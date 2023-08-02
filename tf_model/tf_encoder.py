import numpy as np
import tensorflow as tf
from tf_modules import AutoModule


class PositionalEncoding(tf.Module):
    def __init__(self, module):
        super().__init__()
        self.position = tf.convert_to_tensor(module.position.detach().numpy(), dtype=tf.float32)
        self.div_term = tf.convert_to_tensor(module.div_term.detach().numpy(), dtype=tf.float32)

    def __call__(self, x_len):
        angle_arr = self.position[:x_len] * self.div_term  # [max_len, dim / 2]
        pe_sin = tf.math.sin(angle_arr)
        pe_cos = tf.math.cos(angle_arr)
        pe = tf.reshape(tf.stack([pe_sin, pe_cos], axis=2), (1, x_len, -1))
        return pe


class SlidingATTN(tf.Module):
    def __init__(self, module):
        super(SlidingATTN, self).__init__()
        self.num_heads = module.num_heads
        self.window_size = module.window_size

        self.attn_lin = AutoModule(module.attn_lin)
        self.pos_net = AutoModule(module.pos_net)
        self.v_lin = AutoModule(module.v_lin)
        self.out_lin = AutoModule(module.out_lin)

        self.indices_buffer = tf.convert_to_tensor(module.indices_buffer.detach().numpy(), dtype=tf.int32)

    def __call__(self, x):  # x: [N, L, in_dim], mask: [N, L]
        attn_exp = tf.math.exp(self.attn_lin(x))  # [N, L, num_heads]
        attn = attn_exp / (tf.math.reduce_sum(attn_exp, axis=1, keepdims=True) + 1e-5)
        v = self.v_lin(x)
        g_pool = tf.math.reduce_sum(tf.expand_dims(attn, 3) *
                                    tf.reshape(v, (tf.shape(x)[0], tf.shape(x)[1], self.num_heads, -1)),
                                    axis=1)
        g_pool = tf.reshape(g_pool, (tf.shape(x)[0], -1))  # [N, dim]
        pos_component = tf.math.exp(tf.reshape(self.pos_net(g_pool), (-1, 1, self.window_size + 1, self.num_heads)))

        out = self.checkpoint_fn(attn_exp, v, g_pool, pos_component)
        out = tf.reshape(out, (tf.shape(x)[0], tf.shape(x)[1], -1))  # [N, L, out_dim]
        out = self.out_lin(out)
        return out

    def checkpoint_fn(self, attn_exp, v, g_pool, pos_component):
        attn_win = self.extract_sliding_windows(attn_exp)  # [N, L, window_size, num_heads]
        attn_win = tf.concat([tf.ones((tf.shape(v)[0], tf.shape(v)[1], 1, self.num_heads), dtype=tf.float32),
                              attn_win], axis=2)
        attn_win = attn_win * pos_component  # [N, L, window_size + 1, num_heads]
        attn_win = attn_win / (tf.math.reduce_sum(attn_win, axis=2, keepdims=True) + 1e-5)
        v = self.extract_sliding_windows(v)  # [N, L, window_size, out_dim]
        v = tf.reshape(v, (tf.shape(v)[0], tf.shape(v)[1], self.window_size, self.num_heads,
                           -1))  # [N, L, window_size, num_heads, head_dim]

        out = tf.math.reduce_sum(tf.expand_dims(attn_win[:, :, 1:], 4) * v, axis=2)  # [N, L, num_heads, head_dim]
        out = out + \
              (tf.expand_dims(attn_win[:, :, 0], 3) * tf.reshape(g_pool, (tf.shape(v)[0], 1, self.num_heads, -1)))
        return out

    def extract_sliding_windows(self, x):  # [N, L, dim]
        indices = self.indices_buffer[:tf.shape(x)[1]]
        indices = tf.math.minimum(indices, tf.fill(tf.shape(indices), value=tf.shape(x)[1]))
        x = tf.concat([x,
                       tf.zeros((tf.shape(x)[0], 1, tf.shape(x)[2]), dtype=tf.float32)], axis=1)
        x = tf.gather(x, indices, axis=1)
        return x


class AxisLayerNorm(tf.Module):
    def __init__(self, module):
        super(AxisLayerNorm, self).__init__()
        self.num_points = module.num_points
        self.num_axes = module.num_axes
        self.dim = module.dim

        self.gamma = tf.convert_to_tensor(module.gamma.detach().numpy(), dtype=tf.float32)
        self.beta = tf.convert_to_tensor(module.beta.detach().numpy(), dtype=tf.float32)

    def __call__(self, x):  # [N, L, num_points, num_axes]
        weight = tf.cast((x != 0), tf.float32)
        x = x - (tf.math.reduce_sum(x * weight, axis=self.dim, keepdims=True) /
                 (tf.reduce_sum(weight, axis=self.dim, keepdims=True) + 1e-5))
        x_std = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x) * weight, axis=self.dim, keepdims=True) /
                             (tf.math.reduce_sum(weight, axis=self.dim, keepdims=True) + 1e-5))
        x = x / (x_std + 1e-5)
        x = self.gamma * x + self.beta
        x = x * weight
        return x


class Encoder(tf.Module):
    def __init__(self, module):
        super(Encoder, self).__init__()
        self.num_points = module.num_points
        self.num_axes = module.num_axes
        self.norm_ranges = module.norm_ranges

        self.x_norm = AxisLayerNorm(module.x_norm)
        self.feature_norms = [AxisLayerNorm(feature_norm) for feature_norm in module.feature_norms]

        self.input_dim = module.input_dim
        self.dim = module.dim
        self.num_heads = module.num_heads

        self.input_net = AutoModule(module.input_net)
        self.pos_enc = PositionalEncoding(module.pos_enc)
        self.sliding_attn1 = SlidingATTN(module.sliding_attn1)
        self.sliding_attn_stack = [SlidingATTN(sliding_attn) for sliding_attn in module.sliding_attn_stack]

        self.num_dec_layers = module.num_dec_layers
        self.decoder_dim = module.decoder_dim
        self.out_dim = module.out_dim

        self.out_lin = AutoModule(module.out_lin)

    def __call__(self, x):  # x: [N, L, num_points, num_axes], mask: [N, L]
        normed_x = self.x_norm(x)
        normed_features = tf.concat([self.feature_norms[i](x[:, :, start: end])
                                     for i, (start, end) in enumerate(self.norm_ranges)], axis=2)
        x = tf.concat([normed_x, normed_features], axis=2)
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1))

        input_net_out = self.input_net(x) + self.pos_enc(tf.shape(x)[1])
        sliding_attn_out = input_net_out + self.sliding_attn1(input_net_out)
        for sliding_attn in self.sliding_attn_stack:
            sliding_attn_out = sliding_attn_out + sliding_attn(sliding_attn_out)
        out = tf.reshape(self.out_lin(sliding_attn_out), (tf.shape(x)[0], -1, self.decoder_dim, self.num_dec_layers, 2))
        return out
