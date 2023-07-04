import tensorflow as tf
from tf_model.tf_modules import AutoModule


class MultiHeadAttention(tf.Module):
    def __init__(self, module):
        super(MultiHeadAttention, self).__init__()
        self.dim = module.dim
        self.num_heads = module.num_heads

        self.scaling = module.scaling
        self.out_lin = AutoModule(module.out_lin)

    def __call__(self, q, k, v):  # q: [N, Lq, dim], k: [N, Lk, dim], v: [N, Lv, dim]
        q = tf.reshape(q, (tf.shape(q)[0], tf.shape(q)[1], self.num_heads, self.dim // self.num_heads))  # [N, Lq, num_heads, head_dim]
        k = tf.reshape(k, (tf.shape(k)[0], tf.shape(k)[1], self.num_heads, self.dim // self.num_heads))  # [N, Lk, num_heads, head_dim]
        v = tf.reshape(v, (tf.shape(v)[0], tf.shape(v)[1], self.num_heads, self.dim // self.num_heads))  # [N, Lk, num_heads, head_dim]
        q = tf.transpose(q, (0, 2, 1, 3))  # [N, num_heads, Lq, head_dim]
        k = tf.transpose(k, (0, 2, 3, 1))  # [N, num_heads, head_dim, Lk]
        v = tf.transpose(v, (0, 2, 1, 3))  # [N, num_heads, Lk, head_dim]

        out = self.checkpoint_fn(q, k, v)
        out = tf.transpose(out, (0, 2, 1, 3))  # [N, Lq, num_heads, head_dim]
        out = tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1], -1))  # [N, Lq, model_dim]
        out = self.out_lin(out)
        return out

    def checkpoint_fn(self, q, k, v):
        attn = tf.matmul(q, k)  # [N, num_heads, Lq, Lk]
        attn = tf.nn.softmax(attn / self.scaling, axis=3)  # [N, num_heads, Lq, Lk]
        out = tf.matmul(attn, v)  # [N, num_heads, Lq, head_dim]
        return out


class SelfAttention(tf.Module):
    def __init__(self, module):
        super(SelfAttention, self).__init__()
        self.qkv_lin = AutoModule(module.qkv_lin)
        self.mha = MultiHeadAttention(module.mha)

    def __call__(self, x, kv_cache):  # x: [N, dim], idx: int, kv_cache: [N, L, dim, 2]
        qkv = tf.reshape(self.qkv_lin(x), (tf.shape(x)[0], tf.shape(x)[1], 3))
        q, k, v = qkv[..., 0], qkv[..., 1], tf.nn.elu(qkv[..., 2])  # each [N, dim]
        kv_cache = tf.concat([kv_cache, tf.expand_dims(tf.stack([k, v], axis=2), 1)], axis=1)
        x = tf.squeeze(self.mha(tf.expand_dims(q, 1), kv_cache[..., 0], kv_cache[..., 1]), 1) + x
        return x, kv_cache


class CrossAttention(tf.Module):
    def __init__(self, module):
        super(CrossAttention, self).__init__()
        self.q_lin = AutoModule(module.q_lin)
        self.mha = MultiHeadAttention(module.mha)

    def __call__(self, x, k, v):  # [N, L, dim]
        q = self.q_lin(x)
        return self.mha(q, k, v) + x


class FFNet(tf.Module):
    def __init__(self, module):
        super(FFNet, self).__init__()
        self.net = AutoModule(module.net)

    def __call__(self, x):
        return self.net(x) + x


class DecoderLayer(tf.Module):
    def __init__(self, module):
        super(DecoderLayer, self).__init__()
        self.self_attn = SelfAttention(module.self_attn)
        self.cross_attn = CrossAttention(module.cross_attn)
        self.ff_net = FFNet(module.ff_net)

    def __call__(self, x, k, v, kv_cache):
        # kv_cache: [N, Lp, dim, 2]
        x, kv_cache = self.self_attn(x, kv_cache)
        x = tf.squeeze(self.cross_attn(tf.expand_dims(x, 1), k, v), 1)
        x = self.ff_net(x)
        return x, kv_cache


class Decoder(tf.Module):
    def __init__(self, module):
        super(Decoder, self).__init__()
        self.embedding = tf.convert_to_tensor(module.embedding.weight.detach().numpy(), dtype=tf.float32)
        self.layers = [DecoderLayer(decoder_layer) for decoder_layer in module.layers]
        self.out_lin = AutoModule(module.out_lin)

    def __call__(self, enc_out, tokens, token_pe, kv_cache, idx):
        # enc_out: [N, Lx, dim, num_layers, 2], tokens: [N, Lp], kv_cache: [N, Lp, dim, num_layers, 2]
        x = tf.gather(self.embedding, tokens[:, idx], axis=0) + token_pe[:, idx]  # [N, dim]
        kv_cache_slices = []
        for i, layer in enumerate(self.layers):
            k, v = enc_out[..., i, 0], enc_out[..., i, 1]
            x, layer_kv_cache = layer(x, k, v, kv_cache[..., i, :])
            kv_cache_slices.append(tf.expand_dims(layer_kv_cache, 3))
        tokens = tf.concat([tokens,
                            tf.expand_dims(tf.math.argmax(self.out_lin(x), axis=1), 1)],
                           axis=1)
        kv_cache = tf.concat(kv_cache_slices, axis=3)
        return enc_out, tokens, token_pe, kv_cache, idx + 1
