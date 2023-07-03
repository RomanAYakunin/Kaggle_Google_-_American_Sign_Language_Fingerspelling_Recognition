import torch
import torch.nn as nn
import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, module):
        super(Linear, self).__init__()
        self.weight = tf.convert_to_tensor(module.weight.detach().numpy().T, dtype=tf.float32)
        self.bias = tf.convert_to_tensor(module.bias.detach().numpy(), dtype=tf.float32)

    def __call__(self, x):
        return tf.matmul(x, self.weight) + self.bias


class LayerNorm(tf.Module):
    def __init__(self, module):
        super(LayerNorm, self).__init__()
        self.weight = tf.convert_to_tensor(module.weight.detach().numpy(), dtype=tf.float32)
        self.bias = tf.convert_to_tensor(module.bias.detach().numpy(), dtype=tf.float32)

    def __call__(self, x):
        x = (x - tf.math.reduce_mean(x, axis=-1, keepdims=True)) / \
            tf.math.sqrt(tf.math.reduce_variance(x, axis=-1, keepdims=True) + 1e-5)
        return x * self.weight + self.bias


class ELU(tf.Module):
    def __init__(self):
        super(ELU, self).__init__()

    def __call__(self, x):
        return tf.nn.elu(x)


class Sequential(tf.Module):
    def __init__(self, tf_modules):
        super(Sequential, self).__init__()
        self.tf_modules = tf_modules

    def __call__(self, x):
        for module in self.tf_modules:
            x = module(x)
        return x


class AutoModule(tf.Module):
    def __init__(self, module):
        super(AutoModule, self).__init__()
        if type(module) is nn.Sequential:
            tf_seq_modules = []
            for seq_module in module:
                if not (type(seq_module) is nn.Dropout):
                    tf_seq_modules.append(AutoModule(seq_module))
            self.tf_module = Sequential(tf_seq_modules)
        elif type(module) is nn.Linear:
            self.tf_module = Linear(module)
        elif type(module) is nn.LayerNorm:
            self.tf_module = LayerNorm(module)
        elif type(module) is nn.ELU:
            self.tf_module = ELU()
        else:
            raise RuntimeError('unsupported torch module')

    def __call__(self, x):
        return self.tf_module(x)


