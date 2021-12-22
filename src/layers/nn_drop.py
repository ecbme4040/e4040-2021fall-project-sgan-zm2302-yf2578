import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as Layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from .utils import CustomNormalizationLayer, CustomReverseNormalizationLayer
from tensorflow.keras import backend

from .gan import ClipConstraint



class LeakyReluDropout(Layers.Layer):
    def __init__(self, num_node, constraint = None,
                 drop_prob = None):
        super().__init__()
        if constraint == "clip":
            self.Dense = Dense(num_node, kernel_initializer='he_uniform',
                               kernel_constraint=ClipConstraint())
        else:
            self.Dense = Dense(num_node, kernel_initializer='he_uniform')

            self.leaky_relu = keras.layers.LeakyReLU(alpha=0.2)

        self.leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
        self.batch_norm = keras.layers.BatchNormalization()
        self.drop_prob = drop_prob
    def call(self,  inputs):
        out = self.Dense(inputs)
        out = self.leaky_relu(out)
        out = self.batch_norm(out)
        out = tf.nn.dropout(out, rate=self.drop_prob)
        return out


class DensetanhDropout(Layers.Layer):
    def __init__(self, num_node, constraint=None,
                 keep_prob=None):
        super().__init__()
        if constraint == "clip":
            self.Dense = Dense(num_node, activation='tanh', kernel_initializer='he_uniform',
                               kernel_constraint=ClipConstraint())
        else:
            self.Dense = Dense(num_node, activation='tanh', kernel_initializer='he_uniform')
        self.batch_norm = keras.layers.BatchNormalization()
    def call(self,  inputs):
        out = self.Dense(inputs)
        out = self.batch_norm(out)
        out = tf.nn.dropout(out, keep_prob=self.keep_prob)
        return out

class NNDrop(Layers.Layer):
    def __init__(self, layers,  X_mean, X_std, Y_mean, Y_std, drop_prob, name=None,
                 layer_type = None,
                 constraint = None):
        assert (len(X_mean.shape) == 2) & (X_mean.shape[0] == 1)
        assert (len(X_std.shape) == 2) & (X_std.shape[0] == 1)
        super().__init__(name=name)

        self.drop_prob = drop_prob
        self.normalization_layer = CustomNormalizationLayer(X_mean, X_std)
        self.hidden_layers = []
        for l in range(1, len(layers) - 1):
            if layer_type == "leaky_relu":
                self.hidden_layers.append(LeakyReluDropout(layers[l], constraint=constraint, drop_prob=drop_prob))
            elif layer_type == "dense_tanh":
                self.hidden_layers.append(DensetanhDropout(layers[l], constraint=constraint, drop_prob=drop_prob))
            else:
                raise ValueError("invalid layer_type")


        self.out_layer = Dense(layers[-1], kernel_initializer='he_uniform')
        self.out_reverse_normalization_layer = CustomReverseNormalizationLayer(Y_mean, Y_std)

    def call(self, inputs, n_repeat = 1, return_mean=True, return_var=False ):
        # replicate the inputs
        inputs = tf.repeat(inputs, n_repeat, axis=0, name="generator_repeat")


        # normalization layer
        out = self.normalization_layer(inputs)

        # hidden layer
        for l in self.hidden_layers:
            out = l(out, training=True)

        # output
        out = self.out_layer(out)
        out = self.out_reverse_normalization_layer(out)

        # average
        out = tf.reshape(out, [-1,n_repeat])
        out_mean = tf.reduce_mean(out, axis=1, keepdims=True)

        if return_mean:
            if return_var is True:
                # var
                out_var = tf.math.reduce_variance(out, axis=1, keepdims=True)
                return out_mean, out_var
            else:
                return out_mean

        else:
            return out