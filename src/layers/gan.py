import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as Layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from .utils import CustomNormalizationLayer, CustomReverseNormalizationLayer
from tensorflow.keras import backend
# Generator


# clip constraint
# code from https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value=0.01):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


class LeakyRelu(Layers.Layer):
    def __init__(self, num_node, constraint = None):
        super().__init__()
        if constraint == "clip":
            self.Dense = Dense(num_node, kernel_initializer='he_uniform',
                               kernel_constraint=ClipConstraint())
        else:
            self.Dense = Dense(num_node, kernel_initializer='he_uniform')

            self.leaky_relu = keras.layers.LeakyReLU(alpha=0.2)

        self.leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
        self.batch_norm = keras.layers.BatchNormalization()
    def call(self,  inputs):
        out = self.Dense(inputs)
        out = self.leaky_relu(out)
        out = self.batch_norm(out)
        return out


class Densetanh(Layers.Layer):
    def __init__(self, num_node, constraint=None):
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
        return out








class Generator(Layers.Layer):
    def __init__(self, layers,  X_mean, X_std, Y_mean, Y_std, latent_dim, name = "generator",
                 layer_type = None,
                 constraint = None):
        assert (len(X_mean.shape) == 2) & (X_mean.shape[0] == 1)
        assert (len(X_std.shape) == 2) & (X_std.shape[0] == 1)
        super().__init__(name=name)

        self.latent_dim = latent_dim
        self.z = None

        self.normalization_layer = CustomNormalizationLayer(X_mean, X_std)
        self.hidden_layers = []
        for l in range(1, len(layers)-1):
            if layer_type == "leaky_relu":
                self.hidden_layers.append(LeakyRelu(layers[l], constraint=constraint))
            elif layer_type == "dense_tanh":
                self.hidden_layers.append(Densetanh(layers[l], constraint=constraint))
            else:
                self.hidden_layers.append(Dense(layers[l], activation='tanh', kernel_initializer='he_uniform'))

        self.out_layer = Dense(layers[-1], kernel_initializer='he_uniform' )
        self.out_reverse_normalization_layer = CustomReverseNormalizationLayer(Y_mean, Y_std)

    def call(self, inputs, n_repeat = 1, return_mean=True, return_var=False ):
        # replicate the inputs
        batch = tf.shape(inputs)[0]
        inputs = tf.repeat(inputs, n_repeat, axis=0, name="generator_repeat")
        dim = self.latent_dim
        z = tf.keras.backend.random_normal(shape =(int(batch*n_repeat), dim) )
        self.z = z

        # normalization layer
        out = self.normalization_layer(inputs)

        # concat latent variable
        #out = tf.keras.layers.Concatenate(axis=1)([out, z])
        out = tf.concat([out, z], axis=-1)

        # hidden layer
        for l in self.hidden_layers:
            out = l(out)

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

# Discriminator

class Discriminator(Layers.Layer):
    def __init__(self, layers, X_mean, X_std, Y_mean, Y_std, name="discriminator",
                 layer_type = None,
                 constraint=None):
        assert (len(X_mean.shape) == 2) & (X_mean.shape[0] == 1)
        assert (len(X_std.shape) == 2) & (X_std.shape[0] == 1)
        assert (len(Y_mean.shape) == 2) & (Y_mean.shape[0] == 1)
        assert (len(Y_std.shape) == 2) & (Y_std.shape[0] == 1)
        super().__init__(name=name)

        self.normalization_layer = CustomNormalizationLayer(np.concatenate([X_mean, Y_mean], axis=1),
                                                            np.concatenate([X_std, Y_std], axis=1))
        self.hidden_layers = []
        for l in range(1, len(layers) - 1):
            if layer_type == "leaky_relu":
                self.hidden_layers.append(LeakyRelu(layers[l],constraint=constraint))
            elif layer_type == "dense_tanh":
                self.hidden_layers.append(Densetanh(layers[l], constraint=constraint))
            else:
                self.hidden_layers.append(Dense(layers[l], activation='tanh', kernel_initializer='he_uniform'))

        self.out_layer = Dense(layers[-1], kernel_initializer='he_uniform')

    def call(self, inputs):
        # normalization layer
        out = self.normalization_layer(inputs)

        # hidden layer
        for l in self.hidden_layers:
            out = l(out)

        # output
        out = self.out_layer(out)

        return out

# Posterior Estimator
class PosteriorEstimator(Layers.Layer):
    def __init__(self, layers, X_mean, X_std, Y_mean, Y_std, name = "estimator",
                 layer_type = None,
                 constraint=None):
        assert (len(X_mean.shape) == 2) & (X_mean.shape[0] == 1)
        assert (len(X_std.shape) == 2) & (X_std.shape[0] == 1)
        assert (len(Y_mean.shape) == 2) & (Y_mean.shape[0] == 1)
        assert (len(Y_std.shape) == 2) & (Y_std.shape[0] == 1)
        super().__init__(name=name)

        self.normalization_layer = CustomNormalizationLayer(np.concatenate([X_mean, Y_mean], axis=1),
                                                            np.concatenate([X_std, Y_std], axis=1))
        self.hidden_layers = []
        for l in range(1, len(layers) - 1):
            if layer_type == "leaky_relu":
                self.hidden_layers.append(LeakyRelu(layers[l], constraint=constraint))
            elif layer_type == "dense_tanh":
                self.hidden_layers.append(Densetanh(layers[l], constraint=constraint))
            else:
                self.hidden_layers.append(Dense(layers[l], activation='tanh', kernel_initializer='he_uniform'))

        self.out_layer = Dense(layers[-1], kernel_initializer='he_uniform')

    def call(self, inputs):
        # normalization layer
        out = self.normalization_layer(inputs)

        # hidden layer
        for l in self.hidden_layers:
            out = l(out)

        # output
        out = self.out_layer(out)

        return out





        






