from tensorflow.keras import layers as Layers
import tensorflow as tf
from keras import backend

from .utils import PhysicsNormalizationLayer, PhysicsReverseNormalizationLayer


class DeterministicIDMLayer(Layers.Layer):
    def __init__(self, params_value, params_trainable, lower_bounds, upper_bounds, name="D-IDM",
                 layer_type=None, constraint=None):
        # layer_type and constraint is useless for physics layer, just to be in accordance with GAN layers.
        super().__init__(name)
        # normalization layer
        self.normalization_layer = PhysicsNormalizationLayer(lower_bounds, upper_bounds)
        self.reverse_normalization_layer = PhysicsReverseNormalizationLayer(lower_bounds, upper_bounds)


        normalized_params_value = self.normalization_layer(params_value)

        self.tf_normalized_params = dict()
        # Question: variable is normalized, how to view unnormalized variable in the tensorboard?
        for k, v in normalized_params_value:
            self.tf_normalized_params[k] = tf.Variable(v, dtype="float32", trainable=params_trainable[k])

    def call(self, inputs, action_as="a", dt=1):
        '''
        inputs: (batch, dim=3)
        inputs[:,0]: delta_x
        inputs[:,1]: delta_v
        inputs[:,2]: v
        '''
        tf_params = self.reverse_normalization_layer(self.tf_normalized_params)
        dx = inputs[:, :1]
        dv = inputs[:, 1:2]
        v = inputs[:, 2:]

        in_root = tf_params["a"] * tf_params["b"]
        in_root = tf.math.maximum(in_root, 0, name="max")
        root = tf.sqrt(in_root)
        temp = v * tf_params["T0"] - 0.5 * v * dv / root
        desired_s_n = tf_params["s0"] + temp
        acc = tf_params["a"] * (1 - (v / (tf_params["v0"] + 1e-4)) ** 4 - (desired_s_n / (dx + 1e-4)) ** 2)
        v_prime = v + acc*dt

        if action_as == "a":
            return acc
        elif action_as == "v":
            return v_prime
        else:
            raise ValueError("invalid actin_as")


def clip(x, l=None, u=None):
    return tf.clip_by_value(x, l, u)

class StochasticIDM(Layers.Layer):
    def __init__(self, meta_params_value, meta_params_trainable, lower_bounds, upper_bounds, action_as,
                 dT=1, name="S-IDM",
                 layer_type=None, constraint=None):
        # layer_type and constraint is useless for physics layer, just to be in accordance with GAN layers.
        # params is a dictionary, below is one example:
        # params = {"amax": 15,
        #          "v0": 0}
        # lower/upper bounds are also dictionary with the same architecture with the
        super().__init__(name=name)
        # normalization layer

        self.tf_meta_params = dict()
        # Question: variable is normalized, how to view unnormalized variable in the tensorboard?
        for k, v in meta_params_value.items():
            self.tf_meta_params[k] = tf.Variable([v], shape=(1,),  dtype="float32", trainable=(meta_params_trainable[k]=="True"),
                                                 # constraint=lambda x: tf.clip_by_value(x, float(lower_bounds[k]),
                                                 #                                       float(upper_bounds[k])),
                                                 name=k
                                                 )
            backend.track_variable(self.tf_meta_params[k])
            if meta_params_trainable[k]=="True":
                self._trainable_weights.append(self.tf_meta_params[k])
            else:
                self._non_trainable_weights.append(self.tf_meta_params[k])
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.dT = dT
        self.action_as = action_as

    def call(self, inputs, n_repeat=1, return_mean=True, return_var=False):
        '''
        inputs: (batch, dim=3)
        inputs[:,0]: delta_x
        inputs[:,1]: delta_v
        inputs[:,2]: v
        '''
        action_as = self.action_as
        tf_meta_params = self.tf_meta_params
        tf_params = self.sample_params(tf_meta_params, n_repeat=n_repeat)
        dx = inputs[:, :1]
        dv = inputs[:, 1:2]
        v = inputs[:, 2:]

        in_root = tf_params["amax"] * tf_params["b"]
        in_root = tf.math.maximum(in_root, 0, name="max")
        root = tf.sqrt(in_root)
        temp = v * tf_params["T0"] - 0.5 * v * dv / root
        desired_s_n = tf_params["s0"] + temp
        acc = tf_params["amax"] * (1 - (v / (tf_params["v0"] + 1e-4)) ** 4 - (desired_s_n / (dx + 1e-4)) ** 2)
        acc = tf.clip_by_value(acc, -2,100)
        v_prime = v + acc*self.dT
        if action_as == "a":
            out = acc
        elif action_as == "v":
            out = v_prime
        else:
            raise ValueError("invalid action_as")

        # average
        out = out + tf_meta_params["sigma_external"] * tf.keras.backend.random_normal(shape=out.shape)
        out = tf.reshape(out, [-1, n_repeat])
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

    def sample_params(self, tf_meta_params, n_repeat=None):
        meta_pairs = [("mu_amax", "sigma_amax"),
                      ("mu_v0", "sigma_v0"),
                      ("mu_T0","sigma_T0"),
                      ("mu_s0", "sigma_s0"),
                      ("mu_b", "sigma_b")]
        tf_params = dict()
        for mu_key, sigma_key in meta_pairs:
            param_key = mu_key.split("_")[1]
            tf_params[param_key] = tf_meta_params[mu_key] + tf_meta_params[sigma_key]*tf.keras.backend.random_normal(shape =(1, n_repeat) )
            tf_params[param_key] = tf.clip_by_value(tf_params[param_key], self.lower_bounds[mu_key], self.upper_bounds[mu_key])
        tf_params["sigma_external"] = tf_meta_params["sigma_external"]
        return tf_params

    def clip(self):
        for k in self.tf_meta_params.keys():
            self.tf_meta_params[k].assign( tf.clip_by_value(
                self.tf_meta_params[k],
                self.lower_bounds[k],
                self.upper_bounds[k]
            )
            )




class StochasticHelly(Layers.Layer):
    def __init__(self, meta_params_value, meta_params_trainable, lower_bounds, upper_bounds,
                 dT=1, name="S-Helly",
                 layer_type=None, constraint=None):
        # layer_type and constraint is useless for physics layer, just to be in accordance with GAN layers.
        # params is a dictionary, below is one example:
        # params = {"amax": 15,
        #          "v0": 0}
        # lower/upper bounds are also dictionary with the same architecture with the
        super().__init__(name=name)
        # normalization layer

        self.tf_meta_params = dict()
        # Question: variable is normalized, how to view unnormalized variable in the tensorboard?
        for k, v in meta_params_value.items():
            self.tf_meta_params[k] = tf.Variable([v], shape=(1,),  dtype="float32", trainable=(meta_params_trainable[k]=="True"),
                                                 # constraint=lambda x: tf.clip_by_value(x, float(lower_bounds[k]),
                                                 #                                       float(upper_bounds[k])),
                                                 name=k
                                                 )
            backend.track_variable(self.tf_meta_params[k])
            if meta_params_trainable[k]=="True":
                self._trainable_weights.append(self.tf_meta_params[k])
            else:
                self._non_trainable_weights.append(self.tf_meta_params[k])
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.dT = dT

    def call(self, inputs, n_repeat=1, return_mean=True, return_var=False,
             action_as="v"):
        '''
        inputs: (batch, dim=3)
        inputs[:,0]: delta_x
        inputs[:,1]: delta_v
        inputs[:,2]: v
        '''
        tf_meta_params = self.tf_meta_params
        tf_params = self.sample_params(tf_meta_params, n_repeat=n_repeat)
        dx = inputs[:, :1]
        dv = inputs[:, 1:2]
        v = inputs[:, 2:]

        acc = tf_params["lambdav"]*dv + tf_params["lambdax"]*dx + tf_params["D"]
        #acc = tf.clip_by_value(acc, -3,3)
        v_prime = v + acc*self.dT
        if action_as == "a":
            out = acc
        elif action_as == "v":
            out = v_prime
        else:
            raise ValueError("invalid action_as")
        # average
        out = out + tf_meta_params["sigma_external"] * tf.keras.backend.random_normal(shape=out.shape)
        out = tf.reshape(out, [-1, n_repeat])
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

    def sample_params(self, tf_meta_params, n_repeat=None):
        meta_pairs = [("mu_lambdax", "sigma_lambdax"),
                      ("mu_lambdav", "sigma_lambdav"),
                      ("mu_D","sigma_D")]
        tf_params = dict()
        for mu_key, sigma_key in meta_pairs:
            param_key = mu_key.split("_")[1]
            tf_params[param_key] = tf_meta_params[mu_key] + tf_meta_params[sigma_key]*tf.keras.backend.random_normal(shape =(1, n_repeat) )
            tf_params[param_key] = tf.clip_by_value(tf_params[param_key], self.lower_bounds[mu_key], self.upper_bounds[mu_key])
        tf_params["sigma_external"] = tf_meta_params["sigma_external"]
        return tf_params

    def clip(self):
        for k in self.tf_meta_params.keys():
            self.tf_meta_params[k].assign( tf.clip_by_value(
                self.tf_meta_params[k],
                self.lower_bounds[k],
                self.upper_bounds[k]
            )
            )



