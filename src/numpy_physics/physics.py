import numpy as np


def IDM(Inputs, n_repeat=1, return_mean=True, return_var=False,
         action_as="a"):
    '''
    inputs: (batch, dim=3)
    inputs[:,0]: delta_x
    inputs[:,1]: delta_v
    inputs[:,2]: v
    '''
    tf_meta_params = self.reverse_normalization_layer(self.tf_normalized_meta_params)
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
    v_prime = v + acc * self.dT
    if action_as == "a":
        out = acc
    elif action_as == "v":
        out = v_prime
    else:
        raise ValueError("invalid action_as")

    # average
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