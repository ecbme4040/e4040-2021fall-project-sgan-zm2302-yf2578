import tensorflow as tf
from tensorflow.keras import layers as Layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow import keras


class PhysicsNormalizationLayer(Layers.Layer):
    def __init__(self, lower_bounds, upper_bounds, name="physics_normalization"):
        super().__init__(name=name)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def call(self, physics_params):
        normalized_physics_params = dict()
        for k, v in physics_params:
            l = self.lower_bounds[k]
            u = self.upper_bounds[k]
            normalized_physics_params[k] = (v - l) / (u - l)
        return normalized_physics_params


class PhysicsReverseNormalizationLayer(Layers.Layer):
    def __init__(self, lower_bounds, upper_bounds, name="physics_normalization"):
        super().__init__(name=name)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def call(self, normalized_physics_params):
        physics_params = dict()
        for k, v in normalized_physics_params:
            l = self.lower_bounds[k]
            u = self.upper_bounds[k]
            physics_params[k] = l + (u - l) * v
        return physics_params


class CustomNormalizationLayer(Layers.Layer):
    def __init__(self, mean, std, name="Normalization"):
        super().__init__(name=name)
        self.mean = mean
        self.std = std

    def call(self, inputs):
        return (inputs - self.mean) / self.std


class CustomReverseNormalizationLayer(Layers.Layer):
    def __init__(self, mean, std, name="Normalization"):
        super().__init__(name=name)
        self.mean = mean
        self.std = std

    def call(self, inputs):
        return inputs * self.std + self.mean
