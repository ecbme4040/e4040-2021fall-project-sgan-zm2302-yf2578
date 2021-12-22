import tensorflow as tf
from .loss_calculation import get_losses_fromdata_pinndrop
from .gan import GAN
import numpy as np

class NNDrop(GAN):
    def __init__(self,
                 layer_specs,  # used when building the model
                 hyper_params, name="NNDrop"):
        super().__init__(layer_specs, hyper_params, name=name)

    @tf.function
    def train_step(self, data):
        x_dict, y = data

        with tf.GradientTape(persistent=True) as tape:
            g_loss = get_losses_fromdata_pinndrop(x_dict["train"], y,
                                                                self.nets["primal_generator"],
                                                                self.loss["direct_distance"])



        # update parameters

        for _ in range(self.hyper_params["primal_generator"]["num_step"]):
            grads = tape.gradient(g_loss, self.nets["primal_generator"].trainable_weights)
            self.optimizer["primal_generator"].apply_gradients(
                zip(grads,
                    self.nets["primal_generator"].trainable_weights))

        # evaluate
        y_pred_mean = self.nets["primal_generator"](x_dict["train"], n_repeat = self.hyper_params["global_hyper_params"]["n_repeat"])
        self.compiled_metrics.update_state(y, y_pred_mean)
        output_metrics = {m.name: m.result() for m in self.metrics}
        output_losses = {"g_loss":tf.squeeze(g_loss)}
        return output_metrics, output_losses

    

    def predict(self, x, n_repeat=1, pre_train=False):
        if pre_train is True:
            return self.nets["primal_physics"].call(x, n_repeat=n_repeat, return_mean=False)
        elif pre_train is False:
            return self.nets["primal_generator"].call(x, n_repeat=n_repeat, return_mean=False)
        else:
            raise ValueError("invalid pre_train")
