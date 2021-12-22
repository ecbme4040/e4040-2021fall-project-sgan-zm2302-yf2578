from tensorflow import keras
import tensorflow as tf
from .loss_calculation import get_losses_fromdata, get_losses_fromdata_wgan


class GAN(keras.Model):
    def __init__(self,
                 layer_specs,  # used when building the model
                 hyper_params,  # used when metrics_factory the model
                 name="BasicGAN"):
        super().__init__()
        self.hyper_params = hyper_params  # like
        self.layer_specs = layer_specs
        self.nets = dict()
        #self.mse_metric = keras.metrics.MeanSquaredError(name="mse")d

        for k, v in layer_specs.items():
            # define nets
            self.nets[k] = layer_specs[k]["net"](*layer_specs[k]["net_config"],
                                                 **layer_specs[k]["kwargs"])



    @tf.function
    def train_step(self, data):
        x_dict, y = data

        with tf.GradientTape(persistent=True) as tape:



            g_loss_KL, d_loss, ge_loss_reconstruction = get_losses_fromdata(x_dict["train"], y,
                                                                                 self.nets["primal_generator"],
                                                                                 self.nets["primal_discriminator"],
                                                                                 self.nets["primal_estimator"],
                                                                            self.loss["direct_distance"],
                                                                            self.loss["distributional_comparison"])
            g_loss = g_loss_KL + self.hyper_params["global_hyper_params"]["lam"] * \
                     ge_loss_reconstruction

        # update parameters
        for _ in range(self.hyper_params["primal_discriminator"]["num_step"]):
            grads = tape.gradient(d_loss, self.nets["primal_discriminator"].trainable_weights)
            self.optimizer["primal_discriminator"].apply_gradients(
                zip(grads,
                    self.nets["primal_discriminator"].trainable_weights))

        for _ in range(self.hyper_params["primal_generator"]["num_step"]):
            grads = tape.gradient(g_loss, self.nets["primal_generator"].trainable_weights)
            self.optimizer["primal_generator"].apply_gradients(
                zip(grads,
                    self.nets["primal_generator"].trainable_weights))
        if self.hyper_params["global_hyper_params"]["lam"] > 0.0:
            for _ in range(self.hyper_params["primal_estimator"]["num_step"]):
                grads = tape.gradient(ge_loss_reconstruction, self.nets["primal_estimator"].trainable_weights)
                self.optimizer["primal_estimator"].apply_gradients(
                    zip(grads,
                        self.nets["primal_estimator"].trainable_weights))

        # evaluate
        y_pred_mean = self.nets["primal_generator"](x_dict["train"], n_repeat = self.hyper_params["global_hyper_params"]["n_repeat"])
        self.compiled_metrics.update_state(y, y_pred_mean)
        output_metrics = {m.name: m.result() for m in self.metrics}
        output_losses = {"g_loss":tf.squeeze(g_loss), "g_loss_kl": tf.squeeze(g_loss_KL),
                         "d_loss": tf.squeeze(d_loss), "reconstraction_loss":tf.squeeze(ge_loss_reconstruction)}
        return output_metrics, output_losses

    def train_phys_step(self, data, summary_writer=None):
        x_dict, y = data
        pass


    def test_step(self, data):
        x,y = data
        y_pred_mean = self.nets["primal_generator"](x, n_repeat = self.hyper_params["global_hyper_params"]["n_repeat"])
        self.compiled_metrics.update_state(y, y_pred_mean)
        output = {m.name: m.result() for m in self.metrics}
        return output


    def call(self, x):
        return self.nets["primal_generator"](x)

    def predict(self, x, n_repeat=1):
        return self.nets["primal_generator"](x, n_repeat=n_repeat, return_mean=False)

