import tensorflow as tf
import numpy as np

def get_losses_fromdata(x_train, y, generator, discriminator, estimator,
                        direct_distance_loss_fn,
                        distributional_loss_fn):
    # generate fake data
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    generated_y = generator(x_train)
    generated_xy = tf.concat([x_train, generated_y], axis=1)
    real_xy = tf.concat([x_train, y], axis=-1)

    combined_xy = tf.concat([generated_xy, real_xy], axis=0)

    labels = tf.concat(
        [tf.ones((x_train.shape[0], 1)), tf.zeros((x_train.shape[0], 1))], axis=0
    )
    predicted_logit = discriminator(combined_xy)

    # primal discriminator loss
    d_loss = distributional_loss_fn(labels, predicted_logit)

    # generator loss
    fake_logit = discriminator(generated_xy)
    g_loss_KL = tf.reduce_mean(fake_logit, keepdims=True)

    # ge_loss_reconstruction
    reconstructed_z = estimator(tf.concat([x_train, generated_y], axis=1))
    ge_loss_reconstruction = direct_distance_loss_fn(generator.z, reconstructed_z)

    return g_loss_KL, d_loss, ge_loss_reconstruction


def get_losses_fromdata_wgan(x_train, y, generator, discriminator, estimator,
                        direct_distance_loss_fn,
                        distributional_loss_fn):
    # generate fake data
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    generated_y = generator(x_train)
    generated_xy = tf.concat([x_train, generated_y], axis=1)
    real_xy = tf.concat([x_train, y], axis=-1)



    real_logit = discriminator(real_xy)
    fake_logit = discriminator(generated_xy )

    # primal discriminator loss
    d_loss = tf.reduce_mean(fake_logit) - tf.reduce_mean(real_logit)

    # generator loss
    g_loss_KL = -tf.reduce_mean(fake_logit, keepdims=True)

    # ge_loss_reconstruction
    reconstructed_z = estimator(tf.concat([x_train, generated_y], axis=1))
    ge_loss_reconstruction = direct_distance_loss_fn(generator.z, reconstructed_z)

    return g_loss_KL, d_loss, ge_loss_reconstruction


def get_phys_loss_basic(x, generator, physics,
                        direct_distance_loss_fn,
                        real_y = None):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    physics_y = physics(x)
    if real_y is not None:
        y = real_y
    else:
        y = generator(x)

    phys_loss = direct_distance_loss_fn(y, physics_y)
    phys_loss = tf.reshape(phys_loss, [1,1])
    return phys_loss

def get_phys_loss_mmt(x, generator, physics,
                        direct_distance_loss_fn,
                      n_repeat = 10):
    #assert y.shape[1] == n_repeat, \
    #    "y.shape[1] should equalt to n_repeat. It error may happen if use mmt_gan for pretain." + \
    #    " Only bsc/double gan can be used to pretrain, as they do not require y.shape[1] > 1"
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y_mean, y_var = generator(x, n_repeat=n_repeat, return_mean=True, return_var=True)
    physics_mean, physics_var = physics(x, n_repeat=n_repeat,
                                        return_mean=True, return_var=True)

    norm_y_mean = y_mean / (tf.math.abs(y_mean)+tf.math.abs(physics_mean))
    norm_physics_mean = physics_mean / (tf.math.abs(y_mean)+tf.math.abs(physics_mean))
    norm_y_var = y_var / (y_var+physics_var)
    norm_physics_var = physics_var / (y_var+physics_var)

    mean_diff = direct_distance_loss_fn(norm_y_mean, norm_physics_mean)
    var_diff = direct_distance_loss_fn(norm_y_var, norm_physics_var)
    return mean_diff, var_diff

def get_phys_loss_double(x, generator, physics, discriminator, estimator,
                    direct_distance_loss_fn,
                        distributional_loss_fn,
                         real_y=None):
    # if real_y is not None, real_y is the real-data and is the true label, physics is the target
    # generate fake data
    x= tf.convert_to_tensor(x, dtype=tf.float32)
    y_phy = physics(x)
    if real_y is not None:
        y = real_y
        generated_xy = tf.concat([x, y_phy], axis=1)
        real_xy = tf.concat([x, y], axis=-1)
    else:
        y = generator(x)
        generated_xy = tf.concat([x, y], axis=1)
        real_xy = tf.concat([x, y_phy], axis=-1)

    combined_xy = tf.concat([generated_xy, real_xy], axis=0)


    labels = tf.concat(
        [tf.ones((x.shape[0], 1)), tf.zeros((x.shape[0], 1))], axis=0
    )
    predicted_logit = discriminator(combined_xy)

    # primal discriminator loss
    d_loss = distributional_loss_fn(labels, predicted_logit)

    # generator loss
    fake_logit = discriminator(generated_xy)
    g_loss_KL = tf.reduce_mean(fake_logit, keepdims=True)

    # ge_loss_reconstruction
    reconstructed_z = estimator(tf.concat([x, y], axis=1))
    if real_y is not None:
        ge_loss_reconstruction = None
    else:
        ge_loss_reconstruction = direct_distance_loss_fn(generator.z, reconstructed_z)

    return g_loss_KL, d_loss, ge_loss_reconstruction

def get_twinphys_loss_double(x, generator, physics, discriminator,distributional_loss_fn):
    # if real_y is not None, real_y is the real-data and is the true label, physics is the target
    # generate fake data
    x= tf.convert_to_tensor(x, dtype=tf.float32)

    y = generator(x)
    generated_xy = tf.concat([x, y], axis=1)
    real_xy = tf.concat([x, y_phy], axis=-1)

    combined_xy = tf.concat([generated_xy, real_xy], axis=0)


    labels = tf.concat(
        [tf.ones((x.shape[0], 1)), tf.zeros((x.shape[0], 1))], axis=0
    )
    predicted_logit = discriminator(combined_xy)

    # primal discriminator loss
    d_loss = distributional_loss_fn(labels, predicted_logit)

    # generator loss
    fake_logit = discriminator(generated_xy)
    g_loss_KL = tf.reduce_mean(fake_logit, keepdims=True)


    return g_loss_KL, d_loss

def get_losses_fromdata_pinndrop(x_train, y, generator,
                        direct_distance_loss_fn):
    # generate fake data
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    generated_y = generator(x_train)

    g_loss_data = direct_distance_loss_fn(generated_y, y)

    return g_loss_data


