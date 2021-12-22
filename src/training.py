import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from .helper_funcs import save_dict_to_json, check_and_make_dir

import logging
import os
from src.utils import check_exist_and_delete, check_not_empty_and_delete, delete_file_or_folder, load_json



def training(model, train_feature_dict, train_target, validation_data=None, restore_from=None,
             experiment_dir = None,
             batch_size = None, epochs=None,
             keep_latest_model_max = None,
             n_repeat=None,
             train_NN = True,
             train_PINN = True):
    # Initialize tf.Saver instances to save weights during metrics_factory
    X_train = train_feature_dict["train"]
    X_collocation = train_feature_dict["collocation"]
    y_train = train_target
    begin_at_epoch = 0

    if restore_from is not None:
        logging.info("Restoring parameters from {}".format(restore_from))
        assert os.path.isfile(restore_from), "restore_from is not a file"

        # restore model
        model = keras.models.load_model(restore_from)
        begin_at_epoch = int(restore_from.split('-')[-1].split('.')[0])

        # if current experiment dir is not empty, remove everything except experiment_setting.json
        if len(os.listdir(experiment_dir)) > 1:
            for file_or_folder in os.listdir(experiment_dir):
                if file_or_folder != "experiment_setting.json":
                    delete_file_or_folder(file_or_folder)



    best_eval_mse = 100.0
    # create folders
    check_and_make_dir(os.path.join(experiment_dir, 'best_weights'))
    check_and_make_dir(os.path.join(experiment_dir, 'last_weights'))
    check_and_make_dir(os.path.join(experiment_dir, 'physics_last_weights'))

    # tf summary
    train_log_dir = os.path.join(experiment_dir, "summary", "train")
    eval_log_dir = os.path.join(experiment_dir, "summary", "eval")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    # wright the ground truth distribution
    with eval_summary_writer.as_default():
        tf.summary.histogram(os.path.join("y_val", "true"), validation_data[1], step=0)

    for epoch in range(begin_at_epoch, begin_at_epoch + epochs):
        # Run one epoch
        if epoch%100 == 0:
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + epochs))
        # Compute number of batches in one epoch (one full pass over the metrics_factory set)
        num_steps = X_train.shape[0] // batch_size

        for step in range(num_steps):
            x_batch = X_train[step*batch_size:(step+1)*batch_size,:]
            y_batch = y_train[step*batch_size:(step+1)*batch_size,:]

            # random sample collocation batch
            random_idx = np.random.choice(X_collocation.shape[0], batch_size, replace=False)
            x_batch_collocation = X_collocation.numpy()[random_idx, :]
            x_dict = {"train": x_batch,
                      "collocation": tf.convert_to_tensor(x_batch_collocation)}

            # tf.summary.trace_on(graph=True, profiler=True)
            # # Call only one tf.function when tracing.
            # model.temp_func((x_dict,y_batch))
            # with train_summary_writer.as_default():
            #     tf.summary.trace_export(
            #         name="my_func_trace",
            #         step=0,
            #         profiler_outdir=train_log_dir)

            if train_NN is True:
                train_metrics, train_losses = model.train_step((x_dict,y_batch))
                if "twin_physics" in model.nets.keys():
                    if epoch%10 == 0:
                        physics_params, output_losses = model.train_twin_phys_step((x_dict, y_batch))
                        with train_summary_writer.as_default():
                            for k, v in physics_params.items():
                                tf.summary.scalar(k, v, step=epoch)
                            for k, v in output_losses.items():
                                tf.summary.scalar(k, v, step=epoch)


        model.compiled_metrics.reset_state()

        # write to the summary
        if train_NN is True:
            with train_summary_writer.as_default():
                for k,v in train_losses.items():
                    tf.summary.scalar(k, v, step=epoch)
                for k,v in train_metrics.items():
                    tf.summary.scalar(k, v, step=epoch)
                for layer in model.layers:
                    if layer.name.split("_")[-1] == "physics":
                        continue
                    tf.summary.histogram(os.path.join(layer.name, "weights_layer_0"), layer.get_weights()[0], step=epoch)
                    tf.summary.histogram(os.path.join(layer.name, "weights_all"),
                                         np.concatenate([w.flatten() for w in layer.get_weights()]), step=epoch)

            # save the last physics param



        if train_NN is False:
            continue

        last_save_path = os.path.join(experiment_dir, 'last_weights', f'after-epoch-{epoch+1:d}')
        model.save_weights(last_save_path)

        # only keep lastest N model. N=param.keep_latest_model_max
        N = keep_latest_model_max
        last_save_path_min = f"after-epoch-{epoch+1-N:d}"
        for file_name in os.listdir(os.path.join(experiment_dir, "last_weights")):
            file_dir = os.path.join(experiment_dir, "last_weights", file_name)
            if last_save_path_min in file_dir:
                check_exist_and_delete(file_dir)

        # Evaluate for one epoch on validation set
        #metrics = model.evaluate(*validation_data, return_dict=True)
        eval_metrics=model.test_step(validation_data)
        with eval_summary_writer.as_default():
            for k, v in eval_metrics.items():
                tf.summary.scalar(k, v, step=epoch)

            # the prediction
            with eval_summary_writer.as_default():
                tf.summary.histogram(os.path.join("y_val", "pred"), model.predict(validation_data[0]), step=epoch)



        # If best_eval, best_save_path
        eval_mse = eval_metrics["mean_squared_error"]
        if eval_mse <= best_eval_mse:
            # Store new best accuracy
            best_eval_mse = eval_mse
            # Save weights
            best_save_path = os.path.join(experiment_dir, 'best_weights', f'after-epoch-{epoch+1:d}')

            check_not_empty_and_delete(os.path.join(experiment_dir, 'best_weights'))
            model.save_weights(best_save_path)
            logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
            # Save best eval metrics in a json file in the model directory
            best_json_path = os.path.join(experiment_dir, "metrics_eval_best_weights.json")
            save_dict_to_json(eval_metrics, best_json_path)

        # Save latest eval metrics in a json file in the model directory
    if train_NN is True:
        last_json_path = os.path.join(experiment_dir, "metrics_eval_last_weights.json")
        save_dict_to_json(eval_metrics, last_json_path)





def test_multiple_rounds(model, data_test, test_rounds,
                         save_dir = None,
                         model_alias = None,
                        train_NN = True,
                **kwargs):
    metrics_dict, test_prediction, kl = test(model, data_test, train_NN=train_NN,
                                         **kwargs)
    logging.info("Restoring parameters from {}".format(kwargs["restore_from"]))
    if test_rounds > 1:
        for i in range(test_rounds-1):
            metrics_dict_new,_,_ = test(model, data_test, train_NN = train_NN,
                                    **kwargs)
            for k in metrics_dict.keys():
                metrics_dict[k] += metrics_dict_new[k]
    check_and_make_dir(os.path.join(save_dir, model_alias))
    save_path_metric = os.path.join(save_dir, model_alias,
                                    f"metrics_test.json")
    save_path_prediction = os.path.join(save_dir, model_alias,
                                        f"predictions_test.csv")
    save_path_feature = os.path.join(save_dir, model_alias,
                                        f"features_test.csv")
    save_path_target = os.path.join(save_dir, model_alias,
                                        f"targets_test.csv")
    save_path_kl = os.path.join(save_dir, model_alias,
                                    f"kl_test.csv")


    save_dict_to_json(metrics_dict, save_path_metric)
    np.savetxt(save_path_prediction, test_prediction, delimiter=",")
    np.savetxt(save_path_feature, data_test[0], delimiter=",")
    np.savetxt(save_path_target, data_test[1], delimiter=",")
    np.savetxt(save_path_kl, kl, delimiter=",")






def test(model, data_test,
                restore_from=None,
                metric_functions = None,
                n_samples = None,
                params=None,
         train_NN = True):
    # experiment_dir: where the model json file locates
    # Initialize tf.Saver instances to save weights during metrics_factory

    if restore_from is not None:
        if ("phys" in model.name) and (train_NN is False):
            meta_params = load_json(restore_from)
            layer_specs, hyper_params = model.layer_specs, model.hyper_params
            layer_specs["primal_physics"]["net_config"] = list(layer_specs["primal_physics"]["net_config"])
            layer_specs["primal_physics"]["net_config"][0] = meta_params
            layer_specs["primal_physics"]["net_config"] = tuple(layer_specs["primal_physics"]["net_config"])
            parent_class = model.__class__
            model =parent_class(layer_specs, hyper_params, name=model.name)
        else:
            model.load_weights(restore_from).expect_partial()
    else:
        raise FileExistsError("model not exist in "+ restore_from)

    # make prediction

    test_feature, test_target = data_test
    test_prediction = model.predict(test_feature, n_repeat=n_samples)

    # convert to numpy
    test_target = test_target.numpy()
    test_prediction = test_prediction.numpy()

    metrics_dict = dict()
    kl = None
    for k, func in metric_functions.items():
        if k == 'nlpd':
            use_mean = True if params.nlpd_use_mean == "True" else False
            metrics_dict[k] = [func(test_target, test_prediction,
                                   use_mean = use_mean,
                                   n_bands = params.nlpd_n_bands)]
        elif k == "kl":
            kl = func(test_target, test_prediction)
            metrics_dict[k] = [np.mean(kl)]
        else:
            metrics_dict[k] = [func(test_target, test_prediction)]


    return metrics_dict, test_prediction, kl