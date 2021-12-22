import numpy as np
import math
import argparse
import os
from src.helper_funcs import Params
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/ngsim/ngsim_v_dt=1', help="Directory to save idm data, and find config file")
parser.add_argument('--raw_data_dir', default='raw_data/ngsim/IDOVM_para_and_mid_and_73weight_paraspace_cluster.pickle',
                    help="path of the original pickle file for the time-series trajectory")


def trajecotory_to_feature_label(data, time_step, historical_length, prediction_length,
                                 action_as="a"):
    # data is one element of a list
    # data if of shape (N,5), column contains loc_follower, vel_follower, loc_leader, vel_leader, acc_follower
    # time_step: length of steps. e.g. if time_steps=10, we predict every 1s as the sampling frequency is 0.1s
    # historical_length: number of time_step used. e.g. if (time_steps, historical_length)=(10, 5), it means we record
    #       past 5 seconds at every 1 second
    # prediction_length: similar to historical_length but in the opposite temporal direction.
    # action_as: - "a": use acceleration as label
    #            - "v": use velocity as label

    feature = np.hstack([data[:,2].reshape(-1,1) - data[:,0].reshape(-1,1),  # delta x
                         data[:,3].reshape(-1,1) - data[:,1].reshape(-1,1),  # delta v
                         data[:,1].reshape(-1,1)])                            # v
    if action_as == "a":
        label = data[:,-1].reshape(-1,1)
    elif action_as == "v":
        label = data[:,1].reshape(-1,1)
    else:
        raise ValueError("invalid 'action_as'")

    # temporal matching
    lstm_feature = []
    lstm_label = []
    for i in range(data.shape[0] - (time_step-1)*(historical_length+prediction_length)-1):
        whole_idx = np.arange(i, i+(historical_length+prediction_length)*time_step, time_step)
        historical_idx = whole_idx[:historical_length]
        prediction_idx = whole_idx[historical_length: historical_length+prediction_length]
        lstm_feature.append(np.expand_dims(feature[historical_idx , :], axis=0))
        lstm_label.append(np.expand_dims(label[prediction_idx, :], axis=0))

    lstm_feature = np.vstack(lstm_feature)
    lstm_label = np.vstack(lstm_label)

    # squeeze for NN model, if prediction/historical length is 1
    lstm_feature = np.squeeze(lstm_feature, axis=1)
    lstm_label = np.squeeze(lstm_label, axis=1)

    return lstm_feature, lstm_label



    

    
if __name__ == "__main__":
    args = parser.parse_args()
    data_config_path = os.path.join(args.data_dir, "data_para.json")
    assert os.path.isfile(data_config_path), f"file not found: {data_config_path}"

    # if data already exist
    assert not os.path.exists(os.path.join(args.data_dir, 'train_data.csv')),\
        f" train data already exists: {args.data_dir}"
    assert not os.path.exists(os.path.join(args.data_dir, 'validation_data.csv')),\
        f" validation data already exists: {args.data_dir}"
    assert not os.path.exists(os.path.join(args.data_dir, 'test_data.csv')),\
        f" test data already exists: {args.data_dir}"
    assert not os.path.exists(os.path.join(args.data_dir, 'collocation_data.csv')),\
        f" collocation data already exists: {args.data_dir}"

    params = Params(data_config_path)

    # load pickle file
    with open(args.raw_data_dir, "rb") as f:
        data = pickle.load(f)

    feature = []
    label = []
    for tmp_data in data:
        tmp_feature, tmp_label = trajecotory_to_feature_label(tmp_data, params.time_step,
                                                              params.historical_length,
                                                                params.prediction_length,
                                                                  action_as="v")
        tmp = np.hstack([tmp_feature, tmp_label, tmp_feature[:,-1].reshape(-1,1) - tmp_label.reshape(-1,1)])
        #print(max(abs(tmp[:, -1])))
        feature.append(tmp_feature)
        label.append(tmp_label)


    feature = np.vstack(feature)
    feature = feature.reshape(len(feature), -1)
    label = np.vstack(label)
    label = label.reshape(len(label), -1)
    # shuffle the data
    idx = np.random.choice(feature.shape[0], feature.shape[0], replace=True)
    feature = feature[idx, :]
    label = label[idx, :]

    # split into train, valid, test
    train_size = params.train_pool_size
    valid_size = params.validation_pool_size
    test_size = params.test_pool_size
    col_size = params.collocation_pool_size

    train_feature, remain_feature      =        feature[:train_size], feature[train_size:]
    train_label, remain_label          =        label[:train_size], label[train_size:]

    validation_feature, remain_feature =        remain_feature[:valid_size], remain_feature[valid_size:]
    validation_label, remain_label     =        remain_label[:valid_size], remain_label[valid_size:]

    test_feature, remain_feature       =        remain_feature[:test_size], remain_feature[test_size:]
    test_label, remain_label           =        remain_label[:test_size], remain_label[test_size:]

    collocation_feature, _             =        remain_feature[:col_size], remain_feature[col_size:]


    # k-means for test data to find the label
    scaler = MinMaxScaler()
    feature_scale = scaler.fit_transform(feature)
    # weights = [1,1,1,0.0]
    N_neighbor = params.n_samples
    nbrs = NearestNeighbors(n_neighbors=N_neighbor
                            ).fit(feature_scale)
    _, indices = nbrs.kneighbors(scaler.transform(test_feature), n_neighbors=N_neighbor)

    test_label = label[indices].reshape(-1,N_neighbor)

    np.savetxt(os.path.join(args.data_dir, 'train_feature.csv'), train_feature, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'train_label.csv'), train_label, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'validation_feature.csv'), validation_feature, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'validation_label.csv'), validation_label, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'test_feature.csv'), test_feature, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'test_label.csv'), test_label, delimiter=',')
    np.savetxt(os.path.join(args.data_dir, 'collocation_feature.csv'), collocation_feature, delimiter=',')



