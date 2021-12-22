import os, time
import tensorflow as tf


class Learner():
    def __init__(self, args, state_action_data_path):
        self.args = args
        self.state_action_data = self.load(data)
        self.data_dict = self.get_data_dict()

    def run(self):


    def get_data_dict(self, verbose = True):
        # output: whether use the one-state-to-multi-a neighbouring, X_train and a_train should reshape to (-1,1)
        n_train, n_valid, n_test = self.args.n_train, self.args.n_valid, self.args.n_test
        n_collocation = self.args.n_collocation

        state_action_data = np.random.shuffle(self.state_action_data)
        train_data = state_action_data[:n_train, :]
        valid_data = state_action_data[n_train:(n_train+n_valid), :]
        test_data = state_action_data[(n_train+n_valid):(n_train+n_valid+n_test), :]
        collocation_data = state_action_data[(n_train+n_valid+n_test):(n_train+n_valid+n_test+n_collocation):, :]

        input_train, output_train = train_data[:3,:], train_data[3:,:]
        input_valid, output_valid = valid_data[:3,:], valid_data[3:,:]
        input_test, output_test =   test_data[:3,:], test_data[3:, :]

        data_dict = {"input_train": input_train, "output_train": output_train, "input_valid": input_valid,
                     "output_valid": output_valid, "input_test": input_test, "output_test": output_test,
                     "train_data": train_data, "valid_data": valid_data, 'test_data': test_data,
                     "collocation_data": collocation_data}

        if verbose:
            for key in list(data_dict.keys()):
                print(f"{key} shape: {data_dict[key].shape}")

        return data_dict







