import argparse, os


def parse_cl_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("-n_train", type=int, default=500,
                        help='number of metrics_factory data')

    args = parser.parse_args()
    return args