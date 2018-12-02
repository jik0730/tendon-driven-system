import os
import json
import argparse
import torch

# Define hyper-parameters and simulation parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='exp/freq_dependency')
parser.add_argument('--data_type', default='sine_0.5Hz_10deg_0offset')
# 0 if f_est=0, 1 if oracle, 2 if MLP
parser.add_argument('--ftype', default=0, type=int)


def evaluate(const, params):

    pass


if __name__ == '__main__':
    args = parser.parse_args()
    const_path = os.path.join(args.model_dir, args.data_type, 'const.json')
    params_path = os.path.join(args.model_dir, args.data_type, 'params.json')
    with open(const_path, 'r') as f:
        const = json.load(f)
    with open(params_path, 'r') as f:
        params = json.load(f)
    evaluate(const, params)