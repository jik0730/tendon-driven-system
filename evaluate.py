import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from utils import compute_input_force
from utils import compute_theta_hat
from utils import store_logs
from utils import cast_dict_to_tensor
from pid_controller import PIDController
from friction_est import Friction_EST
from friction_true import RealFriction
from visualization import plot_theta
from target_generator import sin_target_traj
from target_generator import random_walk
from target_generator import sin_freq_variation
from target_generator import step_target_traj

# Define hyper-parameters and simulation parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='exp/freq_dependency')
parser.add_argument('--data_type', default='sine_0.5Hz_10deg_0offset')
parser.add_argument('--eval_type', default='sine_0.5Hz_10deg_0offset')
# 0 if f_est=0, 1 if oracle, 2 if MLP
parser.add_argument('--ftype', default=0, type=int)
args = parser.parse_args()


def evaluate(const, params, ftype):
    print('Start evaluation exp: {} and {} and {}'.format(
        args.model_dir, args.data_type, args.ftype))

    # Total running steps
    T = params['freq'] * params['simT']

    # Target trajectory for evaluation
    if 'sine' in args.eval_type and 'Hz' in args.eval_type:
        target_traj = sin_target_traj(
            params['freq'], params['simT'], sine_type=args.eval_type)
    elif 'random_walk' in args.eval_type:
        target_traj = random_walk(T, args.eval_type)
    elif 'sine_freq_variation' == args.eval_type:
        freq_from = 0.5
        freq_to = 10.
        sys_freq = params['freq']
        simT = params['simT']
        sine_type = 'sine_1Hz_10deg_0offset'
        target_traj = sin_freq_variation(freq_from, freq_to, sys_freq, simT,
                                         sine_type)
    elif 'sine_freq_variation_with_step' == args.eval_step:
        freq_from = 0.5
        freq_to = 10.
        sys_freq = params['freq']
        simT = params['simT']
        sine_type = 'sine_1Hz_10deg_0offset'
        target_traj = sin_freq_variation_with_step(freq_from, freq_to,
                                                   sys_freq, simT, sine_type)
    elif 'step' in args.eval_type:
        target_traj = step_target_traj(T, args.eval_type)
    else:
        raise Exception('I dont know your targets')

    # initiate values
    if 'step' in args.eval_type:
        t_OBS_vals = [
            torch.FloatTensor([0]),
            torch.FloatTensor([0]),
            torch.FloatTensor([0]),
            torch.FloatTensor([0])
        ]
    else:
        t_OBS_vals = [
            torch.FloatTensor([target_traj[0]]),
            torch.FloatTensor([target_traj[1]]),
            torch.FloatTensor([target_traj[2]]),
            torch.FloatTensor([target_traj[3]])
        ]
    f1_EST_vals = [torch.zeros(1)]
    F_EST = torch.zeros(1)

    # for plotting
    time_stamp = []
    target_history = []
    obs_history = []
    est_history = []
    f1_obs_history = []
    f1_est_history = []
    F_est_history = []

    # Define PID controller
    Kp = params['Kp']
    Kd = params['Kd'] * Kp
    Ki = params['Ki'] * Kp
    pid_cont = PIDController(p=Kp, i=Ki, d=Kd, del_t=const['del_t'] * 10)

    # Define models TODO for now we ignore f2.
    f1_OBS_fn = RealFriction(const)
    f1_EST_fn = Friction_EST(params['hdim'])
    if ftype == 2:
        state_dict_path = os.path.join(args.model_dir, args.data_type, 'MLP',
                                       'f1_model')
        state_dict = torch.load(state_dict_path)
        f1_EST_fn.load_state_dict(state_dict)

    for t in range(4, T):
        # current target
        target = target_traj[t]

        # detach nodes for simplicity
        f1 = f1_EST_vals[-1].detach()

        # compute input force (F) at t-2
        if t % 10 == 3:
            const['T_d'] = pid_cont.compute_torque(target, t_OBS_vals[-1])
            F_EST = compute_input_force(const, t_OBS_vals[-1], f1)

        # compute frictions (f) at t-2
        t_dot_OBS = (t_OBS_vals[-2] - t_OBS_vals[-3]) / const['del_t']
        f1_OBS = f1_OBS_fn(t_OBS_vals[-2], t_OBS_vals[-3], F_EST)

        if ftype == 0:
            f1_EST = torch.zeros(1)
        elif ftype == 1:
            f1_EST = f1_OBS_fn(t_OBS_vals[-2], t_OBS_vals[-3], F_EST)
        elif ftype == 2:
            f1_EST = f1_EST_fn(torch.cat([t_OBS_vals[-2], t_dot_OBS, F_EST]))
        else:
            raise NotImplementedError()

        # compute theta_hat (t) at t
        t_OBS = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_OBS)
        t_EST = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_EST)

        # store values to containers
        t_OBS_vals[-4] = t_OBS_vals[-3]
        t_OBS_vals[-3] = t_OBS_vals[-2]
        t_OBS_vals[-2] = t_OBS_vals[-1]
        t_OBS_vals[-1] = t_OBS
        f1_EST_vals[-1] = f1_EST

        # store history for plotting
        time_stamp.append(float(t / params['freq']))
        target_history.append(float(target.numpy()))
        obs_history.append(float(t_OBS.numpy()))
        est_history.append(float(t_EST.detach().numpy()))
        f1_obs_history.append(float(f1_OBS.detach().numpy()))
        f1_est_history.append(float(f1_EST.detach().numpy()))
        F_est_history.append(float(F_EST.detach().numpy()))

        # for debugging
        # if np.isnan(t_OBS.numpy()):
        #     break

    # store values for post-visualization
    params_dir = os.path.join(args.model_dir, args.data_type)
    if ftype == 0:
        eval_log_dir = os.path.join(params_dir, 'f_est=0', 'evaluation',
                                    args.eval_type)
    elif ftype == 1:
        eval_log_dir = os.path.join(params_dir, 'oracle', 'evaluation',
                                    args.eval_type)
    elif ftype == 2:
        eval_log_dir = os.path.join(params_dir, 'MLP', 'evaluation',
                                    args.eval_type)
    if not os.path.exists(eval_log_dir):
        os.makedirs(eval_log_dir)
    store_logs(time_stamp, target_history, obs_history, est_history,
               f1_obs_history, f1_est_history, F_est_history, eval_log_dir)

    # visualize
    plot_theta(time_stamp, target_history, obs_history, est_history,
               eval_log_dir)


if __name__ == '__main__':
    const_path = os.path.join(args.model_dir, args.data_type, 'const.json')
    params_path = os.path.join(args.model_dir, args.data_type, 'params.json')
    with open(const_path, 'r') as f:
        const = json.load(f)
    with open(params_path, 'r') as f:
        params = json.load(f)

    const = cast_dict_to_tensor(const)
    evaluate(const, params, ftype=args.ftype)
