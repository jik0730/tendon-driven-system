"""
Main function implements actual simulation stuffs.
"""
import math
import os
import argparse
import json
import torch
import torch.nn as nn
from utils import compute_input_force
from utils import compute_theta_hat
from utils import degree_to_radian
from utils import current_target
from utils import const  # system constants
from utils import store_logs
from utils import cast_dict_to_float
from pid_controller import PIDController
from friction_est import Friction_EST
from friction_true import ArbitraryFriction
from friction_true import SinFriction
from friction_true import RealFriction
from visualization import plot_theta
from target_generator import sin_target_traj
from target_generator import random_walk
from collections import OrderedDict

# Define hyper-parameters and simulation parameters
parser = argparse.ArgumentParser()
parser.add_argument('--hdim', default=8, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--freq', default=10000, type=int)
parser.add_argument('--simT', default=10, type=int)  # simulation time (sec)
parser.add_argument('--Kp', default=0.0018, type=float)
parser.add_argument('--Kd', default=0.02, type=float)  # Kd = 0.02 * Kp
parser.add_argument('--Ki', default=0., type=float)  # Ki = 0. * Kp
parser.add_argument('--ftype', default=0, type=int)
parser.add_argument('--model_dir', default='exp/basic_tests')
parser.add_argument('--data_type', default='sine_1Hz_10deg')
args = parser.parse_args()


def main():
    # During some time steps...
    # 1. Compute theta_hat by f_TRUE (1)
    # 2. Compute theta_hat by f_EST (2)
    # 3. Compute loss between (1) and (2)
    # 4. Optimize parameters of f_EST

    # Simulation parameters and intial values
    const['del_t'] = float(1 / args.freq)  # sampling time for dynamics
    T = args.freq * args.simT  # number of operation for dynamics

    # initiate values
    t_OBS_vals = [
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1)
    ]
    f1_EST_vals = [torch.zeros(1)]
    F_EST = torch.zeros(1)

    # Target trajectory
    if 'sine' in args.data_type or 'basic_tests' in args.model_dir:
        target_traj = sin_target_traj(
            args.freq, args.simT, sine_type=args.data_type)
    elif 'random_walk' in args.model_dir:
        target_traj = random_walk(30., T)
    else:
        raise Exception('I dont know your targets')

    # NOTE 0 if f_est=0, 1 if f_est is oracle, 2 if f_est is MLP
    friction_type = args.ftype

    # for plotting
    time_stamp = []
    target_history = []
    obs_history = []
    est_history = []
    f1_obs_history = []
    f1_est_history = []
    F_est_history = []

    # Define PID controller
    Kp = args.Kp
    Kd = args.Kd * Kp
    Ki = args.Ki * Kp
    pid_cont = PIDController(p=Kp, i=Ki, d=Kd, del_t=const['del_t'] * 10)

    # Define models TODO for now we ignore f2.
    f1_OBS_fn = RealFriction(const)
    f1_EST_fn = Friction_EST(args.hdim)

    # Define loss_fn, optimizer
    optimizer = torch.optim.Adam(f1_EST_fn.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

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

        if friction_type == 0:
            f1_EST = torch.zeros(1)
        elif friction_type == 1:
            f1_EST = f1_OBS_fn(t_OBS_vals[-2], t_OBS_vals[-3], F_EST)
        elif friction_type == 2:
            f1_EST = f1_EST_fn(torch.cat([t_OBS_vals[-2], t_dot_OBS, F_EST]))
        else:
            raise NotImplementedError()

        # compute theta_hat (t) at t
        t_OBS = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_OBS)
        t_EST = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_EST)

        # Optimization
        print(t, t_OBS, f1_OBS, F_EST, t_dot_OBS, target)
        if friction_type == 2:
            loss = loss_fn(t_EST, t_OBS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('loss={} at t={}'.format(loss.item(), t))

        # store values to containers
        t_OBS_vals[-4] = t_OBS_vals[-3]
        t_OBS_vals[-3] = t_OBS_vals[-2]
        t_OBS_vals[-2] = t_OBS_vals[-1]
        t_OBS_vals[-1] = t_OBS
        f1_EST_vals[-1] = f1_EST

        # store history for plotting
        time_stamp.append(float(t / args.freq))
        target_history.append(float(target.numpy()))
        obs_history.append(float(t_OBS.numpy()))
        est_history.append(float(t_EST.detach().numpy()))
        f1_obs_history.append(float(f1_OBS.detach().numpy()))
        f1_est_history.append(float(f1_EST.detach().numpy()))
        F_est_history.append(float(F_EST.detach().numpy()))

        # for debugging
        # if float(t_OBS.numpy()) > 3:
        #     break

    # store hyper-parameters and settings and trained model
    params_dir = os.path.join(args.model_dir, args.data_type)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    params = OrderedDict(vars(args))
    const_ord = OrderedDict(cast_dict_to_float(const))
    with open(os.path.join(params_dir, 'params.json'), 'w') as f:
        json.dump(params, f)
    with open(os.path.join(params_dir, 'const.json'), 'w') as f:
        json.dump(const_ord, f)
    model_name = os.path.join(params_dir, 'f1_model')
    torch.save(f1_EST_fn.state_dict(), model_name)

    # store values for post-visualization
    if friction_type == 0:
        training_log_dir = os.path.join(params_dir, 'f_est=0', 'training')
    elif friction_type == 1:
        training_log_dir = os.path.join(params_dir, 'oracle', 'training')
    elif friction_type == 2:
        training_log_dir = os.path.join(params_dir, 'MLP', 'training')
    if not os.path.exists(training_log_dir):
        os.makedirs(training_log_dir)

    store_logs(time_stamp, target_history, obs_history, est_history,
               f1_obs_history, f1_est_history, F_est_history, training_log_dir)

    # visualize
    plot_theta(time_stamp, target_history, obs_history, est_history,
               training_log_dir)


if __name__ == '__main__':
    main()