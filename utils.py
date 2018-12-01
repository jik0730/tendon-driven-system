import torch
import math
import os
import pickle as pkl
from scipy.io import savemat

const = {
    'L': torch.FloatTensor([0.1]),
    'b': torch.FloatTensor([0.07]),
    'M': torch.FloatTensor([0.1]),
    'g': torch.FloatTensor([9.81]),
    'k': torch.FloatTensor([100.]),
    'c': torch.FloatTensor([0.07]),
    'I': None,
    'T_d': None,
    'del_t': None,
    'a0': torch.FloatTensor([0.22]),
    'a1': torch.FloatTensor([0.13]),
    'a2': torch.FloatTensor([0.12]),
    'v0': torch.FloatTensor([0.04]),
    'A': torch.FloatTensor([0.5])
}
const['I'] = (0.25) * const['M'] * const['L']**2


def compute_alpha(L, b, theta):
    """
    Compute alpha at time t under the system.
    Args:
        L, b: torch constant
        theta: torch constant representing current angle
    """
    out = torch.cos(theta) * L
    out = out / torch.sqrt(L**2 + b**2 + 2 * b * L * torch.sin(theta))
    out = torch.asin(out)

    if -L * torch.sin(theta) > b:
        return torch.FloatTensor([math.pi]) - out
    else:
        return out


def compute_beta(L, b, theta):
    """
    Compute beta at time t under the system.
    Args:
        L, b: torch constant
        theta: torch constant representing current angle
    """
    out = torch.cos(theta) * L
    out = out / torch.sqrt(L**2 + b**2 - 2 * b * L * torch.sin(theta))
    if out > 1:
        out = torch.ones(1)
    out = torch.asin(out)
    # print(out)

    if L * torch.sin(theta) > b:
        return torch.FloatTensor([math.pi]) - out
    else:
        return out


def compute_input_force(const, theta_1, f1_1, f2_1=torch.zeros(1)):
    """
    Compute F (input_force) at time t under the system given theta_t-1 and f_t-1.
    theta_1 = theta_t-1
    f1_1 = f1_t-1
    f2_1 = f2_t-1
    Args:
        L, b, M, g, k, c: torch constant
        T_d: torch constant computed by PID controller (T_desired)
        theta_1: torch constant representing current angle (t-1)
    Return:
        F_t | theta_t-1, f_t-1
    """
    L = const['L']
    b = const['b']
    M = const['M']
    g = const['g']
    k = const['k']
    c = const['c']
    T_d = const['T_d']

    alpha = compute_alpha(L, b, theta_1)
    beta = compute_beta(L, b, theta_1)

    out = torch.sqrt(L**2 + b**2 + 2 * b * L * torch.sin(theta_1))
    out = k * (c + out - torch.sqrt(L**2 + b**2)) - f2_1
    out = out * L * torch.cos(theta_1 + alpha)
    out = out + T_d - (M * g * L * torch.sin(theta_1)) / 2.
    out = out / (L * torch.cos(theta_1 - beta))
    out = out - f1_1

    if out < 0:
        return torch.zeros(1)
    else:
        return out


def compute_theta_hat(const,
                      theta_1,
                      theta_2,
                      theta_3,
                      F_1,
                      f1_1,
                      f2_1=torch.zeros(1)):
    """
    Compute theta_hat at time t under the system given theta_t-1 and 
    theta_dot_t-2 and theta_dotdot_t-2.
    theta_1 = theta_t-1
    theta_2 = theta_t-2
    theta_3 = theta_t-3
    f1_1 = f1_t-1
    f2_1 = f2_t-1
    F_1 = F_t-1
    Args:
        L, b, M, g, k, c, I: torch constant
        theta_1, 2: torch constant representing current angle at proper time step
        f1, f2, F: frictions and input force
        del_t: sampling time
    """
    L = const['L']
    b = const['b']
    M = const['M']
    g = const['g']
    k = const['k']
    c = const['c']
    I = const['I']
    del_t = const['del_t']

    alpha = compute_alpha(L, b, theta_2)
    beta = compute_beta(L, b, theta_2)

    out = torch.sqrt(L**2 + b**2 + 2 * b * L * torch.sin(theta_2))
    out = k * (c + out - torch.sqrt(L**2 + b**2)) - f2_1
    out = out * L * torch.cos(theta_2 + alpha)
    out = (M * g * L * torch.sin(theta_2)) / 2. - out
    out = out + (F_1 + f1_1) * L * torch.cos(theta_2 - beta)
    out = out / I  # theta_dotdot_t

    theta_dot_2 = (theta_2 - theta_3) / del_t  # t-2
    out = theta_dot_2 + out * del_t  # theta_dot_t-1
    out = theta_1 + out * del_t

    return out


def degree_to_radian(deg):
    if type(deg) == list:
        return [d / 180. * math.pi for d in deg]
    else:
        return deg / 180. * math.pi


def current_target(target, t_cur, T):
    if type(target) == list:
        # TODO
        N = len(target)
        quotient, remainder = T // N, T % N
        if t_cur // quotient == 0:
            pass
        else:
            idx = t_cur // quotient
    else:
        return target


def store_logs(time_stamp, target_history, obs_history, est_history,
               f1_obs_history, f1_est_history, F_est_history, model_dir):
    with open(os.path.join(model_dir, 'time_stamp.pkl'), 'wb') as f:
        pkl.dump(time_stamp, f)
    with open(os.path.join(model_dir, 'target_history.pkl'), 'wb') as f:
        pkl.dump(target_history, f)
    with open(os.path.join(model_dir, 'obs_history.pkl'), 'wb') as f:
        pkl.dump(obs_history, f)
    with open(os.path.join(model_dir, 'est_history.pkl'), 'wb') as f:
        pkl.dump(est_history, f)
    with open(os.path.join(model_dir, 'f1_obs_history.pkl'), 'wb') as f:
        pkl.dump(f1_obs_history, f)
    with open(os.path.join(model_dir, 'f1_est_history.pkl'), 'wb') as f:
        pkl.dump(f1_est_history, f)
    with open(os.path.join(model_dir, 'F_est_history.pkl'), 'wb') as f:
        pkl.dump(F_est_history, f)

    mdict = {
        'time_stamp': time_stamp,
        'target_history': target_history,
        'obs_history': obs_history,
        'est_history': est_history,
        'f1_obs_history': f1_obs_history,
        'f1_est_history': f1_est_history,
        'F_est_history': F_est_history
    }
    savemat(os.path.join(model_dir, 'logs.mat'), mdict=mdict)


def cast_dict_to_float(dictionary):
    for key, val in dictionary.items():
        if type(val) == torch.Tensor:
            dictionary[key] = float(val.detach().numpy())
    return dictionary