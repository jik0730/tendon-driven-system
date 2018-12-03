import torch
import math
import matplotlib.pyplot as plt
from utils import degree_to_radian
from random import seed
from random import random


def point_target(degree):
    target_pt = torch.FloatTensor([degree])
    target_pt = degree_to_radian(target_pt)
    return target_pt


def arbitrary_target_traj(degrees):
    target_traj = [torch.FloatTensor([deg]) for deg in degrees]
    target_traj = degree_to_radian(target_traj)
    return target_traj


def sin_target_traj(sys_freq, simT, sine_type=None):
    split = sine_type.split('_')
    desired_freq = float(split[1][:-2])
    max_degree = float(split[2][:-3])
    offset = float(split[3][:-6]) / 180 * math.pi  # radian

    T = sys_freq * simT
    amp = max_degree / 180 * math.pi  # 1 radian = 57.3 degree
    target_sin_traj = amp * torch.sin(
        torch.linspace(0, simT, steps=T) * desired_freq * 2. *
        math.pi) + offset
    return target_sin_traj


def sin_target_traj_manual(sys_freq, simT, desired_freq, max_degree):
    T = sys_freq * simT
    amp = max_degree / 57.3  # 1 radian = 57.3 degree
    target_sin_traj = amp * torch.sin(
        torch.linspace(0, simT, steps=T) * desired_freq * 2. * math.pi)
    return target_sin_traj


def sin_freq_variation(freq_from, freq_to, sys_freq, simT, sine_type=None):
    split = sine_type.split('_')
    max_degree = float(split[2][:-3])
    amp = max_degree / 180 * math.pi
    T = int(sys_freq * simT)

    t = 0.
    freq = freq_from
    linspace = torch.linspace(0, simT, steps=T)
    idx_from = 0
    target_traj = []
    while t < simT:
        idx_to = min(int(1. / freq * sys_freq) + idx_from, T)
        print('1/freq: {}, t: {}, del_idx: {}'.format(1. / freq, t,
                                                      idx_to - idx_from))
        sin_traj = amp * torch.sin(
            linspace[0:idx_to - idx_from] * freq * 2. * math.pi)
        target_traj.append(sin_traj)
        idx_from = idx_to
        t += 1. / freq
        freq = freq * 20**(1 / 19.)
    target_traj = torch.cat(target_traj)
    # print(target_traj.size())
    # plt.plot(target_traj.numpy())
    # plt.show()
    return target_traj


def sin_freq_variation_with_step(freq_from,
                                 freq_to,
                                 sys_freq,
                                 simT,
                                 sine_type=None):
    split = sine_type.split('_')
    max_degree = float(split[2][:-3])
    amp = max_degree / 180 * math.pi
    T = int(sys_freq * simT)

    t = 0.
    freq = freq_from
    linspace = torch.linspace(0, simT, steps=T)
    idx_from = 0
    target_traj = []
    while t < simT:
        idx_to = min(int(1. / freq * sys_freq) + idx_from, T)
        print('1/freq: {}, t: {}, del_idx: {}'.format(1. / freq, t,
                                                      idx_to - idx_from))
        sin_traj = amp * torch.sin(
            linspace[0:idx_to - idx_from] * freq * 2. * math.pi)
        target_traj.append(sin_traj)
        # add step
        step_traj = step_target_traj(2 * sys_freq, 'step_0deg')
        target_traj.append(step_traj)
        t += 2.
        idx_from = idx_to
        t += 1. / freq
        freq = freq * 20**(1 / 19.)
    target_traj = torch.cat(target_traj)
    # print(target_traj.size())
    # plt.plot(target_traj.numpy())
    # plt.show()
    return target_traj


def random_walk(T, data_type):
    split = data_type.split('_')
    max_degree = float(split[2][:-3])
    SEED = int(split[3][:-4])

    seed(SEED)
    random_walk = list()
    delta_movement = random() * 0.2  # movement is in (0, 0.2)
    random_walk.append(-delta_movement if random() < 0.5 else delta_movement)
    for i in range(1, T):
        delta_movement = random() * 0.2  # movement is in (0, 0.2)
        movement = -delta_movement if random() < 0.5 else delta_movement
        if random_walk[i - 1] + movement > max_degree:
            value = random_walk[i - 1] - movement
        elif random_walk[i - 1] + movement < -max_degree:
            value = random_walk[i - 1] - movement
        else:
            value = random_walk[i - 1] + movement
        random_walk.append(torch.FloatTensor([value]))
    # plt.plot(random_walk)
    # plt.show()
    return degree_to_radian(random_walk)


def step_target_traj(T, data_type):
    split = data_type.split('_')  # step_10deg
    degree = float(split[1][:-3])
    target_traj = torch.ones(T) * degree
    target_traj = degree_to_radian(target_traj)
    return target_traj


if __name__ == '__main__':
    freq_from = 0.5
    freq_to = 10.
    sys_freq = 10000
    simT = 30
    sine_type = 'sine_1Hz_10deg_0offset'
    sin_freq_variation_with_step(freq_from, freq_to, sys_freq, simT, sine_type)
    # random_walk(100000, 'random_walk_30deg_1seed')