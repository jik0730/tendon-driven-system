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


def sin_target_traj(max_degree, T):
    amp = max_degree / 57.3  # 1 radian = 57.3 degree
    target_sin_traj = amp * torch.sin(
        torch.linspace(0, T, steps=T) / (T / 2.) * 2. * math.pi)
    return target_sin_traj


def random_walk(max_degree, T, SEED=1):
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
        random_walk.append(value)
    # plt.plot(random_walk)
    # plt.show()
    return degree_to_radian(random_walk)


if __name__ == '__main__':
    max_degree = 30  # positive
    T = 100000
    random_walk(max_degree, T)