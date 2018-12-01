import os
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# TODO implement any functions for visualization purpose
# 1. friction function visualization (v, \mu)
# 2. system visualization (그림)
# 3. statistic visualization


def plot_theta(time_stamp, target, obs, est, model_dir):
    plt.plot(time_stamp, target, label='target')
    plt.plot(time_stamp, obs, label='obs')
    #plt.plot(range(len(est)), est, label='est')

    plt.legend()
    plt.savefig(os.path.join(model_dir, 'theta_visualization'), dpi=200)
    plt.show()
    plt.clf()


def plot_friction():
    # TODO by sihwan
    pass


def visualize_simulation(time_stamp, target, theta):
    # initial constant
    L = 0.1
    b = 0.07
    sim_T = int(time_stamp[-1]) + 1

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-0.13, 0.13), ylim=(-0.02, 0.13))
    ax.grid(linestyle='--')
    origin_stick, = ax.plot([], [], 'o-', lw=2, c='blue', label='theta_obs')
    origin_target, = ax.plot([], [], 'o--', lw=2, c='red', label='target')
    origin_F, = ax.plot([], [], 'o-', lw=2, c='black')
    origin_spring, = ax.plot([], [], 'o-', lw=2, c='black')
    F_theta, = ax.plot([], [], '--', lw=1, c='black')
    spring_theta, = ax.plot([], [], '--', lw=1, c='black')
    F_arrow = ax.arrow(b, 0, 0.1 - b, 0, color='black')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # initialization function: plot the background of each frame
    def init():
        origin_stick.set_data([], [])
        origin_target.set_data([], [])
        origin_F.set_data([], [])
        origin_spring.set_data([], [])
        F_theta.set_data([], [])
        spring_theta.set_data([], [])
        time_text.set_text('')
        return origin_F, origin_spring, F_theta, spring_theta, F_arrow, origin_target, origin_stick, time_text

    # animation function.  This is called sequentially
    scale = 100

    def animate(i):
        i = i * scale
        origin_stick.set_data([0, L * np.sin(theta[i])],
                              [0, L * np.cos(theta[i])])
        origin_target.set_data([0, L * np.sin(target[i])],
                               [0, L * np.cos(target[i])])
        origin_F.set_data([0, b], [0, 0])
        origin_spring.set_data([0, -b], [0, 0])
        F_theta.set_data([b, L * np.sin(theta[i])], [0, L * np.cos(theta[i])])
        spring_theta.set_data([-b, L * np.sin(theta[i])],
                              [0, L * np.cos(theta[i])])
        time_text.set_text('time: {}'.format(time_stamp[i]))
        return origin_F, origin_spring, F_theta, spring_theta, F_arrow, origin_target, origin_stick, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=int(len(time_stamp) / scale),
        interval=1,
        blit=True,
        repeat=True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    time_stamp_file = sys.argv[1]
    target_file = sys.argv[2]
    theta_file = sys.argv[3]
    with open(time_stamp_file, 'rb') as f:
        time_stamp = pkl.load(f)
    with open(target_file, 'rb') as f:
        target = pkl.load(f)
    with open(theta_file, 'rb') as f:
        theta = pkl.load(f)
    visualize_simulation(time_stamp, target, theta)