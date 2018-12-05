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


def visualize_simulation(time_stamp, target, theta, sim_T=None):
    # initial constant
    L = 0.1
    b = 0.07
    freq = 10000  # NOTE hard-coding
    if not sim_T:
        sim_T = int(time_stamp[-1]) + 1
    time_stamp = time_stamp[:min(freq * sim_T, len(time_stamp))]
    target = target[:min(freq * sim_T, len(target))]
    theta = theta[:min(freq * sim_T, len(theta))]

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
    # plt.show()
    return anim


def visualize_simulation_all(time_stamp, target, thetas, sim_T=None):
    # initial constant
    L = 0.1
    b = 0.07
    freq = 10000  # NOTE hard-coding
    if not sim_T:
        sim_T = int(time_stamp[-1]) + 1

    def adjust_simT(dic, sim_T):
        a = dic['f0']
        b = dic['MLP']
        c = dic['ora']
        dic['f0'] = a[:min(freq * sim_T, len(a))]
        dic['MLP'] = b[:min(freq * sim_T, len(b))]
        dic['ora'] = c[:min(freq * sim_T, len(c))]
        return

    time_stamp = time_stamp[:min(freq * sim_T, len(time_stamp))]
    target = target[:min(freq * sim_T, len(target))]
    adjust_simT(thetas, sim_T)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-0.13, 0.13), ylim=(-0.02, 0.13))
    ax.grid(linestyle='--')
    origin_stick_f0, = ax.plot([], [],
                               'o-',
                               lw=2,
                               c='green',
                               label='[f_est=0]theta_obs',
                               alpha=0.7)
    origin_stick_MLP, = ax.plot([], [],
                                'o-',
                                lw=2,
                                c='blue',
                                label='[MLP]theta_obs',
                                alpha=0.7)
    # origin_stick_ora, = ax.plot([], [],
    #                             'o-',
    #                             lw=2,
    #                             c='green',
    #                             label='[oracle]theta_obs',
    #                             alpha=0.5)
    origin_target, = ax.plot([], [], 'o--', lw=2, c='red', label='target')
    origin_F, = ax.plot([], [], 'o-', lw=2, c='black')
    origin_spring, = ax.plot([], [], 'o-', lw=2, c='black')
    F_theta, = ax.plot([], [], '--', lw=1, c='black')
    spring_theta, = ax.plot([], [], '--', lw=1, c='black')
    F_arrow = ax.arrow(b, 0, 0.1 - b, 0, color='black')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # initialization function: plot the background of each frame
    def init():
        origin_stick_f0.set_data([], [])
        origin_stick_MLP.set_data([], [])
        # origin_stick_ora.set_data([], [])
        origin_target.set_data([], [])
        origin_F.set_data([], [])
        origin_spring.set_data([], [])
        F_theta.set_data([], [])
        spring_theta.set_data([], [])
        time_text.set_text('')
        return origin_F, origin_spring, F_theta, spring_theta, F_arrow, origin_target, origin_stick_f0, origin_stick_MLP, time_text

    # animation function.  This is called sequentially
    scale = 100

    def animate(i):
        i = i * scale
        origin_stick_f0.set_data([0, L * np.sin(thetas['f0'][i])],
                                 [0, L * np.cos(thetas['f0'][i])])
        origin_stick_MLP.set_data([0, L * np.sin(thetas['MLP'][i])],
                                  [0, L * np.cos(thetas['MLP'][i])])
        # origin_stick_ora.set_data([0, L * np.sin(thetas['ora'][i])],
        #                           [0, L * np.cos(thetas['ora'][i])])
        origin_target.set_data([0, L * np.sin(target[i])],
                               [0, L * np.cos(target[i])])
        origin_F.set_data([0, b], [0, 0])
        origin_spring.set_data([0, -b], [0, 0])
        F_theta.set_data([b, L * np.sin(thetas['MLP'][i])],
                         [0, L * np.cos(thetas['MLP'][i])])
        spring_theta.set_data([-b, L * np.sin(thetas['MLP'][i])],
                              [0, L * np.cos(thetas['MLP'][i])])
        time_text.set_text('time: {}'.format(time_stamp[i]))
        return origin_F, origin_spring, F_theta, spring_theta, F_arrow, origin_target, origin_stick_f0, origin_stick_MLP, time_text

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
    # plt.show()
    return anim


if __name__ == '__main__':
    # model_dir = sys.argv[1]
    # time_stamp_file = os.path.join(model_dir, 'time_stamp.pkl')
    # target_file = os.path.join(model_dir, 'target_history.pkl')
    # theta_file = os.path.join(model_dir, 'obs_history.pkl')
    # with open(time_stamp_file, 'rb') as f:
    #     time_stamp = pkl.load(f)
    # with open(target_file, 'rb') as f:
    #     target = pkl.load(f)
    # with open(theta_file, 'rb') as f:
    #     theta = pkl.load(f)

    # sim_T = int(sys.argv[2])
    # gif = visualize_simulation(time_stamp, target, theta, sim_T=sim_T)
    # gif_file = os.path.join(sys.argv[1], 'visualization.gif')
    # gif.save(gif_file, fps=60, dpi=100, writer='imagemagick')

    model_dir = sys.argv[1]
    time_stamp_file = os.path.join(model_dir, 'MLP/training', 'time_stamp.pkl')
    target_file = os.path.join(model_dir, 'MLP/training', 'target_history.pkl')
    theta_file_f0 = os.path.join(model_dir, 'f_est=0/training',
                                 'obs_history.pkl')
    theta_file_MLP = os.path.join(model_dir, 'MLP/training', 'obs_history.pkl')
    theta_file_ora = os.path.join(model_dir, 'oracle/training',
                                  'obs_history.pkl')
    thetas = {}
    with open(time_stamp_file, 'rb') as f:
        time_stamp = pkl.load(f)
    with open(target_file, 'rb') as f:
        target = pkl.load(f)
    with open(theta_file_f0, 'rb') as f:
        thetas['f0'] = pkl.load(f)
    with open(theta_file_MLP, 'rb') as f:
        thetas['MLP'] = pkl.load(f)
    with open(theta_file_ora, 'rb') as f:
        thetas['ora'] = pkl.load(f)

    sim_T = int(sys.argv[2])
    gif = visualize_simulation_all(time_stamp, target, thetas, sim_T=sim_T)
    gif_file = os.path.join(sys.argv[1], 'visualization.gif')
    gif.save(gif_file, fps=60, dpi=100, writer='imagemagick')