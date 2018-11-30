import matplotlib.pyplot as plt

# TODO implement any functions for visualization purpose
# 1. friction function visualization (v, \mu)
# 2. system visualization (그림)
# 3. statistic visualization


def plot_theta(target, obs, est):
    plt.plot(range(len(target)), target, label='target')
    plt.plot(range(len(obs)), obs, label='obs')
    #plt.plot(range(len(est)), est, label='est')

    plt.legend()
    plt.show()
    plt.clf()


def plot_friction():
    # TODO
    pass