"""
TODO
Main function implements actual simulation stuffs.
SimPy??
"""
import torch
import torch.nn as nn
from utils import compute_input_force
from utils import compute_theta_hat
from utils import degree_to_radian
from utils import current_target
from pid_controller import PIDController
from friction_est import Friction_EST
from friction_true import ArbitraryFriction
from friction_true import SinFriction
from visualization import plot_theta

# Define system constants
# TODO accurate values
const = {
    'L': torch.FloatTensor([1.]),
    'b': torch.FloatTensor([1.]),
    'M': torch.FloatTensor([1.]),
    'g': torch.FloatTensor([1.]),
    'k': torch.FloatTensor([1.]),
    'c': torch.FloatTensor([1.]),
    'I': torch.FloatTensor([1.]),
    'T_d': torch.FloatTensor([1.]),
    'del_t': None
}

# Define hyper-parameters
H_DIM = 8
lr = 1e-2


def main():
    # Define models TODO for now we ignore f2.
    f1_OBS_fn = ArbitraryFriction(1)
    f1_EST_fn = Friction_EST(H_DIM)

    # Define loss_fn, optimizer
    optimizer = torch.optim.Adam(f1_EST_fn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # During some time steps...
    # 1. Compute theta_hat by f_TRUE (1)
    # 2. Compute theta_hat by f_EST (2)
    # 3. Compute loss between (1) and (2)
    # 4. Optimize parameters of f_EST

    # Simulation parameters and intial values
    T = 10000
    const['del_t'] = float(1 / T)
    Kp = 1.1
    Kd = 0.004 * Kp
    Ki = 1. * Kp
    pid_cont = PIDController(p=Kp, i=Ki, d=Kd, del_t=const['del_t'] * 10)

    # initiate values
    t_OBS_vals = [
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1)
    ]
    f1_OBS_vals = [torch.zeros(1)]
    t_EST_vals = [
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1)
    ]
    f1_EST_vals = [torch.zeros(1)]

    # for loop for t=4, ...
    target_pt = torch.FloatTensor([10.])  # degree
    target_traj = [10., 20., 30, -5., -20., 0.]
    target_pt = degree_to_radian(target_pt)  # radian
    target_traj = degree_to_radian(target_traj)

    use_est = False
    target_history = []
    obs_history = []
    est_history = []

    for t in range(4, T):
        # current target
        target = current_target(target_pt, t, T)

        # detach nodes for simplicity
        t1 = t_EST_vals[-1].detach()
        t2 = t_EST_vals[-2].detach()
        t3 = t_EST_vals[-3].detach()
        t4 = t_EST_vals[-4].detach()
        f1 = f1_EST_vals[-1].detach()

        # compute input force (F) at t-2
        if t % 10 == 0 or t == 4:
            # input t_OBS
            const['T_d'] = pid_cont.compute_torque(target, t_OBS_vals[-3])
            F_EST = compute_input_force(const, t_OBS_vals[-3], f1)

        # compute frictions (f) at t-2
        t_dot_OBS = (t_OBS_vals[-3] - t_OBS_vals[-4]) / const['del_t']
        # t_dot_EST = (t3 - t4) / const['del_t']
        f1_OBS = f1_OBS_fn(torch.cat([t_OBS_vals[-3], t_dot_OBS, F_EST]))
        if use_est:
            f1_EST = f1_EST_fn(torch.cat([t_OBS_vals[-3], t_dot_OBS, F_EST]))
        else:
            f1_EST = torch.zeros(1)

        # compute theta_hat (t) at t
        t_OBS = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_OBS)
        t_EST = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_EST)

        # Optimization
        if use_est:
            print(t_EST, t_OBS)
            loss = loss_fn(t_EST, t_OBS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss={} at t={}'.format(loss.item(), t))

        # store values to containers
        t_OBS_vals[-4] = t_OBS_vals[-3]
        t_OBS_vals[-3] = t_OBS_vals[-2]
        t_OBS_vals[-2] = t_OBS_vals[-1]
        t_OBS_vals[-1] = t_OBS
        f1_OBS_vals[-1] = f1_OBS
        t_EST_vals[-4] = t_EST_vals[-3]
        t_EST_vals[-3] = t_EST_vals[-2]
        t_EST_vals[-2] = t_EST_vals[-1]
        t_EST_vals[-1] = t_EST
        f1_EST_vals[-1] = f1_EST

        # store history for plotting
        target_history.append(target)
        obs_history.append(float(t_OBS.numpy()))
        est_history.append(float(t_EST.numpy()))

    # visualize
    plot_theta(target_history, obs_history, est_history)


if __name__ == '__main__':
    main()