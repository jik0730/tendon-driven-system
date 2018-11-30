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
from friction_true import RealFriction
from visualization import plot_theta

# Define system constants
# TODO accurate values
const = {
    'L': torch.FloatTensor([0.1]),
    'b': torch.FloatTensor([0.07]),
    'M': torch.FloatTensor([0.1]),
    'g': torch.FloatTensor([9.81]),
    'k': torch.FloatTensor([10.]),
    'c': torch.FloatTensor([0.07]),
    'I': None,
    'T_d': None,
    'del_t': None,
    'a0': torch.FloatTensor([0.25]),
    'a1': torch.FloatTensor([0.09]),
    'a2': torch.FloatTensor([0.00005]),
    'v0': torch.FloatTensor([20.]),
    'A': torch.FloatTensor([1.])
}
const['I'] = (0.25) * const['M'] * const['L']**2

const = {
    'L': torch.FloatTensor([1.]),
    'b': torch.FloatTensor([1.]),
    'M': torch.FloatTensor([1.]),
    'g': torch.FloatTensor([1.]),
    'k': torch.FloatTensor([1.]),
    'c': torch.FloatTensor([1.]),
    'I': torch.FloatTensor([1.]),
    'T_d': None,
    'del_t': None,
    'a0': torch.FloatTensor([0.25]),
    'a1': torch.FloatTensor([0.09]),
    'a2': torch.FloatTensor([0.00005]),
    'v0': torch.FloatTensor([20.]),
    'A': torch.FloatTensor([1.])
}

# Define hyper-parameters
H_DIM = 8
lr = 1e-2


def main():
    # During some time steps...
    # 1. Compute theta_hat by f_TRUE (1)
    # 2. Compute theta_hat by f_EST (2)
    # 3. Compute loss between (1) and (2)
    # 4. Optimize parameters of f_EST

    # Simulation parameters and intial values
    T = 50000
    const['del_t'] = float(1 / T)
    Kp = 1.1
    Kd = 0.004 * Kp
    Ki = 1. * Kp
    pid_cont = PIDController(p=Kp, i=Ki, d=Kd, del_t=const['del_t'] * 10)

    # Define models TODO for now we ignore f2.
    f1_OBS_fn = RealFriction(const)
    # f1_OBS_fn = ArbitraryFriction(1)
    f1_EST_fn = Friction_EST(H_DIM)

    # Define loss_fn, optimizer
    optimizer = torch.optim.Adam(f1_EST_fn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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
    t_OBS_vals_F = t_EST_vals[-3]
    f1_F = f1_EST_vals[-1]

    # for loop for t=4, ...
    target_pt = torch.FloatTensor([10.])  # degree
    target_traj = [
        torch.FloatTensor([10.]),
        torch.FloatTensor([20.]),
        torch.FloatTensor([30.]),
        torch.FloatTensor([-5.]),
        torch.FloatTensor([-20.]),
        torch.FloatTensor([0.])
    ]
    target_pt = degree_to_radian(target_pt)  # radian
    target_traj = degree_to_radian(target_traj)

    # NOTE 0 if f_est=0, 1 if f_est is oracle, 2 if f_est is MLP
    friction_type = 1

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
            const['T_d'] = pid_cont.compute_torque(target, t_OBS_vals_F)
            F_EST = compute_input_force(const, t_OBS_vals_F, f1_F)
            t_OBS_vals_F = t_OBS_vals[-3]
            f1_F = f1

        # compute frictions (f) at t-2
        t_dot_OBS = (t_OBS_vals[-3] - t_OBS_vals[-4]) / const['del_t']
        # t_dot_EST = (t3 - t4) / const['del_t']
        f1_OBS = f1_OBS_fn(t_OBS_vals[-3], t_OBS_vals[-4], F_EST)
        # f1_OBS = f1_OBS_fn(torch.cat([t_OBS_vals[-3], t_dot_OBS, F_EST]))

        if friction_type == 0:
            f1_EST = torch.zeros(1)
        elif friction_type == 1:
            f1_EST = f1_OBS_fn(t_OBS_vals[-3], t_OBS_vals[-4], F_EST)
        elif friction_type == 2:
            f1_EST = f1_EST_fn(torch.cat([t_OBS_vals[-3], t_dot_OBS, F_EST]))
        else:
            raise NotImplementedError()

        # compute theta_hat (t) at t
        t_OBS = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_OBS)
        t_EST = compute_theta_hat(const, t_OBS_vals[-1], t_OBS_vals[-2],
                                  t_OBS_vals[-3], F_EST, f1_EST)

        # Optimization
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
        f1_OBS_vals[-1] = f1_OBS
        t_EST_vals[-4] = t_EST_vals[-3]
        t_EST_vals[-3] = t_EST_vals[-2]
        t_EST_vals[-2] = t_EST_vals[-1]
        t_EST_vals[-1] = t_EST
        f1_EST_vals[-1] = f1_EST

        # store history for plotting
        target_history.append(target)
        obs_history.append(float(t_OBS.numpy()))
        est_history.append(float(t_EST.detach().numpy()))

    # visualize
    plot_theta(target_history, obs_history, est_history)


if __name__ == '__main__':
    main()