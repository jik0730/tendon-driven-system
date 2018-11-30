import torch
import math


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
    out = torch.asin(out)
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

    return out


def compute_theta_hat(const,
                      theta_1,
                      theta_2,
                      theta_3,
                      F_2,
                      f1_2,
                      f2_2=torch.zeros(1)):
    """
    Compute theta_hat at time t under the system given theta_t-1 and 
    theta_dot_t-2 and theta_dotdot_t-2.
    theta_1 = theta_t-1
    theta_2 = theta_t-2
    theta_dot_2 = theta_dot_t-2
    f1_2 = f1_t-2
    f2_2 = f2_t-2
    F_2 = F_t-2
    Args:
        L, b, M, g, k, c, I: torch constant
        theta_1, 2: torch constant representing current angle at proper time step
        theta_dot_2: angle velocity
        f1_2, f2_2, F: frictions and input force
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
    theta_dot_2 = (theta_2 - theta_3) / del_t

    alpha = compute_alpha(L, b, theta_2)
    beta = compute_beta(L, b, theta_2)

    out = torch.sqrt(L**2 + b**2 + 2 * b * L * torch.sin(theta_2))
    out = k * (c + out - torch.sqrt(L**2 + b**2)) - f2_2
    out = out * L * torch.cos(theta_2 + alpha)
    out = (M * g * L * torch.sin(theta_2)) / 2. - out
    out = out + (F_2 + f1_2) * L * torch.cos(theta_2 - beta)
    out = out / I  # theta_dotdot_t-2

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