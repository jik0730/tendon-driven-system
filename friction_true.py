"""
TODO
Implement true friction function HERE.
"""
import torch
from utils import compute_beta


class ArbitraryFriction(object):
    def __init__(self, SEED):
        torch.manual_seed(SEED)
        self.W = torch.rand(3, 1)

    def __call__(self, X):
        """
        Args:
            X: torch tensor of size 3 (theta_t-1, theta_dot_t-1, F_t)
        """
        return torch.matmul(X, self.W)


class SinFriction(object):
    """
    The friction function is sin function only with the frist dimension.
    """

    def __init__(self):
        pass

    def __call__(self, X):
        return torch.sin(X[:, 0:1])


class RealFriction(object):
    """
    The friction function is similar to the real situation.
    """

    def __init__(self, const):
        self.a0 = const['a0']
        self.a1 = const['a1']
        self.a2 = const['a2']
        self.v0 = const['v0']
        self.A = const['A']
        self.L = const['L']
        self.b = const['b']
        self.del_t = const['del_t']

    def __call__(self, theta, theta_prev, F_est):
        theta_dot = (theta - theta_prev) / self.del_t
        v = self.L**2 + self.b**2 - 2 * self.b * self.L * torch.sin(theta)
        v = 1 / torch.sqrt(v)
        v = 0.5 * v * (-2 * self.b * self.L * torch.cos(theta)) * theta_dot

        mu = self.a0 + self.a1 * torch.exp(-(v / self.v0)**2)
        mu = mu * torch.sign(v) + self.a2 * v
        mu = self.A * mu

        beta = compute_beta(self.L, self.b, theta)
        if theta_dot > 0:
            # NOTE F_est must be positive
            # assert F_est > 0, 'F_est is not positive'
            if F_est > 0:
                F_est = torch.FloatTensor([0.])
            return -F_est * (1 - torch.exp(-mu * beta))
        else:
            return F_est * (torch.exp(mu * beta) - 1)
