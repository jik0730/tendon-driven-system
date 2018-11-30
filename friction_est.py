import torch
import torch.nn as nn


class Friction_EST(nn.Module):
    """
    The class represents estimated friction function.
    There can be several possible form of the function, but at this point
    we implement 1 hidden layer MLP.
    """

    def __init__(self, hidden_dim):
        super(Friction_EST, self).__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        """
        Args:
            X: (1x3) vector consisted of (theta_t-1, theta_dot_t-1, F_t)
        """
        out = self.fc1(X)
        out = torch.sigmoid(out)
        out = self.fc2(out)
        return out  # TODO out might be strictly positive