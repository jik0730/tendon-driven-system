import unittest
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from friction_est import Friction_EST
from friction_true import ArbitraryFriction
from friction_true import SinFriction

N = 100000


class SimpleTest(unittest.TestCase):
    """
    friction_true를 friction_est가 directly 현재 셋팅에서 optimize할 수 있는지 보는 테스트.
    """

    def setUp(self):
        self.f1_TRUE = ArbitraryFriction(1)
        self.f1_TRUE_sin = SinFriction()
        self.f1_EST = Friction_EST(8)
        self.X = torch.rand(N, 1) * 7
        # self.X, _ = torch.sort(self.X, dim=0)
        self.X_supp = torch.zeros(N, 2)
        self.Y = self.f1_TRUE(torch.cat([self.X, self.X_supp], dim=1))
        self.Y_sin = self.f1_TRUE_sin(torch.cat([self.X, self.X_supp], dim=1))
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.f1_EST.parameters(), lr=1e-2)

    @unittest.skip
    def test_simple_case(self):
        plt.figure(1)
        plt.plot(
            self.X.numpy(), self.Y.numpy(), '.', color='red', label='true')

        plt.figure(2)
        for i in range(N):
            supp = torch.zeros(1, 2)
            input = torch.cat([self.X[i].view(-1, 1), supp], dim=1)
            loss = self.loss_fn(self.f1_EST(input), self.Y[i].view(-1, 1))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (i + 1) % int(N / 10) == 0:
                Y_hat = self.f1_EST(torch.cat([self.X, self.X_supp], dim=1))
                loss = self.loss_fn(Y_hat, self.Y)
                print(i + 1, loss.item())
                plt.plot(
                    self.X.numpy(), Y_hat.detach().numpy(), label=str(i + 1))
        plt.legend()
        plt.show()

    # @unittest.skip('It works!')
    def test_sin_case(self):
        """
        It works even when mini_batch=1.
        But if mini_batch=1, the system is slow, which is a problem in real situation.
        Also, if input is in some order, it does not work.
        """
        plt.figure(1)
        plt.plot(
            self.X.numpy(),
            self.Y_sin.numpy(),
            '.',
            color='red',
            label='true',
            markersize=5)

        plt.figure(2)
        update_int = 100  # update_interval
        num_inner_updates = 1  # number of inner updates
        for i in range(N):
            if (i + 1) % update_int == 0:
                X = self.X[i + 1 - update_int:i + 1].view(update_int, -1)
                Y = self.Y_sin[i + 1 - update_int:i + 1].view(update_int, -1)
                supp = torch.zeros(update_int, 2)
                input = torch.cat([X, supp], dim=1)

                # plt.figure(3)
                # plt.plot(X.numpy(), Y.numpy(), '.')
                # plt.show()
                # exit()

                for _ in range(num_inner_updates):
                    loss = self.loss_fn(self.f1_EST(input), Y)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

            if (i + 1) % int(N / 10) == 0:
                Y_hat = self.f1_EST(torch.cat([self.X, self.X_supp], dim=1))
                loss = self.loss_fn(Y_hat, self.Y_sin)
                print(i + 1, loss.item())
                plt.plot(
                    self.X.numpy(),
                    Y_hat.detach().numpy(),
                    '.',
                    label=str(i + 1),
                    color='#2b90d9',
                    alpha=(i + 1) / float(N),
                    markersize=5)
        plt.legend()
        plt.show()

    @unittest.skip('It works.')
    def test_sin_case_batch(self):
        plt.figure(1)
        plt.plot(
            self.X.numpy(), self.Y_sin.numpy(), '.', color='red', label='true')

        plt.figure(2)
        for i in range(1000):
            input = torch.cat([self.X, self.X_supp], dim=1)
            loss = self.loss_fn(self.f1_EST(input), self.Y_sin.view(-1, 1))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (i + 1) % 100 == 0:
                Y_hat = self.f1_EST(torch.cat([self.X, self.X_supp], dim=1))
                loss = self.loss_fn(Y_hat, self.Y_sin)
                print(i + 1, loss.item())
                plt.plot(
                    self.X.numpy(),
                    Y_hat.detach().numpy(),
                    '.',
                    label=str(i + 1),
                    color='#2b90d9',
                    alpha=(i + 1) / 1000.)
        plt.legend()
        plt.show()