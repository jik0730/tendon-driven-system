"""
Implement pid controller related functions.
"""


class PIDController(object):
    def __init__(self, p=1., i=1., d=1., del_t=0.1):
        self.p = p
        self.i = i
        self.d = d
        self.del_t = del_t
        self.prev_err = 0.
        self.cum_err = 0.

    def compute_torque(self, target, cur):
        err = (cur - target) / self.del_t
        err_dot = (err - self.prev_err) / self.del_t
        self.cum_err += err * self.del_t
        torque = -(self.p * err + self.d * err_dot + self.i * self.cum_err)
        return torque
