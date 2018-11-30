"""
TODO
This .py file can be used for simulation but TENTATIVE..
"""
import simpy


class TendonDrivenSystem(object):
    """
    This class represents simulation of tendon-driven system.
    """

    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())

    def run(self):
        pass
