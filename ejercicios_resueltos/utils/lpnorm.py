import numpy as np


class LP_norm(object):
    def lp_norm(self, m, p=1):
        abs_m = np.abs(m)
        return np.sum(abs_m ** p, axis=1)**(1/p)

    def get_l0(self, m):
        return np.sum(m > 0, axis=1)

    def get_l1(self, m):
        return self.lp_norm(m)

    def get_l2(self, m):
        return self.lp_norm(m, 2)

    def get_linf(self, m):
        return np.max(m, axis=1)