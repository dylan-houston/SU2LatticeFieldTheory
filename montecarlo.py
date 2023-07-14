# pick random point
# then pick either up or down link at random
# when finding change in action, only find for co-boundary
# if horizontal link is chosen, co-boundary will be the plaquettes starting at (x, y) and (x, y-1)
# if vertical link is chosen, co-boundary will be plaquettes starting at (x, y) and (x - 1, y)

import numpy as np
from scipy.linalg import expm

from lattice2d import LatticeGaugeTheory2D
from su2matrices import SU2Matrix


class LatticeMetropolis:

    def __init__(self, lattice: LatticeGaugeTheory2D, seed=None, step_size=0.1):
        self.lattice = lattice
        self.step_size = step_size

        np.random.seed(seed)

    def site_step(self, x_index, y_index):
        """
        Performs the metropolis update at the site level. Each link will be updated and the change will either be
        accepted or rejected.
        """
        pass

    def matrix_shift(self, matrix):
        """
        Generates an update to the SU2Matrix provided, by generating a random SU2Matrix with two random, complex entries
        that have a real and imaginary part smaller than the step size.

        :param matrix: the SU2Matrix to update.
        """
        # generate a random special unitary matrix
        # these are generated with a standard deviation of the step size, with the mean being the identity matrix
        a_r = np.random.normal(1, self.step_size, 1)
        a_i, b_r, b_i = np.random.normal(0, self.step_size, 3)

        a = a_r + 1j*a_i
        b = b_r + 1j*b_i
        c = -b_r + 1j*b_i
        d = a_r - 1j*b_i

        V = SU2Matrix(a=a, b=b)

        # this matrix and its hermitian conjugate should be equally probable to ensure detail balance, so randomly
        # choose one
        rand = np.random.rand()
        if rand > 0.5:
            V = V.hermitian_conjugate()

        # create the shifted matrix
        return matrix.right_multiply_by(V)

    def metropolis_test(self, delta_S):
        """
        Performs the Metropolis test. Generates some random number r el [0, 1). Ff r < exp(-Delta S) then the change
        will be accepted. If r > exp(-Delta S) then the change will be rejected. If Delta S < 0 the change is
        immediately accepted.

        :param delta_S: the change in the action for the proposed update.
        """
        r = np.random.rand()

        if delta_S <= 0:
            return True

        return r < np.exp(-delta_S)
