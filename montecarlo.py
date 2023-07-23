import numpy as np

from lattice2d import LatticeGaugeTheory2D, plaquette
from su2matrices import SU2Matrix


class LatticeMetropolis:

    def __init__(self, lattice: LatticeGaugeTheory2D, seed=None, step_size=0.1):
        self.lattice = lattice
        self.step_size = step_size

        np.random.seed(seed)

    def site_step(self, x_index, y_index):
        """
        Performs the metropolis update at the site level. The upwards and downwards links will both have candidate
        updates carried out and the metropolis test will be carried out on each one and each change will either be
        accepted or rejected.
        """
        # the matrices to attempt to update
        right_link = self.lattice.rightward_link(y_index, x_index)
        up_link = self.lattice.upward_link(y_index, x_index)

        # calculates the plaquettes in the co-boundary of the links
        up_plaquette_cobound = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index - 1, y_index)
        right_plaquette_cobound = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index, y_index - 1)

        # generates candidate updates
        candidate_right_link = self.matrix_shift(right_link)
        candidate_up_link = self.matrix_shift(up_link)
        self.lattice.update_rightward_link(y_index, x_index, candidate_right_link)
        self.lattice.update_upward_link(y_index, x_index, candidate_up_link)

        # calculates the plaquettes in the co-boundary of the links
        up_plaquette_cobound_new = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index - 1, y_index)
        right_plaquette_cobound_new = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index, y_index - 1)

        # finds the change in action for both updates
        delta_S_up = up_plaquette_cobound_new - up_plaquette_cobound
        delta_S_right = right_plaquette_cobound_new - right_plaquette_cobound

        # perform metropolis test, keeping change if accepted and reverting if rejected
        if not self.metropolis_test(delta_S_up):
            self.lattice.update_upward_link(y_index, x_index, up_link)
        if not self.metropolis_test(delta_S_right):
            self.lattice.update_rightward_link(y_index, x_index, up_link)

    def matrix_shift(self, matrix):
        """
        Generates an update to the SU2Matrix provided, by generating a random SU2Matrix with two random, complex entries
        that have a real and imaginary part smaller than the step size.

        :param matrix: the SU2Matrix to update.
        """
        # generate a random special unitary matrix
        # these are generated with a standard deviation of the step size, with the mean being the identity matrix
        a_r = np.random.normal(1, self.step_size, 1)[0]
        a_i, b_r, b_i = np.random.normal(0, self.step_size, 3)

        a = a_r + 1j * a_i
        b = b_r + 1j * b_i

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
        Performs the Metropolis test. Generates some random number r el [0, 1). If r < exp(-Delta S) then the change
        will be accepted. If r > exp(-Delta S) then the change will be rejected. If Delta S < 0 the change is
        immediately accepted.

        :param delta_S: the change in the action for the proposed update.
        :returns: True if accepted.
        """
        r = np.random.rand()

        if delta_S <= 0:
            return True

        return r < np.exp(-delta_S)
