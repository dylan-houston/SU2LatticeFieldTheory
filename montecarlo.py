# pick random point
# then pick either up or down link at random
# when finding change in action, only find for co-boundary
# if horizontal link is chosen, co-boundary will be the plaquettes starting at (x, y) and (x, y-1)
# if vertical link is chosen, co-boundary will be plaquettes starting at (x, y) and (x - 1, y)

import numpy as np

from lattice2d import LatticeGaugeTheory2D


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
        Generates an update to the matrix provided, by generating an NxN matrix with each entry being a random number
        sampled in the range [-step_size, +step_size].

        :param matrix: the NxN matrix to update.
        """
        N = len(matrix)
        assert len(matrix[0]) == N
        random_numbers_real_part = np.random.random_sample((N, N)) * self.step_size
        random_numbers_complex_part = np.random.random_sample((N, N)) * self.step_size

        for i in range(0, N):
            for j in range(0, N):
                rand = np.random.randint(1, 2, 2)
                random_numbers_real_part[i, j] *= (-1) ** rand[0]
                random_numbers_complex_part[i, j] *= (-1) ** rand[1]

        random_numbers = random_numbers_real_part + 1j * random_numbers_complex_part

        return matrix + random_numbers

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
