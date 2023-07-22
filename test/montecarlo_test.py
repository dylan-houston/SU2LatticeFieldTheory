import unittest

import numpy as np

from montecarlo import LatticeMetropolis
from lattice2d import LatticeGaugeTheory2D
from su2matrices import SU2Matrix


class LatticeMetropolisTest(unittest.TestCase):

    def setUp(self):
        self.lattice = LatticeGaugeTheory2D(random_seed=42)
        self.metropolis = LatticeMetropolis(self.lattice, seed=42)

    def test_metropolis_test(self):
        # Delta S < 0 always accepted
        for _ in range(0, 10):
            rand = np.random.rand() * -10
            self.assertTrue(self.metropolis.metropolis_test(rand))

        # Delta S should be accepted if S = 0
        self.assertTrue(self.metropolis.metropolis_test(0))

        # Generate a candidate r from a seed, then reset the seed so that the same r is generated by LatticeMetropolis
        np.random.seed(42)
        r = np.random.rand()

        # Since acceptance condition is r < e^{-Delta S}, Delta S < ln(1/r) will give acceptance and
        # Delta S >= ln (1/r) will give rejection
        # Delta_S = 2 * ln(1/r), should give rejection
        np.random.seed(42)
        self.assertFalse(self.metropolis.metropolis_test(2 * np.log(1/r)))
        # Delta_S = ln(1/r), should give rejection
        np.random.seed(42)
        self.assertFalse(self.metropolis.metropolis_test(np.log(1/r)))
        # Try Delta_S = 1/2 * ln(1/r), should give acceptance
        np.random.seed(42)
        self.assertTrue(self.metropolis.metropolis_test(1/2 * np.log(1/r)))

    def test_matrix_shift_gives_SU2_matrix(self):
        # generate 100 SU(2) matrices and an update for each
        matrices = np.vectorize(SU2Matrix)(np.empty(100))
        shifted_matrices = np.vectorize(self.metropolis.matrix_shift)(matrices)

        for matrix in shifted_matrices:
            self.assertTrue(matrix.is_special_unitary())

    def test_action_is_real_after_site_metropolis_steps(self):
        # perform 100 metropolis steps, so that some will accept and some will reject
        for _ in range(100):
            random_x = np.random.randint(self.lattice.lattice_width())
            random_y = np.random.randint(self.lattice.lattice_height())
            self.metropolis.site_step(random_x, random_y)
            action = self.lattice.action()
            self.assertEquals(action, np.real(action))
