import unittest

import numpy as np

from lattice2d import LatticeGaugeTheory2D
from su2matrices import SU2Matrix


class LatticeGaugeTheory2DTest(unittest.TestCase):

    def setUp(self):
        self.lattice = LatticeGaugeTheory2D()
        self.uneven_lattice = LatticeGaugeTheory2D(M=50, N=200)

    def test_new_lattice_correct_size(self):
        lattice2 = LatticeGaugeTheory2D(N=200)
        lattice3 = LatticeGaugeTheory2D(M=200)
        self.assertEqual(len(self.uneven_lattice.link_variables), 50)
        self.assertEqual(len(self.uneven_lattice.link_variables[0]), 200)
        self.assertEqual(len(self.lattice.link_variables), 20)
        self.assertEqual(len(self.lattice.link_variables[0]), 20)
        self.assertEqual(len(lattice2.link_variables), 20)
        self.assertEqual(len(lattice2.link_variables[0]), 200)
        self.assertEqual(len(lattice3.link_variables), 200)
        self.assertEqual(len(lattice3.link_variables[0]), 20)

    def test_width_height_methods(self):
        self.assertEqual(self.uneven_lattice.lattice_height(), 50)
        self.assertEqual(self.uneven_lattice.lattice_width(), 200)
        self.assertEqual(self.lattice.lattice_height(), 20)
        self.assertEqual(self.lattice.lattice_width(), 20)

    def test_random_initialisation_correct(self):
        # generate 8 SU(2) matrices from a seed
        matrices = []
        for i in range(0, 8):
            matrices.append(SU2Matrix(seed=42))

        # reset seed back to beginning
        np.random.seed(42)

        # generate a lattice using this seed and check all matrices are expected
        lattice = LatticeGaugeTheory2D(M=2, N=2, random_seed=42)
        for i in range(0, 2):
            for j in range(0, 2):
                self.assertEqual(lattice.links_for_site(i, j)[0], matrices[2 * (i+j)])
                self.assertEqual(lattice.links_for_site(i, j)[1], matrices[2 * (i + j) + 1])

    def test_seed_ignored_when_initialisation_matrix_supplied(self):
        mat1 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        mat2 = SU2Matrix(a=0, b=1, c=-1, d=0)
        mat3 = SU2Matrix(a=1j, b=0, c=0, d=-1j)
        mat4 = SU2Matrix(a=1 / np.sqrt(5), b=2j / np.sqrt(5), c=2j / np.sqrt(5), d=1 / np.sqrt(5))

        lattice1 = LatticeGaugeTheory2D(M=2, N=2, uniform_initialisation_matrix=mat1)
        lattice2 = LatticeGaugeTheory2D(M=2, N=2, uniform_initialisation_matrix=mat2)
        lattice3 = LatticeGaugeTheory2D(M=2, N=2, uniform_initialisation_matrix=mat3)
        lattice4 = LatticeGaugeTheory2D(M=2, N=2, uniform_initialisation_matrix=mat4)

        for i in range(1, 5):
            for j in range(0, 2):
                for k in range(0, 2):
                    self.assertEqual(eval(f'lattice{i}').links_for_site(j, k)[0], eval(f'mat{i}'))
                    self.assertEqual(eval(f'lattice{i}').links_for_site(j, k)[1], eval(f'mat{i}'))

    def test_PBCs(self):
        M = self.lattice.lattice_height()
        N = self.lattice.lattice_width()

        # test wrap around in both directions
        self.assertTupleEqual(self.lattice.index_with_PBCs(M, 0), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, N), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(M + 1, 0), (1, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, N + 1), (0, 1))

        # test wrap around for index > 2 * size
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, 2 * N), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(2 * M, 0), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, 2 * N - 1), (0, N - 1))
        self.assertTupleEqual(self.lattice.index_with_PBCs(2 * M - 1, 0), (M - 1, 0))

        # test for negative indices
        self.assertTupleEqual(self.lattice.index_with_PBCs(-1, 0), (M - 1, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, -1), (0, N - 1))
        self.assertTupleEqual(self.lattice.index_with_PBCs(-M, 0), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, -N), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(-M + 1, 0), (1, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, -N + 1), (0, 1))
        self.assertTupleEqual(self.lattice.index_with_PBCs(-M - 1, 0), (M - 1, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, -N - 1), (0, N - 1))
        self.assertTupleEqual(self.lattice.index_with_PBCs(-2 * M, 0), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, -2 * N), (0, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(-2 * M + 1, 0), (1, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, -2 * N + 1), (0, 1))
        self.assertTupleEqual(self.lattice.index_with_PBCs(-2 * M - 1, 0), (M - 1, 0))
        self.assertTupleEqual(self.lattice.index_with_PBCs(0, -2 * N - 1), (0, N - 1))

        # test wrap around can occur in both directions simultaneously
        self.assertTupleEqual(self.lattice.index_with_PBCs(-1, N), (M - 1, 0))

    def test_links_for_sites(self):
        self.assertEqual(self.lattice.links_for_site(0, 0).all(), self.lattice.link_variables[0, 0].all())
        self.assertEqual(self.lattice.links_for_site(self.lattice.lattice_width(), self.lattice.lattice_height()).all(),
                         self.lattice.link_variables[0, 0].all())
        self.assertEqual(self.lattice.links_for_site(1, 1).all(), self.lattice.link_variables[1, 1].all())
        self.assertEqual(self.lattice.links_for_site(0, 1).all(), self.lattice.link_variables[0, 1].all())
        self.assertEqual(self.lattice.links_for_site(self.lattice.lattice_width() + 1, 0).all(),
                         self.lattice.link_variables[1, 0].all())

    def test_leftwards_link_inverse_of_rightwards(self):
        # check this works for two rows of the lattice
        self.assertEqual(self.lattice.rightward_link(0, 0), self.lattice.leftward_link(0, 1).inverse())
        self.assertEqual(self.lattice.rightward_link(0, 1), self.lattice.leftward_link(0, 2).inverse())
        self.assertEqual(self.lattice.rightward_link(1, 0), self.lattice.leftward_link(1, 1).inverse())
        self.assertEqual(self.lattice.rightward_link(1, 1), self.lattice.leftward_link(1, 2).inverse())

        # and with wrap-around for both rows
        self.assertEqual(self.lattice.rightward_link(0, self.lattice.lattice_width() - 1),
                         self.lattice.leftward_link(0, 0).inverse())
        self.assertEqual(self.lattice.rightward_link(1, self.lattice.lattice_width() - 1),
                         self.lattice.leftward_link(1, 0).inverse())

    def test_downwards_link_inverse_of_upwards(self):
        # check this works for two columns of the lattice
        self.assertEqual(self.lattice.upward_link(0, 0), self.lattice.downward_link(1, 0).inverse())
        self.assertEqual(self.lattice.upward_link(1, 0), self.lattice.downward_link(2, 0).inverse())
        self.assertEqual(self.lattice.upward_link(0, 1), self.lattice.downward_link(1, 1).inverse())
        self.assertEqual(self.lattice.upward_link(1, 1), self.lattice.downward_link(2, 1).inverse())

        # and with wrap-around for both columns
        self.assertEqual(self.lattice.upward_link(0, 0),
                         self.lattice.downward_link(self.lattice.lattice_height() - 1, 0).inverse())
        self.assertEqual(self.lattice.upward_link(0, 1),
                         self.lattice.downward_link(self.lattice.lattice_height() - 1, 1).inverse())

    def test_rightwards_links_updated(self):
        mat1 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        mat2 = SU2Matrix(a=0, b=1, c=-1, d=0)
        mat3 = SU2Matrix(a=1j, b=0, c=0, d=-1j)
        self.assertNotEqual(self.lattice.rightward_link(0, 0), mat1)
        self.assertNotEqual(self.lattice.rightward_link(0, 0), mat2)
        self.assertNotEqual(self.lattice.rightward_link(0, 0), mat3)
        self.lattice.update_rightward_link(0, 0, mat1)
        self.assertEqual(self.lattice.rightward_link(0, 0), mat1)
        self.lattice.update_rightward_link(0, 0, mat2)
        self.assertEqual(self.lattice.rightward_link(0, 0), mat2)
        self.lattice.update_rightward_link(0, 0, mat3)
        self.assertEqual(self.lattice.rightward_link(0, 0), mat3)

    def test_upwards_links_updated(self):
        mat1 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        mat2 = SU2Matrix(a=0, b=1, c=-1, d=0)
        mat3 = SU2Matrix(a=1j, b=0, c=0, d=-1j)
        self.assertNotEqual(self.lattice.upward_link(0, 0), mat1)
        self.assertNotEqual(self.lattice.upward_link(0, 0), mat2)
        self.assertNotEqual(self.lattice.upward_link(0, 0), mat3)
        self.lattice.update_upward_link(0, 0, mat1)
        self.assertEqual(self.lattice.upward_link(0, 0), mat1)
        self.lattice.update_rightward_link(0, 0, mat2)
        self.assertEqual(self.lattice.upward_link(0, 0), mat2)
        self.lattice.update_upward_link(0, 0, mat3)
        self.assertEqual(self.lattice.upward_link(0, 0), mat3)

    def test_action_is_real(self):
        self.assertEqual(np.real(self.lattice.action()), self.lattice.action())


if __name__ == '__main__':
    unittest.main()
