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

    def test_site_has_group_elements(self):
        for i in range(self.lattice.lattice_width()):
            for j in range(self.lattice.lattice_height()):
                self.assertIsInstance(self.lattice.site_group_element(j, i), SU2Matrix)

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
        self.assertEquals(self.lattice.upward_link(0, 0), self.lattice.downward_link(1, 0).inverse())
        self.assertEquals(self.lattice.upward_link(1, 0), self.lattice.downward_link(2, 0).inverse())
        self.assertEqual(self.lattice.upward_link(0, 1), self.lattice.downward_link(1, 1).inverse())
        self.assertEquals(self.lattice.upward_link(1, 1), self.lattice.downward_link(2, 1).inverse())

        # and with wrap-around for both columns
        self.assertEqual(self.lattice.downward_link(0, 0),
                         self.lattice.upward_link(self.lattice.lattice_height() - 1, 0).inverse())
        self.assertEqual(self.lattice.downward_link(0, 1),
                         self.lattice.upward_link(self.lattice.lattice_height() - 1, 1).inverse())

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
        self.lattice.update_upward_link(0, 0, mat2)
        self.assertEqual(self.lattice.upward_link(0, 0), mat2)
        self.lattice.update_upward_link(0, 0, mat3)
        self.assertEqual(self.lattice.upward_link(0, 0), mat3)

    def test_action_is_real(self):
        self.assertEqual(np.real(self.lattice.action()), self.lattice.action())

    def test_action_is_invariant_under_one_gauge_transformation(self):
        action_before = self.lattice.action()
        actions = np.zeros(100)

        # perform 100 tests. On each test pick a random site and a random link from that site and perform the gauge
        # transformation U_ij -> g_i U_ij (g_j)^(-1). Find the action and save it
        for i in range(100):
            random_x = np.random.randint(self.lattice.lattice_width())
            random_y = np.random.randint(self.lattice.lattice_height())

            # generate a random number in [0, 1), if >=0.5 then use the upwards link, otherwise use the rightward link
            random_link = np.random.rand()
            up = random_link >= 0.5

            if up:
                up_link = self.lattice.upward_link(random_y, random_x)
                transformed_up_link = up_link.left_multiply_by(self.lattice.site_group_element(random_y, random_x))
                transformed_up_link = transformed_up_link.right_multiply_by(
                    self.lattice.site_group_element(random_y + 1, random_x).inverse())

                # check the matrix multiplication was actually carried out correctly
                matrix_prod = self.lattice.site_group_element(random_y, random_x).matrix @ up_link.matrix @ \
                    self.lattice.site_group_element(random_y + 1, random_x).inverse().matrix
                self.assertEqual(transformed_up_link.matrix.all(), matrix_prod.all())

                # check lattice action is the same
                self.lattice.update_upward_link(random_y, random_x, transformed_up_link)
                actions[i] = self.lattice.action()

                # revert back to original state
                self.lattice.update_upward_link(random_y, random_x, up_link)
            else:
                right_link = self.lattice.rightward_link(random_y, random_x)
                transformed_right_link = right_link.left_multiply_by(
                    self.lattice.site_group_element(random_y, random_x))
                transformed_right_link = transformed_right_link.right_multiply_by(
                    self.lattice.site_group_element(random_y, random_x + 1).inverse())

                # check the matrix multiplication was actually carried out correctly
                matrix_prod = self.lattice.site_group_element(random_y, random_x).matrix @ right_link.matrix @ \
                    self.lattice.site_group_element(random_y, random_x + 1).inverse().matrix
                self.assertEqual(transformed_right_link.matrix.all(), matrix_prod.all())

                # check lattice action is the same
                self.lattice.update_rightward_link(random_y, random_x, transformed_right_link)
                actions[i] = self.lattice.action()

                # revert back to original state
                self.lattice.update_rightward_link(random_y, random_x, right_link)

        # find the average of the new action difference, must be within 1% of the original action
        action_diff = actions - action_before
        self.assertLessEqual(action_diff.mean(), action_before * 0.01)

    def test_save_configuration(self):
        # check that after updating a link the copy remains the same but the lattice is different
        config = self.lattice.save_configuration_copy()

        # take the [0, 0] rightward link
        a1, b1, c1, d1 = self.lattice.rightward_link(0, 0).get_abcd_values()
        a2, b2, c2, d2 = config[0, 0, 0].get_abcd_values()
        self.assertEqual(a1, a2)
        self.assertEqual(b1, b2)
        self.assertEqual(c1, c2)
        self.assertEqual(d1, d2)

        # stick a random SU(2) matrix in. Yes they could be equal but very low probability
        mat = SU2Matrix()
        self.lattice.update_rightward_link(0, 0, mat)
        a3, b3, c3, d3 = self.lattice.rightward_link(0, 0).get_abcd_values()
        a4, b4, c4, d4 = config[0, 0, 0].get_abcd_values()
        self.assertNotEqual(a3, a4)
        self.assertNotEqual(b3, b4)
        self.assertNotEqual(c3, c4)
        self.assertNotEqual(d3, d4)
        self.assertEqual(a2, a4)
        self.assertEqual(b2, b4)
        self.assertEqual(c2, c4)
        self.assertEqual(d2, d4)

    def test_replace_configuration(self):
        # rejects bad shapes
        bad_arrays = [
            np.empty(1), np.empty(self.uneven_lattice.lattice_height()), np.empty(self.uneven_lattice.lattice_width()),
            np.empty((self.uneven_lattice.lattice_width(), self.uneven_lattice.lattice_height())),
            np.empty((self.uneven_lattice.lattice_height(), self.uneven_lattice.lattice_width())),
            np.empty((self.uneven_lattice.lattice_width(), self.uneven_lattice.lattice_height(), 1)),
            np.empty((self.uneven_lattice.lattice_height(), self.uneven_lattice.lattice_width(), 1)),
            np.empty((self.uneven_lattice.lattice_width(), self.uneven_lattice.lattice_height(), 2)),
            np.empty((self.uneven_lattice.lattice_height(), self.uneven_lattice.lattice_width(), 3))
        ]
        for arr in bad_arrays:
            self.assertRaises(ValueError, self.uneven_lattice.replace_configuration, arr)

        # accepts a correct shape
        current_link_variables = self.uneven_lattice.save_configuration_copy()
        link_variables = np.empty((*self.uneven_lattice.size, 2), dtype=SU2Matrix)
        for row in link_variables:
            for site in row:
                site[0] = SU2Matrix(seed=42)
                site[1] = SU2Matrix(seed=42)
        self.uneven_lattice.replace_configuration(link_variables)
        self.assertEqual(link_variables.all(), self.uneven_lattice.link_variables.all())
        self.assertFalse(np.array_equal(current_link_variables, self.uneven_lattice.link_variables))


if __name__ == '__main__':
    unittest.main()
