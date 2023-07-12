import unittest

import numpy as np

from su2matrices import SU2Matrix, gram_schmidt, random_complex_vector, is_matrix_special_unitary


class SU2UtilityTest(unittest.TestCase):

    def test_special_unitary_check_allows_valid_SU2(self):
        self.assertTrue(is_matrix_special_unitary(np.array([[0, 1j], [1j, 0]])))
        self.assertTrue(is_matrix_special_unitary(np.array([[0, 1], [-1, 0]])))
        self.assertTrue(is_matrix_special_unitary(np.array([[1j, 0], [0, -1j]])))
        self.assertTrue(is_matrix_special_unitary(np.array([[1 / np.sqrt(5), 2j / np.sqrt(5)],
                                                             [2j / np.sqrt(5), 1 / np.sqrt(5)]])))

    def test_special_unitary_check_blocks_det1_not_unitary(self):
        self.assertFalse(is_matrix_special_unitary(np.array([[1, 2], [-1, 1]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[3, 4], [-2, -1]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[-5, 2], [-3, 1]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[1, 2, 3], [0, 1, 0], [0, 0, 1]])))

    def test_special_unitary_check_blocks_unitary_not_det1(self):
        # Pauli matrices not det 1 but are unitary
        self.assertFalse(is_matrix_special_unitary(np.array([[0, 1], [1, 0]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[0, -1j], [1j, 0]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[1, 0], [0, -1]])))

    def test_special_unitary_check_blocks_not_unitary_not_det1(self):
        self.assertFalse(is_matrix_special_unitary(np.array([[1, 1], [1, 1]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[2, 2], [2, 2]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[4, 0], [0, -1]])))
        self.assertFalse(is_matrix_special_unitary(np.array([[4, 5, 5], [1, 1j, 3], [9, 8, -1j]])))

    def test_special_unitary_check_rejects_vector(self):
        self.assertRaises(ValueError, is_matrix_special_unitary, np.array([0, 1, 2]))

    def test_special_unitary_check_rejects_3d_tensor(self):
        self.assertRaises(ValueError, is_matrix_special_unitary, np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_special_unitary_check_rejects_non_square_matrix(self):
        self.assertRaises(ValueError, is_matrix_special_unitary, np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertRaises(ValueError, is_matrix_special_unitary, np.array([[1, 2], [3, 4], [5, 6]]))

    def test_rand_complex_is_correct_size(self):
        for i in range(0, 10):
            rand_vec = random_complex_vector(i)
            self.assertEqual(len(rand_vec), i)

    def test_gram_schmidt_rejects_one_or_fewer_vectors(self):
        self.assertRaises(ValueError, gram_schmidt, [])
        self.assertRaises(ValueError, gram_schmidt, [np.zeros(2)])

    def test_gram_schmidt_produces_orthogonal_vectors(self):
        for i in range(2, 10):
            vectors = []
            for _ in range(0, i):
                vectors.append(random_complex_vector(i))
            gs_vectors = gram_schmidt(vectors)

            # make sure each of the produced vectors is orthogonal to the others
            for j in range(0, i):
                for k in range(j+1, i):
                    self.assertAlmostEqual(np.vdot(gs_vectors[j], gs_vectors[k]), 0)

    def test_gram_schmidt_produces_normal_vectors(self):
        for i in range(2, 10):
            vectors = []
            for _ in range(0, i):
                vectors.append(random_complex_vector(i))
            gs_vectors = gram_schmidt(vectors)

            # make sure each of the produced vectors is normal
            for j in range(0, i):
                self.assertAlmostEqual(np.linalg.norm(gs_vectors[j]), 1)


class SU2MatricesTest(unittest.TestCase):

    def test_special_unitary_allows_SU2(self):
        self.assertTrue(SU2Matrix(a=0, b=1j, c=1j, d=0).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=1, c=-1, d=0).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1j, b=0, c=0, d=-1j).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1/np.sqrt(5), b=2j/np.sqrt(5), c=2j/np.sqrt(5), d=1/np.sqrt(5)).is_special_unitary())

    def test_input_zero_a_b_gives_SU2(self):
        self.assertTrue(SU2Matrix(a=0, b=0, c=1, d=1, seed=42).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=0, c=1, d=1, seed=36).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=0, c=1, d=1, seed=23).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=0, c=1, d=1, seed=11).is_special_unitary())

    def test_input_zero_c_d_gives_SU2(self):
        self.assertTrue(SU2Matrix(a=1, b=1, c=0, d=0, seed=42).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=1, c=0, d=0, seed=36).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=1, c=0, d=0, seed=23).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=1, c=0, d=0, seed=11).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=1, c=0, d=0, seed=19).is_special_unitary())

    def test_input_zero_a_b_c_d_gives_SU2(self):
        self.assertTrue(SU2Matrix(seed=42).is_special_unitary())
        self.assertTrue(SU2Matrix(seed=36).is_special_unitary())
        self.assertTrue(SU2Matrix(seed=23).is_special_unitary())
        self.assertTrue(SU2Matrix(seed=11).is_special_unitary())
        self.assertTrue(SU2Matrix(seed=19).is_special_unitary())

    def test_non_SU2_input_gives_SU2(self):
        # the Pauli matrices are not SU(2), testing each one with several random seeds
        self.assertTrue(SU2Matrix(a=0, b=1, c=1, d=0, seed=42).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=1, c=1, d=0, seed=36).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=1, c=1, d=0, seed=23).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=1, c=1, d=0, seed=11).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=1, c=1, d=0, seed=19).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=-1j, c=1j, d=0, seed=42).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=-1j, c=1j, d=0, seed=36).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=-1j, c=1j, d=0, seed=23).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=-1j, c=1j, d=0, seed=11).is_special_unitary())
        self.assertTrue(SU2Matrix(a=0, b=-1j, c=1j, d=0, seed=19).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=0, c=0, d=-1, seed=42).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=0, c=0, d=-1, seed=36).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=0, c=0, d=-1, seed=23).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=0, c=0, d=-1, seed=11).is_special_unitary())
        self.assertTrue(SU2Matrix(a=1, b=0, c=0, d=-1, seed=19).is_special_unitary())
        # and a more random matrix
        self.assertTrue(SU2Matrix(a=5, b=9, c=10, d=-21, seed=42).is_special_unitary())
        self.assertTrue(SU2Matrix(a=5, b=9, c=10, d=-21, seed=36).is_special_unitary())
        self.assertTrue(SU2Matrix(a=5, b=9, c=10, d=-21, seed=23).is_special_unitary())
        self.assertTrue(SU2Matrix(a=5, b=9, c=10, d=-21, seed=11).is_special_unitary())
        self.assertTrue(SU2Matrix(a=5, b=9, c=10, d=-21, seed=19).is_special_unitary())

    def test_returns_abcd_correctly(self):
        mat = SU2Matrix(a=0, b=1j, c=1j, d=0)
        self.assertEqual(mat.get_abcd_values(), (0, 1j, 1j, 0))

    def test_eq(self):
        mat1 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        mat2 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        self.assertEqual(mat1, mat2)

    def test_inverse(self):
        mat1 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        mat2 = SU2Matrix(a=0, b=1, c=-1, d=0)

        inv1 = mat1.inverse()
        inv2 = mat2.inverse()

        self.assertEqual(np.linalg.inv(mat1.matrix).all(), inv1.matrix.all())
        self.assertEqual(np.linalg.inv(mat2.matrix).all(), inv2.matrix.all())

    def test_right_multiply_valid_su2_result(self):
        matrix1 = SU2Matrix(a=0, b=1j, c=1j, d=0)

        matrix2 = SU2Matrix(a=0, b=1, c=-1, d=0)
        matrix2_list = [[0, 1], [-1, 0]]
        matrix2_arr = np.array(matrix2_list)

        # test all of these succeed and are of the right type
        self.assertIsInstance(matrix1.right_multiply_by(matrix2_list), SU2Matrix)
        self.assertIsInstance(matrix1.right_multiply_by(matrix2_arr), SU2Matrix)
        self.assertIsInstance(matrix1.right_multiply_by(matrix2), SU2Matrix)
        self.assertIsInstance(matrix1.right_multiply_by(matrix2_list, su2_result=False), np.ndarray)
        self.assertIsInstance(matrix1.right_multiply_by(matrix2_arr, su2_result=False), np.ndarray)
        self.assertIsInstance(matrix1.right_multiply_by(matrix2, su2_result=False), np.ndarray)

        # check the same result is obtained regardless of input type
        self.assertEqual(matrix1.right_multiply_by(matrix2), matrix1.right_multiply_by(matrix2_list))
        self.assertEqual(matrix1.right_multiply_by(matrix2_arr), matrix1.right_multiply_by(matrix2_list))

        # check the same result is obtained whether returned as SU2Matrix or as array
        self.assertEqual(matrix1.right_multiply_by(matrix2_list, su2_result=False).all(),
                         matrix1.right_multiply_by(matrix2_list).matrix.all())

        # check the correct result is obtained
        self.assertEqual(matrix1.right_multiply_by(matrix2).matrix.all(), (matrix1.matrix @ matrix2.matrix).all())
        self.assertEqual(matrix2.right_multiply_by(matrix1).matrix.all(), (matrix2.matrix @ matrix1.matrix).all())

    def test_right_multiply_invalid_su2_with_su2_result(self):
        matrix1 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        matrix2 = [[0, -1j], [1j, 0]]
        self.assertRaises(ValueError, matrix1.right_multiply_by, matrix2)

    def test_right_multiply_invalid_su2_with_array_result(self):
        matrix1 = SU2Matrix(a=0, b=1j, c=1j, d=0)
        matrix2 = [[0, -1j], [1j, 0]]
        self.assertEqual(matrix1.right_multiply_by(matrix2, su2_result=False).all(),
                         (matrix1.matrix @ np.array(matrix2)).all())


if __name__ == '__main__':
    unittest.main()
