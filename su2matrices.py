import numpy as np


class SU2Matrix:
    """
    A class used to represent SU(2) group elements as 2x2 matrices.
    """

    def __init__(self, a=0, b=0, c=0, d=0, seed=None):
        """
        Creates an SU2Matrix object. If the values of a, b, c and d are supplied then these will be inserted into the
        matrix. If this results in an SU(2) matrix then this is kept. If this does not result in an SU(2) matrix then a
        and b will be kept, c will be set to -b* and d to a*. The Gram-Schmidt process will be applied (with the rows
        taken as a set of linearly independent vectors), ensuring an SU(2) matrix. If a, b, c and d aren't supplied,
        then a random SU(2) matrix will be generated.

        :param a: Optional. The [0,0] matrix element.
        :param b: Optional. The [0,1] matrix element.
        :param c: Optional. The [1,0] matrix element.
        :param d: Optional. The [1,1] matrix element.
        :param seed: The random seed to use if `a` & `b` are not supplied. Default = `None`.
        """
        self.matrix = np.array([[a, b], [c, d]], dtype=complex)

        if not is_matrix_special_unitary(self.matrix):
            if a == 0 and b == 0 and c == 0 and d == 0:
                np.random.seed(seed)
                a = np.random.random() + np.random.random() * 1j
                b = np.random.random() + np.random.random() * 1j
                c = -np.conj(b)
                d = np.conj(a)
            elif c != 0 and d != 0:
                a = np.conj(d)
                b = -np.conj(c)
            else:
                c = -np.conj(b)
                d = np.conj(a)

            vectors = [np.array([a, b]), np.array([c, d])]

            orthonormal_vectors = gram_schmidt(vectors)

            self.matrix = np.array([[orthonormal_vectors[0][0], orthonormal_vectors[0][1]],
                                    [orthonormal_vectors[1][0], orthonormal_vectors[1][1]]], dtype=complex)

    def hermitian_conjugate(self):
        """
        Returns the Hermitian conjugate of this SU(2) matrix.

        :return: The Hermitian conjugate
        """
        return self.matrix.conj().T

    def is_special_unitary(self):
        """
        Returns whether this is a valid SU(2) matrix.

        :return: True if a valid SU(2) matrix, False otherwise.
        """
        return is_matrix_special_unitary(self.matrix)

    def re_special_unitrise(self):
        pass

    def get_abcd_values(self):
        """
        Returns the elements of the matrix, in the order a, b, c, d for a matrix [[a, b], [c, d]].
        """
        return get_abcd_values_from_2x2_matrix(self.matrix)

    def inverse(self):
        """
        Inverts this matrix. Since this matrix is an SU(2) group element, the inverse will also be an SU(2) group
        element.

        :return: An SU2Matrix object for the inverse of this matrix.
        """
        [a, b], [c, d] = np.linalg.inv(self.matrix)
        return SU2Matrix(a=a, b=b, c=c, d=d)

    def trace(self):
        """
        Returns the trace of this matrix.
        """
        return np.trace(self.matrix)

    def right_multiply_by(self, matrix, su2_result=True):
        """
        Multiplies this matrix by the supplied matrix to the right. Call this matrix U0 and the supplied matrix U, then
        the operation carried out is U0 @ U.

        :param matrix: Either an SU2Matrix or a regular array-based matrix.
        :param su2_result: Whether to return the result as an SU2Matrix. Default=True. If this option is selected but
            the result is not an valid SU(2) element, an exception will be raised.
        :return: Either a numpy array as the result (if su2_result=False) or an SU2Matrix.
        """
        if su2_result:
            if type(matrix) == SU2Matrix:
                # the product of 2 SU(2) matrices will be an SU(2) matrix, so no checks needed
                prod = self.matrix @ matrix.matrix
                a, b, c, d = get_abcd_values_from_2x2_matrix(prod)
                return SU2Matrix(a=a, b=b, c=c, d=d)
            else:
                # if matrix is an SU(2) matrix, but just not an SU2Matrix object the result will be an SU(2) matrix
                # but a check has to be carried out, in case it is not
                prod = self.matrix @ matrix
                if is_matrix_special_unitary(prod):
                    a, b, c, d = get_abcd_values_from_2x2_matrix(prod)
                    return SU2Matrix(a=a, b=b, c=c, d=d)
                else:
                    raise ValueError('Cannot produce an SU2Matrix object from this product.')
        else:
            if type(matrix) == SU2Matrix:
                matrix = matrix.matrix
            return self.matrix @ matrix

    def __eq__(self, other):
        if type(other) == SU2Matrix:
            return self.matrix.all() == other.matrix.all()
        return False


def is_matrix_special_unitary(U: np.ndarray):
    """
    Returns whether the matrix is valid SU(n) matrix.
    Checks whether $det(U) = 1$ and whether $UU\dagger=\mathbb{1}$

    :param U: The matrix to check
    :return: True if a valid SU(n) matrix, False otherwise
    """
    # check if matrix and if square
    if len(U.shape) > 1 and U.shape[0] == U.shape[1]:
        det = np.linalg.det(U)
        u_u_dagger = np.asmatrix(U @ U.conj().T, dtype=complex)
        return np.isclose(det, 1) and np.allclose(u_u_dagger, np.identity(U.shape[0], dtype=complex))

    raise ValueError('Expected a square matrix.')


def gram_schmidt(vectors):
    """
    Performs the Gram-Schmidt process on the supplied basis, producing an orthonormal basis.

    :param vectors: A list of initial basis vectors.
    :return: A list of orthonormal basis vectors.
    """
    if len(vectors) > 1:
        orthonormal_vectors = []

        # create the first normal vector
        e1 = 1 / np.linalg.norm(vectors[0]) * np.asarray(vectors[0], dtype=complex)
        orthonormal_vectors.append(e1)

        # create a set of orthonormal vectors
        for i in range(1, len(vectors)):
            vi = np.asarray(vectors[i], dtype=complex)

            wi = vi
            for ei in orthonormal_vectors:
                # note that <vi, ei> is written as np.vdot(ei, vi), since np.vdot takes the complex conjugate
                # of the first argument not the second
                wi -= np.vdot(ei, vi) * ei

            ei = 1 / np.linalg.norm(wi) * wi
            orthonormal_vectors.append(ei)

        return orthonormal_vectors

    raise ValueError('More than one vector is required to produce an orthonormal set.')


def get_abcd_values_from_2x2_matrix(matrix):
    """
    Returns the elements of the 2x2 matrix, in the order a, b, c, d for a matrix [[a, b], [c, d]].
    """
    return matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]


def random_complex_vector(size):
    rand = np.random.rand(2, size)
    return rand[0] + rand[1]*1j


if __name__ == "__main__":
    SU2Matrix(a=0, b=0, c=1, d=1)
