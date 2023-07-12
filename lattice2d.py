import numpy as np

from su2matrices import SU2Matrix


class LatticeGaugeTheory2D:
    """
    A class that represents a 2D lattice gauge theory.
    """

    def __init__(self, M=20, N=20, beta=4, uniform_initialisation_matrix=None, random_seed=None):
        """
        Sets up a 2D MxN lattice, with SU(2) matrices sitting on the links of the lattice.

        :param M: the height of the lattice
        :param N: the width of the lattice
        :param beta: the parameter of the theory, used in calculating the action
        :param uniform_initialisation_matrix: an `SU2Matrix`. If supplied the lattice, will be constructed with a copy
            of this matrix sitting on each link, if not supplied then there will be a random initialisation.
        :param random_seed: The random seed to use if link matrices are randomly generated. Default=`None`
        """
        # link variables do not sit on the sites, they sit on the links between sites
        # this would imply 4 link variables per site, but traversing backwards along a link (i.e. from j -> i rather
        # than i -> j) uses the same matrix, just inverted. As such, only two link variables are needed per site, taken
        # to be that going upwards and that going rightwards
        self.link_variables = np.empty((M, N, 2), dtype=SU2Matrix)
        self.size = (M, N)
        self.beta = beta

        if uniform_initialisation_matrix is not None:
            if type(uniform_initialisation_matrix) == SU2Matrix:
                a, b, c, d = uniform_initialisation_matrix.get_abcd_values()
                for row in self.link_variables:
                    for site in row:
                        site[0] = SU2Matrix(a=a, b=b, c=c, d=d)
                        site[1] = SU2Matrix(a=a, b=b, c=c, d=d)
            else:
                raise TypeError('The initialisation matrix should be of type SU2Matrix')
        else:
            for row in self.link_variables:
                for site in row:
                    site[0] = SU2Matrix(seed=random_seed)
                    site[1] = SU2Matrix(seed=random_seed)

    def lattice_height(self):
        return self.size[0]

    def lattice_width(self):
        return self.size[1]

    def beta_value(self):
        return self.beta

    def links_for_site(self, i, j):
        """
        Return the upwards and rightwards links for each site. Periodic boundary conditions are used on this lattice, so
        if the supplied indices lie outside the ranges for i and j, [0, M) and [0, N), they will be translated into
        this range, e.g. if i = M, then this corresponds to i = 0, if i = -1 this corresponds to i = M - 1, etc.

        :param i: Site index in the y (up-down) direction
        :param j: Site index in the x (left-right) direction
        :return: [rightward link, upwards link]
        """
        if i >= 0:
            i = i % self.lattice_height()
        else:
            i = (-i - 1) % self.lattice_height()

        if j >= 0:
            j = j % self.lattice_width()
        else:
            j = (-j - 1) % self.lattice_width()

        return self.link_variables[i, j]

    def rightward_link(self, i, j):
        """
        Returns the rightward link matrix of a lattice site. Rightward to mean in the +ve x-direction.

        :param i: Site index in the y (up-down) direction
        :param j: Site index in the x (left-right) direction
        """
        return self.links_for_site(i, j)[0]

    def leftward_link(self, i, j):
        """
        Returns the leftward link matrix of a lattice site. Leftward to mean in the -ve x-direction.

        The leftward link is the inverse of the rightward link for the site (y, x) = (i, j - 1).

        :param i: Site index in the y (up-down) direction
        :param j: Site index in the x (left-right) direction
        """
        return self.links_for_site(i, j - 1)[0].inverse()

    def upward_link(self, i, j):
        """
        Returns the upward link matrix of a lattice site. Upward to mean in the +ve y-direction.

        :param i: Site index in the y (up-down) direction
        :param j: Site index in the x (left-right) direction:
        """
        return self.links_for_site(i, j)[1]

    def downward_link(self, i, j):
        """
        Returns the downward link matrix of a lattice site. Downward to mean in the -ve y-direction.

        The downward link is the inverse of the rightward link for the site (y, x) = (i - 1, j).

        :param i: Site index in the y (up-down) direction
        :param j: Site index in the x (left-right) direction:
        """
        return self.links_for_site(i - 1, j)[1].inverse()

    def action(self):
        """
        Calculates the action of the lattice by summing over all plaquettes.

        :return: The action on the lattice.
        """
        action = 0

        for i in range(0, self.lattice_height()):
            for j in range(0, self.lattice_width()):
                action += plaquette(self, j, i)

        return action


def plaquette(lattice: LatticeGaugeTheory2D, site_x, site_y):
    """
    Calculates the Plaquette, starting at the specified site.

    The calculation is:
    beta * (1 - (1/2)*Re(Tr(U_ij * U_jk * U_kl * U_li))

    :param lattice: the lattice: LatticeGaugeTheory2D
    :param site_x: the x index of the site to start at
    :param site_y: the y index of the site to start to
    :return: The action of the Plaquette
    """
    U_ij = lattice.rightward_link(site_y, site_x)
    U_jk = lattice.upward_link(site_y, site_x + 1)
    U_kl = lattice.leftward_link(site_y + 1, site_x + 1)
    U_li = lattice.downward_link(site_y + 1, site_x)

    return lattice.beta_value() * (1 - 1/2 * np.real(
        U_ij.right_multiply_by(U_jk).right_multiply_by(U_kl).right_multiply_by(U_li).trace()
    ))
