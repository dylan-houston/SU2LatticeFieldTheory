import numpy as np

from su2matrices import SU2Matrix


class LatticeGaugeTheory2D:
    """
    A class that represents a 2D lattice gauge theory.
    """

    def __init__(self, M=20, N=20, beta=4, uniform_initialisation_matrix=None, random_seed=None):
        """
        Sets up a 2D periodic MxN lattice, with SU(2) matrices sitting on the links of the lattice.

        :param M: the height of the lattice
        :param N: the width of the lattice
        :param beta: the parameter of the theory, used in calculating the action
        :param uniform_initialisation_matrix: an `SU2Matrix`. If supplied the lattice, will be constructed with a copy
            of this matrix sitting on each link, if not supplied then there will be a random initialisation.
        :param random_seed: The random seed to use if link matrices are randomly generated. Default=`None`
        """
        self.site_group_elements = np.empty((M, N), dtype=SU2Matrix)
        self.size = (M, N)
        self.beta = beta

        # put an arbitrary group element on each site
        for i in range(M):
            for j in range(N):
                self.site_group_elements[i, j] = SU2Matrix()

        # link variables do not sit on the sites, they sit on the links between sites
        # this would imply 4 link variables per site, but traversing backwards along a link (i.e. from j -> i rather
        # than i -> j) uses the same matrix, just inverted. As such, only two link variables are needed per site, taken
        # to be that going upwards and that going rightwards
        self.link_variables = np.empty((M, N, 2), dtype=SU2Matrix)

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

    def index_with_PBCs(self, y_index, x_index):
        """
        Calculates the correct indices using periodic boundary conditions. If the supplied indices lie outside the
        ranges for i and j, [0, M) and [0, N), they will be translated into this range, e.g. if i = M, then this
        corresponds to i = 0, if i = -1 this corresponds to i = M - 1, etc.

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        :return: (y_index, x_index)
        """
        if y_index >= 0:
            y_index = y_index % self.lattice_height()
        else:
            y_index = (np.abs(y_index // self.lattice_height()) * self.lattice_height() + y_index) % self.lattice_height()

        if x_index >= 0:
            x_index = x_index % self.lattice_width()
        else:
            x_index = (np.abs(x_index // self.lattice_width()) * self.lattice_width() + x_index) % self.lattice_width()

        return y_index, x_index

    def links_for_site(self, y_index, x_index):
        """
        Return the upwards and rightwards links for each site.

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        :return: [rightward link, upwards link]
        """
        y_index, x_index = self.index_with_PBCs(y_index, x_index)

        return self.link_variables[y_index, x_index]

    def site_group_element(self, y_index, x_index):
        """
        Return the group element associated with each site.

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        :return: the SU2Matrix associated with this site.
        """
        y_index, x_index = self.index_with_PBCs(y_index, x_index)

        return self.site_group_elements[y_index, x_index]

    def rightward_link(self, y_index, x_index):
        """
        Returns the rightward link matrix of a lattice site. Rightward to mean in the +ve x-direction.

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        """
        return self.links_for_site(y_index, x_index)[0]

    def leftward_link(self, y_index, x_index):
        """
        Returns the leftward link matrix of a lattice site. Leftward to mean in the -ve x-direction.

        The leftward link is the inverse of the rightward link for the site (y, x) = (i, j - 1).

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        """
        return self.links_for_site(y_index, x_index - 1)[0].inverse()

    def upward_link(self, y_index, x_index):
        """
        Returns the upward link matrix of a lattice site. Upward to mean in the +ve y-direction.

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        """
        return self.links_for_site(y_index, x_index)[1]

    def downward_link(self, y_index, x_index):
        """
        Returns the downward link matrix of a lattice site. Downward to mean in the -ve y-direction.

        The downward link is the inverse of the rightward link for the site (y, x) = (i - 1, j).

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        """
        return self.links_for_site(y_index - 1, x_index)[1].inverse()

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

    def update_rightward_link(self, y_index, x_index, U: SU2Matrix):
        """
        Updates the rightward link matrix of a lattice site. Rightward to mean in the +ve x-direction.

        This will also update the leftward link from site (y, x) = (i, j - 1), as that is the inverse of this link.

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        :param U: The new link SU2Matrix.
        """
        y_index, x_index = self.index_with_PBCs(y_index, x_index)
        self.link_variables[y_index, x_index, 0] = U

    def update_upward_link(self, y_index, x_index, U: SU2Matrix):
        """
        Updates the upward link matrix of a lattice site. Upward to mean in the +ve y-direction.

        This will also update the downward link from site (y, x) = (i - 1, j), as that is the inverse of this link.

        :param y_index: Site index in the y (up-down) direction
        :param x_index: Site index in the x (left-right) direction
        :param U: The new link SU2Matrix.
        """
        y_index, x_index = self.index_with_PBCs(y_index, x_index)
        self.link_variables[y_index, x_index, 1] = U


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
