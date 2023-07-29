from abc import ABC, abstractmethod
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

from lattice2d import LatticeGaugeTheory2D, plaquette
from montecarlo import LatticeMarkovChain


def _remove_fraction_of_data(data_arrays: np.ndarray, sparsity):
    """
    Removes a fixed amount of data from the supplied arrays in a uniform way. Used for plotting purposes when a graph
    would be overpopulated given the full dataset.

    :param data_arrays: A 2D array containing all the data sequences to remove data from. Each data sequence should be
        the same size. Typically, these would be the x, y and error values for each data point.
    :param sparsity: How much data to remove. A sparsity of 0 is no data removed, a sparsity of 1 is all data removed.
        The value must lie in the range (0, 1). Example: A sparsity of 0.75 would keep 1/4 of the data, removing 3 in
        every 4 points.
    :returns: a list of the sparse arrays
    """
    if 0 < sparsity < 1 and len(data_arrays.shape) == 2:
        arr_length = len(data_arrays[0])

        frac = Fraction(sparsity).limit_denominator(100)
        sparsity_denominator = frac.denominator
        sparsity_numerator = frac.numerator

        indices_to_remove = []
        for i in range(arr_length // sparsity_denominator):
            length = sparsity_denominator
            for j in range(sparsity_numerator):
                lower = length * i
                indices_to_remove.append(lower + length - j - 1)

        indices_to_remove = np.array(indices_to_remove, dtype=int)
        indices_to_remove = indices_to_remove[indices_to_remove < arr_length]

        arrays = []
        for array in data_arrays:
            arrays.append(np.delete(array, indices_to_remove))
        return arrays

    raise ValueError('Sparsity must lie in the same (0, 1) and the data_arrays must be a 2D array containing all data '
                     'to sparsify.')


class LatticeOperator(ABC):

    def __init__(self, lattice: LatticeGaugeTheory2D, markov_chain: LatticeMarkovChain):
        self.lattice = lattice
        self.markov_chain = markov_chain

    @abstractmethod
    def operate(self):
        pass

    def operate_on_markov_step(self, step):
        self.markov_chain.revert_lattice_to_config(step)
        val = self.operate()
        self.markov_chain.restore_final_lattice_config()
        return val

    def operate_on_all_configs(self):
        size = self.markov_chain.size()
        values = np.empty(size)

        for i in range(size):
            self.markov_chain.revert_lattice_to_config(i)
            values[i] = self.operate()

        self.markov_chain.restore_final_lattice_config()

        return values

    def plot_exp_value_as_function_of_markov_step(self, title, y_title, filepath=None, error_bars=False, sparsity=0):
        values = self.operate_on_all_configs()
        x = np.arange(0, self.markov_chain.size())
        std_dev = np.sqrt(self.variance())
        if 0 < sparsity < 1:
            x, values, std_dev = _remove_fraction_of_data(np.array([x, values, std_dev]), sparsity)

        fig, ax = plt.subplots(figsize=(10, 10))

        if error_bars:
            ax.errorbar(x, values, yerr=std_dev, fmt='x', capsize=5)
        else:
            ax.scatter(x, values, s=1)

        ax.set_title(title)
        ax.set_ylabel(y_title)
        ax.set_xlabel('Markov Step')

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, dpi=300)
        plt.show()

    def expectation_value(self):
        values = self.operate_on_all_configs()
        return values.mean()

    def variance(self):
        exp_value = self.expectation_value()
        values = self.operate_on_all_configs()
        dev_from_mean2 = (values - exp_value) ** 2

        variance_at_markov_step = np.zeros(self.markov_chain.size())
        variance_at_markov_step[0] = np.nan
        for i in range(1, len(variance_at_markov_step)):
            devs_up_to_i = dev_from_mean2[:i]
            variance_at_markov_step[i] = devs_up_to_i.sum() / i

        return variance_at_markov_step

    def corrected_variance(self):
        pass


class PlaquetteOperator(LatticeOperator):
    """
    A LatticeOperator class that returns the plaquette originating at a given point on the lattice.
    """
    def __init__(self, lattice: LatticeGaugeTheory2D, markov_chain: LatticeMarkovChain, x_index, y_index):
        """
        Creates a PlaquetteOperator that starts at the specified index.

        :param x_index: The x_index on the lattice for the plaquette to begin at.
        :param y_index: The y_index on the lattice for the plaquette to begin at.
        """
        super().__init__(lattice, markov_chain)
        self.x_index = x_index
        self.y_index = y_index

    def operate(self):
        return plaquette(self.lattice, self.x_index, self.y_index)


class AveragePlaquetteOperator(LatticeOperator):
    """
    A LatticeOperator class that returns the average plaquette on the lattice.
    """
    def operate(self):
        # action is sum of plaquettes so avg plaquette is action / n sites
        return self.lattice.action() / (self.lattice.lattice_width() * self.lattice.lattice_height())


class LatticeActionOperator(LatticeOperator):
    """
    A LatticeOperator class that returns the action of the lattice, i.e. the sum of all plaquettes over the lattice.
    """
    def operate(self):
        return self.lattice.action()
