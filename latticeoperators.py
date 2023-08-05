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
        """
        Computes the value of this operator for the current configuration of the lattice.
        """
        pass

    def operate_on_markov_step(self, step):
        """
        Sets the configuration of the lattice to that of the provided Markov step, computes the value of this operator,
        before returning the configuration to the configuration that was previously in place.

        :param step: The index of the configuration in the Markov Chain
        :returns: The value of the operator for this configuration
        """
        current_step = self.markov_chain.get_current_config_index()
        self.markov_chain.revert_lattice_to_config(step)
        val = self.operate()
        self.markov_chain.revert_lattice_to_config(current_step)
        return val

    def operate_on_all_configs(self):
        """
        Finds the value of this operator at each Markov configuration.

        :returns: An array of values of this operator, one value for each step in the Markov Chain.
        """
        current_config = self.markov_chain.get_current_config_index()

        size = self.markov_chain.size()
        values = np.empty(size)

        for i in range(size):
            self.markov_chain.revert_lattice_to_config(i)
            values[i] = self.operate()

        self.markov_chain.revert_lattice_to_config(current_config)

        return values

    def expectation_value(self):
        """
        Finds the expectation value of this operator over the whole Markov chain.
        """
        values = self.operate_on_all_configs()
        return values.mean()

    def expectation_value_at_step(self, markov_step):
        """
        Finds the expectation value of this operator over the chain up to and including the current Markov step.
        """
        values = self.operate_on_all_configs()
        values = values[:markov_step]
        return values.mean()

    def expectation_value_at_each_step(self):
        """
        Finds the expectation value of this operator for each Markov step, where at each step the expectation value is
        calculated using the chain up to and including that point.
        """
        op_values = self.operate_on_all_configs()
        step_exp_values = np.empty(self.markov_chain.size())

        for i in range(len(step_exp_values)):
            op_values_to_step = op_values[:i+1]
            step_exp_values[i] = op_values_to_step.mean()

        return step_exp_values

    def variance_at_each_markov_step(self):
        """
        Finds the variance of this operator at each Markov step.

        :returns: An array containing a value of the variance at each Markov step.
        """
        exp_value = self.expectation_value()
        values = self.expectation_value_at_each_step()
        dev_from_mean2 = (values - exp_value) ** 2

        variance_at_markov_step = np.zeros(self.markov_chain.size())
        variance_at_markov_step[0] = dev_from_mean2[0]
        for i in range(1, len(variance_at_markov_step)):
            devs_up_to_i = dev_from_mean2[:i+1]
            variance_at_markov_step[i] = devs_up_to_i.sum() / i

        return variance_at_markov_step * self.integrated_autocorrelation()

    def integrated_autocorrelation(self, up_to=-1):
        """
        Calculates the integrated autocorrelation function for this operator over the Markov Chain. In theory, the
        integrated autocorrelation is a sum from -infinity to infinity; in reality, it is approximated using a finite
        sum from -M to M where M < N is some value chosen such that the integrated autocorrelation doesn't fall into
        becoming just noise.
        """
        if up_to == -1:
            up_to = self.markov_chain.size()

        expectation_value = self.expectation_value()
        value_at_each_step = self.operate_on_all_configs()

        normalisation_sum = 0
        for i in range(1, self.markov_chain.size()):
            normalisation_sum += (value_at_each_step[i] - expectation_value) ** 2
        normalisation_factor = 1 / (normalisation_sum / self.markov_chain.size())

        autocorrelation_sum = 0
        for t in range(1, up_to):
            autocorrelation_sum += self.autocorrelation(t, expectation_value, value_at_each_step)

        autocorrelation_sum *= normalisation_factor
        integrated_autocorrelation = 1 + 2 * autocorrelation_sum

        return integrated_autocorrelation

    def autocorrelation(self, tau_value, expectation_value, value_at_each_step):
        r"""
        Calculates the autocorrelation for a given tau value, where tau is the value summed over to obtain integrated
        autocorrelation.

        The autocorrelation is defined as:
        $\tau_{\langle O \rangle} = \frac{1}{N-\tau}\sum^{N-\tau}_{n=1}\left(O_n - \langle O
            \rangle \right) \left(O_{n+\tau} - \langle O \rangle \right) $

        :param tau_value: The tau value, tau being the number of steps away in the markov chain the variance is measured
            at.
        :param expectation_value: The expectation value of the complete Markov Chain.
        :param value_at_each_step: The value of the operator as calculated at each step of the Markov Chain.
        """
        value = 0
        for n in range(1, self.markov_chain.size() - tau_value):
            value += (value_at_each_step[n] - expectation_value) * \
                     (value_at_each_step[n + tau_value] - expectation_value)

        value *= 1 / (self.markov_chain.size() - tau_value)

        return value

    def plot_exp_value_as_function_of_markov_step(self, title, y_title, filepath=None, error_bars=False, sparsity=0.0):
        """
        Plots the expectation value of this operator as computed at each Markov step, as a function of Markov step.

        :param title: The plot title.
        :param y_title: The title of the y axis, typically a description of this operator.
        :param filepath: The path at which to save the figure. Default=`None`, the figure is not saved.
        :param error_bars: Whether to include error bars, which will the standard deviation calculated at each Markov
            step. Default=`False`.
        :param sparsity: How much data to remove before plotting. A sparsity of 0 is no data removed, a sparsity of 1 is
            all data removed. The value must lie in the range (0, 1). Example: A sparsity of 0.75 would keep 1/4 of the
            data, removing 3 in every 4 points. Default=0.0.
        """
        values = self.expectation_value_at_each_step()
        x = np.arange(0, self.markov_chain.size())
        std_dev = np.sqrt(self.variance_at_each_markov_step())
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

    def plot_autocorrelation_as_func_of_tau(self, op_title, filepath=None):
        """
        Plots the autocorrelation as a function of tau, ranging from 1 to N, where N is the size of the Markov Chain.

        :param op_title: The name of the operator, as will appear in the title of the plot.
        :param filepath: The path at which to save the figure. Default=`None`, the figure is not saved.
        """
        x = np.arange(1, self.markov_chain.size()+1)
        autocorrelations = np.zeros_like(x)

        expectation_value = self.expectation_value()
        value_at_each_step = self.operate_on_all_configs()

        for tau in range(1, self.markov_chain.size()):
            autocorrelations[tau] = self.autocorrelation(tau, expectation_value, value_at_each_step)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x, autocorrelations, s=1)
        ax.set_title(fr'Variation in {op_title} Autocorrelation $\rho(\tau)$')
        ax.set_xlabel(r'$\tau\$')
        ax.set_ylabel('Autocorrelation')

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, dpi=300)
        plt.show()


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
