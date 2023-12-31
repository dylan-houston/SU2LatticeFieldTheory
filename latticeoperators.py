from abc import ABC, abstractmethod
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

from lattice2d import LatticeGaugeTheory2D, plaquette
from montecarlo import LatticeMarkovChain


class LatticeOperator(ABC):

    def __init__(self, lattice: LatticeGaugeTheory2D, markov_chain: LatticeMarkovChain):
        self.lattice = lattice
        self.markov_chain = markov_chain

        self.autocorrelation_sum_to = None

    @abstractmethod
    def operate(self):
        """
        Computes the value of this operator for the current configuration of the lattice.
        """
        pass

    def operate_on_configs(self, steps):
        """
        Finds the value of this operator at each of the specified Markov configuration.

        :param steps: An array of indices corresponding to the Markov steps to operate on
        :returns: An array of values of this operator, one value for each of the specified steps in the Markov Chain.
        """
        current_config = self.markov_chain.get_current_config_index()

        values = np.empty(len(steps))

        for i in range(len(steps)):
            self.markov_chain.revert_lattice_to_config(steps[i])
            values[i] = self.operate()

        self.markov_chain.revert_lattice_to_config(current_config)

        return values

    def operate_on_markov_step(self, step):
        """
        Sets the configuration of the lattice to that of the provided Markov step, computes the value of this operator,
        before returning the configuration to the configuration that was previously in place.

        :param step: The index of the configuration in the Markov Chain
        :returns: The value of the operator for this configuration
        """
        return self.operate_on_configs(np.array([step]))

    def operate_on_all_configs(self):
        """
        Finds the value of this operator at each Markov configuration.

        :returns: An array of values of this operator, one value for each step in the Markov Chain.
        """
        return self.operate_on_configs(np.arange(self.markov_chain.size(), dtype=int))

    def expectation_value(self):
        """
        Finds the expectation value of this operator over the whole Markov chain.
        """
        values = self.operate_on_all_configs()
        return values.mean()

    def expectation_value_at_steps(self, steps):
        """
        Finds the expectation value of this operator at the given steps in the Markov Chain, where at each step the
        expectation value is calculated using the chain up to and including that point.
        """
        op_values = self.operate_on_configs(steps)
        step_exp_values = np.empty(len(steps))

        for i in range(len(steps)):
            op_values_to_step = op_values[:steps[i]+1]
            step_exp_values[i] = op_values_to_step.mean()

        return step_exp_values

    def expectation_value_at_step(self, markov_step):
        """
        Finds the expectation value of this operator over the chain up to and including the current Markov step.
        """
        return self.expectation_value_at_steps(np.array([markov_step]))

    def expectation_value_at_each_step(self):
        """
        Finds the expectation value of this operator for each Markov step, where at each step the expectation value is
        calculated using the chain up to and including that point.
        """
        return self.expectation_value_at_steps(np.arange(self.markov_chain.size(), dtype=int))

    def variance_at_markov_steps(self, steps, exp_val_variance=False):
        """
        Finds the variance of either this operator or its expectation value, as calculated at the specified Markov steps.
        This variance is not naive and is corrected by the integrated autocorrelation, compensating for the correlated
        values in the Markov Chain.

        :param steps: An array of indices corresponding to the Markov steps to operate on.
        :param exp_val_variance: Whether to calculate the variance of the expectation value or the operator value.
            Default=False, will calculate the variance of the operator's value at that step (subtracting the expectation
            value as the mean used). True will calculate the variance of the expectation value at that step (subtracting
            the mean expectation value as the mean used).
        :returns: An array containing a value of the variance at each Markov step.
        """
        if exp_val_variance:
            # if the variance of the expectation value is wanted then find the mean of the exp. values calculated at
            # each point along the chain
            mean = self.expectation_value_at_each_step().mean()
            values = self.expectation_value_at_steps(steps)
        else:
            # otherwise if the variance of the operators values are being calculated then the mean value (i.e. the
            # expectation value of the chain) should be used
            mean = self.expectation_value()
            values = self.operate_on_configs(steps)

        dev_from_mean2 = (values - mean) ** 2

        variance_at_markov_steps = np.zeros(len(steps))
        for i in range(len(steps)):
            devs_up_to_i = dev_from_mean2[:i+1]
            variance_at_markov_steps[i] = devs_up_to_i.sum() / (steps[i] + 1)

        return variance_at_markov_steps * self.integrated_autocorrelation(mean)

    def variance_at_each_markov_step(self):
        """
        Finds the variance of this operator at each Markov step.

        :returns: An array containing a value of the variance at each Markov step.
        """
        return self.variance_at_markov_steps(np.arange(self.markov_chain.size(), dtype=int))

    def exp_value_variance_at_each_markov_step(self):
        """
        Finds the variance of this operator's expectation value at each Markov step.

        :returns: An array containing a value of the variance at each Markov step.
        """
        return self.variance_at_markov_steps(np.arange(self.markov_chain.size(), dtype=int), exp_val_variance=True)

    def integrated_autocorrelation(self, mean):
        """
        Calculates the integrated autocorrelation function for this operator over the Markov Chain. In theory, the
        integrated autocorrelation is a sum from -infinity to infinity; in reality, it is approximated using a finite
        sum from -M to M where M < N is some value chosen such that the integrated autocorrelation doesn't fall into
        becoming just noise.

        :param mean: The mean to subtract from values when calculating autocorrelation.
        """
        up_to = self.markov_chain.size() if self.autocorrelation_sum_to is None else self.autocorrelation_sum_to

        value_at_each_step = self.operate_on_all_configs()

        normalisation_sum = 0
        for i in range(1, self.markov_chain.size()):
            normalisation_sum += (value_at_each_step[i] - mean) ** 2
        normalisation_factor = 1 / (normalisation_sum / self.markov_chain.size())

        autocorrelation_sum = 0
        for t in range(1, up_to):
            autocorrelation_sum += self.autocorrelation(t, mean, value_at_each_step)

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

    def set_sum_to_for_integrated_autocorrelation(self, sum_to):
        """
        Sets the value of tau to sum to in the integrated autocorrelation.
        """
        if sum_to <= self.markov_chain.size():
            self.autocorrelation_sum_to = sum_to
        else:
            raise ValueError('Cannot sum to greater than the size of the Markov Chain')


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


class PlaquetteCorrelationFunctionOperator(LatticeOperator):
    """
    A LatticeOperator class that calculates the correlation function between two plaquettes at a distance.
    """
    def __init__(self, lattice: LatticeGaugeTheory2D, markov_chain: LatticeMarkovChain, lattice_point_1,
                 lattice_point_2):
        """
        Creates a PlaquetteCorrelationFunctionOperator that calculates the correlation function between the plaquettes
        at two different points on the lattice.

        :param lattice_point_1: The [x, y] lattice coordinates to begin the first plaquette at.
        :param lattice_point_2: The [x, y] lattice coordinates to begin the second plaquette at.
        """

        super().__init__(lattice, markov_chain)

        self.lattice_coords_1 = lattice_point_1
        self.lattice_coords_2 = lattice_point_2

        self.plaquette1 = PlaquetteOperator(lattice, markov_chain, lattice_point_1[0], lattice_point_1[1])
        self.plaquette2 = PlaquetteOperator(lattice, markov_chain, lattice_point_2[0], lattice_point_2[1])

    def operate(self):
        return (self.plaquette1.operate_on_all_configs() * self.plaquette2.operate_on_all_configs()).mean()


# Plotting Functions
def sparse_indices_to_plot(sparsity, data_length):
    """
    Provides a set of indices to calculate the values for when wanting a sparsely populated graph, used when a graph
    would be overpopulated given the full dataset.

    :param sparsity: How much data to remove. A sparsity of 0 is no data removed, a sparsity of 1 is all data removed.
        The value must lie in the range (0, 1). Example: A sparsity of 0.75 would keep 1/4 of the data, removing 3 in
        every 4 points.
    :param data_length: The length of the full dataset.
    :returns: a list of indices to calculate the values for to produce a sparse dataset
    """
    if 0 < sparsity < 1:
        frac = Fraction(sparsity).limit_denominator(100)
        sparsity_denominator = frac.denominator
        sparsity_numerator = frac.numerator

        indices_to_remove = []
        for i in range(data_length // sparsity_denominator):
            length = sparsity_denominator
            for j in range(sparsity_numerator):
                lower = length * i
                indices_to_remove.append(lower + length - j - 1)

        indices_to_remove = np.array(indices_to_remove, dtype=int)
        indices_to_remove = indices_to_remove[indices_to_remove < data_length]

        indices_to_keep = np.delete(np.arange(data_length), indices_to_remove)

        return indices_to_keep

    raise ValueError('Sparsity must lie in the same (0, 1).')


def plot_operator_exp_value_as_function_of_markov_step(operator: LatticeOperator, title, y_title, filepath=None,
                                                       error_bars=False, sparsity=0.0):
    """
    Plots the expectation value of this operator as computed at each Markov step, as a function of Markov step.

    :param operator: The operator to plot the expectation value of.
    :param title: The plot title.
    :param y_title: The title of the y axis, typically a description of this operator.
    :param filepath: The path at which to save the figure. Default=`None`, the figure is not saved.
    :param error_bars: Whether to include error bars, which will the standard deviation calculated at each Markov
        step. Default=`False`.
    :param sparsity: How much data to remove before plotting. A sparsity of 0 is no data removed, a sparsity of 1 is
        all data removed. The value must lie in the range (0, 1). Example: A sparsity of 0.75 would keep 1/4 of the
        data, removing 3 in every 4 points. Default=0.0.
    """
    if 0 < sparsity < 1:
        x = sparse_indices_to_plot(sparsity, operator.markov_chain.size())
    else:
        x = np.arange(operator.markov_chain.size(), dtype=int)

    values = operator.expectation_value_at_steps(x)
    std_dev = np.sqrt(operator.variance_at_markov_steps(x, exp_val_variance=True))

    fig, ax = plt.subplots(figsize=(10, 10))

    if error_bars:
        ax.errorbar(x, values, yerr=std_dev, fmt='x', capsize=3)
    else:
        ax.scatter(x, values, s=1)

    ax.set_title(title)
    ax.set_ylabel(y_title)
    ax.set_xlabel('Markov Step')

    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, dpi=300)
    plt.show()


def plot_autocorrelation_as_func_of_tau(operator, op_title, filepath=None, max_tau=None):
    """
    Plots the autocorrelation as a function of tau, ranging from 1 to either N, where N is the size of the Markov
    Chain, or the specified value.

    :param operator: The operator to plot the expectation value of.
    :param op_title: The name of the operator, as will appear in the title of the plot.
    :param filepath: The path at which to save the figure. Default=`None`, the figure is not saved.
    :param max_tau: The maximum tau value of the autocorrelation function to display.
    """
    max_tau = max_tau if max_tau is not None else operator.markov_chain.size()

    x = np.arange(1, max_tau + 1)
    autocorrelations = np.zeros_like(x)

    expectation_value = operator.expectation_value()
    value_at_each_step = operator.operate_on_all_configs()

    for tau in range(1, max_tau):
        autocorrelations[tau] = operator.autocorrelation(tau, expectation_value, value_at_each_step)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, autocorrelations, s=1)
    ax.set_title(fr'Variation in {op_title} Autocorrelation $\rho(\tau)$')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Autocorrelation')

    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, dpi=300)
    plt.show()
