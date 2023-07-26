from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from lattice2d import LatticeGaugeTheory2D, plaquette
from montecarlo import LatticeMarkovChain


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

    def plot_as_function_of_markov_step(self, title, y_title, filepath=None, error_bars=False, sparsity=1):
        values = self.operate_on_all_configs()
        x = np.arange(0, self.markov_chain.size(), dtype=int)

        # TODO: Implement sparsity, keep only len(values)*sparsity values

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x, values, s=1)

        if error_bars:
            ax.errorbar(x, values, yerr=np.sqrt(self.variance()))

        ax.set_title(title)
        ax.set_ylabel(y_title)

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
