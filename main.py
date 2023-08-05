import matplotlib.pyplot as plt
import numpy as np

from lattice2d import LatticeGaugeTheory2D
from montecarlo import LatticeMarkovChain
from latticeoperators import AveragePlaquetteOperator, LatticeActionOperator


def run_plot_metropolis():
    n_steps = 5000
    M, N = 4, 4
    lattice = LatticeGaugeTheory2D(M, N, random_seed=63)
    markov_chain = LatticeMarkovChain(lattice, step_size=0.25, seed=63)
    step_actions, acceptance_rate = markov_chain.run_metropolis(n_steps)

    action_operator = LatticeActionOperator(lattice, markov_chain)
    action_operator.plot_exp_value_as_function_of_markov_step(f'Action Expectation Value on an {M}x{N} Lattice',
                                                              r'$\langle S \rangle$', error_bars=False, sparsity=0.75)
    action_operator.plot_autocorrelation_as_func_of_tau('Action')


if __name__ == '__main__':
    run_plot_metropolis()
