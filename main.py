import matplotlib.pyplot as plt
import numpy as np

from lattice2d import LatticeGaugeTheory2D
from montecarlo import LatticeMarkovChain
from latticeoperators import AveragePlaquetteOperator, LatticeActionOperator, plot_autocorrelation_as_func_of_tau, \
    plot_operator_exp_value_as_function_of_markov_step


def run_plot_metropolis():
    n_steps = 100
    M, N = 8, 8
    lattice = LatticeGaugeTheory2D(M, N, random_seed=63)
    markov_chain = LatticeMarkovChain(lattice, step_size=0.25, seed=63)
    step_actions, acceptance_rate = markov_chain.run_metropolis(n_steps)

    action_operator = LatticeActionOperator(lattice, markov_chain)
    plot_operator_exp_value_as_function_of_markov_step(action_operator, f'Action Expectation Value on an {M}x{N} Lattice',
                                                       r'$\langle S \rangle$', error_bars=True, sparsity=0.75)
    plot_operator_exp_value_as_function_of_markov_step(action_operator,
                                                       f'Action Expectation Value on an {M}x{N} Lattice',
                                                       r'$\langle S \rangle$')
    plot_autocorrelation_as_func_of_tau(action_operator, 'Action')


if __name__ == '__main__':
    run_plot_metropolis()
