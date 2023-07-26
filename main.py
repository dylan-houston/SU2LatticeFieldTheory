import matplotlib.pyplot as plt
import numpy as np

from lattice2d import LatticeGaugeTheory2D
from montecarlo import LatticeMarkovChain
from latticeoperators import AveragePlaquetteOperator


def run_plot_metropolis():
    n_steps = 300
    M, N = 8, 4
    lattice = LatticeGaugeTheory2D(8, 4, random_seed=42)
    markov_chain = LatticeMarkovChain(lattice, step_size=0.4, seed=42)
    step_actions, acceptance_rate = markov_chain.run_metropolis(n_steps)

    avg_plaquette_operator = AveragePlaquetteOperator(lattice, markov_chain)
    avg_plaquette_operator.plot_as_function_of_markov_step(f'Average Plaquette on an {M}x{N} Lattice',
                                                           'Average Plaquette', error_bars=True)


if __name__ == '__main__':
    run_plot_metropolis()
