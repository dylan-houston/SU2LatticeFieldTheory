import matplotlib.pyplot as plt
import numpy as np

from lattice2d import LatticeGaugeTheory2D
from montecarlo import LatticeMarkovChain


def run_plot_metropolis():
    n_steps = 300
    lattice = LatticeGaugeTheory2D(8, 4, random_seed=42)
    metropolis = LatticeMarkovChain(lattice, step_size=0.4, seed=42)
    step_actions, step_acceptance_rates = metropolis.run_metropolis(n_steps)

    fig, ax = plt.subplots(1)
    fig.suptitle(f'Metropolis Algorithm run for {n_steps} Markov steps 2D SU(2) lattice')

    ax.scatter(np.arange(0, n_steps), step_actions, s=1)
    ax.set_xlabel('Markov Steps')
    ax.set_ylabel('Action')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_plot_metropolis()
