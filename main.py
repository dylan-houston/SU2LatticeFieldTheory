import matplotlib.pyplot as plt
import numpy as np

from lattice2d import LatticeGaugeTheory2D
from montecarlo import LatticeMetropolis


def run_plot_metropolis():
    n_steps = 300
    lattice = LatticeGaugeTheory2D(20, 20, random_seed=42)
    metropolis = LatticeMetropolis(lattice, step_size=0.4, seed=42)
    step_actions, step_acceptance_rates = metropolis.run_metropolis(n_steps)

    fig, (action_ax, acc_rate_ax) = plt.subplots(1, 2)
    fig.suptitle(f'Metropolis Algorithm run for {n_steps} Markov steps 2D SU(2) lattice')

    action_ax.scatter(np.arange(0, n_steps), step_actions, s=1)
    action_ax.set_xlabel('Markov Steps')
    action_ax.set_ylabel('Action')

    acc_rate_ax.scatter(np.arange(0, n_steps), step_acceptance_rates, s=1)
    acc_rate_ax.set_xlabel('Markov Steps')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_plot_metropolis()
