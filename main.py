import matplotlib.pyplot as plt

from lattice2d import LatticeGaugeTheory2D
from montecarlo import LatticeMetropolis


def run_plot_metropolis():
    lattice = LatticeGaugeTheory2D()
    metropolis = LatticeMetropolis(lattice, step_size=0.01)
    step_actions, step_acceptance_rates = metropolis.run_metropolis(100)

    fig, (action_ax, acc_rate_ax) = plt.subplots(1, 2)
    fig.suptitle('Metropolis Algorithm run for 100 Markov steps on a 20x20 2D SU(2) lattice')

    action_ax.plot(step_actions)
    action_ax.set_xlabel('Markov Steps')
    action_ax.set_ylabel('Action')

    acc_rate_ax.plot(step_acceptance_rates)
    acc_rate_ax.set_xlabel('Markov Steps')
    acc_rate_ax.set_ylabel('Acceptance Rate')

    plt.show()


if __name__ == '__main__':
    run_plot_metropolis()
