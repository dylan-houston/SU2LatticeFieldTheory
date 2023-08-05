from time import time

import numpy as np
from scipy.linalg import expm

from lattice2d import LatticeGaugeTheory2D, plaquette
from su2matrices import SU2Matrix, get_abcd_values_from_2x2_matrix


class LatticeMarkovChain:

    def __init__(self, lattice: LatticeGaugeTheory2D, seed=None, step_size=0.1):
        self.lattice = lattice
        self.step_size = step_size

        self.markov_chain_configs = [lattice.save_configuration_copy()]
        self.current_config = len(self.markov_chain_configs) - 1

        np.random.seed(seed)

    def save_configurations_to_file(self, filename):
        """
        Loads the configurations of the Markov Chain to a .npy file. Note that this file contains no information about
        the step size used or the LatticeGaugeTheory2D object, it will only load the link variables for each Markov step.
        """
        np.save(filename, self.markov_chain_configs)

    def load_configurations_from_file(self, filename):
        """
        Loads the configurations of the Markov Chain from a file. Note that this file contains no information about the
        step size used or the LatticeGaugeTheory2D object, it will only load the link variables for each Markov step.
        """
        self.markov_chain_configs = np.load(filename, allow_pickle=True).tolist()

    def run_metropolis(self, n_runs, supress_output=False):
        """
        Runs the metropolis algorithm to update the lattice and produce a Markov Chain. This function returns an array
        of action expectation values up to each point in the Markov Chain and the average acceptance rate.

        :param n_runs: The number of runs of the algorithm, i.e. the number of steps in the Markov Chain.
        :param supress_output: If False then no output is produced.
        :returns: action_expectation_value_at_each_markov_step, acceptance_rate
        """
        if not supress_output:
            print('----- Starting Metropolis -----')
        start_time = time()

        cumulative_action = 0
        average_action_through_chain = np.empty(n_runs)

        acceptance_rates = np.empty(n_runs)

        for run in range(n_runs):
            if not supress_output:
                print(f'Starting run {run + 1}')
            run_start_time = time()

            # create a random order to go through the whole lattice for this step
            site_order = self._generate_random_lattice_step_through_sequence()

            # go through the lattice in the specified order, noting whether the change to the upwards and rightwards
            # link was accepted or not to produce an acceptance rate.
            n_acceptances_in_step = 0
            for y_index, x_index in site_order:
                up_accept, right_accept = self._metropolis_site_step(x_index, y_index)
                n_acceptances_in_step += up_accept + right_accept

            acceptance_rates[run] = n_acceptances_in_step / (self.lattice.lattice_width() *
                                                             self.lattice.lattice_height() * 2)

            # find the average action for this and all other prior configurations in the Markov Chain
            cumulative_action += self.lattice.action()
            average_action_through_chain[run] = cumulative_action / (run + 1)

            # save this configuration
            self.markov_chain_configs.append(self.lattice.save_configuration_copy())

            if not supress_output:
                print(f'Run {run + 1} completed in {time() - run_start_time} seconds.\n'
                      f'    Acceptance Rate = {acceptance_rates[run]}'
                      f'    Action = {average_action_through_chain[run]}')

        if not supress_output:
            print(f'----- Metropolis Completed in {time() - start_time} seconds -----')
            print(f'Average Acceptance Rate {acceptance_rates.mean()}')

        self.current_config = n_runs
        return average_action_through_chain, acceptance_rates.mean()

    def _generate_random_lattice_step_through_sequence(self):
        """
        Generates a random order to step through the lattice in. Each site will be visited in each Markov step, in a
        random order. To run each Markov step stepping through the lattice in a random order, this array can be stepped
        through sequentially.

        :returns: A (M*N, 2) sized array. Each entry it [y_index, x_index].
        """
        pairs = np.empty((self.lattice.lattice_height() * self.lattice.lattice_width(), 2), dtype=int)
        for i in range(self.lattice.lattice_height()):
            row_order = np.arange(0, self.lattice.lattice_width())
            row_pairs = row_order.reshape((self.lattice.lattice_width(), 1))
            row_number = np.ones((self.lattice.lattice_width(), 1)) * i
            row_pairs = np.concatenate((row_number, row_pairs), axis=1)
            pairs[i * self.lattice.lattice_width():
                  i * self.lattice.lattice_width() + self.lattice.lattice_width()] = row_pairs
        np.random.shuffle(pairs)
        return pairs

    def _metropolis_site_step(self, x_index, y_index):
        """
        Performs the metropolis update at the site level. The upwards and downwards links will both have candidate
        updates carried out and the metropolis test will be carried out on each one and each change will either be
        accepted or rejected.
        """
        # the matrices to attempt to update
        right_link = self.lattice.rightward_link(y_index, x_index)
        up_link = self.lattice.upward_link(y_index, x_index)

        # calculates the plaquettes in the co-boundary of the links
        up_plaquette_cobound = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index - 1, y_index)
        right_plaquette_cobound = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index, y_index - 1)

        # generates candidate updates
        candidate_right_link = self._metropolis_matrix_shift(right_link)
        candidate_up_link = self._metropolis_matrix_shift(up_link)
        self.lattice.update_rightward_link(y_index, x_index, candidate_right_link)
        self.lattice.update_upward_link(y_index, x_index, candidate_up_link)

        # calculates the plaquettes in the co-boundary of the links
        up_plaquette_cobound_new = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index - 1, y_index)
        right_plaquette_cobound_new = plaquette(self.lattice, x_index, y_index) + plaquette(self.lattice, x_index, y_index - 1)

        # finds the change in action for both updates
        delta_S_up = up_plaquette_cobound_new - up_plaquette_cobound
        delta_S_right = right_plaquette_cobound_new - right_plaquette_cobound

        # perform metropolis test, keeping change if accepted and reverting if rejected
        up_accept = self._metropolis_test(delta_S_up)
        right_accept = self._metropolis_test(delta_S_right)
        if not up_accept:
            self.lattice.update_upward_link(y_index, x_index, up_link)
        if not right_accept:
            self.lattice.update_rightward_link(y_index, x_index, right_link)

        return [up_accept, right_accept]

    def _metropolis_matrix_shift(self, matrix):
        """
        Generates an update to the SU2Matrix provided, by generating a random SU2Matrix with two random, complex entries
        that have a real and imaginary part smaller than the step size.

        :param matrix: the SU2Matrix to update.
        """
        w, x, y, z = np.random.uniform(-1, 1, 4)

        H = np.array([[x, y + z*1j], [y - z*1j, w]])

        V = expm(-1j * self.step_size * H)
        a, b, c, d = get_abcd_values_from_2x2_matrix(V)
        V = SU2Matrix(a=a, b=b, c=c, d=d)

        # this matrix and its hermitian conjugate should be equally probable to ensure detail balance, so randomly
        # choose one
        rand = np.random.rand()
        if rand > 0.5:
            V = V.hermitian_conjugate()

        # create the shifted matrix
        return matrix.right_multiply_by(V)

    def _metropolis_test(self, delta_S):
        """
        Performs the Metropolis test. Generates some random number r el [0, 1). If r < exp(-Delta S) then the change
        will be accepted. If r > exp(-Delta S) then the change will be rejected. If Delta S < 0 the change is
        immediately accepted.

        :param delta_S: the change in the action for the proposed update.
        :returns: True if accepted.
        """
        r = np.random.uniform()

        if delta_S <= 0:
            return True

        return r < np.exp(-delta_S)

    def trim_first_n_steps(self, n):
        """
        Removes the first n entries from the Markov Chain. Note that the chain will be re-indexed after this, so that
        i' = i - n.

        :param n: The number of steps to remove.
        """
        self.markov_chain_configs = self.markov_chain_configs[n:]

    def revert_lattice_to_config(self, step):
        """
        Replaces the current configuration of the lattice with the one from the specified Markov step.

        :param step: Int, the Markov step.
        :raises ValueError: If the specified step doesn't exist.
        """
        if step < len(self.markov_chain_configs):
            self.lattice.replace_configuration(self.markov_chain_configs[step])
            self.current_config = step
        else:
            raise ValueError('The specified Markov step doesn\'t exist')

    def restore_final_lattice_config(self):
        """
        Restores the final configuration of the lattice in the Markov Chain.
        """
        self.lattice.replace_configuration(self.markov_chain_configs[-1])
        self.current_config = len(self.markov_chain_configs) - 1

    def size(self):
        """
        Returns the size of the Markov Chain.
        """
        return len(self.markov_chain_configs)

    def get_current_config_index(self):
        """
        Returns the index (i.e. Markov step index) of the current configuration of the lattice.
        """
        return self.current_config
