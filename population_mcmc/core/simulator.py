#
# Class to simulate the Population MCMC algorithm
#

import numpy as np
import pandas as pd

from .ode_system import ODESystem
from .log_prior import LogPrior
from .chain import Chain
from .plotter import Plotter


class Simulator:
    """This class simulates the Population MCMC algorithm for a given
    :class:`ODESystem` and table of observed data.
    """

    def __init__(self, ode_system: ODESystem, num_chains: int, y_obs: np.array,
                 param_bounds: np.array, param_names: list[str] = None,
                 max_its: int = 1000, init_phase_its: int = 500):
        """Constructor Method

        ode_system : ODESystem
            The system of ODEs to be solved
        num_chains : int
            The number of internal chains used in the population MCMC
        y_obs : np.array
            The observed data for :math:`y` (a :math:`p \\times n` array where
            :math:`p` is the number of time steps and :math:`n` is the length
            of :math:`y`)
        param_bounds : np.array
            A :math:`2 \\times m` array, where :math:`m` is the number of
            parameters, containing the lower and upper bounds of the
            parameters. These must also contain bounds for the standard
            deviations of the :math:`y_{i}`
        param_names : list[str] [optional, default=None]
            Names for the parameters, to be used when logging to the user
        max_its : int [optional, default=1000]
            The maximal number of iterations the simulator will run for
        init_phase_its : int [optional, default=500]
            The number of iterations for the initial phase of the inference
        """
        self._ode_system = ode_system
        self._num_chains = num_chains
        self._y_obs = y_obs
        # First ensure that the parameter bounds have the correct number of
        # rows
        if param_bounds.shape[0] != 2:
            raise ValueError("param_bounds must have exactly 2 rows")

        # Next we need to check that the number of columns of param_bounds is
        # equal to the number of ODE params plus the number of hyperparameter
        # standard deviations in y that we need to keep track of
        exp_num = ode_system.get_len_theta() + ode_system.get_dim_y()
        if param_bounds.shape[1] != exp_num:
            raise ValueError(f"Expected {exp_num} columns for param_bounds,"
                             f" got {param_bounds.shape[1]}")
        self._param_bounds = param_bounds
        self._num_params = param_bounds.shape[1]
        if not param_names or len(param_names) != self._num_params:
            self._param_names = [f"param_{i + 1}"
                                 for i in range(self._num_params)]
        else:
            self._param_names = param_names
        if init_phase_its > max_its:
            raise ValueError("init_phase_its cannot be more than max_its")
        self._max_its = max_its
        self._init_phase_its = init_phase_its
        self._log_prior = LogPrior(self._param_bounds)
        self._chains = [Chain(j + 1, num_chains, ode_system, self._log_prior,
                              self._param_names)
                        for j in range(num_chains)]

    def _set_uniform_initial_sample(self):
        """Takes a uniform sample of parameters using their bounds for each
        chain, and sets up each chain with these values.
        """
        for j in range(self._num_chains):
            chain = self._chains[j]
            lower_bounds = self._param_bounds[0, :]
            upper_bounds = self._param_bounds[1, :]
            chain.set_params(np.random.uniform(lower_bounds, upper_bounds))

    def run(self):
        """Main method for running the Population MCMC. This method also logs
        data to the console.
        """
        # First, set up all chains with the initial samples
        self._set_uniform_initial_sample()

        # Next, we run the algorithm for (up to) max_its iterations
        for t in range(self._max_its):
            # Mutation step
            i = np.random.randint(0, self._num_chains)
            chain_i = self._chains[i]
            mutated = chain_i.mutate(self._y_obs)
            if mutated:
                print(f"{t} --- {chain_i} mutated successfully")

            # Choose new chain
            j = np.random.randint(0, self._num_chains)
            while i == j:
                j = np.random.randint(0, self._num_chains)
            chain_j = self._chains[j]

            # Exchange step
            exchanged = self._exchange(chain_i, chain_j)
            if exchanged:
                print(f"{t} --- {chain_i} swapped with {chain_j}")
            else:
                pass
                # print(f"{t} --- {chain_i} failed to swap with {chain_j}")

            # Record the parameter history
            for chain in self._chains:
                chain.update_param_history(t)

    def _exchange(self, chain_i: Chain, chain_j: Chain) -> bool:
        """Exchanges the parameter sets of chain_i and chain_j if with
        a certain acceptance probability :math:`min(1, A)` where :math:`A` is
        equal to :math:`\\frac{{\\pi}_{i}(x_{j}){\\pi}_{j}(x_{i})}
        {{\\pi}_{i}(x_{i}){\\pi}_{j}(x_{j})}` and :math:`{\\pi}_{k}` is the
        tempered density of the chain (note this formula changes when we
        take logs).

        Parameters
        ----------
        chain_i : Chain
            First (recently changed) chain
        chain_j : Chain
            Second chain

        Returns
        -------
        bool
            True if exchange occurred, False otherwise
        """
        # Get the densities
        chain_i_params_density = chain_i.density(self._y_obs)
        chain_j_params_density = chain_j.density(self._y_obs)

        # Now we use the tempered densities
        tempering_i = chain_i.get_tempering()
        tempering_j = chain_j.get_tempering()
        pi_i_x_i = chain_i_params_density * (1 - tempering_i)
        pi_i_x_j = chain_j_params_density * (1 - tempering_i)
        pi_j_x_i = chain_i_params_density * (1 - tempering_j)
        pi_j_x_j = chain_j_params_density * (1 - tempering_j)
        A = pi_i_x_j + pi_j_x_i - pi_i_x_i - pi_j_x_j
        acceptance = np.exp(min(0, A))
        u = np.random.uniform()
        if acceptance > u:
            x_i = chain_i.get_params()
            x_j = chain_j.get_params()
            chain_i.set_params(x_j)
            chain_j.set_params(x_i)
            return True
        return False

    def plot_traces(self, chain_id: int = -1):
        """Uses the Plotter class to create trace plots

        Parameters
        ----------
        chain_id : int [optional, default=-1]
            If this parameter is not -1, we plot only one chain. Otherwise,
            plot all chains.
        """
        title = self._ode_system.get_title()
        if 1 <= chain_id <= self._num_chains:
            df = self._chains[chain_id - 1].get_param_history()
        else:
            df = pd.concat([chain.get_param_history()
                            for chain in self._chains], ignore_index=True)
        Plotter.plot_traces(df, title)
