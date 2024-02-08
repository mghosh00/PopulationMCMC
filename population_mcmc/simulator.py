#
# Class to simulate the Population MCMC algorithm
#

import numpy as np

from population_mcmc import ODESystem
from population_mcmc import LogPrior
from population_mcmc import Chain


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
        if param_bounds.shape[0] != 2 or param_bounds.shape[1] < 1:
            raise ValueError("param_bounds must have exactly 2 rows and at "
                             "least one column")
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
        self._chains = [Chain(j + 1, num_chains, ode_system, self._log_prior)
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
            chain_i_new_params = chain_i.proposal()
            chain_i.set_params(chain_i_new_params)

            # Choose new chain
            j = np.random.randint(0, self._num_chains)
            while i == j:
                j = np.random.randint(0, self._num_chains)
            chain_j = self._chains[j]

            # Exchange step
            self._exchange(chain_i, chain_j)

    def _exchange(self, chain_i: Chain, chain_j: Chain):
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
        """
        # Get the densities
        chain_i_params_density = chain_i.density(self._y_obs)
        chain_j_params_density = chain_j.density(self._y_obs)

        # Now we use the tempered densities
        tempering_i = chain_i.get_tempering()
        tempering_j = chain_j.get_tempering()
        pi_i_x_i = chain_i_params_density * (1 - tempering_i)
        pi_i_x_j = chain_j_params_density * (1 - tempering_j)
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
