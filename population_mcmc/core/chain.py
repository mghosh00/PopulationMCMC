#
# Class to represent a single Markov chain for Bayesian inference
#

import numpy as np
import scipy as sp
import pandas as pd

from population_mcmc import ODESystem
from population_mcmc import LogPrior


class Chain:
    """This class represents a single chain for performing Population Markov
    Chain Monte Carlo.
    """

    def __init__(self, chain_id: int, num_chains: int, ode_system: ODESystem,
                 log_prior: LogPrior, param_names: list[str]):
        """Constructor Method

        Parameters
        ----------
        chain_id : int
            The id for this chain. Will lie in :math:`{1,...,N}` where N is the
            number of chains
        num_chains : int
            The total number of chains that will be used. Necessary for
            calculating the tempering parameter
        ode_system : ODESystem
            The system of ODEs to be solved
        log_prior : LogPrior
            The log prior to be used for the chain
        param_names : list[str]
            The list of parameter names
        """
        self._id = chain_id
        self._num_chains = num_chains
        # Checks
        if not 1 <= chain_id <= num_chains:
            raise ValueError("Chain id must lie between 1 and N")
        self._ode_system = ode_system
        self._log_prior = log_prior
        self._current_params = None
        self._tempering = self._calculate_tempering()
        self._param_history = pd.DataFrame(columns=(["t"] + param_names
                                                    + ["id"]))

    def _calculate_tempering(self) -> float:
        """Calculates the tempering parameter for the chain. This lies between
        0 and 1. Note that we use a uniformly spaced tempering scheme.

        Returns
        -------
        float
            The tempering parameter
        """
        return (self._id - 1) / self._num_chains

    def _log_prior_pdf(self) -> float:
        """Calculates the pdf value of the log prior for the current set of
        parameters. We assume a multivariate normal prior centred around the
        midpoint of the uniform bounds. The method used accesses the pdf of the
        prior.

        Returns
        -------
        float
            The log prior for the given parameters
        """
        return self._log_prior(self._current_params)

    def _log_likelihood(self, y_obs: np.array) -> float:
        """Calculates the log likelihood of observed values, `y_obs`, given the
        `current_params`. To do this, the ODE system is solved and a float
        value is returned.

        Parameters
        ----------
        y_obs : np.array
            The observed data for :math:`y` (a :math:`p \\times n` array where
            :math:`p` is the number of time steps and :math:`n` is the length
            of :math:`y`)

        Returns
        -------
        float
            The sum of log likelihoods of `y_obs` given `current_params`
        """
        # This is to remove the standard deviations from the current_params
        num_time_steps, num_vars = y_obs.shape
        num_params = len(self._current_params) - num_vars
        params_only = self._current_params[:num_params]
        y_std_devs = self._current_params[num_params:]

        # Here, we solve the ODE system with the given parameters
        expected_sol_df = self._ode_system.solve(params_only)

        # This is a numpy array of all expected y values at all time steps
        expected_sol = expected_sol_df.iloc[:, 1:].to_numpy()

        # Finally, calculate the log likelihoods comparing the observed to
        # expected solutions, and using the current standard deviations
        log_likelihoods = [0] * num_vars
        for j in range(num_vars):
            for i in range(num_time_steps):
                log_likelihoods[j] += sp.stats.norm.logpdf(y_obs[i][j],
                                                           expected_sol[i][j],
                                                           y_std_devs[j])
        return sum(log_likelihoods)

    def density(self, y_obs: np.array) -> np.array:
        """Calculates the density using the log prior and log likelihood. This
        will be equal to :math:`log(prior) + log(likelihood)`. Note that the
        tempering will be done later as it is not necessary here.

        Parameters
        ----------
        y_obs : np.array
            The observed data for :math:`y` (a :math:`p \\times n` array where
            :math:`p` is the number of time steps and :math:`n` is the length
            of :math:`y`)

        Returns
        -------
        np.array
            The vector of densities for this specific chain.
        """
        log_prior = self._log_prior_pdf()
        log_likelihood = self._log_likelihood(y_obs)
        return log_prior + log_likelihood

    def mutate(self, y_obs: np.array) -> bool:
        """Make a proposal for new parameters using the current ones as means
        and the standard deviations of the prior. Then choose whether to accept
        these parameters based on a Metropolis acceptance algorithm.

        Parameters
        ----------
        y_obs : np.array
            The observed data for :math:`y` (a :math:`p \\times n` array where
            :math:`p` is the number of time steps and :math:`n` is the length
            of :math:`y`)

        Returns
        -------
        bool
            Whether the new parameters were accepted or not
        """
        old_params = np.copy(self._current_params)
        current_density = self.density(y_obs)
        proposal = np.random.normal(self._current_params,
                                    self._log_prior.get_std_devs())

        # In order to calculate the density, we need to set the new parameters
        # as the current parameters of the chain
        self.set_params(proposal)
        new_density = self.density(y_obs)

        # Use the Metropolis algorithm to accept/reject the new parameters
        r = np.exp(new_density - current_density)
        u = np.random.uniform()
        if r > u:
            return True
        else:
            # We revert to the old parameters if we have rejected
            self.set_params(old_params)
            return False

    def update_param_history(self, t: int):
        """Writes a new row of the `self._param_history` df with the current
        parameters.

        Parameters
        ----------
        t : int
            Current step of the iteration
        """
        df = self._param_history
        df.loc[len(df)] = [t] + list(self._current_params) + [self._id]

    def get_params(self) -> np.array:
        """Retrieves the `current_params`

        Returns
        -------
        np.array
            The `current_params`
        """
        return self._current_params

    def set_params(self, params: np.array):
        """Whenever a new proposal for parameters succeeds, we set new
        parameters for the chain. Note that these `params` include the standard
        deviations.

        Parameters
        ----------
        params : np.array
            A new array of parameters to be used for this chain
        """
        self._current_params = params

    def get_tempering(self) -> float:
        """Retrieves the tempering parameter

        Returns
        -------
        float
            The tempering parameter
        """
        return self._tempering

    def get_param_history(self) -> pd.DataFrame:
        """Retrieves the parameter history dataframe

        Returns
        -------
        pd.DataFrame
            The parameter history dataframe
        """
        return self._param_history

    def __str__(self):
        return f"Chain {self._id} with parameters {self._current_params}"
