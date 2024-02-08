#
# Class to represent a single Markov chain for Bayesian inference
#

import numpy as np
import scipy as sp

from population_mcmc import ODESystem
from population_mcmc import LogPrior


class Chain:
    """This class represents a single chain for performing Population Markov
    Chain Monte Carlo.
    """

    def __init__(self, chain_id: int, num_chains: int, ode_system: ODESystem,
                 log_prior: LogPrior):
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

    def _calculate_tempering(self) -> float:
        """Calculates the tempering parameter for the chain. This lies between
        0 and 1. Note that we use a uniformly spaced tempering scheme.

        Returns
        -------
        float
            The tempering parameter
        """
        return (self._id - 1) / self._num_chains

    def set_params(self, theta: np.array):
        """Whenever a new proposal for parameters succeeds, we set new
        parameters for the chain.

        Parameters
        ----------
        theta : np.array
            A new array of parameters to be used for this chain
        """
        self._current_params = theta

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

    def _log_likelihood(self, y_obs: np.array) -> np.array:
        """Calculates the log likelihood of observed values, `y_obs`, given the
        `current_params`. To do this, the ODE system is solved and a float
        value is returned.

        Parameters
        ----------
        y_obs : np.array
            The observed data for y (a :math:`p \times n` array where p is the
            number of time steps and n is the length of y)

        Returns
        -------
        np.array
            The vector of likelihoods of `y_obs` given `current_params`
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
        return np.array(log_likelihoods)

    def tempered_density(self, y_obs: np.array) -> np.array:
        """Calculates the tempered density using the log prior, log likelihood
        and tempering parameter. This will be equal to
        :math:`(log_prior + log_likelihood) * (1 - T)`,
        where T is the tempering parameter.
        Parameters
        ----------
        y_obs : np.array
            The observed data for y (a :math:`p x n` array where p is the
            number of time steps and n is the length of `y`)

        Returns
        -------
        np.array
            The vector of tempered densities for this specific chain.
        """
        log_prior = self._log_prior_pdf()
        log_likelihood = self._log_likelihood(y_obs)
        return (log_prior + log_likelihood) * (1 - self._tempering)

    def proposal(self) -> np.array:
        """Make a proposal for new parameters using the current ones as means
        and the standard deviations of the prior

        Returns
        -------
        np.array
            A proposal for new parameters
        """
        return np.random.normal(self._current_params,
                                self._log_prior.get_std_devs())
