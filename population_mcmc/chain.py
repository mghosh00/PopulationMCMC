#
# Class to represent a single Markov chain for Bayesian inference
#

import numpy as np

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

    def _calculate_log_prior(self) -> float:
        """Calculates the log prior for the current set of parameters. We
        assume a multivariate normal prior centred around the midpoint of the
        uniform bounds. The method used accesses the pdf of the prior.

        Returns
        -------
        float
            The log prior for the given parameters
        """
        return self._log_prior(self._current_params)
