#
# Generates random observed data for the solution of an ode_system given some
# standard deviations away from the true solution
#

import numpy as np

from population_mcmc import ODESystem


class DataGenerator:
    """This class generates random observed data for the solution of an
    :class:`ODESystem` given ground truth parameter values (`true_theta`)
    and some standard deviations away from these values in `sigma`.
    """

    def __init__(self, ode_system: ODESystem, true_theta: np.array,
                 sigma: np.array):
        """Constructor method

        Parameters
        ----------
        ode_system : ODESystem
            The system of ODEs being used to generate the data
        true_theta : np.array
            The ground truth parameter values used in solving the ODE (size =
            number of parameters)
        sigma : np.array
            Standard deviations away from the true data (size = length of
            :math:`y`)
        """
        self._ode_system = ode_system
        self._true_theta = true_theta
        self._sigma = sigma

    def generate_observed_data(self) -> np.array:
        """Generates data in the form of a :math:`p \\times n` array where
        :math:`p` is the number of time steps and :math:`n` is the length of
        :math:`y`.

        Returns
        -------
        np.array
            Observed data generated as a noisy version of the true ODE
            solution
        """
        true_data = self._ode_system.solve(self._true_theta)
        y_true = true_data.iloc[:, 1:].to_numpy()
        return np.random.normal(y_true, self._sigma, size=y_true.shape)
