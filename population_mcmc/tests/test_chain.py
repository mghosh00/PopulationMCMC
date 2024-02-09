import unittest
from unittest import TestCase
from unittest import mock
import numpy as np
import scipy as sp

from population_mcmc.core import ODESystem
from population_mcmc.core import LogPrior
from population_mcmc.core import Chain


class TestChain(TestCase):
    """Test the `Chain` class
    """

    @staticmethod
    def rhs_2d(y, t, a, b, c):
        """RHS for an ODE with exponential solution
        """
        return np.array([a * y[0] + b * y[1],
                         c * y[0] - b * y[1]])

    def setUp(self):
        self._id = 4
        self.num_chains = 6
        self.y_init = np.array([1, 1])
        self.times = np.linspace(0, 1, 100)
        self.ode_system = ODESystem(self.rhs_2d, self.y_init, self.times,
                                    "system_2d")
        self.bounds = np.array([[1, 2, 3], [4, 5, 6]])
        self.prior = LogPrior(self.bounds)
        self.param_names = ["a", "b", "c"]
        self.y_obs = (np.array([np.exp(self.times), np.exp(2 * self.times)])
                      .transpose())
        self.true_params = np.array([1, 0, 2])

    def test___init___invalid(self):
        invalid_id = 7
        with self.assertRaises(ValueError) as ve:
            Chain(invalid_id, self.num_chains, self.ode_system, self.prior,
                  self.param_names)
        self.assertEqual(str(ve.exception), "Chain id must lie "
                                            "between 1 and N")

    def test___init__(self):
        chain = Chain(self.id, self.num_chains, self.ode_system, self.prior,
                      self.param_names)
        self.assertEqual(4, chain._id)
        self.assertEqual(6, chain._num_chains)
        self.assertEqual(self.ode_system, chain._ode_system)
        self.assertEqual(self.prior, chain._log_prior)
        self.assertIsNone(chain._current_params)
        self.assertEqual(0.5, chain._tempering)
        self.assertEqual(["t", "a", "b", "c", "id"],
                         chain._param_history.columns)

    def test__log_prior_pdf(self):
        with mock.patch('population_mcmc.LogPrior.__call__') as mock_call:
            chain = Chain(self._id, self.num_chains, self.ode_system,
                          self.prior, self.param_names)
            mock_call.return_value = 0.0
            prior = chain._log_prior_pdf()
            mock_call.assert_called_once_with(None)
            self.assertEqual(0.0, prior)

    @mock.patch('population_mcmc.Chain._log_prior_pdf')
    @mock.patch('population_mcmc.Chain._log_likelihood')
    def test_density(self, mock_likelihood, mock_prior):
        chain = Chain(self._id, self.num_chains, self.ode_system,
                      self.prior, self.param_names)
        mock_likelihood.return_value = 3.0
        mock_prior.return_value = 4.0
        density = chain.density(self.y_obs)
        self.assertEqual(7.0, density)

if __name__ == "__main__":
    unittest.main()
