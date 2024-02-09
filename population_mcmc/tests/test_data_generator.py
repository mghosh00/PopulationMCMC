import unittest
from unittest import TestCase
import numpy as np

import population_mcmc
from population_mcmc.core import ODESystem
from population_mcmc.core import DataGenerator


class TestDataGenerator(TestCase):
    """Test the `DataGenerator` class
    """

    @staticmethod
    def rhs_basic(y, t, a):
        """RHS for an ODE with exponential solution
        """
        return np.array(a * y)

    def setUp(self):
        self.rhs = self.rhs_basic
        self.y_init = np.array(1.0)
        self.times = np.linspace(0.0, 5.0, 100)
        self.title = "basic_ode"
        self.basic_system = ODESystem(self.rhs, self.y_init, self.times,
                                      self.title)
        self.true_theta_basic = np.array(2.0)
        self.sigma_basic = np.array(0.01)

    def test___init__(self):
        generator = DataGenerator(self.basic_system, self.true_theta_basic,
                                  self.sigma_basic)
        self.assertEqual(self.basic_system, generator._ode_system)
        self.assertEqual(self.true_theta_basic, generator._true_theta)
        self.assertEqual(self.sigma_basic, generator._sigma)

    def test_generate_observed_data_basic(self):
        generator = DataGenerator(self.basic_system, self.true_theta_basic,
                                  self.sigma_basic)
        y_true = np.exp(2 * self.times)
        y_obs = generator.generate_observed_data()
        for i in range(len(y_obs)):

            # Checks that the generated data is within 5 standard deviations
            # of the observed data
            self.assertAlmostEqual(float(y_obs[i]), float(y_true[i]),
                                   delta=5 * float(self.sigma_basic))


if __name__ == "__main__":
    unittest.main()
