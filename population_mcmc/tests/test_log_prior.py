import unittest
from unittest import TestCase
import numpy as np
import scipy as sp

import population_mcmc
from population_mcmc.core.log_prior import LogPrior


class TestDataGenerator(TestCase):
    """Test the `LogPrior` class
    """

    def setUp(self):
        self.bounds = np.array([[1, 2, 3], [2, 3, 4]])

    def test___init___valid(self):
        log_prior = LogPrior(self.bounds)
        self.assertEqual(self.bounds, log_prior._bounds)
        self.assertEqual()


if __name__ == "__main__":
    unittest.main()
