import unittest
from unittest import TestCase
import numpy as np
import scipy as sp

import population_mcmc
from population_mcmc.core.log_prior import LogPrior


class TestLogPrior(TestCase):
    """Test the `LogPrior` class
    """

    def setUp(self):
        self.bounds = np.array([[1, 2, 3], [2, 4, 6]])

    def test___init___valid(self):
        log_prior = LogPrior(self.bounds)
        self.assertEqual(self.bounds.tolist(), log_prior._bounds.tolist())
        self.assertEqual([1.5, 3, 4.5], log_prior._means.tolist())
        self.assertEqual([0.125, 0.25, 0.375],
                         log_prior._std_devs.tolist())

    def test___init___invalid(self):
        invalid_size_bounds = np.array([[2, 3], [3, 4], [4, 5]])
        with self.assertRaises(ValueError) as ve1:
            lp = LogPrior(invalid_size_bounds)
        self.assertEqual(str(ve1.exception), "bounds must have exactly 2 "
                                             "rows and at least one column")
        invalid_order_bounds = np.array([[2, 3, 4], [3, 4, 3]])
        with self.assertRaises(ValueError) as ve2:
            lp = LogPrior(invalid_order_bounds)
        self.assertEqual(str(ve2.exception), "Lower bounds must be in the "
                                             "first row and upper bounds in the "
                                             "second row")

    def test___call___invalid(self):
        invalid_theta = np.array([3, 4])
        with self.assertRaises(ValueError) as ve:
            lp = LogPrior(self.bounds)
            lp(invalid_theta)
        self.assertEqual(str(ve.exception), f"The length of theta must match "
                                            f"the number of columns specified "
                                            f"in self._bounds (2 != 3)")

    def test___call__(self):
        theta = np.array([1.5, 2.5, 3.5])
        expected = sum([sp.stats.norm.logpdf(theta[i], [1.5, 3, 4.5][i],
                                             [0.125, 0.25, 0.375][i])
                        for i in range(3)])
        log_prior = LogPrior(self.bounds)
        actual = log_prior(theta)
        self.assertEqual(expected, actual)

    def test_get_std_devs(self):
        log_prior = LogPrior(self.bounds)
        self.assertEqual([0.125, 0.25, 0.375],
                         log_prior.get_std_devs().tolist())


if __name__ == "__main__":
    unittest.main()
