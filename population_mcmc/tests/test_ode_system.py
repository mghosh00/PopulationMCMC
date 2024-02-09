import unittest
from unittest import TestCase
import numpy as np

from population_mcmc.core import ODESystem


class TestODESystem(TestCase):
    """Test the `ODESystem` class
    """

    @staticmethod
    def rhs_basic(y, t, a):
        """RHS for an ODE with exponential solution
        """
        return np.array(a * y)

    @staticmethod
    def rhs_2d(y, t, a, b, c):
        """RHS for an ODE with exponential solution
        """
        return np.array([a * y[0] + b * y[1],
                         c * y[0] - b * y[1]])

    def setUp(self):
        self.rhs = self.rhs_basic
        self.y_init = np.array(1.0)
        self.times = np.linspace(0.0, 5.0, 100)
        self.title = "basic_ode"

    def test___init__(self):
        system = ODESystem(self.rhs, self.y_init, self.times, self.title)
        self.assertEqual(self.rhs, system._rhs)
        self.assertEqual(self.y_init, system._y_init)
        self.assertEqual(self.times.tolist(), system._times.tolist())
        self.assertEqual(self.title, system._title)
        self.assertEqual(1, system._len_theta)
        system_2 = ODESystem(self.rhs_2d, np.array([1, 2]), self.times,
                             "ode_2d")
        self.assertEqual(3, system_2._len_theta)

    def test_solve_simple(self):
        system = ODESystem(self.rhs, self.y_init, self.times, self.title)
        a = np.array(2)
        exact_y = np.exp(2 * self.times)
        sol_df = system.solve(a)
        self.assertEqual(["t", "y_1"], sol_df.columns.tolist())
        sol_numpy = sol_df.to_numpy()
        self.assertEqual(self.times.tolist(), sol_numpy[:, 0].tolist())
        for i in range(len(exact_y)):
            self.assertAlmostEqual(exact_y[i], sol_numpy[i, 1], places=2)

    def test_get_len_theta(self):
        system = ODESystem(self.rhs, self.y_init, self.times, self.title)
        self.assertEqual(1, system.get_len_theta())

    def test_get_dim_y(self):
        system = ODESystem(self.rhs, self.y_init, self.times, self.title)
        self.assertEqual(1, system.get_dim_y())
        system_2 = ODESystem(self.rhs_2d, np.array([1, 2]), self.times,
                             "ode_2d")
        self.assertEqual(2, system_2.get_dim_y())

    def test_get_title(self):
        system = ODESystem(self.rhs, self.y_init, self.times, self.title)
        self.assertEqual(1, system.get_len_theta())


if __name__ == "__main__":
    unittest.main()
