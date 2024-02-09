#
# Class to numerically solve a system of ODEs given initial data and specific
# parameter values
#

import typing
from inspect import signature
import numpy as np
import pandas as pd
import scipy.integrate as si


class ODESystem:
    """This class takes in the RHS of a system of ODEs in the form:
    :math:`y'(t) = f(y, t; \\theta), y(0) = y_{0},` where :math:`y, f, y_{0}`
    are n-dimensional vectors, :math:`t` is a scalar and :math:`\\theta` is an
    m-dimensional parameter vector.

    """

    def __init__(self, rhs: typing.Callable, y_init: np.array,
                 times: np.array, title: str):
        """Constructor Method

        Parameters
        ----------
        rhs : typing.Callable
            A function :math:`f : R^{n} x R -> R^{n}` which takes in a vector
            :math:`y` and time :math:`t` and returns the RHS of the ODE system,
            given parameters :math:`\\theta`
        y_init : np.array
            The initial values of y to be passed to the system
        times : np.array
            The times to be used for the numerical solution
        title : str
            The title of the ODE System
        """
        self._rhs = rhs
        self._times = times
        self._y_init = y_init
        self._title = title
        rhs_sig = signature(rhs)
        # This is useful for keeping track of the number of parameters which
        # are not hyperparameters
        self._len_theta = len(rhs_sig.parameters) - 2

    def solve(self, theta: np.array) -> pd.DataFrame:
        """Given a parameter array theta, this will solve the ODE as described
        in the class definition.

        Parameters
        ----------
        theta : np.array
            The parameter list for the ODE

        Returns
        -------
        pd.DataFrame
            A dataframe containing the solution, with columns for time and
            each of the different :math:`y_{i}`
        """
        theta_tuple = tuple([theta[i] for i in range(len(theta))]) \
            if theta.shape else tuple(theta)
        sol = si.odeint(self._rhs, self._y_init, self._times, args=theta_tuple)
        df_dict = {'t': self._times}
        for i in range(self.get_dim_y()):
            df_dict[f'y_{i + 1}'] = sol[:, i]
        return pd.DataFrame(df_dict)

    def get_len_theta(self) -> int:
        """Gets _len_theta

        Returns
        -------
        int
            The number of parameters for the ODE system
        """
        return self._len_theta

    def get_dim_y(self) -> int:
        """Calculates and returns the dimension of the vector :math:`y`.

        Returns
        -------
        The dimension of :math:`y`
        """
        return len(self._y_init) if self._y_init.shape else 1

    def get_title(self) -> str:
        """Gets title of system

        Returns
        -------
        The title of the system
        """
        return self._title
