#
# Class to numerically solve a system of ODEs given initial data and specific
# parameter values
#

import typing
import numpy as np
import pandas as pd
import scipy.integrate as si


class ODESystem:
    """This class takes in the RHS of a system of ODEs in the form:
    .. math::
        y'(t) = f(y, t; \\theta), y(0) = y_{0},
    where :math:`y, f, y_{0}` are n-dimensional vectors, :math:`t` is a scalar
    and :math:`\\theta` is an m-dimensional parameter vector.

    """

    def __init__(self, rhs: typing.Callable, y_init: np.array,
                 times: np.array):
        """Constructor Method

        Parameters
        ----------
        rhs : typing.Callable
            A function f : R^{n} x R -> R^{n} which takes in a vector y and
            time t and returns the RHS of the ODE system, given parameters
            \theta
        y_init : np.array
            The initial values of the ODE to be passed to the system
        times : np.array
            The times to be used for the numerical solution
        """
        self._rhs = rhs
        self._times = times
        self._y_init = y_init

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
            each of the different y_i
        """
        theta_tuple = tuple(map(tuple, theta))
        sol = si.odeint(self._rhs, self._y_init, self._times, args=theta_tuple)
        df_dict = {'t': self._times}
        for i in range(self._y_init):
            df_dict[f'y_{i + 1}'] = sol[:, i]
        return pd.DataFrame(df_dict)
