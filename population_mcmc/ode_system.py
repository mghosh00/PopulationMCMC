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
    y'(t) = f(y, t; \theta), y(0) = y_{0}, where y, f, y_{0} are n-dimensional
    vectors, t is a scalar and \theta is an m-dimensional parameter vector.

    """

    def __init__(self, rhs: typing.Callable, y_init: np.array,
                 times: np.array):
        """

        Parameters
        ----------
        rhs : typing.Callable
            A function f : R^{n} x R -> R^{n} which takes in a vector y and
            time t and returns the RHS of the ODE system, given parameters
            \theta.
        y_init : np.array
            The initial values of the ODE to be passed to the system.
        times : np.array
            The times to be used for the numerical solution.
        """
        pass

