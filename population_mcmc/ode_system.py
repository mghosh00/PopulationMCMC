#
# Class to numerically solve a system of ODEs given initial data and specific
# parameter values
#

import numpy as np
import pandas as pd
import scipy.integrate as si


class ODESystem:
    """This class takes in the RHS of a system of ODEs in the form:
        y'(t) = f(y, t; \theta), y(0) = y_{0},
    where y, f, y_{0} are n-dimensional vectors, t is a scalar and \theta is an
    m-dimensional parameter vector.
    """
