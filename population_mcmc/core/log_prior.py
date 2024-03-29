#
# Class to form a multivariate normal log prior given bounds on each parameter
#

import numpy as np
import scipy as sp


class LogPrior:
    """This class will take in `bounds` on each parameter and form a
    multivariate normal prior with means at the midpoints of each interval, and
    standard deviations equal to 1/8 of the length of the interval to capture
    most of the variance.
    """
    def __init__(self, bounds: np.array):
        """Constructor method

        Parameters
        ----------
        bounds : np.array
            A :math:`2 \\times m` array, where :math:`m` is the number of
            parameters, containing the lower and upper bounds of the parameters
        """
        # Checks
        if bounds.shape[0] != 2 or bounds.shape[1] < 1:
            raise ValueError("bounds must have exactly 2 rows and at least "
                             "one column")
        if not all(np.greater(bounds[1, :], bounds[0, :])):
            raise ValueError("Lower bounds must be in the first row and "
                             "upper bounds in the second row")
        self._bounds = bounds
        self._means = self._calculate_means()
        self._std_devs = self._calculate_std_devs()

    def _calculate_means(self) -> np.array:
        """Finds the midpoints of all intervals and returns this as an array.

        Returns
        -------
        np.array
            An m-dimensional array, where :math:`m` is the number of
            parameters, representing the means of the :class:`LogPrior`
        """
        return np.mean(self._bounds, axis=0)

    def _calculate_std_devs(self) -> np.array:
        """Creates standard deviations for the prior.

        Returns
        -------
        np.array
            An m-dimensional array, where :math:`m` is the number of
            parameters, representing the standard deviations of the
            :class:`LogPrior`
        """
        interval_lengths = self._bounds[1, :] - self._bounds[0, :]
        return interval_lengths / 8

    def __call__(self, theta: np.array) -> float:
        """Accesses the pdf of the :class:`LogPrior` at a specific value of
        `theta` (needs to be an m-dimensional array of parameters).

        Parameters
        ----------
        theta: np.array
            An m-dimensional array of parameters

        Returns
        -------
        float
            The value of the log pdf at the given `theta`
        """
        if theta is None:
            raise ValueError("Chain has not been properly initialised yet")
        num_params = len(theta)
        if num_params == self._bounds.shape[1]:
            return sum([sp.stats.norm.logpdf(theta[i], self._means[i],
                                             self._std_devs[i])
                        for i in range(num_params)])
        else:
            raise ValueError(f"The length of theta must match the number of "
                             f"columns specified in self._bounds ({num_params}"
                             f" != {self._bounds.shape[1]})")

    def get_std_devs(self) -> np.array:
        """Gets the standard deviations from this `LogPrior`.

        Returns
        -------
        np.array
            The standard deviations
        """
        return self._std_devs
