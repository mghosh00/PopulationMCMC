# Population-Based Markov Chain Monte Carlo
[![Documentation Status](https://readthedocs.org/projects/populationmcmc/badge/?version=latest)](https://populationmcmc.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mghosh00/PopulationMCMC/graph/badge.svg?token=6CRIQSLSRN)](https://codecov.io/gh/mghosh00/PopulationMCMC)

This project provides a framework for using Bayesian inference to infer parameters from a system of ordinary differential equations (ODEs). The inference is carried out using a population-based MCMC algorithm as described in [this paper](https://link.springer.com/article/10.1007/s11222-007-9028-9) by Jasra et al. in Algorithm 1, and takes inspiration from the [pints.PopulationMCMC](https://pints.readthedocs.io/en/latest/mcmc_samplers/population_mcmc.html#pints.PopulationMCMC) class implemented in the [PINTS repository](https://github.com/pints-team/pints).

## Installation
To install a copy of the project, in the terminal run:

	git clone git@github.com:mghosh00/PopulationMCMC.git

To make sure all dependencies are installed, in the current directory run:

	pip install ./PopulationMCMC

## Using the framework to run the algorithm
From here, navigate to PopulationMCMC/population_mcmc/examples/logistic_growth_example.py to see an example of how to use the package. Below is a step-by-step description of how to run your own examples:

# Set up the ODE System
Firstly, create a `rhs` function, which needs to have the following signature:
```
def rhs(y: np.array, t: float, *theta: tuple[float]) -> np.array
```
where `y` is an n-dimensional array, `*theta` is an m-dimensional parameter tuple, and the return value must be n-dimensional. This represents the ODE $`y'(t) = f(y, t; \theta)`$.

## References
Jasra, A., Stephens, D. A. & Holmes, C. C. (2007).
On population-based simulation for static inference
Statistics and Computing, 17, 263-279.

https://link.springer.com/article/10.1007/s11222-007-9028-9


Clerx, M., Robinson, M., Lambert, B., Lei, C. L., Ghosh, S., Mirams, G. R., & Gavaghan, D. J. (2019).
Probabilistic Inference on Noisy Time Series (PINTS).
Journal of Open Research Software, 7(1), 23.

https://doi.org/10.5334/jors.252
