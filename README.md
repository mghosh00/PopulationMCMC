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

### Set up the ODE System
Firstly, create a `rhs` function, which needs to have the following signature:
```
def rhs(y: np.array, t: float, *theta: tuple[float]) -> np.array
```
where `y` is an $n$-dimensional array, `*theta` is an $m$-dimensional parameter tuple, and the return value must be $n$-dimensional. This represents the ODE:	$`y'(t) = f(y, t; \theta)`$.

Next, `y_init` ($n$-dimensional array containing initial values of $y$) and `times` ($p$-dimensional array containing the range of times for the numerical solve) need to be specified before instantiating an `ODESystem` ([docs here](https://populationmcmc.readthedocs.io/en/latest/core.html#population_mcmc.ODESystem)).

### Generate/import data to perform inference against
From here, there are two options. 
1. Choose ground truth values of the parameters `theta` and hyperparameters `sigma` representing standard deviations for each element of $y$ (`y_std_devs`).
2. Import your own data with unknown parameters to be determined by the model.

For option 1, simply input the `ODESystem` along with the chosen `theta` and `sigma` arrays (length $m$ and $n$ respectively) into the `DataGenerator` class, which will generate some noisy data as a solution to the `ODESystem` in the form of a $p\times n$ array, which we denote as `y_obs`.

For option 2, the data must be converted into a $p\times n$ array, with rows corresponding to timesteps and columns to elements of $y$.

### Run the Population MCMC algorithm
In order to run the `Simulator`, a list of required and optional arguments can be passed to the constructor method, as described in the table below.

|Argument|Type|Description|Default (if optional)|
| --- | --- | --- | --- |
|`ode_system`|`population_mcmc.ODESystem`|The system of ODEs to be solved|Required|
|`num_chains`|`int`|The number of internal chains used in the population MCMC|Required|
|`y_obs`|`numpy.array`|The observed/generated data as described above|Required|
|`param_bounds`|`numpy.array`|A $`2 \times (m+n)`$ array, containing the lower and upper bounds of the `theta` and the `y_std_devs`|Required|
|`param_names`|`list[str]`|Names for the parameters, to be used when logging to the user|`["param_1", ..., "param_(m+n)"]`|
|`max_its`|`int`|The maximum number of iterations that the simulator will run for|`1000`|
|`init_phase_its`|`int`|The number of iterations for the initial phase of the inference|`500`|

Once these have been chosen, call `simulation.run()` to perform the inference, and `simulation.plot_traces()` to plot the traces of all the chains. An optional parameter, representing the chain id, can be passed to the plotting function. In this case, only the specified chain will be plotted (`id = 1` corresponds to the chain with the target distribution).

## References
Jasra, A., Stephens, D. A. & Holmes, C. C. (2007).
On population-based simulation for static inference
Statistics and Computing, 17, 263-279.

https://link.springer.com/article/10.1007/s11222-007-9028-9


Clerx, M., Robinson, M., Lambert, B., Lei, C. L., Ghosh, S., Mirams, G. R., & Gavaghan, D. J. (2019).
Probabilistic Inference on Noisy Time Series (PINTS).
Journal of Open Research Software, 7(1), 23.

https://doi.org/10.5334/jors.252
