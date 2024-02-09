# Population-Based Markov Chain Monte Carlo
[![Documentation Status](https://readthedocs.org/projects/populationmcmc/badge/?version=latest)](https://populationmcmc.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mghosh00/PopulationMCMC/graph/badge.svg?token=6CRIQSLSRN)](https://codecov.io/gh/mghosh00/PopulationMCMC)

This project provides a framework for using Bayesian inference to infer parameters from a system of ordinary differential equations (ODEs). The inference is carried out using a population-based MCMC algorithm as described in [this paper](https://link.springer.com/article/10.1007/s11222-007-9028-9) by Jasra et al. in Algorithm 1, and takes inspiration from the [pints.PopulationMCMC](https://pints.readthedocs.io/en/latest/mcmc_samplers/population_mcmc.html#pints.PopulationMCMC) class implemented in the [PINTS repository](https://github.com/pints-team/pints).

## Explanation of the algorithm
This algorithm follows the method described in the paper by [Jasra et al.](https://link.springer.com/article/10.1007/s11222-007-9028-9).

Population MCMC is a single chain Monte Carlo method, but uses multiple internal chains during the update steps. This package is specifically for systems of ODEs, so we start with the system:	
$$y'(t) = f(y, t; \theta),\qquad y(0) = y_{0}.$$ 
Our goal is to find the posterior, $`p(\theta|y)`$, using Bayesian inference techniques.

The first step of the algorithm is to sample values $`\theta`$ from some prior distribution. Next, we initialise $N$ chains which all have different densities, depending on some tempering parameter, $`T_{i}`$ (where here, density refers to $`likelihood \times prior`$). Denoting each density by $`\pi_{i}(\theta_{i};y)`$ for $`i = 1,...,N`$, we define these densities as	
$$\pi_{i}(\theta_{i};y) = (\pi_{1}(\theta_{i};y))^{1 - T_{i}}.$$ 
Here, the tempering parameters are $`T_{i}=\frac{i - 1}{N}`$ using a Uniformly Spaced tempering scheme as described in the above paper. Note that chain $1$ is the target chain with the density we are trying to get.

Next we do the following steps over each iteration of the algorithm:

1. Mutate
Randomly (with uniform probabiliy) choose chain $i$, and perform a mutation step exactly as in the Metropolis MCMC algorithm (with some acceptance/failure probability).
2. Exchange
Randomly (with uniform probability) choose a different chain $j$, and choose to exchange the parameters with probability $`min(1, A)`$, where
$$A=\frac{\pi_{i}(\theta_{j};y)\pi_{j}(\theta_{i};y)}{\pi_{i}(\theta_{i};y)\pi_{j}(\theta_{j};y)}.$$

Then we terminate the algorithm once the target chain (chain $1$) has suitably converged to the desired parameters.

For the tempered densities, the greater the value of $`T_{i}`$ (corresponding to a higher "temperature"), the flatter the density function will be. This implies that accepting an exchange with a much lower temperature becomes very likely if the higher temperature density has better parameters than the lower temperature density, as $A$ will be large in this case. This is the underlying hope as to why the Population MCMC may perform better than other single chain methods, and it does not take up the same time cost as multi-chain methods as it only updates one chain each time step.

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

## Documentation
All classes are documented on readthedocs.io [here](https://populationmcmc.readthedocs.io/en/latest/).

## References
Jasra, A., Stephens, D. A. & Holmes, C. C. (2007).
On population-based simulation for static inference
Statistics and Computing, 17, 263-279.

https://link.springer.com/article/10.1007/s11222-007-9028-9


Clerx, M., Robinson, M., Lambert, B., Lei, C. L., Ghosh, S., Mirams, G. R., & Gavaghan, D. J. (2019).
Probabilistic Inference on Noisy Time Series (PINTS).
Journal of Open Research Software, 7(1), 23.

https://doi.org/10.5334/jors.252
