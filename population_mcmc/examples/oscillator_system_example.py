#
# Using the framework for a basic oscillator system
#

import numpy as np

from population_mcmc import ODESystem
from population_mcmc import DataGenerator
from population_mcmc import Simulator


# 1. Setting up the ODE system
def oscillator_rhs(y: np.array, t: float, a: float, b: float) -> np.array:
    """RHS of the logistic growth equation
    """
    return np.array([a * y[1],
                     -b * y[0]])


y_init = np.array([0, 1])
times = np.linspace(0, 10, 100)
ode_system = ODESystem(oscillator_rhs, y_init, times, "oscillator")

# 2. Randomly generating data using the DataGenerator
# Here we are using ground truth parameters of a = 2.0 and b = 1.0
true_theta = np.array([2.0, 1.0])
# We allow data to vary with sigma = 0.01 away from the true values for both
true_sigma = np.array([0.01, 0.01])
data_generator = DataGenerator(ode_system, true_theta, true_sigma)
# Get some observed data
y_obs = data_generator.generate_observed_data()

# 3. Running the population MCMC
num_chains = 10
# We set the parameter bounds here
param_bounds = np.array([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 2.0, 2.0]])
param_names = ["a", "b", "sigma_1", "sigma_2"]
# Stick with the default max_its
simulator = Simulator(ode_system, num_chains, y_obs, param_bounds, param_names,
                      max_its=2000)
simulator.run()
simulator.plot_traces()
