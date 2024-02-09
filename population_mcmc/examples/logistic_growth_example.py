#
# Using the framework for logistic growth equation
#

import numpy as np

from population_mcmc import ODESystem
from population_mcmc import DataGenerator
from population_mcmc import Simulator


# 1. Setting up the ODE system
def logistic_rhs(y: float, t: float, r: float, k: float) -> float:
    """RHS of the logistic growth equation
    """
    return r * y * (1 - y / k)


y_init = np.array(5.0)
times = np.linspace(0, 10, 100)
ode_system = ODESystem(logistic_rhs, y_init, times, "logistic_growth")

# 2. Randomly generating data using the DataGenerator
# Here we are using ground truth parameters of r = 3.0 and k = 7.0
true_theta = np.array([3.0, 7.0])
# We allow data to vary with sigma = 0.01 away from the true values
true_sigma = np.array(0.01)
data_generator = DataGenerator(ode_system, true_theta, true_sigma)
# Get some observed data
y_obs = data_generator.generate_observed_data()

# 3. Running the population MCMC
num_chains = 10
# We set the parameter bounds here
param_bounds = np.array([[0.0, 5.0, 0.0], [10.0, 10.0, 2.0]])
param_names = ["r", "k", "sigma"]
# Stick with the default max_its
simulator = Simulator(ode_system, num_chains, y_obs, param_bounds, param_names,
                      max_its=2000)
simulator.run()
simulator.plot_traces()
