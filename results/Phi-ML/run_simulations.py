import pybamm
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")


experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V",
        "Rest for 2 hours",
    ],
    period="10 seconds",
)
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)

cspmax = 50483 * 1.25  #1.25
csnmax = 29583 * 1.13  #1.13

param["Initial concentration in negative electrode [mol.m-3]"] = 0.90 * csnmax
param["Initial concentration in positive electrode [mol.m-3]"] = 0.26 * cspmax
param["Maximum concentration in negative electrode [mol.m-3]"] = csnmax
param["Maximum concentration in positive electrode [mol.m-3]"] = cspmax
param["Negative electrode Bruggeman coefficient (electrolyte)"] = 1.5
param["Positive electrode Bruggeman coefficient (electrolyte)"] = 1.5
param["Separator Bruggeman coefficient (electrolyte)"] = 1.5
param["Positive electrode diffusivity [m2.s-1]"] = 4E-15
param["Negative electrode diffusivity [m2.s-1]"] = 3.3E-14

sim = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=experiment,
    solver=pybamm.CasadiSolver()
)
sim.solve()

# Show all plots
sim.plot()

# Compare with experiments
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)

voltage = sim.solution["Terminal voltage [V]"]
time = sim.solution["Time [h]"]
