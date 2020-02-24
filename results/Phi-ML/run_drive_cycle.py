import pybamm
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

pybamm.set_logging_level("INFO")


model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
# param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
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

filename = "drive_cycle_1"
param["Current function [A]"] = "[current data]US06"

sim = pybamm.Simulation(
    model,
    parameter_values=param,
    solver=pybamm.CasadiSolver(mode="fast")
)
sim.solve()

# Store data
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)

voltage = sim.solution["Terminal voltage [V]"]
time = sim.solution["Time [h]"]
capacity = sim.solution["Discharge capacity [A.h]"]
ce = sim.solution["Electrolyte concentration [mol.m-3]"]

ce_store = np.transpose(
        np.vstack((time(sim.solution.t), ce(sim.solution.t, x=np.linspace(0,1, 100))))
)

np.savetxt(
    "results/Phi-ML/data/ce_" + filename + ".csv",
    ce_store,
    delimiter=",",
    header="# First column is time in hours, the other columns are the variable values at each gridpoint being the second column x = 0"
)

# Sanity check that the data makes sense
plt.figure(1)
plt.plot(np.transpose(ce_store[:,1:]))

plt.figure(num=2, figsize=(6, 4))
plt.plot(time(sim.solution.t), voltage(sim.solution.t))
plt.xlabel("Time [h]")
plt.ylabel("Voltage [V]")

plt.figure(num=3, figsize=(6, 4))
plt.plot(time(sim.solution.t), capacity(sim.solution.t))
plt.xlabel("Time [h]")
plt.ylabel("Discharge capacity [Ah]")

plt.tight_layout()

print(capacity(sim.solution.t[-1]))

# plt.savefig(
#     "/home/ferranbrosa/LGM50/figures/GITT_t_V.png",
#     dpi=300
# )

# Show all plots
sim.plot()