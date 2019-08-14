import pybamm
import numpy as np
import matplotlib.pyplot as plt

# load model
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

param["Negative electrode OCV entropic change"] = "graphite_LGM50_entropic_change.py"
param["Positive electrode OCV entropic change"] =  "nmc_LGM50_entropic_change.py"


param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 100)
solution = model.default_solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(model, mesh, solution)
plot.dynamic_plot()

# other plots
Ueq = pybamm.ProcessedVariable(
    model.variables['X-averaged battery open circuit voltage [V]'], solution.t, solution.y, mesh=mesh
)
Umeas = pybamm.ProcessedVariable(
    model.variables['Measured battery open circuit voltage [V]'], solution.t, solution.y, mesh=mesh
)
etar = pybamm.ProcessedVariable(
    model.variables['X-averaged battery reaction overpotential [V]'], solution.t, solution.y, mesh=mesh
)
etac = pybamm.ProcessedVariable(
    model.variables['X-averaged battery concentration overpotential [V]'], solution.t, solution.y, mesh=mesh
)
Dphis = pybamm.ProcessedVariable(
    model.variables['X-averaged battery solid phase ohmic losses [V]'], solution.t, solution.y, mesh=mesh
)
Dphie = pybamm.ProcessedVariable(
    model.variables['X-averaged battery electrolyte ohmic losses [V]'], solution.t, solution.y, mesh=mesh
)

plt.figure(6)
plt.plot(solution.t,Ueq(solution.t),label="OCV")
plt.plot(solution.t,etar(solution.t),label="reaction op")
plt.plot(solution.t,etac(solution.t),label="concentration op")
plt.plot(solution.t,Dphis(solution.t),label="solid Ohmic")
plt.plot(solution.t,Dphie(solution.t),label="electrolyte Ohmic")
plt.xlabel("t")
plt.ylabel("Voltages [V]")
plt.legend()

plt.show()