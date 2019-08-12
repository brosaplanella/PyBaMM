import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.DFN()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

C_rate = 1
param["Typical current [A]"] = (
    C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)

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

phi_e = pybamm.ProcessedVariable(
    model.variables['Electrolyte potential [V]'], solution.t, solution.y, mesh=mesh
)

print(phi_e(t=solution.t[-1],x=0))

plt.figure(2)
plt.plot(np.linspace(0,1),phi_e(t=solution.t[-1],x=np.linspace(0,1)))
plt.show()
