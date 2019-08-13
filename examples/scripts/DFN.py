import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.DFN()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 10000)
solution = model.default_solver.solve(model, t_eval)

# plot
voltage = pybamm.ProcessedVariable(
    model.variables['Terminal voltage [V]'], solution.t, solution.y, mesh=mesh
)
c_s_n_surf = pybamm.ProcessedVariable(
    model.variables['Negative particle surface concentration'], solution.t, solution.y, mesh=mesh
)
c_s_p_surf = pybamm.ProcessedVariable(
    model.variables['Positive particle surface concentration'], solution.t, solution.y, mesh=mesh
)
print("Terminal voltage: ",voltage(solution.t[-1]))
print("Positive concentration: ",c_s_p_surf(solution.t[-1],x=1))
print("Negative concentration: ",c_s_n_surf(solution.t[-1],x=0))

plot = pybamm.QuickPlot(model, mesh, solution)
plot.dynamic_plot()
