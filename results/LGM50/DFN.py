import pybamm
import numpy as np

# load model
model = pybamm.lithium_ion.DFN()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = pybamm.ParameterValues("input/parameters/lithium-ion/LGM50_parameters.csv")
param.update({
    "Electrolyte conductivity": "electrolyte_conductivity_Petibon2016.py",
    "Electrolyte diffusivity": "electrolyte_diffusivity_Stewart2008.py",
    "Negative electrode OCV": "graphite_LGM50_ocp_CC3.py",
    "Positive electrode OCV": "nmc_LGM50_ocp_CC3.py",
    "Negative electrode diffusivity": "graphite_LGM50_diffusivity_CC3.py",
    "Positive electrode diffusivity": "nmc_LGM50_diffusivity_CC3.py",
    "Negative electrode OCV entropic change": "graphite_entropic_change_Moura.py",
    "Positive electrode OCV entropic change": "lico2_entropic_change_Moura.py",
    "Negative electrode reaction rate": "graphite_electrolyte_reaction_rate.py",
    "Positive electrode reaction rate": "lico2_electrolyte_reaction_rate.py",
    "Typical current [A]": 5,
    "Current function": pybamm.GetConstantCurrent(current=5)
})
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
