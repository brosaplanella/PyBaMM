import pybamm
import numpy as np
import matplotlib.pyplot as plt

## Define model

# 1. Initialise an empty model
model = pybamm.BaseModel()

# 2. Define your variables
c = pybamm.Variable('Concentration', domain = ['negative particle'])
N = - pybamm.grad(c)

# 3. State the governing equations
model.rhs = {c: - pybamm.div(N)}

# 4. State boundary conditions
model.boundary_conditions = {c: {'left': (0 , 'Neumann'), 'right': (2 , 'Neumann')}}

# 5. State initial conditions
model.initial_conditions = {c: pybamm.Scalar(1)}

# 6. State output variables
model.variables = {'Concentration': c}

## Solve model

# Define geometry
r = pybamm.SpatialVariable('r', domain = ['negative particle'], coord_sys = 'spherical polar')
geometry = {'negative particle': {'primary': {r: {'min': pybamm.Scalar(0), 'max': pybamm.Scalar(1)}}}}

# Define parameters
param = pybamm.ParameterValues()
param.process_geometry(geometry)
param.process_model(model)

# Define mesh
submesh_types = {'negative particle': pybamm.Uniform1DSubMesh}
var_pts = {r: 100}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

# Define discretisation
spatial_methods = {'negative particle': pybamm.FiniteVolume}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Define solver
solver = pybamm.ScipySolver()
t = np.linspace(0,1,100)
solution = solver.solve(model, t)

# Postprocessing
c_out = pybamm.ProcessedVariable(model.variables['Concentration'], solution.t, solution.y, mesh)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c_out(solution.t, r=1))
ax1.set_xlabel("t")
ax1.set_ylabel("Surface concentration")
rr = np.linspace(0, 1, 100)
ax2.plot(rr, c_out(t=0.5, r=rr))
ax2.set_xlabel("r")
ax2.set_ylabel("Concentration at t=0.5")
plt.tight_layout()
plt.show()