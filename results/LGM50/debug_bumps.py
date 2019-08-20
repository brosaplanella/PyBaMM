import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# load model
# model = pybamm.lithium_ion.DFN()   # has bumps?
model = pybamm.lithium_ion.SPMe()   # has bumps
# model = pybamm.lithium_ion.SPM()   # has bumps

model.variables.update(
    {
        "Ferran's positive ocp": pybamm.standard_parameters_lithium_ion.U_p_dimensional(
            pybamm.PrimaryBroadcast(
                pybamm.surf(pybamm.standard_variables.c_s_p_xav),
                broadcast_domain=["positive electrode"]
            ),
            pybamm.thermal_parameters.T_ref
        ),
        "Ferran's av positive ocp": pybamm.x_average(
            pybamm.standard_parameters_lithium_ion.U_p_dimensional(
                pybamm.PrimaryBroadcast(
                    pybamm.surf(pybamm.standard_variables.c_s_p_xav),
                    broadcast_domain=["positive electrode"]
                ),
                pybamm.thermal_parameters.T_ref
            )
        ),
        "Ferran's other av positive ocp":
        pybamm.standard_parameters_lithium_ion.U_p_dimensional(
            pybamm.x_average(pybamm.PrimaryBroadcast(
                pybamm.surf(pybamm.standard_variables.c_s_p_xav),
                broadcast_domain=["positive electrode"])),
            pybamm.thermal_parameters.T_ref
        ),
        "Ferran's another av positive ocp": pybamm.x_average(
            pybamm.standard_parameters_lithium_ion.U_p_dimensional(
                pybamm.PrimaryBroadcast(
                    pybamm.surf(pybamm.standard_variables.c_s_p_xav),
                    broadcast_domain=["positive electrode"]
                ),
                pybamm.thermal_parameters.T_ref
            ) / pybamm.standard_parameters_lithium_ion.potential_scale
        ) * pybamm.standard_parameters_lithium_ion.potential_scale,
        "Ferran's negative ocp": pybamm.standard_parameters_lithium_ion.U_n_dimensional(
            pybamm.PrimaryBroadcast(
                pybamm.surf(pybamm.standard_variables.c_s_n_xav),
                broadcast_domain=["negative electrode"]
            ),
            pybamm.thermal_parameters.T_ref
        ),
        "Ferran's av negative ocp": pybamm.x_average(
            pybamm.standard_parameters_lithium_ion.U_n_dimensional(
                pybamm.PrimaryBroadcast(
                    pybamm.surf(pybamm.standard_variables.c_s_n_xav),
                    broadcast_domain=["negative electrode"]
                ),
                pybamm.thermal_parameters.T_ref
            )
        ),
        "Ferran's other av negative ocp":
        pybamm.standard_parameters_lithium_ion.U_n_dimensional(
            pybamm.x_average(pybamm.PrimaryBroadcast(
                pybamm.surf(pybamm.standard_variables.c_s_n_xav),
                broadcast_domain=["negative electrode"]
            )),
            pybamm.thermal_parameters.T_ref
        ),
        "Ferran's another av negative ocp": pybamm.x_average(
            pybamm.standard_parameters_lithium_ion.U_n_dimensional(
                pybamm.PrimaryBroadcast(
                    pybamm.surf(pybamm.standard_variables.c_s_n_xav),
                    broadcast_domain=["negative electrode"]
                ),
                pybamm.thermal_parameters.T_ref
            ) / pybamm.standard_parameters_lithium_ion.potential_scale
        ) * pybamm.standard_parameters_lithium_ion.potential_scale
    }
)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

cspmax = 45000
csnmax = 30000

param["Initial concentration in negative electrode [mol.m-3]"] = 0.98 * csnmax
param["Initial concentration in positive electrode [mol.m-3]"] = 0.05 * cspmax
param["Maximum concentration in negative electrode [mol.m-3]"] = csnmax
param["Maximum concentration in positive electrode [mol.m-3]"] = cspmax

param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 1000)
solution = model.default_solver.solve(model, t_eval)

# plot


def OCV_cathode(sto):
    stretch = 1.062
    sto = stretch * sto

    u_eq = (
        2.16216
        + 0.07645 * np.tanh(30.834 - 54.4806 * sto)
        + 2.1581 * np.tanh(52.294 - 50.294 * sto)
        - 0.14169 * np.tanh(11.0923 - 19.8543 * sto)
        + 0.2051 * np.tanh(1.4684 - 5.4888 * sto)
        + 0.2531 * np.tanh((-sto + 0.56478) / 0.1316)
        - 0.02167 * np.tanh((sto - 0.525) / 0.006)
    )
    return u_eq


def OCV_anode(sto):
    u_eq = (
        0.194
        + 1.5 * np.exp(-120.0 * sto)
        + 0.0351 * np.tanh((sto - 0.286) / 0.083)
        - 0.0045 * np.tanh((sto - 0.849) / 0.119)
        - 0.035 * np.tanh((sto - 0.9233) / 0.05)
        - 0.0147 * np.tanh((sto - 0.5) / 0.034)
        - 0.102 * np.tanh((sto - 0.194) / 0.142)
        - 0.022 * np.tanh((sto - 0.9) / 0.0164)
        - 0.011 * np.tanh((sto - 0.124) / 0.0226)
        + 0.0155 * np.tanh((sto - 0.105) / 0.029)
    )
    return u_eq


voltage = pybamm.ProcessedVariable(
    model.variables['Terminal voltage [V]'], solution.t, solution.y, mesh=mesh
)
c_s_n_surf = pybamm.ProcessedVariable(
    model.variables['Negative particle surface concentration [mol.m-3]'], solution.t,
    solution.y, mesh=mesh
)
c_s_p_surf = pybamm.ProcessedVariable(
    model.variables['Positive particle surface concentration [mol.m-3]'], solution.t,
    solution.y, mesh=mesh
)
c_s_n_nd = pybamm.ProcessedVariable(
    model.variables['Negative particle surface concentration'], solution.t, solution.y,
    mesh=mesh
)
x_averaged_c_s_n_nd = pybamm.ProcessedVariable(
    model.variables['X-averaged negative particle surface concentration'], solution.t,
    solution.y, mesh=mesh
)
c_s_p_nd = pybamm.ProcessedVariable(
    model.variables['Positive particle surface concentration'], solution.t, solution.y,
    mesh=mesh
)
x_averaged_c_s_p_nd = pybamm.ProcessedVariable(
    model.variables['X-averaged positive particle surface concentration'], solution.t,
    solution.y, mesh=mesh
)
time = pybamm.ProcessedVariable(
    model.variables['Time [h]'], solution.t, solution.y, mesh=mesh
)
x_averaged_OCV_p = pybamm.ProcessedVariable(
    model.variables['X-averaged positive electrode open circuit potential [V]'],
    solution.t, solution.y, mesh=mesh
)
x_averaged_OCV_n = pybamm.ProcessedVariable(
    model.variables['X-averaged negative electrode open circuit potential [V]'],
    solution.t, solution.y, mesh=mesh
)
P1 = pybamm.ProcessedVariable(
    model.variables["Ferran's positive ocp"], solution.t, solution.y, mesh=mesh
)
P2 = pybamm.ProcessedVariable(
    model.variables["Ferran's av positive ocp"], solution.t, solution.y, mesh=mesh
)
P3 = pybamm.ProcessedVariable(
    model.variables["Ferran's other av positive ocp"], solution.t, solution.y, mesh=mesh
)
P4 = pybamm.ProcessedVariable(
    model.variables["Ferran's another av positive ocp"], solution.t, solution.y,
    mesh=mesh
)
N1 = pybamm.ProcessedVariable(
    model.variables["Ferran's negative ocp"], solution.t, solution.y, mesh=mesh
)
N2 = pybamm.ProcessedVariable(
    model.variables["Ferran's av negative ocp"], solution.t, solution.y, mesh=mesh
)
N3 = pybamm.ProcessedVariable(
    model.variables["Ferran's other av negative ocp"], solution.t, solution.y, mesh=mesh
)
N4 = pybamm.ProcessedVariable(
    model.variables["Ferran's another av negative ocp"], solution.t, solution.y,
    mesh=mesh
)

plt.figure(1)
plt.plot(time(solution.t), x_averaged_OCV_p(solution.t), label="model OCV")
# plt.plot(time(solution.t), P1(solution.t, x=0), label="P1")  # P1 produces NaNs
plt.plot(time(solution.t), P2(solution.t), label="P2")
plt.plot(time(solution.t), P3(solution.t), label="P3")
plt.plot(time(solution.t), P4(solution.t), label="P4")
plt.xlabel('t [h]')
plt.ylabel('Positive OCP')
plt.legend()

plt.figure(2)
plt.plot(time(solution.t), x_averaged_OCV_n(solution.t), label="SPMe OCV")
# plt.plot(time(solution.t), N1(solution.t, x=0), label="P1")  # N1 produces NaNs
plt.plot(time(solution.t), N2(solution.t), label="N2")
plt.plot(time(solution.t), N3(solution.t), label="N3")
plt.plot(time(solution.t), N4(solution.t), label="N4")
plt.xlabel('t [h]')
plt.ylabel('Negative OCP')
plt.legend()

plt.figure(3)
plt.plot(
    time(solution.t),
    x_averaged_OCV_p(solution.t) - x_averaged_OCV_n(solution.t),
    label="SPMe OCV"
)
plt.plot(time(solution.t), P2(solution.t) - N2(solution.t), label="N2")
plt.plot(time(solution.t), P3(solution.t) - N3(solution.t), label="N3")
plt.plot(time(solution.t), P4(solution.t) - N4(solution.t), label="N4")
plt.xlabel('t [h]')
plt.ylabel('OCP difference')
plt.legend()

plt.figure(5)
plt.plot(
    np.linspace(0, 1, 1E4),
    OCV_cathode(np.linspace(0, 1, 1E4)),
    color="C0", label="positive"
)
plt.plot(
    np.linspace(0, 1, 1E4),
    OCV_anode(np.linspace(0, 1, 1E4)),
    color="C1", label="negative"
)
plt.xlabel("SoC")
plt.ylabel("OCP [V]")
plt.legend()

plt.show()
