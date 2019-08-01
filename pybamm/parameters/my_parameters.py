import pybamm

R = pybamm.Parameter('Particle radius [m]')
D = pybamm.Parameter('Diffusion coefficient [m2.s-1]')
j = pybamm.Parameter('Interfacial current density [A.m-2]')
F = pybamm.Parameter('Faraday constant [C.mol-1]')
c0 = pybamm.Parameter('Initial concentration [mol.m-3]')