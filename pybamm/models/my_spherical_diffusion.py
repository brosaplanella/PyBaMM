import pybamm

class MySphericalDiffusion(pybamm.BaseModel):
    """A model for diffusion in a sphere.

    **Extends:** :class:‘pybamm.BaseModel‘
    """
    def __init__(self, param, name="Spherical Diffusion"):
        # Initialise base class
        super().__init__(name)

        # Add parameters as an attribute of the model
        self.param = param

        # 2. Define your variables and parameters
        c = pybamm.Variable('Concentration', domain = ['negative particle'])
        N = - param.D/param.R * pybamm.grad(c)

        # 3. State the governing equations
        self.rhs = {c: - 1/param.R * pybamm.div(N)}

        # 4. State boundary conditions
        self.boundary_conditions = {c: {'left': (0 , 'Neumann'), 'right': ( -param.j*param.R/(param.F*param.D*param.c0), 'Neumann')}}

        # 5. State initial conditions
        self.initial_conditions = {c: pybamm.Scalar(1)}

        # 6. State output variables
        self.variables = {
            'Concentration': c, 
            'Surface concentration': pybamm.surf(c), 
            'Flux': N, 
            'Concentration [mol.m-3]': c*param.c0, 
            'Surface concentration [mol.m-3]': param.c0*pybamm.surf(c), 
            'Flux [mol.m-2.s-1]': N*param.c0/param.R
        }