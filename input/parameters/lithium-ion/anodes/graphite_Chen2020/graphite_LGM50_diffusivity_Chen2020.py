<<<<<<< HEAD
<<<<<<< HEAD
from pybamm import exp
=======
import autograd.numpy as np

>>>>>>> #573 update final LGM50 parameters
=======
from pybamm import exp
>>>>>>> #573 setup test as experiment, still fails to converge

def graphite_LGM50_diffusivity_Chen2020(sto, T, T_inf, E_D_s, R_g):
    """
      LG M50 Graphite diffusivity as a function of stochiometry, in this case the
      diffusivity is taken to be a constant. The value is taken from [1].
      References
      ----------
      .. [1] Work in progress
      Parameters
      ----------
      sto: :class: `numpy.Array`
         Electrode stochiometry
      T: :class: `numpy.Array`
         Dimensional temperature
      T_inf: double
         Reference temperature
      E_D_s: double
         Solid diffusion activation energy
      R_g: double
         The ideal gas constant
      Returns
      -------
      : double
         Solid diffusivity
   """

<<<<<<< HEAD
<<<<<<< HEAD
    D_ref = 3.3e-14
    arrhenius = exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    return D_ref * arrhenius
=======
    D_ref = 6e-15 * 2.2  # * 2.2 * 5.5 * 10.5
    arrhenius = np.exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    correct_shape = 0 * sto

    return D_ref * arrhenius + correct_shape
>>>>>>> #573 update final LGM50 parameters
=======
    D_ref = 3.3e-14
    arrhenius = exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    return D_ref * arrhenius
>>>>>>> #573 setup test as experiment, still fails to converge
