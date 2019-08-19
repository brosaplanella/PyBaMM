import autograd.numpy as np


def nmc_LGM50_diffusivity_CC3(sto, T, T_inf, E_D_s, R_g):
    """
       LiCo2 diffusivity as a function of stochiometry, in this case the
       diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

       References
       ----------
       .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

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

    D_ref = 1e-15
    arrhenius = np.exp(E_D_s / R_g * (1 / T_inf - 1 / T))

    correct_shape = 0 * sto

    return D_ref * arrhenius + correct_shape
