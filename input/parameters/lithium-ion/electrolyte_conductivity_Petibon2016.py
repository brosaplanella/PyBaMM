import autograd.numpy as np


def electrolyte_conductivity_Petibon2016(c_e, T, T_inf, E_k_e, R_g):
    """
    Conductivity of LiPF6 in EC:EMC as a function of ion concentration. The original
    data and fit are from [1].

    References
    ----------
    .. [1] R Petibon et al. Electrolyte System for High Voltage Li-Ion Cells. Journal
    of the Electrochemical Society 163 (2016): A2571-A2578.

    Parameters
    ----------
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_k_e: double
        Electrolyte conductivity activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    :`numpy.Array`
        Electrolyte conductivity
    """

    sigma_e = 0.95 * c_e / 1000

    return sigma_e
