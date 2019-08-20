import autograd.numpy as np


def electrolyte_conductivity_Nyman2008(c_e, T, T_inf, E_k_e, R_g):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] A Nyman et al. Electrochemical characterisation and modelling of the mass
    transport phenomena in LiPF6-EC-EMC electrolyte. Electrochimica Acta 53 (2008):
    6356-6365.

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
        Solid diffusivity
    """

    sigma_e = (
        0.1297 * (c_e / 1000) ** 3
        - 2.51 * (c_e / 1000) ** 1.5
        + 3.329 * (c_e / 1000)
    )

    arrhenius = np.exp(E_k_e / R_g * (1 / T_inf - 1 / T))

    return sigma_e * arrhenius
