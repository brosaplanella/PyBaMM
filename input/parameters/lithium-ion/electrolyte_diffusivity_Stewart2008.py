import autograd.numpy as np


def electrolyte_diffusivity_Stewart2008(c_e, T, T_inf, E_D_e, R_g):
    """
    Diffusivity of LiPF6 in EC:DEC as a function of ion concentration. The original
    data and fit are from [1].

    References
    ----------
    .. [1] S G Stewart and J Newman. The Use of UV/vis Absorption to Measure Diffusion
    Coefficients in LiPF6 Electrolytic Solutions. Journal of The Electrochemical
    Society 155 (2008): F13-F16.

    Parameters
    ----------
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_D_e: double
        Electrolyte diffusion activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    :`numpy.Array`
        Electrolyte diffusivity
    """

    D_c_e = 1.3*2.582E-9 * np.exp(-2.85E-3 * c_e)
    arrhenius = np.exp(E_D_e / R_g * (1 / T_inf - 1 / T))

    return D_c_e * arrhenius

