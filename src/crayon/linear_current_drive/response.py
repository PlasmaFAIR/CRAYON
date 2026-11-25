"""
Response function chi = sgn(u_par) F(u) H(lambda)
"""

# Third party imports
import numpy as np
import scipy.constants as const
from scipy.integrate import quad

# Local imports
from crayon.linear_current_drive.kinematics import thermal_momentum


def F_func(fc: float, Te: float, u: float, Zeff: float) -> float:  # noqa: N802, N803
    """
    Calculate F function, see eq. (33) in Lin-Liu et al. PoP 2003.
    This is the line of eq. (33), with integral over ``x``.
    Ported from linliu_subroutines.F_func; Te now in eV (was keV).

    Parameters
    ----------
    fc : float
        Effective circulating particle fraction (LL03 eq. 32) [].
    Te : float
        Electron temperature [eV].
    u : float
        Momentum per unit mass, gamma * v [m/s].
    Zeff : float
        Effective ion charge [].

    Returns
    -------
    F : float
        Response function F(u) [].
    """
    ue2 = thermal_momentum(Te) ** 2
    coeff = u**4.0 / ue2**2.0 / fc
    rhohat = (Zeff + 1.0) / fc
    x_array = np.linspace(0, 1, num=31, dtype=float)
    x_array = (x_array[1:] + x_array[:-1]) / 2
    d_x = x_array[1] - x_array[0]

    tmp_int = (
        x_array ** (rhohat + 3)
        * (1.0 / (1 + (u * x_array / const.c) ** 2) ** 1.5)
        * (
            (1 + np.sqrt(1 + (u / const.c) ** 2))
            / (1 + np.sqrt(1 + (u * x_array / const.c) ** 2))
        )
        ** rhohat
    )

    return coeff * d_x * np.sum(tmp_int)


def F_func_gamma(fc: float, Te: float, u: float, Zeff: float) -> float:  # noqa: N802, N803
    """
    Calculate F function from the first form of eq. (33) in Lin-Liu et al.
    PoP 2003, an integral over u' written in terms of the Lorentz factor
    gamma, by adaptive quadrature (scipy.integrate.quad).

    Parameters
    ----------
    fc : float
        Effective circulating particle fraction (LL03 eq. 32) [].
    Te : float
        Electron temperature [eV].
    u : float
        Momentum per unit mass, gamma * v [m/s].
    Zeff : float
        Effective ion charge [].

    Returns
    -------
    F : float
        Response function F(u) [].
    """
    ue2 = thermal_momentum(Te) ** 2
    rhohat = (Zeff + 1.0) / fc

    def lorentz_factor(up):
        return np.sqrt(1 + (up / const.c) ** 2)

    def integrand(up):
        gp = lorentz_factor(up)
        return (up / gp) ** 3 * ((gp - 1) / (gp + 1)) ** (rhohat / 2)

    gamma = lorentz_factor(u)
    integral, _ = quad(integrand, 0.0, u, epsabs=0.0, epsrel=1e-12, limit=200)

    return ((gamma + 1) / (gamma - 1)) ** (rhohat / 2) * integral / ue2**2 / fc
