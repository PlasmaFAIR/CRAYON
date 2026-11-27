"""
Response function chi = sgn(u_par) F(u) H(lambda)
"""

# ruff: noqa: N806

# Third party imports
import numpy as np
import scipy.constants as const
from scipy.integrate import quad

# Local imports
from crayon.linear_current_drive.kinematics import thermal_momentum
from crayon.shared.physics import lorentz_factor_from_normalised_momentum


def F_func(u: float, Te: float, fc: float, Zeff: float) -> float:  # noqa: N802, N803
    """
    Calculate F function, see eq. (33) in Lin-Liu et al. PoP 2003.
    This is the line of eq. (33), with integral over x.

    Parameters
    ----------
    u : float
        Momentum per unit mass, gamma * v [m/s].
    Te : float
        Electron temperature [eV].
    fc : float
        Effective circulating particle fraction (LL03 eq. 32) [].
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

    gamma = lorentz_factor_from_normalised_momentum(u / const.c)
    gamma_x = lorentz_factor_from_normalised_momentum(u * x_array / const.c)

    tmp_int = (
        x_array ** (rhohat + 3)
        * (1.0 / gamma_x**3)
        * ((1.0 + gamma) / (1.0 + gamma_x)) ** rhohat
    )

    return coeff * d_x * np.sum(tmp_int)


def F_func_quad(u: float, Te: float, fc: float, Zeff: float) -> float:  # noqa: N802, N803
    """
    Calculate F function from the first form of eq. (33) in Lin-Liu et al.
    PoP 2003, an integral over u' written in terms of the Lorentz factor
    gamma, by adaptive quadrature (scipy.integrate.quad).

    Parameters
    ----------
    u : float
        Momentum per unit mass, gamma * v [m/s].
    Te : float
        Electron temperature [eV].
    fc : float
        Effective circulating particle fraction (LL03 eq. 32) [].
    Zeff : float
        Effective ion charge [].

    Returns
    -------
    F : float
        Response function F(u) [].
    """
    ue2 = thermal_momentum(Te) ** 2
    rhohat = (Zeff + 1.0) / fc

    def integrand(up):
        gp = lorentz_factor_from_normalised_momentum(up / const.c)
        return (up / gp) ** 3 * ((gp - 1) / (gp + 1)) ** (rhohat / 2)

    gamma = lorentz_factor_from_normalised_momentum(u / const.c)
    integral, _ = quad(integrand, 0.0, u, epsabs=0.0, epsrel=1e-12, limit=200)

    return ((gamma + 1) / (gamma - 1)) ** (rhohat / 2) * integral / ue2**2 / fc


def F_func_MC(  # noqa: N802
    u: float,
    Te: float,  # noqa: N803
    fc: float,
    Zeff: float,  # noqa: N803
    G_func,  # noqa: N803
    Mee_func,  # noqa: N803
    Mei_func,  # noqa: N803
    Oee_func,  # noqa: N803
) -> float:
    """
    Momentum-conserving F(u) of Hu et al. 2021: the variational solution with
    trial function F = (x / gamma^2) sum_i c_i x^i (Hu21 eq. 24, k = 2), whose
    coefficients solve the 5 x 5 system of Hu21 eq. 30 (Hu16 eq. 28).

    Parameters
    ----------
    u : float
        Momentum per unit mass, gamma * v [m/s].
    Te : float
        Electron temperature [eV].
    fc : float
        Effective circulating particle fraction (LL03 eq. 32) [].
    Zeff : float
        Effective ion charge [].
    G_func, Mee_func, Mei_func, Oee_func : scipy.interpolate.interp1d
        Tables from mc_coefficients.get_MC_coefficients.

    Returns
    -------
    F : float
        Response function F(u) [].
    """
    gamma = lorentz_factor_from_normalised_momentum(u / const.c)
    x = u / thermal_momentum(Te)
    theta = (Te * const.eV) / (const.m_e * const.c**2)

    # Hu21's f_tr = 1 / f_c - 1 (eq. 28), the coefficient of Omega.
    f_tr = 1.0 / fc - 1

    # Matrices of Hu21 eq. 30, with Omega_ei = M_ei (Hu21, after eq. 30).
    G = G_func(theta)
    Mei = Mei_func(theta)
    M = Mee_func(theta) + Zeff * Mei
    Omega = Oee_func(theta) + Zeff * Mei
    A = M + f_tr * Omega

    constraint = A[0, 1:]
    A = np.vstack((
        np.hstack((A[1:, 1:], np.reshape(constraint, (4, 1)))),
        np.reshape(np.append(constraint, 0), (1, 5)),
    ))
    G = np.roll(G, -1)

    d = np.linalg.solve(A, G)

    # Trial function, Hu21 eq. 24 with k = 2 and N = 4.
    series = 0
    for i in range(4):
        series += d[i] * (x ** (i + 1))
    F = series * (x / gamma**2)

    # The right-hand side of eq. 30 is G / f_c.
    return F / fc
