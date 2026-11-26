"""
Momenta per unit mass used by the linear current drive model
"""

# Third party imports
import scipy.constants as const

# Local imports
from crayon.shared.physics import normalised_thermal_velocity_nonrelativistic


def thermal_momentum(Te: float) -> float:  # noqa: N803
    """
    Thermal momentum per unit mass u_e = sqrt(2 T_e / m) (LL03, below eq. 17).

    Thin wrapper around crayon.shared.physics
    .normalised_thermal_velocity_nonrelativistic, which returns u_e / c.

    Parameters
    ----------
    Te : float
        Electron temperature [eV].

    Returns
    -------
    ue : float
        Thermal momentum per unit mass [m/s].
    """
    return const.c * normalised_thermal_velocity_nonrelativistic(Te)


def upar_func_gamma(gma: float, y: float, npar: float) -> float:
    """
    Parallel momentum per unit mass u_par, as a function of the
    Lorentz factor gamma (LL03 eqs. 41-42).

    Parameters
    ----------
    gma : float
        Lorentz factor gamma [].
    y : float
        Harmonic number times cyclotron over wave frequency, l w_c / w [].
    npar : float
        Parallel refractive index n_par [].

    Returns
    -------
    upar : float
        Parallel momentum per unit mass [m/s].
    """
    return const.c * (gma - y) / npar


def uper2_func_gamma(gma: float, y: float, npar: float) -> float:
    """
    Square of the perpendicular momentum per unit mass u_perp^2,
    as a function of the Lorentz factor gamma (LL03 eqs. 41-42).

    Parameters
    ----------
    gma : float
        Lorentz factor gamma [].
    y : float
        Harmonic number times cyclotron over wave frequency, l w_c / w [].
    npar : float
        Parallel refractive index n_par [].

    Returns
    -------
    uper2 : float
        Perpendicular momentum per unit mass squared [m^2/s^2].
    """
    return const.c**2 * ((gma**2) - 1 - ((gma - y) / npar) ** 2)


def lmdba_func(Bmax: float, B: float, uper2: float, u: float) -> float:  # noqa: N803
    """
    Pitch-angle variable lambda = (B_max / B) (u_perp^2 / u^2) (LL03 eq. 22).

    Parameters
    ----------
    Bmax : float
        Maximum magnetic field strength on the flux surface [T].
    B : float
        Local magnetic field strength [T].
    uper2 : float
        Perpendicular momentum per unit mass squared [m^2/s^2].
    u : float
        Total momentum per unit mass [m/s].

    Returns
    -------
    lmbda : float
        Pitch-angle variable lambda [].
    """
    return (Bmax / B) * (uper2 / u**2)
