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
