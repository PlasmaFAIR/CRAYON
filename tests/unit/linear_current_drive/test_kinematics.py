"""
Unit tests for linear_current_drive.kinematics.
"""

# Third party imports
import numpy as np
import numpy.testing as npt
import pytest
import scipy.constants as const

# Local imports
from crayon.linear_current_drive.kinematics import thermal_momentum


class TestThermalMomentum:
    """
    Unit tests for thermal_momentum: u_e = sqrt(2 T_e / m).
    """

    params_te = (1.0, 2.0e3, 25.0e3, 100.0e3)

    @staticmethod
    @pytest.mark.parametrize("Te", params_te)
    def test_definition(Te: float):  # noqa: N803
        """
        Equals sqrt(2 T_e / m) with T_e converted from eV to J.
        """
        npt.assert_allclose(
            thermal_momentum(Te),
            np.sqrt(2 * Te * const.eV / const.m_e),
            rtol=1e-14,
        )

    @staticmethod
    def test_known_value():
        """
        u_e at 1 keV is about 1.87554e7 m/s.
        """
        npt.assert_allclose(thermal_momentum(1.0e3), 1.87554e7, rtol=1e-5)

    @staticmethod
    def test_scaling():
        """
        u_e scales as sqrt(Te).
        """
        Te = 1.0e3  # noqa: N806
        factor = 4.0
        npt.assert_allclose(
            thermal_momentum(factor * Te),
            np.sqrt(factor) * thermal_momentum(Te),
            rtol=1e-14,
        )

    @staticmethod
    def test_array():
        """
        Accepts an array of temperatures.
        """
        Te = np.array([1.0, 2.0e3, 25.0e3])  # noqa: N806
        npt.assert_allclose(
            thermal_momentum(Te),
            [thermal_momentum(t) for t in Te],
            rtol=1e-14,
        )
