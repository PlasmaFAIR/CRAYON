"""
Unit tests for linear_current_drive.kinematics.
"""

# Third party imports
import numpy as np
import numpy.testing as npt
import pytest
import scipy.constants as const

# Local imports
from crayon.linear_current_drive.kinematics import (
    lmdba_func,
    thermal_momentum,
    upar_func_gamma,
    uper2_func_gamma,
)


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


class TestMomentum:
    """
    Unit tests for upar_func_gamma and uper2_func_gamma (LL03 eqs. 41-42).
    """

    # Each entry: y = l w_c / w and n_par. Includes l = 2 and negative n_par.
    params = ((0.9, 0.3), (1.0, -0.4), (0.95, 0.6), (2.1, 0.2))

    gamma = np.linspace(1.0, 1.2, 41)

    @staticmethod
    @pytest.mark.parametrize(("y", "npar"), params)
    def test_upar(y: float, npar: float):
        """
        Every point satisfies gamma - y - n_par u_par / c = 0 (LL03 eq. 41).
        """
        gamma = TestMomentum.gamma
        upar = upar_func_gamma(gamma, y, npar)
        npt.assert_allclose(gamma - npar * upar / const.c, y, rtol=1e-14)

    @staticmethod
    @pytest.mark.parametrize(("y", "npar"), params)
    def test_uper(y: float, npar: float):
        """
        Every point satisfies gamma^2 = 1 + (u_par^2 + u_perp^2) / c^2
        (LL03 eq. 42).
        """
        gamma = TestMomentum.gamma
        upar = upar_func_gamma(gamma, y, npar)
        uper2 = uper2_func_gamma(gamma, y, npar)
        npt.assert_allclose(
            1 + (upar**2 + uper2) / const.c**2, gamma**2, rtol=1e-12
        )

    @staticmethod
    def test_cold_resonance():
        """
        For y = 1, the point gamma = 1 is at rest: u_par = u_perp = 0
        (LL03 eqs. 41-42).
        """
        npt.assert_equal(upar_func_gamma(1.0, 1.0, 0.3), 0.0)
        npt.assert_equal(uper2_func_gamma(1.0, 1.0, 0.3), 0.0)


class TestLmdbaFunc:
    """
    Unit tests for lmdba_func (LL03 eq. 22).
    """

    # Each entry: B_max [T], B [T] and |u| [m/s]. Includes B = B_max.
    params = ((2.0, 2.0, 1.0e7), (2.4, 1.6, 3.0e7), (3.0, 1.2, 1.2e8))

    @staticmethod
    @pytest.mark.parametrize(("Bmax", "B", "u"), params)
    def test_u_perp(Bmax: float, B: float, u: float):  # noqa: N803
        """
        All momentum perpendicular (u_par = 0): lambda = B_max / B.
        """
        npt.assert_allclose(lmdba_func(Bmax, B, u**2, u), Bmax / B)

    @staticmethod
    @pytest.mark.parametrize(("Bmax", "B", "u"), params)
    def test_u_par(Bmax: float, B: float, u: float):  # noqa: N803
        """
        All momentum parallel (u_perp = 0): lambda = 0.
        """
        npt.assert_equal(lmdba_func(Bmax, B, 0.0, u), 0.0)

    @staticmethod
    def test_lambda():
        """
        0 <= lambda <= B_max / B (LL03 eq. 22).
        """
        Bmax, B = 2.4, 1.6  # noqa: N806
        y, npar = 1.02, 0.3
        gamma = np.linspace(1.0, 1.5, 100)
        uper2 = uper2_func_gamma(gamma, y, npar)
        # Keep the resonant part of the curve, where u_perp^2 >= 0.
        resonant = uper2 >= 0
        gamma, uper2 = gamma[resonant], uper2[resonant]
        u = const.c * np.sqrt(gamma**2 - 1)  # (LL03 eq. 42).
        lmbda = lmdba_func(Bmax, B, uper2, u)
        assert np.all((lmbda >= 0) & (lmbda <= Bmax / B))
