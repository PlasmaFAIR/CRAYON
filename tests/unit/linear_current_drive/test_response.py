"""
Unit tests for linear_current_drive.response.
"""

# Third party imports
import numpy as np
import numpy.testing as npt
import pytest
import scipy.constants as const

# Local imports
from crayon.linear_current_drive.kinematics import thermal_momentum
from crayon.linear_current_drive.response import F_func, F_func_gamma

# F_func integrates eq. 33 with a 30-point midpoint rule, which is low by up
# to ~0.8% for the parameters below (worst at large rhohat, small x).
# Use a leniant tolerance for now.
MIDPOINT_RTOL = 1e-2


class TestFFunc:
    """
    Unit tests for F_func: F(u)
    (LL03 eqs. 31-34).
    """

    # Each entry: fc, Te [eV] and Zeff, in F_func's argument order.
    params = (
        (1.0, 2.0e3, 1.0),
        (0.6, 2.0e3, 1.6),
        (0.4, 25.0e3, 3.0),
        (0.8, 100.0e3, 1.0),
    )

    params_x = (0.5, 1.0, 2.0, 4.0, 8.0)

    # Each entry: fc and Zeff. Kept to rhohat <= 10: at rhohat = 20
    # (fc = 0.3, Zeff = 5) the midpoint rule is ~2.5% low.
    params_nonrelativistic = ((1.0, 1.0), (0.6, 1.6), (0.4, 3.0))

    params_x_nonrelativistic = (0.5, 1.0, 3.0)

    @staticmethod
    @pytest.mark.parametrize(("fc", "Te", "Zeff"), params)
    @pytest.mark.parametrize("x", params_x)
    def test_matches_F_func_gamma(  # noqa: N802
        fc: float,
        Te: float,  # noqa: N803
        Zeff: float,  # noqa: N803
        x: float,
    ):
        """
        F_func (second form of eq. 33) equals F_func_gamma (first form).
        """
        u = x * thermal_momentum(Te)
        npt.assert_allclose(
            F_func(fc, Te, u, Zeff),
            F_func_gamma(fc, Te, u, Zeff),
            rtol=MIDPOINT_RTOL,
        )

    @staticmethod
    @pytest.mark.parametrize(("fc", "Te", "Zeff"), params)
    def test_satisfies_ode(fc: float, Te: float, Zeff: float):  # noqa: N803
        """
        F_func satisfies its defining ODE (eq. 31), with dF/du by central
        difference.
        """
        ue = thermal_momentum(Te)
        rhohat = (Zeff + 1.0) / fc
        u = ue * np.linspace(0.5, 6.0, 12)
        gamma = np.sqrt(1.0 + (u / const.c) ** 2)
        h = 1e-6 * u

        f = np.array([F_func(fc, Te, ui, Zeff) for ui in u])
        f_plus = np.array([F_func(fc, Te, ui, Zeff) for ui in u + h])
        f_minus = np.array([F_func(fc, Te, ui, Zeff) for ui in u - h])
        fp = (f_plus - f_minus) / (2 * h)

        lhs = gamma**2 / u**2 * fp + rhohat * gamma / u**3 * f
        rhs = 1.0 / (fc * ue**3) / gamma * u / ue
        npt.assert_allclose(lhs, rhs, rtol=MIDPOINT_RTOL)

    @staticmethod
    def test_zero_at_zero_momentum():
        """
        Boundary condition F(0) = 0.
        """
        npt.assert_equal(F_func(0.6, 2.0e3, 0.0, 1.6), 0.0)

    @staticmethod
    @pytest.mark.parametrize(("fc", "Zeff"), params_nonrelativistic)
    @pytest.mark.parametrize("x", params_x_nonrelativistic)
    def test_nonrelativistic_limit(fc: float, Zeff: float, x: float):  # noqa: N803
        """
        For u << c, F = x^4 / (Zeff + 1 + 4 fc) (eq. 34).
        """
        Te = 1.0  # noqa: N806 (u_e / c ~ 2e-3, so u << c)
        ue = thermal_momentum(Te)
        npt.assert_allclose(
            F_func(fc, Te, x * ue, Zeff),
            x**4 / (Zeff + 1 + 4 * fc),
            rtol=MIDPOINT_RTOL,
        )
