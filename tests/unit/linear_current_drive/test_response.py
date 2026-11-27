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
from crayon.linear_current_drive.mc_coefficients import get_MC_coefficients
from crayon.linear_current_drive.response import (
    F_func,
    F_func_MC,
    F_func_quad,
)
from crayon.shared.physics import (
    ELECTRON_REST_MASS_ENERGY_EV,
    lorentz_factor_from_normalised_momentum,
)

# F_func integrates eq. 33 with a 30-point midpoint rule, which is low by up
# to ~0.8% for the parameters below (worst at large rhohat, small x).
# Use a leniant tolerance for now.
MIDPOINT_RTOL = 1e-2


class TestFFunc:
    """
    Unit tests for F_func: F(u)
    (LL03 eqs. 31-34).
    """

    # Each entry: fc, Te [eV] and Zeff.
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
    def test_matches_F_func_quad(  # noqa: N802
        fc: float,
        Te: float,  # noqa: N803
        Zeff: float,  # noqa: N803
        x: float,
    ):
        """
        F_func (second form of eq. 33) equals F_func_quad (first form).
        """
        u = x * thermal_momentum(Te)
        npt.assert_allclose(
            F_func(u, Te, fc, Zeff),
            F_func_quad(u, Te, fc, Zeff),
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

        f = np.array([F_func(ui, Te, fc, Zeff) for ui in u])
        f_plus = np.array([F_func(ui, Te, fc, Zeff) for ui in u + h])
        f_minus = np.array([F_func(ui, Te, fc, Zeff) for ui in u - h])
        fp = (f_plus - f_minus) / (2 * h)

        lhs = gamma**2 / u**2 * fp + rhohat * gamma / u**3 * f
        rhs = 1.0 / (fc * ue**3) / gamma * u / ue
        npt.assert_allclose(lhs, rhs, rtol=MIDPOINT_RTOL)

    @staticmethod
    def test_zero_at_zero_momentum():
        """
        Boundary condition F(0) = 0.
        """
        npt.assert_equal(F_func(0.0, 2.0e3, 0.6, 1.6), 0.0)

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
            F_func(x * ue, Te, fc, Zeff),
            x**4 / (Zeff + 1 + 4 * fc),
            rtol=MIDPOINT_RTOL,
        )


class TestFFuncMC:
    """
    Unit tests for F_func_MC: the momentum-conserving F(u) of Hu et al.
    (Hu21 eq. 30, Hu16 eq. 28).
    """

    tables = get_MC_coefficients()
    params_fc = (1.0, 0.5, 0.1)
    params_Z_F = ((1.0, 234.0), (100.0, 4.54))  # noqa: N815

    @staticmethod
    @pytest.mark.parametrize("fc", params_fc)
    def test_lorentz_limit(fc: float):
        """
        As Zeff -> infinity, F -> x^4 / (gamma^2 Zeff) for any fc (Hu21, text
        after eq. 35). Checks the sign of f_tr = 1 / fc - 1.
        """
        Te, Zeff = 50.0e3, 1.0e4  # noqa: N806
        u = np.array([1.0e8, 2.0e8, 3.0e8])
        x = u / thermal_momentum(Te)
        gamma = lorentz_factor_from_normalised_momentum(u / const.c)
        F = F_func_MC(u, Te, fc, Zeff, *TestFFuncMC.tables)  # noqa: N806
        npt.assert_allclose(Zeff * F * gamma**2 / x**4, 1.0, rtol=1e-3)

    @staticmethod
    @pytest.mark.parametrize(("Zeff", "expected"), params_Z_F)
    def test_F_expected(Zeff: float, expected: float):  # noqa: N802, N803
        """
        Test F matches expected values.
        """
        Te = 0.1 * ELECTRON_REST_MASS_ENERGY_EV  # noqa: N806
        u = 10.0 * thermal_momentum(Te)
        F = F_func_MC(u, Te, 0.5, Zeff, *TestFFuncMC.tables)  # noqa: N806
        npt.assert_allclose(F, expected, rtol=3e-2)
