"""Tests for the MSINDO NDDO ("improved integrals") two-centre kernels.

NDDO keeps the two-centre multipole interactions (dipole-monopole, dipole-
dipole, …) that INDO drops, in the Dewar/MNDO point-charge model (B. Voigt,
Theor. Chim. Acta 31, 289 (1973); MSINDO spss_si.f / sppp_si.f / spsp_si.f /
spsp_pi.f, parameters DA/ASP from einzentren.f).

These integrals are MSINDO-specific (built from its own exponents), so the
non-circular ground truth is the *physics limit*: each point-charge integral
must reproduce the exact classical multipole field at large separation.  The
NDDO Fock assembly (nddofockcl.f + spspfockcl.f + the v2core / pseudofock core
terms) and the end-to-end energy parity vs the oracle are the next increment
(docs/user_guide/msindo.md, NDDO mode).
"""

import math

import pytest

from vibeqc.semiempirical.methods import msindo

# 2nd-/3rd-row main-group elements that carry an s-p dipole.
_SP_ELEMENTS = [6, 7, 8, 9, 13, 14, 15, 16, 17]


@pytest.mark.parametrize("z", _SP_ELEMENTS)
def test_da_positive_and_shrinks_with_exponent(z):
    """The dipole length DA is positive and physically O(1) bohr."""
    da = msindo.nddo_da(z)
    assert 0.0 < da < 3.0


def test_da_decreases_across_a_row():
    """Across a row the orbitals contract, so the s-p dipole length shrinks."""
    # 2nd row C > N > O > F ; 3rd row Al > Si > P > S > Cl.
    assert (msindo.nddo_da(6) > msindo.nddo_da(7)
            > msindo.nddo_da(8) > msindo.nddo_da(9))
    assert (msindo.nddo_da(13) > msindo.nddo_da(14) > msindo.nddo_da(15)
            > msindo.nddo_da(16) > msindo.nddo_da(17))


@pytest.mark.parametrize("z,da_ref", [
    (6, 0.831150), (7, 0.741764), (8, 0.637043), (9, 0.594047),
    (13, 1.431086), (16, 0.959086),
])
def test_da_regression_values(z, da_ref):
    """DA from the einzentren.f formula with the validated MUS/MUP exponents."""
    assert msindo.nddo_da(z) == pytest.approx(da_ref, abs=1e-5)


@pytest.mark.parametrize("zi,zj", [(8, 8), (6, 8), (9, 9), (8, 16), (16, 6)])
def test_dipole_monopole_field_limit(zi, zj):
    """spss_si, sppp_si → DA_i / r² (a bare dipole field) as r→∞.

    The dipole-monopole integral is, at long range, the potential of the
    point dipole DA_i sampled by the j monopole.
    """
    r = 1.0e5
    da_i = msindo.nddo_da(zi)
    assert msindo.nddo_spss_si(zi, zj, r) * r**2 == pytest.approx(da_i, abs=1e-6)
    assert msindo.nddo_sppp_si(zi, zj, r) * r**2 == pytest.approx(da_i, abs=1e-6)


@pytest.mark.parametrize("zi,zj", [(8, 8), (6, 8), (9, 9), (16, 8)])
def test_dipole_dipole_field_limit(zi, zj):
    """spsp_si → −2·DA_i·DA_j/r³ (axial) and spsp_pi → +DA_i·DA_j/r³ (perp).

    These are the two independent components of the classical dipole-dipole
    interaction tensor.  Checked at a moderate r (large enough to be
    asymptotic, small enough to avoid the 1/r³ cancellation losing precision).
    """
    r = 400.0
    di, dj = msindo.nddo_da(zi), msindo.nddo_da(zj)
    assert msindo.nddo_spsp_si(zi, zj, r) * r**3 == pytest.approx(-2.0 * di * dj, rel=2e-4)
    assert msindo.nddo_spsp_pi(zi, zj, r) * r**3 == pytest.approx(di * dj, rel=2e-4)


@pytest.mark.parametrize("zi,zj", [(6, 8), (8, 16), (9, 13)])
def test_dipole_dipole_symmetry(zi, zj):
    """(s_i p_i | s_j p_j) is symmetric under i↔j (both carry a dipole)."""
    r = 3.2
    assert msindo.nddo_spsp_si(zi, zj, r) == pytest.approx(msindo.nddo_spsp_si(zj, zi, r), abs=1e-12)
    assert msindo.nddo_spsp_pi(zi, zj, r) == pytest.approx(msindo.nddo_spsp_pi(zj, zi, r), abs=1e-12)


def test_dipole_monopole_not_symmetric():
    """(s_i p_i | s_j s_j) carries the dipole only on i — not i↔j symmetric."""
    r = 3.0
    assert msindo.nddo_spss_si(8, 9, r) != pytest.approx(msindo.nddo_spss_si(9, 8, r), abs=1e-4)


def test_integrals_decay_with_distance():
    """All NDDO multipole integrals are positive-definite in magnitude and decay."""
    near = abs(msindo.nddo_spss_si(8, 8, 2.0))
    far = abs(msindo.nddo_spss_si(8, 8, 6.0))
    assert near > far > 0.0
