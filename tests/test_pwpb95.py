"""PWPB95 (Goerigk & Grimme, *J. Chem. Theory Comput.* **7**, 291
(2011)) — the third double hybrid in vibe-qc and the first *meta-GGA*
double hybrid, riding the SOS-MP2 surface (c_ss = 0).

Pins:

  1. ``Functional("pwpb95")`` resolves to the published recipe:
     0.50·HF + 0.50·PW6-modified-mPW91 exchange, 0.731·B95 correlation
     for the SCF piece, c_os = 0.269 / c_ss = 0.0 for the MP2
     correction. PWPB95 reparametrises both semilocal libxc components
     (mPW91 ``_bt`` / ``_alpha`` / ``_expo``, B95 ``_css`` / ``_copp``)
     via the per-component external-parameter override machinery.
  2. The functional is a meta-GGA (``XCKind.MGGA``) — B95 correlation
     is τ-dependent — so the SCF step runs the τ-dependent KS path.
  3. ``run_pwpb95`` orchestrates the SCF + MP2 dispatch and returns a
     :class:`DoubleHybridResult`; it is a thin shim over
     :func:`run_double_hybrid`, exactly like ``run_b2plyp`` /
     ``run_dsd_pbep86``.
  4. The combined total agrees with ORCA 6.1 ``! PWPB95`` to grid
     accuracy on H2O / cc-pVDZ.
  5. PWPB95 is *spin-opposite-scaled*: the MP2 correction is
     0.269·e_os with the same-spin component dropped (c_ss = 0).

The cross-code reference is ORCA rather than PySCF (used by the
B2PLYP / DSD-PBEP86 tests): PWPB95 mixes *reparametrised* libxc
components, which PySCF's xc-string parser cannot express, so a
hand-rolled PySCF recipe would silently use stock mPW91 / B95 and not
be a valid reference.
"""

from __future__ import annotations

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    DoubleHybridResult,
    Functional,
    Molecule,
    XCKind,
    run_double_hybrid,
    run_pwpb95,
)

from .conftest import ANGSTROM_TO_BOHR, GEOMETRIES


PWPB95_HF = 0.50
PWPB95_C_OS = 0.269
PWPB95_C_SS = 0.0

# ORCA 6.1 reference — ``! PWPB95 cc-pVDZ NoFrozenCore NORI TightSCF
# Bohrs`` with ``%mp2 RI false end`` (fully conventional, all-electron),
# geometry = tests/conftest.py GEOMETRIES["H2O"] in bohr. The MP2
# correlation energy ORCA prints is already SOS-scaled (0.269·e_os).
ORCA_PWPB95_SCF = -76.30774555965722
ORCA_PWPB95_MP2_SCALED = -0.050663754
ORCA_PWPB95_TOTAL = -76.358409314121


def atoms_to_mol_basis(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    return mol, basis


# ---------------------------------------------------------------------
# Functional resolver.
# ---------------------------------------------------------------------

def test_pwpb95_functional_resolves():
    """``Functional("pwpb95")`` carries the SCF hybrid mix and the
    SOS-MP2-correction coefficients (Goerigk-Grimme 2011)."""
    f = Functional("pwpb95")
    assert f.hf_exchange_fraction == pytest.approx(PWPB95_HF, abs=1e-12)
    assert f.is_hybrid is True
    assert f.is_double_hybrid is True
    assert f.mp2_c_os == pytest.approx(PWPB95_C_OS, abs=1e-12)
    assert f.mp2_c_ss == pytest.approx(PWPB95_C_SS, abs=1e-12)
    assert f.is_range_separated is False


def test_pwpb95_is_meta_gga_double_hybrid():
    """PWPB95 is the first *meta-GGA* double hybrid: B95 correlation is
    τ-dependent, so the composite resolves to XCKind.MGGA (unlike the
    GGA double hybrids B2PLYP / DSD-PBEP86)."""
    assert Functional("pwpb95").kind == XCKind.MGGA
    assert Functional("b2plyp").kind == XCKind.GGA
    assert Functional("dsd-pbep86").kind == XCKind.GGA


# ---------------------------------------------------------------------
# Cross-code parity against ORCA 6.1 ! PWPB95.
# ---------------------------------------------------------------------

def test_run_pwpb95_matches_orca_h2o_ccpvdz():
    """End-to-end parity against ORCA 6.1 ``! PWPB95`` on H2O / cc-pVDZ.

    Both sides run fully conventional (no density fitting) and
    all-electron, so the only cross-code gap is the XC integration
    grid: vibe-qc lands ~1e-6 Ha from ORCA on the SCF piece, the MP2
    piece, and the combined total. The 5e-5 tolerance is ~45× the
    observed gap — loose enough to absorb grid / platform variation,
    tight enough that any recipe error (a wrong reparametrised
    exchange / correlation external parameter, HF fraction, or MP2
    coefficient would shift the energy by milliHartree or more) fails
    the test."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    result = run_pwpb95(mol, basis, density_fit=False,
                        density_fit_mp2=False)

    d_scf = result.rks.energy - ORCA_PWPB95_SCF
    d_mp2 = result.mp2.e_correlation - ORCA_PWPB95_MP2_SCALED
    d_total = result.e_total - ORCA_PWPB95_TOTAL

    assert abs(d_scf) < 5e-5, (
        f"PWPB95 SCF piece vs ORCA = {d_scf:+.3e} Ha "
        f"(vibeqc = {result.rks.energy:.10f}, ORCA = {ORCA_PWPB95_SCF:.10f})"
    )
    assert abs(d_mp2) < 5e-5, (
        f"PWPB95 SOS-MP2 correction vs ORCA = {d_mp2:+.3e} Ha "
        f"(vibeqc = {result.mp2.e_correlation:.10f}, "
        f"ORCA = {ORCA_PWPB95_MP2_SCALED:.10f})"
    )
    assert abs(d_total) < 5e-5, (
        f"PWPB95 total vs ORCA = {d_total:+.3e} Ha "
        f"(vibeqc = {result.e_total:.10f}, ORCA = {ORCA_PWPB95_TOTAL:.10f})"
    )


# ---------------------------------------------------------------------
# Dispatcher internals.
# ---------------------------------------------------------------------

def test_run_pwpb95_total_equals_sum_of_parts():
    """e_total = rks.energy + mp2.e_correlation."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    result = run_pwpb95(mol, basis, density_fit_mp2=True,
                        aux_basis_mp2="cc-pvdz-ri")
    assert isinstance(result, DoubleHybridResult)
    assert result.functional == "pwpb95"
    assert result.e_total == pytest.approx(
        result.rks.energy + result.mp2.e_correlation, abs=1e-14)


def test_run_pwpb95_is_spin_opposite_scaled():
    """PWPB95 is the SOS member of the double-hybrid line: the MP2
    correction is 0.269·e_os exactly, with the same-spin component
    dropped (c_ss = 0). The unscaled same-spin energy is still
    computed and reported on the result — it just does not enter the
    correction."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    result = run_pwpb95(mol, basis, density_fit_mp2=True,
                        aux_basis_mp2="cc-pvdz-ri")
    # e_correlation depends only on the opposite-spin component.
    assert result.mp2.e_correlation == pytest.approx(
        PWPB95_C_OS * result.mp2.e_os, rel=1e-14)
    # The same-spin component is bound (negative) but contributes zero.
    assert result.mp2.e_ss < 0.0
    assert result.mp2.e_correlation == pytest.approx(
        PWPB95_C_OS * result.mp2.e_os + PWPB95_C_SS * result.mp2.e_ss,
        rel=1e-14)


def test_run_pwpb95_via_run_double_hybrid():
    """run_pwpb95(mol, basis) and run_double_hybrid(mol, basis,
    "pwpb95") return identical results — the named wrapper is a thin
    shim over the generic dispatcher."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    direct = run_pwpb95(mol, basis, density_fit_mp2=True,
                        aux_basis_mp2="cc-pvdz-ri")
    via_generic = run_double_hybrid(
        mol, basis, "pwpb95",
        density_fit_mp2=True, aux_basis_mp2="cc-pvdz-ri")
    assert direct.e_total == pytest.approx(via_generic.e_total, abs=1e-14)
    assert direct.rks.energy == pytest.approx(
        via_generic.rks.energy, abs=1e-14)
    assert direct.mp2.e_correlation == pytest.approx(
        via_generic.mp2.e_correlation, abs=1e-14)


def test_run_pwpb95_open_shell_roks_mgga():
    """Open-shell PWPB95 exercises the ROKS meta-GGA SCF half before the
    semicanonical ROHF-MP2 doubles correction."""
    oh = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
        ],
        multiplicity=2,
    )
    basis = BasisSet(oh, "cc-pvdz")

    result = run_pwpb95(
        oh,
        basis,
        density_fit=False,
        density_fit_mp2=True,
        aux_basis_mp2="cc-pvdz-ri",
    )

    assert isinstance(result, DoubleHybridResult)
    assert result.rks.converged
    assert result.rks.functional == "pwpb95"
    assert result.rks.s_squared == pytest.approx(0.75, abs=1e-12)
    assert result.mp2.e_correlation == pytest.approx(
        PWPB95_C_OS * result.mp2.e_os + PWPB95_C_SS * result.mp2.e_ss,
        rel=1e-12,
    )
    assert result.e_total == pytest.approx(
        result.rks.energy + result.mp2.e_correlation, abs=1e-12
    )
