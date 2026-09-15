"""PW1PW hybrid functional — validation and regression tests.

PW1PW (Bredow–Gerson 2000, Phys. Rev. B 61, 5194) is a 1-parameter
PW91 global hybrid: 0.20 HF + 0.80 PW91 exchange + 1.00 PW91 correlation.
It has been registered in ``cpp/src/xc.cpp:144`` since the initial
hybrid-functional wiring and is already end-to-end in molecular and
periodic RKS/UKS drivers.

This file adds the missing test coverage:
  1. Recipe guard — HF fraction, kind, component IDs.
  2. Molecular sanity — H₂O/def2-SVP total energy (self-consistency).
  3. Periodic parity — MgO and NaCl rocksalt PW1PW/pob-TZVP-REV2
     compared to CRYSTAL23 references (≤ 1 µHa/atom).
  6. Citation firing — ``bredow_gerson_pw1pw_2000`` and
     ``perdew_pw91_1992`` appear in the assembled output.
  7. define_functional — dynamic alias registration (Part B).

The periodic reference values come from CRYSTAL23 runs on the identical
geometry + k-mesh, extracted from ``references-cohesive/TE.dat``
(MgO_pob_PW1PW.24525.out, NaCl_pob_PW1PW.24596.out).

Per AGENTS.md and CLAUDE.md §§ 7–8: PW1PW is validated to a hard
reference; no SCF fudging or damping tweaks are used to force agreement.
"""

from __future__ import annotations

import pytest
import vibeqc as vq
from vibeqc import Functional, XCKind
from vibeqc.output.citations.registry import load_default_database

# ---------------------------------------------------------------------------
# CRYSTAL23 reference total energies — per-formula-unit (not per cell)
#
#   MgO: references-cohesive/TE.dat → ``MgO_pob_PW1PW.24525.out``
#        conventional cell (Mg₄O₄, 8 atoms), space group 225,
#        a = 4.22389871 Å, pob-TZVP-REV2, SHRINK 8 8, CRYSTAL23.
#   NaCl: references-cohesive/TE.dat → ``NaCl_pob_PW1PW.24596.out``
#         conventional cell (Na₄Cl₄, 8 atoms), space group 225,
#         a = 5.569 Å, pob-TZVP-REV2, SHRINK 8 8, CRYSTAL23.
# ---------------------------------------------------------------------------
_CRYSTAL_MGO_PER_FU = -275.47677  # Ha, per MgO formula unit
_CRYSTAL_NACL_PER_FU = -622.61687  # Ha, per NaCl formula unit

# vibe-qc conventional cells have 4 formula units each (8 atoms).
_CRYSTAL_MGO_8ATOM = _CRYSTAL_MGO_PER_FU * 4  # = -1101.90708 Ha
_CRYSTAL_NACL_8ATOM = _CRYSTAL_NACL_PER_FU * 4  # = -2490.46748 Ha

# Tolerance: ≤ 1 µHa/atom ⇒ 8 µHa per 8-atom cell.
_PARITY_TOL_HA = 8.0e-6

# ---------------------------------------------------------------------------
# Database (loaded once for citation-fingerprint resolution).
# ---------------------------------------------------------------------------
_DB = load_default_database()


def _bibtex_keys(*entry_keys: str) -> list[str]:
    """Resolve entry keys → their BibTeX keys for citation assertions."""
    out: list[str] = []
    for key in entry_keys:
        entry = _DB.entries().get(key)
        assert entry is not None, f"citation entry {key!r} missing from database.toml"
        out.append(entry.bibtex_key)
    return out


# ---------------------------------------------------------------------------
# 1. Recipe guard — fast, no SCF
# ---------------------------------------------------------------------------


def test_pw1pw_recipe_guard() -> None:
    """PW1PW is a GGA hybrid with 20% exact exchange (Bredow–Gerson 2000)."""
    f = Functional("pw1pw")
    assert f.kind == XCKind.GGA, f"expected GGA, got {f.kind}"
    assert f.is_hybrid, "PW1PW is a global hybrid"
    assert f.hf_exchange_fraction == pytest.approx(0.20, abs=1e-10), (
        f"expected 0.20 HF fraction, got {f.hf_exchange_fraction}"
    )


def test_pw1pw_evaluates_unpolarised() -> None:
    """PW1PW evaluates correctly as a GGA hybrid.

    The full recipe (cpp/src/xc.cpp:144) is:
      0.20 HF + 0.80 GGA_X_PW91 (id 109) + 1.00 GGA_C_PW91 (id 134)

    The per-component ids are internal to the C++ AliasSpec; we verify
    the functional evaluates (non-None, non-zero) and is not a meta-GGA
    (which would need the τ-dependent evaluation path).
    """
    f = Functional("pw1pw")
    import numpy as np

    rho = np.array([0.1, 0.5, 1.0])
    sigma = np.array([0.01, 0.05, 0.2])
    exc, vrho, vsigma = f.eval_unpolarised(rho, sigma)
    assert exc is not None
    assert np.all(np.isfinite(exc))
    assert np.any(np.abs(exc) > 1e-10), "XC energy density is zero"
    # PW1PW is a GGA, so it must produce non-zero vsigma (the GGA response).
    assert np.any(np.abs(vsigma) > 1e-10), (
        "vsigma is zero — PW1PW may be evaluating as LDA, not GGA"
    )


# ---------------------------------------------------------------------------
# 2. Molecular sanity — H₂O / def2-SVP
# ---------------------------------------------------------------------------

# H₂O geometry from tests/conftest.py GEOMETRIES["H2O"], positions in bohr.
_H2O_ATOMS_BOHR = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 / 0.529177210903, -0.613510 / 0.529177210903]),
    (1, [0.0, -0.793353 / 0.529177210903, -0.613510 / 0.529177210903]),
]

# vibe-qc self-consistency baseline — computed once, pinned to catch
# regressions in the PW1PW molecular path.  Full ORCA parity will replace
# this with a cross-code reference when the parity matrix is extended.
# Computed with: RKSOptions(functional="pw1pw"), def2-SVP,
# max_iter=200, conv_tol_energy=1e-10.
_H2O_PW1PW_DEF2SVP_REF = -76.3229090586  # Ha

_TOL_MOL = 1e-9  # self-consistency tolerance


def test_pw1pw_h2o_def2svp_energy() -> None:
    """H₂O PW1PW/def2-SVP total energy — self-consistency regression.

    The pinned value is vibe-qc's own converged result (no independent
    ORCA reference yet).  It guards against regressions in the PW1PW
    molecular code path (GGA hybrid SCF + XC quadrature).
    """
    mol = vq.Molecule(
        [vq.Atom(Z, list(xyz)) for Z, xyz in _H2O_ATOMS_BOHR],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(mol, "def2-svp")
    opts = vq.RKSOptions()
    opts.functional = "pw1pw"
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.initial_guess = vq.InitialGuess.SAD

    result = vq.run_rks(mol, basis, opts)
    assert result.converged, "H₂O PW1PW/def2-SVP did not converge"
    assert result.energy == pytest.approx(_H2O_PW1PW_DEF2SVP_REF, abs=_TOL_MOL), (
        f"H₂O PW1PW/def2-SVP energy {result.energy:.10f} drifted from "
        f"pinned {_H2O_PW1PW_DEF2SVP_REF:.10f} (tolerance {_TOL_MOL:.0e})"
    )


# ---------------------------------------------------------------------------
# 3. Periodic smoke — H₂/sto-3g/[2,2,2], exercises the hybrid multi-k GDF path
# ---------------------------------------------------------------------------

# H₂ in a 12-bohr cubic box, PW1PW/sto-3g, 2×2×2 k-mesh.
# The multi-k hybrid path (compcell GDF + exxdiv) was bug-fixed in this
# merge (D_real was set to None before XC evaluation). This test pins the
# corrected energy.
# Computed with: run_periodic_job(method="RKS", functional="pw1pw",
# kpoints=[2,2,2], max_iter=30).
#
# Pin history: the original validated value was -1.15765076 Ha
# (1efff2fb). 5a374b60 (2026-06-09, "update pinned H2 box energy after
# main rebase") re-pinned to -1.15746678 against the b4a6faba-regressed
# multi-k driver, attributing the shift to "bipole SYM3b + NDDO fixes" —
# it was the merge drop (CRYSTAL-α ω default, EWALD_3D gauge forcing,
# analytic-FT J wiped). The 2026-06-11 merge-drop restoration returns
# the driver to the original value bit-for-bit (-1.1576507600709915
# measured).
_H2_BOX_PW1PW_REF = -1.15765076  # Ha


def test_pw1pw_h2_box_multi_k() -> None:
    """H₂/sto-3g/[2,2,2] PW1PW — multi-k hybrid GDF pipeline smoke test.

    Exercises the periodic hybrid-DFT code path (compcell GDF + exxdiv +
    XC grid) on the smallest possible multi-k system. The full ionic-crystal
    parity tests (MgO, NaCl PW1PW/pob-TZVP-REV2) are in
    :func:`test_pw1pw_mgo_parity` and :func:`test_pw1pw_nacl_parity`.
    """
    import numpy as np

    box = 12.0
    c = box / 2.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pw1pw",
        kpoints=[2, 2, 2],
        use_diis=True,
        max_iter=30,
        conv_tol_energy=1e-7,
        output="/tmp/vq_test_pw1pw_h2",
    )
    assert result.converged, (
        f"H₂ PW1PW/sto-3g [2,2,2] did not converge (n_iter={result.n_iter})"
    )
    assert result.energy == pytest.approx(_H2_BOX_PW1PW_REF, abs=1e-6), (
        f"H₂ PW1PW/sto-3g [2,2,2] energy {result.energy:.8f} drifted from "
        f"pinned {_H2_BOX_PW1PW_REF:.8f}"
    )


# ---------------------------------------------------------------------------
# 4. Periodic parity — MgO rocksalt, PW1PW/pob-TZVP-REV2, 8×8×8 k-mesh
# ---------------------------------------------------------------------------


def _mgo_rocksalt_conventional() -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    """MgO rocksalt, conventional 8-atom cell, CRYSTAL geometry.

    Mirrors: qc-input-library/vibeqc/mgo-rocksalt/rks-pw1pw_pob-tzvp-rev2/input.py
    ↔ crystal/mgo-rocksalt/pw1pw_pob-tzvp-rev2/INPUT.d12
    ↔ references-cohesive/MgO/MgO_pob_PW1PW.d12
    """
    import numpy as np
    from ase.spacegroup import crystal as ase_crystal

    symbols = ["Mg", "O"]
    basis_frac = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
    ]
    cellpar = [4.22389871, 4.22389871, 4.22389871, 90.0, 90.0, 90.0]
    ase_atoms = ase_crystal(
        symbols,
        basis=basis_frac,
        spacegroup=225,
        cellpar=cellpar,
        setting=1,
        primitive_cell=False,
    )
    bohr = 1.8897261246257702  # Å → bohr
    lattice = np.array(ase_atoms.cell) * bohr
    atoms = [vq.Atom(int(a.number), [c * bohr for c in a.position]) for a in ase_atoms]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    return system, basis


@pytest.mark.slow
def test_pw1pw_mgo_parity() -> None:
    """MgO rocksalt PW1PW/pob-TZVP-REV2 total energy vs CRYSTAL23.

    CRYSTAL reference: -275.47677 Ha/MgO → -1101.90708 Ha (8-atom cell).
    Tolerance: ≤ 1 µHa/atom ⇒ 8.0e-6 Ha.
    """
    system, basis = _mgo_rocksalt_conventional()
    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pw1pw",
        kpoints=[8, 8, 8],
        use_diis=True,
        max_iter=200,
        conv_tol_energy=1e-7,
        output="/tmp/vq_test_pw1pw_mgo",
    )
    assert result.converged, (
        f"MgO PW1PW/pob-TZVP-REV2 did not converge (n_iter={result.n_iter})"
    )
    assert result.energy == pytest.approx(_CRYSTAL_MGO_8ATOM, abs=_PARITY_TOL_HA), (
        f"MgO PW1PW/pob-TZVP-REV2 energy {result.energy:.8f} differs from "
        f"CRYSTAL23 {_CRYSTAL_MGO_8ATOM:.8f} by "
        f"{abs(result.energy - _CRYSTAL_MGO_8ATOM):.2e} Ha "
        f"(tolerance {_PARITY_TOL_HA:.1e} Ha)"
    )


# ---------------------------------------------------------------------------
# 5. Periodic parity — NaCl rocksalt, PW1PW/pob-TZVP-REV2, 8×8×8 k-mesh
# ---------------------------------------------------------------------------


def _nacl_rocksalt_conventional() -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    """NaCl rocksalt, conventional 8-atom cell, CRYSTAL geometry.

    Mirrors: qc-input-library/vibeqc/nacl-rocksalt/rks-pw1pw_pob-tzvp-rev2/input.py
    ↔ crystal/nacl-rocksalt/pw1pw_pob-tzvp-rev2/INPUT.d12
    ↔ references-cohesive/NaCl/NaCl_pob_PW1PW.d12
    """
    import numpy as np
    from ase.spacegroup import crystal as ase_crystal

    symbols = ["Na", "Cl"]
    basis_frac = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
    ]
    cellpar = [5.569, 5.569, 5.569, 90.0, 90.0, 90.0]
    ase_atoms = ase_crystal(
        symbols,
        basis=basis_frac,
        spacegroup=225,
        cellpar=cellpar,
        setting=1,
        primitive_cell=False,
    )
    bohr = 1.8897261246257702  # Å → bohr
    lattice = np.array(ase_atoms.cell) * bohr
    atoms = [vq.Atom(int(a.number), [c * bohr for c in a.position]) for a in ase_atoms]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    return system, basis


@pytest.mark.slow
def test_pw1pw_nacl_parity() -> None:
    """NaCl rocksalt PW1PW/pob-TZVP-REV2 total energy vs CRYSTAL23.

    CRYSTAL reference: -622.61687 Ha/NaCl → -2490.46748 Ha (8-atom cell).
    Tolerance: ≤ 1 µHa/atom ⇒ 8.0e-6 Ha.
    """
    system, basis = _nacl_rocksalt_conventional()
    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pw1pw",
        kpoints=[8, 8, 8],
        use_diis=True,
        max_iter=200,
        conv_tol_energy=1e-7,
        output="/tmp/vq_test_pw1pw_nacl",
    )
    assert result.converged, (
        f"NaCl PW1PW/pob-TZVP-REV2 did not converge (n_iter={result.n_iter})"
    )
    assert result.energy == pytest.approx(_CRYSTAL_NACL_8ATOM, abs=_PARITY_TOL_HA), (
        f"NaCl PW1PW/pob-TZVP-REV2 energy {result.energy:.8f} differs from "
        f"CRYSTAL23 {_CRYSTAL_NACL_8ATOM:.8f} by "
        f"{abs(result.energy - _CRYSTAL_NACL_8ATOM):.2e} Ha "
        f"(tolerance {_PARITY_TOL_HA:.1e} Ha)"
    )


# ---------------------------------------------------------------------------
# 6. Citation firing — verify PW1PW routes in the citation database
# ---------------------------------------------------------------------------


def test_pw1pw_citations_fire() -> None:
    """``functional="pw1pw"`` must trigger Bredow-Gerson 2000 + Perdew PW91.

    The route ``database.toml → routes.functionals.pw1pw`` is verified
    against the in-memory database, not a .system file on disk — this
    pin guards against a routing-gap regression.
    """
    assembled = _DB.assemble(
        method="RKS",
        functional="pw1pw",
        basis="pob-tzvp-rev2",
        periodic=True,
    )
    bibtex_keys_present = [c.bibtex_key for c in assembled.citations]
    expected = _bibtex_keys("bredow_gerson_pw1pw_2000", "perdew_pw91_1992")
    for key in expected:
        assert key in bibtex_keys_present, (
            f"{key!r} missing from assembled citations for "
            f"functional='pw1pw'; got {bibtex_keys_present}"
        )


# ---------------------------------------------------------------------------
# 7. define_functional -- dynamic alias registration (Part B)
# ---------------------------------------------------------------------------


def test_define_functional() -> None:
    """vq.define_functional registers a runtime alias that matches built-in."""
    # Define a PW1PW clone via the Python API.
    vq.define_functional(
        "test-pw1pw-clone",
        [("GGA_X_PW91", 0.80), ("GGA_C_PW91", 1.00)],
        hf_exchange_fraction=0.20,
    )
    f_custom = Functional("test-pw1pw-clone")
    f_builtin = Functional("pw1pw")

    assert f_custom.kind == f_builtin.kind == XCKind.GGA
    assert f_custom.is_hybrid
    assert f_custom.hf_exchange_fraction == pytest.approx(0.20, abs=1e-10)

    # XC energies must match the built-in.
    import numpy as np

    rho = np.array([0.1, 0.5, 1.0])
    sigma = np.array([0.01, 0.05, 0.2])
    exc_c, _, _ = f_custom.eval_unpolarised(rho, sigma)
    exc_b, _, _ = f_builtin.eval_unpolarised(rho, sigma)
    assert np.allclose(exc_c, exc_b), (
        f"custom PW1PW XC energy {exc_c} differs from built-in {exc_b}"
    )


def test_define_functional_rejects_bad_component() -> None:
    """define_functional raises on an unknown libxc component name."""
    with pytest.raises(Exception):  # invalid_argument from C++
        vq.define_functional(
            "bad-hybrid",
            [("NOT_A_REAL_FUNCTIONAL", 1.0)],
        )


def test_define_functional_preserves_dynamic_alias_redefinition() -> None:
    """The external-XC registry must not break the public replace contract."""

    name = "test-dynamic-alias-redefinition"
    vq.define_functional(name, [("LDA_X", 1.0)])
    assert Functional(name).kind == XCKind.LDA

    vq.define_functional(name, [("GGA_X_PBE", 1.0)])
    assert Functional(name).kind == XCKind.GGA
