"""RHF end-to-end cross-checks against independent reference programs.

These tests pin vibe-qc's total and orbital energies to within 1e-10 Hartree
of PySCF for H2, H2O, and CH4 across STO-3G and 6-31G*. Any regression in
integrals, Fock build, or SCF convergence shows up as a failing test. The
tetrazine multiroot regression additionally pins root identity against an
out-of-process ORCA calculation.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, RHFOptions, run_job, run_rhf

from .conftest import (
    GEOMETRIES,
    run_pyscf_rhf,
    run_vibeqc_rhf,
)


# (molecule_key, basis_name)
SCF_CASES = [
    ("H2",  "sto-3g"),
    ("H2",  "6-31g*"),
    ("H2O", "sto-3g"),
    ("H2O", "6-31g*"),
    ("CH4", "sto-3g"),
    ("CH4", "6-31g*"),
]

ANGSTROM_TO_BOHR = 1.8897259886


@pytest.mark.parametrize(
    "mol_key,basis_name",
    SCF_CASES,
    ids=[f"{m}-{b}" for m, b in SCF_CASES],
)
def test_total_energy_matches_pyscf(mol_key, basis_name, tight_rhf_opts):
    atoms = GEOMETRIES[mol_key]
    result = run_vibeqc_rhf(atoms, basis_name, tight_rhf_opts)
    ref_energy, _ = run_pyscf_rhf(atoms, basis_name)

    assert result.converged, (
        f"{mol_key}/{basis_name}: vibeqc SCF did not converge "
        f"(last E = {result.energy}, iter = {result.n_iter})"
    )
    diff = result.energy - ref_energy
    assert abs(diff) < 1e-10, (
        f"{mol_key}/{basis_name}: E_vibeqc = {result.energy:.12f}, "
        f"E_pyscf = {ref_energy:.12f}, diff = {diff:+.2e}"
    )


@pytest.mark.parametrize(
    "mol_key,basis_name",
    SCF_CASES,
    ids=[f"{m}-{b}" for m, b in SCF_CASES],
)
def test_mo_energies_match_pyscf(mol_key, basis_name, tight_rhf_opts):
    atoms = GEOMETRIES[mol_key]
    result = run_vibeqc_rhf(atoms, basis_name, tight_rhf_opts)
    _, ref_mo = run_pyscf_rhf(atoms, basis_name)

    vibeqc_mo = np.array(result.mo_energies)
    # Sort both to be invariant to arbitrary degenerate-orbital orderings
    # (neither should matter in practice, but being defensive).
    np.testing.assert_allclose(
        np.sort(vibeqc_mo),
        np.sort(ref_mo),
        atol=1e-9,
        err_msg=f"MO energies differ for {mol_key}/{basis_name}",
    )


def test_tetrazine_ccpvdz_default_matches_orca_hueckel_root():
    """Issue 12 is a root-selection mismatch, not an RHF energy defect.

    Matched ORCA 6.1 calculations find two RHF stationary points on this
    system: the lower root at -294.0434733 Ha and one 49.19 mHa above it
    at -293.9942834 Ha, which ORCA's HCORE run stays in. An offline
    comparison of the complete occupied spaces gave a minimum singular
    value of 0.99999999995. The orbital spectrum below is the
    redistribution-free numerical pin for that independently generated
    artifact: all 21 occupied levels plus the first 11 virtual levels.

    Since the EDIIS energy-model convention fix (2026-08-28) vibe-qc's
    HCORE run no longer stays in the upper root: the default ``ediis_diis``
    accelerator descends past it to the same lower root HUECKEL and PATOM
    find (measured ``||D_hcore - D_hueckel|| = 7.5e-07``,
    ``max|dEps| = 1.9e-08`` over the pinned 32 levels, ``dE = 1.0e-12``).
    That is what EDIIS is for -- Kudin, Scuseria & Cances, J. Chem. Phys.
    116, 8255 (2002), p. 8258: the energy-minimisation driven algorithm
    "is able to reach the lowest minimum among those available from
    different SCF cycles", with better odds from a bad guess than a good
    one. Before the fix the QP minimised a rescaled proxy that was exact
    only at the hull vertices, so the extrapolation could not descend out
    of the upper basin.

    The upper root is still reachable and is pinned below under plain
    Pulay ``DIIS`` from the same HCORE guess, so the cross-code
    observation this test was written for is not lost.
    """
    from vibeqc import InitialGuess, RHFOptions, SCFAccelerator

    a2b = 1.0 / 0.529177210903
    atoms_angstrom = [
        (7, [0.00, 1.20, 0.00]),
        (7, [1.20, 0.00, 0.00]),
        (7, [0.00, -1.20, 0.00]),
        (7, [-1.20, 0.00, 0.00]),
        (6, [1.20, 1.20, 0.00]),
        (6, [-1.20, 1.20, 0.00]),
        (1, [2.07, 2.07, 0.00]),
        (1, [-2.07, 2.07, 0.00]),
    ]
    molecule = Molecule([
        Atom(z, [component * a2b for component in xyz])
        for z, xyz in atoms_angstrom
    ])
    basis = BasisSet(molecule, "cc-pvdz")

    def _options(guess=None, accelerator=None):
        options = RHFOptions()
        options.density_fit = False
        options.cosx = False
        options.max_iter = 120
        options.conv_tol_energy = 1e-8
        options.conv_tol_grad = 1e-6
        if guess is not None:
            options.initial_guess = guess
        if accelerator is not None:
            options.scf_accelerator = accelerator
        return options

    default_options = _options()
    assert default_options.initial_guess == InitialGuess.AUTO
    default = run_rhf(molecule, basis, default_options)
    assert default.guess_selection.effective == InitialGuess.PATOM
    hueckel = run_rhf(molecule, basis, _options(InitialGuess.HUECKEL))
    hcore = run_rhf(molecule, basis, _options(InitialGuess.HCORE))
    # Same HCORE guess, plain Pulay DIIS: the accelerator that carries no
    # energy model still stays in ORCA's upper root.
    hcore_diis = run_rhf(
        molecule,
        basis,
        _options(InitialGuess.HCORE, SCFAccelerator.DIIS),
    )
    # Pure EDIIS must also finish after the global simplex repair; the
    # convex-only QP reached this energy but stalled at the 120-cycle cap.
    hcore_ediis = run_rhf(
        molecule,
        basis,
        _options(InitialGuess.HCORE, SCFAccelerator.EDIIS),
    )

    assert default.converged and hueckel.converged and hcore.converged
    assert hcore_diis.converged and hcore_ediis.converged
    assert molecule.n_electrons() == 42  # 21 doubly occupied orbitals
    assert default.energy == pytest.approx(-294.043473304177, abs=1e-8)
    assert hueckel.energy == pytest.approx(-294.043473304177, abs=1e-8)
    # The default accelerator now descends to the lower root from HCORE too.
    assert hcore.energy == pytest.approx(hueckel.energy, abs=1e-8)
    assert hcore_ediis.energy == pytest.approx(hueckel.energy, abs=1e-8)
    assert np.linalg.norm(default.density - hueckel.density) < 1e-5
    assert np.linalg.norm(hueckel.density - hcore.density) < 1e-5
    # The upper root ORCA's HCORE run finds, still reproduced under DIIS.
    assert hcore_diis.energy == pytest.approx(-293.994283405219, abs=1e-8)
    assert (hcore_diis.energy - hueckel.energy) * 1000 == pytest.approx(
        49.189899, abs=1e-5
    )
    assert np.linalg.norm(hueckel.density - hcore_diis.density) == pytest.approx(
        2.95219407, abs=1e-5
    )

    orca_hueckel_mo_energies = np.array([
        -15.691967, -15.661468, -15.573628, -15.573586,
        -11.370479, -11.370415, -1.511901, -1.335124,
        -1.135338, -0.986258, -0.925953, -0.787174,
        -0.734942, -0.733932, -0.612760, -0.529368,
        -0.511622, -0.500183, -0.451579, -0.301163,
        -0.299465, 0.045885, 0.145846, 0.152585,
        0.180594, 0.218558, 0.230311, 0.245436,
        0.353284, 0.384600, 0.396408, 0.471133,
    ])
    np.testing.assert_allclose(
        np.asarray(hueckel.mo_energies)[:32],
        orca_hueckel_mo_energies,
        rtol=0.0,
        atol=5e-6,
    )


def test_rhf_rejects_odd_electron_count(tight_rhf_opts):
    # Guard-rail test: open-shell systems (here: Li doublet, 3 electrons)
    # physically require UHF or ROHF. run_rhf() must refuse them with a
    # clear error rather than silently producing a garbage number.
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0])], charge=0, multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="even electron count"):
        run_rhf(mol, basis, tight_rhf_opts)


def test_rhf_rejects_nonunit_multiplicity(tight_rhf_opts):
    # H2 triplet is unusual but syntactically valid; RHF (multiplicity=1)
    # should reject it.
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=3,
    )
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="multiplicity"):
        run_rhf(mol, basis, tight_rhf_opts)


def test_rhf_rejects_damping_out_of_range(tight_rhf_opts):
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    tight_rhf_opts.damping = 1.0  # must be strictly < 1
    with pytest.raises(ValueError, match="damping"):
        run_rhf(mol, basis, tight_rhf_opts)


PORPHINE_D2H_ATOMS_ANGSTROM = [
    (7, [0.000, 2.050, 0.000]),
    (7, [2.050, 0.000, 0.000]),
    (7, [0.000, -2.050, 0.000]),
    (7, [-2.050, 0.000, 0.000]),
    (6, [0.000, 3.400, 0.000]),
    (6, [3.400, 0.000, 0.000]),
    (6, [0.000, -3.400, 0.000]),
    (6, [-3.400, 0.000, 0.000]),
    (6, [1.100, 2.050, 0.000]),
    (6, [-1.100, 2.050, 0.000]),
    (6, [2.050, 1.100, 0.000]),
    (6, [2.050, -1.100, 0.000]),
    (6, [1.100, -2.050, 0.000]),
    (6, [-1.100, -2.050, 0.000]),
    (6, [-2.050, 1.100, 0.000]),
    (6, [-2.050, -1.100, 0.000]),
    (6, [2.350, 1.350, 0.000]),
    (6, [1.350, 2.350, 0.000]),
    (6, [-1.350, 2.350, 0.000]),
    (6, [-2.350, 1.350, 0.000]),
    (1, [0.000, 4.450, 0.000]),
    (1, [4.450, 0.000, 0.000]),
    (1, [0.000, -4.450, 0.000]),
    (1, [-4.450, 0.000, 0.000]),
    (1, [3.350, 1.800, 0.000]),
    (1, [1.800, 3.350, 0.000]),
    (1, [-1.800, 3.350, 0.000]),
    (1, [-3.350, 1.800, 0.000]),
    (1, [3.350, -1.800, 0.000]),
    (1, [1.800, -3.350, 0.000]),
    (1, [-1.800, -3.350, 0.000]),
    (1, [-3.350, -1.800, 0.000]),
    (1, [1.100, -1.100, 0.000]),
    (1, [-1.100, 1.100, 0.000]),
    (1, [1.100, 1.100, 0.000]),
    (1, [-1.100, -1.100, 0.000]),
]


def test_porphine_rhf_sto3g_diis_zero_damping_converges():
    """Current-main reproducer for a stale porphine RHF convergence report."""
    from vibeqc import RHFOptions

    mol = Molecule(
        [
            Atom(z, [coord * ANGSTROM_TO_BOHR for coord in xyz])
            for z, xyz in PORPHINE_D2H_ATOMS_ANGSTROM
        ]
    )
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-8

    result = run_rhf(mol, basis, opts)

    assert result.converged
    assert result.n_iter <= 120
    assert result.energy == pytest.approx(-773.7788816750447, abs=1e-8)


def _run_scf(atoms, basis_name, *, use_diis, damping, tol_e=1e-10, tol_g=1e-8):
    from vibeqc import RHFOptions
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    opts = RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = tol_e
    opts.conv_tol_grad = tol_g
    opts.damping = damping
    # These regression guards compare DIIS against *static* damping; pin
    # dynamic_damping off (it now defaults on) so the damping baseline is the
    # classic static-0.5 path the 2x-speedup guard expects.
    opts.dynamic_damping = False
    opts.use_diis = use_diis
    return run_rhf(mol, BasisSet(mol, basis_name), opts)


def test_diis_gives_same_energy_as_damping():
    """DIIS is an accelerator, not a different method — the converged energy
    must match plain damping at full precision."""
    atoms = GEOMETRIES["H2O"]
    r_damp = _run_scf(atoms, "6-31g*", use_diis=False, damping=0.5)
    r_diis = _run_scf(atoms, "6-31g*", use_diis=True, damping=0.0)
    assert r_damp.converged
    assert r_diis.converged
    assert abs(r_damp.energy - r_diis.energy) < 1e-9


def test_diis_converges_faster_than_damping():
    """Regression guard: DIIS should beat plain damping by at least 2x on
    a realistic test case. H2O/6-31G* typically converges in ~14 iters
    with DIIS vs ~55 with damping=0.5."""
    atoms = GEOMETRIES["H2O"]
    r_damp = _run_scf(atoms, "6-31g*", use_diis=False, damping=0.5)
    r_diis = _run_scf(atoms, "6-31g*", use_diis=True, damping=0.0)
    assert r_diis.n_iter * 2 <= r_damp.n_iter, (
        f"DIIS ({r_diis.n_iter} iter) is not >= 2x faster than "
        f"damping ({r_damp.n_iter} iter)"
    )


# ---------------------------------------------------------------------------
# Issue #144 — restricted-stability VERDICT surface (decision-neutral
# prerequisite). The RHF driver now reports the lowest orbital-rotation
# Hessian eigenvalue at the converged restricted solution (Seeger-Pople
# 1977 singlet + triplet sectors) and warns loudly when it is negative.
# Verdict only: no rotation, escape, or promotion — what to do with a
# negative verdict is the maintainer decision in
# agentic-loop/asks/ask-scf144-restricted-stability-contract-2026-08-27.md.
# ---------------------------------------------------------------------------


def _twisted_ethene_90deg():
    """90-degree twisted ethene, textbook geometry (r(C=C) = 1.33 A,
    r(C-H) = 1.08 A, HCH = 120 deg, bohr). The two CH2 planes are
    perpendicular, the classic biradical twist: the default RHF path
    converges to a restricted root that is externally (triplet) unstable.
    Constructed witness — not a library deck; the provenance is the
    geometry constants above."""
    ang = 1.8897259886
    rcc = 1.33 * ang
    rch = 1.08 * ang
    half = rcc / 2.0
    dx = rch * np.cos(np.radians(30.0))
    dy = rch * np.sin(np.radians(30.0))
    return Molecule(
        [
            Atom(6, [-half, 0.0, 0.0]),
            Atom(6, [half, 0.0, 0.0]),
            Atom(1, [-half + dx, dy, 0.0]),
            Atom(1, [-half + dx, -dy, 0.0]),
            Atom(1, [half - dx, 0.0, dy]),
            Atom(1, [half - dx, 0.0, -dy]),
        ],
        charge=0,
        multiplicity=1,
    )


def test_restricted_stability_verdict_flags_unstable_root_and_warns(tmp_path):
    """#144: a converged-but-unstable RHF root is no longer silent.

    The default path on 90-deg twisted ethene lands an externally
    unstable restricted solution; the verdict surface reports a negative
    lowest Hessian eigenvalue, marks internal_instability, and the .out
    carries the loud SCF INTERNAL INSTABILITY warning. At the parent the
    run was silent — converged=True with no signal — which is the defect.
    """
    mol = _twisted_ethene_90deg()
    stem = tmp_path / "twist_on"
    res = run_job(mol, basis="cc-pvdz", method="rhf", output=stem)

    assert res.converged
    # Behavioural discriminator FIRST: at the parent this same run
    # completed with converged=True and no warning at all — that
    # silence is the defect. The emitted artifact is the observable.
    out = stem.with_suffix(".out").read_text()
    assert "SCF INTERNAL INSTABILITY" in out
    assert "EXCITED SCF solution" in out
    # The new verdict surface (missing-symbol at the parent).
    assert res.stability_checked is True
    assert res.stability_analysis_converged is True
    assert res.internal_instability is True
    assert res.stability_eigenvalue < -1e-4


def test_restricted_stability_opt_out_route_no_verdict_same_energy(tmp_path):
    """#144 negative control (L125): the SAME route with the feature off.

    stability_check=False returns the identical excited root but records
    no verdict and emits no warning — the verdict surface itself is what
    changed, not the SCF numerics.
    """
    mol = _twisted_ethene_90deg()
    opts = RHFOptions()
    opts.stability_check = False
    stem = tmp_path / "twist_off"
    res = run_job(
        mol, basis="cc-pvdz", method="rhf", rhf_options=opts, output=stem
    )

    assert res.converged
    out = stem.with_suffix(".out").read_text()
    assert "SCF INTERNAL INSTABILITY" not in out
    assert res.stability_checked is False
    # Same stationary point as the default path (bit-identical energy).
    on_stem = tmp_path / "twist_on"
    res_on = run_job(mol, basis="cc-pvdz", method="rhf", output=on_stem)
    assert res.energy == res_on.energy


def test_restricted_stability_stable_h2_exactly_stable(tmp_path):
    """#144 positive control: a stable solution is exactly stable.

    H2/STO-3G at equilibrium converges to a minimum; the verdict must be
    exactly non-unstable (internal_instability is exactly False, the
    eigenvalue is positive) and no warning may fire.
    """
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    stem = tmp_path / "h2_stable"
    res = run_job(mol, basis="sto-3g", method="rhf", output=stem)

    assert res.converged
    out = stem.with_suffix(".out").read_text()
    assert "SCF INTERNAL INSTABILITY" not in out
    assert res.stability_checked is True
    assert res.stability_analysis_converged is True
    assert res.internal_instability is False
    assert res.stability_eigenvalue > 0.0
