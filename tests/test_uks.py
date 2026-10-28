"""UKS (unrestricted Kohn-Sham DFT) cross-checks against PySCF."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    GridOptions,
    HubbardSite,
    InitialGuess,
    Molecule,
    RKSOptions,
    UKSOptions,
    compute_overlap,
    cosx_grid_options_for_level,
    run_rks,
    run_uks,
)


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _pyscf_uks(
    atoms_bohr,
    basis_name,
    xc,
    *,
    charge=0,
    spin,
    atom_grid=None,
    prune=True,
    conv_tol_grad=1e-10,
    use_newton=False,
):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.charge = charge
    mol.spin = spin  # 2S, not 2S+1
    mol.verbose = 0
    mol.build()
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 5
    if atom_grid is not None:
        mf.grids.atom_grid = atom_grid
    if prune is not True:
        mf.grids.prune = None if prune is None else prune
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = 200
    if use_newton:
        mf = mf.newton()
        mf.max_cycle = 50
    mf.kernel()
    if not mf.converged and not use_newton:
        mf = mf.newton()
        mf.max_cycle = 50
        mf.kernel()
    assert mf.converged, f"PySCF UKS reference did not converge for {xc}"
    s2, _ = mf.spin_square()
    return mf.e_tot, s2


def _fine_m06_grid() -> GridOptions:
    grid = GridOptions()
    grid.n_radial = 120
    grid.n_theta = 23
    grid.n_phi = 46
    return grid


def _tight_uks_opts() -> UKSOptions:
    opts = UKSOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-10
    # 1e-5 matches PySCF's default; UKS on open shells plateaus in the
    # ~1e-6 gradient regime when the energy is already at grid precision.
    opts.conv_tol_grad = 1e-5
    return opts


def _run_uks(atoms_bohr, basis_name, functional, mult, *, opts=None):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
                   multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    o = opts if opts is not None else _tight_uks_opts()
    o.functional = functional
    return run_uks(mol, basis, o)


# (label, atoms_bohr, basis, charge, multiplicity)
OPEN_SHELL_DFT_CASES = [
    ("H-doublet",  [(1, [0.0, 0.0, 0.0])],                               "sto-3g", 0, 2),
    ("OH-doublet",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],                         "sto-3g", 0, 2),
    ("O2-triplet", [(8, [0, 0, -0.6]), (8, [0, 0, 0.6])],                "sto-3g", 0, 3),
]

FUNCTIONALS = [
    ("LDA",    "lda,vwn"),
    ("PBE",    "pbe,pbe"),
    # vibe-qc's "B3LYP" = ORCA/VWN5 definition; PySCF's matching
    # spelling is "b3lyp5".
    ("B3LYP",  "b3lyp5"),
    # Meta-GGA (τ-dependent) functionals — exercise the polarised
    # eval_polarised_mgga path + per-spin V_τ Fock contribution.
    ("TPSS",   "tpss,tpss"),
    ("TPSSh",  "tpssh"),
    ("M06-L",  "m06_l,m06_l"),
    ("M06-2X", "m06_2x,m06_2x"),
    ("R2SCAN", "r2scan,r2scan"),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult",
    OPEN_SHELL_DFT_CASES,
    ids=[c[0] for c in OPEN_SHELL_DFT_CASES],
)
@pytest.mark.parametrize("vq_name,ps_name", FUNCTIONALS,
                         ids=[c[0] for c in FUNCTIONALS])
def test_uks_matches_pyscf(label, atoms, basis_name, charge, mult,
                           vq_name, ps_name):
    opts = _tight_uks_opts()
    pyscf_kwargs = {}
    if label == "OH-doublet":
        # PySCF's UKS Newton solver reaches a stable OH doublet energy on
        # this compact STO-3G grid but leaves the orbital-gradient norm at
        # the same ~1e-6 plateau as vibe-qc, so use the convergence gate
        # both implementations already exercise below.
        pyscf_kwargs.update(conv_tol_grad=1e-5, use_newton=True)
    if vq_name in {"M06-L", "M06-2X", "R2SCAN"}:
        # The M06 family and R2SCAN are visibly grid-sensitive on these
        # compact open-shell STO-3G cells. Pin both implementations to a
        # matched fine product/Lebedev-size grid rather than weakening the
        # parity tolerance.
        opts.grid = _fine_m06_grid()
        pyscf_kwargs.update(
            atom_grid=(120, 590),
            prune=None,
            conv_tol_grad=1e-5,
            use_newton=(label != "H-doublet"),
        )
    result = _run_uks(atoms, basis_name, vq_name, mult, opts=opts)
    assert result.converged, (
        f"{vq_name} / {label}: UKS did not converge in {result.n_iter} iters"
    )
    ref_E, ref_s2 = _pyscf_uks(atoms, basis_name, ps_name,
                                charge=charge, spin=mult - 1,
                                **pyscf_kwargs)
    # Default grid is "medium"; expect < 2e-5 Ha agreement for GGAs
    # and hybrids (the 612-point angular grid on O2 at compressed 1.2-bohr
    # geometry pushes this to ~1e-5). LDA has no gradient integration so
    # converges tighter.
    tol = 1e-6 if vq_name == "LDA" else 2e-5
    assert abs(result.energy - ref_E) < tol, (
        f"{vq_name} / {label}: E_vibeqc = {result.energy:.10f}, "
        f"E_pyscf = {ref_E:.10f}, diff = {result.energy - ref_E:+.2e}"
    )
    # <S^2> close to PySCF's (which itself may have small contamination
    # for UKS on symmetric systems).
    assert abs(result.s_squared - ref_s2) < 5e-3, (
        f"{vq_name} / {label}: <S^2>_vibeqc = {result.s_squared}, "
        f"pyscf = {ref_s2}"
    )


@pytest.mark.parametrize("functional",
                         ["LDA", "PBE", "B3LYP", "TPSS", "TPSSh",
                          "M06-L", "M06-2X", "r2scan", "r2scanh"])
def test_uks_on_closed_shell_matches_rks(functional):
    """Closed-shell UKS must collapse to RKS exactly."""
    atoms = [(8, [0, 0, 0]),
             (1, [0.0, 1.5, -1.16]),
             (1, [0.0, -1.5, -1.16])]
    r_uks = _run_uks(atoms, "sto-3g", functional, mult=1)
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    rks_opts = RKSOptions()
    rks_opts.functional = functional
    rks_opts.conv_tol_energy = 1e-10
    rks_opts.conv_tol_grad = 1e-7
    r_rks = run_rks(mol, basis, rks_opts)
    assert r_uks.converged and r_rks.converged
    assert abs(r_uks.energy - r_rks.energy) < 1e-9
    # Closed-shell: exact <S^2> = 0.
    assert abs(r_uks.s_squared) < 1e-8


def test_uks_s_squared_for_doublet_is_near_075():
    """H atom is a one-electron system with zero possible spin
    contamination: <S^2> should be exactly 0.75 for any functional."""
    for fn in ("LDA", "PBE", "B3LYP"):
        r = _run_uks([(1, [0, 0, 0])], "sto-3g", fn, mult=2)
        assert r.s_squared == pytest.approx(0.75, abs=1e-10)
        assert r.s_squared_deviation == pytest.approx(
            r.s_squared - r.s_squared_ideal,
            abs=1e-14,
        )


def test_uks_default_stability_check_skips_unsupported_response_models():
    """The automatic check runs only when the complete Hessian is available."""
    lda = _run_uks([(1, [0, 0, 0])], "sto-3g", "LDA", mult=2)
    assert lda.converged
    assert lda.stability_checked

    for functional in ("TPSS", "wb97x", "vv10"):
        result = _run_uks(
            [(1, [0, 0, 0])], "sto-3g", functional, mult=2
        )
        assert result.converged
        assert not result.stability_checked


@pytest.mark.parametrize(
    ("functional", "reason"),
    [
        ("TPSS", "tau-dependent polarised fxc"),
        ("wb97x", "range-separated exact-exchange response"),
        ("vv10", "VV10 nonlocal correlation response"),
    ],
)
def test_uks_explicit_incomplete_stability_model_fails_closed(
    functional, reason
):
    """An explicit request must not manufacture an incomplete Hessian."""
    opts = _tight_uks_opts()
    opts.functional = functional
    opts.stability_check = True

    with pytest.raises(RuntimeError, match=reason):
        _run_uks(
            [(1, [0, 0, 0])], "sto-3g", functional, mult=2, opts=opts
        )


def test_uks_explicit_dft_plus_u_stability_check_fails_closed():
    """The Dudarev curvature must not be omitted from an explicit check."""
    opts = _tight_uks_opts()
    opts.functional = "PBE"
    opts.stability_check = True
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)

    with pytest.raises(RuntimeError, match=r"DFT\+U response"):
        run_uks(
            mol,
            BasisSet(mol, "cc-pvdz"),
            opts,
            dft_plus_u=[HubbardSite(0, 0, U_ev=2.0)],
        )


def test_uks_stability_follow_escapes_stretched_h2_saddle():
    """Issue #447: a negative UKS mode must reach a stable determinant.

    At 4 bohr the spin-symmetric PBE/STO-3G solution is an internal saddle,
    38.672682 mHa above the atom-localised broken-symmetry minimum.  Before
    the fix the stability solve found lambda_min = -0.1075884 Ha, but its
    XC-free, one-sign line search rejected every escape and returned the
    symmetric density with zero restarts.
    """
    opts = _tight_uks_opts()
    opts.functional = "PBE"
    opts.initial_guess = InitialGuess.SAD
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 120

    mol = Molecule(
        [Atom(1, [0.0, 0.0, -2.0]), Atom(1, [0.0, 0.0, 2.0])],
        multiplicity=1,
    )
    # Capture the gated native trace so this regression pins that *both*
    # signed Seeger-Pople descents are actually reconverged. The H2 branches
    # are degenerate, so final energy alone cannot distinguish a one-sign
    # implementation from the required two-sign procedure.
    from vibeqc import _vibeqc_core as core
    from vibeqc.output import _cpp_diagnostics as cpp_diagnostics

    diagnostics = []
    saved_level = int(core._get_diagnostics_level())
    restore_sink = (
        cpp_diagnostics._diag_callback
        if cpp_diagnostics._bridge_installed
        else None
    )
    try:
        core._set_diagnostics_level(3)
        core._set_diagnostics_sink(
            lambda tag, level, message: diagnostics.append(
                (tag, level, message)
            )
        )
        result = run_uks(mol, BasisSet(mol, "sto-3g"), opts)
    finally:
        core._set_diagnostics_sink(restore_sink)
        core._set_diagnostics_level(saved_level)

    assert result.converged
    assert result.stability_checked
    assert result.stability_analysis_converged
    assert result.n_stability_restarts >= 1
    assert result.n_iter_before_stability == 2
    assert result.n_iter_before_stability < result.n_iter
    assert not result.internal_instability
    assert result.stability_eigenvalue >= -opts.stability_tol
    assert result.energy == pytest.approx(-0.935641595036327, abs=1e-9)
    assert result.s_squared == pytest.approx(0.8923980872, abs=1e-6)
    assert np.linalg.norm(
        np.asarray(result.density_alpha) - np.asarray(result.density_beta)
    ) > 1.0
    branch_messages = [
        message
        for tag, _level, message in diagnostics
        if tag == "uks-stability" and "event=escape_candidate" in message
    ]
    assert any("sign=-1" in message and "converged=1" in message
               for message in branch_messages)
    assert any("sign=+1" in message and "converged=1" in message
               for message in branch_messages)


def test_uks_stability_follow_rotation_is_finite_beyond_minimal_basis():
    """Issue #447: rank-deficient rotations must remain finite.

    The old UKS-only exponential diagonalised ``K**2`` and evaluated
    ``sqrt(-mu)``.  In def2-SVP, roundoff makes null-space eigenvalues of the
    rank-two H2 rotation slightly positive, so every line-search density was
    NaN and the detected B3LYP saddle was returned with zero restarts.  This
    larger-basis witness complements the STO-3G two-sign regression above.
    """
    opts = _tight_uks_opts()
    opts.functional = "B3LYP"
    opts.initial_guess = InitialGuess.SAD
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 120

    mol = Molecule(
        [Atom(1, [0.0, 0.0, -1.5]), Atom(1, [0.0, 0.0, 1.5])],
        multiplicity=1,
    )
    result = run_uks(mol, BasisSet(mol, "def2-svp"), opts)

    assert result.converged
    assert result.stability_checked
    assert result.stability_analysis_converged
    assert result.n_stability_restarts >= 1
    assert not result.internal_instability
    assert result.stability_eigenvalue >= -opts.stability_tol
    assert result.energy == pytest.approx(-1.0453384673384285, abs=1e-9)
    assert result.s_squared == pytest.approx(0.1745466754, abs=1e-6)
    assert np.isfinite(np.asarray(result.density_alpha)).all()
    assert np.isfinite(np.asarray(result.density_beta)).all()
    assert np.linalg.norm(
        np.asarray(result.density_alpha) - np.asarray(result.density_beta)
    ) > 0.3


def test_uks_reports_nonzero_xc_energy():
    atoms = [(1, [0, 0, 0])]
    r = _run_uks(atoms, "sto-3g", "PBE", mult=2)
    assert r.e_xc != 0.0


def test_uks_rejects_mult_electron_inconsistency():
    with pytest.raises(ValueError, match="inconsistent"):
        Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
                 multiplicity=2)


def test_uks_wb97x_open_shell_o2_triplet():
    """Open-shell UKS with the range-separated hybrid ωB97X — exercises
    the per-spin (cam_alpha·K_σ + cam_beta·K_erf,σ) exchange assembly on
    a genuine open-shell density. O2 triplet / def2-SVP; energy
    cross-checked vs PySCF.dft UKS (≈ 0.6 µHa). <S^2> ≈ 2 for the
    triplet ground state."""
    o2 = [(8, [0, 0, -1.1]), (8, [0, 0, 1.1])]
    r = _run_uks(o2, "def2-svp", "wb97x", mult=3)
    assert r.converged, f"ωB97X UKS on O2 triplet did not converge ({r.n_iter} iters)"
    assert r.energy == pytest.approx(-150.167396, abs=5e-4)
    assert r.s_squared == pytest.approx(2.0, abs=0.05)


def test_uks_b3lyp_sh_def2_svp_clears_commutator_gate():
    """The release-paper SH doublet must not stall on a flat-energy tail.

    This is the smallest deterministic near-miss from DL-F1 in the archived
    open-shell DFT ladder: the energy is stable to about 3e-11 Ha, but the
    UKS commutator remains at 1.342e-6 through the 120-cycle cap.
    """
    mol = Molecule(
        [
            Atom(16, [0.0, 0.0, 0.0]),
            Atom(1, [1.341 * ANGSTROM_TO_BOHR, 0.0, 0.0]),
        ],
        multiplicity=2,
    )
    basis = BasisSet(mol, "def2-svp")

    probe_opts = UKSOptions()
    probe_opts.functional = "B3LYP"
    probe_opts.damping = 0.0
    probe_opts.max_iter = 1
    probe_opts.grid = cosx_grid_options_for_level(3)
    probe = run_uks(mol, basis, probe_opts)

    overlap = np.asarray(compute_overlap(basis))
    s_eval, s_evec = np.linalg.eigh(overlap)
    keep = s_eval > probe_opts.linear_dep_threshold
    orthogonalizer = s_evec[:, keep] / np.sqrt(s_eval[keep])
    ao_norms = []
    orthonormal_norms = []
    for spin in ("alpha", "beta"):
        fock = np.asarray(getattr(probe, f"fock_{spin}"))
        density = np.asarray(getattr(probe, f"density_{spin}"))
        error_ao = fock @ density @ overlap - overlap @ density @ fock
        error_orth = orthogonalizer.T @ error_ao @ orthogonalizer
        ao_norms.append(np.linalg.norm(error_ao))
        orthonormal_norms.append(np.linalg.norm(error_orth))

    assert probe.scf_trace[-1].grad_norm == pytest.approx(
        max(orthonormal_norms), rel=1e-12
    )
    assert abs(max(ao_norms) - max(orthonormal_norms)) > 1e-2

    opts = UKSOptions()
    opts.functional = "B3LYP"
    # This revision-bound pin was measured with the former PATOM default.
    opts.initial_guess = InitialGuess.PATOM
    opts.damping = 0.0
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-8
    opts.grid = cosx_grid_options_for_level(3)

    result = run_uks(mol, basis, opts)

    assert result.converged, (
        f"UKS stalled for {result.n_iter} cycles with final commutator "
        f"{result.scf_trace[-1].grad_norm:.9e}"
    )
    # Revision-bound fixed point; cross-code accuracy is tracked by the
    # retained DFT-ladder comparison. Normalized PATOM changed the old
    # -398.5689395806 Ha pin by 10.8 nHa; averaging the degenerate beta pi
    # frontier of the open-shell PATOM seed (the #336 rule) moves the landing
    # point on this radical's 1.1 uHa grid ripple (vibe-qc#55, note 36789)
    # from -398.56893956981907 to -398.5689396378825, a shift of -6.8e-8 Ha
    # measured on two builds of that change; the ripple minimum
    # -398.5689406860 is not reached. This maintainer-approved re-pin retains
    # the 1 nHa bound. fb8ae35c (#472) diagonalizes the symmetrized projected
    # Fock matrix 0.5 (F' + F'^T) instead of one triangle of F'; that
    # roundoff-level change alone moves the landing point by -2.05e-8 Ha, to
    # -398.56893965843494 in 36 cycles (bisected; restoring the unsymmetrized
    # solve on main gives back -398.5689396378825 in 34 cycles).
    assert result.energy == pytest.approx(-398.56893965843494, abs=1e-9)
    assert result.scf_trace[-1].grad_norm < opts.conv_tol_grad


@pytest.fixture
def sh_read_witness():
    """Exact legal spin densities exposing the SH UKS recovery failure (#55).

    The fixture contains only the normalized PATOM alpha/beta densities for
    this geometry and basis. Retaining them isolates SCF recovery from future
    changes to the initial-guess generator.
    """
    from pathlib import Path

    mol = Molecule([
        Atom(16, [0.0, 0.0, 0.0]),
        Atom(1, [1.341 * ANGSTROM_TO_BOHR, 0.0, 0.0]),
    ], multiplicity=2)
    basis = BasisSet(mol, "def2-svp")
    opts = UKSOptions()
    opts.functional = "B3LYP"
    opts.grid = cosx_grid_options_for_level(3)
    opts.initial_guess = InitialGuess.READ
    opts.damping = 0.0
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    opts.stability_check = False
    with np.load(Path(__file__).parent / "data" / "uks_sh_read_seed.npz") as seed:
        opts.read_density_alpha = seed["alpha"]
        opts.read_density_beta = seed["beta"]
    return mol, basis, opts


@pytest.mark.parametrize("stall_limit", [1, 15])
def test_uks_restart_requires_consecutive_stalls(sh_read_witness, stall_limit):
    """Improvement separates stalled runs instead of advancing a later restart.

    The original default-limit trajectory incorrectly restarts at cycle 37:
    fifteen stalls have accumulated, but only the last ten are consecutive.
    A limit of one remains an active recovery control.
    """
    from vibeqc import _vibeqc_core as core
    from vibeqc.output import _cpp_diagnostics as cpp_diagnostics

    mol, basis, opts = sh_read_witness
    opts.max_iter = 38
    opts.restart_opts.max_stall_iters = stall_limit
    diagnostics = []
    saved_level = int(core._get_diagnostics_level())
    restore_sink = (
        cpp_diagnostics._diag_callback
        if cpp_diagnostics._bridge_installed else None
    )
    try:
        core._set_diagnostics_level(3)
        core._set_diagnostics_sink(
            lambda tag, level, message: diagnostics.append((tag, message))
        )
        result = core.run_uks(mol, basis, opts)
    finally:
        core._set_diagnostics_sink(restore_sink)
        core._set_diagnostics_level(saved_level)
    restarts = [message for tag, message in diagnostics
                if tag == "uks" and message.startswith("restart ")]
    if stall_limit == 15:
        assert not restarts, restarts
    else:
        assert restarts, "The configured one-stall recovery must remain active"
    assert result.n_iter <= opts.max_iter
    if not result.converged:
        assert result.n_iter == opts.max_iter


@pytest.mark.parametrize("max_iter", [2, 5])
def test_stopped_uks_returns_evaluated_spin_focks(sh_read_witness, max_iter):
    """Stopped results retain F[D], not the DIIS search extrapolate (#55)."""
    from vibeqc import _vibeqc_core as core

    mol, basis, opts = sh_read_witness
    opts.max_iter = max_iter
    # Keep the same full operator in both calls, with no coarse-screening
    # or incremental approximation as an alternative explanation.
    opts.incremental_fock = False
    opts.schwarz_threshold_loose = opts.schwarz_threshold
    stopped = core.run_uks(mol, basis, opts)
    assert not stopped.converged
    assert stopped.n_iter == max_iter
    assert any(row.accelerator_step != 0 for row in stopped.scf_trace)

    opts.max_iter = 0
    opts.read_density_alpha = stopped.density_alpha
    opts.read_density_beta = stopped.density_beta
    evaluated = core.run_uks(mol, basis, opts)
    assert evaluated.n_iter == 0
    assert evaluated.energy == pytest.approx(stopped.energy, abs=1e-10)
    overlap = np.asarray(compute_overlap(basis))
    vals, vecs = np.linalg.eigh(overlap)
    keep = vals > opts.linear_dep_threshold
    X = vecs[:, keep] / np.sqrt(vals[keep])
    norms = []
    for spin in ("alpha", "beta"):
        density = np.asarray(getattr(stopped, f"density_{spin}"))
        # READ admission removes antisymmetric roundoff before evaluation.
        # Check that exact map, not bit identity with the pre-admission matrix;
        # do not permit population rescaling or relax the F[D]/residual bounds.
        admitted_density = 0.5 * (density + density.T)
        np.testing.assert_array_equal(getattr(evaluated, f"density_{spin}"), admitted_density)
        fock = np.asarray(getattr(stopped, f"fock_{spin}"))
        np.testing.assert_allclose(fock, getattr(evaluated, f"fock_{spin}"),
                                   atol=1e-11, rtol=0.0)
        residual = fock @ density @ overlap - overlap @ density @ fock
        norms.append(np.linalg.norm(X.T @ residual @ X))
    assert max(norms) == pytest.approx(stopped.scf_trace[-1].grad_norm,
                                     abs=1e-11, rel=0.0)


@pytest.mark.xfail(
    strict=True,
    reason="#55: the SH READ witness stops at max_iter with |grad| ~ 4e-6 under "
    "default recovery; this is the open acceptance case, not a regression",
)
def test_uks_sh_original_read_seed_clears_tight_gate(sh_read_witness):
    """The original tight witness must converge with default recovery (#55)."""
    from vibeqc import _vibeqc_core as core

    mol, basis, opts = sh_read_witness
    result = core.run_uks(mol, basis, opts)
    assert result.converged, (
        result.energy, result.n_iter, result.scf_trace[-1].grad_norm
    )
    overlap = np.asarray(compute_overlap(basis))
    vals, vecs = np.linalg.eigh(overlap)
    keep = vals > opts.linear_dep_threshold
    X = vecs[:, keep] / np.sqrt(vals[keep])
    for spin, count in (("alpha", 9), ("beta", 8)):
        density = np.asarray(getattr(result, f"density_{spin}"))
        fock = np.asarray(getattr(result, f"fock_{spin}"))
        C = np.asarray(getattr(result, f"mo_coeffs_{spin}"))[:, :count]
        assert np.trace(density @ overlap) == pytest.approx(count, abs=1e-10)
        np.testing.assert_allclose(density @ overlap @ density, density, atol=1e-10)
        np.testing.assert_allclose(C @ C.T, density, atol=1e-8)
        error = fock @ density @ overlap - overlap @ density @ fock
        assert np.linalg.norm(X.T @ error @ X) < opts.conv_tol_grad


@pytest.mark.parametrize("scenario", [
    "no-restarts", "pattern-hold", "spin-schedule", "disabled-schedule",
    "atomspin", "short-window",
])
def test_uks_recovery_diagnostic_rejects_unsupported_entry(sh_read_witness, scenario):
    """The real private entry refuses transformed targets before any SCF prefix."""
    from vibeqc import _vibeqc_core as core, SpinlockMode
    from vibeqc.output import _cpp_diagnostics as diagnostics

    mol, basis, opts = sh_read_witness
    if scenario == "no-restarts":
        opts.restart_opts.max_restarts = 0
    elif scenario == "atomspin":
        opts.atomic_spins = [1, -1]
    elif scenario == "short-window":
        opts.max_iter = 50
    else:
        opts.spinlock_mode = (SpinlockMode.PATTERN_HOLD if scenario == "pattern-hold"
                              else SpinlockMode.SPIN_SCHEDULE)
        opts.spinlock_iterations = 0 if scenario == "disabled-schedule" else 60
        opts.spinlock_value = 1
    progress = []
    level = int(core._get_diagnostics_level())
    sink = diagnostics._diag_callback if diagnostics._bridge_installed else None
    try:
        core._set_diagnostics_level(3)
        core._set_diagnostics_sink(lambda tag, severity, msg: progress.append((tag, msg)))
        with pytest.raises(ValueError, match="Invalid bounded UKS recovery diagnostic request"):
            core._diagnose_uks_recovery(mol, basis, opts, 42, 12)
    finally:
        core._set_diagnostics_sink(sink)
        core._set_diagnostics_level(level)
    assert not progress, "An unsupported request must be rejected before setup/replay"


def test_uks_recovery_diagnostic_rejects_residual_below_gate(sh_read_witness):
    """A stalled energy alone cannot authorize an orbital-recovery checkpoint.

    This is a refusal control with explicitly different options, not the #55
    scientific acceptance case. Zero energy tolerance keeps its energy gate
    false while the loose residual tolerance forbids production recovery.
    Exactly one 42-step prefix is allowed and no factorial tail may run.
    """
    import re
    from vibeqc import _vibeqc_core as core
    from vibeqc.output import _cpp_diagnostics as diagnostics

    mol, basis, opts = sh_read_witness
    opts.conv_tol_energy = 0.0
    opts.conv_tol_grad = 1.0
    progress = []
    level = int(core._get_diagnostics_level())
    sink = diagnostics._diag_callback if diagnostics._bridge_installed else None
    try:
        core._set_diagnostics_level(3)
        core._set_diagnostics_sink(lambda tag, severity, msg: progress.append((tag, msg)))
        with pytest.raises(RuntimeError, match="Expected first orbital-recovery checkpoint was not reached"):
            core._diagnose_uks_recovery(mol, basis, opts, 42, 12)
    finally:
        core._set_diagnostics_sink(sink)
        core._set_diagnostics_level(level)
    iterations = [int(re.search(r"iter=(\d+)", msg).group(1))
                  for tag, msg in progress if tag == "progress" and "iter=" in msg]
    assert iterations == list(range(1, 43)), iterations


def _assert_capture_is_passive(core, mol, basis, opts, first, last):
    plain = core.run_uks(mol, basis, opts)
    observed, report = core._run_uks_with_diis_capture(mol, basis, opts, first, last)
    assert observed.energy == plain.energy
    assert observed.converged == plain.converged
    for spin in ("alpha", "beta"):
        for field in ("density", "fock", "mo_coeffs"):
            np.testing.assert_array_equal(getattr(observed, f"{field}_{spin}"),
                                          getattr(plain, f"{field}_{spin}"))
    assert len(observed.scf_trace) == len(plain.scf_trace)
    for a, b in zip(observed.scf_trace, plain.scf_trace):
        for field in ("energy", "grad_norm", "delta_e", "diis_subspace", "accelerator_step"):
            assert getattr(a, field) == getattr(b, field)
    assert [f["iteration"] for f in report["frames"]] == list(range(first, last + 1))
    # A later ordinary call must not inherit a pointer into the expired report.
    assert core.run_uks(mol, basis, opts).energy == plain.energy
    return plain, report


def _check_diis_frame(frame):
    errors = frame["errors"]
    gram = np.array([[np.sum(a * b) for b in errors] for a in errors])
    np.testing.assert_allclose(frame["pushed_gram"], gram, rtol=1e-13, atol=1e-30)
    state = frame["matrices"]
    np.testing.assert_array_equal(errors[-1], np.vstack([
        state["residual_alpha"], state["residual_beta"]]))
    if not frame["attempts"]:
        return None, 0
    solve = frame["attempts"][-1]
    m = len(solve["solution"]) - 1
    return solve, m


def _check_ediis_frame(frame):
    """EDIIS history/model of Kudin, Scuseria & Cances (2002) Eqs. (8), (10)."""
    state, ediis = frame["matrices"], frame["ediis"]
    focks, dens, energies = ediis["focks"], ediis["densities"], ediis["energies"]
    # The newest pushed iterate is this frame's evaluated (F[D], D, E).
    np.testing.assert_array_equal(focks[-1][0], state["physical_fock_alpha"])
    np.testing.assert_array_equal(focks[-1][1], state["physical_fock_beta"])
    np.testing.assert_array_equal(dens[-1][0], state["density_alpha"])
    np.testing.assert_array_equal(dens[-1][1], state["density_beta"])
    assert energies[-1] == frame["values"]["energy"]
    alive = list(range(len(energies)))
    for solve in ediis["attempts"]:
        n = len(alive)
        F = [focks[i] for i in alive]
        D = [dens[i] for i in alive]
        E = [energies[i] for i in alive]
        r = n - 1

        def tr(i, j):
            return sum(np.sum((F[i][b] - F[r][b]) * (D[j][b] - D[r][b])) for b in (0, 1))

        if n > 1:
            linear = np.array([2 * (E[i] - E[r]) - tr(i, i) for i in range(n)])
            quad = np.array([[0.5 * (tr(i, j) + tr(j, i)) for j in range(n)]
                             for i in range(n)])
            scale = max(1.0, np.abs(quad).max(), np.abs(linear).max())
            np.testing.assert_allclose(solve["linear"], linear, rtol=0, atol=1e-12 * scale)
            np.testing.assert_allclose(solve["quadratic"], quad, rtol=0, atol=1e-12 * scale)
        c = solve["coefficients"]
        assert len(c) == n and np.all(c >= 0.0) and abs(c.sum() - 1.0) < 1e-12
        # The simplex minimiser never exceeds the newest vertex's energy.
        assert solve["interpolated_energy"] <= E[r] + 1e-9 * max(1.0, abs(E[r]))
        if solve["erased_index"] >= 0:
            alive.remove(solve["erased_index"])
    assert alive == list(ediis["surviving"])
    assert ediis["attempts"][-1]["erased_index"] == -1
    c = ediis["attempts"][-1]["coefficients"]
    return [sum(ci * focks[i][b] for ci, i in zip(c, alive)) for b in (0, 1)]


def _check_search_fock(frame):
    state, step = frame["matrices"], frame["values"]["accelerator_step"]
    if step == 1:
        solve, m = _check_diis_frame(frame)
        assert solve is not None and solve["usable"] and solve["rank"] == m + 1
        coefficients = solve["solution"][:m]
        assert abs(np.sum(coefficients) - 1) < 1e-10
        expected = sum(c * f for c, f in zip(coefficients, frame["focks"][-m:]))
        np.testing.assert_allclose(expected, np.vstack([
            state["search_fock_alpha"], state["search_fock_beta"]]), rtol=1e-13, atol=1e-13)
    elif step == 3:
        fa, fb = _check_ediis_frame(frame)
        np.testing.assert_allclose(fa, state["search_fock_alpha"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fb, state["search_fock_beta"], rtol=1e-13, atol=1e-13)
    else:
        raise AssertionError(f"unexpected accelerator step {step}")


def test_uks_diis_capture_preserves_trajectory_and_actual_history(sh_read_witness):
    """The observer records the coupled solve without changing SCF choices.

    The witness keeps its default EDIIS_DIIS accelerator: both histories are
    captured every cycle and the branch that reached the SCF is reconstructed.
    """
    from vibeqc import SCFAccelerator
    from vibeqc import _vibeqc_core as core

    mol, basis, opts = sh_read_witness
    assert opts.scf_accelerator == SCFAccelerator.EDIIS_DIIS
    opts.max_iter = 5
    _, report = _assert_capture_is_passive(core, mol, basis, opts, 2, 5)
    nbf = basis.nbasis
    for frame in report["frames"]:
        values = frame["values"]
        assert values["switch_threshold"] == opts.ediis_diis_switch_threshold
        assert values["switch_metric"] == values["gradient"] / (math.sqrt(2.0) * nbf)
        expected_step = 3 if values["switch_metric"] > values["switch_threshold"] else 1
        assert values["accelerator_step"] == expected_step
        _check_diis_frame(frame)
        _check_ediis_frame(frame)
        _check_search_fock(frame)


def test_uks_diis_capture_records_ediis_phase_and_switch(sh_read_witness):
    """EDIIS cycles, the switch to DIIS and both histories are recorded.

    Only the switch threshold differs from the witness (its accelerator stays
    EDIIS_DIIS). At the default 1e-1 every witness cycle is DIIS-consumed (RMS
    metric <= 6e-3); at 5e-4 cycles 2-3 consume EDIIS and cycle 4 switches.
    """
    from vibeqc import _vibeqc_core as core

    mol, basis, opts = sh_read_witness
    opts.max_iter = 8
    opts.ediis_diis_switch_threshold = 5e-4
    _, report = _assert_capture_is_passive(core, mol, basis, opts, 1, 8)
    frames = report["frames"]
    steps = [f["values"]["accelerator_step"] for f in frames]
    assert steps[:4] == [0, 3, 3, 1], steps
    assert set(steps[4:]) == {1}, steps
    # Below diis_start_iter nothing is consumed: the search Fock is F[D].
    first = frames[0]["matrices"]
    for spin in ("alpha", "beta"):
        np.testing.assert_array_equal(first[f"search_fock_{spin}"],
                                      first[f"physical_fock_{spin}"])
    assert frames[0]["ediis"]["disposition"] == "single_iterate"
    for frame in frames[1:]:
        values = frame["values"]
        expected_step = 3 if values["switch_metric"] > values["switch_threshold"] else 1
        assert values["accelerator_step"] == expected_step
        # Both solves run each cycle, whichever return is consumed.
        assert frame["ediis"]["disposition"] in ("extrapolated", "replay_guard_resolved")
        assert frame["disposition"] == "extrapolated"
        _check_search_fock(frame)
    # EDIIS keeps every pushed iterate up to its depth: n = iteration.
    assert [len(f["ediis"]["energies"]) for f in frames] == list(range(1, 9))


def test_uks_diis_capture_accepts_largest_window(sh_read_witness):
    from vibeqc import _vibeqc_core as core

    mol, basis, opts = sh_read_witness
    opts.max_iter = 32
    _, report = core._run_uks_with_diis_capture(mol, basis, opts, 1, 32)
    assert [f["iteration"] for f in report["frames"]] == list(range(1, 33))


@pytest.mark.parametrize("first,last,bound", [
    (0, 3, "starts before iteration 1"),
    (4, 3, "ends before it starts"),
    (1, 33, "spans more than 32 iterations"),
    (100, 121, "ends after max_iter"),
])
def test_uks_diis_capture_rejects_unbounded_windows(sh_read_witness, first, last, bound):
    from vibeqc import _vibeqc_core as core

    mol, basis, opts = sh_read_witness
    with pytest.raises(ValueError, match=f"bounded UKS DIIS capture request: window {bound}"):
        core._run_uks_with_diis_capture(mol, basis, opts, first, last)


def test_uks_diis_capture_rejects_unsupported_accelerator(sh_read_witness):
    from vibeqc import SCFAccelerator
    from vibeqc import _vibeqc_core as core

    mol, basis, opts = sh_read_witness
    opts.scf_accelerator = SCFAccelerator.KDIIS
    with pytest.raises(ValueError, match="capture request: unsupported accelerator"):
        core._run_uks_with_diis_capture(mol, basis, opts, 2, 5)
