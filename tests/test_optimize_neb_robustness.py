"""Optimizer / NEB robustness regressions (2026-05-31 end-to-end audit).

These pin the silent-flag / robustness / edge bugs found in the geometry
optimizers and the NEB driver. The underlying physics was already correct
(geometries <=2e-6 bohr vs geomeTRIC, NEB barrier 8e-13 vs PySCF); the
bugs here are about *reporting* convergence honestly and *failing
gracefully*:

* F1 -- ``optimize_molecule`` (and the BIPOLE relaxers) reported
  ``converged=True`` at a non-stationary geometry because scipy's
  L-BFGS-B sets ``res.success`` on EITHER its gradient (gtol) OR its
  energy-reduction (ftol) criterion. Convergence is now gated on the
  actual max-component force.
* F2 -- a NEB run aborted the whole band with a cryptic C++
  ``RuntimeError: ... not converged`` when one image's SCF hit max_iter;
  it now raises a clear, image-named :class:`vibeqc.neb.NEBImageSCFError`.
* F3 -- NEB cold-start images used a weaker Hcore guess than the public
  ``run_uhf`` / ``run_rks`` drivers (AUTO->SAD), so they needed more SCF
  iterations and could fail at the default ``max_iter`` where an
  equivalent single point converges. Cold starts now seed SAD.
* F4 -- ``optimize_molecule(method="rks", functional=None)`` raised a
  ``TypeError`` from an operator-precedence bug in the SCF dispatch.

CI on this repo runs ``pytest --collect-only`` (no execution), so these
must be run locally; see the audit handover for the de-hook harness.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import (
    Atom,
    BasisSet,
    GradientOptions,
    GridOptions,
    Molecule,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    run_uhf,
)
from vibeqc.molecular_optimize import (
    _gradient_converged,
    _run_molecular_scf,
    optimize_molecule,
)
from vibeqc.neb import (
    NEBImageSCFError,
    _evaluate_image,
    _sad_cold_start_closed,
    _sad_cold_start_open,
    run_neb,
)


def _water_sto3g() -> Molecule:
    # Near the STO-3G water minimum (O-H ~1.83 bohr, angle ~105 deg).
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.45, 1.12]),
            Atom(1, [0.0, -1.45, 1.12]),
        ],
        charge=0,
        multiplicity=1,
    )


def _stretched_h3_doublet(z2: float, z3: float) -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, z2]), Atom(1, [0.0, 0.0, z3])],
        charge=0,
        multiplicity=2,
    )


# --------------------------------------------------------------------------
# F1 -- the shared gradient-convergence gate (pure logic; no SCF)
# --------------------------------------------------------------------------


def test_gradient_converged_helper_truth_table():
    # success + small inf-norm gradient -> converged
    conv, gmax = _gradient_converged(True, np.array([1e-7, 2e-8, -5e-8]), 1e-6)
    assert conv is True and gmax == pytest.approx(1e-7)
    # success but a single large component -> NOT converged (the F1 case);
    # the 2-norm-vs-inf-norm distinction must not hide a large component.
    conv, gmax = _gradient_converged(True, np.array([1e-9, 0.01, 1e-9]), 1e-6)
    assert conv is False and gmax == pytest.approx(0.01)
    # scipy failure -> never converged, regardless of gradient size
    assert _gradient_converged(False, np.array([1e-12]), 1e-6)[0] is False
    # no gradient available -> not converged, grad_max == inf
    conv, gmax = _gradient_converged(True, None, 1e-6)
    assert conv is False and gmax == float("inf")


def test_optimize_molecule_converged_implies_stationary():
    """A 'converged' optimize_molecule result must actually be stationary."""
    tol = 1e-6
    # Loose ftol relative to gtol: scipy's energy-reduction criterion
    # trips while the forces are still large. Pre-fix this falsely
    # reported converged=True.
    res = optimize_molecule(
        _water_sto3g(),
        "sto-3g",
        method="rhf",
        conv_tol_grad=tol,
        conv_tol_energy=1e-3,
        max_iter=200,
    )
    gmax = float(np.max(np.abs(res.gradient))) if len(res.gradient) else float("inf")
    # The optimizer genuinely stopped at a non-stationary geometry here...
    assert gmax > tol, f"test setup no longer exercises the ftol-stop (gmax={gmax:.2e})"
    # ...so it must NOT be reported converged (F1 invariant).
    if res.converged:
        raise AssertionError(
            f"converged=True at a non-stationary geometry: max|grad|={gmax:.3e} "
            f"> conv_tol_grad={tol:.0e}"
        )


def test_optimize_molecule_real_convergence_still_reported():
    """Positive control: the gate must still pass a genuine minimum."""
    tol = 1e-4
    res = optimize_molecule(
        _water_sto3g(),
        "sto-3g",
        method="rhf",
        conv_tol_grad=tol,
        conv_tol_energy=1e-9,
        max_iter=200,
    )
    assert res.converged, "gradient gate rejected a genuinely converged optimization"
    assert float(np.max(np.abs(res.gradient))) <= tol


# --------------------------------------------------------------------------
# F4 -- RKS/UKS dispatch with functional=None must not raise TypeError
# --------------------------------------------------------------------------


def test_run_molecular_scf_rks_default_functional_no_typeerror():
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], charge=0, multiplicity=1)
    basis = BasisSet(mol, "sto-3g")
    # functional=None + default RKSOptions (functional="LDA"): pre-fix this
    # ran `opts.functional = None` and the pybind str setter raised.
    energy, result = _run_molecular_scf(
        mol, basis, "rks", functional=None, rks_options=RKSOptions()
    )
    assert result.converged and energy < 0.0


def test_run_molecular_scf_uks_default_functional_no_typeerror():
    mol = Molecule([Atom(1, [0, 0, 0])], charge=0, multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    energy, result = _run_molecular_scf(
        mol, basis, "uks", functional=None, uks_options=UKSOptions()
    )
    assert result.converged


# --------------------------------------------------------------------------
# F2 -- a non-converged NEB image raises a clear, image-named error
# --------------------------------------------------------------------------


def test_neb_image_nonconverged_raises_clear_error():
    template = _stretched_h3_doublet(1.6, 4.5)
    positions = np.array([[0, 0, 0], [0, 0, 1.6], [0, 0, 4.5]], dtype=float)
    opts = UHFOptions()
    opts.max_iter = 1  # guarantee non-convergence regardless of guess
    with pytest.raises(NEBImageSCFError) as excinfo:
        _evaluate_image(
            positions,
            template,
            "sto-3g",
            "uhf",
            functional=None,
            rhf_options=None,
            uhf_options=opts,
            rks_options=None,
            uks_options=None,
            gradient_options=GradientOptions(),
            grid_options=GridOptions(),
            dispersion_params=None,
            initial_density=None,
            image_index=3,
        )
    message = str(excinfo.value)
    assert "image 3" in message
    assert "did not converge" in message


# --------------------------------------------------------------------------
# F3 -- NEB cold-start seeds SAD (matches the public driver's guess)
# --------------------------------------------------------------------------


def test_sad_cold_start_builds_valid_guess():
    """The cold-start helpers build a real SAD density, not the empty
    (Hcore) fallback."""
    mol = _stretched_h3_doublet(1.4, 4.0)
    basis = BasisSet(mol, "sto-3g")
    S = None
    from vibeqc.neb import _build_scf_common_pieces

    S, _h, _e, _jk = _build_scf_common_pieces(mol, basis)
    n_alpha = (mol.n_electrons() + 1) // 2
    n_beta = mol.n_electrons() - n_alpha

    d_alpha, d_beta = _sad_cold_start_open(mol, basis, n_alpha, n_beta)
    assert d_alpha.shape[0] > 0 and d_beta.shape[0] > 0, "fell back to empty (Hcore)"
    # Open-shell SAD now preserves the atoms' natural Hund occupations; the
    # first SCF diagonalisation enforces the requested global n_alpha/n_beta.
    # The seed itself must still carry the physical total electron count and
    # a majority-alpha spin bias for this doublet.
    n_guess_alpha = float(np.trace(d_alpha @ S))
    n_guess_beta = float(np.trace(d_beta @ S))
    assert n_guess_alpha + n_guess_beta == pytest.approx(
        n_alpha + n_beta, abs=1e-6
    )
    assert n_guess_alpha > n_guess_beta

    # Closed-shell variant on water.
    wat = _water_sto3g()
    wbasis = BasisSet(wat, "sto-3g")
    Sw, _h2, _e2, _jk2 = _build_scf_common_pieces(wat, wbasis)
    d = _sad_cold_start_closed(wat, wbasis)
    assert d.shape[0] > 0
    assert float(np.trace(d @ Sw)) == pytest.approx(wat.n_electrons(), abs=1e-6)


def test_neb_cold_start_converges_where_hcore_would_fail():
    """A NEB cold-start UHF image reaches the same converged minimum the
    public run_uhf driver does (the F3 fix: cold starts seed SAD, not the
    empty Hcore fallback).

    History of the iteration budget below: with the old per-spin DIIS this
    stretched-H3 doublet broke spin symmetry slowly and the guess set the
    count (SAD ~66 iters, Hcore ~82), so a tight cap converged from SAD but
    not Hcore. The 2026-07 spin-coupled DIIS (commit 4f8da2c8) removed that
    gap: it now converges in ~9 iters from either guess, so the guess no
    longer gates convergence on this system. The SAD seeding itself is
    pinned by test_sad_cold_start_builds_valid_guess; what this test still
    guards is the end-to-end cold-start-vs-single-point energy match."""
    template = _stretched_h3_doublet(1.4, 4.0)
    positions = np.array([[0, 0, 0], [0, 0, 1.4], [0, 0, 4.0]], dtype=float)
    opts = UHFOptions()
    # Coupled DIIS converges this doublet in ~9 iters (SAD 9, Hcore 8) to
    # E=-1.56944233, <S^2>~0.754; the cap only needs comfortable margin
    # over that count.
    opts.max_iter = 50

    energy, grad, _density = _evaluate_image(
        positions,
        template,
        "sto-3g",
        "uhf",
        functional=None,
        rhf_options=None,
        uhf_options=opts,
        rks_options=None,
        uks_options=None,
        gradient_options=GradientOptions(),
        grid_options=GridOptions(),
        dispersion_params=None,
        initial_density=None,
        image_index=2,
    )
    assert np.isfinite(energy)
    assert grad.shape == (3, 3)

    # The cold-start now reaches the same minimum the public driver does.
    ref = run_uhf(template, BasisSet(template, "sto-3g"), opts)
    assert ref.converged
    assert energy == pytest.approx(float(ref.energy), abs=1e-6)


# --------------------------------------------------------------------------
# End-to-end CI-NEB barrier (audit follow-up: "pin the H3 barrier")
# --------------------------------------------------------------------------


def test_neb_h3_exchange_barrier_uhf_sto3g():
    """End-to-end climbing-image NEB barrier for the textbook collinear
    H + H₂ → H₂ + H exchange at UHF/STO-3G.

    On the value — the 2026-05-31 audit asked to "pin the H3 ~11.2
    kcal/mol value", but ~11.2 is roughly the *physical* classical
    barrier (≈9.8); minimal-basis UHF/STO-3G substantially
    overestimates it (and DFT underestimates it: PBE/def2-svp gives
    ≈+2.6). The directly-computed UHF/STO-3G barrier (symmetric-saddle
    scan vs separated H₂(eq)+H) is **+23.2 kcal/mol**, and **+18.0**
    relative to this NEB's reactant-endpoint geometry [0, 1.4, 4.4]
    bohr (the spectator H there sits at 4.4 bohr, already partway up
    the entrance channel). So this test pins the genuine UHF/STO-3G
    value, not the physical one. The climbing image lands exactly on
    the symmetric saddle (R_HH ≈ 1.73 bohr, lit. ≈1.757) — the outer
    atoms migrate inward from the endpoint frame, so it is the true
    free saddle, not the constrained R=2.2 midpoint.

    Settings mirror ``test_neb_driver.TestCIClimbingImage`` (proven to
    converge to the saddle). ``n_jobs=1`` because the joblib process
    backend can't pickle the C++ Molecule/options (all NEB tests run
    serial).
    """
    reactant = _stretched_h3_doublet(1.4, 4.4)
    product = _stretched_h3_doublet(3.0, 4.4)
    uhf = UHFOptions()
    uhf.max_iter = 200
    result = run_neb(
        reactant,
        product,
        basis="sto-3g",
        method="UHF",
        uhf_options=uhf,
        n_images=5,
        spring_constant=0.1,
        interpolation="linear",
        max_iter=80,
        conv_tol_force=1e-3,
        n_jobs=1,
        initial_step=0.05,
        climbing_image=True,
        climbing_image_start_fraction=0.3,
    )
    energies = np.asarray(result.energies)
    ts = result.transition_state_index
    assert ts is not None

    # The climbing image sits exactly on the saddle: its energy equals
    # an independent UHF SCF at the converged TS geometry to < 1 mHa.
    ts_mol = result.path.images[ts].system
    e_ref = float(run_uhf(ts_mol, BasisSet(ts_mol, "sto-3g"), uhf).energy)
    assert abs(energies[ts] - e_ref) < 1e-3

    # TS is the symmetric collinear saddle at the true free-saddle
    # separation (≈1.73 bohr) — equal H–H distances, NOT the R=2.2
    # endpoint-frame midpoint.
    z = np.array([a.xyz[2] for a in ts_mol.atoms], dtype=float)
    r_left, r_right = z[1] - z[0], z[2] - z[1]
    assert abs(r_left - r_right) < 1e-2, f"asymmetric TS: {r_left:.4f} vs {r_right:.4f}"
    assert 1.6 < r_left < 1.9, f"R_HH={r_left:.4f} bohr is not the free saddle (~1.73)"

    # Barrier = climbing TS minus the (fixed) reactant-endpoint energy.
    # UHF/STO-3G value, cross-checked against the direct symmetric-
    # saddle scan (+18.0 kcal/mol endpoint-relative; +23.2 separated).
    barrier_kcal = (energies.max() - energies[0]) * 627.509474
    assert 16.5 < barrier_kcal < 19.5, f"barrier {barrier_kcal:.2f} kcal/mol off"
