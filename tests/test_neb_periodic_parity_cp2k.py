"""Periodic-NEB cross-code parity scaffold (vibe-qc ↔ CP2K).

Why this file exists
====================
``run_neb`` on :class:`PeriodicSystem` endpoints ships today, but its
*physical accuracy* is unvalidated: ``tests/test_neb_periodic.py``
only asserts the API holds together ("does NOT assert convergence"),
and no test pins a periodic barrier against an external reference.
Per CLAUDE.md § 7, an unchecked periodic energy path is exactly the
kind of thing that can harbour a silent gauge / Madelung /
image-summing bug (the canonical example: the v0.7.0 self-image leak
that over-bound H₂/STO-3G by ~0.587 Ha in a 30-bohr box). This file
is the anchor that closes that gap — it diffs vibe-qc's per-image
periodic SCF energies against CP2K on the *same* NEB geometries.

It is also the prerequisite the periodic-NEB roadmap step depends on:
before swapping the finite-difference per-image gradient for the
analytic BIPOLE gradient (the J^LR reciprocal-Ewald term — the
headline optimisation in the native periodic NEB work), you need an
external oracle to catch a regression. You cannot safely optimise an
unvalidated path.

Strategy (and its honest limits)
================================
* **Reference is single-point, not band-vs-band.** ``runner_cp2k``
  runs ``RUN_TYPE ENERGY``. So we run vibe-qc's *full* periodic NEB,
  take its relaxed per-image geometries, and have CP2K compute a
  single-point energy on each. We then compare the **relative energy
  profile** ΔEᵢ = Eᵢ − E₀ frame-by-frame. This validates the periodic
  SCF energy surface along the path — the § 7 failure class — without
  needing CP2K's own optimiser. (A true band-vs-band parity needs a
  ``RUN_TYPE BAND`` / ``&MOTION/&BAND`` deck builder added to
  ``runner_cp2k``; that is the next escalation, see PRODUCTION below.)

* **Smoke-level, not chemical-accuracy — by construction.**
  ``runner_cp2k`` is GPW (pseudopotential / GTH basis); vibe-qc
  periodic is all-electron Gaussian. The Hamiltonians differ, so
  *absolute* energies are not comparable and even the *relative*
  profile carries a basis/pseudo gap. The tolerance here
  (:data:`SMOKE_TOL_HA`) is therefore deliberately loose: it catches
  gross, geometry-dependent errors (wrong sign, factor-of-two,
  Madelung-scale self-image leaks — i.e. the § 7 bug class, which is
  ≥ 0.5 Ha) but does **not** assert chemical accuracy. Tightening to
  chemical accuracy is gated on (a) matched basis sets on both sides
  and (b) CP2K **GAPW** (all-electron) support in ``runner_cp2k``
  (its docstring marks GAPW as "M3+ scope"). Both are tracked as the
  escalation below; do not tighten :data:`SMOKE_TOL_HA` without them.

Runtime / gating
================
* ``@pytest.mark.slow`` (registered in ``pyproject.toml``): the
  periodic NEB uses the 6N+1-SCF FD gradient, so even the tiny H₂/box
  exemplar here is ~30 s. Deselected from the default fast run.
* The CP2K leg additionally skips when no ``cp2k.*`` executable is on
  PATH (``runner_cp2k.is_available()``), so laptop / CI runs without
  CP2K land cleanly on the vibe-qc-only consistency test.

Import convention for the regression package mirrors
``tests/test_runner_cp2k.py`` (the suite isn't installed as
``vibeqc.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq

# The regression-suite package isn't installed as ``vibeqc.*``;
# import it relatively the same way run_suite.py / test_runner_cp2k.py do.
REGRESSION_ROOT = Path(__file__).parent.parent / "examples" / "regression"
sys.path.insert(0, str(REGRESSION_ROOT.parent.parent))

from examples.regression.core import runner_cp2k       # noqa: E402
from examples.regression.core.spec import (            # noqa: E402
    AtomFrac,
    MethodSpec,
    PeriodicSpec,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOX_BOHR = 12.0
BOHR_TO_ANG = 0.52917721067

# Smoke-level tolerance on the per-image *relative* energy profile
# ΔEᵢ = Eᵢ − E₀, in Hartree. See the module docstring: this is sized
# to catch § 7-class geometry-dependent leaks (Madelung self-image
# over-binding is ≥ 0.5 Ha), NOT to assert chemical accuracy across
# the all-electron(vibe-qc) ↔ GTH-pseudo(CP2K) basis gap. The real
# deviation observed on a live CP2K run is logged by the parity test
# so a future matched-basis / GAPW setup can calibrate a tighter
# bound — do not tighten this without that calibration.
SMOKE_TOL_HA = 0.05

# vibe-qc atomic number → element symbol, for translating a
# PeriodicSystem geometry into the runner's AtomFrac spec. Light
# elements plus the named surface-NEB production-target metals
# (Fe(100), Pt(111)); extend as the exemplar system grows (mirrors the
# runner's own _GTH_VALENCE extend-as-needed pattern).
_Z_TO_SYMBOL = {
    1: "H", 6: "C", 7: "N", 8: "O", 13: "Al", 26: "Fe", 29: "Cu", 78: "Pt",
}


# ---------------------------------------------------------------------------
# vibe-qc side: a tiny closed-shell periodic NEB the CP2K runner can mirror
# ---------------------------------------------------------------------------
#
# H₂ stretch in a cubic box, neutral singlet, RKS-PBE. Closed-shell so
# CP2K's M1f whitelist (rks/rhf · lda/pbe/blyp/b3lyp) accepts it; H is
# q1 in the GTH set so the pseudopotential is effectively all-electron
# for this element, which keeps the basis gap as small as it gets for a
# GPW reference. This is an *exemplar to exercise the wiring*, not the
# production target — see the PRODUCTION placeholder at the bottom.


def _h2_in_box(d_bohr: float) -> vq.PeriodicSystem:
    """Neutral singlet H₂ along z in a cubic cell, bond length ``d_bohr``."""
    L = np.diag([BOX_BOHR, BOX_BOHR, BOX_BOHR])
    return vq.PeriodicSystem(
        3, L,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, float(d_bohr)])],
    )


def _rks_pbe_opts() -> "vq.PeriodicKSOptions":
    """Periodic RKS options matching the toy-example lattice cutoff."""
    opts = vq.PeriodicKSOptions()
    lat = vq.LatticeSumOptions()
    lat.cutoff_bohr = BOX_BOHR / 2.0
    opts.lattice_opts = lat
    opts.max_iter = 100
    return opts


def _run_small_periodic_neb() -> "vq.NEBResult":
    """Run the tiny H₂/box RKS-PBE periodic NEB (shared by both tests).

    Deliberately cheap: 3 intermediate images, 2 outer iterations,
    serial (``n_jobs=1`` — joblib's process pool is overkill and
    flakier than serial at this scale). It does not (need to)
    converge; both tests work off the *relaxed-so-far* per-image
    geometries + energies, which is all the parity check requires.
    """
    return vq.run_neb(
        _h2_in_box(1.2), _h2_in_box(2.0),
        basis="sto-3g",
        n_images=3,
        method="RKS",
        functional="pbe",
        rks_options=_rks_pbe_opts(),
        interpolation="linear",
        max_iter=2,
        conv_tol_force=5e-2,
        n_jobs=1,
        kpoints=(1, 1, 1),
        fd_step_bohr=5e-3,
        warm_start=True,
    )


@pytest.fixture(scope="module")
def periodic_neb_result() -> "vq.NEBResult":
    """Run the shared periodic NEB band once for the module."""
    return _run_small_periodic_neb()


# ---------------------------------------------------------------------------
# Geometry bridge: vibe-qc PeriodicSystem → runner PeriodicSpec
# ---------------------------------------------------------------------------


def _periodic_system_to_spec(
    system: vq.PeriodicSystem, *, spec_id: str,
) -> PeriodicSpec:
    """Translate a vibe-qc periodic geometry into a CP2K-runner spec.

    Converts the lattice bohr → Å (CP2K's ``A/B/C`` cell block is in
    Å) and Cartesian (bohr) → fractional coordinates (CP2K ``&COORD
    SCALED``). The fractional transform ``frac = cart · L⁻¹`` assumes
    the lattice is stored as **row** vectors — correct for the diagonal
    cubic cells used here; revisit for non-orthogonal cells if the
    exemplar grows.
    """
    L_bohr = np.asarray(system.lattice, dtype=float)
    cart_bohr = np.array([list(a.xyz) for a in system.unit_cell], dtype=float)
    frac = cart_bohr @ np.linalg.inv(L_bohr)
    lattice_ang = tuple(
        tuple(float(x) for x in row) for row in (L_bohr * BOHR_TO_ANG)
    )
    atoms = []
    for a, f in zip(system.unit_cell, frac):
        z = int(a.Z)
        try:
            symbol = _Z_TO_SYMBOL[z]
        except KeyError:  # pragma: no cover - guard for exemplar growth
            raise KeyError(
                f"no element-symbol mapping for Z={z}; add it to "
                f"_Z_TO_SYMBOL in {Path(__file__).name}"
            )
        atoms.append(AtomFrac(symbol=symbol, z=z, frac=(float(f[0]), float(f[1]), float(f[2]))))
    return PeriodicSpec(
        id=spec_id,
        family="neb-parity",
        lattice_ang=lattice_ang,
        space_group="P1",
        atoms=tuple(atoms),
    )


def _cp2k_energies_along_band(
    result: "vq.NEBResult", *, tmp_path: Path,
) -> tuple:
    """Run a CP2K single-point on every NEB image; return energies (Ha).

    Returns ``None`` (and the caller skips) if CP2K reports any frame
    as unavailable / error / unparseable — a partial profile can't be
    compared. The per-frame :class:`CodeRow` notes are surfaced in the
    skip message so a half-configured CP2K install (missing data files,
    unsupported element) is diagnosable.
    """
    method = MethodSpec(id="rks-pbe", scf="rks", xc="pbe")
    energies = []
    notes = []
    for i, img in enumerate(result.path.images):
        spec = _periodic_system_to_spec(img.system, spec_id=f"neb-img-{i}")
        row = runner_cp2k.run_periodic_case(
            run_id="neb-parity",
            target="vibeqc_periodic_neb",
            spec=spec,
            basis_name="dzvp-molopt-sr-gth",
            method=method,
            kmesh=(1, 1, 1),
            log_path=tmp_path / "cp2k_parity.log",
            workdir=tmp_path,
        )
        if row.status in ("unavailable", "error") or row.energy_ha is None:
            notes.append(f"img{i}: status={row.status!r} note={row.note!r}")
            return None, notes
        energies.append(float(row.energy_ha))
    return energies, notes


# ---------------------------------------------------------------------------
# Test 1 — vibe-qc periodic NEB produces an internally-consistent band
# ---------------------------------------------------------------------------
# Runs without CP2K. This is the in-CI (slow) regression anchor for
# "the periodic NEB driver emits a sane band": finite energies, the
# lattice is preserved across every image, the image count is right,
# and the TS index lands on an interior image.


def test_periodic_neb_band_is_internally_consistent(periodic_neb_result):
    result = periodic_neb_result
    n_total = len(result.path.images)
    assert n_total == 5  # 3 intermediate + 2 endpoints

    # All per-image energies are finite (NaN / Inf would mean a broken
    # SCF leaked into the band — the thing § 7 warns about).
    assert np.all(np.isfinite(result.energies))
    assert np.isfinite(result.max_force)

    # The lattice is fixed across the band (variable-cell NEB is out of
    # scope; every image must carry the reactant's cell).
    L0 = np.asarray(result.path.images[0].system.lattice, dtype=float)
    for img in result.path.images[1:]:
        Li = np.asarray(img.system.lattice, dtype=float)
        assert np.allclose(Li, L0), "lattice drifted across NEB images"

    # The highest-energy image is an interior one, not an endpoint.
    ts = result.transition_state_index
    assert ts is not None
    assert 1 <= ts <= n_total - 2


# ---------------------------------------------------------------------------
# Test 2 — vibe-qc periodic energy profile agrees with CP2K (smoke-level)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not runner_cp2k.is_available(),
    reason="CP2K executable not on PATH (set VIBEQC_CP2K_EXECUTABLE); "
           "see module docstring for the compute-reference workflow",
)
def test_periodic_neb_energy_profile_matches_cp2k(periodic_neb_result, tmp_path):
    result = periodic_neb_result

    cp2k_energies, notes = _cp2k_energies_along_band(result, tmp_path=tmp_path)
    if cp2k_energies is None:
        pytest.skip(
            "CP2K present but did not return a full energy profile "
            f"(check data files / element support): {notes}"
        )

    vq_e = np.asarray(result.energies, dtype=float)
    cp_e = np.asarray(cp2k_energies, dtype=float)

    # Compare the *relative* profile ΔEᵢ = Eᵢ − E₀ (the absolute zero
    # differs by the all-electron vs pseudopotential core treatment and
    # is not comparable; the shape along the path is).
    vq_rel = vq_e - vq_e[0]
    cp_rel = cp_e - cp_e[0]
    dev = np.abs(vq_rel - cp_rel)
    max_dev = float(np.max(dev))

    # Informational: a live CP2K run reveals the achievable agreement,
    # so a future matched-basis / GAPW setup can calibrate a tighter
    # SMOKE_TOL_HA. Visible with `pytest -s`.
    print(
        "\nperiodic-NEB CP2K parity (relative profile, Ha):"
        f"\n  vibe-qc ΔE: {np.round(vq_rel, 6).tolist()}"
        f"\n  cp2k    ΔE: {np.round(cp_rel, 6).tolist()}"
        f"\n  max |Δ|  : {max_dev:.6f} Ha   (smoke tol {SMOKE_TOL_HA} Ha)"
    )

    assert max_dev < SMOKE_TOL_HA, (
        f"vibe-qc and CP2K relative energy profiles disagree by "
        f"{max_dev:.4f} Ha > {SMOKE_TOL_HA} Ha smoke tolerance. This is "
        f"the § 7 alarm: a Madelung / gauge / image-summing bug in the "
        f"periodic SCF would show up here as a geometry-dependent shift. "
        f"Investigate against PySCF.pbc.GDF on the same geometries before "
        f"papering over with convergence aids."
    )


# ---------------------------------------------------------------------------
# PRODUCTION escalation — the real surface-NEB parity (not yet runnable)
# ---------------------------------------------------------------------------


def test_h2_pt111_barrier_parity_PRODUCTION():
    """Placeholder for the flagship periodic-NEB parity (always skipped).

    The exemplar above exercises the *wiring*. The production anchor the
    surface-reactions flagship actually needs is a real slab barrier,
    which is a multi-hour compute-reference job, not a unit test. The procedure to
    stand it up (the concrete "next step"):

    1. Build the system natively (no ASE):
           slab, info = vq.slab("Pt", "fcc111", n_layers=4,
                                 vacuum=10.0, supercell=(2, 2, 1))
           reactant = vq.place_adsorbate(slab, "H2", site="top")
           product  = (2 H in adjacent fcc/hcp hollows, relaxed)
       freeze the bottom layers via info.bottom_layer_indices(n).
    2. vibe-qc band:
           vq.run_neb(reactant, product, basis="pob-tzvp",
                      functional="pbe", method="RKS", n_images=7,
                      kpoints=(3, 3, 1), climbing_image=True,
                      freeze_indices=info.bottom_layer_indices(2),
                      conv_tol_force=5e-4)
       Submit on compute-reference via vq (CLAUDE.md § 15), `taskset -c 0-3 nice
       -n 19`. Wall time is dominated by the 6N+1-SCF FD gradient —
       this is exactly the cost the J^LR analytic gradient removes.
    3. CP2K reference. Two tiers, in order:
         (a) single-point GAPW (all-electron) on each vibe-qc image —
             needs `runner_cp2k` extended to METHOD GAPW + all-electron
             basis (its docstring marks this "M3+ scope"). Matched
             all-electron Hamiltonians let you tighten past SMOKE_TOL_HA
             to chemical accuracy.
         (b) full band-vs-band: a `RUN_TYPE BAND` / `&MOTION/&BAND` deck
             builder in `runner_cp2k`, comparing CP2K's own CI-NEB
             barrier to vibe-qc's. The ultimate end-to-end oracle.
    4. Pin the barrier (within ~0.05 eV of published PBE-D, e.g. Watson/
       Wells/Hodgson 2003; Olsen/Kroes 2005) as a `@pytest.mark.slow`
       regression and unblock the roadmap rows
       (`Surface diffusion via NEB`, `NEB + Dimer on Al(110)`).

    See the NEB user guide (test/verification gate) and
    handovers/HANDOVER_SURFACE_REACTIONS.md (Gap 4) for the full plan.
    """
    pytest.skip(
        "production H₂/Pt(111) periodic-NEB parity is a compute-reference job + "
        "needs runner_cp2k GAPW/BAND support; see this test's docstring "
        "and docs/user_guide/neb.md"
    )
