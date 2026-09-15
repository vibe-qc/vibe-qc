"""GitLab #571: the analytic RKS/UKS gradient omits the range-separated
exact-exchange and VV10 terms; consumers must refuse or fall back to FD.

Measured at b151d589a (before the fix), bent O/H/H, STO-3G, analytic
:class:`MolecularSCFProvider` gradient against the optimizer's own
full-energy central finite differences (step 0.005 bohr,
OMP_NUM_THREADS=1), max |analytic - FD| in Ha/bohr with max |g| ~ 0.1:

    pbe      2.1e-6   (control: the FD noise floor)
    hse06    5.6e-3
    vv10     7.0e-4
    wb97x-v  5.6e-2

``cpp/src/gradient.cpp`` differentiates exchange through
``Functional.hf_exchange_fraction()`` (= cam_alpha) only: no
``cam_beta * dK_erf(omega)/dR`` and no VV10 nonlocal gradient (Vydrov and
Van Voorhis, J. Chem. Phys. 133, 244103 (2010), doi:10.1063/1.3521275),
while ``rks.cpp`` includes both in the energy.

Fail-first contract (LEARNINGS L124): at the parent the refusal tests fail
*behaviourally* -- the wrapper returns a gradient and the test measures how
far it sits from the FD gradient -- not on a missing symbol. The negative
control (L125) is the same route with the feature off: PBE through the same
provider and the same wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import Atom, Molecule, run_job
from vibeqc.geomopt.providers import MolecularSCFProvider
from vibeqc.molecular_optimize import (
    _compute_molecular_gradient,
    _gradient_via_central_difference,
)

BASIS = "sto-3g"
FD_STEP = 0.005
# Tolerance for "same surface": the measured PBE analytic-vs-FD noise floor
# is 2.1e-6 Ha/bohr on this molecule; the defect is 3 to 4 orders larger.
SAME_SURFACE_TOL = 1e-5
# Issue-table discrepancies at b151d589a, quoted in the fail-first message.
MEASURED_AT_PARENT = {"hse06": 5.6e-3, "vv10": 7.0e-4, "wb97x-v": 5.6e-2}


def _ohh() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.8]),
            Atom(1, [1.7, 0.0, -0.6]),
        ]
    )


def _h2(r: float = 1.5) -> Molecule:
    return Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, r])])


def _oh_radical() -> Molecule:
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.85])],
        charge=0,
        multiplicity=2,
    )


def _run_rks(mol: Molecule, functional: str):
    basis = vq.BasisSet(mol, BASIS)
    opts = vq.RKSOptions()
    opts.functional = functional
    res = vq.run_rks(mol, basis, opts)
    assert res.converged
    return basis, opts, res


def _run_uks(mol: Molecule, functional: str):
    basis = vq.BasisSet(mol, BASIS)
    opts = vq.UKSOptions()
    opts.functional = functional
    res = vq.run_uks(mol, basis, opts)
    assert res.converged
    return basis, opts, res


def _fd(mol: Molecule, method: str, functional: str, opts) -> np.ndarray:
    kw = {"rks_options": opts} if method == "rks" else {"uks_options": opts}
    return np.asarray(
        _gradient_via_central_difference(
            mol, BASIS, method, functional=functional, step_bohr=FD_STEP, **kw
        ),
        dtype=float,
    ).reshape(-1, 3)


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, n_missing",
    [
        ("lda", 0),
        ("pbe", 0),
        ("pbe0", 0),
        ("b3lyp", 0),
        ("tpss", 0),
        ("hse06", 1),
        ("wb97x", 1),
        ("cam-b3lyp", 1),
        ("vv10", 1),
        ("wb97x-v", 2),
        ("wb97m-v", 2),
    ],
)
def test_helper_counts_missing_terms(name, n_missing):
    missing = vq.functional_gradient_terms_missing(name)
    assert len(missing) == n_missing, missing


def test_helper_none_empty_and_unknown_are_complete():
    # None / "" mean "no functional" (RHF); an unknown name is left for the
    # SCF to reject with its own error, not pre-empted by a capability query.
    assert vq.functional_gradient_terms_missing(None) == []
    assert vq.functional_gradient_terms_missing("") == []
    assert vq.functional_gradient_terms_missing("not-a-functional") == []


def test_helper_accepts_options_and_functional_objects():
    opts = vq.RKSOptions()
    opts.functional = "hse06"
    assert len(vq.functional_gradient_terms_missing(opts)) == 1
    assert len(vq.functional_gradient_terms_missing(vq.Functional("wb97x-v"))) == 2


def test_helper_names_the_terms_and_the_vv10_reference():
    missing = vq.functional_gradient_terms_missing("wb97x-v")
    text = " ".join(missing)
    assert "K_erf(omega)" in text and "omega=0.3000" in text
    assert "VV10" in text and "10.1063/1.3521275" in text


# ---------------------------------------------------------------------------
# Public API refuses (fail-first: at the parent these return a wrong gradient)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("functional", ["hse06", "vv10", "wb97x-v"])
def test_compute_gradient_rks_refuses_incomplete_functional(functional):
    mol = _ohh()
    basis, opts, res = _run_rks(mol, functional)
    try:
        g = np.asarray(vq.compute_gradient_rks(mol, basis, res), dtype=float)
    except NotImplementedError as exc:
        msg = str(exc)
        assert functional in msg and "#571" in msg
        if functional in ("hse06", "wb97x-v"):
            assert "long-range exact-exchange" in msg
        if functional in ("vv10", "wb97x-v"):
            assert "VV10" in msg
        assert "allow_incomplete=True" in msg
        return
    # Parent behaviour: the wrapper returned. Measure the wrong surface.
    dev = float(np.abs(g.reshape(-1, 3) - _fd(mol, "rks", functional, opts)).max())
    pytest.fail(
        f"compute_gradient_rks({functional!r}) returned a gradient "
        f"{dev:.2e} Ha/bohr off the full-energy FD gradient (issue table: "
        f"{MEASURED_AT_PARENT[functional]:.1e} at b151d589a) instead of "
        "refusing (#571)"
    )


def test_compute_gradient_uks_refuses_incomplete_functional():
    mol = _oh_radical()
    basis, opts, res = _run_uks(mol, "hse06")
    try:
        g = np.asarray(vq.compute_gradient_uks(mol, basis, res), dtype=float)
    except NotImplementedError as exc:
        assert "hse06" in str(exc) and "long-range exact-exchange" in str(exc)
        return
    dev = float(np.abs(g.reshape(-1, 3) - _fd(mol, "uks", "hse06", opts)).max())
    pytest.fail(
        f"compute_gradient_uks('hse06') returned a gradient {dev:.2e} Ha/bohr "
        "off the full-energy FD gradient instead of refusing (#571)"
    )


def test_internal_gradient_dispatch_fails_closed_for_neb_and_direct_callers():
    # _compute_molecular_gradient calls the C++ binding directly (bypassing
    # the public wrapper); NEB and any direct caller must get the refusal.
    mol = _h2()
    basis, opts, res = _run_rks(mol, "hse06")
    try:
        g = np.asarray(_compute_molecular_gradient(mol, basis, res, "rks"))
    except NotImplementedError as exc:
        assert "hse06" in str(exc)
        return
    dev = float(np.abs(g.reshape(-1, 3) - _fd(mol, "rks", "hse06", opts)).max())
    pytest.fail(
        f"_compute_molecular_gradient(rks, hse06) returned a gradient "
        f"{dev:.2e} Ha/bohr off FD instead of refusing (#571)"
    )


def test_allow_incomplete_opts_into_the_partial_gradient():
    mol = _h2()
    basis, _opts, res = _run_rks(mol, "hse06")
    g = np.asarray(
        vq.compute_gradient_rks(mol, basis, res, allow_incomplete=True), dtype=float
    )
    assert g.shape == (2, 3) and np.all(np.isfinite(g))


def test_complete_functional_is_not_refused_same_route():
    # Same wrapper, feature off: a global hybrid keeps its analytic gradient.
    mol = _h2()
    basis, _opts, res = _run_rks(mol, "pbe0")
    g = np.asarray(vq.compute_gradient_rks(mol, basis, res), dtype=float)
    assert g.shape == (2, 3) and np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# Optimizer provider: negative control and fallback
# ---------------------------------------------------------------------------


def test_provider_pbe_control_analytic_matches_fd():
    # Negative control (L125): the same provider route with the feature off.
    mol = _ohh()
    prov = MolecularSCFProvider(BASIS, method="rks", functional="pbe")
    assert prov.has_analytic_gradient is True
    _e, g = prov(mol)
    opts = prov._rks_options
    dev = float(np.abs(g.reshape(-1, 3) - _fd(mol, "rks", "pbe", opts)).max())
    assert dev <= SAME_SURFACE_TOL, dev


def test_provider_falls_back_to_fd_for_wb97x_v():
    mol = _ohh()
    prov = MolecularSCFProvider(BASIS, method="rks", functional="wb97x-v")
    e, g = prov(mol)
    # Numeric assertion first so a parent run fails on the number, not on
    # the capability flag: at b151d589a the analytic gradient sits 5.6e-2
    # Ha/bohr off this independent FD gradient.
    opts = prov._rks_options
    g_fd = _fd(mol, "rks", "wb97x-v", opts)
    dev = float(np.abs(g.reshape(-1, 3) - g_fd).max())
    assert dev <= SAME_SURFACE_TOL, (
        f"provider gradient is {dev:.2e} Ha/bohr off the full-energy FD "
        "gradient (#571)"
    )
    assert prov.has_analytic_gradient is False
    assert len(prov.missing_gradient_terms) == 2
    # The energy is the same SCF energy the analytic branch would report.
    basis = vq.BasisSet(mol, BASIS)
    ref = vq.run_rks(mol, basis, opts)
    assert abs(e - float(ref.energy)) < 1e-8


def test_all_three_optimizers_share_the_decision():
    from vibeqc.molecular_optimize import _mean_field_gradient_terms_missing

    for method, functional, n in (
        ("rks", "pbe", 0),
        ("rks", "hse06", 1),
        ("uks", "wb97x-v", 2),
        ("rhf", None, 0),
        ("uhf", None, 0),
    ):
        assert len(_mean_field_gradient_terms_missing(method, functional, None, None)) == n
    # An explicit options functional wins over a default-LDA options struct,
    # exactly as _run_molecular_scf resolves it.
    opts = vq.RKSOptions()
    opts.functional = "wb97x"
    assert len(_mean_field_gradient_terms_missing("rks", None, opts, None)) == 1
    assert len(_mean_field_gradient_terms_missing("rks", "pbe", opts, None)) == 1


def test_run_job_wb97x_v_optimize_converges_on_the_fd_surface(tmp_path):
    pytest.importorskip("ase")
    res = run_job(
        _h2(),
        basis=BASIS,
        method="rks",
        functional="wb97x-v",
        optimize=True,
        max_opt_steps=25,
        output=tmp_path / "h2-wb97xv-opt",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert res.converged
    # The last trajectory frame is a stationary point of the FULL energy
    # (the FD surface), which the partial analytic gradient would not find.
    # This numeric check comes first so a parent run fails on the wrong
    # surface, not on the missing note.
    from ase.io.trajectory import Trajectory
    from ase.units import Bohr

    frames = list(Trajectory(str(tmp_path / "h2-wb97xv-opt.traj")))
    assert frames, "optimizer wrote no trajectory frames"
    last = frames[-1]
    final = Molecule(
        [Atom(int(z), list(xyz)) for z, xyz in zip(last.numbers, last.positions / Bohr)]
    )
    opts = vq.RKSOptions()
    opts.functional = "wb97x-v"
    g_fd = _fd(final, "rks", "wb97x-v", opts)
    assert float(np.abs(g_fd).max()) < 2e-3, (
        f"optimized geometry is not a stationary point of the full wb97x-v "
        f"energy: max |FD gradient| = {float(np.abs(g_fd).max()):.2e} Ha/bohr (#571)"
    )
    out_text = (tmp_path / "h2-wb97xv-opt.out").read_text()
    assert "GitLab #571" in out_text
    assert "full-energy central finite differences" in out_text
    assert "VV10" in out_text and "long-range exact-exchange" in out_text
