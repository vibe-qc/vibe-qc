"""IRC — intrinsic reaction coordinate (mass-weighted reaction-path following).

Pins: the core stabilized steepest descent traces a known analytic path to
both minima (monotonic energy); ``run_irc`` from the planar NH3 transition
state descends to the two pyramidal minima on opposite sides; the
transition vector can be auto-computed (FD Hessian) or supplied; the IRC
citation fires; and the error paths behave.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.irc import IRCResult, _irc_branch, run_irc

ANG = 1.8897259886


# ---------------------------------------------------------------------------
# Core integrator on an analytic potential
# ---------------------------------------------------------------------------


def test_core_traces_both_branches_to_minima():
    # E = -cos(x) + y^2 : saddle at (pi, 0); minima at x=0 and x=2pi.
    def gf(X):
        xx, yy, _ = X[0]
        return -np.cos(xx) + yy * yy, np.array([[np.sin(xx), 2 * yy, 0.0]])

    x_ts = np.array([[np.pi, 1e-4, 0.0]])
    masses = np.array([1.0])
    mode = np.array([[1.0, 0.0, 0.0]])
    for sign, expect in ((+1.0, 2 * np.pi), (-1.0, 0.0)):
        pos, en, conv = _irc_branch(x_ts, gf, masses, mode, sign,
                                    step_size=0.1, max_points=1000,
                                    conv_tol_grad=1e-4)
        assert conv
        assert abs(pos[-1][0, 0] - expect) < 1e-2
        assert all(np.diff(en) <= 1e-9)             # energy monotonically down


# ---------------------------------------------------------------------------
# Molecular IRC: NH3 umbrella inversion
# ---------------------------------------------------------------------------


def _planar_nh3() -> vq.Molecule:
    r = 1.012 * ANG
    at = [vq.Atom(7, [0, 0, 0])]
    for k in range(3):
        p = k * 2 * np.pi / 3
        at.append(vq.Atom(1, [r * np.cos(p), r * np.sin(p), 0.0]))  # planar D3h
    return vq.Molecule(at, 0, 1)


def _h_mean_z(system) -> float:
    return float(np.mean([list(a.xyz)[2] for a in system.atoms[1:]]))


def _h2_cation() -> vq.Molecule:
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, -0.7]), vq.Atom(1, [0.0, 0.0, 0.7])],
        1,
        2,
    )


def test_run_irc_nh3_auto_mode():
    # transition_mode=None → FD Hessian gives the imaginary inversion mode.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_irc(_planar_nh3(), basis="sto-3g", method="RHF",
                      step_size=0.1, max_points=80, conv_tol_grad=2e-3)
    assert res.forward_converged and res.reverse_converged
    fz, rz = _h_mean_z(res.forward_minimum), _h_mean_z(res.reverse_minimum)
    assert abs(fz) > 0.2 and abs(rz) > 0.2          # pyramidalised
    assert fz * rz < 0                              # opposite sides
    assert res.forward_energies[-1] < res.ts_energy
    assert res.reverse_energies[-1] < res.ts_energy
    assert all(np.diff(res.forward_energies) <= 1e-9)
    assert all(np.diff(res.reverse_energies) <= 1e-9)
    # Symmetric inversion: the two barriers match.
    assert res.barrier_forward == pytest.approx(res.barrier_reverse, abs=1e-3)


def test_run_irc_explicit_mode_single_branch():
    # Supplying transition_mode skips the Hessian; one branch only.
    mode = np.zeros((4, 3))
    mode[1:, 2] = 1.0  # H's out of plane (inversion)
    res = run_irc(_planar_nh3(), basis="sto-3g", method="RHF",
                  transition_mode=mode, direction="forward",
                  step_size=0.1, max_points=80, conv_tol_grad=2e-3)
    assert res.forward_converged and len(res.reverse_path) == 0
    assert abs(_h_mean_z(res.forward_minimum)) > 0.2


@pytest.mark.parametrize(
    ("method", "functional"),
    [("ROHF", None), ("ROKS", "lda")],
)
def test_run_irc_restricted_open_shell_explicit_mode(tmp_path, method, functional):
    mode = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    kwargs = {"functional": functional} if functional is not None else {}
    if method == "ROHF":
        kwargs["rohf_options"] = vq.ROHFOptions()
    else:
        kwargs["roks_options"] = vq.ROKSOptions()
        kwargs["fd_step_bohr"] = 5e-4
    res = run_irc(
        _h2_cation(),
        basis="sto-3g",
        method=method,
        transition_mode=mode,
        direction="forward",
        step_size=0.02,
        max_points=0,
        **kwargs,
    )
    assert len(res.forward_path) == 1
    assert len(res.reverse_path) == 0
    assert np.isfinite(res.ts_energy)
    assert np.all(np.isfinite(res.forward_energies))
    assert res.functional == functional

    bib_path, _ = res.write_citations(tmp_path / method.lower())
    citations = bib_path.read_text()
    assert "fukui_irc_1981" in citations
    assert "roothaan_rohf_1960" in citations


def test_run_irc_rohf_auto_mode():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_irc(
            _h2_cation(),
            basis="sto-3g",
            method="ROHF",
            direction="forward",
            max_points=0,
        )
    assert len(res.forward_path) == 1
    assert np.isfinite(res.ts_energy)
    assert res.functional is None


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_irc_citation_route_fires():
    from vibeqc.output.citations import load_default_database
    refs = load_default_database().assemble(method="RHF", basis="sto-3g",
                                            uses_irc=True)
    keys = {getattr(c, "bibtex_key", getattr(c, "key", "")) for c in refs.printable}
    assert "fukui_irc_1981" in keys
    assert "ishida_morokuma_komornicki_irc_1977" in keys


def test_irc_write_citations(tmp_path):
    res = IRCResult(transition_state=_planar_nh3(), ts_energy=-1.0,
                    forward_path=[_planar_nh3()], forward_energies=np.array([-1.1]),
                    reverse_path=[_planar_nh3()], reverse_energies=np.array([-1.1]),
                    forward_converged=True, reverse_converged=True,
                    method="RHF", basis="sto-3g", functional=None)
    bib_path, _ = res.write_citations(str(tmp_path / "irc"))
    blob = open(bib_path).read()
    assert "fukui_irc_1981" in blob and "1981" in blob


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_run_irc_basis_required_for_scf():
    with pytest.raises(ValueError, match="basis set is required"):
        run_irc(_planar_nh3(), method="RHF")


def test_run_irc_mace_requires_transition_mode():
    with pytest.raises(ValueError, match="transition_mode"):
        run_irc(_planar_nh3(), method="mace")          # no mode + no SCF Hessian


def test_run_irc_roks_requires_transition_mode():
    with pytest.raises(NotImplementedError, match="explicit transition_mode"):
        run_irc(_h2_cation(), basis="sto-3g", method="ROKS", functional="lda")


@pytest.mark.parametrize(
    "mode",
    [
        np.ones((1, 3)),
        np.ones(6),
        np.zeros((2, 3)),
        np.full((2, 3), np.nan),
        np.full((2, 3), np.inf),
        np.full((2, 3), 1.0 + 1.0j),
    ],
    ids=[
        "wrong-atom-count",
        "flattened",
        "zero-norm",
        "nan",
        "infinity",
        "complex",
    ],
)
def test_run_irc_rejects_invalid_transition_mode_before_force_setup(
    monkeypatch,
    mode,
):
    import importlib

    irc_module = importlib.import_module("vibeqc.irc")
    build_calls = 0

    def forbidden_force_builder(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("force-provider setup must not run")

    monkeypatch.setattr(irc_module, "_build_force_fn", forbidden_force_builder)

    with pytest.raises(ValueError, match="transition_mode"):
        run_irc(
            _h2_cation(),
            method="mace",
            transition_mode=mode,
            direction="forward",
            max_points=0,
        )

    assert build_calls == 0


@pytest.mark.parametrize(
    "mode_scale",
    [7.0, 1.0e300, 1.0e-300],
    ids=["ordinary", "large", "small"],
)
def test_run_irc_accepts_finite_nonzero_transition_mode(
    monkeypatch,
    mode_scale,
):
    import importlib

    irc_module = importlib.import_module("vibeqc.irc")
    evaluated_positions = []

    def force_provider(coords):
        positions = np.asarray(coords, dtype=float)
        evaluated_positions.append(positions.copy())
        return float(np.sum(positions * positions)), -2.0 * positions

    monkeypatch.setattr(
        irc_module,
        "_build_force_fn",
        lambda *args, **kwargs: (force_provider, ""),
    )

    molecule = _h2_cation()
    unit_mode = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
    ) / np.sqrt(2.0)
    result = run_irc(
        molecule,
        method="mace",
        transition_mode=(mode_scale * unit_mode).tolist(),
        direction="forward",
        max_points=0,
    )

    assert len(evaluated_positions) == 2
    initial_positions = np.array(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
    )
    assert evaluated_positions[1] == pytest.approx(
        initial_positions + 0.1 * unit_mode,
    )
    assert result.forward_energies.shape == (1,)
    assert np.all(np.isfinite(result.forward_energies))


def test_run_irc_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        run_irc(_planar_nh3(), basis="sto-3g", method="RHF",
                transition_mode=np.zeros((4, 3)), direction="sideways")


def test_run_irc_periodic_not_supported():
    L = np.diag([8.0, 8.0, 8.0])
    sysp = vq.PeriodicSystem(3, L, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    with pytest.raises(NotImplementedError, match="only molecular"):
        run_irc(sysp, basis="sto-3g", method="RHF")


def test_run_irc_forwards_gradient_options_to_the_ts_hessian(monkeypatch):
    """#576: ``run_irc`` used to call ``compute_hessian_fd`` without its
    ``gradient_options``, so the TS Hessian differentiated a
    default-constructed (bare-Z, direct four-index) gradient while the IRC
    path itself ran with the caller's options. The kwarg is now forwarded
    verbatim -- ``None`` included, which lets ``compute_hessian_fd`` derive
    the gradient from the SCF options (JK backend + ECP fields)."""
    import importlib

    irc_module = importlib.import_module("vibeqc.irc")
    hessian_module = importlib.import_module("vibeqc.hessian")
    seen: list = []

    def fake_hessian(mol, basis, **kwargs):
        seen.append(kwargs)
        n_dof = 3 * len(mol.atoms)
        modes = np.eye(n_dof)
        return SimpleNamespace(
            frequencies_cm1=np.full(n_dof, -500.0),
            normal_modes=modes,
            masses_amu=np.ones(len(mol.atoms)),
        )

    def force_provider(coords):
        positions = np.asarray(coords, dtype=float)
        return float(np.sum(positions * positions)), -2.0 * positions

    monkeypatch.setattr(hessian_module, "compute_hessian_fd", fake_hessian)
    monkeypatch.setattr(
        irc_module,
        "_build_force_fn",
        lambda *args, **kwargs: (force_provider, ""),
    )

    go = vq.GradientOptions()
    go.density_fit = True
    go.aux_basis = "def2-universal-jfit"
    rhf = vq.RHFOptions()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_irc(_h2_cation(), basis="sto-3g", method="RHF",
                rhf_options=rhf, gradient_options=go,
                direction="forward", max_points=0)
        run_irc(_h2_cation(), basis="sto-3g", method="RHF",
                rhf_options=rhf, direction="forward", max_points=0)

    assert len(seen) == 2
    assert seen[0]["gradient_options"] is go
    assert seen[0]["scf_options"] is rhf
    # No caller object: the kwarg is present and None, so compute_hessian_fd
    # derives the gradient from the SCF options instead of bare defaults.
    assert "gradient_options" in seen[1] and seen[1]["gradient_options"] is None


def test_run_irc_on_an_ecp_system_runs_the_ecp_hamiltonian_and_restores_options():
    """#643: the IRC path forces go through the shared image evaluator, which
    now moves the ECP centres onto every visited geometry and runs the
    ECP-aware SCF (pre-fix: an all-electron warm-start Hamiltonian, E of
    order -1777 Ha on this system instead of -224 Ha). Explicit transition
    mode, two steps, so this is a smoke of the wiring, not a real IRC."""
    o = vq.RHFOptions()
    o.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    o.ecp_library = "ecp10mdf"
    o.max_iter = 200
    o.damping = 0.5
    ts = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 3.2])], 1, 1)
    mode = np.zeros((2, 3))
    mode[1, 2] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_irc(ts, basis="6-31g", method="RHF", rhf_options=o,
                      transition_mode=mode, direction="forward",
                      step_size=0.05, max_points=2)
    assert -230.0 < res.ts_energy < -220.0, res.ts_energy
    assert all(-230.0 < e < -220.0 for e in res.forward_energies)
    assert list(o.ecp_centers[0].xyz) == [0.0, 0.0, 0.0]


def test_run_irc_refuses_ecp_centres_off_the_transition_state():
    o = vq.RHFOptions()
    o.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.05])]
    o.ecp_library = "ecp10mdf"
    ts = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 3.2])], 1, 1)
    mode = np.zeros((2, 3))
    mode[1, 2] = 1.0
    with pytest.raises(ValueError, match="coincides with no atom"):
        run_irc(ts, basis="6-31g", method="RHF", rhf_options=o,
                transition_mode=mode, direction="forward", max_points=1)
