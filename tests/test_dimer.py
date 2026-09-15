"""Dimer method — single-ended saddle-point search (Henkelman-Jónsson 1999).

Pins: the core algorithm finds a known analytic saddle (and reports
positive curvature at a minimum); ``run_dimer`` finds the NH3 planar
inversion TS with vibe-qc SCF forces (planar, one negative curvature, no
centre-of-mass drift); the dimer-method citation fires; and the error
paths + MACE dispatch behave.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.dimer as dimer
from vibeqc.dimer import DimerResult, _dimer_search, run_dimer


def _core(force_fn, x0, N0, **kw):
    opts = dict(dimer_separation=1e-3, max_iter=600, conv_tol_force=1e-5,
                max_step=0.2, initial_step=0.05, n_rotations=4,
                rotation_force_tol=1e-7, frozen_mask=None, progress=False,
                project_translation=False)
    opts.update(kw)
    return _dimer_search(np.asarray(x0, float), force_fn,
                         initial_direction=np.asarray(N0, float), **opts)


# ---------------------------------------------------------------------------
# Core algorithm on analytic potentials
# ---------------------------------------------------------------------------


def test_core_finds_analytic_saddle():
    # E = -cos(x) + y^2 : first-order saddle at (pi, 0), min-mode x̂.
    def ff(x):
        xx, yy, _ = x[0]
        return -np.cos(xx) + yy * yy, -np.array([[np.sin(xx), 2 * yy, 0.0]])

    R, N, C, E, mF, conv, nit, nev = _core(ff, [[np.pi - 0.4, 0.5, 0.0]],
                                           [[1.0, 0.3, 0.0]])
    assert conv and C < 0                       # converged to a saddle
    assert abs(R[0, 0] - np.pi) < 1e-3 and abs(R[0, 1]) < 1e-3
    assert abs(abs(N[0, 0]) - 1.0) < 1e-2       # dimer axis ≈ x̂ (negative mode)


def test_core_rotates_to_min_mode_from_biased_start():
    # Even an initial axis biased toward the high-curvature (y) mode must
    # rotate to the low-curvature (x) mode.
    def ff(x):
        xx, yy, _ = x[0]
        return -np.cos(xx) + yy * yy, -np.array([[np.sin(xx), 2 * yy, 0.0]])

    R, N, C, *_ = _core(ff, [[np.pi - 0.5, -0.6, 0.0]], [[0.2, 1.0, 0.0]])
    assert C < 0 and abs(abs(N[0, 0]) - 1.0) < 1e-2


def test_core_does_not_report_saddle_on_convex_bowl():
    # Pure minimum E = x^2 + 2y^2 + 3z^2 : no negative mode. The dimer
    # climbs out looking for one (correct) — it must NOT falsely report a
    # converged first-order saddle.
    def ff(x):
        xx, yy, zz = x[0]
        return xx*xx + 2*yy*yy + 3*zz*zz, -np.array([[2*xx, 4*yy, 6*zz]])

    R, N, C, E, mF, conv, *_ = _core(ff, [[0.3, 0.4, 0.2]], [[1.0, 0.1, 0.0]],
                                     max_iter=200)
    assert not (conv and C < 0.0)               # no spurious saddle on a bowl


def test_core_progress_preserves_caller_stream_without_stdout(capsys):
    stream = io.StringIO()

    def ff(x):
        return float(np.sum(x * x)), -2.0 * x

    _core(
        ff,
        [[0.3, 0.2, 0.1]],
        [[1.0, 0.0, 0.0]],
        max_iter=1,
        progress=stream,
    )

    assert capsys.readouterr().out == ""
    assert "dimer iter" in stream.getvalue()


# ---------------------------------------------------------------------------
# Molecular integration: NH3 umbrella-inversion transition state
# ---------------------------------------------------------------------------


def _nh3(scale: float) -> vq.Molecule:
    th = np.deg2rad(106.67)
    r = 1.012 * 1.8897259886
    sa = np.sqrt(2 * (1 - np.cos(th)) / 3)
    ca = np.sqrt(1 - sa ** 2)
    at = [vq.Atom(7, [0, 0, 0])]
    for k in range(3):
        p = k * 2 * np.pi / 3
        at.append(vq.Atom(1, [r*sa*np.cos(p), r*sa*np.sin(p), -scale*r*ca]))
    return vq.Molecule(at, 0, 1)


def _h2plus(distance: float = 1.4) -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, -0.5 * distance]),
            vq.Atom(1, [0.0, 0.0, 0.5 * distance]),
        ],
        1,
        2,
    )


@pytest.mark.parametrize("method", ["ROHF", "ROKS"])
def test_run_dimer_restricted_open_shell_public_route(method, tmp_path):
    result = run_dimer(
        _h2plus(),
        basis="sto-3g",
        method=method,
        functional="lda",
        initial_direction=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
        dimer_separation=1.0e-3,
        max_iter=1,
        n_rotations=1,
        fd_step_bohr=1.0e-3,
    )

    assert np.isfinite(result.energy)
    assert np.isfinite(result.curvature)
    assert result.n_force_evals >= 2
    assert result.method == method
    assert result.functional == ("lda" if method == "ROKS" else None)
    bib_path, ref_path = result.write_citations(tmp_path / method.lower())
    citation_text = bib_path.read_text() + ref_path.read_text()
    assert "henkelman" in citation_text.lower()
    assert "roothaan" in citation_text.lower()


def test_run_dimer_nh3_inversion_ts():
    start = _nh3(0.25)                          # near-planar, on the saddle side
    z0 = float(np.mean([list(a.xyz)[2] for a in start.atoms]))
    direction = np.zeros((4, 3))
    direction[1:, 2] = 1.0   # flatten the H's
    res = run_dimer(start, basis="sto-3g", method="RHF",
                    initial_direction=direction, conv_tol_force=1e-3,
                    max_iter=200, max_step=0.15)
    z = np.array([float(list(a.xyz)[2]) for a in res.system.atoms])
    assert res.converged and res.is_saddle
    assert res.curvature < 0.0
    assert (z.max() - z.min()) < 0.02          # planar D3h TS
    assert abs(z.mean() - z0) < 0.05           # no centre-of-mass drift


# ---------------------------------------------------------------------------
# Citations (CLAUDE.md §8)
# ---------------------------------------------------------------------------


def test_dimer_citation_route_fires():
    from vibeqc.output.citations import load_default_database
    refs = load_default_database().assemble(method="RHF", basis="sto-3g",
                                            uses_dimer=True)
    keys = {getattr(c, "bibtex_key", getattr(c, "key", "")) for c in refs.printable}
    assert "henkelman_jonsson_dimer_1999" in keys


def test_dimer_write_citations(tmp_path):
    res = DimerResult(system=_nh3(0.0), energy=-1.0, max_force=1e-4,
                      curvature=-0.1, mode=np.zeros((4, 3)), converged=True,
                      is_saddle=True, n_iter=1, n_force_evals=1,
                      method="RHF", basis="sto-3g", functional=None)
    bib_path, ref_path = res.write_citations(str(tmp_path / "saddle"))
    blob = open(bib_path).read()
    assert "henkelman_jonsson_dimer_1999" in blob
    assert "1999" in blob and "imer" in blob


# ---------------------------------------------------------------------------
# Error paths + MACE dispatch
# ---------------------------------------------------------------------------


def test_run_dimer_basis_required_for_scf():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
    with pytest.raises(ValueError, match="basis set is required"):
        run_dimer(mol, method="RHF")


def test_run_dimer_mace_rejects_dft_plus_u():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
    with pytest.raises(ValueError, match="dft_plus_u is not supported"):
        run_dimer(mol, method="mace", dft_plus_u=[object()])


@pytest.mark.parametrize(
    "kpoints_factory",
    [
        lambda sysp: vq.monkhorst_pack(sysp, [1, 1, 1]),
        lambda sysp: vq.KPoints.gamma(sysp),
    ],
    ids=["native-bloch-kmesh", "kpoints-object"],
)
def test_run_dimer_periodic_accepts_materialized_kmesh(
    monkeypatch, kpoints_factory
):
    """Periodic dimer force evaluations must preserve object-style k-meshes."""
    initial = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [4.0, 4.0, 4.0]), vq.Atom(1, [5.4, 4.0, 4.0])],
        charge=0,
        multiplicity=1,
    )
    user_kpoints = kpoints_factory(initial)
    expected = vq.as_bloch_kmesh(user_kpoints)
    captured = []

    def fake_evaluate_image_periodic(
        positions,
        template,
        basis_name,
        method,
        *,
        kmesh,
        **kwargs,
    ):
        captured.append(kmesh)
        grad = np.asarray(positions, dtype=float) * 0.01
        energy = 0.5 * float(np.sum(np.asarray(positions, dtype=float) ** 2))
        return energy, grad, None

    monkeypatch.setattr(
        dimer,
        "_evaluate_image_periodic",
        fake_evaluate_image_periodic,
    )

    result = run_dimer(
        initial,
        basis="sto-3g",
        method="RHF",
        kpoints=user_kpoints,
        initial_direction=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
        max_iter=1,
        n_rotations=1,
        dimer_separation=1e-3,
    )

    assert isinstance(result, DimerResult)
    assert captured
    for kmesh in captured:
        np.testing.assert_allclose(
            np.asarray(kmesh.kpoints, dtype=float),
            np.asarray(expected.kpoints, dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(kmesh.weights, dtype=float),
            np.asarray(expected.weights, dtype=float),
        )
    if isinstance(user_kpoints, vq.BlochKMesh):
        assert all(kmesh is user_kpoints for kmesh in captured)


def test_run_dimer_mace_dispatch(monkeypatch):
    # Inject a translation/rotation-invariant bond-stretch saddle as a mock
    # ASE calculator (E = -(r - r0)^2, a barrier at r = r0), so run_dimer's
    # MACE path (no SCF/basis) is exercised end to end without torch.
    from ase.calculators.calculator import Calculator, all_changes

    R0_ANG = 0.8

    class _BondSaddle(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            p = atoms.get_positions()            # Angstrom
            d = p[1] - p[0]
            r = float(np.linalg.norm(d))
            u = d / r
            e = -(r - R0_ANG) ** 2               # max (barrier) at r = R0
            f1 = 2.0 * (r - R0_ANG) * u          # force = -dE/dR
            self.results = {"energy": float(e),
                            "forces": np.array([-f1, f1])}

    def _fake_loader(template, mlip_options, is_periodic):
        cell = np.asarray(template.lattice, float) if is_periodic else None
        return (_BondSaddle(), dimer._atomic_numbers_of(template), cell,
                "batatia_mace_mp_2024")

    monkeypatch.setattr(dimer, "_load_mace_model", _fake_loader)
    # H2 with the bond off the barrier top; dimer should climb to r = R0.
    start = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 2.0])], 0, 1)
    res = run_dimer(start, method="mace",
                    initial_direction=np.array([[0, 0, -1.0], [0, 0, 1.0]]),
                    dimer_separation=1e-3, conv_tol_force=1e-4, max_iter=400,
                    max_step=0.2)
    assert res.method == "mace" and res.mace_model_citation == "batatia_mace_mp_2024"
    assert res.converged and res.is_saddle      # found the bond-stretch barrier
    bohr = 1.8897259886
    r_final = np.linalg.norm(np.array(list(res.system.atoms[1].xyz))
                             - np.array(list(res.system.atoms[0].xyz)))
    assert abs(r_final / bohr - R0_ANG) < 0.02  # converged to r = R0 (Å)
