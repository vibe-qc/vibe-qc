"""MSINDO semiempirical backend for ``run_neb`` (method="msindo").

MSINDO (vibe-qc's own Bredow/Geudtner/Jug INDO re-implementation;
``vibeqc.semiempirical.methods.msindo``) supplies a molecular total energy
plus a finite-difference nuclear gradient with no Gaussian basis and no
libint. ``run_neb(method="msindo")`` is the first *semiempirical* image path:
like MACE it needs no basis / functional / k-mesh, but unlike MACE it has no
reusable live object — ``run_msindo`` / ``msindo_gradient_fd`` are stateless
module-level functions, so the band evaluates in parallel across images.

The validation system is the canonical NH3 umbrella inversion (the same
reaction the SCF + MACE NEB examples use): a symmetric isomerization whose
transition state is the planar D3h structure with a *single* imaginary mode
(the umbrella bend). MSINDO supports H + N (s/p), so the whole band runs on
the validated engine.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.neb as neb
from vibeqc.semiempirical.methods.msindo import (
    ANGSTROM_TO_BOHR as A2B,
)
from vibeqc.semiempirical.methods.msindo import (
    msindo_gradient_fd,
    run_msindo,
)

KCAL_PER_HA = 627.509474
_Z_NH3 = [7, 1, 1, 1]


def _nh3(flip: bool = False, theta_deg: float = 106.67, r: float = 1.012):
    """Pyramidal NH3 as a vibe-qc Molecule (bohr); H's below (or above) N.

    ``flip`` mirrors the molecule through the z=0 plane — the product of the
    umbrella inversion, an exact mirror image of the reactant, so the two
    endpoints are degenerate by construction.
    """
    theta = np.deg2rad(theta_deg)
    sin_a = np.sqrt(2.0 * (1.0 - np.cos(theta)) / 3.0)
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    z_sign = +1.0 if flip else -1.0
    coords = [(0.0, 0.0, 0.0)]
    for k in range(3):
        phi = k * 2.0 * np.pi / 3.0
        coords.append((r * sin_a * np.cos(phi),
                       r * sin_a * np.sin(phi),
                       z_sign * r * cos_a))
    atoms = [vq.Atom(z, list(np.array(c, float) * A2B))
             for z, c in zip(_Z_NH3, coords)]
    return vq.Molecule(atoms, 0, 1)


@pytest.fixture(scope="module")
def umbrella():
    """One converged MSINDO umbrella-inversion band, shared across tests.

    Tiny + deterministic: 3 intermediate images, serial (``n_jobs=1``), loose
    force tolerance. Converges in well under 40 outer iterations (~5 s).
    """
    return vq.run_neb(
        _nh3(flip=False), _nh3(flip=True),
        method="msindo", n_images=3, interpolation="idpp",
        max_iter=40, conv_tol_force=5e-3, n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Evaluator: units, convergence guard, dispersion folding
# ---------------------------------------------------------------------------


def test_evaluator_units_match_engine():
    # The NEB evaluator must return exactly what the engine returns: the
    # bohr->Angstrom round-trip uses MSINDO's own constant, so energy and
    # gradient are bit-identical to a direct run_msindo / msindo_gradient_fd.
    coords_ang = np.array([[0.0, 0.0, 0.117],
                           [0.0, 0.757, -0.467],
                           [0.0, -0.757, -0.467]])  # H2O
    Z = [8, 1, 1]
    pos_bohr = coords_ang * A2B
    e, g, dens = neb._evaluate_image_msindo(
        pos_bohr, numbers=np.array(Z), charge=0, multiplicity=1,
        fd_step_bohr=1e-3,
    )
    ref = run_msindo(Z, coords_ang)
    g_ref = msindo_gradient_fd(Z, coords_ang, step=1e-3 / A2B)
    assert e == pytest.approx(ref.total_energy, abs=1e-12)
    assert np.allclose(g, g_ref, atol=1e-12)
    # MSINDO keeps no SCF state across geometries -> no warm-start density.
    assert dens is None


def test_run_neb_nddo_alias_stays_on_the_nddo_surface(monkeypatch, tmp_path):
    """NDDO NEB keeps its analytic surface and citation provenance."""
    import vibeqc.semiempirical.methods.msindo_gradient_analytic as grad_module

    analytic_calls = []
    analytic_gradient = grad_module.msindo_gradient_analytic

    def counted_analytic_gradient(*args, **kwargs):
        analytic_calls.append(1)
        return analytic_gradient(*args, **kwargs)

    def forbidden_fd_gradient(*_args, **_kwargs):
        raise AssertionError("NDDO NEB fell through to the INDO FD evaluator")

    monkeypatch.setattr(
        grad_module,
        "msindo_gradient_analytic",
        counted_analytic_gradient,
    )
    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.msindo.msindo_gradient_fd",
        forbidden_fd_gradient,
    )

    def hf(distance_angstrom):
        return vq.Molecule(
            [
                vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(9, [0.0, 0.0, distance_angstrom * A2B]),
            ],
            0,
            1,
        )

    result = vq.run_neb(
        hf(0.917),
        hf(1.05),
        method="nddo",
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=10.0,
        n_jobs=1,
    )
    expected = run_msindo(
        [1, 9],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]],
        nddo=True,
    )
    assert result.energies[0] == pytest.approx(expected.total_energy, abs=1e-10)
    assert len(analytic_calls) >= 3
    assert result.semiempirical_variant == "nddo"

    qvf = result.write_qvf(str(tmp_path / "msindo_nddo_neb"))
    blob = ""
    with zipfile.ZipFile(qvf) as archive:
        for name in archive.namelist():
            blob += archive.read(name).decode("utf-8", "replace")
    assert "MSINDO-NDDO" in blob
    assert "ahlswede_jug_msindo_1_1999" in blob
    assert "ahlswede_jug_msindo_2_1999" in blob
    assert "dewar_thiel_1977" in blob
    assert "voigt_1973" in blob


def test_evaluator_nonconverged_raises(monkeypatch):
    # A non-converged MSINDO SCF has no valid gradient; the evaluator must
    # raise the clear, image-named NEBImageSCFError before differencing.
    class _Bad:
        converged = False
        total_energy = -1.0
        n_iter = 200

    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.msindo.run_msindo",
        lambda *a, **k: _Bad(),
    )
    pos = np.array([list(a.xyz) for a in _nh3().atoms])
    with pytest.raises(neb.NEBImageSCFError, match="MSINDO SCF did not converge"):
        neb._evaluate_image_msindo(
            pos, numbers=np.array(_Z_NH3), charge=0, multiplicity=1,
            fd_step_bohr=1e-3, image_index=2,
        )


def test_evaluator_dispersion_folded():
    # With dispersion_params, the D3-BJ energy + gradient are added on top of
    # the bare MSINDO energy + FD gradient (which see no dispersion).
    from vibeqc.dispersion import D3BJParams, compute_d3bj

    Z = np.array(_Z_NH3)
    pos = np.array([list(a.xyz) for a in _nh3().atoms])
    params = D3BJParams(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)  # PBE-D3(BJ)
    e0, g0, _ = neb._evaluate_image_msindo(
        pos, numbers=Z, charge=0, multiplicity=1, fd_step_bohr=1e-3)
    e1, g1, _ = neb._evaluate_image_msindo(
        pos, numbers=Z, charge=0, multiplicity=1, fd_step_bohr=1e-3,
        dispersion_params=params)
    mol = vq.Molecule([vq.Atom(int(z), list(c)) for z, c in zip(Z, pos)], 0, 1)
    disp = compute_d3bj(mol, params, with_gradient=True)
    assert (e1 - e0) == pytest.approx(disp.energy, abs=1e-12)
    assert np.allclose(g1 - g0, np.asarray(disp.gradient), atol=1e-12)
    assert e1 < e0  # attractive dispersion lowers the energy


# ---------------------------------------------------------------------------
# Dispatch + the umbrella barrier
# ---------------------------------------------------------------------------


def test_run_neb_msindo_dispatch_no_basis(umbrella):
    # method="msindo" dispatches to the MSINDO evaluator with no basis /
    # functional needed, and produces a finite (n_images + 2) band.
    assert umbrella.method == "msindo"
    assert umbrella.basis is None
    assert len(umbrella.path.images) == 5
    assert np.all(np.isfinite(umbrella.energies))


def test_run_neb_msindo_umbrella_barrier(umbrella):
    e = np.asarray(umbrella.energies)
    ts = umbrella.transition_state_index
    # Symmetric reaction: the product is an exact mirror of the reactant, so
    # the endpoint energies are bit-identical — a strong check that the
    # evaluator is geometry-faithful.
    assert e[0] == pytest.approx(e[-1], abs=1e-10)
    # A genuine, interior barrier (the planar D3h saddle), well above noise.
    assert 0 < ts < len(e) - 1
    assert ts == int(np.argmax(e))
    barrier_kcal = (e[ts] - e[0]) * KCAL_PER_HA
    assert barrier_kcal > 1.0
    # Monotone on each side within FD-gradient noise (1e-4 Ha ~ 0.06 kcal).
    tol = 1e-4
    assert all(e[i] <= e[i + 1] + tol for i in range(ts))
    assert all(e[i] >= e[i + 1] - tol for i in range(ts, len(e) - 1))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_run_neb_msindo_rejects_dft_plus_u():
    with pytest.raises(ValueError, match="dft_plus_u is not supported"):
        vq.run_neb(_nh3(), _nh3(flip=True), method="msindo", n_images=2,
                   dft_plus_u=[object()])


def test_run_neb_msindo_rejects_periodic():
    L = np.diag([12.0, 12.0, 12.0])

    def _p(flip):
        m = _nh3(flip=flip)
        return vq.PeriodicSystem(3, L, list(m.atoms))

    with pytest.raises(NotImplementedError, match="molecular-only"):
        vq.run_neb(_p(False), _p(True), method="msindo", n_images=2)


def test_run_neb_unsupported_method_lists_msindo():
    # The error enumerates the supported set, including MSINDO.
    with pytest.raises(ValueError, match="MSINDO"):
        vq.run_neb(_nh3(), _nh3(flip=True), method="bogus", n_images=2)


# ---------------------------------------------------------------------------
# Citation surface (CLAUDE.md §8) — MSINDO method paper reaches the QVF, and
# libint / the SCF-fallback basis / the run_neb default functional do NOT.
# ---------------------------------------------------------------------------


def test_run_neb_msindo_qvf_citations(umbrella, tmp_path):
    qvf = umbrella.write_qvf(str(tmp_path / "msindo_neb"))
    blob = ""
    with zipfile.ZipFile(qvf) as z:
        for name in z.namelist():
            blob += z.read(name).decode("utf-8", "replace")
    low = blob.lower()
    # MSINDO method papers (Ahlswede & Jug 1999, Parts I + II) + NEB fire.
    assert "ahlswede_jug_msindo_1_1999" in blob
    assert "ahlswede_jug_msindo_2_1999" in blob
    assert "henkelman" in low                       # NEB tangent paper
    # MSINDO is STO/INDO — no Gaussian integrals (no libint), no XC
    # functional, and the "sto-3g" SCF fallback basis must not leak in.
    assert "libint" not in low, "libint wrongly cited for an MSINDO run"
    assert "sto-3g" not in low, "STO-3G wrongly cited (placeholder basis leaked)"


# ---------------------------------------------------------------------------
# Physics validation (slow): the converged saddle is a true first-order TS —
# the FD Hessian at the climbing image has exactly one imaginary mode.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_neb_msindo_single_imaginary_mode():
    from vibeqc.semiempirical.methods.msindo import msindo_optimize

    # Relax the endpoint to the true MSINDO minimum, mirror it for the product
    # (exact umbrella image), and run a converged climbing-image band.
    react_ang, _ = msindo_optimize(_Z_NH3, _coords_ang(_nh3()), fmax=1e-3)
    prod_ang = react_ang.copy()
    prod_ang[:, 2] *= -1.0
    res = vq.run_neb(
        _mol(react_ang), _mol(prod_ang), method="msindo", n_images=5,
        interpolation="idpp", climbing_image=True,
        climbing_image_start_fraction=0.25, max_iter=200,
        conv_tol_force=1e-3, n_jobs=1,
    )
    assert res.converged
    e = np.asarray(res.energies)
    ts_idx = res.transition_state_index
    assert (e[ts_idx] - e[0]) * KCAL_PER_HA > 1.0

    # Cartesian FD Hessian (d grad / d coord) at the transition-state image.
    ts_ang = _coords_ang(res.path.images[ts_idx].system)
    n = len(_Z_NH3) * 3
    H = np.zeros((n, n))
    h = 1e-3  # Angstrom
    for i in range(len(_Z_NH3)):
        for d in range(3):
            cp = ts_ang.copy(); cp[i, d] += h
            cm = ts_ang.copy(); cm[i, d] -= h
            gp = msindo_gradient_fd(_Z_NH3, cp, step=5e-4).ravel()
            gm = msindo_gradient_fd(_Z_NH3, cm, step=5e-4).ravel()
            H[:, i * 3 + d] = (gp - gm) / (2.0 * h / A2B)
    H = 0.5 * (H + H.T)
    w = np.linalg.eigvalsh(H)
    # Exactly one clearly-negative eigenvalue (the umbrella mode); the six
    # translation/rotation modes sit at ~0 and every other mode is positive.
    assert int(np.sum(w < -1e-3)) == 1, f"eigenvalues: {np.sort(w)}"


def _coords_ang(mol):
    return np.array([list(np.array(a.xyz) / A2B) for a in mol.atoms])


def _mol(coords_ang):
    return vq.Molecule(
        [vq.Atom(z, list(np.array(c, float) * A2B))
         for z, c in zip(_Z_NH3, coords_ang)],
        0, 1,
    )
