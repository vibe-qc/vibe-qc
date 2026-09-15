"""MACE machine-learned-potential backend for ``run_neb`` (method="mace").

MACE supplies analytic energy + forces from a pre-trained model with no
SCF, no Gaussian basis, and no k-mesh — so ``run_neb(method="mace")``
evaluates each image with a single forward pass and, for periodic bands,
sidesteps the 6N+1 finite-difference SCFs entirely.

The real MACE stack (PyTorch + e3nn, the optional ``[mace]`` extra) caps
at Python <= 3.13, so the integration is exercised here with a *mock* ASE
calculator (a known analytic potential) injected via monkeypatch — this
covers the real ``run_neb`` dispatch, unit conversion, and citation
routing without torch. A gated end-to-end test runs the real model when
the extra is installed.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.neb as neb


# ---------------------------------------------------------------------------
# Mock MACE calculator: a harmonic potential toward the origin (Angstrom).
#   E[eV] = 0.5 * K * sum(pos_A**2);  F[eV/A] = -K * pos_A.
# Lets us assert the exact eV->Ha / Angstrom->bohr conversion in the NEB
# MACE evaluator, and drive a band, with no torch.
# ---------------------------------------------------------------------------

_K_EV_PER_A2 = 0.7


def _harmonic_calc():
    from ase.calculators.calculator import Calculator, all_changes

    class _HarmonicCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            pos = atoms.get_positions()
            self.results = {
                "energy": 0.5 * _K_EV_PER_A2 * float(np.sum(pos ** 2)),
                "forces": -_K_EV_PER_A2 * pos,
            }

    return _HarmonicCalc()


def _patch_mock_mace(monkeypatch, *, citation="batatia_mace_mp_2024"):
    """Replace the model loader so run_neb uses the harmonic mock calc."""
    def _fake_loader(template, mlip_options, is_periodic):
        cell = np.asarray(template.lattice, float) if is_periodic else None
        return _harmonic_calc(), neb._atomic_numbers_of(template), cell, citation

    monkeypatch.setattr(neb, "_load_mace_model", _fake_loader)


def _h2(d_bohr):
    return vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, float(d_bohr)])], 0, 1)


# ---------------------------------------------------------------------------
# Dispatch + unit conversion
# ---------------------------------------------------------------------------


def test_run_neb_mace_dispatch_and_units(monkeypatch):
    _patch_mock_mace(monkeypatch)
    res = vq.run_neb(_h2(1.4), _h2(2.0), method="mace", n_images=3,
                     interpolation="linear", max_iter=10, conv_tol_force=5e-3)

    assert res.method == "mace"
    assert res.mace_model_citation == "batatia_mace_mp_2024"
    assert len(res.path.images) == 5
    assert np.all(np.isfinite(res.energies))

    # Endpoint energy must equal the harmonic potential at the (fixed)
    # reactant geometry, exactly — verifying the eV->Ha + Angstrom->bohr
    # conversion inside _evaluate_image_mace.
    from ase.units import Bohr, Hartree
    pos_A = np.array([[0, 0, 0], [0, 0, 1.4]], float) * Bohr
    expected_ha = 0.5 * _K_EV_PER_A2 * float(np.sum(pos_A ** 2)) / Hartree
    assert res.energies[0] == pytest.approx(expected_ha, abs=1e-12)


def test_run_neb_mace_no_basis_needed(monkeypatch):
    # basis is optional for MACE (defaults to None); a band runs without it.
    _patch_mock_mace(monkeypatch)
    res = vq.run_neb(_h2(1.4), _h2(2.0), method="mace", n_images=2,
                     interpolation="linear", max_iter=5, conv_tol_force=1e-2)
    assert res.method == "mace" and res.basis is None


def test_run_neb_mace_climbing_image(monkeypatch):
    # Climbing image works on the MACE path (same force kernel).
    _patch_mock_mace(monkeypatch)
    res = vq.run_neb(_h2(1.4), _h2(2.0), method="mace", n_images=3,
                     interpolation="linear", climbing_image=True,
                     climbing_image_start_fraction=0.1, max_iter=30,
                     conv_tol_force=5e-3)
    assert res.transition_state_index is not None
    assert np.all(np.isfinite(res.energies))


def test_run_neb_mace_periodic(monkeypatch):
    # Periodic MACE: the lattice flows to the evaluator (cell + pbc), the
    # band runs, and no k-mesh / FD-gradient is involved.
    _patch_mock_mace(monkeypatch)
    L = np.diag([12.0, 12.0, 12.0])

    def _ph2(d):
        return vq.PeriodicSystem(3, L, [vq.Atom(1, [0, 0, 0]),
                                        vq.Atom(1, [0, 0, float(d)])])

    res = vq.run_neb(_ph2(1.4), _ph2(2.0), method="mace", n_images=2,
                     interpolation="linear", max_iter=5, conv_tol_force=1e-2)
    assert res.method == "mace" and res.is_periodic
    assert np.all(np.isfinite(res.energies))


def test_run_neb_mace_periodic_qvf_omits_lcao_citation(monkeypatch, tmp_path):
    _patch_mock_mace(monkeypatch, citation="batatia_mace_mp_2024")
    lattice = np.diag([12.0, 12.0, 12.0])

    def _ph2(distance):
        return vq.PeriodicSystem(
            3,
            lattice,
            [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, float(distance)])],
        )

    result = vq.run_neb(
        _ph2(1.4),
        _ph2(2.0),
        method="mace",
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
    )
    qvf = result.write_qvf(tmp_path / "periodic_mace_neb")
    with zipfile.ZipFile(qvf) as archive:
        blob = "".join(
            archive.read(name).decode("utf-8", "replace")
            for name in archive.namelist()
        )
    assert "batatia_mace_2022" in blob
    assert "pisani_crystal_1988" not in blob


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_run_neb_basis_required_for_scf():
    with pytest.raises(ValueError, match="basis set is required"):
        vq.run_neb(_h2(1.4), _h2(2.0), method="RHF", n_images=2)


def test_run_neb_mace_rejects_dft_plus_u():
    with pytest.raises(ValueError, match="dft_plus_u is not supported"):
        vq.run_neb(_h2(1.4), _h2(2.0), method="mace", n_images=2,
                   dft_plus_u=[object()])


# ---------------------------------------------------------------------------
# Citation surface (CLAUDE.md §8) — MACE method + model papers reach the QVF
# ---------------------------------------------------------------------------


def test_run_neb_mace_qvf_citations(monkeypatch, tmp_path):
    _patch_mock_mace(monkeypatch, citation="batatia_mace_mp_2024")
    res = vq.run_neb(_h2(1.4), _h2(2.0), method="mace", n_images=2,
                     interpolation="linear", max_iter=5, conv_tol_force=1e-2)
    qvf = res.write_qvf(str(tmp_path / "mace_neb"))

    blob = ""
    with zipfile.ZipFile(qvf) as z:
        for name in z.namelist():
            blob += z.read(name).decode("utf-8", "replace")
    # MACE method paper + per-model foundation paper + the NEB papers fire;
    # libint must NOT (MACE evaluates no Gaussian integrals).
    assert "batatia_mace_2022" in blob          # MACE method paper
    assert "batatia_mace_mp_2024" in blob       # MACE-MPA-0 foundation model
    assert "henkelman" in blob.lower()          # NEB
    assert "libint" not in blob.lower(), "libint wrongly cited for a MACE run"


# ---------------------------------------------------------------------------
# End-to-end with the real model (gated: needs the [mace] extra, Py <= 3.13)
# ---------------------------------------------------------------------------


def _mace_available() -> bool:
    try:
        import mace  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _mace_available(),
                    reason="MACE [mace] extra not installed (needs Python <= 3.13)")
def test_run_neb_mace_e2e():
    # Real MACE-MPA-0 (MIT, ungated) on a tiny H2 stretch. Downloads model
    # weights on first run. Asserts a sane finite band only.
    res = vq.run_neb(_h2(1.2), _h2(2.2), method="mace", n_images=3,
                     interpolation="linear", climbing_image=True,
                     climbing_image_start_fraction=0.1, max_iter=40,
                     conv_tol_force=5e-3)
    assert res.method == "mace"
    assert np.all(np.isfinite(res.energies))
    assert res.transition_state_index is not None
