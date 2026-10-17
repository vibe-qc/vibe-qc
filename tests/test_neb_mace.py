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

import json
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
    def _fake_loader(template, mlip_options, is_periodic, **_kwargs):
        cell = np.asarray(template.lattice, float) if is_periodic else None
        return _harmonic_calc(), neb._atomic_numbers_of(template), cell, citation

    monkeypatch.setattr(neb, "_load_mace_model", _fake_loader)


def _h2(d_bohr):
    return vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, float(d_bohr)])], 0, 1)


@pytest.mark.parametrize("driver,periodic", [
    ("neb", False), ("neb", True),
    ("dimer", False), ("dimer", True), ("irc", False),
])
def test_path_result_retains_loaded_mace_identity(monkeypatch, tmp_path, driver, periodic):
    """Path metadata must identify the loaded model, even if options change."""
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from vibeqc.mlip import MLIPOptions
    from vibeqc.mlip import mace as mace_mod
    from vibeqc.mlip._mace_checkpoint import CheckpointIdentity
    from vibeqc.output.formats.mlip_record import mlip_record_fields

    options = MLIPOptions(model="small", device="cpu", dtype="float64")
    identity = CheckpointIdentity("a" * 64, 123, "https://example.invalid/model")
    calc = _harmonic_calc()
    calc._vibeqc_checkpoint_identity = identity
    calculate = calc.calculate
    loads = []

    def record(*args, **kwargs):
        # Neither a caller edit nor a subsequent calculator attribute change
        # describes the checkpoint that was loaded before the first image.
        options.model, options.dtype = "large", "float32"
        calc._vibeqc_checkpoint_identity = None
        return calculate(*args, **kwargs)

    def loader(**kwargs):
        loads.append(kwargs)
        return calc

    calc.calculate = record
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: (loader, loader, None, Atoms, Bohr, Hartree))

    def endpoint(distance):
        molecule = _h2(distance)
        return (vq.PeriodicSystem(2, np.diag([12., 12., 12.]), molecule.atoms)
                if periodic else molecule)

    mode = np.array([[0., 0., -1.], [0., 0., 1.]])
    if driver == "neb":
        result = vq.run_neb(endpoint(1.4), endpoint(2.), method="mace",
                            mlip_options=options, n_images=1, max_iter=1,
                            interpolation="linear", conv_tol_force=1e6)
        with zipfile.ZipFile(result.write_qvf(tmp_path / "path")) as archive:
            provenance = json.loads(archive.read("manifest.json"))["provenance"]
        assert "mlip" in provenance
        assert provenance["mlip"] == mlip_record_fields(result)["mlip"]
        assert provenance["scf_applicable"] is False
        assert "scf_converged" not in provenance
    elif driver == "dimer":
        result = vq.run_dimer(endpoint(1.4), method="mace", mlip_options=options,
                              initial_direction=mode, max_iter=1)
    else:
        result = vq.run_irc(endpoint(1.4), method="mace", mlip_options=options,
                            transition_mode=mode, max_points=1)

    assert len(loads) == 1
    assert result.model_info.key == "small"
    assert result.device == "cpu" and result.dtype == "float64"
    assert result.weight_identity["sha256"] == identity.sha256
    assert result.weight_identity["size_bytes"] == identity.size_bytes
    assert options.model == "large" and options.dtype == "float32"
    metadata = mlip_record_fields(result)["mlip"]
    assert metadata["license"] == "MIT"
    assert metadata["checkpoint"] == result.weight_identity
    assert "accept_academic_license" not in metadata


@pytest.mark.parametrize("field,value", [
    ("energy", np.nan),
    ("energy", np.inf),
    ("energy", [0.0]),
    ("energy", "invalid"),
    ("forces", np.full((2, 3), np.nan)),
    ("forces", np.full((2, 3), np.inf)),
    ("forces", np.zeros((1, 3))),
    ("forces", np.zeros(3)),
    ("forces", "invalid"),
])
def test_mace_image_rejects_invalid_backend_results(field, value):
    """Bad model output must not become a path energy or broadcast force."""
    calc = _harmonic_calc()
    calculate = calc.calculate

    def corrupt(*args, **kwargs):
        calculate(*args, **kwargs)
        calc.results[field] = value

    calc.calculate = corrupt
    with pytest.raises(ValueError, match=rf"MACE.*{field}"):
        neb._evaluate_image_mace(
            np.array([[0., 0., 0.], [0., 0., 1.4]]),
            calc=calc, numbers=np.array([1, 1]), cell=None,
        )


@pytest.mark.parametrize("field", ["energy", "forces"])
def test_run_neb_mace_stops_at_invalid_endpoint(monkeypatch, field):
    """Reject the first invalid endpoint before evaluating the rest of a band."""
    calc = _harmonic_calc()
    calculate = calc.calculate
    calls = []

    def corrupt(*args, **kwargs):
        calls.append(1)
        calculate(*args, **kwargs)
        calc.results[field] = np.full_like(calc.results[field], np.nan)

    calc.calculate = corrupt
    monkeypatch.setattr(
        neb, "_load_mace_model",
        lambda template, *_, **_kwargs: (calc, neb._atomic_numbers_of(template), None, ""),
    )
    with pytest.raises(ValueError, match=rf"MACE.*{field}"):
        vq.run_neb(_h2(1.4), _h2(2.), method="mace", n_images=2,
                   interpolation="linear", max_iter=1)
    assert len(calls) == 1


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


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("skew", [False, True])
def test_run_neb_mace_preserves_skew_cell_and_periodic_axes(monkeypatch, dim, skew):
    """Every image must describe the same physical cell as the endpoints."""
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from vibeqc.mlip import mace as mace_mod

    calc = _harmonic_calc()
    calculate = calc.calculate
    observed = []
    loads = []

    def record(atoms=None, *args, **kwargs):
        observed.append((atoms.cell.array.copy(), atoms.pbc.copy()))
        return calculate(atoms, *args, **kwargs)

    def loader(**kwargs):
        loads.append(kwargs)
        return calc

    calc.calculate = record
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: (loader, loader, None, Atoms, Bohr, Hartree))
    lattice = np.array([[12., 2., 1.], [0., 11., 3.], [0., 0., 10.]])
    if not skew:
        lattice = np.diag(np.diag(lattice))

    def endpoint(distance):
        return vq.PeriodicSystem(dim, lattice, [vq.Atom(1, [0., 0., 0.]),
                                               vq.Atom(1, [0., 0., distance])])

    result = vq.run_neb(endpoint(1.4), endpoint(2.), method="mace", n_images=2,
                       interpolation="linear", max_iter=1, conv_tol_force=1e6)
    assert result.is_periodic
    assert len(loads) == 1
    assert observed
    for cell, pbc in observed:
        np.testing.assert_allclose(cell, lattice.T * Bohr, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(pbc, np.arange(3) < dim)


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
        provenance = json.loads(archive.read("manifest.json"))["provenance"]
        blob = "".join(
            archive.read(name).decode("utf-8", "replace")
            for name in archive.namelist()
        )
    assert "batatia_mace_2022" in blob
    assert "pisani_crystal_1988" not in blob
    assert provenance.get("scf_applicable") is False
    assert "scf_converged" not in provenance
    assert "n_scf_iterations" not in provenance


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
        provenance = json.loads(z.read("manifest.json"))["provenance"]
        for name in z.namelist():
            blob += z.read(name).decode("utf-8", "replace")
    # MACE method paper + per-model foundation paper + the NEB papers fire;
    # libint must NOT (MACE evaluates no Gaussian integrals).
    assert "batatia_mace_2022" in blob          # MACE method paper
    assert "batatia_mace_mp_2024" in blob       # MACE-MPA-0 foundation model
    assert "henkelman" in blob.lower()          # NEB
    assert "libint" not in blob.lower(), "libint wrongly cited for a MACE run"
    assert provenance.get("scf_applicable") is False
    assert "scf_converged" not in provenance
    assert "n_scf_iterations" not in provenance


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
@pytest.mark.parametrize("periodic", [False, True])
def test_real_mace_path_checkpoint_provenance(tmp_path, periodic):
    """A bounded band records the checkpoint used for upstream-parity energies."""
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from mace.calculators import mace_mp
    from vibeqc.mlip import MLIPOptions

    def endpoint(distance):
        molecule = _h2(distance)
        return (vq.PeriodicSystem(2, np.diag([12., 12., 12.]), molecule.atoms)
                if periodic else molecule)

    result = vq.run_neb(endpoint(1.4), endpoint(2.), method="mace",
                        mlip_options=MLIPOptions(device="cpu", dtype="float64"),
                        n_images=1, max_iter=1, interpolation="linear")
    assert result.model_info.key == "medium-mpa-0"
    assert result.device == "cpu" and result.dtype == "float64"
    assert len(result.weight_identity["sha256"]) == 64
    reference = mace_mp(model="medium-mpa-0", device="cpu", default_dtype="float64")
    for image, energy in zip(result.path.images, result.energies):
        atoms = Atoms(numbers=[1, 1], positions=neb._positions_of(image.system) * Bohr)
        if periodic:
            atoms.set_cell(np.asarray(image.system.lattice).T * Bohr)
            atoms.set_pbc([True, True, False])
        atoms.calc = reference
        assert energy == pytest.approx(atoms.get_potential_energy() / Hartree,
                                       abs=1e-9, rel=0)
    with zipfile.ZipFile(result.write_qvf(tmp_path / "path")) as archive:
        provenance = json.loads(archive.read("manifest.json"))["provenance"]
    assert provenance["mlip"]["model"] == "medium-mpa-0"
    assert provenance["mlip"]["license"] == "MIT"
    assert provenance["mlip"]["checkpoint"] == result.weight_identity
    assert provenance["scf_applicable"] is False


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


@pytest.mark.slow
@pytest.mark.skipif(not _mace_available(),
                    reason="MACE [mace] extra not installed (needs Python <= 3.13)")
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_real_mace_image_matches_direct_calculator(dim):
    """A reused path calculator preserves raw model energies and forces."""
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from vibeqc.mlip import MLIPOptions
    from vibeqc.mlip.mace import mace_calculator

    options = MLIPOptions(device="cpu", dtype="float64")
    calc = mace_calculator(options)
    reference = mace_calculator(options)
    # Skew cells and a short first axis exercise image interactions, rather
    # than only adding vacuum around an effectively molecular calculation.
    cell = np.array([[5., 1., .5], [0., 7., 1.5], [0., 0., 8.]]) if dim else None
    pbc = tuple(axis < dim for axis in range(3)) if dim else None
    for distance in (1.4, 2.):
        positions = np.array([[0., 0., 0.], [.2, .1, distance]])
        atoms = Atoms(numbers=[1, 1], positions=positions * Bohr)
        if cell is not None:
            atoms.set_cell(cell.T * Bohr)
            atoms.set_pbc(pbc)
        atoms.calc = reference
        expected_energy = atoms.get_potential_energy() / Hartree
        expected_gradient = -atoms.get_forces() * Bohr / Hartree
        energy, gradient, density = neb._evaluate_image_mace(
            positions, calc=calc, numbers=np.array([1, 1]), cell=cell, pbc=pbc,
        )
        assert density is None
        assert energy == pytest.approx(expected_energy, abs=1e-12, rel=0)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12, rtol=0)
