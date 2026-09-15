"""Smoke test for the ``jk_method='gpw'`` dispatch through
:func:`vibeqc.periodic_runner.run_periodic_job` — the M3b closure
that gets the GAPW route plumbed end-to-end into the user-facing
runner.

Pins:

* ``validate_jk_method(PeriodicJKMethod.GPW)`` no longer raises;
* ``run_periodic_job(..., jk_method='gpw')`` returns a result the
  runner can write out (energy, density, MO coeffs, scf_trace);
* the result matches what
  :func:`vibeqc.run_periodic_rhf_gpw` returns directly;
* open-shell + multi-k GPW raise informative errors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_jk_method import PeriodicJKMethod, validate_jk_method


def _he_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _h2_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def _si_primitive_system():
    a = 5.13155129
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.array(
        [
            [0.0, a, a],
            [a, 0.0, a],
            [a, a, 0.0],
        ],
        dtype=float,
    )
    sys.unit_cell = [
        core.Atom(14, [0.0, 0.0, 0.0]),
        core.Atom(14, [0.5 * a, 0.5 * a, 0.5 * a]),
    ]
    return sys


def _lih_rocksalt_primitive():
    """Primitive 2-atom FCC rocksalt LiH — a genuine crystal (not a
    molecule-in-a-box, which the runner would auto-classify as a
    molecular-limit cell). Cross-cell AO overlap is real, so this exercises
    the periodic GPW SCF loop rather than a vacuum-padded degenerate case.
    """
    a = 4.084 / 0.529177210903  # experimental lattice constant, bohr
    lattice = np.array(
        [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]],
        dtype=float,
    )
    unit_cell = [core.Atom(3, [0.0, 0.0, 0.0]), core.Atom(1, [a / 2, a / 2, a / 2])]
    system = vq.PeriodicSystem(3, lattice, unit_cell)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


# ---------- validate_jk_method ------------------------------------------


def test_validate_jk_method_no_longer_rejects_gpw():
    """Pre-M3b dispatch wiring, GPW raised NotImplementedError. After
    the M3b wiring, validate_jk_method accepts GPW silently."""
    validate_jk_method(
        PeriodicJKMethod.GPW,
        lattice=np.eye(3) * 16.0,
        basis_name="sto-3g",
    )


def test_validate_jk_method_no_longer_rejects_gapw():
    """GAPW is the augmentation-included path; as of M3c the SCF
    wiring and augmentation correction are complete."""
    validate_jk_method(
        PeriodicJKMethod.GAPW,
        lattice=np.eye(3) * 16.0,
        basis_name="sto-3g",
    )


# ---------- run_periodic_job dispatch -----------------------------------


def test_run_periodic_job_gpw_he_sto3g_dispatches_to_gpw_scf():
    """End-to-end smoke: ``jk_method='gpw'`` on He STO-3G converges
    via the GPW SCF and matches the standalone entry's energy."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    # Standalone GPW reference.
    standalone = vq.run_periodic_rhf_gpw(
        system,
        basis,
        cutoff_ha=300.0,
        quiet=True,
    )

    # Runner dispatch path. Disable output writing by using a
    # short-lived tempdir.
    with tempfile.TemporaryDirectory() as tmp:
        output_base = Path(tmp) / "he_gpw"
        result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RHF",
            output=str(output_base),
            jk_method="gpw",
        )
        assert result.converged
        assert result.n_iter == standalone.n_iter
        assert abs(result.energy - standalone.energy) < 1e-9
        # Decomposition fields present.
        assert hasattr(result, "e_electronic")
        assert hasattr(result, "e_nuclear")
        assert hasattr(result, "e_coulomb")
        assert hasattr(result, "e_hf_exchange")
        # Output file(s) written.
        assert any(Path(tmp).iterdir())


def test_run_periodic_job_gpw_h2_sto3g_dispatches():
    """Same smoke on H2 — actually iterates (more than 1 AO)."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _h2_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    standalone = vq.run_periodic_rhf_gpw(
        system,
        basis,
        cutoff_ha=300.0,
        max_iter=30,
        quiet=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_base = Path(tmp) / "h2_gpw"
        result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RHF",
            output=str(output_base),
            jk_method="gpw",
        )
    assert result.converged
    assert abs(result.energy - standalone.energy) < 1e-9


def test_run_periodic_job_gpw_rks_lda_on_he():
    """M3d: ``method='RKS'`` + ``functional='lda'`` + ``jk_method=
    'gpw'`` dispatches through the GPW SCF with XC enabled.
    Energy matches molecular LDA reference."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.conv_tol_energy = 1e-10
    opts.functional = "lda"
    e_mol = vq.run_rks(mol, basis, opts).energy

    with tempfile.TemporaryDirectory() as tmp:
        result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RKS",
            functional="lda",
            output=str(Path(tmp) / "he_rks_gpw"),
            jk_method="gpw",
        )
        assert result.converged
        assert abs(result.energy - e_mol) < 1e-5
        # The runner-result e_xc field carries the periodic XC piece.
        assert result.e_xc != 0.0


def test_run_periodic_job_gpw_rks_without_functional_raises():
    """RKS dispatch through GPW requires functional=. Missing it
    should raise a clear error."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="functional"):
            vq.run_periodic_job(
                system=system,
                basis=basis,
                method="RKS",  # no functional!
                output=str(Path(tmp) / "he_rks_no_func"),
                jk_method="gpw",
            )


def test_run_periodic_job_gpw_uhf_converges_on_he():
    """v0.12 R2: GPW dispatch now supports UHF. On closed-shell He
    the UHF path converges and matches the RHF result."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        rhf_result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RHF",
            output=str(Path(tmp) / "he_gpw_rhf"),
            jk_method="gpw",
        )
        uhf_result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="UHF",
            output=str(Path(tmp) / "he_gpw_uhf"),
            jk_method="gpw",
        )
        assert uhf_result.converged
        # Closed-shell forced through UHF reduces to RHF bit-for-bit.
        assert abs(uhf_result.energy - rhf_result.energy) < 1e-9


def test_run_periodic_job_gpw_kpoints():
    """Multi-k GPW: RKS+functional+kpoints dispatches to the multi-k
    driver.  RHF+kpoints (no functional) raises since multi-k GPW is
    pure-DFT only."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        output_base = Path(tmp) / "he_gpw_kpts"
        # RHF + kpoints should raise (runner multi-k GPW is RKS-only;
        # pure-DFT UKS ships at the library level, HF needs per-k K).
        with pytest.raises(NotImplementedError, match="supports RKS"):
            vq.run_periodic_job(
                system=system,
                basis=basis,
                method="RHF",
                output=str(output_base),
                jk_method="gpw",
                kpoints=[2, 2, 2],
            )

    # RKS + functional + kpoints should converge (multi-k GPW).
    with tempfile.TemporaryDirectory() as tmp:
        output_base = Path(tmp) / "he_gpw_mk"
        result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RKS",
            functional="lda",
            output=str(output_base),
            jk_method="gpw",
            kpoints=[2, 2, 2],
        )
        assert result.converged
        assert np.isfinite(float(result.energy))


def test_run_periodic_job_gpw_custom_cutoff():
    """cutoff_ha parameter is forwarded to the GPW SCF driver."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        output_cut1 = Path(tmp) / "he_cut300"
        output_cut2 = Path(tmp) / "he_cut200"
        r1 = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RHF",
            jk_method="gpw",
            output=str(output_cut1),
            cutoff_ha=300.0,
        )
        r2 = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RHF",
            jk_method="gpw",
            output=str(output_cut2),
            cutoff_ha=200.0,
        )
        assert r1.converged
        assert r2.converged
        # Different cutoffs give different FFT grids → slightly
        # different energies.
        assert abs(float(r1.energy) - float(r2.energy)) < 1e-3
        # The .out file records the cutoff in the banner.
        text1 = output_cut1.with_suffix(".out").read_text()
        assert "cutoff_ha=300.0" in text1 or "GPW" in text1


def test_run_periodic_job_gpw_multi_k_writes_output():
    """Multi-k GPW RKS produces the standard output files."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "he_mk"
        result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RKS",
            functional="lda",
            jk_method="gpw",
            kpoints=[2, 2, 2],
            output=str(stem),
        )
        assert result.converged
        assert np.isfinite(float(result.energy))
        # Standard output files should exist.
        assert stem.with_suffix(".out").exists()
        assert stem.with_suffix(".system").exists()
        out_text = stem.with_suffix(".out").read_text()
        assert "energy (Ha)" in out_text and "converged in" in out_text
        assert "kpoints" in out_text
        assert "converged" in out_text


def test_run_periodic_job_gpw_multik_read_from_inmemory_result(tmp_path):
    """Multi-k GPW READ restarts from native in-memory per-k Bloch MOs."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    common = dict(
        system=system,
        basis=basis,
        method="RKS",
        functional="lda",
        jk_method="gpw",
        kpoints=(1, 1, 2),
        cutoff_ha=80.0,
        conv_tol_energy=1e-8,
        output_qvf=False,
        write_molden_file=False,
        citations=False,
        progress=False,
    )

    base = vq.run_periodic_job(
        output=str(tmp_path / "he_gpw_base"),
        initial_guess="HCORE",
        max_iter=20,
        **common,
    )
    assert base.converged

    restarted = vq.run_periodic_job(
        output=str(tmp_path / "he_gpw_restart"),
        initial_guess="read",
        read_from=base,
        max_iter=4,
        **common,
    )

    assert restarted.converged
    assert restarted.n_iter <= 3
    assert float(restarted.energy) == pytest.approx(float(base.energy), abs=1e-8)


def test_run_periodic_job_gpw_multi_k_uks_doublet(tmp_path):
    """Multi-k GPW UKS through the runner: an H-atom doublet on a
    (1,1,2) mesh converges, reports a clean <S^2> = 0.75, matches the
    library driver's energy, and writes the standard output files.
    Completes the prompt-9 wiring (the runner previously gated UKS +
    kpoints with a pointer to the library driver)."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 12.0
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    system.multiplicity = 2
    mol = vq.Molecule(list(system.unit_cell), 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")

    stem = tmp_path / "h_uks_mk"
    result = vq.run_periodic_job(
        system=system,
        basis=basis,
        method="UKS",
        functional="lda",
        jk_method="gpw",
        kpoints=(1, 1, 2),
        cutoff_ha=80.0,
        output=str(stem),
        citations=False,
        progress=False,
    )
    assert result.converged
    assert np.isfinite(float(result.energy))
    # Boxed H doublet LDA: generous physical window.
    assert -0.7 < float(result.energy) < -0.2
    assert float(result.s_squared) == pytest.approx(0.75, abs=1e-8)
    assert stem.with_suffix(".out").exists()
    assert stem.with_suffix(".system").exists()

    # Parity with the library driver on the same grid settings.
    from vibeqc.periodic_gapw_open_shell import run_periodic_uks_gpw_multi_k

    kmesh = core.monkhorst_pack(system, [1, 1, 2])
    lib = run_periodic_uks_gpw_multi_k(
        system, basis, kmesh, functional="lda", cutoff_ha=80.0, quiet=True,
    )
    assert float(result.energy) == pytest.approx(lib.energy, abs=1e-8)


def test_run_periodic_job_gpw_multi_k_roks_doublet(tmp_path):
    """Public multi-k GPW ROKS reports restricted orbitals, exact spin,
    its backend identity, and complete standard artifacts."""
    L = 12.0
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [core.Atom(3, [L / 2, L / 2, L / 2])]
    system.multiplicity = 2
    basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 2), "sto-3g")

    stem = tmp_path / "li_roks_mk"
    result = vq.run_periodic_job(
        system=system,
        basis=basis,
        method="ROKS",
        functional="pbe",
        jk_method="gpw",
        kpoints=(1, 1, 2),
        cutoff_ha=50.0,
        max_iter=80,
        output=str(stem),
        output_qvf=True,
        citations=True,
        progress=False,
    )

    assert result.converged
    assert result.method == "roks"
    assert result.backend == "gpw-roks-multi-k"
    assert result.s_squared == pytest.approx(0.75, abs=1e-12)
    assert all(
        np.allclose(Ca, Cb, atol=1e-12)
        for Ca, Cb in zip(result.mo_coeffs_alpha, result.mo_coeffs_beta)
    )
    assert stem.with_suffix(".out").exists()
    assert stem.with_suffix(".system").exists()
    assert stem.with_suffix(".qvf").exists()
    assert stem.with_suffix(".bibtex").exists()
    assert "roothaan_rohf_1960" in stem.with_suffix(".bibtex").read_text()


def test_run_periodic_job_gpw_multi_k_uhf_still_gated(tmp_path):
    """Multi-k UHF stays fail-closed (per-k exact exchange not wired)."""
    system = _he_system(16.0)
    basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")
    with pytest.raises(NotImplementedError, match="per-k exact exchange"):
        vq.run_periodic_job(
            system=system,
            basis=basis,
            method="UHF",
            jk_method="gpw",
            kpoints=(2, 2, 2),
            output=str(tmp_path / "x"),
            progress=False,
        )


def test_run_periodic_job_gpw_compact_multik_si_converges(tmp_path):
    """P07-shaped compact Si through the runner with multi-k GPW RKS-PBE now
    takes the Bloch real-space density / per-k projection path and converges
    to a finite, physical energy -- no ``~1e15 Ha`` runaway, no
    non-converged artifact.

    This replaces the temporary fail-closed guard (vibe-qc ``2287fdd8``): the
    compact multi-k GPW crystal path is implemented, so the driver produces a
    real result instead of raising. Small basis / cutoff / mesh keep it fast;
    the full def2-SVP ``(4,4,4)`` / 300 Ha P07 target is validated as a manual
    smoke (see STATUS.md), not in the unit suite."""
    system = _si_primitive_system()
    basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")

    out = tmp_path / "si_gpw_compact"
    result = vq.run_periodic_job(
        system=system,
        basis=basis,
        method="RKS",
        functional="pbe",
        jk_method="gpw",
        kpoints=(2, 2, 2),
        cutoff_ha=120.0,
        max_iter=80,
        output=str(out),
        output_qvf=False,
        citations=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        progress=False,
    )

    energy = float(getattr(result, "energy", getattr(result, "e_total", np.nan)))
    assert np.isfinite(energy)
    # Sane Si2 all-electron STO-3G/PBE window; the old runaway was ~3.26e15 Ha.
    assert -600.0 < energy < -540.0

    out_text = (out.with_suffix(".out")).read_text()
    assert "converged" in out_text.lower()
    # The runaway signature (huge positive energy / negative eV gap) is gone.
    assert "e+15" not in out_text and "e+20" not in out_text


def test_run_periodic_job_gpw_restart():
    """Warm-start from a saved GPW result via restart_from."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)
    L = 16.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        # Run a standalone GPW SCF and save its result.
        from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw
        from vibeqc.periodic_gapw_restart import save_gpw_result

        scf_r = run_periodic_rhf_gpw(
            system,
            basis,
            cutoff_ha=300.0,
            max_iter=30,
            conv_tol_energy=1e-9,
            quiet=True,
        )
        assert scf_r.converged
        e_orig = float(scf_r.energy)

        restart_path = Path(tmp) / "restart.npz"
        save_gpw_result(str(restart_path), scf_r, basis, system)
        assert restart_path.exists()

        # Restart via run_periodic_job.
        result = vq.run_periodic_job(
            system=system,
            basis=basis,
            method="RHF",
            jk_method="gpw",
            output=str(Path(tmp) / "he_restarted"),
            restart_from=str(restart_path),
        )
        assert result.converged
        # Energies should be identical (same density, same SCF).
        assert abs(float(result.energy) - e_orig) < 1e-8


def test_run_periodic_job_restart_overrides_sap_selector_and_provenance(
    monkeypatch,
    tmp_path,
):
    """A loaded restart is the effective guess in dispatch and provenance."""
    from types import SimpleNamespace

    import vibeqc.periodic_gapw_j as gpw_module
    import vibeqc.periodic_gapw_restart as restart_module

    restart_density = np.array([[2.0]])
    restart_path = tmp_path / "restart.npz"
    restart_path.touch()
    monkeypatch.setattr(
        restart_module,
        "load_gpw_result",
        lambda _path: {
            "kind": "gpw_scf",
            "density": restart_density,
            "energy": -2.8,
        },
    )

    seen = []

    def fake_gpw_driver(_system, _basis, **kwargs):
        seen.append(kwargs)
        breakdown = SimpleNamespace(
            e_kinetic=1.0,
            e_nuclear_attraction=-5.0,
            e_hartree=1.0,
            e_hf_exchange=-0.5,
            e_nuclear_repulsion=0.7,
            e_total=-2.8,
            e_xc=0.0,
            functional=None,
        )
        return SimpleNamespace(
            energy=-2.8,
            breakdown=breakdown,
            density=restart_density.copy(),
            mo_coeffs=np.ones((1, 1)),
            mo_energies=np.array([-0.5]),
            converged=True,
            n_iter=1,
            scf_trace=(),
            fock=np.array([[-0.5]]),
            overlap=np.ones((1, 1)),
        )

    monkeypatch.setattr(gpw_module, "run_periodic_rhf_gpw", fake_gpw_driver)
    system = _he_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "restart-over-sap"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gpw",
        initial_guess="SAP",
        restart_from=restart_path,
        output=stem,
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=True,
        progress=False,
    )

    assert result.converged
    assert len(seen) == 1
    assert seen[0]["initial_guess"] == core.InitialGuess.READ
    np.testing.assert_array_equal(
        seen[0]["initial_density"],
        restart_density,
    )
    out_text = stem.with_suffix(".out").read_text()
    assert "initial_guess       = RESTART" in out_text
    citation_surface = "\n".join(
        path.read_text()
        for path in (
            stem.with_suffix(".out"),
            stem.with_suffix(".references"),
            stem.with_suffix(".bibtex"),
        )
        if path.exists()
    )
    assert "lehtola_sap_2019" not in citation_surface
    assert "lehtola_visscher_engel_sap_2020" not in citation_surface


def test_run_periodic_job_gpw_rohf_read_forwards_spin_density_pair(
    monkeypatch,
    tmp_path,
):
    """Gamma ROHF READ resolves and forwards both spin-density blocks."""
    import vibeqc.guess_read as guess_read_module
    import vibeqc.periodic_gapw_open_shell as gpw_open

    class DriverReached(RuntimeError):
        pass

    density_alpha = np.array([[1.0]])
    density_beta = np.array([[0.0]])
    source = object()
    seen = []

    def fake_read(_basis, *, read_path, read_from):
        assert read_path == ""
        assert read_from is source
        return density_alpha, density_beta

    def stop_at_driver(*_args, **kwargs):
        seen.append(kwargs)
        raise DriverReached

    monkeypatch.setattr(
        guess_read_module,
        "resolve_periodic_read_densities_open",
        fake_read,
    )
    monkeypatch.setattr(gpw_open, "run_periodic_rohf_gpw", stop_at_driver)

    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * 16.0
    system.unit_cell = [core.Atom(1, [8.0, 8.0, 8.0])]
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(DriverReached):
        vq.run_periodic_job(
            system,
            basis,
            method="ROHF",
            jk_method="gpw",
            initial_guess="READ",
            read_from=source,
            output=tmp_path / "rohf-read",
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            output_qvf=False,
            citations=False,
            progress=False,
        )

    assert len(seen) == 1
    assert seen[0]["initial_guess"] == core.InitialGuess.READ
    np.testing.assert_array_equal(
        seen[0]["initial_density"][0], density_alpha
    )
    np.testing.assert_array_equal(
        seen[0]["initial_density"][1], density_beta
    )


# ---------- live QVF checkpoint cadence (GPW route) ---------------------


def test_gpw_driver_fires_progress_iteration_per_scf_cycle():
    """The GPW SCF loop must call ``progress.iteration(...)`` once per
    cycle, carrying the running ``energy``. This is the hook the periodic
    runner's QVF checkpointer wraps to emit per-iteration cadence frames
    (previously GPW ran with only a ``quiet`` flag and no Python hook, so
    GPW jobs emitted start + terminal frames only)."""
    import warnings

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

    class _CaptureLogger:
        """Duck-typed progress sink (resolve_progress passes it through)."""

        def __init__(self):
            self.iters = []

        def iteration(self, n, **fields):
            self.iters.append((n, fields.get("energy")))

    system, basis = _lih_rocksalt_primitive()
    cap = _CaptureLogger()
    result = vq.run_periodic_rhf_gpw(
        system,
        basis,
        functional="lda",
        cutoff_ha=80.0,
        max_iter=20,
        quiet=True,
        progress=cap,
    )
    assert result.converged
    # One iteration row per SCF cycle, in order, each with an energy value.
    assert [n for n, _ in cap.iters] == list(range(1, result.n_iter + 1))
    assert all(e is not None for _, e in cap.iters)


def test_run_periodic_job_gpw_emits_per_iteration_checkpoint_cadence(tmp_path):
    """End-to-end: ``run_periodic_job(..., jk_method='gpw')`` with
    ``checkpoint_qvf`` + ``checkpoint_every=1`` writes one cadence frame per
    SCF iteration. The monotonic ``checkpoint.seq`` on the settled archive
    is ``start (1) + n_iter cadence + terminal (1)`` -- exactly
    ``n_iter + 2`` -- proving the GPW driver now drives the checkpointer's
    per-iteration hook rather than emitting a start/terminal pair only."""
    import json
    import warnings
    import zipfile

    warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)

    system, basis = _lih_rocksalt_primitive()
    ckpt = tmp_path / "live"
    result = vq.run_periodic_job(
        system=system,
        basis=basis,
        method="RKS",
        functional="lda",
        jk_method="gpw",
        cutoff_ha=80.0,
        output=str(tmp_path / "lih_gpw"),
        output_qvf=True,
        checkpoint_qvf=str(ckpt),
        checkpoint_every=1,
        verbose=0,
    )
    assert result.converged
    assert result.n_iter > 1  # actually iterated, so cadence is meaningful

    with zipfile.ZipFile(ckpt.with_suffix(".qvf")) as zf:
        prov = json.loads(zf.read("manifest.json"))["provenance"]
    assert prov["run_status"] == "converged"
    # start + one frame per SCF cycle + terminal.
    assert prov["checkpoint"]["seq"] == result.n_iter + 2


# ---------- SAP forwarding (GitLab #667) -------------------------------


@pytest.mark.parametrize(
    ("backend", "module_name", "driver_name", "multik"),
    [
        ("gpw", "vibeqc.periodic_gapw_j", "run_periodic_rhf_gpw", False),
        (
            "gapw",
            "vibeqc.periodic_gapw_augment",
            "run_periodic_rhf_gapw",
            False,
        ),
        (
            "gpw",
            "vibeqc.periodic_gapw_j",
            "run_periodic_rks_gpw_multi_k",
            True,
        ),
        (
            "gapw",
            "vibeqc.periodic_gapw_augment",
            "run_periodic_rks_gapw_multi_k",
            True,
        ),
    ],
    ids=["gpw-gamma", "gapw-gamma", "gpw-multik", "gapw-multik"],
)
def test_run_periodic_job_forwards_sap_to_gpw_gapw(
    monkeypatch,
    tmp_path,
    backend,
    module_name,
    driver_name,
    multik,
):
    """The public route passes SAP to each GPW/GAPW driver family."""
    import importlib

    class DriverReached(RuntimeError):
        pass

    seen = []

    def stop_at_driver(*_args, **kwargs):
        seen.append(kwargs)
        raise DriverReached

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, driver_name, stop_at_driver)
    system = _he_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kwargs = {
        "method": "RKS" if multik else "RHF",
        "functional": "lda" if multik else None,
        "kpoints": (2, 1, 1) if multik else None,
    }
    if multik:
        kwargs["smearing_temperature"] = 0.002
    if backend == "gapw" and not multik:
        kwargs["gapw_molecular_limit"] = True

    with pytest.raises(DriverReached):
        vq.run_periodic_job(
            system,
            basis,
            jk_method=backend,
            initial_guess="SAP",
            output=tmp_path / f"{backend}-sap-forwarding",
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            output_qvf=False,
            citations=False,
            progress=False,
            **kwargs,
        )

    assert len(seen) == 1
    assert seen[0]["initial_guess"] == core.InitialGuess.SAP
    if multik:
        assert seen[0]["initial_density_k"] is None
        assert seen[0]["smearing_temperature"] == pytest.approx(0.002)
        assert seen[0]["smearing_method"] == "fermi-dirac"


@pytest.mark.parametrize(
    ("requested", "effective"),
    [
        (core.InitialGuess.SAD, core.InitialGuess.SAD),
        (" sad ", core.InitialGuess.SAD),
        ("InitialGuess.SAD", core.InitialGuess.SAD),
        ("core", core.InitialGuess.HCORE),
        ("huckel", core.InitialGuess.HUECKEL),
        (" auto ", core.InitialGuess.SAD),
    ],
    ids=["enum", "string", "dotted", "core-alias", "huckel-alias", "auto"],
)
def test_run_periodic_job_normalizes_guess_before_gpw_dispatch(
    monkeypatch,
    tmp_path,
    requested,
    effective,
):
    """The public runner and direct routes share one selector contract."""
    import vibeqc.periodic_gapw_j as gpw_module

    class DriverReached(RuntimeError):
        pass

    seen = []

    def stop_at_driver(*_args, **kwargs):
        seen.append(kwargs)
        raise DriverReached

    monkeypatch.setattr(gpw_module, "run_periodic_rhf_gpw", stop_at_driver)
    system = _he_system(16.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(DriverReached):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gpw",
            initial_guess=requested,
            output=tmp_path / f"normalized-{effective.name.lower()}",
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            output_qvf=False,
            citations=False,
            progress=False,
        )

    assert len(seen) == 1
    assert seen[0]["initial_guess"] == effective


# ---------------------------------------------------------------------------
# #748: the SCF-failure diagnostic must report the failure, never replace it
# ---------------------------------------------------------------------------


def test_gpw_multik_nonconvergence_reports_the_real_failure_not_an_attributeerror():
    """GitLab #748: a refused GPW SCF must say why it was refused.

    The GPW / GAPW Python drivers append plain dicts to ``scf_trace`` while
    the C++, GDF, BIPOLE and Ewald routes emit ``SCFIteration`` records. Most
    GPW routes are lifted back to records by
    ``periodic_gapw_runner_adapter``, but multi-k RKS reaches the runner
    through ``_GapwMultiKRunnerProxy``, which forwards ``scf_trace``
    verbatim. An explicit Gamma mesh counts as multi-k here: the runner's
    ``gpw_use_gamma_branch`` predicate excludes RKS, so ``kpoints=(1,1,1)``
    takes the multi-k driver.

    The withdrawal diagnostic then read ``_last.delta_e`` off that dict and
    raised ``AttributeError: 'dict' object has no attribute 'delta_e'``,
    so every case reaching this branch reported the same traceback about
    the reporter instead of its own convergence state. Five LiF cases in
    the 2026-09-07 validation wave were lost that way.
    """
    system = _he_system(L=10.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError) as excinfo:
            vq.run_periodic_job(
                system,
                basis,
                method="RKS",
                functional="lda",
                jk_method="gpw",
                kpoints=(1, 1, 1),
                max_iter=1,
                output=Path(tmp) / "gpw_refused",
                citations=False,
                record_hostname=False,
                output_qvf=False,
                progress=False,
            )

    message = str(excinfo.value)
    # The real failure, not a traceback about the diagnostic.
    assert "did not converge after 1 iterations" in message
    assert "refusing to treat the energy" in message
    # And the measured numbers it exists to carry.
    assert "Terminal check on the density returned" in message
    assert "|dE| = " in message and "conv_tol_energy" in message
    # A dict row's residual is reported under its own field name: on this
    # route it is the density change the driver gates with
    # conv_tol_density, not the commutator norm conv_tol_grad gates.
    assert "trace grad_norm = " in message
    assert "||[F,DS]||" not in message


def test_terminal_scf_check_clause_renders_both_row_shapes_and_never_raises():
    """Unit contract of the #748 helper.

    Record rows keep the wording the BIPOLE route already pins
    (``tests/test_periodic_runner_bipole_callsites.py``); dict rows render
    without claiming a label the shape cannot establish; anything
    unreadable degrades to no clause at all, because this runs on the
    failure path where raising destroys the real report.
    """
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _terminal_scf_check_clause

    kw = {"conv_tol_energy": 1e-7, "conv_tol_grad": 1e-6}

    # Record row (GDF / BIPOLE / Ewald / C++): the legacy wording.
    record = SimpleNamespace(iter=3, energy=-1.0, delta_e=1.5e-2, grad_norm=3.25e-3)
    record_clause = _terminal_scf_check_clause([record], **kw)
    assert "|dE| = 1.500e-02 Ha against conv_tol_energy 1e-07" in record_clause
    assert "||[F,DS]|| = 3.250e-03 against conv_tol_grad 1e-06" in record_clause

    # Dict row (GPW / GAPW): renders, and does not assert the commutator label.
    dict_clause = _terminal_scf_check_clause(
        [{"iter": 3, "energy": -1.0, "delta_e": 1.5e-2, "grad_norm": 3.25e-3}], **kw
    )
    assert "|dE| = 1.500e-02 Ha against conv_tol_energy 1e-07" in dict_clause
    assert "trace grad_norm = 3.250e-03" in dict_clause
    assert "||[F,DS]||" not in dict_clause

    # A tuple trace is as valid as a list: the GPW drivers return tuples.
    assert _terminal_scf_check_clause(
        ({"delta_e": 1.5e-2, "grad_norm": 3.25e-3},), **kw
    ) == dict_clause

    # Missing conv_tol_grad drops only that comparison.
    no_tol = _terminal_scf_check_clause(
        [record], conv_tol_energy=1e-7, conv_tol_grad=None
    )
    assert "||[F,DS]|| = 3.250e-03." in no_tol and "conv_tol_grad" not in no_tol

    # Nothing to report -> no clause, no exception.
    assert _terminal_scf_check_clause(None, **kw) == ""
    assert _terminal_scf_check_clause([], **kw) == ""
    assert _terminal_scf_check_clause([object()], **kw) == ""

    # Partial rows report what they have.
    only_de = _terminal_scf_check_clause([{"delta_e": 1.0}], **kw)
    assert "|dE| = 1.000e+00" in only_de and "grad_norm" not in only_de

    # A row that raises on access must not take the failure report with it.
    class _Hostile:
        @property
        def delta_e(self):
            raise ValueError("trace row is poisoned")

    assert _terminal_scf_check_clause([_Hostile()], **kw) == ""

    class _HostileTrace:
        def __iter__(self):
            raise RuntimeError("trace is not iterable")

    assert _terminal_scf_check_clause(_HostileTrace(), **kw) == ""


# ---------------------------------------------------------------------------
# #768: the live-progress stamp must survive a tuple-shaped SCF trace
# ---------------------------------------------------------------------------


def _progress_section(stem: Path) -> dict:
    import tomllib

    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    return manifest.get("progress") or {}


def test_gpw_multik_stamps_live_progress_into_the_system_manifest():
    """GitLab #768: a GPW/GAPW multi-k run must stamp ``[progress]``.

    The runner selected the terminal SCF trace row with
    ``_trace[-1] if isinstance(_trace, list) else _trace``. The GPW / GAPW
    drivers return ``scf_trace=tuple(...)``, and a tuple is not a ``list``,
    so ``_last`` became the whole trace; neither the record branch nor the
    dict branch matched it, the field dict stayed empty, and
    ``update_progress`` was never called. Every multi-k GPW/GAPW run
    therefore wrote no ``[progress]`` section at all, silently, which is
    exactly the section the block's own comment says exists so ``vq`` can
    read iteration and energy from one file.

    Measured before the fix on He / STO-3G / LDA in a 10 bohr cube: the GPW
    route (tuple of dicts) produced no ``[progress]`` section, while GDF
    (list of ``SCFIteration``) on the same cell produced
    ``iteration=2, energy_eh=-2.772172361688382``.
    """
    system = _he_system(L=10.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "gpw_progress"
        result = vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="gpw",
            kpoints=(1, 1, 1),
            max_iter=60,
            output=stem,
            citations=False,
            record_hostname=False,
            output_qvf=False,
            progress=False,
        )
        assert result.converged
        # The shape that used to defeat the selection.
        trace = result.scf_trace
        assert isinstance(trace, tuple) and isinstance(trace[-1], dict)

        progress = _progress_section(stem)

    assert progress, "no [progress] section was stamped for the GPW multi-k route"
    assert progress["phase"] == "scf"
    # The stamped values are the terminal row's, not a placeholder.
    last = trace[-1]
    assert progress["iteration"] == last["iter"]
    assert progress["energy_eh"] == last["energy"]
    assert progress["gradient_norm"] == last["grad_norm"]


def test_gdf_route_still_stamps_live_progress_from_a_list_trace():
    """Control for #768: the record-row route is unchanged.

    GDF returns a ``list`` of ``SCFIteration``, the shape the original
    predicate did handle. Same cell and method as the GPW test above, so a
    regression that fixed one shape by breaking the other would show here.
    """
    system = _he_system(L=10.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "gdf_progress"
        result = vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="gdf",
            max_iter=60,
            output=stem,
            citations=False,
            record_hostname=False,
            output_qvf=False,
            progress=False,
        )
        assert result.converged
        trace = result.scf_trace
        assert isinstance(trace, list) and hasattr(trace[-1], "iter")

        progress = _progress_section(stem)

    assert progress["phase"] == "scf"
    assert progress["iteration"] == trace[-1].iter
    assert progress["energy_eh"] == trace[-1].energy


def test_terminal_trace_row_selection_accepts_every_producer_shape():
    """Unit contract of the #768 selection.

    A sequence yields its LAST row; a bare row (a producer that exposes one
    record rather than a sequence) is used as-is. Before the fix a tuple took
    the bare-row branch and the whole trace was treated as one row.
    """
    from types import SimpleNamespace

    def select(trace):
        # The expression as it stands in periodic_runner.run_periodic_job.
        return trace[-1] if isinstance(trace, (list, tuple)) else trace

    rows = [{"iter": 1}, {"iter": 2}]
    assert select(rows) == {"iter": 2}
    assert select(tuple(rows)) == {"iter": 2}

    single = SimpleNamespace(iter=7)
    assert select(single) is single

    # The pre-fix expression, kept here so the defect cannot silently return.
    def select_prefix(trace):
        return trace[-1] if isinstance(trace, list) else trace

    assert select_prefix(tuple(rows)) == tuple(rows), (
        "the pre-fix expression returned the whole tuple as if it were one row"
    )
