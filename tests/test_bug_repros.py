"""Small release-paper bug reproductions that should stay fixed."""

from __future__ import annotations

import json
from types import SimpleNamespace
import tomllib
import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import Functional, XCKind


def test_bipole_ext_el_pole_wrong_decomposition_fails_closed() -> None:
    """A K != 0 reciprocal e-e contribution must not be called EXT EL-POLE."""
    from vibeqc._vibeqc_core import compute_overlap_lattice
    from vibeqc.bipole_ext_el_pole import compute_ext_el_pole_reciprocal_sum

    system = vq.PeriodicSystem(
        3,
        5.0 * np.eye(3),
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = 1.0
    density = compute_overlap_lattice(basis, system, options)

    with pytest.raises(NotImplementedError, match="penetration-zone"):
        compute_ext_el_pole_reciprocal_sum(density, basis, system)


@pytest.mark.parametrize(
    "name,kind,omega,short_range_hf,long_range_hf",
    [
        ("cam-b3lyp", XCKind.GGA, 0.33, 0.19, 0.65),
        ("camb3lyp", XCKind.GGA, 0.33, 0.19, 0.65),
        ("lc-wpbe", XCKind.GGA, 0.40, 0.0, 1.0),
        ("lcwpbe", XCKind.GGA, 0.40, 0.0, 1.0),
        ("mn15", XCKind.MGGA, None, None, None),
    ],
)
def test_release_paper_functional_spellings_resolve(
    name: str,
    kind: XCKind,
    omega: float | None,
    short_range_hf: float | None,
    long_range_hf: float | None,
) -> None:
    """Release-paper inputs used common CAM-B3LYP/LC-wPBE/MN15 spellings."""
    functional = Functional(name)

    assert functional.kind == kind
    assert functional.is_hybrid

    if omega is not None:
        assert functional.is_range_separated
        assert functional.rsh_omega == pytest.approx(omega, abs=1e-12)
        assert functional.cam_alpha == pytest.approx(short_range_hf, abs=1e-12)
        assert (
            functional.cam_alpha + functional.cam_beta
            == pytest.approx(long_range_hf, abs=1e-12)
        )


def test_unit_cell_molecule_handles_odd_electron_primitive_cell() -> None:
    """Ag/Au primitive cells should not fail basis setup through multiplicity."""
    a = 4.0853 / 0.529177210903
    lattice = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(47, [0.0, 0.0, 0.0])])

    molecule = system.unit_cell_molecule()

    assert molecule.n_electrons() == 47
    assert molecule.multiplicity == 2
    basis = vq.BasisSet(molecule, "pob-tzvp-rev2")
    assert basis.nbasis > 0


@pytest.mark.parametrize(
    "symbol,z,n_core",
    [
        ("Ag", 47, 28),
        ("Au", 79, 60),
        ("W", 74, 60),
    ],
)
def test_pob_tzvp_rev2_loads_release_paper_ecp_metals(
    symbol: str, z: int, n_core: int
) -> None:
    """Release-paper Ag/Au/W P09 route must load POB basis and ECP blocks."""
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 20.0,
        [vq.Atom(z, [0.0, 0.0, 0.0]), vq.Atom(z, [5.0, 5.0, 5.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")

    from vibeqc.periodic_runner import _resolve_ecp_data

    ecp_blocks, ecp_centers, effective_z, total_ncore = _resolve_ecp_data(
        system, basis
    )

    assert basis.nbasis > 0, symbol
    assert len(ecp_blocks) == 2
    assert len(ecp_centers) == 2
    assert total_ncore == 2 * n_core
    assert effective_z == pytest.approx([z - n_core, z - n_core])


def test_rks_pbe_s22_water_dimer_default_accelerator_converges() -> None:
    """S22 water dimer must not stay in EDIIS until the extensive norm is tiny."""
    ang2bohr = 1.0 / 0.529177210903
    atoms = [
        (8, (0.0, 0.0, 0.0)),
        (1, (0.0, 0.793, -0.614)),
        (1, (0.0, -0.793, -0.614)),
        (8, (0.0, 0.0, 2.95)),
        (1, (0.0, 0.793, 2.336)),
        (1, (0.0, -0.793, 2.336)),
    ]
    mol = vq.Molecule(
        [vq.Atom(z, [ang2bohr * c for c in xyz]) for z, xyz in atoms]
    )
    basis = vq.BasisSet(mol, "def2-tzvp")
    opts = vq.RKSOptions()
    opts.functional = "pbe"
    opts.max_iter = 40

    result = vq.run_rks(mol, basis, opts)

    assert result.converged
    assert result.n_iter <= 20
    assert result.energy == pytest.approx(-152.7541893947, abs=1e-8)


def test_large_rks_tail_nonconvergence_retries_with_trah(
    monkeypatch, tmp_path
) -> None:
    """LM02-like large RKS tails should retry from the final density via TRAH."""
    import vibeqc.runner as runner

    atoms = [
        vq.Atom(1, [4.0 * float(i), 0.0, 0.0])
        for i in range(20)
    ]
    mol = vq.Molecule(atoms)
    captured = {}

    def fake_first_attempt(method, molecule, basis, **kwargs):
        assert method == "rks"
        nbf = basis.nbasis
        return SimpleNamespace(
            energy=-9.0,
            n_iter=80,
            converged=False,
            density=np.eye(nbf),
            scf_trace=[],
        )

    def fake_trah_retry(molecule, basis, options):
        captured["options"] = options
        return SimpleNamespace(
            energy=-9.1,
            n_iter=4,
            converged=True,
            scf_trace=[],
        )

    opts = vq.RKSOptions()
    opts.functional = "pbe"
    opts.max_iter = 80
    opts.initial_guess = vq.InitialGuess.SAP
    memory_checks = []
    monkeypatch.setattr(runner, "_run_single_point", fake_first_attempt)
    monkeypatch.setattr(runner, "run_rks", fake_trah_retry)
    monkeypatch.setattr(
        runner,
        "check_memory",
        lambda estimate, **_kwargs: memory_checks.append(estimate),
    )

    run_kwargs = _quiet_run_job_kwargs(tmp_path, "large_rks_retry")
    run_kwargs["citations"] = True
    result = runner.run_job(
        mol,
        basis="sto-3g",
        method="rks",
        functional="pbe",
        rks_options=opts,
        crash_dump=False,
        structured_log=True,
        **run_kwargs,
    )

    assert result.converged
    retry_opts = captured["options"]
    assert retry_opts.initial_guess == vq.InitialGuess.READ
    assert retry_opts.trah_threshold == pytest.approx(1.0e-2)
    assert np.asarray(retry_opts.read_density).shape == (20, 20)
    assert len(memory_checks) == 2
    assert (
        memory_checks[1].by_category["DFT grid + chi"]
        > memory_checks[0].by_category["DFT grid + chi"]
    )
    out_text = (tmp_path / "large_rks_retry.out").read_text(encoding="utf-8")
    assert out_text.count("vibe-qc estimates this calculation will require") == 2
    assert "dense TRAH retry memory preflight" in out_text
    assert "retrying from the final density with TRAH" in out_text
    citation_surface = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            tmp_path / "large_rks_retry.out",
            tmp_path / "large_rks_retry.references",
            tmp_path / "large_rks_retry.bibtex",
        )
        if path.exists()
    )
    assert "lehtola_sap_2019" in citation_surface
    assert "lehtola_visscher_engel_sap_2020" in citation_surface
    memory_events = [
        json.loads(line)
        for line in (tmp_path / "large_rks_retry.scf.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if json.loads(line).get("event") == "memory_estimate"
    ]
    assert len(memory_events) == 2
    assert memory_events[1]["phase"] == "scf.rks.trah_retry"
    assert memory_events[1]["raw_total_bytes"] == memory_checks[1].raw_total_bytes
    with (tmp_path / "large_rks_retry.system").open("rb") as fh:
        manifest = tomllib.load(fh)
    assert manifest["outputs"]["status"] == "complete"


def test_large_rhf_tail_nonconvergence_retries_with_sad(
    monkeypatch, tmp_path
) -> None:
    """Crambin-like large RHF tails should restart once from SAD."""
    import vibeqc.runner as runner

    atoms = [
        vq.Atom(1, [4.0 * float(i), 0.0, 0.0])
        for i in range(20)
    ]
    mol = vq.Molecule(atoms)
    captured = {}

    def fake_first_attempt(method, molecule, basis, **kwargs):
        assert method == "rhf"
        nbf = basis.nbasis
        return SimpleNamespace(
            energy=-9.0,
            n_iter=80,
            converged=False,
            density=np.eye(nbf),
            scf_trace=[],
        )

    def fake_sad_retry(molecule, basis, options):
        captured["options"] = options
        return SimpleNamespace(
            energy=-9.1,
            n_iter=5,
            converged=True,
            scf_trace=[],
        )

    opts = vq.RHFOptions()
    opts.max_iter = 80
    monkeypatch.setattr(runner, "_run_single_point", fake_first_attempt)
    monkeypatch.setattr(runner, "run_rhf", fake_sad_retry)

    result = runner.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        rhf_options=opts,
        crash_dump=False,
        **_quiet_run_job_kwargs(tmp_path, "large_rhf_retry"),
    )

    assert result.converged
    retry_opts = captured["options"]
    assert retry_opts.initial_guess == vq.InitialGuess.SAD
    assert retry_opts.max_iter == 120
    assert retry_opts.soscf_threshold == pytest.approx(0.0)
    assert retry_opts.trah_threshold == pytest.approx(0.0)
    out_text = (tmp_path / "large_rhf_retry.out").read_text(encoding="utf-8")
    assert "restarting once from the SAD initial guess" in out_text
    with (tmp_path / "large_rhf_retry.system").open("rb") as fh:
        manifest = tomllib.load(fh)
    assert manifest["outputs"]["status"] == "complete"


def test_large_rks_tail_options_are_eligible_for_trah_retry() -> None:
    """Large closed-shell RKS tails can retry from the last density."""
    import vibeqc.runner as runner

    ang2bohr = 1.0 / 0.529177210903
    atoms = [
        (6, (0.0, 0.0, 0.0)),
        (6, (1.2573952636169805, 0.0, 0.8891328084339168)),
        (6, (2.514790527233961, 0.0, 0.0)),
        (6, (3.7721857908509415, 0.0, 0.8891328084339168)),
        (6, (5.029581054467922, 0.0, 0.0)),
        (6, (6.2869763180849025, 0.0, 0.8891328084339168)),
        (6, (7.544371581701883, 0.0, 0.0)),
        (6, (8.801766845318863, 0.0, 0.8891328084339168)),
        (1, (-1.09, 0.0, 0.0)),
        (1, (0.327, 0.89013929512623, -0.545)),
        (1, (0.327, -0.89013929512623, -0.545)),
        (1, (1.2573952636169805, 0.89013929512623, 0.11913280843391683)),
        (1, (1.2573952636169805, -0.89013929512623, 0.11913280843391683)),
        (1, (2.514790527233961, 0.89013929512623, 0.77)),
        (1, (2.514790527233961, -0.89013929512623, 0.77)),
        (1, (3.7721857908509415, 0.89013929512623, 0.11913280843391683)),
        (1, (3.7721857908509415, -0.89013929512623, 0.11913280843391683)),
        (1, (5.029581054467922, 0.89013929512623, 0.77)),
        (1, (5.029581054467922, -0.89013929512623, 0.77)),
        (1, (6.2869763180849025, 0.89013929512623, 0.11913280843391683)),
        (1, (6.2869763180849025, -0.89013929512623, 0.11913280843391683)),
        (1, (7.544371581701883, 0.89013929512623, 0.77)),
        (1, (7.544371581701883, -0.89013929512623, 0.77)),
        (1, (9.891766845318862, 0.0, 0.8891328084339168)),
        (1, (9.128766845318863, 0.89013929512623, 1.4341328084339169)),
        (1, (9.128766845318863, -0.89013929512623, 1.4341328084339169)),
    ]
    mol = vq.Molecule(
        [vq.Atom(z, [ang2bohr * c for c in xyz]) for z, xyz in atoms]
    )
    opts = vq.RKSOptions()
    opts.functional = "pbe"
    opts.max_iter = 250
    opts.level_shift = 0.15
    opts.damping = 0.20
    opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS
    nbf = 180
    basis = SimpleNamespace(nbasis=nbf)
    result = SimpleNamespace(
        n_iter=250,
        converged=False,
        density=np.eye(nbf),
    )

    assert runner._rks_tail_trah_retry_supported(
        resolved_method="rks",
        molecule=mol,
        basis=basis,
        functional="pbe",
        rks_options=opts,
        result=result,
    )
    retry = runner._clone_rks_options_for_trah_retry(opts, result)

    assert retry.initial_guess == vq.InitialGuess.READ
    assert retry.trah_threshold == pytest.approx(1.0e-2)
    assert retry.max_iter == 80
    assert retry.level_shift == pytest.approx(0.15)
    assert retry.damping == pytest.approx(0.20)
    assert retry.scf_accelerator == vq.SCFAccelerator.EDIIS_DIIS


def test_large_rhf_tail_options_are_eligible_for_sad_retry() -> None:
    """Large closed-shell RHF tails can restart from SAD."""
    import vibeqc.runner as runner

    atoms = [
        vq.Atom(1, [4.0 * float(i), 0.0, 0.0])
        for i in range(20)
    ]
    mol = vq.Molecule(atoms)
    opts = vq.RHFOptions()
    opts.max_iter = 250
    opts.level_shift = 0.15
    opts.level_shift_warmup_cycles = 3
    opts.level_shift_schedule = [0.15, 0.05, 0.0]
    opts.damping = 0.20
    opts.diis_restart_tau = 2.0e-4
    opts.diis_adaptive_delta = 3.0e-4
    opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS
    opts.cosx = True
    opts.cosx_grid_level = 2
    nbf = 180
    basis = SimpleNamespace(nbasis=nbf)
    result = SimpleNamespace(
        n_iter=250,
        converged=False,
        density=np.eye(nbf),
    )

    assert runner._rhf_tail_sad_retry_supported(
        resolved_method="rhf",
        molecule=mol,
        basis=basis,
        rhf_options=opts,
        result=result,
    )
    retry = runner._clone_rhf_options_for_sad_retry(opts)

    assert retry.initial_guess == vq.InitialGuess.SAD
    assert retry.soscf_threshold == pytest.approx(0.0)
    assert retry.trah_threshold == pytest.approx(0.0)
    assert retry.max_iter == 250
    assert retry.level_shift == pytest.approx(0.15)
    assert retry.level_shift_warmup_cycles == 3
    assert list(retry.level_shift_schedule) == pytest.approx([0.15, 0.05, 0.0])
    assert retry.damping == pytest.approx(0.20)
    assert retry.diis_restart_tau == pytest.approx(2.0e-4)
    assert retry.diis_adaptive_delta == pytest.approx(3.0e-4)
    assert retry.scf_accelerator == vq.SCFAccelerator.EDIIS_DIIS
    assert retry.cosx is True
    assert retry.cosx_grid_level == 2


def test_porphine_rhf_sto3g_default_accelerator_converges(tmp_path) -> None:
    """Porphine RHF must use the robust EDIIS/DIIS default, not plain DIIS."""
    ang2bohr = 1.0 / 0.529177210903
    z_by_symbol = {"H": 1, "C": 6, "N": 7}
    xyz = """
    N  0.000  2.050  0.000
    N  2.050  0.000  0.000
    N  0.000 -2.050  0.000
    N -2.050  0.000  0.000
    C  0.000  3.400  0.000
    C  3.400  0.000  0.000
    C  0.000 -3.400  0.000
    C -3.400  0.000  0.000
    C  1.100  2.050  0.000
    C -1.100  2.050  0.000
    C  2.050  1.100  0.000
    C  2.050 -1.100  0.000
    C  1.100 -2.050  0.000
    C -1.100 -2.050  0.000
    C -2.050  1.100  0.000
    C -2.050 -1.100  0.000
    C  2.350  1.350  0.000
    C  1.350  2.350  0.000
    C -1.350  2.350  0.000
    C -2.350  1.350  0.000
    H  0.000  4.450  0.000
    H  4.450  0.000  0.000
    H  0.000 -4.450  0.000
    H -4.450  0.000  0.000
    H  3.350  1.800  0.000
    H  1.800  3.350  0.000
    H -1.800  3.350  0.000
    H -3.350  1.800  0.000
    H  3.350 -1.800  0.000
    H  1.800 -3.350  0.000
    H -1.800 -3.350  0.000
    H -3.350 -1.800  0.000
    H  1.100 -1.100  0.000
    H -1.100  1.100  0.000
    H  1.100  1.100  0.000
    H -1.100 -1.100  0.000
    """
    atoms = []
    for line in xyz.splitlines():
        if not line.strip():
            continue
        symbol, x, y, z = line.split()
        atoms.append(
            vq.Atom(
                z_by_symbol[symbol],
                [float(x) * ang2bohr, float(y) * ang2bohr, float(z) * ang2bohr],
            )
        )
    mol = vq.Molecule(atoms, multiplicity=1)
    opts = vq.RHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-8

    result = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        rhf_options=opts,
        output_qvf=False,
        crash_dump=False,
        **_quiet_run_job_kwargs(tmp_path, "porphine_rhf_sto3g"),
    )

    assert opts.scf_accelerator == vq.SCFAccelerator.EDIIS_DIIS
    assert result.converged
    assert result.n_iter <= 40
    # The molecular default guess is now PATOM (was AUTO->SAP). With PATOM,
    # porphine/STO-3G converges to the true ground state -773.77889...
    # (the SAP start previously converged to a higher, distinct SCF solution
    # -773.7265 and could stall; 9fbd7e1e documents this as intentional).
    assert result.energy == pytest.approx(-773.7788904532144, abs=1e-8)
    with (tmp_path / "porphine_rhf_sto3g.system").open("rb") as fh:
        manifest = tomllib.load(fh)
    assert manifest["outputs"]["status"] == "complete"


def _quiet_run_job_kwargs(tmp_path, name: str) -> dict:
    return {
        "output": tmp_path / name,
        "progress": False,
        "verbose": 0,
        "write_molden_file": False,
        "write_xyz_file": False,
        "write_population_file": False,
        "citations": False,
    }


def _water() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, -0.98]),
            vq.Atom(1, [0.0, -1.43, -0.98]),
        ]
    )


def test_initial_guess_huckel_enum_alias_covers_feature_runner(tmp_path) -> None:
    """Release feature runner uses InitialGuess.HUCKEL, not only HUECKEL."""
    assert "HUCKEL" in vq.InitialGuess.__members__
    assert vq.InitialGuess.HUCKEL == vq.InitialGuess.HUECKEL

    result = vq.run_job(
        _water(),
        basis="sto-3g",
        method="rhf",
        initial_guess=vq.InitialGuess.HUCKEL,
        **_quiet_run_job_kwargs(tmp_path, "h2o_huckel_enum"),
    )

    assert result.converged
    assert result.energy == pytest.approx(-74.9495661467, abs=1e-8)


def test_run_job_initial_guess_keyword_covers_release_feature_sweep(tmp_path) -> None:
    from vibeqc.guess_fragmo import Fragment

    mol = _water()
    for guess in ("sad", "sap", "hcore", "huckel", "minao"):
        result = vq.run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            initial_guess=guess,
            **_quiet_run_job_kwargs(tmp_path, f"h2o_{guess}"),
        )
        assert result.converged
        assert result.energy == pytest.approx(-74.9495661467, abs=1e-8)

    base = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        **_quiet_run_job_kwargs(tmp_path, "read_base"),
    )
    restart = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        initial_guess="read",
        read_from=base,
        **_quiet_run_job_kwargs(tmp_path, "read_restart"),
    )
    assert restart.converged
    assert restart.energy == pytest.approx(base.energy, abs=1e-10)
    assert restart.n_iter < base.n_iter
    with pytest.raises(ValueError, match="read_from=.*not READ"):
        vq.run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            initial_guess="sad",
            read_from=base,
            **_quiet_run_job_kwargs(tmp_path, "stale_restart"),
        )

    h2 = vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 0.74 / 0.529177210903]),
        ]
    )
    fragmo = vq.run_job(
        h2,
        basis="sto-3g",
        method="rhf",
        initial_guess="fragmo",
        fragments=[Fragment([0], multiplicity=2), Fragment([1], multiplicity=2)],
        **_quiet_run_job_kwargs(tmp_path, "h2_fragmo"),
    )

    assert fragmo.converged
    assert fragmo.energy == pytest.approx(-1.1167593074, abs=1e-10)


def test_run_job_open_shell_sap_preserves_spin_electron_counts(tmp_path) -> None:
    """The public UHF entry must retain the 5/4 SAP occupations (issue 666)."""
    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.8, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_job(
        mol,
        basis="sto-3g",
        method="uhf",
        initial_guess="sap",
        output=str(tmp_path / "oh_sap"),
        verbose=0,
    )
    overlap = np.asarray(vq.compute_overlap(basis))

    assert result.converged
    assert np.trace(np.asarray(result.density_alpha) @ overlap) == pytest.approx(
        5.0, abs=1e-9
    )
    assert np.trace(np.asarray(result.density_beta) @ overlap) == pytest.approx(
        4.0, abs=1e-9
    )


@pytest.mark.parametrize(
    ("method", "molecule", "name"),
    [
        ("rks", _water(), "rks"),
        (
            "uks",
            vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2),
            "uks",
        ),
    ],
)
def test_run_job_dft_sap_routes_are_wired_end_to_end(
    tmp_path,
    method,
    molecule,
    name,
) -> None:
    """Public molecular RKS and UKS both execute an explicit SAP start."""
    result = vq.run_job(
        molecule,
        basis="sto-3g",
        method=method,
        functional="lda",
        initial_guess="sap",
        **_quiet_run_job_kwargs(tmp_path, f"{name}_sap"),
    )

    assert result.converged
    assert np.isfinite(result.energy)


def test_run_job_option_guess_materializes_read_and_fragmo_density(tmp_path) -> None:
    """Generated feature inputs may set opts.initial_guess instead of keyword."""
    from vibeqc.guess_fragmo import Fragment

    mol = _water()
    base = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        **_quiet_run_job_kwargs(tmp_path, "opts_read_base"),
    )

    read_opts = vq.RHFOptions()
    read_opts.initial_guess = vq.InitialGuess.READ
    restart = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        rhf_options=read_opts,
        read_from=base,
        **_quiet_run_job_kwargs(tmp_path, "opts_read_restart"),
    )
    assert restart.converged
    assert np.asarray(read_opts.read_density).shape == (7, 7)
    assert restart.energy == pytest.approx(base.energy, abs=1e-10)

    h2 = vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 0.74 / 0.529177210903]),
        ]
    )
    fragmo_opts = vq.RHFOptions()
    fragmo_opts.initial_guess = vq.InitialGuess.FRAGMO
    fragmo = vq.run_job(
        h2,
        basis="sto-3g",
        method="rhf",
        rhf_options=fragmo_opts,
        fragments=[Fragment([0], multiplicity=2), Fragment([1], multiplicity=2)],
        **_quiet_run_job_kwargs(tmp_path, "opts_h2_fragmo"),
    )
    assert fragmo.converged
    assert np.asarray(fragmo_opts.read_density).shape == (2, 2)
    assert fragmo.energy == pytest.approx(-1.1167593074, abs=1e-10)


def test_v2rdm_run_job_exact_small_space_converges(tmp_path) -> None:
    """The v2RDM route must not report stale projector nonconvergence."""
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])
    opts = vq.V2RDMOptions(
        outer_max_iter=1,
        conv_tol_primal=1e-12,
        verbose=0,
    )
    stem = tmp_path / "v2rdm_nonconverged"

    result = vq.run_job(
        mol,
        basis="sto-3g",
        method="v2rdm",
        v2rdm_options=opts,
        crash_dump=False,
        **_quiet_run_job_kwargs(tmp_path, stem.name),
    )

    assert result.converged
    assert result.energy == pytest.approx(-1.1372759436, abs=1e-8)

    out_text = stem.with_suffix(".out").read_text()
    assert "Method:            v2rdm(p)" in out_text
    assert "Converged:         True" in out_text
    assert "FATAL: v2rdm(p) did not converge" not in out_text

    with stem.with_suffix(".system").open("rb") as fh:
        manifest = tomllib.load(fh)
    assert manifest["outputs"]["status"] == "complete"


def test_run_job_hybrid_tddft_passes_density_for_fxc_kernel(tmp_path) -> None:
    """High-level hybrid TDA must not drop density_ao and run a partial kernel."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vq.run_job(
            _water(),
            basis="sto-3g",
            method="rks",
            functional="pbe0",
            tddft=True,
            tddft_n_states=1,
            **_quiet_run_job_kwargs(tmp_path, "h2o_pbe0_tddft"),
        )

    assert result.converged
    out_text = (tmp_path / "h2o_pbe0_tddft.out").read_text()
    assert "TD-DFT excited states (TDA (pbe0))" in out_text
    assert not any("no density_ao provided" in str(w.message) for w in caught)
    assert not any("NOT full TDA/TDDFT-pbe0" in str(w.message) for w in caught)


def test_run_job_open_shell_hybrid_tddft_passes_spin_densities(tmp_path) -> None:
    """High-level UKS hybrid TDA must include the polarised f_xc kernel."""
    ang2bohr = 1.0 / 0.529177210903
    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 0.97 * ang2bohr]),
        ],
        0,
        2,
    )
    opts = vq.UKSOptions()
    opts.max_iter = 80
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vq.run_job(
            mol,
            basis="sto-3g",
            method="uks",
            functional="pbe0",
            tddft=True,
            tddft_n_states=1,
            uks_options=opts,
            **_quiet_run_job_kwargs(tmp_path, "oh_pbe0_uks_tddft"),
        )

    assert result.converged
    out_text = (tmp_path / "oh_pbe0_uks_tddft.out").read_text()
    assert "TD-DFT excited states (TDA-UHF (pbe0))" in out_text
    assert not any(
        "no spin-polarised density_ao" in str(w.message)
        for w in caught
    )
    assert not any("NOT full TDA/TDDFT-pbe0" in str(w.message) for w in caught)


def test_run_job_open_shell_casida_tddft_is_supported(tmp_path) -> None:
    """High-level UHF/UKS Casida must not be rejected after A/B wiring."""
    mol = vq.Molecule([vq.Atom(3, [0.0, 0.0, 0.0])], 0, 2)
    result = vq.run_job(
        mol,
        basis="sto-3g",
        method="uhf",
        tddft=True,
        tddft_type="casida",
        tddft_n_states=1,
        output_qvf=False,
        **_quiet_run_job_kwargs(tmp_path, "li_uhf_casida_tddft"),
    )

    assert result.converged
    out_text = (tmp_path / "li_uhf_casida_tddft.out").read_text()
    assert "TD-DFT excited states (Casida-UHF)" in out_text
    assert "State" in out_text
    assert "E (eV)" in out_text
    assert "FAILED" not in out_text
    assert "not supported for UHF/UKS" not in out_text


@pytest.mark.parametrize(
    "method,functional",
    [
        ("uhf", None),
        ("uks", "pbe"),
    ],
)
def test_open_shell_oh_capped_optimization_uses_trah_and_fails_closed(
    tmp_path, method: str, functional: str | None
) -> None:
    """A one-step OH smoke run uses TRAH but is not a converged geometry."""
    pytest.importorskip("ase")
    ang2bohr = 1.0 / 0.529177210903
    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 0.97 * ang2bohr]),
        ],
        0,
        2,
    )
    output = tmp_path / f"oh_{method}_one_step"

    with pytest.raises(
        RuntimeError,
        match="did not converge after 1 of 1 allowed steps",
    ):
        vq.run_job(
            mol,
            basis="sto-3g",
            method=method,
            functional=functional,
            optimize=True,
            max_opt_steps=1,
            output=output,
            progress=False,
            verbose=0,
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )

    text = output.with_suffix(".out").read_text()
    assert "open-shell geometry optimization uses TRAH" in text
    assert "FATAL: ASE BFGS geometry optimization did not converge" in text
    assert 'status           = "crashed"' in output.with_suffix(
        ".system"
    ).read_text()


@pytest.mark.slow
def test_krhf_gdf_c_diamond_default_accelerator_does_not_stall() -> None:
    """c-diamond KRHF-GDF kmesh (2,2,2) froze for 8 cycles under the
    default accelerator (2026-07-09): the periodic EDIIS_DIIS switch
    compared the size-extensive commutator norm against the intensive
    0.1 threshold (raw 0.19 > 0.1 held EDIIS; intensive 0.019 < 0.1
    says DIIS), and the pinned EDIIS vertex then replayed the same
    Fock until the FIFO evicted it. Iterations 2-9 printed dE exactly
    0.0; v0.15.28 converged in 10 iterations on the same system.

    Pins: no bit-frozen pre-convergence cycles, an iteration count in
    the plain-DIIS class, and the converged energy (accelerator-path
    independent).
    """
    ang2bohr = 1.0 / 0.529177210903
    a_lat = 3.567 * ang2bohr
    lattice = 0.5 * a_lat * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(6, [0.0, 0.0, 0.0]),
            vq.Atom(6, [0.25 * a_lat] * 3),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-7

    result = vq.run_krhf_periodic_gdf(
        system, basis, (2, 2, 2), opts, progress=False, verbose=0
    )

    assert result.converged
    assert result.n_iter <= 12
    # The stall signature: consecutive iterations with bit-identical
    # energies (dE exactly 0.0) before convergence.
    frozen = [
        t.iter
        for t in result.scf_trace[:-1]
        if t.iter > 1 and t.delta_e == 0.0
    ]
    assert frozen == [], f"bit-frozen SCF iterations: {frozen}"
    # Revision-bound full-BZ Hamiltonian pin, not an external absolute-parity
    # claim. The old -74.7817190818 value predates the Gamma pair-FT mirror
    # correction (f8c213e8). That fix moved this same k222 case to
    # -74.8306916636 Ha, independently matched by the mirror-insensitive
    # neutral-torus full-mesh control at -74.83069166358 Ha/cell. The physical
    # shifted reciprocal support correction (3e5bbd8a) then moved the fixed
    # point by -81.385 microHa to the value below (remeasured 2026-07-18 on
    # origin/main @ 37afbcac). See HANDOVER_GDF_FIT_SCREENING.md.
    assert result.energy == pytest.approx(-74.8307730489, abs=1e-6)


@pytest.mark.parametrize('method', ['rhf', 'rks'])
@pytest.mark.parametrize('has_metadata', [False, True], ids=['legacy-result', 'native-metadata'])
@pytest.mark.parametrize('requested_name', ['AUTO', 'SAP'])
def test_tail_retry_keeps_requested_physical_and_transport_guess(
    monkeypatch, tmp_path, method, has_metadata, requested_name
):
    import vibeqc as vq
    import vibeqc.runner as runner
    from vibeqc.guess import select_initial_guess

    mol = vq.Molecule([vq.Atom(1, [4.0*i, 0.0, 0.0]) for i in range(20)])
    opts = vq.RHFOptions() if method == 'rhf' else vq.RKSOptions()
    opts.initial_guess = getattr(vq.InitialGuess, requested_name)
    opts.max_iter = 80
    if method == 'rks':
        opts.functional = 'pbe'
    source_selection = select_initial_guess(mol, opts.initial_guess)
    attempts = []

    def first_attempt(_method, _molecule, basis, **_kwargs):
        result = SimpleNamespace(energy=-9.0, n_iter=80, converged=False,
                                 density=np.eye(basis.nbasis), scf_trace=[])
        if has_metadata:
            result.guess_selection = source_selection
        return result

    def retry(_molecule, _basis, options):
        attempts.append(options.initial_guess)
        result = SimpleNamespace(energy=-9.1, n_iter=4, converged=True, scf_trace=[])
        if has_metadata:
            result.guess_selection = select_initial_guess(mol, options.initial_guess)
        return result

    monkeypatch.setattr(runner, '_run_single_point', first_attempt)
    monkeypatch.setattr(runner, 'run_' + method, retry)
    monkeypatch.setattr(runner, 'check_memory', lambda *_args, **_kwargs: None)
    result = runner.run_job(
        mol, basis='sto-3g', method=method,
        functional='pbe' if method == 'rks' else None,
        **{method + '_options': opts},
        output=tmp_path / 'retry', progress=False, verbose=0,
        write_molden_file=False, write_xyz_file=False,
        write_population_file=False, output_qvf=False,
        citations=True, crash_dump=False,
    )
    effective = vq.InitialGuess.SAD if method == 'rhf' else source_selection.effective
    transport = vq.InitialGuess.SAD if method == 'rhf' else vq.InitialGuess.READ
    assert attempts == [transport]
    selection = result.guess_selection
    assert (selection.requested, selection.effective, selection.transport) == (
        opts.initial_guess, effective, transport,
    )
    out = (tmp_path / 'retry.out').read_text()
    label = requested_name if selection.requested == effective else f'{requested_name} -> {effective.name}'
    if transport != effective:
        label += f' (transport: {transport.name})'
    assert f'initial_guess = {label}' in out
    references = (tmp_path / 'retry.references').read_text()
    if method == 'rks' and requested_name == 'SAP':
        assert 'lehtola_sap_2019' in references
        assert 'lehtola_visscher_engel_sap_2020' in references
