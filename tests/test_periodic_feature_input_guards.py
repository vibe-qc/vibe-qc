"""Molecular inputs on periodic (k-point) features fail fast with a
targeted message, not a deep AttributeError or a bare TypeError.

Band structures, DOS meshes, k-paths, and Monkhorst-Pack sampling are
defined on the Brillouin zone of a lattice; a Molecule has neither. Before
these guards:

* ``run_periodic_job(system=Molecule)`` died with
  ``AttributeError: 'Molecule' object has no attribute 'lattice'``;
* ``run_job(mol, kpoints=...)`` raised Python's bare
  ``TypeError: unexpected keyword argument`` with no pointer to
  ``run_periodic_job``;
* ``KPoints.band_path`` / ``monkhorst_pack`` / ``band_structure_hcore`` /
  ``band_path_eigenvalues`` crashed at whatever attribute they touched
  first.

Every entry point now raises with the same explanation: molecules have no
Brillouin zone, use run_periodic_job / run_job as appropriate.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq


@pytest.fixture()
def h2_mol():
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])], 0, 1
    )


def test_run_periodic_job_rejects_molecule(h2_mol, tmp_path):
    basis = vq.BasisSet(h2_mol, "sto-3g")
    with pytest.raises(TypeError, match="no lattice or Brillouin zone"):
        vq.run_periodic_job(
            system=h2_mol,
            basis=basis,
            method="RHF",
            output=str(tmp_path / "x"),
            progress=False,
        )


@pytest.mark.parametrize(
    "kwarg",
    [
        {"kpoints": (2, 2, 2)},
        {"dos_kmesh": [4, 4, 4]},
        {"jk_method": "gpw"},
        {"bz_integration": "gilat"},
        {"cutoff_ha": 300.0},
    ],
)
def test_run_job_rejects_pbc_only_kwargs(h2_mol, tmp_path, kwarg):
    """The molecular runner traps the common PBC-only kwargs with a pointer
    to run_periodic_job instead of a bare unexpected-keyword TypeError."""
    with pytest.raises(ValueError, match="periodic-boundary-condition"):
        vq.run_job(
            h2_mol,
            basis="sto-3g",
            output=str(tmp_path / "x"),
            dry_run=True,
            **kwarg,
        )


def test_kpoints_band_path_rejects_molecule(h2_mol):
    from vibeqc.kpoints import KPoints

    with pytest.raises(TypeError, match="Brillouin zone"):
        KPoints.band_path(h2_mol)


def test_kpoints_monkhorst_pack_rejects_molecule(h2_mol):
    from vibeqc.kpoints import KPoints

    with pytest.raises(TypeError, match="Brillouin zone"):
        KPoints.monkhorst_pack(h2_mol, (2, 2, 2))


def test_band_structure_hcore_rejects_molecule(h2_mol):
    from vibeqc.bands import band_structure_hcore

    basis = vq.BasisSet(h2_mol, "sto-3g")
    with pytest.raises(TypeError, match="Brillouin zone"):
        band_structure_hcore(h2_mol, basis, None)


def test_band_path_eigenvalues_rejects_molecule(h2_mol):
    from vibeqc.periodic_gapw_postscf import band_path_eigenvalues

    basis = vq.BasisSet(h2_mol, "sto-3g")
    with pytest.raises(TypeError, match="Brillouin zone"):
        band_path_eigenvalues(
            h2_mol, basis, np.eye(basis.nbasis), [(0.0, 0.0, 0.0)]
        )


def test_periodic_system_still_accepted():
    """The guards key on lattice/unit_cell presence -- a real
    PeriodicSystem passes through to the actual machinery."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.kpoints import KPoints

    L = 10.0
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    kp = KPoints.monkhorst_pack(sysp, (2, 2, 2))
    assert kp.kpoints_cart.shape[0] > 0


def test_run_periodic_job_forwards_bipole_cutoffs(monkeypatch, tmp_path):
    import vibeqc.pbc_bipole_rks as bipole_rks

    class SeenBipoleOptions(RuntimeError):
        pass

    seen = {}

    def fake_run_pbc_bipole_rks(system, basis, kmesh, opts, **kwargs):
        seen["cutoff_bohr"] = float(opts.lattice_opts.cutoff_bohr)
        seen["nuclear_cutoff_bohr"] = float(opts.lattice_opts.nuclear_cutoff_bohr)
        seen["functional"] = kwargs["functional"]
        seen["sr_image_precision"] = kwargs["sr_image_precision"]
        raise SeenBipoleOptions

    monkeypatch.setattr(
        bipole_rks,
        "run_pbc_bipole_rks",
        fake_run_pbc_bipole_rks,
    )
    length = 10.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * length,
        [
            vq.Atom(1, [length / 2.0, length / 2.0, length / 2.0 - 0.7]),
            vq.Atom(1, [length / 2.0, length / 2.0, length / 2.0 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    with pytest.raises(SeenBipoleOptions):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe0",
            jk_method="bipole",
            kpoints=(1, 1, 1),
            bipole_cutoff_bohr=12.0,
            bipole_nuclear_cutoff_bohr=9.0,
            sr_image_precision=2e-5,
            output=tmp_path / "bipole_cutoff",
            convergence="off",
            progress=False,
            citations=False,
            output_qvf=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
        )

    assert seen == {
        "cutoff_bohr": 12.0,
        "nuclear_cutoff_bohr": 9.0,
        "functional": "pbe0",
        "sr_image_precision": 2e-5,
    }


def test_periodic_dry_run_estimate_writes_rks_gradient_memory(
    tmp_path,
    monkeypatch,
):
    import tomllib

    from vibeqc.memory import estimate_periodic_xc_gradient

    L = 8.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * L,
        [
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "pbc_rks_grad_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="PBE",
        optimize=True,
        output=stem,
        progress=False,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    expected = estimate_periodic_xc_gradient(
        n_basis=basis.nbasis,
        n_atoms=len(sysp.unit_cell),
        n_grid_points=len(sysp.unit_cell) * 75 * 17 * 36,
        n_cells=len(vq.direct_lattice_cells(sysp, 18.0)),
        functional_kind="GGA",
        open_shell=False,
    ).total_bytes
    assert body["outputs"]["status"] == "dry_run"
    assert body["memory"]["estimate_bytes"] == expected
    assert body["memory"]["estimate_bytes"] > 0


def test_periodic_dry_run_estimate_writes_gdf_memory(
    tmp_path,
    monkeypatch,
):
    import tomllib

    from vibeqc.memory import estimate_periodic_multik_gdf
    from vibeqc.periodic_runner import _periodic_gdf_aux_basis_size

    L = 8.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * L,
        [
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "pbc_gdf_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="PBE",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        output=stem,
        progress=False,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    # #92 (3d6ed3130) made the preflight size the auxiliary set the GDF driver
    # will actually allocate -- ``aux_basis or default_aux_for(basis.name)``,
    # here def2-svp-jk -- instead of the old ``3 x n_ao`` fallback, which had
    # printed 452 GB for a 1096 GB cache on the NaCl conventional cell. #561:
    # this expectation had kept the fallback, so it under-charged the Lpq
    # factor cache and disagreed with the runner. Resolve it the same way the
    # runner does rather than restating a literal.
    n_aux = _periodic_gdf_aux_basis_size(sysp, basis, None)
    assert n_aux == 36  # def2-svp-jk over 2 H; the fallback would be 3 x 2 = 6

    expected_est = estimate_periodic_multik_gdf(
        n_basis=basis.nbasis,
        n_aux=n_aux,
        n_kpoints=2,
        need_k_pairs=False,
        open_shell=False,
    )
    # Since 2026-08-13 the GDF preflight also counts the dense rsgdf
    # AO-pair FT bundle (handovers/HANDOVER_GDF_FIT_SCREENING.md; the
    # Al2O3/def2-SVP 41-GiB-vs-0.4-GB underestimate incident).
    from vibeqc.aux_basis import rsgdf_dense_g_mesh

    _n_g = int(rsgdf_dense_g_mesh(sysp, 200.0).shape[0])
    expected_est.by_category["GDF dense AO-pair FT bundle"] = (
        basis.nbasis * basis.nbasis * _n_g * 16
    )
    expected = expected_est.total_bytes
    assert body["outputs"]["status"] == "dry_run"
    assert body["memory"]["estimate_bytes"] == expected
    assert body["memory"]["estimate_bytes"] > 0


def test_periodic_skala_dry_run_records_model_memory_and_provenance(
    tmp_path,
    monkeypatch,
):
    import tomllib

    from vibeqc.periodic_runner import _periodic_skala_memory_estimate
    from vibeqc.skala import (
        MODEL_REVISION,
        MODEL_SHA256,
        SOURCE_NOTICE_FILENAME,
    )

    lattice_length = 8.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * lattice_length,
        [
            vq.Atom(1, [4.0, 4.0, 3.3]),
            vq.Atom(1, [4.0, 4.0, 4.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "pbc_skala_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="skala-1.1",
        jk_method="bipole",
        output=stem,
        progress=False,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as handle:
        body = tomllib.load(handle)

    expected = _periodic_skala_memory_estimate(system)
    assert body["memory"]["estimate_bytes"] == expected.total_bytes
    assert body["run"]["skala_model_revision"] == MODEL_REVISION
    assert body["run"]["skala_model_sha256"] == MODEL_SHA256
    assert (
        body["run"]["skala_source_notice_file"] == SOURCE_NOTICE_FILENAME
    )
    assert body["run"]["skala_grid_policy"] == "skala"


@pytest.mark.parametrize(
    ("dispersion", "should_reject"),
    [
        (None, False),
        (False, False),
        ("", False),
        ("none", False),
        ("false", False),
        ("b3lyp5", True),
    ],
)
def test_periodic_skala_dispersion_guard_preserves_disable_spellings(
    tmp_path,
    dispersion,
    should_reject,
):
    lattice_length = 8.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * lattice_length,
        [
            vq.Atom(1, [4.0, 4.0, 3.3]),
            vq.Atom(1, [4.0, 4.0, 4.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"pbc_skala_dispersion_{str(dispersion).lower()}"
    kwargs = dict(
        method="RKS",
        functional="skala-1.1",
        jk_method="gdf",
        dispersion=dispersion,
        output=stem,
        dry_run=True,
        progress=False,
    )

    if should_reject:
        with pytest.raises(
            NotImplementedError,
            match="dispersion is not yet supported with periodic SKALA",
        ):
            vq.run_periodic_job(system, basis, **kwargs)
        assert not stem.with_suffix(".system").exists()
    else:
        assert vq.run_periodic_job(system, basis, **kwargs) is None
        assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "selector",
    [
        {"method": "RKS", "jk_method": "real-gamma"},   # legacy spelling (M1: warns)
        {"method": "aiccm", "variant": "real-gamma"},  # front door (KS inferred)
    ],
    ids=["legacy", "front-door"],
)
def test_periodic_skala_real_gamma_ccm_reaches_dry_run(tmp_path, selector):
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [
            vq.Atom(1, [4.0, 4.0, 3.3]),
            vq.Atom(1, [4.0, 4.0, 4.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "pbc_skala_real_gamma"

    result = vq.run_periodic_job(
        system,
        basis,
        functional="skala-1.1",
        initial_guess="HCORE",
        output=stem,
        dry_run=True,
        progress=False,
        **selector,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize("multiplicity", [1, 2], ids=["rks", "uks"])
def test_periodic_skala_rejects_neutral_bloch_before_dry_run(
    tmp_path,
    multiplicity,
):
    atoms = (
        [vq.Atom(1, [4.0, 4.0, 3.3]), vq.Atom(1, [4.0, 4.0, 4.7])]
        if multiplicity == 1
        else [vq.Atom(1, [4.0, 4.0, 4.0])]
    )
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        atoms,
        0,
        multiplicity,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"pbc_skala_neutral_bloch_{multiplicity}"

    with pytest.raises(
        NotImplementedError,
        match="external XC.*neutral-bloch",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="aiccm",
            variant="neutral-bloch",
            functional="skala-1.1",
            aiccm_lattice_extension=(2, 1, 1),
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "property_kwargs",
    [
        {"coop_cohp": True},
        {"dos_kmesh": [1, 1, 1]},
        {"band_structure": object()},
    ],
)
def test_periodic_skala_rejects_surrogate_hamiltonian_properties_before_scf(
    tmp_path,
    property_kwargs,
):
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [
            vq.Atom(1, [4.0, 4.0, 3.3]),
            vq.Atom(1, [4.0, 4.0, 4.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "pbc_skala_unsupported_property"

    with pytest.raises(
        NotImplementedError,
        match="generic QVF reconstruction cannot include the nonlocal XC",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="skala-1.1",
            jk_method="gdf",
            output=stem,
            output_qvf=True,
            dry_run=True,
            progress=False,
            **property_kwargs,
        )

    assert not stem.with_suffix(".system").exists()


def test_periodic_rohf_gdf_dry_run_charges_open_shell_pair_memory(
    tmp_path,
    monkeypatch,
):
    import tomllib

    from vibeqc.memory import estimate_periodic_multik_gdf
    from vibeqc.periodic_runner import _periodic_gdf_aux_basis_size

    lattice = np.eye(3) * 10.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [5.0, 5.0, 5.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "pbc_rohf_gdf_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="ROHF",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        output=stem,
        progress=False,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    # #92 (3d6ed3130) made the preflight size the auxiliary set the GDF driver
    # will actually allocate -- ``aux_basis or default_aux_for(basis.name)``,
    # here def2-svp-jk -- instead of the old ``3 x n_ao`` fallback, which had
    # printed 452 GB for a 1096 GB cache on the NaCl conventional cell. #561:
    # this expectation had kept the fallback, so it under-charged the Lpq
    # factor cache and disagreed with the runner. Resolve it the same way the
    # runner does rather than restating a literal.
    n_aux = _periodic_gdf_aux_basis_size(sysp, basis, None)
    assert n_aux == 18  # def2-svp-jk over 1 H; the fallback would be 3 x 1 = 3

    expected_est = estimate_periodic_multik_gdf(
        n_basis=basis.nbasis,
        n_aux=n_aux,
        n_kpoints=2,
        need_k_pairs=True,
        open_shell=True,
    )
    # Since 2026-08-13 the GDF preflight also counts the dense rsgdf
    # AO-pair FT bundle (handovers/HANDOVER_GDF_FIT_SCREENING.md).
    from vibeqc.aux_basis import rsgdf_dense_g_mesh

    _n_g = int(rsgdf_dense_g_mesh(sysp, 200.0).shape[0])
    expected_est.by_category["GDF dense AO-pair FT bundle"] = (
        basis.nbasis * basis.nbasis * _n_g * 16
    )
    expected = expected_est.total_bytes
    assert body["outputs"]["status"] == "dry_run"
    assert body["memory"]["estimate_bytes"] == expected


@pytest.mark.parametrize("jk_method", ["gpw", "gapw"])
def test_periodic_dry_run_estimate_writes_gpw_gapw_memory(
    tmp_path,
    monkeypatch,
    jk_method,
):
    import tomllib

    from vibeqc.memory import estimate_periodic_gpw_gapw
    from vibeqc.periodic_gapw_grid import nx_for_axis

    L = 8.0
    cutoff_ha = 60.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * L,
        [
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"pbc_{jk_method}_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="PBE",
        jk_method=jk_method,
        cutoff_ha=cutoff_ha,
        output=stem,
        progress=False,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    n_axis = nx_for_axis(L, cutoff_ha)
    # Mirror periodic_runner.py's own dry-run estimate inputs rather than
    # assuming augmentation is always active for jk_method="gapw": since
    # 4f342c4f, softened_basis() preserves hydrogen's full contraction (it
    # has no core to separate), so an all-hydrogen system like this one
    # genuinely needs no augmentation and augmentation_active is False.
    n_soft_basis = basis.nbasis
    augmentation_active = False
    if jk_method == "gapw":
        from vibeqc.periodic_gapw_augment import softened_basis
        from vibeqc.periodic_runner import _basis_primitive_count

        soft_basis = softened_basis(basis, sysp)
        n_soft_basis = soft_basis.nbasis
        full_prim = _basis_primitive_count(basis)
        soft_prim = _basis_primitive_count(soft_basis)
        augmentation_active = full_prim != soft_prim
    expected = estimate_periodic_gpw_gapw(
        n_basis=basis.nbasis,
        n_grid_points=n_axis**3,
        route=jk_method,
        functional_kind="GGA",
        open_shell=False,
        n_kpoints=1,
        n_soft_basis=n_soft_basis,
        n_atoms=len(sysp.unit_cell),
        augmentation_active=augmentation_active,
    ).total_bytes
    assert body["outputs"]["status"] == "dry_run"
    assert body["memory"]["estimate_bytes"] == expected
    assert body["memory"]["estimate_bytes"] > 0


def test_periodic_run_writes_gdf_memory_before_scf(
    tmp_path,
    monkeypatch,
):
    import vibeqc.periodic_runner as periodic_runner

    class StopBeforeSCF(RuntimeError):
        pass

    def fake_run_krks_periodic_gdf(*args, **kwargs):
        raise StopBeforeSCF

    monkeypatch.setattr(
        periodic_runner,
        "run_krks_periodic_gdf",
        fake_run_krks_periodic_gdf,
    )
    L = 8.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * L,
        [
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "pbc_gdf_live_est"

    with pytest.raises(StopBeforeSCF):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="PBE",
            jk_method="gdf",
            kpoints=(2, 1, 1),
            output=stem,
            convergence="off",
            progress=False,
            citations=False,
            output_qvf=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
        )

    text = stem.with_suffix(".out").read_text()
    assert "vibe-qc estimates this calculation" in text
    assert "GDF Lpq factor cache" in text


# ---------------------------------------------------------------------------
# GPW-GAPW-GILAT-SILENTLY-IGNORED (reported by the general-PBC chat,
# 2026-08-02): `bz_integration="gilat"` was ACCEPTED by the GPW and GAPW
# dispatch and then dropped, because neither driver threads it into its
# occupation logic. The run used Fermi-Dirac while the `.out` recorded
#
#     smearing_method      = fermi-dirac
#     bz_integration       = gilat
#
# at once, and the `.references` sidecar cited Gilat-Raubenheimer 1966 +
# Gilat 1972 (4 hits; 5 in `.bibtex`) for numerics the run never performed.
# Measured on this exact fixture the energy was BITWISE identical with and
# without the flag (-1.152074940754 either way), which is what proves the
# request never reached the occupations. That is a CLAUDE.md section 8
# citation-integrity break, so the flag is now refused on these two routes
# rather than silently ignored -- and refused BEFORE the citation route is
# registered, so the entry cannot fire for a backend that does not honour it.
def _h2_periodic():
    import numpy as np
    from vibeqc import _vibeqc_core as core

    L = 12.0
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    mol = vq.Molecule(
        [
            vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
            vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
        ],
        0,
        1,
    )
    return sysp, vq.BasisSet(mol, "sto-3g")


@pytest.mark.parametrize("jk_method", ["gpw", "gapw"])
@pytest.mark.parametrize("extra", [{}, {"smearing_temperature": 0.01}])
def test_gpw_gapw_refuse_gilat_rather_than_dropping_it(
    jk_method, extra, tmp_path
):
    """GPW/GAPW must refuse a BZ backend they do not implement."""
    sysp, basis = _h2_periodic()
    with pytest.raises(NotImplementedError, match="bz_integration='gilat'"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe",
            jk_method=jk_method,
            kpoints=(2, 1, 1),
            bz_integration="gilat",
            output=str(tmp_path / "x"),
            progress=False,
            **extra,
        )


@pytest.mark.parametrize("jk_method", ["gpw", "gapw"])
def test_gilat_refusal_writes_no_gilat_citation(jk_method, tmp_path):
    """The refusal must precede citation registration: a run that never
    performed Gilat-Raubenheimer must not leave a GR citation behind."""
    sysp, basis = _h2_periodic()
    stem = tmp_path / f"gilat_{jk_method}"
    with pytest.raises(NotImplementedError):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe",
            jk_method=jk_method,
            kpoints=(2, 1, 1),
            bz_integration="gilat",
            output=str(stem),
            progress=False,
        )
    for suffix in (".references", ".bibtex", ".out"):
        path = stem.with_suffix(suffix)
        if path.exists():
            assert "gilat" not in path.read_text().lower(), (
                f"{path.name} cites Gilat-Raubenheimer for a run that was "
                f"refused before it could perform it"
            )


@pytest.mark.parametrize("jk_method", ["gpw", "gapw"])
def test_gpw_gapw_without_bz_integration_still_run(jk_method, tmp_path):
    """The guard is keyed on the flag, not the route: dropping
    ``bz_integration`` leaves these routes working exactly as before."""
    sysp, basis = _h2_periodic()
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="pbe",
        jk_method=jk_method,
        kpoints=(2, 1, 1),
        output=str(tmp_path / f"plain_{jk_method}"),
        progress=False,
    )
    # Pinned pre-fix value on this fixture; the no-flag path is untouched.
    assert float(result.energy) == pytest.approx(-1.152074940754, abs=1e-9)


# ---------------------------------------------------------------------------
# Issue #88 (Ag/Au clause): run_periodic_job refuses closed-shell methods on
# odd-electron cells. The guard (be1d32b6f, 2026-07-02) predates the issue's
# evidence decks by six weeks, so the clause is a stale deck-prep defect, not
# a live wrong answer; these tests pin the refusal so the verdict stays a
# contract rather than a manual sweep note.
# ---------------------------------------------------------------------------


def _h_atom_odd_electron():
    """One hydrogen atom: one electron, closed-shell impossible."""
    L = 12.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * L,
        [vq.Atom(1, [L / 2, L / 2, L / 2])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


@pytest.mark.parametrize(
    "method,functional",
    [("RHF", None), ("RKS", "lda")],
)
def test_run_periodic_job_refuses_closed_shell_on_odd_electron_count(
    method, functional, tmp_path
):
    """#88: a closed-shell run of an odd electron count must refuse, not
    converge to a wrong answer (the failure the issue's Ag/Au rows showed)."""
    sysp, basis = _h_atom_odd_electron()
    kwargs = {} if functional is None else {"functional": functional}
    with pytest.raises(ValueError, match="odd electron count"):
        vq.run_periodic_job(
            sysp,
            basis,
            method=method,
            output=str(tmp_path / f"odd_{method.lower()}"),
            progress=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_cif_file=False,
            write_xsf_structure_file=False,
            **kwargs,
        )


def test_run_periodic_job_closed_shell_even_count_still_runs(tmp_path):
    """#88 negative control (L125): the SAME RHF route with an even
    electron count is untouched by the guard and still converges."""
    sysp, basis = _h2_periodic()
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        output=str(tmp_path / "even_rhf"),
        progress=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
        max_iter=60,
    )
    assert result.converged


# ---------------------------------------------------------------------------
# AICCM front door M0 (handovers/HANDOVER_AICCM_STANDARD_METHOD.md, D-8): the
# selector and manifest contracts live in this T2 consumer file, reached
# through the public runner surface only.
# ---------------------------------------------------------------------------


def test_real_gamma_selector_value_and_rejected_a_prefix():
    """The real-Gamma selector's value is the front-door variant name; the
    A-prefixed spellings stay rejected inputs (D89/D90); the unknown-name
    error lists the valid value and never the rejected one."""
    from vibeqc.periodic_jk_method import PeriodicJKMethod, resolve_jk_method_string

    rg = PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA
    assert rg.value == "real-gamma"
    assert PeriodicJKMethod("real-gamma") is rg
    for spelling in ("real-gamma", "real_gamma", "REAL-GAMMA", " Real_Gamma "):
        assert resolve_jk_method_string(spelling) is rg
    assert "aiccm2026dev-a-real-gamma" not in {m.value for m in PeriodicJKMethod}
    for bad in ("aiccm2026dev-a-real-gamma", "aiccm2026dev-a-direct"):
        with pytest.raises(ValueError, match="prefix denotes the union-and-weight"):
            resolve_jk_method_string(bad)
    with pytest.raises(ValueError, match=r"Valid: .*\breal-gamma\b") as exc:
        resolve_jk_method_string("bogus")
    assert "aiccm2026dev-a-real-gamma" not in str(exc.value)


def test_aiccm_jk_methods_set_is_the_stamp_contract():
    """The private set that gates [run].method_status holds exactly the four
    AICCM selectors (M1 added the neutral-Bloch member), equals the variant
    table's value set, and each member is described as
    ``aiccm (<variant>) ...; experimental`` in the .out, so the manifest
    marker, the front-door table and the "J/K method" line cannot drift
    apart. M3 extends nothing here; it wires an arm that already has a
    member."""
    from vibeqc.periodic_jk_method import (
        AICCM_VARIANT_OF,
        AICCM_VARIANT_ROUTES,
        AICCM_VARIANTS,
        PeriodicJKMethod,
        _AICCM_JK_METHODS,
        describe_jk_method,
    )

    assert _AICCM_JK_METHODS == frozenset(
        {
            PeriodicJKMethod.AICCM2026DEV_A,
            PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
            PeriodicJKMethod.NEUTRAL_BLOCH,
            PeriodicJKMethod.AICCM2026DEV_B,
        }
    )
    assert _AICCM_JK_METHODS == frozenset(AICCM_VARIANT_ROUTES.values())
    assert tuple(AICCM_VARIANT_ROUTES) == AICCM_VARIANTS == (
        "real-gamma", "neutral-bloch", "four-center", "chi"
    )
    assert PeriodicJKMethod.NEUTRAL_BLOCH.value == "neutral-bloch"
    for member in _AICCM_JK_METHODS:
        text = describe_jk_method(member)
        assert text.startswith(f"aiccm ({AICCM_VARIANT_OF[member]})"), text
        assert "experimental" in text.rsplit(";", 1)[-1], text


def test_periodic_manifest_stamps_method_status_only_for_aiccm_routes(tmp_path):
    """Every job dispatched through an AICCM selector writes
    [run].method_status = "experimental" into the .system manifest before any
    compute (a dry-run already carries it) and records the real-Gamma selector
    as "real-gamma"; a shipped route writes no marker, so absence is never a
    claim."""
    import tomllib

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [4.0, 4.0, 3.3]), vq.Atom(1, [4.0, 4.0, 4.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def run_section(tag, **kwargs):
        stem = tmp_path / f"status-{tag}"
        kwargs.setdefault("method", "RHF")
        if (
            kwargs.get("jk_method") in {"real-gamma", "aiccm2026dev-a"}
            or kwargs.get("variant") in {"real-gamma", "four-center"}
        ):
            kwargs.setdefault("initial_guess", "HCORE")
        assert vq.run_periodic_job(
            system, basis, kpoints=(2, 1, 1), output=stem,
            dry_run=True, progress=False, **kwargs,
        ) is None
        with stem.with_suffix(".system").open("rb") as fh:
            return tomllib.load(fh)["run"]

    # Legacy spellings (M1): still dispatch, still stamped, now warned and
    # recorded as the legacy selector surface.
    with pytest.warns(DeprecationWarning, match="variant='real-gamma'"):
        real_gamma = run_section("real-gamma", jk_method="real-gamma")
    assert real_gamma["method_status"] == "experimental"
    for key in ("jk_method_requested", "jk_method_resolved", "jk_method_executed"):
        assert real_gamma[key] == "real-gamma", key
    assert real_gamma["aiccm_variant"] == "real-gamma"
    assert real_gamma["aiccm_selector"] == "legacy-jk_method"
    with pytest.warns(DeprecationWarning, match="variant='chi'"):
        chi = run_section("chi", jk_method="aiccm2026dev-b", aiccm_backend="four_center")
    assert chi["method_status"] == "experimental"
    assert chi["aiccm_variant"] == "chi"
    assert chi["aiccm_selector"] == "legacy-jk_method"
    # Front door: same stamp; jk_method_requested / _resolved / _executed all
    # record the variant's own route; the selector surface is recorded.
    for variant, route in (("real-gamma", "real-gamma"), ("chi", "aiccm2026dev-b")):
        run = run_section(f"fd-{variant}", method="aiccm", variant=variant)
        assert run["method_status"] == "experimental", variant
        assert run["aiccm_variant"] == variant
        assert run["aiccm_selector"] == "front-door"
        for key in ("jk_method_requested", "jk_method_resolved", "jk_method_executed"):
            assert run[key] == route, (variant, key)
    # Negative control: a shipped route carries no marker at all.
    gdf = run_section("gdf", jk_method="gdf")
    assert "method_status" not in gdf and "aiccm_variant" not in gdf


# ---------------------------------------------------------------------------
# AICCM front door M1 (handovers/HANDOVER_AICCM_STANDARD_METHOD.md, D-2, D-3,
# D-8): method="aiccm", variant=... through the public runner surface only;
# nothing here imports vibeqc.periodic.ccm.
# ---------------------------------------------------------------------------

_AICCM_VARIANTS = ("real-gamma", "neutral-bloch", "four-center", "chi")


def _aiccm_h2(multiplicity=1):
    atoms = (
        [vq.Atom(1, [4.0, 4.0, 3.3]), vq.Atom(1, [4.0, 4.0, 4.7])]
        if multiplicity == 1
        else [vq.Atom(1, [4.0, 4.0, 4.0])]
    )
    system = vq.PeriodicSystem(3, np.eye(3) * 8.0, atoms, 0, multiplicity)
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _aiccm_dry(tmp_path, tag, *, multiplicity=1, **kw):
    import tomllib

    system, basis = _aiccm_h2(multiplicity)
    stem = tmp_path / tag
    if (
        kw.get("jk_method") in {"real-gamma", "aiccm2026dev-a"}
        or kw.get("variant") in {"real-gamma", "four-center"}
    ):
        kw.setdefault("initial_guess", "HCORE")
    assert vq.run_periodic_job(
        system, basis, output=stem, dry_run=True, progress=False, **kw
    ) is None
    with stem.with_suffix(".system").open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize(
    "variant,route",
    [
        ("real-gamma", "real-gamma"),
        ("neutral-bloch", "neutral-bloch"),
        ("four-center", "aiccm2026dev-a"),
        ("chi", "aiccm2026dev-b"),
    ],
)
def test_aiccm_front_door_dispatches_each_wired_variant(tmp_path, variant, route):
    m = _aiccm_dry(
        tmp_path, f"fd-{variant}", method="aiccm", variant=variant,
        aiccm_lattice_extension=(2, 1, 1),
    )
    assert m["run"]["jk_method_resolved"] == route == m["run"]["jk_method_executed"]
    assert m["run"]["method_status"] == "experimental"
    assert m["run"]["aiccm_variant"] == variant
    assert m["run"]["aiccm_selector"] == "front-door"


def test_aiccm_front_door_has_no_unwired_variants_left(tmp_path):
    """M3a wired the four-centre construction, the last variant that failed
    closed, so every value of ``variant`` now dispatches. The mechanism that
    used to refuse one stays: ``_UNWIRED_CCM_ROUTES`` is empty rather than
    deleted, so the next formulation fails closed with a library pointer
    instead of an AttributeError."""
    from vibeqc.periodic_jk_method import (
        AICCM_VARIANT_ROUTES,
        _IMPLEMENTED,
        _UNWIRED_CCM_ROUTES,
    )

    assert _UNWIRED_CCM_ROUTES == frozenset()
    for variant, member in AICCM_VARIANT_ROUTES.items():
        assert member in _IMPLEMENTED, variant


def test_aiccm_front_door_requires_a_variant(tmp_path):
    system, basis = _aiccm_h2()
    stem = tmp_path / "fd-novariant"
    with pytest.raises(ValueError, match="requires variant=") as exc:
        vq.run_periodic_job(
            system, basis, method="aiccm", output=stem, dry_run=True, progress=False
        )
    for v in _AICCM_VARIANTS:
        assert v in str(exc.value), v
    assert not stem.with_suffix(".system").exists()
    with pytest.raises(ValueError, match="unknown AICCM variant"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant="bogus", output=stem,
            dry_run=True, progress=False,
        )
    # The bare gamma spellings are not variants either (ruling R1, D-2).
    with pytest.raises(ValueError, match="ruling R1"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant="gamma", output=stem,
            dry_run=True, progress=False,
        )
    # variant= / scf_reference= with any other method fail closed, never ignored.
    for stray in ({"variant": "chi"}, {"scf_reference": "rohf"}):
        with pytest.raises(ValueError, match="require method='aiccm'"):
            vq.run_periodic_job(
                system, basis, method="RHF", output=stem, dry_run=True,
                progress=False, **stray,
            )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "variant,own_route,other",
    [
        ("real-gamma", "real-gamma", "aiccm2026dev-b"),
        ("chi", "aiccm2026dev-b", "gdf"),
        ("chi", "chi-ccm", "real-gamma"),  # legacy spelling of the same route
        ("neutral-bloch", "neutral-bloch", "gdf"),
    ],
)
def test_aiccm_front_door_jk_method_must_match_the_variant(
    tmp_path, variant, own_route, other
):
    import warnings

    system, basis = _aiccm_h2()
    stem = tmp_path / f"fd-conflict-{variant}"
    with pytest.raises(ValueError, match="Leave jk_method='auto'") as exc:
        vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant, jk_method=other,
            aiccm_lattice_extension=(2, 1, 1), output=stem, dry_run=True,
            progress=False,
        )
    assert variant in str(exc.value) and other in str(exc.value)
    assert not stem.with_suffix(".system").exists()
    # Explicitly equal to the variant's route: accepted, and no deprecation.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant, jk_method=own_route,
            aiccm_lattice_extension=(2, 1, 1), output=stem, dry_run=True,
            progress=False,
            initial_guess=("HCORE" if variant == "real-gamma" else "SAD"),
        ) is None


@pytest.mark.parametrize("variant", ["real-gamma", "neutral-bloch", "chi"])
@pytest.mark.parametrize(
    "functional,multiplicity,expected",
    [(None, 1, "RHF"), (None, 2, "UHF"), ("pbe", 1, "RKS"), ("pbe", 2, "UKS")],
)
def test_aiccm_front_door_infers_the_scf_reference(
    tmp_path, variant, functional, multiplicity, expected
):
    """D-3: functional decides HF versus KS; the system's multiplicity and
    electron parity decide restricted versus unrestricted. The inferred
    reference is what [plan].method records."""
    m = _aiccm_dry(
        tmp_path, f"ref-{variant}-{expected}", multiplicity=multiplicity,
        method="aiccm", variant=variant, functional=functional,
        aiccm_lattice_extension=(2, 1, 1),
    )
    assert m["plan"]["method"] == expected
    assert m["run"]["method_status"] == "experimental"


@pytest.mark.parametrize("variant", ["real-gamma", "neutral-bloch", "chi"])
@pytest.mark.parametrize(
    "scf_reference,functional", [("rohf", None), ("roks", "pbe")]
)
def test_aiccm_front_door_restricted_open_shell_fails_closed(
    tmp_path, variant, scf_reference, functional
):
    """scf_reference='rohf' / 'roks' select the references explicitly; no
    wired variant implements them, so the request fails closed naming the
    reference rather than being downgraded to UHF/UKS."""
    system, basis = _aiccm_h2(2)
    stem = tmp_path / f"ro-{variant}-{scf_reference}"
    with pytest.raises(NotImplementedError, match=scf_reference.upper()):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant,
            scf_reference=scf_reference, functional=functional,
            aiccm_lattice_extension=(2, 1, 1), output=stem, dry_run=True,
            progress=False,
            initial_guess=("HCORE" if variant == "real-gamma" else "SAD"),
        )
    assert not stem.with_suffix(".system").exists()
    # The two explicit references check their functional consistency first.
    with pytest.raises(ValueError, match="scf_reference='rohf'"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant, scf_reference="rohf",
            functional="pbe", output=stem, dry_run=True, progress=False,
        )
    with pytest.raises(ValueError, match="scf_reference='roks'"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant, scf_reference="roks",
            output=stem, dry_run=True, progress=False,
        )
    with pytest.raises(ValueError, match="unknown scf_reference"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant, scf_reference="uhf",
            output=stem, dry_run=True, progress=False,
        )


@pytest.mark.parametrize(
    "jk,variant",
    [
        ("aiccm2026dev-b", "chi"), ("chi", "chi"), ("chi-ccm", "chi"),
        ("real-gamma", "real-gamma"), ("real_gamma", "real-gamma"),
    ],
)
def test_aiccm_legacy_jk_spellings_warn_and_name_the_front_door(tmp_path, jk, variant):
    """Every dev-era spelling still dispatches, and warns exactly once per
    call with a message naming method='aiccm' and the matching variant."""
    system, basis = _aiccm_h2()
    stem = tmp_path / f"legacy-{jk}"
    with pytest.warns(DeprecationWarning) as rec:
        assert vq.run_periodic_job(
            system, basis, method="RHF", jk_method=jk, kpoints=(2, 1, 1),
            output=stem, dry_run=True, progress=False,
            initial_guess=("HCORE" if variant == "real-gamma" else "SAD"),
        ) is None
    msgs = [
        str(r.message) for r in rec
        if r.category is DeprecationWarning
        and "deprecated AICCM selector" in str(r.message)
    ]
    assert len(msgs) == 1, msgs
    assert "method='aiccm'" in msgs[0] and f"variant='{variant}'" in msgs[0]
    assert stem.with_suffix(".system").exists()
    # (The .out pointer line is pinned on a real run in
    # tests/test_periodic_ccm_real_gamma_adapter.py; a dry run writes none.)


def test_aiccm_deprecation_is_attributed_to_the_caller_not_to_vibeqc(tmp_path):
    """A DeprecationWarning blamed on a vibe-qc frame is invisible to every
    user: Python's default filters show one only when it is attributed to
    ``__main__`` and swallow the rest. The entry points sit at different
    depths (the resolver three frames down, ``run_periodic_job`` five, its
    output-writer decorator six), so no fixed ``stacklevel`` serves them all;
    the warning must be attributed to the first frame OUTSIDE
    ``python/vibeqc``. Regression pin: before this, a legacy spelling passed
    to ``run_periodic_job`` was blamed on ``periodic_runner.py`` and printed
    nothing at all in a plain script."""
    import warnings
    from pathlib import Path

    import vibeqc

    package_dir = Path(vibeqc.__file__).resolve().parent
    system, basis = _aiccm_h2()

    def attribution(**kwargs):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            vq.run_periodic_job(
                system, basis, kpoints=(2, 1, 1), output=tmp_path / "attrib",
                dry_run=True, progress=False, **kwargs,
            )
        hits = [
            r for r in rec
            if r.category is DeprecationWarning
            and "deprecated AICCM selector" in str(r.message)
        ]
        assert len(hits) == 1, [str(r.message) for r in hits]
        return Path(hits[0].filename).resolve()

    for kwargs in (
        {
            "method": "RHF",
            "jk_method": "real-gamma",
            "initial_guess": "HCORE",
        },
        {"method": "RHF", "jk_method": vq.PeriodicJKMethod.AICCM2026DEV_B},
    ):
        blamed = attribution(**kwargs)
        assert blamed == Path(__file__).resolve(), (kwargs, blamed)
        assert package_dir not in blamed.parents, (kwargs, blamed)

    # The resolver, called directly, blames its own caller too.
    from vibeqc.periodic_jk_method import resolve_jk_method_string

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        resolve_jk_method_string("chi-ccm")
    (hit,) = [r for r in rec if r.category is DeprecationWarning]
    assert Path(hit.filename).resolve() == Path(__file__).resolve()


def test_aiccm_legacy_enum_member_warns_and_neutral_bloch_member_is_refused(tmp_path):
    """A PeriodicJKMethod member passed directly bypasses the string resolver,
    so the runner warns for the three legacy members itself; the front-door-
    only NEUTRAL_BLOCH member is not a jk_method at all."""
    system, basis = _aiccm_h2()
    with pytest.warns(DeprecationWarning, match="variant='chi'"):
        assert vq.run_periodic_job(
            system, basis, method="RHF", jk_method=vq.PeriodicJKMethod.AICCM2026DEV_B,
            kpoints=(2, 1, 1), output=tmp_path / "legacy-member", dry_run=True,
            progress=False,
        ) is None
    with pytest.raises(ValueError, match="only through method='aiccm', variant='neutral-bloch'"):
        vq.run_periodic_job(
            system, basis, method="RHF", jk_method=vq.PeriodicJKMethod.NEUTRAL_BLOCH,
            kpoints=(2, 1, 1), output=tmp_path / "nb-member", dry_run=True,
            progress=False,
        )


def test_aiccm_legacy_construction_spelling_warns_and_dispatches(tmp_path):
    """The dev-era four-centre spelling warns and, since M3a, dispatches: the
    construction it names now has a runner arm. Before M3a this test pinned
    the warning arriving BEFORE the fail-closed pointer."""
    system, basis = _aiccm_h2()
    stem = tmp_path / "legacy-a"
    with pytest.warns(DeprecationWarning, match="variant='four-center'"):
        assert vq.run_periodic_job(
            system, basis, method="RHF", jk_method="aiccm2026dev-a",
            kpoints=(2, 1, 1), output=stem, dry_run=True, progress=False,
            initial_guess="HCORE",
        ) is None
    import tomllib

    with stem.with_suffix(".system").open("rb") as fh:
        run = tomllib.load(fh)["run"]
    assert run["aiccm_variant"] == "four-center"
    assert run["aiccm_selector"] == "legacy-jk_method"


@pytest.mark.parametrize("variant", ["real-gamma", "neutral-bloch", "chi"])
def test_aiccm_front_door_is_silent(tmp_path, variant):
    import warnings

    system, basis = _aiccm_h2()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant,
            aiccm_lattice_extension=(2, 1, 1), output=tmp_path / f"silent-{variant}",
            dry_run=True, progress=False,
            initial_guess=("HCORE" if variant == "real-gamma" else "SAD"),
        ) is None


@pytest.mark.parametrize(
    "jk,needle",
    [
        ("gamma", "ruling R1"), ("gamma-ccm", "ruling R1"), ("gamma_ccm", "ruling R1"),
        ("neutral-bloch", "variant='neutral-bloch'"),
        ("neutral_bloch", "variant='neutral-bloch'"),
        ("bloch-control", "variant='neutral-bloch'"),
        ("gdf-control", "variant='neutral-bloch'"),
        ("aiccm-ri", "variant='neutral-bloch'"),
    ],
)
def test_aiccm_retired_jk_spellings_fail_closed_on_the_runner(tmp_path, jk, needle):
    """D-2 / D-2b on the runner surface (not only the resolver): no .system is
    written, the R1 family names both neutral producers, the GDF-alias family
    names the library entry that runs today; plain GDF is untouched."""
    system, basis = _aiccm_h2()
    stem = tmp_path / f"retired-{jk}"
    with pytest.raises(ValueError) as exc:
        vq.run_periodic_job(
            system, basis, method="RHF", jk_method=jk, kpoints=(2, 1, 1),
            output=stem, dry_run=True, progress=False,
        )
    assert needle in str(exc.value)
    if needle == "ruling R1":
        assert "variant='real-gamma'" in str(exc.value)
    else:
        assert "route='neutral-bloch'" in str(exc.value)
    assert not stem.with_suffix(".system").exists()
    assert vq.run_periodic_job(
        system, basis, method="RHF", jk_method="gdf", kpoints=(2, 1, 1),
        output=tmp_path / "gdf-negative", dry_run=True, progress=False,
    ) is None


@pytest.mark.parametrize("variant", ["real-gamma", "neutral-bloch", "chi"])
def test_aiccm_front_door_torus_keyword_and_not_both_rule(tmp_path, variant):
    """Plan section 3.4: one torus keyword for every variant; the k-mesh
    stays an alias; both together fail closed before any artefact."""
    system, basis = _aiccm_h2()
    initial_guess = "HCORE" if variant == "real-gamma" else "SAD"
    assert vq.run_periodic_job(
        system, basis, method="aiccm", variant=variant,
        aiccm_lattice_extension=(2, 1, 1), output=tmp_path / f"torus-{variant}",
        dry_run=True, progress=False, initial_guess=initial_guess,
    ) is None
    assert vq.run_periodic_job(
        system, basis, method="aiccm", variant=variant, kpoints=(2, 1, 1),
        output=tmp_path / f"alias-{variant}", dry_run=True, progress=False,
        initial_guess=initial_guess,
    ) is None
    stem = tmp_path / f"both-{variant}"
    with pytest.raises(ValueError, match="not both"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant,
            aiccm_lattice_extension=(2, 1, 1), kpoints=(2, 1, 1),
            output=stem, dry_run=True, progress=False,
            initial_guess=initial_guess,
        )
    assert not stem.with_suffix(".system").exists()


def test_aiccm_real_gamma_accepts_the_odd_shell_shorthand(tmp_path):
    """aiccm_wigner_seitz_shells=s is the odd-extension shorthand 2s+1 for
    every variant, so the real-Gamma producer accepts it too (the mesh it
    derives is pinned on a real run in
    tests/test_periodic_ccm_real_gamma_adapter.py)."""
    system, basis = _aiccm_h2()
    assert vq.run_periodic_job(
        system, basis, method="aiccm", variant="real-gamma",
        aiccm_wigner_seitz_shells=(1, 0, 0), output=tmp_path / "shells",
        dry_run=True, progress=False, initial_guess="HCORE",
    ) is None


@pytest.mark.parametrize(
    "unsupported",
    [{"damping": 0.3}, {"level_shift": 0.5}, {"fock_mixing": 0.3}],
    ids=lambda d: next(iter(d)),
)
def test_aiccm_real_gamma_rejects_a_criterion_it_cannot_honour(tmp_path, unsupported):
    """The real-Gamma loop forwards max_iter and conv_tol_energy only; an
    explicit knob it would silently drop fails closed before SCF (CLAUDE.md
    section 7), naming the knob and its value."""
    system, basis = _aiccm_h2()
    stem = tmp_path / "crit"
    with pytest.raises(NotImplementedError, match="variant='real-gamma'") as exc:
        vq.run_periodic_job(
            system, basis, method="aiccm", variant="real-gamma",
            aiccm_lattice_extension=(2, 1, 1), output=stem, dry_run=True,
            progress=False, initial_guess="HCORE", **unsupported,
        )
    (name, value), = unsupported.items()
    assert f"{name}={value!r}" in str(exc.value)
    assert not stem.with_suffix(".system").exists()


def test_aiccm_variant_is_explicit_for_the_dftu_auto_fallback(tmp_path):
    """With method='aiccm' the variant fixes the Coulomb Hamiltonian as firmly
    as an explicit jk_method: a dft_plus_u request is refused by the variant's
    own guard and never re-routed onto BIPOLE by the AUTO fallback."""
    from vibeqc.dft_plus_u import HubbardSite

    system, basis = _aiccm_h2()
    stem = tmp_path / "fd-dftu"
    with pytest.raises(NotImplementedError, match=r"DFT\+U"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant="real-gamma", functional="pbe",
            dft_plus_u=[HubbardSite(0, 0, 1.0)], aiccm_lattice_extension=(2, 1, 1),
            output=stem, dry_run=True, progress=False, initial_guess="HCORE",
        )
    assert not stem.with_suffix(".system").exists()


# ---------------------------------------------------------------------------
# The per-unit-cell artefact contract (AICCM front door M2, #655; the defect
# it closes is #654). Route-agnostic: an artefact that pairs MO coefficients
# with the unit cell's basis and structure must never carry orbitals from a
# larger AO space. Reached through the public runner only.
# ---------------------------------------------------------------------------


def _qvf_wavefunction_shapes(path):
    """Return (n_mo, n_ao, coefficient_floats, complex_encoded) or None.

    ``None`` means the archive carries no ``wavefunction`` section at all,
    which is the correct outcome when the result's orbitals do not span the
    unit-cell basis.
    """
    import json
    import zipfile

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "wavefunction/mo_metadata.json" not in names:
            return None
        meta = json.loads(archive.read("wavefunction/mo_metadata.json"))
        floats = len(archive.read("wavefunction/mo_coefficients.dat")) // 8
    complex_encoded = str(meta.get("coefficient_encoding", "")).startswith("complex")
    return int(meta["n_mo"]), int(meta["n_ao"]), floats, complex_encoded


@pytest.mark.parametrize("variant", ["real-gamma", "neutral-bloch"])
@pytest.mark.parametrize("torus", [(1, 1, 1), (2, 1, 1)])
def test_periodic_qvf_wavefunction_is_consistent_or_absent(tmp_path, variant, torus):
    """#654: a real-Gamma job on a torus larger than one cell used to ship the
    SUPERCELL coefficient matrix in the QVF wavefunction section while the
    archive's basis and structure described the unit cell, and
    ``validate_qvf`` passed it. Measured then: n_mo=4, n_ao=2 and 16 float64
    at a (2,1,1) torus, where a reader reshaping n_mo x n_ao takes 8. Either
    the section is omitted or its metadata describes its own data, and its
    AO count is the unit cell's."""
    import warnings

    from vibeqc.output.formats.qvf import validate_qvf

    system, basis = _aiccm_h2()
    stem = tmp_path / f"qvf-{variant}-{''.join(str(n) for n in torus)}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant,
            aiccm_lattice_extension=torus, output=stem, progress=False,
            output_qvf=True,
            initial_guess=("HCORE" if variant == "real-gamma" else "SAD"),
        )
    assert result.converged
    assert validate_qvf(str(stem.with_suffix(".qvf")))["valid"]
    shapes = _qvf_wavefunction_shapes(stem.with_suffix(".qvf"))
    if shapes is None:
        # Omitted: the .out must say so rather than leaving it unexplained.
        out = stem.with_suffix(".out").read_text(encoding="utf-8")
        assert "QVF wavefunction    = omitted" in out
        return
    n_mo, n_ao, floats, complex_encoded = shapes
    assert n_ao == int(basis.nbasis), (variant, torus, n_ao, basis.nbasis)
    assert floats == n_mo * n_ao * (2 if complex_encoded else 1), (
        variant, torus, shapes
    )


def test_real_gamma_supercell_orbitals_never_reach_a_unit_cell_artefact(tmp_path):
    """The real-Gamma producer retains supercell-Gamma MOs by design (its
    energy and density ARE folded per unit cell), so every artefact that
    pairs orbitals with the unit-cell basis must refuse them above a
    one-cell torus: the Molden sidecar already did, through its own
    capability gate, and an explicit request fails closed; the QVF
    wavefunction section now does too."""
    import warnings

    system, basis = _aiccm_h2()
    stem = tmp_path / "rg-supercell"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vq.run_periodic_job(
            system, basis, method="aiccm", variant="real-gamma",
            aiccm_lattice_extension=(2, 1, 1), output=stem, progress=False,
            output_qvf=True, initial_guess="HCORE",
        )
    # The orbitals really are supercell-wide: two cells times two functions.
    from vibeqc.periodic_runner import _sidecar_mo_ao_dimension

    assert _sidecar_mo_ao_dimension(result) == 2 * int(basis.nbasis)
    assert _qvf_wavefunction_shapes(stem.with_suffix(".qvf")) is None
    assert not stem.with_suffix(".molden").exists()
    assert not stem.with_suffix(".population.txt").exists()
    # An explicit sidecar request is refused rather than silently dropped.
    with pytest.raises(NotImplementedError, match="write_molden_file"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant="real-gamma",
            aiccm_lattice_extension=(2, 1, 1), output=tmp_path / "rg-explicit",
            progress=False, dry_run=True, write_molden_file=True,
            initial_guess="HCORE",
        )


def test_unit_cell_basis_span_helper_is_conservative():
    """The gate refuses a KNOWN mismatch and never withdraws an artefact whose
    dimensions cannot be read, so a result type it does not understand keeps
    the historical behaviour."""
    from vibeqc.periodic_runner import (
        _mo_coeffs_span_unit_cell_basis,
        _sidecar_mo_ao_dimension,
    )

    class _Result:
        def __init__(self, coeffs):
            self.mo_coeffs = coeffs

    class _Basis:
        nbasis = 2

    assert _sidecar_mo_ao_dimension(_Result(None)) is None
    assert _mo_coeffs_span_unit_cell_basis(_Result(None), _Basis()) is True
    assert _mo_coeffs_span_unit_cell_basis(_Result(np.zeros((2, 2))), _Basis()) is True
    assert _mo_coeffs_span_unit_cell_basis(_Result(np.zeros((4, 4))), _Basis()) is False
    # Multi-k results carry a list of per-k blocks with the same AO leading
    # dimension; the first block is representative.
    assert _sidecar_mo_ao_dimension(_Result([np.zeros((2, 2))] * 3)) == 2
    assert _mo_coeffs_span_unit_cell_basis(_Result([np.zeros((2, 2))] * 3), _Basis())
