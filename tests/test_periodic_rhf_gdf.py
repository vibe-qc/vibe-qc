"""Native Γ-only periodic RHF/GDF smoke tests.

These tests exercise the first self-hosted periodic GDF SCF route. They
deliberately do not import PySCF; external-program parity is handled by
the regression harness out of process.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq


def _h2_box(box: float = 12.0):
    c = box / 2.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [
            vq.Atom(1, [c, c, c - 0.7]),
            vq.Atom(1, [c, c, c + 0.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _h2_dimensional_box(dim: int, box: float = 12.0, vacuum: float = 40.0):
    if dim == 1:
        lattice = np.diag([box, vacuum, vacuum])
        centre = np.array([0.5 * box, 0.5 * vacuum, 0.5 * vacuum])
    elif dim == 2:
        lattice = np.column_stack(
            [
                [box, 0.0, 0.0],
                [0.5 * box, 0.5 * np.sqrt(3.0) * box, 0.0],
                [0.0, 0.0, vacuum],
            ]
        )
        centre = lattice @ np.array([0.5, 0.5, 0.5])
    elif dim == 3:
        lattice = np.eye(3) * box
        centre = np.array([0.5 * box, 0.5 * box, 0.5 * box])
    else:
        raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
    system = vq.PeriodicSystem(
        dim,
        lattice,
        [
            vq.Atom(1, (centre + np.array([0.0, 0.0, -0.7])).tolist()),
            vq.Atom(1, (centre + np.array([0.0, 0.0, 0.7])).tolist()),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _h2_hexagonal_box(a: float = 12.0):
    lattice = np.column_stack(
        [
            [a, 0.0, 0.0],
            [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, a],
        ]
    )
    centre = lattice @ np.array([0.5, 0.5, 0.5])
    atoms = [
        vq.Atom(1, (centre + np.array([0.0, 0.0, -0.7])).tolist()),
        vq.Atom(1, (centre + np.array([0.0, 0.0, 0.7])).tolist()),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lih_primitive_rocksalt():
    """Primitive FCC LiH rocksalt cell: 2 atoms / 1 formula unit."""
    a = 4.084 / 0.529177210903
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(3, [0.0, 0.0, 0.0]),
            vq.Atom(1, [a / 2.0, a / 2.0, a / 2.0]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _mgo_p01_primitive():
    """P01 MgO/STO-3G primitive cell from the prompt-11 parity audit."""
    a = 4.212 / 0.529177210903
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, [0.5 * a] * 3)],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _opts(max_iter: int = 40):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    opts.max_iter = max_iter
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    opts.damping = 0.2
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.initial_guess = vq.InitialGuess.HCORE
    return opts


def _ks_opts(functional: str, max_iter: int = 40):
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    opts.max_iter = max_iter
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    opts.damping = 0.2
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.initial_guess = vq.InitialGuess.HCORE
    return opts


def test_gamma_rhf_gdf_converges_without_external_backend():
    system, basis = _h2_box()

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        _opts(),
        aux_basis="def2-svp-jk",
        fock_mixing=0.3,
        progress=False,
    )

    assert result.converged
    assert result.backend == "native-gamma-gdf"
    assert result.fock_mixing == pytest.approx(0.3)
    assert result.aux_basis_name == "def2-svp-jk"
    assert result.n_aux > 0
    assert result.n_fit > 0
    assert -2.0 < result.energy < 0.0
    assert result.density.shape == (basis.nbasis, basis.nbasis)
    assert result.fock.shape == (basis.nbasis, basis.nbasis)
    assert result.overlap.shape == (basis.nbasis, basis.nbasis)
    assert len(result.scf_trace) >= 2
    assert result.free_energy == pytest.approx(result.energy)
    assert result.entropy == pytest.approx(0.0)
    np.testing.assert_allclose(result.occupations, [2.0, 0.0])


@pytest.mark.parametrize("dim", (1, 2))
def test_gamma_rhf_gdf_private_ewald_force_requires_3d(dim):
    system, basis = _h2_dimensional_box(dim)

    with pytest.raises(ValueError, match="requires a fully 3D-periodic system"):
        vq.run_rhf_periodic_gamma_gdf(
            system,
            basis,
            _opts(),
            _force_ewald_jk=True,
            progress=False,
        )


@pytest.mark.parametrize(
    "dim, expected_energy",
    [
        # 1D / 2D vacuum-padded H₂: Steps 1 + 2 of the 2026-05-13
        # gauge fix are no-ops (Ewald-3D is 3D-only on the C++ side,
        # so dim<3 falls through to the bare lattice sum that 1D / 2D
        # wires + slabs always used).  Pinned values unchanged from
        # pre-fix.
        (1, -1.116738328690913),
        (2, -1.116738328690913),
        # 3D-periodic H2 with a molecular-limit GDF cutoff uses the
        # internally consistent native direct-truncated gauge, matching
        # the dim<3 vacuum-padded H2 value. Tight 3D cells with image
        # cells in the GDF cutoff still route to the Ewald-JK fallback.
        (3, -1.116738328690913),
    ],
)
def test_gamma_rhf_gdf_supports_periodic_dimensionalities(dim, expected_energy):
    system, basis = _h2_dimensional_box(dim)

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        _opts(),
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert int(system.dim) == dim
    assert result.converged
    assert result.backend == "native-gamma-gdf"
    assert result.energy == pytest.approx(expected_energy, abs=1e-9)
    assert np.isfinite(result.energy)


def test_gamma_rhf_gdf_smearing_preserves_electron_count():
    system, basis = _h2_box()
    opts = _opts()
    opts.smearing_temperature = 0.02

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert result.converged
    assert result.smearing_temperature == pytest.approx(0.02)
    assert result.entropy >= 0.0
    assert result.free_energy == pytest.approx(
        result.energy - result.smearing_temperature * result.entropy,
        abs=1e-12,
    )
    assert float(np.sum(result.occupations)) == pytest.approx(
        float(system.n_electrons()),
        abs=1e-9,
    )
    assert np.all(result.occupations >= -1e-12)
    assert np.all(result.occupations <= 2.0 + 1e-12)


def test_k_gdf_per_k_density_fold_matches_fractional_builder():
    from vibeqc.periodic_k_density import real_space_density_from_per_k_density

    system, basis = _h2_box()
    opts = _opts()
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    S_lat = vq.compute_overlap_lattice(basis, system, opts.lattice_opts)
    rng = np.random.default_rng(123)

    C_per_k = []
    occ_per_k = []
    D_per_k = []
    for _ in kmesh.kpoints:
        raw = rng.standard_normal((basis.nbasis, basis.nbasis)) + (
            1j * rng.standard_normal((basis.nbasis, basis.nbasis))
        )
        C, _ = np.linalg.qr(raw)
        occ = np.linspace(1.7, 0.3, basis.nbasis)
        D = (C * occ[None, :].astype(complex)) @ C.conj().T
        D = 0.5 * (D + D.conj().T)
        C_per_k.append(C)
        occ_per_k.append(occ)
        D_per_k.append(D)

    folded_from_density = real_space_density_from_per_k_density(
        D_per_k,
        kmesh,
        S_lat.cells,
    )
    folded_from_occupations = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            C_per_k,
            occ_per_k,
            kmesh,
            S_lat.cells,
        )
    )
    for got, expected in zip(
        folded_from_density.blocks,
        folded_from_occupations.blocks,
    ):
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_gamma_rhf_gdf_level_shift_warmup_restarts_unshifted(tmp_path):
    system, basis = _h2_box()
    opts = _opts(max_iter=20)
    opts.level_shift = 0.5
    log_path = tmp_path / "h2-gdf-level-shift.out"
    plog = vq.ProgressLogger(log_path=log_path, verbose=True)

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        level_shift_warmup_cycles=2,
        progress=plog,
    )

    text = log_path.read_text()
    assert result.converged
    assert result.level_shift == pytest.approx(0.5)
    assert result.level_shift_warmup_cycles == 2
    # This molecular-limit cutoff uses native GDF in a direct-truncated
    # gauge; pinning it catches accidental fallback/gauge drift while
    # the main point of the test remains level-shift warmup behaviour.
    assert result.energy == pytest.approx(-1.116738328690913, abs=1e-9)
    assert "level-shift warm-up: 2 cycles at 0.500 Ha" in text
    assert "restart: unshifted Fock with fresh DIIS history" in text


def test_gamma_rhf_gdf_level_shift_warmup_rejects_bad_length():
    system, basis = _h2_box()
    opts = _opts(max_iter=5)
    opts.level_shift = 0.5

    with pytest.raises(ValueError, match="level_shift_warmup_cycles"):
        vq.run_rhf_periodic_gamma_gdf(
            system,
            basis,
            opts,
            aux_basis="def2-svp-jk",
            level_shift_warmup_cycles=-2,
            progress=False,
        )


def test_gamma_rhf_gdf_accepts_skew_hexagonal_lattice():
    system, basis = _h2_hexagonal_box()

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        _opts(),
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert result.converged
    assert result.backend == "native-gamma-gdf"
    assert np.isfinite(result.energy)


@pytest.mark.parametrize("functional", ["pbe", "b3lyp"])
def test_gamma_rks_gdf_converges_without_external_backend(functional):
    system, basis = _h2_box()
    opts = _ks_opts(functional)

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert result.converged
    assert result.functional == functional
    assert result.e_xc != pytest.approx(0.0)
    if functional == "pbe":
        assert result.e_hf_exchange == pytest.approx(0.0)
    else:
        assert result.e_hf_exchange != pytest.approx(0.0)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("functional", ["pbe", "b3lyp"])
def test_gamma_rks_gdf_supports_periodic_dimensionalities(dim, functional):
    system, basis = _h2_dimensional_box(dim)

    result = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        _ks_opts(functional),
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert int(system.dim) == dim
    assert result.converged
    assert result.functional == functional
    assert np.isfinite(result.energy)
    assert result.e_xc != pytest.approx(0.0)
    if functional == "pbe":
        assert result.e_hf_exchange == pytest.approx(0.0)
    else:
        assert result.e_hf_exchange != pytest.approx(0.0)


def test_periodic_jk_auto_resolves_to_native_gdf():
    method = vq.pick_jk_method(
        "auto",
        lattice=np.eye(3) * 12.0,
        basis_name="sto-3g",
        n_atoms=2,
    )
    assert method is vq.PeriodicJKMethod.GDF


def test_run_periodic_job_rks_gdf_smoke(tmp_path):
    system, basis = _h2_box()
    out = tmp_path / "h2-rks-gdf"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pbe",
        output=out,
        aux_basis="def2-svp-jk",
        max_iter=20,
        conv_tol_energy=1e-9,
        damping=0.2,
        fmixing_percent=30.0,
        write_molden_file=False,
        progress=False,
    )

    assert result.converged
    assert result.functional == "pbe"
    assert result.fock_mixing == pytest.approx(0.3)
    assert out.with_suffix(".out").exists()
    assert out.with_suffix(".system").exists()
    assert "fmixing_percent     = 30.0" in out.with_suffix(".out").read_text()


@pytest.mark.parametrize(
    "dim,measure_label",
    [(1, "periodic length"), (2, "periodic area"), (3, "cell volume")],
)
def test_run_periodic_job_logs_periodic_dimensionality(
    tmp_path,
    dim,
    measure_label,
):
    system, basis = _h2_dimensional_box(dim)
    out = tmp_path / f"h2-dim{dim}-rhf-gdf"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        output=out,
        aux_basis="def2-svp-jk",
        max_iter=40,
        conv_tol_energy=1e-9,
        damping=0.2,
        initial_guess="HCORE",
        write_molden_file=False,
        progress=False,
    )

    text = out.with_suffix(".out").read_text()
    assert result.converged
    assert f"dimensionality = {dim}D" in text
    assert measure_label in text
    if dim < 3:
        assert "embedding volume" in text


def test_run_periodic_job_forwards_smearing_to_gdf(tmp_path):
    system, basis = _h2_box()
    out = tmp_path / "h2-rhf-gdf-smearing"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        output=out,
        aux_basis="def2-svp-jk",
        max_iter=40,
        conv_tol_energy=1e-9,
        damping=0.2,
        smearing_temperature="auto",
        smearing_metallic=True,
        initial_guess="HCORE",
        write_molden_file=False,
        progress=False,
    )

    text = out.with_suffix(".out").read_text()
    assert result.converged
    assert result.smearing_temperature == pytest.approx(0.005)
    assert result.free_energy == pytest.approx(
        result.energy - result.smearing_temperature * result.entropy,
        abs=1e-12,
    )
    assert "smearing_source      = auto" in text
    assert "smearing_reason      = metallic=True" in text
    assert "smearing_temperature = 0.005" in text
    assert "free_energy (Ha)" in text
    assert "fermi_level (Ha)" in text
    assert "occ=" in text


def test_run_periodic_job_rejects_negative_smearing(tmp_path):
    system, basis = _h2_box()

    with pytest.raises(ValueError, match="smearing_temperature"):
        vq.run_periodic_job(
            system,
            basis,
            output=tmp_path / "bad-smearing",
            smearing_temperature=-0.01,
            write_molden_file=False,
            progress=False,
        )


def test_run_periodic_job_accepts_smearing_units(tmp_path):
    system, basis = _h2_box()
    out = tmp_path / "h2-rhf-gdf-smearing-ev"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        output=out,
        aux_basis="def2-svp-jk",
        max_iter=40,
        conv_tol_energy=1e-9,
        damping=0.2,
        smearing_temperature=0.1,
        smearing_unit="eV",
        initial_guess="HCORE",
        write_molden_file=False,
        progress=False,
    )

    assert result.converged
    assert result.smearing_temperature == pytest.approx(
        vq.electronvolt_to_hartree_temperature(0.1)
    )


# Out-of-process PySCF ground truth for H2/sto-3g/12-bohr (this exact
# _h2_box cell), KRHF(1,1,1) density_fit(def2-svp-jkfit) exxdiv='ewald':
#   -1.1225839666 Ha   (== RHF(Γ) exxdiv='ewald'; verified 2026-06-14)
# The pre-2026-06-15 default Γ RHF/GDF backend (run_rhf_periodic_gamma_gdf,
# molecular limit / exxdiv=None) gave -1.1167447256 — off by the finite-size
# Madelung shift (~5.8 mHa). The default must now match PySCF exxdiv='ewald'.
_PYSCF_H2_GAMMA_EWALD = -1.1225839666
_GAMMA_GDF_MOLECULAR_LIMIT = -1.1167447256


def test_run_periodic_job_gamma_rhf_gdf_default_is_exxdiv_ewald(tmp_path):
    """run_periodic_job(method='RHF', jk_method='gdf') at Γ with NO explicit
    gdf_method routes through the PySCF-µHa-validated run_pbc_gdf_rhf
    (exxdiv='ewald'), not the legacy molecular-limit gamma driver.

    Regression for the 2026-06-15 default-vs-explicit reconciliation: the
    explicit gdf_method path (MDF wiring) already routed to run_pbc_gdf_rhf
    (exxdiv='ewald'), but the default fell to run_rhf_periodic_gamma_gdf
    (molecular limit), so the same job gave two energies ~5.8 mHa apart
    depending only on whether gdf_method was passed.
    """
    system, basis = _h2_box()
    out = tmp_path / "h2-rhf-gdf-default"
    r = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gdf",
        output=out,
        aux_basis="def2-svp-jk",
        max_iter=40,
        conv_tol_energy=1e-9,
        write_molden_file=False,
        progress=False,
    )
    assert r.converged
    # The exxdiv='ewald' rsgdf driver, not the legacy molecular-limit bridge.
    assert r.backend == "pbc-gdf-rsgdf"
    # Matches PySCF KRHF(1,1,1) exxdiv='ewald' ...
    assert r.energy == pytest.approx(_PYSCF_H2_GAMMA_EWALD, abs=2e-4)
    # ... and is decisively NOT the old molecular-limit default (~5.8 mHa).
    assert abs(r.energy - _GAMMA_GDF_MOLECULAR_LIMIT) > 4e-3


def test_run_periodic_job_gamma_rhf_gdf_forwards_rsgdf_tail_cutoff(
    monkeypatch,
    tmp_path,
):
    from vibeqc.pbc_gdf import PBCGDFResult

    system, basis = _h2_box()
    captured = {}

    def fake_driver(_system, _basis, _options, **kwargs):
        captured["kwargs"] = kwargs
        nbf = _basis.nbasis
        zero = np.zeros((nbf, nbf), dtype=float)
        density = np.zeros((nbf, nbf), dtype=float)
        density[0, 0] = 2.0
        return PBCGDFResult(
            energy=-1.0,
            e_electronic=-1.5,
            e_nuclear=0.5,
            e_coulomb=0.1,
            e_hf_exchange=-0.05,
            e_exxdiv=0.0,
            n_iter=1,
            converged=True,
            mo_energies=np.linspace(-0.5, 0.5, nbf),
            mo_coeffs=np.eye(nbf),
            density=density,
            fock=zero,
            overlap=np.eye(nbf),
            hcore=zero,
            aux_basis_name=kwargs["aux_basis"],
            n_aux=1,
            n_fit=1,
            backend="pbc-gdf-rsgdf",
        )

    monkeypatch.setattr("vibeqc.periodic_runner.run_pbc_gdf_rhf", fake_driver)
    out = tmp_path / "h2-rhf-gdf-tail"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gdf",
        output=out,
        aux_basis="def2-svp-jk",
        rsgdf_ke_cutoff=200.0,
        rsgdf_tail_ke_cutoff=3200.0,
        max_iter=2,
        convergence="off",
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        output_qvf=False,
        progress=False,
    )

    assert result.backend == "pbc-gdf-rsgdf"
    assert captured["kwargs"]["gdf_method"] == "rsgdf"
    assert captured["kwargs"]["rsgdf_ke_cutoff"] == pytest.approx(200.0)
    assert captured["kwargs"]["rsgdf_tail_ke_cutoff"] == pytest.approx(3200.0)
    assert captured["kwargs"]["exxdiv"] == "ewald"
    log_text = out.with_suffix(".out").read_text(encoding="utf-8")
    assert "rsgdf_tail_ke_cutoff = 3200.0" in log_text
    assert "OpenMP threads" in log_text


def test_run_periodic_job_rsgdf_tail_cutoff_fails_closed_off_gdf(tmp_path):
    system, basis = _h2_box()

    with pytest.raises(NotImplementedError, match="rsgdf_tail_ke_cutoff"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            output=tmp_path / "h2-tail-bipole",
            rsgdf_tail_ke_cutoff=3200.0,
            write_molden_file=False,
            output_qvf=False,
            progress=False,
        )


def test_run_periodic_job_kpoints_gamma_rks_tail_cutoff_supported(tmp_path):
    """Γ RKS at kpoints=(1,1,1) with rsgdf_tail_ke_cutoff now RUNS.

    Historically this failed closed: the Γ KS fast path fell back to the
    legacy molecular-limit gamma driver, which has no high-|G| tail
    correction. Since the Finding-§4 Γ-KS routing fix, closed-shell Γ KS
    delegates to run_pbc_gdf_rks (the run_pbc_gdf_rhf KS branch), which
    honours the tail cutoff like the RHF path does — so the capability
    exists and the loud refusal would be a regression in reverse.
    """
    system, basis = _h2_box()

    r = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gdf",
        kpoints=(1, 1, 1),
        output=tmp_path / "h2-rks-k111-tail",
        rsgdf_tail_ke_cutoff=3200.0,
        max_iter=30,
        write_molden_file=False,
        output_qvf=False,
        progress=False,
    )
    assert r.converged
    assert r.backend == "native-gamma-gdf-via-k-gdf"


def test_run_periodic_job_gamma_rks_gdf_uses_pure_gdf_ks_backend(tmp_path):
    """Γ RKS/GDF must not fall through to the legacy molecular-limit bridge.

    The regression-suite RKS/LDA Γ parity rows compare against PySCF.pbc GDF.
    The legacy ``run_rhf_periodic_gamma_gdf(..., functional=...)`` path uses
    the Ewald-J molecular-limit bridge and gives Hartree-scale Coulomb-gauge
    offsets on condensed cells. Closed-shell RKS can use the native GDF UKS
    engine in its singlet limit; pin that dispatch here with a small vacuum
    H2 box so the test stays cheap.
    """
    system, basis = _h2_box()
    out = tmp_path / "h2-rks-gdf-default"

    r = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gdf",
        output=out,
        aux_basis="def2-svp-jk",
        max_iter=20,
        conv_tol_energy=1e-8,
        initial_guess="HCORE",
        write_molden_file=False,
        progress=False,
    )

    assert r.converged
    assert r.backend == "pbc-gdf-rsgdf-uks"
    assert r.functional == "lda"
    assert r.s_squared == pytest.approx(0.0, abs=1e-12)


def test_run_periodic_job_gamma_rhf_gdf_smearing_stays_on_legacy(tmp_path):
    """Finite-T smearing at Γ keeps the legacy molecular-limit GDF driver:
    run_pbc_gdf_rhf has no smearing, so the domain gate must fall back rather
    than silently route a smeared job to a driver that ignores smearing.
    """
    system, basis = _h2_box()
    out = tmp_path / "h2-rhf-gdf-smear"
    r = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gdf",
        output=out,
        aux_basis="def2-svp-jk",
        max_iter=40,
        conv_tol_energy=1e-9,
        smearing_temperature=0.01,
        write_molden_file=False,
        progress=False,
    )
    assert r.converged
    assert r.backend == "ewald-jk-fallback"
    assert r.smearing_temperature == pytest.approx(0.01)


def test_legacy_gamma_gdf_dense_core_result_is_parity_held():
    """Prompt 11: the P01 MgO fallback path must stay visibly held.

    The expensive high-level P01 run can route to the legacy
    ``ewald-jk-fallback`` driver when auto convergence knobs are active. That
    result still belongs to the dense-core Gamma GDF parity-hold class, even
    though it did not reach the explicit ``pbc-gdf-rsgdf`` backend that tags
    itself internally.
    """
    from vibeqc.periodic_runner import _mark_legacy_gamma_gdf_parity_hold

    mgo, _basis = _mgo_p01_primitive()
    result = SimpleNamespace(backend="ewald-jk-fallback")
    log_messages: list[str] = []
    plog = SimpleNamespace(info=log_messages.append)

    with pytest.warns(RuntimeWarning, match="legacy fallback absolute-energy"):
        held = _mark_legacy_gamma_gdf_parity_hold(result, mgo, plog)

    assert held
    assert result.backend == "ewald-jk-fallback+PARITY_HELD"
    assert log_messages
    assert "P01 MgO/STO-3G" in log_messages[0]

    h2, _basis = _h2_box()
    h2_result = SimpleNamespace(backend="ewald-jk-fallback")
    assert not _mark_legacy_gamma_gdf_parity_hold(h2_result, h2, plog)
    assert h2_result.backend == "ewald-jk-fallback"


def test_gdf_backend_parity_hold_marker_is_idempotent():
    from vibeqc.pbc_gdf import _gdf_backend_with_parity_hold

    assert (
        _gdf_backend_with_parity_hold("pbc-gdf-rsgdf-uks", True)
        == "pbc-gdf-rsgdf-uks+PARITY_HELD"
    )
    assert (
        _gdf_backend_with_parity_hold("pbc-gdf-rsgdf-uks+PARITY_HELD", True)
        == "pbc-gdf-rsgdf-uks+PARITY_HELD"
    )
    assert (
        _gdf_backend_with_parity_hold("pbc-gdf-rsgdf-uks", False)
        == "pbc-gdf-rsgdf-uks"
    )


def test_gamma_gdf_result_wrapper_propagates_parity_hold_marker():
    """The KRHF/KRKS Γ result adapter must not drop +PARITY_HELD.

    Regression: ``_wrap_gamma_gdf_result`` overwrote the inner driver's
    backend with the bare literal ``native-gamma-gdf-via-k-gdf``, so a
    dense-core hold computed by ``run_pbc_gdf_rhf`` (P01 MgO class,
    ~-0.5 Ha untailed electronic offset) vanished from the public result
    and a pipeline reading ``result.backend`` saw an un-held absolute
    energy.  The wrapper now forwards the marker (and stays clean for
    un-held results).
    """
    from vibeqc.periodic_k_gdf import (
        PeriodicKRKSGDFResult,
        _wrap_gamma_gdf_result,
    )

    n = 2
    zeros = np.zeros((n, n))

    def _fake_inner(backend):
        return SimpleNamespace(
            energy=-1.0,
            e_electronic=-1.0,
            e_nuclear=0.0,
            n_iter=1,
            converged=True,
            mo_energies=np.array([-0.5, 0.5]),
            mo_coeffs=zeros,
            fock=zeros,
            overlap=zeros,
            hcore=zeros,
            density=zeros,
            scf_trace=[],
            functional="pbe",
            e_xc=-0.1,
            e_coulomb=0.0,
            e_hf_exchange=0.0,
            fock_mixing=0.0,
            level_shift=0.0,
            level_shift_warmup_cycles=0,
            smearing_temperature=0.0,
            fermi_level=0.0,
            entropy=0.0,
            occupations=np.array([2.0, 0.0]),
            aux_basis_name="",
            n_aux=0,
            backend=backend,
            gradient=None,
        )

    info = SimpleNamespace(
        kpoints_cart=np.zeros((1, 3)),
        weights=np.array([1.0]),
    )

    held_inner = _fake_inner("pbc-gdf-rsgdf-rks+PARITY_HELD")
    held_out = _wrap_gamma_gdf_result(
        held_inner, info, functional="pbe", result_cls=PeriodicKRKSGDFResult
    )
    assert held_out.backend == "native-gamma-gdf-via-k-gdf+PARITY_HELD"

    clean_inner = _fake_inner("pbc-gdf-rsgdf-rks")
    clean_out = _wrap_gamma_gdf_result(
        clean_inner, info, functional="pbe", result_cls=PeriodicKRKSGDFResult
    )
    assert clean_out.backend == "native-gamma-gdf-via-k-gdf"


# ============================================================
# Multi-k GDF integration tests (kpoints routing)
# ============================================================


@pytest.mark.parametrize("method", ["RHF"])
def test_kpoints_gamma_equals_single_k_identity(method, tmp_path):
    """Gamma-only [1,1,1] k-mesh must give the same energy as default."""
    system, basis = _h2_box()
    out1 = tmp_path / "h2-gamma"
    out2 = tmp_path / "h2-k111"

    r1 = vq.run_periodic_job(
        system,
        basis,
        method=method,
        output=out1,
        aux_basis="def2-svp-jk",
        max_iter=30,
        conv_tol_energy=1e-9,
        initial_guess="HCORE",
        write_molden_file=False,
        progress=False,
    )
    r2 = vq.run_periodic_job(
        system,
        basis,
        method=method,
        output=out2,
        kpoints=(1, 1, 1),
        aux_basis="def2-svp-jk",
        max_iter=30,
        conv_tol_energy=1e-9,
        initial_guess="HCORE",
        write_molden_file=False,
        progress=False,
    )

    assert r1.converged and r2.converged
    # Default Γ RHF/GDF now routes through run_pbc_gdf_rhf (exxdiv='ewald');
    # the explicit (1,1,1) k-mesh delegates to the same driver via run_krhf,
    # so the two agree on the exxdiv='ewald' energy (no molecular-limit gap).
    assert r1.backend == "pbc-gdf-rsgdf"
    assert r2.backend == "native-gamma-gdf-via-k-gdf"
    assert r1.energy == pytest.approx(r2.energy, abs=1e-10)


@pytest.mark.parametrize("method", ["RHF", "RKS"])
def test_gdf_multi_k_converges(method, tmp_path):
    """Multi-k GDF via kpoints converges on H2/12-bohr."""
    system, basis = _h2_box()
    out = tmp_path / f"h2-mk-{method.lower()}"

    kw = {
        "method": method,
        "output": out,
        "kpoints": (2, 2, 2),
        "aux_basis": "def2-svp-jk",
        "max_iter": 30,
        "conv_tol_energy": 1e-8,
        "initial_guess": "HCORE",
        "write_molden_file": False,
        "progress": False,
    }
    if method == "RKS":
        kw["functional"] = "pbe"

    result = vq.run_periodic_job(system, basis, **kw)

    assert result.converged
    assert result.n_iter >= 2
    assert -2.0 < result.energy < 0.0
    assert out.with_suffix(".out").exists()
    assert out.with_suffix(".system").exists()


def test_lih_p02_222_gdf_postscf_outputs_handle_complex_density(tmp_path):
    """Reduced P02 LiH 2x2x2 GDF-shaped output path: post-SCF writers
    must not emit complex-to-real casts or imaginary-density warnings."""
    import warnings
    from types import SimpleNamespace

    from vibeqc.output.formats.population import compute_population_summary
    from vibeqc.output.formats.qvf import validate_qvf, write_qvf
    from vibeqc.output.plan import OutputPlan
    from vibeqc.periodic_density import evaluate_periodic_density_on_grid
    from vibeqc.periodic_runner import (
        _density_lattice_set_for_output,
        _gamma_proxy_for_multi_k,
    )

    # This mirrors the reduced P02 LiH primitive-cell shape used by the
    # full GDF route without spending minutes in the live SCF/integral build.
    system, basis = _lih_primitive_rocksalt()
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])
    nbf = basis.nbasis
    real_density = np.eye(nbf, dtype=float) * (system.n_electrons() / nbf)
    anti = np.triu(np.ones((nbf, nbf)), 1)
    anti = anti - anti.T
    density_k = [
        real_density.astype(np.complex128) + 2.66e-8j * anti
        for _ in kmesh.kpoints
    ]
    fake_result = SimpleNamespace(
        converged=True,
        energy=-8.0,
        n_iter=6,
        density=density_k,
        mo_coeffs=[np.eye(nbf, dtype=np.complex128) for _ in kmesh.kpoints],
        mo_energies=[np.linspace(-0.5, 0.5, nbf) for _ in kmesh.kpoints],
        occupations=[
            np.array([2.0, 2.0] + [0.0] * max(0, nbf - 2), dtype=float)
            for _ in kmesh.kpoints
        ],
        kpoints_cart=np.asarray(list(kmesh.kpoints), dtype=float),
        kpoint_weights=np.asarray(list(kmesh.weights), dtype=float),
        overlap=None,
    )

    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0
    lat_opts.nuclear_cutoff_bohr = 8.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        D_set = _density_lattice_set_for_output(
            basis,
            system,
            fake_result,
            lat_opts,
        )
        rho, shape = evaluate_periodic_density_on_grid(
            basis,
            system,
            D_set,
            grid_shape=(2, 2, 2),
            ao_image_radius=0,
        )
        pop_result = _gamma_proxy_for_multi_k(fake_result)
        summary = compute_population_summary(
            pop_result,
            basis,
            system.unit_cell_molecule(),
        )
        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "lih-p02-222-gdf",
            method="RKS",
            basis="sto-3g",
            functional="lda",
            output_qvf=True,
            write_population=True,
            job_kind="periodic_scf",
        )
        per_voxel = np.asarray(system.lattice, dtype=float).T / np.array(
            shape,
            dtype=float,
        )
        qvf_path = write_qvf(
            tmp_path / "lih-p02-222-gdf",
            plan,
            system=system,
            result=fake_result,
            method="RKS",
            functional="lda",
            basis="sto-3g",
            volume_data={
                "Electron density": (
                    rho.astype(np.complex128) + 1.0e-10j,
                    np.zeros(3),
                    per_voxel,
                )
            },
            population_summary=summary,
        )

    assert not any(
        "ComplexWarning" in warning.category.__name__
        or "non-negligible imaginary" in str(warning.message)
        for warning in caught
    )
    # NPA is unavailable until occupancy-weighted NAOs exist;
    # every attempted section must succeed cleanly.
    assert summary.errors == {}
    assert "Natural Atomic Orbital" in summary.unavailable["npa"]
    assert summary.npa_atoms == []
    report = validate_qvf(qvf_path)
    assert report["valid"], report["errors"]


def test_multik_gamma_proxy_uses_explicit_gamma_point():
    """Gamma-only artifacts must choose the actual Gamma k point, not
    whichever entry happens to be first in a multi-k result list."""
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _gamma_proxy_for_multi_k

    c_non_gamma = np.eye(2) * 2.0
    c_gamma = np.eye(2) * 3.0
    result = SimpleNamespace(
        mo_coeffs=[c_non_gamma, c_gamma],
        mo_energies=[np.array([2.0, 3.0]), np.array([-1.0, 1.0])],
        occupations=[np.array([0.0, 0.0]), np.array([2.0, 0.0])],
        density=[np.eye(2) * 4.0, np.eye(2) * 5.0],
        kpoints_cart=np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        overlap=None,
    )

    proxy = _gamma_proxy_for_multi_k(result)

    np.testing.assert_allclose(proxy.mo_coeffs, c_gamma)
    np.testing.assert_allclose(proxy.mo_energies, [-1.0, 1.0])
    np.testing.assert_allclose(proxy.occupations, [2.0, 0.0])
    np.testing.assert_allclose(proxy.density, np.eye(2) * 5.0)


def test_multik_gamma_proxy_requires_gamma_metadata():
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _gamma_proxy_for_multi_k

    result = SimpleNamespace(
        mo_coeffs=[np.eye(2), np.eye(2)],
        mo_energies=[np.zeros(2), np.ones(2)],
        occupations=[np.ones(2), np.ones(2)],
        density=[np.eye(2), np.eye(2)],
        overlap=None,
    )

    with pytest.raises(ValueError, match="requires k-point metadata"):
        _gamma_proxy_for_multi_k(result)


def test_qvf_wavefunction_proxy_uses_first_k_when_gamma_absent():
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _qvf_wavefunction_proxy_for_multi_k

    class _System:
        def reciprocal_lattice(self):
            return np.eye(3)

    c_first = np.eye(2, dtype=np.complex128) * (1.0 + 0.25j)
    c_second = np.eye(2, dtype=np.complex128) * 2.0
    result = SimpleNamespace(
        mo_coeffs=[c_first, c_second],
        mo_energies=[np.array([-1.0, 1.0]), np.array([2.0, 3.0])],
        occupations=[np.array([2.0, 0.0]), np.array([0.0, 0.0])],
        density=[np.eye(2), np.eye(2) * 2.0],
        kpoints_cart=np.array([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        overlap=None,
    )

    proxy, k_frac = _qvf_wavefunction_proxy_for_multi_k(result, _System())

    np.testing.assert_allclose(proxy.mo_coeffs, c_first)
    np.testing.assert_allclose(proxy.mo_energies, [-1.0, 1.0])
    assert k_frac == pytest.approx([0.25, 0.0, 0.0])


def test_qvf_wavefunction_proxy_treats_empty_occupations_as_missing():
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _qvf_wavefunction_proxy_for_multi_k

    class _System:
        def reciprocal_lattice(self):
            return np.eye(3)

    result = SimpleNamespace(
        mo_coeffs=[np.eye(2, dtype=np.complex128)],
        mo_energies=[np.array([-1.0, 1.0])],
        occupations=[],
        density=[np.eye(2)],
        kpoints_cart=np.array([[0.0, 0.0, 0.0]]),
        overlap=None,
    )

    proxy, k_frac = _qvf_wavefunction_proxy_for_multi_k(result, _System())

    np.testing.assert_allclose(proxy.mo_coeffs, np.eye(2))
    assert proxy.occupations is None
    assert k_frac == pytest.approx([0.0, 0.0, 0.0])


def test_complex_per_k_density_requires_fold_metadata():
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _density_lattice_set_for_output

    system, basis = _lih_primitive_rocksalt()
    nbf = basis.nbasis
    anti = np.triu(np.ones((nbf, nbf)), 1)
    anti = anti - anti.T
    density_k = [
        np.eye(nbf, dtype=np.complex128) + 1.0e-8j * anti,
        np.eye(nbf, dtype=np.complex128) - 1.0e-8j * anti,
    ]
    result = SimpleNamespace(density=density_k)
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0

    with pytest.raises(ValueError, match="requires k-point and weight metadata"):
        _density_lattice_set_for_output(basis, system, result, lat_opts)


def test_reduced_time_reversal_pair_density_fold_is_real():
    """Output density folding must unfold an implicit -k partner.

    Symmetry-reduced meshes may store only one representative k point whose
    weight includes its time-reversal mate. The real-space density fold must
    use both k and -k; otherwise a complex Hermitian D(k) leaves an imaginary
    density block and QVF/DOS artifacts fail.
    """
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _density_lattice_set_for_output

    system, basis = _lih_primitive_rocksalt()
    nbf = basis.nbasis
    anti = np.triu(np.ones((nbf, nbf)), 1)
    anti = anti - anti.T
    D_k = np.eye(nbf, dtype=np.complex128) + 2.0e-3j * anti
    k_cart = (np.asarray(system.reciprocal_lattice(), dtype=float)
              @ np.array([0.25, 0.0, 0.0]))
    result = SimpleNamespace(
        density=[D_k],
        kpoints_cart=np.asarray([k_cart], dtype=float),
        kpoint_weights=np.asarray([1.0], dtype=float),
    )
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0

    D_set = _density_lattice_set_for_output(basis, system, result, lat_opts)

    assert len(D_set.blocks) > 1
    assert all(not np.iscomplexobj(np.asarray(block)) for block in D_set.blocks)
    assert all(np.isfinite(np.asarray(block)).all() for block in D_set.blocks)


def test_spin_density_proxy_preserves_k_metadata_for_periodic_output():
    """Unrestricted density_alpha/beta wrappers must keep k metadata."""
    from types import SimpleNamespace

    from vibeqc.periodic_runner import (
        _density_lattice_set_for_output,
        _density_proxy_with_k_metadata,
    )

    system, basis = _lih_primitive_rocksalt()
    nbf = basis.nbasis
    anti = np.triu(np.ones((nbf, nbf)), 1)
    anti = anti - anti.T
    D_alpha = [
        0.5 * np.eye(nbf, dtype=np.complex128) + 1.0e-3j * anti,
    ]
    k_cart = (np.asarray(system.reciprocal_lattice(), dtype=float)
              @ np.array([0.25, 0.0, 0.0]))
    result = SimpleNamespace(
        density_alpha=D_alpha,
        density_beta=D_alpha,
        kpoints_cart=np.asarray([k_cart], dtype=float),
        kpoint_weights=np.asarray([1.0], dtype=float),
    )
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 8.0

    proxy = _density_proxy_with_k_metadata(result, result.density_alpha)
    D_set = _density_lattice_set_for_output(basis, system, proxy, lat_opts)

    assert all(not np.iscomplexobj(np.asarray(block)) for block in D_set.blocks)


@pytest.mark.parametrize("method,km", [("RHF", (2, 2, 2)), ("RHF", (1, 1, 2))])
def test_gdf_multi_k_energy_close_to_gamma(method, km):
    """Multi-k GDF energy should be close to gamma for molecular-limit cells."""
    system, basis = _h2_box()

    r_gamma = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        _opts(),
        aux_basis="def2-svp-jk",
        progress=False,
    )
    r_mk = vq.run_krhf_periodic_gdf(
        system,
        basis,
        km,
        _opts(),
        aux_basis="def2-svp-jk",
        progress=False,
    )

    assert r_gamma.converged and r_mk.converged
    # Multi-k uses Ewald gauge; gamma may use direct truncation when
    # n_int_cells=1 (molecular limit).  Difference is ~15 mHa for H2/12-bohr.
    assert abs(r_mk.energy - r_gamma.energy) < 0.02


@pytest.mark.parametrize("dim", [1, 2])
def test_rsgdf_fails_closed_below_3d(dim):
    """RSGDF must REFUSE ``dim < 3``, not converge on it.

    This test previously asserted the opposite: that the 1-D/2-D RSGDF SCF
    converges to a finite negative energy. It does, and the energy is meaningless.
    ``rsgdf_g_mesh`` spans only the periodic axes and pins every non-periodic axis
    at the single point ``G_perp = 0``, so each AO-pair density is replaced by its
    transverse average and the kernel ``4*pi/|G+q|^2/V`` applied to it is a
    transverse-uniform sheet term proportional to ``1/V`` rather than ``1/r``: the
    electron repulsion *vanishes* as the vacuum padding grows. Independently, the
    bare ``V_ne``/``E_nn`` lattice sums combined with a neutral (``G+q=0``-dropped)
    ``J`` leave an uncancelled conditional constant, so the total also diverges with
    the nuclear lattice-sum cutoff.

    Fixed 2026-07-10 (``aux_basis._reject_transverse_collapse``); see
    ``docs/aiccm2026dev_a_lowd_greens.md`` section 0.1. The gauge-correct low-D
    Hamiltonian is the mixed-boundary wire kernel
    (``vibeqc.periodic.ccm.lowd_scf.run_ccm_rhf_wire``, ``dim == 1``).

    The ``bare`` algorithm does not build that mesh and still runs at ``dim < 3``.
    """
    system, basis = _h2_dimensional_box(dim)
    opts = _opts()

    with pytest.raises(NotImplementedError, match="transverse-collapsed"):
        vq.run_rhf_periodic_gamma_gdf(
            system,
            basis,
            opts,
            aux_basis="def2-svp-jk",
            gdf_algorithm="rsgdf",
            progress=False,
        )

    # the bare path is untouched by the fix and must keep working
    r_bare = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        gdf_algorithm="bare",
        progress=False,
    )
    assert r_bare.converged
    assert np.isfinite(r_bare.energy) and r_bare.energy < 0.0


def test_rsgdf_3d_scf_converges_with_known_error():
    """RSGDF on 3D H2 converges but with a known systematic error (~231 mHa
    over-binding vs bare). This is an architectural limitation (sparse G-mesh
    cannot resolve compact aux primitives); the warning emitted by the driver
    documents the status. The test pins the current error magnitude as a
    regression guard — a future dense-FFT-mesh RSGDF fix should decrease this."""
    system, basis = _h2_dimensional_box(3)
    opts = _opts()
    r_bare = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        gdf_algorithm="bare",
        progress=False,
    )
    r_rsgdf = vq.run_rhf_periodic_gamma_gdf(
        system,
        basis,
        opts,
        aux_basis="def2-svp-jk",
        gdf_algorithm="rsgdf",
        progress=False,
    )
    assert r_bare.converged and r_rsgdf.converged
    # RSGDF 3D over-binds by ~231 mHa due to the sparse G-mesh.
    delta_ha = abs(r_bare.energy - r_rsgdf.energy)
    assert delta_ha == pytest.approx(0.2306236417, abs=1e-3)
    assert r_rsgdf.energy < 0.0
