"""Phase C1b tests: Fermi-Dirac smearing in multi-k Ewald drivers.

Contracts exercised:

1. **Field exposure** — ``smearing_temperature`` is on
   ``PeriodicRHFOptions`` / ``PeriodicSCFOptions`` /
   ``PeriodicKSOptions`` (default 0.0).

2. **T = 0 inertness** — ``smearing_temperature = 0`` reproduces the
   pre-C1b SCF dynamics bit-for-bit. Energy, free energy, and entropy
   collapse: ``free_energy == energy``, ``entropy == 0``,
   ``occupations`` are integer-Aufbau {0, 2}.

3. **Electron-count conservation** — under any temperature, the
   total k-weighted occupation across the Brillouin zone equals
   ``system.n_electrons``.

4. **Wide-gap inertness** — on a wide-HOMO-LUMO-gap insulator, low-T
   smearing barely moves the energy because occupations are pinned
   at the Aufbau values. Specifically: at T = 0.01 Ha (~3 mHa
   thermal energy) on H2 / sto-3g (gap ~ 0.6 Ha), the energy and
   free energy match the T=0 result to ~µHa.

5. **Entropy non-negative** — the electronic entropy ``S/k_B`` is
   always ≥ 0 by construction.

6. **Fractional density builder** — the C++
   ``real_space_density_from_kpoints_fractional`` reduces to the
   integer-occupation builder when ``occ`` is bit-equal to ``{0, 2}``.

7. **SCF convergence with smearing** — a multi-k SCF run on a small
   mesh with T = 0.005 Ha converges to a finite, sensible energy
   (the SCF dynamics under smearing are stable on closed-shell
   insulators).

8. **DFT + dispatcher coverage** — periodic RKS and BIPOLE RKS/UKS use
   the same occupation machinery and the public dispatchers never silently
   drop ``smearing_temperature`` or route it to an integer-Aufbau backend.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h2_chain(a: float = 10.0):
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _smearing_options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.damping = 0.3
    o.max_iter = 60
    o.use_diis = True
    o.conv_tol_grad = 1e-5
    return o


def _ks_smearing_options():
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.damping = 0.3
    o.max_iter = 40
    o.use_diis = True
    o.conv_tol_grad = 1e-5
    # Keep the test grid small; this file tests occupation plumbing,
    # not production DFT quadrature convergence.
    o.grid.n_radial = 20
    o.grid.n_theta = 9
    o.grid.n_phi = 18
    return o


# ---------------------------------------------------------------------------
# 1. Field exposure on all three option structs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    vq.PeriodicRHFOptions,
    vq.PeriodicSCFOptions,
    vq.PeriodicKSOptions,
])
def test_smearing_field_default_is_zero(cls):
    o = cls()
    assert hasattr(o, "smearing_temperature")
    assert o.smearing_temperature == 0.0
    o.smearing_temperature = 0.005
    assert o.smearing_temperature == 0.005


def test_temperature_kelvin_helper():
    assert vq.kelvin_to_hartree_temperature(0.0) == 0.0
    assert vq.kelvin_to_hartree_temperature(300.0) == pytest.approx(
        300.0 * vq.KB_HARTREE_PER_K
    )
    assert vq.hartree_to_kelvin_temperature(
        vq.kelvin_to_hartree_temperature(300.0)
    ) == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 2. T=0 inertness — Aufbau path, same energy + zero entropy
# ---------------------------------------------------------------------------

def test_smearing_zero_temperature_is_aufbau():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _smearing_options()
    opts.smearing_temperature = 0.0

    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert r.smearing_temperature == 0.0
    assert r.entropy == 0.0
    assert r.free_energy == pytest.approx(r.energy, abs=1e-14)
    # Hard Aufbau on H2: 1 occupied at occ=2, 1 virtual at occ=0.
    occ_home = r.occupations[0]
    assert np.allclose(occ_home, [2.0, 0.0])


def test_default_options_match_explicit_zero():
    """An options object that doesn't touch smearing_temperature must
    produce identical SCF dynamics to one explicitly set to 0."""
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    o_implicit = _smearing_options()
    o_explicit = _smearing_options()
    o_explicit.smearing_temperature = 0.0

    r_imp = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, o_implicit, omega=0.5, spacing_bohr=0.3,
    )
    r_exp = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, o_explicit, omega=0.5, spacing_bohr=0.3,
    )
    assert r_imp.n_iter == r_exp.n_iter
    assert r_imp.energy == pytest.approx(r_exp.energy, abs=1e-14)


# ---------------------------------------------------------------------------
# 3. Electron-count conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("temperature", [0.001, 0.01, 0.05])
def test_electron_count_preserved_under_smearing(temperature):
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _smearing_options()
    opts.smearing_temperature = temperature

    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    n_total = sum(
        float(w) * float(occ.sum())
        for w, occ in zip(km.weights, r.occupations)
    )
    assert n_total == pytest.approx(float(sysp.n_electrons()), abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Wide-gap inertness — energy doesn't move
# ---------------------------------------------------------------------------

def test_wide_gap_smearing_barely_changes_energy():
    """On wide-gap H2 (~ 0.6 Ha gap) at T = 0.005 Ha (way below the
    gap), occupations stay essentially Aufbau and the energy / free
    energy match the T=0 calculation to ~µHa."""
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _smearing_options()

    opts.smearing_temperature = 0.0
    r0 = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.smearing_temperature = 0.005
    rT = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r0.converged and rT.converged
    # Energy and free energy nearly unchanged because T << gap.
    assert rT.energy == pytest.approx(r0.energy, abs=5e-6)
    assert rT.free_energy == pytest.approx(r0.energy, abs=5e-6)
    # Entropy is essentially zero (occupations near {0, 2}).
    assert rT.entropy < 1e-6


# ---------------------------------------------------------------------------
# 5. Entropy non-negative
# ---------------------------------------------------------------------------

def test_entropy_is_non_negative():
    sysp, basis = _h2_chain(a=10.0)
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _smearing_options()
    opts.smearing_temperature = 0.01

    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert np.isfinite(r.energy)
    assert r.entropy >= 0.0


# ---------------------------------------------------------------------------
# 6. Fractional density builder reduces to integer builder at {0, 2}
# ---------------------------------------------------------------------------

def test_fractional_density_builder_matches_integer_at_aufbau():
    """``real_space_density_from_kpoints_fractional`` with occ exactly
    ``[2, 0, 0, ...]`` produces the same density blocks as the integer
    builder with n_occ = 1 (the closed-shell 2-electron case)."""
    sysp, basis = _h2_chain(a=10.0)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    opts.nuclear_cutoff_bohr = 15.0
    S_lat = vq.compute_overlap_lattice(basis, sysp, opts)
    km = vq.monkhorst_pack(sysp, [2, 2, 2])

    # Random Hermitian per-k C from Hcore-like proxy.
    rng = np.random.default_rng(42)
    nbf = basis.nbasis
    n_k = len(km.kpoints)
    C_per_k = []
    for _ in range(n_k):
        H = rng.standard_normal((nbf, nbf)) + \
            1j * rng.standard_normal((nbf, nbf))
        H = 0.5 * (H + H.conj().T)
        _, C = np.linalg.eigh(H)
        C_per_k.append(C)

    # Integer build: n_occ = 1 → first column gets occ 2, rest 0.
    D_int = vq.real_space_density_from_kpoints(
        C_per_k, [1] * n_k, km, S_lat.cells,
    )
    # Fractional build with the same occupation pattern.
    occ_per_k = [np.array([2.0] + [0.0] * (nbf - 1)) for _ in range(n_k)]
    D_frac = vq._vibeqc_core.real_space_density_from_kpoints_fractional(
        C_per_k, occ_per_k, km, S_lat.cells,
    )
    for slot in range(len(S_lat.cells)):
        np.testing.assert_allclose(
            np.asarray(D_int.blocks[slot]),
            np.asarray(D_frac.blocks[slot]),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 7. SCF convergence with smearing
# ---------------------------------------------------------------------------

def test_smearing_scf_converges_on_small_multik_mesh():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _smearing_options()
    opts.smearing_temperature = 0.005

    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert np.isfinite(r.energy)
    assert np.isfinite(r.free_energy)
    assert r.energy < 0.0    # bound electrons
    # All occupations bounded in [0, 2].
    for occ in r.occupations:
        assert occ.min() >= -1e-12
        assert occ.max() <= 2.0 + 1e-12
    D_expected = vq._vibeqc_core.real_space_density_from_kpoints_fractional(
        r.mo_coeffs,
        r.occupations,
        km,
        r.density.cells,
    )
    for got, expected in zip(r.density.blocks, D_expected.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 8. Periodic RKS + dispatcher coverage for metallic occupations
# ---------------------------------------------------------------------------

def test_rks_smearing_preserves_electron_count():
    sysp, basis = _h2_chain(a=10.0)
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _ks_smearing_options()
    opts.smearing_temperature = 0.005

    r = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert r.smearing_temperature == pytest.approx(0.005)
    assert np.isfinite(r.energy)
    assert np.isfinite(r.free_energy)
    n_total = sum(
        float(w) * float(occ.sum())
        for w, occ in zip(km.weights, r.occupations)
    )
    assert n_total == pytest.approx(float(sysp.n_electrons()), abs=1e-9)
    for occ in r.occupations:
        assert occ.min() >= -1e-12
        assert occ.max() <= 2.0 + 1e-12
    D_expected = vq._vibeqc_core.real_space_density_from_kpoints_fractional(
        r.mo_coeffs,
        r.occupations,
        km,
        r.density.cells,
    )
    for got, expected in zip(r.density.blocks, D_expected.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-9)


def test_rks_dispatcher_gamma_smearing_routes_to_multik_driver():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _ks_smearing_options()
    opts.smearing_temperature = 0.005

    r = vq.run_rks_periodic_scf(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSMultiKEwaldResult)
    assert r.smearing_temperature == pytest.approx(0.005)


def test_rhf_dispatcher_preserves_smearing_temperature():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _smearing_options()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.smearing_temperature = 0.005

    r = vq.run_rhf_periodic_scf(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert isinstance(r, vq.PeriodicRHFMultiKEwaldResult)
    assert r.smearing_temperature == pytest.approx(0.005)


def test_run_periodic_job_accepts_smearing_options(tmp_path):
    sysp, basis = _h2()
    out = tmp_path / "h2-smearing-options"

    r = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        output=out,
        smearing=vq.SmearingOptions.from_user("metal"),
        max_iter=40,
        conv_tol_energy=1e-9,
        initial_guess="HCORE",
        write_molden_file=False,
        progress=False,
    )

    assert r.converged
    assert r.smearing_temperature == pytest.approx(0.005)
    text = out.with_suffix(".out").read_text()
    assert "smearing_method      = fermi-dirac" in text
    assert "smearing_source      = preset:metal" in text


def test_run_periodic_job_accepts_bipole_rks_smearing(tmp_path):
    sysp, basis = _h2(box=8.0)

    r = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="lda",
        jk_method="bipole",
        output=tmp_path / "rks-bipole-smearing",
        smearing_temperature=0.005,
        max_iter=2,
        initial_guess="HCORE",
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert r.smearing_temperature == pytest.approx(0.005)
    assert np.isfinite(r.energy)
    assert np.isfinite(r.free_energy)
    assert len(r.occupations) == 1


def test_run_periodic_job_bipole_optimizer_rejects_smearing_before_dispatch(
    monkeypatch, tmp_path
):
    """A smeared BIPOLE objective is not mislabeled as optimized energy."""
    import vibeqc.bipole_optimize as bipole_optimize

    sysp, basis = _h2(box=8.0)

    def fake_relax_atoms(system, basis_name, kmesh, method, **kwargs):
        pytest.fail("optimizer dispatched before finite-T preflight")

    monkeypatch.setattr(bipole_optimize, "relax_atoms", fake_relax_atoms)

    stem = tmp_path / "rks-bipole-smearing-opt"
    with pytest.raises(
        NotImplementedError,
        match="finite-temperature BIPOLE optimization",
    ):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            output=stem,
            smearing_temperature=0.005,
            max_iter=50,
            conv_tol_energy=1e-9,
            initial_guess="HCORE",
            optimize=True,
            optimize_max_iter=1,
            dry_run=True,
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "kpoints_factory",
    [
        lambda sysp: vq.monkhorst_pack(sysp, [2, 1, 1]),
        lambda sysp: vq.KPoints.monkhorst_pack(sysp, (2, 1, 1)),
    ],
    ids=["native-bloch-kmesh", "kpoints-object"],
)
def test_run_periodic_job_bipole_optimizer_accepts_materialized_kmesh(
    monkeypatch, tmp_path, kpoints_factory
):
    """Pre-built k-mesh objects must survive SCF and optimizer handoff."""
    import vibeqc.bipole_optimize as bipole_optimize
    import vibeqc.pbc_bipole as pbc_bipole

    sysp, basis = _h2(box=8.0)
    user_kpoints = kpoints_factory(sysp)
    expected = vq.as_bloch_kmesh(user_kpoints)
    captured = {}

    def fake_run_pbc_bipole_rhf(system, basis, kmesh, opts, **kwargs):
        captured["scf_kmesh"] = kmesh
        return SimpleNamespace(
            energy=-1.0,
            e_electronic=-1.0,
            e_nuclear=0.0,
            converged=True,
            n_iter=1,
            scf_trace=[
                SimpleNamespace(
                    iter=1,
                    energy=-1.0,
                    delta_e=0.0,
                    grad_norm=0.0,
                    diis_subspace=0,
                )
            ],
        )

    def fake_relax_atoms(system, basis_name, kmesh, method, **kwargs):
        captured["opt_kmesh"] = kmesh
        captured["opt_kwargs"] = kwargs
        return SimpleNamespace(
            system=system,
            energy=-1.0,
            gradient=np.zeros((len(system.unit_cell), 3)),
            n_iter=0,
            converged=True,
        )

    monkeypatch.setattr(pbc_bipole, "run_pbc_bipole_rhf", fake_run_pbc_bipole_rhf)
    monkeypatch.setattr(bipole_optimize, "relax_atoms", fake_relax_atoms)

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        jk_method="bipole",
        kpoints=user_kpoints,
        output=tmp_path / "rhf-bipole-object-kmesh-opt",
        optimize=True,
        optimize_max_iter=1,
        dispersion="none",
        bipole_cutoff_bohr=9.0,
        bipole_nuclear_cutoff_bohr=10.0,
        bipole_exact_zone_bohr=1.25,
        sr_image_precision=2e-7,
        sr_range_screening=True,
        ewald_omega=0.4,
        ewald_precision=2e-9,
        use_mom=True,
        use_exchange_ewald_split=True,
        exchange_exxdiv="none",
        bz_integration="smearing",
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert result.converged
    for key in ("scf_kmesh", "opt_kmesh"):
        np.testing.assert_allclose(
            np.asarray(captured[key].kpoints, dtype=float),
            np.asarray(expected.kpoints, dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(captured[key].weights, dtype=float),
            np.asarray(expected.weights, dtype=float),
        )
    if isinstance(user_kpoints, vq.BlochKMesh):
        assert captured["scf_kmesh"] is user_kpoints
        assert captured["opt_kmesh"] is user_kpoints
    opt_kwargs = captured["opt_kwargs"]
    assert opt_kwargs["nuclear_cutoff_bohr"] == pytest.approx(10.0)
    assert opt_kwargs["exact_zone_bohr"] == pytest.approx(1.25)
    assert opt_kwargs["sr_image_precision"] == pytest.approx(2e-7)
    assert opt_kwargs["ewald_omega"] == pytest.approx(0.4)
    assert opt_kwargs["ewald_precision"] == pytest.approx(2e-9)
    assert opt_kwargs["use_oda"] is False
    assert opt_kwargs["use_mom"] is True
    assert opt_kwargs["use_exchange_ewald_split"] is True
    assert opt_kwargs["exchange_exxdiv"] == "none"
    assert opt_kwargs["bz_integration"] == "smearing"
    assert opt_kwargs["scf_options"].lattice_opts.sr_range_screening is True


def test_run_periodic_job_accepts_bipole_uks_smearing(tmp_path):
    lattice = np.eye(3) * 8.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [4.0, 4.0, 4.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    r = vq.run_periodic_job(
        sysp,
        basis,
        method="UKS",
        functional="lda",
        jk_method="bipole",
        output=tmp_path / "uks-bipole-smearing",
        smearing_temperature=0.005,
        max_iter=10,
        conv_tol_energy=1e-6,
        initial_guess="HCORE",
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert r.smearing_temperature == pytest.approx(0.005)
    assert np.isfinite(r.energy)
    assert np.isfinite(r.free_energy)
    assert len(r.occupations_alpha) == 1
    assert len(r.occupations_beta) == 1


def test_run_periodic_job_accepts_bipole_uhf_smearing(tmp_path):
    lattice = np.eye(3) * 8.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [4.0, 4.0, 4.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    r = vq.run_periodic_job(
        sysp,
        basis,
        method="UHF",
        jk_method="bipole",
        output=tmp_path / "uhf-bipole-smearing",
        smearing_temperature=0.005,
        max_iter=10,
        conv_tol_energy=1e-6,
        initial_guess="HCORE",
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert r.smearing_temperature == pytest.approx(0.005)
    assert np.isfinite(r.energy)
    assert np.isfinite(r.free_energy)
    assert r.occupations_alpha[0].sum() == pytest.approx(1.0, abs=1e-12)
    assert r.occupations_beta[0].sum() == pytest.approx(0.0, abs=1e-12)


def test_run_periodic_job_rejects_unsupported_bipole_rhf_smearing(tmp_path):
    sysp, basis = _h2(box=12.0)

    with pytest.raises(NotImplementedError, match="BIPOLE RHF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="bipole",
            output=tmp_path / "bad-bipole-rhf-smearing",
            smearing_temperature=0.005,
            write_molden_file=False,
            progress=False,
        )


def test_direct_bipole_rhf_driver_rejects_smearing():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])

    rhf_opts = vq.PeriodicRHFOptions()
    rhf_opts.smearing_temperature = 0.005
    with pytest.raises(NotImplementedError, match="BIPOLE RHF"):
        vq.run_pbc_bipole_rhf(
            sysp,
            basis,
            km,
            rhf_opts,
            progress=False,
        )


@pytest.mark.parametrize(
    ("driver", "options_factory", "bz_integration"),
    [
        (vq.run_pbc_bipole_rks, _ks_smearing_options, None),
        (vq.run_pbc_bipole_uks, _ks_smearing_options, None),
        (vq.run_pbc_bipole_rks, _ks_smearing_options, "gilat"),
        (vq.run_pbc_bipole_uhf, _smearing_options, "gilat"),
        (vq.run_pbc_bipole_uks, _ks_smearing_options, "gilat"),
    ],
)
def test_fractional_bipole_density_rejects_legacy_ewald_gauge(
    driver, options_factory, bz_integration
):
    """Legacy J-LR orbital reconstruction assumes integer occupations."""
    sysp, basis = _h2()
    mesh = [2, 1, 1] if bz_integration == "gilat" else [1, 1, 1]
    km = vq.monkhorst_pack(sysp, mesh, use_symmetry=False)
    opts = options_factory()
    if bz_integration is None:
        opts.smearing_temperature = 0.005

    with pytest.raises(NotImplementedError, match="fractional occupations"):
        driver(
            sysp,
            basis,
            km,
            opts,
            bz_integration=bz_integration,
            use_ewald_j_split=True,
            use_exchange_ewald_split=False,
            progress=False,
        )


@pytest.mark.parametrize("driver, options_factory", [
    (vq.run_rhf_periodic_scf, _smearing_options),
    (vq.run_rks_periodic_scf, _ks_smearing_options),
])
def test_direct_truncated_rejects_smearing(driver, options_factory):
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = options_factory()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.DIRECT_TRUNCATED
    opts.smearing_temperature = 0.005

    with pytest.raises(NotImplementedError, match="DIRECT_TRUNCATED"):
        driver(sysp, basis, km, opts)


@pytest.mark.parametrize("driver, options_factory", [
    (vq.run_rhf_periodic_multi_k_ewald3d, _smearing_options),
    (vq.run_rks_periodic_multi_k_ewald3d, _ks_smearing_options),
])
def test_negative_smearing_temperature_rejected(driver, options_factory):
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = options_factory()
    opts.smearing_temperature = -0.001

    with pytest.raises(ValueError, match="smearing_temperature"):
        driver(sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3)
