"""SCF-level test of the Gilat-Raubenheimer occupation hook in the multi-k driver.

On a wide-gap insulator at a full mesh, GR occupations are exactly integer, so a
GR-driven SCF must reproduce the default (Aufbau) total energy. Also exercises
the full-mesh requirement and the smearing-incompatibility guard.

GR method: Gilat & Raubenheimer, Phys. Rev. 144, 390 (1966).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import vibeqc as vq

_driver = vq.run_rhf_periodic_multi_k_ewald3d


def _h2(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_doublet(box: float = 30.0):
    centre = box / 2.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [centre, centre, centre])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _opts(cutoff: float = 12.0):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = max(cutoff, 15.0)
    opts.max_iter = 40
    opts.use_diis = True
    return opts


def test_gilat_matches_aufbau_on_insulator():
    # Wide-gap insulator (isolated H2) at a full 2x2x2 mesh: the Gilat-
    # Raubenheimer net must give integer occupations and thus the same SCF
    # energy as the default Aufbau density build.
    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)

    r_aufbau = _driver(sysp, basis, kmesh, _opts())
    r_gilat = _driver(sysp, basis, kmesh, _opts(), bz_integration="gilat")

    assert r_gilat.converged
    assert r_gilat.energy == pytest.approx(r_aufbau.energy, abs=1e-6)

    # gapped system -> GR cell fractions collapse to integer occupations
    if r_gilat.occupations:
        for occ in r_gilat.occupations:
            occ = np.asarray(occ, dtype=float)
            assert np.allclose(occ, np.round(occ), atol=1e-6)


def test_gilat_rks_matches_aufbau_on_insulator():
    # Same contract for the closed-shell DFT driver (the route a real metal
    # uses): on a gapped system GR reproduces the Aufbau total energy.
    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1], use_symmetry=False)

    def _rks_opts():
        o = vq.PeriodicKSOptions()
        o.functional = "pbe"
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 15.0
        o.max_iter = 40
        o.use_diis = True
        return o

    r_aufbau = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, _rks_opts())
    r_gilat = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _rks_opts(), bz_integration="gilat"
    )
    assert r_gilat.converged
    assert r_gilat.energy == pytest.approx(r_aufbau.energy, abs=1e-6)


def test_runner_gilat_bipole_rks_path_runs(tmp_path, monkeypatch):
    """run_periodic_job forwards Gilat occupations to BIPOLE RKS.

    Asserts the forwarding itself: the resolved scheme must reach the solver
    as ``bz_integration="gilat"``. Through v0.15.23..v0.15.30 the runner
    accepted the keyword and silently dropped it at the run_pbc_bipole_rks
    callsite, and the earlier result-is-not-None form of this test stayed
    green for those eight releases (BUG-PER-002 in
    handovers/HANDOVER_OPEN_BUGS_V015.md). The run uses a converged SCF
    because ``run_periodic_job`` now fails closed on non-convergence
    (the 2026-08-13 periodic SCF gate).
    """
    import vibeqc.pbc_bipole_rks as bipole_rks_mod

    real_rks = bipole_rks_mod.run_pbc_bipole_rks
    seen_kwargs = {}

    def _capture(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return real_rks(*args, **kwargs)

    # periodic_runner imports run_pbc_bipole_rks from the module at call
    # time, so patching the module attribute intercepts the real dispatch.
    monkeypatch.setattr(bipole_rks_mod, "run_pbc_bipole_rks", _capture)

    sysp, basis = _h2(box=18.0)
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="lda",
        jk_method="bipole",
        kpoints=(2, 1, 1),
        bz_integration="gilat",
        max_iter=60,
        output=tmp_path / "gilat_bipole_rks",
        output_qvf=False,
        write_density=False,
        citations=True,
        progress=False,
    )
    assert seen_kwargs, "run_periodic_job never dispatched to run_pbc_bipole_rks"
    assert seen_kwargs.get("bz_integration") == "gilat"
    assert result is not None
    assert getattr(result, "occupations", None)
    references = (tmp_path / "gilat_bipole_rks.references").read_text(
        encoding="utf-8"
    )
    assert "Accurate Numerical Method" in references
    assert "Analysis of Methods" in references
    bibtex = (tmp_path / "gilat_bipole_rks.bibtex").read_text(encoding="utf-8")
    manifest = (tmp_path / "gilat_bipole_rks.system").read_text(encoding="utf-8")
    for key in ("gilat_raubenheimer_1966", "gilat_spectral_1972"):
        assert key in bibtex
        assert key in manifest
    assert 'status           = "complete"' in manifest


@pytest.mark.parametrize(
    "method,functional,module_name,driver_name",
    [
        ("UHF", None, "vibeqc.pbc_bipole_uhf", "run_pbc_bipole_uhf"),
        ("UKS", "lda", "vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
    ],
)
def test_runner_gilat_bipole_open_shell_forwards_per_spin_occupations(
    tmp_path, monkeypatch, method, functional, module_name, driver_name
):
    """Public BIPOLE UHF/UKS must execute, not drop, per-spin GR.

    Runs a converged SCF: ``run_periodic_job`` fails closed on
    non-convergence (the 2026-08-13 periodic SCF gate), so the
    occupation inspection needs the returned converged result.
    """
    driver_module = importlib.import_module(module_name)
    real_driver = getattr(driver_module, driver_name)
    seen_kwargs = {}

    def _capture(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return real_driver(*args, **kwargs)

    monkeypatch.setattr(driver_module, driver_name, _capture)

    sysp, basis = _h_atom_doublet()
    kwargs = {} if functional is None else {"functional": functional}
    stem = tmp_path / f"gilat_bipole_{method.lower()}"
    result = vq.run_periodic_job(
        sysp,
        basis,
        method=method,
        jk_method="bipole",
        kpoints=(2, 1, 1),
        bz_integration="gilat",
        max_iter=40,
        output=stem,
        output_qvf=False,
        write_density=False,
        citations=True,
        progress=False,
        **kwargs,
    )

    assert seen_kwargs.get("bz_integration") == "gilat"
    weights = np.asarray(result.kpoint_weights, dtype=float)
    n_alpha = sum(
        weight * np.asarray(occ, dtype=float).sum()
        for weight, occ in zip(weights, result.occupations_alpha)
    )
    n_beta = sum(
        weight * np.asarray(occ, dtype=float).sum()
        for weight, occ in zip(weights, result.occupations_beta)
    )
    assert n_alpha == pytest.approx(1.0, abs=1e-10)
    assert n_beta == pytest.approx(0.0, abs=1e-10)
    references = stem.with_suffix(".references").read_text(encoding="utf-8")
    assert "Accurate Numerical Method" in references
    assert "Analysis of Methods" in references


@pytest.mark.parametrize(
    "method,functional,expected_energy",
    [
        # Pins refreshed 2026-08-29 (issue #515): the fixed point moved by
        # -8.9897e-6 Ha (identical for UHF and LDA, so shared 1e machinery)
        # when 9cca63c16 (issue #478) sized the BIPOLE 1e Ewald real-space
        # sum from the pinned CRYSTAL alpha instead of the AO overlap cutoff.
        # That is the deliberate monopole-self-image truncation fix; the
        # vacuum-box H doublet is exactly its blast-radius class (image-cell
        # count 1 -> 19). Verified: at 9cca63c16~1 the old pins pass, at
        # 9cca63c16 and at main the values below reproduce to 1e-12.
        #
        # UHF pin refreshed 2026-09-06 (issue #674): 9b8bb42b4 bounds the
        # corrected split's default alpha from below by the exchange cutoff
        # (2.8/30 = 0.0933 -> 0.358 bohr^-1 in this 30-bohr box at a
        # 12-bohr cutoff), which moved the UHF fixed point by -6.028e-6 Ha.
        # Attributed by rerunning with ``ewald_omega=2.8/30`` (the CRYSTAL
        # value the old default resolved to): that reproduces the old pin
        # to 1.4e-12 on the current tree. LDA has no exchange arm, so its
        # alpha and its pin did not move. The 1.4e-12 residual of a
        # same-alpha rerun is machine round-off, so the pin comparison
        # below is at 1e-10: still four orders below every fixed-point
        # move this test has ever caught (8.99e-6, 6.03e-6).
        ("UHF", None, -0.46665742526772),
        ("UKS", "lda", -0.43567023297923124),
    ],
)
def test_bipole_open_shell_gilat_matches_gapped_aufbau_fixed_point(
    tmp_path, method, functional, expected_energy
):
    """GR reduces exactly to Aufbau for a gapped one-electron band."""
    sysp, basis = _h_atom_doublet()
    method_kwargs = {} if functional is None else {"functional": functional}
    results = {}
    for label, bz_integration in (("aufbau", None), ("gilat", "gilat")):
        results[label] = vq.run_periodic_job(
            sysp,
            basis,
            method=method,
            jk_method="bipole",
            kpoints=(2, 1, 1),
            bz_integration=bz_integration,
            max_iter=40,
            output=tmp_path / f"{method.lower()}-{label}",
            output_qvf=False,
            write_density=False,
            write_xyz_file=False,
            write_cif_file=False,
            write_xsf_structure_file=False,
            citations=False,
            progress=False,
            **method_kwargs,
        )

    aufbau = results["aufbau"]
    gilat = results["gilat"]
    assert aufbau.converged and gilat.converged
    assert gilat.energy == pytest.approx(expected_energy, rel=0.0, abs=1e-10)
    assert gilat.energy == pytest.approx(aufbau.energy, rel=0.0, abs=1e-12)
    assert gilat.entropy == pytest.approx(0.0, abs=1e-15)
    assert gilat.s_squared == pytest.approx(0.75, abs=1e-12)
    for occ in gilat.occupations_alpha + gilat.occupations_beta:
        assert np.allclose(occ, np.round(occ), atol=1e-12)


def test_bipole_open_shell_helper_uses_fractional_gilat_microcells():
    """The shared BIPOLE seam must not reduce GR to integer slicing."""
    from vibeqc.pbc_bipole_common import unrestricted_occupations_per_spin

    sysp, _ = _h_atom_doublet(box=10.0)
    kmesh = vq.monkhorst_pack(sysp, [3, 1, 1])
    eps_per_k = [
        np.array([-1.0, 0.2]),
        np.array([0.4, -0.7]),
        np.array([0.3, 0.9]),
    ]
    occupations, _, entropy = unrestricted_occupations_per_spin(
        eps_per_k,
        1,
        smearing_T=0.0,
        weights=kmesh.weights,
        system=sysp,
        kmesh=kmesh,
        bz_integration="gilat",
    )

    electron_count = sum(
        weight * np.asarray(occ).sum()
        for weight, occ in zip(kmesh.weights, occupations)
    )
    assert electron_count == pytest.approx(1.0, abs=1e-10)
    assert entropy == pytest.approx(0.0, abs=1e-15)
    assert any(
        np.any((np.asarray(occ) > 1e-8) & (np.asarray(occ) < 1.0 - 1e-8))
        for occ in occupations
    )


@pytest.mark.parametrize("method,functional", [("UHF", None), ("UKS", "lda")])
def test_bipole_open_shell_gilat_rejects_finite_smearing(
    tmp_path, method, functional
):
    sysp, basis = _h_atom_doublet()
    method_kwargs = {} if functional is None else {"functional": functional}
    with pytest.raises(ValueError, match="T=0 integrator"):
        vq.run_periodic_job(
            sysp,
            basis,
            method=method,
            jk_method="bipole",
            kpoints=(2, 1, 1),
            bz_integration="gilat",
            smearing_temperature=0.01,
            output=tmp_path / f"{method.lower()}-gilat-smearing",
            output_qvf=False,
            citations=False,
            progress=False,
            **method_kwargs,
        )


def test_gilat_uks_matches_aufbau_on_open_shell():
    # Open-shell DFT (H-atom doublet) at a full mesh: per-spin GR occupations
    # are integer for the gapped system, so UKS-GR energy == UKS Aufbau.
    box = 30.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box, [vq.Atom(1, [c, c, c])], charge=0, multiplicity=2
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1], use_symmetry=False)

    def _uks_opts():
        o = vq.PeriodicKSOptions()
        o.functional = "pbe"
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 15.0
        o.max_iter = 60
        o.use_diis = True
        return o

    r_aufbau = vq.run_uks_periodic_multi_k_ewald3d(sysp, basis, kmesh, _uks_opts())
    r_gilat = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _uks_opts(), bz_integration="gilat"
    )
    assert r_gilat.converged
    assert r_gilat.energy == pytest.approx(r_aufbau.energy, abs=1e-6)


def test_gilat_uks_terminal_metadata_matches_reported_spectrum_and_density():
    """The final Gilat state must be regenerated from the reported bands."""
    from vibeqc.bz_integration import gilat_occupations_for_kmesh

    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1], use_symmetry=False)
    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.max_iter = 60
    opts.use_diis = True

    result = vq.run_uks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        bz_integration="gilat",
        progress=False,
    )

    assert result.converged
    expected_alpha, mu_alpha = gilat_occupations_for_kmesh(
        sysp,
        kmesh,
        result.mo_energies_alpha,
        1.0,
        spin_degeneracy=1.0,
    )
    expected_beta, mu_beta = gilat_occupations_for_kmesh(
        sysp,
        kmesh,
        result.mo_energies_beta,
        1.0,
        spin_degeneracy=1.0,
    )
    for got, expected in zip(result.occupations_alpha, expected_alpha):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(result.occupations_beta, expected_beta):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    assert result.fermi_level_alpha == pytest.approx(mu_alpha, abs=1e-12)
    assert result.fermi_level_beta == pytest.approx(mu_beta, abs=1e-12)
    assert result.entropy == pytest.approx(0.0, abs=1e-15)

    expected_density_alpha = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            result.mo_coeffs_alpha,
            result.occupations_alpha,
            kmesh,
            result.density_alpha.cells,
        )
    )
    expected_density_beta = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            result.mo_coeffs_beta,
            result.occupations_beta,
            kmesh,
            result.density_beta.cells,
        )
    )
    for got, expected in zip(
        result.density_alpha.blocks, expected_density_alpha.blocks
    ):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(
        result.density_beta.blocks, expected_density_beta.blocks
    ):
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_gilat_uhf_matches_aufbau_and_conserves_each_spin(monkeypatch):
    """UHF consumes the shared GR kernel independently for alpha and beta."""
    import vibeqc.bz_integration as bz

    real_gilat = bz.gilat_occupations_for_kmesh
    observed = []

    def _capture(system, kmesh, eps_per_k, n_elec, spin_degeneracy=2.0):
        occ, e_fermi = real_gilat(
            system,
            kmesh,
            eps_per_k,
            n_elec,
            spin_degeneracy=spin_degeneracy,
        )
        count = sum(
            float(weight) * float(np.asarray(block).sum())
            for weight, block in zip(kmesh.weights, occ)
        )
        observed.append((float(n_elec), float(spin_degeneracy), count))
        return occ, e_fermi

    monkeypatch.setattr(bz, "gilat_occupations_for_kmesh", _capture)
    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1], use_symmetry=False)
    opts = _opts()
    r_aufbau = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, opts, progress=False
    )
    r_gilat = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        _opts(),
        bz_integration="gilat",
        progress=False,
    )

    assert r_gilat.converged
    assert r_gilat.energy == pytest.approx(r_aufbau.energy, abs=1e-6)
    assert r_gilat.entropy == pytest.approx(0.0, abs=1e-15)
    assert np.isfinite(r_gilat.fermi_level_alpha)
    assert np.isfinite(r_gilat.fermi_level_beta)
    expected_alpha, mu_alpha = real_gilat(
        sysp,
        kmesh,
        r_gilat.mo_energies_alpha,
        1.0,
        spin_degeneracy=1.0,
    )
    expected_beta, mu_beta = real_gilat(
        sysp,
        kmesh,
        r_gilat.mo_energies_beta,
        1.0,
        spin_degeneracy=1.0,
    )
    for got, expected in zip(r_gilat.occupations_alpha, expected_alpha):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(r_gilat.occupations_beta, expected_beta):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    assert r_gilat.fermi_level_alpha == pytest.approx(mu_alpha, abs=1e-12)
    assert r_gilat.fermi_level_beta == pytest.approx(mu_beta, abs=1e-12)
    assert observed
    for target, degeneracy, count in observed:
        assert degeneracy == pytest.approx(1.0)
        assert count == pytest.approx(target, abs=1e-10)


def test_gilat_uhf_matches_aufbau_on_open_shell():
    box = 30.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box, [vq.Atom(1, [c, c, c])], multiplicity=2
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1], use_symmetry=False)
    r_aufbau = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _opts(), progress=False
    )
    r_gilat = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        _opts(),
        bz_integration="gilat",
        progress=False,
    )
    assert r_gilat.converged
    assert r_gilat.energy == pytest.approx(r_aufbau.energy, abs=1e-6)
    assert r_gilat.s_squared == pytest.approx(0.75, abs=1e-10)


def test_gilat_uhf_ibz_mesh_matches_full_mesh():
    """The UHF selector preserves the shared full-BZ/IBZ GR contract."""
    sysp, basis = _h2()
    attach = getattr(vq, "attach_symmetry", None)
    if attach is None:
        pytest.skip("attach_symmetry unavailable")
    attach(sysp)
    kmesh_ibz = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    kmesh_full = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    if len(kmesh_ibz) >= len(kmesh_full):
        pytest.skip("mesh was not symmetry-reduced for this system")
    r_full = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, kmesh_full, _opts(), bz_integration="gilat", progress=False
    )
    r_ibz = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, kmesh_ibz, _opts(), bz_integration="gilat", progress=False
    )
    assert r_full.converged and r_ibz.converged
    assert r_ibz.energy == pytest.approx(r_full.energy, abs=1e-6)


def test_gilat_uhf_rejects_finite_smearing():
    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _opts()
    opts.smearing_temperature = 0.01
    with pytest.raises(ValueError, match="T=0 integrator"):
        vq.run_uhf_periodic_multi_k_ewald3d(
            sysp, basis, kmesh, opts, bz_integration="gilat", progress=False
        )


def test_gilat_uhf_rejects_unknown_selector():
    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    with pytest.raises(ValueError, match="bz_integration must be"):
        vq.run_uhf_periodic_multi_k_ewald3d(
            sysp, basis, kmesh, _opts(), bz_integration="bogus", progress=False
        )


def test_gilat_ibz_mesh_matches_full_mesh_on_exact_exchange():
    """GR on a symmetry-reduced mesh matches GR on the full mesh, for HF.

    Third life of this test, each stage meaningful. Originally it asserted
    IBZ == full and passed only because *both* sides summed the full-range
    exchange as a divergent bare ``1/r`` image sum. When the full mesh
    moved to the corrected Ewald split (2026-07-28), IBZ was made to fail
    closed rather than disagree by 4.6e-5 Ha. Since 2026-07-29 the wedge
    is served properly -- the SCF stays on the irreducible points and the
    density is unfolded to the full BZ by the Pisani/Dovesi star transport
    -- so the parity assertion returns, now meaning something: both sides
    run the same convergent exchange.

    GR occupations are star-invariant (``eps_n(Rk) = eps_n(k)``), so the
    gapped-insulator fixture must reproduce the full-mesh energy tightly.
    """
    sysp, basis = _h2()
    attach = getattr(vq, "attach_symmetry", None)
    if attach is None:
        pytest.skip("attach_symmetry unavailable")
    attach(sysp)
    kmesh_ibz = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    kmesh_full = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    if len(kmesh_ibz) >= len(kmesh_full):
        pytest.skip("mesh was not symmetry-reduced for this system")
    r_full = _driver(sysp, basis, kmesh_full, _opts(), bz_integration="gilat")
    r_ibz = _driver(sysp, basis, kmesh_ibz, _opts(), bz_integration="gilat")
    assert r_full.converged and r_ibz.converged
    assert r_ibz.energy == pytest.approx(r_full.energy, abs=1e-8)
    assert r_ibz.used_kpoint_symmetry_unfolding is True


def test_gilat_ibz_occupations_expand_to_full_bz():
    """The GR IBZ eigenvalue expansion, isolated from the exchange gauge.

    ``eps_n(Rk) = eps_n(k)``, so a GR net built from an irreducible wedge
    must reproduce the full-mesh occupations. Checked on the occupation
    numbers directly rather than through an SCF energy, so it stays valid
    regardless of how exact exchange is treated.
    """
    sysp, basis = _h2()
    attach = getattr(vq, "attach_symmetry", None)
    if attach is None:
        pytest.skip("attach_symmetry unavailable")
    attach(sysp)
    kmesh_ibz = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    kmesh_full = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    if len(kmesh_ibz) >= len(kmesh_full):
        pytest.skip("mesh was not symmetry-reduced for this system")

    from vibeqc.bz_integration.grid import gilat_occupations_for_kmesh

    n_bands = basis.nbasis
    # A gapped two-electron band structure: one filled band, rest empty,
    # identical at every k (isolated H2 in a large box).
    eps_full = [
        np.array([-0.6] + [0.4] * (n_bands - 1)) for _ in kmesh_full.kpoints
    ]
    eps_ibz = [
        np.array([-0.6] + [0.4] * (n_bands - 1)) for _ in kmesh_ibz.kpoints
    ]
    occ_full, _ = gilat_occupations_for_kmesh(sysp, kmesh_full, eps_full, 2)
    occ_ibz, _ = gilat_occupations_for_kmesh(sysp, kmesh_ibz, eps_ibz, 2)
    n_full = sum(
        float(w) * float(np.sum(o))
        for w, o in zip(kmesh_full.weights, occ_full)
    )
    n_ibz = sum(
        float(w) * float(np.sum(o))
        for w, o in zip(kmesh_ibz.weights, occ_ibz)
    )
    assert n_full == pytest.approx(2.0, abs=1e-9)
    assert n_ibz == pytest.approx(n_full, abs=1e-9)


def test_gilat_rejects_finite_smearing():
    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _opts()
    opts.smearing_temperature = 0.01
    with pytest.raises(ValueError, match="T=0 integrator"):
        _driver(sysp, basis, kmesh, opts, bz_integration="gilat")


def test_invalid_bz_integration_rejected():
    sysp, basis = _h2()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    with pytest.raises(ValueError, match="bz_integration must be"):
        _driver(sysp, basis, kmesh, _opts(), bz_integration="bogus")
