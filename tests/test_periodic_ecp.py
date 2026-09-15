"""Periodic ECP regression tests: PBE / pob-TZVP-REV2 on 4d/5d metals.

Phase 14h — validates the inline-primitive ECP path (compute_ecp_lattice_from_primitives)
against PySCF.pbc with the same geometry, basis, and ECP.

Target: ≤ 1 mHa/atom agreement with PySCF.pbc (subprocess runner, CLAUDE.md §10).

Systems:
  - Ag fcc (Z=47, 4-atom conventional cell, ecp28mdf)
  - Au fcc (Z=79, 4-atom conventional cell, ecp60mdf)
  - W bcc  (Z=74, 2-atom conventional cell, ecp60mdf)

Validation procedure (to be run after this code compiles):
  1. Compute each system with vibe-qc at PBE/pob-TZVP-REV2 + ECP.
  2. Run the same system with PySCF.pbc at PBE/pob-TZVP-REV2 + ECP
     (using the subprocess-runner pattern at examples/regression/core/runner_pyscf.py).
  3. Check |E_vibe - E_pyscf| / n_atoms ≤ 1e-3 Ha.
  4. Pin the PySCF value as the inline reference.

The reference values below are PLACEHOLDERS — they MUST be updated once the
PySCF cross-validation is complete.  Replace each ``_PYSCF_REF_*`` constant
with the PySCF total energy and flip the corresponding ``@pytest.mark.skip``
to activate the parity check.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq

ANG = 1.8897261246257702


def _patch_bredow_fetcher(monkeypatch, replacement):
    """Replace ``fetch_bredow_basis_sets`` where ``_resolve_ecp_data`` looks.

    ``_resolve_ecp_data`` reaches the fetcher through ``from .basis_crystal
    import fetch_bredow_basis_sets``, which resolves through ``sys.modules``,
    whereas ``import vibeqc.basis_crystal`` here binds the ``vibeqc`` package
    attribute.  Those are the same object in a healthy interpreter — but a test
    that registers its own copy of the module under the real name splits them,
    and then patching one leaves the live fetcher reachable from the other.
    That is not a hypothetical: it silently disarmed the fail-closed guard below
    for any run where ``tests/basisset_dev/test_ld_penalty_inmemory.py`` was
    collected first.  Fail loudly on the split rather than testing nothing.
    """
    import vibeqc.basis_crystal as basis_crystal

    assert sys.modules["vibeqc.basis_crystal"] is basis_crystal, (
        "a second copy of vibeqc.basis_crystal is registered in sys.modules; "
        "monkeypatching the fetcher would not reach _resolve_ecp_data, so this "
        "ECP-provenance guard would pass without exercising anything"
    )
    monkeypatch.setattr(basis_crystal, "fetch_bredow_basis_sets", replacement)


# =========================================================================
# Helper: build periodic system from qc-input-library parameters
# =========================================================================


def _build_ag_fcc() -> vq.PeriodicSystem:
    """Ag fcc, conventional 4-atom cell, a = 4.062 Å (CRYSTAL reference geometry).

    Space group Fm-3m (no. 225). Atoms at (0,0,0), (0,1/2,1/2), (1/2,0,1/2),
    (1/2,1/2,0) in fractional coordinates.
    """
    lat = np.eye(3) * 4.062 * ANG
    a = 4.062 * ANG
    atoms = [
        vq.Atom(47, [0.0, 0.0, 0.0]),
        vq.Atom(47, [0.0, a / 2, a / 2]),
        vq.Atom(47, [a / 2, 0.0, a / 2]),
        vq.Atom(47, [a / 2, a / 2, 0.0]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def _build_au_fcc() -> vq.PeriodicSystem:
    """Au fcc, conventional 4-atom cell, a = 4.063 Å."""
    lat = np.eye(3) * 4.063 * ANG
    a = 4.063 * ANG
    atoms = [
        vq.Atom(79, [0.0, 0.0, 0.0]),
        vq.Atom(79, [0.0, a / 2, a / 2]),
        vq.Atom(79, [a / 2, 0.0, a / 2]),
        vq.Atom(79, [a / 2, a / 2, 0.0]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def _build_w_bcc() -> vq.PeriodicSystem:
    """W bcc, conventional 2-atom cell, a = 3.191 Å.

    Space group Im-3m (no. 229). Atoms at (0,0,0), (1/2,1/2,1/2) in
    fractional coordinates.
    """
    lat = np.eye(3) * 3.191 * ANG
    a = 3.191 * ANG
    atoms = [
        vq.Atom(74, [0.0, 0.0, 0.0]),
        vq.Atom(74, [a / 2, a / 2, a / 2]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


# =========================================================================
# PySCF reference energies (PLACEHOLDERS — fill in after cross-validation)
#
#   E_vibe   → total energy from vibe-qc PBE/pob-TZVP-REV2 + ECP
#   E_pyscf  → total energy from PySCF.pbc PBE/pob-TZVP-REV2 + ECP
#   ΔE/atom  → |E_vibe - E_pyscf| / n_atoms_cell
#
# The target is ΔE/atom ≤ 1e-3 Ha (1 mHa).
# =========================================================================

# Placeholder — fill in after PySCF cross-validation.
# PySCF energy for Ag fcc, PBE/pob-TZVP-REV2, ECP (ecp28mdf), 4-atom cell.
_PYSCF_REF_AG_FCC = None  # TODO: run runner_pyscf.py and pin the value here

# PySCF energy for Au fcc, PBE/pob-TZVP-REV2, ECP (ecp60mdf), 4-atom cell.
_PYSCF_REF_AU_FCC = None

# PySCF energy for W bcc, PBE/pob-TZVP-REV2, ECP (ecp60mdf), 2-atom cell.
_PYSCF_REF_W_BCC = None


# =========================================================================
# Helper: run a periodic PBE SCF with pob-TZVP-REV2 and check ECP wiring
# =========================================================================


def _run_pbe_ecp(system, kmesh=(2, 2, 2)):
    """Run PBE / pob-TZVP-REV2 on a periodic system with ECP auto-attach.

    Uses a small k-mesh for speed — the ECP path is independent of
    k-point density, so a 2×2×2 mesh is sufficient to exercise the
    full lattice-summed V_ECP code path.
    """
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")

    from vibeqc.kpoints import KPoints

    kp = KPoints.monkhorst_pack(system, kmesh)
    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-7
    opts.smearing_temperature = 0.005  # metals need smearing
    opts.use_diis = True

    # Attach ECP data from pob-TZVP-REV2 inline CRYSTAL ECP blocks.
    from vibeqc.basis_crystal import (
        build_periodic_ecp_data,
        fetch_bredow_basis_sets,
    )

    _, atoms_dict = fetch_bredow_basis_sets(
        names=["pob-tzvp-rev2"], verbose=False, return_atoms=True
    )
    atom_list = atoms_dict.get("pob-tzvp-rev2", [])
    if atom_list:
        ecp_blocks, ecp_centers, eff_z, total_ncore = build_periodic_ecp_data(
            system, atom_list
        )
        if ecp_blocks:
            opts.ecp_primitive_blocks = ecp_blocks
            opts.ecp_home_centers = ecp_centers
            opts.ecp_effective_charges = eff_z
            opts.ecp_total_ncore = total_ncore

    result = vq.run_rks_periodic(system, basis, kp.to_bloch_kmesh(), opts)

    # Sanity: total energy should be strongly negative for all three metals.
    # For Ag fcc (4 atoms), E_total ~ -585 Ha at PBE/ECP (approx).
    # For Au fcc (4 atoms), E_total ~ -540 Ha at PBE/ECP (approx).
    # For W bcc (2 atoms), E_total ~ -120 Ha at PBE/ECP (approx).
    assert result.energy < -10, (
        f"Implausible total energy {result.energy:.6f} Ha — "
        "ECP wiring may be broken (double-counting core electrons?)"
    )

    return result


# =========================================================================
# Tests
# =========================================================================


@pytest.mark.parametrize(
    "builder, expected_blocks, expected_zeff, expected_ncore",
    [
        pytest.param(_build_ag_fcc, 4, 19.0, 4 * 28, id="Ag-fcc"),
        pytest.param(_build_au_fcc, 4, 19.0, 4 * 60, id="Au-fcc"),
        pytest.param(_build_w_bcc, 2, 14.0, 2 * 60, id="W-bcc"),
    ],
)
def test_p09_pob_tzvp_rev2_auto_attaches_ecp_blocks(
    builder,
    expected_blocks,
    expected_zeff,
    expected_ncore,
):
    """P09 Ag/Au/W pob-TZVP-REV2 cells must attach inline CRYSTAL ECP data.

    This is the always-on guard for the release-paper P09 setup. The external
    PySCF parity numbers below remain skipped until those references are
    regenerated, but the in-code ECP provenance path must already know that
    these cells are pseudopotential, not all-electron, calculations.
    """
    system = builder()
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")

    from vibeqc.periodic_runner import _resolve_ecp_data

    ecp_blocks, ecp_centers, eff_z, total_ncore = _resolve_ecp_data(system, basis)

    assert len(ecp_blocks) == expected_blocks
    assert len(ecp_centers) == expected_blocks
    assert eff_z == pytest.approx([expected_zeff] * expected_blocks)
    assert total_ncore == expected_ncore

    np.testing.assert_allclose(
        ecp_centers,
        [list(atom.xyz) for atom in system.unit_cell],
        atol=0.0,
        rtol=0.0,
    )
    for block in ecp_blocks:
        assert block.n_primitive > 0
        assert len(block.exponents) == block.n_primitive
        assert len(block.coefficients) == block.n_primitive
        assert len(block.ams) == block.n_primitive
        assert len(block.ns) == block.n_primitive


@pytest.mark.parametrize("charges", [
    [3., 1.], [1., 1.], [1., 3.], [3.], [3.000000001, 1.], [3., float("nan")],
])
def test_guess_charge_only_metadata_requires_exact_all_electron_identity(charges):
    from vibeqc.guess import guess_ecp_context, validate_guess_ecp

    mol = vq.Molecule([vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [2., 0., 0.])])
    options = SimpleNamespace(ecp_effective_charges=charges)
    context = guess_ecp_context(options, molecule=mol)
    if charges == [3., 1.]:
        assert not context.active
        assert list(context.effective_charges) == []
        validate_guess_ecp("SAD", context, mol)
    else:
        with pytest.raises(ValueError, match="non-empty ecp_primitive_blocks"):
            validate_guess_ecp("SAD", context, mol)
    # Without an explicit parent molecule there is no evidence to clear it.
    with pytest.raises(ValueError, match="non-empty ecp_primitive_blocks"):
        validate_guess_ecp("SAD", guess_ecp_context(options), mol)


@pytest.mark.parametrize("fault", ["core", "fractional_core", "center", "xml"])
def test_all_electron_charges_do_not_mask_orphan_ecp_metadata(fault):
    from vibeqc.guess import guess_ecp_context, validate_guess_ecp

    mol = vq.Molecule([vq.Atom(2, [0., 0., 0.])])
    options = SimpleNamespace(ecp_effective_charges=[2.])
    if fault == "core":
        options.ecp_total_ncore = 2
    elif fault == "fractional_core":
        options.ecp_total_ncore = 0.5
    elif fault == "center":
        options.ecp_home_centers = [[0., 0., 0.]]
    else:
        options.ecp_library = "ecp28mdf"
    with pytest.raises(ValueError):
        validate_guess_ecp("HCORE", guess_ecp_context(options, molecule=mol), mol)


@pytest.mark.parametrize("basis_name", ["pob-tzvp", "pob-tzvp-rev2"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("route", ["gdf", "bipole", "gpw", "gapw"])
@pytest.mark.parametrize("guess", ["AUTO", "HCORE", "SAD", "SAP", "MINAO"])
def test_public_all_electron_pob_guess_preflight(tmp_path, basis_name, mesh, route, guess):
    """#763: real POB metadata must pass the public route/guess preflight."""
    system = vq.PeriodicSystem(3, np.eye(3) * 8., [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [2., 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    vq.run_periodic_job(
        system, basis, method="RKS", functional="lda", jk_method=route,
        kpoints=mesh, initial_guess=guess, dry_run=True, progress=False,
        output=str(tmp_path / "pob"), output_qvf=False,
    )


@pytest.mark.parametrize("charges", [[1., 1.], [3., float("nan")]])
def test_public_pob_incomplete_ecp_fails_before_output(tmp_path, monkeypatch, charges):
    import vibeqc.periodic_runner as runner

    system = vq.PeriodicSystem(3, np.eye(3) * 8., [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [2., 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")
    monkeypatch.setattr(runner, "_resolve_ecp_data", lambda *args: ([], [], charges, 0))
    with pytest.raises(ValueError, match="non-empty ecp_primitive_blocks"):
        vq.run_periodic_job(system, basis, method="RHF", jk_method="gdf",
                            output=str(tmp_path / "invalid"), progress=False)
    assert not list(tmp_path.glob("invalid*"))


@pytest.mark.parametrize(
    "mesh, expected_driver",
    [
        (None, "pbc_gdf"),
        ((1, 1, 1), "periodic_k_gdf"),
        ((3, 1, 1), "periodic_k_gdf"),
    ],
    ids=["default-gamma", "explicit-gamma", "multi-k"],
)
def test_public_all_electron_pob_reaches_normalized_sad(
    tmp_path, monkeypatch, mesh, expected_driver
):
    """The public GDF route reaches the real SAD builder of the driver it
    actually selects, and that seed is normalized and Hermitian.

    Default Gamma (``kpoints=None``) runs ``pbc_gdf``; every explicit mesh,
    the one-point ``(1, 1, 1)`` included, runs the k-point drivers in
    ``periodic_k_gdf`` (#209: the earlier version patched ``pbc_gdf`` for
    the explicit Gamma mesh, so it never observed the builder that ran and
    failed on ``reached == []``). The one-point mesh stays in that general
    k engine because the shipped SR/LR ``rsgdf`` bulk route sets
    ``gamma_info = None`` (``periodic_k_gdf.py``, ``bulk_sr``); with another
    ``gdf_method`` the k driver would delegate a one-point mesh to
    ``run_pbc_gdf_rhf`` and this expectation would move with it. Both
    drivers are probed in every case, so a route that skips the selected
    builder, or selects the other driver, fails here rather than passing
    vacuously.
    """
    from vibeqc import pbc_gdf, periodic_k_gdf

    system = vq.PeriodicSystem(3, np.eye(3) * 8., [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [2., 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")
    drivers = {"pbc_gdf": pbc_gdf, "periodic_k_gdf": periodic_k_gdf}
    reached = []

    def _probe_for(name, builder):
        def probe(*args, **kwargs):
            context = kwargs.get("ecp_context")
            assert context is None or not context.active
            density = builder(*args, **kwargs)
            assert density is not None, f"{name}: builder returned no density"
            metric = np.asarray(kwargs["overlap"])
            if metric.ndim == 3:
                metric = sum(w * s for w, s in zip(kwargs["weights"], metric))
            assert np.trace(density @ metric).real == pytest.approx(4., abs=1e-11)
            np.testing.assert_allclose(density, density.conj().T, atol=1e-13, rtol=0)
            reached.append(name)
            return density
        return probe

    for name, module in drivers.items():
        monkeypatch.setattr(
            module, "initial_density_closed_shell",
            _probe_for(name, module.initial_density_closed_shell),
        )
    # Bound the quadrature cost of this adapter regression. This test checks
    # the selected seed and convergence, not absolute GDF energy accuracy.
    result = vq.run_periodic_job(
        system, basis, method="RHF", jk_method="gdf", aux_basis="def2-svp-jk",
        rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0., convergence="off",
        kpoints=mesh, initial_guess="AUTO", max_iter=80, progress=False,
        output=str(tmp_path / "pob-sad"), output_qvf=False,
    )
    assert reached, "no GDF driver reached its SAD builder"
    assert set(reached) == {expected_driver}, reached
    assert result.converged
    assert result.guess_selection.effective == vq.InitialGuess.SAD


def test_pob_tzvp_rev2_all_electron_sources_do_not_fetch(monkeypatch):
    """Ni/O POB-TZVP-REV2 is all-electron and should resolve locally.

    Prompt 55's P16 artifact warned about a failed live ECP fetch before
    continuing as all-electron. For NiO no ECP is expected, so the resolver
    should use the checked-in H-Br/Ni source files and avoid the network path
    entirely.
    """
    a = 4.17 * ANG
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * a,
        [
            vq.Atom(28, [0.0, 0.0, 0.0]),
            vq.Atom(8, [a / 2, a / 2, a / 2]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")

    def _unexpected_fetch(*args, **kwargs):
        raise AssertionError("all-electron bundled POB sources should not fetch")

    from vibeqc.periodic_runner import _resolve_ecp_data

    _patch_bredow_fetcher(monkeypatch, _unexpected_fetch)

    ecp_blocks, ecp_centers, eff_z, total_ncore = _resolve_ecp_data(system, basis)

    assert ecp_blocks == []
    assert ecp_centers == []
    assert eff_z == pytest.approx([28.0, 8.0])
    assert total_ncore == 0


def test_pob_tzvp_rev2_bundled_ecp_atoms_resolve_offline(monkeypatch):
    """W bcc resolves its ECP from the bundled CRYSTAL records; no fetch."""
    system = _build_w_bcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")

    def _unexpected_fetch(*args, **kwargs):
        raise AssertionError("bundled POB ECP records should not fetch")

    from vibeqc.periodic_runner import _resolve_ecp_data

    _patch_bredow_fetcher(monkeypatch, _unexpected_fetch)

    ecp_blocks, _ecp_centers, eff_z, total_ncore = _resolve_ecp_data(system, basis)
    assert len(ecp_blocks) == 2 and eff_z == pytest.approx([14.0, 14.0])
    assert total_ncore == 2 * 60


def test_pob_tzvp_rev2_ecp_atoms_fail_closed_when_fetch_unavailable(monkeypatch):
    """An element with no bundled record must not fall back to all-electron
    silently when the fetch is unavailable (xenon has no pob-TZVP-rev2
    record at all; the basis object stands in because libint would refuse
    to build it)."""
    from types import SimpleNamespace

    a = 6.0 * ANG
    system = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(54, [0.0, 0.0, 0.0])])
    basis = SimpleNamespace(name="pob-tzvp-rev2")

    def _offline_fetch(*args, **kwargs):
        raise OSError("network unavailable")

    from vibeqc.periodic_runner import _resolve_ecp_data

    _patch_bredow_fetcher(monkeypatch, _offline_fetch)

    with pytest.raises(RuntimeError, match="will not continue as an all-electron"):
        _resolve_ecp_data(system, basis)


@pytest.mark.slow
@pytest.mark.skip(reason="needs PySCF reference value (see _PYSCF_REF_AG_FCC)")
def test_periodic_ecp_ag_fcc_pbe():
    """Ag fcc PBE/pob-TZVP-REV2 ECP: parity with PySCF.pbc ≤ 1 mHa/atom.

    Ag (Z=47): 28 core electrons replaced by ECP (ecp28mdf equivalent),
    19 valence electrons per atom.  4-atom conventional cell → 76 valence
    electrons (38 occupied).

    Reference: runner_pyscf.py PBE/pob-TZVP-REV2 + ECP.
    """
    system = _build_ag_fcc()
    result = _run_pbe_ecp(system)
    # Sanity: net charge per atom = 19 valence
    assert system.n_electrons() == 47 * 4  # 188 total (but SCF uses valence)
    assert abs(result.energy - _PYSCF_REF_AG_FCC) / 4 < 1e-3, (
        f"Ag fcc ECP energy {result.energy:.8f} Ha deviates from PySCF "
        f"{_PYSCF_REF_AG_FCC:.8f} by "
        f"{abs(result.energy - _PYSCF_REF_AG_FCC) / 4:.6f} Ha/atom "
        f"(limit 1e-3 Ha/atom)"
    )


@pytest.mark.slow
@pytest.mark.skip(reason="needs PySCF reference value (see _PYSCF_REF_AU_FCC)")
def test_periodic_ecp_au_fcc_pbe():
    """Au fcc PBE/pob-TZVP-REV2 ECP: parity with PySCF.pbc ≤ 1 mHa/atom.

    Au (Z=79): 60 core electrons replaced by ECP (ecp60mdf equivalent),
    19 valence electrons per atom.  4-atom conventional cell → 76 valence
    electrons (38 occupied).

    Reference: runner_pyscf.py PBE/pob-TZVP-REV2 + ECP.
    """
    system = _build_au_fcc()
    result = _run_pbe_ecp(system)
    assert abs(result.energy - _PYSCF_REF_AU_FCC) / 4 < 1e-3, (
        f"Au fcc ECP energy {result.energy:.8f} Ha deviates from PySCF "
        f"{_PYSCF_REF_AU_FCC:.8f} by "
        f"{abs(result.energy - _PYSCF_REF_AU_FCC) / 4:.6f} Ha/atom "
        f"(limit 1e-3 Ha/atom)"
    )


@pytest.mark.slow
@pytest.mark.skip(reason="needs PySCF reference value (see _PYSCF_REF_W_BCC)")
def test_periodic_ecp_w_bcc_pbe():
    """W bcc PBE/pob-TZVP-REV2 ECP: parity with PySCF.pbc ≤ 1 mHa/atom.

    W (Z=74): 60 core electrons replaced by ECP (ecp60mdf equivalent),
    14 valence electrons per atom.  2-atom conventional cell → 28 valence
    electrons (14 occupied).

    Reference: runner_pyscf.py PBE/pob-TZVP-REV2 + ECP.
    """
    system = _build_w_bcc()
    result = _run_pbe_ecp(system)
    assert abs(result.energy - _PYSCF_REF_W_BCC) / 2 < 1e-3, (
        f"W bcc ECP energy {result.energy:.8f} Ha deviates from PySCF "
        f"{_PYSCF_REF_W_BCC:.8f} by "
        f"{abs(result.energy - _PYSCF_REF_W_BCC) / 2:.6f} Ha/atom "
        f"(limit 1e-3 Ha/atom)"
    )


# =========================================================================
# run_periodic_job fails closed on an ECP-bearing cell (2026-09)
# =========================================================================


def test_run_periodic_job_refuses_ecp_active_cell_on_routes_that_drop_the_ecp(tmp_path):
    """Before the k-point GDF drivers consumed the fields (2026-09-05) the
    runner resolved the pob-TZVP-rev2 ECP blocks, set them on the options,
    and dispatched to a driver that read none of them: 188 electrons went
    into the 19-valence-electron silver basis with bare Z = 47 nuclei.
    Every route whose driver still works that way is refused before any
    SCF work, naming the elements and the route."""
    from vibeqc.periodic_runner import run_periodic_job

    system = _build_ag_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    with pytest.raises(NotImplementedError, match="ECP metadata is not supported by the BIPOLE"):
        run_periodic_job(
            system, basis, method="RKS", functional="pbe", kpoints=[2, 2, 2],
            jk_method="bipole", smearing_temperature=0.005,
            output=tmp_path / "ag_refused", max_iter=2,
        )
    with pytest.raises(NotImplementedError, match=r"does not apply a periodic ECP"):
        run_periodic_job(
            system, basis, method="RKS", functional="pbe", kpoints=[1, 1, 1],
            jk_method="rijcosx", smearing_temperature=0.005,
            output=tmp_path / "ag_refused_rijcosx", max_iter=2,
        )


def _build_agcl_rocksalt():
    """Rocksalt AgCl primitive cell (fcc, a = 5.55 Angstrom): one ECP atom
    (Ag, 28 core electrons removed) and one all-electron atom; 36 valence
    electrons, closed shell."""
    a = 5.55 * ANG
    lat = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
    atoms = [vq.Atom(47, [0.0, 0.0, 0.0]), vq.Atom(17, [a / 2, a / 2, a / 2])]
    return vq.PeriodicSystem(3, lat, atoms)


@pytest.mark.skip(
    reason="#715: RKS/PBE on the AgCl / pob-TZVP-rev2 cell allocates >80 GB in "
    "the periodic XC build and is killed before the first iteration; the RKS "
    "ECP route is pinned on NaCl / LANL2DZ below, the pob frame on RHF/UHF."
)
@pytest.mark.slow
def test_run_periodic_job_gdf_applies_the_pob_ecp(tmp_path):
    """The default k-point GDF route builds the ECP Hamiltonian: V_ne with
    Z_eff, the lattice V_ECP in every Hcore(k), the Z_eff ionic repulsion
    and the valence count. The ionic energy is the observable that the
    charged nuclear frame was used: it equals the Ewald energy of the
    Z_eff point charges, not of the bare nuclei."""
    from vibeqc.periodic_k_gdf import _periodic_ecp_context
    from vibeqc.periodic_runner import _resolve_ecp_data, run_periodic_job

    system = _build_agcl_rocksalt()
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    blocks, centers, eff_z, ncore = _resolve_ecp_data(system, basis)
    assert eff_z == pytest.approx([19.0, 17.0]) and ncore == 28

    result = run_periodic_job(
        system, basis, method="RKS", functional="pbe", kpoints=[1, 1, 1],
        jk_method="gdf", output=tmp_path / "agcl_gdf_ecp", max_iter=40,
    )
    opts = SimpleNamespace(
        ecp_primitive_blocks=blocks, ecp_home_centers=centers,
        ecp_effective_charges=eff_z, ecp_total_ncore=ncore,
    )
    *_, system_v = _periodic_ecp_context(opts, system, "test")
    e_nuc_eff = float(vq.ewald_nuclear_repulsion(system_v))
    e_nuc_bare = float(vq.ewald_nuclear_repulsion(system))
    assert abs(e_nuc_eff - e_nuc_bare) > 100.0
    assert float(result.e_nuclear) == pytest.approx(e_nuc_eff, abs=1e-8)
    # PBE valence energy of AgCl: silver's 19 valence electrons under the
    # Stuttgart ECP (about -146 Ha) plus all-electron chlorine (about
    # -460 Ha). An all-electron run in the valence basis sat near -1600 Ha.
    assert -640.0 < float(result.energy) < -580.0


def test_periodic_hf_options_carry_the_ecp_fields():
    """#88 item 3: every periodic HF request on an ECP basis died with an
    internal AttributeError because PeriodicRHFOptions had no ECP fields
    for the runner to assign. Both option structs carry the same four."""
    for options_type in (vq.PeriodicRHFOptions, vq.PeriodicKSOptions):
        opts = options_type()
        assert list(opts.ecp_primitive_blocks) == []
        assert list(opts.ecp_home_centers) == []
        assert list(opts.ecp_effective_charges) == []
        assert int(opts.ecp_total_ncore) == 0
        opts.ecp_home_centers = [[0.0, 0.0, 0.0]]
        opts.ecp_effective_charges = [19.0, 17.0]
        opts.ecp_total_ncore = 28
        assert [list(c) for c in opts.ecp_home_centers] == [[0.0, 0.0, 0.0]]
        assert list(opts.ecp_effective_charges) == [19.0, 17.0]
        assert opts.ecp_total_ncore == 28


@pytest.mark.slow
def test_run_periodic_job_gdf_applies_the_pob_ecp_for_rhf_and_uhf(tmp_path):
    """#88 items 1 and 3 on the HF routes: Gamma RHF and UHF on an ECP
    cell must reach the k-point GDF drivers (never the run_pbc_gdf_*
    fast paths, which read no ECP field). The observables: the ionic
    energy is the Z_eff Ewald energy on both; the UHF singlet lands on
    the RHF energy; and the alpha density holds 18 valence electrons,
    not the 32 an all-electron split of AgCl would give."""
    from vibeqc.periodic_k_gdf import _periodic_ecp_context
    from vibeqc.periodic_runner import _resolve_ecp_data, run_periodic_job

    system = _build_agcl_rocksalt()
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    blocks, centers, eff_z, ncore = _resolve_ecp_data(system, basis)
    opts = SimpleNamespace(
        ecp_primitive_blocks=blocks, ecp_home_centers=centers,
        ecp_effective_charges=eff_z, ecp_total_ncore=ncore,
    )
    *_, system_v = _periodic_ecp_context(opts, system, "test")
    e_nuc_eff = float(vq.ewald_nuclear_repulsion(system_v))

    rhf = run_periodic_job(
        system, basis, method="RHF", kpoints=[1, 1, 1], jk_method="gdf",
        output=tmp_path / "agcl_gdf_ecp_rhf", max_iter=60,
    )
    assert rhf.converged
    assert float(rhf.e_nuclear) == pytest.approx(e_nuc_eff, abs=1e-8)
    assert -700.0 < float(rhf.energy) < -500.0

    uhf = run_periodic_job(
        system, basis, method="UHF", kpoints=[1, 1, 1], jk_method="gdf",
        output=tmp_path / "agcl_gdf_ecp_uhf", max_iter=60,
    )
    assert uhf.converged
    assert float(uhf.e_nuclear) == pytest.approx(e_nuc_eff, abs=1e-8)
    # Not pinned: uhf.energy == rhf.energy. On this cell the SCF has several
    # closed-shell stationary points, so the two drivers land on different
    # ones. The NaCl / LANL2DZ test below pins RHF == UHF where the landscape
    # is simple. The specific energies once quoted here (SAD RHF -539.673,
    # Hcore RHF -539.899, UHF -540.128, measured 2026-09-06) predate #207 and
    # came from an ECP applied two radial powers too low, with the f
    # projector acting as the local potential; they are not a reference for
    # anything. Whether the guess dependence itself survives the corrected
    # operator is unmeasured (vibeqc#52).
    assert -700.0 < float(uhf.energy) < -500.0
    d_alpha = np.asarray(uhf.density_alpha[0])
    overlap = np.asarray(uhf.overlap[0])
    n_alpha = float(np.trace(d_alpha @ overlap).real)
    assert n_alpha == pytest.approx(18.0, abs=1e-6), (type(uhf), n_alpha)


def test_default_gamma_ecp_cell_dispatches_to_the_k_gdf_drivers(monkeypatch, tmp_path):
    """#88 items 1 and 2: with the default mesh (kpoints=None) the runner
    used to send UHF/UKS to the run_pbc_gdf_* fast paths and RHF/RKS to the
    legacy Gamma driver, none of which read an ECP field, so the run was
    all-electron in a valence basis. An ECP-bearing cell must reach the
    k-point GDF drivers at a one-point mesh. Pinned by interception, so
    no SCF runs."""
    import vibeqc.periodic_runner as pr
    from vibeqc.periodic_runner import run_periodic_job

    class _KDriverReached(Exception):
        pass

    def _k_driver(*args, **kwargs):
        meshes = [
            a for a in (*args, *kwargs.values())
            if isinstance(a, (tuple, list)) and len(a) == 3
            and all(isinstance(x, int) for x in a)
        ]
        raise _KDriverReached(meshes[0] if meshes else None)

    def _fast_path(*args, **kwargs):
        raise AssertionError("a Gamma fast path received an ECP-bearing cell")

    for name in (
        "run_krhf_periodic_gdf", "run_krks_periodic_gdf",
        "run_kuhf_periodic_gdf", "run_kuks_periodic_gdf",
    ):
        monkeypatch.setattr(pr, name, _k_driver)
    for name in (
        "run_pbc_gdf_rhf", "run_pbc_gdf_uhf", "run_pbc_gdf_uks",
        "run_rhf_periodic_gamma_gdf",
    ):
        monkeypatch.setattr(pr, name, _fast_path)

    system = _build_agcl_rocksalt()
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    for method, functional in (("RHF", None), ("RKS", "pbe"), ("UHF", None), ("UKS", "pbe")):
        kwargs = {
            "method": method, "kpoints": None, "jk_method": "gdf",
            "output": tmp_path / f"dispatch_{method}", "progress": False,
        }
        if functional:
            kwargs["functional"] = functional
        with pytest.raises(_KDriverReached) as info:
            run_periodic_job(system, basis, **kwargs)
        assert tuple(info.value.args[0]) == (1, 1, 1), (method, info.value.args)


@pytest.mark.parametrize(
    "driver_name",
    ["run_pbc_gdf_rhf", "run_pbc_gdf_rks", "run_pbc_gdf_uhf", "run_pbc_gdf_uks",
     "run_rhf_periodic_gamma_gdf"],
)
def test_gamma_fast_path_drivers_refuse_ecp_options_when_called_directly(driver_name):
    """The Gamma fast paths build Hcore from bare Z and fill the physical
    count; a direct caller who sets ECP options on them must be refused
    before any work, not silently run all-electron (#88)."""
    from vibeqc import pbc_gdf
    from vibeqc import periodic_rhf_gdf as legacy

    driver = getattr(pbc_gdf, driver_name, None) or getattr(legacy, driver_name)
    opts = vq.PeriodicRHFOptions()
    opts.ecp_total_ncore = 28
    with pytest.raises(NotImplementedError, match="does not apply a periodic ECP"):
        driver(None, None, opts)


def test_legacy_property_rebuild_stays_disabled_for_ecp_cells():
    """The legacy bare-Z property rebuild remains gated. GDF uses the
    independent accepted-state adapter tested by the producer cases below."""
    from vibeqc.periodic_runner import (
        PeriodicJKMethod,
        _qvf_periodic_property_payload_supported,
    )

    assert _qvf_periodic_property_payload_supported(
        PeriodicJKMethod.GDF, uses_external_xc=False
    )
    assert not _qvf_periodic_property_payload_supported(
        PeriodicJKMethod.GDF, uses_external_xc=False, ecp_active=True
    )


def _build_nacl_rocksalt():
    a = 5.64 * 1.8897261246257702
    lattice = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
    return vq.PeriodicSystem(
        3, lattice, [vq.Atom(11, [0.0, 0.0, 0.0]), vq.Atom(17, [a / 2, a / 2, a / 2])], 0, 1
    )


@pytest.mark.parametrize(
    "method, functional, n_valence_alpha",
    [("RHF", None, None), ("UHF", None, 4.0), ("RKS", "pbe", None)],
)
def test_sidecar_ecp_basis_runs_on_the_periodic_gdf_route(
    tmp_path, method, functional, n_valence_alpha
):
    """A basis with a plain .ecp sidecar (LANL2DZ: ECPs on Na and Cl) runs
    on the k-point GDF route with its ECP (#88). Until 2026-09 the periodic
    resolver knew only the pob CRYSTAL records, so this request was refused
    as an ECP-paired basis without an ECP. Pinned: the ionic energy is the
    Z_eff Ewald energy (1 and 7, not 11 and 17), the density holds the 8
    valence electrons (28 all-electron), and RKS runs in ordinary memory."""
    from vibeqc.periodic_k_gdf import _periodic_ecp_context
    from vibeqc.periodic_runner import _resolve_ecp_data, run_periodic_job

    system = _build_nacl_rocksalt()
    # Choose the equivalent nearest Cl image so the home-atom pair selector
    # includes the Na-Cl bond (its documented cutoff is 8 bohr).
    system = vq.PeriodicSystem(3, np.asarray(system.lattice), [
        system.unit_cell[0],
        vq.Atom(17, list(np.asarray(system.unit_cell[1].xyz)
                        - np.asarray(system.lattice)[:, 0])),
    ], 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    blocks, centers, eff_z, ncore = _resolve_ecp_data(system, basis)
    assert len(blocks) == 2 and eff_z == pytest.approx([1.0, 7.0]) and ncore == 20
    opts = SimpleNamespace(
        ecp_primitive_blocks=blocks, ecp_home_centers=centers,
        ecp_effective_charges=eff_z, ecp_total_ncore=ncore,
    )
    *_, system_v = _periodic_ecp_context(opts, system, "test")
    e_nuc_eff = float(vq.ewald_nuclear_repulsion(system_v))
    assert abs(e_nuc_eff - float(vq.ewald_nuclear_repulsion(system))) > 50.0

    kwargs = {
        "method": method, "kpoints": [1, 1, 1], "jk_method": "gdf",
        "aux_basis": "def2-svp-jk", "output": tmp_path / f"nacl_lanl_{method}",
        "max_iter": 80, "progress": False, "output_qvf": True,
        "coop_cohp": True,
    }
    if functional:
        kwargs["functional"] = functional
    result = run_periodic_job(system, basis, **kwargs)
    assert result.converged
    assert float(result.e_nuclear) == pytest.approx(e_nuc_eff, abs=1e-8)
    overlap = np.asarray(result.overlap[0])
    if n_valence_alpha is None:
        density = np.asarray(result.density[0])
        assert float(np.trace(density @ overlap).real) == pytest.approx(8.0, abs=1e-6)
    else:
        d_alpha = np.asarray(result.density_alpha[0])
        assert float(np.trace(d_alpha @ overlap).real) == pytest.approx(
            n_valence_alpha, abs=1e-6
        )
    # Valence-only energy scale: Na+ core + Cl 7e under their ECPs. An
    # all-electron NaCl run in this basis would sit near -620 Ha.
    assert -4.0 < float(result.energy) < -2.0

    import json
    import zipfile
    from vibeqc.bands import _HARTREE_TO_EV, _shell_to_atom
    from vibeqc.output.formats.qvf import validate_qvf

    path = tmp_path / f"nacl_lanl_{method}.qvf"
    validation = validate_qvf(path)
    assert validation['valid'], validation.get('errors')
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        sections = {s['kind']: s for s in manifest['sections']}
        assert {'dos.total', 'dos.projected', 'dos.coop', 'dos.cohp'} <= sections.keys()
        integrated = np.frombuffer(archive.read('dos/cohp_integrated.bin'), dtype='<f8')
    ao_atoms = _shell_to_atom(basis)
    ab = np.ix_(np.flatnonzero(ao_atoms == 0), np.flatnonzero(ao_atoms == 1))
    suffixes = ('_alpha', '_beta') if method == 'UHF' else ('',)
    expected = []
    for suffix in suffixes:
        density = np.asarray(getattr(result, 'density'+suffix)[0])
        fock = np.asarray(getattr(result, 'fock'+suffix)[0])
        expected.append(-np.sum(density.T[ab] * fock[ab]).real * _HARTREE_TO_EV)
    np.testing.assert_allclose(integrated, expected, atol=1e-10, rtol=0)


def test_sidecar_ecp_cell_uhf_singlet_matches_rhf(tmp_path):
    """On a cell with one closed-shell SCF solution the two k-point GDF
    drivers agree on the ECP energy to machine precision (NaCl / LANL2DZ:
    measured 2026-09-06, |E_UHF - E_RHF| = 4e-15 Ha). This is the pin the
    AgCl cell cannot carry (several stationary points, see above)."""
    from vibeqc.periodic_runner import run_periodic_job

    system = _build_nacl_rocksalt()
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    energies = {}
    for method in ("RHF", "UHF"):
        result = run_periodic_job(
            system, basis, method=method, kpoints=[1, 1, 1], jk_method="gdf",
            aux_basis="def2-svp-jk", output=tmp_path / f"nacl_lanl_pair_{method}",
            max_iter=80, progress=False, output_qvf=False,
        )
        assert result.converged
        energies[method] = float(result.energy)
    assert energies["UHF"] == pytest.approx(energies["RHF"], abs=1e-8)
    assert -4.0 < energies["RHF"] < -2.0


def test_pob_heavy_sources_are_bundled_so_ecp_resolution_needs_no_network():
    """The Rb-Po and La-Lu CRYSTAL records ship under sources/; the runtime
    resolution must find silver locally instead of fetching."""
    from vibeqc.periodic_runner import _bundled_pob_source_atoms

    atoms = {int(a.Z): a for a in _bundled_pob_source_atoms("pob-tzvp-rev2")}
    assert 47 in atoms and atoms[47].has_ecp and atoms[47].ecp is not None
    assert 79 in atoms and atoms[79].has_ecp


def test_pob_sidecar_matches_the_periodic_bridge_arrays():
    """One ECP for molecular and periodic runs: the shipped sidecar parses
    to exactly the primitive arrays the CRYSTAL bridge feeds libecpint."""
    from vibeqc.basis_crystal import crystal_ecp_to_libecpint_arrays
    from vibeqc.ecp_metadata import parse_inline_ecp_sidecar, sidecar_path_for
    from vibeqc.periodic_runner import _bundled_pob_source_atoms

    records = parse_inline_ecp_sidecar(sidecar_path_for("pob-tzvp-rev2"))
    checked = 0
    for atom in _bundled_pob_source_atoms("pob-tzvp-rev2"):
        if not (atom.has_ecp and atom.ecp is not None):
            continue
        arrays = crystal_ecp_to_libecpint_arrays(atom.ecp)
        rec = records[int(atom.Z)]
        assert rec.header.ncore == int(atom.Z) - round(atom.ecp.znuc)
        # Compare every primitive, the zero local placeholder included. The
        # exemption that used to stand here hid #207: the bridge emitted no
        # local channel at all, so libecpint promoted the f projector to the
        # local potential on the periodic route while the sidecar-fed
        # molecular route kept it projected -- two operators, one record.
        bridge = sorted(zip(arrays.ams, arrays.ns, arrays.exponents, arrays.coefficients))
        sidecar = sorted(zip(rec.ams, rec.ns, rec.exponents, rec.coefficients))
        assert bridge == sidecar, f"Z={atom.Z}"
        checked += 1
    assert checked == 46


# =========================================================================
# Smoke test: ECP auto-attach via run_periodic_job should not crash
# =========================================================================


def test_ecp_auto_attach_no_crash_on_light_atom():
    """ECP auto-attach is a no-op on light-element pob-TZVP-REV2 systems.

    NaCl rocksalt (Na: Z=11, Cl: Z=17) has no ECP atoms in pob-TZVP-REV2,
    so the _resolve_ecp_data helper should return empty lists and the SCF
    should run all-electron as before.
    """
    a = 5.64 * ANG  # experimental NaCl lattice constant
    lat = np.eye(3) * a
    atoms = [
        vq.Atom(11, [0.0, 0.0, 0.0]),  # Na
        vq.Atom(17, [a / 2, a / 2, a / 2]),  # Cl
    ]
    system = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")

    from vibeqc.periodic_runner import _resolve_ecp_data

    ecp_blocks, _ecp_centers, _eff_z, total_ncore = _resolve_ecp_data(system, basis)
    # NaCl has no ECP-bearing atoms in pob-TZVP-REV2 (Na/Cl are period-3,
    # all-electron in this basis).
    assert len(ecp_blocks) == 0
    assert total_ncore == 0


@pytest.mark.parametrize("route", [
    "native_gamma", "native_rhf", "native_rks", "ewald_rhf", "ewald_uhf",
    "gdf_rhf", "gdf_uhf",
])
@pytest.mark.parametrize("guess", [vq.InitialGuess.AUTO, vq.InitialGuess.MINAO])
def test_direct_all_electron_charge_metadata_preserves_guess_and_energy(route, guess):
    """Charge-only POB records must behave like absent ECP metadata at SCF."""
    import importlib

    system = vq.PeriodicSystem(3, np.eye(3) * 20., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = vq.monkhorst_pack(system, [3, 1, 1])
    if route == "native_gamma":
        driver = vq._vibeqc_core.run_rhf_periodic_gamma
        options_cls = vq.PeriodicRHFOptions
    elif route.startswith("native_"):
        family = route.removeprefix("native_")
        driver = getattr(vq._vibeqc_core, f"run_{family}_periodic")
        options_cls = vq.PeriodicKSOptions if family == "rks" else vq.PeriodicSCFOptions
    elif route.startswith("ewald_"):
        family = route.removeprefix("ewald_")
        module = importlib.import_module(f"vibeqc.periodic_{family}_multi_k_ewald")
        driver = getattr(module, f"run_{family}_periodic_multi_k_ewald3d")
        options_cls = vq.PeriodicRHFOptions
    else:
        module = importlib.import_module("vibeqc.periodic_k_gdf")
        driver = getattr(module, f"run_k{route.removeprefix('gdf_')}_periodic_gdf")
        options_cls = vq.PeriodicRHFOptions

    results = []
    for charges in ([], [1., 1.]):
        opts = options_cls()
        opts.initial_guess = guess
        opts.ecp_effective_charges = charges
        opts.max_iter = 60
        opts.conv_tol_energy = 1e-10
        opts.lattice_opts.cutoff_bohr = 5.
        opts.lattice_opts.nuclear_cutoff_bohr = 5.
        if route == "native_rks":
            opts.functional = "lda"
        args = (system, basis, opts) if route == "native_gamma" else (system, basis, mesh, opts)
        # Both metadata variants use the same finite reciprocal grid.
        kwargs = dict(rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.) if route.startswith("gdf_") else {}
        result = driver(*args, **kwargs)
        assert result.converged
        assert result.guess_selection.requested == guess
        assert result.guess_selection.effective == (
            vq.InitialGuess.SAD if guess == vq.InitialGuess.AUTO else guess
        )
        assert list(opts.ecp_effective_charges) == charges
        results.append(result)
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-10)


@pytest.mark.parametrize("route", ["gamma", "rhf", "rks"])
@pytest.mark.parametrize("fault", [
    "reduced", "reordered", "size", "roundoff", "nonfinite", "core", "centers",
])
def test_native_periodic_charge_identity_does_not_mask_incomplete_ecp(route, fault):
    system = vq.PeriodicSystem(3, np.eye(3) * 20., [
        vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = {"gamma": vq.PeriodicRHFOptions, "rhf": vq.PeriodicSCFOptions,
            "rks": vq.PeriodicKSOptions}[route]()
    opts.ecp_effective_charges = {
        "reduced": [1., 1.], "reordered": [1., 3.], "size": [3.],
        "roundoff": [3.000000001, 1.], "nonfinite": [3., float("nan")],
    }.get(fault, [3., 1.])
    if fault == "core":
        opts.ecp_total_ncore = 2
    elif fault == "centers":
        opts.ecp_home_centers = [[0., 0., 0.]]
    with pytest.raises(ValueError, match="non-empty ecp_primitive_blocks"):
        if route == "gamma":
            vq._vibeqc_core.run_rhf_periodic_gamma(system, basis, opts)
        else:
            driver = getattr(vq._vibeqc_core, f"run_{route}_periodic")
            driver(system, basis, vq.monkhorst_pack(system, [3, 1, 1]), opts)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("guess", [vq.InitialGuess.AUTO, vq.InitialGuess.MINAO])
def test_bipole_charge_metadata_preserves_direct_guess_and_energy(method, mesh, guess):
    system = vq.PeriodicSystem(3, np.eye(3) * 10., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, mesh)
    driver = getattr(vq, f"run_pbc_bipole_{method}")
    results = []
    for charges in ([1., 1.], []):
        opts = vq.PeriodicKSOptions() if method.endswith("ks") else vq.PeriodicRHFOptions()
        opts.initial_guess = guess
        opts.ecp_effective_charges = charges
        opts.max_iter = 60
        opts.conv_tol_energy = 1e-10
        opts.lattice_opts.cutoff_bohr = 6.
        opts.lattice_opts.nuclear_cutoff_bohr = 6.
        if method.endswith("ks"):
            opts.functional = "lda"
        result = driver(system, basis, kmesh, opts, progress=False, verbose=0)
        assert result.converged
        assert result.guess_selection.requested == guess
        assert result.guess_selection.effective == (
            vq.InitialGuess.SAD if guess == vq.InitialGuess.AUTO else guess
        )
        assert list(opts.ecp_effective_charges) == charges
        results.append(result)
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-10)


_ALL_ELECTRON_GDF_ROUTES = [
    "gdf_rhf", "gdf_uhf", "gdf_rks", "gdf_uks", "legacy_gdf", "rohf_gdf",
    "slab_gdf", "rijcosx_rhf", "rijcosx_uhf", "rijcosx_rks", "rijcosx_uks",
]


def _all_electron_gdf_call(route, charges, *, with_basis):
    import importlib

    system = vq.PeriodicSystem(2 if route == "slab_gdf" else 3,
        np.eye(3) * 20., [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g") if with_basis else None
    opts = vq.PeriodicKSOptions() if route.endswith("ks") else vq.PeriodicRHFOptions()
    opts.ecp_effective_charges = charges
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-10
    opts.lattice_opts.cutoff_bohr = 4.
    opts.lattice_opts.nuclear_cutoff_bohr = 4.
    if route == "slab_gdf":
        opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    kwargs = {"progress": False}
    if route.endswith("ks"):
        kwargs["functional"] = "lda"
    if route.startswith("rijcosx_"):
        driver = getattr(importlib.import_module("vibeqc.periodic_rijcosx"),
                         f"run_periodic_{route}")
    elif route.startswith("gdf_"):
        driver = getattr(vq, f"run_pbc_{route}")
    elif route == "legacy_gdf":
        driver = importlib.import_module("vibeqc.periodic_rhf_gdf").run_rhf_periodic_gamma_gdf
    elif route == "rohf_gdf":
        driver = importlib.import_module("vibeqc.periodic_rohf_gdf").run_krohf_periodic_gdf
    else:
        driver = importlib.import_module("vibeqc.periodic_rhf_gdf")._run_krhf_periodic_slab_gdf
    args = (system, basis, (1, 1, 1), opts) if route in ("rohf_gdf", "slab_gdf") else (system, basis, opts)
    return driver, args, kwargs, opts


@pytest.mark.parametrize("route", _ALL_ELECTRON_GDF_ROUTES)
@pytest.mark.parametrize("charges", [[0., 1.], [1.], [1., float("nan")], [1.000000001, 1.]])
def test_all_electron_gdf_routes_reject_orphan_charges_before_setup(route, charges):
    driver, args, kwargs, _ = _all_electron_gdf_call(route, charges, with_basis=False)
    with pytest.raises(NotImplementedError, match="does not apply a periodic ECP"):
        driver(*args, **kwargs)


@pytest.mark.parametrize("route", [r for r in _ALL_ELECTRON_GDF_ROUTES if r != "slab_gdf"])
def test_all_electron_gdf_routes_preserve_charge_only_auto(route):
    results = []
    for charges in ([1., 1.], []):
        driver, args, kwargs, opts = _all_electron_gdf_call(route, charges, with_basis=True)
        result = driver(*args, **kwargs)
        assert result.converged
        assert result.guess_selection.requested == vq.InitialGuess.AUTO
        assert result.guess_selection.effective == vq.InitialGuess.SAD
        assert list(opts.ecp_effective_charges) == charges
        results.append(result)
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-10)


def test_private_slab_gdf_physical_charges_reach_integral_setup(monkeypatch):
    import vibeqc.periodic_rhf_gdf as module

    class ReachedIntegrals(Exception):
        pass

    def stop(*_args):
        raise ReachedIntegrals

    monkeypatch.setattr(module, "compute_overlap_lattice", stop)
    driver, args, kwargs, opts = _all_electron_gdf_call(
        "slab_gdf", [1., 1.], with_basis=False)
    with pytest.raises(ReachedIntegrals):
        driver(*args, **kwargs)
    assert list(opts.ecp_effective_charges) == [1., 1.]


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_gamma_rijcosx_rejects_ecp_core_before_setup(method):
    driver, args, kwargs, opts = _all_electron_gdf_call(
        f"rijcosx_{method}", [1., 1.], with_basis=False)
    opts.ecp_total_ncore = 2
    with pytest.raises(NotImplementedError, match="does not apply a periodic ECP"):
        driver(*args, **kwargs)


def _gaussian_projector_block(*, nonlocal_channel=False):
    block = vq.ECPPrimitiveBlock()
    block.n_primitive = 2 if nonlocal_channel else 1
    block.exponents = [1.25, 0.9] if nonlocal_channel else [1.25]
    block.coefficients = [-2.0, 0.7] if nonlocal_channel else [-2.0]
    block.ams = [1, 0] if nonlocal_channel else [0]
    block.ns = [2] * block.n_primitive
    return block


@pytest.mark.parametrize("case_index", range(8))
def test_displaced_ecp_projectors_against_independent_matrices(case_index):
    """Pin #758's generated angular precision and amplified radial drops."""
    import json
    from pathlib import Path

    from vibeqc.periodic_runner import _resolve_ecp_data

    core = vq._vibeqc_core
    directory = Path(__file__).parent / "data/periodic_gdf"
    specification = json.loads((directory / "nacl_ecp_projectors.json").read_text())
    atoms = specification["atoms"]
    case = specification["cases"][case_index]

    def make_basis(shifts):
        centers = [np.asarray(center) + shift for shift in shifts for _, center in atoms]
        molecule = vq.Molecule([
            vq.Atom(11 if atoms[index % 2][0] == "Na" else 17, center)
            for index, center in enumerate(centers)
        ])
        shells, permutation = [], []
        offset = 0
        for index, center in enumerate(centers):
            for record in specification["basis"][atoms[index % 2][0]]:
                angular = record[0]
                primitives = np.asarray(record[1:])
                for column in range(1, primitives.shape[1]):
                    shells.append(core.ShellInfo(
                        index, angular, True, primitives[:, 0].tolist(),
                        primitives[:, column].tolist(), center,
                    ))
                    order = [1, 2, 0] if angular == 1 else list(range(2*angular + 1))
                    permutation.extend(offset + component for component in order)
                    offset += len(order)
        return core.BasisSet(molecule, shells, "lanl2dz", False), permutation

    home, permutation = make_basis([np.zeros(3)])
    system = vq.PeriodicSystem(
        3, np.asarray(specification["lattice"]).T,
        [vq.Atom(11 if symbol == "Na" else 17, center) for symbol, center in atoms],
    )
    blocks, _, _, _ = _resolve_ecp_data(system, home)
    union, _ = make_basis([np.zeros(3), np.asarray(case["ket_shift"])])
    actual = core.compute_ecp_matrix_from_primitives(
        union, case["projector_center"], [blocks[case["projector_atom"]]],
    )
    with np.load(directory / "nacl_ecp_projectors.npz") as reference:
        expected = reference[case["name"]][np.ix_(permutation, permutation)]
    np.testing.assert_allclose(np.asarray(actual)[:16, 16:], expected, atol=1e-11, rtol=0)


def test_periodic_ecp_home_bra_shifted_ket_gaussian_oracle():
    """A local Gaussian projector has an elementary three-Gaussian integral."""
    core = vq._vibeqc_core
    system = vq.PeriodicSystem(3, np.eye(3)*4.0, [vq.Atom(1, [0.0]*3)])
    basis = core.BasisSet(system.unit_cell_molecule(), [
        core.ShellInfo(0, 0, True, [1.0], [1.0], [0.0]*3),
    ], "ecp-gaussian-oracle", coefficients_pre_normalized=True)
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = options.nuclear_cutoff_bohr = 4.01
    result = core.compute_ecp_lattice_from_primitives(
        basis, system, options, [[0.0]*3], [_gaussian_projector_block()],
    )
    images = [np.asarray(c.r_cart) for c in result.cells]
    expected = []
    for cell in result.cells:
        g = np.asarray(cell.r_cart)
        expected.append(sum(
            -2.0*(np.pi/3.25)**1.5*np.exp(
                -g@g + (g + 1.25*t)@(g + 1.25*t)/3.25 - 1.25*(t@t)
            ) for t in images
        ))
    actual = np.asarray(result.blocks)[:, 0, 0]
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    # The old both-orbitals-shifted builder gave -1.9007794363 on every
    # nearest cell, against the analytic -0.00005879184825 Ha.
    assert np.max(np.abs(actual[1:])) < 6e-5
    for k in ([0.0]*3, [0.23, 0.17, -0.08]):
        phase = np.exp(1j*np.asarray(images)@k)
        np.testing.assert_allclose(phase@actual, phase@expected, atol=3e-12, rtol=0)


@pytest.mark.parametrize("angular_momentum", [0, 1, 2])
def test_periodic_ecp_matches_molecular_cross_blocks(angular_momentum):
    """Nonlocal projectors and spherical transforms use <home|U|shifted>."""
    core = vq._vibeqc_core
    centers = [np.array([0.2, 0.3, 0.0]), np.array([0.8, -0.1, 0.4])]
    system = vq.PeriodicSystem(3, np.eye(3)*5.0, [vq.Atom(1, x) for x in centers])

    def make_basis(positions):
        molecule = vq.Molecule([vq.Atom(1, x) for x in positions])
        return core.BasisSet(molecule, [
            core.ShellInfo(i, angular_momentum, True, [0.8 + 0.2*(i % 2)],
                           [1.0], center)
            for i, center in enumerate(positions)
        ], "ecp-cross-block-oracle", coefficients_pre_normalized=True)

    basis = make_basis(centers)
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = options.nuclear_cutoff_bohr = 5.01
    result = core.compute_ecp_lattice_from_primitives(
        basis, system, options, [centers[0]],
        [_gaussian_projector_block(nonlocal_channel=True)],
    )
    image_centers = [centers[0] + np.asarray(c.r_cart) for c in result.cells]
    for cell, block in zip(result.cells, result.blocks):
        g = np.asarray(cell.r_cart)
        # A molecular union basis gives an independent cross-block readout;
        # the native periodic code must not extract its shifted diagonal.
        union = make_basis(centers + [x + g for x in centers])
        molecular = core.compute_ecp_matrix_from_primitives(
            union, np.asarray(image_centers).ravel().tolist(),
            [_gaussian_projector_block(nonlocal_channel=True) for _ in image_centers],
        )
        np.testing.assert_allclose(
            block, np.asarray(molecular)[:basis.nbasis, basis.nbasis:],
            atol=2e-10, rtol=0,
        )


def _ecp_gradient_case(coords, angular):
    system = vq.PeriodicSystem(3, np.eye(3) * 7., [vq.Atom(1, p) for p in coords])
    basis = vq._vibeqc_core.BasisSet(system.unit_cell_molecule(), [
        vq._vibeqc_core.ShellInfo(a, angular, True, [1.], [1.], p)
        for a, p in enumerate(coords)
    ], 'analytic', coefficients_pre_normalized=False)
    block = vq._vibeqc_core.ECPPrimitiveBlock()
    block.n_primitive = 2
    block.exponents = [1.25, .8]
    block.coefficients = [-2., .7]
    block.ams = [1, 0]
    block.ns = [2, 2]
    return system, basis, [coords[0]], [block]


@pytest.mark.parametrize('angular', [0, 1, 2])
@pytest.mark.parametrize('threads', [1, 2])
def test_periodic_ecp_fixed_density_gradient(angular, threads):
    previous_threads = vq.get_num_threads()
    vq.set_num_threads(threads)
    try:
        coords = [[.2, .3, .4], [2.1, 1.2, .8]]
        system, basis, centres, blocks = _ecp_gradient_case(coords, angular)
        options = vq._vibeqc_core.LatticeSumOptions()
        options.cutoff_bohr = 8.
        options.nuclear_cutoff_bohr = 14.
        cells = vq._vibeqc_core.direct_lattice_cells(system, options.cutoff_bohr)
        rng = np.random.default_rng(739)
        d = rng.normal(size=(basis.nbasis, basis.nbasis))
        d = d @ d.T * .01
        density = vq._vibeqc_core.make_lattice_matrix_set(basis.nbasis, cells,
            [d * np.exp(-.1 * np.dot(c.r_cart, c.r_cart)) for c in cells])
        actual = np.asarray(vq._vibeqc_core.ecp_lattice_gradient_contribution_from_primitives(
            basis, system, density, options, centres, blocks))

        def energy(positions):
            system, basis, centres, blocks = _ecp_gradient_case(positions, angular)
            potential = vq._vibeqc_core.compute_ecp_lattice_from_primitives(basis, system, options, centres, blocks)
            assert len(potential.cells) == len(density.cells)
            return sum(np.sum(np.asarray(d) * np.asarray(v))
                       for d, v in zip(density.blocks, potential.blocks))

        step = 2e-5
        expected = np.empty((2, 3))
        for a in range(2):
            for axis in range(3):
                plus, minus = np.array(coords), np.array(coords)
                plus[a, axis] += step
                minus[a, axis] -= step
                expected[a, axis] = (energy(plus.tolist()) - energy(minus.tolist())) / (2 * step)
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)
        np.testing.assert_allclose(actual.sum(axis=0), 0., atol=2e-12, rtol=0.)

    finally:
        vq.set_num_threads(previous_threads)


@pytest.mark.slow
def test_periodic_ecp_gdf_force_differentiates_the_accepted_valence_energy():
    """Full #739 force: distorted ECP crystal, two finite-difference steps."""
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf, _periodic_ecp_context
    from vibeqc.periodic_runner import _resolve_ecp_data

    original = _build_nacl_rocksalt()
    coordinates = np.array([atom.xyz for atom in original.unit_cell])
    coordinates[1, 2] += .17  # avoid a symmetry-enforced zero-force witness

    def run(displacement, gradient=False):
        xyz = coordinates.copy()
        xyz[1, 2] += displacement
        system = vq.PeriodicSystem(3, np.asarray(original.lattice), [
            vq.Atom(atom.Z, list(position))
            for atom, position in zip(original.unit_cell, xyz)
        ], 0, 1)
        basis = vq.BasisSet(system.unit_cell_molecule(), 'lanl2dz')
        blocks, centers, charges, ncore = _resolve_ecp_data(system, basis)
        options = vq.PeriodicRHFOptions()
        options.ecp_primitive_blocks = blocks
        options.ecp_home_centers = centers
        options.ecp_effective_charges = charges
        options.ecp_total_ncore = ncore
        options.initial_guess = vq.InitialGuess.HCORE
        options.conv_tol_energy = 1e-11
        options.conv_tol_grad = 1e-8
        options.max_iter = 100
        result = run_krhf_periodic_gdf(
            system, basis, (1, 1, 1), options,
            aux_basis='def2-svp-jk', compute_gradient=gradient, progress=False,
        )
        assert result.converged
        ionic_system = _periodic_ecp_context(options, system, 'test')[-1]
        assert result.e_nuclear == pytest.approx(vq.ewald_nuclear_repulsion(ionic_system), abs=1e-9)
        count = sum(w * np.trace(d @ s).real for w, d, s in zip(
            result.kpoint_weights, result.density, result.overlap,
        ))
        assert count == pytest.approx(8., abs=1e-8)
        return result

    accepted = run(0., gradient=True)
    derivative = np.asarray(accepted.gradient)
    assert abs(derivative[1, 2]) > 1e-5
    np.testing.assert_allclose(derivative.sum(axis=0), 0., atol=1e-7, rtol=0)
    finite_differences = []
    for step in (1e-3, 5e-4):
        finite_differences.append((run(step).energy - run(-step).energy) / (2*step))
    extrapolated = (4*finite_differences[1] - finite_differences[0])/3
    assert derivative[1, 2] == pytest.approx(extrapolated, abs=2e-6)
