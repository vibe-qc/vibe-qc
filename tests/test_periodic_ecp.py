"""Periodic ECP regression tests: PBE / pob-TZVP-REV2 on 4d/5d metals.

Phase 14h — validates the inline-primitive ECP path (compute_ecp_lattice_from_primitives)
against PySCF.pbc with the same geometry, basis, and ECP.

Target: ≤ 1 mHa/atom agreement with PySCF.pbc (subprocess runner, CLAUDE.md §10).

Systems:
  - Ag fcc (Z=47, 4-atom conventional cell, ecp28mdf)
  - Au fcc (Z=79, 4-atom conventional cell, ecp60mdf)
  - W bcc  (Z=74, 2-atom conventional cell, ecp60mdf)

Validation procedure:
  1. Compute the system with vibe-qc at PBE/pob-TZVP-REV2 + ECP **on the GDF
     route** — ``run_krks_periodic_gdf``, which is where the periodic ECP
     operator landed (vibeqc#88) and the route vibe-qc#31 is about.
  2. Run the same system with PySCF.pbc out of process
     (``examples/regression/periodic_ecp_metal_reference.py``, CLAUDE.md §10).
  3. Check |E_vibe - E_pyscf| / n_atoms ≤ 1e-3 Ha.
  4. Pin the PySCF value as the inline reference.

W bcc is pinned. Ag fcc and Au fcc are not: see the skip reasons below.
The archived reference and the full statement of which convention was
matched to what live in ``tests/data/periodic_gdf/w_bcc_ecp_reference.json``
and that directory's README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
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
# PySCF reference energies
#
#   E_vibe   → total energy from vibe-qc PBE/pob-TZVP-REV2 + ECP, GDF route
#   E_pyscf  → total energy from PySCF.pbc PBE/pob-TZVP-REV2 + ECP
#   ΔE/atom  → |E_vibe - E_pyscf| / n_atoms_cell
#
# The target is ΔE/atom ≤ 1e-3 Ha (1 mHa).
#
# Both sides must be the *same* physical calculation, and three of the
# conventions involved differ by default between the two codes:
#
#   * sampling — the multi-k GDF drivers resolve a tuple ``kmesh`` with a
#     zero shift, i.e. Γ-centred, which is PySCF ``make_kpts``'s convention.
#     ``KPoints.monkhorst_pack``'s classical default half-shifts every even
#     axis instead; kpoints.py records the two being ~44 mHa apart at
#     (2,2,2), so ``_run_pbe_ecp`` passes the mesh as a plain tuple and the
#     test asserts the resulting k-points against the archived list.
#   * the ECP operator — PySCF's own ``def2-*`` library records for these
#     elements are a *different* published Stuttgart/Cologne parameter set
#     from the ones pob-TZVP-rev2 carries, so the reference is generated
#     from this repository's own ``.ecp`` sidecar.
#   * the energy — vibe-qc's ``result.energy`` is the plain SCF total energy
#     with ``result.free_energy = E - T·S`` kept separately, and PySCF's
#     ``e_tot``/``e_free`` split the same way. The pin is ``e_tot``.
#
# tests/data/periodic_gdf/README.md states the rest (fit, thresholds,
# quadrature) and how to regenerate.
# =========================================================================

_ECP_REFERENCE_DIR = Path(__file__).parent / "data/periodic_gdf"


def _w_bcc_reference(label: str) -> dict:
    """One archived PySCF.pbc W-bcc run, by sampling label."""
    record = json.loads((_ECP_REFERENCE_DIR / "w_bcc_ecp_reference.json").read_text())
    for run in record["runs"]:
        if run["label"] == label:
            assert run["converged"], f"archived {label} run did not converge"
            return run
    raise KeyError(f"no {label!r} run in w_bcc_ecp_reference.json")


# Ag fcc and Au fcc are 4-atom cells: 140 AOs and ~1800 def2-tzvp-jk
# auxiliary functions against W bcc's 70 and 454. The RSGDF fit alone took
# 14 minutes and 1.08 GB for W bcc at 2×2×2 on the development machine, and
# it grows roughly as naux²·nao², so these two are two to three orders of
# magnitude more expensive. They are left unpinned rather than guessed;
# examples/regression/periodic_ecp_metal_reference.py already builds both
# (`--system ag-fcc` / `--system au-fcc`) for whoever has the machine time.
_PYSCF_REF_AG_FCC = None
_PYSCF_REF_AU_FCC = None

# PySCF 2.14.0, W bcc, PBE/pob-TZVP-REV2 + ECP (60-electron core, Z_eff=14),
# 2-atom conventional cell, Γ, Fermi kT = 0.005 Ha, RSGDF ω = 0.4.
# Archived with its full provenance in
# tests/data/periodic_gdf/w_bcc_ecp_reference.{json,system}.
_PYSCF_REF_W_BCC = -135.21491246495432


# =========================================================================
# Helper: run a periodic PBE SCF with pob-TZVP-REV2 and check ECP wiring
# =========================================================================


def _run_pbe_ecp(system, kmesh=(2, 2, 2)):
    """Run PBE / pob-TZVP-REV2 on a periodic system with ECP auto-attach.

    Dispatches to ``run_krks_periodic_gdf`` — the k-point GDF driver that
    consumes the inline ECP primitives (vibeqc#88), and the route
    vibe-qc#31 asks for a cross-code pin on. ``vq.run_rks_periodic``, which
    this helper used to call, is the DIRECT_TRUNCATED multi-k driver; it
    honours the ECP fields too, so it was not wrong, but a reference pinned
    against it would have left the GDF route exactly as unchecked as before.

    ``kmesh`` is handed over as a **plain tuple** on purpose. The multi-k GDF
    drivers resolve a tuple through ``_mp_native(system, mesh, [0, 0, 0],
    False)`` — a zero shift, i.e. the Γ-centred sampling PySCF's
    ``cell.make_kpts`` produces by default. Building a
    ``KPoints.monkhorst_pack`` object instead would take the *classical* MP
    convention, half-shifting every even axis onto a different set of
    k-points; kpoints.py measures the two ~44 mHa apart on MgO/STO-3G at
    (2,2,2), which is three orders of magnitude outside this file's target.
    """
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")

    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-7
    opts.smearing_temperature = 0.005  # metals need smearing
    opts.use_diis = True

    # Attach ECP data from pob-TZVP-REV2 inline CRYSTAL ECP blocks. The
    # bundled records are what run_periodic_job resolves first; reading them
    # keeps the test offline and out of the package tree (#393).
    from vibeqc.basis_crystal import build_periodic_ecp_data
    from vibeqc.periodic_runner import _bundled_pob_source_atoms

    atom_list = _bundled_pob_source_atoms("pob-tzvp-rev2")
    assert atom_list, "pob-tzvp-rev2 resolved no atoms; the ECP would be dropped"
    ecp_blocks, ecp_centers, eff_z, total_ncore = build_periodic_ecp_data(
        system, atom_list
    )
    assert ecp_blocks, "no ECP blocks resolved; this would run all-electron"
    opts.ecp_primitive_blocks = ecp_blocks
    opts.ecp_home_centers = ecp_centers
    opts.ecp_effective_charges = eff_z
    opts.ecp_total_ncore = total_ncore

    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    result = run_krks_periodic_gdf(
        system, basis, tuple(int(n) for n in kmesh), opts, functional="pbe"
    )

    # Sanity: total energy should be strongly negative for all three metals.
    # For Ag fcc (4 atoms), E_total ~ -585 Ha at PBE/ECP (approx).
    # For Au fcc (4 atoms), E_total ~ -540 Ha at PBE/ECP (approx).
    # For W bcc (2 atoms), E_total ~ -135 Ha at PBE/ECP (PySCF.pbc, pinned).
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
@pytest.mark.skip(
    reason="no PySCF.pbc reference generated yet (see _PYSCF_REF_AG_FCC): the "
    "4-atom cell's RSGDF fit is orders of magnitude beyond W bcc's. Generate "
    "with examples/regression/periodic_ecp_metal_reference.py --system ag-fcc"
)
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
@pytest.mark.skip(
    reason="no PySCF.pbc reference generated yet (see _PYSCF_REF_AU_FCC): the "
    "4-atom cell's RSGDF fit is orders of magnitude beyond W bcc's. Generate "
    "with examples/regression/periodic_ecp_metal_reference.py --system au-fcc"
)
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


def _resolve_pob_ecp(system, elements=None):
    """Resolve pob-TZVP-REV2 ECP primitive blocks/centres for ``system``.

    Mirrors the resolution ``_run_pbe_ecp`` performs before attaching ECP
    data to ``PeriodicKSOptions``, exposed standalone so a test can call
    ``compute_ecp_lattice_from_primitives`` directly without running an SCF.
    """
    from vibeqc.basis_crystal import build_periodic_ecp_data
    from vibeqc.periodic_runner import _bundled_pob_source_atoms

    atom_list = _bundled_pob_source_atoms("pob-tzvp-rev2")
    assert atom_list, "pob-tzvp-rev2 resolved no atoms; the ECP would be dropped"
    if elements is not None:
        atom_list = [atom for atom in atom_list if int(atom.Z) in elements]
    ecp_blocks, ecp_centers, eff_z, total_ncore = build_periodic_ecp_data(
        system, atom_list
    )
    assert ecp_blocks, "no ECP blocks resolved; this would run all-electron"
    return ecp_blocks, ecp_centers


@pytest.mark.parametrize("Z", [72, 73, 74], ids=["Hf", "Ta", "W"])
def test_periodic_ecp_n4_radial_power_is_deterministic(Z):
    """compute_ecp_lattice_from_primitives is bit-identical across repeats.

    Hf, Ta and W are the only pob-TZVP-REV2 records carrying an ECP
    primitive with Gaussian radial power n=4 (CRYSTAL NKL=2; every other
    record only ever uses n=2). libecpint stores u.n = n - 2, and its
    generated type2(triples, nbase, ...) reads values[N + u.n + l2] from a
    scratch buffer sized nbase + 2, with nbase fixed at code-generation time
    from angular momentum alone. u.n = 2 read past that buffer: a silent
    heap over-read on the analytic branch, not the quadrature fallback that
    prints "Quadrature failed". Ten single-threaded calls on this cell gave
    nine distinct matrices with |max| between 6e+148 and 1e+150 Ha
    (vibe-qc #358).

    build_libecpint.sh's nbase patch sizes the buffer from the triples and
    u.n, so this asserts the previously flaky quantity is exactly
    reproducible. The W bcc lattice is reused for Hf and Ta: only the
    element (and therefore its ECP primitives) varies, and the geometry is a
    fixed probe, not a physical Hf or Ta structure. Determinism is not
    correctness; the molecular dimer test below checks the values.
    """
    core = vq._vibeqc_core
    system = _build_w_bcc()
    system.unit_cell = [vq.Atom(Z, atom.xyz) for atom in system.unit_cell]
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    ecp_blocks, ecp_centers = _resolve_pob_ecp(system, elements={Z})

    options = vq.LatticeSumOptions()
    options.cutoff_bohr = options.nuclear_cutoff_bohr = 8.0

    previous_threads = vq.get_num_threads()
    vq.set_num_threads(1)
    try:
        results = [
            core.compute_ecp_lattice_from_primitives(
                basis, system, options, ecp_centers, ecp_blocks
            )
            for _ in range(5)
        ]
    finally:
        vq.set_num_threads(previous_threads)

    first_blocks = [np.asarray(block) for block in results[0].blocks]
    assert len(results[0].cells) >= 2, "probe cutoff resolved too few image cells"
    max_abs = max(np.max(np.abs(block)) for block in first_blocks)
    # A physical ECP matrix element on this cell is O(1e2) Ha (Ag/Au/Rb/W
    # controls measured for #358); the corrupted operator overflowed to
    # 1e148-1e150. 1e6 comfortably separates "sane" from "corrupt" without
    # pinning the exact physical magnitude here.
    assert max_abs < 1e6, f"|V_ECP| max = {max_abs:.6e} Ha is not a sane operator"
    for repeat, result in enumerate(results[1:], start=2):
        for cell_index, (expected, block) in enumerate(zip(first_blocks, result.blocks)):
            np.testing.assert_array_equal(
                np.asarray(block), expected,
                err_msg=f"Z={Z}: call {repeat} cell {cell_index} differs from call 1",
            )


@pytest.mark.parametrize(
    "Z, symbol", [(72, "Hf"), (73, "Ta"), (74, "W"), (47, "Ag")],
    ids=["Hf2", "Ta2", "W2", "Ag2-control"],
)
def test_pob_ecp_dimer_matches_pyscf_ecpscalar(Z, symbol):
    """The n=4 ECP operator agrees with PySCF's ECPscalar on a dimer.

    vibe-qc #358 in its molecular form: compute_ecp_matrix_from_primitives
    on W2 at the bcc nearest-neighbour distance returned 20 distinct
    matrices in 20 single-threaded calls with |max| ~ 3.6e+194 Ha once the
    lattice route was already stable, because the first nbase patch
    budgeted N + u.n and missed the + l2 term of the generated switch.

    The two sides share only the pob-TZVP-rev2 orbital basis. vibe-qc takes
    the ECP from the CRYSTAL source through crystal_ecp_to_libecpint_arrays,
    as build_periodic_ecp_data does; PySCF takes the bundled Gaussian
    sidecar through the reference generator's converter and evaluates it
    with its own libcint ECP code. The generalized eigenvalues of (V, S)
    are compared, which does not depend on AO order or normalisation. Both
    builds agreed to 4e-12 Ha when this was written; Ag (n=2 only) is the
    control that the comparison itself is sound.
    """
    import importlib.util

    import scipy.linalg

    pyscf = pytest.importorskip("pyscf")
    from pyscf.gto.basis import parse_gaussian

    from vibeqc.basis_crystal import crystal_ecp_to_libecpint_arrays
    from vibeqc.periodic_runner import _bundled_pob_source_atoms

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "periodic_ecp_metal_reference",
        root / "examples/regression/periodic_ecp_metal_reference.py",
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    d = 5.222234  # W bcc nearest-neighbour distance, bohr
    mol = vq.Molecule([vq.Atom(Z, [0.0, 0.0, 0.0]), vq.Atom(Z, [0.0, 0.0, d])])
    basis = vq.BasisSet(mol, "pob-tzvp-rev2")
    record = next(
        atom for atom in _bundled_pob_source_atoms("pob-tzvp-rev2") if int(atom.Z) == Z
    )
    arrays = crystal_ecp_to_libecpint_arrays(record.ecp)
    assert (4 in arrays.ns) == (Z != 47), "the n=4 records are Hf, Ta and W"
    blocks = []
    for _ in range(2):
        block = vq._vibeqc_core.ECPPrimitiveBlock()
        block.n_primitive = arrays.n_primitives
        block.exponents = list(arrays.exponents)
        block.coefficients = list(arrays.coefficients)
        block.ams = list(arrays.ams)
        block.ns = list(arrays.ns)
        blocks.append(block)
    V = np.asarray(
        vq.compute_ecp_matrix_from_primitives(basis, [0.0, 0.0, 0.0, 0.0, 0.0, d], blocks)
    )
    S = np.asarray(vq.compute_overlap(basis))

    library = root / "python/vibeqc/basis_library/basis"
    ref_mol = pyscf.gto.M(
        atom=[(symbol, (0.0, 0.0, 0.0)), (symbol, (0.0, 0.0, d))],
        basis={symbol: parse_gaussian.load(
            str(library / "pob-tzvp-rev2.g94"), symbol, optimize=False)},
        ecp={symbol: pyscf.gto.basis.parse_ecp(
            reference._sidecar_nwchem(library / "pob-tzvp-rev2.ecp", symbol))},
        unit="Bohr", verbose=0,
    )
    assert ref_mol.nao == basis.nbasis
    expected = scipy.linalg.eigh(
        ref_mol.intor("ECPscalar"), ref_mol.intor("int1e_ovlp"), eigvals_only=True
    )
    actual = scipy.linalg.eigh(V, S, eigvals_only=True)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-8)


@pytest.mark.slow
@pytest.mark.periodic
@pytest.mark.gdf
@pytest.mark.external_reference
def test_periodic_ecp_w_bcc_pbe():
    """W bcc PBE/pob-TZVP-REV2 ECP: parity with PySCF.pbc ≤ 1 mHa/atom.

    W (Z=74): 60 core electrons replaced by ECP (ecp60mdf equivalent),
    14 valence electrons per atom.  2-atom conventional cell → 28 valence
    electrons (14 occupied).

    The cross-code pin vibe-qc#31 asks for: the first periodic **ECP**
    energy this project holds against an independent program. Everything the
    other periodic-ECP tests prove is self-consistent with vibe-qc's own
    conventions, so an error shared between the energy expression and the
    test would survive them.

    Γ, not the helper's default 2×2×2. One k-point is enough to exercise the
    full lattice-summed V_ECP operator — the ECP is a one-electron operator
    built on the real-space image list, and the image list does not depend on
    the k-mesh — and it removes the sampling convention from the comparison
    entirely, since Γ is Γ under either Monkhorst-Pack shift. The archived
    reference also carries the 2×2×2 Γ-centred run for anyone who wants the
    multi-k comparison as well.

    Reference: examples/regression/periodic_ecp_metal_reference.py, archived
    at tests/data/periodic_gdf/w_bcc_ecp_reference.{json,system}.
    """
    reference = _w_bcc_reference("gamma")
    assert reference["energy"] == _PYSCF_REF_W_BCC, (
        "the pinned constant and the archived reference disagree; "
        "one of them was edited without the other"
    )
    # The reference's own frame, re-asserted here so a regenerated artefact
    # cannot silently change what is being compared.
    assert reference["n_atoms"] == 2 and reference["n_electrons"] == 28
    assert reference["sigma"] == 0.005
    assert reference["kpoints_fractional"] == [[0.0, 0.0, 0.0]]

    system = _build_w_bcc()
    result = _run_pbe_ecp(system, kmesh=(1, 1, 1))

    deviation = abs(result.energy - _PYSCF_REF_W_BCC) / 2
    assert deviation < 1e-3, (
        f"W bcc ECP energy {result.energy:.8f} Ha deviates from PySCF "
        f"{_PYSCF_REF_W_BCC:.8f} by "
        f"{deviation:.6f} Ha/atom "
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
    # RIJCOSX applies the ECP since #35 (Gamma drivers and the k-GDF COSX
    # backend); the external-XC GPW adapter still drops it and is refused.
    with pytest.raises(NotImplementedError, match=r"does not apply a periodic ECP"):
        run_periodic_job(
            system, basis, method="RKS", functional="pbe", kpoints=[1, 1, 1],
            jk_method="gpw", smearing_temperature=0.005,
            output=tmp_path / "ag_refused_gpw", max_iter=2,
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


def test_an_ecp_cell_never_reaches_the_bare_z_property_rebuild(tmp_path):
    """The generic QVF band-property rebuild assembles a bare-Z operator, so an
    ECP cell has no DOS from it on any route.

    vibe-qc#32 reads as if ECP cells ship a QVF without DOS/PDOS/COOP/COHP
    because this gate returns ``False``. Until #35 that state was
    unreachable: ``_periodic_route_applies_ecp`` admitted an ECP cell on the
    GDF route only, where the accepted-state adapter serves the payload and
    the gate is never consulted. Since #35 the Gamma RIJCOSX drivers apply
    the ECP too, so ``run_periodic_job`` does reach the ``resolved_jk !=
    GDF`` property branch with ``ecp_active=True``. The decision pinned here:
    the gate withholds the bare-Z rebuild with the ECP reason, and the run
    records the omission in its ``.system`` manifest instead of writing a
    COHP of an operator the SCF never solved.
    """
    import tomllib

    from vibeqc.periodic_runner import (
        PeriodicJKMethod,
        _periodic_route_applies_ecp,
        _qvf_periodic_property_payload_skip_reason,
        _qvf_periodic_property_payload_supported,
        run_periodic_job,
    )

    system = _build_nacl_rocksalt()
    methods = ("RHF", "RKS", "UHF", "UKS", "ROHF", "ROKS")
    reachable_with_ecp = {
        (jk, method)
        for jk in PeriodicJKMethod
        for method in methods
        if _periodic_route_applies_ecp(jk, method, system)
    }
    assert reachable_with_ecp == {
        (PeriodicJKMethod.GDF, m) for m in ("RHF", "RKS", "UHF", "UKS", "ROHF")
    } | {
        (PeriodicJKMethod.RIJCOSX, m) for m in ("RHF", "RKS", "UHF", "UKS")
    }, (
        "a route newly applies a periodic ECP; decide what the bare-Z "
        "property rebuild must do there before admitting it"
    )
    for jk, _method in reachable_with_ecp:
        # The gate never admits the bare-Z rebuild on an ECP cell, and names
        # the operator mismatch for the manifest.
        assert not _qvf_periodic_property_payload_supported(jk, ecp_active=True)
        assert "V_ECP" in (
            _qvf_periodic_property_payload_skip_reason(jk, ecp_active=True) or ""
        )

    # ...and a Gamma RIJCOSX run on an ECP cell records the omission.
    box, basis, _data = _lanl2dz_molecule_in_box((11, 17), 2.36)
    stem = tmp_path / "rijcosx_ecp_no_property_payload"
    result = run_periodic_job(
        box, basis, method="RKS", functional="pbe", kpoints=[1, 1, 1],
        jk_method="rijcosx", aux_basis="def2-svp-jk", output=stem,
        output_qvf=True, citations=False, progress=False,
    )
    assert result.converged
    assert -15.5 < float(result.energy) < -14.5  # valence-only scale
    run = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))["run"]
    assert run["property_hamiltonian"].startswith("omitted ("), run["property_hamiltonian"]
    assert "V_ECP" in run["property_hamiltonian"], run["property_hamiltonian"]


def test_an_omitted_qvf_property_payload_records_why(tmp_path):
    """A gated-off property payload must say so, not vanish silently.

    ``property_hamiltonian`` reaches the ``.system`` manifest only from the
    GDF success branch, so a run whose payload was withheld by
    ``_qvf_periodic_property_payload_supported`` wrote nothing at all: two
    runs of the same cell, one with DOS in its QVF and one without, and no
    field anywhere naming the difference. That silence is the part of
    vibe-qc#32 that is real: a reduced artifact must be stated, not silent.

    BIPOLE is used as the representative gated route; the ECP branch of the
    same gate is unreachable (see above).
    """
    import tomllib

    from vibeqc.periodic_runner import (
        PeriodicJKMethod,
        _qvf_periodic_property_payload_skip_reason,
        _qvf_periodic_property_payload_supported,
        run_periodic_job,
    )

    # The reason is available to callers, and agrees with the boolean gate.
    assert _qvf_periodic_property_payload_skip_reason(PeriodicJKMethod.GDF) is None
    for jk, needle in (
        (PeriodicJKMethod.BIPOLE, "BIPOLE"),
        (PeriodicJKMethod.AICCM2026DEV_B, "AICCM2026DEV-B"),
    ):
        reason = _qvf_periodic_property_payload_skip_reason(jk)
        assert reason and needle in reason, (jk, reason)
        assert not _qvf_periodic_property_payload_supported(jk)
    assert "external" in (
        _qvf_periodic_property_payload_skip_reason(
            PeriodicJKMethod.GDF, uses_external_xc=True
        )
        or ""
    )
    assert "ECP" in (
        _qvf_periodic_property_payload_skip_reason(
            PeriodicJKMethod.GDF, ecp_active=True
        )
        or ""
    )

    # ...and a real run that hits the gate records the reason it hit.
    box = 12.0
    c = box / 2
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "bipole_no_property_payload"
    result = run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="bipole",
        bipole_cutoff_bohr=15.0,
        bipole_nuclear_cutoff_bohr=15.0,
        output=stem,
        output_qvf=True,
        citations=False,
        progress=False,
    )
    assert result.converged
    run = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))["run"]
    assert "property_hamiltonian" in run, (
        "the omitted DOS/PDOS/COOP/COHP payload left no trace in the manifest"
    )
    assert run["property_hamiltonian"].startswith("omitted ("), run[
        "property_hamiltonian"
    ]
    assert "BIPOLE" in run["property_hamiltonian"]


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


def test_ecp_payload_one_electron_term_is_not_the_bare_z_build(tmp_path):
    """vibe-qc#32 closure criterion (b), the half not already pinned above.

    The producer case pins that an ECP cell's QVF carries
    dos.total/projected/coop/cohp and that ``dos/cohp_integrated.bin`` is the
    contraction of the *accepted* ECP density and Fock. This pins the other
    half: that the accepted one-electron frame is nowhere near the bare-Z one
    the legacy rebuild would have used, so the payload could not have come
    from a bare-Z build and cannot be confused with one.

    Both frames are built on the same Ewald-3D lattice options, so the
    difference measured is the physics (Z_eff plus the lattice-summed V_ECP
    against bare Z), not a cutoff artifact. Reported as the ICOHP error the
    legacy rebuild would have made: the 2e part of F is common to both
    frames, so ``F_bare = F_accepted + (V_bare - V_eff)`` and the induced
    ICOHP shift is the Na-Cl block contraction of that difference.
    """
    import json
    import zipfile

    from vibeqc._vibeqc_core import CoulombMethod, LatticeSumOptions, bloch_sum
    from vibeqc.bands import _HARTREE_TO_EV, _shell_to_atom
    from vibeqc.periodic_k_gdf import _ecp_lattice_blocks, _periodic_ecp_context
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
    from vibeqc.periodic_runner import _resolve_ecp_data, run_periodic_job
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    system = _build_nacl_rocksalt()
    system = vq.PeriodicSystem(3, np.asarray(system.lattice), [
        system.unit_cell[0],
        vq.Atom(17, list(np.asarray(system.unit_cell[1].xyz)
                        - np.asarray(system.lattice)[:, 0])),
    ], 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")

    stem = tmp_path / "nacl_lanl_frames"
    result = run_periodic_job(
        system, basis, method="RHF", kpoints=[1, 1, 1], jk_method="gdf",
        aux_basis="def2-svp-jk", output=stem, max_iter=80, progress=False,
        output_qvf=True, coop_cohp=True,
    )
    assert result.converged

    with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        sections = {s["kind"] for s in manifest["sections"]}
        assert {"dos.total", "dos.projected", "dos.coop", "dos.cohp"} <= sections
        icohp = np.frombuffer(
            archive.read("dos/cohp_integrated.bin"), dtype="<f8"
        )
    # Provenance says the payload was projected, not rebuilt.
    import tomllib

    run_fields = tomllib.loads(
        stem.with_suffix(".system").read_text("utf-8")
    )["run"]
    assert run_fields["property_hamiltonian"] == "accepted-scf"

    blocks, centers, eff_z, ncore = _resolve_ecp_data(system, basis)
    ecp_ctx = _periodic_ecp_context(
        SimpleNamespace(
            ecp_primitive_blocks=blocks, ecp_home_centers=centers,
            ecp_effective_charges=eff_z, ecp_total_ncore=ncore,
        ),
        system, "test",
    )
    system_v = ecp_ctx[4]
    lat_opts = LatticeSumOptions()
    lat_opts.coulomb_method = CoulombMethod.EWALD_3D
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat_opts, system)
    gamma = np.zeros(3)
    v_eff = np.asarray(bloch_sum(
        compute_nuclear_lattice_dispatch(basis, system_v, gauge), gamma
    )) + np.asarray(bloch_sum(
        _ecp_lattice_blocks(basis, system, lat_opts, ecp_ctx), gamma
    ))
    v_bare = np.asarray(bloch_sum(
        compute_nuclear_lattice_dispatch(basis, system, gauge), gamma
    ))
    delta = v_bare - v_eff
    assert np.abs(delta).max() > 1.0, (
        "bare-Z and ECP one-electron frames agree; the ECP operator is not "
        "reaching the Hamiltonian at all"
    )

    ao_atoms = _shell_to_atom(basis)
    ab = np.ix_(np.flatnonzero(ao_atoms == 0), np.flatnonzero(ao_atoms == 1))
    density = np.asarray(result.density[0])
    icohp_shift = -np.sum(density.T[ab] * delta[ab]).real * _HARTREE_TO_EV
    assert abs(icohp_shift) > 1.0, (
        f"a bare-Z property rebuild would shift the reported Na-Cl ICOHP by "
        f"{icohp_shift:.3f} eV against the accepted {icohp[0]:.3f} eV; that "
        "is too small to distinguish the two builds, so this test no longer "
        "pins what it claims"
    )


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

    # Native drivers require interacting image cells, as in the restart fixture.
    box_bohr, cutoff_bohr = (6., 8.) if route.startswith("native_") else (20., 5.)
    system = vq.PeriodicSystem(3, np.eye(3) * box_bohr, [
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
        opts.lattice_opts.cutoff_bohr = cutoff_bohr
        opts.lattice_opts.nuclear_cutoff_bohr = cutoff_bohr
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


_ECP_APPLYING_GDF_ROUTES = (
    "rohf_gdf", "rijcosx_rhf", "rijcosx_uhf", "rijcosx_rks", "rijcosx_uks",
)


@pytest.mark.parametrize("route", _ALL_ELECTRON_GDF_ROUTES)
@pytest.mark.parametrize("charges", [[0., 1.], [1.], [1., float("nan")], [1.000000001, 1.]])
def test_all_electron_gdf_routes_reject_orphan_charges_before_setup(route, charges):
    """A reduced charge without its operator never reaches an integral.

    The Gamma fast paths and the slab driver refuse every ECP field; the
    routes that apply the operator since #35 (ROHF on GDF, Gamma RIJCOSX)
    run the orphan record through the k-point GDF drivers' validators, which
    name the missing ``ecp_primitive_blocks``."""
    driver, args, kwargs, _ = _all_electron_gdf_call(route, charges, with_basis=False)
    if route in _ECP_APPLYING_GDF_ROUTES:
        with pytest.raises(ValueError, match="non-empty ecp_primitive_blocks"):
            driver(*args, **kwargs)
    else:
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
    """A core count without an operator is refused before any integral: the
    Gamma RIJCOSX drivers share ``_periodic_ecp_context`` with the k-point
    GDF drivers since #35, and that context names the missing blocks."""
    driver, args, kwargs, opts = _all_electron_gdf_call(
        f"rijcosx_{method}", [1., 1.], with_basis=False)
    opts.ecp_total_ncore = 2
    with pytest.raises(ValueError, match="without ecp_primitive_blocks"):
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


@pytest.mark.parametrize("z", [14, *range(24, 36)])
def test_pob_dzvp_restored_all_electron_sources_resolve_offline(monkeypatch, z):
    from types import SimpleNamespace
    from vibeqc import periodic_runner as runner
    def unexpected(*args, **kwargs):
        raise AssertionError("bundled element must resolve offline")
    _patch_bredow_fetcher(monkeypatch, unexpected)
    system = SimpleNamespace(unit_cell=[SimpleNamespace(Z=z, xyz=[0, 0, 0])])
    assert runner._resolve_ecp_data(system, SimpleNamespace(name="pob-dzvp-rev2")) == ([], [], [float(z)], 0)


@pytest.mark.parametrize("z,has_ecp,should_pass", [(14, False, True), (47, False, False), (47, True, False)])
def test_pob_fetched_record_semantics(monkeypatch, z, has_ecp, should_pass):
    from types import SimpleNamespace
    from vibeqc import periodic_runner as runner
    record = SimpleNamespace(Z=z, has_ecp=has_ecp, ecp=None)
    monkeypatch.setattr(runner, "_bundled_pob_source_atoms", lambda name: [])
    _patch_bredow_fetcher(monkeypatch, lambda **kw: ({}, {"pob-dzvp-rev2": [record]}))
    system = SimpleNamespace(unit_cell=[SimpleNamespace(Z=z, xyz=[0, 0, 0])])
    basis = SimpleNamespace(name="pob-dzvp-rev2")
    if should_pass:
        assert runner._resolve_ecp_data(system, basis) == ([], [], [14.0], 0)
    else:
        with pytest.raises(RuntimeError, match="ECP-bearing record"):
            runner._resolve_ecp_data(system, basis)


def test_pob_incomplete_bundled_ecp_is_not_all_electron(monkeypatch):
    from types import SimpleNamespace
    from vibeqc import periodic_runner as runner
    atom = SimpleNamespace(Z=47, has_ecp=True, ecp=None)
    monkeypatch.setattr(runner, "_bundled_pob_source_atoms", lambda name: [atom])
    system = SimpleNamespace(unit_cell=[SimpleNamespace(Z=47, xyz=[0, 0, 0])])
    with pytest.raises(RuntimeError, match="ECP-bearing record"):
        runner._resolve_ecp_data(system, SimpleNamespace(name="pob-dzvp-rev2"))


def test_native_gamma_physical_operator_is_hermitian_and_reciprocal():
    """Finite Gamma domain, heterogeneous AOs, raw matrices; no SCF shortcut.

    Run on a containing native runtime. This checks operator consistency,
    not absolute bulk energy accuracy or the NaH SCF convergence criterion.
    """
    core = vq._vibeqc_core
    system = vq.PeriodicSystem(3, np.eye(3) * 8., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.5]),
    ])
    basis = core.BasisSet(system.unit_cell_molecule(), [
        core.ShellInfo(0, 0, True, [0.7], [1.0], [0., 0., 0.]),
        core.ShellInfo(1, 0, True, [0.1], [1.0], [0., 0., 3.5]),
    ], "gamma-operator-contract", coefficients_pre_normalized=True)
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = options.nuclear_cutoff_bohr = 10.
    assert not options.pair_complete_1e
    density = np.array([[0.8, 0.2], [0.2, 0.5]])
    direction = np.array([[0.3, -0.1], [-0.1, -0.4]])
    jd = core.build_jk_gamma_physical(basis, system, options, density)
    ja = core.build_jk_gamma_physical(basis, system, options, direction)
    assert not options.pair_complete_1e  # resolver never mutates its caller
    for name in ("J", "K"):
        md, ma = np.asarray(getattr(jd, name)), np.asarray(getattr(ja, name))
        np.testing.assert_allclose(md, md.T, atol=1e-10, rtol=0)
        np.testing.assert_allclose(ma, ma.T, atol=1e-10, rtol=0)
        assert abs(np.sum(density * ma) - np.sum(direction * md)) < 1e-10

    options.pair_complete_1e = True
    fold = lambda result: np.sum(np.asarray(result.blocks), axis=0)
    kinetic = fold(core.compute_kinetic_lattice(basis, system, options))
    nuclear = fold(core.compute_nuclear_lattice_with_charges(
        basis, system, options, [1., 1.]))
    ecp = fold(core.compute_ecp_lattice_from_primitives(
        basis, system, options, [[0., 0., 0.]], [_gaussian_projector_block(nonlocal_channel=True)],
        physical_pair_support=True))
    for matrix in (kinetic, nuclear, ecp):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10, rtol=0)
    h = kinetic + nuclear + ecp
    f = h + np.asarray(jd.J) - .5 * np.asarray(jd.K)
    derivative = np.sum(direction * f)
    # Retain the original small stencil: compensated inner quartet sums
    # resolve the rounding error without weakening this derivative contract.
    step = 1e-4
    energies = []
    for d in (density + step * direction, density - step * direction):
        jk = core.build_jk_gamma_physical(basis, system, options, d)
        energies.append(np.sum(d * h) + .5 * np.sum(d * (np.asarray(jk.J) - .5 * np.asarray(jk.K))))
    assert abs((energies[0] - energies[1]) / (2 * step) - derivative) < 1e-8


# ---------------------------------------------------------------------------
# #35: periodic ECPs on ROHF/GDF, Gamma RIJCOSX and the open-shell Ewald drivers
# ---------------------------------------------------------------------------
#
# Before #35 run_periodic_job applied a periodic ECP on one route (the k-point
# GDF drivers, RHF/RKS/UHF/UKS) and refused an ECP-bearing cell everywhere
# else. The routes below now build the same three pieces: the Z_eff nuclear
# frame for V_ne and the ionic energy, the lattice-summed V_ECP in Hcore, and
# the valence fill with the ECP context threaded into the atomic seeds. The
# fixtures are LANL2DZ cells (ECPs on Na, Mg, Al, Cl; the basis ships no
# fitting set, so aux_basis is explicit).


def _lanl2dz_molecule_in_box(z_pair, bond_ang, *, box_bohr=20.0, multiplicity=1):
    """A diatomic with LANL2DZ ECPs on both atoms, centred in a cubic box.

    Returns ``(system, basis, ecp_data)`` where ``ecp_data`` is the payload
    the runner resolves from the basis' ``.ecp`` sidecar."""
    from vibeqc.periodic_runner import _resolve_ecp_data

    half = 0.5 * box_bohr
    d = bond_ang * ANG
    atoms = [
        vq.Atom(z_pair[0], [half, half, half - d / 2]),
        vq.Atom(z_pair[1], [half, half, half + d / 2]),
    ]
    system = vq.PeriodicSystem(3, np.eye(3) * box_bohr, atoms, 0, multiplicity)
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    return system, basis, _resolve_ecp_data(system, basis)


def _periodic_options_with_ecp(data, *, is_ks=False, max_iter=150):
    from vibeqc.periodic_runner import _attach_periodic_ecp_data

    opts = vq.PeriodicKSOptions() if is_ks else vq.PeriodicRHFOptions()
    _attach_periodic_ecp_data(opts, data)
    opts.max_iter = max_iter
    return opts


def _molecular_options_with_ecp(cls, data, *, cosx=False, functional=None):
    """Molecular SCF options carrying the same inline ECP: the oracle."""
    opts = cls()
    opts.ecp_primitive_blocks = list(data[0])
    opts.ecp_primitive_centers = [list(c) for c in data[1]]
    opts.ecp_effective_charges = list(data[2])
    opts.ecp_total_ncore = int(data[3])
    if cosx:
        opts.density_fit = True
        opts.aux_basis = "def2-svp-jk"
        opts.cosx = True
    if functional:
        opts.functional = functional
    return opts


def _electron_trace(density, overlap):
    return float(np.trace(np.asarray(density) @ np.asarray(overlap)).real)


def _build_mg_fcc():
    """One Mg atom per fcc cell (a = 4.51 Angstrom): a tight 3D cell whose
    only atom carries the LANL2DZ ECP (10 core electrons, 2 valence)."""
    a = 4.51 * ANG
    lat = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(12, [0.0, 0.0, 0.0])], 0, 1)


def _build_al_fcc_doublet():
    """One Al atom per fcc cell (a = 4.05 Angstrom), 3 valence electrons under
    the LANL2DZ ECP, multiplicity 2."""
    a = 4.05 * ANG
    lat = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(13, [0.0, 0.0, 0.0])], 0, 2)


def _gamma_fold(lattice):
    from vibeqc.periodic_gradient import _fold_gamma_real

    return _fold_gamma_real(lattice)


def test_periodic_route_predicate_admits_the_ecp_routes_of_35():
    """The runner's ECP admission: GDF (RHF/RKS/UHF/UKS/ROHF) and RIJCOSX
    (RHF/RKS/UHF/UKS) on 3D cells; nothing on a slab, nothing on the routes
    that still drop the operator."""
    from vibeqc.periodic_jk_method import PeriodicJKMethod
    from vibeqc.periodic_runner import _periodic_route_applies_ecp

    bulk = _build_mg_fcc()
    slab = vq.PeriodicSystem(2, np.eye(3) * 10.0, [vq.Atom(12, [0.0, 0.0, 0.0])], 0, 1)
    for method in ("RHF", "RKS", "UHF", "UKS", "ROHF"):
        assert _periodic_route_applies_ecp(PeriodicJKMethod.GDF, method, bulk)
        assert not _periodic_route_applies_ecp(PeriodicJKMethod.GDF, method, slab)
    for method in ("RHF", "RKS", "UHF", "UKS"):
        assert _periodic_route_applies_ecp(PeriodicJKMethod.RIJCOSX, method, bulk)
        assert not _periodic_route_applies_ecp(PeriodicJKMethod.RIJCOSX, method, slab)
    assert not _periodic_route_applies_ecp(PeriodicJKMethod.RIJCOSX, "ROHF", bulk)
    assert not _periodic_route_applies_ecp(PeriodicJKMethod.GDF, "ROKS", bulk)
    for jk in (
        PeriodicJKMethod.BIPOLE, PeriodicJKMethod.GPW, PeriodicJKMethod.GAPW,
        PeriodicJKMethod.SLAB_EWALD_2D,
    ):
        assert not _periodic_route_applies_ecp(jk, "RHF", bulk)


def test_ecp_context_refuses_a_slab_cell_and_the_ewald_slab_driver_follows():
    """Slabs keep an explicit refusal: the shared context names the 3D
    restriction, and the UKS multi-k Ewald driver (the slab UKS route)
    raises it before any integral."""
    from vibeqc.periodic_k_gdf import _periodic_ecp_context
    from vibeqc.periodic_uks_multi_k_ewald import run_uks_periodic_multi_k_ewald3d

    system, basis, data = _lanl2dz_molecule_in_box((11, 17), 2.36)
    slab = vq.PeriodicSystem(2, np.asarray(system.lattice), list(system.unit_cell), 0, 1)
    opts = _periodic_options_with_ecp(data, is_ks=True)
    opts.functional = "pbe"
    with pytest.raises(NotImplementedError, match="3D cells only"):
        _periodic_ecp_context(opts, slab, "test")
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    with pytest.raises(NotImplementedError, match="3D cells only"):
        run_uks_periodic_multi_k_ewald3d(
            slab, basis, vq.monkhorst_pack(slab, [1, 1, 1]), opts, progress=False, verbose=0,
        )


def test_run_periodic_job_dispatches_ecp_cells_to_rijcosx_and_krohf(monkeypatch, tmp_path):
    """The runner attaches the resolved sidecar ECP and dispatches to the
    Gamma RIJCOSX drivers and to ROHF/GDF instead of refusing (#35). Pinned
    by interception, so no SCF runs."""
    import vibeqc.periodic_rijcosx as rij
    import vibeqc.periodic_runner as pr
    from vibeqc.periodic_runner import run_periodic_job

    class _DriverReached(Exception):
        pass

    def _payload(opts):
        return (
            list(opts.ecp_effective_charges), int(opts.ecp_total_ncore),
            len(opts.ecp_primitive_blocks),
        )

    def _rijcosx(system, basis, opts, *args, **kwargs):
        raise _DriverReached(*_payload(opts))

    def _krohf(system, basis, kmesh, opts, *args, **kwargs):
        raise _DriverReached(*_payload(opts))

    for name in (
        "run_periodic_rijcosx_rhf", "run_periodic_rijcosx_rks",
        "run_periodic_rijcosx_uhf", "run_periodic_rijcosx_uks",
    ):
        monkeypatch.setattr(rij, name, _rijcosx)
    monkeypatch.setattr(pr, "run_krohf_periodic_gdf", _krohf)

    system = _build_nacl_rocksalt()
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    cases = [
        ("RHF", None, "rijcosx"), ("RKS", "pbe", "rijcosx"),
        ("UHF", None, "rijcosx"), ("UKS", "pbe", "rijcosx"),
        ("ROHF", None, "gdf"),
    ]
    for method, functional, jk in cases:
        kwargs = {
            "method": method, "kpoints": [1, 1, 1], "jk_method": jk,
            "aux_basis": "def2-svp-jk", "output": tmp_path / f"dispatch_{method}_{jk}",
            "progress": False,
        }
        if functional:
            kwargs["functional"] = functional
        with pytest.raises(_DriverReached) as info:
            run_periodic_job(system, basis, **kwargs)
        charges, ncore, n_blocks = info.value.args
        assert charges == pytest.approx([1.0, 7.0]), (method, jk)
        assert ncore == 20 and n_blocks == 2, (method, jk)


@pytest.mark.parametrize(
    "method, functional", [("RHF", None), ("UHF", None), ("RKS", "pbe"), ("UKS", "pbe")],
)
def test_gamma_rijcosx_vacuum_envelope_applies_the_sidecar_ecp(method, functional):
    """An NaCl molecule in a 20-bohr box on the vacuum-padded Gamma RIJCOSX
    drivers with its LANL2DZ ECP (#35). The ionic energy is the Z_eff pair
    energy 1 x 7 / d (home-cell nuclei only), the density holds the 8 valence
    electrons, and the energy reproduces the molecular ECP calculation: to
    roundoff for RKS/UKS (same DF J, same home-cell COSX K, same grid;
    measured 0.0 and 5e-13 Ha), and to the route's exchange-quadrature floor
    for RHF/UHF (-4.6e-4 Ha against exact molecular RHF; the all-electron
    NaCl/def2-SVP control on the same path sits -1.1e-3 Ha from its exact
    oracle, so the offset is the COSX grid, not the ECP)."""
    from vibeqc import periodic_rijcosx as rij

    system, basis, data = _lanl2dz_molecule_in_box((11, 17), 2.36)
    d = 2.36 * ANG
    mol = system.unit_cell_molecule()
    assert data[2] == pytest.approx([1.0, 7.0]) and data[3] == 20
    driver = getattr(rij, f"run_periodic_rijcosx_{method.lower()}")
    kwargs = {"aux_basis": "def2-svp-jk", "progress": False}
    if functional:
        kwargs["functional"] = functional
    result = driver(system, basis, _periodic_options_with_ecp(data), **kwargs)
    assert result.converged
    assert float(result.e_nuclear) == pytest.approx(1.0 * 7.0 / d, abs=1e-9)
    if method in ("RHF", "RKS"):
        assert _electron_trace(result.density, result.overlap) == pytest.approx(8.0, abs=1e-8)
    else:
        assert _electron_trace(result.density_alpha, result.overlap) == pytest.approx(4.0, abs=1e-8)
        assert _electron_trace(result.density_beta, result.overlap) == pytest.approx(4.0, abs=1e-8)
    if functional:
        oracle = vq.run_rks(
            mol, basis,
            _molecular_options_with_ecp(vq.RKSOptions, data, cosx=True, functional=functional),
        )
        assert float(result.energy) == pytest.approx(float(oracle.energy), abs=1e-7)
    else:
        exact = vq.run_rhf(mol, basis, _molecular_options_with_ecp(vq.RHFOptions, data))
        # Independent ORCA 6.1.1 reference: ! RHF LANL2DZ VeryTightSCF,
        # neutral singlet Na (0,0,0), Cl (0,0,2.36) in Angstrom, 16 AOs,
        # Hay-Wadt ECPs removing 10 electrons from each atom (8 valence).
        # Pin the exact molecular oracle separately from the COSX grid error.
        assert float(exact.energy) == pytest.approx(-14.968249075143, abs=1e-7)
        assert abs(float(result.energy) - float(exact.energy)) < 3e-3
        # Valence-only scale: all-electron NaCl sits near -620 Ha.
        assert -15.5 < float(result.energy) < -14.5


def test_gamma_rijcosx_vacuum_doublet_splits_the_valence_count():
    """MgCl (Mg 2 + Cl 7 valence electrons under their ECPs, multiplicity 2)
    in a 20-bohr box: the open-shell Gamma RIJCOSX drivers fill 5 alpha and
    4 beta electrons, not 15 and 14, and reproduce the molecular ECP oracle
    (UKS/PBE to roundoff, measured 3e-13 Ha; UHF to -2.1e-4 Ha against the
    exact molecular UHF, with the molecular <S^2> of 0.7513)."""
    from vibeqc import periodic_rijcosx as rij

    system, basis, data = _lanl2dz_molecule_in_box((12, 17), 2.20, multiplicity=2)
    d = 2.20 * ANG
    mol = system.unit_cell_molecule()
    assert data[2] == pytest.approx([2.0, 7.0]) and data[3] == 20

    uhf = rij.run_periodic_rijcosx_uhf(
        system, basis, _periodic_options_with_ecp(data), aux_basis="def2-svp-jk", progress=False,
    )
    assert uhf.converged
    assert float(uhf.e_nuclear) == pytest.approx(2.0 * 7.0 / d, abs=1e-9)
    assert _electron_trace(uhf.density_alpha, uhf.overlap) == pytest.approx(5.0, abs=1e-8)
    assert _electron_trace(uhf.density_beta, uhf.overlap) == pytest.approx(4.0, abs=1e-8)
    exact = vq.run_uhf(mol, basis, _molecular_options_with_ecp(vq.UHFOptions, data))
    assert abs(float(uhf.energy) - float(exact.energy)) < 2e-3
    assert float(uhf.s_squared) == pytest.approx(float(exact.s_squared), abs=1e-3)

    uks = rij.run_periodic_rijcosx_uks(
        system, basis, _periodic_options_with_ecp(data), functional="pbe",
        aux_basis="def2-svp-jk", progress=False,
    )
    assert uks.converged
    oracle = vq.run_uks(
        mol, basis, _molecular_options_with_ecp(vq.UKSOptions, data, cosx=True, functional="pbe"),
    )
    assert float(uks.energy) == pytest.approx(float(oracle.energy), abs=1e-7)
    assert _electron_trace(uks.density_alpha, uks.overlap) == pytest.approx(5.0, abs=1e-8)


def _build_nacl_chain(pad_bohr=16.0, bond_ang=2.36):
    """Na and Cl alternating along z (period 2d), transverse vacuum ``pad_bohr``
    beyond the default 15-bohr exchange cutoff: the one image direction the
    tight-cell Gamma RIJCOSX path still admits after #555."""
    d = bond_ang * ANG
    half = 0.5 * pad_bohr
    lattice = np.diag([pad_bohr, pad_bohr, 2.0 * d])
    atoms = [vq.Atom(11, [half, half, 0.0]), vq.Atom(17, [half, half, d])]
    return vq.PeriodicSystem(3, lattice, atoms, 0, 1)


def test_gamma_rijcosx_tight_cell_hands_the_valence_hamiltonian_to_the_scf(monkeypatch):
    """The tight-cell Gamma RIJCOSX path (GDF J + periodic COSX K) with an ECP
    (#35), on an NaCl chain / LANL2DZ whose exchange images span one lattice
    direction: the Hcore handed to the native SCF is T + V_ne(Z_eff) + V_ECP,
    the Gamma fold over the route's own cell list (symmetric to roundoff),
    the ionic energy is the Z_eff per-cell sum in the route's Ewald gauge,
    the fill is the 8 valence electrons and the SAD seed is a valence
    density. Pinned at the SCF boundary. A dense cell (Mg fcc) is refused by
    the #555 image-rank guard before any integral, with or without the ECP."""
    from vibeqc import periodic_rijcosx as rij
    from vibeqc._vibeqc_core import compute_kinetic_lattice
    from vibeqc.pbc_gdf import _gauge_lat_opts_ewald_3d
    from vibeqc.periodic_k_gdf import _ecp_lattice_blocks, _periodic_ecp_context
    from vibeqc.periodic_runner import _resolve_ecp_data
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    system = _build_nacl_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    data = _resolve_ecp_data(system, basis)
    assert data[2] == pytest.approx([1.0, 7.0]) and data[3] == 20
    opts = _periodic_options_with_ecp(data)
    opts.initial_guess = vq.InitialGuess.SAD
    assert rij._exchange_image_rank(system, opts.lattice_opts) == 1

    class _Boundary(Exception):
        pass

    seen = {}

    def _capture(basis_, n_electrons, S, Hcore, E_nuc, jk, rhf_opts, initial_density, **kwargs):
        seen.update(
            n=int(n_electrons), S=np.asarray(S), H=np.asarray(Hcore), e_nuc=float(E_nuc),
            ncore=int(rhf_opts.ecp_total_ncore), blocks=len(rhf_opts.ecp_primitive_blocks),
            density=np.asarray(initial_density),
        )
        raise _Boundary

    monkeypatch.setattr(rij, "run_rhf_scf_with_jk", _capture)
    with pytest.raises(_Boundary):
        rij.run_periodic_rijcosx_rhf(
            system, basis, opts, aux_basis="def2-svp-jk", tight_cell=True, progress=False,
        )

    ctx = _periodic_ecp_context(opts, system, "test")
    gauge = _gauge_lat_opts_ewald_3d(opts.lattice_opts, system)
    T = _gamma_fold(compute_kinetic_lattice(basis, system, opts.lattice_opts))
    V = _gamma_fold(compute_nuclear_lattice_dispatch(basis, ctx[4], gauge))
    V_ecp = _gamma_fold(_ecp_lattice_blocks(basis, system, gauge, ctx))
    # The origin-ball projector images leave a small antisymmetric part in
    # the raw fold (1.9e-7 here); the route keeps the symmetric part, as the
    # k-point GDF drivers do by Hermitizing Hcore(k).
    assert np.abs(V_ecp).max() > 0.1 and np.abs(V_ecp - V_ecp.T).max() < 1e-5
    np.testing.assert_allclose(
        seen["H"], T + V + 0.5 * (V_ecp + V_ecp.T), atol=1e-10, rtol=0.0)
    # What is left of the asymmetry is V_ne's own origin-ball nuclear image
    # domain (#547), 2.9e-8 on this chain with or without the ECP.
    assert np.abs(seen["H"] - seen["H"].T).max() < 1e-6
    assert seen["e_nuc"] == pytest.approx(
        float(vq.nuclear_repulsion_per_cell(ctx[4], gauge)), abs=1e-10)
    assert seen["n"] == 8 and seen["ncore"] == 20 and seen["blocks"] == 2
    assert _electron_trace(seen["density"], seen["S"]) == pytest.approx(8.0, abs=1e-8)
    # The bare-Z frame the route used before #35 is a different operator.
    V_bare = _gamma_fold(compute_nuclear_lattice_dispatch(basis, system, gauge))
    assert np.abs(V - V_bare).max() > 1.0

    # Dense cell: the #555 guard refuses the tight path first, ECP or not.
    mg = _build_mg_fcc()
    mg_basis = vq.BasisSet(mg.unit_cell_molecule(), "lanl2dz")
    mg_opts = _periodic_options_with_ecp(_resolve_ecp_data(mg, mg_basis))
    with pytest.raises(NotImplementedError, match="span 3 lattice directions"):
        rij.run_periodic_rijcosx_rhf(mg, mg_basis, mg_opts, aux_basis="def2-svp-jk", progress=False)


def test_krohf_gdf_builds_the_valence_frame_with_the_ecp(monkeypatch):
    """ROHF on GDF reaches its fit with V_ne on the Z_eff frame and the
    lattice V_ECP beside S and T (#35). Pinned at the cderi seam, so neither
    the fit nor the SCF runs (the SR/LR fit on this diffuse basis takes
    minutes; the converged collapse gate is the slow test below)."""
    import vibeqc.periodic_k_gdf as kgdf
    import vibeqc.periodic_rohf_gdf as rohf
    import vibeqc.periodic_v_ne as v_ne
    from vibeqc.periodic_runner import _resolve_ecp_data

    system = _build_mg_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    data = _resolve_ecp_data(system, basis)
    opts = _periodic_options_with_ecp(data)

    seen = {"v_ne": [], "ecp_calls": 0}
    real_dispatch = v_ne.compute_nuclear_lattice_dispatch
    real_blocks = kgdf._ecp_lattice_blocks

    def _spy_v_ne(basis_, system_, lat_opts):
        seen["v_ne"].append(([int(a.Z) for a in system_.unit_cell], lat_opts))
        return real_dispatch(basis_, system_, lat_opts)

    def _spy_blocks(*args, **kwargs):
        seen["ecp_calls"] += 1
        return real_blocks(*args, **kwargs)

    class _Seam(Exception):
        pass

    def _cache(*args, **kwargs):
        seen["oneel"] = kwargs.get("oneel_lattices")
        raise _Seam

    monkeypatch.setattr(v_ne, "compute_nuclear_lattice_dispatch", _spy_v_ne)
    monkeypatch.setattr(kgdf, "_ecp_lattice_blocks", _spy_blocks)
    monkeypatch.setattr(rohf, "_build_scf_range_separated_lpq_cache", _cache)
    with pytest.raises(_Seam):
        rohf.run_krohf_periodic_gdf(
            system, basis, (1, 1, 1), opts, aux_basis="def2-svp-jk", progress=False, verbose=0,
        )

    assert [frame for frame, _ in seen["v_ne"]] == [[2]]  # the Z_eff frame, once
    assert seen["ecp_calls"] == 1
    S_lat, T_lat, V_lat, V_ecp_lat = seen["oneel"]
    assert V_ecp_lat is not None
    V_ecp = _gamma_fold(V_ecp_lat)
    assert np.abs(V_ecp).max() > 0.1 and np.abs(V_ecp - V_ecp.T).max() < 1e-10
    # V_ne is linear in the one nuclear charge: the Z_eff frame is 2/12 of the bare one.
    _, lat_opts = seen["v_ne"][0]
    V_bare = _gamma_fold(real_dispatch(basis, system, lat_opts))
    np.testing.assert_allclose(_gamma_fold(V_lat), (2.0 / 12.0) * V_bare, atol=1e-10, rtol=0.0)


@pytest.mark.slow
def test_krohf_gdf_with_the_ecp_collapses_to_krhf_on_a_closed_shell():
    """Mg fcc / LANL2DZ at Gamma: ROHF on GDF with multiplicity 1 reproduces
    the closed-shell k-point GDF energy on the ECP cell (measured 0.0 Ha),
    both on the Z_eff Ewald ionic energy with 2 valence electrons. About
    2.5 minutes per arm: the SR/LR fit on this diffuse basis dominates."""
    from vibeqc.periodic_k_gdf import _periodic_ecp_context, run_krhf_periodic_gdf
    from vibeqc.periodic_rohf_gdf import run_krohf_periodic_gdf
    from vibeqc.periodic_runner import _resolve_ecp_data

    system = _build_mg_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    data = _resolve_ecp_data(system, basis)
    e_nuc = float(vq.ewald_nuclear_repulsion(
        _periodic_ecp_context(_periodic_options_with_ecp(data), system, "test")[4]))
    assert abs(e_nuc - float(vq.ewald_nuclear_repulsion(system))) > 30.0
    common = {"aux_basis": "def2-svp-jk", "progress": False, "verbose": 0}
    rhf = run_krhf_periodic_gdf(system, basis, (1, 1, 1), _periodic_options_with_ecp(data), **common)
    rohf = run_krohf_periodic_gdf(system, basis, (1, 1, 1), _periodic_options_with_ecp(data), **common)
    assert rhf.converged and rohf.converged
    assert float(rhf.e_nuclear) == pytest.approx(e_nuc, abs=1e-8)
    assert float(rohf.e_nuclear) == pytest.approx(e_nuc, abs=1e-8)
    assert float(rohf.energy) == pytest.approx(float(rhf.energy), abs=1e-8)
    assert _electron_trace(rhf.density[0], rhf.overlap[0]) == pytest.approx(2.0, abs=1e-8)
    assert _electron_trace(rohf.density_alpha[0], rohf.overlap[0]) == pytest.approx(1.0, abs=1e-8)
    assert _electron_trace(rohf.density_beta[0], rohf.overlap[0]) == pytest.approx(1.0, abs=1e-8)
    assert -1.5 < float(rhf.energy) < -1.0  # 2 valence electrons, not 12


@pytest.mark.slow
def test_krohf_gdf_with_the_ecp_sits_above_kuhf_on_a_doublet():
    """Al fcc / LANL2DZ, multiplicity 2, at Gamma: ROHF on GDF splits 2 alpha
    and 1 beta valence electrons under the ECP, is spin-pure by construction
    and sits above the k-point UHF energy by the usual spin-restriction cost
    (measured +4.2e-7 Ha on a UHF with <S^2> = 0.75000006). About 2 minutes
    per arm."""
    from vibeqc.periodic_k_gdf import _periodic_ecp_context, run_kuhf_periodic_gdf
    from vibeqc.periodic_rohf_gdf import run_krohf_periodic_gdf
    from vibeqc.periodic_runner import _resolve_ecp_data

    system = _build_al_fcc_doublet()
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    data = _resolve_ecp_data(system, basis)
    assert data[2] == pytest.approx([3.0]) and data[3] == 10
    e_nuc = float(vq.ewald_nuclear_repulsion(
        _periodic_ecp_context(_periodic_options_with_ecp(data), system, "test")[4]))
    common = {"aux_basis": "def2-svp-jk", "progress": False, "verbose": 0}
    uhf = run_kuhf_periodic_gdf(system, basis, (1, 1, 1), _periodic_options_with_ecp(data), **common)
    rohf = run_krohf_periodic_gdf(system, basis, (1, 1, 1), _periodic_options_with_ecp(data), **common)
    assert uhf.converged and rohf.converged
    for result in (uhf, rohf):
        assert float(result.e_nuclear) == pytest.approx(e_nuc, abs=1e-8)
        assert _electron_trace(result.density_alpha[0], result.overlap[0]) == pytest.approx(2.0, abs=1e-8)
        assert _electron_trace(result.density_beta[0], result.overlap[0]) == pytest.approx(1.0, abs=1e-8)
    assert float(rohf.s_squared) == pytest.approx(0.75, abs=1e-10)
    assert float(uhf.s_squared) == pytest.approx(0.75, abs=1e-5)
    gap = float(rohf.energy) - float(uhf.energy)
    assert -1e-10 <= gap < 1e-5, gap


# PySCF KUHF with direct range-separated J/K, exxdiv='ewald', precision
# and SCF tolerance 1e-10, 20-bohr cubic cell, Cl2 at 1.99 Angstrom,
# LANL2DZ/Hay-Wadt ECP, charge/spin 0/0. Explicit cell.energy_nuc() keeps
# the 3D Ewald ionic term even on PySCF versions whose RSJK SCF wrapper
# selects molecular nuclear repulsion. No auxiliary fitting of J or K.
# Reproduce with examples/regression/parity_ecp_ewald_uhf.py.
_CL2_EWALD_UHF_REFERENCE = {
    (1, 1, 1): -29.3455846237986,
    (3, 1, 1): -29.340156129936254,
}

# ORCA 6.1.1: ! UHF LANL2DZ VeryTightSCF; neutral singlet Cl2, 1.99 A.
_CL2_MOLECULAR_UHF_REFERENCE = -29.335833072480


@pytest.fixture
def ecp_derivative_receipt():
    """Small synthetic terminal state for admission checks, not SCF evidence."""
    from vibeqc.periodic_ecp_source import (
        capture_ewald_uks_ecp_source, seal_ewald_uks_ecp_source,
    )
    from vibeqc.periodic_k_gdf import _periodic_ecp_context

    system, basis, data = _lanl2dz_molecule_in_box((17, 17), 1.99)
    opts = _periodic_options_with_ecp(data, is_ks=True)
    opts.functional = "pbe"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    mesh = vq.monkhorst_pack(system, [3, 1, 1])
    source = capture_ewald_uks_ecp_source(
        system, basis, mesh, opts, opts.lattice_opts,
        _periodic_ecp_context(opts, system, "test"), .4,
    )
    result = SimpleNamespace(
        converged=True, ecp_operator=True, ecp_total_ncore=20, functional="pbe",
        omega=.4, smearing_temperature=0., bz_integration="monkhorst-pack",
        effective_lattice_opts=opts.lattice_opts,
    )
    n = basis.nbasis
    for spin in ("alpha", "beta"):
        for field, value in (("mo_coeffs", np.eye(n, dtype=complex)),
                             ("mo_energies", np.arange(n, dtype=float)),
                             ("occupations", np.r_[np.ones(7), np.zeros(n-7)])):
            setattr(result, f"{field}_{spin}", [value.copy() for _ in range(3)])
        setattr(result, f"density_{spin}_k", [np.eye(n, dtype=complex) for _ in range(3)])
        setattr(result, f"density_{spin}", vq.compute_overlap_lattice(basis, system, opts.lattice_opts))
    result.ecp_derivative_source = seal_ewald_uks_ecp_source(source, result)
    return system, basis, mesh, result, source


@pytest.mark.parametrize("field", [
    "mo_coeffs_alpha", "mo_coeffs_beta", "mo_energies_alpha", "mo_energies_beta",
    "occupations_alpha", "occupations_beta", "density_alpha_k", "density_beta_k",
    "density_alpha", "density_beta", "ecp_flags", "missing_density", "promoted_convergence",
])
def test_ecp_gradient_rejects_changed_terminal_state(ecp_derivative_receipt, field):
    from vibeqc.periodic_ecp_source import resolve_ewald_uks_ecp_source, seal_ewald_uks_ecp_source

    system, basis, mesh, result, source = ecp_derivative_receipt
    # Admission of the unchanged state does not build derivative integrals.
    resolve_ewald_uks_ecp_source(system, basis, result, mesh, result.effective_lattice_opts, None)
    if field == "ecp_flags":
        result.ecp_operator, result.ecp_total_ncore = False, 0
    elif field == "missing_density":
        result.density_alpha_k = None
    elif field == "promoted_convergence":
        result.converged = False
        result.ecp_derivative_source = seal_ewald_uks_ecp_source(source, result)
        result.converged = True
    elif field in ("density_alpha", "density_beta"):
        density = getattr(result, field)
        block = np.asarray(density.blocks[0]).copy()
        block[0, 0] += .01
        density.set_block(0, block)
    else:
        getattr(result, field)[-1].flat[0] += .01
    with pytest.raises(ValueError, match="terminal SCF state"):
        vq.compute_gradient_periodic_uks_multi_k(system, basis, result, mesh)


def test_ecp_gradient_requires_terminal_receipt(ecp_derivative_receipt):
    system, basis, mesh, result, source = ecp_derivative_receipt
    result.ecp_derivative_source = source  # Old pre-build-only receipt.
    with pytest.raises(NotImplementedError, match="terminal SCF state; rerun"):
        vq.compute_gradient_periodic_uks_multi_k(system, basis, result, mesh)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("method,physical", [("UHF", False), ("UKS", False), ("UKS", True)])
def test_open_shell_multi_k_ewald_drivers_apply_the_ecp(monkeypatch, method, mesh, physical, record_property):
    """Cl2 (7 + 7 valence electrons under LANL2DZ ECPs) in a 20-bohr box on
    the open-shell multi-k Ewald drivers at Gamma and on a three-point mesh
    (#35): V_ne is built on the Z_eff frame, the lattice V_ECP is built once, the ionic
    energy is the Z_eff per-cell Ewald sum in the driver's forced EWALD_3D
    gauge, and each spin holds 7 valence electrons. UHF is compared with
    an independent periodic reference using the same exchange convention.
    The pure PBE molecular limit has no finite-k exact-exchange term."""
    import vibeqc.periodic_k_gdf as kgdf
    import vibeqc.periodic_v_ne as v_ne
    from vibeqc.pbc_gdf import _gauge_lat_opts_ewald_3d
    from vibeqc.periodic_k_gdf import _periodic_ecp_context
    from vibeqc.periodic_uhf_multi_k_ewald import run_uhf_periodic_multi_k_ewald3d
    from vibeqc.periodic_uks_multi_k_ewald import run_uks_periodic_multi_k_ewald3d

    system, basis, data = _lanl2dz_molecule_in_box((17, 17), 1.99)
    mol = system.unit_cell_molecule()
    assert data[2] == pytest.approx([7.0, 7.0]) and data[3] == 20
    is_ks = method == "UKS"
    opts = _periodic_options_with_ecp(data, is_ks=is_ks)
    opts.lattice_opts.pair_complete_1e = physical
    if is_ks:
        opts.functional = "pbe"
        opts.conv_tol_energy = 1e-12
        opts.conv_tol_grad = 1e-9

    seen = {"v_ne": [], "ecp_calls": 0}
    real_dispatch = v_ne.compute_nuclear_lattice_dispatch
    real_blocks = kgdf._ecp_lattice_blocks

    def _spy_v_ne(basis_, system_, lat_opts):
        seen["v_ne"].append([int(a.Z) for a in system_.unit_cell])
        return real_dispatch(basis_, system_, lat_opts)

    def _spy_blocks(*args, **kwargs):
        seen["ecp_calls"] += 1
        return real_blocks(*args, **kwargs)

    monkeypatch.setattr(v_ne, "compute_nuclear_lattice_dispatch", _spy_v_ne)
    monkeypatch.setattr(kgdf, "_ecp_lattice_blocks", _spy_blocks)
    driver = run_uks_periodic_multi_k_ewald3d if is_ks else run_uhf_periodic_multi_k_ewald3d
    result = driver(system, basis, vq.monkhorst_pack(system, list(mesh)), opts, progress=False, verbose=0)
    assert result.converged
    assert seen["v_ne"] == [[7, 7]] and seen["ecp_calls"] == 1

    ctx = _periodic_ecp_context(opts, system, "test")
    gauge = _gauge_lat_opts_ewald_3d(opts.lattice_opts, system)
    assert float(result.e_nuclear) == pytest.approx(
        float(vq.nuclear_repulsion_per_cell(ctx[4], gauge)), abs=1e-8)
    assert len(result.overlap) == int(np.prod(mesh))
    for da, db, overlap in zip(result.density_alpha_k, result.density_beta_k, result.overlap):
        assert _electron_trace(da, overlap) == pytest.approx(7.0, abs=1e-8)
        assert _electron_trace(db, overlap) == pytest.approx(7.0, abs=1e-8)
    if is_ks:
        assert result.ecp_operator is True
        assert result.ecp_total_ncore == 20
        assert result.ecp_derivative_source is not None
        # A converged result must not silently mix edited Pulay/orbital data
        # with the retained ECP and XC densities. Restoring it stays valid.
        saved = result.mo_energies_alpha[0].copy()
        result.mo_energies_alpha[0][0] += .01
        with pytest.raises(ValueError, match="terminal SCF state"):
            vq.compute_gradient_periodic_uks_multi_k(
                system, basis, result, vq.monkhorst_pack(system, list(mesh)))
        result.mo_energies_alpha[0] = saved
        gradient = np.asarray(vq.compute_gradient_periodic_uks_multi_k(
            system, basis, result, vq.monkhorst_pack(system, list(mesh)),
        ))
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-7, rtol=0)
        finite = []
        for h in (1e-3, 5e-4):
            energies = []
            for sign in (1., -1.):
                # Symmetric bond displacement: g[1,z] = dE/d(bond).
                sd, bd, ed = _lanl2dz_molecule_in_box((17, 17), 1.99 + sign*h/ANG)
                od = _periodic_options_with_ecp(ed, is_ks=True)
                od.functional = "pbe"
                od.lattice_opts.pair_complete_1e = physical
                od.conv_tol_energy = opts.conv_tol_energy
                od.conv_tol_grad = opts.conv_tol_grad
                rd = driver(sd, bd, vq.monkhorst_pack(sd, list(mesh)), od,
                            progress=False, verbose=0)
                assert rd.converged
                energies.append(float(rd.energy))
            finite.append((energies[0] - energies[1])/(2*h))
        extrapolated = (4*finite[1] - finite[0])/3
        record_property("ecp_gradient", gradient.tolist())
        record_property("ecp_bond_fd", extrapolated)
        record_property("ecp_gradient_error", float(gradient[1, 2] - extrapolated))
        np.testing.assert_allclose(finite[0], finite[1], atol=2e-6, rtol=0)
        np.testing.assert_allclose(gradient[1, 2], extrapolated, atol=2e-6, rtol=0)
        exact = vq.run_uks(mol, basis, _molecular_options_with_ecp(vq.UKSOptions, data, functional="pbe"))
    else:
        exact = vq.run_uhf(mol, basis, _molecular_options_with_ecp(vq.UHFOptions, data))
    record_property("periodic_energy", float(result.energy))
    record_property("molecular_energy", float(exact.energy))
    record_property("molecular_limit_error", float(result.energy)-float(exact.energy))
    if is_ks:
        assert abs(float(result.energy) - float(exact.energy)) < 1e-5
    else:
        assert result.exchange_q0 == "BvK-ewald"
        reference = _CL2_EWALD_UHF_REFERENCE[mesh]
        record_property("periodic_reference_error", float(result.energy)-reference)
        assert float(result.energy) == pytest.approx(reference, abs=5e-8, rel=0)
        assert float(exact.energy) == pytest.approx(_CL2_MOLECULAR_UHF_REFERENCE, abs=1e-7, rel=0)
        # The 9.75/4.32 mHa molecular offset is finite-k exchange, not an
        # ECP operator error; never use it as an implementation tolerance.
        assert abs(float(result.energy)-float(exact.energy)) > 1e-3
    assert -30.0 < float(result.energy) < -29.0  # 14 valence electrons, not 34


def test_ecp_uhf_molecular_exchange_gauge_matches_orca(record_property):
    """The explicit molecular-limit exchange retains periodic electrostatics.

    Cl2 has no dipole, so the residual at 20 bohr is a few microhartree;
    removing the finite-k exchange convention does not remove image
    electrostatics. This is an isolated-molecule control, not a bulk gauge.
    """
    from vibeqc.periodic_uhf_multi_k_ewald import run_uhf_periodic_multi_k_ewald3d

    system, basis, data = _lanl2dz_molecule_in_box((17, 17), 1.99)
    result = run_uhf_periodic_multi_k_ewald3d(
        system, basis, vq.monkhorst_pack(system, [1, 1, 1]),
        _periodic_options_with_ecp(data), exchange_exxdiv="none",
        progress=False, verbose=0,
    )
    assert result.converged and result.exchange_q0 == "molecular-limit"
    error = float(result.energy)-_CL2_MOLECULAR_UHF_REFERENCE
    record_property("orca_molecular_limit_error", error)
    assert abs(error) < 5e-6
    assert _electron_trace(result.density_alpha_k[0], result.overlap[0]) == pytest.approx(7., abs=1e-8)
    assert _electron_trace(result.density_beta_k[0], result.overlap[0]) == pytest.approx(7., abs=1e-8)


@pytest.mark.parametrize("angular_momentum,omega", [(None, 0.), (None, .4), (1, .4), (2, .4)])
def test_periodic_jk_trace_replays_original_accumulators(angular_momentum, omega):
    """Observation must preserve the ordinary build and fail closed on limits."""
    core = vq._vibeqc_core
    atom = vq.Atom(1, np.array([0., 0., 0.]))
    system = vq.PeriodicSystem(3, np.eye(3) * 8., [atom])
    basis = vq.BasisSet(system.unit_cell_molecule(), "STO-3G")
    if angular_momentum is not None:
        # Exercise direct p kernels and the general d-shell engine with
        # signed densities, not only the scalar path's accumulator.
        basis = vq.BasisSet(system.unit_cell_molecule(), [
            vq.ShellInfo(0, 0, True, [.3, 1.1], [.8, -.2], [0., 0., 0.]),
            vq.ShellInfo(0, angular_momentum, True, [.4], [1.], [0., 0., 0.]),
        ], "trace-angular-test", False)
    options = vq.LatticeSumOptions()
    cutoff = 10. if angular_momentum is None else 3.
    options.cutoff_bohr = options.nuclear_cutoff_bohr = cutoff
    options.pair_complete_1e = True
    cells = core.physical_eri_lattice_cells(basis, system, cutoff, cutoff)
    nbf = int(basis.nbasis)
    rng = np.random.default_rng(193)
    blocks = [np.array([[.7]]) if angular_momentum is None
              else rng.normal(size=(nbf, nbf)) for _ in cells]
    density = core.make_lattice_matrix_set(nbf, cells, blocks)
    original = core.build_jk_2e_real_space(basis, system, options, density, omega=omega)
    traced = core.trace_jk_2e_real_space(basis, system, options, density, 100000, omega=omega)
    identities, values = traced.identities, traced.values
    assert len(identities) == len(values) > 0
    quartets = {tuple(map(int, row[:8])) + (int(row[12]),) for row in identities}
    assert len(quartets) == traced.matrices.integral_evaluations
    replay = [np.zeros_like(np.asarray(original.J.blocks)),
              np.zeros_like(np.asarray(original.K.blocks))]
    absent = np.iinfo(np.uint64).max
    for identity, value in zip(identities, values):
        g, _, _, _, _, _, _, _, mu, nu, _, _, channel, _ = map(int, identity)
        if mu == absent:
            continue
        integral, den, product, before, after = value
        assert product == den * integral
        assert replay[channel][g, mu, nu] == before
        replay[channel][g, mu, nu] = after
    for channel, name in enumerate(("J", "K")):
        np.testing.assert_array_equal(getattr(traced.matrices, name).blocks,
                                      getattr(original, name).blocks)
        np.testing.assert_array_equal(replay[channel], getattr(original, name).blocks)
    with pytest.raises(ValueError, match="positive record cap"):
        core.trace_jk_2e_real_space(basis, system, options, density, 0, omega=omega)
    with pytest.raises(ValueError, match="within 128 MiB"):
        core.trace_jk_2e_real_space(basis, system, options, density, 2**40, omega=omega)
    with pytest.raises((ValueError, RuntimeError), match="record cap exceeded"):
        core.trace_jk_2e_real_space(basis, system, options, density, 1, omega=omega)
    options.sr_sparse_traversal = False
    with pytest.raises(ValueError, match="ordinary shell-pair"):
        core.trace_jk_2e_real_space(basis, system, options, density, 100000, omega=omega)
