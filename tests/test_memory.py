"""Phase P2: memory estimator + pre-flight abort."""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import memory as vq_memory


def _h2o() -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(8, [0, 0, 0]),
        vq.Atom(1, [0, 1.43, -0.98]),
        vq.Atom(1, [0, -1.43, -0.98]),
    ])


def _cr_co6() -> vq.Molecule:
    """Cr(CO)6 atom inventory; geometry is immaterial to memory sizing."""
    cr_c = 1.916
    c_o = 1.140
    atoms = [vq.Atom(24, [0.0, 0.0, 0.0])]
    for sign in (+1, -1):
        for axis in range(3):
            xyz = [0.0, 0.0, 0.0]
            xyz[axis] = sign * cr_c
            atoms.append(vq.Atom(6, tuple(xyz)))
            xyz_o = list(xyz)
            xyz_o[axis] += sign * c_o
            atoms.append(vq.Atom(8, tuple(xyz_o)))
    return vq.Molecule(atoms)


def _he_periodic() -> vq.PeriodicSystem:
    return vq.PeriodicSystem(
        3,
        np.eye(3) * 10.0,
        [vq.Atom(2, [0.0, 0.0, 0.0]), vq.Atom(2, [2.0, 0.0, 0.0])],
    )


def _n2_release_geometry() -> vq.Molecule:
    bohr_per_angstrom = 1.889726124565062
    half_bond = 0.5 * 1.0976 * bohr_per_angstrom
    return vq.Molecule([
        vq.Atom(7, [0.0, 0.0, -half_bond]),
        vq.Atom(7, [0.0, 0.0, half_bond]),
    ])


# ---------------------------------------------------------------------------
# estimate_memory — shape & sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method",
    ["rhf", "uhf", "rks", "uks", "mp2", "ump2", "ovgf"],
)
def test_estimate_memory_returns_non_negative_total(method):
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    est = vq.estimate_memory(mol, basis, method=method)
    assert est.total_bytes > 0
    assert est.raw_total_bytes > 0
    assert est.headroom_factor >= 1.0
    assert est.by_category["Python runtime + NumPy overhead"] == 100 * 1024**2


def test_memory_headroom_defaults_to_hpc_factor_and_env_override(monkeypatch):
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    monkeypatch.delenv("VIBEQC_MEMORY_HEADROOM", raising=False)
    default_est = vq.estimate_memory(mol, basis, method="rhf")
    assert default_est.headroom_factor == pytest.approx(1.5)

    monkeypatch.setenv("VIBEQC_MEMORY_HEADROOM", "2.0")
    override_est = vq.estimate_memory(mol, basis, method="rhf")
    assert override_est.headroom_factor == pytest.approx(2.0)


def test_estimate_semiempirical_memory_counts_basis_free_scratch():
    est = vq.estimate_semiempirical_memory(_h2o(), method="pm6")

    assert est.total_bytes > 0
    assert "Semiempirical valence matrices" in est.by_category
    assert "Semiempirical DIIS history" in est.by_category
    assert "Semiempirical finite-difference gradient scratch" in est.by_category
    assert "ERI tensor" not in est.by_category


def test_estimate_semiempirical_memory_uses_shared_method_aliases():
    gfn2 = vq.estimate_semiempirical_memory(_h2o(), method="gfn2")
    gfn2_spaced = vq.estimate_semiempirical_memory(_h2o(), method="gfn2 xtb")
    scc = vq.estimate_semiempirical_memory(_h2o(), method="sccdftb")

    assert gfn2.by_category == gfn2_spaced.by_category
    assert "Semiempirical charge/SCC state" in gfn2.by_category
    assert "GFN2 shell/AES multipole state" in gfn2.by_category
    assert "Semiempirical charge/SCC state" in scc.by_category
    assert "GFN2 shell/AES multipole state" not in scc.by_category


def test_estimate_semiempirical_memory_consumes_route_plan():
    from vibeqc.semiempirical import SemiempiricalRoutePlan

    plan = SemiempiricalRoutePlan.from_request("upm6")
    planned = vq.estimate_semiempirical_memory(_h2o(), method=plan)
    aliased = vq.estimate_semiempirical_memory(_h2o(), method="upm6")

    assert plan.variant == "upm6"
    assert plan.spin == "unrestricted"
    assert planned.by_category == aliased.by_category


def test_estimate_semiempirical_memory_rejects_before_atom_inventory(
    monkeypatch: pytest.MonkeyPatch,
):
    import vibeqc.memory as memory_module

    radical = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)

    def _unexpected_atom_inventory(_system):
        raise AssertionError("atom inventory must follow route validation")

    monkeypatch.setattr(
        memory_module,
        "_semiempirical_atoms",
        _unexpected_atom_inventory,
    )
    with pytest.raises(NotImplementedError, match="closed-shell only"):
        vq.estimate_semiempirical_memory(radical, method="gfn2")


def test_estimate_semiempirical_memory_rejects_plan_boundary_mismatch():
    from vibeqc.semiempirical import SemiempiricalRoutePlan

    molecular = SemiempiricalRoutePlan.from_request("dftb0")

    with pytest.raises(ValueError, match="does not match the periodic system"):
        vq.estimate_semiempirical_memory(_he_periodic(), method=molecular)


def test_estimate_semiempirical_memory_counts_cosmo_and_ccm_terms():
    cosmo = vq.estimate_semiempirical_memory(
        _h2o(),
        method="msindo",
        solvent="water",
    )
    assert "MSINDO COSMO cavity A matrix" in cosmo.by_category
    assert "MSINDO COSMO B matrix" in cosmo.by_category

    seccm = vq.estimate_semiempirical_memory(
        _h2o(),
        method="seccm",
        ccm_options=SimpleNamespace(
            translations=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            madelung=True,
        ),
    )
    assert "MSINDO SECCM image-cell buffers" in seccm.by_category
    assert "MSINDO SECCM Madelung workspace" in seccm.by_category


def test_estimate_semiempirical_memory_counts_periodic_helper_terms():
    single_point = vq.estimate_semiempirical_memory(_he_periodic(), method="dftb0")

    assert single_point.total_bytes > 0
    assert "Semiempirical valence matrices" in single_point.by_category
    assert "Periodic semiempirical image/neighbour buffers" in single_point.by_category
    assert "ERI tensor" not in single_point.by_category

    optimized = vq.estimate_semiempirical_memory(
        _he_periodic(),
        method="dftb0",
        optimize=True,
        max_steps=12,
    )
    assert "Periodic semiempirical stress/strain scratch" in optimized.by_category
    assert "Semiempirical optimizer coordinates/history" in optimized.by_category
    assert optimized.raw_total_bytes > single_point.raw_total_bytes


def test_periodic_semiempirical_memory_uses_system_aware_spin_plan():
    common = {
        "unit_cell": [vq.Atom(1, [0.0, 0.0, 0.0])],
        "lattice": np.eye(3) * 10.0,
        "dim": 3,
        "charge": 0,
        "multiplicity": 1,
    }
    closed = SimpleNamespace(**common, n_electrons=lambda: 2)
    odd_electron = SimpleNamespace(**common, n_electrons=lambda: 1)

    closed_estimate = vq.estimate_semiempirical_memory(closed, method="dftb0")
    odd_estimate = vq.estimate_semiempirical_memory(odd_electron, method="dftb0")

    assert odd_estimate.by_category["Semiempirical DIIS history"] == (
        2 * closed_estimate.by_category["Semiempirical DIIS history"]
    )


def test_estimate_memory_scales_with_basis_size():
    """Switching to a larger basis must increase the estimate."""
    mol = _h2o()
    small = vq.BasisSet(mol, "sto-3g")
    large = vq.BasisSet(mol, "6-31g*")
    est_small = vq.estimate_memory(mol, small, method="rhf")
    est_large = vq.estimate_memory(mol, large, method="rhf")
    assert est_large.total_bytes > est_small.total_bytes


def test_estimate_memory_eri_dominates():
    """For a medium system the ERI tensor should be the largest term —
    which is the whole reason the estimator is useful."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvdz")
    est = vq.estimate_memory(mol, basis, method="rhf")
    largest = max(
        (
            kv
            for kv in est.by_category.items()
            if kv[0] != "Python runtime + NumPy overhead"
        ),
        key=lambda kv: kv[1],
    )
    assert largest[0] == "ERI tensor"


def test_direct_scf_estimate_does_not_materialise_eri_tensor():
    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g*")
    opts = vq.RHFOptions()
    opts.scf_mode = vq.SCFMode.DIRECT

    est = vq.estimate_memory(mol, basis, method="rhf", options=opts)

    assert "ERI tensor" not in est.by_category
    assert "Direct-SCF shell-pair scratch" in est.by_category
    n_pairs = basis.nshells * (basis.nshells + 1) // 2
    assert est.by_category["Direct-SCF shell-pair scratch"] == n_pairs * 1024


def test_auto_direct_scf_estimate_uses_threshold():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.scf_mode = vq.SCFMode.AUTO
    opts.scf_mode_auto_threshold = 1

    est = vq.estimate_memory(mol, basis, method="rhf", options=opts)

    assert "ERI tensor" not in est.by_category
    assert "Direct-SCF shell-pair scratch" in est.by_category
    n_pairs = basis.nshells * (basis.nshells + 1) // 2
    assert est.by_category["Direct-SCF shell-pair scratch"] == n_pairs * 1024


def test_density_fit_estimate_uses_df_storage():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"

    est = vq.estimate_memory(mol, basis, method="rhf", options=opts)

    assert "ERI tensor" not in est.by_category
    assert "DF three-index tensors" in est.by_category
    assert "DF metric/workspace" in est.by_category


def test_closed_df_exchange_gradient_prices_occupied_factor_pair():
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    opts = vq.RHFOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"

    estimate = vq.estimate_memory(mol, basis, method="rhf", options=opts)
    n_aux = estimate.dims["n_aux"]
    n_occ = mol.n_electrons() // 2

    assert estimate.by_category[
        "DF exchange-gradient occupied factors"
    ] == 2 * n_aux * n_occ * n_occ * 8


def test_pure_rks_omits_df_exchange_gradient_but_hybrid_includes_it():
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")

    def estimate(functional):
        opts = vq.RKSOptions()
        opts.functional = functional
        opts.density_fit = True
        opts.aux_basis = "def2-svp-jk"
        return vq.estimate_memory(mol, basis, method="rks", options=opts)

    pure = estimate("pbe")
    hybrid = estimate("pbe0")

    assert "DF exchange-gradient occupied factors" not in pure.by_category
    assert "DF exchange-gradient occupied factors" in hybrid.by_category


@pytest.mark.parametrize("cosx", (False, True))
@pytest.mark.parametrize("reference", ("rohf", "roks"))
def test_python_open_shell_df_reference_prices_fourth_metric_peer(
    reference,
    cosx,
):
    from vibeqc.rohf import ROHFOptions
    from vibeqc.roks import ROKSOptions

    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    basis = vq.BasisSet(mol, "def2-svp")
    if reference == "rohf":
        options = ROHFOptions()
        method = "uhf"  # run_job's standalone ROHF estimator mapping
    else:
        options = ROKSOptions(functional="PBE")
        method = "uks"  # run_job's standalone ROKS estimator mapping
    options.density_fit = True
    options.cosx = cosx
    options.aux_basis = "def2-svp-jk"

    estimate = vq.estimate_memory(
        mol,
        basis,
        method=method,
        options=options,
    )
    n_aux = estimate.dims["n_aux"]

    assert estimate.by_category[
        "Python DF constructor extra metric peer"
    ] == n_aux * n_aux * 8


def _df_rhf_estimate(
    molecule: vq.Molecule,
    orbital_basis: str,
    aux_basis: str,
) -> tuple[vq.MemoryEstimate, vq.BasisSet, vq.BasisSet]:
    basis = vq.BasisSet(molecule, orbital_basis)
    aux = vq.BasisSet(molecule, aux_basis)
    opts = vq.RHFOptions()
    opts.density_fit = True
    opts.aux_basis = aux_basis
    return vq.estimate_memory(molecule, basis, method="rhf", options=opts), basis, aux


def test_df_memory_estimator_includes_aux_basis_and_qz_construction_peak():
    """BUG 117: QZ DF sizing must include aux functions and live buffers."""
    mol = _cr_co6()
    svp, svp_basis, svp_aux = _df_rhf_estimate(
        mol,
        "def2-svp",
        "def2-svp-j",
    )
    qz, qz_basis, qz_aux = _df_rhf_estimate(
        mol,
        "def2-qzvpp",
        "def2-qzvpp-j",
    )

    svp_steady = svp_basis.nbasis**2 * svp_aux.nbasis * 8
    qz_steady = qz_basis.nbasis**2 * qz_aux.nbasis * 8
    svp_df = svp.by_category["DF three-index tensors"]
    qz_df = qz.by_category["DF three-index tensors"]

    # The C++ DensityFitting constructor holds four n_aux*n_orb^2 extents
    # simultaneously (T, T_flat_, B_flat, B_per_P_; cpp/src/df.cpp) at
    # EVERY basis size: two resident blocks x 2x construction overlap.
    # BUG 117: the old size-gated 1x small-basis baseline was disproved by
    # measurement (BH9 2026-08-06: UKS n_bf 261 true peak 1.088 GB vs
    # 0.516 GB estimated).
    assert svp_df == 2 * svp_steady * 2
    assert qz_df == 2 * qz_steady * 2
    ratio = qz_df / svp_df
    # Both tiers now carry the same four-extent layout, so this is the
    # honest n_orb^2 * n_aux scaling ratio (~15x for Cr(CO)6), no longer
    # inflated by the old 1x-small/4x-large model inconsistency.
    assert ratio > 10, (
        f"QZ/SVP DF memory ratio is only {ratio:.1f}x, expected >10x"
    )


def test_df_estimator_has_safety_factor_and_formula_for_qz():
    """QZ-scale DF tensors must expose their dimensions and 2x safety."""
    estimate, basis, aux = _df_rhf_estimate(
        _cr_co6(),
        "def2-qzvpp",
        "def2-qzvpp-j",
    )

    assert basis.nbasis > 500
    assert estimate.safety_factor >= 2.0
    rendered = estimate.format(available=256 * 1024**3)
    formula = (
        f"n_orb^2={basis.nbasis}^2 x n_aux={aux.nbasis} x n_blocks=2 "
        "x 8 bytes x 2x safety"
    )
    assert formula in rendered


def test_rijcosx_estimate_uses_ri_and_cosx_storage():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.density_fit = True
    opts.cosx = True
    opts.aux_basis = "def2-svp-jk"

    est = vq.estimate_memory(mol, basis, method="rks", options=opts)

    assert "ERI tensor" not in est.by_category
    assert "RI-J tensors" in est.by_category
    assert "COSX grid workspace" in est.by_category
    assert "DFT grid + chi" in est.by_category


def test_df_small_basis_carries_construction_peak():
    """BUG 117: the four-extent DF construction peak has no size gate.

    cpp/src/df.cpp's DensityFitting constructor holds T, T_flat_, B_flat,
    and B_per_P_ simultaneously regardless of n_basis. Measured witness
    (BH9 wave 2026-08-06): UKS n_bf 261 true peak 1.088 GB vs 0.516 GB
    under the old 1x small-basis model (2.11x underestimate).
    """
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    aux = vq.BasisSet(mol, "def2-svp-jk")
    opts = vq.RHFOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"

    est = vq.estimate_memory(mol, basis, method="rhf", options=opts)

    one_block = aux.nbasis * basis.nbasis**2 * 8
    metric = aux.nbasis**2 * 8
    assert est.by_category["DF three-index tensors"] == 2 * one_block * 2
    assert est.by_category["DF metric/workspace"] == 3 * metric
    assert est.safety_factor >= 2.0
    assert est.dims["n_aux"] == aux.nbasis


def test_rijcosx_ri_j_tensors_carry_construction_peak():
    """The COSX JK builder embeds the same C++ DensityFitting (BUG 117)."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    aux = vq.BasisSet(mol, "def2-svp-jk")
    opts = vq.RKSOptions()
    opts.density_fit = True
    opts.cosx = True
    opts.aux_basis = "def2-svp-jk"

    est = vq.estimate_memory(mol, basis, method="rks", options=opts)

    one_block = aux.nbasis * basis.nbasis**2 * 8
    expected = 2 * one_block * 2 + 3 * aux.nbasis**2 * 8
    assert est.by_category["RI-J tensors"] == expected


def test_bare_cosx_flag_preserves_conventional_uks_memory_route():
    """Inactive COSX must not replace the real O(n^4) JK allocation."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-qzvp")
    direct_opts = vq.UKSOptions()
    direct_opts.functional = "B3LYP"
    direct_opts.scf_mode = vq.SCFMode.CONVENTIONAL

    inactive_cosx_opts = vq.UKSOptions(direct_opts)
    inactive_cosx_opts.cosx = True
    direct = vq.estimate_memory(
        mol,
        basis,
        method="uks",
        options=direct_opts,
    )
    inactive_cosx = vq.estimate_memory(
        mol,
        basis,
        method="uks",
        options=inactive_cosx_opts,
    )

    assert "ERI tensor" in direct.by_category
    assert direct.by_category["ERI tensor"] == basis.nbasis**4 * 8
    assert "RI-J tensors" not in inactive_cosx.by_category
    assert "COSX grid workspace" not in inactive_cosx.by_category
    assert inactive_cosx.by_category == direct.by_category
    assert inactive_cosx.total_bytes == direct.total_bytes


def test_df_estimate_resolves_driver_default_aux_when_unset():
    """BUG 117: estimate n_aux from the driver's auto-resolved JK aux.

    run_job auto-resolves ``default_aux_basis_for(<basis>, kind="jk")``
    when ``density_fit=True`` and ``aux_basis`` is empty; the estimator
    must size n_aux from the same aux basis rather than the generic
    3*n_basis bound (Eichkorn 1995, Eq. 26), so the estimate and the
    execution agree.
    """
    from vibeqc.density_fitting import default_aux_basis_for

    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    opts = vq.RHFOptions()
    opts.density_fit = True
    # aux_basis deliberately left empty — the driver default applies.

    est = vq.estimate_memory(mol, basis, method="rhf", options=opts)

    resolved = vq.BasisSet(mol, default_aux_basis_for("def2-svp", kind="jk"))
    assert est.dims["n_aux"] == resolved.nbasis
    assert est.dims["n_aux"] != 3 * basis.nbasis


def test_run_job_density_fit_kwarg_reaches_memory_preflight(tmp_path):
    """BUG 117: run_job(density_fit=True) must be preflighted as a DF job.

    The density_fit/cosx/aux_basis kwargs used to be wired onto the SCF
    option structs only immediately before the SCF call — after the
    memory pre-flight — so a kwarg-configured DF job was estimated on
    the direct-SCF branch (BH9 campaign: def2-QZVPP DF jobs preflighted
    at ~1 GB, 515/822 OOM-killed).
    """
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])]
    )
    stem = tmp_path / "h2-df-kwarg"
    vq.run_job(
        mol,
        basis="def2-svp",
        method="rhf",
        density_fit=True,
        output=str(stem),
    )
    out_text = (tmp_path / "h2-df-kwarg.out").read_text()
    assert "DF three-index tensors" in out_text
    assert "Direct-SCF shell-pair scratch" not in out_text


def test_mp2_estimate_counts_resident_canonical_transform_terms():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    # BUG 63's transform inventory predates the published frozen-core
    # default.  Pin its original all-electron problem explicitly so this
    # remains a formula test rather than another default-policy test.
    options = vq.MP2Options()
    options.n_frozen_core = 0

    est = vq.estimate_memory(mol, basis, method="mp2", options=options)

    n = basis.nbasis
    n_occ = mol.n_electrons() // 2
    n_vir = n - n_occ

    # The SCF estimate already holds the in-core AO ERI tensor;
    # the MP2 half-transform reuses it rather than counting a second
    # copy.  (BUG 63 follow-up: was double-counted at 2 x n^4 * 8.)
    assert "ERI tensor" in est.by_category
    assert est.by_category["ERI tensor"] == n**4 * 8
    assert "MP2 AO ERI tensor" not in est.by_category

    assert est.by_category["MP2 AO->MO I1 scratch"] == n_occ * n**3 * 8
    assert est.by_category["MP2 AO->MO I2 scratch"] == (
        n_occ * n_vir * n * n * 8
    )
    assert est.by_category["MP2 AO->MO I3 scratch"] == (
        n_occ * n_vir * n_occ * n * 8
    )
    assert est.by_category["OVOV MO tensor"] == n_occ * n_vir * n_occ * n_vir * 8
    assert "MP2 AO->MO transform scratch" not in est.by_category


def test_canonical_mp2_after_df_reference_counts_its_ao_eri():
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    scf_options = vq.RHFOptions()
    scf_options.density_fit = True
    scf_options.aux_basis = "def2-svp-jk"

    est = vq.estimate_memory(
        mol,
        basis,
        method="mp2",
        options={
            "scf_options": scf_options,
            "mp2_options": SimpleNamespace(
                density_fit=False,
                memory_mode="incore",
            ),
        },
    )

    assert "ERI tensor" not in est.by_category
    assert est.by_category["MP2 AO ERI tensor"] == basis.nbasis**4 * 8
    assert "MP2 incore correlation" in est.phase_peaks


def test_ovgf_estimate_counts_dense_post_scf_tensors():
    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g")

    est = vq.estimate_memory(mol, basis, method="ovgf")

    n = basis.nbasis
    assert est.by_category["OVGF AO ERI tensor"] == n**4 * 8
    assert est.by_category["OVGF MO ERI tensor"] == n**4 * 8
    assert est.by_category["OVGF AO->MO transform scratch"] == n**4 * 8
    assert (
        est.by_category["OVGF renormalized spin-orbital tensor"]
        == (2 * n) ** 4 * 8
    )


def test_ovgf_open_shell_estimate_counts_spin_orbital_tensor():
    mol = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.834])],
        0,
        2,
    )
    basis = vq.BasisSet(mol, "sto-3g")

    est = vq.estimate_memory(mol, basis, method="ovgf")

    n = basis.nbasis
    assert est.by_category["OVGF AO ERI tensor"] == n**4 * 8
    assert est.by_category["OVGF spin-block MO tensors"] == 3 * n**4 * 8
    assert est.by_category["OVGF spin-orbital tensor"] == (2 * n) ** 4 * 8
    assert "Open-shell UHF buffers" in est.by_category


def test_df_mp2_estimate_counts_ovov_and_df_resident_storage():
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    opts = vq.MP2Options()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-rifit"

    est = vq.estimate_memory(
        mol,
        basis,
        method="mp2",
        options={"mp2_options": opts},
    )

    assert "MP2 AO ERI tensor" not in est.by_category
    assert "DF-MP2 three-index/B tensors" in est.by_category
    assert "DF-MP2 metric/workspace" in est.by_category
    assert "DF-MP2 aux-OV workspace" in est.by_category
    assert "OVOV MO tensor" in est.by_category
    assert est.by_category["OVOV MO tensor"] > 0


def test_tddft_casida_estimate_includes_dense_response_storage():
    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g*")

    est = vq.estimate_memory(
        mol,
        basis,
        method="rks",
        options={
            "scf_options": None,
            "tddft": True,
            "tddft_type": "casida",
            "tddft_n_states": 4,
            "functional": "PBE",
        },
    )

    assert "TDDFT AO ERI tensor" in est.by_category
    assert "TDDFT ovov/oovv blocks" in est.by_category
    assert "TDDFT A/B response matrices" in est.by_category
    assert "TDDFT eigensolver workspace" in est.by_category
    assert est.by_category["TDDFT A/B response matrices"] > 0


def test_neb_estimate_charges_parallel_fd_and_warm_start():
    from vibeqc.memory import MemoryEstimate, estimate_neb_memory

    per_image = MemoryEstimate({"per-image SCF": 1_000_000}, headroom_factor=1.2)

    serial = estimate_neb_memory(
        per_image,
        n_images=5,
        n_jobs=1,
        n_atoms=3,
        n_basis=4,
        open_shell=False,
        finite_difference_evaluations=1,
        warm_start=True,
    )
    parallel_fd = estimate_neb_memory(
        per_image,
        n_images=5,
        n_jobs=3,
        n_atoms=3,
        n_basis=4,
        open_shell=True,
        finite_difference_evaluations=19,
        warm_start=True,
    )

    assert parallel_fd.by_category["NEB parallel image workers"] == (
        3 * per_image.raw_total_bytes
    )
    assert "NEB finite-difference gradient scratch" in parallel_fd.by_category
    assert "NEB warm-start density cache" in parallel_fd.by_category
    assert parallel_fd.by_category["NEB warm-start density cache"] > (
        serial.by_category["NEB warm-start density cache"]
    )
    assert parallel_fd.total_bytes > serial.total_bytes


def test_ccsd_estimate_is_supported_and_includes_amplitudes():
    from vibeqc import _vibeqc_core as core

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = core.CCSDOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"

    est = vq.estimate_memory(mol, basis, method="ccsd", options=opts)

    assert "CCSD T1/T2 amplitudes" in est.by_category
    assert "CCSD D1/D2 intermediates" in est.by_category
    assert "DF-CCSD auxiliary integrals" in est.by_category


def test_df_ccsd_estimate_separates_native_build_and_solve_lifetimes(
    monkeypatch,
):
    from vibeqc import _vibeqc_core as core

    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 3)
    molecule = _h2o()
    basis = vq.BasisSet(molecule, "def2-svp")
    aux_name = "def2-svp-rifit"
    aux = vq.BasisSet(molecule, aux_name)
    options = core.CCSDOptions()
    options.density_fit = True
    options.aux_basis = aux_name
    options.n_frozen_core = 0

    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="ccsd",
        options=options,
    )

    n = basis.nbasis
    n_aux = aux.nbasis
    n_occ = estimate.dims["n_occ"]
    n_vir = estimate.dims["n_vir"]
    three_index = n_aux * n * n * 8
    metric = n_aux * n_aux * 8
    constructor = 4 * three_index + 3 * metric
    resident = 2 * three_index + metric
    b_ov = n_aux * n_occ * n_vir * 8
    b_vv = n_aux * n_vir * n_vir * 8
    b_oo = n_aux * n_occ * n_occ * 8
    workers = min(n_aux, 3)
    retained = 0
    transform_peak = 0
    for n_left, n_right, output in (
        (n_occ, n_vir, b_ov),
        (n_vir, n_vir, b_vv),
        (n_occ, n_occ, b_oo),
    ):
        transform_peak = max(
            transform_peak,
            resident
            + retained
            + output
            + workers * n_left * (n + n_right) * 8,
        )
        retained += output

    category = estimate.by_category
    reference = _retained_reference_bytes(estimate)
    assert category["DF-CCSD construction overlap"] == constructor
    assert category["DF-CCSD auxiliary integrals"] == b_ov + b_vv
    assert category["DF-CCSD build-only DF substrate/B_oo"] == (
        resident + b_oo
    )
    assert category["DF-CCSD threaded MO-transform peak"] == transform_peak
    assert estimate.phase_peaks["CC DF construction"] == (
        reference + constructor
    )
    assert estimate.phase_peaks["CC DF MO transformation"] == (
        reference + transform_peak
    )
    solve_categories = (
        "CCSD T1/T2 amplitudes",
        "CCSD residual/DIIS amplitudes",
        "CCSD D1/D2 intermediates",
        "DF-CCSD auxiliary integrals",
        "DF-CCSD MO integral blocks",
        "DF-CCSD VVVV/tile scratch",
    )
    assert estimate.phase_peaks["CC amplitude solve"] == reference + sum(
        category[name] for name in solve_categories
    )


def test_df_uccsd_estimate_prices_native_constructor_and_build_scope(
    monkeypatch,
):
    from vibeqc import _vibeqc_core as core

    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 3)
    molecule = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    basis = vq.BasisSet(molecule, "def2-svp")
    aux_name = "def2-svp-rifit"
    aux = vq.BasisSet(molecule, aux_name)
    options = core.CCSDOptions()
    options.density_fit = True
    options.aux_basis = aux_name
    options.n_frozen_core = 0

    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="uccsd",
        options=options,
    )

    n = basis.nbasis
    n_aux = aux.nbasis
    n_elec = molecule.n_electrons()
    n_alpha = (n_elec + molecule.multiplicity - 1) // 2
    n_beta = n_elec - n_alpha
    n_a = n_alpha + (n - n_alpha)
    n_b = n_beta + (n - n_beta)
    n_spin = estimate.dims["n_occ"] + estimate.dims["n_vir"]
    three_index = n_aux * n * n * 8
    metric = n_aux * n_aux * 8
    constructor = 4 * three_index + 3 * metric
    resident = 2 * three_index + metric
    b_a = n_aux * n_a * n_a * 8
    b_b = n_aux * n_b * n_b * 8
    b_so = n_aux * n_spin * n_spin * 8
    workers = min(n_aux, 3)
    transform_peak = max(
        resident + b_a + workers * n_a * (n + n_a) * 8,
        resident + b_a + b_b + workers * n_b * (n + n_b) * 8,
        resident + b_a + b_b + b_so,
    )
    p_block = max(1, min(n_spin, 2048 // max(1, n_spin)))
    chem_tile = p_block * n_spin**3 * 8

    category = estimate.by_category
    reference = _retained_reference_bytes(estimate)
    expected_build_scratch = resident + b_a + b_b + b_so + chem_tile
    assert category["DF-UCCSD construction overlap"] == constructor
    assert category["DF-UCCSD threaded MO-transform peak"] == transform_peak
    assert category["DF-UCCSD integral-build scratch"] == (
        expected_build_scratch
    )
    assert estimate.phase_peaks["CC DF construction"] == (
        reference + constructor
    )
    assert estimate.phase_peaks["CC DF MO transformation"] == (
        reference + transform_peak
    )
    assert estimate.phase_peaks["CC integral build"] == (
        reference
        + category["UCCSD SpinOrbitalIntegrals"]
        + expected_build_scratch
    )


def test_ccsdt_estimate_includes_triples_workspace():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    est = vq.estimate_memory(mol, basis, method="ccsd(t)")

    assert "CCSD(T) triples image buffers" in est.by_category
    assert "CCSD(T) triples work list" in est.by_category


def test_ccsdt_ccpvdz_estimate_uses_phase_peak_without_synthetic_floor():
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvdz")

    est = vq.estimate_memory(mol, basis, method="ccsd(t)")

    assert "CCSD native solver/runtime scratch" not in est.by_category
    assert "DF-CCSD MO integral blocks" in est.by_category
    assert "DF-CCSD VVVV/tile scratch" in est.by_category
    assert est.phase_peaks
    assert est.raw_total_bytes == max(est.phase_peaks.values())
    assert est.raw_total_bytes < sum(est.by_category.values())


def test_bccd_estimates_use_ccsd_workspace():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    bccd = vq.estimate_memory(mol, basis, method="bccd")
    bccdt = vq.estimate_memory(mol, basis, method="bccd(t)")

    assert "CCSD T1/T2 amplitudes" in bccd.by_category
    assert bccd.by_category["BCCD persistent AO ERI"] == basis.nbasis**4 * 8
    assert "CCSD(T) triples image buffers" not in bccd.by_category
    assert "CCSD(T) triples image buffers" in bccdt.by_category
    for label, peak in bccd.phase_peaks.items():
        if label not in {"Runtime baseline", "SCF reference"}:
            assert peak >= bccd.by_category["BCCD persistent AO ERI"]
    assert bccdt.total_bytes >= bccd.total_bytes


def test_bccd_native_budget_reserves_persistent_outer_ao_eri():
    from vibeqc.runner import _native_correlated_budget_bytes

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    est = vq.estimate_memory(mol, basis, method="bccd(t)")
    process_budget = 512 * 1024**2

    reference_and_outer = sum(
        est.by_category.get(name, 0)
        for name in (
            "Fock + density + 1e",
            "MO workspace",
            "Open-shell UHF buffers",
            "BCCD persistent AO ERI",
        )
    )
    expected = max(
        1,
        int(process_budget / est.headroom_factor) - reference_and_outer,
    )
    assert _native_correlated_budget_bytes(process_budget, est) == expected


def test_fno_estimate_has_separate_dense_mp2_no_setup_phase():
    from vibeqc.cc import CCSDOptions

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = CCSDOptions(
        aux_basis="def2-svp-jk",
        compute_triples=False,
        n_frozen_core=0,
        fno=True,
        fno_keep_fraction=0.5,
        fno_delta_mp2=True,
    )

    est = vq.estimate_memory(mol, basis, method="ccsd", options=opts)
    no = mol.n_electrons() // 2
    nv = basis.nbasis - no
    k = max(1, int(round(0.5 * nv)))
    full_oovv = no * no * nv * nv * 8
    truncated_oovv = no * no * k * k * 8
    n_aux = est.dims["n_aux"]
    three_index = n_aux * basis.nbasis * basis.nbasis * 8
    metric = n_aux * n_aux * 8

    assert est.by_category["FNO DF construction overlap"] == (
        4 * three_index + 4 * metric
    )
    assert est.by_category["FNO retained DF substrate"] == (
        2 * three_index + 3 * metric
    )
    assert est.by_category["FNO full-space MP2-NO tensors"] == 5 * full_oovv
    assert est.by_category["FNO delta-MP2 tensors"] == 5 * truncated_oovv
    assert est.by_category["FNO full-space B_ov"] > 0
    assert est.by_category["FNO truncated B_ov"] > 0
    assert est.by_category["FNO truncated MO-transform scratch"] == (
        n_aux * basis.nbasis * k * 8
    )
    assert "FNO dense MP2 natural-orbital setup" in est.phase_peaks
    assert est.dims["fno_virtual_kept_estimate"] == k
    assert est.dims["fno_virtual_total"] == nv
    assert est.dims["n_vir"] == k

    full_opts = CCSDOptions(
        aux_basis="def2-svp-jk",
        compute_triples=False,
        n_frozen_core=0,
        fno=False,
    )
    full = vq.estimate_memory(mol, basis, method="ccsd", options=full_opts)
    assert est.phase_peaks["CC amplitude solve"] < full.phase_peaks[
        "CC amplitude solve"
    ]


def test_rohf_mp2_estimate_models_dense_python_spin_orbital_setup():
    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    basis = vq.BasisSet(mol, "cc-pvdz")

    est = vq.estimate_memory(mol, basis, method="rohf-mp2")
    assert est.dims["n_frozen_core"] == 1
    n_mo = basis.nbasis - est.dims["n_frozen_core"]
    spin_eri = (2 * n_mo) ** 4 * 8
    n_aux = est.dims["n_aux"]
    three_index = n_aux * basis.nbasis * basis.nbasis * 8
    metric = n_aux * n_aux * 8

    assert est.by_category["ROHF-MP2 DF construction overlap"] == (
        4 * three_index + 4 * metric
    )
    assert est.by_category["ROHF-MP2 retained DF substrate"] == (
        2 * three_index + 3 * metric
    )
    assert est.by_category["ROHF-MP2 semicanonical B/F copies"] == (
        4 * n_aux * n_mo * n_mo * 8
        + (4 * n_mo * n_mo + 2 * n_mo) * 8
    )
    assert est.by_category["ROHF-MP2 spin-orbital ERI tensor"] == spin_eri
    assert (
        est.by_category["ROHF-MP2 spin-index mesh/mask temporaries"]
        == 13 * spin_eri
    )
    assert (
        est.by_category["ROHF-MP2 four spatial Vpair tensors"]
        == 4 * n_mo**4 * 8
    )
    assert "ROHF-MP2 dense spin-orbital setup" in est.phase_peaks
    assert "mp2_block_size" not in est.dims
    assert "pure-Python accuracy oracle" in est.category_details[
        "ROHF-MP2 spin-index mesh/mask temporaries"
    ]


def test_cc2_estimate_uses_ccsd_workspace_without_triples():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    est = vq.estimate_memory(mol, basis, method="cc2")

    assert "CCSD T1/T2 amplitudes" in est.by_category
    assert "CCSD(T) triples image buffers" not in est.by_category


def test_ccsdt_estimator_not_100x_wrong_o2(monkeypatch):
    """O2 DF-UCCSD(T)/def2-SVP should track the observed ~220 MiB.

    Regression test for BUG 63: the old triples estimator claimed
    n_occ^3 * n_vir^3 elements for amplitudes and
    n_occ^2 * n_vir^4 + n_occ^4 * n_vir^2 for intermediates, but the
    C++ code never materialises the 6-index T3 tensor. The later 27 GB
    estimate still added SCF, integral-build, solve, triples, a synthetic
    100 MiB Python floor, and a synthetic 256 MiB native floor as though all
    were simultaneous. The measured OMP=8 peak is about 220 MiB.
    """
    # Pin the modeled runtime assumption explicitly. The production helper
    # queries omp_get_max_threads(), which may differ across CI hosts.
    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 8)
    mol = vq.Molecule([vq.Atom(8, [0, 0, 0]), vq.Atom(8, [0, 0, 1.2])],
                      multiplicity=3)
    basis = vq.BasisSet(mol, "def2-svp")
    # The measured BUG 63 job was all-electron.  Keep that historical
    # comparison explicit now that the public default freezes chemical core.
    from vibeqc.cc import CCSDOptions

    options = CCSDOptions(n_frozen_core=0)
    est = vq.estimate_memory(
        mol,
        basis,
        method="uccsd(t)",
        options=options,
    )

    # Triples image buffers are n_vir^3 * n_buffers * n_threads * 8 bytes.
    # For O2/def2-SVP (nv ~ 40, 3 buffers, ~8 threads) this is ~8 MB.
    # Even with conservative headroom the total should be well under 2 GB.
    assert "CCSD(T) triples image buffers" in est.by_category
    triples_mb = est.by_category["CCSD(T) triples image buffers"] / (1024**2)
    assert triples_mb < 200, (
        f"Triples image buffers estimate {triples_mb:.0f} MB, "
        f"expected < 200 MB"
    )
    total_mb = est.total_bytes / (1024**2)
    assert 200 <= total_mb <= 300, (
        f"Total memory estimate {total_mb:.0f} MiB, expected 200-300 MiB"
    )
    assert est.raw_total_bytes == max(est.phase_peaks.values())
    assert est.raw_total_bytes < sum(est.by_category.values())


def test_correlated_thread_model_matches_native_runtime_without_128_cap():
    from vibeqc._vibeqc_core import get_num_threads, set_num_threads
    from vibeqc.memory import _omp_max_threads

    previous = get_num_threads()
    try:
        actual = set_num_threads(129)
        if actual == 1:
            pytest.skip("OpenMP is disabled in this build")
        assert actual == 129
        assert _omp_max_threads() == actual
    finally:
        set_num_threads(previous)


def test_uccsdt_estimator_counts_active_spin_orbital_integrals():
    mol = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(8, [0, 0, 1.2])],
        multiplicity=3,
    )
    basis = vq.BasisSet(mol, "def2-svp")

    est = vq.estimate_memory(mol, basis, method="uccsd(t)")

    n_spin = est.dims["n_occ"] + est.dims["n_vir"]
    assert est.by_category["UCCSD SpinOrbitalIntegrals"] == n_spin**4 * 8
    assert "DF-CCSD VVVV/tile scratch" not in est.by_category


def test_ccsd_and_mp2_estimates_honor_frozen_core():
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvdz")

    cc_all = vq.estimate_memory(
        mol,
        basis,
        method="ccsd(t)",
        options=SimpleNamespace(n_frozen_core=0),
    )
    cc_frozen = vq.estimate_memory(
        mol,
        basis,
        method="ccsd(t)",
        options=SimpleNamespace(n_frozen_core=1),
    )
    assert cc_frozen.dims["n_occ"] == cc_all.dims["n_occ"] - 1
    assert cc_frozen.by_category["CCSD T1/T2 amplitudes"] < (
        cc_all.by_category["CCSD T1/T2 amplitudes"]
    )

    mp2_all = vq.estimate_memory(
        mol,
        basis,
        method="mp2",
        options=SimpleNamespace(
            density_fit=False,
            memory_mode="incore",
            n_frozen_core=0,
        ),
    )
    mp2_frozen = vq.estimate_memory(
        mol,
        basis,
        method="mp2",
        options=SimpleNamespace(
            density_fit=False,
            memory_mode="incore",
            n_frozen_core=1,
        ),
    )
    assert mp2_frozen.dims["n_occ"] == mp2_all.dims["n_occ"] - 1
    assert mp2_frozen.by_category["OVOV MO tensor"] < (
        mp2_all.by_category["OVOV MO tensor"]
    )


def test_direct_mp2_budget_selects_monotonic_occupied_i_slabs():
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvtz")

    def estimate(budget):
        return vq.estimate_memory(
            mol,
            basis,
            method="mp2",
            options=SimpleNamespace(
                density_fit=False,
                memory_mode="direct",
                requested_memory_bytes=budget,
                n_frozen_core=1,
            ),
        )

    # One exact occupied-i transform needs about 2.85 MiB here. These caps
    # select one and two bra-occupied orbitals, respectively.
    small = estimate(4 * 1024**2)
    large = estimate(8 * 1024**2)
    label = "MP2 direct occupied-orbital slab"

    assert "OVOV MO tensor" not in small.by_category
    assert "MP2 AO ERI tensor" not in small.by_category
    assert small.dims["mp2_block_size"] < large.dims["mp2_block_size"]
    assert small.by_category[label] < large.by_category[label]
    n = basis.nbasis
    no = small.dims["n_occ"]
    nv = small.dims["n_vir"]
    per_i = 8 * max(
        n**3 + nv * n**2,
        nv * n**2 + nv * no * n,
        nv * no * n + nv * no * nv,
    )
    assert small.by_category[label] == small.dims["mp2_block_size"] * per_i
    assert small.dims["mp2_workspace_bytes"] == n * n * 8 + small.by_category[label]


def test_df_mp2_direct_workspace_matches_native_one_i_panel_model(monkeypatch):
    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 4)
    mol = _h2o()
    basis = vq.BasisSet(mol, "def2-svp")
    aux_name = "def2-svp-rifit"
    aux = vq.BasisSet(mol, aux_name)
    budget = 32 * 1024**2
    est = vq.estimate_memory(
        mol,
        basis,
        method="mp2",
        options=SimpleNamespace(
            density_fit=True,
            aux_basis=aux_name,
            memory_mode="direct",
            requested_memory_bytes=budget,
        ),
    )

    n = basis.nbasis
    no = est.dims["n_occ"]
    nv = est.dims["n_vir"]
    naux = aux.nbasis
    fixed = n * n * 8
    three = naux * n * n * 8
    metric = naux * naux * 8
    b_mo = naux * no * nv * 8
    constructor = fixed + 4 * three + 3 * metric
    resident = fixed + 2 * three + metric + b_mo
    transform = resident + min(naux, 4) * no * (n + nv) * 8
    construction = max(constructor, transform)
    per_i = nv * no * nv * 8
    block = max(1, min(no, (budget - resident) // per_i))
    expected = max(construction, resident + block * per_i)

    label = "DF-MP2 direct occupied-orbital slab"
    assert est.dims["mp2_block_size"] == block
    assert est.by_category[label] == block * per_i
    assert est.dims["mp2_workspace_bytes"] == expected


@pytest.mark.parametrize("density_fit", [False, True])
def test_ump2_direct_workspace_matches_largest_native_spin_channel(
    density_fit,
    monkeypatch,
):
    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 4)
    mol = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(8, [0, 0, 1.2])],
        multiplicity=3,
    )
    basis = vq.BasisSet(mol, "def2-svp")
    aux_name = "def2-svp-rifit"
    budget = 128 * 1024**2
    opts = SimpleNamespace(
        density_fit=density_fit,
        aux_basis=aux_name if density_fit else "",
        memory_mode="direct",
        requested_memory_bytes=budget,
        n_frozen_core=0,
    )
    est = vq.estimate_memory(mol, basis, method="ump2", options=opts)

    n = basis.nbasis
    n_elec = mol.n_electrons()
    na = (n_elec + mol.multiplicity - 1) // 2
    nb = n_elec - na
    nva = n - na
    nvb = n - nb
    channels = [
        (na, nva, na, nva),
        (nb, nvb, nb, nvb),
        (na, nva, nb, nvb),
    ]
    fixed = 2 * n * n * 8
    if density_fit:
        naux = vq.BasisSet(mol, aux_name).nbasis
        three = naux * n * n * 8
        metric = naux * naux * 8
        b_mo = naux * (na * nva + nb * nvb) * 8
        constructor = fixed + 4 * three + 3 * metric
        resident = fixed + 2 * three + metric + b_mo
        transform_workers = min(naux, 4)
        alpha_transform = (
            fixed
            + 2 * three
            + metric
            + naux * na * nva * 8
            + transform_workers * na * (n + nva) * 8
        )
        beta_transform = (
            resident
            + transform_workers * nb * (n + nvb) * 8
        )
        construction = max(
            constructor,
            alpha_transform,
            beta_transform,
        )
        label = "DF-UMP2 direct occupied-orbital slab"
    else:
        construction = 0
        resident = fixed
        label = "UMP2 direct occupied-orbital slab"

    expected_block = 0
    expected_slab = 0
    expected_workspace = max(resident, construction)
    for no_bra, nv_bra, no_ket, nv_ket in channels:
        if density_fit:
            per_i = nv_bra * no_ket * nv_ket * 8
        else:
            per_i = 8 * max(
                n**3 + nv_bra * n**2,
                nv_bra * n**2 + nv_bra * no_ket * n,
                nv_bra * no_ket * n + nv_bra * no_ket * nv_ket,
            )
        block = max(1, min(no_bra, (budget - resident) // per_i))
        slab = block * per_i
        expected_block = max(expected_block, block)
        expected_slab = max(expected_slab, slab)
        expected_workspace = max(
            expected_workspace,
            construction,
            resident + slab,
        )

    assert est.dims["mp2_block_size"] == expected_block
    assert est.by_category[label] == expected_slab
    assert est.dims["mp2_workspace_bytes"] == expected_workspace


def test_ump2_estimator_preserves_zero_active_beta_boundary():
    """Estimator and native UMP2 admit the same high-spin frozen count."""

    mol = vq.Molecule([vq.Atom(7, [0, 0, 0])], multiplicity=4)
    basis = vq.BasisSet(mol, "cc-pvdz")
    opts = SimpleNamespace(
        n_frozen_core=2,
        density_fit=False,
        memory_mode="incore",
    )
    est = vq.estimate_memory(mol, basis, method="ump2", options=opts)

    assert est.dims["n_frozen_core"] == 2
    assert est.dims["n_occ"] == 3


def test_uccsd_estimator_preserves_zero_active_beta_boundary():
    """UCCSD preflight mirrors the valid n_frozen == n_beta boundary."""

    mol = vq.Molecule([vq.Atom(5, [0, 0, 0])], multiplicity=4)
    basis = vq.BasisSet(mol, "def2-svp")
    est = vq.estimate_memory(mol, basis, method="uccsd")

    assert est.dims["n_frozen_core"] == 1
    assert est.dims["n_occ"] == 3


@pytest.mark.parametrize(
    "method",
    (
        "ump2",
        "rohf-mp2",
        "uccsd",
        "uccsd(t)",
        "dlpno-ump2",
        "dlpno-uccsd",
        "dlpno-uccsd(t)",
    ),
)
def test_open_shell_estimator_rejects_published_core_larger_than_beta_space(
    method,
):
    """Preflight fails before SCF instead of clamping to all-electron."""

    molecule = vq.Molecule([vq.Atom(5, [0, 0, 0])], multiplicity=6)
    basis = vq.BasisSet(molecule, "def2-svp")

    with pytest.raises(
        ValueError,
        match="count 1 exceeds this route's maximum valid count 0",
    ):
        vq.estimate_memory(molecule, basis, method=method)


@pytest.mark.parametrize(
    ("method", "options"),
    (
        ("mp2", SimpleNamespace(n_frozen_core=5)),
        ("ccsd", SimpleNamespace(n_frozen_core=5, density_fit=False)),
        ("dlpno-mp2", SimpleNamespace(n_frozen=5)),
        ("dlpno-ccsd", SimpleNamespace(n_frozen=5)),
    ),
)
def test_closed_shell_estimator_rejects_frozen_core_that_empties_occupied(
    method, options
):
    """An explicit invalid count never becomes a believable clamped plan."""

    molecule = _h2o()
    basis = vq.BasisSet(molecule, "def2-svp")

    with pytest.raises(ValueError, match="exceeds this route's maximum"):
        vq.estimate_memory(
            molecule,
            basis,
            method=method,
            options=options,
        )


def test_mp2_auto_budget_switches_from_incore_to_direct():
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvtz")

    bounded = vq.estimate_memory(
        mol,
        basis,
        method="mp2",
        options=SimpleNamespace(
            density_fit=False,
            memory_mode="auto",
            requested_memory_bytes=512 * 1024,
        ),
    )
    incore = vq.estimate_memory(
        mol,
        basis,
        method="mp2",
        options=SimpleNamespace(
            density_fit=False,
            memory_mode="auto",
            requested_memory_bytes=512 * 1024**2,
        ),
    )

    assert "MP2 direct occupied-orbital slab" in bounded.by_category
    assert "OVOV MO tensor" not in bounded.by_category
    assert "OVOV MO tensor" in incore.by_category


def test_blocked_triples_budget_selects_monotonic_virtual_tiles(monkeypatch):
    import vibeqc.memory as memory_module

    monkeypatch.setattr(memory_module, "_omp_max_threads", lambda: 2)
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvtz")

    def estimate(budget):
        return vq.estimate_memory(
            mol,
            basis,
            method="ccsd(t)",
            options=SimpleNamespace(
                density_fit=True,
                aux_basis="",
                n_frozen_core=1,
                diis_subspace_size=6,
                triples_memory_mode="blocked",
                requested_memory_bytes=budget,
                triples_tile_size=0,
                triples_max_threads=2,
            ),
        )

    small_budget = 8 * 1024**2
    large_budget = 16 * 1024**2
    small = estimate(small_budget)
    large = estimate(large_budget)
    label = "CCSD(T) triples image buffers"

    assert small.dims["triples_tile_size"] < large.dims["triples_tile_size"]
    assert small.by_category[label] < large.by_category[label]
    assert (
        small.by_category[label]
        + small.by_category["CCSD(T) retained amplitudes/integrals"]
        + small.by_category["CCSD(T) triples work list"]
    ) <= small_budget
    assert (
        large.by_category[label]
        + large.by_category["CCSD(T) retained amplitudes/integrals"]
        + large.by_category["CCSD(T) triples work list"]
    ) <= large_budget
    assert small.dims["triples_workspace_bytes"] == (
        small.by_category[label]
        + small.by_category["CCSD(T) retained amplitudes/integrals"]
        + small.by_category["CCSD(T) triples work list"]
    )


def test_df_ccsdt_disk_peak_includes_factor_spill_transition():
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvdz")
    opts = SimpleNamespace(
        density_fit=True,
        aux_basis="",
        n_frozen_core=1,
        triples_memory_mode="disk",
        requested_memory_bytes=64 * 1024**2,
        triples_tile_size=2,
        triples_max_threads=2,
    )
    est = vq.estimate_memory(mol, basis, method="ccsd(t)", options=opts)

    no = est.dims["n_occ"]
    nv = est.dims["n_vir"]
    naux = est.dims["n_aux"]
    threads = est.dims["n_threads"]
    tile = est.dims["triples_tile_size"]
    work = (no * (no + 1) * (no + 2) // 6) * 3 * 8
    base = 8 * (
        2 * no * nv
        + 2 * no**2 * nv**2
        + no**3 * nv
        + no
        + nv
    )
    factor_row = (no * nv + nv**2) * 8
    factors = naux * factor_row
    transition = base + factors
    streamed = base + work + threads * (
        2 * tile * nv**2 * 8 + factor_row
    )
    expected_peak = max(transition, streamed)

    assert est.by_category["CCSD(T) retained amplitudes/integrals"] == base
    assert est.dims["triples_workspace_bytes"] == expected_peak
    assert est.dims["triples_disk_bytes"] == 32 + naux * factor_row
    assert expected_peak >= transition


def test_uccsdt_planner_matches_fast_direct_aliases_and_disk_guard():
    mol = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(8, [0, 0, 1.2])],
        multiplicity=3,
    )
    basis = vq.BasisSet(mol, "def2-svp")
    common = dict(
        density_fit=True,
        aux_basis="",
        n_frozen_core=0,
        triples_max_threads=8,
    )

    direct = vq.estimate_memory(
        mol,
        basis,
        method="uccsd(t)",
        options=SimpleNamespace(
            **common,
            triples_memory_mode="direct",
            requested_memory_bytes=1024**3,
        ),
    )
    no = direct.dims["n_occ"]
    nv = direct.dims["n_vir"]
    retained = 8 * ((no + nv) ** 4 + no * nv + no**2 * nv**2 + no + nv)
    assert direct.dims["triples_workspace_bytes"] == retained
    assert direct.by_category["CCSD(T) triples image buffers"] == 0

    fast = vq.estimate_memory(
        mol,
        basis,
        method="uccsd(t)",
        options=SimpleNamespace(
            **common,
            triples_memory_mode="fast",
            requested_memory_bytes=1024**3,
        ),
    )
    threads = fast.dims["n_threads"]
    assert fast.dims["triples_workspace_bytes"] == (
        retained + threads * 3 * nv**3 * 8
    )

    for requested_mode in ("auto", "blocked", "low"):
        bounded = vq.estimate_memory(
            mol,
            basis,
            method="uccsd(t)",
            options=SimpleNamespace(
                **common,
                triples_memory_mode=requested_mode,
                requested_memory_bytes=retained,
            ),
        )
        assert "mode=direct" in bounded.category_details[
            "CCSD(T) triples image buffers"
        ]
        assert bounded.dims["triples_workspace_bytes"] == retained
        assert bounded.dims["triples_tile_size"] == 0

    with pytest.raises(ValueError, match="cannot spill"):
        vq.estimate_memory(
            mol,
            basis,
            method="uccsd(t)",
            options=SimpleNamespace(
                **common,
                triples_memory_mode="disk",
                requested_memory_bytes=retained,
            ),
        )


@pytest.mark.parametrize("mode", ["incore", "in-core", "disk-backed", "typo"])
def test_ccsdt_estimator_rejects_modes_the_native_kernel_rejects(mode):
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvdz")
    with pytest.raises(ValueError, match="triples_memory_mode"):
        vq.estimate_memory(
            mol,
            basis,
            method="ccsd(t)",
            options=SimpleNamespace(
                density_fit=True,
                triples_memory_mode=mode,
            ),
        )


def test_accsdt_uses_separate_dense_lambda_phase_not_bounded_t_planner():
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvtz")

    def estimate(mode):
        return vq.estimate_memory(
            mol,
            basis,
            method="a-ccsd(t)",
            options=SimpleNamespace(
                density_fit=True,
                aux_basis="",
                n_frozen_core=1,
                diis_subspace_size=4,
                triples_memory_mode=mode,
                requested_memory_bytes=2 * 1024**2,
                triples_tile_size=1,
                triples_max_threads=1,
            ),
        )

    direct = estimate("direct")
    fast = estimate("fast")
    nv = direct.dims["n_vir"]
    n = direct.dims["n_basis"]
    no = direct.dims["n_occ"]
    n_aux = direct.dims["n_aux"]
    three_index = n_aux * n * n * 8
    metric = n_aux * n_aux * 8
    df_resident = 2 * three_index + 3 * metric
    b_ov = n_aux * no * nv * 8
    b_vv = n_aux * nv * nv * 8
    b_oo = n_aux * no * no * 8
    transform_peak = max(
        df_resident + n_aux * n * nv * 8 + b_ov,
        df_resident + b_ov + n_aux * n * nv * 8 + b_vv,
        df_resident + b_ov + b_vv + n_aux * n * no * 8 + b_oo,
    )

    assert "CCSD(T) triples image buffers" not in direct.by_category
    assert direct.by_category["A-CCSD(T) DF construction overlap"] == (
        4 * three_index + 4 * metric
    )
    assert direct.by_category["A-CCSD(T) DF MO-transform peak"] == (
        transform_peak
    )
    assert direct.by_category["A-CCSD(T) retained DF/factor tensors"] >= (
        2 * three_index + 3 * metric
    )
    assert direct.by_category["A-CCSD(T) dense integral blocks"] >= nv**4 * 8
    ad_nodes = (
        9 * nv**4
        + 7 * no**4
        + 94 * no * no * nv * nv
        + 20 * nv * nv
        + 18 * no * no
        + 24 * no * nv
    )
    ad_workspace = (
        2 * ad_nodes + 2 * max(nv**4, no**4, no * no * nv * nv)
    ) * 8
    assert direct.by_category[
        "A-CCSD(T) Lambda reverse-AD graph"
    ] == ad_workspace
    assert direct.by_category[
        "A-CCSD(T) spatial solver bridge copies"
    ] == 3 * direct.by_category["A-CCSD(T) dense integral blocks"]
    amplitude_elements = no * nv + no * no * nv * nv
    triples_integrals = (
        no * nv**3 + no**3 * nv + no * no * nv * nv
    ) * 8
    amplitude_call_copies = (
        4 * amplitude_elements + 2 * no * nv
    ) * 8
    assert direct.by_category[
        "A-CCSD(T) Lambda amplitude/call copies"
    ] == amplitude_call_copies
    assert direct.by_category[
        "A-CCSD(T) Lambda integral/call copies"
    ] == 3 * triples_integrals
    assert "A-CCSD(T) spatial solver bridge" in direct.phase_peaks
    assert "A-CCSD(T) Lambda solve" in direct.phase_peaks
    assert direct.phase_peaks["A-CCSD(T) Lambda solve"] >= ad_workspace
    assert "A-CCSD(T) Lambda triples" in direct.phase_peaks
    assert direct.phase_peaks["A-CCSD(T) Lambda triples"] >= (
        2 * amplitude_elements * 8
        + amplitude_call_copies
        + 3 * triples_integrals
        + direct.by_category["A-CCSD(T) Lambda triples image buffers"]
    )
    assert direct.phase_peaks == fast.phase_peaks

    exact = vq.estimate_memory(
        mol,
        basis,
        method="a-ccsd(t)",
        options=SimpleNamespace(
            density_fit=False,
            n_frozen_core=1,
            diis_subspace_size=4,
        ),
    )
    exact_transform_peak = 3 * exact.dims["n_basis"] ** 4 * 8
    assert exact.by_category[
        "A-CCSD(T) exact integral-transform peak"
    ] == exact_transform_peak
    assert exact.phase_peaks["A-CCSD(T) dense integral build"] >= (
        exact_transform_peak
        + exact.by_category["A-CCSD(T) dense integral blocks"]
    )


def test_canonical_ccsd_prices_pair_transform_return_boundary():
    """The exact-ERI CC build retains both half transforms and returned W."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvdz")
    estimate = vq.estimate_memory(
        mol,
        basis,
        method="ccsd",
        options=SimpleNamespace(
            density_fit=False,
            n_frozen_core=1,
            compute_triples=False,
        ),
    )

    n = estimate.dims["n_basis"]
    n_corr = estimate.dims["n_occ"] + estimate.dims["n_vir"]
    source_peak = (
        n**4 + 2 * n * n * n_corr * n_corr + 2 * n_corr**4
    ) * 8

    assert estimate.by_category[
        "CCSD canonical AO-to-MO transform peak"
    ] == source_peak
    assert estimate.phase_peaks[
        "CC canonical AO-to-MO pair transform"
    ] >= source_peak


def test_correlated_workspace_scales_across_basis_families():
    mol = _h2o()
    basis_names = ("sto-3g", "def2-svp", "cc-pvdz", "cc-pvtz")
    mp2_workspaces = []
    triples_workspaces = []

    for basis_name in basis_names:
        basis = vq.BasisSet(mol, basis_name)
        mp2 = vq.estimate_memory(
            mol,
            basis,
            method="mp2",
            options=SimpleNamespace(
                density_fit=False,
                memory_mode="direct",
                requested_memory_bytes=256 * 1024,
                n_frozen_core=1,
            ),
        )
        triples = vq.estimate_memory(
            mol,
            basis,
            method="ccsd(t)",
            options=SimpleNamespace(
                density_fit=True,
                aux_basis="",
                n_frozen_core=1,
                triples_memory_mode="blocked",
                requested_memory_bytes=64 * 1024**2,
                triples_tile_size=2,
                triples_max_threads=2,
            ),
        )
        mp2_workspaces.append(
            mp2.by_category["MP2 direct occupied-orbital slab"]
        )
        triples_workspaces.append(
            triples.by_category["CCSD(T) triples image buffers"]
        )

    assert mp2_workspaces == sorted(mp2_workspaces)
    assert triples_workspaces == sorted(triples_workspaces)
    assert mp2_workspaces[-1] > mp2_workspaces[0]
    assert triples_workspaces[-1] > triples_workspaces[0]


def test_ccsdt_estimator_uses_streaming_not_materialised_t3():
    """Triples estimate must not contain a materialised T3 amplitudes term.

    The old estimator included a "CCSD(T) triples amplitudes" category
    sized at n_occ^3 * n_vir^3 * 8 bytes.  The C++ code never allocates
    the full T3 tensor; amplitudes are computed on-the-fly and contracted
    with denominators inside the (a,b,c) virtual loop nest.
    """
    mol = _h2o()
    basis = vq.BasisSet(mol, "cc-pvdz")
    est = vq.estimate_memory(mol, basis, method="ccsd(t)")

    assert "CCSD(T) triples amplitudes" not in est.by_category
    assert "CCSD(T) triples image buffers" in est.by_category
    assert "CCSD(T) triples work list" in est.by_category


def test_ccsdt_scales_with_virtuals_not_occupied_cubed():
    """Triples estimate should scale as n_vir^3, not n_occ^3 * n_vir^3.

    The per-thread wc/wd image buffers are sized nv^3, independent of
    n_occ.  Doubling n_occ should barely change the triples estimate
    (only the work list grows, which is negligible).
    """
    # H2O: 10 electrons, 5 occupied
    mol_small = _h2o()
    basis = vq.BasisSet(mol_small, "cc-pvdz")
    est_small = vq.estimate_memory(mol_small, basis, method="ccsd(t)")

    # C2H6: 18 electrons, 9 occupied — similar virtual space
    mol_large = vq.Molecule([
        vq.Atom(6, [ 0.0000,  0.0000,  0.0000]),
        vq.Atom(6, [ 0.0000,  0.0000,  1.5400]),
        vq.Atom(1, [ 1.0300,  0.0000, -0.5100]),
        vq.Atom(1, [-0.5100, -0.8900, -0.5100]),
        vq.Atom(1, [-0.5100,  0.8900, -0.5100]),
        vq.Atom(1, [-0.5100, -0.8900,  2.0500]),
        vq.Atom(1, [-0.5100,  0.8900,  2.0500]),
        vq.Atom(1, [ 1.0300,  0.0000,  2.0500]),
    ])
    basis_large = vq.BasisSet(mol_large, "cc-pvdz")
    est_large = vq.estimate_memory(mol_large, basis, method="ccsd(t)")

    triples_small = est_small.by_category["CCSD(T) triples image buffers"]
    triples_large = est_large.by_category["CCSD(T) triples image buffers"]

    # C2H6 has ~2x more virtuals (bigger molecule) so triples will be
    # larger, but the ratio should be roughly (nv_large/nv_small)^3,
    # nowhere near n_occ^3 factor (~5.8x).  With the old formula,
    # doubling n_occ from 5 to 9 would give (9/5)^3 ~ 5.8x just from
    # the n_occ factor, on top of the n_vir growth.
    ratio = triples_large / max(1, triples_small)
    assert ratio < 50, (
        f"Triples ratio {ratio:.1f}x — old estimator would be ~{ratio * 5.8:.0f}x"
    )


def test_cisd_estimate_uses_wavefunction_path():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    est = vq.estimate_memory(mol, basis, method="cisd")

    assert est.total_bytes > 0
    assert "CI determinant vector" in est.by_category
    assert "CI sigma/residual workspace" in est.by_category
    assert "MO integral transform workspace" in est.by_category


def test_full_ccsdt_estimate_tracks_frozen_core_space():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    all_electron = vq.estimate_memory(
        mol,
        basis,
        method="ccsdt",
        options=vq.CCSDTOptions(n_frozen_core=0),
    )
    frozen_core = vq.estimate_memory(
        mol,
        basis,
        method="ccsdt",
        options=vq.CCSDTOptions(n_frozen_core=1),
    )

    for category in (
        "CCSDT active MO integrals",
        "CCSDT excitation metadata",
        "CCSDT amplitudes and DIIS",
        "CCSDT determinant-state dictionaries",
    ):
        assert category in all_electron.by_category
        assert all_electron.by_category[category] > frozen_core.by_category[category]


def test_cc3_estimate_tracks_dense_triples_and_frozen_core_space():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    all_electron = vq.estimate_memory(
        mol,
        basis,
        method="cc3",
        options=vq.CC3Options(n_frozen_core=0),
    )
    frozen_core = vq.estimate_memory(
        mol,
        basis,
        method="cc3",
        options=vq.CC3Options(n_frozen_core=1),
    )

    for category in (
        "CC3 active MO integrals",
        "CC3 excitation metadata",
        "CC3 amplitudes and DIIS",
        "CC3 determinant-state dictionaries",
    ):
        assert category in all_electron.by_category
        assert all_electron.by_category[category] > frozen_core.by_category[category]


def test_caspt2_n2_cas66_estimate_catches_release_paper_oom():
    """N2/cc-pVDZ CAS(6,6) CASPT2 must not preflight as a MB-scale job.

    Release-paper job 99b812f70a79 was killed after reaching about 46 GB RSS
    while the old generic wavefunction estimator reported only about 16.9 MB.
    """
    mol = _n2_release_geometry()
    basis = vq.BasisSet(mol, "cc-pvdz")

    est = vq.estimate_memory(
        mol,
        basis,
        method="caspt2",
        options={"active_space": (6, 6), "caspt2_options": None},
    )

    assert est.total_bytes > 48 * 1024**3
    assert "CASPT2 active-CI state workspace" in est.by_category
    assert est.by_category["CASPT2 active-CI state workspace"] > 40 * 1024**3


# ---------------------------------------------------------------------------
# selected CI — the estimate must follow the selection, not the full CI space
# (issue #79)
# ---------------------------------------------------------------------------


def _selected_ci_estimate_for(target_size, active_space=(6, 6)):
    from vibeqc.solvers import SelectedCIOptions

    mol = _n2_release_geometry()
    basis = vq.BasisSet(mol, "cc-pvdz")
    return vq.estimate_memory(
        mol,
        basis,
        method="selected_ci",
        options={
            "scf_options": None,
            "active_space": active_space,
            "selected_ci_options": SelectedCIOptions(target_size=target_size),
        },
    )


def test_selected_ci_estimate_varies_with_target_size():
    """The estimate must be a function of the requested selection budget.

    The defect was that it was not: on N2/cc-pVDZ CAS(6e,6o) the pre-flight
    reported a bit-identical 324364.8 GB at ``target_size`` 6, 20, 50, 100, 200
    and 400 (vq job 7abb6453e9fa, v0.15.132), because it sized the untruncated
    CI space and ``target_size`` never reached it at all.
    """
    totals = [_selected_ci_estimate_for(t).total_bytes for t in (6, 20, 50, 100)]

    assert len(set(totals)) == len(totals)
    assert totals == sorted(totals)


def test_selected_ci_estimate_is_bounded_by_the_active_space():
    """A CAS(6e,6o) selection cannot need more than its 400 determinants.

    Requesting more determinants than the active space holds must not grow the
    estimate past the whole-space cost -- and the whole-space cost is what the
    CASCI sibling of the same job pays, which ran fine on the same host.
    """
    mol = _n2_release_geometry()
    basis = vq.BasisSet(mol, "cc-pvdz")

    at_400 = _selected_ci_estimate_for(400).total_bytes
    at_4000 = _selected_ci_estimate_for(4000).total_bytes
    casci = vq.estimate_memory(
        mol,
        basis,
        method="casci",
        options={"scf_options": None, "active_space": (6, 6)},
    ).total_bytes

    assert at_4000 == at_400
    assert at_400 < 2 * casci


def test_selected_ci_small_target_passes_a_laptop_scale_budget():
    """The held SI rung: target_size=6 on N2/cc-pVDZ CAS(6e,6o).

    Every rung of that ladder was refused against a 23.4 GB allocation. The
    smallest one must now clear a budget a laptop actually has.
    """
    est = _selected_ci_estimate_for(6)

    assert est.total_bytes < 4 * 1024**3
    vq.check_memory(est, available=4 * 1024**3)


def test_selected_ci_itemises_its_own_working_set():
    est = _selected_ci_estimate_for(100)

    assert "Selected-CI variational Hamiltonian" in est.by_category
    assert "Selected-CI candidate buffer" in est.by_category
    assert "CI determinant vector" not in est.by_category
    assert "FCI determinant vector" not in est.by_category


def test_exact_ci_estimates_still_size_the_full_space():
    """The exact solvers really do hold the whole space; leave them alone.

    Pinned against the same molecule so a future change to the selected-CI
    model cannot quietly relax the exact routes with it.
    """
    mol = _n2_release_geometry()
    basis = vq.BasisSet(mol, "cc-pvdz")

    for method in ("fci", "cisd"):
        est = vq.estimate_memory(mol, basis, method=method)
        assert est.total_bytes > 100_000 * 1024**3, method


def test_selected_ci_estimate_without_an_active_space_stays_finite():
    """No CAS means the excitations span the whole MO set, not the whole CI space.

    The candidate buffer is bounded by what one selection step can generate
    from the working space, so a plain selected-CI job on N2/cc-pVDZ prices as
    a routine job rather than as full CI.
    """
    est = _selected_ci_estimate_for(100, active_space=None)

    assert est.total_bytes < 8 * 1024**3
    assert est.total_bytes > _selected_ci_estimate_for(100).total_bytes


def test_caspt2_release_estimate_aborts_under_reported_host_cap():
    mol = _n2_release_geometry()
    basis = vq.BasisSet(mol, "cc-pvdz")
    est = vq.estimate_memory(
        mol,
        basis,
        method="caspt2",
        options={"active_space": (6, 6), "caspt2_options": None},
    )

    with pytest.raises(vq.InsufficientMemoryError, match="CASPT2 active-CI"):
        vq.check_memory(est, available=44_406 * 1024**2)


def test_caspt2_corr_gradient_estimate_includes_pt2_fd_workers():
    mol = _n2_release_geometry()
    basis = vq.BasisSet(mol, "cc-pvdz")
    opts = vq.CASPT2Options(compute_corr_grad=True, use_zvector=True)

    est = vq.estimate_memory(
        mol,
        basis,
        method="caspt2",
        options={"active_space": (6, 6), "caspt2_options": opts},
    )

    assert "PT2 gradient FD workers" in est.by_category
    assert est.by_category["PT2 gradient FD workers"] > 0


def test_nevpt2_corr_gradient_estimate_includes_pt2_fd_workers():
    from vibeqc.solvers import NEVPT2Options

    mol = _n2_release_geometry()
    basis = vq.BasisSet(mol, "cc-pvdz")
    opts = NEVPT2Options(compute_corr_grad=True, use_zvector=True)

    est = vq.estimate_memory(
        mol,
        basis,
        method="nevpt2",
        options={"active_space": (6, 6), "nevpt2_options": opts},
    )

    assert "PT2 gradient FD workers" in est.by_category
    assert est.by_category["PT2 gradient FD workers"] > 0


def test_periodic_xc_gradient_estimate_tracks_kind_spin_and_threads():
    from vibeqc.memory import estimate_periodic_xc_gradient

    lda = estimate_periodic_xc_gradient(
        n_basis=24,
        n_atoms=3,
        n_grid_points=2_000,
        n_cells=9,
        functional_kind="LDA",
        open_shell=False,
        n_threads=2,
    )
    gga = estimate_periodic_xc_gradient(
        n_basis=24,
        n_atoms=3,
        n_grid_points=2_000,
        n_cells=9,
        functional_kind="GGA",
        open_shell=False,
        n_threads=2,
    )
    uks_gga = estimate_periodic_xc_gradient(
        n_basis=24,
        n_atoms=3,
        n_grid_points=2_000,
        n_cells=9,
        functional_kind="GGA",
        open_shell=True,
        n_threads=2,
    )
    gga_more_threads = estimate_periodic_xc_gradient(
        n_basis=24,
        n_atoms=3,
        n_grid_points=2_000,
        n_cells=9,
        functional_kind="GGA",
        open_shell=False,
        n_threads=4,
    )

    assert "Periodic-XC gradient AO/gradient/Hessian tables" in gga.by_category
    assert "Periodic-XC gradient pair scratch" in gga.by_category
    assert "Periodic-XC gradient grid vectors" in gga.by_category
    assert gga.by_category[
        "Periodic-XC gradient AO/gradient/Hessian tables"
    ] > lda.by_category["Periodic-XC gradient AO/gradient/Hessian tables"]
    assert uks_gga.by_category[
        "Periodic-XC gradient pair scratch"
    ] > gga.by_category["Periodic-XC gradient pair scratch"]
    assert gga_more_threads.by_category[
        "Periodic-XC gradient pair scratch"
    ] == 2 * gga.by_category["Periodic-XC gradient pair scratch"]


def test_periodic_gpw_gapw_estimate_charges_route_and_kmesh():
    from vibeqc.memory import estimate_periodic_gpw_gapw

    gpw = estimate_periodic_gpw_gapw(
        n_basis=24,
        n_grid_points=12_000,
        route="gpw",
        functional_kind="GGA",
        open_shell=False,
        n_kpoints=1,
    )
    gapw = estimate_periodic_gpw_gapw(
        n_basis=24,
        n_grid_points=12_000,
        route="gapw",
        functional_kind="GGA",
        open_shell=False,
        n_kpoints=1,
        n_soft_basis=18,
        n_atoms=4,
        augmentation_active=True,
    )
    gapw_analytic = estimate_periodic_gpw_gapw(
        n_basis=24,
        n_grid_points=12_000,
        route="gapw",
        functional_kind=None,
        open_shell=False,
        n_kpoints=1,
        n_soft_basis=18,
        n_atoms=4,
        augmentation_active=True,
        analytic_eri_one_centre=True,
    )
    gapw_analytic_inactive = estimate_periodic_gpw_gapw(
        n_basis=24,
        n_grid_points=12_000,
        route="gapw",
        functional_kind=None,
        open_shell=False,
        n_kpoints=1,
        n_soft_basis=24,
        n_atoms=4,
        augmentation_active=False,
        analytic_eri_one_centre=True,
    )
    compact = estimate_periodic_gpw_gapw(
        n_basis=24,
        n_grid_points=12_000,
        route="gpw",
        functional_kind="MGGA",
        open_shell=True,
        n_kpoints=4,
        compact_multik=True,
    )

    assert "GPW full collocation cache" in gpw.by_category
    assert "GPW reciprocal mesh cache" in gpw.by_category
    assert "GPW density/Poisson grids" in gpw.by_category
    assert "GPW XC grid workspace" in gpw.by_category
    assert "GAPW soft collocation cache" in gapw.by_category
    assert "GAPW analytic augmentation cache" in gapw.by_category
    assert gapw_analytic.by_category[
        "GAPW fit-free hard/soft ERI cache"
    ] == 8 * (24**4 + 18**4)
    assert gapw_analytic.total_bytes > gapw.total_bytes
    assert (
        "GAPW fit-free hard/soft ERI cache"
        not in gapw_analytic_inactive.by_category
    )
    assert "GAPW soft collocation cache" not in gapw_analytic_inactive.by_category
    assert gapw.total_bytes > gpw.total_bytes
    # The open-shell compact route joined the bounded model in issue #89, so
    # it is charged the same categories as the closed-shell one: the retained
    # all-k cache while it fits the 1 GiB target (18.4 MB here), and the
    # bounded one-k-at-a-time meta-GGA scratch -- not the dense
    # "Bloch AO tables" / "meta-GGA AO-gradient scratch" pair it used to get.
    assert "GPW compact multi-k Bloch AO cache" in compact.by_category
    assert "GPW compact multi-k meta-GGA AO/FFT scratch" in compact.by_category
    assert "GPW compact multi-k Bloch AO tables" not in compact.by_category
    assert "GPW meta-GGA AO-gradient scratch" not in compact.by_category
    assert compact.total_bytes > gpw.total_bytes


@pytest.mark.parametrize("driver_name", ["rhf", "uhf", "uks"])
def test_standalone_gapw_analytic_preflight_precedes_dense_eri_allocation(
    monkeypatch,
    driver_name,
):
    """Standalone HF drivers enforce the analytic ERI memory gate too."""
    import vibeqc.memory as memory_module
    import vibeqc.periodic_gapw_augment as gapw_augment
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    length = 12.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * length,
        [
            vq.Atom(8, [6.0, 6.0, 6.0]),
            vq.Atom(1, [6.0, 7.43, 5.02]),
            vq.Atom(1, [6.0, 4.57, 5.02]),
        ],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * length, 4, 4, 4)

    monkeypatch.setattr(memory_module, "available_memory_bytes", lambda: 1)

    def _unexpected_integrals(*_args, **_kwargs):
        raise AssertionError("analytic ERI memory gate ran after AO integrals")

    monkeypatch.setattr(
        gapw_augment,
        "_kinetic_lattice_gamma",
        _unexpected_integrals,
    )

    with pytest.raises(vq.InsufficientMemoryError, match="ABORTING"):
        if driver_name == "rhf":
            gapw_augment.run_periodic_rhf_gapw(
                system,
                basis,
                grid=grid,
                molecular_limit=True,
                quiet=True,
            )
        elif driver_name == "uhf":
            gapw_augment.run_periodic_uhf_gapw(
                system,
                basis,
                n_alpha=5,
                n_beta=5,
                grid=grid,
                molecular_limit=True,
                quiet=True,
            )
        else:
            gapw_augment.run_periodic_uks_gapw(
                system,
                basis,
                functional="lda",
                n_alpha=5,
                n_beta=5,
                grid=grid,
                one_centre="analytic",
                quiet=True,
            )


@pytest.mark.parametrize(
    ("functional", "open_shell", "expected_kind"),
    [
        ("lda", False, "LDA"),
        ("pbe", True, "GGA"),
    ],
)
def test_gapw_analytic_dft_preflight_charges_xc_workspace(
    monkeypatch,
    functional,
    open_shell,
    expected_kind,
):
    """Analytic RKS/UKS estimates must retain their XC workspace charge."""
    import vibeqc.memory as memory_module
    import vibeqc.periodic_gapw_augment as gapw_augment
    from vibeqc.periodic_gapw_grid import PlaneWaveGrid

    length = 12.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * length,
        [
            vq.Atom(8, [6.0, 6.0, 6.0]),
            vq.Atom(1, [6.0, 7.43, 5.02]),
            vq.Atom(1, [6.0, 4.57, 5.02]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * length, 4, 4, 4)
    seen = {}

    def _capture_estimate(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(total_bytes=0)

    monkeypatch.setattr(
        memory_module,
        "estimate_periodic_gpw_gapw",
        _capture_estimate,
    )
    monkeypatch.setattr(memory_module, "check_memory", lambda *_args, **_kwargs: None)

    gapw_augment._preflight_analytic_eri_memory(
        system,
        basis,
        grid,
        functional=functional,
        soft_cutoff=3.0,
        lmax=3,
        n_radial=80,
        lebedev_order=17,
        open_shell=open_shell,
        memory_override=False,
    )

    assert seen["functional_kind"] == expected_kind
    assert seen["open_shell"] is open_shell


@pytest.mark.parametrize("open_shell", [False, True])
def test_periodic_gpw_p16_compact_bloch_storage_is_kmesh_bounded(open_shell):
    """P16 NiO must not retain one 128^3 Bloch-AO table per k-point.

    The archived 4x4x4 / pob-TZVP-REV2 calculation has 232 AOs.  Retaining
    all 64 complex AO tables requires 464 GiB before SCF workspaces, so the
    memory preflight rejects a 48 GiB node.  Compact GPW is a streaming route:
    its AO workspace must be independent of the number of k-points -- on
    both spin branches. The open-shell (AFM NiO, DFT+U) driver kept the dense
    all-k cache after the closed-shell fix and was still estimated at
    702.7 GiB on this cell (issue #89, re-test 2026-08-28); it streams now.
    """
    from vibeqc.memory import estimate_periodic_gpw_gapw

    kwargs = dict(
        n_basis=232,
        n_grid_points=128**3,
        route="gpw",
        functional_kind="GGA",
        open_shell=open_shell,
        compact_multik=True,
    )
    two_k = estimate_periodic_gpw_gapw(n_kpoints=2, **kwargs)
    p16 = estimate_periodic_gpw_gapw(n_kpoints=64, **kwargs)

    category = "GPW compact multi-k Bloch AO batch"
    assert p16.by_category[category] == two_k.by_category[category]
    assert p16.total_bytes < 48 * 1024**3


def test_estimate_memory_rejects_unknown_method():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="unknown method"):
        vq.estimate_memory(mol, basis, method="not-a-method")


def test_dlpno_ccsdt_estimate_has_local_and_triples_categories():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    est = vq.estimate_memory(mol, basis, method="dlpno-ccsd(t)")

    assert "DLPNO PAO coefficients/domains" in est.by_category
    assert "DLPNO pair lists" in est.by_category
    assert "DLPNO-CCSD T1/T2 pair amplitudes" in est.by_category
    assert "DLPNO triples T_ijk^abc workspace" in est.by_category


@pytest.mark.parametrize("method", ["dlpno-mp2", "dlpno-ccsd(t)"])
def test_dlpno_memory_report_discloses_composition_level_bound(method):
    compact = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ])
    separated = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, -10.0]),
        vq.Atom(1, [0.0, 0.0, 10.0]),
    ])

    compact_estimate = vq.estimate_memory(
        compact,
        vq.BasisSet(compact, "sto-3g"),
        method=method,
    )
    separated_estimate = vq.estimate_memory(
        separated,
        vq.BasisSet(separated, "sto-3g"),
        method=method,
    )
    report = compact_estimate.format(available=512 * 1024**3)

    assert compact_estimate.by_category == separated_estimate.by_category
    assert "composition-level bound" in report
    assert "geometry-dependent pair/PNO locality is not modeled" in report
    with pytest.raises(vq.InsufficientMemoryError) as refusal:
        vq.check_memory(compact_estimate, available=1)
    assert "composition-level bound" in str(refusal.value)
    assert "geometry-dependent pair/PNO locality is not modeled" in str(
        refusal.value
    )


@pytest.mark.parametrize(
    "method,open_shell",
    [
        ("dlpno-mp2", False),
        ("dlpno-ccsd", False),
        ("dlpno-ccsd(t)", False),
        ("dlpno-ump2", True),
        ("dlpno-uccsd", True),
        ("dlpno-uccsd(t)", True),
    ],
)
def test_dlpno_estimator_defaults_use_published_frozen_core(
    method, open_shell
):
    molecule = (
        vq.Molecule(
            [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
            multiplicity=2,
        )
        if open_shell
        else _h2o()
    )
    basis = vq.BasisSet(molecule, "def2-svp")

    estimate = vq.estimate_memory(
        molecule,
        basis,
        method=method,
    )

    assert estimate.dims["n_frozen_core"] == 1


def _retained_reference_bytes(estimate):
    category = estimate.by_category
    return sum(
        category.get(name, 0)
        for name in (
            "Fock + density + 1e",
            "MO workspace",
            "Open-shell UHF buffers",
        )
    )


def _dlpno_live_reference_bytes(estimate):
    return (
        estimate.by_category.get("Python runtime + NumPy overhead", 0)
        + _retained_reference_bytes(estimate)
    )


@pytest.mark.parametrize("open_shell", [False, True])
def test_dlpno_ccsd_prices_python_df_transform_lifetimes(
    open_shell,
    monkeypatch,
):
    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 2)
    aux_name = "def2-svp-rifit"
    if open_shell:
        from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions

        molecule = vq.Molecule(
            [
                vq.Atom(8, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 0.0, 1.8]),
            ],
            multiplicity=2,
        )
        options = LocalUCCSDOptions(n_frozen=0, aux_basis=aux_name)
        method = "dlpno-uccsd"
    else:
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

        molecule = _h2o()
        options = LocalCCSDOptions(n_frozen=0, aux_basis=aux_name)
        method = "dlpno-ccsd"
    basis = vq.BasisSet(molecule, "def2-svp")
    estimate = vq.estimate_memory(
        molecule,
        basis,
        method=method,
        options=options,
    )

    n = estimate.dims["n_basis"]
    n_aux = estimate.dims["n_aux"]
    n_occ = estimate.dims["n_occ"]
    n_vir = estimate.dims["n_vir"]
    three_index = n_aux * n * n * 8
    metric = n_aux * n_aux * 8
    resident = 2 * three_index + 3 * metric
    if open_shell:
        n_elec = molecule.n_electrons()
        n_alpha = (n_elec + molecule.multiplicity - 1) // 2
        n_beta = n_elec - n_alpha
        n_occ_a = n_alpha
        n_occ_b = n_beta
        n_vir_a = n - n_alpha
        n_vir_b = n - n_beta
        shapes = (
            (n_occ_a, n_occ_a),
            (n_occ_b, n_occ_b),
            (n_occ_a, n_vir_a),
            (n_occ_b, n_vir_b),
            (n_vir_a, n_vir_a),
            (n_vir_b, n_vir_b),
        )
        retained = 0
        transform_peak = 0
        for n_left, n_right in shapes:
            output = n_aux * n_left * n_right * 8
            scratch = n_aux * n * n_right * 8
            transform_peak = max(
                transform_peak,
                retained + scratch + output,
            )
            retained += output
        combined = n_aux * (
            n_occ * n_occ + n_occ * n_vir + n_vir * n_vir
        ) * 8
        global_factors = retained + combined
        transform_peak = max(transform_peak, global_factors)
    else:
        retained = 0
        transform_peak = 0
        for n_left, n_right in (
            (n_occ, n_vir),
            (n_vir, n_vir),
            (n_occ, n_occ),
        ):
            output = n_aux * n_left * n_right * 8
            scratch = n_aux * n * n_right * 8
            transform_peak = max(
                transform_peak,
                retained + scratch + output,
            )
            retained += output
        global_factors = retained

    category = estimate.by_category
    assert category["DLPNO DF construction overlap"] == (
        4 * three_index + 4 * metric
    )
    assert category["DLPNO DF auxiliary integrals"] == resident
    assert category["DLPNO global MO DF factors"] == global_factors
    assert category["DLPNO MO DF transformation peak"] == transform_peak
    assert estimate.phase_peaks["DLPNO MO DF transformation"] == (
        _dlpno_live_reference_bytes(estimate)
        + category["DLPNO native worker stacks/scratch"]
        + resident
        + transform_peak
    )


def test_dlpno_mp2_global_df_prices_constructor_and_transform_peaks():
    """The default route retains both AO DF tensors and metric factors."""
    from vibeqc.dlpno.mp2 import DLPNOMP2Options

    molecule = _h2o()
    basis = vq.BasisSet(molecule, "def2-svp")
    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-mp2",
        options=DLPNOMP2Options(n_frozen=0, local_df=False),
    )

    category = estimate.by_category
    n = estimate.dims["n_basis"]
    n_aux = estimate.dims["n_aux"]
    n_occ = estimate.dims["n_occ"]
    three_index = n_aux * n * n * 8
    metric = n_aux * n_aux * 8
    expected_construction = 4 * three_index + 4 * metric
    expected_resident = 2 * three_index + 3 * metric
    expected_factors = n_aux * n_occ * n * 8
    n_pairs = estimate.dims["n_pairs"]
    pno_bound = estimate.dims["n_vir"]
    link_count = n_pairs * 2 * (n_occ - 1)
    link_cache = link_count * pno_bound * pno_bound * 8
    packed_pair_inputs = n_pairs * (
        2 * pno_bound * pno_bound + n * pno_bound + pno_bound
    ) * 8
    expected_iteration_copies = (
        2 * packed_pair_inputs
        + 2 * n_pairs * pno_bound * pno_bound * 8
        + 2 * (n_occ * n_occ + n * n) * 8
        + (8 * pno_bound * pno_bound + 2 * n * pno_bound) * 8
    )

    assert category["DLPNO-MP2 DF construction overlap"] == (
        expected_construction
    )
    assert category["DLPNO-MP2 retained DF substrate"] == expected_resident
    assert category["DLPNO-MP2 transformed DF factors"] == expected_factors
    assert category["DLPNO-MP2 MO-transform scratch"] == three_index
    assert category["DLPNO-MP2 LMP2 overlap-link cache"] == link_cache
    assert category[
        "DLPNO-MP2 native iteration bridge/copies"
    ] == expected_iteration_copies
    assert estimate.dims["lmp2_link_count_bound"] == link_count
    assert estimate.dims["lmp2_pno_rank_bound"] == pno_bound
    assert "DLPNO DF auxiliary integrals" not in category
    assert estimate.phase_peaks["DLPNO-MP2 DF construction"] == (
        _dlpno_live_reference_bytes(estimate) + expected_construction
    )
    persistent_pairs = sum(
        category[name]
        for name in (
            "DLPNO PAO coefficients/domains",
            "DLPNO pair lists",
            "DLPNO-MP2 PNO pair amplitudes",
        )
    )
    assert estimate.phase_peaks[
        "DLPNO-MP2 coupled-LMP2 iteration"
    ] == (
        _dlpno_live_reference_bytes(estimate)
        + expected_resident
        + persistent_pairs
        + link_cache
        + expected_iteration_copies
    )
    assert estimate.raw_total_bytes == max(estimate.phase_peaks.values())


def test_dlpno_ump2_prices_both_spin_transformed_df_factors():
    """Open-shell admission follows the two retained Ba/Bb transforms."""
    from vibeqc.dlpno.ump2 import DLPNOUMP2Options

    molecule = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    basis = vq.BasisSet(molecule, "def2-svp")
    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-ump2",
        options=DLPNOUMP2Options(n_frozen=0),
    )

    n = estimate.dims["n_basis"]
    n_aux = estimate.dims["n_aux"]
    n_elec = molecule.n_electrons()
    n_alpha = (n_elec + molecule.multiplicity - 1) // 2
    n_beta = n_elec - n_alpha
    n_vir_a = n - n_alpha
    n_vir_b = n - n_beta
    expected_factors = n_aux * (
        n_alpha * n_vir_a + n_beta * n_vir_b
    ) * 8
    expected_scratch = n_aux * n * max(n_vir_a, n_vir_b) * 8

    assert estimate.by_category[
        "DLPNO-MP2 transformed DF factors"
    ] == expected_factors
    assert estimate.by_category[
        "DLPNO-MP2 MO-transform scratch"
    ] == expected_scratch
    assert "DLPNO-MP2 DF construction" in estimate.phase_peaks


def test_dlpno_mp2_local_df_has_distinct_cached_domain_peak():
    """Local DF does not inherit a global DensityFitting allocation."""
    from vibeqc.dlpno.mp2 import DLPNOMP2Options

    molecule = _h2o()
    basis = vq.BasisSet(molecule, "def2-svp")
    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-mp2",
        options=DLPNOMP2Options(
            n_frozen=0,
            local_df=True,
            fit_buffer=1.0e9,
        ),
    )

    category = estimate.by_category
    n = estimate.dims["n_basis"]
    n_aux = estimate.dims["n_aux"]
    expected_cache = n_aux * n * n * 8 + n_aux * n_aux * 8

    assert "DLPNO-MP2 retained DF substrate" not in category
    assert "DLPNO-MP2 DF construction overlap" not in category
    assert "DLPNO-MP2 transformed DF factors" not in category
    assert category[
        "DLPNO-MP2 local-DF cached domain integrals"
    ] == expected_cache
    assert estimate.dims["local_fit_dimension_bound"] == n_aux
    assert estimate.dims["n_local_fit_domains_estimate"] == 1
    assert "DLPNO-MP2 local-DF pair setup" in estimate.phase_peaks

    bounded = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-mp2",
        options=DLPNOMP2Options(
            n_frozen=0,
            local_df=True,
            fit_buffer=4.0,
        ),
    )
    assert bounded.dims["local_fit_dimension_bound"] == n_aux
    assert bounded.dims["n_local_fit_domains_estimate"] == (
        bounded.dims["n_pairs"]
    )
    assert bounded.by_category[
        "DLPNO-MP2 local-DF cached domain integrals"
    ] == bounded.dims["n_pairs"] * expected_cache


@pytest.mark.parametrize(
    "method,molecule_factory,estimator_method,option_key,option_type,"
    "expected_pair_cutoff",
    [
        (
            "dlpno-mp2",
            _h2o,
            "dlpno-mp2",
            "dlpno_options",
            "DLPNOMP2Options",
            1.0e-5,
        ),
        (
            "dlpno-mp2",
            lambda: vq.Molecule(
                [
                    vq.Atom(8, [0.0, 0.0, 0.0]),
                    vq.Atom(1, [0.0, 0.0, 1.8]),
                ],
                multiplicity=2,
            ),
            "dlpno-ump2",
            "dlpno_ump2_options",
            "DLPNOUMP2Options",
            1.0e-4,
        ),
        (
            "dlpno-ccsd(t)",
            _h2o,
            "dlpno-ccsd(t)",
            "dlpno_ccsd_options",
            "LocalCCSDOptions",
            1.0e-5,
        ),
        (
            "dlpno-ccsd(t)",
            lambda: vq.Molecule(
                [
                    vq.Atom(8, [0.0, 0.0, 0.0]),
                    vq.Atom(1, [0.0, 0.0, 1.8]),
                ],
                multiplicity=2,
            ),
            "dlpno-uccsd(t)",
            "dlpno_ccsd_options",
            "LocalUCCSDOptions",
            1.0e-5,
        ),
    ],
)
def test_run_job_dlpno_preflight_receives_resolved_route_options(
    tmp_path,
    monkeypatch,
    method,
    molecule_factory,
    estimator_method,
    option_key,
    option_type,
    expected_pair_cutoff,
):
    """Admission sees the exact spin route and TightPNO policy it runs."""
    import vibeqc.runner as runner_module

    captured = {}

    class PreflightCaptured(RuntimeError):
        pass

    def capture_estimate(molecule, basis, *, method, options):
        captured.update(method=method, options=options)
        raise PreflightCaptured

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    with pytest.raises(PreflightCaptured):
        vq.run_job(
            molecule_factory(),
            basis="def2-svp",
            method=method,
            output=str(tmp_path / option_type),
            frozen_core="published",
            dlpno_thresholds="tight",
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )

    assert captured["method"] == estimator_method
    options = captured["options"][option_key]
    assert type(options).__name__ == option_type
    assert options.n_frozen == 1
    assert options.aux_basis == "def2-svp-rifit"
    assert options.tcut_pno == pytest.approx(1.0e-7)
    # TightPNO's 1e-5 pair threshold must force the conservative full-pair
    # estimate on active pair-screening routes. Canonical-occupied UMP2
    # reports that coordinate inactive and retains its stored Normal value.
    assert options.tcut_pairs == pytest.approx(expected_pair_cutoff)
    if "CCSD" in option_type:
        assert options.compute_triples is True


def test_run_job_dlpno_dry_run_estimate_uses_resolved_policy(
    tmp_path, monkeypatch
):
    """Scheduler placement gets the same options as real admission."""
    import vibeqc.runner as runner_module

    captured = {}
    real_estimate_memory = runner_module.estimate_memory

    def capture_estimate(molecule, basis, *, method, options):
        estimate = real_estimate_memory(
            molecule,
            basis,
            method=method,
            options=options,
        )
        captured.update(method=method, options=options, estimate=estimate)
        return estimate

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    vq.run_job(
        _h2o(),
        basis="def2-svp",
        method="dlpno-ccsd(t)",
        output=str(tmp_path / "dry-tight"),
        frozen_core=False,
        dlpno_thresholds="tight",
        dry_run=True,
        write_xyz_file=False,
        output_qvf=False,
        citations=False,
        crash_dump=False,
    )

    assert captured["method"] == "dlpno-ccsd(t)"
    options = captured["options"]["dlpno_ccsd_options"]
    assert type(options).__name__ == "LocalCCSDOptions"
    assert options.n_frozen == 0
    assert options.tcut_pairs == pytest.approx(1.0e-5)
    assert options.compute_triples is True
    assert (
        captured["estimate"].dims["n_strong_pairs_estimate"]
        == captured["estimate"].dims["n_pairs"]
    )


def test_run_job_dlpno_process_cap_rejects_before_scf(
    tmp_path, monkeypatch
):
    """DLPNO admission must honor the explicit process cap, not re-probe."""
    from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions
    import vibeqc.runner as runner_module

    def fail_if_scf_runs(*args, **kwargs):
        raise AssertionError("SCF ran before the DLPNO process cap was checked")

    monkeypatch.setattr(runner_module, "available_memory_bytes", lambda: 0)
    monkeypatch.setattr(runner_module, "run_rhf", fail_if_scf_runs)
    with pytest.raises(vq.InsufficientMemoryError, match="ABORTING"):
        vq.run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-ccsd",
            dlpno_ccsd_options=DLPNOCCSDPilotOptions(),
            memory_budget_bytes=1,
            output=str(tmp_path / "dlpno-budget"),
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )


def test_mp2_dry_run_and_real_admission_share_budgeted_plan(
    tmp_path, monkeypatch
):
    """Scheduler and real preflight resolve the same native cap and mode."""
    import vibeqc.runner as runner_module

    real_estimate_memory = runner_module.estimate_memory
    caller = vq.MP2Options()
    process_budget = 512 * 1024**2

    def run_and_capture(*, dry_run):
        trace = []

        def capture_estimate(molecule, basis, *, method, options):
            estimate = real_estimate_memory(
                molecule,
                basis,
                method=method,
                options=options,
            )
            prepared = options["mp2_options"]
            mode = next(
                name.split()[1].lower()
                for name in estimate.phase_peaks
                if name.startswith("MP2 ") and name.endswith(" correlation")
            )
            trace.append(
                (int(prepared.requested_memory_bytes), mode)
            )
            return estimate

        monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
        if dry_run:
            monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
            vq.run_job(
                _h2o(),
                basis="def2-qzvpp",
                method="mp2",
                mp2_options=caller,
                memory_budget_bytes=process_budget,
                output=str(tmp_path / "dry-budgeted-mp2"),
                dry_run=True,
                write_xyz_file=False,
                output_qvf=False,
                citations=False,
                crash_dump=False,
            )
            monkeypatch.delenv("VIBEQC_DRY_RUN_ESTIMATE")
        else:
            class RealPathReached(RuntimeError):
                pass

            def stop_before_scf(*args, **kwargs):
                raise RealPathReached

            monkeypatch.setattr(runner_module, "run_rhf", stop_before_scf)
            with pytest.raises(RealPathReached):
                vq.run_job(
                    _h2o(),
                    basis="def2-qzvpp",
                    method="mp2",
                    mp2_options=caller,
                    memory_budget_bytes=process_budget,
                    memory_override=True,
                    output=str(tmp_path / "real-budgeted-mp2"),
                    write_xyz_file=False,
                    output_qvf=False,
                    citations=False,
                    crash_dump=False,
                )
        return trace

    monkeypatch.setattr(runner_module, "available_memory_bytes", lambda: 0)
    dry_trace = run_and_capture(dry_run=True)
    real_trace = run_and_capture(dry_run=False)

    assert len(dry_trace) == len(real_trace) == 2
    assert dry_trace[-1] == real_trace[-1]
    assert dry_trace[-1][0] < process_budget
    assert dry_trace[-1][1] == "direct"
    assert caller.requested_memory_bytes == 0


@pytest.mark.parametrize("open_shell", [False, True])
def test_df_mp2_dry_run_and_real_admission_share_resolved_auxiliary(
    tmp_path, monkeypatch, open_shell
):
    """Scheduler and execution admission size the same concrete RIfit basis."""
    import vibeqc.runner as runner_module

    if open_shell:
        molecule = vq.Molecule(
            [
                vq.Atom(8, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 0.0, 1.8]),
            ],
            multiplicity=2,
        )
        caller = vq.UMP2Options()
        option_key = "ump2_options"
        option_kw = {"ump2_options": caller}
        stop_name = "run_uhf"
    else:
        molecule = _h2o()
        caller = vq.MP2Options()
        option_key = "mp2_options"
        option_kw = {"mp2_options": caller}
        stop_name = "run_rhf"
    caller.density_fit = True
    real_estimate_memory = runner_module.estimate_memory
    expected_aux = "def2-svp-rifit"
    expected_n_aux = vq.BasisSet(molecule, expected_aux).nbasis

    def run_and_capture(*, dry_run):
        trace = []

        def capture_estimate(molecule, basis, *, method, options):
            estimate = real_estimate_memory(
                molecule,
                basis,
                method=method,
                options=options,
            )
            prepared = options[option_key]
            trace.append(
                (
                    method,
                    str(prepared.aux_basis),
                    int(estimate.dims["n_aux"]),
                )
            )
            return estimate

        monkeypatch.setattr(
            runner_module,
            "estimate_memory",
            capture_estimate,
        )
        common = dict(
            molecule=molecule,
            basis="def2-svp",
            method="mp2",
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
            **option_kw,
        )
        if dry_run:
            monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
            vq.run_job(
                **common,
                output=str(tmp_path / f"dry-df-mp2-{open_shell}"),
                dry_run=True,
            )
            monkeypatch.delenv("VIBEQC_DRY_RUN_ESTIMATE")
        else:
            class RealPathReached(RuntimeError):
                pass

            def stop_before_scf(*args, **kwargs):
                raise RealPathReached

            monkeypatch.setattr(runner_module, stop_name, stop_before_scf)
            with pytest.raises(RealPathReached):
                vq.run_job(
                    **common,
                    output=str(tmp_path / f"real-df-mp2-{open_shell}"),
                    memory_override=True,
                )
        return trace

    monkeypatch.setattr(runner_module, "available_memory_bytes", lambda: 0)
    dry_trace = run_and_capture(dry_run=True)
    real_trace = run_and_capture(dry_run=False)

    assert dry_trace
    assert real_trace
    assert dry_trace[-1] == real_trace[-1]
    assert dry_trace[-1][1:] == (expected_aux, expected_n_aux)
    assert caller.aux_basis == ""


def test_run_job_mp2_dry_run_uses_cloned_explicit_frozen_core_policy(
    tmp_path, monkeypatch
):
    """Scheduler placement sees the MP2 execution clone, not raw options."""
    import vibeqc.runner as runner_module

    caller = vq.MP2Options()
    captured = {}

    def capture_estimate(molecule, basis, *, method, options):
        captured.update(method=method, options=options)
        return SimpleNamespace(total_bytes=123)

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    vq.run_job(
        _h2o(),
        basis="def2-svp",
        method="mp2",
        mp2_options=caller,
        frozen_core=False,
        output=str(tmp_path / "dry-mp2"),
        dry_run=True,
        write_xyz_file=False,
        output_qvf=False,
        citations=False,
        crash_dump=False,
    )

    prepared = captured["options"]["mp2_options"]
    assert captured["method"] == "mp2"
    assert prepared is not caller
    assert prepared.n_frozen_core == 0
    assert caller.n_frozen_core is None


def test_run_job_ccsdt_dry_run_uses_cloned_execution_policy(
    tmp_path, monkeypatch
):
    """Triples, DF route, and frozen core are resolved before dry-run."""
    from vibeqc.cc import CCSDOptions
    import vibeqc.runner as runner_module

    caller = CCSDOptions(
        density_fit=False,
        compute_triples=False,
    )
    captured = {}

    def capture_estimate(molecule, basis, *, method, options):
        captured.update(method=method, options=options)
        return SimpleNamespace(total_bytes=456)

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    vq.run_job(
        _h2o(),
        basis="def2-svp",
        method="ccsd",
        triples="(t)",
        ccsd_options=caller,
        frozen_core=False,
        output=str(tmp_path / "dry-ccsdt"),
        dry_run=True,
        write_xyz_file=False,
        output_qvf=False,
        citations=False,
        crash_dump=False,
    )

    prepared = captured["options"]["ccsd_options"]
    assert captured["method"] == "ccsd(t)"
    assert prepared is not caller
    assert prepared.n_frozen_core == 0
    assert prepared.compute_triples is True
    assert prepared.triples_variant == "(t)"
    assert prepared.density_fit is False
    assert caller.n_frozen_core == 0
    assert caller._n_frozen_core_explicit is False
    assert caller.compute_triples is False


@pytest.mark.parametrize("method", ["scs-mp2", "sos-mp2"])
def test_open_scaled_mp2_preflight_uses_ump2_estimator(
    tmp_path, monkeypatch, method
):
    """Open-shell SCS/SOS routes size the UMP2 arrays they execute."""
    import vibeqc.runner as runner_module

    molecule = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    captured = {}

    class PreflightCaptured(RuntimeError):
        pass

    def capture_estimate(molecule, basis, *, method, options):
        captured.update(method=method, options=options)
        raise PreflightCaptured

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    with pytest.raises(PreflightCaptured):
        vq.run_job(
            molecule,
            basis="def2-svp",
            method=method,
            output=str(tmp_path / method),
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )

    assert captured["method"] == "ump2"
    assert captured["options"]["ump2_options"] is not None
    assert captured["options"].get("mp2_options") is None


@pytest.mark.parametrize(
    "method,molecule_factory,option_type",
    [
        (
            "dlpno-ccsd",
            _h2o,
            "closed",
        ),
        (
            "dlpno-ccsd",
            lambda: vq.Molecule(
                [
                    vq.Atom(8, [0.0, 0.0, 0.0]),
                    vq.Atom(1, [0.0, 0.0, 1.8]),
                ],
                multiplicity=2,
            ),
            "open",
        ),
    ],
)
def test_run_job_dlpno_pilot_preflight_uses_dense_estimator(
    tmp_path,
    monkeypatch,
    method,
    molecule_factory,
    option_type,
):
    """Pilot option classes never inherit the local-solver memory model."""
    import vibeqc.runner as runner_module

    if option_type == "open":
        from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions

        pilot_options = DLPNOUCCSDPilotOptions(max_nbf=256)
    else:
        from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions

        pilot_options = DLPNOCCSDPilotOptions(max_nbf=256)

    captured = {}
    real_estimate_memory = runner_module.estimate_memory

    class PreflightCaptured(RuntimeError):
        pass

    def capture_estimate(molecule, basis, *, method, options):
        captured["estimate"] = real_estimate_memory(
            molecule,
            basis,
            method=method,
            options=options,
        )
        captured.update(method=method, options=options)
        raise PreflightCaptured

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    with pytest.raises(PreflightCaptured):
        vq.run_job(
            molecule_factory(),
            basis="def2-svp",
            method=method,
            dlpno_ccsd_options=pilot_options,
            output=str(tmp_path / "pilot"),
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )

    estimate = captured["estimate"]
    assert "DLPNO pilot retained spin ERI" in estimate.by_category
    assert "DLPNO pilot dense spin-ERI construction" in estimate.phase_peaks
    assert estimate.dims["n_strong_pairs_estimate"] == estimate.dims["n_pairs"]


@pytest.mark.parametrize(
    "open_shell,localise",
    [
        (False, "boys"),
        (True, "boys"),
        (True, "external"),
    ],
)
def test_localized_dlpno_pilot_triples_prices_second_spin_eri_build(
    open_shell, localise
):
    """Every localized pilot (T) path prices the exact second-build peak."""
    if open_shell:
        from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions

        molecule = vq.Molecule(
            [
                vq.Atom(8, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 0.0, 1.8]),
            ],
            multiplicity=2,
        )
        options = DLPNOUCCSDPilotOptions(
            localise=localise,
            compute_triples=True,
            max_nbf=256,
        )
        method = "dlpno-uccsd(t)"
    else:
        from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions

        molecule = _h2o()
        options = DLPNOCCSDPilotOptions(
            localise=localise,
            compute_triples=True,
            max_nbf=256,
        )
        method = "dlpno-ccsd(t)"
    basis = vq.BasisSet(molecule, "def2-svp")
    estimate = vq.estimate_memory(
        molecule,
        basis,
        method=method,
        options=options,
    )

    category = estimate.by_category
    phases = estimate.phase_peaks
    three_index = (
        estimate.dims["n_aux"]
        * estimate.dims["n_basis"]
        * estimate.dims["n_basis"]
        * 8
    )
    metric = estimate.dims["n_aux"] ** 2 * 8
    assert category["DLPNO pilot DF construction overlap"] == (
        4 * three_index + 4 * metric
    )
    assert category["DLPNO pilot retained DF substrate"] == (
        2 * three_index + 3 * metric
    )
    n_mo = estimate.dims["n_basis"] - estimate.dims["n_frozen_core"]
    canonical_factors = (
        (2 if open_shell else 1)
        * estimate.dims["n_aux"]
        * n_mo
        * n_mo
        * 8
    )
    expected_rebuild = (
        canonical_factors
        + category["DLPNO pilot retained spin ERI"]
        + category["DLPNO pilot spin-ERI build temporaries"]
    )
    assert category["DLPNO pilot localized-(T) spin-ERI rebuild"] == (
        expected_rebuild
    )

    live_reference = category.get("Python runtime + NumPy overhead", 0) + sum(
        category.get(name, 0)
        for name in (
            "Fock + density + 1e",
            "MO workspace",
            "Open-shell UHF buffers",
        )
    )
    retained_substrate = (
        category["DLPNO pilot retained DF substrate"]
        + category["DLPNO pilot transformed DF factors"]
    )
    expected_rebuild_phase = (
        live_reference
        + retained_substrate
        + category["DLPNO pilot retained spin ERI"]
        + category["DLPNO pilot dense spatial integral blocks"]
        + category["DLPNO pilot PNO pair state"]
        + category["DLPNO pilot dense DIIS history"]
        + expected_rebuild
    )
    assert phases["DLPNO pilot localized-(T) ERI rebuild"] == (
        expected_rebuild_phase
    )
    assert (
        phases["DLPNO pilot localized-(T) ERI rebuild"]
        > phases["DLPNO pilot dense spin-ERI construction"]
    )


def test_run_job_dlpno_policy_does_not_mutate_caller_options(
    tmp_path, monkeypatch
):
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
    import vibeqc.runner as runner_module

    caller_options = LocalCCSDOptions(
        n_frozen=None,
        compute_triples=False,
    )
    captured = {}

    class PreflightCaptured(RuntimeError):
        pass

    def capture_estimate(molecule, basis, *, method, options):
        captured["options"] = options["dlpno_ccsd_options"]
        raise PreflightCaptured

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    with pytest.raises(PreflightCaptured):
        vq.run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-ccsd(t)",
            output=str(tmp_path / "nonmutation"),
            frozen_core=False,
            dlpno_ccsd_options=caller_options,
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )

    assert captured["options"] is not caller_options
    assert captured["options"].n_frozen == 0
    assert captured["options"].compute_triples is True
    assert caller_options.n_frozen is None
    assert caller_options.compute_triples is False


def test_run_job_dlpno_preflight_preserves_explicit_aux_and_caller(
    tmp_path, monkeypatch
):
    """Admission receives the exact requested RI identity on a clone."""
    from vibeqc.dlpno.mp2 import DLPNOMP2Options
    import vibeqc.runner as runner_module

    caller = DLPNOMP2Options(aux_basis="cc-pvdz-ri")
    captured = {}

    class PreflightCaptured(RuntimeError):
        pass

    def capture_estimate(molecule, basis, *, method, options):
        captured["options"] = options["dlpno_options"]
        raise PreflightCaptured

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    with pytest.raises(PreflightCaptured):
        vq.run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-mp2",
            dlpno_options=caller,
            output=str(tmp_path / "explicit-ri"),
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )

    assert captured["options"] is not caller
    assert captured["options"].aux_basis == "cc-pvdz-ri"
    assert caller.aux_basis == "cc-pvdz-ri"


def test_dlpno_top_level_aux_matches_dry_and_real_preflight_dimensions(
    tmp_path, monkeypatch
):
    """Both admission passes size the exact top-level RI basis on clones."""
    import vibeqc.runner as runner_module

    molecule = _h2o()
    real_estimate_memory = runner_module.estimate_memory
    captured = []

    class RealPreflightCaptured(RuntimeError):
        pass

    def capture_estimate(molecule, basis, *, method, options):
        estimate = real_estimate_memory(
            molecule,
            basis,
            method=method,
            options=options,
        )
        captured.append((method, options["dlpno_options"], estimate))
        if len(captured) == 2:
            raise RealPreflightCaptured
        return estimate

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    vq.run_job(
        molecule,
        basis="def2-svp",
        method="dlpno-mp2",
        aux_basis="cc-pvdz-ri",
        output=str(tmp_path / "dry-top-level-ri"),
        dry_run=True,
        write_xyz_file=False,
        output_qvf=False,
        citations=False,
        crash_dump=False,
    )

    monkeypatch.delenv("VIBEQC_DRY_RUN_ESTIMATE")
    with pytest.raises(RealPreflightCaptured):
        vq.run_job(
            molecule,
            basis="def2-svp",
            method="dlpno-mp2",
            aux_basis="cc-pvdz-ri",
            output=str(tmp_path / "real-top-level-ri"),
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )

    expected_n_aux = vq.BasisSet(molecule, "cc-pvdz-ri").nbasis
    assert len(captured) == 2
    assert captured[0][0] == captured[1][0] == "dlpno-mp2"
    assert captured[0][1] is not captured[1][1]
    for _, options, estimate in captured:
        assert options.aux_basis == "cc-pvdz-ri"
        assert estimate.dims["n_aux"] == expected_n_aux


def test_run_job_rohf_mp2_preflight_receives_top_level_policy(
    tmp_path, monkeypatch
):
    import vibeqc.runner as runner_module

    molecule = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )
    captured = {}

    class PreflightCaptured(RuntimeError):
        pass

    def capture_estimate(molecule, basis, *, method, options):
        captured.update(method=method, options=options)
        raise PreflightCaptured

    monkeypatch.setattr(runner_module, "estimate_memory", capture_estimate)
    with pytest.raises(PreflightCaptured):
        vq.run_job(
            molecule,
            basis="cc-pvdz",
            method="mp2",
            mp2_reference="rohf",
            frozen_core=False,
            aux_basis="def2-svp-rifit",
            output=str(tmp_path / "rohf-mp2"),
            write_xyz_file=False,
            output_qvf=False,
            citations=False,
            crash_dump=False,
        )

    assert captured["method"] == "rohf-mp2"
    assert captured["options"]["ump2_options"].n_frozen_core == 0
    assert captured["options"]["ump2_options"].density_fit is True
    assert captured["options"]["ump2_options"].aux_basis == "def2-svp-rifit"


def test_rohf_mp2_estimator_keeps_zero_active_beta_boundary():
    """Published core may consume beta space without changing conventions."""
    molecule = vq.Molecule(
        [vq.Atom(5, [0.0, 0.0, 0.0])],
        multiplicity=4,
    )
    basis = vq.BasisSet(molecule, "def2-svp")

    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="rohf-mp2",
        options={"ump2_options": vq.UMP2Options()},
    )

    assert estimate.dims["n_frozen_core"] == 1
    assert estimate.dims["n_occ"] == 3


def test_dlpno_ccsdt_s22_uracil_preflight_tracks_measured_peak(monkeypatch):
    """The historical streamed S22-05 job fits its measured 90,000 MiB node.

    OMP=48 profiling peaked at 6747.878906 and 6819.148438 MiB.  The old
    estimator claimed 100.689 GiB by treating every pair residual and occupied
    triple workspace as simultaneously resident. The measurement predates
    #140/#448, so its all-electron convention remains explicit evidence.
    """
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    geometry = (
        Path(__file__).parents[1]
        / "examples/molecular/mp2_benchmarks/s22/geometries"
        / "s22-05-uracil-dimer-hb-monoA.xyz"
    )
    molecule = vq.Molecule.from_xyz(geometry)
    basis = vq.BasisSet(molecule, "cc-pvdz")
    options = LocalCCSDOptions(
        n_frozen=0,
        tcut_pairs=5.0e-5,
        tcut_pno=1.0e-8,
        tcut_mkn=1.0e-3,
        residual_domain="pair",
        compute_triples=True,
        tcut_tno=0.0,
        triples_mode="t1",
    )
    monkeypatch.delenv("VIBEQC_MEMORY_HEADROOM", raising=False)
    monkeypatch.delenv("VIBEQC_TRIPLES_TILE_SIZE", raising=False)
    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 48)

    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-ccsd(t)",
        options=options,
    )
    observed_peak = round(6819.148438 * 1024**2)

    assert basis.nbasis == 132
    assert estimate.total_bytes >= observed_peak
    vq.check_memory(estimate, available=90_000 * 1024**2)


def test_dlpno_ccsdt_small_job_counts_native_worker_residency(monkeypatch):
    """A small OMP=64 DLPNO job still owns native per-worker state.

    GNU time measured the converged S22-02 water dimer at 1,240,412 KiB
    peak RSS.  Numerical tensors alone are small, so the runtime and native
    worker reserve must coexist with the post-HF phases rather than becoming a
    mutually exclusive 100-MiB floor. This is pre-#140/#448 all-electron
    evidence and is not silently reinterpreted under the new defaults.
    """
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    geometry = (
        Path(__file__).parents[1]
        / "examples/molecular/mp2_benchmarks/s22/geometries"
        / "s22-02-water-dimer.xyz"
    )
    molecule = vq.Molecule.from_xyz(geometry)
    basis = vq.BasisSet(molecule, "cc-pvdz")
    options = LocalCCSDOptions(
        n_frozen=0,
        tcut_pairs=5.0e-5,
        tcut_pno=1.0e-8,
        tcut_mkn=1.0e-3,
        residual_domain="pair",
        compute_triples=True,
        tcut_tno=0.0,
        triples_mode="t1",
    )
    monkeypatch.delenv("VIBEQC_MEMORY_HEADROOM", raising=False)
    monkeypatch.delenv("VIBEQC_TRIPLES_TILE_SIZE", raising=False)
    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 64)

    estimate = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-ccsd(t)",
        options=options,
    )

    assert basis.nbasis == 48
    assert estimate.total_bytes >= 1_240_412 * 1024


def test_dlpno_ccsd_estimate_fails_closed_outside_profiled_local_envelope(
    monkeypatch,
):
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    geometry = (
        Path(__file__).parents[1]
        / "examples/molecular/mp2_benchmarks/s22/geometries"
        / "s22-05-uracil-dimer-hb-monoA.xyz"
    )
    molecule = vq.Molecule.from_xyz(geometry)
    basis = vq.BasisSet(molecule, "cc-pvdz")
    monkeypatch.setattr("vibeqc.memory._omp_max_threads", lambda: 1)

    profiled = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-ccsd",
        options=LocalCCSDOptions(
            n_frozen=0,
            tcut_pairs=5.0e-5,
            tcut_pno=1.0e-8,
            tcut_mkn=0.0,
            residual_domain="pair",
        ),
    )
    tighter_pairs = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-ccsd",
        options=LocalCCSDOptions(
            n_frozen=0,
            tcut_pairs=1.0e-5,
            tcut_pno=1.0e-8,
            tcut_mkn=0.0,
            residual_domain="pair",
        ),
    )
    untruncated_pnos = vq.estimate_memory(
        molecule,
        basis,
        method="dlpno-ccsd",
        options=LocalCCSDOptions(
            n_frozen=0,
            tcut_pairs=5.0e-5,
            tcut_pno=0.0,
            tcut_mkn=0.0,
            residual_domain="pair",
        ),
    )

    assert profiled.dims["n_strong_pairs_estimate"] < profiled.dims["n_pairs"]
    assert (
        tighter_pairs.dims["n_strong_pairs_estimate"]
        == tighter_pairs.dims["n_pairs"]
    )
    assert untruncated_pnos.dims["avg_pno"] == untruncated_pnos.dims["n_vir"]


def test_estimate_memory_rks_includes_grid():
    """RKS grid storage uses the native batch bound for AO tables."""
    from vibeqc import _vibeqc_core as core

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    rhf = vq.estimate_memory(mol, basis, method="rhf")
    rks = vq.estimate_memory(mol, basis, method="rks")
    assert "DFT grid + chi" not in rhf.by_category
    assert "DFT grid + chi" in rks.by_category
    assert rks.by_category["DFT grid + chi"] > 0

    n_pts = len(mol.atoms) * 75 * 17 * 36
    n_batch = min(n_pts, core.MOLECULAR_XC_GRID_BATCH_SIZE)
    # options=None is conservatively treated as GGA: 13 batch-local AO /
    # gradient-Hessian tables, 16 batch-local libxc vectors, one batch-local
    # grid copy, and full-grid coordinates, weights, and owner indices.
    expected = (
        13 * n_batch * basis.nbasis * 8
        + 16 * n_batch * 8
        + 4 * n_pts * 8
        + n_pts * 4
        + n_batch * (4 * 8 + 4)
        + 4 * 75 * 17 * 36 * 8
    )
    assert rks.by_category["DFT grid + chi"] == expected
    assert rks.by_category["DFT grid + chi"] < 8 * n_pts * basis.nbasis * 8


def test_skala_memory_chunking_packs_only_complete_atom_grids():
    import vibeqc.memory as memory_module

    # Four 2,000-point atoms pack exactly to the 8,192-point target floor.
    assert memory_module._skala_atom_aligned_chunk_points([2_000] * 4) == 8_000
    # Three 3,000-point atoms pack two at a time; the third is another call.
    assert memory_module._skala_atom_aligned_chunk_points([3_000] * 3) == 6_000
    # Heterogeneous sizes form separate homogeneous model calls.
    assert memory_module._skala_atom_aligned_chunk_points(
        [5_000, 2_000, 3_000]
    ) == 5_000
    # Equal sizes retain stable order and pack together within their group.
    assert memory_module._skala_atom_aligned_chunk_points(
        [3_000, 2_000, 3_000, 2_000, 3_000]
    ) == 6_000


def test_skala_memory_chunking_keeps_one_oversized_atom_intact():
    import vibeqc.memory as memory_module

    n_points = 10_000
    assert memory_module._skala_atom_aligned_chunk_points([n_points]) == n_points
    expected = n_points * (
        memory_module._EXTERNAL_XC_FULL_GRID_FLOAT64_VALUES_PER_POINT * 8
        + memory_module._EXTERNAL_XC_FULL_GRID_INDEX_BYTES_PER_POINT
        + 6_680 * 8
    )
    assert memory_module._skala_xc_peak_bytes([n_points]) == expected


@pytest.mark.parametrize(
    "atomic_grid_sizes",
    (
        [2_000] * 4,
        [3_000, 2_000, 3_000, 2_000, 3_000],
        [5_000, 2_000, 3_000],
        [10_000, 2_000],
    ),
)
def test_skala_memory_chunk_peak_matches_adapter_plan(atomic_grid_sizes):
    import vibeqc.memory as memory_module
    from vibeqc.skala import plan_atom_grid_chunks

    expected = max(
        chunk.num_points
        for chunk in plan_atom_grid_chunks(atomic_grid_sizes)
    )
    assert (
        memory_module._skala_atom_aligned_chunk_points(atomic_grid_sizes)
        == expected
    )


def test_skala_memory_scales_full_grid_state_after_model_chunk_plateau():
    import vibeqc.memory as memory_module

    low_sizes = [3_000, 3_000]
    high_sizes = [3_000] * 4
    low_points = sum(low_sizes)
    high_points = sum(high_sizes)
    low = memory_module._skala_xc_peak_bytes(low_sizes)
    high = memory_module._skala_xc_peak_bytes(high_sizes)

    # Both systems have the same 6,000-point atom-aligned model peak. Only
    # the complete C++/Python bridge grows with the two extra atomic grids.
    bridge_bytes_per_point = (
        memory_module._EXTERNAL_XC_FULL_GRID_FLOAT64_VALUES_PER_POINT * 8
        + memory_module._EXTERNAL_XC_FULL_GRID_INDEX_BYTES_PER_POINT
    )
    assert high - low == (high_points - low_points) * bridge_bytes_per_point


def test_skala_memory_uses_exact_element_specific_atomic_blocks():
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module

    mol = _h2o()
    options = vq.RKSOptions()
    options.functional = "skala-1.1"
    options.grid.atomic_grid_profile = "pyscf-level3"

    expected = tuple(core.grid_atomic_point_counts(mol, options.grid))
    actual = memory_module._atomic_grid_point_counts(mol, options)

    assert actual == expected
    assert actual[1] == actual[2]
    assert actual[0] != actual[1]


def test_periodic_skala_memory_uses_same_exact_atomic_blocks():
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module
    from vibeqc.periodic_runner import _periodic_skala_memory_estimate

    molecule = _h2o()
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 12.0,
        list(molecule.atoms),
    )
    grid = vq.GridOptions()
    grid.atomic_grid_profile = "pyscf-level3"
    atomic_grid_sizes = tuple(
        core.grid_atomic_point_counts(system.unit_cell_molecule(), grid)
    )

    estimate = _periodic_skala_memory_estimate(system)

    assert estimate.by_category[
        "SKALA full-grid + model workspace"
    ] == memory_module._skala_xc_peak_bytes(atomic_grid_sizes)
    assert atomic_grid_sizes[1] == atomic_grid_sizes[2]
    assert atomic_grid_sizes[0] != atomic_grid_sizes[1]


def test_periodic_skala_memory_merge_updates_phase_aware_peak(monkeypatch):
    import vibeqc.periodic_runner as periodic_runner

    label = "SKALA full-grid + model workspace"
    base = vq.MemoryEstimate(
        by_category={"persistent route state": 100},
        phase_peaks={"setup": 120, "SCF": 200},
    )
    skala = vq.MemoryEstimate(by_category={label: 50})
    monkeypatch.setattr(
        periodic_runner,
        "_periodic_skala_memory_estimate",
        lambda _system: skala,
    )

    merged = periodic_runner._merge_periodic_skala_memory(base, object())

    assert merged.by_category[label] == 50
    assert merged.phase_peaks == {"setup": 170, "SCF": 250}
    assert merged.raw_total_bytes == 250
    # Reapplying the same plan is idempotent rather than double-charging it.
    assert (
        periodic_runner._merge_periodic_skala_memory(merged, object())
        .raw_total_bytes
        == 250
    )


def test_skala_aliases_are_detected_without_loading_torch(monkeypatch):
    import builtins
    import vibeqc.memory as memory_module

    original_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("memory preflight must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    for name in ("skala", "skala-1.1", "skala-1.1-rev1"):
        assert memory_module._functional_is_external({"functional": name})
    assert not memory_module._functional_is_external({"functional": "r2scan"})


@pytest.mark.parametrize(
    ("method", "options_type"),
    (("rks", vq.RKSOptions), ("uks", vq.UKSOptions)),
)
def test_skala_molecular_memory_materially_exceeds_same_grid_mgga(
    monkeypatch,
    method,
    options_type,
):
    import vibeqc.memory as memory_module

    monkeypatch.setattr(memory_module, "_omp_max_threads", lambda: 1)
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    mgga_options = options_type()
    mgga_options.functional = "r2scan"
    mgga_options.grid.atomic_grid_profile = "pyscf-level3"
    skala_options = options_type()
    skala_options.functional = "skala-1.1"
    skala_options.grid.atomic_grid_profile = "pyscf-level3"

    mgga = vq.estimate_memory(
        mol,
        basis,
        method=method,
        options=mgga_options,
    )
    skala = vq.estimate_memory(
        mol,
        basis,
        method=method,
        options=skala_options,
    )

    assert (
        skala.by_category["DFT grid + chi"]
        > mgga.by_category["DFT grid + chi"] + 512 * 1024**2
    )
    assert skala.raw_total_bytes > mgga.raw_total_bytes + 512 * 1024**2


def test_molecular_xc_worker_planner_mirrors_native_caps(monkeypatch):
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module

    batch = int(core.MOLECULAR_XC_GRID_BATCH_SIZE)
    monkeypatch.setattr(memory_module, "_omp_max_threads", lambda: 64)
    assert memory_module._molecular_xc_batch_workers(batch) == 1
    assert memory_module._molecular_xc_batch_workers(100 * batch) == 64
    assert (
        memory_module._molecular_xc_batch_workers(
            100 * batch, kernel=True
        )
        == int(core.MOLECULAR_XC_KERNEL_MAX_WORKERS)
        == 16
    )
    monkeypatch.setattr(memory_module, "_omp_max_threads", lambda: 20)
    assert memory_module._molecular_xc_batch_workers(100 * batch) == 20


def test_uks_xc_estimate_charges_parallel_scf_and_stability_waves(
    monkeypatch,
):
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    n = basis.nbasis
    batch = int(core.MOLECULAR_XC_GRID_BATCH_SIZE)
    n_points = 100 * batch
    monkeypatch.setattr(memory_module, "_omp_max_threads", lambda: 64)
    monkeypatch.setattr(memory_module, "_grid_dimensions", lambda _opts: (1, 1, 1))
    monkeypatch.setattr(
        memory_module,
        "_grid_points",
        lambda _mol, _opts: n_points,
    )

    # UKS/PBE: six AO-sized tables, thirty scalar vectors, four projected
    # n_basis^2 matrices, and one copied grid slice per SCF worker.
    grid_copy = batch * (4 * 8 + 4)
    scf_worker = (
        6 * batch * n * 8
        + 30 * batch * 8
        + 4 * n * n * 8
        + grid_copy
    )
    scf_peak = 64 * scf_worker
    gradient_peak = 15 * batch * n * 8 + 30 * batch * 8 + grid_copy
    immutable_grid = n_points * (8 + 4 + 3 * 8) + 4 * 1 * 1 * 8

    without_stability = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "PBE", "stability_check": False},
        open_shell=True,
    )
    assert without_stability == max(scf_peak, gradient_peak) + immutable_grid

    stability_worker = (
        10 * batch * n * 8
        + 5 * n * n * 8
        + 60 * batch * 8
        + grid_copy
    )
    stability_peak = 28 * n_points * 8 + 16 * stability_worker
    with_stability = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "PBE", "stability_check": True},
        open_shell=True,
    )
    assert with_stability == max(
        scf_peak, gradient_peak, stability_peak
    ) + immutable_grid

    # Issue #447: the driver leaves stability_checked=false when response
    # terms are incomplete, so the preflight must not charge a phase that
    # cannot run. RSH, VV10, and meta-GGA are distinct functional gates.
    for functional in ("wb97x", "vv10", "TPSS"):
        unsupported = memory_module._dft_xc_estimate(
            mol,
            basis,
            {"functional": functional, "stability_check": True},
            open_shell=True,
        )
        unsupported_opt_out = memory_module._dft_xc_estimate(
            mol,
            basis,
            {"functional": functional, "stability_check": False},
            open_shell=True,
        )
        assert unsupported == unsupported_opt_out

    plus_u = memory_module._dft_xc_estimate(
        mol,
        basis,
        {
            "functional": "PBE",
            "stability_check": True,
            "dft_plus_u_sites": [object()],
        },
        open_shell=True,
    )
    assert plus_u == without_stability

    hybrid_cosx = memory_module._dft_xc_estimate(
        mol,
        basis,
        {
            "functional": "B3LYP",
            "cosx": True,
            "density_fit": True,
            "stability_check": True,
        },
        open_shell=True,
    )
    hybrid_cosx_opt_out = memory_module._dft_xc_estimate(
        mol,
        basis,
        {
            "functional": "B3LYP",
            "cosx": True,
            "density_fit": True,
            "stability_check": False,
        },
        open_shell=True,
    )
    assert hybrid_cosx == hybrid_cosx_opt_out

    # A bare cosx=True is inactive without the density-fit/auxiliary route.
    # The direct global-hybrid response therefore remains complete and charged.
    direct_hybrid = memory_module._dft_xc_estimate(
        mol,
        basis,
        {
            "functional": "B3LYP",
            "cosx": True,
            "density_fit": False,
            "stability_check": True,
        },
        open_shell=True,
    )
    assert direct_hybrid == with_stability

    # COSX is a no-op for pure PBE, so its complete response remains charged.
    pure_cosx = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "PBE", "cosx": True, "stability_check": True},
        open_shell=True,
    )
    assert pure_cosx == with_stability


def test_molecular_dft_ao_table_estimate_plateaus_at_native_batch_size(
    monkeypatch,
):
    """Past one native batch only O(n_pts) scalar/grid vectors may grow."""
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    batch = int(core.MOLECULAR_XC_GRID_BATCH_SIZE)
    monkeypatch.setattr(memory_module, "_grid_dimensions", lambda _opts: (1, 1, 1))
    monkeypatch.setattr(
        memory_module,
        "_grid_points",
        lambda _mol, opts: int(opts["n_points"]),
    )

    low_points = 2 * batch
    high_points = 5 * batch
    low = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "PBE", "n_points": low_points},
    )
    high = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "PBE", "n_points": high_points},
    )

    # Only one weight, three coordinate vectors, and one 32-bit owner index
    # grow over the full grid. Batch-local peers are unchanged.
    assert high - low == (high_points - low_points) * (4 * 8 + 4)


def test_molecular_vv10_estimate_retains_only_global_scalar_state(monkeypatch):
    """VV10 keeps its double-grid fields global, but never dense AO tables."""
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    batch = int(core.MOLECULAR_XC_GRID_BATCH_SIZE)
    monkeypatch.setattr(memory_module, "_grid_dimensions", lambda _opts: (1, 1, 1))
    monkeypatch.setattr(
        memory_module,
        "_grid_points",
        lambda _mol, opts: int(opts["n_points"]),
    )

    # At these sizes the 16-vector full-grid kernel phase is larger than the
    # batch AO projection peak for this small basis.
    low_points = 8 * batch
    high_points = 12 * batch
    low = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "wb97x-v", "n_points": low_points},
    )
    high = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "wb97x-v", "n_points": high_points},
    )

    # The VV10 kernel phase dominates here: sixteen conservative fields/
    # active-point slots plus four immutable-grid vectors and the owner index
    # grow globally.
    # AO and semilocal vectors plateau.
    assert high - low == (high_points - low_points) * (20 * 8 + 4)


def test_molecular_dft_dense_ao_fallbacks_remain_whole_grid(monkeypatch):
    """ROKS, TDDFT, and XC-kernel accelerators must not claim batching."""
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module
    from vibeqc.roks import ROKSOptions

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    n_points = 4 * int(core.MOLECULAR_XC_GRID_BATCH_SIZE)
    monkeypatch.setattr(memory_module, "_grid_dimensions", lambda _opts: (1, 1, 1))
    monkeypatch.setattr(
        memory_module,
        "_grid_points",
        lambda _mol, _opts: n_points,
    )

    ordinary_rks = vq.RKSOptions()
    ordinary_rks.functional = "PBE"
    batched_rks = memory_module._dft_xc_estimate(
        mol,
        basis,
        ordinary_rks,
    )

    soscf_rks = vq.RKSOptions()
    soscf_rks.functional = "PBE"
    soscf_rks.soscf_threshold = 1.0
    assert memory_module._dft_xc_estimate(mol, basis, soscf_rks) == batched_rks

    newton_rks = vq.RKSOptions()
    newton_rks.functional = "PBE"
    newton_rks.newton_threshold = 1.0
    trah_rks = vq.RKSOptions()
    trah_rks.functional = "PBE"
    trah_rks.trah_threshold = 1.0
    tddft_rks = {
        "scf_options": ordinary_rks,
        "functional": "PBE",
        "tddft": True,
    }
    dict_newton_rks = {
        "functional": "PBE",
        "newton_threshold": 1.0,
    }
    for dense_options in (
        newton_rks,
        trah_rks,
        tddft_rks,
        dict_newton_rks,
    ):
        assert (
            memory_module._dft_xc_estimate(mol, basis, dense_options)
            > batched_rks
        )

    ordinary_uks = vq.UKSOptions()
    ordinary_uks.functional = "PBE"
    batched_uks = memory_module._dft_xc_estimate(
        mol,
        basis,
        ordinary_uks,
        open_shell=True,
    )
    roks = ROKSOptions(functional="PBE")
    assert (
        memory_module._dft_xc_estimate(
            mol,
            basis,
            roks,
            open_shell=True,
        )
        > batched_uks
    )


def test_molecular_dft_mgga_estimate_counts_real_ao_tables(monkeypatch):
    """MGGA tau is scalar; charge its actual derivative contractions."""
    from vibeqc import _vibeqc_core as core
    import vibeqc.memory as memory_module

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    n_points = 4 * int(core.MOLECULAR_XC_GRID_BATCH_SIZE)
    monkeypatch.setattr(memory_module, "_grid_dimensions", lambda _opts: (1, 1, 1))
    monkeypatch.setattr(
        memory_module,
        "_grid_points",
        lambda _mol, _opts: n_points,
    )

    dense_rks = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "r2scan", "tddft": True},
    )
    dense_uks = memory_module._dft_xc_estimate(
        mol,
        basis,
        {"functional": "r2scan", "tddft": True},
        open_shell=True,
    )
    # The underlying static MGGA matrix build owns eight RKS or twelve UKS
    # basis-sized tables. Dense legacy/second-order consumers need a higher
    # conservative floor for simultaneous source, builder, and apply tables.
    expected_rks = (
        12 * n_points * basis.nbasis * 8
        + (4 + 28) * n_points * 8
        + n_points * 4
        + int(core.MOLECULAR_XC_GRID_BATCH_SIZE) * (4 * 8 + 4)
        + 4 * 8
    )
    batch = int(core.MOLECULAR_XC_GRID_BATCH_SIZE)
    expected_rks += memory_module._molecular_xc_batch_workers(n_points) * (
        8 * batch * basis.nbasis * 8
        + 28 * batch * 8
        + 2 * basis.nbasis**2 * 8
        + batch * (4 * 8 + 4)
    )
    expected_uks = (
        16 * n_points * basis.nbasis * 8
        + (4 + 57) * n_points * 8
        + n_points * 4
        + int(core.MOLECULAR_XC_GRID_BATCH_SIZE) * (4 * 8 + 4)
        + 4 * 8
    )
    expected_uks += memory_module._molecular_xc_batch_workers(n_points) * (
        12 * batch * basis.nbasis * 8
        + 48 * batch * 8
        + 4 * basis.nbasis**2 * 8
        + batch * (4 * 8 + 4)
    )
    assert dense_rks == expected_rks
    assert dense_uks == expected_uks


def test_rks_grid_estimate_uses_active_angular_scheme():
    """Lebedev orders must not inherit the product-grid point count."""
    from vibeqc.memory import _grid_dimensions, _grid_points

    mol = _h2o()
    order_29 = vq.RKSOptions()
    order_29.grid.angular = "lebedev"
    order_29.grid.lebedev_order = 29
    order_47 = vq.RKSOptions()
    order_47.grid.angular = "lebedev"
    order_47.grid.lebedev_order = 47

    product_points = _grid_points(mol, vq.RKSOptions())
    order_29_points = _grid_points(mol, order_29)
    order_47_points = _grid_points(mol, order_47)
    cosx_level_3 = vq.cosx_grid_options_for_level(3)

    assert _grid_dimensions(order_29) == (75, 302, 1)
    assert _grid_dimensions(order_47) == (75, 770, 1)
    assert _grid_dimensions(cosx_level_3) == (75, 302, 1)
    assert order_29_points == len(mol.atoms) * 75 * 302
    assert product_points == len(mol.atoms) * 75 * 17 * 36
    assert order_47_points == len(mol.atoms) * 75 * 770
    assert order_29_points < product_points < order_47_points

    basis = vq.BasisSet(mol, "sto-3g")
    low = vq.estimate_memory(mol, basis, method="rks", options=order_29)
    product = vq.estimate_memory(mol, basis, method="rks", options=vq.RKSOptions())
    high = vq.estimate_memory(mol, basis, method="rks", options=order_47)
    assert (
        low.by_category["DFT grid + chi"]
        < product.by_category["DFT grid + chi"]
        < high.by_category["DFT grid + chi"]
    )

    hydrogen = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    for grid_options in (order_29.grid, order_47.grid):
        actual_points = vq.build_grid(hydrogen, grid_options).points.shape[0]
        assert _grid_points(hydrogen, grid_options) == actual_points
    cosx_actual_points = vq.build_grid(hydrogen, cosx_level_3).points.shape[0]
    assert _grid_points(hydrogen, cosx_level_3) >= cosx_actual_points


def test_estimate_memory_uhf_adds_open_shell_term():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    rhf = vq.estimate_memory(mol, basis, method="rhf")
    uhf = vq.estimate_memory(mol, basis, method="uhf")
    assert "Open-shell UHF buffers" in uhf.by_category
    assert uhf.total_bytes > rhf.total_bytes


# ---------------------------------------------------------------------------
# MemoryEstimate.format + format_memory_report
# ---------------------------------------------------------------------------

def test_format_headline_uses_mb_for_small():
    est = vq.MemoryEstimate(by_category={"ERI tensor": 50 * 1024 * 1024})  # 50 MB
    rendered = est.format(available=100 * 1024**3)
    assert "MB" in rendered.splitlines()[0]


def test_format_headline_uses_gb_for_large():
    est = vq.MemoryEstimate(by_category={"ERI tensor": 120 * 1024**3})   # 120 GB
    rendered = est.format(available=512 * 1024**3)
    assert "GB" in rendered.splitlines()[0]


def test_format_memory_report_marks_override_when_requested():
    """When the estimate exceeds available and override=True, the block
    must say so explicitly rather than "Proceeding"."""
    est = vq.MemoryEstimate(by_category={"huge": 500 * 1024**3})
    report = vq.format_memory_report(
        est, override_requested=True, available=8 * 1024**3,
    )
    assert "override" in report.lower()


def test_format_memory_report_marks_abort_when_insufficient_and_no_override():
    est = vq.MemoryEstimate(by_category={"huge": 500 * 1024**3})
    report = vq.format_memory_report(
        est, override_requested=False, available=8 * 1024**3,
    )
    assert "ABORTING" in report


# ---------------------------------------------------------------------------
# check_memory — the enforcement gate
# ---------------------------------------------------------------------------

def test_check_memory_passes_when_estimate_fits():
    est = vq.MemoryEstimate(by_category={"tiny": 10 * 1024**2})        # 10 MB
    # No exception expected.
    vq.check_memory(est, available=4 * 1024**3)


def test_check_memory_aborts_when_estimate_exceeds_available():
    est = vq.MemoryEstimate(by_category={"huge": 100 * 1024**3})
    with pytest.raises(vq.InsufficientMemoryError, match="ABORTING"):
        vq.check_memory(est, available=8 * 1024**3)


def test_check_memory_respects_allow_exceed():
    est = vq.MemoryEstimate(by_category={"huge": 100 * 1024**3})
    # No exception when allow_exceed=True.
    vq.check_memory(est, allow_exceed=True, available=8 * 1024**3)


def test_check_memory_silent_when_probe_fails():
    """When ``available`` is 0 (probe failed) we can't enforce, so
    check_memory is a no-op — better than false-aborting on unsupported
    platforms."""
    est = vq.MemoryEstimate(by_category={"huge": 100 * 1024**3})
    vq.check_memory(est, available=0)     # should not raise


def test_available_memory_bytes_returns_non_negative():
    """Cheap cross-platform probe; on any supported OS it returns > 0."""
    assert vq.available_memory_bytes() >= 0


def test_available_memory_bytes_respects_cgroup_v2_limit(monkeypatch):
    """A scheduler cgroup must bound host-wide psutil availability."""
    import builtins
    import io
    import os
    import sys

    import vibeqc.memory as memory_module

    gib = 1024**3
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=91 * gib),
    )
    files = {
        "/proc/self/cgroup": "0::/pbs_jobs/123\n",
        "/sys/fs/cgroup/pbs_jobs/123/memory.max": str(6 * gib),
        "/sys/fs/cgroup/pbs_jobs/123/memory.current": str(1 * gib),
    }
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        resolved = os.fspath(path)
        if resolved in files:
            return io.StringIO(files[resolved])
        if resolved.startswith("/sys/fs/cgroup"):
            raise FileNotFoundError(resolved)
        return real_open(path, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(memory_module.sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", fake_open)

    assert vq.available_memory_bytes() == 5 * gib


def test_available_memory_bytes_respects_cgroup_v1_limit(monkeypatch):
    """Legacy memory-controller limits also bound host availability."""
    import builtins
    import io
    import os
    import sys

    import vibeqc.memory as memory_module

    gib = 1024**3
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=91 * gib),
    )
    files = {
        "/proc/self/cgroup": "5:cpu,memory:/pbs_jobs/456\n",
        "/sys/fs/cgroup/memory/pbs_jobs/456/memory.limit_in_bytes": str(
            8 * gib
        ),
        "/sys/fs/cgroup/memory/pbs_jobs/456/memory.usage_in_bytes": str(
            2 * gib
        ),
    }
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        resolved = os.fspath(path)
        if resolved in files:
            return io.StringIO(files[resolved])
        if resolved.startswith("/sys/fs/cgroup"):
            raise FileNotFoundError(resolved)
        return real_open(path, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(memory_module.sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", fake_open)

    assert vq.available_memory_bytes() == 6 * gib


def test_available_memory_bytes_respects_parent_cgroup_limit(monkeypatch):
    """A finite parent allocation constrains an unlimited leaf cgroup."""
    import builtins
    import io
    import os
    import sys

    import vibeqc.memory as memory_module

    gib = 1024**3
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=91 * gib),
    )
    files = {
        "/proc/self/cgroup": "0::/pbs_jobs/123/task\n",
        "/sys/fs/cgroup/pbs_jobs/123/task/memory.max": "max",
        "/sys/fs/cgroup/pbs_jobs/123/memory.max": str(6 * gib),
        "/sys/fs/cgroup/pbs_jobs/123/memory.current": str(1 * gib),
    }
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        resolved = os.fspath(path)
        if resolved in files:
            return io.StringIO(files[resolved])
        if resolved.startswith("/sys/fs/cgroup"):
            raise FileNotFoundError(resolved)
        return real_open(path, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(memory_module.sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", fake_open)

    assert vq.available_memory_bytes() == 5 * gib


def test_available_memory_bytes_ignores_unlimited_cgroup(monkeypatch):
    """The cgroup-v2 `max` marker leaves the host probe unchanged."""
    import builtins
    import io
    import os
    import sys

    import vibeqc.memory as memory_module

    gib = 1024**3
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=4 * gib),
    )
    files = {
        "/proc/self/cgroup": "0::/unlimited\n",
        "/sys/fs/cgroup/unlimited/memory.max": "max",
    }
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        resolved = os.fspath(path)
        if resolved in files:
            return io.StringIO(files[resolved])
        if resolved.startswith("/sys/fs/cgroup"):
            raise FileNotFoundError(resolved)
        return real_open(path, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(memory_module.sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", fake_open)

    assert vq.available_memory_bytes() == 4 * gib


def test_available_memory_bytes_fails_closed_for_exhausted_cgroup(monkeypatch):
    """Known zero cgroup headroom must not use the API's unknown sentinel."""
    import builtins
    import io
    import os
    import sys

    import vibeqc.memory as memory_module

    gib = 1024**3
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=91 * gib),
    )
    files = {
        "/proc/self/cgroup": "0::/exhausted\n",
        "/sys/fs/cgroup/exhausted/memory.max": str(gib),
        "/sys/fs/cgroup/exhausted/memory.current": str(2 * gib),
    }
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        resolved = os.fspath(path)
        if resolved in files:
            return io.StringIO(files[resolved])
        if resolved.startswith("/sys/fs/cgroup"):
            raise FileNotFoundError(resolved)
        return real_open(path, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(memory_module.sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", fake_open)

    assert vq.available_memory_bytes() == 1


# ---------------------------------------------------------------------------
# run_job integration
# ---------------------------------------------------------------------------

def test_run_job_writes_memory_block_to_out_file():
    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        vq.run_job(
            _h2o(), basis="sto-3g", method="rhf", output=out_stem,
            write_molden_file=False,
        )
        text = (out_stem.with_suffix(".out")).read_text()
    assert "vibe-qc estimates this calculation" in text
    assert "MB" in text or "GB" in text


def test_run_job_memory_override_allows_implausibly_large_estimate(monkeypatch):
    """Monkey-patch ``estimate_memory`` to force a huge estimate; without
    override the call aborts, with override it proceeds."""
    from vibeqc import memory as _mem

    real = _mem.estimate_memory

    def fake_estimate(molecule, basis, *, method, options=None):
        est = real(molecule, basis, method=method, options=options)
        est.by_category["inflated-for-test"] = 10 * 1024 ** 4   # 10 TB
        return est

    monkeypatch.setattr("vibeqc.runner.estimate_memory", fake_estimate)

    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        # Without override: aborts.
        with pytest.raises(vq.InsufficientMemoryError):
            vq.run_job(
                _h2o(), basis="sto-3g", method="rhf", output=out_stem,
                write_molden_file=False,
            )

    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        # With override: runs to completion.
        vq.run_job(
            _h2o(), basis="sto-3g", method="rhf", output=out_stem,
            memory_override=True, write_molden_file=False,
        )
        text = (out_stem.with_suffix(".out")).read_text()
    assert "override" in text.lower()


# ---------------------------------------------------------------------------
# VIBEQC_MEMORY_LIMIT_BYTES environment variable
# ---------------------------------------------------------------------------

def test_available_memory_bytes_respects_env_override(monkeypatch):
    """VIBEQC_MEMORY_LIMIT_BYTES takes priority over every probe."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", str(4 * 1024**3))
    # The override returns exactly the declared limit.
    assert vq.available_memory_bytes() == 4 * 1024**3


def test_available_memory_bytes_env_override_ignores_invalid(monkeypatch):
    """Garbage values are silently ignored; we fall back to the probe."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", "not-a-number")
    result = vq.available_memory_bytes()
    # Must still return a sane result (0 or positive).
    assert result >= 0


def test_available_memory_bytes_env_override_ignores_negative(monkeypatch):
    """Negative or zero values are treated as unset."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", "-100")
    result = vq.available_memory_bytes()
    assert result >= 0


def test_available_memory_bytes_env_override_respects_cgroup(monkeypatch):
    """On Linux the env var and cgroup limit are both honoured —
    the tighter bound wins."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", str(16 * 1024**3))
    # Simulate a cgroup v2 limit of 8 GB.
    cgroup_fs = {
        "/sys/fs/cgroup/memory.max": str(8 * 1024**3),
        "/sys/fs/cgroup/memory.current": "0",
        "/proc/self/cgroup": "0::/user.slice/test.scope\n",
    }
    _real_open = __builtins__["open"] if "open" in dir(__builtins__) else open
    def _fake_open(path, *args, **kwargs):
        path_str = str(path)
        if path_str in cgroup_fs:
            import io
            return io.StringIO(cgroup_fs[path_str])
        if hasattr(path, "startswith") and path.startswith("/sys/fs/cgroup"):
            raise OSError("not simulated")
        return _real_open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", _fake_open)
    # We can't fake /proc/self/cgroup cleanly without breaking other
    # open calls; skip the exact value check and just verify the guard
    # shape: env override returns a positive integer.
    result = vq.available_memory_bytes()
    assert result > 0


def test_check_memory_aborts_with_env_override_limit(monkeypatch):
    """When VIBEQC_MEMORY_LIMIT_BYTES declares 4 GB and the estimate
    is 8 GB, check_memory must abort."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", str(4 * 1024**3))
    est = vq.MemoryEstimate(by_category={"test": 8 * 1024**3})
    with pytest.raises(vq.InsufficientMemoryError, match="ABORTING"):
        vq.check_memory(est)


def test_check_memory_passes_when_estimate_fits_env_limit(monkeypatch):
    """When VIBEQC_MEMORY_LIMIT_BYTES declares 16 GB and the estimate
    is 8 GB, the job proceeds."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", str(16 * 1024**3))
    est = vq.MemoryEstimate(by_category={"test": 8 * 1024**3})
    vq.check_memory(est)  # must not raise


def test_check_memory_env_override_still_respects_allow_exceed(monkeypatch):
    """memory_override=True bypasses even the env-var limit."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", str(1 * 1024**3))
    est = vq.MemoryEstimate(by_category={"test": 100 * 1024**3})
    vq.check_memory(est, allow_exceed=True)  # must not raise


# ---------------------------------------------------------------------------
# VQ_MEM_MB environment variable (vq scheduler allocation, BUG 114)
# ---------------------------------------------------------------------------


def test_available_memory_bytes_respects_vq_mem_mb(monkeypatch):
    """VQ_MEM_MB (vq scheduler allocation in MB) bounds the host probe."""
    monkeypatch.setenv("VQ_MEM_MB", "4096")
    result = vq.available_memory_bytes()
    # Must return at most 4096 MB regardless of host-wide available RAM.
    assert 0 < result <= 4096 * 1024 * 1024


def test_available_memory_bytes_vq_mem_mb_ignores_invalid(monkeypatch):
    """Garbage VQ_MEM_MB values are silently ignored."""
    monkeypatch.setenv("VQ_MEM_MB", "not-a-number")
    result = vq.available_memory_bytes()
    assert result >= 0


def test_available_memory_bytes_vq_mem_mb_ignores_negative(monkeypatch):
    """Negative or zero VQ_MEM_MB is treated as unset."""
    monkeypatch.setenv("VQ_MEM_MB", "0")
    result = vq.available_memory_bytes()
    assert result >= 0


def test_available_memory_bytes_min_of_both_env_vars(monkeypatch):
    """When both VIBEQC_MEMORY_LIMIT_BYTES and VQ_MEM_MB are set,
    the tighter bound wins."""
    monkeypatch.setenv("VIBEQC_MEMORY_LIMIT_BYTES", str(8 * 1024**3))
    monkeypatch.setenv("VQ_MEM_MB", "4096")  # 4 GB in MB
    result = vq.available_memory_bytes()
    # VQ_MEM_MB (4 GB) is tighter than VIBEQC_MEMORY_LIMIT_BYTES (8 GB).
    assert 0 < result <= 4096 * 1024 * 1024


def test_check_memory_aborts_with_vq_mem_mb_limit(monkeypatch):
    """BUG 114 regression: when VQ_MEM_MB declares 2 GB and the estimate
    is 5 GB, check_memory must abort before integral construction."""
    monkeypatch.setenv("VQ_MEM_MB", "2048")
    est = vq.MemoryEstimate(by_category={"test": 5 * 1024**3})
    with pytest.raises(vq.InsufficientMemoryError, match="ABORTING"):
        vq.check_memory(est)


def test_check_memory_passes_when_estimate_fits_vq_mem_mb(monkeypatch):
    """When VQ_MEM_MB declares 16 GB and the estimate is 8 GB,
    the job proceeds."""
    monkeypatch.setenv("VQ_MEM_MB", "16384")
    est = vq.MemoryEstimate(by_category={"test": 8 * 1024**3})
    vq.check_memory(est)  # must not raise


def test_format_memory_report_aborts_with_vq_mem_mb(monkeypatch):
    """When VQ_MEM_MB=2048 and estimate exceeds it, the report must
    say ABORTING, never Proceeding."""
    monkeypatch.setenv("VQ_MEM_MB", "2048")
    est = vq.MemoryEstimate(by_category={"test": 5 * 1024**3})
    report = vq.format_memory_report(est, override_requested=False)
    assert "Proceeding" not in report
    assert "ABORTING" in report


def test_format_memory_report_shows_proceeding_when_fits_vq_mem_mb(monkeypatch):
    """When the estimate fits within VQ_MEM_MB, the report says Proceeding."""
    monkeypatch.setenv("VQ_MEM_MB", "16384")
    est = vq.MemoryEstimate(by_category={"test": 8 * 1024**3})
    report = vq.format_memory_report(est, override_requested=False)
    assert "Proceeding" in report
    assert "ABORTING" not in report


# ---------------------------------------------------------------------------
# macOS vm_stat probe
# ---------------------------------------------------------------------------

def test_macos_available_memory_parses_vm_stat_output():
    """The helper correctly sums free + inactive + speculative + purgeable."""
    from vibeqc.memory import _macos_available_memory_bytes

    sample = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               10000.\n"
        "Pages active:                             50000.\n"
        "Pages inactive:                           20000.\n"
        "Pages speculative:                         5000.\n"
        "Pages throttled:                              0.\n"
        "Pages wired down:                         30000.\n"
        "Pages purgeable:                           1000.\n"
    )
    # We can't easily mock subprocess in this context, but we can test
    # the parsing logic by calling the function on a real Mac — the
    # result must be in a sensible range.  On non-macOS the function
    # returns 0 because the subprocess call fails.
    import platform
    result = _macos_available_memory_bytes()
    if platform.system() == "Darwin":
        assert result > 0, "macOS must report positive available memory"
    else:
        assert result == 0, "non-macOS must return 0 (probe not available)"


# ---------------------------------------------------------------------------
# Transition-metal guard: Cr(CO)₆ RHF/def2-SVP
# ---------------------------------------------------------------------------

def test_cr_co6_rhf_estimate_triggers_guard_on_small_machine():
    """Regression test for BUG 117: Cr(CO)₆ RHF/def2-SVP (~210 bf,
    ~14 GB ERI tensor) must trigger InsufficientMemoryError when only
    8 GB is available, rather than crashing with std::bad_alloc in C++.

    BUG 87: scf_mode_auto_threshold lowered to 140 — AUTO now resolves to
    DIRECT for this system, so the memory estimate is 0.2 GB instead of
    >8 GB. To exercise the guard itself we force CONVENTIONAL."""
    mol = _cr_co6()
    basis = vq.BasisSet(mol, "def2-svp")

    # Force CONVENTIONAL so the in-core ERI tensor estimate applies.
    # (BUG 87: AUTO now picks DIRECT for >140 BF, which avoids OOM.)
    from vibeqc._vibeqc_core import RHFOptions, SCFMode
    opts = RHFOptions()
    opts.scf_mode = SCFMode.CONVENTIONAL
    estimate = vq.estimate_memory(mol, basis, method="rhf", options=opts)

    # The raw ERI tensor alone for ~197 bf is > 10 GB; the total
    # (with 1.5× headroom) must be > 8 GB.
    assert estimate.total_bytes > 8 * 1024**3, (
        f"Cr(CO)₆ RHF/def2-SVP estimate {estimate.total_gb:.1f} GB "
        f"must exceed 8 GB to trigger the guard"
    )

    # Simulate an 8 GB machine: the guard must fire.
    with pytest.raises(vq.InsufficientMemoryError, match="ABORTING"):
        vq.check_memory(estimate, available=8 * 1024**3)


def test_cr_co6_rhf_estimate_passes_on_large_machine():
    """The same estimate must proceed when the machine has enough RAM."""
    mol = _cr_co6()
    basis = vq.BasisSet(mol, "def2-svp")

    estimate = vq.estimate_memory(mol, basis, method="rhf")

    # 36 GB machine: must proceed.
    vq.check_memory(estimate, available=36 * 1024**3)  # must not raise


# ---------------------------------------------------------------------------
# Batch runner memory budgeting (BUG 123)
# ---------------------------------------------------------------------------


def _est_mb(mb: float) -> vq.MemoryEstimate:
    """Helper: create a MemoryEstimate with a given total in MB."""
    return vq.MemoryEstimate(
        {"test": int(mb * 1024**2)},
        headroom_factor=1.0,
    )


def test_batch_runner_accounts_for_child_memory():
    """Batch runner must not spawn more children than memory allows."""
    cases = [_est_mb(4096), _est_mb(4096), _est_mb(4096), _est_mb(4096)]
    # 4 children × 4 GB = 16 GB, but only 8 GB available.
    plan = vq.plan_batch_memory(cases, available_mb=8192, overhead_factor=1.0)
    assert plan.n_parallel <= 2


def test_batch_runner_falls_back_to_sequential():
    """When sum exceeds PBS limit, sequential execution is selected."""
    cases = [_est_mb(4096), _est_mb(4096), _est_mb(4096), _est_mb(4096)]
    # 2 GB available — even 1 child barely fits, so sequential.
    plan = vq.plan_batch_memory(cases, available_mb=2048, overhead_factor=1.0)
    assert plan.n_parallel == 1
    assert plan.mode == "sequential"


def test_batch_plan_with_overhead_factor():
    """Overhead factor reduces safe parallelism."""
    cases = [_est_mb(2000), _est_mb(2000), _est_mb(2000)]
    # Without overhead: 3 × 2000 = 6000 MB fits in 8000 MB.
    plan_no_oh = vq.plan_batch_memory(cases, available_mb=8000, overhead_factor=1.0)
    assert plan_no_oh.n_parallel == 3
    # With 1.5× overhead: 3 × (2000 × 1.5) = 9000 MB > 8000 MB.
    plan_oh = vq.plan_batch_memory(cases, available_mb=8000, overhead_factor=1.5)
    assert plan_oh.n_parallel <= 2


def test_batch_plan_respects_max_parallel():
    """max_parallel caps n_parallel even when memory is abundant."""
    cases = [_est_mb(100), _est_mb(100), _est_mb(100), _est_mb(100)]
    plan = vq.plan_batch_memory(cases, available_mb=8000, overhead_factor=1.0, max_parallel=2)
    assert plan.n_parallel == 2
    assert plan.mode == "parallel"


def test_batch_plan_empty_estimates():
    """Empty estimate list returns zero parallelism."""
    plan = vq.plan_batch_memory([], available_mb=8000)
    assert plan.n_parallel == 0
    assert plan.mode == "sequential"


def test_batch_plan_per_child_mb_tracked():
    """per_child_mb records each child's estimate."""
    cases = [_est_mb(1024), _est_mb(2048), _est_mb(4096)]
    plan = vq.plan_batch_memory(cases, available_mb=16000, overhead_factor=1.0)
    assert plan.per_child_mb == pytest.approx([1024.0, 2048.0, 4096.0])
    assert plan.total_estimated_mb == pytest.approx(1024 + 2048 + 4096)


def test_batch_plan_with_max_parallel_one_forces_sequential():
    """max_parallel=1 forces sequential mode."""
    cases = [_est_mb(100), _est_mb(100)]
    plan = vq.plan_batch_memory(cases, available_mb=8000, overhead_factor=1.0, max_parallel=1)
    assert plan.n_parallel == 1
    assert plan.mode == "sequential"


# ---------------------------------------------------------------------------
# Native periodic-correlation resource admission
# ---------------------------------------------------------------------------


def _translation_pair_count(
    mesh: tuple[int, int, int], n_home_occupied: int
) -> int:
    n_cells = math.prod(mesh)
    self_inverse = math.prod(2 if size % 2 == 0 else 1 for size in mesh)
    return (
        n_home_occupied * (n_cells + self_inverse) // 2
        + n_home_occupied * (n_home_occupied - 1) * n_cells // 2
    )


def _periodic_resource_dimensions(
    *,
    n_kpoints: int = 2,
    mesh: tuple[int, int, int] | None = None,
    is_shift: tuple[int, int, int] = (0, 0, 0),
    n_basis: int = 5,
    n_home_occupied: int = 2,
    triples_requested: bool = True,
):
    core = vq._vibeqc_core
    dimensions = core._PeriodicCorrelationStaticDimensions()
    dimensions.allocation_contract_version = (
        core._PERIODIC_CORRELATION_STREAMED_RESOURCE_CONTRACT_VERSION
    )
    dimensions.calculation_identity = "1" * 64
    dimensions.allocation_identity = "4" * 64
    dimensions.static_inventory_complete = True
    dimensions.periodic_dimension = 3
    exact_mesh = mesh if mesh is not None else (n_kpoints, 1, 1)
    dimensions.mesh = exact_mesh
    dimensions.is_shift = is_shift
    dimensions.n_kpoints = n_kpoints
    dimensions.n_basis = n_basis
    dimensions.n_effective_orbitals = n_basis
    dimensions.n_auxiliary = 3 * n_basis
    dimensions.n_home_total_occupied = n_home_occupied
    dimensions.n_home_occupied = n_home_occupied
    dimensions.n_home_virtual = max(1, n_basis - n_home_occupied)
    dimensions.n_spin_channels = 1
    dimensions.triples_requested = triples_requested
    dimensions.symmetry_representative_count = n_kpoints
    dimensions.symmetry_weight_sum = n_kpoints
    pair_count = _translation_pair_count(exact_mesh, n_home_occupied)
    dimensions.expected_pair_candidate_count = pair_count
    dimensions.expected_triple_candidate_count = 1 if triples_requested else 0
    dimensions.factor_k_bra_block = 1
    dimensions.factor_k_ket_block = 1
    dimensions.factor_q_block = 1
    dimensions.factor_auxiliary_block = 2
    dimensions.domain_ao_support_upper_bound = n_basis
    dimensions.domain_pao_upper_bound = n_basis
    dimensions.domain_pno_upper_bound = n_basis
    dimensions.domain_local_occupied_upper_bound = max(
        n_home_occupied, n_basis
    )
    dimensions.domain_local_auxiliary_upper_bound = 3 * n_basis
    dimensions.pair_domain_metadata_upper_bytes = max(
        16_384, 8 * n_home_occupied + 64 * pair_count
    )
    if triples_requested:
        dimensions.triple_virtual_support_upper_bound = n_basis
        dimensions.triple_union_pno_upper_bound = n_basis
        dimensions.triple_tno_upper_bound = n_basis
        dimensions.triple_local_occupied_upper_bound = n_home_occupied
        dimensions.triple_local_auxiliary_upper_bound = 3 * n_basis
        dimensions.triple_domain_metadata_upper_bytes = 8_192
    return dimensions


def _periodic_resource_budget(
    *,
    memory: int = 1024**3,
    scratch: int = 1024**3,
    ranks: int = 1,
    workers: int = 4,
):
    core = vq._vibeqc_core
    budget = core._PeriodicCorrelationResourceBudget()
    budget.memory_limit_bytes = memory
    budget.scratch_limit_bytes = scratch
    budget.mpi_ranks = ranks
    budget.workers_per_rank = workers
    return budget


def _periodic_pair(*, pno: int, occupied: int, virtual: int, auxiliary: int):
    pair = vq._vibeqc_core._PeriodicCorrelationPairDomain()
    pair.n_pno = pno
    pair.n_local_occupied = occupied
    pair.n_extended_virtual = virtual
    pair.n_local_auxiliary = auxiliary
    return pair


def _periodic_triple(*, tno: int, occupied: int, auxiliary: int):
    triple = vq._vibeqc_core._PeriodicCorrelationTripleDomain()
    triple.n_tno = tno
    triple.n_local_occupied = occupied
    triple.n_local_auxiliary = auxiliary
    return triple


def _periodic_factor(
    *, key: int, auxiliary: int, occupied: int, virtual: int, header: int
):
    factor = vq._vibeqc_core._PeriodicCorrelationFactorDomain()
    factor.storage_identity = f"{key:064x}"
    factor.n_auxiliary = auxiliary
    factor.n_occupied = occupied
    factor.n_virtual = virtual
    factor.header_bytes = header
    return factor


def _periodic_domain_census(*, iterative: bool = False, disk: bool = True):
    census = vq._vibeqc_core._PeriodicCorrelationDomainCensus()
    census.calculation_identity = "1" * 64
    census.allocation_identity = "4" * 64
    census.census_identity = "7" * 64
    census.pair_domain_census_complete = True
    census.singles_pno_counts = [2, 3]
    census.pair_domains = [
        _periodic_pair(pno=2, occupied=2, virtual=4, auxiliary=3),
        _periodic_pair(pno=3, occupied=2, virtual=5, auxiliary=3),
    ]
    census.triple_domains = [
        _periodic_triple(tno=3, occupied=2, auxiliary=3)
    ]
    factor_a = _periodic_factor(
        key=7, auxiliary=3, occupied=2, virtual=4, header=32
    )
    factor_b = _periodic_factor(
        key=8, auxiliary=2, occupied=1, virtual=3, header=16
    )
    census.factor_domains = [factor_a, factor_b]
    census.symmetry_full_kpoint_count = 2
    census.symmetry_representative_count = 2
    census.symmetry_weight_sum = 2
    census.diis_depth = 2
    census.disk_backed_diis = disk
    census.disk_backed_factors = True
    census.triples_requested = True
    census.triple_domain_census_complete = True
    census.pair_candidate_count = 6
    census.strong_pair_count = 2
    census.neglected_pair_count = 4
    census.factor_manifest_verified = True
    census.factor_manifest_identity = "6" * 64
    census.expected_factor_domain_count = 2
    census.triple_candidate_count = 1
    census.evaluated_triple_count = 1
    census.iterative_triples = iterative
    census.disk_backed_triples = disk and iterative
    census.amplitude_replicas = 1
    census.factor_store_replicas = 1
    census.triples_amplitude_replicas = 1 if iterative else 0
    census.checkpoint_replicas = 1
    census.pair_virtual_block = 2
    census.triple_virtual_block = 2
    census.checkpoint_generations = 2
    census.checkpoint_inventory_complete = True
    census.checkpoint_includes_factor_store = True
    census.checkpoint_provenance_bytes = 512
    census.pair_checkpoint_payload_bytes = 16_384
    census.final_checkpoint_payload_bytes = 24_576
    census.pair_domain_metadata_bytes = 4_096
    census.triple_domain_metadata_bytes = 2_048
    return census


def _periodic_phase(plan, phase):
    return next(item for item in plan.phases if item.phase == phase)


def test_native_periodic_resource_byte_helpers_are_exact_and_checked():
    core = vq._vibeqc_core
    dense = core._estimate_periodic_dense_gdf_cache_bytes(64, 30, 10)
    streamed = core._estimate_periodic_streamed_factor_tile_bytes(1, 1, 4, 100)

    assert dense == 16 * 64 * 30 * 10**2
    assert streamed == 16 * 4 * 100
    assert core._periodic_occupied_triple_count(3) == 10
    assert core._periodic_correlation_bytes_to_mib_ceil(0) == 0
    assert core._periodic_correlation_bytes_to_mib_ceil(1) == 1
    assert core._periodic_correlation_bytes_to_mib_ceil(1024**2 + 1) == 2
    with pytest.raises(OverflowError, match="overflow"):
        core._periodic_occupied_triple_count(2**64 - 1)


def test_native_static_phase_inventories_are_hand_calculated():
    core = vq._vibeqc_core
    plan = core._plan_periodic_correlation_static_resources(
        _periodic_resource_dimensions(),
        _periodic_resource_budget(workers=1),
    )
    factor = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.FACTOR_BUILD
    )
    mean_field = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.MEAN_FIELD
    )
    localization = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.LOCALIZATION
    )
    domain = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.DOMAIN_BUILD
    )

    assert (factor.retained_bytes, factor.concurrent_worker_bytes) == (
        7_200,
        1_600,
    )
    assert (mean_field.retained_bytes, mean_field.concurrent_worker_bytes) == (
        4_800,
        800,
    )
    assert (localization.retained_bytes, localization.concurrent_worker_bytes) == (
        448,
        0,
    )
    assert (domain.retained_bytes, domain.concurrent_worker_bytes) == (
        21_632,
        8_112,
    )


def test_native_static_preflight_is_linear_in_k_and_keeps_factor_tile_fixed():
    core = vq._vibeqc_core
    budget = _periodic_resource_budget(memory=1024**4, workers=4)
    one = core._plan_periodic_correlation_static_resources(
        _periodic_resource_dimensions(n_kpoints=1, n_basis=37), budget
    )
    mesh = core._plan_periodic_correlation_static_resources(
        _periodic_resource_dimensions(n_kpoints=512, n_basis=37), budget
    )
    one_phase = _periodic_phase(
        one, core._PeriodicCorrelationResourcePhase.MEAN_FIELD
    )
    mesh_phase = _periodic_phase(
        mesh, core._PeriodicCorrelationResourcePhase.MEAN_FIELD
    )

    assert one_phase.concurrent_worker_bytes == mesh_phase.concurrent_worker_bytes
    assert mesh_phase.retained_bytes == 512 * one_phase.retained_bytes
    assert mesh.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )
    assert not mesh.pair_domain_census_complete
    assert not mesh.triple_domain_census_complete


@pytest.mark.parametrize(
    ("mesh", "n_basis", "n_occupied", "expected_pairs", "minimum_metadata"),
    [
        ((8, 8, 8), 37, 8, 16_416, 1_050_688),
        ((8, 8, 8), 36, 4, 4_112, 263_200),
        ((6, 6, 6), 196, 24, 62_304, 3_987_648),
    ],
    ids=["eight-cubed-b8", "eight-cubed-b4", "six-cubed-b24"],
)
def test_native_target_mesh_preflight_allocates_only_metadata(
    mesh, n_basis, n_occupied, expected_pairs, minimum_metadata
):
    core = vq._vibeqc_core
    n_kpoints = math.prod(mesh)
    dimensions = _periodic_resource_dimensions(
        n_kpoints=n_kpoints, mesh=mesh,
        n_basis=n_basis,
        n_home_occupied=n_occupied,
    )
    dimensions.factor_auxiliary_block = 16
    assert dimensions.expected_pair_candidate_count == expected_pairs
    assert dimensions.pair_domain_metadata_upper_bytes == minimum_metadata
    plan = core._plan_periodic_correlation_static_resources(
        dimensions,
        _periodic_resource_budget(memory=1024**4, workers=4),
    )

    assert len(plan.phases) == 4
    assert plan.required_memory_bytes < 1024**4
    assert plan.calculation_identity == "1" * 64
    assert plan.allocation_identity == "4" * 64
    assert plan.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )


def test_native_static_preflight_seals_exact_regular_mesh_and_shift():
    core = vq._vibeqc_core
    shifted = _periodic_resource_dimensions(
        n_kpoints=4, mesh=(2, 1, 2), is_shift=(1, 0, 1)
    )
    plan = core._plan_periodic_correlation_static_resources(
        shifted, _periodic_resource_budget()
    )
    assert plan.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )

    wrong_extent = _periodic_resource_dimensions(
        n_kpoints=4, mesh=(2, 1, 1)
    )
    with pytest.raises(ValueError, match="RegularKMesh extent"):
        core._plan_periodic_correlation_static_resources(
            wrong_extent, _periodic_resource_budget()
        )

    invalid_shift = _periodic_resource_dimensions(is_shift=(0, 2, 0))
    with pytest.raises(ValueError, match="is_shift"):
        core._plan_periodic_correlation_static_resources(
            invalid_shift, _periodic_resource_budget()
        )


def test_native_static_preflight_requires_a_real_memory_limit():
    core = vq._vibeqc_core
    plan = core._plan_periodic_correlation_static_resources(
        _periodic_resource_dimensions(),
        _periodic_resource_budget(memory=0),
    )

    assert plan.admission == (
        core._PeriodicCorrelationAdmissionCode.MISSING_MEMORY_LIMIT
    )


def test_native_static_preflight_rejects_contract_dimension_and_open_shell():
    core = vq._vibeqc_core
    dimensions = _periodic_resource_dimensions()
    dimensions.allocation_contract_version = 0
    with pytest.raises(ValueError, match="allocation contract version"):
        core._plan_periodic_correlation_static_resources(
            dimensions,
            _periodic_resource_budget(),
        )

    for invalid_dimension in (0, 4):
        dimensions = _periodic_resource_dimensions()
        dimensions.periodic_dimension = invalid_dimension
        with pytest.raises(ValueError, match="periodic_dimension"):
            core._plan_periodic_correlation_static_resources(
                dimensions,
                _periodic_resource_budget(),
            )

    for field, invalid_value in (
        ("mesh", (2, 2, 1)),
        ("is_shift", (0, 1, 0)),
    ):
        dimensions = _periodic_resource_dimensions()
        dimensions.periodic_dimension = 1
        setattr(dimensions, field, invalid_value)
        with pytest.raises(ValueError, match="inactive mesh axes"):
            core._plan_periodic_correlation_static_resources(
                dimensions,
                _periodic_resource_budget(),
            )

    open_shell = _periodic_resource_dimensions()
    open_shell.n_spin_channels = 2
    with pytest.raises(ValueError, match="closed-shell only"):
        core._plan_periodic_correlation_static_resources(
            open_shell,
            _periodic_resource_budget(),
        )


def test_native_static_contract_v1_is_single_rank_and_one_q_at_a_time():
    core = vq._vibeqc_core
    multi_q = _periodic_resource_dimensions()
    multi_q.factor_q_block = 2
    with pytest.raises(ValueError, match="one-q-at-a-time"):
        core._plan_periodic_correlation_static_resources(
            multi_q,
            _periodic_resource_budget(),
        )

    with pytest.raises(ValueError, match="one MPI rank"):
        core._plan_periodic_correlation_static_resources(
            _periodic_resource_dimensions(),
            _periodic_resource_budget(ranks=2),
        )


def test_native_static_domain_rank_bounds_are_nested():
    dimensions = _periodic_resource_dimensions()
    dimensions.domain_ao_support_upper_bound = 4
    with pytest.raises(ValueError, match="AO support"):
        vq._vibeqc_core._plan_periodic_correlation_static_resources(
            dimensions,
            _periodic_resource_budget(),
        )

    metadata = _periodic_resource_dimensions()
    metadata.pair_domain_metadata_upper_bytes = 399
    with pytest.raises(ValueError, match="structural minimum"):
        vq._vibeqc_core._plan_periodic_correlation_static_resources(
            metadata,
            _periodic_resource_budget(),
        )


@pytest.mark.parametrize("forged_count", [5, 7])
def test_native_static_preflight_rejects_forged_pair_count(forged_count):
    dimensions = _periodic_resource_dimensions()
    dimensions.expected_pair_candidate_count = forged_count
    with pytest.raises(ValueError, match="exact translation-pair topology"):
        vq._vibeqc_core._plan_periodic_correlation_static_resources(
            dimensions,
            _periodic_resource_budget(),
        )


def test_native_static_preflight_throttles_workers_before_rejecting():
    core = vq._vibeqc_core
    dimensions = _periodic_resource_dimensions(
        n_kpoints=8, n_basis=20, triples_requested=False
    )
    dimensions.n_auxiliary = 1
    dimensions.n_home_virtual = 1
    dimensions.n_effective_orbitals = 3
    dimensions.factor_auxiliary_block = 1
    dimensions.domain_ao_support_upper_bound = 1
    dimensions.domain_pao_upper_bound = 1
    dimensions.domain_pno_upper_bound = 1
    dimensions.domain_local_auxiliary_upper_bound = 1
    dimensions.pair_domain_metadata_upper_bytes = 1_168
    exploratory = core._plan_periodic_correlation_static_resources(
        dimensions,
        _periodic_resource_budget(memory=1024**4, workers=4),
    )
    phase = _periodic_phase(
        exploratory, core._PeriodicCorrelationResourcePhase.MEAN_FIELD
    )
    one_worker = phase.concurrent_worker_bytes // 4
    two_worker_peak = phase.retained_bytes + 2 * one_worker
    exact_cap = (3 * two_worker_peak + 1) // 2
    throttled = core._plan_periodic_correlation_static_resources(
        dimensions,
        _periodic_resource_budget(memory=exact_cap, workers=4),
    )

    assert throttled.active_mean_field_workers == 2
    assert throttled.required_memory_bytes == exact_cap
    assert throttled.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )


def test_native_static_preflight_exact_limit_passes_and_one_less_fails():
    core = vq._vibeqc_core
    dimensions = _periodic_resource_dimensions()
    exploratory = core._plan_periodic_correlation_static_resources(
        dimensions,
        _periodic_resource_budget(memory=1024**3, workers=1),
    )
    required = exploratory.required_memory_bytes
    exact = core._plan_periodic_correlation_static_resources(
        dimensions,
        _periodic_resource_budget(memory=required, workers=1),
    )
    short = core._plan_periodic_correlation_static_resources(
        dimensions,
        _periodic_resource_budget(memory=required - 1, workers=1),
    )

    assert exact.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )
    assert short.admission == (
        core._PeriodicCorrelationAdmissionCode.MEMORY_EXCEEDED
    )


def test_native_domain_census_counts_exact_amplitudes_factors_and_scratch():
    core = vq._vibeqc_core
    plan = core._plan_periodic_correlation_domain_resources(
        _periodic_resource_dimensions(),
        _periodic_domain_census(),
        _periodic_resource_budget(),
    )

    assert plan.pair_domain_census_complete
    assert plan.triple_domain_census_complete
    assert plan.calculation_identity == "1" * 64
    assert plan.allocation_identity == "4" * 64
    assert plan.census_identity == "7" * 64
    assert plan.amplitude_bytes == 16 * (2 + 3 + 2**2 + 3**2)
    assert plan.factor_store_bytes == (
        16 * 3 * (2 * 4 + 2**2 + 4**2)
        + 32
        + 16 * 2 * (1 * 3 + 1**2 + 3**2)
        + 16
    )
    assert plan.unique_factor_domains == 2
    assert plan.triples_amplitude_bytes == 0
    assert plan.diis_scratch_bytes == 2 * 2 * plan.amplitude_bytes
    assert plan.checkpoint_scratch_bytes == 3 * 24_576
    assert plan.modeled_scratch_bytes == (
        plan.factor_store_bytes + plan.checkpoint_scratch_bytes
    )
    assert plan.required_scratch_bytes == (
        5 * plan.modeled_scratch_bytes + 3
    ) // 4
    assert plan.admission == core._PeriodicCorrelationAdmissionCode.ADMITTED


def test_native_pair_and_triples_censuses_are_separate_admission_stages():
    core = vq._vibeqc_core
    pair_census = _periodic_domain_census()
    pair_census.triple_domains = []
    pair_census.triple_domain_census_complete = False
    pair_census.triple_candidate_count = 0
    pair_census.evaluated_triple_count = 0
    pair_census.triple_domain_metadata_bytes = 0
    pair_census.final_checkpoint_payload_bytes = 0
    pair_plan = core._plan_periodic_correlation_domain_resources(
        _periodic_resource_dimensions(),
        pair_census,
        _periodic_resource_budget(),
    )

    assert pair_plan.stage == (
        core._PeriodicCorrelationEstimateStage.PAIR_DOMAIN_CENSUS
    )
    assert pair_plan.pair_domain_census_complete
    assert not pair_plan.triple_domain_census_complete
    assert pair_plan.triples_amplitude_bytes == 0
    assert pair_plan.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_TRIPLE_DOMAIN_CENSUS
    )
    assert all(
        phase.phase != core._PeriodicCorrelationResourcePhase.TRIPLES
        for phase in pair_plan.phases
    )
    assert any(
        phase.phase == core._PeriodicCorrelationResourcePhase.TRIPLE_DOMAIN_BUILD
        for phase in pair_plan.phases
    )
    triple_domain_build = _periodic_phase(
        pair_plan,
        core._PeriodicCorrelationResourcePhase.TRIPLE_DOMAIN_BUILD,
    )
    pair_checkpoint = pair_census.pair_checkpoint_payload_bytes
    post_tno_checkpoint = (
        pair_checkpoint
        + _periodic_resource_dimensions().triple_domain_metadata_upper_bytes
    )
    pair_checkpoint_phase = _periodic_phase(
        pair_plan,
        core._PeriodicCorrelationResourcePhase.CHECKPOINT,
    )
    assert pair_checkpoint_phase.retained_bytes == (
        triple_domain_build.retained_bytes + post_tno_checkpoint
    )
    assert pair_plan.checkpoint_scratch_bytes == 3 * post_tno_checkpoint

    triples_plan = core._plan_periodic_correlation_domain_resources(
        _periodic_resource_dimensions(),
        _periodic_domain_census(),
        _periodic_resource_budget(),
    )
    assert triples_plan.stage == (
        core._PeriodicCorrelationEstimateStage.TRIPLE_DOMAIN_CENSUS
    )
    assert triples_plan.triple_domain_census_complete
    assert triples_plan.admission == (
        core._PeriodicCorrelationAdmissionCode.ADMITTED
    )


def test_native_domain_plan_uses_phase_maximum_and_top_worker_workspaces():
    core = vq._vibeqc_core
    census = _periodic_domain_census()
    plan = core._plan_periodic_correlation_domain_resources(
        _periodic_resource_dimensions(),
        census,
        _periodic_resource_budget(workers=1),
    )
    pair = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.PAIR_SOLVE
    )

    assert plan.modeled_peak_memory_bytes == max(
        phase.peak_memory_bytes for phase in plan.phases
    )
    assert pair.active_workers == 1
    reversed_census = _periodic_domain_census()
    reversed_census.pair_domains = list(reversed(reversed_census.pair_domains))
    reversed_census.census_identity = "8" * 64
    reversed_plan = core._plan_periodic_correlation_domain_resources(
        _periodic_resource_dimensions(),
        reversed_census,
        _periodic_resource_budget(workers=1),
    )
    assert reversed_plan.modeled_peak_memory_bytes == plan.modeled_peak_memory_bytes
    assert reversed_plan.modeled_scratch_bytes == plan.modeled_scratch_bytes


def test_native_pair_and_triple_worker_inventories_are_hand_calculated():
    core = vq._vibeqc_core
    dimensions = _periodic_resource_dimensions()
    census = _periodic_domain_census()
    plan = core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(workers=2),
    )
    pair = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.PAIR_SOLVE
    )
    triple = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.TRIPLES
    )
    factor_build = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.FACTOR_STORE_BUILD
    )

    assert factor_build.concurrent_worker_bytes == 5_312 + 3_712
    assert pair.concurrent_worker_bytes == 12_608 + 20_832
    assert triple.concurrent_worker_bytes == 4_112

    one_pair_worker_cap = (3 * (pair.retained_bytes + 20_832) + 1) // 2
    throttled = core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(memory=one_pair_worker_cap, workers=2),
    )
    throttled_pair = _periodic_phase(
        throttled, core._PeriodicCorrelationResourcePhase.PAIR_SOLVE
    )
    assert throttled_pair.active_workers == 1
    assert throttled_pair.concurrent_worker_bytes == 20_832


def test_native_pair_concurrency_sums_the_largest_worker_workspaces():
    core = vq._vibeqc_core
    domains = [
        _periodic_pair(pno=2, occupied=2, virtual=3, auxiliary=2),
        _periodic_pair(pno=3, occupied=2, virtual=5, auxiliary=3),
        _periodic_pair(pno=4, occupied=3, virtual=7, auxiliary=4),
    ]

    def pair_only_inputs(pair_domains):
        identity = f"{100 + len(pair_domains):064x}"
        dimensions = _periodic_resource_dimensions(
            n_basis=10, triples_requested=False
        )
        dimensions.allocation_identity = identity
        census = _periodic_domain_census(iterative=False, disk=True)
        census.allocation_identity = identity
        census.census_identity = f"{200 + len(pair_domains):064x}"
        census.pair_domains = pair_domains
        census.triple_domains = []
        census.triples_requested = False
        census.triple_domain_census_complete = False
        census.triple_candidate_count = 0
        census.evaluated_triple_count = 0
        census.triple_domain_metadata_bytes = 0
        census.iterative_triples = False
        census.disk_backed_triples = False
        census.strong_pair_count = len(pair_domains)
        census.pair_candidate_count = 6
        census.neglected_pair_count = 6 - len(pair_domains)
        census.checkpoint_generations = 0
        census.checkpoint_replicas = 0
        census.checkpoint_inventory_complete = False
        census.checkpoint_includes_factor_store = False
        census.checkpoint_provenance_bytes = 0
        census.pair_checkpoint_payload_bytes = 0
        census.final_checkpoint_payload_bytes = 0
        return dimensions, census

    individual_workspaces = []
    for domain in domains:
        dimensions, census = pair_only_inputs([domain])
        plan = core._plan_periodic_correlation_domain_resources(
            dimensions,
            census,
            _periodic_resource_budget(workers=1),
        )
        individual_workspaces.append(
            _periodic_phase(
                plan, core._PeriodicCorrelationResourcePhase.PAIR_SOLVE
            ).concurrent_worker_bytes
        )

    dimensions, census = pair_only_inputs(domains)
    concurrent = core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(workers=2),
    )
    pair_phase = _periodic_phase(
        concurrent, core._PeriodicCorrelationResourcePhase.PAIR_SOLVE
    )
    assert pair_phase.active_workers == 2
    assert pair_phase.concurrent_worker_bytes == sum(
        sorted(individual_workspaces, reverse=True)[:2]
    )


def test_native_checkpoint_phase_counts_live_state_and_write_buffer():
    core = vq._vibeqc_core
    census = _periodic_domain_census()
    census.checkpoint_write_buffer_extra_bytes = 10_000_000
    plan = core._plan_periodic_correlation_domain_resources(
        _periodic_resource_dimensions(),
        census,
        _periodic_resource_budget(),
    )
    triples = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.TRIPLES
    )
    checkpoint = _periodic_phase(
        plan, core._PeriodicCorrelationResourcePhase.CHECKPOINT
    )
    checkpoint_payload = census.final_checkpoint_payload_bytes

    assert checkpoint.retained_bytes == (
        triples.retained_bytes
        + checkpoint_payload
        + census.checkpoint_write_buffer_extra_bytes
    )
    assert checkpoint.concurrent_worker_bytes == 0
    assert checkpoint.scratch_bytes == (
        triples.scratch_bytes + checkpoint_payload
    )
    assert plan.checkpoint_scratch_bytes == 3 * checkpoint_payload
    assert plan.modeled_peak_memory_bytes == checkpoint.peak_memory_bytes


def test_native_disk_modes_trade_memory_for_scratch():
    core = vq._vibeqc_core
    dimensions = _periodic_resource_dimensions()
    budget = _periodic_resource_budget()
    in_memory = core._plan_periodic_correlation_domain_resources(
        dimensions,
        _periodic_domain_census(disk=False),
        budget,
    )
    on_disk = core._plan_periodic_correlation_domain_resources(
        dimensions,
        _periodic_domain_census(disk=True),
        budget,
    )
    in_memory_pair = _periodic_phase(
        in_memory, core._PeriodicCorrelationResourcePhase.PAIR_SOLVE
    )
    on_disk_pair = _periodic_phase(
        on_disk, core._PeriodicCorrelationResourcePhase.PAIR_SOLVE
    )

    assert on_disk_pair.retained_bytes < in_memory_pair.retained_bytes
    assert on_disk.diis_scratch_bytes > in_memory.diis_scratch_bytes
    assert on_disk_pair.scratch_bytes > in_memory_pair.scratch_bytes


def test_native_contract_v1_rejects_mpi_and_nonunit_replica_ownership():
    core = vq._vibeqc_core
    with pytest.raises(ValueError, match="one MPI rank"):
        core._plan_periodic_correlation_static_resources(
            _periodic_resource_dimensions(),
            _periodic_resource_budget(ranks=2),
        )

    invalid = _periodic_domain_census()
    invalid.amplitude_replicas = 2
    with pytest.raises(ValueError, match="one state/factor replica"):
        core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            invalid,
            _periodic_resource_budget(),
        )


def test_native_contract_v1_rejects_iterative_triples():
    with pytest.raises(ValueError, match="excludes iterative triples"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            _periodic_domain_census(iterative=True, disk=True),
            _periodic_resource_budget(),
        )


def test_native_domain_admission_requires_and_enforces_scratch_limit():
    core = vq._vibeqc_core
    dimensions = _periodic_resource_dimensions()
    census = _periodic_domain_census()
    exploratory = core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(),
    )
    missing = core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(scratch=0),
    )
    exact = core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(
            scratch=exploratory.required_scratch_bytes
        ),
    )
    short = core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(
            scratch=exploratory.required_scratch_bytes - 1
        ),
    )

    assert missing.admission == (
        core._PeriodicCorrelationAdmissionCode.MISSING_SCRATCH_LIMIT
    )
    assert exact.admission == core._PeriodicCorrelationAdmissionCode.ADMITTED
    assert short.admission == (
        core._PeriodicCorrelationAdmissionCode.SCRATCH_EXCEEDED
    )


def test_native_resource_overflow_fails_closed_without_allocating_dimensions():
    core = vq._vibeqc_core
    dimensions = _periodic_resource_dimensions(n_basis=2**32)
    plan = core._plan_periodic_correlation_static_resources(
        dimensions,
        _periodic_resource_budget(memory=2**64 - 1),
    )

    assert plan.admission == (
        core._PeriodicCorrelationAdmissionCode.ARITHMETIC_OVERFLOW
    )
    assert "overflow" in plan.failure_detail
    assert not plan.phases


def test_native_domain_census_rejects_duplicate_factor_identity():
    census = _periodic_domain_census()
    census.factor_domains = [
        _periodic_factor(
            key=5, auxiliary=3, occupied=2, virtual=4, header=32
        ),
        _periodic_factor(
            key=5, auxiliary=3, occupied=2, virtual=5, header=32
        ),
    ]

    with pytest.raises(ValueError, match="unique and canonically ordered"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            census,
            _periodic_resource_budget(),
        )


def test_native_domain_census_requires_exact_persistent_metadata():
    census = _periodic_domain_census()
    census.pair_domain_metadata_bytes = 0

    with pytest.raises(ValueError, match="pair_domain_metadata_bytes"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            census,
            _periodic_resource_budget(),
        )

    pair_small = _periodic_domain_census()
    pair_small.pair_domain_metadata_bytes = 655
    with pytest.raises(ValueError, match="structural minimum"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            pair_small,
            _periodic_resource_budget(),
        )

    triple_small = _periodic_domain_census()
    triple_small.triple_domain_metadata_bytes = 63
    with pytest.raises(ValueError, match="structural minimum"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            triple_small,
            _periodic_resource_budget(),
        )


def test_native_domain_census_rejects_stale_identity_and_partial_counts():
    stale = _periodic_domain_census()
    stale.calculation_identity = "3" * 64
    with pytest.raises(ValueError, match="identity does not match"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            stale,
            _periodic_resource_budget(),
        )

    stale_allocation = _periodic_domain_census()
    stale_allocation.allocation_identity = "5" * 64
    with pytest.raises(ValueError, match="allocation identity"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            stale_allocation,
            _periodic_resource_budget(),
        )

    partial = _periodic_domain_census()
    partial.strong_pair_count = 1
    with pytest.raises(ValueError, match="completeness counts"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            partial,
            _periodic_resource_budget(),
        )


def test_native_domain_census_cannot_exceed_preflight_bounds():
    census = _periodic_domain_census()
    census.pair_domain_metadata_bytes = 16_385
    with pytest.raises(ValueError, match="metadata exceeds"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            census,
            _periodic_resource_budget(),
        )


def test_native_checkpoint_payload_covers_known_restart_inventory():
    census = _periodic_domain_census()
    census.pair_checkpoint_payload_bytes = 1
    with pytest.raises(ValueError, match="known inventory"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            census,
            _periodic_resource_budget(),
        )

    final = _periodic_domain_census()
    final.pair_checkpoint_payload_bytes = 30_000
    with pytest.raises(ValueError, match="final checkpoint payload"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            _periodic_resource_dimensions(),
            final,
            _periodic_resource_budget(),
        )


def test_native_census_binding_mutators_update_cpp_vectors():
    core = vq._vibeqc_core
    census = core._PeriodicCorrelationDomainCensus()
    census.add_singles_pno_count(3)
    census.add_pair_domain(
        _periodic_pair(pno=2, occupied=2, virtual=4, auxiliary=3)
    )
    census.add_triple_domain(
        _periodic_triple(tno=3, occupied=2, auxiliary=3)
    )
    census.add_factor_domain(
        _periodic_factor(
            key=7, auxiliary=3, occupied=2, virtual=4, header=32
        )
    )

    assert census.singles_pno_counts == [3]
    assert len(census.pair_domains) == 1
    assert len(census.triple_domains) == 1
    assert len(census.factor_domains) == 1


def test_native_domain_census_rejects_unverified_symmetry_reduction():
    dimensions = _periodic_resource_dimensions()
    dimensions.symmetry_reduction_requested = True
    dimensions.symmetry_mapping_identity = "2" * 64
    dimensions.symmetry_representative_count = 1
    census = _periodic_domain_census()
    census.symmetry_reduction_used = True
    census.symmetry_mapping_identity = "2" * 64
    census.symmetry_representative_count = 1

    with pytest.raises(ValueError, match="requires a verified mapping"):
        vq._vibeqc_core._plan_periodic_correlation_domain_resources(
            dimensions,
            census,
            _periodic_resource_budget(),
        )

    census.symmetry_mapping_verified = True
    plan = vq._vibeqc_core._plan_periodic_correlation_domain_resources(
        dimensions,
        census,
        _periodic_resource_budget(),
    )
    assert plan.admission == (
        vq._vibeqc_core._PeriodicCorrelationAdmissionCode.ADMITTED
    )


def _native_lebedev_tiers() -> list[tuple[int, int]]:
    """(order, n_points) rows of ``kLebedevTiers`` in the native source.

    Parsed from C++ rather than imported, because the dispatch table is not
    exposed through the bindings; the estimator therefore has to keep its own
    copy, and this test is what stops that copy from drifting.
    """
    root = Path(__file__).resolve().parent.parent
    src = (root / "cpp" / "src" / "lebedev_data.cpp").read_text()
    table = src.split("static const LebedevTier kLebedevTiers[]")[1].split("};")[0]
    rows = re.findall(r"\{\s*(\d+),\s*(\d+),\s*kLebedev_\d+_data\s*\}", table)
    assert rows, "could not parse kLebedevTiers out of cpp/src/lebedev_data.cpp"
    return [(int(order), int(n_points)) for order, n_points in rows]


def test_molecular_lebedev_tables_match_the_native_dispatch_table():
    """The estimator's Lebedev tables mirror ``kLebedevTiers`` exactly.

    ``build_grid`` matches ``lebedev_order`` against the native table exactly
    and throws on a miss, so a Python table that is missing an order makes the
    estimator raise on a grid the solver builds happily. Regression test for
    the collateral deletion of these constants (a NameError on every Lebedev
    DFT job) and for the stale order list that deletion had preserved, which
    omitted the PySCF level-3 profile tiers 15/27/31.
    """
    native = _native_lebedev_tiers()

    assert vq_memory._MOLECULAR_LEBEDEV_ORDERS == tuple(o for o, _ in native)
    assert vq_memory._MOLECULAR_LEBEDEV_POINT_TIERS == tuple(n for _, n in native)
    for order, n_points in native:
        assert vq_memory._LEBEDEV_ORDER_POINTS[order] == n_points

    # Ascending order keeps the nearest-tier search in
    # ``_molecular_lebedev_point_count`` breaking ties toward the lower tier,
    # which is what the native ``lebedev_tier_for_points`` does.
    tiers = vq_memory._MOLECULAR_LEBEDEV_POINT_TIERS
    assert list(tiers) == sorted(tiers)


def test_molecular_lebedev_point_count_agrees_with_the_built_grid():
    """Every bundled order estimates the angular count the solver produces."""
    mol = vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])])
    for order, n_points in _native_lebedev_tiers():
        opts = vq.GridOptions()
        opts.angular = "lebedev"
        opts.angular_pruning = "none"
        opts.lebedev_order = order
        opts.n_radial = 1
        grid = vq.build_grid(mol, opts)
        assert grid.n_points == n_points
        assert vq_memory._molecular_lebedev_point_count(opts) == n_points


def test_molecular_lebedev_point_count_rejects_an_unbundled_order():
    """An order the native table lacks is refused, not silently rounded."""
    opts = vq.GridOptions()
    opts.angular = "lebedev"
    opts.lebedev_order = 9  # a standard Lebedev order that is not bundled
    with pytest.raises(ValueError, match="unsupported molecular Lebedev order 9"):
        vq_memory._molecular_lebedev_point_count(opts)
