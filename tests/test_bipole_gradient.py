"""BIPOLE gradient validation.

The BIPOLE gradient (``vibeqc.bipole_gradient``) is a RESEARCH PREVIEW:
RHF/UHF Γ now use gauge-consistent Ewald electrostatics, a fixed-density
FD spheropole Hellmann-Feynman term, local-energy Pulay, and Bloch-CPHF
orbital relaxation, matching exact FD to ~1e-4-1e-7 Ha/bohr. RHF/UHF
multi-k have the corrected W/J^LR convention; their padded radial M5 routes
add the finite-domain density adjoint plus a full real-linear k/spin-coupled
response, pinned on maintained asymmetric [2,1,1] regressions. Gamma-local
RKS/UKS have maintained LDA KS-CPHF regressions; padded multi-k KS,
pair-resolved, and fractional certification remain gated. The production
gradient is the
finite-difference path
``compute_bipole_gradient_fd``, which differentiates converged driver
energies. Asymmetric legacy-gauge HF SCF is unsupported; its retired
nonstationary rows do not certify analytic or finite-difference forces.

These tests therefore:

* lock in the multi-k Pulay ``W(k)→W(g)`` inverse-Bloch fold (the one
  analytic-term bug that was fixed) — fast, no SCF;
* validate the FD *production* gradient (per-method, translational
  invariance, step stability) at Γ and multi-k for RHF/UHF/RKS/UKS;
* pin the gradient's research-preview status (warning + RHF/UHF Γ and
  historical/padded multi-k agreement) so future certification work is
  noticed.

Run:
    .venv/bin/python -m pytest tests/test_bipole_gradient.py -x -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    SCFAccelerator,
    compute_overlap_lattice,
    monkhorst_pack,
)
from vibeqc.bipole_gradient import (
    _bloch_fold_w_matrices,
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
    compute_bipole_gradient_uhf,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
from vibeqc.periodic_gradient_multi_k import _bloch_fold_w_per_k

ANG2BOHR = 1.0 / 0.529177210903


@pytest.mark.parametrize("method", ["rks", "uks"])
@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_ks_gradient_nondefault_xc_radius_matches_fd(method, functional):
    """The gradient must use the same finite AO-image sum as the SCF grid.

    H3 is a doublet, so the UKS rows exercise unequal spin densities. The
    preview's 1e-3-bohr grid-motion difference leaves a few microhartree/bohr
    of error on this coarse H3 grid; the relaxed SCF derivative is checked
    independently at two smaller steps. The old radius mismatch produces
    errors above 1e-3 Ha/bohr in all four rows.
    """
    import importlib
    import vibeqc.bipole_gradient as gradients

    driver = getattr(
        importlib.import_module(f"vibeqc.pbc_bipole_{method}"),
        f"run_pbc_bipole_{method}",
    )
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.becke_image_radius_bohr = 3.0
    opts.grid.n_radial = 16
    opts.grid.n_theta = 7
    opts.grid.n_phi = 12
    opts.functional = functional
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10

    def run(displacement):
        atoms = [
            vq.Atom(1, [0.2, 0.1, 0.0]),
            vq.Atom(1, [1.6 + displacement, 0.1, 0.0]),
        ]
        if method == "uks":
            atoms.append(vq.Atom(1, [2.4, 1.1, 0.2]))
        system = vq.PeriodicSystem(3, 7.0 * np.eye(3), atoms)
        if method == "uks":
            system.multiplicity = 2
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        mesh = monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
        result = driver(
            system, basis, mesh, opts, functional=functional,
            ewald_omega=0.6, ewald_precision=1e-10,
            sr_image_precision=None, use_fock_symmetry=False,
            use_fock_symmetry_reduce=False, progress=False,
        )
        assert result.converged
        return system, basis, mesh, result

    system, basis, mesh, result = run(0.0)
    with pytest.warns(UserWarning, match="preview"):
        analytic = getattr(gradients, f"compute_bipole_gradient_{method}")(
            system, basis, result, lattice_opts=opts.lattice_opts,
            kmesh=mesh, grid_options=opts.grid, becke_image_radius_bohr=3.0,
        )
    finite_differences = []
    for step in (1e-4, 1e-5):
        minus = run(-step)[3].energy
        plus = run(step)[3].energy
        finite_differences.append((plus - minus) / (2.0 * step))
    assert abs(finite_differences[0] - finite_differences[1]) < 1e-7
    tolerance = 1e-7 if method == "rks" else 1e-5
    assert abs(analytic[1, 0] - finite_differences[1]) < tolerance
    assert opts.lattice_opts.becke_image_radius_bohr == 0.0


@pytest.mark.parametrize("method", ["rks", "uks"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("supplied_options", [False, True])
def test_ks_gradient_xc_radius_preserves_lattice_options(
    monkeypatch, method, periodic, supplied_options,
):
    """Public gradient propagation preserves caller cutoffs and screening."""
    import importlib
    import vibeqc.bipole_gradient as gradients

    system = vq.PeriodicSystem(3, 7.0 * np.eye(3), [vq.Atom(2, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 5.0
    opts.becke_image_radius_bohr = 3.0
    opts.grid.n_radial = 8
    opts.grid.n_theta = 5
    opts.grid.n_phi = 8
    driver = getattr(
        importlib.import_module(f"vibeqc.pbc_bipole_{method}"),
        f"run_pbc_bipole_{method}",
    )
    result = driver(
        system, basis, mesh, opts, functional="pbe",
        sr_image_precision=None, use_fock_symmetry=False,
        use_fock_symmetry_reduce=False, progress=False,
    )
    assert result.converged
    options = opts.lattice_opts if supplied_options else None
    if options is not None:
        # These values are only inspected at the gradient boundary. Choosing
        # nondefaults makes omissions in the options copy observable.
        from vibeqc._vibeqc_core import CoulombMethod

        options.coulomb_method = CoulombMethod.EWALD_3D
        options.eri_interaction_cutoff_bohr = 6.5
        options.pair_complete_1e = True
        options.screening_exchange_threshold = 4e-9
        options.screening_overlap_threshold = 1e-7
        options.schwarz_threshold = 2e-9
        options.schwarz_threshold_forces = 3e-10
        options.slab_ewald_alpha = 0.7
        options.sr_range_screening = True
        options.sr_sparse_traversal = False
    source = options if options is not None else LatticeSumOptions()
    # Derive the contract from the native object, independently of the
    # production copy helper's field list, so a missing field is observable.
    fields = [
        field for field in dir(source)
        if not field.startswith("_") and not callable(getattr(source, field))
    ]
    expected = {field: getattr(source, field) for field in fields}

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        actual = kwargs["lattice_opts"]
        for field, value in expected.items():
            if field == "becke_image_radius_bohr" and periodic:
                value = 3.0
            assert getattr(actual, field) == value, field
        if periodic and supplied_options:
            assert actual is not options
        raise Captured

    monkeypatch.setattr(gradients, "_compute_bipole_gradient_corrected_gamma", capture)
    with pytest.warns(UserWarning, match="preview"), pytest.raises(Captured):
        getattr(gradients, f"compute_bipole_gradient_{method}")(
            system, basis, result, lattice_opts=options, kmesh=mesh,
            grid_options=opts.grid, use_periodic_becke=periodic,
            becke_image_radius_bohr=3.0,
        )
    for field, value in expected.items():
        assert getattr(source, field) == value, field


@pytest.fixture(autouse=True)
def _historical_sr_domain_for_analytic_regressions(monkeypatch):
    """Keep historical analytic-vs-FD cases on their original domain.

    Dedicated RHF/UHF/hybrid-RKS tests opt into the supported padded M5 domain
    explicitly. The broader legacy matrix stays unpadded so each regression
    continues to isolate the convention it originally certified; unsupported
    M5 combinations are pinned by the public fail-closed guard below.
    """
    for driver in (
        vq.run_pbc_bipole_rhf,
        vq.run_pbc_bipole_uhf,
        vq.run_pbc_bipole_rks,
        vq.run_pbc_bipole_uks,
    ):
        monkeypatch.setitem(driver.__kwdefaults__, "sr_image_precision", None)
        monkeypatch.setitem(
            driver.__kwdefaults__, "use_fock_symmetry_reduce", False
        )


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
# NOTE on lattice cutoffs: every SCF that enters the corrected
# (Ewald-exchange-split) gauge passes the fold-support preflight
# (`raise_if_bipole_fold_unreliable`, refusal at S-fold drift > 1e-2,
# reliable-support target < 1e-4). Fixture cutoffs below were chosen per
# system from a measured drift-vs-cutoff ladder (2026-08-05, after the
# 2026-08-03 pair-separation image-enumeration fix shifted the measured
# drifts) as the first sampled cutoff whose drift is stably < 1e-4.
# The ladder is non-monotonic (discrete radial cell shells), so don't
# lower a cutoff without re-measuring `s_fold_truncation_drift` on the
# exact fixture geometry + k-mesh.

def _build_mgo_primitive():
    """MgO FCC primitive — 1 Mg + 1 O."""
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


def _build_h2_box(a_bohr: float = 5.0, bond_bohr: float = 1.4):
    """H2 in a cubic box. ``a=5`` bohr gives genuine k-dispersion so the
    multi-k Pulay fold is actually exercised."""
    lattice = a_bohr * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, bond_bohr])]
    return vq.PeriodicSystem(3, lattice, atoms)


def _build_beh2_box(a_bohr: float = 9.0):
    """Linear, *asymmetric* BeH2 (different Be–H lengths) in a cubic box.
    Closed-shell (6 e⁻), 3 atoms not related by symmetry — so the
    per-atom forces are not trivially equal-and-opposite and Σ_A F_A ≈ 0
    is a genuine translational-invariance check."""
    lattice = a_bohr * np.eye(3)
    atoms = [
        vq.Atom(4, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 2.45]),
        vq.Atom(1, [0.0, 0.0, -2.75]),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


def _rhf_opts(cutoff: float = 6.0, conv: float = 1e-9):
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = 100
    opts.use_diis = True
    opts.conv_tol_energy = conv
    opts.initial_guess = InitialGuess.SAD
    return opts


def _ks_opts(cutoff: float = 6.0, conv: float = 1e-9):
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = 100
    opts.use_diis = True
    opts.conv_tol_energy = conv
    opts.initial_guess = InitialGuess.SAD
    return opts


def _small_svwn_multik_opts(cutoff: float = 5.0, conv: float = 1e-9):
    opts = _ks_opts(cutoff=cutoff, conv=conv)
    opts.functional = "svwn"
    opts.grid.n_radial = 20
    opts.grid.angular = "lebedev"
    opts.grid.lebedev_order = 17
    return opts


# ----------------------------------------------------------------------
# 1. Multi-k Pulay W(k)→W(g) inverse-Bloch fold (the fixed bug) — fast
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mesh", [[1, 1, 1], [2, 1, 1], [2, 2, 1]])
def test_pulay_fold_matches_proven_helper(mesh):
    """``_bloch_fold_w_matrices`` (BIPOLE multi-k Pulay term) must use the
    exact same inverse-Bloch convention as the proven
    ``periodic_gradient_multi_k._bloch_fold_w_per_k`` (the non-BIPOLE
    multi-k gradient). Pure-math check, no SCF.

    Before 2026-05-31 the BIPOLE Pulay term broadcast ``W(Γ)`` into every
    cell, which is correct only at Γ and produced wrong-sign multi-k
    forces.
    """
    sysp = _build_h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 6.0
    lat.nuclear_cutoff_bohr = 6.0
    kmesh = monkhorst_pack(sysp, mesh)
    n_k = len(list(kmesh.kpoints))
    nbf = basis.nbasis
    nocc = 1

    rng = np.random.default_rng(20260531 + n_k)
    C_per_k, eps_per_k, W_k_list = [], [], []
    for _ in range(n_k):
        C = rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))
        eps = np.sort(rng.standard_normal(nbf))
        C_per_k.append(C)
        eps_per_k.append(eps)
        C_occ = C[:, :nocc]
        # Matches _build_energy_weighted_density_closed (complex, factor 2).
        W_k_list.append(2.0 * (C_occ * eps[:nocc][None, :]) @ C_occ.conj().T)

    set_matrices = _bloch_fold_w_matrices(
        W_k_list, kmesh, compute_overlap_lattice(basis, sysp, lat)
    )
    set_coeffs = _bloch_fold_w_per_k(
        C_per_k,
        eps_per_k,
        [nocc] * n_k,
        kmesh,
        compute_overlap_lattice(basis, sysp, lat),
    )
    max_diff = max(
        float(
            np.max(
                np.abs(
                    np.asarray(set_matrices.blocks[c])
                    - np.asarray(set_coeffs.blocks[c])
                )
            )
        )
        for c in range(len(set_matrices.cells))
    )
    assert max_diff < 1e-12, f"fold convention mismatch: {max_diff:.3e}"


def test_pulay_fold_reduces_to_real_part_at_gamma():
    """At a single Γ point the fold must reduce to broadcasting Re[W(Γ)]
    into every cell (so the fix is a no-op at Γ, where the old broadcast
    was already correct)."""
    sysp = _build_h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 6.0
    lat.nuclear_cutoff_bohr = 6.0
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    nbf = basis.nbasis
    rng = np.random.default_rng(7)
    W_gamma = rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))

    folded = _bloch_fold_w_matrices(
        [W_gamma], kmesh, compute_overlap_lattice(basis, sysp, lat)
    )
    for c in range(len(folded.cells)):
        assert np.allclose(np.asarray(folded.blocks[c]), W_gamma.real, atol=1e-13)


# ----------------------------------------------------------------------
# 2. Analytic gradient is gated as a research preview
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_analytic_gradient_emits_research_preview_warning():
    """The analytic drivers must warn that the certified surface is still
    narrow/hybrid, so nothing uses them by accident."""
    sysp = _build_h2_box(a_bohr=8.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _rhf_opts()
    # Exercise the maintained symmetric legacy-Gamma control.
    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    # The warning text moved from "RESEARCH PREVIEW" to "maintained
    # preview" in ec00dca4; match the current wording.
    with pytest.warns(UserWarning, match="maintained preview") as caught:
        compute_bipole_gradient_rhf(
            sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh
        )
    assert "asymmetric legacy-HF SCF is unsupported" in str(caught[0].message)


def test_bipole_gradient_rhf_not_converged_warns(monkeypatch):
    """A non-converged result should still emit its own warning."""
    # Legacy-gauge fixture at a deliberately small cutoff: the
    # gauge-independent fold guard (hoisted 2026-08-06) would rightly
    # refuse it, but this test validates plumbing/derivative identities
    # on a FIXED truncated support (both arms use the same functional),
    # so fold reliability is not its subject -- bypass the measurement,
    # same rationale as the retired-G1 sentinels.
    import vibeqc.pbc_bipole_common as _common

    monkeypatch.setattr(
        _common, "s_fold_truncation_drift", lambda *a, **k: 1.0e-9
    )
    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.4)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1  # force non-convergence
    opts.use_diis = False
    opts.conv_tol_energy = 1e-15

    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,  # legacy gauge for the analytic preview
        ewald_precision=1e-6,
        progress=False,
    )
    assert not result.converged
    with pytest.warns(UserWarning, match="not converged"):
        compute_bipole_gradient_rhf(sysp, basis, result, lattice_opts=opts.lattice_opts)


def test_analytic_gradient_rejects_m5_energy_domains():
    """Unsupported M5 method and pair-domain combinations stay gated."""
    from vibeqc.bipole_gradient import (
        _reject_m5_domain_analytic_gradient,
        compute_bipole_gradient_rks,
        compute_bipole_gradient_uks,
    )

    result = SimpleNamespace(sr_image_extent_bohr=18.5)
    with pytest.raises(NotImplementedError, match="padded erfc SR ket-image"):
        compute_bipole_gradient_rhf(None, None, result)

    result = SimpleNamespace(
        sr_image_extent_bohr=None,
        pair_resolved_fock_domain=True,
    )
    with pytest.raises(NotImplementedError, match="pair-resolved Fock domain"):
        compute_bipole_gradient_rhf(None, None, result)

    result = SimpleNamespace(
        sr_image_extent_bohr=18.5,
        pair_resolved_fock_domain=False,
        exchange_ewald_split=True,
        mo_coeffs=[object(), object()],
    )
    _reject_m5_domain_analytic_gradient(result, "rhf")

    result.pair_resolved_fock_domain = True
    with pytest.raises(NotImplementedError, match="pair-resolved Fock domain"):
        _reject_m5_domain_analytic_gradient(result, "rhf")

    result = SimpleNamespace(
        sr_image_extent_bohr=18.5,
        pair_resolved_fock_domain=False,
        exchange_ewald_split=True,
        mo_coeffs_alpha=[object(), object()],
    )
    _reject_m5_domain_analytic_gradient(result, "uhf")

    result.pair_resolved_fock_domain = True
    with pytest.raises(NotImplementedError, match="pair-resolved Fock domain"):
        _reject_m5_domain_analytic_gradient(result, "uhf")

    result = SimpleNamespace(
        sr_image_extent_bohr=18.5,
        pair_resolved_fock_domain=True,
        exchange_ewald_split=True,
        mo_coeffs_alpha=[object()],
        occupations_alpha=[np.array([1.0, 1.0, 0.0])],
        occupations_beta=[np.array([1.0, 0.0, 0.0])],
    )
    _reject_m5_domain_analytic_gradient(result, "uhf")

    result.pair_resolved_fock_domain = True
    _reject_m5_domain_analytic_gradient(result, "uhf")

    result = SimpleNamespace(
        sr_image_extent_bohr=18.5,
        pair_resolved_fock_domain=False,
        exchange_ewald_split=True,
        mo_coeffs_alpha=[object()],
        occupations_alpha=[np.ones(2)],
        occupations_beta=[np.zeros(2)],
    )
    _reject_m5_domain_analytic_gradient(result, "uhf")

    result = SimpleNamespace(
        sr_image_extent_bohr=None,
        pair_resolved_fock_domain=False,
        exchange_ewald_split=True,
        mo_coeffs_alpha=[object()],
        smearing_temperature=0.01,
        occupations_alpha=[np.array([0.8, 0.2])],
        occupations_beta=[np.array([0.7, 0.3])],
    )
    with pytest.raises(NotImplementedError, match="Mermin free energy"):
        _reject_m5_domain_analytic_gradient(result, "uhf")

    result.pair_resolved_fock_domain = True
    with pytest.raises(NotImplementedError, match="Mermin free energy"):
        _reject_m5_domain_analytic_gradient(result, "uhf")

    result = SimpleNamespace(
        sr_image_extent_bohr=18.5,
        pair_resolved_fock_domain=True,
        exchange_ewald_split=True,
        mo_coeffs=[object()],
    )
    _reject_m5_domain_analytic_gradient(result, "rks")

    result = SimpleNamespace(
        sr_image_extent_bohr=18.5,
        pair_resolved_fock_domain=False,
        exchange_ewald_split=True,
        mo_coeffs=[object()],
        smearing_temperature=0.05,
        occupations=[np.array([1.8, 0.2])],
    )
    with pytest.raises(
        NotImplementedError, match="integer-occupation RKS"
    ):
        _reject_m5_domain_analytic_gradient(result, "rks")

    result = SimpleNamespace(
        sr_image_extent_bohr=18.5,
        pair_resolved_fock_domain=False,
        exchange_ewald_split=True,
        mo_coeffs_alpha=[object()],
        occupations_alpha=[np.array([1.0, 0.0])],
        occupations_beta=[np.array([1.0, 0.0])],
    )
    _reject_m5_domain_analytic_gradient(result, "uks")

    result.pair_resolved_fock_domain = True
    _reject_m5_domain_analytic_gradient(result, "uks")

    result.occupations_alpha = [np.array([0.8, 0.2])]
    with pytest.raises(NotImplementedError, match="fractional occupations"):
        _reject_m5_domain_analytic_gradient(result, "uks")


@pytest.mark.parametrize("pair_resolved", [False, True])
def test_gamma_jk_domain_energy_derivative_matches_density_fd(pair_resolved):
    """Native padded-domain J/K derivatives include both projector actions."""
    from vibeqc._vibeqc_core import (
        build_jk_2e_real_space_domains_gamma_derivative,
        direct_lattice_cells,
        make_lattice_matrix_set,
    )

    sysp = _build_h2_box(a_bohr=6.0, bond_bohr=2.0)
    if pair_resolved:
        vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _rhf_opts(cutoff=5.0).lattice_opts
    if pair_resolved:
        from vibeqc.bipole_symmetry_fock import pair_resolved_fock_mapping
        from vibeqc.pair_resolved_truncation import (
            atom_pair_shell_masks,
            mask_lattice_blocks_to_domain,
        )

        mapping = pair_resolved_fock_mapping(sysp, basis, opts)
        output_cells = list(mapping.cells)
        output_masks = list(mapping.output_masks)
        density_cells = list(mapping.density_domain.cells)
        density_masks = atom_pair_shell_masks(
            basis, mapping.density_domain.pairs_by_cell
        )
        internal_extent = max(
            6.1,
            max(
                float(np.linalg.norm(np.asarray(cell.r_cart)))
                for cell in output_cells
            )
            + 1.0e-6,
        )
    else:
        output_cells = list(direct_lattice_cells(sysp, opts.cutoff_bohr))
        output_masks = []
        density_cells = list(direct_lattice_cells(sysp, 12.1))
        density_masks = []
        internal_extent = 6.1
    internal_cells = list(direct_lattice_cells(sysp, internal_extent))
    internal_index = {
        tuple(int(x) for x in cell.index): i
        for i, cell in enumerate(internal_cells)
    }
    output_indices = [
        internal_index[tuple(int(x) for x in cell.index)]
        for cell in output_cells
    ]
    P = np.array([[1.1, -0.2], [-0.2, 0.9]])

    def build(block):
        density = make_lattice_matrix_set(
            basis.nbasis,
            density_cells,
            [np.asarray(block, dtype=float).copy() for _ in density_cells],
        )
        if pair_resolved:
            mask_lattice_blocks_to_domain(
                basis, mapping.density_domain, density
            )
        return build_jk_2e_real_space_domains_gamma_derivative(
            basis,
            sysp,
            opts,
            density,
            internal_cells,
            output_indices,
            0.4,
            True,
            output_masks,
            density_masks,
            np.asarray(block, dtype=float),
        )

    def energy(block, component):
        jk = build(block)
        blocks = jk.J.blocks if component == "J" else jk.K.blocks
        return 0.5 * sum(float(np.sum(block * np.asarray(x))) for x in blocks)

    jk = build(P)
    for component, analytic in (
        ("J", np.asarray(jk.J_gamma_energy_derivative)),
        ("K", np.asarray(jk.K_gamma_energy_derivative)),
    ):
        fd = np.zeros_like(P)
        step = 1.0e-5
        for i in range(P.shape[0]):
            for j in range(i, P.shape[1]):
                direction = np.zeros_like(P)
                direction[i, j] = 1.0
                direction[j, i] = 1.0
                slope = (
                    energy(P + step * direction, component)
                    - energy(P - step * direction, component)
                ) / (2.0 * step)
                fd[i, j] = fd[j, i] = slope if i == j else 0.5 * slope
        np.testing.assert_allclose(analytic, fd, atol=1.0e-9)

    j_forward = sum(np.asarray(block) for block in jk.J.blocks)
    assert np.max(np.abs(jk.J_gamma_energy_derivative - j_forward)) > 1.0e-5


def test_multicell_jk_domain_density_adjoint_matches_directional_fd():
    """The finite-domain J/K adjoint differentiates every density block.

    The padded multi-k energy is a bilinear with different Fock-output and
    density-support domains.  Non-homogeneous blocks make the forward-only
    shortcut observably wrong and pin the transpose action needed by CPHF.
    """
    from vibeqc._vibeqc_core import (
        build_jk_2e_real_space_domains,
        build_jk_2e_real_space_domains_density_derivative,
        direct_lattice_cells,
        make_lattice_matrix_set,
    )

    sysp = _build_h2_box(a_bohr=6.0, bond_bohr=2.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _rhf_opts(cutoff=5.0).lattice_opts
    opts.schwarz_threshold = 0.0
    internal_cells = list(direct_lattice_cells(sysp, 6.1))
    output_cells = list(direct_lattice_cells(sysp, opts.cutoff_bohr))
    internal_index = {
        tuple(int(x) for x in cell.index): i
        for i, cell in enumerate(internal_cells)
    }
    output_indices = [
        internal_index[tuple(int(x) for x in cell.index)]
        for cell in output_cells
    ]
    density_cells = list(direct_lattice_cells(sysp, 12.1))
    density_index = {
        tuple(int(x) for x in cell.index): i
        for i, cell in enumerate(density_cells)
    }
    output_density_indices = [
        density_index[tuple(int(x) for x in cell.index)]
        for cell in output_cells
    ]

    rng = np.random.default_rng(8411)
    density_blocks = [
        rng.normal(size=(basis.nbasis, basis.nbasis))
        for _ in density_cells
    ]
    directions = [
        rng.normal(size=(basis.nbasis, basis.nbasis))
        for _ in density_cells
    ]

    def make_density(blocks):
        return make_lattice_matrix_set(
            basis.nbasis,
            density_cells,
            [np.asarray(block, dtype=float) for block in blocks],
        )

    def build(blocks, *, derivative):
        builder = (
            build_jk_2e_real_space_domains_density_derivative
            if derivative
            else build_jk_2e_real_space_domains
        )
        return builder(
            basis,
            sysp,
            opts,
            make_density(blocks),
            internal_cells,
            output_indices=output_indices,
            omega=0.4,
            compute_exchange=True,
        )

    def energy(blocks, component):
        jk = build(blocks, derivative=False)
        fock_blocks = jk.J.blocks if component == "J" else jk.K.blocks
        return 0.5 * sum(
            float(
                np.sum(
                    blocks[density_pos]
                    * np.asarray(fock_blocks[internal_pos])
                )
            )
            for density_pos, internal_pos in zip(
                output_density_indices, output_indices
            )
        )

    jk = build(density_blocks, derivative=True)
    step = 1.0e-6
    plus = [
        block + step * direction
        for block, direction in zip(density_blocks, directions)
    ]
    minus = [
        block - step * direction
        for block, direction in zip(density_blocks, directions)
    ]
    for component, derivative in (
        ("J", jk.J_density_energy_derivative),
        ("K", jk.K_density_energy_derivative),
    ):
        analytic = sum(
            float(np.sum(np.asarray(block) * direction))
            for block, direction in zip(derivative.blocks, directions)
        )
        finite_difference = (
            energy(plus, component) - energy(minus, component)
        ) / (2.0 * step)
        assert analytic == pytest.approx(finite_difference, abs=1.0e-8)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("name", ["bipole-fd-custom", "sto-3g"])
@pytest.mark.parametrize("through_optimizer", [False, True])
def test_fd_gradient_preserves_explicit_basis(monkeypatch, method, name, through_optimizer):
    """Every displaced driver sees the supplied atom-centered shells."""
    import importlib

    system = _build_h2_box(a_bohr=12.0)
    shells = [
        vq.ShellInfo(i, i * 2, False, [.7 + i, .25], [.6, -.2],
                     np.asarray(atom.xyz))
        for i, atom in enumerate(system.unit_cell)
    ]
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, name, False)
    original = list(basis.shells())
    reference_positions = np.array([a.xyz for a in system.unit_cell])
    calls = []

    def driver(displaced, moved, *args, **kwargs):
        positions = np.array([a.xyz for a in displaced.unit_cell])
        energy = 0.
        for old, new in zip(original, moved.shells(), strict=True):
            assert (new.atom_index, new.l, new.pure) == (old.atom_index, old.l, old.pure)
            np.testing.assert_array_equal(new.exponents, old.exponents)
            np.testing.assert_array_equal(new.coefficients, old.coefficients)
            expected_origin = (np.asarray(old.origin) + positions[old.atom_index]
                               - reference_positions[old.atom_index])
            np.testing.assert_allclose(new.origin, expected_origin, atol=1e-14, rtol=0)
            energy += float(np.dot(new.origin, new.origin))
        calls.append(moved)
        return SimpleNamespace(energy=energy, converged=True)

    module = importlib.import_module(
        "vibeqc.pbc_bipole" if method == "rhf" else f"vibeqc.pbc_bipole_{method}"
    )
    monkeypatch.setattr(module, f"run_pbc_bipole_{method}", driver)
    mesh = monkhorst_pack(system, [1, 1, 1])
    if through_optimizer:
        from vibeqc.bipole_optimize import _compute_forces

        actual = _compute_forces(
            system, basis, None, method.upper(), None, kmesh=mesh,
            basis_name=name, opts=None, functional=None, bipole_kwargs={},
            force_mode="fd", fd_step_bohr=1e-3,
        )
    else:
        actual = compute_bipole_gradient_fd(system, basis, mesh, method=method)
    np.testing.assert_allclose(actual, [2*np.asarray(s.origin) for s in original],
                               atol=1e-9, rtol=0)
    assert len(calls) == 12
    for old, unchanged in zip(original, basis.shells(), strict=True):
        np.testing.assert_array_equal(unchanged.origin, old.origin)


def test_fd_custom_basis_matches_explicit_displaced_scf():
    """Real converged forces follow the supplied basis, not its library name."""
    positions = np.array([[0., 0., 0.], [0., 0., 1.4]])

    def geometry(dz=0.):
        xyz = positions.copy()
        xyz[1, 2] += dz
        system = vq.PeriodicSystem(3, 12*np.eye(3), [vq.Atom(1, p) for p in xyz])
        # Reconstruct these explicit primitives independently of the helper.
        basis = vq.BasisSet(system.unit_cell_molecule(), [
            vq.ShellInfo(i, 0, True, [.7+.4*i], [1.], p)
            for i, p in enumerate(xyz)
        ], "sto-3g", False)
        return system, basis

    system, basis = geometry()
    mesh = monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    options = PeriodicRHFOptions()
    options.lattice_opts.cutoff_bohr = 5.
    options.lattice_opts.nuclear_cutoff_bohr = 8.
    options.initial_guess = InitialGuess.HCORE
    options.conv_tol_energy = 1e-12
    options.conv_tol_grad = 1e-10
    options.max_iter = 100
    kwargs = dict(ewald_omega=.6, ewald_precision=1e-10, sr_image_precision=None,
                  use_fock_symmetry=False, use_fock_symmetry_reduce=False)
    step = 1e-4
    custom = compute_bipole_gradient_fd(
        system, basis, mesh, options, step_bohr=step, **kwargs,
    )
    reloaded = compute_bipole_gradient_fd(
        system, "sto-3g", mesh, options, step_bohr=step, **kwargs,
    )
    energies = []
    for displacement in [-step, step]:
        displaced, moved = geometry(displacement)
        result = run_pbc_bipole_rhf(displaced, moved, mesh, options, progress=False, **kwargs)
        assert result.converged
        energies.append(result.energy)
    manual = (energies[1] - energies[0]) / (2*step)
    assert custom[1, 2] == pytest.approx(manual, abs=1e-9, rel=0)
    assert np.max(np.abs(custom - reloaded)) > 1e-3


def test_fd_gradient_rejects_nonconverged_displacement(monkeypatch):
    """The production FD path must not differentiate a failed displaced SCF."""
    import vibeqc.pbc_bipole as pbc_bipole

    sysp = _build_h2_box(a_bohr=6.0)
    opts = _rhf_opts()
    calls = []

    def fake_run_pbc_bipole_rhf(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(energy=-1.0, converged=False, n_iter=opts.max_iter)

    monkeypatch.setattr(pbc_bipole, "run_pbc_bipole_rhf", fake_run_pbc_bipole_rhf)

    with pytest.raises(
        RuntimeError,
        match=r"RHF SCF did not converge for atom 0, coord x, step \+0\.001",
    ):
        compute_bipole_gradient_fd(
            sysp,
            "sto-3g",
            monkhorst_pack(sysp, [1, 1, 1]),
            opts,
            method="RHF",
            step_bohr=1e-3,
        )

    assert len(calls) == 1

    calls.clear()
    grad = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        monkhorst_pack(sysp, [1, 1, 1]),
        opts,
        method="RHF",
        step_bohr=1e-3,
        require_converged=False,
    )
    assert grad.shape == (2, 3)
    assert np.all(np.isfinite(grad))
    assert len(calls) == 12


def test_ks_analytic_gradient_rejects_fractional_occupations():
    """The *legacy*-gauge KS analytic gradient still requires integer
    occupations and rejects finite-T / fractional ones. (The corrected
    Ewald-exchange-split gauge handles fractional occupations directly via the
    Mermin free-energy form — see
    ``test_gamma_rks_fractional_occupation_matches_fd``.) These SimpleNamespace
    results carry no ``exchange_ewald_split`` flag, so they take the legacy path
    and must raise."""
    from vibeqc.bipole_gradient import (
        compute_bipole_gradient_rks,
        compute_bipole_gradient_uks,
    )

    sysp = _build_h2_box(a_bohr=8.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    rks_smeared = SimpleNamespace(
        converged=True,
        smearing_temperature=0.005,
        occupations=[np.array([1.5, 0.5])],
    )
    with pytest.raises(NotImplementedError, match="finite-temperature"):
        compute_bipole_gradient_rks(
            sysp, basis, rks_smeared, lattice_opts=LatticeSumOptions()
        )

    rks_fractional = SimpleNamespace(
        converged=True,
        smearing_temperature=0.0,
        occupations=[np.array([1.5, 0.5])],
    )
    with pytest.raises(NotImplementedError, match="fractional-occupation"):
        compute_bipole_gradient_rks(
            sysp, basis, rks_fractional, lattice_opts=LatticeSumOptions()
        )

    uks_smeared = SimpleNamespace(
        converged=True,
        smearing_temperature=0.005,
        occupations_alpha=[np.array([0.7, 0.3])],
        occupations_beta=[np.array([0.6, 0.4])],
    )
    with pytest.raises(NotImplementedError, match="finite-temperature"):
        compute_bipole_gradient_uks(
            sysp, basis, uks_smeared, lattice_opts=LatticeSumOptions()
        )


def test_ks_analytic_gradient_emits_multik_warning():
    """Multi-k KS analytic gradients now warn (maintained preview) instead of raising."""
    from vibeqc.bipole_gradient import (
        compute_bipole_gradient_rks,
        compute_bipole_gradient_uks,
    )

    sysp = _build_h2_box(a_bohr=8.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    eye = np.eye(basis.nbasis)
    nbf = basis.nbasis
    lat = LatticeSumOptions()
    D = compute_overlap_lattice(basis, sysp, lat)

    rks_multik = SimpleNamespace(
        converged=True,
        smearing_temperature=0.0,
        occupations=[
            np.array([2.0] + [0.0] * (nbf - 1)),
            np.array([2.0] + [0.0] * (nbf - 1)),
        ],
        mo_coeffs=[eye, eye],
        mo_energies=[np.zeros(nbf), np.zeros(nbf)],
        density=D,
    )
    with pytest.warns(UserWarning, match="multi-k KS analytic"):
        compute_bipole_gradient_rks(sysp, basis, rks_multik, lattice_opts=lat)

    uks_multik = SimpleNamespace(
        converged=True,
        smearing_temperature=0.0,
        occupations_alpha=[
            np.array([1.0] + [0.0] * (nbf - 1)),
            np.array([1.0] + [0.0] * (nbf - 1)),
        ],
        occupations_beta=[
            np.array([1.0] + [0.0] * (nbf - 1)),
            np.array([1.0] + [0.0] * (nbf - 1)),
        ],
        mo_coeffs_alpha=[eye, eye],
        mo_coeffs_beta=[eye, eye],
        mo_energies_alpha=[np.zeros(nbf), np.zeros(nbf)],
        mo_energies_beta=[np.zeros(nbf), np.zeros(nbf)],
        density_alpha=D,
        density_beta=D,
    )
    with pytest.warns(UserWarning, match="multi-k KS analytic"):
        compute_bipole_gradient_uks(sysp, basis, uks_multik, lattice_opts=lat)


# ----------------------------------------------------------------------
# 3. FD production gradient — per method, translational invariance, steps
# ----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "method,ks,functional",
    [
        ("RHF", False, None),
        ("UHF", False, None),
        ("RKS", True, "pbe"),
        ("UKS", True, "pbe"),
    ],
)
def test_fd_gradient_runs_and_balances_all_methods(method, ks, functional):
    """The exact FD gradient must run for every BIPOLE method and obey
    Newton's third law in a 2-atom cell (translational invariance ⇒
    F_0 = -F_1). Tight tolerance — this is the production force path."""
    sysp = _build_h2_box(a_bohr=6.0)
    # 11 bohr: drift 2.3e-5 (6 bohr sat at 9.1e-3, <10% under the refusal).
    opts = _ks_opts(cutoff=11.0) if ks else _rhf_opts(cutoff=11.0)
    grad = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        monkhorst_pack(sysp, [1, 1, 1]),
        opts,
        method=method,
        functional=functional,
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )
    net = np.abs(grad.sum(axis=0))
    assert np.all(np.isfinite(grad))
    assert float(np.max(net)) < 1e-4, f"{method}: |Σ_A F_A| = {net} (≠ 0)"


@pytest.mark.slow
def test_fd_gradient_translational_invariance_three_atoms():
    """Genuine ≥3-atom translational-invariance check (forces are not
    trivially equal-and-opposite). Σ_A F_A ≈ 0 for an asymmetric BeH2."""
    sysp = _build_beh2_box()
    grad = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        monkhorst_pack(sysp, [1, 1, 1]),
        # 13 bohr: drift 5.1e-5 (7 bohr refused at 1.2e-1).
        _rhf_opts(cutoff=13.0),
        method="RHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )
    # All three atoms carry non-trivial force along the molecular axis.
    assert float(np.max(np.abs(grad[:, 2]))) > 1e-3
    net = np.abs(grad.sum(axis=0))
    assert float(np.max(net)) < 2e-4, f"|Σ_A F_A| = {net} (≠ 0)"


@pytest.mark.slow
def test_fd_production_gradient_translational_invariance_ks():
    """Certify the **production** FD gradient (the optimiser/NEB default) is
    translationally invariant for the DFT methods on real asymmetric crystals
    — ``Σ_A F_A ≈ 0`` to FD precision when the SCF converges. This is the
    production guarantee for RKS/UKS forces (the analytic KS gradient is a
    gated research preview; see ``_warn_research_preview``)."""
    # RKS — asymmetric BeH2 (3 atoms not related by symmetry, so a genuine
    # sum-rule check). This row used LiH until 2026-08-05: the diffuse Li
    # 2sp tail (STO-3G exp ~0.064) keeps the LiH fold drift above the 1e-4
    # reliable-support target until a 24 bohr cutoff, where one SCF costs
    # ~8 min — 13 FD SCFs made this test multi-hour. The LiH/24-bohr
    # fixture still runs (single SCF) in
    # test_periodic_xc_grid_motion_correction_matches_moving_fd.
    s_rks = _build_beh2_box(a_bohr=9.0)
    # 13 bohr: drift 5.1e-5 (7 bohr refused at 1.2e-1).
    o_rks = _ks_opts(cutoff=13.0)
    g_rks = compute_bipole_gradient_fd(
        s_rks,
        "sto-3g",
        monkhorst_pack(s_rks, [1, 1, 1]),
        o_rks,
        method="RKS",
        functional="svwn",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )
    assert float(np.max(np.abs(g_rks.sum(axis=0)))) < 1e-5, (
        f"RKS FD ΣF = {g_rks.sum(axis=0)}"
    )
    # UKS — BeH doublet (asymmetric, open-shell)
    s_uks = vq.PeriodicSystem(
        3, 7.0 * np.eye(3), [vq.Atom(4, [0, 0, 0]), vq.Atom(1, [0, 0, 2.4])]
    )
    s_uks.multiplicity = 2
    # 16 bohr: drift 1.5e-5 (7 bohr refused at 4.9e-2).
    o_uks = _ks_opts(cutoff=16.0)
    o_uks.max_iter = 400
    g_uks = compute_bipole_gradient_fd(
        s_uks,
        "sto-3g",
        monkhorst_pack(s_uks, [1, 1, 1]),
        o_uks,
        method="UKS",
        functional="svwn",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )
    assert float(np.max(np.abs(g_uks.sum(axis=0)))) < 1e-4, (
        f"UKS FD ΣF = {g_uks.sum(axis=0)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,functional,mesh",
    [
        ("RHF", None, [2, 2, 1]),
        ("RHF", None, [2, 1, 1]),
        ("UHF", None, [2, 1, 1]),
        ("RKS", "svwn", [2, 1, 1]),
        ("RKS", "pbe", [2, 1, 1]),
    ],
)
def test_fd_production_gradient_multi_k_translational_invariance(
    method, functional, mesh
):
    """The **production** FD gradient must obey Newton's third law at
    multi-k for every method (the multi-k force path). ``Σ_A F_A ≈ 0`` to
    FD precision for H₂ on a k-mesh that samples genuine dispersion.

    (RHF at the metallic-flavoured [2,1,1] mesh was previously avoided
    here: at bond≈1.449 bohr the SCF returned a spurious E=−0.347 Ha basin
    that spiked the FD gradient. That was a DIIS-at-the-fixed-point driver
    bug — singular Pulay B-matrix → garbage post-convergence
    re-diagonalisation — *not* a near-degeneracy and not a force-path bug;
    fixed in ``pbc_bipole.py`` 2026-06-05, regression-pinned in
    ``test_pbc_bipole_diis_converged_basin.py``. RHF [2,1,1] is re-enabled
    here to keep the production force path covered on that mesh.)"""
    sysp = _build_h2_box(a_bohr=5.0, bond_bohr=1.45)
    # 12 bohr: drift 4.1e-5 over both meshes (6 bohr refused at 6.2e-2).
    opts = _ks_opts(cutoff=12.0) if functional else _rhf_opts(cutoff=12.0)
    grad = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        monkhorst_pack(sysp, mesh),
        opts,
        method=method,
        functional=functional,
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )
    assert np.all(np.isfinite(grad))
    net = float(np.max(np.abs(grad.sum(axis=0))))
    assert net < 1e-5, f"{method}/{functional} mesh={mesh}: |Σ_A F_A| = {net:.2e}"


@pytest.mark.slow
def test_fd_gradient_multi_k_step_stability():
    """Multi-k FD gradient must be stable w.r.t. the central-difference
    step (tightened from the old <0.1 to <1e-3)."""
    sysp = _build_h2_box(a_bohr=5.0)
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    # 12 bohr: drift 3.9e-5 on [2,1,1] (6 bohr refused at 6.1e-2).
    opts = _rhf_opts(cutoff=12.0)
    g1 = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )
    g2 = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RHF",
        step_bohr=2e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(g1 - g2)))
    assert max_abs < 1e-3, f"FD step instability: max|Δ| = {max_abs:.2e}"


@pytest.mark.slow
@pytest.mark.parametrize("padded", [False, True])
def test_multik_rhf_corrected_gauge_matches_fd(padded):
    """Multi-k corrected-gauge (Ewald-exchange-split) RHF *analytic* gradient
    matches the production FD gradient on asymmetric BeH₂ [2,1,1]/STO-3G.

    Exercises the landed ``_compute_bipole_gradient_corrected_multi_k``: per-q
    reciprocal exchange (K_LR) + per-k Madelung + the density inverse-Bloch-
    folded onto the gradient template (NOT ``result.density``). The padded
    RHF/UHF cases additionally pin the unequal-domain density adjoint, occupied
    Lagrangian correction, and full real-linear k/spin-coupled CPHF response. The
    historical domain is FD-clean to ~6e-8; padded M5 to ~2.6e-5 Ha/bohr."""
    sysp = _build_beh2_box(a_bohr=9.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    # 13 bohr: drift 5.1e-5 on [2,1,1] (6 bohr refused at 1.2e-1).
    opts = _rhf_opts(cutoff=13.0, conv=1e-10)
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    m5_kwargs = (
        {
            "sr_image_precision": 1.0e-6,
            "use_fock_symmetry_reduce": False,
        }
        if padded
        else {}
    )
    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_exchange_ewald_split=True,
        **m5_kwargs,
    )
    assert result.converged
    if padded:
        assert result.sr_image_extent_bohr > opts.lattice_opts.cutoff_bohr
        assert not result.pair_resolved_fock_domain
    else:
        assert result.sr_image_extent_bohr is None
    # #674: on this 9-bohr cell at the 13-bohr cutoff the corrected split's
    # default alpha is the erfc bound (0.330), above CRYSTAL's 0.311, and the
    # reciprocal envelope scales with it in the energy and in every gradient
    # term; the FD parity below is the >8-bohr energy/gradient consistency
    # check of that contract.
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
    from vibeqc.pbc_bipole_common import ewald_alpha_lower_bound

    _V = float(abs(np.linalg.det(np.asarray(sysp.lattice, dtype=float))))
    assert result.ewald_alpha_bohr_inv > crystal_default_ewald_alpha(_V)
    assert result.ewald_alpha_bohr_inv == pytest.approx(
        ewald_alpha_lower_bound(13.0, 1e-8), rel=1e-9
    )
    g_an = np.asarray(compute_bipole_gradient_rhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh))
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="RHF",
        step_bohr=1e-3, use_ewald_j_split=True, ewald_precision=1e-8,
        **m5_kwargs))
    np.testing.assert_allclose(g_an, g_fd, atol=5e-5 if padded else 1e-5,
        err_msg="multi-k corrected-gauge RHF analytic gradient diverges from FD")


@pytest.mark.slow
@pytest.mark.parametrize("padded", [False, True])
def test_multik_uhf_corrected_gauge_matches_fd(padded):
    """Multi-k corrected-gauge (Ewald-exchange-split) UHF *analytic* gradient
    matches the production FD gradient on an asymmetric BeH doublet
    [2,1,1]/STO-3G.

    The shared ``_compute_bipole_gradient_corrected_multi_k`` detects the
    open-shell result and builds the spin-resolved exchange
    ``2·Σ_σ ∂E_x[P_σ]`` (per-spin K_SR/K_LR/Madelung) with the open
    energy-weighted W; the total density drives the 1e/Coulomb/jellium/J^LR
    terms. The padded case pins the full-support ``D_alpha + D_beta`` used by
    J_SR, the spin-resolved finite-domain transpose, occupied Lagrangian, and
    alpha/beta-coupled real-linear response. Both routes are FD-clean
    (~1e-8)."""
    a = 12.0
    sysp = vq.PeriodicSystem(
        3, a * np.eye(3),
        [vq.Atom(4, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 2.5])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _rhf_opts(cutoff=6.0, conv=1e-10)
    opts.max_iter = 400
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    m5_kwargs = (
        {
            "sr_image_precision": 1.0e-6,
            "use_fock_symmetry_reduce": False,
        }
        if padded
        else {
            "sr_image_precision": None,
            "use_fock_symmetry_reduce": False,
        }
    )
    result = run_pbc_bipole_uhf(
        sysp, basis, kmesh, opts,
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
        **m5_kwargs)
    assert result.converged
    assert getattr(result, "exchange_ewald_split", False)
    if padded:
        assert result.sr_image_extent_bohr > opts.lattice_opts.cutoff_bohr
        assert not result.pair_resolved_fock_domain
    else:
        assert result.sr_image_extent_bohr is None
    g_an = np.asarray(compute_bipole_gradient_uhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh))
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="UHF",
        step_bohr=1e-3, use_ewald_j_split=True, ewald_precision=1e-8,
        **m5_kwargs))
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5,
        err_msg="multi-k corrected-gauge UHF analytic gradient diverges from FD")


@pytest.mark.slow
def test_multik_rks_corrected_gauge_matches_fd():
    """Multi-k corrected-gauge RKS (LDA) *analytic* gradient matches FD on
    H2 [2,1,1]/STO-3G. The shared multi-k core (exchange scaled by the
    functional's HF fraction; zero for SVWN) + the multi-k XC Pulay
    (``xc_lattice_gradient_contribution`` on the Bloch-folded density) +
    grid-motion correction."""
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    # 10 bohr: drift 1.1e-5 on [2,1,1] (5 bohr refused at 2.7e-2).
    opts = _small_svwn_multik_opts(cutoff=10.0, conv=1e-9)
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    result = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts, use_ewald_j_split=True,
        use_exchange_ewald_split=True, ewald_precision=1e-6, progress=False)
    assert result.converged
    g_an = np.asarray(compute_bipole_gradient_rks(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh,
        grid_options=opts.grid))
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="RKS", functional="svwn",
        step_bohr=1e-3, use_ewald_j_split=True, use_exchange_ewald_split=True,
        ewald_precision=1e-6))
    np.testing.assert_allclose(g_an, g_fd, atol=2e-5,
        err_msg="multi-k corrected-gauge RKS analytic gradient diverges from FD")


@pytest.mark.slow
def test_multik_uks_corrected_gauge_matches_fd():
    """Multi-k corrected-gauge UKS (spin-polarized LDA) *analytic* gradient
    matches FD on an H2+ doublet [2,1,1]/STO-3G: the per-spin multi-k XC Pulay
    on top of the shared open-shell core."""
    from vibeqc.bipole_gradient import compute_bipole_gradient_uks
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.55)
    sysp.charge = 1
    sysp.multiplicity = 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    # 10 bohr: drift 1.7e-5 on [2,1,1] (5 bohr refused at 3.6e-2).
    opts = _small_svwn_multik_opts(cutoff=10.0, conv=1e-9)
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    result = run_pbc_bipole_uks(
        sysp, basis, kmesh, opts, use_ewald_j_split=True,
        use_exchange_ewald_split=True, ewald_precision=1e-6, progress=False)
    assert result.converged
    g_an = np.asarray(compute_bipole_gradient_uks(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh,
        grid_options=opts.grid))
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="UKS", functional="svwn",
        step_bohr=1e-3, use_ewald_j_split=True, use_exchange_ewald_split=True,
        ewald_precision=1e-6))
    np.testing.assert_allclose(g_an, g_fd, atol=3e-5,
        err_msg="multi-k corrected-gauge UKS analytic gradient diverges from FD")


# ----------------------------------------------------------------------
# 2b. Fractional-occupation (finite-T smeared, Mermin free-energy) gradient
# ----------------------------------------------------------------------
def _smeared_ks_opts(functional="svwn", T=0.05):
    opts = PeriodicKSOptions()
    opts.functional = functional
    # 12 bohr: drift 1.4e-6 at Γ and on [2,1,1] for the H2 a=8/b=2.6
    # fixture both smeared tests use (6 bohr refused at 2.7e-2).
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 200
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.smearing_temperature = float(T)
    return opts


@pytest.mark.slow
def test_gamma_rks_fractional_occupation_matches_fd():
    """Corrected-gauge Γ RKS *fractional-occupation* (finite-T smeared) analytic
    gradient matches FD on H₂/STO-3G at T=0.05 Ha. The smeared SCF minimises the
    Mermin free energy A = E − T·S, so the production FD path differentiates A;
    the analytic gradient matches it with fractional D and
    W = Σ_i f_i ε_i C_iC_i† and NO occupation-response term (A is stationary
    w.r.t. the occupations at convergence). Genuinely fractional here
    (worst occupation ~0.11 from integer); FD-clean (~2e-8)."""
    from vibeqc.bipole_gradient import (
        compute_bipole_gradient_rks,
        _is_fractional_ks_occupation,
    )
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_h2_box(a_bohr=8.0, bond_bohr=2.6)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _smeared_ks_opts("svwn", 0.05)
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    result = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts, use_ewald_j_split=True,
        use_exchange_ewald_split=True, ewald_precision=1e-8, progress=False)
    assert result.converged
    # The occupations must actually be fractional, else the test is vacuous.
    assert _is_fractional_ks_occupation(result, "rks")
    # A zero commutator does not prove occupation self-consistency when the
    # orbitals are symmetry-fixed.  Rebuilding and diagonalising the physical
    # Fock once from the returned density must leave the Fermi occupations
    # unchanged; the pre-fix driver stopped three iterations in with a
    # 1.22e-5 occupation residual.
    fixed_point_opts = _smeared_ks_opts("svwn", 0.05)
    fixed_point_opts.max_iter = 1
    fixed_point_opts.use_diis = False
    fixed_point = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        fixed_point_opts,
        initial_density=[
            np.asarray(block).copy() for block in result.density.blocks
        ],
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
    np.testing.assert_allclose(
        fixed_point.occupations[0], result.occupations[0], atol=1e-8
    )
    g_an = np.asarray(compute_bipole_gradient_rks(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh,
        grid_options=opts.grid))
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="RKS", functional="svwn",
        step_bohr=1e-3, use_ewald_j_split=True, use_exchange_ewald_split=True,
        ewald_precision=1e-8))
    np.testing.assert_allclose(g_an, g_fd, atol=1e-6, err_msg=(
        "corrected-gauge Γ fractional-occupation RKS gradient diverges from FD"))


@pytest.mark.slow
def test_multik_rks_fractional_occupation_matches_fd():
    """Corrected-gauge *multi-k* RKS fractional-occupation gradient matches FD on
    H₂/STO-3G [2,1,1] at T=0.05 Ha — the multi-k fractional density + per-k
    fractional W (``_build_per_k_density_matrices_frac`` /
    ``_build_energy_weighted_density_closed_frac``). FD-clean (~4e-9)."""
    from vibeqc.bipole_gradient import (
        compute_bipole_gradient_rks,
        _is_fractional_ks_occupation,
    )
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_h2_box(a_bohr=8.0, bond_bohr=2.6)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _smeared_ks_opts("svwn", 0.05)
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    result = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts, use_ewald_j_split=True,
        use_exchange_ewald_split=True, ewald_precision=1e-8, progress=False)
    assert result.converged
    assert _is_fractional_ks_occupation(result, "rks")
    g_an = np.asarray(compute_bipole_gradient_rks(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh,
        grid_options=opts.grid))
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="RKS", functional="svwn",
        step_bohr=1e-3, use_ewald_j_split=True, use_exchange_ewald_split=True,
        ewald_precision=1e-8))
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5, err_msg=(
        "corrected-gauge multi-k fractional-occupation RKS gradient diverges from FD"))


def test_open_shell_fractional_w_builder_reduces_to_integer():
    """The fractional open-shell energy-weighted-density builder must reduce
    EXACTLY to the integer-Aufbau builder when fed an integer occupation vector.

    This pins the open-shell fractional path (``_build_energy_weighted_density_
    open_frac``) without needing a converged smeared-UKS SCF (finite-T UKS is a
    separate smearing-convergence concern). The closed-shell fractional builder
    is FD-validated by ``test_gamma_rks_fractional_occupation_matches_fd``; the
    open-shell one is its per-spin generalisation, so reduction-to-integer plus
    that FD pin certify it."""
    from vibeqc.bipole_gradient import (
        _build_energy_weighted_density_open,
        _build_energy_weighted_density_open_frac,
    )

    rng = np.random.default_rng(7)
    nbf, n_alpha, n_beta, n_k = 5, 3, 2, 2
    Ca = [rng.standard_normal((nbf, nbf)) + 0j for _ in range(n_k)]
    Cb = [rng.standard_normal((nbf, nbf)) + 0j for _ in range(n_k)]
    ea = [rng.standard_normal(nbf) for _ in range(n_k)]
    eb = [rng.standard_normal(nbf) for _ in range(n_k)]
    occ_a = [np.array([1.0] * n_alpha + [0.0] * (nbf - n_alpha)) for _ in range(n_k)]
    occ_b = [np.array([1.0] * n_beta + [0.0] * (nbf - n_beta)) for _ in range(n_k)]

    W_int = _build_energy_weighted_density_open(Ca, ea, Cb, eb, n_alpha, n_beta)
    W_frac = _build_energy_weighted_density_open_frac(Ca, ea, occ_a, Cb, eb, occ_b)
    for wi, wf in zip(W_int, W_frac):
        np.testing.assert_allclose(wf, wi, atol=1e-12, err_msg=(
            "fractional open-shell W builder must match integer builder at "
            "integer Aufbau occupation"))


# ----------------------------------------------------------------------
# 3b. Ewald E_nn gradient kernel (analytic-gradient kernel 1) — exact,
#     no SCF. Validates ∂E_nn/∂R_A against a central FD of the same Ewald
#     energy, the gauge-correct replacement for the truncated direct sum.
# ----------------------------------------------------------------------
def _ewald_opts_for(system):
    """The exact EwaldOptions the BIPOLE energy uses for E_nn on a 3D
    system (α + matched real/reciprocal cutoffs)."""
    from vibeqc._vibeqc_core import EwaldOptions
    from vibeqc.bipole_ext_el_pole import (
        crystal_default_ewald_alpha,
        crystal_ewald_reciprocal_cutoff,
    )

    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    opts = EwaldOptions()
    opts.alpha = crystal_default_ewald_alpha(V)
    opts.real_cutoff_bohr = 12.0
    opts.recip_cutoff_bohr_inv = crystal_ewald_reciprocal_cutoff(V)
    return opts


@pytest.mark.parametrize(
    "build",
    [_build_mgo_primitive, lambda: _build_beh2_box(a_bohr=9.0)],
)
def test_ewald_nuclear_repulsion_gradient_matches_fd(build):
    """``ewald_nuclear_repulsion_gradient`` must reproduce a central FD of
    ``ewald_nuclear_repulsion`` on the same α / cutoffs (the energy's exact
    gauge). Both an ionic (MgO) and a covalent asymmetric (BeH2) cell."""
    from vibeqc._vibeqc_core import (
        ewald_nuclear_repulsion,
        ewald_nuclear_repulsion_gradient,
    )

    sysp = build()
    opts = _ewald_opts_for(sysp)
    atoms = list(sysp.unit_cell)
    lattice = np.asarray(sysp.lattice, dtype=float)

    g_an = np.asarray(ewald_nuclear_repulsion_gradient(sysp, opts))

    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return ewald_nuclear_repulsion(
                    vq.PeriodicSystem(3, lattice, pert), opts
                )

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"Ewald E_nn grad vs FD: max|Δ| = {max_abs:.2e}"
    # Translational invariance: Σ_A ∂E_nn/∂R_A = 0.
    assert float(np.max(np.abs(g_an.sum(axis=0)))) < 1e-8


# ----------------------------------------------------------------------
# 3c. Ewald V_ne gradient kernel (analytic-gradient kernel 2) — the
#     Hellmann–Feynman gradient Σ_g D(g)·∂V_ne(g)/∂R of the BIPOLE Ewald
#     V_ne (erfc V_short + reciprocal V_long + background), vs a central FD
#     of Σ_g D(g)·V_ne(g) at FIXED D (no SCF).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pair_complete", [False, True])
@pytest.mark.parametrize("fixture", ["h2", "chi-he2"])
def test_v_ne_ewald_gradient_matches_fd(fixture, pair_complete):
    """``_v_ne_ewald_gradient`` vs central FD of the V_ne energy
    Σ_g D(g)·V_ne(g) with a fixed random density —
    isolates the explicit (integral) V_ne derivative from the SCF."""
    from vibeqc._vibeqc_core import (
        CoulombMethod,
        LatticeSumOptions,
        compute_overlap_lattice,
    )
    from vibeqc.bipole_ext_el_pole import (
        crystal_default_ewald_alpha,
        crystal_ewald_reciprocal_cutoff,
    )
    from vibeqc.bipole_gradient import _v_ne_ewald_gradient
    from vibeqc.pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )

    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.4)
    basis_name, cutoff = "sto-3g", 9.0
    if fixture == "chi-he2":
        # #704: off-centre Gaussian products need more than alpha-only
        # nuclear support. Keep energy and analytic image sets matched.
        sysp = vq.PeriodicSystem(3, np.diag([8., 16., 16.]), [
            vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [4.3, 0.2, 0.1]),
        ])
        basis_name, cutoff = "6-31g", 15.0
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), basis_name)
    nbf = basis.nbasis
    V = float(abs(np.linalg.det(lattice)))
    alpha = crystal_default_ewald_alpha(V)
    Kmax = crystal_ewald_reciprocal_cutoff(V)

    lat = LatticeSumOptions()
    lat.pair_complete_1e = pair_complete
    lat.cutoff_bohr = cutoff
    lat.nuclear_cutoff_bohr = cutoff
    lat.coulomb_method = CoulombMethod.EWALD_3D

    ncells = len(compute_overlap_lattice(basis, sysp, lat).cells)
    rng = np.random.default_rng(0)
    D = compute_overlap_lattice(basis, sysp, lat)
    for c in range(ncells):
        D.set_block(c, rng.standard_normal((nbf, nbf)) * 0.1)
    Dblocks = [np.asarray(D.blocks[c]).copy() for c in range(ncells)]

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), basis_name)
        Sl = compute_overlap_lattice(b, s, lat)
        ew = _crystal_ewald_options(
            lat,
            alpha_bohr_inv=alpha,
            tolerance=1e-8,
            recip_cutoff_bohr_inv=Kmax,
        )
        Vne, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
            b, s, lat, ew, Sl, precision=1e-8, K_max=Kmax
        )
        return sum(
            float(np.sum(Dblocks[c] * np.asarray(Vne.blocks[c]))) for c in range(ncells)
        )

    g_an = _v_ne_ewald_gradient(sysp, basis, D, lat, alpha)
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return energy_at(pert)

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"V_ne Ewald grad vs FD: max|Δ| = {max_abs:.2e}"


# ----------------------------------------------------------------------
# 3d. Long-range J gradient kernel (analytic-gradient kernel 4) — the
#     Hellmann–Feynman gradient of E_J^LR = ½ Σ_K kernel(K)|ρ̂(K)|² vs a
#     central FD of ½ Σ_g tr[D(g) J^LR(g)] at FIXED D (no SCF). Function
#     is validated standalone; wiring into the analytic driver follows the
#     screened-J_SR kernel (which suppresses the full-Coulomb J).
# ----------------------------------------------------------------------
def test_j_long_range_ewald_gradient_matches_fd():
    """``_j_long_range_ewald_gradient`` vs central FD of the long-range
    Hartree energy on MgO with a fixed symmetric random density. Both
    energy and gradient use the matched CRYSTAL K_max envelope."""
    from vibeqc._vibeqc_core import (
        CoulombMethod,
        LatticeSumOptions,
        compute_overlap_lattice,
    )
    from vibeqc.bipole_ext_el_pole import (
        crystal_default_ewald_alpha,
        crystal_ewald_reciprocal_cutoff,
    )
    from vibeqc.bipole_fock_ewald import (
        _build_j_long_range_cache,
        compute_J_long_range_real_space_blocks,
    )
    from vibeqc.bipole_gradient import _j_long_range_ewald_gradient

    sysp = _build_mgo_primitive()
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    V = float(abs(np.linalg.det(lattice)))
    alpha = crystal_default_ewald_alpha(V)
    Kmax = crystal_ewald_reciprocal_cutoff(V)

    lat = LatticeSumOptions()
    lat.cutoff_bohr = 9.0
    lat.nuclear_cutoff_bohr = 9.0
    lat.coulomb_method = CoulombMethod.EWALD_3D

    ncells = len(compute_overlap_lattice(basis, sysp, lat).cells)
    rng = np.random.default_rng(0)
    D = compute_overlap_lattice(basis, sysp, lat)
    Dblocks = []
    for c in range(ncells):
        M = rng.standard_normal((nbf, nbf)) * 0.1
        M = 0.5 * (M + M.T)
        D.set_block(c, M)
        Dblocks.append(M)

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        Sl = compute_overlap_lattice(b, s, lat)
        Dp = compute_overlap_lattice(b, s, lat)
        for c in range(ncells):
            Dp.set_block(c, Dblocks[c])
        crc = np.array([np.asarray(c.r_cart, float) for c in Sl.cells])
        cache = _build_j_long_range_cache(b, s, crc, alpha, 1e-8, K_max=Kmax)
        blocks = compute_J_long_range_real_space_blocks(Dp, b, s, alpha, cache=cache)
        return 0.5 * sum(float(np.sum(Dblocks[c] * blocks[c])) for c in range(ncells))

    g_an = _j_long_range_ewald_gradient(sysp, basis, D, alpha)
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return energy_at(pert)

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"J^LR Ewald grad vs FD: max|Δ| = {max_abs:.2e}"


def test_j_long_range_ewald_gradient_multi_k_matches_fd():
    """Multi-k ``J^LR`` gradient vs FD of ``1/2 |Σ_k w_k rho_k(K)|^2``.

    This pins the Bloch-summed complex-density derivative at fixed ``D(k)``:
    bra AO derivatives scatter to atom(μ), ket derivatives scatter to atom(ν),
    and all k-points fold through the weighted total rho used by the SCF Fock.
    """
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_overlap_lattice,
        monkhorst_pack,
    )
    from vibeqc.bipole_ext_el_pole import (
        crystal_default_ewald_alpha,
        crystal_ewald_reciprocal_cutoff,
    )
    from vibeqc.bipole_fock_ewald import (
        _bloch_pair_ft_from_cache,
        _build_j_long_range_cache,
    )
    from vibeqc.bipole_gradient import _j_long_range_ewald_gradient_multi_k

    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.45)
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis

    lat = LatticeSumOptions()
    lat.cutoff_bohr = 5.0
    lat.nuclear_cutoff_bohr = 5.0

    D_real = compute_overlap_lattice(basis, sysp, lat)
    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in D_real.cells], dtype=float
    )
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]
    weights = [float(w) for w in kmesh.weights]

    rng = np.random.default_rng(123)
    per_k = []
    for _ in kpts:
        z = rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))
        per_k.append(0.5 * (z + z.conj().T))

    V = float(abs(np.linalg.det(lattice)))
    alpha = crystal_default_ewald_alpha(V)
    Kmax = crystal_ewald_reciprocal_cutoff(V)

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        cache = _build_j_long_range_cache(b, s, cells_r_cart, alpha, 1e-8, K_max=Kmax)
        rho_total = np.zeros(cache.K_vectors.shape[0], dtype=np.complex128)
        for w_k, k_arr, D_k in zip(weights, kpts, per_k):
            ft_bloch = _bloch_pair_ft_from_cache(cache, k_arr)
            rho = np.einsum("mn,mnk->k", D_k, ft_bloch)
            rho_total += w_k * rho
        return 0.5 * float(
            np.real(np.sum(cache.kernel * np.conj(rho_total) * rho_total))
        )

    g_an = _j_long_range_ewald_gradient_multi_k(
        sysp, basis, D_real, kmesh, alpha, per_k
    )
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return energy_at(pert)

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-7, f"multi-k J^LR grad vs FD: max|Δ| = {max_abs:.2e}"


# ----------------------------------------------------------------------
# 3e. Screened short-range J gradient (analytic-gradient kernel 3).
#     eri_lattice_gradient_contribution gains j_scale (J-suppression) +
#     omega (erfc-screened operator). The generalization must (a) be
#     bit-identical to the old call at j_scale=1, omega=0 and (b) the
#     screened J_SR branch must match a FD of the screened-J energy.
# ----------------------------------------------------------------------
def test_eri_lattice_gradient_jscale_omega_parity():
    """j_scale=1, omega=0 must reproduce the legacy full-Coulomb J+K
    gradient bit-for-bit (the generalization is behaviour-preserving)."""
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_overlap_lattice,
        eri_lattice_gradient_contribution,
    )

    sysp = _build_mgo_primitive()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 5.0
    lat.nuclear_cutoff_bohr = 5.0
    ncells = len(compute_overlap_lattice(basis, sysp, lat).cells)
    rng = np.random.default_rng(0)
    D = compute_overlap_lattice(basis, sysp, lat)
    for c in range(ncells):
        M = rng.standard_normal((nbf, nbf)) * 0.1
        D.set_block(c, 0.5 * (M + M.T))

    g_default = np.asarray(eri_lattice_gradient_contribution(basis, sysp, D, lat, 1.0))
    g_gen = np.asarray(
        eri_lattice_gradient_contribution(basis, sysp, D, lat, 1.0, 1.0, 0.0)
    )
    assert float(np.max(np.abs(g_default - g_gen))) < 1e-14


def test_eri_lattice_gradient_screened_jsr_matches_fd():
    """The screened J_SR branch (alpha_hf=0, j_scale=1, omega>0) must match
    a central FD of the screened short-range Coulomb energy
    ½ Σ_g tr[D(g) J_SR(g)] at fixed D."""
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_fock_2e_real_space,
        compute_overlap_lattice,
        eri_lattice_gradient_contribution,
    )
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha

    sysp = _build_mgo_primitive()
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    alpha = crystal_default_ewald_alpha(float(abs(np.linalg.det(lattice))))

    lat = LatticeSumOptions()
    lat.cutoff_bohr = 7.0
    lat.nuclear_cutoff_bohr = 7.0
    ncells = len(compute_overlap_lattice(basis, sysp, lat).cells)
    rng = np.random.default_rng(0)
    D = compute_overlap_lattice(basis, sysp, lat)
    Dblocks = []
    for c in range(ncells):
        M = (
            0.5
            * (rng.standard_normal((nbf, nbf)) + rng.standard_normal((nbf, nbf)).T)
            * 0.1
        )
        D.set_block(c, M)
        Dblocks.append(M)

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        Dp = compute_overlap_lattice(b, s, lat)
        for c in range(ncells):
            Dp.set_block(c, Dblocks[c])
        Jsr = build_fock_2e_real_space(b, s, lat, Dp, 0.0, alpha)
        return 0.5 * sum(
            float(np.sum(Dblocks[c] * np.asarray(Jsr.blocks[c]))) for c in range(ncells)
        )

    g_an = np.asarray(
        eri_lattice_gradient_contribution(basis, sysp, D, lat, 0.0, 1.0, alpha)
    )
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return energy_at(pert)

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"J_SR grad vs FD: max|Δ| = {max_abs:.2e}"


def test_eri_lattice_gradient_screened_ksr_matches_fd():
    """The screened K_SR branch (alpha_hf=1, j_scale=0, omega>0) must match
    a central FD of the exchange energy -1/4 Σ_g tr[D(g) K_SR(g)] at fixed D."""
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_jk_2e_real_space,
        compute_overlap_lattice,
        eri_lattice_gradient_contribution,
    )
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha

    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.4)
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    alpha = crystal_default_ewald_alpha(float(abs(np.linalg.det(lattice))))

    lat = LatticeSumOptions()
    lat.cutoff_bohr = 5.0
    lat.nuclear_cutoff_bohr = 5.0
    ncells = len(compute_overlap_lattice(basis, sysp, lat).cells)
    rng = np.random.default_rng(1)
    D = compute_overlap_lattice(basis, sysp, lat)
    Dblocks = []
    for c in range(ncells):
        M = (
            0.5
            * (rng.standard_normal((nbf, nbf)) + rng.standard_normal((nbf, nbf)).T)
            * 0.1
        )
        D.set_block(c, M)
        Dblocks.append(M)

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        Dp = compute_overlap_lattice(b, s, lat)
        for c in range(ncells):
            Dp.set_block(c, Dblocks[c])
        Ksr = build_jk_2e_real_space(b, s, lat, Dp, alpha).K
        return -0.25 * sum(
            float(np.sum(Dblocks[c] * np.asarray(Ksr.blocks[c])))
            for c in range(ncells)
        )

    g_an = np.asarray(
        eri_lattice_gradient_contribution(basis, sysp, D, lat, 1.0, 0.0, alpha)
    )
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return energy_at(pert)

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"K_SR grad vs FD: max|Δ| = {max_abs:.2e}"


@pytest.mark.parametrize("term", ["j", "k"])
@pytest.mark.parametrize("pair_resolved", [False, True])
def test_eri_lattice_gradient_padded_sr_domain_matches_fd(
    term, pair_resolved
):
    """The M5 internal image ball is differentiated without enlarging the
    Fock-output cell list. Density support independently reaches twice the
    ordinary cutoff, as in the production Fock build.

    This is the low-level contract used by the production RHF Gamma analytic
    gradient: ``c_g`` remains on the ordinary cutoff template, ``P(h)`` is
    available on the 2x-cutoff difference domain, and the translated
    ``(c_lambda, c_sigma)`` ket-pair traversal uses the padded short-range
    radius.
    """
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_jk_2e_real_space_domains,
        direct_lattice_cells,
        eri_lattice_gradient_contribution,
        make_lattice_matrix_set,
    )
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha

    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.4)
    if pair_resolved:
        vq.attach_symmetry(sysp)
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    alpha = crystal_default_ewald_alpha(float(abs(np.linalg.det(lattice))))
    extent = 12.0

    lat = LatticeSumOptions()
    lat.cutoff_bohr = 5.0
    lat.nuclear_cutoff_bohr = 5.0
    if pair_resolved:
        from vibeqc.bipole_symmetry_fock import pair_resolved_fock_mapping

        mapping = pair_resolved_fock_mapping(sysp, basis, lat)
        cells_out = list(mapping.cells)
        output_masks = list(mapping.output_masks)
        cells_density = list(mapping.density_domain.cells)
        extent = max(
            extent,
            max(float(np.linalg.norm(np.asarray(c.r_cart))) for c in cells_out)
            + 1e-6,
        )
    else:
        cells_out = list(direct_lattice_cells(sysp, lat.cutoff_bohr))
        output_masks = []
        cells_density = list(direct_lattice_cells(sysp, 2.0 * lat.cutoff_bohr))
    rng = np.random.default_rng(7)
    M = rng.standard_normal((nbf, nbf)) * 0.1
    M = 0.5 * (M + M.T)
    D = make_lattice_matrix_set(
        nbf, cells_density, [M.copy() for _ in cells_density]
    )
    if pair_resolved:
        from vibeqc.pair_resolved_truncation import (
            mask_lattice_blocks_to_domain,
        )

        mask_lattice_blocks_to_domain(basis, mapping.density_domain, D)

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        density_cells_p = list(direct_lattice_cells(s, 2.0 * lat.cutoff_bohr))
        Dp = make_lattice_matrix_set(
            nbf,
            list(mapping.density_domain.cells)
            if pair_resolved
            else density_cells_p,
            [
                M.copy()
                for _ in (
                    mapping.density_domain.cells
                    if pair_resolved
                    else density_cells_p
                )
            ],
        )
        if pair_resolved:
            mask_lattice_blocks_to_domain(
                b, mapping.density_domain, Dp
            )
        cells_internal = list(direct_lattice_cells(s, extent))
        internal_index = {
            tuple(int(x) for x in cell.index): i
            for i, cell in enumerate(cells_internal)
        }
        output_indices = [
            internal_index[tuple(int(x) for x in cell.index)]
            for cell in cells_out
        ]
        jk = build_jk_2e_real_space_domains(
            b,
            s,
            lat,
            Dp,
            cells_internal,
            output_indices,
            output_masks,
            alpha,
            True,
        )
        blocks = jk.J.blocks if term == "j" else jk.K.blocks
        coefficient = 0.5 if term == "j" else -0.25
        return coefficient * sum(
            float(np.sum(M * np.asarray(blocks[out_idx])))
            for out_idx in output_indices
        )

    cells_internal = list(direct_lattice_cells(sysp, extent))
    g_an = np.asarray(
        eri_lattice_gradient_contribution(
            basis,
            sysp,
            D,
            lat,
            0.0 if term == "j" else 1.0,
            1.0 if term == "j" else 0.0,
            alpha,
            False,
            cells_internal,
            cells_out,
            output_masks,
        )
    )
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):
            pert_plus = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
            pert_minus = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
            xyz_plus = list(pert_plus[a].xyz)
            xyz_minus = list(pert_minus[a].xyz)
            xyz_plus[d] += h
            xyz_minus[d] -= h
            pert_plus[a] = vq.Atom(pert_plus[a].Z, xyz_plus)
            pert_minus[a] = vq.Atom(pert_minus[a].Z, xyz_minus)
            g_fd[a, d] = (energy_at(pert_plus) - energy_at(pert_minus)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    domain = "pair-resolved" if pair_resolved else "radial"
    assert max_abs < 1e-6, (
        f"padded {domain} {term.upper()}_SR grad vs FD: "
        f"max|Δ|={max_abs:.2e}"
    )


def test_eri_lattice_gradient_full_k_bipole_exchange_matches_fd():
    """BIPOLE's legacy full-K branch differentiates -1/4 Σ_g tr[D(g) K(g)].

    The default full-Coulomb periodic K gradient keeps the true-periodic
    DIRECT_TRUNCATED density slots for the low-level periodic-gradient
    regression. BIPOLE's legacy energy, however, contracts the direct
    exchange blocks as an explicit real-space exchange energy; the opt-in
    exchange-energy convention must match a fixed-density central FD.
    """
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_jk_2e_real_space,
        compute_overlap_lattice,
        eri_lattice_gradient_contribution,
    )

    sysp = _build_h2_box(a_bohr=5.0, bond_bohr=1.4)
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    lat = LatticeSumOptions()
    lat.cutoff_bohr = 6.0
    lat.nuclear_cutoff_bohr = 6.0
    D = compute_overlap_lattice(basis, sysp, lat)
    Dblocks = []
    for cell in D.cells:
        idx = np.asarray(cell.index, dtype=int)
        scale = 1.0 if np.all(idx == 0) else (0.2 if abs(idx[0]) == 1 else 0.05)
        block = scale * np.array([[0.5, 0.4], [0.4, 0.5]])
        D.set_block(len(Dblocks), block)
        Dblocks.append(block)

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        Dp = compute_overlap_lattice(b, s, lat)
        for c, block in enumerate(Dblocks):
            Dp.set_block(c, block)
        K = build_jk_2e_real_space(b, s, lat, Dp, 0.0).K
        return -0.25 * sum(
            float(np.sum(Dblocks[c] * np.asarray(K.blocks[c])))
            for c in range(len(Dblocks))
        )

    g_an = np.asarray(
        eri_lattice_gradient_contribution(
            basis,
            sysp,
            D,
            lat,
            1.0,
            0.0,
            0.0,
            exchange_energy_convention=True,
        )
    )
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return energy_at(pert)

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"full-K BIPOLE grad vs FD: max|Δ| = {max_abs:.2e}"


# ----------------------------------------------------------------------
# 3f. Spheropole gradient kernel (hybrid-gradient kernel 5) — the
#     Hellmann–Feynman gradient of the EXT EL-SPHEROPOLE energy vs a
#     central FD of compute_ext_el_spheropole at FIXED P (no SCF).
# ----------------------------------------------------------------------
def test_spheropole_ewald_gradient_matches_fd():
    """``_spheropole_ewald_gradient`` vs an independent central FD of the
    EXT EL-SPHEROPOLE energy (emultipole2 v3) on MgO at a fixed random
    density. Regression guard that the gradient tracks the *current*
    spheropole energy (it caught the 2026-06-01 emultipole2 reimplementation
    that obsoleted the old spheropole kernel-grad path)."""
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_overlap_lattice,
    )
    from vibeqc.bipole_ext_el_pole import compute_ext_el_spheropole
    from vibeqc.bipole_gradient import _spheropole_ewald_gradient

    sysp = _build_mgo_primitive()
    lattice = np.asarray(sysp.lattice, dtype=float)
    atoms = list(sysp.unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 7.0
    lat.nuclear_cutoff_bohr = 7.0
    ncells = len(compute_overlap_lattice(basis, sysp, lat).cells)
    rng = np.random.default_rng(0)
    D = compute_overlap_lattice(basis, sysp, lat)
    Dblocks = []
    for c in range(ncells):
        M = (
            0.5
            * (rng.standard_normal((nbf, nbf)) + rng.standard_normal((nbf, nbf)).T)
            * 0.1
        )
        D.set_block(c, M)
        Dblocks.append(M)

    def energy_at(pert_atoms):
        s = vq.PeriodicSystem(3, lattice, pert_atoms)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        Dp = compute_overlap_lattice(b, s, lat)
        for c in range(ncells):
            Dp.set_block(c, Dblocks[c])
        return compute_ext_el_spheropole(Dp, b, s, lat)

    g_an = _spheropole_ewald_gradient(sysp, basis, D, lat)
    h = 1e-5
    g_fd = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for d in range(3):

            def _shift(sign):
                pert = [vq.Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(pert[a].xyz)
                xyz[d] += sign * h
                pert[a] = vq.Atom(pert[a].Z, xyz)
                return energy_at(pert)

            g_fd[a, d] = (_shift(+1) - _shift(-1)) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"spheropole grad vs FD: max|Δ| = {max_abs:.2e}"


# ----------------------------------------------------------------------
# 4. Analytic/hybrid↔FD agreement at Γ — BIPOLE RHF/UHF gradients.
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_analytic_gradient_matches_fd_gamma():
    """The full hybrid BIPOLE RHF gradient matches the exact FD gradient
    at Γ to ~1e-3 Ha/bohr (in practice ~1e-7 here — FD-step-limited). This
    is the milestone done-signal (formerly xfail).

    Assembles: four analytic Ewald-gauge electrostatic kernels (E_nn,
    V_ne, screened J_SR, J^LR-reciprocal with the mixed Bloch/local ρ̂
    convention), the fixed-density FD spheropole term, the full-Coulomb
    exchange, and the Pulay term
    built from ∂E/∂P(0) consistent with BIPOLE's Γ-only LOCAL energy
    (_corrected_w_gamma_closed). This symmetric Γ control is maintained;
    broader analytic gradients stay gated as a research preview."""
    sysp = _build_h2_box(a_bohr=5.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    # 8 bohr: S-fold drift 5.4e-3 (6 bohr is refused at 6.1e-2).
    opts = _rhf_opts(cutoff=8.0)
    # Legacy gauge on BOTH sides: the analytic gradient implements the
    # Γ-local gauge (projection + full-Coulomb K + spheropole) and
    # refuses Ewald-exchange-split results; the FD reference must
    # differentiate the same legacy energy (option (b), 2026-06-10).
    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_rhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 1e-3, f"analytic vs FD: max|Δ| = {max_abs:.4f} Ha/bohr"
    # Retain argument validation from the retired asymmetric capability row.
    with pytest.raises(ValueError, match="unknown cphf_rhs"):
        compute_bipole_gradient_rhf(
            sysp, basis, result, lattice_opts=opts.lattice_opts,
            kmesh=kmesh, cphf_rhs="not-a-mode",
        )


def test_analytic_rhf_gradient_matches_fd_multi_k(monkeypatch):
    """RHF legacy-gauge multi-k analytic gradient matches FD.

    The multi-k SCF Fock carries the full J^LR background and omits the
    post-SCF spheropole operator.  The analytic Pulay W(k) must therefore use
    the energy derivative ``-0.5*v_bg*S(k) + dE_sph/dP(k)`` correction on top
    of the diagonalized Fock eigenvalue W. The legacy full-Coulomb K gradient
    also must use BIPOLE's exchange-energy convention, not the default
    true-periodic DIRECT_TRUNCATED K slots.
    """
    # Legacy-gauge fixture at a deliberately small cutoff: the
    # gauge-independent fold guard (hoisted 2026-08-06) would rightly
    # refuse it, but this test validates plumbing/derivative identities
    # on a FIXED truncated support (both arms use the same functional),
    # so fold reliability is not its subject -- bypass the measurement,
    # same rationale as the retired-G1 sentinels.
    import vibeqc.pbc_bipole_common as _common

    monkeypatch.setattr(
        _common, "s_fold_truncation_drift", lambda *a, **k: 1.0e-9
    )
    sysp = _build_h2_box(a_bohr=5.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    opts = _rhf_opts()
    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_rhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 1e-3, f"multi-k analytic vs FD: max|Δ| = {max_abs:.4f}"


@pytest.mark.slow
def test_analytic_uhf_gradient_matches_fd_gamma():
    """The analytic BIPOLE *UHF* gradient at Γ matches the exact FD gradient
    for a genuinely spin-polarised system (triplet H₂: n_α=2, n_β=0).

    Exercises the per-spin local-energy Pulay (``_corrected_w_gamma_open``,
    ∂E/∂P_σ = shared − α_HF·K[P_σ]) and the spin-resolved exchange gradient
    (``2·(∂E_x[P_α] + ∂E_x[P_β])``). The triplet has no occ-virt block
    (alpha fully fills both STO-3G MOs, beta is empty), so the post-SCF
    spheropole orbital-relaxation residual (a separate, asymmetric-only
    effect handled by the spheropole Z-vector) does not enter — this is an
    exact (~1e-7) UHF check. RHF/UHF multi-k has maintained [2,1,1] coverage.

    Pins the LEGACY Γ-local gauge explicitly (use_exchange_ewald_split=False
    on both the SCF and the FD reference): the analytic gradient implements
    that gauge only. Under the corrected option-(b) gauge (default at Γ under
    the J split since Phase 4b, 66ac0252) the analytic path refuses — see
    test_analytic_uhf_gradient_refuses_corrected_gauge below."""
    sysp = _build_h2_box(a_bohr=5.0)
    sysp.multiplicity = 3  # triplet: n_α=2, n_β=0
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    # 8 bohr: S-fold drift 5.4e-3 (6 bohr is refused at 6.1e-2).
    opts = _rhf_opts(cutoff=8.0)
    result = run_pbc_bipole_uhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_uhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="UHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 1e-3, f"UHF analytic vs FD: max|Δ| = {max_abs:.4f} Ha/bohr"


@pytest.mark.slow
def test_analytic_uhf_gradient_corrected_gauge_gamma_matches_fd():
    """UHF Γ corrected-gauge analytic gradient (un-refused 2026-06-15).

    Same standard-variational structure as RHF (full Bloch density → no
    Bloch-CPHF), with the exchange spin-resolved (``2·(∂E_x[Pα]+∂E_x[Pβ])``)
    and ``W = W_α + W_β``; the total-density 1e/Coulomb/jellium terms are
    shared. Must match the production FD path. (Was previously a refusal
    test; the Γ corrected-gauge UHF gradient is now supported — multi-k
    still refuses.) Uses an asymmetric BeH doublet so both spin densities
    are non-trivial and the force is on a gentle (FD-clean) curve."""
    a = 12.0
    lattice = a * np.eye(3)
    sysp = vq.PeriodicSystem(
        3, lattice,
        [vq.Atom(4, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 2.5])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _rhf_opts(cutoff=5.0, conv=1e-10)
    opts.max_iter = 400
    result = run_pbc_bipole_uhf(
        sysp, basis, kmesh, opts,
        use_ewald_j_split=True, ewald_precision=1e-8,
        sr_image_precision=1e-6, use_fock_symmetry_reduce=False,
        progress=False,
    )
    assert result.converged
    assert getattr(result, "exchange_ewald_split", False)
    assert result.sr_image_extent_bohr > opts.lattice_opts.cutoff_bohr
    assert result.pair_resolved_fock_domain is False
    g_an = compute_bipole_gradient_uhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    )
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="UHF", step_bohr=1e-3,
        use_ewald_j_split=True, ewald_precision=1e-8,
        sr_image_precision=1e-6, use_fock_symmetry_reduce=False,
    ))
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("pair_resolved", [False, True])
def test_analytic_uhf_gradient_padded_saturated_spin_matches_fd(pair_resolved):
    """Padded Gamma UHF includes the radial or pair-domain J/K adjoint.

    Triplet H2/STO-3G fully occupies the alpha AO space, so its density
    response is purely the occupied-overlap Pulay term. The pair-resolved
    route additionally pins the transpose of its output and density shell-pair
    projectors. Both J and K contributions are required and largely cancel.
    """
    # Keep both rows away from the hard 2*cutoff density-domain boundary:
    # translated pairs sit at n*a and n*a +/- bond, so at a=6, bond=2 every
    # integer cutoff whose double is 6n or 6n+/-2 (10, 11, 12, ...) parks a
    # pair exactly on the boundary and a central displacement changes the
    # discrete domain rather than differentiating one smooth energy branch.
    # 11.5 bohr (radial row): boundary at 23, clear by 1 bohr; drift 5.6e-5
    # (5 bohr refused at 1.3e-1). 11 bohr (pair row, a=6.2): boundary at 22,
    # nearest pair 22.8; drift 2.7e-5 (5 bohr refused at 1.1e-1).
    a_bohr = 6.2 if pair_resolved else 6.0
    sysp = _build_h2_box(a_bohr=a_bohr, bond_bohr=2.0)
    sysp.multiplicity = 3
    if pair_resolved:
        vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _rhf_opts(cutoff=11.0 if pair_resolved else 11.5, conv=1e-10)
    opts.max_iter = 300
    opts.lattice_opts.sr_range_screening = True
    m5_kwargs = {
        "use_ewald_j_split": True,
        "ewald_precision": 1e-8,
        "sr_image_precision": 1e-6,
        "use_fock_symmetry_reduce": pair_resolved,
    }
    # Keep this derivative test on its original fixed radial support;
    # the bounded Ewald alpha may select a shorter automatic extent.
    if not pair_resolved:
        m5_kwargs["sr_image_extent_bohr"] = 28.5690327065
    result = run_pbc_bipole_uhf(
        sysp, basis, kmesh, opts, progress=False, **m5_kwargs
    )
    assert result.converged
    assert result.sr_image_extent_bohr > opts.lattice_opts.cutoff_bohr
    if not pair_resolved:
        # Both analytic and displaced SCFs use this fixed domain.
        assert result.sr_image_extent_bohr == pytest.approx(28.5690327065)
    assert result.pair_resolved_fock_domain is pair_resolved
    g_an = compute_bipole_gradient_uhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    )
    g_fd = np.asarray(
        compute_bipole_gradient_fd(
            sysp,
            "sto-3g",
            kmesh,
            opts,
            method="UHF",
            step_bohr=1e-3,
            **m5_kwargs,
        )
    )
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)



def test_analytic_uhf_gradient_matches_fd_multi_k_high_spin(monkeypatch):
    """UHF legacy-gauge multi-k high-spin plumbing matches FD.

    Triplet H2/STO-3G has alpha fully occupied and beta empty, so there is no
    occ-virt orbital-relaxation space. This pins the open-shell multi-k
    ``J^LR`` total-density path plus the per-spin W(k) background/spheropole
    correction and the BIPOLE exchange-energy convention for the legacy
    full-Coulomb K gradient.
    """
    # Legacy-gauge fixture at a deliberately small cutoff: the
    # gauge-independent fold guard (hoisted 2026-08-06) would rightly
    # refuse it, but this test validates plumbing/derivative identities
    # on a FIXED truncated support (both arms use the same functional),
    # so fold reliability is not its subject -- bypass the measurement,
    # same rationale as the retired-G1 sentinels.
    import vibeqc.pbc_bipole_common as _common

    monkeypatch.setattr(
        _common, "s_fold_truncation_drift", lambda *a, **k: 1.0e-9
    )
    sysp = _build_h2_box(a_bohr=5.0)
    sysp.multiplicity = 3
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    opts = _rhf_opts()
    result = run_pbc_bipole_uhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_uhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="UHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 1e-3, f"UHF multi-k analytic vs FD: max|Δ| = {max_abs:.4f}"


def test_rks_gamma_fock_reconstruction_matches_scf_exchange_fraction(monkeypatch):
    """The Γ CPHF Fock reconstruction must respect the method's α_HF.

    Pure RKS has α_HF=0, so the reconstructed two-electron Bloch Fock must not
    contain the closed-shell RHF ``-1/2 K`` term. The SCF driver also Hermitizes
    the Bloch Fock before diagonalisation; the response helper must do the same
    or the occ-virt stationarity check sees an antisymmetric lattice-fold tail.
    """
    # Legacy-gauge fixture at a deliberately small cutoff: the
    # gauge-independent fold guard (hoisted 2026-08-06) would rightly
    # refuse it, but this test validates plumbing/derivative identities
    # on a FIXED truncated support (both arms use the same functional),
    # so fold reliability is not its subject -- bypass the measurement,
    # same rationale as the retired-G1 sentinels.
    import vibeqc.pbc_bipole_common as _common

    monkeypatch.setattr(
        _common, "s_fold_truncation_drift", lambda *a, **k: 1.0e-9
    )
    from vibeqc._vibeqc_core import Functional, build_xc_periodic
    from vibeqc.bipole_gradient import (
        _build_ks_grid,
        _home_cell_index,
        _reconstruct_bipole_fock_gamma_builder,
    )
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    lattice = 7.6 * np.eye(3)
    sysp = vq.PeriodicSystem(
        3, lattice, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 3.8])]
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    # The fixed-grid legacy preview has an approximately 1e-9 Ha numerical
    # energy floor.  Its purpose here is an independent 1e-8 Fock identity,
    # so demanding sub-floor SCF changes only turns a stable fixed point into
    # a false non-convergence.
    opts = _ks_opts(cutoff=8.0, conv=1e-8)
    opts.conv_tol_grad = 1e-8
    opts.functional = "svwn"
    # This test pins the reconstructed Fock, not the independent FMIXING
    # state machine.  A numerically zero but positive value still activates
    # that post-DIIS path and can prevent the deliberately tight preview SCF
    # from satisfying its convergence gate.
    opts.fock_mixing = 0.0
    # Legacy Γ-local gauge pinned explicitly: the analytic-gradient / CPHF
    # reconstruction helpers implement the legacy assembly; the corrected
    # option-(b) gauge (default at Γ under the J split since Phase 4a/4b)
    # is refused by the analytic path.
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged

    S_g, H_g, f2e, _, _ = _reconstruct_bipole_fock_gamma_builder(
        sysp,
        basis,
        opts.lattice_opts,
        float(result.ewald_alpha_bohr_inv),
        0.0,
    )
    del S_g
    home = _home_cell_index(list(result.density.cells))
    P_home = np.asarray(result.density.blocks[home], dtype=float)
    grid = _build_ks_grid(sysp, opts.grid, True, 10.0)
    xc = build_xc_periodic(
        basis,
        sysp,
        grid,
        Functional("svwn", 1),
        result.density,
        opts.lattice_opts,
    )
    Vxc_g = sum(np.asarray(block, dtype=float) for block in xc.V_xc.blocks)
    Vxc_g = 0.5 * (Vxc_g + Vxc_g.T)
    F_rebuilt = H_g + f2e(P_home, P_home) + Vxc_g
    F_scf = np.real(np.asarray(result.fock[0]))
    F_scf = 0.5 * (F_scf + F_scf.T)
    max_abs = float(np.max(np.abs(F_rebuilt - F_scf)))
    assert max_abs < 1e-8, f"RKS Γ Fock rebuild mismatch: {max_abs:.2e}"


@pytest.mark.slow
def test_rks_lih_physical_fock_confirmation_fixed_support(monkeypatch):
    """The tight LiH SCF confirms a stationary physical Fock within 100 steps.

    This preserves the 2026-07-25 state-machine regression without multiplying
    the diffuse LiH support across thirteen finite-difference SCFs.  The test
    deliberately compares one fixed truncated functional, so it bypasses the
    production fold preflight exactly like the neighbouring Fock-rebuild test;
    fold reliability is covered independently by the production guards.
    """
    import vibeqc.pbc_bipole_common as _common
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    monkeypatch.setattr(
        _common, "s_fold_truncation_drift", lambda *a, **k: 1.0e-9
    )
    lattice = 7.6 * np.eye(3)
    sysp = vq.PeriodicSystem(
        3, lattice, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 3.8])]
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _ks_opts(cutoff=8.0, conv=1e-12)
    opts.max_iter = 100
    opts.conv_tol_grad = 1e-10
    opts.functional = "svwn"
    opts.fock_mixing = 1e-12
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    assert result.n_iter <= 100


@pytest.mark.slow
def test_analytic_rks_gradient_gamma_beh2_matches_fd():
    """Gamma-local RKS includes the KS Bloch-CPHF orbital response.

    Asymmetric BeH2/SVWN supplies a non-trivial closed-shell orbital response.
    The KS Z-vector solves the non-self-adjoint orbital Hessian transpose and
    recovers the relaxed SCF finite-difference gradient.  This row previously
    used diffuse LiH, whose fold drift does not enter the supported regime
    until 18 bohr; thirteen finite-difference SCFs there exceed the lane
    budget.  The compact BeH2 fixture reaches a 5.1e-5 drift at 13 bohr.
    """
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_beh2_box(a_bohr=9.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _ks_opts(cutoff=13.0, conv=1e-12)
    opts.max_iter = 100
    opts.conv_tol_grad = 1e-10
    opts.functional = "svwn"
    opts.fock_mixing = 1e-12
    # Legacy Γ-local gauge pinned explicitly: the analytic-gradient / CPHF
    # reconstruction helpers implement the legacy assembly; the corrected
    # option-(b) gauge (default at Γ under the J split since Phase 4a/4b)
    # is refused by the analytic path.
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_rks(
        sysp,
        basis,
        result,
        lattice_opts=opts.lattice_opts,
        kmesh=kmesh,
        grid_options=opts.grid,
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RKS",
        functional="svwn",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 2e-5, f"RKS Γ analytic vs FD: max|Δ| = {max_abs:.2e}"


@pytest.mark.slow
def test_analytic_uks_gradient_gamma_beh_matches_fd():
    """Gamma-local UKS includes the coupled-spin KS orbital response.

    BeH/SVWN is an asymmetric open-shell doublet: after the fixed-density XC
    terms, the missing orbital response is ~9.7e-4 Ha/bohr.  The UKS Z-vector
    solves the coupled alpha/beta non-self-adjoint Hessian transpose and
    recovers the relaxed SCF finite-difference gradient.
    """
    from vibeqc.bipole_gradient import compute_bipole_gradient_uks
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    sysp = vq.PeriodicSystem(
        3, 7.0 * np.eye(3), [vq.Atom(4, [0, 0, 0]), vq.Atom(1, [0, 0, 2.4])]
    )
    sysp.multiplicity = 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # 11 bohr: S-fold drift 3.9e-3 (7 bohr is refused at 4.9e-2).
    opts.lattice_opts.cutoff_bohr = 11.0
    opts.lattice_opts.nuclear_cutoff_bohr = 11.0
    opts.conv_tol_grad = 1e-10
    opts.conv_tol_energy = 1e-12
    opts.initial_guess = InitialGuess.SAD
    opts.functional = "svwn"
    opts.max_iter = 400
    opts.fock_mixing = 1e-12
    opts.scf_accelerator = SCFAccelerator.DIIS
    result = run_pbc_bipole_uks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_uks(
        sysp,
        basis,
        result,
        lattice_opts=opts.lattice_opts,
        kmesh=kmesh,
        grid_options=opts.grid,
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="UKS",
        functional="svwn",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 2e-5, f"UKS Γ analytic vs FD: max|Δ| = {max_abs:.2e}"


@pytest.mark.slow
def test_analytic_rks_gradient_gamma_beh2_pbe_matches_fd():
    """Gamma-local RKS with PBE (GGA) matches FD. Extends the SVWN regression
    to a gradient-corrected functional, exercising the sigma-Pulay Hessian terms
    in the full gradient assembly."""
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_beh2_box(a_bohr=9.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    # 13 bohr: S-fold drift 5.1e-5; diffuse LiH needs at least 18 bohr.
    opts = _ks_opts(cutoff=13.0, conv=1e-12)
    opts.conv_tol_grad = 1e-10
    opts.functional = "pbe"
    opts.fock_mixing = 1e-12
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_rks(
        sysp,
        basis,
        result,
        lattice_opts=opts.lattice_opts,
        kmesh=kmesh,
        grid_options=opts.grid,
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RKS",
        functional="pbe",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 2e-4, f"RKS Γ PBE analytic vs FD: max|Δ| = {max_abs:.2e}"


@pytest.mark.slow
def test_analytic_uks_gradient_gamma_beh_pbe_matches_fd():
    """Gamma-local UKS with PBE (GGA) matches FD. Extends the SVWN regression
    to a gradient-corrected functional, exercising the per-spin sigma-Pulay
    terms and coupled KS CPHF."""
    from vibeqc.bipole_gradient import compute_bipole_gradient_uks
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    lattice = 7.0 * np.eye(3)
    sysp = vq.PeriodicSystem(
        3, lattice, [vq.Atom(4, [0, 0, 0]), vq.Atom(1, [0, 0, 2.4])]
    )
    sysp.multiplicity = 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    # 11 bohr: S-fold drift 3.9e-3 (7 bohr is refused at 4.9e-2).
    opts = _ks_opts(cutoff=11.0, conv=1e-12)
    opts.conv_tol_grad = 1e-10
    opts.functional = "pbe"
    opts.fock_mixing = 1e-12
    opts.max_iter = 100
    opts.scf_accelerator = SCFAccelerator.KDIIS
    result = run_pbc_bipole_uks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_uks(
        sysp,
        basis,
        result,
        lattice_opts=opts.lattice_opts,
        kmesh=kmesh,
        grid_options=opts.grid,
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="UKS",
        functional="pbe",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 2e-4, f"UKS Γ PBE analytic vs FD: max|Δ| = {max_abs:.2e}"


@pytest.mark.slow
def test_periodic_xc_lattice_gradient_kernel_matches_fd():
    """The periodic XC Pulay gradient kernel
    (``xc_lattice_gradient_contribution``) matches the fixed-grid finite
    difference of ``E_xc[D fixed]`` — the analytic V_xc force.

    Validates the kernel in isolation (the C++ lattice-summed XC Pulay,
    exact for LDA): build the converged RKS density + grid, then central-
    difference ``E_xc`` over displaced *bases* on the **same** grid + density
    (so only the basis-function Pulay term enters — the Becke grid-weight
    derivative, which the kernel neglects like the molecular kernel, is held
    out). Multi-cell BeH-free H₂ on the periodic Becke grid, LDA (SVWN)."""
    from vibeqc._vibeqc_core import (
        Functional,
        build_xc_periodic,
        xc_lattice_gradient_contribution,
    )
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.periodic_grid import build_periodic_becke_grid

    opts = PeriodicKSOptions()
    # 11 bohr: drift 2.6e-5 (7 bohr sat at 9.9e-3, <10% under the refusal).
    opts.lattice_opts.cutoff_bohr = 11.0
    opts.lattice_opts.nuclear_cutoff_bohr = 11.0
    opts.max_iter = 300
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.functional = "svwn"
    lattice = 6.0 * np.eye(3)
    atoms0 = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.5])]
    sysp = vq.PeriodicSystem(3, lattice, atoms0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    func = Functional("svwn", 1)
    grid = build_periodic_becke_grid(
        sysp, grid_options=opts.grid, image_radius_bohr=10.0
    )
    g_an = np.asarray(
        xc_lattice_gradient_contribution(
            basis, sysp, grid, func, result.density, opts.lattice_opts
        )
    )

    h = 1e-4
    g_fd = np.zeros((2, 3))
    for a in range(2):
        for d in range(3):
            ap = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xp = list(ap[a].xyz)
            xp[d] += h
            ap[a] = vq.Atom(ap[a].Z, xp)
            am = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xm = list(am[a].xyz)
            xm[d] -= h
            am[a] = vq.Atom(am[a].Z, xm)
            sp = vq.PeriodicSystem(3, lattice, ap)
            bp = vq.BasisSet(sp.unit_cell_molecule(), "sto-3g")
            sm = vq.PeriodicSystem(3, lattice, am)
            bm = vq.BasisSet(sm.unit_cell_molecule(), "sto-3g")
            ep = build_xc_periodic(
                bp, sp, grid, func, result.density, opts.lattice_opts
            ).e_xc
            em = build_xc_periodic(
                bm, sm, grid, func, result.density, opts.lattice_opts
            ).e_xc
            g_fd[a, d] = (ep - em) / (2.0 * h)
    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"periodic XC Pulay kernel vs fixed-grid FD: {max_abs:.2e}"


def test_periodic_uks_xc_lattice_gradient_kernel_cross_cell_matches_fd():
    """Open-shell periodic XC Pulay kernel matches fixed-grid FD on a genuine
    CROSS-CELL (periodic, P(g≠0) populated) density.

    The companion ``test_periodic_uks_xc_lattice_gradient_kernel_matches_fd``
    uses a home-only SAD density, so it never exercised the bra-image term;
    ``xc_lattice_gradient_contribution_uks`` was home-bra-only while
    ``build_xc_periodic_uks`` is cross-cell (the dense-XC P0 fix), so the kernel
    silently missed the bra-image motion for any periodic density. A converged
    triplet-H₂ UKS density (cross-cell) is the smallest discriminator."""
    from vibeqc._vibeqc_core import (
        Functional,
        build_xc_periodic_uks,
        xc_lattice_gradient_contribution_uks,
    )
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks
    from vibeqc.periodic_grid import build_periodic_becke_grid

    lattice = 6.0 * np.eye(3)
    atoms0 = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.6])]
    sysp = vq.PeriodicSystem(3, lattice, atoms0)
    sysp.multiplicity = 3
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # The cross-cell density needs real image support; 12 bohr keeps that
    # support and converges the measured overlap fold to 5.5e-6.
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 300
    opts.use_diis = True
    opts.conv_tol_energy = 1e-11
    opts.initial_guess = InitialGuess.SAD
    opts.functional = "svwn"
    res = run_pbc_bipole_uks(
        sysp, basis, kmesh, opts,
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
    )
    assert res.converged
    Pa, Pb = res.density_alpha, res.density_beta
    func = Functional("svwn", 2)
    grid = build_periodic_becke_grid(
        sysp, grid_options=opts.grid, image_radius_bohr=10.0
    )
    g_an = np.asarray(xc_lattice_gradient_contribution_uks(
        basis, sysp, grid, func, Pa, Pb, opts.lattice_opts
    ))
    h = 1e-4
    g_fd = np.zeros((2, 3))
    for a in range(2):
        for d in range(3):
            ap = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xp = list(ap[a].xyz); xp[d] += h; ap[a] = vq.Atom(ap[a].Z, xp)
            am = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xm = list(am[a].xyz); xm[d] -= h; am[a] = vq.Atom(am[a].Z, xm)
            sp = vq.PeriodicSystem(3, lattice, ap); sp.multiplicity = 3
            bp = vq.BasisSet(sp.unit_cell_molecule(), "sto-3g")
            sm = vq.PeriodicSystem(3, lattice, am); sm.multiplicity = 3
            bm = vq.BasisSet(sm.unit_cell_molecule(), "sto-3g")
            ep = build_xc_periodic_uks(bp, sp, grid, func, Pa, Pb, opts.lattice_opts).e_xc
            em = build_xc_periodic_uks(bm, sm, grid, func, Pa, Pb, opts.lattice_opts).e_xc
            g_fd[a, d] = (ep - em) / (2.0 * h)
    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-6, f"UKS periodic XC Pulay kernel (cross-cell) vs FD: {max_abs:.2e}"


@pytest.mark.slow
def test_periodic_xc_grid_motion_correction_matches_moving_fd():
    """The KS analytic XC force includes atom-centred grid motion.

    ``xc_lattice_gradient_contribution`` is the fixed-grid AO Pulay term.
    The BIPOLE KS energy rebuilds the periodic Becke grid after displacement,
    so ``_periodic_xc_grid_motion_correction`` must add the missing
    fixed-density derivative of the moving grid points/weights. LiH is
    intentionally asymmetric so this correction is not hidden by symmetry.
    """
    from vibeqc._vibeqc_core import (
        Functional,
        build_xc_periodic,
        xc_lattice_gradient_contribution,
    )
    from vibeqc.bipole_gradient import _periodic_xc_grid_motion_correction
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.periodic_grid import build_periodic_becke_grid

    # 24 bohr: the diffuse Li 2sp tail keeps the LiH fold drift above 1e-4
    # until 24 bohr (3.8e-5; 8 bohr refused at 4.4e-1).
    opts = _ks_opts(cutoff=24.0)
    opts.functional = "svwn"
    lattice = 7.6 * np.eye(3)
    atoms0 = [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 3.8])]
    sysp = vq.PeriodicSystem(3, lattice, atoms0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged

    func = Functional("svwn", 1)
    grid = build_periodic_becke_grid(
        sysp, grid_options=opts.grid, image_radius_bohr=10.0
    )
    fixed = np.asarray(
        xc_lattice_gradient_contribution(
            basis, sysp, grid, func, result.density, opts.lattice_opts
        )
    )
    corr = _periodic_xc_grid_motion_correction(
        sysp,
        basis,
        result.density,
        "svwn",
        opts.lattice_opts,
        opts.grid,
        True,
        10.0,
        fixed,
        step_bohr=1e-3,
    )
    g_an = fixed + corr

    h = 1e-3
    g_fd = np.zeros((2, 3))
    for a in range(2):
        for d in range(3):
            energies = []
            for sign in (+1.0, -1.0):
                aa = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
                xyz = list(aa[a].xyz)
                xyz[d] += sign * h
                aa[a] = vq.Atom(aa[a].Z, xyz)
                sp = vq.PeriodicSystem(3, lattice, aa)
                bp = vq.BasisSet(sp.unit_cell_molecule(), "sto-3g")
                gp = build_periodic_becke_grid(
                    sp, grid_options=opts.grid, image_radius_bohr=10.0
                )
                energies.append(
                    build_xc_periodic(
                        bp,
                        sp,
                        gp,
                        func,
                        result.density,
                        opts.lattice_opts,
                    ).e_xc
                )
            g_fd[a, d] = (float(energies[0]) - float(energies[1])) / (2.0 * h)

    # Omitting grid motion must fail the accuracy requirement decisively.
    # The point-centered partition reduces the correction from the old
    # 8.3e-4 scale to about 1.27e-4 Ha/bohr at this converged support.
    accuracy_tol = 1e-9
    uncorrected_error = float(np.max(np.abs(fixed - g_fd)))
    assert uncorrected_error > 1000 * accuracy_tol
    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < accuracy_tol, f"moving-grid XC correction vs FD: {max_abs:.2e}"


def test_periodic_xc_lattice_gga_gradient_kernel_matches_fd():
    """Closed-shell periodic GGA XC Pulay kernel matches fixed-grid FD.

    This uses a fixed SAD density, not an SCF density, so the test isolates
    the sigma-Pulay Hessian terms in ``xc_lattice_gradient_contribution``.
    """
    from vibeqc._vibeqc_core import (
        Functional,
        GridOptions,
        build_xc_periodic,
        compute_overlap_lattice,
        xc_lattice_gradient_contribution,
    )
    from vibeqc.guess import initial_density_closed_shell
    from vibeqc.periodic_grid import build_periodic_becke_grid

    lattice = 8.0 * np.eye(3)
    atoms0 = [
        vq.Atom(1, [0.1, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.2, 1.5]),
    ]
    sysp = vq.PeriodicSystem(3, lattice, atoms0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 5.0
    S_lat = compute_overlap_lattice(basis, sysp, lat_opts)
    D = initial_density_closed_shell(
        sysp.unit_cell_molecule(),
        basis,
        1,
        InitialGuess.SAD,
        is_periodic=True,
    )
    P = compute_overlap_lattice(basis, sysp, lat_opts)
    for g_idx, cell in enumerate(S_lat.cells):
        is_g0 = (np.asarray(cell.index) == np.array([0, 0, 0])).all()
        P.set_block(g_idx, D if is_g0 else np.zeros_like(D))

    grid_opts = GridOptions()
    grid_opts.n_radial = 20
    grid_opts.angular = "lebedev"
    grid_opts.lebedev_order = 17
    grid = build_periodic_becke_grid(
        sysp, grid_options=grid_opts, image_radius_bohr=8.0
    )
    func = Functional("pbe", 1)
    g_an = np.asarray(
        xc_lattice_gradient_contribution(basis, sysp, grid, func, P, lat_opts)
    )

    h = 1e-4
    g_fd = np.zeros((2, 3))
    for a_idx in range(2):
        for d in range(3):
            ap = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xp = list(ap[a_idx].xyz)
            xp[d] += h
            ap[a_idx] = vq.Atom(ap[a_idx].Z, xp)
            am = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xm = list(am[a_idx].xyz)
            xm[d] -= h
            am[a_idx] = vq.Atom(am[a_idx].Z, xm)
            sp = vq.PeriodicSystem(3, lattice, ap)
            bp = vq.BasisSet(sp.unit_cell_molecule(), "sto-3g")
            sm = vq.PeriodicSystem(3, lattice, am)
            bm = vq.BasisSet(sm.unit_cell_molecule(), "sto-3g")
            ep = build_xc_periodic(bp, sp, grid, func, P, lat_opts).e_xc
            em = build_xc_periodic(bm, sm, grid, func, P, lat_opts).e_xc
            g_fd[a_idx, d] = (ep - em) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 2e-6, (
        f"periodic GGA XC Pulay kernel vs fixed-grid FD: {max_abs:.2e}"
    )


@pytest.mark.parametrize("functional,tol", [("svwn", 1e-6), ("pbe", 2e-6)])
def test_periodic_uks_xc_lattice_gradient_kernel_matches_fd(functional, tol):
    """Open-shell periodic XC Pulay kernel matches fixed-grid FD of E_xc.

    Uses a tiny H3 doublet with fixed SAD alpha/beta densities. No SCF is
    involved; the test isolates the native spin-polarized lattice-XC Pulay
    term that ``compute_bipole_gradient_uks`` adds on the Gamma-local path.
    The PBE case exercises the spin-polarized GGA sigma-Pulay terms.
    """
    from vibeqc._vibeqc_core import (
        Functional,
        GridOptions,
        build_xc_periodic_uks,
        compute_overlap_lattice,
        xc_lattice_gradient_contribution_uks,
    )
    from vibeqc.guess import initial_densities_open_shell
    from vibeqc.periodic_grid import build_periodic_becke_grid

    lattice = 8.0 * np.eye(3)
    atoms0 = [
        vq.Atom(1, [0.2, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.1, 1.4]),
        vq.Atom(1, [0.0, 0.0, 2.8]),
    ]
    sysp = vq.PeriodicSystem(3, lattice, atoms0, charge=0, multiplicity=2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 5.0
    S_lat = compute_overlap_lattice(basis, sysp, lat_opts)
    Da, Db = initial_densities_open_shell(
        sysp.unit_cell_molecule(),
        basis,
        2,
        1,
        InitialGuess.SAD,
        is_periodic=True,
    )
    P_alpha = compute_overlap_lattice(basis, sysp, lat_opts)
    P_beta = compute_overlap_lattice(basis, sysp, lat_opts)
    for g_idx, cell in enumerate(S_lat.cells):
        is_g0 = (np.asarray(cell.index) == np.array([0, 0, 0])).all()
        P_alpha.set_block(g_idx, Da if is_g0 else np.zeros_like(Da))
        P_beta.set_block(g_idx, Db if is_g0 else np.zeros_like(Db))

    grid_opts = GridOptions()
    grid_opts.n_radial = 20
    grid_opts.angular = "lebedev"
    grid_opts.lebedev_order = 17
    grid = build_periodic_becke_grid(
        sysp, grid_options=grid_opts, image_radius_bohr=8.0
    )
    func = Functional(functional, 2)
    g_an = np.asarray(
        xc_lattice_gradient_contribution_uks(
            basis, sysp, grid, func, P_alpha, P_beta, lat_opts
        )
    )

    h = 1e-4
    g_fd = np.zeros((3, 3))
    for a_idx in range(3):
        for d in range(3):
            ap = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xp = list(ap[a_idx].xyz)
            xp[d] += h
            ap[a_idx] = vq.Atom(ap[a_idx].Z, xp)
            am = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
            xm = list(am[a_idx].xyz)
            xm[d] -= h
            am[a_idx] = vq.Atom(am[a_idx].Z, xm)
            sp = vq.PeriodicSystem(3, lattice, ap, charge=0, multiplicity=2)
            bp = vq.BasisSet(sp.unit_cell_molecule(), "sto-3g")
            sm = vq.PeriodicSystem(3, lattice, am, charge=0, multiplicity=2)
            bm = vq.BasisSet(sm.unit_cell_molecule(), "sto-3g")
            ep = build_xc_periodic_uks(
                bp, sp, grid, func, P_alpha, P_beta, lat_opts
            ).e_xc
            em = build_xc_periodic_uks(
                bm, sm, grid, func, P_alpha, P_beta, lat_opts
            ).e_xc
            g_fd[a_idx, d] = (ep - em) / (2.0 * h)

    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < tol, (
        f"periodic UKS XC Pulay kernel vs fixed-grid FD: {max_abs:.2e}"
    )


def test_periodic_uks_xc_grid_motion_correction_matches_moving_fd():
    """Spin-polarized XC grid-motion correction matches moving-grid FD."""
    from vibeqc._vibeqc_core import (
        Functional,
        GridOptions,
        build_xc_periodic_uks,
        compute_overlap_lattice,
        xc_lattice_gradient_contribution_uks,
    )
    from vibeqc.bipole_gradient import _periodic_xc_grid_motion_correction_uks
    from vibeqc.guess import initial_densities_open_shell
    from vibeqc.periodic_grid import build_periodic_becke_grid

    lattice = 8.0 * np.eye(3)
    atoms0 = [
        vq.Atom(1, [0.2, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.1, 1.4]),
        vq.Atom(1, [0.0, 0.0, 2.8]),
    ]
    sysp = vq.PeriodicSystem(3, lattice, atoms0, charge=0, multiplicity=2)
    loaded_basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    basis = vq.BasisSet(
        sysp.unit_cell_molecule(),
        loaded_basis.shells(),
        "<in-memory-grid-motion>",
        True,
    )
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 5.0
    S_lat = compute_overlap_lattice(basis, sysp, lat_opts)
    Da, Db = initial_densities_open_shell(
        sysp.unit_cell_molecule(),
        loaded_basis,
        2,
        1,
        InitialGuess.SAD,
        is_periodic=True,
    )
    P_alpha = compute_overlap_lattice(basis, sysp, lat_opts)
    P_beta = compute_overlap_lattice(basis, sysp, lat_opts)
    for g_idx, cell in enumerate(S_lat.cells):
        is_g0 = (np.asarray(cell.index) == np.array([0, 0, 0])).all()
        P_alpha.set_block(g_idx, Da if is_g0 else np.zeros_like(Da))
        P_beta.set_block(g_idx, Db if is_g0 else np.zeros_like(Db))

    grid_opts = GridOptions()
    grid_opts.n_radial = 20
    grid_opts.angular = "lebedev"
    grid_opts.lebedev_order = 17
    grid = build_periodic_becke_grid(
        sysp, grid_options=grid_opts, image_radius_bohr=8.0
    )
    func = Functional("svwn", 2)
    fixed = np.asarray(
        xc_lattice_gradient_contribution_uks(
            basis, sysp, grid, func, P_alpha, P_beta, lat_opts
        )
    )
    corr = _periodic_xc_grid_motion_correction_uks(
        sysp,
        basis,
        P_alpha,
        P_beta,
        "svwn",
        lat_opts,
        grid_opts,
        True,
        8.0,
        fixed,
        step_bohr=1e-3,
    )
    g_an = fixed + corr

    h = 1e-3
    g_fd = np.zeros((3, 3))
    for a_idx in range(3):
        for d in range(3):
            energies = []
            for sign in (+1.0, -1.0):
                ap = [vq.Atom(x.Z, list(x.xyz)) for x in atoms0]
                xyz = list(ap[a_idx].xyz)
                xyz[d] += sign * h
                ap[a_idx] = vq.Atom(ap[a_idx].Z, xyz)
                sp = vq.PeriodicSystem(3, lattice, ap, charge=0, multiplicity=2)
                bp = vq.BasisSet(sp.unit_cell_molecule(), "sto-3g")
                gp = build_periodic_becke_grid(
                    sp, grid_options=grid_opts, image_radius_bohr=8.0
                )
                energies.append(
                    build_xc_periodic_uks(
                        bp,
                        sp,
                        gp,
                        func,
                        P_alpha,
                        P_beta,
                        lat_opts,
                    ).e_xc
                )
            g_fd[a_idx, d] = (float(energies[0]) - float(energies[1])) / (2.0 * h)

    assert float(np.max(np.abs(corr))) > 1e-5
    max_abs = float(np.max(np.abs(g_an - g_fd)))
    assert max_abs < 1e-9, f"UKS moving-grid XC correction vs FD: {max_abs:.2e}"


@pytest.mark.slow
def test_analytic_rks_gradient_multi_k_svwn_matches_fd():
    """RKS multi-k SVWN analytic gradient with diagonal-Z KS CPHF
    matches finite difference on H2/STO-3G [2,1,1]."""
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_h2_box(a_bohr=5.0, bond_bohr=1.45)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [2, 1, 1])
    # 9 bohr: S(k)-fold drift 9.3e-4 (7 bohr is refused at 6.2e-2).
    opts = _ks_opts(cutoff=9.0, conv=1e-12)
    opts.functional = "svwn"
    opts.fock_mixing = 1e-12
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,  # analytic gradient is legacy-gauge only; pin the SCF to legacy too
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_rks(
        sysp,
        basis,
        result,
        lattice_opts=opts.lattice_opts,
        kmesh=kmesh,
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RKS",
        functional="svwn",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 5e-4, f"RKS multi-k SVWN analytic vs FD: max|Δ| = {max_abs:.2e}"


@pytest.mark.slow
def test_analytic_rks_gradient_gamma_beh2_b3lyp_matches_fd():
    """Gamma-local RKS with B3LYP hybrid (20% HF exchange) matches FD.
    Exercises the alpha_hf > 0 path through the KS CPHF."""
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_beh2_box(a_bohr=9.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    # 13 bohr: S-fold drift 5.1e-5; diffuse LiH needs at least 18 bohr.
    opts = _ks_opts(cutoff=13.0, conv=1e-12)
    opts.conv_tol_grad = 1e-10
    opts.functional = "b3lyp"
    opts.fock_mixing = 1e-12
    result = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged
    grad_analytic = compute_bipole_gradient_rks(
        sysp,
        basis,
        result,
        lattice_opts=opts.lattice_opts,
        kmesh=kmesh,
        grid_options=opts.grid,
    )
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RKS",
        functional="b3lyp",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,
        ewald_precision=1e-8,
    )
    max_abs = float(np.max(np.abs(grad_analytic - grad_fd)))
    assert max_abs < 2e-4, f"RKS Gamma B3LYP analytic vs FD: max|Δ| = {max_abs:.2e}"


def test_k_long_range_ewald_gradient_matches_fd():
    """Reciprocal long-range EXCHANGE gradient kernel (corrected gauge).

    ``_k_long_range_ewald_gradient`` differentiates ``E = -1/4 Tr[D K^LR(D)]``
    at fixed density, where ``K^LR`` sandwiches the density between two
    AO-pair FTs (the exchange analogue of ``J^LR``). Pins the 2026-06-15
    derivation used by the corrected (Ewald-exchange-split) analytic
    gradient; the phaseless/legacy path would miss this term entirely.
    """
    from vibeqc.bipole_gradient import _k_long_range_ewald_gradient
    from vibeqc.bipole_fock_ewald import (
        _build_j_long_range_cache,
        compute_K_long_range_gamma,
    )
    from vibeqc.bipole_ext_el_pole import (
        crystal_default_ewald_alpha,
        crystal_ewald_reciprocal_cutoff,
    )
    from vibeqc.guess import initial_density_closed_shell

    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    cutoff = 6.0

    def build(disp=None):
        pos = [np.zeros(3), np.array([a / 2.0, a / 2.0, a / 2.0])]
        if disp is not None:
            i, x, d = disp
            pos[i] = pos[i].copy()
            pos[i][x] += d
        sysp = vq.PeriodicSystem(
            3, lattice,
            [vq.Atom(12, list(pos[0])), vq.Atom(8, list(pos[1]))],
        )
        return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    sysp0, basis0 = build()
    n_occ = sysp0.n_electrons() // 2
    D_home = np.asarray(
        initial_density_closed_shell(
            sysp0.unit_cell_molecule(), basis0, n_occ,
            InitialGuess.SAD, is_periodic=True,
        ),
        dtype=float,
    )
    V = float(abs(np.linalg.det(lattice)))
    omega = crystal_default_ewald_alpha(V)
    K_max = crystal_ewald_reciprocal_cutoff(V)

    def homog(sysp, basis):
        lat = LatticeSumOptions()
        lat.cutoff_bohr = cutoff
        T = compute_overlap_lattice(basis, sysp, lat)
        for c in range(len(T.cells)):
            T.set_block(c, D_home)
        return T

    def klr_energy(sysp, basis):
        T = homog(sysp, basis)
        cells = np.array([np.asarray(c.r_cart, float) for c in T.cells])
        cache = _build_j_long_range_cache(basis, sysp, cells, omega, 1e-8, K_max=K_max)
        return -0.25 * float(
            np.einsum("ij,ji->", D_home, compute_K_long_range_gamma(cache, D_home))
        )

    g_an = _k_long_range_ewald_gradient(sysp0, basis0, homog(sysp0, basis0), omega)
    h = 1e-4
    g_fd = np.zeros_like(g_an)
    for i in range(2):
        for x in range(3):
            sp, bp = build((i, x, +h))
            sm, bm = build((i, x, -h))
            g_fd[i, x] = (klr_energy(sp, bp) - klr_energy(sm, bm)) / (2.0 * h)
    np.testing.assert_allclose(g_an, g_fd, atol=1e-6)


@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "Green in the 2026-08-06 post-fix run but 19315 s wall (3 MgO/STO-3G SCFs at the "
        "fold-reliable 15 bohr, ~1.7 h each) — over the nightly slow "
        "lane's 1800 s per-test timeout. Skipped pending the "
        "BIPOLE-FIXTURE-CUTOFFS-TRIP-FOLD-GATE slow-lane budget decision "
        "(agentic-loop/bug-claims.md ask to the release/test-health "
        "chat); standalone-green evidence in agentic-loop/runs/"
        "2026-08-05-bipole-fold-fixture-cutoffs/EVIDENCE.md. Un-skip "
        "when the budget lands (nightly accept or weekly tier)."
    )
)
def test_corrected_gauge_rhf_gamma_gradient_matches_fd():
    """Corrected (Ewald-exchange-split) gauge RHF Γ analytic gradient.

    Un-refused 2026-06-15. The corrected gauge is a *standard* variational
    HF gradient (full Bloch density → no Bloch-CPHF) — the legacy assembly
    with the exchange block swapped (K_SR(erfc)+K_LR(recip)+Madelung·SDS,
    no spheropole), the homogeneous-on-gradient-template density, and the
    FULL jellium gradient (v_bg ∝ Tr[D S] is quadratic; the W from the
    variational eigenvalues supplies the cancelling overlap term). Must
    match the production FD path (which differentiates the actual
    corrected-gauge driver energy).

    Restructured 2026-08-05 (fold-support fixture repair): the reliable
    MgO fold cutoff is 15 bohr (S(Γ) drift 2.4e-6; 5 bohr refused at
    9.8e-1 — same support ladder as
    agentic-loop/runs/2026-08-01-bipole-invfold-regression/EVIDENCE.md),
    and one MgO/STO-3G SCF at 15 bohr costs ~1-2 h wall. The full-tensor
    ``compute_bipole_gradient_fd`` comparison (12 displaced SCFs) is
    therefore replaced by a single-coordinate central difference on a
    symmetry-broken geometry: the O atom sits 0.15 bohr off its ideal
    rocksalt site along z, so the compared dE/dz force is genuinely
    nonzero (the ideal site gives identically zero forces by symmetry),
    and only that coordinate is differenced — 3 SCFs total. The
    full-tensor FD parity for the corrected gauge stays covered on the
    cheap H2/BeH2/BeH fixtures above; this row keeps the ionic
    (Madelung/jellium-scale) coverage.
    """
    def mgo_o_displaced(dz):
        a = 4.21 * ANG2BOHR
        lattice = (a / 2.0) * np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
        atoms = [
            vq.Atom(12, [0.0, 0.0, 0.0]),
            vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0 + dz]),
        ]
        return vq.PeriodicSystem(3, lattice, atoms)

    dz0 = 0.15
    sysp = mgo_o_displaced(dz0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _rhf_opts(cutoff=15.0, conv=1e-10)
    run_kwargs = dict(
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
    )
    result = run_pbc_bipole_rhf(sysp, basis, kmesh, opts, **run_kwargs)
    assert result.converged
    assert getattr(result, "exchange_ewald_split", False)
    g_an = np.asarray(compute_bipole_gradient_rhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    ))
    # Analytic translational invariance (no extra SCF).
    assert float(np.max(np.abs(g_an.sum(axis=0)))) < 1e-5

    # Single-coordinate central difference on the O atom's z coordinate.
    step = 1e-3
    energies = []
    for sign in (+1.0, -1.0):
        disp = mgo_o_displaced(dz0 + sign * step)
        disp_basis = vq.BasisSet(disp.unit_cell_molecule(), "sto-3g")
        disp_res = run_pbc_bipole_rhf(
            disp, disp_basis, monkhorst_pack(disp, [1, 1, 1]), opts,
            **run_kwargs,
        )
        assert disp_res.converged
        energies.append(disp_res.energy)
    g_fd_oz = (energies[0] - energies[1]) / (2.0 * step)
    # The off-site force must be genuinely nonzero for parity to mean much.
    assert abs(g_fd_oz) > 1e-3
    np.testing.assert_allclose(g_an[1, 2], g_fd_oz, atol=1e-5)


def test_corrected_gauge_rhf_gamma_large_box_bounded_alpha_matches_fd():
    """GitLab #674 at Γ: H2/STO-3G in a 12-bohr box at a 6-bohr cutoff.

    CRYSTAL's alpha there is 0.233 and the erfc bound of the 6-bohr exchange
    cutoff 0.715, so the corrected-split default takes the bound and the
    J_LR / K_LR envelope scales with it. The Γ corrected analytic gradient
    resolves that envelope at every one of its sites (jellium, V_ne, J_LR,
    K_LR, the Fock reconstruction of the CPHF right-hand side), so the
    full-tensor central difference of the driver energy is the
    energy/gradient consistency check of the bounded-alpha contract on a
    cell wider than the 7.8-bohr threshold.
    """
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
    from vibeqc.pbc_bipole_common import ewald_alpha_lower_bound

    lattice = 12.0 * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, -0.62]), vq.Atom(1, [0.0, 0.0, 0.78])]
    sysp = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _rhf_opts(cutoff=6.0, conv=1e-10)
    run_kwargs = {"use_ewald_j_split": True, "ewald_precision": 1e-8}
    result = run_pbc_bipole_rhf(sysp, basis, kmesh, opts, progress=False, **run_kwargs)
    assert result.converged
    assert getattr(result, "exchange_ewald_split", False)
    assert result.ewald_alpha_bohr_inv > crystal_default_ewald_alpha(12.0**3)
    assert result.ewald_alpha_bohr_inv == pytest.approx(
        ewald_alpha_lower_bound(6.0, 1e-8), rel=1e-9
    )
    g_an = np.asarray(compute_bipole_gradient_rhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    ))
    assert float(np.max(np.abs(g_an.sum(axis=0)))) < 1e-5
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="RHF", step_bohr=1e-3, **run_kwargs
    ))
    # The off-centre H2 carries a genuinely nonzero bond force.
    assert float(np.max(np.abs(g_fd))) > 1e-3
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5, err_msg=(
        "Γ corrected-gauge gradient with the #674 bounded alpha diverges from FD"
    ))


@pytest.mark.slow
@pytest.mark.parametrize("pair_resolved", [False, True])
def test_corrected_gauge_rhf_gamma_padded_domain_matches_fd(pair_resolved):
    """RHF Gamma differentiates the production M5 padded erfc domain.

    The radial case isolates the padded internal ket-image traversal. The
    SYM3b case also pins the pair-resolved output cells and shell-pair masks.
    """
    sysp = _build_h2_box(a_bohr=7.0, bond_bohr=1.4)
    if pair_resolved:
        vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    # 10 bohr: drift 1.5e-5 (5 bohr refused at 3.3e-2).
    opts = _rhf_opts(cutoff=10.0, conv=1e-10)
    m5_kwargs = {
        "use_ewald_j_split": True,
        "ewald_precision": 1e-8,
        "sr_image_precision": 1e-6,
        "use_fock_symmetry_reduce": None if pair_resolved else False,
    }
    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        progress=False,
        **m5_kwargs,
    )
    assert result.converged
    assert result.exchange_ewald_split
    assert result.sr_image_extent_bohr > opts.lattice_opts.cutoff_bohr
    assert opts.lattice_opts.sr_range_screening
    assert result.pair_resolved_fock_domain is pair_resolved

    with pytest.warns(UserWarning, match="maintained preview"):
        g_an = compute_bipole_gradient_rhf(
            sysp, basis, result, lattice_opts=opts.lattice_opts
        )
    g_fd = np.asarray(
        compute_bipole_gradient_fd(
            sysp,
            "sto-3g",
            kmesh,
            opts,
            method="RHF",
            step_bohr=1e-3,
            **m5_kwargs,
        )
    )
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)


@pytest.mark.slow
def test_corrected_gauge_uhf_gamma_pair_domain_matches_fd():
    """UHF Gamma differentiates the attached-symmetry M5 pair domain.

    The FD helper must re-detect symmetry at each displaced geometry. Without
    that propagation every displaced SCF silently fell back to the radial
    domain, leaving a spurious 1.5e-4 Ha/bohr analytic-vs-FD residual.
    """
    a = 12.0
    sysp = vq.PeriodicSystem(
        3,
        a * np.eye(3),
        [vq.Atom(4, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 2.5])],
        multiplicity=2,
    )
    vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = _rhf_opts(cutoff=5.0, conv=1e-10)
    opts.max_iter = 400
    m5_kwargs = {
        "use_ewald_j_split": True,
        "ewald_precision": 1e-8,
        "sr_image_precision": 1e-6,
        "use_fock_symmetry_reduce": True,
    }
    result = run_pbc_bipole_uhf(
        sysp,
        basis,
        kmesh,
        opts,
        progress=False,
        **m5_kwargs,
    )
    assert result.converged
    assert result.exchange_ewald_split
    assert result.pair_resolved_fock_domain

    with pytest.warns(UserWarning, match="maintained preview"):
        g_an = compute_bipole_gradient_uhf(
            sysp, basis, result, lattice_opts=opts.lattice_opts
        )
    g_fd = np.asarray(
        compute_bipole_gradient_fd(
            sysp,
            "sto-3g",
            kmesh,
            opts,
            method="UHF",
            step_bohr=1e-3,
            **m5_kwargs,
        )
    )
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)



@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "pbe0"])
def test_corrected_gauge_rks_gamma_gradient_matches_fd(functional):
    """RKS Γ corrected-gauge analytic gradient (un-refused 2026-06-15).

    The corrected-gauge core (no Bloch-CPHF; exchange scaled by the
    functional's HF fraction — 0 for pure DFT) plus the same fixed-grid
    XC Pulay + grid-motion correction the legacy path uses, with W from
    the KS variational eigenvalues. Pins the padded radial M5 SR+LR domain
    for both pure DFT (LDA) and a hybrid (PBE0, α_HF=0.25). The LDA row
    is the production-surface replacement for the retired exact-FT J
    comparison. Multi-k corrected gauge still refuses."""
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks

    sysp = _build_h2_box(a_bohr=6.0, bond_bohr=1.6)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # 11 bohr: drift 3.1e-5 (5 bohr refused at 1.1e-1).
    opts.lattice_opts.cutoff_bohr = 11.0
    opts.lattice_opts.nuclear_cutoff_bohr = 11.0
    opts.max_iter = 200
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.functional = functional
    m5_kwargs = {
        "sr_image_precision": 1e-6,
        "use_fock_symmetry_reduce": False,
    }
    result = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts,
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
        **m5_kwargs,
    )
    assert result.converged and getattr(result, "exchange_ewald_split", False)
    assert result.sr_image_extent_bohr > opts.lattice_opts.cutoff_bohr
    assert result.pair_resolved_fock_domain is False
    g_an = compute_bipole_gradient_rks(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    )
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="RKS", functional=functional,
        step_bohr=1e-3, use_ewald_j_split=True, ewald_precision=1e-8,
        **m5_kwargs,
    ))
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "pbe0"])
def test_corrected_gauge_rks_gamma_pair_domain_matches_fd(functional):
    """RKS Gamma differentiates the complete SYM3b pair Fock domain.

    Pair masks apply only to the direct SR tensor density.  Reciprocal J/K
    use the pair output template, while the neutralising background remains
    ``v_bg.S(k)`` on the SCF overlap template.  Mixing those domains breaks
    the occupied-overlap adjoint even though the SCF remains stationary.
    """
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks

    sysp = _build_h2_box(a_bohr=6.0, bond_bohr=1.6)
    vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # 11 bohr: drift 3.1e-5 (5 bohr refused at 1.1e-1).
    opts.lattice_opts.cutoff_bohr = 11.0
    opts.lattice_opts.nuclear_cutoff_bohr = 11.0
    opts.max_iter = 200
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.functional = functional
    m5_kwargs = {
        "use_ewald_j_split": True,
        "ewald_precision": 1e-8,
        "sr_image_precision": 1e-6,
        "use_fock_symmetry_reduce": None,
    }
    result = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts, progress=False, **m5_kwargs
    )
    assert result.converged
    assert result.exchange_ewald_split
    assert result.pair_resolved_fock_domain

    with pytest.warns(UserWarning, match="maintained preview"):
        g_an = compute_bipole_gradient_rks(
            sysp, basis, result, lattice_opts=opts.lattice_opts
        )
    g_fd = np.asarray(
        compute_bipole_gradient_fd(
            sysp,
            "sto-3g",
            kmesh,
            opts,
            method="RKS",
            functional=functional,
            step_bohr=1e-3,
            **m5_kwargs,
        )
    )
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["lda", "pbe0"])
@pytest.mark.parametrize("pair_resolved", [False, True])
def test_corrected_gauge_uks_gamma_gradient_matches_fd(
    functional, pair_resolved
):
    """UKS Γ corrected-gauge analytic gradient (un-refused 2026-06-15).

    Open-shell KS: the spin-resolved exchange core + the per-spin XC
    Pulay. Triplet H2 (rho_beta=0) exercises the spin-polarised XC and fully
    occupies the alpha AO space, pinning the padded finite-domain J adjoint.
    The pair row also pins the output and density shell-pair projectors.
    Must match the production FD path; multi-k corrected gauge still refuses.
    """
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks
    from vibeqc.bipole_gradient import compute_bipole_gradient_uks

    # The pair row stays off the hard 2*cutoff density-domain boundary.
    a_bohr = 6.2 if pair_resolved else 6.0
    sysp = _build_h2_box(a_bohr=a_bohr, bond_bohr=2.0)
    sysp.multiplicity = 3
    if pair_resolved:
        vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # 11.5 bohr radial row / 11 bohr pair row (drift 5.6e-5 / 2.7e-5;
    # 5 bohr refused at 1.3e-1 / 1.1e-1). The half-integer radial cutoff
    # keeps the 2*cutoff density-domain boundary (23 bohr) 1 bohr clear of
    # the a=6/bond=2 translated pairs at 22 and 24 bohr; the pair row's
    # a=6.2 lattice clears 22 by 0.8 bohr on its own.
    _cut = 11.0 if pair_resolved else 11.5
    opts.lattice_opts.cutoff_bohr = _cut
    opts.lattice_opts.nuclear_cutoff_bohr = _cut
    opts.max_iter = 300
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.functional = functional
    opts.lattice_opts.sr_range_screening = True
    m5_kwargs = {
        "use_ewald_j_split": True,
        "ewald_precision": 1e-8,
        "sr_image_precision": 1e-6,
        "use_fock_symmetry_reduce": pair_resolved,
    }
    # Keep this derivative test on its original fixed radial support;
    # the bounded Ewald alpha may select a shorter automatic extent.
    if not pair_resolved:
        m5_kwargs["sr_image_extent_bohr"] = 28.5690327065
    result = run_pbc_bipole_uks(
        sysp, basis, kmesh, opts, progress=False, **m5_kwargs
    )
    assert result.converged and getattr(result, "exchange_ewald_split", False)
    assert result.sr_image_extent_bohr > opts.lattice_opts.cutoff_bohr
    if not pair_resolved:
        # Both analytic and displaced SCFs use this fixed domain.
        assert result.sr_image_extent_bohr == pytest.approx(28.5690327065)
    assert result.pair_resolved_fock_domain is pair_resolved
    g_an = compute_bipole_gradient_uks(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    )
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="UKS", functional=functional,
        step_bohr=1e-3, **m5_kwargs,
    ))
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)



# ---------------------------------------------------------------------------
# Meta-GGA (TPSS / M06-L) periodic XC: von Weizsacker tau floor
# ---------------------------------------------------------------------------
# Regression for the cross-cell meta-GGA V_xc blow-up. The periodic cross-cell
# tau assembly (sum over image cells, signed density blocks) can produce a
# kinetic energy density tau slightly BELOW its physical von Weizsacker bound
# tau_W = |grad rho|^2 / (8 rho) -- even negative -- from roundoff at diffuse
# grid points. Unregularised meta-GGAs (TPSS, M06-L) are SINGULAR there: libxc
# returns v_sigma / v_tau ~ 1e16, which (with a CORRECT energy) contaminated the
# V_xc Fock's virtual subspace (eigenvalues ~1e8) and drove the TPSS SCF to a
# spurious +0.3 Ha stationary point. SCAN/r2SCAN self-regularise. The fix clamps
# tau >= tau_W in build_xc_periodic{,_uks} and the gradient kernels (a no-op
# where the physical bound already holds), so the energy, SCF Fock, and analytic
# gradient stay synchronised. See cpp/src/periodic_xc.cpp and
# handovers/HANDOVER_BIPOLE_GRADIENT.md (Case 2).
@pytest.mark.parametrize("functional", ["tpss", "m06-l", "scan"])
def test_periodic_mgga_vxc_bounded_on_cross_cell_density(functional):
    """build_xc_periodic V_xc is bounded for meta-GGAs on a cross-cell density.

    A correct E_xc with a ~1e7 V_xc Fock is the signature of the tau < tau_W
    libxc singularity; with the floor the V_xc is on the same O(1) scale as the
    LDA/GGA/SCAN V_xc (which never hit the bug).
    """
    from vibeqc._vibeqc_core import (
        Functional,
        PeriodicKSOptions,
        bloch_sum,
        build_xc_periodic,
    )
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.periodic_grid import build_periodic_becke_grid

    sysp = _build_h2_box(a_bohr=8.0, bond_bohr=1.4)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # 12 bohr: drift 4.2e-7 (6 bohr sat at 9.5e-3, <10% under the refusal).
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 200
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.functional = "pbe"
    # A genuine cross-cell Bloch density (many image blocks populated).
    res = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts,
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
    )
    assert res.converged
    grid = build_periodic_becke_grid(sysp, image_radius_bohr=10.0)
    xc = build_xc_periodic(
        basis, sysp, grid, Functional(functional, 1), res.density,
        opts.lattice_opts,
    )
    V = np.real(bloch_sum(xc.V_xc, np.zeros(3)))
    # Pre-fix TPSS/M06-L reached ~1e7-1e8 here; physical V_xc is O(1).
    assert np.max(np.abs(V)) < 5.0, (
        f"{functional}: max|V_xc| = {np.max(np.abs(V)):.3e} on a cross-cell "
        f"density (tau < tau_W libxc singularity not screened)"
    )
    assert np.all(np.isfinite(V))


@pytest.mark.parametrize("functional", ["tpss", "m06-l"])
def test_periodic_mgga_scf_eigenvalues_bounded(functional):
    """A periodic meta-GGA BIPOLE SCF converges to a physical solution with
    bounded MO eigenvalues (no virtual-subspace contamination)."""
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    sysp = _build_h2_box(a_bohr=8.0, bond_bohr=1.4)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # 12 bohr: drift 4.2e-7 (6 bohr sat at 9.5e-3, <10% under the refusal).
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 200
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.functional = functional
    res = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts,
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
    )
    assert res.converged
    eps = np.concatenate([np.asarray(e).ravel() for e in res.mo_energies])
    # Pre-fix: TPSS eps ~+/-3e7 (and E=+0.34 Ha); M06-L eps ~+/-1e8.
    assert np.max(np.abs(eps)) < 10.0, (
        f"{functional}: |eps|_max = {np.max(np.abs(eps)):.3e} (Fock virtual "
        f"subspace contaminated by the tau < tau_W libxc singularity)"
    )
    # H2/STO-3G electronic total is ~-1.16 Ha; the pre-fix TPSS basin was +0.34.
    assert -1.3 < res.energy < -1.0, f"{functional}: E = {res.energy:.4f} Ha"


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["tpss", "m06-l"])
def test_corrected_gauge_rks_mgga_gradient_matches_fd(functional):
    """The meta-GGA RKS Gamma analytic gradient matches the production FD path.

    With the tau >= tau_W floor applied identically in the SCF energy and the
    gradient XC kernel, the (previously contaminated) analytic gradient now
    follows the SCF energy surface -- the same ~1e-7 agreement SCAN already had.
    Asymmetric H2 box so the force is non-trivial."""
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.bipole_gradient import compute_bipole_gradient_rks

    sysp = _build_h2_box(a_bohr=6.0, bond_bohr=1.55)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(sysp, [1, 1, 1])
    opts = PeriodicKSOptions()
    # 11 bohr: drift 2.8e-5 (5 bohr refused at 1.1e-1).
    opts.lattice_opts.cutoff_bohr = 11.0
    opts.lattice_opts.nuclear_cutoff_bohr = 11.0
    opts.max_iter = 200
    opts.use_diis = True
    opts.conv_tol_energy = 1e-10
    opts.initial_guess = InitialGuess.SAD
    opts.functional = functional
    res = run_pbc_bipole_rks(
        sysp, basis, kmesh, opts,
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
    )
    assert res.converged and getattr(res, "exchange_ewald_split", False)
    g_an = np.asarray(compute_bipole_gradient_rks(
        sysp, basis, res, lattice_opts=opts.lattice_opts
    ))
    g_fd = np.asarray(compute_bipole_gradient_fd(
        sysp, "sto-3g", kmesh, opts, method="RKS", functional=functional,
        step_bohr=1e-3, use_ewald_j_split=True, ewald_precision=1e-8,
    ))
    np.testing.assert_allclose(g_an, g_fd, atol=1e-5)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("mesh_size", [1, 2])
@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize("precision", [None, 1e-4, 1e-12])
def test_scf_ewald_precision_reaches_public_gradient(
    monkeypatch, method, mesh_size, corrected, precision,
):
    """Check SCF provenance and the real gradient's E_nn boundary together.

    A one-basis He cell keeps all method/gauge dispatches inexpensive. This
    is a domain-selection check, not a force-accuracy claim for every route.
    """
    import importlib
    import vibeqc.bipole_gradient as gradients
    from vibeqc.pbc_bipole_common import _crystal_ewald_options
    from vibeqc.bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff

    system = vq.PeriodicSystem(3, np.eye(3) * 9., [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = monkhorst_pack(system, [mesh_size, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions() if method.endswith("ks") else PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.  # 2R covers the two-cell BvK density torus.
    opts.lattice_opts.nuclear_cutoff_bohr = 4.
    opts.initial_guess = InitialGuess.HCORE
    opts.max_iter = 8
    kwargs = {} if precision is None else {"ewald_precision": precision}
    if method.endswith("ks"):
        opts.grid.n_radial = 8
        opts.grid.n_theta = 5
        opts.grid.n_phi = 8
        opts.becke_image_radius_bohr = 3.
        kwargs["functional"] = "lda"
    module = "vibeqc.pbc_bipole" + ("" if method == "rhf" else "_" + method)
    driver = getattr(importlib.import_module(module), "run_pbc_bipole_" + method)
    result = driver(
        system, basis, mesh, opts, ewald_omega=.4,
        use_exchange_ewald_split=corrected, sr_image_precision=None,
        use_fock_symmetry=False, use_fock_symmetry_reduce=False,
        progress=False, **kwargs,
    )
    assert result.converged
    expected = 1e-8 if precision is None else precision
    captured = []
    leaf_precisions = {}

    def capture(system_arg, options):
        assert system_arg is system
        captured.append(options)
        return np.zeros((1, 3))

    def leaf(name):
        def evaluate(*args, precision=1e-8, **kwargs):
            leaf_precisions.setdefault(name, []).append(precision)
            return np.zeros((1, 3))
        return evaluate

    monkeypatch.setattr(gradients, "ewald_nuclear_repulsion_gradient", capture)
    for name in ("_v_ne_ewald_gradient", "_j_long_range_ewald_gradient",
                 "_k_long_range_ewald_gradient", "_j_long_range_ewald_gradient_multi_k",
                 "_k_long_range_ewald_gradient_multi_k"):
        monkeypatch.setattr(gradients, name, leaf(name))
    with pytest.warns(UserWarning, match="preview"):
        getattr(gradients, "compute_bipole_gradient_" + method)(
            system, basis, result, lattice_opts=opts.lattice_opts, kmesh=mesh,
            **({"grid_options": opts.grid, "becke_image_radius_bohr": 3.}
               if method.endswith("ks") else {}),
        )
    reference = _crystal_ewald_options(
        opts.lattice_opts, alpha_bohr_inv=.4, tolerance=expected,
        recip_cutoff_bohr_inv=bipole_ewald_reciprocal_cutoff(9.**3, .4),
    )
    assert captured[0].tolerance == expected
    assert captured[0].real_cutoff_bohr == reference.real_cutoff_bohr
    assert leaf_precisions["_v_ne_ewald_gradient"] == [expected]
    j_name = "_j_long_range_ewald_gradient" + ("_multi_k" if mesh_size > 1 else "")
    assert leaf_precisions[j_name] == [expected]
    if corrected and method.endswith("hf"):
        k_name = "_k_long_range_ewald_gradient" + ("_multi_k" if mesh_size > 1 else "")
        assert leaf_precisions[k_name] == [expected] * (2 if method == "uhf" else 1)
    assert result.ewald_precision == expected


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_gradient_old_result_defaults_ewald_precision(monkeypatch, method):
    """Older duck-typed results retain the historical 1e-8 contract."""
    import vibeqc.bipole_gradient as gradients

    system = vq.PeriodicSystem(3, 9. * np.eye(3), [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 4.
    density = compute_overlap_lattice(basis, system, lat)
    result = SimpleNamespace(
        exchange_ewald_split=True, ewald_alpha_bohr_inv=.4, converged=True,
        n_iter=1, functional="lda", density=density,
        density_alpha=density, density_beta=density,
        mo_coeffs=[np.eye(1)], mo_energies=[np.array([-1.])],
        mo_coeffs_alpha=[np.eye(1)], mo_coeffs_beta=[np.eye(1)],
        mo_energies_alpha=[np.array([-1.])], mo_energies_beta=[np.array([-1.])],
    )

    class BoundaryReached(Exception):
        pass

    def capture(*args, **kwargs):
        assert kwargs.get("ewald_precision", 1e-8) == 1e-8
        raise BoundaryReached

    monkeypatch.setattr(gradients, "_compute_bipole_gradient_corrected_gamma", capture)
    with pytest.warns(UserWarning, match="preview"), pytest.raises(BoundaryReached):
        getattr(gradients, "compute_bipole_gradient_" + method)(
            system, basis, result, lattice_opts=lat,
        )


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("mesh_size", [1, 2])
@pytest.mark.parametrize("precision", [1e-2, 1e-8])
def test_nondefault_ewald_precision_force_matches_displaced_scf(
    method, mesh_size, precision, record_property,
):
    """A loose image tolerance exposes a different finite nuclear sum.

    The force must differentiate that requested sum, even before convergence
    toward the infinite-lattice limit. Two FD steps exclude step cancellation.
    The default-precision rows are unchanged controls, not retuned references.
    """
    import importlib
    import vibeqc.bipole_gradient as gradients

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.
    opts.lattice_opts.nuclear_cutoff_bohr = 4.
    opts.initial_guess = InitialGuess.HCORE
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.max_iter = 100
    module = "vibeqc.pbc_bipole" + ("" if method == "rhf" else "_uhf")
    driver = getattr(importlib.import_module(module), "run_pbc_bipole_" + method)

    def run(displacement):
        positions = [[.1, .2, .3], [1.5 + displacement, .4, .3]]
        system = vq.PeriodicSystem(
            3, 9. * np.eye(3), [vq.Atom(1, p) for p in positions],
        )
        shells = [vq.ShellInfo(i, 0, False, [a], [1.], p)
                  for i, (a, p) in enumerate(zip([.7, 1.1], positions))]
        basis = vq.BasisSet(system.unit_cell_molecule(), shells, "force-precision", False)
        mesh = monkhorst_pack(system, [mesh_size, 1, 1], use_symmetry=False)
        result = driver(
            system, basis, mesh, opts, ewald_omega=.4, ewald_precision=precision,
            sr_image_precision=None, use_fock_symmetry=False,
            use_fock_symmetry_reduce=False, progress=False,
        )
        assert result.converged
        return system, basis, mesh, result

    system, basis, mesh, result = run(0.)
    with pytest.warns(UserWarning, match="preview"):
        analytic = getattr(gradients, "compute_bipole_gradient_" + method)(
            system, basis, result, lattice_opts=opts.lattice_opts, kmesh=mesh,
        )[1, 0]
    differences = [(run(h)[3].energy - run(-h)[3].energy) / (2 * h)
                   for h in (1e-4, 3e-5)]
    record_property("analytic_gradient", float(analytic))
    record_property("fd_step_1e-4", float(differences[0]))
    record_property("fd_step_3e-5", float(differences[1]))
    record_property("energy", float(result.energy))
    assert abs(differences[0] - differences[1]) < 1e-7
    assert abs(analytic - differences[1]) < 1e-7


@pytest.mark.parametrize("precision", [1e-4, 1e-8, 1e-12])
@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("consumer", ["closed_w", "open_w", "closed_b0", "open_b0", "gamma_fock"])
def test_response_reconstruction_retains_ewald_precision(monkeypatch, precision, physical, consumer):
    """Response matrices must reconstruct the energy's finite nuclear sum."""
    import vibeqc.bipole_gradient as gradients
    import vibeqc.pbc_bipole as driver

    system = vq.PeriodicSystem(3, 9. * np.eye(3), [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lat = LatticeSumOptions()
    lat.cutoff_bohr = lat.nuclear_cutoff_bohr = 4.
    lat.pair_complete_1e = physical
    density = compute_overlap_lattice(basis, system, lat)
    mesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    C = np.eye(1)
    eps = np.array([-1.])

    class BoundaryReached(Exception):
        pass

    def capture(basis_arg, system_arg, lat_arg, ew, overlap, *, precision, **kwargs):
        assert basis_arg is basis and system_arg is system
        assert lat_arg.pair_complete_1e == physical
        assert precision == expected and ew.tolerance == expected
        raise BoundaryReached

    expected = precision
    monkeypatch.setattr(driver, "_compute_nuclear_lattice_ewald_reciprocal_ft", capture)
    calls = {
        "closed_w": (gradients._corrected_w_gamma_closed,
                     (system, basis, density, C, 1, lat, .4, 1.)),
        "open_w": (gradients._corrected_w_gamma_open,
                   (system, basis, density, density, density, C, 1, C, 1, lat, .4, 1.)),
        "closed_b0": (gradients._build_multi_k_bipole_b0_closed,
                      (system, basis, [C, C], [eps, eps], 1, mesh, lat, .4)),
        "open_b0": (gradients._build_multi_k_bipole_b0_open,
                    (system, basis, [C, C], [eps, eps], [C, C], [eps, eps], 1, 1, mesh, lat, .4, 1.)),
        "gamma_fock": (gradients._reconstruct_bipole_fock_gamma_builder,
                       (system, basis, lat, .4)),
    }
    function, args = calls[consumer]
    with pytest.raises(BoundaryReached):
        function(*args, ewald_precision=precision)


def _explicit_response_basis(molecule, name):
    """Independent primitive specification, including a signed contraction."""
    return vq.BasisSet(molecule, [
        vq.ShellInfo(i, 0, False, [.7 + .4*i, .23 + .1*i], [.8, -.15], atom.xyz)
        for i, atom in enumerate(molecule.atoms)
    ], name, False)


def _custom_basis_response_call(consumer, name):
    """A fixed-orbital response probe, not a converged-SCF force fixture."""
    import vibeqc.bipole_gradient as gradients

    system = vq.PeriodicSystem(3, 9.*np.eye(3),
                              [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])])
    basis = _explicit_response_basis(system.unit_cell_molecule(), name)
    lat = LatticeSumOptions()
    lat.cutoff_bohr = lat.nuclear_cutoff_bohr = 5.
    overlap = compute_overlap_lattice(basis, system, lat)
    S = sum(np.asarray(block) for block in overlap.blocks)
    values, vectors = np.linalg.eigh(S)
    rotation = np.array([[np.cos(.3), -np.sin(.3)], [np.sin(.3), np.cos(.3)]])
    C = (vectors / np.sqrt(values)) @ vectors.T @ rotation
    # A positive, narrower multi-k gap gives a measurable diagonal response
    # even for the small KS energy/Fock mismatch of this compact fixture.
    eps = np.array([-2., -1.75 if consumer.startswith("multi_") else 2.])
    P = C[:, :1] @ C[:, :1].T

    def density(factor):
        result = compute_overlap_lattice(basis, system, lat)
        home = gradients._home_cell_index(list(result.cells))
        for i in range(len(result.cells)):
            result.set_block(i, factor * P if i == home else np.zeros_like(P))
        return result

    total, spin = density(2.), density(1.)
    mesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    common = (system, basis)
    closed = common + (total, C, eps, 1, lat, .4, 1.)
    opened = common + (total, spin, spin, C, eps, 1, C, eps, 1, lat, .4, 1.)
    multi_closed = common + ([C, C], [eps, eps], 1, mesh, lat, .4)
    multi_open = common + ([C, C], [eps, eps], 1, [C, C], [eps, eps], 1, mesh, lat, .4)
    grid = vq.GridOptions()
    grid.n_radial = 12
    grid.n_theta = 5
    grid.n_phi = 8
    if consumer.startswith("gamma_rhf_"):
        call = lambda: gradients._bloch_cphf_relaxation(
            *closed, cphf_rhs=consumer.removeprefix("gamma_rhf_"))
    elif consumer.startswith("gamma_uhf_"):
        call = lambda: gradients._bloch_cphf_relaxation_open(
            *opened, cphf_rhs=consumer.removeprefix("gamma_uhf_"))
    elif consumer == "gamma_rks":
        call = lambda: gradients._bloch_cphf_relaxation_ks_closed(
            *closed[:-1], 0., "svwn", grid, False, 5.)
    elif consumer == "gamma_uks":
        call = lambda: gradients._bloch_cphf_relaxation_ks_open(
            *opened[:-1], 0., "svwn", grid, False, 5.)
    elif consumer == "multi_rhf":
        call = lambda: gradients._multi_k_orbital_relaxation_closed_diag(*multi_closed)
    elif consumer == "multi_uhf":
        call = lambda: gradients._multi_k_orbital_relaxation_open(*multi_open, 1.)
    elif consumer == "multi_rks":
        call = lambda: gradients._multi_k_orbital_relaxation_ks_closed_diag(*multi_closed, "svwn")
    elif consumer == "multi_uks":
        call = lambda: gradients._multi_k_orbital_relaxation_ks_open_diag(*multi_open, "svwn")
    else:
        raise AssertionError(consumer)
    return system, basis, call


@pytest.mark.parametrize("name", ["response-custom-unregistered", "sto-3g"])
@pytest.mark.parametrize("consumer", [
    "gamma_rhf_hybrid", "gamma_rhf_seminumeric",
    "gamma_uhf_hybrid", "gamma_uhf_seminumeric",
    "gamma_rks", "gamma_uks", "multi_rhf", "multi_uhf", "multi_rks", "multi_uks",
])
def test_orbital_response_displacements_preserve_shell_inventory(monkeypatch, name, consumer):
    """Reach all ten response paths and inspect the native basis boundary."""
    import vibeqc.bipole_gradient as gradients

    system, basis, call = _custom_basis_response_call(consumer, name)
    original = list(basis.shells())

    class DisplacedBasisChecked(Exception):
        pass

    def check(molecule, shells, supplied_name=None, normalized=False):
        assert not isinstance(shells, str), "response reloaded the library basis"
        assert supplied_name == name and normalized is True
        for old, moved in zip(original, shells, strict=True):
            assert (moved.atom_index, moved.l, moved.pure) == (old.atom_index, old.l, old.pure)
            np.testing.assert_array_equal(moved.exponents, old.exponents)
            np.testing.assert_array_equal(moved.coefficients, old.coefficients)
            np.testing.assert_array_equal(moved.origin, molecule.atoms[moved.atom_index].xyz)
        old_positions = np.array([atom.xyz for atom in system.unit_cell])
        new_positions = np.array([atom.xyz for atom in molecule.atoms])
        assert np.max(np.abs(new_positions - old_positions)) == pytest.approx(1e-4)
        raise DisplacedBasisChecked

    monkeypatch.setattr(gradients, "BasisSet", check)
    with pytest.raises(DisplacedBasisChecked):
        call()


@pytest.mark.parametrize("consumer", [
    "gamma_rhf_hybrid", "gamma_rhf_seminumeric", "gamma_uhf_hybrid", "gamma_uhf_seminumeric",
    "gamma_rks", "gamma_uks", "multi_rhf", "multi_uhf", "multi_rks", "multi_uks",
])
def test_custom_basis_orbital_response_matches_explicit_displacements(monkeypatch, record_property, consumer):
    """Numerical response term agrees with independently specified primitives.

    The fixed orbitals have occupied/virtual coupling. This tests the response
    component, without asserting full legacy-gauge SCF/force acceptance.
    """
    import vibeqc.bipole_gradient as gradients

    _, _, call = _custom_basis_response_call(consumer, "sto-3g")
    actual = call()

    def explicit(template, displaced):
        return _explicit_response_basis(displaced.unit_cell_molecule(), template.name)

    # The reference also intercepts the old name constructor so the exact same
    # independent primitive oracle runs on the parent implementation.
    with monkeypatch.context() as patch:
        patch.setattr(gradients, "_recenter_basis_on_periodic_system", explicit)
        patch.setattr(gradients, "BasisSet", lambda molecule, name: _explicit_response_basis(molecule, name))
        expected = call()
    record_property("response_actual", np.asarray(actual).tolist())
    record_property("response_explicit", np.asarray(expected).tolist())
    assert np.max(np.abs(expected)) > 1e-5
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0.)
    # Repeating with an unregistered label must leave the same numerical term.
    _, _, renamed_call = _custom_basis_response_call(consumer, "response-custom-unregistered")
    np.testing.assert_allclose(renamed_call(), expected, atol=1e-9, rtol=0.)
