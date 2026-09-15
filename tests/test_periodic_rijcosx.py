"""Periodic RIJCOSX — regression tests (M3a: image-cell COSX-K summation).

M3 delivered periodic-correct GDF Coulomb J (compcell/RSGDF Lpq tensor)
with a home-cell-only COSX exchange K.  M3a adds the image-cell exchange
summation: ``compute_cosx_k`` accepts an ``image_cells`` list and, per
grid point, accumulates the analytic A-operator over all lattice cells R
(pseudo-nucleus at r_g − R), summing

    K_{μν} += Σ_R Σ_{λσ} D_{λσ} (μ_0 λ_0 | ν_R σ_R)

— the Coulomb coupling of the home-cell bra pair density to every
image-cell replica of the ket pair density, truncated at
``lattice_opts.cutoff_bohr`` (direct-truncated regularization).

Measured M3a state on the H-chain anchor (2 H, 4-bohr z-cell, 20-bohr xy
pad, STO-3G, D = I):

    ||K_home-only||      = 1.7377   (M3 behavior, cutoff < lattice const)
    ||K_image-summed||   = 3.9393   (M3a, 7 cells at 15-bohr cutoff)
    ||K_direct-ERI||     = 2.7633   (build_jk_gamma_molecular_limit —
                                     cell-diagonal-density double sum)
    ||K_GDF||            = 16.162   (Γ-folded GDF exchange)

The single-lattice-sum model is exact for vacuum-padded cells (image
list collapses to the home cell) and adds the leading 1/R image-replica
exchange class on tight cells, but it is NOT the Γ-folded GDF exchange:
at Γ-only sampling the density matrix couples *all* cell pairs, so the
GDF K contains long-range exchange classes (triple lattice sum /
Ewald-resummed) that no finite single sum reproduces.  Closing that gap
needs the cell-pair double-sum COSX kernel (vs the direct-ERI
reference) and/or the multi-k / divergence-corrected treatment — see
handovers/HANDOVER_RIJCOSX_M3A.md § "M3a measured outcome" for the full analysis.

Status by test:
  test_rijcosx_pbc_molecular_limit_h2    — PASS: RIJCOSX vs GDF on
    vacuum-padded cell (50 bohr box). Both J and K match.
  test_rijcosx_pbc_runner_dispatch       — PASS: run_periodic_job
    dispatches to RIJCOSX path.
  test_rijcosx_runner_dispatches_rks_multik_to_gdf_cosx — PASS:
    public jk_method='rijcosx' reaches the multi-k GDF/COSX backend for
    RKS.
  test_rijcosx_pbc_runner_remaining_multik_dispatches — PASS: public
    jk_method='rijcosx' reaches the multi-k GDF/COSX backend for
    RHF/UHF/UKS.
  test_rijcosx_pbc_runner_single_k_rks_fails_closed — PASS: one-point
    RKS does not silently fall back to a non-COSX route.
  test_rijcosx_pbc_tight_j_parity        — PASS: RIJCOSX J matches GDF J
    to machine precision on tight cell.
  test_rijcosx_pbc_tight_k_image_cells   — PASS: image-cell summation is
    active on tight cells and pinned against the measured M3a values.
  test_cosx_k_image_cells_binding        — PASS: Python binding for
    ``image_cells``; trivial home-only list falls back to the M1 path
    bit-exactly.
  test_rijcosx_pbc_tight_k_gamma_parity  — PASS: the M3b cell-pair
    double sum closes the Γ-folded exchange classes that M3a omitted.

M3b-1 (cell-pair double-sum kernel, ``compute_cosx_k_cell_pair``):
  the Γ-point cell-diagonal-density exchange
  K = Σ_{g,p} Σ_{λσ} D_{λσ} (μ_0 λ_p | ν_g σ_p) — the same object as
  the direct-ERI ``build_jk_gamma_molecular_limit`` K — evaluated
  seminumerically with per-bra-cell shifted AOs and per-relative-shift
  (δ = g − p) analytic A-blocks. This is the engine for multi-k
  periodic COSX (handovers/HANDOVER_RIJCOSX_M3A.md § "M3b direction").

  test_cosx_cell_pair_vacuum_reduces_to_molecular — PASS: home-only
    cell list reproduces the molecular ``compute_cosx_k``.
  test_cosx_cell_pair_vs_direct_eri_chain — PASS: ERI-exact reference
    parity on the H-chain at COSX grid-quadrature accuracy.
  test_cosx_cell_pair_vs_direct_eri_3d    — PASS: same on a 3D cubic
    cell (image sums along all three axes).

M3b-2 (real-space block engine, ``compute_cosx_k_blocks``): K(g)
  exchange blocks from D(g) density blocks — the multi-k periodic COSX
  engine; same object and summation domain as the direct-ERI
  ``build_jk_2e_real_space_explicit`` K. ``compute_cosx_k_cell_pair``
  is now a thin Γ-fold wrapper over this engine.

  test_cosx_k_blocks_cell_diagonal_consistency — PASS: P = {0 ↦ D}
    blocks Γ-fold to the cell-pair kernel's K exactly.
  test_cosx_k_blocks_vs_direct_eri_chain — PASS: per-block ERI-exact
    parity on the H-chain with a decaying synthetic P(g) set.
  test_cosx_k_blocks_vs_direct_eri_3d    — PASS: same on the 3D cube.

M3b-3 (multi-k bridge ``KPointCosxK`` + ``k_exchange='cosx'`` in
  ``run_krhf_periodic_gdf``): D(k) → D(g) → K(g) → K(k). The K(k)
  build is ERI-exact-validated; the SCF backend gap vs GDF is the
  ball-truncated exchange model (supercell-matched truncation is
  M3b-4 — measured on the gapped dimerized H2 chain: 62 mHa at
  kmesh (1,1,4), 17 mHa at (1,1,6)).

  test_cosx_kpoint_bridge_vs_direct_eri — PASS: K(k) from a converged
    multi-k density matches the direct-ERI truncated-exchange fold.
  test_krhf_cosx_backend_scf            — PASS: k_exchange='cosx' SCF
    converges; energy within the measured truncation envelope of the
    GDF backend; flag validation errors fire.

M3b-4a (erfc-SR exchange kernel): the in-tree COSX kernel gains the
  erfc(ω·r)/r short-range Coulomb (seeding-only change; ULP-pinned vs
  libint's erfc_nuclear in test_cosx_kernel.py); the cell-pair caches
  carry ω with a matching erfc Schwarz metric. The K(g) blocks
  localise at the kernel range 1/ω (bra-ket coupling) — alias-immune
  for any k-mesh; the δ cache set stays overlap-bounded as before.

  test_cosx_k_blocks_erfc_sr_vs_direct_eri — PASS: SR blocks vs the
    erfc direct-ERI reference at quadrature accuracy + SR locality.

M3b-4b (SR+LR composition): ``KPointCosxK.k_matrices(lr_complement=
  True)`` adds the smooth LR-erf complement in reciprocal space
  (ket-Bloch pair-FTs, 4π·e^{−|G+q|²/4ω²}/|G+q|²/V weights, G=0
  dropped in the GDF gauge, −π/ω²·S D S†/V/N_k finite-part term).
  ω-invariant at 1e-4; matches the independent RSGDF exchange at the
  DF-fit floor. In the course of validation, a compcell Lpq
  zone-edge deviation (~0.15 in ||K||) was found on vacuum-padded
  systems and escalated to the GDF route.

  test_cosx_k_composed_vs_rsgdf_and_compcell_flag — PASS: the
    acceptance (composed vs RSGDF ~1e-4) + the flagged compcell
    deviation pinned as a characterisation.

M3b-4c (single-gauge production pairing): ``gdf_method='rsgdf'`` in
  ``run_krhf_periodic_gdf`` builds the Lpq cache via the all-FT
  Bloch-pair route, so RSGDF-J + COSX-K share one gauge. Backend
  parity 0.024–0.027 mHa across the (1,1,2)–(1,1,6) ladder, clean
  convergence through (1,1,8) — the dense-mesh stalls were entirely
  the mixed-gauge (compcell-J + exact-K) Fock.

  test_krhf_cosx_backend_scf_single_gauge — PASS: the production
    gate, pinned at (1,1,2).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import (
    Atom,
    BasisSet,
    LatticeSumOptions,
    Molecule,
    PeriodicRHFOptions,
    PeriodicSystem,
    build_grid,
)
from vibeqc import _vibeqc_core as core
from vibeqc.aux_basis import build_lpq_compcell, make_aux_basis_set
from vibeqc.pbc_gdf import _build_j_from_lpq, _build_k_from_lpq, run_pbc_gdf_rhf
from vibeqc.periodic_rijcosx import run_periodic_rijcosx_rhf

_E_TOL = 1e-3  # 1 mHa — COSX-K quadrature error band


def _h2_box(box_bohr: float = 50.0) -> tuple:
    """H₂ in a large cubic box — molecular-limit regime."""
    atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
    lat = np.diag([box_bohr, box_bohr, box_bohr])
    sys = PeriodicSystem(3, lat, atoms)
    mol = Molecule(atoms, 0, 1)
    basis = BasisSet(mol, "sto-3g")
    return sys, basis, mol


def _h_chain() -> tuple:
    """H chain: 2 H per cell in a tight 4-bohr z-cell, 20-bohr xy pad.

    The M3/M3a tight-cell anchor — equally-spaced H atoms along z
    (in-cell bond 2.0 = half the lattice constant).
    """
    Lz = 4.0
    Lpad = 20.0
    lat = np.diag([Lpad, Lpad, Lz])
    atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 2.0])]
    sys = PeriodicSystem(3, lat, atoms)
    mol = Molecule(atoms, 0, 1)
    basis = BasisSet(mol, "sto-3g")
    return sys, basis, mol


def test_rijcosx_pbc_molecular_limit_h2():
    """H2 in a 50-bohr box: RIJCOSX matches GDF to 1 mHa."""
    sys, basis, _ = _h2_box(50.0)

    # GDF reference (validated against direct SCF).
    ref = vq.run_pbc_gdf_rhf(sys, basis, aux_basis="def2-universal-jkfit")
    assert ref.converged
    ref_e = float(ref.energy)

    opts = PeriodicRHFOptions()
    opts.lattice_opts = LatticeSumOptions()
    opts.lattice_opts.cutoff_bohr = 15.0
    opts.conv_tol_energy = 1e-10
    opts.max_iter = 100
    result = run_periodic_rijcosx_rhf(
        sys, basis, opts, aux_basis="def2-universal-jkfit"
    )
    assert result.converged, f"RIJCOSX did not converge: {result.n_iter} iters"

    delta = abs(result.energy - ref_e)
    assert delta < _E_TOL, (
        f"H2/50-bohr: RIJCOSX {result.energy:.8f} vs GDF {ref_e:.8f}, "
        f"Δ={delta:.3e} Ha > {_E_TOL:.1e}"
    )


def test_rijcosx_pbc_runner_dispatch(tmp_path):
    """Verify run_periodic_job dispatches to the RIJCOSX path."""
    from vibeqc.periodic_runner import run_periodic_job

    sys, basis, _ = _h2_box(50.0)
    result = run_periodic_job(
        sys,
        basis,
        method="RHF",
        jk_method="rijcosx",
        aux_basis="def2-universal-jkfit",
        max_iter=50,
        conv_tol_energy=1e-8,
        output=str(tmp_path / "rijcosx_dispatch"),
    )
    assert result.converged, f"run_periodic_job RIJCOSX: {result.n_iter} iters"
    assert -2.0 < result.energy < 0.0, f"energy {result.energy:.6f} out of range"


def test_periodic_cosx_one_center_correction_is_post_scf():
    """Both Gamma COSX builders keep the one-center correction post-SCF.

    ``JKBuilder`` defines the iterated RHF two-electron contribution as
    ``J - 0.5 K``.  The analytic one-center COSX correction is deliberately
    applied only by the finalization hook: putting its quadrature residual
    into every iteration previously prevented tight molecular COSX
    convergence (commit 40dad042).  The periodic builders must follow the
    same lifecycle.
    """
    from vibeqc.periodic_gradient import _fold_gamma_real
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    system, basis, mol = _h2_box(50.0)
    aux = make_aux_basis_set(mol, aux_name="def2-universal-jkfit")
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 15.0
    density = np.array([[1.0, 0.2], [0.2, 0.5]])

    builders = (
        (
            "gamma",
            core.make_periodic_gamma_cosx_jk_builder(
                basis, aux, system, lat_opts
            ),
        ),
        (
            "tight",
            # A single zero Lpq factor keeps this synthetic builder valid;
            # the J value is irrelevant to the lifecycle under test.
            core.make_periodic_tight_cosx_jk_builder(
                basis,
                np.zeros((1, basis.nbasis, basis.nbasis)),
                system,
                lat_opts,
            ),
        ),
    )

    # The final SCF pass must take the opposite path: rebuild the
    # uncorrected J/K energy surface, then add the one-center correction to
    # the returned Fock and orbitals through apply_one_center_correction().
    overlap = _fold_gamma_real(
        core.compute_overlap_lattice(basis, system, lat_opts)
    )
    kinetic = _fold_gamma_real(
        core.compute_kinetic_lattice(basis, system, lat_opts)
    )
    attraction = _fold_gamma_real(
        compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    )
    hcore = kinetic + attraction
    e_nuc = core.nuclear_repulsion_per_cell(system, lat_opts)
    scf_opts = vq.RHFOptions()
    scf_opts.max_iter = 50
    scf_opts.conv_tol_energy = 1e-10
    for name, jk in builders:
        j_matrix = np.asarray(jk.build_J(density))
        k_matrix = np.asarray(jk.build_K(density))
        g_matrix = np.asarray(jk.build_g_rhf(density, 1.0))
        assert (
            np.linalg.norm(g_matrix - (j_matrix - 0.5 * k_matrix)) < 1e-12
        ), name

        result = core.run_rhf_scf_with_jk(
            basis,
            system.n_electrons(),
            overlap,
            hcore,
            e_nuc,
            jk,
            scf_opts,
        )
        assert result.converged, name
        final_density = np.asarray(result.density)
        final_j = np.asarray(jk.build_J(final_density))
        final_k = np.asarray(jk.build_K(final_density))
        uncorrected_fock = hcore + final_j - 0.5 * final_k
        final_correction = np.asarray(result.fock) - uncorrected_fock
        assert np.linalg.norm(final_correction) > 1e-8, name

        expected_energy = (
            e_nuc
            + np.sum(final_density * hcore)
            + 0.5 * np.sum(final_density * final_j)
            - 0.25 * np.sum(final_density * final_k)
        )
        assert abs(result.energy - expected_energy) < 1e-12, name


def test_rijcosx_runner_dispatches_rks_multik_to_gdf_cosx(monkeypatch, tmp_path):
    """The public RIJCOSX route reaches the validated multi-k COSX backend."""
    from vibeqc.periodic_k_gdf import PeriodicKRHFGDFResult
    from vibeqc.periodic_runner import run_periodic_job

    sys, basis, _ = _h2_box(12.0)
    captured = {}

    def fake_run_krks(_system, _basis, kmesh, _options, **kwargs):
        captured["kmesh"] = kmesh
        captured["kwargs"] = kwargs
        nbf = _basis.nbasis
        n_k = 2
        zeros = [np.zeros((nbf, nbf), dtype=complex) for _ in range(n_k)]
        density = [np.diag([2.0] + [0.0] * (nbf - 1)).astype(complex)
                   for _ in range(n_k)]
        return PeriodicKRHFGDFResult(
            energy=-1.0,
            e_electronic=-1.5,
            e_nuclear=0.5,
            n_iter=1,
            converged=True,
            mo_energies=[np.linspace(-0.5, 0.5, nbf) for _ in range(n_k)],
            mo_coeffs=[np.eye(nbf, dtype=complex) for _ in range(n_k)],
            fock=zeros,
            overlap=[np.eye(nbf, dtype=complex) for _ in range(n_k)],
            hcore=zeros,
            density=density,
            kpoints_cart=np.zeros((n_k, 3), dtype=float),
            kpoint_weights=np.full(n_k, 1.0 / n_k),
            functional=str(kwargs["functional"]),
            e_xc=-0.1,
            free_energy=-1.0,
            occupations=[np.array([2.0] + [0.0] * (nbf - 1))
                         for _ in range(n_k)],
            aux_basis_name=str(kwargs["aux_basis"] or "def2-universal-jkfit"),
            n_aux=1,
            backend="native-multi-k-gdf-cosx-rks",
        )

    monkeypatch.setattr("vibeqc.periodic_runner.run_krks_periodic_gdf",
                        fake_run_krks)

    result = run_periodic_job(
        sys,
        basis,
        method="RKS",
        functional="lda",
        jk_method="rijcosx",
        kpoints=(2, 1, 1),
        aux_basis="def2-universal-jkfit",
        output=tmp_path / "rijcosx-rks-multik",
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        output_qvf=False,
        progress=False,
    )

    assert result.energy == pytest.approx(-1.0)
    assert captured["kmesh"] == (2, 1, 1)
    kwargs = captured["kwargs"]
    assert kwargs["functional"] == "lda"
    assert kwargs["use_compcell"] is True
    assert kwargs["k_exchange"] == "cosx"
    assert kwargs["gdf_method"] == "rsgdf"
    assert result.backend == "native-multi-k-gdf-cosx-rks"


@pytest.mark.parametrize(
    ("method", "functional", "backend_suffix"),
    [
        ("RHF", None, "rhf"),
        ("UHF", None, "uhf"),
        ("UKS", "pbe0", "uks"),
    ],
)
def test_rijcosx_pbc_runner_remaining_multik_dispatches(
    tmp_path, method, functional, backend_suffix
):
    """Public jk_method='rijcosx' reaches the non-RKS multi-k backend."""
    from vibeqc.periodic_runner import run_periodic_job

    sys, basis, _ = _dimerized_chain()
    result = run_periodic_job(
        sys,
        basis,
        method=method,
        functional=functional,
        jk_method="rijcosx",
        kpoints=(1, 1, 2),
        aux_basis="def2-universal-jkfit",
        output=tmp_path / f"rijcosx_{method.lower()}",
        max_iter=40,
        conv_tol_energy=1e-8,
        write_molden_file=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=False,
    )
    assert result.converged, f"{method} RIJCOSX did not converge"
    assert np.isfinite(float(result.energy))
    assert getattr(result, "backend", "").endswith(f"cosx-{backend_suffix}")
    if method == "UKS":
        assert getattr(result, "functional", "").lower() == "pbe0"
        assert abs(float(getattr(result, "e_hf_exchange", 0.0))) > 1e-8


def test_rijcosx_pbc_runner_single_k_rks_fails_closed(tmp_path):
    """One-point RKS outside the vacuum-padded envelope must not be
    mislabeled as RIJCOSX; one-point UHF/UKS stay fail-closed entirely.

    The dimerized chain has a 6-bohr active axis, well below the
    vacuum-padded criterion, so Gamma RIJCOSX RKS refuses it (its XC is
    integrated on the molecular grid, exact only for compactly
    supported densities)."""
    from vibeqc.periodic_runner import run_periodic_job

    sys, basis, _ = _dimerized_chain()
    common = dict(
        jk_method="rijcosx",
        kpoints=(1, 1, 1),
        aux_basis="def2-universal-jkfit",
        write_molden_file=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=False,
    )
    with pytest.raises(NotImplementedError, match="vacuum-padded"):
        run_periodic_job(
            sys,
            basis,
            method="RKS",
            functional="pbe0",
            output=tmp_path / "rijcosx_rks_gamma",
            **common,
        )
    with pytest.raises(NotImplementedError, match="vacuum-padded"):
        run_periodic_job(
            sys,
            basis,
            method="UKS",
            functional="pbe0",
            output=tmp_path / "rijcosx_uks_gamma",
            **common,
        )


def test_rijcosx_pbc_tight_j_parity():
    """M3: RIJCOSX J matches GDF J to machine precision on a tight cell.

    Uses the same Lpq tensor for both J-builds (GDF: _build_j_from_lpq
    in Python; RIJCOSX: PeriodicTightCellCOSXJKBuilder::build_J in C++).
    Verifies the C++ builder's GDF J contraction is bit-exact with the
    Python reference.
    """
    sys, basis, mol = _h_chain()

    aux = make_aux_basis_set(mol, aux_name="def2-universal-jkfit")
    Lpq = build_lpq_compcell(sys, basis, aux, molecule=mol, linear_dep_thr=1e-9)

    # Get a converged density from the RIJCOSX SCF.
    result = run_periodic_rijcosx_rhf(
        sys,
        basis,
        aux_basis="def2-universal-jkfit",
        gdf_method="compcell",
        tight_cell=True,
    )
    assert result.converged
    D = result.density

    # Build J from Python Lpq.
    J_py = _build_j_from_lpq(Lpq, D)

    # Build J from C++ builder (same Lpq).
    from vibeqc._vibeqc_core import make_periodic_tight_cosx_jk_builder

    jk = make_periodic_tight_cosx_jk_builder(
        basis, np.ascontiguousarray(Lpq), sys, LatticeSumOptions()
    )
    J_cpp = np.asarray(jk.build_J(D))

    delta = np.linalg.norm(J_py - J_cpp)
    assert delta < 1e-12, f"RIJCOSX J vs Python J: ||ΔJ|| = {delta:.3e} > 1e-12"


def test_rijcosx_pbc_tight_k_image_cells():
    """M3a: the COSX-K image-cell summation is active on tight cells.

    Two builders on the same system and density, differing only in the
    lattice-sum cutoff: 3.5 bohr (< 4-bohr lattice constant → home cell
    only ≡ M3 behavior) vs the 15-bohr default (7 cells along z).  The
    image-summed K must reproduce the measured M3a anchor values
    (module docstring table; D = I, deterministic Becke grid):

        ||K_home||         = 1.7377
        ||K_img||          = 3.9393
        ||K_img - K_home|| = 2.2096

    Bands are ±5 % — the quadrature is deterministic for a fixed grid
    build; the band absorbs BLAS/OMP reduction noise only.
    """
    sys, basis, mol = _h_chain()
    D = np.eye(basis.nbasis)

    aux = make_aux_basis_set(mol, aux_name="def2-universal-jkfit")
    Lpq = np.ascontiguousarray(
        build_lpq_compcell(sys, basis, aux, molecule=mol, linear_dep_thr=1e-9)
    )

    assert len(core.direct_lattice_cells(sys, 3.5)) == 1
    assert len(core.direct_lattice_cells(sys, 15.0)) == 7

    opts_home = LatticeSumOptions()
    opts_home.cutoff_bohr = 3.5
    opts_img = LatticeSumOptions()  # default 15 bohr

    jk_home = core.make_periodic_tight_cosx_jk_builder(basis, Lpq, sys, opts_home)
    jk_img = core.make_periodic_tight_cosx_jk_builder(basis, Lpq, sys, opts_img)

    K_home = np.asarray(jk_home.build_K(D))
    K_img = np.asarray(jk_img.build_K(D))

    n_home = np.linalg.norm(K_home)
    n_img = np.linalg.norm(K_img)
    n_delta = np.linalg.norm(K_img - K_home)

    assert abs(n_home - 1.7377) < 0.09, f"||K_home|| = {n_home:.4f} (exp 1.7377)"
    assert abs(n_img - 3.9393) < 0.20, f"||K_img|| = {n_img:.4f} (exp 3.9393)"
    assert abs(n_delta - 2.2096) < 0.11, (
        f"||K_img - K_home|| = {n_delta:.4f} (exp 2.2096)"
    )
    assert n_img > n_home, "image summation must add exchange on a tight cell"


def test_cosx_k_image_cells_binding():
    """``compute_cosx_k(..., image_cells=...)`` Python binding (M3a).

    On the tight chain with a molecular Becke grid and D = I:

      * a trivial home-only list must fall back to the single-pass
        kernel (same code path as the lattice-only call — equal up to
        OMP grid-point-scheduling summation-order noise), and
      * the full 7-cell list must change K by a finite amount.
    """
    sys, basis, mol = _h_chain()
    D = np.eye(basis.nbasis)
    lat = np.asarray(sys.lattice)

    grid = build_grid(mol, vq.GridOptions())
    cells = core.direct_lattice_cells(sys, 15.0)
    home_only = core.direct_lattice_cells(sys, 3.5)
    assert len(home_only) == 1

    K_lat = np.asarray(core.compute_cosx_k(basis, D, grid, lattice=lat))
    K_home = np.asarray(
        core.compute_cosx_k(basis, D, grid, lattice=lat, image_cells=home_only)
    )
    K_img = np.asarray(
        core.compute_cosx_k(basis, D, grid, lattice=lat, image_cells=cells)
    )

    # Trivial list → fallback to the lattice-only path. The two calls
    # run the identical kernel; the residual is OMP summation-order
    # noise (grid points land on different threads per run), bounded
    # well below 1e-12 at this system size.
    assert np.linalg.norm(K_home - K_lat) < 1e-12
    # Non-trivial list → image contributions present.
    assert np.linalg.norm(K_img - K_lat) > 0.1


# ---------------------------------------------------------------------------
# M3b-1: cell-pair double-sum COSX-K vs the direct-ERI reference
# ---------------------------------------------------------------------------


def test_cosx_cell_pair_vacuum_reduces_to_molecular():
    """M3b-1: home-only cell list reduces to the molecular kernel.

    H2 in a 50-bohr box at the 15-bohr cutoff → the cell list is the
    zero cell only, δ = {0}, and the cell-pair kernel evaluates exactly
    the molecular COSX sum (rectangular pair loop instead of
    triangular+transpose — same math, different summation order, so
    agreement is at accumulated-roundoff level, not bitwise).
    """
    sys, basis, mol = _h2_box(50.0)
    D = np.eye(basis.nbasis)
    grid = build_grid(mol, vq.GridOptions())
    q = core.build_cosx_q(basis, grid)

    cells = core.direct_lattice_cells(sys, 15.0)
    assert len(cells) == 1

    caches = core.build_cosx_cell_pair_caches(basis, cells)
    assert caches.n_cells == 1
    assert caches.n_deltas == 1

    K_cp = np.asarray(
        core.compute_cosx_k_cell_pair(basis, D, grid, caches, q_cached=q)
    )
    K_mol = np.asarray(core.compute_cosx_k(basis, D, grid, q_cached=q))

    delta = np.linalg.norm(K_cp - K_mol)
    assert delta < 1e-9, f"home-only cell-pair vs molecular: {delta:.3e}"


def test_cosx_cell_pair_vs_direct_eri_chain():
    """M3b-1: ERI-exact parity on the H-chain (1D image sums).

    The direct-ERI reference (``build_jk_gamma_molecular_limit``)
    evaluates the identical cell-diagonal double sum analytically on
    the same cell list, so the residual is pure COSX grid-quadrature
    error. Measured anchor (D = I, default molecular Becke grid +
    Q-junction): ||K_ref|| = 2.7633, ||K_cp − K_ref|| = 2.0e-4 —
    versus 1.2519 for the M3a single-lattice-sum K on the same
    reference (a ~6000× closer match; the single sum is a different
    model, not a worse quadrature).
    """
    sys, basis, mol = _h_chain()
    D = np.eye(basis.nbasis)
    grid = build_grid(mol, vq.GridOptions())
    q = core.build_cosx_q(basis, grid)

    cells = core.direct_lattice_cells(sys, 15.0)
    assert len(cells) == 7
    caches = core.build_cosx_cell_pair_caches(basis, cells)
    assert caches.n_deltas >= 5  # 0, ±4, ±8 must survive Schwarz

    K_cp = np.asarray(
        core.compute_cosx_k_cell_pair(basis, D, grid, caches, q_cached=q)
    )

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    K_ref = np.asarray(
        core.build_jk_gamma_molecular_limit(basis, sys, opts, D).K
    )

    delta = np.linalg.norm(K_cp - K_ref)
    assert delta < 2e-3, (
        f"cell-pair COSX vs direct-ERI (chain): ||ΔK|| = {delta:.2e}, "
        f"||K_ref|| = {np.linalg.norm(K_ref):.4f} (measured 2.0e-4)"
    )


def test_cosx_cell_pair_vs_direct_eri_3d():
    """M3b-1: ERI-exact parity on a 3D cubic cell (image sums on all
    three axes).

    H2 in a 6-bohr cube at a 7-bohr cutoff → 7 cells (home + 6 face
    neighbours; 25 Schwarz-surviving δ shifts including the cross-axis
    diagonals). Same comparison as the chain test. Measured (D = I):
    ||K_ref|| = 2.1743, ||K_cp − K_ref|| = 3.7e-4.
    """
    L = 6.0
    atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
    sys = PeriodicSystem(3, np.diag([L, L, L]), atoms)
    mol = Molecule(atoms, 0, 1)
    basis = BasisSet(mol, "sto-3g")
    D = np.eye(basis.nbasis)
    grid = build_grid(mol, vq.GridOptions())
    q = core.build_cosx_q(basis, grid)

    cells = core.direct_lattice_cells(sys, 7.0)
    assert len(cells) == 7
    caches = core.build_cosx_cell_pair_caches(basis, cells)

    K_cp = np.asarray(
        core.compute_cosx_k_cell_pair(basis, D, grid, caches, q_cached=q)
    )

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 7.0
    K_ref = np.asarray(
        core.build_jk_gamma_molecular_limit(basis, sys, opts, D).K
    )

    delta = np.linalg.norm(K_cp - K_ref)
    assert delta < 4e-3, (
        f"cell-pair COSX vs direct-ERI (3D): ||ΔK|| = {delta:.2e}, "
        f"||K_ref|| = {np.linalg.norm(K_ref):.4f} (measured 3.7e-4)"
    )


# ---------------------------------------------------------------------------
# M3b-2: real-space K(g) blocks from D(g) blocks vs the direct-ERI reference
# ---------------------------------------------------------------------------


def _cell_lookup(cells):
    """Map integer cell index tuple → LatticeCell."""
    return {tuple(np.asarray(c.index)): c for c in cells}


def test_cosx_k_blocks_cell_diagonal_consistency():
    """M3b-2: P = {0 ↦ D} blocks Γ-fold to the cell-pair kernel's K.

    The cell-pair wrapper is implemented on top of the block engine,
    so the Γ-fold of the raw blocks, symmetrised and Q-corrected in
    numpy, must reproduce ``compute_cosx_k_cell_pair`` to summation-
    order noise.
    """
    sys, basis, mol = _h_chain()
    D = np.eye(basis.nbasis)
    grid = build_grid(mol, vq.GridOptions())
    q = np.asarray(core.build_cosx_q(basis, grid))

    cells = core.direct_lattice_cells(sys, 15.0)
    caches = core.build_cosx_cell_pair_caches(basis, cells)

    zero = _cell_lookup(cells)[(0, 0, 0)]
    P_diag = core.make_lattice_matrix_set(basis.nbasis, [zero], [D])

    K_set = core.compute_cosx_k_blocks(basis, P_diag, grid, caches)
    K_fold = np.sum([np.asarray(b) for b in K_set.blocks], axis=0)
    K_sym = 0.5 * (K_fold + K_fold.T)
    K_from_blocks = q @ K_sym @ q.T

    K_cp = np.asarray(
        core.compute_cosx_k_cell_pair(basis, D, grid, caches, q_cached=q)
    )
    delta = np.linalg.norm(K_from_blocks - K_cp)
    assert delta < 1e-10, f"block fold vs cell-pair wrapper: {delta:.3e}"


def _chain_p_set(basis, cells):
    """Synthetic decaying density blocks on the chain: P(0) = I,
    P(±a_z) = M / Mᵀ (satisfies P(h) = P(−h)ᵀ)."""
    by_index = _cell_lookup(cells)
    M = np.array([[0.4, 0.1], [0.2, 0.3]])
    p_cells = [by_index[(0, 0, 0)], by_index[(0, 0, 1)], by_index[(0, 0, -1)]]
    p_blocks = [np.eye(basis.nbasis), M, M.T]
    return core.make_lattice_matrix_set(basis.nbasis, p_cells, p_blocks)


def test_cosx_k_blocks_vs_direct_eri_chain():
    """M3b-2: per-block ERI-exact parity on the H-chain.

    ``build_jk_2e_real_space_explicit`` evaluates the identical block
    object analytically on the same cell list and density set; the
    residual is COSX grid-quadrature error (Q-corrected blocks via the
    left Q-junction + pairwise K(g) = K(−g)ᵀ symmetrisation).
    """
    sys, basis, mol = _h_chain()
    grid = build_grid(mol, vq.GridOptions())
    q = np.asarray(core.build_cosx_q(basis, grid))

    cells = core.direct_lattice_cells(sys, 15.0)
    caches = core.build_cosx_cell_pair_caches(basis, cells)
    P_set = _chain_p_set(basis, cells)

    K_set = core.compute_cosx_k_blocks(
        basis, P_set, grid, caches, q_cached=q
    )

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    ref = core.build_jk_2e_real_space_explicit(
        basis, sys, opts, P_set, cells
    )
    K_ref = ref.K

    deltas = [
        np.linalg.norm(np.asarray(K_set.blocks[i]) - np.asarray(K_ref.blocks[i]))
        for i in range(len(cells))
    ]
    worst = max(deltas)
    ref_norm = max(np.linalg.norm(np.asarray(b)) for b in K_ref.blocks)
    assert worst < 2e-3, (
        f"K(g) blocks vs direct ERI (chain): worst ||ΔK(g)|| = {worst:.2e}, "
        f"max ||K_ref(g)|| = {ref_norm:.4f}"
    )


def test_cosx_k_blocks_vs_direct_eri_3d():
    """M3b-2: per-block ERI-exact parity on the 3D cube.

    Face-neighbour density blocks on all three axes (P(h) = P(−h)ᵀ).
    """
    L = 6.0
    atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
    sys = PeriodicSystem(3, np.diag([L, L, L]), atoms)
    mol = Molecule(atoms, 0, 1)
    basis = BasisSet(mol, "sto-3g")
    grid = build_grid(mol, vq.GridOptions())
    q = np.asarray(core.build_cosx_q(basis, grid))

    cells = core.direct_lattice_cells(sys, 7.0)
    caches = core.build_cosx_cell_pair_caches(basis, cells)

    by_index = _cell_lookup(cells)
    M = np.array([[0.3, 0.1], [0.05, 0.25]])
    p_cells = [by_index[(0, 0, 0)]]
    p_blocks = [np.eye(basis.nbasis)]
    for axis in range(3):
        plus = [0, 0, 0]
        plus[axis] = 1
        minus = [0, 0, 0]
        minus[axis] = -1
        p_cells += [by_index[tuple(plus)], by_index[tuple(minus)]]
        p_blocks += [M, M.T]
    P_set = core.make_lattice_matrix_set(basis.nbasis, p_cells, p_blocks)

    K_set = core.compute_cosx_k_blocks(
        basis, P_set, grid, caches, q_cached=q
    )

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 7.0
    ref = core.build_jk_2e_real_space_explicit(
        basis, sys, opts, P_set, cells
    )

    deltas = [
        np.linalg.norm(np.asarray(K_set.blocks[i]) - np.asarray(ref.K.blocks[i]))
        for i in range(len(cells))
    ]
    worst = max(deltas)
    assert worst < 4e-3, (
        f"K(g) blocks vs direct ERI (3D): worst ||ΔK(g)|| = {worst:.2e}"
    )


# ---------------------------------------------------------------------------
# M3b-3: multi-k bridge + k_exchange='cosx' SCF backend
# ---------------------------------------------------------------------------


def _dimerized_chain():
    """Gapped dimerized H2 chain: intra 1.4 / inter 4.6 bohr."""
    atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
    sys = PeriodicSystem(3, np.diag([20.0, 20.0, 6.0]), atoms)
    mol = Molecule(atoms, 0, 1)
    basis = BasisSet(mol, "sto-3g")
    return sys, basis, mol


def test_cosx_kpoint_bridge_vs_direct_eri():
    """M3b-3: K(k) from a converged multi-k density matches the
    direct-ERI fold of the same truncated real-space exchange.

    Measured anchor: ||ΔK(k)|| ≈ 2.5e-4 per k-point (equally-spaced
    H chain, kmesh (1,1,2), GDF-converged density) — pure COSX
    quadrature error; the truncation model is identical on both
    sides by construction.
    """
    from vibeqc.periodic_cosx_k import KPointCosxK
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    sys, basis, mol = _h_chain()
    res = run_krhf_periodic_gdf(
        sys,
        basis,
        kmesh=(1, 1, 2),
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        progress=False,
    )
    assert res.converged

    bridge = KPointCosxK(basis, sys)
    kpts = [np.asarray(k) for k in res.kpoints_cart]
    D_k = [np.asarray(d) for d in res.density]
    K_k = bridge.k_matrices(D_k, kpts)

    P_set = bridge.real_space_density(D_k, kpts)
    opts = LatticeSumOptions()
    ref = core.build_jk_2e_real_space_explicit(
        basis, sys, opts, P_set, bridge.cells
    )
    for i, k in enumerate(kpts):
        K_direct = np.asarray(core.bloch_sum(ref.K, np.asarray(k)))
        K_direct = 0.5 * (K_direct + K_direct.conj().T)
        delta = np.linalg.norm(K_k[i] - K_direct)
        # Quadrature-tier dependent: 2.5e-4 at the XC grid tier,
        # 4.0e-3 at the production COSX tier (the bridge default) —
        # energy-level parity is tier-insensitive (see the SCF tests).
        assert delta < 1e-2, (
            f"bridge K(k[{i}]) vs direct-ERI fold: {delta:.2e} "
            f"(measured 4.0e-3 at the COSX grid tier)"
        )
        # Hermiticity of the folded K.
        assert np.linalg.norm(K_k[i] - K_k[i].conj().T) < 1e-12


def test_krhf_cosx_backend_scf():
    """M3b-3/4b: k_exchange='cosx' SCF on the gapped dimerized chain.

    With the M3b-4b SR+LR composition the COSX K(k) is matrix-level
    exact (~1e-4 vs the independent RSGDF exchange — see
    ``test_cosx_k_composed_vs_rsgdf_and_compcell_flag``). The SCF
    energy still differs from the ``'gdf'`` backend by ~64 mHa at
    (1,1,4) — now understood as the flagged compcell zone-edge
    deviation on padded systems (the 'gdf' backend contracts compcell
    Lpq for BOTH J and K) plus the mixed-gauge Fock (compcell J +
    exact K) on the COSX side. The band documents that measured
    state; it tightens when the GDF route resolves the compcell flag.
    Flag validation also pinned.
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    sys, basis, mol = _dimerized_chain()

    with pytest.raises(ValueError, match="k_exchange"):
        run_krhf_periodic_gdf(
            sys, basis, kmesh=(1, 1, 2), use_compcell=True,
            k_exchange="bogus", progress=False,
        )
    with pytest.raises(ValueError, match="use_compcell"):
        run_krhf_periodic_gdf(
            sys, basis, kmesh=(1, 1, 2), use_compcell=False,
            k_exchange="cosx", progress=False,
        )

    res_gdf = run_krhf_periodic_gdf(
        sys,
        basis,
        kmesh=(1, 1, 4),
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        progress=False,
    )
    res_cosx = run_krhf_periodic_gdf(
        sys,
        basis,
        kmesh=(1, 1, 4),
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        k_exchange="cosx",
        progress=False,
    )
    assert res_gdf.converged and res_cosx.converged
    delta = abs(res_cosx.energy - res_gdf.energy)
    # Measured 0.028 mHa (2026-06-11, Hcore-fold-fixed Fock). Envelope history: 62.3 → 63.9 mHa
    # while the GDF reference ran the flagged q-only compcell builder +
    # mixed-gauge Fock (the b4a6faba-regressed route); the band carried
    # the note "revisit when the compcell flag is resolved by the GDF
    # route". The merge-drop restoration resolved it: gdf_method now
    # defaults to 'rsgdf' (ket-resolved all-FT cderi), making this
    # comparison single-gauge (RSGDF-J on both sides, M3b-4c pairing) —
    # the remaining delta is the genuine COSX-vs-GDF exchange-fit
    # difference (matrix-level ~1e-4 vs RSGDF). Band guards regressions:
    # a jump back to tens of mHa means a gauge split or a broken bridge;
    # exactly 0 would mean backend aliasing.
    assert 1e-6 < delta < 5e-3, (
        f"COSX vs GDF backend at (1,1,4): ΔE = {delta*1e3:.3f} mHa "
        f"(measured 0.028 mHa on the single-gauge default — see docstring)"
    )


# ---------------------------------------------------------------------------
# M3b-4c: single-gauge SCF backend parity (RSGDF-J + {RSGDF, COSX}-K)
# ---------------------------------------------------------------------------


def test_krhf_cosx_backend_scf_single_gauge():
    """M3b-4c production gate: sub-mHa SCF backend parity.

    With ``gdf_method='rsgdf'`` both exchange backends ride the same
    J and share the Bloch pair-FT conventions — the Fock is
    single-gauge and the backend difference isolates the COSX K
    against the RSGDF K at the SCF fixed point. Measured on the
    gapped dimerized chain: ΔE = 0.024 / 0.027 / 0.027 mHa at kmesh
    (1,1,2) / (1,1,4) / (1,1,6), clean convergence through (1,1,8)
    in 6 iterations (the mixed-gauge compcell pairing stalled there
    — see ``test_krhf_cosx_backend_scf``). This test pins the
    (1,1,2) point.
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    sys, basis, mol = _dimerized_chain()
    common = dict(
        kmesh=(1, 1, 2),
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        progress=False,
    )
    res_gdf = run_krhf_periodic_gdf(sys, basis, **common)
    res_cosx = run_krhf_periodic_gdf(sys, basis, k_exchange="cosx", **common)
    assert res_gdf.converged and res_cosx.converged
    delta = abs(res_cosx.energy - res_gdf.energy)
    assert delta < 5e-4, (
        f"single-gauge COSX vs RSGDF backend at (1,1,2): "
        f"ΔE = {delta*1e3:.3f} mHa (measured 0.024 mHa)"
    )


# ---------------------------------------------------------------------------
# M3b-4b: composed SR+LR exchange vs independent references
# ---------------------------------------------------------------------------


def test_cosx_k_composed_vs_rsgdf_and_compcell_flag():
    """M3b-4b acceptance + the flagged compcell zone-edge deviation.

    On the gapped dimerized chain at kmesh (1,1,2), converged density:

    1. **Acceptance:** the composed COSX exchange (real-space erfc-SR
       + reciprocal-space erf-LR + the −π/ω² G=0 finite-part term)
       matches the exchange reconstructed from the independent RSGDF
       Lpq tensors (``build_lpq_bloch_native_fft``) at the DF-fit
       floor (measured ~1e-4 per k). ω-invariance of the composition
       was separately verified at 1e-4 during development.

    2. **Characterisation of a flagged issue (NOT ours):** the
       compcell Lpq exchange (``build_lpq_bloch_compcell`` — what the
       ``k_exchange='gdf'`` backend contracts) deviates from BOTH
       independent routes by ~1.4e-2 at Γ and ~1.5e-1 at the zone
       edge on this vacuum-padded system (ω-invariant,
       cell-list-cutoff-invariant). Escalated to the GDF route
       (handovers/HANDOVER_RIJCOSX_M3A.md § M3b-4b); this assertion documents
       the symptom per CLAUDE.md §7 and should be FLIPPED when the
       compcell fix lands.
    """
    from vibeqc.aux_basis import (
        build_lpq_bloch_compcell,
        build_lpq_bloch_native_fft,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic_cosx_k import KPointCosxK
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    sys, basis, mol = _dimerized_chain()
    aux = make_aux_basis_set(mol, aux_name="def2-universal-jkfit")
    aux_mr = make_modrho_aux_basis(aux, mol)

    res = run_krhf_periodic_gdf(
        sys, basis, kmesh=(1, 1, 2), use_compcell=True,
        aux_basis="def2-universal-jkfit", progress=False,
    )
    assert res.converged
    kpts = [np.asarray(k) for k in res.kpoints_cart]
    D_k = [np.asarray(d) for d in res.density]
    n_k = len(kpts)
    n_bf = basis.nbasis

    def k_from_lpq(builder):
        K = [np.zeros((n_bf, n_bf), dtype=complex) for _ in range(n_k)]
        for i in range(n_k):
            for j in range(n_k):
                L = builder(i, j)
                tmp = np.einsum("Lpr,rs->Lps", L, D_k[j])
                K[i] += (1.0 / n_k) * np.einsum(
                    "Lps,Lqs->pq", tmp, L.conj()
                )
        return [0.5 * (Kk + Kk.conj().T) for Kk in K]

    K_rs = k_from_lpq(
        lambda i, j: build_lpq_bloch_native_fft(
            sys, basis, aux_mr, kpts[i], kpts[j],
            ke_cutoff=200.0, linear_dep_thr=1e-9,
        )
    )
    K_cc = k_from_lpq(
        lambda i, j: build_lpq_bloch_compcell(
            sys, basis, aux, kpts[j] - kpts[i], molecule=mol,
            linear_dep_thr=1e-9,
        )
    )

    bridge = KPointCosxK(basis, sys, omega=0.7)
    K_comp = bridge.k_matrices(D_k, kpts, lr_complement=True)

    # 1. Acceptance: composed COSX vs RSGDF. Quadrature-tier
    #    dependent: ~1e-4 at the XC grid tier (DF-fit floor),
    #    ~1e-3 at the production COSX tier (the bridge default).
    for i in range(n_k):
        delta = np.linalg.norm(K_comp[i] - K_rs[i])
        assert delta < 5e-3, (
            f"composed COSX K vs RSGDF K at k[{i}]: {delta:.2e} "
            f"(measured ~1e-3 at the COSX grid tier)"
        )

    # 2. Flagged compcell deviation (zone-edge k) — documents the
    #    symptom; flip when the GDF route fixes compcell.
    zone_edge = np.linalg.norm(K_rs[1] - K_cc[1])
    assert zone_edge > 0.05, (
        f"compcell zone-edge deviation vanished ({zone_edge:.2e}) — "
        "the GDF-route fix may have landed; update this test and the "
        "k_exchange='cosx' warning + handover § M3b-4b."
    )


def test_cosx_sr_envelope_warning_lih():
    """M3b-6: the SR-envelope criterion fires on ultra-diffuse bases.

    LiH/sto-3g carries Li 2sp extents of 50-56 bohr (α_min = 0.048) —
    outside the real-space SR exchange envelope (measured: ω-invariance
    broken at 2e-2 (15-bohr domain) and 3.6e-2 (40-bohr capped domain);
    no feasible cell domain covers the basis). The bridge must derive
    the domain, cap it, and warn — pointing at the k-space GDF route /
    periodic-adapted bases.
    """
    import warnings

    from vibeqc.periodic_cosx_k import KPointCosxK

    a = 4.084 / 0.529177210903
    A = (a / 2) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    sys_l = PeriodicSystem(
        3, A, [Atom(3, [0, 0, 0]), Atom(1, [a / 2, a / 2, a / 2])]
    )
    basis = BasisSet(sys_l.unit_cell_molecule(), "sto-3g")

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        bridge = KPointCosxK(basis, sys_l, omega=0.9)
    envelope_warnings = [
        w for w in wlist if "ultra-diffuse basis" in str(w.message)
    ]
    assert envelope_warnings, "SR-envelope warning did not fire on LiH"
    assert bridge.sr_cell_cutoff_bohr <= KPointCosxK.SR_CELL_RADIUS_CAP

    # Compact bases stay un-warned and keep at least the one-electron
    # domain (the chain anchor's H/sto-3g: r(1e-4) = 7.4 bohr).
    sys_c, basis_c, _ = _h_chain()
    with warnings.catch_warnings(record=True) as wlist2:
        warnings.simplefilter("always")
        bridge_c = KPointCosxK(basis_c, sys_c, omega=0.9)
    assert not [
        w for w in wlist2 if "ultra-diffuse basis" in str(w.message)
    ]
    assert bridge_c.sr_cell_cutoff_bohr >= 15.0


def test_compcell_q_only_lih_flag():
    """Characterisation of the flagged compcell q-only limitation on a
    tight ionic crystal (LiH primitive FCC) — NOT a COSX issue.

    ``build_lpq_bloch_compcell`` is q-resolved only (signature takes
    the momentum transfer, not (k_bra, k_ket)) — the architecture the
    RSGDF builder's own docstring warns is "wrong by hundreds of Ha"
    on tight ionic cells, exact only in the no-inter-cell-overlap
    limit. Measured SCF consequences (2026-06-10, this is what the
    ``k_exchange='gdf'`` default contracts):

        LiH/sto-3g compcell (2,2,2): E = −64318 Ha, converged=True (!)
        LiH/sto-3g compcell (1,1,2): E = +3461 Ha, not converged
        (physical scale: ≈ −7.9 Ha; the padded-chain zone-edge
        deviation flagged in § M3b-4b is the MILD form of the same
        root cause)

    This test pins the symptom at the Lpq level (no SCF): the
    exchange contraction of the compcell q-only tensor at a q ≠ 0
    pair deviates from the (k_bra, k_ket)-resolved RSGDF tensor at
    O(1) relative on LiH. Escalated to the GDF route
    (handovers/HANDOVER_RIJCOSX_M3A.md § M3b-5); FLIP this assertion when the
    (k_bra, k_ket)-resolved compcell builder lands.
    """
    from vibeqc.aux_basis import (
        build_lpq_bloch_compcell,
        build_lpq_bloch_native_fft,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )

    a = 4.084 / 0.529177210903
    A = (a / 2) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    sys_l = PeriodicSystem(
        3, A, [Atom(3, [0, 0, 0]), Atom(1, [a / 2, a / 2, a / 2])]
    )
    mol = sys_l.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    aux = make_aux_basis_set(mol, aux_name="def2-universal-jkfit")
    aux_mr = make_modrho_aux_basis(aux, mol)

    k0 = np.zeros(3)
    k1 = np.array([0.0, 0.0, 0.4])  # a generic q ≠ 0 transfer
    D = np.eye(basis.nbasis)

    L_cc = build_lpq_bloch_compcell(
        sys_l, basis, aux, k1 - k0, molecule=mol, linear_dep_thr=1e-9
    )
    L_rs = build_lpq_bloch_native_fft(
        sys_l, basis, aux_mr, k0, k1, ke_cutoff=100.0,
        linear_dep_thr=1e-9,
    )

    def k_contract(L):
        tmp = np.einsum("Lpr,rs->Lps", L, D)
        K = np.einsum("Lps,Lqs->pq", tmp, L.conj())
        return 0.5 * (K + K.conj().T)

    K_cc = k_contract(L_cc)
    K_rs = k_contract(L_rs)
    rel = np.linalg.norm(K_cc - K_rs) / np.linalg.norm(K_rs)
    assert rel > 0.5, (
        f"compcell q-only deviation on LiH vanished (rel = {rel:.2e}) "
        "— the (k_bra, k_ket)-resolved compcell fix may have landed; "
        "update this test, the k_exchange warning, and the handover."
    )


# ---------------------------------------------------------------------------
# M3b-4a: erfc-SR exchange blocks vs the erfc direct-ERI reference
# ---------------------------------------------------------------------------


def test_cosx_k_blocks_erfc_sr_vs_direct_eri():
    """M3b-4a: short-range (erfc) exchange blocks, ERI-exact.

    Same harness as the full-Coulomb block test, with ω = 0.6 on both
    sides: the COSX caches/kernel use the erfc-SR Coulomb and the
    direct-ERI reference runs libint's ``erfc_coulomb``. The residual
    is COSX grid-quadrature error. SR localisation is pinned through
    the far K(g) blocks dying (kernel range 1/ω ≈ 1.7 bohr) — NOT
    through the δ-set size: the Schwarz factor is the pair density's
    self-interaction and decays with |δ| through the overlap only.
    """
    OMEGA = 0.6
    sys, basis, mol = _h_chain()
    grid = build_grid(mol, vq.GridOptions())
    q = np.asarray(core.build_cosx_q(basis, grid))

    cells = core.direct_lattice_cells(sys, 15.0)
    caches_sr = core.build_cosx_cell_pair_caches(
        basis, cells, 1e-10, OMEGA
    )
    assert caches_sr.omega == OMEGA

    P_set = _chain_p_set(basis, cells)
    K_set = core.compute_cosx_k_blocks(
        basis, P_set, grid, caches_sr, q_cached=q
    )

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    ref = core.build_jk_2e_real_space_explicit(
        basis, sys, opts, P_set, cells, omega=OMEGA
    )

    deltas = [
        np.linalg.norm(np.asarray(K_set.blocks[i]) - np.asarray(ref.K.blocks[i]))
        for i in range(len(cells))
    ]
    worst = max(deltas)
    assert worst < 2e-3, (
        f"SR K(g) blocks vs erfc direct ERI: worst ||ΔK(g)|| = {worst:.2e}"
    )

    # SR decay (measured at ω = 0.6, vs the full kernel in
    # parentheses): |K(2 cells)| = 5.2e-3 (full: 0.143),
    # |K(3 cells)| = 2.3e-5 (full: 1.0e-2) — the steep Gaussian-like
    # locality of the erfc coupling, softened only by the
    # pair-density tails. Bands at ~4-10× headroom.
    block_norm = {}
    for i, c in enumerate(cells):
        block_norm[tuple(np.asarray(c.index))] = np.linalg.norm(
            np.asarray(K_set.blocks[i])
        )
    assert block_norm[(0, 0, 2)] < 2e-2
    assert block_norm[(0, 0, 3)] < 2e-4


def test_rijcosx_pbc_tight_k_gamma_parity():
    """M3b cell-pair COSX closes the tight-cell Γ exchange gap.

    This is the former M3a xfail promoted to an active regression.  On
    the tight H-chain, the single-lattice-sum image-cell kernel is a
    different exchange model and remains far from the Γ-folded direct
    ERI reference.  The cell-pair double-sum kernel carries the missing
    Γ density-coupled exchange classes and matches the same direct-ERI
    reference at COSX quadrature accuracy.
    """
    sys, basis, mol = _h_chain()
    D = np.eye(basis.nbasis)
    grid = build_grid(mol, vq.GridOptions())
    q = core.build_cosx_q(basis, grid)

    cells = core.direct_lattice_cells(sys, 15.0)
    assert len(cells) == 7
    caches = core.build_cosx_cell_pair_caches(basis, cells)

    K_single = np.asarray(
        core.compute_cosx_k(basis, D, grid, lattice=sys.lattice, image_cells=cells)
    )
    K_cell_pair = np.asarray(
        core.compute_cosx_k_cell_pair(basis, D, grid, caches, q_cached=q)
    )

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    K_ref = np.asarray(
        core.build_jk_gamma_molecular_limit(basis, sys, opts, D).K
    )

    single_gap = np.linalg.norm(K_single - K_ref)
    cell_pair_gap = np.linalg.norm(K_cell_pair - K_ref)
    assert single_gap > 0.5, (
        "single-lattice-sum COSX unexpectedly matched the Γ-folded "
        f"reference: ||ΔK|| = {single_gap:.2e}"
    )
    assert cell_pair_gap < 2e-3, (
        f"cell-pair COSX vs Γ-folded direct-ERI: ||ΔK|| = {cell_pair_gap:.2e}; "
        f"single-sum gap was {single_gap:.2e}"
    )


# ---------------------------------------------------------------------------
# Multi-k one-center correction (G-PBC-005 sub-gate, 2026-07-18)
# ---------------------------------------------------------------------------


def test_cosx_one_center_correction_kernel_match():
    """The correction's analytic and quadrature sides share one kernel.

    ``compute_cosx_one_center_correction`` returns
    K_exact^(1c) − K_quad^(1c) — pure same-atom quadrature error — so it
    must (a) shrink substantially on a denser grid and (b) be continuous
    at the erfc/coulomb branch point (ω → 0 reduces erfc(ω·r)/r to the
    full kernel). A kernel mismatch between the analytic side (libint
    ``erfc_coulomb``) and the quadrature side (M3b-4a attenuated
    seeding) would leave a dense-grid residual and break (a).
    """
    _, basis, mol = _h2_box(50.0)
    D = np.array([[1.0, 0.2], [0.2, 0.5]])
    omega = 0.6

    coarse_opts = vq.GridOptions()
    coarse_opts.n_radial = 35
    coarse_opts.n_theta = 9
    coarse_opts.n_phi = 18
    dense_opts = vq.GridOptions()
    dense_opts.n_radial = 90
    dense_opts.n_theta = 21
    dense_opts.n_phi = 42
    grid_coarse = build_grid(mol, coarse_opts)
    grid_dense = build_grid(mol, dense_opts)

    corr_coarse = np.asarray(
        core.compute_cosx_one_center_correction(
            basis, D, grid_coarse, omega
        )
    )
    corr_dense = np.asarray(
        core.compute_cosx_one_center_correction(basis, D, grid_dense, omega)
    )

    n_coarse = np.linalg.norm(corr_coarse)
    n_dense = np.linalg.norm(corr_dense)
    assert n_coarse > 1e-10, (
        "one-center correction unexpectedly zero on the COSX-tier grid"
    )
    assert n_dense < 0.5 * n_coarse, (
        "one-center correction did not shrink on a denser grid — the "
        "analytic and quadrature kernels disagree: "
        f"coarse {n_coarse:.3e}, dense {n_dense:.3e}"
    )
    assert np.linalg.norm(corr_coarse - corr_coarse.T) < 1e-10

    # erfc/coulomb branch continuity: ω → 0 must reduce to the full
    # kernel evaluated by the ω = 0 (Operator::coulomb) branch.
    corr_full = np.asarray(
        core.compute_cosx_one_center_correction(basis, D, grid_coarse, 0.0)
    )
    corr_eps = np.asarray(
        core.compute_cosx_one_center_correction(
            basis, D, grid_coarse, 1e-8
        )
    )
    assert np.linalg.norm(corr_eps - corr_full) < 1e-6, (
        "erfc branch does not reduce to the Coulomb branch at ω → 0: "
        f"||Δ|| = {np.linalg.norm(corr_eps - corr_full):.3e}"
    )


def test_kpoint_bridge_one_center_correction_folds_d0():
    """``KPointCosxK.one_center_correction`` contracts the folded D(g=0)
    block ((1/N_k)·Σ_k D(k), TRS-projected real) with the bridge's own
    grid and ω."""
    from vibeqc.periodic_cosx_k import KPointCosxK

    sys, basis, mol = _dimerized_chain()
    bridge = KPointCosxK(basis, sys, omega=0.7)

    rng = np.random.default_rng(7)
    A = rng.standard_normal((2, 2))
    D0 = A + A.T
    phase = rng.standard_normal((2, 2)) * 1j
    # A k/−k pair with conjugate imaginary parts: the fold's TRS
    # projection keeps the shared real part.
    D_k = [D0 + (phase + phase.conj().T), D0 + (phase + phase.conj().T).conj()]

    def _sym(M):
        return 0.5 * (M + M.T)

    corr = bridge.one_center_correction(D_k)
    ref = _sym(
        np.asarray(
            core.compute_cosx_one_center_correction(
                basis, np.ascontiguousarray(D0), bridge.grid, bridge.omega
            )
        )
    )
    ref_fold = _sym(
        np.asarray(
            core.compute_cosx_one_center_correction(
                basis,
                np.ascontiguousarray(
                    sum(np.asarray(D) for D in D_k).real / 2.0
                ),
                bridge.grid,
                bridge.omega,
            )
        )
    )
    assert np.linalg.norm(corr - ref_fold) < 1e-12
    assert np.linalg.norm(corr) > 1e-10
    assert np.linalg.norm(corr - corr.T) < 1e-14
    # Single-density fold is the density itself.
    corr_single = bridge.one_center_correction([D0])
    assert np.linalg.norm(corr_single - ref) < 1e-12


def test_krhf_cosx_one_center_lifecycle(monkeypatch):
    """Post-convergence one-center replacement in the multi-k driver.

    Mirrors the molecular / dedicated-Gamma-builder lifecycle
    (commit 40dad042 + the 2026-07-15 Gamma increment): with
    ``k_exchange='cosx'`` the converged ``run_krhf_periodic_gdf``

    1. reports the energy of the UNCORRECTED iterated K surface
       (identical to a run with the correction forced to zero), and
    2. returns Fock matrices and orbitals upgraded by exactly
       ``−0.5·α·C1`` at every k, where ``C1`` is the bridge's
       one-center correction on the converged density.
    """
    from vibeqc.periodic_cosx_k import KPointCosxK
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    sys, basis, mol = _dimerized_chain()
    common = dict(
        kmesh=(1, 1, 2),
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        k_exchange="cosx",
        progress=False,
    )

    # Reference run: correction forced to zero — the pre-fix behavior.
    def _zero_correction(self, D_k, **kwargs):
        return np.zeros((self.basis.nbasis, self.basis.nbasis))

    real_correction = KPointCosxK.one_center_correction
    monkeypatch.setattr(
        KPointCosxK, "one_center_correction", _zero_correction
    )
    res_plain = run_krhf_periodic_gdf(sys, basis, **common)
    assert res_plain.converged

    # Corrected run: record the bridge instance + the applied C1.
    recorded = {}

    def _recording_correction(self, D_k, **kwargs):
        C1 = real_correction(self, D_k, **kwargs)
        recorded["C1"] = C1
        return C1

    monkeypatch.setattr(
        KPointCosxK, "one_center_correction", _recording_correction
    )
    res_corr = run_krhf_periodic_gdf(sys, basis, **common)
    assert res_corr.converged
    assert "C1" in recorded, "converged run never applied the correction"
    C1 = recorded["C1"]
    assert np.linalg.norm(C1) > 1e-10

    # (1) The energy stays on the uncorrected iterated surface.
    assert abs(res_corr.energy - res_plain.energy) < 1e-10
    # Identical SCF trajectory: the correction is post-convergence only.
    assert res_corr.n_iter == res_plain.n_iter
    for Dc, Dp in zip(res_corr.density, res_plain.density):
        assert np.linalg.norm(np.asarray(Dc) - np.asarray(Dp)) < 1e-10

    # (2) The returned Fock carries exactly −0.5·α·C1 (α = 1, HF) at
    # every k; the orbitals were re-diagonalised from it.
    dF_expected = -0.5 * C1
    for i, (Fc, Fp) in enumerate(zip(res_corr.fock, res_plain.fock)):
        dF = np.asarray(Fc) - np.asarray(Fp)
        assert np.linalg.norm(dF - dF_expected) < 1e-10, (
            f"k-point {i}: returned Fock delta does not equal "
            "−0.5·α·C1"
        )
    mo_delta = max(
        float(np.max(np.abs(np.asarray(ec) - np.asarray(ep))))
        for ec, ep in zip(res_corr.mo_energies, res_plain.mo_energies)
    )
    assert mo_delta > 1e-10, (
        "orbital energies were not re-diagonalised from the corrected "
        "Fock"
    )


def test_kuhf_cosx_one_center_lifecycle(monkeypatch):
    """Open-shell sibling of the RHF lifecycle pin: per-spin Focks carry
    ``−α·C1(D_spin)`` post-convergence; the energy stays on the
    uncorrected iterated surface (per-spin exchange enters the
    open-shell Fock as ``−α·K_spin``)."""
    from vibeqc.periodic_cosx_k import KPointCosxK
    from vibeqc.periodic_k_gdf import run_kuhf_periodic_gdf

    sys, basis, mol = _dimerized_chain()
    common = dict(
        kmesh=(1, 1, 2),
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        k_exchange="cosx",
        progress=False,
    )

    def _zero_correction(self, D_k, **kwargs):
        return np.zeros((self.basis.nbasis, self.basis.nbasis))

    real_correction = KPointCosxK.one_center_correction
    monkeypatch.setattr(
        KPointCosxK, "one_center_correction", _zero_correction
    )
    res_plain = run_kuhf_periodic_gdf(sys, basis, **common)
    assert res_plain.converged

    recorded = []

    def _recording_correction(self, D_k, **kwargs):
        C1 = real_correction(self, D_k, **kwargs)
        recorded.append(C1)
        return C1

    monkeypatch.setattr(
        KPointCosxK, "one_center_correction", _recording_correction
    )
    res_corr = run_kuhf_periodic_gdf(sys, basis, **common)
    assert res_corr.converged
    assert len(recorded) == 2, (
        "expected one per-spin correction pair at convergence, got "
        f"{len(recorded)} calls"
    )
    C1_a, C1_b = recorded

    assert abs(res_corr.energy - res_plain.energy) < 1e-10
    for spin, (focks_c, focks_p, C1) in enumerate(
        (
            (res_corr.fock_alpha, res_plain.fock_alpha, C1_a),
            (res_corr.fock_beta, res_plain.fock_beta, C1_b),
        )
    ):
        dF_expected = -1.0 * C1
        for i, (Fc, Fp) in enumerate(zip(focks_c, focks_p)):
            dF = np.asarray(Fc) - np.asarray(Fp)
            assert np.linalg.norm(dF - dF_expected) < 1e-10, (
                f"spin {spin} k-point {i}: Fock delta != −α·C1"
            )


def test_rijcosx_periodic_job_cites_cosx(monkeypatch, tmp_path):
    """A public periodic RIJCOSX job fires the COSX acceleration
    citations (Neese 2009, Izsak-Neese 2011, Helmich-Paris 2021) into
    the references surface. Before 2026-07-18 the runner only passed
    ``acceleration=('rijcosx',)`` for the AICCM2026DEV_B rijcosx
    backend, silently dropping the attribution on the dedicated
    RIJCOSX route (CLAUDE.md section 8)."""
    from vibeqc.periodic_k_gdf import PeriodicKRHFGDFResult
    from vibeqc.periodic_runner import run_periodic_job

    sys, basis, _ = _h2_box(12.0)

    def fake_run_krhf(_system, _basis, kmesh, _options, **kwargs):
        nbf = _basis.nbasis
        n_k = 2
        zeros = [np.zeros((nbf, nbf), dtype=complex) for _ in range(n_k)]
        density = [np.diag([2.0] + [0.0] * (nbf - 1)).astype(complex)
                   for _ in range(n_k)]
        return PeriodicKRHFGDFResult(
            energy=-1.0,
            e_electronic=-1.5,
            e_nuclear=0.5,
            n_iter=1,
            converged=True,
            mo_energies=[np.linspace(-0.5, 0.5, nbf) for _ in range(n_k)],
            mo_coeffs=[np.eye(nbf, dtype=complex) for _ in range(n_k)],
            fock=zeros,
            overlap=[np.eye(nbf, dtype=complex) for _ in range(n_k)],
            hcore=zeros,
            density=density,
            kpoints_cart=np.zeros((n_k, 3), dtype=float),
            kpoint_weights=np.full(n_k, 1.0 / n_k),
            free_energy=-1.0,
            occupations=[np.array([2.0] + [0.0] * (nbf - 1))
                         for _ in range(n_k)],
            aux_basis_name="def2-universal-jkfit",
            n_aux=1,
            backend="native-multi-k-gdf-cosx-rhf",
        )

    monkeypatch.setattr("vibeqc.periodic_runner.run_krhf_periodic_gdf",
                        fake_run_krhf)

    run_periodic_job(
        sys,
        basis,
        method="RHF",
        jk_method="rijcosx",
        kpoints=(2, 1, 1),
        aux_basis="def2-universal-jkfit",
        output=tmp_path / "rijcosx-cites",
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=True,
        output_qvf=False,
        progress=False,
    )

    refs_text = (tmp_path / "rijcosx-cites.references").read_text()
    for key in (
        "neese_rijcosx_2009",
        "izsak_neese_cosx_2011",
        "helmich_paris_cosx_2021",
    ):
        assert key in refs_text, f"missing COSX citation {key}"


# ---------------------------------------------------------------------------
# Weighted / nonuniform full-BZ k-meshes (G-PBC-005 sub-gate, 2026-07-18)
# ---------------------------------------------------------------------------


def _explicit_kpoints(system, pts, ws, mesh):
    """Explicit full-BZ KPoints with declared mesh metadata."""
    from vibeqc.kpoints import KPoints

    lat = np.asarray(system.lattice, dtype=float)
    B = 2.0 * np.pi * np.linalg.inv(lat).T
    pts = np.asarray(pts, dtype=float)
    frac = np.linalg.solve(B, pts.T).T
    return KPoints(
        kpoints_cart=pts,
        kpoints_frac=frac,
        weights=np.asarray(ws, dtype=float),
        kind="explicit",
        mesh=mesh,
        _system=system,
    )


def test_cosx_bridge_weighted_fold_matches_duplicated_uniform():
    """Weight plumbing exactness oracle at the bridge level: a weighted
    k-list is identical to the equivalent duplicated uniform list for
    the D(g) fold, the composed K(k) (SR + LR complement incl. the
    q = 0 G = 0 finite part), and the one-center-correction D(g=0)
    fold."""
    from vibeqc.periodic_cosx_k import KPointCosxK

    sys_c, basis, mol = _dimerized_chain()
    lat = np.asarray(sys_c.lattice, dtype=float)
    B = 2.0 * np.pi * np.linalg.inv(lat).T
    gamma = np.zeros(3)
    X = 0.5 * B[:, 2]

    rng = np.random.default_rng(11)
    A1 = rng.standard_normal((2, 2))
    A2 = rng.standard_normal((2, 2))
    D1 = A1 + A1.T
    D2 = A2 + A2.T

    bridge = KPointCosxK(basis, sys_c, omega=5.0 / 6.0)

    kw = [gamma, X]
    kd = [gamma, X, X]
    w = [1.0 / 3.0, 2.0 / 3.0]

    P_w = bridge.real_space_density([D1, D2], kw, weights=w)
    P_d = bridge.real_space_density([D1, D2, D2], kd)
    for a, b in zip(P_w.blocks, P_d.blocks):
        assert np.linalg.norm(np.asarray(a) - np.asarray(b)) < 1e-13

    K_w = bridge.k_matrices([D1, D2], kw, lr_complement=True, weights=w)
    K_d = bridge.k_matrices([D1, D2, D2], kd, lr_complement=True)
    assert np.linalg.norm(K_w[0] - K_d[0]) < 1e-12
    assert np.linalg.norm(K_w[1] - K_d[1]) < 1e-12
    assert np.linalg.norm(K_w[1] - K_d[2]) < 1e-12

    C_w = bridge.one_center_correction([D1, D2], weights=w)
    C_d = bridge.one_center_correction([D1, D2, D2])
    assert np.linalg.norm(C_w - C_d) < 1e-13

    # Weight validation fails closed.
    with pytest.raises(ValueError):
        bridge.real_space_density([D1, D2], kw, weights=[0.9, 0.2])
    with pytest.raises(ValueError):
        bridge.real_space_density([D1, D2], kw, weights=[1.0])


def test_krhf_cosx_weighted_kpoints_scf():
    """Weighted full-BZ meshes through the multi-k COSX backend
    (G-PBC-005 weighted-mesh sub-gate) + the explicit-KPoints exxdiv
    Madelung fix.

    Measured on the gapped dimerized chain (2026-07-18):

    1. An explicit uniform [Gamma, X] KPoints (declared mesh (1,1,2))
       equals the tuple (1,1,2) run to machine precision. Pre-fix the
       explicit path silently used the primitive-cell Madelung
       (``to_bloch_kmesh`` reports mesh (1,1,1)) and sat 78 mHa off.
    2. A weighted {Gamma 1/3, X 2/3} quadrature equals the duplicated
       uniform [Gamma, X, X] run to machine precision (the weighted
       D(g)/LR plumbing exactness oracle at the SCF fixed point).
    3. The weighted mesh agrees with the k_exchange='gdf' backend on
       the same quadrature at the COSX backend-parity band
       (measured 0.128 mHa; band 5e-4 as in the single-gauge gate).
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    sys_c, basis, mol = _dimerized_chain()
    lat = np.asarray(sys_c.lattice, dtype=float)
    B = 2.0 * np.pi * np.linalg.inv(lat).T
    gamma = np.zeros(3)
    X = 0.5 * B[:, 2]

    common = dict(
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        k_exchange="cosx",
        progress=False,
    )
    r_ref = run_krhf_periodic_gdf(sys_c, basis, kmesh=(1, 1, 2), **common)
    assert r_ref.converged

    kp_unif = _explicit_kpoints(sys_c, [gamma, X], [0.5, 0.5], (1, 1, 2))
    r_expl = run_krhf_periodic_gdf(sys_c, basis, kmesh=kp_unif, **common)
    assert r_expl.converged
    assert abs(r_expl.energy - r_ref.energy) < 1e-10

    kp_w = _explicit_kpoints(
        sys_c, [gamma, X], [1.0 / 3.0, 2.0 / 3.0], (1, 1, 2)
    )
    kp_dup = _explicit_kpoints(
        sys_c, [gamma, X, X], [1.0 / 3.0] * 3, (1, 1, 2)
    )
    r_w = run_krhf_periodic_gdf(sys_c, basis, kmesh=kp_w, **common)
    r_dup = run_krhf_periodic_gdf(sys_c, basis, kmesh=kp_dup, **common)
    assert r_w.converged and r_dup.converged
    assert abs(r_w.energy - r_dup.energy) < 1e-10

    common_gdf = dict(common)
    common_gdf["k_exchange"] = "gdf"
    r_wg = run_krhf_periodic_gdf(sys_c, basis, kmesh=kp_w, **common_gdf)
    assert r_wg.converged
    assert abs(r_w.energy - r_wg.energy) < 5e-4


# ---------------------------------------------------------------------------
# Gamma RKS RIJCOSX (vacuum-padded envelope) — G-PBC-005 sub-gate, 2026-07-18
# ---------------------------------------------------------------------------


def test_rijcosx_gamma_rks_molecular_limit():
    """Gamma RIJCOSX RKS on a vacuum-padded box matches molecular RKS.

    Measured (H2/STO-3G, 50-bohr box, def2-universal-jkfit):
    pbe0 delta 4.7e-5 Ha, pbe delta 4.8e-5 Ha vs vibe-qc molecular RKS
    -- the residual is the shared box-limit/DF piece (identical scale
    with and without exact exchange, so the COSX K adds no error above
    it). Gate at 2e-4."""
    from vibeqc.periodic_rijcosx import run_periodic_rijcosx_rks

    sys50, basis, mol = _h2_box(50.0)

    for functional in ("pbe0", "pbe"):
        r = run_periodic_rijcosx_rks(
            sys50,
            basis,
            functional=functional,
            aux_basis="def2-universal-jkfit",
            progress=False,
        )
        assert r.converged, functional
        opts = vq.RKSOptions()
        opts.functional = functional
        m = vq.run_rks(mol, basis, opts)
        assert m.converged, functional
        assert abs(r.energy - m.energy) < 2e-4, (
            f"{functional}: |dE| = {abs(r.energy - m.energy):.3e}"
        )

    # Tight/short-axis cells fail closed (XC needs the periodic grid).
    sys_chain, basis_chain, _ = _dimerized_chain()
    with pytest.raises(NotImplementedError, match="vacuum-padded"):
        run_periodic_rijcosx_rks(
            sys_chain, basis_chain, functional="pbe0", progress=False
        )


def test_rijcosx_gamma_rks_public_route(tmp_path):
    """The public runner reaches Gamma RIJCOSX RKS on a vacuum-padded
    cell and agrees with the direct driver."""
    from vibeqc.periodic_rijcosx import run_periodic_rijcosx_rks
    from vibeqc.periodic_runner import run_periodic_job

    sys50, basis, mol = _h2_box(50.0)

    r_pub = run_periodic_job(
        sys50,
        basis,
        method="RKS",
        functional="pbe0",
        jk_method="rijcosx",
        kpoints=(1, 1, 1),
        aux_basis="def2-universal-jkfit",
        output=tmp_path / "rijcosx_rks_box",
        write_molden_file=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        output_qvf=False,
        citations=False,
    )
    assert r_pub.converged
    r_drv = run_periodic_rijcosx_rks(
        sys50,
        basis,
        functional="pbe0",
        aux_basis="def2-universal-jkfit",
        progress=False,
    )
    assert abs(r_pub.energy - r_drv.energy) < 1e-10


def test_rijcosx_gamma_uhf_uks_molecular_limit():
    """Gamma RIJCOSX UHF/UKS on a vacuum-padded box match the molecular
    RIJCOSX oracle (apples-to-apples: same DF-J + COSX-K approximation).

    Measured (triplet H2/STO-3G, 50-bohr box, def2-universal-jkfit):
    periodic UHF vs molecular cosx=True UHF 2.0e-5 Ha (the shared
    box-limit/DF floor of the RKS rung); vs EXACT molecular UHF
    1.07e-3 Ha -- and the molecular RIJCOSX itself sits 1.10e-3 from
    exact on this open-shell anchor, so the deviation is the
    method-inherent COSX quadrature error, not a periodic defect."""
    from vibeqc.periodic_rijcosx import (
        run_periodic_rijcosx_uhf,
        run_periodic_rijcosx_uks,
    )

    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    mol = vq.Molecule(atoms, 0, 3)
    basis = vq.BasisSet(mol, "sto-3g")
    sys50 = vq.PeriodicSystem(
        3, np.diag([50.0, 50.0, 50.0]), atoms, multiplicity=3
    )

    r_uhf = run_periodic_rijcosx_uhf(
        sys50, basis, aux_basis="def2-universal-jkfit", progress=False
    )
    assert r_uhf.converged
    o = vq.UHFOptions()
    o.density_fit = True
    o.aux_basis = "def2-universal-jkfit"
    o.cosx = True
    m_cosx = vq.run_uhf(mol, basis, o)
    assert abs(r_uhf.energy - m_cosx.energy) < 2e-4
    m_exact = vq.run_uhf(mol, basis)
    assert abs(r_uhf.energy - m_exact.energy) < 2e-3

    r_uks = run_periodic_rijcosx_uks(
        sys50,
        basis,
        functional="pbe0",
        aux_basis="def2-universal-jkfit",
        progress=False,
    )
    assert r_uks.converged
    ok = vq.UKSOptions()
    ok.functional = "pbe0"
    mk = vq.run_uks(mol, basis, ok)
    assert abs(r_uks.energy - mk.energy) < 2e-3

    # Short-axis cells fail closed for both open-shell drivers.
    sys_chain = vq.PeriodicSystem(
        3, np.diag([20.0, 20.0, 6.0]), atoms, multiplicity=3
    )
    with pytest.raises(NotImplementedError, match="vacuum-padded"):
        run_periodic_rijcosx_uhf(sys_chain, basis, progress=False)
    with pytest.raises(NotImplementedError, match="vacuum-padded"):
        run_periodic_rijcosx_uks(
            sys_chain, basis, functional="pbe0", progress=False
        )


# ---------------------------------------------------------------------------
# Screened-hybrid (HSE-type) multi-k COSX exchange — G-PBC-005, 2026-07-19
# ---------------------------------------------------------------------------


def test_cosx_screened_k_matches_erfc_direct_eri():
    """The screened-exchange composition is ERI-exact and split-invariant.

    ``k_matrices(screened_omega=w_s)`` must reproduce the physical
    erfc(w_s r)/r exchange of the direct-ERI reference
    (``build_jk_2e_real_space_explicit(omega=w_s)``, Bloch-folded) for
    BOTH branches: the SR(split) + reciprocal band composition and the
    pure real-space evaluation (bridge omega == w_s). Measured on the
    dimerized chain at w_s = 0.6 (2026-07-19): composed 5.6e-4, pure
    1.1e-3, composed-vs-pure 5.8e-4, split-invariance (w_p 1.2 vs 0.9)
    2.3e-4 — all at the COSX quadrature floor (||K_ref|| ~ 0.88)."""
    from vibeqc.periodic_cosx_k import KPointCosxK

    sys_c, basis, mol = _dimerized_chain()
    lat = np.asarray(sys_c.lattice, dtype=float)
    B = 2.0 * np.pi * np.linalg.inv(lat).T
    kpts = [np.zeros(3), 0.5 * B[:, 2]]

    rng = np.random.default_rng(3)
    A1 = rng.standard_normal((2, 2))
    A2 = rng.standard_normal((2, 2))
    D_k = [A1 + A1.T, A2 + A2.T]
    W_S = 0.6

    br = KPointCosxK(basis, sys_c, omega=1.2)
    K_comp = br.k_matrices(
        D_k, kpts, lr_complement=True, screened_omega=W_S
    )
    br_pure = KPointCosxK(basis, sys_c, omega=W_S)
    K_pure = br_pure.k_matrices(
        D_k, kpts, lr_complement=True, screened_omega=W_S
    )

    P_set = br.real_space_density(D_k, kpts)
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    cells = core.direct_lattice_cells(sys_c, 15.0)
    ref = core.build_jk_2e_real_space_explicit(
        basis, sys_c, opts, P_set, cells, omega=W_S
    )
    K_ref = [np.asarray(core.bloch_sum(ref.K, k)) for k in kpts]
    K_ref = [0.5 * (K + K.conj().T) for K in K_ref]

    for i in range(2):
        assert np.linalg.norm(K_comp[i] - K_ref[i]) < 2e-3
        assert np.linalg.norm(K_pure[i] - K_ref[i]) < 4e-3

    # Split invariance: a different numerical split, same physics.
    br3 = KPointCosxK(basis, sys_c, omega=0.9)
    K_c3 = br3.k_matrices(
        D_k, kpts, lr_complement=True, screened_omega=W_S
    )
    worst = max(np.linalg.norm(K_comp[i] - K_c3[i]) for i in range(2))
    assert worst < 1e-3, f"split-invariance broke: {worst:.3e}"

    # Guard: screening longer-ranged than the split raises.
    with pytest.raises(ValueError, match="must not exceed"):
        br3.k_matrices(D_k, kpts, lr_complement=True, screened_omega=1.5)


def test_krks_cosx_hse06_screened_scf():
    """HSE06 through the multi-k COSX backend (G-PBC-005 sub-gate).

    Measured (2026-07-19):

    * vacuum 25-bohr box (1,1,2) vs vibe-qc molecular hse06: 5.1e-5 Ha
      (the same box-limit/DF floor as the pbe0/pbe KRKS parity);
    * dimerized chain (1,1,2) vs the independent BIPOLE screened
      backend: 0.20 mHa (COSX quadrature band);
    * PySCF 2.13.1 KRKS hse06 (exxdiv=None) agrees to 7.3e-5 Ha after
      adding the analytic erfc G=0 zero-mode term
      (pi/w_s^2/(V.N_k)).S D S that PySCF's convention drops — pinned
      separately in the gated PySCF test.

    Guards: the Lpq GDF backend keeps rejecting screened hybrids and the
    COSX backend keeps rejecting full-range-arm RS functionals. The
    spin-unrestricted COSX route must reproduce the restricted HSE06
    result at multiplicity one.
    """
    from vibeqc.periodic_k_gdf import (
        run_krks_periodic_gdf,
        run_kuks_periodic_gdf,
    )

    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    mol = vq.Molecule(atoms, 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    common = dict(
        kmesh=(1, 1, 2),
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        progress=False,
    )

    # Vacuum-box molecular-limit oracle.
    sys_box = vq.PeriodicSystem(3, np.diag([25.0, 25.0, 25.0]), atoms)
    r_box = run_krks_periodic_gdf(
        sys_box, basis, functional="hse06", k_exchange="cosx", **common
    )
    assert r_box.converged
    o = vq.RKSOptions()
    o.functional = "hse06"
    m = vq.run_rks(mol, basis, o)
    assert abs(r_box.energy - m.energy) < 2e-4

    # Guards.
    with pytest.raises(NotImplementedError, match="range-separated"):
        run_krks_periodic_gdf(
            sys_box, basis, functional="hse06", k_exchange="gdf", **common
        )
    with pytest.raises(NotImplementedError, match="full-range"):
        run_krks_periodic_gdf(
            sys_box, basis, functional="cam-b3lyp", k_exchange="cosx",
            **common,
        )
    with pytest.raises(NotImplementedError, match="range-separated"):
        run_kuks_periodic_gdf(
            sys_box, basis, functional="hse06", k_exchange="gdf",
            kmesh=(1, 1, 2), aux_basis="def2-universal-jkfit",
            gdf_method="rsgdf", progress=False,
        )
    with pytest.raises(NotImplementedError, match="full-range"):
        run_kuks_periodic_gdf(
            sys_box, basis, functional="cam-b3lyp", k_exchange="cosx",
            kmesh=(1, 1, 2), aux_basis="def2-universal-jkfit",
            gdf_method="rsgdf", progress=False,
        )
    r_uks = run_kuks_periodic_gdf(
        sys_box,
        basis,
        kmesh=(1, 1, 2),
        functional="hse06",
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        k_exchange="cosx",
        progress=False,
    )
    assert r_uks.converged
    assert abs(r_uks.energy - r_box.energy) < 1e-7


def test_kuks_cosx_hse06_triplet_molecular_limit():
    """Open-shell screened COSX acts on each spin density independently.

    Triplet H2 in a 25-bohr box on a (1,1,2) mesh is the smallest
    genuine open-shell route check. The periodic result retains S^2=2
    and agrees with the molecular HSE06 oracle inside the established
    open-shell COSX quadrature/box-limit band.
    """
    from vibeqc.periodic_k_gdf import run_kuks_periodic_gdf

    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    mol = vq.Molecule(atoms, 0, 3)
    basis = vq.BasisSet(mol, "sto-3g")
    system = vq.PeriodicSystem(
        3, np.diag([25.0, 25.0, 25.0]), atoms, multiplicity=3
    )
    result = run_kuks_periodic_gdf(
        system,
        basis,
        kmesh=(1, 1, 2),
        functional="hse06",
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        k_exchange="cosx",
        progress=False,
    )
    assert result.converged
    assert abs(result.s_squared - 2.0) < 1e-8

    options = vq.UKSOptions()
    options.functional = "hse06"
    molecular = vq.run_uks(mol, basis, options)
    assert abs(result.energy - molecular.energy) < 2e-3


def test_krks_cosx_hse06_vs_bipole_backend():
    """Cross-backend parity: multi-k COSX hse06 vs the independent
    BIPOLE screened route on the tight-ish dimerized chain (both share
    the VASP/CRYSTAL G=0 convention). Measured 0.20 mHa; gate 1e-3."""
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf
    from vibeqc.periodic_runner import run_periodic_job

    sys_c, basis, mol = _dimerized_chain()
    r_cosx = run_krks_periodic_gdf(
        sys_c,
        basis,
        kmesh=(1, 1, 2),
        functional="hse06",
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        k_exchange="cosx",
        progress=False,
    )
    assert r_cosx.converged

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        r_bip = run_periodic_job(
            sys_c,
            basis,
            method="RKS",
            functional="hse06",
            jk_method="bipole",
            kpoints=(1, 1, 2),
            output=f"{td}/hse_bipole",
            output_qvf=False,
            citations=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            progress=False,
        )
    assert r_bip.converged
    assert abs(r_cosx.energy - r_bip.energy) < 1e-3, (
        f"COSX vs BIPOLE screened backends: "
        f"{abs(r_cosx.energy - r_bip.energy)*1e3:.3f} mHa"
    )


def test_krks_cosx_hse06_pyscf_parity():
    """External oracle (live, in-test per the toroidal-suite pattern):
    PySCF KRKS hse06 with exxdiv=None plus the analytic erfc G=0
    zero-mode term equals the vibe-qc multi-k COSX hse06 energy.

    PySCF's GDF-SR exchange drops the erfc kernel's FINITE G = 0 value
    (pi/w_s^2); vibe-qc (COSX and BIPOLE alike) includes it — the
    VASP/CRYSTAL convention. Measured after aligning: 7.3e-5 Ha."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf.pbc import dft as pbc_dft
    from pyscf.pbc import gto as pbc_gto

    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    sys_c, basis, mol = _dimerized_chain()
    r = run_krks_periodic_gdf(
        sys_c,
        basis,
        kmesh=(1, 1, 2),
        functional="hse06",
        use_compcell=True,
        aux_basis="def2-universal-jkfit",
        gdf_method="rsgdf",
        k_exchange="cosx",
        progress=False,
    )
    assert r.converged

    cell = pbc_gto.Cell()
    cell.a = np.asarray(sys_c.lattice, dtype=float)
    cell.atom = [["H", tuple(a.xyz)] for a in sys_c.unit_cell]
    cell.basis = "sto-3g"
    cell.unit = "Bohr"
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([1, 1, 2])
    mf = pbc_dft.KRKS(cell, kpts=kpts, xc="hse06").density_fit(
        auxbasis="def2-svp-jk-fit"
    )
    mf.exxdiv = None
    e_pyscf = mf.kernel()
    assert mf.converged

    w_s, c_sr = 0.11, 0.25
    n_k = len(kpts)
    S_k = mf.get_ovlp()
    D_k = mf.make_rdm1()
    zero_mode = 0.0
    for i in range(n_k):
        dK = (np.pi / w_s**2) / (cell.vol * n_k) * (
            S_k[i] @ D_k[i] @ S_k[i]
        )
        zero_mode += (1.0 / n_k) * float(
            np.real(np.trace(D_k[i] @ dK))
        )
    e_aligned = e_pyscf - 0.25 * c_sr * zero_mode

    assert abs(r.energy - e_aligned) < 3e-4, (
        f"COSX hse06 vs convention-aligned PySCF: "
        f"{abs(r.energy - e_aligned)*1e3:.3f} mHa"
    )


# ---------------------------------------------------------------------------
# SAP routing (GitLab #667)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("driver_name", "atomic_number", "multiplicity", "functional"),
    [
        ("run_periodic_rijcosx_rhf", 2, 1, None),
        ("run_periodic_rijcosx_rks", 2, 1, "lda"),
        ("run_periodic_rijcosx_uhf", 1, 2, None),
        ("run_periodic_rijcosx_uks", 1, 2, "lda"),
    ],
    ids=["rhf", "rks", "uhf", "uks"],
)
def test_gamma_rijcosx_sap_reaches_lattice_potential(
    monkeypatch,
    driver_name,
    atomic_number,
    multiplicity,
    functional,
):
    """Each Gamma RIJCOSX SCF entry point executes periodic SAP."""
    import vibeqc.guess as guess_module
    import vibeqc.periodic_rijcosx as rijcosx_module

    class SapPotentialReached(RuntimeError):
        pass

    calls = []

    def stop_at_vsap(basis, system, _grid, table, lattice_opts):
        calls.append((basis, system, table, lattice_opts))
        raise SapPotentialReached

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", stop_at_vsap)
    lattice = np.eye(3) * 16.0
    system = PeriodicSystem(
        3,
        lattice,
        [Atom(atomic_number, [8.0, 8.0, 8.0])],
        multiplicity=multiplicity,
    )
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = (
        vq.PeriodicKSOptions()
        if functional is not None
        else vq.PeriodicRHFOptions()
    )
    options.initial_guess = core.InitialGuess.SAP
    options.max_iter = 0
    kwargs = {"functional": functional} if functional is not None else {}

    with pytest.raises(SapPotentialReached):
        getattr(rijcosx_module, driver_name)(
            system,
            basis,
            options,
            aux_basis="def2-universal-jkfit",
            progress=False,
            **kwargs,
        )

    assert len(calls) == 1
    assert calls[0][0] is basis
    assert calls[0][1] is system
    assert calls[0][2] == "sap_helfem_large"
    assert calls[0][3] is options.lattice_opts


def test_public_gamma_rijcosx_forwards_sap(monkeypatch, tmp_path):
    """The public runner keeps SAP on the Gamma RIJCOSX options object."""
    import vibeqc.periodic_rijcosx as rijcosx_module

    class RijcosxDriverReached(RuntimeError):
        pass

    seen = []

    def stop_at_driver(_system, _basis, options, **_kwargs):
        seen.append(options.initial_guess)
        raise RijcosxDriverReached

    monkeypatch.setattr(
        rijcosx_module,
        "run_periodic_rijcosx_rhf",
        stop_at_driver,
    )
    system, basis, _ = _h2_box(16.0)

    with pytest.raises(RijcosxDriverReached):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="rijcosx",
            initial_guess="SAP",
            output=tmp_path / "rijcosx-sap-forwarding",
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

    assert seen == [core.InitialGuess.SAP]


@pytest.mark.parametrize(
    ("selector", "effective"),
    [
        (core.InitialGuess.AUTO, core.InitialGuess.SAD),
        (" sad ", core.InitialGuess.SAD),
        ("huckel", core.InitialGuess.HUECKEL),
        (core.InitialGuess.MINAO, core.InitialGuess.MINAO),
    ],
)
def test_gamma_rijcosx_guess_normalizes_string_and_enum(selector, effective):
    """Direct RIJCOSX selectors share the canonical normalization policy."""
    import vibeqc.periodic_rijcosx as rijcosx_module

    opts = SimpleNamespace(initial_guess=selector)
    assert (
        rijcosx_module._rijcosx_guess(opts, driver="test_rijcosx")
        == effective
    )


@pytest.mark.parametrize(
    "guess",
    [core.InitialGuess.FRAGMO],
)
def test_gamma_rijcosx_unsupported_guess_fails_before_setup(
    monkeypatch, guess
):
    """RIJCOSX rejects missing seams before auxiliary/integral setup."""
    import vibeqc.periodic_rijcosx as rijcosx_module

    monkeypatch.setattr(
        rijcosx_module,
        "_resolve_aux_basis",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported guess reached RIJCOSX setup"
        ),
    )
    system, basis, _ = _h2_box(16.0)
    options = PeriodicRHFOptions()
    options.initial_guess = guess

    with pytest.raises(NotImplementedError, match=f"initial_guess={guess.name}"):
        rijcosx_module.run_periodic_rijcosx_rhf(
            system, basis, options, progress=False
        )


@pytest.mark.parametrize(
    "guess",
    [
        core.InitialGuess.HCORE,
        core.InitialGuess.SAD,
        core.InitialGuess.SAP,
        core.InitialGuess.HUECKEL,
        core.InitialGuess.MINAO,
    ],
)
def test_gamma_rijcosx_closed_guesses_use_shared_density_adapter(
    monkeypatch, guess
):
    """Every supported RIJCOSX kind reaches the shared density helper."""
    import vibeqc.periodic_rijcosx as rijcosx_module

    system, basis, _ = _h2_box(16.0)
    sentinel = np.full((basis.nbasis, basis.nbasis), 0.125)
    seen = []

    def fake_density(_mol, _basis, n_occ, initial_guess, **kwargs):
        seen.append((n_occ, initial_guess, kwargs))
        return None if initial_guess == core.InitialGuess.HCORE else sentinel

    monkeypatch.setattr(
        rijcosx_module, "initial_density_closed_shell", fake_density
    )
    actual = rijcosx_module._initial_density_closed(
        system, basis, 1, guess, LatticeSumOptions(), np.eye(basis.nbasis)
    )

    assert seen[0][0] == 1
    assert seen[0][1] == guess
    assert seen[0][2]["periodic_system"] is system
    if guess == core.InitialGuess.HCORE:
        assert actual.shape == (0, 0)
    else:
        np.testing.assert_array_equal(actual, sentinel)


# ---------------------------------------------------------------------------
# #707: vacuum-padded drivers drop image-cell electrostatics consistently
# ---------------------------------------------------------------------------


def test_rijcosx_gamma_vacuum_padded_home_cell_electrostatics():
    """Every electrostatic term of the vacuum-padded Gamma RIJCOSX path is a
    home-cell quantity, independent of the box length and lattice cutoffs.

    The molecular-limit builder's J is the molecular DF J (no image
    electrons). Before #707, V_ne and E_nuc summed the image nuclei inside
    ``nuclear_cutoff_bohr`` (25 bohr default) and the COSX K summed the
    image-ket exchange class inside ``cutoff_bohr`` (15 bohr default), so
    every box shorter than those cutoffs was over-bound by the bare
    monopole sums that only the missing image part of J would cancel
    (Makov & Payne 1995: only the neutral-cell total is gauge-invariant).

    Pre-fix, measured on main at 5651aaba1: H atom / STO-3G, UHF, 12-bohr
    box -1.6377388584 Ha (32 image protons: -1.8420 Ha in V_ne, +0.9210 Ha
    in E_nuc; 6 image cells in K: +0.5000 Ha); H2 / STO-3G RHF -1.61675811
    Ha in a 24-bohr box (-0.5000 Ha) and -5.30082663 Ha in a 12-bohr box,
    against molecular RHF -1.11671433 Ha. Post-fix: UHF H atom
    -0.46673838 Ha vs molecular RIJCOSX UHF -0.46672900 Ha (9.4e-6) and
    exact UHF -0.46658185 Ha (1.6e-4, the COSX quadrature floor); RHF H2
    -1.11679649 (12 bohr) / -1.11675738 (24 bohr) vs molecular RHF (8.2e-5 /
    4.3e-5, the shared box-limit/DF floor of the 50-bohr anchor)."""
    from vibeqc.periodic_rijcosx import (
        _home_cell_electrostatics_lat_opts,
        run_periodic_rijcosx_uhf,
    )

    # --- one-electron system: J and K must cancel at the operator level.
    h_sys = PeriodicSystem(
        3, 12.0 * np.eye(3), [Atom(1, [6.0, 6.0, 6.0])], multiplicity=2
    )
    h_mol = h_sys.unit_cell_molecule()
    h_basis = BasisSet(h_mol, "sto-3g")
    lat_opts = PeriodicRHFOptions().lattice_opts
    one_e_opts, exchange_opts = _home_cell_electrostatics_lat_opts(lat_opts)
    assert one_e_opts.nuclear_cutoff_bohr == 0.0
    assert one_e_opts.cutoff_bohr == lat_opts.cutoff_bohr
    assert exchange_opts.cutoff_bohr == 0.0
    assert exchange_opts.nuclear_cutoff_bohr == lat_opts.nuclear_cutoff_bohr
    assert len(core.direct_lattice_cells(h_sys, 0.0)) == 1
    # Home-cell nuclei only: E_nuc of a one-atom cell is zero and V_ne is
    # the molecular nuclear attraction.
    assert core.nuclear_repulsion_per_cell(h_sys, one_e_opts) == 0.0
    aux = make_aux_basis_set(h_mol, aux_name="def2-universal-jkfit")
    jk = core.make_periodic_gamma_cosx_jk_builder(
        h_basis, aux, h_sys, exchange_opts
    )
    D = np.array([[1.0]])
    J = float(jk.build_J(D)[0, 0])
    K = float(jk.build_K(D)[0, 0])
    assert abs(J - 0.7746) < 5e-3  # (1s 1s|1s 1s) of STO-3G H
    # Pre-fix K carried +6/12 Ha of image-ket exchange (1.2746).
    assert abs(J - K) < 1e-3, f"one-electron J-K = {J - K:.3e}"

    r_uhf = run_periodic_rijcosx_uhf(
        h_sys, h_basis, aux_basis="def2-universal-jkfit", progress=False
    )
    assert r_uhf.converged
    o = vq.UHFOptions()
    o.density_fit = True
    o.aux_basis = "def2-universal-jkfit"
    o.cosx = True
    m_cosx = vq.run_uhf(h_mol, h_basis, o)
    m_exact = vq.run_uhf(h_mol, h_basis)
    assert abs(r_uhf.energy - m_cosx.energy) < 1e-4
    assert abs(r_uhf.energy - m_exact.energy) < 1e-3

    # --- closed-shell RHF: boxes inside both default cutoffs match the
    # molecular oracle exactly as the 50-bohr anchor does.
    for box in (12.0, 24.0):
        sys_box, basis, mol = _h2_box(box)
        r = run_periodic_rijcosx_rhf(
            sys_box, basis, aux_basis="def2-universal-jkfit", progress=False
        )
        assert r.converged, box
        assert abs(r.e_nuclear - 1.0 / 1.4) < 1e-12, box
        m = vq.run_rhf(mol, basis)
        assert abs(r.energy - m.energy) < 2e-4, (
            f"box {box}: |dE| = {abs(r.energy - m.energy):.3e}"
        )
