"""MDF (Mixed Density Fitting) cderi tests — Sun-Berkelbach 2017.

Mixed Density Fitting (Sun, Berkelbach, McClain & Chan, J. Chem. Phys.
147, 164119 (2017), doi:10.1063/1.4998644) closes the all-electron
accuracy floor of pure Gaussian DF by adding a plane-wave residual to the
compensated-Gaussian fit. Design: ``docs/design_mdf.md``.

Increment 1 (this file): the cderi builder ``build_lpq_mdf``.

* The **no-PW limit must reproduce ``build_lpq_compcell`` (no AFT)
  bit-for-bit** — same compensated metric ``M``, same 3c ``T``, same
  eigendecomposition. This pins the Gaussian scaffold against the
  validated compcell builder.
* With the PW mesh on, the residual cderi has the documented shape and
  the reconstructed ``W`` is Hermitian + positive-semidefinite.

Later increments add ``get_eri`` parity vs ``pyscf.pbc.df.MDF`` and the
MgO all-electron µHa SCF gate (out-of-process, CLAUDE.md §10).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.aux_basis import (
    build_lpq_compcell,
    build_lpq_mdf,
    make_aux_basis_set,
)


def _h2_box(box_bohr: float = 12.0, sep_bohr: float = 1.4):
    half = 0.5 * sep_bohr
    system = vq.PeriodicSystem(
        3,
        np.diag([box_bohr, box_bohr, box_bohr]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lat_opts(cutoff_bohr: float = 20.0):
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = cutoff_bohr
    lo.nuclear_cutoff_bohr = cutoff_bohr
    return lo


def test_build_lpq_mdf_no_pw_equals_compcell():
    """MDF with the PW mesh disabled ≡ compcell (no AFT), bit-for-bit.

    The no-PW limit is ``J̃ = M``, ``T`` uncorrected, ``L = M^{-½}·T`` —
    identical construction to ``build_lpq_compcell(apply_aft_correction=
    False)`` with the same eta / rcut / threshold. The two share every
    numpy op in the same order, so the result is bit-identical, not merely
    close. This is the increment-1 sanity gate: it proves the Gaussian
    scaffold of ``build_lpq_mdf`` matches the validated compcell builder
    before any PW residual is layered on.
    """
    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = _lat_opts(20.0)

    L_compcell = build_lpq_compcell(
        system,
        basis,
        aux,
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        apply_aft_correction=False,
        linear_dep_thr=1e-9,
        rcut_strategy="pyscf_auto",
    )
    cderi = build_lpq_mdf(
        system,
        basis,
        aux,
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        mdf_ke_cutoff=0.0,  # PW off → compcell-equivalent limit
        linear_dep_thr=1e-9,
        rcut_strategy="pyscf_auto",
    )

    assert cderi.n_pw == 0
    assert cderi.cderi_pw.shape == (0, basis.nbasis, basis.nbasis)
    assert cderi.L_gauss.shape == L_compcell.shape
    # Agreement to round-off, NOT bit-identity. This assertion used
    # np.array_equal until 2026-08-03, which asserted a property the code
    # cannot provide: the 2c/3c lattice kernels parallelise their cell
    # loop with `schedule(dynamic)` and thread-local accumulators, so
    # which cells land on which thread varies between calls and
    # floating-point addition is not associative. Measured directly:
    # four identical calls to compute_2c_eri_lattice differ by up to
    # 8.9e-16, and this test's own delta varies run to run (5.1e-14,
    # 1.8e-15, 1.8e-15). The delta lands in L entries whose exact value
    # is zero by symmetry, amplified from the integral round-off by the
    # 1/sqrt(lambda) orthogonalisation of a near-singular metric.
    #
    # The claim being pinned is that the two builders evaluate the SAME
    # expression, which a tight relative tolerance states correctly. If
    # bit-reproducibility is wanted it has to be bought in the kernel
    # (a deterministic cell-to-thread mapping), not asserted here --
    # see the reproducibility note in HANDOVER_OPEN_BUGS_V015.md.
    scale = max(float(np.max(np.abs(L_compcell))), 1.0)
    delta = float(np.max(np.abs(cderi.L_gauss - L_compcell)))
    assert delta < 1e-11 * scale, (
        "MDF no-PW limit must equal compcell(no AFT) to round-off; "
        f"max |Δ| = {delta:.3e} (scale {scale:.3e})"
    )


def test_build_lpq_mdf_pw_residual_shapes_and_psd():
    """With the PW mesh on, the residual cderi is well-formed and the
    reconstructed ``W`` is Hermitian + positive-semidefinite.

    ``W = Σ_i L·L + Σ_G cderi_pw·conj(cderi_pw)`` is a sum of two
    Gram matrices, hence PSD. The PW residual is symmetric in (μ,ν) at Γ
    (real AO pairs).
    """
    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = _lat_opts(20.0)
    n = basis.nbasis

    cderi = build_lpq_mdf(
        system,
        basis,
        aux,
        molecule=mol,
        lat_opts=lo,
        compcell_eta=0.25,
        mdf_ke_cutoff=20.0,
    )

    assert cderi.n_pw > 0
    assert cderi.cderi_pw.shape == (cderi.n_pw, n, n)
    assert cderi.cderi_pw.dtype == np.complex128
    assert cderi.L_gauss.shape[1:] == (n, n)
    # Real AO pairs at Γ ⇒ the residual is (μ,ν)-symmetric.
    assert np.allclose(
        cderi.cderi_pw, cderi.cderi_pw.transpose(0, 2, 1), atol=1e-10
    )

    # Reconstruct the 4-index W and check Hermiticity + PSD.
    Lg = cderi.L_gauss.reshape(cderi.n_kept_gauss, n * n)
    Wp = cderi.cderi_pw.reshape(cderi.n_pw, n * n)
    W = Lg.T @ Lg + np.real(Wp.conj().T @ Wp)
    assert np.allclose(W, W.T, atol=1e-9)
    w_eig = np.linalg.eigvalsh(W)
    assert w_eig.min() > -1e-8, f"W not PSD: min eig {w_eig.min():.3e}"


def test_build_lpq_mdf_reconstructs_rsgdf_W_and_is_mesh_stable():
    """The MDF cderi (Gaussian L + PW residual + the Eq-23 G=0 V̄ρ̄ term)
    reconstructs the same 4-index ``W`` as the dense-mesh rsgdf fit — which
    is µHa-exact for a light atom in a box — and does so *mesh-stably*.

    Mesh-stability is the proof that the G=0 self-term is right: without
    V̄ρ̄ the L vectors blow up as the residual mesh tightens (the
    ke-independent T_real−T_recip gap, docs/design_mdf.md §4). With it the
    W-agreement plateaus at the higher-multipole floor (~1e-4 on
    H2/def2-svp-jk) instead of diverging.
    """
    from vibeqc.aux_basis import build_lpq_native_fft, make_modrho_aux_basis

    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = _lat_opts(30.0)
    n = basis.nbasis

    # rsgdf dense-mesh reference W (µHa-exact for H2 in a vacuum box).
    Lr = build_lpq_native_fft(
        system, basis, make_modrho_aux_basis(aux, mol),
        ke_cutoff=400.0, lat_opts=lo,
    ).reshape(-1, n * n)
    W_ref = Lr.T @ Lr

    diffs = []
    for ke in (20.0, 40.0, 80.0):
        c = build_lpq_mdf(
            system, basis, aux, molecule=mol, lat_opts=lo,
            compcell_eta=1.0, mdf_ke_cutoff=ke,
        )
        Lg = c.L_gauss.reshape(c.n_kept_gauss, n * n)
        Wp = c.cderi_pw.reshape(c.n_pw, n * n)
        W_mdf = Lg.T @ Lg + np.real(Wp.conj().T @ Wp)
        diffs.append(float(np.abs(W_mdf - W_ref).max()))

    # Matches the dense-mesh fit to the higher-multipole floor ...
    assert max(diffs) < 5e-4, f"W(MDF) vs W(rsgdf): {diffs}"
    # ... and is mesh-stable: tightening the mesh does not blow it up
    # (would be O(1)+ without the G=0 correction).
    assert abs(diffs[-1] - diffs[0]) < 1e-4, f"not mesh-stable: {diffs}"


def test_run_pbc_gdf_rhf_mdf_h2_converges_near_rsgdf():
    """The wired Γ MDF SCF path (gdf_method='mdf', complex combined cderi
    through the complex-aware J/K) converges and lands close to the
    dense-mesh rsgdf reference on a light system — MDF must not regress the
    cases pure Gaussian DF already nails. The residual (~tens of µHa) is
    the neglected higher-multipole G=0 term + the modest residual mesh;
    MDF's payoff is the all-electron/heavy-atom regime, not H2.
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf

    system, basis = _h2_box(box_bohr=12.0)
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 100
    opt.conv_tol_energy = 1e-10
    common = dict(aux_basis="def2-svp-jk", exxdiv="ewald", progress=False)

    r_rsgdf = run_pbc_gdf_rhf(system, basis, opt, gdf_method="rsgdf", **common)
    r_mdf = run_pbc_gdf_rhf(
        system, basis, opt, gdf_method="mdf", mdf_ke_cutoff=40.0, **common
    )

    assert r_rsgdf.converged and r_mdf.converged
    # MDF reproduces the µHa-exact rsgdf energy for a light atom to well
    # within the GDF floor it is designed to remove (~9 Ha on MgO).
    assert abs(r_mdf.energy - r_rsgdf.energy) < 2e-4, (
        f"MDF {r_mdf.energy} vs rsgdf {r_rsgdf.energy} "
        f"(Δ={ (r_mdf.energy - r_rsgdf.energy)*1e3:.4f} mHa)"
    )


def test_run_pbc_gdf_uhf_mdf_triplet_h2():
    """MDF flows through the open-shell Γ driver (run_pbc_gdf_uhf,
    gdf_method='mdf') via the shared _pbc_gdf_gamma_setup + complex-aware
    per-spin J/K. Triplet H2 reproduces PySCF UHF Γ to the MDF
    higher-multipole floor and ⟨S²⟩ is exactly the triplet value.

    PySCF reference (out-of-process, §10; pyscf.pbc.scf.UHF(cell)
    .density_fit()/exxdiv='ewald', H2/sto-3g/12-bohr, cell.spin=2):
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_uhf

    PYSCF_UHF_H2_TRIPLET = -0.53585051
    system, basis = _h2_box(box_bohr=12.0)
    system.multiplicity = 3

    r = run_pbc_gdf_uhf(
        system, basis, aux_basis="def2-svp-jk", exxdiv="ewald",
        gdf_method="mdf", mdf_ke_cutoff=40.0, progress=False,
    )
    assert r.converged
    assert abs(r.energy - PYSCF_UHF_H2_TRIPLET) < 5e-5, (
        f"UHF/MDF triplet {r.energy} vs PySCF {PYSCF_UHF_H2_TRIPLET}"
    )
    assert abs(r.s_squared - 2.0) < 1e-6


def test_run_periodic_job_mdf_threading(tmp_path):
    """run_periodic_job exposes gdf_method='mdf' + mdf_ke_cutoff and routes
    them to the GDF drivers: Γ closed-shell RHF through run_pbc_gdf_rhf,
    multi-k through run_krhf_periodic_gdf. The runner result must match the
    direct-driver MDF energy (the params actually reach the builder)."""
    from vibeqc.periodic_runner import run_periodic_job
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf

    system, basis = _h2_box(box_bohr=12.0)
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 80
    opt.conv_tol_energy = 1e-9
    e_direct = run_pbc_gdf_rhf(
        system, basis, opt, aux_basis="def2-svp-jk", gdf_method="mdf",
        mdf_ke_cutoff=40.0, exxdiv="ewald", progress=False,
    ).energy

    r = run_periodic_job(
        system, basis, method="RHF", jk_method="gdf", aux_basis="def2-svp-jk",
        gdf_method="mdf", mdf_ke_cutoff=40.0, output=str(tmp_path / "mdf_g"),
        max_iter=80, conv_tol_energy=1e-9, initial_guess="HCORE",
        write_molden_file=False,
    )
    assert r.converged
    # Runner Γ RHF/MDF == direct run_pbc_gdf_rhf MDF (params threaded).
    assert abs(r.energy - e_direct) < 1e-9, (
        f"runner MDF {r.energy} vs direct {e_direct}"
    )


@pytest.mark.slow
def test_run_krhf_mdf_multik_h2_matches_pyscf():
    """Multi-k MDF (build_lpq_bloch_mdf via run_krhf_periodic_gdf) on the
    canonical H2/sto-3g/12-bohr kmesh=(2,1,1) cell reproduces the published
    KRHF reference and tracks the validated rsgdf multi-k path.

    The per-k-pair MDF cderi is the q-only real-space Gaussian fit + the
    ket-resolved PW residual (+ the G=0 V̄ρ̄ term on diagonal pairs). This
    is the genuine BZ-sampling test (kmesh=(1,1,1) is a degenerate Γ case
    whose multi-k-vs-direct offset is pre-existing + method-independent).

    Reference: E_HF = -1.12013988 Ha (Sun, Berkelbach, McClain & Chan,
    J. Chem. Phys. 147, 164119 (2017), Table II; pinned in CLAUDE.md §8;
    vibe-qc rsgdf reproduces it exactly).
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    PYSCF_KRHF_H2_211 = -1.12013988
    system, basis = _h2_box(box_bohr=12.0)
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 100
    opt.conv_tol_energy = 1e-9
    common = dict(aux_basis="def2-svp-jk", progress=False)

    r_mdf = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=opt,
        gdf_method="mdf", mdf_ke_cutoff=40.0, **common
    )
    r_rsgdf = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=opt,
        gdf_method="rsgdf", rsgdf_ke_cutoff=200.0, **common
    )
    assert r_mdf.converged and r_rsgdf.converged
    # Multi-k MDF matches the published KRHF reference to the MDF
    # higher-multipole floor (~tens of µHa for this light cell).
    assert abs(r_mdf.energy - PYSCF_KRHF_H2_211) < 5e-4, (
        f"multi-k MDF {r_mdf.energy} vs PySCF {PYSCF_KRHF_H2_211} "
        f"(Δ={(r_mdf.energy - PYSCF_KRHF_H2_211)*1e3:.4f} mHa)"
    )
    # ... and tracks the validated rsgdf multi-k path closely.
    assert abs(r_mdf.energy - r_rsgdf.energy) < 1e-4


@pytest.mark.parametrize("tail", [None, 0.0, 3200.0])
def test_gamma_rsgdf_obsolete_tail_reaches_setup_as_none(monkeypatch, tail):
    """No legacy-tail request may change the production RSGDF setup contract."""
    import warnings
    import vibeqc.pbc_gdf as gdf

    system, basis = _h2_box()
    captured = []

    class SetupReached(Exception):
        pass

    def capture_setup(*args, **kwargs):
        captured.append(kwargs)
        raise SetupReached

    # Avoid even the diagnostic cell enumeration; stop before all integrals.
    monkeypatch.setattr(gdf, "direct_lattice_cells", lambda *args: [])
    monkeypatch.setattr(gdf, "_pbc_gdf_gamma_setup", capture_setup)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(SetupReached):
            gdf.run_pbc_gdf_rhf(
                system, basis, gdf_method="rsgdf", rsgdf_tail_ke_cutoff=tail,
                rsgdf_g_precision=2e-10, progress=False,
            )
    deprecated = [w for w in recorded if issubclass(w.category, DeprecationWarning)
                  and "obsolete for the SR/LR fit" in str(w.message)]
    assert len(deprecated) == (0 if tail is None else 1)
    assert len(captured) == 1
    assert captured[0]["gdf_method"] == "rsgdf"
    assert captured[0]["rsgdf_tail_ke_cutoff"] is None
    assert captured[0]["rsgdf_g_precision"] == 2e-10


@pytest.mark.slow
def test_run_pbc_gdf_rhf_mdf_all_electron_ne_box(monkeypatch):
    """MDF mesh convergence and independent MDF/GDF Ne energy references.

    The production RSGDF route uses real-space short-range integrals plus
    the reciprocal long-range fit. Its obsolete ``tail=0`` argument warns
    and is ignored; it cannot select the old, intentionally deficient
    all-FT mesh used by the former negative control (#276).

    PySCF reference (out-of-process, §10; pyscf.pbc.df.MDF, RHF,
    exxdiv='ewald', cell.unit='B', Ne/sto-3g/10-bohr cube,
    auxbasis='def2-svp-jkfit' == vibe-qc 'def2-svp-jk'):
      PYSCF GDF = -126.61358133   PYSCF MDF = -126.61361316
    (the GDF↔MDF gap is tiny for *this* aux because def2-svp-jkfit already
    resolves Ne's core.) Both reference comparisons use the existing
    0.5 mHa absolute parity envelope of this fixture, accommodating its
    finite-domain/fit differences. This is independent of the measured
    RSGDF-MDF gap and does not claim microhartree or general-cell parity.
    """
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf
    import vibeqc.periodic_k_gdf as kgdf

    PYSCF_MDF_NE = -126.61361316  # def2-svp-jkfit, out-of-process
    PYSCF_GDF_NE = -126.61358133  # same independent reference envelope above
    box = 10.0
    system = vq.PeriodicSystem(3, np.diag([box] * 3), [vq.Atom(10, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 120
    opt.conv_tol_energy = 1e-10
    common = dict(aux_basis="def2-svp-jk", exxdiv="ewald", progress=False)

    e_mdf_lo = run_pbc_gdf_rhf(
        system, basis, opt, gdf_method="mdf", mdf_ke_cutoff=60.0, **common
    ).energy
    e_mdf_hi = run_pbc_gdf_rhf(
        system, basis, opt, gdf_method="mdf", mdf_ke_cutoff=120.0, **common
    ).energy
    # Observe the actual builder without replacing its numerical work.
    # This guards the route as well as the energy, with no extra SCF run.
    sr_lr_calls = []
    build_sr_lr = kgdf._build_scf_range_separated_lpq_cache

    def observed_sr_lr(*args, **kwargs):
        sr_lr_calls.append(kwargs.copy())
        return build_sr_lr(*args, **kwargs)

    monkeypatch.setattr(kgdf, "_build_scf_range_separated_lpq_cache", observed_sr_lr)
    with pytest.warns(DeprecationWarning, match="obsolete for the SR/LR fit"):
        r_rsgdf = run_pbc_gdf_rhf(
            system, basis, opt, gdf_method="rsgdf", rsgdf_ke_cutoff=200.0,
            rsgdf_tail_ke_cutoff=0.0, **common
        )
    assert r_rsgdf.converged
    assert r_rsgdf.rsgdf_tail_ke_cutoff is None
    assert "+PARITY_HELD" not in r_rsgdf.backend
    assert len(sr_lr_calls) == 1
    assert sr_lr_calls[0]["ke_cutoff"] == 200.0
    assert sr_lr_calls[0]["raw_integral_error"] == 1e-10
    e_rsgdf = r_rsgdf.energy

    # (1) MDF reproduces PySCF MDF (same aux) to the higher-multipole floor.
    assert abs(e_mdf_hi - PYSCF_MDF_NE) < 5e-4, (
        f"MDF {e_mdf_hi} vs PySCF MDF {PYSCF_MDF_NE} "
        f"(Δ={(e_mdf_hi - PYSCF_MDF_NE)*1e3:.3f} mHa)"
    )
    # (2) MDF is mesh-converged at a modest ke — the whole point.
    assert abs(e_mdf_hi - e_mdf_lo) < 1e-4, (
        f"MDF not mesh-converged: ke=60 {e_mdf_lo} vs ke=120 {e_mdf_hi}"
    )
    # (3) Modern SR/LR RSGDF is checked against its own external oracle,
    # not against an expected defect or an empirically fitted MDF gap.
    assert abs(e_rsgdf - PYSCF_GDF_NE) < 5e-4, (
        f"SR/LR RSGDF {e_rsgdf} vs PySCF GDF {PYSCF_GDF_NE} "
        f"(delta={(e_rsgdf - PYSCF_GDF_NE)*1e3:.3f} mHa)"
    )


def _mgo_primitive():
    """MgO rocksalt 2-atom primitive (FCC), a = 4.211 Å — the P01
    compact-dense-core class (G-GDF-001)."""
    ang2bohr = 1.8897259886
    a = 4.211 * ang2bohr
    prim = 0.5 * a * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float
    )
    o = 0.5 * (prim[0] + prim[1] + prim[2])
    system = vq.PeriodicSystem(
        3, prim, [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, o.tolist())]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_dense_core_mdf_fails_closed_gamma_and_multik():
    """Compact-dense-core MDF raises instead of returning garbage.

    G-GDF-001 MDF sub-gate (2026-07-29): Γ mdf on MgO/STO-3G is
    non-convergent and trial-dependent at ke=40/60 (+1703/+2143/+418 Ha)
    and falsely reports converged=True at ke=80 at -1651 Ha (PySCF MDF:
    -271.0499); multi-k (1,1,2) mdf lands -11741 Ha non-converged. The
    rsgdf tail remediation named by the parity-hold warning does not
    exist for mdf, so the class fails closed (CLAUDE.md §7) across all
    five driver entry points.
    """
    system, basis = _mgo_primitive()
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 5
    opt.lattice_opts.cutoff_bohr = 18.0
    opt.lattice_opts.nuclear_cutoff_bohr = 18.0
    common = dict(aux_basis="def2-svp-jk", gdf_method="mdf", progress=False)

    with pytest.raises(NotImplementedError, match="dense-core"):
        vq.run_pbc_gdf_rhf(system, basis, opt, **common)
    with pytest.raises(NotImplementedError, match="dense-core"):
        vq.run_pbc_gdf_uhf(system, basis, opt, **common)
    with pytest.raises(NotImplementedError, match="dense-core"):
        vq.run_pbc_gdf_uks(system, basis, opt, functional="pbe", **common)
    with pytest.raises(NotImplementedError, match="dense-core"):
        vq.run_krhf_periodic_gdf(system, basis, (1, 1, 2), opt, **common)
    with pytest.raises(NotImplementedError, match="dense-core"):
        vq.run_kuhf_periodic_gdf(system, basis, (1, 1, 2), opt, **common)


def test_classifier_mdf_not_held_on_tight_basis_vacuum_box():
    """The validated vacuum-box heavy-atom MDF class is NOT tagged.

    MDF fits the steep core exactly in real space, so the rsgdf
    tight-basis/unresolved-reciprocal-mesh hold does not apply to it:
    Ne/STO-3G/10-bohr (zeta_max = 207, the validated 0.14 mHa PySCF-MDF
    parity gate above) must not be classified held for mdf. The legacy
    untailed-RSGDF classifier still identifies this basis as requiring
    reciprocal tail resolution; production SR/LR RSGDF does not use that
    hold and tail=0 does not select the legacy builder. The compact class
    stays held for both classifier labels (and MDF refuses it above).
    """
    from vibeqc.pbc_gdf import _gamma_dense_core_gdf_parity_held

    ne = vq.PeriodicSystem(
        3, np.diag([10.0] * 3), [vq.Atom(10, [5.0] * 3)]
    )
    ne_basis = vq.BasisSet(ne.unit_cell_molecule(), "sto-3g")
    assert _gamma_dense_core_gdf_parity_held(ne, "mdf", ne_basis) is False
    assert _gamma_dense_core_gdf_parity_held(ne, "rsgdf", ne_basis) is True

    mgo, mgo_basis = _mgo_primitive()
    assert _gamma_dense_core_gdf_parity_held(mgo, "mdf", mgo_basis) is True
    assert _gamma_dense_core_gdf_parity_held(mgo, "rsgdf", mgo_basis) is True


@pytest.mark.slow
def test_ne_box_mdf_backend_untagged():
    """The validated Ne-in-box MDF run carries a clean backend tag.

    Before the G-GDF-001 MDF sub-gate closure this run was tagged
    ``+PARITY_HELD`` and warned with the (inapplicable) rsgdf tail
    remediation despite being the validated 0.14 mHa PySCF-MDF parity
    case. Energy is unchanged by the classifier narrowing (pinned to
    the value measured both before and after the change).
    """
    import warnings as _warnings

    system = vq.PeriodicSystem(
        3, np.diag([10.0] * 3), [vq.Atom(10, [5.0] * 3)]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 60
    opt.conv_tol_energy = 1e-9
    opt.lattice_opts.cutoff_bohr = 30.0
    opt.lattice_opts.nuclear_cutoff_bohr = 30.0
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        r = vq.run_pbc_gdf_rhf(
            system, basis, opt, aux_basis="def2-svp-jk",
            gdf_method="mdf", mdf_ke_cutoff=60.0, exxdiv="ewald",
            progress=False,
        )
    assert r.converged
    assert "PARITY_HELD" not in r.backend
    parity_warns = [x for x in w if "parity" in str(x.message).lower()]
    assert not parity_warns
    # Re-pinned 2026-08-14 from -126.6136772952. The linear-dependence fix
    # moved this value, and moved it TOWARDS the oracle: the out-of-process
    # PySCF MDF reference for this cell is -126.61361316, so the deviation
    # went from 0.064 mHa to 0.0024 mHa. The old pin recorded the
    # near-null-mode contamination, not a physical target.
    assert abs(r.energy - (-126.6136107693)) < 1e-6


def test_build_lpq_mdf_pw_changes_the_fit():
    """Turning the PW residual on must actually change the fit — the
    Gaussian L is re-orthogonalised on the PW-dressed metric (which
    re-ranks the aux space, Eq 19–21) and the PW residual adds a
    non-trivial term to ``W``. Guards against a silent no-op."""
    system, basis = _h2_box(box_bohr=12.0)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    lo = _lat_opts(20.0)
    n = basis.nbasis

    off = build_lpq_mdf(
        system, basis, aux, molecule=mol, lat_opts=lo,
        compcell_eta=0.25, mdf_ke_cutoff=0.0,
    )
    on = build_lpq_mdf(
        system, basis, aux, molecule=mol, lat_opts=lo,
        compcell_eta=0.25, mdf_ke_cutoff=20.0,
    )
    assert on.n_pw > 0 and off.n_pw == 0
    # The PW-dressing removes Gaussian↔PW linear dependence, so the
    # retained-mode count drops (the Gaussians that the PWs already span
    # become null in J̃).
    assert on.n_kept_gauss < off.n_kept_gauss

    def reconstruct(c):
        w = c.L_gauss.reshape(c.n_kept_gauss, n * n)
        out = w.T @ w
        if c.n_pw:
            wp = c.cderi_pw.reshape(c.n_pw, n * n)
            out = out + np.real(wp.conj().T @ wp)
        return out

    # The full reconstructed W moves when the PW residual is added.
    assert not np.allclose(reconstruct(off), reconstruct(on), atol=1e-6)


# --------------------------------------------------------------------------
# Linear-dependence threshold: convention + calibrated floor (2026-08-14)
#
# Root cause of the MDF wrong-answer class (registry G-GDF-001): the
# PW-dressed metric J-tilde of Eq. 20 is deliberately singular ("Projecting
# the PWs out of the Gaussian functions in (20) leads to a highly singular
# matrix", Sun 2017 Sec. II B), and the shipped threshold retained modes
# lying BELOW J-tilde's own construction error. Eq. 19 then divides the
# Eq.-23 numerator by their square root.
#
# Measured on MgO/STO-3G/def2-svp-jk at the production rcut against a
# converged 34-bohr reference: the J-tilde spectrum carries up to 5.4e-06
# absolute error, while the smallest retained eigenvalue was 2.9e-08 with a
# 6.6e-08 error bar -- the retained mode was smaller than its own
# uncertainty. Amplification 1/sqrt(lambda) ~ 6e+03.
# --------------------------------------------------------------------------


def test_metric_keep_mask_is_absolute_not_relative():
    """The fitting-metric threshold is ABSOLUTE, matching PySCF.

    Sun 2017 Sec. II B prescribes only "remove the eigenvectors associated
    with small eigenvalues below a threshold" and states no normalisation.
    The authors' own reference implementation compares the raw eigenvalue of
    the un-normalised 2-centre Coulomb metric (PySCF
    ``pbc/df/rsdf_builder.py``, ``eigenvalue_decomposed_metric``), through
    one shared ``linear_dep_threshold`` used identically by its GDF, MDF and
    RSGDF builders; PySCF has no max-eigenvalue-relative convention for a
    fitting metric anywhere.

    vibe-qc previously mixed conventions -- the FFT/rsgdf builders compared
    absolutely, the compcell and MDF builders against
    ``linear_dep_thr * max_eig``. With ``max_eig ~ 10`` the same number meant
    two cuts an order of magnitude apart. This pins the harmonisation.
    """
    from vibeqc.aux_basis import _metric_keep_mask

    # max_eig = 100, so a relative reading of 1e-3 would cut at 1e-1 and
    # drop the 1e-2 mode; the absolute reading keeps it.
    eigvals = np.array([1e-12, 1e-4, 1e-2, 1.0, 100.0])
    keep = _metric_keep_mask(eigvals, 1e-3)
    assert keep.tolist() == [False, False, True, True, True]
    # Strictly-greater comparison, and no dependence on the spectrum's scale:
    # rescaling the metric rescales which modes survive an absolute cut.
    assert _metric_keep_mask(np.array([1e-3]), 1e-3).tolist() == [False]
    # eigvals * 1e-2 = [1e-14, 1e-6, 1e-4, 1e-2, 1.0]; a relative reading
    # would be invariant under this rescaling, an absolute one is not.
    assert _metric_keep_mask(eigvals * 1e-2, 1e-3).tolist() == [
        False, False, False, True, True
    ]

    # The optional noise floor is a SEPARATE, deliberately scale-aware
    # concept: it guards against the metric's own construction error, which
    # scales with the metric. thr = max(1e-6, 1e-4 * 100) = 1e-2.
    assert _metric_keep_mask(eigvals, 1e-6, 1e-4).tolist() == [
        False, False, False, True, True
    ]
    # A floor below the threshold does not move it.
    assert _metric_keep_mask(eigvals, 1e-2, 1e-12).tolist() == [
        False, False, False, True, True
    ]


def test_plain_metric_floor_preserves_validated_compcell_cut():
    """The plain-metric floor reproduces the pre-harmonisation cut exactly.

    The compcell and bare-native builders historically compared
    ``eigvals > linear_dep_thr * max_eig``. Expressing that as an absolute
    comparison plus a scale-aware floor is bit-equivalent at the shipped
    default, which is what keeps the harmonisation from silently loosening
    those validated paths. Measured over 88 decompositions in the compcell
    + periodic-RHF-GDF suites, dropping the floor changed the retained rank
    in exactly one case (LiH ionic FCC, max_eig 2.814: 61 -> 62), admitting
    a mode whose 1/sqrt(lambda) amplification is ~3e+04.
    """
    from vibeqc.aux_basis import (
        _PLAIN_METRIC_MIN_EIG_FRACTION as FRAC,
        _metric_keep_mask,
    )

    rng = np.random.default_rng(20260814)
    for scale in (1e-3, 1.0, 2.814, 1e3):
        w = np.sort(np.abs(rng.normal(size=64))) * scale
        w[:8] = np.array([1e-14, 1e-12, 1e-11, 1e-10, 5e-10, 1e-9, 2e-9, 5e-9])
        floored = _metric_keep_mask(w, FRAC, FRAC)
        relative = w > (FRAC * float(w[-1]))
        absolute = w > FRAC
        # The shipped cut is max(absolute, relative): never looser than
        # either. That is the safety property -- harmonising the convention
        # must not admit modes the validated relative cut rejected.
        assert floored.tolist() == (relative & absolute).tolist(), (
            f"shipped cut is not the stricter of the two at scale {scale}"
        )
        # On a Coulomb metric of realistic scale (max_eig >= 1, the regime
        # every cell in the suites occupies) it reproduces the validated
        # relative cut exactly.
        if float(w[-1]) >= 1.0:
            assert floored.tolist() == relative.tolist(), (
                f"plain-metric floor diverged from the validated relative "
                f"cut at scale {scale} (max_eig {w[-1]:.3e})"
            )


def test_mdf_dressed_metric_threshold_floor():
    """The dressed metric carries its own floor; the no-PW limit does not.

    ``J-tilde`` is only the plain Coulomb metric when the PW block is empty
    (``mdf_ke_cutoff <= 0``), which is the compcell-equivalent sanity limit.
    The floor must not apply there, or that bit-identity test would break.
    """
    from vibeqc.aux_basis import (
        _MDF_DRESSED_METRIC_MIN_EIG_FRACTION as FRAC,
        _mdf_dressed_metric_threshold as resolve,
    )

    # PW block active: a value below the floor is raised to it ...
    assert resolve(1e-9, n_pw=500, max_eig=10.0, where="t") == FRAC * 10.0
    # ... a looser value is honoured (the dense-core class needs ~3e-2) ...
    assert resolve(3e-2, n_pw=500, max_eig=10.0, where="t") == 3e-2
    # ... and with no PW block the caller's value passes through untouched.
    assert resolve(1e-9, n_pw=0, max_eig=10.0, where="t") == 1e-9
    # The floor tracks the metric's scale. J-tilde's magnitude varies by
    # seven orders across cells (max_eig ~ 1e+01 on MgO/Ne, ~9e-06 on a
    # dilute H2 box, where the PWs nearly span the Gaussian aux space); a
    # fixed absolute floor calibrated on the former annihilates the whole
    # auxiliary space of the latter. Here the floor falls below the
    # caller's value, so the caller's value stands.
    assert resolve(1e-9, n_pw=500, max_eig=9.17e-06, where="t") == 1e-9


@pytest.mark.slow
def test_mdf_linear_dep_threshold_ne_box_sweep():
    """Threshold sweep on the validated Ne envelope, vs the PySCF MDF oracle.

    Promotes the 2026-08-03 triage sweep (previously recorded only as prose
    in ``pbc_gdf._reject_dense_core_mdf``) to an executable gate.

    Sun 2017 Sec. III states the trade-off this pins: "A threshold too tight
    would cause numerical instability while a threshold too loose would
    increase the basis set incompleteness error." The measured plateau of
    best accuracy runs 1e-4 .. 3e-3; 1e-1 is past it (incompleteness), and
    the pre-fix effective cut of ~1.1e-08 was inside the noise (instability).
    """
    PYSCF_MDF_NE = -126.61361316  # def2-svp-jkfit, out-of-process (§10)
    system = vq.PeriodicSystem(
        3, np.diag([10.0] * 3), [vq.Atom(10, [5.0] * 3)]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def run(thr):
        opt = vq.PeriodicRHFOptions()
        opt.max_iter = 60
        opt.conv_tol_energy = 1e-9
        opt.lattice_opts.cutoff_bohr = 30.0
        opt.lattice_opts.nuclear_cutoff_bohr = 30.0
        r = vq.run_pbc_gdf_rhf(
            system, basis, opt, aux_basis="def2-svp-jk", gdf_method="mdf",
            mdf_ke_cutoff=60.0, exxdiv="ewald", progress=False,
            gdf_linear_dep_threshold=thr,
        )
        assert r.converged
        return r.energy

    # The shipped default lands on the accuracy plateau: 2.4 uHa from the
    # oracle, against 64 uHa at the pre-fix effective cut -- a 26x
    # improvement in PySCF-MDF parity.
    e_default = run(1e-9)
    assert abs(e_default - PYSCF_MDF_NE) < 1e-5, (
        f"default-threshold MDF {e_default} vs PySCF MDF {PYSCF_MDF_NE} "
        f"(d={(e_default - PYSCF_MDF_NE) * 1e3:+.4f} mHa)"
    )
    # The floor makes anything below it identical to the floor itself
    # (Ne max_eig = 8.09, so the floor is 8.09e-4).
    assert run(1e-12) == e_default
    assert run(1e-4) == e_default
    # Past the plateau the fit loses real auxiliary support (Sun Sec. III
    # incompleteness error) and parity degrades by two orders of magnitude.
    e_loose = run(1e-1)
    assert abs(e_loose - PYSCF_MDF_NE) > 1e-4, (
        "expected incompleteness error at a too-loose threshold; "
        f"got {e_loose}"
    )


@pytest.mark.slow
def test_mdf_dense_core_no_longer_diverges():
    """The dense-core class is bounded rather than divergent.

    MgO/STO-3G Gamma MDF, the G-GDF-001 wrong-answer case. Pre-fix, the
    retained near-null shelf of J-tilde drove the SCF to grossly unphysical
    energies -- +4.67e+05 Ha measured here at a 28-bohr cutoff, and
    trial-dependent (the triage lane recorded -6.3e+04 for the same setup).
    With the threshold above J-tilde's construction error the energy is
    bounded and on the physical scale.

    This class REMAINS gated (``_reject_dense_core_mdf``): the residual is
    ~92 mHa at the loosest useful threshold, which is incompleteness error,
    not production accuracy. What this pins is that the *divergence* is
    gone, so a regression back into it is caught.
    """
    import vibeqc.pbc_gdf as pg

    PYSCF_MDF_MGO = -271.0499046  # out-of-process oracle (§10)
    ang2bohr = 1.8897259886
    a = 4.211 * ang2bohr
    prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    o = 0.5 * (prim[0] + prim[1] + prim[2])
    system = vq.PeriodicSystem(
        3, prim, [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, o.tolist())]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opt = vq.PeriodicRHFOptions()
    opt.max_iter = 100
    opt.conv_tol_energy = 1e-9
    opt.lattice_opts.cutoff_bohr = 28.0
    opt.lattice_opts.nuclear_cutoff_bohr = 28.0

    # The production entry point still fails closed on this class.
    with pytest.raises(NotImplementedError, match="dense-core"):
        vq.run_pbc_gdf_rhf(
            system, basis, opt, aux_basis="def2-svp-jk", gdf_method="mdf",
            mdf_ke_cutoff=40.0, progress=False,
        )

    # Behind the gate, the answer is now bounded instead of divergent.
    orig = pg._reject_dense_core_mdf
    pg._reject_dense_core_mdf = lambda *a, **k: None
    try:
        e = pg.run_pbc_gdf_rhf(
            system, basis, opt, aux_basis="def2-svp-jk", gdf_method="mdf",
            mdf_ke_cutoff=40.0, exxdiv="ewald", progress=False,
            gdf_linear_dep_threshold=3e-2,
        ).energy
    finally:
        pg._reject_dense_core_mdf = orig
    # Pre-fix this was +4.67e+05 Ha. Bounded now, and still short of
    # production accuracy by ~92 mHa -- which is why the gate stays.
    assert abs(e - PYSCF_MDF_MGO) < 0.5, (
        f"dense-core MDF {e} vs PySCF MDF {PYSCF_MDF_MGO} "
        f"(d={(e - PYSCF_MDF_MGO) * 1e3:+.1f} mHa) -- divergence regression"
    )
    assert abs(e - PYSCF_MDF_MGO) > 1e-3, (
        "dense-core MDF unexpectedly reached production accuracy; if the "
        "residual incompleteness error has been fixed, lift the gate in "
        "pbc_gdf._reject_dense_core_mdf and update this test"
    )


def test_accumulated_rcut_budgets_the_whole_lattice_sum():
    """``precision`` must bound the SUM's error, not the largest term's.

    ``estimate_rcut_pyscf`` bounds one shell pair's contribution at
    ``rcut``; a lattice sum drops many at once, so the error that lands in
    the assembled matrix is larger by roughly the number of cells. That is
    why the shipped ``precision = 1e-8`` delivered 2.4e-07 on MgO and
    7.1e-06 on LiH rocksalt (recorded on this issue 2026-08-14), and why
    the dilute H2 box -- fewest neighbours inside ``rcut`` -- was the only
    one that nearly delivered what it asked.

    :func:`estimate_rcut_accumulated` divides the budget by that count, so
    it must reach further than the plain estimator at equal ``precision``,
    and exactly as far as the plain estimator reaches at the divided one.
    """
    from vibeqc.lattice_screening import (
        estimate_rcut_accumulated,
        estimate_rcut_pyscf,
        lattice_cell_count,
    )

    system, basis = _mgo_primitive()
    precision = 1e-12

    plain = estimate_rcut_pyscf(basis, precision=precision)
    accumulated = estimate_rcut_accumulated(basis, system, precision=precision)
    assert accumulated > plain

    # Self-consistency: at the returned radius, the per-term budget the
    # count implies is the one the radius was solved for.
    n_cells = lattice_cell_count(system, accumulated)
    assert n_cells > 1.0
    assert estimate_rcut_pyscf(
        basis, precision=precision / n_cells
    ) == pytest.approx(accumulated, rel=1e-6)

    # Buying orders of accuracy is cheap: the radius grows only as
    # sqrt(log(1/precision)), which is what makes a dedicated metric
    # precision affordable at all.
    assert accumulated < 2.5 * estimate_rcut_pyscf(basis, precision=1e-8)

    # extra_multiplicity carries the element bound to the eigenvalue one.
    assert estimate_rcut_accumulated(
        basis, system, precision=precision, extra_multiplicity=100.0
    ) > accumulated


def test_dressed_metric_is_psd_and_accurate_at_precision_j2c():
    """``J-tilde`` is a Gram matrix, so a negative eigenvalue is an error.

    Sun 2017 Eq. 20 subtracts from the Coulomb metric the Coulomb inner
    product of the aux functions' plane-wave projections. Because the PWs
    are orthogonal in that metric, what is left is
    ``J-tilde_PQ = ((1-Pi) phi_P | (1-Pi) phi_Q)`` with ``Pi`` the PW
    projector -- a Gram matrix of the residuals, hence positive
    SEMI-definite exactly. Eq. 20 is deliberately singular (Sec. II B), so
    eigenvalues at zero are expected and fine; eigenvalues meaningfully
    BELOW zero are construction error and nothing else, which makes
    ``min(eig)`` a direct, cell-intrinsic readout of how well the metric
    was built.

    At the general cell precision MgO/STO-3G/def2-svp-jk carried
    ``min(eig) = -5.4e-06`` and a spectrum error of the same size -- the
    2026-08-14 diagnosis on this issue. With ``J-tilde`` built at
    :data:`~vibeqc.aux_basis._MDF_PRECISION_J2C` both sit at round-off.
    """
    import vibeqc.aux_basis as ab
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.lattice_screening import (
        estimate_rcut_accumulated,
        estimate_rcut_pyscf,
    )

    system, _basis = _mgo_primitive()
    mol = system.unit_cell_molecule()
    aux = vq.BasisSet(mol, "def2-svp-jk")
    modrho = ab.make_modrho_aux_basis(aux, mol)
    compensating = ab.make_compensating_basis(modrho, mol, eta=1.0)
    fused = ab.make_fused_basis(modrho, compensating, mol)
    A = ab.fuse_transform_matrix(modrho, compensating)

    def dressed(cutoff_bohr):
        opts = LatticeSumOptions()
        opts.cutoff_bohr = float(cutoff_bohr)
        opts.nuclear_cutoff_bohr = float(cutoff_bohr)
        M = A @ np.asarray(
            ab.compute_2c_eri_lattice(fused, system, opts)
        ) @ A.T
        M = 0.5 * (M + M.T)
        volume = float(abs(np.linalg.det(np.asarray(system.lattice, float))))
        G = ab.rsgdf_dense_g_mesh(system, 40.0)
        g2 = (G ** 2).sum(axis=1)
        nonzero = g2 > 1e-12
        coulomb = (4.0 * np.pi) / g2[nonzero] / volume
        rho = A @ ab.rsgdf_aux_fourier_transform(fused, G[nonzero])
        projected = np.real((rho.conj() * coulomb[None, :]) @ rho.T)
        dressed_metric = M - 0.5 * (projected + projected.T)
        return np.linalg.eigvalsh(
            0.5 * (dressed_metric + dressed_metric.T)
        )

    reference = dressed(45.0)
    loose = dressed(estimate_rcut_pyscf(fused, precision=1e-8))
    tight = dressed(
        estimate_rcut_accumulated(
            fused, system, precision=ab._MDF_PRECISION_J2C,
            extra_multiplicity=modrho.nbasis,
        )
    )

    # The defect, still reproducible at the general precision.
    assert loose[0] < -1e-6
    assert np.abs(loose - reference).max() > 1e-6
    # ... and gone at the dedicated one. A Gram matrix's eigenvalues may
    # sit at zero but must not go meaningfully below it.
    assert tight[0] > -1e-10, f"J-tilde not PSD: min eig = {tight[0]:.3e}"
    assert np.abs(tight - reference).max() < 1e-10


def test_precision_j3c_is_off_by_default_and_measured_unnecessary():
    """The 3-centre companion is deliberately not switched on.

    ``L = U_keep^T (T - PW proj) / sqrt(lambda)`` divides the 3-centre by
    the same square root as the metric, so on paper it wants the same
    treatment. Measured 2026-09-17 it does not: the metric's error is
    amplified by ``1/lambda`` and the 3-centre's only by
    ``1/sqrt(lambda)``, and at every threshold the dressed metric admits,
    the 3-centre at ``rcut_precision`` is already inside its budget --
    switching it on left Ne, H2 and a full MgO threshold sweep unchanged
    while costing ~20% of the run.
    """
    import vibeqc.aux_basis as ab

    assert ab._MDF_PRECISION_J3C is None
    assert ab._MDF_PRECISION_J2C == 1.0e-12
