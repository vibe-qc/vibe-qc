"""Regression gates from the 2026-05-30 end-to-end periodic audit.

Three independent findings are pinned here:

1. ``KPoints.auto(l)`` was ~1.76x too dense per axis (kept the 2π factor
   in the bohr⁻¹→Å⁻¹ conversion). It must agree per-axis with
   ``KPoints.from_kspacing(2π/l)`` (VASP's documented Auto↔KSPACING
   equivalence). Fixed in ``kpoints.py``.

2. ``madelung_constant_for_cell`` is a 3D Ewald self-energy and silently
   returned a meaningless negative number for 1D/2D cells (then fed it
   into the exxdiv='ewald' K-shift). It must refuse dim<3. Fixed in
   ``madelung.py``.

3. Multi-k GDF (``run_krhf_periodic_gdf(use_compcell=True)``) used to
   return a catastrophically wrong, *converged* energy on tight ionic
   crystals (LiH primitive FCC at kmesh=(2,2,2): ~ -2495 Ha vs PySCF
   -7.92 Ha) — the per-q ``Lpq`` cache assumed a momentum-transfer-only
   cderi, valid only for vacuum boxes. **M2 landed**: the multi-k GDF
   ``Lpq`` is now built via the symmetry-preserving all-FT path resolved
   per ``(k_i, k_j)`` pair, with a k-mesh-aware (supercell) exxdiv
   Madelung. LiH (2,2,2) reaches -7.92039 Ha (+1.6 mHa of PySCF GDF,
   chemical accuracy). The CLAUDE.md §7 energy-sanity guard
   (``run_krhf_periodic_gdf`` raises on a converged non-physical energy)
   stays as a silent backstop. This test flipped from pinning the guard
   to asserting PySCF parity; µHa parity (compensated-charge fused aux)
   is the GDF chat's tracked follow-up.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq


def _si_primitive():
    a = 5.43 / 0.529177210903  # Si conventional a (bohr)
    A = (a / 2) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return vq.PeriodicSystem(
        3, A, [vq.Atom(14, [0, 0, 0]), vq.Atom(14, [a / 4, a / 4, a / 4])]
    )


def _lih_primitive():
    a = 4.084 / 0.529177210903
    A = (a / 2) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    sys = vq.PeriodicSystem(
        3, A, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [a / 2, a / 2, a / 2])]
    )
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    return sys, basis


# --- Finding 1: KPoints.auto density ---------------------------------


@pytest.mark.parametrize("length", [20.0, 25.0, 30.0, 50.0])
def test_auto_consistent_with_from_kspacing(length):
    """auto(l) must match from_kspacing(2π/l) per axis (VASP equivalence).

    The +0.5 round-up in Auto vs the plain ceil in from_kspacing allows a
    one-division difference, but not the ~1.76x over-density the bug had.
    """
    s = _si_primitive()
    m_auto = np.asarray(vq.KPoints.auto(s, length).mesh)
    m_ks = np.asarray(vq.KPoints.from_kspacing(s, 2.0 * np.pi / length).mesh)
    assert np.all(np.abs(m_auto - m_ks) <= 1), (
        f"auto({length})={tuple(m_auto)} must agree within ±1 per axis with "
        f"from_kspacing(2π/{length})={tuple(m_ks)}."
    )


def test_auto_not_overdense_regression():
    """Pin the specific pre-fix symptom: auto(30) on Si was 18×18×18."""
    s = _si_primitive()
    mesh = vq.KPoints.auto(s, 30.0).mesh
    assert max(mesh) <= 13, (
        f"auto(30) on Si primitive returned {tuple(mesh)}; the 1.76x-too-"
        "dense bug (18×18×18) has regressed."
    )


# --- Finding 2: madelung dim<3 guard ---------------------------------


@pytest.mark.parametrize("dim", [1, 2])
def test_madelung_rejects_low_dim(dim):
    from vibeqc.madelung import madelung_constant_for_cell

    lat = np.diag([5.0, 25.0, 25.0]) if dim == 1 else np.diag([5.0, 5.0, 25.0])
    s = vq.PeriodicSystem(dim, lat, [vq.Atom(1, [0, 0, 0])])
    with pytest.raises(NotImplementedError, match="dim=3"):
        madelung_constant_for_cell(s)


def test_madelung_3d_still_works():
    """The 3D path is unchanged and positive (sanity)."""
    from vibeqc.madelung import madelung_constant_for_cell

    a = 4.084 / 0.529177210903
    s = vq.PeriodicSystem(3, np.diag([a, a, a]), [vq.Atom(1, [0, 0, 0])])
    xi = madelung_constant_for_cell(s)
    assert xi > 0.0


# --- Finding 3: multi-k compcell GDF catastrophic on tight ionic -----


@pytest.mark.slow
def test_multik_compcell_gdf_lih_physical():
    """LiH primitive FCC multi-k GDF reaches µHa parity with PySCF.

    M2 milestone landed: ``run_krhf_periodic_gdf(use_compcell=True)`` now
    builds the multi-k GDF ``Lpq`` via the symmetry-preserving all-FT path
    resolved per ``(k_i, k_j)`` pair (``build_lpq_bloch_native_fft``), with
    the exxdiv-ewald shift using the k-mesh-aware (supercell) Madelung. The
    prior per-q compcell ``Lpq`` path converged to ~-2495 Ha (the §7 guard
    caught that silent corruption); the FT path lands at the physical
    energy and the guard goes quiet.

    Published target (out-of-process PySCF; CLAUDE.md §10):
    ``pyscf.pbc.scf.KRHF(cell, 2x2x2).density_fit()``, ``exxdiv='ewald'``,
    ``cell.unit='B'`` -> E = -7.92200323 Ha (GDF). vibe-qc:

      * cutoff 15 bohr / ke=200 (production default): -7.92039 Ha,
        +1.6 mHa (chemical accuracy). The residual is Bloch-pair
        cell-list truncation, NOT a DF-method floor.
      * cutoff 30 bohr / ke=200: -7.92200467 Ha, **-1.4 µHa** — genuine
        µHa parity (PySCF's own GDF↔RSDF differ by ~1-14 µHa here).

    Both vibe-qc values re-measured 2026-08-05 (PySCF 2.13.1 run out of
    process); the reference is self-converged to 0.03 µHa between
    cell.precision 1e-8 and 1e-10. The µHa figure previously read -1.0.
    Note that ``compcell_eta`` and ``apply_aft_correction`` below are
    inert here: ``gdf_method`` defaults to ``"rsgdf"``, whose per-pair
    builder (``build_lpq_bloch_native_fft``) takes neither.

    Raising the lattice cutoff (the Bloch-pair cell list) is the accuracy
    knob; no compensated-charge / MDF builder is needed for LiH.
    """
    sys, basis = _lih_primitive()
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-8
    E_PYSCF_GDF = -7.92200323

    # (a) Production default (cutoff 15 / ke 200): physical + chemical
    # accuracy. The §7 guard (check_energy_sanity=True) must NOT fire.
    r = vq.run_krhf_periodic_gdf(
        sys,
        basis,
        kmesh=(2, 2, 2),
        options=opts,
        aux_basis="def2-svp-jk",
        use_compcell=True,
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    assert r.converged, "multi-k GDF LiH SCF did not converge"
    assert -12.0 < r.energy < -4.0, (
        f"multi-k GDF LiH energy {r.energy:.4f} Ha is non-physical."
    )
    assert abs(r.energy - E_PYSCF_GDF) < 5.0e-3, (
        f"multi-k GDF LiH energy {r.energy:.6f} Ha is "
        f"{1e3 * (r.energy - E_PYSCF_GDF):+.3f} mHa from PySCF GDF (>5 mHa)."
    )

    # (b) µHa parity at the converged lattice cutoff (cutoff 30 / ke 200).
    # The bar Mike set (2026); pins that the FT path reaches µHa, not just
    # chem-acc. ~4 min wall-clock (64 (k_i,k_j) cderi builds) -> @slow.
    opts_uha = vq.PeriodicRHFOptions()
    opts_uha.use_diis = True
    opts_uha.max_iter = 60
    opts_uha.conv_tol_energy = 1e-9
    opts_uha.lattice_opts.cutoff_bohr = 30.0
    opts_uha.lattice_opts.nuclear_cutoff_bohr = 30.0
    r_uha = vq.run_krhf_periodic_gdf(
        sys,
        basis,
        kmesh=(2, 2, 2),
        options=opts_uha,
        aux_basis="def2-svp-jk",
        use_compcell=True,
        compcell_eta=0.25,
        apply_aft_correction=False,
        progress=False,
    )
    assert r_uha.converged
    # 50 µHa bound: comfortably µHa-scale (observed -1.4 µHa), robust to
    # cross-machine numerical drift; clearly distinguishes µHa from mHa.
    assert abs(r_uha.energy - E_PYSCF_GDF) < 5.0e-5, (
        f"multi-k GDF LiH at cutoff 30 / ke 200 is "
        f"{1e6 * (r_uha.energy - E_PYSCF_GDF):+.1f} µHa from PySCF GDF "
        f"{E_PYSCF_GDF:.8f} Ha (>50 µHa — µHa parity regressed)."
    )
