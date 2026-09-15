"""GPW multi-k: S(k) and H(k) must be lattice-summed over ONE R-set.

Registry: GPW-SK-HK-RSET-INCONSISTENT (agentic-loop/bug-claims.md in the
loop checkout). Follow-up to GPW-MULTIK-OVERLAP-NOT-PSD, whose validated
overlap-cutoff remedy was backed out because completing the overlap lattice
sum ALONE moved LiF/def2-SVP/(2,2,2)/PBE from -107.25 Ha to -125.43 Ha --
"and nobody can explain it" (HANDOVER_GPW_MULTIK_OVERLAP_PSD.md).

The explanation, measured here with NO grid, NO SCF and NO density: the
multi-k GPW drivers build the generalized-eigenproblem pencil
``H(k) C = S(k) C eps`` from three INDEPENDENTLY truncated real-space
lattice sums -- S/T from ``multik_one_electron_lattice_options``, V_ne from
a separate bare ``LatticeSumOptions`` (EWALD_3D dispatch), and the local
Hartree+XC block from Bloch-AO grid tables at a hardcoded 25 bohr reach.

Why mixed truncation collapses the pencil: with increasing ``|R|`` each
neglected far-field one-electron term "becomes proportional to S_12^g, the
coefficient of proportionality being independent of omega_1, omega_2 and
g. Such terms do not affect eigenvectors and displace eigenvalues by a
constant inessential quantity" -- Pisani & Dovesi, Int. J. Quantum Chem.
17, 501 (1980), doi:10.1002/qua.560170311, Sec. 4 p. 510 (the origin of
CRYSTAL's shared, overlap-based truncation set; their Eq. (30) sums T^g,
Z^g and the far-field Delta^g over the SAME g set as the Fock Bloch sum).
That cancellation holds EXACTLY when H and S drop the same tail. Complete
S while V_ne keeps its own truncation and the missing H tail is ~ c*S_tail
with S_tail now PRESENT in the metric: the pencil acquires
direction-dependent spurious shifts instead of one constant, and the
"eigenvalue displacement" stops being inessential. Concretely,
``compute_v_ne_ewald_3d_ft_lattice``'s G = 0 term is literally
``-v_short(G=0) * S_mu_nu(g)`` -- a constant times the overlap block --
built over V's OWN cell list.

Measured (2026-08-05 fixer, LiF/def2-SVP, (2,2,2) MP mesh, full converged
radius 51.53 bohr, ke_cutoff 200): the Hcore band-sum proxy
``sum_k w_k 2 sum_i^6 eps_i(k)`` is -103.43 Ha with everything at the flat
15 bohr, -121.04 Ha with S/T converged while V stays at 15 (the ENTIRE
-18 Ha SCF collapse, at the Hcore level), and -103.45 Ha with the uniform
converged R-set. The X-type k-points carry the whole effect; Gamma is
blind to it (the tail enters S and V with the same +1 phases there).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_gapw_j as gpw_j
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_rhf_multi_k_ewald import _canonical_orthogonalizer_complex
from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_lattice


def _lif_primitive():
    a = 4.0351 / 0.529177210903
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=float
    )
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = lattice
    sysp.unit_cell = [
        core.Atom(3, [0.0, 0.0, 0.0]),
        core.Atom(9, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    return sysp


def _lif_basis(sysp):
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    return vq.BasisSet(mol, "def2-svp")


def _band_sum_at_k(T_lat, S_lat, V_lat, k, n_occ=6, threshold=1e-4):
    """Closed-shell band-sum proxy 2 sum_i^{n_occ} eps_i of the Hcore
    pencil at one k, in the canonically-orthogonalized kept subspace (the
    same construction -- and the same 1e-4 threshold -- the compact multi-k
    driver uses)."""
    Tk = np.asarray(core.bloch_sum(T_lat, k))
    Sk = np.asarray(core.bloch_sum(S_lat, k))
    Vk = np.asarray(core.bloch_sum(V_lat, k))
    Sk = 0.5 * (Sk + Sk.conj().T)
    Hk = 0.5 * (Tk + Tk.conj().T) + 0.5 * (Vk + Vk.conj().T)
    X, n_kept = _canonical_orthogonalizer_complex(Sk, threshold=threshold)
    eps = np.linalg.eigvalsh(X.conj().T @ Hk @ X)
    return 2.0 * float(np.real(eps[:n_occ]).sum()), int(n_kept)


def test_hcore_pencil_collapses_when_vne_keeps_its_own_truncation():
    """The -18 Ha is mixed R-set truncation, present with no SCF at all.

    Economical form of the mechanism measurement: one X-type k-point of the
    (2,2,2) mesh (Gamma is blind to the defect), a 35 bohr "converged" S/T
    radius (the LiF/def2-SVP overlap tail is ~1e-4-converged there, against
    ~1e0 at 15 bohr), and a 60 Ha FT mesh for the analytic V_ne build (the
    effect is real-space; the G mesh is irrelevant to it). Measured:
    band-sum -103.52 (uniform 15) / -120.29 (S/T at 35, V at 15) /
    -103.59 (uniform 35). The mixed pencil collapses by -16.7 Ha; the two
    uniform pencils agree to 0.07 Ha.
    """
    sysp = _lif_primitive()
    basis = _lif_basis(sysp)
    kmesh = core.monkhorst_pack(sysp, [2, 2, 2])
    kpts = np.asarray(kmesh.kpoints, dtype=float)
    # The X-type point: the largest |k| entry of the (2,2,2) MP mesh.
    k_x = kpts[int(np.argmax(np.linalg.norm(kpts, axis=1)))]

    def _st(cut):
        lo = core.LatticeSumOptions()
        lo.cutoff_bohr = float(cut)
        return (
            core.compute_kinetic_lattice(basis, sysp, lo),
            core.compute_overlap_lattice(basis, sysp, lo),
        )

    def _vne(cut):
        lo = core.LatticeSumOptions()
        lo.cutoff_bohr = float(cut)
        return compute_v_ne_ewald_3d_ft_lattice(
            basis, sysp, lo, ke_cutoff=60.0
        )

    T15, S15 = _st(15.0)
    T35, S35 = _st(35.0)
    V15 = _vne(15.0)
    V35 = _vne(35.0)

    flat, kept_flat = _band_sum_at_k(T15, S15, V15, k_x)
    mixed, kept_mixed = _band_sum_at_k(T35, S35, V15, k_x)
    uniform, kept_uniform = _band_sum_at_k(T35, S35, V35, k_x)

    # The mixed and uniform arms share the same S -- same kept subspace --
    # so the collapse below is not a linear-dependence-projection artifact.
    # (The flat arm legitimately keeps one direction fewer at this k: the
    # truncated S(k) has a negative eigenvalue there, which is the
    # already-pinned GPW-MULTIK-OVERLAP-NOT-PSD defect.)
    assert kept_mixed == kept_uniform
    assert kept_flat <= kept_uniform

    # The two CONSISTENT pencils agree at the sub-0.2 Ha level ...
    assert abs(uniform - flat) < 0.2, (
        f"uniform-R-set pencils disagree: flat-15 {flat:+.4f} vs "
        f"uniform-35 {uniform:+.4f}"
    )
    # ... while the mixed one collapses by more than 10 Ha (measured
    # -16.7 Ha). This is the whole GPW-SK-HK-RSET-INCONSISTENT defect,
    # reproduced in the Hcore pencil.
    assert mixed - uniform < -10.0, (
        f"expected the mixed-truncation pencil to collapse; got "
        f"mixed {mixed:+.4f} vs uniform {uniform:+.4f}"
    )


def test_multik_gpw_operators_share_one_converged_rset():
    """GUARDS THE FIX (this file's reproducer commit pinned the defect).

    The multi-k drivers must build S/T and V_ne over ONE shared,
    basis-derived R-set: the S/T options carry the
    ``bloch_overlap_cutoff_bohr`` criterion radius (which keeps S(k)
    positive semidefinite), and the V_ne options carry the SAME cutoff
    with the EWALD_3D dispatch selected -- never a fresh
    ``LatticeSumOptions`` with its bare 15-bohr flat default.
    """
    from vibeqc.lattice_screening import bloch_overlap_cutoff_bohr

    sysp = _lif_primitive()
    basis = _lif_basis(sysp)
    lat_opts = gpw_j.multik_one_electron_lattice_options(basis, sysp)
    lat_opts_v = gpw_j.multik_v_ne_lattice_options(basis, sysp)

    r_criterion = bloch_overlap_cutoff_bohr(basis, sysp)
    assert float(lat_opts.cutoff_bohr) == pytest.approx(r_criterion)
    assert float(lat_opts_v.cutoff_bohr) == pytest.approx(r_criterion)
    assert lat_opts_v.coulomb_method == core.CoulombMethod.EWALD_3D
    # And the criterion radius really is beyond the diffuse def2-SVP tail
    # on this compact cell (51.53 bohr measured), not the bare default.
    assert r_criterion > 50.0
