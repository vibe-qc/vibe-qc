"""Analytic nuclear gradient (forces) for the Γ-CCM (``aiccm2026dev-a``).

The finite-difference gate :func:`vibeqc.periodic.ccm.properties.ccm_numerical_gradient`
displaces each *unit-cell* atom ±h and rebuilds the whole ``CCMSystem`` (so every
periodic image moves together — the cyclic Born–von-Kármán constraint), costing
``6·n_basis_atoms`` SCF evaluations. This module computes the same gradient
**analytically**, term by term, by differentiating the WSSC-folded effective
integrals and contracting against the converged CCM densities — no new C++.

The derivation is paper-grade in
``docs/manuscripts/aiccm_a_forces.md`` (§1.1–1.7); each equation here cites the
section it implements. The four terms (each finite-difference-checkable in isolation
against the gate, all DONE for RHF + UHF) are:

1. ``∂V_nn^CCM/∂R``           — this module (§1.5)
2. 1-e ``Σ P ∂h^CCM/∂R``      — L₂ᵀ(T) + L_Vᵀ(V)  (§1.4)
3. overlap Pulay ``−Σ W ∂S``  — L₂ᵀ(W)            (§1.4)
4. 2-e ``Σ Γ ∂(μν|λσ)^CCM/∂R``— L₄ᵀ(Γ) + casscf  (§1.4)

``run_ccm_rhf_gradient`` / ``run_ccm_uhf_gradient`` / ``run_ccm_rks_gradient`` /
``run_ccm_uks_gradient`` / ``run_ccm_mp2_gradient`` / ``run_ccm_ump2_gradient`` /
``run_ccm_ccsd_gradient`` / ``run_ccm_uccsd_gradient`` sum these over the four term
cores (which take explicit ``P``/``W``/``Γ`` arrays, §1.7) + ``V_nn``.
MP2/UMP2/CCSD/UCCSD feed the cores the *relaxed* densities — amplitude/Λ 1-RDM +
Z-vector (CCM-CPHF) orbital response, separable + non-separable 2-PDM, correlation
energy-weighted density (§1.8; spin-coupled CPHF for UMP2, §1.8.4; complex-step CCSD
Lagrangian, §1.9; open-shell UCCSD, §1.9.5). KS/UKS add a fifth term — the *molecular*
XC gradient on the supercell (``xc_pulay_gradient_rks`` / ``xc_pulay_gradient_uks``),
which does NOT fold (the CCM XC is the ordinary molecular functional of the supercell
density) and so is image-summed over supercell copies only. All terms fold back to a
per-unit-cell-atom ``(n_basis_atoms, 3)`` array via the
cyclic image-sum (§1.3): the per-supercell-atom (or per-pad-atom) molecular
gradient is summed over the supercell copies — and padded images — of each
unit-cell atom. The WSSC weights are SUPERCELL-indexed (``W[g][A,B]`` over
``ccm.n_atoms`` atoms), so every scatter/contraction is indexed by supercell atom
``A``; the fold-back to unit-cell ``β`` happens only at the end.

Regime: the four-center term is a dense ``n_pad_ao⁴`` validation path — small /
1-D / 2-D clusters only (the ``_check_padded_eri_size`` guard). 3-D analytic forces
need a scalable weighted ERI-gradient C++ kernel (a later milestone).

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550; Helgaker, Jørgensen & Olsen, *Molecular Electronic-Structure
Theory* (gradient theory, Eq. 12.5.7); Pulay, Mol. Phys. 17, 197 (1969).
"""

from __future__ import annotations

import numpy as np

from vibeqc._vibeqc_core import (
    Atom,
    Molecule,
    compute_external_charge_density_gradient,
    one_electron_gradient_contribution,
    overlap_gradient_contribution,
    two_electron_gradient_casscf,
)

from .padded import (
    PaddedCluster,
    _check_padded_eri_size,
    build_padded_cluster,
    eri_cells,
    wssc_cells,
)

__all__ = [
    "run_ccm_rhf_gradient",
    "run_ccm_uhf_gradient",
    "run_ccm_rks_gradient",
    "run_ccm_uks_gradient",
    "run_ccm_mp2_gradient",
    "run_ccm_ump2_gradient",
    "run_ccm_ccsd_gradient",
    "run_ccm_uccsd_gradient",
    "ccm_nuclear_repulsion_gradient",
    "ccm_kinetic_gradient",
    "ccm_nuclear_attraction_gradient",
    "ccm_overlap_pulay_gradient",
    "ccm_two_electron_gradient",
    "fold_supercell_gradient_to_cell",
    "fold_pad_gradient_to_cell",
]


def fold_supercell_gradient_to_cell(ccm, grad_atom: np.ndarray) -> np.ndarray:
    """Image-sum a per-supercell-atom gradient to per-unit-cell-atom (§1.3b).

    The cyclic constraint moves *every* supercell copy of a unit-cell atom
    together, so the unit-cell-atom gradient is the sum of the gradients of its
    supercell copies::

        grad_cell[β] = Σ_{A : atom_beta[A] = β}  grad_atom[A] .          (§1.3b)

    ``grad_atom`` is ``(n_atoms, 3)`` (supercell atoms); the result is
    ``(n_basis_atoms, 3)`` (unit-cell atoms), matching ``ccm_numerical_gradient``.
    """
    grad_atom = np.asarray(grad_atom, dtype=float)
    if grad_atom.shape != (ccm.n_atoms, 3):
        raise ValueError(
            f"grad_atom must be (n_atoms={ccm.n_atoms}, 3), got {grad_atom.shape}"
        )
    grad_cell = np.zeros((ccm.n_basis_atoms, 3), dtype=float)
    # atom_beta[A] is the unit-cell atom β that supercell atom A is a copy of
    # (CCMSystem._decompose_atoms). np.add.at scatters with repeated indices.
    np.add.at(grad_cell, ccm.atom_beta, grad_atom)
    return grad_cell


def fold_pad_gradient_to_cell(ccm, pad: PaddedCluster, grad_pad: np.ndarray) -> np.ndarray:
    """Image-sum a per-padded-atom gradient to per-unit-cell-atom (§1.3b).

    The padded-cluster atom ``p`` is supercell atom ``A = pad.pad_atom_of[p][0]``
    in image cell ``g``; the cyclic constraint sums every padded image *and* every
    supercell copy of unit-cell atom ``β``::

        grad_cell[β] = Σ_{p : atom_beta[ pad.pad_atom_of[p].A ] = β}  grad_pad[p] .

    ``grad_pad`` is ``(n_pad_atom, 3)``; the result is ``(n_basis_atoms, 3)``.
    """
    grad_pad = np.asarray(grad_pad, dtype=float)
    n_pad = len(pad.pad_atom_of)
    if grad_pad.shape != (n_pad, 3):
        raise ValueError(
            f"grad_pad must be (n_pad_atom={n_pad}, 3), got {grad_pad.shape}"
        )
    # pad atom p -> supercell atom A -> unit-cell atom β.
    cell_of_pad = np.array(
        [int(ccm.atom_beta[A]) for (A, _g) in pad.pad_atom_of], dtype=int
    )
    grad_cell = np.zeros((ccm.n_basis_atoms, 3), dtype=float)
    np.add.at(grad_cell, cell_of_pad, grad_pad)
    return grad_cell


def _pad_molecule(pad: PaddedCluster) -> Molecule:
    """Reconstruct the padded-cluster ``Molecule`` (atom order == basis order).

    :func:`build_padded_cluster` builds this molecule internally to construct the
    basis but does not retain it; the molecular gradient primitives need it for the
    atom positions. Rebuilt in the same ``pad_atom_of`` order so per-atom gradient
    rows line up with :func:`fold_pad_gradient_to_cell`.
    """
    atoms = [
        Atom(int(pad.pad_Z[p]), pad.pad_positions[p].tolist())
        for p in range(len(pad.pad_atom_of))
    ]
    n_elec = sum(int(z) for z in pad.pad_Z)
    return Molecule(atoms, 0, 1 if n_elec % 2 == 0 else 2)


def _l2t_adjoint(ccm, pad: PaddedCluster, c_ccm: np.ndarray) -> np.ndarray:
    """Adjoint of the two-center fold ``L₂`` (§1.4 ``L₂ᵀ``): scatter ``C^CCM`` to padded.

    The fold is ``M^CCM = ½(out + outᵀ)``, ``out_μν = Σ_g w^g_{a(μ)a(ν)} M_pad[μ, ν@g]``
    (``_fold_two_center``). For symmetric ``C^CCM`` and symmetric ``M_pad`` the energy
    contraction ``Σ C^CCM M^CCM`` collapses to ``Σ_{p,q} X_pad[p,q] M_pad[p,q]`` with
    the home-row scatter

        X_pad[μ, ν@g] += w^g_{a(μ)a(ν)} · C^CCM_μν      (home rows μ),

    and only ``sym(X_pad)`` survives contraction with the symmetric ``∂M_pad/∂R``, so
    the padded contraction matrix is ``½(X_pad + X_padᵀ)`` (§1.4c). Returns the
    ``(n_pad_ao, n_pad_ao)`` symmetric padded density to feed a molecular gradient
    routine.
    """
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    n_pad_ao = pad.basis.nbasis
    home = np.arange(pad.n_ref_ao)
    x_pad = np.zeros((n_pad_ao, n_pad_ao))
    for g, w_atom in weights.items():
        cols = pad.cell_to_cols.get(g)
        if cols is None:
            raise ValueError(
                f"padded cluster too small: minimum-image cell {g} not built."
            )
        # w^g_{a(μ)a(ν)} · C^CCM_μν scattered to [home μ, ν@g].
        x_pad[home[:, None], cols[None, :]] += w_atom[ao[:, None], ao[None, :]] * c_ccm
    return 0.5 * (x_pad + x_pad.T)


def _energy_weighted_density(eps, C, n_occ, occ_number) -> np.ndarray:
    """Energy-weighted density ``W = Σ_i n_i ε_i C_i C_iᵀ`` over the lowest ``n_occ``.

    ``occ_number`` is 2 for closed-shell RHF spatial orbitals, 1 for UHF spin
    orbitals (per spin). Returns a symmetric ``(nbf, nbf)`` matrix.
    """
    eps = np.asarray(eps, dtype=float)
    C = np.asarray(C, dtype=float)
    occ = np.zeros(C.shape[1])
    occ[:n_occ] = occ_number
    return (C * (occ * eps)[None, :]) @ C.T


def _overlap_pulay_gradient_W(ccm, W: np.ndarray) -> np.ndarray:
    """Overlap Pulay term core ``−Σ W ∂S^CCM/∂R`` for a given energy-weighted ``W``.

    Folds via the two-center adjoint ``W_pad = L₂ᵀ(W)`` (§1.4) then drives the
    molecular ``overlap_gradient_contribution`` (which already carries the leading
    minus) on the padded cluster, and image-sums (§1.3b). Spin-independent: RHF
    passes ``W``, UHF passes ``W^α + W^β``.
    """
    pad = build_padded_cluster(ccm, wssc_cells(ccm))
    pad_mol = _pad_molecule(pad)
    W_pad = _l2t_adjoint(ccm, pad, np.asarray(W, dtype=float))
    # overlap_gradient_contribution returns −Σ W ∂S/∂R per (padded) atom.
    grad_pad = np.asarray(
        overlap_gradient_contribution(pad.basis, pad_mol, W_pad), dtype=float
    )
    return fold_pad_gradient_to_cell(ccm, pad, grad_pad)


def ccm_overlap_pulay_gradient(ccm, scf) -> np.ndarray:
    """Analytic overlap Pulay term ``−Σ W ∂S^CCM/∂R`` per unit-cell atom, step 3 (RHF).

    The basis-relaxation (Pulay) term of (§1.1), with the closed-shell
    energy-weighted density ``W_μν = Σ_i 2 ε_i C_μi C_νi`` (§1.1c). Returns
    ``(n_basis_atoms, 3)``.
    """
    n_occ = int(round(ccm.supercell.n_electrons() / 2))
    W = _energy_weighted_density(scf.mo_energies, scf.mo_coeffs, n_occ, 2.0)
    return _overlap_pulay_gradient_W(ccm, W)


def _kinetic_gradient_P(ccm, P: np.ndarray) -> np.ndarray:
    """Kinetic Hellmann–Feynman core ``Σ P ∂T^CCM/∂R`` for a given total density ``P``.

    ``T^CCM = L₂(T_pad)`` is a two-center fold, so ``Σ P ∂T^CCM/∂R = Σ L₂ᵀ(P)
    ∂T_pad/∂R``. Drives ``one_electron_gradient_contribution`` on the padded cluster
    with ``P_pad = L₂ᵀ(P)`` and **all-zero effective charges**, which zeroes the V_ne
    derivative kernel and leaves the pure kinetic gradient ("kinetic piece unchanged"
    per the binding), then image-sums (§1.3b). Spin-independent (pass the *total*
    density ``P^α + P^β`` for UHF).
    """
    pad = build_padded_cluster(ccm, wssc_cells(ccm))
    pad_mol = _pad_molecule(pad)
    P_pad = _l2t_adjoint(ccm, pad, np.asarray(P, dtype=float))
    zero_charges = [0.0] * len(pad.pad_atom_of)          # zero V_ne ⇒ kinetic only
    grad_pad = np.asarray(
        one_electron_gradient_contribution(pad.basis, pad_mol, P_pad, zero_charges),
        dtype=float,
    )
    return fold_pad_gradient_to_cell(ccm, pad, grad_pad)


def ccm_kinetic_gradient(ccm, scf) -> np.ndarray:
    """Analytic kinetic Hellmann–Feynman term ``Σ P ∂T^CCM/∂R`` per unit-cell atom (RHF).

    The kinetic half of the 1-e term (§1.4, step 2); ``P = 2 C_occ C_occᵀ``. Returns
    ``(n_basis_atoms, 3)``.
    """
    return _kinetic_gradient_P(ccm, scf.density)


def _nuclear_attraction_gradient_P(ccm, P: np.ndarray) -> np.ndarray:
    """Nuclear-attraction Hellmann–Feynman core ``Σ P ∂V^CCM/∂R`` (§1.4 ``L_Vᵀ``).

    ``ccm_nuclear`` (``padded.py:247``) weights each **nucleus image** ``c=(C₀,g_c)``
    by ``ω_{A,c}=w^{g_c}_{A,C₀}`` and folds the bra/ket pair independently::

        Σ P V^CCM = Σ_c Σ_{g_ν} Σ_μν P_μν · w^{g_ν}_{a(μ)a(ν)}
                      · ½(ω_{a(μ),c}+ω_{a(ν),c}) · I^c[μ, ν@g_ν] ,
        I^c[p,q] = ⟨p| −Z_{C₀}/|r−R_c| |q⟩ .

    Its gradient at fixed ``P`` has (i) the **bra/ket basis-center** derivatives and
    (ii) the **Hellmann–Feynman nucleus-position** derivative
    ``∫φ ∂(−Z/|r−R_c|)/∂R_{C₀} φ``. Built per nucleus image ``c`` by *scattering* ``P``
    (not gathering ``I^c``) into the combined-weighted padded contraction density
    ``D^c_pad`` (mirroring ``ccm_nuclear``'s per-``c`` loop), then calling
    ``compute_external_charge_density_gradient`` with the single point charge
    ``q=−Z_{C₀}`` at ``R_c`` (so ``q·M^c = I^c``): its ``atom_grad`` carries piece (i)
    per padded atom, its ``point_grad`` carries piece (ii) for the nucleus image.
    Piece (i) image-sums via the padded atoms; piece (ii) lands on the unit-cell atom
    owning ``C₀`` (``atom_beta[C₀]``). Spin-independent (pass the *total* density).
    Returns ``(n_basis_atoms, 3)``.
    """
    P = np.asarray(P, dtype=float)
    pad = build_padded_cluster(ccm, wssc_cells(ccm))
    pad_mol = _pad_molecule(pad)
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    home = np.arange(pad.n_ref_ao)
    n_pad_ao = pad.basis.nbasis

    grad_pad = np.zeros((len(pad.pad_atom_of), 3))       # bra/ket basis-center pieces
    grad_cell = np.zeros((ccm.n_basis_atoms, 3))         # nucleus-position (HF) pieces

    # Each nucleus image c=(C0,gc) contributes only if gc is a min-image cell of some
    # pair (A,C0) — mirror ccm_nuclear's skip conditions exactly.
    for c, (C0, gc) in enumerate(pad.pad_atom_of):
        w_cell = weights.get(gc)
        if w_cell is None:
            continue
        omega_c = w_cell[:, C0]                           # ω_{A,c} over supercell atoms A
        if not np.any(omega_c):
            continue
        nucl_w = 0.5 * (omega_c[:, None] + omega_c[None, :])   # ½(ω_{A,c}+ω_{B,c})

        # D^c_pad = L₂ᵀ-style scatter of P with weight w^{g_ν}·½(ω_A+ω_B) (§1.4 L_Vᵀ).
        x_pad = np.zeros((n_pad_ao, n_pad_ao))
        for g_nu, pair_w in weights.items():
            atom_w = pair_w * nucl_w                      # (n_atoms, n_atoms)
            if not np.any(atom_w):
                continue
            cols = pad.cell_to_cols[g_nu]
            x_pad[home[:, None], cols[None, :]] += atom_w[ao[:, None], ao[None, :]] * P
        d_pad = 0.5 * (x_pad + x_pad.T)

        # q = −Z_{C0} at R_c so that q·M^c = I^c (compute_nuclear is signed-attractive).
        z = int(pad.pad_Z[c])
        ext = compute_external_charge_density_gradient(
            pad.basis, pad_mol, d_pad, [-float(z)], [pad.pad_positions[c].tolist()]
        )
        grad_pad += np.asarray(ext.atom_grad, dtype=float)         # piece (i)
        # piece (ii): the single point charge's derivative onto C0's unit-cell atom.
        grad_cell[int(ccm.atom_beta[C0])] += np.asarray(ext.point_grad, dtype=float)[0]

    return fold_pad_gradient_to_cell(ccm, pad, grad_pad) + grad_cell


def ccm_nuclear_attraction_gradient(ccm, scf) -> np.ndarray:
    """Analytic nuclear-attraction Hellmann–Feynman term ``Σ P ∂V^CCM/∂R``, step 2 (V, RHF).

    The subtlest 1-e term; ``P = 2 C_occ C_occᵀ``. See :func:`_nuclear_attraction_gradient_P`.
    Returns ``(n_basis_atoms, 3)``.
    """
    return _nuclear_attraction_gradient_P(ccm, scf.density)


def _two_electron_gradient_gamma(ccm, gamma: np.ndarray) -> np.ndarray:
    """Two-electron Hellmann–Feynman core ``½ Σ Γ ∂(μν|λσ)^CCM/∂R`` for AO 2-RDM ``Γ``.

    The four-center term (§1.4 ``L₄ᵀ``). With ``E_2e = ½ Σ Γ (μν|λσ)^CCM`` the gradient
    at fixed ``Γ`` is ``½ Σ Γ ∂(μν|λσ)^CCM/∂R = ½ Σ L₄ᵀ(Γ) ∂eri_pad/∂R``. ``Γ`` is the
    AO 2-RDM in chemist order ``μνλσ`` — closed-shell HF ``P⊗P − ½P⊗P_swap`` for RHF,
    ``P⊗P − P^α⊗P^α − P^β⊗P^β`` for UHF; the fold/scatter is spin-independent.

    ``L₄ᵀ`` mirrors ``ccm_eri_symmetric`` (``padded.py:361``) **exactly** — the home
    bra pair ``w^0``, the symmetric bridge ``¼(w_μλ+w_νλ+w_μσ+w_νσ)``, the independent
    minimum-image ket fold ``w^{g_e−g_c}``, and the closing bra↔ket transpose — but
    **scatters** ``Γ̄ = ½(Γ + Γ.transpose(2,3,0,1))`` (the adjoint of the closing
    transpose) into ``Γ_pad`` instead of gathering ``eri_pad``::

        Γ_pad[μ, ν@0, λ@g_c, σ@g_e] += w^0_{μν} · bridge · w^{g_e−g_c}_{λσ} · Γ̄_μνλσ .

    ``Γ_pad`` is 8-fold symmetrized and scaled by ½, flattened row-major, and fed to
    ``two_electron_gradient_casscf`` (which returns ``Σ G_pad ∂eri_pad/∂R`` with no
    extra factor, ``gradient.cpp:1485``); image-sum (§1.3b). Dense ``n_pad_ao⁴`` —
    small / 1-D / 2-D only (``_check_padded_eri_size`` guard). Returns
    ``(n_basis_atoms, 3)``.
    """
    gamma = np.asarray(gamma, dtype=float)
    n = ccm.nbf
    gamma_bar = 0.5 * (gamma + np.transpose(gamma, (2, 3, 0, 1)))

    pad = build_padded_cluster(ccm, eri_cells(ccm))
    _check_padded_eri_size(pad, "ccm_two_electron_gradient")
    pad_mol = _pad_molecule(pad)
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    home = np.arange(n)
    keys = list(weights.keys())
    wexp = {g: weights[g][ao[:, None], ao[None, :]] for g in keys}

    n_pad_ao = pad.basis.nbasis
    gamma_pad = np.zeros((n_pad_ao, n_pad_ao, n_pad_ao, n_pad_ao))
    cols_b = pad.cell_to_cols[(0, 0, 0)]                  # home ket-1 (ν@0) columns
    wab = wexp[(0, 0, 0)]                                 # w^0_{μν} (home bra pair)
    for gc in keys:
        wc = wexp[gc]                                     # w^{g_c}_{·λ}
        cols_c = pad.cell_to_cols[gc]
        for ge in keys:
            grel = (ge[0] - gc[0], ge[1] - gc[1], ge[2] - gc[2])
            w_pair = weights.get(grel)
            if w_pair is None:
                continue                                  # λ,σ not a min-image pair
            wcd = w_pair[ao[:, None], ao[None, :]]        # w^{g_e−g_c}_{λσ}
            cols_d = pad.cell_to_cols.get(ge)
            if cols_d is None:
                continue
            wd = wexp[ge]                                 # w^{g_e}_{·σ}
            bridge = 0.25 * (
                wc[:, None, :, None] + wc[None, :, :, None]      # w_μλ, w_νλ
                + wd[:, None, None, :] + wd[None, :, None, :]    # w_μσ, w_νσ
            )
            w4 = wab[:, :, None, None] * bridge * wcd[None, None, :, :]
            if not np.any(w4):
                continue
            # Scatter Γ̄ (cf. ccm_eri_symmetric, which gathers eri_pad here).
            np.add.at(
                gamma_pad,
                (home[:, None, None, None], cols_b[None, :, None, None],
                 cols_c[None, None, :, None], cols_d[None, None, None, :]),
                w4 * gamma_bar,
            )

    # two_electron_gradient_casscf loops over *unique* shell quartets ×degeneracy and
    # assumes the supplied Γ carries the full 8-fold ERI permutation symmetry (cf.
    # _casscf.py:216 — the CASSCF 2-RDM has only 2-fold symmetry, so it is 8-fold
    # symmetrized before the call). Our scatter places Γ̄ into distinct padded slots,
    # which breaks μ↔ν / λ↔σ symmetry on the padded layout. Since ∂eri_pad/∂R is
    # itself 8-fold symmetric, Σ gamma_pad ∂eri = Σ sym₈(gamma_pad) ∂eri, so we feed
    # the 8-fold symmetrization (then the ½ from E_2e = ½ Σ Γ (μν|λσ)).
    g = gamma_pad
    g_sym = (
        g + g.transpose(1, 0, 2, 3) + g.transpose(0, 1, 3, 2) + g.transpose(1, 0, 3, 2)
        + g.transpose(2, 3, 0, 1) + g.transpose(2, 3, 1, 0)
        + g.transpose(3, 2, 0, 1) + g.transpose(3, 2, 1, 0)
    ) / 8.0
    g_pad = 0.5 * g_sym
    grad_pad = np.asarray(
        two_electron_gradient_casscf(pad.basis, pad_mol, g_pad.reshape(-1)),
        dtype=float,
    )
    return fold_pad_gradient_to_cell(ccm, pad, grad_pad)


def ccm_two_electron_gradient(ccm, scf) -> np.ndarray:
    """Analytic two-electron Hellmann–Feynman term ``½ Σ Γ ∂(μν|λσ)^CCM/∂R``, step 4 (RHF).

    Builds the closed-shell HF AO 2-RDM ``Γ^CCM_μνλσ = P_μν P_λσ − ½ P_μλ P_νσ``
    (§0.2; already bra↔ket symmetric) and delegates to
    :func:`_two_electron_gradient_gamma`. Returns ``(n_basis_atoms, 3)``.
    """
    P = np.asarray(scf.density, dtype=float)
    gamma = np.einsum("mn,ls->mnls", P, P) - 0.5 * np.einsum("ml,ns->mnls", P, P)
    return _two_electron_gradient_gamma(ccm, gamma)


def ccm_nuclear_repulsion_gradient(ccm) -> np.ndarray:
    """Analytic ``∂V_nn^CCM/∂R`` per unit-cell atom (a.u.), build-order step 1.

    Differentiates the exact WSSC-weighted nuclear-repulsion pair list that
    :func:`vibeqc.periodic.ccm.padded.ccm_nuclear_repulsion` sums (``padded.py:437``),
    holding the WSSC weights fixed (Lemma 1.2.1: the weights are piecewise-constant
    in the atomic positions, so ``∂w/∂R = 0`` away from Wigner–Seitz boundary ties).

    Energy (per reference cell), with ``d_{ABg} = r_A − (r_B + R_g)``, ``R_g =
    g·A_cluster``, the prime excluding the on-site ``(A=B, g=0)`` term, and the ½
    removing the ordered double-count::

        V_nn^CCM = ½ Σ_g Σ_{A,B}'  w^g_{AB} Z_A Z_B / |d_{ABg}| .            (§1.5a)

    Per-supercell-atom gradient (the A-as-bra and A-as-ket halves are accumulated
    explicitly onto both endpoints, so the result matches a finite difference of
    ``ccm_nuclear_repulsion`` by construction)::

        ∂V_nn^CCM/∂r_A = − Σ_g Σ_B'  w^g_{AB} Z_A Z_B  d_{ABg} / |d_{ABg}|³ ,  (§1.5b)

    then image-summed to per-unit-cell-atom (§1.5c / §1.3b). Returns
    ``(n_basis_atoms, 3)`` (Hartree/bohr).
    """
    weights = ccm.cell_weight_matrices()           # {g: (n_atoms, n_atoms)} supercell-indexed
    pos = np.asarray(ccm.atom_positions, dtype=float)               # (n_atoms, 3) bohr
    Z = np.array([int(a.Z) for a in ccm.supercell.atoms], dtype=float)  # (n_atoms,)
    Ac = ccm.cluster_vectors                        # rows = cluster lattice vectors
    na = ccm.n_atoms
    ZZ = Z[:, None] * Z[None, :]                     # (na, na): Z_A Z_B

    grad_atom = np.zeros((na, 3), dtype=float)
    for g, w_atom in weights.items():
        Rg = np.asarray(g, dtype=float) @ Ac                        # (3,)
        # d_{ABg} = r_A − (r_B + R_g); D[A,B] is the bra→ket vector.
        D = pos[:, None, :] - (pos[None, :, :] + Rg[None, None, :])  # (na, na, 3)
        dist = np.linalg.norm(D, axis=2)                            # (na, na)
        mask = w_atom != 0.0
        if g == (0, 0, 0):
            mask = mask & ~np.eye(na, dtype=bool)                  # drop on-site (A=B, g=0)
        # Guard the masked (incl. self) entries against 1/0 before reciprocals.
        safe = np.where(mask, dist, 1.0)
        inv3 = np.where(mask, 1.0 / safe ** 3, 0.0)                # (na, na)
        coef = 0.5 * w_atom * ZZ * inv3                            # (na, na) = ½ w Z_A Z_B / d³

        # ∂/∂r_A of |d_{ABg}| = +d_{ABg}/|d|  ⇒  ∂(1/|d|)/∂r_A = −d/|d|³ (A as bra);
        # ∂/∂r_B = +d/|d|³ (B as ket). Accumulate both onto the two endpoints.
        contrib = coef[:, :, None] * D                            # (na, na, 3)
        grad_atom += -contrib.sum(axis=1)                         # A as first index (bra)
        grad_atom += +contrib.sum(axis=0)                         # A as second index (ket)

    return fold_supercell_gradient_to_cell(ccm, grad_atom)


def run_ccm_rhf_gradient(ccm, *, method: str = "aiccm2026dev-a", scf=None) -> np.ndarray:
    """Analytic Γ-CCM RHF nuclear gradient ``dE_total/dR`` per unit-cell atom (a.u.).

    The full variational gradient (§1.1), assembled from the four
    finite-difference-gated terms::

        dE/dR =  Σ P ∂h^CCM/∂R         (kinetic L₂ᵀ + nuclear-attraction L_Vᵀ)
              +  ½ Σ Γ ∂(μν|λσ)^CCM/∂R (four-center L₄ᵀ)
              −  Σ W ∂S^CCM/∂R         (overlap Pulay L₂ᵀ)
              +  ∂V_nn^CCM/∂R .        (nuclear repulsion)

    Matches :func:`vibeqc.periodic.ccm.properties.ccm_numerical_gradient` to ~1e-6
    Ha/bohr at a generic geometry (Lemma 1.2.1: away from Wigner–Seitz boundary
    ties), at a fraction of its ``6·n_basis_atoms`` SCF cost. The force is
    ``−gradient``. Returns ``(n_basis_atoms, 3)``; by translational invariance the
    per-cell forces sum to ≈ 0 (Theorem 1.6.1).

    Default ``method="aiccm2026dev-a"`` (the symmetric four-center
    :func:`~vibeqc.periodic.ccm.padded.ccm_eri_symmetric`); RHF only. Dense
    ``n_pad_ao⁴`` four-center — small / 1-D / 2-D clusters only.

    **The SCF must be tightly converged.** Unlike the energy (second-order
    insensitive to a small orbital gradient), the analytic gradient is *first-order*
    sensitive to SCF non-stationarity: the Pulay term ``−Σ W ∂S/∂R`` cancels the
    orbital response only at a stationary point (§1.1), so a loose ``F_ai`` leaks
    directly into the force at O(F_ai). The internal SCF therefore runs at
    ``conv_tol=1e-12`` (DIIS commutator ≪ 1e-6); a hand-passed ``scf`` should be
    converged at least that tightly. With a stationary SCF the gradient matches
    ``ccm_numerical_gradient`` to the finite-difference truncation floor (O(h²); the
    analytic value is exact — validated to 1.1e-8 at h=1e-4 on the H₂ chain).
    """
    if method != "aiccm2026dev-a":
        raise NotImplementedError(
            f"run_ccm_rhf_gradient supports method='aiccm2026dev-a' only, got "
            f"{method!r} (the symmetric four-center is the gated path)."
        )
    if scf is None:
        from .scf import run_ccm_rhf

        # Tight convergence is required for a correct analytic gradient (the Pulay
        # term assumes ∂E/∂κ_ai = 0); conv_tol=1e-12 drives the DIIS commutator far
        # below the 1e-6 SCF floor.
        scf = run_ccm_rhf(ccm, method=method, conv_tol=1e-12, max_iter=512)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_rhf_gradient: the internal SCF did not converge to "
                "conv_tol=1e-12; the analytic gradient needs a stationary SCF. "
                "Converge it yourself and pass scf=."
            )

    return (
        ccm_kinetic_gradient(ccm, scf)
        + ccm_nuclear_attraction_gradient(ccm, scf)
        + ccm_two_electron_gradient(ccm, scf)
        + ccm_overlap_pulay_gradient(ccm, scf)
        + ccm_nuclear_repulsion_gradient(ccm)
    )


def run_ccm_uhf_gradient(ccm, *, method: str = "aiccm2026dev-a", scf=None) -> np.ndarray:
    """Analytic Γ-CCM **UHF** nuclear gradient ``dE_total/dR`` per unit-cell atom (a.u.).

    The open-shell counterpart of :func:`run_ccm_rhf_gradient` (§1.7). The WSSC fold,
    the ``Lᵀ`` adjoints, and the cyclic image-sum are all spin-independent; only the
    densities fed to the (shared) term cores change:

    * 1-e ``Σ P ∂h^CCM/∂R``  — total density ``P = P^α + P^β`` (kinetic + V_ne);
    * overlap Pulay ``−Σ W ∂S^CCM/∂R`` — total energy-weighted density
      ``W = W^α + W^β``, each ``W^σ = Σ_i ε^σ_i C^σ_i C^σ_iᵀ`` (occupation 1 per spin
      orbital — UHF spin densities carry no factor 2);
    * 2-e ``½ Σ Γ ∂(μν|λσ)^CCM/∂R`` — the UHF AO 2-RDM (chemist order)
      ``Γ_μνλσ = P_μν P_λσ − P^α_μλ P^α_νσ − P^β_μλ P^β_νσ`` (total-density Coulomb,
      same-spin exchange; reduces to the RHF ``P⊗P − ½P⊗P`` when ``P^α=P^β=P/2``);
    * ``∂V_nn^CCM/∂R`` — geometry only (spin-independent).

    Like the RHF driver, the analytic force needs a stationary SCF (first-order
    sensitive to ``F^σ_{ai}``), so the internal :func:`run_ccm_uhf` runs at
    ``conv_tol=1e-12``. Default ``method="aiccm2026dev-a"``; dense ``n_pad_ao⁴``
    four-center ⇒ small / 1-D / 2-D clusters only. Returns ``(n_basis_atoms, 3)``.

    .. warning::

       **Fully-occupied-spin edge case.** When a spin manifold is *completely*
       filled (``n_occ_σ == nbf``, e.g. HeH/STO-3G with ``n_α = nbf = 2``), that
       spin density is ``P^σ = S⁻¹`` exactly, so the DIIS commutator
       ``F^σ P^σ S − S P^σ F^σ ≡ 0`` *identically* — the SCF convergence check is
       blind to that spin block and ``run_ccm_uhf`` can stop before ``F^σ`` (hence
       the canonical ``ε^σ`` feeding ``W^σ``) is fully self-consistent, so the Pulay
       term — and the force — can be off by ~1e-5. This is a minimal-basis SCF
       pathology, not a gradient defect: any basis with ``n_occ_σ < nbf`` (the
       generic case) is exact (validated to machine precision vs the molecular UHF
       analytic gradient on LiH⁺/STO-3G and HeH/6-31G). Avoid fully-filled-spin
       minimal-basis clusters, or converge ``F^σ`` independently and pass ``scf=``.
    """
    if method != "aiccm2026dev-a":
        raise NotImplementedError(
            f"run_ccm_uhf_gradient supports method='aiccm2026dev-a' only, got {method!r}."
        )
    if scf is None:
        from .uhf import run_ccm_uhf

        scf = run_ccm_uhf(ccm, method=method, conv_tol=1e-12, max_iter=512)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_uhf_gradient: the internal UHF SCF did not converge to "
                "conv_tol=1e-12; the analytic gradient needs a stationary SCF. "
                "Converge it yourself and pass scf=."
            )

    Pa = np.asarray(scf.density_alpha, dtype=float)       # P^σ = C^σ_occ C^σ_occᵀ (no 2)
    Pb = np.asarray(scf.density_beta, dtype=float)
    P = Pa + Pb                                           # total density
    Wa = _energy_weighted_density(
        scf.mo_energies_alpha, scf.mo_coeffs_alpha, scf.n_alpha, 1.0
    )
    Wb = _energy_weighted_density(
        scf.mo_energies_beta, scf.mo_coeffs_beta, scf.n_beta, 1.0
    )
    W = Wa + Wb
    # UHF AO 2-RDM: total-density Coulomb, same-spin exchange (§1.7; cf. uhf.py:151).
    gamma = (
        np.einsum("mn,ls->mnls", P, P)
        - np.einsum("ml,ns->mnls", Pa, Pa)
        - np.einsum("ml,ns->mnls", Pb, Pb)
    )
    return (
        _kinetic_gradient_P(ccm, P)
        + _nuclear_attraction_gradient_P(ccm, P)
        + _two_electron_gradient_gamma(ccm, gamma)
        + _overlap_pulay_gradient_W(ccm, W)
        + ccm_nuclear_repulsion_gradient(ccm)
    )


def run_ccm_rks_gradient(ccm, functional: str = "pbe", *, method: str = "aiccm2026dev-a",
                         scf=None, grid_options=None) -> np.ndarray:
    """Analytic Γ-CCM **RKS** (closed-shell DFT) nuclear gradient per unit-cell atom (a.u.).

    The KS counterpart of :func:`run_ccm_rhf_gradient` (§1.7). In the KS-CCM Fock
    ``F = h^CCM + J^CCM + V_xc[ρ] (− α K^CCM)`` (:mod:`vibeqc.periodic.ccm.dft`) only
    the Coulomb / exact exchange (the four-center) carries the WSSC fold; **the XC is
    the ordinary molecular semi-local functional of the cluster (supercell) density
    on a molecular Becke grid — it does NOT fold.** So the gradient reuses the HF term
    cores plus a molecular XC gradient on the supercell::

        dE_KS/dR =  Σ P ∂h^CCM/∂R                    (kinetic L₂ᵀ + V_ne L_Vᵀ)
                 +  ½ Σ Γ_KS ∂(μν|λσ)^CCM/∂R         (L₄ᵀ, Γ_KS = P⊗P − (α/2) P⊗P_swap)
                 −  Σ W ∂S^CCM/∂R                    (overlap Pulay L₂ᵀ)
                 +  ∂E_xc/∂R |_supercell             (molecular XC-Pulay, no fold)
                 +  ∂V_nn^CCM/∂R ,

    with ``α`` the functional's HF-exchange fraction (``α=0`` pure ⇒ Coulomb-only
    ``Γ_KS=P⊗P``; ``α=1`` recovers RHF). The XC term is
    ``xc_pulay_gradient_rks`` on ``ccm.supercell`` (the *same* supercell + grid the
    KS-CCM energy uses), image-summed by ``fold_supercell_gradient_to_cell`` (no
    padded images — the XC lives on the supercell, not the padded cluster).

    Like the HF drivers the analytic force needs a stationary SCF, so the internal
    :func:`~vibeqc.periodic.ccm.dft.run_ccm_rks` runs at ``conv_tol=1e-11``. Matches
    the molecular ``compute_gradient_rks`` in the isolated limit (both grid-fixed;
    the molecular DFT gradient neglects the Becke weight derivative, so analytic vs
    FD agrees only to that grid-fixed tolerance — tighten the grid to close it).
    Dense ``n_pad_ao⁴`` four-center ⇒ small / 1-D / 2-D clusters only. Returns
    ``(n_basis_atoms, 3)``.
    """
    from vibeqc import Functional, GridOptions
    from vibeqc._vibeqc_core import xc_pulay_gradient_rks

    if method != "aiccm2026dev-a":
        raise NotImplementedError(
            f"run_ccm_rks_gradient supports method='aiccm2026dev-a' only, got {method!r}."
        )
    if scf is None:
        from .dft import run_ccm_rks

        scf = run_ccm_rks(ccm, functional, method=method, conv_tol=1e-11,
                          max_iter=256, grid_options=grid_options)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_rks_gradient: the internal RKS SCF did not converge to "
                "conv_tol=1e-11; the analytic gradient needs a stationary SCF."
            )

    P = np.asarray(scf.density, dtype=float)             # total density 2 C_occ C_occᵀ
    alpha = float(Functional(functional, 1).hf_exchange_fraction)
    # KS AO 2-RDM (chemist order): total-density Coulomb + α exact exchange. The
    # ½ inside _two_electron_gradient_gamma gives E_2e = ½ Σ Γ_KS (μν|λσ); α=1 ⇒ RHF.
    gamma = (
        np.einsum("mn,ls->mnls", P, P)
        - 0.5 * alpha * np.einsum("ml,ns->mnls", P, P)
    )
    n_occ = int(round(ccm.supercell.n_electrons() / 2))
    W = _energy_weighted_density(scf.mo_energies, scf.mo_coeffs, n_occ, 2.0)

    # XC gradient: molecular XC-Pulay on the supercell density/grid (no fold), the
    # same grid run_ccm_rks built via build_grid(ccm.supercell, grid_options).
    grid = grid_options if grid_options is not None else GridOptions()
    xc_grad_sc = np.asarray(
        xc_pulay_gradient_rks(ccm.supercell, ccm.basis, P, functional, grid),
        dtype=float,
    )

    return (
        _kinetic_gradient_P(ccm, P)
        + _nuclear_attraction_gradient_P(ccm, P)
        + _two_electron_gradient_gamma(ccm, gamma)
        + _overlap_pulay_gradient_W(ccm, W)
        + fold_supercell_gradient_to_cell(ccm, xc_grad_sc)
        + ccm_nuclear_repulsion_gradient(ccm)
    )


def run_ccm_uks_gradient(ccm, functional: str = "pbe", *, method: str = "aiccm2026dev-a",
                         scf=None, grid_options=None) -> np.ndarray:
    """Analytic Γ-CCM **UKS** (open-shell DFT) nuclear gradient per unit-cell atom (a.u.).

    The open-shell counterpart of :func:`run_ccm_rks_gradient` (§1.7): it is to
    :func:`run_ccm_rks_gradient` what :func:`run_ccm_uhf_gradient` is to
    :func:`run_ccm_rhf_gradient`. The WSSC fold, the ``Lᵀ`` adjoints, the cyclic
    image-sum, and the "XC does not fold" fact are all unchanged; only the densities
    fed to the shared cores become spin-resolved:

    * 1-e ``Σ P ∂h^CCM/∂R``  — total density ``P = P^α + P^β``;
    * overlap Pulay          — ``W = W^α + W^β`` (each ``W^σ = Σ_i ε^σ_i C^σ_i C^σ_iᵀ``,
      occupation 1 per spin orbital);
    * 2-e ``½ Σ Γ ∂(μν|λσ)^CCM/∂R`` — the UKS AO 2-RDM
      ``Γ = P⊗P − α(P^α⊗P^α + P^β⊗P^β)`` (total-density Coulomb + ``α``-scaled
      same-spin exchange; ``α=1`` recovers UHF, ``α=0`` is Coulomb-only);
    * XC — the *molecular* UKS XC-Pulay ``xc_pulay_gradient_uks`` on the supercell
      spin densities (no fold), image-summed by ``fold_supercell_gradient_to_cell``;
    * ``∂V_nn^CCM/∂R`` — geometry only.

    Needs a stationary SCF (``run_ccm_uks`` at ``conv_tol=1e-11``) and a ``scf`` that
    carries the per-spin densities (``run_ccm_uks`` populates ``density_alpha/beta``,
    ``mo_*_beta``). Machine-precision vs the molecular ``compute_gradient_uks`` in the
    isolated limit; periodic FD at the grid-fixed floor. Same fully-occupied-spin
    edge caveat as :func:`run_ccm_uhf_gradient`. Dense ``n_pad_ao⁴`` ⇒ small / 1-D /
    2-D only. Returns ``(n_basis_atoms, 3)``.
    """
    from vibeqc import Functional, GridOptions
    from vibeqc._vibeqc_core import xc_pulay_gradient_uks

    if method != "aiccm2026dev-a":
        raise NotImplementedError(
            f"run_ccm_uks_gradient supports method='aiccm2026dev-a' only, got {method!r}."
        )
    if scf is None:
        from .dft import run_ccm_uks

        scf = run_ccm_uks(ccm, functional, method=method, conv_tol=1e-11,
                          max_iter=256, grid_options=grid_options)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_uks_gradient: the internal UKS SCF did not converge to "
                "conv_tol=1e-11; the analytic gradient needs a stationary SCF."
            )
    if scf.density_alpha is None or scf.density_beta is None:
        raise ValueError(
            "run_ccm_uks_gradient needs a UKS result carrying per-spin densities "
            "(scf.density_alpha / density_beta); pass a run_ccm_uks result."
        )

    Pa = np.asarray(scf.density_alpha, dtype=float)       # P^σ = C^σ_occ C^σ_occᵀ (no 2)
    Pb = np.asarray(scf.density_beta, dtype=float)
    P = Pa + Pb
    alpha = float(Functional(functional, 1).hf_exchange_fraction)
    # UKS AO 2-RDM: total-density Coulomb + α same-spin exchange (α=1 ≡ UHF).
    gamma = (
        np.einsum("mn,ls->mnls", P, P)
        - alpha * np.einsum("ml,ns->mnls", Pa, Pa)
        - alpha * np.einsum("ml,ns->mnls", Pb, Pb)
    )
    # Per-spin energy-weighted densities (occupation 1). n_alpha/n_beta from the
    # supercell charge + multiplicity (as run_ccm_uks derives them).
    n_elec = ccm.supercell.n_electrons()
    n_unpaired = int(ccm.supercell.multiplicity) - 1
    n_alpha = (n_elec + n_unpaired) // 2
    n_beta = n_elec - n_alpha
    Wa = _energy_weighted_density(scf.mo_energies, scf.mo_coeffs, n_alpha, 1.0)
    Wb = _energy_weighted_density(scf.mo_energies_beta, scf.mo_coeffs_beta, n_beta, 1.0)
    W = Wa + Wb

    grid = grid_options if grid_options is not None else GridOptions()
    xc_grad_sc = np.asarray(
        xc_pulay_gradient_uks(ccm.supercell, ccm.basis, Pa, Pb, functional, grid),
        dtype=float,
    )

    return (
        _kinetic_gradient_P(ccm, P)
        + _nuclear_attraction_gradient_P(ccm, P)
        + _two_electron_gradient_gamma(ccm, gamma)
        + _overlap_pulay_gradient_W(ccm, W)
        + fold_supercell_gradient_to_cell(ccm, xc_grad_sc)
        + ccm_nuclear_repulsion_gradient(ccm)
    )


def _g_ccm(eri, P):
    """Folded closed-shell Fock response ``G^CCM[P] = J^CCM[P] − ½ K^CCM[P]``.

    ``G[P]_μν = Σ_λσ P_λσ [(μν|λσ)^CCM − ½(μλ|νσ)^CCM]`` on the home basis — the
    two-electron part of the CCM Fock matrix for a (not necessarily idempotent)
    symmetric density ``P``. Used for the Z-vector RHS (§1.8d) and the
    energy-weighted-density oo block (§1.8h).
    """
    J = np.einsum("mnls,ls->mn", eri, P, optimize=True)
    K = np.einsum("mlns,ls->mn", eri, P, optimize=True)
    return J - 0.5 * K


def _mp2_relaxed_densities(ccm, C, eps, n_occ, eri):
    """MP2 relaxed densities on the CCM reference (§1.8; closed shell, canonical).

    Derivation: ``docs/manuscripts/aiccm_a_forces.md`` §1.8 — the MP2 Lagrangian
    with Hylleraas-stationary amplitudes and the Z-vector (CCM-CPHF) elimination of
    the orbital response (Pople, Krishnan, Schlegel & Binkley, Int. J. Quantum
    Chem. Symp. 13, 225 (1979); Handy & Schaefer, J. Chem. Phys. 81, 5031 (1984),
    doi:10.1063/1.447489; Helgaker, Jørgensen & Olsen, Ch. 12). Every ingredient —
    MO integrals ``g``, orbital energies, the CPHF supermatrix ``A`` — is built from
    the *folded* tensors, so the molecular equations apply verbatim (§1.2).

    Returns ``(P_delta, W_delta, gamma_ns)`` in the home AO basis:

    * ``P_delta`` — the relaxed one-particle correction (Eq. 1.8f),
      ``C [[d_oo, −½zᵀ],[−½z, d_vv]] Cᵀ``;
    * ``W_delta`` — the correlation energy-weighted density (Eq. 1.8h), to be
      *added* to ``W_HF`` in the Pulay term;
    * ``gamma_ns`` — the non-separable amplitude 2-PDM (Eq. 1.8g),
      ``4 Σ t̃_ijab C_μi C_νa C_λj C_σb`` (chemist order).

    Validated by the response identity ``dE_MP2/dλ|_{h+λV} = Tr[(P_SCF+P_delta) V]``
    to 2.8e-11 on the H₂ chain (§1.8 validation 1). Dense ``nbf⁴`` MO transform —
    the small/1-D/2-D validation regime shared by the whole dense gradient path.
    """
    C = np.asarray(C, dtype=float)
    eps = np.asarray(eps, dtype=float)
    nbf = C.shape[1]
    n_vir = nbf - n_occ
    o, v = slice(0, n_occ), slice(n_occ, nbf)
    Co, Cv = C[:, o], C[:, v]
    eo, ev = eps[o], eps[v]

    # Full MO four-center g_pqrs = (pq|rs)^CCM (chemist order); nbf⁴ dense.
    g = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C, optimize=True)

    # Eq. 1.8a: canonical amplitudes t_ijab = (ia|jb)/D, t̃ = 2t − t.swap(ab).
    D = (eo[:, None, None, None] + eo[None, :, None, None]
         - ev[None, None, :, None] - ev[None, None, None, :])
    t = g[o, v, o, v].transpose(0, 2, 1, 3) / D                  # t[i,j,a,b]
    tt = 2.0 * t - t.transpose(0, 1, 3, 2)                       # t̃[i,j,a,b]

    # Eq. 1.8c: the unrelaxed correction density d (∂E₂/∂ε_p = d_pp) and the
    # 3-index back-transforms X_pi = Σ t̃ (pa|jb), Y_pa = Σ t̃ (ip|jb).
    d_oo = -2.0 * np.einsum("ikab,jkab->ij", tt, t, optimize=True)
    d_vv = +2.0 * np.einsum("ijac,ijbc->ab", tt, t, optimize=True)
    X = np.einsum("ijab,pajb->pi", tt, g[:, v, o, v], optimize=True)
    Y = np.einsum("ijab,ipjb->pa", tt, g[o, :, o, v], optimize=True)

    # Eq. 1.8d: Z-vector RHS L_ai = 4X_ai − 4Y_ia + 4(Cᵀ G^CCM[C d Cᵀ] C)_ai.
    d_ur = np.zeros((nbf, nbf))
    d_ur[o, o] = d_oo
    d_ur[v, v] = d_vv
    G_ur_mo = C.T @ _g_ccm(eri, C @ d_ur @ C.T) @ C
    L = 4.0 * X[v, :] - 4.0 * Y[o, :].T + 4.0 * G_ur_mo[v, o]    # (n_vir, n_occ)

    # Eq. 1.8e: the CCM-CPHF (Z-vector) solve M z = L with the folded electronic
    # Hessian A_ai,bj = 4(ai|bj) − (ab|ij) − (aj|ib) (Handy-Schaefer 1984).
    A = (4.0 * g[v, o, v, o]                                     # [a,i,b,j]
         - g[v, v, o, o].transpose(0, 2, 1, 3)                   # (ab|ij)
         - g[v, o, o, v].transpose(0, 2, 3, 1))                  # (aj|ib)
    M = A + np.einsum("ai,ab,ij->aibj", ev[:, None] - eo[None, :],
                      np.eye(n_vir), np.eye(n_occ), optimize=True)
    z = np.linalg.solve(
        M.reshape(n_vir * n_occ, n_vir * n_occ), L.reshape(-1)
    ).reshape(n_vir, n_occ)

    # Eq. 1.8f: relaxed one-particle correction P^Δ (ov block −½z).
    d_full = d_ur.copy()
    d_full[o, v] = -0.5 * z.T
    d_full[v, o] = -0.5 * z
    P_delta = C @ d_full @ C.T

    # Eq. 1.8h: the correlation energy-weighted density W^Δ (MO blocks, symmetric;
    # the oo block carries the Fock response of the *full* P^Δ).
    W = np.zeros((nbf, nbf))
    W[o, o] = (X[o, :] + X[o, :].T
               + 0.5 * d_oo * (eo[:, None] + eo[None, :])
               + 2.0 * (C.T @ _g_ccm(eri, P_delta) @ C)[o, o])
    W[v, v] = Y[v, :] + Y[v, :].T + 0.5 * d_vv * (ev[:, None] + ev[None, :])
    W_ov = 2.0 * Y[o, :] - 0.5 * (z * eo[None, :]).T             # [i,a]
    W[o, v] = W_ov
    W[v, o] = W_ov.T
    W_delta = C @ W @ C.T

    # Eq. 1.8g: non-separable amplitude 2-PDM, Γ_ns = 4 t̃ back-transformed (the ½
    # inside _two_electron_gradient_gamma makes ½ Σ Γ_ns ∂g = 2 Σ t̃ ∂(ia|jb)).
    gamma_ns = 4.0 * np.einsum("ijab,mi,na,lj,sb->mnls", tt, Co, Cv, Co, Cv,
                               optimize=True)
    return P_delta, W_delta, gamma_ns


def run_ccm_mp2_gradient(ccm, *, method: str = "aiccm2026dev-a", scf=None) -> np.ndarray:
    """Analytic Γ-CCM **MP2** nuclear gradient ``dE_MP2/dR`` per unit-cell atom (a.u.).

    The closed-shell MP2 *relaxed-density* gradient (§1.8 of
    ``docs/manuscripts/aiccm_a_forces.md``): the HF contraction pipeline (1.1)
    verbatim, with the relaxed densities of Eqs. 1.8f–1.8h::

        dE_MP2/dR =  Σ (P_SCF+P^Δ) ∂h^CCM/∂R          (kinetic L₂ᵀ + V_ne L_Vᵀ)
                  +  ½ Σ (Γ_HF+Γ_sep+Γ_ns) ∂(μν|λσ)^CCM/∂R      (four-center L₄ᵀ)
                  −  Σ (W_HF+W^Δ) ∂S^CCM/∂R                  (overlap Pulay L₂ᵀ)
                  +  ∂V_nn^CCM/∂R ,

    where ``P^Δ`` carries the amplitude density and the Z-vector (CCM-CPHF)
    orbital response, ``Γ_sep = 2P^Δ⊗P − P^Δ⊗P_swap`` is the separable
    (Fock-response) 2-PDM, and ``Γ_ns = 4t̃`` the amplitude 2-PDM. The amplitudes,
    the CPHF supermatrix, and every contraction are built from the *folded*
    integrals — the WSSC fold commutes with the entire relaxed-density machinery
    (§1.2/§1.8). References: Pople-Krishnan-Schlegel-Binkley 1979 (MP2 gradient);
    Handy-Schaefer 1984 (Z-vector); Helgaker-Jørgensen-Olsen Ch. 12.

    Consistency contract: the reference SCF and the correlation must ride the
    *same* folded four-center. A hand-passed ``scf`` must be a
    :func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf` result converged with this
    ``method``'s ERI (tightly — ``conv_tol≲1e-12``; the relaxed gradient is
    first-order sensitive to SCF non-stationarity like the HF one). All-electron
    (no frozen core), closed shell; dense ``nbf⁴``/``n_pad_ao⁴`` ⇒ small / 1-D /
    2-D clusters only. Returns ``(n_basis_atoms, 3)``; the per-cell forces sum to
    ≈ 0 (Theorem 1.6.1 applies to E₂ as well).

    Validation (§1.8): FD gate at the O(h²) floor (``4.3e-6 → 1.1e-6 → 2.7e-7``
    for ``h = 2e-3, 1e-3, 5e-4`` on the (3,1,1) H₂ chain); response identity to
    2.8e-11; isolated-limit vs vibe-qc's independent *molecular* ``run_mp2`` FD to
    3.6e-9; sum rule to 6.7e-16.

    .. warning::

       **FD comparisons need a unique SCF solution.** Where the CCM-RHF ground
       state is symmetry-broken with near-degenerate branch partners (e.g. the
       (2,1,1) H₂ chain, branch splitting ~8e-12 Ha), displaced-geometry SCFs
       branch-hop; ``E_HF`` is branch-insensitive but ``E₂`` differs by ~1e-5
       between branches, so the *FD* MP2 gradient is erratic there while the
       analytic gradient stays well-defined on the reference branch (§1.8 caveat).
    """
    from .scf import _ccm_eri_for_method, run_ccm_rhf

    eri = _ccm_eri_for_method(ccm, method)
    if scf is None:
        scf = run_ccm_rhf(ccm, eri=eri, conv_tol=1e-12, max_iter=512)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_mp2_gradient: the internal SCF did not converge to "
                "conv_tol=1e-12; the analytic gradient needs a stationary SCF. "
                "Converge it yourself and pass scf=."
            )

    n_elec = int(ccm.supercell.n_electrons())
    if n_elec % 2 != 0 or int(ccm.supercell.multiplicity) != 1:
        raise ValueError(
            "run_ccm_mp2_gradient is closed-shell (RMP2); open-shell UMP2 "
            "gradients are follow-on work."
        )
    n_occ = n_elec // 2
    P = np.asarray(scf.density, dtype=float)
    W_hf = _energy_weighted_density(scf.mo_energies, scf.mo_coeffs, n_occ, 2.0)
    gamma_hf = (np.einsum("mn,ls->mnls", P, P)
                - 0.5 * np.einsum("ml,ns->mnls", P, P))

    nbf = P.shape[0]
    if n_occ == 0 or n_occ == nbf:
        # No occupied or no virtual space -> zero correlation -> the RHF gradient.
        P_delta = np.zeros_like(P)
        W_delta = np.zeros_like(P)
        gamma_ns = np.zeros((nbf,) * 4)
    else:
        P_delta, W_delta, gamma_ns = _mp2_relaxed_densities(
            ccm, scf.mo_coeffs, scf.mo_energies, n_occ, eri
        )

    # Γ_sep = 2P^Δ⊗P − P^Δ⊗P_swap (Eq. 1.8g): the separable (Fock-response) 2-PDM
    # pairing the relaxed correction with the SCF density.
    gamma_sep = (2.0 * np.einsum("mn,ls->mnls", P_delta, P)
                 - np.einsum("ml,ns->mnls", P_delta, P))

    return (
        _kinetic_gradient_P(ccm, P + P_delta)
        + _nuclear_attraction_gradient_P(ccm, P + P_delta)
        + _two_electron_gradient_gamma(ccm, gamma_hf + gamma_sep + gamma_ns)
        + _overlap_pulay_gradient_W(ccm, W_hf + W_delta)
        + ccm_nuclear_repulsion_gradient(ccm)
    )


def _g_ccm_uhf(eri, Xa, Xb, spin):
    """Spin-resolved folded Fock response ``G^σ[X^α,X^β] = J[X^α+X^β] − K[X^σ]``.

    The UHF counterpart of :func:`_g_ccm` (occupation-1 spin densities): total-
    density Coulomb, same-spin exchange. ``spin`` is 0 (α) or 1 (β). Used for the
    spin-coupled Z-vector RHS and the UMP2 ``W^Δσ`` oo block (§1.8k/§1.8.4).
    """
    J = np.einsum("mnls,ls->mn", eri, Xa + Xb, optimize=True)
    K = np.einsum("mlns,ls->mn", eri, Xa if spin == 0 else Xb, optimize=True)
    return J - K


def _ump2_relaxed_densities(ccm, Ca, Cb, ea, eb, n_occ_a, n_occ_b, eri):
    """UMP2 relaxed densities on the CCM-UHF reference (§1.8.4; canonical spins).

    The open-shell mirror of :func:`_mp2_relaxed_densities`: per-spin amplitude
    blocks ``t^αα``/``t^ββ`` (antisymmetrized) and ``t^αβ`` (Eq. 1.8j), per-spin
    intermediates ``d^σ``/``X^σ``/``Y^σ`` (spin factors absorbed: same-spin 1,
    cross-spin 2; Eq. 1.8k), the **spin-coupled** Z-vector solve on the stacked ov
    space (Eq. 1.8l), and the per-spin assembly mirroring Eqs. 1.8f–1.8h with
    occupation 1. Every block's closed-shell reduction reproduces the RMP2 set
    (``t^αβ = t``, ``X^σ = 2X_cs``, ``z^σ = ½z_cs``, …) — pinned numerically to
    machine ε by the closed-shell-reduction test.

    Returns ``(P_delta_a, P_delta_b, W_delta, gamma_ns)`` in the home AO basis
    (``W_delta`` already spin-summed; ``gamma_ns`` per Eq. 1.8k's Γ_ns line).
    References: Pople-Krishnan-Schlegel-Binkley 1979; Handy-Schaefer 1984
    (Z-vector); Helgaker-Jørgensen-Olsen Ch. 12.
    """
    Ca = np.asarray(Ca, dtype=float)
    Cb = np.asarray(Cb, dtype=float)
    ea = np.asarray(ea, dtype=float)
    eb = np.asarray(eb, dtype=float)
    nbf = Ca.shape[1]
    nva, nvb = nbf - n_occ_a, nbf - n_occ_b
    oa, va = slice(0, n_occ_a), slice(n_occ_a, nbf)
    ob, vb = slice(0, n_occ_b), slice(n_occ_b, nbf)
    eoa, eva = ea[oa], ea[va]
    eob, evb = eb[ob], eb[vb]

    def mo4(C1, C2, C3, C4):
        return np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C1, C2, C3, C4,
                         optimize=True)

    g_aa, g_bb, g_ab = mo4(Ca, Ca, Ca, Ca), mo4(Cb, Cb, Cb, Cb), mo4(Ca, Ca, Cb, Cb)

    # Eq. 1.8j: same-spin antisymmetrized + opposite-spin amplitudes.
    def same_spin_t(g, eo, ev, o, v):
        g4 = g[o, v, o, v].transpose(0, 2, 1, 3)              # [i,j,a,b] = (ia|jb)
        gbar = g4 - g4.transpose(0, 1, 3, 2)                  # (ia|jb) − (ib|ja)
        D = (eo[:, None, None, None] + eo[None, :, None, None]
             - ev[None, None, :, None] - ev[None, None, None, :])
        return gbar / D

    t_aa = same_spin_t(g_aa, eoa, eva, oa, va)
    t_bb = same_spin_t(g_bb, eob, evb, ob, vb)
    D_ab = (eoa[:, None, None, None] + eob[None, :, None, None]
            - eva[None, None, :, None] - evb[None, None, None, :])
    t_ab = g_ab[oa, va, ob, vb].transpose(0, 2, 1, 3) / D_ab

    # Eq. 1.8k: d^σ (unrelaxed, per spin; ∂E₂/∂ε^σ_p = d^σ_pp).
    d_oo_a = (-0.5 * np.einsum("ikab,jkab->ij", t_aa, t_aa, optimize=True)
              - np.einsum("ikab,jkab->ij", t_ab, t_ab, optimize=True))
    d_vv_a = (+0.5 * np.einsum("ijac,ijbc->ab", t_aa, t_aa, optimize=True)
              + np.einsum("ijac,ijbc->ab", t_ab, t_ab, optimize=True))
    d_oo_b = (-0.5 * np.einsum("ikab,jkab->ij", t_bb, t_bb, optimize=True)
              - np.einsum("kiab,kjab->ij", t_ab, t_ab, optimize=True))
    d_vv_b = (+0.5 * np.einsum("ijac,ijbc->ab", t_bb, t_bb, optimize=True)
              + np.einsum("ijca,ijcb->ab", t_ab, t_ab, optimize=True))

    # Eq. 1.8k: X^σ_pi / Y^σ_pa (same-spin coeff 1 with ḡ; cross-spin coeff 2).
    gbar_pv_aa = g_aa[:, va, oa, va] - g_aa[:, va, oa, va].transpose(0, 3, 2, 1)
    gbar_pv_bb = g_bb[:, vb, ob, vb] - g_bb[:, vb, ob, vb].transpose(0, 3, 2, 1)
    X_a = (np.einsum("ijab,pajb->pi", t_aa, gbar_pv_aa, optimize=True)
           + 2.0 * np.einsum("ijab,pajb->pi", t_ab, g_ab[:, va, ob, vb],
                             optimize=True))
    X_b = (np.einsum("ijab,pajb->pi", t_bb, gbar_pv_bb, optimize=True)
           + 2.0 * np.einsum("ijab,iapb->pj", t_ab, g_ab[oa, va, :, vb],
                             optimize=True))
    gbar_ov_aa = g_aa[oa, :, oa, va] - g_aa[oa, va, oa, :].transpose(0, 3, 2, 1)
    gbar_ov_bb = g_bb[ob, :, ob, vb] - g_bb[ob, vb, ob, :].transpose(0, 3, 2, 1)
    Y_a = (np.einsum("ijab,ipjb->pa", t_aa, gbar_ov_aa, optimize=True)
           + 2.0 * np.einsum("ijab,ipjb->pa", t_ab, g_ab[oa, :, ob, vb],
                             optimize=True))
    Y_b = (np.einsum("ijab,ipjb->pa", t_bb, gbar_ov_bb, optimize=True)
           + 2.0 * np.einsum("ijab,iajp->pb", t_ab, g_ab[oa, va, ob, :],
                             optimize=True))

    # Eq. 1.8k: Z-vector RHS L^σ = X^σ_ai − Y^σ_ia + 2 G^σ[D_ur]_ai.
    dur_a = np.zeros((nbf, nbf))
    dur_a[oa, oa] = d_oo_a
    dur_a[va, va] = d_vv_a
    dur_b = np.zeros((nbf, nbf))
    dur_b[ob, ob] = d_oo_b
    dur_b[vb, vb] = d_vv_b
    Dur_a, Dur_b = Ca @ dur_a @ Ca.T, Cb @ dur_b @ Cb.T
    L_a = (X_a[va, :] - Y_a[oa, :].T
           + 2.0 * (Ca.T @ _g_ccm_uhf(eri, Dur_a, Dur_b, 0) @ Ca)[va, oa])
    L_b = (X_b[vb, :] - Y_b[ob, :].T
           + 2.0 * (Cb.T @ _g_ccm_uhf(eri, Dur_a, Dur_b, 1) @ Cb)[vb, ob])

    # Eq. 1.8l: spin-coupled CCM-CPHF: M^σσ = Δε + 2(ai|bj)−(ab|ij)−(aj|ib),
    # M^σσ̄ = 2(ai|bj)^σσ̄, solved on the stacked ov space (Handy-Schaefer 1984).
    def m_same(g, eo, ev, o, v):
        A = (2.0 * g[v, o, v, o]
             - g[v, v, o, o].transpose(0, 2, 1, 3)
             - g[v, o, o, v].transpose(0, 2, 3, 1))
        return A + np.einsum("ai,ab,ij->aibj", ev[:, None] - eo[None, :],
                             np.eye(len(ev)), np.eye(len(eo)), optimize=True)

    M_aa = m_same(g_aa, eoa, eva, oa, va).reshape(nva * n_occ_a, nva * n_occ_a)
    M_bb = m_same(g_bb, eob, evb, ob, vb).reshape(nvb * n_occ_b, nvb * n_occ_b)
    M_ab = 2.0 * g_ab[va, oa, vb, ob].reshape(nva * n_occ_a, nvb * n_occ_b)
    zvec = np.linalg.solve(np.block([[M_aa, M_ab], [M_ab.T, M_bb]]),
                           np.concatenate([L_a.reshape(-1), L_b.reshape(-1)]))
    z_a = zvec[:nva * n_occ_a].reshape(nva, n_occ_a)
    z_b = zvec[nva * n_occ_a:].reshape(nvb, n_occ_b)

    # Per-spin relaxed corrections (Eq. 1.8f mirror, ov block −½z^σ).
    df_a = dur_a.copy()
    df_a[oa, va] = -0.5 * z_a.T
    df_a[va, oa] = -0.5 * z_a
    df_b = dur_b.copy()
    df_b[ob, vb] = -0.5 * z_b.T
    df_b[vb, ob] = -0.5 * z_b
    PD_a, PD_b = Ca @ df_a @ Ca.T, Cb @ df_b @ Cb.T

    # Per-spin W^Δσ (Eq. 1.8h mirror, occupation 1; oo block carries the spin-
    # resolved Fock response of the full P^Δ pair).
    def w_spin(X, Y, d_oo, d_vv, z, eo, ev, o, v, C, G_full_mo):
        W = np.zeros((nbf, nbf))
        W[o, o] = (0.25 * (X[o, :] + X[o, :].T)
                   + 0.5 * d_oo * (eo[:, None] + eo[None, :])
                   + G_full_mo[o, o])
        W[v, v] = (0.25 * (Y[v, :] + Y[v, :].T)
                   + 0.5 * d_vv * (ev[:, None] + ev[None, :]))
        W_ov = 0.5 * Y[o, :] - 0.5 * (z * eo[None, :]).T
        W[o, v] = W_ov
        W[v, o] = W_ov.T
        return C @ W @ C.T

    Gf_a = Ca.T @ _g_ccm_uhf(eri, PD_a, PD_b, 0) @ Ca
    Gf_b = Cb.T @ _g_ccm_uhf(eri, PD_a, PD_b, 1) @ Cb
    W_delta = (w_spin(X_a, Y_a, d_oo_a, d_vv_a, z_a, eoa, eva, oa, va, Ca, Gf_a)
               + w_spin(X_b, Y_b, d_oo_b, d_vv_b, z_b, eob, evb, ob, vb, Cb, Gf_b))

    # Γ_ns (Eq. 1.8k): 2 t^σσ⊗(C^σ)⁴ + 4 t^αβ⊗(C^α)²(C^β)² (closed-shell
    # reduction: 2·2(t−t_sw) + 4t = 4t̃ = the RMP2 Γ_ns).
    Coa, Cva, Cob, Cvb = Ca[:, oa], Ca[:, va], Cb[:, ob], Cb[:, vb]
    gamma_ns = (2.0 * np.einsum("ijab,mi,na,lj,sb->mnls", t_aa, Coa, Cva, Coa, Cva,
                                optimize=True)
                + 2.0 * np.einsum("ijab,mi,na,lj,sb->mnls", t_bb, Cob, Cvb, Cob, Cvb,
                                  optimize=True)
                + 4.0 * np.einsum("ijab,mi,na,lj,sb->mnls", t_ab, Coa, Cva, Cob, Cvb,
                                  optimize=True))
    return PD_a, PD_b, W_delta, gamma_ns


def run_ccm_ump2_gradient(ccm, *, method: str = "aiccm2026dev-a", scf=None) -> np.ndarray:
    """Analytic Γ-CCM **UMP2** nuclear gradient ``dE_UMP2/dR`` per unit-cell atom (a.u.).

    The open-shell mirror of :func:`run_ccm_mp2_gradient` (§1.8.4 of
    ``docs/manuscripts/aiccm_a_forces.md``): per-spin amplitude blocks
    ``t^αα``/``t^ββ``/``t^αβ``, the **spin-coupled** Z-vector (CCM-CPHF) on the
    stacked ov space, and per-spin relaxed densities fed to the same spin-general
    contraction cores::

        dE_UMP2/dR =  Σ (P^α+P^β+P^Δα+P^Δβ) ∂h^CCM/∂R
                   +  ½ Σ (Γ_HF^UHF + Γ_sep + Γ_ns) ∂(μν|λσ)^CCM/∂R
                   −  Σ (W^α+W^β+W^Δα+W^Δβ) ∂S^CCM/∂R
                   +  ∂V_nn^CCM/∂R ,

    with ``Γ_sep = 2P^Δ_tot⊗P_tot − 2P^Δα⊗P^α − 2P^Δβ⊗P^β`` (total-density
    Coulomb, same-spin exchange) and ``Γ_ns`` per Eq. 1.8k. Reduces **exactly** to
    :func:`run_ccm_mp2_gradient` for a closed shell (pinned at machine ε by the
    reduction test). Same consistency contract, tight-SCF requirement, dense
    small/1-D/2-D regime, and fully-occupied-spin caveat as the UHF/RMP2 drivers.
    Returns ``(n_basis_atoms, 3)``.

    Validation (§1.8.4): closed-shell reduction 6.2e-15; response identity
    1.7e-10; FD gate on the open-shell HeH chain (3,1,1) at the O(h²) floor
    (``4.9e-6 → 1.2e-6 → 3.1e-7`` for ``h = 2e-3, 1e-3, 5e-4``); isolated limit
    vs vibe-qc's independent molecular ``run_ump2`` FD to 1.1e-9; sum rule 1.6e-14.
    """
    from .scf import _ccm_eri_for_method
    from .uhf import run_ccm_uhf

    eri = _ccm_eri_for_method(ccm, method)
    if scf is None:
        scf = run_ccm_uhf(ccm, method=method, conv_tol=1e-12, max_iter=512)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_ump2_gradient: the internal UHF SCF did not converge to "
                "conv_tol=1e-12; the analytic gradient needs a stationary SCF. "
                "Converge it yourself and pass scf=."
            )

    Ca = np.asarray(scf.mo_coeffs_alpha, dtype=float)
    Cb = np.asarray(scf.mo_coeffs_beta, dtype=float)
    ea = np.asarray(scf.mo_energies_alpha, dtype=float)
    eb = np.asarray(scf.mo_energies_beta, dtype=float)
    noa, nob = int(scf.n_alpha), int(scf.n_beta)
    Pa = np.asarray(scf.density_alpha, dtype=float)
    Pb = np.asarray(scf.density_beta, dtype=float)
    P = Pa + Pb

    PD_a, PD_b, W_delta, gamma_ns = _ump2_relaxed_densities(
        ccm, Ca, Cb, ea, eb, noa, nob, eri
    )
    PD = PD_a + PD_b

    # UHF Γ_HF (§1.7) + the separable relaxed part (total-density Coulomb,
    # same-spin exchange; closed-shell reduction = Eq. 1.8g).
    gamma_hf = (np.einsum("mn,ls->mnls", P, P)
                - np.einsum("ml,ns->mnls", Pa, Pa)
                - np.einsum("ml,ns->mnls", Pb, Pb))
    gamma_sep = (2.0 * np.einsum("mn,ls->mnls", PD, P)
                 - 2.0 * np.einsum("ml,ns->mnls", PD_a, Pa)
                 - 2.0 * np.einsum("ml,ns->mnls", PD_b, Pb))
    Wa = _energy_weighted_density(ea, Ca, noa, 1.0)
    Wb = _energy_weighted_density(eb, Cb, nob, 1.0)

    return (
        _kinetic_gradient_P(ccm, P + PD)
        + _nuclear_attraction_gradient_P(ccm, P + PD)
        + _two_electron_gradient_gamma(ccm, gamma_hf + gamma_sep + gamma_ns)
        + _overlap_pulay_gradient_W(ccm, Wa + Wb + W_delta)
        + ccm_nuclear_repulsion_gradient(ccm)
    )


# --------------------------------------------------------------------------- #
# CCSD relaxed-density gradient (closed shell) — §1.9 of aiccm_a_forces.md
# --------------------------------------------------------------------------- #
# The CCSD analytic nuclear gradient is the relaxed-density gradient of Scheiner,
# Scuseria, Lee, Rice & Schaefer, J. Chem. Phys. 87, 5361 (1987); Gauss & Cremer,
# Chem. Phys. Lett. 138, 131 (1987); with the Λ (Handy & Schaefer, J. Chem. Phys.
# 81, 5031 (1984)) / Z-vector orbital response. As for HF/MP2, the WSSC fold
# commutes with the entire relaxed-density machinery (aiccm_a_forces.md §1.2), so
# the CCSD Λ equations, the relaxed 1-/2-particle densities, and the CCM-CPHF
# orbital response are the molecular equations on the *folded* integrals, and the
# final AO contractions reuse the SAME L₂ᵀ / L_Vᵀ / L₄ᵀ adjoint cores as HF/MP2.
#
# Implementation note — the dense validation regime. The CCSD energy itself is a
# spin-orbital n_so⁶ path restricted to small / 1-D / 2-D clusters (ccsd.py). The
# gradient here matches that regime and takes the most *robust* route to the
# relaxed densities: exact complex-step differentiation of the CCSD correlation
# Lagrangian L(t,Λ; f, W) with respect to the MO integrals. Because L is a
# polynomial in the integrals, the complex step h(x+iε)/ε is exact to machine
# precision (Squire & Trapp, SIAM Rev. 40, 110 (1998)), so the response 1-PDM
# γ = ∂L/∂f, the 2-PDM Γ = ∂L/∂W (cumulant), and the generalized Fock ∂L/∂U come
# out without any hand-derived (error-prone) CC density algebra. Λ is obtained
# from the exact complex-step transpose-Jacobian of the CCSD residual (avoiding a
# hand-coded Λ equation). This is O(n_so²)/O(n_so⁴) Lagrangian evaluations —
# affordable for the tiny validation clusters, and philosophically consistent with
# the dense CCSD energy path. (An iterative Λ solver + closed-form densities is the
# scalable-kernel follow-on.)


def _ccsd_fs_from_spatial(m: np.ndarray) -> np.ndarray:
    """Spin-orbital block-diagonal matrix from a spatial matrix ``m[p,q]`` (interleave)."""
    n = m.shape[0]
    nso = 2 * n
    idx = np.arange(nso) // 2
    spin = np.arange(nso) % 2
    same = spin[:, None] == spin[None, :]
    fs = np.zeros((nso, nso), dtype=m.dtype)
    fs[same] = m[np.ix_(idx, idx)][same]
    return fs


def _t3_asym_ijk(x):
    return x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 5, 4, 3)


def _t3_asym_abc(x):
    return x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)


def _t3_connected_numerator(spinints, td, nocc_so):
    """W^abc_ijk = P(i/jk)P(a/bc)[Σ_e t^ae_jk ⟨ei‖bc⟩ − Σ_m t^bc_im ⟨ma‖jk⟩]."""
    nso = spinints.shape[0]
    o, v = slice(0, nocc_so), slice(nocc_so, nso)
    w = np.einsum("aejk,eibc->abcijk", td, spinints[v, o, v, v]) - np.einsum(
        "bcim,majk->abcijk", td, spinints[o, v, o, o]
    )
    return _t3_asym_abc(_t3_asym_ijk(w))


def _t3_disconnected_numerator(fs, spinints, ts, td, nocc_so):
    """V^abc_ijk = P(i/jk)P(a/bc)[t^a_i ⟨jk‖bc⟩ + f_ia t^bc_jk] (Eq. 1.10c).

    The ``f_ov·T₂`` piece is the general-single-determinant fifth-order term
    (Watts, Gauss & Bartlett, J. Chem. Phys. 98, 8718 (1993)); it vanishes at a
    canonical HF reference but its f-derivative is the ov triples density.
    """
    nso = spinints.shape[0]
    o, v = slice(0, nocc_so), slice(nocc_so, nso)
    d = np.einsum("ai,jkbc->abcijk", ts, spinints[o, o, v, v]) + np.einsum(
        "ia,bcjk->abcijk", fs[o, v], td
    )
    return _t3_asym_abc(_t3_asym_ijk(d))


def _t3_A_offdiag(fs, t3, nocc_so):
    """Off-diagonal part of the triples one-body map A[f] (Eq. 1.10b)."""
    nso = fs.shape[0]
    o, v = slice(0, nocc_so), slice(nocc_so, nso)
    f_oo = fs[o, o] - np.diag(np.diag(fs[o, o]))
    f_vv = fs[v, v] - np.diag(np.diag(fs[v, v]))
    return (
        np.einsum("im,abcmjk->abcijk", f_oo, t3)
        + np.einsum("jm,abcimk->abcijk", f_oo, t3)
        + np.einsum("km,abcijm->abcijk", f_oo, t3)
        - np.einsum("ae,ebcijk->abcijk", f_vv, t3)
        - np.einsum("be,aecijk->abcijk", f_vv, t3)
        - np.einsum("ce,abeijk->abcijk", f_vv, t3)
    )


def _t3_denominator(fs, nocc_so):
    """D^abc_ijk = f_ii+f_jj+f_kk−f_aa−f_bb−f_cc (the diagonal of A[f])."""
    eps = np.diag(fs)
    eo, ev = eps[:nocc_so], eps[nocc_so:]
    return (
        eo[None, None, None, :, None, None]
        + eo[None, None, None, None, :, None]
        + eo[None, None, None, None, None, :]
        - ev[:, None, None, None, None, None]
        - ev[None, :, None, None, None, None]
        - ev[None, None, :, None, None, None]
    )


def _ccsd_t_energy_so(ts, td, fs, spinints, nocc_so, *, sweeps=2):
    """Non-canonical (T) energy E_(T) = (1/36) Σ t₃·(W+V), t₃ = A[f]⁻¹W (§1.10).

    The complex-steppable (T) functional: unlike the canonical formula (which
    sees f only through diag(f) and therefore cannot yield the off-diagonal
    triples densities — the ~2.5e-5 heteronuclear gradient error diagnosed
    2026-07-17), this is holomorphic in every element of ``f``: the connected
    t₃ solves the full first-order equation A[f]t₃ = W (Eq. 1.10b) and the
    disconnected numerator carries the f_ov·T₂ term (Eq. 1.10c). At a
    canonical reference it reduces bit-identically to ``ccsd._triples``
    (A = D one-shot, f_ov = 0); a complex-step perturbation makes the
    off-diagonal of A pure O(h), so ``sweeps=2`` preconditioned Jacobi sweeps
    are exact through O(h) (§1.10.4; gated to 1e-16 against a converged FD).
    References: Raghavachari 1989 (canonical (T)); Watts-Gauss-Bartlett 1993
    (the general-reference functional); Scuseria 1991 / WGB 1992 ((T)
    gradient densities, produced here by complex step instead of hand algebra).
    """
    W = _t3_connected_numerator(spinints, td, nocc_so)
    D = _t3_denominator(fs, nocc_so)
    t3 = W / D
    for _ in range(sweeps):
        t3 = (W - _t3_A_offdiag(fs, t3, nocc_so)) / D
    V = _t3_disconnected_numerator(fs, spinints, ts, td, nocc_so)
    return (1.0 / 36.0) * np.einsum("abcijk,abcijk->", t3, W + V)


def _ccsd_solve_amplitudes(fs, spinints, nocc_so, *, max_iter=300, tol=1e-13,
                           ndiis=6):
    """Converge (ts, td) to the CCSD residual zero with DIIS (general f).

    Plain Jacobi (`t += Ω/D`) stalls on harder references (open-shell UHF chains
    reach only ~1e-4); DIIS on the update vector `t + Ω/D` with the residual-scaled
    error `Ω/D` restores fast convergence (~40 iters to 1e-13). Reduces to the
    Jacobi step for `ndiis=1`.
    """
    from .ccsd import _ccsd_residual_so

    nso = spinints.shape[0]
    o, v = slice(0, nocc_so), slice(nocc_so, nso)
    eps = np.diag(fs)
    Dai = (eps[o, None] - eps[None, v]).T
    Dabij = (eps[o, None, None, None] + eps[None, o, None, None]
             - eps[None, None, v, None] - eps[None, None, None, v]).transpose(2, 3, 0, 1)
    ts = np.zeros((nso - nocc_so, nocc_so))
    td = spinints[v, v, o, o] / Dabij
    n1 = ts.size
    hist_t, hist_e = [], []
    for _ in range(max_iter):
        o1, o2 = _ccsd_residual_so(ts, td, fs, spinints, nocc_so)
        err = np.concatenate([(o1 / Dai).ravel(), (o2 / Dabij).ravel()])
        if np.max(np.abs(err)) < tol:
            break
        tvec = np.concatenate([ts.ravel(), td.ravel()]) + err
        hist_t.append(tvec)
        hist_e.append(err)
        if len(hist_t) > ndiis:
            hist_t.pop(0)
            hist_e.pop(0)
        m = len(hist_e)
        if m > 1:
            B = np.zeros((m + 1, m + 1))
            B[-1, :-1] = B[:-1, -1] = -1.0
            for i in range(m):
                for j in range(i, m):
                    B[i, j] = B[j, i] = hist_e[i] @ hist_e[j]
            rhs = np.zeros(m + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:-1]
                tvec = sum(ci * hi for ci, hi in zip(c, hist_t))
            except np.linalg.LinAlgError:
                pass
        ts = tvec[:n1].reshape(ts.shape)
        td = tvec[n1:].reshape(td.shape)
    return ts, td


def _ccsd_lambda_jacobian_nbytes(nso: int, nocc_so: int) -> int:
    """Bytes occupied by the dense real CCSD amplitude Jacobian."""
    nv = int(nso) - int(nocc_so)
    no = int(nocc_so)
    n_amp = nv * no + nv * nv * no * no
    return np.dtype(float).itemsize * n_amp * n_amp


def _ccsd_solve_lambda(
    ts,
    td,
    fs,
    spinints,
    nocc_so,
    *,
    eps_cs=1e-30,
    triples=False,
    solver="dense",
    tol=1e-10,
    max_iter=None,
):
    """Λ via the exact complex-step transpose-Jacobian: ``J^T λ = −η`` (§1.9).

    ``J_{μν} = ∂Ω_μ/∂t_ν`` (Ω the projected CCSD residual) is built column-by-column
    by complex step — exact because Ω is a polynomial in the amplitudes. ``η = ∂E/∂t``.
    With ``triples=True`` the RHS gains ``η_(T) = ∂E_(T)/∂t`` (the T₃→Λ₁/T₃→Λ₂
    inhomogeneity of Scuseria 1991 / WGB 1992, §1.10.3), again by complex step.
    The ``td`` antisymmetry makes ``J`` singular on the redundant subspace.
    ``solver="dense"`` materializes ``J`` and uses ``lstsq`` as the
    small-cluster oracle. ``solver="matrix-free"`` supplies exact
    complex-step ``J*x`` and ``J.T*x`` products to LSMR, retaining its
    minimum-norm least-squares contract without the O(N²) Jacobian storage.
    """
    from .ccsd import _ccsd_residual_so

    if solver not in ("dense", "matrix-free"):
        raise ValueError(
            f"unknown CCSD Lambda solver {solver!r}; expected 'dense' or "
            "'matrix-free'"
        )
    if tol <= 0.0:
        raise ValueError("CCSD Lambda tolerance must be positive")
    if max_iter is not None and max_iter <= 0:
        raise ValueError("CCSD Lambda max_iter must be positive")

    nso = spinints.shape[0]
    nv, no = nso - nocc_so, nocc_so
    n1, n2 = nv * no, nv * nv * no * no
    N = n1 + n2
    o, v = slice(0, no), slice(no, nso)

    def unpack(x):
        return x[:n1].reshape(nv, no), x[n1:].reshape(nv, nv, no, no)

    t0 = np.concatenate([ts.ravel(), td.ravel()]).astype(complex)
    fsC, siC = fs.astype(complex), spinints.astype(complex)
    eta1 = fs[o, v].T + np.einsum("ijab,bj->ai", spinints[o, o, v, v], ts)
    eta2 = 0.25 * spinints[o, o, v, v].transpose(2, 3, 0, 1)
    eta = np.concatenate([eta1.ravel(), eta2.ravel()])
    if triples:
        # η_(T)[k] = ∂E_(T)/∂t_k by complex step of the non-canonical (T)
        # functional (f untouched ⇒ A stays diagonal ⇒ the solve is one-shot).
        for k in range(N):
            p = t0.copy()
            p[k] += 1j * eps_cs
            a1, a2 = unpack(p)
            eta[k] += _ccsd_t_energy_so(a1, a2, fsC, siC, nocc_so).imag / eps_cs
    rhs = -eta

    def j_action(x):
        p = t0 + 1j * eps_cs * np.asarray(x)
        a1, a2 = unpack(p)
        o1, o2 = _ccsd_residual_so(a1, a2, fsC, siC, nocc_so)
        return np.concatenate([o1.imag.ravel(), o2.imag.ravel()]) / eps_cs

    if solver == "dense":
        J = np.empty((N, N))
        for k in range(N):
            unit = np.zeros(N)
            unit[k] = 1.0
            J[:, k] = j_action(unit)
        lam, *_ = np.linalg.lstsq(J.T, rhs, rcond=None)
        return unpack(lam)

    from scipy.sparse.linalg import LinearOperator, lsmr

    def jt_action(x):
        x = np.asarray(x)
        out = np.empty(N)
        for k in range(N):
            p = t0.copy()
            p[k] += 1j * eps_cs
            a1, a2 = unpack(p)
            o1, o2 = _ccsd_residual_so(a1, a2, fsC, siC, nocc_so)
            scalar = (
                np.dot(x[:n1], o1.ravel())
                + np.dot(x[n1:], o2.ravel())
            )
            out[k] = scalar.imag / eps_cs
        return out

    operator = LinearOperator(
        (N, N), matvec=jt_action, rmatvec=j_action, dtype=float
    )
    limit = int(max_iter) if max_iter is not None else max(20, 2 * N)
    result = lsmr(operator, rhs, atol=tol, btol=tol, maxiter=limit)
    lam, istop, iterations, normr = result[:4]
    residual_limit = tol * max(1.0, float(np.linalg.norm(rhs)))
    if istop not in (0, 1, 2, 4, 5) or normr > residual_limit:
        raise RuntimeError(
            "matrix-free CCSD Lambda solve did not converge: "
            f"istop={istop}, iterations={iterations}, residual={normr:.3e}, "
            f"required<={residual_limit:.3e}"
        )
    return unpack(lam)


def _ccsd_lagrangian(ts, td, l1, l2, fs, spinints, nocc_so, *, triples=False):
    """CCSD(+T) correlation Lagrangian ``L = E_corr(t) [+ E_(T)] + ⟨Λ, Ω(t)⟩``.

    Stationary in t, Λ. With ``triples=True``, ``E_(T)`` is the non-canonical
    functional (§1.10) so every complex-step derivative of ``L`` carries the
    complete (T) densities.
    """
    from .ccsd import _ccsd_energy_so, _ccsd_residual_so

    E = _ccsd_energy_so(ts, td, fs, spinints, nocc_so)
    if triples:
        E = E + _ccsd_t_energy_so(ts, td, fs, spinints, nocc_so)
    o1, o2 = _ccsd_residual_so(ts, td, fs, spinints, nocc_so)
    return E + np.sum(l1 * o1) + np.sum(l2 * o2)


def _ccsd_cs_gamma1(ts, td, l1, l2, hf, mo, nocc_so, *, eps_cs=1e-30,
                    triples=False):
    """Response 1-PDM ``γ[p,q] = ∂L/∂hf[p,q]`` (spatial), symmetrized. O(n²) evals."""
    from .ccsd import _spin_block_eri_phys

    n = hf.shape[0]
    g = np.zeros((n, n))
    tsC, tdC, l1C, l2C = (x.astype(complex) for x in (ts, td, l1, l2))
    si = _spin_block_eri_phys(mo.astype(complex))
    for p in range(n):
        for q in range(n):
            hp = hf.astype(complex)
            hp[p, q] += 1j * eps_cs
            g[p, q] = _ccsd_lagrangian(
                tsC, tdC, l1C, l2C, _ccsd_fs_from_spatial(hp), si, nocc_so,
                triples=triples,
            ).imag / eps_cs
    return 0.5 * (g + g.T)


def _ccsd_cs_gamma2(ts, td, l1, l2, hf, mo, nocc_so, *, eps_cs=1e-30,
                    triples=False):
    """Non-separable 2-PDM cumulant ``Γ[p,q,r,s] = ∂L/∂mo`` (spatial chemist).

    Perturbs each 8-fold orbit of the real chemist ERI together, so the returned
    tensor carries the full permutational symmetry. O(n⁴/8) evals.
    """
    from .ccsd import _spin_block_eri_phys

    n = hf.shape[0]
    G = np.zeros((n, n, n, n))
    tsC, tdC, l1C, l2C = (x.astype(complex) for x in (ts, td, l1, l2))
    fsC = _ccsd_fs_from_spatial(hf.astype(complex))
    seen = np.zeros((n, n, n, n), dtype=bool)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if seen[p, q, r, s]:
                        continue
                    orbit = {(p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                             (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)}
                    mp = mo.astype(complex)
                    for (a, b, c, d) in orbit:
                        mp[a, b, c, d] += 1j * eps_cs
                    val = _ccsd_lagrangian(
                        tsC, tdC, l1C, l2C, fsC, _spin_block_eri_phys(mp), nocc_so,
                        triples=triples,
                    ).imag / eps_cs
                    per = val / len(orbit)
                    for (a, b, c, d) in orbit:
                        G[a, b, c, d] = per
                        seen[a, b, c, d] = True
    return G


def _ccsd_cs_genfock(ts, td, l1, l2, hcore_mo, mo, nocc_so, *, eps_cs=1e-30,
                     triples=False):
    """Generalized Fock ``GF[p,q] = ∂L/∂U`` for ``C → C(I+U)``, with the HF Fock
    REBUILT from bare ``h_core`` inside (so the reference-density G[P_ref] response
    under the orbital rotation is captured — the piece a rigid Fock transform
    misses). Antisymmetric ov block → Z-vector RHS; symmetric part → W-density.
    O(n²) evals. With ``triples=True`` the rebuilt (rotated) Fock acquires O(h)
    off-diagonals, which the non-canonical E_(T) solve turns into the previously
    missing (T) orbital-rotation derivative (§1.10.2 property 3).
    """
    from .ccsd import _spin_block_eri_phys

    n = hcore_mo.shape[0]
    GF = np.zeros((n, n))
    tsC, tdC, l1C, l2C = (x.astype(complex) for x in (ts, td, l1, l2))

    def lag_hcore(hc, m):
        spinints = _spin_block_eri_phys(m)
        fs = _ccsd_fs_from_spatial(hc) + np.einsum(
            "piqi->pq", spinints[:, :nocc_so, :, :nocc_so])
        from .ccsd import _ccsd_energy_so, _ccsd_residual_so
        E = _ccsd_energy_so(tsC, tdC, fs, spinints, nocc_so)
        if triples:
            E = E + _ccsd_t_energy_so(tsC, tdC, fs, spinints, nocc_so)
        o1, o2 = _ccsd_residual_so(tsC, tdC, fs, spinints, nocc_so)
        return E + np.sum(l1C * o1) + np.sum(l2C * o2)

    for p in range(n):
        for q in range(n):
            U = np.zeros((n, n), dtype=complex)
            U[p, q] = 1j * eps_cs
            hU = hcore_mo + U.T @ hcore_mo + hcore_mo @ U
            moU = (mo
                   + np.einsum("tp,tqrs->pqrs", U, mo)
                   + np.einsum("tq,ptrs->pqrs", U, mo)
                   + np.einsum("tr,pqts->pqrs", U, mo)
                   + np.einsum("ts,pqrt->pqrs", U, mo))
            GF[p, q] = lag_hcore(hU, moU).imag / eps_cs
    return GF


def _ccsd_zvector_genfock(zeta, C, h_ao, eri, nocc, *, eps_cs=1e-30):
    """Generalized Fock of the Z-vector constraint term (closed shell).

    ``Fz[p,q] = ∂/∂U_pq  Σ_ai ζ_ai F_ai(C(1+U))`` with the Fock rebuilt from
    ``h_AO + G[P[C(1+U)]]`` (occupied-density response included), ``ζ = −z``.
    Its ov-antisymmetric part is exactly ``M ζ`` (the CPHF supermatrix — the
    linearization is gated to 9e-10), so after the Z-vector solve the *total*
    Lagrangian generalized Fock ``GF + Fz`` is stationary (antisym residual
    ~1e-13) and the symmetric part ``¼(Fz + Fzᵀ)`` is the constraint's
    contribution to the energy-weighted density.

    This term was MISSING from the original CCSD assembly (the §1.9.3 claim
    that ``¼C(GF+GFᵀ)Cᵀ`` alone carries the orbital-relaxation weighting is
    wrong): it vanishes-ish on homonuclear symmetric chains where ``z`` is
    small — which is why every H₂-chain gate passed — but produces an
    h-independent analytic error on heteronuclear systems (LiH: 1.77e-5,
    root-caused 2026-07-17, fixed to the 1.6e-7 O(h²) floor by this term).
    The h- and ERI-side couplings of the same constraint were already correct:
    ``P^Δ``'s ``−½z`` in both triangles is the matrix form of the one-sided
    ``ζ = −z`` coupling.
    """
    n = C.shape[1]
    Fz = np.zeros((n, n))
    eye = np.eye(n)
    for p in range(n):
        for q in range(n):
            U = np.zeros((n, n), dtype=complex)
            U[p, q] = 1j * eps_cs
            CU = C @ (eye + U)
            PU = 2.0 * CU[:, :nocc] @ CU[:, :nocc].T
            F_mo = CU.T @ (h_ao + _g_ccm(eri, PU)) @ CU
            Fz[p, q] = (np.sum(zeta * F_mo[nocc:, :nocc])).imag / eps_cs
    return Fz


def run_ccm_ccsd_gradient(
    ccm,
    *,
    method: str = "aiccm2026dev-a",
    scf=None,
    compute_triples: bool = False,
    lambda_solver: str = "dense",
    lambda_tol: float = 1e-10,
    lambda_max_iter: int | None = None,
) -> np.ndarray:
    """Analytic Γ-CCM **CCSD** nuclear gradient ``dE_CCSD/dR`` per unit-cell atom (a.u.).

    The closed-shell CCSD *relaxed-density* gradient (§1.9 of
    ``docs/manuscripts/aiccm_a_forces.md``): the HF contraction pipeline (1.1)
    verbatim with the CCSD relaxed densities, all built from the *folded* MO
    integrals (the WSSC fold commutes with the Λ equations, the relaxed densities,
    and the CCM-CPHF orbital response, §1.2/§1.9)::

        dE_CCSD/dR =  Σ (P_SCF+P^Δ) ∂h^CCM/∂R          (kinetic L₂ᵀ + V_ne L_Vᵀ)
                   +  ½ Σ (Γ_HF+Γ_sep+Γ_ns) ∂(μν|λσ)^CCM/∂R      (four-center L₄ᵀ)
                   −  Σ (W_HF+W^Δ) ∂S^CCM/∂R                  (overlap Pulay L₂ᵀ)
                   +  ∂V_nn^CCM/∂R ,

    where ``P^Δ`` is the CCSD relaxed 1-PDM (amplitude/Λ density + the Z-vector
    orbital response, whose RHS is the ov block of the generalized Fock),
    ``Γ_sep = 2P^Δ⊗P − P^Δ⊗P_swap`` the separable 2-PDM, ``Γ_ns`` the CCSD cumulant,
    and ``W^Δ = ¼ C(GF+GFᵀ)Cᵀ`` the correlation energy-weighted density. All are
    obtained by exact complex-step differentiation of the CCSD correlation
    Lagrangian (see the module comment); the Z-vector reuses the SAME CPHF
    supermatrix ``A`` as MP2/HF. References: Scheiner-Scuseria-Lee-Rice-Schaefer
    1987, Gauss-Cremer 1987 (CCSD gradient); Handy-Schaefer 1984 (Z-vector);
    Stanton-Gauss-Watts-Bartlett 1991 (the amplitude equations).

    Consistency contract: the reference SCF and the correlation ride the *same*
    folded four-center. A hand-passed ``scf`` must be a
    :func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf` result on this ``method``'s ERI,
    converged tightly (``conv_tol≲1e-12``; the relaxed gradient is first-order
    sensitive to SCF non-stationarity). All-electron (no frozen core), closed
    shell. With ``compute_triples=True`` the gradient is **CCSD(T)**: the
    non-canonical (T) functional (§1.10, Watts-Gauss-Bartlett 1993) joins the
    Lagrangian, so the complex-step densities carry the complete (T)
    contributions — including the off-diagonal triples 1-PDM blocks a canonical
    (T) formula cannot produce (the h-independent ~2.5e-5 heteronuclear error
    diagnosed 2026-07-17). Dense spin-orbital / ``n_pad_ao⁴`` ⇒ small /
    1-D / 2-D clusters only. Returns ``(n_basis_atoms, 3)``; the per-cell forces sum
    to ≈ 0 (Theorem 1.6.1).

    ``lambda_solver="matrix-free"`` is the opt-in bounded-memory Λ path. It uses
    exact complex-step Jacobian-vector products with LSMR instead of storing the
    dense amplitude Jacobian. ``lambda_tol`` and ``lambda_max_iter`` control that
    iterative solve; the dense small-system validation oracle remains the default.

    Validation (§1.9): relaxed-1-PDM response identity to 1.9e-9 (H₂) / 1.1e-8
    (LiH, full-relaxation); Γ_ns MO response 4e-12 (H₂) / 1.6e-9 (LiH); geometric
    FD gate at the O(h²) floor (``4.4e-6 → 1.1e-6 → 3.4e-7`` for ``h = 2e-3,
    1e-3, 5e-4`` on the (3,1,1) H₂ chain); **heteronuclear** LiH (1,1,1) vs both
    the CCM energy FD and the independent molecular C++ CCSD FD to ~1.6e-7 (the
    O(h²)/FD floor — this gate requires the Z-vector constraint term ``Fz`` in
    ``W^Δ``, Eq. 1.9c'; see the §1.9.3 erratum); isolated (1,1,1) H₂ limit vs
    the molecular ``run_ccsd`` FD to 3.6e-9; sum rule 2.4e-15. With
    ``compute_triples=True``: LiH CCSD(T) vs CCM FD ``9.5e-8 / 1.6e-7 / 8.9e-8``
    at ``h = 2e-3, 1e-3, 5e-4`` (§1.10.5).

    .. warning::

       **FD comparisons need a unique SCF solution** — same as the MP2 gradient: on
       a symmetry-broken chain with near-degenerate branch partners (the (2,1,1) H₂
       chain), displaced-geometry SCFs branch-hop and the FD correlation reference
       is erratic; gate on (3,1,1). The analytic gradient is well-defined on the
       reference branch.
    """
    from .ccsd import _spin_block_eri_phys
    from .padded import ccm_hcore
    from .scf import _ccm_eri_for_method, run_ccm_rhf

    if method != "aiccm2026dev-a":
        raise NotImplementedError(
            f"run_ccm_ccsd_gradient supports method='aiccm2026dev-a' only, got {method!r}."
        )
    eri = _ccm_eri_for_method(ccm, method)
    if scf is None:
        scf = run_ccm_rhf(ccm, eri=eri, conv_tol=1e-12, max_iter=512)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_ccsd_gradient: the internal SCF did not converge to "
                "conv_tol=1e-12; the analytic gradient needs a stationary SCF."
            )
    n_elec = int(ccm.supercell.n_electrons())
    if n_elec % 2 != 0 or int(ccm.supercell.multiplicity) != 1:
        raise ValueError(
            "run_ccm_ccsd_gradient is closed-shell; open-shell CCSD gradients are "
            "follow-on work."
        )

    C = np.asarray(scf.mo_coeffs, dtype=float)
    if np.iscomplexobj(C):
        C = np.real_if_close(C, tol=1000).real
    C = np.asarray(C, dtype=float)
    eps = np.asarray(scf.mo_energies, dtype=float)
    n = C.shape[1]
    nocc = n_elec // 2
    nocc_so = 2 * nocc
    o, v = slice(0, nocc), slice(nocc, n)
    P = np.asarray(scf.density, dtype=float)

    hf = np.diag(eps)                                    # canonical spatial 1-e MO
    mo = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C, optimize=True)
    spinints = _spin_block_eri_phys(mo)
    fs = _ccsd_fs_from_spatial(hf)

    ts, td = _ccsd_solve_amplitudes(fs, spinints, nocc_so)
    l1, l2 = _ccsd_solve_lambda(
        ts,
        td,
        fs,
        spinints,
        nocc_so,
        triples=compute_triples,
        solver=lambda_solver,
        tol=lambda_tol,
        max_iter=lambda_max_iter,
    )

    gamma = _ccsd_cs_gamma1(ts, td, l1, l2, hf, mo, nocc_so,      # relaxed 1-PDM (sym)
                            triples=compute_triples)
    Gamma = _ccsd_cs_gamma2(ts, td, l1, l2, hf, mo, nocc_so,      # ∂E_corr/∂mo cumulant
                            triples=compute_triples)
    h_ao = np.asarray(ccm_hcore(ccm)[0], float)
    hcore_mo = C.T @ h_ao @ C
    GF = _ccsd_cs_genfock(ts, td, l1, l2, hcore_mo, mo, nocc_so,  # generalized Fock
                          triples=compute_triples)

    # Z-vector (CCM-CPHF): A z = X, X = antisym ov block of the generalized Fock.
    X = GF[v, o] - GF[o, v].T
    A = (4.0 * mo[v, o, v, o]
         - mo[v, v, o, o].transpose(0, 2, 1, 3)
         - mo[v, o, o, v].transpose(0, 2, 3, 1))
    nv = n - nocc
    M = A + np.einsum("ai,ab,ij->aibj", eps[v][:, None] - eps[o][None, :],
                      np.eye(nv), np.eye(nocc), optimize=True)
    z = np.linalg.solve(M.reshape(nv * nocc, nv * nocc), X.reshape(-1)).reshape(nv, nocc)

    # Relaxed 1-PDM: γ (amplitude/Λ) + orbital relaxation −½z in ov/vo (symmetric).
    d_full = gamma.copy()
    d_full[o, v] += -0.5 * z.T
    d_full[v, o] += -0.5 * z
    P_delta = C @ d_full @ C.T

    # 2-PDM AO: non-separable cumulant (2·∂E_corr/∂mo → the ½-core convention, cf.
    # MP2's 4t̃ = 2·2t̃), separable (P^Δ⊗P, MP2-normalization), and HF.
    gamma_ns_ao = 2.0 * np.einsum("pqrs,mp,nq,lr,os->mnlo", Gamma, C, C, C, C, optimize=True)
    gamma_hf = np.einsum("mn,ls->mnls", P, P) - 0.5 * np.einsum("ml,ns->mnls", P, P)
    gamma_sep = (2.0 * np.einsum("mn,ls->mnls", P_delta, P)
                 - np.einsum("ml,ns->mnls", P_delta, P))

    # Energy-weighted density: HF + W^Δ = ¼ C(GF+GFᵀ)Cᵀ + ¼ C(Fz+Fzᵀ)Cᵀ. The
    # second term is the Z-vector constraint's generalized Fock (ζ = −z); with
    # it the total Lagrangian is stationary in ALL orbital rotations (antisym
    # residual ~1e-13). Missing it is the heteronuclear W bug fixed 2026-07-17
    # (see _ccsd_zvector_genfock).
    W_hf = _energy_weighted_density(eps, C, nocc, 2.0)
    Fz = _ccsd_zvector_genfock(-z, C, h_ao, eri, nocc)
    W_delta = 0.25 * C @ (GF + GF.T + Fz + Fz.T) @ C.T

    return (
        _kinetic_gradient_P(ccm, P + P_delta)
        + _nuclear_attraction_gradient_P(ccm, P + P_delta)
        + _two_electron_gradient_gamma(ccm, gamma_hf + gamma_sep + gamma_ns_ao)
        + _overlap_pulay_gradient_W(ccm, W_hf + W_delta)
        + ccm_nuclear_repulsion_gradient(ccm)
    )


# --------------------------------------------------------------------------- #
# UCCSD relaxed-density gradient (open shell) — §1.9.5 of aiccm_a_forces.md
# --------------------------------------------------------------------------- #
# The open-shell mirror of run_ccm_ccsd_gradient: the spin-orbital CCSD residual
# (Stanton 1991) is spin-agnostic in (fs, spinints), so a UHF reference is handled
# by building the UHF spin-orbital tensors (α/β orbitals, occ-first mixed-spin
# ordering) and reusing the ENTIRE closed-shell complex-step machinery — the
# amplitude/Λ solve, the Lagrangian, and the exact complex-step densities. Only the
# assembly becomes spin-resolved (UMP2 bookkeeping, §1.8.4): per-spin relaxed 1-PDMs
# γ^σ, the spin-coupled Z-vector, per-spin generalized Focks, the UHF Γ_HF, and the
# spin-block cumulant. References as CCSD (§1.9); the open-shell relaxed-density
# gradient follows Gauss-Lauderdale-Stanton-Watts-Bartlett, Chem. Phys. Lett. 182,
# 207 (1991).


def _uhf_spinorbital_indices(n, na, nb):
    """(spat, spin, nocc_so): occ-first mixed-spin order [α-occ, β-occ | α-vir, β-vir]."""
    occ = [(p, 0) for p in range(na)] + [(p, 1) for p in range(nb)]
    vir = [(p, 0) for p in range(na, n)] + [(p, 1) for p in range(nb, n)]
    lst = occ + vir
    spat = np.array([p for p, s in lst], dtype=int)
    spin = np.array([s for p, s in lst], dtype=int)
    return spat, spin, len(occ)


def _build_uhf_spinorbital(fa, fb, g_aa, g_bb, g_ab, na, nb):
    """Spin-orbital (fs, spinints, nocc_so) from spatial α/β MO Fock (fa,fb) and
    chemist ERIs g_aa=(αα|αα), g_bb=(ββ|ββ), g_ab=(αα|ββ). ``g_ab`` carries the
    (ββ|αα) block by its bra-ket transpose (built here), so ∂L/∂g_ab already sums
    both cross-spin contributions."""
    n = fa.shape[0]
    spat, spin, nocc_so = _uhf_spinorbital_indices(n, na, nb)
    nso = 2 * n
    dt = np.result_type(fa, g_aa, g_ab)

    fs = np.zeros((nso, nso), dtype=dt)
    same = spin[:, None] == spin[None, :]
    aa = same & (spin[:, None] == 0)
    bb = same & (spin[:, None] == 1)
    fs[aa] = fa[np.ix_(spat, spat)][aa]
    fs[bb] = fb[np.ix_(spat, spat)][bb]

    ia = np.where(spin == 0)[0]
    ib = np.where(spin == 1)[0]
    chem = np.zeros((nso, nso, nso, nso), dtype=dt)
    sp = spat
    chem[np.ix_(ia, ia, ia, ia)] = g_aa[np.ix_(sp[ia], sp[ia], sp[ia], sp[ia])]
    chem[np.ix_(ib, ib, ib, ib)] = g_bb[np.ix_(sp[ib], sp[ib], sp[ib], sp[ib])]
    chem[np.ix_(ia, ia, ib, ib)] = g_ab[np.ix_(sp[ia], sp[ia], sp[ib], sp[ib])]
    chem[np.ix_(ib, ib, ia, ia)] = g_ab.transpose(2, 3, 0, 1)[
        np.ix_(sp[ib], sp[ib], sp[ia], sp[ia])]
    # physicist antisym <PQ||RS> = (PR|QS) − (PS|QR)
    spinints = np.einsum("prqs->pqrs", chem) - np.einsum("psqr->pqrs", chem)
    return fs, spinints, nocc_so


def _uccsd_lagrangian(ts, td, l1, l2, fa, fb, g_aa, g_bb, g_ab, na, nb,
                      *, triples=False):
    from .ccsd import _ccsd_energy_so, _ccsd_residual_so

    fs, spinints, nocc_so = _build_uhf_spinorbital(fa, fb, g_aa, g_bb, g_ab, na, nb)
    E = _ccsd_energy_so(ts, td, fs, spinints, nocc_so)
    if triples:
        # the non-canonical (T) functional (§1.10) is spin-agnostic in
        # (fs, spinints) — same completeness argument as closed shell.
        E = E + _ccsd_t_energy_so(ts, td, fs, spinints, nocc_so)
    o1, o2 = _ccsd_residual_so(ts, td, fs, spinints, nocc_so)
    return E + np.sum(l1 * o1) + np.sum(l2 * o2)


def _uhf_fock_mo(hca, hcb, g_aa, g_bb, g_ab, na, nb):
    """Rebuild the canonical UHF MO Fock (fa, fb) from bare h_core + occ projectors
    (J from the total density, K from the same-spin density)."""
    oa, ob = slice(0, na), slice(0, nb)
    fa = (hca + np.einsum("pqii->pq", g_aa[:, :, oa, oa])
          - np.einsum("piiq->pq", g_aa[:, oa, oa, :])
          + np.einsum("pqjj->pq", g_ab[:, :, ob, ob]))
    fb = (hcb + np.einsum("pqjj->pq", g_bb[:, :, ob, ob])
          - np.einsum("pjjq->pq", g_bb[:, ob, ob, :])
          + np.einsum("iipq->pq", g_ab[oa, oa, :, :]))
    return fa, fb


def _uccsd_cs_gamma1(ts, td, l1, l2, fa, fb, g_aa, g_bb, g_ab, na, nb, which,
                     *, eps_cs=1e-30, triples=False):
    """γ^σ = ∂L/∂f^σ (symmetrized), σ = which ∈ {'a','b'}. O(n²)."""
    n = fa.shape[0]
    G = np.zeros((n, n))
    args = [x.astype(complex) for x in (ts, td, l1, l2)]
    faC, fbC = fa.astype(complex), fb.astype(complex)
    ga, gb, gab = (g.astype(complex) for g in (g_aa, g_bb, g_ab))
    for p in range(n):
        for q in range(n):
            fap, fbp = faC.copy(), fbC.copy()
            (fap if which == "a" else fbp)[p, q] += 1j * eps_cs
            G[p, q] = _uccsd_lagrangian(*args, fap, fbp, ga, gb, gab, na, nb,
                                        triples=triples).imag / eps_cs
    return 0.5 * (G + G.T)


def _uccsd_cs_gamma2(ts, td, l1, l2, fa, fb, g_aa, g_bb, g_ab, na, nb, which,
                     *, eps_cs=1e-30, triples=False):
    """Cumulant spin-block Γ^{block} = ∂L/∂g_block (spatial chemist). 'aa'/'bb'
    carry 8-fold symmetry; 'ab'=(αα|ββ) carries 4-fold (p↔q, r↔s)."""
    n = fa.shape[0]
    G = np.zeros((n, n, n, n))
    args = [x.astype(complex) for x in (ts, td, l1, l2)]
    faC, fbC = fa.astype(complex), fb.astype(complex)
    base = {"aa": g_aa, "bb": g_bb, "ab": g_ab}
    seen = np.zeros((n, n, n, n), dtype=bool)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if seen[p, q, r, s]:
                        continue
                    if which in ("aa", "bb"):
                        orbit = {(p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                                 (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)}
                    else:
                        orbit = {(p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r)}
                    gp = {k: v.astype(complex) for k, v in base.items()}
                    for (a, b, c, d) in orbit:
                        gp[which][a, b, c, d] += 1j * eps_cs
                    val = _uccsd_lagrangian(*args, faC, fbC, gp["aa"], gp["bb"], gp["ab"],
                                            na, nb, triples=triples).imag / eps_cs
                    per = val / len(orbit)
                    for (a, b, c, d) in orbit:
                        G[a, b, c, d] = per
                        seen[a, b, c, d] = True
    return G


def _uccsd_cs_genfock(ts, td, l1, l2, hca, hcb, g_aa, g_bb, g_ab, na, nb, which,
                      *, eps_cs=1e-30, triples=False):
    """Generalized Fock GF^σ = ∂L/∂U^σ (rotate α resp. β orbitals), Fock rebuilt from
    h_core so the reference response is captured (the α rotation also moves the β
    Fock's Coulomb, and vice versa, through the transformed g_ab)."""
    n = hca.shape[0]
    GF = np.zeros((n, n))
    args = [x.astype(complex) for x in (ts, td, l1, l2)]

    def lag(hca_, hcb_, ga_, gb_, gab_):
        fa, fb = _uhf_fock_mo(hca_, hcb_, ga_, gb_, gab_, na, nb)
        return _uccsd_lagrangian(*args, fa, fb, ga_, gb_, gab_, na, nb,
                                 triples=triples)

    hcaC, hcbC = hca.astype(complex), hcb.astype(complex)
    gaC, gbC, gabC = (g.astype(complex) for g in (g_aa, g_bb, g_ab))
    for p in range(n):
        for q in range(n):
            U = np.zeros((n, n), dtype=complex)
            U[p, q] = 1j * eps_cs
            if which == "a":
                hca2 = hcaC + U.T @ hcaC + hcaC @ U
                ga2 = (gaC + np.einsum("tp,tqrs->pqrs", U, gaC)
                       + np.einsum("tq,ptrs->pqrs", U, gaC)
                       + np.einsum("tr,pqts->pqrs", U, gaC)
                       + np.einsum("ts,pqrt->pqrs", U, gaC))
                gab2 = (gabC + np.einsum("tp,tqrs->pqrs", U, gabC)
                        + np.einsum("tq,ptrs->pqrs", U, gabC))    # only α pair
                GF[p, q] = lag(hca2, hcbC, ga2, gbC, gab2).imag / eps_cs
            else:
                hcb2 = hcbC + U.T @ hcbC + hcbC @ U
                gb2 = (gbC + np.einsum("tp,tqrs->pqrs", U, gbC)
                       + np.einsum("tq,ptrs->pqrs", U, gbC)
                       + np.einsum("tr,pqts->pqrs", U, gbC)
                       + np.einsum("ts,pqrt->pqrs", U, gbC))
                gab2 = (gabC + np.einsum("tr,pqts->pqrs", U, gabC)
                        + np.einsum("ts,pqrt->pqrs", U, gabC))    # only β pair
                GF[p, q] = lag(hcaC, hcb2, gaC, gb2, gab2).imag / eps_cs
    return GF


def _uccsd_zvector_genfock(zeta_a, zeta_b, Ca, Cb, h_ao, eri, na, nb, which,
                           *, eps_cs=1e-30):
    """Open-shell Z-vector constraint generalized Fock (the UHF mirror of
    :func:`_ccsd_zvector_genfock`).

    ``Fz^σ[p,q] = ∂/∂U^σ_pq [Σ_ai ζ^α_ai F^α_ai + Σ_ai ζ^β_ai F^β_ai]`` with
    both UHF Fock matrices rebuilt from ``h_AO`` and the rotated spin
    densities — rotating one spin's orbitals perturbs the *other* spin's Fock
    through the total-density Coulomb term, so both constraint pieces
    contribute to either spin's ``Fz``. ``ζ^σ = −z^σ``.
    """
    n = Ca.shape[1]
    Fz = np.zeros((n, n))
    eye = np.eye(n)
    for p in range(n):
        for q in range(n):
            U = np.zeros((n, n), dtype=complex)
            U[p, q] = 1j * eps_cs
            CaU = Ca @ (eye + U) if which == "a" else Ca.astype(complex)
            CbU = Cb @ (eye + U) if which == "b" else Cb.astype(complex)
            Pa = CaU[:, :na] @ CaU[:, :na].T
            Pb = CbU[:, :nb] @ CbU[:, :nb].T
            Fa = CaU.T @ (h_ao + _g_ccm_uhf(eri, Pa, Pb, 0)) @ CaU
            Fb = CbU.T @ (h_ao + _g_ccm_uhf(eri, Pa, Pb, 1)) @ CbU
            val = (np.sum(zeta_a * Fa[na:, :na])
                   + np.sum(zeta_b * Fb[nb:, :nb]))
            Fz[p, q] = val.imag / eps_cs
    return Fz


def run_ccm_uccsd_gradient(
    ccm,
    *,
    method: str = "aiccm2026dev-a",
    scf=None,
    compute_triples: bool = False,
    lambda_solver: str = "dense",
    lambda_tol: float = 1e-10,
    lambda_max_iter: int | None = None,
) -> np.ndarray:
    """Analytic Γ-CCM **UCCSD** nuclear gradient ``dE_UCCSD/dR`` per unit-cell atom (a.u.).

    The open-shell mirror of :func:`run_ccm_ccsd_gradient` (§1.9.5 of
    ``docs/manuscripts/aiccm_a_forces.md``): the spin-orbital CCSD residual is
    spin-agnostic, so a UHF reference is handled by building the UHF spin-orbital
    tensors and reusing the complex-step CCSD machinery, with the assembly made
    spin-resolved (UMP2 bookkeeping, §1.8.4). All AO contractions reuse the SAME
    ``L₂ᵀ/L_Vᵀ/L₄ᵀ`` cores, and the orbital relaxation is the spin-coupled CCM-CPHF
    (same supermatrix ``M`` as UMP2, Eq. 1.8l). Requires a converged
    :func:`~vibeqc.periodic.ccm.uhf.run_ccm_uhf` reference on this ``method``'s ERI
    (tightly — the relaxed gradient is first-order sensitive to SCF
    non-stationarity; pass a tight ``conv_tol_grad``). UCCSD (no ``(T)``); dense
    spin-orbital ⇒ small / 1-D / 2-D clusters only. Returns ``(n_basis_atoms, 3)``;
    per-cell forces sum to ≈ 0.

    Validation (§1.9.5): closed-shell reduction to :func:`run_ccm_ccsd_gradient`
    machine-exact (7.3e-15 plain; 1.6e-13 with ``compute_triples=True``); frozen
    1-e/cumulant responses ~1e-9; relaxed-1-PDM response identity 2.4e-9 (HeH
    chain (3,1,1)); heteronuclear HeH doublet FD gate at the O(h²) floor
    (``4.9e-6 → 1.2e-6`` for ``h = 2e-3 → 1e-3`` with triples) — this gate
    requires the per-spin Z-vector constraint term ``Fz^σ`` in ``W^Δ`` (see
    :func:`_uccsd_zvector_genfock`; the pre-fix "2.6e-6 open-shell FD floor"
    reading was that missing term, §1.9.3 erratum); sum rule 1.6e-14. With
    ``compute_triples=True`` the (T) correction uses the non-canonical
    functional (§1.10) on the UHF spin-orbital tensors.

    ``lambda_solver="matrix-free"`` selects the same bounded-memory iterative
    Λ solve as the closed-shell driver. The dense small-system oracle remains
    the default.
    """
    from .ccsd import _spin_block_eri_phys  # noqa: F401  (parity; not used directly)
    from .padded import ccm_hcore
    from .scf import _ccm_eri_for_method
    from .uhf import run_ccm_uhf

    if method != "aiccm2026dev-a":
        raise NotImplementedError(
            f"run_ccm_uccsd_gradient supports method='aiccm2026dev-a' only, got {method!r}."
        )
    eri = _ccm_eri_for_method(ccm, method)
    if scf is None:
        scf = run_ccm_uhf(ccm, method=method, conv_tol=1e-12, conv_tol_grad=1e-10,
                          max_iter=512)
        if not scf.converged:
            raise RuntimeError(
                "run_ccm_uccsd_gradient: the internal UHF SCF did not converge; the "
                "analytic gradient needs a stationary reference."
            )

    Ca = np.asarray(scf.mo_coeffs_alpha, dtype=float)
    Cb = np.asarray(scf.mo_coeffs_beta, dtype=float)
    ea = np.asarray(scf.mo_energies_alpha, dtype=float)
    eb = np.asarray(scf.mo_energies_beta, dtype=float)
    na, nb = int(scf.n_alpha), int(scf.n_beta)
    n = Ca.shape[1]
    oa, va = slice(0, na), slice(na, n)
    ob, vb = slice(0, nb), slice(nb, n)

    g_aa = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, Ca, Ca, Ca, Ca, optimize=True)
    g_bb = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, Cb, Cb, Cb, Cb, optimize=True)
    g_ab = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, Ca, Ca, Cb, Cb, optimize=True)
    fa, fb = np.diag(ea), np.diag(eb)
    fs, spinints, nocc_so = _build_uhf_spinorbital(fa, fb, g_aa, g_bb, g_ab, na, nb)

    ts, td = _ccsd_solve_amplitudes(fs, spinints, nocc_so)
    l1, l2 = _ccsd_solve_lambda(
        ts,
        td,
        fs,
        spinints,
        nocc_so,
        triples=compute_triples,
        solver=lambda_solver,
        tol=lambda_tol,
        max_iter=lambda_max_iter,
    )

    a5 = (ts, td, l1, l2, fa, fb, g_aa, g_bb, g_ab, na, nb)
    ga1 = _uccsd_cs_gamma1(*a5, "a", triples=compute_triples)
    gb1 = _uccsd_cs_gamma1(*a5, "b", triples=compute_triples)
    Gaa = _uccsd_cs_gamma2(*a5, "aa", triples=compute_triples)
    Gbb = _uccsd_cs_gamma2(*a5, "bb", triples=compute_triples)
    Gab = _uccsd_cs_gamma2(*a5, "ab", triples=compute_triples)

    h_ao = np.asarray(ccm_hcore(ccm)[0], float)
    hca, hcb = Ca.T @ h_ao @ Ca, Cb.T @ h_ao @ Cb
    GFa = _uccsd_cs_genfock(ts, td, l1, l2, hca, hcb, g_aa, g_bb, g_ab, na, nb,
                            "a", triples=compute_triples)
    GFb = _uccsd_cs_genfock(ts, td, l1, l2, hca, hcb, g_aa, g_bb, g_ab, na, nb,
                            "b", triples=compute_triples)

    # spin-coupled Z-vector (CCM-CPHF, Eq. 1.8l): M [za; zb] = [Xa; Xb].
    Xa = GFa[va, oa] - GFa[oa, va].T
    Xb = GFb[vb, ob] - GFb[ob, vb].T

    def m_same(g, eo, ev, o, v):
        A = (2.0 * g[v, o, v, o] - g[v, v, o, o].transpose(0, 2, 1, 3)
             - g[v, o, o, v].transpose(0, 2, 3, 1))
        return A + np.einsum("ai,ab,ij->aibj", ev[:, None] - eo[None, :],
                             np.eye(len(ev)), np.eye(len(eo)), optimize=True)

    nva, nvb = n - na, n - nb
    Maa = m_same(g_aa, ea[oa], ea[va], oa, va).reshape(nva * na, nva * na)
    Mbb = m_same(g_bb, eb[ob], eb[vb], ob, vb).reshape(nvb * nb, nvb * nb)
    Mab = 2.0 * g_ab[va, oa, vb, ob].reshape(nva * na, nvb * nb)
    zvec = np.linalg.solve(np.block([[Maa, Mab], [Mab.T, Mbb]]),
                           np.concatenate([Xa.reshape(-1), Xb.reshape(-1)]))
    za = zvec[:nva * na].reshape(nva, na)
    zb = zvec[nva * na:].reshape(nvb, nb)

    da = ga1.copy()
    da[oa, va] += -0.5 * za.T
    da[va, oa] += -0.5 * za
    db = gb1.copy()
    db[ob, vb] += -0.5 * zb.T
    db[vb, ob] += -0.5 * zb
    PDa, PDb = Ca @ da @ Ca.T, Cb @ db @ Cb.T
    PD = PDa + PDb

    Pa = np.asarray(scf.density_alpha, float)
    Pb = np.asarray(scf.density_beta, float)
    P = Pa + Pb

    # 2-PDM AO: cumulant (spin-block backtransforms; ∂L/∂g_ab already sums the
    # (ββ|αα) block) ×2 for the ½-core; separable (UMP2 form); UHF Γ_HF.
    ns_aa = np.einsum("pqrs,mp,nq,lr,os->mnlo", Gaa, Ca, Ca, Ca, Ca, optimize=True)
    ns_bb = np.einsum("pqrs,mp,nq,lr,os->mnlo", Gbb, Cb, Cb, Cb, Cb, optimize=True)
    ns_ab = np.einsum("pqrs,mp,nq,lr,os->mnlo", Gab, Ca, Ca, Cb, Cb, optimize=True)
    gamma_ns_ao = 2.0 * (ns_aa + ns_bb + ns_ab)
    gamma_hf = (np.einsum("mn,ls->mnls", P, P)
                - np.einsum("ml,ns->mnls", Pa, Pa)
                - np.einsum("ml,ns->mnls", Pb, Pb))
    gamma_sep = (2.0 * np.einsum("mn,ls->mnls", PD, P)
                 - 2.0 * np.einsum("ml,ns->mnls", PDa, Pa)
                 - 2.0 * np.einsum("ml,ns->mnls", PDb, Pb))

    W_hf = (_energy_weighted_density(ea, Ca, na, 1.0)
            + _energy_weighted_density(eb, Cb, nb, 1.0))
    # Z-vector constraint term per spin (ζ^σ = −z^σ) — the open-shell mirror
    # of the closed-shell heteronuclear W fix (see _ccsd_zvector_genfock).
    Fza = _uccsd_zvector_genfock(-za, -zb, Ca, Cb, h_ao, eri, na, nb, "a")
    Fzb = _uccsd_zvector_genfock(-za, -zb, Ca, Cb, h_ao, eri, na, nb, "b")
    W_delta = 0.25 * (Ca @ (GFa + GFa.T + Fza + Fza.T) @ Ca.T
                      + Cb @ (GFb + GFb.T + Fzb + Fzb.T) @ Cb.T)

    return (
        _kinetic_gradient_P(ccm, P + PD)
        + _nuclear_attraction_gradient_P(ccm, P + PD)
        + _two_electron_gradient_gamma(ccm, gamma_hf + gamma_sep + gamma_ns_ao)
        + _overlap_pulay_gradient_W(ccm, W_hf + W_delta)
        + ccm_nuclear_repulsion_gradient(ccm)
    )
