"""Padded-cluster molecular-integral path for the CCM (small-system prototype).

The two-center CCM integrals (:mod:`vibeqc.periodic.ccm.integrals`) fold
vibe-qc's real-space *lattice* blocks against the WSSC weights -- efficient
and the production path. The three-center (nuclear attraction) and
four-center (ERI) CCM terms need integrals between a home basis function
and a *specific translational image* of another (and, for V, against an
arbitrary point charge), which the lattice-block primitives do not expose
per-image / per-nucleus.

This module supplies them by an explicit **padded cluster**: the reference
supercell plus its cluster-lattice images out to ``±image_range`` built as
one molecule, on which ordinary molecular integrals are evaluated. CCM
matrices over the reference cell are then assembled by selecting the
home<->image sub-blocks and applying the WSSC weights. It is O((images .
n_atoms)^k) and therefore a **validation / small-cluster prototype**, not
the production route for large 3-D clusters (that needs a C++ lattice-sum
with screening -- a later milestone). Its correctness is anchored by
cross-checking the two-center fold against the verified lattice path.

References (Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550):

    # Three-center nuclear attraction, eqs (12)-(13):
    #   V^CCM_{mu nu} = - sum_{nu'} sum_{C in WSSC(M) ∪ WSSC(N)} omega_{mu nu' C} I_{mu nu' C}
    #   omega_{mu nu' C} = omega_{mu nu'} * (omega_{mu C} + omega_{nu C}) / 2
    #   I_{mu nu' C} = <mu | Z_C / r_C | nu'>
    # The nucleus weight omega_{A,(C0,g)} is exactly the two-center pair weight
    # W[g][A, C0] already produced by CCMSystem.cell_weight_matrices().
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from vibeqc import (  # type: ignore
    BasisSet,
    compute_kinetic,
    compute_nuclear,
    compute_overlap,
)
from vibeqc._vibeqc_core import Atom, Molecule

__all__ = [
    "PaddedCluster",
    "build_padded_cluster",
    "ccm_overlap_padded",
    "ccm_kinetic_padded",
    "ccm_nuclear",
    "ccm_eri",
    "ccm_eri_symmetric",
    "ccm_hcore",
]


@dataclass
class PaddedCluster:
    """Reference supercell + cluster-lattice images as one molecular system."""

    basis: object
    n_ref_ao: int
    cell_to_cols: dict          # g_tuple -> (n_ref_ao,) pad-AO index of each ref AO's image
    pad_positions: np.ndarray   # (n_pad_atom, 3) bohr
    pad_Z: np.ndarray           # (n_pad_atom,)
    pad_atom_of: list           # pad atom index -> (ref atom A, cell g)
    atom_g_to_pad: dict         # (A, g_tuple) -> pad atom index


def _ao_ranges(basis):
    """List, per atom index, of the AO indices that atom contributes."""
    out: dict = {}
    off = 0
    for sh in basis.shells():
        l = int(sh.l)
        n_ao = (2 * l + 1) if sh.pure else (l + 1) * (l + 2) // 2
        out.setdefault(int(sh.atom_index), []).extend(range(off, off + n_ao))
        off += n_ao
    return out, off


def _padded_eri_max_bytes() -> int:
    """Memory ceiling (bytes) for the dense padded-cluster ERI tensor.

    Overridable via ``VIBEQC_CCM_PADDED_ERI_MAX_GB`` (default 16 GiB). The dense
    direct four-center is an O(n_pad⁴) screening-free *validation* builder; the
    ceiling turns an out-of-memory crash on a too-large cluster into an
    actionable error pointing first at the scalable four-center route for the
    same Γ-CCM construction, then at GDF only as a different declared control.
    """
    env = os.environ.get("VIBEQC_CCM_PADDED_ERI_MAX_GB")
    if env:
        try:
            return int(float(env) * (1024 ** 3))
        except ValueError:
            pass
    return 16 * (1024 ** 3)


def _check_padded_eri_size(pad: "PaddedCluster", what: str) -> int:
    """Guard the dense padded ERI allocation; return the padded AO count.

    The blow-up in :func:`ccm_eri` / :func:`ccm_eri_symmetric` is
    ``compute_eri(pad.basis)`` — a dense ``n_pad⁴`` float64 tensor over the
    *padded* basis (home cell + every image cell of the ±2t fold), which is
    ``n_pad = n_ref_ao × (number of ERI image cells)`` and therefore explodes for
    >2-atom / 3-D cells (e.g. c-diamond 2×2×2: ~80 reference AOs → thousands of
    padded AOs → a TB-scale tensor). Raise a clear, actionable error *before* the
    allocation rather than letting NumPy OOM.
    """
    _, n_pad = _ao_ranges(pad.basis)
    nbytes = (int(n_pad) ** 4) * 8                      # dense float64 (n_pad)^4 ERI
    ceiling = _padded_eri_max_bytes()
    if nbytes > ceiling:
        raise MemoryError(
            f"CCM direct four-center ({what}): the padded-cluster ERI is a dense "
            f"{n_pad}^4 float64 tensor = {nbytes / 1024**3:.1f} GiB, over the "
            f"{ceiling / 1024**3:.1f} GiB ceiling. This builder is an O(n^4), "
            f"screening-free *validation* path for small / 1-D / 2-D clusters "
            f"only (n_pad = n_ref_ao × ERI-image-cells blows up in 3-D). For "
            f"the same union-and-weight Γ-CCM construction, use the scalable C++ "
            f"four-center `run_ccm_rhf_scalable` (the lattice-sum JK builder that "
            f"applies the WSSC weights in the shell-quartet loop, no padded ERI). "
            f"If the intended operator is instead the separately declared neutral "
            f"fitted-torus control, use `run_ccm_rhf_gdf`; it is not a Γ-CCM "
            f"substitute or construction proof. Both routes scale to genuine 3-D. "
            f"Raise the ceiling with "
            f"VIBEQC_CCM_PADDED_ERI_MAX_GB only if you have the RAM."
        )
    return int(n_pad)


def wssc_cells(ccm) -> list:
    """Cluster cells that are a minimum image of some pair (incl. home)."""
    cells = {(0, 0, 0)} | set(ccm.cell_weight_matrices().keys())
    return sorted(cells)


def eri_cells(ccm) -> list:
    """Cells needed for the four-center fold: WSSC cells and their pairwise sums.

    The electron-2 pair reaches ``c`` at a WSSC cell ``g_c`` and ``d`` at the
    minimum image of ``s`` relative to ``c`` (another WSSC cell ``g_d``), so the
    farthest padded image is at ``g_c + g_d`` -- the ``±2t`` range of eq. (18).
    Building only this set (not a full ``±2`` product) keeps the vacuum
    directions of low-dimensional systems from exploding the AO count.
    """
    w = list(ccm.cell_weight_matrices().keys())
    cells = {(0, 0, 0)} | set(w)
    for a in w:
        for b in w:
            cells.add((a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    return sorted(cells)


def build_padded_cluster(ccm, cells) -> PaddedCluster:
    """Build the padded cluster molecule and the home->image AO column maps.

    ``cells`` is an explicit list of cluster-cell index tuples to materialise
    as images (``(0,0,0)`` is always included and placed first so the
    reference AOs lead the basis). Use :func:`wssc_cells` for two/three-center
    work and :func:`eri_cells` for the four-center fold.
    """
    Ac = ccm.cluster_vectors  # rows = cluster vectors
    cells = [(0, 0, 0)] + [
        tuple(int(x) for x in c) for c in cells if tuple(int(x) for x in c) != (0, 0, 0)
    ]

    Zref = [int(a.Z) for a in ccm.supercell.atoms]
    pad_atoms = []
    pad_atom_of = []
    atom_g_to_pad = {}
    for g in cells:
        Rg = np.asarray(g, dtype=float) @ Ac
        for A in range(ccm.n_atoms):
            atom_g_to_pad[(A, g)] = len(pad_atoms)
            pad_atom_of.append((A, g))
            pad_atoms.append(Atom(Zref[A], (ccm.atom_positions[A] + Rg).tolist()))

    n_elec = sum(a.Z for a in pad_atoms)
    pad_mol = Molecule(pad_atoms, 0, 1 if n_elec % 2 == 0 else 2)
    basis = BasisSet(pad_mol, ccm.basis_name)

    ao_of_atom, n_pad_ao = _ao_ranges(basis)
    n_ref_ao = ccm.nbf

    # The home-cell atoms are pad atoms 0..n_atoms-1 in reference order, with the
    # same per-atom basis, so the first n_ref_ao AOs are exactly ccm.basis's AOs.
    home_aos = []
    for A in range(ccm.n_atoms):
        home_aos.extend(ao_of_atom[atom_g_to_pad[(A, (0, 0, 0))]])
    if home_aos != list(range(n_ref_ao)):
        raise RuntimeError("padded home-cell AO ordering does not match reference basis")

    # Local AO index (position within its atom's AO block) for each reference AO.
    ref_ao_atom = ccm.ao_atom
    local = np.zeros(n_ref_ao, dtype=int)
    counter: dict = {}
    for ao in range(n_ref_ao):
        a = int(ref_ao_atom[ao])
        local[ao] = counter.get(a, 0)
        counter[a] = counter.get(a, 0) + 1

    cell_to_cols = {}
    for g in cells:
        cols = np.empty(n_ref_ao, dtype=int)
        for ao in range(n_ref_ao):
            a = int(ref_ao_atom[ao])
            cols[ao] = ao_of_atom[atom_g_to_pad[(a, g)]][local[ao]]
        cell_to_cols[g] = cols

    pad_positions = np.array([list(a.xyz) for a in pad_atoms], dtype=float)
    pad_Z = np.array(Zref * len(cells), dtype=int)
    return PaddedCluster(basis, n_ref_ao, cell_to_cols, pad_positions, pad_Z,
                         pad_atom_of, atom_g_to_pad)


def _fold_two_center(ccm, pad: PaddedCluster, m_pad: np.ndarray) -> np.ndarray:
    """Fold a padded molecular two-center matrix into the CCM matrix (eqs 5-6)."""
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    rows = np.arange(pad.n_ref_ao)
    out = np.zeros((ccm.nbf, ccm.nbf))
    for g, w_atom in weights.items():
        cols = pad.cell_to_cols.get(g)
        if cols is None:
            raise ValueError(
                f"padded cluster too small: minimum-image cell {g} not built "
                "(increase image_range)."
            )
        block = m_pad[np.ix_(rows, cols)]
        out += w_atom[ao[:, None], ao[None, :]] * block
    return 0.5 * (out + out.T)


def ccm_overlap_padded(ccm) -> np.ndarray:
    """``S^CCM`` via the padded path -- for cross-checking the lattice fold."""
    pad = build_padded_cluster(ccm, wssc_cells(ccm))
    return _fold_two_center(ccm, pad, compute_overlap(pad.basis))


def ccm_kinetic_padded(ccm) -> np.ndarray:
    """``T^CCM`` via the padded path -- for cross-checking the lattice fold."""
    pad = build_padded_cluster(ccm, wssc_cells(ccm))
    return _fold_two_center(ccm, pad, compute_kinetic(pad.basis))


def ccm_nuclear(ccm, pad: "PaddedCluster | None" = None) -> np.ndarray:
    """CCM-weighted nuclear-attraction matrix ``V^CCM`` (eqs 12-13).

    Union-of-WSSC three-center weighting. The nucleus weight ``omega_{A,C}``
    is the two-center pair weight ``W[g][A, C0]`` from
    :meth:`CCMSystem.cell_weight_matrices` (nucleus image ``C=(C0,g)``), so no
    new geometry is needed. Returns an ``(nbf, nbf)`` matrix.
    """
    if pad is None:
        pad = build_padded_cluster(ccm, wssc_cells(ccm))
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    rows = np.arange(pad.n_ref_ao)
    aoA, aoB = ao[:, None], ao[None, :]

    out = np.zeros((ccm.nbf, ccm.nbf))
    # Each nucleus image c = (C0, gc) contributes only if gc is a minimum-image
    # cell of (A, C0) for some reference atom A -- i.e. W[gc][:, C0] is non-zero.
    for c, (C0, gc) in enumerate(pad.pad_atom_of):
        w_cell = weights.get(gc)
        if w_cell is None:
            continue
        omega_c = w_cell[:, C0]                       # omega_{A, c} over ref atoms A
        if not np.any(omega_c):
            continue
        # I_{mu nu' c} = <mu | (-Z_{C0}/r_c) | nu'> on the padded basis. NB:
        # vibe-qc's compute_nuclear already returns the signed (attractive,
        # negative) nuclear-attraction matrix, so eq. (12)'s leading minus is
        # already included -- accumulate with += (do not negate again).
        z = int(pad.pad_Z[c])
        nuc_mol = Molecule([Atom(z, pad.pad_positions[c].tolist())], 0, 1 if z % 2 == 0 else 2)
        i_pad = compute_nuclear(pad.basis, nuc_mol)
        nucl_w = 0.5 * (omega_c[:, None] + omega_c[None, :])   # (omega_{A c}+omega_{B c})/2
        for g_nu, pair_w in weights.items():
            atom_w = pair_w * nucl_w                  # omega_{mu nu'} * (...)
            if not np.any(atom_w):
                continue
            block = i_pad[np.ix_(rows, pad.cell_to_cols[g_nu])]
            out += atom_w[aoA, aoB] * block
    return 0.5 * (out + out.T)


def ccm_eri(ccm, pad: "PaddedCluster | None" = None) -> np.ndarray:
    """CCM-weighted, image-folded effective ERI tensor ``(ab|cd)`` (Mulliken).

    Returns ``eff[a,b,c,d]`` over reference (home) AOs such that the standard
    contractions give the cyclic Coulomb and exchange matrices (eqs 23-24):
    ``J_{muν} = S_{rs} D_{rs} eff[mu,ν,r,s]`` and
    ``K_{muν} = S_{rs} D_{rs} eff[mu,s,r,ν]``.

    The four-center weight is eq. (18):
    ``w_{abcd} = w_{ab} . (w_{ac} + w_{bc})/2 . w_{cd}``, with each ``w`` a
    two-center pair weight from :meth:`CCMSystem.cell_weight_matrices`. The bra
    pair (a,b) is kept at the home cell and the ket pair (c,d) is folded over
    its images -- ``c`` over the union WSSC (cell ``g_c``) and ``d`` relative to
    ``c`` (cell ``g_d``, so ``d`` sits at ``g_c+g_d``, the ``±2t`` range). The
    result is then **bra-ket symmetrised** (``1/2(eff + eff^{(cd|ab)})``): folding
    only one pair counts each wrapped interaction once, and the symmetrisation
    splits it 1/2 between the bra-folded and ket-folded representations (the
    boundary-sharing that avoids the double-count) and makes the Fock Hermitian.

    Accuracy: reproduces the supercell-model HF energy (Peintinger PhD thesis
    Tab. 8.3 / JCC 2014 Tab. 2) to ~1e-5 Ha/atom, converging with cluster size;
    generalises to any 3-D lattice and arbitrary orbitals (the weight is
    per-atom-pair, applied to whole AO blocks). NOT yet bit-exact: it is not
    perfectly cyclically invariant, so small high-symmetry clusters show minor
    orbital-degeneracy splitting. Dense ``n_ref_ao**4`` -- small-cluster path.
    The bit-exact version remains open.
    """
    from vibeqc import compute_eri

    if pad is None:
        pad = build_padded_cluster(ccm, eri_cells(ccm))
    _check_padded_eri_size(pad, "ccm_eri")
    eri_pad = np.asarray(compute_eri(pad.basis), dtype=float)
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    n = ccm.nbf
    home = np.arange(n)
    keys = list(weights.keys())
    # Pair weights expanded to AO pairs: wexp[g][x, y] = w of (AO y at cell g) rel AO x.
    wexp = {g: weights[g][ao[:, None], ao[None, :]] for g in keys}

    # Bra pair at home; ket pair (c at g_c in the union WSSC, d at g_c+g_d) folded.
    cols_b = pad.cell_to_cols[(0, 0, 0)]                  # home bra columns
    wab = wexp[(0, 0, 0)]                                 # w_{ab} home pair weight
    eff = np.zeros((n, n, n, n))
    if True:
        for gc in keys:
            wc = wexp[gc]                                # (x, c): use [a,c] and [b,c]
            cross = 0.5 * (wc[:, None, :] + wc[None, :, :])   # (a, b, c)
            if not np.any(cross):
                continue
            cols_c = pad.cell_to_cols[gc]
            for gd in keys:
                wcd = wexp[gd]                           # (c, d)
                if not np.any(wcd):
                    continue
                gcd = (gc[0] + gd[0], gc[1] + gd[1], gc[2] + gd[2])
                cols_d = pad.cell_to_cols.get(gcd)
                if cols_d is None:
                    continue
                w4 = (wab[:, :, None, None]
                      * cross[:, :, :, None]
                      * wcd[None, None, :, :])           # (a, b, c, d)
                if not np.any(w4):
                    continue
                sub = eri_pad[np.ix_(home, cols_b, cols_c, cols_d)]
                eff += w4 * sub
    # Bra-ket symmetrise (eq. Fock = dE/dP): splits each wrapped interaction 1/2
    # between the bra-folded and ket-folded representations and makes J/K Hermitian.
    return 0.5 * (eff + np.transpose(eff, (2, 3, 0, 1)))


# ---------------------------------------------------------------------------
# The 2012 program / P2014 construction ("p2014"), with the bra partner imaged
# ---------------------------------------------------------------------------
#
# ``ccm_eri`` ("union12") keeps the bra partner in the HOME copy of the cluster.
# P2014 Eq. 19 and the original program (``~/gitlab/aiccm``,
# ``integrals/eri_functions.pyx::calcERIsUnion12``: ``for J, wIJ in
# zip(bfng[I].WSC_bfng, ...)`` with ``j = J % d``) sum the bra partner over the
# closed Wigner-Seitz supercell of the first function, images included, and take
# the bridge weight relative to that IMAGED partner. In "union12" a pair whose
# minimum image is a wrapped one enters only as a ket pair through the bra-ket
# symmetrisation, at half weight, and the interaction of two wrapped pairs is
# absent. Found by the paper chat, 2026-09-17 (paper repository,
# ``theory/analysis-2026-09-17/``); the builders below are the faithful form.

#: Nuclear-attraction variants of the "p2014" selector.
#:
#: * ``"program-union"`` -- the 2012 program's ``'union'`` path
#:   (``operators/one_functions.pyx::attractionWSCIJ``): nucleus weight
#:   ``(omega_IC + omega_JC)/2`` with ``omega_JC`` relative to the IMAGED
#:   partner J, and NO pair weight ``omega_IJ`` (every image of J inside the
#:   closed Wigner-Seitz cell of I enters with weight one). This is what the
#:   published P2014 numbers were produced with (two-cell value -0.540663).
#: * ``"printed-eq13"`` -- P2014 Eq. 13 as printed: the same nucleus weight
#:   times the pair weight ``omega_IJ`` (two-cell value -0.540361; the two
#:   agree to 1e-8 from four cells on).
P2014_NUCLEAR_VARIANTS = ("program-union", "printed-eq13")


def _cell_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _cell_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _p2014_anchor_cells(ccm) -> list:
    """Cells an anchor (ket function K, or a nucleus) can sit in with a non-zero
    union weight: inside the closed Wigner-Seitz supercell of a home function I
    (cell ``g_c`` a weight cell) or of its imaged partner J at ``g_b`` (cell
    ``g_c - g_b`` a weight cell). Pruned to the pairs that carry weight."""
    weights = ccm.cell_weight_matrices()
    keys = list(weights.keys())
    nz = {g: weights[g] > 0.0 for g in keys}
    anchors = set()
    for gb in keys:
        a_has_b = nz[gb].any(axis=1)          # home atoms A with a partner in g_b
        b_has_a = nz[gb].any(axis=0)          # partner atoms B (in g_b) that are used
        if not a_has_b.any():
            continue
        for gc in set(keys) | {_cell_add(gb, g) for g in keys}:
            hit = False
            m1 = nz.get(gc)
            if m1 is not None and (a_has_b[:, None] & m1).any():
                hit = True
            m2 = nz.get(_cell_sub(gc, gb))
            if not hit and m2 is not None and (b_has_a[:, None] & m2).any():
                hit = True
            if hit:
                anchors.add(gc)
    return sorted(anchors)


def p2014_nuclear_cells(ccm) -> list:
    """Padded cells for the "p2014" nuclear attraction: imaged partners + nuclei."""
    keys = set(ccm.cell_weight_matrices().keys())
    return sorted({(0, 0, 0)} | keys | set(_p2014_anchor_cells(ccm)))


def p2014_eri_cells(ccm) -> list:
    """Padded cells for the "p2014" four-centre fold: imaged partner ``g_b``,
    anchor ``g_c`` (union of the two Wigner-Seitz supercells) and the ket partner
    at ``g_c + g_d`` -- the +-2 supercell translations of the 2012 program."""
    keys = list(ccm.cell_weight_matrices().keys())
    anchors = _p2014_anchor_cells(ccm)
    cells = {(0, 0, 0)} | set(keys) | set(anchors)
    for gc in anchors:
        for gd in keys:
            cells.add(_cell_add(gc, gd))
    return sorted(cells)


def ccm_nuclear_p2014(ccm, pad: "PaddedCluster | None" = None, *,
                      variant: str = "program-union",
                      symmetrize: bool = True) -> np.ndarray:
    """Nuclear attraction of the 2012 program / P2014 with the partner imaged.

    For a home function I and every image J (cell ``g_b``) inside the closed
    Wigner-Seitz supercell of I, the nucleus image ``C = (C0, g_c)`` carries
    ``(omega_IC + omega_JC)/2`` with ``omega_JC = W[g_c - g_b][J0, C0]`` taken
    relative to the IMAGED partner (:func:`ccm_nuclear` takes it relative to the
    home image of the partner). ``variant`` selects
    :data:`P2014_NUCLEAR_VARIANTS`. ``symmetrize=False`` returns the row-wise
    assembled matrix so a test can measure its asymmetry.
    """
    if variant not in P2014_NUCLEAR_VARIANTS:
        raise ValueError(
            f"ccm_nuclear_p2014: unknown variant {variant!r}; choose from "
            f"{P2014_NUCLEAR_VARIANTS}")
    if pad is None:
        pad = build_padded_cluster(ccm, p2014_nuclear_cells(ccm))
    weights = ccm.cell_weight_matrices()
    keys = list(weights.keys())
    ao = ccm.ao_atom
    rows = np.arange(pad.n_ref_ao)
    aoA, aoB = ao[:, None], ao[None, :]
    n_at = ccm.n_atoms
    zero = np.zeros(n_at)

    out = np.zeros((ccm.nbf, ccm.nbf))
    for c, (C0, gc) in enumerate(pad.pad_atom_of):
        w_i = weights.get(gc)
        om_i = w_i[:, C0] if w_i is not None else zero        # omega_{I, C}
        terms = []
        for gb in keys:
            w_j = weights.get(_cell_sub(gc, gb))
            om_j = w_j[:, C0] if w_j is not None else zero    # omega_{J@g_b, C}
            pair = weights[gb] if variant == "printed-eq13" else (weights[gb] > 0.0)
            atom_w = pair * (0.5 * (om_i[:, None] + om_j[None, :]))
            if np.any(atom_w):
                terms.append((gb, atom_w))
        if not terms:
            continue
        z = int(pad.pad_Z[c])
        nuc_mol = Molecule([Atom(z, pad.pad_positions[c].tolist())], 0, 1 if z % 2 == 0 else 2)
        i_pad = compute_nuclear(pad.basis, nuc_mol)          # signed (attractive)
        for gb, atom_w in terms:
            cols = pad.cell_to_cols.get(gb)
            if cols is None:
                raise ValueError(
                    f"ccm_nuclear_p2014: padded cluster lacks partner cell {gb}.")
            out += atom_w[aoA, aoB] * i_pad[np.ix_(rows, cols)]
    return 0.5 * (out + out.T) if symmetrize else out


def ccm_eri_p2014(ccm, pad: "PaddedCluster | None" = None, *,
                  ket_symmetrize: bool = True) -> np.ndarray:
    """Effective four-centre tensor of the 2012 program / P2014 Eqs. 18-19.

    For a home function I: J over every image (cell ``g_b``) in the closed
    Wigner-Seitz supercell of I, weight ``omega_IJ = W[g_b][I, J0]``; K over
    every image (cell ``g_c``) with the bridge weight
    ``(omega_IK + omega_JK)/2``, ``omega_IK = W[g_c][I, K0]`` and
    ``omega_JK = W[g_c - g_b][J0, K0]`` relative to the IMAGED J; L over every
    image in the cell of K (cell ``g_c + g_d``), weight ``omega_KL = W[g_d]``.
    ``omega_IJ (omega_IK + omega_JK)/2 omega_KL (IJ|KL)`` is accumulated into
    ``V[i, j, k, l]`` with the indices folded onto the cluster.

    The 2012 program contracts ``J_ij = V[i,j,k,l] D_kl`` and
    ``K_ij = (V[i,k,j,l] + V[i,l,k,j]) D_kl / 2``, which for a symmetric density
    is the standard contraction of the KET-SYMMETRISED tensor
    ``Vbar = (V_ijkl + V_ijlk)/2``; ``Vbar`` is eight-fold symmetric (paper
    repository, analysis note of 2026-09-17), so it is what this function
    returns by default and what ``run_ccm_rhf(method="p2014")`` contracts with
    the usual ``"mnrs,rs"`` / ``"msrn,rs"`` einsums. ``ket_symmetrize=False``
    returns the raw ``V`` (first-pair symmetric only) for structure tests.

    Dense ``n_pad**4`` validation path; the integral-direct C++ kernel
    ``"p2014-direct"`` (:func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf_scalable`)
    is the production form.
    """
    from vibeqc import compute_eri

    if pad is None:
        pad = build_padded_cluster(ccm, p2014_eri_cells(ccm))
    _check_padded_eri_size(pad, "ccm_eri_p2014")
    eri_pad = np.asarray(compute_eri(pad.basis), dtype=float)
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    n = ccm.nbf
    home = np.arange(n)
    keys = list(weights.keys())
    wexp = {g: weights[g][ao[:, None], ao[None, :]] for g in keys}
    anchors = _p2014_anchor_cells(ccm)

    eff = np.zeros((n, n, n, n))
    for gb in keys:
        wab = wexp[gb]                                   # (a, b): b imaged at g_b
        if not np.any(wab):
            continue
        cols_b = pad.cell_to_cols.get(gb)
        if cols_b is None:
            raise ValueError(f"ccm_eri_p2014: padded cluster lacks cell {gb}.")
        for gc in anchors:
            w_ac = wexp.get(gc)                          # omega_IK, relative to I
            w_bc = wexp.get(_cell_sub(gc, gb))           # omega_JK, relative to imaged J
            if w_ac is None and w_bc is None:
                continue
            cross = np.zeros((n, n, n))
            if w_ac is not None:
                cross += 0.5 * w_ac[:, None, :]
            if w_bc is not None:
                cross += 0.5 * w_bc[None, :, :]
            bra = wab[:, :, None] * cross                # (a, b, c)
            if not np.any(bra):
                continue
            cols_c = pad.cell_to_cols.get(gc)
            if cols_c is None:
                raise ValueError(f"ccm_eri_p2014: padded cluster lacks cell {gc}.")
            for gd in keys:
                wcd = wexp[gd]                           # (c, d): d at g_c + g_d
                w4 = bra[:, :, :, None] * wcd[None, None, :, :]
                if not np.any(w4):
                    continue
                cols_d = pad.cell_to_cols.get(_cell_add(gc, gd))
                if cols_d is None:
                    raise ValueError(
                        "ccm_eri_p2014: padded cluster lacks cell "
                        f"{_cell_add(gc, gd)} (weighted quartets would be dropped).")
                eff += w4 * eri_pad[np.ix_(home, cols_b, cols_c, cols_d)]
    if not ket_symmetrize:
        return eff
    return 0.5 * (eff + np.transpose(eff, (0, 1, 3, 2)))


def ccm_hcore_p2014(ccm, *, nuclear_attraction: str = "program-union"):
    """``(h, T, V)`` for the "p2014" selector: lattice-fold kinetic energy plus
    :func:`ccm_nuclear_p2014` in the named variant."""
    from .integrals import ccm_kinetic

    t = ccm_kinetic(ccm)
    v = ccm_nuclear_p2014(ccm, variant=nuclear_attraction)
    return t + v, t, v


def ccm_eri_symmetric(ccm, pad: "PaddedCluster | None" = None) -> np.ndarray:
    """Fully bra-ket-symmetric effective ERI tensor -- the ``"aiccm2026dev-a"`` four-center.

    The Sec.13 (``AICCM_ALGORITHM.md``) symmetric replacement for the historical
    eq-18 :func:`ccm_eri`. It contracts identically --
    ``J = S_{rs} D_{rs} eff[mu,ν,r,s]``, ``K = S_{rs} D_{rs} eff[mu,s,r,ν]`` -- but
    the effective tensor is **exactly 8-fold permutationally symmetric for any
    lattice** (machine precision in 1-D, 2-D hexagonal, and 2-D oblique), whereas
    eq 18 breaks r<->s / mu<->ν symmetry (≈1.6e-3 in 1-D, ≈2.1e-2 in 2-D). A genuine
    symmetric ERI set is what a single variational RHF/MP2 energy functional
    requires; eq 18's asymmetry is energetically invisible only in high-symmetry
    1-D (it diverges by ~1.4 mHa/atom on a 2-D hex lattice).

    Two changes vs eq 18, **both** required for exact symmetry:

    * **Symmetric bridge** ``1/4(w_mur + w_νr + w_mus + w_νs)`` -- treats the two ket
      functions r, s identically (eq 18's ``1/2(w_mur + w_νr)`` singles out the ket
      anchor r).
    * **Independent minimum-image fold** -- r (cell ``g_c``) and s (cell ``g_e``)
      are each the minimum image of the *home* bra, and the ket-pair weight
      ``w_rs`` is taken at the relative cell ``g_e - g_c``. eq 18 instead *chains*
      s to r (cell ``g_c + g_d``), which is asymmetric under r<->s even with a
      symmetric weight.

    Summing over all ``(g_c, g_e)`` is manifestly r<->s invariant (the swap maps
    ``(g_c, g_e) -> (g_e, g_c)`` with an identical weight); the home bra gives
    mu<->ν; the closing transpose gives bra<->ket -- so ``eff`` obeys the full
    molecular ERI permutation group. It reduces to the molecular ERIs in the
    interior (all w -> 1 => bridge 1/4.4 = 1). All weights come from
    :meth:`CCMSystem.cell_weight_matrices`, so it inherits the WSC geometry and
    is lattice-general (the reference cell is the WSC).

    Dense ``n_ref_ao**4`` -- small-cluster / validation path. For the same
    union-and-weight Γ-CCM construction at scale, use the screened lattice-sum
    builder :func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf_scalable`.
    """
    from vibeqc import compute_eri

    if pad is None:
        pad = build_padded_cluster(ccm, eri_cells(ccm))
    _check_padded_eri_size(pad, "ccm_eri_symmetric")
    eri_pad = np.asarray(compute_eri(pad.basis), dtype=float)
    weights = ccm.cell_weight_matrices()
    ao = ccm.ao_atom
    n = ccm.nbf
    home = np.arange(n)
    keys = list(weights.keys())
    wexp = {g: weights[g][ao[:, None], ao[None, :]] for g in keys}

    cols_b = pad.cell_to_cols[(0, 0, 0)]                  # home bra columns
    wab = wexp[(0, 0, 0)]                                 # w_muν (home bra pair)
    eff = np.zeros((n, n, n, n))
    for gc in keys:
        wc = wexp[gc]                                     # (x,c): w of r=c@g_c rel x
        cols_c = pad.cell_to_cols[gc]
        for ge in keys:
            grel = (ge[0] - gc[0], ge[1] - gc[1], ge[2] - gc[2])
            w_pair = weights.get(grel)
            if w_pair is None:
                continue                                 # r,s not a minimum-image pair
            wcd = w_pair[ao[:, None], ao[None, :]]        # (c,d): w_rs at g_e - g_c
            cols_d = pad.cell_to_cols.get(ge)
            if cols_d is None:
                continue
            wd = wexp[ge]                                 # (x,d): w of s=d@g_e rel x
            bridge = 0.25 * (
                wc[:, None, :, None] + wc[None, :, :, None]      # w_mur, w_νr
                + wd[:, None, None, :] + wd[None, :, None, :]    # w_mus, w_νs
            )
            w4 = wab[:, :, None, None] * bridge * wcd[None, None, :, :]
            if not np.any(w4):
                continue
            eff += w4 * eri_pad[np.ix_(home, cols_b, cols_c, cols_d)]
    # Bra-ket symmetrise (the remaining generator); mu<->ν and r<->s already hold.
    return 0.5 * (eff + np.transpose(eff, (2, 3, 0, 1)))


def ccm_nuclear_repulsion(ccm) -> float:
    """CCM nuclear repulsion per reference cell ``V_nn^CCM`` (WSSC-weighted).

    ``0.5 S_{A,B} S_{g in minimg(A,B)} w_{AB}[g] Z_A Z_B / |r_A - (r_B + R_g)|``,
    excluding the on-site term ``(A=B, g=0)``. The 1/2 removes the ordered
    double-count of each translationally-equivalent pair.
    """
    weights = ccm.cell_weight_matrices()
    pos = ccm.atom_positions
    Z = np.array([int(a.Z) for a in ccm.supercell.atoms], dtype=float)
    Ac = ccm.cluster_vectors
    e_nn = 0.0
    for g, w_atom in weights.items():
        Rg = np.asarray(g, dtype=float) @ Ac
        is_home = g == (0, 0, 0)
        for A in range(ccm.n_atoms):
            for B in range(ccm.n_atoms):
                w = w_atom[A, B]
                if w == 0.0 or (is_home and A == B):
                    continue
                d = np.linalg.norm(pos[A] - (pos[B] + Rg))
                e_nn += 0.5 * w * Z[A] * Z[B] / d
    return float(e_nn)


def ccm_hcore(ccm):
    """CCM core Hamiltonian ``h^CCM = T^CCM + V^CCM`` and its parts.

    Returns ``(h, T, V)``. ``T^CCM`` uses the efficient lattice fold; ``V^CCM``
    uses the padded path. Both apply the same WSSC weighting.
    """
    from .integrals import ccm_kinetic

    t = ccm_kinetic(ccm)
    v = ccm_nuclear(ccm)
    return t + v, t, v
