"""Phase SYM3b: symmetry enforcement for BIPOLE real-space Fock blocks.

Post-processes the real-space two-electron Fock blocks to enforce
space-group symmetry: every (source-atom, dest-atom, cell) sub-block is
replaced by the correctly rotated sub-block of its orbit representative.

This is a correctness-and-convergence aid -- it does not reduce compute
but enforces exact Fock symmetry, improving SCF stability.

The transformation is the *atom-pair-resolved* one (SYM2c,
``symmetry_lattice_c``): a space-group operation maps atom ``a`` of the
home cell into its image atom in some *other* cell (the per-atom lattice
shift ``s_a``), so the cell index of a transformed (a, b, g) block is
``R.g + s_b - s_a`` -- per atom pair, not one uniform ``g -> R.g`` for the
whole matrix. A naive whole-matrix cell-orbit version of this module
(2026-06-09) was only correct for single-atom-at-origin cells: on MgO
primitive (O off the rotation centre) it scattered Mg-O cross blocks
into wrong cells and shifted the converged RHF energy by ~0.5 Ha.
Regression: ``tests/test_bipole_symmetry_fock.py``
(MgO energy-invariance) -- see handovers/HANDOVER_BIPOLE_PRODUCTION.md Sec.0a
2026-06-10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeCell,
    LatticeSumOptions,
    LatticeMatrixSet,
    PeriodicSystem,
    build_jk_2e_real_space_output_subset_masked,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from .pair_resolved_truncation import (
    PairResolvedDomain,
    atom_pair_shell_masks,
    domain_triples,
    pair_resolved_domain,
)
from .symmetry_integrals import symmorphic_operations
from .symmetry_lattice_c import (
    _cell_tuple,
    compress_lattice_matrix_set_c,
    identify_atom_pair_orbits,
    reconstruct_lattice_matrix_set_c,
)

__all__ = [
    "FockSymmetryMapping",
    "symmetrize_fock_blocks",
    "cell_orbit_mapping",
    "pair_resolved_fock_mapping",
    "representative_cell_indices",
    "representative_shell_masks",
    "build_jk_reduced_symmetrized",
    "build_jk_pair_resolved",
]


class _BlockSetView:
    """Duck-typed stand-in for a ``LatticeMatrixSet`` (``.blocks`` +
    ``.cells``) so ``compress_lattice_matrix_set_c`` can read plain
    Python lists without building a native object."""

    def __init__(self, blocks: List[np.ndarray], cells: Sequence[LatticeCell]):
        self.blocks = blocks
        self.cells = cells


@dataclass
class FockSymmetryMapping:
    """The SYM3b Fock-symmetry mapping (formerly an opaque 5-tuple).

    ``cells`` is the OUTPUT template the mapping's orbits live on: the
    radial ``direct_lattice_cells`` list for the legacy mapping
    (:func:`cell_orbit_mapping`), the pair-resolved cell list for
    :func:`pair_resolved_fock_mapping` (``domain`` is then set and the
    derived fields carry the M1 ``build_jk_2e_real_space_domains``
    inputs, cached once per run).
    """

    orbits: object
    sym_ops: list
    cells: list
    system: PeriodicSystem
    basis: BasisSet
    # Pair-resolved mode (M2) -- all None for the legacy radial mapping.
    domain: Optional[PairResolvedDomain] = None
    # Internal summation cell list (union of the radial cutoff ball and
    # the pair-resolved output cells) + positions of `cells` in it.
    cells_internal: Optional[list] = field(default=None, repr=False)
    output_positions: Optional[List[int]] = field(default=None, repr=False)
    # Qualifying-pair shell masks for every output cell (parallel to
    # `cells`) -- the FULL pair-resolved build's output_shell_masks.
    output_masks: Optional[List[np.ndarray]] = field(default=None, repr=False)
    # M3: the pair-resolved DENSITY support domain (criterion radius
    # 2x cutoff -- the historical density-list convention). The SCF stores
    # the physical Bloch density on these cells; the direct SR builder masks
    # a private copy to the qualifying pairs so its tensor domain is
    # group-invariant without contaminating one-electron/XC/LR contractions.
    density_domain: Optional[PairResolvedDomain] = field(
        default=None, repr=False
    )

    @property
    def pair_resolved(self) -> bool:
        return self.domain is not None


def cell_orbit_mapping(
    system: PeriodicSystem,
    basis: BasisSet,
    cells: Sequence[LatticeCell],
) -> Optional[FockSymmetryMapping]:
    """Build the LEGACY (radial-domain) atom-pair orbit mapping.

    Returns ``None`` if the system carries no symmetry, the operator set
    has no symmorphic subset, or the orbit partition fails. The radial
    triple space is NOT group-closed for off-centre atoms
    (``require_closed=False`` silently skips out-of-list partners --
    the 2026-07-12 SYM3b root cause); prefer
    :func:`pair_resolved_fock_mapping` wherever the caller controls the
    build domains.
    """
    ops = getattr(getattr(system, "symmetry", None), "operations", None)
    if not ops:
        return None
    sym_ops = symmorphic_operations(ops)
    if not sym_ops:
        return None

    try:
        orbits = identify_atom_pair_orbits(
            list(cells),
            sym_ops,
            system,
            require_closed=False,
        )
    except (ValueError, RuntimeError):
        return None

    if orbits.n_orbits == 0:
        return None

    return FockSymmetryMapping(
        orbits=orbits,
        sym_ops=sym_ops,
        cells=list(cells),
        system=system,
        basis=basis,
    )


def pair_resolved_fock_mapping(
    system: PeriodicSystem,
    basis: BasisSet,
    lat_opts_2e: LatticeSumOptions,
) -> Optional[FockSymmetryMapping]:
    """Build the pair-resolved (group-invariant) Fock-symmetry mapping.

    M2 of the pair-resolved truncation workstream
    (HANDOVER_BIPOLE_PRODUCTION.md Sec. 0a 2026-07-12): the output
    triple set is ``{(a, b, h): |r_b + L.h - r_a| <= cutoff}``, which
    the triple action preserves, so the orbit partition runs with
    ``require_closed=True`` -- no silently skipped partners, every
    member reconstructs from its representative. Derived build inputs
    (internal cell list, output positions, qualifying-pair masks,
    extended overlap blocks for the background fold) are computed once
    and cached on the mapping.

    Returns ``None`` when the system carries no usable symmetry (same
    contract as :func:`cell_orbit_mapping`).
    """
    ops = getattr(getattr(system, "symmetry", None), "operations", None)
    if not ops:
        return None
    sym_ops = symmorphic_operations(ops)
    if not sym_ops:
        return None

    cutoff = float(lat_opts_2e.cutoff_bohr)
    domain = pair_resolved_domain(system, cutoff, operations=sym_ops)
    triples = domain_triples(domain)
    orbits = identify_atom_pair_orbits(
        list(domain.cells),
        sym_ops,
        system,
        require_closed=True,
        triples=triples,
    )
    if orbits.n_orbits == 0:
        return None

    # Internal summation domain. The pair-resolved cell set always
    # CONTAINS the radial cutoff ball (every radial cell hosts its
    # qualifying diagonal (a, a) pairs), so the union of the historical
    # traversal domain with the output cells is just the domain cells,
    # and the output subset is the identity. Interaction-resolved
    # internal selection is M4b; this is the minimal structurally-valid
    # choice (a mild traversal extension: |L.h| reaches
    # cutoff + pair_span instead of cutoff).
    radial_keys = {
        _cell_tuple(c.index) for c in direct_lattice_cells(system, cutoff)
    }
    domain_keys = set(domain.cell_keys())
    if not radial_keys <= domain_keys:
        raise RuntimeError(
            "pair_resolved_fock_mapping: the pair-resolved domain lost "
            f"radial cells ({sorted(radial_keys - domain_keys)[:4]} ...) "
            "-- diagonal pairs must keep every radial cell. Internal "
            "invariant violation, not a user error."
        )
    cells_internal = list(domain.cells)
    output_positions = list(range(len(cells_internal)))
    output_masks = atom_pair_shell_masks(basis, domain.pairs_by_cell)

    # M3: pair-resolved density support at the historical 2x-cutoff
    # criterion radius. Same operator set, so the density triple set
    # is orbit-closed by construction too.
    density_domain = pair_resolved_domain(
        system, 2.0 * cutoff, operations=sym_ops
    )

    return FockSymmetryMapping(
        orbits=orbits,
        sym_ops=sym_ops,
        cells=list(domain.cells),
        system=system,
        basis=basis,
        domain=domain,
        cells_internal=cells_internal,
        output_positions=output_positions,
        output_masks=output_masks,
        density_domain=density_domain,
    )


def symmetrize_fock_blocks(
    fock_blocks: List[np.ndarray],
    mapping: FockSymmetryMapping,
    *,
    cells: Optional[Sequence[LatticeCell]] = None,
) -> None:
    """Enforce space-group symmetry on Fock blocks in-place.

    Each atom-pair orbit's members are replaced by the correctly
    rotated representative sub-block,
    ``F(a_mem, b_mem, h_mem) = D_a(R) . F(a_rep, b_rep, h_rep) . D_b(R)ᵀ``
    with the on-atom Wigner-D blocks of the full AO rotation matrix --
    the same transformation the symmetry-reduced S/T integral path uses
    (pinned to machine precision against the explicit build).

    Sub-blocks not covered by any orbit (only possible when the mapping
    carries a pair-resolved triple restriction, or when ``cells`` is
    wider than the mapping template) are left untouched.

    Parameters
    ----------
    fock_blocks : list of (nbf, nbf) ndarray
        Fock matrix blocks per cell (mutated in-place).
    mapping : FockSymmetryMapping from cell_orbit_mapping() /
        pair_resolved_fock_mapping().
    cells : optional
        The template ``fock_blocks`` actually lives on, when it is not
        ``mapping.cells``. Association is BY CELL KEY -- positional
        association across different radial lists is wrong because
        ``direct_lattice_cells`` tie order is not stable across radii
        (the pre-M2 exact-J + enforcement combination silently
        mis-associated blocks this way). Every mapping cell must be
        present in ``cells``; raises otherwise.
    """
    cells_t = list(cells) if cells is not None else mapping.cells
    if cells is None and len(fock_blocks) != len(cells_t):
        raise ValueError(
            f"symmetrize_fock_blocks: {len(fock_blocks)} blocks for "
            f"{len(cells_t)} mapping cells -- when the caller template "
            f"differs from mapping.cells, pass it via cells= (positional "
            f"association across templates scrambles blocks)."
        )
    if cells is not None:
        keys_t = {_cell_tuple(c.index) for c in cells_t}
        missing = [
            k
            for k in (_cell_tuple(c.index) for c in mapping.cells)
            if k not in keys_t
        ]
        if missing:
            raise ValueError(
                f"symmetrize_fock_blocks: the caller template is missing "
                f"mapping cell(s) {missing[:4]}"
                f"{'...' if len(missing) > 4 else ''} -- cannot enforce "
                f"symmetry on a template that does not cover the mapping."
            )
    # Fresh writable copies: the scatter mutates entries in place, and
    # caller blocks may be read-only pybind views. (The pre-M2 version
    # also replaced every list entry with a fresh array.)
    for i in range(len(fock_blocks)):
        fock_blocks[i] = np.array(fock_blocks[i], dtype=float, copy=True)
    view = _BlockSetView(fock_blocks, cells_t)
    reps = compress_lattice_matrix_set_c(view, mapping.orbits, mapping.basis)
    reconstruct_lattice_matrix_set_c(
        reps,
        mapping.orbits,
        mapping.basis,
        mapping.system,
        mapping.sym_ops,
        cells=cells_t,
        into_blocks=fock_blocks,
    )


def representative_cell_indices(mapping: FockSymmetryMapping) -> List[int]:
    """Output (bra) cell indices that hold an atom-pair orbit representative.

    These are the only cells the SYM3b reduced Fock build must compute; every
    other cell's block is recovered from its orbit rep by point-group rotation
    (:func:`symmetrize_fock_blocks`). Indices are positions in the mapping's
    cell list (``mapping.cells``): the radial ``direct_lattice_cells`` list
    for the legacy mapping, the pair-resolved cell list under
    :func:`pair_resolved_fock_mapping`.
    """
    orbits, cells = mapping.orbits, mapping.cells
    tup2idx = {_cell_tuple(c.index): i for i, c in enumerate(cells)}
    return sorted({tup2idx[_cell_tuple(o.representative[2])] for o in orbits.orbits})


def _atom_shell_indices(basis: BasisSet) -> dict:
    """Map atom index -> list of shell indices, in libint shell order (the same
    order the C++ JK builder's ``s1``/``s2`` loops use)."""
    by_atom: dict = {}
    for s_idx, sh in enumerate(basis.shells()):
        by_atom.setdefault(int(sh.atom_index), []).append(s_idx)
    return by_atom


def representative_shell_masks(
    mapping: FockSymmetryMapping,
    rep_cell_indices: List[int],
) -> List[np.ndarray]:
    """Per-rep-cell shell-pair masks selecting the orbit-representative
    atom-pair sub-blocks for the masked C++ build.

    Returns one ``(n_shells*n_shells,)`` uint8 array per entry of
    ``rep_cell_indices`` (same order), with ``mask[s1*n_shells + s2] = 1`` for
    every output shell pair ``(s1 on the home-cell source atom, s2 on the
    dest atom)`` that belongs to an atom-pair-orbit representative whose
    representative cell is that output cell. Every other output shell pair is
    skipped by the build and recovered by :func:`symmetrize_fock_blocks`.
    """
    orbits, cells, basis = mapping.orbits, mapping.cells, mapping.basis
    shells_by_atom = _atom_shell_indices(basis)
    n_shells = sum(len(v) for v in shells_by_atom.values())
    tup2idx = {_cell_tuple(c.index): i for i, c in enumerate(cells)}
    pairs_by_cell: dict = {}
    for orb in orbits.orbits:
        a, b, h = orb.representative
        ci = tup2idx[_cell_tuple(h)]
        pairs_by_cell.setdefault(ci, []).append((int(a), int(b)))
    masks: List[np.ndarray] = []
    for ci in rep_cell_indices:
        mask = np.zeros(n_shells * n_shells, dtype=np.uint8)
        for a, b in pairs_by_cell.get(ci, []):
            for s1 in shells_by_atom.get(a, []):
                base = s1 * n_shells
                for s2 in shells_by_atom.get(b, []):
                    mask[base + s2] = 1
        masks.append(mask)
    return masks


def _padded_internal_cells(
    mapping: FockSymmetryMapping,
    system: PeriodicSystem,
    internal_extent_bohr: float,
):
    """Radial internal cell list at the given extent (never narrower than
    the mapping's own reach) + positions of ``mapping.cells`` in it."""
    reach = 0.0
    for c in mapping.cells:
        reach = max(reach, float(np.linalg.norm(np.asarray(c.r_cart))))
    radius = max(float(internal_extent_bohr), reach + 1e-6)
    cells = list(direct_lattice_cells(system, radius))
    pos = {_cell_tuple(c.index): i for i, c in enumerate(cells)}
    out_positions = [pos[_cell_tuple(c.index)] for c in mapping.cells]
    return cells, out_positions, radius


def build_jk_reduced_symmetrized(
    basis: BasisSet,
    system: PeriodicSystem,
    opts: LatticeSumOptions,
    density: LatticeMatrixSet,
    omega: float,
    mapping: FockSymmetryMapping,
    rep_cell_indices: List[int],
    *,
    internal_extent_bohr: Optional[float] = None,
    output_cell_farming_task_kind: Optional[str] = None,
    output_cell_farming_strategy: str = "cyclic",
):
    """SYM3b: point-group-reduced direct-ERI build of J(g) and K(g).

    Builds J/K only for the orbit-representative atom-pair sub-blocks (the full
    internal lattice sum is kept, so each emitted sub-block is exact), then
    reconstructs the complete symmetric J/K by rotating each rep into its
    orbit. The shell-pair mask is the finer reduction over the whole-cell
    subset: ``symmetrize_fock_blocks`` only reads the representative atom-pair
    sub-blocks (``compress_lattice_matrix_set_c``) and overwrites every
    orbit-covered sub-block on reconstruction, so building only those
    sub-blocks is **bit-identical** to ``symmetrize_fock_blocks`` applied to a
    full build on the same domains (validated to 0.0 on MgO/STO-3G c8/c12).
    Returns the same ``JKLatticeMatrixSets``-shaped object as the full
    builder, so callers (incl. the incremental-Fock ΔD path, since
    reconstruction is linear) are drop-in.

    Under a pair-resolved mapping (M2) the output template is
    ``mapping.cells`` (the pair-resolved cell list) and the build runs
    through the M1 full-domain binding; sub-blocks outside the
    pair-resolved triple set are exact zeros (they belong to no orbit
    and are never built).

    ``internal_extent_bohr`` (M4b: the sr_image_extent + SYM3b-reduce
    composition) pads the internal (c_lam, c_sig) summation ball to the
    given absolute radius -- the M4a ket-image pad expressed through
    the reduced build, for either mapping kind. Set
    ``opts.sr_range_screening`` to keep the padded traversal affordable
    (charge-pair Schwarz screening with angular and contraction
    envelopes; the threshold applies per quartet, not to the total error).
    """
    masks = representative_shell_masks(mapping, rep_cell_indices)
    if mapping.pair_resolved or internal_extent_bohr is not None:
        if internal_extent_bohr is not None:
            cells_internal, output_positions, radius = (
                _padded_internal_cells(mapping, system, internal_extent_bohr)
            )
            from .pbc_bipole_fock import _sr_padded_lat_opts

            opts_used = _sr_padded_lat_opts(opts, radius)
        elif mapping.pair_resolved:
            cells_internal = mapping.cells_internal
            output_positions = mapping.output_positions
            opts_used = opts
        rep_pos_internal = [output_positions[i] for i in rep_cell_indices]
        from .pbc_bipole_fock import _build_jk_domains_output_cells_mpi

        jk = _build_jk_domains_output_cells_mpi(
            basis,
            system,
            opts_used,
            density,
            cells_internal,
            rep_pos_internal,
            masks,
            omega=float(omega),
            task_kind=output_cell_farming_task_kind,
            strategy=output_cell_farming_strategy,
        )
        j_blocks = [
            np.array(jk.J.blocks[p], dtype=float, copy=True)
            for p in output_positions
        ]
        k_blocks = [
            np.array(jk.K.blocks[p], dtype=float, copy=True)
            for p in output_positions
        ]
        symmetrize_fock_blocks(j_blocks, mapping)
        symmetrize_fock_blocks(k_blocks, mapping)
        from types import SimpleNamespace

        nbf = int(basis.nbasis)
        return SimpleNamespace(
            J=make_lattice_matrix_set(nbf, mapping.cells, j_blocks),
            K=make_lattice_matrix_set(nbf, mapping.cells, k_blocks),
            output_cell_farming_execution=getattr(
                jk, "output_cell_farming_execution", None
            ),
        )
    if output_cell_farming_task_kind is not None:
        from ._vibeqc_core import direct_lattice_cells
        from .pbc_bipole_fock import _build_jk_domains_output_cells_mpi

        cells_internal = list(
            direct_lattice_cells(system, float(opts.cutoff_bohr))
        )
        jk = _build_jk_domains_output_cells_mpi(
            basis,
            system,
            opts,
            density,
            cells_internal,
            rep_cell_indices,
            masks,
            omega=float(omega),
            task_kind=output_cell_farming_task_kind,
            strategy=output_cell_farming_strategy,
        )
        n = len(jk.J.cells)
        j_blocks = [np.asarray(jk.J.blocks[c], dtype=float) for c in range(n)]
        k_blocks = [np.asarray(jk.K.blocks[c], dtype=float) for c in range(n)]
        symmetrize_fock_blocks(j_blocks, mapping)
        symmetrize_fock_blocks(k_blocks, mapping)
        for c in range(n):
            jk.J.set_block(c, j_blocks[c])
            jk.K.set_block(c, k_blocks[c])
        return jk
    jk = build_jk_2e_real_space_output_subset_masked(
        basis, system, opts, density, rep_cell_indices, masks, float(omega)
    )
    n = len(jk.J.cells)
    j_blocks = [np.asarray(jk.J.blocks[c], dtype=float) for c in range(n)]
    k_blocks = [np.asarray(jk.K.blocks[c], dtype=float) for c in range(n)]
    symmetrize_fock_blocks(j_blocks, mapping)
    symmetrize_fock_blocks(k_blocks, mapping)
    for c in range(n):
        jk.J.set_block(c, j_blocks[c])
        jk.K.set_block(c, k_blocks[c])
    return jk


def build_jk_pair_resolved(
    basis: BasisSet,
    system: PeriodicSystem,
    opts: LatticeSumOptions,
    density: LatticeMatrixSet,
    omega: float,
    mapping: FockSymmetryMapping,
    *,
    compute_exchange: bool = True,
    internal_extent_bohr: Optional[float] = None,
    output_cell_farming_task_kind: Optional[str] = None,
    output_cell_farming_strategy: str = "cyclic",
):
    """FULL (non-reduced) J/K build on the pair-resolved domains (M2).

    Every qualifying (a, b, h) output sub-block is built explicitly
    (per-cell shell-pair masks; non-qualifying sub-blocks are exact
    zeros); the output template is ``mapping.cells``. Used by the
    enforcement-only SYM3b mode (``use_fock_symmetry`` without the
    reduced build) -- symmetry enforcement stays with the caller.
    ``compute_exchange=False`` skips the K contraction (pure-functional
    J-only builds; ``K`` comes back ``None``).
    ``internal_extent_bohr`` (M4b) pads the internal (c_lam, c_sig)
    summation ball to the given absolute radius (the M4a ket-image pad
    under pair-resolved output domains); pair with
    ``opts.sr_range_screening`` to keep the padded traversal
    affordable.
    """
    if not mapping.pair_resolved:
        raise ValueError(
            "build_jk_pair_resolved: mapping carries no pair-resolved "
            "domain (legacy cell_orbit_mapping?)"
        )
    if internal_extent_bohr is not None:
        cells_internal, output_positions, radius = _padded_internal_cells(
            mapping, system, internal_extent_bohr
        )
        from .pbc_bipole_fock import _sr_padded_lat_opts

        opts_used = _sr_padded_lat_opts(opts, radius)
    else:
        cells_internal = mapping.cells_internal
        output_positions = mapping.output_positions
        opts_used = opts
    from .pbc_bipole_fock import _build_jk_domains_output_cells_mpi

    jk = _build_jk_domains_output_cells_mpi(
        basis,
        system,
        opts_used,
        density,
        cells_internal,
        output_positions,
        mapping.output_masks,
        omega=float(omega),
        compute_exchange=compute_exchange,
        task_kind=output_cell_farming_task_kind,
        strategy=output_cell_farming_strategy,
    )
    from types import SimpleNamespace

    nbf = int(basis.nbasis)
    j_blocks = [
        np.array(jk.J.blocks[p], dtype=float, copy=True)
        for p in output_positions
    ]
    J = make_lattice_matrix_set(nbf, mapping.cells, j_blocks)
    K = None
    if compute_exchange:
        k_blocks = [
            np.array(jk.K.blocks[p], dtype=float, copy=True)
            for p in output_positions
        ]
        K = make_lattice_matrix_set(nbf, mapping.cells, k_blocks)
    return SimpleNamespace(
        J=J,
        K=K,
        output_cell_farming_execution=getattr(
            jk, "output_cell_farming_execution", None
        ),
    )
