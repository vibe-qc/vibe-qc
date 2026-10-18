"""Density-qualified symmetry reduction of the physical direct J/K domain.

Dovesi, Int. J. Quantum Chem. 29, 1755 (1986),
doi:10.1002/qua.560290608, Eqs. (31)-(36), (43): an atom-pair rotation
includes the relative atom-image shift, and Fock reconstruction requires a
covariant density and internal summation domain. Here the finite domain is
our physical pair/midpoint criterion, not CRYSTAL's overlap thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _vibeqc_core as core
from .bipole_symmetry_fock import (
    FockSymmetryMapping, representative_cell_indices, representative_shell_masks,
    symmetrize_fock_blocks,
)
from .pair_resolved_truncation import pair_resolved_domain, domain_triples
from .symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix
from .symmetry_integrals import symmorphic_operations
from .symmetry_lattice import lattice_to_cartesian_rotation
from .symmetry_lattice_c import (
    _atom_ao_slices, _cell_tuple, identify_atom_pair_orbits, operator_triple_actions,
)


def _clear_physical_cutoff_boundaries(system, cells, pair, interaction):
    """Conservative admission of the finite native pair/midpoint predicate.

    Mathematical distance invariance is insufficient at a floating-point
    cutoff tie. Do not change the native predicate or union its domain.
    Refuse reduction if any possible pair/midpoint distance is ambiguous.
    Twice a midpoint difference is ra+rb-rc-rd+Lh; reducing its offset modulo
    L makes this test independent of the atoms' chosen image labels.
    """
    lattice = np.asarray(system.lattice)
    atoms = np.asarray([a.xyz for a in system.unit_cell])
    translations = np.asarray([c.r_cart for c in cells])
    scale = 1. + np.max(np.abs(atoms)) + np.max(np.abs(translations)) + pair + interaction
    slack = 256 * np.finfo(float).eps * scale**2
    for a in atoms:
        for b in atoms:
            delta = translations + b - a
            if np.any(np.abs(np.einsum("ij,ij->i", delta, delta) - pair**2) <= slack):
                return False
    sums = {tuple(np.linalg.solve(lattice, a + b)) for a in atoms for b in atoms}
    if len(sums) > 256:
        return False
    offsets = set()
    for a in sums:
        for b in sums:
            fractional = np.asarray(a) - b
            # Only periodic directions can be absorbed into a lattice label.
            fractional[:system.dim] -= np.rint(fractional[:system.dim])
            offsets.add(tuple(lattice @ fractional))
            if len(offsets) > 256:
                return False
    radius = 2 * interaction + max(np.linalg.norm(o) for o in offsets) + np.sqrt(slack)
    # Bound work for this optional optimization. Full J/K remains available
    # when the geometry certificate would itself be too expensive.
    estimate = len(cells) * max(1., radius / (pair + interaction))**system.dim
    if estimate * len(offsets) > 2_000_000:
        return False
    vectors = np.asarray([c.r_cart for c in core.direct_lattice_cells(system, radius)])
    if len(vectors) * len(offsets) > 2_000_000:
        return False
    for offset in offsets:
        delta = vectors - offset
        if np.any(np.abs(np.einsum("ij,ij->i", delta, delta) - (2 * interaction)**2) <= 4 * slack):
            return False
    return True


@dataclass
class _PhysicalPlan:
    pair_cutoff: float
    interaction_cutoff: float
    cells: list
    checks: list
    representatives: list
    masks: list
    reduced_builds: int = 0
    full_builds: int = 0
    residual_builds: int = 0

    def _density_blocks(self, density):
        # Export native blocks once: each property access copies the vector.
        by_key = {_cell_tuple(c.index): b for c, b in zip(density.cells, density.blocks)}
        zero = np.zeros((density.nbf, density.nbf))
        return np.asarray([by_key.get(_cell_tuple(c.index), zero) for c in self.cells])

    def _blocks_are_covariant(self, blocks):
        if not np.all(np.isfinite(blocks)):
            return False
        tolerance = 64 * np.finfo(float).eps
        for src, dst, a, b, pa, pb, da, db in self.checks:
            expected = da @ blocks[src, a, b] @ db.T
            if not np.allclose(blocks[dst, pa, pb], expected,
                               atol=tolerance, rtol=tolerance):
                return False
        return True

    def density_is_covariant(self, density):
        return self._blocks_are_covariant(self._density_blocks(density))

    def _split_nearly_covariant_density(self, mapping, density):
        """Return a group-invariant density and its explicit remainder.

        Average inverse atom-pair actions on each representative, including
        its little group, then reconstruct (Dovesi 1986, Eqs. 31-36, 43).
        Linearity gives JK(D) = JK(Dsym) + JK(D-Dsym). The second term is
        evaluated on the FULL output domain, never projected or discarded.
        The size test only selects a cheaper execution path; it is not a
        density or Fock error tolerance. Large broken states keep one build.
        """
        blocks = self._density_blocks(density)
        if not np.all(np.isfinite(blocks)):
            return None
        projected = blocks.copy()
        for src, _, a, b, *_ in self.checks:
            projected[src, a, b] = 0.
        for src, dst, a, b, pa, pb, da, db in self.checks:
            projected[src, a, b] += da.T @ blocks[dst, pa, pb] @ db / len(mapping.sym_ops)
        projected_list = list(projected)
        symmetrize_fock_blocks(projected_list, mapping)
        projected = np.asarray(projected_list)
        delta = blocks - projected
        if (np.max(np.abs(delta)) > 1e-8 * max(1., np.max(np.abs(blocks)))
                or not self._blocks_are_covariant(projected)):
            return None
        symmetric = core.make_lattice_matrix_set(density.nbf, self.cells, list(projected))
        # Keep caller-only cells as well. They cannot contribute on this
        # certified physical support, but do not silently drop density data
        # at the native boundary. Missing mapping cells are zero-extended.
        keys = {_cell_tuple(c.index): i for i, c in enumerate(self.cells)}
        residual_cells = list(self.cells)
        residual_blocks = list(delta)
        for cell, block in zip(density.cells, density.blocks):
            if _cell_tuple(cell.index) not in keys:
                residual_cells.append(cell)
                residual_blocks.append(block)
        residual = core.make_lattice_matrix_set(density.nbf, residual_cells, residual_blocks)
        # Give the small arm a smaller screening budget. Equal allocation
        # needlessly changes the dominant contraction's screened term set;
        # allocation proportional to density size would make the residual
        # just as expensive. The square root interpolates between these
        # choices without increasing the sum of the two thresholds.
        fraction = min(.5, np.sqrt(np.max(np.abs(delta)) / np.max(np.abs(blocks))))
        return symmetric, residual, fraction

    def build(self, mapping, opts, density, omega, *, internal_extent_bohr,
              output_cell_farming_task_kind, output_cell_farming_strategy,
              workspace, compute_coulomb, compute_exchange):
        from types import SimpleNamespace
        from .pbc_bipole_fock import _build_jk_domains_output_cells_mpi, _sr_padded_lat_opts

        interaction = (float(internal_extent_bohr) if internal_extent_bohr is not None
                       else float(opts.eri_interaction_cutoff_bohr or opts.cutoff_bohr))
        if (not opts.pair_complete_1e or float(opts.cutoff_bohr) != self.pair_cutoff
                or interaction != self.interaction_cutoff):
            raise ValueError("physical Fock symmetry plan does not match the J/K domain")
        opts_used = _sr_padded_lat_opts(opts, interaction)
        reduced = self.density_is_covariant(density)
        # Without screening the full residual costs as much as the original
        # density, so splitting would only add work.
        split = (self._split_nearly_covariant_density(mapping, density)
                 if not reduced and opts_used.schwarz_threshold > 0. else None)
        if split is not None:
            # Two linear contractions share the original per-quartet
            # screening budget; the remainder receives the smaller share.
            opts_used.schwarz_threshold *= 1. - split[2]
            reduced = True
        jk = _build_jk_domains_output_cells_mpi(
            mapping.basis, mapping.system, opts_used, split[0] if split else density, self.cells,
            self.representatives if reduced else list(range(len(self.cells))),
            self.masks if reduced else [], omega=float(omega),
            task_kind=output_cell_farming_task_kind,
            strategy=output_cell_farming_strategy, workspace=workspace,
            compute_coulomb=compute_coulomb, compute_exchange=compute_exchange,
        )
        if not reduced:
            self.full_builds += 1
            # The native J-only API may still return a zero K container.
            # Keep the accumulator's J-only contract stable when successive
            # density increments switch between reduced and full builds.
            return SimpleNamespace(
                J=jk.J, K=jk.K if compute_exchange else None,
                output_cell_farming_execution=getattr(jk, "output_cell_farming_execution", None),
            )
        self.reduced_builds += 1

        def reconstruct(matrix):
            blocks = list(matrix.blocks)
            symmetrize_fock_blocks(blocks, mapping)
            return core.make_lattice_matrix_set(mapping.basis.nbasis, self.cells, blocks)

        j, k = reconstruct(jk.J), reconstruct(jk.K) if compute_exchange else None
        execution = getattr(jk, "output_cell_farming_execution", None)
        if split is not None:
            residual_opts = _sr_padded_lat_opts(opts, interaction)
            residual_opts.schwarz_threshold *= split[2]
            correction = _build_jk_domains_output_cells_mpi(
                mapping.basis, mapping.system, residual_opts, split[1], self.cells,
                list(range(len(self.cells))), [], omega=float(omega),
                task_kind=output_cell_farming_task_kind,
                strategy=output_cell_farming_strategy, workspace=workspace,
                compute_coulomb=compute_coulomb, compute_exchange=compute_exchange,
            )
            def add(left, right):
                return core.make_lattice_matrix_set(
                    density.nbf, self.cells, [a + b for a, b in zip(left.blocks, right.blocks)],
                )
            j = add(j, correction.J)
            k = add(k, correction.K) if compute_exchange else None
            execution = getattr(correction, "output_cell_farming_execution", None)
            self.residual_builds += 1
        return SimpleNamespace(J=j, K=k, output_cell_farming_execution=execution)


def physical_fock_mapping(system, basis, opts, *, interaction_extent=None):
    """Map the complete physical output domain, without changing truncation.

    If pair lengths are <= R and their midpoints are <= L apart, every
    crossed pair is <= R+L. Thus exchange output AND contracted density
    pairs need R+L, whereas J output only needs R. Native physical cells
    enclose every admitted center; they remain the internal/output template.
    AUTO admission is a caller policy. Each actual build checks its density,
    including stabilizers, and falls back to the full operator if necessary.
    """
    operations = getattr(getattr(system, "symmetry", None), "operations", None)
    ops = symmorphic_operations(operations) if operations else []
    if len(ops) < 2:
        return None
    shells = list(basis.shells())
    atoms = list(system.unit_cell)
    indices = [int(s.atom_index) for s in shells]
    if (indices != sorted(indices) or set(indices) != set(range(len(atoms)))
            or any(not s.pure and s.l != 0 for s in shells)
            or any(not np.array_equal(s.origin, atoms[s.atom_index].xyz) for s in shells)):
        # Existing atom-block reconstruction requires contiguous pure shells
        # on their assigned nuclei. Do not infer this from a basis name.
        return None
    pair = float(opts.cutoff_bohr)
    interaction = (float(interaction_extent) if interaction_extent is not None
                   else float(opts.eri_interaction_cutoff_bohr or pair))
    from .lattice_screening import physical_eri_cells

    cells = physical_eri_cells(basis, system, opts, interaction)
    if not _clear_physical_cutoff_boundaries(system, cells, pair, interaction):
        return None
    slices = _atom_ao_slices(basis)
    try:
        rotations, permutations, shifts, _ = operator_triple_actions(ops, system)
    except ValueError:
        # A model attached with a loose structure tolerance need not supply
        # a usable atom action. AUTO must retain the original full operator.
        return None
    actions = []
    lattice = np.asarray(system.lattice)
    # Certify the actual geometry and radial action before constructing
    # group-closed domains from a possibly approximate symmetry model.
    for op, rotation, perm, shift in zip(ops, rotations, permutations, shifts):
        cart = lattice_to_cartesian_rotation(rotation, lattice)
        ap = atom_permutation_under_op(system, cart, lattice @ np.asarray(op.translation))
        coordinates = np.asarray([a.xyz for a in atoms])
        residual = (coordinates @ cart.T + lattice @ np.asarray(op.translation)
                    - coordinates[perm] + shift @ lattice.T)
        tolerance = 128 * np.finfo(float).eps
        if (np.max(np.abs(cart.T @ cart - np.eye(3))) > tolerance
                or np.max(np.abs(residual)) > tolerance * (1 + np.max(np.abs(coordinates)))):
            return None
        try:
            action = build_ao_permutation_matrix(basis, cart, ap)
        except ValueError:
            # A geometric atom permutation need not preserve custom radial
            # contractions. Such a basis still has a valid full operator.
            return None
        atom_actions = [action[slices[int(perm[a])], slices[a]].copy()
                        for a in range(len(atoms))]
        actions.append(atom_actions)
    domain = pair_resolved_domain(system, pair + interaction, operations=ops)
    positions = {_cell_tuple(c.index): i for i, c in enumerate(cells)}
    if not set(domain.cell_keys()) <= positions.keys():
        return None
    orbits = identify_atom_pair_orbits(
        cells, ops, system, require_closed=True, triples=domain_triples(domain),
    )
    mapping = FockSymmetryMapping(orbits, ops, cells, system, basis)
    checks = []
    # Checking only representative -> member edges misses little-group
    # constraints. Apply every operation, including stabilizers.
    for rotation, perm, shift, atom_actions in zip(rotations, permutations, shifts, actions):
        groups = {}
        for orbit in orbits.orbits:
            a, b, h = orbit.representative
            target = tuple(rotation @ np.asarray(h) + shift[a] - shift[b])
            src, dst = groups.setdefault((a, b), ([], []))
            src.append(positions[h])
            dst.append(positions[target])
        for (a, b), (src, dst) in groups.items():
            pa, pb = int(perm[a]), int(perm[b])
            checks.append((np.asarray(src), np.asarray(dst), slices[a], slices[b],
                           slices[pa], slices[pb], atom_actions[a], atom_actions[b]))
    reps = representative_cell_indices(mapping)
    mapping.physical_plan = _PhysicalPlan(
        pair, interaction, cells, checks, reps, representative_shell_masks(mapping, reps),
    )
    return mapping
