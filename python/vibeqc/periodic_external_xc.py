"""Shared atom-grid adapter for periodic full-grid XC providers.

GPW and GAPW use uniform grids and density decompositions for their Hartree
builders.  A full-grid XC provider has a different contract: it must see the
complete AO density on one atom-centred quadrature before constructing any
non-local features.  This module owns that translation so the GPW/GAPW SCF
drivers do not duplicate the density-folding and Bloch-projection rules.

Primary-source seam: Pöschel et al., 2026, arXiv:2608.19033, Sec. II B-C.
Their GauXC integration bypasses the auxiliary plane-wave mapping and, for
GAPW, evaluates the complete AO density once rather than combining separate
smooth, hard, and soft model evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import _vibeqc_core as _core
from .periodic_grid import build_periodic_becke_grid
from .periodic_k_density import real_space_density_from_per_k_density


def _require_zero_temperature_external_xc(
    temperature: float,
    *,
    where: str,
) -> None:
    """Keep external periodic XC on its state-consistent T=0 envelope.

    The periodic SCF drivers retain an accepted density when their iteration
    budget is exhausted, but their finite-temperature diagnostics are
    currently produced by a subsequent canonical refill. Until those two
    states are unified, exposing ``E - T S`` for a full-grid provider would
    combine energy and entropy from different densities.
    """

    value = float(temperature)
    if not np.isfinite(value):
        raise ValueError(
            f"{where}: smearing_temperature must be finite; got {value!r}"
        )
    if value < 0.0:
        raise ValueError(
            f"{where}: smearing_temperature must be >= 0; got {value!r}"
        )
    if value > 0.0:
        raise NotImplementedError(
            f"{where}: periodic external XC is zero-temperature only. "
            "Set smearing_temperature=0 until finite-iteration Mermin "
            "energy, entropy, occupations, and returned density share one "
            "self-consistent state."
        )


def _cell_index(cell) -> tuple[int, int, int]:
    values = np.asarray(cell.index, dtype=int).reshape(3)
    return (int(values[0]), int(values[1]), int(values[2]))


_DIFFERENCE_BITMAP_MAX_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class _LatticeDifferenceBitmap:
    origin: tuple[int, int, int]
    shape: tuple[int, int, int]
    bits: bytes
    count: int

    def __len__(self) -> int:
        return self.count

    def __contains__(self, index: tuple[int, int, int]) -> bool:
        x, y, z = (index[i] - self.origin[i] for i in range(3))
        nx, ny, nz = self.shape
        # Check each axis before flattening: an out-of-range coordinate must
        # not alias a valid cell by carrying into the next radix.
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            return False
        offset = (x * ny + y) * nz + z
        return bool(self.bits[offset >> 3] & (1 << (offset & 7)))


def _lattice_difference_set(
    ket_indices: Sequence[tuple[int, int, int]],
    bra_indices: Sequence[tuple[int, int, int]],
) -> frozenset[tuple[int, int, int]] | _LatticeDifferenceBitmap:
    """Exact finite ``ket - bra`` support, without a Python loop per pair.

    Encode ket offsets and reversed bra offsets in radices large enough for
    their sums. Integer shifts then form the Minkowski sum with no coordinate
    carries, periodic wrapping, or filling of holes. This is only a support
    representation; the full AO density and its adjoint remain unchanged
    (Pöschel et al., arXiv:2608.19033, Sec. III, Eqs. 7-9).
    """
    ket = set(ket_indices)
    bra = set(bra_indices)
    if not ket or not bra:
        return frozenset()
    ket_min = tuple(min(index[i] for index in ket) for i in range(3))
    ket_max = tuple(max(index[i] for index in ket) for i in range(3))
    bra_min = tuple(min(index[i] for index in bra) for i in range(3))
    bra_max = tuple(max(index[i] for index in bra) for i in range(3))
    shape = tuple(
        ket_max[i] - ket_min[i] + bra_max[i] - bra_min[i] + 1
        for i in range(3)
    )
    nx, ny, nz = shape
    volume = nx * ny * nz
    # Bound scratch independently of lattice skew and displaced atoms. A
    # very sparse bounding box also favors explicit tuple differences.
    if (
        volume > 8 * _DIFFERENCE_BITMAP_MAX_BYTES
        or volume > 64 * len(ket) * len(bra)
    ):
        return frozenset(
            (s[0] - a[0], s[1] - a[1], s[2] - a[2])
            for a in bra for s in ket
        )

    seed = bytearray((volume + 7) // 8)
    for x, y, z in ket:
        offset = (
            ((x - ket_min[0]) * ny + y - ket_min[1]) * nz + z - ket_min[2]
        )
        seed[offset >> 3] |= 1 << (offset & 7)
    ket_bits = int.from_bytes(seed, "little")
    del seed
    required = 0
    for x, y, z in bra:
        shift = (
            ((bra_max[0] - x) * ny + bra_max[1] - y) * nz + bra_max[2] - z
        )
        required |= ket_bits << shift
    return _LatticeDifferenceBitmap(
        origin=tuple(ket_min[i] - bra_max[i] for i in range(3)),
        shape=shape,
        bits=required.to_bytes((volume + 7) // 8, "little"),
        count=required.bit_count(),
    )


def periodic_xc_difference_closed_cells(
    system,
    cutoff_bohr: float,
    image_radius_bohr: float,
) -> tuple[object, ...]:
    """Build the exact cell support for periodic XC density and its adjoint.

    The native XC kernel admits a translated AO cell whenever *any* home-cell
    atom pair is within the requested physical radius.  That is an
    atom-pair-centred domain, not a ball on lattice-cell origins.  Enumerate
    the exact ket and bra image sets with that same criterion, then retain
    every lattice difference ``g = s - a`` needed by the variational
    potential.  This remains invariant when an atom is redescribed in a
    neighbouring unit cell.
    """

    cutoff = float(cutoff_bohr)
    image_radius = float(image_radius_bohr)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(
            "periodic external XC cutoff_bohr must be finite and > 0"
        )
    if not np.isfinite(image_radius) or image_radius <= 0.0:
        raise ValueError(
            "periodic external XC image_radius_bohr must be finite and > 0"
        )
    bra_radius = min(image_radius, cutoff)

    from .pair_resolved_truncation import (
        atom_pair_span,
        pair_resolved_domain,
    )

    ket_cells = pair_resolved_domain(system, cutoff).cells
    bra_cells = pair_resolved_domain(system, bra_radius).cells
    required = _lattice_difference_set(
        [_cell_index(cell) for cell in ket_cells],
        [_cell_index(cell) for cell in bra_cells],
    )
    span = float(atom_pair_span(system))
    difference_bound = cutoff + bra_radius + 2.0 * span + 1.0e-9
    candidates = tuple(
        _core.direct_lattice_cells(system, difference_bound)
    )
    selected = []
    available = set()
    for cell in candidates:
        index = _cell_index(cell)
        if index in required:
            selected.append(cell)
            available.add(index)
    missing = len(required) - len(available)
    if missing:
        raise RuntimeError(
            "periodic external XC could not materialize its exact "
            f"difference domain ({missing} lattice cells missing)"
        )
    cells = tuple(selected)
    if not cells:
        raise RuntimeError("periodic external XC lattice domain is empty")
    return cells


def _external_density_domain_kwargs() -> dict[str, object]:
    """Select the explicit periodic-lattice density contract."""

    density_domain = getattr(_core, "PeriodicXCDensityDomain", None)
    if density_domain is None:
        raise RuntimeError(
            "periodic external XC requires a rebuilt native extension with "
            "PeriodicXCDensityDomain support"
        )
    return {"density_domain": density_domain.PERIODIC_LATTICE}


def _copy_lattice_options(source) -> object:
    """Return a detached ``LatticeSumOptions`` copy."""

    target = _core.LatticeSumOptions()
    for name in (
        "cutoff_bohr",
        "nuclear_cutoff_bohr",
        "becke_image_radius_bohr",
        "coulomb_method",
        "slab_ewald_alpha",
        "screening_overlap_threshold",
        "screening_exchange_threshold",
        "schwarz_threshold",
        "schwarz_threshold_forces",
        "sr_range_screening",
        "sr_sparse_traversal",
        "pair_complete_1e",
        "eri_interaction_cutoff_bohr",
    ):
        if hasattr(source, name) and hasattr(target, name):
            setattr(target, name, getattr(source, name))
    return target


def _grid_options_for_external(
    functional,
    supplied,
    *,
    where: str = "PeriodicExternalXC.prepare",
) -> object:
    """Resolve a provider-compatible atom-grid option object.

    A direct low-level GPW/GAPW call has no high-level runner available to
    apply the provider capability.  In that case construct the requested
    profile.  An explicitly supplied option object is preserved, but its
    profile is checked here so a mismatch fails before grid construction and
    the first SCF iteration.
    """

    capabilities = functional.external_capabilities or {}
    required = str(capabilities.get("required_grid_profile", ""))
    if supplied is None:
        options = _core.GridOptions()
        if required:
            options.atomic_grid_profile = required
        return options

    options = supplied
    if required:
        value = getattr(options, "atomic_grid_profile", None)
        name = str(getattr(value, "name", value)).strip().lower()
        aliases = {
            "generic": "generic",
            "atomicgridprofile.generic": "generic",
            "pyscflevel3": "pyscf-level3",
            "pyscf-level3": "pyscf-level3",
            "atomicgridprofile.pyscflevel3": "pyscf-level3",
        }
        actual = aliases.get(name.replace("_", "-"), name.replace("_", "-"))
        if actual != required:
            raise ValueError(
                f"{where}: external XC functional '{functional.name}' "
                f"requires grid profile '{required}', but the supplied grid "
                f"selects '{actual}'."
            )
    return options


def _reject_reduced_kmesh(kmesh, system) -> None:
    """Require a complete, uniform native Monkhorst-Pack finite torus.

    The inverse Bloch fold used by a full-grid provider is not a generic
    weighted quadrature: its real-space blocks are the discrete Fourier
    transform of one complete Born-von Karman character mesh.  An IBZ,
    explicit weighted list, or band path therefore cannot be accepted even
    when it happens to contain more than one point.
    """

    mapping = list(getattr(kmesh, "ir_mapping", ()) or ())
    kpoints = list(getattr(kmesh, "kpoints", ()) or ())
    mesh = tuple(int(value) for value in getattr(kmesh, "mesh", (1, 1, 1)))
    if len(mesh) != 3 or any(value < 1 for value in mesh):
        raise NotImplementedError(
            "periodic external XC requires a complete Monkhorst-Pack mesh "
            "with three positive mesh dimensions"
        )
    structured_size = int(np.prod(mesh, dtype=int))
    # Native reduced meshes keep one mapping entry per full-BZ point but only
    # store irreducible representatives.  An identity mapping is harmless;
    # any other mapping proves that representative weights would be folded as
    # though they were a complete character mesh.
    mapping_is_identity = not mapping or (
        len(mapping) == structured_size
        and np.array_equal(
            np.asarray(mapping, dtype=int), np.arange(structured_size)
        )
    )
    weights = np.asarray(getattr(kmesh, "weights", ()), dtype=float)
    uniform_weights = (
        weights.shape == (structured_size,)
        and np.all(np.isfinite(weights))
        and np.allclose(
            weights,
            1.0 / float(structured_size),
            rtol=0.0,
            atol=1.0e-14,
        )
    )
    if (
        len(kpoints) != structured_size
        or not mapping_is_identity
        or not uniform_weights
    ):
        raise NotImplementedError(
            "periodic external XC requires a complete, uniformly weighted "
            "Monkhorst-Pack mesh before the real-space density fold; "
            "symmetry-reduced representatives and explicit weighted lists "
            "need AO rotation or a separate quadrature contract"
        )

    shift = tuple(
        int(value) for value in getattr(kmesh, "is_shift", (0, 0, 0))
    )
    if len(shift) != 3 or any(value not in (0, 1) for value in shift):
        raise NotImplementedError(
            "periodic external XC requires native Monkhorst-Pack shift "
            "metadata (three entries in {0, 1})"
        )
    expected = _core.monkhorst_pack(
        system,
        list(mesh),
        list(shift),
        False,
    )
    actual_points = np.asarray(kpoints, dtype=float)
    expected_points = np.asarray(expected.kpoints, dtype=float)
    if actual_points.shape != expected_points.shape or not np.allclose(
        actual_points,
        expected_points,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise NotImplementedError(
            "periodic external XC requires the native ordered "
            "Monkhorst-Pack character mesh; explicit or reordered k-point "
            "lists are not a proven finite torus"
        )


@dataclass(frozen=True)
class PeriodicExternalXC:
    """Prepared full-grid periodic-XC evaluation context."""

    basis: object
    system: object
    functional: object
    grid: object
    lattice_options: object
    cells: tuple[object, ...]

    @classmethod
    def prepare(
        cls,
        basis,
        system,
        functional,
        *,
        grid_options=None,
        image_radius_bohr: float = 10.0,
        lattice_options=None,
        cells: Optional[Sequence[object]] = None,
    ) -> "PeriodicExternalXC":
        """Build the atom grid and lattice domain once for an SCF run."""

        if not bool(getattr(functional, "is_external", False)):
            raise ValueError(
                "PeriodicExternalXC requires a full-grid external functional"
            )
        image_radius = float(image_radius_bohr)
        if not np.isfinite(image_radius) or image_radius <= 0.0:
            raise ValueError(
                "external XC image_radius_bohr must be finite and > 0"
            )
        resolved_grid_options = _grid_options_for_external(
            functional, grid_options
        )

        if lattice_options is None:
            options = _core.LatticeSumOptions()
            options.cutoff_bohr = 25.0
        else:
            options = _copy_lattice_options(lattice_options)
        cutoff = float(options.cutoff_bohr)
        density_radius = float(options.becke_image_radius_bohr)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "periodic external XC cutoff_bohr must be finite and > 0"
            )
        if not np.isfinite(density_radius):
            raise ValueError("periodic external XC density image radius must be finite")
        # Becke, JCP 88, 2547 (1988), Sec. II, Eqs. (3)-(7),
        # DOI: 10.1063/1.454033: the partition weights the integrand; it
        # must not truncate the AO density supplied to a full-grid provider.
        # Match the native density support convention independently of the
        # partition radius: unset means full ket reach, positive is capped.
        density_radius = min(density_radius, cutoff) if density_radius > 0.0 else cutoff
        options.becke_image_radius_bohr = density_radius

        # The XC kernel evaluates AO images a,s within ``cutoff_bohr`` and
        # accumulates the derivative into the difference block g=s-a.  Its
        # output/density domain must therefore contain every exact difference
        # of the atom-pair-complete bra and ket image sets. A radial cell-
        # origin ball (even at ``2 * cutoff``) is insufficient for a
        # multi-atom cell because an intra-cell offset can bring an image AO
        # inside the physical cutoff.
        if cells is None:
            resolved_cells = periodic_xc_difference_closed_cells(
                system,
                float(options.cutoff_bohr),
                density_radius,
            )
        else:
            resolved_cells = tuple(cells)
            present = {_cell_index(cell) for cell in resolved_cells}
            required = {
                _cell_index(cell)
                for cell in periodic_xc_difference_closed_cells(
                    system,
                    float(options.cutoff_bohr),
                    density_radius,
                )
            }
            missing = required.difference(present)
            if missing:
                raise ValueError(
                    "periodic external XC cells are not closed over active "
                    "AO-image differences under the atom-pair-complete "
                    f"periodic domain (missing {len(missing)} cells)"
                )
        if not resolved_cells:
            raise RuntimeError("periodic external XC lattice domain is empty")

        atom_grid = build_periodic_becke_grid(
            system,
            grid_options=resolved_grid_options,
            image_radius_bohr=image_radius,
        )
        return cls(
            basis=basis,
            system=system,
            functional=functional,
            grid=atom_grid,
            lattice_options=options,
            cells=resolved_cells,
        )

    def _gamma_density(self, density: np.ndarray):
        """Construct the finite-torus inverse fold of one Gamma density."""

        matrix = np.asarray(density, dtype=float)
        expected = (int(self.basis.nbasis), int(self.basis.nbasis))
        if matrix.shape != expected:
            raise ValueError(
                f"external XC Gamma density has shape {matrix.shape}; "
                f"expected {expected}"
            )
        matrix = 0.5 * (matrix + matrix.T)
        return _core.make_lattice_matrix_set(
            int(self.basis.nbasis),
            list(self.cells),
            [matrix.copy() for _ in self.cells],
        )

    def build_gamma(self, density: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate external XC for a periodic Gamma-point RKS density."""

        contribution = _core.build_xc_periodic(
            self.basis,
            self.system,
            self.grid,
            self.functional,
            self._gamma_density(density),
            self.lattice_options,
            **_external_density_domain_kwargs(),
        )
        potential = np.asarray(
            _core.bloch_sum(contribution.V_xc, np.zeros(3)), dtype=complex
        )
        potential = 0.5 * (potential + potential.conj().T)
        max_imaginary = float(np.max(np.abs(np.imag(potential)), initial=0.0))
        if max_imaginary > 1.0e-10:
            raise RuntimeError(
                "periodic external XC Gamma potential is not real "
                f"(max imaginary component {max_imaginary:.3e})"
            )
        return float(contribution.e_xc), np.asarray(np.real(potential))

    def build_gamma_uks(
        self,
        density_alpha: np.ndarray,
        density_beta: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate external XC for periodic Gamma-point spin densities."""

        contribution = _core.build_xc_periodic_uks(
            self.basis,
            self.system,
            self.grid,
            self.functional,
            self._gamma_density(density_alpha),
            self._gamma_density(density_beta),
            self.lattice_options,
            **_external_density_domain_kwargs(),
        )
        potentials = []
        for spin, lattice_potential in (
            ("alpha", contribution.V_alpha),
            ("beta", contribution.V_beta),
        ):
            potential = np.asarray(
                _core.bloch_sum(lattice_potential, np.zeros(3)), dtype=complex
            )
            potential = 0.5 * (potential + potential.conj().T)
            max_imaginary = float(
                np.max(np.abs(np.imag(potential)), initial=0.0)
            )
            if max_imaginary > 1.0e-10:
                raise RuntimeError(
                    "periodic external XC Gamma "
                    f"{spin} potential is not real "
                    f"(max imaginary component {max_imaginary:.3e})"
                )
            potentials.append(np.asarray(np.real(potential)))
        return float(contribution.e_xc), potentials[0], potentials[1]

    def build_kpoints(
        self,
        density_k: Sequence[np.ndarray],
        kmesh,
    ) -> tuple[float, list[np.ndarray]]:
        """Evaluate XC from a full-mesh density and return ``V_xc(k)``."""

        _reject_reduced_kmesh(kmesh, self.system)
        blocks = list(density_k)
        if len(blocks) != len(getattr(kmesh, "kpoints", ())):
            raise ValueError(
                "periodic external XC density/k-point size mismatch: "
                f"{len(blocks)} density blocks for "
                f"{len(getattr(kmesh, 'kpoints', ()))} k-points"
            )
        density_real = real_space_density_from_per_k_density(
            blocks,
            kmesh,
            self.cells,
        )
        contribution = _core.build_xc_periodic(
            self.basis,
            self.system,
            self.grid,
            self.functional,
            density_real,
            self.lattice_options,
            **_external_density_domain_kwargs(),
        )
        potentials: list[np.ndarray] = []
        for kpoint in kmesh.kpoints:
            block = np.asarray(
                _core.bloch_sum(
                    contribution.V_xc,
                    np.asarray(kpoint, dtype=float).reshape(3),
                ),
                dtype=complex,
            )
            potentials.append(0.5 * (block + block.conj().T))
        return float(contribution.e_xc), potentials


__all__ = ["PeriodicExternalXC", "periodic_xc_difference_closed_cells"]
