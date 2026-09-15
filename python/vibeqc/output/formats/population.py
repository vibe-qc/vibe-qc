"""Population / properties dump -- ``{stem}.population.txt`` + ``.json``.

The `.out` text log already carries a properties block (Mulliken /
Löwdin charges, Mayer bond orders, dipole moment) for human reading.
This module emits the *machine-readable* siblings -- a per-job
``{stem}.population.txt`` (Chicago-table style) plus a
``{stem}.population.json`` (one JSON object) -- so downstream tools
(plots, regression dashboards, comparison scripts) don't have to
screen-scrape the SCF log.

Both files carry the same content, in different shapes. The TXT
form is human-readable plus copy-pasteable into a spreadsheet
(tab-separated within each block, blank line between blocks). The
JSON form is one object with top-level keys ``mulliken`` /
``loewdin`` / ``hirshfeld`` / ``mayer`` / ``dipole``, suitable for
``json.load`` straight into a dashboard.

Public API
----------

``compute_population_summary(result, basis, molecule)``
    Pulls the four properties via :mod:`vibeqc.properties` and
    returns a :class:`PopulationSummary` carrying the numerical
    data. Best-effort: any property that raises is reported with
    its error message on the returned object; the others still
    succeed. Per the wider output-module rule, a finished SCF
    never gets dragged down by a property-emit failure.

``compute_native_mulliken_population_summary(charges, molecule, method)``
    Adapts a semiempirical engine's validated net atomic Mulliken charges.
    Analyses that require a Gaussian AO basis are represented by stable
    ``unsupported:`` markers rather than fabricated from an unrelated basis.

``unsupported_population_summary(reason)``
    Build the same container for a route whose population properties
    are intentionally gated. The marker is structured and stable so
    published artifacts never carry raw internal exception strings for
    an unsupported route.

``compute_bipole_population_summary(result, basis, molecule, system)``
    BIPOLE periodic populations from lattice density/overlap blocks.
    This is deliberately separate from :mod:`vibeqc.properties`, whose
    formulas are molecular AO-matrix formulas. Pass the SCF lattice
    options and k-mesh when available so the overlap support and
    density-overlap contractions match the SCF route.

``compute_aiccm2026dev_b_population_summary(result, basis, molecule, system)``
    AICCM2026DEV-B periodic populations from finite-torus density/overlap
    blocks. This avoids sending lattice/k-point density containers through
    molecular AO-matrix formulas.

``write_population(stem, result, basis, molecule)``
    Emit ``{stem}.population.txt`` + ``{stem}.population.json``.
    Returns the two written paths.

The Phase-O1 ``.system`` manifest will record both paths in its
``[[outputs.files]]`` rows when the matching ``write_population``
kwarg is set on ``run_job``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ...spin_channels import spin_densities

__all__ = [
    "PopulationSummary",
    "compute_aiccm2026dev_b_population_summary",
    "compute_bipole_population_summary",
    "compute_native_mulliken_population_summary",
    "compute_population_summary",
    "unsupported_population_summary",
    "format_population_txt",
    "format_population_json",
    "write_population",
]


@dataclass
class PopulationSummary:
    """Numerical container for the four standard properties.

    Each field is independently optional -- a Mayer-bond-order failure
    (which can happen on near-singular overlap matrices) doesn't
    suppress the dipole-moment row.
    """

    # Per-atom (length = n_atoms). Charge = Z - q_population.
    mulliken_atoms: list[tuple[int, str, float, float]] = field(
        default_factory=list,
    )  # (atom_idx, symbol, Z, charge)
    loewdin_atoms: list[tuple[int, str, float, float]] = field(
        default_factory=list,
    )
    hirshfeld_atoms: list[tuple[int, str, float, float]] = field(
        default_factory=list,
    )
    # Spin-resolved Mulliken populations for unrestricted (UHF/UKS)
    # wavefunctions.  Each entry is (atom_idx, symbol, Z, spin_population)
    # where spin_population = alpha_population - beta_population.
    mulliken_spin_atoms: list[tuple[int, str, float, float]] = field(
        default_factory=list,
    )
    # Per atom-pair (i, j, order). Only the top entries -- the full
    # bond-order matrix is O(N^2) and most pairs are ~zero.
    mayer_bonds: list[tuple[int, int, str, str, float]] = field(
        default_factory=list,
    )  # (i, j, sym_i, sym_j, order)
    wiberg_bonds: list[tuple[int, int, str, str, float]] = field(
        default_factory=list,
    )  # Wiberg bond indices (top-N)
    npa_atoms: list[tuple[int, str, float, float]] = field(
        default_factory=list,
    )  # NPA charges (atom_idx, symbol, Z, charge)
    # Single dipole-moment dict.
    dipole: dict[str, Any] | None = None
    # Per-section error message; surfaced in .txt as a "section N/A"
    # row and as a "_error" JSON field.
    errors: dict[str, str] = field(default_factory=dict)


_POPULATION_SECTIONS = ("mulliken", "loewdin", "hirshfeld", "mayer", "dipole")
_SPIN_SECTIONS = ("mulliken_spin",)

_BIPOLE_LOEWDIN_UNSUPPORTED = (
    "unsupported: periodic BIPOLE Loewdin populations could not be evaluated "
    "for this density/k-mesh combination"
)
_BIPOLE_MAYER_UNAVAILABLE = (
    "unsupported: periodic BIPOLE Mayer bond orders could not be evaluated "
    "for this density/k-mesh combination"
)
_BIPOLE_HIRSHFELD_UNSUPPORTED = (
    "unsupported: periodic BIPOLE Hirshfeld charges require a periodic "
    "promolecule/grid partition, which is not implemented"
)
_BIPOLE_DIPOLE_UNSUPPORTED = (
    "unsupported: ordinary electric dipole moments are ill-defined for 3D "
    "periodic BIPOLE bulk cells; Berry-phase polarization is not implemented"
)
_AICCM_LOEWDIN_UNAVAILABLE = (
    "unsupported: periodic AICCM2026DEV-B Loewdin populations could not be "
    "evaluated for this density/k-mesh combination"
)
_AICCM_MAYER_UNAVAILABLE = (
    "unsupported: periodic AICCM2026DEV-B Mayer bond orders could not be "
    "evaluated for this finite-torus density"
)
_AICCM_HIRSHFELD_UNSUPPORTED = (
    "unsupported: periodic AICCM2026DEV-B Hirshfeld charges require a "
    "periodic promolecule/grid partition, which is not implemented"
)
_AICCM_DIPOLE_UNSUPPORTED = (
    "unsupported: ordinary electric dipole moments are ill-defined for "
    "periodic AICCM2026DEV-B cells; Berry-phase polarization is not implemented"
)


# ---------------------------------------------------------------------- #
# Computation                                                            #
# ---------------------------------------------------------------------- #


def _rank_bonds(
    bo: np.ndarray,
    atoms: list,
    threshold: float = 0.05,
) -> list[tuple[int, int, float]]:
    """Extract top bond pairs from a bond-order matrix."""
    n = bo.shape[0]
    entries: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = float(bo[i, j])
            if v >= threshold:
                entries.append((i, j, v))
    entries.sort(key=lambda t: -t[2])
    return entries


def compute_population_summary(
    result: Any,
    basis: Any,
    molecule: Any,
    *,
    n_top_bonds: int = 20,
    bond_threshold: float = 0.1,
    nuclear_charges: Any = None,
) -> PopulationSummary:
    """Pull Mulliken / Löwdin / Mayer / dipole via
    :mod:`vibeqc.properties`. Each section is wrapped in its own
    try/except -- partial success is preserved.

    ``nuclear_charges`` -- per-atom charges the SCF Hamiltonian saw
    (``Z - n_core`` on ECP atoms; :func:`vibeqc.ecp_metadata.
    effective_nuclear_charges` on the options that ran). ``None`` is bare
    ``Z``, correct only for an all-electron SCF (GitLab #642)."""
    out = PopulationSummary()

    # Atomic-number -> symbol via the xyz writer's table (same source
    # of truth used elsewhere in vibeqc.output).
    from .xyz import _symbol

    atoms = list(molecule.atoms)

    try:
        try:
            from ...properties import mulliken_charges
        except ImportError:
            from vibeqc.properties import mulliken_charges  # type: ignore[no-redef]
        # ``mulliken_charges`` already returns partial charges
        # (q_A = Z_A - population). Store them directly; subtracting from
        # Z again here would turn the charge back into the population (the
        # cause of the "+5.85 charge on carbon" bug).
        q_mul = np.asarray(
            mulliken_charges(result, basis, molecule, nuclear_charges=nuclear_charges),
            dtype=float,
        )
        rows: list[tuple[int, str, float, float]] = []
        for i, a in enumerate(atoms):
            z = int(a.Z)
            rows.append((i, _symbol(z), float(z), float(q_mul[i])))
        out.mulliken_atoms = rows
    except Exception as exc:
        out.errors["mulliken"] = f"{type(exc).__name__}: {exc}"

    # --- Mulliken spin populations (unrestricted only) ---
    _spin_a, _spin_b = spin_densities(result)
    if _spin_a is not None:
        try:
            from ..._vibeqc_core import compute_overlap as _co
        except ImportError:
            from vibeqc._vibeqc_core import compute_overlap as _co  # type: ignore[no-redef]
        try:
            S = np.asarray(_co(basis))
            P_spin = np.asarray(_spin_a) - np.asarray(_spin_b)
            PS_diag = np.einsum("ij,ji->i", P_spin, S)
            ao_to_atom = np.zeros(basis.nbasis, dtype=int)
            offset = 0
            for i_shell, shell in enumerate(basis.shells()):
                n_func = shell.nfunctions()
                ao_to_atom[offset : offset + n_func] = shell.atom_index
                offset += n_func
            electron_spin = np.bincount(
                ao_to_atom, weights=PS_diag.real, minlength=len(atoms)
            )
            spin_rows: list[tuple[int, str, float, float]] = []
            for i, a in enumerate(atoms):
                z = int(a.Z)
                spin_rows.append((i, _symbol(z), float(z), float(electron_spin[i])))
            out.mulliken_spin_atoms = spin_rows
        except Exception as exc:
            out.errors["mulliken_spin"] = f"{type(exc).__name__}: {exc}"

    try:
        try:
            from ...properties import loewdin_charges
        except ImportError:
            from vibeqc.properties import loewdin_charges  # type: ignore[no-redef]
        # As above: loewdin_charges already returns partial charges.
        q_low = np.asarray(
            loewdin_charges(result, basis, molecule, nuclear_charges=nuclear_charges),
            dtype=float,
        )
        rows = []
        for i, a in enumerate(atoms):
            z = int(a.Z)
            rows.append((i, _symbol(z), float(z), float(q_low[i])))
        out.loewdin_atoms = rows
    except Exception as exc:
        out.errors["loewdin"] = f"{type(exc).__name__}: {exc}"

    try:
        try:
            from ...properties import hirshfeld_charges
        except ImportError:
            from vibeqc.properties import (  # type: ignore[no-redef]
                hirshfeld_charges,
            )
        q_hir = np.asarray(
            hirshfeld_charges(
                result, basis, molecule, nuclear_charges=nuclear_charges
            ).charges,
            dtype=float,
        )
        rows = []
        for i, a in enumerate(atoms):
            z = int(a.Z)
            rows.append((i, _symbol(z), float(z), float(q_hir[i])))
        out.hirshfeld_atoms = rows
    except Exception as exc:
        out.errors["hirshfeld"] = f"{type(exc).__name__}: {exc}"

    try:
        try:
            from ...properties import mayer_bond_orders, prominent_bonds
        except ImportError:
            from vibeqc.properties import (  # type: ignore[no-redef]
                mayer_bond_orders,
                prominent_bonds,
            )
        bo = np.asarray(
            mayer_bond_orders(result, basis, molecule),
            dtype=float,
        )
        # All bonds (i<j, order >= threshold), sorted descending.
        # Truncate to top-N after the call.
        ranked = prominent_bonds(
            bo,
            molecule,
            threshold=bond_threshold,
        )[:n_top_bonds]
        rows_b: list[tuple[int, int, str, str, float]] = []
        for entry in ranked:
            # prominent_bonds returns (i, j, order) or similar; the
            # exact dataclass shape varies -- extract defensively.
            if hasattr(entry, "i"):
                i = int(entry.i)
                j = int(entry.j)
                o = float(entry.order)
            elif isinstance(entry, tuple) and len(entry) >= 3:
                i, j, o = int(entry[0]), int(entry[1]), float(entry[2])
            else:
                continue
            si = _symbol(int(atoms[i].Z))
            sj = _symbol(int(atoms[j].Z))
            rows_b.append((i, j, si, sj, o))
        out.mayer_bonds = rows_b
    except Exception as exc:
        out.errors["mayer"] = f"{type(exc).__name__}: {exc}"

    # --- Dipole ---
    try:
        try:
            from ...properties import dipole_moment
        except ImportError:
            from vibeqc.properties import dipole_moment  # type: ignore[no-redef]
        d = dipole_moment(result, basis, molecule, nuclear_charges=nuclear_charges)
        out.dipole = {
            "x_ebohr": float(d.x),
            "y_ebohr": float(d.y),
            "z_ebohr": float(d.z),
            "total_ebohr": float(d.total),
            "total_debye": float(d.total_debye),
            "origin_bohr": list(d.origin),
        }
    except Exception as exc:
        out.errors["dipole"] = f"{type(exc).__name__}: {exc}"

    # --- Wiberg bond orders (v0.22.0 BOND) ---
    try:
        try:
            from ...bond_analysis import wiberg_bond_orders
        except ImportError:
            from vibeqc.bond_analysis import (
                wiberg_bond_orders,  # type: ignore[no-redef]
            )
        bo_w = np.asarray(wiberg_bond_orders(result, basis, molecule), dtype=float)
        ranked_w = _rank_bonds(bo_w, atoms, threshold=0.05)
        rows_w = []
        for i, j, o in ranked_w[:n_top_bonds]:
            si = _symbol(int(atoms[i].Z))
            sj = _symbol(int(atoms[j].Z))
            rows_w.append((i, j, si, sj, o))
        out.wiberg_bonds = rows_w
    except Exception as exc:
        out.errors["wiberg"] = f"{type(exc).__name__}: {exc}"

    # --- NPA charges (v0.22.0 BOND) ---
    try:
        try:
            from ..._vibeqc_core import compute_overlap as _co
            from ...nbo import npa_charges as _npa
        except ImportError:
            from vibeqc._vibeqc_core import (
                compute_overlap as _co,  # type: ignore[no-redef]
            )
            from vibeqc.nbo import npa_charges as _npa  # type: ignore[no-redef]
        S = np.asarray(_co(basis))
        _npa_a, _npa_b = spin_densities(result)
        if _npa_a is not None:
            P_npa = np.asarray(_npa_a.real + _npa_b.real)
        else:
            P_npa = np.asarray(result.density.real)
        q_npa = np.asarray(_npa(P_npa, S, basis, molecule), dtype=float)
        rows_npa = []
        for i, a in enumerate(atoms):
            z = int(a.Z)
            rows_npa.append((i, _symbol(z), float(z), float(q_npa[i])))
        out.npa_atoms = rows_npa
    except Exception as exc:
        out.errors["npa"] = f"{type(exc).__name__}: {exc}"

    return out


@dataclass
class _LatticeDensityView:
    cells: Sequence[Any]
    blocks: Sequence[np.ndarray]
    nbf: int


def _cell_key(cell: Any) -> tuple[int, int, int]:
    return tuple(int(x) for x in np.asarray(cell.index, dtype=int).reshape(3))


def _basis_ao_to_atom(basis: Any) -> np.ndarray:
    """Return the AO -> atom map using the public BasisSet shell API."""
    per_ao: list[int] = []
    for shell in basis.shells():
        angular = int(shell.l)
        n_func = (
            2 * angular + 1
            if bool(getattr(shell, "pure", True))
            else (angular + 1) * (angular + 2) // 2
        )
        per_ao.extend([int(shell.atom_index)] * n_func)
    return np.asarray(per_ao, dtype=np.int64)


def _as_lattice_density_view(result: Any) -> _LatticeDensityView:
    """Extract the total BIPOLE density as real-space lattice blocks."""
    density = getattr(result, "density", None)
    if density is not None and hasattr(density, "cells") and hasattr(density, "blocks"):
        blocks = [np.asarray(block) for block in density.blocks]
        nbf = int(getattr(density, "nbf", blocks[0].shape[0] if blocks else 0))
        return _LatticeDensityView(cells=list(density.cells), blocks=blocks, nbf=nbf)

    alpha = getattr(result, "density_alpha", None)
    beta = getattr(result, "density_beta", None)
    if alpha is None or beta is None:
        raise TypeError("BIPOLE population requires lattice density blocks")
    if not (
        hasattr(alpha, "cells")
        and hasattr(alpha, "blocks")
        and hasattr(beta, "cells")
        and hasattr(beta, "blocks")
    ):
        raise TypeError("BIPOLE spin densities must be lattice density blocks")

    alpha_cells = list(alpha.cells)
    beta_cells = list(beta.cells)
    alpha_blocks = {
        _cell_key(cell): np.asarray(block)
        for cell, block in zip(alpha_cells, alpha.blocks)
    }
    beta_blocks = {
        _cell_key(cell): np.asarray(block)
        for cell, block in zip(beta_cells, beta.blocks)
    }
    cell_by_key = {_cell_key(cell): cell for cell in alpha_cells}
    for cell in beta_cells:
        cell_by_key.setdefault(_cell_key(cell), cell)

    keys = [_cell_key(cell) for cell in alpha_cells]
    keys.extend(key for key in beta_blocks if key not in alpha_blocks)
    if not keys:
        raise ValueError("BIPOLE spin densities have no lattice cells")

    first_block = next(iter(alpha_blocks.values()), None)
    if first_block is None:
        first_block = next(iter(beta_blocks.values()))
    nbf = int(first_block.shape[0])
    zero = np.zeros((nbf, nbf), dtype=np.result_type(first_block, float))
    blocks: list[np.ndarray] = []
    cells: list[Any] = []
    for key in keys:
        a_block = alpha_blocks.get(key, zero)
        b_block = beta_blocks.get(key, zero)
        if a_block.shape != b_block.shape:
            raise ValueError(
                f"BIPOLE alpha/beta density shape mismatch at cell {key}"
            )
        cells.append(cell_by_key[key])
        blocks.append(np.asarray(a_block) + np.asarray(b_block))
    return _LatticeDensityView(cells=cells, blocks=blocks, nbf=nbf)


def _overlap_lattice_for_density(
    basis: Any,
    system: Any,
    density: _LatticeDensityView,
    lattice_options: Any | None = None,
):
    """Build real-space overlap blocks for the BIPOLE population contraction.

    When the SCF lattice options are available, use their operator support:
    corrected-gauge BIPOLE densities may be wider than the one-electron
    operator lattice, and the energy path contracts by iterating operator
    cells and looking up density blocks by key. Falling back to density
    support is reserved for direct helper calls that do not know the SCF
    cutoff.
    """
    if not density.cells:
        raise ValueError("BIPOLE density has no lattice cells")

    from ..._vibeqc_core import LatticeSumOptions, compute_overlap_lattice

    if lattice_options is not None:
        opts = LatticeSumOptions()
        opts.cutoff_bohr = float(getattr(lattice_options, "cutoff_bohr"))
        opts.nuclear_cutoff_bohr = float(
            getattr(lattice_options, "nuclear_cutoff_bohr", opts.cutoff_bohr)
        )
        return compute_overlap_lattice(basis, system, opts)

    max_norm = max(
        float(np.linalg.norm(np.asarray(cell.r_cart, dtype=float)))
        for cell in density.cells
    )
    opts = LatticeSumOptions()
    opts.cutoff_bohr = max_norm + 1.0e-8
    opts.nuclear_cutoff_bohr = opts.cutoff_bohr
    return compute_overlap_lattice(basis, system, opts)


def _bipole_mulliken_charges(
    density: _LatticeDensityView,
    overlap_lattice: Any,
    basis: Any,
    molecule: Any,
) -> np.ndarray:
    """Mulliken charges from the BIPOLE lattice contraction convention.

    BIPOLE density blocks follow ``rho(r) = sum_g D_mu,nu(g)
    chi_mu(r) chi_nu(r - R_g)``. The electron count is the same
    operator-cell-keyed lattice overlap contraction used by the SCF
    energy, ``sum_g,mu,nu D(g) S(g)`` over the overlap lattice support.
    Mulliken atom populations split each AO-pair overlap population equally
    between the AO centers unless both AOs live on the same atom.
    """
    atoms = list(molecule.atoms)
    n_atoms = len(atoms)
    ao_to_atom = _basis_ao_to_atom(basis)
    if len(ao_to_atom) != density.nbf:
        raise ValueError(
            "BIPOLE AO-to-atom map length does not match density dimension"
        )

    density_by_key = {
        _cell_key(cell): np.asarray(block)
        for cell, block in zip(density.cells, density.blocks)
    }
    electron_pop = np.zeros(n_atoms, dtype=np.float64)
    for cell, s_block_raw in zip(overlap_lattice.cells, overlap_lattice.blocks):
        key = _cell_key(cell)
        if key not in density_by_key:
            raise ValueError(f"BIPOLE density lattice missing overlap cell {key}")
        d_block = np.asarray(density_by_key[key])
        s_block = np.asarray(s_block_raw)
        if d_block.shape != s_block.shape:
            raise ValueError(
                f"BIPOLE density/overlap shape mismatch at cell {key}: "
                f"D{d_block.shape} vs S{s_block.shape}"
            )
        contrib = np.real(np.asarray(d_block * s_block))
        if not np.all(np.isfinite(contrib)):
            raise ValueError(f"BIPOLE Mulliken contribution is non-finite at {key}")
        for mu in range(density.nbf):
            atom_mu = int(ao_to_atom[mu])
            for nu in range(density.nbf):
                value = float(contrib[mu, nu])
                if value == 0.0:
                    continue
                atom_nu = int(ao_to_atom[nu])
                if atom_mu == atom_nu:
                    electron_pop[atom_mu] += value
                else:
                    half = 0.5 * value
                    electron_pop[atom_mu] += half
                    electron_pop[atom_nu] += half

    charges = np.asarray([atom.Z for atom in atoms], dtype=np.float64) - electron_pop
    if not np.all(np.isfinite(charges)):
        raise ValueError("BIPOLE Mulliken charges are non-finite")
    return charges


def _bipole_mayer_bond_orders(
    result: Any,
    overlap_lattice: Any,
    basis: Any,
    system: Any,
    *,
    kmesh: Any | None = None,
) -> np.ndarray:
    """Mayer bond orders from BIPOLE per-k SCF orbitals and overlaps.

    This evaluates the standard periodic k-space density-overlap
    contraction over the same k-mesh as the SCF calculation:

        B_AB = sum_k w_k sum_{mu in A,nu in B}
               (P(k) S(k))_{mu,nu} (P(k) S(k))_{nu,mu}

    For unrestricted BIPOLE, the spin-resolved molecular convention
    ``2 * (alpha + beta)`` is applied to the per-spin density matrices.
    """
    if kmesh is None:
        weights = np.ones(len(getattr(result, "overlap", [])), dtype=float)
    else:
        weights = np.asarray(kmesh.weights, dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum == 0.0:
        raise ValueError("BIPOLE Mayer k-point weights sum to zero")
    if abs(weight_sum - 1.0) > 1.0e-8:
        weights = weights / weight_sum

    ao_to_atom = _basis_ao_to_atom(basis)
    n_atoms = len(system.unit_cell)
    if len(ao_to_atom) != int(overlap_lattice.nbf):
        raise ValueError(
            "BIPOLE AO-to-atom map length does not match overlap dimension"
        )

    bond_orders = np.zeros((n_atoms, n_atoms), dtype=np.float64)

    def _accumulate_channel(
        coeffs_per_k: Sequence[Any],
        overlaps_per_k: Sequence[Any],
        occs_per_k: Sequence[Any],
        *,
        spin_factor: float,
    ) -> None:
        if not coeffs_per_k:
            raise ValueError("BIPOLE Mayer analysis requires MO coefficients")
        if len(coeffs_per_k) != len(overlaps_per_k):
            raise ValueError("BIPOLE Mayer coefficient/overlap k-list mismatch")
        if len(coeffs_per_k) != len(weights):
            raise ValueError("BIPOLE Mayer k-weight count does not match MO list")
        if len(occs_per_k) != len(coeffs_per_k):
            raise ValueError("BIPOLE Mayer occupation k-list mismatch")

        for idx, (coeffs_raw, overlap_raw, occ_raw) in enumerate(
            zip(coeffs_per_k, overlaps_per_k, occs_per_k)
        ):
            coeffs = np.asarray(coeffs_raw, dtype=np.complex128)
            S_k = np.asarray(overlap_raw, dtype=np.complex128)
            S_k = 0.5 * (S_k + S_k.conj().T)
            occ = np.asarray(occ_raw, dtype=float).reshape(-1)
            if coeffs.shape[0] != coeffs.shape[1] or coeffs.shape != S_k.shape:
                raise ValueError(
                    "BIPOLE Mayer coefficient/overlap shape mismatch at "
                    f"k={idx}: C{coeffs.shape} vs S{S_k.shape}"
                )
            if occ.size != coeffs.shape[1]:
                raise ValueError(
                    "BIPOLE Mayer occupation length mismatch at "
                    f"k={idx}: {occ.size} vs {coeffs.shape[1]}"
                )
            weighted_coeffs = coeffs * occ.reshape(1, -1)
            PS = weighted_coeffs @ (coeffs.conj().T @ S_k)
            mayer_ao = PS * PS.T
            wk = float(weights[idx]) * float(spin_factor)
            for mu in range(mayer_ao.shape[0]):
                atom_mu = int(ao_to_atom[mu])
                for nu in range(mayer_ao.shape[1]):
                    atom_nu = int(ao_to_atom[nu])
                    if atom_mu == atom_nu:
                        continue
                    bond_orders[atom_mu, atom_nu] += float(
                        wk * np.real(mayer_ao[mu, nu])
                    )

    overlaps = list(getattr(result, "overlap", []))
    if hasattr(result, "mo_coeffs_alpha") and hasattr(result, "mo_coeffs_beta"):
        n_electrons = int(system.n_electrons())
        multiplicity = int(getattr(system, "multiplicity", 1) or 1)
        n_alpha = (n_electrons + multiplicity - 1) // 2
        n_beta = n_electrons - n_alpha
        coeffs_alpha = list(getattr(result, "mo_coeffs_alpha"))
        coeffs_beta = list(getattr(result, "mo_coeffs_beta"))
        occs_alpha = getattr(result, "occupations_alpha", None)
        if not occs_alpha:
            occs_alpha = [
                np.r_[
                    np.ones(n_alpha),
                    np.zeros(max(0, np.asarray(c).shape[1] - n_alpha)),
                ]
                for c in coeffs_alpha
            ]
        occs_beta = getattr(result, "occupations_beta", None)
        if not occs_beta:
            occs_beta = [
                np.r_[
                    np.ones(n_beta),
                    np.zeros(max(0, np.asarray(c).shape[1] - n_beta)),
                ]
                for c in coeffs_beta
            ]
        _accumulate_channel(coeffs_alpha, overlaps, list(occs_alpha), spin_factor=2.0)
        _accumulate_channel(coeffs_beta, overlaps, list(occs_beta), spin_factor=2.0)
    else:
        coeffs = list(getattr(result, "mo_coeffs", []))
        occupations = getattr(result, "occupations", None)
        if not occupations:
            n_occ = int(system.n_electrons()) // 2
            occupations = [
                np.r_[
                    2.0 * np.ones(n_occ),
                    np.zeros(max(0, np.asarray(c).shape[1] - n_occ)),
                ]
                for c in coeffs
            ]
        _accumulate_channel(coeffs, overlaps, list(occupations), spin_factor=1.0)

    bond_orders = 0.5 * (bond_orders + bond_orders.T)
    if not np.all(np.isfinite(bond_orders)):
        raise ValueError("BIPOLE Mayer bond orders are non-finite")
    return bond_orders


def _bipole_k_weights(result: Any, kmesh: Any | None) -> np.ndarray:
    overlaps = list(getattr(result, "overlap", []))
    if not overlaps:
        raise ValueError("BIPOLE population analysis requires k-point overlaps")
    if kmesh is None:
        weights = np.ones(len(overlaps), dtype=float)
    else:
        weights = np.asarray(kmesh.weights, dtype=float).reshape(-1)
    if weights.size != len(overlaps):
        raise ValueError(
            "BIPOLE population k-weight count does not match overlap list"
        )
    weight_sum = float(weights.sum())
    if weight_sum == 0.0:
        raise ValueError("BIPOLE population k-point weights sum to zero")
    return weights / weight_sum


def _occupation_rows(
    occupations: Any,
    coeffs_per_k: Sequence[Any],
    *,
    n_occupied: int,
    occupation_value: float,
) -> list[np.ndarray]:
    if occupations is not None:
        rows = [np.asarray(row, dtype=float).reshape(-1) for row in occupations]
        if rows:
            return rows
    out: list[np.ndarray] = []
    for coeffs in coeffs_per_k:
        n_orb = int(np.asarray(coeffs).shape[1])
        occ = np.zeros(n_orb, dtype=float)
        occ[: min(n_occupied, n_orb)] = float(occupation_value)
        out.append(occ)
    return out


def _accumulate_bipole_loewdin_populations(
    coeffs_per_k: Sequence[Any],
    overlaps_per_k: Sequence[Any],
    occupations_per_k: Sequence[Any],
    weights: np.ndarray,
    ao_to_atom: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    if not coeffs_per_k:
        raise ValueError("BIPOLE Loewdin analysis requires MO coefficients")
    if len(coeffs_per_k) != len(overlaps_per_k):
        raise ValueError("BIPOLE Loewdin coefficient/overlap k-list mismatch")
    if len(coeffs_per_k) != len(weights):
        raise ValueError("BIPOLE Loewdin k-weight count does not match MO list")
    if len(occupations_per_k) != len(coeffs_per_k):
        raise ValueError("BIPOLE Loewdin occupation k-list mismatch")

    electron_pop = np.zeros(n_atoms, dtype=np.float64)
    for idx, (coeffs_raw, overlap_raw, occ_raw) in enumerate(
        zip(coeffs_per_k, overlaps_per_k, occupations_per_k)
    ):
        coeffs = np.asarray(coeffs_raw, dtype=np.complex128)
        S_k = np.asarray(overlap_raw, dtype=np.complex128)
        S_k = 0.5 * (S_k + S_k.conj().T)
        occ = np.asarray(occ_raw, dtype=float).reshape(-1)
        if coeffs.ndim != 2 or S_k.ndim != 2 or coeffs.shape != S_k.shape:
            raise ValueError(
                "BIPOLE Loewdin coefficient/overlap shape mismatch at "
                f"k={idx}: C{coeffs.shape} vs S{S_k.shape}"
            )
        if occ.size != coeffs.shape[1]:
            raise ValueError(
                "BIPOLE Loewdin occupation length mismatch at "
                f"k={idx}: {occ.size} vs {coeffs.shape[1]}"
            )
        eigvals, eigvecs = np.linalg.eigh(S_k)
        min_eig = float(np.min(eigvals))
        if min_eig < 1.0e-10:
            raise ValueError(
                "BIPOLE Loewdin overlap matrix is near-singular at "
                f"k={idx} (min eigenvalue {min_eig:.2e})"
            )
        S_half = (eigvecs * np.sqrt(eigvals).reshape(1, -1)) @ eigvecs.conj().T
        density_k = (coeffs * occ.reshape(1, -1)) @ coeffs.conj().T
        lowdin_density = S_half @ density_k @ S_half
        diagonal = np.real(np.diag(lowdin_density))
        if diagonal.size != ao_to_atom.size:
            raise ValueError(
                "BIPOLE Loewdin AO-to-atom map length does not match "
                f"k={idx} density dimension"
            )
        if not np.all(np.isfinite(diagonal)):
            raise ValueError(f"BIPOLE Loewdin population is non-finite at k={idx}")
        electron_pop += float(weights[idx]) * np.bincount(
            ao_to_atom,
            weights=diagonal,
            minlength=n_atoms,
        )
    return electron_pop


def _bipole_loewdin_charges(
    result: Any,
    basis: Any,
    system: Any,
    *,
    kmesh: Any | None = None,
) -> np.ndarray:
    """Löwdin charges from k-resolved BIPOLE densities and overlaps.

    This mirrors the molecular formula ``diag(S^1/2 P S^1/2)`` at each
    SCF k-point, then averages the per-k electron populations with the
    SCF k weights. It uses the same per-k density convention as the
    BIPOLE Mayer analysis and avoids the molecular Gamma-proxy path.
    """
    weights = _bipole_k_weights(result, kmesh)
    overlaps = list(getattr(result, "overlap", []))
    ao_to_atom = _basis_ao_to_atom(basis)
    n_atoms = len(system.unit_cell)
    n_electrons = int(system.n_electrons())
    multiplicity = int(getattr(system, "multiplicity", 1) or 1)

    if hasattr(result, "mo_coeffs_alpha") and hasattr(result, "mo_coeffs_beta"):
        n_alpha = (n_electrons + multiplicity - 1) // 2
        n_beta = n_electrons - n_alpha
        coeffs_alpha = list(getattr(result, "mo_coeffs_alpha"))
        coeffs_beta = list(getattr(result, "mo_coeffs_beta"))
        pop_alpha = _accumulate_bipole_loewdin_populations(
            coeffs_alpha,
            overlaps,
            _occupation_rows(
                getattr(result, "occupations_alpha", None),
                coeffs_alpha,
                n_occupied=n_alpha,
                occupation_value=1.0,
            ),
            weights,
            ao_to_atom,
            n_atoms,
        )
        pop_beta = _accumulate_bipole_loewdin_populations(
            coeffs_beta,
            overlaps,
            _occupation_rows(
                getattr(result, "occupations_beta", None),
                coeffs_beta,
                n_occupied=n_beta,
                occupation_value=1.0,
            ),
            weights,
            ao_to_atom,
            n_atoms,
        )
        electron_pop = pop_alpha + pop_beta
    else:
        coeffs = list(getattr(result, "mo_coeffs", []))
        electron_pop = _accumulate_bipole_loewdin_populations(
            coeffs,
            overlaps,
            _occupation_rows(
                getattr(result, "occupations", None),
                coeffs,
                n_occupied=n_electrons // 2,
                occupation_value=2.0,
            ),
            weights,
            ao_to_atom,
            n_atoms,
        )

    charges = (
        np.asarray([atom.Z for atom in system.unit_cell], dtype=np.float64)
        - electron_pop
    )
    if not np.all(np.isfinite(charges)):
        raise ValueError("BIPOLE Loewdin charges are non-finite")
    return charges


def _rank_bipole_mayer_bonds(
    bond_orders: np.ndarray,
    molecule: Any,
    *,
    n_top_bonds: int,
    bond_threshold: float,
) -> list[tuple[int, int, str, str, float]]:
    from .xyz import _symbol

    atoms = list(molecule.atoms)
    rows: list[tuple[int, int, str, str, float]] = []
    for i in range(bond_orders.shape[0]):
        for j in range(i + 1, bond_orders.shape[1]):
            order = float(bond_orders[i, j])
            if order < bond_threshold:
                continue
            rows.append(
                (
                    i,
                    j,
                    _symbol(int(atoms[i].Z)),
                    _symbol(int(atoms[j].Z)),
                    order,
                )
            )
    rows.sort(key=lambda row: -row[4])
    return rows[:n_top_bonds]


def compute_bipole_population_summary(
    result: Any,
    basis: Any,
    molecule: Any,
    system: Any,
    *,
    lattice_options: Any | None = None,
    kmesh: Any | None = None,
    n_top_bonds: int = 20,
    bond_threshold: float = 0.1,
) -> PopulationSummary:
    """Compute BIPOLE periodic population properties where defined.

    Mulliken uses lattice density and overlap blocks in the same
    real-space contraction convention as the BIPOLE SCF energy. Löwdin
    charges and Mayer bond orders use the corresponding k-space
    density-overlap contraction over the SCF k-mesh. Ordinary dipole
    output stays explicitly unsupported because the bulk position
    operator requires a Berry-phase polarization implementation.
    """
    out = PopulationSummary()

    from .xyz import _symbol

    density = _as_lattice_density_view(result)
    overlap_lattice = _overlap_lattice_for_density(
        basis,
        system,
        density,
        lattice_options=lattice_options,
    )
    charges = _bipole_mulliken_charges(density, overlap_lattice, basis, molecule)

    rows: list[tuple[int, str, float, float]] = []
    for i, atom in enumerate(molecule.atoms):
        z = int(atom.Z)
        rows.append((i, _symbol(z), float(z), float(charges[i])))
    out.mulliken_atoms = rows

    try:
        loewdin_charges = _bipole_loewdin_charges(
            result,
            basis,
            system,
            kmesh=kmesh,
        )
        rows = []
        for i, atom in enumerate(molecule.atoms):
            z = int(atom.Z)
            rows.append((i, _symbol(z), float(z), float(loewdin_charges[i])))
        out.loewdin_atoms = rows
    except Exception:
        out.errors["loewdin"] = _BIPOLE_LOEWDIN_UNSUPPORTED

    try:
        mayer = _bipole_mayer_bond_orders(
            result,
            overlap_lattice,
            basis,
            system,
            kmesh=kmesh,
        )
        out.mayer_bonds = _rank_bipole_mayer_bonds(
            mayer,
            molecule,
            n_top_bonds=n_top_bonds,
            bond_threshold=bond_threshold,
        )
    except Exception:
        out.errors["mayer"] = _BIPOLE_MAYER_UNAVAILABLE

    out.errors["hirshfeld"] = _BIPOLE_HIRSHFELD_UNSUPPORTED
    out.errors["dipole"] = _BIPOLE_DIPOLE_UNSUPPORTED
    return out


def _matrix_blocks(value: object) -> list[np.ndarray]:
    lattice_blocks = getattr(value, "blocks", None)
    if lattice_blocks is not None:
        return [np.asarray(block) for block in lattice_blocks]
    array = np.asarray(value)
    if array.ndim == 2:
        return [array]
    return [np.asarray(block) for block in value]  # type: ignore[union-attr]


def _aiccm_effective_charges(result: Any, system: Any) -> np.ndarray:
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    charges = getattr(diagnostics, "effective_nuclear_charges", None)
    if charges is not None:
        return np.asarray(charges, dtype=float)
    return np.asarray([atom.Z for atom in system.unit_cell], dtype=float)


def _aiccm_effective_electron_count(result: Any, system: Any) -> int:
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    electrons = getattr(diagnostics, "effective_electron_count", None)
    if electrons is not None:
        return int(electrons)
    return int(system.n_electrons())


def _aiccm2026dev_b_loewdin_charges(
    result: Any,
    basis: Any,
    system: Any,
) -> np.ndarray:
    """Löwdin charges from finite-torus AICCM2026DEV-B k blocks."""
    weights = np.asarray(getattr(result, "kpoint_weights"), dtype=float).reshape(-1)
    weight_sum = float(weights.sum())
    if weight_sum == 0.0:
        raise ValueError("AICCM2026DEV-B Loewdin k-point weights sum to zero")
    if abs(weight_sum - 1.0) > 1.0e-8:
        weights = weights / weight_sum

    overlaps = _matrix_blocks(getattr(result, "overlap"))
    ao_to_atom = _basis_ao_to_atom(basis)
    n_atoms = len(system.unit_cell)
    effective_electrons = _aiccm_effective_electron_count(result, system)
    multiplicity = int(getattr(system, "multiplicity", 1) or 1)

    if hasattr(result, "mo_coeffs_alpha") and hasattr(result, "mo_coeffs_beta"):
        n_alpha = (effective_electrons + multiplicity - 1) // 2
        n_beta = effective_electrons - n_alpha
        coeffs_alpha = list(getattr(result, "mo_coeffs_alpha"))
        coeffs_beta = list(getattr(result, "mo_coeffs_beta"))
        pop_alpha = _accumulate_bipole_loewdin_populations(
            coeffs_alpha,
            overlaps,
            _occupation_rows(
                getattr(result, "occupations_alpha", None),
                coeffs_alpha,
                n_occupied=n_alpha,
                occupation_value=1.0,
            ),
            weights,
            ao_to_atom,
            n_atoms,
        )
        pop_beta = _accumulate_bipole_loewdin_populations(
            coeffs_beta,
            overlaps,
            _occupation_rows(
                getattr(result, "occupations_beta", None),
                coeffs_beta,
                n_occupied=n_beta,
                occupation_value=1.0,
            ),
            weights,
            ao_to_atom,
            n_atoms,
        )
        electron_pop = pop_alpha + pop_beta
        charges = _aiccm_effective_charges(result, system) - electron_pop
        if not np.all(np.isfinite(charges)):
            raise ValueError("AICCM2026DEV-B Loewdin charges are non-finite")
        return charges

    coeffs_attr = getattr(result, "mo_coeffs", None)
    coeffs = list(coeffs_attr) if coeffs_attr is not None else []
    if coeffs:
        electron_pop = _accumulate_bipole_loewdin_populations(
            coeffs,
            overlaps,
            _occupation_rows(
                getattr(result, "occupations", None),
                coeffs,
                n_occupied=effective_electrons // 2,
                occupation_value=2.0,
            ),
            weights,
            ao_to_atom,
            n_atoms,
        )
        charges = _aiccm_effective_charges(result, system) - electron_pop
        if not np.all(np.isfinite(charges)):
            raise ValueError("AICCM2026DEV-B Loewdin charges are non-finite")
        return charges

    _lw_a, _lw_b = spin_densities(result)
    if _lw_a is not None:
        alpha = _matrix_blocks(_lw_a)
        beta = _matrix_blocks(_lw_b)
        if len(alpha) != len(beta):
            raise ValueError("AICCM2026DEV-B alpha/beta density block mismatch")
        densities = [a + b for a, b in zip(alpha, beta)]
    else:
        densities = _matrix_blocks(getattr(result, "density"))

    if len(densities) != len(overlaps) or len(densities) != weights.size:
        raise ValueError(
            "AICCM2026DEV-B Loewdin density/overlap/k-weight block mismatch"
        )

    electron_pop = np.zeros(n_atoms, dtype=np.float64)
    for idx, (weight, density_raw, overlap_raw) in enumerate(
        zip(weights, densities, overlaps)
    ):
        density = np.asarray(density_raw, dtype=np.complex128)
        overlap = np.asarray(overlap_raw, dtype=np.complex128)
        if density.ndim != 2 or overlap.ndim != 2 or density.shape != overlap.shape:
            raise ValueError(
                "AICCM2026DEV-B Loewdin density/overlap shape mismatch at "
                f"k={idx}: D{density.shape} vs S{overlap.shape}"
            )
        if density.shape[0] != ao_to_atom.size:
            raise ValueError(
                "AICCM2026DEV-B Loewdin AO-to-atom map length does not match "
                f"k={idx} density dimension"
            )
        overlap = 0.5 * (overlap + overlap.conj().T)
        eigvals, eigvecs = np.linalg.eigh(overlap)
        min_eig = float(np.min(eigvals))
        if min_eig < 1.0e-10:
            raise ValueError(
                "AICCM2026DEV-B Loewdin overlap matrix is near-singular at "
                f"k={idx} (min eigenvalue {min_eig:.2e})"
            )
        S_half = (eigvecs * np.sqrt(eigvals).reshape(1, -1)) @ eigvecs.conj().T
        lowdin_density = S_half @ density @ S_half
        diagonal = np.real(np.diag(lowdin_density))
        electron_pop += float(weight) * np.bincount(
            ao_to_atom,
            weights=diagonal,
            minlength=n_atoms,
        )

    charges = _aiccm_effective_charges(result, system) - electron_pop
    if not np.all(np.isfinite(charges)):
        raise ValueError("AICCM2026DEV-B Loewdin charges are non-finite")
    return charges


def compute_aiccm2026dev_b_population_summary(
    result: Any,
    basis: Any,
    molecule: Any,
    system: Any,
    *,
    n_top_bonds: int = 20,
    bond_threshold: float = 0.1,
) -> PopulationSummary:
    """Compute finite-torus AICCM2026DEV-B population properties.

    The AICCM2026DEV-B route stores k-resolved finite-torus matrices rather
    than one molecular AO density. Use its dedicated charge and bond helpers so
    large periodic basis containers never flow into molecular ``einsum`` paths.
    """
    out = PopulationSummary()

    from .xyz import _symbol

    atoms = list(molecule.atoms)
    from ...periodic.chi.properties import (
        aiccm2026dev_b_mayer_bond_orders,
        derive_aiccm2026dev_b_scf_properties,
    )

    props = derive_aiccm2026dev_b_scf_properties(result, system, basis)
    rows: list[tuple[int, str, float, float]] = []
    for i, atom in enumerate(atoms):
        z = int(atom.Z)
        rows.append((i, _symbol(z), float(z), float(props.mulliken_charges[i])))
    out.mulliken_atoms = rows

    try:
        loewdin = _aiccm2026dev_b_loewdin_charges(result, basis, system)
        rows = []
        for i, atom in enumerate(atoms):
            z = int(atom.Z)
            rows.append((i, _symbol(z), float(z), float(loewdin[i])))
        out.loewdin_atoms = rows
    except Exception:
        out.errors["loewdin"] = _AICCM_LOEWDIN_UNAVAILABLE

    try:
        analysis = aiccm2026dev_b_mayer_bond_orders(
            result,
            system,
            basis,
            threshold=bond_threshold,
            max_bonds=n_top_bonds,
        )
        out.mayer_bonds = [
            (
                int(bond.atom_i),
                int(bond.atom_j),
                str(bond.symbol_i),
                str(bond.symbol_j),
                float(bond.order),
            )
            for bond in analysis.bonds
        ]
    except Exception:
        out.errors["mayer"] = _AICCM_MAYER_UNAVAILABLE

    out.errors["hirshfeld"] = _AICCM_HIRSHFELD_UNSUPPORTED
    out.errors["dipole"] = _AICCM_DIPOLE_UNSUPPORTED
    return out


def unsupported_population_summary(reason: str) -> PopulationSummary:
    """Return a structured unsupported-route population summary.

    Use this when the SCF route is valid but the corresponding population
    formulas are not implemented for that route's density representation.
    The stable ``unsupported:`` prefix lets downstream artifact checks
    distinguish a deliberate gate from an unexpected property failure.
    """
    msg = reason.strip()
    if not msg:
        msg = "not available for this route"
    if not msg.startswith("unsupported:"):
        msg = f"unsupported: {msg}"
    out = PopulationSummary()
    out.errors = {section: msg for section in (*_POPULATION_SECTIONS, *_SPIN_SECTIONS)}
    return out


def compute_native_mulliken_population_summary(
    charges: Any,
    molecule: Any,
    method: str,
) -> PopulationSummary:
    """Adapt method-native net atomic Mulliken charges for sidecar output.

    Semiempirical SCC engines define their minimal Slater-orbital populations
    inside the Hamiltonian and expose conventional net charges directly
    (positive means electron-deficient).  Reusing the molecular Gaussian-AO
    property helpers would require a fictitious :class:`BasisSet`, so this
    adapter emits the native Mulliken values and marks every unavailable
    analysis explicitly.
    """
    values = np.asarray(charges, dtype=float).reshape(-1)
    atoms = list(molecule.atoms)
    if values.size != len(atoms):
        raise ValueError(
            f"{method} native Mulliken charge count {values.size} does not "
            f"match atom count {len(atoms)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{method} native Mulliken charges are non-finite")

    molecular_charge = float(getattr(molecule, "charge", 0.0))
    if not np.isclose(
        float(values.sum()), molecular_charge, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(
            f"{method} native Mulliken charges sum to {values.sum():+.8f}, "
            f"expected molecular charge {molecular_charge:+.8f}"
        )

    from .xyz import _symbol

    out = PopulationSummary()
    out.mulliken_atoms = [
        (index, _symbol(int(atom.Z)), float(atom.Z), float(values[index]))
        for index, atom in enumerate(atoms)
    ]
    unavailable = (
        f"unsupported: {method} native population output currently provides "
        "Mulliken atomic charges only"
    )
    for section in (
        "loewdin",
        "hirshfeld",
        "mayer",
        "wiberg",
        "npa",
        "dipole",
    ):
        out.errors[section] = unavailable
    return out


# ---------------------------------------------------------------------- #
# Formatting                                                             #
# ---------------------------------------------------------------------- #


def format_population_txt(summary: PopulationSummary) -> str:
    """Render the summary as a tab-separated tabular text file.

    Sections (in order): Mulliken charges, Löwdin charges, Hirshfeld charges,
    Mayer bond orders (top-N), dipole moment. Each section opens with a
    comment header, then a tab-separated body. A blank line separates
    sections. Sections that errored emit a single
    ``# <section>: N/A -- <error>`` line so the file still loads
    cleanly into a column-oriented parser that uses ``#`` as the
    comment marker (pandas, awk, ...).
    """
    parts: list[str] = [
        "# vibe-qc population / properties dump",
        "# Sections: mulliken charges, löwdin charges,",
        "# hirshfeld charges, mayer bond orders (top-N), dipole moment.",
        "# Tab-separated. Lines starting with '#' are comments.",
        "",
    ]

    parts.append("# === Mulliken atomic charges ===")
    if summary.mulliken_atoms:
        parts.append("# idx\tsymbol\tZ\tcharge")
        for i, sym, z, q in summary.mulliken_atoms:
            parts.append(f"{i}\t{sym}\t{z:.1f}\t{q:+.6f}")
    else:
        err = summary.errors.get("mulliken", "no data")
        parts.append(f"# mulliken: N/A -- {err}")
    parts.append("")

    parts.append("# === Löwdin atomic charges ===")
    if summary.loewdin_atoms:
        parts.append("# idx\tsymbol\tZ\tcharge")
        for i, sym, z, q in summary.loewdin_atoms:
            parts.append(f"{i}\t{sym}\t{z:.1f}\t{q:+.6f}")
    else:
        err = summary.errors.get("loewdin", "no data")
        parts.append(f"# loewdin: N/A -- {err}")
    parts.append("")

    parts.append("# === Hirshfeld atomic charges ===")
    if summary.hirshfeld_atoms:
        parts.append("# idx\tsymbol\tZ\tcharge")
        for i, sym, z, q in summary.hirshfeld_atoms:
            parts.append(f"{i}\t{sym}\t{z:.1f}\t{q:+.6f}")
    else:
        err = summary.errors.get("hirshfeld", "no data")
        parts.append(f"# hirshfeld: N/A -- {err}")
    parts.append("")

    parts.append("# === Mayer bond orders (top entries) ===")
    if summary.mayer_bonds:
        parts.append("# i\tj\tsym_i\tsym_j\torder")
        for i, j, si, sj, o in summary.mayer_bonds:
            parts.append(f"{i}\t{j}\t{si}\t{sj}\t{o:.4f}")
    else:
        err = summary.errors.get("mayer", "no bonds above threshold")
        parts.append(f"# mayer: N/A -- {err}")
    parts.append("")

    parts.append("# === Wiberg bond indices (top entries) ===")
    if summary.wiberg_bonds:
        parts.append("# i\tj\tsym_i\tsym_j\torder")
        for i, j, si, sj, o in summary.wiberg_bonds:
            parts.append(f"{i}\t{j}\t{si}\t{sj}\t{o:.4f}")
    else:
        err = summary.errors.get("wiberg", "no bonds above threshold")
        parts.append(f"# wiberg: N/A -- {err}")
    parts.append("")

    parts.append("# === NPA atomic charges ===")
    if summary.npa_atoms:
        parts.append("# idx\tsymbol\tZ\tcharge")
        for i, sym, z, q in summary.npa_atoms:
            parts.append(f"{i}\t{sym}\t{z:.1f}\t{q:+.6f}")
    else:
        err = summary.errors.get("npa", "no data")
        parts.append(f"# npa: N/A -- {err}")
    parts.append("")

    parts.append("# === Dipole moment ===")
    if summary.dipole is not None:
        d = summary.dipole
        parts.append(f"# x_ebohr\ty_ebohr\tz_ebohr\ttotal_ebohr\ttotal_debye")
        parts.append(
            f"{d['x_ebohr']:.6f}\t{d['y_ebohr']:.6f}\t"
            f"{d['z_ebohr']:.6f}\t{d['total_ebohr']:.6f}\t"
            f"{d['total_debye']:.6f}"
        )
        parts.append(
            f"# origin_bohr = "
            f"({d['origin_bohr'][0]:.6f}, "
            f"{d['origin_bohr'][1]:.6f}, "
            f"{d['origin_bohr'][2]:.6f})"
        )
    else:
        err = summary.errors.get("dipole", "no data")
        parts.append(f"# dipole: N/A -- {err}")
    parts.append("")
    return "\n".join(parts)


def format_population_json(summary: PopulationSummary) -> str:
    """Render the summary as a JSON object, suitable for
    ``json.load``."""
    body: dict[str, Any] = {}
    body["mulliken"] = [
        {"idx": i, "symbol": sym, "Z": z, "charge": q}
        for i, sym, z, q in summary.mulliken_atoms
    ]
    body["loewdin"] = [
        {"idx": i, "symbol": sym, "Z": z, "charge": q}
        for i, sym, z, q in summary.loewdin_atoms
    ]
    body["hirshfeld"] = [
        {"idx": i, "symbol": sym, "Z": z, "charge": q}
        for i, sym, z, q in summary.hirshfeld_atoms
    ]
    body["mayer"] = [
        {"i": i, "j": j, "sym_i": si, "sym_j": sj, "order": o}
        for i, j, si, sj, o in summary.mayer_bonds
    ]
    body["wiberg"] = [
        {"i": i, "j": j, "sym_i": si, "sym_j": sj, "order": o}
        for i, j, si, sj, o in summary.wiberg_bonds
    ]
    body["npa"] = [
        {"idx": i, "symbol": sym, "Z": z, "charge": q}
        for i, sym, z, q in summary.npa_atoms
    ]
    body["dipole"] = summary.dipole
    body["errors"] = dict(summary.errors)
    return json.dumps(body, indent=2, sort_keys=False)


# ---------------------------------------------------------------------- #
# Writers                                                                #
# ---------------------------------------------------------------------- #


def write_population(
    stem: os.PathLike | str,
    result: Any,
    basis: Any,
    molecule: Any,
) -> tuple[Path, Path]:
    """Compute + write ``{stem}.population.txt`` and
    ``{stem}.population.json``. Returns the two written paths."""
    summary = compute_population_summary(result, basis, molecule)
    return write_population_summary(stem, summary)


def write_population_summary(
    stem: os.PathLike | str,
    summary: PopulationSummary,
) -> tuple[Path, Path]:
    """Write the txt + json files from an already-computed
    :class:`PopulationSummary`. Useful when the caller wants to
    inspect / extend the summary before writing."""
    s = Path(os.fspath(stem))
    txt_path = s.parent / (s.name + ".population.txt")
    json_path = s.parent / (s.name + ".population.json")
    s.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(format_population_txt(summary), encoding="utf-8")
    json_path.write_text(format_population_json(summary), encoding="utf-8")
    return (txt_path, json_path)
