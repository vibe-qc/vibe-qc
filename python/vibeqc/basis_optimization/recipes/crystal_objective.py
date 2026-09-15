"""Reusable CRYSTAL14 objective factories for basis-set optimization.

Two factory functions:

* :func:`make_crystal_objective` -- single-structure energy evaluation.
* :func:`make_multi_crystal_objective` -- joint minimization across
  multiple structures.  This is how pob-TZVP was created: the same
  basis is evaluated on many compounds simultaneously, and the
  optimizer minimises S wᵢ . Eᵢ(x).

Both support isolated atoms (via ``emit_input_atom``) and periodic
crystals (via ``emit_input_inline``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from vibe_basis.backends.crystal import emit_input_inline, parse_output_file
from vibe_basis.backends.crystal_atom import emit_input_atom
from vibe_basis.transports.base import Transport, TransportError

from ...basis_crystal import emit_crystal
from ..parametrise import BasisParametrisation


def make_crystal_objective(
    parametrisation: BasisParametrisation,
    transport: Transport,
    *,
    kind: str = "crystal",
    structure=None,
    Z: int | None = None,
    method: str = "rhf",
    crystal_wrapper: str = "crystal",
    cpus: int = 4,
    wall_time_s: int = 7200,
    timeout_s: float = 86_400.0,
    workdir_prefix: str = "objective",
) -> Callable[[np.ndarray], float]:
    """Build a single-structure objective function.

    On each call: unpack basis -> emit .d12 -> submit -> parse -> return
    energy (Ha).  Returns ``np.inf`` on failure.
    """
    if kind == "atom" and Z is None:
        raise ValueError("Z is required for kind='atom'")
    if kind == "crystal" and structure is None:
        raise ValueError("structure is required for kind='crystal'")

    n_calls = [0]

    def objective(x: np.ndarray) -> float:
        n_calls[0] += 1
        iteration = n_calls[0]

        atoms = parametrisation.unpack(x)
        basis_list = [atoms[k] for k in parametrisation.atoms_keys]
        basis_text = emit_crystal(basis_list)

        if kind == "atom":
            deck = emit_input_atom(Z, basis_text, method=method)
        else:
            deck = emit_input_inline(structure, basis_text, method=method)
            if deck is None:
                return float("inf")

        wd = Path(f"{workdir_prefix}_iter_{iteration:03d}")
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        tag = structure.name if kind == "crystal" else f"Z{Z}"
        d12_name = f"{tag}.d12"
        (wd / d12_name).write_text(deck)

        try:
            job = transport.run(
                wd,
                command=[crystal_wrapper, d12_name],
                dest=wd / "fetched",
                cpus=cpus,
                wall_time_s=wall_time_s,
                label=f"obj/{tag}/iter{iteration:03d}",
                timeout_s=timeout_s,
            )
        except (TransportError, Exception):
            return float("inf")

        out_path = job.output_dir / f"{tag}.out"
        try:
            parsed = parse_output_file(out_path)
        except FileNotFoundError:
            return float("inf")

        if not parsed.ok or parsed.energy is None:
            return float("inf")
        return parsed.energy

    return objective


# ---------------------------------------------------------------------------
# Multi-system joint objective -- how pob-TZVP was created
# ---------------------------------------------------------------------------


def make_multi_crystal_objective(
    parametrisation: BasisParametrisation,
    transport: Transport,
    structures: list,
    *,
    method: str = "rhf",
    weights: list[float] | None = None,
    crystal_wrapper: str = "crystal",
    cpus: int = 4,
    wall_time_s: int = 7200,
    timeout_s: float = 86_400.0,
    workdir_prefix: str = "multi",
) -> Callable[[np.ndarray], float]:
    """Build a multi-system joint objective for basis-set optimization.

    **This is how pob-TZVP was produced.**  A single basis is
    evaluated across many compounds simultaneously, and the optimizer
    minimises the weighted sum of total energies:

        L(x) = Sᵢ wᵢ . Eᵢ(x)        where x = basis parameters

    The **same basis (x)** is used for every compound -- the optimizer
    finds the exponents and coefficients that work well *on average*
    across all bonding situations in the test set.  This produces a
    transferable, consistent basis that sacrifices per-system optimality
    for universal applicability.  A basis optimised only for MgO would
    give a lower MgO energy, but would perform poorly on CaO or LiF.
    The joint optimum is the best compromise.

    Failed evaluations (non-converged SCF, transport error) return
    ``np.inf`` so BOBYQA routes around infeasible parameter regions.
    If any single compound fails, the entire evaluation returns inf --
    the optimizer treats the whole parameter vector as infeasible.

    Parameters
    ----------
    parametrisation
        :class:`BasisParametrisation` covering all elements in the
        test set.  For example, for the 13 cubic ionics (LiCl, NaCl,
        LiF, ..., KH), the atoms dict needs: H, Li, Na, K, F, Cl,
        Mg, Ca, O.
    transport
        Transport for CRYSTAL14 submission on each compound.
    structures
        List of :class:`~vibe_basis.io.structures.Structure` objects
        from the structure database.
    method
        ``"rhf"``, ``"hf"``, or a DFT functional keyword (``"pw1pw"``,
        ``"pbe"``, ...).
    weights
        Per-compound weights.  ``None`` (default) means equal
        weights (1.0 for all) -- the pob convention.  Use non-uniform
        weights to emphasise certain bonding situations.
    crystal_wrapper
        Path to ``run-crystal.sh`` on the target machine.
    cpus, wall_time_s, timeout_s
        Passed through to ``transport.run()`` for each compound.

    Returns
    -------
    Callable[[np.ndarray], float]
        Objective: parameter vector -> S wᵢ . Eᵢ in Hartree.
        Returns ``np.inf`` if any compound fails.
    """
    if not structures:
        raise ValueError("at least one structure required")

    ws = weights or [1.0] * len(structures)
    if len(ws) != len(structures):
        raise ValueError(
            f"weights length {len(ws)} != structures length {len(structures)}"
        )

    n_calls = [0]

    def objective(x: np.ndarray) -> float:
        n_calls[0] += 1
        iteration = n_calls[0]

        # Unpack -> ONE set of basis atoms for ALL compounds.
        atoms = parametrisation.unpack(x)
        basis_list = [atoms[k] for k in parametrisation.atoms_keys]
        basis_text = emit_crystal(basis_list)

        total = 0.0

        for s, w in zip(structures, ws):
            deck = emit_input_inline(s, basis_text, method=method)
            if deck is None:
                return float("inf")

            wd = Path(f"{workdir_prefix}_{iteration:03d}") / s.name
            if wd.exists():
                shutil.rmtree(wd)
            wd.mkdir(parents=True)
            d12_name = f"{s.name}.d12"
            (wd / d12_name).write_text(deck)

            try:
                job = transport.run(
                    wd,
                    command=[crystal_wrapper, d12_name],
                    dest=wd / "fetched",
                    cpus=cpus,
                    wall_time_s=wall_time_s,
                    label=f"multi/{s.name}/iter{iteration:03d}",
                    timeout_s=timeout_s,
                )
            except (TransportError, Exception):
                return float("inf")

            out_path = job.output_dir / f"{s.name}.out"
            try:
                parsed = parse_output_file(out_path)
            except FileNotFoundError:
                return float("inf")

            if not parsed.ok or parsed.energy is None:
                return float("inf")

            total += w * parsed.energy

        return total

    return objective
