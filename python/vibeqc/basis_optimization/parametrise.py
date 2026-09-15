"""Pack/unpack a basis <-> flat parameter vector.

The optimiser sees a 1-D ``np.ndarray``; the SCF wants a
:class:`vibeqc.basis_crystal.CrystalAtomBasis` per element. This
module bridges the two.

A parametrisation is the user's declaration of WHICH numbers in the
basis are free to vary and HOW the optimiser sees them
(linear vs log-space, bounds). Everything else stays at its starting
value across the optimisation.

Typical use
-----------

>>> from vibeqc.basis_crystal import parse_crystal_atom_basis_file
>>> H = parse_crystal_atom_basis_file("python/vibeqc/basis_library/sources/pob-TZVP/01_H")
>>> p = BasisParametrisation(
...     atoms={"H": H},
...     free=[FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
...                    field="exponent", transform=Transform.LOG)],
... )
>>> x0 = p.pack()        # current parameter vector
>>> updated = p.unpack(x0 * 1.1)   # nudge upward, returns new atoms dict

The :class:`BasisParametrisation` keeps the *original* basis as
immutable reference data. Each ``unpack`` produces a fresh deep
copy with the free fields overwritten -- the original is never
mutated, so an optimiser can call ``unpack`` repeatedly with
different parameter vectors without leaking state.
"""

from __future__ import annotations

import copy
import enum
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..basis_crystal import CrystalAtomBasis


class Transform(enum.Enum):
    """How the optimiser sees a free parameter.

    ``LINEAR`` -- pass the value unchanged. Suitable for contraction
    coefficients that may be small or negative.

    ``LOG`` -- pass ``log(value)`` to the optimiser; map back via
    ``exp``. The natural transform for Gaussian exponents (always
    positive, span many orders of magnitude). All pob optimisations
    in PT2013 / VO2019 work in log-space on the exponents.
    """

    LINEAR = "linear"
    LOG = "log"

    def to_optim(self, x: float) -> float:
        if self is Transform.LOG:
            if x <= 0.0:
                raise ValueError(f"LOG transform requires positive value, got {x}")
            return math.log(x)
        return x

    def from_optim(self, y: float) -> float:
        if self is Transform.LOG:
            return math.exp(y)
        return y


@dataclass(frozen=True)
class FreeSpec:
    """One free parameter in the optimisation.

    Identifies a specific number in the basis (which element, which
    shell within that element, which primitive within that shell, and
    which field -- the exponent or one of the contraction coefficients)
    and how the optimiser sees it.

    Attributes
    ----------
    symbol : str
        Element symbol, e.g. ``"H"``, ``"Li"``.
    shell_idx : int
        Index into ``atoms[symbol].shells`` (0-based).
    prim_idx : int
        Index into ``atoms[symbol].shells[shell_idx].primitives`` (0-based).
    field : str
        ``"exponent"``, ``"coeff"`` (single-coefficient shells), or
        ``"coeff_s"``/``"coeff_p"`` for SP shells.
    transform : Transform
        How the optimiser sees this parameter.
    bounds : tuple[Optional[float], Optional[float]]
        Lower / upper bound in physical (post-transform) space.
        ``None`` means unbounded on that side. The pob recipe uses
        a hard lower bound of 0.10 (PT2013) or 0.15 / 0.12 (VO2019)
        on exponents to defend against linear dependence.
    label : Optional[str]
        Human-readable label for logs. Defaults to a generated name.
    """

    symbol: str
    shell_idx: int
    prim_idx: int
    field: str
    transform: Transform = Transform.LINEAR
    bounds: tuple[Optional[float], Optional[float]] = (None, None)
    label: Optional[str] = None

    def display(self) -> str:
        return self.label or (
            f"{self.symbol}.shell{self.shell_idx}.prim{self.prim_idx}.{self.field}"
        )


@dataclass
class BasisParametrisation:
    """A basis (one or more atoms) with a designated set of free numbers.

    The class is constructed from an immutable reference basis plus a
    list of free specifications. The reference basis is never mutated;
    :meth:`unpack` returns a fresh copy each call.
    """

    atoms: dict[str, "CrystalAtomBasis"]
    free: list[FreeSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Defensive copy so external mutations of the input dict don't
        # leak into the reference.
        self.atoms = {k: copy.deepcopy(v) for k, v in self.atoms.items()}
        self._validate_free()

    def _validate_free(self) -> None:
        for spec in self.free:
            if spec.symbol not in self.atoms:
                raise KeyError(
                    f"FreeSpec references unknown element {spec.symbol!r}; "
                    f"have {sorted(self.atoms)}"
                )
            atom = self.atoms[spec.symbol]
            shells = list(atom.shells)
            if not (0 <= spec.shell_idx < len(shells)):
                raise IndexError(
                    f"{spec.display()}: shell_idx {spec.shell_idx} "
                    f"out of range [0, {len(shells)})"
                )
            self._read_field(spec, atom)  # raises if prim_idx / field is invalid

    # ---- field accessors ----------------------------------------------------
    #
    # CrystalShell stores primitives as parallel arrays:
    #   shell.exponents       -- list[float], length n_primitives
    #   shell.coefficients    -- list[float], length n_primitives
    #                           (s-side for SP shells; only side for S/P/D/F/G)
    #   shell.coefficients_p  -- list[float], length n_primitives (SP only)

    @staticmethod
    def _validate_prim(spec: FreeSpec, atom: "CrystalAtomBasis") -> None:
        shell = atom.shells[spec.shell_idx]
        if not (0 <= spec.prim_idx < len(shell.exponents)):
            raise IndexError(
                f"{spec.display()}: prim_idx {spec.prim_idx} out of range "
                f"[0, {len(shell.exponents)})"
            )

    @staticmethod
    def _read_field(spec: FreeSpec, atom: "CrystalAtomBasis") -> float:
        BasisParametrisation._validate_prim(spec, atom)
        shell = atom.shells[spec.shell_idx]
        is_sp = shell.shell_type == "SP"
        if spec.field == "exponent":
            return float(shell.exponents[spec.prim_idx])
        if spec.field == "coeff":
            if is_sp:
                raise ValueError(
                    f"{spec.display()}: 'coeff' is ambiguous on SP shells; "
                    f"use 'coeff_s' or 'coeff_p'"
                )
            return float(shell.coefficients[spec.prim_idx])
        if spec.field == "coeff_s":
            if not is_sp:
                raise ValueError(
                    f"{spec.display()}: 'coeff_s' requires SP shell, "
                    f"got shell_type={shell.shell_type!r}"
                )
            return float(shell.coefficients[spec.prim_idx])
        if spec.field == "coeff_p":
            if not is_sp:
                raise ValueError(
                    f"{spec.display()}: 'coeff_p' requires SP shell, "
                    f"got shell_type={shell.shell_type!r}"
                )
            return float(shell.coefficients_p[spec.prim_idx])
        raise ValueError(f"{spec.display()}: unknown field {spec.field!r}")

    @staticmethod
    def _write_field(spec: FreeSpec, atom: "CrystalAtomBasis", value: float) -> None:
        BasisParametrisation._validate_prim(spec, atom)
        shell = atom.shells[spec.shell_idx]
        if spec.field == "exponent":
            shell.exponents[spec.prim_idx] = float(value)
            return
        if spec.field in ("coeff", "coeff_s"):
            shell.coefficients[spec.prim_idx] = float(value)
            return
        if spec.field == "coeff_p":
            shell.coefficients_p[spec.prim_idx] = float(value)
            return
        raise ValueError(f"{spec.display()}: unknown field {spec.field!r}")

    # ---- pack / unpack ------------------------------------------------------

    @property
    def atoms_keys(self) -> list[str]:
        """Stable element-symbol ordering for callers iterating the atoms dict.

        Returns the keys of :attr:`atoms` as a list in insertion order. Used
        by recipe code that needs to zip the per-atom basis dict back into a
        positional list for emit_crystal / inline-deck generation.
        """
        return list(self.atoms.keys())

    def pack(self) -> np.ndarray:
        """Return current values of free parameters in optimiser-space."""
        out = np.empty(len(self.free), dtype=float)
        for i, spec in enumerate(self.free):
            phys = self._read_field(spec, self.atoms[spec.symbol])
            out[i] = spec.transform.to_optim(phys)
        return out

    def unpack(self, x: np.ndarray) -> dict[str, "CrystalAtomBasis"]:
        """Return a fresh atoms dict with free parameters set from `x`."""
        if x.shape != (len(self.free),):
            raise ValueError(
                f"unpack: x has shape {x.shape}, expected ({len(self.free)},)"
            )
        out = {k: copy.deepcopy(v) for k, v in self.atoms.items()}
        for i, spec in enumerate(self.free):
            phys = spec.transform.from_optim(float(x[i]))
            lo, hi = spec.bounds
            if lo is not None and phys < lo:
                raise ValueError(
                    f"{spec.display()}: physical value {phys} below lower bound {lo}"
                )
            if hi is not None and phys > hi:
                raise ValueError(
                    f"{spec.display()}: physical value {phys} above upper bound {hi}"
                )
            self._write_field(spec, out[spec.symbol], phys)
        return out

    def optim_bounds(self) -> list[tuple[Optional[float], Optional[float]]]:
        """Bounds in optimiser space (after the per-parameter transform)."""
        result: list[tuple[Optional[float], Optional[float]]] = []
        for spec in self.free:
            lo, hi = spec.bounds
            lo_o = spec.transform.to_optim(lo) if lo is not None else None
            hi_o = spec.transform.to_optim(hi) if hi is not None else None
            result.append((lo_o, hi_o))
        return result

    def labels(self) -> list[str]:
        return [spec.display() for spec in self.free]

    def __len__(self) -> int:
        return len(self.free)
