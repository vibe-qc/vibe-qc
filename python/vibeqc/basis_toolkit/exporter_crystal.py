"""Export BasisSetData to CRYSTAL basis-set input, shells and ECPs.

CRYSTAL reads one block per atomic number after the geometry (CRYSTAL23
manual, "Basis set input", pp. 23-25): ``Z NSHL`` (``200+Z`` when an
effective core potential follows), then per shell ``ITYB LAT NG CHE SCAL``
and ``NG`` rows ``EXP COE1 [COE2]``. An ECP is the ``INPUT`` block of
"Effective core pseudo-potentials" (pp. 83-84). :func:`to_crystal` writes
those blocks, closed by `` 99 0``, through
:func:`vibeqc.basis_crystal.emit_crystal`, the writer the basis optimizer's
CRYSTAL engines use; paste the text after the geometry's ``END`` and add
the basis block's own ``END``.

Conventions
-----------
* LAT is 0 (S), 1 (SP), 2 (P), 3 (D), 4 (F), 5 (G, CRYSTAL22 and later).
  CRYSTAL has no shell above g, and allows at most 10 primitives in an
  s, sp or p shell and 6 in a d shell (``NG`` limits, p. 25); a record
  beyond either is refused.
* Exponents are written as they are (``SCAL`` 0.0, "as written" for a
  general basis) and every number with shortest round-trip precision.
* ECP terms: ``NKL`` is the literal power of r, so a record's
  Gaussian-format ``power`` n becomes ``NKL = n - 2``. The local channel's
  terms are CRYSTAL's ``M`` local terms (zero-coefficient placeholders are
  dropped); the projected channels are ``M0``-``M4``, so a projector above
  g is refused. ``ZNUC`` is ``Z - n_core``.

CHE
---
CRYSTAL wants a formal electron charge per shell, and the cell's electron
count is the sum of them (manual, basis set input, note 12). A record
carries none, so :func:`to_crystal` assigns the neutral ground state: the
per-l electron totals the SAD guess pins (``ground_config_per_l`` in
``cpp/src/guess.cpp``: Madelung filling plus the NIST anomalies) minus the
ECP's closed core (``valence_config_per_l`` there), filled into the
element's shells of each l in record order, each up to 2(2l+1). Pass
``occupancies`` for anything else (an ECP core outside that table, an
ion, another guess).

Lossy fields
------------
CRYSTAL input carries no metadata, and its d, f and g shells are always
spherical (notes 5-7, p. 30), so a Cartesian record changes meaning.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ._element_data import ATOMIC_NUMBER
from .model import HarmonicType, LossReport

if TYPE_CHECKING:
    from vibeqc.basis_crystal import CrystalAtomBasis

    from .model import BasisSetData, ECPEntry, ElementBasis

__all__ = [
    "to_crystal",
    "to_crystal_atoms",
    "to_crystal_file",
    "loss_report_crystal",
    "ground_state_electrons_per_l",
]

_SHELL_TYPE = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G"}
#: CRYSTAL's documented NG limits for a general basis (p. 25), by LAT.
_MAX_PRIMITIVES = {"S": 10, "SP": 10, "P": 10, "D": 6}

# The two tables below mirror cpp/src/guess.cpp (ground_config_per_l and
# valence_config_per_l); tests/test_basis_formats_crystal.py reads that file
# and pins them equal.

#: Madelung filling order by l: 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
#: 7s 5f 6d 7p.
_FILL_L = (0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1)

#: Neutral ground states that deviate from Madelung filling (NIST Atomic
#: Spectra Database): full {s, p, d, f} electron totals.
_ANOMALIES = {
    24: (7, 12, 5, 0),
    29: (7, 12, 10, 0),
    41: (9, 18, 14, 0),
    42: (9, 18, 15, 0),
    44: (9, 18, 17, 0),
    45: (9, 18, 18, 0),
    46: (8, 18, 20, 0),
    47: (9, 18, 20, 0),
    57: (12, 24, 21, 0),
    58: (12, 24, 21, 1),
    64: (12, 24, 21, 7),
    78: (11, 24, 29, 14),
    79: (11, 24, 30, 14),
    89: (14, 30, 31, 14),
    90: (14, 30, 32, 14),
    91: (14, 30, 31, 16),
    92: (14, 30, 31, 17),
    93: (14, 30, 31, 18),
    96: (14, 30, 31, 21),
    103: (14, 31, 30, 28),
}

#: Closed ECP cores as {s, p, d, f} electron totals: 28 is [Ar]3d10, 60 is
#: [Kr]4d10 4f14 (Andrae et al. 1990, Sec. 1-2).
_CLOSED_CORES = {
    0: (0, 0, 0, 0),
    2: (2, 0, 0, 0),
    10: (4, 6, 0, 0),
    18: (6, 12, 0, 0),
    28: (6, 12, 10, 0),
    36: (8, 18, 10, 0),
    46: (8, 18, 20, 0),
    54: (10, 24, 20, 0),
    60: (8, 18, 20, 14),
    68: (10, 24, 20, 14),
    78: (10, 24, 30, 14),
    80: (12, 24, 30, 14),
    86: (12, 30, 30, 14),
}


def ground_state_electrons_per_l(z: int, n_core: int = 0) -> tuple[int, int, int, int]:
    """Neutral ground-state {s, p, d, f} electron totals of element *z*
    outside an ECP core of *n_core* electrons."""
    totals = _ANOMALIES.get(z)
    if totals is None:
        per_l = [0, 0, 0, 0]
        remaining = z
        for l in _FILL_L:
            if remaining <= 0:
                break
            fill = min(2 * (2 * l + 1), remaining)
            per_l[l] += fill
            remaining -= fill
        totals = tuple(per_l)
    core = _CLOSED_CORES.get(n_core)
    if core is None:
        raise ValueError(
            f"Z={z}: no closed-core occupation for an ECP with {n_core} core "
            f"electrons; pass occupancies"
        )
    valence = tuple(t - c for t, c in zip(totals, core))
    if min(valence) < 0:
        raise ValueError(f"Z={z}: an ECP core of {n_core} electrons does not fit this element")
    return valence


# ---------------------------------------------------------------------------
# Loss report
# ---------------------------------------------------------------------------


def loss_report_crystal(data: BasisSetData) -> LossReport:
    """What CRYSTAL basis-set input cannot carry."""
    report = LossReport()
    for field in ("name", "description", "role", "basis_family", "references", "provenance"):
        report.add_loss(field, "CRYSTAL basis-set input carries no metadata")
    if any(
        shell.harmonic_type == HarmonicType.CARTESIAN
        for element in data.elements.values()
        for shell in element.shells
    ):
        report.add_loss(
            "harmonic_type",
            "CRYSTAL d, f and g shells are spherical; Cartesian shells change meaning",
        )
    return report


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _ecp_entry(data: BasisSetData, element: ElementBasis) -> ECPEntry | None:
    if element.ecp_id is None or not data.ecps:
        return None
    return data.ecps.get(element.ecp_id)


def _crystal_ecp(sym: str, z: int, entry: ECPEntry):
    from vibeqc.basis_crystal import CrystalECP, CrystalECPTerm

    lmax = int(entry.angular_momentum_max)
    terms: list[CrystalECPTerm] = []
    for potential in entry.potentials:
        l = int(potential.angular_momentum)
        if l != lmax and l > 4:
            raise ValueError(f"{sym}: CRYSTAL ECPs project l = 0..4 only; this one projects l = {l}")
        for term in potential.terms:
            if l == lmax and term.coefficient == 0.0:
                continue  # the zero local placeholder; CRYSTAL's M may be 0
            terms.append(
                CrystalECPTerm(
                    ell="local" if l == lmax else l,
                    alpha=float(term.exponent),
                    coefficient=float(term.coefficient),
                    n_pow=int(term.power) - 2,
                )
            )
    counts = [sum(1 for t in terms if t.ell == l) for l in range(5)]
    return CrystalECP(
        znuc=float(z - int(entry.n_core_electrons)),
        m_local=sum(1 for t in terms if t.ell == "local"),
        m_per_ell=tuple(counts),
        terms=terms,
    )


def _assign_che(sym: str, shells, electrons: Sequence[int]) -> list[float]:
    """Fill *electrons* (per l) into *shells* in order, 2(2l+1) per shell."""
    left = list(electrons) + [0] * 8
    che: list[float] = []
    for shell in shells:
        total = 0
        for l in shell.angular_momentum:
            take = min(2 * (2 * l + 1), left[l])
            left[l] -= take
            total += take
        che.append(float(total))
    if any(left):
        raise ValueError(
            f"{sym}: the shells cannot hold the ground-state electrons "
            f"(left over per l: {left[:4]}); pass occupancies"
        )
    return che


def to_crystal_atoms(
    data: BasisSetData,
    elements: Iterable[str] | None = None,
    *,
    occupancies: Mapping[str, Sequence[float]] | None = None,
) -> list[CrystalAtomBasis]:
    """The record's elements as :class:`~vibeqc.basis_crystal.CrystalAtomBasis`,
    ordered by atomic number, with CHE assigned (see the module docstring).

    *elements* limits the output to those symbols. *occupancies* maps a
    symbol to one CHE per shell in record order (an SP shell is one shell).
    """
    from vibeqc.basis_crystal import CrystalAtomBasis, CrystalShell

    wanted = list(data.elements) if elements is None else list(elements)
    missing = [s for s in wanted if s not in data.elements]
    if missing:
        raise KeyError(f"{data.name}: no basis for {missing}")
    atoms: list[CrystalAtomBasis] = []
    for sym in sorted(wanted, key=lambda s: ATOMIC_NUMBER.get(s, 999)):
        element = data.elements[sym]
        z = ATOMIC_NUMBER[sym]
        entry = _ecp_entry(data, element)
        n_core = int(entry.n_core_electrons) if entry is not None else 0
        if occupancies is not None and sym in occupancies:
            che = [float(c) for c in occupancies[sym]]
            if len(che) != len(element.shells):
                raise ValueError(f"{sym}: {len(che)} occupancies for {len(element.shells)} shells")
        else:
            che = _assign_che(sym, element.shells, ground_state_electrons_per_l(z, n_core))
        shells: list[CrystalShell] = []
        for shell, occupancy in zip(element.shells, che):
            am = list(shell.angular_momentum)
            prims = shell.primitives
            if am == [0, 1]:
                n = len(prims) // 2
                crystal = CrystalShell(
                    "SP", occupancy, 1.0,
                    exponents=[p.exponent for p in prims[:n]],
                    coefficients=[p.coefficient for p in prims[:n]],
                    coefficients_p=[p.coefficient for p in prims[n:]],
                )
            elif len(am) == 1 and am[0] in _SHELL_TYPE:
                crystal = CrystalShell(
                    _SHELL_TYPE[am[0]], occupancy, 1.0,
                    exponents=[p.exponent for p in prims],
                    coefficients=[p.coefficient for p in prims],
                )
            else:
                raise ValueError(f"{sym}: CRYSTAL has no shell type for angular momentum {am}")
            limit = _MAX_PRIMITIVES.get(crystal.shell_type)
            if limit is not None and crystal.n_primitives > limit:
                raise ValueError(
                    f"{sym}: a {crystal.shell_type} shell of {crystal.n_primitives} "
                    f"primitives; CRYSTAL reads at most {limit}"
                )
            shells.append(crystal)
        atom = CrystalAtomBasis(Z=z, has_ecp=entry is not None, shells=shells)
        if entry is not None:
            atom.ecp = _crystal_ecp(sym, z, entry)
        atoms.append(atom)
    return atoms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_crystal(
    data: BasisSetData,
    elements: Iterable[str] | None = None,
    *,
    occupancies: Mapping[str, Sequence[float]] | None = None,
) -> str:
    """CRYSTAL basis-set input of *data*: one block per element, ECPs
    included, closed by `` 99 0``."""
    from vibeqc.basis_crystal import emit_crystal

    return emit_crystal(to_crystal_atoms(data, elements, occupancies=occupancies)) + "\n"


def to_crystal_file(
    data: BasisSetData,
    path: str | Path,
    elements: Iterable[str] | None = None,
    *,
    occupancies: Mapping[str, Sequence[float]] | None = None,
) -> LossReport:
    """Write :func:`to_crystal` text to *path* and return the loss report."""
    Path(path).write_text(to_crystal(data, elements, occupancies=occupancies), encoding="utf-8")
    return loss_report_crystal(data)
