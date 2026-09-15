"""Basis-set primitive filtering for periodic SCF.

In a solid, the dense packing of atoms combined with the diffuse
Gaussian primitives in molecular-design basis sets (def2, cc-pVXZ,
6-31G, ...) routinely drives the AO overlap matrix into near-singular
or even non-positive-definite territory. The classical fix is to
*remove* the offending diffuse primitives at basis-construction time,
keeping only the tight-and-medium parts.

PySCF surfaces this as the ``Cell.exp_to_discard`` parameter -- set
``cell.exp_to_discard = 0.1`` and every Gaussian primitive with
exponent a < 0.1 is dropped before SCF starts. This module provides
the equivalent for vibe-qc:

.. code-block:: python

    import vibeqc as vq
    from vibeqc.basis_filter import filter_basis_by_exponent

    mol = vq.Molecule(...)
    basis = vq.BasisSet(mol, "def2-tzvp")
    filtered, report = filter_basis_by_exponent(basis, mol, exp_to_discard=0.1)
    print(f"dropped {report.n_primitives_dropped} primitives, "
          f"{report.n_shells_dropped} entire shells")
    # use ``filtered`` instead of ``basis`` for the SCF.

The "right" fix is usually a basis set *designed* for solids -- the
overlap matrix conditioning is then well-conditioned by construction
and no runtime filtering is needed. References:

* Peintinger, M. F., Vilela Oliveira, D., Bredow, T.
  *J. Comput. Chem.* **34**, 451 (2013) -- **pob-TZVP**, ships as
  ``"pob-tzvp"``.
* Vilela Oliveira, D., Laun, J., Peintinger, M. F., Bredow, T.
  *J. Comput. Chem.* **40**, 2364 (2019) -- **pob-TZVP-rev2**, ships as
  ``"pob-tzvp-rev2"`` and ``"pob-dzvp-rev2"``.
* Laun, J., Vilela Oliveira, D., Bredow, T.
  *J. Comput. Chem.* **39**, 1285 (2018) -- pob period-4 update.
* Laun, J., Bredow, T. *J. Comput. Chem.* **42**, 1064 (2021) -- pob
  sixth period.
* Laun, J., Bredow, T. *J. Comput. Chem.* **43**, 839 (2022) -- pob
  fifth-period rev2.
* VandeVondele, J., Hutter, J. *J. Chem. Phys.* **127**, 114105
  (2007) -- **MOLOPT**, optimised against a condition-number penalty.
* Ye, H.-Z., Berkelbach, T. C. *J. Chem. Theory Comput.* **18**, 1595
  (2022) -- **GTH-cc-pVXZ** for solids.

Implementation details
----------------------

The filter walks every contracted shell, drops primitives with
exponent below the threshold, and re-emits a synthesised ``.g94``
basis-set file in a temporary directory. The new BasisSet is then
loaded via libint's standard data-path mechanism (``LIBINT_DATA_PATH``
swapped temporarily to the temp dir, then restored).

Coefficient un-normalisation: ``ShellInfo.coefficients`` in vibe-qc
mirrors libint's internal contraction coefficients, which include
primitive normalisation. The .g94 format expects *raw* contraction
coefficients (re-normalised by libint at load), so we divide by

    N(a, l) = (2a / pi)^(3/4) . (4a)^(l/2) / √((2l-1)!!)

before emission. Verified against the standard
``basis_library/basis/sto-3g.g94`` for H (matches to four decimals).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ._primitive_norm import libint_primitive_norm as _libint_primitive_norm
from ._vibeqc_core import BasisSet, Molecule, ShellInfo


__all__ = [
    "BasisFilterReport",
    "DroppedPrimitive",
    "filter_basis_by_exponent",
    "clear_filtered_basis_cache",
    "format_basis_filter_report",
    "make_basis",
]


@dataclass
class DroppedPrimitive:
    """One Gaussian primitive that was dropped by the filter.

    Recorded so :func:`format_basis_filter_report` can print every
    drop verbatim to the SCF output -- the user-requested transparency:
    when ``exp_to_discard`` removes basis content, the user must be
    able to see *exactly* what was removed.
    """
    element_z: int
    element_symbol: str
    shell_index: int           # index within the canonical per-element
                               # shell list (0-based)
    shell_l: int               # angular momentum 0=s, 1=p, ...
    shell_letter: str          # "S" / "P" / "D" / ...
    primitive_index: int       # index within the shell (0-based)
    exponent: float            # bohr^-2
    coefficient_libint: float  # the libint-internal contraction coef
    threshold: float           # the exp_to_discard value at filter time


def make_basis(
    molecule: Molecule,
    name: str,
    *,
    exp_to_discard: Optional[float] = None,
) -> BasisSet:
    """Construct a :class:`BasisSet`, optionally filtering diffuse
    primitives -- the recommended user-facing entry point for periodic
    SCF.

    Equivalent to ``BasisSet(molecule, name)`` when ``exp_to_discard``
    is ``None`` (default). When ``exp_to_discard`` is positive, every
    primitive Gaussian with exponent below that threshold is dropped
    via :func:`filter_basis_by_exponent` before the basis is returned.

    Use this wrapper when you'd otherwise reach for PySCF's
    ``cell.exp_to_discard`` parameter -- same semantics, same
    recommended starting point (``exp_to_discard = 0.1`` for tight
    crystals with molecular bases).

    .. note::

        Filtering is a runtime fixup. The cleaner solution is to start
        with a basis set *designed* for solids: ``"pob-tzvp"``
        (Peintinger 2013), ``"pob-tzvp-rev2"`` (Vilela Oliveira 2019),
        or external MOLOPT / GTH-cc-pVXZ. All pob bases ship with
        vibe-qc -- see ``docs/tutorial/pob_tzvp.md``.
    """
    basis = BasisSet(molecule, name)
    if exp_to_discard is None:
        return basis
    filtered, _ = filter_basis_by_exponent(
        basis, molecule, exp_to_discard,
    )
    return filtered


def _filtered_basis_cache_dir() -> Path:
    """Resolve the on-disk cache directory for filtered .g94 files.

    Filtered bases need to live somewhere libint can re-load them by
    name -- the SAD initial-guess code calls
    ``libint2::BasisSet(name, atoms)`` again from inside the C++ SCF
    driver, so a one-shot temp-directory load is not sufficient.

    Resolution order:

    1. ``$VIBEQC_FILTERED_BASIS_DIR`` if set (user override / CI hook).
    2. ``<LIBINT_DATA_PATH>/basis/`` if writable -- the bundled
       ``python/vibeqc/basis_library/basis/`` ships there and libint
       finds it without further setup.
    3. ``$XDG_CACHE_HOME/vibeqc/basis_filter/`` (falls back to
       ``~/.cache/vibeqc/basis_filter/``) -- used when the bundled
       ``basis/`` dir isn't writable (e.g. system-wide install).
       This path also has to be added to ``LIBINT_DATA_PATH``;
       caller is expected to do that via :func:`_ensure_data_path`.

    All filtered .g94 files use the ``_vibeqc_filtered_`` prefix so
    they're trivial to identify and clean up via
    :func:`clear_filtered_basis_cache`.
    """
    override = os.environ.get("VIBEQC_FILTERED_BASIS_DIR")
    if override:
        target = Path(override)
        target.mkdir(parents=True, exist_ok=True)
        return target

    libint_data_path = os.environ.get("LIBINT_DATA_PATH")
    if libint_data_path:
        candidate = Path(libint_data_path) / "basis"
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        cache_root = Path(xdg)
    else:
        cache_root = Path.home() / ".cache"
    target = cache_root / "vibeqc" / "basis_filter"
    target.mkdir(parents=True, exist_ok=True)
    return target


def clear_filtered_basis_cache() -> int:
    """Remove every cached ``_vibeqc_filtered_*.g94`` file.

    Returns the number of files deleted. Safe to call between SCF
    runs; if a previously-filtered :class:`BasisSet` is still alive in
    memory its shell data was copied at construction time and is
    unaffected by the on-disk cleanup.
    """
    cache_dir = _filtered_basis_cache_dir()
    n = 0
    for p in cache_dir.glob("_vibeqc_filtered_*.g94"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


# Element symbol lookup by atomic number Z. Up to Z = 86 (radon)
# covers everything pob-TZVP handles plus most main-group + first/second
# transition rows. Beyond that we fall through to the standard
# IUPAC three-letter prefix and the user gets a clear error.
_SYMBOLS_BY_Z: Tuple[str, ...] = (
    "X",  # placeholder for Z=0
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
)


# Single-letter shell labels for .g94. l = 0..4 -> S, P, D, F, G.
_SHELL_LETTERS: Tuple[str, ...] = ("S", "P", "D", "F", "G", "H", "I")


@dataclass
class BasisFilterReport:
    """Summary returned alongside the filtered :class:`BasisSet`.

    The ``dropped_primitives`` list records every primitive that
    was removed -- element, shell, l-quantum number, exponent, and
    libint-stored coefficient. Use
    :func:`format_basis_filter_report` to print the full per-drop
    accounting to the SCF output.
    """

    threshold: float
    primitives_total: int
    primitives_kept: int
    n_primitives_dropped: int
    n_shells_dropped: int
    new_name: str
    original_name: str
    dropped_primitives: List[DroppedPrimitive] = field(default_factory=list)
    dropped_shells: List[Tuple[int, int]] = field(default_factory=list)
    # ^^^ list of ``(element_z, shell_index)`` for shells whose every
    # primitive was filtered out.

    @property
    def fraction_dropped(self) -> float:
        if self.primitives_total == 0:
            return 0.0
        return self.n_primitives_dropped / self.primitives_total


def _symbol_for_z(z: int) -> str:
    if 0 < z < len(_SYMBOLS_BY_Z):
        return _SYMBOLS_BY_Z[z]
    raise ValueError(
        f"filter_basis_by_exponent: no .g94 element symbol for Z = {z}. "
        "Filtering is currently restricted to Z <= 86; for heavier "
        "elements either extend _SYMBOLS_BY_Z or use a basis-set design "
        "appropriate for solids (pob-tzvp-rev2 / GTH-cc-pVXZ / MOLOPT)."
    )


# ---------------------------------------------------------------------------
# Per-element shell collection
# ---------------------------------------------------------------------------

def _collect_canonical_shells(
    basis: BasisSet,
    molecule: Molecule,
) -> Dict[int, List[ShellInfo]]:
    """Group shells by element, taking the *first occurrence* of each
    element as the canonical shell list.

    Rationale: in standard non-ECP bases, every atom of element ``Z``
    shares the same shell layout (only the origin differs). The .g94
    format groups by element, so we just need one canonical shell list
    per element. If two atoms of the same element somehow had
    different shells (mixed-basis users), we'd silently use the first
    -- flagged with a warning here would be useful for v0.7.x but
    deferred for now.
    """
    atoms = list(molecule.atoms)
    # First-seen atom index per element.
    first_atom_for_z: Dict[int, int] = {}
    for atom_idx, atom in enumerate(atoms):
        z = int(atom.Z)
        if z not in first_atom_for_z:
            first_atom_for_z[z] = atom_idx

    canonical: Dict[int, List[ShellInfo]] = {z: [] for z in first_atom_for_z}
    for sh in basis.shells():
        z = int(atoms[sh.atom_index].Z)
        if sh.atom_index == first_atom_for_z[z]:
            canonical[z].append(sh)
    return canonical


def _emit_g94(
    original_name: str,
    exp_to_discard: float,
    filtered: Dict[int, List[Tuple[int, List[float], List[float]]]],
) -> str:
    """Render a per-element .g94 string from filtered (l, exponents,
    raw_coeffs) tuples. Format mirrors libint's bundled bases:

    ::

        ! comment
        ****
        H     0
        S   3   1.00
              3.42525091             0.15432897
              0.62391373             0.53532814
              0.16885540             0.44463454
        ****
    """
    lines: List[str] = [
        f"! Generated by vibeqc.basis_filter -- filtered {original_name}",
        f"! exp_to_discard = {exp_to_discard}",
        "!",
    ]
    lines.extend(_source_basis_comments(original_name))
    lines.append("****")

    for z in sorted(filtered.keys()):
        shells = filtered[z]
        if not shells:
            continue
        symbol = _symbol_for_z(z)
        # Element header: "<Sym>     0".
        lines.append(f"{symbol}     0")
        for l, alphas, coefs in shells:
            if l < 0 or l >= len(_SHELL_LETTERS):
                raise ValueError(
                    f"filter_basis_by_exponent: shell letter not "
                    f"defined for l = {l}"
                )
            letter = _SHELL_LETTERS[l]
            n_prim = len(alphas)
            # ".g94" header line: "<L>   <n_prim>   <scale=1.00>".
            lines.append(f"{letter}   {n_prim}   1.00")
            for alpha, coef in zip(alphas, coefs):
                lines.append(f"      {alpha:.10f}             {coef:.10f}")
        lines.append("****")
    lines.append("")
    return "\n".join(lines)


def _source_basis_comments(original_name: str) -> List[str]:
    """Copy the source .g94 provenance header into a filtered basis.

    Filtered bases are derived data, so their generated files must retain the
    DOI/origin comments of the source basis.  Search the active libint data
    roots first, then the package's bundled library.  External bases without a
    readable source file simply keep the generated header above.
    """
    candidates: List[Path] = []
    for root in os.environ.get("LIBINT_DATA_PATH", "").split(os.pathsep):
        if root:
            candidates.append(Path(root) / "basis" / f"{original_name}.g94")
    candidates.append(
        Path(__file__).resolve().parent
        / "basis_library"
        / "basis"
        / f"{original_name}.g94"
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        comments: List[str] = []
        try:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped == "****":
                    break
                if stripped.startswith("!"):
                    comments.append(line)
        except OSError:
            continue
        if comments:
            return comments
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_basis_by_exponent(
    basis: BasisSet,
    molecule: Molecule,
    exp_to_discard: float,
    *,
    new_name: Optional[str] = None,
) -> Tuple[BasisSet, BasisFilterReport]:
    """Build a new :class:`BasisSet` with primitives ``a < exp_to_discard``
    removed.

    Parameters
    ----------
    basis
        Source basis to filter.
    molecule
        Molecule the basis is bound to. Needed to map shell origins
        back to elements when re-emitting the .g94 file.
    exp_to_discard
        Threshold (bohr⁻^2). Must be positive. Primitives with
        exponent below this are dropped. ``0.1`` is a sensible
        starting point for tight crystals (matches PySCF's typical
        recommendation).
    new_name
        Synthetic basis name. If ``None`` (default) a hash-based
        unique name is generated so successive filters don't collide.

    Returns
    -------
    new_basis
        A fresh :class:`BasisSet` without the discarded primitives.
        Use this in place of ``basis`` for SCF / property calls.
    report
        :class:`BasisFilterReport` with primitive / shell counts.

    Raises
    ------
    ValueError
        ``exp_to_discard <= 0``, all primitives filtered out, or an
        element with no shipped .g94 symbol mapping (Z > 86).
    """
    if exp_to_discard <= 0:
        raise ValueError(
            f"filter_basis_by_exponent: exp_to_discard must be > 0; "
            f"got {exp_to_discard}"
        )

    canonical = _collect_canonical_shells(basis, molecule)

    n_prim_total = 0
    n_prim_kept = 0
    n_shells_dropped = 0
    dropped_primitives: List[DroppedPrimitive] = []
    dropped_shells: List[Tuple[int, int]] = []

    # filtered[Z] = [(l, [a, ...], [c_raw, ...]), ...]
    filtered: Dict[int, List[Tuple[int, List[float], List[float]]]] = {}
    for z, shells in canonical.items():
        per_element: List[Tuple[int, List[float], List[float]]] = []
        symbol = _symbol_for_z(z)
        for shell_idx, sh in enumerate(shells):
            kept_alphas: List[float] = []
            kept_coefs_raw: List[float] = []
            shell_letter = (
                _SHELL_LETTERS[sh.l] if 0 <= sh.l < len(_SHELL_LETTERS)
                else "?"
            )
            for prim_idx, (alpha, coef_libint) in enumerate(
                zip(sh.exponents, sh.coefficients)
            ):
                n_prim_total += 1
                if alpha >= exp_to_discard:
                    norm = _libint_primitive_norm(alpha, sh.l)
                    if norm == 0.0:
                        continue   # extreme a; skip silently
                    kept_alphas.append(float(alpha))
                    kept_coefs_raw.append(float(coef_libint) / norm)
                    n_prim_kept += 1
                else:
                    # Record the drop for transparency.
                    dropped_primitives.append(DroppedPrimitive(
                        element_z=int(z),
                        element_symbol=symbol,
                        shell_index=int(shell_idx),
                        shell_l=int(sh.l),
                        shell_letter=shell_letter,
                        primitive_index=int(prim_idx),
                        exponent=float(alpha),
                        coefficient_libint=float(coef_libint),
                        threshold=float(exp_to_discard),
                    ))
            if kept_alphas:
                per_element.append((int(sh.l), kept_alphas, kept_coefs_raw))
            else:
                n_shells_dropped += 1
                dropped_shells.append((int(z), int(shell_idx)))
        filtered[z] = per_element

    if n_prim_kept == 0:
        raise ValueError(
            f"filter_basis_by_exponent: threshold {exp_to_discard} dropped "
            "every primitive -- pick a smaller value, or use a basis set "
            "designed for solids (pob-tzvp / pob-tzvp-rev2 / "
            "GTH-cc-pVXZ / MOLOPT)."
        )

    if new_name is None:
        digest = hashlib.sha1(
            f"{basis.name}@{exp_to_discard}".encode("utf-8")
        ).hexdigest()[:8]
        new_name = f"_vibeqc_filtered_{basis.name}_{digest}"

    g94_text = _emit_g94(basis.name, exp_to_discard, filtered)

    # Persist the synthesised .g94 in a cache location libint can
    # re-find by name. Necessary because the C++ SAD initial-guess
    # code (atomic_sad_density in cpp/src/guess.cpp) re-loads the
    # basis from LIBINT_DATA_PATH inside the SCF driver -- a one-shot
    # tempdir load isn't enough. See module docstring +
    # _filtered_basis_cache_dir for the resolution order. Cleanup is
    # the user's responsibility via :func:`clear_filtered_basis_cache`.
    cache_dir = _filtered_basis_cache_dir()
    target = cache_dir / f"{new_name}.g94"
    target.write_text(g94_text)

    # If the cache dir isn't already in LIBINT_DATA_PATH, the bundled
    # path resolution stays untouched but our filtered file is unreachable.
    # Detect that case and either (a) extend LIBINT_DATA_PATH or (b) point
    # at the cache dir's parent.
    libint_dp = os.environ.get("LIBINT_DATA_PATH")
    expected_root = str(cache_dir.parent)   # libint expects <root>/basis/<name>.g94
    if libint_dp != expected_root and (Path(libint_dp or "") / "basis") != cache_dir:
        # Point libint at our cache dir's parent. We don't try to chain
        # multiple data paths (libint reads a single env var), so the
        # bundled bases must already be present at expected_root, OR
        # the user is expected to keep using stock bases that are also
        # findable via the new path. The default cache_dir is exactly
        # ``<LIBINT_DATA_PATH>/basis/`` -- see _filtered_basis_cache_dir
        # -- so this branch only fires when the user overrode
        # VIBEQC_FILTERED_BASIS_DIR or fell through to ~/.cache.
        os.environ["LIBINT_DATA_PATH"] = expected_root

    new_basis = BasisSet(molecule, new_name)
    report = BasisFilterReport(
        threshold=float(exp_to_discard),
        primitives_total=n_prim_total,
        primitives_kept=n_prim_kept,
        n_primitives_dropped=n_prim_total - n_prim_kept,
        n_shells_dropped=n_shells_dropped,
        new_name=new_name,
        original_name=basis.name,
        dropped_primitives=dropped_primitives,
        dropped_shells=dropped_shells,
    )
    return new_basis, report


def format_basis_filter_report(report: BasisFilterReport) -> str:
    """Render a :class:`BasisFilterReport` as a multi-line, fully
    transparent summary of what the filter dropped -- every primitive
    listed with its element, shell, l-letter, exponent, and stored
    coefficient.

    User-facing transparency (per the v0.7 directive: "we need to
    print exactly what we drop into the output"). Designed to be
    handed straight to ``ProgressLogger.write_raw`` from the SCF
    banner so the run's output file is self-documenting and the
    filter doesn't silently change basis content.
    """
    lines: List[str] = []
    lines.append(
        f"Basis-set primitive filter: {report.original_name} -> "
        f"{report.new_name}"
    )
    lines.append("-" * 60)
    lines.append(f"  exp_to_discard threshold : {report.threshold:.4g}")
    lines.append(
        f"  primitives kept / total  : "
        f"{report.primitives_kept} / {report.primitives_total} "
        f"({report.fraction_dropped * 100:.2f}% dropped)"
    )
    lines.append(f"  shells dropped entirely  : {report.n_shells_dropped}")

    if not report.dropped_primitives:
        lines.append("  (no primitives below threshold -- basis unchanged)")
        return "\n".join(lines)

    lines.append("")
    lines.append("  Primitives dropped:")
    lines.append(
        f"  {'elem':>4}  {'shell':>5}  {'l':>2}  "
        f"{'prim':>4}  {'a (bohr⁻^2)':>14}  {'c (libint)':>14}"
    )
    for d in report.dropped_primitives:
        lines.append(
            f"  {d.element_symbol:>4}  {d.shell_index:>5}  "
            f"{d.shell_letter:>2}  {d.primitive_index:>4}  "
            f"{d.exponent:>14.6f}  {d.coefficient_libint:>14.6e}"
        )

    if report.dropped_shells:
        lines.append("")
        lines.append(
            "  Shells removed entirely (every primitive below threshold):"
        )
        for z, shell_idx in report.dropped_shells:
            symbol = _symbol_for_z(z)
            lines.append(f"    {symbol} (Z={z}), shell #{shell_idx}")

    lines.append("")
    lines.append(
        "  This filter removes basis content. Cite the change in any "
        "publication using these results: the drop list above is the "
        "complete record of what was removed from the parent basis."
    )

    return "\n".join(lines)
