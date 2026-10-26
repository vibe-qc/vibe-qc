"""CRYSTAL backend for vibe-basis — targets **CRYSTAL23**.

Drives external CRYSTAL (Dovesi / Erba / Orlando / Saunders et al.)
runs from a basis-set optimization loop. This module owns:

* ``parse_output(text)`` — extract total HF/DFT energy, the CRYSTAL
  version, and convergence flags from a ``.out`` file. Defensive
  against vq's ``STARVED`` mis-kill regression, SCF non-convergence,
  and crashed jobs that never wrote a ``TOTAL ENERGY`` line.
* ``probe_version(text)`` — read the major version off the banner.
* ``emit_input(...)`` / ``emit_input_inline(...)`` — serialize a
  per-element basis + unit-cell description into a ``.d12`` deck.

vibe-basis treats CRYSTAL as an external program: we read its
output text, but never link against any CRYSTAL library or call
its internals from Python.

Pipeline used by vibe-basis (Goal 8 mpei-TZVP, Stages 0-4):

  param vector x_k
    → emit_input(x_k) → mgo.d12 (per-compound)
    → transport.run("crystal", mgo.d12) → mgo.out
    → parse_output(mgo.out) → CrystalEnergyResult
    → objective += w_i * result.energy   (if result.ok)
                  or  np.inf             (if not result.ok)

Version targeting
-----------------
This module was written against CRYSTAL14 and **retargeted to
CRYSTAL23** (vibe-basis 0.3.0). The retarget is deliberately *not* a
rewrite, because the surface we depend on did not move:

* the ``.d12`` input format is stable CRYSTAL09 → 23 for everything
  we emit (geometry block, basis block, SCF block, DFT block);
* the ``TOTAL ENERGY(...)`` and ``SCF ENDED`` lines are unchanged;
* the one real difference — the end-of-run terminator — was already
  handled here: CRYSTAL14 writes a bare rule of ``E``'s, CRYSTAL23
  writes ``EEEEEEEEEE TERMINATION  DATE ...``, and
  ``_EEEE_TERMINATOR_RE`` matches both (see its docstring).

What the retarget adds is :func:`probe_version` and the
``crystal_version`` field on :class:`CrystalEnergyResult`, so a
result records *which* CRYSTAL produced it instead of leaving that
to be assumed. Compatibility here is asserted-and-recorded, not
assumed silently: a run whose banner says ``14`` while the campaign
expects ``23`` is visible in the result rather than blended into the
numbers. :meth:`CrystalEnergyResult.version_matches` is the check;
enforcement is the caller's policy decision, because a *parsed*
older output is still perfectly readable — it is only the *claim*
that it came from the campaign's engine that would be false.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Regex toolbox
# ---------------------------------------------------------------------------
#
# CRYSTAL14 prints lines like::
#
#     CYCLE   8 TOTAL ENERGY(HF)(AU)(   8)         -2.7468175399E+02 DE-4.5E-12
#     CYCLE  12 TOTAL ENERGY(DFT)(AU)(  12)        -2.7547759489E+02 DE-3.2E-11
#
# Three paren groups: the method label, the (AU) unit tag, and the
# cycle counter. The number after the third paren group is the
# energy in Hartree. CRYSTAL uses scientific notation with capital E.
#
# The convergence marker is one of::
#
#     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7468175400E+02
#     == SCF ENDED - CONVERGENCE ON DENSITY MATRIX
#     == SCF ENDED - TOO MANY CYCLES                ← non-converged
#     SCF NOT CONVERGED                              ← rare; manual abort
#
# We treat the first two as success and everything else as failure.

_TOTAL_ENERGY_RE = re.compile(
    r"""
        TOTAL\ ENERGY\s*\(([^)]+)\)\s*\([^)]*\)\s*\(\s*(\d+)\s*\)\s+
        (-?\d+\.\d+[Ee][+\-]?\d+)
    """,
    re.VERBOSE,
)
# Deliberately not anchored to ``^`` — CRYSTAL14 prefixes the
# TOTAL ENERGY line with ``CYCLE  N`` and indents it variably. The
# trailing ``DE...`` discrepancy across CRYSTAL versions (space vs.
# no-space between DE and the number) is not part of what we need;
# we capture method + cycle + energy only.

_SCF_ENDED_RE = re.compile(
    r"""
        SCF\ ENDED\ -\ (CONVERGENCE\ ON\ (?:ENERGY|DENSITY\ MATRIX)|
                        TOO\ MANY\ CYCLES)
    """,
    re.VERBOSE,
)
# CRYSTAL14 emits the SCF ENDED line with a ``==`` prefix and the
# converged energy as a suffix on the same line. We just look for
# the canonical phrase anywhere in the line.

_NOT_CONVERGED_RE = re.compile(r"^\s*SCF NOT CONVERGED")

#: The default CRYSTAL major version this backend targets.
DEFAULT_CRYSTAL_VERSION = 23

_VERSION_BANNER_RE = re.compile(r"\bCRYSTAL(\d{2})\b")
"""Major version off the CRYSTAL start-up banner.

Every CRYSTAL release opens its output with a boxed banner naming
itself, e.g.::

     *******************************************************************
     *                              CRYSTAL23                          *
     *                     public : 1.0.1 - Dec 20th, 2022             *

so a two-digit run of characters right after ``CRYSTAL`` is the major
version (14 / 17 / 23). Matched anywhere in the line because the
banner is centred with variable padding.

Deliberately narrow: ``\\d{2}`` and a word boundary, so it does not
fire on ``CRYSTAL`` in prose, on a path like ``/opt/crystal23/bin``
(lowercase), or on the ``CRYSTAL`` keyword that opens a periodic
``.d12`` geometry block. Returns the *first* match, which is the
banner -- later mentions in an echoed input deck must not override it.
"""

_EEEE_TERMINATOR_RE = re.compile(r"^\s*E{4,}\b")
"""Sentinel for "output file finished cleanly" — its absence flags a
wall-time truncation. Two on-disk forms are matched:

* **CRYSTAL14** prints a bare line of capital-E's near the end of a
  successful output as a section divider::

      EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

* **CRYSTAL23** (the crystal23demo binary on remote compute hosts) ends a
  converged run with a banner line carrying a date/time stamp::

      EEEEEEEEEE TERMINATION  DATE 18 05 2026 TIME 15:14:00.2

So we anchor on "4 or more E's at line start, then a word boundary"
rather than "a line of *only* E's" — the latter (the dropped-then-
restored ``b5d4f20`` regression) mis-classified every converged
CRYSTAL23 output as ``failure_mode="truncated"``."""


def _deck(text: str) -> str:
    """Strip blank lines from a CRYSTAL deck.

    CRYSTAL reads the SCF section line by line as keywords, and a blank
    line is read as an *empty keyword*: ``ERROR **** READM2 **** KEYWORD
    NOT ALLOWED``. No CRYSTAL deck legitimately contains one, and the
    emitters here interpolate optional blocks that are empty when the
    option is off, so filtering once at the end is safer than making
    every interpolation site newline-perfect.
    """
    return "\n".join(line for line in text.splitlines() if line.strip()) + "\n"


def _tolinteg_block(tolinteg) -> str:
    """``TOLINTEG`` block, or empty for CRYSTAL's own defaults."""
    if not tolinteg:
        return ""
    return "TOLINTEG\n" + " ".join(str(int(v)) for v in tolinteg) + "\n"


def probe_version(text: str) -> Optional[int]:
    """Return the CRYSTAL major version from *text*'s banner, or ``None``.

    ``None`` means "the banner was not in the text given" -- which is
    normal for a truncated output, a hand-built fixture, or a tail
    slice. It does **not** mean "wrong version", and callers must not
    treat it as a failure: an output with no banner but a converged
    ``TOTAL ENERGY`` is still a usable energy.

    >>> probe_version(" *                     CRYSTAL23                    *")
    23
    >>> probe_version("no banner here") is None
    True
    """
    m = _VERSION_BANNER_RE.search(text)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None


@dataclass(frozen=True)
class CrystalEnergyResult:
    """Parsed result of one CRYSTAL ``.out`` file.

    Use ``ok`` as the boolean gate before consuming ``energy`` — for
    failures, ``energy`` is ``None`` and ``failure_mode`` reports
    why.

    Attributes
    ----------
    ok
        ``True`` iff the SCF converged AND the file finished cleanly
        (EEEE terminator present) AND a TOTAL ENERGY line was found.
        Optimizer code should branch on this and substitute
        ``np.inf`` for the objective when ``ok`` is False.
    energy
        Total HF/DFT energy in Hartree, per unit cell. ``None`` only
        when no TOTAL ENERGY line was found anywhere (CRYSTAL
        crashed before SCF). For other failure modes the last
        reported cycle's energy is surfaced for diagnosis but the
        caller must NOT use it as a converged value.
    method
        ``"HF"``, ``"DFT"``, ``"UHF"``, etc. as written in the
        ``TOTAL ENERGY(<method>)`` paren group.
    last_cycle
        SCF iteration number on the last TOTAL ENERGY line.
    converged
        ``True`` iff CRYSTAL printed ``SCF ENDED - CONVERGENCE ON
        ...``. Distinct from ``ok``: a converged run that got
        truncated (e.g. by vq STARVED) has ``converged=True`` but
        ``ok=False``.
    truncated
        ``True`` iff the file lacks the trailing EEEE terminator.
    failure_mode
        One of:
            * ``None`` — success (``ok=True``).
            * ``"non_converged"`` — SCF ENDED - TOO MANY CYCLES.
            * ``"scf_not_converged"`` — explicit SCF NOT CONVERGED
              marker (e.g. CRYSTAL aborted due to linear
              dependence).
            * ``"truncated"`` — TOTAL ENERGY + SCF ENDED present,
              but no EEEE terminator; flags the vq v0.5.9 STARVED
              mis-kill regression.
            * ``"no_energy_line"`` — CRYSTAL crashed before
              reaching SCF (basis-file parse error, etc.); the
              caller cannot use ``energy`` (it's ``None``).
    n_lines_scanned
        Debug counter: how many lines of the output we read.
    crystal_version
        CRYSTAL major version off the banner (14 / 17 / 23), or
        ``None`` when the text carried no banner. **Provenance only:
        it deliberately does not affect** ``ok``. A converged energy
        from an output whose banner we could not see is still a
        converged energy; what would be wrong is *claiming* it came
        from a particular CRYSTAL. Use :meth:`version_matches` for
        that check.
    """

    ok: bool
    energy: Optional[float]
    method: Optional[str]
    last_cycle: Optional[int]
    converged: bool
    truncated: bool
    failure_mode: Optional[str]
    n_lines_scanned: int
    crystal_version: Optional[int] = None

    def __post_init__(self) -> None:
        # Coherence: ok ↔ (energy is not None AND converged AND not
        #                  truncated AND failure_mode is None).
        # crystal_version is intentionally absent from this invariant.
        if self.ok:
            assert self.energy is not None
            assert self.converged
            assert not self.truncated
            assert self.failure_mode is None
        else:
            assert self.failure_mode is not None

    def version_matches(self, expected: int = DEFAULT_CRYSTAL_VERSION) -> Optional[bool]:
        """Did this output come from CRYSTAL *expected*?

        Three-valued on purpose:

        * ``True``  -- banner seen and it matches.
        * ``False`` -- banner seen and it does **not** match. The
          energy is still parsed and usable, but attributing it to
          *expected* would be false; a campaign that mixes engines
          silently is exactly the provenance failure this field
          exists to prevent.
        * ``None``  -- no banner in the text, so unknowable. Not a
          failure (see :attr:`crystal_version`).

        Enforcement is the caller's policy: a parity study may
        legitimately want CRYSTAL14 output, so this reports rather
        than raises.
        """
        if self.crystal_version is None:
            return None
        return self.crystal_version == expected


def parse_output(text: str) -> CrystalEnergyResult:
    """Parse a CRYSTAL ``.out`` text into :class:`CrystalEnergyResult`.

    Reads the whole text into memory. CRYSTAL outputs are typically
    < 10 MB even at production sizes, so this is fine; callers can
    stream-truncate longer files if needed.

    The parser uses the *last* ``TOTAL ENERGY`` line in the file
    (the converged value, not earlier SCF cycle prints). It also
    tolerates the absence of either the ``SCF ENDED`` marker or the
    ``EEEE`` terminator and tags the corresponding failure mode.
    """
    last_energy: Optional[float] = None
    last_method: Optional[str] = None
    last_cycle: Optional[int] = None
    converged: bool = False
    seen_eeee: bool = False
    scf_explicitly_not_converged: bool = False
    n_lines = 0
    # Probed once over the whole text rather than per line: the banner
    # is a handful of lines at the top, and a single search is cheaper
    # than a per-line match against every line of a multi-MB output.
    version = probe_version(text)

    for line in text.splitlines():
        n_lines += 1
        m = _TOTAL_ENERGY_RE.search(line)
        if m:
            last_method = m.group(1).strip()
            try:
                last_cycle = int(m.group(2))
            except ValueError:
                pass
            try:
                last_energy = float(m.group(3))
            except ValueError:
                # Should not happen given the regex matched, but be
                # defensive — keep the previous energy.
                pass
            continue
        m_scf = _SCF_ENDED_RE.search(line)
        if m_scf:
            tag = m_scf.group(1)
            if tag.startswith("CONVERGENCE"):
                converged = True
            else:
                # TOO MANY CYCLES — non-convergence.
                converged = False
            continue
        if _NOT_CONVERGED_RE.match(line):
            scf_explicitly_not_converged = True
            converged = False
            continue
        if _EEEE_TERMINATOR_RE.match(line):
            seen_eeee = True

    # ----- Decision tree --------------------------------------------------
    if last_energy is None:
        return CrystalEnergyResult(
            ok=False,
            energy=None,
            method=None,
            last_cycle=None,
            converged=False,
            truncated=not seen_eeee,
            failure_mode="no_energy_line",
            n_lines_scanned=n_lines,

            crystal_version=version,
        )
    if scf_explicitly_not_converged:
        return CrystalEnergyResult(
            ok=False,
            energy=last_energy,
            method=last_method,
            last_cycle=last_cycle,
            converged=False,
            truncated=not seen_eeee,
            failure_mode="scf_not_converged",
            n_lines_scanned=n_lines,

            crystal_version=version,
        )
    if not converged:
        return CrystalEnergyResult(
            ok=False,
            energy=last_energy,
            method=last_method,
            last_cycle=last_cycle,
            converged=False,
            truncated=not seen_eeee,
            failure_mode="non_converged",
            n_lines_scanned=n_lines,

            crystal_version=version,
        )
    if not seen_eeee:
        return CrystalEnergyResult(
            ok=False,
            energy=last_energy,
            method=last_method,
            last_cycle=last_cycle,
            converged=True,
            truncated=True,
            failure_mode="truncated",
            n_lines_scanned=n_lines,

            crystal_version=version,
        )
    return CrystalEnergyResult(
        ok=True,
        energy=last_energy,
        method=last_method,
        last_cycle=last_cycle,
        converged=True,
        truncated=False,
        failure_mode=None,
        n_lines_scanned=n_lines,

        crystal_version=version,
    )


def parse_output_file(path: str | Path) -> CrystalEnergyResult:
    """Convenience wrapper: parse a CRYSTAL ``.out`` from disk.

    Raises ``FileNotFoundError`` if the path doesn't exist (the
    fetch failed); use :func:`parse_output` directly when you
    already have the text in memory (e.g. from ``vq fetch`` stdout).
    """
    p = Path(path)
    return parse_output(p.read_text())


# ===========================================================================
# Input emission: build a CRYSTAL .d12 deck from a Structure + basis + method
# ===========================================================================

# DFT functional keywords CRYSTAL recognizes. Bundle here so the
# emitter accepts the same set everywhere — if the caller passes
# something else, we return None rather than emit a deck CRYSTAL
# would silently reject. ``hf`` / ``rhf`` mean "no DFT block".
_CRYSTAL_DFT_FUNCTIONALS = frozenset(
    {
        "pw1pw",
        "b3lyp",
        "pbe",
        "pbe0",
        "wcgga",
        "blyp",
        "lda",
        "svwn",
        "wc1lyp",
        "hse06",
        "hsesol",
        # SCAN-family meta-GGAs and their global hybrids. CRYSTAL23
        # manual, "Availability of XC functionals": SCAN and r2SCAN as
        # non-empirical mGGAs, plus r2SCAN hybrids at 10 / 25 / 50 %
        # HF exchange. r2SCAN is the functional the pob cohesive-energy
        # reference set is computed at, so the campaign needs it.
        #
        # NOTE: these want a denser integration grid. The manual
        # specifies HUGEGRID, a pruned (300,1454) grid, for the SCAN
        # family, and meta-GGA grid noise is a few kJ/mol -- the same
        # size as the differences a basis campaign resolves.
        "scan",
        "scan0",
        "r2scan",
        "r2scanh",
        "r2scan0",
        "r2scan50",
    }
)

#: Functionals for which the CRYSTAL23 manual recommends a denser
#: integration grid: HUGEGRID "should be used in combination with the
#: SCAN functional".
#:
#: **Off by default, deliberately.** Not one of the 1759 decks in the pob
#: cohesive-energy reference set uses HUGEGRID, and those decks produced
#: every published number this work is gated against. Matching the
#: protocol beats following the manual when the goal is reproducing its
#: results; pass ``hugegrid=True`` to follow the manual instead. Whichever
#: is chosen must be the same for the bulk and its free atoms, or the
#: difference lands in the cohesive energy.
_NEEDS_HUGEGRID = frozenset(
    {"scan", "scan0", "r2scan", "r2scanh", "r2scan0", "r2scan50"}
)

#: Integral-screening thresholds used by every deck in the pob
#: cohesive-energy reference set. Much tighter than CRYSTAL's own
#: defaults, and the difference is not negligible at the sub-kJ/mol
#: tolerance a cohesive-energy gate works to, so this is the default
#: here. Pass ``tolinteg=None`` for CRYSTAL's defaults.
REFERENCE_TOLINTEG = (9, 9, 9, 18, 54)


def _format_lattice_line(struct) -> str:
    """Format the lattice-parameter line per CRYSTAL's syntax.

    The number of values depends on crystal system:
      * cubic                 — ``a``
      * hexagonal / trigonal  — ``a c``
      * tetragonal            — ``a c``
      * orthorhombic          — ``a b c``
      * monoclinic            — ``a b c β``  (β = unique angle)
      * triclinic / fallback  — ``a b c α β γ``
    """
    sys_lower = struct.crystal_system.lower()
    if sys_lower == "cubic":
        return f"{struct.a:.6f}"
    if sys_lower in ("hexagonal", "trigonal", "tetragonal"):
        return f"{struct.a:.6f} {struct.c:.6f}"
    if sys_lower == "orthorhombic":
        return f"{struct.a:.6f} {struct.b:.6f} {struct.c:.6f}"
    if sys_lower == "monoclinic":
        return f"{struct.a:.6f} {struct.b:.6f} {struct.c:.6f} {struct.beta:.4f}"
    return (
        f"{struct.a:.6f} {struct.b:.6f} {struct.c:.6f} "
        f"{struct.alpha:.4f} {struct.beta:.4f} {struct.gamma:.4f}"
    )


def emit_input(
    struct,
    basis: str,
    method: str = "rhf",
    *,
    shrink: int = 8,
    toldee: int = 8,
    tolinteg: Optional[tuple[int, ...]] = REFERENCE_TOLINTEG,
    hugegrid: bool = False,
) -> Optional[str]:
    """Build a CRYSTAL ``.d12`` input deck as a string.

    Parameters
    ----------
    struct
        A :class:`vibe_basis.io.structures.Structure` (typed via duck
        typing so this module doesn't have to import structures —
        keeps the dep DAG one-way).
    basis
        Basis-set name. Uppercased on output; must match a CRYSTAL
        keyword like ``"POB-TZVP"`` / ``"POB-TZVP-REV2"`` / ``"POB-DZVP-REV2"``.
        Candidate bases produced by an optimizer use a custom name
        and a per-element file emitted via a separate code path
        (planned for Stage 1+).
    method
        ``"rhf"`` / ``"hf"`` → no DFT block. Anything in
        :data:`_CRYSTAL_DFT_FUNCTIONALS` → a ``DFT / <functional> /
        END`` block. Anything else → returns ``None`` (skip rather
        than emit a deck CRYSTAL would reject).
    shrink
        Pack-Monkhorst grid (``SHRINK <shrink> <shrink>``). Default 8.
    toldee
        SCF energy-convergence exponent (``TOLDEE <toldee>`` = 10⁻ⁿ
        Hartree). Default 8.

    Returns
    -------
    str
        The .d12 deck text. The caller writes it wherever (the
        Stage-0 recipe will write into the per-compound work
        directory the transport submits from).
    None
        If the structure lacks ``crystal_spacegroup`` /
        ``crystal_asymm_unit`` (CRYSTAL needs the asymm-unit form),
        is AFM (needs ATOMSPIN — REQUIREMENTS-PERIODIC R3), or the
        method isn't a recognized CRYSTAL keyword.

    Notes
    -----
    Per CLAUDE.md § 10, CRYSTAL is an external code — this function
    only writes a text deck; CRYSTAL never runs in-process.

    Format follows the CRYSTAL23 manual recipe for periodic SCF:

      title
      CRYSTAL
      IFLAG IFHR IFSO          (0 0 0 = cell defined by a,b,c,α,β,γ)
      SG_NUMBER                (ITC space group)
      LATTICE_PARAMS           (a [b c α β γ] depending on system)
      NATOMS_ASYM
      Z f_x f_y f_z            (one row per asymm-unit atom)
      BASISSET
      <basis-keyword>
      [DFT
      <functional>
      END]
      SHRINK
      <s> <s>                  (Pack-Monkhorst grid)
      TOLDEE
      <t>                      (10⁻ᵗ Hartree SCF threshold)
      END
    """
    if struct.crystal_spacegroup == 0 or not struct.crystal_asymm_unit:
        return None
    if getattr(struct, "afm_pattern", None):
        # AFM compounds need ATOMSPIN to specify magnetic ordering;
        # CRYSTAL's space-group expansion alone won't produce the
        # correct broken-symmetry state. Skip until R3 lands.
        return None

    method_lower = method.lower()
    method_block = ""
    if method_lower in ("rhf", "hf"):
        pass  # CRYSTAL defaults to HF when no DFT block is given.
    elif method_lower in _CRYSTAL_DFT_FUNCTIONALS:
        # HUGEGRID for the SCAN family, exactly as the atom emitter does.
        # A cohesive energy subtracts the bulk from its own free atoms, so
        # the two must be integrated on the same grid: a mismatch does not
        # cancel, it shows up as a few kJ/mol that looks like chemistry.
        # The plane-wave reference work learned this the same way, holding
        # atoms at a fixed cutoff while sweeping the solid and getting a
        # spurious curve out of it.
        grid = ("\nHUGEGRID"
                if hugegrid and method_lower in _NEEDS_HUGEGRID else "")
        method_block = f"\nDFT\n{method_lower.upper()}{grid}\nEND"
    else:
        return None  # Unknown method — refuse rather than emit garbage.

    tolinteg_block = _tolinteg_block(tolinteg)
    lattice_line = _format_lattice_line(struct)
    atom_lines = "\n".join(
        f"{atom.Z} {atom.fxyz[0]:.6f} {atom.fxyz[1]:.6f} {atom.fxyz[2]:.6f}"
        for atom in struct.crystal_asymm_unit
    )

    title = f"{struct.formula} — {method.upper()} on {basis.upper()} (CRYSTAL23 parity)"
    return (_deck(
        f"{title}\n"
        "CRYSTAL\n"
        "0 0 0\n"
        f"{struct.crystal_spacegroup}\n"
        f"{lattice_line}\n"
        f"{len(struct.crystal_asymm_unit)}\n"
        f"{atom_lines}\n"
        "BASISSET\n"
        f"{basis.upper()}"
        f"{method_block}\n"
        f"{tolinteg_block}"
        "SHRINK\n"
        f"{shrink} {shrink}\n"
        "TOLDEE\n"
        f"{toldee}\n"
        "END\n"
    ))


def emit_input_inline(
    struct,
    inline_basis: str,
    method: str = "rhf",
    *,
    shrink: int = 8,
    toldee: int = 8,
    toldeg: Optional[int] = None,
    tolinteg: Optional[tuple[int, ...]] = REFERENCE_TOLINTEG,
    hugegrid: bool = False,
) -> Optional[str]:
    """Build a CRYSTAL ``.d12`` deck with an inline basis set.

    Like :func:`emit_input` but the basis is embedded in the deck
    rather than referenced by name via ``BASISSET``.  The *inline_basis*
    text is the output of :func:`vibeqc.basis_crystal.emit_crystal` —
    a per-element CRYSTAL-format basis block ending with ``END``.

    This is useful for basis sets that CRYSTAL doesn't know about
    (e.g. ``pob-TZVP-rev2`` is not in CRYSTAL23's built-in library,
    and custom optimizer bases don't have registered keywords).

    The emitted deck has this shape::

        title
        CRYSTAL
        0 0 0
        SG_NUMBER
        LATTICE_PARAMS
        NATOMS_ASYM
        Z f_x f_y f_z
        ENDGEOM      ← geometry block end
        <inline_basis>
        [DFT / <functional> / END]
        SHRINK S S
        TOLDEE T
        [TOLDEG T]   ← optional gradient threshold for OPTGEOM
        END

    Parameters
    ----------
    struct
        A :class:`vibe_basis.io.structures.Structure`.
    inline_basis
        CRYSTAL-format inline basis text as returned by
        ``emit_crystal([atom1, atom2, ...])``.
    method
        ``"rhf"`` / ``"hf"`` (no DFT block) or a CRYSTAL functional
        keyword like ``"pw1pw"``, ``"pbe"``, …
    shrink
        Pack-Monkhorst grid.
    toldee
        SCF energy-convergence exponent (10⁻ⁿ Ha).
    toldeg
        Geometry-optimization gradient threshold.  When set (non-None),
        an ``OPTGEOM / ENDOPT / ENDGEOM`` block is emitted around the
        geometry for a single-point-structure validation.  Omit for
        pure single-point SCF.

    Returns
    -------
    str or None
        The .d12 deck, or None if the structure is unsupported.
    """
    if struct.crystal_spacegroup == 0 or not struct.crystal_asymm_unit:
        return None
    if getattr(struct, "afm_pattern", None):
        return None

    method_lower = method.lower()
    method_block = ""
    if method_lower in ("rhf", "hf"):
        pass
    elif method_lower in _CRYSTAL_DFT_FUNCTIONALS:
        # HUGEGRID for the SCAN family, exactly as the atom emitter does.
        # A cohesive energy subtracts the bulk from its own free atoms, so
        # the two must be integrated on the same grid: a mismatch does not
        # cancel, it shows up as a few kJ/mol that looks like chemistry.
        # The plane-wave reference work learned this the same way, holding
        # atoms at a fixed cutoff while sweeping the solid and getting a
        # spurious curve out of it.
        grid = ("\nHUGEGRID"
                if hugegrid and method_lower in _NEEDS_HUGEGRID else "")
        method_block = f"\nDFT\n{method_lower.upper()}{grid}\nEND"
    else:
        return None

    tolinteg_block = _tolinteg_block(tolinteg)
    lattice_line = _format_lattice_line(struct)
    atom_lines = "\n".join(
        f"{atom.Z} {atom.fxyz[0]:.6f} {atom.fxyz[1]:.6f} {atom.fxyz[2]:.6f}"
        for atom in struct.crystal_asymm_unit
    )

    title = f"{struct.formula} — {method.upper()} with inline basis (CRYSTAL23 parity)"

    toldeg_block = ""
    if toldeg is not None:
        toldeg_block = f"\nTOLDEG\n{toldeg}"

    return (_deck(
        f"{title}\n"
        "CRYSTAL\n"
        "0 0 0\n"
        f"{struct.crystal_spacegroup}\n"
        f"{lattice_line}\n"
        f"{len(struct.crystal_asymm_unit)}\n"
        f"{atom_lines}\n"
        "END\n"
        f"{inline_basis.strip()}\n"
        "ENDBS\n"
        f"{method_block}\n"
        f"{tolinteg_block}"
        "SHRINK\n"
        f"{shrink} {shrink}\n"
        "TOLDEE\n"
        f"{toldee}"
        f"{toldeg_block}\n"
        "END\n"
    ))
