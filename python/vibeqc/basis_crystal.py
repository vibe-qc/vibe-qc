"""CRYSTAL-format basis-set parser and NWChem/.g94 emitter.

Targets the per-element CRYSTAL basis-set files distributed by the
Bredow group (Peintinger--Vilela Oliveira--Bredow, ``pob-*``): one file
per element, named ``ZZ_Sym`` (e.g. ``06_C``), plain-text CRYSTAL input.

Supported today
---------------
All-electron shells (S, SP, P, D, F, G). These cover the light-element
Bredow sets (pob-TZVP and pob-DZVP-rev2 for H-Br). The emitter produces
NWChem/.g94 files that libint loads without modification.

Not yet supported
-----------------
ECP blocks (``INPUT`` header on Rb-I and Cs-Po pob-* archives). Parsing
them is trivial; *applying* them in vibe-qc needs libecpint integration,
which is a separate follow-on. Such files raise ``NotImplementedError``
for now.

CRYSTAL per-element format summary
----------------------------------
Line 1::
    Z  NSHELL
or for ECP-using atoms::
    (200 + Z)  NSHELL

Each shell starts with::
    ITYPE  LAT  NG  CHE  SCAL

where

* ``ITYPE`` -- 0 = user-defined (the only case the Bredow files use).
* ``LAT``   -- shell angular-momentum code:
      0=S, 1=SP, 2=P, 3=D, 4=F, 5=G.
* ``NG``    -- number of primitive Gaussians.
* ``CHE``   -- formal occupancy (electrons) assigned to this shell.
* ``SCAL``  -- scale factor applied to the exponents (1.0 = as-written).

Followed by ``NG`` lines of ``exponent coefficient`` (or ``exponent
s_coef p_coef`` for SP shells).
"""

from __future__ import annotations

import io
import re
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

# CRYSTAL LAT-code -> angular momentum label understood by libint's .g94 loader.
# Libint's BasisSet parser treats "SP" as a general-contraction s+p pair.
_LAT_TO_SHELL = {0: "S", 1: "SP", 2: "P", 3: "D", 4: "F", 5: "G"}

# Element symbols, 1-based index so _ELEMENT_SYMBOLS[Z] returns the symbol.
# Light-element subset -- extend as needed; pob-TZVP only requires through Br.
_ELEMENT_SYMBOLS = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
]


@dataclass
class CrystalShell:
    """One contracted Gaussian shell in CRYSTAL format."""

    shell_type: str  # "S", "SP", "P", "D", "F", "G"
    occupancy: float  # CHE
    scale_factor: float  # SCAL (exponents multiplied by this)
    exponents: list[float] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    # For SP shells only: the p-side coefficients. Length matches exponents;
    # empty for non-SP shells. ``coefficients`` holds the s-side.
    coefficients_p: list[float] = field(default_factory=list)

    @property
    def n_primitives(self) -> int:
        return len(self.exponents)


@dataclass
class CrystalECPTerm:
    """One Gaussian term inside a CRYSTAL ECP angular-momentum block.

    Encodes the contribution

        V_ℓ_term(r) = C . r^n . exp(-a . r^2)

    in the standard Stuttgart-Köln / Hay-Wadt convention. ``ell`` is
    the angular momentum (0=s, 1=p, ..., 4=g) -- or the special label
    ``"local"`` for the M-counted local-potential terms in CRYSTAL's
    eq. 3.18.
    """

    ell: object  # int (0..4) or the literal "local"
    alpha: float  # ALFKL -- exponent
    coefficient: float  # CGKL  -- Gaussian prefactor
    n_pow: int  # NKL   -- power of r prefactor


@dataclass
class CrystalECP:
    """Parsed CRYSTAL effective-core-potential block.

    Built from the ``INPUT``-keyword block that follows a ``200+Z``
    atom header in CRYSTAL-format basis files. Carries everything
    libecpint needs to build the V_ECP operator: the effective
    nuclear charge (``Z_eff = Z - ncore``), and the per-ℓ Gaussian
    expansion of the local + ℓ-projected potentials.

    The CRYSTAL ECP convention has up to five ℓ-projected
    components (ℓ = 0..4) plus a "local" group counted by ``M`` in
    the header. ``terms`` collects them all in source order; group
    them by ``term.ell`` to match a specific potential expansion.
    """

    znuc: float  # effective core charge (= Z - ncore)
    m_local: int  # M in the CRYSTAL header (eq. 3.18)
    m_per_ell: tuple[int, ...] = ()  # (M0, M1, M2, M3, M4)
    terms: list[CrystalECPTerm] = field(default_factory=list)

    @property
    def total_terms(self) -> int:
        return self.m_local + sum(self.m_per_ell)


@dataclass
class CrystalAtomBasis:
    """Parsed per-atom basis set (one Bredow file)."""

    Z: int  # atomic number (without ECP offset)
    has_ecp: bool  # True if file had 200+Z header
    shells: list[CrystalShell] = field(default_factory=list)
    ecp: Optional["CrystalECP"] = None  # populated when has_ecp + INPUT block parsed

    @property
    def element_symbol(self) -> str:
        return (
            _ELEMENT_SYMBOLS[self.Z]
            if 0 < self.Z < len(_ELEMENT_SYMBOLS)
            else f"Z{self.Z}"
        )


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _tokens(stream: Iterator[str]) -> Iterator[tuple[int, str]]:
    """Yield (line_no, token) pairs, skipping blank lines and CRYSTAL
    comments (``!`` introduces a comment-to-end-of-line, matching CRYSTAL)."""
    for lineno, raw in enumerate(stream, start=1):
        # Strip trailing comments and whitespace.
        cleaned = raw.split("!", 1)[0]
        for tok in cleaned.split():
            yield lineno, tok


def parse_crystal_atom_basis(text: str, source: str = "<string>") -> CrystalAtomBasis:
    """Parse a single CRYSTAL per-element basis file.

    ``source`` is used only in error messages.
    """
    tokens = _tokens(iter(text.splitlines()))
    # Single-token pushback so the parser can peek one token *together with
    # its source line number* without consuming it. Holds (lineno, token)
    # pairs; only ever 0 or 1 deep in practice. The line number is what lets
    # the shell-header reader below tell a stray on-line token (the ``1 0``
    # SCAL typo) apart from the next record on the following line.
    pushback: list[tuple[int, str]] = []

    def next_pair() -> tuple[int, str]:
        if pushback:
            return pushback.pop()
        try:
            return next(tokens)
        except StopIteration as exc:
            raise ValueError(f"{source}: unexpected end of file") from exc

    def nxt() -> str:
        return next_pair()[1]

    def peek_pair() -> Optional[tuple[int, str]]:
        """Look at the next (lineno, token) without consuming it; None at EOF."""
        if pushback:
            return pushback[-1]
        try:
            pair = next(tokens)
        except StopIteration:
            return None
        pushback.append(pair)
        return pair

    # ---- Header: (Z or 200+Z) NSHELL ---------------------------------
    z_raw = int(nxt())
    nshell = int(nxt())
    if z_raw >= 200:
        atom = CrystalAtomBasis(Z=z_raw - 200, has_ecp=True)
    elif z_raw >= 1:
        atom = CrystalAtomBasis(Z=z_raw, has_ecp=False)
    else:
        raise ValueError(f"{source}: invalid atomic number {z_raw}")

    # ---- Optional ECP block (``INPUT`` keyword) ----------------------
    # Peek the next token; if it is 'INPUT' we must consume the whole ECP
    # block before the shells.
    try:
        first_pair = next_pair()
    except ValueError:
        first_pair = None
    if first_pair is not None and first_pair[1].upper() == "INPUT":
        # CRYSTAL Manual Sec."Effective core pseudo-potentials" (p. 84):
        #   line 1: ZNUC M M0 M1 M2 M3 M4    (effective charge + 6 counts)
        #   then  M+M0+M1+M2+M3+M4 records, each: alpha coefficient n_factor
        #
        # M is the number of "local" terms (eq. 3.18); M0..M4 are the
        # per-ℓ projections (eq. 3.19) for ℓ = 0..4. Source order in
        # the file is local first, then ℓ=0, ℓ=1, ... which we preserve
        # in the resulting ``CrystalECPTerm`` list.
        try:
            znuc = float(nxt())
            m_local = int(nxt())
            m0 = int(nxt())
            m1 = int(nxt())
            m2 = int(nxt())
            m3 = int(nxt())
            m4 = int(nxt())
        except (ValueError, StopIteration) as exc:
            raise ValueError(
                f"{source}: malformed CRYSTAL ECP INPUT header "
                f"(expected ZNUC M M0 M1 M2 M3 M4): {exc}"
            ) from exc

        ecp = CrystalECP(
            znuc=znuc,
            m_local=m_local,
            m_per_ell=(m0, m1, m2, m3, m4),
        )
        # Read the grouped records in source order.
        groups: list[tuple[object, int]] = [("local", m_local)]
        for ell, m_ell in enumerate((m0, m1, m2, m3, m4)):
            groups.append((ell, m_ell))
        for ell_label, m_count in groups:
            for _ in range(m_count):
                try:
                    alpha = float(nxt())
                    coef = float(nxt())
                    n_pow = int(nxt())
                except (ValueError, StopIteration) as exc:
                    raise ValueError(
                        f"{source}: malformed CRYSTAL ECP record for "
                        f"ell={ell_label!r}: {exc}"
                    ) from exc
                ecp.terms.append(
                    CrystalECPTerm(
                        ell=ell_label,
                        alpha=alpha,
                        coefficient=coef,
                        n_pow=n_pow,
                    )
                )
        atom.ecp = ecp
        # The INPUT block is fully consumed; nothing is pushed back, so the
        # shell loop below reads the first shell header directly.
    elif first_pair is not None:
        pushback.append(first_pair)

    # ---- Shell loop ---------------------------------------------------
    for shell_idx in range(nshell):
        itype = int(nxt())
        lat = int(nxt())
        ng = int(nxt())
        che = float(nxt())
        scal_lineno, scal_tok = next_pair()

        # The CRYSTAL shell header is exactly five tokens:
        #     ITYPE LAT NG CHE SCAL
        # Any further token on the *same physical line* is anomalous. The one
        # known case is the Bredow ``16_S`` typo where SCAL ``1.0`` was
        # mis-keyed as two tokens ``1 0`` (decimal point typed as a space),
        # i.e. ``0 3 1 0.0 1 0``. Without line awareness the stray ``0`` is
        # silently consumed as the d-shell exponent and the real exponent
        # (0.5207... for pob-TZVP, 0.4107... for pob-TZVP-rev2) as the
        # coefficient -- yielding a physically invalid exp=0 function with no
        # error raised. Tolerate that exact ``<int> <int>`` -> ``<int>.<int>``
        # pattern -- mirroring
        # scripts/basisset_dev/pob_basis_verify.py:parse_crystal_file (the
        # canonical verify/regenerate tool) so the two parsers agree -- and
        # reject anything else extra on the header line.
        extra = peek_pair()
        if extra is not None and extra[0] == scal_lineno:
            if "." not in scal_tok and extra[1].isdigit():
                next_pair()  # consume the stray fractional digit
                scal_tok = f"{scal_tok}.{extra[1]}"
            else:
                raise ValueError(
                    f"{source}: shell {shell_idx + 1}: unexpected extra token "
                    f"{extra[1]!r} on the shell-header line (expected exactly "
                    f"5 tokens: ITYPE LAT NG CHE SCAL)."
                )
        try:
            scal = float(scal_tok)
        except ValueError as exc:
            raise ValueError(
                f"{source}: shell {shell_idx + 1}: malformed SCAL value "
                f"{scal_tok!r} in shell header: {exc}"
            ) from exc

        if itype != 0:
            raise ValueError(
                f"{source}: shell {shell_idx + 1}: CRYSTAL built-in basis "
                f"types (ITYPE != 0) are not supported -- only user-defined "
                f"contracted Gaussians."
            )
        shell_type = _LAT_TO_SHELL.get(lat)
        if shell_type is None:
            raise ValueError(
                f"{source}: shell {shell_idx + 1}: unknown LAT code {lat}. "
                f"Known: {_LAT_TO_SHELL}."
            )

        shell = CrystalShell(
            shell_type=shell_type,
            occupancy=che,
            scale_factor=scal,
        )
        for _ in range(ng):
            expn = float(nxt())
            # Apply the CRYSTAL exponent-scale convention.
            #
            # CRYSTAL23 User's Manual, "Basis set input" (p. 25) and its
            # note 2: SCAL is meaningful only for the *standard* Pople
            # STO-nG / n-21G sets (ITYB=1), where SCAL=0. selects the
            # standard Pople scale factor. For a general user basis
            # (ITYB=0 -- the only kind we parse, enforced above) the
            # exponents are already scaled, and CRYSTAL's own GAUSS98
            # exporter writes SCAL=1. for exactly that reason (manual,
            # GAUSS98 note 5). So **both 1.0 and 0.0 mean "unscaled"**
            # here; only a genuine scale factor rescales.
            #
            # Treating 0.0 as a literal factor multiplies every exponent
            # by zero, which the positive-exponent guard below then
            # reports as a mis-parsed shell header. That is how this was
            # found: `emit_crystal` writes SCAL 0.0, so emit -> parse was
            # not a round trip.
            if scal not in (0.0, 1.0):
                expn = expn * (scal**2)
            # A Gaussian primitive exponent must be strictly positive. A zero
            # or negative exponent is physically meaningless and is the
            # tell-tale of a mis-parsed shell header silently shifting the
            # token stream (see the SCAL-typo handling above). Fail loudly
            # rather than emit a bogus diffuse function.
            if expn <= 0.0:
                raise ValueError(
                    f"{source}: shell {shell_idx + 1}: non-positive Gaussian "
                    f"exponent {expn!r}; basis primitive exponents must be "
                    f"> 0 (often a sign the shell header mis-parsed)."
                )
            if shell_type == "SP":
                s_coef = float(nxt())
                p_coef = float(nxt())
                shell.exponents.append(expn)
                shell.coefficients.append(s_coef)
                shell.coefficients_p.append(p_coef)
            else:
                coef = float(nxt())
                shell.exponents.append(expn)
                shell.coefficients.append(coef)
        atom.shells.append(shell)

    return atom


def parse_crystal_atom_basis_file(path: str | Path) -> CrystalAtomBasis:
    """Convenience wrapper: parse a CRYSTAL per-element file from disk."""
    p = Path(path)
    return parse_crystal_atom_basis(p.read_text(), source=str(p))


def parse_crystal_inline_basis(
    text: str, source: str = "<string>"
) -> list[CrystalAtomBasis]:
    """Split a **multi-element** CRYSTAL inline basis block into per-element specs.

    The inverse of :func:`emit_crystal`. Where
    :func:`parse_crystal_atom_basis` reads one element's file, this reads
    the concatenated block that goes into a ``.d12`` deck after
    ``ENDGEOM``::

        12 5                 <- Mg
        0 0 8  2.0 0.0
          ...
        8 4                  <- O
        0 0 8  2.0 0.0
          ...
         99 0                <- terminator

    Why it exists: the basis-optimization engines exchange candidate
    bases as exactly this text (``EnergyEngine.crystal_energy``'s
    ``basis_text``). An engine that computes a crystal needs *every*
    element's block, and an engine computing an isolated atom needs *the
    right one*. Reading only the first block and relabelling it with the
    requested ``Z`` -- which is what the vibe-qc engine used to do --
    silently gives O the Mg basis, which converges and is wrong.

    Splitting is line-structural (the CRYSTAL inline format is one
    record per line), then each per-element slice is handed to
    :func:`parse_crystal_atom_basis` so all the token-level quirks it
    handles -- the ``SCAL`` typo, the exponent-scale convention, the
    positive-exponent guard -- apply identically here.

    ECP ``INPUT`` blocks are rejected rather than skipped: their record
    count is not derivable from the shell headers, so a splitter that
    guessed would mis-frame every following element. :func:`emit_crystal`
    does not emit them (it writes ``200+Z`` with no ECP block), so this
    only fires on hand-written text.

    Parameters
    ----------
    text
        The inline basis block. A trailing ``99 0`` terminator and/or
        ``END`` are optional and are not returned.
    source
        Used only in error messages.

    Returns
    -------
    list[CrystalAtomBasis]
        One entry per element, in source order.

    Raises
    ------
    ValueError
        On a malformed block, a CRYSTAL built-in shell (``ITYPE != 0``),
        an inline ECP, or a duplicated element.
    """
    records: list[str] = []
    for raw in text.splitlines():
        cleaned = raw.split("!", 1)[0].strip()
        if cleaned:
            records.append(cleaned)

    atoms: list[CrystalAtomBasis] = []
    seen: set[int] = set()
    i = 0
    while i < len(records):
        toks = records[i].split()
        if toks[0].upper() in ("END", "ENDBS"):
            break
        if toks[0] == "99":
            break
        if len(toks) < 2:
            raise ValueError(
                f"{source}: line {i + 1}: expected an element header "
                f"'Z NSHELL', got {records[i]!r}"
            )
        try:
            z_raw = int(toks[0])
            nshell = int(toks[1])
        except ValueError as exc:
            raise ValueError(
                f"{source}: line {i + 1}: malformed element header "
                f"{records[i]!r}: {exc}"
            ) from exc

        start = i
        i += 1
        if i < len(records) and records[i].split()[0].upper() == "INPUT":
            raise ValueError(
                f"{source}: element Z={z_raw % 200}: inline ECP (INPUT) blocks "
                f"are not supported by the multi-element splitter -- their "
                f"record count is not derivable from the shell headers, so "
                f"every following element would be mis-framed. Pass "
                f"per-element text to parse_crystal_atom_basis instead."
            )

        for shell_idx in range(nshell):
            if i >= len(records):
                raise ValueError(
                    f"{source}: element Z={z_raw % 200}: block ends after "
                    f"{shell_idx} of {nshell} declared shells"
                )
            header = records[i].split()
            if len(header) < 5:
                raise ValueError(
                    f"{source}: line {i + 1}: shell header needs 5 fields "
                    f"'ITYPE LAT NG CHE SCAL', got {records[i]!r}"
                )
            if int(header[0]) != 0:
                raise ValueError(
                    f"{source}: line {i + 1}: CRYSTAL built-in basis types "
                    f"(ITYPE != 0) carry no primitive lines, so the block "
                    f"cannot be framed; only user-defined shells are supported."
                )
            npg = int(header[2])
            i += 1 + npg

        if i > len(records):
            raise ValueError(
                f"{source}: element Z={z_raw % 200}: block runs past the end "
                f"of the text (a primitive line is missing)"
            )

        atom = parse_crystal_atom_basis(
            "\n".join(records[start:i]), source=source
        )
        if atom.Z in seen:
            raise ValueError(
                f"{source}: element Z={atom.Z} appears more than once; an "
                f"inline basis block must carry one entry per element."
            )
        seen.add(atom.Z)
        atoms.append(atom)

    if not atoms:
        raise ValueError(f"{source}: no element blocks found in inline basis text")
    return atoms


# --------------------------------------------------------------------------
# CRYSTAL -> libecpint bridge (Phase 14g)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LibecpintInlineArrays:
    """Flat primitive arrays in the exact shape ``libecpint::ECPIntegrator::
    set_ecp_basis(...)`` expects for one ECP center.

    Mapping convention (per CRYSTAL23 Sec."Effective core pseudo-potentials"
    p. 84, eqs. 3.18 + 3.19, against libecpint's XML convention where the
    local channel sits at ``lval = max_proj_ell + 1``):

    * CRYSTAL ``M`` (``ell="local"`` term group)
        -> libecpint ``ams[i] = local_ell = max_proj_ell + 1``.
        This is the U_L channel that acts on all angular momenta (no
        projector applied), matching the local-potential semantics in
        CRYSTAL eq. 3.18.

    * CRYSTAL ``M_ℓ`` (``ell in {0..4}`` term groups, with non-zero count)
        -> libecpint ``ams[i] = ℓ``.
        Projected channels; libecpint applies P_ℓ during integration.

    * CRYSTAL ``NKL`` (the literal power of r in eqs. 3.18 + 3.19)
        -> libecpint ``ns[i] = NKL + 2``.
        libecpint takes the Gaussian-format power and subtracts two, so
        a Stuttgart r^0 term is NKL = 0 here and ``ns[i] = 2`` there.

    Both conventions yield the same V_ECP matrix elements when summed
    against the basis. Verified against ecp10mdf.xml (K, Z=19) shell
    layout: local at lval=4 with placeholder zero, projectors at lval=
    0..3 for s/p/d/f, max projector ell = 3.
    """

    # Flattened per-primitive arrays. All four lists have the same
    # length == n_primitives. Pass these straight to libecpint's
    # ``set_ecp_basis(necps=1, ..., shell_lengths=&n_primitives)``
    # (one ECP per atom-center; the n_primitives counts every primitive
    # across all l channels for that ECP).
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]
    ams: tuple[int, ...]  # angular momentum per primitive
    ns: tuple[int, ...]  # r^n power per primitive
    local_ell: int  # libecpint l-index for CRYSTAL's M-local block
    max_proj_ell: int  # highest ℓ with non-zero M_ℓ in CRYSTAL input

    @property
    def n_primitives(self) -> int:
        return len(self.exponents)


def crystal_ecp_to_libecpint_arrays(ecp: CrystalECP) -> LibecpintInlineArrays:
    """Flatten a :class:`CrystalECP` into libecpint primitive arrays.

    The conversion is bookkeeping -- no algebraic rewrite of the ECP.
    Each :class:`CrystalECPTerm` becomes one primitive with its
    (alpha, coefficient) carried over verbatim; the ``ell`` label is
    re-keyed onto libecpint's per-primitive ``am`` convention and the
    radial power is converted between the two conventions (see
    :class:`LibecpintInlineArrays` for both mappings).

    Phase 14g pairs this with a C++ ``compute_ecp_matrix_inline``
    that calls libecpint's ``set_ecp_basis(...)`` (the inline-
    primitive form) instead of ``set_ecp_basis_from_library(...)``
    (the XML form used today). That unblocks the 5th-period pob
    basis sets (Rb-I, Cs-Po) and any future inline-ECP archives
    (vDZP, mixed-row dhf-*).

    Parameters
    ----------
    ecp
        Parsed CRYSTAL ECP block from :func:`parse_crystal_atom_basis`.

    Returns
    -------
    LibecpintInlineArrays
        Bundle ready to feed libecpint's inline-primitive API.

    Raises
    ------
    ValueError
        If a term carries an unrecognised ``ell`` label (not
        ``"local"`` and not in 0..4).
    """
    # Determine the libecpint local-channel l-index. The XML convention
    # places the local at ``max_proj_ell + 1`` so it sits strictly above
    # every projector. When no projectors are present (rare; only valid
    # for trivial test cases), fall back to local_ell=0 -- libecpint will
    # still treat the single channel as the local one since it's
    # max-l.
    max_proj_ell = -1
    for ell_idx, m_count in enumerate(ecp.m_per_ell):
        if m_count > 0:
            max_proj_ell = max(max_proj_ell, ell_idx)
    local_ell = max_proj_ell + 1 if max_proj_ell >= 0 else 0

    exponents: list[float] = []
    coefficients: list[float] = []
    ams: list[int] = []
    ns: list[int] = []

    for term in ecp.terms:
        if term.ell == "local":
            assigned_l = local_ell
        elif isinstance(term.ell, int) and 0 <= term.ell <= 4:
            assigned_l = term.ell
        else:
            raise ValueError(
                f"crystal_ecp_to_libecpint_arrays: term carries "
                f"unrecognised ell={term.ell!r}; expected "
                f"'local' or int in 0..4."
            )
        exponents.append(term.alpha)
        coefficients.append(term.coefficient)
        ams.append(assigned_l)
        # CRYSTAL's NKL is the literal power of r (CRYSTAL23 eqs. 3.18-3.19:
        # r^n C exp(-alpha r^2)); libecpint takes the Gaussian-format power
        # and stores n - 2 (GaussianECP's constructor). Passing NKL through
        # unconverted turned every Stuttgart r^0 term into a singular r^-2
        # one -- 40.9 Ha on an AgCl molecule (#207).
        ns.append(term.n_pow + 2)

    if local_ell not in ams:
        # libecpint reads its local channel off the highest am present. A
        # CRYSTAL record with no local terms (M = 0, as on every pob record)
        # would hand the highest projector -- f -- to all angular momenta
        # instead of projecting it. libecpint's own XML library carries a
        # zero placeholder for exactly this (ecp10mdf.xml: local at lval=4),
        # and :func:`emit_ecp_sidecar` already writes one (#207).
        exponents.append(1.0)
        coefficients.append(0.0)
        ams.append(local_ell)
        ns.append(2)

    return LibecpintInlineArrays(
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        ams=tuple(ams),
        ns=tuple(ns),
        local_ell=local_ell,
        max_proj_ell=max_proj_ell,
    )


# --------------------------------------------------------------------------
# .ecp sidecar emitter (BSE / Gaussian ECP block layout)
# --------------------------------------------------------------------------

_AM_LETTERS = "spdfghik"


def emit_ecp_sidecar(
    atoms: Sequence[CrystalAtomBasis], header_comment: str = ""
) -> str:
    """Write the ECP blocks of CRYSTAL atom records as a ``.ecp`` sidecar.

    The layout is the one ``scripts/basisset_dev/split_ecp_g94.py`` leaves
    next to every ECP-bearing orbital file and
    :func:`vibeqc.ecp_metadata.parse_inline_ecp_sidecar` reads back::

        Ag     0
        Ag-ECP     4     28
        g potential
          1
        2      1.0     0.0
        s-g potential
          ...

    Channels come straight from :func:`crystal_ecp_to_libecpint_arrays`,
    so the sidecar round-trips to the exact primitive arrays the periodic
    driver feeds libecpint; the header ``lmax`` is that bridge's local
    channel and ``ncore = Z - Z_eff``.  A record without local terms gets
    the conventional zero placeholder so the local channel is present.
    Atoms without an ECP are skipped.
    """
    lines: list[str] = []
    for line in header_comment.splitlines():
        lines.append(f"! {line}".rstrip())
    if header_comment:
        lines.append("!")
    for ab in atoms:
        if not (ab.has_ecp and ab.ecp is not None):
            continue
        arrays = crystal_ecp_to_libecpint_arrays(ab.ecp)
        lmax = int(arrays.local_ell)
        ncore = int(ab.Z) - round(float(ab.ecp.znuc))
        sym = ab.element_symbol
        by_l: dict[int, list[tuple[int, float, float]]] = {}
        for exp, coef, am, n in zip(
            arrays.exponents, arrays.coefficients, arrays.ams, arrays.ns
        ):
            by_l.setdefault(int(am), []).append((int(n), float(exp), float(coef)))
        local_letter = _AM_LETTERS[lmax]
        lines.append(f"{sym:<6s} 0")
        lines.append(f"{sym}-ECP {lmax:5d} {ncore:5d}")

        def _channel(label: str, terms: list[tuple[int, float, float]]) -> None:
            lines.append(f"{label} potential")
            lines.append(f"  {len(terms)}")
            for n, exp, coef in terms:
                lines.append(f"{n:d}  {exp!r:<24s} {coef!r}")

        _channel(local_letter, by_l.get(lmax) or [(2, 1.0, 0.0)])
        for ell in range(lmax):
            if ell in by_l:
                _channel(f"{_AM_LETTERS[ell]}-{local_letter}", by_l[ell])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# NWChem / .g94 emitter
# --------------------------------------------------------------------------


def build_periodic_ecp_data(
    system: "PeriodicSystem",
    atoms: list[CrystalAtomBasis],
) -> tuple:
    """Build ECP data for a PeriodicSystem from parsed CRYSTAL atom bases."""
    from vibeqc._vibeqc_core import ECPPrimitiveBlock as _ECPPrimitiveBlock

    by_z: dict[int, CrystalAtomBasis] = {}
    for ab in atoms:
        by_z[ab.Z] = ab

    ecp_primitive_blocks: list = []
    ecp_home_centers: list = []
    effective_charges: list[float] = []
    total_ncore = 0

    for atom in system.unit_cell:
        z = int(atom.Z)
        ab = by_z.get(z)
        if ab is None or not ab.has_ecp or ab.ecp is None:
            effective_charges.append(float(z))
            continue
        ecp_data = ab.ecp
        arrays = crystal_ecp_to_libecpint_arrays(ecp_data)
        block = _ECPPrimitiveBlock()
        block.n_primitive = arrays.n_primitives
        block.exponents = list(arrays.exponents)
        block.coefficients = list(arrays.coefficients)
        block.ams = list(arrays.ams)
        block.ns = list(arrays.ns)
        ecp_primitive_blocks.append(block)
        ecp_home_centers.append(list(atom.xyz))
        z_eff = ecp_data.znuc
        ncore = z - int(z_eff)
        effective_charges.append(z_eff)
        total_ncore += ncore

    return (ecp_primitive_blocks, ecp_home_centers, effective_charges, total_ncore)


def emit_g94(atoms: Sequence[CrystalAtomBasis], header_comment: str = "") -> str:
    """Convert a list of per-element basis specs into a single .g94
    (Gaussian / NWChem format) string that libint can load."""
    lines: list[str] = []
    if header_comment:
        for ln in header_comment.splitlines():
            lines.append(f"! {ln}")
        lines.append("")

    for atom in atoms:
        # ECP-bearing atoms (CRYSTAL Z+200 convention) carry a valence
        # orbital basis in their shells — emit those normally, just skip
        # the ECP block (which libint can't read). The ECP data travels
        # through the libecpint inline-primitive path instead.
        lines.append("****")
        lines.append(f"{atom.element_symbol:<2s}    0")
        for shell in atom.shells:
            # NWChem format uses a leading keyword + NG + 1.00 (unused).
            lines.append(f"{shell.shell_type:<2s}  {shell.n_primitives:2d}   1.00")
            for i, exp in enumerate(shell.exponents):
                if shell.shell_type == "SP":
                    lines.append(
                        f"  {exp:18.10f}  {shell.coefficients[i]:18.10f}"
                        f"  {shell.coefficients_p[i]:18.10f}"
                    )
                else:
                    lines.append(f"  {exp:18.10f}  {shell.coefficients[i]:18.10f}")
    lines.append("****")
    lines.append("")
    return "\n".join(lines)


# CRYSTAL LAT-code <- string lookup for the emitter.
_SHELL_TO_LAT = {"S": 0, "SP": 1, "P": 2, "D": 3, "F": 4, "G": 5}

# Default formal occupancies per shell type (used when CrystalShell.occupancy
# is 0.0 because the data came from a .g94 file that doesn't carry it).
_DEFAULT_OCCUPANCY = {"S": 2, "SP": 2, "P": 6, "D": 10, "F": 14, "G": 18}


def _occ(shell: CrystalShell) -> float:
    """Return formal electron count per shell for CRYSTAL14 neutrality.

    CRYSTAL14 requires the sum of CHE values to equal the nuclear
    charge.  Use proper electron counts: 2.0 for s/sp, 6.0 for p,
    10.0 for d, 14.0 for f.  Polarisation shells (single-primitive,
    occupancy=0.0) stay at 0.0.
    """
    if shell.occupancy > 0.0:
        return shell.occupancy
    # Single-primitive shells with occ=0.0 are polarisation functions.
    if shell.n_primitives == 1:
        return 0.0
    return float(_DEFAULT_OCCUPANCY.get(shell.shell_type, 0))


# --------------------------------------------------------------------------
# CRYSTAL inline-basis emitter (for .d12 files)
# --------------------------------------------------------------------------


def emit_crystal(atoms: Sequence[CrystalAtomBasis]) -> str:
    """Convert per-element basis specs into CRYSTAL inline-basis format.

    Produces text suitable for pasting into a CRYSTAL ``.d12`` input
    deck after the geometry block (``ENDGEOM``).  The format is::

        Z NSHELL
        0 LAT NPG OCC SCALE
          exp1  coeff1
          exp2  coeff2
        ...
         99 0
        END

    where LAT = 0(S), 1(SP), 2(P), 3(D), 4(F), 5(G).  For SP shells
    each primitive line carries both s and p coefficients.

    ECP-bearing atoms (``has_ecp=True``) are emitted with ``200+Z``
    as the header number; the ECP block itself is **not** emitted
    (ECP inline emission needs a separate function -- libecpint
    integration pending).  All-electron atoms emit the raw Z.

    Parameters
    ----------
    atoms
        Sequence of :class:`CrystalAtomBasis` objects, typically
        parsed from the Bredow-group per-element CRYSTAL files via
        :func:`parse_crystal_atom_basis` or :func:`parse_crystal_atom_basis_file`.

    Returns
    -------
    str
        Multi-line CRYSTAL basis-set block ready for insertion into
        a ``.d12`` deck.
    """
    blocks: list[str] = []
    for atom in atoms:
        z_header = atom.Z + 200 if atom.has_ecp else atom.Z
        nshell = len(atom.shells)
        lines: list[str] = [f"{z_header} {nshell}"]
        for shell in atom.shells:
            lat = _SHELL_TO_LAT.get(shell.shell_type)
            if lat is None:
                raise ValueError(
                    f"{atom.element_symbol}: unknown shell type {shell.shell_type!r}"
                )
            npg = shell.n_primitives
            occ = _occ(shell)
            scale = 0.0
            lines.append(f"0 {lat} {npg:3d}  {occ:.1f} {scale:.1f}")
            for i in range(npg):
                exp = shell.exponents[i]
                if shell.shell_type == "SP":
                    c_s = shell.coefficients[i]
                    c_p = shell.coefficients_p[i]
                    lines.append(f"{exp:20.10f}  {c_s:20.10f}  {c_p:20.10f}")
                else:
                    c = shell.coefficients[i]
                    lines.append(f"{exp:20.10f}  {c:20.10f}")
        blocks.append("\n".join(lines))

    # Join per-element blocks with a blank line, then close with END.
    return "\n".join(blocks) + "\n 99 0"


# --------------------------------------------------------------------------
# Bredow-group archive fetcher
# --------------------------------------------------------------------------

# Canonical registry of the all-electron Bredow archives (light elements,
# H-Br). ECP-bearing archives are known but not fetched until libecpint
# support lands; see _BREDOW_ARCHIVES_ECP below.
_BREDOW_BASE = "https://www.chemie.uni-bonn.de/bredow/de/software"

# Two modes of distribution:
#   "archive": URL returns a tar.gz / zip archive.
#   "listing": URL returns an HTML page with per-element subpages; every
#              element page embeds its file contents inside a <p> block.
# The fetcher auto-dispatches on the magic bytes of the response.
_BREDOW_ARCHIVES = {
    # key = local basis name -> (URL, pretty-name)
    "pob-tzvp": (f"{_BREDOW_BASE}/pob-tzvp-tar.gz", "pob-TZVP"),
    "pob-tzvp-rev2": (f"{_BREDOW_BASE}/pob-tzvp-rev2-tar.gz", "pob-TZVP-rev2"),
    # pob-DZVP-rev2 is served only as an HTML listing with embedded file
    # bodies -- the fetcher transparently scrapes it.
    "pob-dzvp-rev2": (f"{_BREDOW_BASE}/pob-dzvp-rev2", "pob-DZVP-rev2"),
}

_BREDOW_ARCHIVES_ECP = {
    # Fetched alongside the matching all-electron archive.
    "pob-dzvp-rb-i": f"{_BREDOW_BASE}/pob-dzvp-rb-i-tar.gz",
    "pob-tzvp-rb-i": f"{_BREDOW_BASE}/pob-tzvp-rb-i-tar.gz",
    "pob-tzvp-rev2-rb-i": f"{_BREDOW_BASE}/pob-tzvp-ref2-rb-i-tar.gz",
    "pob-tzvp-rev2-cs-po": f"{_BREDOW_BASE}/pob-tzvp-rev2-cs-po-tar.gz",
    "pob-tzv-rev2-la-lu": f"{_BREDOW_BASE}/pob-tzv-rev2-la-lu-tar.gz",
}

# Mapping from all-electron basis key to ECP archive keys to fetch
# alongside it.  A missing entry means no ECP archives are known.
_BREDOW_AE_TO_ECP_KEYS: dict[str, list[str]] = {
    "pob-tzvp-rev2": [
        "pob-tzvp-rev2-rb-i",
        "pob-tzvp-rev2-cs-po",
        "pob-tzv-rev2-la-lu",
    ],
    "pob-tzvp": [
        "pob-tzvp-rb-i",
        "pob-tzv-rev2-la-lu",
    ],
    "pob-dzvp-rev2": [
        "pob-dzvp-rb-i",
    ],
}

# Per-basis citation text surfaced to the user on successful fetch.
_BREDOW_CITATIONS = {
    "pob-tzvp": (
        "Peintinger, Vilela Oliveira, Bredow. J. Comput. Chem. 34, 451-459 "
        "(2013). DOI: 10.1002/jcc.23153; Rb-I: Laun, Vilela Oliveira, Bredow. "
        "J. Comput. Chem. 39, 1285-1290 (2018). DOI: 10.1002/jcc.25195"
    ),
    "pob-tzvp-rev2": (
        "Vilela Oliveira, Laun, Peintinger, Bredow. J. Comput. Chem. 40, "
        "2364-2376 (2019). DOI: 10.1002/jcc.26013"
    ),
    "pob-dzvp-rev2": (
        "Vilela Oliveira, Laun, Peintinger, Bredow. J. Comput. Chem. 40, "
        "2364-2376 (2019). DOI: 10.1002/jcc.26013"
    ),
}


def _download(url: str, timeout: float = 60.0, retries: int = 3) -> bytes:
    """Fetch a URL to memory; tolerates both tar.gz archives and single files.

    The Bredow Plone server is fast for tar.gz archives and slower for
    per-element HTML pages. Retries transparently on transient timeouts.
    """
    import time

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as err:
            last_err = err
            time.sleep(1.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


_ELEMENT_LINK_RE = re.compile(
    r'href="(?P<url>[^"]*?/(?P<name>\d{1,3}_[a-z]+))"',
    re.IGNORECASE,
)
# File contents inside a Plone page are wrapped in a single <p>...</p> block
# with <br /> line separators. We extract that and normalize it to plain
# text. The content block is identified by starting with "ZZ_SYM" or by
# having the characteristic "Z N" first data line -- we match the former.
_PLONE_FILE_BLOCK_RE = re.compile(
    r"<p>\s*(?:<strong>[^<]*</strong>\s*<br\s*/>)?(?P<body>.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)


def _unescape_plone_body(body: str) -> str:
    """Convert a Plone <p>...<br />...</p> blob to plain text lines."""
    # Collapse <br/> in any form to newline, drop remaining tags.
    t = re.sub(r"<br\s*/?>", "\n", body)
    t = re.sub(r"</?strong>", "", t)
    t = re.sub(r"<[^>]+>", "", t)  # any stray tag
    # HTML entities we might encounter.
    t = t.replace("&nbsp;", " ").replace("&amp;", "&")
    # Unicode non-breaking spaces frequently appear in scraped text.
    t = t.replace("\u00a0", " ")
    return t.strip()


def _iter_archive_members(data: bytes) -> Iterator[tuple[str, bytes]]:
    """Yield (name, bytes) for every member of a tar.gz or zip archive.

    Bredow's server serves some URLs without extensions (raw tar.gz bytes);
    we autodetect the magic bytes.
    """
    # Tarball (gzip-compressed).
    if data[:2] == b"\x1f\x8b":
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                yield member.name, f.read()
        return
    # Zip.
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                yield name, zf.read(name)
        return
    raise RuntimeError("unknown archive format (not tar.gz or zip)")


def _iter_plone_listing_members(
    data: bytes, base_url: str
) -> Iterator[tuple[str, bytes]]:
    """Walk a Plone listing page and fetch every per-element subpage,
    returning the embedded file body as bytes.

    ``base_url`` is the listing URL -- used to resolve relative links and
    to de-duplicate cross-references.
    """
    html = data.decode("utf-8", errors="replace")
    seen: set[str] = set()
    for match in _ELEMENT_LINK_RE.finditer(html):
        url = match.group("url")
        name = match.group("name")
        # Normalize: ``06_c`` -> ``06_C`` so the parser's filename regex picks it up.
        parts = name.split("_", 1)
        if len(parts) == 2:
            name = f"{parts[0]}_{parts[1].capitalize()}"
        # Stay on the same Bredow basis listing -- reject stray nav links.
        if "/bredow/" not in url:
            continue
        if url in seen:
            continue
        seen.add(url)

        sub_html = _download(url)
        sub_text = sub_html.decode("utf-8", errors="replace")
        # The Plone layout wraps the per-element page in several <p>
        # blocks; the first typically contains just the element header
        # ("<strong>06_C</strong><br />"), the next contains the actual
        # basis. Iterate and pick the first body that looks like CRYSTAL
        # input (starts with ZZ NSHELL and has numeric data).
        body = ""
        for m in _PLONE_FILE_BLOCK_RE.finditer(sub_text):
            candidate = _unescape_plone_body(m.group("body"))
            if not candidate:
                continue
            first_line = candidate.splitlines()[0].split()
            if (
                len(first_line) >= 2
                and first_line[0].isdigit()
                and first_line[1].isdigit()
            ):
                body = candidate
                break
        if not body:
            continue
        yield name, body.encode("utf-8")


_FILENAME_RE = re.compile(r"^\s*(?:.*/)?(\d{2,3})_([A-Z][a-z]?)\s*$")


def fetch_bredow_basis_sets(
    dest_dir: str | Path | None = None,
    names: Iterable[str] | None = None,
    *,
    verbose: bool = True,
    return_atoms: bool = False,
) -> dict[str, Path] | tuple[dict[str, Path], dict[str, list[CrystalAtomBasis]]]:
    """Download the Bredow group all-electron basis sets and convert them
    to .g94 files loadable by libint.

    Parameters
    ----------
    dest_dir
        Target directory for the .g94 output files. Defaults to
        ``basis_library/basis/`` inside the vibe-qc source tree (the path
        already on ``LIBINT_DATA_PATH`` in editable installs).
    names
        Subset of basis-set keys to fetch (from ``_BREDOW_ARCHIVES``).
        ``None`` fetches all supported light-element archives.
    verbose
        Print progress + citation info.

    Returns
    -------
    dict mapping basis name -> path of the generated .g94 file.
    """
    if dest_dir is None:
        # Default target: python/vibeqc/basis_library/custom/ — this is the
        # committed-to-repo staging area inside the Python package.
        # scripts/setup_basis_library.sh copies custom/*.g94 on top of
        # libint's stock bases to populate build/basis_library/basis/,
        # which is what LIBINT_DATA_PATH points at.
        # Writing here keeps the generated files under version control.
        here = Path(__file__).resolve().parent
        dest_dir = here / "basis_library" / "custom"
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    keys = list(names) if names is not None else list(_BREDOW_ARCHIVES)
    results: dict[str, Path] = {}
    parsed_atoms: dict[str, list[CrystalAtomBasis]] = {}

    for key in keys:
        if key not in _BREDOW_ARCHIVES:
            raise KeyError(
                f"Unknown Bredow basis key {key!r}. Known: {sorted(_BREDOW_ARCHIVES)}"
            )
        url, _inner = _BREDOW_ARCHIVES[key]
        if verbose:
            print(f"[fetch] {key:<15s} <- {url}")
        data = _download(url)

        # Autodetect: archive (tar.gz / zip) or HTML listing.
        if data[:2] == b"\x1f\x8b" or data[:4] == b"PK\x03\x04":
            members: Iterable[tuple[str, bytes]] = list(_iter_archive_members(data))
        elif data.lstrip()[:5].lower() in (b"<!doc", b"<html"):
            members = list(_iter_plone_listing_members(data, url))
        else:
            raise RuntimeError(f"{key}: unrecognised server response format")

        # Parse every per-element file into a CrystalAtomBasis, sorted by Z.
        atoms_by_z: dict[int, CrystalAtomBasis] = {}
        for member_name, content in members:
            m = _FILENAME_RE.match(member_name)
            if not m:
                continue  # skip top-level directory entries or READMEs
            atom = parse_crystal_atom_basis(
                content.decode("utf-8", errors="replace"),
                source=member_name,
            )
            atoms_by_z[atom.Z] = atom

        if not atoms_by_z:
            raise RuntimeError(f"{key}: archive contained no parseable elements")

        # ---- Fetch ECP-bearing archives for heavy elements --------------
        # The all-electron archive covers H-Br; heavy elements (Rb-Po,
        # lanthanides) live in separate ECP-bearing archives whose
        # per-element files carry both a valence orbital basis and an
        # inline ECP block. Parse and merge them so the .g94 includes
        # every element the basis family supports.
        ecp_keys = _BREDOW_AE_TO_ECP_KEYS.get(key, [])
        ecp_urls: list[str] = []
        for ek in ecp_keys:
            if ek in _BREDOW_ARCHIVES_ECP:
                ecp_urls.append(_BREDOW_ARCHIVES_ECP[ek])
        for ek, eu in zip(ecp_keys, ecp_urls):
            if verbose:
                print(f"[fetch] {ek:<15s} <- {eu}")
            try:
                edata = _download(eu)
            except Exception as exc:
                if verbose:
                    print(
                        f"        WARNING: download failed ({exc}); "
                        f"skipping heavy elements"
                    )
                continue
            if edata[:2] == b"\x1f\x8b" or edata[:4] == b"PK\x03\x04":
                emembers: Iterable[tuple[str, bytes]] = list(
                    _iter_archive_members(edata)
                )
            else:
                if verbose:
                    print(f"        WARNING: unrecognised format; skipping")
                continue
            n_merged = 0
            for member_name, content in emembers:
                m = _FILENAME_RE.match(member_name)
                if not m:
                    continue
                eatom = parse_crystal_atom_basis(
                    content.decode("utf-8", errors="replace"),
                    source=member_name,
                )
                # ECP atoms carry the orbital valence basis in their
                # shells and the ECP operator data in their .ecp field.
                # Merge: ECP atom replaces any all-electron entry for
                # the same Z (light-element all-electron set doesn't
                # overlap ECP Z ranges, but this is defensive).
                atoms_by_z[eatom.Z] = eatom
                n_merged += 1
            if verbose:
                print(f"        merged {n_merged} ECP-bearing elements")
        # -----------------------------------------------------------------

        ordered = [atoms_by_z[z] for z in sorted(atoms_by_z)]
        if return_atoms:
            parsed_atoms[key] = ordered
        citation = _BREDOW_CITATIONS.get(key, "")
        header = (
            f"vibeqc-generated from Bredow-group archive '{key}'.\n"
            f"Source URL: {url}\n"
            f"Elements: {', '.join(a.element_symbol for a in ordered)}\n"
            + (f"Cite: {citation}\n" if citation else "")
        )
        g94 = emit_g94(ordered, header_comment=header)

        out = dest / f"{key}.g94"
        out.write_text(g94)
        results[key] = out
        if verbose:
            print(f"[write] {out}  ({len(ordered)} elements)")
            if citation:
                print(f"        cite: {citation}")
    if return_atoms:
        return results, parsed_atoms
    return results


# --------------------------------------------------------------------------
# CLI:  python -m vibeqc.basis_crystal [fetch | convert <file.g94> <in.crystal>]
# --------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage:\n"
            "  python -m vibeqc.basis_crystal fetch "
            "[--dest DIR] [name ...]\n"
            "       Download Bredow-group basis sets and convert to "
            ".g94 for libint.\n"
            "  python -m vibeqc.basis_crystal convert IN.crystal OUT.g94\n"
            "       Convert a CRYSTAL per-element file to a single-atom "
            ".g94 block."
        )
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "fetch":
        dest = None
        if rest and rest[0] == "--dest":
            dest = rest[1]
            rest = rest[2:]
        fetch_bredow_basis_sets(dest_dir=dest, names=rest if rest else None)
        return 0
    if cmd == "convert":
        if len(rest) != 2:
            print("convert expects 2 arguments: input output")
            return 2
        atom = parse_crystal_atom_basis_file(rest[0])
        Path(rest[1]).write_text(
            emit_g94([atom], header_comment=f"converted from {rest[0]}")
        )
        return 0
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
