"""Auto-populate :class:`ECPCenter` lists from a basis name + a Molecule.

Phase 14e of the libecpint integration: given a basis like ``lanl2dz``,
``dhf-tzvp``, etc. -- one of the 13 ECP-bearing bases that
``scripts/basisset_dev/split_ecp_g94.py`` produced ``<name>.ecp``
sidecar files for -- read the sidecar, extract per-element ``ncore``,
and produce a ready-to-go ``opts.ecp_centers`` + ``opts.ecp_library``
without making the user do the bookkeeping by hand.

Public surface:

* :func:`parse_sidecar_path(path)` -- read one ``<name>.ecp`` file,
  return a list of :class:`EcpHeader` (one per element block).
* :func:`parse_inline_ecp_sidecar(path)` -- read the same sidecar and
  return libecpint-ready primitive arrays for ECPs whose cores do not
  map onto a bundled XML library (vDZP-style custom ECPs).
* :func:`sidecar_path_for(basis_name)` -- locate ``<name>.ecp`` next
  to ``<name>.g94`` under ``$LIBINT_DATA_PATH/basis/``. Returns
  ``None`` if the basis has no ECP sidecar (i.e. is all-electron).
* :func:`basis_sidecar_replaces_core(mol, basis)` -- report whether a
  positive-core sidecar record applies to any atom in ``mol``.
* :func:`basis_sidecar_has_ecp_operator(mol, basis)` -- stricter detector for
  drivers that cannot apply even a zero-core model potential.
* :func:`molecular_options_replace_core(options)` -- report whether an
  options object explicitly activates a positive-core molecular ECP.
* :func:`molecular_options_request_ecp_operator(options)` -- stricter manual
  option detector for drivers that do not implement an ECP Hamiltonian.
* :func:`molecular_result_has_ecp_operator(result)` -- inspect authoritative
  SCF-result provenance, including zero-core model potentials.
* :func:`validate_gradient_ecp_contract(...)` -- prove that a molecular
  analytic gradient will use the exact XML-library or inline-primitive ECP
  input recorded by the SCF result.
* :func:`refuse_molecular_ecp_derivative_route(...)` -- shared fail-closed
  boundary for derivative drivers that cannot reproduce an ECP Hamiltonian.
* :func:`library_for(basis_name, ncore)` -- pick the right libecpint
  XML library for a given basis-name + ncore combination. Returns
  ``None`` for non-standard ncores like vDZP's per-element customs.
* :func:`auto_ecp_centers(mol, basis_name, library_name=None)` -- the
  one-call helper. Returns ``(ecp_centers, library_name)`` ready to
  drop into ``RHFOptions`` / ``UHFOptions`` / ``RKSOptions`` /
  ``UKSOptions``.
* :func:`ecp_centers_from_dict(spec, molecule, *, basis_name)` --
  dict-based convenience: ``{"I": "dhf"}`` → ``(ecp_centers, library)``.
* :func:`attach_inline_ecp_options_from_basis_sidecar(options, mol, basis)` --
  attach inline primitive sidecar ECP data to molecular SCF options when
  no standard XML library can represent the basis.
* :func:`validate_ecp_required(options, molecule, basis_name)` --
  safety check that raises ``ValueError`` when a basis requires an ECP
  but the options carry no ECP centres (BUG 99 guard).
* :func:`ecp_centre_atom_indices(options, molecule)` -- map every ECP
  centre on the options (XML-library ``ecp_centers`` and inline
  ``ecp_primitive_centers``) onto the atom it sits on; raises
  ``ValueError`` for a centre that coincides with no atom.
* :func:`ecp_centres_follow_atoms(options, reference, displaced)` --
  context manager that repositions those centres from the atoms of one
  geometry to the same atoms of another (finite-difference
  displacements, #576) and restores them on exit.
* :func:`options_carry_ecp(options)` -- ``True`` when either ECP route is
  populated on a molecular SCF options struct.
* :func:`reposition_ecp_centres(options, molecule, atom_indices)` --
  permanently move the centres onto the atoms of ``molecule`` using a
  mapping from :func:`ecp_centre_atom_indices` (drivers that keep one
  options object across geometries: the ASE calculator, the runner's
  optimiser hand-back, #643).
* :func:`is_ecp_paired_basis(basis_name)` --
  naming-convention heuristic: returns ``True`` for ECP-paired basis
  families (dhf-*, *-PP, lanl*, vdzp, x2c-*, *ecp*).

Caveats and legacy helper scope (from `docs/user_guide/ecp.md`):

* **Mixed-library molecules remain unsupported in the XML-only helper.**
  ``auto_ecp_centers`` returns a single ``ecp_library`` string, so it
  raises ``ValueError`` when the molecule's atoms span more than one
  libecpint XML library. The molecular SCF wrappers use the inline-
  primitive sidecar path when a standard XML library cannot represent
  the basis data.
* **Non-standard ncore values** (vDZP's B/C/N/etc with ``ncore=2``,
  ``ncore=3``, etc.) match no standard libecpint library; auto-
  population raises ``NotImplementedError`` because that helper is
  intentionally XML-only. vDZP's inline ECPs are valid data and are
  consumed automatically by ``run_rhf`` / ``run_rks`` / ``run_uhf`` /
  ``run_uks`` through their inline primitive options.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterator, Optional


# ---------- Element-symbol lookup -------------------------------------------
#
# BSE-style sidecars come in mixed-case (Na, Mg) AND all-caps (NA, MG)
# depending on the era of the source data (LANL files are all-caps,
# dhf is mixed). Normalise to the canonical title-case symbol on
# parse.

_SYMBOL_TO_Z: dict[str, int] = {
    sym: z for z, sym in enumerate([
        "",   "H",  "He",
        "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
        "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe",
        "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
        "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
        "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
        "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn",
        # Actinides (Z=87..103) covered for LANL08 / lanl2dz heavy
        # elements. Without these, lanl2dz.ecp's "U-ECP" line fails
        # the parse.
        "Fr", "Ra",
        "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
        "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    ])
}


def _normalise_symbol(raw: str) -> str:
    """Map ``"NA"`` / ``"Na"`` / ``"na"`` to canonical ``"Na"``."""
    s = raw.strip()
    if not s:
        return s
    return s[0].upper() + s[1:].lower()


# ---------- Sidecar parser --------------------------------------------------

# BSE / NWChem ``.g94`` ECP-block convention:
#   <Sym>-ECP <lmax> <ncore>
# Confirmed against LANL2DZ sodium (NA-ECP 2 10 -> lmax=2 d-projection,
# ncore=10 = [Ne]) and dhf-TZVP rubidium (RB-ECP 4 28 -> lmax=4,
# ncore=28 = [Ar]3d¹⁰).
_ECP_HEADER_RE = re.compile(
    r"^\s*([A-Z][A-Za-z]?)-ECP\s+(\d+)\s+(\d+)\s*$"
)


@dataclass(frozen=True)
class EcpHeader:
    """One ``<Sym>-ECP <lmax> <ncore>`` row from a sidecar.

    Attributes
    ----------
    symbol : str
        Canonical title-case element symbol (``"Na"``, ``"Rb"``).
    Z : int
        Atomic number derived from ``symbol``.
    lmax : int
        Maximum angular momentum used in the ECP expansion.
    ncore : int
        Number of replaced core electrons. Subtract from atomic ``Z``
        to get the valence-electron count for the SCF.
    """
    symbol: str
    Z: int
    lmax: int
    ncore: int


@dataclass(frozen=True)
class InlineECPRecord:
    """One parsed Gaussian/NWChem ECP block with libecpint stream data.

    ``ams`` uses libecpint's convention directly: the local channel is
    carried at its declared local angular momentum and projectors use
    their projected angular momentum. For BSE/G94 sidecars this matches
    the ``<local> potential`` / ``<l>-<local> potential`` channel labels.
    """

    header: EcpHeader
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]
    ams: tuple[int, ...]
    ns: tuple[int, ...]

    @property
    def n_primitives(self) -> int:
        return len(self.exponents)


def parse_sidecar_path(path: Path) -> list[EcpHeader]:
    """Parse a ``.ecp`` sidecar file. Returns one :class:`EcpHeader`
    per element block, in source order. Skips comments and blank
    lines; raises :class:`ValueError` on malformed headers.
    """
    text = Path(path).read_text(errors="replace")
    out: list[EcpHeader] = []
    for line in text.splitlines():
        m = _ECP_HEADER_RE.match(line)
        if not m:
            continue
        sym = _normalise_symbol(m.group(1))
        if sym not in _SYMBOL_TO_Z:
            raise ValueError(
                f"{path}: ECP header references unknown element "
                f"{m.group(1)!r}"
            )
        out.append(EcpHeader(
            symbol=sym, Z=_SYMBOL_TO_Z[sym],
            lmax=int(m.group(2)), ncore=int(m.group(3)),
        ))
    return out


_AM_LABEL_TO_L: dict[str, int] = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
    "j": 7,
    "k": 8,
}
_ECP_CHANNEL_RE = re.compile(
    r"^\s*([spdfghijkSPDFGHIJK])(?:-([spdfghijkSPDFGHIJK]|ul|UL))?"
    r"\s+potential\s*$"
)
_ELEMENT_PREAMBLE_RE = re.compile(r"^\s*([A-Z][A-Za-z]?)\s+0\s*$")


def _parse_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def parse_inline_ecp_sidecar(path: Path) -> dict[int, InlineECPRecord]:
    """Parse a Gaussian/NWChem ``.ecp`` sidecar into inline ECP records.

    The sidecars produced by ``scripts/basisset_dev/split_ecp_g94.py`` use
    the usual BSE/Gaussian ECP layout::

        O     0
        O-ECP     3     2
        f potential
          1
        2      1.0     0.0
        s-f potential
          1
        2     10.4    50.7

    The local channel is the channel whose angular momentum equals the
    header ``lmax``; lower channels are the projected terms. The returned
    arrays are in the exact flat shape expected by libecpint
    ``ECPIntegrator::set_ecp_basis``.
    """

    lines = Path(path).read_text(errors="replace").splitlines()
    records: dict[int, InlineECPRecord] = {}
    i = 0
    while i < len(lines):
        clean = lines[i].split("!", 1)[0].strip()
        i += 1
        if not clean:
            continue
        m_header = _ECP_HEADER_RE.match(clean)
        if m_header is None:
            continue

        sym = _normalise_symbol(m_header.group(1))
        if sym not in _SYMBOL_TO_Z:
            raise ValueError(
                f"{path}: ECP header references unknown element "
                f"{m_header.group(1)!r}"
            )
        header = EcpHeader(
            symbol=sym,
            Z=_SYMBOL_TO_Z[sym],
            lmax=int(m_header.group(2)),
            ncore=int(m_header.group(3)),
        )

        exponents: list[float] = []
        coefficients: list[float] = []
        ams: list[int] = []
        ns: list[int] = []
        current_am: int | None = None
        expected_terms: int | None = None
        terms_seen = 0

        while i < len(lines):
            raw = lines[i]
            clean = raw.split("!", 1)[0].strip()
            if not clean:
                i += 1
                continue
            if (
                _ELEMENT_PREAMBLE_RE.match(clean)
                or _ECP_HEADER_RE.match(clean)
                or clean == "****"
            ):
                break

            m_channel = _ECP_CHANNEL_RE.match(clean)
            if m_channel is not None:
                label = m_channel.group(1).lower()
                if label not in _AM_LABEL_TO_L:
                    raise ValueError(
                        f"{path}: unsupported ECP angular momentum label "
                        f"{m_channel.group(1)!r} for {sym}"
                    )
                current_am = _AM_LABEL_TO_L[label]
                expected_terms = None
                terms_seen = 0
                i += 1
                continue

            if current_am is None:
                raise ValueError(
                    f"{path}: ECP primitive appears before a channel label "
                    f"for {sym}: {raw!r}"
                )

            parts = clean.split()
            if len(parts) == 1:
                expected_terms = int(parts[0])
                terms_seen = 0
                i += 1
                continue
            if len(parts) >= 3:
                ns.append(int(float(parts[0])))
                exponents.append(_parse_float(parts[1]))
                coefficients.append(_parse_float(parts[2]))
                ams.append(current_am)
                terms_seen += 1
                if expected_terms is not None and terms_seen > expected_terms:
                    raise ValueError(
                        f"{path}: too many ECP primitives in {sym} "
                        f"channel l={current_am}"
                    )
                i += 1
                continue

            raise ValueError(f"{path}: malformed ECP line for {sym}: {raw!r}")

        if expected_terms is not None and terms_seen != expected_terms:
            raise ValueError(
                f"{path}: ECP channel for {sym} expected {expected_terms} "
                f"terms, parsed {terms_seen}"
            )
        if not exponents:
            raise ValueError(f"{path}: ECP block for {sym} has no primitives")
        if header.lmax not in ams:
            raise ValueError(
                f"{path}: ECP block for {sym} does not include local "
                f"l={header.lmax} channel"
            )
        records[header.Z] = InlineECPRecord(
            header=header,
            exponents=tuple(exponents),
            coefficients=tuple(coefficients),
            ams=tuple(ams),
            ns=tuple(ns),
        )

    return records


def inline_ecp_data_for(mol, basis_name: str) -> tuple:
    """Return inline primitive ECP data for atoms in ``mol`` using ``basis``.

    The result is ``(primitive_blocks, centers, effective_charges,
    total_ncore)``. Empty primitive blocks mean the basis has no matching
    sidecar entries for the molecule. ``effective_charges`` is populated
    only when at least one inline ECP center is present.
    """

    sidecar = sidecar_path_for(basis_name)
    if sidecar is None:
        return [], [], [], 0
    records = parse_inline_ecp_sidecar(sidecar)
    if not records:
        return [], [], [], 0

    from ._vibeqc_core import ECPPrimitiveBlock

    blocks: list = []
    centers: list[list[float]] = []
    effective_charges: list[float] = []
    total_ncore = 0
    for atom in mol.atoms:
        z = int(atom.Z)
        rec = records.get(z)
        if rec is None:
            effective_charges.append(float(z))
            continue
        block = ECPPrimitiveBlock()
        block.n_primitive = rec.n_primitives
        block.exponents = list(rec.exponents)
        block.coefficients = list(rec.coefficients)
        block.ams = list(rec.ams)
        block.ns = list(rec.ns)
        blocks.append(block)
        centers.append(list(atom.xyz))
        effective_charges.append(float(z - rec.header.ncore))
        total_ncore += rec.header.ncore

    if not blocks:
        return [], [], [], 0
    return blocks, centers, effective_charges, total_ncore


def attach_inline_ecp_options_from_basis_sidecar(options, mol, basis) -> None:
    """Populate molecular SCF options from the basis' bundled ECP sidecar.

    The sidecar is the authoritative ECP for a bundled basis, and its
    primitives are attached *inline* on every route.  Until 2026-09 the
    helper preferred a libecpint XML library whenever the sidecar's core
    count happened to match one, which mis-paired vDZP (its Ag/Pt/I/Ba
    potentials are not the Stuttgart small-core ones: AgH/vDZP came out
    85 mHa off), failed on elements the XML lacks (Si, Cl, Ga, Kr), and
    refused any molecule spanning two core sizes (CsI/dhf-TZVP) although
    the sidecar carried both.  The XML libraries remain available to a
    caller who sets ``ecp_centers`` / ``ecp_library`` explicitly.

    Explicit caller-supplied XML centres or primitive blocks take
    precedence; a basis without a sidecar, or a molecule whose atoms the
    sidecar does not cover, leaves the options untouched (all-electron).
    """

    if options is None:
        return
    if getattr(options, "ecp_primitive_blocks", None):
        return
    if getattr(options, "ecp_centers", None):
        return
    basis_name = str(getattr(basis, "name", "") or "").strip()
    if not basis_name:
        return
    blocks, centers, effective_charges, total_ncore = inline_ecp_data_for(
        mol, basis_name
    )
    if not blocks:
        return
    options.ecp_primitive_blocks = blocks
    options.ecp_primitive_centers = centers
    options.ecp_effective_charges = effective_charges
    options.ecp_total_ncore = int(total_ncore)


def sidecar_path_for(basis_name: str) -> Optional[Path]:
    """Locate ``<basis_name>.ecp`` under ``$LIBINT_DATA_PATH/basis/``.

    Returns ``None`` when no sidecar exists (i.e. the basis is
    all-electron). The path resolution mirrors libint's: at vibe-qc
    import time ``__init__.py`` points ``$LIBINT_DATA_PATH`` at the
    bundled ``basis_library/`` directory; basis names resolve under
    ``<that>/basis/<name>.g94`` and we look for ``.ecp`` alongside.
    """
    name = f"{basis_name.lower()}.ecp"
    root = os.environ.get("LIBINT_DATA_PATH")
    if root:
        candidate = Path(root) / "basis" / name
        if candidate.is_file():
            return candidate

    bundled = Path(__file__).resolve().parent / "basis_library" / "basis" / name
    return bundled if bundled.is_file() else None


def basis_sidecar_replaces_core(molecule: object, basis: object) -> bool:
    """Return whether ``basis`` replaces core electrons in ``molecule``.

    ``basis`` may be either a basis-name string or an object exposing its
    name through ``.name``. A sidecar that contains no matching element, or
    only a zero-core model potential for the matching elements, returns
    ``False``.
    """

    basis_name = getattr(basis, "name", basis)
    if not isinstance(basis_name, str) or not basis_name.strip():
        return False
    sidecar = sidecar_path_for(basis_name)
    if sidecar is None:
        return False
    atomic_numbers = {int(atom.Z) for atom in molecule.atoms}
    return any(
        int(header.ncore) > 0 and int(header.Z) in atomic_numbers
        for header in parse_sidecar_path(sidecar)
    )


def basis_sidecar_has_ecp_operator(molecule: object, basis: object) -> bool:
    """Return whether a matching basis sidecar requests any ECP operator.

    Unlike :func:`basis_sidecar_replaces_core`, this includes zero-core model
    potentials. Drivers that do not apply an ECP Hamiltonian must use this
    stricter predicate; correlated consumers may use the core-replacement
    predicate when a genuine zero-core model potential is otherwise safe.
    """

    basis_name = getattr(basis, "name", basis)
    if not isinstance(basis_name, str) or not basis_name.strip():
        return False
    sidecar = sidecar_path_for(basis_name)
    if sidecar is None:
        return False
    atomic_numbers = {int(atom.Z) for atom in molecule.atoms}
    return any(
        int(header.Z) in atomic_numbers
        for header in parse_sidecar_path(sidecar)
    )


_MOLECULAR_ECP_OPERATOR_FIELDS = (
    "ecp_primitive_blocks",
    "ecp_primitive_centers",
    "ecp_home_centers",
    "ecp_centers",
    "ecp_effective_charges",
    "ecp_library",
)


def molecular_options_request_ecp_operator(options: object) -> bool:
    """Return whether options carry any explicit molecular ECP request.

    A nonempty operator field is authoritative even when the aggregate core
    count is zero. A nonzero or malformed count also fails closed because it
    is provenance for an ECP configuration whose operator data may already
    have been consumed or copied elsewhere.
    """

    if options is None:
        return False
    for name in _MOLECULAR_ECP_OPERATOR_FIELDS:
        value = getattr(options, name, None)
        if value is None:
            continue
        try:
            if len(value) > 0:
                return True
        except TypeError:
            if bool(value):
                return True
    try:
        return int(getattr(options, "ecp_total_ncore", 0) or 0) != 0
    except (TypeError, ValueError):
        return True


def molecular_result_has_ecp_operator(result: object) -> bool:
    """Return whether an SCF result was built with an ECP operator.

    New native results carry ``ecp_operator_applied`` because the aggregate
    replaced-core count alone cannot distinguish an all-electron reference
    from a genuine zero-core model potential.  The count remains a
    compatibility fallback for older result-like objects and is validated so
    malformed provenance fails closed rather than reaching a derivative or
    post-SCF consumer.
    """

    if result is None:
        return False
    try:
        ncore = int(getattr(result, "ecp_total_ncore", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid SCF ecp_total_ncore provenance") from exc
    if ncore < 0:
        raise ValueError("invalid negative ecp_total_ncore provenance")

    applied = getattr(result, "ecp_operator_applied", None)
    if applied is None:
        return ncore != 0
    applied = bool(applied)
    if ncore != 0 and not applied:
        raise ValueError(
            "inconsistent SCF ECP provenance: ecp_total_ncore is nonzero "
            "but ecp_operator_applied is false"
        )
    return applied


def _ecp_coordinate_signature(values: object, *, route: str, source: str) -> tuple:
    """Return one validated, exactly comparable Cartesian coordinate."""

    try:
        xyz = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{route}: invalid {source} ECP center") from exc
    if len(xyz) != 3 or not all(isfinite(value) for value in xyz):
        raise ValueError(f"{route}: invalid {source} ECP center")
    return xyz


def _xml_center_signature(center: object, *, route: str, source: str) -> tuple:
    """Return the exact ``(Z, xyz)`` identity of one XML ECP center."""

    try:
        atomic_number = int(center.Z)
        xyz_value = center.xyz
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{route}: invalid {source} XML ECP center") from exc
    if atomic_number <= 0:
        raise ValueError(f"{route}: invalid {source} XML ECP center")
    return atomic_number, _ecp_coordinate_signature(
        xyz_value, route=route, source=f"{source} XML"
    )


def _primitive_block_signature(
    block: object, *, route: str, source: str
) -> tuple:
    """Return all fields that define one inline ECP primitive block."""

    try:
        n_primitive = int(block.n_primitive)
        exponents = tuple(float(value) for value in block.exponents)
        coefficients = tuple(float(value) for value in block.coefficients)
        ams = tuple(int(value) for value in block.ams)
        ns = tuple(int(value) for value in block.ns)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{route}: invalid {source} inline ECP primitive block"
        ) from exc
    if (
        n_primitive < 1
        or len(exponents) != n_primitive
        or len(coefficients) != n_primitive
        or len(ams) != n_primitive
        or len(ns) != n_primitive
        or not all(isfinite(value) for value in exponents)
        or not all(isfinite(value) for value in coefficients)
    ):
        raise ValueError(
            f"{route}: invalid {source} inline ECP primitive block"
        )
    return n_primitive, exponents, coefficients, ams, ns


def _inline_route_signature(
    owner: object,
    molecule: object,
    *,
    route: str,
    source: str,
) -> tuple[Counter, tuple[float, ...], int]:
    """Return the paired primitive/center multiset, charges, and ncore."""

    blocks = list(getattr(owner, "ecp_primitive_blocks", None) or [])
    centers = list(getattr(owner, "ecp_primitive_centers", None) or [])
    if not blocks or len(blocks) != len(centers):
        raise ValueError(
            f"{route}: {source} inline ECP blocks and centers must be "
            "nonempty and paired one-to-one"
        )
    pairs = Counter(
        (
            _ecp_coordinate_signature(
                center, route=route, source=f"{source} inline"
            ),
            _primitive_block_signature(block, route=route, source=source),
        )
        for block, center in zip(blocks, centers)
    )
    try:
        charges = tuple(
            float(value)
            for value in (getattr(owner, "ecp_effective_charges", None) or [])
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{route}: invalid {source} inline ECP effective charges"
        ) from exc
    if (
        len(charges) != len(list(molecule.atoms))
        or not all(isfinite(value) for value in charges)
    ):
        raise ValueError(
            f"{route}: {source} inline ECP effective charges must contain "
            "one finite value per molecule atom"
        )
    try:
        total_ncore = int(getattr(owner, "ecp_total_ncore"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{route}: invalid {source} ecp_total_ncore provenance"
        ) from exc
    if total_ncore < 0:
        raise ValueError(
            f"{route}: invalid negative {source} ecp_total_ncore provenance"
        )
    return pairs, charges, total_ncore


def validate_gradient_ecp_contract(
    molecule: object,
    result: object,
    options: object,
    *,
    route: str,
) -> None:
    """Prove that a gradient reproduces the SCF's exact ECP Hamiltonian.

    High-level RHF/UHF/RKS/UKS results record whether their Hamiltonian is
    all-electron, XML-library ECP, or inline-primitive ECP.  A low-level
    ``run_*_scf_with_jk`` result instead has unknown provenance because its
    caller supplied an arbitrary one-electron matrix.  Unknown provenance is
    never interpreted as all-electron: the derivative fails closed.

    For an ECP reference, this check requires the gradient options to carry
    the same route and exact data.  XML centers are compared as a ``(Z, xyz)``
    multiset because their list order does not change the Hamiltonian. Inline
    blocks are compared together with their paired centers as a multiset;
    per-atom effective charges retain their exact molecule order.
    """

    if result is None or not bool(
        getattr(result, "ecp_provenance_verified", False)
    ):
        raise NotImplementedError(
            f"{route}: SCF ECP provenance is unknown or unverified; the "
            "gradient cannot prove that it differentiates the same "
            "one-electron Hamiltonian"
        )

    try:
        result_ncore = int(getattr(result, "ecp_total_ncore"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{route}: invalid SCF result ecp_total_ncore provenance"
        ) from exc
    if result_ncore < 0:
        raise ValueError(
            f"{route}: invalid negative SCF result ecp_total_ncore provenance"
        )
    if not hasattr(result, "ecp_operator_applied"):
        raise ValueError(
            f"{route}: verified SCF result lacks ecp_operator_applied "
            "provenance"
        )
    result_applied = bool(result.ecp_operator_applied)

    result_xml_centers = list(
        getattr(result, "ecp_xml_centers", None) or []
    )
    result_xml_library = str(
        getattr(result, "ecp_xml_library", "") or ""
    )
    result_inline_blocks = list(
        getattr(result, "ecp_primitive_blocks", None) or []
    )
    result_inline_centers = list(
        getattr(result, "ecp_primitive_centers", None) or []
    )
    result_inline_charges = list(
        getattr(result, "ecp_effective_charges", None) or []
    )
    result_has_xml = bool(result_xml_centers or result_xml_library)
    result_has_inline = bool(
        result_inline_blocks or result_inline_centers or result_inline_charges
    )
    if result_has_xml and result_has_inline:
        raise ValueError(
            f"{route}: SCF result mixes XML and inline ECP provenance"
        )
    if result_applied != bool(result_has_xml or result_has_inline):
        raise ValueError(
            f"{route}: inconsistent SCF ECP operator and route provenance"
        )
    if not result_applied and result_ncore != 0:
        raise ValueError(
            f"{route}: all-electron SCF result reports nonzero "
            "ecp_total_ncore"
        )

    option_xml_centers = list(getattr(options, "ecp_centers", None) or [])
    option_xml_library_raw = str(getattr(options, "ecp_library", "") or "")
    option_inline_blocks = list(
        getattr(options, "ecp_primitive_blocks", None) or []
    )
    option_inline_centers = list(
        getattr(options, "ecp_primitive_centers", None) or []
    )
    option_inline_charges = list(
        getattr(options, "ecp_effective_charges", None) or []
    )
    try:
        option_ncore = int(getattr(options, "ecp_total_ncore", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{route}: invalid GradientOptions ecp_total_ncore"
        ) from exc
    if option_ncore < 0:
        raise ValueError(
            f"{route}: invalid negative GradientOptions ecp_total_ncore"
        )
    option_has_xml = bool(option_xml_centers or option_xml_library_raw)
    option_has_inline = bool(
        option_inline_blocks
        or option_inline_centers
        or option_inline_charges
        or option_ncore
    )
    if option_has_xml and option_has_inline:
        raise ValueError(
            f"{route}: GradientOptions mix XML and inline ECP inputs"
        )

    if not result_applied:
        if option_has_xml or option_has_inline:
            raise ValueError(
                f"{route}: GradientOptions request an ECP operator but the "
                "verified SCF result is all-electron"
            )
        return

    if result_has_xml:
        if not option_has_xml or option_has_inline:
            raise ValueError(
                f"{route}: GradientOptions ECP route does not match the "
                "SCF result's XML-library route"
            )
        if not result_xml_centers or not result_xml_library:
            raise ValueError(
                f"{route}: incomplete SCF result XML ECP provenance"
            )
        option_library = option_xml_library_raw or "ecp10mdf"
        if option_library != result_xml_library:
            raise ValueError(
                f"{route}: GradientOptions XML ECP library "
                f"{option_library!r} does not match the SCF result "
                f"provenance {result_xml_library!r}"
            )
        option_centers = Counter(
            _xml_center_signature(center, route=route, source="GradientOptions")
            for center in option_xml_centers
        )
        result_centers = Counter(
            _xml_center_signature(center, route=route, source="SCF result")
            for center in result_xml_centers
        )
        if option_centers != result_centers:
            raise ValueError(
                f"{route}: GradientOptions XML ECP centers do not match the "
                "exact SCF result provenance"
            )

        # XML GradientOptions have no independent ncore field. Resolve the
        # count from their exact centers/library and compare it with the SCF
        # result instead of accepting an aggregate count as route identity.
        from . import ecp_effective_charges as _xml_effective_charges

        effective = [
            float(value)
            for value in _xml_effective_charges(
                molecule, option_xml_centers, option_library
            )
        ]
        bare = [float(atom.Z) for atom in molecule.atoms]
        if len(effective) != len(bare) or not all(
            isfinite(value) for value in effective
        ):
            raise ValueError(
                f"{route}: invalid XML ECP effective-charge provenance"
            )
        derived_ncore_float = sum(
            nuclear - valence for nuclear, valence in zip(bare, effective)
        )
        derived_ncore = int(round(derived_ncore_float))
        if (
            not isfinite(derived_ncore_float)
            or abs(derived_ncore_float - derived_ncore) > 1e-8
        ):
            raise ValueError(
                f"{route}: XML ECP inputs imply a non-integral replaced-core "
                f"count ({derived_ncore_float})"
            )
        if derived_ncore != result_ncore:
            raise ValueError(
                f"{route}: XML ECP inputs imply {derived_ncore} replaced "
                "core electrons but the SCF result reports "
                f"{result_ncore}"
            )
        return

    if not option_has_inline or option_has_xml:
        raise ValueError(
            f"{route}: GradientOptions ECP route does not match the SCF "
            "result's inline-primitive route"
        )
    result_pairs, result_charges, exact_result_ncore = _inline_route_signature(
        result, molecule, route=route, source="SCF result"
    )
    option_pairs, option_charges, exact_option_ncore = _inline_route_signature(
        options, molecule, route=route, source="GradientOptions"
    )
    if result_pairs != option_pairs:
        raise ValueError(
            f"{route}: GradientOptions inline ECP block/center pairs do not "
            "match the exact SCF result provenance"
        )
    if result_charges != option_charges:
        raise ValueError(
            f"{route}: GradientOptions inline ECP effective charges do not "
            "match the exact SCF result provenance"
        )
    if exact_result_ncore != result_ncore:
        raise ValueError(
            f"{route}: internally inconsistent SCF inline ECP ncore "
            "provenance"
        )
    if exact_option_ncore != result_ncore:
        raise ValueError(
            f"{route}: GradientOptions inline ECP total_ncore "
            f"{exact_option_ncore} does not match the SCF result "
            f"provenance {result_ncore}"
        )


def refuse_molecular_ecp_derivative_route(
    molecule: object,
    basis: object,
    *,
    options: object = None,
    result: object = None,
    route: str,
) -> None:
    """Reject an ECP request on a derivative route that is not ECP-safe.

    The check combines explicit options, automatic basis sidecars, and
    result-carried provenance.  In particular, zero-core model potentials
    are operators and therefore must not slip through a ``ncore == 0`` test.
    """

    active = molecular_options_request_ecp_operator(options)
    active = active or basis_sidecar_has_ecp_operator(molecule, basis)
    active = active or molecular_result_has_ecp_operator(result)
    if not active:
        return
    raise NotImplementedError(
        f"{route}: molecular ECP derivatives are not implemented on this "
        "route. The current derivative kernel would otherwise differentiate "
        "a bare all-electron Hamiltonian. Use an all-electron basis."
    )


def molecular_options_replace_core(options: object) -> bool:
    """Return whether molecular SCF options explicitly replace core.

    Inline primitive inputs carry their aggregate replaced-core count, so
    primitive blocks with ``ecp_total_ncore == 0`` remain valid model
    potentials. XML centers do not expose that count before native ECP
    construction and are therefore conservatively treated as core-replacing.
    Invalid explicit counts also fail closed.
    """

    if options is None:
        return False
    if getattr(options, "ecp_centers", None):
        return True
    if getattr(options, "ecp_primitive_blocks", None):
        try:
            return int(getattr(options, "ecp_total_ncore", 0)) != 0
        except (TypeError, ValueError):
            return True
    try:
        return int(getattr(options, "ecp_total_ncore", 0) or 0) != 0
    except (TypeError, ValueError):
        return True


# ---------- Library-name resolution ----------------------------------------
#
# libecpint ships a fixed set of XML ECP libraries under
# ``python/vibeqc/ecp_library/xml/``: ecp10mdf, ecp28mdf, ecp46mdf,
# ecp60mdf, ecp78mdf, lanl2dz. The basis name + ncore tell us which
# one to point at.

# Standard ncore-keyed mapping for the Stuttgart-Köln MDF series.
# Used when the basis name itself isn't a libecpint library
# (e.g. dhf-tzvp + Rb has ncore=28 -> ecp28mdf).
_NCORE_TO_MDF: dict[int, str] = {
    10: "ecp10mdf",
    28: "ecp28mdf",
    46: "ecp46mdf",
    60: "ecp60mdf",
    78: "ecp78mdf",
}

# Bases whose name maps directly to a libecpint XML library --
# ncore-irrelevant since the same XML covers every element in the
# basis. Add to this set as new ECP basis families ship XML.
_BASIS_TO_LIBRARY: dict[str, str] = {
    "lanl2dz":   "lanl2dz",
    "lanl2dzdp": "lanl2dz",
    "lanl2tz":   "lanl2dz",
    "lanl08":    "lanl2dz",
    "lanl08(d)": "lanl2dz",
    "lanl08(f)": "lanl2dz",
}


def library_for(basis_name: str, ncore: int) -> Optional[str]:
    """Pick the libecpint XML library name for ``(basis_name, ncore)``.

    Resolution order:

    1. If the basis name is one of the LANL family, use
       ``"lanl2dz"`` (every LANL element lives in that single XML).
    2. Otherwise consult :data:`_NCORE_TO_MDF` keyed on ``ncore``
       -- covers def2-TZVP/QZVP heavy elements, dhf-*, x2c-*, and
       any other Stuttgart-Köln MDF-derived basis.
    3. Return ``None`` for non-standard ncores (vDZP's
       per-element customs, ``ncore=2``, ``ncore=3``, etc.).

    The ``None`` return is the auto-populator's "no standard XML
    library matches" signal; molecular SCF wrappers use that signal to
    attach parsed inline primitive ECPs from the sidecar.
    """
    name = basis_name.lower()
    if name in _BASIS_TO_LIBRARY:
        return _BASIS_TO_LIBRARY[name]
    return _NCORE_TO_MDF.get(ncore)


# ---------- The one-call helper --------------------------------------------


def auto_ecp_centers(mol, basis_name: str,
                     library_name: Optional[str] = None) -> tuple:
    """Build ``(ecp_centers, library_name)`` from a Molecule + basis name.

    **Legacy XML-library helper.**  The molecular SCF wrappers attach the
    basis sidecar's own primitives inline (see
    :func:`attach_inline_ecp_options_from_basis_sidecar`); this helper only
    pairs a libecpint XML library by core count and is kept for scripts that
    still call it.  It cannot represent vDZP-style custom cores or a
    molecule spanning two core sizes, and it pairs a basis whose sidecar
    differs from the Stuttgart library with the wrong potential.

    For each atom in ``mol`` whose Z appears in
    ``<basis_name>.ecp``, emits an ``ECPCenter(Z, xyz)`` and resolves
    the right libecpint XML library. The two returned values can be
    assigned directly to ``opts.ecp_centers`` and ``opts.ecp_library``
    on any of the four molecular SCF Options classes.

    Parameters
    ----------
    mol : vibeqc.Molecule
        The molecule whose heavy atoms need ECP centres built.
    basis_name : str
        Basis-set name as you'd pass to ``vq.BasisSet(mol, name)``.
    library_name : Optional[str]
        Override the auto-resolved libecpint XML library. Useful when
        a basis ships a custom XML the user dropped into
        ``$VIBEQC_ECP_SHARE_DIR/xml/``.

    Returns
    -------
    (centers, library_name) : tuple[list[ECPCenter], str]
        Empty list and the input ``library_name`` (or ``""``) if the
        basis is all-electron; otherwise a populated list.

    Raises
    ------
    ValueError
        When the molecule's heavy atoms would need MORE THAN ONE
        libecpint XML library (mixed-row case -- common for dhf with
        a Rb-Cs span). Use inline primitive option fields when a
        single XML library cannot represent the molecule.
    NotImplementedError
        When the basis has at least one ECP atom whose ncore doesn't
        match any standard libecpint library (vDZP-style customs).
    """
    from . import ECPCenter  # local: avoid hard-import for doc builds

    sidecar = sidecar_path_for(basis_name)
    if sidecar is None:
        # All-electron basis. Nothing to build; honour any explicit
        # library_name the caller already provided.
        return [], (library_name or "")

    headers = parse_sidecar_path(sidecar)
    by_z = {h.Z: h for h in headers}
    if not by_z:
        return [], (library_name or "")

    # Walk mol's atoms; emit centres for every ECP-bearing match.
    # ``Molecule.atoms`` is a property returning a list, not a method.
    centres: list = []
    libraries_seen: set[str] = set()
    for atom in mol.atoms:
        h = by_z.get(int(atom.Z))
        if h is None:
            continue  # all-electron atom, skip
        ec = ECPCenter()
        ec.Z = h.Z
        ec.xyz = list(atom.xyz)
        centres.append(ec)
        if library_name is None:
            lib = library_for(basis_name, h.ncore)
            if lib is None:
                raise NotImplementedError(
                    f"Basis {basis_name!r} has a non-standard ECP for "
                    f"{h.symbol} (ncore={h.ncore}, lmax={h.lmax}) -- no "
                    "matching libecpint XML library is bundled. "
                    "auto_ecp_centers is the legacy XML-library helper; "
                    "molecular SCF wrappers consume this vDZP-style "
                    "inline-primitive sidecar automatically. To build "
                    "options by hand, use inline_ecp_data_for(...) and "
                    "assign the ecp_primitive_* option fields."
                )
            libraries_seen.add(lib)

    if library_name is None:
        if len(libraries_seen) == 0:
            # No mol atom matched a sidecar element -- basis carries
            # ECPs but the molecule doesn't need any. Fine; return
            # empty list + empty library.
            return [], ""
        if len(libraries_seen) > 1:
            raise ValueError(
                f"Basis {basis_name!r} would need {sorted(libraries_seen)} "
                "libecpint libraries to cover this molecule (mixed-row "
                "ECP), but vibe-qc's SCF drivers accept exactly one "
                "ecp_library per call through auto_ecp_centers. Split the "
                "molecule into single-library subsets, or use the inline "
                "primitive ecp_primitive_* option fields."
            )
        library_name = next(iter(libraries_seen))
    return centres, library_name


# ---------- Dict-based convenience (element symbol → ECP name) -----------------


def ecp_centers_from_dict(
    spec: dict[str, str],
    molecule,
    *,
    basis_name: Optional[str] = None,
) -> tuple[list, str]:
    """Build ``(ecp_centers, library_name)`` from a ``{symbol: ecp_name}`` dict.

    Convenience wrapper that mirrors the dict-based ECP specification
    common in other QC input formats::

        opts.ecp_centers, opts.ecp_library = ecp_centers_from_dict(
            {"I": "dhf"}, molecule, basis_name="dhf-tzvpp"
        )

    Each key is an element symbol (case-insensitive).  The value is a
    *label* that selects the ECP preset; currently the only recognised
    label is ``"dhf"`` (Stuttgart-Koeln MDF, resolved via *basis_name*
    through :func:`auto_ecp_centers`).

    Parameters
    ----------
    spec : dict[str, str]
        ``{symbol: ecp_label}`` mapping.
    molecule : Molecule
        The molecule whose atom positions anchor the ECP centres.
    basis_name : str or None
        Basis-set name passed to :func:`auto_ecp_centers` for
        library-name resolution.  Required when *spec* contains
        ``"dhf"`` labels.

    Returns
    -------
    (centers, library_name) : tuple[list[ECPCenter], str]
        Ready to assign to ``opts.ecp_centers`` and ``opts.ecp_library``.

    Raises
    ------
    ValueError
        When *spec* contains an unknown element symbol or an
        unrecognised ECP label.
    """
    from . import ECPCenter

    # Normalise spec keys to atomic numbers.
    _by_z: dict[int, str] = {}
    for sym, label in spec.items():
        z = _SYMBOL_TO_Z.get(_normalise_symbol(sym))
        if z is None or z == 0:
            raise ValueError(
                f"Unknown element symbol {sym!r} in ecp_centers dict"
            )
        _by_z[z] = label.lower()

    centres: list = []
    libraries_seen: set[str] = set()
    # Labels that require basis_name for sidecar-based library resolution.
    _LABEL_NEEDS_BASIS = frozenset({"dhf", "lanl2"})
    _known_labels = sorted(_LABEL_NEEDS_BASIS)
    for atom in molecule.atoms:
        z = int(atom.Z)
        label = _by_z.get(z)
        if label is None:
            continue
        ec = ECPCenter()
        ec.Z = z
        ec.xyz = list(atom.xyz)
        centres.append(ec)
        if label in _LABEL_NEEDS_BASIS:
            if basis_name is None:
                raise ValueError(
                    "ecp_centers_from_dict: basis_name is required when "
                    f"spec contains {label!r} label (atom {atom.symbol})"
                )
            # Delegate to the sidecar + library_for for XML resolution.
            # This also validates that the basis + ncore combination
            # maps to a known libecpint XML library.
            _h = _resolve_ecp_header_for(basis_name, z)
            lib = library_for(basis_name, _h.ncore)
            if lib is None:
                raise ValueError(
                    f"ecp_centers_from_dict: basis {basis_name!r} has no "
                    f"standard XML library for {atom.symbol} (ncore={_h.ncore})"
                )
            libraries_seen.add(lib)
        else:
            raise ValueError(
                f"ecp_centers_from_dict: unknown ECP label {label!r} "
                f"for atom {atom.symbol}. "
                f"Supported labels: {_known_labels}"
            )

    if libraries_seen:
        if len(libraries_seen) > 1:
            raise ValueError(
                "ecp_centers_from_dict: molecule requires multiple "
                f"libecpint libraries ({sorted(libraries_seen)}); "
                "the SCF drivers accept exactly one ecp_library per call"
            )
        library_name = next(iter(libraries_seen))
    else:
        library_name = ""

    return centres, library_name


def _resolve_ecp_header_for(basis_name: str, z: int) -> EcpHeader:
    """Return the :class:`EcpHeader` for atomic number *z* from the
    ``<basis_name>.ecp`` sidecar.

    Raises ``ValueError`` when the sidecar is missing or doesn't cover
    element *z*.
    """
    sidecar = sidecar_path_for(basis_name)
    if sidecar is None:
        raise ValueError(
            f"No ECP sidecar found for basis {basis_name!r}"
        )
    headers = parse_sidecar_path(sidecar)
    for h in headers:
        if h.Z == z:
            return h
    raise ValueError(
        f"Basis {basis_name!r} does not include an ECP for "
        f"atomic number {z}"
    )


# ---------- Safety validation (BUG 99) ---------------------------------------
#
# Basis sets whose *name* signals that they are designed for use with an
# effective core potential.  When a user specifies one of these bases but
# does not attach ECP centres, the runtime must refuse instead of silently
# running an all-electron calculation whose energy differs from the correct
# ECP value by thousands of Hartree.
#
# The patterns are matched case-insensitively against the normalised
# basis name (``basis_name.lower()``).  Add new families here when they
# ship their first ECP-bearing member.

import re as _re

_ECP_PAIRED_PATTERNS: list[_re.Pattern] = [
    # dhf-* family: dhf-SVP, dhf-TZVP, dhf-TZVPP, dhf-QZVP, dhf-QZVPP
    _re.compile(r"^dhf-"),
    # Stuttgart-Koeln MDF relativistic bases: x2c-TZVPall, x2c-TZVPall-s, …
    _re.compile(r"^x2c-"),
    # Pseudopotential-tagged correlation-consistent: cc-pV*Z-PP, aug-cc-pV*Z-PP,
    # cc-pwCV*Z-PP, aug-cc-pwCV*Z-PP
    _re.compile(r"-pp$"),
    # LANL effective-core-potential family: lanl2dz, lanl2dzdp, lanl2tz,
    # lanl08, lanl08(d), lanl08(f)
    _re.compile(r"^lanl"),
    # vDZP: variational DZP with per-element custom ECP (B, C, N, O, F)
    _re.compile(r"^vdzp$"),
    # Explicit ECP-labelled bases: ecp-*, *-ECP (catch-all for any future
    # naming convention that includes "ecp" in the name)
    _re.compile(r"ecp"),
]


def is_ecp_paired_basis(basis_name: str) -> bool:
    """Return ``True`` when *basis_name* belongs to a family designed for
    use with an effective core potential.

    **Legacy name heuristic.**  It is wrong in both directions (LANL2DZ and
    dhf-* are all-electron on light atoms; x2c-* is all-electron everywhere;
    pob-TZVP-rev2 and def2-m* are valence-only beyond Kr but carry no
    marker), so no runtime guard uses it any more.  Use
    :func:`vibeqc.basis_registry.ecp_requirements`, which decides per
    element from the sidecar and the registry.  Kept for callers that only
    want a coarse family label.

    Parameters
    ----------
    basis_name : str
        Basis-set name as resolved by ``BasisSet.name`` (case-normalised).
    """
    lower = basis_name.lower()
    return any(p.search(lower) for p in _ECP_PAIRED_PATTERNS)


def validate_ecp_required(options, molecule, basis_name: str) -> None:
    """Raise ``ValueError`` when *basis_name* needs an ECP on an atom of
    *molecule* but *options* carries no ECP.

    This is the BUG 99 guard, decided **per element** through
    :mod:`vibeqc.basis_registry`: a basis needs an ECP on an atom when its
    sidecar replaces core electrons for that element, or when the family is
    valence-only beyond a threshold (def2 and the 3c bases beyond Kr, the
    cc-pVnZ-PP sets, pob-TZVP-rev2) and the atom lies beyond it.  A light
    molecule in LANL2DZ or dhf-TZVP is all-electron and passes; an
    all-electron relativistic basis such as x2c-TZVPall passes on every
    element.  Running a valence-only block all-electron silently produces an
    energy that is off by hundreds or thousands of Hartree, so the guard
    refuses instead.

    Parameters
    ----------
    options : object or None
        One of ``RHFOptions`` / ``UHFOptions`` / ``RKSOptions`` /
        ``UKSOptions``.  ``None`` is silently ignored.
    molecule : Molecule
        The molecule being computed.
    basis_name : str
        Basis-set name as resolved by ``BasisSet.name``.

    Raises
    ------
    ValueError
        When an atom needs an ECP but neither ``ecp_centers`` nor
        ``ecp_primitive_blocks`` is set on *options*.
    """
    if options is None:
        return

    # Already populated -- nothing to validate.
    if getattr(options, "ecp_centers", None):
        return
    if getattr(options, "ecp_primitive_blocks", None):
        return

    from .basis_registry import ecp_requirements

    needed = ecp_requirements(basis_name, (int(atom.Z) for atom in molecule.atoms))
    if not needed:
        return

    symbols = sorted({req.symbol for req in needed})
    without_data = sorted({req.symbol for req in needed if not req.has_data})
    if without_data:
        _cause = (
            f"vibe-qc ships no ECP data for basis={basis_name!r} on "
            f"{without_data}, and the bundled blocks for these elements are "
            "valence-only, so the calculation cannot run in this basis."
        )
    else:
        _cause = (
            "The bundled sidecar carries the ECP for these atoms; the "
            "molecular SCF wrappers attach it automatically, so reaching "
            "this guard means the options bypassed that attachment "
            "(call vibeqc.ecp_metadata.attach_inline_ecp_options_from_"
            "basis_sidecar, or set the ecp_* fields by hand)."
        )
    _msg = (
        f"basis={basis_name!r} replaces core electrons with an effective core "
        f"potential (ECP) on {symbols}, but no ECP centers were configured. "
        f"Atoms that need an ECP: {symbols}. {_cause}"
    )

    # Emit to the structured log (NDJSON) before raising, so the
    # mismatch is permanently recorded even when the caller catches
    # the error.  The import is lazy: ecp_metadata has no output
    # dependency at module level.
    try:
        from vibeqc.output.formats.structured_log import emit as _slog_emit

        _slog_emit(
            "ecp_mismatch_warning",
            basis=basis_name,
            ecp_paired=True,
            ecp_centers_configured=False,
            elements=symbols,
            message=_msg,
        )
    except Exception:
        pass  # structured log is best-effort; never crash on it

    raise ValueError(
        _msg + "\n"
        "Either configure ecp_centers / ecp_primitive_* via RHFOptions / "
        "RKSOptions / UHFOptions / UKSOptions, or choose an all-electron "
        "basis set such as x2c-TZVPall, ANO-RCC or (up to Kr) def2-TZVPP."
    )


# ---------------------------------------------------------------------------
# ECP centres follow their atoms across geometries (#576)
# ---------------------------------------------------------------------------
#
# Both ECP routes pin the potential to ABSOLUTE coordinates on the SCF
# options: ``ecp_centers[i].xyz`` (XML library) and
# ``ecp_primitive_centers[i]`` (inline primitives). The C++ side identifies
# the atom an ECP belongs to purely by position -- ``ecp_effective_charges``
# and ``compute_ecp_one_electron`` in cpp/src/ecp.cpp accept a centre as
# sitting on atom A when ``|R_A - R_centre|^2 < 1e-12`` bohr^2 and the Z
# match, and libecpint's ``ECPIntegrator::init`` assigns ECPs to atom ids
# within 1e-4 bohr (libecpint api.cpp). Move a nucleus by a finite-difference
# step without moving its centre and the atom silently reverts to bare Z while
# V_ECP keeps acting: [ZnH]+/6-31g/ecp10mdf with Zn displaced 0.005 bohr and
# the centre left behind diverges to E = -7.4e10 Ha (#576). Every driver that
# evaluates the SCF at a geometry other than the one the centres were built
# for has to reposition them first; these helpers are that one place.

#: Squared distance (bohr^2) within which a centre counts as sitting on an
#: atom. Same threshold as cpp/src/ecp.cpp (``d2 < 1e-12``): a centre the C++
#: side would not attach to the atom is not one we can follow either.
_ECP_CENTRE_MATCH_D2_BOHR2 = 1e-12


def _match_centre_to_atom(
    xyz, molecule, *, z: Optional[int], what: str
) -> int:
    """Index of the atom of ``molecule`` that ``xyz`` (bohr) sits on.

    ``z`` restricts the match to atoms of that atomic number (XML centres
    carry a ``Z``; inline centres do not). Raises ``ValueError`` naming the
    centre and the nearest atom when nothing is within the C++ threshold.
    """
    best_i, best_d2 = -1, float("inf")
    for i, atom in enumerate(molecule.atoms):
        if z is not None and int(atom.Z) != int(z):
            continue
        d2 = sum((float(atom.xyz[k]) - float(xyz[k])) ** 2 for k in range(3))
        if d2 < best_d2:
            best_i, best_d2 = i, d2
    if best_i < 0 or best_d2 >= _ECP_CENTRE_MATCH_D2_BOHR2:
        nearest = (
            f"; nearest candidate is atom {best_i} at "
            f"{best_d2 ** 0.5:.3e} bohr" if best_i >= 0 else ""
        )
        raise ValueError(
            f"{what} at xyz={[float(v) for v in xyz]} bohr"
            + (f" (Z={int(z)})" if z is not None else "")
            + " coincides with no atom of the molecule (tolerance "
            f"{_ECP_CENTRE_MATCH_D2_BOHR2 ** 0.5:.0e} bohr, the same the "
            "SCF uses to attach an ECP to a nucleus)" + nearest + ". The ECP "
            "options describe a different geometry than the molecule; "
            "rebuild them for this geometry (auto_ecp_centers / "
            "attach_inline_ecp_options_from_basis_sidecar) before running."
        )
    return best_i


def ecp_centre_atom_indices(options, molecule) -> tuple[list[int], list[int]]:
    """Map every ECP centre on ``options`` onto the atom of ``molecule`` it
    sits on.

    Returns ``(xml_indices, inline_indices)``: one atom index per entry of
    ``options.ecp_centers`` and one per entry of
    ``options.ecp_primitive_centers`` (either list empty when that route is
    not in use). Raises ``ValueError`` when a centre coincides with no atom
    within the threshold the C++ SCF itself uses (1e-6 bohr), because such an
    SCF is already inconsistent (bare Z on the atom, V_ECP still acting) and
    no derivative built on it can be trusted.
    """
    xml_indices: list[int] = []
    for k, centre in enumerate(getattr(options, "ecp_centers", None) or []):
        xml_indices.append(
            _match_centre_to_atom(
                centre.xyz, molecule, z=int(centre.Z),
                what=f"ECP centre ecp_centers[{k}]",
            )
        )
    inline_indices: list[int] = []
    for k, xyz in enumerate(
        getattr(options, "ecp_primitive_centers", None) or []
    ):
        inline_indices.append(
            _match_centre_to_atom(
                xyz, molecule, z=None,
                what=f"inline ECP centre ecp_primitive_centers[{k}]",
            )
        )
    return xml_indices, inline_indices


def options_carry_ecp(options) -> bool:
    """``True`` when ``options`` carries ECP centres on either route."""
    if options is None:
        return False
    return bool(getattr(options, "ecp_centers", None)) or bool(
        getattr(options, "ecp_primitive_centers", None)
    )


def effective_nuclear_charges(molecule, options) -> "np.ndarray":
    """Per-atom nuclear charges the valence-only Hamiltonian actually sees,
    in ``molecule.atoms`` order: ``Z_A - n_core,A`` on every atom whose core
    an ECP replaces, the bare ``Z_A`` elsewhere, and bare ``Z`` throughout
    when ``options`` is ``None`` or carries no ECP.

    This is the charge that belongs in every electrostatic property of an
    ECP calculation, not only in the nuclear repulsion the SCF already uses:
    Dolg & Cao, Chem. Rev. 112, 403 (2012), doi:10.1021/cr2001383, Sec. 5
    "Valence-only model Hamiltonian", Eqs. 54-55 -- the cores enter as point
    charges ``Q_lambda`` with ``n_v = n - sum_lambda (Z_lambda - Q_lambda)``
    valence electrons. A density that integrates to ``n_v`` paired with the
    bare ``Z`` in the nuclear dipole term ``sum_A Z_A (R_A - O)`` leaves a
    spurious ``n_core,A (R_A - O)`` per ECP atom, so a neutral molecule's
    dipole becomes origin-dependent by exactly ``-n_core`` per bohr of origin
    shift, and Mulliken/Loewdin/Hirshfeld charges come out ``+n_core,A`` too
    positive (GitLab #642: H2S/LANL2DZ RHF at its centre of mass 0.148 D
    with the wrong sign, -10.000 au per bohr; 1.99496 D and origin-free
    with ``Z_eff = [6, 1, 1]``).

    Both routes the SCF drivers use are honoured, with the same precedence
    as :func:`vibeqc.gradient_options.gradient_options_from_scf`: inline
    primitive blocks first (``options.ecp_effective_charges`` is already
    the per-atom vector :func:`inline_ecp_data_for` builds, bare ``Z`` on
    atoms without a block), then the libecpint XML-library route
    (``vibeqc.ecp_effective_charges`` resolves each centre's ``n_core``
    from the library). Returns a float ``ndarray`` of shape ``(n_atoms,)``.
    """
    import numpy as _np

    bare = _np.asarray([float(a.Z) for a in molecule.atoms], dtype=_np.float64)
    if options is None:
        return bare
    if getattr(options, "ecp_primitive_blocks", None):
        inline = [float(q) for q in (getattr(options, "ecp_effective_charges", None) or [])]
        if len(inline) == len(bare):
            return _np.asarray(inline, dtype=_np.float64)
        # A malformed inline vector cannot be trusted; refuse rather than
        # silently fall back to the bare-Z answer this helper exists to fix.
        raise ValueError(
            "effective_nuclear_charges: options.ecp_effective_charges has "
            f"{len(inline)} entries for {len(bare)} atoms; the inline ECP "
            "route attaches one effective charge per atom "
            "(vibeqc.ecp_metadata.inline_ecp_data_for)."
        )
    centers = getattr(options, "ecp_centers", None)
    if centers:
        from . import ecp_effective_charges as _xml_effective_charges

        library = str(getattr(options, "ecp_library", "") or "")
        q = _xml_effective_charges(molecule, list(centers), library)
        return _np.asarray([float(x) for x in q], dtype=_np.float64)
    return bare


def _xml_centers_from(source) -> list:
    """XML-route centres from a result (``ecp_xml_centers``) or options
    (``ecp_centers``), as native ``ECPCenter`` objects."""
    from . import ECPCenter

    centers = getattr(source, "ecp_xml_centers", None)
    if centers is None or len(centers) == 0:
        centers = getattr(source, "ecp_centers", None)
    out = []
    for c in list(centers or []):
        if isinstance(c, ECPCenter):
            out.append(c)
            continue
        native = ECPCenter()
        native.Z = int(c.Z)
        native.xyz = [float(v) for v in c.xyz]
        out.append(native)
    return out


def effective_nuclear_charges_from(molecule, source) -> "np.ndarray":
    """Per-atom ``Z_eff`` for the Hamiltonian ``source`` describes.

    ``source`` may be a molecular SCF *result* (``ecp_xml_centers`` /
    ``ecp_xml_library`` or ``ecp_primitive_*`` / ``ecp_effective_charges``,
    as stamped by the native drivers and by CPCM) or an options struct
    (:func:`effective_nuclear_charges`).  Both ECP routes are honoured with
    the same precedence as everywhere else: inline primitives first, then
    the XML library.  ``None`` and an all-electron source give bare ``Z``.
    """
    import numpy as _np

    bare = _np.asarray([float(a.Z) for a in molecule.atoms], dtype=_np.float64)
    if source is None:
        return bare
    blocks = getattr(source, "ecp_primitive_blocks", None)
    if blocks is not None and len(blocks) > 0:
        inline = [float(q) for q in (getattr(source, "ecp_effective_charges", None) or [])]
        if len(inline) != len(bare):
            raise ValueError(
                "effective_nuclear_charges_from: the inline ECP route carries "
                f"{len(inline)} effective charges for {len(bare)} atoms"
            )
        return _np.asarray(inline, dtype=_np.float64)
    centers = _xml_centers_from(source)
    if centers:
        from . import ecp_effective_charges as _xml_effective_charges

        library = str(
            getattr(source, "ecp_xml_library", "")
            or getattr(source, "ecp_library", "")
            or ""
        )
        q = _xml_effective_charges(molecule, centers, library)
        return _np.asarray([float(x) for x in q], dtype=_np.float64)
    return bare


def ecp_core_electrons_per_atom(molecule, source) -> list:
    """Electrons an ECP removed on each atom of ``molecule`` under ``source``
    (an SCF result or options), in ``molecule.atoms`` order; zeros for an
    all-electron source.  This is what an element-based frozen-core count
    must subtract before freezing anything (see
    :func:`vibeqc.correlation_conventions.published_frozen_core_orbital_count`).
    """
    q = effective_nuclear_charges_from(molecule, source)
    return [int(round(float(a.Z) - float(qa))) for a, qa in zip(molecule.atoms, q)]


def one_electron_hamiltonian(molecule, basis, source) -> tuple:
    """Return ``(hcore, e_nuc, n_valence, z_eff)`` for the Hamiltonian that
    ``source`` (an SCF result or an options struct) describes.

    All-electron: ``T + V_ne(Z)``, the bare nuclear repulsion, and the
    physical electron count.  With an ECP on either route: ``T +
    V_ne(Z_eff) + V_ECP``, the ``Z_eff`` point-charge repulsion, and the
    valence count.  This is the same construction the native RHF/UHF/RKS/UKS
    drivers use, so a Python driver (ROHF/ROKS, the determinant solvers)
    built on it solves the same Hamiltonian the native reference did.
    """
    import numpy as _np

    from ._vibeqc_core import (
        compute_ecp_matrix_from_primitives,
        compute_kinetic,
        compute_nuclear,
        compute_nuclear_with_charges,
    )

    atoms = list(molecule.atoms)
    T = _np.asarray(compute_kinetic(basis), dtype=float)
    z_eff = effective_nuclear_charges_from(molecule, source)
    blocks = list(getattr(source, "ecp_primitive_blocks", None) or []) if source is not None else []
    centers = _xml_centers_from(source) if (source is not None and not blocks) else []
    if not blocks and not centers:
        hcore = T + _np.asarray(compute_nuclear(basis, molecule), dtype=float)
        return hcore, float(molecule.nuclear_repulsion()), int(molecule.n_electrons()), z_eff

    positions = [list(a.xyz) for a in atoms]
    V = _np.asarray(compute_nuclear_with_charges(basis, positions, z_eff.tolist()), dtype=float)
    if blocks:
        flat = [float(v) for c in getattr(source, "ecp_primitive_centers") for v in c]
        V_ecp = _np.asarray(compute_ecp_matrix_from_primitives(basis, flat, blocks), dtype=float)
    else:
        from . import compute_ecp_matrix

        library = str(
            getattr(source, "ecp_xml_library", "")
            or getattr(source, "ecp_library", "")
            or "ecp10mdf"
        )
        V_ecp = _np.asarray(compute_ecp_matrix(basis, centers, library), dtype=float)
    e_nuc = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = _np.asarray(atoms[i].xyz, dtype=float) - _np.asarray(atoms[j].xyz, dtype=float)
            e_nuc += float(z_eff[i]) * float(z_eff[j]) / float(_np.linalg.norm(d))
    n_core = int(round(sum(float(a.Z) for a in atoms) - float(z_eff.sum())))
    return T + V + V_ecp, e_nuc, int(molecule.n_electrons()) - n_core, z_eff


def reposition_ecp_centres(options, molecule, atom_indices) -> None:
    """Move the ECP centres on ``options`` onto the atoms of ``molecule``.

    ``atom_indices`` is the ``(xml_indices, inline_indices)`` pair from
    :func:`ecp_centre_atom_indices`, established at a geometry where the
    centres coincided with their atoms; ``molecule`` must have the same atoms
    in the same order. The change is permanent (in place, by value: the
    ``ECPCenter`` objects are rebuilt and the lists reassigned, because
    pybind hands list elements out by reference into the underlying
    vector). Drivers that keep one options object alive across geometries
    -- the ASE calculator, the runner's optimiser hand-back (#643) -- use
    this; a driver that only visits a geometry transiently uses the
    restoring :func:`ecp_centres_follow_atoms` instead.
    """
    xml_indices, inline_indices = atom_indices
    if not xml_indices and not inline_indices:
        return
    n_atoms = len(molecule.atoms)
    for i in set(xml_indices) | set(inline_indices):
        if i < 0 or i >= n_atoms:
            raise ValueError(
                f"reposition_ecp_centres: atom index {i} out of range for a "
                f"molecule of {n_atoms} atoms"
            )
    if xml_indices:
        from ._vibeqc_core import ECPCenter

        current = list(options.ecp_centers)
        if len(current) != len(xml_indices):
            raise ValueError(
                "reposition_ecp_centres: options carry "
                f"{len(current)} XML ECP centres but the mapping has "
                f"{len(xml_indices)} entries"
            )
        moved = []
        for centre, i in zip(current, xml_indices):
            atom = molecule.atoms[i]
            if int(atom.Z) != int(centre.Z):
                raise ValueError(
                    f"reposition_ecp_centres: ECP centre Z={int(centre.Z)} "
                    f"mapped onto atom {i} with Z={int(atom.Z)}; the atom "
                    "order changed since the mapping was made"
                )
            ec = ECPCenter()
            ec.Z = int(centre.Z)
            ec.xyz = [float(v) for v in atom.xyz]
            moved.append(ec)
        options.ecp_centers = moved
    if inline_indices:
        if len(list(options.ecp_primitive_centers)) != len(inline_indices):
            raise ValueError(
                "reposition_ecp_centres: options carry "
                f"{len(list(options.ecp_primitive_centers))} inline ECP "
                f"centres but the mapping has {len(inline_indices)} entries"
            )
        options.ecp_primitive_centers = [
            [float(v) for v in molecule.atoms[i].xyz] for i in inline_indices
        ]


@contextmanager
def ecp_centres_follow_atoms(
    options, reference_molecule, displaced_molecule, gradient_options=None
) -> Iterator[None]:
    """Within the block, the ECP centres on ``options`` sit on the atoms of
    ``displaced_molecule`` instead of those of ``reference_molecule``.

    Each centre is identified with the atom of ``reference_molecule`` it
    coincides with (:func:`ecp_centre_atom_indices`) and moved to that
    atom's position in ``displaced_molecule``; the two molecules must have
    the same atoms in the same order. When ``gradient_options`` is given its
    ECP fields are synchronised to the repositioned SCF options for the
    duration of the block as well (:func:`vibeqc.gradient_options.copy_ecp_fields`),
    so the gradient differentiates the Hamiltonian the SCF at
    ``displaced_molecule`` solves. Every touched field is restored on exit,
    also on an exception. A no-op when ``options`` carries no ECP centre or
    ``reference_molecule`` is ``None``.

    ``vibeqc.hessian.compute_hessian_fd`` wraps every displaced SCF +
    gradient evaluation in this (#576).
    """
    if options is None or reference_molecule is None:
        yield
        return
    xml_indices, inline_indices = ecp_centre_atom_indices(
        options, reference_molecule
    )
    if not xml_indices and not inline_indices:
        yield
        return
    n_ref = len(reference_molecule.atoms)
    n_disp = len(displaced_molecule.atoms)
    if n_ref != n_disp:
        raise ValueError(
            "ecp_centres_follow_atoms: reference and displaced molecules "
            f"have different atom counts ({n_ref} vs {n_disp})"
        )
    # pybind11 hands out the list ELEMENTS by reference into the underlying
    # std::vector (mutating ``options.ecp_centers[0].xyz`` changes the
    # options; a saved ``list(options.ecp_centers)`` reads the *new* values
    # after the vector is reassigned), so snapshots hold plain Python data
    # and the restore rebuilds the objects.
    saved_scf = _snapshot_ecp_fields(options)
    saved_grad = (
        _snapshot_ecp_fields(gradient_options)
        if gradient_options is not None else None
    )
    try:
        reposition_ecp_centres(
            options, displaced_molecule, (xml_indices, inline_indices)
        )
        if gradient_options is not None:
            from .gradient_options import copy_ecp_fields

            copy_ecp_fields(gradient_options, options)
        yield
    finally:
        _restore_ecp_fields(options, saved_scf)
        if saved_grad is not None:
            _restore_ecp_fields(gradient_options, saved_grad)


def _snapshot_ecp_fields(obj) -> dict:
    """Plain-Python copy of the six ECP fields of an options struct (values,
    not pybind references), for :func:`_restore_ecp_fields`."""
    return {
        "ecp_centers": [
            (int(c.Z), [float(v) for v in c.xyz])
            for c in (getattr(obj, "ecp_centers", None) or [])
        ],
        "ecp_library": str(getattr(obj, "ecp_library", "") or ""),
        "ecp_primitive_blocks": [
            {
                "n_primitive": int(b.n_primitive),
                "exponents": [float(v) for v in b.exponents],
                "coefficients": [float(v) for v in b.coefficients],
                "ams": [int(v) for v in b.ams],
                "ns": [int(v) for v in b.ns],
            }
            for b in (getattr(obj, "ecp_primitive_blocks", None) or [])
        ],
        "ecp_primitive_centers": [
            [float(v) for v in c]
            for c in (getattr(obj, "ecp_primitive_centers", None) or [])
        ],
        "ecp_effective_charges": [
            float(q) for q in (getattr(obj, "ecp_effective_charges", None) or [])
        ],
        "ecp_total_ncore": int(getattr(obj, "ecp_total_ncore", 0) or 0),
    }


def _restore_ecp_fields(obj, snapshot: dict) -> None:
    """Inverse of :func:`_snapshot_ecp_fields`."""
    from ._vibeqc_core import ECPCenter, ECPPrimitiveBlock

    centres = []
    for z, xyz in snapshot["ecp_centers"]:
        ec = ECPCenter()
        ec.Z = z
        ec.xyz = list(xyz)
        centres.append(ec)
    obj.ecp_centers = centres
    obj.ecp_library = snapshot["ecp_library"]
    blocks = []
    for rec in snapshot["ecp_primitive_blocks"]:
        b = ECPPrimitiveBlock()
        b.n_primitive = rec["n_primitive"]
        b.exponents = list(rec["exponents"])
        b.coefficients = list(rec["coefficients"])
        b.ams = list(rec["ams"])
        b.ns = list(rec["ns"])
        blocks.append(b)
    obj.ecp_primitive_blocks = blocks
    obj.ecp_primitive_centers = [list(c) for c in snapshot["ecp_primitive_centers"]]
    obj.ecp_effective_charges = list(snapshot["ecp_effective_charges"])
    obj.ecp_total_ncore = snapshot["ecp_total_ncore"]
