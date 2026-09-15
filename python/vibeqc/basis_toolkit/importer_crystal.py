"""Bridge: CRYSTAL-format data <-> canonical BasisSetData.

Connects :mod:`vibeqc.basis_crystal` (the legacy CRYSTAL parser) to the
canonical model.  CRYSTAL's per-element files use integer LAT codes and
the ``200+Z`` ECP header convention -- this module translates those into
the typed ``BasisSetData`` / ``ECPEntry`` model.
"""

from __future__ import annotations

from pathlib import Path

from vibeqc.basis_crystal import (
    CrystalAtomBasis,
    parse_crystal_atom_basis,
)

from .model import (
    BasisRole,
    BasisSetData,
    ECPEntry,
    ECPPotential,
    ECPTerm,
    ElementBasis,
    HarmonicType,
    Primitive,
    Provenance,
    Shell,
)

__all__ = ["from_crystal_atom", "from_crystal_file"]


def from_crystal_atom(
    atom: CrystalAtomBasis,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Convert a single parsed CRYSTAL atom basis into :class:`BasisSetData`.

    Parameters
    ----------
    atom:
        A single :class:`CrystalAtomBasis` from :func:`parse_crystal_atom_basis`.
    name:
        Basis set name.  Defaults to ``"<crystal-import-Z{N}>"``.
    role, family, harmonic_type:
        Metadata hints.

    Returns
    -------
    BasisSetData
        Canonical basis with one element entry.
    """
    sym = atom.element_symbol
    shells: list[Shell] = []

    for cs in atom.shells:
        # Map CRYSTAL shell type to angular_momentum.
        am_map = {"S": [0], "SP": [0, 1], "P": [1], "D": [2], "F": [3], "G": [4]}
        am = am_map.get(cs.shell_type)
        if am is None:
            raise ValueError(f"Unknown CRYSTAL shell type: {cs.shell_type!r}")

        # Apply scale factor to exponents.
        scaled_exponents = [e * cs.scale_factor for e in cs.exponents]

        if am == [0, 1]:
            # SP shell: N s-coeffs then N p-coeffs, exponents repeated.
            n = len(scaled_exponents)
            prims: list[Primitive] = []
            for i in range(n):
                prims.append(
                    Primitive(
                        exponent=scaled_exponents[i], coefficient=cs.coefficients[i]
                    )
                )
            for i in range(n):
                prims.append(
                    Primitive(
                        exponent=scaled_exponents[i],
                        coefficient=cs.coefficients_p[i]
                        if i < len(cs.coefficients_p)
                        else 0.0,
                    )
                )
            shells.append(
                Shell(
                    angular_momentum=am,
                    harmonic_type=HarmonicType(harmonic_type),
                    primitives=prims,
                )
            )
        else:
            prims = [
                Primitive(exponent=e, coefficient=c)
                for e, c in zip(scaled_exponents, cs.coefficients)
            ]
            shells.append(
                Shell(
                    angular_momentum=am,
                    harmonic_type=HarmonicType(harmonic_type),
                    primitives=prims,
                )
            )

    # Build ECP if present.
    ecps: dict[str, ECPEntry] | None = None
    if atom.has_ecp and atom.ecp is not None:
        ecp = atom.ecp
        potentials: list[ECPPotential] = []
        current_l: int | None = None
        current_terms: list[ECPTerm] = []

        for term in ecp.terms:
            # Determine L for this term.
            if term.ell == "local":
                continue  # local potential not stored in OVF-Basis ECP model
            l_val = int(term.ell)
            if l_val != current_l:
                if current_l is not None and current_terms:
                    potentials.append(
                        ECPPotential(angular_momentum=current_l, terms=current_terms)
                    )
                current_l = l_val
                current_terms = []
            current_terms.append(
                ECPTerm(
                    coefficient=term.coefficient,
                    exponent=term.alpha,
                    power=term.n_pow,
                )
            )
        if current_l is not None and current_terms:
            potentials.append(
                ECPPotential(angular_momentum=current_l, terms=current_terms)
            )

        n_core = atom.Z - int(ecp.znuc)
        ecps = {
            sym: ECPEntry(
                element=sym,
                n_core_electrons=n_core,
                angular_momentum_max=max(
                    (p.angular_momentum for p in potentials), default=0
                ),
                potentials=potentials,
            )
        }

    return BasisSetData(
        schema_version="1.0.0",
        name=name or f"<crystal-import-Z{atom.Z}>",
        description=f"Imported from CRYSTAL: Z={atom.Z}",
        role=BasisRole(role),
        basis_family=family,
        elements={sym: ElementBasis(element=sym, shells=shells)},
        ecps=ecps,
        provenance=Provenance(origin="crystal"),
    )


def from_crystal_file(
    path: str | Path,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Read a CRYSTAL per-element file and convert to :class:`BasisSetData`."""
    text = Path(path).read_text(encoding="utf-8")
    atom = parse_crystal_atom_basis(text, source=str(path))
    return from_crystal_atom(
        atom, name=name, role=role, family=family, harmonic_type=harmonic_type
    )
