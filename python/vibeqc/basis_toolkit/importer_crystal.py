"""Bridge: CRYSTAL-format data <-> canonical BasisSetData.

Connects :mod:`vibeqc.basis_crystal` (the legacy CRYSTAL parser) to the
canonical model.  CRYSTAL's per-element files use integer LAT codes and
the ``200+Z`` ECP header convention -- this module translates those into
the typed ``BasisSetData`` / ``ECPEntry`` model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .model import (
    SCHEMA_VERSION,
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

if TYPE_CHECKING:
    from vibeqc.basis_crystal import CrystalAtomBasis

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
    # Imported here: vibeqc.basis_crystal imports this package at load time.
    from vibeqc.basis_crystal import scaled_exponents as crystal_scaled_exponents

    sym = atom.element_symbol
    shells: list[Shell] = []

    for cs in atom.shells:
        # Map CRYSTAL shell type to angular_momentum.
        am_map = {"S": [0], "SP": [0, 1], "P": [1], "D": [2], "F": [3], "G": [4]}
        am = am_map.get(cs.shell_type)
        if am is None:
            raise ValueError(f"Unknown CRYSTAL shell type: {cs.shell_type!r}")

        # CRYSTAL's SCAL rule (exponent * SCAL**2, 0 and 1 as written). A
        # parsed shell already carries it in its exponents (scale_factor
        # 1.0); a hand-built one may not.
        scaled_exponents = crystal_scaled_exponents(cs)

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

    # Build ECP if present. The conversion goes through the same bridge
    # the periodic driver and the sidecar writer use
    # (``basis_crystal.crystal_ecp_to_libecpint_arrays``): CRYSTAL's NKL is
    # the literal power of r and libecpint's Gaussian-format n is NKL + 2,
    # and a record without local terms needs the zero local placeholder so
    # the highest projector is not promoted to the local potential (#207).
    # A previous version copied NKL unconverted and dropped the local block.
    ecps: dict[str, ECPEntry] | None = None
    if atom.has_ecp and atom.ecp is not None:
        from vibeqc.basis_crystal import crystal_ecp_to_libecpint_arrays

        arrays = crystal_ecp_to_libecpint_arrays(atom.ecp)
        by_l: dict[int, list[ECPTerm]] = {}
        for exponent, coefficient, am, n in zip(
            arrays.exponents, arrays.coefficients, arrays.ams, arrays.ns
        ):
            by_l.setdefault(int(am), []).append(
                ECPTerm(coefficient=float(coefficient), exponent=float(exponent), power=int(n))
            )
        lmax = int(arrays.local_ell)
        potentials = [ECPPotential(angular_momentum=lmax, terms=by_l[lmax])]
        potentials.extend(
            ECPPotential(angular_momentum=ell, terms=by_l[ell])
            for ell in range(lmax)
            if ell in by_l
        )
        n_core = atom.Z - round(float(atom.ecp.znuc))
        ecps = {
            sym: ECPEntry(
                element=sym,
                n_core_electrons=n_core,
                angular_momentum_max=lmax,
                potentials=potentials,
            )
        }

    return BasisSetData(
        schema_version=SCHEMA_VERSION,
        name=name or f"<crystal-import-Z{atom.Z}>",
        description=f"Imported from CRYSTAL: Z={atom.Z}",
        role=BasisRole(role),
        basis_family=family,
        elements={sym: ElementBasis(element=sym, shells=shells, ecp_id=sym if ecps else None)},
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
    from vibeqc.basis_crystal import parse_crystal_atom_basis

    text = Path(path).read_text(encoding="utf-8")
    atom = parse_crystal_atom_basis(text, source=str(path))
    return from_crystal_atom(
        atom, name=name, role=role, family=family, harmonic_type=harmonic_type
    )
