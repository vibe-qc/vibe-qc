"""Direct integration: :class:`BasisSetData` <-> :class:`vibeqc._vibeqc_core.BasisSet`.

This bridge lets the basis toolkit feed the C++ integral engine directly,
without writing/reading a temporary ``.g94`` file.  It converts the typed
canonical model into the ``ShellInfo`` list that the pybind11-bound
``BasisSet`` constructor accepts.

The main entry point is :func:`to_libint_basis`, which expands SP shells
into separate S and P libint shells and maps element symbols to atom
indices in the molecule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model import BasisRole, BasisSetData

if TYPE_CHECKING:
    from vibeqc._vibeqc_core import BasisSet as _CppBasisSet
    from vibeqc._vibeqc_core import Molecule as _CppMolecule
    from vibeqc._vibeqc_core import ShellInfo

__all__ = ["to_libint_basis", "from_libint_basis"]


def to_libint_basis(
    data: BasisSetData,
    molecule: "_CppMolecule",
    *,
    coefficients_pre_normalized: bool = False,
) -> "_CppBasisSet":
    """Build a C++ :class:`BasisSet` from :class:`BasisSetData`.

    SP shells in the canonical model (``angular_momentum == [0, 1]``) are
    expanded into two separate libint shells -- one S, one P -- sharing the
    same exponents.  This matches libint's internal representation.

    Parameters
    ----------
    data:
        The canonical basis-set data to convert.
    molecule:
        A vibe-qc :class:`Molecule` whose atoms match the elements in
        *data*.  Shell origins are set to the corresponding atom positions
        so libint can build the shell-to-atom map.
    coefficients_pre_normalized:
        Passed through to the C++ ``BasisSet`` constructor.  The default
        ``False`` is correct for G94-convention coefficients (the default
        in the canonical model).

    Returns
    -------
    vibeqc._vibeqc_core.BasisSet
        A C++ basis set ready for integral computation.
    """
    from vibeqc._vibeqc_core import BasisSet as _CppBasisSet
    from vibeqc._vibeqc_core import ShellInfo as _CppShellInfo

    # Build element-symbol -> atom-index map from the molecule.
    shell_infos: list[_CppShellInfo] = []
    # Emit shells in atom-index order so the libint basis matches
    # the molecule's atom ordering.
    for atom_idx in range(len(molecule.atoms)):
        atom = molecule.atoms[atom_idx]
        sym = _symbol_for_z(atom.Z)
        eb = data.elements.get(sym)
        if eb is None:
            continue  # no basis data for this element
        origin = atom.xyz
        for shell in eb.shells:
            am = shell.angular_momentum
            if am == [0, 1]:
                # SP shell: emit separate S and P libint shells.
                n_prim = len(shell.primitives) // 2
                exponents = [shell.primitives[j].exponent for j in range(n_prim)]
                s_coeffs = [shell.primitives[j].coefficient for j in range(n_prim)]
                p_coeffs = [
                    shell.primitives[j + n_prim].coefficient for j in range(n_prim)
                ]

                shell_infos.append(
                    _CppShellInfo(
                        atom_index=atom_idx,
                        l=0,
                        pure=(shell.harmonic_type.value == "spherical"),
                        exponents=exponents,
                        coefficients=s_coeffs,
                        origin=origin,
                    )
                )
                shell_infos.append(
                    _CppShellInfo(
                        atom_index=atom_idx,
                        l=1,
                        pure=(shell.harmonic_type.value == "spherical"),
                        exponents=exponents,
                        coefficients=p_coeffs,
                        origin=origin,
                    )
                )
            else:
                # Pure shell.
                exponents = [p.exponent for p in shell.primitives]
                coefficients = [p.coefficient for p in shell.primitives]
                shell_infos.append(
                    _CppShellInfo(
                        atom_index=atom_idx,
                        l=am[0],
                        pure=(shell.harmonic_type.value == "spherical"),
                        exponents=exponents,
                        coefficients=coefficients,
                        origin=origin,
                    )
                )

    return _CppBasisSet(
        molecule,
        shell_infos,
        data.name,
        coefficients_pre_normalized,
    )


def from_libint_basis(
    basis: "_CppBasisSet",
    molecule: "_CppMolecule | None" = None,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
) -> BasisSetData:
    """Convert a C++ :class:`BasisSet` back into :class:`BasisSetData`.

    This is a *read* path: extract per-shell exponents and coefficients
    from a live libint basis set and wrap them in the canonical model.
    The resulting data can be serialised, diffed, or exported.

    Parameters
    ----------
    basis:
        An existing vibe-qc C++ basis set.
    molecule:
        Optional molecule to resolve atom indices to element symbols.
        If ``None``, element symbols are guessed from shell origins
        (may be inaccurate for multi-element systems).
    name:
        Override name.  Defaults to ``basis.name``.
    role:
        Basis role hint (no role survives libint round-trip).
    family:
        Basis family hint.

    Returns
    -------
    BasisSetData
    """
    from ._element_data import SYMBOL as _Z_TO_SYMBOL
    from .model import ElementBasis, HarmonicType, Primitive, Provenance, Shell

    shells = basis.shells()

    # Build atom-index -> element-symbol map from the molecule.
    atom_z: dict[int, int] = {}
    if molecule is not None:
        for i, atom in enumerate(molecule.atoms):
            atom_z[i] = atom.Z

    # Group shells by element symbol, keeping shells only from the
    # first atom of each element (all same-element atoms carry
    # identical basis data in standard calculations).
    first_atom_for: dict[str, int] = {}
    elem_shells: dict[str, list[Shell]] = {}
    for s in shells:
        z = atom_z.get(s.atom_index, 0)
        sym = _Z_TO_SYMBOL.get(z, f"Z{z}")
        if sym not in first_atom_for:
            first_atom_for[sym] = s.atom_index
        elif s.atom_index != first_atom_for[sym]:
            continue  # skip shells from subsequent atoms of this element

        am = [s.l]
        harmonic = HarmonicType.SPHERICAL if s.pure else HarmonicType.CARTESIAN
        prims = [
            Primitive(exponent=e, coefficient=c)
            for e, c in zip(s.exponents, s.coefficients)
        ]
        elem_shells.setdefault(sym, []).append(
            Shell(angular_momentum=am, harmonic_type=harmonic, primitives=prims)
        )

    elements = {
        sym: ElementBasis(element=sym, shells=sh) for sym, sh in elem_shells.items()
    }

    return BasisSetData(
        schema_version="1.0.0",
        name=name or basis.name,
        description=f"Extracted from C++ BasisSet: {basis.name}",
        role=BasisRole(role),
        basis_family=family,
        elements=elements,
        provenance=Provenance(origin="generated"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symbol_for_z(z: int) -> str:
    from ._element_data import SYMBOL

    return SYMBOL.get(z, f"Z{z}")
