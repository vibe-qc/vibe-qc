"""CRYSTAL isolated-atom .d12 emitter (targets CRYSTAL23).

Uses the same CRYSTAL backend as :mod:`vibe_basis.backends.crystal`,
but for isolated atoms in a large periodic box (P1, 10 Å cubic cell)
instead of crystal structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .crystal import (
    _CRYSTAL_DFT_FUNCTIONALS,
    _deck,
    _NEEDS_HUGEGRID,
    _tolinteg_block,
    REFERENCE_TOLINTEG,
)

_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
}


#: Ground-state spin multiplicity (2S+1) of the neutral free atom.
#:
#: Duplicated from ``vibeqc.atomization._GROUND_STATE_MULTIPLICITY``
#: because vibe-basis must not import vibe-qc
#: (``tests/test_no_vibeqc_dependency.py``). The two are pinned equal by
#: ``tests/basisset_dev/test_atom_reference_convention.py`` on the
#: vibe-qc side, which can see both.
GROUND_STATE_MULTIPLICITY: dict[int, int] = {
    1: 2, 2: 1,                                              # H He
    3: 2, 4: 1, 5: 2, 6: 3, 7: 4, 8: 3, 9: 2, 10: 1,         # Li-Ne
    11: 2, 12: 1, 13: 2, 14: 3, 15: 4, 16: 3, 17: 2, 18: 1,  # Na-Ar
    19: 2, 20: 1,                                            # K Ca
    31: 2, 32: 3, 33: 4, 34: 3, 35: 2, 36: 1,                # Ga-Kr
}


def _spin_blocks(
    Z: int,
    method_lower: str,
    n_unpaired: Optional[int],
    spinlock_cycles: int,
    hugegrid: bool = False,
) -> tuple[int, str, str, str]:
    """Resolve the spin state and build the three spin/symmetry blocks.

    Returns ``(n_unpaired, symmetry_block, method_block, spinlock_block)``.
    Shared by the isolated-atom and counterpoise emitters so the two can
    never drift on the convention that matters most (see this module's
    ``emit_input_atom`` docstring for why 37 kJ/mol rides on it).
    """
    if n_unpaired is None:
        if Z not in GROUND_STATE_MULTIPLICITY:
            raise ValueError(
                f"no tabulated free-atom ground state for Z={Z}; pass "
                f"n_unpaired= explicitly (supported: "
                f"{sorted(GROUND_STATE_MULTIPLICITY)})"
            )
        n_unpaired = GROUND_STATE_MULTIPLICITY[Z] - 1
    if n_unpaired < 0:
        raise ValueError(f"n_unpaired must be >= 0, got {n_unpaired}")

    open_shell = n_unpaired > 0
    is_hf = method_lower in ("rhf", "hf")

    if is_hf:
        method_block = "\nUHF" if open_shell else ""
    else:
        spin = "\nSPIN" if open_shell else ""
        # HUGEGRID only when asked for. The manual prescribes it for
        # the SCAN family, but no deck in the pob reference set uses it,
        # and matching that protocol is what makes our numbers
        # comparable to the published ones. Whichever way it is set, the
        # bulk and its free atoms must agree; see
        # `crystal._NEEDS_HUGEGRID`.
        grid = ("\nHUGEGRID"
                if hugegrid and method_lower in _NEEDS_HUGEGRID else "")
        method_block = f"\nDFT\n{method_lower.upper()}{spin}{grid}\nEND"

    symmetry_block = "SYMMREMO\n" if open_shell else ""
    spinlock_block = (
        f"SPINLOCK\n{n_unpaired} {spinlock_cycles}\n" if open_shell else ""
    )
    return n_unpaired, symmetry_block, method_block, spinlock_block


def emit_input_atom_counterpoise(
    struct,
    atom_index: int,
    inline_basis: str,
    method: str = "rhf",
    *,
    nstar: int = 30,
    rmax: float = 10.0,
    trasremo: bool = True,
    toldee: int = 8,
    n_unpaired: Optional[int] = None,
    spinlock_cycles: int = 50,
    tolinteg: Optional[tuple[int, ...]] = REFERENCE_TOLINTEG,
    hugegrid: bool = False,
) -> Optional[str]:
    """Build a **counterpoise-corrected** free-atom deck via ``ATOMBSSE``.

    The atom is computed *in the ghost basis of its own crystal*: the
    full unit cell is written out, and ``ATOMBSSE`` keeps one atom real
    while the neighbours within ``nstar`` stars and ``rmax`` Å
    contribute basis functions only. The difference from a bare atom is
    the basis-set superposition error.

    Method: the counterpoise correction of Boys and Bernardi,
    *Mol. Phys.* **19**, 553 (1970), doi:10.1080/00268977000101561 --
    each fragment is computed in the *full* basis of the complex, so the
    basis-set superposition error cancels in the difference. Here the
    "complex" is the crystal and the fragment is one atom, which is what
    CRYSTAL's ``ATOMBSSE`` builds.

    The solid-state scheme being reproduced is Vilela Oliveira, Laun,
    Peintinger & Bredow, *J. Comput. Chem.* **40**, 2364 (2019),
    doi:10.1002/jcc.26013 -- the BSSE-corrected pob-rev2 basis sets --
    extended to the fifth period by Laun & Bredow,
    *J. Comput. Chem.* **43**, 839 (2022), doi:10.1002/jcc.26839.

    (vibe-qc's own citation database already carries
    ``boys_bernardi_1970`` routed as ``routes.properties.counterpoise``;
    this docstring is the vibe-basis-side end of the same chain, since a
    driver campaign does not run through ``vibeqc.output``.)

    Why this exists
    ---------------
    The published pob cohesive energies are computed this way: every
    free-atom deck in the reference set carries ``ATOMBSSE``. A cohesive
    energy assembled from *bare* atoms is therefore not comparable to
    them, and the discrepancy is the BSSE of the basis in question,
    which is precisely the quantity a basis-set campaign is trying to
    reduce. Matching the protocol is what makes our numbers and the
    published ones the same quantity.

    Consequence worth knowing: a counterpoise atom energy is
    **system-specific** by construction, since the ghost basis is the
    host crystal's. Unlike a bare free atom it cannot be shared across
    compounds, so the campaign cache must key it by host as well. See
    ``vibe_basis.cache.atom_key``'s ``host`` argument.

    Parameters
    ----------
    struct
        The host :class:`~vibe_basis.io.structures.Structure`. Its
        asymmetric unit is written out in full.
    atom_index
        1-based label of the real atom **in the reference cell**, which
        is CRYSTAL's ``IAT``. Indexes into ``struct.crystal_asymm_unit``.
    inline_basis
        Inline basis text covering **every** element in the cell, not
        just the real atom: the ghosts need their functions.
    method
        ``"rhf"`` / ``"hf"`` or a CRYSTAL functional keyword.
    nstar, rmax
        ``NSTAR`` and ``RMAX``: how many neighbour stars, and out to
        what distance in Å, contribute ghost functions. The reference
        set uses per-system converged values (LiF ``30 / 10.0``, MgO
        ``10 / 5.0``), so treat the defaults as a starting point to
        converge, not a setting to trust.
    toldee
        SCF energy-convergence exponent.
    n_unpaired, spinlock_cycles
        As for :func:`emit_input_atom`.

    Returns
    -------
    str or None
        The .d12 deck, or ``None`` if the structure cannot be expressed
        (no space group, no asymmetric unit, or AFM ordering).

    Notes
    -----
    Per the CRYSTAL23 manual, ``ATOMBSSE`` makes the system **0D**, so no
    reciprocal-space sampling is emitted: *"The system is 0D. No
    reciprocal lattice information is required in the scf input."*
    """
    from .crystal import _format_lattice_line

    if struct.crystal_spacegroup == 0 or not struct.crystal_asymm_unit:
        return None
    if getattr(struct, "afm_pattern", None):
        return None

    method_lower = method.lower()
    if (
        method_lower not in ("rhf", "hf")
        and method_lower not in _CRYSTAL_DFT_FUNCTIONALS
    ):
        raise ValueError(f"unknown method {method!r}")

    n_asym = len(struct.crystal_asymm_unit)
    if not 1 <= atom_index <= n_asym:
        raise ValueError(
            f"atom_index must be a 1-based label into the {n_asym}-atom "
            f"asymmetric unit of {struct.name!r}, got {atom_index}"
        )

    Z = struct.crystal_asymm_unit[atom_index - 1].Z
    n_unpaired, symmetry_block, method_block, spinlock_block = _spin_blocks(
        Z, method_lower, n_unpaired, spinlock_cycles, hugegrid
    )

    lattice_line = _format_lattice_line(struct)
    atom_lines = "\n".join(
        f"{a.Z} {a.fxyz[0]:.6f} {a.fxyz[1]:.6f} {a.fxyz[2]:.6f}"
        for a in struct.crystal_asymm_unit
    )
    # ORIGIN + TRASREMO, exactly as the reference set does for its
    # diamond-structure systems.
    #
    # ATOMBSSE carves a 0D cluster out of the crystal. A symmetry
    # operator carrying a translation cannot map a finite cluster onto
    # itself, so in a **non-symmorphic** group the surviving operators do
    # not close and CRYSTAL stops with
    # "ERROR **** MULTIP **** SYMMOPS DO NOT FORM A GROUP". That is
    # precisely why every space-group-227 system in the pob set (diamond,
    # Si, Ge) failed while the symmorphic rocksalt and zincblende ones
    # did not.
    #
    # ORIGIN "is moved to minimize the number of symmetry operators with
    # finite translation components"; TRASREMO removes what is left. On a
    # symmorphic group there is nothing to minimize and nothing to
    # remove, so emitting both unconditionally is a no-op there -- pinned
    # by a test that the energy is unchanged for a symmorphic system.
    symmetry_prefix = "ORIGIN\nTRASREMO\n" if trasremo else ""
    symbol = _ELEMENT_SYMBOLS.get(Z, f"Z{Z}")
    title = (
        f"{struct.formula} — {symbol} counterpoise atom "
        f"({method.upper()}, ATOMBSSE)"
    )

    return (_deck(
        f"{title}\n"
        "CRYSTAL\n"
        "0 0 0\n"
        f"{struct.crystal_spacegroup}\n"
        f"{lattice_line}\n"
        f"{n_asym}\n"
        f"{atom_lines}\n"
        f"{symmetry_prefix}"
        "ATOMBSSE\n"
        f"{atom_index} {nstar} {rmax:.1f}\n"
        f"{symmetry_block}"
        "END\n"
        f"{inline_basis.strip()}\n"
        "ENDBS\n"
        f"{method_block}\n"
        f"{spinlock_block}"
        f"{_tolinteg_block(tolinteg)}"
        "TOLDEE\n"
        f"{toldee}\n"
        "END\n"
    ))


def emit_input_atom(
    Z: int,
    inline_basis: str,
    method: str = "rhf",
    *,
    box: float = 10.0,
    toldee: int = 8,
    n_unpaired: Optional[int] = None,
    spinlock_cycles: int = 50,
    tolinteg: Optional[tuple[int, ...]] = REFERENCE_TOLINTEG,
    hugegrid: bool = False,
) -> str:
    """Build a CRYSTAL ``.d12`` deck for an isolated atom.

    Uses a large P1 periodic box (default 10 Å) to simulate an
    isolated-atom SCF.  CRYSTAL's molecular mode is available
    but the periodic path with a large box produces the same
    energy and avoids needing a separate MOLECULE parser.

    Spin and symmetry convention
    ----------------------------
    For a **cohesive energy** the free-atom reference must be
    **aspherical and spin-polarised**. This is not a refinement: in the
    plane-wave reference work, spin-restricting the bromine reference
    alone moved KBr's atomization energy by 37 kJ/mol
    (``handovers/HANDOVER_GPAW_PW_REFERENCE.md``, GPAW-PWREF-002), which
    is far larger than the basis-set effects a campaign is trying to
    resolve. Comparing a spin-restricted atom reference against an
    oracle that used a polarised one produces a discrepancy that looks
    like physics.

    So an open-shell atom (``n_unpaired > 0``) gets three things, per the
    CRYSTAL23 manual:

    * ``SYMMREMO`` in the geometry block, removing all symmetry
      operators so the solution may be aspherical (manual, `SYMMREMO`).
    * unrestricted spin: ``UHF`` for Hartree-Fock, or ``SPIN`` inside the
      ``DFT`` block for a functional (manual: *"SPIN unrestricted spin DF
      calculation (default: restricted)"*).
    * ``SPINLOCK NSPIN NCYC`` with ``NSPIN = nalpha - nbeta``, holding the
      occupancy for the first ``NCYC`` cycles. The manual is explicit that
      this is required for an atom: *"UHF and SPINLOCK must be used to
      define a reasonable orbital occupancy"* (the 0D single-atom route).

    A closed-shell atom (``n_unpaired == 0``) gets none of them, since a
    restricted symmetric solution is already the right answer.

    Parameters
    ----------
    Z
        Atomic number.
    inline_basis
        CRYSTAL-format inline basis text for the atom (from
        ``emit_crystal([atom])`` in vibe-qc's basis_crystal).
    method
        ``"rhf"`` / ``"hf"`` or a CRYSTAL functional keyword.
    box
        Cubic lattice constant in Å (default 10.0).
    toldee
        SCF energy-convergence exponent.
    n_unpaired
        Number of unpaired electrons, i.e. ``multiplicity - 1``.
        Defaults to the tabulated neutral ground state for *Z*.
        Passing it explicitly is how a caller requests a non-ground
        state, or supplies an element the table does not cover.
    spinlock_cycles
        ``NCYC`` for ``SPINLOCK``. The lock must be released before
        convergence so the SCF can relax to a true stationary point,
        but late enough that it does not fall back to the spherical
        solution the lock exists to escape.

    Returns
    -------
    str
        The .d12 deck.

    Raises
    ------
    ValueError
        If *method* is unknown, or *Z* has no tabulated ground state and
        no explicit ``n_unpaired``.
    """
    method_lower = method.lower()
    if method_lower not in ("rhf", "hf") and method_lower not in _CRYSTAL_DFT_FUNCTIONALS:
        raise ValueError(f"unknown method {method!r}")

    # SYMMREMO belongs to the geometry block, before its END.
    n_unpaired, symmetry_block, method_block, spinlock_block = _spin_blocks(
        Z, method_lower, n_unpaired, spinlock_cycles, hugegrid
    )
    open_shell = n_unpaired > 0

    symbol = _ELEMENT_SYMBOLS.get(Z, f"Z{Z}")
    state = f"{n_unpaired + 1}-plet" if open_shell else "closed shell"
    title = f"{symbol} — {method.upper()} isolated atom ({state})"

    return (_deck(
        f"{title}\n"
        "CRYSTAL\n"
        "0 0 0\n"
        # Space group 1 (P1, triclinic) takes the FULL cell: a b c and
        # the three angles. Only the cubic groups accept a single
        # parameter. Emitting one number under P1 is a
        # "FORMAT ERROR IN INPUT DECK", which no amount of reading the
        # text reveals.
        "1\n"
        f"{box:.3f} {box:.3f} {box:.3f} 90.000 90.000 90.000\n"
        "1\n"
        f"{Z} 0.5 0.5 0.5\n"
        f"{symmetry_block}"
        "END\n"
        f"{inline_basis.strip()}\n"
        "ENDBS\n"
        f"{method_block}\n"
        f"{spinlock_block}"
        f"{_tolinteg_block(tolinteg)}"
        # The isolated atom sits in a large P1 box, which is still a 3D
        # periodic system, so CRYSTAL wants a real Monkhorst-Pack grid.
        # "0 0" is a format error. One k-point is the right answer here:
        # at a 10 A box the bands are flat and Gamma is the whole zone.
        "SHRINK\n"
        "1 1\n"
        "TOLDEE\n"
        f"{toldee}\n"
        "END\n"
    ))
