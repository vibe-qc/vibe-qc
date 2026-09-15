"""Build an ORCA single-point input deck for one parity / benchmark cell.

Pure string templating — no ORCA, no vibe-qc. Keeps the SCF settings
matched to the vibe-qc side of ``tests/test_parity_hf_dft.py`` as
closely as ORCA's keyword surface allows.
"""
from __future__ import annotations

from .cases import GEOMETRIES, Cell, geometry_bohr_symbols

# vibe-qc method id -> ORCA simple-input functional keyword.
#
# R vs U is left to ORCA: it picks RHF/RKS for multiplicity 1 and
# UHF/UKS for multiplicity > 1 from the `*xyz charge mult` line — the
# same convention runner_orca.py::_ORCA_FUNC_MAP already relies on, so
# "RHF" and "UHF" both map to "HF", etc.
#
# B3LYP: ORCA's *bare* "B3LYP" is the VWN5 variant. vibe-qc's "b3lyp"
# follows that ORCA convention as of commit 822e350 (libxc B3LYP5), so
# bare "B3LYP" is the matching keyword — NOT "B3LYP/G" (the Gaussian
# VWN3 variant). PBE0: ORCA "PBE0" == libxc PBEH == vibe-qc "pbe0".
_METHOD_TO_ORCA = {
    "RHF": "HF",
    "UHF": "HF",
    "RKS-PBE": "PBE",
    "UKS-PBE": "PBE",
    "RKS-B3LYP": "B3LYP",
    "UKS-B3LYP": "B3LYP",
    "RKS-PBE0": "PBE0",
    "UKS-PBE0": "PBE0",
}

# ORCA's name for the def2-universal-JKfit auxiliary basis (Weigend).
# It resolves to the same Gaussians as vibe-qc's / PySCF's
# "def2-universal-jkfit" — so a DF cell is the same mathematical
# approximation in both codes (see test_parity_hf_dft.py::_DF_AUX).
_ORCA_JK_AUX = "def2/JK"
# ORCA's RI-J-only aux (def2-universal-Jfit). RIJCOSX uses RI for J and
# the seminumerical chain-of-spheres kernel for K, so it needs the
# J-only aux — vibe-qc's matching spelling is "def2-universal-jfit".
_ORCA_J_AUX = "def2/J"

# ORCA's basis-set spelling. ORCA accepts "def2-SVP" / "def2-TZVP" /
# "cc-pVDZ" / "cc-pVTZ" case-insensitively; normalise from the
# lowercase vibe-qc spelling.
_BASIS_TO_ORCA = {
    "def2-svp": "def2-SVP",
    "def2-tzvp": "def2-TZVP",
    "cc-pvdz": "cc-pVDZ",
    "cc-pvtz": "cc-pVTZ",
}

# ORCA's MP2 correlation-fitting auxiliary basis ("/C" suffix —
# Weigend-Köhn-Hättig). Used by an `! RI-MP2 <auxC> <basis>` deck.
# vibe-qc's matching RI aux is the orbital basis + "-ri" / "-rifit"
# (resolved on the vibe-qc side when the MP2 parity cells were added).
# Both spellings denote the standard RI-C basis for the orbital set,
# so an RI-MP2 cell is the same mathematical approximation in both
# codes — the assumption is checked in M5.
_BASIS_TO_ORCA_CFIT = {
    "def2-svp": "def2-SVP/C",
    "def2-tzvp": "def2-TZVP/C",
    "cc-pvdz": "cc-pVDZ/C",
    "cc-pvtz": "cc-pVTZ/C",
}

# VeryTightSCF == ORCA's ~1e-9 energy + tight density predefined
# tolerance set. The first runs showed plain TightSCF (~1e-8 energy)
# stops with the density still only converged to ~5e-5 — which leaks
# into E_1e / E_coulomb as a ~1e-4 cross-code difference. VeryTightSCF
# pushes the ORCA density much closer to vibe-qc's (which runs
# conv_tol_energy = 1e-12), so the parity comparison reflects the
# integral / Fock-build / XC implementations, not a convergence gap.
_SCF_TOL_KEYWORD = "VeryTightSCF"
_MAX_ITER = 300


def _orca_mp2_simple_input(cell: Cell) -> str:
    """The ORCA `! ...` line for an MP2 / UMP2 cell.

    ``df=False`` -> ``! MP2 <basis>`` (canonical four-index MP2).
    ``df=True``  -> ``! RI-MP2 <auxC> <basis>`` (RI-MP2).

    ORCA picks the RHF/UHF reference (hence MP2 vs UMP2) from the
    ``*xyz charge mult`` line, exactly as for the SCF cells.

    ``NoFrozenCore`` is mandatory for this historical parity protocol.
    The vibe-qc side explicitly sets ``n_frozen_core=0`` even though its
    unqualified public default is now the published chemical core.  The
    ``%method FrozenCore FC_NONE`` block in :func:`build_orca_input`
    enforces the same in ORCA; the simple-input keyword is the belt to that
    block's braces.
    """
    basis = _BASIS_TO_ORCA.get(cell.basis, cell.basis)
    if cell.df:
        aux_c = _BASIS_TO_ORCA_CFIT.get(cell.basis)
        if aux_c is None:
            raise ValueError(
                f"orca_input: no ORCA correlation-fitting aux mapping "
                f"for basis {cell.basis!r} (RI-MP2 cell {cell.cell_id}) "
                f"— add it to _BASIS_TO_ORCA_CFIT."
            )
        parts = ["RI-MP2", aux_c, basis]
    else:
        parts = ["MP2", basis]
    parts += ["NoFrozenCore", _SCF_TOL_KEYWORD, "Bohrs"]
    return "! " + " ".join(parts)


def orca_simple_input(cell: Cell) -> str:
    """The ORCA `! ...` simple-input line for `cell`."""
    if cell.is_mp2:
        return _orca_mp2_simple_input(cell)
    func = _METHOD_TO_ORCA.get(cell.method)
    if func is None:
        raise ValueError(
            f"orca_input: no ORCA keyword mapping for method "
            f"{cell.method!r} (cell {cell.cell_id})"
        )
    basis = _BASIS_TO_ORCA.get(cell.basis, cell.basis)
    parts = [func, basis]
    if cell.cosx:
        # RIJCOSX = RI-J + seminumerical chain-of-spheres K — matches
        # vibe-qc's cosx=True path. RI-J takes the J-only aux (def2/J).
        parts += [_ORCA_J_AUX, "RIJCOSX"]
    elif cell.df:
        # RIJK = RI on BOTH J and K — matches vibe-qc's density_fit=True
        # path (DF-J + DF-K, parity.py::_make_jk_builders). For a pure
        # GGA (no HF exchange) ORCA simply ignores the K side.
        parts += [_ORCA_JK_AUX, "RIJK"]
    parts.append(_SCF_TOL_KEYWORD)
    # Bohrs: the *xyz block is fed in Bohr (see build_orca_input). vibe-qc
    # and ORCA then sit on identical nuclear coordinates — no per-code
    # Angstrom->Bohr rounding shifting E_nuc between the two codes.
    parts.append("Bohrs")
    return "! " + " ".join(parts)


def build_orca_input(cell: Cell, *, nprocs: int = 1) -> str:
    """Return the full ORCA input deck text for `cell`.

    `nprocs` drives the `%pal` block — 1 (default) is serial, the
    parity-cell setting. The speed benchmark raises it to match the
    `--cpus` it claims on `vq submit`.
    """
    if cell.system not in GEOMETRIES:
        raise ValueError(f"orca_input: unknown system {cell.system!r}")
    multiplicity = cell.spin + 1

    lines = [
        orca_simple_input(cell),
        "%scf",
        f"  MaxIter {_MAX_ITER}",
        "end",
    ]
    if cell.is_mp2:
        # The vibe-qc parity runner explicitly sets n_frozen_core=0. Force
        # ORCA all-electron so the two codes compute the *same* correlation
        # energy. NoFrozenCore on the
        # `!` line + FC_NONE here are belt-and-braces (ORCA honours
        # whichever is more specific; setting both removes ambiguity).
        lines += ["%method", "  FrozenCore FC_NONE", "end"]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.append(f"*xyz {cell.charge} {multiplicity}")
    # Coordinates in Bohr — the `Bohrs` keyword on the `!` line tells
    # ORCA to read them as Bohr, matching vibe-qc's nuclear coordinates
    # exactly (geometry_bohr_symbols does the one shared conversion).
    for symbol, (x, y, z) in geometry_bohr_symbols(cell.system):
        lines.append(f"  {symbol:<2s} {x:>18.12f} {y:>18.12f} {z:>18.12f}")
    lines.append("*")
    lines.append("")  # trailing newline
    return "\n".join(lines)
