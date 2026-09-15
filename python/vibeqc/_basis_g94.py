"""Read libint-style .g94 basis files for external reference scripts.

External reference scripts sometimes need the same bundled pob-* basis
data that vibe-qc uses internally. This module exposes a small
dependency-free conversion to the list representation accepted by
PySCF's input layer; it deliberately does not import PySCF or call any
PySCF functions. Reference programs must still be run out-of-process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


_VIBEQC_BASIS_DIR = (
    Path(__file__).resolve().parent / "basis_library" / "basis"
)


def _g94_atom_blocks(g94_text: str) -> dict[str, str]:
    """Split a .g94 file into per-element blocks (just the shell list).

    Returns ``{symbol_upper: shell_block_string}`` where the shell block
    is the lines between ``{symbol} 0`` and the terminating ``****``,
    in g94 format.
    """
    blocks: dict[str, str] = {}
    cur_sym: Optional[str] = None
    cur_lines: list[str] = []
    for raw in g94_text.splitlines():
        s = raw.rstrip()
        # Lines starting with "!" are comments.
        if s.startswith("!"):
            continue
        toks = s.split()
        if len(toks) == 2 and toks[1] == "0" and toks[0].isalpha():
            # New element header
            if cur_sym is not None and cur_lines:
                blocks[cur_sym.upper()] = "\n".join(cur_lines)
            cur_sym = toks[0]
            cur_lines = []
            continue
        if s.strip() == "****":
            if cur_sym is not None and cur_lines:
                blocks[cur_sym.upper()] = "\n".join(cur_lines)
            cur_sym = None
            cur_lines = []
            continue
        if cur_sym is not None:
            cur_lines.append(raw)
    return blocks


def _g94_block_to_nwchem(symbol: str, g94_block: str) -> str:
    """Convert one element's g94 shell block to NWChem-format text.

    g94 shell header: ``L  nprim  scale``  (e.g., ``S    3   1.00``,
    ``SP   3   1.00`` for combined sp shells).
    NWChem shell header: ``{symbol} {L}`` for plain shells. SP shells
    must be split: emit one ``{symbol} S`` block with the s-coeffs and
    one ``{symbol} P`` block with the p-coeffs (the same exponents).
    """
    out: list[str] = []
    lines = g94_block.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        toks = s.split()
        # Shell header: {L_token} {nprim} {scale}
        if len(toks) >= 2 and toks[0].isalpha() and toks[1].isdigit():
            l_tok = toks[0].upper()
            nprim = int(toks[1])
            prims: list[list[str]] = []
            for _ in range(nprim):
                i += 1
                p = lines[i].split()
                prims.append(p)
            if l_tok == "SP":
                # Two shells: one S with col-1 coeffs, one P with col-2.
                out.append(f"{symbol} S")
                for p in prims:
                    out.append(f"  {p[0]:>22s}  {p[1]:>22s}")
                out.append(f"{symbol} P")
                for p in prims:
                    out.append(f"  {p[0]:>22s}  {p[2]:>22s}")
            else:
                out.append(f"{symbol} {l_tok}")
                for p in prims:
                    # First col is exponent; remaining cols are coeffs.
                    coeff_cols = "  ".join(f"{c:>22s}" for c in p[1:])
                    out.append(f"  {p[0]:>22s}  {coeff_cols}")
            i += 1
        else:
            i += 1
    return "\n".join(out)


_ANGULAR_TO_PYSCF_L = {
    "S": 0,
    "P": 1,
    "D": 2,
    "F": 3,
    "G": 4,
    "H": 5,
}


def _g94_block_to_pyscf_basis(g94_block: str) -> list:
    """Convert one element's g94 block to PySCF's list basis format.

    This is intentionally dependency-free. It mirrors the simple subset
    of Gaussian94 basis syntax used by vibe-qc's bundled basis files:
    plain angular shells and split ``SP`` shells.
    """
    out: list = []
    lines = g94_block.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        toks = s.split()
        if len(toks) >= 2 and toks[0].isalpha() and toks[1].isdigit():
            l_tok = toks[0].upper()
            nprim = int(toks[1])
            prims: list[list[float]] = []
            for _ in range(nprim):
                i += 1
                prims.append([
                    float(x.replace("D", "E")) for x in lines[i].split()
                ])
            if l_tok == "SP":
                s_shell = [0]
                p_shell = [1]
                for p in prims:
                    s_shell.append([p[0], p[1]])
                    p_shell.append([p[0], p[2]])
                out.append(s_shell)
                out.append(p_shell)
            else:
                if l_tok not in _ANGULAR_TO_PYSCF_L:
                    raise NotImplementedError(f"unsupported g94 shell {l_tok!r}")
                shell = [_ANGULAR_TO_PYSCF_L[l_tok]]
                for p in prims:
                    shell.append([p[0], *p[1:]])
                out.append(shell)
        i += 1
    return out


def load_g94_for_pyscf(
    basis_name: str,
    elements: Iterable[str],
) -> dict[str, list]:
    """Return ``{symbol: basis_list}`` for external PySCF scripts.

    The old implementation imported ``pyscf.gto`` and called
    ``gto.basis.parse``. That made PySCF an in-process vibe-qc
    dependency. The helper now returns the equivalent plain Python data
    structure without importing PySCF.
    """

    g94_path = _VIBEQC_BASIS_DIR / f"{basis_name}.g94"
    if not g94_path.is_file():
        raise FileNotFoundError(
            f"vibe-qc bundled basis file not found: {g94_path}"
        )
    text = g94_path.read_text()
    blocks = _g94_atom_blocks(text)
    out: dict[str, list] = {}
    for sym in elements:
        key = sym.upper()
        if key not in blocks:
            raise KeyError(
                f"basis '{basis_name}' has no entry for element {sym!r} "
                f"in {g94_path.name}"
            )
        out[sym] = _g94_block_to_pyscf_basis(blocks[key])
    return out


__all__ = ["load_g94_for_pyscf"]
