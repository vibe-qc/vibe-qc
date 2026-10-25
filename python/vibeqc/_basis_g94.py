"""Hand a bundled basis to an external PySCF script, from the record.

External reference scripts (``examples/periodic_pyscf/``) need the same
bundled pob-* basis data vibe-qc uses internally, in PySCF's list format.
This reads the QVF-Basis record of the bundled library, the source of truth
since #406, and converts it without importing PySCF or any other quantum
chemistry program; reference programs still run out-of-process.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["load_g94_for_pyscf"]


def _shell_to_pyscf(shell) -> list[list]:
    """One record shell -> one or two PySCF shells ``[l, [exp, coeff], ...]``.

    The record stores an SP shell blocked: the N s-contraction primitives,
    then the N p-contraction primitives over the same exponents.
    """
    am = list(shell.angular_momentum)
    prims = shell.primitives
    if am == [0, 1]:
        n = len(prims) // 2
        s_shell = [0] + [[p.exponent, p.coefficient] for p in prims[:n]]
        p_shell = [1] + [[p.exponent, p.coefficient] for p in prims[n:]]
        return [s_shell, p_shell]
    if len(am) != 1:
        raise NotImplementedError(f"unsupported shell angular momentum {am}")
    return [[am[0]] + [[p.exponent, p.coefficient] for p in prims]]


def load_g94_for_pyscf(
    basis_name: str,
    elements: Iterable[str],
) -> dict[str, list]:
    """Return ``{symbol: basis_list}`` for external PySCF scripts.

    ``basis_name`` is resolved the way ``BasisSet`` resolves it (#321), so
    ``"pob-TZVP"`` reads the ``pob-tzvp`` record. Raises ``FileNotFoundError``
    for a name the bundled library has no record for and ``KeyError`` for an
    element the record lacks.
    """
    from .basis_registry import canonical_basis_name
    from .basis_toolkit import library

    stem = canonical_basis_name(basis_name)
    path = library.record_path(stem)
    if not path.is_file():
        raise FileNotFoundError(f"vibe-qc bundled basis record not found: {path}")
    record = library.load_record(stem)
    out: dict[str, list] = {}
    for sym in elements:
        key = str(sym).strip().capitalize()
        if key not in record.elements:
            raise KeyError(
                f"basis '{basis_name}' has no entry for element {sym!r} in {path.name}"
            )
        out[sym] = [pyscf_shell for shell in record.elements[key].shells
                    for pyscf_shell in _shell_to_pyscf(shell)]
    return out
