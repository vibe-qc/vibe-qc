"""Write a candidate basis as a temp .g94 file libint can pick up.

The vibe-qc molecular SCF (`vibeqc.run_rhf`, `run_uhf`, ...) takes a
`vibeqc.BasisSet(mol, name)` where ``name`` resolves to a
``.g94`` file under ``$LIBINT_DATA_PATH/basis/``. There is currently
no in-process API to construct a BasisSet from raw exponent/coefficient
arrays, so the optimisation engine drops a fresh ``.g94`` per
evaluation and asks libint to read it.

This module manages the temp-directory lifecycle, sets
``LIBINT_DATA_PATH`` to a private directory layered ON TOP of the
package's bundled basis library, and produces a unique basis name
per evaluation (so libint never returns a cached parse from a
previous iteration).

Usage
-----

>>> with TempBasisLibrary() as lib:
...     name = lib.write_g94(atoms_dict)
...     # ``$LIBINT_DATA_PATH`` is now lib.path; ``name`` resolves there.
...     basis = vq.BasisSet(mol, name)
...     result = vq.run_rhf(mol, basis)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..basis_crystal import CrystalAtomBasis


class TempBasisLibrary:
    """Temporary $LIBINT_DATA_PATH for emitting candidate basis files.

    On enter: creates a temp directory with a ``basis/`` subdirectory,
    sets ``LIBINT_DATA_PATH``, and returns a context handle. On exit:
    restores the previous environment and deletes the temp tree.

    The class does NOT copy bundled standard basis sets in. Each
    candidate basis is a fresh file with a unique name, addressed
    explicitly. If you also need stock basis sets in the same
    optimisation (rare -- but useful for "fix Li at def2-TZVP, vary
    only H"), pass ``inherit_bundled=True`` and the bundled
    ``basis/`` directory is hard-linked / copied in.
    """

    def __init__(self, *, inherit_bundled: bool = False, prefix: str = "vibeqc-opt-"):
        self._inherit_bundled = inherit_bundled
        self._prefix = prefix
        self.path: Optional[Path] = None
        self._previous_libint: Optional[str] = None

    def __enter__(self) -> "TempBasisLibrary":
        tmp = Path(tempfile.mkdtemp(prefix=self._prefix))
        (tmp / "basis").mkdir(parents=True, exist_ok=True)
        if self._inherit_bundled:
            # Find the package-bundled basis library.
            from .. import basis_library  # type: ignore[attr-defined]
            bundled = Path(basis_library.__file__).parent / "basis"
            for f in bundled.glob("*.g94"):
                # Hard-link if we can (cheap), copy otherwise.
                try:
                    os.link(f, tmp / "basis" / f.name)
                except OSError:
                    shutil.copy2(f, tmp / "basis" / f.name)
        self.path = tmp
        self._previous_libint = os.environ.get("LIBINT_DATA_PATH")
        os.environ["LIBINT_DATA_PATH"] = str(tmp)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._previous_libint is None:
            os.environ.pop("LIBINT_DATA_PATH", None)
        else:
            os.environ["LIBINT_DATA_PATH"] = self._previous_libint
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)

    # ---- emission -----------------------------------------------------------

    def write_g94(self, atoms: dict[str, "CrystalAtomBasis"], *,
                  basis_name: Optional[str] = None) -> str:
        """Emit a fresh .g94 file from the atoms dict; return its basis name.

        The default basis name uuids one per call so libint's BasisSet
        constructor never returns a cached parse from a previous
        iteration.
        """
        if self.path is None:
            raise RuntimeError("TempBasisLibrary used outside its context")
        from ..basis_crystal import emit_g94  # local import: optional dep
        name = basis_name or f"opt-{uuid.uuid4().hex[:8]}"
        target = self.path / "basis" / f"{name}.g94"
        text = emit_g94(list(atoms.values()))
        target.write_text(text)
        return name
