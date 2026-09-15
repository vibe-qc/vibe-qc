"""Derive an artifact path from a job's output *stem*.

``run_job(output=...)`` / ``run_periodic_job(output=...)`` take a path
**stem**, not a filename.  The documented contract
(:doc:`docs/user_guide/output_files.md`, "``output=`` ... path stem; files
become ``{output}.out``, ``{output}.molden``, ``{output}.traj``") is that
the suffix is *appended*.

:meth:`pathlib.Path.with_suffix` does not append -- it *substitutes*,
treating everything after the final dot of the final path component as a
replaceable extension.  For any stem that carries a dot -- a lattice
constant, a scale factor, a tolerance, a version -- the two differ, and
the difference destroys data silently::

    Path("output_scale_4.08").with_suffix(".out")  -> output_scale_4.out
    Path("output_scale_4.12").with_suffix(".out")  -> output_scale_4.out
    Path("lih-gdf-rhf-a4.0082-k222").with_suffix(".out")
                                                   -> lih-gdf-rhf-a4.out

The calculations still exit 0; only the captured output is truncated onto
a shared name, so a scan of nine lattice points writes nine times to one
file and eight results cease to exist.  Nothing warns.  GitLab issue #254
records three independent instances, the third of which converted the
resulting ``FileNotFoundError`` into a confident and *wrong*
``route-mismatch`` verdict on seven healthy calculations.

:func:`stem_sibling` is the append-based replacement.  Use it for every
artifact path derived from a job stem.

Idempotence, and why it is part of the contract
-----------------------------------------------
Some call sites deliberately pass a stem that already ends in the target
suffix, to name a *compound* artifact: the periodic runner asks for
``{stem}.opt.xyz`` and hands that to the extended-XYZ writer, which then
re-derives ``.xyz`` from it.  Under substitution that re-derivation was a
no-op; under naive appending it would produce ``{stem}.opt.xyz.xyz``.
:func:`stem_sibling` therefore returns the path unchanged when the name
already ends with exactly the requested suffix, which keeps those
compound names working while still appending for every ordinary stem.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["stem_sibling"]


def stem_sibling(stem: os.PathLike | str, suffix: str) -> Path:
    """Return the artifact ``{stem}{suffix}`` beside *stem*.

    Parameters
    ----------
    stem
        The job's output stem, e.g. ``Path("output-h2o")`` or
        ``Path("eos/output_scale_4.08")``.  Dots in the final component
        are part of the name and are preserved.
    suffix
        The artifact extension, leading dot included (``".out"``,
        ``".scf.jsonl"``, ``".structure.xsf"``).

    Returns
    -------
    Path
        ``stem`` with *suffix* appended -- or ``stem`` unchanged when its
        final component already ends with exactly *suffix* (see the
        module docstring on compound artifact names).

    Raises
    ------
    ValueError
        If *suffix* is empty or does not begin with ``"."``.  Both are
        caller bugs that would otherwise produce a silently misnamed
        artifact, which is the failure mode this helper exists to end.
    """
    if not suffix or not suffix.startswith("."):
        raise ValueError(
            "stem_sibling(suffix=...) must be a non-empty extension "
            f"beginning with '.', got {suffix!r}"
        )
    path = Path(os.fspath(stem))
    name = path.name
    if len(name) > len(suffix) and name.endswith(suffix):
        return path
    return path.parent / (name + suffix)
