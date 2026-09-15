"""Convenience helper for emitting citation siblings from a standalone runner.

The molecular and periodic ``run_job`` driver inlines its own citation
emission so it can interleave warnings into the live ``.out`` log. Most
standalone dispatchers -- ``run_b2plyp``, ``run_dsd_pbep86``,
``run_pwpb95``, ``run_wb97x_d``, ``run_mp2``, ``run_dftb0``,
``run_scc_dftb``, ``run_cpcm_scf`` -- don't open a ``.out`` file, but a
user who wants per-call provenance can still ask for the ``.bibtex``
+ ``.references`` siblings.

:func:`emit_citations` is that one-shot path. It is best-effort
(non-fatal on routing miss or writer crash, mirroring the runner-side
behaviour) and returns the path of the written ``.bibtex`` plus the
plain-text ``## References`` block in case the caller wants to splice
it into its own log.

Typical use inside a standalone runner::

    def run_b2plyp(mol, basis, *, output=None, ...):
        result = run_double_hybrid(mol, basis, "b2plyp", ...)
        if output is not None:
            emit_citations(
                output,
                method="b2plyp",
                basis=basis,
                functional="b2plyp",
                dispersion=dispersion,
            )
        return result

The kwargs mirror :meth:`CitationDatabase.assemble` one-for-one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .bibtex import write_bibtex
from .plain import format_references_block, write_references
from .registry import (
    AssembledCitations,
    load_default_database,
)


__all__ = ["emit_citations"]


def emit_citations(
    output_stem: os.PathLike | str,
    *,
    method: str | None = None,
    basis: str | None = None,
    functional: str | None = None,
    dispersion: str | None = None,
    scf_accelerator: str | None = None,
    periodic: bool = False,
    uses_ecp: bool = False,
    uses_fftw_poisson: bool = False,
    uses_ewald_ao_ft: bool = False,
    uses_ase: bool = False,
    direct_scf: bool = False,
    uses_cpcm: bool = False,
    solvent_variant: str | None = None,
    extra_libraries: Iterable[str] = (),
    **assemble_overrides: object,
) -> tuple[AssembledCitations, Path, Path, str]:
    """Assemble citations for the given job context and write the
    ``{stem}.bibtex`` + ``{stem}.references`` siblings.

    Returns ``(citations, bibtex_path, references_path, block_text)``.
    The block text is the ``## References`` section a caller can
    append to its own text log. The call is non-fatal -- a routing
    miss surfaces as :attr:`AssembledCitations.warnings`, and a
    writer crash propagates (callers should wrap in try/except if
    they want to ignore writer failures).

    The commonly-used assembly knobs are named explicitly above; any
    other :meth:`CitationDatabase.assemble` keyword (e.g.
    ``uses_tddft`` / ``tddft_variant`` from the TDDFT drivers,
    ``uses_gradient`` / ``properties=[...]`` from a property runner)
    can be passed through ``**assemble_overrides``. They are forwarded
    verbatim so a standalone driver can fire its feature route without
    this signature having to enumerate every flag.
    """
    stem = Path(os.fspath(output_stem))
    db = load_default_database()
    _assemble_kwargs: dict[str, object] = {
        "method": method,
        "basis": basis,
        "functional": functional,
        "dispersion": dispersion,
        "scf_accelerator": scf_accelerator,
        "periodic": periodic,
        "uses_ecp": uses_ecp,
        "uses_fftw_poisson": uses_fftw_poisson,
        "uses_ewald_ao_ft": uses_ewald_ao_ft,
        "uses_ase": uses_ase,
        "direct_scf": direct_scf,
        "uses_cpcm": uses_cpcm,
        "solvent_variant": solvent_variant,
        "extra_libraries": extra_libraries,
    }
    _assemble_kwargs.update(assemble_overrides)
    citations = db.assemble(**_assemble_kwargs)
    bib_path = write_bibtex(stem, citations)
    ref_path = write_references(stem, citations)
    block = format_references_block(citations)
    return citations, bib_path, ref_path, block
