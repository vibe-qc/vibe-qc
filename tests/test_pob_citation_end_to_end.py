"""End-to-end pob-* citation surface — emitted DOIs reach disk.

Third and final pin in the pob-* citation-restoration audit. The
prior pins close two surfaces:

* **File-level** — ``tests/test_basis_citation_coverage.py``'s
  ``test_inline_doi_in_header`` asserts each of ``pob-tzvp``,
  ``pob-tzvp-rev2``, ``pob-dzvp-rev2`` carries a concrete DOI in its
  ``! Cite: …`` header line (basis_library/basis/ + custom/). Pinned
  by commit ``0e6f6df4`` after restoration in ``8267c9c9``.
* **Route-table** — ``python/vibeqc/output/citations/database.toml``
  maps each pob-* name to ``entries.peintinger_pob_tzvp_2013`` (and
  ``entries.vilela_oliveira_pob_rev2_2019`` for the -rev2 pair).
  Asserted by ``tests/test_citations.py::test_pob_tzvp_rev2_pulls_*``
  at the assembly layer.

This file is the **output-channel pin**: when a user actually
runs an SCF with ``run_job(..., basis="pob-*", ...)`` the
publication DOI must appear in at least one citation file that
``run_job`` writes to disk (``.references``, ``.bibtex``, or the
in-``.out`` references block). The file-level + route-table pins
guarantee the data is *available*; this one guarantees the
pipeline actually *emits* it.

Audit history:
    22aface3  stripped the per-publication ``! Cite: …`` headers
    8267c9c9  restored them on pob-*.g94 (May 24, 2026)
    0e6f6df4  pinned them at file level + clarified the convention
    f2a7419c  CHANGELOG note for 0e6f6df4
    (this file) pins the output channel

Per AGENTS.md § 8 + CLAUDE.md § 1 + § 8.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vibeqc as vq


# DOI fingerprints that MUST appear in the emitted citation surface
# for each pob-* basis. These pin the values currently set in
# python/vibeqc/output/citations/database.toml:
#   entries.peintinger_pob_tzvp_2013      → 10.1002/jcc.23153
#   entries.vilela_oliveira_pob_rev2_2019 → 10.1002/jcc.26013
# The -rev2 bases route to *both* papers; pob-tzvp routes to the
# 2013 paper only.
POB_DOI_FINGERPRINTS: dict[str, tuple[str, ...]] = {
    "pob-tzvp":      ("10.1002/jcc.23153",),
    "pob-tzvp-rev2": ("10.1002/jcc.23153", "10.1002/jcc.26013"),
    "pob-dzvp-rev2": ("10.1002/jcc.23153", "10.1002/jcc.26013"),
}


def _h2_molecule() -> vq.Molecule:
    """Smallest pob-compatible molecular fixture. H₂ is closed-shell
    so the test runs through RHF rather than UHF; pob-* covers H
    upward, so this is the cheapest run that exercises the pob
    citation route through the full ``run_job`` pipeline."""
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),  # bohr
    ])


@pytest.mark.parametrize("basis_name", sorted(POB_DOI_FINGERPRINTS))
def test_pob_citation_surface_carries_doi(
    tmp_path: Path, basis_name: str,
) -> None:
    """Run ``run_job`` with the pob-* basis on H₂ and assert every
    expected DOI lands in at least one of the citation-surface files
    ``run_job`` writes (``.references``, ``.bibtex``, or the
    in-``.out`` references block).

    Whether the DOI arrives via the route table (primary channel) or
    via the inline ``! Cite: …`` header readback (secondary channel)
    is intentionally not asserted — the contract is "the DOI surfaces
    at the user output", not "the DOI surfaces via channel X". Either
    channel satisfies CLAUDE.md § 8.
    """
    mol = _h2_molecule()
    # Sanitize "*" out of stems just in case (no pob-* name contains
    # one today, but the suffix sweep is cheap insurance).
    stem = tmp_path / f"h2_{basis_name.replace('*', 's')}"

    vq.run_job(
        mol,
        basis=basis_name,
        method="rhf",
        output=str(stem),
        verbose=0,
    )

    # Citation surface = every file ``run_job`` writes that is
    # designed to carry references.
    candidate_paths = [
        stem.with_suffix(".references"),
        stem.with_suffix(".bibtex"),
        stem.with_suffix(".out"),
    ]
    present = [p for p in candidate_paths if p.is_file()]
    assert present, (
        f"run_job emitted no citation-surface files for "
        f"basis={basis_name!r}. Expected at least one of "
        f"{[p.name for p in candidate_paths]} to land on disk."
    )
    surface_text = "\n".join(p.read_text(errors="replace") for p in present)

    for doi in POB_DOI_FINGERPRINTS[basis_name]:
        assert doi in surface_text, (
            f"DOI {doi!r} missing from emitted citation surface for "
            f"basis={basis_name!r}. Channels checked: "
            f"{[p.name for p in present]}. The route table "
            f"(database.toml) AND the inline ! Cite: header in "
            f"basis_library/basis/{basis_name}.g94 both carry this "
            f"DOI; neither reached the user output."
        )
