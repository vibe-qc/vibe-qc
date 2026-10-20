"""Fetch the composite-method orbital bases (def2-mSVP, def2-mTZVP,
def2-mTZVPP, MINIX, vDZP) used by the v0.9.0 composite 3c stack.

Pulls from two upstream sources:

* **Basis Set Exchange** (``basis_set_exchange`` Python package) —
  primary source where available. Currently has: def2-mTZVP,
  def2-mTZVPP, Grimme vDZP.

* **Psi4 source distribution** (raw.githubusercontent.com) — secondary
  source for bases that BSE does not yet ship under their composite
  3c name. Used for: def2-mSVP, MINIX. Psi4 itself is LGPL-3, but the
  basis-set numerical data shipped under
  ``psi4/share/psi4/basis/*.gbs`` is the published parameter content
  from the originating publication (cited in each file's header) —
  Weigend-Ahlrichs 2005 for the def2 family, Sure-Grimme 2013 for
  MINIX, etc. The numerical content is therefore treated the same way
  BSE / NWChem / Q-Chem treat it: redistributable published
  scientific data with attribution to the originating publication.

  vibe-qc rewrites the per-file header on import to:
    * cite the **originating publication** (not Psi4),
    * record the secondary distribution channel
      (Psi4 master branch, snapshot commit X), and
    * carry the standard vibe-qc license note (CLAUDE.md § 1 + § 8).

Each basis becomes a QVF-Basis record under ``python/vibeqc/basis_library/qvf/``
(kind ``bse``, with the source and citation in its provenance), and ``basis/``
is re-rendered, so it ships
inside the wheel. ``scripts/setup_basis_library.sh`` then mirrors
with every other record for libint runtime discovery.

Run once when bundling a new 3c basis; the resulting .g94 files are
committed to the repo. Users never re-fetch.

Provenance is recorded in each file's header — see CHANGELOG entry
"Bundle 3c orbital bases" for the maintainer-side audit trail of
which upstream commit each numerical content was sourced from.

Usage:
    python scripts/fetch_3c_orbital_bases.py [--dry-run] [--list]
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import urllib.request
from pathlib import Path

import basis_set_exchange as bse


# ---------------------------------------------------------------------------
# Source 1 — Basis Set Exchange (canonical archive).
#
# (BSE basis name, vibe-qc filename without .g94 suffix, header
# citation).
# ---------------------------------------------------------------------------
BSE_TARGETS: list[tuple[str, str, str]] = [
    (
        "def2-mtzvp",
        "def2-mtzvp",
        "Brandenburg, Bannwarth, Hansen, Grimme, J. Chem. Phys. 148, "
        "064104 (2018); DOI: 10.1063/1.5012601 [B97-3c parent basis]",
    ),
    (
        "def2-mtzvpp",
        "def2-mtzvpp",
        "Grimme, Hansen, Ehlert, Mewes, J. Chem. Phys. 154, 064103 "
        "(2021); DOI: 10.1063/5.0040021 [r²SCAN-3c parent basis]",
    ),
    (
        "Grimme vDZP",
        "vdzp",
        "Müller, Hansen, Grimme, J. Chem. Phys. 158, 014103 (2023); "
        "DOI: 10.1063/5.0133026 [ωB97X-3c parent basis; carries "
        "Stuttgart small-core ECPs for heavier elements]",
    ),
]

# ---------------------------------------------------------------------------
# Source 2 — Psi4 GitHub raw, for bases BSE doesn't archive under the
# composite-method name.
#
# (psi4 .gbs URL, vibe-qc filename without .g94 suffix, originating-
# publication citation).
#
# Psi4 license: LGPL-3 (the *code*). The numerical content of these
# basis files is the published parameter data from the cited
# publication; vibe-qc rewrites the header to attribute the original
# publication and notes Psi4 as the secondary distribution channel.
# ---------------------------------------------------------------------------
_PSI4_BASE = (
    "https://raw.githubusercontent.com/psi4/psi4/master/"
    "psi4/share/psi4/basis"
)
PSI4_TARGETS: list[tuple[str, str, str]] = [
    (
        f"{_PSI4_BASE}/def2-msvp.gbs",
        "def2-msvp",
        "Grimme, Brandenburg, Bannwarth, Hansen, J. Chem. Phys. 143, "
        "054107 (2015); DOI: 10.1063/1.4927476 [PBEh-3c parent basis; "
        "modified def2-SVP per Table II of the cited paper]",
    ),
    (
        f"{_PSI4_BASE}/minix.gbs",
        "minix",
        "Sure & Grimme, J. Comput. Chem. 34, 1672 (2013); DOI: "
        "10.1002/jcc.23317 [HF-3c parent basis; MINIS + Huzinaga p-"
        "augmentation per Table 1 of the cited paper]",
    ),
]


def _vibeqc_header(
    basis_label: str,
    citation: str,
    source: str,
) -> str:
    """Build the vibe-qc-canonical .g94 header. Prepended to every
    fetched basis file regardless of upstream source so the citation
    chain is uniform."""
    today = datetime.date.today().isoformat()
    return (
        f"! ----------------------------------------------------------\n"
        f"! Basis set: {basis_label}\n"
        f"! Bundled with vibe-qc v0.9.0 (composite 3c stack)\n"
        f"! Originating publication: {citation}\n"
        f"! Fetched from: {source}\n"
        f"! Bundled date: {today}\n"
        f"! License: numerical parameter data from a published\n"
        f"!   scientific paper; redistribution is standard practice\n"
        f"!   (see python/vibeqc/data_library/README.md § License).\n"
        f"! ----------------------------------------------------------\n"
        f"\n"
    )


def _strip_psi4_preamble(gbs_text: str) -> str:
    """Psi4 .gbs files start with a ``spherical`` directive line +
    a freeform comment block (lines starting with ``!``). libint's
    .g94 parser doesn't read the ``spherical`` directive and expects
    the per-element blocks to begin after the comment header. We
    strip the ``spherical`` line and forward the upstream comments
    verbatim into the file as additional context — they get prepended
    to vibe-qc's canonical header by the caller."""
    lines = gbs_text.splitlines()
    out: list[str] = []
    seen_spherical = False
    for ln in lines:
        stripped = ln.strip()
        if not seen_spherical and stripped == "spherical":
            seen_spherical = True
            continue
        out.append(ln)
    return "\n".join(out) + ("\n" if not gbs_text.endswith("\n") else "")


# Regex matching an ECP block header. Three upstream conventions seen:
#   * Psi4 .gbs:  ``RB-ECP     3     28``   (all-caps element symbol,
#                                            n_core l_max — Psi4 uses
#                                            uppercase consistently in
#                                            ECP block headers)
#   * BSE  .g94:  ``B-ECP     3     2``     (standard symbol case)
#   * Generic:    ``<Element>-ECP   gen``    (gen-form for per-shell
#                                            arbitrary contractions)
#
# Element symbol regex must accept BOTH ``[A-Z][a-z]?`` (Bse/standard)
# AND ``[A-Z]{1,2}`` (Psi4 uppercase) — the union is ``[A-Za-z][A-Za-z]?``.
_ECP_HEADER_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z]?)-ECP\s+(?:gen|\d+\s+\d+)\s*$"
)


def _strip_embedded_ecps(g94_text: str) -> tuple[str, str]:
    """Split a basis-set file into ``(orbital_basis_text, ecp_text)``.

    libint's .g94 parser handles only the orbital basis-set blocks;
    upstream sources (BSE for vDZP, Psi4 for def2-mSVP) embed the
    matching effective-core-potential blocks in the same file. The
    libint parser treats the ECP block headers as orbital shell
    definitions and dies with 'invalid angular momentum label'.

    We split at the `****` separators (which delimit every per-element
    block, basis or ECP) and route each chunk based on whether its
    header line matches an orbital `<Element>  0` shape or an ECP
    `<Element>-ECP ...` shape.

    Both halves keep their leading comments + the `****` separators
    so the resulting files are self-contained and re-parsable by
    libint (orbital side) / libecpint (ECP side).
    """
    # Split keeping the **** markers — we need them in each output.
    chunks = re.split(r"(^\s*\*+\s*$)", g94_text, flags=re.MULTILINE)

    orbital_parts: list[str] = []
    ecp_parts: list[str] = []
    current_is_ecp = False
    current_buffer: list[str] = []
    for chunk in chunks:
        if re.match(r"^\s*\*+\s*$", chunk):
            # Hit a **** separator → flush the buffered block into
            # the right pile, then start a new buffer.
            if current_buffer:
                blob = "".join(current_buffer)
                (ecp_parts if current_is_ecp else orbital_parts).append(blob)
                current_buffer = []
            # Send the separator itself to the current pile so the
            # block structure stays intact in the orbital output.
            # ECP output uses its own markers anyway.
            orbital_parts.append(chunk)
            current_is_ecp = False
            continue
        # Inspect this chunk to decide if it's a basis block or an
        # ECP block. Discriminator: scan up to the first 8 non-comment
        # lines for an ECP header.
        #
        # Why look multiple lines deep: Psi4 .gbs prefaces every ECP
        # block with a basis-style ``<ELEMENT>     0`` line followed
        # by the actual ``<ELEMENT>-ECP   <n_core> <l_max>`` marker
        # one line later. Looking only at the first non-comment line
        # mis-detects the chunk as a basis block.
        is_ecp = False
        non_comment_seen = 0
        for ln in chunk.splitlines():
            stripped = ln.strip()
            if not stripped or stripped.startswith("!"):
                continue
            non_comment_seen += 1
            if _ECP_HEADER_RE.match(stripped):
                is_ecp = True
                break
            if non_comment_seen >= 8:
                break
        current_is_ecp = is_ecp
        current_buffer.append(chunk)
    # Flush trailing block.
    if current_buffer:
        blob = "".join(current_buffer)
        (ecp_parts if current_is_ecp else orbital_parts).append(blob)

    orbital_text = "".join(orbital_parts)
    ecp_text = "".join(ecp_parts)
    return orbital_text, ecp_text


def _fetch_psi4_url(url: str) -> str:
    """Pull a single .gbs file from Psi4's master branch."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "vibeqc-3c-basis-fetcher/0.9.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(
                f"{url}: HTTP {resp.status} (expected 200)"
            )
        return resp.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be fetched without writing files.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the targets and exit.",
    )
    args = parser.parse_args()

    n_total = len(BSE_TARGETS) + len(PSI4_TARGETS)
    if args.list:
        print(f"BSE library version: {bse.version()}")
        print(f"3c composite-method orbital bases to bundle ({n_total}):")
        print()
        print("  From BSE:")
        for src, dst, cite in BSE_TARGETS:
            print(f"    {src:<18s} -> {dst}.g94")
            print(f"      cite: {cite[:80]}{'...' if len(cite) > 80 else ''}")
        print()
        print("  From Psi4 distribution:")
        for url, dst, cite in PSI4_TARGETS:
            print(f"    {url.split('/')[-1]:<18s} -> {dst}.g94")
            print(f"      cite: {cite[:80]}{'...' if len(cite) > 80 else ''}")
        return 0

    from vibeqc.basis_toolkit import Upstream, library

    out_dir = library.records_dir()

    print(f"BSE library version: {bse.version()}")
    print(f"Record output dir:   {out_dir}")
    print()

    n_written = 0
    n_skipped = 0

    def _emit(basis_label: str, dst: str, cite: str, body: str, source: str):
        """Common emission tail: split basis from ECP, write each to its
        own file with the canonical vibe-qc header. Returns a
        (n_basis_elements, n_ecp_elements) summary."""
        nonlocal n_written, n_skipped
        orbital_text, ecp_text = _strip_embedded_ecps(body)
        description = _vibeqc_header(
            basis_label=basis_label, citation=cite, source=source,
        ).replace("! ", "").replace("!", "").strip()
        record = library.record_from_text(
            dst, orbital_text, ecp_text if ecp_text.strip() else None,
            kind="bse", description=description,
            upstream=[Upstream(name=source.split(" (")[0], identifier=basis_label,
                               url=source[source.find("(") + 1:].rstrip(")") or None)],
        )
        record_path = library.record_path(dst)
        if record_path.is_file() and library.load_record(dst).content_sha256() == record.content_sha256():
            n_skipped += 1
            orb_msg = "= (unchanged)"
        else:
            library.save_record(record, record_path)
            n_written += 1
            orb_msg = "+"
        return len(record.elements), len(record.ecps or {}), orb_msg

    # -- BSE targets ---------------------------------------------------------
    for src, dst, cite in BSE_TARGETS:
        if args.dry_run:
            print(f"  [dry-run] BSE  {src} -> {dst}.g94")
            continue
        try:
            upstream = bse.get_basis(src, fmt="gaussian94", header=True)
        except (KeyError, Exception) as exc:
            print(f"  ! {src}: BSE fetch failed ({exc})", file=sys.stderr)
            continue
        n_b, n_e, msg = _emit(
            basis_label=src, dst=dst, cite=cite, body=upstream,
            source=f"Basis Set Exchange v{bse.version()} "
                   f"(https://www.basissetexchange.org)",
        )
        suffix = f"({n_b} elements" + (f", {n_e} ECPs" if n_e else "") + ")"
        print(f"  {msg} BSE  {src:<18s} -> {dst}.g94 {suffix}")

    # -- Psi4 targets --------------------------------------------------------
    for url, dst, cite in PSI4_TARGETS:
        if args.dry_run:
            print(f"  [dry-run] Psi4 {url.split('/')[-1]} -> {dst}.g94")
            continue
        try:
            upstream = _fetch_psi4_url(url)
        except Exception as exc:
            print(f"  ! {url}: fetch failed ({exc})", file=sys.stderr)
            continue
        body = _strip_psi4_preamble(upstream)
        n_b, n_e, msg = _emit(
            basis_label=dst, dst=dst, cite=cite, body=body,
            source=f"Psi4 master branch ({url}) — numerical content "
                   f"from the originating publication, Psi4 is the "
                   f"secondary distribution channel",
        )
        suffix = f"({n_b} elements" + (f", {n_e} ECPs" if n_e else "") + ")"
        print(f"  {msg} Psi4 {dst:<18s} -> {dst}.g94 {suffix}")

    print()
    print(f"Wrote {n_written} new/updated records, kept {n_skipped} unchanged.")
    library.render_library(library.renderings_dir())
    print("Re-rendered basis/; route the citation keys in database.toml and run")
    print("  python scripts/render_basis_library.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
