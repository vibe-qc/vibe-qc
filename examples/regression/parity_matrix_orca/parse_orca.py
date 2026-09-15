"""Parse an ORCA 6.x SCF output into the cross-code parity decomposition.

This is the ORCA *reference* side of the HF/DFT parity matrix — the
counterpart of ``_pyscf_decompose`` in ``tests/test_parity_hf_dft.py``.
It reads a plain ORCA single-point ``.out`` and returns a dict the
parity comparison can assert against, piece by piece.

**What ORCA actually prints — and the one place the original handover
was wrong.** The retired ORCA-parity handover rev 2 expected separate
``Coulomb Energy :`` and ``Exchange Energy :`` lines. ORCA 6.1.1's
standard ``TOTAL SCF ENERGY`` block does *not* emit those — it prints a
single combined::

    Nuclear Repulsion  :   ...   Eh      -> e_nuc
    One Electron Energy:   ...   Eh      -> e_1e   = tr(D . H_core)
    Two Electron Energy:   ...   Eh      -> e_2e   (see below)

and, for DFT, a ``DFT components`` block with ``E(XC) : ... Eh``.
``Two Electron Energy`` is everything electronic beyond the one-electron
term:

    HF :   e_2e = e_coulomb + e_exchange_hf
    DFT:   e_2e = e_coulomb + alpha_HF * e_exchange_hf + e_xc

So ORCA does not expose the J / K split. The parity comparison
therefore uses a single combined bucket, ``e_coulomb_plus_exchange``::

    e_coulomb_plus_exchange = e_2e - e_xc          # e_xc = 0 for HF

which on the vibe-qc side is ``e_coulomb + e_exchange`` from
``vibeqc.parity.decompose_energy_*``. Five comparison buckets instead
of six (e_nuc, e_1e, e_coulomb_plus_exchange, e_xc, e_total) plus the
MO eigenvalues — still localises a discrepancy to integrals
(e_nuc / e_1e), the Fock build (e_coulomb_plus_exchange), or the XC
quadrature (e_xc); only the J-vs-K resolution is lost.

**The hybrid "final integration" wrinkle.** For a *hybrid* functional
ORCA recomputes the DFT exchange energy on a finer grid at the very
end ("Recomputing exchange energy using gridx3"). The result:

* ``Nuclear Repulsion`` + ``One Electron Energy`` + ``Two Electron
  Energy`` sum to the **SCF-grid** total — internally consistent.
* the ``Total Energy`` line in the block, and ``FINAL SINGLE POINT
  ENERGY``, are the **post-final-integration** total — the SCF-grid
  total plus the ``Exchange energy change after final integration``
  (typically ~1e-5 Ha; exactly zero / absent for HF and pure GGA).

vibe-qc decomposes its result on its *own* SCF grid and does no
separate final integration, so the comparable ORCA quantity is the
**SCF-grid total** = ``e_nuc + e_1e + e_2e``. That is what this parser
returns as ``e_total``; the post-integration value is kept as
``e_total_final`` for provenance.

**Mandatory self-check** (mirrors ``_pyscf_decompose``'s): the parsed
``(e_nuc + e_1e + e_2e) + e_final_integration_delta`` must reconstruct
ORCA's printed ``Total Energy`` / ``FINAL SINGLE POINT ENERGY``. A
wrong parser, a sign slip, a missed final-integration term, or a label
drift between ORCA versions fails loudly here instead of feeding bad
reference numbers into the parity assertions.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union


class OrcaParseError(RuntimeError):
    """Raised when an ORCA output can't be parsed into a decomposition."""


# A signed float as ORCA prints energies: "-122.87357160719932".
_FLOAT = r"[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?"

# "<label> : <value> Eh   <value> eV" — the TOTAL SCF ENERGY block.
_SCF_COMPONENT = {
    "e_nuc": re.compile(rf"^\s*Nuclear Repulsion\s*:\s*({_FLOAT})\s*Eh", re.M),
    "e_1e": re.compile(rf"^\s*One Electron Energy\s*:\s*({_FLOAT})\s*Eh", re.M),
    "e_2e": re.compile(rf"^\s*Two Electron Energy\s*:\s*({_FLOAT})\s*Eh", re.M),
    "e_total_block": re.compile(rf"^\s*Total Energy\s*:\s*({_FLOAT})\s*Eh", re.M),
}
# "E(XC) : ... Eh" — present only for DFT runs.
_E_XC = re.compile(rf"^\s*E\(XC\)\s*:\s*({_FLOAT})\s*Eh", re.M)
# "FINAL SINGLE POINT ENERGY  -76.008426802834" — the canonical total.
_FSP = re.compile(rf"^\s*FINAL SINGLE POINT ENERGY\s+({_FLOAT})", re.M)
# Hybrid functionals only: ORCA recomputes E_x on a finer grid at the
# end. "Exchange energy change after final integration :  0.000031544 Eh"
# is (post-integration - SCF-grid); absent for HF and pure GGA.
_FINAL_INTEGRATION = re.compile(
    rf"Exchange energy change after final integration\s*:\s*({_FLOAT})", re.M)
# "Program Version 6.1.1  -  RELEASE -"
_VERSION = re.compile(r"Program Version\s+([0-9][0-9.]*)")

# --- MP2 / RI-MP2 post-SCF block --------------------------------------
# ORCA 6.x prints, regardless of canonical vs RI:
#     MP2 CORRELATION ENERGY   :   -0.xxxxxxxx Eh   (or "RI-MP2 ...")
#     MP2 TOTAL ENERGY:            -76.xxxxxxxx Eh
# The opposite-spin / same-spin split appears under the MP2 pair-energy
# summary (not every output mode prints it — handled as optional).
_MP2_CORR = re.compile(
    rf"^\s*(?:RI-)?MP2 CORRELATION ENERGY\s*:\s*({_FLOAT})",
    re.M | re.I)
_MP2_TOTAL = re.compile(
    rf"^\s*MP2 TOTAL ENERGY:\s*({_FLOAT})", re.M | re.I)
# Opposite-spin (αβ) / same-spin (αα+ββ) pair-correlation energies.
_MP2_OS = re.compile(
    rf"(?:Opposite[- ]Spin|alpha-beta)\s+(?:pair )?(?:correlation\s+)?"
    rf"energy\s*:?\s*({_FLOAT})", re.I)
_MP2_SS = re.compile(
    rf"(?:Same[- ]Spin|alpha-alpha\s*\+\s*beta-beta)\s+(?:pair )?"
    rf"(?:correlation\s+)?energy\s*:?\s*({_FLOAT})", re.I)
# Orbital-energy table rows: "  NO   OCC          E(Eh)            E(eV)"
# followed by "   0   2.0000     -20.557409      -559.3955".
_ORB_HEADER = re.compile(r"^\s*NO\s+OCC\s+E\(Eh\)\s+E\(eV\)", re.M)
_ORB_ROW = re.compile(
    rf"^\s*(\d+)\s+({_FLOAT}|\d+)\s+({_FLOAT})\s+({_FLOAT})\s*$", re.M
)
_SPIN_UP = re.compile(r"SPIN UP ORBITALS", re.I)
_SPIN_DOWN = re.compile(r"SPIN DOWN ORBITALS", re.I)

# The parser self-check budget. ORCA's TOTAL SCF ENERGY block prints
# 14 significant figures; reconstructing the total from its own
# components is exact to printing precision. 1e-8 catches a real
# parse/sign error with headroom for the last-digit rounding.
_SELF_CHECK_TOL = 1e-8


def _last(pattern: re.Pattern, text: str) -> Union[str, None]:
    """Return the last capture group `pattern` matches in `text`, or None.

    ORCA reprints the SCF blocks once per geometry step; a single-point
    job has exactly one, but taking the *last* match is correct for
    both and robust to a concatenated multi-run file.
    """
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def _to_float(raw: str) -> float:
    # ORCA is C/C++; it never prints Fortran 'D' exponents, but be safe.
    return float(raw.replace("D", "e").replace("d", "e"))


def _parse_orbital_block(text: str, start: int, end: int) -> List[float]:
    """Pull the E(Eh) column out of one orbital-energy table.

    `start`/`end` bound the slice of `text` holding exactly one table
    (one header row + its data rows). Rows stop at the first line that
    isn't a NO/OCC/E(Eh)/E(eV) tuple — e.g. ORCA's
    "*Only the first 10 virtual orbitals were printed.".
    """
    block = text[start:end]
    energies: List[float] = []
    started = False
    for line in block.splitlines():
        row = _ORB_ROW.match(line)
        if row:
            started = True
            energies.append(_to_float(row.group(3)))
        elif started and line.strip():
            # First non-row, non-blank line after the data ends the table.
            break
    return energies


def _parse_orbital_energies(text: str) -> Dict[str, Any]:
    """Return {'mo_energies': [...]} or {'mo_energies_alpha/beta': [...]}.

    Closed-shell ORCA prints one orbital table; open-shell prints a
    "SPIN UP ORBITALS" table then a "SPIN DOWN ORBITALS" table. ORCA
    truncates the virtual list (default: first 10 virtuals) — the
    parity comparison only asserts on the overlapping leading block
    (occupied + whatever virtuals both codes printed), so the
    truncation is harmless.
    """
    headers = list(_ORB_HEADER.finditer(text))
    if not headers:
        raise OrcaParseError("no ORBITAL ENERGIES table found in ORCA output")

    # Use the last ORBITAL ENERGIES section (one per geometry step).
    section_start = text.rfind("ORBITAL ENERGIES")
    section_headers = [h for h in headers if h.start() >= section_start]
    if not section_headers:
        section_headers = headers[-1:]

    up = _SPIN_UP.search(text, section_start)
    down = _SPIN_DOWN.search(text, section_start)
    if up and down:
        alpha = _parse_orbital_block(text, section_headers[0].end(),
                                     down.start())
        beta_header = next(
            (h for h in section_headers if h.start() > down.start()), None)
        if beta_header is None:
            raise OrcaParseError(
                "open-shell ORCA output: SPIN DOWN block has no orbital table")
        beta = _parse_orbital_block(text, beta_header.end(), len(text))
        if not alpha or not beta:
            raise OrcaParseError("open-shell ORCA output: empty orbital table")
        return {
            "mo_energies_alpha": alpha,
            "mo_energies_beta": beta,
        }

    energies = _parse_orbital_block(text, section_headers[-1].end(), len(text))
    if not energies:
        raise OrcaParseError("ORCA orbital-energy table parsed empty")
    return {"mo_energies": energies}


def parse_orca_output(source: Union[str, Path]) -> Dict[str, Any]:
    """Parse an ORCA single-point ``.out`` into a parity decomposition.

    `source` is either the output text or a path to it. Returns a dict
    with the same named energy pieces ``vibeqc.parity.decompose_energy_*``
    produces, *as far as ORCA exposes them*:

      ``e_nuc``, ``e_1e`` — directly parsed.
      ``e_two_electron`` — ORCA's combined ``Two Electron Energy``.
      ``e_xc`` — ORCA's ``E(XC)`` (DFT only; 0.0 for HF).
      ``e_coulomb_plus_exchange`` — ``e_two_electron - e_xc``; the
        bucket the parity comparison compares against vibe-qc's
        ``e_coulomb + e_exchange``.
      ``e_total`` — the **SCF-grid** total (= e_nuc + e_1e + e_2e),
        the quantity vibe-qc's SCF-grid decomposition is comparable to.
      ``e_total_final`` — ORCA's post-final-integration total
        (``FINAL SINGLE POINT ENERGY``); equals ``e_total`` for HF and
        pure GGA, differs by ~1e-5 Ha for hybrids.
      ``e_final_integration_delta`` — the hybrid finer-grid E_x
        correction (0.0 for HF / pure GGA).
      ``mo_energies`` or ``mo_energies_alpha`` / ``mo_energies_beta``.
      ``code``, ``code_version`` — provenance.
      ``e_total_residual`` — the self-check residual (kept for the
        cache record / report; the assert already fired if it was big).

    Raises :class:`OrcaParseError` on a missing block or a failed
    self-check.
    """
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8", errors="replace")
    else:
        text = str(source)

    components: Dict[str, float] = {}
    for key, pattern in _SCF_COMPONENT.items():
        raw = _last(pattern, text)
        if raw is None and key != "e_total_block":
            raise OrcaParseError(
                f"ORCA output missing the '{key}' line of the "
                f"TOTAL SCF ENERGY block — not a completed SCF run?"
            )
        if raw is not None:
            components[key] = _to_float(raw)

    e_xc_raw = _last(_E_XC, text)
    e_xc = _to_float(e_xc_raw) if e_xc_raw is not None else 0.0

    fsp_raw = _last(_FSP, text)
    if fsp_raw is None:
        raise OrcaParseError(
            "ORCA output missing 'FINAL SINGLE POINT ENERGY' — the run "
            "did not finish (SCF non-convergence / crash / killed)."
        )
    e_total_final = _to_float(fsp_raw)

    e_nuc = components["e_nuc"]
    e_1e = components["e_1e"]
    e_2e = components["e_2e"]
    # The SCF-grid total — internally consistent with the printed
    # components, and the quantity vibe-qc's SCF-grid decomposition is
    # comparable to (vibe-qc does no separate final integration).
    e_total = e_nuc + e_1e + e_2e

    # Hybrid functionals: the finer-grid E_x recompute (0.0 / absent
    # for HF and pure GGA).
    fi_raw = _last(_FINAL_INTEGRATION, text)
    e_final_integration_delta = _to_float(fi_raw) if fi_raw is not None else 0.0

    # --- Mandatory self-check (mirrors _pyscf_decompose) -----------------
    # ORCA's identity, accounting for the hybrid final integration:
    #   FINAL SINGLE POINT ENERGY = (E_nuc + E_1e + E_2e) + delta_FI
    # A wrong parse / sign / missed final-integration term / version
    # label drift breaks this — fail loudly, don't poison the matrix.
    reconstructed = e_total + e_final_integration_delta
    residual = reconstructed - e_total_final
    if abs(residual) > _SELF_CHECK_TOL:
        raise OrcaParseError(
            f"ORCA parser self-check FAILED: (E_nuc + E_1e + E_2e) + "
            f"delta_final_integration = {reconstructed:.10f} but FINAL "
            f"SINGLE POINT ENERGY = {e_total_final:.10f} (residual "
            f"{residual:.3e} Eh, tol {_SELF_CHECK_TOL:.0e}). The parser "
            f"is misreading a line, missed the hybrid final-integration "
            f"term, or this ORCA version changed the output format."
        )
    # The block's 'Total Energy' line is the post-final-integration
    # total too — a second, independent cross-check against the FSP line.
    e_total_block = components.get("e_total_block")
    if e_total_block is not None and abs(e_total_block - e_total_final) > 1e-7:
        raise OrcaParseError(
            f"ORCA parser self-check FAILED: 'Total Energy' block value "
            f"{e_total_block:.10f} disagrees with 'FINAL SINGLE POINT "
            f"ENERGY' {e_total_final:.10f} by "
            f"{e_total_block - e_total_final:.3e} Eh."
        )

    version = _last(_VERSION, text) or "unknown"

    out: Dict[str, Any] = {
        "code": "orca",
        "code_version": version,
        "e_nuc": e_nuc,
        "e_1e": e_1e,
        "e_two_electron": e_2e,
        "e_xc": e_xc,
        "e_coulomb_plus_exchange": e_2e - e_xc,
        "e_total": e_total,
        "e_total_final": e_total_final,
        "e_final_integration_delta": e_final_integration_delta,
        "e_total_residual": residual,
    }
    out.update(_parse_orbital_energies(text))
    return out


def parse_orca_mp2_output(source: Union[str, Path]) -> Dict[str, Any]:
    """Parse an ORCA MP2 / RI-MP2 ``.out`` into a parity decomposition.

    The MP2 counterpart of :func:`parse_orca_output`. ORCA's MP2 run
    prints a normal ``TOTAL SCF ENERGY`` block (the HF reference) plus
    a post-SCF MP2 block. This returns the named pieces the MP2 parity
    comparison asserts against — mirroring the dict
    :func:`vibeqc.parity.decompose_energy_rmp2` produces:

      ``e_hf``          — the SCF reference total (HF / UHF), parsed
                          from the ``TOTAL SCF ENERGY`` block's
                          ``Total Energy`` line.
      ``e_corr``        — ``MP2 CORRELATION ENERGY`` (canonical or RI;
                          the ``RI-`` prefix is optional in the label).
      ``e_total``       — ``MP2 TOTAL ENERGY`` if printed, else
                          ``e_hf + e_corr``.
      ``e_os`` / ``e_ss`` — opposite-spin / same-spin pair-correlation
                          energies, or ``None`` when ORCA's output mode
                          did not print the split.
      ``e_total_residual`` — ``(e_hf + e_corr) - e_total``; the
                          self-check residual (the assert below already
                          fired if it was large).
      ``code`` / ``code_version`` — provenance.

    This is deliberately self-contained — it does not import
    ``examples/molecular/mp2_benchmarks/parse_orca_mp2.py`` (which
    itself wraps *this* module for the SCF block; importing it back
    would form a cycle). Canonical-MP2 and RI-MP2 only — the B2PLYP /
    SCS / SOS variants of the mp2_benchmarks parser are out of scope
    for the per-intermediate parity matrix.

    Raises :class:`OrcaParseError` on a missing block or failed
    self-check.
    """
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8", errors="replace")
    else:
        text = str(source)

    # HF reference: the TOTAL SCF ENERGY block's "Total Energy" line.
    scf_raw = _last(_SCF_COMPONENT["e_total_block"], text)
    if scf_raw is None:
        raise OrcaParseError(
            "ORCA MP2 output missing the SCF 'Total Energy' line — "
            "not a completed MP2 run?")
    e_hf = _to_float(scf_raw)

    corr_raw = _last(_MP2_CORR, text)
    if corr_raw is None:
        raise OrcaParseError(
            "ORCA MP2 output missing the 'MP2 CORRELATION ENERGY' line "
            "— the post-SCF MP2 step did not run / did not finish.")
    e_corr = _to_float(corr_raw)

    total_raw = _last(_MP2_TOTAL, text)
    if total_raw is not None:
        e_total = _to_float(total_raw)
    else:
        # Some ORCA modes print only FINAL SINGLE POINT ENERGY.
        fsp_raw = _last(_FSP, text)
        e_total = _to_float(fsp_raw) if fsp_raw is not None else e_hf + e_corr

    os_raw = _last(_MP2_OS, text)
    ss_raw = _last(_MP2_SS, text)
    e_os = _to_float(os_raw) if os_raw is not None else None
    e_ss = _to_float(ss_raw) if ss_raw is not None else None

    # Mandatory self-check: e_hf + e_corr must reconstruct e_total.
    residual = (e_hf + e_corr) - e_total
    if abs(residual) > _SELF_CHECK_TOL:
        raise OrcaParseError(
            f"ORCA MP2 parser self-check FAILED: e_hf ({e_hf:.10f}) + "
            f"e_corr ({e_corr:.10f}) - e_total ({e_total:.10f}) = "
            f"{residual:.3e} Eh (tol {_SELF_CHECK_TOL:.0e}). The parser "
            f"is misreading a line or this ORCA version changed the MP2 "
            f"output format.")

    version = _last(_VERSION, text) or "unknown"
    return {
        "code": "orca",
        "code_version": version,
        "e_hf": e_hf,
        "e_corr": e_corr,
        "e_total": e_total,
        "e_os": e_os,
        "e_ss": e_ss,
        "e_total_residual": residual,
    }


if __name__ == "__main__":  # pragma: no cover - manual spot-check helper
    import json
    import sys

    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <orca-output.out>", file=sys.stderr)
        raise SystemExit(2)
    parsed = parse_orca_output(Path(sys.argv[1]))
    mo = parsed.pop("mo_energies", None)
    mo_a = parsed.pop("mo_energies_alpha", None)
    mo_b = parsed.pop("mo_energies_beta", None)
    print(json.dumps(parsed, indent=2, sort_keys=True))
    if mo is not None:
        print(f"mo_energies: {len(mo)} values, first 6 = {mo[:6]}")
    if mo_a is not None:
        print(f"mo_energies_alpha: {len(mo_a)} values, first 6 = {mo_a[:6]}")
        print(f"mo_energies_beta:  {len(mo_b)} values, first 6 = {mo_b[:6]}")
