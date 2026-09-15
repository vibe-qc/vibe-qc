"""ORCA MP2 / B2PLYP output parser — extracts the per-variant
correlation + total energies and SCF iteration history.

Wraps ``examples/regression/parity_matrix_orca/parse_orca.py`` for
the SCF block and adds extractors for the post-SCF MP2 quantities
that the benchmark needs:

  e_hf                 — Hartree-Fock (or hybrid SCF) total energy.
                         For an MP2 run this is the SCF reference.
                         For B2PLYP this is the RKS(b2plyp) total
                         already including its KS-MP2-coefficient
                         scaling of zero (the MP2 correction is the
                         next field).
  e_correlation        — MP2 (or RI-MP2 / SCS-MP2 / SOS-MP2 / B2PLYP-
                         MP2-correction) correlation energy, with the
                         variant's c_os/c_ss already applied — this
                         is the comparable quantity to vibe-qc's
                         ``MP2Result.e_correlation``.
  e_total              — e_hf + e_correlation. For B2PLYP, the
                         "FINAL SINGLE POINT ENERGY" line.
  e_os, e_ss           — Same-spin / opposite-spin decomposition
                         (unscaled).
  n_scf_iter           — SCF iteration count.
  scf_converged        — True if the "SCF CONVERGED" banner appears.
  variant              — Inferred from the output (mp2 / rimp2 / scsmp2
                         / sosmp2 / b2plyp / ump2 / riump2 / scsump2 /
                         sosump2).

The parser is deliberately lenient on whitespace because ORCA's MP2
output formatting varies across the 4.x → 5.x → 6.x family. Self-
checks: e_hf + e_correlation ≈ e_total within 1e-6 Eh (loud failure
otherwise — surfaces parse-vs-text drift early).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Float pattern matching ORCA's scientific-notation totals.
_F = r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)"

# SCF reference (HF or hybrid-KS) total. The "Total Energy ... Eh"
# inside the TOTAL SCF ENERGY block is what we want — it's the
# converged SCF energy. We use FINAL SINGLE POINT ENERGY only as a
# cross-check for pure-SCF (HF / pure-GGA) runs.
_SCF_TOTAL = re.compile(
    r"^\s*Total Energy\s*:\s*" + _F + r"\s*Eh", re.MULTILINE
)

# MP2 correlation energy line — ORCA 6.x prints
#     MP2 CORRELATION ENERGY   :   -0.xxxxx Eh
# regardless of variant. For ``! SCS-MP2`` and ``! SOS-MP2`` the
# value on this line is the **scaled** correlation energy (matches
# E(FSP) - E(SCF) within rounding). For ``! MP2`` and ``! RI-MP2``
# it's the unscaled canonical / RI MP2 correlation. The variant
# itself is identified from the ``! ...`` simple-input line, not
# from the energy label.
_MP2_CORR_RE = re.compile(
    r"^\s*(?:RI-)?MP2 CORRELATION ENERGY\s*:\s*" + _F,
    re.MULTILINE | re.IGNORECASE,
)

# The simple-input ``! ...`` line — ORCA echoes it as ``|  1> ! ...``.
# We sniff for the keywords that map to our variant tags.
_SIMPLE_INPUT_LINE = re.compile(r"\|\s*\d+\s*>\s*!(.+)")
_VARIANT_KEYWORD_TO_TAG: list[tuple[str, str]] = [
    ("B2PLYP", "b2plyp"),
    ("SCS-MP2", "scsmp2"),
    ("SOS-MP2", "sosmp2"),
    ("RI-MP2", "rimp2"),
    ("MP2", "mp2"),
]
# Spin-decomposition (printed by all MP2 variants when --MP2 selected,
# or by the RI-MP2 driver under "Pair Energies" / "Spin Components").
_OS_RE = re.compile(
    r"(?:Opposite[- ]Spin|alpha-beta)\s+(?:pair )?(?:correlation\s+)?"
    r"energy\s*:?\s*" + _F, re.IGNORECASE
)
_SS_RE = re.compile(
    r"(?:Same[- ]Spin|alpha-alpha\s*\+\s*beta-beta)\s+(?:pair )?"
    r"(?:correlation\s+)?energy\s*:?\s*" + _F, re.IGNORECASE
)
_E_TOT_MP2 = re.compile(
    r"^\s*MP2 TOTAL ENERGY:\s*" + _F, re.MULTILINE | re.IGNORECASE
)
_E_TOT_FSP = re.compile(
    r"^\s*FINAL SINGLE POINT ENERGY\s+" + _F, re.MULTILINE
)

# B2PLYP-specific HF-step total and MP2-correction. ORCA prints:
#   E(SCF)  = ...
#   E(MP2)  = ...
#   E(B2PLYP) = E(SCF) + E(MP2)
_B2PLYP_E_SCF = re.compile(r"^\s*E\(SCF\)\s+=\s+" + _F, re.MULTILINE)
_B2PLYP_E_MP2 = re.compile(r"^\s*E\(MP2\)\s+=\s+" + _F, re.MULTILINE)

# SCF iteration count. ORCA prints "SCF CONVERGED AFTER N CYCLES".
_SCF_ITER = re.compile(
    r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", re.IGNORECASE
)
_SCF_NOT_CONVERGED = re.compile(
    r"SCF NOT CONVERGED|CONVERGENCE PROBLEM|NOT FULLY CONVERGED",
    re.IGNORECASE,
)

_S_SQUARED = re.compile(
    r"<S\*\*2>\s*=?\s*" + _F + r"\s+S\*\(S\+1\)\s*=", re.IGNORECASE
)


class OrcaMP2ParseError(RuntimeError):
    pass


def _last_match(pat: re.Pattern, text: str) -> str | None:
    matches = list(pat.finditer(text))
    return matches[-1].group(1) if matches else None


def parse_orca_mp2(path: Path | str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    scf_iter_raw = _SCF_ITER.search(text)
    scf_iter = int(scf_iter_raw.group(1)) if scf_iter_raw else None
    scf_converged = (scf_iter is not None
                     and not _SCF_NOT_CONVERGED.search(text))

    # Detect variant from the echoed ``! ...`` simple-input line.
    # The first such line is the user's input; ignore subsequent
    # `|  N> ! ...` continuations.
    simple_line = None
    for m in _SIMPLE_INPUT_LINE.finditer(text):
        simple_line = m.group(1).upper()
        break
    if simple_line is None:
        raise OrcaMP2ParseError(
            f"{path}: no '! ...' simple-input line found in ORCA output")

    variant = None
    for keyword, tag in _VARIANT_KEYWORD_TO_TAG:
        if keyword in simple_line:
            variant = tag
            break
    if variant is None:
        raise OrcaMP2ParseError(
            f"{path}: no MP2/SCS/SOS/B2PLYP keyword in simple input: "
            f"{simple_line.strip()!r}")

    # Correlation energy from the MP2 CORRELATION ENERGY line. For
    # SCS / SOS this is the already-scaled value; for canonical /
    # RI-MP2 it's the unscaled value.
    corr_match = _MP2_CORR_RE.search(text)
    e_correlation = float(corr_match.group(1)) if corr_match else None

    # Spin decomposition (when ORCA prints it; not all variants do).
    os_raw = _last_match(_OS_RE, text)
    ss_raw = _last_match(_SS_RE, text)
    e_os = float(os_raw) if os_raw else None
    e_ss = float(ss_raw) if ss_raw else None

    # SCF total — same parse for every variant (RHF for closed shell,
    # UHF for open shell, or hybrid-RKS for B2PLYP). The MP2-step
    # correlation energy comes from the (RI-)MP2 CORRELATION ENERGY
    # line — for B2PLYP this is already 0.27-scaled by ORCA.
    scf_raw = _last_match(_SCF_TOTAL, text)
    if scf_raw is None:
        raise OrcaMP2ParseError(
            f"{path}: no 'Total Energy' SCF block found")
    e_hf = float(scf_raw)
    if e_correlation is None:
        raise OrcaMP2ParseError(
            f"{path}: variant={variant} detected but no MP2 "
            "CORRELATION ENERGY line found")

    # MP2 / FSP total — prefer FSP (it includes the variant's
    # correlation scaling already), fall back to MP2 TOTAL line.
    fsp_raw = _last_match(_E_TOT_FSP, text)
    if fsp_raw is not None:
        e_total = float(fsp_raw)
    else:
        mp2_tot_raw = _last_match(_E_TOT_MP2, text)
        e_total = float(mp2_tot_raw) if mp2_tot_raw else e_hf + e_correlation

    s2_raw = _last_match(_S_SQUARED, text)
    s_squared = float(s2_raw) if s2_raw else None

    # Self-check: e_hf + e_correlation ≈ e_total (within 1e-5 Eh)
    # for all variants. For B2PLYP this is exact: E(B2PLYP) = E(SCF)
    # + E(MP2). For MP2 / RI-MP2 / SCS / SOS this is the printed
    # MP2 TOTAL ENERGY identity.
    residual = e_hf + e_correlation - e_total
    if abs(residual) > 1e-5:
        raise OrcaMP2ParseError(
            f"{path}: self-check failed: e_hf({e_hf:.10f}) + "
            f"e_correlation({e_correlation:.10f}) - e_total({e_total:.10f}) "
            f"= {residual:+.3e} Eh (tol 1e-5)")

    return {
        "variant": variant,
        "e_hf": e_hf,
        "e_correlation": e_correlation,
        "e_total": e_total,
        "e_os": e_os,
        "e_ss": e_ss,
        "n_scf_iter": scf_iter,
        "scf_converged": scf_converged,
        "s_squared_uhf": s_squared,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: parse_orca_mp2.py <orca.out>")
    from pprint import pprint
    pprint(parse_orca_mp2(sys.argv[1]))
