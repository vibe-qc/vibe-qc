"""Markdown summary report — the bug-hunt hand-back document."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .case import CaseRecord, CodeRow


_STATUS_ICON = {
    "pass":         "✓",
    "marginal":     "≈",
    "fail":         "✗",
    "reference":    "·",
    "error":        "!",
    "unavailable":  "—",
    "pending":      "?",
}


def _fmt_e(val) -> str:
    return f"{val:>14.8f}" if isinstance(val, float) else " " * 14


def _fmt_d(val) -> str:
    return f"{val:>+11.3e}" if isinstance(val, float) else " " * 11


def _fmt_mha(val, *, signed: bool = False) -> str:
    if not isinstance(val, float):
        return " " * 9
    return f"{val:>+9.3f}" if signed else f"{val:>9.3f}"


def _fmt_size(row: CodeRow) -> str:
    atoms = row.n_atoms if row.n_atoms is not None else "?"
    bf = row.n_basis_functions if row.n_basis_functions is not None else "?"
    return f"{atoms}/{bf}"


def _verdict_lines(rows: Iterable[CodeRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def render_summary(
    *, run_id: str, env: dict, cases: List[CaseRecord], out_path: Path,
    csv_path: Path,
) -> None:
    """Write the markdown summary to ``out_path``.

    Sections (in order):
      1. Run metadata
      2. Headline counts
      3. Per-case verdict table
      4. Failure dossiers (one per fail / marginal / error case)
      5. Performance ranking
      6. Action items (auto-generated)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[CodeRow] = [r for c in cases for r in c.rows]

    lines: List[str] = []
    a = lines.append

    a(f"# vibe-qc regression run `{run_id}`")
    a("")

    # ---- 1. Run metadata ----------------------------------------------
    a("## Run metadata")
    a("")
    a(f"- **vibe-qc**: `{env.get('vibeqc_version', '?')}` "
      f"(branch `{env.get('git_branch', '?')}`, "
      f"sha `{env.get('git_sha', '?')}`)")
    a(f"- **python**: `{env.get('python', '?')}`  "
      f"`{env.get('python_executable', '?')}`")
    a(f"- **pyscf**: `{env.get('pyscf_version', 'n/a')}`   "
      f"**ase**: `{env.get('ase_version', 'n/a')}`   "
      f"**orca**: `{env.get('orca_command', 'n/a')}`")
    a(f"- **machine**: `{env.get('platform', '?')}` "
      f"({env.get('hostname', '?')})")
    a(f"- **CSV**: [`{csv_path.name}`]({csv_path.name})")
    a("")

    # ---- 2. Headline counts -------------------------------------------
    a("## Headline")
    a("")
    counts = _verdict_lines(r for r in all_rows if r.status != "reference")
    if counts:
        chips = " · ".join(
            f"{_STATUS_ICON.get(k, '?')} **{k}**: {v}"
            for k, v in sorted(counts.items())
        )
        a(chips)
    else:
        a("(no non-reference rows)")
    a("")

    # ---- 3. Per-case verdict table ------------------------------------
    a("## Per-case results")
    a("")
    a("| system | basis | method | kmesh | code | atoms/bf | E (Ha) | "
      "ΔE (Ha) | Δ (mHa) | abs Δ/atom | n_iter | wall (s) | severity | status |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in cases:
        for r in case.rows:
            icon = _STATUS_ICON.get(r.status, "?")
            e_str = _fmt_e(r.energy_ha) if r.energy_ha is not None else " " * 14
            d_str = (
                _fmt_d(r.delta_ha_vs_ref)
                if r.delta_ha_vs_ref is not None else " " * 11
            )
            mha_str = _fmt_mha(r.delta_mha_vs_ref, signed=True).strip()
            per_atom = _fmt_mha(r.abs_delta_mha_per_atom_vs_ref).strip()
            n_str = str(r.n_iter) if r.n_iter is not None else "—"
            sev = r.severity or "—"
            a(f"| {r.system_id} | {r.basis} | {r.method_id} | {r.kmesh} | "
              f"{r.code} | {_fmt_size(r)} | `{e_str.strip()}` | "
              f"`{d_str.strip()}` | `{mha_str}` | `{per_atom}` | "
              f"{n_str} | {r.wall_s:.1f} | {sev} | "
              f"{icon} {r.status} |")
    a("")

    # ---- 3b. Size-normalized accuracy ranking -------------------------
    comparable = [
        r for r in all_rows
        if r.status != "reference" and r.abs_delta_mha_vs_ref is not None
    ]
    if comparable:
        a("## Accuracy ranking")
        a("")
        a("| system | basis | method | code | status | abs Δ (mHa) | "
          "abs Δ/atom | abs Δ/electron | abs Δ/bf | atoms | electrons | bf | ref |")
        a("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(
            comparable,
            key=lambda row: row.abs_delta_mha_vs_ref or 0.0,
            reverse=True,
        )[:30]:
            a(f"| {r.system_id} | {r.basis} | {r.method_id} | {r.code} | "
              f"{_STATUS_ICON.get(r.status, '?')} {r.status} | "
              f"{_fmt_mha(r.abs_delta_mha_vs_ref).strip()} | "
              f"{_fmt_mha(r.abs_delta_mha_per_atom_vs_ref).strip()} | "
              f"{_fmt_mha(r.abs_delta_mha_per_electron_vs_ref).strip()} | "
              f"{_fmt_mha(r.abs_delta_mha_per_basis_function_vs_ref).strip()} | "
              f"{r.n_atoms if r.n_atoms is not None else '—'} | "
              f"{r.n_electrons if r.n_electrons is not None else '—'} | "
              f"{r.n_basis_functions if r.n_basis_functions is not None else '—'} | "
              f"{r.ref_code or '—'} |")
        a("")

    # ---- 4. Failure dossiers -----------------------------------------
    failing = [
        (c, r) for c in cases for r in c.rows
        if r.status in ("fail", "marginal", "error")
    ]
    if failing:
        a("## Failure dossiers")
        a("")
        for case, r in failing:
            log_link = (
                f"[`{case.verbose_log_path.name}`](verbose/{case.verbose_log_path.name})"
                if case.verbose_log_path is not None else "—"
            )
            a(f"### {_STATUS_ICON.get(r.status, '?')} "
              f"{r.system_id} / {r.basis} / {r.method_id} / {r.code}")
            a("")
            a(f"- **status**: `{r.status}`")
            a(f"- **note**: {r.note or '—'}")
            if r.energy_ha is not None:
                a(f"- **E**: `{r.energy_ha:.10f}` Ha")
            if r.delta_ha_vs_ref is not None:
                a(f"- **Δ vs `{r.ref_code}`**: `{r.delta_ha_vs_ref:+.6e}` Ha")
            if r.abs_delta_mha_vs_ref is not None:
                norm_bits = [f"{r.abs_delta_mha_vs_ref:.3f} mHa"]
                if r.abs_delta_mha_per_atom_vs_ref is not None:
                    norm_bits.append(
                        f"{r.abs_delta_mha_per_atom_vs_ref:.3f} mHa/atom"
                    )
                if r.abs_delta_mha_per_electron_vs_ref is not None:
                    norm_bits.append(
                        f"{r.abs_delta_mha_per_electron_vs_ref:.3f} mHa/electron"
                    )
                if r.abs_delta_mha_per_basis_function_vs_ref is not None:
                    norm_bits.append(
                        f"{r.abs_delta_mha_per_basis_function_vs_ref:.3f} mHa/bf"
                    )
                a(f"- **normalized Δ**: `{', '.join(norm_bits)}`")
            if r.severity:
                a(f"- **eigs preflight severity**: `{r.severity}` "
                  f"(min S eig = `{r.min_eigval_S}`)")
            a(f"- **verbose log**: {log_link}")
            a("")
    else:
        a("## Failure dossiers")
        a("")
        a("(none)")
        a("")

    # ---- 4b. Availability / skip report -------------------------------
    unavailable = [
        (c, r) for c in cases for r in c.rows
        if r.status == "unavailable"
    ]
    if unavailable:
        a("## Skipped / unavailable rows")
        a("")
        a("| system | basis | method | code | note |")
        a("|---|---|---|---|---|")
        for _case, r in unavailable:
            note = (r.note or "unavailable").replace("|", "\\|")
            a(f"| {r.system_id} | {r.basis} | {r.method_id} | {r.code} | {note} |")
        a("")

    # ---- 5. Performance ranking --------------------------------------
    a("## Performance (slowest first)")
    a("")
    a("| system | basis | method | code | wall (s) | n_iter | s/iter |")
    a("|---|---|---|---|---|---|---|")
    perf_rows = sorted(
        [r for r in all_rows if r.wall_s > 0],
        key=lambda r: r.wall_s, reverse=True,
    )
    for r in perf_rows[:30]:
        s_per = (
            f"{r.wall_s / r.n_iter:.2f}" if (r.n_iter and r.n_iter > 0) else "—"
        )
        a(f"| {r.system_id} | {r.basis} | {r.method_id} | "
          f"{r.code} | {r.wall_s:.1f} | "
          f"{r.n_iter if r.n_iter is not None else '—'} | {s_per} |")
    a("")

    # Performance ratio vibe-qc / pyscf per case ------------------------
    ratio_lines: List[str] = []
    for case in cases:
        v = next((r for r in case.rows if r.code == "vibeqc"), None)
        p = next((r for r in case.rows if r.code == "pyscf"), None)
        if v and p and v.wall_s > 0 and p.wall_s > 0:
            ratio = v.wall_s / p.wall_s
            ratio_lines.append(
                f"| {v.system_id} | {v.basis} | {v.method_id} | "
                f"{v.wall_s:.1f} | {p.wall_s:.1f} | {ratio:.2f}× |"
            )
    if ratio_lines:
        a("### vibe-qc vs pyscf wall ratio")
        a("")
        a("| system | basis | method | vibeqc (s) | pyscf (s) | ratio |")
        a("|---|---|---|---|---|---|")
        for line in ratio_lines:
            a(line)
        a("")

    # ---- 5b. Experimental references (NIST CCCBDB) -------------------
    # Phase 2 — when the run was invoked with
    # ``--include-experimental-reference cccbdb``, each molecular case
    # carries an ``ExperimentalReference``. Surface the experimental
    # values with full NIST-DOI citation for direct comparison against
    # the computed-energy table above.
    refs_present = [c for c in cases if getattr(c, "experimental_reference", None) is not None]
    if refs_present:
        a("## Experimental references")
        a("")
        a("From NIST CCCBDB (Standard Reference Database 101). "
          "Citation: doi:10.18434/T47C7Z. Each row's units are NIST's "
          "tabulated units; the report writer does not project them "
          "into Hartree (atomization energy is D₀, ZPE-included — "
          "add ZPE = 0.5·Σν to compare against an electronic-only "
          "computed value).")
        a("")
        a("| system | formula | AE D₀ (kcal/mol) | IE (eV) | μ (D) | "
          "α (a.u.) | vib fundamentals (cm⁻¹) | NIST source |")
        a("|---|---|---|---|---|---|---|---|")
        for case in refs_present:
            ref = case.experimental_reference
            ae_str = (
                f"{ref.atomization_energy_kcal_per_mol:.2f}"
                if ref.atomization_energy_kcal_per_mol is not None else "—"
            )
            ie_str = (
                f"{ref.ionization_energy_ev:.3f}"
                if ref.ionization_energy_ev is not None else "—"
            )
            dip_str = (
                f"{ref.dipole_moment_debye:.3f}"
                if ref.dipole_moment_debye is not None else "—"
            )
            alpha_str = (
                f"{ref.polarizability_au:.3f}"
                if ref.polarizability_au is not None else "—"
            )
            vibs = (
                ", ".join(f"{v:.0f}" for v in ref.vibrational_fundamentals_cm_inv)
                if ref.vibrational_fundamentals_cm_inv else "—"
            )
            src = (
                f"[{ref.provenance.source_db} {ref.provenance.source_id}]"
                f"({ref.provenance.source_url})"
            ) if ref.provenance else "—"
            a(f"| {case.system_id} | {ref.formula} | {ae_str} | "
              f"{ie_str} | {dip_str} | {alpha_str} | {vibs} | {src} |")
        a("")
        a("> NIST CCCBDB Release 22 (May 2022), R. D. Johnson III, ed. "
          "Standard Reference Database 101, "
          "[doi:10.18434/T47C7Z](https://doi.org/10.18434/T47C7Z).")
        a("")

    # ---- 6. Action items ---------------------------------------------
    a("## Action items (auto-generated)")
    a("")
    actions = _action_items(cases)
    if actions:
        for action in actions:
            a(f"- {action}")
    else:
        a("(no action items — all cases pass within tolerance)")
    a("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _action_items(cases: List[CaseRecord]) -> List[str]:
    actions: List[str] = []

    def _delta_label(row: CodeRow) -> str:
        if row.delta_ha_vs_ref is None:
            return "unavailable"
        bits = [f"{row.delta_ha_vs_ref:+.3e} Ha"]
        if row.delta_mha_vs_ref is not None:
            bits.append(f"{row.delta_mha_vs_ref:+.3f} mHa")
        if row.abs_delta_mha_per_atom_vs_ref is not None:
            bits.append(f"{row.abs_delta_mha_per_atom_vs_ref:.3f} mHa/atom")
        return ", ".join(bits)

    for case in cases:
        v = next((r for r in case.rows if r.code == "vibeqc"), None)
        p = next((r for r in case.rows if r.code == "pyscf"), None)
        if v is None:
            continue
        tag = f"[{v.system_id} / {v.basis} / {v.method_id}]"
        if v.status == "fail":
            actions.append(
                f"**BUG** {tag}: vibeqc Δ vs ref = "
                f"`{_delta_label(v)}` — {v.note}"
            )
        elif v.status == "marginal":
            actions.append(
                f"**INVESTIGATE** {tag}: vibeqc Δ marginal "
                f"(`{_delta_label(v)}`) — verify whether this is "
                f"basis/grid noise or a real drift"
            )
        elif v.status == "error":
            actions.append(
                f"**ERROR** {tag}: vibeqc did not produce a comparable result "
                f"— {v.note}"
            )
        if v.severity in ("error", "critical"):
            actions.append(
                f"**LINEAR-DEP** {tag}: eigs_preflight reports "
                f"`{v.severity}` — consider `make_basis(..., exp_to_discard=0.1)` "
                f"or a tighter cutoff"
            )
        if (v and p
                and v.wall_s > 0 and p.wall_s > 0
                and v.wall_s / p.wall_s > 5.0):
            actions.append(
                f"**PERF** {tag}: vibeqc {v.wall_s:.1f}s vs pyscf "
                f"{p.wall_s:.1f}s ({v.wall_s / p.wall_s:.1f}× slower)"
            )
    return actions
