"""``vqfetch`` console-script entry point.

CLI surface (v1):

    vqfetch optimade --formula MgO [--provider mp]
                     [--basis pob-tzvp] [--quick]
                     [--out examples/regression/systems/periodic/]
                     [--input-script examples/]

    vqfetch cod --id 1011027 [...same flags...]

    vqfetch mp --id mp-1265 [...same flags...]

    vqfetch canonical <slug> [--quick] [...same out flags...]

Every successful invocation prints exactly two paths to stdout, one
per line: the SPEC file then the input script. Errors -> stderr,
non-zero exit. No interactive prompts.

See ``docs/tutorial/external_data_fetcher.md`` Sec. 8.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from examples.regression.core.spec import MoleculeSpec, PeriodicSpec, TestSystem

from . import __version__ as FETCHER_VERSION
from .cache import FetchCache, FetchCacheMiss
from .canonical_set import find as find_canonical
from .client_cod import fetch_cod
from .client_mp import fetch_mp
from .client_optimade import fetch_optimade
from .emit_input import emit_input_script
from .consistency import _cmd_consistency
from .emit_spec import emit_spec_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PERIODIC_SYSTEMS_DIR = (
    _REPO_ROOT / "examples" / "regression" / "systems" / "periodic"
)
DEFAULT_MOLECULE_SYSTEMS_DIR = (
    _REPO_ROOT / "examples" / "regression" / "systems" / "molecules"
)
DEFAULT_INPUT_SCRIPT_DIR = _REPO_ROOT / "examples"


# ---- Shared option wiring --------------------------------------------------


def _add_common_emit_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--basis",
        default=None,
        help="Override recommended basis. Default: heuristic (Sec. 6.3).",
    )
    p.add_argument(
        "--method",
        default="rks-lda",
        help="SCF method for the emitted input script "
        "(rhf | rks-lda | rks-pbe | rks-blyp | rks-b3lyp). "
        "Periodic SPECs default to rks-lda; molecules to rhf.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode -> recommended_basis = sto-3g.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory for the emitted SPEC module. "
        "Default: examples/regression/systems/{periodic,molecules}/.",
    )
    p.add_argument(
        "--input-script",
        default=str(DEFAULT_INPUT_SCRIPT_DIR),
        help="Output directory for the emitted input script. Default: examples/.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache reads (still writes through after a live fetch).",
    )
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Refuse live HTTP -- only read from the on-disk cache. "
        "Useful for offline / reproducible runs.",
    )
    p.add_argument(
        "--slug",
        default=None,
        help="Override the auto-generated SPEC id slug.",
    )


def _resolve_out_dir(spec, override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    return (
        DEFAULT_PERIODIC_SYSTEMS_DIR
        if isinstance(spec, PeriodicSpec)
        else DEFAULT_MOLECULE_SYSTEMS_DIR
    )


def _resolve_method_for_spec(spec, requested: str) -> str:
    """Periodic SPECs honour `--method`; molecules default to rhf."""
    if isinstance(spec, MoleculeSpec) and requested == "rks-lda":
        # If the user didn't override, molecules default to rhf
        # (sto-3g RKS-LDA on a molecule is unusual; rhf reads more
        # naturally on the emitted input).
        return "rhf"
    return requested


def _resolve_basis(spec, override: Optional[str]) -> str:
    if override:
        return override
    if spec.recommended_basis:
        return spec.recommended_basis
    return "sto-3g"


def _emit_pair(
    spec: TestSystem,
    *,
    out_override: Optional[str],
    input_script_dir: str,
    basis_override: Optional[str],
    method_override: str,
) -> tuple[Path, Path]:
    out_dir = _resolve_out_dir(spec, out_override)
    method = _resolve_method_for_spec(spec, method_override)
    basis = _resolve_basis(spec, basis_override)
    spec_path = emit_spec_module(spec, out_dir)
    input_path = emit_input_script(
        spec,
        Path(input_script_dir),
        basis=basis,
        method=method,
    )
    return spec_path, input_path


def _print_paths_and_exit(spec_path: Path, input_path: Path) -> int:
    print(spec_path)
    print(input_path)
    return 0


# ---- Subcommand dispatchers ------------------------------------------------


def _cmd_optimade(args: argparse.Namespace) -> int:
    cache = FetchCache()

    # Bulk mode: read formula list, fetch each.
    if args.bulk:
        bulk_path = Path(args.bulk).expanduser()
        if not bulk_path.is_file():
            print(f"vqfetch: --bulk file not found: {bulk_path}", file=sys.stderr)
            return 1
        formulas = [
            line.strip() for line in bulk_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        ok = 0
        for formula in formulas:
            try:
                results = fetch_optimade(
                    formula=formula, provider=args.provider, cache=cache,
                    use_cache=not args.no_cache, cache_only=args.cache_only,
                    quick=args.quick, slug_override=args.slug,
                )
            except FetchCacheMiss as exc:
                print(f"vqfetch optimade {formula}: cache miss -- {exc}", file=sys.stderr)
                continue
            if not results:
                print(f"vqfetch optimade {formula}: no results", file=sys.stderr)
                continue
            spec_path, input_path = _emit_pair(
                results[0],
                out_override=args.out,
                input_script_dir=args.input_script,
                basis_override=args.basis,
                method_override=args.method,
            )
            print(spec_path)
            print(input_path)
            ok += 1
        print(f"\n{ok}/{len(formulas)} succeeded", file=sys.stderr)
        return 0 if ok > 0 else 1

    try:
        results = fetch_optimade(
            formula=args.formula,
            provider=args.provider,
            cache=cache,
            use_cache=not args.no_cache,
            cache_only=args.cache_only,
            quick=args.quick,
            slug_override=args.slug,
        )
    except FetchCacheMiss as exc:
        print(
            f"vqfetch: cache miss -- refusing to fetch live (--cache-only): {exc}",
            file=sys.stderr,
        )
        return 2
    if not results:
        print(
            f"vqfetch optimade: no results for formula={args.formula!r}",
            file=sys.stderr,
        )
        return 1
    spec_path, input_path = _emit_pair(
        results[0],
        out_override=args.out,
        input_script_dir=args.input_script,
        basis_override=args.basis,
        method_override=args.method,
    )
    return _print_paths_and_exit(spec_path, input_path)


def _cmd_cod(args: argparse.Namespace) -> int:
    cache = FetchCache()
    try:
        spec = fetch_cod(
            cod_id=args.id,
            cache=cache,
            use_cache=not args.no_cache,
            cache_only=args.cache_only,
            quick=args.quick,
            slug_override=args.slug,
        )
    except FetchCacheMiss as exc:
        print(
            f"vqfetch: cache miss -- refusing to fetch live (--cache-only): {exc}",
            file=sys.stderr,
        )
        return 2
    spec_path, input_path = _emit_pair(
        spec,
        out_override=args.out,
        input_script_dir=args.input_script,
        basis_override=args.basis,
        method_override=args.method,
    )
    return _print_paths_and_exit(spec_path, input_path)


def _cmd_mp(args: argparse.Namespace) -> int:
    try:
        spec = fetch_mp(
            mp_id=args.id,
            api_key=os.environ.get("MP_API_KEY"),
            quick=args.quick,
            slug_override=args.slug,
        )
    except FetchCacheMiss as exc:
        print(
            f"vqfetch: cache miss -- refusing to fetch live (--cache-only): {exc}",
            file=sys.stderr,
        )
        return 2
    spec_path, input_path = _emit_pair(
        spec,
        out_override=args.out,
        input_script_dir=args.input_script,
        basis_override=args.basis,
        method_override=args.method,
    )
    return _print_paths_and_exit(spec_path, input_path)


def _cmd_reference(args: argparse.Namespace) -> int:
    """Phase-2 reference-data fetch (CCCBDB or WebBook)."""
    import dataclasses
    import json

    cache = FetchCache()

    if args.source == "webbook":
        from .references.client_webbook import MoleculeNotFound, fetch_webbook

        fetcher = fetch_webbook
    else:
        from .references.client_cccbdb import MoleculeNotFound, fetch_cccbdb

        fetcher = fetch_cccbdb

    # Bulk mode: read CAS list, fetch each, collect results.
    if args.bulk:
        bulk_path = Path(args.bulk).expanduser()
        if not bulk_path.is_file():
            print(f"vqfetch: --bulk file not found: {bulk_path}", file=sys.stderr)
            return 1
        cas_list = [
            line.strip() for line in bulk_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        out_dir = (
            Path(args.out).expanduser()
            if args.out
            else (_REPO_ROOT / "examples" / "regression" / "references")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for cas in cas_list:
            try:
                ref = fetcher(cas=cas, cache=cache, use_cache=not args.no_cache, cache_only=args.cache_only)
            except (FetchCacheMiss, MoleculeNotFound) as exc:
                print(f"vqfetch reference {cas}: {exc}", file=sys.stderr)
                continue
            out_path = out_dir / f"{cas}.json"
            payload = dataclasses.asdict(ref)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(out_path)
            ok += 1
        print(f"\n{ok}/{len(cas_list)} succeeded", file=sys.stderr)
        return 0 if ok > 0 else 1

    try:
        ref = fetcher(
            cas=args.cas,
            cache=cache,
            use_cache=not args.no_cache,
            cache_only=args.cache_only,
        )
    except FetchCacheMiss as exc:
        print(
            f"vqfetch: cache miss -- refusing to fetch live (--cache-only): {exc}",
            file=sys.stderr,
        )
        return 2
    except MoleculeNotFound as exc:
        print(f"vqfetch reference: {exc}", file=sys.stderr)
        return 1

    out_dir = (
        Path(args.out).expanduser()
        if args.out
        else (_REPO_ROOT / "examples" / "regression" / "references")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.cas}.json"
    # Convert dataclass -> JSON-friendly dict (Provenance flattens cleanly).
    payload = dataclasses.asdict(ref)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    return 0


def _cmd_canonical(args: argparse.Namespace) -> int:
    """Resolve a canonical-set slug via id-lookup (unambiguous)."""
    entry = find_canonical(args.slug)
    spec = fetch_optimade(
        optimade_id=f"{entry.primary_provider}/{entry.primary_id}",
        use_cache=not args.no_cache,
        cache_only=args.cache_only,
        quick=args.quick,
        slug_override=entry.slug,
    )[0]
    # Family override: id-lookup keeps the family heuristic from
    # _attrs_to_periodic_spec, which is formula-keyed and may not match
    # the canonical entry's curated family field. Force it here.
    if spec.family != entry.family:
        from dataclasses import replace

        spec = replace(spec, family=entry.family)
    spec_path, input_path = _emit_pair(
        spec,
        out_override=args.out,
        input_script_dir=args.input_script,
        basis_override=args.basis,
        method_override=args.method,
    )
    return _print_paths_and_exit(spec_path, input_path)


# ---- list-candidates rendering ---------------------------------------------


def _render_candidates_table(specs: list) -> str:
    """Render a list of PeriodicSpec candidates as a tabular string.

    Columns: provider, ID, space group, lattice constant (a, Å),
    atom count, dedup-merge count.
    """

    def _dedup_merge_count(spec) -> int:
        p = getattr(spec, "provenance", None)
        if p is None:
            return 0
        notes = getattr(p, "notes", "") or ""
        if "also at:" not in notes:
            return 0
        tail = notes.split("also at:", 1)[1].strip()
        if not tail:
            return 0
        return tail.count(",") + 1

    header = f"{'provider':<12} {'id':<16} {'sg':<12} {'a (Å)':>10} {'atoms':>6} {'merged':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    for s in specs:
        provider = (
            (s.provenance.source_db or "?").removeprefix("OPTIMADE/")
            if s.provenance
            else "?"
        )
        sid = (s.provenance.source_id or "?") if s.provenance else "?"
        sg = s.space_group or "?"
        a_ang = s.lattice_ang[0][0]
        n_atoms = len(s.atoms)
        n_merged = _dedup_merge_count(s)
        lines.append(
            f"{provider:<12} {sid:<16} {sg:<12} {a_ang:10.4f} {n_atoms:>6} {n_merged:>7}"
        )
    return "\n".join(lines)


def _cmd_list_candidates(args: argparse.Namespace) -> int:
    """vqfetch list-candidates -- multi-candidate query + tabular report."""
    cache = FetchCache()
    try:
        specs = fetch_optimade(
            formula=args.formula,
            provider=args.provider,
            cache=cache,
            use_cache=not args.no_cache,
            cache_only=args.cache_only,
            max_results=args.max_results,
            dedup=not args.no_dedup,
        )
    except FetchCacheMiss as exc:
        print(
            f"vqfetch: cache miss -- refusing to fetch live (--cache-only): {exc}",
            file=sys.stderr,
        )
        return 2
    if not specs:
        print(
            f"vqfetch list-candidates: no results for formula={args.formula!r}",
            file=sys.stderr,
        )
        return 1
    print(_render_candidates_table(specs))
    print(f"\n{len(specs)} candidate(s) shown. Pick one:")
    print(f"  vqfetch optimade --formula {args.formula}")
    if specs[0].provenance:
        p = specs[0].provenance.source_db.removeprefix("OPTIMADE/")
        print(
            f"  # OR id-lookup:  vqfetch optimade --optimade-id {p}/{specs[0].provenance.source_id}"
        )
    return 0


# ---- Top-level argparse wiring --------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vqfetch",
        description=(
            "External structure fetcher for vibe-qc benchmarking. "
            "Pulls geometries from open databases (OPTIMADE / Materials "
            "Project / COD / NOMAD) and emits a regression SPEC plus "
            "an executable vibe-qc input script."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"vqfetch {FETCHER_VERSION}",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_opt = sub.add_parser(
        "optimade",
        help="Federated OPTIMADE query (default).",
        description="Query the OPTIMADE federation by chemical formula.",
    )
    p_opt.add_argument(
        "--formula", required=True, help='chemical_formula_reduced, e.g. "MgO".'
    )
    p_opt.add_argument(
        "--provider",
        default=None,
        help='Short key ("mp", "cod", "oqmd") or full base URL.',
    )
    p_opt.add_argument(
        "--bulk", default=None,
        help="Path to a file with one formula per line. Per-row failure "
             "tolerance -- one bad formula doesn't abort the sweep.",
    )
    _add_common_emit_flags(p_opt)
    p_opt.set_defaults(func=_cmd_optimade)

    p_cod = sub.add_parser("cod", help="COD direct CIF download.")
    p_cod.add_argument("--id", required=True, help="COD entry id, e.g. 1011027.")
    _add_common_emit_flags(p_cod)
    p_cod.set_defaults(func=_cmd_cod)

    p_mp = sub.add_parser("mp", help="Materials Project lookup.")
    p_mp.add_argument(
        "--id", required=True, help='Materials Project id, e.g. "mp-1265".'
    )
    _add_common_emit_flags(p_mp)
    p_mp.set_defaults(func=_cmd_mp)

    p_can = sub.add_parser(
        "canonical",
        help="Run a canonical-set slug through its primary source.",
    )
    p_can.add_argument("slug", help="Canonical-set slug (see canonical_set.py).")
    _add_common_emit_flags(p_can)
    p_can.set_defaults(func=_cmd_canonical)

    # ---- Phase 2: reference-data fetcher --------------------------------
    p_ref = sub.add_parser(
        "reference",
        help="Pull experimental property data from NIST CCCBDB.",
        description=(
            "Fetch experimental + computed reference values "
            "(atomization energy, enthalpies of formation, vibrational "
            "frequencies, dipole, IE) from NIST CCCBDB by CAS number. "
            "Emits a JSON ExperimentalReference record."
        ),
    )
    p_ref.add_argument(
        "--cas",
        required=True,
        help='CAS registry number, hyphenated, e.g. "7732-18-5" for water.',
    )
    p_ref.add_argument(
        "--source",
        default="cccbdb",
        choices=("cccbdb", "webbook"),
        help="Reference source: cccbdb (SRD 101, small molecules) or webbook (SRD 69, any size).",
    )
    p_ref.add_argument(
        "--out",
        default=None,
        help="Output directory for the JSON record. "
        "Default: examples/regression/references/.",
    )
    p_ref.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache reads (still writes through after a live fetch).",
    )
    p_ref.add_argument(
        "--cache-only",
        action="store_true",
        help="Refuse live HTTP -- only read from the on-disk cache.",
    )
    p_ref.add_argument(
        "--bulk", default=None,
        help="Path to a file with one CAS per line. Per-row failure "
             "tolerance -- one bad CAS doesn't abort the sweep.",
    )
    p_ref.set_defaults(func=_cmd_reference)

    # ---- Phase 1: list-candidates (v0.13.x, VFETCH-X1) ------------------
    p_lc = sub.add_parser(
        "list-candidates",
        help="Search and display candidate structures (tabular).",
        description=(
            "Query the OPTIMADE federation for multiple candidate "
            "structures, dedup by structural hash, and display a "
            "tabular report ranked by space-group plurality."
        ),
    )
    p_lc.add_argument(
        "--formula",
        required=True,
        help='chemical_formula_reduced, e.g. "MgO".',
    )
    p_lc.add_argument(
        "--provider",
        default=None,
        help='Short key ("mp", "cod", "oqmd") or full base URL.',
    )
    p_lc.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum candidates to display (default: 10).",
    )
    p_lc.add_argument(
        "--no-dedup",
        action="store_true",
        help="Show structurally-identical entries separately "
        "(default: dedup by structural hash).",
    )
    p_lc.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache reads (still writes through after a live fetch).",
    )
    p_lc.add_argument(
        "--cache-only",
        action="store_true",
        help="Refuse live HTTP -- only read from the on-disk cache.",
    )
    p_lc.set_defaults(func=_cmd_list_candidates)

    # ---- Phase 1: cross-provider consistency (v0.13.x, VFETCH-X6) ----
    p_con = sub.add_parser(
        "consistency",
        help="Cross-provider lattice-constant and space-group comparison.",
        description=(
            "Fetch the same formula from every OPTIMADE provider, "
            "compare lattice constants and space groups, and emit a "
            "Markdown report flagging disagreements."
        ),
    )
    p_con.add_argument(
        "--formula", required=True,
        help='chemical_formula_reduced, e.g. "MgO".',
    )
    p_con.add_argument(
        "--provider", default=None,
        help='Narrow to one provider for sanity-check.',
    )
    p_con.add_argument(
        "--no-cache", action="store_true",
        help="Bypass cache reads.",
    )
    p_con.add_argument(
        "--cache-only", action="store_true",
        help="Refuse live HTTP.",
    )
    p_con.add_argument(
        "--out", default=None,
        help="Write report to file instead of stdout.",
    )
    p_con.set_defaults(func=_cmd_consistency)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
