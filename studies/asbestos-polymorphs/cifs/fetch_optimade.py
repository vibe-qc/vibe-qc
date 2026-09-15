"""One-shot OPTIMADE fetcher for the asbestos-polymorphs CIFs.

This is a STOP-GAP. The proper vibe-qc OPTIMADE fetcher is being
worked on in a separate chat (v0.8 deliverable). Once that lands,
delete this script.

Strategy: query Materials Project's OPTIMADE endpoint by reduced
formula only (the simpler filter that's most widely supported), shortlist
matches, materialise each as an ASE Atoms via OPTIMADE's lattice +
sites attributes, and write CIF.

Reachable today (probed at run time):
  * MP   — has many ICSD-derived entries; works without auth for
           read-only OPTIMADE queries.
  * AFLOW — DFT-only; unlikely to have natural mineral structures.
  * COD  — qiserver mirror is 500-erroring today; main host
           unreachable from this sandbox. AMCSD has these directly
           but doesn't have an OPTIMADE endpoint.

Run from the repo root:
    python3 studies/asbestos-polymorphs/cifs/fetch_optimade.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
TIMEOUT = 12

PROVIDERS = [
    ("mp",    "https://optimade.materialsproject.org/v1"),
    # COD endpoints commented out — qiserver returns 500 today, main
    # host is unreachable from many networks. Uncomment when the
    # qiserver mirror is back up.
    # ("cod_qiserver", "https://qiserver.ugr.es/cod/optimade/v1"),
    # ("cod_main",     "https://www.crystallography.net/cod/optimade/v1"),
]

# (output_filename, reduced_formula, mineral_label, expected SG)
MINERALS = [
    ("lizardite-1T",       "H4Mg3O9Si2",     "lizardite",     "P-3 1 m"),
    ("chrysotile-clino",   "H4Mg3O9Si2",     "chrysotile",    "Cc"),
    ("antigorite-m17",     "H4Mg3O9Si2",     "antigorite",    "Pm"),
    ("tremolite",          "Ca2H2Mg5O24Si8", "tremolite",     "C2/m"),
    ("anthophyllite-Mg",   "H2Mg7O24Si8",    "anthophyllite", "Pnma"),
    ("riebeckite",         "Fe5H2Na2O24Si8", "riebeckite",    "C2/m"),
    ("grunerite",          "Fe7H2O24Si8",    "grunerite",     "C2/m"),
]


_HEADERS = {
    "Accept": "application/json",
    # Materials Project (and some other providers) block default
    # Python-urllib/* User-Agent strings with 403. Identify ourselves
    # as a polite OPTIMADE client so they let us through.
    "User-Agent": "vibeqc-asbestos-fetcher/0.1 (one-shot; OPTIMADE)",
}


def http_get_json(url: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"    ! GET failed: {type(exc).__name__}: {exc}")
        return None


def query_provider(base: str, formula: str, page_limit: int = 5) -> list[dict]:
    """Simple formula-only filter — most widely supported."""
    flt = f'chemical_formula_reduced="{formula}"'
    qs = urllib.parse.urlencode({
        "filter": flt,
        "page_limit": page_limit,
    })
    url = f"{base}/structures?{qs}"
    print(f"  GET {url}")
    payload = http_get_json(url)
    if payload is None:
        return []
    return payload.get("data", []) or []


def optimade_entry_to_atoms(entry: dict):
    """Convert an OPTIMADE structure entry to an ASE Atoms object.

    Uses ``attributes.lattice_vectors`` (3×3 in Å) +
    ``attributes.cartesian_site_positions`` + ``species_at_sites``.
    """
    try:
        from ase import Atoms
    except ImportError as exc:
        raise RuntimeError("Requires ASE: pip install ase") from exc

    a = entry.get("attributes", {})
    lattice = a.get("lattice_vectors")
    positions = a.get("cartesian_site_positions")
    species_at_sites = a.get("species_at_sites") or []
    species_defs = a.get("species") or []
    name_to_chem = {
        s.get("name"): (s.get("chemical_symbols") or [None])[0]
        for s in species_defs
    }
    symbols = [name_to_chem.get(name, name) for name in species_at_sites]
    if (lattice is None or positions is None or
            any(s is None for s in symbols)):
        raise ValueError(
            f"entry {entry.get('id')!r} missing required OPTIMADE attrs"
        )
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)
    return atoms


def write_cif(atoms, out_path: Path) -> None:
    from ase.io import write as ase_write
    ase_write(str(out_path), atoms, format="cif")


def _spacegroup_of(atoms) -> tuple[str, int]:
    """Return (HM symbol, Hall number) for an ASE Atoms via spglib.
    Falls back to ('?', 0) if spglib isn't available."""
    try:
        import spglib  # noqa: F401
    except ImportError:
        return ("?", 0)
    try:
        from ase.spacegroup.symmetrize import check_symmetry
        sym = check_symmetry(atoms, symprec=1e-3, verbose=False)
        return (sym.international, sym.number)
    except Exception:
        # Fallback: ASE's deprecated get_spacegroup is still informative
        from ase.spacegroup import get_spacegroup
        sg = get_spacegroup(atoms)
        return (sg.symbol.strip(), sg.no)


def _sg_matches(found_sym: str, found_no: int, expected: str) -> bool:
    """Lax space-group match: number first, then HM-symbol substring."""
    expected = (expected or "").strip()
    if expected == "":
        return True
    # Map common HM symbols to numbers for the asbestos minerals.
    EXPECTED_NUMBERS = {
        "P-3 1 m": 162,
        "Cc": 9,
        "Pm": 6,
        "C2/m": 12,
        "Pnma": 62,
    }
    want_no = EXPECTED_NUMBERS.get(expected)
    if want_no is not None and found_no == want_no:
        return True
    # Symbol-based fallback (case-insensitive substring).
    found = (found_sym or "").lower().replace(" ", "")
    want  = expected.lower().replace(" ", "")
    return want in found


def try_providers_for(mineral) -> Optional[Path]:
    out_name, formula, label, sg_expected = mineral
    out_path = HERE / f"{out_name}.cif"
    if out_path.exists():
        print(f"  already present: {out_path.name}")
        return out_path
    for provider, base in PROVIDERS:
        entries = query_provider(base, formula)
        if not entries:
            print(f"  [{provider}] 0 candidate(s)")
            continue
        print(f"  [{provider}] {len(entries)} candidate(s)")
        # Materialise + space-group-verify each. Accept first SG match.
        for entry in entries:
            tag = entry.get("id", "?")
            n_sites = (entry.get("attributes") or {}).get("nsites") or "?"
            try:
                atoms = optimade_entry_to_atoms(entry)
            except Exception as exc:
                print(f"    {tag}: cannot materialise — {type(exc).__name__}: {exc}")
                continue
            sg_sym, sg_no = _spacegroup_of(atoms)
            ok = _sg_matches(sg_sym, sg_no, sg_expected)
            verdict = "✓" if ok else "✗"
            print(f"    {verdict} {tag}: nsites={n_sites} "
                  f"sg={sg_sym} (#{sg_no})  want={sg_expected}")
            if not ok:
                continue
            write_cif(atoms, out_path)
            with out_path.open("a") as f:
                f.write(
                    f"\n# OPTIMADE provenance: provider={provider}, "
                    f"id={tag}, formula={formula}, "
                    f"expected_mineral={label}, "
                    f"sg_found={sg_sym} (#{sg_no})\n"
                )
            print(f"  ✓ wrote {out_path.name} from {tag}")
            return out_path
        print(f"  [{provider}] no SG-matching candidate")
    print(f"  ✗ {out_name}: no provider yielded a SG-matching CIF — "
          f"need AMCSD download (see CIFS_NEEDED.md)")
    return None


def main():
    print(f"Fetching CIFs into {HERE}")
    print()
    results = []
    for mineral in MINERALS:
        out_name = mineral[0]
        print(f">>> {out_name}  ({mineral[2]}, {mineral[1]}, expect {mineral[3]})")
        path = try_providers_for(mineral)
        results.append((out_name, path))
        time.sleep(0.5)  # be polite
        print()

    print("=" * 50)
    print("Summary")
    print("=" * 50)
    n_ok = 0
    for name, path in results:
        flag = "✓" if path else "✗"
        if path: n_ok += 1
        print(f"  {flag}  {name}")
    print(f"  {n_ok}/{len(results)} fetched")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
