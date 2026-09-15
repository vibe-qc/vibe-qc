"""Assemble the self-contained ATOMBSSE sweep payload for a vq host.

    .venv/bin/python examples/basisset_dev/bsse_sweep/build_payload.py /tmp/payload
    vq submit compute-reference -d /tmp/payload --cpus 8 -- python run_sweep.py

Restricted to named atom sites (a re-run of the ones a previous sweep
could not settle):

    ... build_payload.py /tmp/payload --sites 'Si|1,Ge|1,LiF|1'
    ... build_payload.py /tmp/payload --sites @sites.txt

The payload ships its own ``vibe_basis`` source and pre-generated inline
basis blocks, so the compute host needs only a CRYSTAL binary and
stdlib Python: no installed vibe-basis, no vibe-qc, no native build.
That is deliberate. The alternative, relying on whatever is deployed on
the host, couples a long sweep to a fleet version that may roll under
it (CLAUDE.md section 15), and the basis text is exactly the input whose
provenance the run must pin.

**Rebuild the payload rather than resubmitting an old one.** The decks it
carries are emitted by the ``vibe_basis`` it copies in, so an emitter fix
(the 0.9.0 ``ORIGIN``/``TRASREMO`` one, say) or a protocol default change
(0.8.0's ``TOLINTEG 9 9 9 18 54`` and dropped ``HUGEGRID``) reaches the
sweep only through a fresh build.

Systems are selected by what can actually be emitted: a space group and
asymmetric unit, no AFM ordering, no ECP, a pob-TZVP-REV2 source for
every element, and a tabulated free-atom ground state. The last one
currently excludes the 3d transition metals, because
``emit_input_atom_counterpoise`` refuses an element whose multiplicity
it would have to guess.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vibe-basis" / "src"))
sys.path.insert(0, str(ROOT / "python"))

from vibe_basis.backends.crystal_atom import GROUND_STATE_MULTIPLICITY
from vibe_basis.io.structures import STRUCTURES
from vibeqc.basis_crystal import emit_crystal, parse_crystal_atom_basis_file

BASIS_SRC = ROOT / "python/vibeqc/basis_library/sources/pob-TZVP-rev2"


def parse_sites(spec: str) -> dict[str, set[int]]:
    """``"Si|1,LiF|1"`` or ``"@file"`` -> ``{"Si": {1}, "LiF": {1}}``."""
    if spec.startswith("@"):
        spec = Path(spec[1:]).read_text()
    wanted: dict[str, set[int]] = {}
    for token in spec.replace("\n", ",").split(","):
        token = token.strip()
        if not token or token.startswith("#"):
            continue
        system, _, site = token.partition("|")
        if not site.isdigit():
            raise SystemExit(f"bad site spec {token!r}; want 'System|index'")
        wanted.setdefault(system.strip(), set()).add(int(site))
    if not wanted:
        raise SystemExit("--sites selected nothing")
    return wanted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", nargs="?", default="payload")
    ap.add_argument(
        "--sites",
        help="restrict to 'System|site' entries (comma/newline separated, or "
             "@path). Omit for every emittable system and site.",
    )
    args = ap.parse_args()

    # A site restriction rides in the payload as sites.json; run_sweep.py
    # honours it. Keeping it in the payload rather than in an env var means
    # the submitted artefact records exactly which sites the job was for,
    # which is what a resumed or re-fetched run has to be read against.
    wanted = parse_sites(args.sites) if args.sites else None

    out = Path(args.output).resolve()
    (out / "bases").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).parent / "run_sweep.py", out / "run_sweep.py")

    src = out / "src"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(ROOT / "vibe-basis/src/vibe_basis", src / "vibe_basis",
                    ignore=shutil.ignore_patterns("__pycache__"))

    sources = {int(p.name.split("_")[0]): p
               for p in BASIS_SRC.iterdir() if p.name[:2].isdigit()}
    chosen, skipped = [], []
    for name, s in STRUCTURES.items():
        if wanted is not None and name not in wanted:
            continue
        if s.crystal_spacegroup == 0 or not s.crystal_asymm_unit:
            skipped.append((name, "no space group / asymmetric unit")); continue
        if getattr(s, "afm_pattern", None):
            skipped.append((name, "AFM ordering")); continue
        if s.ecp_library:
            skipped.append((name, "needs an ECP")); continue
        Zs = sorted({a.Z for a in s.crystal_asymm_unit})
        if set(Zs) - set(sources):
            skipped.append((name, f"no pob source for {sorted(set(Zs)-set(sources))}")); continue
        if set(Zs) - set(GROUND_STATE_MULTIPLICITY):
            skipped.append((name, f"no tabulated ground state for "
                                  f"{sorted(set(Zs)-set(GROUND_STATE_MULTIPLICITY))}")); continue
        (out / "bases" / f"{name}.txt").write_text(
            emit_crystal([parse_crystal_atom_basis_file(str(sources[z])) for z in Zs]))
        chosen.append(name)

    (out / "systems.json").write_text(json.dumps(sorted(chosen), indent=1))

    sites_file = out / "sites.json"
    if wanted is None:
        sites_file.unlink(missing_ok=True)
        n_sites = sum(len(STRUCTURES[n].crystal_asymm_unit) for n in chosen)
    else:
        missing = sorted(set(wanted) - set(chosen))
        if missing:
            raise SystemExit(f"--sites named systems the payload cannot emit: {missing}")
        selected = []
        for name in sorted(chosen):
            n_asym = len(STRUCTURES[name].crystal_asymm_unit)
            for site in sorted(wanted[name]):
                if not 1 <= site <= n_asym:
                    raise SystemExit(
                        f"{name}|{site}: asymmetric unit has {n_asym} site(s)")
                selected.append(f"{name}|{site}")
        sites_file.write_text(json.dumps(selected, indent=1))
        n_sites = len(selected)

    print(f"payload -> {out}")
    print(f"  {len(chosen)} systems, {n_sites} atom sites"
          + ("" if wanted is None else " (restricted by --sites)"))
    for name, why in sorted(skipped):
        print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    main()
