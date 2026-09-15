"""Fetch missing basis sets from the Basis Set Exchange.

Implements action item #1 of
``docs/basisset_dev/ROADMAP_BASIS_LIBRARY.md``: pulls the high-
priority all-electron bases that the 2026-05-08 review flagged
as missing from the vibe-qc bundle.

Usage::

    .venv/bin/pip install basis-set-exchange
    .venv/bin/python scripts/basisset_dev/fetch_from_bse.py --priority high
    ./scripts/setup_basis_library.sh        # promote custom/ → basis/

The fetcher writes to ``python/vibeqc/basis_library/custom/`` so
the existing setup pipeline picks them up unchanged. Each file
gets a ``! vibeqc-fetched ...`` header naming the source and date.

What this fetcher does NOT do (by design):

* It does not pull bases blocked on libecpint or relativistic
  Hamiltonian. The "blocked" list in the roadmap stays out of
  scope here.
* It does not touch ``basis/`` directly — that's the setup
  script's job.
* It does not run libint to verify each fetched basis loads —
  add a regression test (
  ``tests/basisset_dev/test_basis_library_load.py``) for that.

The set of names it knows about is a Python dict at the bottom
of this file, derived from
``docs/basisset_dev/ROADMAP_BASIS_LIBRARY.md``. Update both
together.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "python" / "vibeqc" / "basis_library"
)
CUSTOM_DIR = PACKAGE_ROOT / "custom"


# (vibe-qc canonical name → BSE basis-set identifier, priority, comment)
# Priorities: "highest", "high", "medium", "low" — match the roadmap.
# Canonical names are lowercased; libint resolves names case-insensitively.
TARGETS: list[tuple[str, str, str, str]] = [
    # ---- pcseg family (Jensen segmented polarisation-consistent) ----
    ("pcseg-0", "pcseg-0", "high", "DZ unpolarised; replaces 3-21G in DFT roles"),
    ("pcseg-1", "pcseg-1", "highest", "DFT-optimal DZ"),
    (
        "pcseg-2",
        "pcseg-2",
        "highest",
        "DFT-optimal TZ — best on diet-GMTKN55 (Pitman 2024)",
    ),
    ("pcseg-3", "pcseg-3", "medium", ""),
    ("pcseg-4", "pcseg-4", "low", ""),
    ("aug-pcseg-0", "aug-pcseg-0", "medium", ""),
    ("aug-pcseg-1", "aug-pcseg-1", "high", ""),
    ("aug-pcseg-2", "aug-pcseg-2", "high", "TDDFT, anions"),
    ("aug-pcseg-3", "aug-pcseg-3", "low", ""),
    ("aug-pcseg-4", "aug-pcseg-4", "low", ""),
    # ---- pc family (Jensen general-contraction; less common) ----
    ("pc-0", "pc-0", "medium", ""),
    ("pc-1", "pc-1", "medium", ""),
    ("pc-2", "pc-2", "medium", ""),
    ("pc-3", "pc-3", "low", ""),
    ("pc-4", "pc-4", "low", ""),
    ("aug-pc-0", "aug-pc-0", "low", ""),
    ("aug-pc-1", "aug-pc-1", "low", ""),
    ("aug-pc-2", "aug-pc-2", "medium", ""),
    ("aug-pc-3", "aug-pc-3", "low", ""),
    ("aug-pc-4", "aug-pc-4", "low", ""),
    # ---- Pople with diffuse augmentation ----
    ("6-31+g**", "6-31+G**", "high", "Pople DZ + diffuse + polarisation"),
    ("6-31++g**", "6-31++G**", "high", "best DZ basis per Pitman 2024"),
    ("6-311+g**", "6-311+G**", "medium", ""),
    ("6-311+g(2d,p)", "6-311+G(2d,p)", "low", ""),
    ("6-311++g", "6-311++G", "low", ""),
    ("6-311++g**", "6-311++G**", "medium", ""),
    # ---- Karlsruhe modified def2 (3c carriers; data only — needs gCP+D3/D4) ----
    ("def2-mtzvp", "def2-mTZVP", "high", "B97-3c carrier (with gCP+D3)"),
    ("def2-mtzvpp", "def2-mTZVPP", "high", "r2SCAN-3c carrier (with gCP+D4)"),
    # ---- Karlsruhe relativistic counterparts ----
    # (Data only — meaningful use awaits a relativistic Hamiltonian.)
    ("dhf-sv(p)", "dhf-SV(P)", "medium", "blocked on rel. Hamiltonian for use"),
    ("dhf-svp", "dhf-SVP", "medium", "same"),
    ("dhf-tzvp", "dhf-TZVP", "medium", "same"),
    ("dhf-tzvpp", "dhf-TZVPP", "medium", "same"),
    ("dhf-qzvp", "dhf-QZVP", "low", "same"),
    ("dhf-qzvpp", "dhf-QZVPP", "low", "same"),
    ("x2c-tzvpall", "x2c-TZVPall", "high", "blocked on X2C Hamiltonian for use"),
    ("x2c-tzvpall-s", "x2c-TZVPall-s", "medium", "NMR-tuned X2C"),
    ("x2c-tzvpall-2c", "x2c-TZVPall-2c", "medium", "two-component X2C"),
    # ---- Grimme vDZP (ωB97X-3c carrier; needs ECP for use) ----
    ("vdzp", "Grimme vDZP", "high", "ωB97X-3c carrier — uses ECPs (libecpint blocker)"),
    # ---- Hay-Wadt LANL ECPs (legacy; libecpint blocker for use) ----
    ("lanl2dz", "LANL2DZ", "low", "deprecated — UI label: prefer def2-TZVP+def2-ECP"),
    ("lanl2dzdp", "LANL2DZdp", "low", "same"),
    ("lanl2tz", "LANL2TZ", "low", "same"),
    ("lanl08", "LANL08", "low", "same"),
    ("lanl08(d)", "LANL08(d)", "low", "same"),
    ("lanl08(f)", "LANL08(f)", "low", "same"),
    # ---- Dunning "tight d" for hypervalent ----
    ("cc-pv(d+d)z", "cc-pV(D+d)Z", "medium", "tight-d hypervalent"),
    ("cc-pv(t+d)z", "cc-pV(T+d)Z", "medium", ""),
    ("cc-pv(q+d)z", "cc-pV(Q+d)Z", "medium", ""),
    ("cc-pv(5+d)z", "cc-pV(5+d)Z", "low", ""),
    ("aug-cc-pv(d+d)z", "aug-cc-pV(D+d)Z", "medium", ""),
    ("aug-cc-pv(t+d)z", "aug-cc-pV(T+d)Z", "medium", ""),
    ("aug-cc-pv(q+d)z", "aug-cc-pV(Q+d)Z", "medium", ""),
    ("aug-cc-pv(5+d)z", "aug-cc-pV(5+d)Z", "low", ""),
    # ---- Dunning partially-augmented (jul-/jun-, the "ma-" equivalents in BSE) ----
    ("jul-cc-pv(t+d)z", "jul-cc-pV(T+d)Z", "medium", "partial aug, smaller than aug-"),
    ("jun-cc-pv(t+d)z", "jun-cc-pV(T+d)Z", "medium", "even smaller partial aug"),
    ("jul-cc-pv(d+d)z", "jul-cc-pV(D+d)Z", "low", ""),
    ("jul-cc-pv(q+d)z", "jul-cc-pV(Q+d)Z", "low", ""),
    # ---- Dunning core-valence ----
    ("cc-pcvdz", "cc-pCVDZ", "medium", ""),
    ("cc-pcvtz", "cc-pCVTZ", "medium", ""),
    ("cc-pcvqz", "cc-pCVQZ", "medium", ""),
    ("aug-cc-pcvdz", "aug-cc-pCVDZ", "medium", ""),
    ("aug-cc-pcvtz", "aug-cc-pCVTZ", "medium", ""),
    ("aug-cc-pcvqz", "aug-cc-pCVQZ", "medium", ""),
    # ---- Dunning pseudopotential-adapted (Peterson/Figgen small-core PPs) ----
    # These carry their ECP blocks in the same BSE file; the promotion in
    # scripts/setup_basis_library.sh splits each into <name>.g94 + <name>.ecp.
    # 39 elements apiece (Cu-Kr, Y-Xe, Hf-Rn). The matching *-PP-RIFIT fitting
    # sets already ship, and covered fewer elements than these orbital sets do
    # -- see registry.toml for the per-basis default_ri records.
    ("cc-pvdz-pp", "cc-pVDZ-PP", "high", "small-core PP; heavy-element orbital set"),
    ("cc-pvtz-pp", "cc-pVTZ-PP", "high", "small-core PP; heavy-element orbital set"),
    ("cc-pvqz-pp", "cc-pVQZ-PP", "medium", "small-core PP; heavy-element orbital set"),
    ("cc-pv5z-pp", "cc-pV5Z-PP", "low", "small-core PP; heavy-element orbital set"),
    ("cc-pwcvdz-pp", "cc-pwCVDZ-PP", "medium", "PP core-valence"),
    ("cc-pwcvtz-pp", "cc-pwCVTZ-PP", "medium", "PP core-valence"),
    ("cc-pwcvqz-pp", "cc-pwCVQZ-PP", "low", "PP core-valence"),
    ("cc-pwcv5z-pp", "cc-pwCV5Z-PP", "low", "PP core-valence"),
    ("aug-cc-pvdz-pp", "aug-cc-pVDZ-PP", "high", "PP + diffuse"),
    ("aug-cc-pvtz-pp", "aug-cc-pVTZ-PP", "high", "PP + diffuse"),
    ("aug-cc-pvqz-pp", "aug-cc-pVQZ-PP", "medium", "PP + diffuse"),
    ("aug-cc-pv5z-pp", "aug-cc-pV5Z-PP", "low", "PP + diffuse"),
    ("aug-cc-pwcvdz-pp", "aug-cc-pwCVDZ-PP", "medium", "PP core-valence + diffuse"),
    ("aug-cc-pwcvtz-pp", "aug-cc-pwCVTZ-PP", "medium", "PP core-valence + diffuse"),
    ("aug-cc-pwcvqz-pp", "aug-cc-pwCVQZ-PP", "low", "PP core-valence + diffuse"),
    ("aug-cc-pwcv5z-pp", "aug-cc-pwCV5Z-PP", "low", "PP core-valence + diffuse"),
    # ---- ANO families (multireference) ----
    ("ano-rcc-vdz", "ANO-RCC-VDZ", "medium", ""),
    ("ano-rcc-vdzp", "ANO-RCC-VDZP", "medium", ""),
    ("ano-rcc-vtz", "ANO-RCC-VTZ", "medium", ""),
    ("ano-rcc-vtzp", "ANO-RCC-VTZP", "high", "multiref TM standard"),
    ("ano-rcc-vqzp", "ANO-RCC-VQZP", "medium", ""),
    ("ano-r", "ANO-R", "medium", ""),
    ("ano-r0", "ANO-R0", "medium", ""),
    ("ano-r1", "ANO-R1", "medium", ""),
    ("ano-r2", "ANO-R2", "medium", ""),
    ("ano-r3", "ANO-R3", "low", ""),
    # ---- Polarisability ----
    ("sadlej-pvtz", "Sadlej pVTZ", "medium", "polarisability"),
    ("sadlej+", "Sadlej+", "low", ""),
    # ---- Property-specific NMR (data only — needs NMR kernel for use) ----
    ("pcs-0", "pcS-0", "medium", "NMR shielding (blocked on NMR kernel)"),
    ("pcs-1", "pcS-1", "medium", "same"),
    ("pcs-2", "pcS-2", "medium", "same"),
    ("pcs-3", "pcS-3", "low", "same"),
    ("aug-pcs-1", "aug-pcS-1", "medium", "same"),
    ("aug-pcs-2", "aug-pcS-2", "medium", "same"),
    ("pcseg-s-0", "pcSseg-0", "medium", "segmented NMR shielding"),
    ("pcseg-s-1", "pcSseg-1", "medium", "same"),
    ("pcseg-s-2", "pcSseg-2", "medium", "same"),
    ("pcj-0", "pcJ-0", "medium", "NMR J-coupling (blocked on NMR kernel)"),
    ("pcj-1", "pcJ-1", "medium", "same"),
    ("pcj-2", "pcJ-2", "medium", "same"),
    ("pcj-3", "pcJ-3", "low", "same"),
    # ---- Sapporo-DKH3 (relativistic alternatives to dhf/x2c) ----
    ("sapporo-dkh3-dzp", "Sapporo-DKH3-DZP", "low", "DKH3 alternative to dhf-SVP"),
    ("sapporo-dkh3-tzp", "Sapporo-DKH3-TZP", "low", "DKH3 alternative to dhf-TZVP"),
    ("sapporo-dkh3-qzp", "Sapporo-DKH3-QZP", "low", ""),
    # ---- Cologne DKH2 ----
    ("cologne-dkh2", "Cologne DKH2", "low", "DKH2 light-element basis"),
    ("sarc-dkh2", "SARC-DKH2", "low", "Stuttgart Augmented Rel. Core DKH2"),
    ("sarc2-qzv-dkh2", "SARC2-QZV-DKH2", "low", ""),
    ("sarc2-qzvp-dkh2", "SARC2-QZVP-DKH2", "low", ""),
]


def fetch(canonical: str, bse_name: str, *, comment: str = "") -> Path:
    """Fetch one basis from BSE and write it to ``custom/<canonical>.g94``."""
    try:
        from basis_set_exchange import api  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "fetch_from_bse needs basis-set-exchange installed; "
            "pip install basis-set-exchange"
        ) from exc

    target = CUSTOM_DIR / f"{canonical}.g94"
    qvf_target = CUSTOM_DIR.parent / "qvf" / f"{canonical}.qvf.json"
    text = api.get_basis(
        name=bse_name,
        elements=None,  # all elements BSE has for this basis
        fmt="gaussian94",
        header=True,  # include BSE's own provenance header
    )
    today = datetime.date.today().isoformat()
    vibeqc_header = (
        f"! vibeqc-fetched from Basis Set Exchange on {today}.\n"
        f"! BSE name: {bse_name}\n"
        f"! Canonical name in vibe-qc: {canonical}\n"
        f"! Note: {comment}\n"
        f"!\n"
    )
    target.write_text(vibeqc_header + text)

    # Also write a QVF-Basis structured sidecar.
    try:
        from vibeqc.basis_toolkit import from_g94, save_qvf_json

        data = from_g94(vibeqc_header + text, name=canonical)
        qvf_target.parent.mkdir(exist_ok=True)
        save_qvf_json(data, qvf_target)
    except Exception:
        pass  # QVF is a best-effort sidecar; don't fail the fetch

    return target


PRIORITY_ORDER = {"highest": 0, "high": 1, "medium": 2, "low": 3}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--priority",
        choices=("highest", "high", "medium", "low", "all"),
        default="high",
        help="fetch all targets at this priority OR HIGHER (default: high)",
    )
    p.add_argument(
        "--name",
        action="append",
        default=None,
        help="fetch one specific canonical name; may be repeated",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list targets and their status, do not fetch",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be fetched, do not download",
    )
    args = p.parse_args(argv)

    threshold = 4 if args.priority == "all" else PRIORITY_ORDER[args.priority]

    if args.name:
        candidates = [
            (c, b, pri, com) for (c, b, pri, com) in TARGETS if c in set(args.name)
        ]
    else:
        candidates = [
            (c, b, pri, com)
            for (c, b, pri, com) in TARGETS
            if PRIORITY_ORDER[pri] <= threshold
        ]

    if args.list:
        for c, b, pri, com in candidates:
            present = "✅" if (CUSTOM_DIR / f"{c}.g94").exists() else "❌"
            print(f"{present} {c:24s} ({pri:7s}) ← BSE:{b:24s}  {com}")
        return 0

    rc = 0
    fetched: list[str] = []
    for c, b, pri, com in candidates:
        if (CUSTOM_DIR / f"{c}.g94").exists():
            print(f"= skip {c} (already shipped)")
            continue
        if args.dry_run:
            print(f"+ would fetch {c} from BSE:{b}")
            continue
        try:
            target = fetch(c, b, comment=com)
            print(f"+ {c}  →  {target.relative_to(PACKAGE_ROOT.parents[1])}")
            fetched.append(c)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {c} ({b}): {exc}", file=sys.stderr)
            rc = 1

    if fetched and not args.dry_run:
        print()
        print(f"fetched {len(fetched)} bases. Now run:")
        print("    ./scripts/setup_basis_library.sh")
        print("to promote custom/ → basis/, then run the load-test:")
        print(
            "    .venv/bin/python -m pytest tests/basisset_dev/test_basis_library_load.py"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
