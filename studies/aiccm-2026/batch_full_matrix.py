#!/usr/bin/env python
"""Full-matrix molecular benchmark batch — methods x basis x RIJCOSX/GridX x convergers.

Uses an explicit external host rotation for vq submissions. Ten small molecules, each run through a
combinatorial matrix of method, basis, RIJCOSX/GridX tier, and SCF convergence
accelerator, so every combination is stress-tested.

Writes one .py job script per combination to ``batch_scripts/``, plus a
``run_mol_case.sh`` launcher (finds the vibeqc-dev venv python, same pattern as
``studies/aiccm-2026/run.sh``). Then emits ``vq submit`` commands referencing those files.

Matrix (configurable via CLI filters):
  molecules (10)   methods (5)    bases (4)    COSX (4)       convergers (7)
  ------------     -----------    --------     -----------     -------------------
  h2o, nh3, …      rhf, rks/pbe,  sto-3g,     off,            ediis-diis, kdiis,
                    mp2, ccsd,     6-31g*,     legacy,         ad-cdiis, newton,
                    ccsd(t)        def2-svp,   auto-gridx,     trah, quadratic,
                                   cc-pvdz      gridx3          damping

Usage:
    # Dry run — write scripts + review vq commands
    python batch_full_matrix.py
    # Launch a subset
    python batch_full_matrix.py --method rhf rks --molecule h2o nh3 | sh
    # Launch the full matrix (~5,600 jobs)
    python batch_full_matrix.py | sh
    # Run locally (single machine, no vq)
    python batch_full_matrix.py --local | sh
    # Just generate scripts (no emission)
    python batch_full_matrix.py --dry-run

Output: <stem>.out per job in $VQ_WORKDIR (vq) or ./batch_scripts/ (local).
"""

from __future__ import annotations

import site_settings

import argparse
import os
import textwrap
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent

# ── Molecules ──────────────────────────────────────────────────────────────

MOLECULES: dict[str, str] = {
    "h2o": textwrap.dedent("""\
        3
        water
        O    0.000000    0.000000    0.117349
        H    0.000000    1.431437   -0.469396
        H    0.000000   -1.431437   -0.469396
    """),
    "nh3": textwrap.dedent("""\
        4
        ammonia
        N    0.000000    0.000000    0.119752
        H    0.000000    0.937505   -0.279394
        H    0.811891   -0.468752   -0.279394
        H   -0.811891   -0.468752   -0.279394
    """),
    "ch4": textwrap.dedent("""\
        5
        methane
        C    0.000000    0.000000    0.000000
        H    0.627611    0.627611    0.627611
        H   -0.627611   -0.627611    0.627611
        H    0.627611   -0.627611   -0.627611
        H   -0.627611    0.627611   -0.627611
    """),
    "hf": textwrap.dedent("""\
        2
        hydrogen fluoride
        F    0.000000    0.000000    0.092032
        H    0.000000    0.000000   -1.644310
    """),
    "co2": textwrap.dedent("""\
        3
        carbon dioxide
        C    0.000000    0.000000    0.000000
        O    0.000000    0.000000    1.161579
        O    0.000000    0.000000   -1.161579
    """),
    "h2co": textwrap.dedent("""\
        4
        formaldehyde
        C    0.000000    0.000000   -0.528316
        O    0.000000    0.000000    0.677740
        H    0.000000    0.929765   -1.123724
        H    0.000000   -0.929765   -1.123724
    """),
    "c2h4": textwrap.dedent("""\
        6
        ethene
        C    0.000000    0.000000    0.667293
        C    0.000000    0.000000   -0.667293
        H    0.000000    0.921822    1.238404
        H    0.000000   -0.921822    1.238404
        H    0.000000    0.921822   -1.238404
        H    0.000000   -0.921822   -1.238404
    """),
    "hcn": textwrap.dedent("""\
        3
        hydrogen cyanide
        C    0.000000    0.000000   -0.505778
        N    0.000000    0.000000    0.655565
        H    0.000000    0.000000   -1.570744
    """),
    "h2s": textwrap.dedent("""\
        3
        hydrogen sulfide
        S    0.000000    0.000000    0.103729
        H    0.000000    0.945339   -0.829688
        H    0.000000   -0.945339   -0.829688
    """),
    "ch3oh": textwrap.dedent("""\
        6
        methanol
        C    0.000000    0.666351   -0.044991
        O    0.000000   -0.755953    0.127892
        H    0.000000    1.121861    0.915158
        H    0.890101    0.988783   -0.611431
        H   -0.890101    0.988783   -0.611431
        H    0.000000   -1.141879   -0.759821
    """),
}

# ── Method × basis × COSX × converger matrices ─────────────────────────────

METHODS: list[tuple[str, str | None, bool]] = [
    # (run_job method,  functional,  is_post_scf)
    ("rhf", None, False),
    ("rks", "pbe", False),
    ("mp2", None, True),
    ("ccsd", None, True),
    ("ccsd(t)", None, True),
]

BASIS_SETS: list[str] = ["sto-3g", "6-31g*", "def2-svp", "cc-pvdz"]

COSX_LEVELS: list[tuple[str, bool, int]] = [
    # (key,           cosx,   grid_level)
    ("off", False, 0),
    ("legacy", True, 0),
    ("auto-gridx", True, -1),
    ("gridx3", True, 3),
]

CONVERGERS: list[tuple[str, str, str]] = [
    # (key,                    label,                  python_snippet)
    (
        "ediis-diis",
        "EDIIS+DIIS",
        """
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS
""",
    ),
    (
        "kdiis",
        "KDIIS",
        """
opts.scf_accelerator = vq.SCFAccelerator.KDIIS
""",
    ),
    (
        "ad-cdiis",
        "AD-CDIIS",
        """
opts.scf_accelerator = vq.SCFAccelerator.AD_CDIIS
opts.diis_adaptive_delta = 1e-4
""",
    ),
    (
        "ediis-diis-newton",
        "EDIIS+DIIS+Newton",
        """
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS
opts.newton_threshold = 1.0
""",
    ),
    (
        "ediis-diis-trah",
        "EDIIS+DIIS+TRAH",
        """
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS
opts.trah_threshold = 1.0
""",
    ),
    (
        "quadratic",
        "EDIIS+DIIS+Quadratic",
        """
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS
opts.quadratic_fallback_iter = 30
opts.quadratic_fallback_shift = 0.1
opts.quadratic_fallback_max_step = 0.1
""",
    ),
    (
        "damping",
        "Damping-only",
        """
opts.use_diis = False
opts.damping = 0.7
opts.fock_mixing = 0.3
""",
    ),
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _safe_stem_part(s: str) -> str:
    """Replace shell-unsafe characters for filename-safe stems."""
    return s.replace("*", "s").replace("(", "").replace(")", "")


def _output_stem(
    mol: str, method: str, basis: str, cosx_key: str, converger_key: str
) -> str:
    return (
        f"{mol}__{_safe_stem_part(method)}__"
        f"{_safe_stem_part(basis)}__{cosx_key}__{converger_key}"
    )


def _auto_aux_basis_code(orbital_basis: str) -> str:
    return textwrap.dedent(f"""\
    try:
        _aux = vq.default_aux_basis_for("{orbital_basis}", kind="jk")
    except Exception:
        _aux = "def2-universal-jfit"
    """)


def _is_rhf_method(method: str) -> bool:
    return method in ("rhf", "mp2", "ccsd", "ccsd(t)")


def generate_job_script(
    mol_name: str,
    xyz: str,
    method: str,
    functional: str | None,
    basis_name: str,
    cosx: bool,
    cosx_grid_level: int,
    conv_code: str,
    stem: str,
) -> str:
    """Generate one self-contained Python job script."""

    kw_parts = [f'output="{stem}"', "write_molden_file=False"]
    if functional:
        kw_parts.append(f'functional="{functional}"')

    # Build converger + COSX blocks as properly-indented lines
    conv_lines = [
        l for l in textwrap.dedent(conv_code).strip().split("\n") if l.strip()
    ]

    cosx_lines: list[str] = []
    if cosx:
        cosx_lines = [
            f"opts.cosx = True",
            f"opts.cosx_grid_level = {cosx_grid_level}",
            f"opts.density_fit = True",
        ]
        aux_code = _auto_aux_basis_code(basis_name).strip()
        cosx_lines.extend(aux_code.split("\n"))
        cosx_lines.extend(
            [
                f"opts.aux_basis = _aux",
                f"opts.cosx_variant = vq.CosxVariant.AUTO",
            ]
        )

    rf = "rhf" if _is_rhf_method(method) else "rks"

    # Assemble: indent each option line by 4 spaces
    all_option_lines = conv_lines + cosx_lines
    option_block = "\n".join(f"    {l}" for l in all_option_lines)

    return textwrap.dedent(f"""\
    import vibeqc as vq

    xyz = {repr(xyz)}
    mol = vq.Molecule.from_xyz_str(xyz)

    opts = vq.{rf.upper()}Options()
    opts.use_diis = True
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-6
{option_block}

    result = vq.run_job(
        mol,
        basis="{basis_name}",
        method="{method}",
        {", ".join(kw_parts)},
        {rf}_options=opts,
    )

    e = getattr(result, 'energy', None) or getattr(result, 'e_total', 0.0)
    n_iter = getattr(result, 'n_iter', '?')
    conv = getattr(result, 'converged', '?')
    print(f"E={{e:.10f}} Ha  n_iter={{n_iter}}  converged={{conv}}")
    if hasattr(result, 'ccsd'):
        cc = result.ccsd
        print(f"E_corr={{cc.e_ccsd_correlation:.10f}}  E_T={{cc.e_t:.10f}}")
        print(f"CCSD n_iter={{cc.n_iter}}  converged={{cc.converged}}")
    if hasattr(result, 'mp2'):
        print(f"E_corr={{result.mp2.e_correlation:.10f}}")
    """)


def _write_launcher(path: Path) -> None:
    """Write the run_mol_case.sh launcher (same pattern as run.sh)."""
    path.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        cands=(
            "$HOME/gitlab/vibeqc-dev/.venv/bin/python"
            "$HOME/vibeqc-dev/.venv/bin/python"
            "$HOME/gitlab/vibeqc/.venv/bin/python"
        )
        PY=""
        for p in "${cands[@]}"; do
            [ -x "$p" ] && PY="$p" && break
        done
        if [ -z "$PY" ] && command -v python >/dev/null 2>&1 && python -c "import vibeqc" 2>/dev/null; then
            PY="python"
        fi
        : "${PY:?run_mol_case.sh: could not locate a vibeqc-enabled python}"
        echo "run_mol_case.sh: using interpreter $PY" >&2
        exec "$PY" "$@"
    """)
    )
    os.chmod(path, 0o755)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full-matrix molecular benchmark batch"
    )
    ap.add_argument(
        "--local", action="store_true", help="emit commands for local execution (no vq)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="write scripts only, don't emit commands"
    )
    ap.add_argument(
        "--method", nargs="*", default=None, choices=[m[0] for m in METHODS]
    )
    ap.add_argument("--molecule", nargs="*", default=None, choices=list(MOLECULES))
    ap.add_argument("--basis", nargs="*", default=None, choices=BASIS_SETS)
    ap.add_argument(
        "--cosx", nargs="*", default=None, choices=[c[0] for c in COSX_LEVELS]
    )
    ap.add_argument(
        "--converger", nargs="*", default=None, choices=[c[0] for c in CONVERGERS]
    )
    ap.add_argument(
        "--scripts-dir",
        default="batch_scripts",
        help="output directory for .py job scripts (default: batch_scripts/)",
    )
    ap.add_argument(
        "--cores", type=int, default=4, help="CPU cores per vq job (default: 4)"
    )
    args = ap.parse_args()
    hosts = ("local",) if args.local or args.dry_run else site_settings.molecular_hosts()

    mols = [m for m in MOLECULES if args.molecule is None or m in args.molecule]
    meths = [m for m in METHODS if args.method is None or m[0] in args.method]
    bases = [b for b in BASIS_SETS if args.basis is None or b in args.basis]
    cosxes = [c for c in COSX_LEVELS if args.cosx is None or c[0] in args.cosx]
    convs = [c for c in CONVERGERS if args.converger is None or c[0] in args.converger]

    n_total = len(mols) * len(meths) * len(bases) * len(cosxes) * len(convs)

    print(f"# Full-matrix molecular benchmark")
    print(f"#   molecules:  {len(mols)}  ({', '.join(mols)})")
    print(f"#   methods:    {len(meths)}  ({', '.join(m[0] for m in meths)})")
    print(f"#   bases:      {len(bases)}")
    print(f"#   COSX:       {len(cosxes)}  ({', '.join(c[0] for c in cosxes)})")
    print(f"#   convergers: {len(convs)}  ({', '.join(c[0] for c in convs)})")
    print(f"#   total jobs: {n_total}")
    print(f"#   scripts:    {args.scripts_dir}/*.py")
    print()

    # Set up output directory
    scripts_dir = Path(args.scripts_dir)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    _write_launcher(scripts_dir / "run_mol_case.sh")

    # Generate all scripts
    written = 0
    for i_mol, mol_name in enumerate(mols):
        machine = hosts[i_mol % len(hosts)]
        xyz = MOLECULES[mol_name]

        for method, functional, _post_scf in meths:
            for basis_name in bases:
                for cosx_key, cosx, grid_level in cosxes:
                    for conv_key, _conv_label, conv_code in convs:
                        stem = _output_stem(
                            mol_name, method, basis_name, cosx_key, conv_key
                        )
                        script = generate_job_script(
                            mol_name,
                            xyz,
                            method,
                            functional,
                            basis_name,
                            cosx,
                            grid_level,
                            conv_code,
                            stem,
                        )
                        (scripts_dir / f"{stem}.py").write_text(script)
                        written += 1

    print(f"# Wrote {written} job scripts to {scripts_dir.resolve()}/\n")

    if args.dry_run:
        print("# Dry run complete. Scripts written, no commands emitted.")
        print(f"# To run: cd {scripts_dir.resolve()} && ls *.py | wc -l")
        return

    # Emit commands
    if args.local:
        print("# ---- Local execution ----")
        for py_file in sorted(scripts_dir.glob("*.py")):
            if py_file.name == "run_mol_case.sh":
                continue
            print(
                f"cd {scripts_dir.resolve()} && python3 run_mol_case.sh {py_file.name}"
            )
    else:
        # Group by machine
        by_machine: dict[str, list[str]] = {}
        for i_mol, mol_name in enumerate(mols):
            machine = hosts[i_mol % len(hosts)]
            by_machine.setdefault(machine, [])
        for py_file in sorted(scripts_dir.glob("*.py")):
            if py_file.name == "run_mol_case.sh":
                continue
            # Parse molecule name from the stem to assign machine
            mol_from_stem = py_file.stem.split("__")[0]
            i_mol = mols.index(mol_from_stem) if mol_from_stem in mols else 0
            machine = hosts[i_mol % len(hosts)]
            by_machine.setdefault(machine, []).append(
                f"vq submit {machine} -c {args.cores} "
                f"-d {scripts_dir.resolve()}/ -- "
                f"bash run_mol_case.sh {py_file.name}"
            )

        for machine in sorted(by_machine):
            cmds = by_machine[machine]
            print(f"\n# ---- {machine} ({len(cmds)} jobs) ----")
            for cmd in cmds:
                print(cmd)

    print(f"\n# Pipe to sh to launch: python batch_full_matrix.py | sh")


if __name__ == "__main__":
    main()
