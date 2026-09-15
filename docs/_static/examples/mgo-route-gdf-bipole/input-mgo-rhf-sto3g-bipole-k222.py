from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


MIN_MAIN_SHA = "3e3eef829"  # degenerate Gamma blocks re-expressed on a real basis
ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211
A_BOHR = A_ANG * ANG2BOHR


def build_mgo_primitive(vq):
    lattice = (A_BOHR / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    o_xyz = lattice @ np.array([0.5, 0.5, 0.5])
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, o_xyz.tolist()),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


def copy_input_snapshot(target: Path) -> None:
    source = Path(__file__).resolve()
    shutil.copy2(source, target)


def run_case(vq, case: dict, root: Path) -> dict:
    case_dir = root / case["slug"]
    case_dir.mkdir(parents=True, exist_ok=True)
    input_name = f"input-{case['slug']}.py"
    copy_input_snapshot(case_dir / input_name)
    stem = case_dir / f"output-{case['slug']}"

    system = build_mgo_primitive(vq)
    if case["jk_method"] == "bipole":
        vq.attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    kwargs = {
        "method": "RHF",
        "jk_method": case["jk_method"],
        "kpoints": case["kpoints"],
        "output": stem,
        "max_iter": 80,
        "conv_tol_energy": 1.0e-8,
        "initial_guess": "SAD",
        "use_diis": True,
        "diis_start_iter": 2,
        "damping": 0.0,
        "fmixing_percent": 30.0,
        "level_shift": 0.60,
        # Capability-aware auto, NOT True. Molden has no periodic
        # representation, so the most a periodic run can export is the
        # Gamma block, truncated to the home cell. BIPOLE's Gamma block is
        # real to eigensolver roundoff here and exports; multi-k GDF's is
        # not (a degenerate frontier orbital stays complex after global
        # phase removal), so that case correctly gets no Molden. The QVF
        # is the faithful artefact for both.
        "write_molden_file": None,
        "write_density": False,
        "write_xyz_file": True,
        "write_xsf_structure_file": True,
        "write_cif_file": True,
        # Capability-aware auto, NOT True. BIPOLE contracts the real-space
        # lattice density and the full SCF k-mesh, so it yields a genuine
        # crystal population. Multi-k GDF has no periodic population
        # convention: it would fall back to a molecular analysis of one
        # Bloch block, which does not conserve charge (the retired k222
        # GDF fixture reported Mg +1.5687 / O -0.9104 on a neutral cell,
        # summing to +0.658). run_periodic_job refuses that, so an explicit
        # True aborts the GDF case before its SCF.
        "write_population_file": None,
        "output_qvf": True,
        "record_hostname": False,
        "progress": True,
        "verbose": 2,
    }
    if case["jk_method"] == "bipole":
        kwargs.update(
            {
                "bipole_cutoff_bohr": 12.0,
                "bipole_nuclear_cutoff_bohr": 12.0,
                "ewald_precision": 1.0e-8,
                "sr_image_precision": 1.0e-6,
                "sr_range_screening": True,
            }
        )
    else:
        kwargs.update(
            {
                "gdf_method": "rsgdf",
                "rsgdf_ke_cutoff": 200.0,
            }
        )

    print(f"\n=== {case['slug']} ===", flush=True)
    t0 = time.perf_counter()
    result = vq.run_periodic_job(system, basis, **kwargs)
    elapsed = time.perf_counter() - t0

    n_elec = int(system.n_electrons())
    n_occ = n_elec // 2
    mo_energies = getattr(result, "mo_energies", None)
    homo = lumo = gap = None
    if mo_energies is not None:
        arrays = [np.asarray(eps, dtype=float).reshape(-1) for eps in mo_energies]
        if arrays and all(arr.size > n_occ for arr in arrays):
            homo = max(float(arr[n_occ - 1]) for arr in arrays)
            lumo = min(float(arr[n_occ]) for arr in arrays)
            gap = lumo - homo

    record = {
        "slug": case["slug"],
        "label": case["label"],
        "jk_method": case["jk_method"],
        "kpoints": case["kpoints"],
        "basis": "sto-3g",
        "a_angstrom": A_ANG,
        "energy_hartree_per_primitive_cell": float(result.energy),
        "converged": bool(getattr(result, "converged", False)),
        "n_iter": int(getattr(result, "n_iter", -1)),
        "n_basis": int(basis.nbasis),
        "n_electrons": n_elec,
        "homo_hartree": homo,
        "lumo_hartree": lumo,
        "gap_hartree": gap,
        "elapsed_seconds": elapsed,
        "artifacts": sorted(p.name for p in case_dir.glob("*")),
    }
    (case_dir / f"summary-{case['slug']}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return record


def main() -> int:
    os.environ.setdefault("VIBEQC_NO_HOSTNAME", "1")
    import vibeqc as vq
    from vibeqc.output.formats.qvf import validate_qvf

    program_git_dir = os.environ.get("VQ_PROGRAM_GIT_DIR")
    program_git_path = Path(program_git_dir) if program_git_dir else None
    program_git_dir_exists = bool(program_git_path and program_git_path.exists())
    program_sha = os.environ.get("VQ_PROGRAM_GIT_SHA") or os.environ.get(
        "VQ_PROGRAM_COMMIT"
    )
    if program_git_dir_exists:
        program_sha = subprocess.check_output(
            ["git", "-C", program_git_dir, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    elif program_git_dir:
        print(
            "Ignoring VQ_PROGRAM_GIT_DIR because it is not present on this "
            f"execution host: {program_git_dir}",
            flush=True,
        )
    if program_git_dir_exists:
        is_current_enough = subprocess.run(
            [
                "git",
                "-C",
                program_git_dir,
                "merge-base",
                "--is-ancestor",
                MIN_MAIN_SHA,
                "HEAD",
            ],
            text=True,
            check=False,
        )
        if is_current_enough.returncode != 0:
            raise RuntimeError(
                "managed vibeqc-dev runtime is "
                f"{program_sha}; it does not contain {MIN_MAIN_SHA}"
            )
    root = Path("mgo-route-fixtures").resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# MgO GDF/BIPOLE route fixtures\n\n"
        "Generated by `vq submit ... --program vibeqc-dev` for the "
        "documentation route-comparison tutorial.\n"
    )

    provenance = {
        "minimum_main_sha": MIN_MAIN_SHA,
        "program_git_sha_env": program_sha,
        "program_git_dir_env_present": bool(program_git_dir),
        "program_git_dir_exists": program_git_dir_exists,
        "program": os.environ.get("VQ_PROGRAM"),
        "program_python": os.environ.get("VQ_PROGRAM_PYTHON"),
        "program_bin": os.environ.get("VQ_PROGRAM_BIN"),
        "vibeqc_version": getattr(vq, "__version__", None),
        "python": sys.executable,
    }
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    cases = [
        {
            "slug": "mgo-rhf-sto3g-gdf-k222",
            "label": "MgO primitive RHF/STO-3G GDF 2x2x2",
            "jk_method": "gdf",
            "kpoints": [2, 2, 2],
        },
        {
            "slug": "mgo-rhf-sto3g-bipole-k222",
            "label": "MgO primitive RHF/STO-3G BIPOLE 2x2x2",
            "jk_method": "bipole",
            "kpoints": [2, 2, 2],
        },
    ]

    requested_slugs = set(sys.argv[1:])
    if requested_slugs:
        known_slugs = {case["slug"] for case in cases}
        unknown = sorted(requested_slugs - known_slugs)
        if unknown:
            raise ValueError(f"unknown case slug(s): {unknown}")
        cases = [case for case in cases if case["slug"] in requested_slugs]

    failures = []
    summaries = []
    for case in cases:
        try:
            summaries.append(run_case(vq, case, root))
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append({"slug": case["slug"], "error": repr(exc), "traceback": tb})
            case_dir = root / case["slug"]
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / f"failure-{case['slug']}.txt").write_text(tb)
            print(tb, flush=True)

    for qvf in sorted(root.glob("*/*.qvf")):
        result = validate_qvf(qvf)
        print(f"validate_qvf {qvf}: {result['valid']} {result.get('errors', [])}")
        if not result["valid"]:
            failures.append(
                {
                    "slug": qvf.parent.name,
                    "error": "validate_qvf failed",
                    "traceback": json.dumps(result, indent=2, default=str),
                }
            )

    bundle = {
        "provenance": provenance,
        "summaries": summaries,
        "failures": failures,
    }
    (root / "summary.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
