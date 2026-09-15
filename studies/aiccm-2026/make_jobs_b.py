#!/usr/bin/env python3
"""Emit complete ``vq`` inputs for the χ-CCM benchmark stream."""

from __future__ import annotations

import site_settings

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path

from b_routes import POST_HF_ROUTES, ROUTES, SCF_ROUTES, unsupported_reason
import testset


def _submit_dir() -> str:
    """Payload directory for ``vq submit -d``, relative to the checkout root.

    README_B.md runs this script from the repository root, so the emitted
    ``-d`` argument must be ``studies/aiccm-2026/``. A copy of this script
    living outside a vibe-qc checkout (the qc-input-library copy) keeps
    self-deriving the directory from its own name.
    """
    here = Path(__file__).resolve().parent
    for ancestor in here.parents:
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "python" / "vibeqc").is_dir():
            return f"{here.relative_to(ancestor).as_posix()}/"
    return f"{here.name}/"


SUBMIT_DIR = _submit_dir()
# Site assignments are external; the scientific route plan stays here.
HOST_BY_TIER = site_settings.host_map("tier_hosts")
COVERAGE_SYSTEMS = ("h-chain", "graphene", "lih-rocksalt")
COVERAGE_HOSTS = site_settings.host_map("coverage_hosts")
PAPER1_SYSTEMS = (
    "uniform-h-chain",
    "h-chain",
    "c-diamond",
    "mgo",
    "al2o3-corundum",
    "nacl-rocksalt",
)
PAPER1_ROUTES = (
    "rhf-4c",
    "rhf-ri",
    "rhf-rijcosx",
    "rks-pbe-ri",
    "rks-pbe0-4c",
    "rks-pbe0-ri",
    "rks-pbe0-rijcosx",
    "dlpno-ccsd-t",
)
PAPER1_MESH_BY_SYSTEM = {
    "uniform-h-chain": (8, 1, 1),
    "h-chain": (8, 1, 1),
    "c-diamond": (2, 2, 2),
    "mgo": (2, 2, 2),
    # Corundum is the expensive oxide gate. The one-cell four-center smoke is
    # still available via run_case_b.py, but the paper profile uses a nontrivial
    # character mesh so RI-RKS and RIJCOSX are real finite-torus calculations.
    "al2o3-corundum": (2, 1, 1),
    "nacl-rocksalt": (2, 2, 2),
}
INVESTIGATION_SYSTEMS = ("c-diamond", "si-diamond")
INVESTIGATION_ROUTE = "rks-pbe-ri"
INVESTIGATION_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("default-diagnostics", ()),
    ("ediis-diis", ("--scf-accelerator", "EDIIS_DIIS", "--max-iter", "200")),
    (
        "damping-diis40",
        ("--damping", "0.5", "--diis-start", "40", "--max-iter", "200"),
    ),
    (
        "level-shift",
        ("--level-shift", "0.3", "--level-shift-warmup", "5", "--max-iter", "200"),
    ),
    (
        "stress",
        (
            "--level-shift",
            "0.3",
            "--level-shift-warmup",
            "5",
            "--damping",
            "0.2",
            "--dynamic-damping",
            "--scf-accelerator",
            "EDIIS_DIIS",
            "--max-iter",
            "200",
        ),
    ),
)
JOB_CPUS = 4
JOB_MEMORY_MB = 8000
DIRECT_CUTOFF_BOHR = 15.0
_LIH_ROCKSALT_GEOMETRY_SIGNATURE = (
    3,
    (
        (0.0, 3.858820746486, 3.858820746486),
        (3.858820746486, 0.0, 3.858820746486),
        (3.858820746486, 3.858820746486, 0.0),
    ),
    (
        (3, (0.0, 0.0, 0.0)),
        (1, (3.858820746486, 3.858820746486, 3.858820746486)),
    ),
)



@dataclass(frozen=True)
class Job:
    system: str
    route: str
    host: str
    command: str
    wall_time_seconds: int
    variant: str | None = None


def _is_localhost_vq_host(host: str | None) -> bool:
    return site_settings.is_local_host(host)


def _routes_for(profile: str, system: str, meta: dict) -> tuple[str, ...]:
    if profile == "coverage":
        if system not in COVERAGE_SYSTEMS:
            return ()
        routes = list(SCF_ROUTES)
        if system == "lih-rocksalt":
            routes.extend(POST_HF_ROUTES)
        return tuple(routes)
    if profile == "scf":
        return SCF_ROUTES
    if profile == "posthf":
        return POST_HF_ROUTES
    if profile == "paper":
        routes = list(SCF_ROUTES)
        if meta["tier"] == "A" and int(meta["dim"]) == 3:
            routes.extend(POST_HF_ROUTES)
        return tuple(routes)
    if profile == "paper1":
        return PAPER1_ROUTES if system in PAPER1_SYSTEMS else ()
    if profile == "investigate":
        return (INVESTIGATION_ROUTE,) if system in INVESTIGATION_SYSTEMS else ()
    if profile == "full":
        return tuple(ROUTES)
    raise ValueError(f"unknown profile {profile!r}")


def _known_input_support_reason(
    system: str,
    basis_name: str,
    route_name: str,
    meta: dict,
    mesh: tuple[int, int, int],
    direct_cutoff_bohr: float,
) -> str | None:
    """Fail closed on fleet inputs with measured critical fold support."""

    if (
        system == "lih-rocksalt"
        and basis_name.casefold() == "sto-3g"
        and ROUTES[route_name].backend == "four_center"
        and tuple(mesh) == (2, 2, 2)
        and direct_cutoff_bohr == DIRECT_CUTOFF_BOHR
        and _geometry_signature(meta) == _LIH_ROCKSALT_GEOMETRY_SIGNATURE
    ):
        return (
            "the fixed LiH-rocksalt/STO-3G 2x2x2, 15-bohr direct input has max-k "
            "overlap-fold drift 7.0985e-2 (>1e-2); no resource-qualified "
            "converged replacement is pinned"
        )
    return None


def _geometry_signature(meta: dict) -> tuple:
    """Return the canonical registry geometry used by a fleet decision."""

    periodic_system = meta["build"]()
    lattice = tuple(
        tuple(round(float(value), 12) for value in row)
        for row in periodic_system.lattice
    )
    atoms = tuple(
        (
            int(atom.Z),
            tuple(round(float(value), 12) for value in atom.xyz),
        )
        for atom in periodic_system.unit_cell
    )
    return int(periodic_system.dim), lattice, atoms


def _job_argv(
    system: str,
    route_name: str,
    meta: dict,
    mesh: tuple[int, int, int],
    *,
    basis_name: str,
    local_mode: str,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    max_iter = "120"
    filtered_extra: list[str] = []
    skip_next = False
    for index, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--max-iter" and index + 1 < len(extra_args):
            max_iter = extra_args[index + 1]
            skip_next = True
            continue
        filtered_extra.append(arg)
    return [
        "bash",
        "run.sh",
        "--b",
        system,
        route_name,
        "--basis",
        basis_name,
        "--aux-basis",
        "def2-svp-jk",
        "--mesh",
        *(str(value) for value in mesh),
        "--gdf-method",
        "rsgdf",
        "--rsgdf-ke-cutoff",
        "200",
        "--max-iter",
        max_iter,
        "--energy-tol",
        "1e-8",
        "--gradient-tol",
        "1e-6",
        "--local-mode",
        local_mode,
        *filtered_extra,
        "--direct-cutoff-bohr",
        str(DIRECT_CUTOFF_BOHR),
    ]


def iter_jobs(
    profile: str,
    *,
    system_filter: str | None = None,
    route_filter: str | None = None,
    host_override: str | None = None,
    basis_override: str | None = None,
    local_mode: str = "exact",
    local: bool = False,
) -> tuple[list[Job], list[tuple[str, str, str]]]:
    """Build runnable jobs and explicit unsupported entries."""

    if local:
        host_override = "local"
    elif host_override is not None:
        host_override = site_settings.explicit_remote_host(host_override)

    jobs: list[Job] = []
    unsupported: list[tuple[str, str, str]] = []
    systems = [system_filter] if system_filter else list(testset.SYSTEMS)
    for system in systems:
        meta = testset.SYSTEMS[system]
        basis_name = basis_override or str(meta["basis"])
        routes = _routes_for(profile, system, meta)
        for route_name in routes:
            if route_filter and route_name != route_filter:
                continue
            route = ROUTES[route_name]
            mesh = tuple(int(value) for value in meta["nrep_4c"])
            if profile == "paper1":
                mesh = PAPER1_MESH_BY_SYSTEM[system]
            if profile == "coverage" and route.post_hf:
                # Keep the all-route fleet gate small enough to be diagnostic.
                # The full profile retains the paper registry's 2x2x2 LiH torus.
                mesh = (2, 1, 1)
            reason = _known_input_support_reason(
                system,
                basis_name,
                route_name,
                meta,
                mesh,
                DIRECT_CUTOFF_BOHR,
            ) or unsupported_reason(meta, route_name, mesh)
            if reason:
                unsupported.append((system, route_name, reason))
                continue
            host = host_override or (
                site_settings.required_host("paper_host")
                if profile == "paper1"
                else (
                    COVERAGE_HOSTS[system]
                    if profile == "coverage"
                    else (site_settings.required_host("posthf_host") if route.post_hf else HOST_BY_TIER[meta["tier"]])
                )
            )
            if profile == "paper1":
                wall = 172800 if (route.post_hf or meta["tier"] == "C") else 86400
            else:
                wall = 86400 if route.post_hf else (21600 if meta["tier"] == "C" else 14400)
            variants = (
                INVESTIGATION_VARIANTS
                if profile == "investigate"
                else (("", ()),)
            )
            for variant, extra_args in variants:
                argv = _job_argv(
                    system,
                    route_name,
                    meta,
                    mesh,
                    basis_name=basis_name,
                    local_mode=local_mode,
                    extra_args=extra_args,
                )
                jobs.append(
                    Job(
                        system=system,
                        route=route_name,
                        host=host,
                        command=shlex.join(argv),
                        wall_time_seconds=wall,
                        variant=variant or None,
                    )
                )
    return jobs, unsupported


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "coverage",
            "scf",
            "posthf",
            "paper",
            "paper1",
            "full",
            "investigate",
        ),
        default="coverage",
        help=(
            "coverage enumerates each implementation and emits only the "
            "currently runnable fixed inputs; paper1 is the first AICCM "
            "article matrix; full is the Cartesian matrix"
        ),
    )
    parser.add_argument("--system", choices=sorted(testset.SYSTEMS))
    parser.add_argument("--route", choices=tuple(ROUTES))
    parser.add_argument("--host", help="send every emitted job to one vq host")
    parser.add_argument(
        "--basis",
        help=(
            "override the registry basis for emitted jobs, e.g. pob-tzvp-rev2 "
            "for CRYSTAL-matched article runs"
        ),
    )
    parser.add_argument("--local-mode", choices=("exact", "pno"), default="exact")
    parser.add_argument("--local", action="store_true", help="emit direct Python commands")
    args = parser.parse_args(argv)

    # A vq batch must stay on the queue fleet. `--local` is the explicit path
    # for direct smoke commands and avoids emitting `vq submit` altogether.
    if not args.local and _is_localhost_vq_host(args.host):
        parser.error(
            f"--host {args.host!r} would submit the χ-CCM-B batch to this laptop; "
            "pass a configured remote host or use --local"
        )

    jobs, unsupported = iter_jobs(
        args.profile,
        system_filter=args.system,
        route_filter=args.route,
        host_override=args.host,
        local=args.local,
        basis_override=args.basis,
        local_mode=args.local_mode,
    )
    print(
        f"# aiccm2026dev-b profile={args.profile}: {len(jobs)} runnable jobs, "
        f"{len(unsupported)} intentionally unsupported combinations"
    )
    print(
        "# Preview only: do not pipe this output to a shell; review the "
        "independent D77/D93/D103/D104 qualification gates in README_B.md."
    )
    for system, route, reason in unsupported:
        print(f"# UNSUPPORTED {system} {route}: {reason}")
    for job in jobs:
        if args.local:
            print(job.command)
            continue
        suffix = "" if job.variant is None else f"-{job.variant}"
        name = f"b-{job.system}-{job.route}{suffix}"
        tags = "--tag aiccm2026dev-b"
        if job.variant is not None:
            tags += " --tag investigate"
        print(
            f"vq submit {shlex.quote(job.host)} -d {shlex.quote(SUBMIT_DIR)} "
            f"--cpus {JOB_CPUS} "
            f"--mem-mb {JOB_MEMORY_MB} --wall-time-seconds {job.wall_time_seconds} "
            f"--job-name {shlex.quote(name)} {tags} -- "
            f"{job.command}"
        )


if __name__ == "__main__":
    main()
