"""Run a vibe-qc SCF case end-to-end with verbose logging.

Two entry points: :func:`run_periodic_case` (rocksalts, oxides, …)
and :func:`run_molecule_case` (H₂ / H₂O / Ne / …).
"""
from __future__ import annotations

import json
import math
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import vibeqc as vq
from vibeqc.options_dump import format_options
from vibeqc.output.formats.qvf import qvf_wf_data, scf_history_from_result, write_qvf
from vibeqc.output.formats.system_info import write_system_manifest
from vibeqc.output.plan import OutputPlan
from vibeqc.progress import ProgressLogger

from . import input_identity
from .case import CodeRow
from .periodic_matrix import periodic_state_request
from .spec import MethodSpec, MoleculeSpec, PeriodicSpec

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_DENSE_IONIC_EWALD_RETIRED = "EWALD_3D is retired for dense ionic crystals"
_GAMMA_GDF_SOLID_HOLD_FAMILIES = {"rare_gas_fcc", "rocksalt"}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _method_payload(method: MethodSpec) -> dict:
    return asdict(method)


def _artifact_stem(artifact_dir: Optional[Path]) -> Optional[Path]:
    if artifact_dir is None:
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / "vibeqc"


def _periodic_job_output_stem(
    *,
    artifact_stem: Optional[Path],
    log_path: Path,
) -> Path:
    if artifact_stem is not None:
        return artifact_stem.parent / "vibeqc_job"
    return log_path.with_name(f"{log_path.stem}_job")


def _finalize_artifacts(
    *,
    artifact_dir: Optional[Path],
    log_path: Path,
    row: CodeRow,
    wall_s: float,
    qvf_source: Optional[Path] = None,
) -> None:
    stem = _artifact_stem(artifact_dir)
    if stem is None:
        return
    out_path = stem.with_suffix(".out")
    if log_path.exists():
        out_path.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        out_path.write_text(row.note or "vibe-qc runner produced no verbose log\n")
    write_system_manifest(
        out_path,
        wall_seconds=wall_s,
        basename=stem.name,
        record_hostname=True,
    )
    if qvf_source is not None:
        qvf_target = stem.with_suffix(".qvf")
        if qvf_source.is_file():
            shutil.copy2(qvf_source, qvf_target)
        else:
            msg = (
                "QVF artifact requested but the vibe-qc runner did not "
                f"produce {qvf_source}"
            )
            (artifact_dir / "qvf_error.log").write_text(msg + "\n", encoding="utf-8")
            row.note = f"{row.note}; {msg}" if row.note else msg
    _write_json(stem.with_suffix(".parsed.json"), asdict(row))
    _write_json(artifact_dir / "parsed.json", asdict(row))
    for name in ("stdout.log", "stderr.log"):
        p = artifact_dir / name
        if not p.exists():
            p.write_text("", encoding="utf-8")


def _write_molecule_qvf_artifact(
    *,
    artifact_dir: Path,
    mol,
    basis,
    basis_name: str,
    method: MethodSpec,
    scf_result,
    wall_s: float,
    row: CodeRow,
) -> None:
    """Write a case-local QVF for a molecular regression calculation."""
    stem = _artifact_stem(artifact_dir)
    if stem is None:
        return
    if not bool(getattr(scf_result, "converged", False)):
        msg = "QVF artifact requested but SCF did not converge"
        (artifact_dir / "qvf_error.log").write_text(msg + "\n", encoding="utf-8")
        row.note = f"{row.note}; {msg}" if row.note else msg
        return

    plan = OutputPlan.from_run_job_kwargs(
        output=stem,
        method=method.scf,
        basis=basis_name,
        functional=method.xc,
        write_molden_file=False,
        write_xyz=False,
        write_population=False,
        citations=False,
        crash_dump=False,
        output_qvf=True,
    )
    wf_data = None
    if hasattr(scf_result, "mo_energies") or hasattr(scf_result, "mo_energies_alpha"):
        try:
            wf_data = qvf_wf_data(scf_result, basis, mol)
        except Exception as exc:
            (artifact_dir / "qvf_wavefunction_warning.log").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
    try:
        write_qvf(
            stem,
            plan,
            molecule=mol,
            result=scf_result,
            method=method.scf.upper(),
            basis=basis_name,
            functional=method.xc,
            wall_seconds=wall_s,
            wf_data=wf_data,
            scf_history_data=scf_history_from_result(scf_result),
        )
    except Exception as exc:
        msg = f"QVF artifact failed: {type(exc).__name__}: {str(exc)[:180]}"
        (artifact_dir / "qvf_error.log").write_text(msg + "\n", encoding="utf-8")
        row.note = f"{row.note}; {msg}" if row.note else msg


def _classify_periodic_exception(row: CodeRow, exc: Exception) -> None:
    row.status = "error"
    row.note = f"{type(exc).__name__}: {str(exc)[:160]}"
    if _DENSE_IONIC_EWALD_RETIRED in str(exc):
        row.status = "unavailable"
        row.note = (
            "unsupported retired EWALD_3D dense-ionic diagnostic: "
            f"{str(exc)[:220]}"
        )


def _gamma_gdf_solid_absolute_parity_held(
    spec: PeriodicSpec,
    basis_name: str,
    method: MethodSpec,
    kmesh: Tuple[int, int, int],
) -> bool:
    """Known validation hold for undersized STO-3G Gamma RKS-LDA/GDF tails."""
    if kmesh != (1, 1, 1):
        return False
    if basis_name.lower() != "sto-3g":
        return False
    if method.id != "rks-lda":
        return False
    return spec.family in _GAMMA_GDF_SOLID_HOLD_FAMILIES


def _gamma_gdf_solid_hold_after_tail(
    system: "vq.PeriodicSystem",
    basis: "vq.BasisSet",
    spec: PeriodicSpec,
    basis_name: str,
    method: MethodSpec,
    kmesh: Tuple[int, int, int],
    rsgdf_tail_ke_cutoff: Optional[float],
) -> bool:
    if not _gamma_gdf_solid_absolute_parity_held(spec, basis_name, method, kmesh):
        return False
    if rsgdf_tail_ke_cutoff is None:
        # Current vibe-qc auto-sizes the dense-core Gamma RSGDF high-|G|
        # tail when the caller leaves the cutoff unset. Only explicit
        # undersized diagnostic tails remain held.
        return False
    try:
        from vibeqc.pbc_gdf import _gamma_dense_core_gdf_parity_held

        return _gamma_dense_core_gdf_parity_held(
            system,
            "rsgdf",
            ao_basis=basis,
            tail_ke_cutoff=float(rsgdf_tail_ke_cutoff),
        )
    except Exception:
        return True


def _build_periodic_system(spec: PeriodicSpec) -> "vq.PeriodicSystem":
    # spec.lattice_ang ROWS are the lattice vectors, so r = frac @ L. The
    # orientation-free lattice_vectors form avoids the #445 shear that a
    # row matrix passed as ``lattice`` (columns) gives a non-symmetric cell.
    lat_bohr = np.asarray(spec.lattice_ang, dtype=float) * ANGSTROM_TO_BOHR
    atoms = []
    for at in spec.atoms:
        cart_bohr = np.asarray(at.frac, dtype=float) @ lat_bohr
        atoms.append(vq.Atom(int(at.z), [float(x) for x in cart_bohr]))
    # vibe-qc's periodic charge and multiplicity are per primitive cell,
    # the convention periodic_state_request resolves.
    return vq.PeriodicSystem(3, lattice_vectors=list(lat_bohr), unit_cell=atoms,
                             charge=int(spec.charge),
                             multiplicity=int(spec.multiplicity))


def _input_identity(system, basis, basis_name: str, method: MethodSpec) -> dict:
    """What this run actually built, for physical-input qualification (#648)."""
    atoms = [(int(a.Z), *map(float, a.xyz)) for a in system.unit_cell]
    shells = []
    for shell in basis.shells():
        alpha = np.asarray(shell.exponents, dtype=float)
        # libint coefficients include each primitive's normalization,
        # proportional to alpha^((2l+3)/4); remove it.
        d = np.asarray(shell.coefficients, dtype=float) / alpha ** ((2 * shell.l + 3) / 4)
        shells.append((shell.atom_index, shell.l, shell.pure, alpha, [d]))
    if method.scf in ("rhf", "uhf"):
        xc = input_identity.xc_identity(kind="hf", hf_fraction=1.0, omega=0.0)
    else:
        xc = input_identity.xc_identity_from_functional(vq.Functional(method.xc or "LDA"))
    return {**_geometry_basis_ecp(system, atoms, shells, basis), "xc": xc}


def _geometry_basis_ecp(system, atoms, shells, basis) -> dict:
    lattice_rows = np.asarray(system.lattice, dtype=float).T
    return {
        "geometry": input_identity.geometry_identity(lattice_rows, atoms),
        "basis": input_identity.basis_identity(shells),
        "ecp": _resolved_ecp_identity(system, basis),
    }


def _resolved_ecp_identity(system, basis) -> dict:
    """Use the public driver's resolver, including zero-core potentials.

    This is the resolved preflight input, not an attestation of later integral
    execution. Unknown or ambiguous centers remain unverified.
    """
    from vibeqc.periodic_runner import _resolve_ecp_data

    try:
        blocks, centers, charges, total_core = _resolve_ecp_data(system, basis)
        atoms = list(system.unit_cell)
        if not blocks and not charges:
            charges = [float(a.Z) for a in atoms]
        if len(charges) != len(atoms) or len(blocks) != len(centers):
            raise ValueError("inconsistent ECP center/count metadata")
        cores, terms = {}, {str(i): [] for i in range(len(atoms))}
        for i, (a, charge) in enumerate(zip(atoms, charges)):
            core = float(a.Z) - charge
            if not math.isfinite(core) or abs(core - round(core)) > 1e-10:
                raise ValueError("nonintegral ECP core count")
            cores[str(i)] = int(round(core))
        if sum(cores.values()) != total_core:
            raise ValueError("inconsistent total ECP core count")
        seen = set()
        for block, center in zip(blocks, centers):
            matches = [i for i, a in enumerate(atoms)
                       if np.allclose(a.xyz, center, rtol=0, atol=1e-10)]
            if len(matches) != 1 or matches[0] in seen:
                raise ValueError("ambiguous ECP center-to-atom mapping")
            atom = matches[0]
            seen.add(atom)
            if not block.ams or not all(len(v) == block.n_primitive for v in
                                       (block.ams, block.ns, block.exponents, block.coefficients)):
                raise ValueError("inconsistent ECP primitive arrays")
            # libecpint's highest angular channel is local. Its n includes
            # the radial Jacobian; the potential itself contains r^(n-2).
            local = max(block.ams)
            terms[str(atom)] = [[-1 if l == local else int(l), int(n) - 2, float(e), float(c)]
                                for l, n, e, c in zip(block.ams, block.ns,
                                                     block.exponents, block.coefficients)]
        if any(core and int(atom) not in seen for atom, core in cores.items()):
            raise ValueError("missing ECP operator for a core-replaced atom")
        return input_identity.ecp_identity(cores, terms, "periodic production ECP resolver")
    except Exception as exc:
        return input_identity.unverified(f"ECP operator resolution failed: {type(exc).__name__}: {exc}")


def _format_geometry_block(spec: PeriodicSpec) -> str:
    lat = np.asarray(spec.lattice_ang, dtype=float)
    lines = [
        f"[geometry: {spec.id}]",
        f"  family            = {spec.family}",
        f"  space_group       = {spec.space_group}",
        f"  lattice (Å, rows) = {lat[0].tolist()}",
        f"                      {lat[1].tolist()}",
        f"                      {lat[2].tolist()}",
        f"  n_atoms_per_cell  = {len(spec.atoms)}",
        "  atoms (frac):",
    ]
    for at in spec.atoms:
        f0, f1, f2 = at.frac
        lines.append(
            f"    Z={at.z:>2}  {at.symbol:>2}  "
            f"({f0:7.4f}, {f1:7.4f}, {f2:7.4f})"
        )
    if spec.notes:
        lines.append(f"  notes             = {spec.notes}")
    if spec.citation:
        lines.append(f"  citation          = {spec.citation}")
    return "\n".join(lines)


def _make_options(method: MethodSpec, spec: PeriodicSpec, *,
                  cutoff_bohr: float, nuclear_cutoff_bohr: float,
                  conv_tol_energy: float, max_iter: int):
    if method.scf in ("rks", "uks"):
        opts = vq.PeriodicKSOptions()
        opts.functional = (method.xc or "LDA").upper()
    elif method.scf in ("rhf", "uhf"):
        opts = vq.PeriodicRHFOptions()
    else:
        raise NotImplementedError(
            f"runner_vibeqc: scf={method.scf!r} not wired in this iteration; "
            f"only rhf, uhf, rks and uks are supported."
        )
    opts.lattice_opts.cutoff_bohr = float(cutoff_bohr)
    opts.lattice_opts.nuclear_cutoff_bohr = float(nuclear_cutoff_bohr)
    opts.conv_tol_energy = float(conv_tol_energy)
    opts.max_iter = int(max_iter)
    opts.damping = float(spec.default_damping)
    if spec.default_initial_guess.upper() == "SAD":
        opts.initial_guess = vq.InitialGuess.SAD
    elif spec.default_initial_guess.upper() == "HCORE":
        opts.initial_guess = vq.InitialGuess.HCORE
    if method.scf in ("rks", "uks"):
        opts.use_periodic_becke = bool(spec.default_use_periodic_becke)
    return opts


def run_periodic_case(
    *, run_id: str, target: str, code_version: str, spec: PeriodicSpec,
    basis_name: str, method: MethodSpec, kmesh: Tuple[int, int, int],
    jk_method: str = "gdf",
    spacing_bohr: Optional[float] = None,
    cutoff_bohr: Optional[float] = None,
    nuclear_cutoff_bohr: Optional[float] = None,
    omega: Optional[float] = None,
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    log_path: Path,
    artifact_dir: Optional[Path] = None,
    write_qvf_artifact: bool = False,
) -> CodeRow:
    """Run one vibe-qc periodic SCF case and return its CodeRow.

    Writes a complete verbose log section to ``log_path`` (truncated by
    ProgressLogger on construction).
    """
    if jk_method not in ("gdf", "bipole"):
        raise ValueError("periodic regression jk_method must be 'gdf' or 'bipole'")
    if jk_method != "gdf" and rsgdf_tail_ke_cutoff is not None:
        raise ValueError("rsgdf_tail_ke_cutoff requires jk_method='gdf'")
    if spacing_bohr is None:
        spacing_bohr = spec.default_spacing_bohr
    if cutoff_bohr is None:
        cutoff_bohr = spec.default_cutoff_bohr
    if nuclear_cutoff_bohr is None:
        nuclear_cutoff_bohr = spec.default_nuclear_cutoff_bohr
    if omega is None:
        omega = spec.default_omega
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    route_options = (
        {"bipole_cutoff_bohr": cutoff_bohr,
         "bipole_nuclear_cutoff_bohr": nuclear_cutoff_bohr,
         "ewald_omega": omega}
        if jk_method == "bipole"
        else {"rsgdf_tail_ke_cutoff": rsgdf_tail_ke_cutoff}
    )

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=basis_name, method_id=method.id,
        kmesh="x".join(str(k) for k in kmesh),
        code="vibeqc", code_version=code_version,
        n_atoms=len(spec.atoms),
    )
    stem = _artifact_stem(artifact_dir)
    job_output = _periodic_job_output_stem(
        artifact_stem=stem,
        log_path=log_path,
    )
    try:
        state = periodic_state_request(spec, method, code="vibeqc", jk_method=jk_method)
    except ValueError as exc:
        row.status = "unavailable"
        row.note = f"vibeqc: state refused before execution: {exc}"
        return row
    if artifact_dir is not None:
        _write_json(
            artifact_dir / "input.json",
            {
                "kind": "periodic",
                "run_id": run_id,
                "target": target,
                "system": asdict(spec),
                "state": state,
                "basis": basis_name,
                "method": _method_payload(method),
                "kmesh": list(kmesh),
                "jk_method": jk_method,
                "route_options": route_options,
                "settings": {
                    "spacing_bohr": spacing_bohr,
                    "cutoff_bohr": cutoff_bohr,
                    "nuclear_cutoff_bohr": nuclear_cutoff_bohr,
                    "omega": omega,
                    "conv_tol_energy": conv_tol_energy,
                    "max_iter": max_iter,
                    "rsgdf_tail_ke_cutoff": rsgdf_tail_ke_cutoff,
                },
                "expected_output_stem": str(stem) if stem is not None else None,
                "production_output_stem": str(job_output),
            },
        )

    plog = ProgressLogger(log_path=log_path, verbose=4)
    plog.banner(
        f"vibe-qc | {spec.id} | {basis_name} | {method.id} | "
        f"kmesh={kmesh} | target={target}"
    )
    plog.write_raw(_format_geometry_block(spec))

    try:
        system = _build_periodic_system(spec)
        row.charge = int(system.charge)
        row.electrostatic_background = state["electrostatic_background"]
        basis = vq.make_basis(system.unit_cell_molecule(), basis_name)
        plog.info(
            f"basis:  {basis_name} / {basis.nbasis} bf / {basis.nshells} shells"
        )
        row.n_basis_functions = int(basis.nbasis)
        row.n_basis_shells = int(basis.nshells)
        # The spec counts all electrons; a scalar ECP removes paired core
        # electrons. Resolve the same basis metadata as the public driver.
        from vibeqc.periodic_runner import _resolve_ecp_data
        _, _, _, total_core = _resolve_ecp_data(system, basis)
        row.n_electrons = int(system.n_electrons()) - total_core
        unpaired = int(system.multiplicity) - 1
        if (total_core < 0 or total_core % 2 or row.n_electrons < unpaired
                or (row.n_electrons - unpaired) % 2):
            raise ValueError("invalid periodic ECP valence electron/spin counts")
        row.n_alpha = (row.n_electrons + unpaired) // 2
        row.n_beta = (row.n_electrons - unpaired) // 2
        plog.info(
            f"system: {len(system.unit_cell)} atoms / {row.n_electrons} explicit electrons "
            f"({total_core} replaced core; alpha {row.n_alpha}, beta {row.n_beta}; "
            f"{state['convention']})"
        )
        if artifact_dir is not None:
            _write_json(artifact_dir / "input_identity.json",
                        _input_identity(system, basis, basis_name, method))

        opts = _make_options(
            method, spec,
            cutoff_bohr=cutoff_bohr,
            nuclear_cutoff_bohr=nuclear_cutoff_bohr,
            conv_tol_energy=conv_tol_energy,
            max_iter=max_iter,
        )

        # These options configure the overlap preflight. The public driver
        # receives the explicit request below and records its resolved options.
        plog.banner("Preflight settings")
        plog.write_raw(format_options(opts, title=type(opts).__name__))
        plog.write_raw(format_options(
            opts.lattice_opts, title="LatticeSumOptions",
        ))

        # Standalone EIGS preflight — pre-SCF overlap diagnostic. The
        # SCF driver also runs scf_preflight_overlap_check; we run
        # eigs_preflight here so the verbose log records the diagnostic
        # *separately* from the SCF startup, with the formatter that
        # ships in vq.format_eigs_report.
        plog.banner("EIGS preflight (pre-SCF overlap diagnostic)")
        eigs = vq.eigs_preflight(
            system, basis,
            lattice_opts=opts.lattice_opts,
        )
        plog.write_raw(vq.format_eigs_report(eigs))
        row.severity = eigs.worst_severity
        row.min_eigval_S = float(eigs.worst_min_eigenvalue)

        # This preflight mesh is only for logging. Attaching symmetry here
        # changes later public-driver admission: scalar BIPOLE ECP currently
        # requires the full mesh, so do not mutate its system into an IBZ job.
        use_mesh_symmetry = not (jk_method == "bipole" and total_core)
        if kmesh == (1, 1, 1):
            kpts = vq.KPoints.gamma(system)
        else:
            if use_mesh_symmetry:
                vq.attach_symmetry(system)
            kpts = vq.KPoints.monkhorst_pack(system, list(kmesh), symmetry=use_mesh_symmetry)
        mesh_kind = "IBZ-reduced" if use_mesh_symmetry else "full-mesh"
        plog.info(f"kpts: {len(kpts)} {mesh_kind} point(s) for mesh {kmesh}")

        plog.banner("SCF")
        t0 = time.perf_counter()
        job_kpoints = None if kmesh == (1, 1, 1) else kmesh
        job_kpoints_label = "Gamma" if job_kpoints is None else f"kpoints={job_kpoints}"
        plog.info(
            "periodic JK route: "
            f"run_periodic_job(jk_method={jk_method!r}, {job_kpoints_label})"
        )
        plog.info(f"periodic route controls: {route_options}")
        if rsgdf_tail_ke_cutoff is not None:
            plog.info(
                "periodic GDF tail: "
                f"rsgdf_tail_ke_cutoff={float(rsgdf_tail_ke_cutoff):.6g}"
            )
        result = vq.run_periodic_job(
            system,
            basis,
            method=method.scf.upper(),
            functional=method.xc,
            jk_method=jk_method,
            kpoints=job_kpoints,
            output=job_output,
            max_iter=max_iter,
            conv_tol_energy=conv_tol_energy,
            damping=spec.default_damping,
            initial_guess=spec.default_initial_guess,
            **({"atomic_spins": state["atomic_spins"]} if state["atomic_spins"] else {}),
            **route_options,
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            output_qvf=bool(write_qvf_artifact),
            progress=plog,
        )
        wall = time.perf_counter() - t0
        # BIPOLE retains its applied ECP core count. A conflicting or missing
        # receipt cannot substantiate the preflight valence state.
        observed_core = getattr(result, "ecp_total_ncore", None)
        if ((observed_core is not None and observed_core != total_core)
                or (jk_method == "bipole" and total_core and observed_core is None)):
            raise ValueError("periodic result ECP core count does not confirm the requested state")

        row.wall_s = wall
        row.converged = bool(getattr(result, "converged", False))
        row.n_iter = int(getattr(result, "n_iter", 0))
        row.energy_ha = float(getattr(result, "energy", float("nan")))
        if len(spec.atoms) > 0 and row.energy_ha is not None:
            row.energy_per_atom_ha = row.energy_ha / len(spec.atoms)
        plog.banner("Result")
        plog.info(
            f"E/cell = {row.energy_ha:.10f} Ha   "
            f"({row.n_iter} iters, wall {wall:.1f} s)"
        )
        backend = str(getattr(result, "backend", "") or "")
        if "+PARITY_HELD" in backend or (jk_method == "gdf" and _gamma_gdf_solid_hold_after_tail(
            system,
            basis,
            spec,
            basis_name,
            method,
            kmesh,
            rsgdf_tail_ke_cutoff,
        )):
            row.status = "unavailable"
            row.note = (
                "vibeqc: Gamma GDF solid absolute-energy parity held for "
                "this STO-3G RKS-LDA lane; use a validated high-G tail or "
                "multi-k route before parity claims"
            )
        elif not row.converged:
            row.note = "vibeqc: not converged within max_iter"

    except Exception as exc:
        _classify_periodic_exception(row, exc)
        plog.banner("EXCEPTION")
        plog.write_raw(traceback.format_exc())

    _finalize_artifacts(
        artifact_dir=artifact_dir,
        log_path=log_path,
        row=row,
        wall_s=row.wall_s,
        qvf_source=(
            job_output.with_suffix(".qvf")
            if write_qvf_artifact
            else None
        ),
    )
    return row


# ---------------------------------------------------------------------------
# Molecular runner
# ---------------------------------------------------------------------------


def _build_molecule(spec: MoleculeSpec) -> "vq.Molecule":
    atoms = []
    for at in spec.atoms:
        x, y, z = at.xyz_ang
        xyz_bohr = (
            x * ANGSTROM_TO_BOHR,
            y * ANGSTROM_TO_BOHR,
            z * ANGSTROM_TO_BOHR,
        )
        atoms.append(vq.Atom(int(at.z), list(xyz_bohr)))
    return vq.Molecule(atoms, charge=int(spec.charge),
                       multiplicity=int(spec.multiplicity))


def _format_molecule_block(spec: MoleculeSpec) -> str:
    lines = [
        f"[geometry: {spec.id}]",
        f"  family            = {spec.family}",
        f"  charge            = {spec.charge}",
        f"  multiplicity      = {spec.multiplicity}",
        f"  n_atoms           = {len(spec.atoms)}",
        "  atoms (Å):",
    ]
    for at in spec.atoms:
        x, y, z = at.xyz_ang
        lines.append(
            f"    Z={at.z:>2}  {at.symbol:>2}  "
            f"({x:9.5f}, {y:9.5f}, {z:9.5f})"
        )
    if spec.notes:
        lines.append(f"  notes             = {spec.notes}")
    if spec.citation:
        lines.append(f"  citation          = {spec.citation}")
    return "\n".join(lines)


def _resolve_aux_basis(method: MethodSpec, basis_name: str, kind: str) -> str:
    """Pick the aux basis for a DF case.

    `method.aux_basis` overrides; otherwise fall back to vibe-qc's
    autodetector (`vq.default_aux_basis_for`) which maps the orbital
    basis to the canonical companion (e.g. ``def2-svp`` →
    ``def2-svp-jkfit`` for kind=``jk``, ``def2-svp-rifit`` for kind=``ri``).
    Bindings raise if `density_fit=True` is set with an empty
    `aux_basis`, so this MUST resolve to something non-empty.
    """
    if getattr(method, "aux_basis", "") and method.aux_basis:
        return method.aux_basis
    try:
        return vq.default_aux_basis_for(basis_name, kind=kind)
    except Exception as exc:
        raise RuntimeError(
            f"runner_vibeqc: cannot autodetect aux basis for "
            f"orbital basis {basis_name!r} (kind={kind!r}): {exc}. "
            f"Set MethodSpec.aux_basis explicitly."
        ) from exc


def _make_molecular_options(method: MethodSpec, basis_name: str, *,
                            conv_tol_energy: float, max_iter: int):
    if method.scf == "rhf":
        opts = vq.RHFOptions()
    elif method.scf == "rks":
        opts = vq.RKSOptions()
        opts.functional = (method.xc or "LDA").upper()
    elif method.scf == "uhf":
        opts = vq.UHFOptions()
    elif method.scf == "uks":
        opts = vq.UKSOptions()
        opts.functional = (method.xc or "LDA").upper()
    else:
        raise NotImplementedError(
            f"runner_vibeqc.run_molecule_case: scf={method.scf!r} "
            f"not implemented in this iteration."
        )
    opts.conv_tol_energy = float(conv_tol_energy)
    opts.max_iter = int(max_iter)
    if method.df:
        opts.density_fit = True
        opts.aux_basis = _resolve_aux_basis(method, basis_name, kind="jk")
    return opts


def _make_mp2_options(method: MethodSpec, basis_name: str):
    """Construct MP2Options / UMP2Options for the post-SCF correlation.

    The MP2 RI aux basis is *different* from the SCF JK aux (RIFIT vs
    JKFIT in the def2 family), so we re-resolve via
    `default_aux_basis_for(..., kind='ri')` rather than reusing the
    SCF aux.
    """
    if method.scf == "rhf":
        opts = vq.MP2Options()
    elif method.scf == "uhf":
        opts = vq.UMP2Options()
    else:
        raise NotImplementedError(
            f"runner_vibeqc.run_molecule_case: MP2 on scf={method.scf!r} "
            f"not wired"
        )
    # The regression runner's ORCA and PySCF comparators are explicitly
    # all-electron.  Keep archived rows on that exact protocol after the
    # public #140 default moves to the published chemical core.
    opts.n_frozen_core = 0
    if method.df:
        opts.density_fit = True
        opts.aux_basis = _resolve_aux_basis(method, basis_name, kind="ri")
    return opts


def run_molecule_case(
    *, run_id: str, target: str, code_version: str, spec: MoleculeSpec,
    basis_name: str, method: MethodSpec,
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    log_path: Path,
    artifact_dir: Optional[Path] = None,
    write_qvf_artifact: bool = False,
) -> CodeRow:
    """Run one vibe-qc molecular SCF case and return its CodeRow."""
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=basis_name, method_id=method.id, kmesh="mol",
        code="vibeqc", code_version=code_version,
        n_atoms=len(spec.atoms),
    )
    stem = _artifact_stem(artifact_dir)
    if artifact_dir is not None:
        _write_json(
            artifact_dir / "input.json",
            {
                "kind": "molecule",
                "run_id": run_id,
                "target": target,
                "system": asdict(spec),
                "basis": basis_name,
                "method": _method_payload(method),
                "settings": {
                    "conv_tol_energy": conv_tol_energy,
                    "max_iter": max_iter,
                },
                "expected_output_stem": str(stem) if stem is not None else None,
            },
        )

    plog = ProgressLogger(log_path=log_path, verbose=4)
    plog.banner(
        f"vibe-qc | {spec.id} | {basis_name} | {method.id} | "
        f"target={target}"
    )
    plog.write_raw(_format_molecule_block(spec))

    try:
        mol = _build_molecule(spec)
        plog.info(
            f"molecule: {len(spec.atoms)} atoms / "
            f"{int(mol.n_electrons())} electrons / "
            f"charge={spec.charge} / mult={spec.multiplicity}"
        )
        row.n_electrons = int(mol.n_electrons())

        basis = vq.make_basis(mol, basis_name)
        plog.info(
            f"basis:    {basis_name} / {basis.nbasis} bf / {basis.nshells} shells"
        )
        row.n_basis_functions = int(basis.nbasis)
        row.n_basis_shells = int(basis.nshells)

        opts = _make_molecular_options(
            method, basis_name,
            conv_tol_energy=conv_tol_energy, max_iter=max_iter,
        )
        plog.banner("Active settings")
        plog.write_raw(format_options(opts, title=type(opts).__name__))
        if method.df:
            plog.info(
                f"density-fit: enabled, aux_basis={opts.aux_basis!r}"
            )

        plog.banner("SCF")
        t0 = time.perf_counter()
        if method.scf == "rhf":
            scf_result = vq.run_rhf(mol, basis, opts)
        elif method.scf == "rks":
            scf_result = vq.run_rks(mol, basis, opts)
        elif method.scf == "uhf":
            scf_result = vq.run_uhf(mol, basis, opts)
        elif method.scf == "uks":
            scf_result = vq.run_uks(mol, basis, opts)
        else:
            raise NotImplementedError(method.scf)
        scf_wall = time.perf_counter() - t0
        scf_converged = bool(getattr(scf_result, "converged", False))
        scf_iter = int(getattr(scf_result, "n_iter", 0))
        scf_energy = float(getattr(scf_result, "energy", float("nan")))
        plog.info(
            f"  SCF E = {scf_energy:.10f} Ha   "
            f"({scf_iter} iters, wall {scf_wall:.2f} s)"
        )

        # Post-SCF (currently MP2 / UMP2). The post-HF call returns the
        # total correlated energy (E_SCF + E_corr); we report that as
        # the row energy so cross-code Δ comparisons line up.
        post_wall = 0.0
        result_energy = scf_energy
        if method.post == "mp2":
            plog.banner("MP2")
            mp2_opts = _make_mp2_options(method, basis_name)
            plog.write_raw(format_options(
                mp2_opts, title=type(mp2_opts).__name__,
            ))
            if method.df:
                plog.info(
                    f"density-fit: enabled (MP2 RI), "
                    f"aux_basis={mp2_opts.aux_basis!r}"
                )
            t0 = time.perf_counter()
            if method.scf == "rhf":
                mp2_result = vq.run_mp2(mol, basis, scf_result, mp2_opts)
            elif method.scf == "uhf":
                mp2_result = vq.run_ump2(mol, basis, scf_result, mp2_opts)
            else:
                raise NotImplementedError(
                    f"runner_vibeqc: MP2 on scf={method.scf!r} not wired"
                )
            post_wall = time.perf_counter() - t0
            # MP2 result objects expose .energy (E_total) and .e_corr;
            # be tolerant of attribute names across vibe-qc versions.
            for cand in ("energy", "e_total", "total_energy"):
                if hasattr(mp2_result, cand):
                    result_energy = float(getattr(mp2_result, cand))
                    break
            else:
                e_corr = float(getattr(mp2_result, "e_corr", 0.0))
                result_energy = scf_energy + e_corr
            plog.info(
                f"  MP2 E = {result_energy:.10f} Ha   "
                f"(wall {post_wall:.2f} s)"
            )

        wall = scf_wall + post_wall
        row.wall_s = wall
        row.converged = scf_converged
        row.n_iter = scf_iter
        row.energy_ha = result_energy
        if len(spec.atoms) > 0:
            row.energy_per_atom_ha = row.energy_ha / len(spec.atoms)
        plog.banner("Result")
        plog.info(
            f"E = {row.energy_ha:.10f} Ha   "
            f"({row.n_iter} SCF iters + post={method.post}, "
            f"total wall {wall:.2f} s)"
        )
        if not row.converged:
            row.note = "vibeqc: SCF did not converge within max_iter"
        if write_qvf_artifact and artifact_dir is not None:
            _write_molecule_qvf_artifact(
                artifact_dir=artifact_dir,
                mol=mol,
                basis=basis,
                basis_name=basis_name,
                method=method,
                scf_result=scf_result,
                wall_s=wall,
                row=row,
            )

    except Exception as exc:
        row.note = f"{type(exc).__name__}: {str(exc)[:160]}"
        plog.banner("EXCEPTION")
        plog.write_raw(traceback.format_exc())

    _finalize_artifacts(
        artifact_dir=artifact_dir,
        log_path=log_path,
        row=row,
        wall_s=row.wall_s,
    )
    return row
