"""Run the ORCA molecular reference via ASE's OrcaCalculator.

ORCA is *the* canonical molecular-QC reference for vibe-qc's parity
tests. We drive it through ``vibeqc.benchmark.make_orca_calculator``
which already handles binary discovery (``ASE_ORCA_COMMAND``,
``ORCA_PATH``, ``$PATH``) and graceful "not installed" fall-throughs.

ORCA is molecule-only here — periodic ORCA needs the experimental
``%pal`` block + tight cell input that vibe-qc's regression suite
isn't trying to drive yet. Wave 2+ may add it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional

from .case import CodeRow
from .spec import MethodSpec, MoleculeSpec

EV_TO_HARTREE = 1.0 / 27.211386245988
_ORCA_J_AUX = "def2/J"
_ORCA_JK_AUX = "def2/JK"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# Map (scf, xc) → ORCA simple-input keyword.
_ORCA_FUNC_MAP = {
    ("rhf", None):     "HF",
    ("uhf", None):     "HF",                # ORCA picks UHF from %scf
    ("rks", "lda"):    "LDA",               # ORCA's LDA == Slater + VWN5
    ("rks", "pbe"):    "PBE",
    ("rks", "blyp"):   "BLYP",
    # B3LYP flavor pairing: vibe-qc's bare "b3lyp" (and explicit
    # "b3lyp5") is the ORCA/VWN5 flavor == ORCA's *bare* "B3LYP";
    # vibe-qc's "b3lyp/g" / "b3lypg" == ORCA's "B3LYP/G" (the
    # Gaussian/VWN-RPA variant). ORCA expresses both flavors.
    ("rks", "b3lyp"):   "B3LYP",
    ("rks", "b3lyp5"):  "B3LYP",
    ("rks", "b3lyp/g"): "B3LYP/G",
    ("rks", "b3lypg"):  "B3LYP/G",
    ("uks", "lda"):    "LDA",
    ("uks", "pbe"):    "PBE",
    ("uks", "blyp"):   "BLYP",
    ("uks", "b3lyp"):   "B3LYP",
    ("uks", "b3lyp5"):  "B3LYP",
    ("uks", "b3lyp/g"): "B3LYP/G",
    ("uks", "b3lypg"):  "B3LYP/G",
    # Double-hybrid functionals — ORCA handles the SCF + RI-MP2
    # correction internally, but needs an auxiliary basis for the
    # correlation step.  AutoAux (Stoychev 2017) supplies one
    # automatically; without it ORCA 6.1 exits with
    #   ERROR: RI-MP2 needs an AuxC basis but none was defined!
    # BUG 36 / BUG 36 regress (QCIL wave-041).
    ("rks", "b2plyp"):      "B2PLYP",
    ("rks", "dsd-pbep86"):   "DSD-PBEP86",
    ("rks", "pwpb95"):       "PWPB95",
    ("uks", "b2plyp"):      "B2PLYP",
    ("uks", "dsd-pbep86"):   "DSD-PBEP86",
    ("uks", "pwpb95"):       "PWPB95",
}


def _orca_simpleinput(method: MethodSpec, basis_name: str) -> str:
    if method.post == "mp2":
        # ORCA: "MP2" or "RI-MP2" simple-input keyword. UHF reference
        # picked automatically when multiplicity > 1.
        # NoFrozenCore: pin to all-electron MP2 to match PySCF and the
        # explicit vibe-qc regression recipe; ORCA's bare MP2 freezes the
        # 1s on heavy atoms which
        # would introduce a ~10⁻⁴ Ha cross-code Δ at sto-3g.
        method_kw = "RI-MP2" if method.df else "MP2"
        parts = [method_kw, basis_name, "TightSCF", "NoFrozenCore", "SP"]
        if method.df:
            parts.append("def2/JK def2-SVP/C")
        return " ".join(parts)

    if method.post in ("ccsd", "ccsd(t)"):
        # ORCA CCSD(T) simple-input.  UHF reference picked automatically
        # when multiplicity > 1.  NoFrozenCore pins all-electron
        # correlation.  The %mdci MaxIter block goes in the orcablocks
        # side channel (run_molecule_case) — ORCA's 50-iteration CC
        # default is too tight for cyclopropene and similar systems.
        # BUG 109.
        method_kw = "CCSD(T)"
        parts = [method_kw, basis_name, "TightSCF", "NoFrozenCore", "SP"]
        return " ".join(parts)

    key = (method.scf, (method.xc or "").lower() or None)
    func_kw = _ORCA_FUNC_MAP.get(key)
    if func_kw is None:
        raise NotImplementedError(
            f"runner_orca: no ORCA simpleinput mapping for "
            f"scf={method.scf!r} xc={method.xc!r}"
        )
    # TightSCF == ORCA's predefined ~1e-8 conv tolerance set; ample for
    # the µHa-tolerance molecular parity tests.
    # SP requests a single-point energy; no gradients, no opt.
    parts = [func_kw, basis_name, "TightSCF", "SP"]
    if method.df:
        if _method_needs_hf_exchange(method):
            # RIJK = density-fitting on both J (Coulomb) and K (exchange);
            # it needs ORCA's universal JK auxiliary basis.
            parts.insert(1, "RIJK")
            parts.append(_ORCA_JK_AUX)
        else:
            # Pure GGAs have no K build. ORCA accepts RI (not RIJ) as
            # the simple-input request for the Coulomb fit, paired with
            # the J-only def2 auxiliary slot.
            parts.insert(1, "RI")
            parts.append(_ORCA_J_AUX)
    if _is_double_hybrid(method):
        # ORCA requires an auxiliary basis for the RI-MP2 correlation
        # step in double-hybrid functionals.  AutoAux (Stoychev, Auer,
        # Neese, JCTC 13, 554 (2017)) generates one automatically.  The
        # SCF-side DF aux (def2/JK above) covers the JK fitting; the
        # /C (correlation-fitting) aux is separate and ORCA 6.1 raises
        # an error when it is missing.  BUG 36 / BUG 36 regress.
        # vibe-qc's calibrated double-hybrid model component is explicitly
        # all-electron even though standalone MP2 now defaults to the
        # published chemical core (#140). Keep the ORCA side on that same
        # model definition instead of inheriting ORCA's frozen-core default.
        parts.append("AutoAux")
        parts.append("NoFrozenCore")
    return " ".join(parts)


def _orca_blocks(method: MethodSpec, max_iter: int) -> str:
    """Return block-form controls for one molecular reference deck.

    ``NoFrozenCore`` remains on correlated simple-input lines, but ORCA 6.1
    can ignore that shorthand for canonical UMP2 on some open-shell
    topologies. The explicit ``%method`` selector is therefore the protocol
    authority for every MP2/CC or double-hybrid reference.
    """

    blocks = f"%scf\n  MaxIter {int(max_iter)}\nend\n"
    if method.post in ("mp2", "ccsd", "ccsd(t)") or _is_double_hybrid(
        method
    ):
        blocks += "\n%method\n  FrozenCore FC_NONE\nend\n"
    if method.post in ("ccsd", "ccsd(t)"):
        # ORCA's 50-iteration CC default is too tight: cyclopropene
        # CCSD(T)/def2-TZVP hits the wall at residual >0.003 (BUG 109).
        blocks += "\n%mdci\n  MaxIter 100\nend\n"
    return blocks


_DOUBLE_HYBRID_XC = {"b2plyp", "dsd-pbep86", "pwpb95"}


def _is_double_hybrid(method: MethodSpec) -> bool:
    """Return True when *method* is a double-hybrid functional.

    Double hybrids run an SCF step with a fraction of exact exchange
    followed by a post-SCF MP2 correlation correction; ORCA's
    simple-input keyword (B2PLYP / DSD-PBEP86 / PWPB95) dispatches
    both internally but needs an auxiliary basis for the RI-MP2 step.
    """
    xc = (method.xc or "").lower()
    return xc in _DOUBLE_HYBRID_XC


def _method_needs_hf_exchange(method: MethodSpec) -> bool:
    if method.scf in ("rhf", "uhf"):
        return True
    xc = (method.xc or "").lower()
    return xc in {"b3lyp", "b3lyp5", "b3lyp/g", "b3lypg", "pbe0"} or _is_double_hybrid(method)


def _orca_version(orca_command: str) -> str:
    """Best-effort ORCA version probe (ORCA's own output is verbose)."""
    try:
        # ORCA prints "ORCA <version>  -" on the first non-blank line of
        # any run; cheapest probe is "orca <empty>" which prints help+ver
        # to stdout and exits non-zero. Capture and parse.
        proc = subprocess.run(
            [orca_command],
            capture_output=True, text=True, timeout=5,
        )
        for line in (proc.stdout + proc.stderr).splitlines():
            if "Program Version" in line or "ORCA" in line and "-" in line:
                return line.strip()[:80]
    except Exception:
        pass
    # Fall back: file basename encodes version (e.g. orca_6_1_1_*)
    return Path(orca_command).parent.name or "unknown"


def run_molecule_case(
    *, run_id: str, target: str, spec: MoleculeSpec, basis_name: str,
    method: MethodSpec,
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    log_path: Path,
    workdir: Path,
    artifact_dir: Optional[Path] = None,
) -> CodeRow:
    """Run one ORCA molecular case via ASE's OrcaCalculator.

    Returns a CodeRow with status='unavailable' if ORCA isn't found —
    so the suite stays runnable on machines without ORCA installed.
    """
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=basis_name, method_id=method.id, kmesh="mol",
        code="orca", code_version="unknown",
        n_atoms=len(spec.atoms),
        n_electrons=sum(at.z for at in spec.atoms) - int(spec.charge),
    )
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            artifact_dir / "input.json",
            {
                "kind": "molecule",
                "run_id": run_id,
                "target": target,
                "system_id": spec.id,
                "atoms": [
                    {"symbol": at.symbol, "z": at.z, "xyz_ang": at.xyz_ang}
                    for at in spec.atoms
                ],
                "charge": spec.charge,
                "multiplicity": spec.multiplicity,
                "basis": basis_name,
                "method": {
                    "id": method.id,
                    "scf": method.scf,
                    "xc": method.xc,
                    "post": method.post,
                    "df": method.df,
                },
                "conv_tol_energy": conv_tol_energy,
                "max_iter": max_iter,
            },
        )
        for name in ("stdout.log", "stderr.log"):
            (artifact_dir / name).write_text("", encoding="utf-8")

    def _log(text: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
            fh.flush()

    _log("\n" + "=" * 78)
    _log(
        f"  orca | {spec.id} | {basis_name} | {method.id} | mol | "
        f"target={target}"
    )
    _log("=" * 78)

    try:
        from vibeqc.benchmark import (                            # noqa: WPS433
            find_orca_command, make_orca_calculator,
        )
    except ImportError as exc:
        row.status = "unavailable"
        row.note = f"vibeqc.benchmark missing: {exc}"
        _log(row.note)
        return row

    orca_cmd = find_orca_command()
    if orca_cmd is None:
        row.status = "unavailable"
        row.note = (
            "orca binary not found via $ASE_ORCA_COMMAND / $ORCA_COMMAND / "
            "$ORCA_PATH / $PATH"
        )
        _log(row.note)
        return row
    row.code_version = _orca_version(orca_cmd)
    _log(f"  orca binary: {orca_cmd}")
    _log(f"  orca version: {row.code_version}")

    try:
        from ase import Atoms                                     # noqa: WPS433
    except ImportError as exc:
        row.status = "unavailable"
        row.note = f"ase not installed: {exc}"
        _log(row.note)
        return row

    try:
        atoms = Atoms(
            symbols=[at.symbol for at in spec.atoms],
            positions=[list(at.xyz_ang) for at in spec.atoms],
            charges=([float(spec.charge)]
                     + [0.0] * (len(spec.atoms) - 1)) if spec.charge else None,
        )

        simple = _orca_simpleinput(method, basis_name)
        # MaxIter goes in %scf; charge / multiplicity go on the *xyz line
        # (ORCA's preferred path for ground-state SCF; see ORCA manual
        # § 3.2 "Simple input").
        scfblock = _orca_blocks(method, max_iter)

        case_workdir = (
            artifact_dir
            if artifact_dir is not None
            else workdir / f"{spec.id}__{basis_name}__{method.id}"
        )
        case_workdir.mkdir(parents=True, exist_ok=True)
        old_cwd = os.getcwd()
        try:
            os.chdir(case_workdir)
            calc = make_orca_calculator(
                orcasimpleinput=simple,
                orcablocks=scfblock,
                orca_command=orca_cmd,
                label="orca",
                charge=int(spec.charge),
                mult=int(spec.multiplicity),
            )
            if calc is None:
                row.status = "unavailable"
                row.note = "make_orca_calculator returned None"
                _log(row.note)
                return row
            atoms.calc = calc
            _log(f"  orca simpleinput: {simple}")
            t0 = time.perf_counter()
            e_eV = atoms.get_potential_energy()
            wall = time.perf_counter() - t0
            row.wall_s = wall
            row.energy_ha = float(e_eV) * EV_TO_HARTREE
            row.converged = True
            if len(spec.atoms) > 0:
                row.energy_per_atom_ha = row.energy_ha / len(spec.atoms)
            _log(
                f"  E = {row.energy_ha:.10f} Ha   "
                f"(wall {wall:.2f} s, e_eV={e_eV:.6f})"
            )
        finally:
            os.chdir(old_cwd)

    except Exception as exc:
        row.status = "error"
        row.note = f"{type(exc).__name__}: {str(exc)[:160]}"
        _log(f"  EXCEPTION:\n{traceback.format_exc()}")

    if artifact_dir is not None:
        _write_json(artifact_dir / "parsed.json", row.__dict__)
    return row
