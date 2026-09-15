"""GPAW plane-wave-limit atomization-energy reference generator (out-of-process).

This is vibe-qc's sanctioned, productive **VASP replacement** for generating
plane-wave-limit atomization-energy references — the reference data the revised
``pob`` Gaussian basis sets are optimised against (Peintinger & Bredow,
"Approaching the plane wave limit with Gaussian basis sets").

§10 / §1 BOUNDARY (non-negotiable): GPAW is **GPLv3+**. Neither this module nor
anything under ``python/vibeqc/`` may import GPAW or call it in-process — that
would create a GPL derivative of MPL-2.0 vibe-qc. GPAW is executed only in a
*separate* interpreter through :func:`subprocess.run` below; we serialise the
calculation to JSON, read one machine-readable result line back, and reproduce
GPAW's **numbers**, never its code. (Mirrors ``core/runner_gpaw.py``.)

Method notes:
* GPAW is grid/PAW (Blöchl, *Phys. Rev. B* **50**, 17953 (1994); GPAW: Mortensen
  *et al.* *Phys. Rev. B* **71**, 035109 (2005), Enkovaara *et al.* *J. Phys.:
  Condens. Matter* **22**, 253202 (2010)). Its absolute total is PAW-referenced,
  so only **differences** are cross-code comparable. An atomization energy
  ``E_at = Σ E(free atom) − E(solid)/n_fu`` is exactly such a difference — the
  per-atom PAW reference cancels — so it *is* comparable to VASP and to
  experiment.
* r2SCAN (Furness *et al.* *J. Phys. Chem. Lett.* **11**, 8208 (2020)) is a
  meta-GGA: its kinetic-energy density τ carries high Fourier components, so the
  plane-wave total energy converges **slowly** with the cutoff (∝ E_cut^(-3/2)
  asymptotically). Reaching the true PW limit therefore needs either a high
  cutoff or an extrapolation — both provided here.
* Free open-shell atoms are converged spin-polarised with Hund occupations and a
  robust Davidson + small-Mixer setup (plain meta-GGA atom SCF does not
  converge); closed-shell atoms run spin-paired.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .systems import PwSystem

HARTREE_TO_KJMOL = 2625.499639  # CODATA 2018
_RESULT_MARKER = "VIBEQC-PWREF-RESULT:"
_DEFAULT_CONVERGE_TOL_KJMOL = 1.0  # 1 kJ/mol per atom convergence threshold

# vibe-qc functional name -> GPAW/libxc xc string. Meta-GGAs use the explicit
# libxc compound name: GPAW's bare "R2SCAN"/"SCAN" aliases are NOT registered
# in 25.7.0 (NameError), but the libxc "MGGA_X_*+MGGA_C_*" spelling works
# self-consistently in PW mode.
PW_XC: Dict[str, str] = {
    "lda": "LDA",
    "pbe": "PBE",
    "pbesol": "PBEsol",
    "scan": "MGGA_X_SCAN+MGGA_C_SCAN",
    "r2scan": "MGGA_X_R2SCAN+MGGA_C_R2SCAN",
}


# --------------------------------------------------------------------------- #
# External GPAW worker — runs in a SEPARATE interpreter (subprocess only).
# --------------------------------------------------------------------------- #
_GPAW_WORKER = r"""
import json, sys, time, traceback
RESULT_MARKER = "VIBEQC-PWREF-RESULT:"

def emit(p): print(RESULT_MARKER + json.dumps(p, sort_keys=True), flush=True)

def run(payload):
    from ase.build import bulk
    from ase import Atoms
    from ase.units import Hartree
    from gpaw import GPAW, PW
    import gpaw

    xc = payload["xc"]
    cut = float(payload["cutoff_ev"])
    conv_e = float(payload["conv_energy"]) * Hartree
    t0 = time.perf_counter()

    if payload["kind"] == "solid":
        kw = {"a": float(payload["a"])}
        if "c" in payload and payload["c"] is not None:
            kw["c"] = float(payload["c"])
        atoms = bulk(payload["formula"], payload["structure"], **kw)
        k = tuple(int(x) for x in payload["kmesh"])
        calc = GPAW(mode=PW(cut), xc=xc, kpts=k, txt=None,
                    convergence={"energy": conv_e}, maxiter=int(payload["max_iter"]))
        atoms.calc = calc
        e = atoms.get_potential_energy()
        emit({"status": "ok", "energy_ha": e / Hartree, "n_atoms": len(atoms),
              "code_version": getattr(gpaw, "__version__", "unknown"),
              "converged": bool(calc.scf.converged), "wall_s": time.perf_counter() - t0})
        return

    if payload["kind"] == "custom_solid":
        # Non-cubic / non-standard structures built from cell parameters
        # and fractional coordinates.  cellpar = [a, b, c, alpha, beta, gamma]
        # in Angstrom and degrees.  symbols + positions in fractional coords.
        from ase.spacegroup import crystal as _crystal
        symbols = payload["symbols"]
        basis = payload["basis"]           # list of (x, y, z) fractional
        sg = int(payload["spacegroup"])
        cp = [float(x) for x in payload["cellpar"]]  # a, b, c, alpha, beta, gamma
        atoms = _crystal(symbols, basis=basis, spacegroup=sg, cellpar=cp)
        k = tuple(int(x) for x in payload["kmesh"])
        calc = GPAW(mode=PW(cut), xc=xc, kpts=k, txt=None,
                    convergence={"energy": conv_e}, maxiter=int(payload["max_iter"]))
        atoms.calc = calc
        e = atoms.get_potential_energy()
        emit({"status": "ok", "energy_ha": e / Hartree, "n_atoms": len(atoms),
              "code_version": getattr(gpaw, "__version__", "unknown"),
              "converged": bool(calc.scf.converged), "wall_s": time.perf_counter() - t0})
        return

    if payload["kind"] == "atom":
        sym = payload["symbol"]
        L = float(payload["box_ang"])
        hund = int(payload["magmom"]) > 0
        atoms = Atoms(sym, cell=(L, L + 0.3, L + 0.6), pbc=True)
        atoms.center()
        maxiter = int(payload["max_iter"])

        # Elements where the etdm-fdpw direct-minimisation is known to
        # stall (single-electron systems, closed-shell with large Nv, etc.):
        # use the SCF eigensolver instead.
        _DIRECTMIN_FAILS = {"H", "He", "K", "Ca", "Al", "Ga", "Cs", "P", "Ti",
                            "Br"}
        if sym in _DIRECTMIN_FAILS:
            # SCF route: Davidson + mixing + smearing.  Slower per-iteration
            # but robust for these atoms.  Two deliberate deviations from the
            # GPAW defaults (the GPAW-PWREF-001 K/Ca fix):
            #   * full payload maxiter (the old 150-iteration cap starved
            #     K/Ca, whose semicore meta-GGA SCF needs more iterations);
            #   * explicit eigenstates criterion 1e-6 eV^2/e: with the default
            #     4e-8, K/Ca plateau near 1e-7 indefinitely while the energy
            #     is already stable to 1e-6 eV.  The atomization target
            #     precision is 0.1 kJ/mol = 1e-3 eV, three orders above the
            #     energy criterion, so 1e-6 eV^2/e is physically converged.
            #     (Validated: Al reproduces its production energy unchanged.)
            # K additionally suffers intermittent charge-sloshing excursions
            # (energy spikes of 10-400 eV every ~80 iterations) under the
            # default beta=0.05 mixer, and its density residual plateaus at
            # ~1e-3 while energy+eigenstates are fully converged.  Heavier
            # damping (beta 0.02, nmaxold 8) lets it reach a self-consistent
            # fixed point; the density criterion is relaxed to 1.5e-3 for it.
            # (Validated 2026-07-18: two independent converged runs, default
            # vs nbands=-6 Davidson subspace, agree on E(K/r2SCAN/800eV) to
            # 0.04 kJ/mol: -0.1884425 vs -0.1884270 Ha.)
            _SCF_SLOSHY = {"K"}
            if sym in _SCF_SLOSHY:
                mixer = {"backend": "pulay", "beta": 0.02,
                         "nmaxold": 8, "weight": 100}
                convergence = {"energy": conv_e, "density": 1.5e-3,
                               "eigenstates": 1e-6}
            else:
                mixer = {"backend": "pulay", "beta": 0.05,
                         "nmaxold": 5, "weight": 100}
                convergence = {"energy": conv_e, "eigenstates": 1e-6}
            calc = GPAW(mode=PW(cut), xc=xc, nbands=-4,
                        spinpol=hund,
                        mixer=mixer,
                        occupations={"name": "fermi-dirac", "width": 0.001,
                                     "fixmagmom": True},
                        hund=hund,
                        eigensolver={"name": "dav", "niter": 4},
                        symmetry="off", txt=None, maxiter=maxiter,
                        convergence=convergence)
        else:
            # Direct minimisation (etdm-fdpw) — the preferred, robust
            # free-atom recipe.  No density mixing, fixed-uniform occupations,
            # conditional-hund (off for closed-shell).  Validated across
            # the periodic table at 800-1000 eV.
            calc = GPAW(mode=PW(cut), xc=xc, nbands=-4,
                        mixer={"backend": "no-mixing"},
                        occupations={"name": "fixed-uniform"},
                        hund=hund, eigensolver={"name": "etdm-fdpw"},
                        symmetry="off", txt=None, maxiter=maxiter,
                        convergence={"energy": conv_e})
        atoms.calc = calc
        e = atoms.get_potential_energy()  # raises if it fails to converge
        emit({"status": "ok", "energy_ha": e / Hartree,
              "code_version": getattr(gpaw, "__version__", "unknown"),
              "magmom": float(atoms.get_magnetic_moment()),
              "n_valence": int(calc.setups[0].Nv),
              "converged": bool(getattr(calc.scf, "converged", True)),
              "wall_s": time.perf_counter() - t0})
        return

    raise ValueError("unknown kind %r" % payload["kind"])

def main():
    payload = json.loads(sys.stdin.read())
    try:
        run(payload)
    except ModuleNotFoundError as exc:
        if exc.name in ("gpaw", "ase"):
            emit({"status": "unavailable", "note": str(exc)})
        else:
            emit({"status": "error", "note": "%s: %s" % (type(exc).__name__, exc),
                  "traceback": traceback.format_exc()})
    except Exception as exc:
        emit({"status": "error", "note": "%s: %s" % (type(exc).__name__, str(exc)[:200]),
              "traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
"""


def _gpaw_python() -> str:
    """Interpreter that runs GPAW. ``VIBEQC_GPAW_PYTHON`` overrides; default is
    the current one (still a separate *process*, so the §10 boundary holds)."""
    return os.environ.get("VIBEQC_GPAW_PYTHON", sys.executable)


def _run_external(payload: dict, log: Optional[Path] = None) -> dict:
    # Inherit parent environment so PYTHONPATH (for gpaw source checkout)
    # and OMP/threading variables reach the worker subprocess.
    env = os.environ.copy()
    proc = subprocess.run(
        [_gpaw_python(), "-c", _GPAW_WORKER],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
    )
    if log is not None:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(
                f"\n# {payload.get('kind')} {payload.get('formula', payload.get('symbol', ''))} "
                f"cut={payload.get('cutoff_ev')}\n"
            )
            if proc.stdout:
                fh.write(proc.stdout)
            if proc.stderr:
                fh.write("# stderr:\n" + proc.stderr)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            res = json.loads(line[len(_RESULT_MARKER) :])
            res.setdefault("returncode", proc.returncode)
            return res
    if (
        "No module named 'gpaw'" in proc.stderr
        or "No module named 'ase'" in proc.stderr
    ):
        return {"status": "unavailable", "note": "gpaw/ase not importable"}
    return {
        "status": "error",
        "note": f"no result marker (rc={proc.returncode}): {proc.stderr[-300:]}",
    }


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CutoffPoint:
    cutoff_ev: float
    e_solid_ha: float
    e_atoms_ha: float  # Σ free-atom energies for one formula unit
    atomization_kjmol: float  # per formula unit
    solid_converged: bool
    atoms_converged: bool


@dataclass(frozen=True)
class PwAtomizationResult:
    system_id: str
    functional: str
    points: Tuple[CutoffPoint, ...]
    pw_limit_kjmol: float  # cutoff-extrapolated atomization, kJ/mol/f.u.
    extrapolation: str  # how pw_limit_kjmol was obtained
    atom_energies_ha: Dict[str, float]
    atom_valence: Dict[str, int]
    vasp900_kjmol: Optional[float]
    code_version: str
    kmesh: Tuple[int, int, int]
    atom_box_ang: float
    geometry_a_ang: float
    wall_s: float

    @property
    def delta_vs_vasp_kjmol(self) -> Optional[float]:
        if self.vasp900_kjmol is None:
            return None
        return self.pw_limit_kjmol - self.vasp900_kjmol

    @property
    def highest_cutoff_kjmol(self) -> float:
        return self.points[-1].atomization_kjmol


def _extrapolate_pw_limit(points: Sequence[CutoffPoint]) -> Tuple[float, str]:
    """Plane-wave-limit atomization energy.

    The PW basis-set error of the total energy decays asymptotically as
    E_cut^(-3/2); the same form is fit to the atomization energy using the two
    highest cutoffs. With one cutoff, the single value is returned as-is.
    """
    if len(points) < 2:
        return points[-1].atomization_kjmol, "single-cutoff (no extrapolation)"
    p1, p2 = points[-2], points[-1]
    x1, x2 = p1.cutoff_ev**-1.5, p2.cutoff_ev**-1.5
    slope = (p1.atomization_kjmol - p2.atomization_kjmol) / (x1 - x2)
    e_inf = p2.atomization_kjmol - slope * x2
    return (
        e_inf,
        f"E_cut^(-3/2) extrapolation from {int(p1.cutoff_ev)}/{int(p2.cutoff_ev)} eV",
    )


# Free-atom energies recur across compounds (O in MgO, Al2O3, BeO, …), so cache
# them per (element, functional, cutoff, box) for the lifetime of the process —
# a full-set run then computes each atom once. Clear with _ATOM_CACHE.clear().
_ATOM_CACHE: Dict[tuple, dict] = {}


def _atom_energy(
    sym: str,
    magmom: int,
    cutoff: float,
    xc: str,
    box_ang: float,
    conv_energy: float,
    max_iter: int,
    log: Optional[Path] = None,
) -> dict:
    """Out-of-process GPAW free-atom result dict, memoised across systems.

    ``magmom`` is the Hund unpaired-electron count; it selects the worker's
    open- vs closed-shell path (it is fixed per element, so it stays out of the
    cache key)."""
    key = (sym, xc, round(cutoff, 3), round(box_ang, 3))
    hit = _ATOM_CACHE.get(key)
    if hit is not None:
        return hit
    res = _run_external(
        {
            "kind": "atom",
            "symbol": sym,
            "magmom": magmom,
            "box_ang": box_ang,
            "xc": xc,
            "cutoff_ev": cutoff,
            "conv_energy": conv_energy,
            "max_iter": max_iter,
        },
        log=log,
    )
    if res.get("status") == "ok":
        _ATOM_CACHE[key] = res
    return res


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compute_pw_atomization(
    system: PwSystem,
    *,
    functional: str = "r2scan",
    cutoffs_ev: Sequence[float] = (800.0, 1000.0, 1200.0),
    kmesh: Tuple[int, int, int] = (6, 6, 6),
    atom_box_ang: float = 12.0,
    conv_energy_ha: float = 1e-6,
    max_iter: int = 500,
    log: Optional[Path] = None,
) -> PwAtomizationResult:
    """Compute the plane-wave-limit atomization energy of ``system`` with GPAW.

    For each cutoff the solid and every free atom are evaluated at the *same*
    cutoff/setups (so the PAW reference cancels), the per-formula-unit
    atomization energy is formed, and the series is extrapolated to the PW limit.
    All GPAW work happens out-of-process (§10). Returns a
    :class:`PwAtomizationResult`; raises ``RuntimeError`` if GPAW is unavailable
    or a calculation fails.
    """
    fkey = functional.lower()
    if fkey not in PW_XC:
        raise ValueError(f"unknown functional {functional!r}; have {sorted(PW_XC)}")
    xc = PW_XC[fkey]
    cutoffs = [float(c) for c in cutoffs_ev]
    t_start = time.perf_counter()

    code_version = "unknown"

    points: List[CutoffPoint] = []
    atom_e_at_top: Dict[str, float] = {}
    atom_valence: Dict[str, int] = {}
    for ci, cut in enumerate(cutoffs):
        print(f"  [{ci + 1}/{len(cutoffs)}] cutoff {cut:.0f} eV ...", flush=True)
        if system.structure == "custom" and system.custom_kwargs is not None:
            solid = _run_external(
                {
                    "kind": "custom_solid",
                    "symbols": system.custom_kwargs["symbols"],
                    "basis": system.custom_kwargs["basis"],
                    "spacegroup": system.custom_kwargs["spacegroup"],
                    "cellpar": system.custom_kwargs["cellpar"],
                    "kmesh": list(kmesh),
                    "xc": xc,
                    "cutoff_ev": cut,
                    "conv_energy": conv_energy_ha,
                    "max_iter": max_iter,
                },
                log=log,
            )
        else:
            solid = _run_external(
                {
                    "kind": "solid",
                    "formula": system.formula,
                    "structure": system.structure,
                    "a": system.a,
                    "c": system.c,
                    "kmesh": list(kmesh),
                    "xc": xc,
                    "cutoff_ev": cut,
                    "conv_energy": conv_energy_ha,
                    "max_iter": max_iter,
                },
                log=log,
            )
        if solid["status"] != "ok":
            raise RuntimeError(
                f"GPAW solid {system.id} @ {cut} eV: "
                f"{solid.get('status')}: {solid.get('note')}"
            )
        code_version = solid.get("code_version", code_version)
        e_solid = float(solid["energy_ha"])

        e_atoms = 0.0
        atoms_conv = True
        for sym, count in system.composition:
            res = _atom_energy(
                sym,
                system.magmom(sym),
                cut,
                xc,
                atom_box_ang,
                conv_energy_ha,
                max_iter,
                log=log,
            )
            if res["status"] != "ok":
                raise RuntimeError(
                    f"GPAW atom {sym} @ {cut} eV: "
                    f"{res.get('status')}: {res.get('note')}"
                )
            e_atoms += count * float(res["energy_ha"])  # respect stoichiometry
            atoms_conv = atoms_conv and bool(res.get("converged", False))
            atom_e_at_top[sym] = float(res["energy_ha"])
            atom_valence[sym] = int(res.get("n_valence", 0))

        # E_at per formula unit; E_atoms already sums one formula unit's atoms
        atomization_ha = e_atoms - e_solid / system.n_fu_per_cell
        points.append(
            CutoffPoint(
                cutoff_ev=cut,
                e_solid_ha=e_solid,
                e_atoms_ha=e_atoms,
                atomization_kjmol=atomization_ha * HARTREE_TO_KJMOL,
                solid_converged=bool(solid.get("converged", False)),
                atoms_converged=atoms_conv,
            )
        )

    pw_limit, how = _extrapolate_pw_limit(points)
    return PwAtomizationResult(
        system_id=system.id,
        functional=fkey,
        points=tuple(points),
        pw_limit_kjmol=pw_limit,
        extrapolation=how,
        atom_energies_ha=atom_e_at_top,
        atom_valence=atom_valence,
        vasp900_kjmol=system.vasp900_kjmol,
        code_version=code_version,
        kmesh=tuple(kmesh),
        atom_box_ang=atom_box_ang,
        geometry_a_ang=system.a,
        wall_s=time.perf_counter() - t_start,
    )


# --------------------------------------------------------------------------- #
# Per-element cutoff-convergence check
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ElementConvergence:
    """Cutoff-convergence report for a single element."""

    symbol: str
    functional: str
    energies_kjmol: Dict[float, float]  # cutoff_ev → atom energy (kJ/mol)
    converged: bool  # True if Δ between two highest cutoffs < threshold
    delta_kjmol: float  # Δ between two highest cutoffs
    recommendation: str  # human-readable verdict


def check_element_convergence(
    symbols: Sequence[str],
    magmoms: Dict[str, int],
    *,
    functional: str = "r2scan",
    cutoffs_ev: Sequence[float] = (800.0, 1000.0, 1200.0),
    atom_box_ang: float = 12.0,
    conv_energy_ha: float = 1e-6,
    max_iter: int = 500,
    threshold_kjmol: float = _DEFAULT_CONVERGE_TOL_KJMOL,
    log: Optional[Path] = None,
) -> Tuple[ElementConvergence, ...]:
    """Check per-element free-atom cutoff convergence.

    For each element in *symbols*, compute the free-atom energy at every cutoff
    and report whether the two highest cutoffs agree within *threshold_kjmol*.
    Elements that fail the check likely need a higher PW cutoff in production —
    heavier atoms (Cl, S, Ga, As, …) and those with semicore PAW setups are
    the most common offenders.
    """
    fkey = functional.lower()
    if fkey not in PW_XC:
        raise ValueError(f"unknown functional {functional!r}; have {sorted(PW_XC)}")
    xc = PW_XC[fkey]
    cutoffs = sorted(float(c) for c in cutoffs_ev)

    results: List[ElementConvergence] = []
    for i, sym in enumerate(symbols):
        print(f"  [{i + 1}/{len(symbols)}] {sym} ...", flush=True)
        magmom = magmoms.get(sym, 0)
        energies: Dict[float, float] = {}
        for cut in cutoffs:
            res = _atom_energy(
                sym, magmom, cut, xc, atom_box_ang, conv_energy_ha, max_iter, log=log
            )
            if res["status"] != "ok":
                raise RuntimeError(
                    f"GPAW atom {sym} @ {cut} eV (convergence check): "
                    f"{res.get('status')}: {res.get('note')}"
                )
            energies[cut] = float(res["energy_ha"]) * HARTREE_TO_KJMOL

        # convergence: compare the two highest cutoffs
        hi, lo = cutoffs[-1], cutoffs[-2]
        delta = abs(energies[hi] - energies[lo])
        ok = delta < threshold_kjmol
        if ok:
            rec = f"{sym} converged to {delta:.2f} kJ/mol at {hi:.0f} eV (Nv={res.get('n_valence', '?')})"
        else:
            rec = (
                f"{sym} NOT converged: Δ={delta:.2f} kJ/mol between "
                f"{lo:.0f} and {hi:.0f} eV (Nv={res.get('n_valence', '?')}) — "
                f"consider higher cutoff"
            )
        print(f"    → {rec}", flush=True)
        results.append(
            ElementConvergence(
                symbol=sym,
                functional=fkey,
                energies_kjmol=energies,
                converged=ok,
                delta_kjmol=delta,
                recommendation=rec,
            )
        )
        # Clear per-element cache entry so a higher cutoff can be re-run fresh.
        # Keep the cache for other elements/systems unchanged.

    return tuple(results)
