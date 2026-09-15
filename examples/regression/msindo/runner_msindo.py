"""Out-of-process MSINDO reference runner for vibe-qc parity validation.

MSINDO (Bredow/Geudtner/Jug; (C) Mulliken Center for Theoretical Chemistry,
University of Bonn) is executed as an external subprocess and its text output is
parsed independently — vibe-qc never imports MSINDO (CLAUDE.md §10). Build the
oracle binary with ``build_oracle.sh`` and point ``MSINDO_ORACLE`` at it.

Example
-------
    from runner_msindo import run_msindo
    r = run_msindo([("O", 0.0, 0.0, 0.1173),
                    ("H", 0.0, 0.7572, -0.4692),
                    ("H", 0.0, -0.7572, -0.4692)])
    print(r.total_energy)   # -17.0182087674
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Default oracle location (override with $MSINDO_ORACLE).
DEFAULT_ORACLE = os.environ.get("MSINDO_ORACLE", "/tmp/msindo_bin")

_NUM = r"(-?\d+\.\d+)"


@dataclass
class MsindoResult:
    """Parsed scalar results of an MSINDO run (energies in Hartree)."""

    total_energy: float | None = None
    electronic_energy: float | None = None
    binding_energy: float | None = None
    dipole_debye: float | None = None
    homo_lumo_gap_ev: float | None = None
    mp2_correlation_energy: float | None = None
    scf_cycles: int | None = None
    converged: bool = False
    raw_output: str = field(default="", repr=False)


@dataclass
class CcmResult:
    """Parsed results of a periodic CCM MSINDO run (energies in Hartree).

    ``madelung_nuclear_energy`` is MSINDO's "MADELUNG-NUCLEAR ENERGY" — the
    Wigner-Seitz-weighted core–core point-charge sum (long-range Madelung folded
    in unless ``NOEWALD``).  ``ws_cells`` is the per-atom Wigner-Seitz neighbour
    table (1-based origin atom ids + partial-count weights) dumped by
    ``PRINTOPTS=CCMWSC``; ``wrong_neighbors`` flags the ``neighbors.f`` validity
    failure (a non-cyclic cluster)."""

    total_energy: float | None = None
    electronic_energy: float | None = None
    binding_energy: float | None = None
    madelung_nuclear_energy: float | None = None
    n_translations: int | None = None
    cell_length: float | None = None
    scf_cycles: int | None = None
    converged: bool = False
    wrong_neighbors: bool = False
    ws_cells: list[dict] = field(default_factory=list)
    raw_output: str = field(default="", repr=False)


def build_input(
    geometry: list[tuple[str, float, float, float]],
    *,
    title: str = "vibe-qc MSINDO parity run",
    keywords: str = "CARTES RHF",
) -> str:
    """Render an MSINDO Cartesian (Angstrom) input deck.

    Structure: ``===`` closes the (empty) empirical-parameter override block;
    then the title; then the keyword / geometry / variable sections, each
    terminated by ``:END``.
    """
    lines = ["===", title, keywords, ":END"]
    for sym, x, y, z in geometry:
        lines.append(f"{sym:<2} {x:18.10f} {y:18.10f} {z:18.10f}")
    lines.append(":END")  # close geometry section
    lines.append(":END")  # close (empty) variables section
    return "\n".join(lines) + "\n"


def parse_output(text: str) -> MsindoResult:
    """Parse MSINDO stdout into an :class:`MsindoResult`."""

    def _f(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    cyc = re.search(r"SCF CONVERGED IN\s+(\d+)\s+CYCLES", text, re.IGNORECASE)
    return MsindoResult(
        total_energy=_f(r"TOTAL ENERGY\s*=\s*" + _NUM),
        electronic_energy=_f(r"ELECTRONIC ENERGY\s+" + _NUM),
        binding_energy=_f(r"BINDING ENERGY\s*=\s*" + _NUM),
        dipole_debye=_f(r"DIPOLE MOMENT\s*=\s*" + _NUM),
        homo_lumo_gap_ev=_f(r"HOMO-LUMO GAP:\s*" + _NUM),
        # MP2 correlation energy: RHF prints "= …", UHF prints ": …".
        mp2_correlation_energy=_f(r"MP2 CORRELATION ENERGY\s*[=:]\s*" + _NUM),
        scf_cycles=int(cyc.group(1)) if cyc else None,
        converged=cyc is not None,
        raw_output=text,
    )


def run_msindo(
    geometry: list[tuple[str, float, float, float]],
    *,
    oracle: str | os.PathLike[str] = DEFAULT_ORACLE,
    title: str = "vibe-qc MSINDO parity run",
    keywords: str = "CARTES RHF",
    timeout: float = 300.0,
) -> MsindoResult:
    """Run the MSINDO oracle on a Cartesian (Angstrom) geometry and parse it."""
    oracle = Path(oracle)
    if not oracle.exists():
        raise FileNotFoundError(
            f"MSINDO oracle not found at {oracle}. Build it with build_oracle.sh "
            f"and/or set $MSINDO_ORACLE."
        )
    deck = build_input(geometry, title=title, keywords=keywords)
    with tempfile.TemporaryDirectory() as workdir:
        proc = subprocess.run(
            [str(oracle)],
            input=deck,
            capture_output=True,
            text=True,
            cwd=workdir,  # MSINDO writes scratch (fort.*) into cwd
            timeout=timeout,
        )
    return parse_output(proc.stdout + proc.stderr)


# --------------------------------------------------------------------------- #
# Periodic Cyclic Cluster Model (CCM)
# --------------------------------------------------------------------------- #

_VECT_NAMES = ("VECTA", "VECTB", "VECTC")
_CCM_DIM = {1: "CCM1D", 2: "CCM2D", 3: "CCM3D"}


def build_ccm_input(
    real_atoms: list[tuple[str, float, float, float]],
    translations: list[tuple[float, float, float]],
    *,
    title: str = "vibe-qc MSINDO CCM parity run",
    rhf: bool = True,
    ewald: bool = True,
    max_cycles: int | None = 200,
    printopts: str | None = "CCMWSC",
    extra_keywords: str = "",
) -> str:
    """Render a periodic CCM input deck (Cartesian, Angstrom).

    ``real_atoms`` is the cluster supercell; ``translations`` are its 1–3 lattice
    (translation) vectors.  Following MSINDO (``→VECTA,VECTB,VECTC``) the lattice
    vectors are defined by *dummy* atoms: a shared base dummy ``XX`` at the origin
    (atom index ``n+1``) plus one tip dummy per vector (``n+2``, ``n+3``, ``n+4``),
    so ``VECTA(n+1,n+2)`` is the difference ``tip-base``.  The vectors must point
    *outside* the cluster (different from EMBED).  Madelung embedding (Ewald) is
    on by default; pass ``ewald=False`` to emit ``NOEWALD``.
    """
    if not 1 <= len(translations) <= 3:
        raise ValueError("CCM needs 1 (1-D), 2 (2-D) or 3 (3-D) translations")
    n = len(real_atoms)
    base = n + 1
    vparts = [f"{_VECT_NAMES[k]}({base},{n + 2 + k})"
              for k in range(len(translations))]
    kw = ["CARTES", "RHF" if rhf else "UHF", _CCM_DIM[len(translations)]] + vparts
    if not ewald:
        kw.append("NOEWALD")
    if max_cycles:
        kw.append(f"MAXCYC {max_cycles}")
    if printopts:
        kw.append(f"PRINTOPTS={printopts}")
    if extra_keywords:
        kw.append(extra_keywords)
    lines = ["===", title, " ".join(kw), ":END"]
    for sym, x, y, z in real_atoms:
        lines.append(f"{sym:<2} {x:18.10f} {y:18.10f} {z:18.10f}")
    lines.append(f"XX {0.0:18.10f} {0.0:18.10f} {0.0:18.10f}")  # base dummy
    for ax, ay, az in translations:
        lines.append(f"XX {ax:18.10f} {ay:18.10f} {az:18.10f}")  # tip dummy
    lines.append(":END")  # close geometry
    lines.append(":END")  # close (empty) variables
    return "\n".join(lines) + "\n"


def parse_ccm_output(text: str) -> CcmResult:
    """Parse periodic-CCM MSINDO stdout into a :class:`CcmResult`."""

    def _f(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    cyc = re.search(r"SCF CONVERGED IN\s+(\d+)\s+CYCLES", text, re.IGNORECASE)
    ntr = re.search(r"NO\. OF TRANSLATIONS:\s*(\d+)", text)

    # Wigner-Seitz neighbour table (PRINTOPTS=CCMWSC): per atom, a block of
    # integer lines ("atom n1 n2 …", wrapped 10/line) then a block of float
    # weight lines (also wrapped), terminated by "TOTAL WEIGHT:".  3-D cells have
    # 26+ neighbours so both blocks span several lines.  Weights print at F6.2,
    # i.e. only 2 decimals (0.125 → "0.12") — lossy for sub-1/4 shares.
    def _all_int(toks):
        return bool(toks) and all(re.fullmatch(r"-?\d+", t) for t in toks)

    def _all_float(toks):
        return bool(toks) and all(re.fullmatch(r"-?\d+\.\d+", t) for t in toks)

    ws_cells: list[dict] = []
    lines = text.splitlines()
    hdr = next((i for i, l in enumerate(lines)
                if re.search(r"ATOM\s+Q\s+X\s+Y\s+Z\s+NEIGHBORS", l)), None)
    if hdr is not None:
        j, n = hdr + 1, len(lines)
        while j < n:
            if not lines[j].strip():
                j += 1
                continue
            if not _all_int(lines[j].split()):
                break  # left the WS table
            ints: list[int] = []
            while j < n and _all_int(lines[j].split()):
                ints += [int(t) for t in lines[j].split()]
                j += 1
            weights: list[float] = []
            while j < n and _all_float(lines[j].split()):
                weights += [float(t) for t in lines[j].split()]
                j += 1
            ws_cells.append({
                "atom": ints[0], "neighbors": ints[1:], "weights": weights,
            })
            while j < n and lines[j].strip().startswith("TOTAL WEIGHT"):
                j += 1

    return CcmResult(
        total_energy=_f(r"TOTAL ENERGY\s*=\s*" + _NUM),
        electronic_energy=_f(r"ELECTRONIC AND NUCLEAR-POINT-CHARGE ENERGY\s+" + _NUM),
        binding_energy=_f(r"CCM BINDING ENERGY\s*=\s*" + _NUM),
        madelung_nuclear_energy=_f(r"MADELUNG-NUCLEAR ENERGY:\s*" + _NUM),
        n_translations=int(ntr.group(1)) if ntr else None,
        cell_length=_f(r"LENGTH OF CELL:\s*" + _NUM),
        scf_cycles=int(cyc.group(1)) if cyc else None,
        converged=cyc is not None,
        wrong_neighbors="WRONG NUMBER OF NEIGHBORS" in text,
        ws_cells=ws_cells,
        raw_output=text,
    )


def run_ccm(
    real_atoms: list[tuple[str, float, float, float]],
    translations: list[tuple[float, float, float]],
    *,
    oracle: str | os.PathLike[str] = DEFAULT_ORACLE,
    title: str = "vibe-qc MSINDO CCM parity run",
    rhf: bool = True,
    ewald: bool = True,
    max_cycles: int | None = 200,
    printopts: str | None = "CCMWSC",
    extra_keywords: str = "",
    timeout: float = 300.0,
) -> CcmResult:
    """Run the MSINDO oracle on a periodic cyclic cluster and parse it."""
    oracle = Path(oracle)
    if not oracle.exists():
        raise FileNotFoundError(
            f"MSINDO oracle not found at {oracle}. Build it with build_oracle.sh "
            f"and/or set $MSINDO_ORACLE."
        )
    deck = build_ccm_input(
        real_atoms, translations, title=title, rhf=rhf, ewald=ewald,
        max_cycles=max_cycles, printopts=printopts, extra_keywords=extra_keywords,
    )
    with tempfile.TemporaryDirectory() as workdir:
        proc = subprocess.run(
            [str(oracle)],
            input=deck,
            capture_output=True,
            text=True,
            cwd=workdir,
            timeout=timeout,
        )
    return parse_ccm_output(proc.stdout + proc.stderr)


def run_ccm_gradient(
    real_atoms: list[tuple[str, float, float, float]],
    translations: list[tuple[float, float, float]],
    *,
    oracle: str | os.PathLike[str] = DEFAULT_ORACLE,
    ewald: bool = True,
    timeout: float = 300.0,
) -> list[list[float]]:
    """Oracle analytic CCM nuclear gradient (Ha/bohr) at the input geometry.

    Triggers the gradient via ``CARTOPT ANALY GRADONLY`` — MSINDO sets the
    analytic-gradient flag only inside the optimization driver, then ``GRADONLY``
    dumps ``FIRST DERIVATIVES [H/BOHR]`` for the input geometry and stops.
    Returns one ``[dEx, dEy, dEz]`` per real atom."""
    oracle = Path(oracle)
    if not oracle.exists():
        raise FileNotFoundError(f"MSINDO oracle not found at {oracle}.")
    deck = build_ccm_input(real_atoms, translations, ewald=ewald,
                           max_cycles=200, printopts=None,
                           extra_keywords="CARTOPT ANALY GRADONLY")
    with tempfile.TemporaryDirectory() as workdir:
        proc = subprocess.run([str(oracle)], input=deck, capture_output=True,
                              text=True, cwd=workdir, timeout=timeout)
    text = proc.stdout + proc.stderr
    i = text.find("FIRST DERIVATIVES")
    if i < 0:
        raise RuntimeError("no gradient dumped (CARTOPT ANALY GRADONLY)")
    grad: list[list[float]] = []
    row = re.compile(r"\s*\d+\s+\S+\s+" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM)
    for line in text[i:].splitlines()[2:]:
        m = row.match(line)
        if not m:
            if grad:
                break
            continue
        grad.append([float(m.group(k)) for k in (1, 2, 3)])
    return grad


def run_cis_gradient(
    geometry: list[tuple[str, float, float, float]],
    state: int = 0,
    *,
    oracle: str | os.PathLike[str] = DEFAULT_ORACLE,
    n_roots: int = 6,
    title: str = "vibe-qc MSINDO CIS gradient parity",
    timeout: float = 300.0,
) -> list[list[float]]:
    """Oracle analytic CIS **singlet** excited-state gradient (Ha/bohr).

    Triggers the rigorous full-TDA Davidson CIS gradient via the keyword line
    ``CARTES RHF CIS DAVIDSONCIS SROI <n_roots> CISGRAD(<state+2>) GRADONLY``.
    MSINDO numbers the gradient target ``REFSTATE = state + 2`` (REFSTATE 1 is
    the ground state, 2 is S1, …); ``GRADONLY`` dumps ``FIRST DERIVATIVES
    [H/BOHR]`` for the input geometry and stops.

    Singlet only: with ``GRADONLY`` the oracle stops at the singlet gradient,
    which is emitted before the (later) triplet block, so the analytic triplet
    gradient is validated by finite difference instead.  Returns one
    ``[dEx, dEy, dEz]`` per atom."""
    oracle = Path(oracle)
    if not oracle.exists():
        raise FileNotFoundError(f"MSINDO oracle not found at {oracle}.")
    keywords = (
        f"CARTES RHF CIS DAVIDSONCIS SROI {n_roots} "
        f"CISGRAD({state + 2}) GRADONLY"
    )
    deck = build_input(geometry, title=title, keywords=keywords)
    with tempfile.TemporaryDirectory() as workdir:
        proc = subprocess.run([str(oracle)], input=deck, capture_output=True,
                              text=True, cwd=workdir, timeout=timeout)
    text = proc.stdout + proc.stderr
    i = text.find("FIRST DERIVATIVES")
    if i < 0:
        raise RuntimeError("no CIS gradient dumped (CIS DAVIDSONCIS CISGRAD GRADONLY)")
    grad: list[list[float]] = []
    row = re.compile(r"\s*\d+\s+\S+\s+" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM)
    for line in text[i:].splitlines()[2:]:
        m = row.match(line)
        if not m:
            if grad:
                break
            continue
        grad.append([float(m.group(k)) for k in (1, 2, 3)])
    return grad


if __name__ == "__main__":
    import json
    import sys

    ref_path = Path(__file__).with_name("molecular_reference.json")
    data = json.loads(ref_path.read_text())
    print(f"{'molecule':<6} {'parsed TOTAL':>18} {'reference':>18}  match")
    ok = True
    for mol in data["molecules"]:
        geom = [(s, x, y, z) for s, x, y, z in mol["geometry"]]
        mult = mol.get("multiplicity", 1)
        charge = mol.get("charge", 0)
        kw = "CARTES RHF" if mult == 1 else "CARTES UHF"
        if mult != 1:
            kw += f" MULTIP {mult}"
        if charge != 0:
            kw += f" CHARGE {charge}"
        try:
            res = run_msindo(geom, title=mol["name"], keywords=kw)
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(2)
        ref = mol["reference"]["total_energy"]
        got = res.total_energy
        good = got is not None and abs(got - ref) < 1e-6
        ok &= good
        print(f"{mol['name']:<6} {got!s:>18} {ref:>18.10f}  {'OK' if good else 'MISMATCH'}")
    sys.exit(0 if ok else 1)
