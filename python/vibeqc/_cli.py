"""``vibeqc`` CLI -- v0.12-prep starter, GPAW-equivalent surface.

GPAW ships ``gpaw info`` / ``gpaw gpw <file>`` / etc.  This module
provides the equivalent ``vibeqc`` command (no hyphen -- distinct from
the user-facing ``vibe-qc`` brand CLI in :mod:`vibeqc.cli`):

* ``vibeqc info`` -- print vibeqc version, linked library versions,
  Python + platform info.
* ``vibeqc gpw <restart.npz>`` -- summarise a GPW restart file
  (kind, energy, converged, n_iter, basis, atoms, lattice, file size)
  plus the Lippert-Hutter (1997) GPW citation pointer.
* ``vibeqc dos <restart.npz> [--sigma=0.01] [--bins=200]`` -- compute
  Gaussian-broadened DOS from a GPW restart and print
  ``(energy_Ha, dos)`` pairs to stdout.
* ``vibeqc run <script.py>`` -- execute a user Python script with
  ``vibeqc`` preloaded in the namespace (equivalent to
  ``.venv/bin/python <script>`` for users who don't want to remember
  the venv path).
* ``vibeqc opt <structure.xyz> [--basis ...] [--functional ...]
  [--cutoff ...]`` -- read an XYZ via ASE, wrap in a large cubic box,
  run a BFGS geometry optimisation (max 20 steps, fmax = 0.05) under
  :class:`vibeqc.VibeqcGPW`, and write the optimised geometry to
  ``<structure>.opt.xyz``.
* ``vibeqc bands <restart.npz> [--k-path "G,X,M,G"] [--n-points N]``
  -- load a GPW restart, build a (cubic) high-symmetry k-path, and
  print eigenvalues at every k-point.

Wired into ``pyproject.toml`` as::

    [project.scripts]
    vibeqc = "vibeqc._cli:main"
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from typing import Sequence

import numpy as np

_LIPPERT_CITATION = (
    "Cite the GPW route with Lippert & Hutter (1997), DOI 10.1080/00268979709482119"
)


# ---------------------------------------------------------------------------
# vibeqc info
# ---------------------------------------------------------------------------


def _cmd_info(args: argparse.Namespace) -> int:
    """Print version + linked-library + platform information."""
    import vibeqc

    print(f"vibeqc {vibeqc.__version__}")
    print(f"Python {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")
    print(f"Package: {os.path.dirname(vibeqc.__file__)}")

    print("")
    print("Linked libraries:")
    try:
        versions = vibeqc.library_versions()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  (library_versions() probe failed: {exc!r})")
        versions = {}

    # Stable order: project first, then the native deps.
    preferred = [
        "vibe-qc",
        "libint",
        "libxc",
        "spglib",
        "libecpint",
        "fftw3",
        "blas",
        "dftd3",
        "dftd4",
    ]
    seen: set[str] = set()
    for key in preferred:
        if key in versions:
            print(f"  {key:<10s} {versions[key]}")
            seen.add(key)
    for key, value in versions.items():
        if key in seen or key == "vibeqc":  # vibeqc is an alias
            continue
        print(f"  {key:<10s} {value}")

    return 0


# ---------------------------------------------------------------------------
# vibeqc gpw <restart.npz>
# ---------------------------------------------------------------------------


def _format_lattice(lattice: np.ndarray) -> str:
    rows = []
    for row in np.asarray(lattice, dtype=float):
        rows.append(f"    [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}]")
    return "\n".join(rows)


def _z_to_symbol(z: int) -> str:
    # Minimal periodic-table lookup. Cover the common QC range; fall
    # back to "Z=<n>" for anything outside.
    table = [
        "X",  # 0 placeholder
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
    ]
    if 0 <= z < len(table):
        return table[z]
    return f"Z={z}"


def _cmd_gpw(args: argparse.Namespace) -> int:
    """Summarise a GPW restart file."""
    from vibeqc import load_gpw_result

    path = args.path
    if not os.path.exists(path):
        print(f"vibeqc gpw: no such file: {path}", file=sys.stderr)
        return 2

    data = load_gpw_result(path)
    size_bytes = os.path.getsize(path)

    print(f"GPW restart file: {path}")
    print(f"  file size:    {size_bytes} bytes ({size_bytes / 1024.0:.1f} KiB)")
    print(f"  kind:         {data['kind']}")
    print(f"  energy:       {data['energy']:.10f} Ha")
    print(f"  converged:    {data['converged']}")
    print(f"  n_iter:       {data['n_iter']}")
    print(f"  basis_name:   {data['basis_name']}")

    functional = data.get("breakdown_functional") or ""
    print(f"  functional:   {functional or '(HF / none)'}")

    Z = np.asarray(data["unit_cell_Z"])
    xyz = np.asarray(data["unit_cell_xyz"])
    print(f"  n_atoms:      {Z.shape[0]}")
    print("  atoms (symbol  x  y  z, bohr):")
    for z, pos in zip(Z, xyz):
        sym = _z_to_symbol(int(z))
        print(
            f"    {sym:<3s}  {float(pos[0]):12.6f} "
            f"{float(pos[1]):12.6f} {float(pos[2]):12.6f}"
        )

    print("  lattice (bohr):")
    print(_format_lattice(data["lattice"]))

    if data["kind"] == "gpw_multi_k_scf":
        kpoints = np.asarray(data["kpoints"])
        print(f"  n_kpoints:    {kpoints.shape[0]}")

    print("")
    print(_LIPPERT_CITATION)
    return 0


# ---------------------------------------------------------------------------
# vibeqc dos <restart.npz>
# ---------------------------------------------------------------------------


def _cmd_coop(args: argparse.Namespace) -> int:
    """Print COOP/COHP pair table from a .qvf archive."""
    import json
    import zipfile

    path = args.path
    if not os.path.exists(path):
        print(f"vibeqc coop: no such file: {path}", file=sys.stderr)
        return 2

    try:
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as exc:
        print(f"vibeqc coop: cannot read QVF manifest: {exc}", file=sys.stderr)
        return 2

    sections = {s.get("kind", ""): s for s in manifest.get("sections", [])}
    coop_sec = sections.get("dos.coop")
    cohp_sec = sections.get("dos.cohp")

    if coop_sec is None and cohp_sec is None:
        print(
            f"vibeqc coop: {path} contains no dos.coop or dos.cohp section",
            file=sys.stderr,
        )
        return 1

    if args.json:
        result: dict[str, Any] = {}
        with zipfile.ZipFile(path, "r") as zf:
            if coop_sec is not None:
                result["coop"] = _section_as_dict(zf, coop_sec)
            if cohp_sec is not None:
                result["cohp"] = _section_as_dict(zf, cohp_sec)
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        with zipfile.ZipFile(path, "r") as zf:
            if coop_sec is not None:
                _print_section_table(
                    zf, coop_sec, "COOP", "ICOOP", sort_by_abs=args.sort
                )
            if cohp_sec is not None:
                if coop_sec is not None:
                    print()
                _print_section_table(
                    zf, cohp_sec, "-COHP", "-ICOHP", sort_by_abs=args.sort
                )

    return 0


def _section_as_dict(
    zf: "zipfile.ZipFile",
    section: dict,
) -> dict[str, Any]:
    """Read a dos.coop/dos.cohp section into a plain dict for JSON output."""
    import json
    import struct

    members = section.get("members", {})
    meta_member = members.get("meta")
    int_member = members.get("integrated")

    if meta_member is None or int_member is None:
        return {"error": "section missing meta or integrated member"}

    meta_raw = zf.read(meta_member["path"])
    meta = json.loads(meta_raw)
    int_raw = zf.read(int_member["path"])

    dtype = int_member.get("dtype", "float64")
    shape = int_member.get("shape", [0])
    n_spin = meta.get("n_spin", 1)
    if n_spin == 2 and len(shape) >= 2:
        n_pairs = shape[1]
        n_entries = shape[0] * shape[1]
    else:
        n_pairs = shape[0] if shape else 0
        n_entries = n_pairs
    if dtype == "float64":
        fmt_char = "d"
    else:
        raise ValueError(f"Unsupported integrated dtype: {dtype}")
    integrated = list(struct.unpack(f"<{n_entries}{fmt_char}", int_raw))

    pairs = meta.get("pairs", [])
    return {
        "fermi_energy_ev": meta.get("fermi_energy_ev", 0.0),
        "sigma_ev": meta.get("sigma_ev", 0.0),
        "n_spin": meta.get("n_spin", 1),
        "pairs": [
            {
                "i": p.get("i"),
                "j": p.get("j"),
                "symbol_i": p.get("symbol_i", "?"),
                "symbol_j": p.get("symbol_j", "?"),
                "distance_ang": p.get("distance_ang"),
                "integrated": integrated[ip] if ip < len(integrated) else None,
            }
            for ip, p in enumerate(pairs)
        ],
    }


def _print_section_table(
    zf: "zipfile.ZipFile",
    section: dict,
    label: str,
    ilabel: str,
    sort_by_abs: bool = False,
) -> None:
    """Print a formatted pair table from a dos.coop/dos.cohp section."""
    import json
    import struct

    members = section.get("members", {})
    meta_member = members.get("meta")
    int_member = members.get("integrated")

    if meta_member is None or int_member is None:
        print(f"# {label}: section is missing meta or integrated member")
        return

    meta_raw = zf.read(meta_member["path"])
    meta = json.loads(meta_raw)
    int_raw = zf.read(int_member["path"])

    dtype = int_member.get("dtype", "float64")
    shape = int_member.get("shape", [0])
    n_pairs = shape[0] if shape else 0
    if dtype == "float64":
        fmt_char = "d"
        item_size = 8
    else:
        raise ValueError(f"Unsupported integrated dtype: {dtype}")
    integrated = struct.unpack(f"<{n_pairs}{fmt_char}", int_raw)

    pairs = meta.get("pairs", [])
    fermi_ev = meta.get("fermi_energy_ev", 0.0)
    sigma_ev = meta.get("sigma_ev", 0.0)

    print(f"# {label}  (E_F = {fermi_ev:.3f} eV,  σ = {sigma_ev:.3f} eV)")
    print(f"# {'pair':>12s}  {'distance/Å':>10s}  {ilabel:>12s}")
    print("# " + "-" * 42)

    # Build index list, optionally sorted by |integrated|
    indices = list(range(n_pairs))
    if sort_by_abs:
        indices.sort(key=lambda p: abs(integrated[p]), reverse=True)

    for p in indices:
        pair = pairs[p] if p < len(pairs) else {}
        sym_i = pair.get("symbol_i", "?")
        sym_j = pair.get("symbol_j", "?")
        i = pair.get("i", p)
        j = pair.get("j", p + 1)
        d = pair.get("distance_ang", None)
        name = f"{sym_i}{i}-{sym_j}{j}"
        dist_str = f"{d:.2f}" if d is not None else "-"
        ival = integrated[p]
        print(f"  {name:>10s}  {dist_str:>10s}  {ival:>+12.4f}")


def _cmd_mayer(args: argparse.Namespace) -> int:
    """Print periodic Mayer bond orders from a .qvf archive."""
    import json
    import zipfile

    path = args.path
    if not os.path.exists(path):
        print(f"vibeqc mayer: no such file: {path}", file=sys.stderr)
        return 2

    try:
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as exc:
        print(f"vibeqc mayer: cannot read QVF manifest: {exc}", file=sys.stderr)
        return 2

    sections = {s.get("kind", ""): s for s in manifest.get("sections", [])}
    bo_sec = sections.get("bond_orders")
    if bo_sec is None:
        print(
            f"vibeqc mayer: {path} contains no bond_orders section",
            file=sys.stderr,
        )
        return 1

    members = bo_sec.get("members", {})
    bo_member = members.get("bond_orders")
    if bo_member is None:
        print(
            "vibeqc mayer: bond_orders section has no bond_orders member",
            file=sys.stderr,
        )
        return 1

    with zipfile.ZipFile(path, "r") as zf:
        data = json.loads(zf.read(bo_member["path"]))
    method = data.get("method", "?")
    pairs = data.get("pairs", [])
    threshold = float(args.threshold)

    if args.json:
        # JSON mode: show all pairs (threshold only applies to table output)
        json.dump({"method": method, "pairs": pairs}, sys.stdout, indent=2)
        print()
    else:
        print(f"# Mayer bond orders (method={method}, threshold={threshold:.2f})")
        print(f"# {'pair':>12s}  {'distance/Å':>10s}  {'order':>10s}")
        print("# " + "-" * 38)
        for p in pairs:
            order = p.get("order", 0.0)
            if order < threshold:
                continue
            sym_i = p.get("symbol_i", "?")
            sym_j = p.get("symbol_j", "?")
            i = p.get("i", 0)
            j = p.get("j", 0)
            d = p.get("distance_ang", None)
            name = f"{sym_i}{i}-{sym_j}{j}"
            dist_str = f"{d:.2f}" if d is not None else "-"
            print(f"  {name:>10s}  {dist_str:>10s}  {order:>10.4f}")

    return 0


def _cmd_dos(args: argparse.Namespace) -> int:
    """Compute Gaussian-broadened DOS from a GPW restart file."""
    from vibeqc import gaussian_dos, load_gpw_result

    path = args.path
    if not os.path.exists(path):
        print(f"vibeqc dos: no such file: {path}", file=sys.stderr)
        return 2

    sigma = float(args.sigma)
    bins = int(args.bins)
    if bins <= 1:
        print(
            f"vibeqc dos: --bins must be >= 2 (got {bins})",
            file=sys.stderr,
        )
        return 2

    data = load_gpw_result(path)

    # Closed-shell weighting from the restart: load_gpw_result doesn't
    # build a runner-shaped result with an .occupations field, so we
    # weight by 2 for occupied (n_elec // 2) and 0 for virtual using
    # the unit-cell Z totals.
    Z = np.asarray(data["unit_cell_Z"])
    n_elec = int(Z.sum())
    n_occ = max(n_elec // 2, 0)

    if data["kind"] == "gpw_multi_k_scf":
        eigs_k = np.asarray(data["mo_energies_k"], dtype=float)
        kweights = np.asarray(data["kweights"], dtype=float)
        flat_eps = []
        flat_w = []
        for ik in range(eigs_k.shape[0]):
            eps_k = eigs_k[ik]
            w = np.zeros_like(eps_k)
            w[:n_occ] = 2.0 * float(kweights[ik])
            flat_eps.append(eps_k)
            flat_w.append(w)
        eps_all = np.concatenate(flat_eps)
        w_all = np.concatenate(flat_w)
    else:
        eps_all = np.asarray(data["mo_energies"], dtype=float).ravel()
        w_all = np.zeros_like(eps_all)
        w_all[:n_occ] = 2.0

    pad = 10.0 * sigma
    e_lo = float(eps_all.min()) - pad
    e_hi = float(eps_all.max()) + pad
    if not (e_hi > e_lo):
        e_hi = e_lo + max(20.0 * sigma, 1e-6)
    e_grid = np.linspace(e_lo, e_hi, bins)

    e_out, dos = gaussian_dos(
        eps_all,
        e_grid=e_grid,
        sigma=sigma,
        weights=w_all,
    )

    print(f"# vibeqc dos {path}")
    print(f"# sigma_Ha = {sigma:.6g}  bins = {bins}")
    print("# energy_Ha    dos")
    for e_val, d_val in zip(e_out, dos):
        print(f"{e_val:.10f}  {d_val:.10f}")
    return 0


# ---------------------------------------------------------------------------
# vibeqc run <script.py>
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a user Python script with ``vibeqc`` preloaded.

    Equivalent in spirit to ``.venv/bin/python <script.py>``: we
    exec the script in a clean global namespace that already has
    ``vibeqc`` (aliased as ``vq``) imported, so users can launch
    a script via the ``vibeqc`` console entry point without having
    to remember the per-laptop venv path.

    A ``.qvf`` argument is a *job container* instead (QVF spec 5.9):
    the declarative ``job.spec`` it carries is reconstructed and run
    through :func:`vibeqc.run_job` / :func:`vibeqc.run_periodic_job`.
    No embedded script is executed on this path -- the container's
    ``run.record`` input, if any, is a record, not a program.
    """
    import runpy

    import vibeqc

    path = args.script
    if not os.path.exists(path):
        print(f"vibeqc run: no such file: {path}", file=sys.stderr)
        return 2

    if str(path).endswith(".qvf"):
        from .qvf_job import run_container

        try:
            run_container(
                path,
                force=bool(getattr(args, "force", False)),
                output=getattr(args, "output", None),
            )
        except ValueError as exc:
            print(f"vibeqc run: {exc}", file=sys.stderr)
            return 2
        return 0

    # Clean global namespace with vibeqc preloaded. ``__name__`` is
    # set to "__main__" so ``if __name__ == "__main__":`` blocks in
    # user scripts fire correctly.
    init_globals = {
        "vibeqc": vibeqc,
        "vq": vibeqc,
    }
    try:
        runpy.run_path(path, init_globals=init_globals, run_name="__main__")
    except SystemExit as exc:  # pragma: no cover - user-script-driven
        return int(exc.code) if exc.code is not None else 0
    return 0


# ---------------------------------------------------------------------------
# vibeqc opt <structure.xyz>
# ---------------------------------------------------------------------------


def _cmd_opt(args: argparse.Namespace) -> int:
    """Quick BFGS geometry optimisation under :class:`VibeqcGPW`.

    Reads the input structure via ASE, wraps it in a generous cubic
    box (max extent + 8 bohr padding), and runs BFGS for up to 20
    steps with ``fmax = 0.05 eV/Å``. The initial / final energies
    and atomic positions go to stdout; the optimised geometry is
    written to ``<structure>.opt.xyz``.
    """
    from ase.io import read, write
    from ase.optimize import BFGS
    from ase.units import Bohr

    from vibeqc import VibeqcGPW

    path = args.path
    if not os.path.exists(path):
        print(f"vibeqc opt: no such file: {path}", file=sys.stderr)
        return 2

    atoms = read(path)

    # Wrap in a generous cubic box. Padding is in bohr per the
    # task spec; convert to Å for ASE.
    pad_bohr = 8.0
    pad_ang = pad_bohr * Bohr
    positions = atoms.get_positions()
    max_extent = float(np.ptp(positions, axis=0).max()) if len(atoms) else 0.0
    box_ang = max_extent + pad_ang
    # Centre the atoms in the box.
    atoms.set_cell([box_ang, box_ang, box_ang])
    atoms.set_pbc([True, True, True])
    atoms.center()

    functional = args.functional if args.functional else None
    calc = VibeqcGPW(
        basis=args.basis,
        functional=functional,
        cutoff_ha=float(args.cutoff),
    )
    atoms.calc = calc

    print(f"vibeqc opt: {path}")
    print(f"  basis       = {args.basis}")
    print(f"  functional  = {functional or '(HF)'}")
    print(f"  cutoff_ha   = {args.cutoff}")
    print(f"  box (Å)     = {box_ang:.6f} (cubic)")
    print(f"  n_atoms     = {len(atoms)}")

    e_initial = float(atoms.get_potential_energy())
    print(f"  initial energy = {e_initial:.10f} eV")
    print("  initial positions (Å):")
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        print(f"    {sym:<3s}  {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05, steps=20)

    e_final = float(atoms.get_potential_energy())
    print(f"  final energy   = {e_final:.10f} eV")
    print(f"  delta_E        = {(e_final - e_initial):.10f} eV")
    print("  final positions (Å):")
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        print(f"    {sym:<3s}  {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

    stem, _ext = os.path.splitext(path)
    out_path = f"{stem}.opt.xyz"
    write(out_path, atoms)
    print(f"  wrote optimised geometry: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# vibeqc bands <restart.npz>
# ---------------------------------------------------------------------------


# High-symmetry points in fractional reciprocal-lattice coordinates.
# Cubic-cell convention -- these are the standard simple-cubic labels.
_BAND_HSP = {
    "G": np.array([0.0, 0.0, 0.0]),
    "GAMMA": np.array([0.0, 0.0, 0.0]),
    "X": np.array([0.5, 0.0, 0.0]),
    "M": np.array([0.5, 0.5, 0.0]),
    "R": np.array([0.5, 0.5, 0.5]),
}


def _parse_k_path(spec: str, n_points: int, lattice: np.ndarray):
    """Parse ``"G,X,M,G"`` into a Cartesian-bohr⁻¹ k-path.

    The lattice is in bohr; reciprocal vectors are
    ``b = 2pi (a⁻¹)ᵀ``. ``n_points`` is the number of k-points per
    segment between adjacent labels (endpoints inclusive on the
    first segment, then exclusive to avoid duplicating waypoints).
    """
    labels_in = [s.strip().upper() for s in spec.split(",") if s.strip()]
    if len(labels_in) < 2:
        raise ValueError(
            f"--k-path must list at least two high-symmetry points (got {labels_in!r})"
        )
    unknown = [lab for lab in labels_in if lab not in _BAND_HSP]
    if unknown:
        raise ValueError(
            f"--k-path: unknown high-symmetry label(s) {unknown!r}; "
            f"known: {sorted(set(_BAND_HSP))}"
        )

    # Reciprocal lattice (Cartesian bohr⁻¹) for the supplied real-
    # space lattice.
    a = np.asarray(lattice, dtype=float)
    b = 2.0 * np.pi * np.linalg.inv(a).T

    frac_waypoints = [_BAND_HSP[lab] for lab in labels_in]
    cart_pts: list[np.ndarray] = []
    cart_labels: list[str] = []
    for seg_idx in range(len(frac_waypoints) - 1):
        f0 = frac_waypoints[seg_idx]
        f1 = frac_waypoints[seg_idx + 1]
        # Endpoint inclusive on first segment, exclusive afterward to
        # avoid duplicating shared waypoints.
        start_excl = 0 if seg_idx == 0 else 1
        for j in range(start_excl, n_points):
            t = float(j) / float(n_points - 1)
            frac_k = (1.0 - t) * f0 + t * f1
            cart_k = frac_k @ b
            cart_pts.append(cart_k)
            if j == 0:
                cart_labels.append(labels_in[seg_idx])
            elif j == n_points - 1:
                cart_labels.append(labels_in[seg_idx + 1])
            else:
                cart_labels.append("")
    return cart_labels, np.asarray(cart_pts, dtype=float)


def _cmd_bands(args: argparse.Namespace) -> int:
    """Compute band-structure eigenvalues along a k-path."""
    from vibeqc import band_path_eigenvalues, load_gpw_result

    path = args.path
    if not os.path.exists(path):
        print(f"vibeqc bands: no such file: {path}", file=sys.stderr)
        return 2

    n_points = int(args.n_points)
    if n_points < 2:
        print(
            f"vibeqc bands: --n-points must be >= 2 (got {n_points})",
            file=sys.stderr,
        )
        return 2

    data = load_gpw_result(path)
    lattice = np.asarray(data["lattice"], dtype=float)

    try:
        labels, k_cart = _parse_k_path(args.k_path, n_points, lattice)
    except ValueError as exc:
        print(f"vibeqc bands: {exc}", file=sys.stderr)
        return 2

    functional = data.get("breakdown_functional") or None
    eigenvalues = band_path_eigenvalues(
        data["system"],
        data["basis"],
        data["density"],
        k_cart,
        functional=functional,
    )

    n_bands_print = min(5, eigenvalues.shape[1])
    print(f"# vibeqc bands {path}")
    print(f"# k_path       = {args.k_path}")
    print(f"# n_points/seg = {n_points}")
    print(f"# functional   = {functional or '(HF)'}")
    print(
        "# label      kx (bohr^-1)   ky (bohr^-1)   kz (bohr^-1)   "
        f"eigenvalues_Ha[0:{n_bands_print}]"
    )
    for label, k_vec, eps_k in zip(labels, k_cart, eigenvalues):
        eps_str = "  ".join(f"{float(eps_k[i]):14.8f}" for i in range(n_bands_print))
        print(
            f"  {label:<6s}  {float(k_vec[0]):12.6f}  {float(k_vec[1]):12.6f}  "
            f"{float(k_vec[2]):12.6f}    {eps_str}"
        )
    return 0


# ---------------------------------------------------------------------------
# vibeqc tddft
# ---------------------------------------------------------------------------


def _cmd_tddft(args: argparse.Namespace) -> int:
    """TD-DFT (TDA) excited-state calculation on an XYZ structure."""
    from ase.io import read

    import vibeqc
    from vibeqc import Molecule
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.tddft import run_tddft_casida, run_tddft_tda

    # Read structure.
    atoms = read(args.path)
    coords = atoms.get_positions()  # Angstrom
    # Convert to bohr (CODATA 2018).
    ANG_TO_BOHR = 1.8897259886
    coords_bohr = coords * ANG_TO_BOHR
    numbers = atoms.get_atomic_numbers()
    mol = Molecule(
        [vibeqc.Atom(int(Z), c) for Z, c in zip(numbers, coords_bohr)],
        charge=args.charge,
        multiplicity=args.multiplicity,
    )
    basis = BasisSet(mol, args.basis)
    func = args.functional if args.functional else None

    print(f"TD-DFT on {args.path}")
    print(f"  basis = {args.basis}")
    print(f"  functional = {func or 'HF'}")
    print(f"  n_states = {args.n_states}")
    print(f"  method = {'Casida' if args.casida else 'TDA'}")
    print()

    # Ground-state SCF.
    if args.multiplicity > 1:
        from vibeqc._vibeqc_core import UHFOptions, UKSOptions, run_uhf, run_uks

        if func:
            opts = UKSOptions()
            opts.conv_tol_energy = 1e-8
            result = run_uks(mol, basis, func, opts)
        else:
            opts = UHFOptions()
            opts.conv_tol_energy = 1e-8
            result = run_uhf(mol, basis, opts)
    elif func:
        from vibeqc._vibeqc_core import RKSOptions, run_rks

        opts = RKSOptions()
        opts.conv_tol_energy = 1e-8
        result = run_rks(mol, basis, func, opts)
    else:
        from vibeqc._vibeqc_core import RHFOptions, run_rhf

        opts = RHFOptions()
        opts.conv_tol_energy = 1e-8
        result = run_rhf(mol, basis, opts)

    if not result.converged:
        print("ERROR: SCF did not converge.", file=sys.stderr)
        return 1

    print(f"SCF converged in {result.n_iter} iterations")
    print(f"Ground-state energy = {result.energy:.10f} Ha")
    print()

    # TD-DFT.
    if args.multiplicity > 1:
        from vibeqc.tddft import run_tddft_tda_uhf

        from vibeqc.correlation_conventions import effective_electron_count

        _n_eff = effective_electron_count(mol, result)
        n_occ_a = getattr(result, "n_occ_a", _n_eff // 2)
        n_occ_b = getattr(result, "n_occ_b", _n_eff // 2)
        td = run_tddft_tda_uhf(
            mol,
            basis,
            np.asarray(result.mo_energies_alpha, dtype=float),
            np.asarray(result.mo_energies_beta, dtype=float),
            np.asarray(result.mo_coeffs_alpha, dtype=float),
            np.asarray(result.mo_coeffs_beta, dtype=float),
            n_occ_a,
            n_occ_b,
            n_states=args.n_states,
            functional=func,
        )
    else:
        from vibeqc.correlation_conventions import effective_electron_count

        n_occ = effective_electron_count(mol, result) // 2
        mo_e = np.asarray(result.mo_energies, dtype=float)
        mo_c = np.asarray(result.mo_coeffs, dtype=float)

        if args.casida:
            td = run_tddft_casida(
                mol, basis, mo_e, mo_c, n_occ, n_states=args.n_states, functional=func
            )
        else:
            td = run_tddft_tda(
                mol, basis, mo_e, mo_c, n_occ, n_states=args.n_states, functional=func
            )

    print(f"Excited states ({td.method}, n_states={td.n_states}):")
    print(
        f"  {'State':>5s}  {'E (eV)':>10s}  {'λ (nm)':>10s}"
        f"  {'f_osc':>10s}  {'Dominant transition'}"
    )
    for st in td.states:
        dom = ", ".join(
            f"{o}→{v} ({abs(a):.3f})" for o, v, a in st.dominant_amplitudes[:3]
        )
        print(
            f"  {st.index:>5d}  {st.excitation_energy_ev:>10.4f}"
            f"  {st.wavelength_nm:>10.1f}  {st.oscillator_strength:>10.4f}"
            f"  {dom}"
        )
    return 0


# ---------------------------------------------------------------------------
# vibeqc tddft-opt
# ---------------------------------------------------------------------------


def _cmd_tddft_opt(args: argparse.Namespace) -> int:
    """Excited-state geometry optimisation on S1 via FD gradient."""
    from ase.io import read, write
    from ase.optimize import LBFGS

    import vibeqc
    from vibeqc import Molecule
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.excited_gradient import (
        cis_state_gradients_fd,
        make_hf_cis_energy_fn,
    )

    ANG_TO_BOHR = 1.8897259886

    atoms = read(args.path)
    Z = atoms.get_atomic_numbers()
    coords0_ang = atoms.get_positions()
    func = args.functional if args.functional else None

    print(f"TD-DFT S{args.state} optimisation on {args.path}")
    print(f"  basis = {args.basis}")
    print(f"  functional = {func or 'HF'}")
    print(f"  max_steps = {args.max_steps}")
    print(f"  fmax = {args.fmax} eV/Ang")
    print()

    # Build CIS energy function (HF reference for TDA).
    fn = make_hf_cis_energy_fn(
        [int(z) for z in Z],
        args.basis,
        charge=args.charge,
        spin="singlet" if args.multiplicity == 1 else "triplet",
        n_states=max(args.state, 5),
    )

    # Initial energy.
    ss0 = fn(coords0_ang)
    e0 = ss0.e_ground + ss0.excitation_energies[args.state - 1]
    print(f"Initial S{args.state} energy = {e0:.10f} Ha")
    print(
        f"  (E_SCF = {ss0.e_ground:.10f}, omega = {ss0.excitation_energies[args.state - 1]:.10f})"
    )

    # L-BFGS optimisation loop.
    coords = coords0_ang.copy()
    step = 0

    # Simple L-BFGS with SciPy.
    from scipy.optimize import minimize

    def _energy_and_grad(x):
        c = x.reshape(-1, 3)
        grads = cis_state_gradients_fd(fn, c, states=[args.state])
        g = grads.get(args.state)
        ss = fn(c)
        e = ss.e_ground + ss.excitation_energies[args.state - 1]
        # Convert Ha/bohr -> eV/Ang.
        g_ev_ang = g * (27.211386245988 / 0.529177210903)
        return e, g_ev_ang.flatten()

    def _callback(xk):
        nonlocal step
        step += 1
        ss = fn(xk.reshape(-1, 3))
        e = ss.e_ground + ss.excitation_energies[args.state - 1]
        g = cis_state_gradients_fd(fn, xk.reshape(-1, 3), states=[args.state])
        gmax = np.max(np.abs(g[args.state])) * 27.211386245988 / 0.529177210903
        print(f"  step {step:>3d}: E = {e:.10f} Ha, |grad|_max = {gmax:.6f} eV/Ang")
        if step >= args.max_steps:
            return True  # stop

    res = minimize(
        _energy_and_grad,
        coords.flatten(),
        method="L-BFGS-B",
        jac=True,
        callback=_callback,
        options={"gtol": args.fmax, "maxiter": args.max_steps},
    )

    coords_opt = res.x.reshape(-1, 3)
    ss_final = fn(coords_opt)
    e_final = ss_final.e_ground + ss_final.excitation_energies[args.state - 1]
    print()
    print(f"Optimised S{args.state} energy = {e_final:.10f} Ha")
    print(f"Delta E = {(e_final - e0) * 27.211386245988:.6f} eV")

    # Write optimised geometry.
    out = args.path.replace(".xyz", f".s{args.state}_opt.xyz")
    atoms_opt = atoms.copy()
    atoms_opt.set_positions(coords_opt)
    write(out, atoms_opt)
    print(f"Optimised geometry written to {out}")
    return 0


# ---------------------------------------------------------------------------
# vibeqc nto
# ---------------------------------------------------------------------------


def _cmd_nto(args: argparse.Namespace) -> int:
    """Print NTO weights from a .qvf archive."""
    import json
    import zipfile

    with zipfile.ZipFile(args.path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    nto_secs = [s for s in manifest.get("sections", []) if "nto" in s.get("id", "")]
    if not nto_secs:
        print("No NTO sections found in this QVF.")
        return 1

    if args.json:
        with zipfile.ZipFile(args.path, "r") as zf:
            result = []
            for s in nto_secs:
                sid = s["id"]
                parts = sid.split("_")
                state_idx = parts[2].lstrip("S") if len(parts) > 2 else "?"
                role = parts[-1]
                spin = ""
                if len(parts) >= 5 and parts[-2] not in ("hole", "electron"):
                    spin = parts[-2]
                # Read mo_metadata for weights
                mm_path = s["members"]["mo_metadata"]["path"]
                mm = json.loads(zf.read(mm_path))
                e_ev = mm.get("excitation_energy_ev")
                occs = mm.get("occupations", [])
                result.append(
                    {
                        "id": sid,
                        "state_index": state_idx,
                        "role": role,
                        "spin": spin,
                        "excitation_energy_ev": e_ev,
                        "nto_weights": occs,
                        "n_mo": mm.get("n_mo"),
                    }
                )
        print(json.dumps(result, indent=2))
        return 0

    print(f"NTO sections in {args.path}:")
    with zipfile.ZipFile(args.path, "r") as zf:
        for s in nto_secs:
            sid = s["id"]
            mm_path = s["members"]["mo_metadata"]["path"]
            mm = json.loads(zf.read(mm_path))
            occs = mm.get("occupations", [])
            e_ev = mm.get("excitation_energy_ev")
            weight_info = ""
            if occs:
                top = occs[:5]
                weight_info = f" NTO weights: {[f'{w:.4f}' for w in top]}"
            e_info = f" E={e_ev:.4f} eV" if e_ev else ""
            print(f"  {sid:<35s}{e_info}{weight_info}")
    return 0


# ---------------------------------------------------------------------------
# vibeqc meci
# ---------------------------------------------------------------------------


def _cmd_meci(args: argparse.Namespace) -> int:
    """MECI optimisation between S1 and S2 via penalty function + L-BFGS."""
    import numpy as np
    from ase.io import read, write

    from vibeqc.excited_gradient import (
        cis_state_gradients_fd,
        make_hf_cis_energy_fn,
    )

    ANG_TO_BOHR = 1.8897259886
    HA_TO_EV = 27.211386245988

    atoms = read(args.path)
    Z = atoms.get_atomic_numbers()
    coords = atoms.get_positions().copy()

    print(f"MECI optimisation on {args.path}")
    print(f"  basis = {args.basis}")
    print(f"  max_steps = {args.max_steps}")
    print(f"  fmax = {args.fmax} eV/Ang")
    print()

    fn = make_hf_cis_energy_fn(
        [int(z) for z in Z],
        args.basis,
        charge=args.charge,
        spin="singlet" if args.multiplicity == 1 else "triplet",
        n_states=3,
    )

    ALPHA = 0.02
    SIGMA_START = 3.0
    SIGMA_MAX = 100.0
    SIGMA_FACTOR = 2.0
    INNER_STEPS = 3  # L-BFGS steps per sigma value

    sigma = SIGMA_START
    step = 0

    ss0 = fn(coords)
    e1 = ss0.e_ground + ss0.excitation_energies[0]
    e2 = ss0.e_ground + ss0.excitation_energies[1]
    gap_ev = (e2 - e1) * HA_TO_EV
    print(f"Initial: gap = {gap_ev:.6f} eV, avgE = {0.5 * (e1 + e2) * HA_TO_EV:.6f} eV")
    print()

    while step < args.max_steps and gap_ev > 1e-4 and sigma <= SIGMA_MAX:
        # Compute gradient at current geometry
        grads = cis_state_gradients_fd(fn, coords, states=[1, 2])
        ss = fn(coords)
        e1 = ss.e_ground + ss.excitation_energies[0]
        e2 = ss.e_ground + ss.excitation_energies[1]
        de = e2 - e1
        gap_ev = de * HA_TO_EV

        # Penalty function gradient
        g1 = grads[1]  # Ha/bohr, (n_atoms, 3)
        g2 = grads[2]
        g_avg = 0.5 * (g1 + g2)
        if de > -ALPHA:
            pref = sigma * (2 * de / (de + ALPHA) - de * de / (de + ALPHA) ** 2)
            g_pen = pref * (g2 - g1)
        else:
            g_pen = 0.0
        g_tot = g_avg + g_pen

        f_val = 0.5 * (e1 + e2) + (
            sigma * de * de / (de + ALPHA) if de > -ALPHA else 0.0
        )
        gmax = np.max(np.abs(g_tot)) * HA_TO_EV / ANG_TO_BOHR

        step += 1
        print(
            f"  step {step:>3d}: gap = {gap_ev:.6f} eV,"
            f" |g|_max = {gmax:.6f} eV/Ang, sigma = {sigma:.1f}"
        )

        if gmax < args.fmax and gap_ev < 0.01:
            print("  Converged.")
            break

        # Steepest descent step with adaptive step size
        g_norm = np.linalg.norm(g_tot)
        if g_norm > 1e-12:
            step_dir = -g_tot / g_norm
            # Line search: try a step, accept if energy decreases
            alpha = 0.1  # initial step in Angstrom
            for _ in range(5):
                c_new = coords + alpha * step_dir
                ss_new = fn(c_new)
                e1_n = ss_new.e_ground + ss_new.excitation_energies[0]
                e2_n = ss_new.e_ground + ss_new.excitation_energies[1]
                de_n = e2_n - e1_n
                f_new = 0.5 * (e1_n + e2_n) + (
                    sigma * de_n * de_n / (de_n + ALPHA) if de_n > -ALPHA else 0.0
                )
                if f_new < f_val:
                    coords = c_new
                    break
                alpha *= 0.5

        # Increase sigma to drive gap to zero
        if step % INNER_STEPS == 0:
            sigma = min(sigma * SIGMA_FACTOR, SIGMA_MAX)

    print()
    print(f"Final:   gap = {gap_ev:.6f} eV")
    print(f"         avgE = {0.5 * (e1 + e2) * HA_TO_EV:.6f} eV")

    out = args.path.replace(".xyz", ".meci.xyz")
    atoms_opt = atoms.copy()
    atoms_opt.set_positions(coords)
    write(out, atoms_opt)
    print(f"MECI geometry written to {out}")
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vibeqc",
        description=(
            "vibeqc -- command-line companion to the vibe-qc quantum-"
            "chemistry library. GPAW-style sub-commands: info / gpw / "
            "dos."
        ),
    )
    sub = p.add_subparsers(dest="subcommand", required=False)

    p_info = sub.add_parser(
        "info",
        help="Print vibeqc version + linked library versions + platform.",
    )
    p_info.set_defaults(func=_cmd_info)

    p_gpw = sub.add_parser(
        "gpw",
        help="Summarise a GPW restart (.npz) file.",
    )
    p_gpw.add_argument(
        "path",
        help="Path to a .npz restart written by vibeqc.save_gpw_result.",
    )
    p_gpw.set_defaults(func=_cmd_gpw)

    p_dos = sub.add_parser(
        "dos",
        help="Compute Gaussian-broadened DOS from a GPW restart file.",
    )
    p_dos.add_argument(
        "path",
        help="Path to a .npz restart written by vibeqc.save_gpw_result.",
    )
    p_dos.add_argument(
        "--sigma",
        type=float,
        default=0.01,
        help="Gaussian broadening s in Hartree (default 0.01 Ha ~ 0.27 eV).",
    )
    p_dos.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of energy grid points (default 200).",
    )
    p_dos.set_defaults(func=_cmd_dos)

    p_coop = sub.add_parser(
        "coop",
        help="Print COOP/COHP pair table from a .qvf archive.",
    )
    p_coop.add_argument(
        "path",
        help="Path to a .qvf archive produced by run_periodic_job(coop_cohp=True).",
    )
    p_coop.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of a formatted table.",
    )
    p_coop.add_argument(
        "--sort",
        action="store_true",
        help="Sort pairs by descending |integrated| value.",
    )
    p_coop.set_defaults(func=_cmd_coop)

    p_mayer = sub.add_parser(
        "mayer",
        help="Print periodic Mayer bond orders from a .qvf archive.",
    )
    p_mayer.add_argument(
        "path",
        help="Path to a .qvf archive produced by run_periodic_job(output_qvf=True).",
    )
    p_mayer.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of a formatted table.",
    )
    p_mayer.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum bond order to display in table mode (default 0.05). Ignored for --json.",
    )
    p_mayer.set_defaults(func=_cmd_mayer)

    p_nto = sub.add_parser(
        "nto",
        help="Print NTO weights from a .qvf archive.",
    )
    p_nto.add_argument(
        "path",
        help="Path to a .qvf archive produced with tddft=True, nto=True.",
    )
    p_nto.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON.",
    )
    p_nto.set_defaults(func=_cmd_nto)

    p_run = sub.add_parser(
        "run",
        help=(
            "Run a Python script with vibeqc preloaded, or execute a "
            "pending .qvf job container (QVF spec 5.9)."
        ),
    )
    p_run.add_argument(
        "script",
        help=(
            "Path to a Python script to execute (run as __main__), or "
            "a .qvf job container to run."
        ),
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help=(
            ".qvf containers only: run even when provenance.run_status "
            "is not 'pending' (re-runs a settled archive)."
        ),
    )
    p_run.add_argument(
        "--output",
        default=None,
        help=(
            ".qvf containers only: output stem for the run's files "
            "(default: the container path without its .qvf suffix)."
        ),
    )
    p_run.set_defaults(func=_cmd_run)

    p_opt = sub.add_parser(
        "opt",
        help=(
            "Quick BFGS geometry optimisation on an XYZ structure via "
            "the VibeqcGPW ASE Calculator."
        ),
    )
    p_opt.add_argument(
        "path",
        help="Path to an XYZ structure file readable by ASE.",
    )
    p_opt.add_argument(
        "--basis",
        default="sto-3g",
        help="libint basis-set name (default sto-3g).",
    )
    p_opt.add_argument(
        "--functional",
        default="lda",
        help=("libxc functional name; pass an empty string '' for HF. Default lda."),
    )
    p_opt.add_argument(
        "--cutoff",
        type=float,
        default=300.0,
        help="Plane-wave grid cutoff in Hartree (default 300.0).",
    )
    p_opt.set_defaults(func=_cmd_opt)

    p_bands = sub.add_parser(
        "bands",
        help=(
            "Compute eigenvalues along a high-symmetry k-path from a "
            "GPW restart (cubic-cell labels: G/X/M/R)."
        ),
    )
    p_bands.add_argument(
        "path",
        help="Path to a .npz restart written by vibeqc.save_gpw_result.",
    )
    p_bands.add_argument(
        "--k-path",
        default="G,X,M,G",
        help=(
            "Comma-separated high-symmetry labels (default 'G,X,M,G'). "
            "Known: G=Gamma=(0,0,0), X=(0.5,0,0), M=(0.5,0.5,0), "
            "R=(0.5,0.5,0.5)."
        ),
    )
    p_bands.add_argument(
        "--n-points",
        type=int,
        default=30,
        help="Number of k-points per segment (default 30).",
    )
    p_bands.set_defaults(func=_cmd_bands)

    p_tddft = sub.add_parser(
        "tddft",
        help=("Run a TD-DFT (TDA) excited-state calculation on an XYZ structure."),
    )
    p_tddft.add_argument(
        "path",
        help="Path to an XYZ structure file readable by ASE.",
    )
    p_tddft.add_argument(
        "--basis",
        default="def2-svp",
        help="libint basis-set name (default def2-svp).",
    )
    p_tddft.add_argument(
        "--functional",
        default="",
        help=(
            "libxc functional name; pass empty string for HF/CIS. "
            "Default empty (HF/CIS)."
        ),
    )
    p_tddft.add_argument(
        "--n-states",
        type=int,
        default=5,
        help="Number of excited states (default 5).",
    )
    p_tddft.add_argument(
        "--casida",
        action="store_true",
        help="Use full Casida instead of TDA.",
    )
    p_tddft.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge (default 0).",
    )
    p_tddft.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (1=singlet, 2=doublet, default 1).",
    )
    p_tddft.set_defaults(func=_cmd_tddft)

    p_tddft_opt = sub.add_parser(
        "tddft-opt",
        help=(
            "Excited-state geometry optimisation. Minimises the S1 "
            "total energy (E_SCF + omega) via FD gradient + L-BFGS."
        ),
    )
    p_tddft_opt.add_argument(
        "path",
        help="Path to an XYZ structure file readable by ASE.",
    )
    p_tddft_opt.add_argument(
        "--basis",
        default="def2-svp",
        help="libint basis-set name (default def2-svp).",
    )
    p_tddft_opt.add_argument(
        "--functional",
        default="",
        help="libxc functional name; pass empty for HF/CIS. Default HF.",
    )
    p_tddft_opt.add_argument(
        "--state",
        type=int,
        default=1,
        help="Excited state index to optimise (1=S1, default).",
    )
    p_tddft_opt.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum optimisation steps (default 20).",
    )
    p_tddft_opt.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force convergence in eV/Ang (default 0.01).",
    )
    p_tddft_opt.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge (default 0).",
    )
    p_tddft_opt.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (1=singlet, default 1).",
    )
    p_tddft_opt.set_defaults(func=_cmd_tddft_opt)

    p_meci = sub.add_parser(
        "meci",
        help=(
            "Minimum-energy conical intersection (MECI) optimisation between S1 and S2."
        ),
    )
    p_meci.add_argument(
        "path",
        help="Path to an XYZ structure file readable by ASE.",
    )
    p_meci.add_argument(
        "--basis",
        default="def2-svp",
        help="libint basis-set name (default def2-svp).",
    )
    p_meci.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum optimisation steps (default 30).",
    )
    p_meci.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force convergence in eV/Ang (default 0.01).",
    )
    p_meci.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge (default 0).",
    )
    p_meci.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (1=singlet, default 1).",
    )
    p_meci.set_defaults(func=_cmd_meci)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    if getattr(args, "subcommand", None) is None:
        parser.print_help(sys.stderr)
        return 1

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
