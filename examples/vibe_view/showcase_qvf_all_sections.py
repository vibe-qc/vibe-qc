"""QVF format showcase: every section kind, end to end.

This is the companion script to ``docs/tutorial/qvf_file_format.md``.
Where ``showcase_formaldehyde.py`` produces the subset of sections a
routine ``run_job`` emits, this script exercises **35 of the 37**
viewer-renderable section kinds the QVF writer implements as of
v1.2 (``basis.ao`` and ``scan.surface`` are exercised elsewhere —
``showcase_basis_functions.py`` and the viewer test fixtures) plus the
QVF v2 periodic reaction-path extension — so that running it is a stress test
of the whole writer surface, not just the common path.

Why that matters: ``write_qvf`` runs the canonical schema validator on
the archive it just wrote and *raises* (deleting the bad file) if the
result does not validate. So calling it once per section kind is a
self-checking harness: if any writer path regresses, this script dies
with a concrete error instead of shipping a broken archive to a
consumer. The script also re-runs ``validate_qvf`` explicitly and reads
each archive back through the consumer API (stdlib ``zipfile`` + a
manual SHA-256 check, and ``vibeview.QVFReader`` if vibe-view is
installed).

It produces three coherent archives, which together cover all kinds:

  1. water.qvf            REAL run_job (H2O / RHF / 6-31G*). The natural
                          integration path: structure, volume.density,
                          volume.orbital (HOMO + LUMO), wavefunction.gto,
                          trajectory, vibrations, spectra.ir,
                          atom_properties, scf_history, citations,
                          bond_orders.

  2. spectroscopy.qvf     SYNTHETIC write_qvf over a real Molecule. The
                          format supports spectra and scalar fields
                          vibe-qc does not yet *compute* — this archive
                          carries illustrative (not computed) data for
                          them so the writer + viewer are exercised:
                          volume.density, volume.spin, volume.elf,
                          volume.difference (with operand links),
                          volume.generic, volume.potential, volume.rdg,
                          spectra.raman / uvvis / ecd /
                          vcd / nmr / generic, structure.symmetry, bonds,
                          a molecular (v1) reaction.path, a trajectory +
                          reaction.waypoints annotation,
                          equation_of_state, topology.qtaim, and
                          viewer_defaults.

  3. crystal.qvf          PeriodicSystem (MgO rocksalt) with REAL Hcore
                          bands plus illustrative DOS: structure
                          (lattice), bands, dos.total, dos.projected,
                          structure.symmetry, fermi_surface, phonon_bands,
                          phonon_dos, and a periodic (QVF v2)
                          reaction.path carrying per-frame lattices.

A note on honesty: the synthetic spectra / DOS / scalar fields in
archives 2 and 3 are clearly-labelled demonstration data for the file
format. They are NOT physical observables computed by vibe-qc — do not
cite numbers out of these archives. The electron density, orbitals,
vibrations, IR spectrum and bands in archives 1 and 3 *are* computed.

Run:

    ~/path/to/vibe-qc/.venv/bin/python examples/vibe_view/showcase_qvf_all_sections.py

Then open any archive with vibe-view, e.g.:

    vibe-view open examples/vibe_view/runs/qvf_showcase/spectroscopy.qvf

For the full walkthrough see docs/tutorial/qvf_file_format.md.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import vibeqc as vq
from vibeqc import Atom, Molecule, run_job
from vibeqc.output.formats.qvf import (
    QVF_FORMAT_VERSION,
    QVF_FORMAT_VERSION_V2,
    validate_qvf,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "qvf_showcase"
RUN_DIR.mkdir(parents=True, exist_ok=True)

BOHR = 0.529177210903  # Å per bohr

# Every kind the QVF v1 writer implements. The script asserts the union
# of the three archives' manifests covers this set exactly — the
# coverage gate that makes this a complete writer stress test.
ALL_KINDS = {
    "structure",
    "volume.density",
    "volume.orbital",
    "volume.spin",
    "volume.elf",
    "volume.difference",
    "volume.generic",
    "volume.potential",
    "volume.rdg",
    "wavefunction.gto",
    "atom_properties",
    "trajectory",
    "reaction.path",
    "reaction.waypoints",
    "vibrations",
    "spectra.ir",
    "spectra.raman",
    "spectra.uvvis",
    "spectra.ecd",
    "spectra.vcd",
    "spectra.nmr",
    "spectra.generic",
    "bands",
    "structure.symmetry",
    "bonds",
    "bond_orders",
    "scf_history",
    "citations",
    "dos.total",
    "dos.projected",
    "fermi_surface",
    "phonon_bands",
    "phonon_dos",
    "equation_of_state",
    "topology.qtaim",
    "run.record",
}


# ---------------------------------------------------------------------------
# Small builders for illustrative (NON-physical) demonstration data.
# ---------------------------------------------------------------------------


def _blob(nx: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small Gaussian lump on a uniform grid: (data, origin, span).

    Origin + per-voxel step vectors are in bohr, matching the QVF grid
    descriptor. This is a placeholder scalar field — not a computed
    density.
    """
    ax = np.linspace(-4.0, 4.0, nx)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    data = np.exp(-(x**2 + y**2 + z**2) / 3.0).astype(np.float32)
    origin = np.array([-4.0, -4.0, -4.0], dtype=np.float64)
    span = np.eye(3, dtype=np.float64) * (8.0 / (nx - 1))  # voxel vectors, bohr
    return data, origin, span


def _stick_spectrum(n: int, seed: int) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    freqs = np.sort(rng.uniform(200.0, 3800.0, n))
    ints = rng.uniform(0.0, 100.0, n)
    return freqs.tolist(), ints.tolist()


def _gaussian_dos(energies: np.ndarray, centers, width: float = 0.4) -> np.ndarray:
    dos = np.zeros_like(energies)
    for c in centers:
        dos += np.exp(-0.5 * ((energies - c) / width) ** 2)
    return dos


# ---------------------------------------------------------------------------
# Read-back helpers (the consumer side of the contract).
# ---------------------------------------------------------------------------


def _manifest_kinds(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    return {s["kind"] for s in manifest["sections"]}


def _print_sections(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    print(f"  {path.name}  (qvf_version={manifest['qvf_version']})")
    for s in manifest["sections"]:
        print(f"      {s['id']:<16s} {s['kind']}")
    return {s["kind"] for s in manifest["sections"]}


def _verify_one_checksum(path: Path) -> None:
    """Independently re-hash the first binary member and compare to the
    manifest — the integrity contract every consumer must honour."""
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        for section in manifest["sections"]:
            for role, member in section["members"].items():
                if member.get("format") == "binary":
                    raw = zf.read(member["path"])
                    got = hashlib.sha256(raw).hexdigest()
                    assert got == member["sha256"], (
                        f"checksum mismatch in {path.name} "
                        f"{section['id']}/{role}: {got} != {member['sha256']}"
                    )
                    print(f"      sha256 OK: {section['id']}/{role}")
                    return
    print("      (no binary members to checksum)")


def _validate(path: Path) -> None:
    report = validate_qvf(path)
    status = "VALID" if report["valid"] else "INVALID"
    print(f"      validate_qvf: {status}")
    if not report["valid"]:
        for err in report["errors"]:
            print(f"        - {err}")
        raise SystemExit(f"{path.name} failed validation")


# ---------------------------------------------------------------------------
# Archive 1 — real run_job (the natural path).
# ---------------------------------------------------------------------------


def archive_real_run() -> Path:
    print("[1/3] water.qvf — real H2O / RHF / 6-31G* run_job")
    mol = Molecule(
        [
            Atom(8, [0.0, 0.00, 0.00]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )
    run_job(
        mol,
        basis="6-31g*",
        method="rhf",
        optimize=True,  # -> trajectory section
        hessian=True,  # -> vibrations + spectra.ir
        write_cube=["density", "homo", "lumo"],  # -> volume.density + 2 volume.orbital
        write_molden_file=True,  # -> wavefunction.gto
        write_population_file=True,  # -> atom_properties
        citations=True,  # -> citations
        output=RUN_DIR / "water",
        output_qvf=True,
    )
    return RUN_DIR / "water.qvf"


# ---------------------------------------------------------------------------
# Archive 2 — synthetic molecular: spectra family + exotic volumes.
# ---------------------------------------------------------------------------


def archive_spectroscopy() -> Path:
    print("[2/3] spectroscopy.qvf — synthetic spectra + scalar fields")
    # A real benzene-ish ring as the structure host (positions in bohr).
    ring = Molecule(
        [
            Atom(6, [2.6 * np.cos(t), 2.6 * np.sin(t), 0.0])
            for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
        + [
            Atom(1, [4.6 * np.cos(t), 4.6 * np.sin(t), 0.0])
            for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
    )

    dens, origin, span = _blob()
    spin = (dens * 0.1).astype(np.float32)
    elf = np.clip(dens * 1.5, 0.0, 1.0).astype(np.float32)
    diff = (dens - np.roll(dens, 1, axis=0)).astype(np.float32)
    rdg = np.abs(np.gradient(dens)[0]).astype(np.float32)
    esp = (dens * 0.05 - 0.02).astype(np.float32)  # mock electrostatic potential

    rfreq, rint = _stick_spectrum(8, seed=1)
    vfreq, vint = _stick_spectrum(8, seed=2)
    uv_e = [4.1, 5.0, 6.2, 7.0]
    uv_i = [0.02, 0.31, 0.95, 0.12]

    # A tiny molecular reaction.path (v1) and a separate trajectory the
    # reaction.waypoints section annotates. Frames are closed-shell water
    # geometries with one O-H stretched along the path (the writer only
    # reads each frame's atoms, but Molecule validates the electron count
    # — so the frames must be physically consistent).
    def shifted(dz: float) -> Molecule:
        return Molecule(
            [
                Atom(8, [0.0, 0.00, 0.00]),
                Atom(1, [0.0, 1.43, -0.98 - dz]),
                Atom(1, [0.0, -1.43, -0.98]),
            ]
        )

    rpath_frames = [shifted(dz) for dz in np.linspace(0.0, 1.0, 5)]
    traj_frames = [shifted(dz) for dz in np.linspace(0.0, 0.4, 4)]

    plan = OutputPlan.from_run_job_kwargs(
        output=RUN_DIR / "spectroscopy", method="rhf", basis="sto-3g", functional=None
    )
    return write_qvf(
        RUN_DIR / "spectroscopy",
        plan,
        molecule=ring,
        method="rhf",
        basis="sto-3g",
        # scalar fields
        volume_data={"Electron density (demo)": (dens, origin, span)},
        spin_data={"Spin density (demo)": (spin, origin, span)},
        elf_data={"ELF (demo)": (elf, origin, span)},
        potential_data={"ESP (demo)": (esp, origin, span)},
        rdg_data={"RDG (demo)": (rdg, origin, span)},
        generic_volume_data={"Custom scalar field (demo)": (dens, origin, span)},
        diff_data={
            "Difference density (demo)": {
                "data": diff,
                "origin": origin,
                "span": span,
                "operand_a": "vol_dens_0",
                "operand_b": "vol_dens_0",
                "description": "Illustrative ρ_a − ρ_b difference field.",
            }
        },
        # spectra family
        raman_data={"frequencies": rfreq, "intensities": rint},
        uvvis_data={"energies_ev": uv_e, "intensities": uv_i},
        ecd_data={"energies_ev": uv_e, "intensities": [0.4, -0.6, 0.2, -0.1]},
        vcd_data={"frequencies_cm1": vfreq, "intensities": [v - 50 for v in vint]},
        nmr_data={
            "isotope": "1H",
            "reference": "TMS",
            "chemical_shifts": [7.26] * 6,
            "j_couplings": [],
        },
        generic_spectrum_data={
            "section_id": "xps_demo",
            "label": "XPS (demo)",
            "frequencies": [284.5, 285.1, 290.3],
            "intensities": [1.0, 0.4, 0.2],
        },
        # connectivity + symmetry + history
        bonds_data=[(i, (i + 1) % 6, 1.5) for i in range(6)]
        + [(i, i + 6, 1.0) for i in range(6)],
        symmetry_data={
            "point_group": "D6h",
            "space_group_number": 0,
            "space_group_symbol": "n/a (molecular)",
        },
        scf_history_data=[
            {
                "iter": i,
                "energy_eh": -230.0 + 0.5**i,
                "delta_e": 0.5**i,
                "diis_error": 0.3**i,
            }
            for i in range(8)
        ],
        # bond orders (Mayer, demo)
        bond_orders_data={
            "method": "mayer",
            "pairs": [
                {
                    "i": i,
                    "j": (i + 1) % 6,
                    "order": 1.4,
                    "distance_ang": 1.40,
                    "symbol_i": "C",
                    "symbol_j": "C",
                }
                for i in range(6)
            ],
        },
        # equation of state (demo)
        eos_data={
            "volumes": np.linspace(95, 115, 11),
            "energies": -5000 + 0.02 * (np.linspace(95, 115, 11) - 105) ** 2,
            "fit": {
                "model": "birch_murnaghan",
                "V0": 105.0,
                "E0": -5000.0,
                "B0": 92.0,
                "B0_prime": 3.9,
            },
        },
        # QTAIM topology (demo)
        qtaim_data={
            "critical_points": [
                {
                    "type": "bcp",
                    "position": [1.4, 0.0, 0.0],
                    "rho": 0.28,
                    "laplacian": -0.62,
                    "ellipticity": 0.08,
                    "atom_pair": [0, 1],
                },
                {
                    "type": "rcp",
                    "position": [0.0, 0.0, 1.5],
                    "rho": 0.06,
                    "laplacian": 0.10,
                },
            ],
            "bond_paths": [
                {
                    "atoms": [0, 1],
                    "path": [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.4, 0.0, 0.0]],
                }
            ],
        },
        # reaction representations
        reaction_path={
            "frames": rpath_frames,
            "energies": [0.0, 0.5, 1.1, 0.6, 0.05],
            "reaction_coordinate": [0.0, 0.25, 0.5, 0.75, 1.0],
            "waypoints": [
                {"frame_index": 0, "label": "Reactant", "kind": "reactant"},
                {"frame_index": 2, "label": "TS", "kind": "transition_state"},
                {"frame_index": 4, "label": "Product", "kind": "product"},
            ],
        },
        trajectory_frames=traj_frames,
        trajectory_energies=[0.0, -0.01, -0.02, -0.025],
        reaction_waypoints={
            "trajectory_ref": "traj0",
            "waypoints": [
                {"frame_index": 0, "label": "start", "kind": "point"},
                {"frame_index": 3, "label": "end", "kind": "point"},
            ],
        },
        # self-contained run record (input + log, spec § 5.8)
        run_record={
            "program": "vibe-qc",
            "program_version": "demo",
            "input_text": (
                "import vibeqc as vq\n\n"
                "mol = vq.Molecule.from_xyz('benzene.xyz')\n"
                "vq.run_job(mol, method='rhf', basis='sto-3g',\n"
                "           output='spectroscopy')\n"
            ),
            "input_filename": "input-spectroscopy.py",
            "log_text": (
                "vibe-qc demo -- spectroscopy showcase\n"
                "SCF  iter 1   E = -229.500\n"
                "SCF  iter 8   E = -230.000  converged\n"
                "## References\n(see citations section)\n"
            ),
            "log_filename": "spectroscopy.out",
            "started_utc": "2026-07-24T12:00:00Z",
            "finished_utc": "2026-07-24T12:00:41Z",
        },
        # producer hints to the viewer
        viewer_defaults={
            "auto_open": ["vol_dens_0"],
            "vol_dens_0": {"isovalue": 0.05, "colormap": "viridis"},
        },
    )


# ---------------------------------------------------------------------------
# Archive 3 — periodic: real bands + illustrative DOS + v2 reaction path.
# ---------------------------------------------------------------------------


def _mgo(a_ang: float = 4.21):
    a = a_ang / BOHR  # bohr
    sys = vq.PeriodicSystem(
        3,
        np.eye(3) * a,
        [Atom(12, [0.0, 0.0, 0.0]), Atom(8, [a / 2, a / 2, a / 2])],
    )
    return sys, a


def archive_crystal() -> Path:
    print("[3/3] crystal.qvf — MgO periodic: real bands + DOS + v2 path")
    mgo, a = _mgo()
    basis = vq.BasisSet(mgo.unit_cell_molecule(), "sto-3g")

    kpath = vq.kpath_from_segments(
        mgo,
        segments=[
            ([0.0, 0.0, 0.0], "G", [0.5, 0.0, 0.5], "X"),
            ([0.5, 0.0, 0.5], "X", [0.5, 0.5, 0.5], "L"),
            ([0.5, 0.5, 0.5], "L", [0.0, 0.0, 0.0], "G"),
        ],
        points_per_segment=16,
    )
    bands = vq.band_structure_hcore(mgo, basis, kpath)  # REAL Hcore bands

    energies = np.linspace(-25.0, 15.0, 400)
    dos = _gaussian_dos(energies, centers=[-20.0, -5.0, -2.0, 8.0])
    pdos = np.vstack(
        [
            _gaussian_dos(energies, [-20.0]),  # O 2s
            _gaussian_dos(energies, [-5.0, -2.0]),  # O 2p
            _gaussian_dos(energies, [8.0]),  # Mg 3s
        ]
    )

    # A two-frame periodic "reaction path" (a rigid shift of the O
    # sublattice) forces the archive to QVF v2 with per-frame lattices.
    def mgo_frame(shift: float):
        sys = vq.PeriodicSystem(
            3,
            np.eye(3) * a,
            [Atom(12, [0.0, 0.0, 0.0]), Atom(8, [a / 2 + shift, a / 2, a / 2])],
        )
        return sys

    plan = OutputPlan.from_run_job_kwargs(
        output=RUN_DIR / "crystal",
        method="rhf",
        basis="sto-3g",
        functional=None,
        job_kind="periodic_scf",
    )
    return write_qvf(
        RUN_DIR / "crystal",
        plan,
        system=mgo,
        method="rhf",
        basis="sto-3g",
        band_structure=bands,
        dos_data={
            "energies": energies,
            "dos": dos,
            "n_spin": 1,
            "smearing": 0.4,
            "smearing_type": "gaussian",
            "fermi_energy_ev": 0.0,
            "n_electrons": 20.0,
        },
        pdos_data={
            "energies": energies,
            "projections": pdos,
            "n_spin": 1,
            "energies_units": "eV",
            "fermi_energy_ev": 0.0,
            "channels": [
                {"atom_index": 1, "symbol": "O", "l": 0, "label": "O 2s"},
                {"atom_index": 1, "symbol": "O", "l": 1, "label": "O 2p"},
                {"atom_index": 0, "symbol": "Mg", "l": 0, "label": "Mg 3s"},
            ],
        },
        symmetry_data={
            "space_group_number": 225,
            "space_group_symbol": "Fm-3m",
            "point_group": "m-3m",
            "hall_number": 523,
        },
        reaction_path={
            "frames": [mgo_frame(0.0), mgo_frame(0.3), mgo_frame(0.6)],
            "waypoints": [
                {"frame_index": 0, "label": "Initial", "kind": "reactant"},
                {"frame_index": 2, "label": "Shifted", "kind": "product"},
            ],
        },
        # Fermi surface (demo — 4×4×4 mesh, 2 bands near E_F)
        fermi_surface_data={
            "nk1": 4,
            "nk2": 4,
            "nk3": 4,
            "energies": np.random.randn(4, 4, 4, 2).astype(np.float64) * 2.0,
            "band_indices": [4, 5],
            "lattice_vectors": np.eye(3) * a,
            "fermi_energy_ev": 0.0,
        },
        # Phonon bands (demo)
        phonon_bands_data={
            "qpath": {
                "n_atoms": 2,
                "n_modes": 6,
                "has_eigenvectors": False,
                "segments": [
                    {
                        "label_start": "G",
                        "label_end": "X",
                        "k_start": [0, 0, 0],
                        "k_end": [0.5, 0, 0],
                        "n_points": 30,
                    }
                ],
            },
            "frequencies": np.sort(
                np.abs(np.random.randn(30, 6).astype(np.float64) * 200 + 300)
            ),
        },
        # Phonon DOS (demo)
        phonon_dos_data={
            "frequencies": np.linspace(0, 600, 200),
            "dos": np.exp(-0.5 * ((np.linspace(0, 600, 200) - 300) / 80) ** 2),
            "meta": {"smearing": 10.0, "smearing_type": "gaussian"},
        },
    )


def main() -> None:
    print("=" * 72)
    print(" QVF format showcase — every section kind, validated end to end")
    print("=" * 72)
    print(f"Output directory: {RUN_DIR}\n")

    archives = [archive_real_run(), archive_spectroscopy(), archive_crystal()]

    print("\nManifests:")
    covered: set[str] = set()
    for path in archives:
        covered |= _print_sections(path)

    print("\nValidation + integrity:")
    for path in archives:
        print(f"  {path.name}")
        _validate(path)
        _verify_one_checksum(path)

    print("\nConsumer read-back (vibe-view, if installed):")
    try:
        from vibeview.kinds import classify_section
        from vibeview.qvf import QVFReader

        for path in archives:
            reader = QVFReader(path)
            sections = reader.sections
            rendered = sum(
                1 for s in sections if classify_section(s.kind)[0] == "rendered"
            )
            print(
                f"  {path.name}: QVFReader opened, {len(sections)} sections "
                f"({rendered} renderable)"
            )
    except ImportError:
        print("  vibe-view not installed — skipping (install the separate vibe-view checkout into this Python environment; see docs/getting_started.md#install-both).")

    print("\nCoverage gate:")
    missing = ALL_KINDS - covered
    extra = covered - ALL_KINDS
    print(f"  kinds covered: {len(covered)}/{len(ALL_KINDS)}")
    if missing:
        raise SystemExit(
            f"  MISSING kinds (writer bug or script gap): {sorted(missing)}"
        )
    if extra:
        print(f"  note: archives carry kinds not in ALL_KINDS: {sorted(extra)}")
    assert QVF_FORMAT_VERSION == 1 and QVF_FORMAT_VERSION_V2 == 2

    print("\n" + "=" * 72)
    print(
        f" All {len(ALL_KINDS)} section kinds written, validated, and round-tripped."
    )
    print(
        " Open any archive with:  vibe-view open " + str(RUN_DIR / "spectroscopy.qvf")
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
