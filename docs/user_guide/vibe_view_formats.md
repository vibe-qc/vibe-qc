---
myst:
  html_meta:
    "description": "Open or import chemistry files from Gaussian, ORCA, VASP, molecular dynamics tools, and other producers with vibe-view."
    "og:title": "vibe-view input formats and interoperability"
---

# Input formats and interoperability

Source commands on this page run from the **separate vibe-view checkout**.
Clone [mpei/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](../tutorial/vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

vibe-view is a viewer and conversion tool for quantum-chemistry data. QVF is
its richest input, but the producer does not have to be vibe-qc. You can open
common structure and volume files directly or turn them into a persistent QVF
archive.

## The three names

The spelling depends on where you use the name:

| Context | Name | Example |
|---|---|---|
| Product and shell command | **vibe-view** | `vibe-view open water.qvf` |
| Python distribution | **vibeview** | a wheel named `vibeview-2.15.2-...whl` |
| Python import package | **vibeview** | `from vibeview import QVFReader` |
| Independent source repository | [mpei/vibe-view](https://github.com/vibe-qc/vibe-view) | `./scripts/install.sh` |

The distribution is not on PyPI yet. A bare `pip install vibeview` or
`pipx install vibeview` therefore does not install it today. Use the checkout
installer or the downloadable wheel described in
[Getting started with vibe-view alone](../tutorial/vibe_view_getting_started.md).

```{important}
The legacy 2.15.2 wheel carries the `formats`,
persistent `import`, and importer-plugin workflow on this page.
`vibe-view --help` is the source of truth for an installed version.
```

## Open now or import once

Use `open` when you only want to look at a file:

```sh
vibe-view open molecule.xyz
vibe-view open density.cube
vibe-view open POSCAR
```

The conversion exists in memory for that viewer session. The source file is
not changed and no QVF is written.

Use `import` when you want a reusable, checksummed QVF:

```sh
vibe-view import molecule.xyz
# writes molecule.qvf beside molecule.xyz

vibe-view import density.cube -o density.qvf
vibe-view import ./results/*.xyz -o ./qvf-results/
vibe-view import ./results -o ./qvf-results/
```

One input defaults to a sibling file with the `.qvf` suffix. Multiple inputs,
or a recursively searched directory, default to separate archives under
`./vibe-view-imports`.
Existing outputs are protected; pass `--force` only when replacing them is
intentional. If an extension is ambiguous, select an importer explicitly:

```sh
vibe-view import calculation.data --from xyz -o calculation.qvf
```

An import processes each source independently. It does not merge an XYZ,
Cube, log, and wavefunction sidecar into one calculation archive. A producer
that wants the full viewer surface should write QVF directly; see
[Adopting QVF in another code](../tutorial/qvf_adapt_and_writer.md).

List the importers available in the current environment with:

```sh
vibe-view formats
```

The list includes built-ins, optional importers whose dependencies are
missing, and third-party importers registered through the
`vibeview.importers` Python entry-point group. The command is the source of
truth for the installed version.

### Author a third-party importer

An importer plugin declares one entry point whose name is also the stable
format name used by `vibe-view import --from`. For example, in the plugin's
`pyproject.toml`:

```toml
[project.entry-points."vibeview.importers"]
orca-output = "my_orca_importer:importer"
```

Expose either an `ImporterSpec` or a zero-argument factory that returns one:

```python
from pathlib import Path

from vibeview.importers import IMPORTER_API_VERSION, ImporterSpec


def convert_orca_output(path: Path) -> bytes:
    # Parse path and return one complete QVF archive as bytes.
    return build_qvf_bytes(path)


def importer() -> ImporterSpec:
    return ImporterSpec(
        format_name="orca-output",
        description="ORCA text output",
        extensions=(".out",),
        convert=convert_orca_output,
        data_kinds=("structure", "properties"),
        api_version=IMPORTER_API_VERSION,
    )
```

`extensions` must include the leading dot. Use `stems=("NAME",)` for exact
extensionless filenames, or a `probe(path) -> bool` callback when names are
not sufficient. `convert` receives a `pathlib.Path` and returns a complete
QVF as bytes or a readable binary file object; vibe-view copies and closes a
returned file object. Importer API version 1 is the current contract. Format
names used by built-ins are reserved, and duplicate plugin names or duplicate
extension/stem claims fail closed instead of choosing one package silently.
vibe-view validates every member and checksum in the returned QVF before it
writes or opens it, isolates a broken plugin from other importers, and reports
plugin availability or load errors in `vibe-view formats`.

## Built-in format and capability matrix

"Kept" means the information reaches the temporary or persistent QVF. It
does not mean every possible record in the source format is interpreted.

| Input | Extra | Kept | Important limits |
|---|---|---|---|
| QVF (`.qvf`) | none | All valid sections, metadata, and checksums | This is the only built-in path for a complete calculation record. |
| XYZ (`.xyz`) | none | One molecular geometry, coordinates treated as Å | No cell, bonds, properties, or reliable multi-frame trajectory import. |
| CIF (`.cif`) | none | Explicit atom sites and cell parameters | Symmetry operations are not expanded. Supply an already expanded cell when the CIF stores only an asymmetric unit. |
| Gaussian Cube (`.cube`) | none | Structure and one scalar volume | The volume kind is inferred from comments and defaults to density. No wavefunction, spectra, or calculation provenance. `.cub` is not a recognized suffix. |
| PDB (`.pdb`) | none | Coordinates, atom/residue/chain names, B factors, and `CRYST1` cell | Explicit connectivity is not imported. Use a cleaned single model; alternate locations and multiple models are not resolved. |
| Tripos Mol2 (`.mol2`) | none | Molecular geometry and explicit bond orders | Charges, force-field types beyond element recognition, and most substructure metadata are not retained. |
| Gaussian input (`.gjf`, `.com`) | none | Cartesian molecular geometry | Route, basis, constraints, and Z-matrix input are not interpreted. Gaussian output, checkpoint, and formatted-checkpoint files are not supported. |
| GROMACS (`.gro`) | none | Coordinates and a three-length orthorhombic box | Velocities and residue metadata are not retained; triclinic box terms are not imported. |
| MDL Mol/SDF (`.mol`, `.sdf`) | none | First V2000 molecule and bond orders | Later SDF records and data fields are ignored; V3000 is not supported. |
| vibe-qc Python input (`.py`) | none | Statically parsed structure and selected calculation settings | The script is not executed. This parser is for vibe-qc input syntax, not arbitrary Python. |
| ASE trajectory (`.traj`), extended XYZ (`.extxyz`), VASP (`.vasp`, `.poscar`, `POSCAR`, `CONTCAR`) | `[ase]` | One ASE-readable geometry, cell, and periodic flags | Only these names are routed to ASE. Per-atom arrays, trajectories, calculator results, and other ASE formats are not retained by the built-in importer. |

The browser can infer display bonds when a format carries no connectivity.
Those visual bonds are not evidence that the source contained bond orders.

## Routes from common chemistry programs

| Producer | Best route today | Not imported directly |
|---|---|---|
| Gaussian | Open/import `.gjf` or `.com` for input geometry; use a `.cube` for one orbital, density, or potential | `.log`, `.out`, `.chk`, `.fchk` |
| ORCA | Export `.xyz` for geometry or `.cube` for one scalar field | `.out`, `.gbw`, `.hess`, `.molden.input` |
| VASP | Install `[ase]`, then open/import `POSCAR` or `CONTCAR` | `CHGCAR`, `LOCPOT`, `WAVECAR`, `DOSCAR`, `PROCAR` |
| CP2K, Quantum ESPRESSO, Psi4, PySCF, and similar codes | Export XYZ/CIF for structure or Cube for one volume; alternatively add a QVF writer | Native logs, restart files, and code-specific wavefunction files |
| Molecular-dynamics tools | Use PDB, GRO, Mol2, SDF, XYZ, or one of the explicitly listed ASE routes | Multi-frame trajectories are not yet a general import path |

For loose Molden, XSF, formatted-checkpoint, and GBW-style terminal workflows,
[MolTUI](../tutorial/moltui_terminal_viewer.md) may be a better fit. It is a
separate format-oriented viewer. For rich portable results, use the
[QVF toolkit](../qvf/index.md), whose reference writers are independent of
vibe-qc.

## Which commands accept loose inputs?

`open` accepts the built-in loose formats and converts them in memory.
`import` creates a QVF that every QVF-oriented command can consume:

```sh
vibe-view import calculation.cube -o calculation.qvf
vibe-view show calculation.qvf
vibe-view capture calculation.qvf -o calculation.png
vibe-view validate calculation.qvf
```

Commands such as `show`, `tui`, `capture`, `info`, `validate`, `export`,
`slice`, and `merge` operate on QVF archives. Import first when your source is
a loose file.

## Diagnose an environment before blaming the file

```sh
vibe-view doctor
vibe-view formats
vibe-view capture-selftest
```

`doctor` reports the installed version, Python environment, packaged
resources, optional modes, and source-desktop support. `formats` reports
built-in, optional, and plugin importer availability.
`capture-selftest` isolates the VTK/OpenGL capture path from the contents of a
particular archive. Browser server logs default to
`~/.cache/vibe-view/vibe-view.log` unless `XDG_CACHE_HOME` is set.
