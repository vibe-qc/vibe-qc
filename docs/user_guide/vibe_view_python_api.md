---
myst:
  html_meta:
    "description": "The supported vibe-view Python API: open a QVF from a path, bytes or a file object, read structures, volumes and tables, render PNGs headlessly, compare and slice files, and launch the viewer from a script."
    "og:title": "vibe-view Python API"
    "og:description": "Script vibe-view instead of driving its CLI: the seventeen supported names in vibeview.__all__, with a runnable example for each."
---

# vibe-view Python API

Source commands on this page run from the **separate vibe-view checkout**.
Clone [mpei/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](../tutorial/vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

Everything the [command reference](vibe_view_cli.md) does from a shell, you can
do from Python. This page documents the **supported** surface: the names
exported from the top-level `vibeview` package.

The product and command are spelled `vibe-view`; the import package is spelled
`vibeview`.

## What is supported, and what is not

```{important}
**`vibeview.__all__` is the API contract.** A name you import from the top-level
package is supported and will not move without a deprecation note in the
CHANGELOG. A name you reach by importing a submodule — `vibeview.capture`,
`vibeview.tables`, `vibeview.align` — is an internal detail that can be renamed
or restructured in any release.
```

Some existing examples in
[Working with vibe-view](../tutorial/working_with_vibe_view.md) import from
submodules. Those still work, but prefer the top-level import for new code:

```python
from vibeview import capture_structure   # supported
```

instead of

```python
from vibeview.capture import capture_structure   # internal path
```

## Install

The **base install covers this entire page except {func}`vibeview.launch_qvf`.** Reading,
inspecting, comparing, slicing and headless PNG capture all work with no extra,
because PyVista, matplotlib and plotly are core dependencies:

```sh
pip install .
```

Only the interactive browser viewer needs the `[viewer]` extra, which adds the
trame web-server stack:

```sh
pip install '.[viewer]'
```

```{note}
The `capture` extra exists but is deliberately **empty**. It documents that a
headless screenshot host needs nothing beyond the base install; you never have
to install it.
```

## One argument type for every function

Every function on this page takes a `QVFSource` as its first argument. That is
a union, not a class, anything in it works interchangeably:

```python
str | Path | bytes | bytearray | memoryview | IO[bytes]
```

So the same call reads a file on disk, an in-memory archive, or an open file
object:

```python
from pathlib import Path
from vibeview import info

info("water.qvf")                        # path as a string
info(Path("water.qvf"))                  # Path
info(Path("water.qvf").read_bytes())     # raw bytes, no disk round-trip
with open("water.qvf", "rb") as fh:      # any seekable binary file object
    info(fh)
```

A `QVFReader` is also accepted anywhere a source is, which is how you avoid
reopening the archive, see [Reading a file repeatedly](#reading-a-file-repeatedly).

An unreadable or corrupt file raises `QVFOpenError`.

## Inspecting a file

`info` returns the full metadata dict; `sections` lists what the archive
contains; `has_section` tests one by id; `validate` re-checks the schema and
every SHA-256 in the manifest.

```python
from vibeview import info, sections, has_section, validate

info("analysis.qvf")["calculation"]
# 'analysis'

sections("analysis.qvf")
# [{'id': 'structure', 'kind': 'structure'},
#  {'id': 'atom_properties', 'kind': 'atom_properties'},
#  {'id': 'bond_orders', 'kind': 'bond_orders'},
#  {'id': 'scf_history', 'kind': 'scf_history'}]

has_section("analysis.qvf", "scf_history")
# True

validate("analysis.qvf")
# {'valid': True, 'n_sections': 4, 'n_members': 5}
```

`validate` is the one to call before trusting a file you did not produce: it
hashes every member and reports `valid: False` rather than raising, so it is
safe to run over a directory.

## Reading the structure

```python
from vibeview import get_structure, export_xyz

get_structure("water.qvf")
# {'atoms': [{'symbol': 'O', 'atomic_number': 8, 'position': [0.0, 0.0, 0.1173]},
#            {'symbol': 'H', 'atomic_number': 1, 'position': [0.0, 0.7572, -0.4692]},
#            ...],
#  'pbc': [False, False, False]}

print(export_xyz("water.qvf"))
# 3
# vibe-view export
# O      0.000000     0.000000     0.117300
# H      0.000000     0.757200    -0.469200
# ...
```

For a periodic system `get_structure` also carries `lattice_vectors`, and `pbc`
reports which axes are periodic.

## Volumes: summary vs. voxels

{func}`vibeview.get_volume` returns a **JSON-serializable summary**, the grid geometry
and the value range, not the voxels:

```python
from vibeview import get_volume

get_volume("density.qvf", "rho")
# {'kind': 'volume.density',
#  'origin': [-1.5, -1.5, -1.5],
#  'voxel_vectors': [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
#  'shape': [6, 6, 6],
#  'data_shape': [6, 6, 6],
#  'data_min': 0.0027385002467781305, 'data_max': 0.997209906578064}
```

It returns `None`, rather than raising, when the section is absent or is not
a volume, so it doubles as a probe.

When you want the **array itself**, go through a reader:

```python
from vibeview import QVFReader

with QVFReader("density.qvf") as r:
    grid = r.read_volume_grid("rho")   # origin, voxel_vectors, shape
    data = r.read_volume_data("rho")   # ndarray, here (6, 6, 6) float32
```

## Tables

`get_table` returns `(headers, rows)`, the same data the `vibe-view table`
command prints, ready for `csv.writer` or a DataFrame:

```python
from vibeview import get_table

headers, rows = get_table("analysis.qvf", "atom_properties")
# headers: ['atom', 'element', 'mulliken', 'loewdin', 'hirshfeld']
# rows[0]: [1, 'O', -0.34, -0.2, None]
```

```{note}
Three kinds are tabulatable: **`vibrations`**, **`atom_properties`** and
**`wavefunction.gto`**. Any other kind raises `ValueError` naming the valid set.
A missing value is `None`, not a blank string — check for it before formatting.
```

## Comparing two files

```python
from vibeview import diff

diff("hf.qvf", "pbe.qvf")
# {'energy_a_eh': ..., 'energy_b_eh': ..., 'delta_e_eh': ...,
#  'delta_e_kcal_mol': ..., 'delta_e_ev': ...,
#  'geo_rmsd_a': ..., 'n_sections_a': ..., 'n_sections_b': ...,
#  'kinds_only_a': [], 'kinds_only_b': [], 'kinds_common': ['structure'],
#  'converged_a': ..., 'converged_b': ...}
```

Every numeric field is `None` when the inputs cannot support it, comparing two
structure-only files gives `delta_e_eh: None` rather than a spurious zero. Test
for `None` before arithmetic.

## Rendering

`capture_structure` and `capture_volume` write PNGs with no display attached
and return `True` on success. Both are core, no extra needed:

```python
from vibeview import capture_structure, capture_volume

capture_structure("water.qvf", "water.png")
capture_volume("density.qvf", "rho", "rho.png")
```

```{tip}
On a headless machine set `PYVISTA_OFF_SCREEN=True` before importing, or
PyVista warns that it may segfault without an X server.
```

`render_terminal` returns a string of braille glyphs rather than writing a file
- useful for logs, CI output and SSH sessions:

```python
from vibeview import render_terminal

print(render_terminal("water.qvf", size=(60, 12), plain=True))
```

`plain=True` drops ANSI colour, which is what you want when the destination is
a log file. Pass a `section_id` to pick a section; omitted, it renders the
file's default.

## Slicing

`slice_qvf` writes a new archive containing a subset of sections and returns
the output `Path`. Use `keep` or `drop`:

```python
from vibeview import slice_qvf

out = slice_qvf("analysis.qvf", "structure_only.qvf", keep=["structure"])
sections(out)
# [{'id': 'structure', 'kind': 'structure'}]
```

This is the cheap way to hand a colleague a 2 MB structure out of a 400 MB
archive that also holds volumes.

## Reading a file repeatedly

Each module-level call opens and closes the archive. When you make several
calls against one file, open a `QVFReader` once and pass **it** as the source -
every function on this page accepts one:

```python
from vibeview import QVFReader, info, get_structure, export_xyz

with QVFReader("analysis.qvf") as r:
    meta = info(r)
    geom = get_structure(r)
    xyz = export_xyz(r)
```

The reader is also the lower-level surface, with a `read_*` method per section
kind. Note that `sections` is a **property** here, returning rich `Section`
objects, unlike the module-level `sections()` function, which returns plain
`{id, kind}` dicts:

```python
with QVFReader("analysis.qvf") as r:
    [s.id for s in r.sections]
    # ['structure', 'atom_properties', 'bond_orders', 'scf_history']

    r.read_scf_history("scf_history")
    # SCFHistoryData(iterations=[{'iter': 1, 'energy_eh': -75.0, ...}, ...])

    r.read_bond_orders("bond_orders")
    # BondOrdersData(method='mayer', pairs=[{'i': 0, 'j': 1, 'order': 0.98, ...}])
```

Each `read_*` method takes the **section id**, not the kind. Ids often match
the kind in small files, but not always, take them from `r.sections`.

Without the context manager, call `r.close()` yourself.

## Launching the viewer

{func}`vibeview.launch_qvf` boots the interactive Trame server. This is the one function
that needs the `[viewer]` extra:

```python
from vibeview import launch_qvf

launch_qvf("water.qvf", host="127.0.0.1", port=8080, open_browser=True)
```

It blocks while the server runs. Set `open_browser=False` on a remote host and
forward the port; `print_banner_to_stdout=False` silences the startup banner
when you are embedding it.

## Full surface

| Name | Purpose | Extra |
|---|---|---|
| `QVFReader` | Open once, read many sections |, |
| `QVFSource` | The accepted source union (a type, not a class) |, |
| `QVFOpenError` | Raised when a file cannot be opened |, |
| `info` | Full metadata dict |, |
| `sections` | `{id, kind}` for every section |, |
| `has_section` | Test one section id |, |
| `validate` | Schema + SHA-256 integrity check |, |
| `get_structure` | Atoms, `pbc`, `lattice_vectors` |, |
| `export_xyz` | Structure as an XYZ string |, |
| `get_volume` | Grid geometry + value range (no voxels) |, |
| `get_table` | `(headers, rows)` for tabulatable kinds |, |
| `diff` | Energy delta, section overlap, geometry RMSD |, |
| `capture_structure` | Structure → PNG, headless |, |
| `capture_volume` | Volume → PNG, headless |, |
| `render_terminal` | Section → braille text |, |
| `slice_qvf` | Subset of sections → new QVF |, |
| `launch_qvf` | Interactive browser viewer | `[viewer]` |

## Extending vibe-view

There is **no plugin API today**, and this section says plainly what that means
rather than implying an extension surface that does not exist.

What you *can* do now is composition: the functions above are ordinary Python,
so a script that walks a directory, validates each archive, captures a PNG and
writes a summary CSV needs nothing from vibe-view but the calls on this page.
For most "extend vibe-view" requests, batch rendering, custom reports, feeding
another tool, that is the whole answer, and it is stable.

What is **not** available is in-process extension: you cannot register a custom
representation, a new importer, or your own panel in the viewer and have
vibe-view discover it. Those seams exist inside the codebase, but they are
internal, not entry points, not documented, and not stable across releases.

A real plugin API would need, at minimum:

- **A discovery mechanism**, a `vibeview.plugins` entry-point group, so an
  installed package registers itself without the user editing vibe-view.
- **A stable representation interface**, the contract a custom renderer
  implements to add a style alongside ball-and-stick, ribbons and surfaces.
- **An importer hook**, third-party formats resolving through the same
  `converters.is_supported_path` predicate the browse page and the opener
  already share, so a plugin format is offered wherever a native one is.
- **A panel-injection contract** for the Trame UI, which is the piece most
  entangled with internals today.
- **A versioning policy**, since a plugin API is a promise to not break
  third-party code, the reason it is not worth declaring casually.

If you need one of these, say what you are building in an
[issue](https://github.com/vibe-qc/vibe-qc/issues). Which seam gets
stabilized first should follow a real use case rather than a guess.

## See also

- [vibe-view command reference](vibe_view_cli.md), the same capability from a shell.
- [Working with vibe-view](../tutorial/working_with_vibe_view.md), end-to-end
  walkthroughs, including a scripted workflow.
- [Supported formats](vibe_view_formats.md), what vibe-view reads besides QVF.
