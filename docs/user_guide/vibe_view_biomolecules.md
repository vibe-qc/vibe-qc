# Biomolecules: cartoon rendering, chains and residues

A `structure` section can carry optional **biomolecule metadata**: chain ids,
residues, secondary structure and b-factors. When it does, vibe-view can draw
the structure as a **cartoon** (a ribbon through the backbone) instead of
plotting every atom, colour that ribbon by chain, secondary structure, residue
type or b-factor, and highlight a residue range you name.

This page covers the whole path: getting the metadata into a `.qvf`, what the
viewer does with it, and what happens when it is absent.

```{note}
Cartoon rendering is a **viewer** feature over **optional format** fields. A
`structure` section carrying none of them is a perfectly valid plain structure,
and a consumer that does not understand the fields must ignore them. Nothing
here changes how an ordinary molecular or periodic calculation is written.
```

## Getting the metadata into an archive

### From a PDB file

The simplest route: vibe-view imports `.pdb` directly, and the importer carries
the per-atom biomolecular columns through into the `structure` section. Open a
PDB in the viewer, or convert it, and the cartoon is immediately available.

The per-atom fields the PDB columns map to are `atom_name` (columns 13 to 16,
for example `"CA"`), `residue_name` (18 to 20), `residue_seq` (23 to 26),
`chain_id` (column 22) and `b_factor` (61 to 66).

### From a vibe-qc job

`write_qvf` takes a `biomolecule_data` context argument, a dict with four
independently optional keys that become peer keys on the `structure` **section
object**:

```python
from vibeqc.output.formats.qvf import write_qvf

write_qvf(
    "protein", plan,
    molecule=mol,
    biomolecule_data={
        "chains": ["A", "B"],
        "residues": [
            {"name": "MET", "seq": 1, "chain": "A", "atom_indices": [0, 1, 2, 3]},
            {"name": "ALA", "seq": 2, "chain": "A", "atom_indices": [4, 5, 6, 7]},
            # ... in chain order, which is the order the ribbon is drawn in
        ],
        "secondary_structure": [
            {"type": "helix", "chain": "A", "start_seq": 12, "end_seq": 27},
            {"type": "sheet", "chain": "A", "start_seq": 34, "end_seq": 39},
        ],
        "b_factors": [12.4, 15.1, ...],   # one per atom, in payload order
    },
)
```

Four things worth getting right:

* **`atom_indices` are 0-based** into the `structure` payload's `atoms` array.
* **`residues` go in chain order**, the order a ribbon is drawn in, *not*
  sorted by `seq`. Insertion codes and non-monotonic numbering are both legal,
  so sorting would silently reorder them.
* **`seq` is not a global key.** PDB residue numbers are four columns wide and
  wrap at 9999, so a solvated system reuses them. Identify a residue by chain
  plus seq, never by seq alone.
* **`secondary_structure` ranges are inclusive**, and `type` is one of `helix`,
  `sheet` or `coil`.

The full normative contract is in the
[format specification](../qvf/spec.md), section 5.1, under "Biomolecule
metadata".

### Supplied beats inferred

A consumer can derive chains, residues and secondary structure on its own, from
the per-atom fields and from CA geometry. vibe-view does exactly that when the
archive supplies nothing. But where a producer *does* supply them, the supplied
values win: a producer knows what geometry cannot reveal, such as an alpha from
a pi from a 3-10 helix, or a beta bridge from a sheet.

Two related rules keep b-factors unambiguous:

* A per-atom `b_factor` beats the section-level `b_factors` array, because it
  cannot desynchronize from its own atom.
* A `b_factors` array whose length differs from the atom count is ignored
  outright rather than applied partially.

## Drawing the cartoon

In the viewer's **Display** card, set **Representation** to
**Cartoon (biomolecule)**. The ribbon replaces the atom spheres and bonds.

```{note}
A cartoon needs a backbone. On a structure with no residue identity (an XYZ, or
a molecule you just computed) the picker refuses the switch and says so in the
status line rather than leaving you looking at an unchanged scene. Rendering
through the Python API on such a structure falls back to ball-and-stick.
```

The structure renderer keeps geometry loading separate from bond inference.
Its initial periodicity check and cartoon render share cached geometry, while
other startup metadata paths also stay geometry-only. Opening a protein
therefore does not construct connectivity that the ribbon never draws. If you
later switch to a bond-bearing representation, the renderer initializes that
connectivity once and caches it.

### Ribbon colour

The **Ribbon colour** picker appears next to the representation once the
cartoon is active. Four modes:

| Mode | Colours by |
|---|---|
| **Chain** (default) | one hue per chain |
| **Secondary structure** | helix / sheet / loop |
| **Residue type** | the residue's identity |
| **B-factor** | a ramp over this structure's b-factor range |

The b-factor ramp is normalized over the structure you have open, so the
colours are meaningless without the range they span: selecting it reports that
range in the status line. If the file carries no b-factors at all, the ribbon
falls back to chain colour and the status line says so, rather than leaving the
mode looking like it silently failed.

### Highlighting residues

The **residue selection** field (shown when the structure has CA-bearing
residues) takes a selection string. It works in every structure
representation. In the cartoon, matching ribbon samples render **white** over
the active colour mode. In ball-and-stick and space-filling, every atom in a
matching residue renders white. In sticks-only and wireframe, a bond renders
white when both endpoints belong to matching residues; a boundary bond to the
next residue remains unhighlighted so the selection does not bleed.

| Selection | Selects |
|---|---|
| `A` | all of chain A |
| `A/24-38` | residues 24 to 38 of chain A |
| `*/24-38` | residues 24 to 38 in every chain |
| `A/24-38, B/10` | comma or space separated terms |

The field reports what it matched: the residue count and the chains involved,
or "nothing selected". A chain id it does not recognize, or a term it cannot
read, is called out separately. That summary is the point of the control: a
selection matching nothing has to say so, or it reads as a render bug.

A chainless PDB groups its atoms under an empty chain id, which you cannot
type. Reach those residues with the `*` wildcard.

## Outside the browser

**Terminal mode** does not draw a ribbon, but it does read the same metadata
for colouring:

```sh
vibe-view show protein.qvf --color-by chain
vibe-view show protein.qvf --color-by secondary
vibe-view show protein.qvf --color-by bfactor
```

See [Terminal mode](vibe_view_terminal_mode.md) for the rest of that surface.

**Headless PNG capture** takes the representation as a keyword:

```python
from vibeview import capture_structure

capture_structure("protein.qvf", "ribbon.png", representation="cartoon")
```

Like every other SDK function it also accepts an already-open `QVFReader`, and
a reader you pass in stays open.

`capture_structure` renders the ribbon with the default chain colouring and no
residue selection; those two are viewer controls. For full control from
Python, drive `StructureRenderer.add_to_plotter` directly, which accepts
`representation`, `cartoon_color_mode` and `residue_selection`.

## See also

* [vibe-view: interactive viewer](vibe_view.md), the rest of the viewer.
* [Terminal mode](vibe_view_terminal_mode.md), the same metadata over SSH.
* [Working with vibe-view](../tutorial/working_with_vibe_view.md), the wider
  feature tour: compare mode, exports, capture, presets.
* [QVF format specification](../qvf/spec.md) section 5.1, the normative
  biomolecule contract.
