# vibe-view example calculations

Sample `run_job` / `run_periodic_job` inputs whose output is meant to
be opened in [vibe-view](../../docs/user_guide/vibe_view.md), the
GPU-accelerated interactive 3D viewer for vibe-qc's `.qvf` archives.

Every script here passes `output_qvf=True` so that a `{stem}.qvf`
lands next to the usual `.out` / `.molden` / `.bibtex` siblings.

## Running

These calculation inputs remain in **vibe-qc**. The viewer is installed from
its separate [vibe-view repository](https://github.com/vibe-qc/vibe-view);
follow [viewer setup](../../docs/tutorial/vibe_view_getting_started.md) first.
The commands below use the core environment's Python to generate the archive
and the separately installed `vibe-view` command on PATH to open it. You may
also invoke the viewer by its absolute executable path. The two tools exchange
QVF files and do not need the same virtual environment. The example paths
below are in this core checkout; they are not installed into the viewer
checkout. For a viewer-only sample, use its `vibe-view examples` command.

```sh
# From the vibe-qc checkout root
.venv/bin/python examples/vibe_view/showcase_formaldehyde.py
vibe-view open examples/vibe_view/runs/h2co_showcase/h2co.qvf
```

For the optional `vibeview.QVFReader` check in `showcase_qvf_all_sections.py`,
install both projects explicitly into one environment as shown in
[Install both](../../docs/getting_started.md#install-both). Core archive generation
and checksum validation work without the viewer or the QVF reference toolkit.

vibe-view boots the Trame server at `http://127.0.0.1:8080` and opens
your browser there. Press Ctrl+C in the terminal to stop the server.

The server runs until you stop it, so opening a second example on the
same port aborts with `port 8080 is already in use`; stop the first
(`pkill -f "vibe-view open"`) or pass `--port`. If an already-open tab
stops responding to the mouse, its server was stopped; relaunch and use
the new tab. ([Troubleshooting](../../docs/user_guide/vibe_view.md#common-pitfalls).)

## Scripts

| Script | What it shows |
|---|---|
| `showcase_formaldehyde.py` | The common viewer-renderable sections on a small molecular calculation. RHF / 6-31G* on H2CO with FD Hessian (vibrational modes + IR spectrum), HOMO + LUMO + density isosurfaces, the embedded `wavefunction.gto`, Mulliken / Loewdin charges, SCF history, and the auto-assembled citation bundle. ~60 seconds on a modern laptop. |
| `showcase_nacl_rocksalt.py` | Every periodic-specific rendering path: NaCl rocksalt at RKS-PBE / STO-3G with unit-cell wireframe + replication controls, `volume.density` replication, ionic `atom_properties`, and citations. |
| `showcase_co_on_mgo.py` | CO adsorbed on an MgO(100) slab — slab rendering plus adsorbate-interaction density. |
| `showcase_qvf_all_sections.py` | The whole format surface. Produces three archives that between them exercise **27 of the 35** viewer-renderable section kinds the QVF writer implements (plus the v2 periodic reaction path) and validates each. The viewer renders 26 of them as standalone panels; the 27th, `bonds`, carries no panel of its own — its connectivity is folded into the structure renderer. The two kinds not covered — `basis.ao` and `scan.surface` — are exercised by [`showcase_basis_functions.py`](showcase_basis_functions.py) and the viewer's test fixtures. The companion to [Tutorial 47](../../docs/tutorial/qvf_file_format.md); run it as a writer stress test. |
| `showcase_basis_functions.py` | Evaluates individual atomic-orbital basis functions on a 3D grid (H2O / pob-TZVP) and packs them as `basis.ao` sections — the producer-side counterpart to vibe-view's basis-function picker; every written AO can be selected and rendered as an isosurface. |
| `showcase_natural_orbitals.py` | CASSCF natural orbitals in a QVF archive: H2/6-31G/CAS(2,2), the textbook two-configuration case. Generates a local archive corresponding to [`tests/data/h2_natural_orbitals.qvf` in vibe-view](https://github.com/vibe-qc/vibe-view/blob/main/tests/data/h2_natural_orbitals.qvf), the fixture pinning the viewer's HONO/LUNO path against computed (not invented) data. See [Multireference: CASSCF, CASPT2, NEVPT2 § Natural orbitals and QVF wavefunction export](../../docs/tutorial/casscf_multireference.md#natural-orbitals-and-qvf-wavefunction-export). |
| `showcase_neb_reaction.py` | Two `reaction.path`-headlined archives (plus a DFT+U variant) — the section vibe-view's reaction renderer animates frame-by-frame with reactant/TS/product waypoints and an energy profile. See [NEB reaction path](../../docs/tutorial/neb_reaction_path.md). |
| `showcase_msindo_h2s.py` | H2S via MSINDO (semiempirical INDO, exercising the 3d-polarization/d-shell path with a Ne frozen core) — basis-set-free, Slater orbitals only. See [MSINDO](../../docs/tutorial/msindo.md). |

## Adding a new showcase

Pick a system / method combination that exercises a sub-set of
sections vibe-view renders that the existing scripts do not. Open
a feature request first if you want to land it: the goal is to keep
this directory small and pedagogical, not exhaustive. Per-feature
how-tos belong in `docs/tutorial/`; this directory is for runnable
inputs you can hand straight to a `python` interpreter.

## See also

- [User guide § vibe-view](../../docs/user_guide/vibe_view.md):
  install, launch, UI walkthrough, per-section behaviour.
- [Tutorial 45: vibe-view walkthrough](../../docs/tutorial/vibe_view_walkthrough.md):
  end-to-end worked example with screenshots.
- [Tutorial 47: the QVF file format](../../docs/tutorial/qvf_file_format.md):
  every section kind, written and validated end to end.
- [QVF format design](../../docs/design_qvf_format.md): the
  archive format these scripts produce.
