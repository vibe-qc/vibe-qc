---
myst:
  html_meta:
    "description": "Licensing inventory for vibe-qc: own MPL 2.0 license, vendored / linked native libraries, bundled basis sets, ECP libraries, and external data sources fetched by vqfetch."
    "og:title": "vibe-qc, license + bundled-data inventory"
    "og:description": "Full license-and-attribution inventory: MPL 2.0 source, linked libraries (libint LGPL, FFTW GPL, libxc MPL, spglib BSD, libecpint MIT, …), bundled basis sets, ECPs, vqfetch data sources."
---

# License + bundled-data inventory

vibe-qc is distributed under the
[**Mozilla Public License 2.0**](https://www.mozilla.org/en-US/MPL/2.0/).
The full license text lives in the repository root as
[`LICENSE`](https://github.com/vibe-qc/vibe-qc/blob/main/LICENSE).

This page covers **three** related but distinct things, in the
order users tend to ask:

1. **vibe-qc's own license** (your right to use it).
2. **Linked native libraries** (whose licenses transitively
   bind any binary distribution of vibe-qc).
3. **Bundled data**, basis sets, ECP libraries, and external
   structures / reference data fetched by `vqfetch`.

If you want the legal short version: vibe-qc itself is MPL 2.0
and you can use it freely in academic and commercial work. **The
combined binary that links FFTW3 is effectively GPL v2** (the
restriction comes from FFTW, not from us); see
[FFTW: GPL v2 dependency](#fftw-gpl-v2-dependency-the-binary-distribution-caveat)
below.

## 1. vibe-qc's own license, MPL 2.0

vibe-qc's source code is released under the
[Mozilla Public License Version 2.0](https://www.mozilla.org/en-US/MPL/2.0/).
What that means in practice:

* You can use vibe-qc in commercial and non-commercial projects.
* If you modify vibe-qc's own source files **and redistribute
  them**, your modifications to those files must be released
  under MPL 2.0. You don't have to open-source code that
  merely *uses* vibe-qc.
* No patent traps: MPL 2.0 includes an explicit patent grant.
* No warranty, see the license text for the usual limitations
  of liability.
* MPL 2.0 is **GPL-compatible** (vibe-qc has no "Incompatible
  With Secondary Licenses" notice), so vibe-qc source can be
  combined with GPL'd code into a "Larger Work" under
  MPL 2.0 § 3.3.

## 2. Linked native libraries

vibe-qc dynamically links the following native libraries at
runtime. Their licenses transitively bind any binary
distribution of vibe-qc.

| Library | License | Source | Linkage |
|---------|---------|--------|---------|
| [libint](https://github.com/evaleev/libint) | **LGPL 3.0** (library) + GPL 3.0 (compiler) | [LICENSE](https://github.com/evaleev/libint/blob/master/LICENSE) | dynamic |
| [libxc](https://libxc.gitlab.io/) | MPL 2.0 | [COPYING](https://gitlab.com/libxc/libxc/-/blob/master/COPYING) | dynamic |
| [Eigen](https://eigen.tuxfamily.org/) | MPL 2.0 | [COPYING.MPL2](https://gitlab.com/libeigen/eigen/-/blob/master/COPYING.MPL2) | header-only |
| [spglib](https://github.com/spglib/spglib) | BSD 3-Clause | [COPYING](https://github.com/spglib/spglib/blob/develop/COPYING) | dynamic |
| [pybind11](https://github.com/pybind/pybind11) | BSD 3-Clause | [LICENSE](https://github.com/pybind/pybind11/blob/master/LICENSE) | header-only |
| [FFTW3](https://www.fftw.org/) | **GPL v2 or later** (commercial alt available) | [License](https://www.fftw.org/doc/License-and-Copyright.html) | dynamic |
| [libecpint](https://github.com/robashaw/libecpint) | MIT | [LICENSE](https://github.com/robashaw/libecpint/blob/master/LICENSE) | dynamic |
| [libcerf](https://github.com/jschueller/libcerf) (in libecpint) | MIT | [LICENSE](https://github.com/jschueller/libcerf/blob/master/LICENSE) | static (in libecpint) |
| [pugixml](https://pugixml.org/) (in libecpint) | MIT | [LICENSE.md](https://github.com/zeux/pugixml/blob/master/LICENSE.md) | static (in libecpint) |
| BLAS, Apple Accelerate (macOS, default) | Apple SDK license (system-bundled) | shipped with macOS, no redistribution | dynamic (framework) |
| BLAS, [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) (Linux, recommended; system *or* vendored) | BSD 3-Clause | [LICENSE](https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE) | dynamic |
| BLAS, [reference netlib BLAS](https://www.netlib.org/blas/) (Linux fallback) | modified BSD | [LICENSE](https://github.com/Reference-LAPACK/lapack/blob/master/LICENSE) | dynamic |
| LAPACK + LAPACKE, [Reference-LAPACK](https://github.com/Reference-LAPACK/lapack) (Linux; bundled inside OpenBLAS when vendored) | modified BSD | [LICENSE](https://github.com/Reference-LAPACK/lapack/blob/master/LICENSE) | dynamic |

**Note on vendored OpenBLAS** (opt-in via
`WITH_OPENBLAS=1 ./scripts/setup_native_deps.sh`, see
[installation](installation.md)): when built from source into
`third_party/openblas/install/`, the resulting `libopenblas.so`
bundles netlib LAPACK's Fortran sources + the LAPACKE C interface.
Both inherit OpenBLAS's BSD 3-Clause license (it is the
redistributed work). Net license posture is unchanged from a
system-installed OpenBLAS, fully compatible with MPL 2.0.

### Vendored browser assets (vq web console)

The `vq` web console ships one third-party browser asset. It is a static
file served to the viewer's browser, not linked into any binary, so it
has no bearing on the combined-work analysis above.

| Asset | Version | License | Source | Path |
|-------|---------|---------|--------|------|
| [htmx](https://htmx.org/) | 2.0.4 | **0BSD** (Zero-Clause BSD) | [LICENSE](https://github.com/bigskysoftware/htmx/blob/master/LICENSE) | [vibe-queue companion asset](https://github.com/vibe-qc/vibe-queue/blob/main/src/vq/web/static/htmx.min.js) (not bundled here) |

Vendored 2026-08-05 (it was previously fetched from `unpkg.com` at page
render time, which made the console silently non-functional on fleets
without egress and left an unpinned third-party script in the trust path
of a surface that can kill jobs). 0BSD grants redistribution outright
with no attribution condition. Per-file provenance, the upstream SHA-256,
and the re-fetch procedure are recorded beside the asset in
`htmx.min.js.PROVENANCE`.

**Compatibility summary**:

* MPL 2.0 (libxc, Eigen, vibe-qc itself), fully compatible.
* BSD 3-Clause (spglib, pybind11, OpenBLAS), fully compatible (permissive).
* **0BSD** (htmx, vendored browser asset), fully compatible -- the most
  permissive of the set; no attribution or notice obligation.
* Modified BSD (reference LAPACK / LAPACKE / netlib BLAS), fully
  compatible (permissive, with a no-endorsement clause).
* MIT (libecpint, libcerf, pugixml), fully compatible (permissive).
* **LGPL 3.0** (libint library), compatible via dynamic
  linking, which is the LGPL's intended use. Static linking
  would require either (a) supplying object files for relinking
  or (b) releasing the whole work under LGPL. **We dynamic-link;
  no static-linking case exists.**
* **GPL v2 or later** (FFTW3), see warning below.
* **Apple SDK license** (Accelerate framework, macOS-only), the
  framework ships with macOS, is not redistributed with vibe-qc,
  and is only linked when vibe-qc itself is built and run on
  macOS. No transitive license burden.

### FFTW: GPL v2 dependency, the binary distribution caveat

**This matters for you only if you want to redistribute vibe-qc
binaries (or a bundled application that embeds vibe-qc).** It
does **not** affect:

* Academic users running vibe-qc on their own data.
* Source-code distribution of vibe-qc itself (still MPL 2.0).
* Modifying vibe-qc and using your modified version internally.

But it **does** affect:

* Anyone shipping a precompiled binary of vibe-qc (including
  Conda / PyPI wheels that bake in `libfftw3.so`), the
  combined binary is effectively **GPL v2 or later** because
  FFTW's GPL terms govern the linked work.
* Anyone wanting to embed vibe-qc in a closed-source product.

The mechanism is standard GPL-vs-MPL combination: vibe-qc's MPL
2.0 source is GPL-compatible (no "Incompatible With Secondary
Licenses" notice), so the combined work *can* be distributed,
but recipients receive the combined work under **both** MPL
(for the vibe-qc parts) and GPL (for the whole combined binary,
because FFTW dominates).

**FFTW commercial license**: MIT's Technology Licensing Office
sells a non-GPL commercial FFTW license (see
[FFTW License page](https://www.fftw.org/doc/License-and-Copyright.html)).
This is an option for commercial redistribution, but the price
makes it impractical for hobby projects.

**Roadmap**: a future minor release will add a build-time option
to swap FFTW3 for [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft)
(BSD 3-Clause) or [KissFFT](https://github.com/mborgerding/kissfft)
(BSD 3-Clause) so binary distribution can be GPL-free for
commercial users. Tracked as roadmap entry "FFT backend
abstraction" in `docs/roadmap.md` (post-v1.0 milestone).

### Python package dependencies

vibe-qc's core Python dependencies (installed automatically with
`pip install vibe-qc`) are permissively licensed and
MPL-2.0-compatible:

| Package | License | Use in vibe-qc |
|---------|---------|----------------|
| [seekpath](https://github.com/giovannipizzi/seekpath) | BSD 3-Clause | canonical HPKOT band paths (Hinuma et al., *Comp. Mat. Sci.* **128**, 140, 2017) |
| [jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT | QVF manifest validation against the canonical JSON Schema |
| [joblib](https://joblib.readthedocs.io/) | BSD 3-Clause | NEB image parallelisation (loky process pool) |

Optional extras pull additional Python packages on demand (see
`pyproject.toml` `[project.optional-dependencies]`):

| Extra | Package | License | Use |
|-------|---------|---------|-----|
| `[dispersion]` | [dftd3](https://github.com/dftd3/dftd3) | LGPL 3.0+ | optional reference D3 implementation and coverage beyond the native H-Ar range |
| `[fetch]` | [optimade](https://github.com/Materials-Consortia/optimade-python-tools) | MIT | federated OPTIMADE client for `vqfetch` |
| `[fetch]` | [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | MIT | HTML parsing for NIST CCCBDB / WebBook / ATcT reference-data scrapers |
| `[fetch]` | [lxml](https://lxml.de/) | BSD 3-Clause | XML / HTML tree builder for reference-data scrapers |
| `[ml]` | [scikit-learn](https://scikit-learn.org/) | BSD 3-Clause | deserialises the bundled k-point spacing predictor (K8-F RandomForest model); see *ML k-point predictor* below |
| `[naming]` | [RDKit](https://www.rdkit.org/) | BSD 3-Clause | local IUPAC naming in `vibeqc_naming`; soft-imported, falls back to the CACTUS REST route when absent |
| `[skala]` | [PyTorch](https://pytorch.org/) | BSD 3-Clause | optional CPU inference and automatic differentiation for the Microsoft SKALA-1.1 XC checkpoint |
| `[trexio]` | [trexio](https://github.com/TREX-CoE/trexio) | BSD 3-Clause | TREXIO wavefunction files (`run_job(trexio=True)`, `write_trexio` / `read_trexio`); the PyPI wheel bundles its own HDF5 library; imported lazily, never vendored or redistributed |

### vibe-view's Python dependencies

[vibe-view](https://github.com/vibe-qc/vibe-view) is now a separate
MPL-2.0 project. Its own `pyproject.toml` declares its dependencies. The
following inventory records the viewer integration context, not libraries
shipped by the core repository. Consult the companion release for its current
dependency and license inventory.



All are permissive and MPL-2.0-compatible. Licenses below are as declared by
the distributions themselves:

| Scope | Package | License | Use in vibe-view |
|-------|---------|---------|------------------|
| core | [PyVista](https://pyvista.org/) | MIT | mesh construction, marching-cubes isosurfaces, offscreen PNG capture |
| core | [VTK](https://vtk.org/) (via PyVista) | BSD 3-Clause | the underlying rendering and compute-filter engine |
| core | [plotly](https://plotly.com/python/) | MIT | interactive band / DOS / spectra charts |
| core | [matplotlib](https://matplotlib.org/) | PSF-based (BSD-compatible) | static chart rendering for headless capture |
| core | [click](https://click.palletsprojects.com/) | BSD 3-Clause | the `vibe-view` CLI |
| core | [pydantic](https://docs.pydantic.dev/) | MIT | QVF manifest models |
| core | [jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT | manifest validation against the canonical schema |
| core | [NumPy](https://numpy.org/) | BSD 3-Clause (with 0BSD / MIT / Zlib / CC0 components) | array backbone, and the whole terminal-mode rasterizer |
| `[viewer]` | [trame](https://kitware.github.io/trame/) | Apache 2.0 | the browser viewer's server + widget layer |
| `[viewer]` | [trame-vtk](https://github.com/Kitware/trame-vtk) | BSD | VTK bridge for the trame view |
| `[viewer]` | [trame-vuetify](https://github.com/Kitware/trame-vuetify) | MIT | Vuetify widget set for the trame UI |
| `[viewer]` | [uvicorn](https://www.uvicorn.org/) | BSD 3-Clause | ASGI server behind trame |
| `[tui]` | [Textual](https://textual.textualize.io/) | MIT | the interactive terminal viewer (`vibe-view tui`) |
| `[ase]` | [ASE](https://wiki.fysik.dtu.dk/ase/) | LGPL 2.1+ | reads the long tail of structure formats (VASP POSCAR/CONTCAR, .extxyz, .traj); used as a library, never as a QC engine |
| `[smiles]` | [RDKit](https://www.rdkit.org/) | BSD 3-Clause | parses a SMILES string and embeds it in 3D (ETKDG) for *Build from SMILES*; the geometry is then refined by vibe-qc |
| `[jupyter]` | [IPython](https://ipython.org/) | BSD 3-Clause | the `%vibeview` cell magic |
| `[jupyter]` | [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) | BSD 3-Clause | inline viewer widgets in a notebook |

Nothing here is vendored or redistributed: every one is resolved by pip at
install time. Terminal mode's rasterizer is vibe-view's own code and needs
only NumPy, so `vibe-view show` carries no dependency beyond the core set;
Textual is required by the interactive app alone.

## 3. Bundled data

vibe-qc ships several kinds of data alongside the source:

* **Basis sets** in `python/vibeqc/basis_library/`
  (`basis/` from libint upstream + `custom/` for vibe-qc's
  pob-* additions).
* **ECP libraries** in `third_party/libecpint/install/share/libecpint/xml/`
  (Stuttgart-Köln MDF + LANL families).
* **Reference data fetched at runtime** by `vqfetch` (no
  bundled data; pulled on demand from external sources).
* **MACE foundation-model weights** fetched at runtime by the
  optional `[mace]` extra (no bundled data; pulled on demand into
  the XDG cache). Covered in *MACE* below.
* **Microsoft SKALA-1.1 model weights** fetched at runtime by the optional
  `[skala]` extra (no bundled data; pulled on demand into the XDG cache).
  Covered in *Microsoft SKALA-1.1* below.
* **PM6 / PM7 / OMx semiempirical parameters** in
  `python/vibeqc/semiempirical/methods/`: every shipped PM6 and PM7
  value (the inline H/C/N/O/F sets as well as the full
  `pm6_mopac_params.toml` / `pm7_mopac_params.toml` caches) is copied
  from MOPAC's Apache-2.0 parameter sources and carries that license;
  OM1/OM2/OM3 values are transcribed from the published Dral/Thiel
  tables. Covered below.
* **GFN2-xTB parameters** fetched at runtime from the Grimme group's
  `xtb` (LGPL-3.0-or-later; no bundled data, pulled on demand into the
  XDG cache). Covered in *GFN2-xTB parameters* below.
* **MSINDO method parameters** embedded in the INDO engine
  (`cpp/.../methods/indo/` + `python/vibeqc/semiempirical/methods/msindo.py`,
  H-F): semiempirical parameters redistributed by permission, see
  *MSINDO* below.
* **gCP geometric-counterpoise parameters** in
  `python/vibeqc/data_library/gcp/`: per-basis-set correction
  parameters from Kruse & Grimme 2012, transcribed from the
  published paper and verified against the mctc-gcp Fortran
  reference. Covered in *gCP parameter tables* below.
* **ML k-point predictor model** in
  `python/vibeqc/data_library/k8f_predictor.pkl`: a bundled
  scikit-learn RandomForestRegressor that recommends Monkhorst-Pack
  KSPACING for periodic calculations, based on the
  Choudhary-Tavazza 2020 convergence predictor. Gated by
  `VIBEQC_ML_KPOINTS=1`. Covered in *ML k-point predictor* below.
* **Python fetch extras** (optimade, beautifulsoup4, lxml):
  optional PyPI packages pulled by `pip install 'vibe-qc[fetch]'`
  for `vqfetch` structures and reference-data scraping. Listed in
  the *Python package dependencies* table above.

Each is covered separately.

### PM6 / PM7 / OMx, NDDO semiempirical parameters

vibe-qc includes an in-house implementation of the PM6, PM7 and
OM1/OM2/OM3 NDDO-family executable paths. These methods are **not yet
production-parity validated** against MOPAC or the original OMx
reference implementations, but their user-facing routes are available
for development, benchmarking, and guarded pre-screening workflows.

The *implementation* is vibe-qc's own; the *parameter values* are not.
Every PM6 and PM7 number vibe-qc ships is copied from the open-source
MOPAC parameter sources `parameters_for_PM6_C.F90` /
`parameters_for_PM7_C.F90`, which are **Apache-2.0**. This holds for the
inline five-element sets as well as for the bundled full caches: all 118
inline PM6 values (88 element + 30 diatomic) and all 95 inline PM7 values
(65 element + 30 diatomic) reproduce those files exactly at their six
published decimals, and none of the PM6 values appears in Stewart 2007.
The inline PM7 pair table that shipped between `d333f5adc` (2026-08-12)
and 2026-09-06 carried 28 `alpb`/`xfac` values of unrecorded origin that
matched neither MOPAC nor the bundled cache; the maintainer ruled on #440
that they do not ship, and they were replaced by MOPAC's values (the
inline and cached PM7 tables are now pinned against each other in
`tests/test_nddo_parameter_identity.py`). The Stewart papers are cited as
the **method** references (what the numbers parameterize); MOPAC is cited
as the **source of the values**.

| Item | Terms |
|---|---|
| vibe-qc PM6 / PM7 / OMx implementation | **MPL 2.0** |
| inline PM6 H/C/N/O/F values in `pm6_params.py` | Apache-2.0; copied from MOPAC `parameters_for_PM6_C.F90`, attribution in the file header |
| bundled `pm6_mopac_params.toml` cache | Apache-2.0 MOPAC source provenance recorded in the file header |
| inline PM7 H/C/N/O/F values in `pm7_params.py` | Apache-2.0; copied from MOPAC `parameters_for_PM7_C.F90`, attribution in the file header |
| bundled `pm7_mopac_params.toml` cache | Apache-2.0 MOPAC source provenance recorded in the file header |
| OM1 values in `omx_params.py` | Transcribed from Dral et al. 2016, Table 1 (originally Kolb & Thiel 1993, Table 2); published parameter table, not MOPAC-derived; cite Dral et al. 2016 + Kolb & Thiel 1993 |
| OM2 values in `omx_params.py` | Transcribed from Dral et al. 2016, Table 2 (DOI `10.1021/acs.jctc.5b01046`), the first full publication of the OM2 parameter values; published parameter table, not MOPAC-derived; cite Dral et al. 2016 (values) + Weber & Thiel 2000 (method, DOI `10.1007/s002149900083`) |
| OM3 values in `omx_params.py` | Transcribed from Dral et al. 2016, Table 3 (originally Scholten 2003); published parameter table, not MOPAC-derived; cite Dral et al. 2016 + Scholten 2003 |

**Apache-2.0 attribution.** The MOPAC parameter sources carry
`Copyright 2021 Virginia Polytechnic Institute and State University`,
licensed under the Apache License, Version 2.0
(<https://github.com/openmopac/mopac>). Apache-2.0 § 4(c)/(d) requires
that notice to travel with the redistributed material; it is retained in
the header of every file that carries the values
(`pm6_params.py`, `pm7_params.py`, `pm6_mopac_params.toml`,
`pm7_mopac_params.toml`). The runtime `ParameterSetMetadata` for these
sets reports `license = "Apache-2.0"` accordingly, under the identities
`published:pm6-mopac-{inline,full}-v1` and
`published:pm7-mopac-{inline,full}-v1` (the PM7 identities describe
parameter provenance only; the PM7 Hamiltonian itself stays gated).

**OMx provenance.** The OM1/OM2/OM3 values are *not* MOPAC-derived and
carry no Apache-2.0 obligation. Each set is transcribed from the parameter
tables of Dral et al., *J. Chem. Theory Comput.* **12**, 1082 (2016), DOI
`10.1021/acs.jctc.5b01046`: OM1 from Table 1 (originally Kolb & Thiel,
*J. Comput. Chem.* **14**, 775 (1993), DOI `10.1002/jcc.540140704`), OM2
from Table 2, and OM3 from Table 3 (originally Scholten, PhD thesis,
Heinrich-Heine-Universität Düsseldorf, 2003). Dral 2016 Table 2 is the
first full publication of the OM2 values: the OM2 method paper, Weber &
Thiel, *Theor. Chem. Acc.* **103**, 495 (2000), DOI
`10.1007/s002149900083`, is cited for the formalism, and its own Table 2
is an orthogonalization-energy table for H3-, not a parameter table
(maintainer wording 2026-09-06, #272). In the citation database the
entry each set's values were transcribed from carries
`role = "parameter-source"` (Dral 2016) and the formalism papers carry
`role = "method"`, and a test pins that split. The runtime
`ParameterSetMetadata` for these sets reports
`license = "published parameter table"` with the Dral 2016 DOI and
inherits nothing from the PM6 sets (maintainer ruling 2026-08-28, #272:
cite the papers, never an inherited license string). The original method
paper of each set is auto-cited alongside Dral 2016 through
`routes.methods.om1` / `om2` / `om3`.

**Cite** (auto-surfaced in `.out` / `.bibtex` via
`routes.methods.pm6` / `pm7` / `om1` / `om2` / `om3`): Stewart, *J. Mol.
Model.* **13**, 1173 (2007), DOI
`10.1007/s00894-007-0233-4`; Stewart, *J. Mol. Model.* **19**, 1 (2013),
DOI `10.1007/s00894-012-1667-x`; Moussa & Stewart, *J. Open Source
Softw.* **11**, 8025 (2026), DOI `10.21105/joss.08025`; Dral et al.,
*J. Chem. Theory Comput.* **12**, 1082 (2016), DOI
`10.1021/acs.jctc.5b01046`; Kolb & Thiel, *J. Comput. Chem.* **14**, 775
(1993), DOI `10.1002/jcc.540140704`; Weber & Thiel, *Theor. Chem. Acc.*
**103**, 495 (2000), DOI `10.1007/s002149900083`; Scholten, PhD thesis,
Heinrich-Heine-Universität Düsseldorf (2003).

### GFN2-xTB parameters, fetched on demand, never bundled

`method="gfn2_xtb"` runs vibe-qc's own GFN2-xTB engine, but the 86-element
parameter set it needs is **not** vibe-qc's and is **not** shipped. The
values come from the Grimme group's `xtb`
([grimme-lab/xtb](https://github.com/grimme-lab/xtb),
`param_gfn2-xtb.txt`), which is **LGPL-3.0-or-later**.

| Item | Terms |
|---|---|
| vibe-qc GFN2-xTB engine (`cpp/.../methods/xtb/`, `gfn2.py`) | **MPL 2.0** |
| `param_gfn2-xtb.txt` parameter values | **LGPL-3.0-or-later** (upstream `xtb`); fetched on demand, never bundled |
| D4 reference data used by the GFN2 dispersion term | see *3c. Dispersion-correction source* below |

This is the `vqfetch`-style on-demand pattern CLAUDE.md § 1 prescribes when
redistribution terms are restrictive: nothing is vendored, the file is
pulled from upstream on first use into the XDG cache
(`$VIBEQC_GFN2_CACHE_DIR`, else `$XDG_CACHE_HOME/vibeqc/`, else
`~/.cache/vibeqc/`), and the run record carries its identity. A GFN2
`.system` manifest records `gfn2_cache_sha256` and
`gfn2_d4_refdata_sha256`, so a reported energy is tied to the exact
parameters that produced it, and a substituted or truncated cache fails
closed rather than producing silently different numbers.

Consequence a user has to plan for: a host with **no outbound network and
no seeded cache cannot run GFN2-xTB**. Seed the cache from a networked
host, per "Where the parameter cache lives, and seeding it for offline
hosts" in [the user guide](user_guide/semiempirical.md).
Bundling the parameter file to remove that constraint would be a
redistribution decision under LGPL-3.0 and is deliberately not taken here
(issue #127).

**Cite** (auto-surfaced via `routes.methods.gfn2_xtb`): Bannwarth,
Ehlert & Grimme, *J. Chem. Theory Comput.* **15**, 1652 (2019), DOI
`10.1021/acs.jctc.8b01176`.

### MSINDO, INDO semiempirical method (`method="msindo"`)

vibe-qc includes an **independent re-implementation** of the MSINDO
semiempirical INDO method (Bredow, Geudtner & Jug; © Mulliken Center
for Theoretical Chemistry, University of Bonn). The engine
(`python/vibeqc/semiempirical/methods/msindo.py` and the C++ core under
`cpp/include/vibeqc/semiempirical/methods/indo/`) was written from the
published method and **copies no MSINDO source code**; it is validated
out-of-process against a reference MSINDO build
(`examples/regression/msindo/`, CLAUDE.md § 10).

The bundled **MSINDO parameters** (orbital exponents, frozen-core,
resonance, ionization-potential and anti-penetration parameters for the
full H-Xe set, Z=1..54) are redistributed **by permission of the
copyright holder**. They ship as the published `datas.f` parameter table
in `python/vibeqc/semiempirical/methods/msindo_params.json` (regenerate
with `examples/regression/msindo/gen_msindo_params.py`).

MSINDO's optional **NDDO mode** is a separate parametrization (the
program's `nddoparam.f`, applied for H, Li-F, Na-Cl). Those overrides
ship under the same permission as the published `nddoparam.f` table in
`python/vibeqc/semiempirical/methods/msindo_params_nddo.json` (regenerate
with `examples/regression/msindo/gen_msindo_params_nddo.py`).

| Item | Terms |
|---|---|
| vibe-qc's MSINDO re-implementation | **MPL 2.0** (as the rest of vibe-qc) |
| MSINDO method + bundled parameters | Used by permission of the Mulliken Center for Theoretical Chemistry, University of Bonn (Prof. T. Bredow), under an agreement with vibe-qc's maintainer |

**Cite** (auto-surfaced in `.out` / `.bibtex` via `routes.methods.msindo`):
Ahlswede & Jug, *J. Comput. Chem.* **20**, 563 (1999)
[DOI](https://doi.org/10.1002/(SICI)1096-987X(19990430)20:6%3C563::AID-JCC1%3E3.0.CO;2-2);
Ahlswede & Jug, *J. Comput. Chem.* **20**, 572 (1999)
[DOI](https://doi.org/10.1002/(SICI)1096-987X(19990430)20:6%3C572::AID-JCC2%3E3.0.CO;2-1).
NDDO calculations additionally auto-surface Voigt, *Theor. Chim. Acta* **31**,
289 (1973) [DOI](https://doi.org/10.1007/BF00527556), and Dewar & Thiel,
*Theor. Chim. Acta* **46**, 89 (1977)
[DOI](https://doi.org/10.1007/BF00548085).

### Microsoft SKALA-1.1, optional neural XC functional (`[skala]` extra)

vibe-qc provides an MPL-2.0 adapter for the Microsoft SKALA-1.1
exchange-correlation model. It does not bundle or import Microsoft's
PySCF-based runtime. It implements the pinned checkpoint protocol directly;
the atom-grid chunk-planning contract and interface conventions adapt the
MIT-licensed Microsoft source. The Microsoft source-repository notice is
retained as
`LICENSES/MICROSOFT-SKALA-MIT` in source and wheel distributions, as well as
beside the downloaded checkpoint as conservative attribution. The optional
extra declares PyTorch, under BSD 3-Clause, as its only direct dependency;
PyTorch is not redistributed by vibe-qc. These terms do not conflict with
vibe-qc's MPL 2.0 source license.

For installation, model-cache handling, supported SCF routes, and
reproducibility guidance, see the [Microsoft SKALA-1.1 user guide](user_guide/skala.md).

The adapter's exact level-3 atomic-grid profile adapts the period tables,
Treutler radii, NWChem pruning rule, and Becke partition adjustment from
PySCF 2.14.0. Those modified C++ portions carry a source notice and remain
subject to PySCF's Apache License 2.0 terms. Every source and wheel
distribution includes the unmodified upstream license and NOTICE as
`LICENSES/PYSCF-2.14.0-LICENSE` and
`LICENSES/PYSCF-2.14.0-NOTICE`. Apache-2.0 is compatible with the MPL-2.0
licensing of vibe-qc's own modifications.

The redistribution audit checked these immutable upstream artifacts:

| Item | Immutable source | Terms | vibe-qc treatment |
|------|------------------|-------|-------------------|
| Microsoft SKALA source | [commit `4f3f072b`](https://github.com/microsoft/skala/tree/4f3f072b820183d4c77b47bcea6df9736655d12a) | MIT, Copyright (c) Microsoft Corporation | no runtime implementation imported; checkpoint interface and atom-grid chunk planning adapted with the MIT notice retained |
| Official SKALA-1.1 CPU checkpoint | [`microsoft/skala-1.1` revision `99b5ed87`](https://huggingface.co/microsoft/skala-1.1/tree/99b5ed87e5f69d9216e1f9e30148b922eaea1241) | MIT model-card metadata; shared by Microsoft Research AI for Science | fetched on demand, never bundled; the source-repository Microsoft MIT notice is retained conservatively beside it |
| PySCF 2.14 exact-grid source | [tag `v2.14.0`](https://github.com/pyscf/pyscf/tree/v2.14.0/pyscf/dft) | Apache-2.0, Copyright 2014-2024 The PySCF Developers | selected algorithms and tables adapted with modification notice, license, and NOTICE retained |

The checkpoint is `skala-1.1-rev1.fun` with SHA-256
`7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd`.
vibe-qc verifies that digest before every TorchScript deserialization and
writes immutable provenance plus the conservatively retained Microsoft
source-repository notice as `MICROSOFT-SKALA-SOURCE-MIT.txt` beside the cached
file. Provenance records the model-card `MIT` label and the source notice in
separate, explicitly scoped fields. That copied source notice does not
establish that redistributing the cached TorchScript artifact satisfies every
applicable term. Anyone redistributing it must independently audit the artifact
and preserve all applicable notices and terms. The normal vibe-qc source and
binary distributions contain neither the model weights nor Microsoft's
upstream SKALA package, runtime, or verbatim source tree; vibe-qc's independent
adapter retains the notice for the protocol and chunk-planning concepts it
adapted.

**Training data are not shipped.** Inference needs only the checkpoint.
vibe-qc does not fetch, copy, or redistribute the SKALA training corpus, so
licenses on individual training subsets do not enter the vibe-qc
distribution.

**Audit conclusion.** The Microsoft source is MIT licensed, and the official
checkpoint's model-card metadata labels it MIT. The audit found no copyright
or data-license blocker to shipping this adapter and its hash-pinned on-demand
fetch path. It does not clear redistribution or bundling of the checkpoint;
that would require an authoritative Microsoft checkpoint license/notice
package and an audit of third-party code or IR serialized in the artifact.
The SKALA name is used only to identify the compatible Microsoft model;
vibe-qc uses no Microsoft logo and does not imply endorsement.

**Patent scope.** MIT grants broad copyright permissions but, unlike MPL 2.0,
does not contain an express patent-license clause. No separate `PATENTS` file
or specific patent assertion was found in the reviewed upstream material.
This repository review is not a formal patent clearance or a legal opinion.
A distributor that requires a commercial freedom-to-operate opinion should
obtain advice from counsel or written assurance from Microsoft.

**Citations** (auto-surfaced for `functional="skala-1.1"` and its aliases):
Luise et al., *Accurate and scalable exchange-correlation with deep learning*,
arXiv:2506.14665 (2025), DOI `10.48550/arXiv.2506.14665`; and Pöschel et al.,
*Molecular Implementation of the Machine-Learned Skala Exchange-Correlation
Functional in CP2K through GauXC*, arXiv:2608.19033 (2026), DOI
`10.48550/arXiv.2608.19033`. Because vibe-qc's pinned SKALA parity profile
adapts the PySCF 2.14 level-3 construction and ordering, the route also cites
Sun et al.,
*Recent developments in the PySCF program package*, J. Chem. Phys. 153,
024109 (2020), DOI `10.1063/5.0006074`.

### MACE, optional ML interatomic potential (`[mace]` extra)

`method="mace"` interfaces [ACEsuit MACE](https://github.com/ACEsuit/mace),
a pre-trained machine-learning interatomic potential, via the
**optional** `[mace]` extra (`pip install 'vibe-qc[mace]'`). MACE is
**not** in the default install and is **not** linked into the vibe-qc
binary, none of the licenses below bind a normal vibe-qc distribution;
they apply only to a user who opts in. (Python ≤3.13 only; see the
roadmap MACE entry.)

**Code, the `[mace]` extra pulls these Python packages:**

| Package | License |
|---------|---------|
| [mace-torch](https://github.com/ACEsuit/mace) | MIT |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause |
| [e3nn](https://github.com/e3nn/e3nn) | MIT |

All permissive and MPL-2.0-compatible.

**Foundation-model weights, fetched on demand, never bundled.** The
MACE *code* is MIT, but the trained *weights* are licensed separately
and download on first use into the XDG cache (`~/.cache/mace/`, the
`vqfetch`-style on-demand pattern); vibe-qc ships none of them:

| Model family | License | Commercial use |
|--------------|---------|----------------|
| MACE-MP-0 / MPA-0 (materials) | **MIT** | ✅ yes |
| MACE-OFF23 (organic) | **ASL** | ❌ academic only |

The [Academic Software License](https://github.com/gabor1/ASL) (ASL)
is a GPLv2-derived **non-commercial** license. vibe-qc's default model
is the **MIT** MACE-MPA-0. ASL model families such as MACE-OFF23 are
reachable only through explicit model selection and are gated behind
`MLIPOptions(accept_academic_license=True)` or `VIBEQC_ACCEPT_ASL=1`,
so a commercial user cannot pull them inadvertently. Unknown or unregistered
MACE model keys are rejected before backend import or weight download.
Accepting the ASL does not authorize an unregistered model family, URL, or
local weight path. A model enters the supported public scope only after
registration with explicit provenance and licensing. **Citation**: a
`method="mace"` run auto-emits the
MACE method paper + the selected foundation-model paper to its `.bibtex` /
`.references`, cite them in published work.

### 3a. Bundled basis sets

```
python/vibeqc/basis_library/
├── basis/         ← assembled from libint 2.13.1 upstream + custom overlay
├── custom/        ← source of truth for vibe-qc's own additions
└── README.md
```

#### Standard `basis/` files inherited from libint 2.13.1

The standard `.g94` files assembled into `basis/` come from the
450+ basis sets that ship in the libint 2.13.1 upstream distribution
at `share/libint/2.13.1/basis/`. We inherit libint's distribution
decisions: if libint ships a basis set, vibe-qc redistributes it
under the same terms libint redistributes it under (LGPL 3.0 for the
library, basis sets themselves typically inherited from the
[Basis Set Exchange](https://www.basissetexchange.org)). The runtime
`basis/` directory also includes the custom overlay documented below.

**Citation requirement**. The `.g94` file headers preserve the
original publication references (e.g. ANO-RCC carries a 6-line
reference list naming Widmark, Roos, Malmqvist, Veryazov,
Lindh). Users **must cite the originating publication** for any
basis set used in published work, this is a community norm,
not a license requirement. The vibe-qc reflexive
[`vibeqc.print_banner()`](running.md) does not currently print
the basis-set citation; printing it is queued as a v0.x.x
roadmap item.

**Per-family citation hints** (extend with the `.g94` file's
own header for the authoritative reference):

| Family | Per-family citation |
|--------|---------------------|
| sto-Ng | Hehre, Stewart, Pople, *J. Chem. Phys.* **51**, 2657 (1969) |
| 6-31g family | Hehre, Ditchfield, Pople, *J. Chem. Phys.* **56**, 2257 (1972); Hariharan, Pople, *Theor. Chim. Acta* **28**, 213 (1973) |
| 6-311g family | Krishnan, Binkley, Seeger, Pople, *J. Chem. Phys.* **72**, 650 (1980) |
| cc-pVxZ family | Dunning, *J. Chem. Phys.* **90**, 1007 (1989); Kendall, Dunning, Harrison, *J. Chem. Phys.* **96**, 6796 (1992) |
| def2 family | Weigend, Ahlrichs, *Phys. Chem. Chem. Phys.* **7**, 3297 (2005); Weigend, *Phys. Chem. Chem. Phys.* **8**, 1057 (2006) |
| ANO-RCC family | Multiple, see the per-element references inside [`basis/ano-rcc.g94`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/basis_library/basis/ano-rcc.g94) |
| RIFIT / JKFIT auxiliaries | Weigend, *Phys. Chem. Chem. Phys.* **8**, 1057 (2006); Hellweg et al. *Theor. Chem. Acc.* **117**, 587 (2007); Eichkorn et al. *Theor. Chem. Acc.* **97**, 119 (1997) |

#### `custom/`, vibe-qc's own additions

The POB custom basis sets in `custom/` are
**originated** by Mike Peintinger (vibe-qc's author / current
maintainer) and Tilo Bredow's group at the University of Bonn,
either as PhD work or as collaborator-level contributions. **No
external license clearance is required.**

| File | Citation |
|------|----------|
| `pob-tzvp.g94` | M. F. Peintinger, D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.* **34**, 451 (2013). [DOI:10.1002/jcc.23153](https://doi.org/10.1002/jcc.23153); Rb-I: J. Laun, D. Vilela Oliveira, T. Bredow, *J. Comput. Chem.* **39**, 1285 (2018). [DOI:10.1002/jcc.25195](https://doi.org/10.1002/jcc.25195) |
| `pob-dzvp-rev2.g94` | D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow, *J. Comput. Chem.* **40**, 2364 (2019). [DOI:10.1002/jcc.26013](https://doi.org/10.1002/jcc.26013) |
| `pob-tzvp-rev2.g94` | J. Laun, T. Bredow, *J. Comput. Chem.* **42**, 1064 (2021). [DOI:10.1002/jcc.26521](https://doi.org/10.1002/jcc.26521) |

Source URL (Bredow group archive):
<https://www.chemie.uni-bonn.de/bredow/de/software/pob-tzvp-tar.gz>.
The bundled `.g94` files reproduce this archive's content with
minor formatting normalisation.

The 46 heavy pob-TZVP-rev2 records (Rb-I, Cs-Po, La-Lu; CRYSTAL
`Z+200` format with an inline ECP block) are bundled under
`basis_library/sources/pob-TZVP-rev2/` from the group's
`pob-tzvp-ref2-rb-i`, `pob-tzvp-rev2-cs-po` and `pob-tzv-rev2-la-lu`
archives (retrieved 2026-09-05), and `basis/pob-tzvp-rev2.ecp` is the
same ECP data emitted as a Gaussian-style sidecar. The potentials are
the Stuttgart-Cologne energy-consistent pseudopotentials the pob authors
paired with their valence sets; they are published numerical parameter
data, cited per element through the pob-TZVP-rev2 paper and the
Stuttgart references in `python/vibeqc/output/citations/database.toml`.

The 16 pob-TZVP records for Rb-I (Laun, Vilela Oliveira and Bredow, 2018;
same `Z+200` format) are bundled under `basis_library/sources/pob-TZVP/`
from the group's `pob-tzvp-rb-i` archive (`pob-TZVP-Rb-I.tar.gz`, retrieved
2026-09-13), with `basis/pob-tzvp.ecp` as their sidecar. Their potentials
are the same Stuttgart-Cologne ECPs the pob-TZVP-rev2 records carry, block
for block. They are bundled with the Bredow group's agreement, like the
rest of the pob family. They were deliberately not taken from the Basis
Set Exchange, whose pob-TZVP copy carries the sulfur d-polarisation
column-swap defect (#228).

#### The def2 family beyond Kr

libint's def2 files stop at Kr. `custom/def2-{sv(p),svp,svpd,tzvp,tzvpd,
tzvpp,tzvppd,qzvp,qzvpd,qzvpp,qzvppd}.g94` are libint's H-Kr files, byte
for byte, followed by the Rb-Rn valence blocks from the Basis Set Exchange
(Turbomole 7.3 data; the diffuse `-d` sets carry no lanthanides), and
`custom/def2-*.ecp` are the matching def2-ECP blocks from the same BSE
records (`scripts/basisset_dev/merge_def2_heavy_blocks.py`, retrieved
2026-09-05). Same CC-BY-4.0 terms and attribution obligations as the other
BSE-fetched files; the per-element def2-ECP references are in each
sidecar's header and in the citation database.

#### Single-element completions of libint's files

`custom/cc-pvqz.g94`, `custom/cc-pv6z.g94` and `custom/sto-3g.g94` (added
2026-09-10) are libint's own files with their element data unchanged, plus one
element block each taken from the Basis Set Exchange 0.12 catalogue by
`scripts/basisset_dev/merge_bse_element_blocks.py`. libint omits these
elements because each was added to its set by a later, separate publication:

| Basis | Element | Originating publication |
|---|---|---|
| cc-pVQZ | Ca | Koput, Peterson, *J. Phys. Chem. A* **106**, 9595 (2002) |
| cc-pV6Z | Be | Prascher, Woon, Peterson, Dunning, Wilson, *Theor. Chem. Acc.* **128**, 69 (2011) |
| STO-3G | Xe | Pietro, Blurock, Hout, Hehre, DeFrees, Stewart, *Inorg. Chem.* **20**, 3650 (1981) |

Each routes its own citation in `database.toml` rather than inheriting the
parent set's, since the parent papers do not cover these elements. Same terms
as the other BSE-derived files.

`custom/pob-dzvp-rev2.g94` gained Si and Cr-Br the same way on 2026-09-10.
Those elements are part of the same publication that already covers the
bundled ones (Vilela Oliveira, Laun, Peintinger, Bredow, *J. Comput. Chem.*
**40**, 2364 (2019)), so the existing citation route is unchanged; our file
was simply a partial import of one paper. BSE's copy was checked against the
Bredow-group-archive-derived file first and agrees on all 19 shared elements.

**`pob-tzvp` is deliberately not completed from BSE.** Its 16 missing Rb-I
blocks (Laun, Vilela Oliveira, Bredow, *J. Comput. Chem.* **39**, 1285 (2018))
exist in the catalogue, but BSE's copy of pob-TZVP is an upstream
distribution carrying the sulfur d-polarisation column-swap defect vibe-qc
fixed by regenerating from the Bredow archive: our sulfur is 5s4p1d, BSE's is
5s4p with no d function at all. Those blocks should come from the archive.

#### The Peterson/Figgen PP correlation-consistent family

`custom/{,aug-}cc-p{V,wCV}{D,T,Q,5}Z-PP.g94` (16 files, added 2026-09-08) are
the Basis Set Exchange 0.12 records fetched by
`scripts/basisset_dev/fetch_from_bse.py`, each covering the same 39 elements
(Cu-Kr, Y-Xe, Hf-Rn). BSE ships the Stuttgart-Köln MCDHF ECP blocks inside the
orbital file; `scripts/setup_basis_library.sh` splits each into
`basis/<name>.g94` plus a `basis/<name>.ecp` sidecar. Same terms and
attribution obligations as the other BSE-fetched files below.

The matching `-PP-RIFIT` auxiliary sets already shipped; these are the orbital
sets they fit. Both the orbital blocks and the ECP blocks carry their own
publications, and per-element they are different papers, so every one of the
16 names routes the full union in
`python/vibeqc/output/citations/database.toml`:

| Elements | Orbital set | Stuttgart-Köln ECP |
|---|---|---|
| Cu, Zn, Ag, Cd, Au, Hg | Peterson, Puzzarini, *Theor. Chem. Acc.* **114**, 283 (2005) | Figgen, Rauhut, Dolg, Stoll, *Chem. Phys.* **311**, 227 (2005) |
| Ga-As, In-Sb, Pb, Bi | Peterson, *J. Chem. Phys.* **119**, 11099 (2003) | Metz, Stoll, Dolg, *J. Chem. Phys.* **113**, 2563 (2000) |
| Tl | Peterson, *J. Chem. Phys.* **119**, 11099 (2003) | Metz, Schweizer, Stoll, Dolg, Liu, *Theor. Chem. Acc.* **104**, 22 (2000) |
| Se-Kr, Te, Xe, Po-Rn | Peterson, Figgen, Goll, Stoll, Dolg, *J. Chem. Phys.* **119**, 11113 (2003) | same |
| I | as above, with Peterson, Shepler, Figgen, Stoll, *J. Phys. Chem. A* **110**, 13877 (2006) | same |
| Y-Pd | Peterson, Figgen, Dolg, Stoll, *J. Chem. Phys.* **126**, 124101 (2007) | same |
| Hf-Pt | Figgen, Peterson, Dolg, Stoll, *J. Chem. Phys.* **130**, 164108 (2009) | same |

The `cc-pwCV*Z-PP` sets add Peterson, Yousaf, *J. Chem. Phys.* **133**, 174116
(2010) for the post-d core-valence blocks. The `aug-` sets add no publication
of their own: BSE attributes each diffuse block to the same paper as its
parent set.

#### ECP sidecars for the 3c composite bases

| Sidecar | Content | Source | Citation |
|------|---------|--------|----------|
| `vdzp.ecp` | vDZP's custom-core potentials (82 elements) | BSE 0.12 vDZP record | Müller, Hansen, Grimme, *J. Chem. Phys.* **158**, 014103 (2023) |
| `def2-msvp.ecp` | the def2-ECP, Rb-Rn | Psi4 `def2-msvp.gbs` (BSE-derived) | Andrae 1990; Metz 2000; Peterson 2003; Leininger 1996; Kaupp 1991; Dolg 1989 (per element in the file header) |
| `def2-mtzvp.ecp`, `def2-mtzvpp.ecp` | the same def2-ECP data | copied from `def2-msvp.ecp`; the def2-m* bases pair with the def2-ECP beyond Kr like their parents | same |

These are published numerical parameter data in the same category as the
BSE-fetched basis sets above; the per-element originating references are
kept in each file header.

#### BSE-fetched and BSE-constructed basis sets

This section covers two tranches, 148 files plus one. The sixteen
Peterson/Figgen PP sets have their own heading above and are BSE-fetched on
the same terms; the def2 beyond-Kr blocks likewise. The heading used to carry
a "(149 .g94 files)" total, which counted only these two tranches and so drifted
as later BSE-derived files landed. `custom/` holds 176 `.g94` files today, of
which three (the pob family) are vibe-qc originated; the rest are BSE-fetched
or BSE-constructed by one route or another, so no single number in this
section is the whole inventory. `python/vibeqc/basis_library/` is the
authority.

The `basissetdev` branch carried 148 basis-set ``.g94`` files in
`python/vibeqc/basis_library/custom/` fetched from the
[Basis Set Exchange](https://www.basissetexchange.org) (BSE) on
2026-05-08, now shipping in v0.9.0. The v0.15.x release-paper
work adds one further BSE-derived Pople basis,
`6-311+g3df2p.g94`, constructed from exact BSE component bases
because BSE 0.12 does not publish the literal `6-311+G(3df,2p)`
name.

##### Legal basis for redistribution

Basis-set parameters, lists of Gaussian exponents and contraction
coefficients, are **numerical data**, not creative expression.  Under
U.S. copyright law, facts and data are not copyrightable
(*Feist Publications, Inc. v. Rural Telephone Service Co.*, 499 U.S.
340 (1991)).  The BSE project itself states this explicitly in its
documentation: the basis-set parameters are scientific data extracted
from the peer-reviewed literature and are treated as public-domain
facts; the BSE's BSD-3-Clause license covers the **software** (the
website, API, and curation infrastructure), not the numerical data.

This is the established norm in computational chemistry: every
major quantum-chemistry code (PySCF, Psi4, NWChem, ORCA, Gaussian,
Q-Chem, Molpro, GAMESS) redistributes basis-set parameters obtained
from the BSE or directly from the originating publications without
separate licensing agreements.

##### Attribution, our obligations

Although the data are not copyright-encumbered, **scientific
attribution is mandatory**.  Every ``.g94`` file shipped by vibe-qc
carries a header crediting the BSE source and (where available) the
originating publication.  Users are required to cite the originating
publication for any basis set they use in published work; the
canonical references are surfaced through the citation database at
`python/vibeqc/output/citations/database.toml` (route ``basis_sets``).

##### Publisher policies for supporting information

All 148 files originate from peer-reviewed publications whose
supporting information (SI) includes the basis-set parameters:

| Publisher | SI redistribution policy | Representative journals |
|-----------|--------------------------|------------------------|
| American Chemical Society (ACS) | SI is supplementary material accompanying the article; redistribution of factual data within SI is standard practice | J. Chem. Theory Comput., J. Phys. Chem., J. Am. Chem. Soc. |
| American Institute of Physics (AIP) | SI is part of the published record; AIP's author rights permit reuse of factual data | J. Chem. Phys. |
| Wiley | SI accompanies the article; redistribution of extracted data is standard in the field | J. Comput. Chem., Int. J. Quantum Chem., Angew. Chem. |
| Royal Society of Chemistry (RSC) | SI is supplementary material; RSC's author re-use rights cover factual data | Phys. Chem. Chem. Phys. |
| Springer / Czech Academy | SI data is part of the published record | Theor. Chem. Acc., Collect. Czech. Chem. Commun. |

##### Per-family inventory

| Family | Files | First author (year) | Journal | Publisher |
|--------|-------|---------------------|---------|-----------|
| Jensen pcseg-{0..4}, aug-pcseg-{0..4} | 10 | Jensen (2014) | JCTC | ACS |
| Jensen pc-{0..4}, aug-pc-{0..4} | 10 | Jensen (2001-2002) | J. Chem. Phys. | AIP |
| Pople diffuse (6-31+G**, 6-311+G**, etc.) | 5 | Ditchfield/Hehre/Pople (1971); Frisch/Pople/Binkley (1984) | J. Chem. Phys. | AIP |
| Pople RI-J auxiliary | 2 | Weigend (2006) | PCCP | RSC |
| Karlsruhe def2 3c carriers | 2 | Brandenburg et al. (2018) | J. Chem. Phys. | AIP |
| Karlsruhe dhf-* (Dirac-Hartree-Fock) | 6 | Weigend/Baldes (2010) | J. Chem. Phys. | AIP |
| Karlsruhe x2c-* (exact two-component) | 3 | Pollak/Weigend (2017) | JCTC | ACS |
| Grimme vDZP (omegaB97X-3c carrier) | 1 | Muller/Hansen/Grimme (2023) | J. Chem. Phys. | AIP |
| LANL ECP family | 6 | Hay/Wadt (1985) | J. Chem. Phys. | AIP |
| Dunning cc-pV(n+d)Z + aug- | 5 | Dunning/Peterson/Wilson (2001) | J. Chem. Phys. | AIP |
| Dunning jul-/jun-cc-pV(n+d)Z | 4 | Papajak/Truhlar (2010) | JCTC | ACS |
| Dunning cc-pCV*Z + aug- | 6 | Woon/Dunning (1995) | J. Chem. Phys. | AIP |
| ANO-RCC contracted | 5 | Roos et al. (2004) | J. Phys. Chem. A | ACS |
| ANO-R (Zobel-Widmark-Veryazov) | 5 | Zobel et al. (2020) | JCTC | ACS |
| Sadlej pVTZ / Sadlej+ | 2 | Sadlej (1988) | Collect. Czech. Chem. Commun. | Czech Acad. |
| Jensen pcS-{0..3}, aug-pcS-{1,2} | 6 | Jensen (2008) | JCTC | ACS |
| Jensen pcSseg-{0..2} | 3 | Jensen (2015) | JCTC | ACS |
| Jensen pcJ-{0..3} | 4 | Jensen (2006) | JCTC | ACS |
| Sapporo-DKH3-{DZP,TZP,QZP} | 3 | Noro/Sekiya/Koga (2012) | Theor. Chem. Acc. | Springer |
| Cologne DKH2 | 1 | Dolg et al. | - | - |
| SARC-DKH2 / SARC2-QZ* | 3 | Pantazis/Neese (2009, 2019) | JCTC | ACS |

**All 142 BSE-fetched files (plus 3 pob-* vibe-qc files, separate): numerical data from peer-reviewed publications.
Redistribution of basis-set parameters is the established norm
in computational chemistry; no file is encumbered by a license
that prohibits redistribution as part of a QC code.**#### SAP atomic-potential helpers

Two additional `.g94` files in `basis/` are **not orbital basis
sets**, they are tabulated Gaussian expansions of the radial
atomic effective potentials used by the
**Superposition of Atomic Potentials (SAP)** initial guess
(`InitialGuess.SAP`, see `docs/roadmap.md` §G2c). The `.g94`
container is a convenience: the same loader can read both,
but consuming these files as an orbital basis would produce
nonsense by design, the test suite guards against that via
`SAP_PREFIXES = ("sap_",)` in
[`tests/basisset_dev/test_basis_library_load.py`](https://github.com/vibe-qc/vibe-qc/blob/main/tests/basisset_dev/test_basis_library_load.py).

| File | Reference radial potential | Use |
|------|---------------------------|-----|
| `sap_helfem_large.g94` | All-electron, Helfem fully-numerical atomic SCF | non-relativistic default for `InitialGuess.SAP` |
| `sap_grasp_large.g94`  | Relativistic Dirac-Hartree-Fock, GRASP | optional default for x2c / DKH calculations |

**Citation**:

> S. Lehtola, L. Visscher, E. Engel, *Efficient Implementation
> of the Superposition of Atomic Potentials Initial Guess for
> Electronic Structure Calculations in Gaussian Basis Sets*,
> *J. Chem. Phys.* **152**, 144105 (2020).
> [DOI:10.1063/5.0004046](https://doi.org/10.1063/5.0004046)

The reference is preserved verbatim in each `.g94`'s file
header. Published, redistributable under standard scientific-
data conventions; users **must cite the originating publication**
when reporting energies obtained with the SAP initial guess.

### 3b. Bundled ECP libraries

```
third_party/libecpint/install/share/libecpint/xml/
├── ecp10mdf.xml      ← Stuttgart-Köln MDF, 10-electron core
├── ecp28mdf.xml      ← Stuttgart-Köln MDF, 28-electron core
├── ecp46mdf.xml      ← Stuttgart-Köln MDF, 46-electron core
├── ecp60mdf.xml      ← Stuttgart-Köln MDF, 60-electron core
├── ecp78mdf.xml      ← Stuttgart-Köln MDF, 78-electron core
└── lanl2dz.xml       ← Los Alamos LANL2DZ
```

These XML files ship **inside the libecpint distribution**
(MIT-licensed, by Robert A. Shaw). vibe-qc inherits them by
linking libecpint and pointing at libecpint's data directory at
runtime. No additional clearance or redistribution decision is
made by vibe-qc, we ship what libecpint ships. They are consulted
only for an explicit `ecp_centers` + `ecp_library` request; a bundled
basis' own `.ecp` sidecar (§ 3a) is what the SCF wrappers attach.

**Citation requirement** (community norm, not legal):

| ECP family | Citation |
|-----------|----------|
| Stuttgart-Köln MDF (`ecp10mdf` … `ecp78mdf`) | Originally: Andrae, Häußermann, Dolg, Stoll, Preuß, *Theor. Chim. Acta* **77**, 123 (1990) and family. Per-element references in libecpint's source repository. |
| LANL2DZ (`lanl2dz`) | Hay, Wadt, *J. Chem. Phys.* **82**, 270 (1985); 299 (1985); 284 (1985). |
| libecpint software | R. A. Shaw, J. G. Hill, *J. Chem. Phys.* **147**, 074108 (2017); Shaw, *J. Chem. Phys.* **159**, 014103 (2023). |

### 3c. Dispersion-correction source, fully native, MPL 2.0

vibe-qc ships its own implementation of the EEQ atomic-charge model
(`cpp/include/vibeqc/eeq_charges.hpp` + `cpp/src/eeq_charges{,_data}.cpp`),
written from the published equations in

* Caldeweyher, Ehlert, Hansen, Neugebauer, Spicher, Bannwarth, Grimme,
  *J. Chem. Phys.* **150**, 154122 (2019), supporting information.

Per-element EEQ parameter values (χ, η, κχ, γ) are scientific data
transcribed from the above paper's Table S2. Covalent radii are from
Pyykkö & Atsumi, *Chem. Eur. J.* **15**, 188 (2009). Numerical
values from published scientific data are not subject to expressive
copyright; vibe-qc cites the original papers in the relevant headers
(`cpp/src/eeq_charges_data.cpp`) and in any user-facing output that
quotes EEQ charges.

The Grimme-group reference Fortran implementations
([`multicharge`](https://github.com/grimme-lab/multicharge),
Apache-2.0; [`mctc-lib`](https://github.com/grimme-lab/mctc-lib),
LGPL-3.0-or-later) were consulted during development for cross-
validation; no source from either was copied into the tree, and
all files in vibe-qc carry the project's MPL-2.0 SPDX header.

The native D3(BJ) implementation includes the H-Ar corner of the
coordination-number-dependent C6 grid (1,443 active references), the D3
covalent radii, and all 156 D3(BJ) damping-parameter sets. These numerical
scientific constants were extracted from simple-dftd3 revision
`cc9c1f634ea06ba8c283f174aa43faa08b3b98ab`
(LGPL-3.0-or-later). The pinned-source SHA-256 checks and provenance are in
`scripts/extract_d3_c6_reference.py` and
`scripts/extract_d3bj_parameters.py`; the generated files copy no Fortran
implementation code. The underlying C6 model is Grimme, Antony, Ehrlich &
Krieg, *J. Chem. Phys.* **132**, 154104 (2010), and the BJ damping fits are
Grimme, Ehrlich & Goerigk, *J. Comput. Chem.* **32**, 1456 (2011), plus the
fit-specific references retained in the pinned upstream TOML.

The optional `dftd3` PyPI package (LGPL-3.0-or-later; see the
`[dispersion]` extras in `pyproject.toml`) remains available as the
independent reference implementation and supplies coverage beyond the native
H-Ar range.

The optional `dftd4` PyPI package (LGPL-3.0-or-later; see the
`[dispersion]` extras in `pyproject.toml`) is the current path to
full D4 dispersion energies via `vibeqc.compute_d4` /
`run_b2plyp(dispersion="d4")` / `run_dsd_pbep86(dispersion="d4")`.

Both optional `dftd3` and `dftd4` backends are Python-side runtime
dependencies only;
the LGPL Fortran libraries live in their respective wheels, never
in the vibe-qc tree. When a native D4 dispersion-energy backend
lands (replacing the optional Python dep), it will be a clean-room
reimplementation from the same paper, MPL 2.0 throughout, matching
the EEQ implementation shipped today.

### 3d. gCP parameter tables, Kruse-Grimme 2012

The geometric counterpoise correction (gCP) is a semiempirical
basis-set superposition error (BSSE) estimator. vibe-qc ships
per-basis-set gCP parameter tables in
`python/vibeqc/data_library/gcp/`, transcribed from the published
method and verified against the reference Fortran implementation
([mctc-gcp](https://github.com/grimme-lab/gcp) by S. Grimme's
group).

| Basis set | File | Status |
|-----------|------|--------|
| def2-SVP | `def2-svp.toml` | complete (H-Kr, Z=1..36) |
| def2-TZVP | `def2-tzvp.toml` | complete (H-Kr, Z=1..36) |
| def2-mSVP | `def2-msvp.toml` | complete |
| def2-mTZVP | `def2-mtzvp.toml` | complete |
| def2-mTZVPP | `def2-mtzvpp.toml` | complete |
| MINIS | `minis.toml` | complete |
| MINIX | `minix.toml` | complete |
| vDZP | `vdzp.toml` | complete |

Each `.toml` file carries the originating publication, DOI, and
redistribution terms in its `[metadata]` block. The gCP parameters
are numerical results from a published scientific paper; their
redistribution as part of a quantum-chemistry code is standard
practice.

**Cite** (auto-surfaced in `.out` / `.bibtex` via
the gCP loader): Kruse & Grimme, *J. Chem. Phys.* **136**, 154101
(2012), DOI `10.1063/1.3700154`.

<span id="3c-external-data-fetched-by-vqfetch-no-bundled-data"></span>

(external-data-licensing)=
### 3e. External data fetched by `vqfetch` (no bundled data)

`vqfetch` (the v0.8.0 external-data integration) does **not
bundle** any external data. It pulls structures and reference
data **on demand** from public databases, caches results
locally with full provenance metadata, and surfaces the
per-record license terms back to the caller.

| Source | Default license per record | Attribution requirement |
|--------|---------------------------|-------------------------|
| [COD](http://www.crystallography.net/cod/) (Crystallography Open Database) | CC0 / public domain | None legally; CIF authorship is preserved in provenance |
| [Materials Project](https://next-gen.materialsproject.org/) | CC-BY 4.0 | Cite Materials Project per their terms |
| [NOMAD](https://nomad-lab.eu/) | CC-BY 4.0 (data); CC0 (metadata) | Cite the contributing author + NOMAD as source |
| [OPTIMADE](https://www.optimade.org/) federation | per-provider | varies, vqfetch records the provider in provenance |
| [NIST CCCBDB](https://cccbdb.nist.gov/) | US Government work, public domain in US | Cite NIST Standard Reference Database 101 (DOI 10.18434/T47C7Z) |

The provenance contract: **every `vqfetch`-pulled record carries
the source DB, ID, URL, original DOI (where available), license
string, and fetched-at timestamp**. This makes per-record
attribution straightforward in publications and avoids any
"where did this come from" auditability gap.

vqfetch does **not** ship pre-fetched data. The local cache
(`~/.cache/vqfetch/` per XDG) is populated only by user
queries. If a user republishes work containing
vqfetch-pulled records, the per-record license is displayed in
the SCF log and recorded in the per-run `.system` manifest.

### 3f. Semiempirical (DFTB) parameters, fully native, MPL 2.0

vibe-qc ships its own implementation of the **DFTB0 / SCC-DFTB**
semiempirical methods
([`python/vibeqc/semiempirical/`](https://github.com/vibe-qc/vibe-qc/tree/main/python/vibeqc/semiempirical),
[`cpp/src/semiempirical/`](https://github.com/vibe-qc/vibe-qc/tree/main/cpp/src/semiempirical)),
with parameter tables constructed **from scratch** in
`cpp/src/semiempirical/parameters.cpp`:

* **STO exponents** from Slater's rules.
* **On-site energies** from periodic-trend scaling.
* **Hubbard U** from approximate per-element estimates.
* **Repulsive-pair form** R⁻¹² with an original
  `default_repulsive_A` estimator.

The parameter set covers **H, U (87 elements)**. **Zero values
are sourced from dftb.org parameter sets** (`mio`, `3ob`,
`matsci`, etc.), no redistribution concern. This was confirmed
by the semiempirical chat in the v0.9.0 cycle (commit
`2d2c24eb`).

**Honest scope:** these parameters are **approximate,
order-of-magnitude**, *not* production-grade DFTB. Intended use
is fast screening and geometry preoptimisation before HF/DFT
refinement. The runtime banner and the `.system` manifest both
record that vibe-qc is using its in-house parameter set rather
than a published reference. See
[`docs/semiempirical_stage1_design.md`](semiempirical_stage1_design.md)
and [`docs/semiempirical_roadmap.md`](semiempirical_roadmap.md)
for the full provenance + scope writeup.

**Method-paper citations** (preserved per `.bibtex` sibling via
the route `methods` / `dftb` in
[`database.toml`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/output/citations/database.toml)):

* DFTB0, Porezag, Frauenheim, Köhler, Seifert, Kaschner,
  *Phys. Rev. B* **51**, 12947 (1995).
* SCC-DFTB, Elstner *et al.*, *Phys. Rev. B* **58**, 7260 (1998).
* Slater-rule STO ζ scaffolding, Hehre, Stewart, Pople, *J. Chem.
  Phys.* **51**, 2657 (1969) (STO-NG context).

The parameter tables themselves are vibe-qc's own work and ship
under MPL-2.0 alongside the rest of the in-tree implementation.

### 3g. ML k-point predictor (K8-F), Choudhary-Tavazza 2020

vibe-qc ships a bundled scikit-learn RandomForestRegressor at
`python/vibeqc/data_library/k8f_predictor.pkl`. The model
recommends Monkhorst-Pack KSPACING for periodic calculations via
`KPoints.recommend(predictor="ml")`, gated by the environment
variable `VIBEQC_ML_KPOINTS=1`.

The model is a vibe-qc original trained on a synthetic dataset
built from physical heuristics (cell volumes, band-gap character,
dimensionality), consistent with the scaling behaviour observed in
the Choudhary & Tavazza convergence predictor. The bundled `.pkl`
file is **MPL 2.0** (as the rest of vibe-qc).

**Optional Python dependency.** Deserialising the model requires
`scikit-learn` (BSD 3-Clause), which is installed on demand via
`pip install 'vibe-qc[ml]'`. The package is imported lazily at
first prediction call; without it, `predictor="ml"` raises a clear
`ImportError` pointing to the `[ml]` extra. `scikit-learn` is also
pulled by the `[test]` and `[dev]` extras so the bundled predictor
regression tests run in CI.

**Cite** (auto-surfaced via `routes.methods.ml_kpredictor`):
Choudhary, K. & Tavazza, F., *Convergence and machine learning
predictions of Monkhorst-Pack k-points and plane-wave cut-off in
high-throughput DFT calculations*, *Comput. Mater. Sci.* **161**,
300-308 (2019), DOI `10.1016/j.commatsci.2019.02.006`.

### 3h. QVF format and reference toolkit: separate Apache-2.0 project

The [QVF repository](https://github.com/vibe-qc/qvf) owns the
specification, schema, registry, conformance corpus, and reference
implementations. It is Apache-2.0 and independent of the core repository's
MPL-2.0 code. See its [LICENSE](https://github.com/vibe-qc/qvf/blob/main/LICENSE)
for the governing terms and its own distribution inventory.

vibe-qc implements the published format independently. Neither its Python
package nor its C++ core imports or links the QVF reference toolkit. Tests
compare independent implementations through the published corpus; this does
not create a runtime dependency.

The six QVF reference pages in these docs are pinned, vendored documentation
with upstream tag/commit provenance. Changes belong in QVF and must be
re-vendored deliberately. Legacy pre-split QVF writer and viewer packages
are preserved outside the product repositories in a private historical
archive; current distributions belong to their respective companion projects.

## Website artwork typography

The website artwork generator uses an unmodified Manrope variable font,
Copyright 2018 The Manrope Project Authors, under the SIL Open Font License
1.1. The font, license and pinned upstream provenance are kept under
`website/artwork/fonts/`. Generated wordmarks and social cards contain outlined
lettering; the font is not loaded by the website or chemistry runtime.
See the [font license](https://github.com/google/fonts/blob/fb629caaa15ad25c051089c98f09cf6c8e30a86b/ofl/manrope/OFL.txt).

## Citing vibe-qc itself

A vibe-qc-specific citation will be added once the JCC release
paper is published. Until then, please cite:

* Repository: <https://github.com/vibe-qc/vibe-qc>
* Version + DOI: per the
  [`CITATION.cff`](https://github.com/vibe-qc/vibe-qc/blob/main/CITATION.cff)
  file in the repository root.

Plus any **basis-set citation** + **ECP citation** + **functional
citation** for the methods you actually used. The
[citing guide](citing.md) collects the standard set.

## Reporting a licensing concern

If you spot a licensing gap, an attribution that's not being
surfaced correctly, a basis set we shouldn't be bundling, or an
incompatibility we missed, please email
[mpei@vibe-qc.com](mailto:mpei@vibe-qc.com). We take this
seriously, vibe-qc is an open-source project run by one
person, and getting any of this wrong has real consequences for
the project's ability to be used.
