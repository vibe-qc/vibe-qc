---
myst:
  html_meta:
    "description": "Effective core potentials (ECPs) in vibe-qc, libecpint 1.0.7 integration, Stuttgart-Köln MDF + LANL2DZ libraries, manual ECPCenter recipe + auto-helper."
    "og:title": "vibe-qc, effective core potentials (libecpint)"
    "og:description": "ECP integrals via libecpint 1.0.7. Six ECP libraries shipped (ecp10mdf, ecp28mdf, ecp46mdf, ecp60mdf, ecp78mdf, lanl2dz). Pt UHF/LANL2DZ = −118.227 Ha."
---

# Effective core potentials (ECPs)

vibe-qc ships
[libecpint 1.0.7](https://github.com/robashaw/libecpint)
vendored as a runtime dependency. ECP integrals, reduced-Z
nuclear-attraction, and valence-electron-count accounting are
wired through every molecular SCF driver (RHF, UHF, RKS, UKS).
Heavy-element chemistry, including the d-block, the
lanthanides, and the actinides, is reachable without
constructing the all-electron basis.

The banner lists libecpint as a linked dependency:

```
linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7
```

## What ships

Six ECP libraries bundled inside libecpint, MIT-licensed (see
[`docs/license.md`](../license.md#3b-bundled-ecp-libraries)):

| Library | Family | Elements in the bundled XML | Citation family |
|---------|--------|------------------------------|-----------------|
| `ecp10mdf` | Stuttgart-Köln MDF, 10-electron core | K-Kr | Andrae, Häußermann, Dolg, Stoll, Preuß, *Theor. Chim. Acta* 77, 123 (1990) and follow-ups |
| `ecp28mdf` | Stuttgart-Köln MDF, 28-electron core | Rb-Xe | (ditto, per-element refs) |
| `ecp46mdf` | Stuttgart-Köln MDF, 46-electron core | Cs, Ba | (ditto) |
| `ecp60mdf` | Stuttgart-Köln MDF, 60-electron core | Hf-Rn, Ac-U | (ditto) |
| `ecp78mdf` | Stuttgart-Köln MDF, 78-electron core | Fr, Ra | (ditto) |
| `lanl2dz` | Hay-Wadt LANL | Na-La, Hf-Bi, U-Pu | Hay, Wadt, *J. Chem. Phys.* 82, 270 + 299 + 284 (1985) |

These are the elements actually present in each bundled file, and they are
narrower than the family names imply: `ecp46mdf` carries two elements, not
every 46-core one. **No bundled XML covers Ce-Lu**; the lanthanides are
reached through a basis' own `.ecp` sidecar (pob-TZVP-rev2 carries La-Lu),
never through `ecp_library`. The same numbers appear again under
"The bundled XML libraries, for explicit pairing only" below.

The accompanying valence-only orbital basis sets (the `lanl2dz` family,
the dhf-* sets, vDZP, the def2-m* composite bases, pob-TZVP-rev2 and the
16-member cc-pVnZ-PP family) are bundled together with their own `.ecp`
sidecars; see the next section.

## Automatic attachment, decided per element

Every bundled basis that replaces core electrons ships its ECP as a
`.ecp` sidecar next to its `.g94` file: the def2 family from Rb to Rn
(def2-SV(P) through def2-QZVPPD, with the def2-ECP), the LANL and dhf
families, vDZP, the 3c composite bases def2-mSVP / def2-mTZVP /
def2-mTZVPP, pob-TZVP-rev2, and the Peterson/Figgen PP sets
(`{,aug-}cc-p{V,wCV}{D,T,Q,5}Z-PP`, Cu-Kr / Y-Xe / Hf-Rn, carrying the
Stuttgart-Köln MCDHF potentials with 10-, 28- and 60-electron cores). For molecular `run_job` calls and the direct molecular SCF
wrappers, the sidecar's own primitives are attached **inline** for every
atom that has a block there, and nothing is attached for an atom that has
none. Thus `run_job(mol, basis="lanl2dz", method="rhf")` runs the
valence-electron ECP Hamiltonian on sulfur, while water in LANL2DZ is the
all-electron D95V calculation it always was. A basis that is valence-only
somewhere but ships no ECP there (the lanthanides in the diffuse def2-*D
sets) is refused with a message naming the elements rather than run
all-electron in a valence basis.  The cc-pVnZ-PP sets were in that list
until their orbital files shipped (2026-09-08); they now resolve from
their own sidecars.

The decision is `vibeqc.basis_registry.ecp_requirements(basis, atomic_numbers)`:
the sidecar first, then the family rules in
`python/vibeqc/basis_library/registry.toml`, and nothing else. Basis
*names* no longer decide anything, so x2c-TZVPall (all-electron on every
element) runs, and pob-TZVP-rev2 on silver attaches its ECP although its
name carries no marker.

Two explicit recipes remain when you want to override the selection:

1. **Inline primitives** (`ecp_primitive_blocks`, `ecp_primitive_centers`,
   `ecp_effective_charges`, `ecp_total_ncore`) built from any sidecar with
   `vibeqc.ecp_metadata.inline_ecp_data_for(mol, basis_name)`, or from a
   CRYSTAL record with `vibeqc.basis_crystal.crystal_ecp_to_libecpint_arrays`.
   This is what the automatic attachment sets.
2. **Manual `ECPCenter` recipe** (`ecp_centers` + `ecp_library`): one
   `ECPCenter` per heavy atom and the name of a bundled libecpint XML
   library. Use it to pair an all-electron orbital basis with a Stuttgart
   or LANL potential deliberately, as the Zn²⁺/6-31G example below does.

`vq.auto_ecp_centers(...)` still exists as the legacy XML-library helper
behind recipe 2. It pairs a library by core count, so it cannot represent
vDZP's custom cores, refuses a molecule spanning two core sizes, and for
vDZP's silver picks a different potential from the one the sidecar
carries (85 mHa on AgH). New scripts should rely on the automatic
attachment instead.

## Manual recipe (always available)

```python
import vibeqc as vq

# Pt atom at the origin.
mol = vq.Molecule([vq.Atom(78, [0.0, 0.0, 0.0])])
basis = vq.BasisSet(mol, "lanl2dz")    # valence-only orbital basis

# One ECPCenter per heavy atom.
ec = vq.ECPCenter()
ec.Z = 78                              # atomic number of the centre
ec.xyz = [0.0, 0.0, 0.0]               # bohr (Cartesian)

# Multiplicity lives on the Molecule; rebuild it with the right
# spin state if you're not already starting from one.
mol = vq.Molecule([vq.Atom(78, [0.0, 0.0, 0.0])], multiplicity=3)
basis = vq.BasisSet(mol, "lanl2dz")

# Wire the ECP onto the SCF options.
opts = vq.UHFOptions()
opts.ecp_centers = [ec]
opts.ecp_library = "lanl2dz"           # which ECP XML library

result = vq.run_uhf(mol, basis, opts)
print(result.energy)                   # -118.227 Ha (22 BFs, 125 SCF iters)
```

The same `ecp_centers` + `ecp_library` flags are accepted by
`run_rhf`, `run_rks`, and `run_uks`. Multiple heavy atoms →
build one `ECPCenter` per atom and pass them all in:

```python
# Pt-Pt dimer, 10-bohr separation. Singlet (paired) lives on
# the Molecule, not on the SCF options.
mol = vq.Molecule(
    [
        vq.Atom(78, [0.0, 0.0, -5.0]),
        vq.Atom(78, [0.0, 0.0, +5.0]),
    ],
    multiplicity=1,
)
basis = vq.BasisSet(mol, "lanl2dz")

opts = vq.UHFOptions()
opts.ecp_centers = [
    vq.ECPCenter(Z=78, xyz=[0.0, 0.0, -5.0]),
    vq.ECPCenter(Z=78, xyz=[0.0, 0.0, +5.0]),
]
opts.ecp_library = "lanl2dz"
```

## Electron count and charge convention

`Molecule.charge` is always the **physical ionic charge**, and
`Molecule.n_electrons()` is always the **full physical electron
count**, including the core electrons the ECP will replace. The
SCF subtracts the replaced cores itself: the per-element core
sizes come from the ECP library (`ECPHcore.total_ncore` on the
C++ side), and only the remaining valence electrons fill
orbitals in the valence-only basis.

Worked example, [ZnH]+ with `ecp10mdf`:

```python
mol = vq.Molecule(
    [vq.Atom(30, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 3.0])],
    charge=1,          # the physical ionic charge, nothing more
    multiplicity=1,
)
# Z_total = 31, charge +1  ->  n_electrons() = 30 (physical)
# ecp10mdf replaces 10 core electrons on Zn (3s 3p 3d 4s stay
# in valence)                ->  20 valence electrons
# closed shell              ->  10 doubly occupied orbitals
```

Things that follow from the convention:

* **Never fold the core count into `charge`.** An input that
  passes `charge = n_core + ionic_charge` (a convention some
  codes and some pre-v0.12 vibe-qc scripts used) now describes
  a different, over-stripped ion: the SCF would remove the core
  a second time. The drivers raise if the cores outnumber the
  electrons, with a reminder that `n_electrons()` must be the
  full physical count.
* **Parity and multiplicity arithmetic work on either count.**
  Every shipped core (`ecpNNmdf`, `lanl2dz`) replaces an even
  number of electrons, so the physical and valence counts have
  the same parity; `run_rhf`'s even-electron check and the
  UHF/UKS `n_alpha`/`n_beta` split behave as expected.
* **Gradients require matching ECP provenance.** When `GradientOptions`
  mirrors the SCF options' XML-library or inline-primitive ECP input (see
  § Gradients below), the gradient drivers rebuild the energy-weighted
  density from exactly the orbitals the SCF occupied. Before the current
  wrapper guards, leaving these fields unset caused the bare-Z bug flagged
  by the 2026-05-18 audit. Public gradient wrappers now require a verified
  high-level SCF result and refuse a missing, different, or mixed ECP route
  before native derivative work.
* `vq.ecp_effective_charges(mol, ecp_centers, library_name)`
  returns the per-atom `Z_eff = Z - n_core` vector if you need
  the accounting explicitly (atoms without an ECP keep bare Z).
* **Electrostatic properties use the same charges.** The valence
  density integrates to the valence count, so the nuclear term of
  the dipole moment, `sum_A Z_A (R_A - O)`, and the reference charge
  in the Mulliken, Löwdin and Hirshfeld populations, `q_A = Z_A -
  pop_A`, must use `Z_eff` on ECP atoms (Dolg and Cao, Chem. Rev.
  112, 403 (2012), Sec. 5: the cores are point charges `Q = Z -
  n_core` in every term of the valence-only Hamiltonian). Until
  GitLab #642 (2026-09-03) these used bare `Z`: H2S/LANL2DZ RHF
  reported 0.148 D with the wrong sign, the dipole moved by exactly
  `-n_core = -10` au per bohr of origin shift, and the Mulliken
  charges summed to +10 instead of 0; the corrected dipole is
  1.99496 D and origin-free, and the FD Hessian's dipole
  derivatives (hence IR intensities) satisfy the translational sum
  rule `sum_A d mu / d R_A = 0`. `run_job`, the FD Hessian, the ASE
  calculator and the population writers pass the charges
  automatically from the options the SCF ran with. A direct caller
  of `vq.properties.dipole_moment` / `mulliken_charges` /
  `loewdin_charges` / `hirshfeld_charges` on an ECP system passes
  `nuclear_charges=vibeqc.ecp_metadata.effective_nuclear_charges(mol,
  opts)`; the default is bare `Z`, correct only all-electron.
  Hirshfeld partitions with all-electron free-atom weights even on
  ECP systems, so its per-atom split there is approximate although
  the charges now sum to the molecular charge.

## Phase-14e auto-helper (`vq.auto_ecp_centers`)

The Phase 14e helper short-circuits the boilerplate above.
Given a molecule and a basis name, it parses the bundled
`.ecp` sidecar, picks the matching libecpint XML library, and
returns ready-to-drop `(ecp_centers, library_name)`:

```python
import vibeqc as vq

mol = vq.Molecule(
    [
        vq.Atom(78, [0.0, 0.0, -5.0]),
        vq.Atom(78, [0.0, 0.0, +5.0]),
    ],
    multiplicity=1,                    # singlet (paired)
)
basis = vq.BasisSet(mol, "lanl2dz")

opts = vq.UHFOptions()
opts.ecp_centers, opts.ecp_library = vq.auto_ecp_centers(mol, "lanl2dz")

result = vq.run_uhf(mol, basis, opts)
# Same -118.227 Ha as the manual recipe, but no per-atom ECPCenter
# wiring needed. Tutorial / sample-script-friendly.
```

The helper is a **legacy** surface since 2026-09: tutorials and example
scripts need no ECP wiring at all, because the SCF wrappers attach the
sidecar themselves, and the helper cannot represent vDZP's custom cores
or a molecule spanning two core sizes. Where it applies, its numerics
match the manual recipe: **Pt/lanl2dz UHF = −118.227 Ha** (singlet) at
22 BFs in 125 iters; **Si/lanl2dz RHF = −3.475 Ha**.

```{important}
**Status (shipped on main).** `vq.auto_ecp_centers` is exported
from `vibeqc.__init__` and works out of the box on a `pip install`
of vibe-qc. The basissetdev paper-writing branch still tracks
basis-set-side work (the BSE-fetched basis sets, ongoing
re-optimisations) but the ECP auto-helper itself has landed in
`main` since v0.8.0. The manual recipe below remains useful when
you want to override the Stuttgart-MDF default mapping per
element row.
```

## When you need it

An ECP is **required**, and attached automatically, on every atom whose
block in the chosen basis is valence-only:

* `lanl2dz`, `lanl2tz`, `lanl08*`: Na and heavier.
* `dhf-{svp,sv(p),tzvp,tzvpp,qzvp,qzvpp}`: Rb and heavier (the Dirac-Fock
  small-core potentials).
* `vdzp`: its custom cores on B-F, Al-Ar, Ga-Kr, and Rb-Rn.
* `def2-sv(p)`, `def2-svp`, `def2-tzvp`, `def2-tzvpp`, `def2-qzvp`,
  `def2-qzvpp` and their diffuse `-d` variants: Rb-Rn (the diffuse sets
  skip the lanthanides), with the def2-ECP.
* `def2-msvp`, `def2-mtzvp`, `def2-mtzvpp`: Rb-Rn, with the def2-ECP.
* `pob-tzvp-rev2`: Rb-I, Cs-Po, La-Lu, with the Stuttgart-Cologne potentials
  of its CRYSTAL records.
* `{,aug-}cc-p{V,wCV}{D,T,Q,5}Z-PP`: Cu-Kr, Y-Xe, Hf-Rn, with the
  Stuttgart-Köln MCDHF potentials (10-, 28- and 60-electron cores). All 16
  sets carry the same potential; they differ only in the orbital blocks.

An ECP is **not needed**, and none is attached, when the block is
all-electron: every basis on H-Kr except vDZP, all of `x2c-*`, `ano-rcc*`,
`sarc*`, `sapporo-dkh3-*` and `cologne-dkh2` on every element, and the
LANL / dhf families on light atoms.

A request that needs an ECP vibe-qc does not ship is **refused**: the
diffuse def2-*D sets carry no lanthanide block. (The cc-pVnZ-PP orbital
sets were the other example here until they shipped on 2026-09-08.)

## The bundled XML libraries, for explicit pairing only

The six libecpint XML libraries are consulted only when a script sets
`ecp_centers` + `ecp_library` itself. Their element coverage is narrower
than the names suggest, and their numbers are rounded copies of the
published sets, which is why a bundled basis' own sidecar is preferred:

| Library | Elements in the bundled XML | Core |
|---------|-----------------------------|------|
| `ecp10mdf` | K-Kr | 10 |
| `ecp28mdf` | Rb-Xe | 28 |
| `ecp46mdf` | Cs, Ba | 46 |
| `ecp60mdf` | Hf-Rn, Ac-U | 60 |
| `ecp78mdf` | Fr, Ra | 78 |
| `lanl2dz` | Na-La, Hf-Bi, U-Pu | varies |

The def2-ECP that the def2 family pairs with is Wood-Boring for Y-Cd and
Hf-Hg and Dirac-Fock elsewhere; it ships as the `.ecp` sidecar of every
def2 orbital file (and of the def2-m* composite bases), not as an XML
library. AgH in def2-TZVP reproduces PySCF's def2-TZVP + def2-ECP to the
microhartree (`tests/test_ecp_correlated.py`).

## Python API surface

```python
import vibeqc as vq

# Version probe
vq.libecpint_version()              # "libecpint 1.0.7 (vendored, MAX_L=5)"
vq.library_versions()["libecpint"]  # same

# Per-atom ECP descriptor (manual path)
ec = vq.ECPCenter()
ec.Z = 78
ec.xyz = [0.0, 0.0, 0.0]

# AO matrix V_ECP_{μν} = ⟨χ_μ | V_ECP | χ_ν⟩  (spherical basis)
V_ecp = vq.compute_ecp_matrix(
    basis,                          # vibeqc.BasisSet
    ecp_centers=[ec],
    library_name="lanl2dz",
)                                   # → numpy.ndarray (n_bf, n_bf)

# Auto-helper (shipped on main; see § Phase-14e above)
ecp_centers, library_name = vq.auto_ecp_centers(mol, "lanl2dz")

# Drive any SCF with ECPs:
opts = vq.UHFOptions()
opts.ecp_centers = [ec]
opts.ecp_library = "lanl2dz"
result = vq.run_uhf(mol, basis, opts)
```

The same `ecp_centers` + `ecp_library` API works on every
molecular SCF driver: `run_rhf`, `run_uhf`, `run_rks`,
`run_uks`.

### Periodic SCF with ECPs

The native direct drivers `vq.run_rks_periodic` and
`vq.run_rhf_periodic_gamma` apply a periodic ECP: `PeriodicKSOptions` and
`PeriodicRHFOptions` carry `ecp_primitive_blocks`, `ecp_home_centers`,
`ecp_effective_charges` and `ecp_total_ncore`, H_core uses Z_eff nuclear
attraction + the lattice-summed V_ECP, the electron count is reduced to
the valence count, and the ionic repulsion uses Z_eff. Build the fields
from the bundled pob-TZVP-rev2 records with
`vibeqc.basis_crystal.build_periodic_ecp_data(system, atoms)` where
`atoms` comes from `vibeqc.periodic_runner._bundled_pob_source_atoms`;
the heavy records ship in `basis_library/sources/pob-TZVP-rev2/`, so this
needs no network. Ag fcc / PBE / pob-TZVP-rev2 on this route agrees with
PySCF.pbc to within 1 mHa per atom (`tests/test_periodic_ecp.py`).

`run_periodic_job` applies a periodic ECP on its default route: the
k-point GDF drivers (`jk_method="gdf"`, RHF / RKS / UHF / UKS, 3D cells)
build V_ne from a Z_eff nuclear frame, Bloch-sum the lattice-summed V_ECP
into every Hcore(k), use the Z_eff Ewald repulsion and fill the valence
count, so

```python
run_periodic_job(system, basis, method="RKS", functional="pbe",
                 kpoints=[4, 4, 4], smearing_temperature=0.005, output="ag_pbe")
```

on a pob-TZVP-rev2 Ag cell is a 19-valence-electron-per-atom calculation.
The `.system` manifest and the ionic energy record it: `result.e_nuclear`
is the Ewald energy of the Z_eff charges. `method="RHF"` / `"UHF"` take the
same route; until 2026-09 `PeriodicRHFOptions` had no ECP fields and every
periodic HF request on an ECP basis died on the runner's assignment
(#88). The UHF / UKS drivers split alpha and beta from the valence count
(AgCl in pob-TZVP-rev2: 18 / 18 electrons, not 32 / 32). With the default
mesh (`kpoints=None`) an ECP-bearing cell is routed to the same k-point
drivers at a one-point mesh; the `run_pbc_gdf_*` Gamma fast paths and the
legacy Gamma driver read no ECP field and refuse ECP options when called
directly. The runner resolves the ECP from the bundled pob-TZVP-rev2
records for that family and from the basis' `.ecp` sidecar for every other
basis, so `BasisSet(cell, "lanl2dz")` or def2-TZVP on a silver cell carry
their ECP on this route too (LANL2DZ ships no fitting set; pass
`aux_basis=` explicitly). Every other periodic route
**refuses** an ECP-bearing cell before any SCF work, naming the elements:
RIJCOSX, the Ewald open-shell drivers, ROHF / ROKS on GDF, slabs, BIPOLE,
and the external-XC GPW / GAPW / AICCM adapters. Until
2026-09 the runner set the ECP fields on the options and those drivers
silently ignored them (Ag fcc in pob-TZVP-rev2 was dispatched with 188
electrons into a 19-valence-electron basis); the refusal is what replaced
that.

## Gradients, forces, geometry optimization

Analytic nuclear gradients are ECP-aware (2026-05-18 audit
remediation). The gradient must differentiate the same
Hamiltonian the SCF solved, so `GradientOptions` mirrors the
SCF options' ECP fields:

```python
go = vq.GradientOptions()
go.ecp_centers = opts.ecp_centers   # same list as the SCF call
go.ecp_library = opts.ecp_library
grad = vq.compute_gradient(mol, basis, result, go)
```

When `ecp_centers` is set, the nuclear-repulsion and
nuclear-attraction derivative pieces switch to the effective
charges `Z_eff = Z - n_core` and a `dV_ECP/dR` term (libecpint
first derivatives) is added. Historically, an empty `ecp_centers` list let
an XML-ECP reference reach the bare-Z all-electron derivative, which was
wrong at the 0.1 Ha/bohr scale (regression-pinned in
`tests/test_ecp_gradient.py`). Current public wrappers compare the exact
`(Z, xyz)` centre multiset and normalized library recorded on the SCF result.
They also refuse results from the low-level arbitrary-Hcore SCF entry points,
whose ECP provenance is deliberately unknown.

### Inline-primitive ECPs

ECPs reach the SCF by **two** routes, and the gradient must
follow whichever one ran. Bases whose per-element cores have no
matching libecpint XML library -- vDZP, def2-mSVP, and
CRYSTAL/pob data -- are attached as *inline primitive blocks*
(see the auto-attach section above), which leaves `ecp_centers`
**empty**. Mirror those fields instead:

```python
go = vq.GradientOptions()
go.ecp_primitive_blocks = opts.ecp_primitive_blocks
go.ecp_primitive_centers = opts.ecp_primitive_centers
go.ecp_effective_charges = opts.ecp_effective_charges   # per ATOM
go.ecp_total_ncore = opts.ecp_total_ncore
grad = vq.compute_gradient(mol, basis, result, go)
```

The XML-library and inline-primitive routes are mutually exclusive; setting
both is an error. `ecp_effective_charges` has one entry **per atom** (bare `Z`
on atoms carrying no ECP), while `ecp_primitive_blocks` and
`ecp_primitive_centers` have one entry per ECP centre. Public gradient
wrappers compare each complete primitive block together with its paired
centre, the effective-charge vector in atom order, and `ecp_total_ncore`
against the verified SCF result.

Leaving these unset on an inline-ECP SCF used to trigger the same bare-Z
fallback described above: measured at 0.16 Ha/bohr on CH4/vDZP, ~1.9x the
true force (#574, pinned in `tests/test_ecp_gradient.py`). It now fails the
provenance check instead.

### What wires this for you

One helper builds the mirror, and every internal consumer goes
through it:

```python
from vibeqc.gradient_options import gradient_options_from_scf

go = gradient_options_from_scf(opts)   # JK backend + ECP fields, both routes
grad = vq.compute_gradient(mol, basis, result, go)
```

* The ASE calculator (`vibeqc.ase.VibeQC`) uses it for
  `atoms.get_forces()` and `optimize=True`.
* The FD Hessian (`compute_hessian_fd`) uses it whenever you pass
  no `gradient_options` (#576). That covers the molecular runner's
  `hessian=True` frequency / thermochemistry path and `run_irc`'s
  transition-state Hessian, which pass none. An explicit
  `gradient_options` is honoured for its JK-backend fields, but its
  ECP fields are re-synchronised to the SCF options at every
  displaced geometry.
* `compute_hessian_fd` also **moves the ECP centres with the
  displaced atom**. Both routes pin the potential to absolute
  coordinates (`ecp_centers[i].xyz`, `ecp_primitive_centers[i]`),
  and the SCF attaches a centre to a nucleus only within 1e-6 bohr
  (libecpint's own assignment tolerance is 1e-4 bohr). Displace the
  nucleus without the centre and the atom silently reverts to bare
  `Z` while `V_ECP` keeps acting: pre-#576, [ZnH]+/6-31G/ecp10mdf
  diverged to -7e10 Ha at the first 0.005 bohr Zn displacement, and
  the all-electron atoms' Hessian columns were the bare-Z ones.
  ECP options whose centres sit on no atom of the molecule are now
  refused up front with a `ValueError` naming the centre.

Every other molecular driver that evaluates the SCF away from the
geometry the centres were built for follows the same rule (#643):

* `run_dimer` / `run_irc` path forces (the shared image evaluator)
  move the centres onto each visited geometry, run the SCF through
  the ECP-aware drivers instead of the all-electron warm-start seam,
  and mirror the ECP fields onto the gradient when you pass no
  `gradient_options`; your options come back describing the start
  geometry. A top-level `grid_options` override is refused on ECP
  RKS/UKS images (set the grid on the KS options instead).
* The ASE calculator (`vibeqc.ase.VibeQC`) moves the centres onto
  the current atoms at every `calculate()`. The centre-to-atom map
  is fixed at the first geometry where the centres sit on the atoms;
  afterwards the `*_options` you passed describe the **last**
  geometry evaluated, not the one you built them for. On ECP systems
  the calculator's `hessian` property uses the FD route, because the
  analytic kernels take no ECP options.
* The native optimisers (`optimize_molecule`, `optimize_molecule_brent`,
  the runner's `optimizer_backend="native"` / `"brent"`) follow the
  centres per step and restore your options on return; `run_job`
  attaches the ECP for the start geometry before any optimiser runs
  and moves the centres onto the optimised geometry before the final
  single point and the Hessian.
* Centres that sit on no atom of the start geometry are refused up
  front with a `ValueError` naming the centre, on every one of these
  paths.

`run_neb` still refuses molecular ECP paths (its warm-start image loop
is all-electron by construction).

Post-HF and multireference *derivative* routes (MP2 / CC optimisations,
CASSCF, CASPT2 and NEVPT2 gradients and Hessians) remain unsupported with
ECPs: their derivative boundaries cannot yet preserve and verify the exact
ECP operator, so they fail before SCF or trajectory work. Their energies
are a different matter; see the next section.

## Correlated, open-shell and determinant methods on an ECP reference

Every SCF result records how many core electrons its ECP removed
(`ecp_total_ncore`), and since 2026-09 every consumer partitions its
occupied space from that valence count instead of
`Molecule.n_electrons()`:

* **MP2, SCS/SOS-MP2, UMP2, CCSD(T), UCCSD, BCCD, FNO-CCSD, A-CCSD(T),
  the DLPNO family and the double hybrids** accept an ECP reference.
  H2S/LANL2DZ MP2 and CCSD agree with PySCF to 1e-6 Ha
  (`tests/test_ecp_correlated.py`).
* **The frozen-core convention subtracts the ECP core per atom.** The
  published element table freezes `fc_A` electrons on atom A; on an ECP
  reference `max(0, fc_A - n_core,A) / 2` orbitals are frozen, so LANL2DZ
  on sulfur (ten removed, ten in the table) freezes nothing and the
  Dirac-Fock small core on iodine (28 removed, 36 in the table) keeps its
  4s4p frozen. `resolve_frozen_core_count(molecule, request,
  reference=scf_result)` is the one resolver; the native kernels refuse
  the element-table default (`n_frozen_core=-1`) on an ECP reference so an
  explicit count always comes from that resolver.
* **ROHF and ROKS** build `T + V_ne(Z_eff) + V_ECP`, the `Z_eff`
  repulsion and the valence count exactly like the native drivers, on
  either ECP route, and record the same provenance fields on their
  results. `ROHFOptions` carries the ECP fields.
* **The determinant family** (CISD, CC3, CCSDT, selected CI, DMRG, v2RDM,
  transcorrelated CI, FCI) builds its Hamiltonian from the reference's
  operator and valence count.
* **The CAS / MR-PT family** (CASCI, CASSCF, MRCI, NEVPT2, CASPT2) and
  **OVGF** accept an ECP reference since 2026-09 (#740). The CAS routes are
  handed the same reference-built Hamiltonian as the determinant family, so
  they already carried the ECP operator; what moved is the active-space
  partition, which counted physical electrons and so returned the removed
  core to the frozen block. H2S/LANL2DZ CASCI(4e,4o) agrees with PySCF's
  CAS on the same ECP to 5e-8 Ha (`tests/test_ecp_correlated.py`). OVGF
  takes its occupied count from the valence count at both its closed- and
  open-shell sites.

  A manual ECP given only through `rhf_options` / `uhf_options`, with no
  basis sidecar, also reaches these routes now. `run_job` builds their
  reference SCF internally, and until 2026-09 that construction ignored the
  caller's options, so such an ECP was silently dropped and the run completed
  on a bare-Z Hamiltonian. Sidecar-attached ECPs were never affected.

  Two caveats. A CASSCF **energy** on an ECP reference is returned, but its
  analytic nuclear gradient is not: that kernel has no ECP derivative, and
  the runner now skips it rather than failing the energy run. And CASSCF's
  orbital optimization has a basin-selection defect (#712) that is present
  on all-electron systems too, so its ECP support is only as good as its
  all-electron support.

The shipped analytic Hessian kernels do not accept ECP inputs. ECP frequency
and thermochemistry work therefore uses `compute_hessian_fd`, which takes
finite differences of the supported ECP-aware analytic gradient while moving
every ECP centre with its atom. Ground-state RHF/UHF/RKS/UKS optimization is
supported through the native, Brent, geomopt, ASE, dimer, and IRC paths.
Excited-state gradients and excited-state optimizations remain unavailable
with ECPs.

CPCM energies support both ECP routes, the bundled sidecars inline and an
explicit XML library, and retain exact high-level SCF provenance even
though the solvent macro-iterations use a low-level JK entry point
internally. Every ECP+CPCM gradient, Hessian, and optimization route is
still unsupported. Population analyses and dipoles use the same per-atom
effective nuclear charges as the valence-only Hamiltonian.

Diagnostic helpers:
`vq.ecp_effective_charges(mol, ecp_centers, library_name)`,
`vq.compute_ecp_gradient_contribution(basis, mol, ecp_centers, D)`,
and its inline counterpart
`vq.compute_ecp_gradient_contribution_from_primitives(basis, mol,
centers, primitives, D)`.

## Phase-14f/g: CRYSTAL `INPUT` ECP-block parser + inline application

Phase 14f shipped on `main` a parser for **CRYSTAL `INPUT`-format
ECP blocks** so basis sets distributed in CRYSTAL's ECP format (the
5th-period pob bases: Rb-I, Cs-Po) parse without raising:

```python
parsed = vq.basis_crystal.parse_crystal_atom_basis(crystal_input_block)
parsed.ecp                          # CrystalECP dataclass
parsed.ecp.terms                    # list of CrystalECPTerm
```

Phase 14g wires those parsed inline ECP terms through libecpint's
`set_ecp_basis(…)` inline-primitive API (no XML library needed):

```python
from vibeqc.basis_crystal import (
    build_periodic_ecp_data,
    crystal_ecp_to_libecpint_arrays,
)

# Convert CRYSTAL ECP → libecpint flat arrays.
arrays = crystal_ecp_to_libecpint_arrays(parsed.ecp)
# arrays.exponents, .coefficients, .ams, .ns -- ready for
# vq.compute_ecp_matrix_from_primitives(basis, center_xyz, [block])

# Or build periodic ECP data for run_periodic_job in one call:
ecp_blocks, ecp_centers, eff_z, ncore = build_periodic_ecp_data(
    system, atom_list
)
```

This is the path `run_periodic_job` uses internally for pob-TZVP-REV2
heavy elements. Applications that need ECPs from non-XML sources
(vDZP custom cores, mixed-row dhf-* archives) can use the same
`ECPPrimitiveBlock` + `compute_ecp_matrix_from_primitives` building
blocks directly.

## Citations

For published work using ECPs:

* **libecpint software**: R. A. Shaw, J. G. Hill,
  *J. Chem. Phys.* **147**, 074108 (2017); R. A. Shaw,
  *J. Chem. Phys.* **159**, 014103 (2023).
* **Stuttgart-Köln MDF family**: Andrae, Häußermann, Dolg,
  Stoll, Preuß, *Theor. Chim. Acta* **77**, 123 (1990) and
  per-element follow-ups in the libecpint repository
  documentation.
* **Hay-Wadt LANL family**: Hay, Wadt, *J. Chem. Phys.* **82**,
  270 (1985); 299 (1985); 284 (1985).

The libecpint upstream project documents the per-element
citations in
[its source repository](https://github.com/robashaw/libecpint).

## See also

* [`basis_sets.md`](basis_sets.md), orbital basis selection,
  including which orbital bases are designed to be paired with
  an ECP.
* [`density_fitting.md`](density_fitting.md), DF / RIJCOSX
  Fock build, fully ECP-aware.
* [`docs/license.md`](../license.md#3b-bundled-ecp-libraries)
  - full ECP licensing inventory.
