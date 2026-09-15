---
myst:
  html_meta:
    "description": "Choosing a basis set in vibe-qc, practical survey of small, medium, and large basis sets for molecular and periodic calculations, broken out by chemical and material category."
    "og:title": "Basis sets in vibe-qc, a practical survey"
    "og:description": "Pople, def2, cc-pVnZ, pcseg, pob-*, MOLOPT, PseudoDojo, NAOs, what to use when, and what vibe-qc ships today."
---

# Basis sets

Modern quantum chemistry has converged on a small number of
well-engineered basis-set families. This page is a practical guide
to picking the right one for your system, with explicit notes on
what vibe-qc ships today and what's on the roadmap.

## Reaching a basis vibe-qc does not bundle

The bundled library carries 255 `.g94` files. The
[Basis Set Exchange](https://www.basissetexchange.org) catalogues 776, so a
set you want may simply not be here. `vibeqc.basis_fetch` renders one from
BSE into a per-user cache, in the same `.g94` (plus `.ecp` sidecar) layout
libint and libecpint already read:

```python
from vibeqc.basis_fetch import fetch_basis_from_bse

got = fetch_basis_from_bse("sadlej pvtz", elements=[1, 8])
print(got.g94_path, got.bse_version, got.elements)
```

It needs the optional extra, which is **not** installed by default:

```
pip install 'vibe-qc[bse]'
```

Three things worth knowing before you rely on it.

**It does not touch the network.** The `basis-set-exchange` distribution
ships the whole catalogue as package data (~334 MB), so a lookup is a local
read: it works on an air-gapped machine, and what you get is pinned by that
package's version, which is recorded in the provenance file written beside
every rendered basis. That is also why it is an optional extra rather than a
dependency.

**It supplies whole sets, and refuses to patch one.** Pass `elements=` and a
record that does not cover all of them is refused, with nothing written to
the cache. Filling a gap in a bundled basis by grafting BSE blocks for the
missing elements would put two sources under one basis name with no reviewed
provenance for the join, so it is not offered. If a bundled basis is missing
an element you need, the coverage guard refuses the calculation, and that is
the intended outcome.

**Nothing resolves differently until you ask.** `run_job` takes the flag,
and only then:

```python
vq.run_job(mol, basis="sadlej pvtz", method="rhf", fetch_from_bse=True)
```

`VIBEQC_FETCH_BSE=1` does the same for a whole session. Without either, an
unbundled name refuses exactly as it always has. A **bundled** name is a
complete no-op even with the flag on: no lookup, no cache entry, and
resolution untouched.

When a render does happen, vibe-qc composes a library root that carries the
current one *plus* the rendered files, and points libint at that. It is a
superset on purpose: pointing libint at the cache alone would hide the
bundled library, so a run using a fetched orbital basis with a bundled
auxiliary would then fail on the auxiliary. Each render prints a warning
naming the basis, the catalogue version and the file, because a non-bundled
basis is something a reader of your results needs to know about.

Every rendered file carries a header saying it is not a bundled vibe-qc
basis, the catalogue version it came from, and BSE's own originating
publications. Cite them as you would for a bundled set; see
[license](../license.md) § 3a for the terms, which are the same.

**Every run records where its basis data came from.** The `.system` manifest
gains a `[basis_library]` section naming the library root libint actually
read, plus one `[[basis_library.fetched]]` row per rendered set:

```toml
[basis_library]
resolved_root  = "/.../build/basis_library"
fetched_count  = 1

[[basis_library.fetched]]
name            = "sadlej pvtz"
bse_version     = "0.12"
element_count   = 25
provenance_path = "/.../cache/basis/sadlej pvtz.provenance.json"
```

`resolved_root` is written on **every** run, fetching or not. vibe-qc prefers
a build overlay over the committed library, so a result produced against a
stale overlay is otherwise indistinguishable from one produced against
current data; now the manifest says which answered. Branch on
`fetched_count`, not on `fetched`: an empty array of tables is dropped from
the file, so on an ordinary run the `fetched` key is absent rather than
empty.

## What vibe-qc ships today

```{admonition} Available now
:class: tip

* **90 standard molecular bases** bundled with libint (Pople,
  Dunning, def2, Sapporo, ANO, core-valence). Use the standard
  names (`"sto-3g"`, `"6-31g*"`, `"cc-pvdz"`, `"def2-tzvp"`, …).
* **pob-TZVP, pob-TZVP-rev2, pob-DZVP-rev2** — the
  Peintinger-Vilela-Oliveira-Bredow family for periodic
  calculations on solids (project author's design papers).
* **`vq.make_basis(mol, name, exp_to_discard=...)`** — primitive
  filter that drops diffuse Gaussians below an exponent
  threshold. PySCF-style; useful for using molecular bases in
  tight ionic crystals. See
  [user guide § linear dependence](linear_dependence.md).
```

```{admonition} Basis-set library coverage today
:class: note

Most of the families discussed below resolve to a real load on a
clean install. The bundled
[`python/vibeqc/basis_library/`](https://github.com/vibe-qc/vibe-qc/tree/main/python/vibeqc/basis_library)
ships 255 Gaussian `.g94` basis files: 90 libint-inherited standard
files plus 165 custom / BSE-derived overlay files, with matching
QVF-Basis sidecars for every set. (A further 11 custom files carry the same
names as libint's own def2 records and replace them, which is how the def2
family reaches past Kr.) That includes pob-TZVP,
pob-DZVP-rev2, pob-TZVP-rev2, the def2 series, the cc-pVxZ
family and the Pople sets. Forty-four of them are valence-only
somewhere and ship their ECP as a `.ecp` sidecar that the molecular SCF
wrappers attach automatically, per element: the def2 family from Rb to Rn
(the libint H-Kr files extended with the Basis Set Exchange blocks and the
def2-ECP), the LANL and dhf families, vDZP, the 3c bases def2-mSVP /
def2-mTZVP / def2-mTZVPP, pob-TZVP-rev2, and the 16 Peterson/Figgen
`{,aug-}cc-p{V,wCV}{D,T,Q,5}Z-PP` sets (Cu-Kr, Y-Xe, Hf-Rn).
`x2c-TZVPall` is all-electron
(it needs no ECP, and none is attached). See [user guide § ECPs](ecp.md).
```

## Using a basis set

```python
from vibeqc import BasisSet
basis = BasisSet(mol, "pob-tzvp")
print(basis.nbasis)      # number of basis functions
```

For periodic systems with diffuse molecular bases, optionally
filter the most-diffuse primitives:

```python
import vibeqc as vq
basis = vq.make_basis(mol, "def2-tzvp", exp_to_discard=0.1)
# — drops every primitive with α < 0.1 and prints the drop log
```

:::{admonition} Discarding primitives reduces, but does not remove, the lattice-sum requirement
:class: warning

A diffuse AO tail forces the real-space lattice sums (overlap, kinetic,
nuclear attraction) to reach further before they converge.
`exp_to_discard` shortens that tail, at the cost of changing the basis
and therefore the answer, but it does not make the requirement go away.
Measure it, don't guess:

```python
from vibeqc.lattice_screening import bloch_overlap_cutoff_bohr
print(bloch_overlap_cutoff_bohr(basis, system))   # bohr
```

On a rocksalt LiF primitive cell (a = 4.03 Å), measured:

| basis | full | with `exp_to_discard=0.1` |
|---|---|---|
| STO-3G | 37.1 bohr | 25.7 bohr |
| def2-SVP | 51.5 bohr | 25.0 bohr (23 → 18 functions) |
| def2-TZVP | 45.3 bohr | 25.0 bohr (45 → 40 functions) |

{py:func}`vibeqc.lattice_screening.bloch_overlap_cutoff_bohr` carries a
25 bohr floor, so no amount of primitive-discarding brings the
requirement below that. Compact ionic cells, and cells containing light
loosely bound atoms (Li especially), are the demanding cases; a cell
like MgO/STO-3G is far less so.

Where the cutoff is settable, set it: `run_periodic_job(...,
bipole_cutoff_bohr=...)` on the BIPOLE route. To check how truncated a
given choice actually is, `vibeqc.pbc_bipole_common.s_fold_truncation_drift`
compares the folded overlap against the same sum at 1.5× the cutoff,
using one-electron integrals only. A drift below `1e-4` is converged;
above `1e-2` the BIPOLE drivers refuse to start. Measured on
LiF/STO-3G/Γ, the drift predicts the total-energy error to within about
a factor of three across four decades, so a drift of `1e-3` implies
roughly 3 mHa.
:::

## Background and terminology

A basis set is the finite set of analytic functions used to expand
the molecular orbitals. Modern molecular codes use **contracted
Gaussian-type orbitals (cGTOs)**, with the primitive radial part
$G_\alpha(r) = N e^{-\alpha r^2}$. Solids more often use plane
waves $e^{i\mathbf{k}\cdot \mathbf{r}}$, numerical atom-centred
orbitals, or augmented plane waves.

Key descriptors:

- **Zeta level (n-zeta)**, number of basis functions used to
  describe each valence orbital. STO-3G is single-zeta. 6-31G,
  def2-SVP, cc-pVDZ are double-zeta. def2-TZVP, cc-pVTZ, pcseg-2
  are triple-zeta. cc-pVQZ, def2-QZVP are quadruple-zeta.
- **Polarisation functions**, higher-angular-momentum functions
  (d on C, p on H, f on transition metals, etc.) that allow the
  electron density to deform on bond formation. **Modern usage
  demands them**: unpolarised basis sets like 6-31G or
  cc-pVDZ-without-d should never be used for production.
- **Diffuse functions**, low-exponent functions needed for
  anions, weakly bound species, Rydberg states, and many response
  properties (polarisabilities, NMR shielding, TDDFT excitations).
  Indicated by `+`, `++`, `aug-`, or `ma-` prefixes.
- **Contraction**, segmented vs. general. Segmented (Pople,
  def2, pcseg) is faster in most integral codes. Generally
  contracted (cc, ANO) is more compact and accurate per primitive
  but needs a code that exploits it.
- **Effective core potential (ECP)**, replaces inner-shell
  electrons with a parameterised potential. Reduces cost, builds
  in scalar relativistic effects. The Stuttgart/Cologne ECPs
  paired with def2 valence sets are the de facto standard above
  Kr.
- **Polarisation-consistent vs. correlation-consistent**,
  cc-pVnZ was optimised to recover correlation energy in
  CCSD/CCSD(T). pc-n / pcseg-n was optimised to converge HF and
  DFT energies. For DFT, pcseg-n is technically the better-targeted
  choice, but def2-TZVP achieves nearly the same accuracy at lower
  cost in practice.

A note on **basis set superposition error (BSSE)** and **basis
set incompleteness error (BSIE)**: these are large at the
double-zeta level (several kcal/mol on noncovalent interactions)
and are the reason a triple-zeta basis is now the recommended
floor for any quantitative molecular work. The geometrical
counterpoise correction (gCP) of Kruse and Grimme is the cheapest
practical fix and is built into the 3c composite methods below.

## Choosing a molecular basis set

### Small, fast-screening / pedagogical tier

| Basis | Zeta | Notes |
|---|---|---|
| **STO-3G** | SZ | Single-zeta minimal, no polarisation. Useful only for initial guesses, very large systems where qualitative geometries suffice, or pedagogy. |
| **MINI / MINIs** (Huzinaga) | SZ | Slightly better than STO-3G atomic energies. Niche use. |
| **3-21G** | DZ (unpol) | Old Pople split-valence. Faster than 6-31G but unpolarised. Avoid for production. |
| **pcseg-0 / pc-0** | DZ (unpol) | Jensen DZ without polarisation. Better-balanced than 3-21G. |
| **6-31G** | DZ (unpol) | Historical workhorse without polarisation. Pitman et al. (2024) showed unpolarised DZ basis sets perform poorly and should not be used. |

**Bottom line.** Support STO-3G (legacy + pedagogical) and 3-21G
(rare speed-dominated contexts). All other small basis sets
should carry a "not for production" warning.

### Medium, polarised double-zeta, the everyday tier

| Basis | Notes |
|---|---|
| **def2-SVP** | Karlsruhe segmented DZ with polarisation on heavy atoms. The most widely used DZ basis in modern DFT codes. Best speed/accuracy balance for routine geometry optimisation. |
| **vDZP** (Grimme 2023) | Custom polarised DZ used inside ωB97X-3c. Large-core ECPs and deep contraction. On TorsionNet206 drug-like benchmark, vDZP-based methods give MAEs of 0.4-0.5 kcal/mol, comparable to triple-zeta hybrid methods. |
| **6-31++G(d,p)** | Pitman et al. (2024) found this the best-performing DZ basis on diet-GMTKN55. Diffuse functions matter. |
| **6-31G(d,p) / 6-31G\*\*** | Classic Pople workhorse. Acceptable for organic structures, inferior to def2-SVP at similar cost. |
| **pcseg-1** | Jensen segmented DZ-polarised. Slightly better than def2-SVP for HF/DFT energies, less widely supported. |
| **cc-pVDZ** | Dunning DZ. Designed for correlation; for pure DFT less efficient than pcseg-1 or def2-SVP. Useful as a stepping stone in cc-pVnZ extrapolations. |
| **def2-mSVP** | Modified def2-SVP used inside PBEh-3c and B3LYP-3c. Not for standalone use. |
| **LANL2DZ** | Hay-Wadt ECP plus DZ valence. Historically common for transition metals but inferior to def2-SVP/TZVP plus def2-ECP. **Should be deprecated.** |

**Bottom line.** def2-SVP is the default. vDZP is a strong contender
for any system with heavy atoms.

### Large, triple-zeta and above, production accuracy

| Basis | Notes |
|---|---|
| **def2-TZVP** | The de facto reference for hybrid-DFT calculations on molecules of any size where it is affordable. Within 0.5 kcal/mol of def2-QZVPD for main-group thermochemistry. |
| **def2-TZVPP** | Extra polarisation on hydrogen. Recommended for transition metals, weak interactions, properties beyond geometries. |
| **pcseg-2** | Best-performing TZ basis for DFT thermochemistry on diet-GMTKN55. Slightly better than def2-TZVP, less widely available. |
| **cc-pVTZ / aug-cc-pVTZ** | Dunning TZ, the reference for post-HF (MP2, CCSD(T)). Use aug- variant for anions, noncovalent, response. |
| **def2-QZVP / def2-QZVPP** | Quadruple-zeta. For benchmarking small molecules, basis-set-limit reference, double-hybrid functionals. |
| **cc-pVQZ / aug-cc-pVQZ** | Quadruple-zeta correlation-consistent. Standard for CCSD(T)/CBS extrapolation in `(T,Q)` or `(Q,5)` schemes. |
| **pcseg-3 / pcseg-4** | Larger pc-n family members. Excellent DFT convergence but rarely needed in practice. |
| **def2-TZVPD / ma-def2-TZVP** | TZ with diffuse functions. Use for anions, TDDFT, polarisabilities. The "ma-" (minimally augmented) Truhlar-Zheng variants are a cheaper alternative to full aug-. |
| **cc-pV5Z, cc-pV6Z, aug-cc-pV5Z** | For very small systems and CBS extrapolation only. |
| **ANO-RCC-VTZP, ANO-RCC-VQZP** | Generally contracted relativistic ANOs. Standard for CASPT2/NEVPT2/MS-CASPT2. Compact per primitive, ideal for multireference. |

**Bottom line.** def2-TZVP is the production-quality default.
cc-pVTZ family is the reference for post-HF. ANO-RCC is mandatory
for multireference.

### By chemical category

#### Organic molecules (C, H, N, O, S, halogens, P)

| Tier | Recommendation |
|---|---|
| Small (geometry, screening) | def2-SVP or B97-3c (def2-mTZVP + D3 + gCP) |
| Medium (production) | **def2-TZVP** with a hybrid functional (ωB97X-D, ωB97M-V, M06-2X, B3LYP-D3) |
| Large (benchmark) | aug-cc-pVTZ → aug-cc-pVQZ extrapolation, or ωB97M-V/def2-QZVPP |
| Anions, Rydberg, response | aug- variants required (aug-cc-pVTZ, ma-def2-TZVP, aug-pcseg-2) |

Pople basis sets (6-31G(d), 6-311G(2df,p)) remain in widespread use
due to inertia. The polarised 6-311G family is poorly parameterised
(Pitman et al. 2024 explicitly recommends avoiding it). For new
work, support 6-31G(d) and 6-31++G(d,p) for compatibility but
encourage def2.

#### Metal-organic / organometallics (TM complexes, MOFs)

| Tier | Recommendation |
|---|---|
| Small (screening) | def2-SVP (with def2-ECP for 4d/5d) plus dispersion correction |
| Medium (production) | **def2-TZVP / def2-TZVPP** on the metal, def2-TZVP on ligands. Use a TM-tested functional: TPSSh, B3LYP\*-D3(BJ), or PWPB95-D3(BJ) for benchmarking. |
| Large (benchmark) | def2-QZVPP, x2c-TZVPPall for explicit relativity |
| Multireference | **ANO-RCC-VTZP / VQZP** with CASPT2, NEVPT2, or DMRG-CASPT2 |
| Spin states | Truhlar's group recommends def2-TZVP for spin-splitting energies; extra polarisation does not help |

Avoid LANL2DZ for new work, Stuttgart/Cologne ECPs (built into
def2) are more accurate.

For 4d / 5d transition metals, scalar relativistic effects matter.
Either use def2-ECP, or for explicit treatment use dhf-TZVP /
x2c-TZVPall (Pollak-Weigend 2017) or DKH-recontracted def2
(def2-TZVP-DKH).

#### Inorganic main-group (clusters, oxides, halides, hypervalent)

| Tier | Recommendation |
|---|---|
| Small | def2-SVP (with care for hypervalent S, P needing extra d) |
| Medium | **def2-TZVPPD** (the extra d and diffuse matter) or aug-cc-pV(T+d)Z for second-row hypervalent |
| Large | def2-QZVP, aug-cc-pwCVTZ-DK for explicit core-valence |
| Anions, halide complexes | aug- variants required |
| Heavy main group (Sn, Pb, Bi) | def2-TZVP plus ECP, or all-electron x2c-TZVPall |

For second-row hypervalent (SO₃²⁻, SF₆, PF₅, ClO₄⁻), the "tight d"
variants (e.g. aug-cc-pV(T+d)Z, Dunning et al. 2001) are essential.
The def2 family already includes appropriate d-functions for these
atoms.

#### Biomolecules (peptides, nucleic acids, lipids, sugars)

| Tier | Recommendation |
|---|---|
| Screening (large fragments) | **r2SCAN-3c** (def2-mTZVPP + D4 + gCP) or B97-3c |
| Production (single-point on optimised fragments) | def2-TZVP with ωB97X-D or B3LYP-D3(BJ) |
| Noncovalent / hydrogen bonding | def2-TZVPPD or ma-def2-TZVP, with explicit dispersion (D4 preferred) |
| Vibrational, NMR | property-specific basis sets (pcS-n for shielding, pcJ-n for J couplings) |

The 3c composite methods are particularly well-suited here because
gCP corrects for the BSSE that dominates large-biomolecule
binding-energy errors. Recent benchmarks (Behara et al. 2024,
TorsionNet206) showed B3LYP-D3BJ/6-31G(d) is now clearly inferior
to vDZP-based or 3c methods at comparable cost.

#### Pharma / drug-like molecules

| Tier | Recommendation |
|---|---|
| High-throughput conformer / torsion scans | **ωB97X-3c (vDZP)** or r2SCAN-3c |
| Reference single-points for force-field parametrisation | ωB97M-V/def2-TZVPPD or DLPNO-CCSD(T)/def2-TZVPP |
| QM/MM cores | def2-SVP (QM region), with def2-TZVP single-point refinement |

vDZP deserves particular attention here. The Rowan/Wagen
TorsionNet206 study showed vDZP-based methods give 0.4-0.5
kcal/mol MAE on torsion energies (versus CCSD(T)/def2-TZVP),
comparable to standard hybrid functionals with triple-zeta basis
sets. The use of ECPs makes it especially efficient for
halogenated and metal-containing drug candidates.

## Choosing a basis set for solids

Periodic codes split into three philosophical camps:

1. **Plane waves + pseudopotentials.** VASP, Quantum ESPRESSO,
   ABINIT, CASTEP, GPAW. A single cutoff energy parameter controls
   the basis. No BSSE. Smooth convergence. Heavy reliance on the
   quality of the pseudopotential or PAW dataset.
2. **Gaussian basis sets.** CRYSTAL, CP2K (mixed Gaussian-plane
   wave), Turbomole-Riper, FHI-aims (in some modes). Inherits
   molecular-style basis sets, but they need re-optimisation for
   solids, diffuse functions in molecular bases lead to linear
   dependencies in periodic systems.
3. **Numerical atomic orbitals or augmented plane waves.**
   FHI-aims, SIESTA, OpenMX (NAO); WIEN2k, exciting, ELK
   (LAPW+lo). Highly accurate, smaller basis sizes but slower per
   evaluation.

**vibe-qc is family 2** (Gaussian basis, with periodic-image
folding for solid-state). For users moving back and forth between
codes, the surveys below cover all three.

### Plane-wave pseudopotential libraries

The pseudopotential is the basis-set-equivalent choice for
plane-wave codes. The cutoff is the convergence parameter.

| Library | Code | Type | Characteristics |
|---|---|---|---|
| **VASP PAW (PBE / PBE.54 / PBE.64)** | VASP | PAW | The de facto industry standard for materials science. `_GW`, `_sv` / `_pv` variants for semicore. Closed source. Cutoffs 250-600 eV by element. |
| **SSSP precision (1.3.0)** | Quantum ESPRESSO | mixed PAW + USPP + ONCV | EPFL/Marzari curated. Most accurate open-source PSP library: avg Δ-factor below 0.4 meV/atom against all-electron references (Prandini et al. 2018). |
| **SSSP efficiency (1.3.0)** | QE | mixed | Same protocol with lower cutoffs, faster, slightly less accurate. Standard for high-throughput. |
| **PseudoDojo (ONCV, NC SR/FR)** | abinit, QE, others | norm-conserving (ONCV) | Hamann-Schlüter-Chiang; scalar and fully-relativistic versions. State of the art for systematic, hybrid-functional, and GW calculations. |
| **SG15** | many | ONCV | Schlipf-Gygi 2015. Solid alternative to PseudoDojo, well-tested. |
| **GBRV** | QE, ABINIT | USPP, PAW | Garrity-Bennett-Rabe-Vanderbilt high-throughput library. |
| **JTH PAW** | ABINIT | PAW | Jollet-Torrent-Holzwarth. ABINIT-native PAW. |

The 2016 multi-code Δ-factor study (Lejaeghere et al., *Science*)
showed that all of these libraries now agree with all-electron
LAPW codes (WIEN2k, exciting) to within roughly 1 meV/atom on
equation-of-state data, the practical floor for DFT precision.

### Gaussian basis sets for periodic systems

| Basis | Code | Notes |
|---|---|---|
| **pob-TZVP** | CRYSTAL, **vibe-qc** | Peintinger-Vilela Oliveira-Bredow 2013 (project author). TZ polarised for solids, derived from def2-TZVP by re-optimisation to remove diffuse linear dependencies. CRYSTAL community standard. |
| **pob-TZVP-rev2** | CRYSTAL, **vibe-qc** | Vilela Oliveira et al. 2019. Improved performance for transition metals. **Use this over the original pob-TZVP for any new work.** |
| **pob-DZVP-rev2** | CRYSTAL, **vibe-qc** | DZ companion to pob-TZVP-rev2. |
| **MOLOPT (SZV/DZVP/TZVP/TZV2P)** | CP2K | Optimised by VandeVondele-Hutter for numerical stability in periodic systems with the GPW approach. CP2K standard. |
| **dcm-TZVP** | CRYSTAL | Daga-Civalleri-Maschio 2020. System-specific re-optimisations for diamond, graphene, carbyne. Illustrative of all-purpose-library limits. |
| **MOLOPT-aug (aug-MOLOPT-ae)** | CP2K | Augmented MOLOPT for excited states (TDDFT, BSE-GW). Recent. |

vibe-qc users default to **pob-TZVP-rev2** for ionic crystals; see
the *pob basis sets in detail* section further down this page and
[user guide § linear dependence](linear_dependence.md) for the
design philosophy.

### Numerical atomic orbitals + (L)APW+lo

| Basis / tier | Code | Notes |
|---|---|---|
| **FHI-aims tier 1** | FHI-aims | "Light" species defaults. Polarised DZ-equivalent. Sub-meV precision for many energy differences. |
| **FHI-aims tier 2** | FHI-aims | "Tight" species defaults. Roughly QZVP-quality but smaller. FHI-aims production standard. |
| **FHI-aims tier 3 / 4** | FHI-aims | Reference quality. For benchmarking only. |
| **tier2_aug2** | FHI-aims | tier 2 plus two lowest-l aug-cc functions. Required for excited-state and weakly-bound-anion calculations. |
| **SIESTA SZ / DZ / DZP / TZP** | SIESTA | Numerical pseudo-atomic orbitals with confinement. DZP is the standard SIESTA production basis. |
| **(L)APW+lo** | WIEN2k, exciting | Augmented plane waves. The all-electron gold standard for periodic DFT. Slower than PAW but no pseudopotential approximation. |

### By solid type

#### Metals (elemental, alloys, intermetallics)

| Tier | Recommendation |
|---|---|
| Small (screening) | PAW with default cutoff (e.g. VASP `ENCUT` from POTCAR), dense k-mesh (Methfessel-Paxton smearing) |
| Medium (production) | PseudoDojo standard ONCV or VASP PAW with semicore states (`_sv`, `_pv`) for early TMs, cutoff 400-500 eV |
| Large (benchmark) | LAPW+lo (WIEN2k, exciting) with all-electron references |
| Magnetic systems | Always include semicore states. For 3d magnetism use `_pv` or `_sv` variants. For spin-orbit use fully-relativistic ONCV. |

Common pitfalls: forgetting semicore p-states in early transition
metals (Sc-Cr), too-low cutoff for d-block elements, underconverged
k-meshes. Choudhary-Tavazza 2019 NIST study on 30,000+ materials
provides good convergence heuristics by element.

#### Oxides (binary oxides, perovskites, TM oxides)

| Tier | Recommendation |
|---|---|
| Small | PAW with O standard, screening-functional (PBEsol or SCAN) |
| Medium | **VASP PAW with O_h or O_s + high cutoff (520 eV minimum, often 600-700 eV)**. Materials Project standard is 520 eV with O standard. SSSP precision in QE. |
| Large | PAW + DFT+U (Dudarev) or hybrid (HSE06) for correlated TM oxides. For benchmarking, all-electron LAPW. |
| Strongly correlated | DFT+U with carefully chosen U (e.g. Materials Project tabulated values), or hybrid (HSE06, PBE0). |
| Defects, polarons | Hybrid (HSE06) and large supercell. |

Oxygen p-states are deep and the O 2s semicore needs a cutoff
substantially above what alkali-halide systems require. The 520 eV
Materials Project default is widely accepted.

#### Semiconductors (Si, Ge, III-V, II-VI, halide perovskites)

| Tier | Recommendation |
|---|---|
| Small | PAW with PBE, modest cutoff (300-400 eV) |
| Medium | **PAW or ONCV (PseudoDojo) with PBE for structure**, hybrid (HSE06) or G₀W₀ for band gaps. Include spin-orbit for heavy elements (GaAs, halide perovskites). |
| Large (band-gap accuracy) | **G₀W₀ on top of HSE06 or PBE**, with NAO tier 2 + aug2 (FHI-aims) or LAPW + high-energy local orbitals (HLO). |
| Excitonic absorption | BSE-GW with appropriate basis (aug-MOLOPT-ae in CP2K, tier2+aug2 in FHI-aims, LAPW+HLO+lo in exciting). |

For halide perovskites and other heavy-element semiconductors,
scalar + spin-orbit relativity is essential. Use fully-relativistic
PseudoDojo or VASP PAW with `LSORBIT = .TRUE.`.

## Composite "3c" methods

These bundle a tailored basis with corrections (geometric
counterpoise gCP, dispersion D3/D4, sometimes a modified short-range
correction).

| Method | Functional | Basis | Year | Notes |
|---|---|---|---|---|
| **HF-3c** | Hartree-Fock | MINIX | 2013 | Cheapest. Useful for very large systems where mean-field qualitative behaviour suffices. |
| **PBEh-3c** | PBE0 (42% HF) | def2-mSVP | 2015 | Hybrid. Strong for noncovalent geometries. |
| **HSE-3c** | HSE | def2-mSVP | 2016 | Screened-exchange variant of PBEh-3c. Same basis and short-range potential, but its own D3(BJ) damping. |
| **B97-3c** | B97 GGA | def2-mTZVP | 2018 | Workhorse GGA. Especially recommended for transition-metal systems. |
| **r2SCAN-3c** | r²SCAN meta-GGA | def2-mTZVPP | 2021 | Current "Swiss army knife" recommendation. Excellent default for routine work, geometries, thermochemistry, noncovalent. |
| **B3LYP-3c** | B3LYP | def2-mSVP | 2022 | Tailored for IR spectra (B3LYP frequencies are particularly accurate). |
| **ωB97X-3c** | ωB97X-V | **vDZP** | 2023 | Range-separated hybrid plus vDZP. Often the best 3c for thermochemistry and barrier heights. |

The published 3c benchmark MAEs on GMTKN55 are competitive with
much more expensive calculations. **First-class support for the
modified bases (def2-mSVP, def2-mTZVP, def2-mTZVPP, vDZP) plus
gCP and D4 corrections is the path to making these turnkey**,
roadmap item.

## Property-specific basis sets

Critical to know about, but not "general-purpose" so they don't
appear in the main rankings:

| Property | Basis family | Reference |
|---|---|---|
| NMR shielding | **pcS-n** (Jensen) | Jensen 2008, *J. Chem. Theory Comput.* **4**, 719 |
| NMR J-coupling | **pcJ-n** (Jensen) | Jensen 2006, *J. Chem. Theory Comput.* **2**, 1360 |
| NMR (relativistic) | **x2c-TZVPall-s** (Franzke-Weigend) | Franzke et al. 2019, *Phys. Chem. Chem. Phys.* **21**, 16658 |
| EPR (g-tensor, hyperfine) | **EPR-II, EPR-III, x2c-TZVPall-s** | Barone 1996; Franzke et al. |
| Core-level X-ray (XPS, NEXAFS) | **cc-pCVnZ, aug-cc-pCVnZ, ccX-DK** | Peterson et al.; Hanson-Heine et al. |
| Polarisability, hyperpolarisability | **Sadlej, aug-cc-pVnZ, LPolX** | Sadlej 1988; Bauernschmitt-Ahlrichs 1996 |
| Excited states (TDDFT) | aug-cc-pVTZ, ma-def2-TZVPP, aug-pcseg-2 | Many |

## Master recommendation table

Compact view of the recommended sets at each tier and category.
Everything listed is freely available from the
[Basis Set Exchange](https://www.basissetexchange.org) or built
into common open-source codes.

| Domain | Small (fast) | Medium (production) | Large (benchmark) |
|---|---|---|---|
| Organic molecules | def2-SVP, B97-3c | **def2-TZVP** | def2-QZVPP, aug-cc-pVQZ |
| Metal-organic | def2-SVP + def2-ECP | **def2-TZVPP** + def2-ECP | def2-QZVPP, ANO-RCC-VTZP (multiref) |
| Inorganic main-group | def2-SVP | def2-TZVPPD | aug-cc-pV(Q+d)Z, def2-QZVPPD |
| Biomolecules | r2SCAN-3c, B97-3c | def2-TZVP + D4 | DLPNO-CCSD(T)/def2-TZVPP |
| Pharma / drug-like | ωB97X-3c (vDZP) | def2-TZVPPD | ωB97M-V/def2-QZVPP |
| Excited states | def2-SVPD | ma-def2-TZVPP, aug-pcseg-2 | aug-cc-pVQZ, tier2+aug2 |
| Multireference | ANO-RCC-VDZP | ANO-RCC-VTZP | ANO-RCC-VQZP |
| Heavy elements (relativistic) | x2c-SVPall, dhf-SVP | x2c-TZVPall, dhf-TZVPP | x2c-QZVPPall |
| Solids: metals (PW) | PseudoDojo standard | **PseudoDojo stringent / SSSP precision** | LAPW+lo (WIEN2k) |
| Solids: oxides (PW) | VASP PAW (520 eV) | **VASP PAW + HSE06**, SSSP precision | LAPW+lo, all-electron |
| Solids: semiconductors (PW) | PAW + PBE | **PAW + G₀W₀ on HSE06** | LAPW+HLO, BSE-GW |
| Solids: Gaussian periodic | pob-DZVP | **pob-TZVP-rev2**, MOLOPT TZVP | dcm-TZVP (system-specific) |
| Solids: NAO all-electron | FHI-aims tier 1 (light) | **FHI-aims tier 2 (tight)** | FHI-aims tier 3, LAPW+lo |

Bold entries are the "if you support exactly one thing in this
category, support this" recommendations.

## pob-\* basis sets in detail

The pob-\* sets were designed by the project author and
collaborators around the CRYSTAL code's requirements for
solid-state calculations. Standard molecular TZVP bases contain
very diffuse functions that, when periodic images overlap across
unit cells, produce near-linear-dependent basis sets, SCF
convergence suffers or fails outright. The pob family tightens
those functions to make them usable in solids.

### Coverage

| Name | Elements | Reference |
|---|---|---|
| pob-TZVP | H-Br (32) | [Peintinger, Vilela Oliveira, Bredow, *J. Comput. Chem.* **34**, 451 (2013)](https://doi.org/10.1002/jcc.23153) |
| pob-TZVP-rev2 | H-Br (32) | [Vilela Oliveira, Laun, Peintinger, Bredow, *J. Comput. Chem.* **40**, 2364 (2019)](https://doi.org/10.1002/jcc.26013) |
| pob-DZVP-rev2 | subset (19) | *(same as above)* |

### Heavy elements

The pob-\* archives for Rb-I (up to Z=53), Cs-Po (to Z=84), and
La-Lu (Z=57-71) require effective core potentials. vibe-qc's
CRYSTAL parser *recognises* the ECP blocks in those files and
the libecpint integration ships in v0.4.0+, see the
[roadmap](../roadmap.md) for the full ECP coverage state.

### Method scope

The pob-\* basis sets are **optimised for DFT and hybrid-DFT
work on solids**, that's the regime the parameter
re-optimisations in Peintinger 2013 / Vilela Oliveira 2019
targeted, and the regime where the bundled `.g94` files have
been validated. **For pure Hartree-Fock or correlated methods
(MP2, CCSD, ...) on periodic systems, pob-\* is not the right
basis choice**: SCF convergence under defaults is unreliable
(e.g. MgO RHF/pob-TZVP requires aggressive `LEVSHIFT` and 200+
cycles even in CRYSTAL14 itself), and the basis was never
recommended for correlated treatment.

For periodic RHF / RKS-without-hybrid-mixing parity work, prefer
either STO-3G (for cheap, robust baselines) or a hardened
def2-family basis with diffuse primitives discarded
(`exp_to_discard=0.1`). The bundled
[`baseline_sto3g/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/crystal_parity/baseline_sto3g)
reference inputs follow this principle.

## Re-fetching the pob-\* library

Because we ship the `.g94` conversions in the repository, a fresh
clone already has them under `basis_library/custom/`. If you want
to re-fetch from the Bredow group server (e.g. for a future
revision):

```sh
python -m vibeqc.basis_crystal fetch
./scripts/setup_basis_library.sh
```

The fetcher writes to `basis_library/custom/`; the setup script
assembles `basis_library/basis/` from libint's stock set plus
those customs.

## Parsing a CRYSTAL-format basis directly

```python
from vibeqc.basis_crystal import parse_crystal_atom_basis_file, emit_g94

atom = parse_crystal_atom_basis_file("path/to/06_C")
print(atom.element_symbol, len(atom.shells))
with open("my-basis.g94", "w") as f:
    f.write(emit_g94([atom]))
```

Drop the resulting `my-basis.g94` into `basis_library/custom/` and
run `./scripts/setup_basis_library.sh` to pick it up.

### CRYSTAL format in one page

The per-element files that Bredow, CRYSTAL's library, and most
other solid-state QC projects distribute use CRYSTAL's native input
format:

```
Z  NSHELL                           ← header: atomic number, shell count
ITYPE  LAT  NG  CHE  SCAL           ← shell 1 header
  exp₁  coef₁                       ← NG primitive lines
  exp₂  coef₂
  ...
ITYPE  LAT  NG  CHE  SCAL           ← shell 2 header
  ...
```

Each shell header has:

| Token  | Meaning |
|---|---|
| `ITYPE` | Format code; `0` = user-defined contracted Gaussians (what the Bredow files use). |
| `LAT`   | Shell angular-momentum code: `0=S`, `1=SP`, `2=P`, `3=D`, `4=F`, `5=G`. |
| `NG`    | Number of primitive Gaussians in the contraction. |
| `CHE`   | Formal occupancy of the shell (electrons). |
| `SCAL`  | Scale factor; primitive exponents are multiplied by $\text{SCAL}^2$. |

For ECP-bearing atoms (heavy elements, Rb-I and beyond in the
pob family), the first line uses `200+Z` and an `INPUT` block
follows the header describing the ECP.

`vibeqc.basis_crystal.parse_crystal_atom_basis_file(path)` returns
a dataclass that mirrors this structure:

```python
>>> atom = parse_crystal_atom_basis_file("pob-TZVP/06_C")
>>> atom.Z, atom.element_symbol
(6, 'C')
>>> atom.shells[0].shell_type, atom.shells[0].n_primitives
('S', 6)
>>> atom.shells[0].exponents[:2]
[13575.349682, 2035.2333680]
```

`emit_g94(list_of_atom_basis)` renders them to libint's `.g94`
(NWChem-format) text.

## Key references

### Reviews and best-practice guides

* Bursch, M., Mewes, J.-M., Hansen, A., Grimme, S. *Best-Practice
  DFT Protocols for Basic Molecular Computational Chemistry.*
  *Angew. Chem. Int. Ed.* **2022**, *61*, e202205735.
  [doi:10.1002/anie.202205735](https://doi.org/10.1002/anie.202205735)
* Karton, A. *Good Practices in Database Generation for
  Benchmarking Density Functional Theory.* *WIREs Comput. Mol.
  Sci.* **2025**, *15*, e1737.
  [doi:10.1002/wcms.1737](https://doi.org/10.1002/wcms.1737)
* Goerigk, L., Hansen, A., Bauer, C., Ehrlich, S., Najibi, A.,
  Grimme, S. *A look at the density functional theory zoo with
  the advanced GMTKN55 database.* *Phys. Chem. Chem. Phys.*
  **2017**, *19*, 32184.
  [doi:10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G)
* Pitman, S. J., Evans, A. K., Ireland, R. T., Lempriere, F.,
  McKemmish, L. K. *Benchmarking Basis Sets for DFT
  Thermochemistry: Why Unpolarized Basis Sets and the Polarized
  6-311G Family Should Be Avoided.* *J. Phys. Chem. A* **2023**,
  *127*, 10295.
  [doi:10.1021/acs.jpca.3c05573](https://doi.org/10.1021/acs.jpca.3c05573)

### Pople family

* Hehre, W. J., Stewart, R. F., Pople, J. A. *J. Chem. Phys.*
  **1969**, *51*, 2657 (STO-nG).
* Binkley, J. S., Pople, J. A., Hehre, W. J. *J. Am. Chem. Soc.*
  **1980**, *102*, 939 (3-21G).
* Hariharan, P. C., Pople, J. A. *Theor. Chim. Acta* **1973**,
  *28*, 213 (6-31G(d,p)).
* Krishnan, R., Binkley, J. S., Seeger, R., Pople, J. A. *J. Chem.
  Phys.* **1980**, *72*, 650 (6-311G).

### Dunning correlation-consistent

* Dunning, T. H. Jr. *J. Chem. Phys.* **1989**, *90*, 1007 (cc-pVnZ).
* Kendall, R. A., Dunning, T. H. Jr., Harrison, R. J. *J. Chem.
  Phys.* **1992**, *96*, 6796 (aug-cc-pVnZ).
* Dunning, T. H. Jr., Peterson, K. A., Wilson, A. K. *J. Chem.
  Phys.* **2001**, *114*, 9244 (cc-pV(n+d)Z).

### Karlsruhe def2 family

* Weigend, F., Ahlrichs, R. *Phys. Chem. Chem. Phys.* **2005**,
  *7*, 3297 (def2 series).
* Weigend, F. *Phys. Chem. Chem. Phys.* **2006**, *8*, 1057
  (def2/J auxiliary).
* Rappoport, D., Furche, F. *J. Chem. Phys.* **2010**, *133*,
  134105 (def2-XVPD diffuse-augmented).

### Jensen polarisation-consistent

* Jensen, F. *J. Chem. Phys.* **2001**, *115*, 9113 (pc-n).
* Jensen, F. *J. Chem. Phys.* **2002**, *117*, 9234 (aug-pc-n).
* Jensen, F. *J. Chem. Theory Comput.* **2014**, *10*, 1074 (pcseg-n).

### Relativistic and ANO

* Pollak, P., Weigend, F. *J. Chem. Theory Comput.* **2017**,
  *13*, 3696 (x2c-XVPall).
* Franzke, Y. J., Spiske, L., Pollak, P., Weigend, F. *J. Chem.
  Theory Comput.* **2020**, *16*, 5658 (x2c-QZVPPall).
* Roos, B. O., Lindh, R., Malmqvist, P.-Å., Veryazov, V., Widmark,
  P.-O. *J. Phys. Chem. A* **2005**, *109*, 6575 (ANO-RCC TM).
* Zobel, J. P., Widmark, P.-O., Veryazov, V. *J. Chem. Theory
  Comput.* **2020**, *16*, 278 (ANO-R).

### Composite 3c methods

* Sure, R., Grimme, S. *J. Comput. Chem.* **2013**, *34*, 1672
  (HF-3c).
* Grimme, S., Brandenburg, J. G., Bannwarth, C., Hansen, A. *J.
  Chem. Phys.* **2015**, *143*, 054107 (PBEh-3c).
* Brandenburg, J. G., Bannwarth, C., Hansen, A., Grimme, S. *J.
  Chem. Phys.* **2018**, *148*, 064104 (B97-3c).
* Grimme, S., Hansen, A., Ehlert, S., Mewes, J.-M. *J. Chem.
  Phys.* **2021**, *154*, 064103 (r2SCAN-3c).
* Müller, M., Hansen, A., Grimme, S. *J. Chem. Phys.* **2023**,
  *158*, 014103 (ωB97X-3c, vDZP).
* Wagen, C. C. *The vDZP Basis Set Is Effective for Many Density
  Functionals.* Rowan publication, **2024** (TorsionNet206
  benchmarks).

### Periodic Gaussian basis sets

* Peintinger, M. F., Vilela Oliveira, D., Bredow, T. *J. Comput.
  Chem.* **2013**, *34*, 451 (pob-TZVP).
  [doi:10.1002/jcc.23153](https://doi.org/10.1002/jcc.23153)
* Vilela Oliveira, D., Laun, J., Peintinger, M. F., Bredow, T.
  *J. Comput. Chem.* **2019**, *40*, 2364 (pob-TZVP-rev2).
  [doi:10.1002/jcc.26013](https://doi.org/10.1002/jcc.26013)
* Laun, J., Vilela Oliveira, D., Bredow, T. *J. Comput. Chem.*
  **2018**, *39*, 1285.
* VandeVondele, J., Hutter, J. *J. Chem. Phys.* **2007**, *127*,
  114105 (MOLOPT).
* Daga, L. E., Civalleri, B., Maschio, L. *J. Chem. Theory
  Comput.* **2020**, *16*, 2192 (dcm-TZVP).

### Plane wave and PAW pseudopotentials

* Blöchl, P. E. *Phys. Rev. B* **1994**, *50*, 17953.
* Kresse, G., Joubert, D. *Phys. Rev. B* **1999**, *59*, 1758
  (VASP PAW).
* Hamann, D. R. *Phys. Rev. B* **2013**, *88*, 085117 (ONCV).
* van Setten, M. J. et al. *Comput. Phys. Commun.* **2018**,
  *226*, 39 (PseudoDojo).
* Schlipf, M., Gygi, F. *Comput. Phys. Commun.* **2015**, *196*,
  36 (SG15).
* Prandini, G., Marrazzo, A., Castelli, I. E., Mounet, N.,
  Marzari, N. *npj Comput. Mater.* **2018**, *4*, 72 (SSSP).
* Lejaeghere, K. et al. *Science* **2016**, *351*, aad3000
  (Δ-factor benchmark across codes).
* Garrity, K. F., Bennett, J. W., Rabe, K. M., Vanderbilt, D.
  *Comput. Mater. Sci.* **2014**, *81*, 446 (GBRV).

### Numerical atomic orbitals

* Blum, V. et al. *Comput. Phys. Commun.* **2009**, *180*, 2175
  (FHI-aims).
* Soler, J. M. et al. *J. Phys.: Condens. Matter* **2002**, *14*,
  2745 (SIESTA).

### Database

* Pritchard, B. P., Altarawy, D., Didier, B., Gibson, T. D.,
  Windus, T. L. *J. Chem. Inf. Model.* **2019**, *59*, 4814.
  [Basis Set Exchange](https://www.basissetexchange.org)
