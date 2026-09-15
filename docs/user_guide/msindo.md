# MSINDO (semiempirical INDO)

See also the [tutorial](../tutorial/msindo.md) for a worked walkthrough.

**MSINDO** (Modified Symmetrically-orthogonalised INDO) is a semiempirical
SCF-MO method of the INDO family, built on **Slater-type orbitals** with a
symmetric (Löwdin) orthogonalisation folded into the one-electron resonance
integrals.  It was developed by Ahlswede, Bredow, Geudtner & Jug at the Mulliken
Center for Theoretical Chemistry, University of Bonn.

vibe-qc carries an **independent re-implementation** of MSINDO, it imports no
MSINDO source code.  The method is validated out-of-process against a reference
MSINDO build (see [CLAUDE.md § 10](../CLAUDE.md) and
`examples/regression/msindo/`); validated INDO routes and the closed-shell NDDO
scope reproduce reference MSINDO total energies to **well under 1 µHa**.

> **Licensing.** vibe-qc's re-implementation ships under MPL-2.0; the bundled
> MSINDO parameters are redistributed by permission of the Mulliken Center /
> University of Bonn.  See [the licence inventory](../license.md#msindo-indo-semiempirical-method-methodmsindo).

---

## What is MSINDO?

MSINDO belongs to the INDO (Intermediate Neglect of Differential Overlap) family
of semiempirical methods.  Like CNDO and INDO, it invokes the **zero differential
overlap** (ZDO) approximation, neglecting all two-electron integrals that involve
overlap densities `χ_μ(r)χ_ν(r)` where `χ_μ` and `χ_ν` are centred on different
atoms.  All that remains is a one-centre Coulomb/exchange block and a two-centre
monopole `γ_AB`.

The defining feature that sets MSINDO apart from other INDO implementations is
its **symmetric (Löwdin) S⁻¹ᐟ² orthogonalization**, folded into the one-electron
resonance integrals.  This means:

- The basis is **implicitly orthonormal**, the working overlap matrix is **I**
  (the identity), and the Roothaan-Hall equation is the standard eigenvalue
  problem `FC = Cε`.
- The orthogonalization correction (`HORTH`) modifies each atom-pair's resonance
  block, coupling *all* pairs globally.  A small geometry change on one pair
  therefore shifts the orthogonalization of every other pair, the origin of the
  "global Löwdin coupling" that makes a naive per-pair FD gradient fail.
- MSINDO uses **Slater-type orbitals** (not Gaussians) with s, p, and
  **polarization d** functions (up to 9 basis functions per atom: `s, p_x, p_y,
  p_z, d_z², d_xz, d_yz, d_x²−y², d_xy`).  All integrals, overlap, Coulomb γ,
  nuclear attraction, kinetic, are evaluated analytically in the STO basis.

The INDO parametrization covers **H-Xe** (Z = 1-54), with NDDO parameters for
**H, Li-F, Na-Cl**.  Each element carries valence exponents (`MUS`, `MUP`,
`MUD`), frozen-core pseudopotential exponents (`TAU*`) and charges (`FCP*`),
ionisation potentials (`IPOT*`), and orbital-exponent Slater-Condon scaling
(`MUSE`, `MUPE`, `MUDE`).  Closed-shell H-Xe INDO energies are validated against
the MSINDO oracle; open-shell UHF is validated on the documented light-element
and heavy d/p-block fixtures.

### Theory in one paragraph

The SCF loop builds:

1. **Core Hamiltonian H⁰**, per-atom diagonal `ENEG(Z, shell)` plus per-pair
   resonance blocks built from STO overlap, the nuclear attraction
   pseudopotential `V2CORE`, the orthogonalization correction `HORTH`, the
   empirical resonance `DELTAH`, and the local→global rotation `ROTINT`.

2. **Two-electron GMUNU**, one-centre Slater-Condon Coulomb/exchange packed
   into a 9×9 block per atom; two-centre monopole Coulomb γ matrix.

3. **Fock**, `F = H + G(P)`, where the INDO G-matrix builds Coulomb from the
   total density and exchange from the same-spin density (UHF) or half the
   total density (RHF), with the one-centre INDO Coulomb/exchange rules and
   the EINZI d-shell hybrid block.

4. **Energy**, `E_elec = ½ Tr[P(H+F)]`, `E_total = E_elec + Σ_{I<J} CZ_I·CZ_J/R_IJ`,
   `E_binding = E_total − Σ ATENG(Z)`.

### NDDO mode

MSINDO's **default** mode in the reference program is **NDDO** (Neglect of
Diatomic Differential Overlap) rather than INDO.  NDDO retains the two-centre
dipole-monopole and dipole-dipole integrals that INDO discards.  It is a
**separate parametrization** (`nddoparam.f`), the orbital exponents, IPOTs,
and one-centre factors all differ from the INDO set. vibe-qc's validated
closed-shell molecular energy and analytic-gradient scope is H, Li-F, and
Na-Cl. Al through Cl (Z = 13-17) use a 3s3p3d basis with polarization d
functions; the shipped energy and analytic derivative include their source
SPDD terms. Open-shell NDDO remains unsupported and fails closed.

---

## Supported elements

| Range | Z | Elements | Notes |
|---|---|---|---|
| **Main group (INDO)** | 1-20 | H-Ca | H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca |
| **Transition metals (INDO)** | 21-30 | Sc-Zn | 3d valence shell, EINZI d-block |
| **Post-3d main group (INDO)** | 31-36 | Ga-Kr | 4s4p valence + [Ar]3d frozen core |
| **4d / 5s block (INDO)** | 37-48 | Rb-Cd | Rb/Sr, Y-Mo, Tc-Pd trajectory-SCF fixtures, Ag/Cd |
| **5p main group (INDO)** | 49-54 | In-Xe | In/Sn plus Sb-Xe trajectory-SCF fixtures |
| **NDDO, closed-shell molecular** | 1,3-9,11-17 | H, Li-F, Na-Cl | Separate parametrization; RHF energy and analytic gradient, including SPDD terms for Al-Cl |

Elements beyond Xe are not bundled; the engine raises a clean
`NotImplementedError` for any Z outside the supported set.

### Valence shells per element

| Element(s) | Valence | Basis |
|---|---|---|
| H, He | 1s | s only |
| Li, Be | 2s | s/p (2p polarization on hydrogen-bonded H via `PON=1`) |
| B-Ne | 2s2p | s/p |
| Na, Mg | 3s | s/p |
| Al-Ar | 3s3p3d | s/p/d (3d = polarization) |
| K, Ca | 4s | s/p |
| Sc-Zn | 3d4s | s/p/d (3d = valence) |
| Ga-Br | 4s4p + [Ar]3d core | s/p/d (3d frozen core + 4d polarization) |

For Na and heavier elements, the inner shells (1s, then 2s2p, then 3s3p) are
replaced by the `V2CORE` frozen-core pseudopotential, Slater overlaps of the
valence orbitals with the core Slater functions, weighted by `FCP*` charges,
added to the core Hamiltonian diagonal.

---

## Running MSINDO

### Through `run_job` (recommended)

`method="msindo"` plugs into the normal `run_job` flow, no basis set is
required (the Slater basis is built in):

```python
from vibeqc import Atom, Molecule, run_job

# H2S (bohr in Molecule; run_job handles Å/bohr automatically)
h2s = Molecule([
    Atom(16, [0.0, 0.0, 0.0]),
    Atom(1, [1.819, 0.0, 1.751]),
    Atom(1, [-1.819, 0.0, 1.751]),
])

result = run_job(h2s, method="msindo")
print(result.energy)        # total energy (Hartree)
```

`run_job` writes the usual job artefacts (`.out`, `.bibtex`, `.system`,
`.xyz`, etc.) and emits the MSINDO citations automatically.

### Through the direct engine API

For programmatic use, e.g. in a script that scans a potential-energy surface,
computes finite-difference gradients, or drives an external optimizer, call
`run_msindo` directly:

```python
from vibeqc.semiempirical.methods import msindo

r = msindo.run_msindo(
    [16, 1, 1],
    [[0.0, 0.0, 0.0], [0.9627, 0.0, 0.9264], [-0.9627, 0.0, 0.9264]],
)  # coords are Å
print(r.total_energy, r.binding_energy, r.converged, r.n_iter)
```

The direct API returns a `MsindoResult` dataclass:

```python
@dataclass
class MsindoResult:
    total_energy: float      # total energy (Hartree)
    electronic_energy: float # electronic part = ½Tr[P(H+F)]
    binding_energy: float    # total − Σ ATENG
    mo_energies: np.ndarray  # MO eigenvalues (Hartree)
    density: np.ndarray      # density matrix P (n_basis × n_basis)
    n_iter: int              # SCF iterations
    converged: bool          # did the SCF converge?
```

### NDDO mode

Pass `nddo=True` to `run_msindo` for the NDDO (default MSINDO) mode:

```python
r_indo = msindo.run_msindo([9, 1], [[0, 0, 0], [0, 0, 0.917]])
r_nddo = msindo.run_msindo([9, 1], [[0, 0, 0], [0, 0, 0.917]], nddo=True)
print(r_indo.total_energy)   # -24.3089965036 Ha  (INDO)
print(r_nddo.total_energy)   # -23.0436304974 Ha  (NDDO)
```

Closed-shell (RHF) only; H, Li-F, Na-Cl.  Open-shell or out-of-range elements
raise `NotImplementedError`.

---

## Open-shell (UHF)

Pass a `multiplicity` (2S+1) > 1, or a `charge` that makes the valence-electron
count odd, and the engine runs spin-unrestricted MSINDO (two spin densities, a
Coulomb term from the total density and exchange from the same-spin density;
`fockop.f`).  Validated against reference MSINDO `UHF` to < 3 × 10⁻⁸ Ha on OH•,
NO•, the O₂ triplet, CH₃• and NH₂•.

```python
from vibeqc import Atom, Molecule, run_job

# OH radical (doublet)
oh = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.833])], multiplicity=2)
print(run_job(oh, method="msindo").energy)   # -16.3330865643 Ha

# O2 triplet ground state
o2 = Molecule([Atom(8, [0, 0, 0]), Atom(8, [0, 0, 2.283])], multiplicity=3)
print(run_job(o2, method="msindo").energy)   # -31.5123087290 Ha

# Charged system (e.g. H2O+ cation)
h2o_plus = Molecule([Atom(8, [0, 0, 0]),
                     Atom(1, [0, 1.43, 1.11]),
                     Atom(1, [0, -1.43, 1.11])],
                    charge=+1, multiplicity=2)
print(run_job(h2o_plus, method="msindo").energy)
```

The UHF engine includes the s/p/d one-centre Fock terms. Its production claim is
fixture-based: the light s/p radical set and the named heavy trajectory-SCF
fixtures are validated against the oracle, while near-degenerate heavy cases
remain basin-sensitive and are tracked in `handovers/HANDOVER_MSINDO_SCF.md`.

### UHF reduces to RHF for closed-shell

When `multiplicity=1` and `nelec` is even, the engine runs the closed-shell
RHF path, the UHF path is only invoked for genuine open-shell systems.  Setting
`multiplicity=1` on a system with unpaired electrons triggers the UHF path
automatically (the odd electron count selects it).

---

## Periodic: Cyclic Cluster Model (CCM)

MSINDO treats periodic solids and surfaces with the **Cyclic Cluster Model**
(Bredow, Geudtner & Jug, *J. Comput. Chem.* **22**, 89 (2001); Peintinger &
Bredow, *J. Comput. Chem.* **35**, 839 (2014)): a finite cyclic cluster (a
supercell) in which every atom carries a Wigner-Seitz cell describing its
periodic environment, with the long-range electrostatics added by a classical
Ewald (Madelung) embedding.  vibe-qc's implementation lives in
`vibeqc.semiempirical.methods.msindo_ccm` and is validated out-of-process against
reference MSINDO (every number below reproduces the oracle to < 1 µHa).

### Quick-start: bulk MgO

```python
from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

# Bulk MgO - rocksalt Mg4O4 cell (Angstrom), 3 lattice vectors (CCM3D)
a = 4.21
fcc = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
Z = [12] * 4 + [8] * 4
coords = ([[f * a for f in p] for p in fcc]
          + [[(p[0] + 0.5) * a, p[1] * a, p[2] * a] for p in fcc])
cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
print(run_ccm(Z, coords, cell, madelung=True).total_energy)   # ≈ -66.20383741 Ha
```

`run_ccm(atomic_numbers, coords_angstrom, translations_angstrom, *, madelung=)`
takes the cluster geometry and the **1-3 supercell lattice vectors**, one for a
polymer (CCM1D), two for a surface (CCM2D), three for a bulk (CCM3D).
`madelung=True` switches on the Ewald embedding (a 1-D finite sum, the 2-D
surface Ewald, or the 3-D bulk Ewald, chosen by the number of vectors); leave it
off (`NOEWALD`) for non-polar/covalent systems.

The 2-D and 3-D direct and reciprocal sums are truncated by physical Ewald
decay (`exp(-30)`), with coefficient bounds derived from the direct and
reciprocal lattice metrics. Both sums range over sets of *physical* vectors,
`{K != 0 : |K| <= K_cut}` and `{L : |v + L| <= r_cut}`, never over a fixed box
of integer coefficients. Equivalent unimodular choices of the translation vectors
therefore give the same Madelung energy and analytic gradient, even when a
short lattice vector needs large near-cancelling coefficients in one
representation. A pathologically unreduced representation that would require
an unsafe candidate allocation fails closed and asks for a reduced lattice
basis.

The 2-D surface kernel is the Parry Ewald: eq. (7) of Parry, *Surf. Sci.*
**49**, 433 (1975) (erratum *Surf. Sci.* **54**, 195 (1976)), with the `K = 0`
term in the per-pair form of de Leeuw & Perram, *Mol. Phys.* **37**, 1313
(1979), eqs. (8) and (12). Both halves of that split converge absolutely and
exponentially, so truncating each at `exp(-30)` is well defined. The
conditional convergence of a bare Coulomb lattice sum, where the answer depends
on the shape of the summation region, is removed by the Ewald split itself and
is not a property of eq. (7).

### Through `run_job`

The same calculation runs through the standard `run_job` entry point with
`method="ccm"`, passing the lattice vectors in a `CCMOptions`.  This is the
recommended path for a one-shot energy: it writes the usual job artefacts
(`.out`, `.system`, …) and routes the periodic-CCM citations (Peintinger &
Bredow 2014; Bredow, Geudtner & Jug 2001) into the `.out` / `.bibtex` /
`.references` **and** the `.system` manifest's `[citations]` provenance block.

```python
from vibeqc import Atom, Molecule, run_job
from vibeqc.molecule import ANGSTROM_TO_BOHR as A2B
from vibeqc.semiempirical.methods.msindo_ccm import CCMOptions

a = 4.21
fcc = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
mg = [Atom(12, [f * a * A2B for f in p]) for p in fcc]
ox = [Atom(8, [(p[0] + 0.5) * a * A2B, p[1] * a * A2B, p[2] * a * A2B])
      for p in fcc]
mgo = Molecule(mg + ox)                       # Atom coords are bohr
cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]      # CCMOptions lattice is Angstrom

res = run_job(mgo, method="ccm",
              ccm_options=CCMOptions(translations=cell, madelung=True))
print(res.energy)   # ≈ -66.20383741 Ha
```

Through `run_job`, CCM is **closed-shell (RHF) single-point only**.  Each
boundary is a clean error, never a silent gap (CLAUDE.md §7): an open-shell
cell (`multiplicity != 1`) or `optimize=True` raises `NotImplementedError`
(use `ccm_optimize`, below, for relaxation), a call without `ccm_options`
raises `ValueError`, and an element beyond the engine's scope (H-Xe) raises
`NotImplementedError` naming the supported set.

The public reaction-path driver is a separate validated workflow:
`run_neb(..., method="seccm", ccm_options=...)` uses the analytic cyclic
gradient for each image. It accepts molecular endpoints carrying an explicit
cyclic cell, or `PeriodicSystem` endpoints when the `CCMOptions.translations`
match the active lattice vectors exactly. The latter can be written directly
as a periodic reaction-path QVF. The `run_neb(method="seccm")` driver remains
closed-shell MSINDO only. DFTB0 also has a narrowly gated 1-D H/C
reaction-path route through `run_neb(method="dftb0", seccm_topology=...)`.
The other semiempirical Hamiltonians expose experimental direct SECCM
adapters through `run_scc_dftb_seccm`, `run_pm6_seccm`, `run_omx_seccm`,
and `run_gfn2_seccm`; see {doc}`semiempirical`.

### Valid cyclic clusters

**The cluster must be a valid cyclic cluster**, a supercell with no atom lying
on a Wigner-Seitz cell face, so each other atom appears exactly once in every
atom's WS cell (`round(Σ WS weight) == NATOMS − 1`).  An invalid cluster raises
`ValueError`; an odd-`N` evenly-spaced cell is the most robust shape.  Madelung
runs additionally require an **electroneutral** cell.

The cluster shape must also satisfy stoichiometric constraints for the Ewald
embedding, the total charge of the cell must be zero, and the atoms should
carry chemically sensible charges (the Madelung potential is built from the
SCF net charges).

### Surfaces, adsorption and relaxation

A 2-D slab (two lattice vectors) gives a surface; adding adsorbate atoms to the
cell models a periodic adlayer.  `ccm_gradient_fd` returns the nuclear gradient
(Ha/bohr) and `ccm_optimize` relaxes selected atoms (e.g. an adsorbate on a
frozen slab):

```python
from vibeqc.semiempirical.methods.msindo_ccm import ccm_optimize

# relax the adsorbate atoms, holding the slab (indices 0..7) fixed
relaxed_coords, result = ccm_optimize(
    Z, coords, cell, madelung=True, frozen=list(range(8)))
```

The adsorption energy is `E[slab+adsorbate] − E[slab] − E[adsorbate(gas)]`.  For
H₂O on MgO(100) this is −6.7 kcal/mol at fixed geometry and −8.2 kcal/mol after
relaxation.  A full worked example, bulk → surface → fixed/relaxed adsorption, 
is in [`examples/semiempirical/11_msindo_ccm.py`](../../examples/semiempirical/11_msindo_ccm.py).

> CCM runs both through `run_job(method="ccm", …)` (above, single-point
> energies with the full citation/manifest surface) and through the lower-level
> `msindo_ccm` Python API (gradients, relaxation, energy decomposition).
> Reaction paths run through `run_neb(method="seccm", ccm_options=...)` and
> use the analytic fixed-topology gradient.
> Adsorption examples target the ionic oxides MgO and Al₂O₃, INDO/CCM is not
> parametrized for metallic bonding, so metal surfaces are out of method scope.

### CCM gradient details

CCM checks restricted orbital stability when the initial core-Hamiltonian
occupied/virtual gap is at most `1e-6` Ha. In this domain, tiny coordinate
changes can rotate the initial occupied projector and send DIIS to different
stationary states even when their final orbital gaps are large. A negative
orbital-rotation curvature identifies a saddle: the driver searches both
signs of that mode, reconverges lower-energy seeds, and checks the selected
state again. The density used for analytic and finite-difference gradients
follows the same selection. The Hamiltonian and integer occupations are
unchanged.

`max_iter` bounds the cumulative SCF iterations across these restarts. An
unresolved stability analysis, an exhausted restart budget, or failure to
escape a detected saddle raises a diagnostic error. The result records
`stability_checked`, `stability_analysis_converged`, `stability_eigenvalue`
(the energy Hessian eigenvalue), and `n_stability_restarts`; `run_job` also
prints the verdict and cites the stability algorithm. This is a local
minimum check within restricted HF, not a global-minimum guarantee or a
test against unrestricted spin solutions. Initial guesses with a resolved
gap keep their established SCF path. See [Lehtola et al. (2020), section
10](https://doi.org/10.3390/molecules25051218) and issue #249.

`ccm_gradient_fd` computes the nuclear gradient by central finite differences
(Ha/bohr), holding the **Wigner-Seitz topology fixed** while displacing atoms.
This matches MSINDO's fixed-weight analytic gradient.  The WS is built once
before the gradient loop; within each displacement step, the image positions
follow the displaced central atoms but the origin labels and weights stay
constant.  This avoids the ~1 × 10⁻² Ha/bohr discontinuity that a naive
rebuild-the-WS-per-displacement FD would produce at WS face boundaries.

`ccm_gradient_analytic`
(`vibeqc.semiempirical.methods.msindo_ccm_gradient_analytic`) returns the
**analytic** CCM nuclear gradient, the exact derivative of the CCM total
energy, including the analytic 1-D (finite ±T,±2T lattice sum), 2-D
(Parry-Heyes slab Ewald) and 3-D (Ewald bulk) Madelung gradient. It reproduces
the FD gradient to < 1 × 10⁻⁶ Ha/bohr and the MSINDO oracle to < 2 × 10⁻⁵ on
mirror-symmetric Wigner-Seitz cells. (On *asymmetric* ionic Madelung cells the
oracle's own central-atom-only assembly deviates from the FD of its own energy
by up to ~2 × 10⁻² Ha/bohr; the vibe-qc gradient stays exact.) Closed-shell
(RHF) only. For H-Kr, both `ccm_gradient_analytic` and `ccm_gradient_fd` use
the C++ bindings and match the Python reference to ~10⁻¹⁰ Ha/bohr. Rb-Xe
uses the deterministic Python reference, as does an installation without the
native binding. Once a native calculation is selected, an exception or
nonconverged result propagates rather than falling back to Python.

---

## Gradients, geometry optimization, and transition states

### Molecular gradients

The finite-difference gradient `msindo_gradient_fd` remains available as a
diagnostic reference:

```python
from vibeqc.semiempirical.methods.msindo import msindo_gradient_fd

Z = [8, 1, 1]
xyz = [[0, 0, 0], [0, 0.757, -0.467], [0, -0.757, -0.467]]  # Å
g = msindo_gradient_fd(Z, xyz)  # Ha/bohr, shape (n_atoms, 3)
```

The production closed-shell analytic gradient `msindo_gradient_analytic` uses
the C++ `m_indo.gradient_analytic` kernel for INDO and the complete NDDO
multipole contraction, including the Al-Cl SPDD terms. Both differentiate the
energy actually returned by their route and match a central finite-difference
step ladder to better than 1e-6 Ha/bohr. `run_job(..., method="msindo",
nddo=True)` exposes the NDDO gradient lazily for H, Li-F, and Na-Cl. Open-shell
NDDO gradients fail closed.

### Geometry optimization

`msindo_optimize` wraps scipy's L-BFGS-B:

```python
from vibeqc.semiempirical.methods.msindo import msindo_optimize

result = msindo_optimize(Z, xyz_start, fmax=4.5e-4, max_iter=200)
print(result.energy, result.coords_angstrom, result.n_iter)
```

Returns `MsindoOptimizeResult` with `energy`, `coords_angstrom`, `n_iter`,
`fmax`, `converged`, and `history` (list of `(energy, grad_max, coords)` per
step).

### Nudged Elastic Band (NEB)

MSINDO is the first semiempirical backend for vibe-qc's NEB driver:

```python
from vibeqc import run_neb

# NH3 umbrella inversion: reactant + product = pyramidal NH3 + mirror image
result = run_neb(reactant_mol, product_mol, method="msindo",
                 n_images=7, climbing_image=True)
ts = result.transition_state_index
barrier_kcal = (result.energies[ts] - result.energies[0]) * 627.509474
print(f"{barrier_kcal:.2f}")   # ≈ 8 kcal/mol
```

Each image cost is 6N SCF calls per outer iteration (FD gradient), so NEB
with MSINDO is practical for up to ~10-15 atoms.  Band evaluation is
parallelised across images.  The TS at the climbing image is a genuine
first-order saddle (validated by FD Hessian: exactly one imaginary mode).
Full example: `examples/semiempirical/22_msindo_neb.py`.

---

## Excited states (CIS)

MSINDO excited states use vibe-qc's **general** CIS / Tamm-Dancoff kernel
(`vibeqc.excited`) over the INDO integral set, the same reference-agnostic code
that HF and DFT use, fed the INDO `ERIProvider` blocks instead of libint ones.
`msindo_cis` returns vertical singlet or triplet excitation energies and the CIS
amplitudes:

```python
from vibeqc.semiempirical.methods.msindo import msindo_cis

Z = [8, 1, 1]
xyz = [[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]]   # Å
s = msindo_cis(Z, xyz, spin="singlet", n_states=4)
print(s.excitation_energies_ev[0])     # S1 (eV)
```

> **Rigorous vs. scaled CIS.** This is the *rigorous* INDO CIS. Reference MSINDO
> additionally applies empirical CIS scaling knobs (`SCALEDCIS`, `UJCORR` /
> `CISCORR`) for spectroscopic fitting, so its excitation energies differ from
> the unscaled values here. vibe-qc ships the rigorous variant by design.

### Excited-state gradients

A CIS excited-state *total* energy is `E(R) = E_SCF(R) + ω_state(R)`, so its
nuclear gradient is obtained by central-differencing that total energy.
`msindo_cis_gradient_fd` does this with **state tracking**, at each displaced
geometry it follows the CIS root whose amplitude vector overlaps the reference
state's, so the gradient stays on one state even where roots reorder
(`vibeqc.excited_gradient`):

```python
from vibeqc.semiempirical.methods.msindo import msindo_cis_gradient_fd

# Gradient (Ha/bohr) of the first singlet excited state (S1).
g = msindo_cis_gradient_fd(Z, xyz, state=1, spin="singlet")

# state=0 is the ground state - identical to msindo_gradient_fd.
```

The state index is spectroscopic: `0` = ground state, `k ≥ 1` = the `k`-th CIS
root. The same machinery is reference-agnostic, 
`vibeqc.excited_gradient.make_hf_cis_energy_fn` gives the identical gradients for
a Hartree-Fock reference.

#### Analytic CIS gradient (CPHF / Z-vector)

`vibeqc.semiempirical.methods.msindo_cis.cis_gradient` computes the **analytic**
excited-state nuclear gradient via the CPHF Z-vector relaxed density and the CIS
two-particle density, contracting the per-pair integral derivatives, the same
engine as the ground-state analytic gradient, plus the orbital-relaxation
response (`A·Z = −L`):

```python
from vibeqc.semiempirical.methods.msindo_cis import cis_gradient, run_cis

cis = run_cis(Z, xyz, spin="singlet", n_states=4)
g = cis_gradient(Z, xyz, cis, state=0, spin="singlet")   # S1 gradient (Ha/bohr)
```

It reproduces `cis_gradient_fd` to ~1e-6 Ha/bohr for both singlet and triplet
**non-degenerate** states (closed-shell only; open-shell raises). At or near a
state degeneracy the per-state CIS gradient is mathematically ill-defined, so
**`msindo_cis_gradient_fd` remains the documented, robust fallback**, and is
what the state-tracking / conical-intersection paths above use, since they must
operate where roots cross. Reference for the analytic CIS gradient: Foresman,
Head-Gordon, Pople & Frisch, *J. Phys. Chem.* **96**, 135 (1992).

### Conical intersections (MECI)

A minimum-energy conical intersection (MECI) is the lowest-energy geometry where
two states are degenerate. `msindo_meci` optimizes one with the method-agnostic
penalty-function optimizer in `vibeqc.conical`, fed the INDO finite-difference
CIS gradients:

```python
from vibeqc.semiempirical.methods.msindo import msindo_meci

# Optimize the S1/S0 minimum-energy crossing point.
res = msindo_meci(Z, xyz, lower_state=0, upper_state=1, spin="singlet")
print(res.gap_ev, res.converged)        # state gap (eV) → ~0 at the MECI
print(res.coords_angstrom)              # the MECI geometry
```

The optimizer uses the **Levine-Coe-Martínez** penalty
`F_σ = ½(E_I+E_J) + σ·ΔE²/(ΔE+α)`, escalating σ until the gap closes. It needs
only the two state energies and their gradients, **no nonadiabatic derivative
coupling vector**, which is exactly what the finite-difference CIS gradients
supply. The upper/lower roles are assigned by energy at each step, so the
objective stays continuous through a state crossing, and a trial step into a
non-convergent region is turned into a smooth barrier that retreats rather than
crashing the run.

> This validates the *optimizer mechanics* on INDO states, not reference-MSINDO
> CIS energies (which are empirically scaled, see the note above). The
> optimizer itself is reference-agnostic: any two-state `(E, ∇E)` callback
> (`vibeqc.conical.make_cis_meci_fn` from an HF or MSINDO CIS energy function)
> drives `vibeqc.conical.optimize_conical_intersection`.

A runnable end-to-end demonstration (excited-state gradient + S1/S0 MECI search
on H₂O) is `examples/semiempirical/26_msindo_conical.py`.

---

## Configuration interaction (CISD)

`msindo_cisd` runs **variational CISD**, the Hamiltonian diagonalised in the
space of all single and double substitutions out of the closed-shell INDO
reference. It does this by handing the INDO MO integrals to vibe-qc's
**reference-agnostic CI solver** (`vibeqc.solvers.cisd`): the converged INDO
reference's one-electron core Hamiltonian and the `_indo_ao_eri` two-electron
tensor are transformed to the MO basis, then the same validated Slater-Condon
engine that powers vibe-qc's FCI/CASCI builds and diagonalises the CISD matrix.
It is **not** a port of MSINDO's own selecting determinant-CI driver
(`rhfcisd.f`), there are no empirical CIS scalings and no determinant selection.

```python
from vibeqc.semiempirical.methods.msindo import msindo_cisd

r = msindo_cisd([8, 1, 1],
                [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])  # H2O, Å
print(r.e_total)            # CISD total energy (Ha)
print(r.e_corr)             # correlation energy (E_CISD − E_SCF)
print(r.reference_weight)   # squared CI weight of the reference determinant
for desc, coeff, weight in r.dominant_configurations(4):
    print(f"{desc:24s} c={coeff:+.4f}  w={weight:.4f}")
```

`frozen_core=k` freezes the lowest `k` MOs out of the correlation; `nroots=n`
reports the lowest `n` CI roots (`r.ci.e_totals`); `output="stem"` writes the
`.bibtex` / `.references` citing MSINDO + the variational-CISD method.

Because the singles-plus-doubles space is *exact* for two correlated electrons,
MSINDO CISD on a two-electron system reproduces the INDO **FCI** to machine
precision; in general `E_FCI ≤ E_CISD ≤ E_SCF`, and the singles-only subspace
(`vibeqc.solvers.cisd(..., max_excitation=1)`) reproduces the `msindo_cis`
singlet+triplet excitation structure. These are the validation anchors (vs
vibe-qc's own FCI/CIS), since reference-MSINDO's CISD uses empirical scalings and
a selecting driver and is therefore not a parity target.

> The determinant-space size grows as `(n_occ·n_vir)²`, so CISD is intended for
> small molecules; `max_det` (default 20000) guards against an oversized space.

---

## Post-SCF correlation

### MP2 (closed-shell)

`msindo_mp2` runs canonical MP2 over the converged INDO (or NDDO) MOs:

```python
from vibeqc.semiempirical.methods.msindo import msindo_mp2

r = msindo_mp2([8, 1, 1],
               [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])  # H2O
print(r.e_mp2, r.e_corr, r.e_total)   # MP2 correction (Ha), correlation, total

# SCS-MP2 and SOS-MP2 scaling
r_scs = msindo_mp2(..., variant="scs")
r_sos = msindo_mp2(..., variant="sos")
```

The MP2 kernel uses the general `vibeqc.correlation.mp2_energy` with the INDO
ERI provider.  NDDO-MP2 (`msindo_mp2(nddo=True)`) is also supported and
validated sub-µHa vs the oracle, the NDDO multipole integrals enter only
through the converged MOs, not the correlation integral tensor.

Through `run_job`: `method="msindo"`, `correlation="mp2"` (or `"scs-mp2"`,
`"sos-mp2"`).  The result carries a `.mp2` attribute.

### UMP2 (open-shell)

```python
from vibeqc.semiempirical.methods.msindo import msindo_ump2

# O2 triplet
r = msindo_ump2([8, 8], [[0, 0, 0], [0, 0, 2.283]], multiplicity=3)
print(r.e_corr, r.e_total, r.e_corr_aa, r.e_corr_bb, r.e_corr_ab)
```

The αα channel is the correct 1.0-weighted version, reference MSINDO's
`mp2uhf.f` line 62 carries a spurious ¼ (the ββ channel at line 77 uses the
correct factor 1.0, and the routine's own formula header confirms 1.0), so
the vibe-qc UMP2 ships the corrected formulation.

---

## Ionization potentials and electron affinities (GF2 / OVGF)

`msindo_ovgf` computes **quasiparticle** ionization potentials and electron
affinities from the one-particle Green's function, the diagonal second-order
self-energy (GF2), solved self-consistently through the Dyson equation, over the
*rigorous* INDO integral set (the same tensor that drives MSINDO-MP2). It
corrects the Koopmans (orbital-energy) estimate for relaxation and correlation.

```python
import numpy as np
from vibeqc.semiempirical.methods.msindo import msindo_ovgf

xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]   # H2O (Å)
r = msindo_ovgf([8, 1, 1], xyz)
print(r.ip_homo_ev)     # HOMO ionization potential, eV
print(r.ea_lumo_ev)     # LUMO electron affinity, eV (negative: no bound anion)
```

* **Electron affinities** are surfaced as `EA = −ε_qp(LUMO)`. In a minimal
  valence basis (and any non-augmented basis) the diagonal-GF2 EA is **qualitative
  only**, the virtual orbitals are not variational and there are no diffuse
  functions for a bound anion, so a positive value does *not* establish a stable
  anion. Use an augmented Gaussian basis (HF/DFT reference) for quantitative EAs.

* **Open shell.** Pass a `multiplicity > 1`, or an odd valence-electron count,
  and `msindo_ovgf` runs the spin-unrestricted (UHF) reference and returns a
  `MsindoUOVGFResult` with the **spin-resolved** self-energy (the
  electron-propagator analogue of UMP2): per-spin frontier orbitals, the first
  vertical IP (`first_ip_ev`), and the EA (`ea_ev`). s/p elements (H-F) only.

  ```python
  d = 1.079
  ch3 = [[0, 0, 0]] + [[d*np.cos(a), d*np.sin(a), 0]
                       for a in (0.0, 2*np.pi/3, 4*np.pi/3)]
  r = msindo_ovgf([6, 1, 1, 1], ch3)        # CH3• doublet (inferred)
  print(r.first_ip_ev)                      # ~10.6 eV (SOMO ionization)
  ```

* **Renormalized GF2.** Bare GF2 systematically *over*-corrects the IP. Pass
  `renormalize=True` to dress the second-order self-energy with the full diagonal
  **third-order** self-energy and a geometric screening, which trims the
  overcorrection back toward experiment (H₂O HOMO IP: bare 13.13 → renorm 12.87
  eV). Small systems only (the third-order contractions scale steeply).

  ```python
  msindo_ovgf([8, 1, 1], xyz, renormalize=True).ip_homo_ev   # 12.87 eV
  ```

> **What vibe-qc ships, precisely.** This is the *rigorous* ZDO self-energy and a
> transparently-documented *renormalized second-order Green's function*. It is
> **not** reference MSINDO's `ovgfrhf_neu` (a cruder *factorized*-integral
> approximation), nor the literal von Niessen-Schirmer-Cederbaum OVGF "A/B/C"
> (whose exact screening partition is in the 1984 review). Cederbaum 1975 / von
> Niessen 1984 are cited as the lineage. The general engine lives in
> `vibeqc.propagator` and serves HF/DFT (`method="ovgf"`) the same way.

---

## Implicit solvation (COSMO)

MSINDO solvation reuses vibe-qc's reference-pluggable CPCM/COSMO engine
(`vibeqc.solvation`): the cavity, the segment interaction matrix, the dielectric
screening, the apparent-surface-charge solve, and the polarisation energy are all
shared with the HF/DFT solvation path. Only the **solute↔cavity coupling** is
MSINDO-specific, and it is a *distributed-multipole* coupling, two-centre Slater
penetration integrals `⟨μ(A)|1/|r−sᵢ||ν(A)⟩` (the `V2INT` integrals), with the
solute charge vector built from the core charges, the diagonal AO populations
`P_μμ`, and the same-atom `2·P_μν` p-p / d-d pairs. (A point-charge "monopole"
shortcut over-screens by ~17× because a point ESP is far too singular at the
near-surface cavity points; the distributed multipole is required.)

The screening uses the Klamt-Schüürmann conductor factor
`f(ε) = (ε−1)/(ε+0.5)`, and the reaction field is folded into the SCF directly,
so the density and the surface charge converge together in a single SCF (usually
one extra cycle over the gas phase):

```python
from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo

Z   = [8, 1, 1]
xyz = [(0.0, 0.0, 0.0), (0.0, 0.757, 0.587), (0.0, -0.757, 0.587)]  # Å

r = msindo_cosmo(Z, xyz, epsilon=78.39)        # water
print(r.total_energy)   # in-solvent total (Ha)
print(r.e_solv)         # solvation energy = total − gas (Ha)
```

It is closed-shell (RHF) for the validated H-Xe INDO set; `epsilon` must be > 1.

The same calculation runs through the standard `run_job` entry point with
`solvent=` (the GEPOL cavity is used; the in-solvent total becomes
`result.energy`, with `result.e_solv` the solvation energy):

```python
from vibeqc import Atom, Molecule, run_job

h2o = Molecule([Atom(8, [0, 0, 0]),
                Atom(1, [0, 1.43, 1.11]), Atom(1, [0, -1.43, 1.11])])  # bohr
r = run_job(h2o, method="msindo", solvent="water")   # or solvent=78.39
print(r.energy, r.e_solv)
```

`solvent="vacuum"` (ε ≤ 1) short-circuits to the gas-phase run; open-shell +
solvent raises (COSMO is RHF-only).

### Cavity (`cavity=`)

* **`cavity="gepol"` (default), exact oracle parity.** MSINDO's own GEPOL/SAS
  tessellation: a pentakisdodecahedron (60 triangular faces) around each atom at
  the COSMO `COSMOR` radius, refined for segment centres + areas, with the
  Klamt self-energy `A_ii = 1.07/√area` and a `1/r` off-diagonal capped at
  `½·min(A_ii, A_jj)`. This reproduces reference MSINDO to **~2 × 10⁻⁹ Ha**
  (H₂O ε=78.39: total −17.0264718 vs oracle −17.0264718; H₂O / HF / CH₄ / H₂S
  all ≤ 2e-9). vibe-qc builds each atom's cavity independently, matching the
  oracle's `NOSYM` run; the reference's *default* symmetry path transforms
  symmetry-equivalent atoms' segments and can differ by up to ~10 µHa.
* **`cavity="lebedev"`, the shared CPCM cavity.** vibe-qc's Lebedev-on-Bondi
  cavity (the one HF/DFT CPCM uses). The multipole coupling is identical, on
  the *same* cavity it agrees with the Gaussian ESP coupling to ~0.1 mHa, but
  the absolute E_solv carries that cavity convention: H₂O ε=78.39 gives −6.83
  vs the oracle's −8.26 mHa (the larger Bondi×1.2 cavity under-screens). The
  `radii`, `radii_scale`, `solvent_probe_radius_ang`, and `n_points_per_sphere`
  arguments tune this cavity.

---

## Molecular dynamics

MSINDO drives vibe-qc's general molecular-dynamics engine (`vibeqc.md`):

```python
from vibeqc.semiempirical.methods.msindo import run_msindo_md

traj = run_msindo_md(Z, coords_ang, dt_fs=0.5, n_steps=500, T_K=300,
                     thermostat="berendsen")
print(traj.positions.shape)     # (n_steps, n_atoms, 3)
print(traj.energies)           # total energies per step (Ha)
```

Available thermostats: NVE, Berendsen, Nosé-Hoover NVT.  Velocity-Verlet
integration, Maxwell-Boltzmann initialisation, COM removal.

### Well-tempered metadynamics

```python
from vibeqc.semiempirical.methods.msindo import run_msindo_metadynamics
from vibeqc.metadynamics import DistanceCV

traj, bias = run_msindo_metadynamics(
    Z, coords_ang, collective_variables=[DistanceCV(0, 1)],  # bond distance
    dt_fs=0.5, n_steps=2000, T_K=300,
)
```

Free energy is recovered as `F = −γ/(γ−1)·V_bias` (well-tempered ensemble).
Available CVs: `DistanceCV`, `AngleCV` (with analytic Cartesian gradients).
Full examples: `examples/semiempirical/24_msindo_md.py` and
`25_msindo_metadynamics.py`.

---

## SCF convergence

The closed-shell SCF is accelerated with **Pulay DIIS**, the error vector is the
commutator `[F, P]` in the symmetrically (Löwdin) orthogonalised basis, where the
overlap is the identity.  Convergence is tested on the residual ‖[F, P]‖ rather
than on the energy step: under DIIS a small extrapolation step is *not* a
converged density, so a step-size test would stop early at the wrong point.

Parameters (pass to `run_msindo`):

| Parameter | Default | Meaning |
|---|---|---|
| `max_iter` | 200 | Maximum SCF iterations |
| `conv_tol` | 1e-9 | Convergence threshold on ‖[F,P]‖ (Ha) |
| `diis_start` | 2 | DIIS activated after this many iterations |
| `diis_nvec` | 6 | Number of DIIS error vectors stored |

A bare fixed-point iteration oscillates indefinitely on floppy, realistically
sized molecules. The primary DIIS trajectory converges small molecules in a
handful of iterations and a 27-atom alcohol (C₈H₁₈O, 54 basis functions) in
~25. Reported aggregate work also includes any root-selection probe and strict
refinement described below. Because both backends converge to the same
stationary point on the true residual, the C++ and Python engines agree to
**machine precision** (< 10⁻¹³ Ha) on every reference molecule.

For closed-shell H-Ar INDO single points and ground-state analytic gradients,
Hcore/DIIS remains the primary trajectory. The solver also evaluates the
method's extended-Hückel start with WICHT iteration as a finite alternate-basin
probe. The probe retains the established 2000-cycle recovery floor even when
primary DIIS converges, so merely crossing the caller's strict-DIIS iteration
cap cannot change which basin is considered. Every converged probe density is
restarted under strict commutator DIIS before the stationary energies are
compared. The alternate density is accepted only when the strict solve
converges below the primary energy by more than the requested convergence
resolution. The WICHT energy-only result is never reported directly. Acetamide
exercises the lower-basin selection; cyclopropene exercises rejection of a
higher alternate basin. Reported iteration counts include every attempted
trajectory. Converged jobs outside the validated H-Ar scope retain their
established path without a proactive probe. MP2, OVGF, CIS/CISD, and
CIS-gradient reference-SCF routes are unchanged by this single-point fix.

If primary DIIS exhausts its iteration limit, the same WICHT-to-strict sequence
retains its extended recovery budget, including outside the proactive H-Ar
scope. This recovery is exercised by the exact archived Thiel tetrazine
geometry, whose ordinary DIIS trajectory otherwise remains trapped in a cycle.
NDDO and open-shell jobs retain their existing SCF routes. The validated
heavy-element trajectory set continues to use the
MSINDO-faithful WICHT energy path directly because strict DIIS changes its
intended oracle-matched occupations. Their analytic-gradient SCF route remains
the pre-existing strict stationary DIIS solve; this change does not reinterpret
nonstationary WICHT densities as variational analytic-gradient densities.

---

## Validation

`examples/semiempirical/10_msindo_validate.py` runs the full reference set and
prints the deviation from reference MSINDO:

```
molecule     vibe-qc total         reference      Δ (Ha)  conv
H2           -1.1732166873     -1.1732166872   -1.40e-10  OK
HF          -24.3089965039    -24.3089965036   -2.94e-10  OK
N2          -19.2858118281    -19.2858118276   -4.56e-10  OK
CO          -21.6961833926    -21.6961833869   -5.75e-09  OK
CH4          -8.2956064184     -8.2956064178   -5.83e-10  OK
H2O         -17.0182087695    -17.0182087674   -2.06e-09  OK
H2S         -11.2354667006    -11.2354666980   -2.63e-09  OK   (3d)
SiH4         -6.3561206300     -6.3561206258   -4.16e-09  OK   (3d)
PH3          -8.3298767931     -8.3298767745   -1.86e-08  OK   (3d)
AlCl3       -44.2269575574    -44.2269575516   -5.77e-09  OK   (3d)
OH          -16.3330865712    -16.3330865643   -6.85e-09  OK   (UHF doublet)
NO          -25.3584589711    -25.3584589422   -2.89e-08  OK   (UHF doublet)
O2          -31.5123087277    -31.5123087290   +1.32e-09  OK   (UHF triplet)
CH3          -7.6239129200     -7.6239129136   -6.39e-09  OK   (UHF doublet)
NH2         -10.7659659754    -10.7659659674   -8.01e-09  OK   (UHF doublet)
```

The regression harness (`examples/regression/msindo/`) can rebuild the reference
energies from source (`build_oracle.sh`) and re-run the comparison
(`runner_msindo.py`), vibe-qc never imports MSINDO; it executes a reference
binary out-of-process and parses its output (CLAUDE.md § 10).

---

## Atomic charges

Because MSINDO works in an orthogonal basis, Mulliken/Löwdin charges coincide:
`q_A = CZ_A − Σ_{μ∈A} P_μμ`.  For H₂S the engine gives **S −0.25, H +0.12**, 
the expected slight polarity.  See the validation example for the snippet.

In the direct API, the charge per atom is:

```python
blocks, _ = msindo._atom_blocks(Z)
for i_atom, (lo, hi) in enumerate(blocks):
    pop = float(res.density.diagonal()[lo:hi].sum())
    q = msindo.eff_core_charge(Z[i_atom]) - pop
    print(f"Atom {i_atom} (Z={Z[i_atom]}): charge = {q:+.3f}")
```

---

## Visualization

A MSINDO result (geometry + atomic charges) can be packaged into a `.qvf`
archive and explored in **vibe-view**.  See
`examples/vibe_view/showcase_msindo_h2s.py` for an end-to-end script that
computes H₂S with MSINDO, writes a `.qvf`, and renders a structure-with-charges
image.

---

## Feature matrix

| Capability | Status |
|---|---|
| **Closed-shell (RHF)** | ✅ INDO (H-Xe); NDDO H, Li-F, Na-Cl |
| **Open-shell (UHF)** | ✅ INDO UHF for validated light-element and heavy d/p-block fixtures |
| **Elements** | H-Xe (Z 1-54): full INDO parameter scope |
| **s / p shells** | ✅ Python **and** C++ backend |
| **d shell** (polarization and transition-metal valence) | ✅ Python **and** C++ backend for validated INDO routes; closed-shell NDDO SPDD for Al-Cl |
| **Periodic (CCM)** | ✅ 1-D/2-D/3-D, Ewald Madelung, gradients + relaxation |
| **CCM via `run_job`** | ✅ `method="ccm"` with `CCMOptions` |
| **SECCM reaction paths** | ✅ closed-shell MSINDO via `run_neb(method="seccm", ccm_options=...)` |
| **Molecular gradients** | ✅ finite difference; closed-shell INDO analytic gradients via C++; closed-shell NDDO analytic gradients via Python |
| **Geometry optimization** | ✅ L-BFGS-B via `msindo_optimize`; finite-difference gradients for INDO and NDDO |
| **Transition states (NEB)** | ✅ `run_neb(method="msindo")` |
| **Excited states (CIS / TDA)** | ✅ singlet/triplet |
| **Excited-state gradients** | ✅ finite-difference, state-tracked |
| **Conical intersections (MECI)** | ✅ penalty-function optimizer |
| **Implicit solvation (COSMO)** | ✅ GEPOL (exact oracle parity) + Lebedev cavity |
| **MP2 (closed-shell)** | ✅ INDO + NDDO references; SCS/SOS scaling |
| **UMP2 (open-shell)** | ✅ spin-resolved αα/ββ/αβ |
| **Variational CISD** | ✅ generic CI over INDO MOs |
| **Quasiparticle IPs/EAs (GF2/OVGF)** | ✅ closed + open shell, renormalized |
| **Molecular dynamics** | ✅ NVE + Berendsen + Nosé-Hoover NVT |
| **Metadynamics** | ✅ well-tempered, DistanceCV + AngleCV |
| **Atomization energies** | ✅ `run_job(atomization=True)` |
| ROHF | ⛔ raises `NotImplementedError` |
| Cs and heavier (Z > 54) | ⛔ not parametrized |
| Open-shell analytic molecular gradients | ⛔ finite-difference fallback only |
| Open-shell d-block UHF | 🔶 validated on named fixtures; not a blanket route guarantee |
| Open-shell COSMO | ⛔ RHF-only |

---

## Comparison with the original MSINDO manual

For users familiar with the Fortran MSINDO program, here is the mapping of
the original keyword-driven input to vibe-qc's Python API:

### Wavefunction keywords

| Original MSINDO keyword | vibe-qc Python equivalent |
|---|---|
| `RHF` | Default (neutral, closed-shell) |
| `UHF` | `multiplicity > 1` in `Molecule` or `run_msindo(..., multiplicity=n)` |
| `ROHF` | `NotImplementedError` |
| `NDDO` | `run_msindo(..., nddo=True)` |
| `MULTIP n` | `Molecule(..., multiplicity=n)` or `run_msindo(..., multiplicity=n)` |
| `CHARGE n` | `Molecule(..., charge=n)` or `run_msindo(..., charge=n)` |
| `DFTD3` | `run_job(..., dispersion_params=D3Params(...))` (D3-BJ from vibe-qc's D3 engine) |

### Geometry

| Original input | vibe-qc equivalent |
|---|---|
| `CARTES` (Cartesian geometry) | The default, coords are Cartesian Angstrom |
| `AN x y z` | `[z, [x, y, z]]` in the coordinate list |
| Internal (Z-matrix) coordinates | Not supported; convert to Cartesian first |

### CCM

| Original keyword | vibe-qc equivalent |
|---|---|
| `CCM3D VECTA(i,j) VECTB(k,l) VECTC(m,n)` | `run_ccm(Z, coords, [vec_a, vec_b, vec_c], madelung=True)` |
| `CCM2D VECTA(i,j) VECTB(k,l)` | `run_ccm(Z, coords, [vec_a, vec_b], madelung=True)` |
| `CCM1D VECTA(i,j)` | `run_ccm(Z, coords, [vec_a], madelung=True)` |
| `NOEWALD` | `run_ccm(..., madelung=False)` |
| `WEIGHTFCT n` | Handled internally (uses the MSINDO default step function) |
| Dummy atom `XX` for VECT endpoints | Not needed, vectors are passed directly as coordinates |
| `CARTOPT` | `ccm_optimize(...)` for relaxation |
| `FULLOPT` / `LATTICEOPT` | Not yet implemented |

### SCF control

| Original keyword | vibe-qc equivalent |
|---|---|
| `DELEN n` | `run_msindo(..., conv_tol=n)` |
| `NAV` | SCF acceleration is selected internally; no user-facing damping/DIIS control |
| `IDEN n` | Closed-shell H-Ar `run_msindo`: Hcore/DIIS primary plus an internal extended-Hückel/WICHT basin probe; other routes unchanged |
| `MAXCYC n` | `run_msindo(..., max_iter=n)`; root selection can use the internal WICHT recovery budget |

### Solvation

| Original keyword | vibe-qc equivalent |
|---|---|
| `COSMO DIELEC n` | `msindo_cosmo(..., epsilon=n)` or `run_job(..., solvent=n)` |
| `SOLVENT WATER` | `run_job(..., solvent="water")` or `solvent=78.39` |
| `NOSYM` | Default (independent per-atom cavities) |

### Excited states

| Original keyword | vibe-qc equivalent |
|---|---|
| `DAVIDSONCIS` | `msindo_cis(..., spin="singlet"/"triplet", n_states=n)` |
| `SROI n` / `TROI n` | `msindo_cis(..., n_states=n)` |
| `CISGRAD n` | `msindo_cis_gradient_fd(..., state=n)` |
| `SCALEDCIS` / `CISCORR` | Not applied (vibe-qc ships rigorous CIS) |

### Miscellaneous

| Original keyword | vibe-qc equivalent |
|---|---|
| `PRINTOPTS=...` | Dump matrices via `_scf_rhf` debug hooks (developer use) |
| `DOFF=1` (remove d from Al-Cl) | Not exposed; d is always present |
| `PON=1` (2p on H for H-bonds) | Not yet implemented |

---

## Backend architecture (for developers)

MSINDO in vibe-qc is implemented as two cooperating engines:

1. **Python reference engine** (`python/vibeqc/semiempirical/methods/msindo.py`):
   covers the validated INDO H-Xe surface and supplies the complete analytic
   multipole derivative for closed-shell NDDO H, Li-F, and Na-Cl, including the
   source SPDD contraction for Al-Cl. Open-shell NDDO remains gated.

2. **C++ engine** (`cpp/include/vibeqc/semiempirical/methods/indo/`):
   header-only, Eigen-based, and pybind-exposed. Production molecular energy
   calls use this native path across the validated INDO and closed-shell NDDO
   scopes. Closed-shell INDO analytic gradients are native; the NDDO analytic
   contraction runs in Python over the exact same energy definition.

The integral kernel (`msindo_integrals.py` / `msindo_integrals.hpp`) is a
faithful port of MSINDO's `aintgs.f`/`bintgs.f`/`c2int.f`/`v2int.f`/`overlap.f`
STO engine: analytical Slater overlaps, two-centre Coulomb, nuclear attraction,
Slater-Condon one-centre integrals, and the HARMTR rotation transformation.

### Key routines (for those reading the code)

| Vibe-qc function | Original MSINDO routine | Purpose |
|---|---|---|
| `_build_core_and_gamma` | `intov.f` + `intdrv.f` | Builds H⁰ and two-centre γ |
| `_build_fock` / `_build_fock_uhf` | `fockcl.f` / `fockop.f` | INDO Fock matrix |
| `_scf_rhf` / `_scf_uhf` | `scfclo.f` / `scfopn.f` | DIIS-accelerated SCF loop |
| `_pair_blocks` | `intdrv.f` → rotation | Per-pair local→global integral assembly |
| `one_center_gmunu` | `gij1.f` | One-centre Coulomb/exchange block |
| `_v2core_one_side` | `v2core.f` | Frozen-core pseudopotential |
| `_lmunu_local` / `_add_einzi_dblock` | `lmunu.f` / `fockcl.f` EINZI | Löwdin + d-hybrid blocks |
| `_nddo_fock_extra` | `nddofockcl.f` + `spspfockcl.f` | NDDO multipole 2e Fock |
| `slater_condon` | `calsla.f` / `einzentren.f` | Slater-Condon F⁰/F²/G¹ params |
| `eneg` / `ateng` | `atomic_reference.f` | Per-element ENEG/ATENG |
| `build_wigner_seitz` (CCM) | `neighbors.f` | WS cell construction |
| `_madkonst_2d` / `_madkonst_3d` | `madelkonst.f` | Ewald Madelung constants |

---

## Tutorial: end-to-end workflow

### 1. Single-point energy

```python
from vibeqc import Atom, Molecule, run_job

# Build a water molecule (bohr)
h2o = Molecule([
    Atom(8,  [0.0, 0.0, 0.0]),
    Atom(1,  [0.0, 1.43, 1.11]),
    Atom(1,  [0.0, -1.43, 1.11]),
])

result = run_job(h2o, method="msindo")
print(f"Total energy: {result.energy:.10f} Ha")
print(f"Binding energy: {result.binding_energy:.6f} Ha")
```

### 2. Geometry optimization

```python
from vibeqc.semiempirical.methods.msindo import msindo_optimize

Z = [8, 1, 1]
xyz_start = [[0, 0, 0], [0, 0.8, 0.6], [0, -0.8, 0.6]]  # Å

result = msindo_optimize(Z, xyz_start)
print(f"Optimized energy: {result.energy:.10f} Ha")
print(f"Optimized geometry (Å):\n{result.coords_angstrom}")
```

### 3. Solvation energy

```python
from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo

# Water in water
r = msindo_cosmo([8, 1, 1],
                 [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]],
                 epsilon=78.39)
print(f"In-water energy: {r.total_energy:.10f} Ha")
print(f"Solvation energy: {r.e_solv * 1000:.3f} mHa")
```

### 4. Periodic CCM, bulk → surface → adsorption

Full script: `examples/semiempirical/11_msindo_ccm.py`.  In brief:

```python
from vibeqc.semiempirical.methods.msindo_ccm import run_ccm, ccm_gradient_fd, ccm_optimize

a = 4.21   # MgO lattice constant (Å)

# Bulk MgO rocksalt cell
fcc = [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]
Z_bulk = [12,12,12,12, 8,8,8,8]
C_bulk = [[f*a for f in p] for p in fcc[:4]] + \
         [[(p[0]+0.5)*a, p[1]*a, p[2]*a] for p in fcc[:4]]
cell_3d = [[a,0,0], [0,a,0], [0,0,a]]

e_bulk = run_ccm(Z_bulk, C_bulk, cell_3d, madelung=True).total_energy

# Surface slab (2-D)
Z_slab = [12,8,8,12, 8,12,12,8]
C_slab = [[0,0,0], [a*0.5,0,0], [0,a*0.5,0], [a*0.5,a*0.5,0],
          [0,0,a*0.5], [a*0.5,0,a*0.5], [0,a*0.5,a*0.5], [a*0.5,a*0.5,a*0.5]]
cell_2d = [[a,0,0], [0,a,0]]

e_slab = run_ccm(Z_slab, C_slab, cell_2d, madelung=True).total_energy

# Add water adsorbate
Z_ads = Z_slab + [8,1,1]
C_ads = C_slab + [[a*0.5, 0, a*0.5+2.2],
                  [a*0.5+0.76, 0, a*0.5+2.76],
                  [a*0.5-0.76, 0, a*0.5+2.76]]
e_sys = run_ccm(Z_ads, C_ads, cell_2d, madelung=True).total_energy

# Gas-phase water for the adsorption energy
from vibeqc.semiempirical.methods import msindo
e_h2o_gas = msindo.run_msindo([8,1,1],
    [[0,0,0.1173], [0,0.7572,-0.4692], [0,-0.7572,-0.4692]]).total_energy

E_ads = e_sys - e_slab - e_h2o_gas  # Hartree
print(f"Adsorption energy: {E_ads * 627.509:.2f} kcal/mol")
```

### 5. CIS excited states

```python
from vibeqc.semiempirical.methods.msindo import msindo_cis

Z = [8, 1, 1]
xyz = [[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]]   # Å

singlets = msindo_cis(Z, xyz, spin="singlet", n_states=5)
for i, ev in enumerate(singlets.excitation_energies_ev, 1):
    print(f"S{i}  {ev:.4f} eV")
```

### 6. NEB, reaction path

```python
from vibeqc import Atom, Molecule, run_neb
import numpy as np

# NH3 umbrella flip: reactant and its mirror
def nh3(flip=False):
    r, angle = 1.012, np.deg2rad(106.67)
    sin_a = np.sqrt(2*(1-np.cos(angle))/3)
    coords = [[0,0,0]]
    for k in range(3):
        phi = k * 2*np.pi/3
        z = -r*np.sqrt(1-sin_a**2) if not flip else +r*np.sqrt(1-sin_a**2)
        coords.append([r*sin_a*np.cos(phi), r*sin_a*np.sin(phi), z])
    return Molecule([Atom(7, c) for c in coords[:1]]
                  + [Atom(1, c) for c in coords[1:]])

res = run_neb(nh3(False), nh3(True), method="msindo", n_images=7, climbing=True)
print(f"Barrier: {res.barrier_kcal:.2f} kcal/mol")
```

---

## Citing

When you publish MSINDO results from vibe-qc, cite the method papers (vibe-qc
emits these automatically):

* B. Ahlswede, K. Jug, *Consistent modifications of SINDO1: I. Approximations
  and parameters*, **J. Comput. Chem. 20, 563 (1999)**.
* B. Ahlswede, K. Jug, *Consistent modifications of SINDO1: II. Applications to
  first- and second-row elements*, **J. Comput. Chem. 20, 572 (1999)**.

For NDDO calculations, also cite:

* B. Voigt, *On Bridging the Gap between the INDO and the NDDO Schemes*,
  **Theor. Chim. Acta 31, 289 (1973)**, DOI `10.1007/BF00527556`.
* M. J. S. Dewar, W. Thiel, *A semiempirical model for the two-center
  repulsion integrals in the NDDO approximation*, **Theor. Chim. Acta 46,
  89 (1977)**, DOI `10.1007/BF00548085`.

For periodic (CCM) calculations, cite:

* T. Bredow, G. Geudtner, K. Jug, *The Cyclic Cluster Model: A quantum
  chemical approach to periodic systems*, **J. Comput. Chem. 22, 89 (2001)**.
* M. F. Peintinger, T. Bredow, *The Cyclic Cluster Model revisited*,
  **J. Comput. Chem. 35, 839 (2014)**.

For COSMO solvation runs, also cite the conductor model and the CPCM/GEPOL
cavity/solver papers (emitted automatically when you pass `output=`):

* A. Klamt, G. Schüürmann, *COSMO: a new approach to dielectric screening in
  solvents…*, **J. Chem. Soc. Perkin Trans. 2, 799 (1993)**.

See also [the citation guide](citations.md).
