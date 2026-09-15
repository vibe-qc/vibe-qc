# Periodic KS-DFT

Switch from HF to DFT by using `PeriodicKSOptions` instead of
`PeriodicSCFOptions` and calling the KS dispatcher
`run_rks_periodic_scf` (the recommended user-facing entry point, 
mirrors `run_rhf_periodic_scf` from [Periodic HF on a 1D H chain](periodic_hf.md)):

```python
import numpy as np
from vibeqc import (
    Atom, BasisSet, KPoints, PeriodicSystem, PeriodicKSOptions,
    run_rks_periodic_scf,
)

unit_cell = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
sysp = PeriodicSystem(
    dim=1,
    lattice=np.diag([4.0, 30.0, 30.0]),
    unit_cell=unit_cell,
)
basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
kpoints = KPoints.monkhorst_pack(sysp, [6, 1, 1])

opts = PeriodicKSOptions()
opts.functional = "PBE"
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 40.0
opts.conv_tol_energy = 1e-10
# Default: opts.lattice_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED.
# Flip to CoulombMethod.EWALD_3D for any tight 3D bulk — see "Coulomb
# method" below.

result = run_rks_periodic_scf(sysp, basis, kpoints, opts)

print(f"E_KS per cell  = {result.energy:.10f}")
print(f"  E_Coulomb    = {result.e_coulomb:.10f}")
print(f"  E_xc         = {result.e_xc:.10f}")
print(f"  E_HF_exch    = {result.e_hf_exchange:.10f}  (0 for pure DFT)")
```

For Γ-only calculations there is the matching
`run_rks_periodic_gamma_scf(sysp, basis, opts)` (no kmesh argument)
- internally it just dispatches the multi-k driver at a `[1,1,1]`
mesh, since vibe-qc's C++ stack doesn't ship a separate Γ-only KS
entry point.

## Functional support

All 500+ libxc functionals work in principle. Today's validation
covers:

* **LDA** (Slater + VWN5)
* **PBE**, pure GGA
* **BLYP**, pure GGA

Hybrid functionals (B3LYP, PBE0, …) work in the code path, the
`exchange_scale` parameter on `build_fock_2e_real_space` routes the
HF-exchange fraction through the lattice sum, but have not yet
been validated end-to-end against CRYSTAL reference numbers.

## Coulomb method

The KS dispatcher (``run_rks_periodic_scf`` /
``run_rks_periodic_gamma_scf``) inspects
``opts.lattice_opts.coulomb_method`` exactly like the RHF
dispatcher:

- ``CoulombMethod.DIRECT_TRUNCATED`` (default), routes to
  ``run_rks_periodic`` (the original KS C++ driver, conditionally
  convergent in 3D). Fine for 1D / 2D and for 3D molecular-limit
  cells with lattice $\gtrsim 15$ bohr.
- ``CoulombMethod.EWALD_3D``, full Gaussian-charge Ewald,
  unconditionally convergent. Shipped end-to-end:
  - Γ-only via ``run_rks_periodic_gamma_scf`` (Phase 15c-1)
  - multi-k via ``run_rks_periodic_scf`` (Phase 15c-2)
  - open-shell counterparts ``run_uks_periodic_gamma_scf`` /
    ``run_uks_periodic_scf`` (Phase 15c-3), see [Open-shell systems (UHF, UKS, UMP2)](open_shell.md)
    for the molecular open-shell story; the periodic UKS dispatcher
    follows the same pattern.

For any **quantitative 3D bulk DFT** (lattice $\lesssim 12$ bohr,
metal oxides, dense ionic crystals), use ``EWALD_3D``. The
[periodic Becke partition](tight_cell_dft.md) takes care of the
*other* tight-cell pathology, DFT integration weights, and the
two flags compose: turn both on for tight crystals.

```python
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.use_periodic_becke = True       # the integration-weight fix
opts.becke_image_radius_bohr = 10.0
```

See the [Ewald user guide](../user_guide/ewald.md) for the
ω-invariance story and tunable parameters; the `tight_cell_dft` tutorial for the
periodic Becke partition.

## Grid quality

Reuse the molecular-DFT grid controls:

```python
opts.grid.n_radial = 99
opts.grid.n_theta  = 29
opts.grid.n_phi    = 58
```

The grid is built over the unit-cell atoms. Since v0.9.x, the default
is the **periodic** Becke partition (`use_periodic_becke=True`), which
removes the small (<1e-4 Ha) boundary artefacts the plain molecular
partition produces on tight unit cells where image-atom Voronoi cells
intrude on the reference cell, see [Tight-cell DFT with the periodic
Becke partition](tight_cell_dft.md) for the full derivation and
convergence data.

```{tip}
**SCF won't converge?** [`examples/debug/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/debug)
ships nine focused diagnostic scripts. For DFT-specific cases
the most relevant are ``scf_easy_periodic.py`` (Ne / Ar / diamond
C must converge — known-good baseline), ``scf_sad_vs_hcore.py``
(does the v0.6.1 SAD initial guess rescue HCORE divergences?),
and ``scf_iteration_recorder.py`` (4-panel forensics plot of any
divergent trajectory). [Periodic SCF convergence: damping, DIIS, and level shifts](periodic_scf_convergence.md)
walks through level shift / DIIS / damping / smearing aids in
detail.
```

## Theory

The Kohn-Sham equations at a single $\mathbf{k}$-point look
identical to the molecular KS equations of
[Molecular DFT (RKS)](molecular_dft.md):

$$
\mathbf{F}^\text{KS}(\mathbf{k}) \, \mathbf{C}(\mathbf{k}) = \mathbf{S}(\mathbf{k}) \, \mathbf{C}(\mathbf{k}) \, \boldsymbol{\varepsilon}(\mathbf{k}),
$$

with $\mathbf{F}^\text{KS}(\mathbf{k})$ assembled from its real-space
lattice sum via the Bloch transform of
[Periodic HF on a 1D H chain](periodic_hf.md). The core ideas for both machines, 
the Bloch summation, the Monkhorst-Pack k-integration, the
direct-truncation caveat for the 3D Coulomb sum, are identical to
periodic HF and documented there.

What's new in periodic KS-DFT is the **exchange-correlation
contribution**. The density at a point is now Bloch-summed:

$$
\rho(\mathbf{r}) = \frac{1}{N_\mathbf{k}} \sum_\mathbf{k} \sum_n^\text{occ}(\mathbf{k})
  |\psi_{n\mathbf{k}}(\mathbf{r})|^2,
$$

where the sum is over $N_\mathbf{k}$ sampled k-points and over the
occupied bands at each. The numerical integration of
$V^\text{xc}_{\mu\nu}(\mathbf{T})$ on the DFT grid then needs to
account for periodic images of the basis functions. vibe-qc builds the
grid from the *unit-cell* atomic centers; the plain molecular Becke
partition is accurate for well-separated periodic cells (the
molecular-limit regime) but produces small partition-boundary
artefacts for tight cells. The **periodic Becke partition** is the fix,
and has been the default (`use_periodic_becke=True`) since v0.9.x, see
[Tight-cell DFT with the periodic Becke partition](tight_cell_dft.md).

### Hybrid DFT in periodic systems

Hybrid functionals (B3LYP, PBE0, …) mix a fraction $\alpha$ of HF
exchange into the XC energy. In a periodic system this translates
to an $\mathcal{O}(N_\text{bf}^4)$ lattice sum of the exchange-type
two-electron integrals

$$
K_{\mu\nu}(\mathbf{T}) = \sum_{\lambda\sigma, \mathbf{T}', \mathbf{T}''}
  P_{\lambda\sigma}(\mathbf{T}'' - \mathbf{T}') \,
  (\mu \mathbf{0} \, \sigma \mathbf{T}' \, | \, \nu \mathbf{T} \, \lambda \mathbf{T}''),
$$

instead of the $\mathcal{O}(N_\text{bf}^3)$ sum of local XC matrix
elements for pure DFT. That's 10-100× more expensive, the reason
solid-state DFT defaults to pure GGAs (PBE, BLYP). vibe-qc's
`build_fock_2e_real_space(..., exchange_scale=α)` routes the hybrid
exchange through the same lattice-summed ERI machinery used by HF.

### Basis-set considerations for solids

Two basis-set issues specific to periodic calculations, summarized
from [Basis-set convergence](basis_convergence.md):

1. **Linear dependencies.** Diffuse molecular basis functions
   (`aug-cc-pVXZ`, `6-311++G**`) overlap strongly across
   neighboring periodic cells, pushing the overlap matrix toward
   singularity. vibe-qc's canonical-orthogonalisation path handles
   mild cases; stronger contamination needs a redesigned basis.
2. **Solid-state basis families.** pob-TZVP (Peintinger-Vilela
   Oliveira-Bredow 2013) is the standard starting point, it's a
   systematic re-derivation of TZVP with the smallest-exponent
   primitives either removed or replaced, and ships with vibe-qc.

## Resources

~250 MB peak RAM, ~2 s on one core (Apple M2 baseline) for the
H₂-chain PBE / 6-k example. Add ~1 s per k-point for the XC grid
build inside each iteration (75-radial × 302-angular Lebedev shells
per atom by default). Use the [periodic Becke partition](tight_cell_dft.md)
on tight cells to avoid the over-counting of integration weights
that hurts molecular-Becke DFT.

## References

- **Periodic KS-DFT.** The theoretical framework is Kohn-Sham 1965
  (see [Molecular DFT (RKS)](molecular_dft.md) for full citation)
  applied to crystalline orbitals per Pisani-Dovesi 1980 (see
  [Periodic HF on a 1D H chain](periodic_hf.md)).
- **Periodic Becke partition.** M. D. Towler, A. Zupan, and M.
  Causà, "Density functional theory in periodic systems using
  local Gaussian basis sets," *Comput. Phys. Commun.* **98**, 181
  (1996). The published derivation of Becke fuzzy-cell numerical
  XC integration extended to crystalline Gaussian-basis systems
  (the scheme used by CRYSTAL); it builds on the molecular
  multicenter partition of A. D. Becke, "A multicenter numerical
  integration scheme for polyatomic molecules," *J. Chem. Phys.*
  **88**, 2547 (1988).
- **Solid-state basis sets (pob-TZVP).** M. F. Peintinger, D. V.
  Oliveira, and T. Bredow, "Consistent Gaussian basis sets of
  triple-zeta valence with polarization quality for solid-state
  calculations," *J. Comput. Chem.* **34**, 451 (2013).
- **CRYSTAL reference.** A. Erba, J. K. Desmarais, S. Casassa, B.
  Civalleri, L. Donà, I. J. Bush, B. Searle, L. Maschio, L. Edith-
  Daga, A. Cossard, C. Ribaldone, E. Ascrizzi, N. L. Marana, J.-P.
  Brandenburg, and R. Dovesi, "CRYSTAL23: a program for
  computational solid state physics and chemistry," *J. Chem.
  Theory Comput.* **19**, 6891 (2023).
