---
myst:
  html_meta:
    "description": "Open-shell periodic UHF on the BIPOLE route in vibe-qc: a single Mg+ cation in a vacuum-padded cubic cell, driven with run_pbc_bipole_uhf at the Gamma point, converged against cell size and corrected with the first-order Makov-Payne term to recover the molecular UHF reference. Background on the leading 1/L image interaction and where the correction applies."
    "og:title": "vibe-qc tutorial - open-shell Mg+ in a periodic box: BIPOLE UHF with Makov-Payne"
---

# Open-shell Mg⁺• in a periodic box: BIPOLE UHF with Makov-Payne

This tutorial exercises the `run_pbc_bipole_uhf` driver
(open-shell periodic SCF via the BIPOLE Fock build
at the Γ point) on a single Mg⁺• cation embedded in
a vacuum-padded simulation cell. BIPOLE open-shell energies are a
production route (see
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)).
It compares the converged
periodic energy to a molecular UHF reference, with the
**Makov-Payne charge-correction** bound on the residual
periodic-image interaction.

## Background

A charged cell cannot be summed over an infinite lattice as it stands: the
Coulomb energy of a periodic array of net charges diverges. Every periodic
route therefore adds a uniform neutralising background (the $\mathbf G = 0$
term of the Ewald sum is dropped), and the energy per cell that comes out is
the energy of the ion *plus* its interaction with its own images and with
the background. For an isolated charge $q$ in a cubic cell of edge $L$,
Makov and Payne ({vibeqc-cite}`makov_payne_1995`) showed that this spurious
interaction has the expansion

$$
E(L) = E(\infty) - \frac{\alpha_M q^2}{2L} - \frac{2\pi q\,Q}{3L^3} + \mathcal O(L^{-5}),
$$

where $\alpha_M$ is the Madelung constant of the lattice of point charges in
the background (2.837297 for simple cubic), and $Q$ is the quadrupole moment
of the charge distribution. The leading term is pure electrostatics and can
be subtracted exactly; the quadrupole term needs the density and decays two
powers faster, which is why the tutorial stops at first order and checks the
residual by growing the box instead. The doublet is handled by UHF, and
$\langle S^2\rangle$ stays at $3/4$ (0.7501 in the runs below) because the
one unpaired $3s$ electron sits outside a closed-shell Ne-like core whose
spin polarisation is negligible.

**You will**:

- Build a vacuum-padded simulation cell for a single Mg⁺•
  cation.
- Run UHF at increasing cell sizes (10 Å → 20 Å → 30 Å).
- Compute the Makov-Payne correction and bound the residual
  periodic-image error.
- Compare against a molecular UHF reference for the same
  cation.

**Further reading**: [`user_guide/bipole.md`](../user_guide/bipole.md) for
the BIPOLE open-shell drivers, [`user_guide/multi_k_scf.md`](../user_guide/multi_k_scf.md)
for the multi-k open-shell GDF alternative, and
[`user_guide/ewald.md`](../user_guide/ewald.md) for the
Madelung / Makov-Payne theory.

## Setup

A small helper builds the system: a single Mg⁺• at the centre of a cubic
vacuum-padded cell of edge `L`, with the +1 charge and doublet
multiplicity carried on the `PeriodicSystem` itself.

```python
import numpy as np
import vibeqc as vq

# Single Mg+ at the centre of a cubic vacuum cell.
def make_mg_cation_cell(L_angstrom: float):
    """Mg+ at cell centre, vacuum-padded cubic cell.

    +1 charge and doublet multiplicity belong to the PeriodicSystem
    itself, not to the SCF options.
    """
    L = L_angstrom / 0.529177           # Angstrom to bohr
    lattice = np.eye(3) * L              # 3x3 lattice (bohr)
    return vq.PeriodicSystem(
        3,                               # dim = 3
        lattice,
        [vq.Atom(12, [L / 2, L / 2, L / 2])],
        charge=+1,                       # +1 cation
        multiplicity=2,                  # doublet, UHF spin-unpaired
    )
```

## Vacuum convergence sweep

Running `run_pbc_bipole_uhf` at a sequence of growing cell sizes shows
the energy converging as the box expands, while `<S²>` stays pinned at
the doublet value of ¾:

```python
for L in (10.0, 15.0, 20.0, 30.0):
    cell = make_mg_cation_cell(L)
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(cell, [1, 1, 1])      # Gamma-only

    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 15.0

    result = vq.run_pbc_bipole_uhf(cell, basis, kmesh, opts)
    print(f"  L={L:5.1f} A: E = {result.energy:.6f} Ha   "
          f"<S^2> = {result.s_squared:.4f}")
```

Expected: monotonic convergence from below as the cell grows, the
periodic-image interaction (Makov-Payne 1st-order term) goes
as $-1/L$ for an isolated charge in a uniform compensating
background. On a vibe-qc 0.17.x development build (Apple M2, STO-3G):

```
  L= 10.0 Å: E = -197.047508 Ha   ⟨S²⟩ = 0.7501
  L= 15.0 Å: E = -197.018490 Ha   ⟨S²⟩ = 0.7501
  L= 20.0 Å: E = -197.005006 Ha   ⟨S²⟩ = 0.7501
  L= 30.0 Å: E = -196.991995 Ha   ⟨S²⟩ = 0.7501
```

⟨S²⟩ = 0.75 = ¾ ✓ (exact for a doublet with one unpaired
electron). If ⟨S²⟩ drifts above ~0.76, that's spin
contamination, typical for transition-metal radicals; for
main-group alkali / alkali-earth cations like Mg⁺• it stays
near-exact.

## Makov-Payne correction

The exact dependence on cell size for a single charge in a
neutral-background-compensated periodic cell is, to leading
order in 1/L (see the Background section):

```
E(L) = E(∞) - α_M · q² / (2L) + O(1/L³)
```

where `α_M ≈ 2.837297` is the Madelung constant for a simple-
cubic lattice with a uniform compensating background and
`q = +1` for Mg⁺•. The image interaction *lowers* the periodic
energy (a lattice of charges in a neutralising background is
bound), so the *corrected* energy adds the term back:

```python
def makov_payne_e_inf(L_bohr: float, e_per_cell_ha: float, q: float = 1.0) -> float:
    """First-order Makov-Payne correction for an isolated charge."""
    alpha_M = 2.837297                  # SC Madelung constant
    return e_per_cell_ha + alpha_M * q**2 / (2 * L_bohr)


for L in (10.0, 15.0, 20.0, 30.0):
    L_bohr = L / 0.529177
    cell = make_mg_cation_cell(L)
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(cell, [1, 1, 1])

    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 15.0

    e_periodic = vq.run_pbc_bipole_uhf(cell, basis, kmesh, opts).energy
    e_corr = makov_payne_e_inf(L_bohr, e_periodic)
    print(f"  L={L:5.1f} A:"
          f"   E_periodic = {e_periodic:.6f} Ha"
          f"   E_corrected = {e_corr:.6f} Ha")
```

Expected: the **corrected** energies agree with each other and with
the molecular UHF reference far better than the raw ones, and the
residual shrinks as $1/L^3$ (the quadrupole term the first-order
formula omits):

```
  L= 10.0 Å:   E_periodic = -197.047508 Ha   E_corrected = -196.972437 Ha
  L= 15.0 Å:   E_periodic = -197.018490 Ha   E_corrected = -196.968442 Ha
  L= 20.0 Å:   E_periodic = -197.005006 Ha   E_corrected = -196.967470 Ha
  L= 30.0 Å:   E_periodic = -196.991995 Ha   E_corrected = -196.966971 Ha
```

Against the molecular reference of the next section (-196.966761 Ha)
the raw error at 10 Å is 81 mHa and the corrected error 5.7 mHa; at
30 Å they are 25 mHa and 0.21 mHa. Doubling $L$ from 10 to 20 Å divides
the corrected residual by 8, the $L^{-3}$ signature. If the corrected
energies do *not* line up but the uncorrected ones do show 1/L scaling,
your basis set is too small to be behaving as an isolated charge (try
increasing to def2-svp).

## Molecular UHF reference

A molecular `run_uhf` on the same isolated Mg⁺• cation gives the
`E(infinity)` the periodic-plus-Makov-Payne approach is trying to
reproduce, with no image correction of its own:

```python
mg_cation = vq.Molecule(
    [vq.Atom(12, [0.0, 0.0, 0.0])],
    charge=+1,
    multiplicity=2,                # doublet
)
basis = vq.BasisSet(mg_cation, "sto-3g")

mol_result = vq.run_uhf(mg_cation, basis)
print(f"\nMolecular UHF Mg+ reference:  {mol_result.energy:.6f} Ha")
print(f"<S^2>:                         {mol_result.s_squared:.4f}")
```

```
Molecular UHF Mg+ reference:  -196.966761 Ha
<S^2>:                         0.7501
```

The molecular reference at sto-3g is **independent of any
periodic-image correction**: it is the direct `E(infinity)`
you are trying to reproduce with the periodic + Makov-Payne
approach.

The periodic-corrected `E_corr` at L = 30 Å should match
`mol_result.energy` to a few tenths of a mHa (0.21 mHa above), with the
residual falling as $L^{-3}$. **If it doesn't**, check:

1. Are you using the same basis set for both?
2. Are both at multiplicity = 2 (doublet)?
3. Is the SCF converged to comparable tolerance?
4. Is the cell really vacuum-padded (no atom touching the
   cell boundary)?

## What this tells you

- **Open-shell periodic SCF works at Γ and at a k-mesh.**
  `run_pbc_bipole_uhf` and `run_pbc_bipole_uks` cover the
  open-shell paths via the BIPOLE Fock build (Gamma and multi-k, 3-D),
  and multi-k UHF / UKS via GDF (`run_kuhf_periodic_gdf`,
  `run_kuks_periodic_gdf`, or `run_periodic_job(..., jk_method="gdf")`)
  is public on `dim=1` and `dim=3`; only the slab (`dim=2`) open-shell
  case remains fail-closed. A single ion in a big box needs only Γ, which
  is why this page uses a `(1, 1, 1)` mesh.
- **Charged periodic systems need the Makov-Payne
  correction.** The "raw" periodic energy includes the
  spurious self-interaction with the periodic images of the
  same charge; subtracting the leading 1/L term recovers the
  isolated-cation energy.
- **The molecular limit is reachable.** With a big enough
  vacuum cell + Makov-Payne, the periodic UHF and the molecular
  UHF agree to a few tenths of a mHa at 30 Å, with the residual
  vanishing as $L^{-3}$. This is the regression test the open-shell
  periodic drivers were validated against.

## Where this generalises

- **Open-shell ions in solids**: e.g. Fe³⁺ in α-Fe₂O₃; the
  same Makov-Payne correction applies for any net-charged
  defect.
- **Antiferromagnetic ground states**: requires a broken-
  symmetry initial guess for the spin density; see the
  [roadmap](../roadmap.md).
- **Open-shell radicals in molecular contexts**: drop the
  periodic system entirely; use molecular `run_uhf` with the
  same UHF options.
- **Open-shell DFT (UKS)**: replace `run_pbc_bipole_uhf`
  with `run_pbc_bipole_uks` and add a `functional=`.

## See also

- [`user_guide/bipole.md`](../user_guide/bipole.md), the BIPOLE
  route this page runs, including `run_pbc_bipole_uks`.
- [`user_guide/multi_k_scf.md`](../user_guide/multi_k_scf.md), 
  multi-k periodic SCF, closed- and open-shell, on GDF.
- [`user_guide/ewald.md`](../user_guide/ewald.md), Madelung
  + Ewald-summed nuclear attraction; the theoretical basis
  for the Makov-Payne correction.
- [Open-shell systems (UHF, UKS, UMP2)](open_shell.md), 
  molecular UHF basics.
- [Tight-cell DFT with the periodic Becke partition](tight_cell_dft.md), 
  periodic SCF in dense ionic insulators.
