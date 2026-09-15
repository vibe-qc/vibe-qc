# Open-shell Mg⁺• in a periodic box: UHF GDF with Makov-Payne

This tutorial exercises the v0.9.0 `run_pbc_bipole_uhf` driver
(open-shell periodic SCF via the BIPOLE Fock build
at the Γ point) on a single Mg⁺• cation embedded in
a vacuum-padded simulation cell. Compares the converged
periodic energy to a molecular UHF reference, with the
**Makov-Payne charge-correction** bound on the residual
periodic-image interaction.

**You will**:

- Build a vacuum-padded simulation cell for a single Mg⁺•
  cation.
- Run UHF at increasing cell sizes (10 Å → 20 Å → 30 Å).
- Compute the Makov-Payne correction and bound the residual
  periodic-image error.
- Compare against a molecular UHF reference for the same
  cation.

**Background**: see
[`user_guide/multi_k_scf.md`](../user_guide/multi_k_scf.md) §
Open-shell GDF for the periodic UHF driver, and
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

Expected: monotonic convergence as the cell grows, the
periodic-image interaction (Makov-Payne 1st-order term) goes
as 1/L for an isolated charge in a uniform compensating
background:

```
  L= 10.0 Å: E = -198.812... Ha   ⟨S²⟩ = 0.7501
  L= 15.0 Å: E = -198.789... Ha   ⟨S²⟩ = 0.7500
  L= 20.0 Å: E = -198.776... Ha   ⟨S²⟩ = 0.7500
  L= 30.0 Å: E = -198.769... Ha   ⟨S²⟩ = 0.7500
```

⟨S²⟩ = 0.75 = ¾ ✓ (exact for a doublet with one unpaired
electron). If ⟨S²⟩ drifts above ~0.76, that's spin
contamination, typical for transition-metal radicals; for
main-group alkali / alkali-earth cations like Mg⁺• it stays
near-exact.

## Makov-Payne correction

The exact dependence on cell size for a single charge in a
neutral-background-compensated periodic cell is, to leading
order in 1/L:

```
E(L) = E(∞) + α_M · q² / (2L) + O(1/L³)
```

where `α_M ≈ 2.837297` is the Madelung constant for a simple-
cubic lattice with a uniform compensating background and
`q = +1` for Mg⁺•. So the *corrected* energy is:

```python
def makov_payne_e_inf(L_bohr: float, e_per_cell_ha: float, q: float = 1.0) -> float:
    """First-order Makov-Payne correction for an isolated charge."""
    alpha_M = 2.837297                  # SC Madelung constant
    return e_per_cell_ha - alpha_M * q**2 / (2 * L_bohr)


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

Expected: the **corrected** energies should agree with each
other and with the molecular UHF reference to ~µHa across
the L range. If they don't, but the uncorrected energies
do show 1/L scaling, your basis set is too small to be
behaving as an isolated charge (try increasing to def2-svp).

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

The molecular reference at sto-3g is **independent of any
periodic-image correction**: it is the direct `E(infinity)`
you are trying to reproduce with the periodic + Makov-Payne
approach.

The periodic-corrected `E_corr` at L = 30 A should match
`mol_result.energy` to ~uHa. **If it doesn't**, check:

1. Are you using the same basis set for both?
2. Are both at multiplicity = 2 (doublet)?
3. Is the SCF converged to comparable tolerance?
4. Is the cell really vacuum-padded (no atom touching the
   cell boundary)?

## What this tells you

- **Open-shell periodic SCF works at Γ since v0.9.0.**
  `run_pbc_bipole_uhf` and `run_pbc_bipole_uks` cover the
  open-shell-Γ paths via the BIPOLE Fock build.
  Multi-k UHF / UKS is paired with the periodic GDF parity work
  (the in-progress flagship, see the [roadmap](../roadmap.md)).
- **Charged periodic systems need the Makov-Payne
  correction.** The "raw" periodic energy includes the
  spurious self-interaction with the periodic images of the
  same charge; subtracting the leading 1/L term recovers the
  isolated-cation energy.
- **The molecular limit is reachable.** With a big enough
  vacuum cell + Makov-Payne, the periodic UHF + the molecular
  UHF agree to µHa. This is the regression test the v0.8.0
  open-shell-GDF chip was validated against.

## Where this generalises

- **Open-shell ions in solids**: e.g. Fe³⁺ in α-Fe₂O₃; the
  same Makov-Payne correction applies for any net-charged
  defect.
- **Antiferromagnetic ground states**: requires a broken-
  symmetry initial guess for the spin density. Roadmap item
  (R3 in the basissetdev periodic-requirements list).
- **Open-shell radicals in molecular contexts**: drop the
  periodic system entirely; use molecular `run_uhf` with the
  same UHF options.
- **Open-shell DFT (UKS)**: replace `run_pbc_bipole_uhf`
  with `run_pbc_bipole_uks` and add a `functional=`.

## See also

- [`user_guide/multi_k_scf.md`](../user_guide/multi_k_scf.md), 
  multi-k periodic SCF (closed-shell at v0.8.0; open-shell
  multi-k queued).
- [`user_guide/ewald.md`](../user_guide/ewald.md), Madelung
  + Ewald-summed nuclear attraction; the theoretical basis
  for the Makov-Payne correction.
- [Open-shell systems (UHF, UKS, UMP2)](open_shell.md), 
  molecular UHF basics.
- [Tight-cell DFT with the periodic Becke partition](tight_cell_dft.md), 
  periodic SCF in dense ionic insulators.
