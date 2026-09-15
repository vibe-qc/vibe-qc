---
myst:
  html_meta:
    "description": "Step-by-step Γ-CCM / aiccm2026dev-a tutorial: build a cyclic cluster, check the 8-fold ERI symmetry, run the full HF→MP2→CCSD(T) stack, compare RI and RIJCOSX, handle ionic systems, and extract properties."
    "og:title": "vibe-qc tutorial -- Γ-CCM / aiccm2026dev-a step by step"
---

# Γ-CCM / aiccm2026dev-a: step by step

```{warning}
**Γ-CCM is experimental.** Every `run_ccm_*` SCF driver in this tutorial emits
`vibeqc.AICCM2026DevAExperimentalWarning` -- numbers are research-grade; check
the parity gates on the [reference page](../aiccm2026dev_a.md) before using an
energy quantitatively. Silence it with
`warnings.filterwarnings("ignore", category=vibeqc.AICCM2026DevAExperimentalWarning)`.
```

This tutorial walks through **Γ-CCM**, the union-and-weight/Wigner-Seitz
integral-weighting line selected by `aiccm2026dev-a`. Every step is a runnable Python
fragment -- paste them in order. The companion user guide is
[Γ-CCM reference](../aiccm2026dev_a.md); the independent χ-CCM line has
its own [tutorial](aiccm2026dev_b.md).

```{seealso}
The [H-chain ancestor tutorial](h8_chain_ccm_ancestor.md) derives the
CCM-Bloch equivalence on a 1-D chain -- read that first if the
cluster-size ↔ k-mesh duality is new.
```

## 1. Build a cyclic cluster

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem

BOHR = 1.0 / 0.529177210903

# H4 alternating chain: 4 atoms/cell, a=4.0 bohr
pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
unit = PeriodicSystem(
    3,                                    # 3-D periodicity
    np.diag([4.0 * BOHR, 40.0, 40.0]),   # lattice: a=4, b=c=40 (vacuum)
    [Atom(1, p) for p in pos],
    charge=0, multiplicity=1,
)

# 4-cell chain along x (= 16 H atoms in the supercell)
ccm = CCMSystem(unit, (4, 1, 1), "sto-3g")
print(f"n_atoms = {ccm.n_atoms}, nbf = {ccm.nbf}")
```

`CCMSystem` builds the Wigner-Seitz supercell environment automatically.
The `(4, 1, 1)` means 4 replicas along the first lattice vector, 1
along the second and third -- a 1-D chain.

## 2. The 8-fold symmetry gate

The historical `union12` weight breaks ERI permutation symmetry in
non-orthorhombic lattices. The symmetric `aiccm2026dev-a` weight
restores it. Check it:

```python
from vibeqc.periodic.ccm.padded import build_padded_cluster, ccm_eri, ccm_eri_symmetric, eri_cells

pad = build_padded_cluster(ccm, eri_cells(ccm))

def max_asym(V):
    nrm = np.linalg.norm(V)
    return max(np.linalg.norm(V - V.transpose(1, 0, 2, 3)) / nrm,
               np.linalg.norm(V - V.transpose(0, 1, 3, 2)) / nrm,
               np.linalg.norm(V - V.transpose(2, 3, 0, 1)) / nrm)

V_union = ccm_eri(ccm, pad)                # historical product weight
V_sym   = ccm_eri_symmetric(ccm, pad)      # symmetric four-center
print(f"union12 asymmetry: {max_asym(V_union):.2e}")
print(f"aiccm2026dev-a asymmetry: {max_asym(V_sym):.2e}")
```

On the H4 chain the two weights give the same energy (the 1-D WS symmetry
hides the asymmetry). In 2-D hexagonal and oblique lattices they diverge
by ~1.4 mHa/atom -- the [reference page](../aiccm2026dev_a.md) has the
numbers.

## 3. Hartree-Fock

```python
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable

rhf     = run_ccm_rhf(ccm, method="aiccm2026dev-a")
scalable = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")  # C++ kernel, reaches 3-D

print(f"RHF (Python)   E/atom = {rhf.energy_per_atom:.8f} Ha  "
      f"{rhf.n_iter} iters, converged={rhf.converged}")
print(f"RHF (scalable) E/atom = {scalable.energy_per_atom:.8f} Ha  "
      f"{scalable.n_iter} iters, converged={scalable.converged}")
```

The 1-D H4 chain gold reference is −0.542875 Ha/atom. Your number will
approach it as the cluster grows: at `nrep=(4,1,1)` it is −0.542676; at
`(8,1,1)` it is −0.542870.

## 4. Kohn-Sham DFT

Any libxc functional works. Pure and hybrid:

```python
from vibeqc.periodic.ccm.dft import run_ccm_rks

for func in ("pbe", "pbe0", "b3lyp"):
    rks = run_ccm_rks(ccm, func, method="aiccm2026dev-a")
    print(f"RKS/{func:6s} E/atom = {rks.energy_per_atom:.8f} Ha")
```

## 5. MP2 and CCSD(T) correlation

```python
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd

mp2 = run_ccm_mp2(ccm, method="aiccm2026dev-a")
cc  = run_ccm_ccsd(ccm, method="aiccm2026dev-a", compute_triples=True)

print(f"MP2     Ecorr = {mp2.e_correlation:.8f} Ha")
print(f"CCSD    Ecorr = {cc.e_ccsd_correlation:.8f} Ha")
print(f"CCSD(T) Ecorr = {cc.e_correlation:.8f} Ha, (T) = {cc.e_t:.2e}")
```

The post-HF stack uses the CCM MO integrals directly -- formally
molecular algebra over the cyclic-cluster orbitals. The molecular limit
is exact: CCSD(T) matches `run_ccsd` to the DF gap and (T) to ~1e-8.

## 6. Weighted approximations and neutral-torus controls

The WSSC RIJCOSX route is a construction-specific weighted approximation. The
GDF route below is instead a neutral fitted-torus control and is not Γ-CCM by
name:

```python
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rhf_rijcosx

hf_4c  = run_ccm_rhf(ccm, method="aiccm2026dev-a")
hf_ri  = run_ccm_rhf_gdf(ccm)      # neutral-torus character/Bloch control
hf_cox = run_ccm_rhf_rijcosx(ccm)  # WSSC RI-J + chain-of-spheres K research route

print(f"4-center  E/atom = {hf_4c.energy_per_atom:.8f} Ha")
print(f"RI (GDF)  E/atom = {hf_ri.energy_per_atom:.8f} Ha")
print(f"RIJCOSX   E/atom = {hf_cox.energy_per_atom:.8f} Ha")
```

Do not interpret `hf_ri - hf_4c` as a pure RI fitting error: the two calls can
represent different finite constructions/operators. `run_ccm_rhf_gdf` may be
compared with its real-Gamma neutral control for same-Hamiltonian Fourier
validation, but that result is not evidence about Γ-CCM/χ-CCM equality.

## 7. Cluster sizing by interaction radius

Instead of an explicit `(N1, N2, N3)` mesh, size the cluster by a
real-space cutoff -- the dual of choosing a k-mesh density:

```python
from vibeqc.periodic.ccm import nrep_for_interaction_range

# Derive the minimal cluster that encloses a sphere of R_c = 12 bohr
ccm_r = CCMSystem.from_interaction_range(unit, 12.0, "sto-3g")
print(f"R_c = 12 bohr → nrep = {ccm_r.nrep}, n_atoms = {ccm_r.n_atoms}")

# Or in Angstrom
ccm_a = CCMSystem(unit, basis="sto-3g", interaction_range_ang=6.5)
print(f"R_c = 6.5 Å   → nrep = {ccm_a.nrep}")

# Convergence scan -- the CCM analogue of a k-point convergence study
from vibeqc.periodic.ccm import ccm_interaction_range_scan

scan = ccm_interaction_range_scan(unit, "sto-3g",
                                  radii_bohr=[6, 9, 12, 15, 18, 24, 30],
                                  method="aiccm2026dev-a")
print(f"Converged to 0.1 mHa/atom at R_c = {scan.converged_radius(1e-4)} bohr")
```

The relationship is `Δk = π/R_c`: a larger real-space radius ⇔ a denser
k-mesh. On the 1-D H2 chain the energy converges to 0.1 mHa/atom by
`R_c ≈ 9` bohr and to 10 µHa/atom by `R_c ≈ 18` bohr.

## 8. Diagnosing the union-and-weight / neutral-control gap

The union-and-weight four-center and neutral (Ewald) fitted-torus control can
differ by a Madelung-scale shift on ionic systems. The following constructs
both operators explicitly; it does not replace one construction with the
other:

```python
from vibeqc.periodic.ccm import ccm_eri_neutral, ccm_neutral_background_constant
from vibeqc.periodic.ccm.scf import run_ccm_rhf

# Build a LiH chain (ionic)
a = 3.2 * BOHR
lih = PeriodicSystem(
    1, np.diag([a, 40.0, 40.0]),
    [Atom(3, [0, 0, 0]), Atom(1, [a/2, 0, 0])],
)
ccm_lih = CCMSystem(lih, (6, 1, 1), "sto-3g")

# The Madelung background constant ξ
xi = ccm_neutral_background_constant(ccm_lih)
print(f"Cell Madelung constant ξ = {xi:.4f}")

# Neutral four-center g_eff
g_neutral = ccm_eri_neutral(ccm_lih, ke_cutoff=120.0)

# Bare vs neutral SCF
scf_bare    = run_ccm_rhf(ccm_lih, method="aiccm2026dev-a")
scf_neutral = run_ccm_rhf(ccm_lih, eri=g_neutral)
print(f"Bare    4c E/atom = {scf_bare.energy_per_atom:.8f} Ha")
print(f"Neutral 4c E/atom = {scf_neutral.energy_per_atom:.8f} Ha")
print(f"Δ = {(scf_bare.energy_per_atom - scf_neutral.energy_per_atom) * 1000:.2f} mHa/atom")
```

If the intended study is the neutral fitted-torus control, use the same neutral
reference for HF/KS and post-HF because changing the background shifts the
occupied-virtual denominators. Do not publish that control as Γ-CCM. Conversely,
quantitative ionic Γ-CCM results need union-and-weight convergence and external
validation rather than silent substitution. See
[the operator gap](../aiccm2026dev_a.md#union-and-weight-versus-neutral-torus-control-gap)
for the full explanation.

## 9. Properties

```python
from vibeqc.periodic.ccm import (
    ccm_homo_lumo_gap, ccm_mulliken_charges,
    ccm_lowdin_charges, ccm_dipole,
)

# Gap (folded-Γ spectrum on the supercell)
homo, lumo, gap = ccm_homo_lumo_gap(rhf, ccm)
print(f"Gap: HOMO = {homo:.4f}  LUMO = {lumo:.4f}  Δ = {gap:.4f} Ha")

# Per-cell charges
mq = ccm_mulliken_charges(rhf, ccm)
lq = ccm_lowdin_charges(rhf, ccm)
for i, (m, l) in enumerate(zip(mq.per_cell_charges, lq.per_cell_charges)):
    print(f"  Atom {i}: Mulliken = {m:+.4f}  Löwdin = {l:+.4f}")
print(f"  Translational spread = {mq.translational_spread:.4e}")

# Finite-cluster dipole (≈ 0 for centrosymmetric)
mu = ccm_dipole(rhf, ccm)
print(f"Dipole = ({mu.x:.2e}, {mu.y:.2e}, {mu.z:.2e}) e·bohr")
```

## 10. Wannier localization and symmetry

```python
from vibeqc.periodic.ccm import localise_ccm, localization_density_residual
from vibeqc.periodic.ccm import analyze_ccm_symmetry, ccm_symmetry_invariance_residuals

# Localize the occupied orbitals (Pipek-Mezey)
loc = localise_ccm(rhf, ccm, method="pm")
print(f"Localization objective: {loc.objective_initial:.4f} → {loc.objective_final:.4f}")
print(f"Density residual (must be ≈ 0): {localization_density_residual(rhf, loc):.2e}")
for i, (center, spread) in enumerate(zip(loc.centers, loc.spreads)):
    print(f"  Wannier {i}: center = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
          f"  spread = {spread:.4f} bohr²")

# Symmetry analysis (spglib)
sym = analyze_ccm_symmetry(ccm)
print(f"Space group: {sym.space_group_number} ({sym.space_group_symbol})")
s_res, t_res = ccm_symmetry_invariance_residuals(ccm, sym)
print(f"S-invariance residual: {s_res:.2e}")
print(f"T-invariance residual: {t_res:.2e}")
```

## Next steps

- The [Γ-CCM reference](../aiccm2026dev_a.md) has the complete method
  stack, the DLPNO local-correlation entry point, open-shell UCCSD(T),
  and the full feature matrix.
- The [test-set runner](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README.md) runs any of 28
  curated systems through any route with one command.
- For the independent χ-CCM line, see the
  [χ-CCM tutorial](aiccm2026dev_b.md) and
  [user guide](../user_guide/aiccm2026dev_b.md).
