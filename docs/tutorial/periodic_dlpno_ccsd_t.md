---
myst:
  html_meta:
    "description": "Tutorial: DLPNO-CCSD(T) on crystals with run_periodic_job. An H2 molecular crystal at the Gamma point and on Gamma-centred k meshes folded to the Born-von Karman supercell, rock-salt LiH, an open-shell H3 cell on a UHF reference, the refusals, and the finite-size behaviour of the per-cell energies."
    "og:title": "vibe-qc tutorial: DLPNO-CCSD(T) on a crystal"
---

# DLPNO-CCSD(T) on a crystal

This tutorial runs coupled-cluster correlation on periodic systems with one
call, `run_periodic_job(..., method="dlpno-ccsd(t)")`. You start from a
molecular crystal of H2 at the Gamma point, converge the energy per cell
against the k mesh, repeat it in a polarised basis, move to an ionic crystal
(rock-salt LiH), and finish with an open-shell cell on a UHF reference. Along
the way you learn what the route folds, what it reports per cell and what it
refuses. The [user guide](../user_guide/periodic_local_correlation.md) has
the full option tables and the theory in more depth.

```{admonition} Status: preview
:class: warning

The periodic DLPNO route is reachable, maintained and tested, and at full
domains it reproduces canonical k-point CCSD(T) (PySCF KRCCSD(T)) to the fit
tolerance. Its accuracy at the shipped thresholds on crystals with a real
band structure, and its cost against cell size, are not measured yet, so the
`.out` block says the route "is not production-gated yet". The label
vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

**Prerequisites.** A working vibe-qc install
([installation](../installation.md)), the
[molecular DLPNO tutorial](dlpno_local_correlation.md) for what PNOs and
the thresholds do, and the [k-point tutorial](kpoints_brillouin_bloch.md)
for meshes and Bloch sums. Run every script from an empty scratch
directory: vibe-qc writes its outputs into the current directory.

All numbers on this page were measured with the vibe-qc 0.17.15 development
build (main, 2026-10-11) and reproduced to 1e-9 Ha on v0.18.0 (the H2
crystal at Gamma, on (2, 2, 2) and in def2-SVP on (1, 1, 2), and both H3
meshes) at the shipped defaults (NormalPNO, published
frozen core, `triples_mode="t1"` for RHF and `"t0"` for UHF). Wall times come
from a loaded laptop shared with other jobs and show orders of magnitude
only.

## The H2 molecular crystal at the Gamma point

The test crystal is a simple cubic lattice of H2 molecules: a 6 bohr cube
with one molecule of bond length 1.4 bohr along $z$. It is small enough to
run in seconds and has a real gap, which the route needs.

```python
import numpy as np
import vibeqc as vq

a, d = 6.0, 1.4                       # bohr
system = vq.PeriodicSystem(3, np.diag([a, a, a]), [
    vq.Atom(1, [a / 2, a / 2, a / 2 - d / 2]),
    vq.Atom(1, [a / 2, a / 2, a / 2 + d / 2]),
])
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

res = vq.run_periodic_job(system, basis, method="dlpno-ccsd(t)",
                          output="h2_gamma", progress=False)

c = res.correlation
print(f"E_HF/cell    {c.e_hf_per_cell:.10f}")
print(f"E_corr/cell  {c.e_corr_per_cell:.10f}")
print(f"E_tot/cell   {c.e_total_per_cell:.10f}")
print(c.kmesh, c.n_cells, c.source_kind, c.same_source,
      c.exchange_q0, c.localisation, c.backend)
print("pairs", res.dlpno_ccsd.n_pairs, "iterations", res.dlpno_ccsd.n_iter)
```

```text
E_HF/cell    -1.2385562896
E_corr/cell  -0.0094929996
E_tot/cell   -1.2480492892
(1, 1, 1) 1 gamma-gdf True BvK-ewald pipek-mezey periodic-gdf-dlpno-ccsd(t)
pairs 1 iterations 3
```

The call ran in about two seconds. Reading the last line from left to right:

- **`(1, 1, 1)` and `1`.** The mesh is the Gamma point, so the supercell is
  the unit cell.
- **`gamma-gdf`.** The correlation source is the Gamma adapter: the
  reference is the multi-k GDF engine at one k point, with its classic RSGDF
  fit retained.
- **`True`.** `same_source` means the correlation factors are that same fit,
  so the reference and the correlation see one Hamiltonian.
- **`BvK-ewald`.** The exchange zero mode of the reference Fock operator is
  the Born-von Karman Ewald convention, and the amplitude equations keep it.
- **`pipek-mezey`.** The adapter localises the active occupied orbitals with
  Pipek-Mezey, which is safe on a torus.
- **`periodic-gdf-dlpno-ccsd(t)`.** The citation route key, which also
  names the engine in the `.system` manifest.

One occupied orbital per cell gives one pair, so this first run is a
consistency check rather than a local-correlation calculation. The (T)
correction is zero to machine precision: in a minimal basis a two-electron
cell has no triples to speak of.

## Converge the energy per cell against the k mesh

A Gamma-only calculation on a 6 bohr cube is far from the bulk limit. Pass
a Gamma-centred mesh with `kpoints=`; the route runs the unreduced mesh,
folds it to the real BvK supercell of $N_k = n_1 n_2 n_3$ cells, and solves
one real supercell problem.

```python
for mesh in [(1, 1, 2), (2, 2, 2), (3, 3, 3)]:
    tag = "h2_sto-3g_" + "x".join(map(str, mesh))
    res = vq.run_periodic_job(system, basis, method="dlpno-ccsd(t)",
                              kpoints=mesh, output=tag, progress=False)
    c, cc = res.correlation, res.dlpno_ccsd
    print(mesh, c.n_cells, f"{c.e_hf_per_cell:.10f}",
          f"{c.e_corr_per_cell:.10f}", f"{c.e_t_per_cell:.2e}",
          f"{c.e_total_per_cell:.10f}", cc.n_pairs, cc.n_screened, cc.n_iter)
```

| Mesh | Cells | E_HF per cell, Ha | E_corr (CCSD) per cell, Ha | E(T) per cell, Ha | E_total per cell, Ha | Pairs kept (screened) | Wall, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gamma | 1 | -1.2385562896 | -0.0094929996 | 0 | -1.2480492892 | 1 (0) | 2 to 8 |
| (1, 1, 2) | 2 | -1.1725050397 | -0.0156121011 | 0 | -1.1881171408 | 3 (0) | 2 |
| (2, 2, 2) | 8 | -1.1203818266 | -0.0158587108 | 0 | -1.1362405374 | 12 (24) | 27 |
| (3, 3, 3) | 27 | -1.1177121108 | -0.0172763688 | -7.1e-07 | -1.1349891876 | 54 | 46 |

Three things stand out.

- **The reference moves most.** The Hartree-Fock energy per cell rises by
  0.12 Ha from Gamma to (3, 3, 3). Exact exchange on a finite mesh carries
  an error of order $1/N_k$, and the Gamma point in a 6 bohr cube puts the
  images of each molecule only 6 bohr away. The
  [multi-k SCF guide](../user_guide/multi_k_scf.md) explains this tail of
  the BvK-ewald gauge and how to extrapolate it.
- **The correlation energy grows in magnitude with the supercell.** It goes
  from -9.5 mHa at Gamma to -17.3 mHa on (3, 3, 3). On a small supercell
  the virtual space is a coarse sample of the Brillouin zone, so a pair
  misses part of its excitation space, and pairs between neighbouring
  molecules only appear once the supercell holds them. Converge the
  correlation energy per cell on its own mesh series; it does not converge
  at the same rate as the reference.
- **Pairs are screened on the larger supercells.** On (2, 2, 2) the 8
  occupied orbitals give 36 pairs; 12 are kept and 24 fall below the pair
  prescreening threshold `tcut_pairs`. Their dropped energy is estimated
  from the semicanonical MP2 pair energies. This is where local correlation
  starts to pay off on a crystal.

## The output block

Every run writes the correlation block to the `.out` file. This is the
(2, 2, 2) STO-3G run:

```text
  DLPNO-CCSD(T) local PNO-space (M3c) on the periodic GDF reference (RI: def2-svp-jk)
  -----------------------------------------------------------------------------------
  reference                      = KRHF/GDF, fit_source=classic; the retained fit is the correlation factors (same source)
  k-mesh / cells                 = (2, 2, 2) / 8 (Gamma-centred, folded to the real BvK supercell)
  correlation source             = bvk-supercell, identity d87dad4272cf3a96... (full in .system)
  exchange q=0                   = BvK-ewald
  occupied localisation          = pipek-mezey (active block, by the adapter)
  E(RHF reference) / cell        =    -1.1203818266 Ha
  Threshold convention           = NormalPNO
  Truncation thresholds          = tcut_pairs 0.0001  tcut_pno 3.33e-07  tcut_mkn 0.001  tcut_tno 0.0
  Residual domain                = pno
  PNO density norm               = mp2
  PNO MP2 correction             = on
  frozen core / cell             = 0 orbitals (ORCA 6.1 published count-only default)
  Triples algorithm              = rotated-occupied (T1) compatibility correction (triples_mode='t1')
  pairs / avg PNOs               = 12 / 1.3
  E(PNO truncation corr)         =     0.0000000000 Ha
  E(CCSD correlation) / cell     =    -0.0158587108 Ha
  triples / avg TNOs             = 92 / 8.0   (smallest domain: 8)
  E((T) correction) / cell       =    -0.0000000000 Ha
  E(DLPNO-CCSD(T) corr) / cell   =    -0.0158587108 Ha
  E(DLPNO-CCSD(T) total) / cell  =    -1.1362405374 Ha
  supercell E(corr) / E(total)   =    -0.1268696862 Ha /    -9.0899242989 Ha   (8 cells)
```

The source line now reads `bvk-supercell`: the per-k overlap, Fock matrix
and retained fit were inverse-transformed to the real supercell and the
supercell orbitals rebuilt from the folded Fock matrix. The identity hash
names that exact source; the full value is in the `.system` manifest as
`dlpno_source_identity`, next to `dlpno_kmesh`, `dlpno_n_cells`,
`dlpno_e_corr_per_cell` and the rest of the `dlpno_*` fields. The last line
gives the supercell totals, which are $N_k$ times the per-cell values.

## A polarised basis

STO-3G is a teaching basis. In def2-SVP each H carries a p shell, the pairs
get real PNO spaces and the triples correction becomes visible:

```python
basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
for mesh in [(1, 1, 1), (1, 1, 2), (2, 2, 2)]:
    ...  # same loop as above
```

| Mesh | E_HF per cell, Ha | E_corr (CCSD) per cell, Ha | E(T) per cell, Ha | E_total per cell, Ha | Pairs | Avg PNOs | Wall, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gamma | -1.2561489570 | -0.0191910611 | 0 | -1.2753400181 | 1 | 9.0 | 7 |
| (1, 1, 2) | -1.1889076672 | -0.0262675726 | -6.2e-05 | -1.2152368592 | 3 | 11.7 | 26 |
| (2, 2, 2) | -1.1321596850 | -0.0285352331 | -3.4e-05 | -1.1607285144 | 20 | 12.0 | 40 |

The correlation energy per cell is now about twice the STO-3G value and
still growing on (2, 2, 2). The average of 11.7 PNOs per pair on (1, 1, 2)
shows what the PNO compression buys: each pair keeps a dozen virtual
directions out of the supercell's virtual space.

## An ionic crystal: rock-salt LiH

The same call works for any gapped 3-D cell. Rock-salt LiH in its primitive
fcc cell, $a = 4.084$ Å:

```python
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
a = 4.084 * BOHR_PER_ANGSTROM
lattice = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
centre = 0.5 * lattice.sum(axis=0)
lih = vq.PeriodicSystem(dim=3, lattice=lattice, unit_cell=[
    vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, centre.tolist())])
basis = vq.BasisSet(lih.unit_cell_molecule(), "sto-3g")

for mesh in [(1, 1, 1), (2, 2, 2)]:
    res = vq.run_periodic_job(lih, basis, method="dlpno-ccsd(t)",
                              kpoints=mesh, output="lih_" + "x".join(map(str, mesh)),
                              progress=False)
    c = res.correlation
    print(mesh, f"{c.e_hf_per_cell:.10f}", f"{c.e_corr_per_cell:.10f}",
          f"{c.e_t_per_cell:.2e}", f"{c.e_total_per_cell:.10f}")
```

| Mesh | E_HF per cell, Ha | E_corr (CCSD) per cell, Ha | E(T) per cell, Ha | E_total per cell, Ha | Pairs / avg PNOs | Wall, s |
|---|---:|---:|---:|---:|---:|---:|
| Gamma | -8.3351540806 | -0.0068114824 | -2.5e-06 | -8.3419680624 | 3 / 4.0 | 40 |
| (2, 2, 2) | -7.9222475699 | -0.0162333820 | -2.8e-04 | -7.9387617758 | 92 / 14.1 | 77 |

The published count-only frozen-core convention (ORCA 6.1) freezes no
orbital on lithium, so both occupied orbitals of the cell, the Li 1s and the
hydride pair, are correlated: three pairs at Gamma, 92 on the eight-cell
supercell. Two lessons carry over from H2 and are sharper here. The Gamma-point reference of a compact ionic cell is 0.41 Ha per cell
away from the (2, 2, 2) value: never correlate a Gamma-only reference of a
small primitive cell and call it a crystal. And the triples correction is a
hundred times larger on the (2, 2, 2) supercell than at Gamma, because the
supercell holds triples that span neighbouring cells.

## An open-shell cell

A cell with `multiplicity > 1` runs on a KUHF reference and the spin-orbital
DLPNO-UCCSD solver, with spin-pure PNOs and the semicanonical local (T0)
triples. Nothing changes in the call. Here an H3 doublet in an 8 bohr cube:

```python
h3 = vq.PeriodicSystem(3, np.diag([8.0, 8.0, 8.0]), [
    vq.Atom(1, [3.3, 4.0, 4.0]), vq.Atom(1, [4.7, 4.0, 4.0]),
    vq.Atom(1, [4.0, 6.0, 4.0])], multiplicity=2)
basis = vq.BasisSet(h3.unit_cell_molecule(), "sto-3g")
for mesh in [(1, 1, 1), (1, 1, 2)]:
    res = vq.run_periodic_job(h3, basis, method="dlpno-ccsd(t)", kpoints=mesh,
                              output="h3_" + "x".join(map(str, mesh)), progress=False)
    c, pd = res.correlation, res.periodic_dlpno
    print(mesh, c.reference_kind, pd.n_occ_alpha, pd.n_occ_beta, c.backend,
          f"{c.e_hf_per_cell:.10f}", f"{c.e_corr_per_cell:.10f}")
```

| Mesh | Reference | Occupied alpha / beta (supercell) | E_HF per cell, Ha | E_corr per cell, Ha | E_total per cell, Ha | Wall, s |
|---|---|---|---:|---:|---:|---:|
| Gamma | UHF | 2 / 1 | -1.4867525495 | -0.0127577114 | -1.4995102609 | 16 |
| (1, 1, 2) | UHF | 4 / 2 | -1.4719338411 | -0.0143995238 | -1.4863333650 | 13 |

In this minimal basis no triple survives the TNO construction: the `.out`
reports the triples as "not executed for this result" and the (T) per cell is
zero. In a double-zeta basis the open-shell (T0) runs as on molecules. The
route key becomes `periodic-gdf-dlpno-uccsd(t)`, and the manifest gains
`dlpno_reference_kind`, `dlpno_n_occ_alpha` and `dlpno_n_vir_alpha`. Pass
`dlpno_ccsd_options=LocalUCCSDOptions(...)` (from
`vibeqc.dlpno.uccsd_local_solver`) to change the open-shell solver's
controls; passing it for a singlet cell selects the UHF reference.

## What the route refuses

Every refusal fires before the reference SCF, so a request the route cannot
honour costs nothing. These are the measured messages, abbreviated:

```python
for kw in [dict(functional="pbe"), dict(jk_method="bipole"),
           dict(kpoints=(2, 2, 2), symmetry_reduce_k=True),
           dict(smearing="fermi", smearing_temperature=0.01),
           dict(optimize=True)]:
    try:
        vq.run_periodic_job(system, basis, method="dlpno-ccsd(t)",
                            output="refused", progress=False, **kw)
    except (ValueError, NotImplementedError) as exc:
        print(type(exc).__name__, str(exc)[:110])
```

| Request | Exception | Reason in the message |
|---|---|---|
| `functional="pbe"` | `ValueError` | DLPNO-CCSD(T) correlates a Hartree-Fock reference; drop `functional=`. |
| `jk_method="bipole"` or `"gpw"` | `NotImplementedError` | The front door runs on the GDF route only; a BIPOLE reference has no certified same-source correlation factors. |
| `symmetry_reduce_k=True` | `NotImplementedError` | A symmetry-reduced reference keeps its orbitals on the irreducible wedge only. Run the full mesh. |
| a `KPoints` object for `kpoints=` | `ValueError` | Pass a mesh tuple; a `KPoints` object may carry a half shift or a symmetry reduction. |
| smearing | `NotImplementedError` | The engine needs an integer-aufbau determinant. |
| `optimize=True` or `hessian=True` | `NotImplementedError` | DLPNO-CCSD(T) gradients do not exist. |
| `dim=1` or `dim=2` | `NotImplementedError` | The GDF route runs on 3-D cells; embed a chain or slab in a vacuum box. |
| `LocalCCSDOptions(localise="pipek-mezey")` | `ValueError` | The adapter localises the occupied orbitals itself; the engine's localiser is not periodic-safe. |

The last one is worth understanding: the molecular engine localises with
Boys orbitals, whose dipole integrals are not defined on a torus. The
adapter therefore localises with Pipek-Mezey populations, which are, and
asks the engine to leave the orbitals alone.

## Theory

**Bloch orbitals and the BvK supercell.** A Gamma-centred mesh of $N_k$
points samples exactly the crystal momenta that are periodic on a supercell
of $N_k$ cells, the Born-von Karman torus. The canonical orbitals of that
torus are the Bloch orbitals $\psi_{n\mathbf k}$ of the mesh, and real
supercell orbitals follow from the inverse Bloch transform of the per-k Fock
and overlap blocks,

$$
\mathbf F(\mathbf T) = \frac{1}{N_k}\sum_{\mathbf k} e^{-i\mathbf k\cdot\mathbf T}\,\mathbf F(\mathbf k),
\qquad
\mathbf T \in \text{supercell translations},
$$

so the folded problem is a real molecular problem on the supercell whose
orbital energies equal the union of the per-k spectra. The adapter checks
that equality to 1e-6 Ha and refuses a fold that breaks it
({vibeqc-cite}`monkhorst_pack_1976` for the meshes,
{vibeqc-cite}`mcclain_berkelbach_periodic_cc_2017` for coupled cluster on
the same torus).

**Local correlation on the torus.** The occupied supercell orbitals are
localised by maximising the Pipek-Mezey functional
$\sum_i \sum_A (Q_A^{ii})^2$ of atomic orbital populations
({vibeqc-cite}`pipek_mezey_1989`). Pair distances for `coupling_radius` and
the local triples use the minimum image on the supercell. From there on the
calculation is the molecular DLPNO-CCSD(T) of Riplinger and Neese
({vibeqc-cite}`riplinger_dlpno_2013`, {vibeqc-cite}`riplinger_triples_2013`):
semicanonical MP2 pair densities define the pair natural orbitals,

$$
\mathbf D^{ij} = \frac{1}{1+\delta_{ij}}\left(\tilde{\mathbf T}^{ij\dagger}\mathbf T^{ij} + \tilde{\mathbf T}^{ij}\mathbf T^{ij\dagger}\right),
\qquad
\mathbf D^{ij}\mathbf d^{ij}_a = n^{ij}_a\,\mathbf d^{ij}_a ,
$$

PNOs with occupation $n^{ij}_a$ below `tcut_pno` are dropped and their MP2
energy is added back, weak pairs below `tcut_pairs` are kept at the MP2
level, and the CCSD amplitude equations are solved with every term in the
PNO space of the amplitude that carries the contracted index. The default
closed-shell `(T1)` evaluates the canonical (T) on the rotated amplitudes
({vibeqc-cite}`guo_dlpno_t1_2018`); the open-shell route uses the
semicanonical (T0) of {vibeqc-cite}`guo_openshell_dlpno_triples_2020` on the
spin-orbital solver.

**Finite-size behaviour.** The reference and the correlation energy per cell
converge at different rates. The HF exchange error of the BvK-ewald gauge
decays as $1/N_k$ after the Madelung correction of the $\mathbf q = 0$
singularity ({vibeqc-cite}`gygi_baldereschi_exx_1986`), while the
correlation energy per cell converges as the supercell admits the long-range
pairs and the low-momentum excitations, at a rate that depends on the
system and is in general different from the reference's. Use
the same mesh for every structure you compare, converge each part on its
own series, and extrapolate the two separately when you need a bulk value.
Recent work on periodic local correlation discusses these limits on larger
cells ({vibeqc-cite}`nejad_periodic_dlpno_pbc_2025`,
{vibeqc-cite}`zhu_periodic_dlpno_megacell_2025`).

## Variations

- **Thresholds.** `dlpno_thresholds="tight"` or `"loose"` changes the PNO,
  pair and domain cuts exactly as on `run_job`. Compare a tight run with
  the default on your smallest mesh before trusting a mesh series.
- **Frozen core.** `frozen_core=False` correlates all electrons;
  `frozen_core=2` freezes two orbitals per cell.
- **Local triples.** `dlpno_ccsd_options=LocalCCSDOptions(triples_mode="local")`
  runs the compiled local (T0) kernel, whose cost grows more slowly with
  the supercell than the canonical `(T1)`.
- **The library route.** `vibeqc.periodic_dlpno.run_periodic_dlpno_ccsd(system,
  "def2-svp", kmesh=(2, 2, 2))` returns the same numbers without the output
  files, which is convenient inside a mesh-convergence script.

## Pitfalls

- **A Gamma-only reference of a small cell is not a crystal.** The LiH and
  H2 tables above show reference errors of 0.1 to 0.4 Ha per cell at Gamma.
- **Supercell cost.** The engine solves a supercell of $N_k$ cells, and the
  default `(T1)` costs a canonical (T) on that supercell. A (3, 3, 3) mesh
  of a 10-atom cell is a 270-atom molecular problem.
- **Minimal-basis triples.** In STO-3G the (T) correction of H2 is zero or
  close to it. Read triples from at least a double-zeta basis.
- **Fock convention.** The amplitude equations keep the Ewald exchange term
  of the SCF Fock operator. A code that strips it, such as PySCF's default
  `KRCCSD`, reports a lower finite-mesh correlation energy with the same
  thermodynamic limit; compare like with like.

## See also

- [Local correlation on crystals](../user_guide/periodic_local_correlation.md),
  the user guide with the option tables, the adapters and the evidence.
- [DLPNO methods](../user_guide/dlpno_mp2.md) for thresholds, triples modes
  and residual domains.
- [Local correlation with DLPNO](dlpno_local_correlation.md), the molecular
  tutorial.
- [Multi-k SCF](../user_guide/multi_k_scf.md) for the reference and its
  finite-size tail.
- [Design: one DLPNO engine for molecules and crystals](../design_dlpno_correlation_source.md).
