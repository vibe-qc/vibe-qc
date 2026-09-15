---
myst:
  html_meta:
    "description": "The cyclic cluster model (CCM) in vibe-qc: the project's signature real-space approach to periodic systems. What a cyclic cluster is (Wigner-Seitz image folding + Madelung embedding), how to run bulk, surface, and adsorption calculations at the semi-empirical MSINDO level today, and the ab-initio CCM roadmap. Lineage: Bredow-Geudtner-Jug 2001 and Peintinger-Bredow 2014."
    "og:title": "vibe-qc tutorial - the cyclic cluster model (CCM)"
---

# The cyclic cluster model (CCM)

The cyclic cluster model is vibe-qc's **signature approach to periodic
systems**, the feature the whole project was built around. Where the
[k-point routes](periodic_methods_compared.md) (GDF, BIPOLE, GPW/GAPW)
sample the Brillouin zone in *reciprocal* space, CCM stays in **real
space**: it treats a crystal as a single finite *cyclic* cluster and
recovers the infinite solid as that cluster grows. The two pictures are the
same physics seen from opposite sides, the cluster-size $\leftrightarrow$
k-mesh-density equivalence made concrete in the
[H-chain ancestor tutorial](h8_chain_ccm_ancestor.md).

```{admonition} What ships today, and what is coming
:class: important
This tutorial covers the production semi-empirical MSINDO route and the
ab-initio theory. The cyclic-cluster boundary for the other semiempirical
Hamiltonians is implemented as experimental adapters: the narrow 1-D H/C
DFTB0-SECCM route, SCC-DFTB, and PM6 run
full Wigner-Seitz weighted supercell calculations for closed-shell clusters
(SCC-DFTB adds an opt-in Madelung/Ewald embedding for charged cells and a
fixed-topology analytic gradient, including the explicit embedding
derivative). OM2/OM3 and GFN2-xTB also run multi-replica weighted
supercells. OM1-SECCM fails closed because its defining analytic
core-valence ECP is not implemented. Each executable adapter reproduces its
molecular driver in the 1x1x1 molecular limit. The supported
envelopes and call shapes are in the [semiempirical user
guide](../user_guide/semiempirical.md). The ab-initio CCM is now implemented as two experimental,
actively-developed lines: **Γ-CCM / `aiccm2026dev-a`** (the
union-and-weight/Wigner--Seitz integral-weighting approach, with
HF/KS/MP2/UMP2/CCSD(T)) and **χ-CCM / `aiccm2026dev-b`** (the
finite-translation-group character approach, with SCF and finite-torus
correlation). Γ-CCM and χ-CCM[^xccm-convention] are compared at a declared
common exchange-q=0 convention. Both are separately documented for a
prospective construction-level comparison, but the current reportable route
map is empty - see [Γ-CCM](../aiccm2026dev_a.md) and
[χ-CCM](../user_guide/aiccm2026dev_b.md). Production-grade correlated
scaling remains on the roadmap. The A namespace also contains neutral
fitted-torus UCCSD(T) and DLPNO controls; those controls are neither Γ-CCM
nor χ-CCM results.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.
```

## What a cyclic cluster is

Take a finite cluster carved from the crystal, a supercell of $N$ atoms.
In an ordinary cluster the atoms at the edge have the wrong environment
(they face vacuum). CCM fixes this by giving **every atom a Wigner-Seitz
(WS) cell** of its periodic surroundings: each *other* atom of the cluster
contributes through its single nearest translational image that falls
inside that WS cell. Atoms sitting exactly on a WS face are shared between
neighbours with a fractional weight, and the long-range electrostatics that
a finite sum would miss are restored by a classical **Ewald (Madelung)
embedding**.

The construction has one defining validity rule: for every atom, the WS
weights of all the images inside its cell must sum to $N-1$: every *other*
atom appears exactly once, counting fractional shares. A cluster that
violates this is simply not a valid cyclic cluster, and vibe-qc says so
rather than returning a wrong number. As the cluster grows, the WS
environment of each atom converges and the CCM energy approaches the true
periodic limit.

This is what makes CCM attractive for **defects and surfaces**: the cluster
*is* the system, there is no Brillouin-zone sampling to converge, and a
local perturbation (a vacancy, an adsorbate) is treated directly in real
space against a correctly embedded periodic background.

## Running CCM: bulk MgO

A bulk calculation passes three lattice vectors (so CCM3D). Coordinates and
lattice vectors are in Angstrom; `madelung=True` switches on the Ewald
embedding (recommended for ionic systems):

```python
from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

a = 4.21                                          # MgO lattice constant (Angstrom)
fcc = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
Z = [12]*4 + [8]*4                                # Mg4 O4 conventional cell
C = ([[fx*a, fy*a, fz*a] for fx, fy, fz in fcc]
     + [[(fx+0.5)*a, fy*a, fz*a] for fx, fy, fz in fcc])
trans = [(a, 0, 0), (0, a, 0), (0, 0, a)]         # 3 vectors -> CCM3D (bulk)

res = run_ccm(Z, C, trans, madelung=True)
print(res.total_energy, res.n_iter, res.converged)
```

```text
bulk MgO rocksalt CCM3D:  E = -66.20383741 Ha   (11 cycles)   converged=True
```

The number of lattice vectors selects the dimensionality: **1 vector $\to$
CCM1D** (a polymer), **2 $\to$ CCM2D** (a surface), **3 $\to$ CCM3D**
(bulk). Every CCM energy vibe-qc reports here is validated out-of-process
against reference MSINDO to better than 1 µHa (CLAUDE.md § 10).

The same job through the high-level `run_job` entry point, where the
cluster atoms live in a `Molecule` and the lattice rides in `CCMOptions`:

```python
import vibeqc as vq
from vibeqc.semiempirical.methods.msindo_ccm import CCMOptions

res = vq.run_job(
    mol,                                          # Molecule with the Mg4 O4 cluster
    method="ccm",
    ccm_options=CCMOptions(
        translations=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        madelung=True,
    ),
)
```

## The flagship: surfaces and adsorption

CCM's real strength is surface chemistry, vibe-qc's flagship workflow. A
**two-vector cluster is a periodic slab** (CCM2D), and an adsorption energy
is just the difference of three CCM/MSINDO single points,

$$
E_\text{ads} = E[\text{slab}+\text{adsorbate}]
 - E[\text{slab}] - E[\text{adsorbate, gas}].
$$

For a water molecule on the MgO(100) surface:

```python
from vibeqc.semiempirical.methods import msindo
from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

trans = [(2*a, 0, 0), (0, 2*a, 0)]                # 2 vectors -> CCM2D (surface)
e_slab  = run_ccm(Z_slab, C_slab, trans, madelung=True).total_energy
e_sysw  = run_ccm(Z_slab + [8, 1, 1], C_slab + water_xyz, trans, madelung=True).total_energy
e_gas   = msindo.run_msindo([8, 1, 1], water_gas_xyz).total_energy
e_ads   = e_sysw - e_slab - e_gas
```

```text
slab E      = -66.439593 Ha
slab+H2O E  = -83.468520 Ha
gas  H2O E  = -17.018209 Ha
E_ads       = -0.010718 Ha = -6.73 kcal/mol     (fixed geometry)
```

The forces that relax the adsorbate come from `ccm_gradient_fd`, and
`ccm_optimize` relaxes the adsorbate on a frozen slab to the relaxed
binding energy. The complete worked script (bulk, the MgO(100) slab, the
adsorption energy, the gradient, and a relaxed adsorbate) is
[`examples/semiempirical/11_msindo_ccm.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/semiempirical/11_msindo_ccm.py).

## Scope today

CCM ships at the semi-empirical MSINDO level, and each boundary is a clean
error rather than a silent wrong answer (CLAUDE.md § 7):

- **Closed-shell (RHF) only**, a cell with `multiplicity != 1` raises.
- **Single point through `run_job`**, relax geometries with
  `ccm_optimize` (finite-difference forces, fixed Wigner-Seitz cells).
- **Elements H to Zn**, the MSINDO engine's supported set.

## How the ab-initio CCM works

The semi-empirical engine above and the ab-initio track share one idea:
impose the cyclic Born-von-Kármán boundary conditions (the same conditions
derived in [k-points, the Brillouin zone, and Bloch
phase](kpoints_brillouin_bloch.md)) directly on a finite cluster, and
evaluate every interaction in real space at the single point $\Gamma$. What
changes between the semi-empirical and the ab-initio case is *which*
integrals appear and *how far* the cyclic sum has to reach.

**The cyclic arrangement.** The $N = N_1 N_2 N_3$ atoms carved from the
crystal are the *real* atoms of the cluster, the unit cell; the translation
vectors $\mathbf{t}$ generate *virtual* images of them. Closing the cluster
into a ring (1D), a torus (2D), or a hypertorus (3D) replaces each atom's
open environment with the periodic one it would have in the solid. Because
translational symmetry survives, the atomic-orbital basis maps onto a Bloch
basis,

$$
\phi_\mu^{\mathbf{k}} = \frac{1}{\sqrt{N}}\sum_{\mathbf{t}}^{N}
e^{i\mathbf{k}\cdot\mathbf{t}}\,\mu^{\mathbf{t}},
$$

and at $\mathbf{k}=0$, keeping only the reference cell,
$\phi_\mu^{\mathbf{k}} \approx \mu^{0}$: a cyclic-cluster basis is directly
comparable to a molecular one, which is what lets molecular post-HF
machinery carry over unchanged. The $N_j$ are odd, since translation always
adds matched $+\mathbf{t}_j$ and $-\mathbf{t}_j$ images.

**The Wigner-Seitz weighting (the heart of the method).** Each atom
interacts only out to the boundary of its own Wigner-Seitz supercell, a hard
cutoff at $\pm\tfrac12\mathbf{t}$. An atom sitting exactly on that boundary
is shared between equidistant images, so a naive sum counts its interaction
more than once. The fix that keeps the model exact is to weight every such
interaction by the number $n_{\nu'}$ of translationally equivalent images of
that atom inside the Wigner-Seitz supercell,

$$
\omega_{\mu\nu'} = \frac{1}{n_{\nu'}},
\qquad
S_{\mu\nu} = \sum_{\nu'}^{n_{\nu'}} \omega_{\mu\nu'}\,\langle\mu|\nu'\rangle.
$$

This one rule does two things at once: it enforces charge neutrality and it
preserves the point symmetry of the cluster. It is the integral-level form
of the $N-1$ validity rule stated above, every other atom contributes
exactly once, counting fractional shares.

**Why the ab-initio reach is longer.** In the semi-empirical CCM only one-
and two-center terms appear, and the $\pm\tfrac12\mathbf{t}$ reach suffices.
The ab-initio Fock operator adds three-center (nuclear attraction) and
four-center (Coulomb and exchange) integrals, whose weights are built by
*averaging* the two-center weights over the union of the participating
atoms' Wigner-Seitz cells. The three-center term needs a full
$\pm\mathbf{t}$ translation of the cluster; the four-center term reaches
$\pm\tfrac32\mathbf{t}$ and needs translations out to $\pm2\mathbf{t}$. This
growth of the interaction radius from $\pm\tfrac12\mathbf{t}$ to
$\pm\tfrac32\mathbf{t}$ is the central complication the ab-initio CCM has to
solve and that the semi-empirical model never faced.

With the weighted integrals in hand, the cyclic Hartree-Fock-Roothaan
equations and the total energy take their familiar molecular form,

$$
\mathbf{F}^{\mathrm{CCM}}\mathbf{C} = \mathbf{S}^{\mathrm{CCM}}\mathbf{C}\,\mathbf{E},
\qquad
\mathbf{F}^{\mathrm{CCM}} = \mathbf{h}^{\mathrm{CCM}}
+ \sum_n\bigl(2\mathbf{J}_n^{\mathrm{CCM}} - \mathbf{K}_n^{\mathrm{CCM}}\bigr),
$$

and are solved self-consistently exactly as for a molecule. That structural
identity, a molecular SCF over a set of weight-corrected integrals, is
precisely why the v2.x track can climb from HF to MP2 to coupled cluster:
each correlated method is applied to what is, formally, a molecular
reference. The full derivation, including the three- and four-center
weighting tables, is in the project author's dissertation (see the
[references](#references) below).

## The road to ab-initio CCM

The semi-empirical engine is the proving ground; the ab-initio CCM is
now implemented as two independent, actively developed lines that are
being compared head-to-head:

- **[Γ-CCM / `aiccm2026dev-a`](../aiccm2026dev_a.md)** - the
  union-and-weight/Wigner-Seitz integral-weighting CCM. Ships with HF,
  KS-DFT (any libxc functional), MP2, UMP2, CCSD(T), analytic gradients,
  and a suite of properties (gap, charges, dipole, forces). The same Python
  namespace also exposes neutral fitted-torus UCCSD(T) and DLPNO controls;
  their module location does not assign them the Γ-CCM construction.

- **[χ-CCM / `aiccm2026dev-b`](../user_guide/aiccm2026dev_b.md)** -
  the finite-character (Γ-centred character-mesh) CCM. Ships with RHF/RKS/UHF/UKS
  via four-center, RI, and RIJCOSX backends in 3D, plus finite-torus RI-MP2,
  DLPNO-MP2, DLPNO-CCSD, and DLPNO-CCSD(T) in 3D, with explicit finite-torus
  convention recording. Every 1D/2D B absolute-energy backend fails closed
  pending a shared wire/slab Coulomb convention.

Both lines are experimental and under active head-to-head study, but no
Γ-CCM/χ-CCM approach delta is currently reportable. Their separate routes are
benchmarked
against CRYSTAL23 on the 28-system AICCM-2026 benchmark test set
([`studies/aiccm-2026/`](https://github.com/vibe-qc/vibe-qc/tree/main/studies/aiccm-2026)).
The full track is in the [roadmap](../roadmap.md).

## References

The cyclic cluster model in vibe-qc follows:

- Bredow, Geudtner and Jug, *J. Comput. Chem.* **22**, 89 (2001), the
  semi-empirical CCM that the current engine ports.
- Peintinger and Bredow, *J. Comput. Chem.* **35**, 839 (2014),
  [doi:10.1002/jcc.23550](https://doi.org/10.1002/jcc.23550), the
  cluster-choice problem and the ab-initio CCM foundation.

The ab-initio derivation above, the Wigner-Seitz weighting of the two-,
three-, and four-center integrals, is worked out in full in the primary
source: M. F. Peintinger, *Elektronenstruktur von Molekülkristallen*
(dissertation, Rheinische Friedrich-Wilhelms-Universität Bonn, 2013), of
which the 2014 paper is the published account.

A `method="ccm"` run writes the two papers into the `.bibtex` sidecar
automatically; see [auto_citations](auto_citations.md).

## See also

- [CCM reference](../user_guide/cyclic_cluster_model.md), the conceptual and
  API reference (keywords, defaults, and the theory in depth).
- [The H-chain CCM ancestor](h8_chain_ccm_ancestor.md), the cyclic-cluster
  $\leftrightarrow$ Born-von-Karman / k-points equivalence on a 1D chain.
- [MSINDO](msindo.md), the semi-empirical engine CCM is built on.
- [k-points, the Brillouin zone, and Bloch phase](kpoints_brillouin_bloch.md),
  the reciprocal-space alternative to CCM.
- [Choosing a periodic method](periodic_methods_compared.md), the k-point
  production routes (GDF, BIPOLE, GPW/GAPW).
