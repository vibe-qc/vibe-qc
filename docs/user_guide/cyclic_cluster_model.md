---
myst:
  html_meta:
    "description": "Reference description of the cyclic cluster model (CCM) in vibe-qc: a real-space, Born-von-Karman approach to periodic systems. The Wigner-Seitz image-folding construction, Madelung embedding, convergence to the periodic limit, the current semi-empirical MSINDO implementation with its keywords and defaults, scope limits, and the ab-initio CCM roadmap."
    "og:title": "vibe-qc - the cyclic cluster model (CCM) reference"
---

# The cyclic cluster model (CCM)

The cyclic cluster model is vibe-qc's **signature approach to periodic
systems**. This page is the conceptual and API reference; for a hands-on,
step-by-step run see the
[tutorial](../tutorial/cyclic_cluster_model.md).

```{admonition} Status: experimental
:class: warning

The semiempirical cyclic cluster model (SECCM) is an experimental route
overall. The MSINDO route is the most mature member: its closed-shell
energies are validated within the documented scope (closed-shell, H to Xe,
out-of-process MSINDO parity to better than 1 uHa). The DFTB0, SCC-DFTB,
GFN2-xTB and PM6 adapters are gated experimental and library-only: they are
reached through `vibeqc.semiempirical.seccm`, not through a production
runner route, and each route plan carries `maturity = "experimental"`
(`vibeqc.semiempirical.routes`), which the runner writes into the `.system`
manifest as `route_maturity`. SECCM numbers are research output, not
validated results, and the route is not part of the release gate. The
ab-initio lines (Gamma-CCM and chi-CCM, `method="aiccm"`) are experimental
as well; see [AICCM](aiccm.md). The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

## Overview

CCM computes the electronic structure of a crystal, surface, or polymer by
treating a finite cluster under cyclic (Born-von-Karman) boundary
conditions. The historical formulation evaluates that problem in real
space. For any one already specified block-circulant finite Hamiltonian, its
real-supercell and full Gamma-centred character-mesh representations are
exactly Fourier equivalent. The independent χ-CCM implementation
(`jk_method="aiccm2026dev-b"`) uses that internal identity. It does not identify
the χ construction with the union-and-weight Γ-CCM construction. The infinite-crystal result is
recovered by enlarging the cluster. CCM is the original motivating feature
of vibe-qc and its intended long-term flagship for periodic chemistry.

Because the cluster *is* the system, CCM is naturally suited to localised
phenomena (point defects, surfaces, adsorbates), where a single perturbation
sits inside a correctly embedded periodic background. It is the real-space
counterpart to the [k-point routes](periodic_methods.md) (GDF, BIPOLE,
GPW/GAPW): the same physics approached from opposite sides, since the
cluster-size limit and the k-mesh-density limit coincide.

## How a cyclic cluster is built

### Wigner-Seitz image folding

Each atom of the cluster carries a Wigner-Seitz (WS) cell describing its
periodic surroundings. Every *other* atom contributes to that atom through
its single nearest translational image that falls inside the WS cell. An
atom lying on a WS face is shared between the equidistant faces with a
fractional weight ($1/n$ for $n$ equidistant faces): keeping only one of the
tied images would give the central atom a non-symmetric interaction region
with a dipole moment (Bredow, Geudtner & Jug, J. Comput. Chem. 22, 89 (2001),
pp. 90-91). "Equidistant" is decided with MSINDO's `neighbors.f` tolerance,
$10^{-5}$ bohr on the image distance (a physical length: a topology built
from angstrom coordinates with `length_unit="angstrom"` converts it to
$5.3 \times 10^{-6}$ Å rather than reusing the raw number, #669), so a
geometry given to finite
precision (a coordinate file with six decimals, or coordinates and
translations converted with different CODATA constants, which differ by
$4 \times 10^{-10}$) keeps every shared image; a pure roundoff tolerance
silently dropped them and moved the MgO (2,2,2) SECCM energy by 0.06 Ha
(issues #592, #593). vibe-qc's construction is otherwise a faithful port of
MSINDO's neighbour-list routine.

The construction is valid only if, for every atom, the WS weights of all the
images inside its cell sum to $N-1$: every other atom of the cluster appears
exactly once, counting fractional shares. vibe-qc enforces this rule and
rejects a cluster that violates it rather than returning a wrong number.

In equations: let $A$ be the central atom, $B$ another cluster atom, and
$\mathbf g$ a supercell translation such that the image $\mathbf R_B +
\mathbf g$ lies in or on the Wigner-Seitz cell of $A$. The image weight and
the sum rule are

$$
w_{AB\mathbf g} = \frac{1}{n_{AB\mathbf g}},
\qquad
\sum_{B\neq A}\sum_{\mathbf g\in\mathrm{WS}(A)} w_{AB\mathbf g} = N - 1
\quad\text{for every } A,
$$

where $n_{AB\mathbf g}$ is the number of equidistant faces the image sits on
(1 in the interior, 2 on a face, up to 8 at a corner). Every two-centre
quantity of the semiempirical Hamiltonian is then folded through these
weights. For the MSINDO engine the resonance block, the core attraction and
the two-electron Coulomb term $\gamma$ between atom $A$ and atom $B$ become

$$
\gamma^{\mathrm{CCM}}_{AB} = \sum_{\mathbf g\in\mathrm{WS}(A)} w_{AB\mathbf g}\;
\gamma_{AB}\big(|\mathbf R_A - \mathbf R_B - \mathbf g|\big),
\qquad
H^{\mathrm{CCM}}_{\mu\nu} = \sum_{\mathbf g\in\mathrm{WS}(A)} w_{AB\mathbf g}\;
H_{\mu\nu}\big(\mathbf R_A,\ \mathbf R_B + \mathbf g\big),
\quad \mu\in A,\ \nu\in B,
$$

so an atom sees each neighbour once, through its nearest image, and a
neighbour tied between two faces contributes half from each. This is the
prescription of Bredow, Geudtner and Jug
({vibeqc-cite}`bredow_geudtner_jug_ccm_2001`), and it is the same idea as
Evjen's fractional weighting of boundary charges in Madelung sums
({vibeqc-cite}`evjen_1932`), applied to integrals instead of point charges.
For the non-orthogonal adapters the overlap matrix is stitched with the same
weights (Peintinger and Bredow, {vibeqc-cite}`peintinger_ccm_2014`, eq. 5),
and the ab-initio generalisation carries the pair weights into three- and
four-centre integrals; the approach to the bulk limit with cluster size was
mapped for INDO by Noga, Baňacký and co-workers
({vibeqc-cite}`noga_ccm_1999`), and the Kohn-Sham and DFTB versions by
Janetzko and co-workers ({vibeqc-cite}`janetzko_ccm_2008`) and Claff
({vibeqc-cite}`claff_dftb_ccm_2012`).

### Madelung embedding

A finite cluster misses the conditionally-convergent long-range Coulomb tail
of the infinite lattice. CCM restores it with a classical Madelung embedding,
summed in 1, 2, or 3 dimensions according to the cluster's periodicity. With
the net atomic charges $q_B = Z_B^{\mathrm{core}} - P_{BB}$ (core charge
minus the diagonal population in the orthogonalised MSINDO basis), the
embedding potential at atom $A$ is the full Ewald potential of the periodic
point-charge lattice minus the part the Wigner-Seitz neighbours already
contribute through their explicit two-centre integrals,

$$
V^{\mathrm{Mad}}_A = \sum_B q_B\,M_{AB} \;-\;
\sum_{(B,\mathbf g)\in\mathrm{EWS}(A),\ d>0} \frac{w_{AB\mathbf g}\,q_B}{d_{AB\mathbf g}},
\qquad
d_{AB\mathbf g} = |\mathbf R_A - \mathbf R_B - \mathbf g|,
$$

where $M_{AB}$ is the geometry-only Ewald Madelung matrix (reciprocal, direct
erfc and background terms, `madelkonst.f`) and the Ewald neighbour set
$\mathrm{EWS}(A)$ is the Fock set plus the atom itself, because the periodic
images of $A$'s own charge act on $A$. The potential enters the SCF
self-consistently as a diagonal Fock shift and a classical energy,

$$
F_{\mu\mu} \;\to\; F_{\mu\mu} - V^{\mathrm{Mad}}_A \quad (\mu\in A),
\qquad
E_{\mathrm{Mad}} = \frac12\sum_A Z_A^{\mathrm{core}}\,V^{\mathrm{Mad}}_A ,
$$

so the cluster electrons feel the infinite ionic lattice while the double
counting between the explicit neighbours and the Ewald sum is removed
exactly. The long-range term matters for metal oxides and other ionic
crystals, where Janetzko and co-workers measured its effect on cyclic-cluster
energies ({vibeqc-cite}`janetzko_ccm_long_range_2002`). In the
high-level MSINDO `run_job` entry point, `CCMOptions` controls the embedding
and defaults `madelung=True`. The direct `run_ccm`,
gradient, and optimization helpers default it to `False`, so callers of those
APIs must opt in explicitly. The embedding is recommended for ionic systems.

The MSINDO kernel differs by dimension, and only the 2-D and 3-D branches are
Ewald summations. 3-D uses a reciprocal + direct (erfc) Ewald split with
tin-foil boundary conditions, and 2-D uses the Parry/Heyes 2-D Ewald
(`madelkonst.f`). **1-D is not an Ewald sum**: it is the historical truncated
direct lattice sum of `ccm1dmadelsum.f`, which adds only the ±1 and ±2
translational images of each Wigner-Seitz neighbour's point charge. It is kept
frozen for MSINDO compatibility, so a 1-D cluster must be electroneutral for
the truncated sum to be meaningful, and a 1-D run cites no Ewald method. The
SECCM adapters (SCC-DFTB, GFN2-xTB) instead run a converged
background-corrected Parry-type wire Ewald at 1-D; see
[Semiempirical methods](semiempirical.md).

The experimental OM2/OM3-SECCM adapter does not yet implement that
self-consistent embedding. The primary CCM derivations require long-range
interactions for ionic crystals, and ionicity cannot be classified reliably
before the SCF calculation. OMx topologies therefore fail closed by default
whenever they contain multiple replicas, wrapped image records, or fractional
image ownership. An explicit
`allow_truncated_electrostatics=True` permits only an algorithm-mechanics
probe with the long-range contribution omitted; its result records the
acknowledgement and is not a quantitative solid-state prediction.

### Indefinite cyclic overlap

For the non-orthogonal semiempirical Hamiltonians (DFTB0, SCC-DFTB,
GFN2-xTB) the cyclic overlap matrix is assembled with the same WS image
weights as the Hamiltonian (Peintinger and Bredow, J. Comput. Chem. 35
(2014), eq. 5). That stitched matrix is a Gram matrix only in the
molecular limit; for small clusters or cells whose atoms sit on WS faces
it can acquire non-positive eigenvalues, which makes the generalized
eigenproblem ill-posed and the SCC iteration anti-screen. The paper's
remedy is canonical orthogonalization: the non-positive subspace is
screened, the solve proceeds in the orthonormalized basis, and the
coefficients are back-rotated. The SECCM adapters apply exactly this
screening whenever the cyclic overlap fails the positive-definiteness
gate; cells that pass the gate keep the legacy generalized solve
bit-for-bit. If the screened subspace cannot hold the occupied manifold
the adapter still fails closed with `overlap matrix is not positive
definite`. This is the standard treatment of near-linear dependence in
quantum chemistry and the paper's own prescription for the C-point
approximation breakdown.

### Convergence to the periodic limit

As the cluster grows, the WS environment of each atom converges to the bulk
environment and the CCM total energy approaches the true periodic limit. The
cluster-size convergence series is the CCM analogue of k-mesh convergence in
a Bloch calculation; the equivalence is made explicit in the
[H-chain ancestor tutorial](../tutorial/h8_chain_ccm_ancestor.md).

## Running CCM

CCM ships today at the **semi-empirical MSINDO** level. There are two entry
points. The high-level one goes through `run_job`, with the cluster atoms in
a `Molecule` and the lattice in `CCMOptions`:

```python
import vibeqc as vq
from vibeqc.semiempirical.methods.msindo_ccm import CCMOptions

res = vq.run_job(
    mol,
    method="ccm",
    ccm_options=CCMOptions(
        translations=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        madelung=True,
    ),
)
```

The direct one is `run_ccm`, taking the geometry and lattice as arrays:

```python
from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

res = run_ccm(atomic_numbers, coords_angstrom, translations_angstrom,
              madelung=True)
```

With `citations=True`, the high-level `run_job` path takes its
route-specific citation inputs from the concrete runtime route plan. The
generated `.out`, `.bibtex`, `.references`, and `.system` files therefore
include both MSINDO engine papers and the CCM construction papers, plus only
the dimensional electrostatics references for the kernel that actually ran.
The direct `run_ccm` helper returns a result but does not coordinate job
output files.

The same Wigner-Seitz topology also drives the experimental SECCM adapters
for the other semiempirical Hamiltonians. DFTB0-SECCM is a narrow 1-D H/C
route. SCC-DFTB-SECCM, PM6-SECCM, OM2/OM3-SECCM, and GFN2-xTB-SECCM run
full Wigner-Seitz weighted supercell calculations in one, two, or three
cyclic dimensions, subject to their documented parameter and embedding
gates. OM1-SECCM fails closed because its defining analytic core-valence ECP
is not implemented. On the default T = 0 path, each executable adapter
reproduces its molecular driver bit-for-bit in the molecular limit (1x1x1
cyclic cluster). When the cyclic overlap has non-positive eigenvalues, the
non-orthogonal adapters (DFTB0, SCC-DFTB, GFN2-xTB) screen them by canonical
orthogonalization as described above.
At finite T, GFN2-SECCM keeps the same molecular state but reports the Mermin
free energy in its `energy` field. SCC-DFTB-SECCM adds charge
self-consistency, an opt-in Madelung/Ewald embedding for charged cells
(1-D background-corrected wire Ewald or 2-D Parry/Heyes Ewald; 3-D embedding
fails closed), a
fixed-topology analytic gradient, and an opt-in electronic temperature.
OMx-SECCM carries the Peintinger-Bredow three-center weighting (eq-13
production default, Janetzko eq-10 as an opt-in comparison path). Its exact
one-replica molecular limit, with zero-translation full-weight records, runs
normally; any cyclic-image topology requires the explicit non-quantitative
truncated-electrostatics acknowledgement described above until a variational
Madelung/Ewald operator is implemented.
GFN2-xTB-SECCM runs the full shell-resolved supercell SCC (image-block H0,
shell gamma, multipole AES, GAM3) and exposes an opt-in electronic
temperature for metallic supercells. A small electronic temperature
(0.001 to 0.005 Ha) is recommended for metals so degenerate-frontier
occupations stay branch-free; the quasi-Newton mixers
(`scc_mixer="newton"` or the tblite-faithful `"broyden_eyert"`, whose
`charge_mixing` defaults to xtb's bromix 0.4 with a full-budget
history) converge metallic cells in a handful of iterations where
simple mixing needs a hundred or more. Earlier releases documented a
spurious charge-density-wave basin on fcc Cu that the physical-basin
gate failed closed; that basin was an artifact of the d-first
transition-metal parameter rotation fixed with issue 43, and current
trees converge those cells into the physical basin under every mixer.
See {doc}`semiempirical` for the supported envelopes and call shapes.

GFN2-SECCM remains experimental: the H0 coordination numbers are assembled
from the directed WS inventory with its ownership weights (issue 354, fixed),
but the default AES moment-integral channel is still dependent on the
coordinate representative typed for a torus site (issue 348). The separate
periodic Gamma GFN2 route remains outside quantitative claims under issues
296 and 316; the multi-k route fails closed under issue 351.

An independent experimental ab-initio RHF/RKS route is also selectable
through the periodic runner. Its `kpoints` tuple is the cyclic-cluster size,
not an independently chosen integration mesh:

```python
res = vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="aiccm2026dev-b",
    kpoints=(4, 4, 4),
    aiccm_backend="four_center",  # or "ri" / "rijcosx"
)
print(res.energy, res.aiccm2026dev_b.density_idempotency_error)
```

This development route is neutral, closed-shell, zero-temperature, and
currently 3D-only. Four-centre, RI, and RIJCOSX backends support 3D RHF/RKS,
including hybrids where the selected backend implements their exchange.
Every 1D/2D backend fails closed until a shared neutral wire/slab Green
function and matching electrostatic terms are derived; see the
[χ-CCM page](aiccm2026dev_b.md) for that caveat.
Its mathematical definition, boundary weights, Coulomb gauge, variational
space, and disagreements with the historical weighting are documented in
the [independent derivation](../design_aiccm2026dev_b.md).

### Keywords and defaults

`CCMOptions` and the direct helpers expose the same core controls, but retain
different compatibility defaults for `madelung`:

| Keyword | Default | Meaning |
|---|---|---|
| `translations` | *(required)* | The 1 to 3 supercell lattice vectors, in Angstrom. The count sets the dimensionality: 1 vector is CCM1D (a polymer), 2 is CCM2D (a surface), 3 is CCM3D (bulk). |
| `madelung` | `True` in `CCMOptions`; `False` in `run_ccm`, `ccm_gradient_fd`, `ccm_gradient_analytic`, and `ccm_optimize` | Add the long-range Madelung electrostatic embedding. Recommended for ionic systems; the dimensionality of the embedding follows the number of translation vectors, and so does the kernel (2-D/3-D Ewald, 1-D truncated direct sum -- see [Madelung embedding](#madelung-embedding)). |
| `max_iter` | `200` | Cumulative SCF iteration budget of the energy, including every stability restart. A run that exhausts it raises; it is not silently accepted. |
| `conv_tol` | `1e-9` | Energy-route SCF threshold on both the commutator `[F,P]` and the energy change. |
| `gradient_conv_tol` | `1e-10` | The tighter threshold the analytic-gradient callback re-converges with. |
| `stability` | `"reference"` | SCF state policy after the stability audit: `"reference"` keeps the reference program's state and flags a detected saddle, `"lowest"` descends to a stable minimum. See [SECCM stability audit](msindo.md#seccm-stability-audit). |

Nuclear gradients (Ha/bohr) come in two flavours, both holding the
Wigner-Seitz topology fixed: `ccm_gradient_fd` (central finite differences) and
`ccm_gradient_analytic` (the analytic derivative, exact for 1-D / 2-D / 3-D
including the Madelung embedding, closed-shell).  Both are C++-backed.
`ccm_optimize` relaxes the structure (selected atoms) on the CCM energy
surface.

### Stress and lattice optimisation

`ccm_strain_derivative_analytic` returns the analytic strain derivative
`W[a, b] = dE/deps_ab` in Hartree, for the deformation `x -> (1 + eps) x` of
atoms and lattice vectors together. `ccm_stress_analytic` returns the stress
`W / V` in Ha/bohr^3 for a 3-D cell, where `V` is the volume of the cyclic
cluster. Positive entries are tensile, and the pressure is `-trace / 3`. Both
take the same keywords as `ccm_gradient_analytic`, `stability` included, and
are native for H to Kr with the Python reference beyond.

The derivative is taken with the Wigner-Seitz topology and weights frozen at
the reference geometry, like the nuclear gradients. Exact-tie images, shared
between cells with fractional weights as in MgO B1 2x2x2, make the energy
with a rebuilt topology non-differentiable under shear, because a shear
breaks the tie and changes the weights discontinuously. A uniform scaling
keeps every tie, so the pressure is also the volume derivative of an
equation-of-state scan. The 1-D truncated and 3-D Ewald Madelung embeddings
are included. The 2-D Parry/Heyes embedding is not, and asking for it
raises.

`ccm_optimize_lattice_isotropic` scales atoms and lattice together until the
analytic pressure vanishes. For a cubic crystal that is the full lattice
optimisation, and it reproduces the minimum of a Birch-Murnaghan fit
(`vibeqc.eos.fit_birch_murnaghan`) to an energy-volume scan.

```python
from vibeqc.semiempirical.methods.msindo_ccm import ccm_optimize_lattice_isotropic
from vibeqc.semiempirical.methods.msindo_ccm_gradient_analytic import ccm_stress_analytic

sigma = ccm_stress_analytic(Z, coords, translations, madelung=True)
opt = ccm_optimize_lattice_isotropic(Z, coords, translations, madelung=True)
print(opt.scale, opt.energy, opt.pressure_gpa)
```

### Scope and validation

Each limit below is a clean error, not a silent wrong answer:

- **Closed-shell (RHF) only.** A cell with `multiplicity != 1` raises.
- **Single point through `run_job`.** `optimize=True` raises; use
  `ccm_optimize` for geometry relaxation.
- **Elements H to Xe**, the MSINDO engine's supported set.

CCM energies are validated out-of-process against reference MSINDO to better
than 1 µHa.

Every converged closed-shell CCM SCF is followed by a restricted
orbital-stability audit (see [MSINDO](msindo.md#seccm-stability-audit));
the `.out` prints its verdict, the cyclic-cluster dimension, the
electrostatics kernel (naming the 1-D truncated sum as such), the SCF
controls and state policy that ran, and the largest net atomic charge. The
`.system` manifest records the same facts under `[run]` as
`seccm_periodic_dimension`, `seccm_electrostatics_kernel`,
`seccm_scf_max_iter`, `seccm_scf_conv_tol`, `seccm_gradient_conv_tol`,
`seccm_stability_policy` and `seccm_max_abs_net_charge`. Every point also
carries its stability record, `seccm_stability_checked`,
`seccm_stability_analysis_converged`,
`seccm_stability_eigenvalue`, `seccm_stability_restarts` and
`seccm_internal_instability`, so a scan or equation-of-state consumer can
separate stable minima from flagged saddles. A flagged saddle is a
reference-state result, not a stable-minimum claim.
These fields also appear in QVF `provenance.run_fields`. A checked audit
that did not converge has no stability verdict, even when the SCF converged
and no instability was reported. An unchecked audit likewise cannot certify
a minimum. Older records missing the audit-convergence field carry unknown
audit status; absence must not be interpreted as success.

## The ab-initio CCM roadmap

The semi-empirical engine is the proving ground; the destination is
**ab-initio CCM**, the project's v2.x track: HF-CCM (v2.0), density fitting
in CCM (v2.1), MP2-CCM (v2.2), orbital localisation and local correlation,
CCSD and CCSD(T)-CCM, and projection-based embedding (v2.8) as the capstone.
The full sequence is in the [roadmap](../roadmap.md).

The earlier ab-initio development track carries pair-derived Wigner--Seitz
weights into three- and four-centre integrals. The independent χ-CCM track
does not assume those formulae: it weights only tied
representatives of complete finite-translation equivalence classes, which
preserves ERI permutation symmetry and a scalar RHF functional. The
[derivation](../design_aiccm2026dev_b.md) gives both the argument and the
recorded disagreement.

The four-center Coulomb and exchange weighting is itself under active
comparison: two versions, `union12` and `aiccm2026dev-a`, are kept side by side
and cataloged on the [experimental features](../experimental/index.md) page.
The `aiccm2026dev-a` Γ-CCM line uses the union-and-weight/Wigner--Seitz
integral-weighting construction; the `aiccm2026dev-b` χ-CCM line uses the
finite-translation-group character construction. They are distinct approaches
compared at a declared common exchange-q=0 convention. The convention does not
assign them different Coulomb kernels, and equality for any specified operator
and route remains evidence to establish.

## References

- Bredow, Geudtner and Jug, *J. Comput. Chem.* **22**, 89 (2001),
  {vibeqc-cite}`bredow_geudtner_jug_ccm_2001`: the semi-empirical CCM that
  the current engine ports.
- Peintinger and Bredow, *J. Comput. Chem.* **35**, 839 (2014),
  [doi:10.1002/jcc.23550](https://doi.org/10.1002/jcc.23550),
  {vibeqc-cite}`peintinger_ccm_2014`: the cluster-choice problem and the
  ab-initio CCM foundation.
- Noga and co-workers, INDO cyclic-cluster convergence to the bulk limit
  (1999), {vibeqc-cite}`noga_ccm_1999`; Janetzko and co-workers on
  long-range interactions in metal oxides (2002),
  {vibeqc-cite}`janetzko_ccm_long_range_2002`, and on the Kohn-Sham CCM
  formalism (2008), {vibeqc-cite}`janetzko_ccm_2008`; Claff's DFTB cyclic
  cluster implementation (2012), {vibeqc-cite}`claff_dftb_ccm_2012`.
- Evjen, *Phys. Rev.* **39**, 675 (1932), {vibeqc-cite}`evjen_1932`: the
  fractional weighting of boundary charges that the Wigner-Seitz face
  weights generalise.
- Peintinger, *Elektronenstruktur von Molekülkristallen* (dissertation,
  Rheinische Friedrich-Wilhelms-Universität Bonn, 2013): the primary
  derivation of the ab-initio CCM, published as the 2014 paper above.

## See also

- [Tutorial: the cyclic cluster model](../tutorial/cyclic_cluster_model.md):
  the worked walkthrough (bulk, surface, and adsorption).
- [MSINDO](msindo.md): the semi-empirical engine CCM is built on.
- [Periodic-SCF methods](periodic_methods.md): the k-point routes, the
  reciprocal-space alternative to CCM.
- [χ-CCM guide](aiccm2026dev_b.md): the separate experimental API,
  backend matrix, and dimensional caveats.
- [χ-CCM derivation](../design_aiccm2026dev_b.md): the independent
  finite-torus RHF/RKS definition and validation plan.
