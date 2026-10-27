# BIPOLE: Ewald-J-split periodic HF and DFT

```{admonition} Feature status
:class: note

The BIPOLE drivers (`run_pbc_bipole_rhf`, `_uhf`, `_rks`, `_uks`)
form vibe-qc's BIPOLE periodic workstream. All four methods support
multi-k. The exact Ewald-J-split route is the supported production
path. The experimental quartet-level bipolar far-field is disabled;
`use_multipole_far_field=True` fails before SCF setup. Its current
two-translation dispatch does not cover the same periodic quartet
domain as the exact three-translation Fock contraction.

Energies and finite-difference forces are production: fixed-cell atomic
optimization uses the finite-difference BIPOLE force path by default.
Variable-cell and coupled atom/cell optimization fail closed. The analytic
gradients are a maintained preview (each analytic driver warns
`UserWarning: ... the BIPOLE analytic gradient is a maintained preview`), and
the quartet multipole far field is gated. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

For routine calculations, prefer:

* [`run_rhf_periodic_gamma_gdf`](density_fitting.md) for Γ-only
  hybrid / pure DFT with native Gaussian density fitting.
* [`run_rhf_periodic_multi_k_ewald3d`](multi_k_scf.md) for multi-k
  with 3D Ewald for the long-range Coulomb piece.
```

## What BIPOLE is

CRYSTAL splits the periodic Fock build into two halves and keeps
them in **separate gauges**:

* **One-electron Coulomb** (`V_ne`) and the **nuclear-nuclear
  energy** (`E_nn`), 3D Ewald (point-charge ↔ Gaussian-pair
  doesn't decay on the nucleus side, so direct truncation diverges
  on charged-nucleus crystals like MgO). Both share a single
  Ewald α (a single shared Ewald state).
* **Two-electron Coulomb + exchange** (`F^{2e}` = J + K),
  CRYSTAL's native code uses direct-space BIPOLE screening and
  multipole-far-pair replacement. The current vibe-qc parity path
  implements the same electrostatic gauge as
  `J_SR(ω) + J_LR(ω) + V_bg·S - 1/2 K_full`, where `ω` is the same
  Ewald α used by `V_ne` / `E_nn` and `V_bg = -π N_e/(α²V)`.
  This replaces the older broken `run_rhf_periodic_multi_k_ewald3d`
  composition, whose long-range J did not share CRYSTAL's gauge.

In equations, with $\omega$ the shared Ewald parameter, $V$ the cell volume,
$N_e$ the electrons per cell and $\mathbf D$ the density matrix, the
two-electron Coulomb matrix of the exact route is

$$
\mathbf J(\omega) = \mathbf J^{SR}(\omega) + \mathbf J^{LR}(\omega) + V_{bg}\,\mathbf S,
\qquad
V_{bg} = -\frac{\pi N_e}{\omega^2 V},
$$

$$
J^{SR}_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma}
\sum_{\mathbf g,\mathbf h}\Big(\mu\,\nu^{\mathbf g}\,\Big|\,
\frac{\operatorname{erfc}(\omega r_{12})}{r_{12}}\,\Big|\,\lambda^{\mathbf h}\,\sigma^{\mathbf h+\mathbf g'}\Big),
\qquad
J^{LR}_{\mu\nu} = \frac{4\pi}{V}\sum_{\mathbf G\neq 0}
\frac{e^{-|\mathbf G|^2/4\omega^2}}{|\mathbf G|^2}\,
\tilde\rho_{\mu\nu}(\mathbf G)^{*}\,\tilde\rho(\mathbf G),
$$

where $J^{SR}$ is a direct-space sum of erfc-screened Gaussian ERIs over the
image cells $\mathbf g, \mathbf h$ inside the Fock domain, $\tilde\rho_{\mu\nu}(\mathbf
G)$ is the analytic Fourier transform of the AO pair and $\tilde\rho(\mathbf G)
= \sum_{\lambda\sigma} D_{\lambda\sigma}\tilde\rho_{\lambda\sigma}(\mathbf G)$
that of the density; the uniform-background term $V_{bg}\mathbf S$ restores
the charge neutrality that dropping $\mathbf G = 0$ removed. The same
$\omega$ sizes the real- and reciprocal-space halves of $V_{ne}$ and
$E_{nn}$, which is what makes the three Coulomb channels cancel their
conditionally convergent constants. The derivation and the comparison with
the CRYSTAL multipole treatment are in
[periodic methods, section 2.1](periodic_methods.md#21-theory); the BIPOLE
name follows Saunders and co-workers ({vibeqc-cite}`saunders_bipole_1992`).

## What today's implementation does

`vibeqc.run_pbc_bipole_rhf` ships:

* The full real-space one-electron pipeline at Ewald gauge:
  $S(g)$, $T(g)$, $V_{ne}(g)$ at `opts.lattice_opts.cutoff_bohr`,
  Bloch-summed to per-k $S(k)$ and $H_{core}(k)$, canonical
  orthogonalisation $X(k)$. The singular erfc part of $V_{ne}$ is
  analytic via libint, and the smooth reciprocal/background part is
  analytic by shifted AO-pair Fourier transforms. A tightened
  unpruned Lebedev grid is still available as an explicit diagnostic
  fallback via `v_ne_grid_options`.

  Canonical overlap filtering can retain different orbital counts at
  different k points. UHF and UKS seed occupations in each retained
  orbital space while keeping density blocks in the full AO basis.
  `linear_dep_threshold` controls this filtering; increasing it changes
  the variational space and requires a separate convergence check.

  During density damping, the convergence and DIIS commutator uses the
  same mixed density as the physical Fock build. The orbital projectors
  used for level shifts remain separate from that evaluated density.

  For physical 3D RKS/UKS, an unset density-image radius now grows the XC
  support independently of the integral cutoffs, using the largest contracted
  Gaussian AO envelope at amplitude `1e-4` (the same estimate as GDF/CCM).
  The kernel receives the complete active bra/ket difference domain. This is
  an amplitude heuristic, not a bound on total density or XC-energy error;
  converge the image support and quadrature separately. Becke partition
  weights do not determine AO support ({vibeqc-cite}`becke_grid_1988`).

  A positive `options.lattice_opts.becke_image_radius_bohr` keeps an explicit
  finite density radius, capped at `cutoff_bohr`. Set it to `cutoff_bohr` to
  reproduce the former shared finite support. Historical index-ball and
  explicit molecular-home density domains retain their previous settings.
  `result.resolved_domain.xc_lattice` records any separate expanded XC
  options; the one- and two-electron integral snapshots retain their original
  cutoffs. Analytic KS gradients refuse independently expanded XC support
  until that derivative is implemented. Use finite differences with the same
  SCF options; no broader force qualification is implied.

  Built-in periodic meta-GGA XC applies the chain rule when its kinetic-density
  floor is active: both the potential and fixed-grid gradient differentiate
  the same floored energy. The floor is a numerical extension outside the
  physical feature domain; it is not a new functional. External providers
  continue to receive the raw kinetic-density features. This kernel correction
  does not broaden the supported analytic-force envelope.

  Native built-in XC reuses its torus grouping plan when the density and
  potential retain exactly the same screened pair indices. Different screens
  retain separate plans: a negligible density contribution can still have a
  nonzero potential derivative. This changes planning work only, preserving
  pair order, image support, signed mesh phases and numerical contractions.

  Declared full multi-k meshes must contain every reciprocal-grid address
  exactly once for the current cell and shift. Stale cells, off-grid points
  and duplicate addresses are rejected before integral work or a dry-run
  manifest is written. Point order and reciprocal-lattice wraps are preserved.

  Symmetry-reduced k-point input is checked against the native reduction
  of the current cell, mesh and shift before expansion to the full grid.
  Stale coordinates, duplicate representatives and contradictory star maps
  are rejected. Consistently reordered representatives and reciprocal-lattice
  wraps are accepted. Rebuild the k-mesh after changing the cell or symmetry,
  or provide the full mesh. This admission check does not enable an IBZ-only
  BIPOLE SCF.

  At finite temperature, RKS/UHF/UKS evaluate entropy from the natural
  occupations of the same density used for each iteration's energy,
  including mixed and restart densities. An atomic or restart seed whose
  occupations lie outside the fermionic interval first selects a physical
  Fock matrix; a Fermi refill then supplies an admissible starting density.
  This seed step is not reported as an accepted thermodynamic iteration.

  Single-range CAM and long-range-corrected hybrids, including CAM-B3LYP,
  LC-wPBE and wB97X, are available in corrected 3D RKS/UKS with physical
  AO-pair support at Gamma and complete multi-k meshes. Their exchange is
  `(alpha + beta) K_full - beta K_erfc(omega)`, following
  {vibeqc-cite}`yanai_camb3lyp_2004`. Only `K_full` receives the selected
  BvK Ewald exchange correction. The direct erfc sum retains its finite
  zero mode and receives no probe-charge correction. Reference calculations
  must match this convention separately for each kernel; applying a screened
  probe-charge correction changes the finite-mesh energy.

  Public dry-run preflight applies the same CAM/LC kernel and domain
  admission as the live driver, before writing a successful run manifest.

  Historical index-ball domains, uncorrected gauges and multiple screening
  ranges such as HISS remain refused for these hybrids. Analytic gradients
  remain refused for all range-separated hybrids. Fixed-density kernel
  comparisons and closed-shell spin limits are covered by tests; matched
  crystal total-energy, support convergence and performance qualification
  remain open under #657/#593. This does not widen the 0.18 release matrix.

  The **real-space** half of that one-electron Ewald sum is sized from
  the resolved Ewald `α`, not just from `nuclear_cutoff_bohr`. At
  CRYSTAL's `α = 2.8 / V^{1/3}`, the screened kernel at the nearest
  image is `erfc(α·L) = erfc(2.8) ≈ 7.5e-5` for a cubic cell of *any*
  side, and it never shrinks with the box, so the driver raises the
  one-electron cutoff to `sqrt(-ln ewald_precision) / α` when your
  `nuclear_cutoff_bohr` falls short of it, and says so in the SCF log.
  Tightening `ewald_precision` therefore widens the real-space
  envelope as well as the reciprocal one. The reciprocal radius is the
  larger of the calibrated CRYSTAL envelope and
  `2 α sqrt(-ln ewald_precision)`, so its Gaussian damping factor meets
  the requested precision. SCF, gradients and response reconstruction use
  this same radius, recorded in `resolved_domain`. Without the precision
  floor the reciprocal damping stayed near `1.4e-8` even for tighter inputs
  (#677). The damping target is not an omitted-energy bound: charge factors,
  shell multiplicities and basis support still require convergence checks.
  This follows the Ewald lattice sums and screening-parameter invariance in
  Saunders et al., Eq. (44), Property 11 and Eq. (55),
  [doi:10.1080/00268979200102671](https://doi.org/10.1080/00268979200102671).
  Your `cutoff_bohr`, the electronic J/K cell set, is unchanged. Before this (issue #478) a
  fixed cutoff truncated an image tail decaying only as `1/L`, leaving
  a `+0.226·N_e²/L` mHa monopole self-image in the Γ molecular limit.

  On the physical pair-complete domain, the analytic nuclear build also
  uses the general periodic builder's per-product source radius. The native
  bound limits omitted nuclear attraction to `1e-16` hartree per matrix
  element before floating-point summation error, within the resolved nuclear
  cutoff. AO-pair support, Ewald parameters and reciprocal terms are unchanged.
  This uses the point-charge limit of
  [Ye and Berkelbach (2021)](https://doi.org/10.1063/5.0046617), SI Section S4,
  Eqs. (S8)-(S13), with a conservative packing-ball bound replacing the
  asymptotic lattice-tail estimate in Eq. (S18). The historical origin-centred
  domain retains its full nuclear source list.
* A shared-gauge Ewald-J two-electron build by default for 3D
  systems (`use_ewald_j_split=None`, or `True` explicitly):
  short-range J from direct erfc-screened ERIs, long-range J from
  reciprocal AO-pair Fourier transforms, the matching electron
  background potential, and full direct-space exchange.
* The standard SCF inner loop: Bloch-sum, direct real-space
  energy evaluation (`Σ_g D(g)M(g)`),
  optional `_MultiKPulayDIIS`, optional `LEVSHIFT`, diagonalisation,
  optional MOM occupied-subspace reorder, and density rebuild. ODA is
  fail-closed on this route: a line-search mixture generally has no
  single orbital representation, so accepting it would make the returned
  density and orbital payload describe different states.
* The matching nuclear-nuclear Ewald sum so $E_{nn}$ shares the α
  used by $V_{ne}$.

The default `sr_image_precision=1e-8` sizes the electronic short-range
interaction reach from the smeared erfc factor. It matches the default Ewald
radial tolerance. The former `1e-6` default omitted an image shell in the
100-bohr Ne molecular limit, leaving a `9.21e-8` hartree Coulomb-cancellation
residual. The job runner, optimizer and NEB use the same default. Explicit
precision values, explicit low-level radii and the `None` diagnostic opt-out
remain available. An explicit CHI formulation retains its separately pinned
SR policy.

Explicit SR radii must be finite and exceed the electronic cutoff. The
precision-derived radius requires a finite, positive Ewald screening
parameter. Invalid radii are rejected before constructing the image domain;
the low-level exact-zone builder likewise requires a finite, positive zone
strictly below the electronic cutoff.

This radial criterion does not bound the summed lattice energy error. Since
the #531 increment every padded run also reports a monopole-shell estimate of
the short-range Coulomb energy it omits beyond the resolved reach $R$:

$$
\Delta E_{\mathrm{SR}} \approx \tfrac{1}{2} N_e^2 \sum_{\mathbf g \ne 0,\; |\mathbf g| > R - d}
\frac{\operatorname{erfc}(\sqrt{m_\omega}\, s_{\mathbf g})}{s_{\mathbf g}},
\qquad s_{\mathbf g} = \max(R, |\mathbf g| - d),
$$

with $d$ the AO-pair cutoff plus the largest minimum-image atom separation
in the cell (so two descriptions of one crystal give one estimate).
The minimum-image search uses a Minkowski-reduced lattice basis, so an
equivalent sheared basis cannot inflate this separation or change budget
admission. Reduction changes only the distance search, not physical supports.
The slowest smeared pair decay satisfies
$1/m_\omega = 1/\gamma_{\min} + 1/\omega^2$, and $N_e$ is the number of
electrons per cell. For two s-type Gaussian charges the
erfc-screened interaction is exactly $qq'[\operatorname{erf}(\sqrt m\,s) -
\operatorname{erf}(\sqrt{m_\omega}\,s)]/s$, so the kernel factor bounds every
omitted pair at its minimum midpoint separation; the electron-nuclear and
nuclear-nuclear channels are Ewald-complete, which is why an omitted
electron-electron image is not cancelled (Dovesi et al. (1983), Section II,
Eqs. (12a)-(12e) and (15)). The estimate accounts for the charge amplitude and
the image multiplicities, is invariant under origin shifts and equivalent-atom
relabelling, and on the 100-bohr Ne molecular limit at the former `1e-6`
default gives `1.8e-7` hartree against the measured `9.21e-8` residual, while
the shipped `1e-8` default leaves `6.4e-10`. It is a monopole-shell estimate at
the slowest decay, not a certificate for an arbitrary density: the sum of
absolute AO-pair charges of an SCF density can exceed $N_e$, and higher
angular products carry polynomial prefactors bounded separately by the
quartet screen. The SCF log prints it, the `.out` resolved-domain block
discloses its fields, and `resolved_domain.sr_truncation_estimate` records
them.

The opt-in `sr_truncation_budget_ha` keyword of the four direct drivers
(`bipole_sr_truncation_budget_ha` on `run_periodic_job`) turns the estimate
into a contract: the precision-derived reach grows by whole image shells
until the estimate is below the budget, an explicit `sr_image_extent_bohr`
whose estimate exceeds it is rejected, and `sr_image_precision=None` is
refused. The runner also refuses it with `optimize=True` or NEB, because
displaced geometries could resolve different image shells. Without a budget
nothing changes: defaults, reaches and energies are as before.

Ewald and SR tolerances remain independently adjustable; tightening one does
not silently change the other. Check convergence of the Coulomb components
and total energy for the chosen basis, cell and density. Dovesi et al. (1983),
Section II, Eqs. (13)-(15), explain why separately approximated Coulomb
expressions can differ despite their formal equality
([DOI: 10.1103/PhysRevB.28.5781](https://doi.org/10.1103/PhysRevB.28.5781)).

For result auditing, `energy_components[-1]` and `scf_trace[-1]` describe
the terminal density returned by the driver. The separate
`initial_density_energy_components` field preserves the component breakdown
from the density entering the first SCF cycle. In a legacy-gauge SAD run this
is the density convention corresponding to CRYSTAL CYC0; after even one
diagonalisation it is generally not the same state as the terminal row.

Real-space energy contractions export lattice-cell keys in a native batch.
This avoids constructing a NumPy index array for every cell on each contraction.
The operator support, cell order and energy arithmetic are unchanged. Each call
reads the current density blocks; no density or operator values are cached.
Contiguous double-precision block products are reduced in bounded batches,
keeping each product's memory order and the original sequential sum over cells.
This retains NumPy's within-block pairwise reduction along the fast memory
axis, as described in the [NumPy sum notes](https://numpy.org/doc/stable/reference/generated/numpy.sum.html#notes).
Batch storage is limited to 8 MiB for products and their stacked copy. Large
individual blocks and other data types use the original scalar reduction.

The default split-Hartree projection also avoids a complex Fourier-tensor
copy per output cell. It uses
`Re[w conj(FT)] = Re[conj(w) FT]`, conjugating the private weighted-density
vector once in place. The finite reciprocal sum, cell mapping and physical
support are unchanged. This reduces projection workspace; it does not bound
the retained Fourier cache or establish full-SCF timing parity with CRYSTAL23.

The direct erfc build selects potentially contributing lattice pairs before
traversing them. It uses the existing Schwarz and separation tests, preserves
their accepted contributions and summation order, and retains the requested
internal, output, and density domains. This does not loosen a cutoff or an
SCF tolerance. For numerical comparisons, setting
`options.lattice_opts.sr_sparse_traversal = False` retains exhaustive
enumeration with the same screening. Low-level J/K results expose
`cell_triples_considered`, `cell_triples_possible`, and
`shell_quartets_considered` to compare work independently of machine load.
Cell searches in domains of at least 128 images can use Cartesian bins of the
supplied coordinates, with a 32 MiB construction-memory limit. Bin ranges are
expanded conservatively, and every candidate retains the original squared
distance test. Small domains and grids that exceed the memory limit use the
balanced spatial tree. This changes candidate lookup only: the finite domains,
screening predicates and sorted contraction order are preserved, including for
sheared and lower-dimensional lattices. It uses spatial grouping as discussed
by Izmaylov et al. (2006), Section II.B, without introducing a new box-level
integral or density bound.
Shell-prefix bounds reject groups before quartet expansion, and surviving
integrals reuse primitive shell pairs in a bounded worker-local cache.
The cell-pair gate cache allocates rows only when traversal reaches them,
avoiding a quadratic clearing pass for mostly unvisited image pairs. Concurrent
shell-pair workers share each initialized row; dense traversal retains the same
two-bit gate states and work counters. Screening, summation order, and supplied
domains are unchanged.
Gate rows, their pointer tables and output translation tables share a 64 MiB
batch budget, including rows being initialized concurrently. If a row cannot
fit, the affected output restarts through the direct path; it is never accepted
with missing contributions.
Within the sparse shell-pair traversal, Coulomb and exchange use separate
worker tasks because they accumulate into separate matrices. Each task retains
the complete ordered sum for its output shell pair. This exposes more work
to the existing workers and shortens their sorting lists, with the same
per-worker scratch budget. Pure Coulomb and pure exchange builds use only
their requested channel. This is an implementation choice addressing integral
processing cost, one of the direct-SCF priorities in Almlöf et al. (1982),
Section II.A, [doi:10.1002/jcc.540030314](https://doi.org/10.1002/jcc.540030314);
it introduces no new integral bound or density approximation.
Exchange builds with at least eight workers schedule output shell pairs by
decreasing estimated work, using their screened pair populations and AO sizes.
This aims to reduce idle time at the end of each output batch while preserving
each complete sum.
The priority table shares the 64 MiB pair-list budget. Unlike the simple static
distribution of many analytic tasks discussed by Bush et al. (2011), Section 3,
[doi:10.1098/rspa.2010.0563](https://doi.org/10.1098/rspa.2010.0563), this is a
local cost hint for the existing dynamic worker queue; it changes no admission
test and does not implement CRYSTAL's distributed-memory algorithm.
Quartet work entries use fixed integer fields for the two cell and two shell
indices. Their numeric order is the same as the direct contraction order;
decoding uses shifts instead of division. Two 24-bit cell indices and two
8-bit shell indices fill a 64-bit entry. Domains above 16384 cells can therefore
retain the sparse traversal, subject to the existing pair/work-list budgets
and the gate storage bound. Domains whose pointer and translation tables alone
exceed that bound retain the direct fallback. The work-list budget is unchanged.
This complements the hierarchy of Schwarz and distance screening described by
Izmaylov, Scuseria and Frisch, J. Chem. Phys. **125**, 104103 (2006),
[doi:10.1063/1.2347713](https://doi.org/10.1063/1.2347713).
The implementation uses the supplied density images instead of assuming
insulating density decay. Causà et al., J. Phys. Chem. **92**, 909-913 (1988),
[doi:10.1021/j100315a010](https://doi.org/10.1021/j100315a010), discuss why
exchange ranges differ between insulating and metallic systems.
For all-s-shell quartets, the sparse short-range traversal evaluates the scalar
fundamental integral directly, using the existing Libint primitive-pair screens
and Boys evaluator. Other angular classes keep the general integral engine.
This follows Ahlrichs (2006), Eqs. (39), (40) and (52),
[doi:10.1039/b605188j](https://doi.org/10.1039/b605188j).
Primitive order, integral precision, finite image domains and density contraction
order are preserved; this is not an additional screening approximation.
Quartets with exactly one p shell use the first angular recurrence directly,
following Ahlrichs (2006), Eqs. (24), (31) and (52). They retain the general
engine's canonical primitive ordering and Cartesian or solid-harmonic output
convention. Quartets with two p shells use the second recurrence and, when both
p shells are on the same electron, the angular-transfer relation of Eq. (34)
after primitive contraction. Three-p quartets use the vertical recurrence
through the canonical `ps|ds` intermediate, then transfer angular momentum
on the ket after contraction to obtain `ps|pp`, following Eqs. (31)-(34).
The evaluator restores the caller's pair order after the Cartesian or
solid-harmonic transformation. For selected p-shell axes, a fixed-size transform
uses each configured spherical row's single Cartesian term, preserving the actual
Libint coefficient and its rounding. It retains separate per-axis transformations and
the zero-initialized accumulation; it does not approximate the coefficients by
one. Configurations with more than one term per row retain the generic transform.
This uses Schlegel and Frisch (1995), Eq. (15),
[doi:10.1002/qua.560540202](https://doi.org/10.1002/qua.560540202).
Four-p and higher angular classes continue through the general engine.
The direct s/p kernels specialize the pure erfc fundamental integral for Boys
orders zero through three. They use the same Libint Boys evaluator and the
subtraction in Ahlrichs (2006), Eq. (52), with fixed-size scratch arrays. The
specialization removes general erf/erfc mixture dispatch without changing
primitive screening, recurrence order or contraction order.
Unrestricted HF and hybrids share an ERI evaluation when both spin densities
need the same quartet. Each spin retains its own screening predicate and
summation order. Incremental builds prepare both density changes before the
joint traversal, keeping separate references and reset schedules. This uses
the compressed shell-density bounds and incremental Fock relation of
Almlöf et al. (1982), Eqs. (17)-(20),
[doi:10.1002/jcc.540030314](https://doi.org/10.1002/jcc.540030314).
The shared traversal serves the unreduced serial padded route, including
dedicated screened exchange. Existing symmetry, MPI output-cell farming and
older native runtimes retain their separate builds. The native paired-domain
API requires matching ordered density cell templates; its two results report
the same shared work counters, which must not be added together.
Logarithms of the compressed shell-density bounds are prepared once per build
and reused by product screening. Each density update rebuilds them. A smaller
spin bound retains its own scalar logarithm, and the final screening predicates
and conservative candidate-search margin are unchanged.
Exchange leaves can refine their density maximum using 64 classes of integer
lattice indices modulo four. Each class contains the maximum over every
supplied density image for the shell pair, including distant images and both
spin channels when paired. This does not assume periodic density or decay.
The class table is rebuilt for each density update and is limited to 8 MiB;
larger shell sets retain the global shell-pair maximum. Final integral gates
and the summation order remain unchanged, while earlier rejection can reduce
candidate counters. This adapts the density-bound hierarchy of Izmaylov et al.
(2006), Eqs. (51)-(56), to nonmonotonic finite-mesh density images.
Before geometric traversal, exchange also tests a density-weighted Schwarz
maximum for each product subtree. A rejection skips every contained ket pair;
the final Schwarz product association and accepted integral set are unchanged.
These maxima depend on the range parameter and belong to the kernel snapshot,
separately from the shared geometric tree. The table is limited to 8 MiB per
retained kernel, with the existing traversal as fallback. This applies the
batched screening hierarchy of Izmaylov et al. (2006), Sections II.B and II.D,
using maxima of the prepared Schwarz factors.
The whole-tree maximum is also tested before constructing the charge search
sphere. A rejected ket shell pair therefore avoids the radius lookup and
traversal setup, using the same kernel-specific bound and multiplication order.
Bra pairs are grouped by partner shell and sorted by their Schwarz factor
within each range. For a fixed density shell pair, the ket maximum and density
maximum are constant, so rejection can stop the remaining range. These maxima
still include every supplied image; no density decay or periodicity is assumed.
Compact range offsets share the 64 MiB pair-list budget, and accepted quartets
are sorted into the original order before accumulation.
Final exchange screening reuses the cross-pair charge factor and ket displacement
already resolved by the same traversal. This removes repeated geometry lookups
without retaining state across density updates or changing any screening test.
Indexed exchange admission also keeps the ket index supplied by the tree.
Each output batch prepares its finite-cell translations once and shares them
across shell pairs. Surviving leaves use the prepared lookup instead of repeating
the translation search. It neither wraps images nor fills holes in the supplied
domain; its storage is included when choosing the gate batch size.
The product-center hierarchy uses subtree jump indices for iterative traversal.
Its aligned nodes pack the leaf offset and count into one integer.
The lattice cells are partitioned once in double precision, with a single cell
ordering shared across shell pairs. Each pair retains its own product-center
boxes and envelope maxima. Leaves bound their actual entries; internal nodes
combine their children's bounds, avoiding repeated partitioning and scans of
whole subtrees. This does not assume identical product centers across pairs.
Hierarchy boxes use outward-rounded single-precision endpoints, reducing each
node from 128 to 80 bytes. They enclose the original double-precision boxes;
radii, masses, exponents, leaf envelopes and integral arithmetic remain double
precision. The existing entry budget proves that packed offsets fit. The tree
memory cap is unchanged and includes the temporary shared partition during
preparation. It retains the same leaf screening and early termination,
without a recursive call for each tree node. When the combined envelope and index memory budgets
have room, leaves also retain contiguous copies of the product envelopes.
Cell positions, indices and image-class labels follow the shared partition
order and are stored once across shell pairs. Each pair retains its own
Gaussian envelopes. This reduces leaf-cache memory and keeps repeated scans
contiguous without changing screening or contraction order. The copies use
spare capacity within the existing combined
192 MiB preparation limit; larger domains retain the original indexed lookup.
The copies belong to the geometry workspace and preserve the bounds and
summation order. Accepted quartets are sorted before contraction, independently
of their order in the spatial partition.
This follows the separation of geometry preparation and contraction in
Izmaylov et al. (2006), Section II.D.
For normal positive distances, power-of-two bounds on the distance logarithm
can settle a product-screening decision without evaluating that logarithm.
The bounds include rounding slack; ambiguous decisions and exceptional distances
retain the original scalar logarithm and screening threshold.
During an SCF calculation, the Fock context retains geometry-only preparation
for the direct erfc build: shifted shells, Schwarz factors, and charge/product
search bounds. The workspace checks the exact shell primitives, contractions,
centers, angular conventions, ordered cell indices and coordinates, lattice,
kernel parameter and screening modes before reuse. A geometry change replaces
retained preparation. Up to two kernel states share the same shifted shells
and, when applicable, Gaussian-product tree. This covers the Ewald split and
HSE screened-exchange parameters without rebuilding geometry when they
alternate. Each kernel retains its own Schwarz and charge bounds; a third
kernel evicts the least recently used state. In-flight builds retain their
own snapshots. Density values and support, output masks, density-weighted bounds,
integral engines and accumulated J/K matrices are rebuilt for each call.
This follows the separation of two-index preparation from the density
contraction described in Section II.D of Izmaylov et al. (2006). Each Fock
context owns its workspace; there is no process-global geometry cache.
For unrestricted screened hybrids, exactly equal alpha and beta lattice
densities share one screened-exchange traversal within a Fock build. Density
blocks, ordered cell indices and coordinates must all match; no tolerance or
spin-multiplicity assumption is used. Each spin retains its own Fock and energy
contraction, and exchange matrices are not reused across Fock builds.
The short-range Ewald J/K build for unrestricted HF and global hybrids also
shares a traversal for exactly matching spin inputs. Incremental builds compare
the actual density increments after each spin's independent history and reset
schedule, so equal current densities alone do not trigger reuse. Returned spin
matrices remain independent for in-place Fock assembly. This reuses the same
density contraction in Izmaylov et al. (2006), Eq. (7); it does not introduce a
spin approximation or alter integral screening.
The dedicated screened-exchange traversal skips Coulomb candidates and
contractions: its J contribution is already supplied by the base Fock build.
The native domain API offers `compute_coulomb=False` for this purpose and
returns zero J blocks while preserving the selected K domain, output masks
and contraction order. Padded and symmetry-reduced screened builds use this
selection when supported by the native runtime; older runtimes retain the
combined J/K traversal. This separates the screened exchange operator of
Heyd et al. (2003), Section II.A, Eq. (6),
[doi:10.1063/1.1564060](https://doi.org/10.1063/1.1564060), from the independently
assembled Coulomb term without changing either operator.
Setup also reuses each native lattice export within a pass. HF home-cell
initial guesses iterate over one exported density cell list; overlap-fold
checks and analytic Ewald nuclear assembly reuse their cell and block lists.
The native accessors return complete vectors, so indexing a fresh export
inside a cell loop otherwise introduces quadratic copy work. Cell ordering,
initial densities, nuclear terms and fold-convergence arithmetic are unchanged.
Padded-domain projection, MPI output collection and screened-exchange Fock
updates export each native block vector once per pass. This avoids repeated
conversion of the entire lattice inside per-cell loops. Cell ordering, copied
output blocks and the exchange contraction are unchanged; symmetry reads the
updated Fock blocks after the screened correction.
On both admitted symmetry routes, pure DFT requests only J, including
incremental and final-density builds.
The reduced helper returns `K=None`, so no exchange
matrices are built, reconstructed or stored in the incremental accumulator.
Very small Gaussian-product envelopes retain logarithmic upper bounds, keeping
the screening tree usable when their direct masses underflow. Source domains
and screening tolerances remain unchanged.
Whole-output bounds avoid allocating cell-pair gate tables for blocks whose
contributions are all screened out; these blocks remain present as zeros in
the result. Exchange tree queries use a separate density maximum for each
shell pair, taken over every supplied density image. The final quartet tests
and contraction order remain unchanged.
Dense candidate lists switch to a bitmap when the full pair domain fits
within the same 2 MiB scratch budget. Larger domains fall back to exhaustive
enumeration if their candidate list fills.

Per-cell AO-pair Fourier transforms use the existing native Gaussian kernel
for supported pure shells through L=6. Each ket translation is evaluated
separately, retaining its ordered AO pairs and complex phase. This accelerates
reciprocal-cache setup without adding screening. Set
`VIBEQC_AOPAIR_FT_BACKEND=python` to use the Python reference implementation.

Reciprocal J-cache normalization updates its owned Fourier array in place,
avoiding a second array of the same size. The retained reciprocal cache can
still dominate memory for large cells or reciprocal cutoffs; this reduces
temporary storage without changing its support or numerical values.

Multi-k long-range exchange groups target/source pairs by their existing
momentum-transfer channel and evaluates source Bloch phases in batches.
Unrestricted HF and hybrid KS share those panels across both spins. The
physical AO-pair support, full source mesh and weights are preserved.
A nonzero momentum-transfer channel with no reciprocal vectors inside the
resolved cutoff contributes zero to that finite long-range sum. This can
occur in elongated cells with an explicit small Ewald splitting parameter.
No Fourier panels are allocated for an empty channel; the diagonal channel
and its existing zero-mode correction are unchanged. Convergence with the
real and reciprocal supports remains necessary.
The default method builders retain at most 256 MiB of nonzero-transfer Fourier
panel payload and request at most 64 MiB of new panel payload per batch.
Small calculations reuse retained panels; larger calculations recompute panels
that do not fit. These limits exclude native scratch, metadata, densities,
outputs and the shared J cache, so they are not total process-memory limits.
The low-level `compute_K_long_range_mesh` helper accepts `max_cached_bytes`
and `max_batch_bytes` overrides. Oversized panels are contracted in reciprocal
vector slices and are not retained. Budgets below one AO-pair reciprocal
slice are refused before Fourier evaluation. The diagnostic per-target helper
retains its existing cache behavior. This optimization does not qualify symmetry reduction,
change the exchange zero-mode convention or establish CRYSTAL23 speed parity.

Per-k density reconstruction uses the native torus contraction when available.
It exports only the selected representative density blocks, rather than every
block in the finite real-space support. Minimum-distance representatives, stable
distance ties and Fourier accumulation order are unchanged. Selected blocks
are fresh copies, so subsequent density updates are always read from the current
lattice matrix set. The Python Fourier fallback remains available.
Its Hermitian projection is evaluated before writing the result, preserving
conjugate off-diagonal entries even when the unprojected fold is asymmetric.

The reverse transform of already formed per-k density matrices now folds
their Hermitian parts directly. Mixed densities no longer require an
eigensolve and orbital reconstruction before the inverse Bloch sum. The
native density and fractional-orbital interfaces share the same cell fold,
with the supplied k weights, phase convention and real projection. Negative
eigenvalues from intermediate mixed densities are retained. This follows
the Bloch convention of [Sun et al., Eq. 3](https://doi.org/10.1063/1.4998644)
without introducing their density-fitting approximation or new symmetry
admission. External-functional XC, including the SKALA host path, uses this
direct density transform.

Periodic meta-GGA and external-functional XC builds reuse the AO-gradient
times density matrix product when forming the density gradient and kinetic
energy density. Each worker retains one gradient product at a time. The image
domain, quadrature and screening settings are unchanged; the external provider still
receives the complete grid in one call.

External-functional XC setup forms the exact finite AO-image difference
support using a bounded integer bitmap. Wide or sparse index boxes retain
explicit tuple differences. This removes per-pair Python array construction
while preserving holes, cell order and missing-cell checks. It changes no
quadrature, spin convention, symmetry mapping or Fock channel. The complete
AO density and energy-derivative contract follows the molecular SKALA
interface described by [Pöschel et al., Sec. III, Eqs. 7-9](https://arxiv.org/abs/2608.19033);
the support representation is an implementation choice, not a new physical
approximation or a validation of periodic SKALA.

AO value and gradient evaluation reuses small Cartesian and spherical
transform vectors within each shell's grid-point loop. Each thread owns its
scratch storage, including when building screened periodic image tables.
This removes repeated allocations without changing AO conventions, grid
points, image support or screening.

BIPOLE RKS and UKS request XC potentials directly in k-point order. On a
complete uniform mesh, the native builder checks that the supplied real-space
density obeys the mesh's signed periodicity, including half-step shifts. It
then groups retained AO-image pairs with identical torus residues and neighbor
sets, sums their screened AO columns, and contracts each group once. This
preserves missing image differences, separate density and potential screens,
spin channels and the full external-provider grid. It applies to semilocal,
hybrid and external XC functionals, including the SKALA host interface.
Pair planning on small uniform meshes uses a temporary index of at most
4,096 residue-pair bucket pointers to avoid repeated map searches. It retains
the original ordered buckets and pair lists. Larger meshes, or indexes with
more slots than retained pairs, keep the general map lookup.
When a group's bra and ket image lists are identical, both contractions use
the same signed AO sum, including its gradients, within that grid batch.

For UKS, each rectangle's signed AO sums are also shared between the alpha
and beta density contractions. Each spin retains its own density, gradient,
kinetic-density fields and ordered reduction. Density construction also shares
cached immutable AO sums across workers, using the bounded list-selection
policy and group/byte limits described below for projection. The density cache
is released before projection; density products and spin-dependent fields
are never shared between spins. For uncached lists, each density worker
retains its two AO sum buffers within one grid batch. An identical ordered image list
reuses its sum; an exact ordered prefix is extended by adding only the new
images. Other lists rebuild the sum, preserving signed addition order and
the existing workspace allowance. This applies to native
semilocal XC and the external-provider feature path, without changing the
complete-grid callback. It evaluates the same spin-resolved AO bilinear form
as Pöschel et al., Eq. (2), [doi:10.48550/arXiv.2608.19033](https://doi.org/10.48550/arXiv.2608.19033);
sharing its AO operands is an implementation optimization.

GGA density gradients reuse one transposed-density contraction for all three
Cartesian directions, avoiding two matrix products per retained rectangle.
The transpose preserves nonsymmetric lattice density blocks. This algebraic
reassociation can change results at floating-point roundoff; meta-GGA and
external-provider kinetic-density contractions retain their original path.

Potential projection also reuses repeated ordered image sums and their
weighted gradients across groups within one grid batch. Reuse is bounded by
64 groups, eight groups per worker, and 64 MiB within the existing torus
workspace reservation. Plans with at most 2048 rectangles count their repeated
ordered image lists exactly and prefer those avoiding the most image additions;
this uses at most 4096 candidate records. Larger plans retain a bounded prefix
of candidates. The signed sums and contraction order are unchanged. Uncached
groups retain the original contraction.
Within each output block, uncached AO sums use the same ordered-prefix reuse
as density construction, and contraction buffers are reused across groups.
Identical bra and ket lists share their weighted gradient within each spin.
UKS projection shares signed AO sums and group lookup between alpha and beta,
including uncached rectangles. Each spin keeps its own potential weights,
weighted gradients and ordered matrix accumulation. The cache budget counts
both spin-dependent gradient tables. The same projection is used for native
semilocal XC and the full-grid external-provider adjoint, following the
spin-resolved derivative in Pöschel et al., Eq. (9).
No density values or AO tables persist between batches or SCF builds;
quadrature, screening, signed image order, and output accumulation order stay
unchanged. Independent potential output blocks are assigned dynamically to
workers to balance unequal rectangle counts and image support. Each block
retains its original rectangle accumulation order, and density reductions
retain their static worker assignment. With `perf_log(native=True)`, the native
RKS and UKS profiles separate setup, AO collocation, pair planning, density
formation, functional evaluation and potential projection. Grouped and pairwise
projection have separate phase names; the complete external-provider route is
reported as one phase. These elapsed wall times measure the native call and
do not alter numerical decisions. Timing is disabled by default.

Nonperiodic initial densities, incompatible meshes and groups that would cost
more than the original pairs retain the pair calculation. The original grid
batch boundaries are preserved; grouping is also skipped when its extra
scratch would exceed the native workspace limit. Native
`build_xc_periodic_kpoints` and `build_xc_periodic_uks_kpoints` return per-k
potentials and report whether grouping was used. Consumers needing individual
real-space XC matrices or their derivatives continue to use
`build_xc_periodic` and `build_xc_periodic_uks`. A contraction optimization
alone does not establish full-SCF performance parity with a reference code.

The vendored Libxc build evaluates the PBE and r2SCAN correlation spin
fractions as `2*rho_spin/(rho_alpha+rho_beta)`. This is algebraically equal
to `1 +/- zeta`, but avoids cancellation when one spin is empty. Density
thresholds, spin cutoffs and functional regularization are unchanged. The
retained shifted-mesh empty-spin checks require the same energy and full
potential tolerances as nonzero imbalanced spins. This conditioning fix also
applies to hybrids using those correlation kernels. Other functionals and
physical accuracy at the fully polarized endpoint require their own checks.

The algebra follows the spin-scaling definition after Eq. 3 of
[Perdew, Burke and Ernzerhof](https://doi.org/10.1103/PhysRevLett.77.3865),
p. 3866. It preserves the r2SCAN regularization described by
[Furness et al.](https://doi.org/10.1021/acs.jpclett.0c02405).

For angular-grid convergence studies, `GridOptions` also supports Lebedev
orders 59, 65, 77 and 95, with 1202, 1454, 2030 and 3074 points per radial
shell. Select `angular="lebedev"`, the requested `lebedev_order`, and
`angular_pruning="none"` to compare uniform rules. These are explicit choices;
the default grid and named presets are unchanged. Check XC energy, potential
and SCF convergence as well as radial and image support before replacing a
converged product grid with a smaller Lebedev grid.

For Gamma and multi-k RHF/UHF/RKS/UKS, `run_periodic_job` accepts
`conv_tol_grad` as a positive finite SCF commutator tolerance, including
hybrid and external-functional calculations. It is independent of the
atomic-force tolerance used for geometry optimization. Omitting it retains
the driver default of `1e-6`; an explicit value is preserved under
`convergence="auto"`. The `.out` file reports the executed value. For example,
`conv_tol_energy=1e-9, conv_tol_grad=1e-8` tightens both SCF checks without
changing the basis, lattice support or quadrature.

RKS also checks that diagonalizing the physical Fock reproduces the evaluated
density to `conv_tol_grad`. A small commutator can coexist with an update above this
fixed-point tolerance. If that provisional update fails density closure and
SCF acceleration is active, RKS retries the orbital update once using the
already computed accelerated Fock. It evaluates the resulting density in the
next SCF iteration before considering convergence. The retry uses the
configured occupation rules and retains the ordinary stationarity guards for
Fock mixing and level shifts. It adds no Fock or XC build. The progress stream
reports this retry explicitly.

All four ab initio BIPOLE drivers repeat the density-closure check on the
final non-incremental Fock, using their occupation rules, full mesh weights
and real-space density support. UHF and UKS check both spin densities.
Passing energy and commutator checks alone is insufficient. A failed density
check continues SCF when iterations remain; at the iteration limit the result
is non-converged. The check adds no Fock or XC build. Finalization reuses the
exact Fock; RKS and UKS also reuse its canonical bands. HF retains its
semicanonical output convention. Evaluated density, energy and entropy remain
authoritative, including fractional occupations, MOM and spinlock. The separate
experimental chi XC path retains its existing convergence checks.

The loop's Fock is built incrementally (`use_incremental_fock`, the
J$_{SR}$/K$_{SR}$ increments from $\Delta D$ under density-envelope
screening), and the accumulated screening error of that chain makes its
energy differ from a non-incremental build of the same density by a small,
structural amount, 2.1e-8 Ha on LiH/STO-3G (2,2,2) at a 25-bohr cutoff. The
convergence decision is therefore taken on the exact operator (GitLab #116):
when the loop's own criteria are met, the committed density is rebuilt
non-incrementally, its energy and commutator norm are judged against
`conv_tol_energy` and `conv_tol_grad`, and if they fail the incremental
chain is re-synced to that exact build and the remaining iterations use
full-density J/K builds. Reseeding alone can accumulate screening error
again. The fallback preserves the physical symmetry and paired-spin paths,
including the J-only pure-DFT path; it changes neither the operator domain
nor its screening thresholds. Healthy runs keep differential builds.
This is a conservative response to the accumulated screening error analyzed
by [Häser and Ahlrichs (1989)](https://doi.org/10.1002/jcc.540100111),
Sec. IV, Eqs. (30)-(35), rather than their optimized difference-density
error estimator. The post-loop refresh re-applies the same test to the same
rebuild, so a loop can no longer exit on a state the terminal check would
refuse, and no tolerance is widened. Both
values are logged and emitted as `scf_terminal_check` structured events
(`phase` `in_loop` / `post_loop`, `energy_loop`, `energy_exact`, `delta_e`,
`grad_norm_loop`, `grad_norm_exact`, `passed`), and a refused run's
`RuntimeError` names the terminal delta and commutator norm against the
tolerances. `scf_trace[-1].delta_e` is that terminal delta. On a
Monkhorst-Pack mesh the commutator norm and the terminal electron count
are measured on the Born-von-Karman torus fold of the real-space density
in every gauge, including the direct gauge of 1-D and 2-D cells, whenever
the density cell list covers the torus: the stored density is periodic on a
cell list wider than the torus, and a plain Bloch sum over that list would
overcount it. A list shorter than the torus keeps the plain sum.

That exact rebuild is the expensive post-SCF phase. It screens the full
density, whose image support can be much larger than the localized initial
guess or a late density increment. Its cost is measured separately.
Before it starts, the ordinary `.out` file also receives a flushed record
of the provisional SCF energy and iteration, including runs without a
structured log. The phase completion records build status and wall time;
it does not itself certify convergence, which the terminal check judges.
The historical #115 LiH (2,2,2) case with six basis functions spent 22,000 s
there after late iterations of 819 s. The build is announced before it
runs, and `scf_exact_confirmation_begin` records the loop's provisional
energy and iteration first, so a job killed inside it retains that
provisional result in the structured log; `scf_exact_confirmation_end` carries the
wall time. The post-loop `scf_final_density_begin`/`_end` pair then
reports `reused` (no second build) on a converged run, or the one
mandatory cold build on a max-iteration exit if no exact build already exists
for that density. A failed confirmation at the iteration cap also reuses its
exact build and remains non-converged. A build that raises ends
with `status="raised"` rather than a successful end.

When a native-functional RKS or UKS confirmation is refused and SCF continues,
the next iteration reuses its XC quadrature only if the committed density is
unchanged and undamped. J/K construction and the later exact confirmation still
run. This one-shot reuse retains the XC energy and potential together; it does
not reuse the assembled Fock or its DFT+U correction. External-functional
callbacks and the separate chi XC path retain their existing evaluation sequence.

`vibeqc.run_pbc_bipole_uhf` uses the same one-electron and Ewald-J
machinery with unrestricted spin densities:

* $D_{tot}(g) = D_\alpha(g) + D_\beta(g)$ drives the Hartree operator.
* $F^{2e}_\alpha(g) = J_{tot}(g) - K[D_\alpha](g)$ and
  $F^{2e}_\beta(g) = J_{tot}(g) - K[D_\beta](g)$.
* The UHF energy is evaluated as
  $E_{2e} = \frac{1}{2}\mathrm{tr}[D_\alpha F^{2e}_\alpha] +
  \frac{1}{2}\mathrm{tr}[D_\beta F^{2e}_\beta]$ in the same
  real-space lattice contraction convention as RHF.
* Even-electron SAD guesses are spin-averaged for CRYSTAL CYC0 parity:
  $D_\alpha = D_\beta = D_{SAD}/2$. The requested multiplicity enters
  through the alpha/beta occupation counts after diagonalisation.

What is still gated:

* True IBZ orbit expansion for the multi-k long-range-J density
  transform. Until that lands, symmetry-reduced Monkhorst-Pack inputs
  are accepted by internally expanding them back to the full mesh for
  Ewald-J correctness. This is a usability bridge, not a performance
  speedup.
* ECP derivatives, historical pair domains, symmetry-enabled ECP cells and
  external-XC or experimental source adapters. Scalar ECP single points now
  run on the neutral 3D corrected physical-pair RHF/RKS/UHF/UKS routes,
  at Gamma or a full k mesh. Supply an explicit spin-consistent multiplicity.
  The public runner resolves the basis sidecar; direct drivers require the
  complete inline primitive, center, effective-charge and removed-core payload.
  Coulomb attraction and ionic repulsion use effective charges, occupations
  use the valence count, and basis/XC partitioning retains bare species.
  Projectors reuse the midpoint-centered physical support from #652, with
  radius `max(cutoff_bohr, nuclear_cutoff_bohr)` of the resolved one-electron
  domain. The Ewald resolver can change these supports; inspect the recorded
  radius rather than assuming the requested nuclear cutoff was used.
  The returned `ecp_source` records the exact scalar operator, support and raw
  block identities. It is not derivative or finite-tail certification.
  Cl2/LANL2DZ HF Gamma and 3x1x1 reference controls and PBE0 single-point
  checks are bounded implementation evidence for #658, not general bulk-ECP
  or CRYSTAL23 parity and not an extension of the completed 0.18 release.
  This follows the scalar valence Hamiltonian of Hay and Wadt, Sec. II,
  Eqs. (5), (10)-(11), (15)-(16), [doi:10.1063/1.448799](https://doi.org/10.1063/1.448799).
* Methfessel-Paxton and Marzari-Vanderbilt smearing. BIPOLE implements
  Fermi-Dirac occupations (including the Mermin label); other flavours fail
  rather than being relabelled Fermi-Dirac calculations.
* Periodic Hessians, TDDFT, and COOP/COHP. The former generic paths used
  molecular or fixed-cutoff surrogate operators and now fail before SCF.
  QVF structure, wavefunction, population, and SCF-history content remains
  available, but surrogate DOS/PDOS/COOP/COHP arrays are omitted.
* Closed-shell multi-k READ restarts. A complete conversion from the saved
  Bloch density to the driver's real-space block convention is still needed,
  so the unsupported request fails instead of reading only Gamma.

The driver is an equation- and component-validation surface plus a supported
exact Ewald-J SCF implementation. Results from a different periodic Coulomb
algorithm are useful references, but total-energy agreement is not assumed
until gauge, basis, k mesh, and numerical supports have all been matched.

## Why ship it at all?

Two reasons:

1. **CRYSTAL parity surface.** Comparisons against external reference
   calculations help track the multipole branch's accuracy. The historical
   `tests/demos/` path is no longer present; current diagnostics live in
   [`examples/regression/bipole_parity/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/bipole_parity).

2. **Multipole infrastructure.** Moment, interaction-tensor, dispatch,
   and contractor prototypes remain available for method development.
   They are not called by a production driver. The legacy G1 cell-level
   path is retired, and the quartet-level replacement fails closed until
   its periodic translation domain, penetration criterion, screened
   kernel, symmetry reconstruction, exchange composition, and gradients
   have equation-level end-to-end validation.

## BIPOLE phase status

| Phase | Status | What |
|---|---|---|
| 1 | landed | Shell-pair Cartesian multipole moments (C++ via libint) |
| 2 | landed | Multipole-multipole Coulomb interaction tensor |
| 3 | landed | IDIPC geometric dispatch per quartet |
| 4a | landed | Cell-level multipole moments from density |
| 4b | landed | Neutral-cell spherical-sample dipole surface diagnostic |
| 4b proper | gated | CRYSTAL EXT EL-POLE shell-multipole/penetration decomposition; the former complete-AO-density reciprocal helper fails closed |
| 5 (Γ-only) | landed (2026-05-18) | Γ-only end-to-end SCF with `use_ewald_j_split=True` |
| 5 + DIIS | landed (2026-05-18 late) | DIIS-compatible via D-consistent energy/error formulation |
| 5 (multi-k) | landed (dense-k sign-off, 2026-05-20) | Per-k `F_J^LR(k)`, MgO/diamond/Si STO-3G within 1 mHa of CRYSTAL14 |
| 5K (RKS) | landed (2026-05-20) | BIPOLE RKS driver with libxc V_xc + hybrid support |
| 5U | landed (2026-05-20) | UHF BIPOLE with MOM/level-shift parity to RHF; historical ODA hook now fail-closed |
| 5UK | landed (2026-05-20) | BIPOLE UKS driver with spin-polarised V_xc |
| 5b | retired | Cell-level multipole far-field research artifact; not driver-reachable |
| 6a | **fail-closed (2026-08-13)** | Quartet-level prototype retained for redesign; explicit driver requests raise before SCF |
| 6b | landed | `compute_ext_el_spheropole` higher-l terms |
| POLIPO | **landed (2026-08-11)** | Native C++ multipole moment engine (McMurchie-Davidson Hermite kernel, OpenMP); 6/6 libint parity tests pass; ~200× slower than libint (optimization pending) |
| 7 | roadmap | CRYSTAL14 numerical-parity sign-off on the 15 demo set; BIPOLE parity tests cover LiH/NaCl/MgO/diamond/Si to < 1 mHa |

Phase 1-5 plus the exact Ewald-J split give the production BIPOLE
energy route; it does not call the gated EXT EL-POLE helper or either
multipole far-field prototype. Phase 4b-proper and Phase 6-7 remain
method-development work.

Dense-k STO-3G sign-off results at cutoff 14 bohr:

| Demo | vibe-qc E_total (Ha/cell) | Δ vs CRYSTAL14 |
|---|---:|---:|
| MgO SHRINK 8 | -271.2177748509 | +0.369 mHa |
| diamond SHRINK 8 | -74.8771393842 | -0.145 mHa |
| silicon SHRINK 8 | -571.3214715798 | -0.659 mHa |

## Quick start: SCF with the Ewald J-split

The Ewald-J split is automatic for 3D parity work. Full multi-k meshes
remain fastest to reason about, but IBZ-reduced Monkhorst-Pack inputs
are accepted and expanded internally to the full mesh until true IBZ
orbit expansion is wired.
Pass `use_ewald_j_split=False` only when intentionally reproducing the
legacy direct-only diagnostic branch.

```python
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

system, basis = build_mgo_sto3g()    # 15 demo geometries in crystal_demos/

# Γ-only:
kmesh = monkhorst_pack(system, [1, 1, 1])
# Multi-k:
# kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
# kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
# The second form is expanded internally to the full mesh for Ewald-J.

opts = PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 14.0
opts.lattice_opts.nuclear_cutoff_bohr = 14.0
opts.initial_guess = InitialGuess.SAD
opts.use_diis = True
opts.diis_start_iter = 2
opts.damping = 0.0
opts.max_iter = 15

result = run_pbc_bipole_rhf(
    system, basis, kmesh, opts,
    # use_ewald_j_split=None defaults to the exact Ewald-J-split
    # F^{2e} build for 3D systems.
    ewald_precision=1e-8,
)
print(f"E_total = {result.energy:+.6f} Ha (CRYSTAL: -271.218)")
```

For open-shell systems, use the UHF scaffold:

```python
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

result = run_pbc_bipole_uhf(
    system, basis, kmesh, opts,
    ewald_precision=1e-6,
)
```

Constraints (2026-05-20):

* **True IBZ acceleration is not implemented.** The k-space ρ̂(K)
  formula does not yet expand IBZ symmetry orbits directly. If an
  IBZ-reduced Monkhorst-Pack mesh carries `ir_mapping` metadata, the
  driver expands it to the corresponding full mesh internally.
  Explicit non-uniform custom k-meshes without that metadata are still
  rejected on the Ewald-J path.
* **The public BIPOLE route requires dim = 3.**
  `run_periodic_job(..., jk_method="bipole")` rejects dim < 3 before
  setup. The four direct `run_pbc_bipole_*` APIs retain a low-dimensional
  `DIRECT_TRUNCATED` diagnostic when `use_ewald_j_split=None`; passing
  `True` for 2D / 1D raises.

## Restricted exact bielectronic zone (`exact_zone_bohr`)

On diffuse bases the S(k) fold gate legitimately demands a large
`cutoff_bohr` (LiH/STO-3G needs ~37 bohr for a 1e-4 fold drift), but
the *fold* requirement is one-electron and cheap; historically the
expensive exact erfc J/K traversal grew with it. `exact_zone_bohr`
(all four RHF/RKS/UHF/UKS drivers, opt-in; screened hybrids and
ROHF/ROKS fail closed) bounds the exact-ERI output zone
independently: operator cells beyond the zone carry the reciprocal
`J_LR` channel plus the neutralising background alone -- the Ewald
far-field model of the density. This is the bielectronic-zone /
monoelectronic-zone partition of the CRYSTAL Coulomb machinery
(Dovesi, Pisani, Roetti, Saunders, Phys. Rev. B 28, 5781 (1983),
Eq. (24a); Pisani, Dovesi, Roetti, Lecture Notes in Chemistry 48
(1988), Sec. II.4b/II.4d) expressed in the Ewald-split gauge.

```python
result = run_pbc_bipole_rhf(
    system, basis, kmesh, opts,      # opts.lattice_opts.cutoff_bohr = 20.0
    exact_zone_bohr=12.0,            # exact erfc quartets only to 12 bohr
    ewald_precision=1e-8,
)
```

The omitted short-range tail is a *measured*, per-case quantity --
choose the zone with a small ladder exactly like the SR pad
(`sr_image_extent_bohr`). Measured fixed-density anchors (pad 20,
2026-08-06): MgO/STO-3G tail 2.7e-3 Ha at zone 10 and 3.2e-4 Ha at
zone 12; LiH/STO-3G (most-diffuse exponent 0.048) keeps mHa-scale
tail out to ~14 bohr and needs a correspondingly wider zone. Guard
rails: requires the corrected exchange split and the padded SR path,
must sit strictly below `cutoff_bohr`, is rejected with SYM3b
symmetry reduction / pair-resolved domains, and the analytic-gradient
preview fails closed on it (FD forces remain production). The
resolved zone is recorded on the result (`result.exact_zone_bohr`)
and in the run log.

## Where to look in the code

```
python/vibeqc/pbc_bipole.py
    run_pbc_bipole_rhf: the closed-shell RHF driver.

python/vibeqc/pbc_bipole_uhf.py
    run_pbc_bipole_uhf: the spin-unrestricted UHF scaffold.

python/vibeqc/pbc_bipole_rks.py
    run_pbc_bipole_rks: the closed-shell RKS (DFT) driver.

python/vibeqc/pbc_bipole_uks.py
    run_pbc_bipole_uks: the open-shell UKS (spin-DFT) driver.

python/vibeqc/bipole_multipole.py
    Shell-pair multipole moments (Phase 1) + interaction tensor
    (Phase 2).

python/vibeqc/bipole_dispatch.py
    IDIPC geometric dispatch (Phase 3).

python/vibeqc/bipole_cell_moments.py
    Cell-level multipole moments from a density block (Phase 4a).

python/vibeqc/bipole_ext_el_pole.py
    AO-density Fourier helpers, a reciprocal electronic Ewald diagnostic,
    the spheropole, and a fail-closed EXT EL-POLE compatibility entry point.

python/vibeqc/bipole_fock_multipole.py
    Retired multipole far-field J research artifact (Phase 5b):
    cell-level multipole-multipole interactions for far cell pairs.
    The exact Ewald-J split is the production path.

python/vibeqc/bipole_fock_ewald.py
    Ewald J-split F²e build: short-range erfc + long-range
    reciprocal + background (used by all four drivers).

python/vibeqc/periodic_jk_method.py
    PeriodicJKMethod.BIPOLE enum, selectable via
    run_periodic_job(..., jk_method="bipole").

examples/periodic/input-bipole-*.py
    Example inputs: MgO RHF, MgO RKS PBE, Li UHF, and
    run_periodic_job with jk_method="bipole".
```

## Finite-temperature results

For finite-temperature RKS/UKS, `energy`, `entropy` and `free_energy` describe
the committed density, with `free_energy = energy - smearing_temperature * entropy`.
Final canonical band eigenpairs and their refilled occupations describe the
Fock operator evaluated at that density. On an iteration-limited return they
can predict a different next density; their refilled entropy must not replace
the committed density's entropy. Exact convergence checks use the same
committed-density free energy before and after final canonicalization.

## Forces and analytic-gradient preview

```{admonition} Status: preview
:class: note

The BIPOLE analytic gradient is a maintained preview: the corrected-gauge
(Ewald-exchange-split) Gamma RHF/UHF and integer-occupation RKS/UKS drivers,
and padded radial multi-k RHF/UHF with the finite-domain orbital response,
ship and are finite-difference validated; padded multi-k KS, multi-k
pair-resolved domains and padded fractional occupations fail closed with
`NotImplementedError`. Every analytic driver emits a plain `UserWarning`
whose text begins `compute_bipole_gradient_<kind>: the BIPOLE analytic
gradient is a maintained preview` (KS drivers: `the BIPOLE KS analytic
gradient is a maintained preview`). Production forces are the
finite-difference path below. See
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

Production BIPOLE atomic forces use
``compute_bipole_gradient_fd``.  It finite-differences the same total
energy reported by the SCF driver, so it is the default path used by
geometry optimization and NEB for all four BIPOLE methods.

Physical pair and product-midpoint cutoffs are sharp. A displacement that
adds or removes a retained integral can produce a small energy jump, so
neither a finite-difference quotient nor the analytic preview is a smooth
force at that boundary. Check force stability against both displacement size
and domain cutoff; reducing the displacement alone can amplify a boundary
jump. Analytic comparisons require unchanged support across the displaced
geometries, in addition to the capability restrictions below.

```python
from vibeqc.bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
    compute_bipole_gradient_uhf,
    compute_bipole_gradient_rks,
    compute_bipole_gradient_uks,
)

# Production force path:
grad = compute_bipole_gradient_fd(system, "sto-3g", kmesh, opts, method="RHF")
print(f"max|grad| = {np.max(np.abs(grad)):.4e} Ha/bohr")
```

For a programmatic, imported, filtered or optimized basis, pass the actual
`BasisSet` instead of its name. Each displaced calculation preserves its
exponents, normalized coefficients and shell representation, moving each
shell with its owning atom. A name loads the reference basis once before the
displacements. Analytic orbital-response displacements also retain the
supplied shells in their overlap and Fock reconstructions, including the
Gamma hybrid/seminumeric CPHF and multi-k response paths.

The analytic drivers are still a research-preview surface. In the
**corrected (Ewald-exchange-split) gauge**, the BIPOLE default, maintained
Gamma cases cover RHF/UHF and integer-occupation RKS/UKS with padded radial
or pair-resolved domains. Padded radial multi-k RHF/UHF also include the
finite-domain orbital response. RHF uses a transpose response solve with
consistent complex rotations and k-point weights; an unsatisfied response
equation still raises ``NotImplementedError``. Its dense response matrix
adds a scaling cost for large orbital spaces.

Padded multi-k KS, multi-k pair-resolved domains, and padded fractional
occupations remain gated. Finite-temperature KS analytic gradients are
maintained on the historical corrected-gauge domain; finite-temperature UHF
analytic gradients remain unsupported. Meta-GGA τ-Pulay is landed (the term
validated to 1e-9 against an independent reimplementation; SCAN/r2SCAN are
well-behaved, TPSS/M06-L carry a known SCF-eigenvalue residual).

In the legacy (``use_exchange_ewald_split=False``) gauge, symmetric HF
controls remain maintained; asymmetric HF SCF is unsupported. Gamma-local
zero-smearing RKS/UKS include XC Pulay, moving-grid and KS-response terms.
Legacy multi-k KS uses diagonal-Z + corrected W + J^LR + XC Pulay and
**warns** that the full coupled-perturbed response is not included.
Legacy-gauge KS calls with finite-temperature/fractional occupations raise
``NotImplementedError``. See the [BIPOLE status](../bipole_status.md) for
the maintained validation cases. Use FD for production forces in all cases.

``compute_bipole_gradient_fd`` costs about 6N SCFs for N atoms. It
fails fast if any displaced SCF point does not converge, rather than
differentiating a failed iterate; use ``require_converged=False`` only
for diagnostics.

## Structure optimization

Atomic positions can be optimized at a fixed lattice using
:mod:`vibeqc.bipole_optimize`:

```python
from vibeqc.bipole_optimize import relax_atoms

# Atomic positions only (FD gradients by default, L-BFGS-B)
result = relax_atoms(system, "sto-3g", kmesh, method="RHF",
                     max_iter=30, conv_tol_grad=1e-4)

# One-shot via high-level API
from vibeqc.periodic_runner import run_periodic_job
result = run_periodic_job(system, basis, method="RHF", jk_method="bipole",
                          optimize=True, optimize_cell=False)
```

`relax_atoms` also accepts a `BasisSet` in place of the basis name. The
high-level BIPOLE optimizer retains the supplied basis for both the energy
objective and finite-difference forces, including custom bases whose names
do not exist in the library. Trajectory output keeps the basis name as its
display label.

SAD constructs atomic states from the supplied basis shells, including
in-memory bases with arbitrary display names and modified primitives retaining
a library name. It preserves their normalized coefficients and angular
representation. Equal-element atoms with different local bases have separate
atomic states; identical local bases share the atomic calculation.

Atomic relaxation preserves the input lattice. The historical
`relax_cell`, `relax_cell_gradient`, and `relax_full` entry points remain
importable but raise `NotImplementedError`: their strain convention and
coupled convergence were not certified on one terminal geometry. The
force-virial helper is available only as a diagnostic and is not the
periodic stress:

```python
from vibeqc.bipole_gradient import compute_stress_tensor
virial = compute_stress_tensor(system, gradient)  # diagnostic 3×3, Ha/bohr³
```

## Dimensionality support

| Dim | Coulomb | Gradient | Optimization | Notes |
|---|---|---|---|---|
| 3D | Ewald J-split (default) or DIRECT_TRUNCATED | FD production; analytic research preview | Fixed-cell atoms | Variable-cell entry points fail closed |
| 2D (surfaces) | DIRECT_TRUNCATED only | Diagnostic coverage only | Fails closed | Public BIPOLE rejects; use SLAB_EWALD_2D or slab GDF |
| 1D (wires) | DIRECT_TRUNCATED only | Diagnostic coverage only | Fails closed | Public BIPOLE and AUTO reject; model a wire as a `dim=3` cell with transverse vacuum |

For 2D/1D, only the direct BIPOLE drivers automatically fall back to
``DIRECT_TRUNCATED`` when ``use_ewald_j_split=None``. Treat that branch as a
vacuum-padded molecular-limit diagnostic, not a low-dimensional Coulomb
model. Model a wire as a `dim=3` cell with transverse vacuum, and use
SLAB_EWALD_2D (or the bounded closed-shell slab GDF route) for genuine
surfaces.

## Orbital and population sidecars

BIPOLE runs (RHF, RKS, UHF, UKS, restricted-open, Γ-only and
multi-k alike) write both sidecars. What each one means differs,
and the difference matters:

| Sidecar | Option | What BIPOLE writes |
|---|---|---|
| `{out}.population.{txt,json}` | `write_population_file` | A **crystal** population. Mulliken contracts the real-space lattice density with overlap on the recorded SCF one-electron domain; Löwdin and Mayer contract over the full SCF k-mesh. No Γ restriction. Dipole stays unsupported: the bulk position operator needs a Berry-phase polarization. |
| `{out}.molden` | `write_molden_file` | The **Γ-block orbitals only**, truncated to the home cell. Requires the k-mesh to contain Γ. |

For UHF and UKS, QVF atom properties also store Mulliken spin populations
from the accepted alpha-minus-beta lattice density on that same overlap
domain. Restricted closed-shell archives omit the spin member.

```{admonition} Prefer QVF and vibe-view over Molden for crystals
:class: important

Molden is a molecular format with no lattice vectors, no k-points, and
no complex coefficients, so a periodic `.molden` is necessarily a
narrow projection: the Γ block, evaluated over home-cell basis
functions with no image sum, and therefore clipped wherever an orbital
straddles a cell face. Orbitals at k ≠ 0 are irreducibly complex and
cannot be written at all; vibe-qc refuses rather than emitting a file
a viewer would misread.

Set `output_qvf=True` and open the archive with `vibe-view`. QVF
carries the lattice, the k-point metadata, the complex Bloch
wavefunction, and torus-periodic grids that wrap across cell faces.
See [QVF and vibe-view](../visualization.md) and
[Molden and periodic systems](../tutorial/orbital_visualization.md#molden-and-periodic-systems).
```

BIPOLE density volumes are evaluated from the density returned by the SCF,
including every translated lattice block. For a full uniform
Monkhorst-Pack mesh, vibe-qc performs an exact Born-von-Karman transform of
that returned density and verifies the inverse transform against every
source block before evaluating Bloch AOs. Alpha and beta densities are
verified separately. A refold-certified BvK density uses the returned SCF
Bloch overlap to validate its electron count. Its padded density-cell list
does not define a replacement overlap domain. Malformed overlaps or
contradictory mesh metadata are refused. The archive receives `volume.density`
only when this analytic count, periodic AO-image convergence, and the numerical
primitive-cell electron count all agree. If k-point provenance is incomplete,
the returned blocks cannot be refolded exactly, or the requested real-space
grid is too coarse, the calculation remains successful and QVF is written
without a density volume, together with a nonfatal output diagnostic. This
boundary never regenerates density from orbitals or a surrogate Hamiltonian.
The QVF file's `.system` outcome retains an omitted volume's error in
`component_errors["volume.density"]`, independently of whether the archive
itself was written successfully. Tight core functions can require much finer
sampling than the default `density_spacing_bohr=0.2`: Ne/STO-3G in a 7-bohr
cell gives 10.000022 electrons at 0.05 bohr, within the 0.005-electron charge
check. Refinement changes only the artifact sampling, not the SCF density or
its normalization.

All three options are tri-state. `None` (the default) emits the
sidecar when the route can produce it truthfully; `True` is a
guarantee and raises before the SCF starts if the route cannot;
`False` opts out. A shifted Monkhorst-Pack mesh has no Γ point, so
`write_molden_file=True` raises there; pass `None` to let the run
proceed without the orbital file.

## References

- **Pisani, Dovesi, Roetti 1988**, *Hartree-Fock Ab Initio Treatment
  of Crystalline Systems*, Lecture Notes in Chemistry 48, Springer.
  [doi:10.1007/978-3-642-93385-1](https://doi.org/10.1007/978-3-642-93385-1).
  Chapter II.4c defines the periodic bipolar quartet expansion and its
  common lattice-translation sum.
- **Saunders et al. 1992**, "On the electrostatic potential in
  crystalline systems where the charge density is expanded in Gaussian
  functions," *Molecular Physics* **77**, 629.
  [doi:10.1080/00268979200102671](https://doi.org/10.1080/00268979200102671).
  Defines the Gaussian-tail and penetration quantities used by the
  electrostatic-potential construction.
- **CRYSTAL14**, R. Dovesi, V. R. Saunders, C. Roetti, R.
  Orlando, C. M. Zicovich-Wilson, F. Pascale, B. Civalleri, K.
  Doll, N. M. Harrison, I. J. Bush, P. D'Arco, M. Llunell, M.
  Causà, Y. Noël, "CRYSTAL14: A program for the *ab initio*
  investigation of crystalline solids," *Int. J. Quantum Chem.*
  **114**, 1287 (2014).
  [doi:10.1002/qua.24658](https://doi.org/10.1002/qua.24658). The
  modern reference for the EXT EL-POLE / EXT EL-SPHEROPOLE
  conventions vibe-qc is targeting.
- **CRYSTAL23**, A. Erba et al., "CRYSTAL23: a program for
  computational solid state physics and chemistry," *J. Chem.
  Theory Comput.* **19**, 6891 (2023).
  [doi:10.1021/acs.jctc.2c00958](https://doi.org/10.1021/acs.jctc.2c00958).

BIPOLE runs through `run_periodic_job(..., jk_method="bipole")`
now route the methodology references through the citation database
([user_guide/citations](citations.md)) via `routes.methods["bipole"]`.
The 1988 monograph, Saunders 1992, and Dovesi 2014 fire automatically
on every BIPOLE run.

## See also

- [Periodic-SCF methods (comparative tour)](periodic_methods.md):
  side-by-side comparison of BIPOLE vs GDF vs GPW/GAPW, plus
  the surface-reactions workflow walkthrough.
- [Multi-k periodic SCF](multi_k_scf.md), the production multi-k
  RHF/RKS path today.
- [Density fitting](density_fitting.md), the production Γ-only
  hybrid-DFT path via native GDF.
- [Ewald summation](ewald.md), the long-range Coulomb machinery
  used by the legacy `run_*_periodic_*_ewald3d` drivers.

## Quartet-level bipolar far-field

The quartet-level implementation is currently unavailable from all four
drivers. The supported setting is the default:

```python
result = run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="bipole",
    use_multipole_far_field=False,
)
```

Passing `True` raises `NotImplementedError`. A future implementation must
first show that its skipped exact quartets and multipole add-back cover the
same three-translation periodic domain in Eq. II.4.10 of the 1988 monograph.
It must also validate independent Coulomb and exchange penetration criteria,
screened interactions, nontrivial symmetry operations, and energy-gradient
consistency. The low-level prototype modules are not a supported numerical
API and their isolated unit tests do not certify an SCF route.

## Resolved finite-domain settings

The four ab initio BIPOLE drivers return a `resolved_domain` record. It contains
separate one-electron and two-electron lattice options, the common Ewald alpha,
real and reciprocal reaches, Ewald precision, short-range interaction reach
and its omitted-shell Coulomb estimate, energy and force screening settings,
exact output zone and the selected XC grid.
An explicit nuclear quadrature and the exact-FT J diagnostic cutoff are recorded
when requested. `resolved_domain.as_dict()` is serializable;
`resolved_domain.settings_sha256` identifies these settings. It does not identify
the geometry, basis, native library or the retained integral tensor, and is not a
convergence or accuracy certificate. The ordinary SCF output discloses the record.

The same record accompanies exact and incremental Fock builds. Changing their
lattice options after resolution raises an error before accumulator reuse.
The check also covers the exact-FT J diagnostic selection and its cutoff.
Each driver copies caller lattice options before preparing its finite operator,
so a nuclear-cutoff clamp does not change a later run that reuses those options.
Analytic gradient previews restore omitted lattice and grid settings from the
result, and reject explicitly conflicting settings. Explicit original SCF lattice
options undergo the same neutral-cell cutoff preparation on a private copy
before comparison. The reconstructed nuclear Ewald alpha, real and reciprocal
reaches and precision must all agree with the recorded state. Both multi-k KS orbital
response helpers use the selected grid profile, periodic or molecular grid, and
image radius. Older results without this record retain the previous defaults.
Explicit nuclear-grid and exact-FT J diagnostic runs need the finite-difference
force path with matching SCF options; the analytic preview does not certify those
operators. Existing analytic-force capability restrictions still apply.
Recorded non-`AUTO` XC density domains (`MOLECULAR_HOME` or
`PERIODIC_LATTICE`) also raise `NotImplementedError` in the analytic preview:
its Pulay, grid-motion and orbital-response helpers currently use `AUTO`.
Use `compute_bipole_gradient_fd` with the same `xc_density_domain` and SCF
options until all analytic consumers support that selection consistently.

### Default policy and historical reproduction

`run_pbc_bipole_rhf`, `run_pbc_bipole_uhf`, `run_pbc_bipole_rks` and
`run_pbc_bipole_uks` now default to `historical_domain=False`, selecting
`physical_pairs`. AO-product separation determines pair support, while
short-range quartet interaction reach is a separate midpoint distance. Exchange
retains its AC/BD integral products. This reuses the physical nuclear, J/K and
derivative kernels; it does not enlarge a historical index ball by an arbitrary
amount.

For reproduction, pass `historical_domain=True` to a direct driver or
`bipole_historical_domain=True` to `run_periodic_job`. This selects
`historical_index_ball`, which is not invariant under atom-image relabelling.
The named keyword **overrides** `options.lattice_opts.pair_complete_1e`, including
an explicitly assigned False. Migrate an old historical input to the named
keyword. The shared native default remains False for other consumers;
ROHF/ROKS, GDF, semiempirical methods and SECCM are outside this migration.
The runner rejects an explicit historical request on an inactive route.

The runner forwards this choice through its RHF/UHF/RKS/UKS dispatches and
geometry optimization. Direct optimization and `compute_bipole_gradient_fd`
accept `historical_domain` through their driver keywords. Pass the same policy,
Ewald, screening and grid controls at every displaced geometry. The analytic
preview restores the SCF record when controls are omitted; an original requested
lattice-options snapshot or the resolved snapshot is accepted, but unrelated
changes are rejected. Caller options are never changed by policy resolution.
KS preview calls validate that snapshot before preparing the separate XC image
radius, so supplying the original options agrees with omitting them. Disabling
automatic SR padding with `sr_image_precision=None` still retains the physical
exchange density and output support in Gamma and multi-k HF derivatives. The
one-electron and reciprocal contractions retain their own pair support.
The shared analytic-FT Hartree derivative used by other periodic Ewald drivers
does not yet support physical pair domains; it refuses that combination. Use
finite differences of the same periodic SCF energy for that separate route.

With attached crystal symmetry, physical AUTO attempts density-qualified
direct J/K reduction on supported domains. Set `use_fock_symmetry_reduce=False`
(public runner: `symmetry_reduce_fock=False`) to request the full build.
Unrestricted state-selective controls, including atomic spin patterns, MOM,
spin locks and supplied spin densities, keep AUTO on the full build.
Explicit `True` requests reduction subject to the same geometry and basis
admission checks and the existing state-control restrictions. Approximate
symmetry models that fail the strict atom-action check retain full J/K.
The mapping covers exchange output pairs out to the pair cutoff plus the
interaction cutoff. It retains
the original physical internal, output and density cell lists and all native
quartet admission tests. It does not use the historical pair truncation.
Mapping preparation batches integer symmetry images by atom pair, avoiding
a separate small array operation for every pair and cell. It retains the
same fixed-point closure, cell order and cutoff-boundary admission checks.
Only one pair/operator image batch is held at a time, rather than an array
covering all symmetry operations and cells.
Fock reconstruction also batches members sharing an atom rotation. It keeps
the two Wigner-D matrix products and scatters only the covered atom blocks;
blocks outside the orbit space retain their contents. Temporary matrix batches
are limited to about 1 MiB, or one matrix triple for unusually large atom blocks.
For output sets up to 64 MiB, reconstruction also batches the scatter into an
owned dense buffer. The public block list still contains separate writable
matrix views, and input arrays are unchanged. The transient extra output copy
is capped at 64 MiB; larger sets retain per-block copying and scatter. Rotation
arithmetic, cell-key association and uncovered atom blocks are unchanged.

Every actual input density, including each spin and incremental density update,
is checked under the atom-pair action, including operations that fix a pair.
A nearly symmetric density can be split into its group average and an explicit
remainder. The average uses reduced direct J/K, while the remainder uses the
full direct domain. Their sum preserves the asymmetric contribution. The two
contractions share the original Schwarz screening budget, with a smaller
threshold for the smaller remainder. Larger symmetry breaking, disabled
screening or an unsuccessful average takes the ordinary full build.
The input density is unchanged. Reciprocal exchange, XC, and assembled Fock
matrices retain their original paths. The result reports
`physical_fock_symmetry_reduced_builds` and
`physical_fock_symmetry_full_builds`; the latter counts full-build fallbacks.
`physical_fock_symmetry_residual_builds` counts reduced builds that also required
a full-domain residual contraction. The split uses J/K linearity and does not
accumulate corrections between calls or replace exact convergence checks.
This implements the generating-pair reconstruction of Dovesi (1986),
Eqs. (31)-(36) and (43), [doi:10.1002/qua.560290608](https://doi.org/10.1002/qua.560290608),
with the physical pair/midpoint domain instead of CRYSTAL's overlap thresholds.

The current reconstruction requires atom-centered, contiguous pure shells
(Cartesian s is also supported), matching radial contractions and a usable
symmorphic subgroup. Geometry must satisfy the action to rounding accuracy.
Pairs or midpoint differences too close to a hard cutoff disable reduction:
rounded native cutoff decisions need not be identical across an orbit. The
geometry check also has a bounded work budget. An incompatible shell layout,
absent subgroup, ambiguous cutoff or exhausted check budget keeps the full
build. Explicit historical projection (`use_fock_symmetry=True`,
public runner: `symmetry_stabilize=True`) remains refused. Existing exact-zone,
broken-spin-control, ECP and COSX capability refusals remain in force.
In 3D with Ewald J, physical BIPOLE also requires corrected exchange. Explicit
`use_exchange_ewald_split=False` and an automatic fallback caused by incomplete
Monkhorst-Pack metadata are refused before SCF. Use a complete mesh for the
physical operator, or request the historical domain explicitly for legacy-gauge
replay. Supported low-dimensional direct calculations are unaffected.

At equal nominal radii, the two policies can retain different terms and yield
different energies, forces and runtimes. Old energy references must be explained
component by component and independently verified before replacement. Tolerances,
SCF damping and cutoff defaults have not been loosened or inflated. Production
relabelling, twelve-crystal and performance acceptance remain pending; the
settings record is not a numerical correctness certificate.

SAP uses all-space quadrature for localized initial-guess AO products. The XC
settings in this record describe the per-cell functional and must not replace
that SAP quadrature. No semiempirical method uses this BIPOLE record.

The [physical-domain design and acceptance matrix](../bipole_domain_contract.md)
separates the supported kernels, the approved default migration and the
remaining numerical acceptance.
