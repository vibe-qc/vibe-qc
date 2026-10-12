# Roadmap history

The releases from `v0.1.0` to `v0.15.0`, and the plans written for them.

This page exists because the plan and the record were the same document and
had stopped being readable as either. The [roadmap](roadmap.md) is what
vibe-qc intends to do next; everything below is what it did, and what it
thought at the time it would do. Read the sections here as dated evidence,
not as current status.

Three things to know before quoting anything from this page:

* **Plans here are not promises, and several were overtaken.** Composite 3c
  methods, the periodic stress, the equation-of-state helper, phonons and BSE
  basis fetching all shipped in some form, not always the form planned and
  rarely in the milestone assigned. The
  [release table](roadmap.md#release-table) and
  [`docs/features.md`](features.md) are the current answer for any of them.
* **"Codename TBD (candidates ...)" blocks are historical.** Codenames are
  assigned at the cut; `RELEASE_CODENAMES` in `python/vibeqc/banner.py` is the
  source of truth, mirrored by the roadmap's release table.
* **Milestone numbers in these sections are where an item was sent, not where
  it landed.** `v0.16.0` and `v0.17.0` each took the number of a planned
  milestone they did not deliver, which renumbered the ladder twice.

Sections that describe a companion repository point at that repository. The
queue, the viewer and the file format have their own trackers and roadmaps
since the 2026-09-08 split.

## Milestones, `v0.1.0` to `v0.5.x`

### `v0.1.0`, shipped 2026-04-18

Molecular HF/DFT/MP2 stack validated against PySCF; periodic scaffold
with one-electron infrastructure, Γ-only RHF, multi-k RHF, and
multi-k KS-DFT (LDA/GGA) all validated in the molecular-limit regime;
spglib integration; pob-\* basis sets integrated from a CRYSTAL-format
parser; documentation scaffold; MPL 2.0 license.

### `v0.2.0`, quantitative 3D bulk

Phase 12e: Ewald electrostatics and the exact BIPOLE J split, shipped in
sub-phases. The separate quartet multipole replacement remains fail-closed.

- ✅ **12e-a**, classical Ewald for point-charge nuclear lattice sum.
  `CoulombMethod.EWALD_3D` routes `nuclear_repulsion_per_cell` through
  Ewald; Madelung constants of NaCl, CsCl, ZnS, simple-cubic jellium
  reproduced to literature precision. See
  [user_guide/ewald.md](user_guide/ewald.md).
- ✅ **12e-b**, erfc-screened nuclear attraction as an Ewald
  building block. Public API `vibeqc.compute_nuclear_erfc_lattice`;
  real-space, exponentially convergent.
- ✅ **12e-c-1**, Gaussian-charge Ewald for the long-range *V(g)*
  via grid integration of the smooth complement. α-invariant to 1e-4.
- ✅ **12e-c-2**, erfc-screened ERIs for the short-range Ewald
  J/K. `omega` parameter on `build_jk_gamma_molecular_limit` and
  `build_fock_2e_real_space`.
- ✅ **12e-c-3**, long-range Hartree J via FFTW3 Poisson convolution.
  ✅ `12e-c-3a` adds FFTW3 as a build dependency and ships
  `solve_poisson_erf_screened` / `solve_poisson_coulomb`.
  ✅ `12e-c-3b` ships `build_j_long_range` + `auto_grid` (J_LR via
  density-on-grid sampling, FFT convolution, AO-pair re-integration).
  ⛔ `12e-c-3c`, quartet multipole far-pair replacement. Low-level
  research prototypes exist, but all SCF drivers default to the exact route
  and reject explicit activation before setup.
- ✅ **12e-c-4**, end-to-end `CoulombMethod.EWALD_3D` SCF dispatch.
  ✅ `12e-c-4a` composed Ewald-3D Hartree J + ω-invariance
  validation. ✅ `12e-c-4b` Γ-point periodic RHF SCF using EWALD_3D
  (`run_rhf_periodic_gamma_ewald3d`). ✅ `12e-c-4c-i` multi-cell
  periodic density grid + periodic J_LR. ✅ `12e-c-4c-ii` Pulay DIIS
  in the Γ-Ewald driver. ✅ `12e-c-4c-iii-a` multi-k Ewald Fock
  builder. ✅ `12e-c-4c-iii-b` multi-k Ewald RHF SCF driver
  (`run_rhf_periodic_multi_k_ewald3d`). ✅ `12e-c-4c-iv`
  Madelung-cancellation helpers for neutral crystals.
  ✅ `12e-c-4-end-to-end` `CoulombMethod`-aware SCF dispatcher
  (`run_rhf_periodic_scf` + `run_rhf_periodic_gamma_scf`) routing on
  `lattice_opts.coulomb_method`, plus internal-consistency
  bulk-benchmark suite covering H₂ / LiH / MgO / Ne (convergence,
  ω-invariance, EOS scan, multi-k = Γ-only at [1,1,1] equivalence).
  ✅ `12e-c-4c-iii-c` Pulay DIIS in the multi-k Ewald driver, per-k
  commutator error vectors with a k-weighted Frobenius inner product
  on the Pulay B matrix; lifts tight-cell [2,2,2] meshes from
  non-converging-under-pure-damping to converged in O(few) iters.
  ✅ CRYSTAL cross-check pass on LiH / NaCl / MgO / Si at published
  geometries; the v0.2.0 / v0.3.0 milestones were never separately
  tagged (absorbed into the v0.4.0 ship); the Ewald SCF chain and
  hybrid periodic DFT were validated against CRYSTAL reference
  energies as part of that combined release.
- ✅ **12f**, periodic Becke weight partition for DFT integration in
  tight crystals. Ships C++ ``build_grid_periodic(grid_mol,
  partition_atom_positions, opts)`` (extends the Becke fuzzy-cell
  partition denominator over home + image atoms within a radius),
  Python helpers ``vibeqc.build_periodic_becke_grid`` and
  ``vibeqc.extended_partition_atoms``, and the
  ``PeriodicKSOptions.use_periodic_becke`` /
  ``becke_image_radius_bohr`` flags wiring it into ``run_rks_periodic``.
  Tight-cell volume sanity (Σ_g w_g ≈ V_cell) and molecular-limit
  equivalence both validated.

#### Concurrent infrastructure shipped in 0.2.x

These landed alongside the Ewald work and aren't gated on it:

- ✅ **P1**, OpenMP shared-memory parallelism across hot kernels
  (overlap, kinetic, nuclear, ERI, gradient, Fock build, XC grid
  integration). Engine-per-thread + thread-local accumulator pattern.
- ✅ **P1.1**, gradient parallelism completion + `set_num_threads()` /
  `get_num_threads()` API + `num_threads=` keyword on `run_job` +
  per-section timing report. See
  [user_guide/scf_convergence.md](user_guide/scf_convergence.md).
- ✅ **P1.2**, close remaining OpenMP parallelism gaps: dipole
  integrals, the AO→MO transform and energy accumulation in MP2 /
  UMP2, and the per-cell loops in `build_xc_periodic`. With this,
  every compute-heavy kernel vibe-qc ships is OpenMP-parallel.
- ✅ **Reliability**, canonical orthogonalisation for
  linearly-dependent AO bases (handles tight diffuse-basis cases
  like H₂ at 0.5 bohr / aug-cc-pVTZ); pre-flight linear-dependence
  diagnostic; empty-BasisSet rejection (catches missing
  `LIBINT_DATA_PATH` entries); Functional dtor crash fix (the *real*
  cause of the meta-GGA segfault flagged in the `functional_comparison` tutorial).
- ✅ **P2**, pre-flight memory estimator with abort-on-overflow and
  `memory_override=` escape hatch. `vq.estimate_memory()`,
  `vq.check_memory()`, `InsufficientMemoryError`. See
  [user_guide/memory.md](user_guide/memory.md).
- ✅ **I/O workflow**, `run_job` driver writing `.out` (banner +
  geometry + memory budget + SCF trace + orbital table + properties +
  timing), `.molden` for orbital viewers, optional `.traj`.

### `v0.2.5`, space-group symmetry exploitation (core)

The biggest single periodic-performance lever we can pull: use the
crystal's space-group operators to reduce real-space matrix block
storage, integral evaluation, and (later) k-point sampling. For
high-symmetry ionic crystals (NaCl, MgO; O_h, |G|=48) this is an
order-of-magnitude speedup and a `|G|`-fold memory reduction. For
low-symmetry cells (P1) it's a no-op with no cost, the phase is
strictly a performance enhancement, never a correctness change.

What's already shipped as substrate:

- spglib integration (v0.1.0), detects the space group and returns
  the operator set ``{(R_i, t_i)}``.
- IBZ k-point reduction (v0.1.0), ``vibeqc.irreducible_kpoints``.

What Phase SYM adds:

- ✅ **SYM1**, AO-basis representation of every symmetry operator.
  For each operator ``R`` and each shell of angular momentum ``l``,
  compute the Wigner D-matrix ``D^l(R)`` that maps χ under R. Pure
  spherical-harmonic basis (vibe-qc forces ``set_pure(true)``) keeps
  the rep block-diagonal in l. Validated against the standard Wigner
  identities: ``D^l(R_1 R_2) = D^l(R_1) D^l(R_2)`` and
  ``D^l(R)^T D^l(R) = I``. A follow-up commit extended ``wigner_d_real``
  to improper rotations (det R = −1) via the inversion factorisation,
  unlocking the full O(3) coverage that real point groups need.
- ✅ **SYM2**, real-space matrix-block reduction. Three sub-phases:
  ✅ **SYM2a** ships ``build_ao_permutation_matrix``, the full
  ``(n_bf, n_bf)`` AO permutation matrix ``P(R)`` composing SYM1's
  Wigner D's with the per-atom permutation induced by ``(R, t)``.
  ✅ **SYM2b** ships orbit identification + LatticeMatrixSet
  compression for the **origin-fixed-atom** case: store one
  representative per ``h ↦ R·h`` orbit; reconstruct via
  ``block(R·h) = P(R) · block(h) · P(R)^T``.
  ✅ **SYM2c** ships **atom-pair-resolved orbits** for the general
  (non-origin-fixed) case, covers ionic crystals like NaCl where
  sub-lattices pick up shifts under most operators. Orbits are
  triples ``(a, b, h)`` under
  ``(a, b, h) ↦ (π(a), π(b), R·h + s_a − s_b)``.
- ✅ **SYM3a**, symmetry-reduced **storage** for one-electron
  lattice integrals. Ships ``vibeqc.compute_overlap_lattice_with_orbits``,
  ``compute_kinetic_lattice_with_orbits``, and
  ``compute_nuclear_lattice_with_orbits`` plus the
  ``OrbitReducedLatticeMatrix`` view, ``symmorphic_operations``
  filter, and ``verify_lattice_matrix_set_symmetry`` debug witness.
  Storage shrinks by the orbit compression ratio (~|G|× on
  high-symmetry crystals; 6.6× on simple cubic Pm-3m He / sto-3g
  with cutoff 10 bohr). Round-trip compress / reconstruct is
  exact to machine precision (~10⁻¹⁶) for S, T, V on Pm-3m.
- ✅ **SYM3b**, symmetry-reduced **compute**: integral kernels skip
  shell-pair × cell triples equivalent under the point group and
  scatter the result to all orbit members, so the ``|G|``-fold
  reduction hits wall-clock instead of just memory. Shipped for
  BIPOLE's real-space direct-ERI Fock build (`754de6fd1`,
  2026-06-16, ~3x faster) and extended to the wedge/IBZ erfc
  exchange arm (2026-07-29, 45.2 → 9.4 s/iteration, 4.8x). Not yet
  applied to every kernel vibe-qc ships.

Validation: reproduce the v0.2.0 bulk energies (LiH, NaCl, Si) to
machine precision *with* and *without* symmetry enabled. Measure
wall-clock reduction across the symmetry-enabled / disabled pair on
simple-cubic (|G|=48) vs monoclinic (|G|=4) vs triclinic (|G|=1).

The remaining symmetry pieces (**SYM4** IBZ Fock build, **SYM5**
symmetrized gradient, **SYM6** SALC/irrep projection during SCF) ship
in later milestones where they pay for themselves:

- SYM4 bundles with Phase C1 (v0.4.0, tight-cell SCF convergence).
- SYM5 bundles with Phase G1 (v0.5.0, periodic gradients).
- SYM6 ships when we need irrep-controlled SCF for transition-metal
  / open-shell periodic cases (v0.4.0 or later).

(The ``SYM`` label avoids a collision with Phase ``S1``, implicit
solvation, in v0.5.0. Tag namespaces: ``SYM*`` = symmetry core;
``S1`` alone = solvation.)

Reference path: public group-theory texts (Bradley & Cracknell;
Altmann & Herzig) and PySCF.pbc.symm are enough to implement and
cross-check; CRYSTAL source access is helpful for edge-case coverage
but not required.

### `v0.3.0`, visualization + post-SCF properties

User-facing observables and the writers viewers actually consume.

- ✅ **V1** cube files (molecular density + MOs, multi-volume MO
  stacks). Read by VMD, Avogadro, PyMOL, ChimeraX. See
  [user_guide/volumetric_data.md](user_guide/volumetric_data.md).
- ✅ **V2** Molden export (libint→Molden AO permutation handled).
- ✅ **V3b** XSF + BXSF writers (periodic structure, periodic
  volumetric, BXSF Fermi-surface band data). Read by VESTA / XCrySDen.
- ✅ **V4** `vibeqc.bands.band_structure` + `kpath_from_segments` +
  matplotlib plotter (`vibeqc.plot.band_structure_figure`). See
  [user_guide/band_structure.md](user_guide/band_structure.md).
- ✅ **V5** `vibeqc.bands.density_of_states` over a Monkhorst-Pack mesh
  + matplotlib plotter (vertical or horizontal, combined bands+DOS panel).
- ✅ **18** Mulliken / Löwdin charges + Mayer bond orders. See
  [user_guide/properties.md](user_guide/properties.md).
- ✅ **19** dipole moment (RHF/UHF/RKS/UKS), validated against PySCF
  to ~1e-6 a.u. on H₂O / 6-31G\*.
- ✅ **V3** periodic Bloch-orbital cube / XSF writers. Ships
  ``vibeqc.evaluate_bloch_orbital``, ``make_primitive_cell_grid``,
  ``write_cube_mo_periodic``, ``write_xsf_density``, ``write_xsf_mo``
  in ``vibeqc.periodic_orbitals``, periodic AO evaluator with the
  proper Bloch phase factors so periodic MOs export to formats VESTA /
  XCrySDen / VMD read directly.
- ✅ **V5b** projected DOS, atom-projected and (atom, l)-projected
  via a Mulliken-style projector against the Bloch-summed overlap.
  Ships ``vibeqc.density_of_states_projected``,
  ``density_of_states_projected_hcore``, ``ao_groups_per_atom``,
  ``ao_groups_per_atom_l``, and the ``ProjectedDensityOfStates``
  dataclass (re-exported from ``vibeqc.bands``).
- ✅ **BOND1** COOP/COHP bonding analysis + periodic Mayer bond
  orders. Ships ``vibeqc.compute_coop_cohp`` (C++ accelerated),
  ``periodic_mayer_bond_orders``, QVF ``dos.coop`` / ``dos.cohp`` /
  ``bond_orders`` sections, ``coop_figure`` / ``cohp_figure``
  plotters, ``bands_cohp_figure`` combo panel, and the
  ``vibeqc coop`` / ``vibeqc mayer`` CLI tools. See
  [user_guide/coop_cohp.md](user_guide/coop_cohp.md).
- ✅ **NO** natural orbitals + idempotency diagnostic on SCF
  results. Ships ``vibeqc.natural_orbitals(D, S)`` returning a
  ``NaturalOrbitals`` dataclass (occupations + coefficients) and
  ``vibeqc.idempotency_deviation(D, S)`` for SCF density health
  checks. Useful for post-SCF visualization, bonding analysis, and
  catching density-matrix drift.
- **16** hybrid periodic DFT (PBE0, B3LYP) validated against CRYSTAL
  reference energies.

### `v0.4.0`, shipped 2026-04-27, heavy elements + open-shell periodic + tight-cell SCF

**Second tagged release**, and the first one that's defensibly
"production-grade for real solid-state chemistry." The line of
reasoning was: tagging earlier ("research preview" at v0.2.0 /
v0.3.0) would have shipped a code that hit a wall the moment a user
tried a metal, oxide-surface, or magnetic system and watched SCF
oscillate. v0.4.0 added the SCF-convergence machinery (Saunders-
Hillier level shift, Fermi-Dirac smearing), end-to-end EWALD_3D
across {RHF, UHF, RKS, UKS} × {Γ-only, multi-k}, ECPs via libecpint,
and the doc-side polish (six new tutorials, push-deployed CI). See
the [v0.4.0 changelog](changelog.md) for the full release notes.

This release also unblocked the bulk of the CRYSTAL "Modeling
Specific Systems" tutorials, see the
[parity matrix](roadmap.md#crystal-solid-state-chemistry) for the new
✅ / 🟡 entries.

- ✅ **C1a**, Saunders-Hillier level shifting for periodic SCF.
  ``PeriodicRHFOptions.level_shift`` / ``PeriodicSCFOptions.level_shift``
  / ``PeriodicKSOptions.level_shift`` (Hartree, default 0.0) add
  ``b · (S − ½ S D S)`` to F before diagonalisation, raising virtual
  MO eigenvalues by ``b`` while leaving the SCF fixed point unchanged.
  Wired into the Γ-Ewald and multi-k Ewald drivers; reported MO
  energies are physical (un-shifted). Molecular RHF/UHF/RKS/UKS level
  shifting deferred to a follow-up.
- ✅ **C1b**, Fermi-Dirac smearing + fractional occupations in the
  multi-k Ewald RHF/RKS drivers. ``PeriodicRHFOptions.smearing_temperature``
  / ``PeriodicSCFOptions.smearing_temperature`` /
  ``PeriodicKSOptions.smearing_temperature`` (Hartree, default 0.0).
  ``vq.kelvin_to_hartree_temperature(T_K)`` converts user-facing
  Kelvin temperatures to the ``k_B T`` value stored in the options.
  When set, occupations follow ``n_i = 2 / (1 + exp((ε_i − μ) / T))``
  with bisection on μ to satisfy the per-cell electron-count
  constraint; introduces an electronic-entropy contribution to the
  free energy ``A = E − T S``. The multi-k Ewald result objects
  carry ``free_energy``, ``entropy``, ``fermi_level``, and per-k
  ``occupations`` arrays. Convergence is checked on the free energy
  (variational under smearing). C++ side ships
  ``real_space_density_from_kpoints_fractional`` for the
  inverse-Bloch fold over fractional occupations.
  Currently periodic Ewald RHF/RKS only, molecular RHF/UHF/RKS/UKS
  smearing and periodic UKS smearing are follow-ups.
- **C1c**, second-order / quadratically-convergent SCF as a fallback
  when DIIS + smearing still oscillate. Targets the hardest tight-cell
  cases.
- **14** libecpint integration (ECPs), unlocks pob-\* for Rb-I, Cs-Po,
  La-Lu. Metal oxides, perovskites, lanthanoids accessible.
  ✅ **14a** vendored libecpint build (v1.0.7 pinned alongside
  pugixml 1.15 and libcerf 3.3 via ``scripts/build_libecpint.sh``;
  same self-contained pattern as ``third_party/libint/``). CMake
  ``find_package(ecpint)`` + RPATH baked so the compiled module
  resolves ``@rpath/libecpint.1.dylib`` from the vendored install
  with no Homebrew / system fallback. ``vibeqc.libecpint_version()``
  smoke test passes; full regression unaffected by the new link.
  ✅ **14b** ECP matrix elements via libecpint's built-in XML
  library. ``vibeqc.compute_ecp_matrix(basis, ecp_centers,
  library_name, share_dir)`` returns ``V_ECP_{μν} = ⟨χ_μ|V_ECP|χ_ν⟩``
  in the spherical AO basis. Cartesian → spherical transform per
  shell-pair via libint's ``solidharmonics::tform_cols``/``tform_rows``
  primitives. Supports the standard Stuttgart-Köln ECPs (ecp10mdf,
  ecp28mdf, ecp46mdf, ecp60mdf, ecp78mdf) plus LANL2DZ. ``share_dir``
  defaults to the vendored ``third_party/libecpint/install/share``
  baked at build time. Empirical witness: Zn / 6-31G + ecp10mdf →
  27×27 symmetric V_ECP, eigenvalues bounded in [-0.5, 472] Ha
  (core projector pushes core orbitals up, physically correct).
  ✅ **14c** ECP wiring into all four molecular SCF drivers (RHF /
  UHF / RKS / UKS). Added ``ecp_centers: List[ECPCenter]`` and
  ``ecp_library: str`` fields on ``RHFOptions`` / ``UHFOptions`` /
  ``RKSOptions`` / ``UKSOptions`` (defaults: empty list, empty string
  = ``"ecp10mdf"``); when ``ecp_centers`` is non-empty the SCF adds
  ``V_ECP`` to ``Hcore`` once before the iteration loop. Empty list
  reproduces the all-electron path bit-for-bit. Empirical witness:
  Zn²⁺ at 6-31G with ecp10mdf converges in 9 iterations to E ≈
  −551.5 Ha (10-electron core replaced; 18 valence electrons in 3d¹⁰).
  ✅ **14e** validation against PySCF (which also links libecpint).
  Caught a real correctness bug: when ECPs are used the bare nuclear
  V double-counts because libint's ``compute_nuclear`` uses the full
  Z while the ECP already encodes the core's nuclear contribution.
  Fixed by adding ``compute_ecp_one_electron(basis, mol, ecp_centers,
  ecp_library)`` that builds V_n with effective charges
  ``Z_eff = Z − ncore``, computes V_ecp via libecpint, and computes
  the effective nuclear repulsion ``Σ_{A<B} Z_eff_A Z_eff_B / R_AB``.
  Each of the four molecular SCF drivers now routes through this
  shared helper. Empirical witness: vibe-qc Zn²⁺ / 6-31G + LANL2DZ
  matches PySCF to better than 1 µHa (E = −62.456043 Ha, exact
  to printed precision).
  ⏳ 14d CRYSTAL parser ECP block reader.
- **15** periodic UHF / UKS, open-shell periodic SCF for magnetic
  systems, defects, spin-polarized transition-metal compounds.
  ✅ **15a** Γ-point periodic UHF with the EWALD_3D Coulomb dispatch
  (``vibeqc.run_uhf_periodic_gamma_ewald3d``). Independent α / β
  density matrices, Hartree J via the composed Ewald split + per-spin
  K via the full-range real-space builder, spin-coupled Pulay DIIS
  (one B matrix from the stacked (e_α; e_β) error, one coefficient set
  for both Focks), ⟨S²⟩ contamination diagnostic. Closed-shell H₂ matches RHF to
  ~10⁻¹⁰ Ha; H atom doublet reports ⟨S²⟩ = 0.75 = S(S+1) exactly;
  PySCF cross-check on H atom UHF passes within the Makov-Payne
  finite-box bound (~50 mHa).
  ✅ **15b** multi-k periodic UHF (``run_uhf_periodic_multi_k_ewald3d``)
  on top of the multi-k Ewald RHF substrate. Spin-coupled Pulay DIIS
  (concatenated α + β per-k block lists with duplicated k-weights, one
  coefficient set for both spins), per-spin level shift, ⟨S²⟩ from the
  Γ-block MOs. Six lattice-ERI
  builds per iteration vs three for the closed-shell driver
  (J_SR(D_total) + J_LR(D_total) + 2 × {J_full(D_σ) + F_full(D_σ)}
  for K extraction). Closed-shell H₂ at [1,1,1] mesh matches the
  multi-k Ewald RHF energy bit-for-bit; H atom doublet reports
  ⟨S²⟩ = 0.75 exactly.
  Surfaced and fixed a quiet binding bug: ``LatticeMatrixSet.blocks``
  returns a fresh Python list copy each access, so
  ``lms.blocks[i] = matrix`` writes to a transient list and never
  reaches the C++ vector, silently a no-op. Affected the multi-k
  Ewald RHF damping path (``_damp_lattice_matrix``) which appeared
  to work because every existing test runs with DIIS active.
  Fixed by exposing a ``LatticeMatrixSet.set_block(i, M)`` method
  that mutates the underlying C++ storage in place.
  ✅ **15c** periodic Ewald-DFT (closed- and open-shell, Γ-only and
  multi-k). Shipped end-to-end in three sub-phases:
 - **15c-1** Γ-only periodic RKS SCF with EWALD_3D Coulomb
    (``run_rks_periodic_gamma_scf`` dispatch / ``run_rks_periodic_gamma_ewald3d``
    backend). DFT counterpart of 12e-c-4b (Γ-only RHF Ewald):
    ``F = Hcore + J_ewald(ω, D) + V_xc[ρ(D)] − (α/2) K(D)`` with
    ``α = func.hf_exchange_fraction``.
 - **15c-2** multi-k periodic RKS SCF with EWALD_3D Coulomb
    (``run_rks_periodic_scf`` dispatch / ``run_rks_periodic_multi_k_ewald3d``
    backend). DFT counterpart of 12e-c-4c-iii (multi-k RHF Ewald).
 - **15c-3** Γ + multi-k periodic UKS SCF with EWALD_3D Coulomb
    (``run_uks_periodic_gamma_scf`` / ``run_uks_periodic_scf``).
    Per-spin Fock construction; closed-shell H₂ at any k-mesh
    matches the RKS Ewald energy bit-for-bit. Open-shell DFT
    companion to 15a/15b (UHF Ewald).

### `v0.5.0`, shipped 2026-04-30, molecular Hessian + full k-point sampling

> **Codename: Wilson's Otter.** First release under the
> Scientist's-Animal codename policy (see § Release codenames).
> E. Bright Wilson, FG-matrix vibrations pioneer, co-author of
> *Molecular Vibrations* (1955), anchors the milestone's primary
> flagship feature: molecular Hessian → vibrational frequencies →
> thermochemistry.

> **Milestone shape note.** The original v0.5 plan bundled dispersion,
> solvation, periodic forces + stress + phonons, and a job queue all
> into a single 30-50-day release. After a user-driven scope
> rebalance the kitchen-sink was split into smaller single-headline
> releases (v0.5 → v0.13, see below). Each subsequent minor release
> has exactly one flagship feature so bug reports against a specific
> version are actionable, regression hunts are bounded, and ship
> cadence stays brisk. v0.5 itself absorbs **two** flagships because
> the K-Phase work landed during the same development cycle as the
> molecular Hessian, both were ready to ship together.

🎯 **Primary headline feature**, molecular analytic Hessian + IR
intensities + thermochemistry. Turns vibe-qc from "computes energies"
into "does chemistry": frequencies, ZPE, ΔG, TS verification,
KIE-ready, all at once.

🎯 **Secondary headline feature**, full k-point sampling support
(K-Phase). Public ``vibeqc.KPoints`` builder with seven construction
modes, Monkhorst-Pack, Γ-centered, shifted, Γ-only, IBZ-reduced,
HPKOT band paths via seekpath, explicit user lists, density-based
auto-mesh (KPPRA / kspacing / VASP ``Auto``). Closes the "manual
k-list construction" gap versus CRYSTAL's ``SHRINK`` and VASP's
``KPOINTS``.

#### Molecular Hessian (Phase 17)

- ✅ **17a-1** finite-difference Hessian + harmonic frequencies
  (Wilson FG analysis, mass-weighted projection, trans/rot zero-mode
  enforcement).
- ✅ **17a-2** IR intensities, dipole-derivative tensor,
  Wilson-Decius-Cross intensity formula in km/mol.
- ✅ **17a-3** thermochemistry, translational + rotational +
  vibrational partition functions → ZPE / U / H / S / G at any T, p.
- ✅ **17b-1** CPHF infrastructure, Krylov solver with PySCF-
  compatible orthonormality fix and orbital-energy response.
- ✅ **17b-2** skeleton 2nd-derivative integral wrappers around the
  vendored libint deriv-order=2 build.
- ✅ **17b-3** analytic RHF Hessian, assembles the full closed-shell
  HF Hessian from skeleton + CPHF mo1 / mo_e1. Matches PySCF analytic
  Hessian to <1e-7 Ha/bohr².
- ✅ **17c** analytic UHF Hessian, per-spin extension. Validated
  against PySCF on triplet O₂ and OH radical (LGMRES converges where
  vanilla GMRES stalls on small open-shell CPHF).
- ✅ **17d** analytic RKS Hessian, KS extension with libxc fxc kernel
  in the CPKS response, FD-on-gradient skeleton (avoids 2nd-deriv AO
  + Becke partition machinery). LDA + hybrid (B3LYP) covered.
- ✅ **17e** analytic UKS Hessian, closes the open-shell-DFT branch.
  LDA + pure-GGA covered analytically; hybrid-GGA UKS (B3LYP UKS)
  marked xfail and falls back to the FD path until **17e-2** lands
  the polarized-GGA fxc kernel (queued for v0.5.1).

#### Restricted open-shell (ROHF / ROKS)

- ✅ **18a** molecular ROHF (`method="rohf"`, `vibeqc.run_rohf`),
  spin-pure single determinant via Roothaan's single effective Fock
  (Coulson coupling; Roothaan 1960). Pure-Python driver on the JK seam
  (`python/vibeqc/rohf.py`); ⟨S²⟩ = S(S+1) exactly. Coupling + energy
  math verified directly (closed-shell→RHF reduction, SCF stationarity);
  PySCF-ROHF parity + ROHF-on-singlet==RHF tests in the dev-box suite.
- ✅ **18b** molecular ROKS (`method="roks"`, `vibeqc.run_roks`), KS
  counterpart reusing the Roothaan coupling with the spin-polarised XC
  potential. LDA / GGA / global hybrids; meta-GGA / RSH / double hybrids
  gated. KS Fock builder verified to reduce exactly to ROHF (α_HF=1,
  XC=0); PySCF-ROKS parity + ROKS-on-singlet==RKS tests in the dev-box
  suite.
- ✅ **18c** ROHF/ROKS finite-difference geometry optimisation +
  spin-pure CAS reference (`cas_reference="rohf"`).
- ✅ **18d** analytic ROHF gradient (`compute_rohf_gradient`; W = Dα·Fα·Dα +
  Dβ·Fβ·Dβ, FD-verified) → analytic ROHF optimisation + harmonic
  frequencies; ROHF in the ASE `VibeQC` calculator; `cas_reference="rohf"`;
  atomization / scans.
- ✅ **18e** periodic ROHF (EWALD_3D), Γ-point
  (`run_rohf_periodic_gamma_ewald3d`) **and multi-k / full-BZ**
  (`run_rohf_periodic_multi_k_ewald3d`): reuse the build-validated periodic
  J/K/Madelung + the Roothaan coupling (complex-Hermitian at non-Γ k). Needs
  PySCF.pbc KROHF parity on a build (CLAUDE.md §7).
- ✅ **18f** standalone 3D Gamma-point GPW ROHF (`run_periodic_rohf_gpw`):
  smooth-grid GPW Hartree J plus per-spin exact K and the Ewald
  one-electron/nuclear gauge, coupled through the shared Roothaan SCF loop
  with integer 2/1/0 occupations. The same maintained-preview envelope is
  wired through `run_periodic_job(method="ROHF", jk_method="gpw")`; unsupported
  dimensions, k meshes, gradients, response properties, restart densities,
  and electronic smearing fail closed. `smearing_alpha` is nuclear smoothing,
  not electronic smearing.
- ✅ **18g** 3D Gamma-point GPW ROKS (`run_periodic_roks_gpw` and
  `run_periodic_job(method="ROKS", jk_method="gpw")`): the spin-polarised
  GPW XC build and functional-scaled exact exchange feed the shared Roothaan
  loop, preserving one orbital set and integer 2/1/0 occupations. LDA, GGA,
  meta-GGA, and global hybrids share the Gamma GPW UKS envelope.
- ✅ **18h** 3D full-mesh multi-k pure-DFT GPW ROKS
  (`run_periodic_roks_gpw_multi_k` and the public runner): one complex-safe
  restricted orbital set per k point, integer occupations, exact spin, and
  the existing multi-k GPW LDA/GGA/meta-GGA Bloch-density regimes. Per-k exact
  exchange and therefore multi-k hybrids remain gated.
- 🟡 **18i** periodic ROHF and ROKS on the BIPOLE (multi-k Ewald) route now
  ship: `run_periodic_job(method="ROHF"|"ROKS", jk_method="bipole")` reaches
  `run_rohf_periodic_multi_k_ewald3d` / `run_roks_periodic_multi_k_ewald3d`
  at Gamma and on full Monkhorst-Pack meshes (2026-07-27). Still pending:
  analytic ROKS gradient (molecular XC-gradient primitive), the GAPW route
  for ROHF/ROKS and ROKS on GDF, electronic/metal smearing, and multi-k GPW
  exact exchange (M5b-ROKS / remaining M7b).
- 🟡 **18j** periodic ROHF on the **native GDF** route (2026-08-01):
  `run_krohf_periodic_gdf` (`python/vibeqc/periodic_rohf_gdf.py`) runs
  restricted-open-shell HF on the per-(k_i,k_j) density-fitted cderi cache,
  Gamma reached as a `(1,1,1)` mesh of the one code path. Multiplicity 1
  reproduces `run_krhf_periodic_gdf` exactly; Li/STO-3G/10-bohr Gamma is
  +2.86e-5 Ha from out-of-process PySCF KROHF/GDF and collapses to
  +2.2e-7 Ha with the rsgdf mesh. Exchange, the one O(n_k^2) term, contracts
  the cderi against the occupied MO blocks (`n_occ/nbf` of the flops, two
  GEMMs per k-pair). The 2026-08-02 public increment wires the same engine to
  `run_periodic_job(method="ROHF", jk_method="gdf")` at Gamma and on full
  Monkhorst-Pack meshes. ROKS on GDF, gradients, optimization, restarts and
  smearing remain gated.

#### K-Phase, full k-point sampling support

- ✅ **K1** public ``vibeqc.KPoints`` Python builder with
  Monkhorst-Pack / Γ-centered / shifted / Γ-only constructors;
  classical-MP auto-shift convention (even meshes shift=1, odd
  meshes shift=0); back-compat with native ``BlochKMesh`` via
  ``as_bloch_kmesh()``.
- ✅ **K2** IBZ reduction via spglib, wires the ``use_symmetry=True``
  C++ codepath. Hex/trigonal cells (SG 143-194) refuse non-zero
  MP shifts with an actionable error pointing at ``gamma_centred``
  (the classical (½,½,½) offset breaks the three-fold symmetry).
  ``KPoints.symmetry_reduce()`` builder method.
- ✅ **K3** band-path k-points via **seekpath** (HPKOT convention,
  Hinuma 2017). ``KPoints.band_path(sys)`` autodetects Bravais type
  from spglib spacegroup, returns the canonical Hinuma path; manual
  override via ``scheme="manual"`` + segments.
- ✅ **K4** explicit user-supplied list, ``KPoints.from_list(sys,
  k_frac, weights=None)`` with auto-normalisation.
- ✅ **K5** density-based auto-mesh, ``KPoints.from_kppra(sys,
  n_kpts_per_atom, metallic=False)`` (AFLOW convention, Curtarolo
  2012); ``KPoints.from_kspacing(sys, dk_2pi_per_A)`` (Materials
  Project / ASE); ``KPoints.auto(sys, length_A)`` (VASP ``Auto``
  mode). The ``metallic`` flag bumps the default density and warns
  if smearing isn't enabled.
- ✅ **K7** unified integration, ``run_rhf_periodic_scf`` and
  ``run_rks_periodic_scf`` accept ``KPoints`` directly via the
  boundary helper ``as_bloch_kmesh()``. Demo:
  [examples/periodic/input-k-mesh-convergence.py](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-k-mesh-convergence.py).
- ✅ **K8** AUTO recommender, ``KPoints.recommend(sys)`` classifies
  metal/insulator character (band gap → ``is_metal`` → classifier hook
  → SAFE metal default), *picks* Δk for that character, and returns a
  symmetry-reduced mesh bundled with a recommended ``SmearingOptions``
  and a human-readable rationale. Γ-centred for hex/trigonal; Γ-only
  collapse for large supercells; vacuum axes pinned to 1. Metal BZ
  integration is selectable via ``bz_integration=``, temperature
  smearing (default) or the parameter-free Gilat-Raubenheimer net
  (``"gilat"``: no smearing; runs on the efficient IBZ mesh, expanded
  to the full BZ inside the driver). Optional
  ``verify=True`` convergence ladder (returns a ``KPointConvergence``)
  and a pluggable ``predictor=`` ML Δk hook (no model bundled). Δk and
  gap cutoffs are named constants in one tunable config block. See
  [k_points.md § AUTO mode](user_guide/k_points.md).

#### Other v0.5 deliverables

- ✅ **D1** Grimme D3-BJ dispersion correction, **shipped early in
  0.2.x**. ``D1a`` C++ framework (CN-dependent c6, BJ damping,
  gradient); ``D1b`` reference ``dftd3`` Python backend; ``D1c``
  wired through ``run_job`` + ASE calculator. See
  [Dispersion corrections (D3-BJ)](tutorial/dispersion.md).
- ✅ **M1** ORCA ``.hess`` ASCII writer, drops vibe-qc Hessians
  into the moltui / chemcraft / avogadro / VMD-nmwiz visualization
  ecosystem. ``vq.write_orca_hess(path, mol, hessian_result, ...)``.
- ✅ **M2** Multi-XYZ trajectory + ``.opt`` writers, geometry-
  optimization animations, NEB images, normal-mode movies, MD
  snapshots. ``vq.write_xyz_trajectory(path, frames)``,
  ``vq.write_opt_trajectory(path, frames, energies)`` (auto-formats
  ``"step N  E = …  |grad| = …"`` into the comment line),
  ``vq.normal_mode_trajectory(mol, hess, mode_index)`` helper for
  one-line normal-mode-movie construction. Reads in moltui / OVITO
  / ASE / Avogadro / PyMOL.
- ✅ **ASE-A/B/C** ``vibeqc.ase`` Calculator extensions, exposes
  Hessian, dipole, polarizability, free energy as ASE properties so
  Atoms-driven workflows (ase.vibrations, ase.optimize, ase.neb)
  drive vibe-qc end-to-end without per-property glue code.
- **DOC1-light** *runnable example inputs for the canonical Hessian /
  IR / thermo workflows*. Trimmed from the original DOC1 scope
  (which covered every tutorial) to just the v0.5 headline calcs:
  H₂O HF/STO-3G frequencies, B3LYP IR intensities, NH₃ inversion ΔG.
  Generated logs and visualization files are reproduced locally rather
  than committed.

### `v0.5.x`, observability patch series

The v0.5.0 ship surfaced a recurring need: users running multi-minute
periodic / large-basis calcs want to see *what's happening* during
the run and *where the time went* afterwards. Two sibling patch
releases address this, both extend the v0.5.0 surface, neither
breaks API.

- **v0.5.1, live SCF logging.** *(Shipped 2026-04-30.)*
  Real-time per-iteration progress to stdout / file. Shows the user
  that the SCF is actually doing something during a 20-min build,
  with iteration energy / |ΔE| / max-gradient / DIIS-error /
  per-iteration wall-time. Opt-in via run-options flag and / or env
  var. Closed the user-facing reason the SCF "hangs" silently.

- **v0.5.2, performance / debug log (M3).** Opt-in post-mortem
  breakdown to a sibling ``output.debug`` (or ``output.perf``) file
  recording per-major-code-path:

 - Wall-time + CPU-time per phase (overlap / kinetic / nuclear /
    ERI / J / K / XC quadrature / Bloch sums / DIIS / eigensolver)
 - Memory footprint snapshots at major SCF transitions (peak +
    current RES, swap on Linux)
 - Thread utilisation (``omp_get_max_threads()`` vs measured
    per-phase utilisation; flags under-parallelised hot paths)
 - Cell-list / grid statistics for periodic + DFT runs (n_cells,
    n_grid_points, integral-screening efficiency)
 - Per-iteration SCF breakdown (ΔE, max-grad, DIIS B-matrix
    condition number, error-vector L2 norm, Fock-build time,
    density-update time)

  Implementation: ``python/vibeqc/perf.py`` ``PerfTracker`` + a
  ``perf_log(path)`` context manager. Off by default; enabled via
  ``VIBEQC_PERFLOG=output.debug`` env var, programmatic
  ``with vq.perf_log("output.debug"): ...``, or
  ``run_job(..., perf_log="output.debug")``. Hot-loop C++
  instrumentation guarded behind ``#ifdef VIBEQC_PERFLOG`` so
  release builds without it pay zero runtime cost.

  Pairs with v0.5.1 live logging, live shows progress; perf gives
  the post-mortem breakdown. Together they close the observability
  gap that surfaced in real-world v0.5.0 usage.

**Deferred from the original v0.5 scope**, each gets its own
focused minor release below:

| Item | New milestone |
|------|---------------|
| Periodic atomic gradients (G1) | v0.6 |
| Periodic stress tensor (G2) + slab builder (B1) | v0.7 (stress partly shipped; see the v0.8.0 stress section) |
| Phonons (Phase 21) | v0.8 (Γ-point only; see the v0.8.0 phonons section) |
| Implicit solvation (S1) | v0.9 |
| Grimme D4 dispersion (D2) | v0.10 |
| Geometry-symmetrize pipeline (GS) + DOC1-full | v0.11 |
| vq queue maturity + cluster-bridge docs | v0.12 |
| Generalised-Regular k-grids (K6) | v0.13 |
| ML k-mesh predictor (K-ML) | post-1.0 |

> **Note (2026-09-18, pass 2):** The earlier note here claimed every item
> above "shipped in its assigned milestone". That was wrong for most rows,
> and the "New milestone" column is best read as where the item was *sent*,
> not where it landed. Against this page's own sections:
>
> | Item | Assigned | What actually happened |
> |---|---|---|
> | Periodic atomic gradients (G1) | v0.6 | shipped in v0.6 |
> | Periodic stress (G2) + slab builder (B1) | v0.7 | slipped to the v0.8.0 stress section, and the stress is finite-difference to this day; no analytic periodic stress exists |
> | Phonons (Phase 21) | v0.8 | Gamma-point only, on the GAPW route; dispersion (21b) never shipped |
> | Implicit solvation (S1) | v0.9 | deferred; CPCM shipped in v0.12.0 |
> | Grimme D4 dispersion (D2) | v0.10 | earlier and later: the dftd4 backend in v0.8.0, native D4 validated for H-Ne in v0.15.0 |
> | Geometry-symmetrize (GS) + DOC1-full | v0.11 | GS1-GS3 shipped in v0.13.0 |
> | vq queue maturity + cluster-bridge docs | v0.12 | moved out with the queue; vibe-queue owns it |
> | Generalised-Regular k-grids (K6) | v0.13 | shipped in v0.13 |
> | ML k-mesh predictor (K-ML) | post-1.0 | not shipped, as planned |
>
> The v0.5.x observability patch series (v0.5.1 live logging, v0.5.2
> perf/debug log) did ship.


## Milestones and plans, `v0.6.0` to `v0.15.0`

These sections are in the order they were written, so release inventories and
the plans that preceded them interleave. Headings are unchanged from the
roadmap, so a link that used to reach one of these anchors on `roadmap.md`
reaches it here.

### `v0.6.0`, periodic atomic gradients, *Pulay's Owl*

> **Codename: Pulay's Owl**, Pulay's correction terms (1969) are
> the conceptual core of this release; Pulay also gave us DIIS
> (1980), which the SCF stack underneath the gradient pass relies
> on. Owl: nocturnal, sees what others miss, sub-meV displacement-
> derived forces.

🎯 **Headline feature**, analytic forces on solids. Unlocks
geometry optimization for surfaces, molecular crystals, defect
cells. The single biggest "vibe-qc can now do solids the way ORCA
does molecules" deliverable; gates v0.8 stress and v0.9 phonons.

- ✅ **G1a** Hellmann-Feynman force (electronic + nuclear, no
  Pulay), Γ-only RHF gradient.
- ✅ **G1b** Pulay corrections, basis-set-dependent ∂χ/∂R terms via
  the libint deriv-order=1 path (already vendored for molecular
  gradients). Γ-only RKS gradient via molecular-fallback XC Pulay
  (LDA exact; GGA σ-coupled term as v0.6.x patch).
- ✅ **G1c** Multi-k RHF + RKS gradient drivers via Bloch-folded
  density.
- ✅ **G1d** Open-shell UKS multi-k gradient (pure DFT only;
  hybrid-UKS via per-spin periodic K is v0.6.x).
- ✅ **G1e** ASE bridge, `vibeqc.ase_periodic.atoms_to_periodic_system`
  + `periodic_forces` round-trip eV/Å forces with Newton's-3rd-law
  obeyed.
- ⏳ **G1f** Validation against CRYSTAL gradient outputs on small
  reference systems (LiH, MgO, diamond Si). Tracked as v0.6.x.
- ⏳ **G1a-2** Periodic K-piece gradient routing bug, masked by
  Sx4 default screening (`schwarz_threshold_forces = 1e-14`), root
  cause still tracked. Setting `schwarz_threshold_forces = 0.0`
  reproduces the historical 6e-3 Ha/bohr disagreement vs FD on
  H-chain a=2 Å. Production users with default thresholds are
  unaffected; HF / hybrid-DFT users probing the regression handle
  should use the FD reference until the root-cause fix lands.

Sibling work shipped alongside the gradient flagship in this same
v0.6.0 release:

- ✅ **Sx4**, Cauchy-Schwarz screening on the gradient ERI pass
  (`eri_lattice_gradient_contribution`). Default
  `schwarz_threshold_forces = 1e-14` (100× tighter than the
  energy-side default since plain Schwarz on derivatives is
  non-rigorous, matching CP2K's `EPS_SCHWARZ_FORCES` convention).
  Per-quartet + cell-level skip just like the energy side.
- ✅ Banner print at SCF entry on every periodic SCF surface
  (RHF / RKS Γ-only + multi-k + UKS) so the version + codename
  shows in any `.out` file before the SCF header.
- ✅ G1d export (`compute_gradient_periodic_uks_multi_k`) at top-level
  `vibeqc.*` namespace; was inadvertently dropped during a rebase
  before v0.5.5.

### `v0.6.x`, molecular ORCA parity (sibling validation track)

🎯 **Same idea as the v0.7 PySCF.pbc parity, but for the molecular
stack and against ORCA.** Runs in parallel with v0.7 and ships in
patch releases as fixtures land.

Why both PySCF *and* ORCA: they implement HF / DFT / MP2 with
*different* code paths (different integrators, different DIIS
variants, different XC quadrature implementations). A 1:1 vibe-qc
match against both is a much stronger correctness signal than
against either alone. ORCA also exposes the SCF iteration trace
cleanly via the ``.out`` file, so iter-by-iter SCF convergence
(energy, |ΔE|, max-grad, DIIS error) can be compared in addition
to the final total energy.

**Settings discipline** (per user): vibe-qc and ORCA must use the
*same* SCF settings, same convergence thresholds, same DIIS
config, same XC grid, same initial guess (Hcore vs SAD). Start
with loose criteria; tighten step by step as parity holds. ORCA
defaults are documented in §SCF Convergence of the ORCA manual;
mirror them by:
  * RHF/UHF: ``conv_tol_energy = 1e-6 Ha``, DIIS subspace 8,
    Hcore guess.
  * RKS/UKS: same, plus ``grid.level = 4`` (ORCA's "Grid4").
  * MP2: ``conv_tol = 1e-7 Ha`` for the underlying SCF.

- **MOL1** Test fixtures: H₂, H₂O, CH₄, NH₃, HF, CO, N₂,
  formaldehyde, methanol, ethanol (small molecules ORCA's
  tutorials ship with). Multiple basis sets: STO-3G, 6-31G(d),
  cc-pVDZ, def2-TZVP. Methods: RHF, UHF (radicals, OH·, CH₃·,
  NO·), RKS / UKS for LDA / PBE / B3LYP, RMP2, UMP2.
- **MOL2** ORCA parity runs through subprocess fixtures. External
  outputs are generated on demand and kept out of the committed tree;
  re-run the parity matrix when ORCA version bumps.
- **MOL3** ``tests/test_molecular_orca_parity.py``, parametrised
  over MOL1 fixtures, asserts vibeqc ↔ ORCA agreement on:

  | Quantity | Loose (initial) | Tight (final) |
  |----------|-----------------|---------------|
  | Final total energy | 1e-4 Ha | 1e-7 Ha |
  | SCF iter count | within ± 5 | within ± 2 |
  | Per-iter energy trace | 1e-3 Ha | 1e-6 Ha |
  | Final MO energies (sorted) | 1e-3 Ha | 1e-6 Ha |
  | Mulliken charges | 0.05 e | 1e-3 e |

  Tighten one row at a time once each loosened tolerance passes
  on all fixtures.

- **MOL4** ORCA ``.out`` parser
  (``tests/fixtures/orca_reference/_parser.py``) so parity tests
  read cached ORCA output rather than re-running ORCA on every CI.

Existing PySCF molecular parity (since v0.4.x), RHF / UHF / RKS /
UKS / MP2 / UMP2 on H₂ / H₂O / CH₄ × {STO-3G, 6-31G*, cc-pVDZ} to
1e-10 Ha total + 1e-9 MO, is solid (78 tests, all green at
v0.6.2). ORCA parity adds an *independent* second reference and
the iteration-trace match.

### `v0.7.0`, periodic SCF correctness + PySCF.pbc parity

> **Löwdin's Compass.**
>
> **Promoted to flagship** after v0.6.1 / v0.6.2 surfaced that the
> v0.6.x periodic SCF is correct in the atomic limit but *off by
> orders of magnitude* on dense ionic crystals (LiH conventional
> cell: ~−1060 Ha vs ~−32 Ha expected). The Madelung-correction
> formula in v0.6.1 fixes the single-charge-in-vacuum case but
> doesn't capture the structural Madelung of complex ionic
> arrangements. Periodic stress + slab builder bumped to v0.8.

🎯 **Headline feature**, vibe-qc periodic SCF gives the same
absolute energies as PySCF.pbc on realistic systems, then beats
it on speed via CRYSTAL-style space-group symmetry exploitation.

#### Phase 1, PySCF.pbc parity (port their gauge convention)

The bug is in the gauge consistency between V_nuc lattice sum and
Ewald-3D Hartree J. PySCF's ``pyscf.pbc.df.fft.FFTDF`` and
``pyscf.pbc.tools.pbc.madelung`` already solve this, port their
convention into vibe-qc's
``vibeqc.ewald_composed.build_j_ewald_3d`` +
``vibeqc.nuclear_repulsion_per_cell``.

- **PBC1a** Audit ``pyscf.pbc.df.fft.FFTDF`` and
  ``pyscf.pbc.tools.pbc.madelung``. Document the gauge convention
  they use, single neutralizing background applied to nuclei +
  electrons together, not separately. Document where vibe-qc
  diverges.
- **PBC1b** Refactor ``build_j_ewald_3d`` to use the PySCF
  convention. Drop the G=0 pinning of J alone in favour of the
  combined-system neutralizing background. Replace the
  v0.6.1-shipped ``madelung_energy_correction`` formula with the
  cell-dependent Madelung from
  ``pyscf.pbc.tools.pbc.madelung``-equivalent.
- **PBC1c** Rebuild the multi-k Ewald drivers on top of the
  refactored J. Verify Bloch-sum consistency.
- **PBC1d** Rebuild the periodic XC contribution on the same
  gauge, V_xc must integrate against the same density that J
  sees.

#### Phase 2, PySCF parity test set

Realistic systems with PySCF reference inputs/outputs checked into
the repo. Both vibe-qc and PySCF run side-by-side; absolute energy
agreement to ≤ 1 mHa per cell.

- **PBC2a** **Test fixtures, start with what PySCF converges
  trivially.** The point is to match PySCF where PySCF runs
  cleanly with default settings (no SAD / SAP guess, no level
  shift, no aggressive damping). If PySCF needs a feature, we
  understand we need it too, but we add it when its absence
  blocks parity, not preemptively. Easy systems first
  (high-symmetry, gapped insulators, primitive or small
  conventional cells):

  | System              | Cell      | Atoms | Gap   | Why easy |
  |---------------------|-----------|-------|-------|----------|
  | H₂ in big box       | cubic     | 2     | huge  | molecular limit, trivial |
  | Solid Ne (FCC)      | cubic conv| 4     | huge  | rare-gas, no bonding ambiguity |
  | LiH rocksalt        | cubic conv| 8     | wide  | textbook ionic insulator |
  | NaCl rocksalt       | cubic conv| 8     | wide  | textbook ionic insulator |
  | MgO rocksalt        | cubic conv| 8     | wide  | gap ~7 eV, classic CRYSTAL test |
  | Diamond C           | cubic conv| 8     | wide  | classic covalent, large gap |
  | Si diamond          | cubic conv| 8     | mid   | standard semiconductor benchmark |
  | BN cubic            | cubic conv| 8     | wide  | mixed covalent / ionic |

  Hard systems (need fancy features) are explicitly OUT of the
  PBC2 scope and added only when needed:

  | System       | Why hard                       | Feature blocked on |
  |--------------|--------------------------------|--------------------|
  | Al / Cu metal| metallic, Fermi smearing       | smearing-aware DIIS / OT |
  | TM oxides    | open-shell, broken-symmetry    | SAD-UHF + GUESSMIX |
  | Surfaces     | 2D Ewald + slab                | Shipped: vacuum-free 2D Ewald SCF (Parry/de Leeuw) for RHF/RKS/UKS, Gamma + multi-k; `dim=2` is the AUTO route and bulk builders fail closed; slab-GDF 2D-truncated Coulomb (closed-shell RHF/RKS energies + analytic gradients + `optimize=True`, 2026-07-30). Open: gradients on the direct slab route, slab smearing/UHF, GPW 2D-truncated Coulomb |
  | FCC primitive cells | non-orthorhombic        | native FFTDF/GDF parity and benchmarks |

  STO-3G first for speed; pob-TZVP after STO-3G parity holds.

  **Sourcing fixtures from PySCF's tutorials.** Use the
  smallest cells that PySCF.pbc's own tutorials / examples ship
  with, they're already laptop-friendly (run in seconds to a
  minute). Concretely:

 - ``examples/pbc/00-intro.py``           → H₂ chain at small a
 - ``examples/pbc/03-band_structure.py``  → diamond C (8-atom conv)
 - ``examples/pbc/12-mole_2_cell.py``     → small Si supercell
 - ``examples/pbc/17-disabled_for_isdf.py`` → various small cells
 - ``examples/pbc/df_intro.py``            → small LiH

  Cell-size budget for the parity tests: each fixture must run
  on a developer laptop (8-16 cores, 16-32 GB RAM) in under
  ~60 s for vibe-qc and PySCF combined. That keeps the CI
  pipeline fast and lets the fixtures double as fast iteration
  cases during the gauge refactor.
- **PBC2b** Each fixture has a paired
  ``examples/parity/<system>/pyscf_reference.py`` and
  ``examples/parity/<system>/vibeqc.py``, plus a checked-in
  ``pyscf_reference.energies.json`` so the parity test runs
  against the cached PySCF numbers without re-running PySCF on
  every CI. Re-run periodically when PySCF version bumps.
- **PBC2c** ``tests/test_periodic_pyscf_parity.py``,
  parametrised over all PBC2a fixtures, asserts vibe-qc ↔ PySCF
  agreement to 1 mHa absolute.
- **PBC2d** Extend to pob-TZVP (the basis set that's standard
  for ionic-crystal PBC work) once STO-3G parity is solid. pob-
  TZVP / pob-DZVP-rev2 / pob-TZVP-rev2 already ship in
  ``basis_library``.

#### Phase 3, CRYSTAL-style speedups

Once PySCF parity is solid, add the CRYSTAL-style
symmetry-driven optimisations that make CRYSTAL fast on
LCAO PBC:

- **PBC3a** Space-group symmetry on the unit-cell density,
  symmetrise D after each SCF iteration so symmetry-equivalent
  blocks share one set of integrals. Mirrors CRYSTAL23's
  ``SYMM`` defaults.
- **PBC3b** Symmetry-equivalent atom + shell-pair pruning in the
  ERI lattice sum. CRYSTAL exploits ``f_n`` (number of operators
  mapping pair n → equivalent pair) to reduce the unique
  shell-pair count by a large factor (3-12× depending on
  spacegroup).
- **PBC3c** TOLINTEG-style 5-vector thresholds (overlap,
  penetration, exchange-overlap, pseudo-overlap; see CRYSTAL23
  manual §SCF) layered on top of v0.5.5 / v0.5.6 Schwarz.
- **PBC3d** Symmetry-equivalent k-point folding in the Bloch
  sums (already partially in place via ``KPoints.symmetry=True``;
  full integration with the symmetrised D update).

#### Stress + slab, moved to v0.8

The original v0.7 stress + slab work moves to v0.8 because the
gauge-consistency fix in PBC1 changes the periodic energy
expression and therefore the Pulay-aware analytic stress derivation.
We need correct absolute energies before the energy *gradient*
with respect to lattice strain is well-defined.

### `v0.8.0`, periodic stress tensor + slab builder

> **Codename TBD** *(candidates: Born's Hare, Neese's Magpie,
> Vinet's Pangolin)*.

> **Status, 2026-09-13: partly shipped before the repository split.**
> Finite-difference stress over strain exists for GPW and GAPW
> (`compute_stress_gpw`, `compute_stress_gapw`), the semiempirical routes and
> MACE, and the ASE GPW calculator exposes `stress`. There is no analytic
> (G3a) stress, none on the GDF or BIPOLE routes, and cell optimization is
> refused on GDF, slab GDF and AICCM. The equation of state is a third-order
> Birch-Murnaghan fit helper (`vibeqc.eos.fit_birch_murnaghan`), with no
> Vinet form and no volume-sweep driver.

🎯 **Headline feature**, periodic stress tensor σ_αβ → cell-
parameter optimization, exact equation-of-state, surface energies.
Was originally v0.7; promoted out of the way of the periodic-SCF-
correctness flagship.

- **G3a** Analytic stress from the SCF energy (Pulay-aware).
  **Still open.** What shipped instead is a finite-difference stress
  (`compute_stress_gpw` / `compute_stress_gapw`, and one native batched FD
  pass on the semiempirical routes); no ab initio route has an analytic
  periodic stress.
- **G3b** ASE calculator `stress` property; `vc-relax` routine
  driven by ``ase.optimize.BFGS`` on cell + coordinates.
  **Half shipped:** the ASE `stress` property ships on the GPW calculator
  and returns the FD stress in eV/A^3. `vc-relax` does not: the runner
  refuses `optimize_cell=True` on GPW, GAPW, GDF and BIPOLE, and only the
  full-k semiempirical routes relax a cell.
- **G3c** Equation-of-state utilities (Birch-Murnaghan, Vinet) on
  a few-volume EOS sweep. **Half shipped:** `vibeqc.eos` provides the
  third-order Birch-Murnaghan fit with ill-conditioned-fit guards; Vinet
  and the sweep driver are open.
- **B1** slab / surface builder, thin wrapper over
  ``ase.build.surface``, gated on **C1** so the resulting SCFs
  actually converge for the metallic surfaces it produces.

### `v0.8.0`, phonons

> **Codename TBD** *(candidates: Debye's Stork, Born's Hedgehog)*.

> **Status, 2026-09-13: 21a shipped before the repository split; 21b and 21c
> remain open.** Γ-point finite-difference phonons exist on GAPW
> (`PhononCalculator`, `compute_dynamical_matrix_fd`) and on the BIPOLE route
> that basis optimization uses, and `run_periodic_job(hessian=True)` returns
> finite-difference frequencies. There is no phonon dispersion, no
> quasi-harmonic thermodynamics, and no ZPE from `run_periodic_job`.

🎯 **Headline feature**, Γ-point + finite-displacement phonon
dispersion + quasi-harmonic thermodynamics for solids.

- **21a** Γ-only periodic FD Hessian, straight extension of the
  molecular FD Hessian (17a-1) to periodic, single supercell.
- **21b** Phonon dispersion, FD on G1 forces across a phonopy-style
  supercell, dynamical matrix at arbitrary q-points.
- **21c** Quasi-harmonic approximation thermodynamics, V(T), C_p,
  thermal expansion from the EOS fit plus 21b phonons. The BM3 fit
  (`vibeqc.eos.fit_birch_murnaghan`) and Gamma-point phonons ship; 21b
  dispersion is what this still waits on.

### `v0.8.0`, basis-set library expansion

> **Codename TBD** *(candidates: Pople's Beaver, Dunning's Marmot,
> Weigend's Mole)*.

🎯 **Headline feature**, first-class support for every basis-set
family discussed in [user guide § basis sets](user_guide/basis_sets.md).
Shipping every recognised family closes the prerequisite gap that
gates GMTKN55 / S22 / S66 / TM-benchmark ingestion in the
regression-suite test-set queue further down this page (item 2:
"BSE-driven basis-set ingestion").

The driving design choice: adopt the **Basis Set Exchange (BSE)**
JSON format as the canonical loader so every family below works
without hand-coded `.g94` files. The BSE catalog (Pritchard et al.
2019) is authoritative; vibe-qc tracks it.

- **BS1** **Karlsruhe def2 family** (the molecular default). def2-SVP,
  def2-TZVP, def2-TZVPP, def2-QZVP, def2-QZVPP + def2-ECP for
  Rb-Rn + diffuse-augmented variants (def2-XVPD per Rappoport &
  Furche 2010, ma-def2-XVP minimally-augmented per Truhlar-Zheng).
- **BS2** **Dunning correlation-consistent**. cc-pVnZ for
  n ∈ {D,T,Q,5,6}, aug-cc-pVnZ, cc-pV(n+d)Z (tight-d for second-row
  hypervalent per Dunning 2001), cc-pwCVnZ for explicit core-valence.
  Generally contracted; requires the integral path to handle that
  efficiently.
- **BS3** **Jensen polarisation-consistent**. pcseg-n for
  n ∈ {0,1,2,3,4}, aug-pcseg-n. The pure-DFT-optimal segmented
  alternative to def2-TZVP; Pitman 2024 endorses pcseg-2 as the best
  TZ basis for DFT thermochemistry.
- **BS4** **Pople family completion + deprecation warnings**.
  STO-3G, 3-21G, 6-31G(d,p), 6-31++G(d,p), 6-311G(d,p),
  6-311+G(2d,p), 6-311+G(2df,p). Surface a UI warning on the 6-311G
  family per Pitman 2024 and on unpolarised DZ in general.
- **BS5** **ANO-RCC and ANO-R** for multireference. Generally
  contracted; gates CASSCF / CASPT2 / NEVPT2 ingestion (Roos 2005,
  Zobel 2020).
- **BS6** **Relativistic counterparts**. x2c-SVPall / x2c-TZVPall /
  x2c-QZVPPall (Pollak-Weigend 2017, Franzke 2020), dhf-SVP /
  dhf-TZVP / dhf-QZVP. Pairs with an X2C / DKH2 / ZORA Hamiltonian
  in the relativistic phase.
- **BS7** **Composite-method bases**. def2-mSVP, def2-mTZVP,
  def2-mTZVPP, vDZP, each tied to its 3c composite (HF-3c,
  PBEh-3c, B97-3c, r²SCAN-3c, ωB97X-3c). Gates the composite-method
  shortcuts (`r2scan-3c` keyword that bundles basis + D4 + gCP +
  modified short-range correction).
- **BS8** **Periodic Gaussian, extras**. Already-shipped pob-*
  family stays; add MOLOPT (SZV / DZVP / TZVP / TZV2P) for
  cross-code parity with CP2K, and dcm-TZVP for system-specific
  reference work.
- **BS9** **Auxiliary basis sets for RI / DF**. def2/J, def2/JK,
  cc-pVnZ-RIJK, def2-TZVP/C. Required prerequisite for the
  RI / DF acceleration track (`v0.7.x` of the acceleration roadmap).
- **BS10** **Property-specific**. pcS-n (NMR shielding, Jensen
  2008), pcJ-n (J-coupling, Jensen 2006), x2c-TZVPall-s
  (relativistic NMR/EPR, Franzke 2019), EPR-II / EPR-III, Sadlej
  (polarisability). Wired alongside their property machinery
  (NMR / EPR / response).
- **BS11** **F12 orbital + CABS auxiliary bases**. cc-pVnZ-F12
  (n = D, T, Q) plus their complementary auxiliary basis sets
  (cc-pVnZ-F12-CABS / cc-pVnZ-JKFIT), required inputs for the
  explicitly correlated MP2-F12 work (Psi4 sweep, below).

This sweep is the prerequisite for several other v0.8 / v0.9 / v1.0
items: composite-method shortcuts (BS7), high-throughput TM
benchmarking (BS1 + BS6), property runs (BS10), and DLPNO-CCSD(T)
benchmark targets (BS1 + BS9 RI auxiliaries). User guide:
[`user_guide/basis_sets.md`](user_guide/basis_sets.md) is the
authoritative survey + recommendation surface.

### `v0.8.0`, XC functional library expansion

> **Codename TBD** *(candidates: Becke's Hummingbird, Perdew's Pangolin,
> Grimme's Gecko)*.

🎯 **Headline feature**, first-class support for every XC functional
the test-set queue and the v0.9 composite-method track depend on,
with the range-separated-hybrid (RSH) and meta-GGA machinery wired
in. libxc resolves any registered name today, but several
chemistry-critical functionals need infrastructure beyond a name
lookup, RSH needs erf/erfc Coulomb splitting in the K build,
double-hybrids need MP2 wired into the SCF, custom hybrids like
PW1PW need their fractions registered, and grid coverage for
post-PBE meta-GGAs needs auditing.

This entry consolidates "Functional additions", prerequisite #4 of
the regression-suite test-set queue further down this page, into a
formal v0.8.0 milestone. Pairs with the basis-set sweep (BS7 wires
the composite-method bases; F-series here wires the matching
functionals).

- **F1** **Range-separated hybrid machinery** (HSE06, ωB97X-V,
  ωB97M-V, ωB97X). Needs the K-build to handle range-separated
  Coulomb (erf / erfc kernels) so the short-range and long-range
  Fock exchange can be mixed at different fractions. ωB97M-V is
  what current benchmarks (Bursch 2022, Najibi-Goerigk 2018) settle
  on as the best general-purpose hybrid; HSE06 is the de-facto
  solid-state hybrid. Both are gated behind RSH.
- **F2** **PW1PW (Bredow's 1-parameter hybrid)**, 20 % HF + 80 %
  PW91 exchange, PW91 correlation. Not a libxc preset; requires
  registering the custom mixing as a named functional. **Gates
  parity against the pob-TZVP paper SI**
  ([Peintinger, Vilela Oliveira, Bredow 2013, *J. Comput. Chem.*
  **34**, 451](https://doi.org/10.1002/jcc.23153), published HF
  *and* PW1PW totals for ~60 reference compounds; the regression
  suite consumes Table 2 once PW1PW is wired). Smallest-effort
  item in this entry.
- **F3** **r²SCAN and the r²SCAN-3c stack**. Meta-GGA τ-dependent
  XC; r²SCAN itself ships in libxc but the periodic Becke-grid
  path needs auditing for τ-density evaluation correctness.
  **Gates** the headline 3c composite (`r2scan-3c`), combined
  with BS7 (def2-mTZVPP) + D4 dispersion + gCP, this becomes the
  Grimme-recommended "Swiss army knife" default for routine work.
- **F4** **Double-hybrid functionals** (DSD-PBEP86-D4, B2PLYP-D4,
  PWPB95-D4, ωB97M(2), revDSD-PBEP86-D4). Double-hybrids do MP2 on
  top of a hybrid SCF; this requires the SCF→post-SCF dispatch
  (already exists) and a configurable MP2 fraction. **Gates** the
  GMTKN55 top-tier benchmarking (these functionals win on
  WTMAD-2 by ~50 % over the best plain hybrid).
- **F5** **D4 dispersion** (Caldeweyher et al. 2019). vibe-qc
  ships D3(BJ) today; D4 is the modern successor with charge-
  dependent dispersion coefficients and is what every 3c composite
  + every modern functional benchmark expects. Companion to F1-F4
  rather than a separate functional, but listed here because
  composites depend on `(functional + D4)` as a single unit.
- **F6** **libxc coverage audit + UI surfacing**. libxc 7.x
  registers ~600 XC functionals; the
  [user-guide functional table](user_guide/functionals.md) lists
  six. Audit which of the chemistry-relevant ones (Bursch 2022
  table, Najibi-Goerigk WTMAD-2 winners, Mardirossian-Head-Gordon
  ωB97 family) are reachable today and surface them by name with
  per-functional notes (kind, HF fraction, D-correction default,
  recommended use-case). Drive-by: surface a deprecation warning
  on functionals that newer benchmarks have superseded (e.g. plain
  B3LYP-without-dispersion for noncovalent binding).

This sweep gated the composite-methods work, and that work has since
shipped: `run_job(method=...)` takes the recipe names
`vibeqc.list_composites()` returns. See the
[composite 3c methods](#composite-3c-methods) section for what is molecular
only and what still needs recalibration. User guide:
[`user_guide/functionals.md`](user_guide/functionals.md) is the
authoritative recommendation surface.

### `v0.8.0`, molecular methods polish (RIJCOSX + JKBuilder + B3LYP-VWN5)

> **Codename TBD** *(candidates: Whitten's Bridge, Kohn's Tortoise,
> Becke's Sandpiper)*.

🎯 **Headline feature**, molecular SCF / J-K dispatch unified
behind a polymorphic `JKBuilder` abstract base; RIJCOSX
(chain-of-spheres exchange, Neese 2009) shipped for all four SCF
flavours with analytic gradient (Bykov 2015 frozen-grid); B3LYP
convention corrected to VWN5 (Becke 1993 / Stephens 1994 / Gaussian
/ ORCA convention). Sourced from the molecular methods main chat's
8-stacked-feature-branch handover (2026-05-09).

Sub-items:

- **MM1** **`JKBuilder` polymorphic dispatch** (branch
  `feature/jk-builder-refactor`, commit `09ce308`). Polymorphic
  abstract base with concrete classes `FourIndexJKBuilder` /
  `DFJKBuilder` / `COSXJKBuilder` / `PeriodicGammaJKBuilder`. Each
  factory returns `shared_ptr<JKBuilder>`. Clean architectural
  unification of molecular + periodic-Γ J/K dispatch.
- **MM2** **`PeriodicGammaJKBuilder` adapter** (branch
  `feature/periodic-jk-builder`, commit `3c80016`). Plugs the
  existing periodic-Γ Fock build into the new `JKBuilder` base.
- **MM3** **`run_*_scf_with_jk` SCF body factor** (branches
  `feature/run-rhf-scf-with-jk` `a5d5bf4` and
  `feature/run-scf-with-jk-uhf-rks-uks` `31cb3cd`). Same SCF body
  drives RHF / UHF / RKS / UKS molecular AND periodic-Γ
  calculations, the only difference is the `JKBuilder`
  instance passed in.
- **MM4** **RIJCOSX SCF + analytic gradient** (branch
  `feature/rijcosx`, commit `2dacff6`). Chain-of-spheres
  exchange (Neese, Wennmohs, Hansen, Becker, *Chem. Phys.*
  **356**, 98, 2009); analytic gradient via the frozen-grid
  approximation (Bykov, Petrenko, Stoychev, Auer, Neese,
  *Mol. Phys.* **113**, 1961, 2015). All four SCF flavours;
  sparse default 110-Lebedev × Treutler-Ahlrichs radial COSX
  grid; Q-junction overlap fit. Validation: vibe-qc vs ORCA
  6.1.1 RIJCOSX on glycine / def2-TZVP, max |Δ| = 0.13 mHa
  across 6 K-using methods (RHF / UHF / RKS-B3LYP / RKS-PBE0 /
  UKS-B3LYP / UKS-PBE0).
- **MM5** **`b3lyp` → libxc id 475 (VWN5)** (commit `822e350`,
  lands the intent of branch `feature/b3lyp-vwn5-fix`,
  `b3273d6`). **Behaviour break**: B3LYP energies move by
  ~10-15 mHa per heavy atom. Was id 402 (VWN-RPA, libxc
  default and Gaussian convention); now id 475 (VWN5, the
  ORCA / ADF convention). Pre-fix vs ORCA's bare `B3LYP`
  disagreed by ~150 mHa on 10-atom organics. New **`b3lyp/g`
  alias** → libxc id 402 selects the Gaussian-compatible VWN3
  variant, mirroring ORCA's `B3LYP` / `B3LYP/G` keyword pair.
  **Document loudly in v0.8.0 release notes + the functionals
  user-guide page**, plus a cross-code-parity sidebar in the
  JCC release paper.
- **MM6** **`pbe0` alias** (branch `feature/pbe0-functional`,
  commit `4c5e4c4`). Resolves to libxc id 406 (PBEh).
- **MM7** **New options fields**: `cosx: bool`, `cosx_grid:
  GridOptions` on RHF / UHF / RKS / UKS Options + GradientOptions.
  Default grid via `vq.cosx.default_cosx_grid_options()`.
- **MM8** **New COSX C++ kernels**: `compute_cosx_k`,
  `compute_cosx_k_gradient_contribution`.

**Open known bug at merge time** (carries into v0.8.x
maintenance): DF gradient on glycine / def2-TZVP / RHF
disagrees with direct gradient by ~115 mHa/bohr. Energies
match to ~60 µHa, so the bug is in gradient assembly. Likely
d/f-shell deriv kernel issue; appears only when high angular
momentum + density fitting combine. Test suite covers
def2-SVP on ≤ 5 atoms (where the bug doesn't trip).
**Workaround**: use `density_fit=False` for production
gradient runs above ~50 BFs / def2-TZVP-class basis sets.
Spawned as separate task; goes to the molecular methods
continuation chat for the fix.

**Continuation chats** (post-handover):
- "Continue vibe-qc molecular methods main chat", picks up
  EDIIS+DIIS hybrid, SOSCF/TRAH, periodic DF, periodic COSX,
  multi-k JKBuilder generalisation, wave-2 functionals
  (M06-2X, ωB97X-D, TPSSh, r²SCAN).
- "PW1PW + weighted-sum xc alias path", implements PW1PW
  (gates parity against the pob-TZVP paper SI Table 2) plus
  the weighted-sum aliasing infrastructure for HSE-3c, B97-3c,
  etc.

### `v0.8.0`, periodic SCF chain (restricted HF/DFT, all Bravais, no-fit + DF + RIJCOSX)

> **Codename TBD** *(candidates: Grimme's Gecko, Wannier's
> Marmoset, Brillouin's Gerbil)*.

🎯 **Headline feature**, restricted (closed-shell) periodic
HF and DFT made rock-solid for v0.8.0, exercised across **all
Bravais lattices** (cubic / tetragonal / orthorhombic /
hexagonal / monoclinic / triclinic, in 1D / 2D / 3D) and
all three Coulomb-method modes:

1. **Standard (no fitting)**, direct 4-center periodic ERIs.
2. **DF (GDF)**, native Gaussian density fitting.
3. **RIJCOSX**, RI-J + chain-of-spheres semi-numerical
   exchange, with the four CRYSTAL/ORCA-style tuning knobs
   (COSX grid density, ``!NOCOSX`` last-cycle, no-RI last-
   cycle, UHF/RHF scaling note).

**Scope correction (2026-05-13):** open-shell UHF / UKS (Γ
and multi-k) slip to **v0.8.x**. The earlier v0.8.0 bundle
that included UHF/UKS predates the current rock-solid-restricted-
first directive. The PSCF1 + PSCF5 sub-items below describe
work that needs to be rewritten on the new gamma-GDF scaffold
(the old PySCF-Lpq feature branches were retired on
2026-05-13).

Sourced from the periodic-QC main-dev chat (current ownership
chain documented in
`project_periodic_rhf_gdf_spike.md`).

Sub-items:

- **PSCF0** (NEW, restricted-foundation) **Restricted RHF/RKS rock-solid on all Bravais**.
  The v0.8.0 release gate. Three Coulomb modes, applied uniformly:
  * **PSCF0.std**, Standard (no fitting): direct 4c-ERIs.
    Audit DIRECT_TRUNCATED + EWALD_3D across all Bravais; CRYSTAL14
    cross-check on cubic / hexagonal / triclinic.
  * **PSCF0.df**, DF (GDF): native Lpq via
    ``aux_basis.build_lpq_native`` (Γ) and
    ``build_lpq_bloch_native`` (multi-k, shipped on
    ``feature/multi-k-gdf-scf-loop`` 2026-05-13 with the SCF
    loop).
  * **PSCF0.cosx**, RIJCOSX: RI-J via the DF path + chain-of-
    spheres exchange on the periodic Becke grid. Molecular
    RIJCOSX rewritten from scratch on the new ``JKBuilder``
    architecture (the stale ``feature/rijcosx`` branch was
    retired). Four tuning knobs (COSX grid density,
    ``!NOCOSX`` last cycle, no-RI last cycle, UHF/RHF scaling
    note).
- **PSCF1** *(SLIPPED TO v0.8.x, open-shell)*
  **`run_uhf_periodic_gamma_gdf`** + `PeriodicUHFGDFResult`
  (α/β fields, ⟨Ŝ²⟩). **Status 2026-05-13:** the original
  ``feature/uhf-periodic-gdf`` branch (and its validation
  companion) was retired on 2026-05-13 because it was built
  on the old PySCF-Lpq scaffold that codex replaced. Native
  UHF Γ-GDF is a fresh rewrite on top of the new
  ``periodic_rhf_gdf.py`` foundation, ~1 day. Historical
  validation numbers from the old branch are preserved below
  for reference. **Validation parity numbers
  (refreshed 2026-05-09 from the UHF-A.1 chat historical design record)**:
 - Closed-shell sanity: MgO sto-3g UHF (mult=1) ≡ RHF to
    **1.4×10⁻¹² Ha**.
 - H atom in 20 Å vacuum-padded cubic cell, mult=2: vs
    `pyscf.pbc.scf.UHF + GDF` → **5.6×10⁻¹⁷ Ha (literally
    machine zero)**.
 - OH radical 9 e⁻, mult=2: vs PySCF UHF → **2.5×10⁻⁹ Ha**.
 - **Iter-1 J/K parity vs `pyscf.pbc.UHF.get_jk` to machine
    precision** (max\|ΔJ\|≤10⁻¹⁴, max\|ΔK_σ\|≤4×10⁻¹⁵) on
    H, OH, NO, MgO closed-shell.

  **Caveat at this release**: GDF UHF lacks Saunders-Hillier
  level shift; ‖[F,DS]‖ can stall on UHF radicals (OH, NO)
  even though energy parity is sub-nHa. `result.converged`
  may report `False` while the energy is correct. **Workaround**:
  trust the energy if the iter trace stabilises; OR fall back
  to `run_uhf_periodic_gamma_ewald3d` (legacy, has level
  shift) for cells where the Ewald path is valid; OR wait for
  the level-shift port (~10 lines, slated for v0.7.x). NO
  converging to a different stationary point than PySCF is a
  known UHF instability, not a vibe-qc bug, H-atom and OH
  suffice as parity oracles.

  **Design point worth noting**: one J build per iter on
  `D_α + D_β`, two per-spin K builds; per-spin Madelung
  correction `madelung·S·D_σ·S` matches PySCF's
  `_ewald_exxdiv_for_G0` per-spin convention.
- **PSCF2** **Multi-k periodic SCF** (restricted, closed-shell):
  `run_krhf_periodic_gdf(..., kmesh)` +
  `run_krks_periodic_gdf(..., kmesh, functional)` + result
  classes. **Status 2026-05-13:** native SCF loop landed on
  ``feature/multi-k-gdf-scf-loop`` @ `c19e451`, built on top
  of ``aux_basis.build_lpq_bloch_native(q_cart)``. 15 local
  tests green (cubic / hexagonal / triclinic; Γ delegation
  matches gamma driver to 3.6×10⁻¹⁵ Ha on H₂). Real-system
  parity vs CRYSTAL14 + PySCF KRHF on compute-reference is the v0.8.0
  gate.
- **PSCF3** **Streaming-Lpq for multi-k production scale**.
  Paper-grade asbestos blocker: dense `Lpq[k1,k2]` storage
  hits multi-TB on real systems (lizardite/[2,2,2] ≈ 553 GB;
  antigorite-m17/[2,2,2] ≈ 5 TB). Streaming chunks the Lpq
  tensor along k-pairs; chemistry unchanged, peak-memory
  dramatically reduced.
- **PSCF4** **Native-GDF Lpq drop-in**. Drops the PySCF
  dependency on the integral side. Already shipped: C++
  `compute_2c_eri_lattice` / `compute_3c_eri_lattice`
  (`4958b4e` / `3c0c239`) + `aux_basis.build_lpq_native`
  (~5% residual on native 2c metric; design at
  [`docs/design_native_gdf.md`](design_native_gdf.md)).
  Slice 3d (compcell + Ewald-split for the long-range
  periodic 2c metric) closes the residual at v0.8.0.
- **PSCF5** *(SLIPPED TO v0.8.x, open-shell)*
  **UKS + PW1PW Γ-only**. Same scaffold caveat as PSCF1: needs
  rewrite on the new gamma-GDF foundation. PW1PW + the
  per-component Functional scaling that lands with it stays
  on the v0.8.x slate.
  Concrete in-flight pieces:
 - **C++ Functional**: per-component scaling +
    `extra_hf_fraction` infrastructure. Built; new aliases
    `pbe0` and `pw1pw` resolve correctly (PW1PW α=0.20,
    PBE0 α=0.25). **`pw1pw` is the first non-libxc-built-in
    custom hybrid in vibe-qc**; per-component scaling in
    `Functional::Impl` makes future custom hybrids
    (mPW1PW, BHandHLYP, …) a one-line alias-table addition.
 - `python/vibeqc/periodic_uks_gdf.py`, α/β GDF driver
    (V_xc from PySCF as in RKS GDF; will move to native
    once `build_xc_periodic` V_xc gauge bug closes in v0.9).
 - `run_periodic_job(method="UKS", functional="pbe"|"pbe0"
    |"pw1pw"|"b3lyp")` dispatch.

  **Gates parity against the pob-TZVP paper SI Table 2**
  (Peintinger-Vilela Oliveira-Bredow JCC **34**, 451 (2013))
 - authoritative ~60-compound oracle for periodic hybrids.
  Cite Bredow & Gerson PRB **61**, 5194 (2000) for the
  PW1PW recipe.

  **Workaround for users today**: `run_uks_periodic_gamma_ewald3d`
  is **broken on ionic crystals** (same Fock-gauge issue as
  the RHF Ewald path). Don't use; wait for the GDF UKS
  dispatch.
- **PSCF6** **`run_periodic_job(..., kmesh, method,
  multiplicity)` extended dispatch**. Single high-level entry
  point handles RHF/UHF/RKS/UKS × Γ-only/multi-k via the
  same `run_periodic_job` keyword API.

**Already on `main` (foundation work)**:

- `run_rhf_periodic_gamma_gdf` Γ-only RHF + RKS GDF driver
  (`3fd242a` + `b196337` + `ece5cb4`)
- `BasisSet` ShellInfo ctor + `aux_basis.modrho_renormalise`
  + slice-3c diagnostics (`91662ce` / `4ba3ac1` / `c933503`)
- C++ periodic 2c/3c ERI bindings + native Lpq builder
  (`4958b4e` / `3c0c239`)
- Asbestos-polymorphs scaffolding + OPTIMADE fetcher
  (`76ccda3` / `1bec644`)

**Headline parity result** (already validated on `main`):
MgO/sto-3g/Γ RHF parity vs PySCF.GDF, **ΔE = −6.8×10⁻¹² Ha
(machine precision)**. 8-system × 3-method × 2-basis sweep:
**20/23 sub-µHa parity** (3 outliers convergence-protocol
differences). UHF closed-shell parity gate passes
(UHF M=1 ≡ RHF on MgO/sto-3g/Γ to <10⁻⁹ Ha).

**Open known bugs** (carry into v0.8.x maintenance):

- Si-diamond × 3 + C-diamond / PBE / pob-TZVP converge to
  different stationary point vs PySCF (~mHa); suspected
  DIIS-tightness difference, not algorithmic.
- Multi-k dense `Lpq[k1, k2]` RAM ceiling (workaround until
  PSCF3 ships: small-system × full-kmesh OR full-system ×
  small-kmesh).
- pob-TZVP missing Ne entry (`KeyError`); workaround branch
  `fix/pob-tzvp-add-ne` exists.

**Library-only design philosophy** (de-PySCF audit
2026-05-09): vibe-qc's runtime dependencies are **libxc +
libint + spglib + libecpint only**, no other QC programs as
engines. PySCF appears in `pyproject.toml` only in `[test]`
and `[dev]` extras, never base. ASE is opt-in only via
`vibeqc[ase]`. The 3 core files that currently import PySCF
(`periodic_rhf_gdf.py`, `_basis_g94.py`, `benchmark.py`)
are the closure points for the de-PySCF migration:
removing PySCF from `periodic_rhf_gdf.py` is the v0.8.0
slice-3 modrho work (PSCF4); the other two close in v0.9+.
**This is a strong differentiator vs codes that require
ASE / PySCF / etc. as engines**, vibe-qc + libint + libxc +
spglib + libecpint = a closed dependency graph among
open-source MIT/LGPL/MPL libraries with no QC-program
runtime peer.

**v0.9.0+ carry-forward** (from periodic SCF chain):

- **Native V_ne gauge alignment** (`periodic_rhf_gdf.py:53-57`)
 - closes the gauge-difference workaround that the
  GDF-default driver applies today.
- **Fix `build_xc_periodic` O(10 Ha)/atom V_xc gauge bug**
  (`periodic_rhf_gdf.py:541`). Loud, not silent. Default
  driver routes around it via PySCF Becke grid; v0.9 ports
  V_xc to a native gauge-aligned grid. Same bug surfaces in
  the UKS path (`build_xc_periodic_uks`); UKS GDF driver
  sidesteps via the same PySCF route.
- **Drop `_basis_g94.load_g94_for_pyscf()`** once no caller
  needs it. Closes one of the 3 PySCF-importing core files.
- **Migrate `[test]` extra off PySCF long-term**. Even the
  test deps want to drop PySCF eventually; cross-code parity
  oracles shift to ORCA + CRYSTAL reference data.

**Remaining GDF accuracy residuals** (deferred research, **no
schedule pressure**, the µHa *capability* already ships; these
are redundant refinements). Tracked here so they don't float as
undated "open" items in a handover.

- **compcell+AFT µHa at the cheap default cutoff.** Γ GDF
  already reaches µHa via `gdf_method='rsgdf'` (LiH primitive
  −0.5 µHa, H₂ sub-µHa) and multi-k reaches µHa via the
  lattice-cutoff knob, so this is a *third* µHa path, not a
  capability gap. The compcell + 2c+3c AFT path lands sub-mHa
  today (`apply_aft_correction=True`, `libint` convention,
  η≈1.0); pushing it to µHa needs reconciling vibe-qc's bare
  lattice-sum integrals (libint scale) with the `j2c_p`/`j3c_p`
  AFT corrections (libcint convention) into one convention,
  plus a value-comparing test. The `libcint`-convention combo is
  a known +178 mHa foot-gun, guarded by a runtime warning
  (`9252d37a`).
- ✅ **MgO (and all-electron) µHa via Mixed Density Fitting
  (MDF).** Steep core × core pair-FT exceeds ke=200's Gmax
  bandwidth: MgO lands −515 mHa at ke=200 (−25 mHa at ke=800);
  brute-force µHa needs ke≈6400 (hours/SCF). MDF (Sun-Berkelbach
  2017 §III) shipped in v0.13.0 (see below), validated on
  light-atom/vacuum-box cells; the dense-core (e.g. ionic MgO)
  regime now fails closed instead of silently mis-fitting
  (2026-07-29) rather than returning a µHa claim outside its
  validated envelope.

### `v0.8.0`, basis-set library + ECP integration completion

> **Codename TBD** *(candidates: Stuttgart's Pangolin,
> Hay-Wadt's Wallaby)*.

🎯 **Headline feature**, completes the basis-set + ECP
infrastructure: 87 new BSE-fetched bases bundled (142 → 236
.g94 files), libecpint integration end-to-end, and Phase 14e
auto-population of `ecp_centers` from BasisSet ECP info so
vDZP / dhf-* / x2c-* SCF works without manual ECP wiring.
Sourced from the basissetdev chat's 2026-05-09 historical design record.

Sub-items:

- **BSC1** **87 new BSE-fetched bases bundled** (commit
  `a5f4906` on `basissetdev` branch; pending rebase). Includes:
  pcseg-{0..4}, aug-pcseg-{0..4}, pc-{0..4}, Pople diffuse,
  def2-mTZVP/mTZVPP, dhf-* and x2c-* relativistic, Grimme vDZP,
  full LANL family, cc-pV(n+d)Z + aug-, cc-pCV{D,T,Q}Z,
  ANO-RCC-VxZP variants, ANO-R0..3, Sadlej, pcS/pcSseg/pcJ,
  Sapporo / Cologne / SARC DKH. **Closes most of BS1-BS6** in
  the v0.8.0 basis-set library expansion entry above.
- **BSC2** **libecpint integration end-to-end** (commit
  `add8163`; banner now lists `libecpint 1.0.7`). `.g94` ECP-
  block splitter (Phase 14d); `total_ncore` valence-electron
  accounting in all 4 SCF drivers. Pt UHF/LANL2DZ converges
  (E = −118.227 Ha) as the smoke test.
- **BSC3** **Phase 14e, auto-populate `ecp_centers`** from
  BasisSet ECP info so vDZP / dhf-* / x2c-* SCF works without
  manual `[ECPCenter(Z=, xyz=)]` wiring. Needs libecpint
  `set_ecp_basis(...)` inline-primitive feed for vDZP.
- **BSC4** **CRYSTAL `200+Z` ECP parser** → 5th-period pob.
  Currently the CRYSTAL parser recognises ECP blocks but
  raises `NotImplementedError`; this closes that gap.
- **BSC5** **Pob S d-polarisation column-swap fix** (commit
  `f059b37`). Latent bug since 2013: exp/coeff column swapped
  on the S d-polarisation entry in upstream pob-TZVP / -rev2 /
  pob-DZVP-rev2; CRYSTAL's parser tolerated it; libint exposed
  it. All 3 pob bases now byte-clean to the corrected upstream.
  **Cross-code-parity success story for the JCC paper.**
- **BSC6** **Basis-set optimisation engine scaffold** (commit
  `0ac5fb9`). Parametrise / objective / scipy + iminuit drivers
  / TempBasisLibrary. Live H-atom UHF smoke validated end-to-
  end. Foundation for v0.9 multi-compound LD-aware basis
  optimisation.
- **BSC7** **13 cubic-ionic test-set inputs** (commit `6a77356`)
  covering PT2013 SI Table 4 reference systems. Self-contained;
  reproducibility-package quality.

**User-facing API** (post-rebase): `RHFOptions.ecp_centers`,
`.ecp_library` on all 4 SCF; `vq.ECPCenter` class;
`vq.compute_ecp_matrix(...)`; `vq.libecpint_version()`; 87
new basis-set names resolvable via `vq.BasisSet(mol, name)`.

**Open known issues** (carry into v0.8.x): heavy-atom load
test cumulative-OOMs the laptop on full 366-case batch
(workaround: route via `vq submit` to compute-reference); 3c composite
methods (r²SCAN-3c, B97-3c, ωB97X-3c) basis carriers ship
but gCP + D3/D4 aren't wired through the molecular runner
(don't promise turnkey 3c, workaround: `def2-TZVP` + matching
ECP + D3(BJ) until v0.9.0 composite-methods milestone).

**Branch state**: `basissetdev` is rebased onto current `main`
and 6 commits ahead. **NOT merged.** User reserved the rebase
for "after the papers ship", coordinate timing with the
release chat.

### `v0.8.0`, vq queue v0.4: moved out to vibe-queue

The queue's plan is not vibe-qc's to keep. `vq` is the
[vibe-queue repository](https://github.com/vibe-qc/vibe-queue) since the
2026-09-08 split, with its own tracker, releases and roadmap, and a plan
maintained here could only go stale or contradict the one that is actually
followed.

What used to sit here was the v0.4 milestone (cgroups v2 enforcement, a
read-only web UI, a per-job event log), an inventory of what vq v0.1 to v0.3
had shipped, and a v0.5 to v1.0 forward plan. It is preserved in this file's
history; `git log -- docs/roadmap.md` reaches it.

The design spec lives in that repository:
[`docs/SPEC.md`](https://github.com/vibe-qc/vibe-queue/blob/main/docs/SPEC.md).

### `v0.8.0`, `vqfetch` external-data integration ✅ **shipped**

> **Codename** *Grimme's Gecko*.

🎯 **Shipped at v0.8.0** (the 2026-05-14 scope-widened gate:
Phase 1 structures + Phase 2 CCCBDB references + geometry bridge +
multi-candidate API). 80 tests, 12 commits merged to `main`, live
compute-reference MgO round-trip verified at E = −950.4204308512 Ha.

**What landed:**

- **VFETCH1** `vqfetch` console script + `vibeqc.fetch` package
  (`optimade | cod | mp | canonical` subcommands; `[fetch]` extra).
- **VFETCH2** OPTIMADE federation primary + MP/COD/NOMAD secondaries.
  5 canonical structures round-trip-verified (MgO/NaCl/LiH rocksalts;
  Si/C diamonds).
- **VFETCH4** Schema: `Provenance`, `ExperimentalReference`,
  `ReferenceKind`, `TestSystem` union on `PeriodicSpec` / `MoleculeSpec`.
- **Phase 2** NIST CCCBDB experimental-reference fetcher
  (`vqfetch reference`, `fetch_cccbdb()`) + `experimental_geometry_
to_molecule_spec` bridge + `--include-experimental-reference` in the
  regression suite.
- **Multi-candidate** `fetch_optimade(max_results=N, dedup=True)`
  Python API with structural-hash dedup + space-group plurality ranking.

**Docs:** `docs/user_guide/external_structures.md`,
`docs/user_guide/reference_data.md`,
`docs/tutorial/external_data_fetcher.md`,
`examples/input-vqfetch-optimade-mgo.py`,
`examples/input-vqfetch-reference-h2o.py`.

### `v0.8.x`, `vqfetch` polish (deferred → v0.13.x)

> The 9 items below were deferred from the v0.8.0 ship gate (Mike,
> 2026-05-14). None blocked the tag. They moved to the v0.13.x
> maintenance window below and are now checked off there on ``main``.

### `v0.8.0`, asbestos polymorphs study scaffolding (consumer of dev features)

> **Note (2026-05-09)**: The asbestos polymorphs work is a
> separate paper / publication, Paper 1 (bulk hybrid DFT)
> and Paper 2 (surfaces + Fenton), in a venue that is **not**
> the vibe-qc v0.8.0 JCC release paper. The release paper
> may include a one-sentence forward-pointer noting the
> asbestos paper(s) as an independent application of
> vibe-qc, but the study itself ships under its own
> editorial timeline.

🎯 **What ships at v0.8.0 from this entry**: the
`studies/asbestos-polymorphs/` scaffolding (already on
`main` since v0.7.0; see commit list below) is in tree and
runnable on compute-reference. The actual Paper 1 calculations require
v0.8.0 features other dev chats deliver (periodic stress +
slab + the asks below); the paper writeup is an independent
publication.

This is a **study chat, not a feature chat**. It consumes
periodic-SCF features delivered by other dev chats and
produces application-grade calculations + the bulk-DFT
reference paper (Paper 1) + future surface/Fenton paper
(Paper 2). Methodological extension of the Torino-CRYSTAL
B3LYP line (Demichelis 2016 hydrous silicates; Prencipe 2009
vibrational / IR spectroscopy of hydrous Mg silicates) to an
open-source code that any chemist can install and run.

Already on `main` since v0.7.0:

- `studies/asbestos-polymorphs/` scaffolding (~`76ccda3` →
  `9044593`):
 - `_minerals.py` CIF → `PeriodicSystem` library
 - `cifs/` directory with `CIFS_NEEDED.md`, OPTIMADE fetcher,
    tremolite CIF
 - Per-mineral inputs for **5 closed-shell minerals × 2
    functionals** (PBE0 + PW1PW)
 - 2 Fe-mineral README markers (Fe-bearing variants gated
    on UKS + broken-symmetry support)
- README rewritten 2026-05-09 → **two-paper plan**, **B3LYP
  the primary functional** for paper 1, antigorite-m17 (~290
  atoms) deferred from Paper 1.

**Paper 1 systems** (closed-shell, in scope):

| Mineral | Functional | Basis | Properties |
|---|---|---|---|
| lizardite-1T | B3LYP / PBE0 / PW1PW | pob-DZVP-rev2 | lattice params, ΔAH⁰, band gap |
| tremolite | B3LYP / PBE0 / PW1PW | pob-DZVP-rev2 | same |
| chrysotile-clino-layer | B3LYP / PBE0 / PW1PW | pob-DZVP-rev2 | same |
| anthophyllite-Mg | B3LYP / PBE0 / PW1PW | pob-DZVP-rev2 | same |
| (5th system TBD from CIFS_NEEDED.md) | - | - | - |
| antigorite-m17 (~290 atoms) | DEFERRED | - | Paper 2 / v0.9+ |

**Paper 2 systems** (Fe-bearing, broken-symmetry; v0.9+):
crocidolite, amosite, etc., surfaces + Fenton chemistry
(H₂O₂ / OH• / Fe-tet) for asbestos carcinogenicity (IARC
100C); blocked on UKS GDF + periodic counterpoise BSSE +
slab + adsorbate workflow.

### v0.8.0 asks from the asbestos chat (asks, NOT deliverables)

The asbestos study chat raises three v0.8.0 feature requests
to other dev chats. Two are already roadmapped; one is new:

1. **NEW ASK**, **Periodic Γ Hessian on the GDF chain**, for
   OH-stretch fingerprints. Prencipe 2009 reproduction target.
   Not currently on any v0.8.0 entry. Periodic-QC dev chat
   should scope this; if it lands in v0.8.0, the asbestos
   paper gets a vibrational-spectroscopy validation panel.
2. **Periodic stress tensor / elastic constants**, already
   v0.8.0 (G2 series; "periodic stress tensor + slab builder"
   entry). Chip-tracking unconfirmed by the asbestos chat;
   asbestos consumes this for Paper 1 lattice-parameter
   validation.
3. **Slab builder + dipole correction**, already v0.8.0
   scope; Paper 2 dependency.

### Operational notes

- Paper 1 calculations require `vq submit` to compute-reference;
  asbestos-scale SCF (50-290-atom unit cells) OOM-crashed the
  laptop on 2026-05-08. All heavy calcs route through
  `vq submit` per the `reference_vq_queue` user-memory file.
- vq pipeline validated for asbestos workflow (smoke job
  `280a7aeb56f2`, 2026-05-09).
- Currently awaiting user-supplied AMCSD CIFs (lizardite-1T
  first), then closed-shell submit cascade.

### Citations for Paper 1

- **IARC Monograph 100C**, asbestos carcinogenicity (motivates
  the application; introduction)
- **Gualtieri (ed.), EMU Notes vol. 18 (2017), ch. 2**,
  asbestos mineralogy reference
- **Demichelis et al., *CrystEngComm* (2016)**, hydrous-
  silicate CRYSTAL B3LYP precedent
- **Prencipe et al., *Phys. Chem. Min.* (2009)**, IR
  spectroscopy of hydrous Mg silicates with B3LYP/CRYSTAL

### `v0.9.0`, shipped 2026-05-22, Knowles's Kingfisher

> **Knowles's Kingfisher.** Planned as the "composite 3c methods" release
> below; actually shipped wavefunction-methods scaffolding (MP2->MP3->MP4->
> CCD->CCSD) + full semiempirical stack (GFN2-xTB, PM6, OM1/OM2/OM3) +
> native-D4 backend (experimental) + basissetdev integration.
> See CHANGELOG. Composite 3c methods (below) shipped later, before the
> repository split; see their status note.

### Composite "3c" methods

> **Status, 2026-09-13: shipped before the repository split, molecular only.**
> `run_job(method=...)` accepts the composite recipe names that
> `vibeqc.list_composites()` lists, including `r2scan-3c`, `pbeh-3c` and
> `hse-3c`; see [composite methods](user_guide/composites.md). HSE-3c has no
> periodic route yet, and the periodic flavours need recalibration.

🎯 **Headline feature**, keyword-shortcut support for the
modern Grimme-school **composite "3c" methods**: a single name
(`r2scan-3c`, `ωb97x-3c`, `b97-3c`, `pbeh-3c`, `hf-3c`,
`b3lyp-3c`, `hse-3c`) bundles a tuned basis + dispersion + gCP
counterpoise + (sometimes) a modified short-range correction
into one user-facing method. These are the highest-leverage
addition for routine users today: GMTKN55 MAEs comparable to
much-more-expensive plain-functional + triple-zeta calculations,
at small-basis cost.

**Sits on top of v0.8.0**: BS7 (composite-method bases:
def2-mSVP, def2-mTZVP, def2-mTZVPP, vDZP) + F1 (range-separated
hybrids for ωB97X-3c) + F3 (r²SCAN family for r²SCAN-3c) + F5
(D4 dispersion). Once those land, the remaining work is the
*wiring*, registering the keyword shortcuts and the
counterpoise machinery, which is a small layer on top.

- **C1** **Keyword-shortcut machinery**. `vq.run_job(method="r2scan-3c", ...)`
  resolves to the right (functional, basis, dispersion, gCP)
  tuple per the Grimme group's published recipes. Recipe table
  registered as a static dict; failure if a recipe references a
  basis or functional that's not yet shipped (forces ordering
  with v0.8.0).
- **C2** **gCP, geometric counterpoise correction** (Kruse &
  Grimme, *J. Chem. Phys.* **136**, 154101, 2012). Pure-Python
  addon over the SCF; per-atom-pair correction; needs only the
  geometry + element list + basis-set name. Standalone API
  `vq.compute_gcp(mol, basis_name)` plus auto-wiring inside the
  3c keyword shortcuts.
- **C3** **Modified short-range corrections per composite**.
  PBEh-3c uses a *modified* PBE0 with re-tuned HF-exchange
  fraction + range-separation parameters; same shape for
  HSE-3c, B3LYP-3c. Per-recipe in the C1 table.
- **C4** **Per-recipe parameter table**. One static dict mapping
  `composite_name → (functional, basis, dispersion, gCP, sr_mod)`.
  Sourced from the original publication tables; pinned to specific
  paper versions. Editable in one place.
- **C5** **Per-composite documentation pages** under
  `user_guide/composites/` (or one consolidated chapter): when to
  use each composite (system class), MAE on benchmarks (GMTKN55,
  TorsionNet206, S22, X23), citation, runtime cost relative to
  the parent functional + basis without the corrections.

Per-composite slot (recommended use, adapted from the
[basis-set survey § Composite "3c" methods](user_guide/basis_sets.md#composite-3c-methods)):

| Composite | Best for | Cost vs. plain TZ-hybrid |
|---|---|---|
| **r²SCAN-3c** | "Swiss army knife", geometries, thermochemistry, noncovalent, transition-metal complexes | ~10-20× cheaper |
| **ωB97X-3c** | Drug-like / pharma; barrier heights; vDZP basis with ECPs | ~5-10× cheaper |
| **B97-3c** | Workhorse GGA; especially good for transition metals | ~30× cheaper |
| **PBEh-3c** | Strong noncovalent geometries | ~20× cheaper |
| **HF-3c** | Mean-field qualitative; very large systems | ~100× cheaper |
| **B3LYP-3c** | IR spectra (B3LYP frequencies are particularly accurate) | ~20× cheaper |
| **HSE-3c** | Solid-state range-separated variant (gates on the periodic GDF SCF) | ~20× cheaper |

User guide chapter, to be authored alongside the implementation.
Cross-link from [tutorial set](tutorial/index.md) once a
"recommended first calculation" recipe per composite is settled.

### `v0.9.0`, implicit solvation (deferred; CPCM shipped in v0.12.0)

> **Codename:** Knowles's Kingfisher (release) / this feature deferred.

🎯 **Headline feature** (originally planned), CPCM / COSMO molecular implicit solvation.
Aqueous chemistry is most users' default reaction medium.

- **S1a** Continuum cavity, solvent-excluded surface (SES) /
  solvent-accessible surface (SAS) tessellation, Bondi radii, GEPOL
  defaults.
- **S1b** CPCM Fock / energy contribution, apparent surface charge
  iterated to self-consistency inside the SCF loop.
- **S1c** Solvation gradient, analytic ∂E_solv/∂R for geometry
  optimization in solvent.
- **S1d** Solvent presets (water, DMSO, MeCN, DCM, ...) wired
  through `run_job` + ASE calculator.

### `v0.10.0`, shipped 2026-05-27, Pisani's Penguin

> **Pisani's Penguin.** Planned as the Grimme D4 dispersion release below;
> actually shipped periodic-SCF maturity: multi-k EDIIS, SYM6 crystal
> symmetry, BIPOLE L_max>=2, GPW M2-full iterative SCF, CRYSTAL gauge
> unification, full DFTB stack + DFT+U, NEB, PWPB95, periodic D3-BJ.
> See CHANGELOG. Native-D4 backend shipped experimental in v0.9.0;
> D4 dispersion (below) shipped as dftd4 backend in v0.8.0.

### Grimme D4 dispersion (shipped: dftd4 backend v0.8.0; native-D4 validated for H-Ne v0.15.0)

🎯 **Headline feature**, Grimme's D4 dispersion correction with
charge-dependent C6 coefficients via partial-Hirshfeld atomic
populations. Strict refinement over the D3 already shipped in v0.2.x.

- **D5a** Charge-dependent c6ab table + parameter set (Caldeweyher
  2019).
- **D5b** Hirshfeld atomic charges from the SCF density.
- **D5c** D4 energy + gradient + stress.
- **D5d** ``dftd4`` reference Python backend as optional dep
  (mirrors the D1b pattern).

### `v0.11.0`, shipped 2026-06-03, Sun's Stingray

> **Sun's Stingray.** Planned as the geometry-symmetrize release below;
> actually shipped periodic-GDF chain closure: compcell + multi-k + AFT
> (RSGDF from Sun-Berkelbach 2017). See CHANGELOG. Geometry-symmetrize
> features (GS1-GS3 below) shipped in v0.13.0.

### Geometry symmetrize + full example coverage (shipped: GS1-GS3 v0.13.0)

🎯 **Headline feature**, `vq.symmetrise(system, symprec)` snaps
imperfect input cells to the nearest perfect symmetry, refining the
lattice. Critical for making large calculations feasible: the SYM2b
orbit-compression machinery requires symmorphic + origin-fixed
atoms, which a typical user input rarely has out of the box.

- **GS1** ``vq.symmetrise(system, symprec=1e-4)``, calls spglib
  ``standardize_cell``; returns a new ``PeriodicSystem`` plus a
  metadata report (RMS displacement, spacegroup before / after).
- **GS2** File readers (``read_poscar`` / ``read_cif`` /
  ``from_xyz`` for periodic XYZ) gain a ``symmetrise=True``
  keyword that runs GS1 on the parsed system. Plus
  ``vq.detect_spacegroup(system, symprec)`` non-modifying
  introspection helper.
- **GS3** Wire SYM2b orbit-compression to take the standardised
  cell so it actually triggers in practice.
- **DOC1-full** *example coverage across the entire documentation*,
  the original DOC1 scope. Links every numbered tutorial to runnable
  inputs while leaving ``.out`` / ``.molden`` / ``.cube`` / ``.traj``
  artifacts as local regeneration products,
  with a 30-line excerpt inline + download link in each tutorial.

### `v0.12.0`, shipped 2026-06-12, algorithmic-density release

> **Knuth's Beaver.**

🎯 **Headline feature**, a dense algorithmic release rather than a
single queue milestone. Shipped in full: DLPNO-CCSD infrastructure
(M1 pair classification, PAO domains, PNOs, M2 DLPNO-MP2, M3a DLPNO-CCSD
pilot); TDDFT (Casida + TDA, RHF/RKS/UHF/UKS); CASCI / CASSCF /
internally-contracted CASPT2 / NEVPT2; CIS/TDA excited-state
gradients (FD, state-tracked); GPW production routing; BIPOLE
analytic-gradient hardening; MSINDO periodic CCM and 4th-row coverage;
MACE MLIP integration; an experimental MPI substrate; MD/metadynamics; and the
Green's-function propagator. See ``CHANGELOG.md`` for the full patch-by-
patch inventory.

Subsequent work in v0.14.0-v0.15.0 resolved the CASSCF analytic-gradient
P0 bug (now FD-tight to ~1e-7, default path) and completed the DLPNO stack
(DLPNO-CCSD(T), U-DLPNO-MP2, U-DLPNO-CCSD(T) pilot). See the v0.15.0
section for the comprehensive shipped catalogue.

**Audit note (2026-06-16).** The earlier ``vibeqc.jobs`` /
``vqc-jobs`` sketch was superseded before the tag by the ``vq`` CLI.
Since the 2026-09-08 split, it belongs to the independent vibe-queue repository. Do not add a second
queue under ``python/vibeqc/`` to "complete" the old roadmap text:
the maintained queue surface is ``vq`` plus
the core [integration guide](user_guide/queue.md) and the independent
[vibe-queue documentation](https://vibe-qc.com/vibe-queue/docs/).

### `v0.13.0`, shipped 2026-06-17, Generalised Regular k-grids

> **Wisesa's Fox.**

🎯 **Headline feature**, Generalised Regular (GR) grids
(Wisesa, McGill & Mueller 2016, *Comp. Mat. Sci.* 110, 1;
Balasubramanian, Dwaraknath & Mueller 2021). Searches over
reciprocal sublattices for the grid with the largest minimum
periodic distance per k-point, typically ~2x more efficient than
axis-aligned MP for non-cubic / anisotropic cells. All K6a-c shipped.

Additional v0.13.0-v0.13.1 shipped content: ROHF/ROKS molecular + periodic;
megacell periodic MP2 + DLPNO-CCSD(T); MSINDO to Xe (Z 1-54); BIPOLE-CRYSTAL
parity; Gilat-Raubenheimer metallic DOS; selected-CI CASSCF backend;
CASPT2 engine="cases"; BDIIS basis-set optimiser; AUTO k-point
recommender (K8); corrected-gauge BIPOLE gradients; READ/ATOMSPIN/SPINLOCK
initial-guess + convergence features; Mixed Density Fitting (MDF); and the
complete `vqfetch` polish pass (VFETCH-X1 through X9). See CHANGELOG.
See CHANGELOG.

### `v0.14.0`, shipped 2026-06-22, Bartlett's Goose

> **Bartlett's Goose.** Comprehensive wavefunction-methods release.

🎯 Named flagship (canonical CCSD(T) for RHF/UHF/ROHF) plus a dense
algorithmic haul that shipped during the cycle:

- ✅ **Canonical CCSD(T)**, closed-shell + UHF + ROHF references, validated
  against PySCF to ~2e-11 Ha. Higher-order triples variants + broader CI/CC
  ladder still raise `NotImplementedError`.
- ✅ **DLPNO-CCSD(T)** closed-shell, reducing-scaling local solver + local (T),
  bit-for-bit canonical reproduction at full domains.
- ✅ **DLPNO-UMP2 + DLPNO-UCCSD(T) pilot** (UHF reference, canonical-virtual PNO
  rung; O(N⁶), `max_nbf=64` capped). Reduced-scaling open-shell engine remains
  open.
- ✅ **Frozen Natural Orbitals (FNO)** for closed-shell CCSD(T).
- ✅ **Davidson iterative diagonalization** (D3), LOBPCG (D5), Jacobi-Davidson
  (D6, experimental), and the **unified solver framework** (7 registered solvers,
  `solver=` keyword on `run_job`).
- ✅ **VV10 nonlocal correlation** (un-gates ωB97X-V); ωB97M-V meta-GGA hybrid.
- ✅ **r²SCAN01 meta-GGA**, molecular + periodic.
- ✅ **Periodic SAP + MINAO initial guesses**; **FRAGMO** fragment-superposition
  guess (molecular).
- ✅ **Periodic density mixers** (Anderson + Broyden) in multi-k RKS driver.
- ✅ **Methfessel-Paxton & Marzari-Vanderbilt smearing** (M6/M7).
- ✅ **Periodic ECP integrals** from CRYSTAL-format inline data.
- ✅ **Basis-optimisation gradient** (RHF/UHF/RKS/UKS analytic gradient for
  contraction coefficients; BDIIS basis-set optimiser).
- ✅ **GPAW plane-wave-limit references**; GAPW partition-of-unity fix.
- ✅ **2D (slab) Ewald**: energy primitive, plus the full vacuum-free slab SCF
  (`CoulombMethod.SLAB_EWALD_2D`; RHF/RKS/UKS, Gamma + multi-k), wired as the
  `jk_method="auto"` route for `dim=2`. The `dim=2` API takes two in-plane
  lattice vectors (`vibeqc.slab_2d`); the third column is synthesized and the
  total is invariant to it. Analytic gradients and smearing on a slab are open.
- ✅ **vibe-view** QVF v1.0 complete (Fermi surface, NCI, potential, EOS,
  phonon band+DOS, density-difference, RMSD/Kabsch compare).
- ✅ **Semiempirical:** MSINDO CID, ROMP2, NDDO-MP2, CCM periodic to Xe;
  GFN2-xTB PES collapse fix.
- ✅ **Unified CASSCF/CASPT2/NEVPT2 gradient surface** across optimizers.

Molecular-first discipline (CLAUDE.md §7/§10) enforced throughout: every method
validated out-of-process, un-gated on `main`, before periodic extension begins.
Density-fitting prerequisites (20a-20e) shipped earlier (v0.7.2-v0.8.0).

### `v0.15.0`, shipped 2026-06-26, comprehensive feature freeze

> **Neese's Cheetah.** Feature-freeze release; maintainer scope decision was
> "add everything we have ready, even if not roadmap-due; rework the roadmap
> rather than gate scope by it." Frozen at e94221f2.

🎯 Contains every feature that landed on `main` through the freeze and was not
already tagged in a prior release. Major items:

**Coupled-cluster + local correlation:**
- ✅ **DLPNO-(T1)** with off-diagonal localised-Fock coupling and
  full-domain parity to canonical (T).
- ✅ **DLPNO-MP2 local density fitting**; sparse pair lists default-on.
- ✅ **DLPNO-CCSD per-pair solver** + integral transformation; `tcut_pairs`
  default-on.
- ✅ **Open-shell DLPNO-UMP2** (UHF reference, local DF) and
  **DLPNO-UCCSD(T) pilot** (spin-orbital, canonical-virtual PNO).
- ✅ **A-namespace neutral-control DLPNO** (U-DLPNO-MP2 +
  U-DLPNO-CCSD(T) on the neutral fitted-torus operator). These controls have
  no Γ-CCM or χ-CCM construction identity.

**Multireference:**
- ✅ **CASSCF analytic nuclear gradient** -- P0 resolved, FD-tight to ~1e-7
  (default path). State-averaged + open-shell gradients fall back to FD.
- ✅ **CASPT2/NEVPT2 gradients** experimental (full-energy FD).
- ✅ **Geometry-optimizer framework** with delocalized-internal-coordinate
  citations (RFO, P-RFO, GDIIS, FIRE, trust-region).
- ✅ **Selected-CI kernel** past the 2M-determinant wall; heat-bath presorted
  candidate walks.
- ✅ **Matrix-free RDM CASPT2** (`caspt2(engine="cases")`).

**Periodic SCF + basis:**
- ✅ **Periodic-XC cross-cell fix** (dense-crystal DFT XC error resolved).
- ✅ **Periodic COSX** (M3b-1 through M3b-6: real-space K(g) kernel, SR+LR
  composed exchange, RIJCOSX Γ-only, multi-k bridge).
- ✅ **Multi-k GDF J/K fix** (tight-cell over-binding resolved).
- ✅ **Mixed Density Fitting (MDF)** for periodic GDF.
- ✅ **Periodic basis-optimisation gradient** (Phases P0-P3).
- ✅ **GAPW experimental warning** correctly gates the runner path.

**Cyclic Cluster Model (experimental, gated):**
- ✅ **Γ-CCM / `aiccm2026dev-a`** -- union-and-weight/Wigner--Seitz
  integral-weighting CCM. HF, KS-DFT,
  MP2/UMP2, and CCSD(T). Integral-direct
  scalable J/K (Phase 3b). Properties, localization, symmetry analysis.
- ✅ **A-namespace neutral fitted-torus controls** -- canonical
  MP2/UMP2/CCSD(T)/UCCSD(T) and restricted/unrestricted DLPNO methods on their
  matching neutral SCF reference. Their historical module location does not
  make them Γ-CCM or χ-CCM results.
- ✅ **χ-CCM / `aiccm2026dev-b`** -- finite-translation-group character CCM.
  RHF/RKS/UHF/UKS via four-center, RI, and RIJCOSX in 3D. RI-MP2,
  DLPNO-MP2, DLPNO-CCSD, and DLPNO-CCSD(T) in 3D. Every 1D/2D absolute-energy
  backend fails closed pending a shared wire/slab Coulomb convention. The
  restricted four-center RHF route has an exact MPI slice over complete
  direct-ERI output blocks; other χ SCF and post-HF MPI work remains open.
- ✅ **28-system benchmark test set** across all 7 crystal systems, CRYSTAL23
  comparison harness.

**Properties + analysis:**
- ✅ **COOP/COHP** (C++ accelerated, QVF `dos.coop`/`dos.cohp`, plotters, CLI).
- ✅ **Periodic Mayer bond orders** (k-space generalisation, QVF `bond_orders`).
- ✅ **QTAIM** (critical-point search + bond-path tracing, analytic C++ Hessian,
  QVF `topology.qtaim`).
- ✅ **Fat bands** (Mulliken-projected band weights, QVF `bands` projections).

**IO + infrastructure:**
- ✅ **QVF v1.2** -- four new section kinds (localized orbitals, bond orders,
  fat bands, QTAIM), NTO writer dispatch, runner auto-population.
- ✅ **TD-DFT NTO writer** path ready (the compute side is planned in the multireference and
  excited-state milestone).
- ✅ **vibe-view:** QVF v1.2 consumption, GROMACS .gro + Gaussian .gjf/.com
  import.
- ✅ **basis_toolkit** (`python/vibeqc/basis_toolkit/`) -- importers (BSE, CRYSTAL,
  G94, NWChem, ORCA), exporters (G94, NWChem, ORCA), validator, QVF basis
  integration, `vibeqc basis` CLI.

**Experimental / gated features** (this block is also the standing
ungated-validation inventory: the tracker umbrella #96 was retired here on
2026-09-06; none of these is a reproducible defect, and a limb that becomes
one gets its own `kind::gate` issue):
- 🟡 **Jacobi-Davidson eigensolver** (full MINRES recurrence + harmonic Ritz
  pending). Davidson is the production solver; JD is not claimed validated
  at 0.15.
- 🟡 **GPLHR** (generalized preconditioned locally harmonic residual) --
  framework-registered, not C++-wired into SCF drivers; not claimed
  validated at 0.15.
- 🟡 **DMRG** -- the two-site DMRG/MPS solver behind `method="dmrg"`
  (`docs/user_guide/non_hf_solvers.md`) is experimental and not claimed
  validated at 0.15.
- 🟡 **Selected semiempirical periodic extensions** -- experimental and not
  claimed validated at 0.15; the SECCM lines are tracked per issue.
- 🟡 **2D slab Ewald** -- the default 2D route, excluded from validation
  claims; tracked under #85.
- 🟢 **Native-D4 dispersion** -- un-gated + parity-validated for H-Ne
  (v0.15.0: per-atom Eq.-6 extraction + correlated CPKS/PBE38 reference
  C6 + r4r2 fix; ~5 % C6 / <0.05 kcal/mol vs dftd4). `dftd4` stays the
  default for full periodic-table coverage; extending the reference
  catalogue past period 2 is the prerequisite for making native default.
- 🟡 **GFN2-xTB** -- gated behind `GFN2ExperimentalWarning`; structural
  correctness fixed, still experimental (gating: #43, #68 to #71, #76).
- 🟡 **GAPW** -- gated behind `GAPWExperimentalWarning`; partition-of-unity
  fix landed, still experimental (gating: #65).
- 🟡 **BIPOLE meta-GGA gradients** -- the cell-level multipole far-field
  (G1) is retired, and the quartet-level replacement fails closed at every
  SCF driver.
- 🟢 **IC-CASPT2 analytic gradient** -- the unshifted single-state explicit
  engine has a coupled orbital/CI Lagrangian, full-energy-FD-pinned on H2 and
  core-containing LiH. Shifted, direct-large, selected-reference, and
  open-shell variants remain on full-energy FD. NEVPT2 uses relaxed
  full-energy FD.
- 🟡 **AICCM / CCM** -- both `-a` and `-b` lines gated behind experimental
  warnings.
- 🟢 **Microsoft SKALA-1.1 neural XC** -- hash-pinned, on-demand official
  checkpoint with molecular RKS/UKS/ROKS fixed-geometry single points and the
  pinned PySCF-2.14 level-3 parity profile accepted by vibe-qc. Analytic
  derivatives and second-order response fail closed. Periodic AO-grid routes
  are available only as an explicitly experimental adaptation; periodic
  dispersion is rejected until corrected energy and citation provenance can
  be preserved.

**CCM long-term plan.** The two independent AICCM lines remain experimental
and do not auto-enable. Production-grade items (scalable dense-3-D neutral
four-center, mixed-boundary Green's functions for 1-D/2-D, production
reduced-scaling correlation at production scale, scalable 3-D analytic CCM
gradient, bulk polarization, projection-based embedding) are individually
sequenced rather than gated as a monolithic v2.x track. While the release-paper
line is open, these stay behind the 0.15.x hardening gate; the next post-paper
joint deliverable is a shared wire/slab Green's function for low-dimensional
systems.


## Pre-split process notes and superseded proposals

Written when the work was split across chats and branches rather than across
repositories. Kept for provenance. None of it describes how vibe-qc is
developed today; `AGENTS.md` is the current answer.

### Automated regression / parity test suite (separate chat)

The 8-system × 2-basis acid test (NaCl / MgO / Al₂O₃ / LiH ×
{sto-3g, pob-dzvp-rev2}) plus the dual-target runner (release vs dev)
were developed in the regression suite. Wave 1 covers the linear-dep
program above; waves 2/3 generalise to all SCF methods and become
the project's continuous-correctness dashboard.

**Status as of v0.7.3 (2026-05-08):**

* Waves 1.0 through 1.7 shipped across v0.7.0 → v0.7.3. Suite covers
  ~30 molecules + 4 periodic systems with three-way vibe-qc / PySCF /
  ORCA cross-code parity at single-point SCF.
* Per-case subprocess isolation (v0.7.2, "Boys' Crucible") catches
  C-level crashes in any one runner as `error` rows without bricking
  the dispatcher, the design payoff that lets the suite ride out
  ongoing PySCF.pbc / vibe-qc DF SIGSEGVs.
* Wave-1.5+1.6 added 5 noncovalent dimers from S22 (water / NH3 /
  formamide / methane / benzene-T) + 3 X23 molecular crystals (urea,
  ice Ih, benzene). See `examples/regression/REFERENCES.md` for the
  full bibliography map.

#### Test-set expansion (queued)

Per the basis-set / test-set survey (2026-05, full document hosted
in the docs chat; bibliography distillation in
`examples/regression/REFERENCES.md` §B), the following benchmark sets
are *queued for incorporation into the regression suite* once
prerequisite vibe-qc features land. Each line names the set, its
prerequisite, and the version slot.

* **GMTKN55** (1505 reaction energies / 2462 single-points; Goerigk
  2017), needs **multi-component reaction-energy machinery** in the
  dispatcher (a "case" becomes `E_products − E_reactants`, not a
  single-point). v0.8 territory.
* **ACCDB / ASCDB / GSCDB137** aggregators, same prerequisite as
  GMTKN55. ASCDB's 200-point statistical reduction is the best CI
  candidate once the machinery exists.
* **diet-150-GMTKN55** (Gould 2018), 150-system genetic-algorithm
  reduction; the recommended basis-set-convergence test surface per
  Pitman 2024. Drop-in once GMTKN55 ingest works.
* **W4-17 atomization energies** (Karton 2017), needs **isolated-
  atom calculations** as part of every reaction (E_atomization =
  E_molecule − ΣE_atom). Special case of multi-component machinery.
* **S22 full / S22A / S66 / S66x8 / A24** noncovalent, geometries
  available now from GMTKN55 / refdata; *interaction-energy*
  comparison requires multi-component (E_int = E_AB − E_A − E_B,
  with optional counterpoise correction).
* **MOR41 / ROST61 / MOBH35 / WCCR10 / TMC151 / MME55 / SSE17**
  transition-metal benchmarks, need **(1)** def2 family + def2-ECP
  in vibe-qc's basis library (BSE adapter), **(2)** TM-tested
  functional support (ωB97M-V, B3LYP*-D3BJ, PWPB95, TPSSh) where
  not already present, **(3)** robust UHF / UKS for first-row TM
  open shells. Spread across v0.8, v1.0.
* **TorsionNet206** (Behara 2024) drug-like torsion curves, needs
  **automated torsion driver** (rigid scan over a defined dihedral)
  + def2-TZVP + ωB97M-V or vDZP (composite). v1.0 territory.
* **ROT34 / LB12 / HMGB11 / MGBL150** geometry benchmarks, need
  **molecular geometry-optimiser harness** in the regression suite
  (today only single-points are tested). v0.8 if the optimiser
  surfaces gradients reliably; otherwise later.
* **CCCBDB** (NIST experimental thermochemistry, 2186 molecules),
  needs **scraping or third-party mirror** + same multi-component
  machinery as W4-17.
* **X23 / ICE13 / DMC-ICE13 / POLY59** molecular crystals, geometry
  data partially shipping today (urea / ice / benzene); **lattice-
  energy comparison** needs both **(1)** dispersion correction
  (DFT-D3 / D4 / VV10, Phase D1/D2 in v0.5.0) and **(2)** the
  multi-component machinery (E_lattice = E_crystal/Z − E_isolated).
* **OMC25** (27M molecular crystals; Gharakhanyan 2026), bulk
  ingestion problem; not a regression target, more a stress-test
  surface for high-throughput periodic SCF once the rest works.

**Prerequisites in priority order** (gating most of the above):

1. **Multi-component reaction-energy machinery** in `run_suite.py`
 - converts a "case" from a single SCF + cross-code Δ into a
   composite `Σ stoichiometric_coeff × E_component` with cross-code
   Δ on the composite. Probably v0.8.0 milestone.
2. **BSE-driven basis-set ingestion** so def2 / cc-pVnZ / pcseg-n
   / aug-cc-pVTZ are first-class without hand-coded `.g94` files.
   Required for any of the §3 / §4 benchmark recommendations.
   v0.8, v0.9.
3. **Dispersion corrections** (DFT-D3, DFT-D4, VV10), Phase D1/D2,
   slated v0.5.x in the parity-audit list above. Required for
   meaningful X23 / S22 / S66 lattice / interaction energies.
4. **Functional additions**: PW1PW (Bredow's 1-parameter hybrid;
   prerequisite for the Peintinger 2013 / Vilela Oliveira 2019
   reference tables, see `examples/regression/REFERENCES.md` §A).
   ωB97X / ωB97M-V family for modern thermochemistry. r²SCAN family
   for the 3c composites. Each is its own libxc-name + grid /
   range-separation wiring.
5. **Composite-method support** (HF-3c, B97-3c, r²SCAN-3c, ωB97X-3c)
 - bundles small basis + dispersion + gCP. Highest-leverage
   addition for routine users; ties together (2)-(4). v0.9, v1.0.

The §B survey list is the full discovery surface; this roadmap entry
records *which* of those we plan to wire and *what blocks each*.


### Shipped in v0.17.1: on-demand BSE basis fetcher

This pre-split proposal refers to the
[archived `basissetdev`](https://vibe-qc.com/docs/)
branch, which was not transferred to the new repository. It carries 87 basis sets fetched directly from
the [Basis Set Exchange](https://www.basissetexchange.org)
(pcseg-N, dhf-*, x2c-*, vDZP, full LANL family, ANO-RCC-V*,
ANO-R0..3, Sadlej, pcS, Sapporo, Cologne SARC DKH, etc.).
**These are NOT shipped in v0.8.0** per the user's standing
rule that basissetdev stays a paper-writing branch and does
not merge into `main`.

Any future implementation lands in the new core history. The architectural
proposal is to **not bundle** the 87 BSE sets directly, and
instead add a `vibeqc.basis.fetch_from_bse(name)` helper
modeled on `vqfetch`:

* Bases pulled from BSE on first use, cached locally
  (`~/.cache/vibeqc/basis/`) per XDG.
* Per-basis-set citation surfaced in the SCF log + recorded in
  the per-run `.system` manifest.
* Wheel size unaffected (~half the size compared to bundling
  236 .g94 files).
* Avoids any per-basis-set redistribution clearance question
  for sets whose original publishers might prefer not to be
  bundled in third-party software (BSE itself doesn't make a
  per-record license claim, so the conservative path is
  fetch-on-demand rather than bundle-and-hope).

Estimated effort: ~half-day; mirrors the existing `vqfetch`
plumbing closely. Gates on the basissetdev merge timing (which
is itself gated on Mike's standing rule per
`.release-status/v0.8.0/basissetdev.md`).
