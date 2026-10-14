# Experimental chi scalar semilocal ECPs

Classic physical-pair chi supports scalar semilocal effective core potentials
in neutral three-dimensional RI RHF, UHF and shared-orbital ROHF. The same
sealed references feed canonical MP2, UMP2, Knowles RMP2, and RHF/UHF CCSD with
the defined canonical triples correction. This is a source capability, not
native scientific acceptance. Chi remains experimental, D93 fitted/post-HF
remains not-qualified, and neither whole R5 nor production completion is claimed.

Spin-orbit ECPs, SR ECP combinations, historical-index support, charged cells,
lower dimensions, fractional occupations, KS/hybrid correlation, ROHF-CC and
ECP gradients remain unsupported. Existing all-electron defaults and the
chi/CCM construction firewall are unchanged. The public runner's existing
UMP2 frozen-space restriction still applies; RHF/UHF CC and the low-level
canonical consumers retain their supported spin-resolved active spaces.

## Hamiltonian and finite support

For each ion, `Z_eff = Z - N_ECP`. The one-electron Hamiltonian is
`h = T + V_Coulomb(Z_eff) + U_local + sum_lm |lm> Delta_U_l <lm|`.
The Gaussian short-range terms contain no additional Coulomb tail. The
ionic Ewald energy uses the same effective charges, and the unchanged
electron-electron fitted operator acts only on the remaining explicit
electrons. Bare element labels and the orbital basis remain intact.

This separation follows [Shaw and Hill (2017)](https://doi.org/10.1063/1.4986887),
section I, equations 1-3 and section II, equations 4-16, and the CRYSTAL23
manual, section 3.2, equations 3.16-3.19. Scalar averaged-relativistic
potentials are distinct from the spin-orbit terms in equations 3.20-3.28.
See also [Andrae et al. (1990)](https://doi.org/10.1007/BF01114537), section 2.
The existing periodic Coulomb zero-mode/background convention is retained;
its ionic term uses the Ewald decomposition, not an isolated ion-pair sum
([Ewald (1921)](https://doi.org/10.1002/andp.19213690304), section IV.1,
equations 37-40). Electronic and ionic conventions must match in comparisons.

The existing normalized spherical native libecpint integrals evaluate local
and nonlocal components separately. The largest angular channel is local;
its zero-coefficient placeholder is preserved even for a purely nonlocal
component. Native radial `ns` means a literal power `ns-2`, including when
converting a CRYSTAL radial parameterization. ECP-paired bases require complete
primitive data; changing nuclear charges alone cannot enable this route.

The opt-in native `physical_pair_support=True` mode requires the pair-complete
one-electron enclosure and applies the actual AO shell-center distance cutoff.
For each admitted pair, projector images use the shared AO-midpoint nuclear
selector with the explicit nuclear real-space radius. This is a finite
definition, not a certified ECP tail bound. The Bloch sum uses the positive
translated-ket phase `exp(+i k.R)`. Individual translation blocks are not
symmetrized; reverse translations supply the transpose partners. Full complex
Hermiticity, common translations and changes of lattice representative are
checked independently. All energies and counts are per primitive cell.

Final physical Focks include all four one-electron components. Stationary
acceptance and integer occupations use `N_explicit = sum(Z)-charge-N_ECP`.
Both RHF/UHF drivers and the shared-spatial ROHF optimizer consume that same
Hamiltonian. ECPs introduce no additional two-electron correlation term.

## Library, runner and retained identities

Use the same explicit `ChiCoulombControls` as in
[fitted support](chi_fitted_support.md), with `source='classic'` and
`support_policy='physical_pairs'`. The runner resolves the bundled basis
sidecar through its existing ECP machinery. A library caller supplies the
equivalent complete periodic options before preparing the reference:

```python
from vibeqc import PeriodicRHFOptions
from vibeqc.ecp_metadata import inline_ecp_data_for
from vibeqc.periodic.chi.reference import prepare_chi_hf_reference

options = PeriodicRHFOptions()
options.conv_tol_energy = 1e-11
options.conv_tol_grad = 1e-9
blocks, centers, charges, ncore = inline_ecp_data_for(
    system.unit_cell_molecule(), basis.name)
options.ecp_primitive_blocks = blocks
options.ecp_home_centers = centers
options.ecp_effective_charges = charges
options.ecp_total_ncore = ncore
reference = prepare_chi_hf_reference(
    system, basis, options=options, method="RHF", mesh=(1, 1, 1),
    backend="ri", fit_source="classic", coulomb_controls=controls,
    aux_basis="def2-universal-jkfit")
```

Use `PeriodicRHFOptions` for all three references, with the tight reference
convergence tolerances required by `prepare_chi_hf_reference`. Inline custom
potentials use the same four fields. XML recipes,
molecular `ecp_primitive_centers`, and spin-orbit payloads are not this route.
Missing or inconsistent operator, center, charge, core or spin data refuse
before native construction. A dry run records a request, never execution.

The ECP identity binds atom mapping, full primitive coefficients/exponents/
powers/angular channels, effective charges, removed cores and explicit
populations. The executed one-electron source adds the actual lattice cells,
shell origins and lossless pair/projector selector specification before
screening. It does not claim every candidate caused an integral evaluation.
Reference seals and portable identities also bind the retained kinetic,
ionic-Coulomb, local and nonlocal arrays. Changing any ECP data invalidates
reference, cache and correlation continuation even if Coulomb factors match.

Output through `vibeqc.output` reports the executed ECP identity, component
energies and ionic Ewald term, plus physical, ECP-removed, explicit,
explicitly frozen and correlated electron counts. Existing ECP and basis
citations apply. Execution metadata is required; an accepted basis name or
keyword alone is insufficient evidence of an ECP Hamiltonian.

## Frozen spaces and persistence

`frozen_core` counts only occupied orbitals among the explicit electrons,
per character and spin. It never adds `N_ECP`. For sulfur/LANL2DZ the ten ECP
core electrons are already absent: H2S has eight explicit electrons, and
freezing one occupied orbital per spin leaves six correlated electrons.
SH has seven explicit electrons; ROHF freezing one closed orbital leaves
five. Supplying an ECP core count as a frozen-space request cannot remove
those already absent orbitals; invalid active spaces refuse.

Match reference-program conventions explicitly. ORCA 6.1 section 2.21 counts
ECP-removed electrons in `NewNCore`: ten ECP electrons plus one frozen spatial
orbital correspond to twelve core electrons in that convention. A request
to correlate all explicit electrons must disable additional frozen orbitals.
Do not assume an identical integer keyword means an identical active space.

[HF persistence](chi_restart.md) uses semantic
`vibeqc.chi.hf-restart/v2` for ECP references, retaining full operator data,
component arrays and effective electron counts. All-electron v1 remains
readable and writable. This method version is separate from the ZIP envelope
version. Restoration is native-free and validates source, populations,
physical Focks and energies before issuing a fresh process-local seal.
Canonical correlation restart binds that portable reference identity;
changed-ECP continuation and v2-to-v1 relabeling refuse.

## Checks and later numerical acceptance

Focused checks use analytic normalized Gaussian local/projector integrals,
an independent finite image enumeration with molecular native integrals,
effective-charge and Ewald scaling, complex covariance, electron/spin and
frozen-space bookkeeping, source/checkpoint mutation refusal, and independent
finite-space correlation contractions. These are implementation checks, not
an ORCA/CRYSTAL scientific verdict.

[`examples/periodic/chi_ecp.py`](../examples/periodic/chi_ecp.py) provides
H2S/RHF and SH/UHF/ROHF inputs with LANL2DZ at Gamma and three characters,
explicit finite support, correlation and checkpoint/resume switches. Outputs
must be outside the checkout. Its conservative memory and record reservations
are not eager allocations or scheduler resource limits. Validator must use a
runtime containing this native increment, submit through `vq`, and record
actual source/native identities, convergence, energy components and counts.
The delivery issue holds the single reproducible acceptance matrix.

Match geometry/cell, orbital basis and exact ECP parameters, auxiliary/fitting
approximation, charge/spin, zero-mode conventions, active spaces, convergence,
units and tolerances. CRYSTAL periodic comparisons require compatible finite
support or independently converged limits. ORCA molecular results require a
cell-size and support ladder; a finite periodic value is not expected to equal
an isolated molecule exactly. RMP2 partition and CC triples definitions must
match their respective guides. An unavailable or mismatched reference leaves
an explicit uncertainty, never an inferred scientific PASS.
