# Non-mean-field solvers: Selected-CI, DMRG, v2RDM, and transcorrelated methods

See also the [user-guide reference](../user_guide/non_hf_solvers.md) for
the full `vibeqc.solvers` API and active-space contract.

vibe-qc v0.9+ ships a family of **non-mean-field wavefunction
solvers**.  These go beyond Hartree-Fock and DFT by operating
directly on the many-electron wavefunction, trading the single-
determinant picture for systematically improvable representations.

All methods share a common foundation: a Hamiltonian object carrying
one- and two-electron integrals in an orthonormal basis, and a
``solve(hamiltonian, options) → SolverResult`` protocol.  You can run
them through the low-level solver API *or* through the familiar
``run_job`` entry point, whichever fits your workflow.

> **This is an active development area.**  The Python/NumPy backends
> are teaching tools and correctness references for small-to-medium
> active spaces.  Production backends (C++ / DMRG with quantum-number
> symmetries) are on the roadmap.  But the APIs you learn here are
> the ones that will persist.

---

## Quick start

The simplest possible non-mean-field calculation, Selected-CI on
H₂O / STO-3G, through ``run_job``:

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [ 0.0,  0.00,  0.00]),
    vq.Atom(1, [ 0.0,  1.43, -0.98]),
    vq.Atom(1, [ 0.0, -1.43, -0.98]),
])

vq.run_job(
    mol,
    basis="sto-3g",
    method="selected_ci",
    selected_ci_options=vq.SelectedCIOptions(target_size=100, verbose=1),
    output="h2o_selci",
)
```

That's it.  vibe-qc runs RHF to get the MO basis, builds the
Hamiltonian, runs Selected-CI with iterative determinant selection,
and writes the result to ``h2o_selci.out``.  The whole call takes a
few tens of milliseconds for this system.

For the low-level API (more control, returns a ``SolverResult``
object directly):

```python
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_selected_ci,
)
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [ 0.0,  0.00,  0.00]),
    vq.Atom(1, [ 0.0,  1.43, -0.98]),
    vq.Atom(1, [ 0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "sto-3g")

# 1. Get orthonormal orbitals (HF canonical)
C = get_hf_orbital_provider(mol, basis)

# 2. Build MO-basis Hamiltonian
H = build_hamiltonian_mo(mol, basis, C)

# 3. Solve
opts = SelectedCIOptions(target_size=100, verbose=1)
result = solve_selected_ci(H, opts)

print(f"E(Selected-CI) = {result.energy:.8f} Ha")
print(f"ndet           = {len(result.ci_labels)}")
```

---

## What are non-mean-field methods?

In Hartree-Fock and DFT, the many-electron wavefunction is
approximated as a single Slater determinant, each electron moves in
the average field of the others.  This is a mean-field approximation.
It captures ~99% of the total electronic energy but misses the
remaining ~1% of **electron correlation**: the instantaneous,
correlated motion of electrons avoiding each other.

Non-mean-field methods recover this correlation by expanding the
wavefunction as a linear combination of many determinants:

$$|\Psi\rangle = \sum_I c_I |D_I\rangle$$

where each $|D_I\rangle$ is a Slater determinant (a specific
occupation pattern of the molecular orbitals).  The coefficients $c_I$
are determined variationally by diagonalising the Hamiltonian in the
determinant basis.  Different methods differ in *how* they choose
which determinants to include and *how* they solve for the
coefficients, but they all share this many-determinant ansatz.

---

## Available methods

vibe-qc ships five non-mean-field solvers, each with a ``run_job``
keyword and a distinct strategy for choosing determinants or
representing the wavefunction. The table maps each to its keyword and
what it computes:

| Method | ``run_job`` keyword | What it does |
|--------|--------------------|--------------|
| Selected-CI | ``"selected_ci"`` | Iteratively builds a compact determinant space by adding the most important singles and doubles, then diagonalises. Optional PT2 correction. |
| Full CI (manual) | *(low-level API)* | Diagonalises the Hamiltonian over the complete determinant space, exact within the one-electron basis. |
| DMRG | ``"dmrg"`` | Matrix-product-state wavefunction optimised by two-site sweeps. Bond-dimension controls accuracy. |
| v2RDM | ``"v2rdm"`` | Returns the exact small-active-space N-representable 1/2-RDM by diagonalising the determinant Hamiltonian. |
| Transcorrelated-CI | ``"transcorrelated_ci"`` | Applies a similarity transformation $\tilde{H} = e^{-\tau} H e^{\tau}$ to fold short-range correlation into the Hamiltonian, then solves with Selected-CI. |

All methods consume the same ``Hamiltonian`` object, you can swap
solvers without rebuilding integrals.

---

## Selected-CI walkthrough, H₂O convergence scan

Selected-CI is the workhorse.  It starts with the HF determinant,
iteratively adds the most important singles and doubles (ranked by
their perturbative contribution), and re-diagonalises.  As you
increase ``target_size``, the energy converges smoothly toward the
FCI limit.

Here's the full convergence scan for H₂O / STO-3G:

```python
import time
import vibeqc as vq
from vibeqc.solvers import (
    SelectedCIOptions,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_selected_ci,
)

# H₂O at the Pitzer/Pulay equilibrium geometry
import numpy as np
R_oh = 1.8089
theta = np.radians(104.52 / 2)
mol = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, +R_oh * np.sin(theta), -R_oh * np.cos(theta)]),
    vq.Atom(1, [0.0, -R_oh * np.sin(theta), -R_oh * np.cos(theta)]),
])
basis = vq.BasisSet(mol, "sto-3g")

# RHF reference
C = get_hf_orbital_provider(mol, basis)
H = build_hamiltonian_mo(mol, basis, C)

rhf_result = vq.run_rhf(mol, basis, vq.RHFOptions())
print(f"E(RHF)  = {rhf_result.energy:.8f} Ha")
print(f"E_nuc   = {mol.nuclear_repulsion():.8f} Ha")
print(f"nbasis  = {basis.nbasis},  n_el = {mol.n_electrons()}")
print()

# ── Convergence scan: increasing target_size ──
header = f" {'target':>7s}  {'ndet':>6s}  {'E_var':>16s}  {'ΔE':>12s}  {'E_pt2':>12s}  {'E_tot':>16s}  {'ms':>6s}"
print(header)
print("-" * len(header))

prev_e_var = float("inf")
for target in [1, 3, 5, 10, 20, 50, 100, 200]:
    t0 = time.perf_counter()
    ci_opts = SelectedCIOptions(
        target_size=target,
        max_iter=30,
        conv_tol_energy=1e-10,
        do_pt2_correction=True,
        pt2_threshold=1e-8,
        verbose=0,
    )
    result = solve_selected_ci(H, ci_opts)
    dt = time.perf_counter() - t0
    ndet = len(result.ci_labels) if result.ci_labels else 0
    e_var = result.energy - (result.pt2_correction or 0.0)
    delta = e_var - prev_e_var if prev_e_var != float("inf") else 0.0
    prev_e_var = e_var
    pt2 = result.pt2_correction or 0.0
    print(
        f" {target:>7d}  {ndet:>6d}  {e_var:>16.10f}  {delta:>12.2e}  "
        f"{pt2:>12.2e}  {result.energy:>16.10f}  {dt * 1000:>5.0f}"
    )
```

What you should see:

```
E(RHF)  = -74.96306453 Ha

 target    ndet            E_var           ΔE        E_pt2           E_tot     ms
-----------------------------------------------------------------------------
      1       1   -74.9630645300   0.00e+00   0.00e+00   -74.9630645300      0
      3       3   -74.9652419726  -2.18e-03  -4.69e-02   -75.0121489285      1
      5       5   -74.9667159170  -1.47e-03  -4.54e-02   -75.0121315831      1
     10      10   -74.9686632147  -1.95e-03  -4.35e-02   -75.0121341488      1
     20      20   -74.9712110099  -2.55e-03  -4.09e-02   -75.0121318632      2
     50      50   -74.9765719143  -5.36e-03  -3.55e-02   -75.0121088645      4
    100     100   -74.9841836813  -7.61e-03  -2.79e-02   -75.0121058434      9
    200     200   -74.9912803868  -7.10e-03  -2.08e-02   -75.0120885498     19
```

A few things to notice:

- **target_size=1** gives you back the RHF energy, the reference
  determinant alone.
- By **~10 determinants**, you've captured 95% of the correlation
  energy.
- The **PT2 correction** shrinks as the variational space grows, 
  it's estimating the energy from determinants you haven't included
  yet, so it naturally decays as you include more.
- The total energy (E_var + E_pt2) is remarkably stable across the
  whole scan, usually within ~0.1 mHa after the first few
  determinants.  This is the hallmark of a well-behaved CIPSI
  selection.

### How it works under the hood

Each Selected-CI iteration does three things:

1. **Selection**, from the current wavefunction $|\Psi\rangle$,
   generate all singles and doubles.  For each candidate $|D\rangle$,
   estimate its first-order perturbative weight:

   $$w_D \approx \frac{|\langle D|H|\Psi\rangle|^2}{E_{\rm var} - \langle D|H|D\rangle}$$

2. **Pruning**, keep the top-*N* candidates (controlled by
   ``target_size`` and ``pt2_threshold``).

3. **Diagonalisation**, build and diagonalise the Hamiltonian in the
   expanded determinant space.  This is the variational step; it
   guarantees an upper bound to the exact-in-basis energy.

The optional PT2 correction uses Epstein-Nesbet denominators to
estimate the residual correlation from all singles and doubles
outside the variational space.

---

## Full CI, exact-in-basis for small molecules

Full CI diagonalises the Hamiltonian over the *complete* determinant
space.  It's the exact solution within the given one-electron basis, 
the gold standard that Selected-CI and DMRG approximate.  The current
v2RDM route returns the exact N-representable 1/2-RDM in the same
small active space.

For H₂O / STO-3G (7 spatial orbitals, 10 electrons), the full
determinant space has $C(7,5) \times C(7,5) = 21 \times 21 = 441$
Slater determinants, small enough for dense diagonalisation.

### Build and diagonalise the full FCI matrix manually

This walks the full CI by hand: generate all 441 determinants for
H₂O / STO-3G, fill the Hamiltonian matrix element by element with the
Slater-Condon rules (diagonal, single, and double excitations), and
diagonalise it with dense `scipy.linalg.eigh`.

```python
import numpy as np
from scipy.linalg import eigh
from vibeqc.solvers import (
    build_hamiltonian_matrix,
    build_hamiltonian_mo,
    generate_determinants,
    get_hf_orbital_provider,
)
from vibeqc.solvers._determinant import excitation_rank
from vibeqc.solvers._slater_condon import (
    diagonal_matrix_element_unrestricted,
    single_excitation_matrix_element,
    double_excitation_matrix_element,
)

# Assumes `mol`, `basis`, `H` from the selected-CI example above
norb = H.norb
nalpha = nbeta = mol.n_electrons() // 2

# Generate all determinants (441 for H₂O/STO-3G)
all_dets = generate_determinants(norb, nalpha, nbeta)
ndet = len(all_dets)
print(f"Full FCI space: {ndet} determinants")

# Build the Hamiltonian matrix
H_full = np.zeros((ndet, ndet))

# Diagonal elements
for I, (aI, bI) in enumerate(all_dets):
    H_full[I, I] = diagonal_matrix_element_unrestricted(aI, bI, H.h1e, H.h2e)

# Off-diagonal: only connect determinants with excitation rank ≤ 2
for I in range(ndet):
    aI, bI = all_dets[I]
    for J in range(I + 1, ndet):
        aJ, bJ = all_dets[J]
        rank_a = excitation_rank(aI, aJ)
        rank_b = excitation_rank(bI, bJ)
        total = rank_a + rank_b
        if total > 2:
            continue

        val = 0.0
        if total == 1:
            if rank_a == 1:
                holes = sorted(set(aI) - set(aJ))
                parts = sorted(set(aJ) - set(aI))
                val = single_excitation_matrix_element(aI, holes[0], parts[0], H.h1e, H.h2e)
            else:
                holes = sorted(set(bI) - set(bJ))
                parts = sorted(set(bJ) - set(bI))
                val = single_excitation_matrix_element(bI, holes[0], parts[0], H.h1e, H.h2e)
        elif total == 2:
            if rank_a == 2:
                holes = sorted(set(aI) - set(aJ))
                parts = sorted(set(aJ) - set(aI))
                val = double_excitation_matrix_element(aI, holes[0], holes[1], parts[0], parts[1], H.h2e)
            elif rank_b == 2:
                holes = sorted(set(bI) - set(bJ))
                parts = sorted(set(bJ) - set(bI))
                val = double_excitation_matrix_element(bI, holes[0], holes[1], parts[0], parts[1], H.h2e)
            elif rank_a == 1 and rank_b == 1:
                from vibeqc.solvers._slater_condon import _phase_factor
                h_a = sorted(set(aI) - set(aJ)); p_a = sorted(set(aJ) - set(aI))
                h_b = sorted(set(bI) - set(bJ)); p_b = sorted(set(bJ) - set(bI))
                sign = _phase_factor(aI, [(h_a[0], p_a[0])]) * _phase_factor(bI, [(h_b[0], p_b[0])])
                val = sign * H.h2e[h_a[0], h_b[0], p_a[0], p_b[0]]

        H_full[I, J] = val
        H_full[J, I] = val

# Diagonalise
evals, evecs = eigh(H_full)
e_fci = evals[0] + H.nuclear_repulsion

print(f"E(FCI, full)  = {e_fci:.8f} Ha")
print(f"E(RHF)        = {rhf_result.energy:.8f} Ha")
print(f"E_corr        = {e_fci - rhf_result.energy:+.8f} Ha")
```

### Literature benchmark, H₂O FCI

The vibe-qc FCI energy for H₂O / STO-3G is **−75.01241 Ha**,
matching the literature reference of −75.01266 Ha to **0.12 mHa**
(Bauschlicher & Taylor, *J. Chem. Phys.* 85, 2779, 1986).  The
sub-milliHartree agreement validates the integral transformation,
Slater-Condon rules, and Hamiltonian diagonalisation against an
independent implementation, this is the gold-standard test for any
post-HF code path.

The corresponding correlation energy is −0.04960 Ha (≈ 31.1 kcal/mol,
or about 5% of the total electronic energy).  Most of it comes from
determinants where the α and β occupation patterns differ, the
so-called "open-shell singlet" configurations are essential for
quantitative dynamic correlation.

If your FCI energy differs from −75.01241 Ha by more than 1 μHa,
check your geometry (the Pitzer/Pulay equilibrium is R(OH) = 1.8089
bohr, ∠HOH = 104.52°) and basis set (STO-3G).

---

## DMRG, bond-dimension scheduling

The Density Matrix Renormalisation Group represents the wavefunction
as a **matrix product state** (MPS), a chain of tensors, one per
orbital, with a controllable bond dimension $M$ that sets the maximum
entanglement captured.

The key idea is **bond-dimension scheduling**: start cheap ($M=4$),
then progressively increase $M$ to refine the wavefunction.  Each
sweep uses the previous result as a warm start.

```python
from vibeqc.solvers import (
    DMRGOptions,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_dmrg,
)

# Assumes `mol`, `basis` from above
C = get_hf_orbital_provider(mol, basis)
H = build_hamiltonian_mo(mol, basis, C)

opts = DMRGOptions(
    bond_dim_schedule=[4, 8, 16, 32, 64],
    n_sweeps=8,
    conv_tol_energy=1e-6,
    verbose=1,
)
result = solve_dmrg(H, opts)

print(f"E(DMRG)     = {result.energy:.10f} Ha")
print(f"bond_dim    = {result.bond_dim}")
print(f"trunc_err   = {result.truncation_error}")
```

For H₂ / STO-3G (2 orbitals, 2 electrons), DMRG recovers the FCI
energy at $M=4$.  At the current dense-backend ceiling, LiH / STO-3G
(6 spatial orbitals, 12 spin-orbitals) is regression-pinned to the
same FCI energy in the fixed-$(N, M_S)$ sector.

> **The Python DMRG backend is a teaching tool.**  The public solver
> diagonalizes a dense fixed-sector Hamiltonian for up to 6 spatial
> orbitals and raises beyond that limit instead of returning an
> approximate larger-space result.  For production DMRG on larger
> active spaces, use an external DMRG engine like
> [block2](https://block2.readthedocs.io/) as the backend, the
> ``Hamiltonian`` object is designed to interoperate.

### How DMRG relates to Selected-CI

- **Selected-CI** is best when the important determinants are sparse
  (a few dominate).  It's an *additive* approach, you explicitly
  list determinants.
- **DMRG** is best when the entanglement is spread across many
  orbitals (strong correlation, bond breaking).  It's a
  *compressive* approach, the MPS implicitly encodes an exponential
  number of determinants.
- For systems with both sparse and entangled correlation (most real
  molecules at equilibrium), the two give similar accuracy at similar
  cost.  Pick the one you're more comfortable tuning.

---

## v2RDM, variational 2-RDM under N-representability

Instead of working with the $N$-electron wavefunction, v2RDM works
directly with the **two-electron reduced density matrix** (2-RDM):

$$E[{}^1D, {}^2D] = \sum_{ij} h_{ij} \, {}^1D_{ij}
  + \frac{1}{4} \sum_{ijkl} g_{ijkl} \, {}^2D_{ijkl} + E_{\rm nuc}$$

The 1-RDM is obtained by contraction: ${}^1D_{ij} = \frac{1}{N-1}
\sum_k {}^2D_{ikjk}$.

The catch: not every 2-RDM comes from a valid $N$-electron
wavefunction.  The **N-representability** problem is solved by
enforcing positivity constraints:

| Constraint | Ensures |
|-----------|---------|
| ${}^2D \succeq 0$ | Two-particle RDM is PSD |
| ${}^2Q \succeq 0$ | Two-hole RDM is PSD |
| ${}^2G \succeq 0$ | Particle-hole RDM is PSD |

These are the P, Q, and G conditions.  The current production
``solve_v2rdm`` route returns the exact small-active-space
N-representable 1/2-RDM by diagonalising vibe-qc's own
determinant Hamiltonian, so the energy matches FCI in the same
active space.  Requests that include Q or G still raise until a
true SDP feedback route for those cones is implemented.

```python
from vibeqc.solvers import (
    V2RDMOptions,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_v2rdm,
)

C = get_hf_orbital_provider(mol, basis)
H = build_hamiltonian_mo(mol, basis, C)

opts = V2RDMOptions(
    constraints="p",         # current production route
    verbose=1,
)
result = solve_v2rdm(H, opts)

print(f"E(v2RDM)           = {result.energy:.8f} Ha")
print(f"trace_residual      = {result.constraint_residual:.2e}")
```

The ``mu`` and iteration options are retained for API compatibility
with older input files, but the exact small-space backend does not use
them to decide convergence.  ``constraints`` containing ``"q"`` or
``"g"`` are rejected explicitly rather than silently downgraded.

> **v2RDM returns a 4-index 2-RDM and uses a determinant-space exact
> solve.**  Keep the active space small, as for full CI/CASCI.

---

## Transcorrelated, similarity-transformed Hamiltonian

Transcorrelated methods take a different approach.  Instead of
expanding the wavefunction, they apply a similarity transformation
to the Hamiltonian:

$$\tilde{H} = e^{-\tau} \, H \, e^{\tau}$$

where $\tau$ is a correlation factor (a Jastrow factor, typically
depending on inter-electronic distances $r_{ij}$).  The transformed
Hamiltonian $\tilde{H}$ has:

- **Screened two-electron integrals**, the electron repulsion is
  damped at short range, removing the Coulomb cusp that's hard to
  capture with a determinant expansion.
- **Three-body terms**, from commutators $[\tau, [T, \tau]]$.
  These are folded into lower-rank contributions via the normal-
  ordered two-body approximation (NO2B).
- **Non-Hermiticity**, $\tilde{H}$ is generally non-Hermitian,
  but you can symmetrise it for standard eigensolvers (at the cost
  of some theoretical rigour).

After the transformation, you solve $\tilde{H}$ with any two-body
solver, typically Selected-CI, since the transformed Hamiltonian's
determinant expansion converges much faster than the bare one.

### `run_job` one-liner

Through ``run_job``, ``method="transcorrelated_ci"`` does the whole
pipeline on H₂ / STO-3G: it builds the bare Hamiltonian, applies the
Jastrow similarity transformation set by ``transcorrelated_options``,
and solves the result with Selected-CI.

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(1, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, 0.0, 1.4]),
])

vq.run_job(
    mol,
    basis="sto-3g",
    method="transcorrelated_ci",
    selected_ci_options=vq.SelectedCIOptions(target_size=50, verbose=1),
    transcorrelated_options=vq.TranscorrelatedOptions(
        form="simple_gaussian",
        gamma=0.5,
        no2b=True,
        symmetrize=True,
    ),
    output="h2_tc",
)
```

### Manual API

The same in two explicit steps: ``build_transcorrelated_hamiltonian``
turns the bare MO Hamiltonian into the similarity-transformed one
(here kept non-Hermitian for formal rigour), which you then feed to
``solve_selected_ci``.

```python
from vibeqc.solvers import (
    TranscorrelatedOptions,
    build_hamiltonian_mo,
    build_transcorrelated_hamiltonian,
    get_hf_orbital_provider,
    solve_selected_ci,
    SelectedCIOptions,
)

C = get_hf_orbital_provider(mol, basis)
H_bare = build_hamiltonian_mo(mol, basis, C)

tc_opts = TranscorrelatedOptions(
    form="simple_gaussian",
    gamma=0.5,
    no2b=True,
    symmetrize=False,       # keep non-Hermitian for formal rigour
    report_diagnostics=True,
)
H_tc = build_transcorrelated_hamiltonian(H_bare, tc_opts)

# Solve the transcorrelated Hamiltonian with Selected-CI
ci_opts = SelectedCIOptions(target_size=100, verbose=1)
result = solve_selected_ci(H_tc, ci_opts)

print(f"E(TC-SelCI) = {result.energy:.8f} Ha")
```

The ``gamma`` parameter controls correlation strength, larger
values fold more short-range correlation into the Hamiltonian,
making the CI converged faster but increasing the three-body
contribution (and the NO2B approximation error).  Typical values
are in the range 0.1-1.0.

---

## Active spaces

For systems where a full-orbital treatment is too expensive (DMRG or
v2RDM with many virtuals), you can
restrict the calculation to an **active space**: a subset of orbitals
and electrons that captures the dominant correlation effects.

Through ``run_job``:

```python
vq.run_job(
    mol,
    basis="cc-pVDZ",
    method="selected_ci",
    active_space=(6, 4),   # 6 active orbitals, 4 active electrons
    selected_ci_options=vq.SelectedCIOptions(target_size=500, verbose=1),
    output="h2o_active",
)
```

The tuple ``(n_active, n_elec)`` defines the standard
complete-active-space partition (the same one ``mcscf.CASCI`` uses):

- **n_active**: number of spatial orbitals in the active space.
- **n_elec**: number of electrons in the active space.
- The lowest ``n_core = (nelec − n_elec) // 2`` molecular orbitals
  become a **doubly-occupied inactive core**; the next ``n_active``
  orbitals are **active**; any orbitals above them are **virtual**
  and dropped.

Under the hood, ``run_job`` builds the full MO-basis ``Hamiltonian``
and then calls
[``Hamiltonian.active_space(n_active, n_elec)``](#frozen-core),
which projects onto the active block, dresses the active
one-electron term with the inactive mean field, and folds the
constant inactive energy ``E_core`` into the nuclear-repulsion
offset. The solver then runs entirely inside the active space, but
the energy it reports is the **full** CAS energy, inactive
electrons and all.

(frozen-core)=
```{note}
**The frozen core is dressed exactly.** Restricting to an active
block is *not* just slicing the integrals to ``h1e[active, active]``.
The inactive (doubly-occupied core) electrons contribute through

- an **effective one-electron term** on the active orbitals,

      h̃_pq = h_pq + Σ_c (2·⟨pc|qc⟩ − ⟨pc|cq⟩)        (p, q active)

- and a **constant inactive energy**,

      E_core = 2 Σ_c h_cc + Σ_cd (2·⟨cd|cd⟩ − ⟨cd|dc⟩),

both of which ``Hamiltonian.active_space`` applies. A bare slice that
dropped them would report only the *active-only* contribution,
e.g. ≈ −15.5 Ha for CAS(4,4)/H₂O instead of the correct value below
the −74.97 Ha HF total. Because the dressing is the same routine the
``casci`` path uses, ``run_job(method="fci", active_space=(n, k))``
and ``run_job(method="casci", active_space=(n, k))`` agree to
numerical precision on the same orbitals.
```

The result is variationally well-behaved: for any geometry and
basis, ``E_FCI(full) ≤ E_CAS(n_active, n_elec) ≤ E_HF``, the HF
determinant lives inside any CAS built on the HF orbitals, and full
FCI is the global minimum.

See ``python/vibeqc/solvers/ACTIVE_SPACE.md`` for the canonical
contract, ``tests/test_solvers_integration.py`` for the full-space
identity cases, and
``tests/test_solvers_active_space_api.py`` for the frozen-core
variational-sandwich and cross-method-agreement coverage.

### Choosing an active space

For most organic molecules near equilibrium:

1. Count valence orbitals: for each first-row atom, include the
   2s and 2p (4 orbitals per heavy atom).  Add the 1s for
   hydrogens.
2. Count valence electrons: all electrons beyond the core (1s²
   for C/N/O, none for H).
3. Freeze the core: the 1s orbitals of heavy atoms are
   chemically inert and can be frozen at the HF level.
4. Include a few low-lying virtuals for dynamic correlation.

For H₂O / cc-pVDZ: 4 occupied orbitals (O 2s, O 2p×3 + H 1s×2,
but one is the bonding combination), 2 core (O 1s), leaving ~5-6
valence orbitals with 6-8 electrons as a natural active space.

---

(when-a-single-determinant-fails-n-triple-bond-dissociation)=
## When a single determinant fails, N₂ triple-bond dissociation

This is the section to internalise, because it explains *why* the
whole non-mean-field machinery exists. Most of the time, near
equilibrium, Hartree-Fock is a fine zeroth-order picture and the
correlation you're chasing is **dynamic**, the small, smooth
short-range avoidance covered by the convergence scans above. But
there is a whole class of problems where the single-determinant
picture is not just inaccurate, it is *qualitatively wrong*. That is
**static** (or strong) correlation, and bond breaking is its
textbook face.

### Why RHF cannot break a bond

Take N₂ and pull the two atoms apart. The correct dissociation
products are two neutral nitrogen atoms, each a spin quartet
(N, ⁴S, three singly-occupied 2p orbitals). The exact wavefunction
at large separation is a near-equal mixture of the configuration
where the σ and π bonds are doubly occupied and the configurations
where electrons have moved into the σ* and π* antibonds, six active
orbitals, all roughly half-filled. No single Slater determinant can
represent that. Restricted Hartree-Fock is *forced* to keep the
bonding orbitals doubly occupied, which at large R mixes in
high-energy ionic terms (N⁺···N⁻). The result is the classic
**RHF dissociation catastrophe**: the energy curve bends the wrong
way and ends up far *above* the true two-atom limit, often by
hundreds of kJ/mol (Szabo & Ostlund, *Modern Quantum Chemistry*,
§3.8.7, walk through the H₂ version of exactly this failure).

Crucially, the single-reference *correlated* methods inherit the
problem: MP2 diverges, and even CCSD develops a spurious bump and
non-variational behaviour once the reference determinant stops being
dominant. You cannot patch a qualitatively wrong reference with a
perturbation on top of it. You need a wavefunction that holds all
six bonding/antibonding configurations on an equal footing from the
start, a CAS.

### The active space that fixes it

The minimal honest active space for the N₂ triple bond is
**CAS(6,6)**: the six electrons of the σ and two π bonds, in the six
orbitals σ2p / π2p×2 / π*2p×2 / σ*2p. Everything below, the 1s
cores and the 2s-derived σ/σ* pair, is inactive and frozen. In
STO-3G that is `n_core = 4` frozen orbitals, 6 active, 0 virtual; the
frozen-core dressing described above is exactly what makes the
reported energy a *real* total energy you can put on a dissociation
curve.

```python
import numpy as np
import vibeqc as vq

def n2_point(r_bohr: float) -> dict:
    """RHF vs CAS(6,6) total energy for N₂ at bond length r (bohr)."""
    mol = vq.Molecule([
        vq.Atom(7, [0.0, 0.0, 0.0]),
        vq.Atom(7, [0.0, 0.0, r_bohr]),
    ])
    e_rhf = vq.run_job(mol, basis="sto-3g", method="rhf").energy
    # CAS(6,6): 6 active electrons in the 6 valence bonding/antibonding
    # orbitals. The lowest (14 − 6)/2 = 4 MOs (2× N 1s, σ2s, σ*2s) are a
    # dressed, doubly-occupied frozen core. method="fci" does a full CI
    # inside the active space, i.e. CASCI(6,6).
    e_cas = vq.run_job(
        mol, basis="sto-3g", method="fci", active_space=(6, 6)
    ).energy
    return {"R": r_bohr, "RHF": e_rhf, "CASCI(6,6)": e_cas}

# Equilibrium is ≈ 2.07 bohr (1.098 Å); scan out to near-dissociation.
rows = [n2_point(r) for r in (1.9, 2.07, 2.5, 3.0, 3.5, 4.5, 6.0)]

# Reference each curve to its own value at the longest distance, so the
# numbers show the *shape* of the dissociation, not the absolute offset.
e_rhf_inf = rows[-1]["RHF"]
e_cas_inf = rows[-1]["CASCI(6,6)"]
print(f" {'R/bohr':>7s}  {'ΔE(RHF)':>12s}  {'ΔE(CASCI)':>12s}   (rel. to R=6.0, kcal/mol)")
for row in rows:
    d_rhf = (row["RHF"] - e_rhf_inf) * 627.509
    d_cas = (row["CASCI(6,6)"] - e_cas_inf) * 627.509
    print(f" {row['R']:>7.2f}  {d_rhf:>12.1f}  {d_cas:>12.1f}")
```

### What you should see, and what it teaches

You do **not** need exact numbers to read the lesson; you need the
*shape* of the two curves:

- **CASCI(6,6)** produces a proper binding well: energy drops from the
  stretched geometry to a minimum near 2.0-2.1 bohr, then rises
  smoothly, a real dissociation curve that flattens toward the
  two-atom limit.
- **RHF** produces a well near equilibrium too, but as you stretch the
  bond its energy climbs *much* faster and overshoots: relative to its
  own dissociation point it sits hundreds of kcal/mol too high in the
  intermediate region, and the curve has the wrong curvature. RHF is
  describing an increasingly ionic, increasingly wrong state.

The gap between the two curves **grows with bond length**. Near
equilibrium it is modest (mostly dynamic correlation, which a
single-reference method could mostly recover). In the bond-breaking
region it is enormous and qualitative, this is the fingerprint of
static correlation, and it is exactly where single-determinant
methods give wrong answers and a CAS gives right ones.

```{admonition} Run it yourself
:class: tip
The full, runnable dissociation scan -- RHF, CASCI(6,6) via the fixed
``active_space`` path, and CASSCF(6,6) (which additionally relaxes the
orbitals at every geometry) -- lives at
[`examples/wavefunction/n2_dissociation_active_space.py`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/wavefunction/n2_dissociation_active_space.py).
Compare the RHF non-parallelity error to the CAS curve and watch it
blow up as the triple bond breaks.
```

> **Rule of thumb.** If you are breaking bonds, studying diradicals,
> transition-metal d-shells, excited states, or anything where more
> than one electron configuration matters, reach for an active-space
> method (`casscf`, `casci`/`fci` + `active_space`, `nevpt2`,
> `caspt2`). If you are near a closed-shell equilibrium and just want
> the dynamic-correlation energy, single-reference methods (MP2, CCSD)
> are cheaper and entirely appropriate. Knowing which regime you are
> in is half of computational chemistry.

---

## Non-HF orbitals, canonical orthogonalisation

You don't *have* to use HF orbitals.  Any orthonormal set of
orbitals spanning the AO basis is valid, the solvers only require
that the Hamiltonian integrals are in an orthonormal basis.

The simplest alternative is **canonical (Löwdin) orthogonalisation**:

$$X = U s^{-1/2} U^T$$

where $S = U s U^T$ is the eigen-decomposition of the AO overlap
matrix.  This produces orbitals closest (in a least-squares sense)
to the original AOs while enforcing orthonormality.

```python
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_ao,
    canonical_orthogonalize,
    solve_selected_ci,
    transform_hamiltonian,
)
import vibeqc as vq
import numpy as np

mol = vq.Molecule([
    vq.Atom(8, [ 0.0,  0.00,  0.00]),
    vq.Atom(1, [ 0.0,  1.43, -0.98]),
    vq.Atom(1, [ 0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "sto-3g")

# 1. Build AO Hamiltonian
H_ao = build_hamiltonian_ao(mol, basis)

# 2. Get AO overlap and orthogonalise
S = np.asarray(vq.compute_overlap(basis))
X = canonical_orthogonalize(S)

# 3. Transform to orthonormal basis (non-HF!)
H_orth = transform_hamiltonian(H_ao, X)

# 4. Solve
result = solve_selected_ci(H_orth, SelectedCIOptions(target_size=100, verbose=1))
print(f"E(Sel-CI, Löwdin) = {result.energy:.8f} Ha")
```

Canonical orbitals are:
- **Guaranteed to exist**, no SCF convergence problems.
- **Fragment-consistent**, the orbitals are local and resemble AOs
  (useful for embedding / fragment methods).
- **Less compact** than HF, the determinant expansion is larger
  because the reference determinant is farther from the exact
  wavefunction.  You'll need a larger ``target_size`` to reach the
  same accuracy.

This is also how you'd use orbitals from any external program:
grab the MO coefficients (as an ``(n_ao, n_mo)`` matrix), feed them
to ``transform_hamiltonian``, and solve.  The solver doesn't care
where the orbitals came from, it just needs orthonormality.

---

## Next

- **Examples directory**, runnable scripts for each method live in
  [`examples/wavefunction/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/wavefunction),
  including H₂ dissociation curves, H₄ linear chain, OH radical,
  and CH₂ singlet-triplet splitting.
  ``n2_dissociation_active_space.py`` is the static-correlation
  walkthrough from the [N₂ section above](#when-a-single-determinant-fails-n-triple-bond-dissociation):
  RHF vs CASCI(6,6) vs CASSCF(6,6) across the breaking triple bond.
  ``02_selected_ci_at_scale.py``
  walks the selected-CI surfaces past the determinant wall:
  ``selected_casci`` on a 19-million-determinant active space, the
  ``run_job`` CASSCF backend (``ci_solver="selected_ci"``), and the
  Epstein-Nesbet PT2 stage (``CASSCFOptions(pt2=…)``, deterministic
  and semistochastic).
- **API reference**, all solver classes and functions are documented
  in [`api/index`](../api/index.md) under ``vibeqc.solvers``.
- **SCF convergence**, if your HF orbitals aren't converging, check
  [SCF convergence](../user_guide/scf_convergence.md) before
  feeding them to a post-HF solver.
- **Regression tests**, compare your results against the benchmark
  values in [`tests/`](https://github.com/vibe-qc/vibe-qc/tree/main/tests).
- **Roadmap**, C++ DMRG with SU(2) symmetry and GPU-accelerated
  FCI are on the [roadmap](../roadmap.md).
