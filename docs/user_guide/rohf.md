---
myst:
  html_meta:
    "description": "Restricted open-shell Hartree-Fock (ROHF) and Kohn-Sham (ROKS) in vibe-qc: when to use ROHF vs UHF, the run_job / run_rohf invocation, analytic gradients, geometry optimisation and frequencies, spin-pure CAS references, and a worked open-shell radical example."
    "og:title": "vibe-qc - ROHF and ROKS (restricted open-shell SCF)"
    "og:description": "Spin-pure restricted open-shell references for radicals and high-spin systems: ROHF / ROKS theory, invocation, gradient and optimisation support, and the ROHF-vs-UHF decision."
---

# ROHF and ROKS (restricted open-shell SCF)

vibe-qc has two mean-field references for open-shell molecules:
the **unrestricted** family (`UHF` / `UKS`) and the **restricted
open-shell** family (`ROHF` / `ROKS`) documented here. Both describe the
same radical or high-spin state; they differ in one decision: whether
the α and β electrons are allowed independent spatial orbitals.

This page is the dedicated reference for ROHF/ROKS: when to pick them over
UHF/UKS, how to run them, and what gradient/optimisation/post-SCF support
exists. For the broader open-shell setup (multiplicity, charge) see
[`molecules.md`](molecules.md#configuring-open-shell-systems); for the
Fock-build modes that apply to every SCF method see
[`scf_modes.md`](scf_modes.md).

## When to use ROHF vs UHF

| | **UHF / UKS** (unrestricted) | **ROHF / ROKS** (restricted open shell) |
|---|---|---|
| Spatial orbitals | independent α and β sets | one common set: closed + open + virtual |
| Spin purity | **not** a spin eigenfunction; ⟨S²⟩ deviates from `S(S+1)` (*spin contamination*) | **spin-pure by construction**: ⟨S²⟩ = `S(S+1)` *exactly* |
| Energy | variationally lowest single determinant | slightly higher than UHF (extra orbital constraint) |
| Best for | quick open-shell SCF, bond breaking / strong static correlation where symmetry breaking is wanted | **spin-pure references** for post-HF (ROHF-MP2 / ROHF-CC), a clean **CAS starting point**, and properties sensitive to spin contamination (hyperfine, spin densities) |

**Rule of thumb.** Reach for ROHF/ROKS when spin contamination would
corrupt the answer or the downstream method (a spin-pure CAS reference,
an ROHF-CCSD(T) energy, a clean spin density). Reach for UHF/UKS when you
want the lowest single-determinant energy, are breaking bonds, or need a
large fast DFT relaxation. UHF's ⟨S²⟩ is reported in every open-shell run
so you can see the contamination directly.

## Running ROHF / ROKS

The spin partition (`n_alpha − n_beta = multiplicity − 1`) is read from
`Molecule.multiplicity`, exactly as for UHF. Use the dedicated drivers
(`run_rohf` / `run_roks`) for a result object, or `run_job` for the full
end-to-end run with output files + citations.

```python
import vibeqc as vq

# Hydroxyl radical (doublet). See examples/molecular/input-oh-rohf.py.
oh = vq.Molecule(
    atoms=[vq.Atom(8, [0.0, 0.0, 0.0]),
           vq.Atom(1, [0.0, 0.0, 1.834])],   # bohr
    multiplicity=2,
)
basis = vq.BasisSet(oh, "6-31g*")

rohf = vq.run_rohf(oh, basis)        # spin-pure: <S^2> = 0.75 exactly
print(rohf.energy, rohf.s_squared)

# End-to-end (writes .out / .molden / .references / .bibtex):
vq.run_job(oh, basis="6-31g*", method="rohf", output="oh_rohf")
```

The Kohn-Sham counterpart is **ROKS**, the same Roothaan coupling with a
spin-polarised XC potential:

```python
roks = vq.run_roks(oh, basis, functional="pbe")          # spin-pure KS
vq.run_job(oh, basis="6-31g*", method="roks", functional="b3lyp",
           output="oh_roks")          # see examples/molecular/input-ch3-roks.py
```

ROKS supports **LDA, GGA, meta-GGA, global-hybrid, and
range-separated-hybrid** functionals (B3LYP, PBE0, TPSS, r²SCAN,
ωB97X, ωB97M-V, …). **Double hybrids** (B2PLYP, DSD-PBEP86, PWPB95)
run through `vibeqc.run_double_hybrid` (spin-pure ROKS SCF + a
semicanonical ROHF-MP2 doubles correction); a direct `run_roks` on a
double-hybrid functional points you there because the SCF energy alone
is incomplete. Convergence knobs live on
`ROHFOptions` / `ROKSOptions` (`max_iter`, `conv_tol_energy`,
`conv_tol_grad`, `use_diis`, `level_shift`, `density_fit` / `aux_basis`).

For degenerate ROKS terms, the KS driver averages the frontier shell
instead of forcing one arbitrary component to be the SOMO. Linear
OH(2Pi), for example, is represented with occupations
`2, 2, 2, 1.5, 1.5, 0, ...` across the pi pair; this preserves the
restricted open-shell symmetry and prevents the larger-basis ROKS/PBE
oscillation that an integer `2, 2, 2, 2, 1, 0, ...` assignment produces.

## Convergence

The Roothaan loop carries the same convergence controls as the rest of
the tree: `damping` / `dynamic_damping`, the DIIS family via
`scf_accelerator`, and the Saunders-Hillier `level_shift` with its
`level_shift_warmup_cycles` and `level_shift_schedule` policy. See
[`scf_convergence.md`](scf_convergence.md) for what each one does. Two
restricted-open specifics:

**High-spin transition metals need a held level shift.** A high-spin 3d
complex (FeCl₃ at multiplicity 6, Fe(III) d⁵) settles onto a long
plateau where the energy is converged to `1e-11` Ha while `||[F,DS]||`
creeps down about a decade per hundred iterations and misses
`conv_tol_grad`. That is the DIIS history going near-linearly-dependent,
not an oscillation, so damping it makes it worse. Hold a shift:

```python
opts = vq.ROHFOptions()
opts.level_shift = 0.6
opts.level_shift_warmup_cycles = 0    # hold it; do not auto-release
result = vq.run_job(fecl3, basis="def2-tzvp", method="rohf",
                    rohf_options=opts)
```

Measured over two independent sets of 8 repeats at def2-SVP
(300-iteration cap), against the shipped defaults: **16/16 converged
versus 8/16**, median **55 iterations** with a tight 48 to 59 spread,
against a default that scatters from 63 to 245 and whose median moves by
a factor of two between sets. The shift removes the run-to-run chaos,
not just the mean cost, and that is the part that reproduces. Setting
`scf_accelerator = "r_cdiis"` on top improves the unshifted odds on its
own (14/16), but adds nothing measurable once the shift is held.

Do **not** substitute a damping-first schedule (`damping=0.5`,
`diis_start_iter=11`): on this system it converges tidily onto a
different solution 0.10 Ha higher. How to read the trace, and why
`initial_guess='GWH'` is unavailable here, is in
[`scf_convergence.md`](scf_convergence.md#a-stalled-tail-is-usually-the-diis-history-not-the-state).

**And converging is not the same as being right.** At def2-TZVP the
state the default guess reaches is *itself* wrong: 0.520 Ha above the
correct high-spin d⁵ solution, breaking the molecular symmetry (three
inequivalent Cl on a D₃ₕ molecule). The same held shift reaches the
right one in 48 iterations, reproducing ORCA 6.1.1 to 7×10⁻⁸ Ha. Read
[Convergence is not correctness](scf_convergence.md#convergence-is-not-correctness)
before trusting any high-spin transition-metal ROHF number: it lists the
symmetry and spin-population checks that catch this, and why the guess
that would avoid it is unavailable to this driver.

**Damping is off by default in practice.** The Roothaan loop skips
density mixing once DIIS is extrapolating, and `diis_start_iter`
defaults to 1, so the default `damping = 0.5` only ever applies to the
first iteration. Raise `diis_start_iter` to give damping room.

## Orbital eigenvalues and the Guest-Saunders convention

ROHF has no unique Fock matrix -- the occupied-occupied, virtual-virtual,
and occupied-virtual blocks must be constructed from different combinations
of Coulomb and exchange.  Every quantum chemistry code picks a different
"effective" Fock to diagonalise for orbital eigenvalues, and while the
total energy is convention-independent, the individual eigenvalues differ.

For integer-occupation ROHF and ROKS, vibe-qc uses the
**Guest-Saunders** convention (Guest & Saunders, Mol. Phys. **28**, 819,
1974), which is ORCA's default and a common production ROHF choice. After the
SCF converges, the code diagonalises the appropriate Fock sub-blocks
independently within each subspace:

- **Closed** (doubly occupied): `Fc = (Fα + Fβ) / 2`
- **Open** (singly occupied): `Fα` (alpha Fock -- gives Koopmans IPs)
- **Virtual**: `Fc = (Fα + Fβ) / 2`

The resulting eigenvalues are the Guest-Saunders canonical orbital energies.
They remain in closed/open/virtual occupation-block order and are sorted only
within each block; a global energy sort would detach the eigenvalues and
coefficients from their occupations. Because the transformation is
block-diagonal within each subspace, the density matrix is invariant and the
total energy is unchanged.

Fractional ROKS frontier shells are different. If a degenerate shell crosses
an integer closed/open or open/virtual boundary, applying those integer
Guest-Saunders slices would put symmetry partners in different blocks and can
split them. Such jobs retain the converged Roothaan eigenpairs instead. Their
energies are globally ascending and aligned with the fractional occupations,
and the convention is reported as
``rohf_canonicalization = "roothaan-fractional-shell"``. Integer-shell jobs
continue to report ``"guest-saunders"``.

**How this differs from the Roothaan convention.**  The Roothaan effective
Fock (Rev. Mod. Phys. 32, 179, 1960) uses `Fc` in the open-open block,
giving SOMO eigenvalues that are averages of alpha and beta Fock expectation
values.  Guest-Saunders uses `Fα` there, which typically gives SOMO
eigenvalues 0.1-0.4 Ha lower and better Koopmans-theorem ionisation
potentials.  The original Roothaan eigenvalues are preserved as
``result.mo_energies_roothaan`` for reference.

The selected convention is recorded in ``{output}.system`` under
``[run] rohf_canonicalization`` so downstream tools and comparison scripts can
interpret the numbers correctly.

## Gradients, geometry optimisation, frequencies

**ROHF has an analytic gradient** (`vibeqc.compute_rohf_gradient`), so
geometry optimisation and harmonic frequencies run at full speed:

```python
opt = vq.run_job(oh, basis="6-31g*", method="rohf", optimize=True)
frq = vq.run_job(oh, basis="6-31g*", method="rohf", hessian=True)  # FD of the analytic gradient
```

Density-fitted ROHF uses the matching analytic DF gradient when
`ROHFOptions.density_fit=True` and `aux_basis` is set; the ASE force and
optimisation routes mirror those settings automatically. Direct
`compute_rohf_gradient` calls must pass a `GradientOptions` with the same DF
settings. RIJCOSX and ECP ROHF gradients remain unavailable and fail closed.

**ROKS** geometry optimisation uses **finite-difference forces** for now
(the analytic molecular XC-gradient term is pending), correct but
slower; prefer UKS for large fast DFT relaxations. See
[`geometry_optimization.md`](geometry_optimization.md) for the optimiser
itself.

## Beyond single points

ROHF/ROKS also drive:

* **Spin-pure CAS references**: `cas_reference="rohf"` gives CASSCF/CASCI
  a spin-pure starting determinant (cleaner than UHF natural orbitals);
  see [`non_hf_solvers.md`](non_hf_solvers.md).
* **ROHF-reference CCSD / CCSD(T)**: `run_job(method="ccsd(t)",
  ccsd_reference="rohf")` runs the spin-orbital coupled-cluster kernel on
  the spin-pure reference; see [`ccsd.md`](ccsd.md).
* **Semicanonical ROHF-MP2**: `run_job(method="mp2", mp2_reference="rohf")`
  (or `vibeqc.run_rohf_mp2`) computes second-order Møller-Plesset
  correlation on the spin-pure reference, including the Brillouin singles
  term that UMP2 lacks (Knowles et al. 1991). Needs a basis with a bundled
  RI auxiliary (cc-pVDZ / def2; Pople bases raise a clear error).
* **Atomization energies** (`run_job(atomization=True)`, with spin-pure
  free-atom references), **PES scans** (`vibeqc.scan`), and the
  **molecular optimiser** (`vibeqc.molecular_optimize.optimize_molecule`).
* **The ASE calculator**: `VibeQC(restricted_open=True)` runs ROHF with
  analytic forces; see [`ase_integration.md`](ase_integration.md).

## Periodic ROHF

Restricted open-shell HF is available for **periodic** systems at the
Γ-point and with full Brillouin-zone sampling (EWALD_3D), via the
standalone drivers `run_rohf_periodic_gamma_ewald3d` and
`run_rohf_periodic_multi_k_ewald3d`. A second standalone entry point,
`run_periodic_rohf_gpw`, provides 3D Gamma-point HF with GPW Hartree J,
per-spin exact K, and the Ewald one-electron/nuclear gauge. It uses the
shared Roothaan loop to retain one spatial-orbital set and integer 2/1/0
occupations.

Both engines are available through the unified dispatcher.
`method="ROHF"` alone (AUTO) selects the BIPOLE-route
corrected-Ewald-exchange engine, which also accepts a full Monkhorst-Pack
mesh via `kpoints=[n1, n2, n3]`:

```python
result = vibeqc.run_periodic_job(
    system,
    basis,
    method="ROHF",          # jk_method="bipole" is the AUTO default
    kpoints=[3, 1, 1],      # omit for Gamma
)
```

The 3D Gamma GPW route is also available through the unified dispatcher:

```python
result = vibeqc.run_periodic_job(
    system,
    basis,
    method="ROHF",
    jk_method="gpw",
)

roks = vibeqc.run_periodic_job(
    system,
    basis,
    method="ROKS",
    functional="pbe",
    jk_method="gpw",
)
```

The GPW routes keep the same restricted 2/1/0 orbital contract and fail
closed outside their validated envelope. ROKS adds the spin-polarised
periodic GPW XC build for LDA, GGA, meta-GGA, and global hybrids. The
GPW routes are not GDF, GAPW, 1D/2D, or multi-k implementations (GDF and
BIPOLE routes for periodic ROHF are available through the unified
AUTO dispatcher above). Gradients, restart densities, and electronic
smearing for the GPW routes are still pending. The `smearing_alpha`
keyword on the standalone drivers controls nuclear-charge smoothing,
not electronic Fermi smearing. See
[`periodic_methods.md`](periodic_methods.md) § 3.8 for the periodic SCF
context.

## Theory

ROHF uses **Roothaan's single effective Fock operator** (Roothaan,
*Rev. Mod. Phys.* **32**, 179 (1960)). The MOs are partitioned into
closed, open, and virtual blocks with integer occupations (2/1/0). A
single effective Fock matrix is assembled from the per-spin Fα/Fβ via
Coulson's coupling so that one diagonalisation yields a spin eigenfunction
with ⟨S²⟩ = `S(S+1)` exactly. ROKS reuses the identical coupling but with
Fα/Fβ carrying the spin-polarised KS exchange-correlation potential and,
for degenerate frontier terms, uniform fractional occupations within the
open block. Its energy uses E_xc[ρα, ρβ] in place of full exact exchange. The
`roothaan_rohf_1960` citation fires automatically in any ROHF/ROKS run
(including CAS jobs that use `cas_reference="rohf"`).

## See also

* [`molecules.md`](molecules.md#configuring-open-shell-systems): open-shell
  configuration (multiplicity, charge) and the UHF/ROHF choice in context.
* [`functionals.md`](functionals.md): functional families (ROKS supports
  LDA / GGA / meta-GGA / global and range-separated hybrids).
* [`scf_modes.md`](scf_modes.md): AUTO / CONVENTIONAL / DIRECT Fock-build
  modes, shared by all SCF methods.
* [`non_hf_solvers.md`](non_hf_solvers.md): CAS / multireference methods
  (`cas_reference="rohf"`).
* [`ccsd.md`](ccsd.md): coupled cluster, incl. `ccsd_reference="rohf"`.
