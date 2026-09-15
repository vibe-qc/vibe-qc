# CC2, CC3, CCSD, CCSDT, QCISD, and perturbative triples

Canonical coupled cluster is the small-molecule accuracy reference in
vibe-qc. `method="ccsd"` runs DF-CCSD; `method="ccsd(t)"` adds the
standard perturbative triples correction. The same choice can be made
explicitly with `triples="none"` or `triples="(t)"`. Density fitting is
the default integral route; pass `density_fit=False` (or
`CCSDOptions(density_fit=False)`) for the conventional route with exact
four-index integrals (see
[Conventional (non-DF) integrals](#conventional-non-df-integrals)).
Closed-shell QCISD and QCISD(T) are available through `method="qcisd"`
and `method="qcisd(t)"`. Ground-state closed-shell CC2 is available through
`method="cc2"` or `method="ci", citype="cc2"`. Full iterative CCSDT is
available through `method="ccsdt"` or `method="ci", citype="ccsdt"` for
benchmark-scale molecular calculations. Ground-state CC3 is available through
`method="cc3"` or `method="ci", citype="cc3"` for closed-shell benchmark
calculations.

Both closed- and open-shell references are handled automatically: a
closed-shell (singlet) molecule runs on an RHF reference through the
spin-adapted kernel, and an open-shell molecule (`multiplicity > 1`)
runs on a UHF reference through the spin-orbital UCCSD(T) kernel by
default.  For a spin-pure restricted-open-shell reference, pass
``ccsd_reference="rohf"``: the same spin-orbital kernel is used,
with ROHF supplying identical alpha/beta spatial orbitals and
per-spin Fock matrices.

See the [canonical CCSD(T) tutorial](../tutorial/coupled_cluster_ccsd_t.md)
for a worked, paste-and-run walkthrough of every reference type below.

## At a glance

| | |
|---|---|
| High-level entry point | `run_job(method="cc2")`, `run_job(method="ccsd", triples=..., ccsd_reference=...)`, `run_job(method="cc3")`, `run_job(method="ccsdt")`, `run_job(method="qcisd")` |
| Low-level entry point | `run_ccsd` / `run_uccsd` / `run_rohf_ccsd` / `run_uccsd_from_mos`; `cc3`; `ccsdt` |
| Reference | CCSD: RHF, UHF, or ROHF; CC3: RHF; CCSDT: RHF or common-orbital ROHF |
| Integral form | CCSD: density-fitted or exact four-index; CC3/CCSDT: exact four-index MO Hamiltonian |
| Frozen-core default | ORCA 6.1 chemical-core count; `frozen_core=False` selects all-electron |
| Scope | CCSD: dense O(N^6) small-molecule pilot; CC3/CCSDT: dense determinant-space benchmark routes |
| Validation | in-repo spin-orbital/FCI anchors and out-of-process PySCF/ORCA comparisons |

## Quick start

```python
import vibeqc as vq

mol = vq.Molecule(
    [
        vq.Atom(8, [0.000, 0.000, 0.000]),
        vq.Atom(1, [0.000, 1.499, -1.160]),
        vq.Atom(1, [0.000, -1.499, -1.160]),
    ],
)

result = vq.run_job(mol, basis="cc-pvdz", method="ccsd(t)", output="h2o_ccsd_t")
print(result.ccsd.e_ccsd_correlation)
print(result.ccsd.e_t)
print(result.energy_total)
```

`run_job` writes the usual `.out` / citation sidecars and attaches the
post-SCF result as `result.ccsd`.

## Frozen-core convention

The high-level MP2/CCSD and DLPNO correlation routes use one convention for
RHF, UHF, and ROHF references. By default vibe-qc follows ORCA 6.1 Table
2.69: ORCA's frozen-electron count is converted to closed-shell spatial
orbitals and summed over the atoms. The `.out` method block reports that
resolved orbital count.

Use the same public escape on every route when reproducing an ORCA
`NoFrozenCore` deck or an archived all-electron value:

```python
all_electron = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="ccsd(t)",
    frozen_core=False,
)
```

Low-level APIs retain `n_frozen_core` as an explicit spatial-orbital count;
`vq.chemical_core_orbital_count(mol)` returns the project/ORCA-table value.
This is a count-only convention: vibe-qc removes the corresponding
lowest-energy occupied orbitals and does not implement ORCA's extra
atomic-orbital-character reorder. Review or override the count for unusual
all-electron orbital orderings. Molecular ECP references are currently
rejected because their variational electron count differs from the molecule's
physical count; `frozen_core=False` does not restore ECP-removed electrons.

## CC2

`method="cc2"` implements the ground-state second-order approximate coupled
cluster model of Christiansen, Koch, and Jorgensen. It solves the complete
singles projection together with the first-order doubles equation built from
the `T1`-transformed Hamiltonian and the diagonal `[F,T2]` commutator:

```python
result = vq.run_job(mol, basis="cc-pvdz", method="cc2", output="h2o_cc2")
print(result.ccsd.e_ccsd_correlation)  # CC2 correlation energy
print(result.energy_total)
```

CC2 currently supports closed-shell RHF references, frozen occupied cores,
density fitting, and the conventional exact-integral route selected by
`CCSDOptions(density_fit=False)`. It is an energy-only route and does not
accept `triples=`. The dense implementation shares CCSD's integral tensors and
small-molecule memory envelope.

## CC3

`method="cc3"` implements the iterative CC3 model of Koch and coworkers.
Singles and doubles are optimized to convergence while connected triples are
regenerated from the CC3 triples equation on every iteration and coupled back
into the singles and doubles residuals:

```python
result = vq.run_job(
    mol,
    basis="sto-3g",
    method="cc3",
    cc3_options=vq.CC3Options(
        conv_tol_energy=1.0e-10,
        conv_tol_residual=1.0e-8,
    ),
    output="h2o_cc3",
)
print(result.e_correlation)
print(result.e_total)
print(result.t3_norm)
```

The high-level route uses canonical RHF orbitals and requires a closed-shell
singlet. It freezes chemical-core orbitals by default; use
`CC3Options(n_frozen_core=0)` for an all-electron calculation. The low-level
`vq.cc3(hamiltonian, options)` entry point accepts an orthonormal
spatial-orbital `Hamiltonian` and leaves frozen-core selection explicit.

CC3 is an energy-only, exact-integral benchmark route. The current
determinant-space implementation evaluates the defining commutators directly,
so its memory and runtime grow combinatorially and it is intended for compact
orbital spaces. Iteration details are written at the `verbose` output level;
the standard `.out` contains the converged summary and amplitude norms.

## Full CCSDT

`method="ccsdt"` solves the full coupled-cluster singles, doubles, and triples
equations. The implementation evaluates the defining projected similarity
transform directly in determinant space, including all connected and
disconnected terms generated by `T1 + T2 + T3`:

```python
result = vq.run_job(
    mol,
    basis="sto-3g",
    method="ccsdt",
    ccsdt_options=vq.CCSDTOptions(
        conv_tol_energy=1.0e-10,
        conv_tol_residual=1.0e-8,
    ),
    output="h2o_ccsdt",
)
print(result.e_correlation)
print(result.e_total)
print(result.t3_norm)
```

The high-level route uses RHF orbitals for closed shells and common-orbital
ROHF orbitals for open shells. It freezes chemical-core orbitals by default;
set `CCSDTOptions(n_frozen_core=0)` for an all-electron calculation. The
low-level `vq.ccsdt(hamiltonian, options)` function accepts an orthonormal
spatial-orbital `Hamiltonian` and leaves frozen-core selection explicit.

CCSDT is an energy-only, exact-integral benchmark route. Its determinant-space
algorithm grows combinatorially, so it is intended for compact orbital spaces
and method validation rather than large production molecules. It does not
accept the perturbative `triples=` selector or `CCSDOptions`. Iteration details
are written at the `verbose` output level; the normal `.out` contains the
converged summary and amplitude norms.

## Triples selector

The high-level driver accepts a `triples=` selector for the perturbative
triples mode:

```python
vq.run_job(mol, basis="cc-pvdz", method="ccsd", triples="none")
vq.run_job(mol, basis="cc-pvdz", method="ccsd", triples="(t)")
vq.run_job(mol, basis="cc-pvdz", method="ccsd", triples="[t]")
vq.run_job(mol, basis="cc-pvdz", method="ccsd", triples="A-CCSD(T)")
```

`method="ccsd(t)"` is equivalent to `method="ccsd", triples="(t)"`.
Passing `triples="none"` disables the perturbative correction even when
the method keyword is `ccsd(t)`, and the output / citation method label
follows the effective choice.

`triples="[t]"` selects the fourth-order bracket correction CCSD[T],
which is identically the original CCSD+T(CCSD) of Urban and coworkers
(1985); the spelling `triples="+T(CCSD)"` is accepted as a synonym.
The standard `(T)` is `[T]` plus the fifth-order singles-triples
coupling; the result reports both pieces separately as
`result.ccsd.e_t4` and `result.ccsd.e_t5_st`, and the `.out` block
prints the decomposition. CCSD[T] is closed-shell only; the open-shell
kernel implements the standard `(T)`.

`triples="A-CCSD(T)"` selects the closed-shell asymmetric/Lambda triples
correction. The output and citation route record the effective method as
`a-ccsd(t)` and the result exposes the Lambda residual as
`result.ccsd.lambda_residual_norm`.

## Requested-memory triples

Closed-shell DF-CCSD(T) selects its triples storage strategy from an explicit
budget instead of a fixed molecule-size threshold. The result records the
resolved mode, tile, worker count, workspace, and scratch extent:

```python
opts = vq.CCSDOptions(
    triples="(t)",
    triples_memory_mode="auto",  # auto | fast | blocked | direct | disk
    requested_memory_bytes=4 * 1024**3,
    triples_scratch_directory="/local/scratch",
)
result = vq.run_job(
    mol,
    basis="cc-pvtz",
    method="ccsd(t)",
    ccsd_options=opts,
    memory_budget_bytes=4 * 1024**3,
)
print(result.ccsd.triples_memory_mode_used)
print(result.ccsd.triples_workspace_bytes)
```

`blocked` tiles external virtual indices, `direct` regenerates the needed DF
factor contractions one scalar output at a time, and `disk` streams the DF
factors through scratch. `auto` tries in-core, blocked, and direct execution
before considering disk. The planner may reduce the
active triples worker count so the aggregate, rather than per-thread,
workspace fits. A request below the irreducible live-state floor fails before
the correction begins.

Open-shell UCCSD(T) supports `fast` and scalar `direct`; `blocked` resolves to
that scalar implementation because there is no useful intermediate virtual
tile. `disk` fails closed because the current open-shell solver still retains
the dense spin-orbital four-index integral tensor. A-CCSD(T) uses a separate
dense Lambda-triples implementation and rejects bounded triples controls. The
closed-shell FNO setup also forms dense MP2 natural-orbital arrays before the
final requested-memory triples kernel.

This does not turn the dense CCSD iteration into a reduced-scaling solver.
The UHF/ROHF spin-orbital path also retains a dense four-index integral set.
Formic-acid-dimer/cc-pVTZ, FeCl3, and similar production-size coverage is
therefore still gated on measured end-to-end bounded-memory runs.

## AutoCI-style selector

`run_job` also accepts the shared `citype=` spelling planned for the
single-reference CI/CC ladder:

```python
vq.run_job(mol, basis="sto-3g", method="ci", citype="cisd")
vq.run_job(mol, basis="cc-pvdz", method="ci", citype="ccsd(t)")
vq.run_job(mol, basis="sto-3g", method="ci", citype="cc3")
vq.run_job(mol, basis="sto-3g", method="ci", citype="ccsdt")
```

Supported values are `cisd`, `cc2`, `ccsd`, `ccsd(t)`, `cc3`, `ccsdt`, `ccd`,
`lccd`, `lccsd`, `cepa(0)`..`cepa(3)`, `qcisd`, and `qcisd(t)`.

For CISD, pass `cisd_options=vq.CISDOptions(nroots=..., max_det=...)`
to request multiple roots or raise the determinant-space guard.

## Coupled-pair variants: CCD, LCCD, LCCSD, CEPA(n)

The classic coupled-pair ladder between MP2 and CCSD is available as
variants of the same closed-shell DF kernel, either directly as
`method=` strings or through `citype=`:

```python
vq.run_job(mol, basis="cc-pvdz", method="ccd")       # CCSD without singles
vq.run_job(mol, basis="cc-pvdz", method="lccd")      # linearized CCD
vq.run_job(mol, basis="cc-pvdz", method="lccsd")     # linearized CCSD == cepa(0)
vq.run_job(mol, basis="cc-pvdz", method="cepa(1)")   # Meyer's CEPA
```

What each variant does:

| Variant | Definition |
|---|---|
| `ccd` | CCSD with the singles amplitudes frozen at zero (Čížek 1966). |
| `lccd` | Linearized CCD: the doubles residual truncated to terms linear in T2. |
| `lccsd` / `cepa(0)` | Linearized CCSD (singles + doubles, all nonlinear terms dropped). |
| `cepa(1)`, `cepa(2)`, `cepa(3)` | The linearized residual plus Meyer's pair-specific EPV shifts (coupled-electron-pair approximation). |

All variants require a closed-shell (RHF) reference; an open-shell
molecule raises `NotImplementedError` (use `ccsd` / `ccsd(t)` with the
UHF or ROHF reference instead). None of them defines a `(T)`
correction, so `triples=` is rejected. On the low-level API the same
selection is `vq.CCSDOptions(cc_variant="cepa(1)", compute_triples=False)`
with `vq.run_ccsd`.

Two implementation properties are worth knowing. First, there is no
second equation set: the linearized variants extract the linear part of
the canonical CCSD residual algebraically exactly (a polynomial-stencil
construction, machine-precision-validated against the in-repo
spin-orbital anchor). Second, the CEPA shift convention is the
closed-shell table of Wennmohs and Neese (Chem. Phys. 343, 217 (2008))
as implemented by ORCA's MDCI module, validated to sub-microhartree
agreement against out-of-process ORCA 6.1 `RI-CEPA/n` with the same
auxiliary basis. Note that CEPA(1..3) energies depend on the choice of
occupied orbitals: vibe-qc uses canonical MOs, which corresponds to
ORCA's `%mdci Localize false` (ORCA localizes internal orbitals by
default and will differ by a few tenths of a millihartree on that
account).

## QCISD and QCISD(T)

QCISD is available as `method="qcisd"` or
`run_job(method="ci", citype="qcisd")`. QCISD(T) is available as
`method="qcisd(t)"`, `method="qcisd", triples="(t)"`, or
`citype="qcisd(t)"`.

Like the coupled-pair variants, QCISD currently uses the closed-shell
RHF kernel only. Open-shell molecules raise `NotImplementedError`; use
`ccsd` / `ccsd(t)` with the UHF or ROHF reference for radicals.

Implementation detail: QCISD is built by exact two-parameter monomial
selection from the validated CCSD residual. It keeps the published
QCISD operator set, including the T2-quadratic disconnected quadruple
term, and uses the CI-like energy expression without the quadratic
`T1*T1` contribution. QCISD(T) follows the original
Pople-Head-Gordon-Raghavachari convention: on QCISD amplitudes the
triples increment is `E[T] + 2*E_ST`, not the CCSD(T) `E[T] + E_ST`
combination. This is pinned against out-of-process ORCA 6.1
`RI-QCISD(T)` with canonical orbitals and the same RI auxiliary basis.

## Low-level API

Use the low-level API when you already have a converged RHF reference or
need explicit iteration controls:

```python
import vibeqc as vq

basis = vq.BasisSet(mol, "cc-pvdz")
hf = vq.run_rhf(mol, basis, vq.RHFOptions())

opts = vq.CCSDOptions(
    triples="(t)",
    n_frozen_core=vq.chemical_core_orbital_count(mol),
    conv_tol_energy=1e-10,
    conv_tol_residual=1e-9,
)
cc = vq.run_ccsd(mol, basis, hf, opts)
print(cc.e_ccsd_t)
```

If `opts.aux_basis` is empty and `density_fit=True`, `run_ccsd`
auto-resolves the matching RI auxiliary basis for the orbital basis.
The legacy boolean `compute_triples` remains supported on
`CCSDOptions`; the `triples` property is the human-readable front end to
the same setting.

## Conventional (non-DF) integrals

For strict parity against conventional CCSD(T) in other programs, run
the coupled-cluster step on exact four-index MO integrals instead of the
RI factorisation. Pass `density_fit=False`; no auxiliary basis is needed
(or used), so this also works for basis sets without a registered RI
auxiliary:

```python
cc = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="ccsd(t)",
    frozen_core=False,
    ccsd_options=vq.CCSDOptions(density_fit=False),
)
```

The top-level `density_fit=` kwarg selects the same route and needs no
options struct. This is the form to use from a payload, which has no
`ccsd_options=` surface:

```python
cc = vq.run_job(mol, basis="cc-pvdz", method="ccsd(t)", density_fit=False)
```

`density_fit=` is a whole-job request: it picks the SCF's JK build *and*
the coupled-cluster integral route. Leaving it unset (the default) is
not the same as `density_fit=False`; unset means "no opinion", and each
step keeps its own default, which for the SCF is the four-index path and
for coupled cluster is **DF on**. Passing both `density_fit=` and a
`ccsd_options=` whose `density_fit` disagrees raises `ValueError` rather
than silently picking a winner.

> **Behaviour change.** Earlier releases wired `density_fit=` onto the
> SCF option structs only, so `run_job(method="ccsd(t)",
> density_fit=False)` ran DF-CCSD(T) and reported it as such, with no
> other indication that the request had been dropped, and
> payload-driven jobs had no way to reach the conventional route at all.
> If you passed `density_fit=False` meaning "direct SCF" on a
> coupled-cluster job, you now get the conventional CC route as well,
> which is far more expensive. **Omit the kwarg** to restore the old
> behaviour: unset already gives a direct four-index SCF and leaves the
> CC step density-fitted.

Everything downstream of the integral assembly (amplitude equations,
DIIS, the `(T)` correction) is identical to the DF path; only the source
of the `(pq|rs)` blocks changes. The `.out` block reports the route
honestly: `Algorithm = CCSD(T)` (no `DF-` prefix) and
`Density fitting = off`, with no RI auxiliary basis line. Both the
closed-shell (RHF) and the open-shell (UHF / ROHF) kernels support the
conventional route; all-electron conventional CCSD(T) and UCCSD(T) are
validated against PySCF's conventional kernels to a few nanohartree on
H2O and the OH radical in cc-pVDZ (`tests/test_ccsd_canonical_noDF.py`).

The conventional route holds the AO ERI tensor (`nbf^4` doubles) and the
correlated-window MO tensor in memory during the integral build, so it
is a small-molecule route by design. The RI error of the default DF
route is orders of magnitude below chemical accuracy (about 1e-5 to
1e-4 Ha on the correlation energy with the matched `-ri` auxiliary), so
DF remains the right default for everything except integral-exact
cross-code parity work. FNO-CCSD(T) and the DLPNO pilots remain DF-only.

## Open-shell (UHF) references

For an open-shell molecule (`multiplicity > 1`), `run_job(method="ccsd(t)")`
runs UHF and then the spin-orbital UCCSD(T) kernel automatically. No extra
flags are needed; the result is attached as `result.ccsd` with the same
`CCSDResult` fields as the closed-shell path.

```python
import vibeqc as vq

ch3 = vq.Molecule([...], charge=0, multiplicity=2)   # methyl radical
result = vq.run_job(ch3, basis="cc-pvdz", method="ccsd(t)")
print(result.ccsd.e_ccsd_t)
```

The open-shell kernel evaluates the spin-orbital coupled-cluster equations
directly on the UHF reference, so it reproduces canonical UCCSD/UCCSD(T)
(validated against PySCF `cc.UCCSD`/`UCCSD(T)` to well under a microhartree).
Use the low-level `run_uccsd(mol, basis, uhf_result, CCSDOptions(...))` when
you already have a converged UHF reference:

```python
basis = vq.BasisSet(ch3, "cc-pvdz")
uhf = vq.run_uhf(ch3, basis, vq.UHFOptions())
cc = vq.run_uccsd(ch3, basis, uhf, vq.CCSDOptions(triples="(t)"))
```

Open-shell SCF is sometimes harder to converge than closed-shell: if the
UHF reference does not converge, CCSD is skipped (no coupled-cluster numbers
are produced from an unconverged reference). For difficult radicals, pass a
`uhf_options=vq.UHFOptions(...)` with a level shift / higher `max_iter` to
`run_job`.

## Frozen natural orbitals (FNO)

Canonical CCSD(T) cost is dominated by the size of the virtual space. Frozen
natural orbitals shrink it: the dominant virtual *natural orbitals* of the MP2
density are kept and the rest are discarded before CCSD(T), recovering almost
all of the correlation energy at a fraction of the cost (DePrince and Sherrill,
*J. Chem. Theory Comput.* **9**, 2687 (2013)).

Turn it on with `fno=True` on `CCSDOptions`:

```python
import vibeqc as vq

# occupation-threshold selection (default 1e-5): keep every natural orbital
# whose MP2 occupation is at or above the threshold
vq.run_job(mol, basis="cc-pvtz", method="ccsd(t)",
           ccsd_options=vq.CCSDOptions(fno=True))

# or keep a fixed fraction of the virtual space
vq.run_job(mol, basis="cc-pvtz", method="ccsd(t)",
           ccsd_options=vq.CCSDOptions(fno=True, fno_keep_fraction=0.6))
```

The retained virtuals are semicanonicalized (the virtual Fock block is
re-diagonalized) so the perturbative `(T)` stays exact in the truncated space.
By default a **delta-MP2** correction `E_MP2[full] - E_MP2[trunc]` is added back
(`fno_delta_mp2=True`), which recovers most of the small correlation lost to
truncation. The `.out` file reports how many natural orbitals were kept and the
size of the delta-MP2 correction.

The result is an `FNOCCSDResult` with the same fields as `CCSDResult` plus
`n_virtual_kept`, `n_virtual_total`, and `delta_mp2`. The low-level entry point
is `vq.cc.run_fno_ccsd(mol, basis, rhf_result, options)`.

FNO options:

| Option | Default | Meaning |
|---|---|---|
| `fno` | `False` | enable the FNO virtual-space truncation |
| `fno_occ_threshold` | `1e-5` | keep natural orbitals with occupation `>=` this |
| `fno_keep_fraction` | `None` | if set, keep this fraction of virtuals (overrides the threshold) |
| `fno_delta_mp2` | `True` | add the `E_MP2[full] - E_MP2[trunc]` correction |

With no truncation (`fno_keep_fraction=1.0` or `fno_occ_threshold=0`) FNO-CCSD(T)
reproduces canonical CCSD(T) to machine precision, since CCSD is invariant to
virtual rotation and the semicanonicalization restores the canonical `(T)`.

## Result Fields

`CCSDResult` exposes:

- `e_hf`
- `e_ccsd_correlation`
- `e_ccsd`
- `e_t`
- `e_ccsd_t`
- `e_total`
- `n_iter`, `converged`
- `t1_norm`, `t2_norm`
- `cc_trace`

The per-iteration `cc_trace` entries are `CCSDIteration` records with
energy, energy change, residual norms, and the active DIIS subspace.

## Limits

The current canonical engine is deliberately conservative:

- RHF (closed-shell), UHF (open-shell default), and ROHF
  (`ccsd_reference="rohf"`) references are all supported. All three drive
  the same coupled-cluster equations (the closed-shell path spin-adapts
  them; the open-shell paths evaluate them in spin-orbital form).
- Dense O(N^6) pilot implementation for small molecules. Closed-shell
  uses the spin-adapted kernel; open-shell uses the spin-orbital kernel
  (same equations, validated to machine precision against the in-repo
  spin-orbital reference), which carries the usual spin-orbital factor in
  cost and memory.
- Density fitting is the default integral path; the conventional
  exact-integral route (`density_fit=False`, see above) is supported for
  cross-code parity work on small molecules. FNO and the DLPNO pilots
  require density fitting.
- Frozen occupied cores are supported with the ORCA 6.1 chemical-core default,
  the uniform `frozen_core=False` all-electron escape, or an explicit
  low-level `n_frozen_core` count. The virtual space can be truncated with
  frozen natural orbitals (`fno=True`, see above); explicit frozen-virtual
  lists are not supported.
- FNO is closed-shell (RHF reference) only so far; open-shell FNO is a
  roadmap item.
- `A-CCSD(T)` and `CCSD[T]` / `CCSD+T(CCSD)` are implemented for
  closed-shell CCSD only, not for QCISD.
- Full CCSDT is available for RHF and common-orbital ROHF references. Its
  determinant-space implementation uses exact MO integrals and is intended
  for compact benchmark spaces; it is separate from the O(N^6) CCSD kernels.
- Ground-state CC3 is available for closed-shell RHF references. Its
  determinant-space implementation uses exact MO integrals and is intended
  for compact benchmark spaces.

## Native multi-k periodic reference under development

An internal C++ reference path now connects actual finite-source periodic
Gaussian RHF, IAO/Wannier localization, selected real PAO spaces, streamed
Gaussian factors, CCSD and occupied-Fock-coupled triples. It is tested on
tiny one-, two- and three-point helium-cell Hamiltonians, including an exact
two-electron determinant check and out-of-process canonical PySCF comparisons
using the same integrals. One- and two-cell He2 fixtures also exercise
nontrivial occupied-Fock coupling. Native actual-source pair PNO generation
now implements the periodic pair-density normalization, explicit occupation
cutoffs and recanonicalization against the original virtual Fock block.
Those pair spaces now feed native coupled-MP2 and projected-CCSD references
with ragged amplitudes. CCSD uses independent singles spaces and evaluates
the complete common-frame residual through scalar amplitude accessors before
projecting its result. This is a Galerkin reference, not yet the production
DLPNO interaction/domain approximation. An explicitly opted-in native branch
now connects those pair amplitudes to union-based triple natural orbitals,
amplitude-first projected moments, occupied-Fock-coupled triples and total
energy in one HF-origin workflow. Full-rank results agree with independent
coupled equations and the common-space reference. Distinct retained ranks,
empty triple spaces, convergence failures and cancellation are tested.
Explicit frozen-core masks are tested through both complete workflows,
including the inactive-core contribution to the HF reference energy.
This is not yet the production pair-specific
periodic DLPNO-CCSD(T) implementation and is not selected by
`run_periodic_job`. Its diagnostic Python bindings enforce tiny bounds.

The reference uses explicit full Gamma-centered meshes, frozen-occupied
masks, finite AO-image and reciprocal cutoffs, and error budgets for real
integral/Fock projections. It has no infinite-cutoff or auxiliary-basis
quality certificate. A selected subsystem correlation energy is kept
separate from the HF energy per cell. Only a converged calculation spanning
all active occupied translations and the complete common virtual basis may
report the finite-torus correlation and total energies per cell. The pair
references additionally report their explicit PNO and TNO truncations; no named
production threshold preset or missing-PNO energy correction is implied.
The pair-local branch retains each stage's result, admits the complete live
storage before allocating, and exposes scalar progress callbacks. An
unconverged stage prevents later stages and any publishable total energy.
These native byte inventories are not a whole-process RSS guarantee.

An additional explicit native branch now selects occupied-orbital domains from signed
Mulliken populations and one-step PAO tails, with explicit cuts and actual
HF/basis/localization identity checks. A common-to-pair PAO embedding is
tested against original AO overlap and Fock matrices. Translation-unique
initial pair unions and density/PNO generation inside an admitted pair frame
now feed the integrated HF-origin MP2/CCSD/TNO/triples workflow. Pair occupations retain their
actual generation dimension; only the resulting PNO coefficients are exported
to the common frame. The actual local PAO-to-PNO rotation is also retained,
with its complete memory inventory and immutable numerical receipt.
The builder generates one pair geometry at a time. The pair-MP2 owner retains
each original PAO domain, real space and embedding for later contractions;
diagonal singles borrow their original embedding without an extra copy.
Truncating MP2 PNOs therefore does not truncate the parent singles domain.
Tiny full-domain results match the common-generation reference; rectangular
domain results are checked against independent projected equations. This
opt-in branch does not implement CCSD extended domains or local-auxiliary fitting.

The native Gaussian factor producer also supports exact rectangular ranges
of the certified common orbital basis. These blocks reproduce the full
factor panel, including complex density orientations, with block-sized
temporary output. They do not remove the existing retained common-factor
store or certify the accuracy of the correlation auxiliary basis.

A bounded selected-density Gram producer now integrates actual reciprocal
Gaussian panels directly into Coulomb blocks, without taking a global factor
store as input. Tiny multi-k tests cover both conjugate density orientations,
translated orbitals and frozen core against an independent all-q contraction.
An internal native consumer now generates pair PNOs directly from these
blocks in the original PAO pair space. It retains the local rotation and
projected exchange block, and transfers them into a distinctly tagged pair
space without replaying a common-factor provider. A bounded native driver
connects actual HF, Gaussian bases, localization and domain owners to these
pair spaces and the coupled-MP2 solver, without a detached integral or
amplitude array bridge. It retains every original pair geometry, including
diagonal domains needed by independent CCSD singles when doubles ranks vanish.
Source storage and numerical payloads are checked after progress callbacks;
count-first admission includes retained sibling pairs and nested workspaces.
Tiny multi-k tests compare with independent projected equations, including
frozen core, positive PNO cuts and empty ranks. All placed occupied pairs and
the common virtual export are still retained. Connecting this source to all
local-correlation consumers is still in progress; the integrated CCSD(T)
solver continues to require the original factor store.

Mixed target/source PNO factor blocks and their all-q interaction blocks
are now tested without a joint-orthonormality assumption. An optional
authenticated pair-geometry path expands the preserved local rotations
directly, and computes cross-pair overlaps with the original AO overlap.
Direct-Gram mixed factor panels and all-q Coulomb/exchange/overlap blocks
require both original pair geometries and carry their actual Gram-source
identities, never fabricated provider receipts. Tests connect native MP2
pair owners and their retained geometries directly to this block producer.
This does not yet qualify a complete direct-source CCSD residual or triples.
A separate native streaming kernel passes independent tests for the complete
bare particle-hole group, including both permutations and exchange terms.
The actual all-q Gaussian interaction producer now contracts this group
against a frozen converged CCSD snapshot, including retained pair geometry,
canonical pair reversal and empty-rank sources. A bounded verifier checks
the exact retained amplitude payload before and after consumption. The
tests compare against the original three bare CCSD intermediate seeds,
expanded and projected independently in the common space. The same native
producer also accepts validated current trial amplitudes in the exact
retained pair frames. Separately tested split-iteration kernels remove the
original bare group before adding exactly one replacement, preserving the
remaining common-frame terms. A generic replacement does not authorize
physical energies or downstream triples. A separate native all-q Gaussian
wrapper now provides that qualification and passes tiny multi-k tests
through CCSD, TNO generation and occupied-coupled triples, including
distinct pair domains and frozen core. Each contraction is admitted against
the parent solver's memory limit before allocation. The single native
selected-HF driver now also passes these tests with an explicit opt-in
physical particle-hole replacement, preserving live progress and releasing
the retained geometry before triples. The remaining dressed local
contractions still require development.
The new factor projection is not certified bitwise identical to the
original common-factor provider.

The symmetry work now includes an internal C++ Seitz-operation kernel for
one AO coefficient panel at a time. It retains fractional translations,
integer atom-image phases and optional scalar time reversal, and checks
closure of the entire regular mesh, including its shift. It supports native
pure and Cartesian shells through angular momentum six without assuming
that Cartesian rotation matrices are Euclidean-unitary. Admission counts
the borrowed source, fixed workspace, retained panel and replica inventory;
increasing the number of k points creates no all-k coefficient store.
Independent numerical tests cover nonsymmorphic phases and MgO/diamond
pob-TZVP-rev2 coefficient panels on an 8x8x8 mesh, without running SCF.
An internal native orbital-sewing validator now builds the full matrix
action separately in the frozen-core, correlated-occupied and virtual
subspaces of an immutable full-k state. It audits metric orthonormality,
reconstruction, cross-subspace leakage, stationary Fock equations and
energy intertwining. Independent phases and degenerate unitary mixing are
retained without band matching or numerical repair; degeneracy does not
permit mixing across the frozen-core boundary. Only one source/target
operation is processed at a time, with explicit work and memory caps.
Rank-reduced inputs receive retained-subspace scope only.

Neither numerical operation authenticates a physical BIPOLE source or
enables reduced-k correlation. Even a full-AO stationary-state check does
not establish Hamiltonian covariance for other densities. An authenticated
native BIPOLE source, integral-domain and density-response covariance,
group-consistent symmetry reconstruction and symmetry-representative
CCSD/triples execution remain required. Scalar time reversal is not a
magnetic symmetry certificate.

The native reciprocal-Ewald work now supplies an internal selected integral
kernel, `bipole_ewald_gram`. It uses the explicitly ordered integer cell list
to evaluate the AO-pair Fourier factors, with no implicit distance cutoff
or density-dependent screening. A single momentum-transfer channel and
bounded reciprocal panels produce a rectangular integral block. The same
block can be contracted for the long-range Coulomb and exchange operators;
there is no Gaussian fitting auxiliary or hidden k-point/spin weight.
Optional reciprocal-conjugacy and cell-inversion checks reject failed
closure without averaging. These checks are not space-group admission.

This kernel includes only the nonzero reciprocal part of the erf interaction.
Two further internal C++ kernels, `bipole_erfc_panel` and
`bipole_erfc_bloch`, evaluate selected physical short-range erfc quartets and
their complex multi-k contribution. Each request supplies an ordered slab of
exact integer image triples `(g,p,s)` for `(mu_0 nu_g | lambda_p sigma_s)`.
The fold uses `exp(i kL.g + i(q+kR).p - i kR.s)`, without an implicit k-point
weight, spin factor or symmetry repair. Integer modular phases retain exact
half- and quarter-turn characters even at large representable cell labels.
Segmented pure shells are bounded by the linked integral backend's supported
angular momentum; Cartesian shells beyond s and general contractions are
not accepted by this experimental leaf.

The raw integral slab, one complex output/compensation block, backend
workspace, input owners and other live replica allocations have composed
memory/work admission. This correctness-first implementation repeats shell
quartets for selected elements; it is not yet the scalable shell traversal.
The private Python diagnostics deliberately admit only tiny bounded panels.

An experimental native finite-source context now binds the exact AO content,
original lattice, regular k mesh/shift, ordered short-range image multiset,
ordered reciprocal/overlap cell list and common attenuation parameter. A
selected integral panel recomputes both branches from that context and adds
the explicit q=0 subtraction
`-pi/(Omega*omega^2)*conj(Bminus_left(0))*Bminus_right(0)`.
The overlap factors use the same native explicit-cell Fourier primitive at
zero reciprocal vector and negative Bloch k, not supplied SCF overlaps.
Only an explicitly selected G0-omitted convention is admitted; existing HF
exchange-singularity defaults do not change. Reciprocal block sizes and
resource caps are execution controls, not scientific source inputs.

The context retains no mutable basis or image pointers. Each call verifies
the borrowed input content before integral allocation, and its immutable
result retains the source owner and component receipts. SR and LR numerical
outputs have sequential lifetimes within one enclosing memory/work budget.
This is a finite selected-integral reference, not yet a production HF match,
space-group certificate, representative-only CCSD/(T) driver or target-cell
energy. In particular, fixed finite cutoffs need not give attenuation-
independent energies, and a probe-charge exchange correction is not silently
added to the common Coulomb integrals.

The finite source also has a bounded native density-recontraction witness.
For selected AO pairs at one target k point it streams the same integral
source into separate J and K matrices, with exactly one uniform full-mesh
`1/Nk` weight. The density is supplied in full-grid `C*C^dagger` AO order;
neither density nor output is transposed, made real, or Hermitized. General
complex matrices are admitted to test the complete linear operator, not just
time-reversal-compatible SCF densities. No spin factor, Fock construction,
probe-charge correction or physical-Hamiltonian/symmetry certificate is
implied. A restricted spin-summed-density consumer would form `H+J-K/2`.
The selected output, compensation, borrowed full-grid density and one live
child integral panel have enclosing memory/work admission. No all-q or
four-index AO tensor is stored. This is a correctness witness, not a scalable
replacement for the production BIPOLE J/K traversal.

A private long-range bridge now compares the native source with the actual
HF Fourier and contraction routines before their final Hermitization. The
HF wrappers share those unprojected contractions. An indexed exchange-cache
entry point uses native integer k-mesh transfers, avoiding the legacy
Cartesian cache's nine-decimal rounding of fractional momentum. Complete
real and imaginary density matrix units match on tiny one-, two- and
three-point meshes, including shifted and multiaxis cases, against both
native four-index integrals and independent Gaussian Fourier integrals.

This establishes the tested long-range arithmetic on matched finite cell
lists and reciprocal envelopes. The existing SCF drivers still select the
legacy Cartesian cache; the exact-address bridge is private. Short-range
quartet supports, zero/probe-charge conventions and the complete physical
HF/correlation source contract remain open. The HF cache can retain all
q/k panels, so this bridge does not provide scalable correlation storage.

A private native overlap audit addresses one prerequisite for the zero-mode
match. At a selected full-mesh k point it evaluates the source's
`B = Bminus(0;k)` from the immutable basis, lattice and ordered cell list.
It compares the supplied HF overlap separately with `conj(B)` and `B^T`,
using an explicit absolute element tolerance. With both equalities, the
source's exchange subtraction takes the form
`-pi/(Omega*omega^2*Nk) * S(k)*D(k)*S(k)` for arbitrary complex density.
The HF probe-charge term adds `xi_M * S(k)*D(k)*S(k)` separately, using the
existing full-mesh gauge constant. It is not part of the finite source.

The overlap audit verifies source inputs and admits the borrowed overlap,
one Fourier panel, workspace and replica/node inventory before numerical
evaluation. Its receipt binds the source, k point, supplied overlap,
computed Fourier factors and tolerance. An asymmetric cell list can match
`conj(B)` while failing `B^T`; the audit reports both residuals without
Hermitizing either matrix. A passing result establishes these selected-k
matrix relations only, with no HF provenance or full-operator certificate.

A private native support audit now checks one declared four-center Seitz
mapping against that immutable finite source. In the AO-transport convention
`f_destination = W*f_source + tau + ell_source`, relative cells transform
as `R' = W*R + ell_anchor - ell_center`. The audit checks the left and right
Fourier-product cell lists separately, and reanchors all three short-range
quartet image labels to the first center. It preserves exact integer labels
and repeated-image multiplicities, reports the first mismatch, and uses
constant scratch with an explicit quadratic comparison budget. Finite cells
are never reduced modulo the k mesh.

This distinguishes cell inversion from nonsymmorphic support compatibility:
the retained two-site screw witness has an inversion-closed overlap cell list
but fails the declared screw action. The mapping metadata is supplied by the
caller; a positive audit establishes only support equality for that mapping.

A second private audit derives those shifts from the existing native AO
Bloch validator for a selected **shell quartet**. It checks species-preserving
atom mapping, exact radial/angular shell matching, Cartesian isometry and
whole-mesh compatibility, including shifted grids. The geometry must use the
immutable source's exact lattice. Zero coefficient columns avoid constructing
an AO rotation matrix; the validated atom/shell maps are retained and shared
with the finite-support audit. Angular shells can mix their AO components,
so their mapped shell indices are not an AO permutation. The receipt binds
the source, geometry, shell ownership, operation, selection and tolerances.
Combined memory/work caps cover both validators before payload processing.

This geometry-bound audit still reports support closure separately: the screw
maps the atoms and radial shells correctly, but an off-diagonal pair can fail
finite-cell closure. Reciprocal support, numerical covariance, full-group
closure and matching HF still require separate admission. Neither audit nor
the small four-index inversion tests enable production symmetry reduction.

Widening a common cell window does not generally make this exact. For a
nonempty finite cell set `C` shared by every AO pair, diagonal pairs require
`W*C = C`; an off-diagonal pair then also requires
`C + ell_anchor - ell_center = C`. A finite nonempty set cannot be invariant
under a nonzero translation. Thus unequal site shifts require pair-dependent
support to obtain exact all-pair closure, even when the numerical truncation
error of a wide common window is small.

A private geometric support builder now constructs pair-dependent SR
quartet labels for an explicit new domain policy. In the actual ERI ordering
`(a,b|c,d)`, it limits both AO-product separations by `r_pair` and their
unweighted geometric midpoint separation by `r_mid`. It enumerates `g` for
the left product, `h = s-p` for the right product, and `p` for the midpoint
separation, returning the integer labels `(g,p,p+h)`. Exchange uses the
crossed ERI ordering `(a,c|b,d)`; an additional `a,b` output-pair mask would
remove allowed exchange terms. A radius of zero means zero separation,
not a disabled cutoff.

The bounded Python planner uses exact rational values of the supplied
binary64 geometry/radii and exact inverse-row bounds. It admits each
candidate box before traversal and counts the retained labels before output
allocation. Immutable labels and a geometry/policy receipt accompany the
logical storage/work census. Finite labels are never wrapped onto a BvK
mesh. Tests cover all eight ERI permutations, large site reanchoring, exact
cutoff boundaries and the molecular home-cell limit. For the two-site screw
witness, all 16 s-shell quartet supports close; selected native multi-k SR
integrals reproduce the mapped complex phases and independent Gaussian
quadrature values.

This is a source-development component, not the production HF truncation
rule. It does not authenticate supplied centers as basis/atom ownership,
bind a complete Ewald operator, or enable production symmetry reduction.
The exact boundary predicate differs from legacy floating-point masks.
The LR product support, S/T/V terms and native HF contraction traversal
still need to adopt and validate one common declared policy.

The same private geometry planner now builds a single AO-product cell list
from `|ra-rb-A R| <= r_pair`, using the exact binary64 distance policy.
Its immutable output feeds the native explicit-cell Fourier panel and a new
LR Gram entry point with independent ordered left and right cell lists.
Each list applies only to its selected product panel; the kernel does not
replace them with a shared union. It streams the existing reciprocal source
and inventories both lists, counting an identical borrowed view once.
The shared-cell API delegates to this kernel and retains the shared-list
scientific identity when both ordered contents are identical, even if their
storage differs.

For physical product domains, reversal relates two lists by `C_ba = -C_ab`.
An individual off-diagonal list need not be invariant under inversion.
The native LR diagnostics report each list's inversion closure separately;
this is distinct from a pair-reversal or space-group certificate. Selected
product domains must still be bound to their actual AO rows by the caller.
These private component APIs do not replace production HF traversal or own
a complete pair-dependent SR/LR/zero-mode source.

A native finite-product source now composes these SR/LR components and the
`q=0` subtraction under one immutable source identity. It binds the basis,
lattice, mesh, kernel controls, SR images, both ordered product cell lists,
and the allowed AO-pair interval on each axis. Selected numerical tiles must
lie within those source-owned intervals. Every evaluation rechecks the
supplied basis and both supports before allocating its numerical output.
The zero-mode Fourier factors use the corresponding left/right LR lists;
nonzero transfers allocate no zero-mode factors. Smaller selected tiles and
reciprocal streaming leave the source declaration unchanged.

This source owns the numerical declaration and its hashes, while the caller
retains the basis and label buffers. It does not certify that the labels
were generated by a particular geometric cutoff, that another quartet uses
a compatible domain, or that the complete HF operator matches. Common-list
support audits, overlap gates and J/K recontractions reject a finite-product
source: their single-list contracts cannot establish the corresponding
pair-dependent relations. A production source covering all relevant domains,
its HF contraction traversal and full group validation remain required.

A private physical-source owner now derives every AO center from the native
basis and binds one pair-distance and midpoint policy to the common lattice,
mesh and kernel declaration. It generates each ordered AO quartet's SR images
and independent LR product lists on demand, then constructs its native
finite-product source. Policy identity remains distinct even when two cutoff
choices happen to retain identical labels. The owner snapshots caller controls
and provides restartable full-quartet descriptor ranges without storing an
all-quartet table. It applies the same rule to crossed exchange quartets.

This owner is restricted to the existing small native diagnostic interface.
Explicit budgets cover the owner plus one domain and its construction peak;
retaining multiple domains, panels or workers requires additional inventory.
Descriptor-range admission covers descriptor work, while constructing and
evaluating domains requires separate admission. Tiny complete-quartet tests
compare arbitrary complex-density J/K recontractions against independent
SR/LR/zero components and the zero-mode overlap identities. These are component
validation tests, not a new production HF contraction driver. Production whole-walk
admission, HF traversal, S/T/V and nuclear terms, and full group certification
remain open before representative-only correlation can consume this policy.

The physical owner can now start a private native J/K reduction stream for
one target k point. The stream owns an immutable full-grid complex density,
requests singleton panels in ordered quartet, density-k, J/K order, and
performs the contractions and compensated accumulation in C++. It checks the
common basis/lattice/mesh/kernel, locks one source throughout each quartet,
and records one cell-list identity per ordered AO product across both roles.
Missing, reordered or incompatible panels cannot produce a finalized result;
rejected contributions leave the stream unchanged. Reuse is allowed when two
consecutive requests refer to the same integral. There is no Nao^4 receipt or
integral cache, and the arithmetic adds exactly one 1/Nk weight.

Explicit stream caps reserve the full incoming-panel numerical envelope and
the density/output/receipt state, including replicas and caller inventory.
Panel producers must perform their own native preflight before evaluation;
the stream rechecks each supplied result against its reserved envelope.
Geometry generation and any additional simultaneously retained domains or
panels require separate admission. This is a native finite density-action
component, not a production HF builder. Its policy hash records the physical
owner's declaration; the reducer does not independently regenerate geometric
supports, prove pair reversal or validate a space group. S/T/V, nuclear Ewald,
HF gauge matching and the production correlation connection remain required.

The private physical owner's `plan_jk` and `contract_jk` now connect that
stream to the physical-domain producer for one target k and a full-grid
complex density. Before scanning or copying the density or evaluating any
integrals, the admission pass visits every ordered AO quartet and preflights
every required native panel. Its shared budget counts both geometry passes,
the complete numerical work reservation, replicas and the simultaneous
stream/domain/panel storage. A late-quartet refusal therefore cannot leave a
partially evaluated J/K action.

Execution replays one domain and one singleton panel at a time and sends each
panel directly to the native reducer. The controls are snapshotted, the
producer's panel caps are narrowed to the stream reservation, and a compact
digest checks the complete domain replay before finalization. Only a finalized
native result is returned. There is no all-quartet cache, Python integral
contraction, or production symmetry certificate. This closes the private
physical producer/reducer connection; matching the production HF operator and
connecting representative-only correlation remain open.

The private `plan_jk_all` and `contract_jk_all` interfaces extend this admission
to every target in the declared k mesh. All target preflights finish before
the first density snapshot or integral. Execution uses one immutable common
density and returns a complete tuple of finalized native results in mesh
order. A late-target refusal cannot expose a partial grid. The density layout
is rechecked at the copy boundary, and native validation rejects nonfinite
values before any integral work.

The whole-mesh budget sums all target envelopes and reserves the common
density snapshot separately. This conservatively counts shared storage more
than once and includes each retained native result's own density copy; it is
a bounded diagnostic interface, not a production scaling claim. Native
per-target caps and the aggregate shared budget must both admit the work.
No all-quartet cache or Python numerical contraction is introduced. Each
result retains its existing source and payload identities and its false
Hamiltonian and symmetry certification flags.

The Bloch kernel can optionally require exact image-multiset closure under
all eight real-ERI quartet permutations and first-cell reanchoring. The
check preserves repeated-image multiplicities, rejects incomplete support
before integral evaluation, and neither adds images nor averages values.
Its additional work is explicitly capped at eight image-count-squared
comparisons, with constant scratch storage. A successful immutable result
carries an image-support receipt; disabling the check leaves it uncertified.
This receipt covers only the supplied common image multiset. It does not
certify AO-dependent masks, density screening, space-group operations,
other Ewald contributions, or a complete physical Hamiltonian.

These short-range contributions do not establish one permutation-closed
finite Hamiltonian. Separate finite J/K contractions can match while their
accepted image supports differ under quartet crossing or home-cell
reanchoring. The source must therefore retain and certify its common weighted
support, independently of the SCF density, before correlation or space-group
reduction is admitted. Neither new result claims a physical-Hamiltonian or
symmetry certificate. Background and exchange zero-mode terms, the common
finite-domain energy derivative, and a matching immutable HF source remain
integration requirements. The exact reciprocal
enumeration and finite cell support must be matched explicitly; a shared
BIPOLE label does not establish equality to an existing SCF calculation.
Memory/work admission includes both selected Fourier panels, output and
compensation, one reciprocal buffer, borrowed input owners and declared
replica/backend overhead. It is not an operating-system RSS guarantee.

Production local-CCSD contractions and factor storage,
spatial-symmetry reconstruction, scalable execution/checkpoints, and the
MgO/diamond/corundum target calculations remain development requirements.
The current restricted reference does not support antiferromagnetic CaMnO3.
No production 8x8x8 or 6x6x6 DLPNO-CCSD(T) capability is implied by these
tiny-reference tests.

## See also

* [AICCM A-namespace correlation APIs](../aiccm2026dev_a.md):
  `run_ccm_ccsd(method="aiccm2026dev-a")` is union-and-weight Γ-CCM
  CCSD(T). `run_ccm_uccsd` uses a neutral fitted-torus control and is not a
  Γ-CCM or χ-CCM result.
* [χ-CCM / aiccm2026dev-b](aiccm2026dev_b.md): periodic DLPNO-CCSD(T)
  on the finite torus (χ-CCM, 3-D).
* [Open-shell AICCM](../experimental/aiccm2026dev_open_shell.md):
  union-and-weight Γ-CCM open-shell methods, A-namespace neutral-control
  UCCSD(T), and χ-CCM open-shell methods.
* [Coupled-cluster tutorial](../tutorial/coupled_cluster_ccsd_t.md).
