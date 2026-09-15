# Local correlation with DLPNO

Canonical MP2 and coupled cluster scale steeply with system size
because they correlate every occupied orbital with every other one in
a delocalised basis. Domain-based local pair-natural-orbital (DLPNO)
methods exploit the short-sightedness of dynamic correlation instead:
occupied orbitals are localised, each orbital *pair* gets its own
compact virtual space (the pair natural orbitals), and pairs that are
far apart are estimated rather than solved. The art is that all of
this is controlled by a handful of thresholds, and as the thresholds
go to zero, the canonical answer comes back *exactly*.

This tutorial runs DLPNO-MP2 on water, shows the threshold
convergence against canonical RI-MP2, and finishes with the
reduced-scaling DLPNO-CCSD solver and the local DLPNO-CCSD(T). The
matching ready-to-run scripts are
[`input-h2o-dlpno-mp2.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-mp2.py),
[`input-h2o-dlpno-ccsd.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-ccsd.py),
and
[`input-h2o-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-ccsd-t.py).

All four public MP2/CCSD DLPNO routes, restricted and unrestricted, now use
the same project convention: ORCA 6.1 chemical-core counts and the supported
subset of Liakos NormalPNO. Pass `frozen_core=False` for the uniform
all-electron escape, and use `dlpno_thresholds="loose"`, `"normal"`, or
`"tight"` for a named published preset. The local open-shell CCSD route uses
the same three Liakos coordinates for its spin-orbital pairs. It does not yet
claim Saitow 2017's additional `TScaleTCutPairs=0.33`, which assumes a single
ROHF/QRO-like spatial-orbital set rather than independent UHF alpha and beta
spaces.

## One-call DLPNO-MP2

The whole method is reachable through `run_job` with
`method="dlpno-mp2"`. This runs DLPNO-MP2 on water in a cc-pVDZ basis and
pulls the correlation energy off the result object:

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])

result = vq.run_job(mol, basis="cc-pvdz", method="dlpno-mp2", output="h2o_dlpno")
r = result.dlpno_mp2
print(r.e_corr, result.energy_total)
```

The `.out` file carries the decomposed solver block, iterated pair
energies, the PNO-truncation correction, and the distant-pair
estimate:

```{note}
The numerical block below is retained output from the pre-#140/#448
all-electron MP2 recipe (`tcut_pairs=1e-6`, `tcut_pno=1e-8`,
`tcut_mkn=1e-3`). It is not current-default output.
```

```
  DLPNO-MP2 (Pinski 2015; RI: cc-pvdz-ri)
  ------------------------------------------------------------------------------
  E(RHF reference)       =   -76.0243138804 Ha
  pairs kept / screened  = 15 / 0   (frozen core: 0)
  avg PNOs per pair      =           14.9
  E(iterated pairs)      =    -0.2004258190 Ha
  E(PNO truncation corr) =    -0.0000037342 Ha
  E(distant-pair est.)   =     0.0000000000 Ha
  E(DLPNO-MP2 corr)      =    -0.2004295532 Ha
  E(DLPNO-MP2 total)     =   -76.2247434336 Ha
```

Under that historical all-electron recipe, water had 15 surviving orbital
pairs (5 localised valence+core occupieds → 5 diagonal + 10 off-diagonal
pairs) and kept ~15 of the 19 virtuals per pair at `TCutPNO = 1e-8`. The
current chemical-core/NormalPNO result has different counts and must be read
from its own `.out` rather than inferred from this archived block.

## The experiment every DLPNO user should run once

How much correlation energy does the PNO truncation cost, and does
the method really converge to the canonical answer? Sweep `TCutPNO`
against canonical RI-MP2 with the *same* fitting basis:

```python
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.mp2 import DLPNOMP2Options, run_dlpno_mp2

basis = vq.BasisSet(mol, "cc-pvdz")
hf = vq.run_rhf(mol, basis)

mp2_opts = vq.MP2Options()
mp2_opts.density_fit = True
mp2_opts.aux_basis = "cc-pvdz-ri"
canonical = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="mp2",
    frozen_core=False,
    mp2_options=mp2_opts,
    output="h2o_canonical_all_electron",
).mp2

df = DensityFitting(basis, vq.BasisSet(mol, "cc-pvdz-ri"),
                    aux_basis_name="cc-pvdz-ri")

for label, opts in (
    ("1e-07", DLPNOMP2Options(n_frozen=0, tcut_pairs=1e-6,
                               tcut_pairs_weak=1e-4, tcut_pno=1e-7,
                               tcut_pno_weak=1e-6, tcut_mkn=1e-3)),
    ("1e-08", DLPNOMP2Options(n_frozen=0, tcut_pairs=1e-6,
                               tcut_pairs_weak=1e-4, tcut_pno=1e-8,
                               tcut_pno_weak=1e-7, tcut_mkn=1e-3)),
    ("1e-09", DLPNOMP2Options(n_frozen=0, tcut_pairs=1e-6,
                               tcut_pairs_weak=1e-4, tcut_pno=1e-9,
                               tcut_pno_weak=1e-8, tcut_mkn=1e-3)),
    ("0*",    DLPNOMP2Options(n_frozen=0, tcut_pairs=0.0,
                               tcut_pairs_weak=0.0, tcut_pno=0.0,
                               tcut_pno_weak=0.0, tcut_mkn=0.0)),
):
    r = run_dlpno_mp2(mol, basis, hf, df, opts)
    avg = sum(r.pno_per_pair.values()) / len(r.pno_per_pair)
    print(f"{label:>6s}  {r.e_corr:14.8f}  "
          f"{r.e_corr / canonical.e_correlation:9.4%}  {avg:5.1f}")
```

Output:

This table is deliberately retained as **pre-#140/#448 all-electron,
explicit-threshold evidence**. The code above pins that convention on both
sides; none of these rows is a current-default energy.

```
 1e-07     -0.20028030   99.7706%   14.0
 1e-08     -0.20042955   99.8450%   14.9
 1e-09     -0.20053971   99.8998%   15.7
    0*     -0.20074079  100.0000%   19.0
```

Three things to read off this table:

* **The error is controlled by one knob.** Tightening `TCutPNO` by 10×
  buys roughly a factor-of-2 smaller truncation error at the cost of
  ~1 extra PNO per pair.
* **The last row is the exactness limit**, with pair, PNO, weak-pair, and
  domain screens all zero (hence the `*`): DLPNO-MP2 reproduces
  canonical RI-MP2 to better than 1 µHa. This identity is asserted
  permanently in `tests/test_dlpno_mp2.py`, on canonical *and*
  Boys-localised occupieds.
* **Recovery percentages are measured, not advertised.** Remaining DLPNO
  accuracy and reduced-scaling work is tracked in the canonical
  [GitLab issue list](https://github.com/vibe-qc/vibe-qc/issues).

## Frozen core and distant-pair screening

The default frozen-core count follows ORCA 6.1 Table 2.69. ORCA tabulates
frozen electrons; vibe-qc converts that closed-shell count to spatial
orbitals, sums it over the atoms, and reports the resolved orbital count in
the method block. The same high-level switch works for canonical/DLPNO and
restricted/unrestricted MP2/CCSD routes:

```python
default = vq.run_job(
    mol, basis="cc-pvdz", method="dlpno-mp2",
    dlpno_thresholds="normal",
    output="h2o_fc",
)
all_electron = vq.run_job(
    mol, basis="cc-pvdz", method="dlpno-mp2",
    frozen_core=False,       # ORCA NoFrozenCore
    dlpno_thresholds="tight",
    output="h2o_all_electron_tight",
)
```

Frozen-core DLPNO-MP2 is validated against a canonical frozen-core
reference to ≤ 1 µHa. Pair screening only acts on pairs whose
localised-orbital centroids are ≥ 8 bohr apart (nothing in a water
monomer qualifies); the screened pairs contribute through the
`e_distant` component instead of silently vanishing.

At the low level, `n_frozen` remains an explicit spatial-orbital count. For
example,
`options_from_dlpno_thresholds(LocalCCSDOptions, "tight")` returns a
`LocalCCSDOptions` object ready for `dlpno_ccsd_options=...`. Only thresholds
implemented by that solver are applied and reported; an omitted field is
unsupported, not silently zero.

## DLPNO-CCSD: the reduced-scaling local solver

The same PNO machinery carries the coupled-cluster ansatz.
`method="dlpno-ccsd"` runs the **reduced-scaling local solver**: each
pair's amplitudes remain in its PNO basis, while the default residual is
contracted in Riplinger and Neese's atom-based extended PAO domain and then
projected back. No full-system amplitude tensor is formed. It is FCI-anchored;
in the full-domain limit it reproduces canonical closed-shell CCSD bit-for-bit
(≤ 1 µHa; on H₂ that is FCI). The project default is the supported subset of
NormalPNO with `residual_domain="extended"`. The retained recovery percentages
below describe their explicitly labelled historical recipes rather than a
newly measured current-default value.

```python
result = vq.run_job(mol, basis="cc-pvdz", method="dlpno-ccsd",
                    output="h2o_cc")
cc = result.dlpno_ccsd
print(f"E_corr(CCSD) = {cc.e_corr:.8f}  total = {result.energy_total:.8f}")
```

Output:

```
E_corr(CCSD) = -0.21017221  total = -76.23448609
```

That exact line is retained pre-#140/#448 all-electron/old-threshold output,
not a new-default energy. Current runs use chemical cores and NormalPNO and
report their own resolved count and supported thresholds.

That archived run retained 15 pairs and about 14 PNOs per pair from the
cc-pVDZ virtual space. For a current custom run, pass
`dlpno_ccsd_options=LocalCCSDOptions(tcut_pno=1e-9)` (from
`vibeqc.dlpno.ccsd_local_solver`) to tighten the PNO truncation toward
the canonical limit, or tune `coupling_radius` (bohr; **default 12**),
how far each pair's occupied coupling set reaches, the lever that bounds
the occupied coupling. It is exact at a radius wider than the molecule
and a controlled approximation below the PNO truncation error once
fragments separate: in a chain of H₂ molecules the average
coupled-occupied count saturates while the system grows, instead of
tracking it. On a water this small everything is within the default
radius, so it is a no-op; the radius earns its keep on extended systems
(set `coupling_radius=0` for exact full coupling).

DLPNO fits the correlation integrals against an RI auxiliary basis, which
the runner auto-resolves from the orbital basis (the `RI:` line in the
log). Correlation-consistent (`cc-pVnZ`) and `def2` orbital bases have a
bundled default RI aux; **minimal / Pople bases** (`sto-3g`, `6-31g*`, …)
do not, and a DLPNO run on one raises
`NotImplementedError: No default ri-aux registered …`. Supply an aux
explicitly through the option object to override:
`dlpno_ccsd_options=LocalCCSDOptions(aux_basis="cc-pvdz-ri")` (closed
shell) or `DLPNOUCCSDPilotOptions(aux_basis="cc-pvdz-ri")` (open shell).
DLPNO on a minimal orbital basis is rarely physically meaningful, but the
override keeps it available for smoke tests and parity checks.

### CCSD(T): DLPNO-(T1)

`method="dlpno-ccsd(t)"` runs the same local solver and then the
rotated-occupied **(T1) compatibility route** on the converged amplitudes
(`vibeqc.dlpno.triples_local`). The perturbative triples are evaluated per
occupied triple in a TNO domain after diagonalising and rotating the occupied
space. This removes the (T0) semicanonical approximation, but finite PNO/TNO
domains remain a truncation. The open-shell local solver separately exposes
`triples_mode="t1-iterative"`, a spin-orbital generalization of the 2018 Guo
et al. closed-shell Eq. (2); Guo et al., J. Chem. Phys. 152, 024116 (2020) is
the dedicated open-shell method-family source. Unlike the rotated route, it
can preserve a screened local triple list.
At full domains it reproduces canonical CCSD(T) to machine precision (the
(T1) parity ratchet). Set
``triples_mode="local"`` for the older DLPNO-(T0) (diagonal localised Fock,
~0.1 kcal/mol looser).  The (T) vanishes identically for two-electron
systems.

```python
result = vq.run_job(mol, basis="cc-pvdz", method="dlpno-ccsd(t)",
                    output="h2o_cct")
cc = result.dlpno_ccsd
print(f"E_corr(CCSD) = {cc.e_corr:.8f}  (T) = {cc.e_t:.8f}  "
      f"total = {result.energy_total:.8f}")
# E_corr(CCSD) = -0.21017219  (T) = -0.00267814  total = -76.23716421
```

The commented numerical line is likewise retained pre-#140/#448 evidence;
the unqualified call above now follows the project defaults and should not be
expected to reproduce it.

The (T) uses a spatial closed-shell kernel (no spin-orbital redundancy).
The O(N⁶) correctness pilot is still available by passing a
`DLPNOCCSDPilotOptions`, useful as an independent benchmark anchor.

## Is DLPNO-CCSD(T) accurate enough?

The exactness limit above answers the question in principle, but production
runs use truncated domains. The table below is retained, explicitly
all-electron pre-#140/#448 evidence, not a measurement of the current
chemical-core/NormalPNO default. The checked-in
[`benchmark-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/benchmark-dlpno-ccsd-t.py)
reproduces this precise def2-SVP recipe: `tcut_pno=1e-7`,
`tcut_pairs=1e-4`, `tcut_mkn=0`, `residual_domain="pair"`, and
DLPNO-(T1).

| Molecule | Canonical CCSD(T) (Ha) | DLPNO error (kcal/mol) | Recovery |
|---|---:|---:|---:|
| H₂O  | -76.177021  | +0.083 | 99.95 % |
| NH₃  | -56.357944  | -0.123 | 100.01 % |
| CH₄  | -40.360454  | -0.670 | 100.54 % |
| HF   | -100.141044 | +0.128 | 99.95 % |
| CO   | -112.955086 | +0.006 | 100.01 % |
| N₂   | -109.180282 | -0.727 | 100.33 % |
| H₂CO | -114.125245 | -0.843 | 100.30 % |

Its mean absolute error was **0.37 kcal/mol**. A separate historical run
tightened only the PNO cutoff to `tcut_pno=1e-8` and gave **0.17 kcal/mol**,
alongside a reported ORCA 6.1 value of 0.16 on the same set. The exact
generator for that second snapshot is retained at
[Archived `3e69f1960`](https://vibe-qc.com/docs/);
its results were:

| Molecule | Canonical CCSD(T) (Ha) | DLPNO error (kcal/mol) | Recovery |
|---|---:|---:|---:|
| H₂O  | -76.177021  | +0.135 | 99.90 % |
| NH₃  | -56.357944  | +0.046 | 99.99 % |
| CH₄  | -40.360454  | -0.073 | 100.06 % |
| HF   | -100.141044 | +0.101 | 99.93 % |
| CO   | -112.955086 | +0.288 | 99.86 % |
| N₂   | -109.180282 | +0.401 | 99.82 % |
| H₂CO | -114.125245 | +0.117 | 99.94 % |

Both tables are recipe-specific historical evidence. The first is generated
by the checked-in script; the second belongs to the exact historical script
revision linked above. Neither table validates the new default.

### Legacy fast-scan defaults and the high-accuracy recipe

An earlier pre-#140/#448 fast-scan paired `tcut_pno=1e-7` with DLPNO-(T0)
(MAE 0.30); that distinct historical generator is retained at
[Archived `49492b718`](https://vibe-qc.com/docs/).
The looser PNO caused CCSD over-recovery that accidentally cancelled the
(T0) semicanonical error. The rotated-occupied (T1) removes that error while
retaining TNO-domain truncation, and `tcut_pno=1e-8` removes
the CCSD over-recovery -- the combined historical recipe gave the reported
MAE on this benchmark. Finite domains remain an approximation.

The historical all-electron recipe remains reproducible explicitly:
```python
LocalCCSDOptions(n_frozen=0, tcut_pno=1e-7, tcut_pairs=1e-4,
                 tcut_mkn=0.0, residual_domain="pair",
                 triples_mode="t1", compute_triples=True)
```
For an even tighter, O(N⁷) exact-(T) oracle:
```python
LocalCCSDOptions(n_frozen=0, tcut_pno=1e-8, tcut_pairs=1e-4,
                 tcut_mkn=0.0, residual_domain="pair",
                 triples_mode="exact", compute_triples=True)
```

Re-run the checked-in script for the first table. Use the two immutable
historical revisions for the tightened and (T0) snapshots; their protocols
must not be inferred from today's defaults.

### The point is reduced scaling

Accuracy is only half the story; DLPNO exists because its cost grows
far more slowly than canonical CCSD(T)'s O(N⁷). The companion reach
demonstration in the benchmark (water clusters, same basis) makes the
contrast concrete: canonical CCSD(T) becomes intractable past roughly
three to four waters, while DLPNO-CCSD(T) carries on. See the reach
walkthrough in
[`benchmark-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/benchmark-dlpno-ccsd-t.py)
and the remaining DLPNO entries in
the canonical [GitLab issue list](https://github.com/vibe-qc/vibe-qc/issues).

## When to reach for DLPNO

* **DLPNO-MP2**, whenever you would run RI-MP2 and the system is big enough
  that you care. It supports closed-shell RHF and open-shell UHF references;
  energies only (no gradients yet). Historical recovery percentages earlier
  on this page belong to their explicit all-electron threshold sweep, not the
  current default.
* **DLPNO-CCSD**, the reduced-scaling local solver: coupled-cluster
  quality with PNO truncation control, full-domain == canonical CCSD.
  `coupling_radius` defaults to 12 bohr, a no-op on compact molecules,
  bounding the occupied coupling on extended systems (set 0 for the exact
  full-coupling reference). The default extended-domain contraction runs in
  the compiled residual kernel once per distinct (coupling set, extended
  domain) group, so a compact molecule pays one canonical-shaped residual per
  iteration rather than one per pair; the explicit legacy
  `residual_domain="pair"` mode uses the same kernel in its pair-PNO basis.
  `tcut_pno` cuts pair-density occupation numbers, and which density is used
  is selectable with `pno_norm` (`"legacy"` by default, `"mp2"` for Riplinger
  and Neese's published Eq. 23, `"iepa"` for the pre-2013 LPNO convention);
  see the user guide. The solver carries a generous, overridable `max_nbf`
  guard.
* **DLPNO-CCSD(T)**, the local DLPNO-(T) on the converged local amplitudes
  (per-triple TNO domains), exact == canonical CCSD(T) at full domains. Both
  closed- and open-shell public routes use their reduced-scaling local solver;
  the opt-in open-shell ``triples_mode="t1-iterative"`` correction is an
  experimental, capped dense spin-orbital generalization and is not itself a
  reduced-scaling triples phase.
  `DLPNOCCSDPilotOptions` / `DLPNOUCCSDPilotOptions` explicitly select the
  dense, capped correctness pilots for benchmark anchors.

Every `dlpno-*` job emits its method papers (Pinski 2015, Riplinger
2013, Foster-Boys 1960, …) into the `.references` / `.bibtex` outputs
automatically.

## See also

* [DLPNO methods user guide](../user_guide/dlpno_mp2.md), full
  threshold table, result-object reference, limitations.
* [MP2 and double hybrids](../user_guide/mp2_and_double_hybrids.md), 
  the canonical methods DLPNO converges to.
* Tutorial [38, RI-MP2 auxiliary-basis verification](ri_mp2_aux_verification.md)
  - the RI fitting error underneath everything here.
