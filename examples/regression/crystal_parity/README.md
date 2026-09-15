# CRYSTAL14 parity inputs

CRYSTAL14 `.d12` inputs used as bug-confirmation / parity oracles
for vibe-qc's periodic SCF chain. Treated as **external reference
programs** per CLAUDE.md § 10 - run out-of-process on compute-reference via
`vq submit -d ./<dir> ... -- bash <run-crystal.sh> <input.d12>`,
parse the resulting `<input>.out`, compare to vibe-qc.

## `mgo-rhf-pobtzvp-no-stabilisers.d12`

**Purpose:** confirm that MgO rocksalt RHF/POB-TZVP **does**
converge under default SCF settings (DIIS on, no FMIXING, no
LEVSHIFT). This is the bug-confirmation companion to vibe-qc's
gamma RHF GDF: if CRYSTAL14 converges normally here, then a
vibe-qc gamma run on the same system that oscillates / diverges
is a real algorithmic / build bug — **not** a SCF-convergence
problem to be papered over with damping.

(Per memory `feedback_periodic_scf_oscillations`: oscillation on
a simple ionic insulator like MgO is evidence of a bug, not a
convergence-protocol issue.)

The MgO conventional cubic cell, a = 4.21 Å, BASISSET POB-TZVP,
SHRINK [8, 8] (Monkhorst-Pack + Gilat secondary mesh), TOLDEE 8
(1e-8 SCF tolerance). No `DFT` block → CRYSTAL defaults to HF.
No `FMIXING` block → default mixing of zero. No `LEVSHIFT` block
→ no virtual level shift.

## Submission recipe (compute-reference)

Install the separate [vibe-queue repository](https://github.com/vibe-qc/vibe-queue)
using the [queue setup guide](../../../docs/user_guide/queue.md). The absolute
wrapper path below is on the execution host. `run-reference.sh` defaults to
`~/gitlab/vibe-queue` there; set `VIBEQC_QUEUE_CHECKOUT` in the job environment
if the queue checkout is elsewhere. It is not a directory in vibe-qc.

```bash
# from this directory, after committing + pulling on compute-reference
cd examples/regression/crystal_parity
vq submit -d . --cpus 14 --wall-time-seconds 7200 -- \
    bash /home/USER/gitlab/vibe-queue/contrib/run-crystal.sh \
    mgo-rhf-pobtzvp-no-stabilisers.d12
```

The `.out` file produced is parsed for: converged-or-not, final
total energy, SCF iteration count, the iter-by-iter trace. Diff
that against vibe-qc's own iteration trace on the same system.
