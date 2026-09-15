"""Converger profile — the single source of truth for the
"same converger in ORCA and vibe-qc" goal of this benchmark.

The two codes can never produce a bit-identical SCF trajectory (integral
precision, initial-guess details, internal step orderings all differ),
but the *converger family* and the *convergence criteria* can be made
to match. The values below are the ones quoted by both the vibe-qc
input scripts and the ORCA ``! ...`` / ``%scf`` blocks in this
benchmark.

Rationale per knob:

* **Accelerator: plain DIIS** (Pulay 1980, 1982). vibe-qc's default
  is the EDIIS+DIIS hybrid (Garza-Scuseria 2012); ORCA's default
  stack augments DIIS with SOSCF (Neese 2000) and the trust-region
  Augmented-Hessian (TRAH). We turn both extras off so both codes
  run plain Pulay DIIS.

* **Energy tolerance: 1e-8 Ha**. ORCA ``SCFCONV8`` ≡ TolE 1e-8 +
  TolErr 1e-7 + TolMaxP 1e-7 + TolRMSP 1e-8. vibe-qc's matching
  ``conv_tol_energy = 1e-8`` + ``conv_tol_grad = 1e-6`` covers the
  same regime — tight enough that MP2 differences are not masked by
  unconverged density.

* **Max iter: 200**. Generous enough for the cc-pVTZ dimers in S22
  with plain DIIS (no EDIIS bootstrap).

* **DIIS subspace = 8**. Both code defaults; matched explicitly.

* **Initial guess**: vibe-qc SAD; ORCA PModel. Both are
  atomic-density-based and behave similarly; SAD/PModel differ in
  fine details but not in the dominant convergence behaviour. We
  document the choice rather than forcing HCORE (which doesn't
  converge cleanly on cc-pVTZ for the bigger S22 dimers).

* **Linear-dependence threshold: 1e-7**. Both code defaults.
"""

from __future__ import annotations

CONV_TOL_ENERGY = 1.0e-8       # Hartree, |E[k] - E[k-1]|
CONV_TOL_GRAD = 1.0e-6         # Frobenius norm of [F,P]
MAX_ITER = 200
DIIS_SUBSPACE = 8
LINEAR_DEP_THRESHOLD = 1.0e-7

# vibe-qc SCFAccelerator enum string — see python/vibeqc/__init__.py.
VIBEQC_ACCELERATOR = "DIIS"

# ORCA simple-input keywords that pin "plain DIIS" behaviour:
#
#   NoSOSCF   — disable second-order step (Neese 2000)
#   NoTRAH    — disable trust-region augmented Hessian
#   SCFCONV8  — 1e-8 energy, 1e-7 max-density, 1e-8 rms-density
#
# We do NOT add ``KDIIS off`` — KDIIS is opt-in and stays off by default.
ORCA_SIMPLE_TOLS = "NoSOSCF NoTRAH SCFCONV8 NoFrozenCore NoAutoStart"
# NoFrozenCore - match the generated vibe-qc decks' explicit
#                n_frozen_core=0 / frozen_core=False historical protocol.
# NoAutoStart  — don't read a sibling .gbw orbital file. Without
#                this, repeated ORCA runs in the same directory
#                start from the previous calculation's orbitals
#                and converge in 2-3 iter, masking the actual
#                from-scratch SCF behaviour.
ORCA_SCF_BLOCK = (
    "%scf\n"
    f"  MaxIter {MAX_ITER}\n"
    f"  DIISMaxEq {DIIS_SUBSPACE}\n"
    "  Guess PModel\n"
    "end\n"
    # Block-form FC override. ORCA 6.1's ``! NoFrozenCore`` simple-
    # input flag is honoured for closed-shell MP2 and for some
    # open-shell topologies (OH, O2), but NOT for canonical UMP2 on
    # CH3 — and presumably other polyatomic open-shell cases.
    # The %method block always wins and pins all-electron
    # correlation across every variant, matching the explicit all-electron
    # vibe-qc benchmark recipe. Verified by inspection
    # of the ``Freezing NCore=2 ...`` ORCA banner — absent under
    # this block, present without it on CH3 UMP2.
    "%method\n"
    "  FrozenCore FC_NONE\n"
    "end\n"
)


def vibeqc_options_snippet(opts_var: str = "opts") -> str:
    """The shared converger settings as a vibe-qc options-object snippet
    that the per-variant generator embeds verbatim into an input script.

    ``opts_var`` is the variable name the snippet assumes (e.g. for
    RHFOptions / UHFOptions / RKSOptions / UKSOptions instances).
    """
    return (
        f"{opts_var}.conv_tol_energy = {CONV_TOL_ENERGY:.1e}\n"
        f"{opts_var}.conv_tol_grad = {CONV_TOL_GRAD:.1e}\n"
        f"{opts_var}.max_iter = {MAX_ITER}\n"
        f"{opts_var}.scf_accelerator = vq.SCFAccelerator.DIIS\n"
        f"{opts_var}.diis_subspace_size = {DIIS_SUBSPACE}\n"
        f"{opts_var}.damping = 0.0\n"
        f"{opts_var}.linear_dep_threshold = {LINEAR_DEP_THRESHOLD:.1e}\n"
    )
