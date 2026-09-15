#!/usr/bin/env python
"""AICCM-2026 test-set driver — one (system, route) case per invocation.

Runs one benchmark case and writes a small JSON result for the comparison
harness. Designed to be submitted to ``vq``:

    vq submit compute-study  -d studies/aiccm-2026/ -- python run_case.py mgo aiccm-ks
    vq submit compute-small  -d studies/aiccm-2026/ -- python run_case.py lih-rocksalt gdf
    vq submit local -d studies/aiccm-2026/ -- python run_case.py h-chain aiccm-hf

Routes — a mixed harness containing Γ-CCM construction routes, separately
declared neutral fitted-torus controls, and ordinary periodic references:

    aiccm-hf       HF, four-center cyclic cluster (no RI)
    aiccm-hf-direct HF, neutral-torus (k-free) real-Gamma SCF control with the
                   derived exchange-q0 seam. This is a representation control,
                   not union-and-weight Γ-CCM evidence (periodic/ccm/direct.py)
    aiccm-ks       KS,  four-center cyclic cluster (no RI; --functional, default pbe)
    aiccm-ri       HF neutral fitted-torus Bloch/GDF representation control
    aiccm-ks-ri    KS with an RI (density-fitted) Coulomb (--functional)
    aiccm-rijcosx  HF with RIJCOSX (WSSC RI-J + chain-of-spheres K)
    aiccm-mp2      MP2 correlation (UMP2 for open-shell), on the CCM MO integrals
                   -- the BARE four-center reference. Dense, so 1-D/2-D only;
                   kept unchanged under its own name so published bare numbers
                   stay reproducible (Task #12 option (c), maintainer 2026-09-05)
    aiccm-ccsd     CCSD(T) correlation (closed-shell), on the CCM MO integrals
                   -- likewise the bare four-center reference
    aiccm-ri-mp2   MP2 (UMP2 open-shell) on the NEUTRAL fitted-torus RI
                   reference. Reaches 3-D, and refuses dim<3: the two are
                   COMPLEMENTARY, not a replacement (the bare route is dense
                   so 1-D/2-D only; the RI cderi needs a real 3-D reciprocal
                   mesh -- a transverse-collapsed one gives a sheet term
                   proportional to 1/V, not 1/r). Use the wire kernel
                   (periodic.ccm.lowd_scf.run_ccm_rhf_wire) for 1-D. A
                   declared neutral control, NOT a Gamma-CCM construction
                   result
    aiccm-ri-ccsd  CCSD on the same neutral RI reference (closed-shell)
    aiccm-viz      HF + vibe-view visualization of the crystalline orbitals AND
                   the localized (Wannier) orbitals — _canonical.qvf + _wannier.qvf
                   + density / CO / Wannier .cube + .molden (paper / cover figures)
    aiccm-localize demo: Wannier localization (Task 1) — JSON reports the
                   unitary-invariance gate (density/energy preserved) + centers/spreads
    aiccm-symmetry demo: space-group symmetry (Task 3) — JSON reports the space
                   group, cluster-invariant order, S/T invariance, pair reduction
    aiccm-pao      demo: DLPNO projected atomic orbitals (Task 2) — JSON reports
                   n_pao (= virtual dim) + S-orthogonality to the occupied space
    bipole         vibe-qc periodic reference, jk_method="bipole" (k-mesh)
    gdf            vibe-qc periodic reference, jk_method="gdf"    (k-mesh)

k-point sampling: the union-and-weight Γ-CCM routes use a Γ-supercell whose
cluster size `nrep` replaces BZ sampling. `aiccm-ri`/`aiccm-ks-ri` instead
evaluate a separately declared neutral fitted-torus control over the equivalent
multi-k mesh = `nrep` on the unit cell. They are representation controls, not
an alternative implementation of Γ-CCM. The `bipole`/`gdf` reference routes are
ordinary k-sampled periodic calculations.

RELIABILITY CAVEAT (low-D neutral controls): every route in
`NEEDS_3D_COULOMB`, including multi-k GDF, neutral real-Γ, and the periodic
`gdf`/`bipole` references, now fails closed for `dim < 3`. Their shared
dimension-collapsed reciprocal mesh pins transverse components to zero and is
not a wire or slab Coulomb kernel. Historical pre-guard runs over-bound
vacuum-padded low-dimensional cells and changed with vacuum or lattice-sum
cutoff; those energies are retracted. For a union-and-weight low-D study use
the four-center Γ-CCM routes (`aiccm-hf`/`aiccm-ks`). The separately derived
mixed-boundary wire route exists only for its stated `dim=1` scope; no shared
neutral wire/slab control is currently defined. Do not assume that a route gap
cancels in an energy difference or leaves correlation unaffected. Both sides
of a difference and every correlated reference must use the same specified
operator, and any cancellation requires an explicit convergence test.

The CRYSTAL23 reference for each system is the ``.d12`` named in ``testset.py``
(``crystal_ref``) inside ``~/gitlab/qc-input-library``; run it with CRYSTAL23.

Usage: python run_case.py <system> <route> [--basis B] [--functional F]
                          [--nrep i j k] [--kmesh i j k] [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import probe_host
import testset
import vibeqc as vq
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd, run_ccm_ri_ccsd
from vibeqc.periodic.ccm.dft import run_ccm_rks, run_ccm_uks
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2, run_ccm_ri_mp2
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rhf_rijcosx, run_ccm_rks_gdf
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable
from vibeqc.periodic.ccm.uhf import run_ccm_uhf
from vibeqc.periodic.ccm.ump2 import run_ccm_ump2

METHOD = "aiccm2026dev-a"

ROUTES = [
    "aiccm-hf",
    "aiccm-hf-direct",
    "aiccm-ks",
    "aiccm-ri",
    "aiccm-ks-ri",
    "aiccm-rijcosx",
    "aiccm-ri-cosx",
    "aiccm-ks-ri-cosx",
    "aiccm-mp2",
    "aiccm-ccsd",
    "aiccm-ri-mp2",
    "aiccm-ri-ccsd",
    "aiccm-dlpno-mp2",
    "aiccm-dlpno-ccsd",
    "aiccm-uccsd",
    "aiccm-properties",
    "aiccm-viz",
    "aiccm-localize",
    "aiccm-symmetry",
    "aiccm-pao",
    "bipole",
    "gdf",
]

NEEDS_3D_COULOMB = testset.NEEDS_3D_COULOMB   # single source of truth

#: Per-job path exported by ``run.sh`` inside a fresh private directory.
_attestation_path = os.environ.get("AICCM_PROBE_ATTESTATION")
_ATTESTATION = Path(_attestation_path) if _attestation_path else None


def _core_built_at() -> str:
    """Return the current core-path mtime as a human-readable audit field."""

    try:
        from vibeqc import _vibeqc_core

        core = Path(getattr(_vibeqc_core, "__file__", "") or "")
        if core.is_file():
            return datetime.datetime.fromtimestamp(core.stat().st_mtime).isoformat(
                timespec="seconds"
            )
    except Exception:  # noqa: BLE001 -- audit metadata must fail closed
        pass
    return "unknown"


def _provenance() -> dict:
    """Record process-bound evidence for the producer about to run.

    The cached v2 probe is preflight evidence.  This process independently
    checks the copied producer payload, the shifted-mesh native kernel, and the
    current deployment before any calculation starts.  If only the core-path
    digest changed since preflight, the full v2 checks are repeated here.
    """
    import platform

    commit, source_clean = probe_host.imported_source_state()
    prov = {
        "host": (platform.node() or "unknown").split(".")[0],
        "vibeqc_version": str(getattr(vq, "__version__", "unknown")),
        "vibeqc_commit": commit,
        "source_clean": source_clean,
        "core_built": _core_built_at(),
        "core_build_id": probe_host.core_build_id(),
        "probe_version": "not-recorded",
        "probe_script_id": "unknown",
        "probe_passed": False,
        "probe_attestation_id": "unknown",
        "probe_attestation": None,
        "producer_payload_id": "unknown",
    }
    if os.environ.get("AICCM_SKIP_PROBE") != "1":
        attestation = probe_host.producer_attestation(
            _ATTESTATION,
            _producer_payload(),
        )
        if attestation is not None:
            prov.update(
                host=attestation["host"],
                vibeqc_version=attestation["vibeqc_version"],
                vibeqc_commit=attestation["source_commit"],
                source_clean=attestation["source_clean"],
                core_build_id=attestation["core_build_id"],
                core_built=_core_built_at(),
            )
            prov["probe_version"] = probe_host.PRODUCER_ATTESTATION_VERSION
            prov["probe_script_id"] = attestation["attestor_script_id"]
            prov["probe_passed"] = True
            prov["probe_attestation_id"] = probe_host.probe_attestation_id(
                attestation
            )
            prov["probe_attestation"] = attestation
            prov["producer_payload_id"] = attestation["producer_payload_id"]
    return prov


def _producer_payload() -> dict[str, object]:
    """Identify every executable input copied by the directory launcher."""

    here = Path(__file__).resolve().parent
    return {
        "version": probe_host.PRODUCER_PAYLOAD_VERSION,
        "stream": METHOD,
        "files": {
            "producer": probe_host.file_sha256(__file__),
            "probe_host": probe_host.file_sha256(probe_host.__file__),
            "testset": probe_host.file_sha256(testset.__file__),
            "launcher": probe_host.file_sha256(here / "run.sh"),
        },
    }


def _attestation_required() -> bool:
    return _ATTESTATION is not None and os.environ.get("AICCM_SKIP_PROBE") != "1"


def _require_attested_launch(provenance: dict[str, object]) -> None:
    """Refuse a normal directory launch before it computes untrusted numbers."""

    if _attestation_required() and provenance.get("probe_passed") is not True:
        raise SystemExit(3)


def _finalize_provenance(provenance: dict[str, object]) -> bool:
    """Bind the result to an identity stable through result assembly."""

    finalized = probe_host.finalize_producer_attestation(
        provenance.get("probe_attestation"),
        _producer_payload(),
    )
    provenance["core_built"] = _core_built_at()
    if finalized is None:
        provenance.update(
            probe_passed=False,
            probe_attestation_id="unknown",
            probe_attestation=None,
            producer_payload_id="unknown",
        )
    else:
        provenance.update(
            host=finalized["host"],
            vibeqc_version=finalized["vibeqc_version"],
            vibeqc_commit=finalized["source_commit"],
            source_clean=finalized["source_clean"],
            core_build_id=finalized["core_build_id"],
            probe_version=probe_host.PRODUCER_ATTESTATION_VERSION,
            probe_script_id=finalized["attestor_script_id"],
            probe_passed=finalized.get("probe_passed") is True,
            probe_attestation_id=probe_host.probe_attestation_id(finalized),
            probe_attestation=finalized,
            producer_payload_id=finalized["producer_payload_id"],
        )
    return provenance.get("probe_passed") is True


def _write_result(path: Path, result: dict, provenance: dict[str, object]) -> None:
    """Finalize producer evidence and write one raw benchmark record."""

    _finalize_provenance(provenance)
    result["provenance"] = provenance
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if _attestation_required() and provenance.get("probe_passed") is not True:
        raise SystemExit(3)


def _dim_lt_3_reason(dim):
    return (
        f"route needs the 3-D periodic Coulomb mesh; dim={dim} fails closed. The "
        "neutral cderi pins every non-periodic axis at G_perp=0, so its kernel is a "
        "transverse-uniform sheet term ~1/V, not 1/r. Any energy it produced would "
        "vanish with the vacuum padding and diverge with the nuclear lattice-sum "
        "cutoff. No neutral fitted-torus control is defined below 3-D. Use "
        "aiccm-hf for the union-and-weight Γ-CCM construction at dim<3; the "
        "separately derived mixed-boundary wire kernel (run_ccm_rhf_wire) is "
        "available only for its stated dim=1, light-element scope."
    )


# routes whose primary result is an HF (functional=None) quantity
_HF_ROUTES = {
    "aiccm-hf",
    "aiccm-hf-direct",
    "aiccm-ri",
    "aiccm-rijcosx",
    "aiccm-ri-cosx",
    "aiccm-mp2",
    "aiccm-ccsd",
    "aiccm-ri-mp2",
    "aiccm-ri-ccsd",
    "aiccm-dlpno-mp2",
    "aiccm-dlpno-ccsd",
    "aiccm-uccsd",
    "aiccm-properties",
    "aiccm-viz",
    "aiccm-localize",
    "aiccm-symmetry",
    "aiccm-pao",
}

_GAMMA_CCM_CONSTRUCTION_ROUTES = frozenset(
    {
        "aiccm-hf",
        "aiccm-ks",
        "aiccm-mp2",
        "aiccm-ccsd",
        "aiccm-properties",
        "aiccm-viz",
        "aiccm-localize",
        "aiccm-symmetry",
        "aiccm-pao",
    }
)
_NEUTRAL_BLOCH_CONTROL_ROUTES = frozenset({"aiccm-ri", "aiccm-ks-ri"})
_NEUTRAL_COSX_CONTROL_ROUTES = frozenset(
    {"aiccm-ri-cosx", "aiccm-ks-ri-cosx"}
)
_NEUTRAL_POSTHF_RESEARCH_ROUTES = frozenset(
    {
        "aiccm-dlpno-mp2",
        "aiccm-dlpno-ccsd",
        "aiccm-uccsd",
        # Task #12 option (c): canonical correlation on the neutral RI
        # reference, parallel to the bare four-centre aiccm-mp2/aiccm-ccsd
        # rather than replacing them. These are declared neutral controls and
        # must NOT join _GAMMA_CCM_CONSTRUCTION_ROUTES (ruling R1, D89).
        "aiccm-ri-mp2",
        "aiccm-ri-ccsd",
    }
)
_PERIODIC_REFERENCE_ROUTES = frozenset({"bipole", "gdf"})


def _route_semantics(route: str) -> dict[str, str]:
    """Return explicit approach, construction, representation, and route role.

    ``METHOD`` identifies this mixed harness's producer stream; it is not a
    claim that every record is a Γ-CCM construction result. These fields make
    that distinction machine-readable on every emitted record.
    """

    if route in _GAMMA_CCM_CONSTRUCTION_ROUTES:
        return {
            "ccm_approach": "gamma-ccm",
            "ccm_construction": "union-and-weight",
            "evaluation_representation": "real-supercell-four-center",
            "route_role": "ccm-approach",
        }
    if route == "aiccm-hf-direct":
        return {
            "ccm_approach": "not-applicable",
            "ccm_construction": "neutral-fitted-torus",
            "evaluation_representation": "real-gamma-supercell",
            "route_role": "representation-control",
        }
    if route in _NEUTRAL_BLOCH_CONTROL_ROUTES:
        return {
            "ccm_approach": "not-applicable",
            "ccm_construction": "neutral-fitted-torus",
            "evaluation_representation": "bloch-gdf",
            "route_role": "representation-control",
        }
    if route in _NEUTRAL_COSX_CONTROL_ROUTES:
        return {
            "ccm_approach": "not-applicable",
            "ccm_construction": "neutral-fitted-torus-cosx",
            "evaluation_representation": "bloch-gdf-cosx",
            "route_role": "approximate-exchange-control",
        }
    if route in _NEUTRAL_POSTHF_RESEARCH_ROUTES:
        return {
            "ccm_approach": "not-applicable",
            "ccm_construction": "neutral-fitted-torus-research",
            "evaluation_representation": "real-supercell-neutral-cderi",
            "route_role": "post-hf-research-route",
        }
    if route == "aiccm-rijcosx":
        return {
            "ccm_approach": "not-applicable",
            "ccm_construction": "research-wssc-cosx-undeclared",
            "evaluation_representation": "real-supercell-wssc-cosx",
            "route_role": "research-control",
        }
    if route in _PERIODIC_REFERENCE_ROUTES:
        return {
            "ccm_approach": "not-applicable",
            "ccm_construction": "not-applicable",
            "evaluation_representation": "periodic-k-mesh",
            "route_role": "periodic-reference",
        }
    raise ValueError(f"route semantics are not declared for {route!r}")

# Exchange-q=0 convention recorded per route (D1 reproducibility gate / FR-1).
# The -a routes call run_ccm_* directly (no run_periodic_job → no .system manifest),
# so the convention is stamped into the result JSON here instead. The neutral
# fitted-torus controls (multi-k GDF / RI, exxdiv="ewald") carry the
# "BvK-ewald" seam label. A matching B label is necessary but does not by itself
# attest a common operator or define a Γ/χ approach comparison. The bare symmetric
# four-center routes are a *distinct object* (the G=0 self-image is retained → the
# Madelung over-binding on 3-D — covalent and ionic alike), recorded as
# "bare-four-center" so assert_matched_exchange_q0 compares like with like and the
# bare routes are not silently matched to the neutral controls.
_NEUTRAL_KERNEL_ROUTES = {
    "aiccm-ri", "aiccm-ks-ri",
    "aiccm-ri-cosx", "aiccm-ks-ri-cosx",  # fitted-torus RIJCOSX control
    "aiccm-dlpno-mp2", "aiccm-dlpno-ccsd", "aiccm-uccsd",
    "aiccm-hf-direct",  # neutral-torus real-Gamma control, exxdiv='ewald' seam
}

# The older research route combines WSSC RI-J with molecular-supercell COSX.
# It has neither the neutral fitted-torus control operator nor a derived exchange
# seam, so assigning the BvK-Ewald label would be false provenance.  Keep the
# route runnable for research comparisons, but give its undeclared convention
# a unique label that can never pass an approach-comparison gate.
_RESEARCH_WSSC_ROUTES = {"aiccm-rijcosx"}


def _exchange_q0_for_route(route: str) -> str:
    """The exchange-q=0 convention this route's kernel carries."""
    from vibeqc.periodic.exchange_convention import exchange_q0_label
    if route in _RESEARCH_WSSC_ROUTES:
        return "research-wssc-cosx-undeclared"
    if route in _NEUTRAL_KERNEL_ROUTES or route in ("gdf", "bipole"):
        return exchange_q0_label("ewald")  # neutral-control BvK-Madelung seam
    return "bare-four-center"               # bare WSSC 1/r four-center, no Ewald seam


def _ccm(
    system, meta, basis, nrep, *, interaction_range=None, interaction_range_ang=None
):
    """Build the cluster, sized by an explicit ``nrep`` or a real-space radius.

    The interaction-range path (Task A) is the real-space dual of k-point
    density: the minimal cluster whose supercell Wigner–Seitz cell encloses a
    sphere of ``interaction_range`` around every atom is derived automatically.
    """
    unit = testset.build(system)
    if interaction_range is not None:
        return CCMSystem(unit, basis=basis, interaction_range=float(interaction_range))
    if interaction_range_ang is not None:
        return CCMSystem(
            unit, basis=basis, interaction_range_ang=float(interaction_range_ang)
        )
    return CCMSystem(unit, tuple(nrep), basis)


def _aiccm(ccm, functional, open_shell):
    """HF / KS four-center (no RI)."""
    if functional is None:  # HF
        # Closed-shell HF uses the scalable C++ lattice-sum JK builder
        # (`run_ccm_rhf_scalable`): it applies the WSSC four-center weights during
        # the shell-quartet loop instead of materialising the O(n_pad^4) padded
        # ERI, so it reaches genuine 3-D supercells (c-diamond 2×2×2) that OOM the
        # dense `run_ccm_rhf` — matches it to µHa. Open-shell keeps the dense UHF
        # (no scalable UHF builder yet; the 3-D benchmark systems are closed-shell).
        # Cauchy-Schwarz pre-screening (BFCut ~ 1e-10, Neese RIJCOSX 2009 §3.1) on
        # the integral-direct kernel skips negligible shell quartets so the compute
        # drops from O(nbf^4) toward O(nbf^2) at scale (energy unchanged to <1e-10);
        # this is what makes the four-center route finish on 3-D (c-diamond timed
        # out unscreened). Memory is already O(nbf^2) (direct, no dense tensor).
        r = (run_ccm_uhf(ccm, method=METHOD) if open_shell
             else run_ccm_rhf_scalable(ccm, method=METHOD, schwarz_threshold=1e-10))
    else:  # KS
        r = (run_ccm_uks(ccm, functional, method=METHOD) if open_shell
             else run_ccm_rks(ccm, functional, method=METHOD, schwarz_threshold=1e-10))
    return (
        r.energy,
        r.energy_per_atom,
        bool(r.converged),
        dict(nbf=ccm.nbf, n_atoms=ccm.n_atoms, n_cells=ccm.n_cells,
             backend=str(getattr(r, "backend", "") or ""),
             parity_held=bool(getattr(r, "parity_held", False))),
    )


def _aiccm_ri(ccm, functional):
    """RI (density-fitted) Coulomb via multi-k GDF — HF (functional=None) or KS."""
    r = run_ccm_rks_gdf(ccm, functional) if functional else run_ccm_rhf_gdf(ccm)
    return (r.energy, r.energy_per_atom, bool(r.converged),
            dict(n_atoms=ccm.n_atoms,
                 backend=str(getattr(r, "backend", "") or ""),
                 parity_held=bool(getattr(r, "parity_held", False))))


def _aiccm_ri_cosx(ccm, functional):
    """RIJCOSX on the neutral fitted-torus GDF/COSX control.

    RI-J plus seminumerical COSX-K uses the multi-k GDF representation
    (``k_exchange="cosx"``) for HF (functional=None) or a hybrid. This is a
    same-control backend check, not Γ-CCM construction evidence. COSX needs the
    compensated-cell multi-k grid (``use_compcell=True``).
    """
    kw = dict(k_exchange="cosx", use_compcell=True)
    r = (run_ccm_rks_gdf(ccm, functional, **kw) if functional
         else run_ccm_rhf_gdf(ccm, **kw))
    return (r.energy, r.energy_per_atom, bool(r.converged),
            dict(n_atoms=ccm.n_atoms,
                 backend=str(getattr(r, "backend", "") or ""),
                 parity_held=bool(getattr(r, "parity_held", False))))


def _aiccm_rijcosx(ccm):
    r = run_ccm_rhf_rijcosx(ccm)
    return (
        r.energy,
        r.energy_per_atom,
        bool(r.converged),
        dict(nbf=ccm.nbf, n_atoms=ccm.n_atoms,
             backend=str(getattr(r, "backend", "") or ""),
             parity_held=bool(getattr(r, "parity_held", False))),
    )


def _aiccm_hf_direct(ccm, ke_cutoff):
    """Neutral-torus real-Gamma HF control with the derived exchange-q0 seam.

    One real generalized eigenproblem over the supercell AOs. It is a same-H
    representation control for ``aiccm-ri`` at the shared ``BvK-ewald``
    convention, not a union-and-weight Γ-CCM construction route (see
    ``periodic/ccm/direct.py``). Energies are per supercell in the result
    object; normalised like the other cluster-backed routes.
    """
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct

    r = run_ccm_rhf_direct(ccm, ke_cutoff=ke_cutoff)
    # exchange_q0 is stamped route-wide by _exchange_q0_for_route; assert the
    # result's own record agrees so the JSON can never drift from the route.
    from vibeqc.periodic.exchange_convention import assert_matched_exchange_q0

    assert_matched_exchange_q0(
        [r.exchange_q0, _exchange_q0_for_route("aiccm-hf-direct")],
        context="aiccm-hf-direct",
    )
    return (
        r.energy,
        r.energy_per_atom,
        bool(r.converged),
        dict(nbf=ccm.nbf, n_atoms=ccm.n_atoms,
             backend=str(getattr(r, "backend", "") or ""),
             parity_held=bool(getattr(r, "parity_held", False))),
    )


def _aiccm_corr(ccm, level, open_shell):
    """MP2 / UMP2 / CCSD(T) correlation on the CCM MO integrals.

    Primary energy is the total (HF + correlation); the breakdown
    (HF, correlation, (T)) is recorded in ``info``.
    """
    info = dict(nbf=ccm.nbf, n_atoms=ccm.n_atoms, n_cells=ccm.n_cells)

    def _stamp_reference(scf_ref):
        # IID 344: the record's backend names the operator that produced the
        # SCF reference the correlation is built on; a held reference holds
        # everything built on it.
        info.update(
            backend=str(getattr(scf_ref, "backend", "") or ""),
            parity_held=bool(getattr(scf_ref, "parity_held", False)),
        )

    if level == "mp2":
        r = (run_ccm_ump2 if open_shell else run_ccm_mp2)(ccm, method=METHOD)
        info.update(
            e_hf=float(r.e_hf),
            e_correlation=float(r.e_correlation),
            e_correlation_per_atom=float(r.e_correlation_per_atom),
        )
        _stamp_reference(getattr(r, "scf", None) or getattr(r, "uhf", None))
        conv = bool(getattr(getattr(r, "scf", None), "converged", True))
        return float(r.e_total), float(r.e_total_per_atom), conv, info
    # CCSD(T): closed-shell only
    if open_shell:
        raise SystemExit(
            "aiccm-ccsd: open-shell CCSD not implemented; "
            "use aiccm-mp2 (UMP2) for open-shell correlation."
        )
    r = run_ccm_ccsd(ccm, method=METHOD, compute_triples=True)
    info.update(
        e_hf=float(r.e_hf),
        e_ccsd_correlation=float(r.e_ccsd_correlation),
        e_t=float(r.e_t),
        e_correlation=float(r.e_correlation),
        e_correlation_per_atom=float(r.e_correlation_per_atom),
    )
    _stamp_reference(getattr(r, "scf", None))
    return float(r.e_total), float(r.e_total_per_atom), bool(r.converged), info


def _aiccm_ri_corr(ccm, level, open_shell, ke_cutoff):
    """Canonical MP2 / UMP2 / CCSD on the NEUTRAL fitted-torus RI reference.

    Task #12 option (c), maintainer 2026-09-05: this lands *beside* the bare
    four-centre ``aiccm-mp2`` / ``aiccm-ccsd`` routes rather than replacing or
    renaming them, so the published 1-D/2-D bare numbers stay reproducible
    under the name they were published with while 3-D correlation goes through
    the neutral reference.

    Unlike the bare routes this has a clean RI decomposition, so it reaches
    3-D. The two are complementary rather than one superseding the other: the
    bare route is dense and therefore 1-D/2-D only, while this one refuses
    ``dim < 3``, because ``ccm_neutral_cderi`` needs a genuine 3-D reciprocal
    mesh (a transverse-collapsed one makes the Coulomb kernel a sheet term
    proportional to 1/V rather than 1/r). That refusal comes from
    ``aux_basis._reject_transverse_collapse`` and names the 1-D alternative,
    ``periodic.ccm.lowd_scf.run_ccm_rhf_wire``.

    It is a declared neutral fitted-torus control, NOT a union-and-weight
    Gamma-CCM correlation result -- the same standing the ``aiccm-dlpno-*``
    routes have (ruling R1; ``docs/aiccm2026dev_a_followon.md`` section 2.0).

    ``run_ccm_ri_mp2`` / ``run_ccm_ri_ccsd`` default to ``reference="direct"``
    since D-5 (2026-09-05); the default is taken deliberately here so this
    route tracks the method's own reference convention.
    """
    info = dict(nbf=ccm.nbf, n_atoms=ccm.n_atoms, n_cells=ccm.n_cells)
    if level == "ccsd" and open_shell:
        raise SystemExit(
            "aiccm-ri-ccsd: open-shell CCSD not implemented; "
            "use aiccm-ri-mp2 (UMP2) for open-shell correlation."
        )
    if level == "mp2":
        r = run_ccm_ri_mp2(ccm, ke_cutoff=ke_cutoff)
    else:
        r = run_ccm_ri_ccsd(ccm, ke_cutoff=ke_cutoff, compute_triples=False)
    scf_ref = getattr(r, "scf", None) or getattr(r, "uhf", None)
    info.update(
        e_hf=float(r.e_hf),
        e_correlation=float(r.e_correlation),
        e_correlation_per_atom=float(r.e_correlation_per_atom),
        # IID 344: name the operator that produced the reference this
        # correlation is built on; a held reference holds everything on it.
        backend=str(getattr(scf_ref, "backend", "") or ""),
        parity_held=bool(getattr(scf_ref, "parity_held", False)),
        reference_route="direct",
    )
    conv = bool(getattr(scf_ref, "converged", True))
    return float(r.e_total), float(r.e_total_per_atom), conv, info


def _neutral_reference(ccm, ke_cutoff):
    """The neutral four-center ``g_eff = Σ_P L⊗L`` and its cderi ``L`` (small clusters).

    Internal reference for harness routes that explicitly declare the neutral
    fitted-torus control. It is not the union-and-weight Γ-CCM correlation
    reference (see ``docs/aiccm2026dev_a_followon.md`` § 2.0).
    """
    from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi

    g = ccm_eri_neutral(ccm, ke_cutoff=ke_cutoff)
    L = ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff)
    return g, L


def _n_core_supercell(ccm):
    """Standard frozen-core orbital count over the whole supercell (1s for Li-Ne,
    +2s2p for Na-Ar, +3s3p for K-Kr; none for H/He)."""
    def core(z):
        return 0 if z <= 2 else 1 if z <= 10 else 5 if z <= 18 else 9
    return int(sum(core(int(a.Z)) for a in ccm.supercell.atoms))


def _aiccm_dlpno(ccm, level, tcut_pno, ke_cutoff, frozen_core=False):
    """DLPNO-MP2 / DLPNO-CCSD(T) on the neutral fitted-torus control.

    This harness route is a local-correlation study of that separately declared
    control, not a Γ-CCM local-correlation result.

    **Memory-lean path** (default): the fold cderi ``L`` (per-q unit-cell fits, no
    supercell-FFT wall), the lean neutral SCF ``run_ccm_rhf_ri_neutral(cderi=L)``
    (no dense ``n_ref^4``), and the per-pair-coupled DLPNO-CCSD(T) with the
    scalable local (T) (``triples_mode="local"``) -- so the dense AO/MO four-center
    is **never formed** (the ``ccm_dlpno_ccsd`` union path built ``nbf_sc^4`` ~
    500 GB on MgO/pob-tzvp). ``n_frozen`` freezes the atomic cores (standard
    frozen-core) to shrink the correlation space further. Matches the dense gate
    path to the DLPNO(T0) accuracy (~1e-6 on the h-chain control).

    ⚠ **dim=3 caveat (2026-07-10, D2 KMP2 anchor, benchmark chat's job
    ``2874a52c8d39``): this route's HF reference is the strict-zero neutral SCF,
    and that reference is WRONG at dim=3 — its error grows with cluster size (it
    deletes the free-space exchange monopole, ``handovers/HANDOVER_D2_EXXDIV.md``).
    The correlation inherits the distortion: c-diamond (2,2,2) CCM-neutral RI-MP2
    gave E_corr = −59.0 mHa/atom against out-of-process PySCF KMP2@ewald −29.9
    (~2×). Do NOT curate dim=3 DLPNO/RI correlation from this route until it is
    re-based on the ewald reference (``run_ccm_rhf_gdf`` / direct-torus). The fix
    touches the correlation-reference architecture (benchmark/-b chat's domain) —
    raise it as an ask, do not change it here unilaterally. The dim<3 numbers are
    already retracted for the transverse-collapse reason; this is the dim=3 caveat.**
    """
    from vibeqc.periodic.ccm import ccm_dlpno_ccsd_coupled, ccm_dlpno_mp2
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_ri_neutral

    L = ccm_neutral_cderi_fold(ccm, ke_cutoff=ke_cutoff)   # per-q fold, memory-lean
    scf = run_ccm_rhf_ri_neutral(ccm, cderi=L)             # no dense g
    n_frozen = _n_core_supercell(ccm) if frozen_core else 0
    info = dict(
        nbf=ccm.nbf,
        n_atoms=ccm.n_atoms,
        n_cells=ccm.n_cells,
        reference="neutral",
        cderi="fold",
        tcut_pno=tcut_pno,
        ke_cutoff=ke_cutoff,
        n_frozen=n_frozen,
        # IID 344: the SCF reference's operator identity + hold state; a
        # held reference holds everything built on it.
        backend=str(getattr(scf, "backend", "") or ""),
        parity_held=bool(getattr(scf, "parity_held", False)),
    )
    if level == "mp2":
        d = ccm_dlpno_mp2(ccm, scf, cderi=L, localize="pm", tcut_pno=tcut_pno)
        info.update(
            e_hf=float(d.e_hf),
            e_correlation=float(d.e_corr),
            e_correlation_per_atom=float(d.e_corr_per_atom),
            n_pairs=int(d.n_pairs),
            e_pno_correction=float(d.e_pno_correction),
        )
        return float(d.e_total), float(d.e_total / ccm.n_atoms), bool(d.converged), info
    d = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="pm", tcut_pno=tcut_pno,
                               triples_mode="local", n_frozen=n_frozen)
    info.update(
        e_hf=float(d.e_hf),
        e_ccsd_correlation=float(d.e_ccsd_correlation),
        e_t=float(d.e_t),
        e_correlation=float(d.e_correlation),
        e_correlation_per_atom=float(d.e_correlation / ccm.n_atoms),
    )
    return float(d.e_total), float(d.e_total / ccm.n_atoms), bool(d.converged), info


def _aiccm_uccsd(ccm, ke_cutoff):
    """Open-shell UCCSD(T) on the declared neutral fitted-torus control (Task D)."""
    from vibeqc.periodic.ccm import run_ccm_uccsd
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

    L = ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff)
    u = run_ccm_uccsd(ccm, cderi=L, compute_triples=True)
    info = dict(
        nbf=ccm.nbf,
        n_atoms=ccm.n_atoms,
        n_cells=ccm.n_cells,
        reference="neutral",
        # IID 344: the UHF reference's operator identity + hold state.
        backend=str(getattr(u.uhf, "backend", "") or ""),
        parity_held=bool(getattr(u.uhf, "parity_held", False)),
        n_alpha=int(u.uhf.n_alpha),
        n_beta=int(u.uhf.n_beta),
        e_hf=float(u.e_hf),
        e_ccsd_correlation=float(u.e_ccsd_correlation),
        e_t=float(u.e_t),
        e_correlation=float(u.e_correlation),
        e_correlation_per_atom=float(u.e_correlation_per_atom),
    )
    return float(u.e_total), float(u.e_total_per_atom), bool(u.converged), info


def _aiccm_properties(ccm, out: Path, stem_name: str, with_gradient: bool):
    """Task C properties: gap + Mulliken/Löwdin populations + dipole (+ optional
    numerical gradient), with the canonical orbitals + density emitted for
    vibe-view (.qvf / .cube)."""
    from vibeqc.periodic.ccm import (
        ccm_dipole,
        ccm_homo_lumo_gap,
        ccm_lowdin_charges,
        ccm_mulliken_charges,
        ccm_numerical_gradient,
    )

    mol, b = ccm.supercell, ccm.basis
    open_shell = mol.n_electrons() % 2 != 0
    # scalable closed-shell SCF reaches genuine 3-D; open-shell uses the dense UHF.
    scf = (
        run_ccm_uhf(ccm, method=METHOD)
        if open_shell
        else run_ccm_rhf_scalable(ccm, method=METHOD)
    )

    gap = ccm_homo_lumo_gap(scf, ccm)
    mul = ccm_mulliken_charges(scf, ccm)
    low = ccm_lowdin_charges(scf, ccm)
    mu = ccm_dipole(scf, ccm)
    info = dict(
        nbf=ccm.nbf,
        n_atoms=ccm.n_atoms,
        n_cells=ccm.n_cells,
        shell="open" if open_shell else "closed",
        gap=float(gap.gap),
        homo=float(gap.homo),
        lumo=float(gap.lumo),
        gap_spin=gap.spin,
        mulliken_per_cell=[float(x) for x in mul.charges_per_cell],
        lowdin_per_cell=[float(x) for x in low.charges_per_cell],
        mulliken_translational_spread=float(mul.translational_spread),
        dipole_au=[float(x) for x in mu],
        dipole_norm=float(np.linalg.norm(mu)),
    )

    # vibe-view output: canonical orbitals (.qvf, resolution-independent) + density.
    from vibeqc import write_cube_density

    stem = str(out / stem_name)
    base = os.path.basename(stem)
    files: list[str] = []
    try:
        write_cube_density(
            stem + "_density.cube",
            np.asarray(scf.density),
            b,
            mol,
            spacing=_VIZ_SPACING,
            padding=3.0,
        )
        files.append(base + "_density.cube")
        if not open_shell:
            p = _write_orbital_qvf(
                stem + "_canonical",
                mol,
                b,
                ccm.basis_name,
                scf,
                scf.mo_coeffs,
                scf.mo_energies,
            )
            files.append(os.path.basename(str(p)))
    except Exception as ex:  # noqa: BLE001
        files.append(f"viz FAILED: {type(ex).__name__}: {ex}")
    info["viz_files"] = files

    if with_gradient:
        grad = ccm_numerical_gradient(ccm, method=METHOD)
        info["gradient_au"] = grad.round(8).tolist()
        info["max_force_au"] = float(np.max(np.abs(grad)))
        info["sum_force_au"] = (
            (-grad.sum(axis=0)).round(10).tolist()
        )  # ≈0 (transl. inv.)

    return float(scf.energy), float(scf.energy_per_atom), bool(scf.converged), info


_VIZ_SPACING = 0.20  # paper-quality grid (bohr); the .qvf is resolution-independent
_VIZ_MAX_CUBES = 16  # cap per orbital set (canonical / Wannier) to bound file count


def _write_orbital_qvf(stem, mol, b, basis_name, result, mo_coeffs, mo_energies):
    """vibe-view .qvf carrying the given MO set (canonical COs or Wanniers)."""
    import dataclasses

    from vibeqc.output.formats.qvf import qvf_wf_data, write_qvf
    from vibeqc.output.plan import OutputPlan

    res = dataclasses.replace(
        result, mo_coeffs=np.asarray(mo_coeffs), mo_energies=np.asarray(mo_energies)
    )
    plan = OutputPlan.from_run_job_kwargs(
        output=stem,
        method="RHF",
        basis=str(basis_name),
        functional=None,
        write_molden_file=True,
        output_qvf=True,
    )
    wf = qvf_wf_data(res, b, mol)
    return write_qvf(
        stem,
        plan,
        molecule=mol,
        result=res,
        method="RHF",
        basis=b,
        functional=None,
        wf_data=wf,
        wall_seconds=0.0,
    )


def _aiccm_viz(ccm, out: Path, stem_name: str):
    """HF + paper-quality visualization of the crystalline orbitals AND the
    localized (Wannier) orbitals for vibe-view.

    The figure the paper wants: on the *same* cluster, the delocalized canonical
    crystalline orbitals vs the compact localized Wannier functions. Writes into
    ``out`` (closed-shell):

    * ``<stem>_canonical.qvf`` — vibe-view-native wavefunction.gto of the canonical
      COs (vibe-view resamples any CO / the density at any resolution → cover
      quality), + ``<stem>.molden`` + ``<stem>_density.cube``;
    * ``<stem>_co_NN.cube`` — frontier canonical COs (delocalized Bloch states);
    * ``<stem>_wannier.qvf`` + ``<stem>_wannier_NN.cube`` — the localized Wannier
      functions (Pipek-Mezey, Task 1), centered on bonds/atoms;
    * Wannier centers + spreads recorded in the JSON.

    The visualized property is the HF/Wannier orbitals (the CCM post-HF stack
    returns correlation energies only — no relaxed density / natural orbitals).
    """
    from vibeqc import write_cube_density, write_cube_mo, write_molden
    from vibeqc.periodic.ccm.localize import localise_ccm

    mol, b = ccm.supercell, ccm.basis
    stem = str(out / stem_name)
    base = os.path.basename(stem)
    files: list[str] = []
    sp = _VIZ_SPACING
    open_shell = mol.n_electrons() % 2 != 0

    if open_shell:  # density cube only (UHF total density)
        r = run_ccm_uhf(ccm, method=METHOD)
        try:
            write_cube_density(
                stem + "_density.cube",
                np.asarray(r.density),
                b,
                mol,
                spacing=sp,
                padding=3.0,
            )
            files.append(base + "_density.cube")
        except Exception as ex:  # noqa: BLE001
            files.append(f"density.cube FAILED: {type(ex).__name__}: {ex}")
        return (
            r.energy,
            r.energy_per_atom,
            bool(r.converged),
            dict(nbf=ccm.nbf, n_atoms=ccm.n_atoms, viz_files=files, viz_shell="open"),
        )

    # Scalable C++ lattice-sum JK (FR-2): the dense four-center is un-storable on
    # 3-D (c-diamond (2,2,2) padded ERI ~ 55,800 TiB); scalable reproduces the
    # dense four-center to machine ε (1-D + 2-D parity gated) and reaches 3-D.
    # The route gap may include a uniform ∝S Fock component, but that observation
    # does not identify the two constructions. The reported energy and orbitals
    # remain those of union-and-weight Γ-CCM; aiccm-ri is a separately declared
    # 3-D neutral fitted-torus control, not an accuracy replacement for Γ-CCM.
    r = run_ccm_rhf_scalable(ccm, method=METHOD)
    C = np.asarray(r.mo_coeffs)
    eps = np.asarray(r.mo_energies)
    nocc = mol.n_electrons() // 2
    info = dict(nbf=ccm.nbf, n_atoms=ccm.n_atoms, viz_shell="closed")

    def _try(label, fn):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            files.append(f"{label} FAILED: {type(ex).__name__}: {ex}")

    # --- canonical crystalline orbitals (delocalized) ---
    def _canonical_qvf():
        p = _write_orbital_qvf(stem + "_canonical", mol, b, ccm.basis_name, r, C, eps)
        files.append(os.path.basename(str(p)))

    _try("canonical.qvf", _canonical_qvf)
    _try(
        "molden",
        lambda: (
            write_molden(stem + ".molden", mol, b, r),
            files.append(base + ".molden"),
        ),
    )
    _try(
        "density.cube",
        lambda: (
            write_cube_density(
                stem + "_density.cube",
                np.asarray(r.density),
                b,
                mol,
                spacing=sp,
                padding=3.0,
            ),
            files.append(base + "_density.cube"),
        ),
    )

    def _co_cubes():
        lo, hi = max(0, nocc - 2), min(C.shape[1], nocc + 2)  # HOMO-1..LUMO+1
        for idx in range(lo, hi):
            write_cube_mo(
                stem + f"_co_{idx:02d}.cube", C, idx, b, mol, spacing=sp, padding=3.0
            )
            files.append(base + f"_co_{idx:02d}.cube")

    _try("co.cubes", _co_cubes)

    # --- localized Wannier functions (compact, on bonds/atoms) ---
    def _wannier():
        w = localise_ccm(r, ccm, method="pipek-mezey")
        F = np.asarray(r.fock)
        eps_loc = np.einsum(
            "mi,mn,ni->i", w.C_loc, F, w.C_loc
        )  # diagonal Fock "energies"
        n_show = min(w.n_occ, _VIZ_MAX_CUBES)
        for i in range(n_show):
            write_cube_mo(
                stem + f"_wannier_{i:02d}.cube",
                w.C_loc,
                i,
                b,
                mol,
                spacing=sp,
                padding=3.0,
            )
            files.append(base + f"_wannier_{i:02d}.cube")
        info["wannier_centers"] = np.asarray(w.centroids).tolist()
        info["wannier_spreads"] = np.asarray(w.spreads).tolist()
        info["n_wannier"] = int(w.n_occ)
        try:
            p = _write_orbital_qvf(
                stem + "_wannier", mol, b, ccm.basis_name, r, w.C_loc, eps_loc
            )
            files.append(os.path.basename(str(p)))
        except Exception as ex:  # noqa: BLE001
            files.append(f"wannier.qvf FAILED: {type(ex).__name__}: {ex}")

    _try("wannier", _wannier)

    info["viz_files"] = files
    return r.energy, r.energy_per_atom, bool(r.converged), info


def _aiccm_localize(ccm):
    """Demonstrate Task 1 (Wannier localization): run HF, localize, report the
    unitary-invariance gate (density/energy preserved) + Wannier centers/spreads."""
    from vibeqc.periodic.ccm.localize import localise_ccm, localization_density_residual

    r = run_ccm_rhf_scalable(ccm, method=METHOD)  # FR-2: scalable reaches 3-D
    w = localise_ccm(r, ccm, method="pipek-mezey")
    info = dict(
        nbf=ccm.nbf,
        n_atoms=ccm.n_atoms,
        n_wannier=int(w.n_occ),
        density_residual=float(localization_density_residual(r, w)),
        objective_initial=float(w.objective_initial),
        objective_final=float(w.objective_final),
        wannier_spreads=np.asarray(w.spreads).round(4).tolist(),
        wannier_centers=np.asarray(w.centroids).round(4).tolist(),
    )
    return r.energy, r.energy_per_atom, bool(r.converged), info


def _aiccm_symmetry(ccm):
    """Demonstrate Task 3 (space-group symmetry): space group, cluster-invariant
    subgroup, AO-map correctness (S/T invariant), the union-V finding, and the
    symmetry-unique atom-pair integral reduction."""
    from vibeqc.periodic.ccm.integrals import ccm_kinetic, ccm_overlap
    from vibeqc.periodic.ccm.symmetry import (
        analyze_ccm_symmetry,
        ccm_symmetry_unique_atom_pairs,
    )

    sym = analyze_ccm_symmetry(ccm)
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)

    def _resid(M):
        return max(
            (float(np.linalg.norm(o.P.T @ M @ o.P - M)) for o in sym.invariant_ops),
            default=0.0,
        )

    info = dict(
        nbf=ccm.nbf,
        n_atoms=ccm.n_atoms,
        spacegroup=sym.number,
        symbol=sym.international_symbol,
        point_group=sym.point_group,
        crystal_order=sym.crystal_order,
        cluster_order=sym.cluster_order,
        overlap_invariance=_resid(np.asarray(ccm_overlap(ccm))),
        kinetic_invariance=_resid(np.asarray(ccm_kinetic(ccm))),
        n_unique_pairs=orb.n_unique,
        pair_reduction_factor=round(orb.reduction_factor, 3),
    )
    try:  # union-V symmetry finding — heavier (3-center padded build); guarded
        from vibeqc.periodic.ccm.padded import ccm_nuclear

        info["nuclear_invariance_union"] = _resid(np.asarray(ccm_nuclear(ccm)))
    except Exception as ex:  # noqa: BLE001
        info["nuclear_invariance_union"] = f"skipped: {type(ex).__name__}"
    return None, None, True, info  # analysis route — no energy


def _aiccm_pao(ccm):
    """Demonstrate Task 2 (DLPNO PAO): build projected atomic orbitals, report
    the count (= virtual dim) and S-orthogonality to the occupied space."""
    from vibeqc.periodic.ccm.dlpno import ccm_pao, pao_occupied_orthogonality

    r = run_ccm_rhf_scalable(ccm, method=METHOD)  # FR-2: scalable reaches 3-D
    C_pao, n_pao = ccm_pao(r, ccm)
    n_occ = ccm.supercell.n_electrons() // 2
    info = dict(
        nbf=ccm.nbf,
        n_atoms=ccm.n_atoms,
        n_occ=n_occ,
        n_virtual=ccm.nbf - n_occ,
        n_pao=int(n_pao),
        pao_occ_orthogonality=float(pao_occupied_orthogonality(r, ccm, C_pao)),
    )
    return r.energy, r.energy_per_atom, bool(r.converged), info


def _periodic(system, basis, kmesh, functional, jk, open_shell, out):
    unit = testset.build(system)
    b = vq.BasisSet(unit.unit_cell_molecule(), basis)
    method = (
        ("UKS" if open_shell else "RKS")
        if functional
        else ("UHF" if open_shell else "RHF")
    )
    r = vq.run_periodic_job(
        unit,
        b,
        method=method,
        functional=functional,
        jk_method=jk,
        kpoints=tuple(kmesh),
        output=str(out),
        max_iter=120,
        conv_tol_energy=1e-7,
        write_molden_file=False,
        write_density=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
        write_poscar_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    n_unit = len(list(unit.unit_cell_molecule().atoms))
    return (
        float(r.energy),
        float(r.energy) / n_unit,
        bool(getattr(r, "converged", True)),
        dict(kmesh=list(kmesh), n_atoms_unit=n_unit,
             backend=str(getattr(r, "backend", "") or ""),
             parity_held=bool(
                 getattr(r, "parity_held",
                         "+PARITY_HELD" in str(getattr(r, "backend", "") or ""))
             )),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=sorted(testset.SYSTEMS))
    ap.add_argument("route", choices=ROUTES)
    ap.add_argument("--basis", default=None, help="override basis (default: registry)")
    ap.add_argument("--functional", default="pbe", help="XC for KS routes")
    ap.add_argument("--nrep", type=int, nargs=3, default=None)
    ap.add_argument("--kmesh", type=int, nargs=3, default=None)
    ap.add_argument(
        "--interaction-range",
        type=float,
        default=None,
        help="real-space WSC interaction radius R_c (bohr); derives nrep "
        "(Task A, the real-space dual of k-point density)",
    )
    ap.add_argument(
        "--interaction-range-ang", type=float, default=None, help="same, in angstrom"
    )
    ap.add_argument(
        "--tcut-pno",
        type=float,
        default=0.0,
        help="PNO occupation truncation for the DLPNO routes (0 = canonical)",
    )
    ap.add_argument(
        "--ke-cutoff",
        type=float,
        default=200.0,
        help="plane-wave cutoff for the neutral cderi (correlation routes)",
    )
    ap.add_argument(
        "--gradient",
        action="store_true",
        help="also compute the numerical nuclear gradient (aiccm-properties)",
    )
    ap.add_argument(
        "--frozen-core",
        action="store_true",
        help="freeze atomic cores in the DLPNO correlation routes (memory + time)",
    )
    ap.add_argument("--out", default=os.environ.get("VQ_WORKDIR", "."))
    args = ap.parse_args()

    provenance = _provenance()
    _require_attested_launch(provenance)

    meta = testset.SYSTEMS[args.system]
    basis = args.basis or meta["basis"]
    nrep = args.nrep or meta["nrep_4c"]
    kmesh = args.kmesh or meta["kmesh"]
    open_shell = bool(meta.get("open_shell", False))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    route_functional = None if args.route in _HF_ROUTES else args.functional
    route_semantics = _route_semantics(args.route)

    # Build the cyclic cluster once (radius path derives nrep). Periodic
    # references (bipole/gdf) don't use it.
    ccm_obj = None
    if args.route not in ("bipole", "gdf"):
        ccm_obj = _ccm(
            args.system,
            meta,
            basis,
            nrep,
            interaction_range=args.interaction_range,
            interaction_range_ang=args.interaction_range_ang,
        )

    t0 = time.time()
    if args.route in NEEDS_3D_COULOMB and int(meta["dim"]) != 3:
        # Fail closed, don't crash: vibeqc raises NotImplementedError here since
        # 2026-07-10 (rsgdf_dense_g_mesh). Record the fact so curate.py stores a
        # first-class "unavailable" row instead of a hole, and so nobody mistakes a
        # missing file for a missing run. See _dim_lt_3_reason for the physics.
        result = dict(
            system=args.system, route=args.route, basis=basis,
            functional=route_functional, nrep=list(nrep), dim=meta["dim"],
            klass=meta["klass"], energy=None, energy_per_atom=None,
            converged=False, status="unavailable", reason=_dim_lt_3_reason(meta["dim"]),
            crystal_ref=meta["crystal_ref"], walltime_s=0.0,
            provenance=provenance,
            **route_semantics,
        )
        _write_result(
            out / f"{args.system}__{args.route}.json",
            result,
            provenance,
        )
        return

    if args.route in ("aiccm-hf", "aiccm-ks"):
        e, epa, conv, info = _aiccm(ccm_obj, route_functional, open_shell)
    elif args.route == "aiccm-hf-direct":
        e, epa, conv, info = _aiccm_hf_direct(ccm_obj, args.ke_cutoff)
    elif args.route in ("aiccm-ri", "aiccm-ks-ri"):
        e, epa, conv, info = _aiccm_ri(ccm_obj, route_functional)
    elif args.route in ("aiccm-ri-cosx", "aiccm-ks-ri-cosx"):
        e, epa, conv, info = _aiccm_ri_cosx(ccm_obj, route_functional)
    elif args.route == "aiccm-rijcosx":
        e, epa, conv, info = _aiccm_rijcosx(ccm_obj)
    elif args.route == "aiccm-mp2":
        e, epa, conv, info = _aiccm_corr(ccm_obj, "mp2", open_shell)
    elif args.route == "aiccm-ccsd":
        e, epa, conv, info = _aiccm_corr(ccm_obj, "ccsd", open_shell)
    elif args.route in ("aiccm-ri-mp2", "aiccm-ri-ccsd"):
        e, epa, conv, info = _aiccm_ri_corr(
            ccm_obj,
            "mp2" if args.route.endswith("mp2") else "ccsd",
            open_shell,
            args.ke_cutoff,
        )
    elif args.route in ("aiccm-dlpno-mp2", "aiccm-dlpno-ccsd"):
        level = "mp2" if args.route.endswith("mp2") else "ccsd"
        e, epa, conv, info = _aiccm_dlpno(ccm_obj, level, args.tcut_pno,
                                          args.ke_cutoff, frozen_core=args.frozen_core)
    elif args.route == "aiccm-uccsd":
        e, epa, conv, info = _aiccm_uccsd(ccm_obj, args.ke_cutoff)
    elif args.route == "aiccm-properties":
        e, epa, conv, info = _aiccm_properties(
            ccm_obj, out, f"{args.system}_props", args.gradient
        )
    elif args.route == "aiccm-viz":
        e, epa, conv, info = _aiccm_viz(ccm_obj, out, f"{args.system}_viz")
    elif args.route == "aiccm-localize":
        e, epa, conv, info = _aiccm_localize(ccm_obj)
    elif args.route == "aiccm-symmetry":
        e, epa, conv, info = _aiccm_symmetry(ccm_obj)
    elif args.route == "aiccm-pao":
        e, epa, conv, info = _aiccm_pao(ccm_obj)
    else:  # bipole / gdf periodic reference
        e, epa, conv, info = _periodic(
            args.system,
            basis,
            kmesh,
            args.functional,
            args.route,
            open_shell,
            out / f"{args.system}_{args.route}",
        )

    nrep_used = list(ccm_obj.nrep) if ccm_obj is not None else list(nrep)
    # The backend string (``+PARITY_HELD`` suffix included) is a load-bearing
    # record field: a consumer must be able to tell a held number from a
    # converged one without re-reading the .err sidecar (IID 344). Routes
    # whose result type carries no backend record "". ``parity_held`` is the
    # structured form -- routes stamp it from the result object's own flag;
    # the fallback derives it from the backend marker so the two can never
    # disagree on a backend-carrying route.
    backend = info.pop("backend", "")
    parity_held = bool(info.pop("parity_held", "+PARITY_HELD" in backend))
    result = dict(
        system=args.system,
        route=args.route,
        basis=basis,
        functional=route_functional,
        nrep=nrep_used,
        interaction_range=args.interaction_range,
        wsc_inscribed_radius=(
            round(ccm_obj.wsc_inscribed_radius, 4) if ccm_obj is not None else None
        ),
        dim=meta["dim"],
        klass=meta["klass"],
        energy=e,
        energy_per_atom=epa,
        converged=conv,
        backend=backend,
        parity_held=parity_held,
        four_center_quantitative=meta.get("four_center_quantitative"),
        crystal_ref=meta["crystal_ref"],
        exchange_q0=_exchange_q0_for_route(args.route),
        walltime_s=round(time.time() - t0, 1),
        provenance=provenance,
        **info,
    )
    result.update(route_semantics)
    _write_result(
        out / f"{args.system}__{args.route}.json",
        result,
        provenance,
    )


if __name__ == "__main__":
    main()
