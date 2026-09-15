#!/usr/bin/env python3
"""Regenerate the per-file release-tier manifest without losing measurements."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("suite_manifest.json")
LANES = Path(__file__).with_name("lane_manifest.json")

T0_FILES: set[str] = set()

PRODUCTION_T1_FILES = {
    "tests/test_release_sentinels.py",
}

FILE_OWNERS = {
    "tests/test_audit_20260530_periodic.py": "periodic-SCF",
}

# Hand-curated per-file classification that ``derive`` cannot reconstruct from
# lane_manifest.json.  Every field named here survives regeneration; fields left
# out stay derived, so a lane change still flows through.
#
# This table exists because regeneration rebuilds all ~700 rows from scratch.
# Adding a test file obliges a chat to rerun this script (tests/conftest.py
# refuses an unclassified file), and without the table that rerun silently
# retiers and re-rationalises another chat's row inside a diff far too large to
# eyeball.  ``tests/test_test_gate_lanes.py`` pins regeneration as a no-op so
# the clobber fails a test instead of riding along unnoticed.
#
# Note that the release cut selects its blocking set with ``--tier T1`` off this
# manifest, not off a lane (scripts/test_gate/README.md: the lanes are
# "compatibility views, not the release-cut source of truth").  A ``tier`` pinned
# here therefore widens or narrows the release gate on its own; pin one only as
# the file's owning chat, and say why in the rationale.
CURATED: dict[str, dict[str, str]] = {
    "tests/test_contributor_workflow_contract.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "new",
        "rationale": (
            "Core contributor-policy contract retained after operational loop "
            "tests moved to vibe-qc-agentic-loop: automated fixers post the "
            "verification brief after a successful push and before landing "
            "closes the progress lease. Pure text policy check; no native work."
        ),
    },
    "tests/test_release_ref_selection.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "Hermetic real-Git installer release branch and stable-tag selection; no native calculation",
    },
    "tests/test_aiccm_site_settings.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "Hermetic external study profile and interpreter dispatch boundaries; no native calculation",
    },
    "tests/test_git_clone_pinned.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "Sub-minute Bash-level regressions for scripts/_verify_source.sh git_clone_pinned (#241): the commit pin refuses a moved tag, VIBEQC_GIT_REFERENCE_DIR serves the pinned commit from a local repository offline and only from one holding that commit, upstream failures retry a bounded number of times, and the clone deadline escalates to SIGKILL and honours Ctrl-C. Throwaway local repositories only; no network, no native build.",
    },
    "tests/test_basis_fetch.py": {
        "rationale": "Sub-second guard on the on-demand BSE basis renderer: cache-location precedence, the optional-extra error, the .g94 / .ecp split, atomic writes, version-keyed cache invalidation, and the refusal that keeps a record not covering the requested elements out of the cache. Most cases use a stand-in catalogue so no optional distribution is needed; two skip without the [bse] extra. No SCF.",
    },
    "tests/test_naming_solids.py": {
        "rationale": "Composition, dimensionality, occupancy and symmetry contracts for solid naming",
    },
    "tests/test_example_queue_checkouts.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "regression-suite/cross-code",
        "disposition": "demote T2",
        "rationale": "Hermetic example job staging and shell dispatch against a separate queue checkout",
    },
    "tests/test_symmetry_ao.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "verified inventory or parity coverage retained post-cut",
    },
    "tests/test_symmetry_core.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "verified inventory or parity coverage retained post-cut",
    },
    "tests/test_symmetry_integrals.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "basis/integrals",
        "disposition": "demote T2",
        "rationale": "verified inventory or parity coverage retained post-cut",
    },
    "tests/test_symmetry_integrals_reduced.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "basis/integrals",
        "disposition": "demote T2",
        "rationale": "verified inventory or parity coverage retained post-cut",
    },
    "tests/test_symmetry_lattice.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "verified inventory or parity coverage retained post-cut",
    },
    "tests/test_symmetry_lattice_c.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "verified inventory or parity coverage retained post-cut",
    },
    "tests/test_symmetry_salc.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": "verified inventory or parity coverage retained post-cut",
    },
    "tests/test_symmetry_shared.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Shared bounded native group/cocycle and semilinear block transport, "
            "orbit stabilizers and adjoints, explicit evidence boundaries, SALC "
            "rotation-quotient and common AO radial-layout consumers, and "
            "source-distinct molecular/chi diagnostics. No production "
            "symmetry reduction or gradient admission."
        ),
    },

    "tests/test_basis_alias_parity.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "basis/integrals",
        "disposition": "demote T2",
        "rationale": (
            "Sub-second parity guard between registry.toml's alias table and "
            "the C++ basis_data_name, which resolved names independently and "
            "disagreed (#743). Constructs tiny water bases; no SCF."
        ),
    },
    "tests/test_basis_overlay_freshness.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "basis/integrals",
        "disposition": "demote T2",
        "rationale": (
            "Sub-second guard that the resolved basis library is not a stale "
            "build overlay shadowing newer committed data. Pure path/size "
            "logic plus one read of the resolved def2-svp; no SCF."
        ),
    },
    "tests/test_bz_integration_gilat_scf.py": {
        "maturity": "verified",
        "owner": "BIPOLE",
        "rationale": (
            "Open-shell Gilat and Aufbau fixed points retain absolute energy "
            "pins after the bounded Ewald split (#708). BIPOLE lane membership "
            "must preserve this file's existing verified maturity."
        ),
    },
    "tests/test_bipole_finite_source.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny common selected SR/LR/zero-mode BIPOLE integral composition "
            "and independent Gaussian density-response witnesses. This finite "
            "source does not certify a matched production HF Hamiltonian, "
            "space-group reduction or DLPNO-CCSD(T) energy."
        ),
    },
    "tests/test_periodic_sap.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "under-review inventory or parity coverage retained post-cut",
    },
    "tests/test_bipole_erfc_panel.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny native real-space erfc quartet slabs, independent Gaussian "
            "moments and explicit finite-support/Bloch recontractions. Raw "
            "integrals do not certify a matched BIPOLE HF Hamiltonian or "
            "symmetry-reduced correlated energy."
        ),
    },
    "tests/test_bipole_physical_products.py": {
        'maturity': 'under-review',
        'tier': 'T2',
        'owner': 'periodic-SCF',
        'disposition': 'demote T2',
        'rationale': 'Tiny exact pair-distance supports, independent-product native Ewald Gram and Fourier/overlap symmetry witnesses. No complete production HF Hamiltonian or correlated energy is certified.',
    },
    "tests/test_bipole_ewald_gram.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny native explicit-cell Fourier and reciprocal Ewald selected "
            "integral witnesses. Independent Gaussian moments and J/K "
            "recontractions do not qualify a complete BIPOLE Hamiltonian, "
            "SCF reference, symmetry reduction or correlated energy."
        ),
    },
    "tests/test_periodic_orbital_sewing.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny native one-Seitz orbital-subspace witnesses with independent "
            "complex metric, gauge, core-mask, stationarity and resource checks. "
            "Numerical state sewing is not physical BIPOLE-source admission "
            "or symmetry-reduced CCSD/triples energy validation."
        ),
    },
    "tests/test_periodic_ao_bloch_transport.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny native Seitz AO coefficient transport checked against "
            "real-torus Fourier and Gaussian point-value oracles, including "
            "nonsymmorphic phases, shifted meshes, pure/Cartesian shells "
            "and bounded resources. No SCF or correlated-energy run."
        ),
    },
    "tests/test_ecp_correlated.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "molecular correlation",
        "disposition": "demote T2",
        "rationale": (
            "PySCF-parity pins for MP2, CCSD, UMP2, ROHF and CISD on LANL2DZ "
            "ECP references (H2S / SH sized), the ECP-aware frozen-core "
            "convention, the lifted run_job routes, and the BasisSet "
            "per-atom coverage guard. Skips without pyscf."
        ),
    },
    "tests/test_basis_registry.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "basis/integrals",
        "disposition": "demote T2",
        "rationale": (
            "Pins the basis registry: per-element ECP decision (sidecar "
            "first, family rules second, never the name), the one "
            "default-fit table shared by molecular and periodic callers, "
            "and inline sidecar attachment on AgH-sized SCFs. Executed "
            "reproductions of the 2026-09-05 review defects."
        ),
    },
    "tests/test_basis_qvf_parity.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "basis/integrals",
        "disposition": "demote T2",
        "rationale": (
            "Text-only parity between every runtime .g94/.ecp pair and its "
            "QVF sidecar (elements, ECP elements, core counts, links). Sub-"
            "second; guards the library data, not a method."
        ),
    },
    "tests/test_hermitian_jacobi.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny independent Hermitian eigensystem residual, orthogonality, "
            "conjugation and numerical-boundary checks for the allocation-"
            "transparent native Jacobi reference used by periodic correlation. "
            "No SCF or target-sized matrix runs."
        ),
    },
    "tests/test_pair_natural_orbitals.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "molecular correlation",
        "disposition": "demote T2",
        "rationale": (
            "Pins native restricted spin-adapted pair density and PNO "
            "selection against the defining matrix equations, with explicit "
            "pair multiplicity, zero-cutoff completeness and bounded memory. "
            "This is local algebra, not an end-to-end periodic DLPNO method."
        ),
    },
    "tests/test_pbc_pair_complete_consumers.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "release/test-health",
        "disposition": "demote T2",
        "rationale": (
            "Physical one-electron, nuclear and quartet domains are aligned "
            "by integer cell key, including wider exchange output support. "
            "Independent Gaussian sums, image relabelling, density and nuclear "
            "finite differences, and coupled Ewald split checks cover their "
            "consumers. Native regression anchors follow reviewed grid and "
            "Schwarz corrections; independent periodic references are retained. "
            "Finite-domain checks do not establish infinite-periodic "
            "convergence or approve a default change (#429/#724)."
        ),
    },
    "tests/test_periodic_correlation_metric_factorization.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Pins the admitted native one-q principal metric inverse square "
            "root, original auxiliary AO rows, rank/negative cutoffs, state "
            "and payload provenance, consumed ownership, memory lifetimes, "
            "and block/conjugation covariance against tiny NumPy oracles. "
            "No target-sized matrix, SCF, factor store or energy runs."
        ),
    },
    "tests/test_periodic_correlation_three_center.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny physical all-reciprocal three-center tiles checked against "
            "independent Gaussian moments and spectral whitening, including "
            "q gauges, canonical ket phases, tiling, seals and allocation "
            "admission. Finite-image reference only; no SCF or energy runs."
        ),
    },
    "tests/test_periodic_correlation_factor_store.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny physical factor stream to exclusive preallocated private "
            "scratch, checked exact codec offsets, sequential completion, "
            "verified random reads, corruption/cancellation and resource "
            "admission. No restart or target-sized factor-store claim."
        ),
    },
    "tests/test_periodic_correlation_wannier.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny exact finite-torus occupied gauge/Fourier transport with "
            "home-cell-only storage, explicit frozen masks, translation and "
            "time-reversal checks, and native numerical byte admission. "
            "Not a localization optimizer or end-to-end correlated energy."
        ),
    },
    "tests/test_periodic_correlation_occupied_fock.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny physical occupied Fock projections and finite-torus "
            "translation blocks, with independent placed-Wannier checks, "
            "frozen masks, complex gauge covariance and bounded workspace. "
            "No eigenvalue substitution or correlated-energy claim."
        ),
    },
    "tests/test_periodic_correlation_pao_domain.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny native retained-virtual PAO domain projections against "
            "explicit finite-torus matrices, including full and reduced SCF "
            "rank, frozen masks, Hermiticity, time reversal, memory and "
            "provenance. No full domain selection or correlation-energy run."
        ),
    },
    "tests/test_periodic_correlation_pao_space.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Tiny PAO canonical orthogonalization and selected-space Fock "
            "diagonalization, audited independent metric and retained "
            "projector identities, explicit rank/negative controls and "
            "two-stage bounded-memory admission. No energy run."
        ),
    },
    "tests/test_periodic_correlation_local_factors.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Selected occupied/PAO coefficient panels and bounded one-q "
            "global-auxiliary local factors, checked against independent "
            "Gaussian reciprocal projectors and finite-torus Fourier "
            "oracles. Pins complex orientation, normalization and caps."
        ),
    },
    "tests/test_periodic_correlation_pair_residual.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded real target-pair MP2 residual and streamed inter-pair "
            "projections, checked against dense full-space Fock and linear "
            "MP2 algebra with unequal domains, source orientations and "
            "fail-closed work/ownership gates. Not an iterative solver."
        ),
    },
    "tests/test_periodic_correlation_density_factors.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native bounded occupied/occupied and virtual/virtual RI "
            "density factors with charged densities and independent "
            "Gaussian reciprocal oracles, orientations and byte/work caps."
        ),
    },
    "tests/test_periodic_correlation_pair_integrals.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native all-q complex pair Coulomb assembly from bounded "
            "physical VO/OV factors, independent reciprocal-kernel checks, "
            "partial blocks and outer-plus-inner live memory admission."
        ),
    },
    "tests/test_periodic_correlation_diabatic_seed.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded native diabatic starting gauges, independent dense "
            "Cholesky/SVD checks, physical time-reversal sewing at all even "
            "mesh self-inverse points and explicit frozen/rank/work gates."
        ),
    },
    "tests/test_periodic_correlation_pao_overlap.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded native cross-domain PAO overlaps, independent finite "
            "tori with complex/reduced spaces, translations, actual-owner "
            "inventory deduplication and strict work/byte admission."
        ),
    },
    "tests/test_bounded_restricted_triples_target.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded target triples residual with all occupied-Fock "
            "couplings and explicit supplied moments; independent tensor "
            "projections, linear solve and canonical energy contractions. "
            "Not a moment producer or converged periodic triples driver."
        ),
    },
    "tests/test_bounded_restricted_ccsd_target.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded native real spatial target CCSD residual, checked at "
            "arbitrary amplitudes against spatial and spin-orbital algebra. "
            "Pins extended-domain coupling, callback and memory/work caps. "
            "Internal arithmetic leaf, not a periodic energy method."
        ),
    },
    "tests/test_bounded_restricted_ccsd_solver.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded native local-domain CCSD reference iteration with "
            "same-snapshot residual/energy checks, independent spin-orbital "
            "fixed points, nonzero singles, progress and strict resource caps. "
            "Not the scalable pair-specific periodic DLPNO driver."
        ),
    },
    "tests/test_periodic_correlation_local_orbital_factors.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Unified native local OO/OV/VO/VV factor panel in one verified "
            "AO tile traversal, independent Gaussian reciprocal oracles, "
            "complex phases, frozen/reduced spaces and exact live inventory."
        ),
    },
    "tests/test_periodic_correlation_bloch_iao.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native retained-space Bloch IAO algebra, original-AO projector "
            "oracles, full frozen-plus-active reconstruction, Schur/rank "
            "admission and exact bounded numerical inventory."
        ),
    },
    "tests/test_periodic_correlation_real_local_basis.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual expanded local-orbital time reversal at every k/trim, "
            "finite-torus metric and direct Fock projections with explicit "
            "real-conversion error budgets and checked live memory."
        ),
    },
    "tests/test_periodic_gaussian_source_context.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "SCF-independent native Gaussian source context: actual basis "
            "content, exact mesh, direct/reciprocal geometry and compiled "
            "finite Hamiltonian policies; bounded census and canonical wire."
        ),
    },
    "tests/test_periodic_gaussian_reciprocal_source.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "State-independent finite reciprocal source with native context, "
            "actual record seals and q/conjugate closure; unchanged v1 "
            "enumeration checks plus exact count/wire/storage admission."
        ),
    },
    "tests/test_periodic_correlation_iao_optimizer.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native TR-constrained IAO PM reference steepest ascent and "
            "Armijo search: Riemannian convergence, all-TRIM orbital "
            "audits and explicit accepted-state resource/limit semantics."
        ),
    },
    "tests/test_periodic_gaussian_metric.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Density-independent native Gaussian raw metric and consuming "
            "principal whitener, shared finite-source numerical kernels, "
            "legacy bit parity and explicit current resource admission."
        ),
    },
    "tests/test_periodic_gaussian_three_center.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Density-independent native Gaussian three-center tiles from "
            "actual context, reciprocal source and whitener, independent "
            "Gaussian oracles and explicit image/work/lifetime admission."
        ),
    },
    "tests/test_periodic_gaussian_bloch_iao.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native bounded Gaussian overlap to Bloch IAO connection, "
            "actual cross-adjoint, full-mesh TR and S11 numerical audits, "
            "with source receipts distinct from HF authentication."
        ),
    },
    "tests/test_periodic_gaussian_localization.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Connected native Gaussian-to-IAO localization and Wannier "
            "construction, preserving accepted-iterate failure semantics "
            "and whole-driver resource/source accounting."
        ),
    },
    "tests/test_periodic_correlation_occupied_pao_domain.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native signed Mulliken and one-step PAO-tail occupied domains, "
            "typed localization lineage, exact finite-torus phases and bounded resources."
        ),
    },
    "tests/test_periodic_gaussian_occupied_pao_domain.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual native HF-origin occupied domains with original Gaussian "
            "input authentication and internally derived AO atom labels."
        ),
    },
    "tests/test_periodic_correlation_real_pao_embedding.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native common-to-pair PAO embedding with original AO metric "
            "containment and Fock projection error gates."
        ),
    },
    "tests/test_periodic_correlation_pair_pao_domain.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native translation-unique initial pair PAO unions with exact "
            "torus routing, sealed occupied domains and bounded owner accounting."
        ),
    },
    "tests/test_periodic_gaussian_pair_pao_domain.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual-HF translated pair domains with native shell atom mapping, "
            "Gaussian input receipts and full phase admission."
        ),
    },
    "tests/test_periodic_gaussian_embedded_pair_pnos.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual-source pair-frame MP2 density and PNO generation before "
            "common-frame export, with explicit projection errors and dimensions."
        ),
    },
    "tests/test_periodic_gaussian_pair_domain_builder.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Native actual-HF domain builder and streamed physical pair embeddings.",
    },
    "tests/test_periodic_gaussian_embedded_pair_space.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Tagged native generation frames and common-frame pair integrals and overlaps.",
    },
    "tests/test_periodic_gaussian_domain_pair_mp2.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Streamed actual selected pair domains through coupled MP2 with original diagonal embeddings.",
    },
    "tests/test_periodic_gaussian_domain_pair_ccsd_t.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Selected-domain MP2 through independent singles, CCSD, TNOs and occupied-coupled triples.",
    },
    "tests/test_periodic_gaussian_selected_domain_pair_ccsd_t.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "One native actual-HF selected-domain entry through pair CCSD and coupled triples.",
    },
    "tests/test_periodic_gaussian_local_orbital_factor_blocks.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Bounded rectangular actual-source factor panels with exact full-panel assembly and unchanged source convention.",
    },
    "tests/test_bounded_restricted_pair_ccsd_interaction.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Native mixed-PNO semi-joint CCSD contribution with independent equation, projection and resource tests.",
    },
    "tests/test_bounded_restricted_pair_ccsd_particle_hole.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Streamed complete bare particle-hole group with independent common-space projection oracle and strict source inventory.",
    },
    "tests/test_bounded_restricted_pair_ccsd_solver_payload.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Final ragged CCSD snapshot verification with independent payload codec and bounded validation work.",
    },
    "tests/test_bounded_restricted_ccsd_target_split.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Bare particle-hole exclusion with independent residual reconstruction and exact callback savings.",
    },
    "tests/test_bounded_restricted_pair_ccsd_solver_split.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Exactly-once bare particle-hole replacement at immutable CCSD snapshots with explicit producer phases and failure gates.",
    },
    "tests/test_periodic_gaussian_pair_ccsd_split.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Actual pair-space CCSD split integration against a native scalar reference, with unqualified physical totals blocked.",
    },
    "tests/test_periodic_gaussian_pair_particle_hole_snapshot.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Actual all-q particle-hole groups at current numerical snapshots with exact native coefficient storage and source inventories.",
    },
    "tests/test_periodic_gaussian_pair_ccsd_physical_particle_hole.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Native all-q Gaussian particle-hole replacement during CCSD and continuation through TNO and coupled triples, with source and resource gates.",
    },
    "tests/test_periodic_gaussian_selected_physical_particle_hole_ccsd_t.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Single native actual-HF call through selected pair MP2, source-qualified all-q CCSD, TNOs and triples with complete source lifetimes and admission.",
    },
    "tests/test_periodic_gaussian_pair_ccsd_owner_lifetime.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Equal-content native basis replacement during CCSD callbacks must reject stale borrowed Fock storage before dereference.",
    },
    "tests/test_periodic_gaussian_retained_pair_geometry.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Persistent native pair generation geometry, owner lifetime and complete downstream numerical inventories.",
    },
    "tests/test_periodic_gaussian_pair_particle_hole.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Physical multi-k pair interactions contracted against a frozen CCSD snapshot with independent bare residual oracle.",
    },
    "tests/test_periodic_gaussian_mixed_pair_factors.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Actual Gaussian factors for independent PNO frames without a joint orthonormality assumption.",
    },
    "tests/test_periodic_gaussian_pair_interaction.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": "Streamed all-q mixed pair integrals and original-S overlap for a native local CCSD interaction.",
    },
    "tests/test_periodic_gaussian_one_electron.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded native Gaussian Bloch overlap and kinetic integrals, "
            "original normalization, independent Hermiticity/time-reversal "
            "audits and explicit finite-image work admission."
        ),
    },
    "tests/test_periodic_gaussian_nuclear.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native finite Ewald nuclear attraction and repulsion from "
            "actual nuclei, independent Boys and Gaussian integral oracles, "
            "with bounded image traversal and source receipts."
        ),
    },
    "tests/test_periodic_gaussian_fock.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Streamed native full-k Gaussian J/K response, independent "
            "dense actual-factor contractions and explicit memory/work "
            "admission without a converged-HF claim."
        ),
    },
    "tests/test_bounded_periodic_rhf_solver.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native full-k numerical RHF iteration with same-density "
            "energy and physical residual checks, bounded provider work, "
            "and last-evaluated-state limit semantics."
        ),
    },
    "tests/test_periodic_gaussian_rhf.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Connected actual Gaussian S/T/V/J/K to native full-k RHF, "
            "with final occupied-density Fock reconstruction and independent "
            "mean-field capture; tiny finite-source reference only."
        ),
    },
    "tests/test_periodic_gaussian_local_orbital_factors.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual Gaussian-HF-origin all-orientation local factor panels, "
            "bounded native AO streaming and independent finite-torus contractions."
        ),
    },
    "tests/test_periodic_gaussian_real_local_provider.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Matched finite-Gaussian HF recipe feeding audited real factor rows, "
            "without a full AO store or a production local-auxiliary claim."
        ),
    },
    "tests/test_periodic_gaussian_density_gram.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual-source selected-density Gram blocks with bounded q/qbar "
            "slabs, independent all-q contractions and no retained factor store."
        ),
    },
    "tests/test_periodic_gaussian_gram_pair_pnos.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual PAO-domain Gram PNO seeds and consuming pair spaces, "
            "with original-Fock rotation and no common factor-row provider."
        ),
    },
    "tests/test_periodic_gaussian_gram_pair_mp2.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual HF/domain-to-Gram-PNO coupled MP2 without a common factor "
            "provider, plus independent projected equations and source/memory guards."
        ),
    },
    "tests/test_periodic_gaussian_selected_local_ccsd_t.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual native Gaussian HF through localization, selected real PAOs, "
            "same-recipe factors and CCSD plus coupled triples on tiny meshes. "
            "Per-cell energies require complete finite-torus basis coverage."
        ),
    },
    "tests/test_periodic_gaussian_pair_pnos.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual-source finite-periodic pair PNOs, explicit density normalization, "
            "truncation and original-Fock recanonicalization with bounded storage. "
            "No coupled-MP2 or production DLPNO energy claim."
        ),
    },
    "tests/test_periodic_gaussian_pair_space.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Actual-source streamed projected pair integrals and certified common-basis "
            "overlaps with explicit ownership transfer and roundoff budgets."
        ),
    },
    "tests/test_bounded_restricted_pair_mp2_solver.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Ragged pair-MP2 reference iteration against independent projected linear "
            "systems, same-snapshot convergence and bounded on-demand overlaps."
        ),
    },
    "tests/test_periodic_gaussian_pair_mp2.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Native actual-HF-origin pair PNO generation, projected integrals and "
            "coupled MP2 reference with whole-call admission and tiny canonical oracles."
        ),
    },
    "tests/test_bounded_restricted_pair_ccsd_amplitudes.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Bounded native ragged singles/doubles common-frame accessors with distinct "
            "singles ranks, explicit immutable owners and independent expansion oracles."
        ),
    },
    "tests/test_bounded_restricted_ccsd_energy.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Scalar accessor CCSD energy with the full disconnected singles product, "
            "exact callback census and independent spatial/spin-orbital checks."
        ),
    },
    "tests/test_bounded_restricted_pair_ccsd_solver.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Ragged Galerkin CCSD reference iterator preserving common-frame contractions "
            "before target projection; not a production DLPNO energy certificate."
        ),
    },
    "tests/test_periodic_gaussian_pair_ccsd.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Actual finite-source MP2 pair frames and independently generated singles PNOs "
            "feed native Galerkin CCSD with immutable provenance and whole-call admission."
        ),
    },
    "tests/test_periodic_gaussian_frozen_core_correlation.py": {
        "maturity": "under-review", "tier": "T2", "owner": "periodic-SCF", "disposition": "demote T2",
        "rationale": (
            "Actual per-k frozen masks through HF, active localization, PAOs, pair MP2 "
            "and CCSD/triples, with independent core normal ordering and tiny FCI."
        ),
    },
    "tests/test_periodic_correlation_real_pao_space.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native explicitly real PAO preparation, interval-qualified "
            "rank and original complex operator audits, actual expanded "
            "time reversal and degenerate-space regression."
        ),
    },
    "tests/test_periodic_correlation_real_local_ccsd_t.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native finite Gaussian real-row provider to CCSD and coupled "
            "triples connection, explicit Fock projection and sequential "
            "resource/work admission, independent small-system oracles. "
            "Not a Hamiltonian-matched or production per-cell DLPNO driver."
        ),
    },
    "tests/test_periodic_correlation_real_local_provider.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native finite-source real rows from actual q/aux panels, "
            "density-reversal and conjugacy gates, all-orientation ERI "
            "error bounds and independent Decimal scalar-dot checks."
        ),
    },
    "tests/test_periodic_correlation_iao_pm.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "All-k native unorthogonalized IAO p4 charges and explicit "
            "real Euclidean gradient, independent finite-torus and "
            "directional-difference checks, active/core and memory gates. "
            "An objective leaf, not a converged localization optimizer."
        ),
    },
    "tests/test_bounded_restricted_triples_moments.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native bounded restricted W/U triples moments, all six "
            "permutations, exact Brillouin-reference gate and independent "
            "canonical/spin-orbital energy checks with memory/work caps."
        ),
    },
    "tests/test_bounded_restricted_triples_solver.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded common-space coupled triples reference iteration: "
            "all occupied Fock legs, independent linear solves and "
            "canonical spin-orbital rotation checks, exact inventory. "
            "Not a target-size TNO storage or periodic energy driver."
        ),
    },
    "tests/test_bounded_restricted_local_triples_solver.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded ragged occupied-coupled triples with supplied moments, "
            "independent common-frame Galerkin linear solves, repeated-axis "
            "covariance, zero-rank neighbours and explicit resource gates. "
            "Not a physical moment producer or production DLPNO driver."
        ),
    },
    "tests/test_bounded_restricted_triple_natural_orbitals.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Bounded union-based triple natural orbitals from connected "
            "CCSD pair amplitudes, independent published density/SVD "
            "oracles, repeated-edge multiplicities and original-Fock "
            "semicanonicalization. No physical source or triples energy claim."
        ),
    },
    "tests/test_bounded_restricted_local_triples_moments.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native amplitude-first TNO projection and streamed triples "
            "moments, independent dense preprojection and truncation "
            "witnesses, exact query/resource accounting and source guards. "
            "A numerical adapter, not a production periodic energy driver."
        ),
    },
    "tests/test_periodic_gaussian_triple_spaces.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual finite-Gaussian CCSD connected doubles to immutable "
            "triple natural-orbital batches; independent density/projector "
            "oracles, multi-k/frozen-core/source/resource gates. "
            "Geometry only, not a triples-energy driver."
        ),
    },
    "tests/test_periodic_gaussian_local_triples.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Actual finite-Gaussian CCSD/TNO amplitudes to occupied-coupled "
            "triples and complete-common-basis total energy. Independent "
            "linear, spin and preprojected-moment oracles with frozen-core, "
            "source, convergence, lifetime and complete-owner resource gates."
        ),
    },
    "tests/test_periodic_gaussian_selected_local_pair_ccsd_t.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "One native HF-origin selected-space workflow through pair-PNO "
            "MP2, Galerkin CCSD, union TNOs and occupied-coupled triples. "
            "Tiny multi-k/frozen-core and truncation checks, early-stop "
            "semantics, live progress and whole-workflow admission. "
            "Explicit experimental finite-source reference, not production DLPNO."
        ),
    },
    "tests/test_periodic_aopair_cross_fourier_panel.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Native bounded rectangular two-basis AO Fourier panels, "
            "independent Gaussian moments and cross overlaps, exact "
            "same-basis parity, Bloch phases and image/byte admission."
        ),
    },
    "tests/test_periodic_aopair_fourier_panel.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "Pins bounded native AO-pair Fourier tiles against analytic "
            "Gaussian moments and native overlap, including finite image "
            "cutoffs, explicit ket phases, AO normalization and tile caps. "
            "The image enumeration is a reference numerical policy, not "
            "yet the certified production correlation-factor source."
        ),
    },
    "tests/test_seccm_scc_dftb_basin.py": {
        "maturity": "experimental",
        "tier": "T3",
        "owner": "semiempirical/seccm",
        "disposition": "keep",
        "rationale": (
            "pins GitLab #302: the one rocksalt(100) row that "
            "run_scc_dftb_seccm still returns with converged = True and no "
            "diagnostic set -- Mg/O L3 at electronic_temperature = 0.005, "
            "whose frontier holds 1.337 electrons in the LUMO and the same "
            "in the HOMO, so the applied occupation cannot resolve the two "
            "at all. The gap guard cannot see it: that guard compares the "
            "gap to an absolute 1e-8 Ha epsilon and this row's gap is "
            "2.44e-8 Ha, 2.4x the epsilon but 5e-6 of the smearing width. "
            "The file also records the filed charge premise as refuted "
            "rather than gated -- the overlap is positive definite, tr(DS) "
            "is exact and the density is N-representable, so the Mg "
            "excursion is the Mulliken 1955 Sec. 3 partitioning artefact, "
            "not a wrong metric -- which is why no |q| <= n_val gate exists "
            "to test. Four sub-second runs. T3/experimental to match every "
            "other SECCM file (EXPERIMENTAL_PATTERNS 'tests/test_seccm*.py' "
            "derives the same values); pinned here so the rationale for a "
            "wrong-answer regression is not regenerated into boilerplate."
        ),
    },
    "tests/test_periodic_exchange_gauge_band_folding.py": {
        "maturity": "production",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "non-blocking T2",
        "rationale": (
            "pins GitLab #560: which exact-exchange q -> 0 gauge the EWALD_3D "
            "HF route uses at each mesh, and that a Gamma mesh and a real mesh "
            "band-fold once both are on the corrected (exxdiv='ewald') one. "
            "Each side is anchored to an out-of-process PySCF 2.14.0 number, "
            "so the file catches a gauge regression that no vibe-qc-internal "
            "comparison can see -- the pre-fix tree was self-consistent within "
            "each mesh and wrong only across them. Non-blocking T2 rather than "
            "T1 because it is ~10 s of H2/STO-3G SCF whose failure mode is a "
            "convention drift on one periodic route, not a build or API "
            "contract; the slow (2,2,2)-supercell half of the same identity "
            "stays in the T3 AICCM lane it was found in."
        ),
    },
    "tests/test_out_live_scf_rows.py": {
        "maturity": "production",
        "tier": "T2",
        "owner": "molecular SCF",
        "disposition": "non-blocking T2",
        "rationale": (
            "pins GitLab #482's two-sided contract: a SIGTERM'd molecular SCF "
            "keeps the iteration rows it had already completed in its .out, "
            "and a completed run's .out still carries exactly ONE iteration "
            "table. The second half is the one that needs a test -- the live "
            "rows and write_scf_trace both render correct energies, so a "
            "duplicated table is invisible except by counting headers. The "
            "kill case spawns a subprocess (an in-process SIGTERM would take "
            "pytest with it) and is marked slow: ~12 s of uracil-dimer "
            "cc-pVDZ RHF before the signal. Non-blocking T2 because the "
            "failure is observability on an already-failed job, not a wrong "
            "number -- but it guards the artifact an operator opens first "
            "for exactly the runs where nothing else survived."
        ),
    },
    "tests/test_msindo.py": {
        "rationale": (
            "pins GitLab #538 and #660: the direct closed-shell MSINDO-NDDO analytic "
            "gradient differentiates the exact shipped energy, including the "
            "HSP core, dipole-monopole, dipole-dipole, and Al-Cl SPDD terms. "
            "Source energy/gradient and Fock-branch fixtures, a three-step "
            "finite-difference ladder, translation, rotation, permutation, "
            "charge, and spin controls prevent a finite but incomplete energy "
            "or derivative from being reported again."
        ),
    },
    "tests/test_msindo_cpp.py": {
        "rationale": (
            "pins the public/native half of GitLab #538 and #660: converged "
            "closed-shell NDDO results expose the validated analytic gradient "
            "lazily across H/Li-F/Na-Cl, and Al-Cl source-energy plus "
            "native/Python parity controls require the SPDD branches on every "
            "production energy path."
        ),
    },
    "tests/test_neb_msindo.py": {
        "rationale": (
            "pins MSINDO reaction-path routing, including the GitLab #538 "
            "NDDO end-to-end control: method='nddo' must evaluate its NDDO "
            "energy and analytic-gradient surface and must never fall through "
            "to the older special INDO finite-difference evaluator; its QVF "
            "must also retain the NDDO-specific Dewar-Thiel and Voigt citations."
        ),
    },
    "tests/test_changelog_released_sections.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "release",
        "disposition": "blocking T1",
        "rationale": (
            "pins the GitLab #229 guard: every released CHANGELOG.md section must "
            "match its SHA-256 pin in scripts/test_gate/changelog_pins.toml, so an "
            "entry for unreleased work cannot reach a released section through a "
            "rebase unnoticed, as it did in #227. Plain text and hashing, plus "
            "throwaway git repositories for the audit command and the pre-push "
            "hook; the test imports nothing from vibeqc and runs no SCF. Blocking "
            "T1 because a candidate's release notes are part of what it proves."
        ),
    },
    "tests/test_tddft_davidson.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "molecular correlation",
        "disposition": "blocking T1",
        "rationale": (
            "pins the iterative (matrix-free) TDA path against the dense "
            "reference it replaces above TDA_DAVIDSON_MIN_DIM. Three layers "
            "have to hold: the matrix-free operator must equal the assembled "
            "Casida A times a vector, its analytic diagonal must equal A's "
            "diagonal (it is supplied rather than probed, so it is not "
            "self-checking), and the iterative lowest roots must be the dense "
            "lowest roots. Blocking T1 because a silently wrong excitation "
            "energy is a wrong-answer class and the two paths are chosen by a "
            "dimension threshold, so a defect would surface only on large "
            "systems where no dense cross-check is affordable. Also guards "
            "against null eigenvector columns, the shape GitLab #503 took. "
            "Small synthetic operators plus one sto-3g fixture; seconds."
        ),
    },
    "tests/test_bipole_per_k_band_accessor.py": {
        # Rationale only; maturity/tier/owner stay derived from the lane.
        "rationale": (
            "pins GitLab #505: vibe-qc exposed no public way to ask a "
            "periodic result for its per-k bands, so every consumer "
            "duck-typed the result object and the same class of mistake was "
            "made twice in five days, in two different campaign carriers, "
            "each time after a certified SCF and each time burning a "
            "full-node wall -- #15 indexed into a degenerate manifold, and "
            "#135 wrapped a per-k list as one pseudo-k behind a guard that "
            "counted k-points instead of bands. Runs a real converged "
            "multi-k BIPOLE SCF (H2/STO-3G, 6-bohr cube, k=(2,1,1), "
            "E = -1.1944690859739715 Ha) rather than a constructed object, "
            "because the whole defect is that the shape of a REAL result "
            "was not what a reader assumed; pins its frontier values, that "
            "PBCBipoleRHFResult still has no mo_energies_k, that the #135 "
            "wrapped shape and a ragged k-set are refused loudly, that the "
            "n_occ guard is on the band axis, and that a degenerate "
            "manifold reduces to a scalar. Seconds-scale SCF."
        ),
    },
    "tests/test_trexio_reference_gate.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "output/docs",
        "disposition": "non-blocking T2",
        "rationale": (
            "Pins tests/trexio_reference.py, the shared locator for the "
            "out-of-process TREXIO reference interpreter (#253): without "
            "VIBEQC_TREXIO_PYTHON the convention checks skip with a reason "
            "naming the gate that did not run, and fail instead when "
            "VIBEQC_REQUIRE_TREXIO_REFERENCE is set. Pure Python, needs "
            "neither trexio nor pyscf; T2 with its two TREXIO siblings."
        ),
    },
    "tests/test_output_trexio_extended.py": {
        "maturity": "verified", "tier": "T2", "owner": "output/docs",
        "disposition": "non-blocking T2",
        "rationale": "TREXIO ECP Hamiltonian reconstruction, molecular and periodic READ guesses, complex k/spin blocks, density/integral contractions and all storage kinds on both optional backends.",
    },
    "tests/test_output_trexio.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "output/docs",
        "disposition": "non-blocking T2",
        "rationale": (
            "TREXIO writer + round-trip reader (GitLab #573, increment 1). "
            "Pins the AO conventions against the TREXIO specification's own "
            "worked example, exact H2O/def2-SVP RHF and OH UHF round trips "
            "on both back ends with C^T S C = 1 in the overlap rebuilt from "
            "the file's basis group, the ECP / basis-free refusals, the "
            "run_job(trexio=...) wiring with its manifest row and citation, "
            "and an out-of-process PySCF energy rebuild that needs "
            "VIBEQC_TREXIO_PYTHON to name an interpreter with trexio + pyscf: "
            "it skips naming the gate that did not run, or fails when "
            "VIBEQC_REQUIRE_TREXIO_REFERENCE is set (#253). "
            "output-docs lane, owner output/docs, like every sibling output "
            "writer test; T2 because the artefact is opt-in and no default "
            "output changes."
        ),
    },
    "tests/test_output_artifact_stem_dot_safety.py": {
        # Rationale only -- maturity/tier/owner stay derived from the
        # output-docs lane, matching every sibling in that family. This
        # defect is a strong T1 candidate (it silently destroys results on
        # runs that exit 0), but promoting it would make this the one T1
        # file in an all-T2 family and would widen the release gate; that
        # call belongs to the output/IO owning chat, and is raised to them
        # in handovers/HANDOVER_OUTPUT_LOGGER.md rather than taken here.
        "rationale": (
            "pins GitLab #254: output= is a documented path STEM, but every "
            "artifact path was derived with Path.with_suffix, which "
            "substitutes rather than appends. Any stem carrying a dot -- a "
            "lattice constant, a scale factor, a tolerance -- truncated at "
            "that dot, so a nine-point EOS ladder declared ONE .out path "
            "instead of nine (measured at the parent) and a 24-member bundle "
            "kept 6 .out files. The jobs exit 0 and nothing warns; the loss "
            "is visible only by counting files, which is why it recurred "
            "three times and, on the third, turned the resulting "
            "FileNotFoundError into a confident and wrong route-mismatch "
            "verdict on seven healthy calculations. Pins the plan-level "
            "declaration, injectivity of every artefact role across a real "
            "EOS ladder, two real SCF runs that must not overwrite each "
            "other, the dot-free negative control, the .opt.xyz "
            "compound-name no-op the periodic runner depends on, and a "
            "source guard over the three files that name artefacts -- the "
            "substitution idiom is one keystroke from returning at any of "
            "the ~87 call sites the fix converted."
        ),
    },
    "tests/test_gradient_missing_terms.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "molecular/DFT",
        "disposition": "blocking T1",
        "rationale": (
            "pins the GitLab #571 fail-closed / fall-back contract: the "
            "analytic RKS/UKS kernel omits the range-separated exact-exchange "
            "gradient (cam_beta*dK_erf/dR) and the VV10 nonlocal gradient, so "
            "compute_gradient_rks / _uks refuse for hse06 / vv10 / wb97x-v "
            "unless allow_incomplete=True, and every molecular optimizer "
            "(L-BFGS-B, Brent, geomopt provider, ASE calculator) walks the "
            "full-energy central-FD surface for such functionals. Measured "
            "before the fix on O/H/H STO-3G: 5.6e-3 / 7.0e-4 / 5.6e-2 Ha/bohr "
            "off FD against a 2.1e-6 PBE noise floor, and run_job(optimize=True) "
            "used the wrong surface silently. Blocking T1 because it is a "
            "wrong-answer defect on a production route, with a same-route PBE "
            "negative control."
        ),
    },
    "tests/test_aux_basis_coverage.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "basis/integrals",
        "disposition": "blocking T1",
        "rationale": (
            "pins the GitLab #480 fail-closed guard: density fitting must "
            "refuse an auxiliary basis that contributes zero functions to a "
            "centre carrying orbital functions, on both routes -- the SCF JK "
            "route (density_fit=True) and the post-SCF RI route, which the "
            "DLPNO methods and run_rohf_mp2 reach with no density_fit flag "
            "at all -- through both run_rhf and run_job. Both entry points "
            "are pinned because runner binds the bare C++ SCF functions (it "
            "is imported from vibeqc/__init__.py before the Python wrappers "
            "are defined), so a guard placed only in the wrappers passes the "
            "run_rhf case while leaving run_job -- the entry point the issue "
            "was filed against -- silently wrong. Pre-fix, libint2 returned "
            "an auxiliary BasisSet with no "
            "shells on an uncovered element instead of raising, so the SCF "
            "converged cleanly -- tight gradient, nothing in the .out -- to "
            "-1.15 Ha (LiH/cc-pVQZ) through -49.56 Ha (NaH/cc-pVDZ) below "
            "the conventional answer. Blocking T1 because a regression here "
            "reopens a silent converged-wrong-answer class on a supported "
            "option with no diagnostic anywhere in the output, which is what "
            "the pre-cut gate exists to catch, and because the guard is the "
            "only thing standing between an unbackfilled basis file and a "
            "published number. Also pins the positive side: the "
            "def2-universal-jkfit control still reproduces the conventional "
            "energy to ~1e-5 Ha, covered systems still run unchanged, and "
            "the eleven known shipped-data gaps stay detected so a future "
            "backfill shows up as a named failure rather than as a silent "
            "change in which molecules are refused."
        ),
    },
    "tests/test_ecp_derivative_guards.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "basis/integrals",
        "disposition": "blocking T1",
        "rationale": (
            "pins exact XML and inline molecular ECP provenance across all "
            "four public analytic-gradient wrappers, including rejection of "
            "unverified low-level Hcore results and same-count/different-"
            "Hamiltonian inputs. It also pins the remaining unsupported "
            "analytic-Hessian, ROHF, multireference, CPCM, and HF-CIS "
            "derivative boundaries plus effective-electron occupations in "
            "molecular and periodic QVF fallbacks. Blocking T1 because "
            "differentiating a bare or different Hamiltonian can emit "
            "plausible scientific wrong answers."
        ),
    },
    "tests/test_aic7_driver.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "agentic-loop/AICCM",
        "disposition": "blocking T1",
        "rationale": (
            "GitLab issue #415 AIC7 orchestration contract: incomplete wave "
            "plans are partial with a nonzero exit, budget-limited rungs "
            "retain explicit kill provenance, comma and plus method lists "
            "round-trip through stable tags, target meshes run first, and a "
            "bundled real-Gamma rung succeeds only when every method does. "
            "The SHA-bound historical overlay keeps six false-green jobs "
            "partial without modifying their sealed evidence. "
            "Also pins GitLab issue #120 since 2026-08-28: the kill deadlines "
            "are sized from the reservation vq exported rather than from a "
            "constant -- the budget defaults to the whole granted wall less "
            "the finalization margin (a 4 h grant used to be refused outright "
            "by the 18 h constant, and a 24 h grant left 6 h unspent), the "
            "per-rung cap becomes a floor each rung may exceed up to its share "
            "of what is left (10800 s -> 28680 s on a 3-rung 24 h ladder), an "
            "explicit flag still pins a hard value, no reservation leaves both "
            "historical constants exact, and a shared allotment still cannot "
            "outlive the budget. "
            "Pure driver tests with synthetic rung outcomes; no SCF work."
        ),
    },
    "tests/test_scf_stability_phase_timing.py": {
        "maturity": "production",
        "tier": "T2",
        "owner": "molecular/DFT",
        "disposition": "demote T2",
        "rationale": (
            "GitLab issue #205 truthful stability-phase visibility: only "
            "real UHF/UKS stability work yields nonzero result timings and "
            ".out/perf phases; opt-out, nonconverged, and unsupported routes "
            "remain exact zero. Rendered loop plus stability totals also "
            "reconstruct within a bound derived from millisecond rounding. "
            "Tier remains T2 because this is performance observability, not "
            "a release-blocking scientific-result contract."
        ),
    },
    "tests/test_uks.py": {
        "maturity": "production",
        "tier": "T2",
        "owner": "molecular/DFT",
        "disposition": "demote T2",
        "rationale": (
            "GitLab issue #447 UKS stability-follow contract: a converged "
            "negative mode is followed with the complete KS energy, both "
            "signed internal descents are reconverged, and stretched "
            "H2/PBE reaches the independently seeded stable determinant "
            "instead of returning the 38.672682 mHa-higher symmetric "
            "saddle. Also pins fail-closed handling when meta-GGA, "
            "range-separated exchange, VV10, or DFT+U response terms are "
            "unavailable. Tier remains T2 because the file includes the "
            "broader cross-code UKS functional matrix."
        ),
    },
    "tests/test_progress.py": {
        "maturity": "production",
        "tier": "T2",
        "owner": "molecular/DFT",
        "disposition": "demote T2",
        "rationale": (
            "GitLab issue #25 live molecular-SCF observability contract: "
            "native RHF/UHF/RKS/UKS iteration callbacks reach the terminal, "
            "structured log, and canonical manifest heartbeat before the "
            "driver returns; reset sequences are attempt-labelled, terminal "
            "replay is exact-once, helper phases remain ordered before "
            "job_end, and callback state is isolated across nested, failed, "
            "periodic, and concurrent jobs"
        ),
    },
    "tests/test_frozen_core_convention.py": {
        "maturity": "production",
        "tier": "T2",
        "owner": "molecular correlation",
        "disposition": "non-blocking T2",
        "rationale": (
            "pins the maintainer-ratified GitLab #140 convention: native "
            "MP2/UMP2, canonical CCSD/UCCSD and closed/open-shell DLPNO "
            "routes all resolve the ORCA 6.1 Table 2.69 count-only default, "
            "instead of the former CCSD=chemical-core versus MP2/DLPNO="
            "all-electron split. It transcribes boundary elements across the "
            "published table, runs four public methods in both spin cases on "
            "def2-SVP H2O/OH, and requires the same count and convention in "
            "the .out, .system manifest and structured-log completion event. "
            "The DLPNO-MP2 legs also pin the #448 NormalPNO disclosure, "
            "including UMP2's unsupported/inactive fields. The open-shell "
            "DLPNO-CCSD leg dominates at about 40 s. Non-blocking T2 because "
            "this broad eight-calculation cross-route audit is too heavy for "
            "the pre-cut core profile; the molecular-correlation affected "
            "lane runs it on every convention-source touch."
        ),
    },
    "tests/test_runner_dlpno_ccsd.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "molecular correlation",
        "disposition": "demote T2",
        "rationale": (
            "end-to-end run_job coverage for the closed/open-shell local "
            "DLPNO-CCSD/(T) routes, including exact-pilot selection, size and "
            "auxiliary-basis guards, method citations, and the #417/#448 "
            "artifact contract. The convention sweep makes this a required "
            "consumer because runner resolves NormalPNO and frozen core, "
            "records the supported threshold subset and convention in .out "
            "and .system, and must preserve explicit option passthrough. T2 "
            "because it executes several real correlated calculations and "
            "the DLPNO implementation remains under review."
        ),
    },
    "tests/test_release_paper_dlpno_m16.py": {
        "maturity": "verified",
        "tier": "T2",
        "owner": "molecular correlation",
        "disposition": "demote T2",
        "rationale": (
            "preserves the release-paper M16 n-butane/cc-pVDZ DLPNO evidence "
            "under its exact pre-#140/#448 recipe: NoFrozenCore plus the old "
            "MP2/local-CCSD threshold values. It separately pins the new "
            "NormalPNO option defaults, so archived ORCA gaps cannot be "
            "silently relabelled as current-default validation. The real "
            "106-function MP2 comparison is slow-marked and belongs in T2; "
            "the ordinary molecular-correlation lane exercises its cheap "
            "contract and mocked artifact leg while excluding the slow run."
        ),
    },
    "tests/test_scf_max_iter_boundary.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "molecular SCF",
        "disposition": "blocking T1",
        "rationale": (
            "pins the issue #392 input envelope: every public molecular SCF "
            "route (RHF/UHF/RKS/UKS/ROHF/ROKS) preserves max_iter=0 as a "
            "coherent ORCA NOITER initial-guess evaluation with full finite "
            "result shapes, n_iter=0 and no SCF trace, while rejecting "
            "negative, boolean and non-integer budgets before driver work. "
            "It also pins one-iteration/default controls and prevents UHF/UKS "
            "spin schedules from hiding work under NOITER. Cheap H/H2 STO-3G "
            "SCFs. Blocking T1 because an empty or zero-energy NOITER result "
            "is a silent wrong-answer class."
        ),
    },
    "tests/test_xc_external_provider.py": {
        "maturity": "production",
        "tier": "T2",
        "owner": "molecular/DFT",
        "disposition": "demote T2",
        "rationale": (
            "full-grid external-XC host contract across molecular and "
            "periodic AO projection, registered-provider capability gates, "
            "ROKS first-order support, pointwise/response/gradient fail-closed "
            "behaviour, and separation of generic providers from SKALA grid, "
            "provenance, and calibrated model-memory policy"
        ),
    },
    "tests/test_periodic_external_xc_gpw.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic/DFT",
        "disposition": "new",
        "rationale": (
            "experimental full-grid external-XC routing for GPW Gamma and "
            "full-mesh multi-k RKS plus GAPW Gamma RKS: complete periodic AO "
            "density, difference-closed lattice domain, variational finite "
            "differences, reduced-mesh refusal, FFT-XC bypass, and proof that "
            "GAPW retains its Hartree augmentation without evaluating the "
            "external functional on separate smooth/hard/soft densities"
        ),
    },
    "tests/test_skala_adapter.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "molecular/DFT",
        "disposition": "new",
        "rationale": (
            "SKALA-1.1 external-XC adapter contract: immutable model identity, "
            "SHA-256-before-TorchScript trust boundary, MIT notice/provenance "
            "cache, lazy optional-PyTorch loading, callback shape validation, "
            "and integrated-energy adjoints. Scientific acceptance remains "
            "separate and requires matched molecular and periodic calculations"
        ),
    },
    "tests/test_skala_grid_profile.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "molecular/DFT",
        "disposition": "new",
        "rationale": (
            "vibe-qc SKALA parity profile against PySCF 2.14.0: exact "
            "element-specific radial and angular point counts, NWChem pruning, "
            "Treutler radial quadrature, heteroatomic Becke adjustment, stable "
            "atom-local shell-tier ordering, periodic partition-number parity, "
            "and fail-closed atomic-number range"
        ),
    },
    "tests/test_ao_convention_invariants.py": {
        "maturity": "verified",
        "tier": "T1",
        "owner": "basis/integrals",
        "disposition": "blocking T1",
        "rationale": (
            "pins the libint solid-harmonic ordering and Cartesian "
            "normalization conventions that ao_eval.cpp, the molden writer, and "
            "the QVF writer each assume independently; both are "
            "silent-wrongness failure modes. Blocking T1 because "
            "grid-vs-engine parity "
            "(test_evaluate_ao_reproduces_the_integral_basis) is the invariant "
            "every DFT XC quadrature rests on, and this defect class shipped "
            "once already (v0.15.0-v0.15.114, via the basis_toolkit Cartesian "
            "import route; handovers/HANDOVER_AO_CONVENTION.md). Measured "
            "2.9 s / 150 MB -- cheap enough for the blocking set"
        ),
    },
    "tests/test_bipole_quartet_tensor_cache_fallback.py": {
        "maturity": "production",
        "owner": "BIPOLE",
        "disposition": "new",
        "rationale": (
            "covers build_quartet_tensor_cache's pure-Python fallback, which "
            "had no test at all and shipped unable to execute. 7e94c7f18 put "
            "a function-local import of quartet_effective_screening_parameter "
            "inside the if _use_cpp_tensor branch; Python makes that name "
            "function-local for the whole body, so the else branch raised "
            "UnboundLocalError on every ewald_omega > 0 build. Reachable in "
            "practice -- any _vibeqc_core built before 1a26e6954 lacks the "
            "native symbols, so the fallback is the live path there -- but "
            "invisible, because the sole call site swallows it with "
            "except Exception: pass and silently drops the cache. The crash "
            "was fixed incidentally by 3f33caee9 (hoisting that import while "
            "fixing an unrelated mu_eff key collision); nothing pinned it, so "
            "these tests keep it fixed. Also pins store-key/lookup-key "
            "agreement through cache.get, that the probe covers every kernel "
            "the C++ branch calls, and an AST guard on the shadowing "
            "invariant itself rather than the one symptom symbol. Pure "
            "Python, no SCF, no compiler -- 0.05 s. Tier left derived (T2 via "
            "the pbc-bipole lane) rather than pinned: the crash-class failure "
            "mode and the cost argue for blocking T1 on the "
            "test_cmake_sources_exist reasoning, but widening the release "
            "gate is the owning chat's call (CLAUDE.md s11)"
        ),
    },
    "tests/test_ccm_guard_attribution.py": {
        "maturity": "experimental",
        "tier": "T3",
        "owner": "periodic/aiccm",
        "disposition": "new",
        "rationale": (
            "GitLab IID 498: every public neutral CCM route must name ITSELF "
            "when the IID 291 vacuum-padding guard fires. run_ccm_rhf_gdf and "
            "run_ccm_rks_gdf both delegate to the shared _ccm_gdf helper, "
            "which hardcoded who='run_ccm_rhf_gdf', so an rks caller was sent "
            "to a function it never called. The four unshared entry points "
            "each passed their own name and were fine, which is why the one "
            "shared path went unnoticed. A wrong function name in a "
            "fail-closed message is cheap to ship and expensive to debug. "
            "Pins all nine public neutral routes so a tenth caller added to a "
            "shared helper cannot inherit a ninth's identity. Guard-only: "
            "every case raises before any SCF runs, so the file is seconds. "
            "T3/experimental per the Gamma-CCM research lane"
        ),
    },
    "tests/test_ccm_neutral_tail_parity.py": {
        "maturity": "experimental",
        "tier": "T3",
        "owner": "periodic/aiccm",
        "disposition": "new",
        "rationale": (
            "GitLab IID 307 wrong-answer regression: the RSGDF high-|G| tail "
            "is the Hamiltonian-identity seam between run_ccm_rhf_direct and "
            "run_ccm_rhf_gdf, which are documented as evaluating the same "
            "block-circulant neutral Hamiltonian. At nrep=(1,1,1) the GDF "
            "sibling delegates to the Gamma fast path, which auto-sizes a "
            "tail on the tight-core class, while the direct/fold route had no "
            "tail_ke_cutoff plumbing at all -- so the two ran different "
            "Hamiltonians and disagreed by -4.99e-01 Ha/cell on MgO/STO-3G, "
            "with no result field reporting it. Every end-to-end gate here "
            "carries its own untailed falsifying control, because the "
            "theorem1_parity gate that should have caught this reported a "
            "pass-shaped result over an EMPTY domain and had never once "
            "returned a negative (LEARNINGS L95/L99). 15 tests, no slow "
            "marker, measured 266 s as a file: the end-to-end numerics run on "
            "fcc Be (a 2.1x reciprocal ball) and diamond (7.8x), the rest are "
            "classifier-level and free. The canonical MgO cell is pinned at "
            "the classifier level ONLY -- its end-to-end gate was written, "
            "measured, and removed: the tailed MgO fold did not finish inside "
            "33 min and ran at ~1 core throughout. Fix the unscreened fold "
            "tail before re-adding it. T3/experimental per the Gamma-CCM "
            "research lane"
        ),
    },
    "tests/test_cmake_sources_exist.py": {
        "maturity": "production",
        "tier": "T1",
        "owner": "release/test-health",
        "disposition": "blocking T1",
        "rationale": (
            "pins that every source and vibeqc header a CMakeLists names "
            "exists on disk. Blocking T1 because the failure mode is an "
            "unbuildable tree for every chat that runs a fresh configure, and "
            "ordinary main pushes run no CI to catch it (CLAUDE.md s2). This "
            "shipped once already: 58d980d656 (2026-08-06) added "
            "src/bipole_dispatch.cpp to the vibeqc_core source list and an "
            "#include of vibeqc/bipole_dispatch.hpp to bindings.cpp while both "
            "files stayed untracked in the authoring clone; nine commits "
            "landed on top before 8a07eda80 supplied them. Filesystem-only -- "
            "no configure, no compiler, 0.3 s"
        ),
    },
    "tests/test_gfn2_parameter_identity.py": {
        "rationale": (
            "canonical GFN2 parameter identity and immutable execution snapshots"
        ),
    },
    "tests/test_periodic_gfn2_aes.py": {
        "owner": "semiempirical",
        "rationale": (
            "periodic Gamma GFN2-xTB with the lattice-summed gamma and the "
            "Bannwarth 2019 AES: lattice-relabelling invariance and "
            "cell-face continuity (#296), gradient and all-nine stress "
            "finite differences (#338), Elstner cutoff convergence"
        ),
    },
    "tests/test_periodic_shell_gamma.py": {
        "owner": "semiempirical",
        "rationale": (
            "shared lattice-summed shell gamma kernel: Elstner 1998 closed "
            "forms and series, Klopman-Ohno remainder, 1-D/2-D/3-D Ewald "
            "Madelung anchors, record-driven assembly covariance, gradient "
            "and strain finite differences"
        ),
    },
    "tests/test_nddo_parameter_identity.py": {
        "rationale": (
            "canonical NDDO parameter identity and immutable execution snapshots"
        ),
    },
    "tests/test_semiempirical_parameter_identity.py": {
        "rationale": (
            "canonical semiempirical parameter identity and immutable execution snapshots"
        ),
    },
    "tests/test_semiempirical_kpoint_band_edges.py": {
        "rationale": (
            "issue #426 measured band-edge/gap result surface: gap convention "
            "at a fractionally occupied degenerate edge (equals pooled aufbau, "
            "not the occupancy partition -- sec8-r4 F1), is_metallic, "
            "negative-gap anti-vacuity, and edge absence as NaN"
        ),
    },
    # The next four entries preserve hand-classifications that other chats had
    # written directly into the generated JSON without a CURATED backing row
    # (recovered verbatim 2026-08-06, when a regeneration would otherwise have
    # clobbered them -- the exact failure
    # test_suite_manifest_matches_a_fresh_generator_run exists to catch).
    "tests/test_bipole_exact_zone.py": {
        "disposition": "new",
        "rationale": (
            "BIPOLE-EXACT-ZONE increment 1: restricted exact bielectronic "
            "zone (builder, resolver, SCF tail, gradient fail-closed)"
        ),
    },
    "tests/test_bipole_contractor_parity.py": {
        "rationale": "C++ vs Python far-field contractor parity tests",
    },
    "tests/test_bipole_exact_vs_far_field_energy.py": {
        "rationale": "uHa-scale energy pin for far-field validation",
    },
    "tests/test_bipole_exchange_q0_gauge.py": {
        "rationale": (
            "pins the single definition of the corrected-split q+G=0 "
            "exchange gauge constant and its disclosure in the energy "
            "report; #82's +0.64 Ha cross-code offset was 99.19 % the "
            "reference's own uncorrected SHRINK-2 finite-size error, "
            "invisible because the gauge was never reported"
        ),
    },
    "tests/test_bipole_final_density_phase.py": {
        "rationale": (
            "pins that the BIPOLE post-SCF non-incremental Fock rebuild "
            "announces itself, reports its wall time, and records the "
            "loop energy before it starts; #115 lost a converged 24 h "
            "result to >= 5.22 h of silence in that phase"
        ),
    },
    "tests/test_bipole_far_field_convergence.py": {
        "rationale": "far-field convergence tests for Saunders 1992 bipolar pipeline",
    },
    "tests/test_bipole_far_field_scf.py": {
        "rationale": "end-to-end far-field SCF accuracy pin tests",
    },
    "tests/test_bipole_parity.py": {
        "rationale": "C++ vs Python parity unit tests for BIPOLE components",
    },
    "tests/test_bipole_sph_to_cart.py": {
        "rationale": "spherical-to-Cartesian conversion tests for BIPOLE L=4 pipeline",
    },
    "tests/test_cosx_cartesian_parity.py": {
        "owner": "basis/integrals",
        "disposition": "keep",
        "rationale": (
            "pins COSX's analytic half against exact ERI for Cartesian "
            "shells, where its grid half and its kernel previously disagreed "
            "on the libint normalization convention"
        ),
    },
    "tests/test_scf_oscillation.py": {
        "maturity": "production",
        "owner": "molecular/SCF",
        "disposition": "new",
        "rationale": (
            "IID 152 pins the scale-aware molecular oscillation detector "
            "against archived contracting SCF traces, a DIIS-restart "
            "boundary, and a genuine no-aids limit cycle. The two complete "
            "archived molecular cases are slow-marked in the same file; "
            "tier remains derived T2 through molecular-scf-dft"
        ),
    },
    "tests/test_iao_ibo.py": {
        "owner": "analysis/localization",
        "rationale": (
            "IAO/IBO published-value and Jacobi-increment coverage "
            "(handovers/HANDOVER_IBO.md)"
        ),
    },
    "tests/test_periodic_fold_cutoff_adequacy.py": {
        "maturity": "production",
        "owner": "BIPOLE",
        "rationale": (
            "BUG 124 root cause: AO lattice-sum adequacy at the default cutoff"
        ),
    },
    "tests/test_periodic_gpw_rset_consistency.py": {
        "rationale": (
            "GPW-SK-HK-RSET-INCONSISTENT reproducer: uniform lattice-sum R-set "
            "for the multi-k pencil"
        ),
    },
    "tests/test_periodic_molden_gamma_export.py": {
        "maturity": "verified",
        "tier": "T1",
        "owner": "periodic-SCF",
        "disposition": "blocking T1",
        "rationale": (
            "pins the periodic Molden Gamma-block export contract: real "
            "coefficients, both spin blocks, and fail-closed refusal at k != 0"
        ),
    },
    "tests/test_periodic_density_artifact_capability.py": {
        "maturity": "production",
        "owner": "aiccm-real-gamma",
        "rationale": (
            "density-artefact capability contract (#679): a completed "
            "periodic calculation must not be discarded at finalization. "
            "Pins spin-channel presence by content, the one-entry Gamma "
            "per-k container, the absent Bloch mesh on real-space "
            "supercell-Gamma results, the up-front refusal for per-k routes "
            "at true multi-k, and that D:S certification still refuses an "
            "uncertifiable density"
        ),
    },
    "tests/test_periodic_runner_method_gate.py": {
        "maturity": "production",
        "owner": "BIPOLE",
        "rationale": (
            "run_periodic_job method-gate contract; error-text only, no physics"
        ),
    },
    # Preserve hand-classifications that the BIPOLE, agentic-loop, grid and
    # POLIPO chats wrote straight into the generated JSON without a CURATED
    # backing row, which is what made the manifest disagree with a fresh
    # generator run. Values are copied verbatim from those rows: the point is
    # to keep each owner's wording and classification, not to re-judge it.
    'tests/test_bipole_contractor_parity.py': {
        "maturity": 'verified',
        "rationale": (
            "C++ far-field contractor energy and Fock parity (14.9x "
            "speedup)"
        ),
    },
    'tests/test_bipole_cpp_wiring.py': {
        "disposition": 'new',
        "rationale": (
            "Verify C++ far-field gradient contractor is importable and "
            "auto-wired in build_bipolar_far_field_gradient_contribution"
        ),
    },
    'tests/test_bipole_far_field_convergence.py': {
        "maturity": 'verified',
        "rationale": 'far-field energy convergence with box size for H2/STO-3G',
    },
    'tests/test_bipole_far_field_scf.py': {
        "maturity": 'verified',
        "rationale": 'end-to-end far-field SCF accuracy pin tests',
    },
    'tests/test_bipole_fd_gradient.py': {
        "disposition": 'new',
        "rationale": (
            "Validate analytic far-field gradient: finite/non-zero, dM/dA "
            "contribution measurable, physically sensible signs"
        ),
    },
    'tests/test_bipole_gradient_parity.py': {
        "disposition": 'new',
        "rationale": (
            "C++ far-field gradient contractor parity vs Python reference "
            "(MgO STO-3G)"
        ),
    },
    'tests/test_bipole_pair_moments_parity.py': {
        "maturity": 'verified',
        "rationale": 'C++ pair-centre moment shift machine-precision parity',
    },
    'tests/test_bipole_production_far_field.py': {
        "disposition": 'new',
        "rationale": (
            "MgO/Diamond production SCF validation with far-field "
            "auto-enabled (Gamma + multi-k)"
        ),
    },
    'tests/test_bipole_scf_parity.py': {
        "disposition": 'new',
        "rationale": (
            "End-to-end SCF parity: far-field vs exact 4-centre ERIs at "
            "same cutoff (H2/LiH STO-3G)"
        ),
    },
    'tests/test_bipole_symmetry_reconstruction_parity.py': {
        "rationale": (
            "permutation-symmetry reconstruction vs standard kernel-apply "
            "parity"
        ),
    },
    'tests/test_bipole_unrestricted_far_field.py': {
        "maturity": 'verified',
        "rationale": 'closed-shell UHF/UKS far-field J parity with exact path',
    },
    'tests/test_dual_engine_optimize.py': {
        "owner": 'BIPOLE',
        "disposition": 'new',
        "rationale": (
            'IID 257 regression: pins that the dual-engine optimization '
            'record carries the computed BIPOLE SCF verdict '
            '(bipole_converged and bipole_n_iter) and that delta_mha is '
            'gated on convergence instead of serialized regardless. '
            'Pure Python, no SCF, no compiler; both engines are stubbed'
        ),
    },
    'tests/test_ccm_conv_tol_grad.py': {
        "disposition": 'new',
        "rationale": (
            'IID 295 regression: pins that the closed-shell CCM SCF exit '
            'predicate honors a caller-supplied conv_tol_grad and that the '
            'hard-coded 1e-6 commutator literal no longer appears in the '
            'scf.py/ri.py exit gates. Pure Python, no native integrals; '
            'the fixed-point loop test needs no compiled core'
        ),
    },
    'tests/test_smearing_frontier_resolution.py': {
        "disposition": 'new',
        "rationale": (
            'IID 543 regression: pins that the Python global-Aufbau twin '
            '(smearing/apply.py, behind the ab-initio multi-k routes) REPORTS '
            'a T=0 frontier it cannot resolve instead of silently returning a '
            'raced integer occupation. Behavioural fail-first at the exact '
            '3.2e-07 Ha splitting IID 424 measured, including the direct '
            'demonstration that flipping the sign of that splitting moves a '
            'whole electron between k-points while both fills report success. '
            'Also pins the equalized Weinert-Davenport ensemble below the '
            'roundoff tolerance is NOT reported, and mechanically pins the '
            'Python threshold EQUAL to the C++ '
            'KPointOccupationOptions.min_resolvable_frontier_gap default so '
            'the two independent fills cannot drift. Pure Python, synthetic '
            'two-k spectra, no SCF'
        ),
    },
    'tests/test_solvation_fine_cavity_gradient.py': {
        "disposition": 'new',
        "rationale": (
            'IID 729: analytic nuclear derivatives of the COSMO FINE cavity. '
            'Klamt & Diedenhofen 2018 assert differentiability but give no '
            'algebra, so every stage of the derivation is FD-verified '
            'independently -- an error in one stage yields a smooth, '
            'plausible, wrong total. Stages: PD gradients (plus the EXACT '
            'translation identity sum_A grad_RA PD = -grad_r PD, which needs '
            'no finite differences); iso-point edge Jacobians, including that '
            'the FD motion is confined to the vertex own grid edge, the '
            'rank-one property the derivation rests on; triangle and '
            'per-vertex area gradients; and segment area and position '
            'Jacobians covering BOTH projected and unprojected segments, '
            'since step 6 pushes about half the segments onto their atom '
            'sphere and a derivation omitting that term passes on half the '
            'data. Also pins that vertices must be Newton-refined onto the '
            'iso-surface first: the paper quadratic interpolation leaves '
            '|PD-1| up to 6e-2, and differentiating the constraint PD=1 at a '
            'point that does not satisfy it gives a 0.1-1 percent error. The '
            'FD harness REBUILDS the marching box at every displaced '
            'geometry, because build_fine_cavity anchors it to the molecule: '
            'freezing it would verify the derivative of a function nobody '
            'computes and would hide the box-translation term entirely. '
            'Vertices are matched by grid-edge LATTICE INDEX rather than by '
            'coordinate, since the box translates, and by edge rather than by '
            'vertex index, since welding sorts by coordinate so indices '
            'permute once an atom moves -- either mistake looks exactly like '
            'a wrong derivative. Segment assignment is held fixed, which is a '
            'real caveat the Lebedev cavity shares through its drop threshold'
        ),
    },
    'tests/test_solvation_segment_partition.py': {
        "disposition": 'new',
        "rationale": (
            'IID 757: the Becke fuzzy-cell partition that makes the CFC '
            'coarsening continuous. Klamt & Diedenhofen 2018 assign each basis '
            'point to exactly one segment by nearest centre, so a point on a '
            'boundary flipped at an infinitesimal displacement and its WHOLE '
            'area moved -- three of 1230 points reassigning shifted one '
            'segment weight by 0.333 bohr^2 and jumped the energy by 6e-7 Ha, '
            'reading a secant slope of 0.0111 where the analytic gradient and '
            'every other interval said 0.0233. The paper does not address it; '
            'its own treatment of related discrete problems is to avoid '
            'differentiating them (COC and triple-segment area gradients '
            '"neglected", symmetric coincidences broken with "a small '
            'geometrical noise function", p. 1652). So the fix is Becke 1988 '
            '(doi:10.1063/1.454033), the canonical way to make a '
            'point-to-centre assignment continuous, at Becke published k = 3 '
            'and hence with no fitted parameter -- mu is a RATIO of distances, '
            'so the smoothing width is set by the centre spacing, the only '
            'length in the problem. Pins the switch against the published '
            'recurrence and its order EQUAL to the C++ GridOptions.becke_k '
            'default so the cavity and the DFT molecular grid cannot drift '
            'apart; the partition of unity, which is exact and is what '
            'conserves the cavity total area that COSMO-RS sigma profiles '
            'read; that the hard-assignment limit is recovered EXACTLY at a '
            'centre, so the smooth version still reduces to the paper where '
            'the assignment is unambiguous; that weight transfers monotonically '
            'and passes through 1/2 at a midpoint, which is the defect '
            'actually being removed; and that there is NO nearest-neighbour '
            'truncation of the product, because s(mu)=0 needs mu>=1 so distant '
            'factors are near 1 and not equal to it (16 nearest centres moved '
            'the weights by 1.2e-3) and because the cutoff is itself '
            'discontinuous when the m-th and (m+1)-th centres swap, which '
            'would reintroduce a smaller copy of the very defect. Adjoints are '
            'FD-verified for BOTH the points and the centres, and the exact '
            'sum_S dW_S = 0 identity is checked with no finite differences at '
            'all -- it is why the total-area derivative is immune to partition '
            'errors. Also pins chunk-size independence and that coincident '
            'centres are refused. Pure geometry, no SCF'
        ),
    },
    'tests/test_solvation_charge_representation.py': {
        "disposition": 'new',
        "rationale": (
            'IID 744: the surface-charge representation. The point-charge '
            'kernel 1/r_ij is singular at contact and Lange & Herbert 2010 '
            '(doi:10.1063/1.3511297, p. 244111-8) state that point charges '
            'NECESSITATE a switching function to keep segments apart. A CFC '
            'has none and cannot have one, so the kernel made A indefinite at '
            '10 of 29 grid spacings on water, worst eigenvalue -8.62, which '
            'breaks the eq. 2.25 variational condition that makes the '
            'apparent-charge solve a MINIMUM rather than an arbitrary '
            'stationary point -- run_cpcm_scf reported convergence and a '
            'negative e_solv throughout. Fixed with the York & Karplus 1999 '
            'Gaussian kernel (doi:10.1021/jp992097l eq. 56), under which A is '
            'the Gram matrix of normalized Gaussians in the Coulomb inner '
            'product and positive definite for ANY geometry, so definiteness '
            'is structural rather than a property of the grid. Pins the whole '
            'parameter chain, because the fix must introduce no fitted number '
            'of its own: York-Karplus Table 1 transcribed verbatim; the '
            'bridge zeta = C_S pi sqrt(2) checked by feeding Lange-Herbert '
            'Lebedev C_S = 1.104 through it and recovering York-Karplus '
            'independently fitted 4.901-4.907 to four digits, two papers and '
            'two fitting procedures agreeing; the area-form exponent checked '
            'against eq. 61 plus the published radius scaling; and the '
            'identity of the two diagonals at matched width, which is why no '
            'new parameter is needed at all. Also pins that the off-diagonal '
            'reduces to 1/r to machine precision beyond ~2 bohr, so the '
            'correction is local and cannot move a well-separated energy; '
            'that it is bounded at exact coincidence where 1/r diverges; that '
            'coincident segments give a SINGULAR rather than indefinite A, '
            'which is the honest answer since two coincident patches carry '
            'one degree of freedom; that each cavity declares the '
            'representation its own construction requires, so a third '
            'construction cannot inherit a precondition it does not meet; and '
            'that the Lebedev A stays BIT-IDENTICAL, the compatibility anchor '
            'without which every shipped solvation reference would move. '
            'Records separately that vibe-qc uses the GEPOL C_S = 1.0694 on '
            'Lebedev grids where the literature Lebedev value is 1.104, a 3 '
            'percent narrow implied Gaussian width, NOT changed here because '
            'it would move every reference and is a maintainer decision. '
            'Pure linear algebra, no SCF'
        ),
    },
    'tests/test_solvation_cavity_derivative.py': {
        "disposition": 'new',
        "rationale": (
            'The cavity-derivative seam. Every geometry dependence of the '
            'solvated energy reaches the cavity through exactly two '
            'per-segment quantities, positions and areas, so a cavity term is '
            'written once against per-segment adjoints and each construction '
            'supplies its own chain rule. Pins the three properties that make '
            'the split safe. (1) cavity_A_adjoints is verified by perturbing '
            'segment positions and areas DIRECTLY, no atoms involved, on BOTH '
            'constructions -- the assertion that this half does not secretly '
            'depend on one of them. (2) The distinct left/right form u^T A v '
            'is pinned now, while the energy gradient only ever asks for '
            'q^T A q and so cannot distinguish the symmetrised u_i v_j + '
            'u_j v_i factor from a bare 2 q_i q_j; Direct COSMO-RS needs the '
            'general form, and losing that factor of two once already cost a '
            '3.5e-4 Ha/bohr residual. (3) The Lebedev contraction is compared '
            'against a VERBATIM COPY of the pre-refactor routine kept in the '
            'test file as an independent oracle, so the refactor is shown to '
            'move no numbers rather than merely to still pass. Plus the two '
            'EXACT translation invariants, sum_A dp_S/dR_A = I and '
            'sum_A dw_S/dR_A = 0, on both constructions: no finite '
            'differences, and the check that exposed the FINE cavity missing '
            'box-translation term at 0.34 against components of order 15. '
            'Dispatch refuses an unknown cavity rather than letting it '
            'inherit another construction chain rule. Pure geometry, no SCF'
        ),
    },
    'tests/test_solvation_fine_cavity.py': {
        "disposition": 'new',
        "rationale": (
            'IID 558: the COSMO FINE Cavity of Klamt & Diedenhofen 2018 '
            '(doi:10.1002/jcc.25342) -- pseudo-density iso-surface, marching '
            'tetrahedra, geodesic segment grids and the segment coarsening. '
            'Pure geometry, no SCF. No reference segment set is published, so '
            'the tests pin what CAN be checked: the isolated-atom PD=1 '
            'iso-surface is EXACTLY the vdW sphere for any a1, which is the '
            'strongest available check that eq. 8 uses radius-scaled relative '
            'distances; the geodesic grids hit the COC magic numbers 10n^2+2 '
            'with no duplicated edge or corner points; the marching-tetrahedra '
            'mesh is CLOSED at every spacing (the paper requires alternating '
            'mirror tetrahedra so neighbouring cubes share a face diagonal, '
            'and a cracked cavity leaks field rather than screening it); '
            'sphere area AND volume both converge, which is what tests the '
            'triangle-orientation pass, since area is a norm and survives '
            'mixed winding while volume cancels to zero -- the exact failure '
            'observed before that pass existed; segment coarsening conserves '
            'the triangulated area exactly (sub-threshold basis points are '
            'joined, not dropped); segments sit on or outside their atom '
            'sphere per step 6, because a screening charge inside the surface '
            'it screens is a physical error; area is stable under rigid '
            'rotation, which is not free for an axis-aligned marching grid; '
            'and the pair term demonstrably inflates the surface past the '
            'union of spheres, which is the entire purpose of the '
            'construction'
        ),
    },
    'tests/test_solvation_cosmors.py': {
        "disposition": 'new',
        "rationale": (
            'IID 558: the COSMO-RS/COSMOSPACE thermodynamic layer and the '
            'conductor-surface record it consumes. Method-independent by '
            'construction, so all but one test needs no SCF. Pins the Klamt '
            '1998 section 5.1 constants verbatim against the paper (a_eff, '
            "alpha', f_corr, r_av, c_hb, sigma_hb, lambda, radii, dispersion) "
            'plus beta and r_eff derived from them, and pins that the set '
            'RECORDS the DMol/BPW91/DNP/NSPA92 protocol it was fitted to. '
            'Physics rather than regurgitation: the misfit energy vanishes '
            'exactly for the ideally paired contact and is symmetric and '
            'quadratic; the H-bond term is zero for nonpolar and for '
            'same-sign pairs, stabilising for polar ones, and lands at a few '
            'kcal/mol per contact which is the units check; the sigma '
            'potential is verified by RE-SUBSTITUTING it into eq. 18 rather '
            'than by trusting the loop stopped, is symmetric for a symmetric '
            'ensemble, and the segment form (eq. 23-25) agrees with the '
            'profile form (eq. 18) when f_corr is switched off so the only '
            'difference is that term; the activity coefficient of a solute '
            'in itself is exactly 1. Also pins deterministic surface '
            'round-trip, schema-version refusal, finite sigma on zero-area '
            'segments (a switched cavity really makes them), None rather '
            'than zero for a missing ideal screening energy, and refusal to '
            'substitute a zero dispersion constant for an element the 1998 '
            'fit never covered'
        ),
    },
    'tests/test_solvation_engine.py': {
        "disposition": 'new',
        "rationale": (
            'IID 554 regression: pins the ONE generic reaction-field step '
            'that the Gaussian macro-iteration and the MSINDO fock_extra '
            'hook now share, against a stub provider so it needs no SCF. '
            'Pins q = -f A^-1 (V_elec + V_core) and e_pol = 1/2 q.V_total '
            'against the closed form they replaced, that the core and '
            'electronic energy shares sum exactly to e_pol (MSINDO adds only '
            'the core share, so a broken split would silently half- or '
            'double-count), that q scales linearly in f across variants, and '
            'that with_charges reuses the potentials WITHOUT a second ESP '
            'pass -- the accelerator seam has to be both shared and cheap or '
            'q-DIIS costs more than it saves. Also pins loud failure on a '
            'singular cavity rather than NaN charges reaching the Fock '
            'matrix, and asserts both shipped routes still import the shared '
            'step so a third copy cannot quietly reappear'
        ),
    },
    'tests/test_solvation_screening.py': {
        "disposition": 'new',
        "rationale": (
            'IID 548/549 regression: pins that the dielectric screening '
            'factor has ONE derivation per language and that the two are '
            'bitwise equal across both variants on a 9-point dielectric '
            'grid, so neither can be changed alone -- the same mechanical '
            'pin pattern as the smearing frontier threshold. Before this, '
            'f(e) was written out independently in four places and nothing '
            'tied any two together; #546 is what that costs. Also pins the '
            'Klamt 1993 one-parameter family (CPCM x=0, COSMO x=1/2) so a '
            'new variant is a table entry not a branch, the exact '
            'epsilon=inf conductor limit f=1 where the closed form is NaN, '
            'that SolventResult carries the ScreeningModel that actually '
            'built q (verified by rescaling q back to the unscaled '
            'conductor solution), that gas phase is ABSENT screening rather '
            'than an out-of-range epsilon, that the gas-phase gradient '
            'equals the plain gas-phase gradient to machine precision '
            '(#549; ~1 ulp apart, thread-count dependent), and that '
            'the C++ MSINDO reaction field reads a variant argument instead '
            'of baking the COSMO form in'
        ),
    },
    'tests/test_solvation_variant_gradient.py': {
        "disposition": 'new',
        "rationale": (
            'IID 546 regression: pins that the solvation cavity-gradient '
            'term screens with the factor that actually built q '
            '(SolventResult.solvent_variant) instead of a hard-coded '
            '"cpcm" literal, so a variant="cosmo" run stops '
            'differentiating a different energy than it reports. '
            'Behavioural fail-first at eps=2.27: the shipped '
            'analytic-vs-central-FD residual PLATEAUS at 4.99e-05 Ha/bohr '
            'across h = 4e-3/2e-3/1e-3 -- the signature of a systematic '
            'error, not FD truncation -- while the corrected gradient '
            'converges O(h^2) to 4.70e-07 like the unaffected CPCM '
            'control, whose correction is exactly zero. Isolates the '
            'cavity term at 18.1 percent wrong at benzene and 0.6 percent '
            'at water. Validates cavity segment counts are identical at '
            'every displaced geometry in every stencil, so analytic and '
            'FD sample one energy surface. Carries a pure-Python unit pin '
            'of the factor selection that needs no SCF, plus a '
            'CPCM negative control'
        ),
    },
    'tests/test_semiempirical_unresolvable_frontier_guard.py': {
        "disposition": 'new',
        "rationale": (
            'IID 434 regression: pins that a T=0 k-point Aufbau fill REFUSES '
            'a frontier it cannot resolve instead of cutting it and reporting '
            'success. Behavioural fail-first -- pre-fix run_dftb0_kpoints '
            'returns a raced result on the constructed Z31 zigzag chain at '
            'a=16 bohr (frontier gap 3.99e-07 Ha); post-fix it raises. Also '
            'pins what must NOT change: the smallest cut gap any natural '
            'chain in the scanned family reaches (1.44e-05 Ha at a=9) still '
            'runs, an exactly degenerate frontier still takes the equalized '
            'T->0 ensemble branch #424 depends on, finite-T is the working '
            'remedy, and the 1e-6 boundary is straddled from both sides '
            '(1.18e-06 admitted, 6.91e-07 refused). Pure Python over the '
            'pybind11 DFTB0 k-point driver, no SCF, seconds'
        ),
    },
    'tests/test_ccm_scf_trace.py': {
        "disposition": 'new',
        "rationale": (
            'IID 493 diagnosis instrumentation: pins the per-cycle CCM SCF '
            'trace (CCMSCFIteration on CCMSCFResult.scf_trace) to the loop it '
            'describes -- one row per cycle, last row energy bitwise equal to '
            'the returned energy, delta_e equal to the row-to-row difference, '
            'and the final grad_max below the same commutator gate the exit '
            'predicate tests, so a stalled run can be read against its own '
            'convergence criterion. Also pins that recording the trace cannot '
            'perturb the energy. Pure Python on LiH/sto-3g (6 cycles, ~1 s)'
        ),
    },
    'tests/test_ccm_lindep_screening.py': {
        "disposition": 'new',
        "rationale": (
            'IIDs 289/297 regression: pins that CCM lindep_tol screens (canonical '
            'orthogonalisation projects below-threshold overlap directions) '
            'instead of refusing, that the shared Python SCF loops fail closed '
            'below the occupation count, and that the C++-driver DFT/scalable '
            'routes forward the threshold into linear_dep_threshold; also pins '
            'the exact indefinite LiH overlap, diagnostic classification, '
            'non-finite refusal, occupied-rank-safe native-host screening, and '
            'well-conditioned bitwise noninterference'
        ),
    },
    'tests/test_semiempirical_kpoint_scc_branch_determinism.py': {
        "disposition": 'new',
        "rationale": (
            'IID 424 regression: pins the symmetric SCC fixed point with '
            'margin (conv_tol_charge 1e-12, n_iter 1, |dq| < 1e-12) on the '
            'six homonuclear B/Ga chain inputs whose converged branch '
            'flipped across fleet hosts at build 127682e39, plus '
            'fresh-process byte-identity on the flagship case. Fails 5/6 '
            'on the pre-#316 image selection; tier stays derived T2 '
            'through the semiempirical lane'
        ),
    },
    'tests/test_grid_level.py': {
        "maturity": 'verified',
        "owner": 'basis/integrals',
        "rationale": 'BUG 80 regression: DFT grid-level presets and default switch',
    },
    'tests/test_polipo_parity.py': {
        "maturity": 'experimental',
        "owner": 'BIPOLE',
        "rationale": (
            "POLIPO vs libint Cartesian moment parity + spherical "
            "conversion validation"
        ),
        "tier": 'T3',
    },
    # Same preservation, for rows the semiempirical and molecular/SCF chats
    # added by hand after the block above was written.
    'tests/test_pm7.py': {
        "maturity": 'under-review',
        "owner": 'semiempirical',
        "rationale": 'under-review inventory or parity coverage retained post-cut',
    },
    'tests/test_semiempirical_pair_image_selection.py': {
        # Copied verbatim from the committed manifest row (issue #316 chat's
        # hand-classification, registered here so regeneration keeps it).
        "rationale": (
            'issue #316 pair-distance image-selection regressions: '
            'band-folding identity, wide-supercell recovery, translation '
            'invariance, molecular-limit guard'
        ),
    },
    'tests/test_incremental_fock_disengage.py': {
        "disposition": 'keep T2',
        "maturity": 'production',
        "owner": 'molecular/SCF',
        "rationale": (
            "IIDs 129/418 regression coverage for direct-SCF Fock-map "
            "transitions: high-gradient drift-stall disengagement, "
            "configured tight-screen enforcement, and gradient-eligible "
            "energy certification on two consecutive same-map, "
            "nonincremental full-density rows, including Liakos and S22-19"
        ),
    },
    'tests/test_incremental_fock_open_shell.py': {
        "maturity": 'production',
        "owner": 'molecular/SCF',
        "rationale": (
            "IID 468 fused-G scale-homogeneous screening plus direct-SCF "
            "incremental J/K slot parity for RKS, UHF, and UKS"
        ),
    },
    'tests/test_scf_restart.py': {
        "disposition": 'keep T2',
        "owner": 'molecular/SCF',
        "rationale": (
            "BUG 64 regression coverage for deterministic SCF restart and "
            "transition-metal convergence"
        ),
    },
    'tests/test_state_fingerprint.py': {
        "disposition": 'keep T2',
        "maturity": 'production',
        "owner": 'molecular/SCF',
        "rationale": (
            "BUG 88 state fingerprint and multi-guess convergence "
            "regression coverage"
        ),
    },
    "tests/test_cwd_artifact_guard.py": {
        # Rationale and owner only; maturity/tier stay derived from the lane.
        "owner": "release/test-health",
        "rationale": (
            "self-check for the GitLab #508 working-directory guard in "
            "tests/conftest.py, which fails a session that leaves job "
            "artifacts in the directory pytest was invoked from. A test "
            "passing a bare relative stem to output= writes six files into "
            "the checkout under test; #508 surfaced only because the next "
            "fleet roll refused to deploy over the dirty managed tree, days "
            "later and from an unrelated guard. This guard is the mechanical "
            "pin that issue asks for, so it needs a pin of its own: a guard "
            "that silently stops guarding reproduces the exact failure mode "
            "it exists to catch. Pure filesystem and pure functions, no SCF; "
            "milliseconds."
        ),
    },
    "tests/test_kmesh_address.py": {
        # Maturity and rationale only; tier/owner/disposition stay derived
        # from the pbc-core lane this file was added to.
        "maturity": "verified",
        "rationale": (
            "exact-integer contract for cpp/src/kmesh_address.cpp, the "
            "addressing layer the periodic correlated methods do their "
            "crystal-momentum bookkeeping on. Two things here cannot be "
            "recovered by a later chat if they drift. First, byte-neutrality: "
            "bloch.cpp now builds its unreduced Monkhorst-Pack grid through "
            "fractional_at, and the test pins (2m+s)/(2N) as the SAME double "
            "as the historical (m+s/2)/N -- not merely close -- because every "
            "periodic reference value in the tree was measured against the "
            "old expression. Second, the Monkhorst-Pack even/odd convention: "
            "the half-step shift reproduces the classical 1976 Eq. (3) set on "
            "an EVEN mesh, and the comments in bloch.hpp and crystal.hpp said "
            "odd until 2026-09-02. Carried as verified rather than the lane's "
            "under-review default because nothing here is a scientific "
            "method under review: every assertion is an identity between "
            "integers or a bitwise float comparison, with no SCF, no "
            "tolerance and no reference energy. Pure integer arithmetic and "
            "one 6-bohr cubic cell; well under a second."
        ),
    },
    "tests/test_periodic_correlation_admission.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "pins the in-process C++ admission boundary between a certified "
            "full-k RHF state and future periodic local-correlation work. It "
            "checks exact mesh/shift, orbital and frozen-core dimensions, "
            "immutable shared ownership, state-byte accounting before the "
            "resource gate, and fail-closed memory and symmetry behavior. "
            "All fixtures are tiny constructed matrices; no SCF, target-sized "
            "allocation, or production correlation path runs. T2 while the "
            "periodic DLPNO executor remains under review."
        ),
    },
    "tests/test_periodic_correlation_factor_stream.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "pins the native q-major factor-stream transport contract after "
            "periodic-correlation admission. Independent integer and digest "
            "oracles check exact momentum addressing, ragged tile coverage, "
            "AO/payload layout, source-bound provenance, fixed digest known "
            "answers, transaction failures, SHA limits, bounded NumPy "
            "transport, explicit tile/workspace/count/byte caps, and "
            "allocation-free 8x8x8/6x6x6 shape counts. "
            "The only payload is a named deterministic nonphysical pattern; "
            "no integral, factor store, SCF, energy, or target chemistry runs. "
            "T2 while the numerical periodic DLPNO executor remains under "
            "review."
        ),
    },
    "tests/test_periodic_correlation_factor_build_admission.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "pins the native count-only admission contract for a future "
            "numerical periodic factor builder. Independent scalar oracles "
            "check centred transfer representatives, full-auxiliary "
            "whitening panels, backend workspaces, random-access publication, "
            "atomic scratch generations, exact caps, and allocation-free "
            "8x8x8/6x6x6 target-shaped counts. It constructs no target state, "
            "integral, factor tensor, or energy. T2 while the numerical "
            "periodic DLPNO executor remains under review."
        ),
    },
    "tests/test_periodic_correlation_reciprocal_metric.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "pins the constant-space native one-q reciprocal-source manifest "
            "and the bounded raw Hermitian metric assembled from it. "
            "Independent fixed-FMA source/payload-wire and analytic two-s "
            "Gaussian oracles check Gamma zero omission, non-Gamma q/-q "
            "covariance, exact-real negative-Nyquist canonicalisation, panel-"
            "block bit invariance, SHA limits, source/census/plan provenance, "
            "admission-before-Fourier ordering, immutable matrix copies, hard "
            "diagnostic caps, and two historical binary64 bound "
            "counterexamples without exposing or retaining a G list. Every "
            "successful numerical metric is at most 4x4 with at most three k "
            "points and six accepted reciprocal vectors; no three-center "
            "integral, factor tensor, SCF, energy, target mesh, or target "
            "chemistry runs. T2 while the numerical periodic DLPNO executor "
            "remains under review."
        ),
    },
    "tests/test_periodic_auxiliary_fourier.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "pins the native bounded one-center auxiliary-Gaussian Fourier "
            "panel and its versioned basis-content identity. The existing "
            "Python RSGDF formula and an independent big-endian digest oracle "
            "cover pure spherical L=0..6, contractions, translated centers, "
            "Gamma, zero-copy SoA input, exact output caps, unsupported shell "
            "conventions, signed-zero/name invariance, and a fixed SHA-256 "
            "known answer. All fixtures are tiny custom bases and reciprocal "
            "panels; no SCF, factor tensor, energy, or target chemistry runs. "
            "T2 while the numerical periodic DLPNO executor remains under "
            "review."
        ),
    },
    "tests/test_periodic_correlation_pair_topology.py": {
        "maturity": "under-review",
        "tier": "T2",
        "owner": "periodic-SCF",
        "disposition": "demote T2",
        "rationale": (
            "pins the native translation-unique occupied-pair topology that "
            "follows static periodic-correlation admission. An independent "
            "placed-pair orbit oracle exhausts small meshes and verifies "
            "ordering, multiplicities, identities, ownership, and fail-closed "
            "shift behavior; larger project shapes exercise allocation-free "
            "counts only. No SCF, domain construction, amplitudes, or target "
            "chemistry runs. T2 while the periodic DLPNO executor remains "
            "under review."
        ),
    },
    "tests/test_spin_channel_detection.py": {
        "disposition": "keep",
        "rationale": (
            "regression guard for value-based open-shell detection "
            "(None-valued spin densities)"
        ),
    },
}

CURATED_FIELDS = frozenset({"maturity", "tier", "owner", "disposition", "rationale"})

DEFAULT_RATIONALE = {
    "T0": "cheap build/API contract required on every push",
    "T1": "owned stable sentinel on the pre-cut blocking path",
    "T2": "{maturity} inventory or parity coverage retained post-cut",
    "T3": "experimental AICCM/CCM/SECCM research coverage; never release blocking",
}

# Ordered strength of the two classification fields a regeneration can weaken.
# ``blocking_tiers`` in the emitted policy block is ``["T0", "T1"]``, so any
# move out of that pair narrows the release gate; T2 -> T3 additionally moves a
# file into the tier the gate never runs.
TIER_RANK = {"T0": 3, "T1": 2, "T2": 1, "T3": 0}
MATURITY_RANK = {"production": 3, "verified": 2, "under-review": 1, "experimental": 0}

# Every string ``classify`` can produce without a CURATED rationale.  A prior
# rationale outside this set was written by hand, so replacing it with a member
# of the set is a clobber, not a refresh.
BOILERPLATE_RATIONALES = frozenset(
    template.format(maturity=maturity)
    for template in DEFAULT_RATIONALE.values()
    for maturity in MATURITY_RANK
)

EXPERIMENTAL_PATTERNS = (
    "tests/test_aiccm*.py",
    "tests/test_*seccm*.py",
    "tests/test_ccm*.py",
    "tests/test_lowd_greens*.py",
    "tests/test_msindo_ccm*.py",
    "tests/test_periodic_aiccm*.py",
    # test_periodic_ccm_real_gamma_adapter.py (the real-Γ runner adapter)
    # matched none of the patterns above and was silently classified as
    # T2/under-review shipped-feature coverage owned by another lane,
    # landing it in full-fast and slow-nightly -- lanes whose own
    # descriptions exclude experimental AICCM. Added 2026-08-23.
    "tests/test_periodic_ccm*.py",
    "tests/test_periodic_megacell*.py",
    "tests/test_periodic_toroidal_mp2.py",
    "tests/test_seccm*.py",
)


def matches(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def selected(path: str, lane: dict) -> bool:
    return matches(path, lane.get("include") or ()) and not matches(
        path, lane.get("exclude") or ()
    )


def derive(
    path: str, lanes: dict, blocking_files: set[str]
) -> tuple[str, str, str, str]:
    if matches(path, EXPERIMENTAL_PATTERNS):
        return "experimental", "T3", "implementing chat", "keep"
    if path in T0_FILES:
        return "verified", "T0", "release/test-health", "keep"
    if path in blocking_files:
        maturity = (
            "production"
            if path in PRODUCTION_T1_FILES or selected(path, lanes["molecular-scf-dft"])
            else "verified"
        )
        return maturity, "T1", "release/test-health", "keep"

    precedence = (
        ("molecular-scf-dft", "production", "molecular/DFT"),
        ("molecular-correlation", "under-review", "correlation/multireference"),
        ("semiempirical", "under-review", "semiempirical"),
        ("pbc-gdf", "under-review", "GDF"),
        ("pbc-bipole", "under-review", "BIPOLE"),
        ("pbc-gapw-gpw", "under-review", "GAPW/GPW"),
        ("pbc-core", "under-review", "periodic-SCF"),
        ("embedding-surfaces-neb", "under-review", "surface/embedding"),
        ("basis-ecp-integrals", "verified", "basis/integrals"),
        ("external-reference", "verified", "regression-suite/cross-code"),
        ("output-docs", "verified", "output/docs"),
    )
    for lane_name, maturity, owner in precedence:
        if selected(path, lanes[lane_name]):
            return maturity, "T2", owner, "demote T2"
    owner = FILE_OWNERS.get(path, "release/test-health")
    return "verified", "T2", owner, "demote T2"


def classify(
    path: str, lanes: dict, blocking_files: set[str]
) -> tuple[str, str, str, str, str]:
    """Return ``(maturity, tier, owner, disposition, rationale)`` for one file.

    The lane-derived classification is computed first, then any ``CURATED``
    fields for the path are laid over it.
    """
    maturity, tier, owner, disposition = derive(path, lanes, blocking_files)
    rationale = DEFAULT_RATIONALE[tier].format(maturity=maturity)

    curated = CURATED.get(path)
    if curated is not None:
        maturity = curated.get("maturity", maturity)
        tier = curated.get("tier", tier)
        owner = curated.get("owner", owner)
        disposition = curated.get("disposition", disposition)
        # A curated tier without a curated rationale would otherwise be
        # described by the *derived* tier's boilerplate.
        rationale = curated.get(
            "rationale", DEFAULT_RATIONALE[tier].format(maturity=maturity)
        )
    return maturity, tier, owner, disposition, rationale


def validate(paths: list[str], lanes: dict, blocking_files: set[str]) -> None:
    """Fail loudly on a stale ``CURATED`` entry or a tier-policy violation.

    ``run_full_suite.load_suite_manifest`` enforces the same tier invariants when
    the gate reads the manifest; checking here reports them against the table
    that caused them rather than against a generated file.
    """
    known = set(paths)
    for path, fields in sorted(CURATED.items()):
        if path not in known:
            raise SystemExit(
                f"CURATED names {path}, which is not a discovered test file; "
                "drop or repoint the entry"
            )
        if not fields:
            raise SystemExit(f"CURATED[{path}] overrides nothing; drop the entry")
        unknown = sorted(set(fields) - CURATED_FIELDS)
        if unknown:
            raise SystemExit(
                f"CURATED[{path}] has unsupported field(s): {', '.join(unknown)}"
            )
    for path in paths:
        maturity, tier, owner, _, _ = classify(path, lanes, blocking_files)
        if maturity == "experimental" and tier != "T3":
            raise SystemExit(f"{path}: experimental file must be T3, not {tier}")
        if tier in {"T0", "T1"} and not owner:
            raise SystemExit(f"{path}: blocking file has no owner")


def detect_demotions(
    rows: list[dict],
    prior: dict[str, dict],
    allowed: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Return one message per row this regeneration would silently weaken.

    ``CURATED`` keeps a registered row's classification across regeneration, but
    it only protects rows somebody remembered to register.  An *un*registered
    hand-classification is still rebuilt from the lanes, and the guard test
    ``test_suite_manifest_matches_a_fresh_generator_run`` cannot tell the two
    resolutions apart: the committed manifest and a fresh run disagree either
    way, and rerunning the generator makes the test green *by discarding the
    hand-classification*.  That is the destructive resolution the test's own
    message inadvertently recommends, and the failure mode that lost
    ``tests/test_periodic_molden_gamma_export.py`` its blocking-T1 row.

    Comparing against the committed manifest closes that hole: a T0/T1 -> T2/T3
    move, a maturity downgrade, or a hand-written rationale reverting to tier
    boilerplate is refused at the point of regeneration, naming the rows to
    register.  Deliberate demotions pass through ``--allow-demotion``.
    """
    findings: list[str] = []
    for row in rows:
        path = row["file"]
        if path in allowed:
            continue
        old = prior.get(path)
        if not old:
            continue

        reasons: list[str] = []
        old_tier, new_tier = old.get("tier"), row["tier"]
        if TIER_RANK.get(new_tier, -1) < TIER_RANK.get(old_tier, -1):
            gate = (
                " (drops out of the blocking set)"
                if old_tier in {"T0", "T1"} and new_tier not in {"T0", "T1"}
                else ""
            )
            reasons.append(f"tier {old_tier} -> {new_tier}{gate}")

        old_maturity, new_maturity = old.get("maturity"), row["maturity"]
        if MATURITY_RANK.get(new_maturity, -1) < MATURITY_RANK.get(old_maturity, -1):
            reasons.append(f"maturity {old_maturity} -> {new_maturity}")

        old_rationale, new_rationale = old.get("rationale"), row["rationale"]
        if (
            old_rationale
            and old_rationale != new_rationale
            and old_rationale not in BOILERPLATE_RATIONALES
            and new_rationale in BOILERPLATE_RATIONALES
        ):
            reasons.append(
                "hand-written rationale replaced by tier boilerplate\n"
                f"        was: {old_rationale!r}\n"
                f"        now: {new_rationale!r}"
            )

        if reasons:
            findings.append(f"  {path}: " + "\n  ".join(reasons))
    return findings


def memory_tier(path: str, peak_mb: float | None) -> str:
    if path in {"tests/test_direct_scf_smoke.py", "tests/test_pw1pw.py"}:
        return "xl"
    if peak_mb is None:
        return "unmeasured"
    if peak_mb > 32000:
        return "xl"
    if peak_mb > 6000:
        return "heavy"
    if peak_mb >= 1000:
        return "medium"
    return "light"


def discover_paths(root: Path = ROOT) -> list[str]:
    """Every classified test file, in the manifest's row order."""
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "tests").rglob("test_*.py")
    )


def load_lanes() -> tuple[dict, set[str]]:
    """Return the lane table and the files the blocking lane selects."""
    lanes = json.loads(LANES.read_text(encoding="utf-8"))["lanes"]
    blocking_files = set(lanes["release-core-cheap"]["include"])
    blocking_files |= {
        str(node).split("::", 1)[0]
        for node in lanes["release-core-cheap"].get("nodes") or []
    }
    return lanes, blocking_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", action="append", default=[])
    parser.add_argument(
        "--verified-sha",
        help="commit tested by --measurements (defaults to the current HEAD)",
    )
    parser.add_argument(
        "--allow-demotion",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "permit this file's classification to be weakened by the "
            "regeneration (repeatable). Only the file's owning chat should "
            "pass it, and the reason belongs in the commit message."
        ),
    )
    args = parser.parse_args()

    lanes, blocking_files = load_lanes()

    prior = {}
    if OUTPUT.is_file():
        prior = {
            row["file"]: row
            for row in json.loads(OUTPUT.read_text(encoding="utf-8")).get("tests", [])
        }
    measured = {}
    for source in args.measurements:
        for line in Path(source).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            measured[record["file"].split("::", 1)[0]] = record

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    verified_sha = args.verified_sha or sha
    paths = discover_paths()
    validate(paths, lanes, blocking_files)
    rows = []
    for path in paths:
        maturity, tier, owner, disposition, rationale = classify(
            path, lanes, blocking_files
        )
        measurement = measured.get(path)
        old = prior.get(path, {})
        runtime = (
            measurement.get("elapsed_s")
            if measurement is not None
            else old.get("measured_runtime_s")
        )
        peak = (
            measurement.get("peak_rss_mb")
            if measurement is not None
            else old.get("measured_peak_rss_mb")
        )
        rows.append(
            {
                "file": path,
                "maturity": maturity,
                "tier": tier,
                "owner": owner,
                "memory_tier": memory_tier(path, peak),
                "measured_runtime_s": runtime,
                "measured_peak_rss_mb": peak,
                "disposition": disposition,
                "rationale": rationale,
                "last_verified_sha": (
                    verified_sha
                    if measurement and measurement.get("status") in {"PASS", "NOTESTS"}
                    else old.get("last_verified_sha")
                ),
            }
        )
    findings = detect_demotions(rows, prior, allowed=set(args.allow_demotion))
    if findings:
        raise SystemExit(
            "refusing to write scripts/test_gate/suite_manifest.json: this "
            f"regeneration would weaken {len(findings)} committed row(s).\n"
            + "\n".join(findings)
            + "\n\nEach of these is a hand-classification the lanes cannot "
            "reconstruct. Record it in this script's CURATED table -- as the "
            "file's owning chat, copying the committed values verbatim -- so "
            "regeneration preserves it. Do not resolve this by committing the "
            "regenerated manifest; that discards the classification. If a "
            "demotion is deliberate, rerun with --allow-demotion <path> for "
            "each file and say why in the commit message."
        )

    doc = {
        "schema_version": 1,
        "generated_at_sha": sha,
        "test_file_count": len(rows),
        "policy": {
            "blocking_tiers": ["T0", "T1"],
            "advisory_tiers": ["T2", "T3"],
            "experimental_tier": "T3",
            "scientific_acceptance": False,
        },
        "tests": rows,
    }
    OUTPUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
