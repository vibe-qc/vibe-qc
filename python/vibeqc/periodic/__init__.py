"""Periodic / solid-state model subpackage.

Hosts the two Ab Initio Cyclic Cluster Model lines as sibling subpackages:
:mod:`vibeqc.periodic.ccm`, the Γ-CCM ``aiccm2026dev-a`` construction and the
neutral fitted-torus producers, plus the runner adapter for
``run_periodic_job(method="aiccm", variant="real-gamma")`` in
:mod:`vibeqc.periodic.ccm.real_gamma_runner`, and :mod:`vibeqc.periodic.chi`,
the χ-CCM construction (``variant="chi"``; dev-era selector
``aiccm2026dev-b``). Neither package imports the other (D56/D74; pinned by an
``ast`` guard in ``tests/test_test_gate_lanes.py``); the front door that
selects a variant is :mod:`vibeqc.periodic_runner` with
:mod:`vibeqc.periodic_jk_method`, not this package. The bulk of vibe-qc's
periodic machinery still lives in the flat
``vibeqc.periodic_*`` / ``vibeqc.pbc_*`` modules; new structured
solid-state models are collected here.
"""
