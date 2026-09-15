"""ORCA reference cells + ORCA-vs-vibe-qc speed benchmark.

The ORCA axis of the HF/DFT cross-code parity matrix
(``tests/test_parity_hf_dft.py`` already certifies vibe-qc vs PySCF).
ORCA runs out-of-process on compute-reference through the ``vq`` queue — vibe-qc
never imports ORCA (CLAUDE.md § 10). This package holds:

* :mod:`parse_orca` — ORCA ``.out`` -> parity-decomposition dict, with
  a mandatory self-check.
* :mod:`orca_input` — (system, basis, method) -> ORCA input deck.
* :mod:`orca_vq` — submit / poll / fetch an ORCA job via the ``vq`` CLI.
* :mod:`cases` — the cell definitions, shared by the parity matrix and
  the speed benchmark.
* :mod:`run_parity` — on-demand: per cell, run-or-cache ORCA, compare
  against a live vibe-qc decomposition, emit a markdown report.
* :mod:`run_speed_benchmark` — on-demand: method-matched and
  default-matched ORCA-vs-vibe-qc wall-clock timing on compute-reference.

``cache/`` holds the committed ORCA decomposition JSONs so
``tests/test_parity_vs_orca.py`` runs fast in CI without ORCA / vq.
"""
