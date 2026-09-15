"""χ-CCM, the finite-translation-group character line: ``variant="chi"`` of
``run_periodic_job(method="aiccm", ...)`` (dev-era selector ``aiccm2026dev-b``,
still accepted with a ``DeprecationWarning``).

The independent finite-torus comparator to the Γ-CCM line in
:mod:`vibeqc.periodic.ccm`. The two constructions are implemented separately
on purpose: this package imports **nothing** from :mod:`vibeqc.periodic.ccm`
(D56 and D74 in ``docs/aiccm2026dev_b_decisions.md`` record why sharing
construction code would make the paper comparison circular), and the two
packages are siblings under :mod:`vibeqc.periodic` so that the rule stays
structurally obvious. The rule is pinned by
``tests/test_test_gate_lanes.py::test_chi_and_ccm_lines_import_nothing_from_each_other``
(an ``ast`` walk in both directions); the front door that dispatches to both
lines lives in ``vibeqc.periodic_runner`` / ``vibeqc.periodic_jk_method``,
never here. Keep it that way.

Modules, with the package-root file names they carried before the 2026-09-02
layout move (``handovers/HANDOVER_AICCM_LAYOUT_MERGE.md``), for grep
continuity with the handovers and the CHANGELOG:

* ``scf``           was ``periodic_aiccm2026dev_b.py``: the RHF/RKS/UHF/UKS
  drivers, the finite-torus convention and exchange-assembly records, the
  cyclic mesh and Wigner-Seitz representative helpers, diagnostics.
* ``gradient``      was ``periodic_aiccm2026dev_b_gradient.py``.
* ``localization``  was ``periodic_aiccm2026dev_b_localization.py``.
* ``pno``           was ``periodic_aiccm2026dev_b_pno.py``.
* ``posthf``        was ``periodic_aiccm2026dev_b_posthf.py``.
* ``properties``    was ``periodic_aiccm2026dev_b_properties.py``.
* ``symmetry``      was ``periodic_aiccm2026dev_b_symmetry.py``.

The public entry points stay re-exported from the package root
(``vibeqc.run_aiccm2026dev_b_rhf`` and friends, see ``vibeqc/__init__.py``);
nothing is re-exported here, so import a name from the module that defines it.
"""
