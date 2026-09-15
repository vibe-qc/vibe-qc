"""CRYSTAL eval-version demo systems — geometries + parity runner.

15 reference inputs from https://www.crystalsolutions.eu/try-it.html
mirrored as ``.d12`` files in this directory. ``builders.py`` exposes
``build_<system>()`` for each; ``runner.py`` exposes
:func:`run_demo_parity` which wires a builder + a CRYSTAL14 reference
energy + ``run_pbc_bipole_rhf`` into a pass/fail check.

See ``README.md`` for the system list, basis tier, and recommended
order. See the per-demo ``parity_<system>.py`` scripts in
``examples/regression/crystal_parity/`` for the runnable entry points.
"""
from . import builders, runner

__all__ = ["builders", "runner"]
