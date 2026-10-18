"""Composed optimisation pipelines.

Each module here ties parametrise + objective + driver into a
specific protocol you can run end-to-end. Stage 1
(``single_atom``) is the smoke test; Stage 2 (``one_compound``) is
the in-process periodic refit on LiH; stages 3-4 (multi-compound,
LD-aware multi-compound) are added as the periodic-feature
dependencies (R1-R5 in REQUIREMENTS-PERIODIC.md) land. ``global_then_polish`` is driver composition rather than a
stage: a CMA-ES pre-pass handed to BDIIS for the polish, for any
objective whose starting point may sit in the wrong basin.
"""
