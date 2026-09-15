"""Composed optimisation pipelines.

Each module here ties parametrise + objective + driver into a
specific protocol you can run end-to-end. Stage 1
(``single_atom``) is the smoke test; stages 2-4 (one-compound
solid, multi-compound, LD-aware multi-compound) are added as the
periodic-feature dependencies (R1-R5 in REQUIREMENTS-PERIODIC.md)
land.
"""
