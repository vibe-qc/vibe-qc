"""End-to-end recipe drivers for vibe-basis.

Implemented modules:
* ``pob_parity`` — Goal 8 Stage 0 driver.  Reproduces the
  PT2013 pob-TZVP HF parity pipeline (structures → emit .d12 →
  submit → parse → compare against PT2013 SI Table 2).
* ``remote`` — Submit an entire recipe as a single vq job.
  Package a recipe (e.g. ``pob_parity``) as a self-contained
  Python script, submit it to vq, poll, fetch results.

Planned modules:
* ``mpei_recipe`` — the Hartree-Fock-optimized mpei-TZVP
  sibling (Goal 8 Stages 1-4).
* ``optimize`` — NLopt BOBYQA / iminuit MIGRAD + HESSE /
  scipy L-BFGS-B wrappers with common OptResult.

Each recipe orchestrates a parametrization, a backend choice, a
transport, an optimizer, and acceptance gates.  The recipes are
also where the per-paper test-set definitions live (which 13 /
31 / 75 compounds at which lattice constants, weights, etc.).
"""
