"""Optimizer benchmark suite — systematic comparison of gradient-based
geometry-optimization methods for quantum chemistry.

Compares ASE optimizers (BFGS, LBFGS, FIRE, …) and vibe-qc's native
scipy L-BFGS-B across a curated set of molecular and periodic test
systems. Measures steps to convergence, SCF evaluations, wall-clock
time, and final energy accuracy.

Usage::

    python examples/regression/optimizer_benchmark/run_benchmark.py \\
        --systems h2o,ch4,c2h4,benzene,h2o_dimer \\
        --optimizers BFGSLineSearch,LBFGS,FIRE,MDMin,native \\
        --method rks --functional PBE --basis def2-svp \\
        --fmax 0.01 --max-steps 200 \\
        --output results/optimizer_benchmark.json

Design: follows vibe-qc's §10 rule (no imports from other QC programs).
All optimizers drive vibe-qc's own SCF + gradient through the ASE
Calculator bridge (ase.py) or the native optimize_molecule() path.
"""
