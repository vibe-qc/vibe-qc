"""Method catalog — one entry per (scf, xc, post) recipe.

Wave 1 only enables a small subset; wave 2 expands to the full matrix.
"""
from __future__ import annotations

from typing import Dict

from ..core.spec import MethodSpec

METHODS: Dict[str, MethodSpec] = {
    "rks-lda": MethodSpec(
        id="rks-lda", scf="rks", xc="lda",
        spin="closed", periodic=True, molecular=True,
    ),
    "rks-pbe": MethodSpec(
        id="rks-pbe", scf="rks", xc="pbe",
        spin="closed", periodic=True, molecular=True,
    ),
    "rhf": MethodSpec(
        id="rhf", scf="rhf",
        spin="closed", periodic=True, molecular=True,
    ),
    "uks-lda": MethodSpec(
        id="uks-lda", scf="uks", xc="lda",
        spin="open", periodic=True, molecular=True,
    ),
    "uks-pbe": MethodSpec(
        id="uks-pbe", scf="uks", xc="pbe",
        spin="open", periodic=True, molecular=True,
    ),
    "uhf": MethodSpec(
        id="uhf", scf="uhf",
        spin="open", periodic=True, molecular=True,
    ),
    # GGA + hybrid DFT (vibe-qc supports LDA, PBE, BLYP, B3LYP via LibXC).
    "rks-blyp": MethodSpec(
        id="rks-blyp", scf="rks", xc="blyp",
        spin="closed", periodic=False, molecular=True,
    ),
    "rks-b3lyp": MethodSpec(
        id="rks-b3lyp", scf="rks", xc="b3lyp",
        spin="closed", periodic=False, molecular=True,
    ),
    "uks-blyp": MethodSpec(
        id="uks-blyp", scf="uks", xc="blyp",
        spin="open", periodic=False, molecular=True,
    ),
    "uks-b3lyp": MethodSpec(
        id="uks-b3lyp", scf="uks", xc="b3lyp",
        spin="open", periodic=False, molecular=True,
    ),
    # Post-HF correlation (vibe-qc has run_mp2 + run_ump2; molecular only).
    "mp2": MethodSpec(
        id="mp2", scf="rhf", post="mp2",
        spin="closed", periodic=False, molecular=True,
    ),
    "ump2": MethodSpec(
        id="ump2", scf="uhf", post="mp2",
        spin="open", periodic=False, molecular=True,
    ),
    # Density-fitting variants. Molecular-only for now (PySCF.pbc + ORCA
    # periodic DF needs separate plumbing). vibe-qc lacks DF entirely;
    # these cases test pyscf-DF vs orca-DF (and psi4-DF when available)
    # cross-code agreement and document the vibe-qc gap.
    "rhf-df": MethodSpec(
        id="rhf-df", scf="rhf", df=True,
        spin="closed", periodic=False, molecular=True,
    ),
    "rks-lda-df": MethodSpec(
        id="rks-lda-df", scf="rks", xc="lda", df=True,
        spin="closed", periodic=False, molecular=True,
    ),
    "rks-pbe-df": MethodSpec(
        id="rks-pbe-df", scf="rks", xc="pbe", df=True,
        spin="closed", periodic=False, molecular=True,
    ),
    "rks-b3lyp-df": MethodSpec(
        id="rks-b3lyp-df", scf="rks", xc="b3lyp", df=True,
        spin="closed", periodic=False, molecular=True,
    ),
    "mp2-df": MethodSpec(
        id="mp2-df", scf="rhf", post="mp2", df=True,
        spin="closed", periodic=False, molecular=True,
    ),
}
