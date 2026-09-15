"""MP2 / UMP2 per-intermediate decomposition — self-consistency pins.

Companion to ``tests/test_parity_hf_dft.py``. ``vibeqc.parity``'s
:func:`decompose_energy_rmp2` / :func:`decompose_energy_ump2` thread an
already-computed MP2 result (HF reference + correlation + channel
split) into a single named-piece dict, so a cross-code MP2 comparison
can assert each piece independently and localise a discrepancy to:

* HF reference (E_nuc / E_1e / E_J / E_K — already pinned by
  ``test_parity_hf_dft.py``);
* MP2 correlation total (``e_corr``);
* same-spin / opposite-spin channel split (``e_ss`` / ``e_os`` for
  RMP2, ``e_aa`` / ``e_bb`` / ``e_ab`` for UMP2).

This file is the **self-consistency** half: no PySCF / ORCA, just
verifies that the decomposition's invariants hold to machine precision
on a representative slice (canonical RMP2 / RI-MP2 / canonical UMP2 /
RI-UMP2 × {H2O closed shell, OH· doublet}). The cross-code parity
half — vibe-qc-RI-MP2 vs ORCA-RI-MP2 totals — is owned by the
cosx-perf chat's drive of ``examples/molecular/mp2_benchmarks``
(commit ``748baaa``); this file is the per-intermediate layer
underneath that benchmark.

Invariants asserted:

* ``e_total_residual`` = (``e_hf + e_corr``) - reported ``e_total``.
  Machine precision on canonical MP2 because the underlying C++
  result computes the same sum.
* ``e_channel_residual`` = (Σ channel values) - ``e_corr``. Machine
  precision on canonical MP2 (e_ss + e_os = e_corr by construction);
  non-zero on SCS / SOS scaled correlation (a different test
  surface, not covered here).
* The nested ``hf_decomp`` payload must itself self-consistently
  reconstruct the HF total — pinned via the same residual gate the
  HF/DFT self-consistency family uses.
"""

from __future__ import annotations

import pytest

from vibeqc import (
    Atom, BasisSet, InitialGuess, Molecule,
    MP2Options, RHFOptions, UHFOptions, UMP2Options,
    run_mp2, run_rhf, run_uhf, run_ump2,
)
from vibeqc.parity import decompose_energy_rmp2, decompose_energy_ump2


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_A = ANGSTROM_TO_BOHR

# Same canonical geometries as test_parity_hf_dft.py's _GEOMS — keep
# both files in lockstep so cells refer to identical nuclear
# coordinates.
H2O = [(8, [0.0, 0.0, 0.0]),
       (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
       (1, [0.0, -0.793353 * _A, -0.613510 * _A])]
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * _A])]

# Closed-shell aux basis (Weigend RIfit); both vibe-qc and ORCA accept
# this name. Sufficient to exercise the DF-MP2 path here; the
# benchmark-axis basis-pair coverage is in mp2_benchmarks.
_AUX = "cc-pvtz-ri"


# (sysname, basis, density_fit, aux_basis)
_RMP2_CASES = [
    ("H2O", "def2-svp", False, ""),
    ("H2O", "cc-pvdz", False, ""),
    ("H2O", "cc-pvtz", True, _AUX),       # RI-MP2
]
_UMP2_CASES = [
    ("OH", "def2-svp", False, ""),
    ("OH", "cc-pvtz", True, _AUX),         # RI-UMP2
]


def _make_mol(atoms, *, mult: int = 1) -> Molecule:
    return Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms],
                    multiplicity=mult)


@pytest.mark.parametrize("sysname,basis_name,df,aux", _RMP2_CASES,
                         ids=[f"{s}-{b}{'-RI' if df else ''}"
                              for s, b, df, _ in _RMP2_CASES])
def test_rmp2_decomposition_self_consistent(sysname, basis_name, df, aux):
    """``decompose_energy_rmp2`` is self-consistent at machine precision."""
    atoms = {"H2O": H2O}[sysname]
    mol = _make_mol(atoms)
    basis = BasisSet(mol, basis_name)
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-9
    rhf_opts.initial_guess = InitialGuess.SAD
    rhf = run_rhf(mol, basis, rhf_opts)
    assert rhf.converged

    mp2_opts = MP2Options()
    if df:
        mp2_opts.density_fit = True
        mp2_opts.aux_basis = aux
    mp2 = run_mp2(mol, basis, rhf, mp2_opts)

    payload = decompose_energy_rmp2(
        mol, basis, rhf, mp2, density_fit=df, aux_basis=aux)

    # 1. The HF + correlation reconstruction must match the C++-side
    #    e_total exactly — both computations sum the same two floats.
    assert abs(payload["e_total_residual"]) < 1e-12, (
        f"{sysname}/{basis_name}{'/RI' if df else ''}: e_hf + e_corr "
        f"reconstruction disagrees with MP2Result.e_total by "
        f"{payload['e_total_residual']:.3e}"
    )

    # 2. Canonical MP2 stores e_correlation = e_ss + e_os exactly —
    #    same-spin and opposite-spin channels sum to the unscaled
    #    correlation by construction.
    assert abs(payload["e_channel_residual"]) < 1e-12, (
        f"{sysname}/{basis_name}{'/RI' if df else ''}: e_ss + e_os "
        f"channel sum disagrees with e_corr by "
        f"{payload['e_channel_residual']:.3e} — only SCS/SOS-scaled "
        f"correlation should exceed machine precision here."
    )

    # 3. The nested HF decomposition itself must self-consistently
    #    reconstruct the HF reference total — same 1e-9 floor the
    #    HF/DFT self-consistency family asserts.
    hf = payload["hf_decomp"]
    assert abs(hf["e_total_residual"]) < 1e-9, (
        f"{sysname}/{basis_name}: nested HF decomposition is not "
        f"self-consistent (residual {hf['e_total_residual']:.3e})"
    )


@pytest.mark.parametrize("sysname,basis_name,df,aux", _UMP2_CASES,
                         ids=[f"{s}-{b}{'-RI' if df else ''}"
                              for s, b, df, _ in _UMP2_CASES])
def test_ump2_decomposition_self_consistent(sysname, basis_name, df, aux):
    """``decompose_energy_ump2`` is self-consistent at machine precision."""
    atoms = {"OH": OH}[sysname]
    mol = _make_mol(atoms, mult=2)
    basis = BasisSet(mol, basis_name)
    uhf_opts = UHFOptions()
    uhf_opts.conv_tol_energy = 1e-11
    uhf_opts.conv_tol_grad = 1e-8
    uhf_opts.max_iter = 300
    uhf_opts.initial_guess = InitialGuess.SAD
    uhf = run_uhf(mol, basis, uhf_opts)
    assert uhf.converged

    ump2_opts = UMP2Options()
    if df:
        ump2_opts.density_fit = True
        ump2_opts.aux_basis = aux
    ump2 = run_ump2(mol, basis, uhf, ump2_opts)

    payload = decompose_energy_ump2(
        mol, basis, uhf, ump2, density_fit=df, aux_basis=aux)

    # Same three invariants as the closed-shell case.
    assert abs(payload["e_total_residual"]) < 1e-12, (
        f"{sysname}/{basis_name}{'/RI' if df else ''}: e_hf + e_corr "
        f"reconstruction disagrees with UMP2Result.e_total by "
        f"{payload['e_total_residual']:.3e}"
    )

    # UMP2 has a 3-channel split: αα + ββ + αβ = e_correlation.
    assert abs(payload["e_channel_residual"]) < 1e-12, (
        f"{sysname}/{basis_name}{'/RI' if df else ''}: e_aa + e_bb + "
        f"e_ab channel sum disagrees with e_corr by "
        f"{payload['e_channel_residual']:.3e}"
    )

    hf = payload["hf_decomp"]
    assert abs(hf["e_total_residual"]) < 1e-9, (
        f"{sysname}/{basis_name}: nested UHF decomposition is not "
        f"self-consistent (residual {hf['e_total_residual']:.3e})"
    )
