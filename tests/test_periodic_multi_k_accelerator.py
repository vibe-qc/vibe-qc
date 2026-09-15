"""Multi-k periodic SCF accelerator helper — construction + first-call
contracts.

End-to-end multi-k SCF parity (every accelerator must converge to the
same fixed point on a small periodic system) lands with M2e once the
multi-k Ewald drivers are wired to
:class:`MultiKPeriodicSCFAccelerator` in M2d. The tests here pin the
mechanical contracts the wiring will rely on:

  * Construction for every mode in the ``SCFAccelerator`` family.
  * First call returns the input Fock list unchanged (need at least
    two iterates to extrapolate; the per-k :class:`_MultiKPulayDIIS`
    and :class:`_MultiKKDIIS` follow the same Pulay convention as the
    Γ-point classes from M1).
  * ``subspace_size`` advances after each call.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_scf_accelerators import (
    MultiKPeriodicSCFAccelerator,
    MultiKPeriodicUHFAccelerator,
)


_CLOSED_SHELL_ACCELERATORS = [
    vq.SCFAccelerator.DIIS,
    vq.SCFAccelerator.KDIIS,
    vq.SCFAccelerator.EDIIS,
    vq.SCFAccelerator.EDIIS_DIIS,
    vq.SCFAccelerator.ADIIS,
    vq.SCFAccelerator.ADIIS_DIIS,
    vq.SCFAccelerator.R_CDIIS,
    vq.SCFAccelerator.AD_CDIIS,
]
_BRIDGED_ACCELERATORS = [
    vq.SCFAccelerator.EDIIS,
    vq.SCFAccelerator.EDIIS_DIIS,
    vq.SCFAccelerator.ADIIS,
    vq.SCFAccelerator.ADIIS_DIIS,
]


def _h2_chain_1d_and_kmesh(nk: int = 4):
    sysp = vq.PeriodicSystem(
        1, np.diag([6.0, 30.0, 30.0]),
        [vq.Atom(1, [0.0, 15.0, 15.0]),
         vq.Atom(1, [1.4, 15.0, 15.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [nk, 1, 1])
    return sysp, basis, km


def _make_opts(accel):
    o = vq.PeriodicRHFOptions()
    o.scf_accelerator = accel
    o.diis_subspace_size = 6
    return o


def _hermitian_complex(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (A + A.conj().T)


@pytest.mark.parametrize("accel", _CLOSED_SHELL_ACCELERATORS)
def test_multi_k_accelerator_constructs(accel):
    """Every accelerator family member must construct without raising
    when handed a PeriodicRHFOptions whose scf_accelerator is that
    member. ``subspace_size`` starts at 0."""
    opts = _make_opts(accel)
    a = MultiKPeriodicSCFAccelerator(opts)
    assert a.subspace_size == 0


_NATIVE_PER_K_ACCELERATORS = [
    vq.SCFAccelerator.DIIS,
    vq.SCFAccelerator.KDIIS,
    vq.SCFAccelerator.R_CDIIS,
    vq.SCFAccelerator.AD_CDIIS,
]
# Pulay first-call semantics — DIIS / KDIIS return the input Fock list
# untouched while the rolling history has fewer than two entries. The
# bridged accelerators (EDIIS / ADIIS / EDIIS_DIIS) also obey this
# convention through the C++ block-vector ``EDIIS::extrapolate_blocks``
# / ``ADIIS::extrapolate_blocks`` kernels, which return the input Fock
# unchanged at history depth 1. Test this for the native paths since
# the bridged paths need real (Hermitian, MO-consistent) inputs for
# the QP to behave physically — the random Hermitian inputs in
# :func:`_make_inputs` are fine for *construction* and *first-call
# Pulay convention* but the QP behaviour on synthetic noise is not a
# meaningful test (and is covered end-to-end against the SCF
# reference by ``test_periodic_accelerator_uniformity.py``).


def _make_inputs():
    sysp, basis, km = _h2_chain_1d_and_kmesh(nk=4)
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 12.0
    lat_opts.nuclear_cutoff_bohr = 15.0
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    n_bf = S_lat.blocks[0].shape[0]
    n_k = len(km.kpoints)
    F_k = [_hermitian_complex(n_bf, seed=100 + i) for i in range(n_k)]
    err_k = [_hermitian_complex(n_bf, seed=200 + i) for i in range(n_k)]
    D_k = [_hermitian_complex(n_bf, seed=300 + i) for i in range(n_k)]
    C_k = [
        _hermitian_complex(n_bf, seed=400 + i) + 1e-3 * np.eye(n_bf)
        for i in range(n_k)
    ]
    return {
        "F_k": F_k, "err_k": err_k, "D_k": D_k, "C_k": C_k,
        "cells": S_lat.cells, "kpoints": list(km.kpoints),
        "weights": list(km.weights),
    }


@pytest.mark.parametrize("accel", _NATIVE_PER_K_ACCELERATORS)
def test_multi_k_native_first_call_returns_input_unchanged(accel):
    """Pulay convention on the per-k native paths (DIIS, KDIIS): the
    first iterate has no history to extrapolate against, so
    ``extrapolate_rhf`` returns the input Fock list element-wise."""
    d = _make_inputs()
    opts = _make_opts(accel)
    a = MultiKPeriodicSCFAccelerator(opts)
    out = a.extrapolate_rhf(
        d["F_k"],
        error_k_list=d["err_k"],
        density_k_list=d["D_k"],
        energy=-1.0,
        mo_coeffs_k_list=d["C_k"],
        n_occ=1,
        weights=d["weights"],
        cells=d["cells"],
        kpoints=d["kpoints"],
    )
    assert len(out) == len(d["F_k"])
    for o, f in zip(out, d["F_k"]):
        np.testing.assert_allclose(o, f, atol=1e-12,
            err_msg=f"first call did not return input for {accel.name}")


@pytest.mark.parametrize("accel", _BRIDGED_ACCELERATORS)
def test_multi_k_bridged_paths_construct_and_first_call_returns_input(accel):
    """The bridged accelerators (EDIIS / ADIIS / EDIIS_DIIS) construct
    and obey the Pulay first-call convention through the C++
    ``EDIIS::extrapolate_blocks`` / ``ADIIS::extrapolate_blocks``
    kernels: the first iterate has only itself in history, so the
    coefficient set is ``c = [1.0]`` and the extrapolated Fock equals
    the input (per stacked-real-block; unstacked per-k Hermitian
    matches the input bit-for-bit modulo the √w_k round-trip).

    The stacked-real-block bridge introduced in M2e
    (:func:`per_k_to_stacked_real_blocks` /
    :func:`stacked_real_blocks_to_per_k`) replaces the rejected
    per-cell Bloch bridge whose ``Σ_k w_k²`` quadratic-cross-term
    weighting misweighted the QP's energy functional relative to
    the linear term in Hartree — see the
    "per-k ↔ stacked-real-block bridge" comment in
    :mod:`vibeqc.periodic_scf_accelerators` for the derivation.
    """
    d = _make_inputs()
    opts = _make_opts(accel)
    a = MultiKPeriodicSCFAccelerator(opts)
    out = a.extrapolate_rhf(
        d["F_k"],
        error_k_list=d["err_k"],
        density_k_list=d["D_k"],
        energy=-1.0,
        mo_coeffs_k_list=d["C_k"],
        n_occ=1,
        weights=d["weights"],
        cells=d["cells"],
        kpoints=d["kpoints"],
    )
    assert len(out) == len(d["F_k"])
    for o, f in zip(out, d["F_k"]):
        np.testing.assert_allclose(
            o, f, atol=1e-10,
            err_msg=f"first call did not return input for {accel.name}",
        )


@pytest.mark.parametrize("accel", _NATIVE_PER_K_ACCELERATORS)
def test_multi_k_accelerator_subspace_grows(accel):
    """After two extrapolate calls every mode reports subspace_size
    ≥ 1 (DIIS / KDIIS) or ≥ 2 (block-vector EDIIS / ADIIS use the
    C++ subspace counter). This catches a class of wiring bugs where
    the history isn't being pushed."""
    sysp, basis, km = _h2_chain_1d_and_kmesh(nk=4)
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 12.0
    lat_opts.nuclear_cutoff_bohr = 15.0
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    n_bf = S_lat.blocks[0].shape[0]
    n_k = len(km.kpoints)

    F_k = [_hermitian_complex(n_bf, seed=500 + i) for i in range(n_k)]
    D_k = [_hermitian_complex(n_bf, seed=600 + i) for i in range(n_k)]
    err_k = [_hermitian_complex(n_bf, seed=700 + i) for i in range(n_k)]
    C_k = [_hermitian_complex(n_bf, seed=800 + i) + 1e-3 * np.eye(n_bf)
            for i in range(n_k)]

    opts = _make_opts(accel)
    a = MultiKPeriodicSCFAccelerator(opts)
    for energy in (-1.0, -1.01):
        a.extrapolate_rhf(
            F_k,
            error_k_list=err_k,
            density_k_list=D_k,
            energy=energy,
            mo_coeffs_k_list=C_k,
            n_occ=1,
            weights=list(km.weights),
            cells=S_lat.cells,
            kpoints=list(km.kpoints),
        )
    assert a.subspace_size >= 2


def test_ediis_diis_switch_uses_intensive_metric():
    """The EDIIS_DIIS hybrid must compare the *intensive* RMS
    commutator metric against ``ediis_diis_switch_threshold``, not the
    size-extensive k-weighted Frobenius norm — the molecular-driver
    convention since ddd8d325 (``ediis_diis_switch_metric`` in
    cpp/include/vibeqc/ediis.hpp).

    Regression for the c-diamond fcc-primitive/STO-3G KRHF-GDF kmesh
    (2,2,2) stall (2026-07-09): with the raw norm at 0.19 > 0.1 the run
    was held in the EDIIS regime (whose QP then pinned the initial
    iterate — 8 frozen cycles), while the intensive metric 0.019 < 0.1
    says the run belongs to DIIS.

    Setup: k-weighted raw error norm 0.15 (> threshold 0.1) but
    intensive metric 0.15 / n_bf < 0.1. The hybrid's second call must
    return the DIIS branch — bit-identical to a plain-DIIS accelerator
    fed the same history — rather than the EDIIS-bridged result.
    """
    d = _make_inputs()
    n_bf = d["F_k"][0].shape[0]
    assert n_bf >= 2  # intensive metric = raw / n_bf < threshold below

    def _scaled_errors(err_k, target_raw_norm):
        raw = np.sqrt(sum(
            float(w) * float(np.real(np.vdot(e.ravel(), e.ravel())))
            for w, e in zip(d["weights"], err_k)
        ))
        return [e * (target_raw_norm / raw) for e in err_k]

    err_1 = _scaled_errors(d["err_k"], 0.15)
    err_2 = _scaled_errors(
        [_hermitian_complex(n_bf, seed=900 + i) for i in range(len(err_1))],
        0.15,
    )

    hybrid = MultiKPeriodicSCFAccelerator(
        _make_opts(vq.SCFAccelerator.EDIIS_DIIS)
    )
    diis_ref = MultiKPeriodicSCFAccelerator(
        _make_opts(vq.SCFAccelerator.DIIS)
    )

    out_h = out_d = None
    for it, (err, energy) in enumerate(
        ((err_1, -1.0), (err_2, -1.02)), start=1
    ):
        F_k = [f + 0.01 * it * np.eye(n_bf) for f in d["F_k"]]
        kwargs = dict(
            error_k_list=err,
            density_k_list=d["D_k"],
            energy=energy,
            mo_coeffs_k_list=d["C_k"],
            n_occ=1,
            weights=d["weights"],
            cells=d["cells"],
            kpoints=d["kpoints"],
        )
        out_h = hybrid.extrapolate_rhf(F_k, **kwargs)
        out_d = diis_ref.extrapolate_rhf(F_k, **kwargs)

    for o_h, o_d in zip(out_h, out_d):
        np.testing.assert_allclose(
            o_h, o_d, atol=1e-12,
            err_msg=(
                "EDIIS_DIIS hybrid did not take the DIIS branch below "
                "the intensive switch metric"
            ),
        )


@pytest.mark.parametrize("accel", _CLOSED_SHELL_ACCELERATORS)
def test_uhf_accelerator_constructs_for_every_mode(accel):
    """UHF multi-k accelerator constructs for every mode in the
    ``SCFAccelerator`` family. DIIS uses two per-k Pulay histories;
    KDIIS uses a spin-coupled :class:`_MultiKKDIISOpenShell`;
    EDIIS / ADIIS / EDIIS_DIIS bridge through the stacked-real-block
    representation introduced in M2e and feed a single spin-coupled
    history to the C++ block-vector kernel (mirroring the open-shell
    EDIIS / ADIIS convention)."""
    opts = vq.PeriodicSCFOptions()
    opts.scf_accelerator = accel
    opts.diis_subspace_size = 4
    a = MultiKPeriodicUHFAccelerator(opts)
    assert a.subspace_size == 0
