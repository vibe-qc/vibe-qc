"""Frozen-natural-orbital (FNO) DF-CCSD(T).

FNO truncates the virtual space to the dominant MP2 natural orbitals before
CCSD(T) (DePrince and Sherrill, J. Chem. Theory Comput. 9, 2687 (2013),
doi:10.1021/ct400250u).  The decisive always-on gate is invariance: with no
truncation the FNO path reproduces canonical CCSD(T) to machine precision,
because CCSD is invariant to virtual rotation and the mandatory
semicanonicalization restores the canonical-orbital (T).  The C++ entry point
is run_ccsd_from_mos (cpp/src/ccsd.cpp); the FNO orchestration is
vibeqc.cc.run_fno_ccsd.
"""

from __future__ import annotations

import weakref

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, run_job, run_rhf
from vibeqc.cc import (
    CCSDOptions,
    chemical_core_orbital_count,
    run_ccsd,
    run_fno_ccsd,
)

from .conftest import GEOMETRIES

AUX = "cc-pvdz-ri"
MICRO_HA = 1e-6


def _rhf(mol_key="H2O", basis_name="cc-pvdz"):
    atoms = GEOMETRIES[mol_key]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, basis_name)
    o = RHFOptions()
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    o.density_fit = True
    o.aux_basis = AUX
    hf = run_rhf(mol, basis, o)
    assert hf.converged
    return mol, basis, hf


def _canonical(mol, basis, hf, frozen=1, triples="(t)"):
    opts = CCSDOptions(
        aux_basis=AUX,
        triples=triples,
        n_frozen_core=frozen,
    )
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_residual = 1e-9
    return run_ccsd(mol, basis, hf, opts)


def _fno(mol, basis, hf, *, frozen=1, keep_fraction=None, occ_threshold=1e-5,
         delta_mp2=True, triples="(t)"):
    opts = CCSDOptions(
        aux_basis=AUX, triples=triples, n_frozen_core=frozen,
        fno=True, fno_keep_fraction=keep_fraction,
        fno_occ_threshold=occ_threshold, fno_delta_mp2=delta_mp2,
    )
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_residual = 1e-9
    return run_fno_ccsd(mol, basis, hf, opts)


def test_fno_no_truncation_matches_canonical():
    """keep_fraction=1.0 (no truncation) reproduces canonical CCSD(T) to uHa.

    This is the invariance gate: the full NO rotation + semicanonicalization
    is unitary on the virtual space, so CCSD is unchanged, and the
    semicanonical (T) equals the canonical (T).  Always-on (no PySCF).
    """
    mol, basis, hf = _rhf()
    can = _canonical(mol, basis, hf)
    fno = _fno(mol, basis, hf, keep_fraction=1.0)

    assert fno.converged
    assert fno.n_virtual_kept == fno.n_virtual_total
    assert abs(fno.delta_mp2) < MICRO_HA  # no truncation -> no delta-MP2
    assert abs(fno.e_ccsd_correlation - can.e_ccsd_correlation) < MICRO_HA
    assert abs(fno.e_t - can.e_t) < MICRO_HA
    assert abs(fno.e_ccsd_t - can.e_ccsd_t) < MICRO_HA


def test_fno_zero_threshold_matches_canonical():
    """occ_threshold=0 keeps every NO and reproduces canonical CCSD(T)."""
    mol, basis, hf = _rhf()
    can = _canonical(mol, basis, hf)
    fno = _fno(mol, basis, hf, occ_threshold=0.0)
    assert fno.n_virtual_kept == fno.n_virtual_total
    assert abs(fno.e_ccsd_t - can.e_ccsd_t) < MICRO_HA


def test_fno_accsdt_no_truncation_matches_canonical():
    """A full-space FNO rotation preserves A-CCSD(T) and Lambda convergence."""
    mol, basis, hf = _rhf()
    can = _canonical(mol, basis, hf, triples="A-CCSD(T)")
    fno = _fno(
        mol,
        basis,
        hf,
        keep_fraction=1.0,
        triples="A-CCSD(T)",
    )

    assert can.converged and fno.converged
    assert fno.lambda_residual_norm < 1e-8
    assert abs(fno.delta_mp2) < MICRO_HA
    assert abs(fno.e_ccsd_correlation - can.e_ccsd_correlation) < MICRO_HA
    assert abs(fno.e_t - can.e_t) < MICRO_HA
    assert abs(fno.e_ccsd_t - can.e_ccsd_t) < MICRO_HA


def test_fno_truncation_recovers_most_correlation():
    """Truncating to 60% of virtuals + delta-MP2 recovers most correlation."""
    mol, basis, hf = _rhf()
    can = _canonical(mol, basis, hf)
    fno = _fno(mol, basis, hf, keep_fraction=0.6)

    assert fno.converged
    assert fno.n_virtual_kept < fno.n_virtual_total
    # CCSD recovers less correlation than the full space; the truncated total
    # stays within a few mHa of canonical thanks to delta-MP2.
    assert fno.e_ccsd_t > can.e_ccsd_t          # less correlation (higher E)
    assert abs(fno.e_ccsd_t - can.e_ccsd_t) < 5e-3
    assert fno.delta_mp2 < 0.0                   # recovers lost MP2 correlation


def test_fno_delta_mp2_reduces_truncation_error():
    """delta-MP2 brings the truncated energy closer to canonical than raw."""
    mol, basis, hf = _rhf()
    can = _canonical(mol, basis, hf)
    raw = _fno(mol, basis, hf, keep_fraction=0.6, delta_mp2=False)
    cor = _fno(mol, basis, hf, keep_fraction=0.6, delta_mp2=True)
    assert abs(cor.e_ccsd_t - can.e_ccsd_t) < abs(raw.e_ccsd_t - can.e_ccsd_t)
    assert abs(raw.delta_mp2) == 0.0


def test_fno_releases_dense_mp2_setup_before_native_cc(monkeypatch):
    """Full/truncated B and g arrays are dead at the final native boundary."""
    import vibeqc.cc as cc_module
    import vibeqc.density_fitting as df_module

    mol, basis, hf = _rhf(basis_name="sto-3g")
    real_df = df_module.DensityFitting
    real_einsum = np.einsum
    b_refs = []
    g_refs = []

    class TrackingDF:
        def __init__(self, *args, **kwargs):
            self.inner = real_df(*args, **kwargs)

        def mo_transform(self, *args, **kwargs):
            value = self.inner.mo_transform(*args, **kwargs)
            b_refs.append(weakref.ref(value))
            return value

    def tracking_einsum(subscripts, *operands, **kwargs):
        value = real_einsum(subscripts, *operands, **kwargs)
        if subscripts == "Pia,Pjb->ijab":
            g_refs.append(weakref.ref(value))
        return value

    class NativeBoundaryReached(Exception):
        pass

    def final_native_call(*_args, **_kwargs):
        assert len(b_refs) == 2
        assert len(g_refs) == 2
        assert all(ref() is None for ref in b_refs)
        assert all(ref() is None for ref in g_refs)
        raise NativeBoundaryReached

    monkeypatch.setattr(df_module, "DensityFitting", TrackingDF)
    monkeypatch.setattr(np, "einsum", tracking_einsum)
    monkeypatch.setattr(cc_module, "_run_ccsd_from_mos", final_native_call)

    opts = CCSDOptions(
        aux_basis=AUX,
        compute_triples=False,
        n_frozen_core=0,
        fno=True,
        fno_keep_fraction=0.5,
        fno_delta_mp2=True,
    )
    with pytest.raises(NativeBoundaryReached):
        run_fno_ccsd(mol, basis, hf, opts)


def test_fno_routes_through_run_ccsd_options():
    """run_ccsd(options.fno=True) dispatches to the FNO path."""
    mol, basis, hf = _rhf()
    opts = CCSDOptions(aux_basis=AUX, compute_triples=True, n_frozen_core=1,
                       fno=True, fno_keep_fraction=1.0)
    res = run_ccsd(mol, basis, hf, opts)
    assert hasattr(res, "n_virtual_kept")        # FNOCCSDResult
    assert res.converged
    assert res.triples_memory_mode_used == "fast"
    assert res.triples_tile_size_used > 0
    assert res.triples_threads_used > 0
    assert res.triples_workspace_bytes > 0
    assert res.triples_disk_bytes == 0


@pytest.mark.parametrize("mode", ["direct", "disk"])
def test_fno_forwards_bounded_triples_plan_and_cleans_scratch(mode, tmp_path):
    mol, basis, hf = _rhf(basis_name="sto-3g")
    scratch = tmp_path / mode
    scratch.mkdir()
    opts = CCSDOptions(
        aux_basis=AUX,
        compute_triples=True,
        n_frozen_core=0,
        fno=True,
        fno_keep_fraction=0.5,
        triples_memory_mode=mode,
        requested_memory_bytes=64 * 1024**2,
        triples_tile_size=1,
        triples_max_threads=1,
        triples_scratch_directory=str(scratch),
    )

    result = run_fno_ccsd(mol, basis, hf, opts)

    assert result.converged
    assert result.triples_memory_mode_used == mode
    assert result.triples_workspace_bytes <= opts.requested_memory_bytes
    if mode == "disk":
        assert result.triples_disk_bytes > 0
    else:
        assert result.triples_disk_bytes == 0
    assert list(scratch.iterdir()) == []


def test_fno_via_run_job(tmp_path):
    """run_job(method='ccsd(t)', ccsd_options=CCSDOptions(fno=True)) works."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    res = run_job(
        mol, basis="cc-pvdz", method="ccsd(t)",
        ccsd_options=CCSDOptions(fno=True, fno_keep_fraction=1.0),
        output=str(tmp_path / "fno_via_run_job"),
    )
    assert res.ccsd.converged
    assert hasattr(res.ccsd, "n_virtual_kept")
    assert res.ccsd.n_frozen == chemical_core_orbital_count(mol)
    assert res.ccsd.e_t < 0.0


@pytest.mark.parametrize("method", ["ccsd", "ccsd(t)"])
def test_fno_run_job_cites_taube_bartlett(method, tmp_path):
    """The realized FNO composition retains its defining citation."""
    atoms = GEOMETRIES["H2"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    stem = tmp_path / method.replace("(", "_").replace(")", "")
    result = run_job(
        mol,
        basis="cc-pvdz",
        method=method,
        ccsd_options=CCSDOptions(fno=True, fno_keep_fraction=1.0),
        output=str(stem),
        verbose=0,
    )
    assert result.ccsd.converged
    references = stem.with_suffix(".references").read_text()
    assert "taube_bartlett_fno_2008" in references
    assert "10.1063/1.2902285" in references


def test_run_job_default_fno_matches_canonical_frozen_core(tmp_path):
    """Default FNO run_job uses the same frozen core as canonical CCSD(T)."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    n_core = chemical_core_orbital_count(mol)

    canonical = run_job(
        mol, basis="cc-pvdz", method="ccsd(t)",
        output=str(tmp_path / "default_fno_canonical"),
    )
    explicit_core = run_job(
        mol,
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=CCSDOptions(n_frozen_core=n_core),
        output=str(tmp_path / "default_fno_explicit_core"),
    )
    fno = run_job(
        mol,
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=CCSDOptions(fno=True, fno_keep_fraction=1.0),
        output=str(tmp_path / "default_fno_fno"),
    )

    assert fno.ccsd.n_frozen == n_core
    assert fno.ccsd.n_virtual_kept == fno.ccsd.n_virtual_total
    assert abs(fno.ccsd.delta_mp2) < MICRO_HA
    assert canonical.ccsd.e_ccsd_t == pytest.approx(
        explicit_core.ccsd.e_ccsd_t, abs=MICRO_HA
    )
    assert fno.ccsd.e_ccsd_t == pytest.approx(
        canonical.ccsd.e_ccsd_t, abs=MICRO_HA
    )


def test_run_job_fno_preserves_explicit_all_electron_core(tmp_path):
    """Explicit n_frozen_core=0 must not be coerced to run_job's default core.

    The release-paper M12 FNO audit passes CCSDOptions(fno=True,
    n_frozen_core=0). Pre-fix, run_job could not distinguish that explicit
    all-electron request from the wrapper default and silently restored the
    chemical-core freeze, producing the frozen-core-sized table outlier.
    """
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])

    canonical = run_job(
        mol,
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=CCSDOptions(n_frozen_core=0),
        output=str(tmp_path / "all_electron_canonical"),
    )
    fno = run_job(
        mol,
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=CCSDOptions(
            fno=True,
            fno_keep_fraction=1.0,
            n_frozen_core=0,
        ),
        output=str(tmp_path / "all_electron_fno"),
    )

    assert fno.ccsd.n_frozen == 0
    assert fno.ccsd.n_virtual_kept == fno.ccsd.n_virtual_total
    assert abs(fno.ccsd.delta_mp2) < MICRO_HA
    assert fno.ccsd.e_ccsd_t == pytest.approx(
        canonical.ccsd.e_ccsd_t, abs=MICRO_HA
    )


def test_fno_rejects_open_shell():
    """FNO-CCSD requires a closed-shell reference."""
    mol, basis, hf = _rhf()
    open_mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES["H2O"]],
        charge=0, multiplicity=3,
    )
    with pytest.raises(ValueError, match="closed-shell"):
        run_fno_ccsd(open_mol, basis, hf, CCSDOptions(aux_basis=AUX, fno=True))


def test_fno_rejects_bad_keep_fraction():
    mol, basis, hf = _rhf()
    with pytest.raises(ValueError, match="fno_keep_fraction"):
        run_fno_ccsd(mol, basis, hf,
                     CCSDOptions(aux_basis=AUX, fno=True, fno_keep_fraction=1.5))
