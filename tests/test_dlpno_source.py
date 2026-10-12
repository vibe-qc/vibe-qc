"""The correlation-source contract, increment A of
docs/design_dlpno_correlation_source.md.

The molecular adapter is a re-packaging of the engine's inputs: every result
obtained through a source equals the legacy positional call to the bit, the
overlap the source carries is the one the engine uses, and the source
identity reaches the result, the ``.system`` manifest and the structured
log.
"""

from __future__ import annotations

import json
import tomllib

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, UHFOptions, run_job, run_rhf, run_uhf
from vibeqc._vibeqc_core import Atom, Molecule, compute_overlap
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd
from vibeqc.dlpno.source import (
    CONTRACT_VERSION,
    CorrelationSource,
    CorrelationSourceError,
    molecular_source,
    source_identity,
)
from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions, run_local_dlpno_uccsd

AUX = "def2-svp-rifit"
# Bohr. Water and the hydroxyl radical, small enough for seconds in STO-3G.
H2O = [(8, [0.0, 0.0, 0.2217]), (1, [0.0, 1.4305, -0.8867]), (1, [0.0, -1.4305, -0.8867])]
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.832])]


def _closed_shell(atoms=H2O, basis_name="sto-3g"):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    return mol, basis, rhf, df


def _open_shell(atoms=OH, basis_name="sto-3g"):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=2)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.max_iter = 200
    uhf = run_uhf(mol, basis, opts)
    assert uhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    return mol, basis, uhf, df


def _is_sha256(text):
    return isinstance(text, str) and len(text) == 64 and all(c in "0123456789abcdef" for c in text)


class TestIdentity:
    def test_identity_is_a_stable_sha256_of_the_provenance(self):
        mol, basis, rhf, df = _closed_shell()
        first = molecular_source(mol, basis, rhf, df)
        second = molecular_source(mol, basis, rhf, df)
        assert first.kind == "molecular-df"
        assert first.reference_kind == "rhf"
        assert _is_sha256(first.identity)
        assert first.identity == second.identity
        assert first.identity == source_identity(first.provenance)
        assert first.provenance["aux_basis"] == AUX
        assert first.provenance["basis"] == "sto-3g"
        assert len(first.provenance["atoms"]) == 3
        assert source_identity({"contract": CONTRACT_VERSION}) == source_identity({})

    def test_identity_changes_with_the_auxiliary_basis_and_the_geometry(self):
        mol, basis, rhf, df = _closed_shell()
        base = molecular_source(mol, basis, rhf, df)
        other_aux = molecular_source(mol, basis, rhf, df, aux_basis_name="cc-pvdz-ri")
        assert other_aux.identity != base.identity
        moved = [(8, [0.0, 0.0, 0.2217]), (1, [0.0, 1.45, -0.8867]), (1, [0.0, -1.45, -0.8867])]
        mol2, basis2, rhf2, df2 = _closed_shell(moved)
        assert molecular_source(mol2, basis2, rhf2, df2).identity != base.identity

    def test_open_shell_reference_is_detected(self):
        mol, basis, uhf, df = _open_shell()
        source = molecular_source(mol, basis, uhf, df)
        assert source.reference_kind == "uhf"
        assert source.provenance["multiplicity"] == 2

    def test_rejects_an_unknown_reference_kind_and_factors_without_a_transform(self):
        mol, basis, rhf, df = _closed_shell()
        with pytest.raises(CorrelationSourceError, match="reference_kind"):
            CorrelationSource(
                kind="molecular-df", reference_kind="rks", molecule=mol,
                basis=basis, reference=rhf, factors=df,
            )
        with pytest.raises(CorrelationSourceError, match="mo_transform"):
            CorrelationSource(
                kind="molecular-df", reference_kind="rhf", molecule=mol,
                basis=basis, reference=rhf, factors=object(),
            )
        with pytest.raises(CorrelationSourceError, match="square"):
            CorrelationSource(
                kind="molecular-df", reference_kind="rhf", molecule=mol,
                basis=basis, reference=rhf, factors=df,
                overlap=np.zeros((2, 3)),
            )


class TestEngineThroughTheSource:
    def test_closed_shell_results_are_bit_identical(self):
        mol, basis, rhf, df = _closed_shell()
        options = LocalCCSDOptions(compute_triples=True)
        legacy = run_local_dlpno_ccsd(mol, basis, rhf, df, options)
        source = molecular_source(mol, basis, rhf, df)
        via = run_local_dlpno_ccsd(None, None, None, None, options, source=source)
        assert via.converged and legacy.converged
        assert via.e_corr == legacy.e_corr
        assert via.e_t == legacy.e_t
        assert via.e_total == legacy.e_total
        assert via.n_pairs == legacy.n_pairs
        assert via.source_kind == legacy.source_kind == "molecular-df"
        assert via.source_identity == legacy.source_identity == source.identity

    def test_the_source_overlap_is_the_one_the_engine_uses(self):
        mol, basis, rhf, df = _closed_shell()
        options = LocalCCSDOptions()
        implicit = molecular_source(mol, basis, rhf, df)
        assert implicit.overlap is None
        explicit = CorrelationSource(
            kind="molecular-df", reference_kind="rhf", molecule=mol,
            basis=basis, reference=rhf, factors=df,
            overlap=np.asarray(compute_overlap(basis)),
            provenance=dict(implicit.provenance),
        )
        assert explicit.identity == implicit.identity
        assert explicit.overlap_matrix() is explicit.overlap
        a = run_local_dlpno_ccsd(None, None, None, None, options, source=implicit)
        b = run_local_dlpno_ccsd(None, None, None, None, options, source=explicit)
        assert a.e_corr == b.e_corr

    def test_rejects_a_source_of_the_wrong_type(self):
        mol, basis, rhf, df = _closed_shell()
        with pytest.raises(TypeError, match="CorrelationSource"):
            run_local_dlpno_ccsd(mol, basis, rhf, df, LocalCCSDOptions(), source=df)

    def test_open_shell_results_are_bit_identical(self):
        mol, basis, uhf, df = _open_shell()
        options = LocalUCCSDOptions(compute_triples=True)
        legacy = run_local_dlpno_uccsd(mol, basis, uhf, df, options)
        source = molecular_source(mol, basis, uhf, df)
        via = run_local_dlpno_uccsd(None, None, None, None, options, source=source)
        assert via.converged and legacy.converged
        assert via.e_corr == legacy.e_corr
        assert via.e_t == legacy.e_t
        assert via.source_kind == legacy.source_kind == "molecular-df"
        assert via.source_identity == legacy.source_identity == source.identity


class TestRunJobRecordsTheSource:
    def test_closed_shell_manifest_and_log_carry_the_identity(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], charge=0, multiplicity=1)
        res = run_job(mol, basis="def2-svp", method="dlpno-ccsd", output="h2", structured_log=True)
        identity = res.dlpno_ccsd.source_identity
        assert _is_sha256(identity)
        manifest = tomllib.loads((tmp_path / "h2.system").read_text())
        assert manifest["run"]["dlpno_source_kind"] == "molecular-df"
        assert manifest["run"]["dlpno_source_identity"] == identity
        records = [
            json.loads(line)
            for line in (tmp_path / "h2.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(row for row in records if row["event"] == "dlpno_ccsd_done")
        assert done["source_kind"] == "molecular-df"
        assert done["source_identity"] == identity

    def test_open_shell_manifest_and_log_carry_the_identity(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mol = Molecule([Atom(z, p) for z, p in OH], charge=0, multiplicity=2)
        res = run_job(
            mol, basis="sto-3g", method="dlpno-ccsd", output="oh", structured_log=True,
            dlpno_ccsd_options=LocalUCCSDOptions(aux_basis="cc-pvdz-ri"),
        )
        identity = res.dlpno_ccsd.source_identity
        assert _is_sha256(identity)
        manifest = tomllib.loads((tmp_path / "oh.system").read_text())
        assert manifest["run"]["dlpno_source_kind"] == "molecular-df"
        assert manifest["run"]["dlpno_source_identity"] == identity
        records = [
            json.loads(line)
            for line in (tmp_path / "oh.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(row for row in records if row["event"] == "dlpno_uccsd_done")
        assert done["source_identity"] == identity
