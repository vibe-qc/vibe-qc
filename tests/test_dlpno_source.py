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


@pytest.mark.parametrize("changed", ["factors", "fock", "overlap", "orbitals"])
def test_real_torus_identity_tracks_actual_numerical_source(changed):
    from types import SimpleNamespace

    from vibeqc import PeriodicSystem
    from vibeqc.dlpno.source import LpqFactors, SourceConventions, _real_torus_source

    system = PeriodicSystem(3, np.eye(3) * 8,
                            [Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    arrays = dict(factors=np.ones((1, 2, 2)), fock=np.diag([-1., .5]),
                  overlap=np.eye(2), orbitals=np.eye(2))
    reference = SimpleNamespace(
        converged=True, energy=-1., fock=arrays["fock"], overlap=arrays["overlap"],
        mo_coeffs=arrays["orbitals"],
    )

    def build():
        return _real_torus_source(
            system, basis, reference, LpqFactors(arrays["factors"]),
            kind="test-real-torus",
            conventions=SourceConventions(exchange_q0="bvk-ewald", kmesh=(1, 1, 1)),
            factor_tensor=arrays["factors"], factor_representation="full-auxiliary",
            aux_basis_name="test-auxiliary",
        )

    source = build()
    assert source.identity == build().identity
    assert source.identity == source_identity(source.provenance)
    arrays[changed].flat[0] += .001
    # Geometry, basis names, reference energy and descriptor are unchanged.
    assert source.identity != build().identity


@pytest.mark.parametrize("changed", [
    "fock_alpha", "fock_beta", "mo_coeffs_alpha", "mo_coeffs_beta",
    "mo_energies_alpha", "mo_energies_beta", "overlap", "factors",
])
def test_real_uhf_torus_source_preserves_and_identifies_both_spins(changed):
    from types import SimpleNamespace

    from vibeqc import PeriodicSystem
    from vibeqc.dlpno.source import SourceConventions, _real_uhf_torus_source

    system = PeriodicSystem(3, np.eye(3) * 8,
                            [Atom(2, [0., 0., 0.]), Atom(1, [0., 0., 1.4])],
                            multiplicity=2)
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    arrays = dict(
        fock_alpha=np.diag([-1., -.2]), fock_beta=np.diag([-.8, .4]),
        mo_coeffs_alpha=np.eye(2), mo_coeffs_beta=np.array([[0., 1.], [1., 0.]]),
        mo_energies_alpha=np.array([-1., -.2]), mo_energies_beta=np.array([-.8, .4]),
        overlap=np.diag([1., 1.1]), factors=np.ones((1, 2, 2)),
    )
    reference = SimpleNamespace(
        converged=True, energy=-2., ecp_total_ncore=0,
        **{k: v for k, v in arrays.items() if k != "factors"},
    )

    class CompactFactors:
        n_aux = 2

        @property
        def three_center(self):
            raise AssertionError("provenance must not expand compact factors")

        def mo_transform(self, left, right=None):
            raise AssertionError("provenance must not transform factors")

    factors = CompactFactors()

    def build():
        return _real_uhf_torus_source(
            system, basis, reference, factors, kind="test-uhf-torus",
            conventions=SourceConventions(exchange_q0="bvk-ewald", kmesh=(1, 1, 2),
                                          same_source=True),
            factor_tensor=arrays["factors"], factor_representation="home-auxiliary",
            aux_basis_name="test-fit",
        )

    before = {k: v.copy() for k, v in arrays.items()}
    first = build()
    assert first.reference is reference
    assert first.reference_kind == "uhf"
    assert first.factors is factors
    assert first.provenance["multiplicity"] == 2
    assert first.provenance["factors"]["shape"] == [1, 2, 2]
    assert first.provenance["n_aux"] == 2
    assert first.provenance["spin_channels"]["alpha"] != first.provenance["spin_channels"]["beta"]
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, before[key])
    arrays[changed].flat[0] += .01
    assert build().identity != first.identity


@pytest.mark.parametrize("invalid", ["rhf", "unconverged", "beta_shape", "beta_complex"])
def test_real_uhf_torus_source_refuses_invalid_spin_state_without_mutation(invalid):
    from types import SimpleNamespace

    from vibeqc import PeriodicSystem
    from vibeqc.dlpno.source import LpqFactors, SourceConventions, _real_uhf_torus_source

    system = PeriodicSystem(3, np.eye(3) * 8,
                            [Atom(2, [0., 0., 0.]), Atom(1, [0., 0., 1.4])],
                            multiplicity=2)
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    reference = SimpleNamespace(
        converged=True, energy=-2., overlap=np.eye(2),
        mo_coeffs_alpha=np.eye(2), mo_coeffs_beta=np.eye(2),
        fock_alpha=np.diag([-1., -.2]), fock_beta=np.diag([-.8, .4]),
        mo_energies_alpha=np.array([-1., -.2]), mo_energies_beta=np.array([-.8, .4]),
    )
    if invalid == "rhf":
        reference.reference_kind = "rhf"
    elif invalid == "unconverged":
        reference.converged = False
    elif invalid == "beta_shape":
        reference.mo_coeffs_beta = np.ones((1, 2))
    else:
        reference.fock_beta = reference.fock_beta.astype(complex) + .1j
    before = {k: v.copy() for k, v in vars(reference).items() if isinstance(v, np.ndarray)}
    factors = LpqFactors(np.ones((1, 2, 2)))
    with pytest.raises(CorrelationSourceError):
        _real_uhf_torus_source(
            system, basis, reference, factors, kind="test-uhf-torus",
            conventions=SourceConventions(), factor_tensor=factors.three_center,
            factor_representation="home-auxiliary", aux_basis_name="test-fit",
        )
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(reference, key), value)


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


def _translation_pair_metric_model(mesh=(4, 1, 1), n_bands=2):
    """Cell-local virtual domains in a nonorthogonal finite AO frame."""
    from itertools import product
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    cells = tuple(product(*(range(n) for n in mesh)))
    n_cells = len(cells)
    n_local = n_bands + 2
    metric_frame = np.kron(np.eye(n_cells), np.diag(np.linspace(0.85, 1.2, n_local)))
    lookup = {cell: k for k, cell in enumerate(cells)}
    for cell, k in lookup.items():
        neighbor = ((cell[0] + 1) % mesh[0], cell[1], cell[2])
        metric_frame[k*n_local:(k+1)*n_local,
                     lookup[neighbor]*n_local:(lookup[neighbor]+1)*n_local] += .035*np.eye(n_local)
    coeff = np.linalg.inv(metric_frame)
    overlap = metric_frame.T @ metric_frame
    energies = np.tile(np.r_[np.linspace(-1.4, -.7, n_bands), .4, .9], n_cells)
    fock = metric_frame.T @ np.diag(energies) @ metric_frame
    occupied_indices = [cell*n_local+b for cell in range(n_cells) for b in range(n_bands)]
    occupied = coeff[:, occupied_indices]
    plan = build_translation_pair_plan(n_bands, mesh)
    rng = np.random.default_rng(1925)
    pair_coefficients = []
    for displacement in plan.displacements:
        domain = sorted({0, lookup[tuple(displacement)]})
        indices = [cell*n_local+b for cell in domain for b in range(n_bands, n_local)]
        rotation = np.linalg.qr(rng.normal(size=(len(indices), len(indices))))[0]
        pair_coefficients.append(coeff[:, indices] @ rotation)
    return overlap, fock, occupied, pair_coefficients


def _translate_pair_test_columns(coefficients, shift, mesh):
    n_local = coefficients.shape[0] // np.prod(mesh)
    shaped = coefficients.reshape(*mesh, n_local, coefficients.shape[1])
    return np.roll(shaped, shift, axis=(0, 1, 2)).reshape(coefficients.shape)


@pytest.mark.parametrize('mesh,n_bands', [((3, 1, 1), 1), ((4, 1, 1), 2), ((2, 2, 1), 2)])
def test_translation_pair_transport_matches_unreduced_ao_spaces(mesh, n_bands):
    from itertools import product
    from vibeqc.dlpno.translation import TranslationPairSpaces

    overlap, fock, occupied, bases = _translation_pair_metric_model(mesh, n_bands)
    snapshots = [a.copy() for a in [overlap, fock, occupied, *bases]]
    spaces = TranslationPairSpaces(mesh, overlap, fock, occupied, bases)
    cells = tuple(product(*(range(n) for n in mesh)))
    n_occ = occupied.shape[1]
    all_unordered = set()
    for row, (i, j) in enumerate(spaces.representative_pairs):
        members = set()
        for shift in cells:
            translated = _translate_pair_test_columns(occupied[:, [i, j]], shift, mesh)
            images = tuple(np.argmax(np.abs(occupied.T @ overlap @ translated), axis=0))
            members.add(tuple(sorted(images)))
        assert not all_unordered.intersection(members)
        all_unordered.update(members)
        assert spaces.energy_weights[row] == pytest.approx(len(members)*(2-(i == j))/len(cells))
    assert all_unordered == {(i, j) for i in range(n_occ) for j in range(i, n_occ)}
    assert np.sum(spaces.energy_weights) == pytest.approx(n_occ*n_occ/len(cells))

    placed = {}
    for i in range(n_occ):
        for j in range(n_occ):
            resolved = spaces.resolve(i, j)
            expected = _translate_pair_test_columns(bases[resolved.row], resolved.translation, mesh)
            np.testing.assert_allclose(spaces.coefficients(i, j), expected, atol=2e-14)
            ri, rj = spaces.representative_pairs[resolved.row]
            images = _translate_pair_test_columns(occupied[:, [ri, rj]], resolved.translation, mesh)
            if resolved.transposed:
                images = images[:, ::-1]
            np.testing.assert_allclose(images, occupied[:, [i, j]], atol=2e-14)
            placed[i, j] = expected

    pairs = tuple(placed)
    for index, target in enumerate(pairs):
        source = pairs[(7*index+3) % len(pairs)]
        left, right = placed[target], placed[source]
        np.testing.assert_allclose(spaces.overlap_between(target, source), left.T @ overlap @ right, atol=3e-14)
        np.testing.assert_allclose(spaces.fock_between(target, source), left.T @ fock @ right, atol=3e-14)
    for before, after in zip(snapshots, [overlap, fock, occupied, *bases]):
        np.testing.assert_array_equal(before, after)
    # Source copies own their data: subsequent caller mutations cannot change transport.
    first = spaces.coefficients(0, 0).copy()
    overlap[:] = 0
    fock[:] = 0
    occupied[:] = 0
    bases[0][:] = 0
    np.testing.assert_array_equal(spaces.coefficients(0, 0), first)


@pytest.mark.parametrize('fault', ['metric', 'fock', 'gauge', 'orthogonality', 'virtuality', 'stabilizer', 'complex', 'nonfinite'])
def test_translation_pair_space_refuses_unproved_symmetry_without_mutation(fault):
    from vibeqc.dlpno.translation import TranslationPairSpaces
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    mesh = (4, 1, 1)
    overlap, fock, occupied, bases = _translation_pair_metric_model(mesh)
    if fault == 'metric':
        overlap[0, 0] += .2
    elif fault == 'fock':
        fock[0, 0] += .2
    elif fault == 'gauge':
        occupied[:, 0] *= -1
    elif fault == 'orthogonality':
        bases[0][:, 0] *= 1.01
    elif fault == 'virtuality':
        bases[0] = occupied[:, :1].copy()
    elif fault == 'stabilizer':
        plan = build_translation_pair_plan(2, mesh)
        row = next(k for k, ((i, j), d) in enumerate(zip(plan.band_pairs, plan.displacements))
                   if i == j and tuple(d) == (2, 0, 0))
        # A single arbitrary direction of a two-endpoint domain is not closed.
        bases[row] = bases[row][:, :1].copy()
    elif fault == 'complex':
        bases[0] = bases[0].astype(complex) + .01j
    elif fault == 'nonfinite':
        fock[0, 0] = np.nan
    arrays = [overlap, fock, occupied, *bases]
    before = [a.copy() for a in arrays]
    with pytest.raises(ValueError):
        TranslationPairSpaces(mesh, overlap, fock, occupied, bases)
    for a, b in zip(arrays, before):
        np.testing.assert_array_equal(a, b)


def test_translation_pair_space_preserves_complete_frozen_band():
    from vibeqc.dlpno.translation import TranslationPairSpaces
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    mesh = (3, 1, 1)
    overlap, fock, all_occupied, full_bases = _translation_pair_metric_model(mesh)
    active, frozen = all_occupied[:, 1::2], all_occupied[:, ::2]
    # Reuse the one-band rows from the two-band catalogue with matching displacements.
    full_plan = build_translation_pair_plan(2, mesh)
    lookup = {tuple(d): full_bases[r] for r, ((i, j), d) in enumerate(zip(full_plan.band_pairs, full_plan.displacements)) if i == j == 0}
    plan = build_translation_pair_plan(1, mesh)
    bases = [lookup[tuple(d)] for d in plan.displacements]
    spaces = TranslationPairSpaces(mesh, overlap, fock, active, bases, frozen_occupied=frozen)
    assert spaces.occupied_fock.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(frozen.T @ overlap @ spaces.coefficients(i, j), 0, atol=2e-14)
    bad_bases = list(bases)
    bad_bases[0] = frozen[:, :1]
    with pytest.raises(ValueError):
        TranslationPairSpaces(mesh, overlap, fock, active, bad_bases, frozen_occupied=frozen)
    with pytest.raises(ValueError):
        TranslationPairSpaces(mesh, overlap, fock, active, bases, frozen_occupied=frozen[:, :1])


def test_translation_pair_tensors_enforce_endpoint_sewing_and_own_data():
    from vibeqc.dlpno.translation import TranslationPairSpaces

    mesh = (4, 1, 1)
    overlap, fock, occupied, bases = _translation_pair_metric_model(mesh)
    spaces = TranslationPairSpaces(mesh, overlap, fock, occupied, bases)
    tensors = [np.eye(v.shape[1]) for v in bases]
    validated = spaces.validate_tensors(tensors)
    for source, owned in zip(tensors, validated):
        assert not np.shares_memory(source, owned)
        assert not owned.flags.writeable
        with pytest.raises(ValueError):
            owned.setflags(write=True)
    for i in range(occupied.shape[1]):
        for j in range(occupied.shape[1]):
            v_ij, v_ji = spaces.coefficients(i, j), spaces.coefficients(j, i)
            t_ij = v_ij @ spaces.orient_tensor(i, j, validated) @ v_ij.T
            t_ji = v_ji @ spaces.orient_tensor(j, i, validated) @ v_ji.T
            np.testing.assert_allclose(t_ij, t_ji.T, atol=3e-14)

    bad_diagonal = [t.copy() for t in tensors]
    diagonal_row = next(r for r, (i, j) in enumerate(spaces.representative_pairs) if i == j)
    bad_diagonal[diagonal_row][0, 1] += .1
    with pytest.raises(ValueError, match='symmetric'):
        spaces.validate_tensors(bad_diagonal)
    bad_half_cell = [t.copy() for t in tensors]
    row = next(iter(spaces.stabilizers))
    bad_half_cell[row][0, 1] += .1
    with pytest.raises(ValueError, match='stabilizer'):
        spaces.validate_tensors(bad_half_cell)
    for t in tensors:
        np.testing.assert_array_equal(t, np.eye(t.shape[0]))


@pytest.mark.parametrize('pair', [(-1, 0), (0, 8), (True, 0), (0.0, 1)])
def test_translation_pair_resolver_refuses_invalid_active_indices(pair):
    from vibeqc.dlpno.translation import TranslationPairSpaces

    overlap, fock, occupied, bases = _translation_pair_metric_model()
    spaces = TranslationPairSpaces((4, 1, 1), overlap, fock, occupied, bases)
    with pytest.raises(ValueError, match='indices'):
        spaces.resolve(*pair)


def _translation_builder_source(*, frozen=False, same_source=False, mesh=(3, 1, 1)):
    """Compact factors on a nonorthogonal, finite model Hamiltonian."""
    from types import SimpleNamespace
    from vibeqc.periodic.chi.posthf import _BRealDFAdapter
    from vibeqc.dlpno.source import SourceConventions

    overlap, fock, occupied, _ = _translation_pair_metric_model(mesh)
    nk, nbf = int(np.prod(mesh)), overlap.shape[0]
    rng = np.random.default_rng(2025)
    home = rng.normal(size=(2, nbf, nbf)) * .06
    home = .5*(home+home.swapaxes(1, 2))
    compact = _BRealDFAdapter(home, None, mesh)

    class OnlyTransforms:
        n_aux = compact.n_aux

        def __init__(self):
            self.calls = []
            self.fail = False

        @property
        def three_center(self):
            raise AssertionError('builder must not request expanded AO factors')

        @property
        def home_factors(self):
            raise AssertionError('builder must use the public factor transform contract')

        def mo_transform(self, left, right):
            self.calls.append((left.shape, right.shape))
            assert left.shape[1] == 2
            assert right.shape[1] <= nbf-occupied.shape[1]
            if self.fail:
                raise RuntimeError('intentional factor failure')
            return compact.mo_transform(left, right)

    core = occupied[:, ::2] if frozen else np.empty((nbf, 0))
    active = occupied[:, 1::2] if frozen else occupied.copy()
    # The source's frozen block is first; the explicit active input is cell-major.
    ref_occ = np.column_stack((core, active))
    reference = SimpleNamespace(
        converged=True, reference_kind='rhf', energy=-4.*nk, ecp_total_ncore=0,
        fock=fock, overlap=overlap, mo_coeffs=ref_occ,
    )
    molecule = SimpleNamespace(multiplicity=1, n_electrons=lambda: 4*nk)
    source = CorrelationSource(
        kind='model-real-torus', reference_kind='rhf', molecule=molecule,
        basis=SimpleNamespace(nbasis=nbf, name='finite-model'), reference=reference,
        overlap=overlap, factors=OnlyTransforms(),
        conventions=SourceConventions(kmesh=mesh, exchange_q0='test-finite',
                                      bloch_normalisation='real-bvk-supercell',
                                      same_source=same_source),
        provenance={'model': 'translation-pair-builder'},
    )
    return source, active, core, home


@pytest.mark.parametrize('frozen,same_source', [(False, False), (True, False), (True, True)])
def test_translation_builder_preserves_compact_factors_and_source_partition(frozen, same_source):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    source, active, core, home = _translation_builder_source(frozen=frozen, same_source=same_source)
    arrays = [source.overlap, source.reference.fock, source.reference.mo_coeffs, active, core, home]
    before = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    built = build_translation_pair_pnos(source, (3, 1, 1), active, frozen_occupied=core)
    assert built.source_identity == source.identity
    assert built.source_kind == source.kind
    assert built.same_source is same_source
    assert len(built.identity) == 64
    assert built.max_translation_error < 1e-10
    assert built.spaces.plan.n_occupied == active.shape[1]
    assert tuple(built.pno_ranks) == tuple(built.pao_ranks)
    assert built.pno_correction_per_cell == pytest.approx(0., abs=2e-13)
    assert len(source.factors.calls) >= 3*built.spaces.plan.n_representatives
    for basis in built.spaces.pair_coefficients:
        np.testing.assert_allclose(np.column_stack((active, core)).T @ source.overlap @ basis,
                                   0., atol=2e-12)
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)
        assert not array.flags.writeable
    repeat = build_translation_pair_pnos(source, (3, 1, 1), active, frozen_occupied=core)
    assert repeat.identity == built.identity


@pytest.mark.parametrize('fault', ['uhf', 'unconverged', 'missing_overlap', 'mesh', 'normalization',
                                 'electrons', 'partition', 'gauge', 'fock'])
def test_translation_builder_refuses_invalid_source_before_factor_work(fault):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    source, active, core, home = _translation_builder_source(frozen=True)
    if fault == 'uhf':
        source.reference_kind = 'uhf'
    elif fault == 'unconverged':
        source.reference.converged = False
    elif fault == 'missing_overlap':
        source.overlap = None
    elif fault == 'mesh':
        from dataclasses import replace
        source.conventions = replace(source.conventions, kmesh=(1, 1, 3))
    elif fault == 'normalization':
        from dataclasses import replace
        source.conventions = replace(source.conventions, bloch_normalisation='molecular')
    elif fault == 'electrons':
        source.molecule.n_electrons = lambda: 11
    elif fault == 'partition':
        # The occupied union is unchanged, but freezing a mixed active band is wrong.
        active, core = (active+core)/np.sqrt(2), (core-active)/np.sqrt(2)
    elif fault == 'gauge':
        active[:, 0] *= -1
    elif fault == 'fock':
        source.reference.fock[0, 0] += .1
    before = [a.copy() for a in (active, core, home, source.reference.fock)]
    with pytest.raises((ValueError, TypeError)):
        build_translation_pair_pnos(source, (3, 1, 1), active, frozen_occupied=core)
    assert source.factors.calls == []
    for array, original in zip((active, core, home, source.reference.fock), before):
        np.testing.assert_array_equal(array, original)


def test_translation_builder_factor_failure_preserves_callers():
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    source, active, core, home = _translation_builder_source(frozen=True)
    source.factors.fail = True
    arrays = (source.overlap, source.reference.fock, source.reference.mo_coeffs, active, core, home)
    before = [a.copy() for a in arrays]
    with pytest.raises(RuntimeError, match='intentional factor failure'):
        build_translation_pair_pnos(source, (3, 1, 1), active, frozen_occupied=core)
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)


def test_translation_builder_audits_all_origins_before_pno_truncation():
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    source, active, core, home = _translation_builder_source()
    transform = source.factors.mo_transform
    last = active[:, 4:5]

    def noncovariant(left, right):
        value = transform(left, right)
        # Corrupt one occupied at the final origin; home/generator-only audit misses it.
        if np.max(np.abs(last.T @ source.overlap @ left)) > .9:
            value = value.copy()
            value[:, 0] *= 1.1
        return value

    source.factors.mo_transform = noncovariant
    with pytest.raises(ValueError, match='translat|covarian'):
        build_translation_pair_pnos(source, (3, 1, 1), active, tcut_pno=1e9)


def test_translation_builder_identity_binds_consumed_tensors_and_options():
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    source, active, core, home = _translation_builder_source()
    first = build_translation_pair_pnos(source, (3, 1, 1), active)
    # A mutable parent descriptor alone cannot identify the actual tensors consumed.
    source.reference.fock *= 1.01
    changed = build_translation_pair_pnos(source, (3, 1, 1), active)
    assert changed.source_identity == first.source_identity
    assert changed.identity != first.identity
    other_norm = build_translation_pair_pnos(source, (3, 1, 1), active, pno_norm='legacy')
    assert other_norm.identity != changed.identity
    home *= 1.01
    other_factors = build_translation_pair_pnos(source, (3, 1, 1), active)
    assert other_factors.identity != changed.identity


@pytest.mark.parametrize('normalization', ['real-bvk-supercell', 'unnormalised-ao-bloch-sums-uniform-weights'])
def test_translation_source_snapshots_ecp_counts_and_preserves_conventions(normalization):
    from dataclasses import replace
    from vibeqc.dlpno.translation_source import admit_translation_source

    source, active, core, home = _translation_builder_source(frozen=True, same_source=True)
    source.conventions = replace(source.conventions, bloch_normalisation=normalization)
    source.molecule.n_electrons = lambda: 18
    source.reference.ecp_total_ncore = 6
    inputs = admit_translation_source(source, (3, 1, 1), active, frozen_occupied=core)
    assert inputs.n_electrons == 12
    assert inputs.n_frozen == 3
    assert inputs.conventions['bloch_normalisation'] == normalization
    assert inputs.conventions['exchange_q0'] == 'test-finite'
    assert inputs.same_source is True
    assert inputs.factors is source.factors
    original_s = inputs.overlap.copy()
    original_active = inputs.occupied.copy()
    source.overlap[:] = 0
    active[:] = 0
    source.provenance['model'] = 'changed caller descriptor'
    np.testing.assert_array_equal(inputs.overlap, original_s)
    np.testing.assert_array_equal(inputs.occupied, original_active)
    assert inputs.provenance['source_provenance']['model'] == 'translation-pair-builder'
    assert not inputs.overlap.flags.writeable
    with pytest.raises(ValueError):
        inputs.overlap.setflags(write=True)
    with pytest.raises(TypeError):
        inputs.conventions['exchange_q0'] = 'changed'


def test_translation_builder_refuses_minimum_rank_splitting_half_cell_space():
    from types import SimpleNamespace
    from vibeqc.dlpno.source import LpqFactors, SourceConventions
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    # Degenerate virtual Fock and zero pair density leave localized AO directions.
    # The historical keep-one fallback must not silently violate half-cell sewing.
    occupied = np.eye(6)[:, [0, 3]]
    source = CorrelationSource(
        kind='zero-pair-model', reference_kind='rhf',
        molecule=SimpleNamespace(multiplicity=1, n_electrons=lambda: 4),
        basis=SimpleNamespace(nbasis=6), overlap=np.eye(6),
        reference=SimpleNamespace(converged=True, energy=-2., mo_coeffs=occupied,
                                  fock=np.diag([-1., .5, .5, -1., .5, .5])),
        factors=LpqFactors(np.zeros((1, 6, 6))),
        conventions=SourceConventions(kmesh=(2, 1, 1), bloch_normalisation='real-bvk-supercell'),
    )
    with pytest.raises(ValueError, match='stabilizer'):
        build_translation_pair_pnos(source, (2, 1, 1), occupied, tcut_pno=1e-7)
    full = build_translation_pair_pnos(source, (2, 1, 1), occupied, tcut_pno=0.)
    assert full.pno_ranks == (4, 4)


@pytest.mark.parametrize('mesh', [(3, 1, 1), (4, 1, 1), (2, 2, 1)])
@pytest.mark.parametrize('frozen', [False, True])
def test_translation_selection_recovers_phase_free_gauge_from_rotated_projectors(mesh, frozen):
    from vibeqc.dlpno.translation_selection import select_translation_occupied

    source, active, core, home = _translation_builder_source(mesh=mesh, frozen=frozen,
                                                           same_source=True)
    rng = np.random.default_rng(1093)

    def rotate(columns):
        return columns @ np.linalg.qr(rng.normal(size=(columns.shape[1], columns.shape[1])))[0]

    source.reference.mo_coeffs = np.column_stack((rotate(core), rotate(active)))
    arrays = (source.overlap, source.reference.fock, source.reference.mo_coeffs, home)
    before = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    selected = select_translation_occupied(source, mesh, n_frozen=core.shape[1])
    assert selected.source_identity == source.identity
    assert selected.source_kind == source.kind
    assert selected.same_source is True
    assert selected.n_frozen == core.shape[1]
    assert source.factors.calls == []
    n_cells = int(np.prod(mesh))
    for actual, expected in ((selected.occupied, active), (selected.frozen_occupied, core)):
        np.testing.assert_allclose(actual @ actual.T @ source.overlap,
                                   expected @ expected.T @ source.overlap, atol=3e-13)
        np.testing.assert_allclose(actual.T @ source.overlap @ actual,
                                   np.eye(actual.shape[1]), atol=3e-13)
        n_bands = actual.shape[1] // n_cells
        grid = np.arange(n_cells).reshape(mesh)
        for axis, extent in enumerate(mesh):
            if extent > 1:
                shift = tuple(int(a == axis) for a in range(3))
                images = np.roll(grid, -1, axis=axis).ravel()
                indices = (images[:, None]*n_bands+np.arange(n_bands)[None, :]).ravel()
                np.testing.assert_allclose(_translate_pair_test_columns(actual, shift, mesh),
                                           actual[:, indices], atol=3e-13)
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    np.testing.assert_allclose(selected.frozen_occupied.T @ source.overlap @ selected.occupied,
                               0., atol=3e-13)
    with pytest.raises(TypeError):
        selected.provenance['source_inputs']['source_provenance']['model'] = 'changed'
    repeated = select_translation_occupied(source, mesh, n_frozen=core.shape[1])
    assert repeated.identity == selected.identity
    np.testing.assert_array_equal(repeated.occupied, selected.occupied)
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)

    # The parent descriptor is unchanged, while the selection identity records
    # the actual supplied orbital tensors. Gauge selection depends on projectors.
    source.reference.mo_coeffs = np.column_stack((rotate(core), rotate(active)))
    rotated = select_translation_occupied(source, mesh, n_frozen=core.shape[1])
    assert rotated.source_identity == selected.source_identity
    assert rotated.identity != selected.identity
    assert rotated.active_seed_aos == selected.active_seed_aos
    assert rotated.frozen_seed_aos == selected.frozen_seed_aos
    np.testing.assert_allclose(rotated.occupied, selected.occupied, atol=3e-13)
    np.testing.assert_allclose(rotated.frozen_occupied, selected.frozen_occupied, atol=3e-13)
    assert source.factors.calls == []


def _translation_selection_projector_source(occupied):
    from types import SimpleNamespace
    from vibeqc.dlpno.source import SourceConventions

    nbf, nocc = occupied.shape

    def forbidden_transform(*args):
        raise AssertionError('occupied frame selection must not transform factors')

    return CorrelationSource(
        kind='projector-frame-model', reference_kind='rhf',
        molecule=SimpleNamespace(multiplicity=1, n_electrons=lambda: 2*nocc),
        basis=SimpleNamespace(nbasis=nbf), overlap=np.eye(nbf),
        reference=SimpleNamespace(converged=True, energy=-2., mo_coeffs=occupied,
                                  fock=.5*np.eye(nbf)-1.5*occupied@occupied.T),
        factors=SimpleNamespace(mo_transform=forbidden_transform),
        conventions=SourceConventions(kmesh=(2, 1, 1), bloch_normalisation='real-bvk-supercell'),
    )


def test_translation_selection_resolves_equal_seed_scores_by_ao_index():
    from vibeqc.dlpno.translation_selection import select_translation_occupied

    occupied = np.zeros((6, 2))
    occupied[[0, 1], 0] = 1/np.sqrt(2)
    occupied[[3, 4], 1] = 1/np.sqrt(2)
    source = _translation_selection_projector_source(occupied)
    first = select_translation_occupied(source, (2, 1, 1))
    assert first.active_seed_aos == (0,)
    source.reference.mo_coeffs = occupied @ np.array([[.6, -.8], [.8, .6]])
    repeated = select_translation_occupied(source, (2, 1, 1))
    assert repeated.active_seed_aos == (0,)
    np.testing.assert_allclose(first.occupied, repeated.occupied, atol=4e-15)


def test_translation_selection_refuses_projector_without_common_ao_seed_family():
    from vibeqc.dlpno.translation_selection import select_translation_occupied

    # One band at each momentum, but AO 0 supports only the even sector and
    # AO 1 only the odd sector. A mixed seed could work; neither AO orbit does.
    occupied = np.array([[1., 0.], [0., 1.], [1., 0.], [0., -1.]])/np.sqrt(2)
    source = _translation_selection_projector_source(occupied)
    original = source.reference.mo_coeffs.copy()
    with pytest.raises(ValueError, match='common-seed'):
        select_translation_occupied(source, (2, 1, 1))
    np.testing.assert_array_equal(source.reference.mo_coeffs, original)


@pytest.mark.parametrize('fault', ['frozen_projector', 'fock', 'uhf', 'mesh',
                                 'odd_electrons', 'n_frozen', 'all_frozen',
                                 'bool_frozen', 'seed_lindep', 'tolerance'])
def test_translation_selection_refuses_unsupported_inputs_without_factor_work(fault):
    from dataclasses import replace
    from vibeqc.dlpno.translation_selection import select_translation_occupied

    source, active, core, home = _translation_builder_source(frozen=True)
    options = {'n_frozen': core.shape[1]}
    if fault == 'frozen_projector':
        # Keep the full occupied projector but spoil each frozen/active band's
        # covariance by mixing only one origin across the frozen partition.
        C = source.reference.mo_coeffs
        C[:, [0, 3]] = C[:, [0, 3]] @ np.array([[.6, -.8], [.8, .6]])
    elif fault == 'fock':
        source.reference.fock[0, 0] += .1
    elif fault == 'uhf':
        source.reference_kind = 'uhf'
    elif fault == 'mesh':
        source.conventions = replace(source.conventions, kmesh=(1, 1, 3))
    elif fault == 'odd_electrons':
        source.molecule.n_electrons = lambda: 11
    elif fault == 'n_frozen':
        options['n_frozen'] = 1
    elif fault == 'all_frozen':
        options['n_frozen'] = 6
    elif fault == 'bool_frozen':
        options['n_frozen'] = True
    elif fault == 'seed_lindep':
        options['seed_lindep'] = 0.
    elif fault == 'tolerance':
        options['tolerance'] = np.nan
    arrays = (source.overlap, source.reference.fock, source.reference.mo_coeffs, home)
    before = [array.copy() for array in arrays]
    with pytest.raises((ValueError, TypeError)):
        select_translation_occupied(source, (3, 1, 1), **options)
    assert source.factors.calls == []
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize('mesh', [(3, 1, 1), (4, 1, 1), (2, 2, 1)])
def test_translation_domains_match_unreduced_mulliken_unions(mesh):
    from vibeqc.dlpno.translation_domains import select_translation_pair_domains
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    S, _, C, _ = _translation_pair_metric_model(mesh)
    groups = (np.array([0, 2]), np.array([1, 3]))
    selected = select_translation_pair_domains(mesh, S, C, groups, tcut_mkn=.01)
    nk, nbf_cell = int(np.prod(mesh)), 4
    # Independently threshold every placed orbital, without representative routing.
    products = C * (S @ C)
    masks = np.array([[abs(np.sum(products[cell*nbf_cell+group, i])) > .01
                      for cell in range(nk) for group in groups]
                     for i in range(C.shape[1])])
    plan = build_translation_pair_plan(2, mesh)
    for row, ((i, band_j), shift) in enumerate(zip(plan.band_pairs, plan.displacements)):
        j = 2*int(np.ravel_multi_index(tuple(shift), mesh))+band_j
        union = masks[i] | masks[j]
        expected = sorted(ao for atom in np.flatnonzero(union)
                          for ao in atom//2*nbf_cell+groups[atom%2])
        np.testing.assert_array_equal(selected.pair_ao_domains[row], expected)
        # Equal-band half-cell unions must be closed under their stabilizer.
        if i == band_j and np.any(shift) and np.all((2*shift) % np.asarray(mesh) == 0):
            indicator = np.zeros(S.shape[0], dtype=bool)
            indicator[selected.pair_ao_domains[row]] = True
            shifted = np.roll(indicator.reshape(*mesh, nbf_cell), tuple(shift), axis=(0, 1, 2))
            np.testing.assert_array_equal(shifted.ravel(), indicator)
    assert all(not array.flags.writeable for array in selected.pair_ao_domains)
    with pytest.raises(TypeError):
        selected.provenance['tcut_mkn'] = 0
    all_aos = select_translation_pair_domains(mesh, S, C, groups, tcut_mkn=0)
    assert all(len(domain) == len(S) for domain in all_aos.pair_ao_domains)


@pytest.mark.parametrize('fault', ['overlap', 'phase', 'groups', 'duplicate', 'negative', 'empty'])
def test_translation_domains_reject_bad_gauge_and_domains_without_mutation(fault):
    from vibeqc.dlpno.translation_domains import select_translation_pair_domains

    mesh = (3, 1, 1)
    S, _, C, _ = _translation_pair_metric_model(mesh)
    groups = [np.array([0, 2]), np.array([1, 3])]
    cutoff = .01
    if fault == 'overlap':
        S[0, 0] += .1
    elif fault == 'phase':
        C[:, 0] *= -1
    elif fault == 'groups':
        groups[1] = np.array([1])
    elif fault == 'duplicate':
        groups[1] = np.array([1, 2, 3])
    elif fault == 'negative':
        cutoff = -1
    else:
        cutoff = 1e10
    before = [x.copy() for x in (S, C, *groups)]
    with pytest.raises(ValueError):
        select_translation_pair_domains(mesh, S, C, groups, tcut_mkn=cutoff)
    for actual, expected in zip((S, C, *groups), before):
        np.testing.assert_array_equal(actual, expected)


def test_translation_domain_cutoff_is_strict_and_atom_map_counts_cartesian_shells():
    from types import SimpleNamespace
    from vibeqc.dlpno.translation_domains import derive_translation_atom_groups, select_translation_pair_domains

    C = np.zeros((4, 1))
    C[:2, 0] = np.sqrt([.75, .25])
    exact = float(C[1, 0]**2)
    chosen = select_translation_pair_domains((1, 1, 1), np.eye(4), C,
                                             ([0, 2], [1, 3]), tcut_mkn=exact)
    np.testing.assert_array_equal(chosen.pair_ao_domains[0], [0, 2])
    shells = [SimpleNamespace(atom_index=2*cell+atom, l=angular, pure=pure)
              for cell in range(2) for atom, angular, pure in [(0, 2, False), (1, 2, True)]]
    source = SimpleNamespace(basis=SimpleNamespace(nbasis=22, shells=lambda: shells),
                             molecule=SimpleNamespace(atoms=[None]*4))
    groups = derive_translation_atom_groups(source, (2, 1, 1))
    np.testing.assert_array_equal(groups[0], np.arange(6))
    np.testing.assert_array_equal(groups[1], np.arange(6, 11))
    shells[-1].atom_index = 0
    with pytest.raises(ValueError, match='cell-major'):
        derive_translation_atom_groups(source, (2, 1, 1))


def _automatic_translation_source(mesh, frozen):
    """Compact finite factors, arbitrary input gauge and non-diagonal occupied Fock."""
    from types import SimpleNamespace

    source, active, core, home = _translation_builder_source(mesh=mesh, frozen=frozen)
    nk = int(np.prod(mesh))
    # Cyclic nearest-cell coupling is invariant under the column cell action.
    bands = active.shape[1]//nk
    hop = np.roll(np.eye(nk), 1, axis=0)
    coupling = .025*np.kron(hop+hop.T, np.eye(bands))
    source.reference.fock += source.overlap @ active @ coupling @ active.T @ source.overlap
    rng = np.random.default_rng(127)
    active_rotation = np.linalg.qr(rng.normal(size=(active.shape[1], active.shape[1])))[0]
    core_rotation = np.linalg.qr(rng.normal(size=(core.shape[1], core.shape[1])))[0]
    source.reference.mo_coeffs = np.column_stack((core @ core_rotation, active @ active_rotation))
    source.molecule.atoms = [None]*(2*nk)
    shells = [SimpleNamespace(atom_index=2*cell+atom, l=0, pure=True)
              for cell in range(nk) for atom in [0, 1, 0, 1]]
    source.basis.shells = lambda: shells
    return source, active, core, home


@pytest.mark.parametrize('mesh,frozen', [((3, 1, 1), False), ((4, 1, 1), True), ((2, 2, 1), False)])
def test_translation_automatic_driver_matches_unreduced_canonical_limit(mesh, frozen, record_property):
    from itertools import product
    from scipy.linalg import eigh
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2

    source, active, core, home = _automatic_translation_source(mesh, frozen)
    arrays = [source.overlap, source.reference.fock, source.reference.mo_coeffs, home]
    before = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    result = run_translation_dlpno_mp2(source, mesh, n_frozen=core.shape[1],
                                     conv_tol_energy=1e-13, conv_tol_residual=1e-12)
    solution = result.solution
    assert solution.converged
    assert solution.n_iter > 2
    # Dense independent finite-system MP2: no representative plan or native
    # transport. Expand auxiliary translates ONLY inside this bounded oracle.
    nbf, nk = source.basis.nbasis, int(np.prod(mesh))
    index = np.arange(nbf).reshape(*mesh, nbf//nk)
    expanded = np.concatenate([home[:, np.roll(index, shift, axis=(0, 1, 2)).ravel()][:, :,
                               np.roll(index, shift, axis=(0, 1, 2)).ravel()]
                               for shift in product(*(range(n) for n in mesh))])
    eps, canonical = eigh(source.reference.fock, source.overlap)
    nocc = core.shape[1]+active.shape[1]
    eo, ev = eps[core.shape[1]:nocc], eps[nocc:]
    ov = np.einsum('Pmn,mi,na->Pia', expanded, canonical[:, core.shape[1]:nocc], canonical[:, nocc:])
    K = np.einsum('Pia,Pjb->ijab', ov, ov)
    denom = eo[:, None, None, None]+eo[None, :, None, None]-ev[None, None, :, None]-ev[None, None, None, :]
    T = K/denom
    expected = np.sum(K*(2*T-T.swapaxes(2, 3)))/nk
    record_property('e_corr_per_cell', solution.e_corr_per_cell)
    record_property('unreduced_e_corr_per_cell', expected)
    record_property('absolute_energy_error', abs(solution.e_corr_per_cell-expected))
    record_property('maximum_raw_residual', solution.max_residual)
    record_property('evaluated_states', solution.n_iter)
    assert solution.e_corr_per_cell == pytest.approx(expected, abs=2e-11)
    assert solution.e_hf_per_cell == pytest.approx(source.reference.energy/nk, abs=1e-14)
    assert solution.max_residual < 1e-12
    assert solution.e_pno_correction_per_cell == pytest.approx(0., abs=2e-12)
    assert not result.pair_build.same_source
    for array, initial in zip(arrays, before):
        np.testing.assert_array_equal(array, initial)
        assert not array.flags.writeable
    assert len(source.factors.calls) > 0
    assert result.identity == run_translation_dlpno_mp2(
        source, mesh, n_frozen=core.shape[1], conv_tol_energy=1e-13, conv_tol_residual=1e-12,
    ).identity


def test_translation_automatic_driver_failure_preserves_input_and_source_identity():
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2

    source, _, core, home = _automatic_translation_source((3, 1, 1), True)
    arrays = [source.overlap, source.reference.fock, source.reference.mo_coeffs, home]
    before = [array.copy() for array in arrays]
    identity = source.identity
    source.factors.fail = True
    with pytest.raises(RuntimeError, match='intentional factor failure'):
        run_translation_dlpno_mp2(source, (3, 1, 1), n_frozen=core.shape[1])
    for array, initial in zip(arrays, before):
        np.testing.assert_array_equal(array, initial)
    assert source.identity == identity


def test_translation_automatic_nonzero_domains_match_unreduced_galerkin(record_property):
    from types import SimpleNamespace
    from scipy.linalg import eigh
    from tests.test_dlpno_mp2 import _translation_unreduced_galerkin_solution
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2

    mesh = (3, 1, 1)
    source, _, core, home = _automatic_translation_source(mesh, True)
    result = run_translation_dlpno_mp2(source, mesh, n_frozen=core.shape[1],
                                     tcut_mkn=.01, tcut_pno=1e-5,
                                     conv_tol_energy=1e-13, conv_tol_residual=1e-12)
    assert result.solution.converged
    nbf, nk = source.basis.nbasis, 3
    assert max(len(d) for d in result.domain_selection.pair_ao_domains) < nbf
    S, F, C = source.overlap, source.reference.fock, result.occupied_selection.occupied
    _, canonical = eigh(F, S)
    V = canonical[:, core.shape[1]+C.shape[1]:]
    cells = np.arange(nbf).reshape(nk, nbf//nk)
    permutations = [np.eye(nbf)[np.roll(cells, shift, axis=0).ravel()] for shift in range(nk)]
    factors = np.concatenate([np.einsum('am,Pmn,bn->Pab', P, home, P) for P in permutations])
    ov = np.einsum('Pmn,mi,na->Pia', factors, C, V)
    model = SimpleNamespace(n_cells=nk, virtual=V, overlap=S, fvv=V.T@F@V, foo=C.T@F@C,
                            exchange=np.einsum('Pia,Pjb->ijab', ov, ov),
                            occ_translations=[C.T@S@P@C for P in permutations],
                            vir_translations=[V.T@S@P@V for P in permutations])
    _, expected = _translation_unreduced_galerkin_solution(model, result.pair_build, np.arange(C.shape[1]))
    record_property('e_pair_per_cell', result.solution.e_pair_per_cell)
    record_property('unreduced_pair_energy_per_cell', expected)
    record_property('absolute_energy_error', abs(result.solution.e_pair_per_cell-expected))
    record_property('maximum_raw_residual', result.solution.max_residual)
    assert result.solution.e_pair_per_cell == pytest.approx(expected, abs=2e-11)
    assert result.solution.e_corr_per_cell == pytest.approx(
        expected+result.pair_build.pno_correction_per_cell, abs=2e-11)
