"""Saved BIPOLE state crossing into the draft operator API; no native calls."""
import argparse
from dataclasses import replace
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys
import types

import numpy as np
import scipy
from scipy.linalg import eigh

parser = argparse.ArgumentParser()
parser.add_argument('--operator-source', type=Path, required=True)
parser.add_argument('--state-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = {name: sha(args.operator_source / name) for name in ('operators.py', 'eigensolver.py')}
# Load the unmodified two pure-Python modules without package initialization,
# which would otherwise import an unrelated/stale native extension.
package = types.ModuleType('_operator_crossing')
package.__path__ = [str(args.operator_source)]
sys.modules[package.__name__] = package
operators = importlib.import_module('_operator_crossing.operators')
solvers = importlib.import_module('_operator_crossing.eigensolver')
def refuse_native():
    raise AssertionError('This offline audit must never load a native core')
solvers._get_core = refuse_native

inp = json.loads((args.state_dir/'execution-input.json').read_text())
manifest = json.loads((args.state_dir/'manifest.json').read_text())
assert sha(args.state_dir/'state.npz') == manifest['input_sha256']['state.npz']
assert sha(args.state_dir/'execution-input.json') == manifest['input_sha256']['execution-input.json']
with np.load(args.state_dir/'state.npz', allow_pickle=False) as z:
    fock, overlap = z['fock'].copy(), z['overlap'].copy()
    density_real, indices = z['density_real'].copy(), z['density_cell_indices'].copy()
lattice, kpoints, weights = (np.array(inp[n]) for n in ('lattice', 'kpoints', 'weights'))
assert fock.shape == overlap.shape == (8, 44, 44)
assert kpoints.shape == (8, 3) and lattice.shape == (3, 3)
np.testing.assert_array_equal(weights, np.full(8, 1/8))
cart = indices @ lattice.T
norms = np.sum(cart*cart, axis=1)
representatives = []
for residue in itertools.product(range(2), repeat=3):
    choices = np.flatnonzero(np.all(indices % 2 == residue, axis=1))
    assert len(choices)
    representatives.append(choices[np.argmin(norms[choices])])
density = np.einsum('kg,gij->kij', np.exp(1j*kpoints@cart[representatives].T), density_real[representatives])
np.testing.assert_allclose(density, density.conj().transpose(0, 2, 1), rtol=0, atol=1e-12)
np.testing.assert_allclose(np.einsum('kg,kij,k->gij', np.exp(-1j*kpoints@cart.T), density, weights),
                           density_real, rtol=0, atol=1e-12)
rows = []
refusals = 0
for ik, (f, s, d) in enumerate(zip(fock, overlap, density)):
    metadata = operators.OperatorMetadata(kind='ks', basis_id='si-physical-ao', generation=12,
        spin='restricted', kpoint=tuple(kpoints[ik]), symmetry_block=None,
        occupations=tuple([2.]*14+[0.]*30))
    fin, sin = f.copy(), s.copy()
    op = operators.FockOperator(matrix=fin, overlap=sin, metadata=metadata)
    fin[:] = 123.; sin[:] = 456.
    np.testing.assert_array_equal(op.matrix, f)
    np.testing.assert_array_equal(op.overlap, s)
    assert not op.matrix.flags.writeable and not op.overlap.flags.writeable
    result = operators.solve_operator(op, solvers.SolverOptions(tol=1e-9), solver='dense')
    expected_e, expected_c = eigh(f, s, driver='gvd')
    expected_d = 2*expected_c[:, :14]@expected_c[:, :14].conj().T
    orbital_d = 2*result.eigenvectors[:, :14]@result.eigenvectors[:, :14].conj().T
    np.testing.assert_allclose(result.eigenvalues, expected_e, rtol=0, atol=1e-9)
    np.testing.assert_allclose(orbital_d, expected_d, rtol=0, atol=1e-10)
    # This is a complex AO coordinate change, NOT source-symmetry admission.
    phases = np.exp(1j*np.linspace(.13, 2.17, len(f))*(ik+1))
    transport = lambda a: phases[:, None]*a*phases.conj()[None, :]
    moved = operators.FockOperator(matrix=transport(f), overlap=transport(s),
        metadata=replace(metadata, basis_id='si-rephased-ao'))
    transformed = operators.solve_operator(moved, solvers.SolverOptions(tol=1e-9), solver='dense')
    moved_d = 2*transformed.eigenvectors[:, :14]@transformed.eigenvectors[:, :14].conj().T
    np.testing.assert_allclose(transformed.eigenvalues, result.eigenvalues, rtol=0, atol=1e-9)
    np.testing.assert_allclose(moved_d, transport(orbital_d), rtol=0, atol=1e-10)
    trace = lambda a,b: float(np.trace(a@b).real)
    np.testing.assert_allclose(trace(transport(d), transport(s)), trace(d,s), rtol=0, atol=1e-11)
    np.testing.assert_allclose(trace(transport(d), transport(f)), trace(d,f), rtol=0, atol=1e-9)
    # Frame/spin/k/symmetry labels must reject accidental reuse before native work.
    for changed in (dict(basis_id='different-ao'), dict(spin='alpha'),
                    dict(kpoint=(.123,.234,.345)), dict(symmetry_block='unqualified-wedge')):
        bad = operators.FockOperator(matrix=f, overlap=s, metadata=replace(metadata, **changed))
        try:
            operators.solve_operator(bad, solvers.SolverOptions(tol=1e-9),
                                     solver='hermitian_davidson', previous=result)
        except ValueError as exc:
            assert 'different operator frame' in str(exc)
            refusals += 1
        else:
            raise AssertionError('Incompatible recycled frame was accepted')
    probe = np.column_stack((expected_c[:,0], expected_c[:,-1]))
    np.testing.assert_allclose(op.apply(probe), f@probe, rtol=0, atol=1e-12)
    # Evaluated density is retained; the new projector is diagnostic only.
    rows.append(dict(k_index=ik, electrons=trace(d,s),
        eigenvalue_error=float(np.max(abs(result.eigenvalues-expected_e))),
        projector_error=float(np.max(abs(orbital_d-expected_d))),
        evaluated_to_orbital_density=float(np.max(abs(d-orbital_d))),
        max_ao_residual=float(np.max(result.residuals)),
        max_projected_residual=float(np.max(result.projected_residuals)),
        metric_error=result.orthogonality_error,
        gauge_projector_error=float(np.max(abs(moved_d-transport(orbital_d)))),
        gauge_eigenvalue_error=float(np.max(abs(transformed.eigenvalues-result.eigenvalues))),
        max_complex_gauged_fock=float(np.max(abs(transport(f).imag)))))
assert sha(args.state_dir/'state.npz') == manifest['input_sha256']['state.npz']
assert all(sha(args.operator_source/n)==h for n,h in sources.items())
assert not any('_vibeqc_core' in key for key in sys.modules)
report = dict(status='passed', operator_source_sha256=sources,
    state_sha256=sha(args.state_dir/'state.npz'), input_sha256=sha(args.state_dir/'execution-input.json'),
    script_sha256=sha(Path(__file__)), numpy=np.__version__, scipy=scipy.__version__,
    dense_solves=16, frame_refusals=refusals, rows=rows,
    weighted_electrons=float(weights@np.array([r['electrons'] for r in rows])),
    scope='Author offline compatibility witness for saved Si PBE full-222 BIPOLE state and unmodified draft operator Python dense API. No new native build, Fock evaluation, SCF, symmetry reduction, MANTA numerical equivalence, CRYSTAL accuracy acceptance or performance claim.')
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
