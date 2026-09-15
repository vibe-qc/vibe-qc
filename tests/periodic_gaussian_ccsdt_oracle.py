"""Opt-in, out-of-process canonical oracle for the same finite Gaussian HF.

This is a test runner, never imported by the vibe-qc runtime. Native export
and PySCF consumption MUST run as separate processes; no PySCF periodic DF,
standard source cuts, fetched basis, or synthetic large-gap SCF is involved.
Only He K=1/2/3 and He2 K=1/2, at most four occupied and four virtual
orbitals in the complete finite torus, are admitted. The He2 full-space
rotation fixture explicitly uses the existing PM gradient control2e-6;
SCF/CCSD/(T) tolerances are unchanged. Its stricter PM1e-11 failure is a
separate regression, not silently converted to convergence here.

Primary definitions: Sun 2017 doi:10.1063/1.4998644 Eqs.3,13,16,21;
Nejad 2025 doi:10.1063/5.0290816 Eqs.7-16; Raghavachari 1989
doi:10.1016/S0009-2614(89)87395-6 Sec.5. PySCF's installed real RCCSD
_ChemistsERIs._common_init_ reconstructs F from the supplied effective h and
active-density J/K; that reconstruction is independently audited below.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _guard(arguments):
    import psutil

    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    command = [sys.executable, "-m", "tests.periodic_gaussian_ccsdt_oracle", *arguments]
    process = subprocess.Popen(command, env=env, start_new_session=True)
    owner = psutil.Process(process.pid)
    cap, peak, deadline, failure = 256*1024**2, 0, time.monotonic()+120, None
    while process.poll() is None:
        try:
            members = [owner, *owner.children(recursive=True)]
        except psutil.NoSuchProcess:
            members = []
        rss = 0
        for member in members:
            try:
                rss += member.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        peak = max(peak, rss)
        if rss > cap or time.monotonic() > deadline:
            failure = "RSS" if rss > cap else "120-second deadline"
            for member in reversed(members):
                try:
                    member.terminate()  # psutil verifies the PID's creation identity.
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(members, timeout=3)
            for member in alive:
                try:
                    member.kill()
                except psutil.NoSuchProcess:
                    pass
            break
        time.sleep(0.05)
    status = process.wait()
    print(json.dumps(dict(stage="watchdog", peak_bytes=peak, cap_bytes=cap, failure=failure)), flush=True)
    return 125 if failure else status


def _eightfold(g):
    import numpy as np
    # Average the eight physically identical representations, after an
    # independently measured defect gate. Then select exact eightfold slots.
    permutations = (g, g.swapaxes(0, 1), g.swapaxes(2, 3), g.swapaxes(0, 1).swapaxes(2, 3))
    mean = sum(permutations+tuple(x.transpose(2, 3, 0, 1) for x in permutations))/8
    n = g.shape[0]
    out = np.empty_like(mean)
    for p in range(n):
        for q in range(p+1):
            for r in range(n):
                for s in range(r+1):
                    if p*(p+1)//2+q < r*(r+1)//2+s:
                        continue
                    value = mean[p, q, r, s]
                    for i, j in ((p, q), (q, p)):
                        for k, l in ((r, s), (s, r)):
                            out[i, j, k, l] = out[k, l, i, j] = value
    defect = float(np.max(abs(g-out)))
    if defect > 1e-11:
        raise ValueError(f"actual Gaussian eightfold projection is substantive: {defect}")
    return np.ascontiguousarray(out), defect


def _native(nk, output, system):
    import numpy as np
    from itertools import product
    from tests.test_periodic_gaussian_local_orbital_factors import _bundle, _dense_panel
    from tests.test_periodic_gaussian_fock import _q_data
    from tests.test_periodic_correlation_pao_domain import _make as domain
    from tests.test_periodic_correlation_real_pao_space import _make as real_space
    from tests.test_periodic_correlation_real_local_basis import _make as real_basis
    from tests.test_periodic_correlation_density_factors import _virtual, _occupied_columns, _virtual_columns
    from tests.test_periodic_gaussian_real_local_provider import _make as real_provider
    from tests.test_periodic_correlation_real_local_ccsd_t import _run, _options, _plan
    from tests.test_bounded_restricted_ccsd_solver import _run as dense_ccsd, _options as ccsd_options
    from tests.test_bounded_restricted_triples_solver import _run as dense_triples, _options as triples_options

    if output.exists():
        raise FileExistsError("oracle export must not overwrite an existing artifact")
    connected = None
    if system == "he2":
        if nk not in (1, 2):
            raise ValueError("two-helium oracle admits only Gamma and K2")
        from tests.test_periodic_gaussian_selected_local_ccsd_t import (
            _he2_bundle, _he2_controls, _run as actual_run,
            _prepare_leaves, _plan as actual_plan,
            _original_hamiltonian_check,
        )
        b = _he2_bundle((nk, 1, 1))
        controls = _he2_controls(b)
        # Existing control is explicit only for this full-space fixture.
        pm_tolerance = controls[0].localization.optimizer.riemannian_gradient_tolerance
        if pm_tolerance != 2e-6:
            raise ValueError("He2 oracle PM control changed from its explicit fixture value")
        if actual_plan(b, controls).work_units_upper_bound > 10**17:
            raise ValueError("two-helium connected reference exceeds tiny diagnostic work cap")
        def outer_event(e):
            print(json.dumps(dict(stage="native-connected", system=system, nk=nk,
                solver=str(e.stage), callback=e.callback_count)), flush=True)
        connected = actual_run(b, controls=controls, callback=outer_event)
        if not connected.converged or not connected.periodic_energy_per_cell:
            raise ValueError("actual two-helium full-torus native reference did not converge")
        _prepare_leaves(b, localization_options=controls[0].localization)
        o, v, n = 2*nk, 2*nk, 4*nk
        maximum_iterations = controls[0].correlation.ccsd.maximum_iterations
    else:
        b = _bundle((nk, 1, 1), rows=[[0, cell] for cell in range(nk)])
        b.domain = domain(b.reference, np.array(list(product(range(nk), range(2))), dtype=np.uint64))
        b.real_space = real_space(b.reference, b.domain)
        b.space = b.real_space.space
        b.selected = _virtual(0, nk, 0)
        b.basis = real_basis(b, b.rows, b.selected)
        o, v, n = nk, nk, 2*nk
        pm_tolerance, maximum_iterations = None, 40
    assert b.space.retained_dimension == v
    wrapper = real_provider(b)
    b.provider = wrapper.provider
    options = _options(ccsd=ccsd_options(maximum_iterations=maximum_iterations),
        triples=triples_options(maximum_iterations=maximum_iterations))
    if connected is None:
        plan = _plan(b, options)
        if plan.work_units_upper_bound > 20000000000 or plan.integral_calls_upper_bound > 100000000:
            raise ValueError("same-Hamiltonian native reference exceeds the tiny diagnostic work cap")
    def event(e):
        print(json.dumps(dict(stage="native-correlation", nk=nk, solver=str(e.stage),
            ccsd_iteration=e.ccsd.iteration, triples_iteration=e.triples.iteration)), flush=True)
    result = _run(b, options=options, progress=event) if connected is None else connected.correlation
    if not result.converged:
        raise ValueError("actual finite He native CCSD/(T) did not converge within admitted iterations")
    if connected is not None and connected.provider_identity_sha256 != wrapper.identity_sha256:
        raise ValueError("separately exposed native factors differ from connected actual-HF provider")
    actual_panels = []
    for q in range(nk):
        source, _, w = _q_data(b, q)
        actual_panels.append(_dense_panel(b, source, w))
    actual_panels = np.asarray(actual_panels).reshape(-1, n, n)
    g_complex = np.einsum("Qqp,Qrs->pqrs", actual_panels.conj(), actual_panels)
    imaginary = float(np.max(abs(g_complex.imag)))
    if imaginary > 1e-11:
        raise ValueError("actual Gaussian finite-torus ERI has substantive imaginary lanes")
    g, symmetry_projection = _eightfold(g_complex.real)
    native_g = np.array([b.provider.integral(*index).value for index in product(range(n), repeat=4)]).reshape((n,)*4)
    eri_difference = float(np.max(abs(native_g-g)))
    if eri_difference > 1e-11:
        raise ValueError("native real rows disagree with independent same-source AO contraction")
    # Independently reconstruct physical F from all actual k-point matrices,
    # not from orbital-energy labels or the native packed real-F certificate.
    f_complex = np.zeros((n, n), complex)
    for k in range(nk):
        columns = np.column_stack((_occupied_columns(b, k, b.rows), _virtual_columns(b, k, b.selected)))
        f_complex += columns.conj().T @ b.reference.state.fock(k) @ columns/nk
    f = f_complex.real
    native_f = np.array([[b.basis.fock(i, j) for j in range(n)] for i in range(n)])
    f_error = max(float(np.max(abs(f_complex.imag))), float(np.max(abs(f-native_f))))
    if f_error > 1e-11:
        raise ValueError("independent actual HF Fock differs from native selected basis")
    used_f = (f+f.T)/2
    used_f[:o, o:] = used_f[o:, :o] = 0.0
    used_f[o:, o:] = np.diag(np.diag(used_f[o:, o:]))
    f_projection = float(np.linalg.norm(f_complex-used_f))
    if f_projection > 1e-10:
        raise ValueError("actual HF semicanonical projection is substantive")
    if system == "he2":
        _original_hamiltonian_check(b, dict(o=o, v=v, fock=used_f, eri=g))
        if np.max(abs(used_f[:o, :o]-np.diag(np.diag(used_f[:o, :o])))) < 1e-3:
            raise ValueError("two-helium oracle lost its substantive occupied Fock coupling")
    # Independent complete-active occupied rotation; no core-active mixing.
    rotation = np.eye(n)
    if o > 1:
        raw = np.random.default_rng(712+nk).normal(size=(o, o))
        rotation[:o, :o], _ = np.linalg.qr(raw)
    rotated_f = rotation.T @ used_f @ rotation
    rotated_f = (rotated_f+rotated_f.T)/2
    rotated_g, rotation_projection = _eightfold(np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl", g, rotation, rotation, rotation, rotation, optimize=True))
    rotated = dict(o=o, v=v, fock=np.ascontiguousarray(rotated_f), eri=rotated_g)
    rcc, rt = None, None
    # The direct-integral diagnostic input cap remains o<=3. K2 He2 still
    # compares the actual noncanonical localized native result with PySCF's
    # independent full-active canonicalization, without widening that seam.
    if o <= 3:
        rcc = dense_ccsd(rotated, options=ccsd_options(maximum_iterations=maximum_iterations))
        if not rcc.final_snapshot.converged:
            raise ValueError("occupied-rotated native CCSD failed its unchanged tolerances")
        rotated.update(t1=rcc.t1, t2=rcc.t2, foo=np.ascontiguousarray(rotated_f[:o, :o]),
            fvv=np.ascontiguousarray(rotated_f[o:, o:]), fov=np.ascontiguousarray(rotated_f[:o, o:]))
        rt = dense_triples(rotated, options=triples_options(maximum_iterations=maximum_iterations))
        if not rt.final_snapshot.converged:
            raise ValueError("occupied-rotated native coupled(T) failed its unchanged tolerances")
    metadata = dict(system=system, nk=nk, occupied=o, virtual=v,
        hf_source=b.hf.reference_source_identity_sha256, provider=wrapper.identity_sha256,
        hf_energy_per_cell=b.hf.diagnostics.captured_energy_per_cell,
        native_ccsd=result.ccsd.final_snapshot.correlation_energy,
        native_triples=result.triples.final_snapshot.triples_energy,
        rotated_ccsd=None if rcc is None else rcc.final_snapshot.correlation_energy,
        rotated_triples=None if rt is None else rt.final_snapshot.triples_energy,
        extra_random_rotation_evaluated=rcc is not None,
        pm_riemannian_gradient_tolerance=pm_tolerance,
        original_foo_offdiagonal=float(np.max(abs(used_f[:o, :o]-np.diag(np.diag(used_f[:o, :o]))))),
        native_iterations=result.ccsd.final_snapshot.iteration, native_triples_iterations=result.triples.final_snapshot.iteration,
        maximum_eri_imaginary=imaginary, eri_symmetry_projection=symmetry_projection,
        native_independent_eri_difference=eri_difference, native_independent_fock_difference=f_error,
        fock_projection_norm=f_projection, rotation_eri_projection=rotation_projection,
        rotated_foo_offdiagonal=float(np.max(abs(rotated_f[:o, :o]-np.diag(np.diag(rotated_f[:o, :o]))))),
        complete_finite_torus=True, infinite_source_accuracy=False, frozen_core_count=b.reference.state.n_frozen_core)
    np.savez(output, fock=used_f, eri=g, metadata=np.asarray(json.dumps(metadata)))
    print(json.dumps(dict(stage="native-export", **metadata)), flush=True)


def _load_tiny_artifact(path):
    import numpy as np
    import zipfile

    if path.stat().st_size > 65536:
        raise ValueError("oracle artifact exceeds tiny64KiB cap")
    # Preflight uncompressed size AND every NPY header before np.load can
    # allocate from a caller-supplied shape. A compressed-size gate alone
    # would not protect the tiny diagnostic from an oversized array header.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if {entry.filename for entry in entries} != {"fock.npy", "eri.npy", "metadata.npy"} or len(entries) != 3:
            raise ValueError("oracle archive fields are not the three expected arrays")
        if sum(entry.file_size for entry in entries) > 65536:
            raise ValueError("oracle uncompressed artifact exceeds64KiB")
        shapes = {}
        for entry in entries:
            with archive.open(entry) as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError("oracle requires tiny version1 NPY entries")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
                if dtype.hasobject or fortran:
                    raise ValueError("oracle requires nonobject C-order arrays")
                if entry.filename == "metadata.npy":
                    if shape != () or dtype.kind != "U" or dtype.itemsize > 32768:
                        raise ValueError("oracle metadata exceeds its tiny string extent")
                    size = dtype.itemsize
                else:
                    dimensions = 2 if entry.filename == "fock.npy" else 4
                    if dtype != np.dtype("float64") or len(shape) != dimensions or not all(2 <= d <= 8 for d in shape):
                        raise ValueError("oracle numeric shape exceeds tiny bounds")
                    size = int(np.prod(shape))*8
                if stream.tell()+size != entry.file_size:
                    raise ValueError("oracle NPY payload length disagrees with bounded header")
                shapes[entry.filename] = shape
        if len(set(shapes["fock.npy"])) != 1 or shapes["eri.npy"] != (shapes["fock.npy"][0],)*4:
            raise ValueError("oracle Fock and ERI dimensions differ")
    with np.load(path, allow_pickle=False) as archive:
        return archive["fock"], archive["eri"], json.loads(str(archive["metadata"]))


def _pyscf(path):
    import numpy as np
    import pyscf
    from pyscf import ao2mo, cc, gto, lib, scf

    lib.num_threads(1)
    f, g, data = _load_tiny_artifact(path)
    o, v, n = data["occupied"], data["virtual"], f.shape[0]
    system, nk = data.get("system", "he"), data["nk"]
    if not ((system == "he" and nk in (1, 2, 3) and o == nk)
            or (system == "he2" and nk in (1, 2) and o == 2*nk)):
        raise ValueError("oracle artifact is not an admitted complete He/He2 torus")
    if not (1 <= o == v <= 4) or n != o+v or f.shape != (n, n) or g.shape != (n,)*4:
        raise ValueError("oracle input shape exceeds tiny complete He torus contract")
    if not np.isfinite(f).all() or not np.isfinite(g).all():
        raise ValueError("oracle contains nonfinite numeric input")
    eo, uo = np.linalg.eigh(f[:o, :o])
    ev, uv = np.linalg.eigh(f[o:, o:])
    u = np.zeros((n, n))
    u[:o, :o], u[o:, o:] = uo, uv
    canonical_f = u.T @ f @ u
    canonical_g, projection = _eightfold(np.einsum("pqrs,pi,qj,rk,sl->ijkl", g, u, u, u, u, optimize=True))
    offdiag = float(np.max(abs(canonical_f-np.diag(np.diag(canonical_f)))))
    if offdiag > 1e-11 or min(ev)-max(eo) <= 0.01:
        raise ValueError("actual HF cannot be used as the canonical insulating oracle")
    dm = np.diag(np.r_[np.full(o, 2.0), np.zeros(v)])
    j = np.einsum("pqrs,sr->pq", canonical_g, dm)
    exchange = np.einsum("prqs,rs->pq", canonical_g, dm)
    h = canonical_f-j+0.5*exchange
    # Empty Mole is only a PySCF custom-integral carrier. No fictitious atom,
    # basis, SCF run or physical source is assigned to it. All frozen-core
    # contributions already in the actual F remain in this effective h.
    mol = gto.M(verbose=0)
    mol.nelectron = 2*o
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.chkfile = None
    mf.max_memory = 64
    mf._eri = ao2mo.restore(8, canonical_g, n)
    mf.get_hcore = lambda *args: h
    mf.get_ovlp = lambda *args: np.eye(n)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = np.eye(n), np.diag(dm), np.r_[eo, ev]
    # Keep the actual finite-torus reference, including frozen-core/nuclear
    # constants, without claiming PySCF ran or converged this SCF. PySCF's
    # get_e_hf recomputes energy_tot for this deliberately unconverged carrier.
    reference_total = float(data["hf_energy_per_cell"]*data["nk"])
    electronic_reference = float(0.5*np.einsum("pq,qp", dm, h+canonical_f))
    reference_constant = reference_total-electronic_reference
    mf.energy_nuc = lambda: reference_constant
    mf.e_tot = reference_total
    rebuilt = mf.get_fock(dm=dm)
    reconstructed_error = float(np.max(abs(rebuilt-canonical_f)))
    if reconstructed_error > 1e-12:
        raise ValueError("PySCF custom-integral F rebuild differs from actual HF F")
    reconstructed_energy_error = abs(float(mf.energy_tot(dm=dm))-reference_total)
    if reconstructed_energy_error > 1e-12:
        raise ValueError("PySCF custom-integral reference energy differs from actual HF")
    solver = cc.CCSD(mf)
    solver.max_memory = 64
    solver.incore_complete, solver.async_io = True, False
    solver.max_cycle, solver.conv_tol, solver.conv_tol_normt = 100, 1e-13, 1e-11
    solver.diis_space = 6
    solver.level_shift, solver.iterative_damping = 0, 1.0
    eris = solver.ao2mo()
    if np.max(abs(eris.fock-canonical_f)) > 1e-12:
        raise ValueError("PySCF ERI initializer changed actual HF F")
    def event(values):
        print(json.dumps(dict(stage="pyscf-ccsd", nk=data["nk"], iteration=int(values["istep"])+1)), flush=True)
    solver.callback = event
    ecc, _, _ = solver.kernel(eris=eris)
    if not solver.converged:
        raise ValueError("canonical PySCF CCSD did not converge")
    if abs(float(solver.e_hf)-reference_total) > 1e-12:
        raise ValueError("PySCF CCSD reference energy differs from actual HF")
    triples = float(solver.ccsd_t(eris=eris))
    result = dict(stage="comparison", pyscf_version=pyscf.__version__, **data,
        pyscf_ccsd=float(ecc), pyscf_triples=triples,
        ccsd_difference=float(ecc-data["native_ccsd"]), triples_difference=float(triples-data["native_triples"]),
        rotated_ccsd_difference=None if data["rotated_ccsd"] is None else float(ecc-data["rotated_ccsd"]),
        rotated_triples_difference=None if data["rotated_triples"] is None else float(triples-data["rotated_triples"]),
        canonical_eri_projection=projection, canonical_fock_offdiagonal=offdiag,
        pyscf_reconstructed_fock_error=reconstructed_error,
        pyscf_reconstructed_reference_energy_error=reconstructed_energy_error)
    print(json.dumps(result), flush=True)
    if abs(result["ccsd_difference"]) > 2e-11 or abs(result["triples_difference"]) > 2e-12:
        raise ValueError("same-Hamiltonian canonical/native CCSD(T) comparison failed")
    if (result["rotated_ccsd_difference"] is not None
            and (abs(result["rotated_ccsd_difference"]) > 2e-11
                or abs(result["rotated_triples_difference"]) > 2e-12)):
        raise ValueError("same-Hamiltonian occupied-rotation CCSD(T) comparison failed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "guard":
        return _guard(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("native", "pyscf"))
    parser.add_argument("--nk", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--system", choices=("he", "he2"), default="he")
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "native":
        _native(args.nk, args.artifact, args.system)
    else:
        _pyscf(args.artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
