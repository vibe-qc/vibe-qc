"""Neutral-point SCC map analysis for an aligned-plane stress fixture.

This intentionally retains the historical IID 141 geometry: aligned
single-species square planes with alternating Mg and O labels. It is a
synthetic numerical stress test, not B1 rocksalt(100), B2/CsCl(001), or
any claimed MgO polymorph. No physical surface conclusion may be drawn
from its energies, charges, or SCC stability.

Reconstructs the GFN2-SECCM SCC map f(dq) in Python around the neutral
point using the max_iter=1 failed-exit diagnostics (hamiltonian == H0,
overlap == S, shell_charges == f(0)) and the Python reimplementation of
the 2-D Parry/Heyes KO kernel. The Jacobian J = df/dq at dq = 0 is
built by finite differences of the one-shot map (no SCC iteration), and
its eigenvalues are compared to the damped-simple-mixing stability
bound |1 + m (lambda - 1)| < 1. A lambda < -19 at m = 0.1 is the
sign-flipping (period-2) instability seen in scc_max_change_trace.

Bare map (include_aes=False): V = Gamma . dq only, so the Python map
reproduces the C++ map exactly.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh as _seigh

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc._vibeqc_core.semiempirical import (
    gfn2_enumerate_shells,
    SemiempiricalBasis,
)
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.seccm._adapter_common import (
    flatten_topology_records,
    topology_length_unit_scale,
)
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def _dual_height_bounds(
    vector_1: np.ndarray, vector_2: np.ndarray, cutoff: float
) -> tuple[int, int]:
    """Return coefficient-complete bounds for a 2-D lattice sphere."""
    area = float(np.linalg.norm(np.cross(vector_1, vector_2)))
    if area <= 0.0:
        raise ValueError("2-D lattice vectors must be linearly independent")
    height_1 = area / float(np.linalg.norm(vector_2))
    height_2 = area / float(np.linalg.norm(vector_1))
    return (
        int(math.ceil(cutoff / height_1)) + 1,
        int(math.ceil(cutoff / height_2)) + 1,
    )


def _aligned_plane_stress_fixture(layers: int):
    a = 4.212
    plane_z = [k * a / 2.0 for k in range(layers)]
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(layers):
        species = 12 if k % 2 == 0 else 8
        for i in range(2):
            for j in range(2):
                atoms.append(np.array([i * a, j * a, plane_z[k]]))
                zs.append(species)
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    translations = [
        np.array([2.0 * a, 0.0, 0.0]),
        np.array([0.0, 2.0 * a, 0.0]),
    ]
    prim = [np.array([a, 0.0, 0.0]), np.array([0.0, a, 0.0])]
    topology = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=(2, 2, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def _gamma_reference_2d(shells, coords_bohr, topology, alpha) -> np.ndarray:
    """Python reimplementation of ewald_shell_gamma_2d (tblite
    get_amat_3d convention, 2-D Parry/Heyes Coulomb). Mirrors the C++
    cutoffs and enumeration exactly."""
    T = np.asarray(topology.translations)
    a1, a2 = T[0], T[1]
    normal = np.cross(a1, a2)
    area = float(np.linalg.norm(normal))
    nhat = normal / area
    twopi = 2.0 * math.pi
    b1 = twopi * np.cross(a2, nhat) / area
    b2 = twopi * np.cross(nhat, a1) / area

    k_cut = alpha * math.sqrt(120.0)
    # Coefficient-complete bounds use the dual heights.  Norm quotients
    # miss short vectors formed by near-cancelling coefficients in a skew
    # basis and make the finite Ewald sum basis-dependent.
    kmax = _dual_height_bounds(b1, b2, k_cut)
    kvecs: list[np.ndarray] = []
    for i in range(-kmax[0], kmax[0] + 1):
        for j in range(-kmax[1], kmax[1] + 1):
            if i == j == 0:
                continue
            k = i * b1 + j * b2
            if float(k @ k) > k_cut * k_cut:
                continue
            kvecs.append(k)

    r_cut = math.sqrt(30.0) / alpha
    fractional = np.column_stack((coords_bohr @ b1 / twopi, coords_bohr @ b2 / twopi))
    pair_fractional_span = np.ptp(fractional, axis=0)
    nmax = (
        int(math.ceil(r_cut * np.linalg.norm(b1) / twopi + pair_fractional_span[0]))
        + 1,
        int(math.ceil(r_cut * np.linalg.norm(b2) / twopi + pair_fractional_span[1]))
        + 1,
    )
    images = [
        i * a1 + j * a2
        for i in range(-nmax[0], nmax[0] + 1)
        for j in range(-nmax[1], nmax[1] + 1)
    ]
    self_const = 2.0 * alpha / math.sqrt(math.pi)
    recip_pref = math.pi / area
    k0_pref = twopi / area
    hardness = np.array([max(float(s.hardness), 1.0e-6) for s in shells])

    n = len(shells)
    gamma = np.zeros((n, n))
    for a in range(n):
        atom_a = int(shells[a].atom_idx)
        for b in range(a, n):
            atom_b = int(shells[b].atom_idx)
            eta_ab = 2.0 / (hardness[a] + hardness[b])
            d = coords_bohr[atom_b] - coords_bohr[atom_a]
            same_site = atom_a == atom_b
            value = 0.0
            for shift in images:
                r = float(np.linalg.norm(d + shift))
                if r < 1.0e-12 or r > r_cut:
                    continue
                value += math.erfc(alpha * r) / r
            zc = float(d @ nhat)
            for k in kvecs:
                km = float(np.linalg.norm(k))
                krij = float(k @ d)
                value += (
                    recip_pref
                    * math.cos(krij)
                    / km
                    * (
                        math.exp(km * zc) * math.erfc(alpha * zc + km / (2.0 * alpha))
                        + math.exp(-km * zc)
                        * math.erfc(-alpha * zc + km / (2.0 * alpha))
                    )
                )
            alpha_z = alpha * zc
            value -= k0_pref * (
                zc * math.erf(alpha_z)
                + math.exp(-alpha_z * alpha_z) / (alpha * math.sqrt(math.pi))
            )
            if same_site:
                value -= self_const
            value += 1.0 / eta_ab if same_site else 0.0
            for image in topology.cells[atom_a]:
                if image.origin != atom_b:
                    continue
                r = float(np.linalg.norm(image.disp))
                if r < 1.0e-12:
                    continue
                f = 1.0 / math.sqrt(r * r + eta_ab * eta_ab)
                value += image.weight * (f - 1.0 / r)
            gamma[a, b] = value
            gamma[b, a] = value
    return gamma


def _params_gam3_free():
    from vibeqc.semiempirical.methods.gfn2_params import (
        _compute_pairwise_repulsive,
        _read_cached_toml,
    )

    data = _read_cached_toml()
    params = _se_cxx.xtb.GFN2ParameterSet()
    rep = {}
    for blob in data.get("element", []):
        Z = int(blob["Z"])
        ed = _se_cxx.xtb.GFN2ElementData()
        ed.Z = Z
        ed.gam = float(blob.get("gam", 0.5))
        ed.gam3 = 0.0
        ed.alpha = float(blob.get("alpha", 1.0))
        ed.dpol = float(blob.get("dpol", 0.0))
        ed.qpol = float(blob.get("qpol", 0.0))
        ed.mp_rad = float(blob.get("mp_rad", 0.0))
        ed.mp_vcn = float(blob.get("mp_vcn", 0.0))
        for sh in blob.get("shells", []):
            ed.add_shell(
                int(sh.get("l", 0)),
                float(sh.get("en", 0.0)),
                float(sh.get("zeta", 1.0)),
                float(sh.get("k_en", 1.0)),
                float(sh.get("kcn", 0.0)),
                float(sh.get("poly", 0.0)),
                int(sh.get("n", 0)),
            )
        params.add_element(ed)
        rep[Z] = (
            float(blob.get("repa", 0.0)),
            float(blob.get("repb", 0.0)),
        )
    _compute_pairwise_repulsive(params, rep)
    return params


def _one_shot(molecule, topology, max_iter=1, include_aes=True, params=None):
    if params is None:
        params = load_gfn2_params()
    (
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        primitive_vectors,
        replicas,
    ) = flatten_topology_records(molecule, topology, route_name="GFN2-SECCM")
    group = topology.finite_group
    native = _se_cxx._run_gfn2_seccm_from_records(
        molecule,
        params,
        translations,
        central,
        origin,
        np.asarray(shell_labels, dtype=np.int32),
        weights,
        multiplicities,
        np.asarray(displacements, dtype=float),
        primitive_vectors,
        replicas,
        float(group.geometry_tolerance) * topology_length_unit_scale(topology),
        max_iter=max_iter,
        conv_tol_charge=1.0e-6,
        charge_mixing=0.1,
        electronic_temperature=0.0,
        madelung=False,
        madelung_s_weighted=False,
        madelung_no_self=False,
        ewald_gamma=True,
        include_aes=include_aes,
    )
    return native


def analyze(layers: int) -> None:
    molecule, topology = _aligned_plane_stress_fixture(layers)
    native = _one_shot(molecule, topology, max_iter=1, include_aes=False)

    H0 = np.asarray(native.hamiltonian, dtype=float)
    S = np.asarray(native.overlap, dtype=float)
    f0 = np.asarray(native.shell_charges, dtype=float)
    n_occ = int(native.n_occ)
    n_basis = int(native.n_basis)
    assert H0.shape == (n_basis, n_basis)

    params = load_gfn2_params()
    basis = SemiempiricalBasis.build(molecule, params, 0)
    shells = list(gfn2_enumerate_shells(basis, molecule, params))
    n_shells = len(shells)
    assert n_shells == f0.size

    ao_shell = np.zeros(n_basis, dtype=int)
    for si, sh in enumerate(shells):
        for mu in range(int(sh.bf_start), int(sh.bf_start) + int(sh.n_funcs)):
            ao_shell[mu] = si

    atoms_bohr = np.array([np.asarray(a.xyz) for a in molecule.atoms])
    T = np.asarray(topology.translations)
    area = float(np.linalg.norm(np.cross(T[0], T[1])))
    alpha = 0.85 * math.sqrt(math.pi) / math.sqrt(area)
    gamma = _gamma_reference_2d(shells, atoms_bohr, topology, alpha)

    def f(v: np.ndarray) -> np.ndarray:
        V = gamma @ v
        H = H0.copy()
        for mu in range(n_basis):
            for nu in range(n_basis):
                H[mu, nu] += 0.5 * S[mu, nu] * (V[ao_shell[mu]] + V[ao_shell[nu]])
        eps, C = _seigh(H, S)
        order = np.argsort(eps)
        D = 2.0 * C[:, order[:n_occ]] @ C[:, order[:n_occ]].T
        DS = D @ S
        fv = np.array(
            [
                sum(DS[mu, mu] for mu in range(n_basis) if ao_shell[mu] == si)
                for si in range(n_shells)
            ]
        )
        return fv

    # n0 calibration: f(0) from the C++ must be reproduced exactly.
    n0 = f(np.zeros(n_shells)) - f0
    fv_cal = f(np.zeros(n_shells)) - n0
    print(f"  f(0) calibration error: {np.abs(fv_cal - f0).max():.3e}")

    def f_map(v: np.ndarray) -> np.ndarray:
        return f(v) - n0

    delta = 1.0e-4
    J = np.zeros((n_shells, n_shells))
    for k in range(n_shells):
        e = np.zeros(n_shells)
        e[k] = delta
        J[:, k] = (f_map(e) - f0) / delta

    evals = np.linalg.eigvals(J)
    real = np.sort(evals.real)
    print(f"  J eigenvalues: min={real.min():+.3f} max={real.max():+.3f}")
    n_unstable = sum(abs(1.0 + 0.1 * (l - 1.0)) > 1.0 + 1e-9 for l in real)
    print(f"  modes outside the m=0.1 stability bound: {n_unstable}")

    # Simulate the damped simple-mixing iteration (the C++ loop: at
    # iteration k, max_change = |f(dq_k) - dq_k|, then
    # dq_{k+1} = m f(dq_k) + (1 - m) dq_k) and compare the trace with the
    # native run's scc_max_change_trace (the same bare map, T = 0).
    def simulate(m, n_iter):
        dq = np.zeros(n_shells)
        trace = []
        for _ in range(n_iter):
            fn = f_map(dq)
            trace.append(float(np.abs(fn - dq).max()))
            dq = m * fn + (1.0 - m) * dq
        return trace, dq

    trace_py, dq_final = simulate(0.1, 120)
    print(
        f"  python simulation tail (last 12 of 120): "
        f"{np.array2string(np.array(trace_py[-12:]), precision=3)}"
    )
    native_full = _one_shot(
        molecule,
        topology,
        max_iter=120,
        include_aes=False,
        params=_params_gam3_free(),
    )
    trace_cpp = np.asarray(native_full.scc_max_change_trace)
    print(
        f"  native run tail (last 12 of {trace_cpp.size}): "
        f"{np.array2string(trace_cpp[-12:], precision=3)}"
    )
    n_common = min(len(trace_py), trace_cpp.size)
    print(
        f"  max |py - cpp| over the shared prefix ({n_common} it): "
        f"{np.abs(np.array(trace_py[:n_common]) - trace_cpp[:n_common]).max():.3e}"
    )

    # Dissect the cycle: run the simulation to the saturated cycle and
    # report the per-layer atomic-charge pattern of the cycle states.
    dq = np.zeros(n_shells)
    for _ in range(400):
        dq = 0.1 * f_map(dq) + 0.9 * dq
    atom_idx = np.array([int(sh.atom_idx) for sh in shells])
    dq_atom = -np.array([sum(dq[atom_idx == a]) for a in range(4 * layers)])
    per_layer = np.array([sum(dq_atom[4 * l : 4 * l + 4]) for l in range(layers)])
    print(
        f"  cycle state per-layer atomic charges (py, gam3 off): "
        f"{np.array2string(per_layer, precision=3)}"
    )

    # Newton search for the fixed point of the bare map (T = 0, Aufbau).
    # The map is piecewise-linear (occupation switching), so the Jacobian
    # is re-built by finite differences at every Newton step and the step
    # is damped. Then: is the fixed point locally attracting under
    # simple mixing, and can a smaller mixing reach it from the neutral
    # start?
    def jacobian(q):
        J = np.zeros((n_shells, n_shells))
        for k in range(n_shells):
            e = np.zeros(n_shells)
            e[k] = delta
            J[:, k] = (f_map(q + e) - f_map(q)) / delta
        return J

    def newton(q0, n_iter=60):
        q = q0.copy()
        for _ in range(n_iter):
            r = f_map(q) - q
            if np.abs(r).max() < 1.0e-10:
                return q, True
            J = jacobian(q)
            try:
                step = np.linalg.solve(J - np.eye(n_shells), r)
            except np.linalg.LinAlgError:
                return q, False
            damping = 1.0
            for _ in range(12):
                q_new = q - damping * step
                if np.abs(f_map(q_new) - q_new).max() < np.abs(r).max():
                    break
                damping *= 0.5
            q = q - damping * step
        return q, False

    q_star, newton_ok = newton(np.zeros(n_shells))
    print(f"  newton converged: {newton_ok}")
    if newton_ok:
        r_star = np.abs(f_map(q_star) - q_star).max()
        J_star = jacobian(q_star)
        ev_star = np.sort(np.linalg.eigvals(J_star).real)
        print(
            f"  |f(q*) - q*| = {r_star:.2e}, J(q*) eigenvalues "
            f"min={ev_star.min():+.3f} max={ev_star.max():+.3f}"
        )
        damped = 1.0 + 0.1 * (ev_star - 1.0)
        print(
            f"  damped-map eigenvalues at q*: "
            f"min={damped.min():+.3f} max={damped.max():+.3f}"
        )
        print(
            f"  q* per-layer charges: "
            f"{np.array2string(-np.array([sum(q_star[atom_idx == a]) for a in range(4 * layers)]).reshape(layers, 4).sum(axis=1), precision=3)}"
        )
        # Is q* attracting under simple mixing from the neutral start
        # with a smaller m?
        for m in (0.1, 0.02, 0.005):
            dq = np.zeros(n_shells)
            for _ in range(600):
                dq = m * f_map(dq) + (1.0 - m) * dq
            dist = np.abs(dq - q_star).max()
            print(f"  m={m}: 600 it from neutral, |dq - q*|_inf = {dist:.3e}")

    if real.min() < -5.0:
        w, V = np.linalg.eig(J)
        k = int(np.argmin(w.real))
        mode = V[:, k].real
        mode /= np.abs(mode).max()
        atom_idx = np.array([int(sh.atom_idx) for sh in shells])
        per_layer = np.array([sum(mode[atom_idx == 4 * l]) for l in range(layers)])
        print(f"  most-negative mode (lambda={w[k].real:+.2f}):")
        print(f"    per-layer sums: {np.array2string(per_layer, precision=3)}")


def main() -> None:
    for layers in (2, 4):
        print(
            f"== {layers}-layer aligned-plane synthetic stress fixture, "
            "bare SCC map (aes off) =="
        )
        analyze(layers)


if __name__ == "__main__":
    main()
