"""Solver-wall mechanism probe (IID 130 / IID 101 family).

Rebuilds the bare GFN2-SECCM SCC map (ewald_gamma, aes off, gam3 off) in
Python for the elongated fcc Cu 2x2x4 cell (the wall) and the 2x2x2 cell
(the converging control), then isolates the channel that carries the wall
by swapping the H0/S fold and the gamma kernel between the production
Wigner-Seitz weighting and an all-weights-1 fold (~ the Bloch-Gamma
supercell map that xtb converges in 16 iterations).

For each variant: S spectrum, band gap / DOS at the Fermi level, FD
Jacobian spectrum at the neutral point, a line-searched Newton hunt for
the fixed point, and a simple-mixing trajectory character.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy.linalg import eigh as seigh

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as se
from vibeqc._vibeqc_core.semiempirical import (
    SemiempiricalBasis,
    gfn2_enumerate_shells,
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
Z = 29


def gam3_free_params():
    from vibeqc.semiempirical.methods.gfn2_params import (
        _compute_pairwise_repulsive,
        _read_cached_toml,
    )

    data = _read_cached_toml()
    params = se.xtb.GFN2ParameterSet()
    rep = {}
    for blob in data.get("element", []):
        zn = int(blob["Z"])
        ed = se.xtb.GFN2ElementData()
        ed.Z = zn
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
        rep[zn] = (float(blob.get("repa", 0.0)), float(blob.get("repb", 0.0)))
    _compute_pairwise_repulsive(params, rep)
    return params


def build(a_angstrom, replicas):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    mol = Molecule([Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1)
    topo = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return mol, topo


def one_shot(mol, topo, params, weights_override=None, temperature=0.0, max_iter=1):
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
    ) = flatten_topology_records(mol, topo, route_name="GFN2-SECCM")
    if weights_override is not None:
        weights = weights_override(np.asarray(weights, dtype=float)).tolist()
    group = topo.finite_group
    native = se._run_gfn2_seccm_from_records(
        mol,
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
        float(group.geometry_tolerance) * topology_length_unit_scale(topo),
        max_iter=max_iter,
        conv_tol_charge=1.0e-6,
        charge_mixing=0.1,
        electronic_temperature=temperature,
        madelung=False,
        madelung_s_weighted=False,
        madelung_no_self=False,
        ewald_gamma=True,
        include_aes=False,
    )
    rec = dict(
        translations=translations,
        central=central,
        origin=origin,
        shell_labels=np.asarray(shell_labels, dtype=np.int32),
        weights=weights,
        multiplicities=multiplicities,
        displacements=np.asarray(displacements, dtype=float),
    )
    return native, rec


def gamma_matrix(mol, params, rec, alpha):
    return np.asarray(
        se._gfn2_seccm_ewald_shell_gamma_from_records(
            mol,
            params,
            rec["translations"],
            rec["central"],
            rec["origin"],
            rec["shell_labels"],
            rec["weights"],
            rec["multiplicities"],
            rec["displacements"],
            alpha,
        )
    )


def fermi_occ(eps, n_occ, temp):
    """Match kpoints_occupations.cpp: factor-2 FD, +-50 clamp."""
    if temp <= 0.0:
        occ = np.zeros_like(eps)
        occ[:n_occ] = 2.0
        return occ
    target = 2.0 * n_occ

    def count(mu):
        arg = np.clip((eps - mu) / temp, -50.0, 50.0)
        return (2.0 / (1.0 + np.exp(arg))).sum()

    lo, hi = eps.min() - 1.0, eps.max() + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if count(mid) < target:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    arg = np.clip((eps - mu) / temp, -50.0, 50.0)
    return 2.0 / (1.0 + np.exp(arg))


class BareMap:
    def __init__(self, H0, S, gamma, ao_shell, n_shells, n_occ, temp, label):
        self.H0, self.S, self.gamma = H0, S, gamma
        self.ao_shell, self.n_shells, self.n_occ = ao_shell, n_shells, n_occ
        self.temp = temp
        self.label = label
        self.n0 = None
        # canonical orthogonalization mirror: screen S modes below tol
        s_eval, s_evec = np.linalg.eigh(S)
        self.s_eval = s_eval
        keep = s_eval > 1.0e-8
        self.X = s_evec[:, keep] / np.sqrt(s_eval[keep])
        self.screened = int((~keep).sum())

    def raw_pops(self, dq):
        V = self.gamma @ dq
        Vao = V[self.ao_shell]
        H = self.H0 + 0.5 * self.S * (Vao[:, None] + Vao[None, :])
        Ht = self.X.T @ H @ self.X
        eps, Ct = np.linalg.eigh(Ht)
        C = self.X @ Ct
        occ = fermi_occ(eps, self.n_occ, self.temp)
        D = (C * occ) @ C.T
        pops_ao = np.einsum("ij,ji->i", D, self.S)
        pops = np.zeros(self.n_shells)
        np.add.at(pops, self.ao_shell, pops_ao)
        return pops, eps

    def calibrate(self, f0_native):
        pops0, _ = self.raw_pops(np.zeros(self.n_shells))
        self.n0 = pops0 - f0_native

    def __call__(self, dq):
        pops, _ = self.raw_pops(dq)
        return pops - self.n0

    def spectrum_at(self, dq):
        _, eps = self.raw_pops(dq)
        return eps


def analyze(mapf, tag):
    n = mapf.n_shells
    f0 = mapf(np.zeros(n))
    eps = mapf.spectrum_at(np.zeros(n))
    homo, lumo = eps[mapf.n_occ - 1], eps[mapf.n_occ]
    ef = 0.5 * (homo + lumo)
    near = int(((eps > ef - 0.01) & (eps < ef + 0.01)).sum())
    print(
        f"  [{tag}] |f(0)|_inf={np.abs(f0).max():.4f}  gap={lumo - homo:+.6f}"
        f"  levels within +-10 mHa of E_F: {near}"
        f"  S: min_eig={mapf.s_eval.min():+.3e} screened={mapf.screened}"
    )
    # FD Jacobian at 0
    delta = 1.0e-4
    J = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = delta
        J[:, k] = (mapf(e) - f0) / delta
    ev = np.sort(np.linalg.eigvals(J).real)
    print(f"  [{tag}] J(0) eigenvalues: min={ev.min():+.2f} max={ev.max():+.2f}")

    # simple mixing m=0.1
    dq = np.zeros(n)
    trace = []
    for _ in range(400):
        fv = mapf(dq)
        trace.append(float(np.abs(fv - dq).max()))
        dq = 0.1 * fv + 0.9 * dq
    print(
        f"  [{tag}] simple m=0.1, 400 it: res first/last "
        f"{trace[0]:.3f} -> {trace[-1]:.3e}; tail(6) "
        f"{np.array2string(np.array(trace[-6:]), precision=4)}"
    )
    conv_simple = trace[-1] < 1e-6

    # line-searched Newton from neutral
    def jac(q, f_q):
        Jl = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = delta
            Jl[:, k] = (mapf(q + e) - f_q) / delta
        return Jl

    q = np.zeros(n)
    ok = False
    for it in range(80):
        fq = mapf(q)
        r = fq - q
        rn = np.abs(r).max()
        if rn < 1.0e-9:
            ok = True
            break
        Jl = jac(q, fq)
        try:
            step = np.linalg.solve(Jl - np.eye(n), r)
        except np.linalg.LinAlgError:
            break
        damping = 1.0
        accepted = False
        for _ in range(14):
            qn = q - damping * step
            if np.abs(mapf(qn) - qn).max() < rn:
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            # fall back to a damped simple step
            qn = q + 0.1 * r
            if np.abs(mapf(qn) - qn).max() >= rn:
                break
        q = qn
    rfin = np.abs(mapf(q) - q).max()
    print(
        f"  [{tag}] newton: converged={ok} final |f(q)-q|_inf={rfin:.3e} "
        f"|q|_inf={np.abs(q).max():.3f}"
    )
    if ok:
        Js = jac(q, mapf(q))
        evs = np.sort(np.linalg.eigvals(Js).real)
        print(
            f"  [{tag}] J(q*) eigenvalues: min={evs.min():+.2f} "
            f"max={evs.max():+.2f}"
        )
    return dict(conv_simple=conv_simple, newton=ok, q=q, res=rfin)


def run_cell(replicas, a, temps):
    print(f"== fcc Cu {replicas} a={a} ==", flush=True)
    mol, topo = build(a, replicas)
    params = gam3_free_params()

    native_ws, rec_ws = one_shot(mol, topo, params)
    H0 = np.asarray(native_ws.hamiltonian, dtype=float)
    S = np.asarray(native_ws.overlap, dtype=float)
    f0_ws = np.asarray(native_ws.shell_charges, dtype=float)
    n_occ = int(native_ws.n_occ)
    n_basis = int(native_ws.n_basis)

    basis = SemiempiricalBasis.build(mol, params, 0)
    shells = list(gfn2_enumerate_shells(basis, mol, params))
    n_shells = len(shells)
    ao_shell = np.zeros(n_basis, dtype=int)
    for si, sh in enumerate(shells):
        for mu in range(int(sh.bf_start), int(sh.bf_start) + int(sh.n_funcs)):
            ao_shell[mu] = si

    T = np.asarray(topo.translations)
    volume = abs(np.linalg.det(T))
    alpha = math.sqrt(math.pi) / volume ** (1.0 / 3.0)
    gamma_ws = gamma_matrix(mol, params, rec_ws, alpha)

    # Bloch-fold variant: same records, every weight = 1
    try:
        native_bl, rec_bl = one_shot(
            mol, topo, params, weights_override=lambda w: np.ones_like(w)
        )
        H0_bl = np.asarray(native_bl.hamiltonian, dtype=float)
        S_bl = np.asarray(native_bl.overlap, dtype=float)
        f0_bl = np.asarray(native_bl.shell_charges, dtype=float)
        gamma_bl = gamma_matrix(mol, params, rec_bl, alpha)
        have_bloch = True
    except Exception as exc:  # noqa: BLE001
        print(f"  bloch-fold one-shot REFUSED: {exc}")
        have_bloch = False

    for temp in temps:
        print(f"-- T = {temp} --", flush=True)
        m_ws = BareMap(H0, S, gamma_ws, ao_shell, n_shells, n_occ, temp, "ws")
        if temp == 0.0:
            m_ws.calibrate(f0_ws)
            cal = np.abs(m_ws(np.zeros(n_shells)) - f0_ws).max()
            print(f"  [ws] f(0) calibration vs native: {cal:.3e}")
        else:
            nat_t, _ = one_shot(mol, topo, params, temperature=temp)
            f0_t = np.asarray(nat_t.shell_charges, dtype=float)
            m_ws.calibrate(f0_t)
            cal = np.abs(m_ws(np.zeros(n_shells)) - f0_t).max()
            print(f"  [ws] f(0) calibration vs native (T={temp}): {cal:.3e}")
        analyze(m_ws, f"ws T={temp}")

        if have_bloch:
            m_bl = BareMap(
                H0_bl, S_bl, gamma_bl, ao_shell, n_shells, n_occ, temp, "bloch"
            )
            if temp == 0.0:
                m_bl.calibrate(f0_bl)
            else:
                nat_bt, _ = one_shot(
                    mol,
                    topo,
                    params,
                    weights_override=lambda w: np.ones_like(w),
                    temperature=temp,
                )
                m_bl.calibrate(np.asarray(nat_bt.shell_charges, dtype=float))
            analyze(m_bl, f"bloch T={temp}")

            # channel crosses: which fold carries the wall?
            m_x1 = BareMap(
                H0_bl, S_bl, gamma_ws, ao_shell, n_shells, n_occ, temp, "x1"
            )
            m_x1.n0 = m_bl.n0
            analyze(m_x1, f"H0/S=bloch gamma=ws T={temp}")
            m_x2 = BareMap(
                H0, S, gamma_bl, ao_shell, n_shells, n_occ, temp, "x2"
            )
            m_x2.n0 = m_ws.n0
            analyze(m_x2, f"H0/S=ws gamma=bloch T={temp}")


def main():
    run_cell((2, 2, 2), 3.615, temps=[0.001, 0.005])
    run_cell((2, 2, 4), 3.7958, temps=[0.001, 0.005])


if __name__ == "__main__":
    sys.exit(main())
