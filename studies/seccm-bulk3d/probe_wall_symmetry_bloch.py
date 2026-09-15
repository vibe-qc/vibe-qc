"""Solver-wall mechanism probe, part 3: the coordination-number symmetry
break and the Bloch-fold comparison.

(A) Cyclic-symmetry check on the WS-folded H0: in an exact cyclic
    assembly every Cu atom is equivalent, so the on-site H0 diagonal
    must be identical across atoms. SECCM computes H0 coordination
    numbers molecularly (issue 354), so end/middle layers of the
    elongated cell get different CN -> different on-site levels.
(B) The periodic Gamma driver (run_gfn2_xtb_gamma) builds the Bloch-
    fold H0/S with periodic CNs: spectrum comparison at E_F.
(C) Cross maps: Bloch H0/S + SECCM ewald gamma (the 'xtb map' from our
    own pieces) vs WS H0/S + same gamma. Jacobian + Newton on each.
"""

from __future__ import annotations

import sys

import numpy as np

from probe_wall_map_mechanism import (
    BareMap,
    build,
    gam3_free_params,
    gamma_matrix,
    one_shot,
)
from probe_wall_spectra_thresholds import newton_hunt, setup
from vibeqc._vibeqc_core import Atom as CoreAtom, PeriodicSystem
from vibeqc._vibeqc_core.semiempirical.xtb import (
    XTBSccOptions,
    run_gfn2_xtb_gamma,
)

BOHR = 1.8897259886


def onsite_spread(tag, H0, ao_per_atom, n_atoms):
    d = np.diag(H0)
    per_atom = d.reshape(n_atoms, ao_per_atom)
    # s-level = first AO of each atom
    s_levels = per_atom[:, 0]
    print(
        f"[{tag}] on-site s-level per atom: min={s_levels.min():+.6f} "
        f"max={s_levels.max():+.6f} spread={np.ptp(s_levels):.3e}"
    )
    print(f"[{tag}]   values: {np.array2string(s_levels, precision=5)}")
    return np.ptp(s_levels)


def bloch_system(a_angstrom, replicas):
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
    lattice = np.array([n1 * prim[0], n2 * prim[1], n3 * prim[2]]) * BOHR
    system = PeriodicSystem()
    system.lattice = lattice
    system.unit_cell = [
        CoreAtom(29, (np.asarray(c) * BOHR).tolist()) for c in atoms
    ]
    return system


def main():
    for replicas, a in (((2, 2, 2), 3.615), ((2, 2, 4), 3.7958)):
        n_atoms = replicas[0] * replicas[1] * replicas[2]
        print(f"== fcc Cu {replicas} a={a} ==", flush=True)
        ctx = setup(replicas, a)
        ao_per_atom = ctx["n_basis"] // n_atoms

        # (A) cyclic-symmetry check on WS H0
        onsite_spread("WS-fold", ctx["H0"], ao_per_atom, n_atoms)

        # (B) Bloch fold via the periodic Gamma driver
        params = gam3_free_params()
        system = bloch_system(a, replicas)
        opts = XTBSccOptions()
        opts.max_iter = 1
        opts.auto_stabilize = False
        opts.electronic_temperature = 0.005
        try:
            res = run_gfn2_xtb_gamma(system, params, opts, cutoff_bohr=20.0)
        except Exception as exc:  # noqa: BLE001
            print(f"[bloch] driver REFUSED: {exc}")
            continue
        S_bl = np.asarray(res.overlap_gamma, dtype=float)
        H_bl = np.asarray(res.hamiltonian_gamma, dtype=float)
        ev_s = np.linalg.eigvalsh(S_bl)
        print(
            f"[bloch] S: min={ev_s.min():+.4e} neg={(ev_s < 0).sum()} "
            f"below_0.05={(ev_s < 0.05).sum()}"
        )
        onsite_spread("bloch", H_bl, ao_per_atom, n_atoms)

        # spectra at E_F: WS vs Bloch (both with their own S)
        from scipy.linalg import eigh as seigh

        n_occ = ctx["n_occ"]
        for tag, H, S in (("WS", ctx["H0"], ctx["S"]), ("BL", H_bl, S_bl)):
            se_, sv_ = np.linalg.eigh(S)
            keep = se_ > 1e-10
            X = sv_[:, keep] / np.sqrt(se_[keep])
            eps = np.linalg.eigvalsh(X.T @ H @ X)
            homo, lumo = eps[n_occ - 1], eps[n_occ]
            ef = 0.5 * (homo + lumo)
            near = int(((eps > ef - 0.01) & (eps < ef + 0.01)).sum())
            print(
                f"[{tag}] H0 spectrum: gap={lumo - homo:+.6f} "
                f"levels within +-10 mHa of E_F: {near}"
            )

        # (C) cross maps at T=0.005: Bloch H0/S + SECCM ewald gamma
        temp = 0.005
        n = ctx["n_shells"]
        nat_t, _ = one_shot(
            ctx["mol"], ctx["topo"], ctx["params"], temperature=temp
        )
        f0_t = np.asarray(nat_t.shell_charges, dtype=float)
        m_ws = BareMap(
            ctx["H0"], ctx["S"], ctx["gamma"], ctx["ao_shell"], n, n_occ,
            temp, "ws",
        )
        m_ws.calibrate(f0_t)
        n0 = m_ws.n0

        m_bl = BareMap(
            H_bl, S_bl, ctx["gamma"], ctx["ao_shell"], n, n_occ, temp, "bl"
        )
        m_bl.n0 = n0

        for tag, m in (("WS+ewald", m_ws), ("BLOCH+ewald", m_bl)):
            f0v = m(np.zeros(n))
            delta = 1.0e-4
            J = np.zeros((n, n))
            for k in range(n):
                e = np.zeros(n)
                e[k] = delta
                J[:, k] = (m(e) - f0v) / delta
            ev = np.sort(np.linalg.eigvals(J).real)
            dq = np.zeros(n)
            for _ in range(300):
                dq = 0.1 * m(dq) + 0.9 * dq
            r300 = np.abs(m(dq) - dq).max()
            print(
                f"[{tag}] |f(0)|={np.abs(f0v).max():.3f} "
                f"J(0) in [{ev.min():+.2f}, {ev.max():+.2f}] "
                f"simple300 res={r300:.3e}",
                flush=True,
            )
            ok, q = newton_hunt(m, n, tag)
            if ok:
                qa = np.zeros(n_atoms)
                np.add.at(qa, ctx["shell_atom"], -q)
                print(
                    f"[{tag}] q* atomic charges: "
                    f"{np.array2string(qa, precision=3)}"
                )


if __name__ == "__main__":
    sys.exit(main())
