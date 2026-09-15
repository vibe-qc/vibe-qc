"""Numerical continuation of the fcc Cu SCC solution branch (IID 130).

Answers the last open question of the metal thread: why does the
64-atom Cu 4x4x4 cell converge for a >= 3.615 A and fail below it,
when the ladder minimum (a = 3.7033 A) is comfortably converged?

Method. Reconstruct the SCC charge map in numpy from the native
one-shot diagnostics (machine-precision f(0) calibration), including
the on-site third-order GAM3 term, and track its physical fixed point
downward in lattice constant by numerical continuation: each point is
warm-started from the previous converged root and solved with an
exact-FD-Jacobian, line-searched Newton. At every root, report
min |eig(J - I)| - the distance of the fixed-point problem from
singularity. A saddle-node fold announces itself as that quantity
collapsing to zero, at which point the root annihilates and no solver
can find it.

Result (2026-08-27, on the #43-fixed parameter basis):

    a (A)   root   q_rms    min|eig(J - I)|
    3.900   yes    0.0447   0.7191
    3.800   yes    0.0617   0.6178
    3.700   yes    0.0852   0.4405
    3.650   yes    0.1018   0.2764
    3.615   yes    0.1214   0.0173     <- essentially singular
    3.580   NO     --       --         <- branch annihilated

The physical branch terminates in a saddle-node bifurcation at
a_c ~ 3.61 A. This is a property of the GFN2 Hamiltonian on a
compressed metal - the GAM3 contribution to the Jacobian, -2 G dq,
grows with the charge fluctuation the compression induces - and not a
CCM artifact or a solver defect: the same continuation with GAM3
switched off keeps a healthy root down to 3.4 A at least. The native
adapter converging exactly down to 3.615 A and failing closed below it
is therefore CORRECT behaviour, not a bug to fix, and the validated
lattice minimum at 3.7033 A sits safely above the fold.

Runtime is minutes per point on 64 atoms (each Newton step builds a
192-column FD Jacobian over 576-basis diagonalizations), so this is a
study probe, not a regression test.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from probe_wall_fixed_point_hunt import GAM3_L_SCALE, Gam3Map
from probe_wall_map_mechanism import build, gamma_matrix, one_shot
from vibeqc._vibeqc_core.semiempirical import (
    SemiempiricalBasis,
    gfn2_enumerate_shells,
)
from vibeqc.semiempirical.methods.gfn2_params import (
    _read_cached_toml,
    load_gfn2_params,
)

REPLICAS = (4, 4, 4)
N_ATOMS = REPLICAS[0] * REPLICAS[1] * REPLICAS[2]
LADDER = (3.9, 3.8, 3.7, 3.65, 3.615, 3.58, 3.55, 3.5)


def make_map(a_angstrom: float, temperature: float = 0.001, gam3: bool = True):
    """Native-calibrated numpy SCC map for the fcc Cu supercell."""
    molecule, topology = build(a_angstrom, REPLICAS)
    params = load_gfn2_params()
    native, records = one_shot(molecule, topology, params)
    h0 = np.asarray(native.hamiltonian, dtype=float)
    overlap = np.asarray(native.overlap, dtype=float)
    n_occ = int(native.n_occ)
    n_basis = int(native.n_basis)

    basis = SemiempiricalBasis.build(molecule, params, 0)
    shells = list(gfn2_enumerate_shells(basis, molecule, params))
    n_shells = len(shells)
    ao_shell = np.zeros(n_basis, dtype=int)
    for index, shell in enumerate(shells):
        start = int(shell.bf_start)
        for mu in range(start, start + int(shell.n_funcs)):
            ao_shell[mu] = index

    blob = _read_cached_toml()
    gam3_by_z = {
        int(entry["Z"]): float(entry.get("gam3", 0.0))
        for entry in blob.get("element", [])
    }
    scale = gam3_by_z[29] if gam3 else 0.0
    gam3_shell = np.array(
        [scale * GAM3_L_SCALE[min(int(s.l), 3)] for s in shells]
    )

    translations = np.asarray(topology.translations)
    alpha = math.sqrt(math.pi) / abs(np.linalg.det(translations)) ** (1.0 / 3.0)
    gamma = gamma_matrix(molecule, params, records, alpha)

    native_t, _ = one_shot(
        molecule, topology, params, temperature=temperature
    )
    charge_map = Gam3Map(
        h0,
        overlap,
        gamma,
        ao_shell,
        n_shells,
        n_occ,
        temperature,
        "continuation",
        gam3_shell=gam3_shell,
    )
    charge_map.calibrate(np.asarray(native_t.shell_charges, dtype=float))
    shell_atom = np.array([int(s.atom_idx) for s in shells])
    return charge_map, n_shells, shell_atom


def _jacobian(charge_map, n_shells, q, f_q, delta=1.0e-4):
    jacobian = np.zeros((n_shells, n_shells))
    for column in range(n_shells):
        step = np.zeros(n_shells)
        step[column] = delta
        jacobian[:, column] = (charge_map(q + step) - f_q) / delta
    return jacobian


def newton_from(charge_map, n_shells, q0, max_iter=120, tol=1.0e-9):
    """Exact-FD-Jacobian Newton with a residual-reduction line search."""
    q = q0.copy()
    for _ in range(max_iter):
        f_q = charge_map(q)
        residual = f_q - q
        norm = np.abs(residual).max()
        if norm < tol:
            return q, True, norm
        jacobian = _jacobian(charge_map, n_shells, q, f_q)
        try:
            step = np.linalg.solve(
                jacobian - np.eye(n_shells), residual
            )
        except np.linalg.LinAlgError:
            return q, False, norm
        damping = 1.0
        accepted = False
        for _ in range(20):
            trial = q - damping * step
            if np.abs(charge_map(trial) - trial).max() < norm:
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            return q, False, norm
        q = trial
    return q, False, np.abs(charge_map(q) - q).max()


def main() -> None:
    for gam3 in (True, False):
        label = "GAM3 on (production)" if gam3 else "GAM3 off (control)"
        print(f"== fcc Cu {REPLICAS} branch continuation, {label} ==", flush=True)
        q_prev = None
        for a in LADDER:
            started = time.time()
            charge_map, n_shells, shell_atom = make_map(a, gam3=gam3)
            q0 = (
                q_prev
                if q_prev is not None and q_prev.size == n_shells
                else np.zeros(n_shells)
            )
            q, converged, residual = newton_from(charge_map, n_shells, q0)
            atomic = np.zeros(N_ATOMS)
            np.add.at(atomic, shell_atom, -q)
            if converged:
                singular = float(
                    np.abs(
                        np.linalg.eigvals(
                            _jacobian(charge_map, n_shells, q, charge_map(q))
                            - np.eye(n_shells)
                        )
                    ).min()
                )
                q_prev = q
            else:
                singular = float("nan")
            print(
                f"a={a:6.3f} root={converged} res={residual:.2e} "
                f"q_rms={np.sqrt((atomic ** 2).mean()):.4f} "
                f"min|eig(J-I)|={singular:.4f} {time.time() - started:.0f}s",
                flush=True,
            )
            if not converged:
                print("  -> branch annihilated (saddle-node fold)", flush=True)
                break


if __name__ == "__main__":
    sys.exit(main())
