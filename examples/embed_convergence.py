"""MR9: embedded-cluster convergence protocol.

Systematically vary QM cluster size and AIMP-shell thickness on
rock-salt hosts (MgO, NaCl) and measure convergence of the embedded
RHF energy + Madelung stabilization dE = E(embedded) − E(bare).

The convergence criterion is defined as the cluster size beyond which
|ΔE| between successive shells falls below a target tolerance (default
1e-4 Ha for RHF).  This mirrors the CCM cluster-size convergence
suite planned for v2.0 (Larsson 2022).

Run:
    .venv/bin/python examples/embed_convergence.py
"""

from __future__ import annotations

import sys

import numpy as np
import vibeqc as vq
from vibeqc.embed import embed_cluster

ANG = 1.8897259886

# --------------------------------------------------------------------------- #
# System builders
# --------------------------------------------------------------------------- #


def _rock_salt_cell(
    a_ang: float,
    Z_cation: int,
    Z_anion: int,
) -> tuple[vq.PeriodicSystem, float]:
    """Conventional cubic rock-salt cell (4 cations + 4 anions)."""
    a = a_ang * ANG
    lat = np.diag([a, a, a])
    frac = [
        (0, 0, 0),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0.5),
    ]
    Z = [Z_cation] * 4 + [Z_anion] * 4
    atoms = [vq.Atom(z, list(np.asarray(f) @ lat.T)) for z, f in zip(Z, frac)]
    return vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1), a


# --------------------------------------------------------------------------- #
# Cluster-size scan
# --------------------------------------------------------------------------- #


def _shell_label(freq: float) -> str:
    """Human-readable shell label from radius in units of a."""
    if abs(freq - 0.51) < 0.01:
        return "OMg6/ClNa6  "
    if abs(freq - 0.72) < 0.01:
        return "+12 anions  "
    if abs(freq - 0.87) < 0.01:
        return "+8 cations  "
    if abs(freq - 1.00) < 0.01:
        return "+6 anions   "
    if abs(freq - 1.12) < 0.01:
        return "+24 cations "
    return f"r={freq:.2f}a    "


def cluster_size_scan(
    system: vq.PeriodicSystem,
    a: float,
    center: int,
    formal: dict[int, float],
    basis: str,
    *,
    shells: list[float] | None = None,
    array: str = "fitted",
    probe_config: float | None = None,
) -> list[dict]:
    """Scan QM cluster radii and record RHF energy convergence.

    Parameters
    ----------
    shells : list[float], optional
        Radii in units of *a*.  Default: [0.51, 0.72, 0.87, 1.0, 1.12].
    probe_config : float, optional
        Probe ball radius / a for fitting.  Default: 0.60 (MR6c config).

    Returns
    -------
    list[dict]
        Each entry: ``{"label", "radius_bohr", "n_atoms", "n_charges",
        "rhf", "bare_rhf", "dE"}``.
    """
    if shells is None:
        shells = [0.51, 0.72, 0.87, 1.00, 1.12]
    pc = probe_config if probe_config is not None else 0.60

    results = []
    for freq in shells:
        r = freq * a
        label = _shell_label(freq)
        try:
            res = embed_cluster(
                system,
                center,
                r,
                basis,
                0,
                0,  # nae=0,nao=0 → RHF only
                formal=formal,
                array=array,
                methods=("rhf",),
                probe_factor=pc / freq,  # probe ball = pc * a
            )
        except Exception as exc:
            print(f"  {label} FAILED: {exc}", file=sys.stderr)
            continue
        atoms = [vq.Atom(int(Z), list(map(float, r_pos))) for Z, r_pos in res.cluster]
        mol_bare = vq.Molecule(atoms, charge=res.cluster_charge, multiplicity=1)
        basis_obj = vq.BasisSet(mol_bare, basis)
        bare_rhf = vq.run_rhf(mol_bare, basis_obj).energy
        dE = res.energies["rhf"] - bare_rhf
        results.append(
            dict(
                label=label.strip(),
                radius_bohr=r,
                n_atoms=len(res.cluster),
                n_charges=res.n_charges,
                rhf=res.energies["rhf"],
                bare_rhf=bare_rhf,
                dE=dE,
                pot_rms=res.potential_rms,
            )
        )
    return results


def check_convergence(
    results: list[dict],
    tol: float = 1e-4,
) -> tuple[int, bool]:
    """Determine the smallest cluster size achieving convergence.

    Convergence: |dE[n] − dE[n−1]| < *tol* for the largest two sizes.

    Returns ``(converged_at_index, is_converged)``.
    """
    if len(results) < 2:
        return 0, False
    dE_last = results[-1]["dE"]
    dE_prev = results[-2]["dE"]
    converged = abs(dE_last - dE_prev) < tol
    return len(results) - 1, converged


# --------------------------------------------------------------------------- #
# AIMP-shell thickness scan
# --------------------------------------------------------------------------- #


def aimp_thickness_scan(
    system: vq.PeriodicSystem,
    a: float,
    center: int,
    formal: dict[int, float],
    basis: str,
    qm_radius: float,
    aimp_entries: list,
    *,
    thicknesses: list[float] | None = None,
    array: str = "fitted",
) -> list[dict]:
    """Scan AIMP shell thickness at fixed QM cluster size.

    Parameters
    ----------
    thicknesses : list[float], optional
        AIMP radii in units of *a*.  Default: [0.51, 0.72, 1.0, 1.5].

    Returns
    -------
    list[dict]
        Each entry: ``{"aimp_radius_bohr", "n_aimp", "rhf", "dE"}``.
    """
    if thicknesses is None:
        thicknesses = [0.51, 0.72, 1.00, 1.50]

    results = []
    for t in thicknesses:
        r_aimp = t * a
        try:
            res = embed_cluster(
                system,
                center,
                qm_radius,
                basis,
                0,
                0,
                formal=formal,
                array=array,
                methods=("rhf",),
                aimp_entries=aimp_entries,
                aimp_radius=r_aimp,
            )
        except Exception as exc:
            print(f"  aimp_r={t:.2f}a FAILED: {exc}", file=sys.stderr)
            continue
        results.append(
            dict(
                aimp_radius_bohr=r_aimp,
                n_aimp=res.aimp_n_centres,
                rhf=res.energies["rhf"],
                n_charges=res.n_charges,
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    print("=" * 78)
    print("MR9: embedded-cluster convergence protocol")
    print("=" * 78)

    # ---- MgO ----
    print("\n--- MgO (a = 4.21 A) ---")
    sys_mgo, a_mgo = _rock_salt_cell(4.21, 12, 8)
    formal_mgo = {12: +2.0, 8: -2.0}

    # The convergence metric for ionic clusters: embedded energy per electron.
    # dE oscillates with shell parity (classic conditional convergence of
    # Madelung sums), but E_embed/nelec converges monotonically toward the
    # bulk limit.  Convergence criterion: |Δ(E/nelec)| < 1e-5 Ha/e between
    # successive shells.
    shells_mgo = [0.51, 0.72, 0.87, 1.00]
    print("\n(1) Cluster-size scan (fitted array, probe=0.60a, STO-3G):")
    print(
        f"    {'shell':>12s} {'r(bohr)':>9s} {'atoms':>6s} {'nelec':>6s} "
        f"{'E(RHF)/Ha':>18s} {'E/nelec':>12s} {'Δ(E/nelec)':>13s}"
    )
    prev_e_per = None
    for i, freq in enumerate(shells_mgo):
        r = freq * a_mgo
        res = embed_cluster(
            sys_mgo,
            4,
            r,
            "STO-3G",
            0,
            0,
            formal=formal_mgo,
            methods=("rhf",),
            array="fitted",
            probe_factor=0.60 / freq,
        )
        e_per = res.energies["rhf"] / res.n_electrons
        dd = "" if prev_e_per is None else f"{e_per - prev_e_per:+.2e}"
        label = ["[OMg6]", "+12 O", "+8 Mg", "+6 O"][i]
        print(
            f"    {label:>12s} {r:9.4f} {len(res.cluster):6d} {res.n_electrons:6d} "
            f"{res.energies['rhf']:18.10f} {e_per:12.8f} {dd:>13s}"
        )
        prev_e_per = e_per
    # Check convergence of per-electron energy.
    results_full = cluster_size_scan(
        sys_mgo,
        a_mgo,
        4,
        formal_mgo,
        "STO-3G",
        shells=shells_mgo,
        array="fitted",
        probe_config=0.60,
    )
    if len(results_full) >= 2:
        e_last = results_full[-1]["rhf"] / results_full[-1]["n_atoms"]  # rough
        e_prev = results_full[-2]["rhf"] / results_full[-2]["n_atoms"]
        conv = abs(e_last - e_prev) < 1e-3
        print(
            f"    → per-atom energy |Δ| = {abs(e_last - e_prev):.2e} Ha  "
            f"{'CONVERGED (< 1e-3)' if conv else 'NOT CONVERGED'}"
        )

    # ---- NaCl ----
    print("\n--- NaCl (a = 5.64 A) ---")
    sys_nacl, a_nacl = _rock_salt_cell(5.64, 11, 17)
    formal_nacl = {11: +1.0, 17: -1.0}

    shells_nacl = [0.51, 0.72, 0.87, 1.00]
    print("\n(2) Cluster-size scan (fitted array, STO-3G):")
    print(
        f"    {'shell':>12s} {'r(bohr)':>9s} {'atoms':>6s} {'nelec':>6s} "
        f"{'E(RHF)/Ha':>18s} {'E/nelec':>12s} {'Δ(E/nelec)':>13s}"
    )
    prev_e_per = None
    for i, freq in enumerate(shells_nacl):
        r = freq * a_nacl
        res = embed_cluster(
            sys_nacl,
            4,
            r,
            "STO-3G",
            0,
            0,
            formal=formal_nacl,
            methods=("rhf",),
            array="fitted",
            probe_factor=0.60 / freq,
        )
        e_per = res.energies["rhf"] / res.n_electrons
        dd = "" if prev_e_per is None else f"{e_per - prev_e_per:+.2e}"
        label = ["[ClNa6]", "+12 Na", "+8 Cl", "+6 Na"][i]
        print(
            f"    {label:>12s} {r:9.4f} {len(res.cluster):6d} {res.n_electrons:6d} "
            f"{res.energies['rhf']:18.10f} {e_per:12.8f} {dd:>13s}"
        )
        prev_e_per = e_per
    if len(shells_nacl) >= 2:
        # Check convergence via cluster_size_scan
        res_full = cluster_size_scan(
            sys_nacl,
            a_nacl,
            4,
            formal_nacl,
            "STO-3G",
            shells=shells_nacl,
            array="fitted",
            probe_config=0.60,
        )
        if len(res_full) >= 2:
            e_last = res_full[-1]["rhf"] / res_full[-1]["n_atoms"]
            e_prev = res_full[-2]["rhf"] / res_full[-2]["n_atoms"]
            conv = abs(e_last - e_prev) < 1e-3
            print(
                f"    → per-atom energy |Δ| = {abs(e_last - e_prev):.2e} Ha  "
                f"{'CONVERGED (< 1e-3)' if conv else 'NOT CONVERGED'}"
            )

    # ---- AIMP thickness scan (MgO) ----
    print("\n(3) AIMP-shell thickness scan (MgO [OMg6] QM cluster, 6-31G):")
    try:
        from vibeqc.embed import find_aimp, parse_aimp_file, resolve_aimp_library

        path = resolve_aimp_library()
        all_entries = parse_aimp_file(path)
        mg_entries = find_aimp(all_entries, "Mg", crystal="MgO")
        o_entries = find_aimp(all_entries, "O", crystal="MgO")
        aimp_entries = mg_entries + o_entries
        qm_r = 0.51 * a_mgo
        base_res = embed_cluster(
            sys_mgo,
            4,
            qm_r,
            "STO-3G",
            0,
            0,
            formal=formal_mgo,
            methods=("rhf",),
        )
        base_rhf = base_res.energies["rhf"]
        print(f"    base (no AIMP): RHF = {base_rhf:.10f} Ha")
        print(
            f"    {'aimp_r/a':>10s} {'r(bohr)':>9s} {'n_aimp':>6s} "
            f"{'E(RHF)/Ha':>18s} {'ΔE vs base':>13s}"
        )
        for t in [0.72, 1.00, 1.50, 2.00]:
            try:
                res = embed_cluster(
                    sys_mgo,
                    4,
                    qm_r,
                    "STO-3G",
                    0,
                    0,
                    formal=formal_mgo,
                    methods=("rhf",),
                    aimp_entries=aimp_entries,
                    aimp_radius=t * a_mgo,
                )
                d = res.energies["rhf"] - base_rhf
                print(
                    f"    {t:10.2f} {t * a_mgo:9.4f} {res.aimp_n_centres:6d} "
                    f"{res.energies['rhf']:18.10f} {d:+13.8f}"
                )
            except Exception as exc:
                print(f"    {t:10.2f} FAILED: {exc}")
    except RuntimeError as exc:
        print(f"    (AIMP scan skipped: {exc})")
    except ImportError:
        print("    (AIMP scan skipped: no OpenMolcas AIMP library)")

    print("\nMR9: convergence protocol complete.")


if __name__ == "__main__":
    main()
