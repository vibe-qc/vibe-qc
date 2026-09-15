"""MgO Γ fixed-density component audit: BIPOLE vs PySCF (out-of-process).

The forensic tool that root-caused the BIPOLE ionic-crystal
absolute-energy bug on 2026-06-10 (handovers/HANDOVER_BIPOLE_PRODUCTION.md §0a,
late session): at PySCF's converged density, compare every BIPOLE
energy component against PySCF's, in both a Γ-locality projected
(home-cell-only) warm start and the full Bloch warm start.

Pre-fix findings it captured (driver state before the option (b)
Ewald-exchange-split landing, commit history 2026-06-10):
  * E_nn matches to µHa.
  * T / V_ne / J match PySCF (to cutoff truncation) with the
    UNPROJECTED density and deviate (−0.26 / +0.20 / −1.02 Ha) with
    the projected one.
  * K (legacy full-Coulomb direct builder): projected undercounts by
    0.95 Ha; Bloch input "overcounts ~3×" — that series is formally
    divergent with non-decaying P, the value was a cutoff artefact.
  * The spheropole (+3.0 Ha) has no PySCF counterpart — a double-count
    in vibe-qc's explicit-jellium gauge.

Post-fix expectation (Ewald exchange split active — the default at
3D Γ): the *bloch* column reproduces every PySCF component to cutoff
truncation — at cutoff 10: e_exchange −24.899 vs −24.935, e_total
−271.032 vs −271.050 (+17 mHa); the spheropole row prints "—". The
*projected* column remains a diagnostic of the pre-fix locality
convention and no longer sums to a meaningful total.

Stage 1 (PySCF, run in its own interpreter — never imported by
vibe-qc, CLAUDE.md §10):

    python mgo_component_audit.py pyscf /tmp/mgo_pyscf_gamma.npz

Stage 2 (vibe-qc):

    python mgo_component_audit.py vibeqc /tmp/mgo_pyscf_gamma.npz
"""

from __future__ import annotations

import sys

import numpy as np

A_ANG = 4.21
ANG2BOHR = 1.0 / 0.529177210903


def stage_pyscf(out_path: str) -> None:
    from pyscf.pbc import gto as pbc_gto
    from pyscf.pbc import scf as pbc_scf

    lattice_ang = (A_ANG / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    half = A_ANG / 2.0
    cell = pbc_gto.M(
        atom=f"Mg 0 0 0; O {half} {half} {half}",
        a=lattice_ang.tolist(),
        basis="sto-3g",
        unit="A",
        verbose=0,
    )
    mf = pbc_scf.RHF(cell).density_fit()
    mf.conv_tol = 1e-10
    e = mf.kernel()
    D = mf.make_rdm1()
    hcore = mf.get_hcore()
    vj, vk = mf.get_jk(dm=D)
    T = cell.pbc_intor("int1e_kin")
    np.savez(
        out_path,
        D=D,
        S=mf.get_ovlp(),
        e_total=e,
        e_nuc=float(mf.energy_nuc()),
        e1=float(np.einsum("ij,ji->", D, hcore).real),
        e_kin=float(np.einsum("ij,ji->", D, T).real),
        e_j=0.5 * float(np.einsum("ij,ji->", D, vj).real),
        e_k=-0.25 * float(np.einsum("ij,ji->", D, vk).real),
    )
    print(f"PySCF MgO Γ RHF: E = {e:.10f} → {out_path}")


def stage_vibeqc(npz_path: str, cutoff: float = 10.0) -> None:
    import warnings

    warnings.simplefilter("ignore")
    import vibeqc as vq
    from vibeqc._vibeqc_core import (
        InitialGuess,
        LatticeSumOptions,
        PeriodicRHFOptions,
        compute_overlap_lattice,
        monkhorst_pack,
    )
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    dat = np.load(npz_path)
    a = A_ANG * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    sysp = vq.PeriodicSystem(
        3, lattice,
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis

    # AO map: PySCF groups s-shells before p per atom
    # (Mg: 1s,2s,3s,2p,3p | O: 1s,2s,2p); libint uses declaration order
    # (Mg: 1s,2s,2p,3s,3p). p components x,y,z in both. Validated via
    # S(Γ) in the 2026-06-10 session.
    spec = [0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10, 11, 12, 13]
    P = np.zeros((nbf, nbf))
    for v_i, p_i in enumerate(spec):
        P[v_i, p_i] = 1.0
    D = P @ dat["D"] @ P.T

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = 1
    opts.use_diis = False
    opts.initial_guess = InitialGuess.SAD
    kmesh = monkhorst_pack(sysp, [1, 1, 1])

    # Warm-start template: the Ewald-exchange-split driver stores the
    # SCF density on a 2×-cutoff cell list (every P(b−a) difference the
    # builder traversal forms must be resolvable) — initial_density
    # blocks must match that list.
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 2.0 * cutoff
    lat.nuclear_cutoff_bohr = 2.0 * cutoff
    tmpl = compute_overlap_lattice(basis, sysp, lat)

    def run_with(blocks):
        r = run_pbc_bipole_rhf(
            sysp, basis, kmesh, opts,
            use_ewald_j_split=True, ewald_precision=1e-8,
            progress=False, initial_density=blocks,
        )
        return r.energy_components[0]

    home = [
        i for i in range(len(tmpl.cells))
        if (tmpl.cells[i].index == np.array([0, 0, 0])).all()
    ][0]
    proj = [np.zeros_like(D) for _ in range(len(tmpl.cells))]
    proj[home] = D.copy()
    bloch = [D.copy() for _ in range(len(tmpl.cells))]

    names = (
        "e_kinetic", "e_nuclear_attraction", "e_j_short_range",
        "e_j_long_range", "e_exchange", "e_ext_el_spheropole",
        "e_nuclear_repulsion", "e_total",
    )
    c_proj = run_with(proj)
    c_bloch = run_with(bloch)

    def _fmt(c, nm):
        v = getattr(c, nm)
        return f"{float(v):>14.6f}" if v is not None else f"{'—':>14s}"

    print(f"{'term':24s} {'projected':>14s} {'bloch':>14s}")
    for nm in names:
        print(f"{nm:24s} {_fmt(c_proj, nm)} {_fmt(c_bloch, nm)}")
    print("\nPySCF reference:")
    print(f"{'e_kin':24s} {float(dat['e_kin']):>14.6f}")
    print(f"{'e_ne':24s} {float(dat['e1']) - float(dat['e_kin']):>14.6f}")
    print(f"{'e_j':24s} {float(dat['e_j']):>14.6f}")
    print(f"{'e_k':24s} {float(dat['e_k']):>14.6f}")
    print(f"{'e_nuc':24s} {float(dat['e_nuc']):>14.6f}")
    print(f"{'e_total':24s} {float(dat['e_total']):>14.6f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "vibeqc"
    path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mgo_pyscf_gamma.npz"
    if mode == "pyscf":
        stage_pyscf(path)
    else:
        stage_vibeqc(path)
