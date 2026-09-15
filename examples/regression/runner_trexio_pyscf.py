#!/usr/bin/env python3
"""Out-of-process TREXIO cross-check against PySCF (CLAUDE.md § 10).

Run this script in a separate process with ``trexio`` and ``pyscf``
installed, and never import it from ``python/vibeqc/``. It reads a TREXIO
file that vibe-qc wrote, rebuilds a PySCF ``Mole`` from the file's nucleus,
basis and optional ECP groups, and
then, with PySCF's *own* integrals:

1. checks that the stored MOs are orthonormal in PySCF's overlap,
   ``C S_pyscf C^T = 1`` (pins AO ordering, sign and normalization);
2. compares PySCF's overlap with the file's ``ao_1e_int.overlap`` when the
   group is present (pins the basis group itself);
3. rebuilds the SCF energy from the stored MOs and occupations,
   ``E = E_nn + tr(D h) + 1/2 [tr(D J) - tr(D_a K_a) - tr(D_b K_b)]``,
   and compares it with the file's ``state.energy`` (pins everything at
   once: a wrong permutation, phase or factor moves the energy by mHa).
   For a CI expansion, contracts its determinant coefficients with PySCF's
   FCI Hamiltonian instead, including frozen orbitals present in the file.

Conventions taken from the TREXIO specification (trex.org, v2.6.1) and
from PySCF, independent of vibe-qc's implementation:

* TREXIO orders real solid harmonics ``m = 0, +1, -1, +2, -2, ...``;
  PySCF orders ``p`` as ``(px, py, pz)`` and ``l >= 2`` as ``m = -l..+l``
  (the same mapping ``trexio-tools``' PySCF converter applies).
* The TREXIO radial function is ``N_s sum_k f_ks a_ks exp(-g r^2)``;
  PySCF's ``Mole`` takes raw contraction coefficients and normalizes each
  contracted function itself, so the file's contracted functions must
  already be unit-normalized for the two basis sets to coincide -- the
  script verifies that precondition from the file's own overlap diagonal
  (or, without an overlap group, assumes it and says so in the verdict).
* ``mo.coefficient`` is ``[mo.num, ao.num]``; ``mo.spin`` 0/1 splits UHF
  spin-orbital blocks.

Prints one machine-readable line ``VIBEQC-TREXIO-PYSCF-RESULT:{json}`` and
optionally writes the same JSON to ``--json``. Exit status 0 on a pass,
1 on a fail, 2 on a usage or read error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESULT_MARKER = "VIBEQC-TREXIO-PYSCF-RESULT:"

_SYMBOLS = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
)


def trexio_to_pyscf_perm(shell_ang_mom):
    """``perm[j]`` = PySCF AO index of TREXIO AO ``j`` (both spherical)."""
    perm = []
    offset = 0
    for L in shell_ang_mom:
        if L == 0:
            local = [0]
        elif L == 1:
            # TREXIO (pz, px, py) <- PySCF (px, py, pz)
            local = [2, 0, 1]
        else:
            # TREXIO (0, +1, -1, +2, -2, ...) <- PySCF (m = -L..+L)
            local = [L]
            for k in range(1, L + 1):
                local += [L + k, L - k]
        perm += [offset + i for i in local]
        offset += 2 * L + 1
    return np.asarray(perm, dtype=int)


def read_file(path: str):
    import trexio

    with trexio.File(path, mode="r", back_end=trexio.TREXIO_AUTO) as f:
        d = {}
        if trexio.has_pbc_periodic(f) and trexio.read_pbc_periodic(f):
            raise SystemExit("this reference runner requires a molecular wavefunction")
        if trexio.has_mo_coefficient_im(f) and np.any(trexio.read_mo_coefficient_im(f)):
            raise SystemExit("this reference runner requires real molecular orbitals")
        n_atoms = trexio.read_nucleus_num(f)
        d["charges"] = np.asarray(trexio.read_nucleus_charge(f), dtype=float)
        d["coords"] = np.asarray(trexio.read_nucleus_coord(f), dtype=float).reshape(n_atoms, 3)
        d["n_up"] = int(trexio.read_electron_up_num(f))
        d["n_dn"] = int(trexio.read_electron_dn_num(f))
        d["ecp"] = {}
        if trexio.has_ecp_num(f):
            for name in ("z_core", "max_ang_mom_plus_1", "nucleus_index", "ang_mom",
                         "power", "exponent", "coefficient"):
                d["ecp"][name] = np.asarray(getattr(trexio, "read_ecp_" + name)(f))
        if trexio.read_basis_type(f) != "Gaussian":
            raise SystemExit("only Gaussian basis groups are handled")
        d["shell_atom"] = [int(i) for i in trexio.read_basis_nucleus_index(f)]
        d["shell_l"] = [int(L) for L in trexio.read_basis_shell_ang_mom(f)]
        d["shell_factor"] = np.asarray(trexio.read_basis_shell_factor(f), dtype=float)
        d["prim_shell"] = [int(s) for s in trexio.read_basis_shell_index(f)]
        d["exponent"] = np.asarray(trexio.read_basis_exponent(f), dtype=float)
        d["coefficient"] = np.asarray(trexio.read_basis_coefficient(f), dtype=float)
        d["prim_factor"] = np.asarray(trexio.read_basis_prim_factor(f), dtype=float)
        if trexio.has_basis_r_power(f) and any(int(p) for p in trexio.read_basis_r_power(f)):
            raise SystemExit("r_power != 0 is not handled")
        if trexio.has_ao_cartesian(f) and int(trexio.read_ao_cartesian(f)) != 0:
            raise SystemExit("Cartesian AOs are not handled")
        d["ao_num"] = int(trexio.read_ao_num(f))
        d["ao_norm"] = (
            np.asarray(trexio.read_ao_normalization(f), dtype=float)
            if trexio.has_ao_normalization(f)
            else np.ones(d["ao_num"])
        )
        mo_num = int(trexio.read_mo_num(f))
        d["mo_coeff"] = np.asarray(trexio.read_mo_coefficient(f), dtype=float).reshape(
            mo_num, d["ao_num"]
        )
        d["mo_occ"] = np.asarray(trexio.read_mo_occupation(f), dtype=float)
        d["mo_spin"] = (
            np.asarray(trexio.read_mo_spin(f), dtype=int)
            if trexio.has_mo_spin(f)
            else np.zeros(mo_num, dtype=int)
        )
        d["overlap"] = (
            np.asarray(trexio.read_ao_1e_int_overlap(f), dtype=float).reshape(
                d["ao_num"], d["ao_num"]
            )
            if trexio.has_ao_1e_int_overlap(f)
            else None
        )
        d["hcore"] = (np.asarray(trexio.read_ao_1e_int_core_hamiltonian(f))
                      if trexio.has_ao_1e_int_core_hamiltonian(f) else None)
        d["energy"] = float(trexio.read_state_energy(f)) if trexio.has_state_energy(f) else None
        d["nuclear_repulsion"] = (
            float(trexio.read_nucleus_repulsion(f)) if trexio.has_nucleus_repulsion(f) else None
        )
        d["trexio_version"] = trexio.__version__
        d["determinants"] = None
        if trexio.has_determinant_list(f):
            n = trexio.read_determinant_num(f)
            d["determinants"] = trexio.read_determinant_list(f, 0, n)[0]
            d["ci_coefficients"] = trexio.read_determinant_coefficient(f, 0, n)[0]
    return d


def build_mole(d):
    """A PySCF Mole from the nucleus + basis groups, one basis entry per atom."""
    from pyscf import gto

    prim_shell = np.asarray(d["prim_shell"])
    atoms = []
    basis = {}
    potentials = {}
    core = d["ecp"].get("z_core", np.zeros(len(d["charges"])))
    for a, (z, xyz) in enumerate(zip(d["charges"] + core, d["coords"])):
        label = f"{_SYMBOLS[int(round(z))]}{a}"  # unique per centre
        atoms.append([label, tuple(float(x) for x in xyz)])
        shells = []
        for s, (atom, L) in enumerate(zip(d["shell_atom"], d["shell_l"])):
            if atom != a:
                continue
            sel = np.nonzero(prim_shell == s)[0]
            # The file's contracted radial function is N_s sum f a exp(-g r^2);
            # its unit normalization is checked separately, so PySCF's own
            # contraction normalization reproduces it from the a_ks alone.
            shells.append(
                [int(L)] + [[float(d["exponent"][k]), float(d["coefficient"][k])] for k in sel]
            )
        basis[label] = shells
        if d["ecp"]:
            ecp = d["ecp"]
            selected = np.flatnonzero(ecp["nucleus_index"] == a)
            channels = []
            # PySCF's public ECP input uses -1 for the local channel and
            # groups exponents/coefficients by the NWChem radial power n+2.
            # https://pyscf.org/pyscf_api_docs/pyscf.gto.html#pyscf.gto.mole.format_ecp
            for angular in sorted(set(ecp["ang_mom"][selected].tolist())):
                radial = [[] for _ in range(7)]
                for i in selected[ecp["ang_mom"][selected] == angular]:
                    power = int(ecp["power"][i]) + 2
                    if not 0 <= power < len(radial):
                        raise SystemExit("ECP radial power is outside this PySCF reference runner's range")
                    radial[power].append([float(ecp["exponent"][i]), float(ecp["coefficient"][i])])
                channels.append([-1 if angular == ecp["max_ang_mom_plus_1"][a] else int(angular), radial])
            if channels:
                potentials[label] = [int(core[a]), channels]
    n_el = d["n_up"] + d["n_dn"]
    charge = int(round(float(d["charges"].sum()))) - n_el
    spin = d["n_up"] - d["n_dn"]
    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = "Bohr"
    mol.basis = basis
    mol.ecp = potentials
    mol.charge = charge
    mol.spin = spin
    mol.cart = False
    mol.verbose = 0
    mol.build()
    return mol


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trexio_file")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--tol-energy", type=float, default=1e-8)
    parser.add_argument("--tol-overlap", type=float, default=1e-8)
    args = parser.parse_args(argv)

    try:
        d = read_file(args.trexio_file)
        mol = build_mole(d)
        import pyscf
        from pyscf import scf
    except SystemExit as exc:
        print(RESULT_MARKER + json.dumps({"verdict": "error", "reason": str(exc)}))
        return 2

    perm = trexio_to_pyscf_perm(d["shell_l"])
    if perm.shape[0] != d["ao_num"] or mol.nao != d["ao_num"]:
        print(RESULT_MARKER + json.dumps({"verdict": "error", "reason": "ao count mismatch"}))
        return 2

    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    if d["ecp"]:
        h += mol.intor("ECPscalar")
    hcore_diff = None
    if d["hcore"] is not None:
        h_file = np.empty_like(h)
        h_file[np.ix_(perm, perm)] = d["hcore"]
        hcore_diff = float(np.max(np.abs(h_file - h)))

    # Precondition for the basis construction: unit-normalized contracted AOs.
    ao_norm_ok = bool(np.allclose(d["ao_norm"], 1.0, atol=1e-12))
    diag_ok = True
    overlap_max_diff = None
    if d["overlap"] is not None:
        diag_ok = bool(np.allclose(np.diag(d["overlap"]), 1.0, atol=1e-10))
        S_file_pyscf_order = np.empty_like(S)
        S_file_pyscf_order[np.ix_(perm, perm)] = d["overlap"]
        overlap_max_diff = float(np.abs(S_file_pyscf_order - S).max())

    # MO blocks in PySCF AO order: C[:, i] is MO i.
    blocks = {}
    for spin in sorted(set(int(s) for s in d["mo_spin"])):
        sel = np.nonzero(d["mo_spin"] == spin)[0]
        C = np.empty((d["ao_num"], sel.shape[0]))
        C[perm, :] = d["mo_coeff"][sel, :].T
        blocks[spin] = (C, d["mo_occ"][sel])

    orth_err = 0.0
    for C, _ in blocks.values():
        orth_err = max(orth_err, float(np.abs(C.T @ S @ C - np.eye(C.shape[1])).max()))

    # Energy from the stored MOs and occupations with PySCF's integrals.
    if len(blocks) == 1:
        C, occ = blocks[0]
        dm_total = (C * occ) @ C.T
        # closed shell: D_a = D_b = D/2
        dm_a = dm_b = 0.5 * dm_total
    else:
        Ca, occa = blocks[0]
        Cb, occb = blocks[1]
        dm_a = (Ca * occa) @ Ca.T
        dm_b = (Cb * occb) @ Cb.T
        dm_total = dm_a + dm_b
    J, _ = scf.hf.get_jk(mol, dm_total, hermi=1)
    _, Ka = scf.hf.get_jk(mol, dm_a, hermi=1)
    _, Kb = scf.hf.get_jk(mol, dm_b, hermi=1)
    e_one = float(np.einsum("ij,ji->", dm_total, h))
    e_two = 0.5 * float(np.einsum("ij,ji->", dm_total, J) - np.einsum("ij,ji->", dm_a, Ka) - np.einsum("ij,ji->", dm_b, Kb))
    e_nuc = float(mol.energy_nuc())
    e_pyscf = e_nuc + e_one + e_two
    if d["determinants"] is not None:
        from pyscf import ao2mo, fci
        if set(blocks) != {0}:
            raise SystemExit("CI reference requires one common spatial MO basis")
        C = blocks[0][0]
        norb = C.shape[1]
        nelec = (d["n_up"], d["n_dn"])
        vector = np.zeros((fci.cistring.num_strings(norb, nelec[0]),
                           fci.cistring.num_strings(norb, nelec[1])))
        for determinant, coefficient in zip(d["determinants"], d["ci_coefficients"]):
            words = np.asarray(determinant).view(np.uint64).reshape(2, -1)
            bits = [sum(int(word) << (64 * j) for j, word in enumerate(channel)) for channel in words]
            addresses = [fci.cistring.str2addr(norb, count, bit) for count, bit in zip(nelec, bits)]
            vector[tuple(addresses)] = coefficient
        e_pyscf = float(fci.direct_spin1.energy(C.T @ h @ C, ao2mo.full(mol, C), vector, norb, nelec)) + e_nuc

    energy_diff = None if d["energy"] is None else float(abs(e_pyscf - d["energy"]))
    nuc_diff = (
        None if d["nuclear_repulsion"] is None else float(abs(e_nuc - d["nuclear_repulsion"]))
    )
    checks = {
        "ao_normalization_is_one": ao_norm_ok,
        "file_overlap_diagonal_is_one": diag_ok,
        "orthonormal_in_pyscf_overlap": orth_err < args.tol_overlap,
        "overlap_matches_pyscf": overlap_max_diff is None or overlap_max_diff < args.tol_overlap,
        "hamiltonian_matches_pyscf": hcore_diff is None or hcore_diff < args.tol_energy,
        "energy_matches_state_energy": energy_diff is not None and energy_diff < args.tol_energy,
    }
    verdict = {
        "file": str(Path(args.trexio_file)),
        "n_atoms": int(d["charges"].shape[0]),
        "n_ao": int(d["ao_num"]),
        "n_mo": int(d["mo_coeff"].shape[0]),
        "spin_blocks": len(blocks),
        "wavefunction": "CI" if d["determinants"] is not None else "SCF",
        "ecp_core_electrons": int(np.sum(d["ecp"].get("z_core", 0))),
        "max_orthonormality_error": orth_err,
        "overlap_max_diff": overlap_max_diff,
        "hcore_max_diff": hcore_diff,
        "energy_file": d["energy"],
        "energy_pyscf_from_mos": e_pyscf,
        "energy_diff": energy_diff,
        "nuclear_repulsion_diff": nuc_diff,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
        "pyscf_version": pyscf.__version__,
        "trexio_version": d["trexio_version"],
    }
    line = RESULT_MARKER + json.dumps(verdict, sort_keys=True)
    print(line, flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
