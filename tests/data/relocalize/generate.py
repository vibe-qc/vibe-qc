"""Regenerate small protocol fixtures with this checkout's native core.

Run: .venv/bin/python tests/data/relocalize/generate.py
Test data only; no user job outputs are produced. Periodic inputs use native
Gaussian image-summed metrics and a synthetic occupied Hamiltonian, not SCF.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import vibeqc as vq
from vibeqc.output.formats.qvf import _basis_shell_payload
from vibeqc.relocalize import AO_CONVENTION, BLOCH_CONVENTION, _encode, localize

HERE = Path(__file__).parent


def write(name, payload):
    (HERE / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    positions = [[0.6, 0.0, 0.0], [2.0, 0.0, 0.0]]
    atoms = [vq.Atom(1, p) for p in positions]
    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(mol, "sto-3g")
    rhf = vq.run_rhf(mol, basis, vq.RHFOptions())
    assert rhf.converged
    request = {
        "protocol": "vibeqc.relocalize",
        "protocol_version": 1,
        "id": "h2",
        "operation": "localize",
        "method": "ibo",
        "system": {
            "kind": "molecular",
            "atomic_numbers": [1, 1],
            "positions_bohr": positions,
            "charge": 0,
            "multiplicity": 1,
            "spin": "restricted",
        },
        "basis": {
            "name": "sto-3g",
            "ao_convention": AO_CONVENTION,
            "uses_ecp": False,
            "shells": _basis_shell_payload(basis)[0],
        },
        "orbitals": {
            "coefficients": _encode(np.asarray(rhf.mo_coeffs)[:, :1].T),
            "occupations": [2.0],
        },
    }
    write("h2.request.json", request)
    write("h2.result.json", localize(request))
    lattice = np.diag([4.0, 10.0, 10.0])
    system = vq.PeriodicSystem(1, lattice, atoms)
    cells = vq.direct_lattice_cells(system, 12.0)
    s_lattice = vq.compute_overlap_lattice_explicit(basis, system, cells)
    for name, nk in [("periodic_gamma", 1), ("periodic_complex", 2)]:
        periodic = copy.deepcopy(request)
        periodic.update(id=name, method="aiccm-wannier", allow_experimental=True)
        periodic["system"]["kind"] = "periodic"
        ks = np.zeros((nk, 3))
        ks[:, 0] = np.arange(nk) / nk
        ss = np.array(
            [
                np.asarray(
                    vq.bloch_sum(s_lattice, 2 * np.pi * np.linalg.inv(lattice).T @ k)
                )
                for k in ks
            ]
        )
        cs = []
        for i, s in enumerate(ss):
            # The second k block has an intrinsic relative complex phase,
            # so dropping imaginary components changes the occupied projector.
            c = np.array([1, 1 if i == 0 else 0.4 + 0.8j], dtype=complex)
            c /= np.sqrt((c.conj() @ s @ c).real)
            cs.append(c[None])
        periodic["orbitals"] = {
            "coefficients": _encode(np.array(cs)),
            "occupations": [[2.0]] * nk,
        }
        periodic["periodic"] = {
            "lattice_bohr": lattice.tolist(),
            "dimension": 1,
            "pbc": [True, False, False],
            "mesh": [nk, 1, 1],
            "kpoints_fractional": ks.tolist(),
            "weights": [1 / nk] * nk,
            "overlap": _encode(ss),
            "representation": "finite_bvk",
            "bloch_convention": BLOCH_CONVENTION,
            "orbital_energies_hartree": [[-1.0, 0.5]] * nk,
        }
        write(name + ".request.json", periodic)
        write(name + ".result.json", localize(periodic))
    missing = copy.deepcopy(request)
    missing.pop("orbitals")
    write("missing_orbitals.request.json", missing)
    write(
        "missing_orbitals.error.json", {"code": "invalid_source", "field": "orbitals"}
    )
    unsupported = copy.deepcopy(periodic)
    unsupported["method"] = "ibo"
    write("periodic_ibo.request.json", unsupported)
    write(
        "periodic_ibo.error.json",
        {"code": "unsupported_periodic_method", "field": "method"},
    )


if __name__ == "__main__":
    main()
