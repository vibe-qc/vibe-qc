"""READ initial guess -- restart an SCF from a prior calculation's orbitals.

The READ guess (``InitialGuess.READ``, or the input aliases ``"read"`` /
``"moread"`` / ``"coread"``) starts the SCF from a density built out of a
*prior* calculation rather than a from-scratch atomic guess. The prior
orbitals come from one of three sources:

* an **in-memory result** -- a previous ``RHFResult`` / ``UHFResult`` /
  ``RKSResult`` / ``UKSResult`` object, passed as ``run_*(..., read_from=prev)``;
* a vibe-qc **``.qvf`` file** (its ``wavefunction.gto`` section for
  molecular / Γ-point restarts, or its ``x_vibeqc.bloch_wavefunction``
  section for all-k periodic restarts), via
  ``opts.read_path = "prev.qvf"``;
* a **Molden ``.molden`` or ORCA ``.molden.input`` file**, via
  ``opts.read_path = "prev.molden"``.
  Both raw Gaussian contraction coefficients and the primitive-normalized
  convention emitted by exporters such as ORCA's ``orca_2mkl`` are detected
  from the MO orthonormality condition.

The prior density is expressed in the *current* AO basis. When the prior
geometry and basis match the current ones the density is used directly;
otherwise it is projected onto the current basis with the same least-squares
AO projector MINAO uses,

    D_current = P . D_prior . Pᵀ ,    P = S_tt⁻¹ . S_tm ,

where ``S_tt`` is the current overlap and ``S_tm = <current AO | prior AO>``
the cross-basis overlap. Projecting the *density matrix* (rather than the
orbitals) needs no occupation bookkeeping, so a single path covers
RHF / UHF / RKS / UKS. Native and Python SCF results carry their source basis, as do the file
sources, so geometry changes use the same explicit projection contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .spin_channels import is_open_shell_result

from ._primitive_norm import libint_primitive_norm as _primitive_normalisation
from ._vibeqc_core import (
    Atom,
    BasisSet,
    Molecule,
    ShellInfo,
    compute_overlap,
    compute_overlap_two_basis,
)

_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
_SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
_QVF_BLOCH_WF_KIND = "x_vibeqc.bloch_wavefunction"
_MOLDEN_ORTHONORMALITY_TOL = 1.0e-5


def _real_density(value) -> np.ndarray:
    density = np.asarray(value)
    if density.ndim != 2 or density.shape[0] != density.shape[1] or not np.all(np.isfinite(density)):
        raise ValueError("READ: density must be finite and square")
    if np.iscomplexobj(density) and np.max(np.abs(density.imag), initial=0) > 1e-10:
        raise ValueError("READ: molecular/Gamma density must be real; use an all-k restart for complex data")
    density = np.asarray(density.real, dtype=float)
    if not np.allclose(density, density.T, rtol=0, atol=1e-8):
        raise ValueError("READ: density must be Hermitian")
    density = 0.5 * (density + density.T)
    if density.size and np.linalg.eigvalsh(density).min() < -1e-8:
        raise ValueError("READ: density must be positive semidefinite")
    return density


def _validate_mo_payload(coefficients, occupations, max_occupation):
    if (coefficients.ndim != 2 or occupations.ndim != 1
        or coefficients.shape[1] != occupations.size
        or not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(occupations))
        or np.any(occupations < -1e-12) or np.any(occupations > max_occupation + 1e-12)):
        raise ValueError("READ: invalid coefficients or occupations outside the spin capacity")


class _Prior:
    """Prior-calculation state in its *own* basis: per-spin densities plus the
    basis they live in. Legacy external result-like objects may omit it."""

    __slots__ = ("d_alpha", "d_beta", "basis")

    def __init__(self, d_alpha, d_beta, basis):
        self.d_alpha = _real_density(d_alpha)
        self.d_beta = _real_density(d_beta)
        self.basis = basis


# ---- in-memory result ----------------------------------------------------


def _is_unrestricted_result(obj) -> bool:
    # Value-based, not hasattr-based: the CCM runner adapters declare both
    # spin fields and leave them None on a closed-shell run, so an attribute
    # test would route a closed-shell restart source down the spin branch and
    # hand ``None`` to ``gamma_density``.  See :mod:`vibeqc.spin_channels`.
    return is_open_shell_result(obj)


def _is_restricted_result(obj) -> bool:
    return hasattr(obj, "density") and not _is_unrestricted_result(obj)


def _prior_from_result(src) -> _Prior:
    points = getattr(src, "restart_kpoints", None)
    if points is None:
        points = getattr(src, "kpoints_cart", None)
    if points is None:
        mesh = getattr(src, "kmesh", None)
        points = getattr(mesh, "kpoints", getattr(mesh, "kpoints_cart", None))
    lattice = getattr(src, "restart_lattice", None)
    is_gamma = (_is_gamma_restart(points, lattice) if lattice is not None else
                points is not None and np.asarray(points).shape == (1, 3)
                and np.allclose(points, 0, rtol=0, atol=1e-10))
    if points is not None and not is_gamma:
        raise ValueError("READ: a non-Gamma source needs a compatible all-k reader")

    def gamma_density(value):
        if hasattr(value, "cells") or np.asarray(value).ndim == 3:
            if not is_gamma:
                raise ValueError("READ: a lattice/per-k source needs its Gamma snapshot or the all-k reader")
            if hasattr(value, "cells"):
                from .pbc_bipole_common import home_cell_block
                return _real_density(home_cell_block(value))
            if len(value) != 1:
                raise ValueError("READ: Gamma source must contain one density block")
            return _real_density(value[0])
        return _real_density(value)
    if _is_unrestricted_result(src):
        return _Prior(gamma_density(src.density_alpha), gamma_density(src.density_beta),
                      getattr(src, "restart_basis", None))
    if _is_restricted_result(src):
        half = 0.5 * gamma_density(src.density)
        return _Prior(half, half, getattr(src, "restart_basis", None))
    raise TypeError(
        "READ source must be a prior SCF result (with a `density` or "
        "`density_alpha`/`density_beta` attribute) or a .qvf / Molden path; "
        f"got {type(src).__name__}."
    )


# ---- density assembly from MO coefficients + occupations -----------------


def _density_from_mos(C: np.ndarray, occ: np.ndarray, max_occupation=2) -> np.ndarray:
    """D = C diag(occ) Cᵀ, with C shaped (n_ao, n_mo)."""
    C = np.asarray(C)
    if not np.all(np.isfinite(C)):
        raise ValueError("READ: coefficients must be finite")
    if np.iscomplexobj(C) and np.max(np.abs(C.imag), initial=0) > 1e-10:
        raise ValueError("READ: molecular coefficients must be real")
    C = np.asarray(C.real, dtype=float)
    occ = np.asarray(occ)
    if np.iscomplexobj(occ):
        raise ValueError("READ: occupations must be real")
    occ = np.asarray(occ, dtype=float)
    _validate_mo_payload(C, occ, max_occupation)
    return (C * occ[None, :]) @ C.T


def _density_from_complex_mos(C: np.ndarray, occ: np.ndarray, max_occupation=2) -> np.ndarray:
    """D = C diag(occ) C†, with C shaped (n_ao, n_mo)."""
    C = np.asarray(C, dtype=complex)
    occ = np.asarray(occ)
    if np.iscomplexobj(occ):
        raise ValueError("READ: occupations must be real")
    occ = np.asarray(occ, dtype=float)
    _validate_mo_payload(C, occ, max_occupation)
    return (C * occ[None, :]) @ C.conj().T


# ---- QVF (.qvf) -----------------------------------------------------------


def _valid_molecule(atoms) -> Molecule:
    """Build a Molecule with a consistent (charge=0) multiplicity. Only the
    atom positions + Z matter for reconstructing the prior *basis*; the
    multiplicity is irrelevant to the AO set, so we pick the lowest valid one
    for the electron-count parity rather than tracking the prior spin state."""
    n_elec = sum(int(a.Z) for a in atoms)
    multiplicity = 1 if n_elec % 2 == 0 else 2
    return Molecule(atoms, multiplicity=multiplicity)


def _molecule_from_atoms_angstrom(symbols_z, positions_ang) -> Molecule:
    atoms = [
        Atom(int(z), [float(p[0]) * _BOHR_PER_ANGSTROM,
                      float(p[1]) * _BOHR_PER_ANGSTROM,
                      float(p[2]) * _BOHR_PER_ANGSTROM])
        for z, p in zip(symbols_z, positions_ang)
    ]
    return _valid_molecule(atoms)


def _basis_from_qvf_shells(prior_mol, shells_json, pure_top) -> BasisSet:
    """Reconstruct a BasisSet from a QVF ``basis.json`` shell list. The stored
    contraction coefficients are de-normalised (raw basis-file values), so the
    BasisSet is built with ``coefficients_pre_normalized=False``."""
    atom_xyz = [list(a.xyz) for a in prior_mol.atoms]
    shells: List[ShellInfo] = []
    for sh in shells_json:
        center = int(sh["center"])
        if center < 0 or center >= len(atom_xyz):
            raise ValueError("READ: QVF shell center is outside the source atom list")
        shells.append(ShellInfo(
            atom_index=center,
            l=int(sh["l"]),
            pure=bool(sh.get("pure", pure_top)),
            exponents=[float(x) for x in sh["exponents"]],
            coefficients=[float(c) for c in sh["coefficients"]],
            origin=atom_xyz[center],
        ))
    return BasisSet(prior_mol, shells, "qvf_read",
                    coefficients_pre_normalized=False)


def _prior_basis_from_qvf(
    zf: zipfile.ZipFile,
    manifest: dict,
    sections: dict,
    basis_json: dict,
):
    # The archive's shell data is the source of truth. A name may now refer
    # to a different/custom basis and cannot replace malformed archived data.
    structure = sections.get("structure")
    if structure is None:
        raise ValueError("READ: QVF restart requires its source structure and basis shells")
    struct = json.loads(zf.read(structure["members"]["structure"]["path"]).decode("utf-8"))
    zs = [int(atom["atomic_number"]) for atom in struct["atoms"]]
    xyz = [atom["position"] for atom in struct["atoms"]]
    prior_mol = _molecule_from_atoms_angstrom(zs, xyz)
    if not basis_json.get("shells"):
        raise ValueError("READ: QVF restart is missing its source basis shells")
    return _basis_from_qvf_shells(
        prior_mol, basis_json["shells"], bool(basis_json.get("pure", True))
    )


def _prior_from_qvf_bloch_gamma(
    zf: zipfile.ZipFile,
    path: str,
    manifest: dict,
    sections: dict,
    wf: dict,
) -> _Prior:
    members = wf["members"]
    meta = _qvf_read_json(zf, members, "mo_metadata")
    if meta.get("spin") not in ("restricted", "unrestricted"):
        raise ValueError("READ: invalid QVF Bloch spin metadata")
    blocks = meta.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 1:
        raise ValueError(
            f"READ: {path} has {0 if not blocks else len(blocks)} Bloch k blocks; "
            "Gamma-only READ requires one Gamma point. Use a compatible "
            "all-k READ request for this archive."
        )
    if meta.get("spin") == "restricted" and meta.get("coefficient_encoding") != "complex_split_last_axis":
        raise ValueError(
            "READ: QVF Bloch restart payload must store coefficients with "
            "coefficient_encoding='complex_split_last_axis'."
        )

    block = blocks[0]
    fractional = np.asarray(block.get("k_point"), dtype=float)
    if (fractional.shape != (3,) or not np.all(np.isfinite(fractional))
        or not np.allclose(fractional - np.rint(fractional), 0, rtol=0, atol=1e-10)):
        raise NotImplementedError("READ: a non-Gamma Bloch source requires a compatible all-k restart")
    if meta["spin"] == "unrestricted":
        basis_json = _qvf_read_json(zf, members, "basis")
        basis = _prior_basis_from_qvf(zf, manifest, sections, basis_json)
        pair = _spin_density_k_from_qvf_bloch_payload(path, expected_n_k=1, n_basis=basis.nbasis)
        return _Prior(_real_density(pair[0][0]), _real_density(pair[1][0]), basis)
    coeff_name = str(block.get("mo_coefficients", "mo_coefficients_k0"))
    occ_name = str(block.get("occupations", "occupations_k0"))
    coeff_rows = _complex_from_split_last_axis(
        _qvf_read_binary(zf, members, coeff_name),
        label="QVF Gamma Bloch coefficient block",
    )
    if coeff_rows.ndim != 2:
        raise ValueError(
            "READ: QVF Gamma Bloch coefficient block must decode to a "
            f"2-D [n_mo, n_ao] matrix; got shape {coeff_rows.shape}."
        )
    C = coeff_rows.T
    occ = np.asarray(_qvf_read_binary(zf, members, occ_name), dtype=float)
    if occ.ndim != 1 or C.shape[1] != occ.shape[0]:
        raise ValueError(
            "READ: QVF Gamma Bloch occupations do not match coefficient shape."
        )

    total = _density_from_complex_mos(C, occ)
    total = 0.5 * (total + total.conj().T)
    if total.size:
        max_imag = float(np.max(np.abs(total.imag)))
        max_real = float(np.max(np.abs(total.real)))
    else:
        max_imag = max_real = 0.0
    if max_imag > max(1.0e-10, 1.0e-8 * max(max_real, 1.0)):
        raise ValueError(
            "READ: QVF Gamma Bloch payload produced a complex density; "
            "use a multi-k READ request for non-Gamma k-points."
        )
    half = 0.5 * total.real
    basis_json = _qvf_read_json(zf, members, "basis")
    prior_basis = _prior_basis_from_qvf(zf, manifest, sections, basis_json)
    return _Prior(half, half, prior_basis)


def _prior_from_qvf(path: str) -> _Prior:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        sections = {s.get("kind"): s for s in manifest.get("sections", [])}
        bloch_wf = sections.get(_QVF_BLOCH_WF_KIND)
        if bloch_wf is not None:
            return _prior_from_qvf_bloch_gamma(
                zf, path, manifest, sections, bloch_wf
            )
        wf = sections.get("wavefunction.gto")
        if wf is None:
            raise ValueError(
                f"READ: {path} has no 'wavefunction.gto' section (no MO "
                "coefficients to restart from). Re-run the source job with "
                "MO output enabled.")
        members = wf["members"]

        def _read_json(role):
            return json.loads(zf.read(members[role]["path"]).decode("utf-8"))

        def _read_bin(role):
            spec = members[role]
            raw = zf.read(spec["path"])
            arr = np.frombuffer(raw, dtype=np.dtype(spec.get("dtype", "float64")))
            return arr.reshape(tuple(spec["shape"]))

        meta = _read_json("mo_metadata")
        fractional = np.asarray(meta.get("k_point", [0, 0, 0]), dtype=float)
        if (fractional.shape != (3,) or not np.all(np.isfinite(fractional))
            or not np.allclose(fractional - np.rint(fractional), 0, rtol=0, atol=1e-10)):
            raise NotImplementedError("READ: selected non-Gamma QVF orbitals do not define a Gamma restart")
        basis_json = _read_json("basis")

        prior_basis = _prior_basis_from_qvf(zf, manifest, sections, basis_json)

        spin = meta.get("spin", "restricted")
        if spin not in ("restricted", "unrestricted"):
            raise ValueError("READ: QVF spin metadata must be restricted or unrestricted")
        if spin == "unrestricted":
            Ca = _read_bin("mo_coefficients_alpha").T  # (n_ao, n_mo)
            Cb = _read_bin("mo_coefficients_beta").T
            occ_a = np.asarray(meta["alpha"]["occupations"], dtype=float)
            occ_b = np.asarray(meta["beta"]["occupations"], dtype=float)
            return _Prior(_density_from_mos(Ca, occ_a, 1),
                          _density_from_mos(Cb, occ_b, 1), prior_basis)
        C = _read_bin("mo_coefficients").T  # (n_ao, n_mo), libint AO order
        occ = np.asarray(meta["occupations"], dtype=float)
        half = 0.5 * _density_from_mos(C, occ)  # per-spin (restricted)
        return _Prior(half, half, prior_basis)


# ---- Molden (.molden) -----------------------------------------------------


def _prior_from_molden(path: str) -> _Prior:
    from .output.formats.molden import _build_ao_permutation

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    atoms_z: List[int] = []
    atoms_xyz: List[List[float]] = []      # bohr
    shell_specs: List[Tuple[int, int, List[float], List[float]]] = []
    mo_spin: List[str] = []
    mo_occ: List[float] = []
    mo_occ_seen: List[bool] = []
    mo_cols: List[List[float]] = []
    mo_indices: List[List[int]] = []

    section = None
    atom_unit_bohr = True
    cur_atom = -1
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i].strip()
        i += 1
        if not raw:
            continue
        low = raw.lower()
        if raw.startswith("["):
            if low.startswith("[atoms]"):
                section = "atoms"
                atom_unit_bohr = not ("angs" in low)
            elif low.startswith("[gto]"):
                section = "gto"
            elif low.startswith("[mo]"):
                section = "mo"
            elif low.startswith("[5d]") or low.startswith("[7f]") \
                    or low.startswith("[9g]") or low.startswith("[5d7f]"):
                section = section  # spherical-harmonic flags; vibe-qc is pure
            else:
                section = "other"
            continue

        if section == "atoms":
            parts = raw.split()
            # sym idx Z x y z
            z = int(parts[2])
            xyz = [float(parts[3]), float(parts[4]), float(parts[5])]
            if not atom_unit_bohr:
                xyz = [c * _BOHR_PER_ANGSTROM for c in xyz]
            atoms_z.append(z)
            atoms_xyz.append(xyz)
        elif section == "gto":
            parts = raw.split()
            if len(parts) == 2 and parts[1] == "0":
                # "<atom_index> 0" header
                cur_atom = int(parts[0]) - 1
            elif parts[0].lower() in _SHELL_L:
                l = _SHELL_L[parts[0].lower()]
                nprim = int(parts[1])
                exps: List[float] = []
                coefs: List[float] = []
                for _ in range(nprim):
                    a, c = lines[i].split()[:2]
                    i += 1
                    exps.append(float(a.replace("D", "E").replace("d", "E")))
                    coefs.append(float(c.replace("D", "E").replace("d", "E")))
                shell_specs.append((cur_atom, l, exps, coefs))
        elif section == "mo":
            if low.startswith("sym="):
                mo_spin.append("Alpha")
                mo_occ.append(0.0)
                mo_occ_seen.append(False)
                mo_cols.append([])
                mo_indices.append([])
            elif low.startswith("ene="):
                pass
            elif low.startswith("spin="):
                mo_spin[-1] = raw.split("=", 1)[1].strip()
            elif low.startswith("occup="):
                mo_occ[-1] = float(raw.split("=", 1)[1])
                mo_occ_seen[-1] = True
            else:
                # "<ao_index> <coeff>"
                parts = raw.split()
                if len(parts) >= 2:
                    mo_indices[-1].append(int(parts[0]))
                    mo_cols[-1].append(
                        float(parts[1].replace("D", "E").replace("d", "E")))

    # Reconstruct the prior geometry. Molden exporters disagree on whether
    # [GTO] stores raw Gaussian contraction coefficients (vibe-qc and many
    # other writers) or the primitive-normalized values used internally
    # (ORCA's orca_2mkl). Treating the latter as raw double-normalizes the
    # basis and silently changes Tr(D S); tetrazine/cc-pVDZ measured 42 ->
    # 32.79 electrons (issue 12).
    prior_mol = _valid_molecule(
        [Atom(z, xyz) for z, xyz in zip(atoms_z, atoms_xyz)])
    shells = [
        ShellInfo(atom_index=ai, l=l, pure=True, exponents=exps,
                  coefficients=coefs, origin=atoms_xyz[ai])
        for (ai, l, exps, coefs) in shell_specs
    ]
    nbf = sum(2 * l + 1 for _, l, _, _ in shell_specs)
    expected_indices = list(range(1, nbf + 1))
    for orbital, (occupation, occupation_seen, indices, coefficients) in enumerate(
        zip(mo_occ, mo_occ_seen, mo_indices, mo_cols), start=1
    ):
        if not occupation_seen:
            raise ValueError(
                f"READ: Molden MO {orbital} is missing its Occup field."
            )
        capacity = 1.0 if any(s.lower().startswith("b") for s in mo_spin) else 2.0
        if not np.isfinite(occupation) or not -1.0e-12 <= occupation <= capacity + 1.0e-12:
            raise ValueError(
                f"READ: Molden MO {orbital} has invalid occupation "
                f"{occupation!r}."
            )
        if not all(np.isfinite(coefficient) for coefficient in coefficients):
            raise ValueError(
                f"READ: Molden MO {orbital} contains a non-finite MO "
                "coefficient."
            )
        complete = (
            len(coefficients) == nbf
            and sorted(indices) == expected_indices
        )
        if occupation > 1.0e-12 and not complete:
            raise ValueError(
                f"READ: occupied Molden MO {orbital} has "
                f"{len(coefficients)} AO coefficients; expected exactly "
                f"{nbf}, indexed once each from 1 through {nbf}. Refusing "
                "an incomplete starting density."
            )
        if complete and indices != expected_indices:
            ordered = [0.0] * nbf
            for ao_index, coefficient in zip(indices, coefficients):
                ordered[ao_index - 1] = coefficient
            mo_cols[orbital - 1] = ordered

    def _candidate(primitive_normalized: bool):
        candidate_shells = shells
        if primitive_normalized:
            # Convert the exporter's internal primitive-normalized values
            # back to the raw Gaussian coefficients accepted by BasisSet.
            # This matches the Molden writer's de-normalization; BasisSet
            # then reapplies the primitive normalization on construction.
            candidate_shells = [
                ShellInfo(
                    atom_index=ai,
                    l=l,
                    pure=True,
                    exponents=exps,
                    coefficients=[
                        coefficient / _primitive_normalisation(exponent, l)
                        for exponent, coefficient in zip(exps, coefs)
                    ],
                    origin=atoms_xyz[ai],
                )
                for ai, l, exps, coefs in shell_specs
            ]
        return BasisSet(
            prior_mol,
            candidate_shells,
            "molden_read",
            coefficients_pre_normalized=False,
        )

    def _orthonormality_error(candidate) -> float:
        """Return max |Cocc.T S Cocc - I| for complete occupied MOs."""
        ao_perm_candidate = _build_ao_permutation(candidate)
        inv_candidate = np.argsort(ao_perm_candidate)
        nbf_candidate = candidate.nbasis
        overlap = np.asarray(compute_overlap(candidate), dtype=float)
        errors: List[float] = []
        spin_labels = ("Alpha", "Beta") if any(
            spin.lower().startswith("b") for spin in mo_spin
        ) else ("Alpha",)
        for label in spin_labels:
            cols = [
                col for spin, occupation, col in zip(mo_spin, mo_occ, mo_cols)
                if spin.lower().startswith(label[0].lower())
                and occupation > 1.0e-12
                and len(col) == nbf_candidate
            ]
            if not cols:
                continue
            coefficients = np.asarray(cols, dtype=float).T[inv_candidate, :]
            gram = coefficients.T @ overlap @ coefficients
            errors.append(float(np.max(np.abs(
                gram - np.eye(gram.shape[0], dtype=float)
            ))))
        if not errors or not all(np.isfinite(error) for error in errors):
            return float("inf")
        return max(errors)

    # Infer the producer convention from C.T S C = I instead of a brittle
    # title/program-name check. Ties retain the historical raw-coefficient
    # interpretation. The two measured regimes are widely separated:
    # O(1e-11) vs O(1) for both measured vibe-qc and ORCA outputs.
    raw_basis = _candidate(False)
    normalized_basis = _candidate(True)
    raw_error = _orthonormality_error(raw_basis)
    normalized_error = _orthonormality_error(normalized_basis)
    best_error = min(raw_error, normalized_error)
    if not np.isfinite(best_error) or best_error > _MOLDEN_ORTHONORMALITY_TOL:
        raise ValueError(
            "READ: Molden orbitals are not orthonormal under either the raw "
            "or primitive-normalized [GTO] contraction convention "
            f"(max |C.T S C - I|: raw={raw_error:.3e}, "
            f"primitive-normalized={normalized_error:.3e}; tolerance "
            f"{_MOLDEN_ORTHONORMALITY_TOL:.1e}). Refusing a corrupted "
            "starting density."
        )
    prior_basis = (
        normalized_basis if normalized_error < raw_error else raw_basis
    )

    # Molden lists AOs in molden order; permute back to libint order so the
    # coefficients line up with `prior_basis`. _build_ao_permutation gives
    # ``C_molden = C_libint[ao_perm]``, so the inverse is argsort(ao_perm).
    ao_perm = _build_ao_permutation(prior_basis)
    inv = np.argsort(ao_perm)
    nbf = prior_basis.nbasis

    def _spin_density(spin_label: str) -> np.ndarray:
        cols, occs = [], []
        for s, occ, col in zip(mo_spin, mo_occ, mo_cols):
            if (s.lower().startswith(spin_label[0].lower())
                    and occ > 1.0e-12 and len(col) == nbf):
                cols.append(col)
                occs.append(occ)
        if not cols:
            return np.zeros((nbf, nbf))
        C_molden = np.asarray(cols, dtype=float).T   # (n_ao, n_mo), molden order
        C = C_molden[inv, :]                          # -> libint order
        return _density_from_mos(C, np.asarray(occs, dtype=float), 1)

    has_beta = any(s.lower().startswith("b") for s in mo_spin)
    if has_beta:
        return _Prior(_spin_density("Alpha"), _spin_density("Beta"), prior_basis)
    # Restricted molden: a single Alpha block with Occup 2.0 -> split evenly.
    cols, occs = [], []
    for occ, col in zip(mo_occ, mo_cols):
        if occ > 1.0e-12 and len(col) == nbf:
            cols.append(col)
            occs.append(occ)
    C = (np.asarray(cols, dtype=float).T[inv, :] if cols
         else np.zeros((nbf, 0)))
    half = 0.5 * _density_from_mos(C, np.asarray(occs, dtype=float))
    return _Prior(half, half, prior_basis)


# ---- source dispatch + projection ----------------------------------------


def _prior_state(options, read_from) -> _Prior:
    if read_from is not None and not isinstance(read_from, (str, os.PathLike)):
        return _prior_from_result(read_from)
    path = read_from if isinstance(read_from, (str, os.PathLike)) else getattr(options, "read_path", "") or ""
    if not path:
        raise ValueError(
            "initial_guess=READ needs a source: pass run_*(..., "
            "read_from=prior_result) for an in-memory restart, or set "
            "opts.read_path to a .qvf / .molden / .molden.input file.")
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if ext == ".qvf":
        return _prior_from_qvf(os.fspath(path))
    if os.fspath(path).lower().endswith((".molden", ".molden.input")):
        return _prior_from_molden(os.fspath(path))
    raise ValueError(
        f"READ: unsupported read_path extension {ext!r} (expected .qvf or "
        ".molden; ORCA's .molden.input suffix is also accepted). For an "
        "in-memory restart pass read_from=prior_result.")


def _to_current_basis(D_prior: np.ndarray, prior_basis, basis, threshold=1e-7) -> np.ndarray:
    """Map a prior-basis density onto the current basis: direct when the
    bases coincide, otherwise the MINAO-style least-squares projection."""
    nbf = basis.nbasis
    D_prior = _real_density(D_prior)
    if prior_basis is None:
        raise ValueError(
            "READ: source result has no AO-basis snapshot; its geometry cannot "
            "be verified. Use a current result, a QVF/Molden source, or explicitly "
            "supply an already-projected current-basis read_density."
        )
    S_tt = np.asarray(compute_overlap(basis), dtype=float)
    S_tm = np.asarray(compute_overlap_two_basis(basis, prior_basis), dtype=float)
    from .guess import _canonical_orthogonalizer
    if D_prior.shape != (prior_basis.nbasis, prior_basis.nbasis):
        raise ValueError("READ: density shape does not match its source basis")
    X = _canonical_orthogonalizer(S_tt, threshold, normalize_diagonal=False)
    P = X @ X.T @ S_tm
    D = P @ D_prior @ P.T
    return 0.5 * (D + D.T)


# ---- public resolvers used by the run_* wrappers -------------------------


def _target_read_counts(options, molecule):
    from .ecp_metadata import effective_nuclear_charges
    electrons = int(round(float(np.sum(effective_nuclear_charges(molecule, options))))) - molecule.charge
    spin = int(molecule.multiplicity) - 1
    if electrons < 0 or spin < 0 or spin > electrons or (electrons + spin) % 2:
        raise ValueError("READ: electron count and multiplicity are inconsistent")
    return (electrons + spin) // 2, (electrons - spin) // 2


def _normalized_read_density(density, overlap, electrons):
    from .guess import normalize_density_guess
    density = _real_density(density)
    if density.shape != overlap.shape:
        raise ValueError("READ: density shape does not match the target basis")
    count = float(np.trace(density @ overlap))
    if electrons == 0:
        return np.zeros_like(density)
    if abs(count - electrons) <= 1e-12:
        return density
    return normalize_density_guess(density, overlap, electrons)


def resolve_read_density_closed(options, molecule, basis, read_from) -> np.ndarray:
    """Project and normalize a closed-shell READ source in the target AO metric."""
    na, nb = _target_read_counts(options, molecule)
    if na != nb:
        raise ValueError("READ: closed-shell calculation requires equal spin populations")
    overlap = np.asarray(compute_overlap(basis))
    preloaded = np.asarray(getattr(options, "read_density", None) if getattr(options, "read_density", None) is not None else [])
    if read_from is None and not getattr(options, "read_path", "") and preloaded.size:
        density = preloaded
    else:
        prior = _prior_state(options, read_from)
        density = _to_current_basis(
            prior.d_alpha + prior.d_beta, prior.basis, basis,
            float(getattr(options, "linear_dep_threshold", 1e-7)),
        )
    return _normalized_read_density(density, overlap, na + nb)


def resolve_read_densities_open(options, molecule, basis, read_from) -> Tuple[np.ndarray, np.ndarray]:
    """Project both spins and enforce the shared target-population contract."""
    na, nb = _target_read_counts(options, molecule)
    overlap = np.asarray(compute_overlap(basis))
    preloaded = [np.asarray(getattr(options, "read_density_" + spin, None) if getattr(options, "read_density_" + spin, None) is not None else []) for spin in ("alpha", "beta")]
    if read_from is None and not getattr(options, "read_path", "") and all(d.size for d in preloaded):
        densities = preloaded
    else:
        prior = _prior_state(options, read_from)
        densities = [
            _to_current_basis(d, prior.basis, basis, float(getattr(options, "linear_dep_threshold", 1e-7)))
            for d in (prior.d_alpha, prior.d_beta)
        ]
    from .guess import normalize_read_spin_density_guess
    densities = [_real_density(d) for d in densities]
    if any(d.shape != overlap.shape for d in densities):
        raise ValueError("READ: spin density shapes do not match the target basis")
    # Share the periodic READ policy, including empty-channel spin transfer
    # and preservation of populated source channels' magnetic patterns.
    return normalize_read_spin_density_guess(*densities, overlap, na, nb)


# ---- periodic Gamma / file-backed restart --------------------------------
#
# The Gamma/file-backed periodic READ path resolves a prior g=0 cell density
# in the *current* cell basis with the same MINAO-style least-squares projector
# the molecular path uses, evaluated with the cell (g=0, minimal-image)
# overlaps: ``compute_overlap`` / ``compute_overlap_two_basis`` on the cell
# bases. For a same-geometry restart the bases coincide and the projector is
# the identity (exact); for a shifted geometry (NEB / scan) it is a sound
# starting guess that the SCF refines. The resulting g=0 cell block Bloch-sums
# to a k-independent ``D(k)`` for the first Fock build. Closed-shell in-memory
# multi-k READ is handled separately by ``resolve_periodic_read_density_k_closed``.


def _periodic_prior(read_density, read_path, read_from) -> Optional[_Prior]:
    """Return a prior in its own basis, or ``None`` when a pre-resolved density
    is supplied (already in the current cell basis -- no further projection)."""
    if read_density is not None and np.asarray(read_density).size:
        return None
    if not (read_path or "") and read_from is None:
        raise ValueError(
            "periodic initial_guess=READ needs a source: pass "
            "run_periodic_job(..., read_from=prior_result) for an in-memory "
            "restart, or initial_guess inputs pointing at a .qvf / .molden / "
            ".molden.input file."
        )

    class _PathOpts:
        pass

    o = _PathOpts()
    o.read_path = read_path or ""
    return _prior_state(o, read_from)


def resolve_periodic_read_density_closed(
    basis, *, read_density=None, read_path="", read_from=None
) -> np.ndarray:
    """Closed-shell (periodic RHF / RKS) READ density at g=0, in the current
    cell basis. Accepts a pre-resolved ``read_density`` (used as-is) or a
    source (``read_path`` / ``read_from``) to read + project from."""
    prior = _periodic_prior(read_density, read_path, read_from)
    if prior is None:
        D = _real_density(read_density)
        if D.shape != (basis.nbasis, basis.nbasis):
            raise ValueError("READ: density shape does not match the target basis")
        return D
    return _to_current_basis(prior.d_alpha + prior.d_beta, prior.basis, basis)


def resolve_periodic_read_densities_open(
    basis, *, read_density_alpha=None, read_density_beta=None,
    read_path="", read_from=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Open-shell (periodic UHF / UKS) per-spin READ densities at g=0, in the
    current cell basis."""
    have_pre = (
        read_density_alpha is not None
        and np.asarray(read_density_alpha).size
        and read_density_beta is not None
        and np.asarray(read_density_beta).size
    )
    if have_pre:
        Da = _real_density(read_density_alpha)
        Db = _real_density(read_density_beta)
        if Da.shape != (basis.nbasis, basis.nbasis) or Db.shape != Da.shape:
            raise ValueError("READ: spin density shapes do not match the target basis")
        return Da, Db
    prior = _periodic_prior(None, read_path, read_from)
    return (_to_current_basis(prior.d_alpha, prior.basis, basis),
            _to_current_basis(prior.d_beta, prior.basis, basis))


def _multik_blocks(src, coeff_names: Sequence[str], occ_names: Sequence[str]):
    def blocks(names, single_ndim):
        for name in names:
            value = getattr(src, name, None)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return list(value)
            if isinstance(value, np.ndarray):
                if value.ndim == single_ndim:
                    return [value]
                if value.ndim == single_ndim + 1 and len(value) > 0:
                    return list(value)
        return None
    return blocks(coeff_names, 2), blocks(occ_names, 1)


def _qvf_read_json(zf: zipfile.ZipFile, members: dict, role: str, *, verify=False):
    spec = members[role]
    raw = zf.read(spec["path"])
    if verify and hashlib.sha256(raw).hexdigest() != spec.get("sha256"):
        raise ValueError(f"READ: QVF member {role} failed SHA-256 verification")
    return json.loads(raw.decode("utf-8"))


def _qvf_read_binary(zf: zipfile.ZipFile, members: dict, role: str, *, verify=False) -> np.ndarray:
    spec = members[role]
    raw = zf.read(spec["path"])
    if verify and hashlib.sha256(raw).hexdigest() != spec.get("sha256"):
        raise ValueError(f"READ: QVF density member {role} failed SHA-256 verification")
    arr = np.frombuffer(raw, dtype=np.dtype(spec.get("dtype", "float64")))
    return arr.reshape(tuple(spec["shape"]))


def _complex_from_split_last_axis(arr: np.ndarray, *, label: str) -> np.ndarray:
    payload = np.asarray(arr, dtype=np.float64)
    if payload.ndim < 1 or payload.shape[-1] != 2:
        raise ValueError(
            f"{label} must use complex_split_last_axis with trailing "
            f"[real, imag] components; got shape {payload.shape}."
        )
    return payload[..., 0] + 1j * payload[..., 1]


def _spin_density_k_from_qvf_bloch_payload(path, *, expected_n_k=None, n_basis=None):
    """Read the complete physical spin state; never infer a missing channel."""
    from .guess import _validated_density_guess_block

    with zipfile.ZipFile(path) as archive:
        sections = json.loads(archive.read("manifest.json"))["sections"]
        wf = next((s for s in sections if s["kind"] == _QVF_BLOCH_WF_KIND), None)
        if wf is None:
            return None
        members = wf["members"]
        meta = _qvf_read_json(archive, members, "mo_metadata")
        if meta.get("spin") == "restricted":
            return None
        _qvf_read_json(archive, members, "mo_metadata", verify=True)
        _qvf_read_json(archive, members, "basis", verify=True)
        structure = next((s for s in sections if s["kind"] == "structure"), None)
        if structure is None:
            raise ValueError("READ: QVF spin restart requires its source structure")
        _qvf_read_json(archive, structure["members"], "structure", verify=True)
        if (meta.get("spin") != "unrestricted"
            or meta.get("schema_version") != "1.1"
            or meta.get("restart_kind") != "spin_density_bloch_kpoints"
            or meta.get("density_encoding") != "complex_split_last_axis"
            or meta.get("density_layout") != "ao_row_ao_column"
            or meta.get("density_components") != ["real", "imag"]
            or meta.get("k_point_units") != "fractional_reciprocal"):
            raise ValueError("READ: unsupported QVF spin density restart contract")
        blocks = meta.get("blocks")
        if (not isinstance(blocks, list) or not blocks or meta.get("n_kpoints") != len(blocks)
            or (expected_n_k is not None and len(blocks) != expected_n_k)):
            raise ValueError("READ: QVF spin density k-block counts differ")
        n_ao = int(meta["n_ao"])
        if n_ao <= 0 or n_ao != meta["n_ao"] or (n_basis is not None and n_ao != n_basis):
            raise ValueError("READ: QVF spin density AO dimension does not match the basis")
        points = np.asarray([b.get("k_point") for b in blocks], dtype=float)
        weights = np.asarray([b.get("k_weight") for b in blocks], dtype=float)
        if (points.shape != (len(blocks), 3) or not np.all(np.isfinite(points))
            or weights.shape != (len(blocks),) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("READ: invalid QVF spin restart quadrature")
        pair = ([], [])
        used = set()
        for i, block in enumerate(blocks):
            if block.get("k_index") != i:
                raise ValueError("READ: QVF spin density k indices must be complete and ordered")
            for channel, spin in zip(pair, ("alpha", "beta")):
                name = block.get(f"density_{spin}")
                if not isinstance(name, str) or name not in members or name in used:
                    raise ValueError("READ: QVF spin restart requires a distinct member for every channel and k point")
                used.add(name)
                density = _complex_from_split_last_axis(
                    _qvf_read_binary(archive, members, name, verify=True), label="QVF spin density")
                channel.append(_validated_density_guess_block(density, (n_ao, n_ao)))
        return pair


def _density_k_from_qvf_bloch_payload(
    path: str,
    *,
    expected_n_k: Optional[int],
    n_basis: Optional[int],
) -> List[np.ndarray]:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        sections = list(manifest.get("sections", []))
        wf = next(
            (s for s in sections if s.get("kind") == _QVF_BLOCH_WF_KIND),
            None,
        )
        if wf is None:
            raise NotImplementedError(
                "multi-k periodic READ from QVF requires an "
                f"'{_QVF_BLOCH_WF_KIND}' all-k restart section. This archive "
                "does not contain one; older QVF files carry only "
                "'wavefunction.gto', which describes at most one selected "
                "k-point."
            )

        members = wf["members"]
        meta = _qvf_read_json(zf, members, "mo_metadata")
        if meta.get("spin") != "restricted":
            raise NotImplementedError(
                "multi-k periodic READ from QVF is currently closed-shell "
                "only; the all-k Bloch payload is not restricted-spin."
            )
        if meta.get("coefficient_encoding") != "complex_split_last_axis":
            raise ValueError(
                "multi-k periodic READ QVF payload must store coefficients "
                "with coefficient_encoding='complex_split_last_axis'."
            )
        blocks = meta.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(
                "multi-k periodic READ QVF payload has no per-k metadata blocks."
            )
        if expected_n_k is not None and len(blocks) != int(expected_n_k):
            raise ValueError(
                "multi-k periodic READ QVF k-count does not match the target "
                f"mesh: got {len(blocks)}, expected {int(expected_n_k)}."
            )

        densities: List[np.ndarray] = []
        for pos, block in enumerate(blocks):
            ik = int(block.get("k_index", pos))
            coeff_name = str(block.get("mo_coefficients", f"mo_coefficients_k{ik}"))
            occ_name = str(block.get("occupations", f"occupations_k{ik}"))
            coeff_rows = _complex_from_split_last_axis(
                _qvf_read_binary(zf, members, coeff_name),
                label=f"QVF Bloch coefficient block {ik}",
            )
            if coeff_rows.ndim != 2:
                raise ValueError(
                    f"QVF Bloch coefficient block {ik} must decode to a "
                    f"2-D [n_mo, n_ao] matrix; got shape {coeff_rows.shape}."
                )
            C = coeff_rows.T  # vibe-qc AO-major convention: [n_ao, n_mo]
            occ = np.asarray(
                _qvf_read_binary(zf, members, occ_name),
                dtype=np.float64,
            )
            if occ.ndim != 1:
                raise ValueError(
                    f"QVF Bloch occupation block {ik} must be 1-D; got "
                    f"shape {occ.shape}."
                )
            if C.shape[1] != occ.shape[0]:
                raise ValueError(
                    f"QVF Bloch block {ik} has {C.shape[1]} orbitals but "
                    f"{occ.shape[0]} occupations."
                )
            if n_basis is not None and C.shape[0] != int(n_basis):
                raise ValueError(
                    f"QVF Bloch block {ik} has {C.shape[0]} AO rows; target "
                    f"basis has {int(n_basis)}."
                )
            D = _density_from_complex_mos(C, occ)
            densities.append(0.5 * (D + D.conj().T))
        return densities


def _is_gamma_restart(points, lattice):
    """One point is Gamma only if it is a reciprocal-lattice vector."""
    points, lattice = np.asarray(points), np.asarray(lattice)
    if points.shape != (1, 3) or lattice.shape != (3, 3):
        return False
    fractional = points @ lattice / (2 * np.pi)
    return bool(np.all(np.isfinite(fractional))
                and np.allclose(fractional - np.rint(fractional), 0, rtol=0, atol=1e-10))


def _qvf_has_bloch_payload(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    return any(section["kind"] == _QVF_BLOCH_WF_KIND for section in manifest["sections"])


def _physical_periodic_density_blocks(source, names):
    """Prefer the actual SCF density over orbitals from a later diagonalization."""
    for name in names:
        density = getattr(source, name, None)
        if density is None:
            continue
        if name == "density":
            from ._vibeqc_core import PeriodicRHFResult, PeriodicKSResult
            if (isinstance(source, (PeriodicRHFResult, PeriodicKSResult))
                    and len(getattr(source, "mo_coeffs_k", ())) > 0
                    and not _is_gamma_restart(source.restart_kpoints, source.restart_lattice)):
                # Native density is P(g=0), not an all-k physical density.
                continue
        if hasattr(density, "cells"):
            from .pbc_bipole_common import bvk_torus_density_matrices, home_cell_block
            context = _restart_periodic_context(source)
            if _is_gamma_restart(context[2], context[1]):
                return [home_cell_block(density).astype(complex)]
            mesh = getattr(source, "kmesh", None)
            mesh_shape = getattr(mesh, "mesh", getattr(source, "restart_mesh", None))
            if mesh_shape is None:
                raise ValueError("READ: lattice density requires its source Born-von Karman mesh")
            return bvk_torus_density_matrices(density, list(context[2]), mesh_shape)
        if isinstance(density, (list, tuple, np.ndarray)):
            blocks = np.asarray(density, dtype=complex)
            if blocks.ndim == 2:
                return [blocks.copy()]
            if blocks.ndim == 3:
                return [d.copy() for d in blocks]
    return None


def _periodic_spin_density_blocks(source, *, expected_n_k=None, n_basis=None):
    """Read both spin channels, or return None for a spin-free source.

    Explicit but incomplete spin data must not fall back to a total density.
    Validate each channel before summing: corruption could cancel in the total.
    """
    from .guess import _validated_density_guess_block

    names = [f"{field}_{spin}{suffix}" for spin in ("alpha", "beta")
             for field in ("density", "mo_coeffs", "occupations") for suffix in ("_k", "")]
    if not any(getattr(source, name, None) is not None for name in names):
        return None
    pair = []
    for spin in ("alpha", "beta"):
        blocks = _physical_periodic_density_blocks(
            source, (f"density_{spin}_k", f"density_{spin}"))
        if blocks is None:
            if any(getattr(source, name, None) is not None
                   for name in (f"density_{spin}_k", f"density_{spin}")):
                raise ValueError("READ: multi-k restart needs a complete density list for each spin")
            coeffs, occs = _multik_blocks(
                source, (f"mo_coeffs_{spin}_k", f"mo_coeffs_{spin}"),
                (f"occupations_{spin}_k", f"occupations_{spin}"))
            if coeffs is None or occs is None or len(coeffs) != len(occs):
                raise ValueError("READ: multi-k restart needs a complete density list for each spin")
            blocks = [_density_from_complex_mos(c, o, max_occupation=1)
                      for c, o in zip(coeffs, occs)]
        pair.append(blocks)
    count = len(pair[0])
    if not count or len(pair[1]) != count or (expected_n_k is not None and count != expected_n_k):
        raise ValueError("READ: source spin density k-block counts differ or do not match the mesh")
    shape = (int(n_basis), int(n_basis)) if n_basis is not None else np.asarray(pair[0][0]).shape
    return tuple([_validated_density_guess_block(d, shape) for d in blocks] for blocks in pair)


def _load_periodic_read_density_k_closed(
    *,
    read_from=None,
    read_path: str = "",
    expected_n_k: Optional[int] = None,
    n_basis: Optional[int] = None,
) -> List[np.ndarray]:
    """Closed-shell multi-k READ density blocks ``D(k)`` in the current AO basis.

    Accepts in-memory native multi-k results or QVF archives containing the
    vibe-qc all-k Bloch restart payload. Molden and legacy QVF
    ``wavefunction.gto``-only sources remain fail-closed because they do not
    carry every per-k complex coefficient block.
    """
    if read_path or isinstance(read_from, (str, os.PathLike)):
        path = os.fspath(read_path or read_from)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".qvf":
            pair = _spin_density_k_from_qvf_bloch_payload(
                path, expected_n_k=expected_n_k, n_basis=n_basis)
            if pair is not None:
                return [a + b for a, b in zip(*pair)]
            return _density_k_from_qvf_bloch_payload(
                path,
                expected_n_k=expected_n_k,
                n_basis=n_basis,
            )
        raise NotImplementedError(
            "multi-k periodic READ from non-QVF file sources is not "
            "implemented: a multi-k restart needs all per-k complex Bloch "
            "coefficients and occupations. Use a current vibe-qc .qvf archive "
            "or an in-memory multi-k result."
        )
    if read_from is None:
        raise ValueError(
            "multi-k periodic READ needs an in-memory source result with "
            "per-k MO coefficients and occupations."
        )

    pair = _periodic_spin_density_blocks(
        read_from, expected_n_k=expected_n_k, n_basis=n_basis)
    if pair is not None:
        return [a + b for a, b in zip(*pair)]

    physical = _physical_periodic_density_blocks(
        read_from, ("density_k", "density_per_k", "density"))
    if physical is not None:
        if expected_n_k is not None and len(physical) != int(expected_n_k):
            raise ValueError("multi-k periodic READ source density k-count does not match the target mesh")
        for block in physical:
            if block.ndim != 2 or block.shape[0] != block.shape[1]:
                raise ValueError("multi-k periodic READ density blocks must be square")
            if n_basis is not None and block.shape != (int(n_basis), int(n_basis)):
                raise ValueError("multi-k periodic READ density AO dimension does not match target basis")
        return physical

    coeffs, occs = _multik_blocks(
        read_from,
        ("mo_coeffs_k", "mo_coeffs"),
        ("occupations_k", "occupations"),
    )
    if coeffs is None or occs is None:
        raise TypeError(
            "multi-k periodic READ needs an in-memory closed-shell result "
            "with per-k MO coefficients (`mo_coeffs_k` or `mo_coeffs`) and "
            "per-k occupations (`occupations_k` or `occupations`)."
        )
    if len(coeffs) != len(occs):
        raise ValueError(
            "multi-k periodic READ source has mismatched k blocks: "
            f"{len(coeffs)} coefficient blocks but {len(occs)} occupation blocks."
        )
    if expected_n_k is not None and len(coeffs) != int(expected_n_k):
        raise ValueError(
            "multi-k periodic READ source k-count does not match the target "
            f"mesh: got {len(coeffs)}, expected {int(expected_n_k)}."
        )

    densities: List[np.ndarray] = []
    for ik, (C_raw, occ_raw) in enumerate(zip(coeffs, occs)):
        C = np.asarray(C_raw, dtype=complex)
        occ = np.asarray(occ_raw)
        if C.ndim != 2:
            raise ValueError(
                f"multi-k periodic READ coefficient block {ik} must be 2D; "
                f"got shape {C.shape}."
            )
        if occ.ndim != 1:
            raise ValueError(
                f"multi-k periodic READ occupation block {ik} must be 1D; "
                f"got shape {occ.shape}."
            )
        if C.shape[1] != occ.shape[0]:
            raise ValueError(
                f"multi-k periodic READ block {ik} has {C.shape[1]} MOs but "
                f"{occ.shape[0]} occupations."
            )
        if n_basis is not None and C.shape[0] != int(n_basis):
            raise ValueError(
                f"multi-k periodic READ block {ik} has {C.shape[0]} AO rows; "
                f"target basis has {int(n_basis)}."
            )
        D = _density_from_complex_mos(C, occ)
        densities.append(0.5 * (D + D.conj().T))
    return densities


def _restart_periodic_context(source):
    """Read basis, lattice and quadrature provenance without guessing a mesh."""
    if isinstance(source, (str, os.PathLike)):
        with zipfile.ZipFile(source) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            sections = {s["kind"]: s for s in manifest["sections"]}
            if "structure" not in sections:
                raise ValueError("READ: QVF restart requires its source structure and basis shells")
            structure = _qvf_read_json(archive, sections["structure"]["members"], "structure")
            lattice = np.asarray(structure["lattice_vectors"], dtype=float).T * _BOHR_PER_ANGSTROM
            wf = sections.get(_QVF_BLOCH_WF_KIND, sections.get("wavefunction.gto"))
            basis = _prior_basis_from_qvf(
                archive, manifest, sections, _qvf_read_json(archive, wf["members"], "basis"))
            meta = _qvf_read_json(archive, wf["members"], "mo_metadata")
            if wf["kind"] == _QVF_BLOCH_WF_KIND:
                blocks = meta["blocks"]
                fractional = np.asarray([block["k_point"] for block in blocks], dtype=float)
                points = fractional @ (2 * np.pi * np.linalg.inv(lattice))
                weights = [block.get("k_weight") for block in blocks]
                if any(w is None for w in weights):
                    if len(blocks) != 1:
                        raise ValueError("READ: legacy all-k QVF lacks quadrature weights; re-export the source")
                    weights = [1.0]
            else:
                fractional = np.asarray(meta.get("k_point", [0, 0, 0]), dtype=float)
                if fractional.shape != (3,) or not np.allclose(fractional - np.rint(fractional), 0, atol=1e-10):
                    raise NotImplementedError("READ: selected non-Gamma QVF orbitals do not define an all-k restart")
                points, weights = np.zeros((1, 3)), [1.0]
            return basis, lattice, np.asarray(points), np.asarray(weights, dtype=float)
    basis = getattr(source, "restart_basis", None)
    lattice = getattr(source, "restart_lattice", None)
    points = getattr(source, "restart_kpoints", None)
    weights = getattr(source, "restart_weights", None)
    mesh = getattr(source, "kmesh", None)
    if points is None:
        points = getattr(source, "kpoints_cart", None)
    if weights is None:
        weights = getattr(source, "kpoint_weights", None)
    if mesh is not None:
        if points is None:
            points = getattr(mesh, "kpoints", None)
        if weights is None:
            weights = getattr(mesh, "weights", None)
    if basis is None or lattice is None or points is None or weights is None:
        raise ValueError("READ: periodic result requires source basis, lattice, k points and weights")
    return basis, np.asarray(lattice), np.asarray(points), np.asarray(weights)


def _map_periodic_restart(densities, source_context, basis, system, kmesh):
    """Exact mesh permutation, or an explicit Gamma on-site seed projection.

    A non-Gamma mesh cannot be interpolated by nearest neighbours: its phases
    and occupations are physical data. Different multi-k quadratures fail
    closed until a finite real-space interpolation contract is provided.
    """
    prior_basis, lattice, points, weights = source_context
    target_lattice = np.asarray(system.lattice)
    target_points = np.asarray(kmesh.kpoints)
    target_weights = np.asarray(kmesh.weights)
    if (lattice.shape != (3, 3) or not np.all(np.isfinite(lattice))
        or not np.allclose(lattice, target_lattice, rtol=0, atol=1e-9)):
        raise NotImplementedError("READ: periodic restart requires the same lattice")
    for p, w in ((points, weights), (target_points, target_weights)):
        if (p.ndim != 2 or p.shape[1] != 3 or w.shape != (len(p),)
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(w))
            or np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("READ: invalid source or target k-point quadrature")
    if len(densities) != len(points):
        raise ValueError("READ: source density and k-point block counts differ")
    source_frac = points @ lattice / (2 * np.pi)
    target_frac = target_points @ lattice / (2 * np.pi)
    if _is_gamma_restart(points, lattice):
        density = _to_current_basis(densities[0], prior_basis, basis)
        return [np.asarray(density, dtype=complex).copy() for _ in target_points]
    if prior_basis.nbasis != basis.nbasis:
        raise NotImplementedError("READ: changed multi-k basis needs a Bloch cross-basis projector")
    # Equal AO labels/dimensions alone do not prove equal functions or centers.
    ss = np.asarray(compute_overlap(prior_basis))
    tt = np.asarray(compute_overlap(basis))
    cross = np.asarray(compute_overlap_two_basis(basis, prior_basis))
    if np.max(np.abs(np.diag(ss) + np.diag(tt) - 2 * np.diag(cross)), initial=0) > 1e-9:
        raise NotImplementedError("READ: changed multi-k geometry/basis needs a Bloch cross-basis projector")
    order = []
    for point, weight in zip(target_frac, target_weights):
        differences = source_frac - point
        matches = np.flatnonzero(np.all(np.abs(differences - np.rint(differences)) < 1e-10, axis=1))
        if len(matches) != 1 or not np.isclose(weights[matches[0]], weight, rtol=0, atol=1e-12):
            raise NotImplementedError("READ: incompatible k meshes; only exact permutations or Gamma projection are supported")
        order.append(int(matches[0]))
    if len(set(order)) != len(points) or len(order) != len(points):
        raise ValueError("READ: duplicate or incomplete k-point mapping")
    return [densities[i].copy() for i in order]


def resolve_periodic_read_density_k_closed(
    *, read_from=None, read_path="", expected_n_k=None, n_basis=None,
    basis=None, system=None, kmesh=None,
):
    """Load a closed-shell restart and validate its physical mesh when supplied.

    The legacy dimension-only loader remains available to archive inspection
    callers. Calculation routes supply all three target context arguments.
    The driver normalizes the mapped blocks in its actual weighted overlap.
    """
    if basis is None and system is None and kmesh is None:
        return _load_periodic_read_density_k_closed(
            read_from=read_from, read_path=read_path,
            expected_n_k=expected_n_k, n_basis=n_basis)
    if basis is None or system is None or kmesh is None:
        raise ValueError("READ: target basis, system and kmesh must be provided together")
    source = read_path or read_from
    context = _restart_periodic_context(source)
    if isinstance(source, (str, os.PathLike)) and not _qvf_has_bloch_payload(source):
        prior = _periodic_prior(None, read_path, read_from)
        densities = [prior.d_alpha + prior.d_beta]
    else:
        densities = _load_periodic_read_density_k_closed(
            read_from=read_from, read_path=read_path,
            expected_n_k=len(context[2]), n_basis=context[0].nbasis)
    return _map_periodic_restart(densities, context, basis, system, kmesh)


def resolve_periodic_read_densities_k_open(
    *, read_from=None, read_path="", basis, system, kmesh,
):
    """Map complete spin-resolved restart densities to a compatible k mesh.

    Gamma result/QVF sources use the explicit on-site projection contract.
    Multi-k sources carry either both physical spin densities or a complete
    restricted total, split equally before target population normalization.
    Full source geometry/quadrature metadata is required in either case;
    incomplete explicit spin payloads are errors.
    """
    source = read_path or read_from
    context = _restart_periodic_context(source)
    is_file = isinstance(source, (str, os.PathLike))
    if is_file and not _qvf_has_bloch_payload(source):
        # The context reader admits legacy file data only at Gamma. A modern
        # all-k section takes precedence even when it contains just one point.
        prior = _periodic_prior(None, read_path, read_from)
        spin_blocks = ([prior.d_alpha], [prior.d_beta])
    else:
        spin_blocks = (_spin_density_k_from_qvf_bloch_payload(
            source, expected_n_k=len(context[2]), n_basis=context[0].nbasis) if is_file else
            _periodic_spin_density_blocks(
                read_from, expected_n_k=len(context[2]), n_basis=context[0].nbasis))
        if spin_blocks is None:
            total = _load_periodic_read_density_k_closed(
                read_from=read_from, read_path=read_path,
                expected_n_k=len(context[2]), n_basis=context[0].nbasis)
            spin_blocks = ([0.5 * d for d in total], [0.5 * d for d in total])
    return tuple(_map_periodic_restart(blocks, context, basis, system, kmesh)
                 for blocks in spin_blocks)


__all__ = [
    "resolve_read_density_closed",
    "resolve_read_densities_open",
    "resolve_periodic_read_density_closed",
    "resolve_periodic_read_densities_open",
    "resolve_periodic_read_density_k_closed",
    "resolve_periodic_read_densities_k_open",
]
