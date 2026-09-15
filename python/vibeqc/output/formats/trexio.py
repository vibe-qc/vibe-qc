"""TREXIO writer and round-trip reader for vibe-qc molecular SCF results.

TREXIO (Posenitskiy et al., J. Chem. Phys. 158, 174801 (2023),
doi:10.1063/5.0148161) is the TREX Centre of Excellence's open
wavefunction container: one HDF5 file, or a directory of per-group text
files, carrying the nuclei, the Gaussian basis, the AO conventions, the MO
coefficients and optionally integrals and density matrices in a layout
that every TREXIO-aware program (Quantum Package, CHAMP, QMC=Chem,
TurboRVB, QMCkl, PySCF and ORCA through ``trexio-tools``, ...) reads
without a per-program converter. The ``trexio`` Python package
(BSD-3-Clause) is an optional extra, ``pip install 'vibe-qc[trexio]'``;
it is imported lazily inside the two public functions and never bundled.

Group coverage of this increment
--------------------------------

Written: ``metadata``, ``nucleus``, ``electron``, ``basis`` (Gaussian),
``ao``, ``mo``, ``ao_1e_int`` (overlap, kinetic, potential_n_e,
core_hamiltonian) and ``state`` (the SCF total energy).

Not written: ``ecp`` (a run that used an effective core potential is
refused rather than written with an incomplete Hamiltonian), ``rdm``
(the one-particle density of a single determinant is ``diag(occupation)``
in the MO basis and is recoverable from ``mo.occupation``), ``ao_2e_int``,
``mo_1e_int`` / ``mo_2e_int``, ``cell`` / ``pbc`` (this writer is
molecular; Bloch orbitals are refused), ``determinant`` / ``csf``.

Conventions (TREXIO specification ``trex.org`` at the v2.6.1 tag)
-----------------------------------------------------------------

The TREXIO paper is not in the companion literature library (it is on
the library wishlist), so every convention below was taken from the
library's own specification text and its worked examples, and from the
converter shipped with ``trexio-tools`` for PySCF, and pinned by the
out-of-process PySCF check in ``examples/regression/runner_trexio_pyscf.py``.

* Units. "All data are stored in atomic units": ``nucleus.coord`` in
  bohr (vibe-qc's internal unit, no conversion), energies in Hartree.
* Radial functions (basis group). ``R_s(r) = N_s r^{n_s}
  sum_k f_ks a_ks exp(-gamma_ks r^2)`` with ``r_power`` ``n_s = 0`` for
  Gaussians. vibe-qc stores libint's contraction coefficient ``c_eff``,
  which already carries both the primitive and the contracted
  normalization (``vibeqc._primitive_norm``). We therefore write
  ``prim_factor`` ``f_ks = N(alpha, l) = (2 alpha/pi)^(3/4)
  (4 alpha)^(l/2) / sqrt((2l-1)!!)`` -- the norm of the axial Cartesian
  primitive ``x^l exp(-alpha r^2)``, which reproduces the specification's
  own H2 example (``f = 10.00625`` for ``l=0, alpha=33.87``; ``2.18428``
  for ``l=1, 1.407``; ``1.81360`` for ``l=2, 1.057``) -- together with
  ``coefficient`` ``a_ks = c_eff / f_ks`` and ``shell_factor`` ``N_s = 1``.
  The product ``f_ks a_ks`` is then exactly libint's ``c_eff``, so the
  stored radial function *is* the one vibe-qc evaluated. Measured on
  H2O/def2-SVP, ``c_eff / N`` agrees with PySCF's ``bas_ctr_coeff`` to
  1e-8 on every shell, which is what the ``trexio-tools`` PySCF converter
  writes as ``coefficient`` (with ``prim_factor`` equal to this same
  ``N``), so a vibe-qc file and a PySCF file carry the basis identically.
* Angular functions (ao group). vibe-qc forces spherical-harmonic shells
  (``set_pure(true)``), so ``ao.cartesian = 0`` and every AO is a real
  regular solid harmonic ``S_l^m = sqrt(4 pi/(2l+1)) r^l Y_l^m``. All
  ``2l+1`` members of a shell share one norm, and libint's pure
  functions are unit-normalized (``compute_overlap`` diagonal is 1 to
  1e-15), so ``ao.normalization = 1`` for every AO.
* AO order within a shell. TREXIO orders real solid harmonics
  ``m = 0, +1, -1, +2, -2, ..., +l, -l`` (for ``p``: ``pz, px, py``);
  libint orders them ``m = -l, ..., +l`` (for ``p``: ``py, pz, px``), the
  same facts the Molden writer relies on (:mod:`.molden`). The
  permutation is applied to the MO coefficient rows and to both indices
  of every AO matrix.
* MO layout (mo group). ``mo.coefficient`` is ``[mo.num, ao.num]``
  row-major, i.e. row ``i`` is MO ``i``; vibe-qc keeps MOs as columns of
  an ``(n_ao, n_mo)`` matrix, so the matrix is transposed after the AO
  permutation. "For UHF wave functions, mo.num is the number of
  spin-orbitals": unrestricted results (and restricted open-shell ones,
  whose alpha and beta occupations differ) are written as the alpha
  block (``mo.spin = 0``) followed by the beta block (``mo.spin = 1``)
  with occupations 1/0; a closed-shell restricted result is written once
  with occupations 2/0. ``mo.class`` is ``"Inactive"`` for an occupied
  orbital and ``"Virtual"`` for an empty one (a labelling choice among
  the specification's ``Core / Inactive / Active / Virtual / Deleted``).
* Electrons. ``electron.up_num`` / ``dn_num`` follow from the electron
  count and the multiplicity; ``electron.num`` is their sum.
* One-electron integrals. ``ao_1e_int.potential_n_e`` is
  ``<p|V_ne|q>`` with ``V_ne = -Z_A/|r - R_A|``, and
  ``core_hamiltonian = kinetic + potential_n_e`` (no ECP term, since
  ECP runs are refused).
* Energy. ``state.energy`` carries the SCF total energy
  (``state.num = 1``, ``state.id = 0``).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..._primitive_norm import libint_primitive_norm
from ...spin_channels import spin_densities
from .molden import _element, _pure_molden_to_libint, _result_occupations

__all__ = [
    "TrexioMOBlock",
    "TrexioData",
    "read_trexio",
    "write_trexio",
]

#: Back ends accepted by :func:`write_trexio` / :func:`read_trexio`.
BACKENDS = ("hdf5", "text")

_TREXIO_EXTRA_HINT = (
    "the optional 'trexio' package (TREXIO Python API, BSD-3-Clause) is not "
    "installed. Install it with:  pip install 'vibe-qc[trexio]'"
)

# Electron-count agreement required between the SCF density and the
# molecule's nominal electron count. A converged all-electron SCF matches
# to roundoff (measured 4e-15 on H2O/def2-SVP); an ECP run is off by the
# whole core, so the threshold only has to sit well below one electron.
_ELECTRON_COUNT_TOL = 1.0e-6


def _require_trexio():
    """Import the optional ``trexio`` package or raise a clear ImportError."""
    try:
        import trexio  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(f"write_trexio / read_trexio: {_TREXIO_EXTRA_HINT}") from exc
    return trexio


def _resolve_backend(trexio, backend: str) -> int:
    key = str(backend).strip().lower()
    if key == "hdf5":
        return trexio.TREXIO_HDF5
    if key == "text":
        return trexio.TREXIO_TEXT
    if key == "auto":
        return trexio.TREXIO_AUTO
    raise ValueError(
        f"TREXIO backend must be one of {BACKENDS} (or 'auto' when reading); "
        f"got {backend!r}."
    )


def _pure_trexio_to_libint(L: int) -> List[int]:
    """Index permutation: ``trexio_order[j] = libint slot of TREXIO AO j``.

    TREXIO (trex.org, "Atomic orbitals (ao group)"): real solid harmonics
    are ordered ``0, +1, -1, +2, -2, ..., +l, -l``, and "for p orbitals in
    spherical coordinates the ordering is 0, +1, -1 which corresponds to
    pz, px, py". libint orders a pure shell ``m = -l, ..., +l`` and its
    pure p shell is ``(py, pz, px)`` (:func:`.molden._pure_molden_to_libint`,
    verified by the PySCF Molden round trip). For ``L >= 2`` TREXIO's order
    coincides with Molden's, so the Molden permutation is reused; for
    ``L = 1`` Molden wants ``(px, py, pz)`` while TREXIO wants
    ``(pz, px, py)``, i.e. libint slots ``(1, 2, 0)``.
    """
    if L == 1:
        return [1, 2, 0]
    return _pure_molden_to_libint(L)


def _build_ao_permutation(shell_ang_mom: Sequence[int]) -> np.ndarray:
    """Global AO permutation ``perm`` with ``C_trexio = C_libint[perm, :]``.

    Walks the shells in their stored order (libint's native order for a
    vibe-qc :class:`BasisSet`; the file's shell order when reading) and
    stitches the per-shell spherical reorderings together.
    """
    perm: List[int] = []
    offset = 0
    for L in shell_ang_mom:
        local = _pure_trexio_to_libint(int(L))
        perm.extend(offset + i for i in local)
        offset += len(local)
    return np.asarray(perm, dtype=np.intp)


def _remove_existing(path: Path, backend: str) -> None:
    """Remove a previous artefact at ``path`` so TREXIO can create it fresh.

    TREXIO opens an existing file in write mode and then refuses every
    attribute that is already present, so a rewrite must start from a
    clean target. A text-back-end artefact is a directory; only a
    directory that looks like one (nothing but ``*.txt`` files and the
    library's ``.lock``) is removed, so a stray path cannot delete user
    data.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        entries = list(path.iterdir())
        if all(e.name == ".lock" or e.suffix == ".txt" for e in entries):
            shutil.rmtree(path)
            return
        raise FileExistsError(
            f"write_trexio: {path} is a directory that does not look like a "
            f"TREXIO text-back-end artefact (found "
            f"{sorted(e.name for e in entries)[:5]}); refusing to remove it."
        )


def _vibeqc_provenance() -> Dict[str, str]:
    """``{"version": ..., "sha": ...}`` for the metadata group; never raises."""
    version = "unknown"
    sha = "unknown"
    try:
        from ... import __version__ as _version  # noqa: WPS433 - lazy, no cycle at call time

        version = str(_version)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from ...banner import build_info

        info = build_info() or {}
        sha = str(info.get("sha_full") or info.get("sha") or "unknown")
    except Exception:  # pragma: no cover - defensive
        pass
    return {"version": version, "sha": sha}


def _mo_type_label(result: Any) -> str:
    name = type(result).__name__
    if name.endswith("Result"):
        name = name[: -len("Result")]
    return name or "MO"


@dataclass
class _MOBlockOut:
    """One spin block on its way to the file (libint AO order)."""

    spin: int
    coefficients: np.ndarray  # (n_ao, n_mo), columns are MOs
    energies: np.ndarray
    occupations: np.ndarray


def _real_2d(array: Any, what: str) -> np.ndarray:
    arr = np.asarray(array)
    if np.iscomplexobj(arr):
        raise ValueError(
            f"write_trexio: {what} are complex. This writer covers molecular "
            "results; Bloch orbitals of a periodic run are not written in "
            "this increment."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"write_trexio: {what} must be a 2-D (n_ao, n_mo) array; got "
            f"shape {arr.shape}."
        )
    return np.ascontiguousarray(arr, dtype=float)


def _real_1d(array: Any, n: int, what: str) -> np.ndarray:
    arr = np.asarray(array)
    if np.iscomplexobj(arr):
        arr = arr.real
    arr = np.ascontiguousarray(arr, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(
            f"write_trexio: {what} has {arr.shape[0]} entries for {n} orbitals."
        )
    return arr


def _collect_mo_blocks(
    result: Any, n_alpha: int, n_beta: int, multiplicity: int
) -> List[_MOBlockOut]:
    """Turn a vibe-qc SCF result into TREXIO spin blocks.

    Restricted closed-shell: one block, occupations 2/0 (``mo.num`` is the
    number of spatial orbitals). Unrestricted, or restricted with
    ``n_alpha != n_beta`` (ROHF / ROKS): alpha then beta spin-orbital
    blocks with occupations 1/0, as the specification prescribes for UHF
    wave functions.
    """
    has_alpha = hasattr(result, "mo_coeffs_alpha")
    has_restricted = hasattr(result, "mo_coeffs") and not has_alpha
    if not (has_alpha or has_restricted):
        raise TypeError(
            "write_trexio: result must expose either `mo_coeffs` (restricted) "
            "or `mo_coeffs_alpha` + `mo_coeffs_beta` (unrestricted); got "
            + type(result).__name__
        )

    if has_restricted:
        coeffs = _real_2d(result.mo_coeffs, "MO coefficients")
        n_mo = coeffs.shape[1]
        energies = _real_1d(result.mo_energies, n_mo, "MO energies")
        if multiplicity == 1 and n_alpha == n_beta:
            carried = _result_occupations(result, "occupations", n_mo)
            if carried is not None:
                occ = np.asarray(carried, dtype=float)
            else:
                occ = np.zeros(n_mo)
                occ[:n_alpha] = 2.0
            return [_MOBlockOut(0, coeffs, energies, occ)]
        # Restricted open shell: the same spatial orbitals carry
        # different alpha / beta occupations, which the spin-free
        # spin-orbital layout represents exactly.
        occ_a = np.zeros(n_mo)
        occ_a[:n_alpha] = 1.0
        occ_b = np.zeros(n_mo)
        occ_b[:n_beta] = 1.0
        return [
            _MOBlockOut(0, coeffs, energies, occ_a),
            _MOBlockOut(1, coeffs, energies, occ_b),
        ]

    ca = _real_2d(result.mo_coeffs_alpha, "alpha MO coefficients")
    cb = _real_2d(result.mo_coeffs_beta, "beta MO coefficients")
    ea = _real_1d(result.mo_energies_alpha, ca.shape[1], "alpha MO energies")
    eb = _real_1d(result.mo_energies_beta, cb.shape[1], "beta MO energies")
    carried_a = _result_occupations(result, "occupations_alpha", ca.shape[1])
    carried_b = _result_occupations(result, "occupations_beta", cb.shape[1])
    if carried_a is not None and carried_b is not None:
        occ_a = np.asarray(carried_a, dtype=float)
        occ_b = np.asarray(carried_b, dtype=float)
    else:
        occ_a = np.zeros(ca.shape[1])
        occ_a[:n_alpha] = 1.0
        occ_b = np.zeros(cb.shape[1])
        occ_b[:n_beta] = 1.0
    return [_MOBlockOut(0, ca, ea, occ_a), _MOBlockOut(1, cb, eb, occ_b)]


def _scf_electron_count(result: Any, overlap: np.ndarray) -> Optional[float]:
    """``tr(D S)`` of the converged SCF density, or ``None`` when absent."""
    density = None
    _tx_a, _tx_b = spin_densities(result)
    if _tx_a is not None:
        da = np.asarray(_tx_a)
        db = np.asarray(_tx_b)
        if da.shape == overlap.shape and db.shape == overlap.shape:
            density = da + db
    elif getattr(result, "density", None) is not None:
        d = np.asarray(result.density)
        if d.shape == overlap.shape:
            density = d
    if density is None:
        return None
    if np.iscomplexobj(density):
        density = density.real
    return float(np.trace(np.asarray(density, dtype=float) @ overlap))


def write_trexio(
    path: os.PathLike | str,
    molecule: Any,
    basis: Any,
    result: Any,
    *,
    backend: str = "hdf5",
    description: str = "",
    author: Optional[str] = None,
    mo_type: Optional[str] = None,
    write_integrals: bool = True,
    uses_ecp: bool = False,
    overwrite: bool = True,
) -> Path:
    """Write a molecular SCF result as a TREXIO file.

    Parameters
    ----------
    path
        Target path. With ``backend="hdf5"`` this is one file
        (``.h5`` / ``.trexio.h5`` by convention); with ``backend="text"``
        it is a *directory* that TREXIO fills with one ``<group>.txt`` per
        group.
    molecule, basis
        The :class:`~vibeqc.Molecule` and :class:`~vibeqc.BasisSet` the
        SCF ran on. The basis must be spherical (vibe-qc's default) and
        nucleus-centred.
    result
        ``RHFResult`` / ``UHFResult`` / ``RKSResult`` / ``UKSResult`` (or any
        object exposing ``mo_coeffs`` + ``mo_energies``, or their
        ``_alpha`` / ``_beta`` pairs, plus the converged ``density``).
    backend
        ``"hdf5"`` (default) or ``"text"``.
    description
        Free text for ``metadata.description``; the vibe-qc version and git
        revision are appended.
    author
        Optional ``metadata.author`` entry. Nothing is written when
        omitted, so no user identity leaks into the artefact by default.
    mo_type
        ``mo.type`` label; defaults to the result class (``"RHF"``,
        ``"UKS"``, ...).
    write_integrals
        Also write the ``ao_1e_int`` group (overlap, kinetic, nuclear
        attraction, core Hamiltonian) from the public integral API. The
        overlap is computed regardless, for the electron-count check.
    uses_ecp
        Pass ``True`` when the run used an effective core potential. The
        writer then refuses: this increment does not write the ``ecp``
        group, and a file whose nuclei carry full charges but whose
        orbitals describe valence electrons only would be silently wrong.
        The same refusal fires from the density's electron count when the
        caller did not say so.
    overwrite
        Replace an existing artefact at ``path`` (default). ``False``
        raises :class:`FileExistsError` instead.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    trexio = _require_trexio()
    from ..._vibeqc_core import compute_kinetic, compute_nuclear, compute_overlap

    if uses_ecp:
        raise ValueError(
            "write_trexio: this run used an effective core potential, and the "
            "TREXIO `ecp` group is not written in this increment. Writing the "
            "file anyway would store full nuclear charges next to valence-only "
            "orbitals -- an incomplete Hamiltonian that no reader could detect. "
            "Run all-electron, or wait for the ECP increment of #573."
        )

    back_end = _resolve_backend(trexio, backend)
    if str(backend).lower() == "auto":
        raise ValueError("write_trexio: backend must be 'hdf5' or 'text'.")
    target = Path(os.fspath(path))
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"write_trexio: {target} already exists.")
        _remove_existing(target, backend)
    target.parent.mkdir(parents=True, exist_ok=True)

    # ---- nuclei -------------------------------------------------------- #
    atoms = list(molecule.atoms)
    n_atoms = len(atoms)
    charges = np.asarray([float(atom.Z) for atom in atoms])
    coords = np.asarray([[float(c) for c in atom.xyz] for atom in atoms]).reshape(
        n_atoms, 3
    )
    labels = [_element(int(atom.Z)) for atom in atoms]

    # ---- electrons ----------------------------------------------------- #
    n_electrons = int(molecule.n_electrons())
    multiplicity = int(molecule.multiplicity)
    n_alpha = (n_electrons + multiplicity - 1) // 2
    n_beta = n_electrons - n_alpha
    if n_beta < 0 or n_alpha - n_beta != multiplicity - 1:
        raise ValueError(
            f"write_trexio: {n_electrons} electrons cannot realise "
            f"multiplicity {multiplicity}."
        )

    # ---- basis --------------------------------------------------------- #
    shells = list(basis.shells())
    nucleus_index: List[int] = []
    shell_ang_mom: List[int] = []
    shell_index: List[int] = []
    exponents: List[float] = []
    coefficients: List[float] = []
    prim_factor: List[float] = []
    for s, shell in enumerate(shells):
        if not shell.pure:
            raise ValueError(
                "write_trexio: Cartesian shells are not supported "
                f"(shell {s}, L={shell.l}). vibe-qc forces set_pure(true) at "
                "construction -- this BasisSet was created outside the normal "
                "path."
            )
        atom_index = int(shell.atom_index)
        if not (0 <= atom_index < n_atoms):
            raise ValueError(
                f"write_trexio: shell {s} sits on atom index {atom_index}, but "
                f"the molecule has {n_atoms} atoms."
            )
        origin = np.asarray(shell.origin, dtype=float)
        if not np.allclose(origin, coords[atom_index], atol=1.0e-10):
            raise ValueError(
                f"write_trexio: shell {s} is centred at {origin.tolist()} "
                f"but atom {atom_index} sits at {coords[atom_index].tolist()}; "
                "TREXIO basis functions are nucleus-centred and floating "
                "centres are not written in this increment."
            )
        L = int(shell.l)
        nucleus_index.append(atom_index)
        shell_ang_mom.append(L)
        for alpha, c_eff in zip(shell.exponents, shell.coefficients):
            # trex.org, basis group: R_s = N_s r^0 sum_k f_ks a_ks exp(-g r^2)
            # with f_ks the axial-primitive norm (vibeqc._primitive_norm) and
            # a_ks = c_eff / f_ks, so that f_ks a_ks is libint's own c_eff.
            f_ks = libint_primitive_norm(float(alpha), L)
            shell_index.append(s)
            exponents.append(float(alpha))
            coefficients.append(float(c_eff) / f_ks)
            prim_factor.append(f_ks)
    shell_num = len(shells)
    prim_num = len(exponents)
    if shell_num == 0 or prim_num == 0:
        raise ValueError("write_trexio: the basis carries no shells.")
    shell_factor = [1.0] * shell_num
    r_power = [0] * shell_num

    # ---- ao ------------------------------------------------------------ #
    ao_perm = _build_ao_permutation(shell_ang_mom)
    ao_num = int(ao_perm.shape[0])
    if ao_num != int(basis.nbasis):
        raise ValueError(
            f"write_trexio: {ao_num} spherical AOs from the shell list but the "
            f"BasisSet reports nbasis={basis.nbasis}."
        )
    ao_shell = [s for s, L in enumerate(shell_ang_mom) for _ in range(2 * L + 1)]
    ao_normalization = [1.0] * ao_num

    # ---- integrals + electron-count guard ------------------------------ #
    overlap = np.asarray(compute_overlap(basis), dtype=float)
    if overlap.shape != (ao_num, ao_num):
        raise ValueError(
            f"write_trexio: overlap is {overlap.shape} for ao_num={ao_num}."
        )
    scf_count = _scf_electron_count(result, overlap)
    if scf_count is not None and abs(scf_count - n_electrons) > _ELECTRON_COUNT_TOL:
        raise ValueError(
            f"write_trexio: the SCF density integrates to {scf_count:.6f} "
            f"electrons but the molecule has {n_electrons}. The run most "
            "likely used an effective core potential; the TREXIO `ecp` group "
            "is not written in this increment, so the file is refused rather "
            "than written with an incomplete Hamiltonian."
        )

    # ---- molecular orbitals -------------------------------------------- #
    blocks = _collect_mo_blocks(result, n_alpha, n_beta, multiplicity)
    for block in blocks:
        if block.coefficients.shape[0] != ao_num:
            raise ValueError(
                f"write_trexio: MO coefficients have {block.coefficients.shape[0]} "
                f"AO rows for ao_num={ao_num}."
            )
    mo_num = sum(block.coefficients.shape[1] for block in blocks)
    # trex.org, mo group: coefficient is [mo.num, ao.num] row-major, so
    # every row is one MO in TREXIO's AO order.
    mo_coefficient = np.concatenate(
        [block.coefficients[ao_perm, :].T for block in blocks], axis=0
    )
    mo_energy = np.concatenate([block.energies for block in blocks])
    mo_occupation = np.concatenate([block.occupations for block in blocks])
    mo_spin = np.concatenate(
        [np.full(block.coefficients.shape[1], block.spin, dtype=np.int64) for block in blocks]
    )
    mo_class = ["Inactive" if occ > 0.0 else "Virtual" for occ in mo_occupation]
    occupied_electrons = float(mo_occupation.sum())
    if abs(occupied_electrons - n_electrons) > _ELECTRON_COUNT_TOL:
        raise ValueError(
            f"write_trexio: the MO occupations sum to {occupied_electrons:.6f} "
            f"electrons but the molecule has {n_electrons}."
        )

    # ---- write --------------------------------------------------------- #
    provenance = _vibeqc_provenance()
    full_description = (
        f"{description.strip()} " if description.strip() else ""
    ) + (
        f"Written by vibe-qc {provenance['version']} "
        f"(git {provenance['sha']}); {_mo_type_label(result)} orbitals, "
        f"spherical AOs in TREXIO order, basis {getattr(basis, 'name', '?')}."
    )

    with trexio.File(str(target), mode="w", back_end=back_end) as f:
        # metadata
        trexio.write_metadata_code_num(f, 1)
        trexio.write_metadata_code(f, [f"vibe-qc {provenance['version']}"])
        if author:
            trexio.write_metadata_author_num(f, 1)
            trexio.write_metadata_author(f, [str(author)])
        trexio.write_metadata_description(f, full_description)

        # nucleus
        trexio.write_nucleus_num(f, n_atoms)
        trexio.write_nucleus_charge(f, charges)
        trexio.write_nucleus_coord(f, coords)
        trexio.write_nucleus_label(f, labels)
        trexio.write_nucleus_repulsion(f, float(molecule.nuclear_repulsion()))

        # electron
        trexio.write_electron_num(f, n_electrons)
        trexio.write_electron_up_num(f, n_alpha)
        trexio.write_electron_dn_num(f, n_beta)

        # basis
        trexio.write_basis_type(f, "Gaussian")
        trexio.write_basis_shell_num(f, shell_num)
        trexio.write_basis_prim_num(f, prim_num)
        trexio.write_basis_nucleus_index(f, nucleus_index)
        trexio.write_basis_shell_ang_mom(f, shell_ang_mom)
        trexio.write_basis_shell_factor(f, shell_factor)
        trexio.write_basis_r_power(f, r_power)
        trexio.write_basis_shell_index(f, shell_index)
        trexio.write_basis_exponent(f, exponents)
        trexio.write_basis_coefficient(f, coefficients)
        trexio.write_basis_prim_factor(f, prim_factor)

        # ao
        trexio.write_ao_cartesian(f, 0)
        trexio.write_ao_num(f, ao_num)
        trexio.write_ao_shell(f, ao_shell)
        trexio.write_ao_normalization(f, ao_normalization)

        # mo
        trexio.write_mo_type(f, mo_type or _mo_type_label(result))
        trexio.write_mo_num(f, mo_num)
        trexio.write_mo_coefficient(f, np.ascontiguousarray(mo_coefficient))
        trexio.write_mo_energy(f, mo_energy)
        trexio.write_mo_occupation(f, mo_occupation)
        trexio.write_mo_spin(f, mo_spin)
        trexio.write_mo_class(f, mo_class)

        # ao_1e_int (TREXIO AO order on both indices)
        if write_integrals:
            ix = np.ix_(ao_perm, ao_perm)
            kinetic = np.asarray(compute_kinetic(basis), dtype=float)
            potential = np.asarray(compute_nuclear(basis, molecule), dtype=float)
            trexio.write_ao_1e_int_overlap(f, np.ascontiguousarray(overlap[ix]))
            trexio.write_ao_1e_int_kinetic(f, np.ascontiguousarray(kinetic[ix]))
            trexio.write_ao_1e_int_potential_n_e(
                f, np.ascontiguousarray(potential[ix])
            )
            trexio.write_ao_1e_int_core_hamiltonian(
                f, np.ascontiguousarray((kinetic + potential)[ix])
            )

        # state: the SCF total energy
        energy = getattr(result, "energy", None)
        if energy is not None:
            trexio.write_state_num(f, 1)
            trexio.write_state_id(f, 0)
            trexio.write_state_current_label(f, "ground state")
            trexio.write_state_energy(f, float(energy))

    return target


# ---------------------------------------------------------------------- #
# Reader                                                                 #
# ---------------------------------------------------------------------- #


@dataclass
class TrexioMOBlock:
    """One spin block of molecular orbitals in libint AO order."""

    spin: int
    coefficients: np.ndarray  # (n_ao, n_mo); columns are MOs
    energies: np.ndarray
    occupations: np.ndarray
    classes: List[str]


@dataclass
class TrexioData:
    """Contents of a TREXIO file as vibe-qc-shaped arrays.

    Every AO-indexed array is kept in the file's own (TREXIO) AO order;
    :attr:`ao_perm` maps it back to libint order (``libint_index =
    ao_perm[trexio_index]``), and :meth:`mo_blocks` / :meth:`basis_set` do
    that mapping for the caller.
    """

    labels: List[str]
    charges: np.ndarray
    coords: np.ndarray  # (n_atoms, 3), bohr
    n_up: int
    n_dn: int
    shell_atom: List[int]
    shell_ang_mom: List[int]
    shell_factor: np.ndarray
    prim_shell: List[int]
    exponents: np.ndarray
    coefficients: np.ndarray  # a_ks
    prim_factor: np.ndarray  # f_ks
    ao_perm: np.ndarray
    mo_type: str
    mo_coefficient: np.ndarray  # (mo_num, ao_num), TREXIO layout
    mo_energy: np.ndarray
    mo_occupation: np.ndarray
    mo_spin: np.ndarray
    mo_class: List[str]
    overlap: Optional[np.ndarray] = None
    kinetic: Optional[np.ndarray] = None
    potential_n_e: Optional[np.ndarray] = None
    core_hamiltonian: Optional[np.ndarray] = None
    nuclear_repulsion: Optional[float] = None
    energy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- derived views ----------------------------------------------------- #

    @property
    def n_electrons(self) -> int:
        return int(self.n_up + self.n_dn)

    @property
    def multiplicity(self) -> int:
        return int(self.n_up - self.n_dn + 1)

    @property
    def charge(self) -> int:
        return int(round(float(self.charges.sum()))) - self.n_electrons

    def molecule(self):
        """Rebuild the :class:`~vibeqc.Molecule` (bohr, charge, multiplicity)."""
        from ..._vibeqc_core import Atom, Molecule

        atoms = [
            Atom(int(round(float(z))), [float(x) for x in xyz])
            for z, xyz in zip(self.charges, self.coords)
        ]
        return Molecule(atoms, charge=self.charge, multiplicity=self.multiplicity)

    def shells(self) -> list:
        """Rebuild libint-ready ``ShellInfo`` objects in the file's shell order.

        The stored coefficient of each primitive is ``a_ks``; libint wants
        the effective contraction coefficient ``N_s f_ks a_ks`` that
        multiplies the bare Gaussian, so the factors are folded back in and
        the shells are marked ``coefficients_pre_normalized`` downstream.
        """
        from ..._vibeqc_core import ShellInfo

        prim_shell = np.asarray(self.prim_shell)
        out = []
        for s, (atom, L) in enumerate(zip(self.shell_atom, self.shell_ang_mom)):
            sel = np.nonzero(prim_shell == s)[0]
            exps = [float(self.exponents[k]) for k in sel]
            coefs = [
                float(self.shell_factor[s] * self.prim_factor[k] * self.coefficients[k])
                for k in sel
            ]
            origin = [float(x) for x in self.coords[atom]]
            out.append(ShellInfo(int(atom), int(L), True, exps, coefs, origin))
        return out

    def basis_set(self, molecule=None, name: str = "<trexio>"):
        """Rebuild the :class:`~vibeqc.BasisSet` on ``molecule`` (default: rebuilt)."""
        from ..._vibeqc_core import BasisSet

        mol = molecule if molecule is not None else self.molecule()
        return BasisSet(mol, self.shells(), name, True)

    def mo_blocks(self) -> List[TrexioMOBlock]:
        """MO coefficients per spin, as ``(n_ao, n_mo)`` in libint AO order."""
        n_ao = int(self.ao_perm.shape[0])
        blocks: List[TrexioMOBlock] = []
        for spin in sorted({int(s) for s in self.mo_spin}):
            sel = np.nonzero(self.mo_spin == spin)[0]
            stored = self.mo_coefficient[sel, :]  # (n_sel, n_ao) TREXIO order
            libint = np.empty((n_ao, sel.shape[0]), dtype=float)
            libint[self.ao_perm, :] = stored.T
            blocks.append(
                TrexioMOBlock(
                    spin=spin,
                    coefficients=libint,
                    energies=np.asarray(self.mo_energy[sel], dtype=float),
                    occupations=np.asarray(self.mo_occupation[sel], dtype=float),
                    classes=[self.mo_class[int(k)] for k in sel],
                )
            )
        return blocks

    def to_libint_order(self, matrix: np.ndarray) -> np.ndarray:
        """Return an AO-indexed matrix from TREXIO AO order in libint order."""
        arr = np.asarray(matrix)
        out = np.empty_like(arr)
        out[np.ix_(self.ao_perm, self.ao_perm)] = arr
        return out


def read_trexio(path: os.PathLike | str, *, backend: str = "auto") -> TrexioData:
    """Read a TREXIO file written by :func:`write_trexio` (or a compatible one).

    Supported: ``basis.type = "Gaussian"`` with ``r_power = 0``, spherical
    AOs (``ao.cartesian = 0``) with ``ao.normalization = 1``, no ``ecp``
    group. Anything else is refused with a message naming the unsupported
    feature rather than silently reinterpreted.
    """
    trexio = _require_trexio()
    back_end = _resolve_backend(trexio, backend)
    target = Path(os.fspath(path))
    if not target.exists():
        raise FileNotFoundError(f"read_trexio: {target} does not exist.")

    with trexio.File(str(target), mode="r", back_end=back_end) as f:
        if trexio.has_ecp_num(f):
            raise ValueError(
                f"read_trexio: {target} carries an `ecp` group, which this "
                "increment does not read."
            )
        n_atoms = int(trexio.read_nucleus_num(f))
        charges = np.asarray(trexio.read_nucleus_charge(f), dtype=float)
        coords = np.asarray(trexio.read_nucleus_coord(f), dtype=float).reshape(n_atoms, 3)
        labels = (
            list(trexio.read_nucleus_label(f))
            if trexio.has_nucleus_label(f)
            else [_element(int(round(z))) for z in charges]
        )
        nuclear_repulsion = (
            float(trexio.read_nucleus_repulsion(f))
            if trexio.has_nucleus_repulsion(f)
            else None
        )

        n_up = int(trexio.read_electron_up_num(f))
        n_dn = int(trexio.read_electron_dn_num(f))

        basis_type = str(trexio.read_basis_type(f))
        if basis_type != "Gaussian":
            raise ValueError(
                f"read_trexio: basis.type={basis_type!r} is not supported "
                "(only Gaussian)."
            )
        shell_num = int(trexio.read_basis_shell_num(f))
        prim_num = int(trexio.read_basis_prim_num(f))
        shell_atom = [int(i) for i in trexio.read_basis_nucleus_index(f)]
        shell_ang_mom = [int(L) for L in trexio.read_basis_shell_ang_mom(f)]
        shell_factor = (
            np.asarray(trexio.read_basis_shell_factor(f), dtype=float)
            if trexio.has_basis_shell_factor(f)
            else np.ones(shell_num)
        )
        if trexio.has_basis_r_power(f):
            r_power = [int(p) for p in trexio.read_basis_r_power(f)]
            if any(p != 0 for p in r_power):
                raise ValueError(
                    "read_trexio: basis.r_power != 0 (Slater-type radial "
                    "prefactors) is not supported."
                )
        prim_shell = [int(s) for s in trexio.read_basis_shell_index(f)]
        exponents = np.asarray(trexio.read_basis_exponent(f), dtype=float)
        coefficients = np.asarray(trexio.read_basis_coefficient(f), dtype=float)
        prim_factor = (
            np.asarray(trexio.read_basis_prim_factor(f), dtype=float)
            if trexio.has_basis_prim_factor(f)
            else np.ones(prim_num)
        )
        if len(shell_atom) != shell_num or len(prim_shell) != prim_num:
            raise ValueError("read_trexio: inconsistent basis group dimensions.")

        cartesian = int(trexio.read_ao_cartesian(f)) if trexio.has_ao_cartesian(f) else 0
        if cartesian != 0:
            raise ValueError(
                "read_trexio: ao.cartesian != 0 (Cartesian AOs) is not "
                "supported; vibe-qc evaluates spherical-harmonic shells."
            )
        ao_num = int(trexio.read_ao_num(f))
        ao_shell = [int(s) for s in trexio.read_ao_shell(f)]
        expected_ao_shell = [
            s for s, L in enumerate(shell_ang_mom) for _ in range(2 * L + 1)
        ]
        if ao_shell != expected_ao_shell:
            raise ValueError(
                "read_trexio: ao.shell does not enumerate 2l+1 consecutive AOs "
                "per shell in shell order; this reader assumes that layout."
            )
        if trexio.has_ao_normalization(f):
            ao_norm = np.asarray(trexio.read_ao_normalization(f), dtype=float)
            if not np.allclose(ao_norm, 1.0, atol=1.0e-12):
                raise ValueError(
                    "read_trexio: ao.normalization != 1 is not supported for "
                    "spherical AOs (every real solid harmonic of a shell shares "
                    "one norm in vibe-qc)."
                )
        ao_perm = _build_ao_permutation(shell_ang_mom)
        if ao_perm.shape[0] != ao_num:
            raise ValueError(
                f"read_trexio: ao.num={ao_num} but the shells span "
                f"{ao_perm.shape[0]} spherical AOs."
            )

        mo_num = int(trexio.read_mo_num(f))
        mo_type = str(trexio.read_mo_type(f)) if trexio.has_mo_type(f) else "MO"
        mo_coefficient = np.asarray(
            trexio.read_mo_coefficient(f), dtype=float
        ).reshape(mo_num, ao_num)
        mo_energy = (
            np.asarray(trexio.read_mo_energy(f), dtype=float)
            if trexio.has_mo_energy(f)
            else np.full(mo_num, np.nan)
        )
        mo_occupation = (
            np.asarray(trexio.read_mo_occupation(f), dtype=float)
            if trexio.has_mo_occupation(f)
            else np.full(mo_num, np.nan)
        )
        mo_spin = (
            np.asarray(trexio.read_mo_spin(f), dtype=np.int64)
            if trexio.has_mo_spin(f)
            else np.zeros(mo_num, dtype=np.int64)
        )
        mo_class = (
            list(trexio.read_mo_class(f)) if trexio.has_mo_class(f) else [""] * mo_num
        )

        def _matrix(has, read):
            if not has(f):
                return None
            return np.asarray(read(f), dtype=float).reshape(ao_num, ao_num)

        overlap = _matrix(trexio.has_ao_1e_int_overlap, trexio.read_ao_1e_int_overlap)
        kinetic = _matrix(trexio.has_ao_1e_int_kinetic, trexio.read_ao_1e_int_kinetic)
        potential = _matrix(
            trexio.has_ao_1e_int_potential_n_e, trexio.read_ao_1e_int_potential_n_e
        )
        hcore = _matrix(
            trexio.has_ao_1e_int_core_hamiltonian,
            trexio.read_ao_1e_int_core_hamiltonian,
        )

        energy = float(trexio.read_state_energy(f)) if trexio.has_state_energy(f) else None

        metadata: Dict[str, Any] = {}
        if trexio.has_metadata_code(f):
            metadata["code"] = list(trexio.read_metadata_code(f))
        if trexio.has_metadata_author(f):
            metadata["author"] = list(trexio.read_metadata_author(f))
        if trexio.has_metadata_description(f):
            metadata["description"] = str(trexio.read_metadata_description(f))
        if trexio.has_metadata_package_version(f):
            metadata["package_version"] = str(trexio.read_metadata_package_version(f))

    return TrexioData(
        labels=labels,
        charges=charges,
        coords=coords,
        n_up=n_up,
        n_dn=n_dn,
        shell_atom=shell_atom,
        shell_ang_mom=shell_ang_mom,
        shell_factor=shell_factor,
        prim_shell=prim_shell,
        exponents=exponents,
        coefficients=coefficients,
        prim_factor=prim_factor,
        ao_perm=ao_perm,
        mo_type=mo_type,
        mo_coefficient=mo_coefficient,
        mo_energy=mo_energy,
        mo_occupation=mo_occupation,
        mo_spin=mo_spin,
        mo_class=mo_class,
        overlap=overlap,
        kinetic=kinetic,
        potential_n_e=potential,
        core_hamiltonian=hcore,
        nuclear_repulsion=nuclear_repulsion,
        energy=energy,
        metadata=metadata,
    )
