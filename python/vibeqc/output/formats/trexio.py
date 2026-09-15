"""TREXIO wavefunction export, native reconstruction and general data I/O.

TREXIO (Posenitskiy et al., J. Chem. Phys. 158, 174801 (2023),
doi:10.1063/5.0148161) is the TREX Centre of Excellence's open
wavefunction container: one HDF5 file, or a directory of per-group text
files, carrying the nuclei, the Gaussian basis, the AO conventions, the MO
coefficients and optionally integrals and density matrices. The ``trexio``
Python package (BSD-3-Clause) is an optional extra,
``pip install 'vibe-qc[trexio]'``. It is imported lazily and never bundled.

The result exporter covers molecular and periodic Gaussian wavefunctions,
including ECPs, complex orbitals and one-particle densities. The general
``read_trexio_fields`` / ``write_trexio_fields`` API transports every group
exposed by the installed TREXIO library, including sparse integrals, higher
RDMs, amplitudes, determinants, CSFs, numerical bases and Jastrow parameters.
Native basis reconstruction is available for spherical Gaussian orbitals.

Conventions follow the TREXIO specification and its worked Gaussian
normalization examples. Radial contraction coefficients include libint's
primitive normalization through explicit prim_factor; shell_factor and
AO normalization are one on export. Spherical AOs are permuted from
libint's -l,...,+l order to TREXIO's 0,+1,-1,... order. Molecule and basis
reconstruction restores those factors and the native order. See the
user-guide TREXIO page and the independent PySCF regression runner.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..._primitive_norm import libint_primitive_norm
from ...spin_channels import spin_densities
from .molden import _element, _pure_molden_to_libint
from ._trexio_data import TrexioSparse, atomic_target, read_fields, write_fields

__all__ = [
    "TrexioMOBlock",
    "TrexioData",
    "read_trexio",
    "write_trexio",
    "TrexioSparse",
    "read_trexio_fields",
    "write_trexio_fields",
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


def _coefficient_2d(array: Any, what: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(
            f"write_trexio: {what} must be a 2-D (n_ao, n_mo) array; got "
            f"shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"write_trexio: {what} contains non-finite values.")
    return np.ascontiguousarray(arr, dtype=complex if np.iscomplexobj(arr) else float)


def _real_1d(array: Any, n: int, what: str) -> np.ndarray:
    arr = np.asarray(array)
    if (np.iscomplexobj(arr) and np.any(arr.imag)) or not np.all(np.isfinite(arr)):
        raise ValueError(f"write_trexio: {what} must be finite and real.")
    arr = np.ascontiguousarray(arr.real, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(
            f"write_trexio: {what} has {arr.shape[0]} entries for {n} orbitals."
        )
    return arr


def _collect_mo_blocks(
    result: Any, n_alpha: int, n_beta: int, multiplicity: int, *, overlap=None,
) -> List[_MOBlockOut]:
    """Turn a vibe-qc SCF result into TREXIO spin blocks.

    Restricted closed-shell: one block, occupations 2/0 (``mo.num`` is the
    number of spatial orbitals). Unrestricted, or restricted with
    ``n_alpha != n_beta`` (ROHF / ROKS): alpha then beta spin-orbital
    blocks with occupations 1/0, as the specification prescribes for UHF
    wave functions.
    """
    has_alpha = getattr(result, "mo_coeffs_alpha", None) is not None
    has_restricted = getattr(result, "mo_coeffs", None) is not None and not has_alpha
    if not (has_alpha or has_restricted):
        raise TypeError(
            "write_trexio: result must expose either `mo_coeffs` (restricted) "
            "or `mo_coeffs_alpha` + `mo_coeffs_beta` (unrestricted); got "
            + type(result).__name__
        )

    def energies(attr, n):
        values = getattr(result, attr, None)
        return np.full(n, np.nan) if values is None else _real_1d(values, n, attr)

    def occupations(attr, C, density, fallback, capacity):
        values = getattr(result, attr, None)
        if values is not None and np.asarray(values).size:
            return _real_1d(values, C.shape[1], attr)
        if density is not None and overlap is not None:
            # Preserve promoted and fractional SCF populations, including
            # ROHF/ROKS ensembles that expose densities but no occupation
            # aliases. Snap only numerical roundoff at integer populations.
            rdm = C.conj().T @ overlap @ np.asarray(density) @ overlap @ C
            values = np.real(np.diag(rdm)).copy()
            for integer in range(int(capacity) + 1):
                values[np.abs(values - integer) < 1e-12] = float(integer)
            return values
        return fallback

    if has_restricted:
        coeffs = _coefficient_2d(result.mo_coeffs, "MO coefficients")
        n_mo = coeffs.shape[1]
        orbital_energies = energies("mo_energies", n_mo)
        if (multiplicity == 1 and n_alpha == n_beta) or getattr(result, "trexio_spatial_orbitals", False):
            occ = np.zeros(n_mo)
            occ[:n_alpha] = 2.0
            occ = occupations("occupations", coeffs, getattr(result, "density", None), occ, 2.)
            return [_MOBlockOut(0, coeffs, orbital_energies, occ)]
        # Restricted open shell: the same spatial orbitals carry
        # different alpha / beta occupations, which the spin-free
        # spin-orbital layout represents exactly.
        occ_a = np.zeros(n_mo)
        occ_a[:n_alpha] = 1.0
        occ_b = np.zeros(n_mo)
        occ_b[:n_beta] = 1.0
        densities = spin_densities(result)
        occ_a = occupations("occupations_alpha", coeffs, densities[0], occ_a, 1.)
        occ_b = occupations("occupations_beta", coeffs, densities[1], occ_b, 1.)
        return [
            _MOBlockOut(0, coeffs, orbital_energies, occ_a),
            _MOBlockOut(1, coeffs, orbital_energies, occ_b),
        ]

    ca = _coefficient_2d(result.mo_coeffs_alpha, "alpha MO coefficients")
    cb = _coefficient_2d(result.mo_coeffs_beta, "beta MO coefficients")
    ea = energies("mo_energies_alpha", ca.shape[1])
    eb = energies("mo_energies_beta", cb.shape[1])
    occ_a = np.zeros(ca.shape[1])
    occ_a[:n_alpha] = 1.0
    occ_b = np.zeros(cb.shape[1])
    occ_b[:n_beta] = 1.0
    densities = spin_densities(result)
    occ_a = occupations("occupations_alpha", ca, densities[0], occ_a, 1.)
    occ_b = occupations("occupations_beta", cb, densities[1], occ_b, 1.)
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


def read_trexio_fields(path, *, backend="auto", fields=None, chunk_size=65536):
    """Read raw TREXIO fields in file units and order, without conversion.

    Keys are public TREXIO API suffixes, e.g. ``mo_coefficient`` or
    ``rdm_2e``. Sparse tensors are :class:`TrexioSparse`; buffered arrays
    are read in bounded chunks. Partial files and all basis types are valid.
    """
    trexio = _require_trexio()
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    with trexio.File(str(target), "r", _resolve_backend(trexio, backend)) as f:
        return read_fields(trexio, f, fields=fields, chunk_size=chunk_size)


def write_trexio_fields(path, fields, *, backend="hdf5", overwrite=True, chunk_size=65536):
    """Write raw fields in TREXIO conventions, replacing only after success.

    All groups exposed by the optional library are supported. Scalars and
    dimensions precede arrays. Sparse values require :class:`TrexioSparse`.
    File-owned version/unsafe metadata is regenerated; determinant count is
    validated against the list. No numerical data is inferred or converted.
    """
    trexio = _require_trexio()
    if str(backend).strip().lower() not in BACKENDS:
        raise ValueError("write_trexio: backend must be 'hdf5' or 'text'.")
    back_end = _resolve_backend(trexio, backend)
    target = Path(path)
    with atomic_target(trexio, target, overwrite=overwrite) as temporary:
        with trexio.File(str(temporary), "w", back_end) as f:
            write_fields(trexio, f, fields, chunk_size=chunk_size)
    return target


def _complex_field(fields, name, value):
    value = np.asarray(value)
    fields[name] = np.ascontiguousarray(value.real)
    if np.iscomplexobj(value) and np.any(value.imag):
        if not hasattr(_require_trexio(), "write_" + name + "_im"):
            # TREXIO 2.6 has complex MO/operator fields but real RDMs.
            # C^H S D S C can acquire harmless imaginary roundoff even for
            # a real diagonal MO density. Never discard a physical part.
            scale = max(1., float(np.max(np.abs(value), initial=0)))
            if np.max(np.abs(value.imag), initial=0) > 1e-12 * scale:
                raise ValueError(f"TREXIO: the installed format cannot represent complex {name}.")
            return
        fields[name + "_im"] = np.ascontiguousarray(value.imag)


def _periodic_blocks(result, n_alpha, n_beta, multiplicity):
    """Normalize Gamma, per-k list and GPW *_k result layouts."""
    names = ("mo_coeffs", "mo_energies", "occupations", "mo_occupations", "mo_coeffs_alpha",
             "mo_coeffs_beta", "mo_energies_alpha", "mo_energies_beta",
             "occupations_alpha", "occupations_beta")
    values = {}
    for name in names:
        value = getattr(result, name + "_k", None)
        if value is None:
            value = getattr(result, name, None)
        if value is not None:
            values[name] = value
    coefficient = values.get("mo_coeffs_alpha", values.get("mo_coeffs"))
    if coefficient is None:
        raise ValueError("TREXIO: periodic result has no Gaussian MO coefficients.")
    per_k = isinstance(coefficient, (list, tuple)) or np.asarray(coefficient).ndim == 3
    count = len(coefficient) if per_k else 1
    out = []
    for k in range(count):
        local = {name: (value[k] if per_k and len(value) else value)
                 for name, value in values.items()}
        restricted_occ = local.pop("mo_occupations", None)
        if restricted_occ is not None:
            # ROHF/ROKS carry common spatial occupations, including a
            # fractionally occupied degenerate open shell. Their UHF-style
            # MO aliases do not always expose spin-occupation aliases.
            # Match the producers' _spin_occupations convention exactly.
            restricted_occ = np.asarray(restricted_occ)
            local.setdefault("occupations", restricted_occ)
            local.setdefault("occupations_alpha", np.minimum(restricted_occ, 1.))
            local.setdefault("occupations_beta", np.maximum(restricted_occ - 1., 0.))
        out.append(_collect_mo_blocks(SimpleNamespace(**local), n_alpha, n_beta, multiplicity))
    return out


def _periodic_sampling(result, count, kpoints, weights):
    if kpoints is None:
        for name in ("kpoints_cart", "restart_kpoints"):
            kpoints = getattr(result, name, None)
            if kpoints is not None:
                break
    if weights is None:
        for name in ("kpoint_weights", "restart_weights"):
            weights = getattr(result, name, None)
            if weights is not None:
                break
    mesh = getattr(result, "kmesh", None) or getattr(result, "restart_mesh", None)
    if mesh is not None:
        kpoints = mesh.kpoints if kpoints is None else kpoints
        weights = mesh.weights if weights is None else weights
    if count == 1:
        kpoints = [[0., 0., 0.]] if kpoints is None else kpoints
        weights = [1.] if weights is None else weights
    points, weights = np.asarray(kpoints, dtype=float), np.asarray(weights, dtype=float)
    if points.shape != (count, 3) or not np.all(np.isfinite(points)):
        raise ValueError("TREXIO: supply Cartesian kpoints with shape (n_k, 3), in inverse bohr.")
    if (weights.shape != (count,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("TREXIO: k-point weights must be nonnegative and sum to one.")
    return points, weights


def write_trexio(
    path, molecule, basis, result, *, backend="hdf5", description="",
    author=None, mo_type=None, write_integrals=True, uses_ecp=False,
    overwrite=True, ecp_source=None, system=None, kpoints=None, weights=None,
    write_rdm=True, write_mo_integrals=False, write_eri=False,
    extra_fields=None, ci_result=None, n_core=0, root=0,
) -> Path:
    """Export a Gaussian molecular or periodic wavefunction.

    ``system`` is a PeriodicSystem; ``kpoints`` are Cartesian inverse-bohr
    coordinates and ``weights`` are normalized Brillouin-zone weights.
    ECP parameters come from the result, or explicit applied ``ecp_source``
    options. All coordinates and energies are in atomic units.

    ``write_integrals`` writes molecular one-electron integrals (including
    ECP), or available Gamma overlap/core matrices. Multi-k AO matrices
    cannot fit TREXIO's AO matrix dimensions and are omitted. Optional
    ``write_mo_integrals`` writes molecular one-electron MO matrices;
    ``write_eri`` computes molecular AO ERIs (quartic memory, opt-in).
    ``write_rdm`` stores the one-particle MO density for molecular results.
    ``extra_fields`` supplies additional fields in TREXIO conventions;
    collisions with generated fields are errors. Use ``write_rdm=False``
    when providing a correlated RDM explicitly. Writes preserve an existing
    target if validation or writing fails.
    """
    _require_trexio()
    ci_payload = getattr(result, "trexio_wavefunction", None)
    if ci_result is not None and ci_payload is not None:
        raise ValueError("TREXIO: ci_result requires an SCF reference; this result already carries a CI wavefunction.")
    if ci_result is not None or ci_payload is not None:
        from ._trexio_ci import ci_wavefunction
        if system is not None:
            raise ValueError("TREXIO: determinant export requires molecular common spatial orbitals.")
        if ci_payload is None:
            C = np.asarray(result.mo_coeffs)
            if hasattr(ci_result, "cas"):
                C = C @ np.asarray(ci_result.mo_rotation)
                ci_result = ci_result.cas
            ci_payload = dict(coefficients=C, ci=ci_result, n_core=n_core, reference=result, root=root)
        else:
            ci_payload = dict(ci_payload, root=root)
        result = ci_wavefunction(**ci_payload)
        ecp_source = result.trexio_ecp_source if ecp_source is None else ecp_source
        supplied = dict(extra_fields or {})
        if supplied.keys() & result.trexio_fields.keys():
            raise ValueError("TREXIO: CI-generated fields collide with extra_fields.")
        supplied.update(result.trexio_fields)
        extra_fields, write_rdm = supplied, False
        mo_type = mo_type or "CI"
    from ..._vibeqc_core import compute_kinetic, compute_nuclear, compute_overlap
    from ._trexio_ecp import ecp_fields

    if system is not None:
        molecule = system.unit_cell_molecule()
    elif (kpoints is not None or weights is not None or any(
            getattr(result, name, None) is not None for name in
            ("restart_lattice", "kpoints_cart", "restart_kpoints", "kmesh",
             "mo_coeffs_k", "mo_coeffs_alpha_k"))):
        raise ValueError("TREXIO: periodic results require system= to preserve their lattice.")
    if system is not None and (write_eri or write_mo_integrals):
        raise ValueError("TREXIO: computed ERIs/MO integrals currently require a molecular result.")
    atoms = list(molecule.atoms)
    n_atoms = len(atoms)
    coords = np.asarray([atom.xyz for atom in atoms], dtype=float).reshape(n_atoms, 3)
    labels = [_element(int(atom.Z)) for atom in atoms]
    source = result if ecp_source is None else ecp_source
    fields, charges = ecp_fields(molecule, source, required=uses_ecp)
    n_electrons = int(molecule.n_electrons()) - int(np.sum(fields.get("ecp_z_core", 0)))
    multiplicity = int(molecule.multiplicity)
    n_alpha = (n_electrons + multiplicity - 1) // 2
    n_beta = n_electrons - n_alpha
    if hasattr(result, "trexio_spin_counts") and result.trexio_spin_counts != (n_alpha, n_beta):
        raise ValueError("TREXIO: CI determinants disagree with the molecule's spin populations.")
    if n_beta < 0 or n_alpha - n_beta != multiplicity - 1:
        raise ValueError(f"TREXIO: {n_electrons} electrons cannot realise multiplicity {multiplicity}.")

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
        if not np.allclose(origin, coords[atom_index], rtol=0, atol=1.0e-10):
            raise ValueError(
                f"write_trexio: shell {s} is centred at {origin.tolist()} "
                f"but atom {atom_index} sits at {coords[atom_index].tolist()}; "
                "TREXIO basis functions are nucleus-centred; represent "
                "floating functions using explicit zero-charge centers."
            )
        L = int(shell.l)
        if len(shell.exponents) != len(shell.coefficients):
            raise ValueError(f"TREXIO: shell {s} has inconsistent primitive arrays.")
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

    # Molecular integrals must reproduce the actual all-electron/ECP Hamiltonian.
    matrices = {}
    overlap = None
    nuclear_repulsion = None
    if system is None:
        overlap = np.asarray(compute_overlap(basis), dtype=float)
        scf_count = _scf_electron_count(result, overlap)
        if scf_count is not None and abs(scf_count - n_electrons) > _ELECTRON_COUNT_TOL:
            raise ValueError(f"TREXIO: the SCF density integrates to {scf_count:.6f} electrons "
                             f"but the molecule has {n_electrons}; effective core potential "
                             "parameters may be missing.")
        if "ecp_num" in fields:
            from ...ecp_metadata import one_electron_hamiltonian
            from ..._vibeqc_core import compute_nuclear_with_charges
            hcore, nuclear_repulsion, _, _ = one_electron_hamiltonian(molecule, basis, source)
            kinetic = np.asarray(compute_kinetic(basis))
            potential = np.asarray(compute_nuclear_with_charges(basis, coords.tolist(), charges.tolist()))
            matrices = {"overlap": overlap, "kinetic": kinetic, "potential_n_e": potential,
                        "ecp": hcore - kinetic - potential, "core_hamiltonian": hcore}
        else:
            nuclear_repulsion = float(molecule.nuclear_repulsion())
            if write_integrals or write_mo_integrals:
                kinetic, potential = np.asarray(compute_kinetic(basis)), np.asarray(compute_nuclear(basis, molecule))
                matrices = {"overlap": overlap, "kinetic": kinetic, "potential_n_e": potential,
                            "core_hamiltonian": kinetic + potential}
        by_k = [_collect_mo_blocks(result, n_alpha, n_beta, multiplicity, overlap=overlap)]
        points, k_weights = np.zeros((1, 3)), np.ones(1)
    else:
        by_k = _periodic_blocks(result, n_alpha, n_beta, multiplicity)
        points, k_weights = _periodic_sampling(result, len(by_k), kpoints, weights)
        lattice = np.asarray(system.lattice, dtype=float)
        reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
        for i, axis in enumerate(("a", "b", "c")):
            fields["cell_" + axis] = lattice[:, i]
            fields["cell_g_" + axis] = reciprocal[:, i]
        fields.update(cell_two_pi=1, pbc_periodic=1, pbc_k_point_num=len(by_k),
                      pbc_k_point=points @ np.linalg.inv(reciprocal).T,
                      pbc_k_point_weight=k_weights)
        # TREXIO k points are reduced coordinates in its reciprocal basis.
        description += f" vibe-qc periodic_dim={int(system.dim)};"
        nuclear_repulsion = getattr(result, "e_nuclear", None)
        if len(by_k) == 1 and np.allclose(points[0], 0., rtol=0, atol=1e-12):
            for attr, name in (("overlap", "overlap"), ("hcore", "core_hamiltonian")):
                value = getattr(result, attr + "_k", None)
                if value is None or np.asarray(value).size == 0:
                    value = getattr(result, attr, None)
                if value is not None:
                    value = np.asarray(value)
                    if value.shape == (1, ao_num, ao_num):
                        value = value[0]
                    if value.shape == (ao_num, ao_num):
                        matrices[name] = value

    blocks = [block for group in by_k for block in group]
    if any(block.coefficients.shape[0] != ao_num for block in blocks):
        raise ValueError("TREXIO: MO coefficient AO dimension disagrees with the basis.")
    mo_coefficient = np.concatenate([b.coefficients[ao_perm, :].T for b in blocks])
    mo_energy = np.concatenate([b.energies for b in blocks])
    mo_occupation = np.concatenate([b.occupations for b in blocks])
    mo_spin = np.concatenate([np.full(b.coefficients.shape[1], b.spin, dtype=int) for b in blocks])
    mo_kpoint = np.concatenate([np.full(b.coefficients.shape[1], k, dtype=int)
                               for k, group in enumerate(by_k) for b in group])
    occupied = float(np.dot(mo_occupation, k_weights[mo_kpoint]))
    spin_resolved = any(b.spin == 1 for b in blocks)
    capacity = 1. if spin_resolved else 2.
    if (not np.all(np.isfinite(mo_occupation)) or np.any(mo_occupation < 0)
            or np.any(mo_occupation > capacity + _ELECTRON_COUNT_TOL)
            or abs(occupied - n_electrons) > _ELECTRON_COUNT_TOL):
        raise ValueError(f"TREXIO: MO occupations sum to {occupied:.6f} electrons, expected {n_electrons}.")
    if spin_resolved:
        for spin, expected in ((0, n_alpha), (1, n_beta)):
            sel = mo_spin == spin
            count = np.dot(mo_occupation[sel], k_weights[mo_kpoint[sel]])
            if abs(count - expected) > _ELECTRON_COUNT_TOL:
                raise ValueError(f"TREXIO: spin {spin} occupations contain {count} electrons, expected {expected}.")
    mo_num = len(mo_energy)
    provenance = _vibeqc_provenance()
    full_description = (f"{description.strip()} Written by vibe-qc {provenance['version']} "
                        f"(git {provenance['sha']}); {_mo_type_label(result)} orbitals, "
                        f"spherical AOs in TREXIO order, basis {getattr(basis, 'name', '?')}.").strip()
    temperature = float(getattr(result, "smearing_temperature", 0.) or 0.)
    if temperature:
        full_description += f" electronic_temperature_hartree={temperature:.17g};"
    fields.update(metadata_code_num=1, metadata_code=[f"vibe-qc {provenance['version']}"],
                  metadata_description=full_description,
                  nucleus_num=n_atoms, nucleus_charge=charges, nucleus_coord=coords, nucleus_label=labels,
                  electron_num=n_electrons, electron_up_num=n_alpha, electron_dn_num=n_beta,
                  basis_type="Gaussian", basis_shell_num=shell_num, basis_prim_num=prim_num,
                  basis_nucleus_index=nucleus_index, basis_shell_ang_mom=shell_ang_mom,
                  basis_shell_factor=shell_factor, basis_r_power=r_power, basis_shell_index=shell_index,
                  basis_exponent=exponents, basis_coefficient=coefficients, basis_prim_factor=prim_factor,
                  ao_cartesian=0, ao_num=ao_num, ao_shell=ao_shell, ao_normalization=ao_normalization,
                  mo_type=mo_type or _mo_type_label(result), mo_num=mo_num,
                  mo_occupation=mo_occupation, mo_spin=mo_spin,
                  mo_class=["Inactive" if occ > 0 else "Virtual" for occ in mo_occupation])
    if np.all(np.isfinite(mo_energy)):
        fields["mo_energy"] = mo_energy
    if hasattr(result, "trexio_classes"):
        fields["mo_class"] = result.trexio_classes
    _complex_field(fields, "mo_coefficient", mo_coefficient)
    if system is not None:
        fields["mo_k_point"] = mo_kpoint
    if author:
        fields.update(metadata_author_num=1, metadata_author=[str(author)])
    if nuclear_repulsion is not None:
        fields["nucleus_repulsion"] = float(nuclear_repulsion)
    energy = getattr(result, "energy", None)
    if energy is not None:
        fields.update(state_num=getattr(result, "trexio_state_num", 1),
                      state_id=getattr(result, "trexio_state", 0),
                      state_current_label=(f"state {result.trexio_state}" if hasattr(result, "trexio_state") else "ground state"),
                      state_energy=float(energy))
    if write_integrals:
        for name, matrix in matrices.items():
            _complex_field(fields, "ao_1e_int_" + name, matrix[np.ix_(ao_perm, ao_perm)])
    if write_mo_integrals:
        for name, matrix in matrices.items():
            transformed = np.zeros((mo_num, mo_num), dtype=mo_coefficient.dtype)
            offset = 0
            for block in blocks:
                n = block.coefficients.shape[1]
                transformed[offset:offset+n, offset:offset+n] = block.coefficients.conj().T @ matrix @ block.coefficients
                offset += n
            _complex_field(fields, "mo_1e_int_" + name, transformed)
    if write_rdm and system is None:
        # Use the carried density, including correlated/natural-orbital
        # densities, rather than substituting an Aufbau determinant.
        densities = spin_densities(result)
        rdm = np.zeros((mo_num, mo_num), dtype=mo_coefficient.dtype)
        offset = 0
        spin_rdms = [np.zeros_like(rdm), np.zeros_like(rdm)]
        for block in blocks:
            n = block.coefficients.shape[1]
            density = (densities[block.spin] if len(blocks) == 2 else getattr(result, "density", None))
            if density is None:
                value = np.diag(block.occupations)
            else:
                C = block.coefficients
                value = C.conj().T @ overlap @ np.asarray(density) @ overlap @ C
            rdm[offset:offset+n, offset:offset+n] = value
            if len(blocks) == 2:
                spin_rdms[block.spin][offset:offset+n, offset:offset+n] = value
            offset += n
        _complex_field(fields, "rdm_1e", rdm)
        if len(blocks) == 2:
            _complex_field(fields, "rdm_1e_up", spin_rdms[0])
            _complex_field(fields, "rdm_1e_dn", spin_rdms[1])
    if write_eri:
        from ..._vibeqc_core import compute_eri
        eri = np.asarray(compute_eri(basis)).reshape((ao_num,) * 4)
        # Native compute_eri is chemists' (pq|rs); TREXIO is <pr|qs>.
        eri = eri[np.ix_(ao_perm, ao_perm, ao_perm, ao_perm)].transpose(0, 2, 1, 3)
        fields["ao_2e_int_eri"] = TrexioSparse.from_dense(eri)
    if extra_fields:
        collisions = fields.keys() & extra_fields.keys()
        if collisions:
            raise ValueError(f"TREXIO extra_fields collide with generated fields: {sorted(collisions)}")
        fields.update(extra_fields)
    return write_trexio_fields(path, fields, backend=backend, overwrite=overwrite)


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
    k_point: int = 0


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
    fields: Dict[str, Any] = field(default_factory=dict)
    ao_normalization: Optional[np.ndarray] = None
    mo_k_point: Optional[np.ndarray] = None

    def write(self, path, *, backend="hdf5", overwrite=True):
        """Write the raw ``fields`` mapping, preserving all imported groups.

        Edit ``fields`` to change the file. The named attributes are derived
        views for native conversion and are not the serialization authority.
        """
        return write_trexio_fields(path, self.fields, backend=backend, overwrite=overwrite)

    @property
    def periodic(self):
        return bool(self.fields.get("pbc_periodic", 0))

    @property
    def lattice(self):
        """Real-space lattice in bohr, columns are vectors, or None."""
        if not all("cell_" + axis in self.fields for axis in ("a", "b", "c")):
            return None
        return np.column_stack([self.fields["cell_" + axis] for axis in ("a", "b", "c")])

    @property
    def kpoints(self):
        """Cartesian k points in inverse bohr, or None."""
        if "pbc_k_point" not in self.fields:
            return None
        if all("cell_g_" + axis in self.fields for axis in ("a", "b", "c")):
            reciprocal = np.column_stack([self.fields["cell_g_" + a] for a in ("a", "b", "c")])
            if not self.fields.get("cell_two_pi", 0):
                reciprocal = 2 * np.pi * reciprocal
        elif self.lattice is not None:
            reciprocal = 2 * np.pi * np.linalg.inv(self.lattice).T
        else:
            raise ValueError("TREXIO: Cartesian k points require a cell.")
        return np.asarray(self.fields["pbc_k_point"]) @ reciprocal.T

    @property
    def kpoint_weights(self):
        return self.fields.get("pbc_k_point_weight")

    @property
    def atomic_numbers(self):
        # nucleus.charge is Z_eff when an ECP replaces core electrons.
        return self.charges + np.asarray(self.fields.get("ecp_z_core", np.zeros(len(self.charges))))

    def periodic_system(self, *, dim=None):
        """Rebuild a periodic system. Foreign files default to three dimensions."""
        import re
        from ..._vibeqc_core import PeriodicSystem
        if not self.periodic or self.lattice is None:
            raise ValueError("TREXIO: a periodic system requires pbc.periodic and all cell vectors.")
        if dim is None:
            match = re.search(r"vibe-qc periodic_dim=([123]);", self.metadata.get("description", ""))
            dim = int(match.group(1)) if match else 3
        return PeriodicSystem(dim=dim, lattice=self.lattice, unit_cell=self.molecule().atoms,
                              charge=self.charge, multiplicity=self.multiplicity)

    def ecp_options(self, options=None):
        """Populate molecular or periodic SCF options with the stored ECP."""
        from ._trexio_ecp import ecp_options
        return ecp_options(self.fields, self.coords, options)

    def restart_result(self):
        """Return a density-bearing source accepted by READ guess drivers."""
        if "pbc_periodic" not in self.fields and any(
                name.startswith(("cell_", "pbc_")) for name in self.fields):
            raise ValueError("TREXIO: restart of cell data requires an explicit pbc.periodic flag.")
        blocks = self.mo_blocks()
        densities = self.density_matrices()
        if not blocks:
            raise ValueError("TREXIO: restart requires molecular orbitals.")
        values = {"restart_basis": self.basis_set()}
        if self.periodic:
            points, weights = self.kpoints, self.kpoint_weights
            if points is None or weights is None:
                raise ValueError("TREXIO: periodic restart requires k points and weights.")
            values.update(restart_lattice=self.lattice, restart_kpoints=points, restart_weights=weights)
            expected = list(range(len(points)))
        else:
            expected = [0]
        spins = sorted(set(b.spin for b in blocks))
        if spins not in ([0], [0, 1]):
            raise ValueError("TREXIO: restart requires a complete restricted or alpha/beta orbital set.")
        for spin in spins:
            if [b.k_point for b in blocks if b.spin == spin] != expected:
                raise ValueError("TREXIO: restart requires every k point in each spin channel.")
        if spins == [0] and all(name in self.fields for name in ("rdm_1e_up", "rdm_1e_dn")):
            for spin, name in (("alpha", "rdm_1e_up"), ("beta", "rdm_1e_dn")):
                spin_rdm = self._block_rdm(name)
                channel = []
                for block in blocks:
                    sel = (self.mo_spin == 0) & (self.mo_k_point == block.k_point)
                    C = block.coefficients
                    channel.append(C @ spin_rdm[np.ix_(sel, sel)] @ C.conj().T)
                values["density_" + spin] = channel if self.periodic else channel[0]
            return SimpleNamespace(**values)
        for spin in spins:
            pairs = [(b, d) for b, d in zip(blocks, densities) if b.spin == spin]
            suffix = ("_alpha" if spin == 0 else "_beta") if len(spins) == 2 else ""
            for name, data in (("density", [d for _, d in pairs]),
                               ("mo_coeffs", [b.coefficients for b, _ in pairs]),
                               ("occupations", [b.occupations for b, _ in pairs])):
                values[name + suffix] = data if self.periodic else data[0]
        return SimpleNamespace(**values)

    def _require_native_basis(self):
        if not self.shell_ang_mom or self.fields.get("basis_type") != "Gaussian":
            raise ValueError("TREXIO: native basis reconstruction requires a Gaussian basis.")
        if self.fields.get("ao_cartesian", 0):
            raise ValueError("TREXIO: Cartesian AOs are preserved in fields; native reconstruction requires spherical AOs.")
        if any(np.any(self.fields.get(name, 0)) for name in ("basis_coefficient_im", "basis_exponent_im")):
            raise ValueError("TREXIO: native basis reconstruction requires real Gaussian primitives.")
        if np.any(self.fields.get("basis_r_power", 0)):
            raise ValueError("TREXIO: native Gaussian reconstruction requires basis.r_power=0.")
        required = {"basis_nucleus_index", "basis_shell_index", "basis_exponent", "basis_coefficient"}
        if self.fields and not required <= self.fields.keys():
            raise ValueError(f"TREXIO: incomplete Gaussian basis: {sorted(required - self.fields.keys())}")
        if any(i < 0 or i >= len(self.coords) for i in self.shell_atom):
            raise ValueError("TREXIO: basis nucleus index out of range.")
        if any(i < 0 or i >= len(self.shell_ang_mom) for i in self.prim_shell):
            raise ValueError("TREXIO: primitive shell index out of range.")
        if np.any(self.exponents <= 0) or not np.all(np.isfinite(self.exponents)):
            raise ValueError("TREXIO: Gaussian exponents must be finite and positive.")
        if any(not np.all(np.isfinite(values)) for values in
               (self.coords, self.shell_factor, self.prim_factor, self.coefficients)):
            raise ValueError("TREXIO: Gaussian positions, coefficients and normalization factors must be finite.")
        if set(self.prim_shell) != set(range(len(self.shell_ang_mom))):
            raise ValueError("TREXIO: every Gaussian shell must have primitives.")


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

        if not len(self.charges) or self.coords.shape != (len(self.charges), 3):
            raise ValueError("TREXIO: molecule reconstruction requires nucleus charges and coordinates.")
        if not np.allclose(self.atomic_numbers, np.rint(self.atomic_numbers), rtol=0, atol=1e-10):
            raise ValueError("TREXIO: noninteger nuclear charges cannot become atomic numbers.")
        if self.fields and not {"electron_up_num", "electron_dn_num"} <= self.fields.keys():
            raise ValueError("TREXIO: molecule reconstruction requires spin-resolved electron counts.")
        atoms = [
            Atom(int(round(float(z))), [float(x) for x in xyz])
            for z, xyz in zip(self.atomic_numbers, self.coords)
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

        self._require_native_basis()
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

        self._require_native_basis()
        mol = molecule if molecule is not None else self.molecule()
        return BasisSet(mol, self.shells(), name, True)

    def mo_blocks(self) -> List[TrexioMOBlock]:
        """MO coefficients per k point and spin in native AO order.

        Fold ``ao.normalization`` into the coefficients; native shell
        primitives already contain the radial factors. Complex values stay
        complex. No orbitals or energies are fabricated for partial files.
        """
        self._require_native_basis()
        if "ao_cartesian" not in self.fields:
            raise ValueError("TREXIO: native MO conversion requires an explicit ao.cartesian flag.")
        n_ao = len(self.ao_perm)
        if self.mo_coefficient.shape != (len(self.mo_spin), n_ao) or not n_ao:
            raise ValueError("TREXIO: MO conversion requires a complete AO/MO layout.")
        norm = np.ones(n_ao) if self.ao_normalization is None else self.ao_normalization
        if not np.all(np.isfinite(norm)) or np.any(norm == 0):
            raise ValueError("TREXIO: AO normalizations must be finite and nonzero.")
        kpoints = np.zeros(len(self.mo_spin), dtype=int) if self.mo_k_point is None else self.mo_k_point
        blocks = []
        for k in sorted(set(kpoints.tolist())):
            for spin in sorted(set(self.mo_spin[kpoints == k].tolist())):
                sel = np.flatnonzero((self.mo_spin == spin) & (kpoints == k))
                stored = self.mo_coefficient[sel, :] * norm
                libint = np.empty((n_ao, len(sel)), dtype=stored.dtype)
                libint[self.ao_perm, :] = stored.T
                blocks.append(TrexioMOBlock(
                    spin=int(spin), k_point=int(k), coefficients=libint,
                    energies=self.mo_energy[sel], occupations=self.mo_occupation[sel],
                    classes=[self.mo_class[int(i)] for i in sel]))
        return blocks

    def to_libint_order(self, matrix: np.ndarray) -> np.ndarray:
        """Convert an AO operator matrix, including AO normalization factors."""
        self._require_native_basis()
        if "ao_cartesian" not in self.fields:
            raise ValueError("TREXIO: native AO conversion requires an explicit ao.cartesian flag.")
        arr = np.asarray(matrix)
        if arr.shape != (len(self.ao_perm), len(self.ao_perm)):
            raise ValueError("TREXIO: expected a square AO matrix.")
        norm = np.ones(len(self.ao_perm)) if self.ao_normalization is None else self.ao_normalization
        if not np.all(np.isfinite(norm)) or np.any(norm == 0):
            raise ValueError("TREXIO: AO normalizations must be finite and nonzero.")
        out = np.empty_like(arr, dtype=np.result_type(arr, float))
        out[np.ix_(self.ao_perm, self.ao_perm)] = arr / np.outer(norm, norm)
        return out

    def _block_rdm(self, name):
        rdm = _field_complex(self.fields, name)
        if rdm is None:
            return None
        n = len(self.mo_spin)
        if (rdm.shape != (n, n) or not np.all(np.isfinite(rdm))
                or not np.allclose(rdm, rdm.conj().T, rtol=0, atol=1e-10)):
            raise ValueError(f"TREXIO: {name} must be a finite Hermitian MO matrix.")
        same_block = ((self.mo_spin[:, None] == self.mo_spin[None, :])
                      & (self.mo_k_point[:, None] == self.mo_k_point[None, :]))
        if np.any(np.abs(rdm[~same_block]) > 1e-12):
            raise ValueError(f"TREXIO: {name} contains coherence between spin/k-point blocks; "
                             "native block densities cannot represent it.")
        return rdm

    def density_matrices(self):
        """AO densities per returned MO block, using the stored RDM if present.

        Each block is unweighted; periodic integration applies kpoint_weights.
        Missing occupations and missing RDMs are an error.
        """
        result = []
        rdm = self._block_rdm("rdm_1e")
        if rdm is None and all(name in self.fields for name in ("rdm_1e_up", "rdm_1e_dn")):
            rdm = self._block_rdm("rdm_1e_up") + self._block_rdm("rdm_1e_dn")
        for block in self.mo_blocks():
            if rdm is None:
                if not np.all(np.isfinite(block.occupations)):
                    raise ValueError("TREXIO: a density requires occupations or rdm_1e.")
                density = (block.coefficients * block.occupations) @ block.coefficients.conj().T
            else:
                sel = (self.mo_spin == block.spin) & (self.mo_k_point == block.k_point)
                density = block.coefficients @ rdm[np.ix_(sel, sel)] @ block.coefficients.conj().T
            result.append(density)
        return result


def _field_complex(fields, name):
    if name not in fields:
        if name + "_im" in fields:
            raise ValueError(f"TREXIO: {name}_im is present without its real part.")
        return None
    array = np.asarray(fields[name])
    if name + "_im" in fields:
        array = array + 1j * np.asarray(fields[name + "_im"])
    return array


def read_trexio(path: os.PathLike | str, *, backend: str = "auto") -> TrexioData:
    """Read every present TREXIO group and expose native conversion helpers.

    ``fields`` retains raw data in TREXIO conventions, including sparse
    integrals, ECPs, complex coefficients, RDMs and non-Gaussian bases. Only
    requests to construct native objects impose vibe-qc basis restrictions.
    Optional energies/occupations are NaN in the convenience views when absent.
    """
    fields = read_trexio_fields(path, backend=backend)
    n_atoms, shell_num, prim_num, ao_num, mo_num = (
        int(fields.get(name, 0)) for name in
        ("nucleus_num", "basis_shell_num", "basis_prim_num", "ao_num", "mo_num"))
    charges = np.asarray(fields.get("nucleus_charge", np.empty(0)), dtype=float)
    coords = np.asarray(fields.get("nucleus_coord", np.empty((0, 3))), dtype=float)
    shell_ang_mom = list(fields.get("basis_shell_ang_mom", []))
    ao_perm = np.empty(0, dtype=int)
    if shell_ang_mom and not fields.get("ao_cartesian", 0) and "ao_shell" in fields:
        if min(shell_ang_mom) < 0 or max(shell_ang_mom) > 20:
            raise ValueError("TREXIO: invalid spherical angular momentum.")
        ao_shell = np.asarray(fields["ao_shell"])
        if ao_shell.shape != (ao_num,) or np.any(ao_shell < 0) or np.any(ao_shell >= shell_num):
            raise ValueError("TREXIO: invalid ao.shell mapping.")
        # AO shells may be interleaved; occurrence order within each shell
        # remains TREXIO's m=0,+1,-1,... order.
        ao_perm = np.empty(ao_num, dtype=int)
        offset = 0
        for shell, L in enumerate(shell_ang_mom):
            sel = np.flatnonzero(ao_shell == shell)
            if len(sel) != 2 * L + 1:
                raise ValueError("TREXIO: each spherical shell must contain 2l+1 AOs.")
            ao_perm[sel] = offset + np.asarray(_pure_trexio_to_libint(int(L)))
            offset += len(sel)
    spin = np.asarray(fields.get("mo_spin", np.zeros(mo_num)), dtype=int)
    kpoint = np.asarray(fields.get("mo_k_point", np.zeros(mo_num)), dtype=int)
    if np.any((spin != 0) & (spin != 1)):
        raise ValueError("TREXIO: mo.spin must be 0 or 1.")
    if np.any(kpoint < 0) or np.any(kpoint >= int(fields.get("pbc_k_point_num", 1))):
        raise ValueError("TREXIO: mo.k_point is outside the stored k-point mesh.")
    coefficient = _field_complex(fields, "mo_coefficient")
    numbers = charges + np.asarray(fields.get("ecp_z_core", np.zeros(len(charges))))
    return TrexioData(
        labels=list(fields.get("nucleus_label", [_element(int(round(z))) for z in numbers])),
        charges=charges, coords=coords,
        n_up=int(fields.get("electron_up_num", 0)), n_dn=int(fields.get("electron_dn_num", 0)),
        shell_atom=list(fields.get("basis_nucleus_index", [])), shell_ang_mom=shell_ang_mom,
        shell_factor=np.asarray(fields.get("basis_shell_factor", np.ones(shell_num))),
        prim_shell=list(fields.get("basis_shell_index", [])),
        exponents=np.asarray(fields.get("basis_exponent", [])),
        coefficients=np.asarray(fields.get("basis_coefficient", [])),
        prim_factor=np.asarray(fields.get("basis_prim_factor", np.ones(prim_num))),
        ao_perm=ao_perm, mo_type=str(fields.get("mo_type", "MO")),
        mo_coefficient=np.empty((0, ao_num)) if coefficient is None else coefficient,
        mo_energy=np.asarray(fields.get("mo_energy", np.full(mo_num, np.nan))),
        mo_occupation=np.asarray(fields.get("mo_occupation", np.full(mo_num, np.nan))),
        mo_spin=spin, mo_k_point=kpoint, mo_class=list(fields.get("mo_class", [""] * mo_num)),
        ao_normalization=np.asarray(fields.get("ao_normalization", np.ones(ao_num))),
        overlap=_field_complex(fields, "ao_1e_int_overlap"),
        kinetic=_field_complex(fields, "ao_1e_int_kinetic"),
        potential_n_e=_field_complex(fields, "ao_1e_int_potential_n_e"),
        core_hamiltonian=_field_complex(fields, "ao_1e_int_core_hamiltonian"),
        nuclear_repulsion=fields.get("nucleus_repulsion"), energy=fields.get("state_energy"),
        metadata={k[9:]: v for k, v in fields.items() if k.startswith("metadata_")}, fields=fields,
    )
