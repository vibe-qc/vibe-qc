"""Initial-guess helpers for the Python periodic SCF drivers.

Wraps the C++ ``GuessEngine`` so the dispatch logic lives in one place.
The Python periodic drivers (``periodic_*_ewald*.py``,
``periodic_rhf_gdf.py``) previously each inlined an
``if guess == InitialGuess.SAD: ... else: ...`` block; this module
collapses all of them onto two helpers that mirror the engine's
``build_closed_shell`` / ``build_open_shell`` entry points.

The shared route registry declares each concrete construction and restart
capability. The native resolver selects SAD for ordinary periodic AUTO, or
HCORE for adapters exposing only that construction. ``READ`` payloads are
validated in Python before driver setup. Callers that pass the periodic system and lattice
options also get ``SAP`` through the lattice SAP Fock helper, ``MINAO`` through
the periodic projection helper, and ``HUECKEL`` through the lattice GWH Fock
helper. Route-local periodic drivers implement ``PATOM`` with their in-field
Fock step before calling this generic helper.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    InitialGuess,
    GuessSelection,
    LatticeSumOptions,
    Molecule,
    PeriodicSystem,
    _resolve_initial_guess_for_molecule,
    bloch_sum,
    compute_huckel_fock_lattice,
    compute_kinetic_lattice,
    compute_minao_density_periodic,
    compute_overlap_lattice,
    compute_vsap_lattice,
    _guess_closed_shell_density,
    _guess_open_shell_density,
    _normalize_guess_density,
    _normalize_guess_spin_densities,
    _normalize_atomic_guess,
    compute_overlap,
)

from ._initial_guess import coerce_initial_guess



# Concrete construction capabilities, shared by direct drivers and public
# preflight. AUTO is resolved against this set by the native policy.
_PERIODIC_ATOMIC_GUESSES = frozenset((
    InitialGuess.HCORE, InitialGuess.SAD, InitialGuess.SAP,
    InitialGuess.PATOM, InitialGuess.HUECKEL, InitialGuess.MINAO,
))


def copy_initial_guess_options(source, target):
    """Preserve source payload and Hamiltonian context across option adapters.

    Targets without a source contract reject populated fields instead of
    silently erasing them. Empty fields remain compatible with older shapes.
    """
    from ._vibeqc_core import SpinlockMode

    target.initial_guess = coerce_initial_guess(source.initial_guess)
    fields = (
        "read_density", "read_density_alpha", "read_density_beta", "read_density_k", "read_path",
        "atomic_spins", "spinlock_mode", "spinlock_value", "spinlock_iterations",
        "ecp_centers", "ecp_library", "ecp_primitive_blocks",
        "ecp_primitive_centers", "ecp_home_centers", "ecp_effective_charges",
        "ecp_total_ncore",
    )
    for name in fields:
        if not hasattr(source, name):
            continue
        value = getattr(source, name)
        if hasattr(target, name):
            setattr(target, name, value)
            continue
        if name == "spinlock_mode":
            populated = value != SpinlockMode.OFF
        elif isinstance(value, np.ndarray):
            populated = value.size != 0
        else:
            populated = bool(value)
        if populated:
            raise NotImplementedError(
                f"{type(target).__name__} cannot carry initial-guess field {name}"
            )
    return target


def select_periodic_driver_guess(
    system, options, *, route, method, driver, multi_k=False,
    restart_supplied=False,
):
    """Shared preflight for option-based native/Ewald public entry points."""
    from ._vibeqc_core import SpinlockMode

    requested = coerce_initial_guess(options.initial_guess)
    is_open_shell = method in ("UHF", "UKS", "ROHF", "ROKS")
    if not is_open_shell:
        if getattr(options, "atomic_spins", ()):
            raise NotImplementedError(
                f"{driver}: restricted closed-shell SCF cannot represent atomic_spins"
            )
    if (method in ("RHF", "RKS", "ROHF", "ROKS")
            and getattr(options, "spinlock_mode", SpinlockMode.OFF) != SpinlockMode.OFF
            and int(getattr(options, "spinlock_iterations", 0)) > 0):
        raise NotImplementedError(f"{driver}: restricted SCF cannot represent SPINLOCK")
    mol = system.unit_cell_molecule()
    context = guess_ecp_context(options, molecule=mol)
    # Validate orphan/malformed context even on a route with no ECP operator.
    validate_guess_ecp(InitialGuess.HCORE, context, mol)
    if route == "ewald" and context.active:
        raise NotImplementedError(
            f"{driver}: ECP Hamiltonians are not implemented by the Python Ewald "
            "route; use a supported GDF or native DIRECT_TRUNCATED route"
        )
    return select_initial_guess(
        mol, requested, is_periodic=True, is_open_shell=is_open_shell,
        supported=periodic_guess_capabilities(
            route, method, dim=system.dim, multi_k=multi_k, transport="k"),
        atomic_spins=getattr(options, "atomic_spins", None),
        restart_supplied=restart_supplied, driver=driver, ecp_context=context,
    )


def guess_ecp_context(options=None, *, molecule: Optional[Molecule] = None):
    """Copy the selected ECP operator, accepting explicit all-electron charges.

    Periodic POB metadata includes physical nuclear charges even without an
    ECP. With the parent molecule supplied, this redundant charge-only record
    can be recognized exactly. Reduced charges or any operator/core metadata
    remain subject to the native ECP validator; no operator is inferred.
    """
    from ._vibeqc_core import _GuessECPContext

    context = _GuessECPContext()
    if options is None:
        return context
    context.xml_centers = list(getattr(options, "ecp_centers", ()) or ())
    context.xml_library = str(getattr(options, "ecp_library", "") or "")
    context.primitive_blocks = list(getattr(options, "ecp_primitive_blocks", ()) or ())
    centers = getattr(options, "ecp_primitive_centers", None)
    if centers is None:
        centers = getattr(options, "ecp_home_centers", ())
    context.primitive_centers = list(centers or ())
    context.effective_charges = list(getattr(options, "ecp_effective_charges", ()) or ())
    context.total_ncore = int(getattr(options, "ecp_total_ncore", 0) or 0)
    if (molecule is not None and context.effective_charges
            and not context.xml_centers and not context.xml_library
            and not context.primitive_blocks and not context.primitive_centers
            and not getattr(options, "ecp_total_ncore", 0)
            and list(context.effective_charges) == [float(a.Z) for a in molecule.atoms]):
        context.effective_charges = []
    return context


def _has_guess_ecp_metadata(options, *, molecule: Optional[Molecule] = None) -> bool:
    """Detect an ECP request, including orphaned operator/core metadata.

    Unsupported routes reject populated operators without parsing them. Only
    charge-only records exactly matching a known parent are all-electron.
    """
    def present(name):
        value = getattr(options, name, None)
        if value is None:
            return False
        try:
            return len(value) > 0
        except TypeError:
            return bool(value)

    if any(present(name) for name in (
        "ecp_centers", "ecp_library", "ecp_primitive_blocks",
        "ecp_primitive_centers", "ecp_home_centers", "ecp_total_ncore",
    )):
        return True
    if not present("ecp_effective_charges"):
        return False
    return molecule is None or bool(
        guess_ecp_context(options, molecule=molecule).effective_charges)


def validate_guess_ecp(kind, context=None, molecule=None):
    """Apply the native atomic-potential/reference capability contract."""
    from ._vibeqc_core import _validate_guess_ecp

    context = context if context is not None else guess_ecp_context()
    _validate_guess_ecp(coerce_initial_guess(kind), context, molecule)
    return context


def periodic_guess_capabilities(
    route: str, method: str = "RHF", *, dim: int = 3,
    multi_k: bool = False, transport: str = "source",
) -> frozenset:
    """Construction and payload contract for a concrete periodic adapter.

    ``source`` is the public result/QVF/Molden seam. Direct callers can name
    ``lattice`` or ``k`` when they supply complete density blocks themselves.
    A matrix at Gamma is never implicitly a full lattice or all-k restart.
    """
    route, method = route.lower(), method.upper()
    if route == "aiccm-hcore":
        return frozenset((InitialGuess.HCORE,))
    if route not in ("ewald", "gdf", "rijcosx", "bipole", "gpw", "gapw", "native", "gpw-native"):
        raise ValueError(f"unknown periodic initial-guess route {route!r}")
    kinds = set(_PERIODIC_ATOMIC_GUESSES)
    if int(dim) != 3:
        kinds.discard(InitialGuess.SAP)
    read = True
    if route in ("gdf", "rijcosx"):
        read = method != "ROKS"
    elif route == "ewald":
        read = True
    elif route == "bipole":
        read = True
    elif route in ("gpw", "gapw") and multi_k:
        read = method in ("RHF", "RKS", "UHF", "UKS", "ROKS")
    if read:
        kinds.add(InitialGuess.READ)
    return frozenset(kinds)


def prepare_molecular_guess_source(
    method, options, molecule, basis, *, read_from=None, fragments=None,
):
    """Validate and resolve restart/fragment payloads for every molecular API."""
    import os
    selection = select_initial_guess(
        molecule, getattr(options, "initial_guess"),
        is_open_shell=method in ("uhf", "uks", "rohf", "roks"),
        atomic_spins=getattr(options, "atomic_spins", None),
        ecp_context=guess_ecp_context(options),
    )
    guess = selection.requested
    if fragments is not None and guess != InitialGuess.FRAGMO:
        raise ValueError(
            "fragments=... was supplied but the resolved SCF initial guess is "
            f"{guess.name}, not FRAGMO; pass initial_guess='fragmo'."
        )
    if read_from is not None and guess != InitialGuess.READ:
        raise ValueError(
            "read_from=... was supplied but the resolved SCF initial guess is "
            f"{guess.name}, not READ; pass initial_guess='read'."
        )

    if guess == InitialGuess.READ:
        _read_obj = read_from
        if isinstance(read_from, (str, os.PathLike)):
            options.read_path = os.fspath(read_from)
            _read_obj = None
        from .guess_read import (
            resolve_read_densities_open,
            resolve_read_density_closed,
        )

        if method in ("rhf", "rks"):
            options.read_density = resolve_read_density_closed(
                options, molecule, basis, _read_obj
            )
        else:
            da, db = resolve_read_densities_open(options, molecule, basis, _read_obj)
            options.read_density_alpha = da
            options.read_density_beta = db
    elif guess == InitialGuess.FRAGMO:
        # A nested reference may already carry the parent-prepared fragment
        # density. Preserve it through solvent/retry option copies; the native
        # preparation boundary still validates its shape and populations.
        if fragments is None:
            fields = (("read_density",) if method in ("rhf", "rks") else
                      ("read_density_alpha", "read_density_beta"))
            if all(np.asarray(getattr(options, name, None)).size > 0
                   and getattr(options, name, None) is not None for name in fields):
                return
        from .guess_fragmo import (
            resolve_fragmo_densities_open,
            resolve_fragmo_density_closed,
        )

        if method in ("rhf", "rks"):
            options.read_density = resolve_fragmo_density_closed(
                options, molecule, basis, fragments
            )
        else:
            da, db = resolve_fragmo_densities_open(options, molecule, basis, fragments)
            options.read_density_alpha = da
            options.read_density_beta = db


def select_initial_guess(
    mol: Molecule,
    initial_guess: object,
    *,
    is_periodic: bool = False,
    is_open_shell: bool = False,
    supported: Iterable[object] = tuple(InitialGuess.__members__.values()),
    atomic_spins: Optional[Sequence[int]] = None,
    restart_supplied: bool = False,
    driver: str = "SCF",
    ecp_context=None,
) -> GuessSelection:
    """Validate the request before resolving an explicit route capability.

    A supplied density uses READ transport. It does not conceal malformed
    selectors or incompatible spin seeds. An internal retry can preserve this
    selection on its result while changing only the transport to READ.
    """
    requested = coerce_initial_guess(initial_guess)
    capabilities = tuple(coerce_initial_guess(k) for k in supported)
    try:
        effective = resolve_initial_guess(
            mol, requested, is_periodic=is_periodic, is_open_shell=is_open_shell,
            supported=capabilities,
        )
    except NotImplementedError as exc:
        valid = ", ".join(sorted(k.name for k in capabilities))
        raise NotImplementedError(
            f"{driver}: initial_guess={requested.name} is not implemented by this "
            f"route (supported: {valid})"
        ) from exc
    validate_guess_ecp(effective, ecp_context, mol)
    if atomic_spins is not None and len(atomic_spins):
        if not is_open_shell or len(atomic_spins) != len(mol.atoms):
            raise ValueError(
                f"{driver}: atomic_spins requires one tag per atom "
                "on an open-shell route"
            )
        if any(s not in (-1, 0, 1) for s in atomic_spins):
            raise ValueError(f"{driver}: atomic_spins tags must be -1, 0 or 1")
        if effective != InitialGuess.SAD or restart_supplied:
            raise ValueError(
                f"{driver}: atomic_spins requires SAD without a restart density"
            )
    transport = InitialGuess.READ if restart_supplied else effective
    capabilities = frozenset(capabilities)
    if transport not in capabilities:
        raise NotImplementedError(
            f"{driver}: initial_guess={requested.name} resolves to {effective.name}; "
            f"{transport.name} is not implemented by this route"
        )
    return GuessSelection(
        requested, InitialGuess.READ if restart_supplied else effective, transport
    )

# Initial guesses not built generically by this Python periodic seam. SAP,
# HUECKEL, and MINAO are special-cased when the caller supplies a
# PeriodicSystem and LatticeSumOptions; callers without that context still fail
# closed here. PATOM is handled by the route-local in-field step. ``AUTO`` is
# safe (resolves to SAD); ``HCORE`` / ``SAD`` / ``READ`` all work here.
_PERIODIC_UNSUPPORTED_GUESSES = frozenset({
    InitialGuess.SAP,
    InitialGuess.PATOM,
    InitialGuess.HUECKEL,
    InitialGuess.MINAO,
})

_PERIODIC_CONTEXT_GUESSES = frozenset({
    InitialGuess.SAP,
    InitialGuess.HUECKEL,
    InitialGuess.MINAO,
})


def _validated_density_guess_block(density, shape):
    """Validate one physical density before normalization or spin conversion."""
    density = np.asarray(density, dtype=complex)
    if (density.ndim != 2 or density.shape[0] != density.shape[1]
        or density.shape != shape or not np.all(np.isfinite(density))
        or not np.allclose(density, density.conj().T, rtol=0, atol=1e-8)):
        raise ValueError("READ: per-k density must be finite, Hermitian and match the basis")
    if density.size and np.linalg.eigvalsh(density).min() < -1e-8:
        raise ValueError("READ: per-k density must be positive semidefinite")
    return 0.5 * (density + density.conj().T)


def _validated_density_k_guess(densities, overlaps, weights):
    """Validate a complete Bloch density and return its weighted population."""
    blocks = [np.asarray(d, dtype=complex) for d in densities]
    metrics = [np.asarray(s, dtype=complex) for s in overlaps]
    weights = np.asarray(weights, dtype=float)
    if len(blocks) != len(metrics):
        raise ValueError("READ: density and overlap k-block counts differ")
    guess_overlap_metric(metrics, weights)
    blocks = [_validated_density_guess_block(d, m.shape) for d, m in zip(blocks, metrics)]
    count = float(sum(w * np.trace(d @ m).real for w, d, m in zip(weights, blocks, metrics)))
    if not np.isfinite(count):
        raise ValueError("READ: source population must be finite")
    return blocks, count


def _rescale_density_k_guess(blocks, count, electrons):
    if (not np.isfinite(electrons)) or electrons < 0:
        raise ValueError("READ: invalid target electron population")
    if electrons == 0:
        return [np.zeros_like(d) for d in blocks]
    if count <= 0:
        raise ValueError("READ: positive target electrons require positive source population")
    scale = 1.0 if abs(count - electrons) <= 1e-12 else electrons / count
    result = [scale * d for d in blocks]
    if any(not np.all(np.isfinite(d)) for d in result):
        raise ValueError("READ: density normalization overflow")
    return result


def normalize_density_k_guess(densities, overlaps, weights, electrons):
    """Normalize the weighted total, retaining relative k-point occupations."""
    blocks, count = _validated_density_k_guess(densities, overlaps, weights)
    return _rescale_density_k_guess(blocks, count, electrons)


def normalize_spin_density_k_guess(alpha, beta, overlaps, weights, n_alpha, n_beta):
    """Normalize a READ spin pair over the complete Bloch mesh.

    Preserve each populated source channel's spatial pattern and relative
    k occupations. An empty channel needing electrons is seeded from the
    other channel. Unlike an unseeded atomic guess, a projected READ source
    can carry local magnetic order even when its global spin is zero.
    """
    metrics, weights = list(overlaps), np.asarray(weights, dtype=float)
    a, na = _validated_density_k_guess(alpha, metrics, weights)
    b, nb = _validated_density_k_guess(beta, metrics, weights)
    if any(not np.isfinite(n) or n < 0 or int(n) != n for n in (n_alpha, n_beta)):
        raise ValueError("READ: invalid target spin populations")
    # Independent rescaling cannot populate a zero source density. Transfer
    # the other spin's shape only in that case; averaging two populated
    # channels would erase broken symmetry after a geometry projection.
    if na <= 0 and n_alpha > 0:
        a, na = b, nb
    if nb <= 0 and n_beta > 0:
        b, nb = a, na
    return (_rescale_density_k_guess(a, na, n_alpha),
            _rescale_density_k_guess(b, nb, n_beta))


def normalize_read_spin_density_guess(alpha, beta, overlap, n_alpha, n_beta):
    """Single-matrix READ adapter for the shared weighted spin policy."""
    a, b = normalize_spin_density_k_guess(
        [alpha], [beta], [overlap], [1.], n_alpha, n_beta,
    )
    if not any(np.iscomplexobj(d) for d in (alpha, beta)):
        return a[0].real, b[0].real
    return a[0], b[0]


def _lattice_restart_k_blocks(density, kmesh):
    from .pbc_bipole_common import bvk_torus_density_matrices, home_cell_block
    return ([home_cell_block(density).astype(complex)] if len(kmesh.kpoints) == 1 else
            bvk_torus_density_matrices(density, list(kmesh.kpoints), kmesh.mesh))


def normalize_periodic_lattice_restart(density, overlaps, weights, electrons, kmesh):
    """Validate prepared real-lattice blocks through their physical k density."""
    return periodic_restart_lattice_density(
        _lattice_restart_k_blocks(density, kmesh), overlaps, weights, electrons, kmesh, density.cells,
    )


def normalize_periodic_lattice_spin_restart(alpha, beta, overlaps, weights, n_alpha, n_beta, kmesh):
    """Normalize both lattice spin densities using their complete Bloch state."""
    a, b = normalize_spin_density_k_guess(
        _lattice_restart_k_blocks(alpha, kmesh), _lattice_restart_k_blocks(beta, kmesh),
        overlaps, weights, n_alpha, n_beta,
    )
    return (
        periodic_restart_lattice_density(a, overlaps, weights, n_alpha, kmesh, alpha.cells),
        periodic_restart_lattice_density(b, overlaps, weights, n_beta, kmesh, beta.cells),
    )


def real_lattice_restart_block(block, n_basis):
    """Offsite real blocks need not be individually symmetric; preserve that."""
    block = np.asarray(block)
    if block.shape != (n_basis, n_basis) or not np.all(np.isfinite(block)):
        raise ValueError("READ: lattice density blocks must be finite and match the AO basis")
    if np.iscomplexobj(block) and np.max(np.abs(block.imag), initial=0) > 1e-10:
        raise ValueError("READ: real lattice adapter cannot discard complex density blocks")
    return np.asarray(block.real, dtype=float)


def periodic_restart_lattice_density(
    densities, overlaps, weights, electrons, kmesh, cells, *, retained_k_system=None,
):
    """Validate an all-k restart before folding to a real BvK lattice density.

    The real lattice representation requires time-reversal-compatible blocks.
    Verify the round trip so complex information can never be discarded by
    the native real-space fold. A mesh or cell list missing representatives
    likewise cannot silently approximate the supplied state. A driver retaining
    the full D(k) may pass its system: then validate time reversal directly and
    construct only the real-space blocks its ordinary SCF uses. No information
    is inferred back from that potentially truncated cell list.
    """
    from .periodic_k_density import real_space_density_from_per_k_density
    from .pbc_bipole_common import bvk_torus_density_matrices, home_cell_block
    blocks = normalize_density_k_guess(densities, overlaps, weights, electrons)
    if retained_k_system is not None:
        # With full D(k) retained, the Fourier fold is a view of the state,
        # not its storage. Real lattice blocks require D(-k)=conj(D(k)).
        fractional = (np.asarray(kmesh.kpoints)
                      @ np.asarray(retained_k_system.lattice) / (2.0 * np.pi))
        w = np.asarray(weights, dtype=float)
        for i, point in enumerate(fractional):
            sums = fractional + point
            partners = np.flatnonzero(np.max(np.abs(sums - np.rint(sums)), axis=1) < 1e-8)
            if (len(partners) != 1
                    or not np.isclose(w[i], w[partners[0]], atol=1e-12, rtol=0)
                    or not np.allclose(blocks[i], blocks[partners[0]].conj(), atol=1e-8, rtol=1e-10)):
                raise NotImplementedError(
                    "READ: a real lattice fold requires a complete "
                    "time-reversal-compatible Bloch density"
                )
        return real_space_density_from_per_k_density(blocks, kmesh, cells)
    lattice = real_space_density_from_per_k_density(blocks, kmesh, cells)
    folded = ([home_cell_block(lattice).astype(complex)] if len(blocks) == 1 else
              bvk_torus_density_matrices(lattice, list(kmesh.kpoints), kmesh.mesh))
    if len(folded) != len(blocks) or any(
        not np.allclose(a, b, atol=1e-8, rtol=1e-10)
        for a, b in zip(blocks, folded)
    ):
        raise NotImplementedError(
            "READ: the supplied Bloch density cannot be represented exactly "
            "on this real BvK lattice; require a complete time-reversal-compatible mesh"
        )
    return lattice


def periodic_result_selection(system, initial_guess, *, restarted=False):
    """Describe a validated periodic construction using the execution resolver."""
    return select_initial_guess(
        system.unit_cell_molecule(),
        InitialGuess.AUTO if initial_guess is None else initial_guess,
        is_periodic=True, restart_supplied=restarted,
    )


def resolve_initial_guess(
    mol: Molecule,
    initial_guess: object,
    *,
    is_periodic: bool = False,
    is_open_shell: bool = False,
    supported: Optional[Iterable[object]] = None,
) -> InitialGuess:
    """Return the concrete guess selected by the unified engine policy.

    User-facing enum and string spellings first pass through the canonical
    coercer.  ``AUTO`` is then resolved by the same native, molecule-aware
    policy used by the production molecular drivers.  Keeping both steps here
    prevents Python periodic routes from growing their own alias or AUTO
    tables.
    """
    kind = coerce_initial_guess(initial_guess)
    effective = _resolve_initial_guess_for_molecule(
        mol,
        kind,
        bool(is_periodic),
        bool(is_open_shell),
        None if supported is None else [coerce_initial_guess(item) for item in supported],
    )
    return effective


def _coerce_periodic_driver_guess(
    initial_guess: object,
    *,
    driver: str,
    supported: Iterable[object],
    restart_supplied: bool = False,
) -> InitialGuess:
    """Validate the selector and route before applying restart precedence.

    ``None`` retains its public compatibility meaning of AUTO. A supplied
    density may supersede a supported construction, but cannot make an
    unsupported or malformed selector valid.
    """
    return select_initial_guess(
        Molecule([]),
        InitialGuess.AUTO if initial_guess is None else initial_guess,
        is_periodic=True, supported=tuple(supported),
        restart_supplied=restart_supplied, driver=driver,
    ).transport


def _gamma_fold(lat_set) -> np.ndarray:
    """Return the real Gamma-folded matrix for a lattice matrix set."""
    mat = np.zeros_like(np.asarray(lat_set.blocks[0], dtype=float))
    for block in lat_set.blocks:
        mat += np.asarray(block, dtype=float)
    return 0.5 * (mat + mat.T)


def _canonical_orthogonalizer(
    S: np.ndarray, threshold: float = 1e-7, *, normalize_diagonal: bool = True
) -> np.ndarray:
    """Canonical orthogonalizer for a real or complex Hermitian overlap."""
    S_arr = np.asarray(S)
    S = 0.5 * (S_arr + S_arr.conj().T)
    diag = np.real(np.diag(S))
    if not normalize_diagonal or np.any(diag <= 0.0):
        eigvals, eigvecs = np.linalg.eigh(S)
        mask = eigvals >= threshold
        if not np.any(mask):
            raise RuntimeError(
                "periodic initial guess: no overlap eigenvalue above "
                f"threshold {threshold:.1e}; basis is fully linearly dependent"
            )
        return eigvecs[:, mask] / np.sqrt(eigvals[mask])

    scale = 1.0 / np.sqrt(diag)
    S_norm = S * np.outer(scale, scale)
    eigvals, eigvecs = np.linalg.eigh(S_norm)
    mask = eigvals >= threshold
    if not np.any(mask):
        raise RuntimeError(
            "periodic initial guess: no overlap eigenvalue above threshold "
            f"{threshold:.1e} after sqrt-diag normalisation; basis is fully "
            "linearly dependent"
        )
    X_norm = eigvecs[:, mask] / np.sqrt(eigvals[mask])
    return scale[:, None] * X_norm


def _density_from_fock(
    F: np.ndarray,
    S: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    """Diagonalise a closed-shell Fock-mode guess against ``S``."""
    X = _canonical_orthogonalizer(S)
    if int(n_occ) > X.shape[1]:
        raise RuntimeError(
            "periodic initial guess: fewer retained overlap directions "
            f"({X.shape[1]}) than occupied orbitals ({int(n_occ)})"
        )
    F_arr = np.asarray(F)
    F = 0.5 * (F_arr + F_arr.conj().T)
    Fp = X.conj().T @ F @ X
    _, Cp = np.linalg.eigh(0.5 * (Fp + Fp.conj().T))
    C = X @ Cp
    D = 2.0 * C[:, :int(n_occ)] @ C[:, :int(n_occ)].conj().T
    return 0.5 * (D + D.conj().T)


def _densities_from_fock_open_shell(
    F: np.ndarray,
    S: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalise a Fock-mode guess and fill alpha/beta orbitals."""
    n_alpha_i = int(n_alpha)
    n_beta_i = int(n_beta)
    n_occ = max(n_alpha_i, n_beta_i)
    S_arr = np.asarray(S)
    shape = S_arr.shape
    if n_occ <= 0:
        Z = np.zeros(shape, dtype=S_arr.dtype)
        return Z, Z.copy()

    X = _canonical_orthogonalizer(S)
    if n_occ > X.shape[1]:
        raise RuntimeError(
            "periodic initial guess: fewer retained overlap directions "
            f"({X.shape[1]}) than occupied spin orbitals ({n_occ})"
        )
    F_arr = np.asarray(F)
    F = 0.5 * (F_arr + F_arr.conj().T)
    Fp = X.conj().T @ F @ X
    _, Cp = np.linalg.eigh(0.5 * (Fp + Fp.conj().T))
    C = X @ Cp

    def build(n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((C.shape[0], C.shape[0]), dtype=C.dtype)
        D = C[:, :n] @ C[:, :n].conj().T
        return 0.5 * (D + D.conj().T)

    return build(n_alpha_i), build(n_beta_i)


def periodic_fock_guess_k(
    system: PeriodicSystem,
    basis: BasisSet,
    kpoints_cart: Sequence[Sequence[float]],
    initial_guess: object,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    kinetic_lattice=None,
    overlap_lattice=None,
    ecp_context=None,
) -> Optional[Tuple[np.ndarray, ...]]:
    """Return a periodic Fock-mode guess at each Cartesian k-point.

    SAP produces ``T(k) + V_SAP(k)`` and HUECKEL produces the lattice GWH
    Fock.  Supplying the route's already-built kinetic/overlap lattice keeps
    the guess in exactly the same real-space truncation convention as its SCF
    pencil.  The caller remains responsible for overlap orthogonalisation and
    occupations, which keeps global-k Aufbau and smearing route-local.

    Non-Fock guesses return ``None``.  Capability validation belongs at the
    driver boundary via :func:`_coerce_periodic_driver_guess`; returning None
    here is an artifact-mode result, not permission to substitute Hcore.
    """
    kind = coerce_initial_guess(initial_guess)
    if kind == InitialGuess.AUTO:
        kind = resolve_initial_guess(
            system.unit_cell_molecule(), kind, is_periodic=True
        )
    ecp_context = validate_guess_ecp(kind, ecp_context, system.unit_cell_molecule())
    if kind not in (InitialGuess.SAP, InitialGuess.HUECKEL):
        return None
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    if kind == InitialGuess.SAP:
        from .periodic_grid import build_periodic_becke_grid

        kinetic = (
            kinetic_lattice
            if kinetic_lattice is not None
            else compute_kinetic_lattice(basis, system, opts)
        )
        grid = build_periodic_becke_grid(system)
        if ecp_context.active:
            from ._vibeqc_core import compute_vsap_ecp_lattice
            potential = compute_vsap_ecp_lattice(basis, system, opts, ecp_context)
        else:
            potential = compute_vsap_lattice(
                basis, system, grid, "sap_helfem_large", opts)
        fock_lattice = None
    else:
        overlap = (
            overlap_lattice
            if overlap_lattice is not None
            else compute_overlap_lattice(basis, system, opts)
        )
        fock_lattice = compute_huckel_fock_lattice(
            system.unit_cell_molecule(), basis, overlap, ecp_context
        )

    fock_matrices = []
    for k_cart in kpoints_cart:
        k = np.asarray(k_cart, dtype=float)
        if kind == InitialGuess.SAP:
            F_k = np.asarray(bloch_sum(kinetic, k)) + np.asarray(
                bloch_sum(potential, k)
            )
        else:
            F_k = np.asarray(bloch_sum(fock_lattice, k))
        fock_matrices.append(0.5 * (F_k + F_k.conj().T))
    return tuple(fock_matrices)


def periodic_sap_fock_k(
    system: PeriodicSystem,
    basis: BasisSet,
    kpoints_cart: Sequence[Sequence[float]],
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    kinetic_lattice=None,
) -> Tuple[np.ndarray, ...]:
    """Compatibility wrapper for the former SAP-only constructor."""
    result = periodic_fock_guess_k(
        system,
        basis,
        kpoints_cart,
        InitialGuess.SAP,
        lattice_opts=lattice_opts,
        kinetic_lattice=kinetic_lattice,
    )
    assert result is not None  # SAP is a Fock-mode guess by construction.
    return result


def _proportional_spin_split(
    D_total: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a total density by the requested per-spin electron counts."""
    D_arr = np.asarray(D_total, dtype=float)
    D = 0.5 * (D_arr + D_arr.T)
    n_elec = int(n_alpha) + int(n_beta)
    if n_elec <= 0:
        Z = np.zeros_like(D)
        return Z, Z.copy()
    return (float(n_alpha) / n_elec) * D, (float(n_beta) / n_elec) * D


def _check_periodic_guess_supported(initial_guess: InitialGuess) -> None:
    """Raise a clear NotImplementedError if ``initial_guess`` has no periodic
    implementation. No-op for the supported guesses (HCORE/SAD/READ/AUTO)."""
    if initial_guess == InitialGuess.FRAGMO:
        raise NotImplementedError(
            "FRAGMO requires fragment atom/image ownership and Bloch assembly; "
            "use run_periodic_job(..., initial_guess='FRAGMO', fragments=...) "
            "or the native periodic wrappers with fragments=."
        )
    if initial_guess in _PERIODIC_UNSUPPORTED_GUESSES:
        # SAP/HUECKEL/MINAO reach this branch only for callers that did not
        # supply periodic context. PATOM still needs in-field driver plumbing
        # on the Python periodic side.
        raise NotImplementedError(
            f"{initial_guess.name} not yet implemented for periodic systems "
            "via this driver; use SAD, HCORE, or READ for a periodic SCF -- "
            "AUTO also works (it resolves to SAD for periodic cells)."
        )



def guess_overlap_metric(overlap, weights=None) -> np.ndarray:
    """Return S or sum_k w_k S(k), retaining complex Hermitian dtype."""
    matrices = np.asarray(overlap)
    if matrices.ndim == 2:
        if weights is not None:
            raise ValueError("initial guess: k weights require a list of overlaps")
        metric = matrices
    elif matrices.ndim == 3:
        if weights is None:
            raise ValueError("initial guess: multi-k overlap requires k weights")
        w = np.asarray(weights, dtype=float)
        if (
            w.shape != (len(matrices),) or not np.all(np.isfinite(w))
            or np.any(w < 0) or not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("initial guess: finite nonnegative k weights must sum to one")
        if matrices.shape[1] != matrices.shape[2] or not np.all(np.isfinite(matrices)):
            raise ValueError("initial guess: every k overlap must be finite and square")
        if not np.allclose(matrices, matrices.conj().transpose(0, 2, 1), atol=1e-10, rtol=1e-10):
            raise ValueError("initial guess: every k overlap must be Hermitian")
        metric = np.einsum("k,kij->ij", w, matrices)
    else:
        raise ValueError("initial guess: overlap must be a matrix or list of matrices")
    if metric.shape[0] != metric.shape[1] or not np.all(np.isfinite(metric)):
        raise ValueError("initial guess: overlap must be finite and square")
    if not np.allclose(metric, metric.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("initial guess: overlap must be Hermitian")
    return 0.5 * (metric + metric.conj().T)


def normalize_density_guess(density, overlap, electrons, *, weights=None):
    """Normalize an already projected density-mode artifact in its SCF metric."""
    metric = guess_overlap_metric(overlap, weights)
    result = np.asarray(_normalize_guess_density(density, metric, electrons))
    return result if np.iscomplexobj(density) else result.real


def normalize_spin_density_guess(
    alpha, beta, overlap, n_alpha, n_beta, *, weights=None,
):
    """Use the same population and spin-pattern contract as the native engine."""
    metric = guess_overlap_metric(overlap, weights)
    da, db = _normalize_guess_spin_densities(
        alpha, beta, metric, n_alpha, n_beta
    )
    if not any(np.iscomplexobj(item) for item in (alpha, beta)):
        return np.asarray(da).real, np.asarray(db).real
    return np.asarray(da), np.asarray(db)


def patom_spin_density_step(alpha, beta, hcore, overlap, n_alpha, n_beta,
                            build_j, build_k, *, threshold=1e-7):
    """One full-HF in-field step from SAD, independent of the target XC.

    J sees the total density and K each spin density. This is the same
    PATOM contract used by the native engine; RO methods impose common
    orbitals in their subsequent SCF iteration.
    """
    x = _canonical_orthogonalizer(overlap, threshold)
    coulomb = np.asarray(build_j(np.asarray(alpha) + np.asarray(beta)))
    result = []
    for density, count in ((alpha, n_alpha), (beta, n_beta)):
        if count > x.shape[1]:
            raise ValueError("PATOM: insufficient independent orbitals for spin population")
        fock = np.asarray(hcore) + coulomb - np.asarray(build_k(density))
        fock = (fock + fock.conj().T) * 0.5
        _, vectors = np.linalg.eigh(x.conj().T @ fock @ x)
        occupied = (x @ vectors)[:, :count]
        result.append(occupied @ occupied.conj().T)
    return tuple(result)


def patom_ewald_spin_step(
    basis, system, alpha_real, beta_real, hcore_k, overlap_k, kmesh,
    n_alpha, n_beta, lattice_opts, omega, build_j_blocks, exchange_cache,
    exchange_g0, *, sr_image_extent=None,
):
    """Full-HF PATOM construction shared by restricted-open Ewald routes.

    Use the route's J gauge, short-range exchange image domain, reciprocal
    exchange cache and finite-mesh zero-mode correction without target XC.
    """
    from ._vibeqc_core import build_jk_2e_real_space, compute_overlap_lattice, bloch_sum
    from .bipole_fock_ewald import compute_K_long_range_at_k
    from .pbc_bipole_common import bvk_torus_density_matrices
    from .periodic_k_density import real_space_density_from_per_k_density

    total = compute_overlap_lattice(basis, system, lattice_opts)
    for index, (a, b) in enumerate(zip(alpha_real.blocks, beta_real.blocks)):
        total.set_block(index, np.asarray(a) + np.asarray(b))
    j_blocks = build_j_blocks(total)
    j_lattice = compute_overlap_lattice(basis, system, lattice_opts)
    for index, block in enumerate(j_blocks):
        j_lattice.set_block(index, block)
    kpoints = list(kmesh.kpoints)
    weights = list(kmesh.weights)
    j_k = [np.asarray(bloch_sum(j_lattice, k)) for k in kpoints]
    output = []
    for density_real, count in ((alpha_real, n_alpha), (beta_real, n_beta)):
        density_k = bvk_torus_density_matrices(density_real, kpoints, kmesh.mesh)
        if sr_image_extent is None:
            k_sr = build_jk_2e_real_space(
                basis, system, lattice_opts, density_real, omega,
            ).K
        else:
            from .pbc_bipole_fock import _sr_image_padded_jk
            k_sr = _sr_image_padded_jk(
                basis, system, lattice_opts, density_real, float(omega),
                float(sr_image_extent),
            ).K
        refined = []
        for index, (k, h, j, overlap) in enumerate(zip(kpoints, hcore_k, j_k, overlap_k)):
            exchange = np.asarray(bloch_sum(k_sr, k))
            exchange += compute_K_long_range_at_k(
                exchange_cache, np.asarray(k), kpoints, weights, density_k,
            )
            exchange += exchange_g0 * (overlap @ density_k[index] @ overlap)
            x = _canonical_orthogonalizer(overlap)
            fock = h + j - exchange
            _, vectors = np.linalg.eigh(x.conj().T @ ((fock + fock.conj().T) * 0.5) @ x)
            occupied = (x @ vectors)[:, :count]
            refined.append(occupied @ occupied.conj().T)
        output.append(real_space_density_from_per_k_density(
            refined, kmesh, list(alpha_real.cells),
        ))
    return tuple(output)


def _density_guess_metric(basis, overlap, weights, periodic_system, lattice_opts):
    if overlap is not None:
        return guess_overlap_metric(overlap, weights)
    if weights is not None:
        raise ValueError("initial guess: weights supplied without overlaps")
    if periodic_system is not None:
        opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
        return _gamma_fold(compute_overlap_lattice(basis, periodic_system, opts))
    # Context-free compatibility calls have only a molecular AO metric.
    # Every production periodic route supplies its actual overlap explicitly.
    return np.asarray(compute_overlap(basis))

def initial_density_closed_shell(
    mol: Molecule,
    basis: BasisSet,
    n_occ: int,
    initial_guess: object,
    *,
    is_periodic: bool = True,
    periodic_system: Optional[PeriodicSystem] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
    read_density: Optional[np.ndarray] = None,
    read_path: str = "",
    overlap=None,
    weights=None,
    ecp_context=None,
) -> Optional[np.ndarray]:
    """Return the closed-shell initial density matrix, or ``None`` when
    the engine produced no density (HCORE -- the driver should
    diagonalise Hcore itself; SAP/HUECKEL are diagonalised here when the
    periodic context is available).

    For periodic callers the returned matrix lives at the g=0 cell;
    other cells are zero. Bloch-summing g=0 recovers a sensible D(k)
    for the first Fock build.

    ``READ`` short-circuits the engine: the prior g=0 cell density is
    taken from ``read_density`` (pre-resolved + projected by
    run_periodic_job, e.g. from an in-memory ``read_from``) or, if that
    is empty, read + projected from ``read_path`` (.qvf / .molden, including
    ORCA's .molden.input suffix). The
    returned density is injected at g=0 like any other density-mode guess.
    """
    def finish(density):
        metric = _density_guess_metric(
            basis, overlap, weights, periodic_system, lattice_opts
        )
        return normalize_density_guess(density, metric, 2 * int(n_occ))

    kind = resolve_initial_guess(
        mol,
        initial_guess,
        is_periodic=is_periodic,
        is_open_shell=False,
    )
    ecp_context = validate_guess_ecp(kind, ecp_context, mol)
    if kind == InitialGuess.READ:
        from .guess_read import resolve_periodic_read_density_closed
        return finish(resolve_periodic_read_density_closed(
            basis, read_density=read_density, read_path=read_path))
    if (
        is_periodic
        and kind in (InitialGuess.SAP, InitialGuess.HUECKEL)
        and periodic_system is not None
    ):
        opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
        S_set = compute_overlap_lattice(basis, periodic_system, opts)
        fock_k = periodic_fock_guess_k(
            periodic_system,
            basis,
            (np.zeros(3),),
            kind,
            lattice_opts=opts,
            overlap_lattice=S_set,
            ecp_context=ecp_context,
        )
        assert fock_k is not None
        return finish(_density_from_fock(
            fock_k[0],
            _gamma_fold(S_set),
            int(n_occ),
        ))
    if (
        is_periodic
        and kind == InitialGuess.MINAO
        and periodic_system is not None
    ):
        opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
        S_set = compute_overlap_lattice(basis, periodic_system, opts)
        S_gamma = _gamma_fold(S_set)
        D = compute_minao_density_periodic(
            mol, basis, periodic_system, S_gamma, 2 * int(n_occ), opts, ecp_context)
        D = np.asarray(D, dtype=float)
        return finish(D)
    if is_periodic:
        _check_periodic_guess_supported(kind)
    D = _guess_closed_shell_density(
        mol, basis, int(n_occ), kind,
        is_periodic=is_periodic,
        ecp_context=ecp_context,
        overlap=_density_guess_metric(
            basis, overlap, weights, periodic_system, lattice_opts
        ).real,
    )
    if D is None:
        return None
    D = np.asarray(D, dtype=float)
    return finish(D)  # symmetrise (no-op modulo FP cleanup)


def initial_densities_open_shell(
    mol: Molecule,
    basis: BasisSet,
    n_alpha: int,
    n_beta: int,
    initial_guess: object,
    *,
    is_periodic: bool = True,
    periodic_system: Optional[PeriodicSystem] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
    atomic_spins: Optional[Sequence[int]] = None,
    read_density_alpha: Optional[np.ndarray] = None,
    read_density_beta: Optional[np.ndarray] = None,
    read_path: str = "",
    overlap=None,
    weights=None,
    ecp_context=None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(D_alpha, D_beta)`` for open-shell, or ``None`` when the
    engine produced no density (HCORE fallback).

    Periodic SAP and HUECKEL use a spin-resolved Fock diagonalisation when the
    caller supplies the periodic context. Periodic MINAO uses the projected
    total density and the same proportional spin split as the molecular engine.
    SAD keeps the engine's spin-aware atomic-density packaging (or the
    explicit ATOMSPIN broken-symmetry density).

    ``atomic_spins`` (ATOMSPIN) tags each unit-cell atom +1 (majority
    alpha) / -1 (majority beta) / 0 (unpolarised). When set, the engine
    assembles a block-diagonal broken-symmetry SAD density at g=0 (which
    Bloch-sums to a broken-symmetry ``D(k)``) instead of the spin-symmetric
    proportional split. Consulted only for the SAD guess; empty/None keeps
    the symmetric split. See ``GuessEngine::build_open_shell``.

    ``READ`` short-circuits the engine with the prior per-spin g=0 cell
    densities -- ``read_density_alpha`` / ``read_density_beta`` (pre-resolved
    + projected) or, if empty, read + projected from ``read_path``.
    """
    def finish(densities):
        metric = _density_guess_metric(
            basis, overlap, weights, periodic_system, lattice_opts
        )
        if atomic_spins is not None and len(atomic_spins):
            da, db = _normalize_atomic_guess(
                mol, basis, *densities, metric, int(n_alpha), int(n_beta),
                [int(s) for s in atomic_spins],
            )
            if not any(np.iscomplexobj(d) for d in densities):
                return np.asarray(da).real, np.asarray(db).real
            return np.asarray(da), np.asarray(db)
        return normalize_spin_density_guess(
            *densities, metric, int(n_alpha), int(n_beta)
        )

    if atomic_spins is not None and any(s not in (-1, 0, 1) for s in atomic_spins):
        raise ValueError("atomic_spins tags must be -1, 0 or 1")
    kind = resolve_initial_guess(
        mol,
        initial_guess,
        is_periodic=is_periodic,
        is_open_shell=True,
    )
    if (
        atomic_spins is not None and len(atomic_spins)
        and kind != InitialGuess.SAD
    ):
        raise RuntimeError(
            "GuessEngine: atomic_spins (ATOMSPIN broken-symmetry seed) "
            f"requires the SAD guess; resolved guess is {kind.name}"
        )
    ecp_context = validate_guess_ecp(kind, ecp_context, mol)
    if kind == InitialGuess.READ:
        from .guess_read import resolve_periodic_read_densities_open
        densities = resolve_periodic_read_densities_open(
            basis, read_density_alpha=read_density_alpha,
            read_density_beta=read_density_beta, read_path=read_path)
        return normalize_read_spin_density_guess(
            *densities, _density_guess_metric(
                basis, overlap, weights, periodic_system, lattice_opts,
            ), int(n_alpha), int(n_beta),
        )
    if (
        is_periodic
        and kind in _PERIODIC_CONTEXT_GUESSES
        and periodic_system is not None
    ):
        opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
        S_set = compute_overlap_lattice(basis, periodic_system, opts)
        S_gamma = _gamma_fold(S_set)
        n_elec = int(n_alpha) + int(n_beta)
        if kind in (InitialGuess.SAP, InitialGuess.HUECKEL):
            fock_k = periodic_fock_guess_k(
                periodic_system,
                basis,
                (np.zeros(3),),
                kind,
                lattice_opts=opts,
                overlap_lattice=S_set,
                ecp_context=ecp_context,
            )
            assert fock_k is not None
            return finish(_densities_from_fock_open_shell(
                fock_k[0],
                S_gamma,
                int(n_alpha),
                int(n_beta),
            ))
        if kind == InitialGuess.MINAO:
            D = compute_minao_density_periodic(
                mol, basis, periodic_system, S_gamma, n_elec, opts, ecp_context)
            return finish(_proportional_spin_split(D, int(n_alpha), int(n_beta)))
    if is_periodic:
        _check_periodic_guess_supported(kind)
    result = _guess_open_shell_density(
        mol, basis, int(n_alpha), int(n_beta), kind,
        is_periodic=is_periodic,
        ecp_context=ecp_context,
        atomic_spins=[int(s) for s in atomic_spins] if atomic_spins is not None else [],
        overlap=_density_guess_metric(
            basis, overlap, weights, periodic_system, lattice_opts
        ).real,
    )
    if result is None:
        return None
    Da, Db = result
    Da = np.asarray(Da, dtype=float)
    Db = np.asarray(Db, dtype=float)
    return finish((Da, Db))


__all__ = [
    "_coerce_periodic_driver_guess",
    "initial_density_closed_shell",
    "initial_densities_open_shell",
    "periodic_fock_guess_k",
    "periodic_sap_fock_k",
    "resolve_initial_guess",
]


def patom_full_hf_focks_k(system, basis, kmesh, hcore, overlaps,
                          n_alpha, n_beta, lattice_opts, *, atomic_spins=None):
    """Full periodic HF seed operator for pure-DFT GPW/GAPW adapters.

    The initial one-electron operator is the target route's H(k). Hartree
    and full exchange use the shared analytic Ewald builders for this single
    seed step, including the finite-mesh exchange correction. The subsequent
    DFT iterations keep their ordinary grid Hartree and XC operators.
    """
    from . import _vibeqc_core as core
    from .periodic_corrected_exchange import CorrectedEwaldExchange
    from .periodic_fock_multi_k import build_periodic_j_ewald3d_k_from_k_density
    from .periodic_k_density import real_space_density_from_per_k_density

    points, weights = list(kmesh.kpoints), list(kmesh.weights)
    cells = list(core.compute_overlap_lattice(basis, system, lattice_opts).cells)
    exchange = CorrectedEwaldExchange.build(
        basis, system, np.asarray([c.r_cart for c in cells]), kmesh, 0.5,
        where="PATOM full-HF seed")
    a, b = initial_densities_open_shell(
        system.unit_cell_molecule(), basis, n_alpha, n_beta, InitialGuess.SAD,
        is_periodic=True, periodic_system=system, lattice_opts=lattice_opts,
        overlap=overlaps, weights=weights, atomic_spins=atomic_spins)
    alpha = [np.asarray(a, dtype=complex).copy() for _ in points]
    beta = [np.asarray(b, dtype=complex).copy() for _ in points]
    total = [a + b for a, b in zip(alpha, beta)]
    coulomb = build_periodic_j_ewald3d_k_from_k_density(
        basis, system, total, points, weights, cells, omega=0.5)
    result = []
    for densities in (alpha, beta):
        real = real_space_density_from_per_k_density(densities, kmesh, cells)
        short = core.build_jk_2e_real_space(basis, system, lattice_opts, real, 0.5).K
        correction = exchange.k_space_terms_all_k(overlaps, densities)
        result.append([
            np.asarray(h) + j - np.asarray(core.bloch_sum(short, k)) - c
            for h, j, k, c in zip(hcore, coulomb, points, correction)])
    return tuple(result)
