"""Numerical implementation of the standalone relocalization protocol v1.

The public process entry point is ``python -m vibeqc_relocalize``. Molecular
algorithms are those of :mod:`vibeqc.iao`; the separately named experimental
periodic method uses :mod:`vibeqc.periodic.chi.localization`. No SCF is run
unless ``source='fresh_rhf'`` is explicit. See docs/relocalization_worker.md.
"""

from __future__ import annotations

import itertools

import numpy as np
from vibeqc_relocalize import RequestError, package_version

from . import Atom, BasisSet, Molecule, RHFOptions, compute_overlap, run_rhf
from ._primitive_norm import libint_primitive_norm
from ._vibeqc_core import ShellInfo
from .iao import LOCALIZATION_METHODS, analyse_localization, iao_unsupported_reason

TOL = 1e-7
AO_CONVENTION = "qvf-gto-v1"
BLOCH_CONVENTION = "exp(+2pi*i*k.R)"


def _require(condition, message, field=None, code="invalid_request"):
    if not condition:
        raise RequestError(code, message, field)


def _fields(value, required, optional=(), field="request"):
    _require(isinstance(value, dict), f"{field} must be an object", field)
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    _require(not missing, f"Missing fields: {sorted(missing)}", field)
    _require(not extra, f"Unknown fields: {sorted(extra)}", field)


def _int(value, low, high, field):
    _require(
        type(value) is int and low <= value <= high,
        f"{field} must be an integer in [{low}, {high}]",
        field,
    )
    return value


def _array(value, field, shape=None):
    try:
        # Preserve each JSON scalar's type until validation: numpy otherwise
        # promotes mixed [true, 1.0] to floats and loses the invalid boolean.
        raw = np.asarray(value, dtype=object)
        _require(
            all(
                isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, (bool, np.bool_))
                for v in raw.flat
            ),
            f"{field} must contain real numbers, not booleans or strings",
            field,
        )
        array = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, RequestError):
            raise
        raise RequestError(
            "invalid_request", f"{field} is not a numeric array", field
        ) from exc
    _require(np.all(np.isfinite(array)), f"{field} must be finite", field)
    if shape is not None:
        _require(
            array.shape == shape,
            f"{field} must have shape {shape}, got {array.shape}",
            field,
        )
    return array


def _matrix(value, field):
    _fields(value, ("encoding", "data"), field=field)
    array = _array(value["data"], field + ".data")
    if value["encoding"] == "real":
        return array
    _require(
        value["encoding"] == "complex_split_last_axis",
        "Expected real or complex_split_last_axis encoding",
        field,
    )
    _require(
        array.ndim >= 1 and array.shape[-1] == 2,
        "Complex arrays end in [real, imag] pairs",
        field,
    )
    return array[..., 0] + 1j * array[..., 1]


def _encode(array):
    array = np.asarray(array)
    _require(
        np.all(np.isfinite(array)),
        "Non-finite numerical result",
        code="numerical_validation_failed",
    )
    if np.iscomplexobj(array):
        return {
            "encoding": "complex_split_last_axis",
            "data": np.stack((array.real, array.imag), axis=-1).tolist(),
        }
    return {"encoding": "real", "data": array.tolist()}


def _metric(overlap, n):
    _require(overlap.shape == (n, n), f"overlap must have shape {(n, n)}", "overlap")
    _require(
        np.max(np.abs(overlap - overlap.conj().T)) < 1e-10,
        "overlap must be Hermitian",
        "overlap",
    )
    eigenvalues = np.linalg.eigvalsh(overlap)
    _require(
        eigenvalues[0] > max(eigenvalues[-1] * 1e-10, 1e-14),
        "overlap must be positive definite and well conditioned",
        "overlap",
    )


def _orth_error(c, s):
    return float(np.linalg.norm(c.conj().T @ s @ c - np.eye(c.shape[1])))


def _audit(before, after, overlap):
    orth = _orth_error(after, overlap)
    u = before.conj().T @ overlap @ after
    delta = after - before @ u
    residual = float(np.sqrt(max(0.0, np.trace(delta.conj().T @ overlap @ delta).real)))
    unitary = float(np.linalg.norm(u.conj().T @ u - np.eye(u.shape[0])))
    _require(
        max(orth, residual, unitary) < TOL,
        f"Localized subspace audit failed: orth={orth:g}, residual={residual:g}, unitary={unitary:g}",
        code="numerical_validation_failed",
    )
    return {
        "input_orthonormality_error": _orth_error(before, overlap),
        "orthonormality_error": orth,
        "subspace_residual": residual,
        "unitary_error": unitary,
        "tolerance": TOL,
    }, u


def _basis(spec, molecule, positions, supplied):
    _fields(spec, ("ao_convention", "uses_ecp"), ("name", "shells"), "basis")
    _require(
        spec["ao_convention"] == AO_CONVENTION,
        "Only the explicit qvf-gto-v1 AO convention is supported",
        "basis.ao_convention",
        "unsupported_basis",
    )
    _require(
        spec["uses_ecp"] is False,
        "ECP inputs are unsupported",
        "basis.uses_ecp",
        "unsupported_basis",
    )
    name = spec.get("name", "archived")
    _require(
        isinstance(name, str) and 0 < len(name) <= 128,
        "Invalid basis name",
        "basis.name",
    )
    if "shells" not in spec:
        _require(
            not supplied,
            "Supplied orbitals require exact archived shells; a basis name is insufficient",
            "basis.shells",
            "incomplete_basis",
        )
        _require("name" in spec, "fresh_rhf requires a basis name or shells", "basis")
        basis = BasisSet(molecule, name)
        # A named ECP basis may load orbital shells even though the caller did
        # not supply an ECP. Reject known sidecars instead of doing all-electron SCF.
        from .basis_registry import ecp_requirements
        from .ecp_metadata import basis_sidecar_has_ecp_operator

        _require(
            not ecp_requirements(name, [a.Z for a in molecule.atoms])
            and not basis_sidecar_has_ecp_operator(molecule, basis),
            "Named ECP bases are unsupported",
            "basis.name",
            "unsupported_basis",
        )
        return basis
    shells = spec["shells"]
    _require(
        isinstance(shells, list) and 0 < len(shells) <= 512,
        "Invalid shell list",
        "basis.shells",
    )
    native = []
    n_ao = 0
    for i, shell in enumerate(shells):
        field = f"basis.shells[{i}]"
        _fields(
            shell, ("center", "l", "pure", "exponents", "coefficients"), field=field
        )
        center = _int(shell["center"], 0, len(positions) - 1, field + ".center")
        ell = _int(shell["l"], 0, 6, field + ".l")
        _require(type(shell["pure"]) is bool, "pure must be boolean", field)
        exponents = _array(shell["exponents"], field + ".exponents")
        _require(
            exponents.ndim == 1 and 0 < exponents.size <= 128 and np.all(exponents > 0),
            "Expected 1 to 128 positive primitive exponents",
            field,
        )
        coefficients = _array(
            shell["coefficients"], field + ".coefficients", exponents.shape
        )
        _require(np.any(coefficients != 0), "Empty contraction", field)
        # Inverse of output.formats.qvf._basis_shell_payload. No contraction
        # renormalization, AO permutation or per-Cartesian-component correction.
        embedded = [
            float(c) * libint_primitive_norm(float(a), ell)
            for a, c in zip(exponents, coefficients)
        ]
        native.append(
            ShellInfo(
                center,
                ell,
                shell["pure"],
                exponents.tolist(),
                embedded,
                positions[center].tolist(),
            )
        )
        n_ao += 2 * ell + 1 if shell["pure"] else (ell + 1) * (ell + 2) // 2
    _require(n_ao <= 512, "At most 512 AOs are supported", "basis", "resource_limit")
    _require(
        {s.atom_index for s in native} == set(range(len(positions))),
        "Every atom must own at least one shell",
        "basis.shells",
    )
    return BasisSet(molecule, native, name, coefficients_pre_normalized=True)


def _occupied(spec, shape):
    _fields(spec, ("coefficients", "occupations"), field="orbitals")
    c_rows = _matrix(spec["coefficients"], "orbitals.coefficients")
    _require(
        c_rows.shape == shape,
        f"Coefficient shape must be {shape}, got {c_rows.shape}",
        "orbitals.coefficients",
    )
    occ = _array(spec["occupations"], "orbitals.occupations", shape[:-1])
    _require(
        np.all(occ == 2.0),
        "Only the complete doubly occupied restricted subspace is supported",
        "orbitals.occupations",
        "unsupported_occupations",
    )
    return c_rows.swapaxes(-1, -2)


def _molecular(request, molecule, basis, nocc):
    _require(
        request["method"] in LOCALIZATION_METHODS,
        "Method does not support molecules",
        "method",
        "unsupported_method",
    )
    reason = iao_unsupported_reason(molecule)
    _require(
        reason is None, reason or "Unsupported reference", "system", "unsupported_basis"
    )
    overlap = np.asarray(compute_overlap(basis))
    _metric(overlap, basis.nbasis)
    if request.get("source", "supplied") == "fresh_rhf":
        options = RHFOptions()
        options.max_iter = 100
        scf = run_rhf(molecule, basis, options)
        _require(
            scf.converged,
            "Explicit RHF calculation did not converge",
            code="scf_not_converged",
        )
        occupied = np.asarray(scf.mo_coeffs)[:, :nocc].copy()
    else:
        occupied = _occupied(request["orbitals"], (nocc, basis.nbasis))
        _require(
            not np.iscomplexobj(occupied),
            "Molecular localizers require real encoding; complex orbitals are unsupported",
            "orbitals.coefficients",
            "unsupported_complex",
        )
    _require(
        _orth_error(occupied, overlap) < TOL,
        "Supplied occupied orbitals are not orthonormal in the reconstructed AO metric",
        "orbitals.coefficients",
        "nonorthonormal_input",
    )
    analysis = analyse_localization(molecule, basis, occupied, method=request["method"])
    audit, rotation = _audit(occupied, analysis.coefficients, overlap)
    populations = analysis.atom_populations
    _require(
        np.max(np.abs(populations.sum(axis=1) - 1)) < TOL,
        "IAO populations do not span the supplied subspace",
        code="numerical_validation_failed",
    )
    return {
        "coefficients": _encode(analysis.coefficients.T),
        "occupations": [2.0] * nocc,
        "rotation": _encode(rotation),
        "representation": "molecular_ao",
        "atom_populations": populations.tolist(),
        "population_model": "iao-mini",
        "centroids_bohr": analysis.centroids.tolist(),
        "centroid_model": "position_expectation",
        "n_centres": analysis.n_centres.tolist(),
        "centre_threshold": 0.1,
        "charges": analysis.charges.tolist(),
        "charge_model": "iao-mini",
        "validation": audit,
        "warnings": [],
        "experimental": False,
    }


def _periodic(request, basis, positions, numbers, nocc):
    _require(
        request["method"] == "aiccm-wannier",
        "Periodic ibo, boys and pipek-mezey are not supported by protocol v1",
        "method",
        "unsupported_periodic_method",
    )
    _require(
        request.get("allow_experimental") is True,
        "aiccm-wannier requires allow_experimental=true",
        "allow_experimental",
        "experimental_opt_in_required",
    )
    _require(
        request.get("source", "supplied") == "supplied",
        "Periodic fresh SCF is unsupported",
        "source",
        "unsupported_source",
    )
    periodic = request["periodic"]
    _fields(
        periodic,
        (
            "lattice_bohr",
            "dimension",
            "pbc",
            "mesh",
            "kpoints_fractional",
            "weights",
            "overlap",
            "representation",
            "bloch_convention",
            "orbital_energies_hartree",
        ),
        field="periodic",
    )
    _require(
        periodic["representation"] == "finite_bvk",
        "Explicit finite_bvk provenance is required; arbitrary infinite-crystal Bloch data are not interchangeable",
        "periodic.representation",
        "unsupported_periodic_representation",
    )
    _require(
        periodic["bloch_convention"] == BLOCH_CONVENTION,
        "Unsupported Bloch phase convention",
        "periodic.bloch_convention",
        "unsupported_basis",
    )
    dim = _int(periodic["dimension"], 1, 3, "periodic.dimension")
    pbc = periodic["pbc"]
    _require(
        isinstance(pbc, list)
        and len(pbc) == 3
        and all(type(v) is bool for v in pbc)
        and pbc == [i < dim for i in range(3)],
        "Periodic axes must be the first dimension lattice vectors",
        "periodic.pbc",
        "unsupported_pbc",
    )
    lattice = _array(periodic["lattice_bohr"], "periodic.lattice_bohr", (3, 3))
    _require(
        abs(np.linalg.det(lattice)) > 1e-10,
        "Lattice must be nonsingular (vectors in columns)",
        "periodic.lattice_bohr",
    )
    mesh = periodic["mesh"]
    _require(
        isinstance(mesh, list) and len(mesh) == 3,
        "mesh must have three entries",
        "periodic.mesh",
    )
    mesh = tuple(_int(v, 1, 128, "periodic.mesh") for v in mesh)
    _require(
        all(mesh[i] == 1 for i in range(dim, 3)),
        "Nonperiodic mesh axes must equal 1",
        "periodic.mesh",
    )
    nk = int(np.prod(mesh))
    _require(
        nk * basis.nbasis <= 128,
        "Finite torus is limited to 128 AOs",
        "periodic.mesh",
        "resource_limit",
    )
    translations = np.asarray(
        list(itertools.product(*(range(n) for n in mesh))), dtype=int
    )
    kpoints = _array(
        periodic["kpoints_fractional"], "periodic.kpoints_fractional", (nk, 3)
    )
    expected = translations / np.asarray(mesh)
    _require(
        np.max(np.abs(kpoints - expected)) < 1e-12,
        "Expected the complete unshifted cyclic mesh in lexicographic order, coordinates in [0,1)",
        "periodic.kpoints_fractional",
        "unsupported_kmesh",
    )
    weights = _array(periodic["weights"], "periodic.weights", (nk,))
    _require(
        np.max(np.abs(weights - 1 / nk)) < 1e-12,
        "Full cyclic mesh requires uniform normalized weights",
        "periodic.weights",
        "unsupported_kmesh",
    )
    coefficients = _occupied(request["orbitals"], (nk, nocc, basis.nbasis))
    energies = _array(
        periodic["orbital_energies_hartree"], "periodic.orbital_energies_hartree"
    )
    _require(
        energies.ndim == 2
        and energies.shape[0] == nk
        and nocc < energies.shape[1] <= basis.nbasis,
        "Reference energies must include occupied bands and at least one virtual band at every k",
        "periodic.orbital_energies_hartree",
        "incomplete_band_manifold",
    )
    _require(
        np.all(np.diff(energies, axis=1) >= 0),
        "Reference band energies must be sorted at each k",
        "periodic.orbital_energies_hartree",
    )
    gap = float(energies[:, nocc].min() - energies[:, nocc - 1].max())
    _require(
        gap > 1e-6,
        "Periodic localization requires an isolated occupied manifold with gap > 1e-6 hartree",
        "periodic.orbital_energies_hartree",
        "unsupported_metallic_manifold",
    )
    overlaps = _matrix(periodic["overlap"], "periodic.overlap")
    _require(
        overlaps.shape == (nk, basis.nbasis, basis.nbasis),
        "One square AO overlap block is required per k point",
        "periodic.overlap",
    )
    for c, s in zip(coefficients, overlaps):
        _metric(s, basis.nbasis)
        _require(
            _orth_error(c, s) < TOL,
            "Occupied block is not S(k)-orthonormal",
            "orbitals.coefficients",
            "nonorthonormal_input",
        )
    # The finite real-space metric route requires S(-k)=conj(S(k)); no
    # imaginary coefficient components are removed, even at Gamma.
    lookup = {tuple(t): i for i, t in enumerate(translations)}
    for i, t in enumerate(translations):
        j = lookup[tuple((-t) % np.asarray(mesh))]
        _require(
            np.max(np.abs(overlaps[j] - overlaps[i].conj())) < 1e-10,
            "Overlap is incompatible with the real finite-torus metric (time reversal)",
            "periodic.overlap",
            "unsupported_complex_metric",
        )
    from .periodic.chi.localization import localize_aiccm2026dev_b_occupied_blocks
    from .properties import _shell_to_atom

    atoms = np.asarray(_shell_to_atom(basis), dtype=int)
    primitive_fractional = positions[atoms] @ np.linalg.inv(lattice).T
    fractional = np.concatenate([primitive_fractional + t for t in translations])
    atom_indices = np.concatenate([atoms + i * len(numbers) for i in range(nk)])
    result = localize_aiccm2026dev_b_occupied_blocks(
        coefficients,
        overlaps,
        kpoints,
        mesh,
        nocc,
        fractional,
        atom_indices,
        lattice,
        dimension=dim,
        method="wannier",
    )
    audit, rotation = _audit(
        result.canonical_wannier_coefficients, result.coefficients, result.overlap
    )
    audit["density_invariance_error"] = float(result.density_invariance_error)
    audit["translation_projector_error"] = float(result.translation_projector_error)
    populations = result.atomic_populations
    _require(
        np.max(np.abs(populations.sum(axis=1) - 1)) < TOL,
        "Finite-torus populations are not normalized",
        code="numerical_validation_failed",
    )
    warnings = [
        "Experimental finite-torus localization; centres are a projected circular AO-centre approximation, not exact position expectations."
    ]
    if np.any(result.aliasing_detected):
        warnings.append(
            "Localized weight reaches the cyclic antipode; enlarge the source torus before interpreting local domains."
        )
    return {
        "coefficients": _encode(result.coefficients.T),
        "occupations": [2.0] * (nk * nocc),
        "rotation": _encode(rotation),
        "representation": "finite_torus_ao",
        "reference_gap_hartree": gap,
        "translations": translations.tolist(),
        "lattice_bohr": lattice.tolist(),
        "atom_populations": populations.tolist(),
        "population_model": "lowdin_finite_torus",
        "centroids_bohr": result.centers_bohr.tolist(),
        "centroid_model": result.spread_model,
        "n_centres": (populations > 0.1).sum(axis=1).tolist(),
        "centre_threshold": 0.1,
        "charges": (np.tile(numbers, nk) - 2 * populations.sum(axis=0)).tolist(),
        "charge_model": "lowdin_finite_torus",
        "spreads_bohr2": result.spreads_bohr2.tolist(),
        "aliasing_fraction": result.aliasing_fraction.tolist(),
        "aliasing_detected": result.aliasing_detected.tolist(),
        "validation": audit,
        "experimental": True,
        "warnings": warnings,
    }


def localize(request):
    """Execute a validated protocol request; return a JSON-compatible result."""
    _fields(
        request,
        (
            "protocol",
            "protocol_version",
            "id",
            "operation",
            "method",
            "system",
            "basis",
        ),
        ("source", "orbitals", "periodic", "allow_experimental"),
    )
    _require(
        request["protocol"] == "vibeqc.relocalize"
        and type(request["protocol_version"]) is int
        and request["protocol_version"] == 1
        and request["operation"] == "localize",
        "Expected a protocol v1 localize request",
        code="unsupported_protocol",
    )
    _require(isinstance(request["method"], str), "method must be a string", "method")
    _require(
        type(request.get("allow_experimental", False)) is bool,
        "allow_experimental must be boolean",
        "allow_experimental",
    )
    system = request["system"]
    _fields(
        system,
        ("kind", "atomic_numbers", "positions_bohr", "charge", "multiplicity", "spin"),
        field="system",
    )
    _require(
        system["kind"] in ("molecular", "periodic"),
        "kind must be molecular or periodic",
        "system.kind",
    )
    numbers = system["atomic_numbers"]
    _require(
        isinstance(numbers, list) and 0 < len(numbers) <= 256,
        "Invalid atomic_numbers list",
        "system.atomic_numbers",
    )
    numbers = [_int(z, 1, 118, "system.atomic_numbers") for z in numbers]
    positions = _array(
        system["positions_bohr"], "system.positions_bohr", (len(numbers), 3)
    )
    charge = _int(system["charge"], -512, 512, "system.charge")
    _require(
        system["spin"] == "restricted"
        and type(system["multiplicity"]) is int
        and system["multiplicity"] == 1,
        "Only restricted singlets are supported",
        "system.spin",
        "unsupported_spin",
    )
    electrons = sum(numbers) - charge
    _require(
        electrons > 0 and electrons % 2 == 0,
        "Restricted singlet requires a positive even electron count",
        "system.charge",
    )
    source = request.get("source", "supplied")
    _require(
        source in ("supplied", "fresh_rhf"),
        "source must be supplied or fresh_rhf",
        "source",
        "unsupported_source",
    )
    _require(
        ("orbitals" in request) == (source == "supplied"),
        "supplied requires orbitals; fresh_rhf forbids them (no fallback)",
        "orbitals",
        "invalid_source",
    )
    _require(
        ("periodic" in request) == (system["kind"] == "periodic"),
        "Periodic metadata is required exactly when system.kind is periodic",
        "periodic",
    )
    molecule = Molecule(
        [Atom(z, p.tolist()) for z, p in zip(numbers, positions)],
        charge=charge,
        multiplicity=1,
    )
    basis = _basis(request["basis"], molecule, positions, source == "supplied")
    nocc = electrons // 2
    _require(
        nocc <= basis.nbasis <= 512,
        "Electron count or basis exceeds supported AO dimensions",
        "basis",
        "resource_limit",
    )
    if system["kind"] == "molecular":
        result = _molecular(request, molecule, basis, nocc)
    else:
        result = _periodic(request, basis, positions, numbers, nocc)
    result.update(
        method=request["method"],
        source=source,
        scf_performed=source == "fresh_rhf",
        spin="restricted",
        backend_version=package_version(),
        ao_convention=AO_CONVENTION,
        convergence="not_certified_by_kernel",
        citation_features={
            "ibo": ["ibo"],
            "boys": ["foster_boys", "ibo"],
            "pipek-mezey": ["pipek_mezey", "ibo"],
            "aiccm-wannier": ["wannier"],
        }[request["method"]],
    )
    return result


def probe(report):
    """Exercise native integrals, every advertised localizer, and explicit RHF."""
    from .output.formats.qvf import _basis_shell_payload

    # Two occupied orbitals, so the witness exercises rotations, not just import.
    positions = np.array([[0, 0, 0], [1.4, 0, 0], [0, 3, 0], [1.4, 3, 0]])
    molecule = Molecule(
        [Atom(1, p.tolist()) for p in positions], charge=0, multiplicity=1
    )
    basis = BasisSet(molecule, "sto-3g")
    overlap = np.asarray(compute_overlap(basis))
    _metric(overlap, 4)
    values, vectors = np.linalg.eigh(overlap)
    occupied = vectors[:, :2] / np.sqrt(values[:2])
    shells = _basis_shell_payload(basis)[0]
    request = {
        "protocol": "vibeqc.relocalize",
        "protocol_version": 1,
        "id": "probe",
        "operation": "localize",
        "method": "ibo",
        "system": {
            "kind": "molecular",
            "atomic_numbers": [1] * 4,
            "positions_bohr": positions.tolist(),
            "charge": 0,
            "multiplicity": 1,
            "spin": "restricted",
        },
        "basis": {
            "name": "sto-3g",
            "ao_convention": AO_CONVENTION,
            "uses_ecp": False,
            "shells": shells,
        },
        "orbitals": {"coefficients": _encode(occupied.T), "occupations": [2, 2]},
    }
    report["native_core_ready"] = True
    for method in LOCALIZATION_METHODS:
        try:
            localize(dict(request, method=method))
            report["methods"][method]["ready"] = True
        except Exception as exc:  # noqa: BLE001 - process/probe boundary
            report["methods"][method]["readiness_error"] = str(exc)
    try:
        # Actual RHF solve is independent of the supplied-orbital readiness.
        fresh = dict(request, source="fresh_rhf")
        fresh.pop("orbitals")
        localize(fresh)
        report["fresh_rhf"]["ready"] = True
    except Exception as exc:  # noqa: BLE001 - process/probe boundary
        report["fresh_rhf"]["readiness_error"] = str(exc)
    try:
        periodic = dict(request, method="aiccm-wannier", allow_experimental=True)
        periodic["system"] = dict(request["system"], kind="periodic")
        # Two k points and genuinely complex coefficients exercise the native
        # Fourier transform and complex projector audit without periodic SCF.
        periodic["orbitals"] = {
            "coefficients": _encode(np.array([occupied.T, 1j * occupied.T])),
            "occupations": [[2, 2], [2, 2]],
        }
        periodic["periodic"] = {
            "lattice_bohr": np.diag([8.0, 9.0, 10.0]).tolist(),
            "dimension": 1,
            "pbc": [True, False, False],
            "mesh": [2, 1, 1],
            "kpoints_fractional": [[0, 0, 0], [0.5, 0, 0]],
            "weights": [0.5, 0.5],
            "overlap": _encode(np.array([overlap, overlap])),
            "representation": "finite_bvk",
            "bloch_convention": BLOCH_CONVENTION,
            "orbital_energies_hartree": [[-1, -0.8, 0.3, 0.8]] * 2,
        }
        localize(periodic)
        report["methods"]["aiccm-wannier"]["ready"] = True
    except Exception as exc:  # noqa: BLE001 - process/probe boundary
        report["methods"]["aiccm-wannier"]["readiness_error"] = str(exc)
    report["ready"] = any(m["ready"] for m in report["methods"].values())
