"""Owned ECP inputs for the Ewald UKS derivative consumer.

This is a provider receipt using the shared source vocabulary, not another
Fock state or integral engine. Native ECP objects are copied into JSON values.
"""
import hashlib
import os

import numpy as np

from .bipole_domain import GRID_FIELDS, LATTICE_FIELDS, OptionSnapshot
from .fock_state import SourceProvenance


_PROVIDER = "ewald-uks-ecp-derivative-v1"


def _frame(system, basis, kmesh):
    return dict(
        dimension=int(system.dim), lattice=np.asarray(system.lattice).tolist(),
        charge=int(system.charge), multiplicity=int(system.multiplicity),
        atoms=[dict(Z=int(a.Z), xyz=list(a.xyz)) for a in system.unit_cell],
        shells=[dict(atom_index=int(s.atom_index), l=int(s.l), pure=bool(s.pure),
                     exponents=list(s.exponents), coefficients=list(s.coefficients),
                     origin=list(s.origin)) for s in basis.shells()],
        kpoints=[np.asarray(k).tolist() for k in kmesh.kpoints],
        weights=list(kmesh.weights),
    )


def capture_ewald_uks_ecp_source(system, basis, kmesh, options, lattice_options,
                               context, omega):
    """Freeze the ECP Hamiltonian and numerical support before its build."""
    if context is None:
        return None
    blocks, centers, charges, ncore, _ = context
    electrons = int(system.n_electrons()) - int(ncore)
    spin = int(system.multiplicity) - 1
    return SourceProvenance.capture(
        _PROVIDER, frame=_frame(system, basis, kmesh),
        blocks=[dict(n_primitive=int(b.n_primitive), exponents=list(b.exponents),
                     coefficients=list(b.coefficients), ams=list(b.ams), ns=list(b.ns))
                for b in blocks],
        centers=[list(c) for c in centers], effective_charges=list(charges),
        ncore=int(ncore), spin_populations=[(electrons + spin)//2, (electrons - spin)//2],
        lattice_options=OptionSnapshot.capture(lattice_options, LATTICE_FIELDS).values,
        grid_options=OptionSnapshot.capture(options.grid, GRID_FIELDS).values,
        use_periodic_becke=bool(options.use_periodic_becke),
        becke_image_radius_bohr=float(options.becke_image_radius_bohr),
        functional=str(options.functional), omega=float(omega),
        hartree_backend=os.environ.get("VIBEQC_J_EWALD3D_BACKEND", "analytic_ft").lower(),
        nuclear_backend=os.environ.get("VIBEQC_VNE_EWALD3D_BACKEND", "analytic_ft").lower(),
        hartree_ke=float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0")),
        nuclear_ke=float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0")),
    )


def _tuple(value):
    return tuple(_tuple(x) for x in value) if isinstance(value, list) else value


def _array_identity(value):
    array = np.asarray(value)
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


def _terminal_state(result):
    """Identify exactly the state read by this derivative consumer.

    The core/Pulay, projector and moving-XC terms read different views of
    the same SCF state. Retain hashes, not a second set of density matrices.
    This is a provider-local receipt using the shared SourceProvenance API.
    """
    state = dict(converged=bool(result.converged),
                 ecp_operator=bool(result.ecp_operator),
                 ecp_total_ncore=int(result.ecp_total_ncore),
                 smearing_temperature=float(result.smearing_temperature),
                 bz_integration=result.bz_integration)
    for spin in ("alpha", "beta"):
        for field in ("mo_coeffs", "mo_energies", "occupations", "density"):
            name = f"{field}_{spin}" + ("_k" if field == "density" else "")
            state[name] = [_array_identity(value) for value in getattr(result, name)]
        density = getattr(result, f"density_{spin}")
        state[f"density_{spin}"] = dict(
            nbf=int(density.nbf),
            cells=[dict(index=list(map(int, cell.index)),
                        r_cart=list(map(float, cell.r_cart))) for cell in density.cells],
            blocks=[_array_identity(block) for block in density.blocks],
        )
    return state


def seal_ewald_uks_ecp_source(source, result):
    """Bind the pre-build Hamiltonian receipt to the returned terminal state."""
    return SourceProvenance.capture(
        source.provider, **source.as_dict()["details"], terminal_state=_terminal_state(result),
    )


def resolve_ewald_uks_ecp_source(system, basis, result, kmesh, lattice_options,
                               grid_options):
    """Restore the accepted source and admit only the qualified force envelope."""
    from ._vibeqc_core import Atom, ECPPrimitiveBlock, GridOptions, LatticeSumOptions, PeriodicSystem

    source = getattr(result, "ecp_derivative_source", None)
    if not isinstance(source, SourceProvenance) or source.provider != _PROVIDER:
        raise NotImplementedError("ECP analytic gradient requires the retained Ewald UKS source")
    details = source.as_dict()["details"]
    if "terminal_state" not in details:
        raise NotImplementedError(
            "ECP gradient requires a retained terminal SCF state; rerun the SCF")
    try:
        state = _terminal_state(result)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("ECP gradient terminal SCF state is missing or invalid") from exc
    if not state["converged"] or state != details["terminal_state"]:
        raise ValueError("ECP gradient terminal SCF state differs from its retained source")
    if details["frame"] != _frame(system, basis, kmesh):
        raise ValueError("ECP gradient geometry, basis or k-mesh differs from the SCF source")
    if (details["functional"] != result.functional
            or details["omega"] != float(result.omega)
            or details["ncore"] != int(result.ecp_total_ncore)):
        raise ValueError("ECP gradient result differs from its retained SCF source")
    if (int(system.dim) != 3 or details["hartree_backend"] != "analytic_ft"
            or details["nuclear_backend"] != "analytic_ft"
            or details["hartree_ke"] != 200.0 or details["nuclear_ke"] != 200.0):
        raise NotImplementedError("ECP gradient requires the qualified 3D analytic-FT source")
    lat = OptionSnapshot(_tuple(details["lattice_options"]))
    lat.require_match(lattice_options, context="ECP gradient")
    grid = OptionSnapshot(_tuple(details["grid_options"]))
    if grid_options is not None:
        grid.require_match(grid_options, context="ECP gradient")
    for name, count in zip(("occupations_alpha", "occupations_beta"), details["spin_populations"]):
        occupations = getattr(result, name, ())
        if len(occupations) != len(kmesh.kpoints):
            raise NotImplementedError("ECP gradient requires retained valence spin occupations")
        for occupied in occupations:
            occupied = np.asarray(occupied)
            if occupied.ndim != 1 or count > occupied.size:
                raise NotImplementedError("ECP gradient requires uniform integer valence spin occupations")
            expected = np.zeros_like(occupied)
            expected[:count] = 1.0
            if not np.array_equal(occupied, expected):
                raise NotImplementedError("ECP gradient requires uniform integer valence spin occupations")
    positions = np.asarray([a.xyz for a in system.unit_cell])
    for shell in basis.shells():
        atom = int(shell.atom_index)
        if (atom < 0 or atom >= len(positions)
                or np.linalg.norm(positions[atom] - shell.origin) >= 1e-10):
            raise NotImplementedError("ECP gradient requires atom-centered basis shells")
    for center in details["centers"]:
        if np.count_nonzero(np.linalg.norm(positions - center, axis=1) < 1e-10) != 1:
            raise NotImplementedError("ECP gradient requires each projector center to follow one atom")
    blocks = []
    for record in details["blocks"]:
        block = ECPPrimitiveBlock()
        for name, value in record.items():
            setattr(block, name, value)
        blocks.append(block)
    ionic = PeriodicSystem(system.dim, system.lattice,
        [Atom(round(q), list(a.xyz)) for q, a in zip(details["effective_charges"], system.unit_cell)],
        system.charge, system.multiplicity)
    return (ionic, blocks, details["centers"], lat.restore(LatticeSumOptions),
            grid.restore(GridOptions), details["use_periodic_becke"],
            details["becke_image_radius_bohr"])
