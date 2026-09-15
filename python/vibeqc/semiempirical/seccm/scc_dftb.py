"""SCC-DFTB energy over a frozen SECCM topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibeqc._vibeqc_core import Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc.semiempirical.parameters import default_parameters
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan

from ._adapter_common import flatten_topology_records, topology_length_unit_scale
from .topology import SECCMTopology


SCCDFTB_SECCM_PARAMETER_SET = "vibeqc-inhouse-dftb-screening-v1"


@dataclass(frozen=True)
class SCCDFTBSECCMResult:
    """One finite-cluster SCC-DFTB-SECCM result, per primitive cell.

    At finite electronic temperature ``e_electronic`` and
    ``cyclic_electronic_energy`` include the Mermin ``-T*S`` term, so the
    reported components close exactly to ``energy`` and
    ``total_cyclic_energy`` respectively.
    """

    energy: float
    e_electronic: float
    e_repulsive: float
    e_scc: float
    e_madelung: float
    free_energy: float
    entropy: float
    smearing_temperature: float
    total_cyclic_energy: float
    cyclic_electronic_energy: float
    cyclic_repulsive_energy: float
    cyclic_scc_energy: float
    cyclic_madelung_energy: float
    homo_lumo_gap: float
    #: Guard epsilon (Ha) actually applied to ``homo_lumo_gap``.
    finite_torus_gap_tolerance: float
    #: True when ``homo_lumo_gap`` is at or below
    #: ``finite_torus_gap_tolerance`` and the positive-gap requirement was
    #: waived because finite electronic temperature resolves the occupation.
    #: A waived row is admissible but is **not** a gapped row: its energy is
    #: a Mermin free energy on a numerically degenerate frontier.  At
    #: ``electronic_temperature = 0`` such a state is rejected instead.
    gap_guard_waived: bool
    #: The occupation the accepted density was actually built from: the
    #: Fermi-Dirac occupations at ``smearing_temperature``, or the integer
    #: Aufbau occupation at ``electronic_temperature = 0``.
    occupations: np.ndarray
    #: ``max_i |f_i - f_i^Aufbau|`` over :attr:`occupations`, exactly ``0.0``
    #: on the T = 0 branch by construction.
    #:
    #: Weinert and Davenport, Phys. Rev. B 45, 13709 (1992), Eqs. (8) and
    #: (10'): a fractional-occupation functional differs from the
    #: fixed-integer-occupation one by exactly the ``-T*S`` term this route
    #: subtracts, so a row with a nonzero deviation is a stationary point of a
    #: *different* functional and must not be compared against an Aufbau row.
    #: A value at or above ``1.0`` means the chemical potential has reached
    #: the Aufbau LUMO: the frontier is unresolved and the row is metallic,
    #: not merely thermally broadened.  That is the state
    #: :attr:`gap_guard_waived` structurally cannot see, because it compares
    #: the gap to an absolute epsilon in Ha while the scale that resolves an
    #: occupation under smearing is kT (issue 302).
    aufbau_occupation_deviation: float
    #: Roundoff epsilon applied to :attr:`aufbau_occupation_deviation`, so the
    #: verdict is reproducible from the record alone.
    aufbau_occupation_tolerance: float
    #: True when the applied occupation is the integer Aufbau occupation to
    #: within :attr:`aufbau_occupation_tolerance`, i.e. when
    #: :attr:`energy` is on the Aufbau surface rather than the Mermin
    #: free-energy surface.  Always true at ``electronic_temperature = 0``.
    aufbau_occupation: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    overlap: np.ndarray
    hamiltonian: np.ndarray
    charges: np.ndarray
    gradient: np.ndarray | None
    gradient_method: str | None
    n_basis: int
    n_occ: int
    n_iter: int
    group_order: int
    n_records: int
    converged: bool
    normalization: str
    parameter_set: str
    parameter_identity: str
    parameter_sha256: str
    topology_fingerprint: str
    reference_geometry_fingerprint: str
    route_plan: SemiempiricalRoutePlan


def _frozen_array(value) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result




def _records(molecule: Molecule, topology: SECCMTopology):
    """Flatten the validated WS records into the C++ record arrays."""
    return flatten_topology_records(
        molecule, topology, route_name="SCC-DFTB-SECCM"
    )


def run_scc_dftb_seccm(
    molecule: Molecule,
    topology: SECCMTopology,
    *,
    parameter_set: str = SCCDFTB_SECCM_PARAMETER_SET,
    max_iter: int = 500,
    conv_tol_charge: float = 1.0e-8,
    charge_mixing: float = 0.2,
    madelung: bool = False,
    gamma_form: str = "klopman_ohno",
    ewald_gamma: bool = False,
    use_diis: bool = False,
    diis_subspace: int = 6,
    use_broyden: bool = False,
    broyden_memory: int = 6,
    broyden_damping: float = 0.4,
    electronic_temperature: float = 0.0,
    compute_gradient: bool = False,
) -> SCCDFTBSECCMResult:
    """Evaluate the SCC-DFTB-SECCM finite-cluster route.

    The route is restricted to closed-shell, insulating clusters with one to
    three cyclic dimensions. It evaluates the complete frozen WS record set,
    iterates the supercell Mulliken charge fluctuations to self-consistency
    with vector Aitken relaxation, and normalizes the total energy by the
    finite-group order.

    ``gamma_form`` selects the functional form of the SCC-DFTB gamma:
    ``"klopman_ohno"`` (the default, the shipped in-house form
    ``1/sqrt(R^2 + eta^2)``) or ``"elstner"`` (Elstner et al. 1998
    Eq. 17/18, ``gamma = 1/R - S``, ``tau = 16/5 U``). The Wigner-Seitz
    truncation of the Klopman-Ohno remainder has no thermodynamic limit --
    an ``R^-3`` tail with a pair-dependent amplitude -- while the Elstner
    remainder decays exponentially, which is why 3-D embedding requires it
    (issue #211). The default is unchanged pending the staged molecular
    cutover (maintainer decision D1); see
    ``handovers/HANDOVER_GFN2_PERIODIC_ELECTROSTATICS.md``.

    ``ewald_gamma=True`` sums the Coulomb tail *into* the gamma kernel --
    full bare-Coulomb Ewald plus the WS-folded remainder and the on-site
    hardness, in one, two or three dimensions -- rather than adding it
    afterwards as a separate embedding potential. It is mutually exclusive
    with ``madelung``, which is the same physics by the other route.

    ``madelung=True`` switches on long-range Madelung/Ewald embedding in
    one or two dimensions. Its Mulliken-charge potential joins the ordinary
    SCC potential in the overlap-weighted variational Hamiltonian, and its
    classical self-energy joins the Mermin free energy. Three-dimensional
    embedding fails closed because the current Ohno correction has no
    thermodynamic limit. Unembedded 3-D finite-torus calculations require
    odd replica counts on all three axes; even-replica Nyquist ownership is
    not yet validated and fails closed.
    ``use_diis=True`` switches the charge iteration from the
    default vector-Aitken relaxation to the damped Pulay-DIIS accelerator;
    the stiff charge response of an embedded charged supercell usually
    needs it. ``use_broyden=True`` switches it to the modified Broyden
    quasi-Newton mixer (CP2K/tblite pattern) with ``broyden_memory``
    history vectors and ``broyden_damping`` as the initial mixing
    fraction; it accelerates ordinary chains (the 2-cell regression
    converges faster than DIIS). Charged polar 1-D chains with ODD
    replica counts sit at a near-degenerate T=0 frontier and need a small
    ``electronic_temperature`` instead of a stronger mixer.
    ``compute_gradient=True`` differentiates the converged per-cell energy
    analytically on the frozen record set. Neutral cells use the
    fixed-charge form (same structure as the molecular SCC-DFTB gradient:
    M = -W + D . 1/2 (kappa hbar - V_A - V_B) contracted with the live
    overlap derivatives, the WS-weighted repulsive derivative, and the
    gamma derivative over live record displacements). Embedded cells
    (``madelung=True``) add the fixed-charge Madelung derivative. The total
    charge interaction is variational, so no coupled-perturbed charge term
    remains and both paths report ``gradient_method="analytic"``.
    ``electronic_temperature`` switches on
    Fermi-Dirac fractional occupations (Ha; 0 = hard Aufbau, the default
    and the molecular-parity path): charged polar 1-D chains with the
    converged background-corrected Ewald embedding sit at a near-degenerate
    T=0 frontier whose response no mixer contracts, and a small
    temperature (e.g. 0.005 Ha) smooths the occupation so the SCC
    converges; the reported energy is then the Mermin free energy
    A = E - T*S (``free_energy`` alias) with ``entropy`` and
    ``smearing_temperature`` recorded. The occupation actually applied is
    recorded too, as ``occupations`` plus ``aufbau_occupation`` and
    ``aufbau_occupation_deviation``: a smeared row is a stationary point of
    the fractional-occupation functional, not of the Aufbau one, so screen on
    ``aufbau_occupation`` before comparing a smeared energy against a T = 0
    reference. Screening on ``gap_guard_waived`` alone is not enough -- that
    flag compares the gap to an absolute epsilon in Ha, and a frontier can be
    fully unresolved (equally occupied, i.e. metallic) at a gap far above it
    when the gap is small in units of kT. No dispersion, no stress, and no
    topology derivative are implied.
    """
    if parameter_set != SCCDFTB_SECCM_PARAMETER_SET:
        raise ValueError(
            "SCC-DFTB-SECCM supports only parameter_set="
            f"{SCCDFTB_SECCM_PARAMETER_SET!r}"
        )
    _GAMMA_FORM_BY_NAME = {
        "klopman_ohno": _se_cxx.ShellGammaForm.KlopmanOhno,
        "elstner": _se_cxx.ShellGammaForm.Elstner,
    }
    if gamma_form not in _GAMMA_FORM_BY_NAME:
        raise ValueError(
            f"gamma_form must be one of {sorted(_GAMMA_FORM_BY_NAME)}"
        )
    if madelung and ewald_gamma:
        raise ValueError(
            "SCC-DFTB-SECCM madelung and ewald_gamma are mutually exclusive: "
            "the Ewald-summed gamma already carries the Coulomb tail"
        )
    if (
        madelung
        and topology.dimensionality == 3
        and gamma_form != "elstner"
    ):
        raise NotImplementedError(
            "SCC-DFTB-SECCM 3-D Madelung embedding is unavailable with the "
            "Klopman-Ohno gamma: its Ohno correction has no thermodynamic "
            "limit. Pass gamma_form='elstner' for a kernel that does (#211)."
        )
    if int(molecule.charge) != 0 and not madelung:
        raise NotImplementedError(
            "charged SCC-DFTB-SECCM cells require the opt-in Madelung/Ewald "
            "embedding; pass madelung=True for a charged supercell"
        )
    group = topology.finite_group
    # Noga et al. (1999) adopt odd counts as a natural central-cell
    # convention, not a mathematical requirement; Peintinger and Bredow
    # (2014) define fractional WSSC boundary ownership. The observed even
    # branch diverges, and no route-specific Nyquist convention is validated.
    if (
        topology.dimensionality == 3
        and group is not None
        and any(replica % 2 == 0 for replica in group.replicas[:3])
    ):
        raise NotImplementedError(
            "SCC-DFTB-SECCM 3-D even-replica/Nyquist ownership is not "
            "validated; all active 3-D replica counts must be odd"
        )
    route_plan = SemiempiricalRoutePlan.from_request(
        "scc_dftb",
        boundary="seccm",
        properties=("energy", "gradient")
        if compute_gradient
        else ("energy",),
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
    ).with_seccm_runtime(
        periodic_dimension=topology.dimensionality,
        electrostatics_family="madelung" if madelung else "none",
        electronic_temperature=electronic_temperature,
    )
    (
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        primitive_vectors,
        replicas,
    ) = _records(molecule, topology)

    assert group is not None
    params = default_parameters()
    # Loudly flag placeholder repulsive pairs before any number is
    # produced (issue #306): fixed-geometry SECCM differences stay valid,
    # absolute energies/EOS fits for such systems are not chemistry.
    from vibeqc.semiempirical.dftb0 import warn_placeholder_repulsives

    warn_placeholder_repulsives(
        params,
        [int(atom.Z) for atom in molecule.atoms],
        route="run_scc_dftb_seccm",
    )
    native = _se_cxx._run_scc_dftb_seccm_from_records(
        molecule,
        params,
        translations,
        central,
        origin,
        np.asarray(shell_labels, dtype=np.int32),
        weights,
        multiplicities,
        np.asarray(displacements, dtype=float),
        primitive_vectors,
        replicas,
        float(group.geometry_tolerance) * topology_length_unit_scale(topology),
        int(max_iter),
        float(conv_tol_charge),
        float(charge_mixing),
        bool(madelung),
        _GAMMA_FORM_BY_NAME[gamma_form],
        bool(ewald_gamma),
        bool(use_diis),
        int(diis_subspace),
        bool(use_broyden),
        int(broyden_memory),
        float(broyden_damping),
        float(electronic_temperature),
        bool(compute_gradient),
    )
    if not bool(native.converged):
        raise RuntimeError(
            "C++ SCC-DFTB-SECCM did not converge within the iteration "
            "budget"
        )
    gradient = (
        _frozen_array(native.gradient)
        if bool(native.has_gradient)
        else None
    )
    gradient_method = None
    if compute_gradient:
        gradient_method = "analytic"
    return SCCDFTBSECCMResult(
        energy=float(native.energy),
        e_electronic=float(native.e_electronic),
        e_repulsive=float(native.e_repulsive),
        e_scc=float(native.e_scc),
        e_madelung=float(native.e_madelung),
        free_energy=float(native.free_energy),
        entropy=float(native.entropy),
        smearing_temperature=float(native.smearing_temperature),
        total_cyclic_energy=float(native.total_cyclic_energy),
        cyclic_electronic_energy=float(native.cyclic_electronic_energy),
        cyclic_repulsive_energy=float(native.cyclic_repulsive_energy),
        cyclic_scc_energy=float(native.cyclic_scc_energy),
        cyclic_madelung_energy=float(native.cyclic_madelung_energy),
        homo_lumo_gap=float(native.homo_lumo_gap),
        finite_torus_gap_tolerance=float(native.finite_torus_gap_tolerance),
        gap_guard_waived=bool(native.gap_guard_waived),
        occupations=_frozen_array(native.occupations),
        aufbau_occupation_deviation=float(native.aufbau_occupation_deviation),
        aufbau_occupation_tolerance=float(native.aufbau_occupation_tolerance),
        aufbau_occupation=bool(native.aufbau_occupation),
        mo_energies=_frozen_array(native.mo_energies),
        mo_coeffs=_frozen_array(native.mo_coeffs),
        density=_frozen_array(native.density),
        overlap=_frozen_array(native.overlap),
        hamiltonian=_frozen_array(native.hamiltonian),
        charges=_frozen_array(native.charges),
        gradient=(
            _frozen_array(gradient)
            if gradient is not None
            else None
        ),
        gradient_method=gradient_method,
        n_basis=int(native.n_basis),
        n_occ=int(native.n_occ),
        n_iter=int(native.n_iter),
        group_order=int(native.group_order),
        n_records=int(native.n_records),
        converged=bool(native.converged),
        normalization="per_primitive_cell",
        parameter_set=parameter_set,
        parameter_identity=str(native.parameter_identity),
        parameter_sha256=str(native.parameter_sha256),
        topology_fingerprint=topology.topology_fingerprint,
        reference_geometry_fingerprint=topology.reference_geometry_fingerprint,
        route_plan=route_plan,
    )


__all__ = [
    "SCCDFTB_SECCM_PARAMETER_SET",
    "SCCDFTBSECCMResult",
    "run_scc_dftb_seccm",
]
