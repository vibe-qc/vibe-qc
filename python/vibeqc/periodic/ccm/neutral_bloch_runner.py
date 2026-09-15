"""Runner adapter for the neutral fitted-torus Bloch producer (EXPERIMENTAL).

The sibling of :mod:`vibeqc.periodic.ccm.real_gamma_runner` for
``run_periodic_job(method="aiccm", variant="neutral-bloch")``. Ruling R1
(2026-08-21, :mod:`vibeqc.periodic.ccm.route`): Paper-1 Γ-CCM denotes the
**neutral finite-BvK-torus construction**, which has two admissible
producers. ``variant="real-gamma"`` evaluates it in the real-Γ supercell
representation; this module wires the other one, the Bloch representation
(:mod:`vibeqc.periodic.ccm.ri` ``run_ccm_{rhf,uhf,rks,uks}_gdf``), into the
runner. The two are Fourier-related evaluations of the *same* specified
Hamiltonian; neither is evidence for the union-and-weight four-centre
construction (``variant="four-center"``).

Why this adapter is thin where ``real_gamma_runner`` is thick: the real-Γ
drivers return per-**supercell** API dataclasses that have to be folded and
divided by ``N_c`` before the output stage can consume them, whereas the
Bloch producer already runs the production multi-k GDF driver on the unit
cell over the torus's own Γ-centred mesh, so
:class:`~vibeqc.periodic.ccm.ri.CCMGDFResult` carries a per-**unit-cell**
energy and wraps the very object ``jk_method="gdf"`` hands the output stage
(``.raw``). This adapter therefore performs **no fold and no division**: it
returns ``.raw`` with the convention record attached, which is what keeps the
Molden, density-grid, DOS and QVF artefacts working (each of those stages
``getattr``-guards its inputs and silently drops the artefact when they are
absent, so returning the ``CCMGDFResult`` wrapper instead would degrade the
run without an error).

What the arm is NOT: it is not ``jk_method="gdf"`` under another name. The
producer adds the neutral-torus guards the plain arm has no reason to carry
(the vacuum-padded refusal and the positive-converged-energy refusal, both
IID 291), it records the exchange-q=0 convention as a result field, and its
mesh IS the BvK torus rather than a Bloch sampling choice. The retired
``jk_method`` aliases ``neutral-bloch`` / ``bloch-control`` / ``gdf-control``
/ ``aiccm-ri``, which used to mean plain unit-cell GDF here, fail closed
(D-2b) precisely so the word keeps one meaning.

Experimental research surface: the producers emit
``AICCM2026DevAExperimentalWarning`` and the manifest record carries
``experimental=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..._vibeqc_core import PeriodicSystem

__all__ = [
    "CCMNeutralBlochConvention",
    "NEUTRAL_BLOCH_METHODS",
    "run_neutral_bloch_scf",
]

#: The SCF references this arm implements. ROHF/ROKS have no producer in
#: :mod:`vibeqc.periodic.ccm.ri` (there is no ``run_ccm_rohf_gdf``), so the
#: front door's ``scf_reference="rohf" | "roks"`` fails closed here rather
#: than being downgraded to an unrestricted reference.
NEUTRAL_BLOCH_METHODS = ("RHF", "RKS", "UHF", "UKS")


def _attach_returned_lattice_density(
    raw,
    system: PeriodicSystem,
    basis_name: str,
    mesh: tuple[int, int, int],
    *,
    options,
) -> None:
    """Return the producer's exact inverse-Bloch density representation.

    The generic output path deliberately refuses to invent a lattice density
    from an arbitrary result's per-k matrices. This producer is different: it
    owns the authenticated, complete Gamma-centred BvK mesh that generated
    ``raw``. Fold that state here, while the provenance is still available,
    and attach it alongside (never instead of) the production per-k density.
    The output path independently refolds and validates the returned lattice
    set before writing an artifact.
    """
    from vibeqc import make_basis, monkhorst_pack
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.periodic_k_symmetry import density_set_from_k_matrices

    kmesh = monkhorst_pack(system, list(mesh))
    expected_kpoints = np.asarray(kmesh.kpoints, dtype=float).reshape(-1, 3)
    expected_weights = np.asarray(kmesh.weights, dtype=float).reshape(-1)
    returned_kpoints = np.asarray(
        getattr(raw, "kpoints_cart", None), dtype=float
    )
    returned_weights = np.asarray(
        getattr(raw, "kpoint_weights", None), dtype=float
    ).reshape(-1)
    if (
        returned_kpoints.shape != expected_kpoints.shape
        or returned_weights.shape != expected_weights.shape
        or not np.isfinite(returned_kpoints).all()
        or not np.isfinite(returned_weights).all()
        or not np.allclose(
            returned_kpoints, expected_kpoints, rtol=0.0, atol=1.0e-12
        )
        or not np.allclose(
            returned_weights, expected_weights, rtol=0.0, atol=1.0e-14
        )
    ):
        raise ValueError(
            "neutral-Bloch SCF result does not carry the exact Gamma-centred "
            "BvK mesh requested by the producer"
        )

    basis = make_basis(system.unit_cell_molecule(), basis_name)
    lattice_opts = getattr(options, "lattice_opts", None)
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()

    has_alpha = hasattr(raw, "density_alpha")
    has_beta = hasattr(raw, "density_beta")
    if has_alpha != has_beta:
        raise ValueError(
            "neutral-Bloch SCF result carries only one spin density channel"
        )
    if has_alpha:
        setattr(
            raw,
            "density_alpha_lattice",
            density_set_from_k_matrices(
                system, basis, lattice_opts, kmesh, raw.density_alpha
            ),
        )
        setattr(
            raw,
            "density_beta_lattice",
            density_set_from_k_matrices(
                system, basis, lattice_opts, kmesh, raw.density_beta
            ),
        )
    else:
        setattr(
            raw,
            "density_lattice",
            density_set_from_k_matrices(
                system, basis, lattice_opts, kmesh, raw.density
            ),
        )


@dataclass(frozen=True)
class CCMNeutralBlochConvention:
    """What the neutral Bloch producer actually executed, for the record.

    Every field is a **read** off the producer's result or off the inputs it
    was handed, never a value re-derived at the call site (the D86
    discipline that keeps the two producers' records auditable against each
    other).
    """

    #: ``"neutral-bloch"``: the front-door variant name, which is also the
    #: ``jk_method_executed`` value and the ``routes.methods`` citation key.
    route: str
    #: The q=0 exchange-divergence convention the run applied, read from
    #: :attr:`~vibeqc.periodic.ccm.ri.CCMGDFResult.exchange_q0` (which is
    #: itself keyed on the driver's computed exact-exchange energy).
    exchange_q0: str
    #: ``"active"`` iff exact exchange is present, else ``"inactive"``.
    exchange_q0_applicability: str
    #: vibe-qc lattice matrices hold lattice vectors as columns; recorded so
    #: a cross-code reader never has to guess.
    lattice_vector_convention: str
    #: The Born-von Karman torus this run represents: the cyclic cluster
    #: size along a1, a2, a3, which is also the Γ-centred k-mesh executed.
    nrep: tuple
    #: ``N_c``, the number of unit cells on the torus.
    n_cells: int
    #: Whether the CCM pair-star (space-group / time-reversal) reduction of
    #: the Lpq fits was requested. The runner arm defaults it OFF so the
    #: executed Hamiltonian is the production multi-k GDF one; the reduction
    #: path reconstructs fits from a star and its finite cell-list residual
    #: is documented as quantitative rather than algebraically zero.
    pair_symmetry: bool
    #: Expensive per-k-pair fit builds actually performed.
    gdf_pair_builds: int
    #: Fit builds the unreduced production route would need.
    gdf_pair_total: int
    #: ``gdf_pair_total / gdf_pair_builds`` (one when reduction is inactive).
    gdf_pair_reduction_factor: float
    #: The RSGDF high-|G| tail cutoff the inner driver applied, or ``None``.
    #: Read from the driver result: the two producers recording different
    #: values here was worth -4.99e-01 Ha/cell on MgO/STO-3G (IID 307).
    rsgdf_tail_ke_cutoff: Optional[float]
    #: The inner driver's backend string, carried verbatim including any
    #: ``+PARITY_HELD`` marker.
    backend: str
    #: Whether the inner multi-k driver held this result for parity.
    parity_held: bool
    #: Executed Fock mixing and level shift, READ off the multi-k result
    #: rather than assumed (D86). Unlike the real-Γ loops, which implement
    #: neither and record structural zeros, these drivers do implement both,
    #: so the recorded value is whatever the SCF ran with.
    executed_fock_mixing: float
    executed_level_shift: float
    experimental: bool = True


def run_neutral_bloch_scf(
    system: PeriodicSystem,
    basis_name: str,
    method: str,
    mesh: Sequence[int],
    *,
    functional: Optional[str] = None,
    aux_basis: Optional[str] = None,
    symmetry: bool = False,
    return_lattice_density: bool = False,
    **driver_kwargs,
):
    """Run the neutral Γ-CCM Bloch producer on the BvK torus ``mesh``.

    Returns the production multi-k GDF result (``PeriodicKRHFGDFResult`` and
    friends) with ``neutral_bloch`` (a
    :class:`CCMNeutralBlochConvention`), ``ccm_result`` (the producer's
    :class:`~vibeqc.periodic.ccm.ri.CCMGDFResult` wrapper) and
    ``parity_held`` attached, so ``run_periodic_job``'s output stage consumes
    it exactly as it consumes a ``jk_method="gdf"`` result.

    Parameters
    ----------
    system
        The periodic **unit cell**. The torus is ``mesh``; the producer
        replicates the cell over it.
    basis_name
        Basis-set name; the producer rebuilds the basis on the unit cell.
    method
        One of :data:`NEUTRAL_BLOCH_METHODS` (case-insensitive).
    mesh
        The BvK torus ``(N1, N2, N3)``: three positive integers. It is a
        cluster size, not a Bloch sampling choice, so a shifted mesh has no
        meaning here and the runner rejects one before this call.
    functional
        Required for the KS references, ignored for the HF ones.
    symmetry
        Request the CCM pair-star reduction of the Lpq fits. **Off by
        default on purpose**: with it off the arm executes the same
        Hamiltonian as the production multi-k GDF route on the same mesh,
        which is what makes the Fourier identity with ``variant="real-gamma"``
        a statement about the construction rather than about a fit
        approximation. The producer only takes the reduction path at all on a
        partially-periodic replica mesh with the RSGDF backend.
    return_lattice_density
        Also return the exact lattice-cell representation of the converged
        per-k density. The runner requests this for density-grid and QVF
        output; the original per-k matrices remain untouched.
    **driver_kwargs
        Forwarded verbatim to the producer and on into
        ``run_k{rhf,rks,uhf,uks}_periodic_gdf`` (``options``, ``progress``,
        ``gdf_method``, ``rsgdf_ke_cutoff``, ``mdf_ke_cutoff``,
        ``fock_mixing``, the density-mixer family, ...).
    """
    from .ri import (
        run_ccm_rhf_gdf,
        run_ccm_rks_gdf,
        run_ccm_uhf_gdf,
        run_ccm_uks_gdf,
    )
    from .system import CCMSystem

    method_upper = str(method).strip().upper()
    nrep = tuple(int(n) for n in mesh)
    if len(nrep) != 3 or any(n < 1 for n in nrep):
        raise ValueError(
            "run_neutral_bloch_scf: mesh must be three positive integers "
            f"(the BvK torus nrep); got {mesh!r}"
        )
    if method_upper not in NEUTRAL_BLOCH_METHODS:
        raise NotImplementedError(
            "variant='neutral-bloch' implements "
            f"{', '.join(NEUTRAL_BLOCH_METHODS)}; got method={method_upper!r}. "
            "The Bloch producer has no restricted-open-shell entry in "
            "vibeqc.periodic.ccm.ri."
        )
    if method_upper in ("RKS", "UKS") and functional is None:
        raise ValueError(
            f"run_neutral_bloch_scf: method={method_upper!r} requires "
            "functional=..."
        )

    ccm = CCMSystem(system, nrep, basis_name)
    if method_upper == "RHF":
        res = run_ccm_rhf_gdf(
            ccm, aux_basis=aux_basis, symmetry=symmetry, **driver_kwargs
        )
    elif method_upper == "RKS":
        res = run_ccm_rks_gdf(
            ccm, functional, aux_basis=aux_basis, symmetry=symmetry,
            **driver_kwargs,
        )
    elif method_upper == "UHF":
        # The open-shell producers take no ``symmetry`` keyword: it would be
        # forwarded into run_kuhf_periodic_gdf, which has no such parameter.
        res = run_ccm_uhf_gdf(ccm, aux_basis=aux_basis, **driver_kwargs)
    else:  # UKS
        res = run_ccm_uks_gdf(
            ccm, functional, aux_basis=aux_basis, **driver_kwargs
        )

    raw = res.raw
    if return_lattice_density:
        _attach_returned_lattice_density(
            raw,
            system,
            basis_name,
            nrep,
            options=driver_kwargs.get("options"),
        )
    convention = CCMNeutralBlochConvention(
        route="neutral-bloch",
        exchange_q0=str(res.exchange_q0 or ""),
        exchange_q0_applicability=str(res.exchange_q0_applicability),
        lattice_vector_convention="columns",
        nrep=nrep,
        n_cells=int(ccm.n_cells),
        pair_symmetry=bool(symmetry),
        gdf_pair_builds=int(res.gdf_pair_builds),
        gdf_pair_total=int(res.gdf_pair_total),
        gdf_pair_reduction_factor=float(res.gdf_pair_reduction_factor),
        rsgdf_tail_ke_cutoff=getattr(res, "rsgdf_tail_ke_cutoff", None),
        backend=str(res.backend or ""),
        parity_held=bool(res.parity_held),
        executed_fock_mixing=float(getattr(raw, "fock_mixing", 0.0) or 0.0),
        executed_level_shift=float(getattr(raw, "level_shift", 0.0) or 0.0),
    )
    # The multi-k GDF results are plain (non-frozen) dataclasses, so the
    # record rides on the object the output stage already understands. The
    # chi line attaches its own convention the same way.
    setattr(raw, "neutral_bloch", convention)
    setattr(raw, "ccm_result", res)
    setattr(raw, "parity_held", bool(res.parity_held))
    return raw
