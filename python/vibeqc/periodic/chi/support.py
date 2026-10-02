"""Explicit finite-support contract for the experimental direct chi RHF/UHF routes.

This is an adapter to the shared periodic kernels, not a second construction of
Coulomb or exchange. A support record is provenance, never an accuracy verdict.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


POLICIES = ("physical_pairs", "historical_index_ball")


def resolve_support_policy(policy, *, backend, dimension, ecp=False):
    """An omitted policy preserves legacy routing; explicit requests fail closed."""
    if policy is None:
        return None
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError("support_policy must be physical_pairs or historical_index_ball")
    if backend != "four_center" or dimension != 3 or ecp:
        raise NotImplementedError(
            "explicit chi support_policy requires all-electron 3D four_center RHF/UHF; "
            "fitted, ECP, lower-dimensional and post-HF support are not qualified")
    return policy


# A provenance limit, never a physical cutoff. Refuse before SCF rather than
# drop cells or produce an incomplete record. This bounded increment is not a
# streaming production trace facility.
MAX_SUPPORT_TERMS = 2_000_000


class DirectSupportRecorder:
    def __init__(self, policy):
        self.policy = policy

    def preflight(self, *, system, basis, lattice, nuclear):
        if getattr(lattice, "gamma_only_0", False):
            raise NotImplementedError("chi support recording requires full lattice sums, not gamma_only_0")
        inverse = np.linalg.inv(np.asarray(system.lattice))
        origins = np.asarray([shell.origin for shell in basis.shells()])
        span = max(np.linalg.norm(a-b) for a in origins for b in origins)
        def ball_bound(radius):
            return math.prod(2*math.ceil(float(radius*np.linalg.norm(row)))+3 for row in inverse)
        pair_count = len(origins)**2 * ball_bound(lattice.cutoff_bohr + span)
        source_count = len(system.unit_cell)*ball_bound(nuclear.nuclear_cutoff_bohr)
        if pair_count * source_count > MAX_SUPPORT_TERMS:
            raise MemoryError("chi exact support provenance exceeds its bounded term budget; "
                              "no physical domain was reduced")

    def __call__(self, **kwargs):
        return capture_direct_support(policy=self.policy, **kwargs)


@dataclass(frozen=True)
class ChiExecutedSupport:
    """Immutable canonical JSON; callers receive a copy from as_dict()."""
    payload: str

    def as_dict(self):
        return json.loads(self.payload)

    @property
    def identity(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()


def _vectors(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return []
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("executed support vectors must be finite (n,3)")
    return a.tolist()


def _cells(cells):
    result = []
    seen = set()
    for cell in cells:
        index = np.asarray(cell.index)
        if index.shape != (3,) or not np.isfinite(index).all() or not np.equal(index, np.rint(index)).all():
            raise ValueError("executed cell labels must be integral triples")
        key = tuple(int(x) for x in index)
        if key in seen:
            raise ValueError("duplicate executed cell label")
        seen.add(key)
        result.append(dict(index=key, cart_bohr=_vectors([cell.r_cart])[0]))
    return result


def capture_direct_support(*, policy, system, basis, domain, one_electron,
                           density, fock, j_cache, x_cache, nuclear_support, kpoints, weights, v_ne_cache,
                           spin_density=None, spin_fock=None):
    """Capture actual operator/enclosure arrays and native nuclear membership.

    Quartet predicates and screening settings accompany the cell enclosures:
    a cell in an enclosure is not a claim that every shell quartet was computed.
    """
    if domain.policy != policy:
        raise RuntimeError("chi requested and executed support policies differ")
    if (not domain.use_ewald_j_split or not domain.exchange_ewald_split
            or domain.exchange_exxdiv != "ewald" or domain.exact_zone_bohr is not None):
        raise NotImplementedError("chi support recording requires full corrected Ewald J/K")
    if j_cache is None or (x_cache is None and len(kpoints) != 1) or nuclear_support is None or v_ne_cache is None:
        raise RuntimeError("missing executed chi reciprocal or nuclear support")
    origins = np.asarray([shell.origin for shell in basis.shells()], dtype=float)
    snapshot = domain.as_dict()
    lat = dict(domain.lattice_1e.values)
    erfc = dict(domain.nuclear_erfc.values)
    pair_cutoff = float(lat["cutoff_bohr"])
    physical = policy == "physical_pairs"
    masks = []
    midpoints = []
    midpoint_ids = {}
    # This maps shell pairs to the cells actually returned by V_ne. The native
    # selector below reuses precisely the source-image predicate of that kernel.
    for g, cell in enumerate(one_electron["V_ne"].cells):
        translation = np.asarray(cell.r_cart)
        for p, a in enumerate(origins):
            for q, b in enumerate(origins):
                shifted_b = b + translation
                delta = a - shifted_b
                if physical and float(delta @ delta) > pair_cutoff**2:
                    continue
                midpoint = tuple(.5 * (a + shifted_b))
                if midpoint not in midpoint_ids:
                    midpoint_ids[midpoint] = len(midpoints)
                    midpoints.append(midpoint)
                masks.append((p, q, g, midpoint_ids[midpoint]))
    from ... import _vibeqc_core as core
    if physical:
        sources = core.physical_nuclear_source_indices(
            system, np.asarray(midpoints).reshape(-1, 3), erfc["nuclear_cutoff_bohr"])
        source_convention = "AO midpoint; native PairNuclearImageSelector"
    else:
        # Historical V_ne repeats the same origin-centered source-image ball
        # for every AO pair; keep that distinct from physical NN Ewald pairs.
        images = core.direct_lattice_cells(system, erfc["nuclear_cutoff_bohr"])
        sources = [[[atom, *map(int, cell.index)] for cell in images
                    for atom in range(len(system.unit_cell))]]
        source_convention = "origin-centered image ball, all AO pairs"
    channels = [{"q_fractional": [0., 0., 0.],
                 "vectors_bohr_inv": _vectors(j_cache.K_vectors)}]
    for key, channel in sorted({}.items() if x_cache is None else x_cache.q_channels.items()):
        channels.append(dict(q_fractional=np.frombuffer(key, dtype=float).tolist(),
                             vectors_bohr_inv=_vectors(channel.K_vectors)))
    from ...aux_basis import _basis_fingerprint
    record = dict(
        schema="vibeqc.chi.executed-support.v1", policy=policy,
        scope="experimental all-electron 3D direct RHF; no fitted/post-HF qualification",
        lattice_bohr=_vectors(system.lattice),
        nuclei=[dict(Z=int(a.Z), xyz_bohr=list(a.xyz)) for a in system.unit_cell],
        shell_origins_bohr=_vectors(origins), basis_sha256=_basis_fingerprint(basis),
        resolved_domain=snapshot, provenance_term_limit=MAX_SUPPORT_TERMS,
        exchange_transfer_fractional=[[([0.,0.,0.] if x_cache is None else np.frombuffer(x_cache._canonical_q(np.asarray(k)-np.asarray(kp))[0], dtype=float).tolist())
                                       for kp in kpoints] for k in kpoints],
        operator_cells={name: _cells(value.cells) for name, value in one_electron.items()},
        density_cells=_cells(density.cells), terminal_fock_cells=_cells(fock.cells),
        reciprocal_pair_cells_cart_bohr=_vectors(j_cache.cells_r_cart),
        coulomb_reciprocal_vectors_bohr_inv=_vectors(j_cache.K_vectors),
        exchange_channels=channels,
        kpoints_cart_bohr_inv=_vectors(kpoints), weights=list(map(float, weights)),
        nuclear_repulsion=dict(
            energy_ha=float(nuclear_support.energy), alpha=float(nuclear_support.alpha),
            real_cutoff_bohr=float(nuclear_support.real_cutoff_bohr),
            reciprocal_cutoff_bohr_inv=float(nuclear_support.reciprocal_cutoff_bohr_inv),
            real_pair_displacements_bohr=[_vectors(row) for row in nuclear_support.real_pair_displacements],
            reciprocal_vectors_bohr_inv=_vectors(nuclear_support.reciprocal_vectors),
            convention="actual Ewald traversal; ordered (a*n_atoms+b), centered representatives, self omitted"),
        nuclear_attraction=dict(convention=source_convention,
            reciprocal_vectors_bohr_inv=_vectors(v_ne_cache.K_vectors),
            reciprocal_pair_cells_cart_bohr=_vectors(v_ne_cache.cells_r_cart),
            membership="shared native selector replay on returned V_ne shell pairs before screening",
            shell_pair_cell_midpoint=masks, midpoints_bohr=_vectors(midpoints),
            source_atom_image_indices=sources, resolved_erfc_options=erfc),
        quartet_support=dict(
            convention=("AB/CD for J; AC/BD for K; both product lengths <= pair radius, "
                        "midpoint separation <= interaction radius") if physical else
                       "historical absolute image domains",
            membership="geometric support before numerical screening; cell arrays are enclosures",
            options=dict(domain.lattice_2e.values)),
        normalization="energies per primitive cell; full cyclic character weights",
    )
    if spin_density is not None or spin_fock is not None:
        if (spin_density is None or spin_fock is None or
                set(spin_density) != {"alpha", "beta"} or
                set(spin_fock) != {"alpha", "beta"}):
            raise ValueError("UHF support requires both spin density and Fock cells")
        record["scope"] = "experimental all-electron 3D direct UHF; no fitted/post-HF qualification"
        record["spin_density_cells"] = {k: _cells(v.cells) for k, v in spin_density.items()}
        record["spin_terminal_fock_cells"] = {k: _cells(v.cells) for k, v in spin_fock.items()}
    return ChiExecutedSupport(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
