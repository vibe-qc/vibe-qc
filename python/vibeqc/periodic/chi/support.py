"""Explicit finite-support contract for the experimental direct chi HF/KS routes.

This is an adapter to the shared periodic kernels, not a second construction of
Coulomb or exchange. A support record is provenance, never an accuracy verdict.
"""
from dataclasses import dataclass
import hashlib
import json

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
            "explicit chi support_policy requires all-electron 3D four_center HF/KS; "
            "fitted, ECP, lower-dimensional and post-HF support are not qualified")
    return policy


# A limit on stored nuclear membership entries: shell-pair mappings, unique
# midpoints and selected atom/image labels. Never multiply cell enclosures to
# estimate this deduplicated record, or use the limit as a physical cutoff.
MAX_SUPPORT_TERMS = 2_000_000


class DirectSupportRecorder:
    def __init__(self, policy, *, exact_exchange=True, full_range_exchange=None):
        self.policy = policy
        self.exact_exchange = bool(exact_exchange)
        self.full_range_exchange = (self.exact_exchange if full_range_exchange is None
                                    else bool(full_range_exchange))
        if self.full_range_exchange and not self.exact_exchange:
            raise ValueError("full-range support cannot be active without exchange")

    def preflight(self, *, system, basis, lattice, nuclear):
        if getattr(lattice, "gamma_only_0", False):
            raise NotImplementedError("chi support recording requires full lattice sums, not gamma_only_0")
        from ... import _vibeqc_core as core
        # Replay the same cell enclosure and pair predicate as V_ne before any
        # integral/SCF work. Capture below uses the actual returned V_ne cells.
        cells = (core.pair_complete_lattice_cells(basis, system, lattice.cutoff_bohr)
                 if self.policy == "physical_pairs" else
                 core.direct_lattice_cells(system, lattice.cutoff_bohr))
        _nuclear_membership(self.policy, system,
            np.asarray([shell.origin for shell in basis.shells()], dtype=float),
            cells, lattice.cutoff_bohr, nuclear.nuclear_cutoff_bohr)

    def __call__(self, **kwargs):
        return capture_direct_support(policy=self.policy, exact_exchange=self.exact_exchange,
                                      full_range_exchange=self.full_range_exchange, **kwargs)


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


def _require_capacity(terms):
    if terms > MAX_SUPPORT_TERMS:
        raise MemoryError("chi exact support provenance exceeds its bounded term budget; "
                          "no physical domain was reduced")


def _nuclear_membership(policy, system, origins, cells, pair_cutoff, source_cutoff):
    """Lossless v1 membership, in producer order, with bounded native panels."""
    from ... import _vibeqc_core as core
    physical = policy == "physical_pairs"
    masks, midpoints, midpoint_ids = [], [], {}
    for g, cell in enumerate(cells):
        translation = np.asarray(cell.r_cart)
        for p, a in enumerate(origins):
            for q, b in enumerate(origins):
                shifted_b = b + translation
                delta = a - shifted_b
                if physical and float(delta @ delta) > pair_cutoff**2:
                    continue
                midpoint = tuple(.5 * (a + shifted_b))
                if midpoint not in midpoint_ids:
                    _require_capacity(len(masks) + len(midpoints) + 2)
                    midpoint_ids[midpoint] = len(midpoints)
                    midpoints.append(midpoint)
                _require_capacity(len(masks) + len(midpoints) + 1)
                masks.append((p, q, g, midpoint_ids[midpoint]))
    terms = len(masks) + len(midpoints)
    sources = []
    if physical:
        # Every midpoint is traversed once. Native selection also bounds actual
        # labels before allocation; paneling preserves the old order and JSON.
        for start in range(0, len(midpoints), 64):
            try:
                rows = core.physical_nuclear_source_indices(
                    system, np.asarray(midpoints[start:start+64]).reshape(-1, 3), source_cutoff)
            except ValueError as exc:
                if "Nuclear support record exceeds bounded term budget" not in str(exc):
                    raise
                raise MemoryError("chi exact support provenance exceeds its bounded term budget; "
                                  "no physical domain was reduced") from exc
            if len(rows) != len(midpoints[start:start+64]):
                raise RuntimeError("incomplete native nuclear support panel")
            for labels in rows:
                terms += len(labels)
                _require_capacity(terms)
                sources.append(labels)
        convention = "AO midpoint; native PairNuclearImageSelector"
    else:
        # The historical origin-centered ball is stored only once, just as in
        # the original v1 record; AO-pair repetition is not materialized.
        images = core.direct_lattice_cells(system, source_cutoff)
        terms += len(images) * len(system.unit_cell)
        _require_capacity(terms)
        sources = [[[atom, *map(int, cell.index)] for cell in images
                    for atom in range(len(system.unit_cell))]]
        convention = "origin-centered image ball, all AO pairs"
    return masks, midpoints, sources, convention


def capture_direct_support(*, policy, system, basis, domain, one_electron,
                           density, fock, j_cache, x_cache, nuclear_support, kpoints, weights, v_ne_cache,
                           spin_density=None, spin_fock=None, exact_exchange=True,
                           full_range_exchange=None):
    """Capture actual operator/enclosure arrays and native nuclear membership.

    Quartet predicates and screening settings accompany the cell enclosures:
    a cell in an enclosure is not a claim that every shell quartet was computed.
    """
    if domain.policy != policy:
        raise RuntimeError("chi requested and executed support policies differ")
    if (not domain.use_ewald_j_split or not domain.exchange_ewald_split
            or domain.exchange_exxdiv != "ewald" or domain.exact_zone_bohr is not None):
        raise NotImplementedError("chi support recording requires full corrected Ewald J/K")
    full_range_exchange = exact_exchange if full_range_exchange is None else full_range_exchange
    if j_cache is None or (full_range_exchange and x_cache is None and len(kpoints) != 1) or nuclear_support is None or v_ne_cache is None:
        raise RuntimeError("missing executed chi reciprocal or nuclear support")
    origins = np.asarray([shell.origin for shell in basis.shells()], dtype=float)
    snapshot = domain.as_dict()
    lat = dict(domain.lattice_1e.values)
    erfc = dict(domain.nuclear_erfc.values)
    pair_cutoff = float(lat["cutoff_bohr"])
    physical = policy == "physical_pairs"
    masks, midpoints, sources, source_convention = _nuclear_membership(
        policy, system, origins, one_electron["V_ne"].cells,
        pair_cutoff, erfc["nuclear_cutoff_bohr"])
    channels = [{"q_fractional": [0., 0., 0.],
                 "vectors_bohr_inv": _vectors(j_cache.K_vectors)}]
    for key, channel in sorted({}.items() if x_cache is None else x_cache.q_channels.items()):
        channels.append(dict(q_fractional=np.frombuffer(key, dtype=float).tolist(),
                             vectors_bohr_inv=_vectors(channel.K_vectors)))
    from ...aux_basis import _basis_fingerprint
    record = dict(
        schema="vibeqc.chi.executed-support.v1", policy=policy,
        scope="experimental all-electron 3D direct RHF/RKS; no fitted/post-HF qualification",
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
    record["exact_exchange_applicability"] = "active" if exact_exchange else "inactive"
    record["full_range_exchange_applicability"] = "active" if full_range_exchange else "inactive"
    if not full_range_exchange:
        record["exchange_transfer_fractional"] = []
        record["exchange_channels"] = []
    if not exact_exchange:
        record["scope"] = "experimental all-electron 3D direct semilocal KS; no post-HF qualification"
        record["exchange_transfer_fractional"] = []
        record["exchange_channels"] = []
    if spin_density is not None or spin_fock is not None:
        if (spin_density is None or spin_fock is None or
                set(spin_density) != {"alpha", "beta"} or
                set(spin_fock) != {"alpha", "beta"}):
            raise ValueError("UHF support requires both spin density and Fock cells")
        if exact_exchange:
            record["scope"] = "experimental all-electron 3D direct UHF/UKS; no fitted/post-HF qualification"
        record["spin_density_cells"] = {k: _cells(v.cells) for k, v in spin_density.items()}
        record["spin_terminal_fock_cells"] = {k: _cells(v.cells) for k, v in spin_fock.items()}
    return ChiExecutedSupport(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
