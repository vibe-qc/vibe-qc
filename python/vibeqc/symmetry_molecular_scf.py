"""Explicit molecular method entry for a source-qualified fixed source.

Davidson (doi:10.1063/1.430484) and Dupuis--King
(doi:10.1002/qua.560110408) remain the quartet implementation's algorithms.
This layer owns the Hamiltonian, native method call and evaluated output
state; it adds no integral kernel, orbital optimizer or state projection.
"""
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from hashlib import sha256
import math
import json
import os
from threading import Lock
from time import perf_counter
from types import MappingProxyType

import numpy as np

from . import _vibeqc_core as core
from .solvers.operators import OperatorFrame, _snapshot
from .symmetry_shared import Budget, _array, _freeze, _integer, _tolerance
from .symmetry_molecular_hf import ScreenedMolecularHF, ScreenedMolecularJK, _calibration_cannot_pay
from .symmetry_quartets import qualify_manta_quartets, _shell_signature, QuartetWork
from .symmetry_quartets_ordered import qualified_molecular_source
from .symmetry_quartets_hierarchy import MolecularHierarchicalBounds, _native_budget


@dataclass(frozen=True)
class MolecularSymmetryOptions:
    """Explicit real molecular operations, fixed source error and resources.

    ``budget`` covers source qualification, bounds and J/K, plus a separate
    dense method/output reservation. It does not cap native SCF host memory.
    ``expected_builds`` is per calibrated HF signature, not a speed guarantee.
    """
    rotations: np.ndarray
    translations: np.ndarray
    source_tolerance: float
    maximum_omitted_norm: float
    budget: Budget
    screening: str = 'hierarchical'
    executor: str = 'native'
    expected_builds: int | None = None

    def __post_init__(self):
        if not isinstance(self.budget, Budget):
            raise TypeError('symmetry budget must be Budget')
        _array(self.rotations, 'float64', 3)
        _array(self.translations, 'float64', 2)
        ng = len(self.rotations)
        if ng < 1 or self.rotations.shape != (ng, 3, 3) or self.translations.shape != (ng, 3):
            raise ValueError('molecular symmetry operation shape mismatch')
        self.budget.admit(4096+2*(self.rotations.nbytes+self.translations.nbytes), 128*ng)
        if self.screening not in ('hierarchical', 'audit'):
            raise ValueError('screening must be hierarchical or audit')
        if self.executor not in ('native', 'numpy', 'ordered', 'auto'):
            raise ValueError('invalid molecular symmetry executor')
        if self.executor == 'auto':
            _integer(self.expected_builds, 'expected builds', positive=True)
        elif self.expected_builds is not None:
            raise ValueError('expected_builds requires executor="auto"')
        for name in ('source_tolerance', 'maximum_omitted_norm'):
            object.__setattr__(self, name, _tolerance(getattr(self, name), name))
        object.__setattr__(self, 'rotations', _freeze(self.rotations))
        object.__setattr__(self, 'translations', _freeze(self.translations))


@dataclass(frozen=True)
class MolecularRISymmetryOptions:
    """Explicit finite Coulomb RI source and complete Abelian characters.

    The auxiliary basis and native metric cutoffs define the Hamiltonian.
    Complex characters are unitary spatial actions, not time reversal.
    Work estimates select execution only; they never change source rank.
    """
    rotations: np.ndarray
    translations: np.ndarray
    characters: np.ndarray
    auxiliary_basis: str
    source_tolerance: float
    budget: Budget
    executor: str = 'blocked'
    expected_builds: int | None = None
    absolute_cutoff: float = 1e-12
    relative_cutoff: float = 1e-10
    negative_relative_tolerance: float = 1e-10
    mode_block: int = 8
    factor_export: str = 'producer'

    def __post_init__(self):
        if not isinstance(self.budget, Budget):
            raise TypeError('symmetry budget must be Budget')
        _array(self.rotations, 'float64', 3)
        _array(self.translations, 'float64', 2)
        _array(self.characters, 'complex128', 2)
        ng = len(self.rotations)
        if (ng < 1 or self.rotations.shape != (ng, 3, 3)
                or self.translations.shape != (ng, 3) or self.characters.shape != (ng, ng)):
            raise ValueError('molecular RI operation/character shape mismatch')
        self.budget.admit(4096+2*sum(a.nbytes for a in
            (self.rotations, self.translations, self.characters)), 128*ng*ng)
        if type(self.auxiliary_basis) is not str or not self.auxiliary_basis.strip():
            raise ValueError('an explicit auxiliary basis name is required')
        if self.executor not in ('blocked', 'dense', 'auto'):
            raise ValueError('invalid molecular RI executor')
        if self.executor == 'auto':
            _integer(self.expected_builds, 'expected builds', positive=True)
        elif self.expected_builds is not None:
            raise ValueError('expected_builds requires executor="auto"')
        for name in ('source_tolerance', 'absolute_cutoff', 'relative_cutoff', 'negative_relative_tolerance'):
            object.__setattr__(self, name, _tolerance(getattr(self, name), name))
        if self.source_tolerance > 1e-8:
            raise ValueError('source tolerance must not exceed 1e-8')
        _integer(self.mode_block, 'RI mode block', positive=True)
        if self.factor_export not in ('producer', 'ordered-tiles'):
            raise ValueError('factor_export must be producer or ordered-tiles')
        for name in ('rotations', 'translations', 'characters'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class MolecularSymmetryResult:
    """Owned evaluated D/F/energy and density-defining output orbitals.

    Other native convergence/trace diagnostics are forwarded. No native
    unwrapping hook is provided: unscreened post-SCF/gradient drivers cannot
    silently reuse this screened reference as their own Hamiltonian.
    """
    _native: object = field(repr=False)
    _values: object = field(repr=False)
    evaluation: object
    _provenance_json: str = field(repr=False)

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        if name.startswith('s_squared'):
            raise AttributeError('spin determinant diagnostic requires idempotent evaluated densities')
        return getattr(self._native, name)

    @property
    def fock_build(self):
        return self.evaluation.build

    @property
    def symmetry_provenance(self):
        return json.loads(self._provenance_json)

    @property
    def citation_numerics(self):
        return (('symmetry_molecular_ri',) if self.symmetry_provenance['source_kind'] == 'abelian-ri'
            else ('shared_symmetry',))

    @property
    def run_fields(self):
        return dict(self.fock_build.output_fields(),
            orbital_state_relation='density-defining-natural-orbitals-physical-rayleigh-quotients',
            scf_stop_reason='converged' if self.converged else ('noiter' if self.n_iter == 0 else 'iteration_limit'),
            terminal_commutator_norm=self.fock_build.commutator_norm,
            symmetry_source=self.fock_build.source.as_dict(),
            symmetry_execution=dict(self.symmetry_provenance),
            orbital_kind='natural')


def _density_orbitals(channel):
    """Metric natural orbitals preserve fractional stopped/READ densities.

    The frame is an output coordinate system; no symmetry of its retained
    space, partner gauge or occupied subspace is inferred from this solve.
    """
    x, s = channel.operator.frame.X, channel.operator.frame.overlap
    density = channel.density
    reduced = x.T @ s @ density @ s @ x
    occupations, vectors = np.linalg.eigh(reduced)
    order = np.argsort(-occupations, kind='stable')
    occupations, c = occupations[order], x @ vectors[:, order]
    rebuilt = (c*occupations) @ c.T
    if np.linalg.norm(rebuilt-density) > 1e-9*max(1., np.linalg.norm(density)):
        raise ValueError('evaluated density leaves the declared retained metric space')
    energies = np.diag(c.T @ channel.operator.matrix @ c)
    return _snapshot(c), _snapshot(energies), _snapshot(occupations)


def _validate_options(method, options):
    if (options.max_iter < 0 or not math.isfinite(options.conv_tol_energy)
            or options.conv_tol_energy <= 0 or not math.isfinite(options.conv_tol_grad)
            or options.conv_tol_grad <= 0):
        raise ValueError('nonnegative iterations and finite positive SCF tolerances required')
    if options.initial_guess not in (core.InitialGuess.HCORE, core.InitialGuess.READ):
        raise ValueError('qualified molecular symmetry currently requires HCORE or supplied-density READ')
    if options.read_path:
        raise ValueError('supply a same-basis READ density, not a restart file')
    for name in ('ecp_centers', 'ecp_library', 'ecp_primitive_blocks', 'ecp_primitive_centers',
                 'ecp_effective_charges', 'ecp_total_ncore', 'dft_plus_u_sites', 'dft_plus_u_ao_groups',
                 'density_fit', 'aux_basis', 'cosx'):
        if getattr(options, name):
            raise ValueError(f'qualified all-electron molecular source does not admit {name}')
    if options.scf_mode == core.SCFMode.CONVENTIONAL:
        raise ValueError('qualified symmetry uses direct retained tiles, not conventional AO4 storage')
    if (not math.isfinite(options.linear_dep_threshold) or options.linear_dep_threshold < 0):
        raise ValueError('finite nonnegative linear dependence threshold required')
    if method == 'uhf' and (options.multi_guess_seeds or options.atomic_spins
                           or options.spinlock_mode != core.SpinlockMode.OFF):
        raise ValueError('qualified molecular symmetry does not yet admit multi-guess or spin schedules')


def _qualify_one_electron(plan, overlap, hcore):
    # W=U^-1 has source rows and image columns. Covariant AO tensors
    # transform as W^T A W, including nonorthogonal Cartesian shells.
    residuals = [0., 0.]
    n = len(overlap)
    for g, row in enumerate(plan.inverse_blocks):
        w = np.zeros((n, n))
        for shell, block in enumerate(row):
            target = int(plan.orbits.destinations[g, shell])
            a, b = plan.offsets[shell:shell+2]
            c, d = plan.offsets[target:target+2]
            w[a:b, c:d] = block
        for index, matrix in enumerate((overlap, hcore)):
            residuals[index] = max(residuals[index], float(np.linalg.norm(w.T@matrix@w-matrix)))
    if not all(math.isfinite(r) and r <= plan.source_tolerance for r in residuals):
        raise ValueError('one-electron source fails AO metric/operator equivariance')
    return dict(overlap_residual=residuals[0], hcore_residual=residuals[1],
        scalar_products=4*plan.group.order*n**3, one_electron_build_calls=3)


def _qualify_ri_one_electron(source, overlap, hcore):
    # U C = C chi, D -> U D U^dagger, A -> U^-dagger A U^-1.
    # Test U^dagger A U = A on the actual S/H; do not project either matrix.
    residuals = [0., 0.]
    for characters in source.irreps.characters.T:
        u = (source.orbital_coefficients*characters[source.orbital_labels])@source.orbital_inverse
        for index, matrix in enumerate((overlap, hcore)):
            residuals[index] = max(residuals[index], float(np.linalg.norm(u.conj().T@matrix@u-matrix)))
    if not all(math.isfinite(r) and r <= source.source_tolerance for r in residuals):
        raise ValueError('one-electron source fails AO metric/operator equivariance')
    return dict(overlap_residual=residuals[0], hcore_residual=residuals[1],
        scalar_products=5*len(source.irreps.characters)*source.n**3, one_electron_build_calls=3)


def _signature_requirements(molecule, basis, budget):
    atoms = core._symmetry_molecule_count(molecule, budget.native())
    shell_bytes = core._symmetry_basis_snapshot_bytes(basis, budget.native())
    size = 4096+2048*atoms+2*shell_bytes
    budget.admit(size, 1)
    return size


def _geometry_signature(molecule, basis):
    atoms = molecule.atoms
    if not atoms or not all(np.isfinite(a.xyz).all() for a in atoms):
        raise ValueError('a finite nonempty molecular geometry is required')
    if any(a.Z and b.Z and a.xyz == b.xyz for i, a in enumerate(atoms) for b in atoms[:i]):
        raise ValueError('coincident charged nuclei are not admitted')
    for shell in basis.shells():
        if not 0 <= shell.atom_index < len(atoms) or not np.array_equal(shell.origin, atoms[shell.atom_index].xyz):
            raise ValueError('basis centers and molecular geometry disagree')
    # Charge and multiplicity specify the state, not this all-electron
    # Hamiltonian. Preserve nuclei/order and every basis normalization bit.
    return tuple((a.Z, tuple(a.xyz)) for a in atoms), _shell_signature(basis.shells())


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PreparedMolecularSymmetry:
    """Process-local qualified quartet or RI source for independent HF states.

    Construction qualifies once (ordered-only sources need no tile audit).
    The handle owns that exact native producer, fixed mask/factors and one-electron
    matrices, not an SCF state. It is
    neither a disk cache nor a producer-issued algebraic equivariance seal.
    One method at a time may use its native workspace. New geometry, basis,
    operations, kernel or screening policy requires a new handle.
    """
    options: MolecularSymmetryOptions | MolecularRISymmetryOptions
    identity: str
    _jk: object = field(repr=False)
    _signature: tuple = field(repr=False)
    _overlap: np.ndarray = field(repr=False)
    _hcore: np.ndarray = field(repr=False)
    _nuclear: float
    _method_bytes: int
    _receipt_json: str = field(repr=False)
    _lock: object = field(repr=False)
    _pid: int

    def __init__(self, molecule, basis, *, symmetry_reduction):
        from . import _attach_inline_ecp_from_basis_sidecar, _coerce_molecular_basis
        if type(symmetry_reduction) not in (MolecularSymmetryOptions, MolecularRISymmetryOptions):
            raise TypeError('source preparation requires molecular symmetry options')
        config = symmetry_reduction
        basis = _coerce_molecular_basis(molecule, basis)
        # Named basis sidecars must not introduce an unqualified ECP source.
        check = core.RHFOptions(); check.initial_guess = core.InitialGuess.HCORE
        _attach_inline_ecp_from_basis_sidecar(check, molecule, basis)
        _validate_options('rhf', check)
        n = basis.nbasis
        method_bytes = 65536+1024*n*n
        config.budget.admit(method_bytes, (128+5*len(config.rotations))*n**3)
        source_budget = Budget(config.budget.maximum_bytes-method_bytes,
            config.budget.maximum_work)
        started = perf_counter()
        # Admit both the persistent signature and a comparison snapshot
        # before Python atom/radial payload copies.
        signature_bytes = _signature_requirements(molecule, basis, config.budget)
        config.budget.admit(method_bytes+signature_bytes, 1)
        source_budget = Budget(source_budget.maximum_bytes-signature_bytes,
            source_budget.maximum_work)
        signature = _geometry_signature(molecule, basis)
        is_ri = type(config) is MolecularRISymmetryOptions
        if is_ri:
            from .symmetry_ri import prepare_abelian_molecular_ri, MolecularRIJK
            auxiliary = _coerce_molecular_basis(molecule, config.auxiliary_basis)
            _attach_inline_ecp_from_basis_sidecar(check, molecule, auxiliary)
            _validate_options('rhf', check)
            native_options = core._manta.RIOptions()
            for name in ('absolute_cutoff', 'relative_cutoff', 'negative_relative_tolerance', 'mode_block'):
                setattr(native_options, name, getattr(config, name))
            source = prepare_abelian_molecular_ri(molecule, basis, auxiliary,
                config.rotations, config.translations, config.characters,
                source_tolerance=config.source_tolerance, budget=source_budget,
                ri_options=native_options, factor_export=config.factor_export)
            jk = MolecularRIJK(source, budget=source_budget, executor=config.executor)
            source_identity, source_bytes = source.source_identity, source.required_bytes
            source_fields = dict(source_kind='abelian-ri', fitting=source.receipt)
        else:
            provider = core._manta.ReferenceJKBuilder(basis, core._manta.KernelRequest(), _native_budget(source_budget))
            bounds = (MolecularHierarchicalBounds(basis, provider,
                maximum_omitted_norm=config.maximum_omitted_norm, budget=source_budget)
                if config.screening == 'hierarchical' else None)
            ordered_only = bounds is not None and (config.executor == 'ordered' or
                (config.executor == 'auto' and _calibration_cannot_pay(config.expected_builds)))
            plan = qualify_manta_quartets(molecule, basis, config.rotations, config.translations,
                provider=provider, source_tolerance=config.source_tolerance, budget=source_budget,
                schedule='streaming', maximum_omitted_norm=config.maximum_omitted_norm, bounds=bounds,
                qualification='ordered-source' if ordered_only else 'equivariance')
            jk = ScreenedMolecularJK(plan.screening, budget=source_budget, executor=config.executor)
            source_identity, source_bytes = plan.screening.source_identity, plan.screening.required_bytes
            source_fields = dict(source_kind='quartets',
                source_audit_work=asdict(plan.qualification_work),
                qualification=asdict(qualified_molecular_source(plan.screening)))
        setup_seconds = perf_counter()-started
        started = perf_counter()
        s = np.asarray(core.compute_overlap(basis))
        h = np.asarray(core.compute_kinetic(basis)+core.compute_nuclear(basis, molecule))
        one_electron = (_qualify_ri_one_electron(source, s, h) if is_ri
            else _qualify_one_electron(plan, s, h))
        one_electron['seconds'] = perf_counter()-started
        nuclear = molecule.nuclear_repulsion()
        if not math.isfinite(nuclear):
            raise ValueError('nonfinite molecular nuclear energy')
        identity = sha256(b'vibeqc-prepared-molecular-symmetry-v1\0')
        identity.update(source_identity.encode())
        identity.update(s.tobytes()); identity.update(h.tobytes())
        identity.update(float(nuclear).hex().encode())
        receipt = dict(identity=identity.hexdigest(), source_setup_seconds=setup_seconds,
            one_electron_qualification=one_electron,
            source_identity=source_identity, **source_fields,
            method_logical_reservation=method_bytes,
            signature_logical_reservation=signature_bytes,
            persistent_matrix_bytes=s.nbytes+h.nbytes,
            source_logical_bytes=source_bytes)
        for key, value in dict(options=config, identity=identity.hexdigest(), _jk=jk,
                _signature=signature, _overlap=_snapshot(s), _hcore=_snapshot(h),
                _nuclear=nuclear, _method_bytes=method_bytes,
                _receipt_json=json.dumps(receipt, allow_nan=False), _lock=Lock(),
                _pid=os.getpid()).items():
            object.__setattr__(self, key, value)

    @property
    def preparation_receipt(self):
        """Cost paid once by construction; a fresh JSON-compatible copy."""
        return json.loads(self._receipt_json)

    def __reduce_ex__(self, protocol):
        raise TypeError('prepared molecular source is process-local; requalify in another process')

    @contextmanager
    def _lease(self, molecule, basis):
        if self._pid != os.getpid():
            raise ValueError('prepared molecular source belongs to another process')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('prepared molecular source is already in use')
        try:
            reservation = _signature_requirements(molecule, basis, self.options.budget)
            if reservation > self.preparation_receipt['signature_logical_reservation']:
                raise ValueError('prepared source geometry or basis inventory changed; prepare a new source')
            if _geometry_signature(molecule, basis) != self._signature:
                raise ValueError('prepared source geometry or basis changed; prepare a new source')
            if type(self.options) is MolecularSymmetryOptions:
                qualified_molecular_source(self._jk.screening)
            yield
        finally:
            self._lock.release()

    def _method_builder(self, method, *, single_evaluation=False):
        # Admit the method-local capture before any new cost probes. Only
        # scalar route receipts are published back to the prepared source.
        child = self._jk.fork(capture_terminal=True)
        measured, reused = [], []
        if self.options.executor == 'auto':
            existing = dict(self._jk.route_calibrations)
            seen = set()
            def on_request(builder, densities, targets):
                if targets in seen:
                    return
                if targets in existing:
                    reused.append(targets)
                else:
                    # Resource admission in build_channels precedes this
                    # hook. Invalid trial inputs must not start cost probes.
                    if not isinstance(densities, tuple) or len(densities) != len(targets):
                        raise ValueError('density/channel dimensions disagree before route calibration')
                    for density in densities:
                        builder._density(density)
                    route = builder.calibrate_routes(targets=targets,
                        expected_builds=1 if single_evaluation else self.options.expected_builds)
                    # The NOITER decision belongs to this invocation, not
                    # a later iterative use of the same prepared source.
                    if not single_evaluation:
                        self._jk._routes[targets] = route
                    measured.append(route)
                seen.add(targets)
            child._route_request = on_request
        return child, measured, reused


def prepare_molecular_symmetry(molecule, basis, *, symmetry_reduction):
    """Prepare once, then pass the handle as ``run_rhf/uhf`` symmetry_reduction.

    Preparations contain no spin state, density, occupied/virtual projector
    or method convergence data. Read ``preparation_receipt`` for setup cost.
    Each method result reports only work performed by that method call.
    """
    return PreparedMolecularSymmetry(molecule, basis, symmetry_reduction=symmetry_reduction)


def run_symmetry_scf(method, molecule, basis, options, *, symmetry_reduction,
                     integral_engine=None, coulomb=None, exchange=None,
                     read_from=None, fragments=None, dft_plus_u=None):
    from . import _attach_inline_ecp_from_basis_sidecar, _coerce_molecular_basis
    if type(symmetry_reduction) not in (MolecularSymmetryOptions, MolecularRISymmetryOptions, PreparedMolecularSymmetry):
        raise TypeError('symmetry_reduction requires MolecularSymmetryOptions, MolecularRISymmetryOptions or PreparedMolecularSymmetry')
    if any(value is not None for value in (integral_engine, coulomb, exchange, read_from, fragments)) or dft_plus_u:
        raise ValueError('qualified symmetry cannot be combined with another engine, projected READ, FRAGMO or DFT+U')
    if method not in ('rhf', 'uhf'):
        raise ValueError('qualified molecular symmetry currently admits RHF and UHF')
    cls = core.RHFOptions if method == 'rhf' else core.UHFOptions
    if options is None:
        opts = cls()
        opts.initial_guess = core.InitialGuess.HCORE
    elif type(options) is cls:
        opts = cls(options)
    else:
        raise TypeError(f'{method} requires {cls.__name__}')
    _validate_options(method, opts)
    basis = _coerce_molecular_basis(molecule, basis)
    _attach_inline_ecp_from_basis_sidecar(opts, molecule, basis)
    _validate_options(method, opts)
    ne, mult = molecule.n_electrons(), molecule.multiplicity
    if ne < 0 or mult < 1 or (ne-mult+1) % 2 or ne < mult-1:
        raise ValueError('electron count and multiplicity disagree')
    if method == 'rhf' and (mult != 1 or ne % 2):
        raise ValueError('RHF requires a closed-shell singlet')
    reused_source = type(symmetry_reduction) is PreparedMolecularSymmetry
    prepared = (symmetry_reduction if reused_source else
        prepare_molecular_symmetry(molecule, basis, symmetry_reduction=symmetry_reduction))
    with prepared._lease(molecule, basis):
        return _run_prepared(method, molecule, basis, opts, prepared, reused_source)


def _run_prepared(method, molecule, basis, opts, prepared, reused_source):
    from .output import record
    ne, mult = molecule.n_electrons(), molecule.multiplicity
    na, nb = (ne+mult-1)//2, (ne-mult+1)//2
    config = prepared.options
    # Native NOITER returns after exactly one initial-density J/K evaluation
    # (rhf.cpp/uhf.cpp). The separate orbital optimizer has its own schedule;
    # its iteration limit is not used as a build-count certificate here.
    single_evaluation = opts.max_iter == 0 and opts.orbital_optimizer != 'opentrustregion'
    jk, routes, reused_routes = prepared._method_builder(method, single_evaluation=single_evaluation)
    is_ri = type(config) is MolecularRISymmetryOptions
    s, h, nuclear = prepared._overlap, prepared._hcore, prepared._nuclear
    n = basis.nbasis
    preparation = prepared.preparation_receipt
    one_electron = dict(preparation['one_electron_qualification'], reused=reused_source)
    if reused_source:
        one_electron.update(scalar_products=0, one_electron_build_calls=0, seconds=0.)
    setup_seconds = 0. if reused_source else preparation['source_setup_seconds']
    if is_ri:
        # No quartet audit is claimed. The immutable preparation receipt
        # retains setup counts; this method records only fresh fitting work.
        fitting = preparation['fitting']
        source_fields = dict(source_kind='abelian-ri', fitting_source=fitting,
            fitting_setup_work={k: 0 if reused_source else fitting[k] for k in
                ('overlap_build_calls', 'metric_shell_calls', 'coupling_shell_calls',
                 'metric_primitive_products', 'coupling_primitive_products')})
        source_label, citations = 'Abelian RI', list(jk.source.citation_numerics)
    else:
        plan = jk.screening._plan
        source_fields = dict(source_kind='quartets', screening=plan.screening.receipt,
            source_audit_work=asdict(QuartetWork() if reused_source else plan.qualification_work))
        source_label, citations = plan.screening.policy, ['shared_symmetry']
    # Same unscaled AO overlap cutoff convention as the native HF host.
    eigenvalues, vectors = np.linalg.eigh(s)
    keep = eigenvalues >= opts.linear_dep_threshold
    if not np.any(keep) or np.any(eigenvalues[keep] <= 0):
        raise ValueError('no positive retained overlap space')
    x = vectors[:, keep]/np.sqrt(eigenvalues[keep])
    frame = OperatorFrame(basis_id=preparation['source_identity'] if is_ri else plan.source_identity,
        n=n, ordering='native molecular AO',
        image_gauge='molecular', overlap=s, X=x)
    hf = ScreenedMolecularHF(jk, frame=frame, hcore=h, nuclear_energy=nuclear)
    scf_started = perf_counter()
    if method == 'rhf':
        native = core.run_rhf_scf_with_jk(basis, ne, s, h, nuclear, jk, opts, molecule=molecule)
        densities = (np.asarray(native.density),)
    else:
        native = core.run_uhf_scf_with_jk(basis, na, nb, s, h, nuclear, jk, opts, molecule=molecule)
        densities = (np.asarray(native.density_alpha), np.asarray(native.density_beta))
    scf_elapsed_seconds = perf_counter()-scf_started
    scf_calibration_seconds = sum(r.calibration_seconds for r in routes)
    scf_seconds = scf_elapsed_seconds-scf_calibration_seconds
    scf_totals = jk.totals
    evaluation = hf.evaluate(densities, method=method.upper(), generation=native.n_iter,
        reuse_evaluated=True)
    build = evaluation.build
    values = dict(energy=build.energy.total, e_electronic=build.energy.electronic,
        e_nuclear=nuclear, overlap=frame.overlap, orbital_kind='natural',
        ecp_provenance_verified=True, ecp_operator_applied=False, ecp_total_ncore=0)
    for suffix, channel in zip(('',) if method == 'rhf' else ('_alpha', '_beta'), build.channels):
        c, energies, occupations = _density_orbitals(channel)
        values.update({f'mo_coeffs{suffix}': c, f'mo_energies{suffix}': energies,
            f'occupations{suffix}': occupations, f'density{suffix}': channel.density,
            f'fock{suffix}': channel.operator.matrix})
    if method == 'uhf':
        values['density'] = _snapshot(sum(densities))
        if all(np.linalg.norm(d@s@d-d) <= 1e-8*max(1., np.linalg.norm(d)) for d in densities):
            ideal = .25*(na-nb)*(na-nb+2)
            spin = ideal+nb-float(np.trace(densities[0]@s@densities[1]@s))
            values.update(s_squared=spin, s_squared_ideal=ideal, s_squared_deviation=spin-ideal)
    provenance = dict(**source_fields, source_setup_seconds=setup_seconds,
        one_electron_qualification=one_electron,
        prepared_source=dict(preparation, reused=reused_source),
        reused_routes=reused_routes, routes=[asdict(r) for r in routes], scf_work=asdict(scf_totals),
        route_calibration='on-demand', single_evaluation_bound=single_evaluation,
        scf_seconds=scf_seconds, scf_elapsed_seconds=scf_elapsed_seconds,
        scf_calibration_seconds=scf_calibration_seconds,
        terminal_calibration_seconds=sum(r.calibration_seconds for r in routes)-scf_calibration_seconds,
        terminal_work=asdict(evaluation.receipt), output_state_bytes=build.storage_nbytes,
        terminal_capture=jk.terminal_capture,
        method_logical_reservation=prepared._method_bytes, native_energy_delta=abs(native.energy-build.energy.total),
        electronic_error_bound=evaluation.electronic_error_bound,
        fock_error_bounds=evaluation.fock_error_bounds)
    result = MolecularSymmetryResult(native, MappingProxyType(values), evaluation,
        json.dumps(provenance, allow_nan=False))
    record('molecular_symmetry_source',
        f'  Molecular symmetry: {config.executor}; fixed {source_label} source.\n'
        f'  SCF J/K shell calls: {scf_totals.work.shell_calls}; '
        f'contraction products: {scf_totals.scalar_products}; terminal FockBuild: {build.identity}.\n',
        symmetry_source=build.source.as_dict(), citation_numerics=citations)
    return result
