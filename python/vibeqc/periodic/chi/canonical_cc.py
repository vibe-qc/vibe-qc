"""Experimental same-source RHF/UHF CCSD and canonical spin-orbital (T).

The HF canonical frame, retained metric and finite character group are borrowed,
never rebuilt. The supercell spin-orbital Hamiltonian has g/Nk integrals;
its correlation energy is divided by Nk. See McClain et al. (2017), section 3.
The source construction is exclusively chi; the algebra has no CCM dependency.
"""
from dataclasses import dataclass
import itertools
import json
import numpy as np
from .canonical_mp2 import _positive_integer
from ._cc_equations import residual, _ccsd_energy_so, triples


@dataclass(frozen=True)
class ChiCCOptions:
    compute_triples: bool = True
    max_iter: int = 100
    energy_tolerance: float = 1e-10
    residual_tolerance: float = 1e-8
    denominator_tolerance: float = 1e-12
    memory_byte_cap: int = 512 * 1024**2

    def validate(self):
        if not isinstance(self.compute_triples,bool):
            raise TypeError('compute_triples must be boolean')
        _positive_integer(self.max_iter, 'max_iter')
        _positive_integer(self.memory_byte_cap, 'memory_byte_cap')
        for name in ('energy_tolerance','residual_tolerance','denominator_tolerance'):
            value = getattr(self,name)
            if isinstance(value,bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(name+' must be positive and finite')


class FactorIntegrals:
    """On-demand antisymmetric spin blocks; all-virtual blocks are never cached.

    Occupied/virtual labels retain (character, canonical MO, spin). Factors
    with matching q are contracted with the R5.1 reversed second pair, so
    independent q/-q auxiliary gauges are never identified. No rank truncation.
    """
    def __init__(self, adapter, coefficients, occupied, frozen, batch):
        self.adapter, self.batch = adapter, batch
        # A single frame is the legacy RHF interface. UHF supplies two actual
        # frames, with independent MO and occupied dimensions; no padding or
        # spatial-orbital identification is performed.
        if isinstance(coefficients, np.ndarray) and coefficients.ndim == 3:
            coefficients = (coefficients, coefficients)
            occupied, frozen = (occupied, occupied), (frozen, frozen)
        self.c = tuple(np.asarray(c) for c in coefficients)
        if len(self.c) != 2 or len(occupied) != 2 or len(frozen) != 2:
            raise ValueError('CC factor transport requires two spin channels')
        nk = adapter.nk
        nm = tuple(c.shape[-1] for c in self.c)
        ranges = [(range(frozen[s], occupied[s]), range(occupied[s], nm[s]))
                  for s in (0, 1)]
        # Retain the historical interleaved order in the same-root RHF limit.
        self.labels = [(k,p,s) for space in (0,1) for k in range(nk)
                       for p in range(max(nm)) for s in (0,1)
                       if p in ranges[s][space]]
        self.no = nk*sum(occupied[s]-frozen[s] for s in (0,1))
        self.shape = (len(self.labels),)*4
        self.cache = {}
        self.mo_factors = {}
        # MO pair factors retain every auxiliary row, including null rows.
        # Account for their cache and transformation temporaries before use.
        self.factor_workspace_bytes = 64*sum(
            x.shape[0]*sum(m*m for m in nm) for x in adapter.operator.values())

    def pair_factor(self, left, spin):
        key = (left, spin)
        if key not in self.mo_factors:
            k,l = left
            c = self.c[spin]
            ao = self.adapter.operator[left]
            transformed = np.empty((len(ao),c.shape[-1],c.shape[-1]),complex)
            for start in range(0,len(ao),self.batch):
                transformed[start:start+self.batch] = np.einsum(
                    'pi,Ppq,qj->Pij',c[k].conj(),ao[start:start+self.batch],c[l],optimize=True)
            self.mo_factors[key] = transformed
        return self.mo_factors[key]

    def coulomb_panel(self, kp, mp, kr, rm, ks, sm, kq, mq, sp, sq):
        if self.adapter.pair_groups[kp,kr] != self.adapter.pair_groups[ks,kq]:
            raise ValueError('CC integral factors must share one momentum metric space')
        left = self.pair_factor((kp,kr),sp)[:,mp,:][:,rm]
        right = self.pair_factor((ks,kq),sq)[:,:,mq][:,sm]
        panel = np.zeros((len(rm),len(sm)),complex)
        for start in range(0,len(left),self.batch):
            panel += left[start:start+self.batch].T @ right[start:start+self.batch].conj()
        return panel

    def __getitem__(self, slices):
        axes = tuple(tuple(range(*s.indices(self.shape[0]))) for s in slices)
        cacheable = not all(all(i>=self.no for i in axis) for axis in axes)
        if cacheable and axes in self.cache:
            return self.cache[axes]
        out = np.zeros(tuple(map(len,axes)),complex)
        labels = self.labels
        # Work in one bra pair and one momentum-resolved virtual/occupied panel.
        for ip,p in enumerate(axes[0]):
            kp,mp,sp = labels[p]
            for iq,q in enumerate(axes[1]):
                kq,mq,sq = labels[q]
                for kr in range(self.adapter.nk):
                    ks = self.adapter.conserved[kp,kq,kr]
                    for exchange in (False,True):
                        rr = [(j,labels[r][1]) for j,r in enumerate(axes[2])
                              if labels[r][0]==kr and labels[r][2]==(sq if exchange else sp)]
                        ss = [(j,labels[s][1]) for j,s in enumerate(axes[3])
                              if labels[s][0]==ks and labels[s][2]==(sp if exchange else sq)]
                        if not rr or not ss:
                            continue
                        ri,rm = zip(*rr); si,sm = zip(*ss)
                        if exchange:
                            panel = self.coulomb_panel(kp,mp,ks,sm,kr,rm,kq,mq,sp,sq).T
                        else:
                            panel = self.coulomb_panel(kp,mp,kr,rm,ks,sm,kq,mq,sp,sq)
                        out[ip,iq][np.ix_(ri,si)] += (-1 if exchange else 1)*panel/self.adapter.nk
        if cacheable:
            self.cache[axes] = out
        return out

    def allowed_excitation(self, occupied, virtual):
        """Conserve spin population and momentum for a packed excitation."""
        holes = [self.labels[i] for i in occupied]
        particles = [self.labels[a] for a in virtual]
        if sum(x[2] for x in holes) != sum(x[2] for x in particles):
            return False
        k = self.adapter.k_fractional
        delta = sum((k[x[0]] for x in holes), np.zeros(3)) - sum(
            (k[x[0]] for x in particles), np.zeros(3))
        return bool(np.max(abs(delta-np.rint(delta))) < 1e-10)

    def excitation_masks(self):
        """Denominators are needed only for allowed, distinct excitations."""
        occ = np.asarray(self.labels[:self.no])
        vir = np.asarray(self.labels[self.no:])
        one = (vir[:,None,0] == occ[None,:,0]) & (vir[:,None,2] == occ[None,:,2])
        two = ((vir[:,None,None,None,2]+vir[None,:,None,None,2])
               == (occ[None,None,:,None,2]+occ[None,None,None,:,2]))
        conserved = self.adapter.conserved[
            occ[None,None,:,None,0], occ[None,None,None,:,0], vir[:,None,None,None,0]]
        two &= conserved == vir[None,:,None,None,0]
        two &= ~np.eye(len(vir), dtype=bool)[:,:,None,None]
        two &= ~np.eye(len(occ), dtype=bool)[None,None,:,:]
        return one, two


def workspace_estimate(no,nv):
    """Conservative complex128 array workspace (excludes borrowed AO factors).

    Include all cached non-vvvv blocks, row slices, intermediates, amplitudes,
    eight DIIS vectors/errors and NumPy contraction temporaries. Refuse before
    allocating any MO integrals or amplitudes. BLAS/library overhead is external.
    """
    return 16*(4*((no+nv)**4-nv**4)+80*(no*nv+no*no*nv*nv)+8*nv**3)


def _solver_checkpoint(t1, t2, trace, previous, vectors, errors, converged):
    """A safe numerical state at a complete iteration boundary.

    An unfinished boundary contains the *next* amplitudes; its trace records
    all evaluated iterations. A converged boundary contains the accepted ones.
    The first unavailable energy change has the finite sentinel -1 on disk.
    """
    numeric_trace = np.array(trace, dtype=float).reshape(-1, 4)
    if len(numeric_trace):
        numeric_trace[0, 3] = -1.
    size = t1.size + t2.size
    return dict(iteration=len(trace), converged=bool(converged),
        t1=np.array(t1, copy=True), t2=np.array(t2, copy=True),
        trace=numeric_trace, previous=np.array([previous], dtype=complex),
        diis_vectors=np.array(vectors, dtype=complex).reshape(-1, size),
        diis_errors=np.array(errors, dtype=complex).reshape(-1, size))


def _restore_solver_state(state, no, nv):
    """Validate solver dimensions and history before consuming amplitudes."""
    if set(state) != {'iteration', 'converged', 't1', 't2', 'trace', 'previous',
                      'diis_vectors', 'diis_errors'}:
        raise ValueError('CC checkpoint solver fields are incompatible')
    iteration = state['iteration']
    if (isinstance(iteration, bool) or not isinstance(iteration, int)
            or iteration < 1 or not isinstance(state['converged'], bool)):
        raise ValueError('CC checkpoint iteration state is invalid')
    size = no*nv + no*no*nv*nv
    expected = {'t1': (nv, no), 't2': (nv, nv, no, no),
                'trace': (iteration, 4), 'previous': (1,)}
    for key, shape in expected.items():
        if np.shape(state[key]) != shape or not np.isfinite(state[key]).all():
            raise ValueError('CC checkpoint '+key+' dimensions or values are invalid')
    trace = np.asarray(state['trace'])
    if (np.iscomplexobj(trace) or not np.array_equal(trace[:, 0], np.arange(1, iteration+1))
            or trace[0, 3] != -1. or np.any(trace[:, 2] < 0.)
            or np.any(trace[1:, 3] < 0.)):
        raise ValueError('CC checkpoint convergence trace is invalid')
    vectors, errors = np.asarray(state['diis_vectors']), np.asarray(state['diis_errors'])
    expected_diis = min(iteration-int(state['converged']), 8)
    if (vectors.shape != (expected_diis, size) or errors.shape != vectors.shape
            or not np.isfinite(vectors).all() or not np.isfinite(errors).all()):
        raise ValueError('CC checkpoint DIIS history is incompatible')
    previous = complex(state['previous'][0])
    # Converged boundaries retain the preceding energy for the final delta.
    if not state['converged'] and abs(previous.real-trace[-1, 1]) > 1e-14:
        raise ValueError('CC checkpoint preceding energy differs from its history')
    restored_trace = [tuple(row) for row in trace]
    restored_trace[0] = (*restored_trace[0][:3], np.inf)
    restored_trace = [(int(i), float(e), float(r), float(d)) for i,e,r,d in restored_trace]
    return (np.array(state['t1'], dtype=complex, copy=True),
            np.array(state['t2'], dtype=complex, copy=True), previous,
            [np.array(x, copy=True) for x in vectors],
            [np.array(x, copy=True) for x in errors], restored_trace)


def solve(integrals, eps, options, *, restart_state=None, progress_callback=None,
          checkpoint_interval=1):
    """Solve or continue numerical CC iterations with unchanged equations.

    ``restart_state``/callbacks are internal numerical transport: the caller
    must bind the Hamiltonian, orbital energies and controls. The persistent
    source contract is implemented by ``evaluate_same_source_cc`` below.
    """
    options.validate()
    interval = _positive_integer(checkpoint_interval, 'checkpoint_interval')
    eps = np.asarray(eps)
    if eps.ndim != 1 or np.iscomplexobj(eps) or not np.isfinite(eps).all():
        raise ValueError('CC orbital energies must be finite real canonical values')
    no = integrals.no
    nv = len(eps)-no
    if min(no,nv)<=0 or integrals.shape != (len(eps),)*4:
        raise ValueError('CC requires nonempty occupied and virtual active spaces')
    estimate = workspace_estimate(no,nv) + getattr(integrals, 'factor_workspace_bytes', 0)
    if estimate > options.memory_byte_cap:
        raise MemoryError(f'CC workspace estimate {estimate} exceeds cap {options.memory_byte_cap}')
    d1 = eps[:no][None,:]-eps[no:][:,None]
    d2 = d1[:,None,:,None]+d1[None,:,None,:]
    masks = getattr(integrals, 'excitation_masks', None)
    m1, m2 = masks() if masks else (np.ones_like(d1,dtype=bool),np.ones_like(d2,dtype=bool))
    if (np.any(d1[m1]>=-options.denominator_tolerance)
            or np.any(d2[m2]>=-options.denominator_tolerance)):
        raise ValueError('CC denominator is nonnegative or too small; no regularization')
    t1 = np.zeros((nv,no),complex)
    t2 = np.divide(np.asarray(integrals[slice(no,None),slice(no,None),slice(no),slice(no)],dtype=complex),
                   d2,out=np.zeros(d2.shape,complex),where=m2)
    f = np.diag(eps)
    previous = None
    vectors, errors, trace = [], [], []
    start = 1
    if restart_state is not None:
        t1,t2,previous,vectors,errors,trace = _restore_solver_state(restart_state,no,nv)
        if restart_state['iteration'] > options.max_iter:
            raise ValueError('CC checkpoint iteration exceeds max_iter')
        if np.max(abs(t1[~m1]), initial=0.) > 1e-12 or np.max(abs(t2[~m2]), initial=0.) > 1e-12:
            raise ValueError('CC checkpoint amplitudes violate spin/momentum spaces')
        if (np.max(abs(t2+t2.swapaxes(0,1))) > 1e-10
                or np.max(abs(t2+t2.swapaxes(2,3))) > 1e-10):
            raise ValueError('CC checkpoint doubles violate antisymmetric spaces')
        for history in (vectors,errors):
            for entry in history:
                h1, h2 = entry[:no*nv].reshape(nv,no), entry[no*nv:].reshape(nv,nv,no,no)
                if (np.max(abs(h1[~m1]),initial=0.) > 1e-12
                        or np.max(abs(h2[~m2]),initial=0.) > 1e-12
                        or np.max(abs(h2+h2.swapaxes(0,1))) > 1e-10
                        or np.max(abs(h2+h2.swapaxes(2,3))) > 1e-10):
                    raise ValueError('CC checkpoint DIIS history violates excitation spaces')
        if restart_state['converged']:
            r1,r2 = residual(t1,t2,f,integrals,no)
            energy = _ccsd_energy_so(t1,t2,f,integrals,no)
            norm = max(np.max(abs(r1)),np.max(abs(r2)))
            delta = abs(energy-previous)
            if (not np.isfinite(energy) or not np.isfinite(norm)
                    or norm > options.residual_tolerance or delta > options.energy_tolerance
                    or abs(energy.imag) > options.energy_tolerance
                    or abs(energy.real-trace[-1][1]) > options.energy_tolerance
                    or abs(norm-trace[-1][2]) > options.residual_tolerance
                    or abs(delta-trace[-1][3]) > options.energy_tolerance):
                raise ValueError('CC checkpoint completed amplitudes fail convergence revalidation')
            return float(energy.real),t1,t2,tuple(trace),estimate
        start = restart_state['iteration'] + 1
        norm, delta = trace[-1][2:]
    for iteration in range(start,options.max_iter+1):
        r1,r2 = residual(t1,t2,f,integrals,no)
        energy = _ccsd_energy_so(t1,t2,f,integrals,no)
        norm = max(np.max(abs(r1)),np.max(abs(r2)))
        delta = np.inf if previous is None else abs(energy-previous)
        if not np.isfinite(energy) or not np.isfinite(norm):
            raise RuntimeError('CCSD diverged to nonfinite amplitudes/energy')
        trace.append((iteration,float(energy.real),float(norm),float(delta)))
        if norm <= options.residual_tolerance and delta <= options.energy_tolerance:
            if abs(energy.imag)>options.energy_tolerance:
                raise RuntimeError('CCSD correlation energy is not real')
            if progress_callback is not None:
                progress_callback(_solver_checkpoint(t1,t2,trace,previous,vectors,errors,True))
            return float(energy.real),t1,t2,tuple(trace),estimate
        previous = energy
        vector = np.r_[t1.ravel(),t2.ravel()]
        error = np.r_[np.divide(r1,d1,out=np.zeros_like(r1),where=m1).ravel(),
                      np.divide(r2,d2,out=np.zeros_like(r2),where=m2).ravel()]
        vectors.append(vector+error); errors.append(error)
        if len(vectors)>8:
            vectors.pop(0); errors.pop(0)
        updated = vector+error
        if len(vectors)>1:
            m = len(vectors)
            b = np.empty((m+1,m+1)); b[-1,:]=-1; b[:,-1]=-1; b[-1,-1]=0
            for i,j in itertools.product(range(m),repeat=2):
                b[i,j]=np.vdot(errors[i],errors[j]).real
            rhs = np.r_[np.zeros(m),-1.]
            try:
                weights = np.linalg.solve(b,rhs)[:m]
                updated = sum(w*x for w,x in zip(weights,vectors))
            except np.linalg.LinAlgError:
                pass  # Singular DIIS only changes the iteration, not the equations.
        t1 = updated[:no*nv].reshape(nv,no)
        t2 = updated[no*nv:].reshape(nv,nv,no,no)
        if progress_callback is not None and (iteration % interval == 0 or iteration == options.max_iter):
            progress_callback(_solver_checkpoint(t1,t2,trace,previous,vectors,errors,False))
    raise RuntimeError(f'CCSD did not converge in {options.max_iter} iterations; residual={norm:.3e}, energy_change={delta:.3e}')


@dataclass(frozen=True)
class CanonicalCCResult:
    e_hf_per_cell: float
    e_ccsd_per_cell: float
    e_triples_per_cell: float
    active_spaces: tuple
    source_identity: str
    reference_identity: str
    retained_metric_ranks: tuple
    source_descriptor_json: str
    scf_acceptance_json: str
    method: str
    iterations: int
    residual_norm: float
    energy_change: float
    max_t1: float
    max_t2: float
    workspace_bytes: int
    t1: object
    t2: object
    e_nuclear_per_cell: float = 0.0
    normalization: str = 'supercell integrals / Nk; correlation energy / Nk; per primitive cell'
    method_maturity: str = 'experimental'
    quantitative_status: str = 'not-qualified'
    public_admission: str = 'not-requested: developer evaluation'
    converged: bool = True
    triples_variant: str = 'canonical-spin-orbital-hermitian-(T)/v1; Raghavachari et al. (1989), Eq. 14'
    reference_method: str = 'RHF'
    canonical_residuals: tuple = ()
    spin_labels: tuple = ()
    restart_mode: str = 'fresh'
    resumed_cc_iterations: int = 0
    resumed_triples_work: int = 0
    triples_work_completed: int = 0
    triples_work_total: int = 0
    diis_restart_policy: str = 'fresh'
    convergence_history: tuple = ()
    ecp_json: str = '{}'
    electron_counts_json: str = '{}'

    @property
    def e_corr_per_cell(self):
        return self.e_ccsd_per_cell+self.e_triples_per_cell

    @property
    def e_total_per_cell(self):
        return self.e_hf_per_cell+self.e_corr_per_cell

    @property
    def e_total_electronic_per_cell(self):
        return self.e_total_per_cell-self.e_nuclear_per_cell

    @property
    def energy(self):
        return self.e_total_per_cell

    @property
    def backend(self):
        return 'chi-same-source-'+self.method.lower()


def evaluate_same_source_cc(reference, *, compute_triples=None, frozen_core=0,
                            auxiliary_batch_size=64, options=None,
                            checkpoint_path=None, resume_from=None, checkpoint_interval=1):
    """Evaluate sealed stationary RHF/UHF without any SCF/refit fallback.

    A resume binds the portable physical reference, canonical active spaces,
    source and equation controls. Only the iteration budget may increase.
    Publication is atomic through ``vibeqc.output.checkpoint``. With no new
    destination, resuming updates the original archive. Checkpoint integrity
    is separate from the unchanged experimental scientific acceptance.
    """
    from .canonical_mp2 import prepare_canonical_reference
    from .reference import ChiHFReference
    if not isinstance(reference, ChiHFReference):
        raise TypeError('chi CC requires a sealed ChiHFReference')
    if reference.method not in ('RHF','UHF'):
        raise NotImplementedError('chi CC requires RHF or UHF; ROHF/KS are unsupported')
    options = ChiCCOptions() if options is None else options
    if not isinstance(options, ChiCCOptions):
        raise TypeError('options must be ChiCCOptions')
    options.validate()
    compute_triples = options.compute_triples if compute_triples is None else compute_triples
    if not isinstance(compute_triples,bool):
        raise TypeError('compute_triples must be boolean')
    hf, adapter, channels, frozen, batch, source, report = prepare_canonical_reference(
        reference, frozen_core=frozen_core, auxiliary_batch_size=auxiliary_batch_size,
        denominator_tolerance=options.denominator_tolerance)
    if adapter.operator.descriptor.source != 'classic':
        raise NotImplementedError('chi CC requires classic full-Coulomb factors; SR CC remains unsupported')
    integrals = FactorIntegrals(adapter,tuple(c for c,e,n in channels),
        tuple(n for c,e,n in channels),frozen,batch)
    eps = np.array([channels[spin][1][k,p] for k,p,spin in integrals.labels])
    canonical_residuals = []
    overlap = np.asarray(hf.overlap)
    for spin,(c,e,n) in enumerate(channels):
        suffix = '' if reference.method == 'RHF' else ('_alpha','_beta')[spin]
        fock = np.asarray(getattr(hf,'fock'+suffix))
        density = np.asarray(getattr(hf,'density'+suffix))/(2 if reference.method == 'RHF' else 1)
        canonical_residuals.append((
            float(np.max(abs(c.conj().transpose(0,2,1) @ overlap @ c-np.eye(c.shape[-1])))),
            float(np.max(abs(fock @ c-(overlap @ c)*e[:,None,:]))),
            float(np.max(abs(c[:,:,:n] @ c[:,:,:n].conj().transpose(0,2,1)-density)))))
    interval = _positive_integer(checkpoint_interval, 'checkpoint_interval')
    destination = checkpoint_path if checkpoint_path is not None else resume_from
    persistent = destination is not None or resume_from is not None
    cc_state, triple_state = None, None
    resumed_iterations, resumed_work = 0, 0
    restart_mode, diis_policy = 'fresh', 'fresh'
    binding = None
    if persistent:
        from vibeqc.output.checkpoint import read_restart_state, write_restart_state
        from .checkpoint import portable_reference_identity
        binding = dict(reference=portable_reference_identity(reference),
            reference_method=reference.method, frozen=list(map(int,frozen)),
            auxiliary_batch_size=int(batch), compute_triples=compute_triples,
            energy_tolerance=float(options.energy_tolerance),
            residual_tolerance=float(options.residual_tolerance),
            denominator_tolerance=float(options.denominator_tolerance),
            memory_byte_cap=int(options.memory_byte_cap))
    if resume_from is not None:
        metadata, saved = read_restart_state(resume_from)
        if metadata.get('schema') != 'vibeqc.chi.canonical-cc/v1':
            raise ValueError('unsupported chi CC checkpoint schema; migration is not available')
        stage = metadata.get('stage')
        if (stage not in ('cc_iteration','cc_converged','triples','completed')
                or metadata.get('kind') != ('completed_result' if stage == 'completed' else 'correlation_state')
                or metadata.get('binding') != binding or (stage == 'triples' and not compute_triples)):
            raise ValueError('CC checkpoint source, spaces or numerical controls are incompatible')
        old_limit = metadata.get('max_iter')
        if isinstance(old_limit,bool) or not isinstance(old_limit,int) or options.max_iter < old_limit:
            raise ValueError('CC checkpoint max_iter may only increase')
        array_names = {'t1','t2','trace','previous','diis_vectors','diis_errors','eps','spin_labels'}
        if stage in ('triples','completed') and compute_triples:
            array_names.add('triples_energy')
        if (set(saved) != array_names or not np.array_equal(saved['eps'],eps)
                or not np.array_equal(saved['spin_labels'],np.asarray(integrals.labels,dtype=int))):
            raise ValueError('CC checkpoint canonical orbital energies or spin spaces are incompatible')
        cc_state = dict(iteration=metadata.get('iteration'),converged=stage != 'cc_iteration',
            **{name:saved[name] for name in ('t1','t2','trace','previous','diis_vectors','diis_errors')})
        resumed_iterations = cc_state['iteration']
        if not isinstance(resumed_iterations,int) or isinstance(resumed_iterations,bool):
            raise ValueError('CC checkpoint iteration count is invalid')
        if stage in ('triples','completed') and compute_triples:
            triple_state = dict(metadata.get('triples',{}),energy=saved['triples_energy'])
            resumed_work = triple_state.get('completed_work',0)
            if stage == 'completed' and resumed_work != triple_state.get('total_work'):
                raise ValueError('CC checkpoint completed triples correction has incomplete coverage')
        elif 'triples' in metadata:
            raise ValueError('CC checkpoint stage disagrees with triples progress')
        restart_mode = 'restored-completed-result' if stage == 'completed' else 'resumed-correlation'
        diis_policy = 'reused-compatible-history'

    def publish(stage):
        if destination is None:
            return
        metadata = dict(schema='vibeqc.chi.canonical-cc/v1',
            kind='completed_result' if stage == 'completed' else 'correlation_state',
            stage=stage,binding=binding,max_iter=int(options.max_iter),
            iteration=cc_state['iteration'],diis_restart_policy=diis_policy)
        arrays = {name:cc_state[name] for name in
                  ('t1','t2','trace','previous','diis_vectors','diis_errors')}
        arrays.update(eps=eps,spin_labels=np.asarray(integrals.labels,dtype=int))
        if triple_state is not None:
            metadata['triples'] = {k:v for k,v in triple_state.items() if k != 'energy'}
            arrays['triples_energy'] = triple_state['energy']
        write_restart_state(destination,metadata,arrays)

    def cc_progress(state):
        nonlocal cc_state
        cc_state = state
        publish('cc_converged' if state['converged'] else 'cc_iteration')

    ecc,t1,t2,trace,estimate = solve(integrals,eps,options,restart_state=cc_state,
        progress_callback=cc_progress if persistent else None,checkpoint_interval=interval)

    def triples_progress(state):
        nonlocal triple_state
        triple_state = state
        publish('triples')

    et = triples(t1,t2,eps,integrals,denominator_tolerance=float(options.denominator_tolerance),
        restart_state=triple_state,progress_callback=triples_progress,
        checkpoint_interval=interval) if compute_triples else 0.
    adapter.validate()
    reference.validate(reference.request_identity,reference.method)
    publish('completed')
    t1.setflags(write=False); t2.setflags(write=False)
    from .ecp import electron_accounting
    accounting = electron_accounting(hf, frozen, tuple(c[2] for c in channels))
    return CanonicalCCResult(float(hf.energy),ecc/adapter.nk,et/adapter.nk,
        tuple((int(frozen[s]),n-int(frozen[s]),c.shape[-1]-n) for s,(c,e,n) in enumerate(channels)),
        adapter.source_identity,reference.state_identity,adapter.retained_ranks,
        json.dumps(source.as_dict(),sort_keys=True),json.dumps(dict(report),sort_keys=True),
        ('U' if reference.method == 'UHF' else '')+('CCSD(T)' if compute_triples else 'CCSD'),
        trace[-1][0],trace[-1][2],trace[-1][3],float(np.max(abs(t1))),float(np.max(abs(t2))),
        estimate,t1,t2,e_nuclear_per_cell=float(getattr(hf,'e_nuclear',0.)),
        reference_method=reference.method,canonical_residuals=tuple(canonical_residuals),
        spin_labels=tuple(integrals.labels),restart_mode=restart_mode,
        resumed_cc_iterations=resumed_iterations,resumed_triples_work=resumed_work,
        triples_work_completed=triple_state['completed_work'] if triple_state else 0,
        triples_work_total=triple_state['total_work'] if triple_state else 0,
        diis_restart_policy=diis_policy,convergence_history=trace,
        ecp_json=json.dumps(getattr(hf, 'chi_ecp', None) or {}, sort_keys=True),
        electron_counts_json=json.dumps(accounting, sort_keys=True))
