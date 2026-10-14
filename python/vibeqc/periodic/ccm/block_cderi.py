"""Block-circulant storage of the real-Γ neutral cderi and its Fock build.

The real-Γ producer (:func:`~vibeqc.periodic.ccm.direct.run_ccm_rhf_direct`,
``variant="real-gamma"``) fits the neutral torus Coulomb operator per momentum
transfer ``q`` from unit-cell Bloch-pair fits ``L_{k,k+q}[P,mu,nu]`` and, on
its historical path, folds them onto the torus into one dense real array
``L[A,(mu c),(nu c')]`` of ``8 n_aux n_mu^2 N_c^3`` bytes
(:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi_fold`). That array is
a twisted block-circulant matrix in every auxiliary row,

    B_q[P,(mu c),(nu c')] = N_c^{-1/2} e^{-i q.R_c'} Lt_q[P,mu,nu; c-c'],
    Lt_q[P; g] = (1/N_c) sum_ka e^{+i ka.R_g} L_{ka,ka+q}[P],

so only one block row (over the ``N_c`` torus offsets ``g``) is independent,
and that block row is the inverse mesh transform of the per-bra fits. This
module keeps the fits themselves, per kept ``q``-channel (one representative
of each ``+-q`` pair, the same channel list the fold uses): ``16 N_kept N_c
n_aux n_mu^2`` bytes, ``O(N_c^2)`` instead of ``O(N_c^3)``, and evaluates the
Coulomb and exchange operators of the SAME contraction the dense array
defines (``J = sum_A L_A Tr(L_A D)``, ``K = sum_A L_A D L_A``) on the
circulant projection of the torus density in the mesh representation:

    gamma_P = (1/N_c) sum_k conj Tr[L_{k,k}[P] Dh(k)],     Jh(k) = sum_P L_{k,k}[P] gamma_P
    Kh_q(k) = (1/N_c) sum_P L_{k,k+q}[P] Dh(k+q) L_{k,k+q}[P]^H,   Kh = sum_{q kept} w_q^2 Kh_q
    M(g)    = Re (1/N_c) sum_k e^{-i k.R_g} Mh(k),         M_sc[(mu c),(nu c')] = M(c'-c)

with ``Dh(k) = sum_g e^{+i k.R_g} D(g)``, ``D(g)`` the cell-averaged torus
blocks (the projector the real-Γ XC builder already uses), ``w_q`` the fold's
``+-q`` stacking weight, and the exchange contracted through the occupied-rank
factor ``Dh(k+q) = X X^H`` (the per-k sibling of
:func:`~vibeqc.periodic.ccm.ri.psd_density_factor`). For a translation-
invariant iterate this is an exact reassociation of the dense contraction; for
one that is not, it evaluates ``J(P D)``/``K(P D)`` with ``P`` the self-adjoint
circulant projector, so ``F = dE/dD`` holds for the composed functional
``E[P(D)]`` whose stationary points on the circulant manifold are those of
``E[D]``. The exchange ``q = 0`` seam ``K -> K + xi_N S D S`` enters per k as
``xi_N (Sh X)(Sh X)^H`` on the same projected density; the hybrid fraction is
applied by the KS loop to whatever this object returns, exactly as with the
dense array.

Flop counts per SCF iteration: the dense factored exchange costs
``6 N_c^4 n_aux n_mu^2 n_occ``, this build ``16 N_kept N_c n_aux n_mu^2 n_occ``
(``n_occ`` per cell), a ratio of ``0.75 N_c^2``.

**Wigner-Seitz partition (paper issue O25).** The independent block row is
indexed by the torus offset ``g``, and ``Lt_q[P; g]`` is the fit coefficient
of the AO pair ``(mu, 0 | nu, g)`` (Bloch-summed auxiliary at ``q``), which
decays with the minimum-image distance ``d(g)`` of the offset under the
cluster (supercell) lattice as the AO-pair overlap does. The offsets are
ordered by ``d(g)`` (:func:`~vibeqc.periodic.ccm.wigner_seitz.minimum_image`),
and a threshold omits every block beyond it by projecting the stored fits,
``L' = sum_{g kept} e^{-i ka.R_g} Lt[g]``; the omitted Frobenius norm is
reported. Default: no omission (the storage is exact). The auxiliary index is
left in its Bloch form on purpose: its real-space decay is Coulombic and not
truncatable.

Cite Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
doi:10.1063/1.4998644 for the periodic GDF factorisation the fits realise,
and Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550 for the Wigner-Seitz supercell ordering.
"""

from __future__ import annotations

import numpy as np

from vibeqc.output import Level, write

__all__ = [
    "CCMBlockCderi",
    "BlockChannel",
    "fold_channel_dense",
    "is_block_cderi",
]

#: The fold's null-row pruning threshold (squared Frobenius norm of one
#: auxiliary row of the dense array), reproduced by :meth:`CCMBlockCderi.to_dense`.
_NULL_ROW_NORM2 = 1e-14

#: Relative eigenvalue cut of the per-k density factor -- the rule of
#: :func:`~vibeqc.periodic.ccm.ri.psd_density_factor`.
_FACTOR_TOL = 1e-12

EXCHANGE_FORMS = ("transpose", "pair")


def is_block_cderi(obj) -> bool:
    """``True`` for a :class:`CCMBlockCderi` (duck-typed on its build API)."""
    return isinstance(obj, CCMBlockCderi)


def fold_channel_dense(fits, kpts, kq, r_c, alpha, weight):
    """Fold one channel's per-bra fits onto the torus: ``(w Re B_q, w Im B_q)``.

    Verbatim the accumulation of ``ccm_neutral_cderi_fold``'s channel fold
    (2026-09-24 form): ``fits[ai]`` is the canonical-frame ``L_{ka, ka+q}``
    for bra ``kpts[ai]``, ``kq`` the channel momentum, ``r_c`` the torus cell
    vectors in CCMSystem order, ``alpha = N_c^{-3/2}``. Shared by the fold and
    by :meth:`CCMBlockCderi.to_dense` so the two dense arrays are one code
    path.
    """
    n_c = len(fits)
    acc = None
    for ai in range(n_c):
        l_ab = np.asarray(fits[ai])
        n_mu = int(l_ab.shape[1])
        kb_cart = kpts[ai] + kq
        if acc is None:
            n_sc = n_c * n_mu
            acc = np.zeros((l_ab.shape[0], n_sc, n_sc), dtype=complex)
        pha = np.exp(+1j * (r_c @ kpts[ai]))
        phb = np.exp(-1j * (r_c @ kb_cart))
        acc.reshape(-1, n_c, n_mu, n_c, n_mu)[...] += np.einsum(
            "c,d,Amn->Acmdn", pha, phb, l_ab, optimize=True)
    acc *= alpha
    return (weight * np.ascontiguousarray(acc.real),
            weight * np.ascontiguousarray(acc.imag))


def _hermitian_psd_factor(dhat, *, tol=_FACTOR_TOL, who="CCMBlockCderi"):
    """Rank-revealing factor ``X`` of a Hermitian PSD ``Dh(k)``: ``Dh = X X^H``.

    The per-k sibling of :func:`~vibeqc.periodic.ccm.ri.psd_density_factor`
    (same eigenvalue rule: eigenpairs below ``tol`` relative to the largest
    are dropped, an aufbau ``Dh(k)`` returns rank ``n_occ(k)``; a
    significantly indefinite input raises rather than being projected).
    """
    dhat = np.asarray(dhat)
    w, U = np.linalg.eigh(dhat)
    w_max = max(float(w[-1]), 0.0)
    if float(w[0]) < -1e-8 * max(w_max, 1.0):
        raise ValueError(
            f"{who}: a mesh density block is significantly indefinite "
            f"(min eig {float(w[0]):.3e}); the factored exchange requires an "
            "aufbau-PSD density.")
    keep = w > tol * max(w_max, 1.0)
    return np.ascontiguousarray(U[:, keep] * np.sqrt(w[keep])[None, :])


class BlockChannel:
    """The per-bra fits of one kept momentum-transfer channel.

    ``fits[ai]`` is ``L_{ka, ka+q}[P,mu,nu]`` (complex, canonical auxiliary
    frame) for the bra ``ka = kpts[ai]`` of the fold's mesh, ``q = kpts[qi]``,
    the ket UNFOLDED (``ka + q`` in cartesian coordinates, one whitening frame
    per channel, exactly as the fold builds them). ``weight`` is the fold's
    ``+-q`` stacking weight (``1`` self-paired, ``sqrt 2`` for a pair
    representative).
    """

    __slots__ = ("qi", "label", "weight", "self_paired", "fits")

    def __init__(self, qi, label, weight, self_paired, fits):
        self.qi = int(qi)
        self.label = tuple(int(x) for x in label)
        self.weight = float(weight)
        self.self_paired = bool(self_paired)
        self.fits = np.ascontiguousarray(fits, dtype=complex)
        if self.fits.ndim != 4 or self.fits.shape[2] != self.fits.shape[3]:
            raise ValueError(
                "BlockChannel: fits must be (N_c, n_aux, n_mu, n_mu); got "
                f"{self.fits.shape}")

    @property
    def n_aux(self) -> int:
        return int(self.fits.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.fits.nbytes)


class CCMBlockCderi:
    """Per-q block storage of the real-Γ neutral cderi and its J/K builds.

    Built by :func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi_blocks`
    (the fold's own builder, stopping before the fold). Every contraction
    the dense array supports is available: :meth:`coulomb`, :meth:`exchange`
    / :meth:`exchange_operator` (the SCF's ``j_of_D`` / ``k_of_D``),
    :meth:`mo_factor` (the correlated consumers' three-index MO factor),
    :meth:`to_dense` (the fold's array itself, for consumers that need it).
    The channel list is the unit a distribution along ``q`` would farm.
    """

    def __init__(self, *, nrep, kpts, frac, cells, r_c, a_lat, channels,
                 info=None):
        self.nrep = tuple(int(n) for n in nrep)
        self.n_c = int(np.prod(self.nrep))
        self.kpts = np.ascontiguousarray(kpts, dtype=float).reshape(self.n_c, 3)
        self.frac = np.ascontiguousarray(frac, dtype=int).reshape(self.n_c, 3)
        self.cells = np.ascontiguousarray(cells, dtype=int).reshape(self.n_c, 3)
        self.r_c = np.ascontiguousarray(r_c, dtype=float).reshape(self.n_c, 3)
        self.a_lat = np.ascontiguousarray(a_lat, dtype=float).reshape(3, 3)
        self.channels = list(channels)
        if not self.channels:
            raise ValueError("CCMBlockCderi: no channels")
        self.n_mu = int(self.channels[0].fits.shape[2])
        for ch in self.channels:
            if ch.fits.shape[0] != self.n_c or ch.fits.shape[2] != self.n_mu:
                raise ValueError(
                    "CCMBlockCderi: channel fits must be (N_c, n_aux, n_mu, "
                    f"n_mu) = ({self.n_c}, *, {self.n_mu}, {self.n_mu}); got "
                    f"{ch.fits.shape} for channel {ch.label}")
        self.n_sc = self.n_c * self.n_mu
        self.info = dict(info or {})
        self.ws_cutoff_bohr = None
        self.ws_report = None
        self._imag_residual = 0.0

        nvec = np.asarray(self.nrep, dtype=int)
        # Integer-label index (mesh labels mod nrep) and torus tables.
        self._label_index = {
            tuple(int(x) for x in self.frac[i]): i for i in range(self.n_c)}
        if len(self._label_index) != self.n_c:
            raise ValueError("CCMBlockCderi: BvK mesh labels are not unique")
        cell_index = {
            tuple(int(x) for x in self.cells[i]): i for i in range(self.n_c)}
        plus = np.empty((self.n_c, self.n_c), dtype=int)
        diff = np.empty((self.n_c, self.n_c), dtype=int)
        for c in range(self.n_c):
            for g in range(self.n_c):
                plus[c, g] = cell_index[tuple(int(x) for x in
                                              (self.cells[c] + self.cells[g]) % nvec)]
                diff[c, g] = cell_index[tuple(int(x) for x in
                                              (self.cells[g] - self.cells[c]) % nvec)]
        self._plus = plus      # plus[c, g]  = index of cell c + g (mod N)
        self._diff = diff      # diff[c, c'] = index of cell c' - c (mod N)
        # Mesh phases E[k, g] = exp(+i k.R_g).
        self._E = np.exp(1j * (self.kpts @ self.r_c.T))
        # Per channel: the ket index of (ka + q) and the partner bra k' = -ka - q.
        self._ket = {}
        self._partner = {}
        q0 = None
        for ch in self.channels:
            fq = self.frac[ch.qi]
            self._ket[ch.qi] = np.array([
                self._label_index[tuple(int(x) for x in (self.frac[a] + fq) % nvec)]
                for a in range(self.n_c)], dtype=int)
            self._partner[ch.qi] = np.array([
                self._label_index[tuple(int(x) for x in (-self.frac[a] - fq) % nvec)]
                for a in range(self.n_c)], dtype=int)
            if ch.label == (0, 0, 0):
                q0 = ch
        if q0 is None:
            raise ValueError("CCMBlockCderi: the q = 0 channel is missing")
        self._q0 = q0

    # ---- sizes -----------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def aux_ranks(self) -> tuple:
        return tuple(ch.n_aux for ch in self.channels)

    @property
    def nbytes(self) -> int:
        """Bytes held by the stored fits."""
        return int(sum(ch.nbytes for ch in self.channels))

    @property
    def dense_nbytes_upper_bound(self) -> int:
        """Bytes the fold's dense array would take before null-row pruning."""
        rows = 2 * sum(ch.n_aux for ch in self.channels)
        return int(8 * rows * self.n_sc * self.n_sc)

    @property
    def shape(self) -> tuple:
        """The dense array's shape upper bound ``(rows, n_sc, n_sc)``."""
        return (2 * sum(ch.n_aux for ch in self.channels), self.n_sc, self.n_sc)

    @property
    def imag_residual(self) -> float:
        """Largest imaginary part discarded by the last mesh-to-torus transform.

        Not an error measure: a ``+-q`` pair representative's contribution is
        complex by construction (its partner channel, not stored, carries the
        conjugate), and the ``w^2 Re`` of the fold's real stacking discards
        exactly this part; only self-paired channels give a real sum on their
        own. It is zero to rounding for the round trip of a real block row and
        on meshes where every channel is self-paired (N = 2).
        """
        return float(self._imag_residual)

    def describe(self) -> str:
        mib = self.nbytes / 2.0 ** 20
        dense_mib = self.dense_nbytes_upper_bound / 2.0 ** 20
        ranks = sorted(set(self.aux_ranks))
        return (f"block storage: {self.n_channels} q-channels x {self.n_c} fits, "
                f"aux ranks {ranks[0]}..{ranks[-1]}, {mib:.1f} MiB "
                f"(dense array at most {dense_mib:.1f} MiB)")

    # ---- torus <-> mesh transforms -----------------------------------------
    def torus_blocks(self, m_sc):
        """Cell-averaged torus blocks ``M(g)[mu,nu] = (1/N_c) sum_c M[(mu c),(nu c+g)]``.

        Exact block extraction for a block-circulant matrix; the circulant
        projection ``P(M)`` otherwise (linear, self-adjoint).
        """
        m4 = np.asarray(m_sc).reshape(self.n_c, self.n_mu, self.n_c, self.n_mu)
        idx_c = np.arange(self.n_c)[:, None]
        gathered = m4[idx_c, :, self._plus, :]        # (c, g, mu, nu)
        return gathered.mean(axis=0)

    def bloch(self, blocks):
        """``Mh(k) = sum_g e^{+i k.R_g} M(g)`` for every mesh point."""
        return np.tensordot(self._E, np.asarray(blocks), axes=(1, 0))

    def unbloch_real(self, mhat):
        """``M(g) = Re (1/N_c) sum_k e^{-i k.R_g} Mh(k)``; records the discarded
        imaginary part on :attr:`imag_residual` (see there: not an error)."""
        out = np.tensordot(np.conj(self._E).T, np.asarray(mhat), axes=(1, 0))
        out /= float(self.n_c)
        self._imag_residual = float(np.max(np.abs(out.imag))) if out.size else 0.0
        return np.ascontiguousarray(out.real)

    def embed(self, blocks):
        """Block-circulant torus matrix from its block row: ``M[(mu c),(nu c')] = M(c'-c)``."""
        blocks = np.asarray(blocks)
        m4 = blocks[self._diff]                        # (c, c', mu, nu)
        return np.ascontiguousarray(
            m4.transpose(0, 2, 1, 3).reshape(self.n_sc, self.n_sc))

    def density_bloch(self, d_sc):
        """``Dh(k)`` of a torus density (its circulant projection)."""
        return self.bloch(self.torus_blocks(d_sc))

    # ---- J and K -----------------------------------------------------------
    def coulomb(self, d_sc):
        """RI-consistent neutral Coulomb ``J`` of the torus density (q = 0 channel)."""
        dhat = self.density_bloch(d_sc)
        l0 = self._q0.fits                             # (k, P, mu, nu) = L_{k,k}
        # gamma_P = (1/N_c) sum_k conj(Tr[L_{k,k}[P] Dh(k)]): real at q = 0.
        gamma = np.einsum("kPrs,ksr->P", l0, dhat, optimize=True)
        gamma = np.conj(gamma) / float(self.n_c)
        jhat = np.einsum("kPmn,P->kmn", l0, gamma, optimize=True)
        jhat *= self._q0.weight ** 2
        return self.embed(self.unbloch_real(jhat))

    def _factors(self, dhat):
        """Zero-padded per-k density factors ``X_k`` as one ``(N_c, n_mu, r)`` array."""
        facs = [_hermitian_psd_factor(dhat[k]) for k in range(self.n_c)]
        r = max((f.shape[1] for f in facs), default=0)
        out = np.zeros((self.n_c, self.n_mu, max(r, 1)), dtype=complex)
        for k, f in enumerate(facs):
            out[k, :, :f.shape[1]] = f
        return out

    def _exchange_bloch(self, dhat, *, form="transpose"):
        """``Kh(k) = sum_{q kept} w_q^2 Kh_q(k)`` through the density factors."""
        if form not in EXCHANGE_FORMS:
            raise ValueError(
                f"CCMBlockCderi: exchange_form must be one of {EXCHANGE_FORMS}, "
                f"got {form!r}")
        n_c, n_mu = self.n_c, self.n_mu
        x = self._factors(dhat)                        # (k, mu, r)
        r = int(x.shape[2])
        khat = np.zeros((n_c, n_mu, n_mu), dtype=complex)
        for ch in self.channels:
            l = ch.fits                                # (a, P, mu, nu)
            n_aux = ch.n_aux
            ket = self._ket[ch.qi]
            # B[a, P, mu, i] = sum_rho L_{ka,ka+q}[P,mu,rho] X_{ka+q}[rho,i]
            b = np.matmul(l, x[ket][:, None, :, :])    # (a, P, mu, r)
            bf = b.transpose(0, 2, 1, 3).reshape(n_c, n_mu, n_aux * r)
            scale = ch.weight ** 2 / float(n_c)
            if form == "transpose":
                # Kh_q(ka) = (1/N_c) sum_{P,i} B[P,mu,i] conj(B[P,nu,i]):
                # L_{-ka-q,-ka}[P,s,n] = L_{ka,ka+q}[P,n,s] (real AOs).
                khat += scale * np.matmul(bf, np.conj(bf).transpose(0, 2, 1))
            else:
                # The dense contraction verbatim: the second factor is the
                # stored fit at the partner bra k' = -ka-q, conjugated
                # elementwise, no pair-density symmetry assumed.
                partner = self._partner[ch.qi]
                c = np.matmul(np.conj(x[ket]).transpose(0, 2, 1)[:, None, :, :],
                              np.conj(l[partner]))    # (a, P, r, nu)
                cf = c.reshape(n_c, n_aux * r, n_mu)
                khat += scale * np.matmul(bf, cf)
        return khat

    def exchange_operator(self, S=None, eta=0.0, *, form="transpose"):
        """The SCF's ``k_of_D``: ``D_sc -> K_sc (+ eta S D S)`` on the projected density.

        ``eta`` is the exchange-``q=0`` seam ``xi_N`` (or a screened kernel's
        ``sigma_0``); ``S`` the torus overlap, block-circulant by construction,
        whose mesh blocks are extracted once here.
        """
        if form not in EXCHANGE_FORMS:
            raise ValueError(
                f"CCMBlockCderi: exchange_form must be one of {EXCHANGE_FORMS}, "
                f"got {form!r}")
        eta = float(eta)
        shat = None
        if eta != 0.0:
            if S is None:
                raise ValueError("CCMBlockCderi.exchange_operator: eta != 0 needs S")
            shat = self.bloch(self.torus_blocks(S))

        def k_of_D(d_sc):
            dhat = self.density_bloch(d_sc)
            khat = self._exchange_bloch(dhat, form=form)
            if shat is not None:
                # xi_N S D S = xi_N (Sh X)(Sh X)^H per k, on the same projection.
                x = self._factors(dhat)
                sx = np.matmul(shat, x)
                khat += eta * np.matmul(sx, np.conj(sx).transpose(0, 2, 1))
            return self.embed(self.unbloch_real(khat))

        return k_of_D

    def exchange(self, d_sc, S=None, eta=0.0, *, form="transpose"):
        """``K_sc (+ eta S D S)`` of one torus density."""
        return self.exchange_operator(S, eta, form=form)(d_sc)

    # ---- consumers ---------------------------------------------------------
    def to_dense(self):
        """The fold's dense real array (``{w Re B_q, w Im B_q}`` stacked, null
        rows pruned) from the stored fits: the same code path as
        ``ccm_neutral_cderi_fold``, so the two agree to the bit."""
        alpha = float(self.n_c) ** -1.5
        blocks = []
        for ch in self.channels:
            re_block, im_block = fold_channel_dense(
                [ch.fits[a] for a in range(self.n_c)], self.kpts,
                self.kpts[ch.qi], self.r_c, alpha, ch.weight)
            blocks.append(re_block)
            blocks.append(im_block)
        out = np.concatenate(blocks, axis=0)
        keep = np.einsum("Amn,Amn->A", out, out) > _NULL_ROW_NORM2
        return np.ascontiguousarray(out[keep])

    def mo_factor(self, c_left, c_right):
        """The three-index MO factor ``B[A,i,a] = sum_IJ L[A,I,J] C_l[I,i] C_r[J,a]``
        of the dense array, built per channel without forming it.

        With the mesh transform of the real supercell coefficients
        ``u(k)[mu,i] = sum_c e^{+i k.R_c} C[(mu c),i]``:
        ``T_q[P,i,a] = N_c^{-3/2} sum_ka u_l(ka)^T L_{ka,ka+q}[P] conj(u_r(ka+q))``
        and the rows are ``{w Re T_q, w Im T_q}`` in the fold's channel order
        (null rows are kept: they are exactly the transforms of the fold's
        pruned rows and contribute nothing to any product). This is what
        ``run_ccm_mp2`` contracts for ``(ia|jb)``; the cost is the MO
        transform's own, the supercell orbitals not being Bloch-adapted.
        """
        c_l = np.asarray(c_left, dtype=float)
        c_r = np.asarray(c_right, dtype=float)
        if c_l.shape[0] != self.n_sc or c_r.shape[0] != self.n_sc:
            raise ValueError(
                f"CCMBlockCderi.mo_factor: coefficient rows must be {self.n_sc}; "
                f"got {c_l.shape[0]} and {c_r.shape[0]}")
        n_l, n_r = int(c_l.shape[1]), int(c_r.shape[1])
        u_l = np.tensordot(self._E, c_l.reshape(self.n_c, self.n_mu, n_l), axes=(1, 0))
        u_r = np.tensordot(self._E, c_r.reshape(self.n_c, self.n_mu, n_r), axes=(1, 0))
        alpha = float(self.n_c) ** -1.5
        rows = []
        for ch in self.channels:
            ket = self._ket[ch.qi]
            t1 = np.matmul(u_l.transpose(0, 2, 1)[:, None, :, :], ch.fits)   # (a, P, i, nu)
            t2 = np.matmul(t1, np.conj(u_r[ket])[:, None, :, :])             # (a, P, i, r)
            t = t2.sum(axis=0) * alpha
            rows.append(ch.weight * np.ascontiguousarray(t.real))
            rows.append(ch.weight * np.ascontiguousarray(t.imag))
        return np.concatenate(rows, axis=0)

    # ---- Wigner-Seitz partition (O25) ---------------------------------------
    def ws_distances(self):
        """Minimum-image distance (bohr) and boundary multiplicity of every
        torus offset ``g`` under the cluster (supercell) lattice."""
        from .wigner_seitz import minimum_image

        cluster = (self.a_lat * np.asarray(self.nrep, dtype=float)[None, :]).T  # rows
        cells, weights = minimum_image(self.r_c, cluster)
        dist = np.array([
            float(np.linalg.norm(self.r_c[g] - cells[g][0] @ cluster))
            for g in range(self.n_c)])
        mult = np.array([len(c) for c in cells], dtype=int)
        return dist, mult

    def ws_shells(self):
        """``[(distance_bohr, n_offsets)]`` in increasing distance."""
        dist, _ = self.ws_distances()
        shells = []
        for d in sorted(set(np.round(dist, 6).tolist())):
            shells.append((float(d), int(np.sum(np.abs(dist - d) < 1e-5))))
        return shells

    def apply_ws_cutoff(self, cutoff_bohr):
        """Omit every block row whose offset lies beyond ``cutoff_bohr``
        (minimum image), in place, and record the omitted norm.

        ``None`` records the shells and omits nothing. The projection
        ``L' = sum_{g kept} e^{-i ka.R_g} Lt[g]`` with
        ``Lt[g] = (1/N_c) sum_k e^{+i k.R_g} L_k`` is idempotent, so applying
        it twice, or with a larger cutoff after a smaller one, changes
        nothing beyond the first application.
        """
        dist, mult = self.ws_distances()
        order = np.argsort(dist, kind="stable")
        shells = self.ws_shells()
        report = {
            "cutoff_bohr": None if cutoff_bohr is None else float(cutoff_bohr),
            "shells": shells,
            "offset_order": [int(g) for g in order],
            "n_offsets": int(self.n_c),
            "n_omitted": 0,
            "omitted_norm": 0.0,
            "omitted_norm_rel": 0.0,
            "total_norm": None,
        }
        total2 = 0.0
        for ch in self.channels:
            total2 += float(np.sum(np.abs(ch.fits) ** 2)) / float(self.n_c)
        report["total_norm"] = float(np.sqrt(total2))
        self.ws_cutoff_bohr = report["cutoff_bohr"]
        if cutoff_bohr is None:
            self.ws_report = report
            return report
        cutoff = float(cutoff_bohr)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError(
                f"CCMBlockCderi: ws_cutoff_bohr must be finite and >= 0, got {cutoff!r}")
        keep = dist <= cutoff + 1e-9
        report["n_omitted"] = int(np.sum(~keep))
        if report["n_omitted"] == 0:
            self.ws_report = report
            return report
        omitted2 = 0.0
        e = self._E                                   # E[k, g]
        for ch in self.channels:
            flat = ch.fits.reshape(self.n_c, -1)      # (k, P mu nu)
            lt = (e.T @ flat) / float(self.n_c)       # Lt[g] = (1/N) sum_k E[k,g] L_k
            omitted2 += float(np.sum(np.abs(lt[~keep]) ** 2))
            lt[~keep] = 0.0
            ch.fits = np.ascontiguousarray(
                (np.conj(e) @ lt).reshape(ch.fits.shape))   # L'_k = sum_g conj(E[k,g]) Lt[g]
        report["omitted_norm"] = float(np.sqrt(omitted2))
        report["omitted_norm_rel"] = (
            float(np.sqrt(omitted2) / report["total_norm"]) if total2 > 0 else 0.0)
        self.ws_report = report
        write(
            f"  CCMBlockCderi: Wigner-Seitz cutoff {cutoff:g} bohr omits "
            f"{report['n_omitted']}/{self.n_c} block rows (min-image shells "
            f"{[round(d, 3) for d, _ in shells]}), omitted norm "
            f"{report['omitted_norm']:.3e} ({report['omitted_norm_rel']:.3e} relative)\n",
            Level.VERBOSE)
        return report
