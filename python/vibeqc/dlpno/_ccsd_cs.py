"""Spatial closed-shell (RHF) DF-CCSD residual + solver.

The spin-orbital reference kernel (`_ccsd_ref`) is the FCI-anchored
ground truth, but it carries a 16x spin-block redundancy and is the
wrong basis to localise from -- the DLPNO pair amplitudes are *spatial*.
This module is the spatial closed-shell DF-CCSD residual, a faithful
numpy transcription of the validated C++ kernel (`cpp/src/ccsd.cpp`,
closed-shell spin integration of Stanton-Gauss-Watts-Bartlett 1991;
DePrince & Sherrill, J. Chem. Phys. 139, 174102 (2013)), validated here
by converged-energy parity against the spin-orbital kernel.

It is both a real cost reduction for the DLPNO-CCSD pilot's
per-iteration step (spatial, not spin-orbital) and the exact template
the per-pair *local* residual mirrors -- the local version is these same
contractions evaluated in each pair's PNO basis with cross-pair
PNO-overlap projections; remaining production gates live in
``handovers/HANDOVER_GATED_ITEMS.md``.

Density-fitted Coulomb integrals (pq|rs) = S_P B^P_{pq} B^P_{rs}; dense
full-space here (the oracle). The diagonal Fock is kept inside
F_ae / F_mi, so the returned residuals are true residuals and the
amplitude update is t += R / D.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CSCCSDResult:
    e_corr: float = 0.0
    e_hf: float = 0.0
    e_total: float = 0.0
    n_iter: int = 0
    converged: bool = False
    t1: np.ndarray | None = None
    t2: np.ndarray | None = None
    trace: list = field(default_factory=list)


@dataclass
class CSCCSDLambdaResult:
    """Closed-shell CCSD Lambda amplitudes for the spatial residual oracle."""

    l1: np.ndarray | None = None
    l2: np.ndarray | None = None
    residual_norm: float = 0.0
    condition_number: float = 0.0
    n_amplitudes: int = 0
    n_iter: int = 0
    converged: bool = False


def _blocks(B_ov, B_vv, B_oo, *, include_vvvv=True):
    """Chemist-notation Coulomb blocks from the DF B-tensors.

    ``include_vvvv=False`` omits the O(nv^4) (ab|cd) block, which only the
    particle-particle ladder consumes. Callers that contract that term
    elsewhere (the DLPNO pair-space ladder of #700) must not pay to build or
    hold it: on the extended domain it is the single largest array here.
    """
    out = dict(
        ovov=np.einsum("Pia,Pjb->iajb", B_ov, B_ov, optimize=True),  # (ia|jb)
        ovvv=np.einsum("Pia,Pbc->iabc", B_ov, B_vv, optimize=True),  # (ia|bc)
        ooov=np.einsum("Pij,Pka->ijka", B_oo, B_ov, optimize=True),  # (ij|ka)
        oooo=np.einsum("Pij,Pkl->ijkl", B_oo, B_oo, optimize=True),  # (ij|kl)
        oovv=np.einsum("Pij,Pab->ijab", B_oo, B_vv, optimize=True),  # (ij|ab)
    )
    if include_vvvv:
        out["vvvv"] = np.einsum(
            "Pab,Pcd->abcd", B_vv, B_vv, optimize=True
        )  # (ab|cd)
    return out


def _sum_to_shape(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Undo NumPy broadcasting for one reverse-mode edge."""
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value


class _ResidualAD:
    """Small tensor reverse-mode node used by the CCSD residual VJP.

    The residual is already expressed as tensor contractions. Recording those
    contractions, rather than individual scalar operations, keeps the reverse
    pass at the same polynomial scaling as one residual evaluation.
    """

    __array_priority__ = 10_000

    def __init__(self, value, parents=()):
        self.value = np.asarray(value, dtype=float)
        self.parents = tuple(parents)
        self.gradient = None

    @property
    def shape(self):
        return self.value.shape

    def copy(self):
        return _ResidualAD(self.value.copy(), ((self, lambda grad: grad),))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        if not axes:
            axes = tuple(reversed(range(self.value.ndim)))
        inverse = tuple(np.argsort(axes))
        return _ResidualAD(
            self.value.transpose(*axes),
            ((self, lambda grad: grad.transpose(*inverse)),),
        )

    @property
    def T(self):
        return self.transpose()

    def __array_function__(self, func, types, args, kwargs):
        del types
        if func is np.einsum:
            return _residual_ad_einsum(*args, **kwargs)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, _ResidualAD):
            return _ResidualAD(
                self.value + other.value,
                (
                    (self, lambda grad: _sum_to_shape(grad, self.shape)),
                    (other, lambda grad: _sum_to_shape(grad, other.shape)),
                ),
            )
        return _ResidualAD(
            self.value + other,
            ((self, lambda grad: _sum_to_shape(grad, self.shape)),),
        )

    __radd__ = __add__

    def __neg__(self):
        return _ResidualAD(-self.value, ((self, lambda grad: -grad),))

    def __sub__(self, other):
        return self + (-other if isinstance(other, _ResidualAD) else -np.asarray(other))

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, _ResidualAD):
            return _ResidualAD(
                self.value * other.value,
                (
                    (
                        self,
                        lambda grad: _sum_to_shape(
                            grad * other.value, self.shape
                        ),
                    ),
                    (
                        other,
                        lambda grad: _sum_to_shape(
                            grad * self.value, other.shape
                        ),
                    ),
                ),
            )
        return _ResidualAD(
            self.value * other,
            (
                (
                    self,
                    lambda grad: _sum_to_shape(grad * other, self.shape),
                ),
            ),
        )

    __rmul__ = __mul__


def _residual_ad_einsum(subscripts, *operands, **kwargs):
    """Evaluate an einsum and retain tensor-level reverse contractions."""
    if not any(isinstance(item, _ResidualAD) for item in operands):
        return np.einsum(subscripts, *operands, **kwargs)
    if "->" not in subscripts or "..." in subscripts:
        raise ValueError("CCSD residual AD requires explicit einsum indices")

    lhs, output_labels = subscripts.split("->")
    input_labels = lhs.split(",")
    values = [
        item.value if isinstance(item, _ResidualAD) else np.asarray(item)
        for item in operands
    ]
    value = np.einsum(subscripts, *values, **kwargs)
    parents = []
    for target, operand in enumerate(operands):
        if not isinstance(operand, _ResidualAD):
            continue
        other_labels = [
            labels for index, labels in enumerate(input_labels) if index != target
        ]
        other_values = [
            array for index, array in enumerate(values) if index != target
        ]
        reverse_spec = ",".join([output_labels, *other_labels])
        reverse_spec += "->" + input_labels[target]

        def reverse(grad, spec=reverse_spec, arrays=tuple(other_values)):
            return np.einsum(spec, grad, *arrays, optimize=True)

        parents.append((operand, reverse))
    return _ResidualAD(value, parents)


def _residual_ad_reverse(outputs, seeds):
    ordered = []
    seen = set()

    def visit(node):
        key = id(node)
        if key in seen:
            return
        seen.add(key)
        for parent, _ in node.parents:
            visit(parent)
        ordered.append(node)

    for output in outputs:
        visit(output)
    for output, seed in zip(outputs, seeds, strict=True):
        seed = np.asarray(seed, dtype=float)
        output.gradient = (
            seed
            if output.gradient is None
            else output.gradient + seed
        )
    for node in reversed(ordered):
        if node.gradient is None:
            continue
        for parent, reverse in node.parents:
            contribution = reverse(node.gradient)
            parent.gradient = (
                contribution
                if parent.gradient is None
                else parent.gradient + contribution
            )


def cs_ccsd_residual(
    t1, t2, f_oo, f_vv, f_ov, V, *,
    include_ladder=True, include_ring=True, return_ring=False,
):
    """True closed-shell CCSD residuals (R1, R2) at amplitudes (t1, t2).

    Direct transcription of cpp/src/ccsd.cpp::compute_residuals. Integral
    blocks in ``V`` are chemist-notation (pq|rs); the Fock diagonal is
    inside F_ae/F_mi so R vanishes at the CCSD fixed point.

    ``include_ladder=False`` omits the particle-particle (W_abef) ladder term
    ``R2_ij^ab += sum_ef tau_ij^ef W_abef`` and everything built only for it.
    That term is the whole O(nv^4) cost of this routine and ``V["vvvv"]`` is
    used nowhere else, so the caller may also omit that block entirely.

    The DLPNO extended-domain path (issue #700) uses this: the ladder's
    contracted indices e,f are carried by ``tau_ij``, which lives in pair ij's
    own PNO space, and its free indices a,b are projected into that same space
    afterwards, so the term is computable there at ``n_pno^4`` instead of the
    extended basis' ``n_ext^4`` -- exactly, not approximately (validated to
    5e-16 relative). This is unlike the ring terms, whose contracted indices
    belong to the *source* pair and must not be projected first; doing that is
    the truncation issue #98 was about.
    """
    ovov = V["ovov"]; ovvv = V["ovvv"]; ooov = V["ooov"]
    oooo = V["oooo"]; oovv = V["oovv"]
    # Only the ladder consumes (ae|bf); let the caller omit the block.
    vvvv = V["vvvv"] if include_ladder else None

    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    taut = t2 + 0.5 * np.einsum("ia,jb->ijab", t1, t1)

    # ---- one-body intermediates ----
    # F_me = f_me + S_nf t_nf [2(me|nf) - (mf|ne)]
    F_me = f_ov + (
        2.0 * np.einsum("nf,menf->me", t1, ovov)
        - np.einsum("nf,mfne->me", t1, ovov)
    )
    # F_ae = f_ae - 1/2 S_m f_me t_ma + S_mf t_mf [2(mf|ae) - (me|af)]
    #            - S_mnf taut_mn^af [2(me|nf) - (mf|ne)]
    F_ae = (
        f_vv
        - 0.5 * np.einsum("me,ma->ae", f_ov, t1)
        + 2.0 * np.einsum("mf,mfae->ae", t1, ovvv)
        - np.einsum("mf,meaf->ae", t1, ovvv)
        - 2.0 * np.einsum("mnaf,menf->ae", taut, ovov)
        + np.einsum("mnaf,mfne->ae", taut, ovov)
    )
    # F_mi = f_mi + 1/2 S_e t_ie f_me + S_ne t_ne [2(mi|ne) - (ni|me)]
    #            + S_nef taut_in^ef [2(me|nf) - (mf|ne)]
    F_mi = (
        f_oo
        + 0.5 * np.einsum("ie,me->mi", t1, f_ov)
        + 2.0 * np.einsum("ne,mine->mi", t1, ooov)
        - np.einsum("ne,nime->mi", t1, ooov)
        + 2.0 * np.einsum("inef,menf->mi", taut, ovov)
        - np.einsum("inef,mfne->mi", taut, ovov)
    )

    # ---- T1 residual ----
    R1 = f_ov.copy()
    R1 += np.einsum("ie,ae->ia", t1, F_ae)
    R1 -= np.einsum("mi,ma->ia", F_mi, t1)
    R1 += np.einsum("imae,me->ia", 2.0 * t2 - t2.transpose(1, 0, 2, 3), F_me)
    # + S_nf t_nf [2(nf|ia) - (ni|af)]
    R1 += 2.0 * np.einsum("nf,nfia->ia", t1, ovov)
    R1 -= np.einsum("nf,niaf->ia", t1, oovv)
    # + S_mef [2 t_im^ef - t_mi^ef] (mf|ae)
    R1 += np.einsum("imef,mfae->ia", 2.0 * t2 - t2.transpose(1, 0, 2, 3), ovvv)
    # - S_mne [2 t_mn^ae - t_nm^ae] (mi|ne)
    R1 -= np.einsum("mnae,mine->ia", 2.0 * t2 - t2.transpose(1, 0, 2, 3), ooov)

    # ---- two-body intermediates ----
    # W_mnij = (mi|nj) + S_e t_je (mi|ne) + S_e t_ie (nj|me) + 1/2 S_ef tau_ij^ef (me|nf)
    Wmnij = (
        oooo.transpose(0, 2, 1, 3).copy()           # (mi|nj) -> [m,n,i,j]
        + np.einsum("je,mine->mnij", t1, ooov)
        + np.einsum("ie,njme->mnij", t1, ooov)
        + 0.5 * np.einsum("ijef,menf->mnij", tau, ovov)
    )
    # W_abef = (ae|bf) - S_m t_mb (mf|ae) - S_m t_ma (me|bf) + 1/2 S_mn tau_mn^ab (me|nf)
    Wabef = (
        (
            vvvv.transpose(0, 2, 1, 3).copy()       # (ae|bf) -> [a,b,e,f]
            - np.einsum("mb,mfae->abef", t1, ovvv)
            - np.einsum("ma,mebf->abef", t1, ovvv)
            + 0.5 * np.einsum("mnab,menf->abef", tau, ovov)
        )
        if include_ladder
        else None
    )

    # Ring intermediates W1, W2, WX, indexed [m,e,j,b]. Built only when the
    # ring terms are contracted here or handed back for a PNO-space
    # contraction (#700): they have no other consumer, so skipping them
    # removes their O(o^2 n_ext^2) storage as well as their build.
    _need_ring = include_ring or return_ring
    t_njfb = t2.transpose(0, 1, 2, 3)               # [n,j,f,b]
    # build via einsum, mirroring the C++ scalar loops
    if _need_ring:
        W1 = (
            ovov.transpose(0, 1, 2, 3).copy()           # (me|jb) -> [m,e,j,b]
            + np.einsum("jf,mebf->mejb", t1, ovvv)
            - np.einsum("nb,njme->mejb", t1, ooov)
            + np.einsum("njfb,menf->mejb",
                        t2.transpose(0, 1, 2, 3), ovov)            # t_nj^fb (me|nf)
            - 0.5 * np.einsum("jnfb,menf->mejb", t2, ovov)         # -1/2 t_jn^fb (me|nf)
            - np.einsum("jf,nb,menf->mejb", t1, t1, ovov)          # -t_jf t_nb (me|nf)
            - 0.5 * np.einsum("njfb,mfne->mejb",
                              t2.transpose(0, 1, 2, 3), ovov)      # -1/2 t_nj^fb (mf|ne)
        )
        W2 = (
            ovov.copy()                                  # (me|jb)
            - oovv.transpose(0, 3, 1, 2).copy()          # -(mj|be) -> [m,e,j,b]
            + np.einsum("jf,mebf->mejb", t1, ovvv)
            - np.einsum("jf,mfbe->mejb", t1, ovvv)
            - np.einsum("nb,njme->mejb", t1, ooov)
            + np.einsum("nb,mjne->mejb", t1, ooov)
        )
        # the t-quadratic ring parts of W2 and WX
        tss = 0.5 * (t2 - t2.transpose(0, 1, 3, 2)) + np.einsum("jf,nb->jnfb", t1, t1)
        W2 += (
            -np.einsum("jnfb,menf->mejb", tss, ovov)
            + np.einsum("jnfb,mfne->mejb", tss, ovov)
            + 0.5 * np.einsum("njfb,menf->mejb", t2.transpose(0, 1, 2, 3), ovov)
        )
        WX = (
            -oovv.transpose(0, 3, 1, 2).copy()           # -(mj|be)
            - np.einsum("jf,mfbe->mejb", t1, ovvv)
            + np.einsum("nb,mjne->mejb", t1, ooov)
            + 0.5 * np.einsum("jnfb,mfne->mejb", t2, ovov)
            + np.einsum("jf,nb,mfne->mejb", t1, t1, ovov)
        )
    else:
        W1 = W2 = WX = None

    # ---- T2 residual ----
    Fh_be = F_ae - 0.5 * np.einsum("mb,me->be", t1, F_me)
    Fh_mj = F_mi + 0.5 * np.einsum("je,me->mj", t1, F_me)

    R2 = ovov.transpose(0, 2, 1, 3).copy()           # (ia|jb) -> [i,j,a,b]
    R2 += np.einsum("mnij,mnab->ijab", Wmnij, tau)
    if include_ladder:
        R2 += np.einsum("abef,ijef->ijab", Wabef, tau)

    # The two branches are spelled out rather than assembled from a common
    # part plus a conditional one: floating-point summation is not
    # associative, so folding the ring terms in separately would perturb the
    # default route's last bits and break the exact-equality assertions the
    # #700 gate relies on.
    if include_ring:
        half = (
            np.einsum("ijae,be->ijab", t2, Fh_be)
            - np.einsum("imab,mj->ijab", t2, Fh_mj)
            + np.einsum("imae,mejb->ijab", t2 - t2.transpose(1, 0, 2, 3), W1)
            + np.einsum("imae,mejb->ijab", t2, W2)
            + np.einsum("mjae,meib->ijab", t2, WX)
            - np.einsum("ie,ma,mejb->ijab", t1, t1, ovov)
            - np.einsum("je,ma,mibe->ijab", t1, t1, oovv)
            + np.einsum("ie,jbae->ijab", t1, ovvv)   # S_e t_ie (ae|jb)=(jb|ae)
            - np.einsum("ma,mijb->ijab", t1, ooov)   # S_m t_ma (mi|jb)
        )
    else:
        # #700: the three ring terms are contracted in each source pair's own
        # PNO space by the caller instead. Everything else is unchanged.
        half = (
            np.einsum("ijae,be->ijab", t2, Fh_be)
            - np.einsum("imab,mj->ijab", t2, Fh_mj)
            - np.einsum("ie,ma,mejb->ijab", t1, t1, ovov)
            - np.einsum("je,ma,mibe->ijab", t1, t1, oovv)
            + np.einsum("ie,jbae->ijab", t1, ovvv)   # S_e t_ie (ae|jb)=(jb|ae)
            - np.einsum("ma,mijb->ijab", t1, ooov)   # S_m t_ma (mi|jb)
        )
    R2 += half + half.transpose(1, 0, 3, 2)
    if return_ring:
        return R1, R2, (W1, W2, WX)
    return R1, R2


def cs_ccsd_pair_ladder(
    t2_ij, t1L, tau_L, B_ov_L, B_vv_L, u_i, v_j, w_i, w_j
):
    """Particle-particle ladder for ONE pair, in that pair's own PNO space.

    Returns ``R2_ab = sum_ef tau_ij^ef W_abef`` with

        W_abef = (ae|bf) - sum_m t_m^b (mf|ae) - sum_m t_m^a (me|bf)
                         + 1/2 sum_mn tau_mn^ab (me|nf)

    (issue #700; Riplinger and Neese 2013 Sec. II C contract every term in the
    source pair's PNO space rather than a common large basis).

    The FREE indices a,b are projected into this pair's space no matter what,
    so every quantity carrying them -- ``t1L``, ``tau_L``, ``B_*_L`` -- may be
    projected here exactly. The CONTRACTED indices e,f are the delicate part:

    * ``tau_ij = t2_ij + t1_i x t1_j`` and only the ``t2_ij`` piece lives in
      this pair's PNO space. The singles are expanded in the DIAGONAL pairs'
      PNOs, so ``t1_i x t1_j`` has support outside it and projecting it here
      first loses that -- worth ~1e-7 Ha on a water dimer, far above the
      block-by-block tolerance this route is held to.
    * That piece is rank one in e and f, so it is contracted exactly in the
      mixed basis instead, through four precomputed density-fitting vectors:

          u_i[P,a] = sum_v (a v|P) t1_i[v]      v_j[P,b] = sum_v (b v|P) t1_j[v]
          w_i[P,m] = sum_v (m v|P) t1_i[v]      w_j[P,m] = sum_v (m v|P) t1_j[v]

      with v running over the FULL virtual space, so nothing is truncated.

    W_abef is never formed: folding the contracted amplitude into each term
    through the fitting index first makes this ``O(naux n^3)`` rather than the
    ``O(n^4)`` of building the intermediate, which is the ladder's whole cost.

    Parameters
    ----------
    t2_ij : (n, n)          target pair's doubles, in its own PNO space.
    t1L : (nL, n)           coupled singles, projected into this pair's space.
    tau_L : (nL, nL, n, n)  coupled tau^mn, projected into this pair's space.
    B_ov_L : (naux, nL, n), B_vv_L : (naux, n, n)   pair-space DF factors.
    u_i, v_j : (naux, n)    mixed-basis singles vectors defined above.
    w_i, w_j : (naux, nL)   likewise, occupied side.
    """

    # ---- t2_ij piece: entirely inside this pair's PNO space ----
    G = np.einsum("Pae,ef->Paf", B_vv_L, t2_ij, optimize=True)
    R = np.einsum("Paf,Pbf->ab", G, B_vv_L, optimize=True)          # (ae|bf)

    X = np.einsum("Pmf,Paf->ma", B_ov_L, G, optimize=True)
    R -= np.einsum("ma,mb->ab", X, t1L, optimize=True)              # -t_m^b (mf|ae)

    H = np.einsum("Pbf,ef->Pbe", B_vv_L, t2_ij, optimize=True)
    Y = np.einsum("Pme,Pbe->mb", B_ov_L, H, optimize=True)
    R -= np.einsum("ma,mb->ab", t1L, Y, optimize=True)              # -t_m^a (me|bf)

    Q = np.einsum("Pme,ef->Pmf", B_ov_L, t2_ij, optimize=True)
    Z = np.einsum("Pmf,Pnf->mn", Q, B_ov_L, optimize=True)
    R += 0.5 * np.einsum("mn,mnab->ab", Z, tau_L, optimize=True)    # +1/2 tau_mn^ab (me|nf)

    # ---- t1_i x t1_j piece: rank one, contracted in the mixed basis ----
    R += np.einsum("Pa,Pb->ab", u_i, v_j, optimize=True)            # (ae|bf)
    R -= np.einsum("Pm,Pa,mb->ab", w_j, u_i, t1L, optimize=True)    # -t_m^b (mf|ae)
    R -= np.einsum("Pm,Pb,ma->ab", w_i, v_j, t1L, optimize=True)    # -t_m^a (me|bf)
    Z1 = np.einsum("Pm,Pn->mn", w_i, w_j, optimize=True)
    R += 0.5 * np.einsum("mn,mnab->ab", Z1, tau_L, optimize=True)   # +1/2 tau_mn^ab (me|nf)
    return R


def run_cs_ccsd(
    f_mo, B_ov, B_vv, B_oo, n_occ, e_hf=0.0, *,
    max_iter=100, conv_tol=1e-10, conv_tol_residual=1e-7, diis_size=6,
) -> CSCCSDResult:
    """Spatial closed-shell DF-CCSD on a converged RHF reference."""
    return run_cs_ccsd_blocks(
        f_mo,
        _blocks(B_ov, B_vv, B_oo),
        n_occ,
        e_hf=e_hf,
        max_iter=max_iter,
        conv_tol=conv_tol,
        conv_tol_residual=conv_tol_residual,
        diis_size=diis_size,
    )


def _reshape_4d_to_flat(arr, a, b, c, d):
    """Reshape a 4D numpy array to a 2D Fortran-order flat matrix."""
    import numpy as np

    return np.asfortranarray(np.asarray(arr).reshape(a * b, c * d))


def _run_cs_ccsd_blocks_cpp(
    f_mo, V, n_occ, e_hf=0.0, *,
    max_iter=100, conv_tol=1e-10, conv_tol_residual=1e-7, diis_size=6,
) -> CSCCSDResult | None:
    """Run CCSD through the C++ spatial solver (when the native binding
    is available).  Returns None if the native module cannot be loaded.
    """
    try:
        from .._vibeqc_core import spatial_ccsd_solve
    except ImportError:
        return None

    import numpy as np

    no = n_occ
    nv = V["vvvv"].shape[0]
    f_oo = np.asfortranarray(f_mo[:no, :no])
    f_vv = np.asfortranarray(f_mo[no:, no:])
    f_ov = np.asfortranarray(f_mo[:no, no:])

    ovov_f = _reshape_4d_to_flat(V["ovov"], no, nv, no, nv)
    oooo_f = _reshape_4d_to_flat(V["oooo"], no, no, no, no)
    ooov_f = _reshape_4d_to_flat(V["ooov"], no, no, no, nv)
    oovv_f = _reshape_4d_to_flat(V["oovv"], no, no, nv, nv)
    ovvv_f = _reshape_4d_to_flat(V["ovvv"], no, nv, nv, nv)
    vvvv_f = _reshape_4d_to_flat(V["vvvv"], nv, nv, nv, nv)

    T1, T2_flat, e_corr, n_iter, converged = spatial_ccsd_solve(
        f_oo, f_vv, f_ov,
        ovov_f, oooo_f, ooov_f, oovv_f, ovvv_f, vvvv_f,
        float(e_hf),
        int(max_iter), float(conv_tol), float(conv_tol_residual),
        int(diis_size),
    )

    result = CSCCSDResult(e_hf=float(e_hf))
    result.e_corr = float(e_corr)
    result.e_total = float(e_hf) + float(e_corr)
    result.n_iter = int(n_iter)
    result.converged = bool(converged)
    result.t1 = np.asarray(T1)
    result.t2 = np.asarray(T2_flat).reshape(no, no, nv, nv)
    return result


def run_cs_ccsd_blocks(
    f_mo, V, n_occ, e_hf=0.0, *,
    max_iter=100, conv_tol=1e-10, conv_tol_residual=1e-7, diis_size=6,
) -> CSCCSDResult:
    """Spatial closed-shell CCSD from prebuilt chemist-notation MO blocks.

    Uses the native C++ solver (spatial_ccsd_solve) when available;
    falls back to the pure-Python numpy transcription otherwise.
    """
    cpp_result = _run_cs_ccsd_blocks_cpp(
        f_mo, V, n_occ, e_hf=e_hf,
        max_iter=max_iter, conv_tol=conv_tol, conv_tol_residual=conv_tol_residual,
        diis_size=diis_size,
    )
    if cpp_result is not None:
        return cpp_result

    # ---- Pure-Python fallback ----
    no = n_occ
    nv = V["vvvv"].shape[0]
    f_oo = f_mo[:no, :no]; f_vv = f_mo[no:, no:]; f_ov = f_mo[:no, no:]
    eps_o = np.diag(f_oo); eps_v = np.diag(f_vv)
    Dia = eps_o[:, None] - eps_v[None, :]
    Dijab = (
        eps_o[:, None, None, None] + eps_o[None, :, None, None]
        - eps_v[None, None, :, None] - eps_v[None, None, None, :]
    )
    ovov = V["ovov"]
    Loovv = 2.0 * ovov.transpose(0, 2, 1, 3) - ovov.transpose(0, 2, 3, 1)

    t1 = f_ov / Dia
    t2 = ovov.transpose(0, 2, 1, 3) / Dijab

    def energy(t1, t2):
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
        return float(2.0 * np.einsum("ia,ia->", f_ov, t1)
                     + np.einsum("ijab,ijab->", tau, Loovv))

    amp_hist: list = []; res_hist: list = []
    result = CSCCSDResult(e_hf=float(e_hf))
    e_prev = energy(t1, t2)
    for it in range(max_iter):
        R1, R2 = cs_ccsd_residual(t1, t2, f_oo, f_vv, f_ov, V)
        t1n = t1 + R1 / Dia
        t2n = t2 + R2 / Dijab

        amp_hist.append(np.concatenate([t1n.ravel(), t2n.ravel()]))
        res_hist.append(np.concatenate([(t1n - t1).ravel(), (t2n - t2).ravel()]))
        if len(amp_hist) > diis_size:
            amp_hist.pop(0); res_hist.pop(0)
        nh = len(amp_hist)
        if nh >= 2:
            Bm = np.empty((nh + 1, nh + 1)); Bm[-1, :] = -1.0
            Bm[:, -1] = -1.0; Bm[-1, -1] = 0.0
            for a_ in range(nh):
                for b_ in range(nh):
                    Bm[a_, b_] = float(np.dot(res_hist[a_], res_hist[b_]))
            rhs = np.zeros(nh + 1); rhs[-1] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)[:nh]
                flat = sum(c[k] * amp_hist[k] for k in range(nh))
                t1n = flat[: t1.size].reshape(t1.shape)
                t2n = flat[t1.size:].reshape(t2.shape)
            except np.linalg.LinAlgError:
                pass

        t1, t2 = t1n, t2n
        e = energy(t1, t2)
        result.trace.append(
            {
                "iter": it + 1,
                "e_corr": e,
                "delta_e": e - e_prev,
                "r1_norm": float(np.linalg.norm(R1)),
                "r2_norm": float(np.linalg.norm(R2)),
                "diis_subspace": nh,
            }
        )
        if (
            abs(e - e_prev) < conv_tol
            and float(np.linalg.norm(R1) + np.linalg.norm(R2))
            < conv_tol_residual
        ):
            result.converged = True
            result.n_iter = it + 1
            break
        e_prev = e

    if result.n_iter == 0:
        result.n_iter = max_iter
    result.e_corr = energy(t1, t2)
    result.e_total = result.e_hf + result.e_corr
    result.t1, result.t2 = t1, t2
    return result


def cs_ccsd_energy(t1, t2, f_ov, V) -> float:
    """Closed-shell CCSD correlation-energy functional for this residual."""
    ovov = V["ovov"]
    Loovv = 2.0 * ovov.transpose(0, 2, 1, 3) - ovov.transpose(0, 2, 3, 1)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    return float(
        2.0 * np.einsum("ia,ia->", f_ov, t1)
        + np.einsum("ijab,ijab->", tau, Loovv)
    )


def cs_ccsd_energy_gradient(t1, t2, f_ov, V) -> tuple[np.ndarray, np.ndarray]:
    """Analytic gradient of :func:`cs_ccsd_energy` with respect to T1/T2."""
    del t2  # The doubles gradient is independent of the current T2 amplitudes.
    ovov = V["ovov"]
    Loovv = 2.0 * ovov.transpose(0, 2, 1, 3) - ovov.transpose(0, 2, 3, 1)
    g1 = (
        2.0 * f_ov
        + np.einsum("kjcb,jb->kc", Loovv, t1, optimize=True)
        + np.einsum("ikac,ia->kc", Loovv, t1, optimize=True)
    )
    return g1, Loovv.copy()


def cs_ccsd_residual_vjp(
    t1,
    t2,
    seed1,
    seed2,
    f_oo,
    f_vv,
    f_ov,
    V,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``(dR/dT)^T`` without forming the CCSD Jacobian.

    Reverse differentiation is performed over the tensor contractions in
    :func:`cs_ccsd_residual`. The cost and storage therefore track a small
    multiple of one residual evaluation instead of scaling with the number of
    amplitudes as the dense finite-difference oracle does.
    """
    ad_t1 = _ResidualAD(t1)
    ad_t2 = _ResidualAD(t2)
    r1, r2 = cs_ccsd_residual(ad_t1, ad_t2, f_oo, f_vv, f_ov, V)
    _residual_ad_reverse((r1, r2), (seed1, seed2))
    return ad_t1.gradient, ad_t2.gradient


def run_cs_ccsd_lambda_iterative(
    t1,
    t2,
    f_oo,
    f_vv,
    f_ov,
    V,
    *,
    max_iter: int = 100,
    residual_tol: float = 1e-8,
    diis_size: int = 6,
) -> CSCCSDLambdaResult:
    """Solve the spatial CCSD Lambda equations with analytic VJP products."""
    if max_iter < 1:
        raise ValueError("run_cs_ccsd_lambda_iterative: max_iter must be positive")
    if residual_tol <= 0.0:
        raise ValueError(
            "run_cs_ccsd_lambda_iterative: residual_tol must be positive"
        )
    if diis_size < 0:
        raise ValueError("run_cs_ccsd_lambda_iterative: diis_size must be >= 0")

    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    eps_o = np.diag(f_oo)
    eps_v = np.diag(f_vv)
    d1 = eps_o[:, None] - eps_v[None, :]
    d2 = (
        eps_o[:, None, None, None]
        + eps_o[None, :, None, None]
        - eps_v[None, None, :, None]
        - eps_v[None, None, None, :]
    )
    if np.any(np.abs(d1) < 1e-14) or np.any(np.abs(d2) < 1e-14):
        raise np.linalg.LinAlgError(
            "run_cs_ccsd_lambda_iterative: singular orbital-energy denominator"
        )

    gradient1, gradient2 = cs_ccsd_energy_gradient(t1, t2, f_ov, V)
    # The closed-shell residual is evaluated over the pair-exchange-symmetric
    # T2 manifold. Its Lagrange multiplier is the spin-adapted combination
    # M2 = 2 L2 - L2(ij,ba), while M1 = 2 L1. Solve for M, then invert this
    # metric before returning the physical Lambda amplitudes used by (AT).
    multiplier1 = 2.0 * t1
    multiplier2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    amplitude_history = []
    residual_history = []
    residual_norm = float("inf")

    for iteration in range(1, max_iter + 1):
        sigma1, sigma2 = cs_ccsd_residual_vjp(
            t1, t2, multiplier1, multiplier2, f_oo, f_vv, f_ov, V
        )
        residual1 = gradient1 + sigma1
        residual2 = gradient2 + sigma2
        residual2 = 0.5 * (
            residual2 + residual2.transpose(1, 0, 3, 2)
        )
        residual_norm = float(
            np.linalg.norm(residual1) + np.linalg.norm(residual2)
        )
        if residual_norm < residual_tol:
            l1 = 0.5 * multiplier1
            l2 = (
                2.0 * multiplier2
                + multiplier2.transpose(0, 1, 3, 2)
            ) / 3.0
            return CSCCSDLambdaResult(
                l1=l1,
                l2=l2,
                residual_norm=residual_norm,
                condition_number=float("nan"),
                n_amplitudes=int(l1.size + l2.size),
                n_iter=iteration,
                converged=True,
            )

        step1 = residual1 / d1
        step2 = residual2 / d2
        multiplier1_new = multiplier1 + step1
        multiplier2_new = multiplier2 + step2
        multiplier2_new = 0.5 * (
            multiplier2_new
            + multiplier2_new.transpose(1, 0, 3, 2)
        )

        amplitude_history.append(
            _flatten_amplitudes(multiplier1_new, multiplier2_new)
        )
        residual_history.append(_flatten_amplitudes(step1, step2))
        if len(amplitude_history) > diis_size:
            amplitude_history.pop(0)
            residual_history.pop(0)
        n_history = len(amplitude_history)
        if diis_size > 0 and n_history >= 2:
            b_matrix = np.empty((n_history + 1, n_history + 1))
            b_matrix[-1, :] = -1.0
            b_matrix[:, -1] = -1.0
            b_matrix[-1, -1] = 0.0
            for row in range(n_history):
                for column in range(n_history):
                    b_matrix[row, column] = float(
                        np.dot(residual_history[row], residual_history[column])
                    )
            rhs = np.zeros(n_history + 1)
            rhs[-1] = -1.0
            try:
                coefficients = np.linalg.solve(b_matrix, rhs)[:n_history]
            except np.linalg.LinAlgError:
                pass
            else:
                flat = sum(
                    coefficients[index] * amplitude_history[index]
                    for index in range(n_history)
                )
                multiplier1_new, multiplier2_new = _unflatten_amplitudes(
                    flat, multiplier1.shape, multiplier2.shape
                )

        multiplier1, multiplier2 = multiplier1_new, multiplier2_new

    sigma1, sigma2 = cs_ccsd_residual_vjp(
        t1, t2, multiplier1, multiplier2, f_oo, f_vv, f_ov, V
    )
    projected_residual2 = gradient2 + sigma2
    projected_residual2 = 0.5 * (
        projected_residual2
        + projected_residual2.transpose(1, 0, 3, 2)
    )
    residual_norm = float(
        np.linalg.norm(gradient1 + sigma1)
        + np.linalg.norm(projected_residual2)
    )
    l1 = 0.5 * multiplier1
    l2 = (
        2.0 * multiplier2 + multiplier2.transpose(0, 1, 3, 2)
    ) / 3.0
    return CSCCSDLambdaResult(
        l1=l1,
        l2=l2,
        residual_norm=residual_norm,
        condition_number=float("nan"),
        n_amplitudes=int(l1.size + l2.size),
        n_iter=max_iter,
        converged=False,
    )


def _flatten_amplitudes(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(t1).ravel(), np.asarray(t2).ravel()])


def _unflatten_amplitudes(
    flat: np.ndarray,
    t1_shape: tuple[int, ...],
    t2_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    n1 = int(np.prod(t1_shape))
    t1 = flat[:n1].reshape(t1_shape)
    t2 = flat[n1:].reshape(t2_shape)
    return t1, t2


def _residual_flat_from_vector(
    flat: np.ndarray,
    t1_shape: tuple[int, ...],
    t2_shape: tuple[int, ...],
    f_oo: np.ndarray,
    f_vv: np.ndarray,
    f_ov: np.ndarray,
    V: dict,
) -> np.ndarray:
    t1, t2 = _unflatten_amplitudes(flat, t1_shape, t2_shape)
    r1, r2 = cs_ccsd_residual(t1, t2, f_oo, f_vv, f_ov, V)
    return _flatten_amplitudes(r1, r2)


def cs_ccsd_residual_jacobian(
    t1,
    t2,
    f_oo,
    f_vv,
    f_ov,
    V,
    *,
    step: float = 1e-5,
) -> np.ndarray:
    """Dense finite-difference Jacobian dR/dT for the spatial CCSD residual.

    This is the small-system reference backend for the Lambda-equation
    prerequisite.  It is intentionally explicit: the production A-CCSD(T)
    route can later replace the matrix build with analytic sigma-vector
    products while keeping these tests as the oracle.
    """
    if step <= 0.0:
        raise ValueError("cs_ccsd_residual_jacobian: step must be positive")
    x0 = _flatten_amplitudes(t1, t2)
    jac = np.empty((x0.size, x0.size), dtype=float)
    t1_shape, t2_shape = np.shape(t1), np.shape(t2)
    for p in range(x0.size):
        dx = np.zeros_like(x0)
        dx[p] = step
        rp = _residual_flat_from_vector(
            x0 + dx, t1_shape, t2_shape, f_oo, f_vv, f_ov, V
        )
        rm = _residual_flat_from_vector(
            x0 - dx, t1_shape, t2_shape, f_oo, f_vv, f_ov, V
        )
        jac[:, p] = (rp - rm) / (2.0 * step)
    return jac


def run_cs_ccsd_lambda(
    t1,
    t2,
    f_oo,
    f_vv,
    f_ov,
    V,
    *,
    step: float = 1e-5,
    residual_tol: float = 1e-8,
    rcond: float = 1e-12,
) -> CSCCSDLambdaResult:
    """Solve the closed-shell CCSD Lambda equations for the spatial oracle.

    The Lagrangian is ``E(T) + lambda.R(T)``. At converged CCSD amplitudes,
    stationarity with respect to the right amplitudes gives

        (dR/dT)^T lambda = -dE/dT.

    The returned ``l1``/``l2`` use the same shapes and conventions as
    ``t1``/``t2``.  This routine is a correctness oracle, not the final
    large-system sigma-vector implementation.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    g1, g2 = cs_ccsd_energy_gradient(t1, t2, f_ov, V)
    gradient = _flatten_amplitudes(g1, g2)
    jac = cs_ccsd_residual_jacobian(t1, t2, f_oo, f_vv, f_ov, V, step=step)
    lhs = jac.T
    rhs = -gradient
    try:
        lam = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(lhs, rhs, rcond=rcond)[0]
    residual = lhs @ lam - rhs
    l1, l2 = _unflatten_amplitudes(lam, t1.shape, t2.shape)
    return CSCCSDLambdaResult(
        l1=l1,
        l2=l2,
        residual_norm=float(np.linalg.norm(residual)),
        condition_number=float(np.linalg.cond(lhs)),
        n_amplitudes=int(lam.size),
        converged=bool(np.linalg.norm(residual) < residual_tol),
    )


# =========================================================================
# Spatial closed-shell (T) correction (Raghavachari 1989), classwise spin
# integration. A faithful numpy transcription of the validated C++ kernel
# `cpp/src/ccsd.cpp::triples::compute_triples` (the (T) analogue of
# cs_ccsd_residual <- cpp/src/ccsd.cpp::compute_residuals).
#
# Spin-orbital (validated reference, `_ccsd_ref.so_triples_correction`):
#   G(p1p2p3;q1q2q3) = S_e t_{p2p3}^{q1 e} <e p1||q2 q3>
#                    - S_m t_{p1 m}^{q2 q3} <m q1||p2 p3>
#   W  = P(i/jk) P(a/bc) G                 (connected)
#   Wd = P(i/jk) P(a/bc) t_{p1}^{q1} <p2 p3||q2 q3>   (disconnected)
#   E(T) = (1/36) S_patterns S_{ijkabc} W (W + Wd) / D
# For a closed-shell reference the pattern sum collapses to two classes:
#   E(T) = (1/18) S_aaa + (1/2) S_aab
# with external spins (aaa;aaa) and (aab;aab). Each spin-orbital block
# reduces to spatial integrals (chemist notation): ovvv=(ia|bc),
# ooov=(ij|ka), ovov=(ia|jb); 2x the dimensions / 8x the spin-orbital
# triples are gone. Assumes canonical orbitals in the denominators.
# =========================================================================

# P(i/jk) images and signs, applied to both occupied and virtual sides.
_T_PERMS = (((0, 1, 2), 1.0), ((1, 0, 2), -1.0), ((2, 1, 0), -1.0))
# The two closed-shell spin classes: ((occ spins, vir spins), weight).
_T_CLASSES = (((0, 0, 0), (0, 0, 0), 1.0 / 18.0), ((0, 0, 1), (0, 0, 1), 0.5))


def _cs_triple_buffers(
    i, j, k, t1, t2, ovvv, ooov, ovov, so_pat, sv_pat, f_ov=None
):
    """Connected/disconnected closed-shell triples buffers for one spin class."""
    occ = (i, j, k)
    nv = t1.shape[1]
    wc = np.zeros((nv, nv, nv))
    wd = np.zeros((nv, nv, nv))
    for po, sgn_o in _T_PERMS:
        for pv, sgn_v in _T_PERMS:
            sign = sgn_o * sgn_v
            p1, p2, p3 = occ[po[0]], occ[po[1]], occ[po[2]]
            s1, s2, s3 = so_pat[po[0]], so_pat[po[1]], so_pat[po[2]]
            z1, z2, z3 = sv_pat[pv[0]], sv_pat[pv[1]], sv_pat[pv[2]]

            # ---- connected G image (term A - term B) ----
            G = np.zeros((nv, nv, nv))  # axes (q1, q2, q3)
            # term A: S_e t_{p2p3}^{q1 e} <e p1||q2 q3>
            for se in (0, 1):
                t_dir = s2 == z1 and s3 == se
                t_swp = s2 == se and s3 == z1
                v_cou = se == z2 and s1 == z3
                v_exc = se == z3 and s1 == z2
                if not (t_dir or t_swp) or not (v_cou or v_exc):
                    continue
                tv = np.zeros((nv, nv))            # (q1, e)
                if t_dir:
                    tv = tv + t2[p2, p3]
                if t_swp:
                    tv = tv - t2[p2, p3].T
                vv = np.zeros((nv, nv, nv))        # (e, q2, q3)
                if v_cou:
                    vv = vv + ovvv[p1].transpose(1, 2, 0)  # (p1 q3|e q2)
                if v_exc:
                    vv = vv - ovvv[p1].transpose(1, 0, 2)  # (p1 q2|e q3)
                G = G + np.einsum("qe,eyz->qyz", tv, vv, optimize=True)
            # term B: - S_m t_{p1 m}^{q2 q3} <m q1||p2 p3>
            for sm in (0, 1):
                t_dir = s1 == z2 and sm == z3
                t_swp = s1 == z3 and sm == z2
                v_cou = sm == s2 and z1 == s3
                v_exc = sm == s3 and z1 == s2
                if not (t_dir or t_swp) or not (v_cou or v_exc):
                    continue
                vvB = np.zeros((ooov.shape[0], nv))   # (m, q1)
                if v_cou:
                    vvB = vvB + ooov[:, p2, p3, :]    # (m p2|p3 q1)
                if v_exc:
                    vvB = vvB - ooov[:, p3, p2, :]    # (m p3|p2 q1)
                tvB = np.zeros((t2.shape[0], nv, nv))  # (m, q2, q3)
                if t_dir:
                    tvB = tvB + t2[p1]
                if t_swp:
                    tvB = tvB - t2[p1].transpose(0, 2, 1)
                G = G - np.einsum("mq,myz->qyz", vvB, tvB, optimize=True)
            wc += sign * np.moveaxis(G, [0, 1, 2], list(pv))

            # ---- disconnected image: t_{p1}^{q1} <p2 p3||q2 q3> ----
            if s1 != z1:
                continue
            v_cou = s2 == z2 and s3 == z3
            v_exc = s2 == z3 and s3 == z2
            if not (v_cou or v_exc):
                continue
            vvd = np.zeros((nv, nv))               # (q2, q3)
            base = ovov[p2, :, p3, :]              # (p2 q2|p3 q3)
            if v_cou:
                vvd = vvd + base
            if v_exc:
                vvd = vvd - base.T
            Dimg = np.einsum("q,yz->qyz", t1[p1], vvd, optimize=True)
            if f_ov is not None:
                tvd = np.zeros((nv, nv))
                if v_cou:
                    tvd = tvd + t2[p2, p3]
                if v_exc:
                    tvd = tvd - t2[p2, p3].T
                Dimg += np.einsum(
                    "q,yz->qyz", f_ov[p1], tvd, optimize=True
                )
            wd += sign * np.moveaxis(Dimg, [0, 1, 2], list(pv))
    return wc, wd


def cs_triple_energy(i, j, k, t1, t2, ovvv, ooov, ovov, eps_o, eps_v):
    """Spatial closed-shell (T) energy of one occupied triple (i, j, k).

    Sums the two spin classes over the 3x3 P(i/jk)xP(a/bc) permutation
    images, building the connected (wc) and disconnected (wd) rank-3
    (a,b,c) buffers, then contracts wc(wc+wd)/D over the (full or local)
    virtual space. Integral blocks are chemist-notation over whatever
    virtual space `ovvv`/`ooov`/`ovov` and `eps_v` span.
    """
    e_t = 0.0
    for so_pat, sv_pat, weight in _T_CLASSES:
        wc, wd = _cs_triple_buffers(
            i, j, k, t1, t2, ovvv, ooov, ovov, so_pat, sv_pat
        )
        d = (
            eps_o[i] + eps_o[j] + eps_o[k]
            - eps_v[:, None, None] - eps_v[None, :, None] - eps_v[None, None, :]
        )
        e_t += weight * float(np.sum(wc * (wc + wd) / d))
    return e_t


def cs_lambda_triple_energy(
    i, j, k, t1, t2, l1, l2, ovvv, ooov, ovov, eps_o, eps_v, f_ov=None
):
    """Closed-shell A-CCSD(T)/Lambda triples energy for one occupied triple.

    The right-hand triples moment is the usual connected CCSD(T) numerator
    ``W(T2)``.  The left-hand moment is built with the CCSD multipliers,
    ``W(L2) + Wd(L1)``. For canonical orbitals (``f_ov = 0``), replacing
    ``L`` by ``T`` therefore reduces exactly to :func:`cs_triple_energy`,
    which is pinned in the tests.
    """
    e_t = 0.0
    for so_pat, sv_pat, weight in _T_CLASSES:
        wc_r, _ = _cs_triple_buffers(
            i, j, k, t1, t2, ovvv, ooov, ovov, so_pat, sv_pat
        )
        wc_l, wd_l = _cs_triple_buffers(
            i, j, k, l1, l2, ovvv, ooov, ovov, so_pat, sv_pat, f_ov
        )
        d = (
            eps_o[i] + eps_o[j] + eps_o[k]
            - eps_v[:, None, None] - eps_v[None, :, None] - eps_v[None, None, :]
        )
        e_t += weight * float(np.sum(wc_r * (wc_l + wd_l) / d))
    return e_t


def cs_triples_correction(t1, t2, ovvv, ooov, ovov, eps_o, eps_v):
    """Full spatial closed-shell (T): S over all occupied (i,j,k) triples.

    The classwise spin weights absorb the ijk multiplicity, so the loop
    runs over every (i,j,k) in [0,no)^3 (not distinct i<j<k). Equals
    `_ccsd_ref.so_triples_correction` on the same amplitudes.
    """
    no = t1.shape[0]
    e_t = 0.0
    for i in range(no):
        for j in range(no):
            for k in range(no):
                e_t += cs_triple_energy(i, j, k, t1, t2, ovvv, ooov, ovov,
                                        eps_o, eps_v)
    return e_t


def cs_lambda_triples_correction(
    t1, t2, l1, l2, ovvv, ooov, ovov, eps_o, eps_v, f_ov=None
):
    """Full spatial closed-shell A-CCSD(T) Lambda triples correction.

    Uses the native C++ solver (spatial_lambda_triples) when available;
    falls back to the pure-Python loop over occupied triples otherwise.
    """
    import numpy as np

    if f_ov is None:
        f_ov = np.zeros((t1.shape[0], t1.shape[1]))

    try:
        from .._vibeqc_core import spatial_lambda_triples
    except ImportError:
        spatial_lambda_triples = None

    if spatial_lambda_triples is not None:
        no = t1.shape[0]
        nv = t1.shape[1]
        return float(spatial_lambda_triples(
            np.asfortranarray(t1),
            np.asfortranarray(np.asarray(t2).reshape(no * no, nv * nv)),
            np.asfortranarray(l1),
            np.asfortranarray(np.asarray(l2).reshape(no * no, nv * nv)),
            np.asfortranarray(f_ov),
            _reshape_4d_to_flat(ovvv, no, nv, nv, nv),
            _reshape_4d_to_flat(ooov, no, no, no, nv),
            _reshape_4d_to_flat(ovov, no, nv, no, nv),
            np.asarray(eps_o), np.asarray(eps_v),
        ))

    # ---- Pure-Python fallback ----
    no = t1.shape[0]
    e_t = 0.0
    for i in range(no):
        for j in range(no):
            for k in range(no):
                e_t += cs_lambda_triple_energy(
                    i, j, k, t1, t2, l1, l2, ovvv, ooov, ovov,
                    eps_o, eps_v, f_ov
                )
    return e_t
