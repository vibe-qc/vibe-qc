// Analytic Fourier transform of AO pair densities χ_μ(r)·χ_ν(r).
//
// Periodic GDF needs the Bloch-summed pair-FT
//
//     FT^{(+k)}_μν(G) = Σ_R exp(+i k·R) · ∫ χ_μ(r) χ_ν(r − R) e^{−iG·r} dr
//
// for every (μ, ν) AO pair, every reciprocal-space sample G, and a
// caller-supplied real-space cell list {R}. The result is the Fourier
// coefficient of the AO-pair density at crystal momentum k and is
// what the all-FT-Bloch RSGDF 3c-tensor builder
// (``vibeqc.aux_basis.build_lpq_native_fft``) contracts against the
// auxiliary FT to produce the GDF Lpq tensor.
//
// Algorithm. Per shell pair (μ_shell, ν_shell), the pair density on
// each cell decomposes via the Gaussian product theorem into a
// Hermite expansion in the Cartesian product Gaussian; the analytic
// FT of every term lives on a single (π/γ)^{3/2} · exp(−G²/4γ) ·
// exp(−iG·P) primitive plus a polynomial chain (McMurchie-Davidson
// 1978, Helgaker, Jørgensen & Olsen 2000 §9.5; Sun 2017 derives the
// periodic specialisation that PySCF's ``ft_aopair`` ships). For
// L=0 only the closed-form Gaussian-product piece is needed; higher
// L uses the full Hermite recursion and Cartesian-to-spherical transform.
//
// Memory. The C++ kernel performs the Bloch sum **streaming over
// the cell index** so the per-cell intermediate
// ``(n_cells, n_orb, n_orb, n_G)`` (which OOMs on MgO 8-atom
// conventional rocksalt — see
// ``handovers/HANDOVER_GDF_V0_11_2026_05_29.md``)
// never materialises. The output extent is the bounded
// ``(n_orb, n_orb, n_G)``.
//
// References
// ----------
// - McMurchie, L. E. & Davidson, E. R. (1978). *J. Comput. Phys.*
//   **26**, 218. DOI 10.1016/0021-9991(78)90092-X.
// - Helgaker, T., Jørgensen, P. & Olsen, J. *Molecular
//   Electronic-Structure Theory* (Wiley, 2000), §9.3-9.5.
// - Sun, Q. (2017). *J. Chem. Phys.* **147**, 164119.
//   DOI 10.1063/1.4998644. (Periodic specialisation; ``ft_aopair``.)

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "basis.hpp"
#include "periodic.hpp"
#include "periodic_auxiliary_fourier.hpp"

namespace vibeqc {

// Bloch-summed AO-pair FT result. Dense row-major, contiguous, owns
// its own buffer. ``operator()`` is bounds-free for tight inner loops;
// callers that want range-checked access should index ``data``
// directly through ``stride()``.
struct AOPairFTTensor {
    std::vector<std::complex<double>> data;
    std::size_t n_orb = 0;
    std::size_t n_G = 0;

    std::complex<double>& operator()(std::size_t mu, std::size_t nu,
                                     std::size_t k) noexcept {
        return data[(mu * n_orb + nu) * n_G + k];
    }
    std::complex<double> operator()(std::size_t mu, std::size_t nu,
                                    std::size_t k) const noexcept {
        return data[(mu * n_orb + nu) * n_G + k];
    }
};

// One bounded AO-pair Fourier panel in the physical libint AO convention.
// Rows are flattened pairs mu * n_ket_basis + nu in [pair_begin, pair_begin +
// n_pairs), so intervals may cross shell boundaries. The vector axis is
// contiguous. Unlike the legacy AOPairFTTensor kernels, no full AO square or
// image list is allocated. output_bytes is the sole variable-size allocation.
struct AOPairFourierPanel {
    std::vector<std::complex<double>> data;
    std::uint64_t pair_begin = 0;
    std::size_t n_pairs = 0;
    std::size_t n_vectors = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t image_candidate_count = 0;
    std::uint64_t retained_pair_image_count = 0;
    std::uint64_t fixed_numeric_workspace_bytes = 0;
};

// Exact size of the fixed MD coefficient and Fourier-power arrays used by
// the serial panel evaluator. Ordinary scalar/control stack storage, input
// BasisSet storage, and the output vector object are not included. There is
// no variable-size heap workspace besides AOPairFourierPanel::data.
std::uint64_t ao_pair_fourier_fixed_numeric_workspace_bytes() noexcept;

// Sun et al., JCP 147, 164119 (2017), doi:10.1063/1.4998644, Eq. 16:
//   rho_mu,nu(p; k_ket) = sum_R exp(+i k_ket.R)
//       integral chi_mu(r) chi_nu(r-R) exp(-i p.r) dr.
// A correlation caller supplies p = G + q, q = k_ket - k_bra. This routine
// applies no Coulomb weight, cell-volume factor, k weight, or spin factor.
// It includes sqrt(4*pi/(2*L+1)) on each L>0 axis, unlike the legacy raw
// pair-FT API. Pure spherical L<=6 and s Cartesian shells are supported.
//
// This finite-cutoff reference sums image labels lexicographically in each
// pair's bounding box and retains |A_mu - A_nu - R| <= image_cutoff_bohr.
// The padded long-double geometric bounds and binary64 separation predicate
// are a numerical enumeration policy, NOT a certified source manifest or a
// bound on the omitted infinite-image tail. Source certification belongs to
// the later factor-source layer. This primitive supports only 3D systems.
// maximum_image_candidates caps the SUM of box candidates across the pair
// interval, before result allocation. It does not multiply by n_vectors.
// All extents, lanes, shell conventions, and candidate counts are preflighted
// before output allocation. The cap must be positive; empty panels are valid.
// No approximate Gamma transpose/mirror shortcut is used.
AOPairFourierPanel ao_pair_gaussian_fourier_panel(
    const BasisSet& basis,
    const PeriodicSystem& system,
    AuxiliaryFourierVectorView vectors,
    const Eigen::Vector3d& k_ket_cart,
    std::uint64_t pair_begin,
    std::uint64_t pair_count,
    double image_cutoff_bohr,
    std::uint64_t maximum_image_candidates,
    std::uint64_t output_byte_cap);

// Rectangular two-basis overload with the identical finite-image arithmetic.
// mu belongs to bra_basis and nu to ket_basis; flatten as mu*n_ket_basis+nu.
// Both bases are independently validated and borrowed, never merged/copied.
// The single-basis overload delegates here with the same basis on both axes.
// At p=0 this is the finite-image cross overlap S12(k_ket), with no 1/Nk
// or cell-volume factor (Sun Eq.10). It does not certify omitted image tails,
// matching to a particular HF overlap source, or a minimal-basis choice.
// Owned memory remains exactly selected_pairs*n_vectors*16 bytes plus the
// existing fixed numeric workspace; caller inventories both borrowed bases.
AOPairFourierPanel ao_pair_gaussian_fourier_panel(
    const BasisSet& bra_basis,
    const BasisSet& ket_basis,
    const PeriodicSystem& system,
    AuxiliaryFourierVectorView vectors,
    const Eigen::Vector3d& k_ket_cart,
    std::uint64_t pair_begin,
    std::uint64_t pair_count,
    double image_cutoff_bohr,
    std::uint64_t maximum_image_candidates,
    std::uint64_t output_byte_cap);

// Explicit finite shared-cell support for the reciprocal Ewald arm. Integer
// labels are tightly packed [cell_count,3]; the native original direct
// lattice constructs each Cartesian R. No supplied Cartesian cell payload,
// AO-pair distance cutoff, screening, mirror or Hermiticity projection is
// used. Cell order is preserved; duplicates are rejected. Asymmetric lists
// are valid numerical input, not a certificate of inversion closure.
struct AOPairFourierCellView {
    const std::int64_t* indices = nullptr;
    std::uint64_t cell_count = 0, element_count = 0;
};
struct AOPairFourierCellPanelCaps {
    // Pair-cell visits is the unmultiplied pair_count*cell_count census;
    // work_units additionally charges validation and every vector replay.
    std::uint64_t maximum_cells = 0, maximum_pair_cell_visits = 0;
    std::uint64_t maximum_work_units = 0, maximum_output_bytes = 0;
};
struct AOPairFourierCellPanelPlan {
    std::uint64_t n_basis = 0, n_vectors = 0, pair_begin = 0, n_pairs = 0;
    std::uint64_t cell_count = 0, output_bytes = 0;
    // Active primitive/contraction/origin/max-ln-coefficient lanes. Basis
    // object/container/lookup descriptors are separately charged below.
    std::uint64_t borrowed_basis_numeric_bytes = 0, borrowed_cell_bytes = 0;
    std::uint64_t basis_control_storage_bytes = 0, fixed_control_storage_bytes = 0;
    std::uint64_t fixed_numeric_workspace_bytes = 0, fixed_scalar_numeric_bytes = 0;
    std::uint64_t pair_cell_visits = 0, duplicate_comparisons = 0;
    std::uint64_t metadata_work_units = 0, validation_work_units = 0;
    std::uint64_t contraction_work_units = 0, work_units = 0;
};

// Allocation-free shape/census admission. No primitive values, cell labels,
// lattice/vector payloads or Bloch phases are read. Metadata work is capped
// incrementally before walking each shell/contraction. All four caps must be
// positive; zero cells/pairs/vectors produce valid empty sums. The plan is
// a numerical/control inventory, not a complete caller/node memory envelope.
// The caller inventories the borrowed vector lanes, original System (including
// unrelated retained atom/symmetry owners), and any enclosing owners separately.
AOPairFourierCellPanelPlan plan_ao_pair_gaussian_fourier_cell_panel(
    const BasisSet&, std::uint64_t n_vectors, std::uint64_t pair_begin,
    std::uint64_t pair_count, std::uint64_t cell_count,
    const AOPairFourierCellPanelCaps&);

// Same physical libint AO and +i*k.R convention as the bounded distance
// panel above (Sun 2017 Eq.16), pure L<=6 / Cartesian s. Call with -k for
// the BIPOLE inverse-density convention. Pair rows and vector columns are
// both bounded. The only variable allocation is the output complex array;
// duplicate detection uses the pre-admitted O(cell_count^2) comparison loop.
// Input owners remain immutable/alive during the call. Labels outside the
// exactly representable binary64 integer range are rejected. Returned image
// counts both equal pair_count*cell_count; fixed_numeric_workspace_bytes
// includes the separate plan's MD and scalar reservations. No omitted-tail,
// old-HF source equality, Ewald zero-mode or physical provenance claim.
AOPairFourierPanel ao_pair_gaussian_fourier_cell_panel(
    const BasisSet&, const PeriodicSystem&, AuxiliaryFourierVectorView,
    const Eigen::Vector3d& k_cart, std::uint64_t pair_begin,
    std::uint64_t pair_count, AOPairFourierCellView,
    const AOPairFourierCellPanelCaps&);

// Bloch-summed AO-pair FT, all (μ, ν) blocks at every G.
//
//     out_μν(G) = Σ_R exp(+i k·R) · FT_μν(G; R)
//
// where ``FT_μν(G; R)`` is the molecular-style AO-pair FT with the
// ket Gaussian translated by R. The +ik sign matches vibe-qc's
// Bloch-sum convention (cf. ``vibeqc.pbc_bipole._bloch_sum_blocks``).
//
// Parameters
// ----------
// basis
//     AO basis. Currently REQUIRES all shells to have L=0; mixed-L
//     bases throw. The general-L McMurchie-Davidson path lands in a
//     subsequent milestone (Item 1b in the integrals-chat handover).
// G_vectors
//     ``(n_G, 3)``, Cartesian reciprocal-space samples (inverse bohr).
// R_g_list
//     ``(n_g, 3)``, Cartesian lattice translation vectors (bohr).
//     Pass a single row ``(0, 0, 0)`` for the molecular limit.
// k_cart
//     ``(3,)``, Cartesian crystal momentum (inverse bohr).
//
// Returns
// -------
// AOPairFTTensor with shape ``(n_orb, n_orb, n_G)`` complex128.
//
// Throws
// ------
// std::invalid_argument
//     If ``basis`` contains any shell with L > 0, or if the array
//     extents don't satisfy the documented contract.
AOPairFTTensor ao_pair_fourier_transform_bloch_ss(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Vector3d& k_cart);

// General-L Bloch-summed AO-pair FT.
//
// Implements the full McMurchie-Davidson Hermite-chain pair-FT
// + Cartesian-to-spherical transform on both AO axes. Handles
// arbitrary mixed-L bases up to L = ``vibeqc::cart_to_sph_data::kMaxL``
// (currently 6). libint's build-time max_am is configurable
// (scripts/_libint_max_am.sh, default 5 at derivative order 0) but is
// capped at kMaxL for exactly this reason — a higher libint would
// produce shells this transform rejects.
//
// Parameters and return contract are identical to the s-only path
// above. Pure spherical-harmonic shells only (``ShellInfo::pure ==
// true``); Cartesian (non-pure) shells with L > 0 are not supported
// — vibe-qc forces pure spherical for L ≥ 2 by default so this
// should not arise in practice.
//
// Throws ``std::invalid_argument`` on
//   * a Cartesian-ordered shell with L > 0, or
//   * any shell with L > ``vibeqc::cart_to_sph_data::kMaxL``.
AOPairFTTensor ao_pair_fourier_transform_bloch(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Vector3d& k_cart,
    double screen_tol = 0.0);

// Per-cell reciprocal-weight-contracted AO-pair FT.
//
//   out[g][m,n] = Re( sum_G w(G) . conj( FT_mn(G; R_g) ) )
//
// with ``FT_mn(G; R_g) = int chi_m(r) chi_n(r - R_g) e^{-iG.r} dr`` and
// ``w(G) = reciprocal_weights``. This is the reciprocal half of the
// Ewald-split periodic nuclear attraction consumed by
// ``vibeqc.periodic_v_ne.compute_v_ne_ewald_3d_ft_lattice``.
//
// Unlike the Bloch entry points above there is NO Bloch sum: each cell
// keeps its own block. That has two consequences the caller benefits
// from. The weight contraction is folded into the innermost Hermite
// reduction, so the dense ``(n_orb, n_orb, n_G)`` tensor never
// materialises (Re() commutes out because every remaining factor is
// real). And because ``(shell-pair, cell)`` tasks write disjoint output
// blocks, the whole flattened task space parallelises with no reduction
// -- where the Bloch kernels can only parallelise over shell pairs,
// which starves the threads on a small basis with many image cells.
//
// Returns a ``(n_g, n_orb, n_orb)`` row-major real buffer. ``screen_tol``
// has the same conservative shell-pair/cell meaning as above.
std::vector<double> ao_pair_fourier_transform_weighted_per_cell(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Ref<const Eigen::VectorXcd>& reciprocal_weights,
    double screen_tol = 0.0);

// Multi-k batched general-L Bloch-summed AO-pair FT.
//
// Evaluates the Bloch pair-FT above for EVERY crystal momentum row of
// ``k_carts`` (``(n_k, 3)``) on one shared reciprocal support: the
// per-cell Cartesian McMurchie-Davidson work is k-independent (the ket
// momentum enters only through the Bloch phase exp(+i k·R)), so the
// batch costs one pair-FT pass plus n_k cheap phase folds instead of
// n_k full passes. This is the exact work-sharing companion of the
// RSGDF q-metric cache for the 3c side (multi-k GDF drivers batch the
// pairs that share one momentum transfer q; see
// ``vibeqc.aux_basis.build_lpq_bloch_native_fft`` and
// handovers/HANDOVER_OPEN_BUGS_V015.md, production-k-sampling item).
//
// Returns one tensor per row of ``k_carts``, each ``(n_orb, n_orb,
// n_G)``. Agrees with n_k single-k calls to floating-point rounding
// (the phase fold is applied after the cart→sph transform rather than
// fused into it). Same shell/L contract and ``screen_tol`` semantics
// as the single-k kernel.
//
// ``pair_weights`` is an OPTIONAL ``(n_orb, n_orb)`` row-major real
// weight applied per AO pair as the result is stored. Empty means "no
// weighting" and leaves the kernel byte-unchanged. The consumers of
// this tensor (the RSGDF cderi accumulators) all scale it by a
// per-pair factor before contracting, and that scaling is a pass over
// the LARGEST array in the build -- single-threaded in NumPy, and
// measurably anti-scaling against the threads this kernel itself uses
// (handovers/HANDOVER_GDF_OUTSTANDING.md, OpenMP profile 2026-08-05).
// Folding it into this store costs nothing: the store already touches
// every element, and it happens inside the parallel region.
//
// The arithmetic is chosen to be BIT-IDENTICAL to the NumPy
// ``complex128_array *= float64_array`` it replaces, which casts the
// real factor to complex and runs the full complex product:
//
//   real = v.re * w - v.im * 0.0
//   imag = v.re * 0.0 + v.im * w
//
// Plain componentwise scaling (``v.re * w``, ``v.im * w``) is NOT the
// same: the two differ in the SIGN OF ZERO whenever a component is
// itself zero. A weight of exactly 0.0 means "masked pair" and stores
// a true +0.0 (matching a NumPy masked assignment, not a multiply).
std::vector<AOPairFTTensor> ao_pair_fourier_transform_bloch_multi(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& k_carts,
    double screen_tol = 0.0,
    const std::vector<double>& pair_weights = {});

// Density- and reciprocal-kernel-weighted Gamma-point centre derivative.
//
// Evaluates
//
//   d/dR_A Re sum_G q(G) conj[
//       sum_mu,nu W_mu,nu sum_g FT_mu,nu(G; R_g)]
//
// without materialising the five-dimensional per-cell AO-pair derivative
// tensors. ``pair_weights`` is in the raw AO-pair-FT convention; callers
// that use the RSGDF/libint calibration must include those per-AO factors
// in W. ``reciprocal_weights`` contains q(G), including any cell-volume
// normalization. Every shell is scattered through ``ShellInfo::atom_index``;
// ``n_atoms`` may exceed the number of atoms represented in the basis.
//
// The implementation differentiates the ket Cartesian Gaussian with the
// exact angular-momentum shift recurrence and obtains the bra derivative
// from translational covariance,
//
//   d_A FT + d_B FT = -i G FT.
//
// Pure spherical shells through the same L bound as the value kernel are
// supported. Returns ``(n_atoms, 3)`` in the units implied by q(G).
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
ao_pair_fourier_transform_gamma_gradient_weighted(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic,
                                         Eigen::Dynamic, Eigen::RowMajor>>&
        pair_weights,
    const Eigen::Ref<const Eigen::VectorXcd>& reciprocal_weights,
    std::size_t n_atoms);

// G-resolved-weight variant of the Gamma-point centre derivative above.
//
// Evaluates
//
//   d/dR_A Re sum_G sum_mu,nu Q_mu,nu(G) conj[ sum_g FT_mu,nu(G; R_g) ]
//
// where the separable ``pair_weights (n_orb, n_orb)`` x
// ``reciprocal_weights (n_G,)`` product of the scalar kernel is replaced
// by one complex G-resolved pair weight Q of shape (n_orb, n_orb, n_G).
// For any separable Q_mu,nu(G) = sum_p W^p_mu,nu q^p(G) the result equals
// the sum of scalar-kernel calls over p up to floating-point
// reassociation. Derivative recurrences, spherical transform and atom
// scatter are identical to the scalar kernel.
//
// ``pair_weights_g`` is a C-contiguous row-major (n_orb, n_orb, n_G)
// complex buffer, read shared across the OpenMP workers; shape/contiguity
// validation is the binding's job. Returns ``(n_atoms, 3)``.
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
ao_pair_fourier_transform_gamma_gradient_gweighted(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const std::complex<double>* pair_weights_g,
    std::size_t n_atoms);

// Bloch-phased G-resolved-weight centre derivative
// (handovers/HANDOVER_GDF_GRADIENT_DEFERRED.md § 4, rung 1).
//
// Evaluates
//
//   d/dR_A Re sum_G sum_mu,nu Q_mu,nu(G)
//       conj[ sum_g exp(+i k.R_g) FT_mu,nu(G; R_g) ]
//
// i.e. the gweighted kernel above with the ket Bloch phase of
// ``ao_pair_fourier_transform_bloch`` threaded into the cell sum. The
// phase is atom-position independent, so the derivative acts only on
// the Gaussian-product centre exactly as in the Gamma kernel; under
// the conjugation the per-cell factor composes to exp(-i k.R_g).
// ``k_cart = 0`` reduces to the Gamma kernel bit-for-bit (the phase is
// exactly (1, 0)). Same Q layout, derivative recurrences, spherical
// transform, atom scatter, and OpenMP structure as the Gamma kernel.
// Returns ``(n_atoms, 3)``.
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
ao_pair_fourier_transform_bloch_gradient_gweighted(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Vector3d& k_cart,
    const std::complex<double>* pair_weights_g,
    std::size_t n_atoms);

}  // namespace vibeqc
