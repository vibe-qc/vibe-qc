"""Selected Configuration Interaction solver.

Implements a CIPSI-style (Configuration Interaction using a Perturbative
Selection made Iteratively) selected-CI algorithm:

1. Start with a reference determinant (usually the HF determinant).
2. Generate all singles and doubles from the current space.
3. Estimate each candidate's first-order perturbative contribution:
       ΔE_D ≈ |<D|H|Ψ_0>|^2 / (E_0 - <D|H|D>)
4. Add the top-N candidates to the variational space.
5. Diagonalize the Hamiltonian in the expanded space.
6. Repeat until convergence (energy change, determinant count, or PT2 estimate).

Optionally, compute a final PT2 correction using the Epstein-Nesbet denominator.

Supports both spin-restricted (closed-shell ``Det``) and spin-unrestricted
(``SpinDet`` = ``(alpha_occ, beta_occ)``) determinant spaces.

References
----------
* Huron, Malrieu & Rancurel, *J. Chem. Phys.* 58, 5745 (1973).
* Evangelisti, Daudey & Malrieu, *Chem. Phys.* 75, 91 (1983).
* Tubman et al., *JCTC* 17, 151 (2021) -- ASCI / SHCI modern variants.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from math import comb

from vibeqc.output import warn, write

from ._common import Hamiltonian, SolverOptions, SolverResult
from ._determinant import Det, SpinDet, reference_determinant
from ._slater_condon import (
    build_hamiltonian_matrix,
    build_hamiltonian_matrix_unrestricted,
    diagonal_matrix_element,
    diagonal_matrix_element_unrestricted,
    hamiltonian_matrix_element_unrestricted,
    pair_excitation_matrix_element,
)


@dataclass
class SelectedCIOptions(SolverOptions):
    """Options for the Selected-CI solver.

    Attributes
    ----------
    target_size : int
        Maximum number of determinants in the variational space.
    max_iter : int
        Maximum selection + diagonalization cycles.
    conv_tol_energy : float
        Convergence threshold on total energy change (Hartree).
    pt2_threshold : float
        Perturbative threshold for selecting new determinants.
        Lower = more determinants selected per iteration.
    selection_growth_factor : float
        Max ratio by which the determinant space can grow each iteration.
    max_det_per_iter : int
        Hard cap on new determinants per iteration.
    do_pt2_correction : bool
        Compute final PT2 energy correction after convergence.
    spin_restricted : bool
        Default False (issue #107): the solver walks SpinDet =
        (alpha_occ, beta_occ) determinants, the full S_z sector of the
        active space, so a converged run is the CAS/full-CI answer.
        True selects the closed-shell (seniority-zero, DOCI) basis, in
        which each Det is a tuple of doubly-occupied spatial orbitals
        excited by whole pairs, coupled through the pair-move integral
        ``g_{iiaa}`` (issue #639).  That basis is a closed subspace: it
        cannot represent an open-shell configuration, so on a system
        whose exact state has seniority > 0 it converges to the DOCI
        energy, not the CAS limit; a restricted run whose candidate pool
        empties short of ``target_size`` reports ``converged=False``
        with ``stop_reason="space_exhausted"`` and warns.
    use_davidson : bool
        Use Davidson iterative diagonalization for ndet > davidson_threshold.
    davidson_threshold : int
        Switch to Davidson when ndet exceeds this count.
    davidson_max_iter : int
        Maximum Davidson iterations.
    davidson_conv_tol : float
        Davidson residual convergence tolerance.
    """

    pt2_threshold: float = 1e-6
    selection_growth_factor: float = 2.0
    max_det_per_iter: int = 5000
    do_pt2_correction: bool = True
    # Default False since issue #107 (maintainer decision 2026-08-27): the
    # restricted basis cannot reach the CAS limit (N2/cc-pVDZ CAS(6e,6o):
    # -108.9594672752 Ha at every target_size against the CASCI oracle
    # -109.0217750033 Ha, a silent 62.3 mHa), and the unrestricted basis
    # reaches it (ndet 96, bit-identical to the oracle).
    spin_restricted: bool = False
    use_davidson: bool = True
    davidson_threshold: int = 50
    davidson_max_iter: int = 100
    davidson_conv_tol: float = 1e-8
    # ── selected-CASCI kernel knobs (:func:`selected_casci`) ──
    #: Coefficient magnitude above which a variational determinant
    #: contributes to candidate generation (max over roots).
    significant_coeff: float = 0.01
    #: For an M_s = 0 active space, close the selected set under the
    #: alpha/beta swap so the truncated roots stay (near-)spin-pure.
    spin_complete: bool = True
    #: Heat-bath prefilter (Holmes-Tubman-Umrigar 2016): during candidate
    #: generation, drop the contribution of generator determinant I to
    #: candidate D when ``|H_DI| * max_k |c_I^k| < select_eps``.  0
    #: (default) keeps exact CIPSI numerators.  The C++ backend walks
    #: presorted |H|-ordered double-excitation lists under this
    #: threshold, so enumeration stops at the first sub-threshold entry
    #: instead of visiting every virtual pair; the Python kernel applies
    #: the same predicate by brute force (identical candidate sets, the
    #: oracle for the walk).
    select_eps: float = 0.0


@dataclass
class _SelectionCandidate:
    """Internal: a candidate determinant with its PT2 weight estimate."""

    det: Union[Det, SpinDet]
    weight: float
    coupling: float  # |<D|H|Ψ_0>|


class SelectedCISolver:
    """CIPSI-style selected configuration interaction solver.

    Supports both spin-restricted (``spin_restricted=True``) and
    spin-unrestricted (``spin_restricted=False``) determinant spaces.
    """

    def __init__(self, options: Optional[SelectedCIOptions] = None):
        self.options = options or SelectedCIOptions()
        self._rng = np.random.default_rng(self.options.random_seed)

    def solve(
        self,
        hamiltonian: Hamiltonian,
        options: Optional[SelectedCIOptions] = None,
    ) -> SolverResult:
        opts = options or self.options
        h1e = hamiltonian.h1e
        h2e = hamiltonian.h2e
        enuc = hamiltonian.nuclear_repulsion
        norb = hamiltonian.norb
        nelec = hamiltonian.nelec

        if nelec % 2 != 0 and opts.spin_restricted:
            raise ValueError(
                "Spin-restricted Selected-CI requires an even number of electrons. "
                "Use spin_restricted=False for open-shell systems."
            )

        # ── Initialize with reference determinant ─────────────────────
        if opts.spin_restricted:
            nocc = nelec // 2
            ref_det = tuple(range(nocc))
            det_space: list = [ref_det]
        else:
            nalpha = (nelec + hamiltonian.ms2) // 2
            nbeta = (nelec - hamiltonian.ms2) // 2
            ref_a, ref_b = reference_determinant(nalpha, nbeta, spin_restricted=False)
            det_space: list = [(ref_a, ref_b)]

        energy_trace: list[float] = []
        prev_energy = float("inf")
        converged = False
        stop_reason: str = "max_iter"

        t_start = time.perf_counter()

        # Save last diagonalization results for PT2 and final result
        eigenvalues = None
        eigenvectors = None
        coeffs = None

        for iteration in range(opts.max_iter):
            # ── Build + diagonalize H in current space ────────────────
            ndet = len(det_space)
            if ndet <= 0:
                break

            if opts.spin_restricted:
                H_mat = build_hamiltonian_matrix(det_space, h1e, h2e)
            else:
                H_mat = build_hamiltonian_matrix_unrestricted(det_space, h1e, h2e)

            # Use Davidson for larger spaces to avoid O(N^3) dense diagonalization
            if opts.use_davidson and ndet > opts.davidson_threshold:
                eigenvalues, eigenvectors = self._davidson(
                    H_mat, det_space, h1e, h2e, opts
                )
            else:
                eigenvalues, eigenvectors = scipy_eigh(H_mat)
            e_var = eigenvalues[0] + enuc
            coeffs = eigenvectors[:, 0]
            energy_trace.append(e_var)

            delta_e = abs(e_var - prev_energy)
            prev_energy = e_var

            if opts.verbose >= 1:
                write(
                    f"  Selected-CI iter {iteration + 1:3d}: "
                    f"ndet={ndet:6d}, E_var={e_var:.10f}, ΔE={delta_e:.2e}\n"
                )

            # ── Check convergence ─────────────────────────────────────
            if delta_e < opts.conv_tol_energy and iteration > 0:
                converged = True
                stop_reason = "energy"
                if opts.verbose:
                    write(
                        f"  Selected-CI converged in {iteration + 1} iterations.\n"
                    )
                break

            if ndet >= opts.target_size:
                if opts.verbose:
                    write(
                        f"  Selected-CI: reached target size {opts.target_size}.\n"
                    )
                converged = True
                stop_reason = "target_size"
                break

            # ── Selection step ────────────────────────────────────────
            candidates = self._select_candidates(
                det_space,
                coeffs,
                h1e,
                h2e,
                norb,
                opts,
                spin_restricted=opts.spin_restricted,
                e_var_0=eigenvalues[0],
            )

            if not candidates and ndet < opts.target_size:
                # BUG #107: a fixed pt2_threshold can empty the candidate
                # pool far below target_size.  Retry with progressively
                # relaxed thresholds so target_size stays meaningful
                # (still capped by max_det_per_iter / growth factor).
                #
                # The floor is one ulp of the current total energy, not
                # zero.  A candidate whose second-order estimate is below
                # the resolution of E_var in double precision cannot move
                # E_var, so admitting it grows ndet and nothing else.
                # Measured on N2/cc-pVDZ CAS(6e,6o), spin_restricted=True
                # (#107): the floor-0 relaxation admitted ten determinants
                # of weight 4e-28..4e-33 Ha against a 2.4e-14 Ha ulp, the
                # energy stayed bit-identical, and the run then reported
                # stop_reason "energy" for a pool that was in fact empty.
                # The unrestricted ladder on the same system admits nothing
                # below 9e-10 Ha and is unchanged by the floor.
                _floor = float(np.finfo(float).eps) * max(abs(float(e_var)), 1.0)
                _thr = opts.pt2_threshold
                while True:
                    _thr = max(_thr * 0.1, _floor)
                    candidates = self._select_candidates(
                        det_space,
                        coeffs,
                        h1e,
                        h2e,
                        norb,
                        opts,
                        spin_restricted=opts.spin_restricted,
                        e_var_0=eigenvalues[0],
                        pt2_threshold_override=_thr,
                    )
                    if candidates or _thr <= _floor:
                        break
                if candidates and opts.verbose >= 1:
                    write(
                        f"  Selected-CI: candidate pool starved at "
                        f"pt2_threshold={opts.pt2_threshold:.1e}; "
                        f"relaxed to {_thr:.1e}.\n"
                    )

            if not candidates:
                if opts.verbose:
                    write(
                        "  Selected-CI: candidate pool exhausted below target "
                        f"size (ndet={ndet} < {opts.target_size}); no new "
                        "determinants above any threshold.\n"
                    )
                # The connected space is complete: a fixed point of the
                # selection procedure, but NOT the same as the energy
                # criterion -- the stop_reason field distinguishes the two.
                converged = True
                stop_reason = "space_exhausted"
                break

            # Add top candidates to space
            new_dets = [c.det for c in candidates]
            det_space_set = set(det_space)
            for d in new_dets:
                if d not in det_space_set:
                    det_space.append(d)

            # Safety: don't exceed target size by too much
            if len(det_space) > int(opts.target_size * opts.selection_growth_factor):
                # Trim back to target_size based on coefficient magnitude
                coeff_abs = np.abs(coeffs)
                keep_idx = np.argsort(-coeff_abs)[: opts.target_size]
                det_space = [det_space[i] for i in keep_idx]

        # ── PT2 correction ────────────────────────────────────────────
        pt2_corr = 0.0
        if opts.do_pt2_correction and eigenvalues is not None and coeffs is not None:
            pt2_corr = self._compute_pt2(
                det_space,
                eigenvalues[0],
                coeffs,
                h1e,
                h2e,
                norb,
                opts,
                spin_restricted=opts.spin_restricted,
            )

        # ── Spin-restricted reachability warning (issue #107) ─────────
        #
        # spin_restricted=True walks the *seniority-zero* basis: closed-shell
        # determinants over spatial orbitals, excited by whole pairs
        # (see _select_candidates_restricted).  That subspace holds
        # C(norb, nelec/2) determinants and cannot represent any open-shell
        # (seniority > 0) configuration, so for a system whose CAS solution
        # has open-shell character the variational energy converges to a
        # limit that is NOT the CAS energy -- and converges honestly, so
        # `converged=True` with stop_reason "energy" is reported.
        #
        # Measured on N2/cc-pVDZ CAS(6e,6o) (2026-09-03, main 65201761f):
        # the restricted ladder stops at ndet=6 with E = -108.9594672752 Ha
        # and E(PT2) numerically 0 at every target_size from 6 to 200, while
        # spin_restricted=False reaches ndet=96 and E = -109.0217750033 Ha,
        # bit-identical to the CASCI oracle.  The 62.3 mHa between them is
        # not a selection failure, which is why relaxing the selection
        # thresholds alone did not move it.  (Until #639 the restricted
        # basis was not even a seniority-zero CI: its pair-move coupling was
        # evaluated with the one-electron Brillouin formula and vanished for
        # HF orbitals, so the ladder above froze at ndet=6 and H2/STO-3G with
        # both closed-shell determinants returned E_HF.  With the g_{iiaa}
        # coupling the restricted basis is the DOCI Hamiltonian: H2/STO-3G
        # gives the FCI energy -1.1372759436 Ha, and the N2 CAS(6e,6o) ladder
        # reaches its 20-determinant DOCI limit -108.9850687702 Ha -- still
        # 36.7 mHa above CASCI, because the missing configurations are
        # open-shell, which is the point of this warning.)
        #
        # A user who asked for `target_size` determinants and got far fewer
        # has hit this ceiling, so that is the trigger: it fires exactly when
        # the budget could not be spent, and stays quiet when the restricted
        # space genuinely satisfied the request.  When the pool emptied
        # short of the target the run is NOT reported converged: the closed
        # subspace cannot certify that the configurations it omits are
        # unimportant (they are worth 62 mHa above), so `converged=True`
        # would be the silent-wrong-answer label #107 was filed against.
        if opts.spin_restricted and len(det_space) < opts.target_size:
            _seniority_zero_space = comb(norb, nelec // 2)
            if stop_reason == "space_exhausted":
                converged = False
            warn(
                f"spin-restricted selected-CI stopped on {len(det_space)} "
                f"determinants, short of the requested target_size "
                f"{opts.target_size}: the restricted basis holds only the "
                f"{_seniority_zero_space} closed-shell (seniority-zero) "
                f"determinants of this space and cannot represent open-shell "
                f"configurations: its limit is the seniority-zero (DOCI) "
                f"energy. If the target is the CAS/full-CI energy, this run "
                f"cannot reach it -- rerun with spin_restricted=False (the "
                f"default).",
                method="selected_ci",
            )

        dt = time.perf_counter() - t_start

        return SolverResult(
            energy=e_var + pt2_corr,
            method=f"selected_ci(ndet={len(det_space)})",
            converged=converged,
            stop_reason=stop_reason,
            n_iter=iteration + 1,
            energy_trace=energy_trace,
            ci_coeffs=coeffs,
            ci_labels=det_space,
            pt2_correction=pt2_corr,
        )

    def _select_candidates(
        self,
        det_space: list,
        coeffs: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        opts: SelectedCIOptions,
        *,
        spin_restricted: bool = True,
        e_var_0: float = 0.0,
        pt2_threshold_override: Optional[float] = None,
    ) -> list[_SelectionCandidate]:
        """Generate candidates via singles/doubles from top determinants.

        When ``spin_restricted=False``, ``det_space`` is ``list[SpinDet]`` and
        all five spin-sector excitation classes are explored (a singles,
        b singles, aa doubles, bb doubles, ab doubles).

        ``pt2_threshold_override`` temporarily replaces
        ``opts.pt2_threshold`` for the final weight filter (the
        starvation relaxation in :meth:`solve`; ``None`` = use the
        option value).
        """
        if spin_restricted:
            return self._select_candidates_restricted(
                det_space,
                coeffs,
                h1e,
                h2e,
                norb,
                opts,
                e_var_0=e_var_0,
                pt2_threshold_override=pt2_threshold_override,
            )
        else:
            return self._select_candidates_unrestricted(
                det_space,
                coeffs,
                h1e,
                h2e,
                norb,
                opts,
                e_var_0=e_var_0,
                pt2_threshold_override=pt2_threshold_override,
            )

    def _select_candidates_restricted(
        self,
        det_space: list[Det],
        coeffs: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        opts: SelectedCIOptions,
        *,
        e_var_0: float = 0.0,
        pt2_threshold_override: Optional[float] = None,
    ) -> list[_SelectionCandidate]:
        """Spin-restricted candidate selection (closed-shell determinants).

        The seniority-zero basis couples only through single pair moves
        ``i^2 -> a^2`` with ``H_aI = g_{iiaa}`` (``_slater_condon``
        module docstring; GitLab #639); a two-pair move is a four-electron
        excitation and has no coupling, so it is never a candidate.
        """
        candidates: dict[Det, _SelectionCandidate] = {}
        det_set = set(det_space)

        # Focus on determinants with significant weight
        c_abs = np.abs(coeffs)
        c_threshold = 0.01
        significant = np.where(c_abs > c_threshold)[0]
        if len(significant) == 0:
            significant = [np.argmax(c_abs)]

        for I in significant:
            occ_I = det_space[I]
            occ_set = set(occ_I)
            vir_list = sorted(set(range(norb)) - occ_set)

            # Pair moves
            for i in occ_set:
                for a in vir_list:
                    new_det = tuple(sorted((occ_set - {i}) | {a}))
                    if new_det in candidates or new_det in det_set:
                        continue
                    H_Ia = pair_excitation_matrix_element(i, a, h2e)
                    coupling = abs(H_Ia * coeffs[I])
                    diag_new = diagonal_matrix_element(new_det, h1e, h2e)
                    denom = e_var_0 - diag_new
                    if abs(denom) < 1e-12:
                        continue
                    weight = coupling * coupling / abs(denom)
                    candidates[new_det] = _SelectionCandidate(
                        det=new_det, weight=weight, coupling=coupling
                    )

        # Sort by weight descending, filter by threshold
        sorted_cands = sorted(candidates.values(), key=lambda c: c.weight, reverse=True)
        _thr = (
            opts.pt2_threshold
            if pt2_threshold_override is None
            else pt2_threshold_override
        )
        selected = [c for c in sorted_cands if c.weight > _thr]
        max_new = max(
            10,
            min(
                opts.max_det_per_iter,
                int(len(det_space) * (opts.selection_growth_factor - 1.0)),
            ),
        )
        return selected[:max_new]

    def _select_candidates_unrestricted(
        self,
        det_space: list[SpinDet],
        coeffs: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        opts: SelectedCIOptions,
        *,
        e_var_0: float = 0.0,
        pt2_threshold_override: Optional[float] = None,
    ) -> list[_SelectionCandidate]:
        """Spin-unrestricted candidate selection over all five excitation classes."""
        candidates: dict[SpinDet, _SelectionCandidate] = {}
        det_set = set(det_space)

        # Focus on determinants with significant weight
        c_abs = np.abs(coeffs)
        c_threshold = 0.01
        significant = np.where(c_abs > c_threshold)[0]
        if len(significant) == 0:
            significant = [np.argmax(c_abs)]

        for I in significant:
            aI, bI = det_space[I]
            occ_a = set(aI)
            occ_b = set(bI)
            vir_a = sorted(set(range(norb)) - occ_a)
            vir_b = sorted(set(range(norb)) - occ_b)

            # ── a single excitations ──────────────────────────────────
            for i in occ_a:
                for a in vir_a:
                    new_a = tuple(sorted((occ_a - {i}) | {a}))
                    new_det = (new_a, bI)
                    if new_det in candidates or new_det in det_set:
                        continue
                    coupling = abs(
                        hamiltonian_matrix_element_unrestricted(
                            new_a, bI, aI, bI, h1e, h2e
                        )
                        * coeffs[I]
                    )
                    diag_new = diagonal_matrix_element_unrestricted(new_a, bI, h1e, h2e)
                    denom = e_var_0 - diag_new
                    if abs(denom) < 1e-12:
                        continue
                    weight = coupling * coupling / abs(denom)
                    candidates[new_det] = _SelectionCandidate(
                        det=new_det, weight=weight, coupling=coupling
                    )

            # ── b single excitations ──────────────────────────────────
            for i in occ_b:
                for a in vir_b:
                    new_b = tuple(sorted((occ_b - {i}) | {a}))
                    new_det = (aI, new_b)
                    if new_det in candidates or new_det in det_set:
                        continue
                    coupling = abs(
                        hamiltonian_matrix_element_unrestricted(
                            aI, new_b, aI, bI, h1e, h2e
                        )
                        * coeffs[I]
                    )
                    diag_new = diagonal_matrix_element_unrestricted(aI, new_b, h1e, h2e)
                    denom = e_var_0 - diag_new
                    if abs(denom) < 1e-12:
                        continue
                    weight = coupling * coupling / abs(denom)
                    candidates[new_det] = _SelectionCandidate(
                        det=new_det, weight=weight, coupling=coupling
                    )

            # ── aa double excitations ─────────────────────────────────
            a_list = sorted(occ_a)
            for idx_i in range(len(a_list)):
                i = a_list[idx_i]
                for idx_j in range(idx_i + 1, len(a_list)):
                    j = a_list[idx_j]
                    for idx_a in range(len(vir_a)):
                        a = vir_a[idx_a]
                        for idx_b in range(idx_a + 1, len(vir_a)):
                            b = vir_a[idx_b]
                            new_a = tuple(sorted((occ_a - {i, j}) | {a, b}))
                            new_det = (new_a, bI)
                            if new_det in candidates or new_det in det_set:
                                continue
                            coupling = abs(
                                hamiltonian_matrix_element_unrestricted(
                                    new_a, bI, aI, bI, h1e, h2e
                                )
                                * coeffs[I]
                            )
                            diag_new = diagonal_matrix_element_unrestricted(
                                new_a, bI, h1e, h2e
                            )
                            denom = e_var_0 - diag_new
                            if abs(denom) < 1e-12:
                                continue
                            weight = coupling * coupling / abs(denom)
                            candidates[new_det] = _SelectionCandidate(
                                det=new_det, weight=weight, coupling=coupling
                            )

            # ── bb double excitations ─────────────────────────────────
            b_list = sorted(occ_b)
            for idx_i in range(len(b_list)):
                i = b_list[idx_i]
                for idx_j in range(idx_i + 1, len(b_list)):
                    j = b_list[idx_j]
                    for idx_a in range(len(vir_b)):
                        a = vir_b[idx_a]
                        for idx_b in range(idx_a + 1, len(vir_b)):
                            b = vir_b[idx_b]
                            new_b = tuple(sorted((occ_b - {i, j}) | {a, b}))
                            new_det = (aI, new_b)
                            if new_det in candidates or new_det in det_set:
                                continue
                            coupling = abs(
                                hamiltonian_matrix_element_unrestricted(
                                    aI, new_b, aI, bI, h1e, h2e
                                )
                                * coeffs[I]
                            )
                            diag_new = diagonal_matrix_element_unrestricted(
                                aI, new_b, h1e, h2e
                            )
                            denom = e_var_0 - diag_new
                            if abs(denom) < 1e-12:
                                continue
                            weight = coupling * coupling / abs(denom)
                            candidates[new_det] = _SelectionCandidate(
                                det=new_det, weight=weight, coupling=coupling
                            )

            # ── ab double excitations ─────────────────────────────────
            for i in occ_a:
                for j in occ_b:
                    for a in vir_a:
                        for b in vir_b:
                            new_a = tuple(sorted((occ_a - {i}) | {a}))
                            new_b = tuple(sorted((occ_b - {j}) | {b}))
                            new_det = (new_a, new_b)
                            if new_det in candidates or new_det in det_set:
                                continue
                            coupling = abs(
                                hamiltonian_matrix_element_unrestricted(
                                    new_a, new_b, aI, bI, h1e, h2e
                                )
                                * coeffs[I]
                            )
                            diag_new = diagonal_matrix_element_unrestricted(
                                new_a, new_b, h1e, h2e
                            )
                            denom = e_var_0 - diag_new
                            if abs(denom) < 1e-12:
                                continue
                            weight = coupling * coupling / abs(denom)
                            candidates[new_det] = _SelectionCandidate(
                                det=new_det, weight=weight, coupling=coupling
                            )

        # Sort by weight descending, filter by threshold
        sorted_cands = sorted(candidates.values(), key=lambda c: c.weight, reverse=True)
        _thr = (
            opts.pt2_threshold
            if pt2_threshold_override is None
            else pt2_threshold_override
        )
        selected = [c for c in sorted_cands if c.weight > _thr]
        max_new = max(
            10,
            min(
                opts.max_det_per_iter,
                int(len(det_space) * (opts.selection_growth_factor - 1.0)),
            ),
        )
        return selected[:max_new]

    def _compute_pt2(
        self,
        det_space: list,
        e_var: float,
        coeffs: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        opts: SelectedCIOptions,
        *,
        spin_restricted: bool = True,
    ) -> float:
        """Epstein-Nesbet PT2 correction from all singles/doubles outside space."""
        if spin_restricted:
            return self._compute_pt2_restricted(
                det_space, e_var, coeffs, h1e, h2e, norb
            )
        else:
            return self._compute_pt2_unrestricted(
                det_space, e_var, coeffs, h1e, h2e, norb
            )

    def _compute_pt2_restricted(
        self,
        det_space: list[Det],
        e_var: float,
        coeffs: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
    ) -> float:
        """Coherent EN-PT2 for spin-restricted (seniority-zero) determinants.

        Perturber numerators ``S_I H_aI c_I`` accumulate over ALL in-space
        generators before squaring (Sharma et al, JCTC 13, 1595 (2017),
        Eq. 4).  The pre-2026-06-11 implementation summed ``(c_I H_aI)^2``
        per (generator, perturber) pair, neglecting cross-generator
        interference in the numerator.
        """
        det_set = set(det_space)
        num: dict = {}

        for det_I, c_I in zip(det_space, coeffs):
            c_I = float(c_I)
            if c_I == 0.0:
                continue
            occ_set = set(det_I)
            vir_list = sorted(set(range(norb)) - occ_set)

            # Pair moves -- the only perturbers a closed-shell determinant
            # couples to (H_aI = g_{iiaa}; two-pair moves have zero coupling).
            for i in occ_set:
                for a in vir_list:
                    new_det = tuple(sorted((occ_set - {i}) | {a}))
                    if new_det in det_set:
                        continue
                    H_Ia = pair_excitation_matrix_element(i, a, h2e)
                    if H_Ia != 0.0:
                        num[new_det] = num.get(new_det, 0.0) + c_I * H_Ia

        e2 = 0.0
        for new_det, v in num.items():
            denom = e_var - diagonal_matrix_element(new_det, h1e, h2e)
            if abs(denom) > 1e-12:
                e2 += v * v / denom
        return e2

    def _compute_pt2_unrestricted(
        self,
        det_space: list[SpinDet],
        e_var: float,
        coeffs: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
    ) -> float:
        """Coherent EN-PT2 for SpinDet spaces.

        Routes through the shared module-level kernel
        (:func:`_en_pt2_deterministic_python`, the oracle of the C++
        ``selected_ci_en_pt2_deterministic``): coherent perturber
        numerators over all five excitation classes, Epstein-Nesbet
        denominators on the same (electronic) frame as ``e_var``.
        Replaces the pre-2026-06-11 per-(generator, perturber)
        incoherent sum.
        """
        e2, _ = _en_pt2_deterministic_python(
            list(det_space),
            np.asarray(coeffs, dtype=float),
            float(e_var),
            h1e,
            h2e,
            norb,
            0.0,
        )
        return e2

    def _davidson(
        self,
        H_mat: np.ndarray,
        det_space: list,
        h1e: np.ndarray,
        h2e: np.ndarray,
        opts: SelectedCIOptions,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Davidson iterative diagonalization using the pre-built H matrix
        for sigma-vector products: s = H @ v.

        Returns (eigenvalues[0:1], eigenvectors[:, 0:1]) mimicking scipy_eigh."""
        ndet = len(det_space)
        n_guess = min(4, ndet)
        max_iter = opts.davidson_max_iter
        tol = opts.davidson_conv_tol

        # Initial subspace: diagonal-preconditioned guess
        diag = np.diag(H_mat)
        idx = np.argsort(diag)[:n_guess]
        V = np.zeros((ndet, n_guess))
        for k, i in enumerate(idx):
            V[i, k] = 1.0

        # Orthonormalize initial guess
        V, _ = np.linalg.qr(V)

        for _ in range(max_iter):
            # Subspace Hamiltonian
            HV = H_mat @ V
            H_sub = V.T @ HV
            evals_sub, evecs_sub = np.linalg.eigh(H_sub)
            e_sub = evals_sub[0]
            c_sub = evecs_sub[:, 0]

            # Ritz vector and residual
            v = V @ c_sub
            sigma = HV @ c_sub
            r = sigma - e_sub * v
            r_norm = np.linalg.norm(r)

            if r_norm < tol:
                break

            # Precondition: (diag - e_sub)^{-1}
            denom = diag - e_sub
            denom[np.abs(denom) < 1e-12] = 1e-12
            t = r / denom

            # Orthogonalize against current subspace
            t = t - V @ (V.T @ t)
            t_norm = np.linalg.norm(t)
            if t_norm < 1e-14:
                break
            t = t / t_norm

            # Expand subspace
            V = np.column_stack([V, t])
            # Re-orthonormalize
            V, _ = np.linalg.qr(V)

        # Return in scipy_eigh-compatible format
        eigenvalues = np.array([e_sub])
        eigenvectors = v.reshape(ndet, 1)
        return eigenvalues, eigenvectors


def solve_selected_ci(
    hamiltonian: Hamiltonian,
    options: Optional[SelectedCIOptions] = None,
) -> SolverResult:
    """One-shot Selected-CI solve.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        One- and two-electron integrals in an orthonormal spatial-orbital basis.
    options : SelectedCIOptions, optional

    Returns
    -------
    SolverResult
    """
    solver = SelectedCISolver(options)
    return solver.solve(hamiltonian)


# ── Selected-CI as an active-space CASCI kernel (roadmap 25i) ───────────────
#
# :func:`selected_casci` mirrors :func:`vibeqc.solvers.casci`'s interface so
# the CASSCF macro-iteration (and anything else built on CASCIResult) can use
# a selected-CI wavefunction where the full determinant space is out of reach.
# The selected space is grown by the multi-root CIPSI criterion (Huron,
# Malrieu & Rancurel, J. Chem. Phys. 58, 5745 (1973)): a candidate
# determinant D outside the variational space is scored by its summed
# Epstein-Nesbet second-order estimate over the targeted roots,
#
#     w_D = S_k |<D|H|Ψ_k>|^2 / |E_k - <D|H|D>| ,
#
# and the top candidates above ``pt2_threshold`` enter the space; the
# Hamiltonian is re-diagonalized exactly in the enlarged space.  The energy
# is variational at every cycle (an eigenvalue of H projected on the selected
# space), which is what the CASSCF orbital gradient differentiates.
#
# Backends: the pure-Python engine below (the validation oracle) and the
# C++ kernel (cpp/src/selected_ci.cpp: sparse Slater-Condon H + block
# Davidson over uint64 SpinDet bitmasks, same selection criterion).
# ``VIBEQC_SELECTED_CI_BACKEND=auto|python|cpp``; auto keeps small full
# spaces on the Python path and dispatches beyond-dense-wall actives
# (full determinant count > _SELECTED_CI_CPP_THRESHOLD) to C++.

_SELECTED_CI_CPP_THRESHOLD = 100_000


def _dets_to_masks(dets):
    import numpy as _np

    a = _np.array([sum(1 << p for p in d[0]) for d in dets], dtype=_np.uint64)
    b = _np.array([sum(1 << p for p in d[1]) for d in dets], dtype=_np.uint64)
    return a, b


def _masks_to_dets(masks_a, masks_b, n_act):
    out = []
    for ma, mb in zip(masks_a, masks_b):
        ia, ib = int(ma), int(mb)
        out.append(
            (
                tuple(p for p in range(n_act) if (ia >> p) & 1),
                tuple(p for p in range(n_act) if (ib >> p) & 1),
            )
        )
    return out


def _connected_spin_dets(a_occ, b_occ, n_act):
    """Yield all SpinDets singly/doubly connected to ``(a_occ, b_occ)``.

    Five excitation classes: a/b singles, aa/bb/ab doubles.  Membership
    filtering and matrix elements are the caller's job.
    """
    occ_a, occ_b = set(a_occ), set(b_occ)
    vir_a = [p for p in range(n_act) if p not in occ_a]
    vir_b = [p for p in range(n_act) if p not in occ_b]
    a_list, b_list = sorted(occ_a), sorted(occ_b)

    def _sub(occ, out, inn):
        return tuple(sorted((occ - out) | inn))

    for i in a_list:  # a singles
        for a in vir_a:
            yield (_sub(occ_a, {i}, {a}), b_occ)
    for i in b_list:  # b singles
        for a in vir_b:
            yield (a_occ, _sub(occ_b, {i}, {a}))
    for ii in range(len(a_list)):  # aa doubles
        for jj in range(ii + 1, len(a_list)):
            for aa in range(len(vir_a)):
                for bb in range(aa + 1, len(vir_a)):
                    yield (
                        _sub(occ_a, {a_list[ii], a_list[jj]}, {vir_a[aa], vir_a[bb]}),
                        b_occ,
                    )
    for ii in range(len(b_list)):  # bb doubles
        for jj in range(ii + 1, len(b_list)):
            for aa in range(len(vir_b)):
                for bb in range(aa + 1, len(vir_b)):
                    yield (
                        a_occ,
                        _sub(occ_b, {b_list[ii], b_list[jj]}, {vir_b[aa], vir_b[bb]}),
                    )
    for i in a_list:  # ab doubles
        for j in b_list:
            for a in vir_a:
                for b in vir_b:
                    yield (
                        _sub(occ_a, {i}, {a}),
                        _sub(occ_b, {j}, {b}),
                    )


def _cipsi_candidates(dets, ci_cols, e_roots, h1a, h2a, n_act, opts):
    """Multi-root CIPSI selection scores for determinants outside the space.

    Accumulates the *coherent* coupling ``<D|H|Ψ_k> = S_I H_DI c_I^k`` over
    every significant variational determinant I (not just the first finder),
    then scores ``w_D = S_k |<D|H|Ψ_k>|^2 / |E_k - H_DD|`` (Epstein-Nesbet
    denominators).  Returns ``[(w_D, D), ...]`` sorted descending.
    """
    det_set = set(dets)
    nroots = ci_cols.shape[1]
    c_abs = np.abs(ci_cols).max(axis=1)
    significant = np.where(c_abs > opts.significant_coeff)[0]
    if len(significant) == 0:
        significant = [int(np.argmax(c_abs))]

    # coherent numerators per candidate: D -> S_I H_DI c_I^k  (k = root)
    select_eps = float(getattr(opts, "select_eps", 0.0) or 0.0)
    num: dict = {}
    for idx in significant:
        a_i, b_i = dets[idx]
        c_row = ci_cols[idx]
        cmax = c_abs[idx]
        for cand in _connected_spin_dets(a_i, b_i, n_act):
            if cand in det_set:
                continue
            h_di = hamiltonian_matrix_element_unrestricted(
                cand[0], cand[1], a_i, b_i, h1a, h2a
            )
            if h_di == 0.0:
                continue
            # Heat-bath prefilter: same predicate (and the same float
            # expression) as the C++ kernel's presorted walk.
            if select_eps > 0.0 and abs(h_di) * cmax < select_eps:
                continue
            acc = num.get(cand)
            if acc is None:
                num[cand] = h_di * c_row
            else:
                num[cand] = acc + h_di * c_row

    scored = []
    for cand, vec in num.items():
        h_dd = diagonal_matrix_element_unrestricted(cand[0], cand[1], h1a, h2a)
        w = 0.0
        for k in range(nroots):
            denom = abs(e_roots[k] - h_dd)
            if denom < 1e-10:
                denom = 1e-10
            w += vec[k] * vec[k] / denom
        if w > opts.pt2_threshold:
            scored.append((w, cand))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _spin_partner(det):
    """The alpha/beta-swapped partner of an M_s = 0 SpinDet."""
    return (det[1], det[0])


def selected_casci(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_active_elec: int,
    n_active_orb: int,
    n_core: int = 0,
    nuclear_repulsion: float = 0.0,
    e_hf: float = 0.0,
    ms2: Optional[int] = None,
    *,
    nroots: int = 1,
    options: Optional[SelectedCIOptions] = None,
    det_guess: Optional[list] = None,
):
    """Selected-CI in an active space, with :func:`casci`-compatible output.

    Drop-in alternative to :func:`vibeqc.solvers.casci` for active spaces
    beyond the determinant-CI wall: the variational space is a *selected*
    subset of the CAS determinants, grown by the multi-root CIPSI criterion
    (module notes above) until the energy is stable, no candidate scores
    above ``options.pt2_threshold``, or ``options.target_size`` is reached.
    The returned energies are variational (exact eigenvalues of H in the
    selected space), so 1-/2-RDMs built from the returned wavefunction
    (:func:`vibeqc.solvers.make_rdm12` handles truncated determinant lists)
    reproduce the energy exactly, the property the CASSCF orbital gradient
    needs.

    Parameters mirror :func:`casci`; extras:

    Parameters
    ----------
    options : SelectedCIOptions, optional
        ``target_size`` / ``max_iter`` / ``conv_tol_energy`` /
        ``pt2_threshold`` / ``max_det_per_iter`` / ``significant_coeff`` /
        ``spin_complete`` / ``select_eps`` are honored (the legacy
        full-space solver knobs ``spin_restricted``, ``davidson_*``,
        ``do_pt2_correction`` are not used by this kernel).
    det_guess : list[SpinDet], optional
        Starting selected space (e.g. the previous CASSCF macro-iteration's
        list; the selected set is a good space across small orbital
        rotations).  Default: the aufbau reference determinant (plus its
        single excitations when ``nroots > 1``).

    Returns
    -------
    CASCIResult
        ``determinants`` is the selected SpinDet list (insertion order),
        ``n_det`` its size; ``e_totals`` / ``ci_coeffs_all`` follow the
        ``nroots`` convention of :func:`casci`.
    """
    from math import comb

    from ._casci import CASCIResult, _frozen_core_dressing

    opts = options or SelectedCIOptions()
    norb_total = h1e_mo.shape[0]
    if n_core + n_active_orb > norb_total:
        raise ValueError(
            f"n_core ({n_core}) + n_active_orb ({n_active_orb}) > norb ({norb_total})"
        )
    if ms2 is None:
        ms2 = n_active_elec % 2
    n_alpha = (n_active_elec + ms2) // 2
    n_beta = n_active_elec - n_alpha
    n_act = n_active_orb
    if n_alpha < 0 or n_beta < 0 or n_alpha > n_act or n_beta > n_act:
        raise ValueError(
            f"Cannot place {n_active_elec} electrons (ms2={ms2}) in "
            f"{n_act} active orbitals"
        )
    n_det_full = comb(n_act, n_alpha) * comb(n_act, n_beta)
    if nroots < 1 or nroots > n_det_full:
        raise ValueError(f"nroots ({nroots}) must be in [1, {n_det_full}]")

    active = slice(n_core, n_core + n_act)
    e_core, h1a = _frozen_core_dressing(h1e_mo, h2e_mo, n_core, active)
    h2a = np.ascontiguousarray(h2e_mo[active, active, active, active])

    # ── backend dispatch (see module notes above) ────────────────────────
    import os

    backend = os.environ.get("VIBEQC_SELECTED_CI_BACKEND", "auto")
    if backend not in ("auto", "cpp", "python"):
        raise ValueError(
            f"VIBEQC_SELECTED_CI_BACKEND must be auto|cpp|python, got {backend!r}"
        )
    use_cpp = backend == "cpp" or (
        backend == "auto" and n_det_full > _SELECTED_CI_CPP_THRESHOLD
    )
    if use_cpp:
        try:
            from .._vibeqc_core import (
                SelectedCIOptionsCpp as _CppOpts,
            )
            from .._vibeqc_core import (
                selected_ci_solve as _cpp_solve,
            )
        except ImportError:
            use_cpp = False
            if backend == "cpp":
                raise
    if use_cpp:
        copts = _CppOpts()
        copts.nroots = nroots
        copts.max_cycles = opts.max_iter
        copts.target_size = opts.target_size
        copts.max_new_per_cycle = opts.max_det_per_iter
        copts.conv_tol_energy = opts.conv_tol_energy
        copts.pt2_threshold = opts.pt2_threshold
        copts.significant_coeff = opts.significant_coeff
        copts.spin_complete = bool(opts.spin_complete)
        copts.select_eps = float(opts.select_eps)
        if det_guess:
            ga, gb = _dets_to_masks(list(dict.fromkeys(tuple(d) for d in det_guess)))
        else:
            ga = np.empty(0, dtype=np.uint64)
            gb = np.empty(0, dtype=np.uint64)
        eri_chem = np.ascontiguousarray(h2a.transpose(0, 2, 1, 3))
        direct = _cpp_solve(
            np.ascontiguousarray(h1a),
            eri_chem,
            n_act,
            n_alpha,
            n_beta,
            copts,
            ga,
            gb,
        )
        e_const = e_core + nuclear_repulsion
        e_vals = np.asarray(direct.eigenvalues)
        ci_mat = np.asarray(direct.ci)
        e_totals = [float(e) + e_const for e in e_vals[:nroots]]
        dets_out = _masks_to_dets(direct.dets_a, direct.dets_b, n_act)
        return CASCIResult(
            e_total=e_totals[0],
            e_corr=e_totals[0] - e_hf,
            ci_coeffs=np.ascontiguousarray(ci_mat[:, 0]),
            determinants=dets_out,
            n_det=len(dets_out),
            n_active_orb=n_act,
            e_core=e_const,
            nroots=nroots,
            e_totals=e_totals,
            ci_coeffs_all=(
                np.ascontiguousarray(ci_mat[:, :nroots]) if nroots > 1 else None
            ),
        )

    spin_complete = bool(opts.spin_complete) and ms2 == 0

    def _completed(det_list):
        if not spin_complete:
            return det_list
        out = list(det_list)
        seen = set(det_list)
        for d in det_list:
            p = _spin_partner(d)
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    # ── starting space ───────────────────────────────────────────────────
    if det_guess:
        dets = list(dict.fromkeys(tuple(d) for d in det_guess))
        for a_occ, b_occ in dets:
            if len(a_occ) != n_alpha or len(b_occ) != n_beta:
                raise ValueError(
                    "det_guess electron counts do not match the active space"
                )
    else:
        ref_a = tuple(range(n_alpha))
        ref_b = tuple(range(n_beta))
        dets = [(ref_a, ref_b)]
        if nroots > 1:
            # seed enough variational freedom for the requested roots:
            # all single excitations of the reference (cheap, and the
            # selection grows the rest)
            singles = []
            occ_a, occ_b = set(ref_a), set(ref_b)
            for i in sorted(occ_a):
                for a in range(n_act):
                    if a not in occ_a:
                        singles.append((tuple(sorted((occ_a - {i}) | {a})), ref_b))
            for i in sorted(occ_b):
                for a in range(n_act):
                    if a not in occ_b:
                        singles.append((ref_a, tuple(sorted((occ_b - {i}) | {a}))))
            dets += singles
    dets = _completed(dets)
    if len(dets) < nroots:
        raise ValueError(
            f"starting selected space ({len(dets)} determinants) is smaller "
            f"than nroots ({nroots}); supply a det_guess"
        )

    # ── grow-and-diagonalize loop ────────────────────────────────────────
    e_prev = None
    for cycle in range(opts.max_iter):
        H = build_hamiltonian_matrix_unrestricted(dets, h1a, h2a)
        w, V = scipy_eigh(H)
        e_roots = w[:nroots]
        ci_cols = V[:, :nroots]

        if e_prev is not None:
            if float(np.max(np.abs(e_roots - e_prev))) < opts.conv_tol_energy:
                break
        e_prev = e_roots.copy()

        if len(dets) >= opts.target_size or len(dets) >= n_det_full:
            break
        if cycle == opts.max_iter - 1:
            # no growth on the final cycle: the returned wavefunction must
            # live in the space that was actually diagonalized (max_iter=1
            # is the "frozen selection" solve the CASSCF FD probes use)
            break

        scored = _cipsi_candidates(dets, ci_cols, e_roots, h1a, h2a, n_act, opts)
        if not scored:
            break
        room = min(
            opts.max_det_per_iter,
            opts.target_size - len(dets),
        )
        new = [d for _, d in scored[:room]]
        dets.extend(new)
        if spin_complete:
            dets = _completed(dets)

    e_const = e_core + nuclear_repulsion
    e_totals = [float(e) + e_const for e in e_roots]
    return CASCIResult(
        e_total=e_totals[0],
        e_corr=e_totals[0] - e_hf,
        ci_coeffs=np.ascontiguousarray(ci_cols[:, 0]),
        determinants=dets,
        n_det=len(dets),
        n_active_orb=n_act,
        e_core=e_const,
        nroots=nroots,
        e_totals=e_totals,
        ci_coeffs_all=(np.ascontiguousarray(ci_cols) if nroots > 1 else None),
    )


#: <S^2> classification tolerance for selected (truncated) CI roots.  The
#: dense path (:func:`._ms_caspt2._spin_pure_roots`) uses 1e-4: its roots
#: are exact eigenvectors, spin-pure to solver precision.  A truncated
#: selected space breaks S^2 symmetry, so eigenvector <S^2> drifts from
#: S(S+1) by O(truncation).  With ``spin_complete`` (ab-swap closure,
#: the default at M_s=0) the space is invariant under the spin-flip
#: operator, whose eigenvalue on an |S, M_s=0> state is (-1)^S: even-S
#: and odd-S roots cannot mix, so the adjacent-sector channel
#: (singlet-triplet) is symmetry-blocked and the residual drift comes
#: from sectors >= 2 away (ΔS^2 >= 6).  0.25 accepts that drift while
#: staying far inside the S(S+1) inter-sector gap of 2.
_SPIN_PURE_S2_TOL = 0.25


def _spin_pure_selected_roots(
    h1e_mo,
    h2e_mo,
    n_active_elec,
    n_active_orb,
    n_core=0,
    nuclear_repulsion=0.0,
    ms2=None,
    *,
    nroots=1,
    options=None,
    det_guess=None,
):
    """Lowest ``nroots`` spin-pure roots (S = ms2/2) of a selected CI.

    Selected-space analogue of :func:`._ms_caspt2._spin_pure_roots`:
    solve :func:`selected_casci` with a root buffer (the fixed-M_s
    determinant sector interleaves higher-S roots a CSF code never
    sees), classify each root by <S^2> (evaluated exactly for the
    returned vector via ``|S₊psi|^2 + M_s(M_s+1)`` on the selected
    determinant list) and keep the first ``nroots`` with
    S(S+1) ≈ (ms2/2)(ms2/2+1) (tolerance :data:`_SPIN_PURE_S2_TOL`).

    The CIPSI growth targets the buffered root set coherently, so part
    of the selection budget describes the to-be-discarded higher-spin
    roots; at the full-selection limit the result is identical to the
    dense filter.  When the natural starting space (``det_guess``, or
    aufbau + singles) is smaller than the root buffer, it is extended
    by excitation-connected determinants (BFS over singles + doubles)
    until the buffered solve is well-posed; with ``max_iter=1``
    (frozen-selection solves, e.g. CASSCF FD Hessian probes) the
    warm-start list is never smaller than the buffer in practice, so
    the frozen space is exactly the list that was passed in.

    Returns ``(ci_cols (n_det, nroots), energies, s2s, casci_result)``
    where ``casci_result`` is the buffered (unfiltered) result of the
    final solve, mirroring the dense helper's return shape.
    """
    from math import comb

    from ._ms_caspt2 import _s2_expectation

    if ms2 is None:
        ms2 = n_active_elec % 2
    n_act = n_active_orb
    n_alpha = (n_active_elec + ms2) // 2
    n_beta = n_active_elec - n_alpha
    n_det_full = comb(n_act, n_alpha) * comb(n_act, n_beta)
    s_target = 0.5 * ms2
    s2_target = s_target * (s_target + 1.0)

    opts = options or SelectedCIOptions()

    # At the full-selection limit (target_size >= full CAS and tight
    # pt2 threshold), delegate to the dense spin-pure solver.  The
    # selected-CI eigensolver converges to different eigenvectors
    # than the dense solver even with all determinants, causing SA-RDM
    # differences that break CASSCF convergence (P1 gate red).
    if opts.target_size >= n_det_full and opts.pt2_threshold <= 1e-14:
        from ._ms_caspt2 import _spin_pure_roots as _dense_spin_pure
        return _dense_spin_pure(
            h1e_mo, h2e_mo, n_core, n_act, n_active_elec, ms2, nroots,
            nuclear_repulsion=nuclear_repulsion, ci_guess=det_guess)

    spin_complete = bool(opts.spin_complete) and ms2 == 0

    def _completed(det_list):
        if not spin_complete:
            return det_list
        out = list(det_list)
        seen = set(out)
        for d in det_list:
            p = _spin_partner(d)
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def _seed_for(n_min):
        """A starting space of >= ``n_min`` determinants (<= full space)."""
        if det_guess:
            seed = list(dict.fromkeys(tuple(d) for d in det_guess))
        else:
            ref_a = tuple(range(n_alpha))
            ref_b = tuple(range(n_beta))
            seed = [(ref_a, ref_b)]
        seed = _completed(seed)
        seen = set(seed)
        frontier = list(seed)
        while len(seed) < n_min and frontier:
            new = []
            for a_occ, b_occ in frontier:
                for c in _connected_spin_dets(a_occ, b_occ, n_act):
                    if c not in seen:
                        seen.add(c)
                        new.append(c)
            if spin_complete:
                for d in list(new):
                    p = _spin_partner(d)
                    if p not in seen:
                        seen.add(p)
                        new.append(p)
            seed.extend(new)
            frontier = new
        return seed

    n_buf = min(max(2 * nroots + 4, nroots), n_det_full)
    attempts = 0
    while True:
        res = selected_casci(
            h1e_mo,
            h2e_mo,
            n_active_elec,
            n_act,
            n_core,
            nuclear_repulsion=nuclear_repulsion,
            ms2=ms2,
            nroots=n_buf,
            options=opts,
            det_guess=_seed_for(n_buf),
        )
        cols = res.ci_coeffs_all if n_buf > 1 else res.ci_coeffs[:, None]
        sel, s2s = [], []
        s2s_all = []
        for k in range(n_buf):
            s2 = _s2_expectation(cols[:, k], res.determinants, n_act, ms2)
            s2s_all.append(s2)
            if abs(s2 - s2_target) < _SPIN_PURE_S2_TOL:
                sel.append(k)
                s2s.append(s2)
            if len(sel) == nroots:
                break
        if len(sel) == nroots:
            ci_cols = np.ascontiguousarray(cols[:, sel])
            energies = [res.e_totals[k] for k in sel]
            return ci_cols, energies, s2s, res
        attempts += 1
        next_buf = min(2 * n_buf, n_det_full)
        if attempts >= 3 or next_buf == n_buf or next_buf > res.n_det:
            raise ValueError(
                f"only {len(sel)} spin-pure roots with S={s_target:g} "
                f"found among the lowest {n_buf} roots of the selected "
                f"space ({res.n_det} determinants); requested "
                f"nroots={nroots}.  Root <S^2> values: "
                f"{[round(s, 4) for s in s2s_all]}.  Widen the selection "
                f"(target_size / pt2_threshold) or reduce nroots."
            )
        # Reuse the grown selected space as the next, larger-buffer seed.
        det_guess = list(res.determinants)
        n_buf = next_buf


@dataclass
class SelectedCIPT2Options:
    """Options for the Epstein-Nesbet PT2 stage on a selected wavefunction.

    Mirrors :func:`selected_ci_pt2`'s knobs for the ``run_job`` surface
    (``CASSCFOptions(pt2=SelectedCIPT2Options(...))`` with
    ``ci_solver="selected_ci"``).  ``n_samples = 0`` is the fully
    deterministic mode at ``eps2``; ``n_samples >= 2`` is the
    semistochastic mode (deterministic at ``eps2_loose`` plus the
    sampled unbiased difference down to ``eps2``).
    """

    eps2: float = 1e-8
    n_samples: int = 0
    sample_size: int = 200
    eps2_loose: float = 1e-6
    seed: int = 0


def _en_pt2_deterministic_python(dets, ci, e0, h1a, h2a, n_act, eps2=0.0):
    """Brute-force coherent Epstein-Nesbet PT2: the oracle for the C++ kernel.

    Eq. 5 of Sharma, Holmes, Jeanmairet, Alavi & Umrigar, J. Chem.
    Theory Comput. 13, 1595 (2017), doi:10.1021/acs.jctc.6b01028:

        ΔE₂ = S_a ( S_i^{(e₂)} H_ai c_i )^2 / (E₀ - H_aa)

    over perturbers ``a`` outside the selected space, the inner sum
    keeping contributions with |H_ai c_i| >= ``eps2`` (the same
    drop-when-< convention as the selection prefilter and the C++
    walk).  ``e0`` is the root's ACTIVE-SPACE variational eigenvalue
    (the frame of ``h1a``); perturbers with |E₀ - H_aa| < 1e-10 are
    skipped (intruder guard, matching the C++ kernel).  NOTE the
    coherent numerator: contributions from every generator I sum
    BEFORE squaring.  :class:`SelectedCISolver`'s ``do_pt2_correction``
    routes through this kernel too (migrated from a per-(generator,
    perturber) incoherent sum on 2026-06-11).  Returns
    ``(e2, n_perturbers)``.
    """
    det_set = set(dets)
    num: dict = {}
    for (a_i, b_i), c in zip(dets, ci):
        c = float(c)
        if c == 0.0:
            continue
        for cand in _connected_spin_dets(a_i, b_i, n_act):
            if cand in det_set:
                continue
            h = hamiltonian_matrix_element_unrestricted(
                cand[0], cand[1], a_i, b_i, h1a, h2a
            )
            if h == 0.0:
                continue
            if eps2 > 0.0 and abs(h) * abs(c) < eps2:
                continue
            num[cand] = num.get(cand, 0.0) + h * c
    e2 = 0.0
    for cand, v in num.items():
        denom = e0 - diagonal_matrix_element_unrestricted(cand[0], cand[1], h1a, h2a)
        if abs(denom) < 1e-10:
            continue
        e2 += v * v / denom
    return e2, len(num)


def selected_ci_pt2(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_active_elec: int,
    n_active_orb: int,
    n_core: int = 0,
    *,
    result,
    eps2: float = 1e-8,
    n_samples: int = 0,
    sample_size: int = 200,
    eps2_loose: float = 1e-6,
    seed: int = 0,
):
    """Epstein-Nesbet PT2 correction(s) for a selected-CI wavefunction.

    Post-processing on a :func:`selected_casci` result (SHCI's
    perturbative stage: Sharma, Holmes, Jeanmairet, Alavi & Umrigar,
    JCTC 13, 1595 (2017)).  Per root, the second-order correction over
    perturbers outside the selected space:

    * ``n_samples = 0`` (default): fully deterministic (Eq. 5), with
      contributions screened at |H_ai c_i| >= ``eps2`` and the
      double-excitation enumeration pruned by the heat-bath walk.  The
      perturber map holds every surviving perturber: memory grows as
      eps2 shrinks (the SHCI memory bottleneck the stochastic mode
      removes).
    * ``n_samples >= 2``: semistochastic (Eq. 11) -- deterministic part
      at the loose threshold ``eps2_loose`` plus the sampled unbiased
      difference estimator (Eq. 10) of E2[eps2] - E2[eps2_loose], with
      ``n_samples`` batches of ``sample_size`` determinants drawn with
      replacement from p_i ∝ |c_i| (Eq. 7).  ``eps2_loose == eps2``
      yields zero stochastic noise (the difference vanishes
      batch-wise); ``eps2_loose = float("inf")`` is the fully
      stochastic estimate.  C++ backend only.

    Parameters mirror :func:`selected_casci` for the integral/active
    frame; ``result`` is the converged CASCIResult whose
    ``determinants`` / ``ci_coeffs(_all)`` / ``e_totals`` / ``e_core``
    are used.  Returns a list (one dict per root) with keys
    ``e_pt2``, ``e_total`` (variational + PT2, full-frame),
    ``stderr`` (0.0 for deterministic runs) and ``n_perturbers``
    (deterministic perturber count; None for stochastic batches).

    Backend: ``VIBEQC_SELECTED_CI_BACKEND`` -- ``auto`` uses C++ when
    built (PT2 is enumeration-bound; the Python path is the
    machine-precision oracle for small systems and rejects the
    stochastic mode).
    """
    import os

    from ._casci import _frozen_core_dressing

    if n_samples != 0 and n_samples < 2:
        raise ValueError("n_samples must be 0 (deterministic) or >= 2")
    if n_samples and not (eps2_loose >= eps2):
        raise ValueError("eps2_loose must be >= eps2")

    n_act = n_active_orb
    active = slice(n_core, n_core + n_act)
    _, h1a = _frozen_core_dressing(h1e_mo, h2e_mo, n_core, active)
    h1a = np.ascontiguousarray(h1a)
    h2a = np.ascontiguousarray(h2e_mo[active, active, active, active])

    backend = os.environ.get("VIBEQC_SELECTED_CI_BACKEND", "auto")
    if backend not in ("auto", "cpp", "python"):
        raise ValueError(
            f"VIBEQC_SELECTED_CI_BACKEND must be auto|cpp|python, got {backend!r}"
        )
    use_cpp = backend in ("auto", "cpp")
    if use_cpp:
        try:
            from .._vibeqc_core import (  # noqa: F401
                selected_ci_en_pt2_deterministic as _cpp_det,
            )
            from .._vibeqc_core import (
                selected_ci_en_pt2_stochastic as _cpp_stoch,
            )
        except ImportError:
            use_cpp = False
            if backend == "cpp":
                raise
    if not use_cpp and n_samples:
        raise NotImplementedError(
            "the semistochastic PT2 mode is C++-backend only; the "
            "Python oracle covers the deterministic limit"
        )

    nroots = result.nroots
    cols = (
        result.ci_coeffs_all
        if result.ci_coeffs_all is not None
        else result.ci_coeffs[:, None]
    )
    dets = list(result.determinants)
    out = []
    if use_cpp:
        ma, mb = _dets_to_masks(dets)
        eri_chem = np.ascontiguousarray(h2a.transpose(0, 2, 1, 3))
    for k in range(nroots):
        ci_k = np.ascontiguousarray(cols[:, k], dtype=float)
        e0 = float(result.e_totals[k]) - float(result.e_core)
        if use_cpp:
            if n_samples:
                if np.isfinite(eps2_loose):
                    e2_loose, n_pert = _cpp_det(
                        h1a,
                        eri_chem,
                        n_act,
                        ma,
                        mb,
                        ci_k,
                        e0,
                        eps2_loose,
                        True,
                    )
                else:  # fully stochastic: no deterministic part
                    e2_loose, n_pert = 0.0, None
                mean, sem = _cpp_stoch(
                    h1a,
                    eri_chem,
                    n_act,
                    ma,
                    mb,
                    ci_k,
                    e0,
                    eps2,
                    eps2_loose,
                    n_samples,
                    sample_size,
                    seed,
                    True,
                )
                e2 = e2_loose + mean
                out.append(
                    dict(
                        e_pt2=e2,
                        e_total=float(result.e_totals[k]) + e2,
                        stderr=sem,
                        n_perturbers=n_pert,
                    )
                )
            else:
                e2, n_pert = _cpp_det(
                    h1a, eri_chem, n_act, ma, mb, ci_k, e0, eps2, True
                )
                out.append(
                    dict(
                        e_pt2=e2,
                        e_total=float(result.e_totals[k]) + e2,
                        stderr=0.0,
                        n_perturbers=n_pert,
                    )
                )
        else:
            e2, n_pert = _en_pt2_deterministic_python(
                dets, ci_k, e0, h1a, h2a, n_act, eps2
            )
            out.append(
                dict(
                    e_pt2=e2,
                    e_total=float(result.e_totals[k]) + e2,
                    stderr=0.0,
                    n_perturbers=n_pert,
                )
            )
    return out
