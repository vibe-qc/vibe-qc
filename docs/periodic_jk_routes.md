# Periodic JK routes, method-family parity policy

vibe-qc ships several periodic Coulomb/exchange (JK) routes (see the
`jk_method` table in
[`docs/user_guide/periodic_methods.md`](user_guide/periodic_methods.md)).
Each is its **own** implementation (CLAUDE.md §10), we never import an
external QC program to compute energies. We *do* validate each route against
an external program out-of-process, and **which** program matters.

## The rule

> **Validate each periodic JK route against a program from the same
> method family, at matched settings. Do not cross-validate against a
> program from a different family and treat the difference as a bug.**

For the CCM family, Γ-CCM and χ-CCM[^xccm-convention] are distinct
union-and-weight and finite-translation-group character approaches. They are
compared at a declared common exchange-q=0 convention, which is a comparison
constraint rather than an identification of the constructions.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.

| vibe-qc route | method family | parity reference | why same-family |
|---|---|---|---|
| **GDF** (`jk_method='gdf'`) | Gaussian density fitting | **PySCF** `pbc.df.GDF` / KRHF·KRKS `density_fit()` | Both fit the periodic density into an auxiliary Gaussian basis and build J/K from the fit. The exchange divergence is handled the same way (`exxdiv='ewald'` ↔ vibe-qc's Γ Ewald-exchange correction), so the residual is a genuine implementation diff, resolvable to µHa. |
| **Γ-CCM / AICCM2026DEV-A** (`run_periodic_job(method="aiccm", variant="four-center")`; library `method='aiccm2026dev-a'`) | union-and-weight/Wigner--Seitz integral-weighting CCM; HF/KS/MP2/CCSD(T) on the finite BvK torus; the runner supports 3-D RHF/RKS/UHF/UKS | Construction-specific internal gates; **CRYSTAL23** for the infinite-size limit out-of-process | Any finite-size equality with a periodic route is a specified-operator validation target, not a consequence of the Γ-CCM name. CRYSTAL23 is the external oracle for the thermodynamic limit at matched basis and k-mesh. |
| **χ-CCM / AICCM2026DEV-B** (`run_periodic_job(method="aiccm", variant="chi")`; legacy `jk_method='aiccm2026dev-b'` warns) | finite-translation-group character CCM on a finite BvK torus, translation-invariant RHF/RKS; four-centre, RI, or RIJCOSX | Exact character/real-Gamma Fourier identity for the same χ-defined Hamiltonian; **CRYSTAL** or **PySCF** only out of process with the same finite-size exchange convention | The χ-internal Fourier identity is mathematical and must hold at every cluster size. It is not Γ-CCM construction evidence. External dense-mesh agreement tests the infinite-size limit only after Coulomb and `exxdiv` conventions are matched. |
| **BIPOLE** (`jk_method='bipole'`) | exact erfc short-range J/K plus reciprocal AO-pair-FT long-range J in one Ewald gauge | **CRYSTAL** (CRYSTAL14/17/23) as an out-of-process component and convergence reference | The supported vibe-qc route does not execute CRYSTAL's bipolar quartet replacement, so matching `TOLINTEG` and `SHRINK` does not make the algorithms identical. Compare converged total energies only after separately matching the electrostatic gauge, basis, k mesh, smearing, exchange convention, and real/reciprocal numerical supports. |
| **GPW / GAPW** (`jk_method='gpw'`/`'gapw'`) | Gaussian-and-plane-waves (Lippert-Hutter / Krack-Parrinello) | **CP2K** (Quickstep) / **GPAW** | Both collocate the Gaussian density on a smooth real-space grid and solve Poisson by FFT, with per-atom augmentation for all-electron accuracy (GAPW ↔ CP2K GAPW / GPAW PAW). |

For any already specified block-circulant finite Hamiltonian, including a
Γ-CCM Hamiltonian when its assembled operator has that structure, complete
character/Bloch and real-Gamma evaluations are connected by the finite Fourier
transform. That internal representation identity validates an evaluation of
the specified Hamiltonian; it does not equate the union-and-weight and
finite-character constructions.

> **Retired:** `jk_method='fft_poisson'` (Γ-only EWALD_3D), see
> [`docs/user_guide/periodic_methods.md`](user_guide/periodic_methods.md).
> It returned a wrong energy when periodic AO images overlap and is
> superseded by GDF/BIPOLE/GPW; the internal Γ-only EWALD drivers remain for
> dilute periodic systems and fail closed on the image-overlap regime.

## Why cross-family parity is a trap

Comparing **BIPOLE against PySCF** (different families) conflates two
unrelated things:

1. **Method differences that are correct on both sides.** PySCF's GDF uses
   an auxiliary-basis density fit with a specific `exxdiv` finite-size
   exchange correction at Γ. CRYSTAL can use a direct-space bipolar quartet
   replacement, while the supported vibe-qc BIPOLE route uses exact
   erfc-screened short-range ERIs plus reciprocal AO-pair Fourier transforms.
   At a given **k**-mesh these can legitimately differ through their distinct
   numerical supports and finite-size conventions.
2. **Real bugs.** A genuine gauge or image-summing error in BIPOLE.

When (1) and (2) are mixed, a real bug hides inside the "expected"
cross-family spread, and a benign method difference looks like a regression.
The current BIPOLE/CRYSTAL comparison is therefore not a same-algorithm
identity test. Matched `TOLINTEG`, `SHRINK`, `SMEAR`, `FMIXING`, and
`LEVSHIFT` are necessary but not sufficient; the electrostatic gauge,
exchange convention, and each real/reciprocal support must also be audited.
Component comparisons remain useful, but an unexplained total residual cannot
be classified as implementation error from the route labels alone.

## How parity runs, the §10 subprocess pattern

External programs are executed **out-of-process** and their output parsed;
nothing under `python/vibeqc/` or `cpp/` imports them (CLAUDE.md §10). The
runners live at `examples/regression/core/runner_<program>.py`:

* `runner_pyscf.py`, spawns PySCF in a child interpreter, parses the
  `VIBEQC-PYSCF-RESULT:` JSON marker. Reference for **GDF**.
* `runner_crystal.py`, dispatches a CRYSTAL `.d12` via `vq submit` to a
  compute host, polls, fetches the workspace, and parses
  `TOTAL ENERGY(HF|DFT)(AU)` from the `.out`. Reference for **BIPOLE**.
  Needs `VIBEQC_CRYSTAL_WRAPPER` (path to `run-crystal.sh`) and a host with a
  CRYSTAL binary; heavy runs go via `vq` payloads, never on the laptop
  (CLAUDE.md §15).
* `runner_cp2k.py` / `runner_gpaw.py`, reference for **GPW/GAPW**.

CRYSTAL reference decks live in
[`examples/crystal23_demo/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/crystal23_demo/) (e.g.
`mgo_sto3g.d12`, `diamond_sto3g.d12`, `be_sto3g.d12`,
`graphite_sto3g.d12`) and the parity harness + sealed `.out` references in
[`examples/regression/crystal_parity/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/crystal_parity/).
MgO rocksalt is the BIPOLE flagship anchor; diamond, graphite, and Be extend
coverage to covalent, 2-D, and metallic (smearing) cells.

## Primary gate vs secondary cross-check

* The **primary** parity gate for a route is its same-family reference at
  matched settings (the row above).
* A **cross-family** comparison (e.g. BIPOLE vs PySCF, or GDF vs CRYSTAL) may
  still be informative as a *secondary* sanity check, but it must be
  **labelled as such** in the test/example and must **not** be the gating
  assertion, its tolerance is the method spread, not a bug bound.

## Current status (2026-06-16)

* **GDF ↔ PySCF**, in place, correct family: the µHa-validated `pbc.df.GDF`
  pins (`tests/test_pbc_gdf_*`). Unchanged by the re-basing.
* **BIPOLE PySCF pins relabeled**, the `exxdiv='ewald'` Γ pins in
  `tests/test_bipole_fock_ewald_exchange.py` are now marked a **secondary
  cross-family check** of BIPOLE's Γ exchange-split *convention*, not its
  primary parity gate.
* **BIPOLE ↔ CRYSTAL, finding (2026-06-16): the "matched-k" SHRINK-2-2
  comparison is exchange-convention-confounded. BIPOLE matches CRYSTAL's
  *k-converged* value, NOT its coarse-mesh value.**
  * *§10 oracle validated locally.* A local CRYSTAL23 binary (via the
    `run-crystal.sh` wrapper, out-of-process per §10, no `vq` needed)
    reproduces the sealed MgO SHRINK-8-8 reference to all digits
    (−271.21814375 Ha/FU), and the sealed SHRINK-2-2 reference is
    **−271.85640 Ha/FU** (3 IBZ k-points).
  * *Converged BIPOLE result (this run).* A fold-converged BIPOLE multi-k
    SCF, MgO/STO-3G, **SHRINK 2 2** (full 8-k mesh), corrected exchange
    gauge (`exchange_exxdiv='ewald'`), `cutoff_bohr = 12`, from SAD, 10 iters,
    metric `Σ_k w_k Tr[D(k)S(k)] = 20.000` (= N_elec → physical basin, not the
    cutoff-8 metric-invalid state), gives **E = −271.21530 Ha/FU**:

    | reference | Δ (BIPOLE − ref) |
    |---|---|
    | CRYSTAL **SHRINK 2 2** (−271.85640, the matched-k seal) | **+641.1 mHa** |
    | PySCF KRHF [2,2,2] `exxdiv='ewald'` (−271.213356) | −1.9 mHa |
    | CRYSTAL **SHRINK 8 8**, k-converged (−271.21814) | **+2.9 mHa** |

  * *Why BIPOLE ≠ CRYSTAL SHRINK 2 2, not a bug, an exchange finite-size
    **convention** difference.* BIPOLE's corrected gauge applies the
    probe-charge Ewald (Madelung) finite-size exchange correction
    `(ξ_M − π/(V_sc·ω²))·S(k)D(k)S(k)` (PySCF-equivalent `exxdiv='ewald'`),
    which removes the leading 1/N_k^{1/3} exchange finite-size error, so
    BIPOLE's SHRINK-2-2 energy is **already k-converged**. CRYSTAL applies
    **no** such correction: its SHRINK-2-2 exchange samples the q→0 term at the
    coarse mesh's smallest |q| and over-binds its own converged SHRINK-8-8
    value by 638 mHa. Hence BIPOLE-coarse ≈ converged ≈ CRYSTAL-SHRINK-8-8 (to
    ~3 mHa, STO-3G truncation scale), while CRYSTAL-coarse is the outlier.
  * *Methodology correction (maintainer-confirmed 2026-06-16).*
    The matched-k SHRINK-2-2-vs-SHRINK-2-2 premise (seal `5a443377`) assumed
    BIPOLE-coarse over-binds like CRYSTAL-coarse, so the k-incompleteness would
    cancel. It does not, BIPOLE-coarse is exxdiv-corrected. The sealed
    cross-family CRYSTAL gate is now BIPOLE-SHRINK-2-2 (exxdiv) ↔
    CRYSTAL-**SHRINK-8-8** (k-converged): +2.9 mHa at c12, pinned `@slow` in
    `tests/test_pbc_bipole_multik_ewald_split.py::test_mgo_shrink22_converged_matches_crystal_kconverged`
    (+ example `parity_mgo_shrink22_crystal_convention.py`). The −271.85640
    matched-k value is CRYSTAL's *under-converged* coarse number, **retired** as
    a BIPOLE target. To reproduce CRYSTAL's coarse-mesh exchange one would need
    a CRYSTAL-style *uncorrected* finite-k exchange mode (neither
    `exxdiv='ewald'` nor `'none'` matches it), out of scope; the dense-k
    comparison is the physically meaningful one.
  * *Converged-cutoff confirmation (c16, S(k)-fold 2.3e-6, SYM3b-accelerated).*
    At the fully-converged cutoff, BIPOLE SHRINK 2 2 = **−271.21343 Ha/FU**,
    matching **PySCF [2,2,2] `exxdiv='ewald'` (−271.213356) to 0.075 mHa**, the
    same-convention reference. So at convergence BIPOLE reproduces the PySCF
    [2,2,2] value, and the c12 *+2.9 mHa vs CRYSTAL SHRINK-8-8* was partly a
    lattice-truncation coincidence: the converged value is **+4.7 mHa from
    CRYSTAL SHRINK-8-8**, and that residual is the [2,2,2] mesh's own finite-k
    error (PySCF [2,2,2] is +4.8 mHa from SHRINK-8-8 too, exxdiv accelerates
    but does not fully k-converge the coarse mesh). Net: the *tight*
    cross-validation is BIPOLE = PySCF [2,2,2] (same convention, 0.075 mHa); a
    CRYSTAL comparison is convention-confounded at SHRINK-2-2 (+643 mHa) and
    k-mesh-confounded at SHRINK-8-8 (+4.7 mHa). The compute-small vq ladder confirmed the
    c12 rung bit-for-bit; its c16/c18 rungs hit a payload warm-start bug
    (`initial_density` wants per-cell, not per-k, blocks), c16 was re-run
    from SAD locally.
  * *Efficiency (PBC-audit E2/E3, BIPOLE multi-k lane).* The c12 SHRINK-2-2
    SCF is ~10 iters × ~75 s ≈ 12.5 min, dominated by the C++
    `build_jk_2e_real_space` direct-ERI rebuild (audit **E1**). The BIPOLE
    multi-k **E2** caches (J^LR FT, K^LR q-channel B-tensors, V_ne) are already
    built once before the SCF loop and reused; **E3** (the `…FT_BACKEND=python`
    slow pin) is *not* triggered, STO-3G drives the C++ streaming Bloch-FT
    kernel (cache builds in seconds). The residual per-iter cost is the
    irreducible direct ERI build; cutting it further needs integral storage
    (conventional-SCF; memory-heavy) or real-space point-group reduction,
    the latter **shipped as SYM3b** (`use_fock_symmetry_reduce=True`): build
    J(g)/K(g) only at the atom-pair-orbit representative cells (14/55 at c12,
    full internal sum) and reconstruct by rotation, bit-identical to
    `symmetrize_fock_blocks(full build)`. **~3× faster** (~25 s/iter vs ~75 s);
    the `@slow` CRYSTAL gate now uses it (~4½ min, was ~13½). The later
    atom-pair shell-pair mask also shipped, increasing the measured c12
    speedup to about 4.4× while preserving bit-identical reconstruction.
* **GPW/GAPW ↔ CP2K/GPAW**, runners exist; gate pending.

See also: [Archived `handovers/HANDOVER_CROSS_CODE_PARITY.md`](https://vibe-qc.com/docs/)
(the shipping `python/vibeqc/parity.py` energy-decomposition certification
matrix), [`docs/license.md`](license.md) (the external-program provenance
recorded in the `.system` manifest), and CLAUDE.md §10 / §15.
