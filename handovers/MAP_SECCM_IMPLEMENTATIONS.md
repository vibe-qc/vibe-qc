> Public workstream note. Line numbers cite `main` at `9be848b` (2026-09-16); the
> SCC-DFTB, GFN2 and PM6 adapter files moved by a few lines with #293/#294.

# MAP: the SECCM implementations, cross-referenced

## 1. Scope and method

Consolidated on 2026-09-16 from eight per-implementation maps of
a per-session clone (`claude-seccm-unify`) at HEAD `e873908` (verified: `git log -1` =
`e873908 2026-09-16 Preserve FRAGMO request diagnostics ...`; the pm6 and msindo maps cite `9670ee6`,
which is the HEAD of the sibling checkout `claude-semi-empiric`, not of the mapped tree -- every line
number in those two maps was spot-checked against `e873908` and held). The repository was read-only
throughout. Columns and the files they cover: **DFTB0-SECCM** = `cpp/src/semiempirical/seccm/dftb0.cpp`
(D0), `cpp/include/vibeqc/semiempirical/seccm/dftb0.hpp` (D0H), `python/vibeqc/semiempirical/seccm/dftb0.py`
(D0P); **SCC-DFTB-SECCM** = `seccm/scc_dftb.{cpp,hpp,py}` (S, SH, SP); **GFN2-SECCM** =
`seccm/gfn2.{cpp,hpp,py}` (G, GH, GP); **PM6-SECCM** = `seccm/pm6.{cpp,hpp,py}` (P6, P6H, P6P);
**OMx-SECCM** = `seccm/omx.{cpp,hpp,py}` (OX, OXH, OXP); **MSINDO CCM (legacy)** =
`python/vibeqc/semiempirical/methods/msindo_ccm.py` (MC), `msindo_ccm_gradient_analytic.py` (MG),
`msindo_ccm_stability.py` (STB), `msindo.py` (MS), `cpp/include/vibeqc/semiempirical/methods/indo/ccm_engine.hpp`
(CE), `indo_engine.hpp` (IE), `python/vibeqc/semiempirical/runner.py` (RN); **periodic Gamma GFN2** =
`cpp/src/semiempirical/methods/xtb/periodic_gfn2.cpp` (PG2), its header (PG2H), `gfn2_driver.{cpp,hpp}`
(MD, MDH) as the molecular reference, `python/vibeqc/semiempirical/periodic.py` (PY),
`cpp/src/semiempirical/gradient.cpp` (GR). Shared layer: `seccm/seccm_common.h` (SC), `seccm/ewald_1d.h`
(E1), `core/periodic_gamma.{hpp,cpp}` (PGH/PGC), `core/charge_mixer.hpp` (CM), `core/basin_gates.hpp` (BG),
`kpoints_occupations.cpp` (KO), `seccm/topology.{py,hpp}` (TP/TH), `seccm/_adapter_common.py` (AC),
`cpp/src/bindings.cpp` (B), `python/vibeqc/semiempirical/routes.py` (RY),
`python/vibeqc/output/citations/registry.py` (RG). Tests: `tests/test_ccm_semiempirical.py` (T),
`tests/test_dftb0_seccm.py` (TD), `tests/test_seccm_scc_dftb_basin.py` (TB),
`tests/test_semiempirical_route_plan.py` (TR), `tests/test_msindo_ccm.py` (TM), `tests/test_msindo_cpp.py`
(TMC). Where two maps disagreed or a claim looked surprising, the cited lines were opened with `sed -n`;
each such check is marked "(verified)" below. Every factual row keeps a `file:line` citation.

## 2. The matrix

Column key in every table: **D0** DFTB0-SECCM | **S** SCC-DFTB-SECCM | **G** GFN2-SECCM | **P6** PM6-SECCM |
**OX** OMx-SECCM | **MS** MSINDO CCM (legacy) | **PG** periodic Gamma GFN2 (reference driver).

### 2a. Topology: source, fields, normalisation, gates, molecular limit

| concern | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| record source | Python `SECCMTopology` flattened by a local copy D0P:135-165; rebuilt B:718-757; validated B:779-880, SC:83-246 | `flatten_topology_records` AC:16-124 via SP:112-116; same rebuild/validation | AC:16-124 via GP:820-830; same | AC via P6P:197-207; same | AC via OXP:146; same | bare `build_wigner_seitz(C,T)` MC:670, 810 (alias of `build_seccm_topology` TP:1084); C++ rebuilds its own with TH:222-269 at CE:712, 859, 1084 | pair-cutoff enumeration `atom_pair_interaction_cells` PG2:215; records closure PG2:158-183, unit weight; spans-lattice guard PG2:37-94 |
| fields consumed | `weight, origin, disp` (repulsion D0:181-186, 350-358); `image_shell_label` (S/H0 SC:452-457; gradient D0:293-301); `ownership_multiplicity` validator only SC:177-185 | `origin, weight, disp` (S:704-709; SC:614-654; E1:171-196; S:958-1046); label SC:452-457, S:903-939; multiplicity validator only | `origin, weight, disp, label` G:239-251, 283-291, 316-346, 601-609; labels via `collect_shell_keys` G:199-202; multiplicity validation only | `origin, weight, disp` only P6:354-357, 694-702; label/multiplicity in validator and trivial test | `origin, weight, disp, label` OX:413-415, 521, 534-537; multiplicity NOT read, recounted OX:522-526 | `origin, weight, disp` MC:207-227, 629-639; no labels | (a, b, cell) with `r_cart`, weight 1.0 PG2:158-183 |
| local re-derivation | none (D0:37 comment; no `build_wigner_seitz` call) | `_ewald_ws_cells` adds self record CE:181-189 (S:409); `seccm_image_records` shift SC:478-497 (S:997) | `seccm_pair_weight` G:110-129 (faithful AES); `seccm_image_records` G:157, 1055, 1107; `_ewald_ws_cells` G:1686 | `_ewald_ws_cells` P6:234 only | union `WSSC(M) u WSSC(N-image)` OX:243-265 (unique) | two builders per C++-routed run MC:670 + CE:712 | `gamma_cells` re-enumerated when remainder range exceeds cutoff PG2:282-295 |
| per-cell normalisation | `/ group_order` D0:190-194; gradient D0:361 | `/ group_order` S:721-728; gradient S:1072 | `/ group_order` G:1130-1153; delegation G:1452-1471 | `/ group_order` P6:746-751 | `/ group_order` OX:899-904 | per supercell, no division MC:776-778 (tests divide by n TM:85-87) | one cell by construction |
| dimension gate | 1-3 (D0:37-52; SC:111-114) | 1-3; 3-D madelung needs Elstner S:255-261 (verified) | 1-3; 3-D `ewald_gamma` refuses KO G:1567-1574 | 1-3 unembedded; 3-D madelung refused P6:213-218 (verified) | 1-3, no OMx-specific gate (OXH; SC:111-114) | 1-3 MC:97-135; CE:49-53 | 3-D lattice only |
| replica gate | none, D2 (D0:54-58, verified) | re-checked S:262-282; 3-D all-odd S:283-291, SP:247-255 (verified) | none | none | none | n/a (no finite group) | n/a |
| molecular-limit detection | none; arithmetic identity (TD:83-104, T:146-160) | none (T:193-223, 266-283) | `is_trivial_molecular_records && !madelung && include_aes` G:1305-1307 (verified) | `is_trivial_molecular_records` gates the ack only P6:212, 219-220 | `group_order > 1 \|\| !trivial` gates the ack OX:333-334 (verified); OXP:150-156 | none; dilute-chain test TM:840-849 | none; isolated molecule refused PG2:217-219 |
| delegation | none | none | to `run_gfn2_xtb` G:1397-1555; forwards 7 options G:1401-1412 (verified) | none | none | none | n/a |

### 2b. Per-term record sums and raw home-cell coordinate use

| term | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| overlap S | `w * S_block[label]` per record, home block unit weight SC:405-466 (D0:123-124) | same SC:405-467 (S:343-345) | `w * S_block[label]` G:270-291 | no S matrix (P6:661); STO overlap only inside resonance per record P6:562-570 | no S; per-record `nddo_sto_overlap` OX:473-476, 595-629, 684-687 | Löwdin-orthonormal molecular S (MS:1996) | cell sum masked by cutoff PG2:266-268 |
| H0 / core | `w * 0.5 kappa S hbar_avg`, diag on-site with -0.5 fallback SC:442-462 | same | per-label `build_gfn2_hamiltonian_zero_image_with_cn` with cyclic CN `sum w f_CN(|disp|)` G:239-263 | one-centre home P6:287-288; two-centre `w*` electron-core/exact tensors P6:447-508 | one-centre OX:401; core attraction, resonance, VORT eq-8, ECP over records; eq-9 three-centre over union with `w*x_C` OX:443-717 | `HK1` all records MC:207-217; `HKL2` only `k<j` records, transpose MC:218-221 | cell sum PG2:270-277; periodic CN PG2:269 |
| repulsion / core-core | `0.5 w V_rep(|disp|)` D0:180-189 | `0.5 w` S:703-712 | `0.5 w` G:337-346 | `0.5 w pm6_core_core_repulsion` P6:689-712 | `0.5 w f_ko Z Z / R` OX:738-741 | `0.5 cz cz w / R` MC:629-639, CE:644-654 | home `b>a` w=1, images all `b` w=0.5 PG2:305-321 |
| second-order / two-electron | none | `w * g(|disp|)` SC:595-657 or Ewald-split SC:568-584 | `w * gamma(si,sj,|disp|)` G:307-332 or Ewald-split G:152-157 | exact H-X / sp-sp tensors + monopole gammas per record P6:390-543; exchange P6:511-588 | `w * f_ko * c2int` channel blocks OX:719-736 | `_gamma_shell` on `k<j` records MC:222-227 | Ewald-split PG2:296-297 over unit-weight `gamma_records` |
| third-order | none | none | on-site `-G dq^2`, `G q^3/3` G:583-588, 1110-1116 | n/a | n/a | n/a | PG2:446-453, 538-544 |
| AES / multipoles | none | none | ad-hoc: potential WS-weighted G:601-648, moments from raw molecule G:1705, 1244-1246; faithful: lattice sums G:356-367, 555-570 | inside NDDO pair tensors | none | none | atom-resolved Bannwarth, joint state PG2:299-303, 462-473 |
| dispersion | none (D0H:23-25) | none | none (GH:33-34) | none | none | none | none in C++ (Python post-SCF only, `methods/gfn2.py:769-779`) |
| raw home-cell coordinates | S/H0 blocks from home atoms + label shift SC:412-419 (`disp` unused there) | same; Ewald half on `R_b - R_a` PGC:718-727; gradient `C0` CE:985 | same; AES moments from raw molecule GH:248-255; Ewald half | never (`disp` only) | never (only `Z`) | image at `coords[k] + disp` MC:207-217 | positions + cell shift; no wrapping PG2H:14-16 |

### 2c. Second-order kernel

| concern | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| form, default | none | KO default, Elstner opt-in S:371-375; SP:214-221 | KO default GH:245; Elstner only with `ewald_gamma` GP:848-853 | NDDO: exact multipoles or monopole gamma with KO fallback `1/sqrt(R^2+eta^2)`, `eta=0.5(rho_a+rho_b)` P6:181, 412-437, 583-586 | KO-scaled analytic STO monopole `omx_gamma_k` OX:82-89, `f_ko` OX:129-133 | MSINDO `_gamma_shell` MC:222-227 | Elstner default PG2H:91-97 (D1) |
| KO average | n/a | `InverseHardnessMean` SC:601-625, 650-653 | `HardnessMean` G:148-150 | n/a | n/a | n/a | `HardnessMean` PG2:208-209 |
| WS-truncated vs Ewald-split | n/a | WS-truncated `weighted_dftb_gamma` SC:595-657; `ewald_gamma` -> `ewald_dftb_gamma` SC:568-584 | WS-truncated loop G:307-332; `ewald_gamma` -> `seccm_shell_gamma` G:133-172 | WS-truncated only | WS-truncated only | WS-truncated only | Ewald-split only |
| builder | n/a | `build_periodic_shell_gamma` PGC:706-767 (Ewald path); hand-rolled KO SC:631-657 (default) | `build_periodic_shell_gamma` (Ewald path); `gfn2_shell_gamma_at_distance` loop (default) | local | local | local | `build_periodic_shell_gamma` PG2:296-297 |
| alpha | n/a | `seccm_ewald_alpha` SC:510-532, never overridden S:373, 853 | SC:510-532 or test override G:152-154 (T:3758 pins alpha-independence) | n/a | n/a | n/a | kernel default PGC:414-427 |
| on-site block | n/a | `U_a`, 0.4 fallback SC:542-543, 639-640 | Ewald `potential(0, exclude_self)` + `shell_gamma_onsite` PGC:723, 756-765; `molecular_onsite` subtraction G:159-170 | one-centre NDDO P6:277-343 | one-centre Pople OX:780-847 | `one_center_gmunu` MC:195-205 | as G Ewald path |
| 3-D policy | n/a | `ewald_gamma` allowed with either form (conditional KO convergence documented PGH:17-22, not refused) | `ewald_gamma` + KO refused G:1567-1574 | n/a (no lattice sum) | n/a | n/a | native |

### 2d. Long-range embedding (Madelung) and the `ewald_gamma` alternative

| concern | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| 1-D kernel | none | `wire_madkonst_1d` E1:154-200 (alpha 4/L, `n_real` 8, `m_max` 24 E1:164-167, verified) | same E1 (G:1688-1689) | same E1 (P6:236-238) | none | truncated +-2-shell bare sum MC:251-271, CE:205-222 | Ewald inside gamma (PGC:601-635) |
| 2-D kernel | none | `_madkonst_2d` CE:573-624 (S:412-413) | same (G:1691-1692) | same (P6:239) | none | same CE / Python twin MC:548-604 | PGC:566-599 |
| 3-D kernel | none | `_madkonst_3d` CE:363-405 (S:414-416, verified) | same (G:1694-1695, verified) | refused P6:213-218 (verified) | none | `_madkonst_3d` MC:370-399 | PGC:442-475 with background |
| deposit convention | none | S-weighted `H -= 0.5 S (V_mu+V_nu)` with `V = gamma dq + V_mad` S:127-144 | MSINDO diagonal `H(mu,mu) -= V_mad` G:715-717; S-weighted opt-in G:707-713; `madelung_no_self` G:454-459 | F-only diagonal `F(mu,mu) -= V_mad` P6:613-623 | n/a | diagonal `-mad[i]` MC:764-766 | n/a |
| energy half | none | `0.5 dq.V_mad` S:629-631 | `0.5 dq.V_eff` G:1120-1123 (verified) | core half `0.5 Z V` P6:734-744; electronic half inside `Tr[P(H+F)]`; `e_madelung` is inside `e_electronic` P6:748-751 | n/a | `0.5 cz.V` MC:767 | n/a |
| SMADEL subtraction | n/a | `_madelung_potential_ewald` CE:627-641 (verified) | same | same | n/a | same MC:606-618 | n/a (remainder split instead) |
| gates | n/a | `madelung` xor `ewald_gamma` S:245-249; charged only with `madelung` S:244; 3-D needs Elstner S:255-261; 3-D odd replicas S:283-291 | `madelung` xor `ewald_gamma` G:1352-1356; charged refused SC:103-106; 3-D madelung runs (T:4804) | 3-D refused; unembedded nontrivial needs `allow_truncated_electrostatics` P6:219-230 | `allow_truncated_electrostatics` OX:333-345; no embedding at all | electroneutrality `\|sum q\| > 1e-3` MC:692-700 | n/a |
| `ewald_gamma` alternative | none | SC:568-584 (`Phi_Ewald + sum w [gamma - 1/r] + delta U`) | G:133-172 | none (`periodic_gamma.hpp` unused, P6:1-16) | none (OX:39-42) | none | is the kernel |

### 2e. SCC / SCF loop

| concern | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| guess | n/a (no SCF) | `dq = 0` S:425; no `initial_charges` | `dq = 0` or validated `initial_shell_charges` G:976-979, 1304-1330 | `D = 0` P6:258 | `D = 0` OX:761 | Hcore `eigh` Aufbau MS:1985-1993 | `dq = 0` (PG2:444 loop) |
| eigensolve / screening | `SelfAdjoint(S)`; `min eig <= 1e-10` -> `canonical_orthogonalizer` D0:138-158, SC:336-400; else generalized D0:158-166 | same S:356-367, 155-171 | same with hard-coded 1e-10 G:1625-1635, 725-741 | `SelfAdjoint(F_eff)` P6:661-666, no screening | `SelfAdjoint(F_eff)` OX:925-927 | `eigh` in Löwdin basis MS:1996-2030 | generalized on `(H_scc, S_gamma)` PG2:475-477 |
| occupations, T | hard Aufbau D0:173-176; no T in signature D0H:43-59 | FD via `compute_closed_shell_kpoint_occupations` S:172-184 or Aufbau S:185-188; T default 0 SH:111-152 | FD G:745-752 / Aufbau G:753-757; T default 0 GH:165 | Aufbau P6:667-668; no T | Aufbau OX:930-931; no T | integer Aufbau MS:2030; no T | FD at 0.001 unless explicit PG2:345-352; shared KO:249 |
| gap guard and waiver | `finite_torus_homo_lumo_gap` D0:167-171, no waiver (verified) | S:599-621: throw if non-finite or (`!smeared && gap<=tol`); `gap_guard_waived = smeared && gap_below` (verified) | G:1165-1176: gap recorded, rejected only when `T<=0`; waiver otherwise (verified); delegated copy G:1522-1540 | local `eps(n_occ)-eps(n_occ-1)` P6:681-686, no waiver (verified) | local OX:889-897, no waiver (verified) | none; stability audit when Hcore gap `<= 1e-6` STB:72-76, `ccm_stability.hpp:101-108` | none mapped |
| basin gates | none | none (only `\|dq\| <= n_val+2` extrapolation fallback S:502-513, 531-540) | BG:42-60, 103-120 at G:1183-1185, 1266-1268 | none | none | none | BG via PG2:96-111, 522-529 |
| residual, tolerance | n/a | `max\|dq_new-dq\| < 1e-8` S:481-483 | inf-norm of reduced state `< 1e-6` G:1079-1083 | `max\|dD\|` and `max\|[F,D]\| < 1e-7` P6:670-680 | `max\|[F,D]\| < tol && \|dE\| < tol`, 1e-7 OX:883-885 | `max\|[F,P]\| < tol && \|dE\| < tol`, 1e-9 MS:2003-2006 | full reduced state `< 1e-6` PG2:507-514 |
| retry ladder / checkpoint | n/a | none; executed T always the request S:646, 685 (verified) | primary `min(500,total)`; rungs `{0.01,2000,0}`, `{0.05,500,0.005}`, `{0.05,600,0.05}` if Z>18; T override unconditional G:1712-1766 (verified) | none P6:769-772; Python `RuntimeError` P6P:232-235 | none OX:935-938 | up to 6 stability restarts STB:91-103 | 500/25 contraction checkpoint PG2:431-442, 610-613; one restart `{0.01, min(2000,rem)}` PG2:615-641 (verified) |
| attempt ledger | none | none (`record_applied_occupation` S:69-93 only) | `stamp_supercell_attempt` G:904-937, `prepend_attempt_history` G:939-963; Python `_project_attempts` GP:453-569 | none | none | stability quartet MC:165-180 | none (PG2H:43-79); traces concatenated PG2:635-640 |

### 2f. Mixer

| concern | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| classes | none | local vector Aitken S:576-595 (default); `vibeqc::DIIS` on `(dq, resid)` S:452-455, 518-575; `BroydenMixer` S:460-464, 485-517 | `SimpleMixer`/`BroydenMixer`/`EyertBroydenMixer`/`ChargeDIISMixer` G:1016-1033 (CM:88-221); inline Newton G:804-887 | Fock-commutator Pulay DIIS, `DIIS_MAX=8` P6:264, 625-659 | `DiisState` MAX 8 OX:194-228 (verbatim `omx_fock.cpp:863-898`) | Pulay DIIS on `[F,P]`, max 8 MS:1996-2030 | Eyert Broyden(`min(max_iter,64)`, 0.4) default PG2:382-389; DIIS/Broyden PG2:390-403; Aitken simple PG2:582-605 |
| state vector | n/a | atomic `dq` (n_atoms) S:481 | `[dq_shell]` or `[dq; mu(3n); theta(6n)]` G:500-549 | stored `F_i`, errors `[F,D]` | up to 8 `(F_i, e_i)` | `f_hist, e_hist` (8) | `[dq; mu; theta]` PG2:113-152 |
| damping | n/a | `charge_mixing` 0.2 ceiling, floor `min(0.01,mix)` S:426-427, 576-595; no finite-T one-way branch (molecular has it, `cpp/src/semiempirical/scc_dftb.cpp:291-294`, verified) | simple step `min(cm, 0.1)` when GAM3 G:1013-1015; Newton cap 0.5 G:802 | window `damping_start=min(64,max(2,max_iter/3))`, `+32`, `D=0.5D+0.5D_new` P6:261-268, 676 | window `max_iter/2`, `+64` OX:773-775, 932 | none; Madelung rebuilt undamped MC:762-767 | `simple_step=min(cm,0.1)` PG2:370-372 |
| refusals | n/a | DIIS xor Broyden S:312-316; ranges S:292-311 | Newton+faithful G:988-994; Newton+delegation G:1391-1396 | none | none | none | Newton "SECCM-only" PG2:404-406 |

### 2g. Energy assembly

| concern | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| terms | `sum 2 eps + E_rep` D0:173-198 | `tr(D H0) - TS + 0.5 dq.g.dq + E_mad + E_rep` S:622-641, 713-716 | `band0 + es + aes + 3rd + mad + rep - TS` G:1128-1129 (verified) | `0.5 Tr[P(H+F)] + core + 0.5 Z V_mad` P6:714-744 | `0.5 Tr[D(H+F)] + E_core` OX:874-881 | `0.5 Tr[P(H+F)] + 0.5 cz.V + core` MC:776-778 | `band0 + es + aes + 3rd + rep` PG2:546 |
| `energy` = E or A | E (no T) | A per cell (`-TS` inside `energy`) S:635-636 | A per cell G:1132; GH:59 "Mermin A at T > 0" (verified) | E (no T) | E (no T) | E (no T) | E; `free_energy = E - TS` PG2:551-554 (verified); molecular same MD:1051-1053 (verified) |
| `free_energy` field | absent D0H:19-41 | `= energy` S:727 | `= energy` G:1133; delegation `energy = molecular.free_energy/N` G:1449-1456 (verified) | absent | absent | absent | separate PG2H:44-54 |
| per-cell | `/group_order` | `/group_order` | `/group_order` | `/group_order` | `/group_order` | per supercell | per cell |
| closure pinned | TD:176-192 | T:193-223 (1e-12) | GH:63-66; T:2173-2222 | T:1196-1222 (1e-14) | T:1651-1661 | TM:672-693 (5e-9 vs oracle) | n/a |

### 2h. Gradient and stress

| concern | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| method | analytic, fixed topology D0:223-367 | analytic S:743-1078 | none (GP:816; route `{energy}` RY:527-533) | Python central FD, 6 SCFs/atom, step 1e-4 P6P:63-129 | none (RY:609-610) | FD MC:780-877 / CE:818-897; analytic MG:59-363 / CE:1056-1211 | analytic GR:2243-2509; molecular GR:1992-2222 |
| what is differentiated | `sum w [D dH0 - W dS]` via `overlap_lattice_gradient_contribution` D0:345-347; repulsive D0:349-360 | overlap term S:898-955; repulsive S:957-969; gamma untailed S:999-1023 / Ewald S:988-997; Madelung 1-D S:1036-1050, else `_add_madelung_gradient` S:1051-1069 (verified) | n/a | full energy at displaced geometry, `rebuild_displacements` P6P:82-97 | n/a | per-record HK1/HL1/HKL2/gamma/nuclear MG:340-355; Madelung direct assembly MG:383-470 | overlap, AES, CN chain, lattice gamma gradient/strain, repulsion GR:2311-2426 |
| occupation handling | W at T=0 D0:269-275 | W with Fermi weights recomputed at `opts.electronic_temperature` S:823-846 | n/a | Aufbau | n/a | Aufbau | `result.occupations` GR:2272-2281; molecular W at T=0 GR:2025-2029 |
| refusals | screened result only via shape check D0:238-247 (implicit) | not converged S:731-734; same madelung/ewald/3-D refusals S:763-779; screened NOT refused | n/a | parameter snapshot must not change P6P:109-116 | n/a | not converged MG:192-193; closed shell MG:96 | no cutoff provenance GR:2438-2444 |
| stress | none | none (`periodic_shell_gamma_strain_derivative` PGH:241-247 unused) | none | none | none | none | `periodic_shell_gamma_strain_derivative` GR:2391-2401 |

### 2i. Provenance and result fields

| field | D0 | S | G | P6 | OX | MS | PG |
|---|---|---|---|---|---|---|---|
| `gamma_form` | n/a | NOT on result (SH:48-109; SP:333-378, verified) | `requested_gamma_form` GH:126-142; `hamiltonian_identity` GP:975-991, RY:739-759 | n/a | n/a | n/a | PG2H:74-78 |
| executed temperature | n/a | `smearing_temperature = opts.electronic_temperature` S:646, 685 (requested == executed; no ladder) | `smearing_temperature` = executed G:1135, 1458; `resolved_electronic_temperature` GP:975-980 | n/a | n/a | `electronic_temperature=0.0` stamped RY:1318, 1340-1345 | `smearing_temperature` PG2:417-422; Python Gamma path returns bare tuple PY:1020 |
| mixer | n/a | not recorded | attempt `solver` name G:904-937; `run_controls.scc_mixer` GP:871-879 | n/a | n/a | n/a | not recorded |
| cutoffs | none (pair_cutoff 0 implicit, SC:424-425) | none | none | none | none | none | `cutoff_bohr` PG2H:74 |
| parameter identity | `parameter_identity/sha256` B:614-629, D0P:209-210 | B:11844-11846, SP:373-375 | B:12586-12597, GP:1038-1040 | B:625-626, P6P:280-282 | B:12463-12474, OXP:241 | dropped by `run_seccm` RN:1149-1157 (constructor accepts RN:252-271) | k-route only PY:658-682 |
| `translation_symmetry_charge_spread` | no | no | yes, ungated GP:1032-1036, AC:191-223 | no | no | no | no |
| `attempts` | no | no | yes GH:120-121, GP:465-522 | no | no | stability quartet | no |
| `route_plan.electrostatics_family` | default `"none"` (D0P:81-87 passes `periodic_dimension` only; RY:1318, verified) | `"madelung" \| "none"` SP:266 (verified); `"ewald_gamma"` accepted by RY:68 but never emitted | requested/resolved family in `with_gfn2_seccm_runtime` GP:975-982 (verified) | default `"none"` even with `madelung=True` (P6P:188-196 passes `periodic_dimension` only, verified) | default `"none"` + `three_center_weighting`, `truncated_electrostatics_acknowledged` OXP:175-181, OX:767-771 | `"madelung" \| "none"` with `method_key="msindo"` -> `truncated_1d` RN:1153-1161, RY:131-135 | none (bare tuple PY:1020) |

## 3. Same physics implemented more than once

Ordered by refactor impact. Each entry: the physics, every location, and the verdict.

1. **Madelung dimension dispatch (4 copies).**
   - S:409-416 energy: `dim == 1 / 2 / 3` -> `wire_madkonst_1d / _madkonst_2d / _madkonst_3d` (verified).
   - S:860-868 gradient: `dim == 1` -> wire, else `_madkonst_2d` -- no 3-D branch (verified).
   - G:1684-1696: `1 / 2 / 3` (verified).
   - P6:233-240: `1` -> wire, else `_madkonst_2d`; dim 3 refused upstream at P6:213-218 (verified).
   - Deposit side: `indo::_add_madelung_gradient` dispatches `1 / 2 / 3` itself (CE:983-1030, verified).
   - **Differ:** the S gradient hands a 3-D cell to `_madkonst_2d`, which reads `translations[0..1]` only
     (CE:471-473); the gradient's potential and its deposit then use different lattices.

2. **GFN2 stabilisation ladder (3 budgets).**
   - G:1712-1766: `use_stab = has_gam3 && total >= 2500`; primary `min(500, total)`; rungs `{0.01, 2000, 0}`,
     `{0.05, 500, 0.005}`, `{0.05, 600, 0.05}` for Z > 18; T override unconditional at G:1757; no
     `auto_stabilize` flag (verified).
   - MD:846-861, 1163-1221: `&& opts.auto_stabilize`; primary `min(700, total)`, advisory, extended while
     contracting (MD:865-875, 1149-1155); rungs `{0.01, 1800, 0}`, `{0.05, 500, 0.005}`, `{0.05, 600, 0.05}`;
     T applied only when `!electronic_temperature_explicit` MD:1195-1211 (verified).
   - PG2:431-442, 615-641: 500-step contraction checkpoint, one restart `{0.01, min(2000, rem)}`, no ladder,
     no ledger (verified).
   - `has_extended_period_element` G:371-375 = MD:506-510.
   - **Differ** (budget shape, first-rung length, explicit-T handling, flag surface).

3. **1-D wire quadrature (2 copies) plus a physically different third 1-D kernel.**
   - E1:68-146: `gauss_legendre`, `wire_F`, `wire_m0`; fixed `n_real = 8`, `m_max = 24` (E1:166-167).
   - PGC:289-380: same three functions with rho-derivatives; `r_cut = sqrt(30)/alpha` (PGC:428-440),
     `m_max` from `k_cut` (PGC:496); `reciprocal_1d` PGC:601-606 cites E1 as its source (verified).
   - MC:251-271 / CE:205-222: MSINDO's truncated +-2-shell bare sum, "kept frozen for MSINDO" (E1:3-4).
   - **Differ:** two truncation rules for one formula (bit-equality unverified); MSINDO is a different kernel.

4. **Madelung/Ewald in MSINDO vs seccm.**
   - 2-D/3-D matrices: S, G, P6 call CE:573 / CE:363 directly; MSINDO Python twins MC:274-618 vs CE:181-655,
     pinned 1e-6..1e-10 TMC:766-815; argument order differs (`_ewald_lattice_2d(ews, T)` vs `(T, ews)` CE:459-461).
   - Potential wrappers around CE:627-641: G:450-461 `effective_madelung_potential`, S:405-407 lambda,
     P6:614-615, MC:606-618.
   - Madelung gradient: MG:383-614 vs CE:909-1054 (the C++ copy is reused by S:1062).
   - **Agree** on 2-D/3-D values; **differ** on 1-D (item 3).

5. **Image-record conversion (3 shapes plus an unpacker).**
   - SC:478-497 `seccm_image_records` -> `ImageRecord{a, b, shift = disp - (R_b - R_a), w}` (verified).
   - CE:181-189 `_ewald_ws_cells` -> `WSNeighbor` lists plus a self record `{i, 1.0, 0}` (verified).
   - `cpp/src/semiempirical/periodic_scc_dftb.cpp:184-204` and PG2:158-183: pair-cutoff records,
     `shift = r_cart`, weight 1.0 (verified; the shared map cited this file without its directory).
   - B:15850: array unpacker for the Python seams.
   - Adapters running both `ewald_gamma` and `madelung` build two (G:1055, 1686; S:409, 997).
   - **Differ by design** (weights, self record); one adapter interface would remove two of them.

6. **Ewald alpha convention (4 sites).**
   - SC:510-532: 3-D `sqrt(pi)/cbrt(V)`, 2-D `0.85 sqrt(pi)/sqrt(A)`, 1-D `4/L` (verified).
   - PGC:414-427: identical, selected when `alpha <= 0` (verified).
   - CE:294 (3-D) and CE:473 (2-D) (verified); E1:164 and E1:221 (`4/L`) (verified).
   - MSINDO Python restates them at MC:319, 433.
   - **Agree**; `seccm_ewald_alpha` only passes the kernel its own default.

7. **Gap guard (6 sites).**
   - Shared `finite_torus_homo_lumo_gap` SC:380-386 (NaN when no LUMO, #151) used at D0:167-171 (no waiver),
     S:599-621 (waiver iff smeared), G:1165-1176 (waiver iff T > 0) (all verified).
   - Local `eps(n_occ) - eps(n_occ-1)` at G:1522-1540 (delegation), P6:681-686, OX:889-897 (verified).
   - MSINDO: none; the #249 stability audit runs instead when the Hcore gap is `<= 1e-6` (STB:72-76).
   - **Agree** on the predicate; **differ** on waiver availability and on when `homo_lumo_gap` is recorded
     (G before rejection G:1168; S only on the converged path S:642).

8. **SCF drivers (7 loops).**
   - S:466-597 with `scc_seccm_charge_map` S:105-211; G:965-1285 `run_supercell_scc`; P6:258-687;
     OX:761-938 (verbatim `omx_fock.cpp:943-999`); MS:1945-2043 / IE:2270-2331 (shared with molecular
     MSINDO); PG2:444-614; MD:631-1223 (hybrid simple/DIIS polyalgorithm MD:738-808).
   - **Differ** in residual (charge inf-norm / density + commutator / commutator + energy / full reduced
     state / dq-only), guess, and tolerance.

9. **Aitken / DIIS / Broyden wrappers.**
   - Aitken: S:576-595 vs molecular `cpp/src/semiempirical/scc_dftb.cpp:286-310` (adds a finite-T one-way
     branch at 291-294, verified) vs PG2:582-605 (floor `min(0.005, step)`).
   - Charge DIIS: S:518-575 (`vibeqc::DIIS` on `n_atoms x 1`) vs `periodic_scc_dftb.cpp:364-400` vs
     `ChargeDIISMixer` CM:115-145 (used by G:1030).
   - Guards: Broyden S:485-517 duplicates the DIIS guards; `|dq| <= n_val + 2` at S:502-513, 531-540 and
     `periodic_scc_dftb.cpp:377-388`; drift projection S:498-501, 551-554.
   - Fock DIIS: P6:625-659, OX:194-228, MS:1996-2030 with three damping windows (P6 `max_iter/3, +32`;
     OX `max_iter/2, +64`; MS none).
   - **Differ.**

10. **WS topology builders (3).**
    - TP:999-1084 (Minkowski-reduced, candidate provenance, `rebuild_displacements` TP:601).
    - TH:222-269 (coefficient box, no reduction, no provenance); `WignerSeitzCells` is an alias of
      `SECCMTopology` (TP:983), not a class in MC.
    - Tie tolerance TP:70 == TH:89 ("must stay bit-identical" TH:87-88); validity TP:398-403 == TH:47-55;
      label translation SC:63-71 == TH:60-68 == TP:1635.
    - MSINDO builds both per C++-routed run (MC:670 + CE:712).
    - **Agree** on weights by construction; **differ** in contract (AC:43-60 demands a bound finite group
      and length unit; MC:670 passes neither).

11. **Gradient bookkeeping helpers.**
    - `collect_shell_keys` SC:248-262 vs D0:292-303 vs S:900-911.
    - `shell_lattice_cells` SC:280-292 vs D0:310-321 vs S:919-929.
    - `ao_info_for_basis` SC:299-313 vs D0:259-267 vs S:803-811.
    - `PeriodicSystem` build SC:414-419 vs D0:336-344 vs S:944-952; block scatter `-w M` D0:322-334 vs S:930-942.
    - Repulsive derivative D0:349-360 vs S:958-969 (S adds `d < 1e-12` skip).
    - W: D0:269-275 vs S:823-846 vs GR:103-106; M: D0:278-290 vs S:884-896 vs GR:124-135.
    - **Agree** (byte-identical loops) apart from the S guard.

12. **`flatten_topology_records` and the trivial-record test.**
    - AC:16-124 vs D0P:135-165 (omits AC:53-56 dimensionality and AC:71-78 orbit validation, verified).
    - `is_trivial_molecular_records` SC:268-278 vs B:685-711 (verified identical rule) vs OXP:150-153.
    - **Agree** on the rule; composition differs (OX adds `group_order > 1`, G adds `!madelung && include_aes`).

13. **Ad-hoc AES loop.**
    - G:601-648 vs MD:906-929: SECCM adds weight `w`, pair vector `-disp`, gamma at `|disp|`; moments from
      the raw molecule G:1705, 1244-1246 (representative-dependent, GH:248-255).
    - **Differ.**

14. **`n_occ`, GAM3 l-scaling, reference occupations, V3 / E_3rd.**
    - G:1613-1617, 1661-1670 vs MD:720-736 vs PG2:246-263; V3 G:582-588 / MD:887-892 / PG2:449-453;
      E_3rd G:1110-1116 / MD:1031-1037 / PG2:538-544.
    - **Agree** (identical constants).

15. **Klopman-Ohno pair kernel.**
    - SC:631-657 (hand-rolled, `InverseHardnessMean`) vs PGC:117-125, 181-188 `shell_gamma_pair` vs
      `hamiltonian.cpp:197-217` vs OX:82-89 (gss-based) vs P6:426-427 (rho-based).
    - Hubbard fallback 0.4 at SC:543, 639-640, 649; S:1001-1007; `hamiltonian.cpp:217`.
    - **Agree** in form; S evaluates the KO value locally but its derivative via `shell_gamma_pair_derivative`
      (S:1016).

16. **Eigensolve / screening three-way branch.**
    - D0:138-158, S:150-162 + 356-367, G:1625-1635 + 732; P6:661 and OX:925 skip screening.
    - **Agree.**

17. **Fermi-Dirac occupations and entropy.**
    - S:172-184, 825-832; G:745-752; PG2:486-489 all call KO:249; MD:371-428 has its own bisection and
      computes entropy only when T > 0 (MD:1040-1049, verified).
    - **Agree** on the kernel; bookkeeping differs (`record_applied_occupation` S:69-93 has no G counterpart).

18. **Repulsive / core-core `0.5 w` loop.**
    - D0:180-189, S:703-712, G:337-346, P6:689-712, OX:738-741, MC:629-639 / CE:644-654; PG2:305-321
      (home `b > a` w = 1, images 0.5).
    - **Agree.**

19. **Replica / group-order validation.** B:817-870 vs S:262-291 (adds odd-3-D). **Agree** on the shared part.

20. **Acknowledgement gate.** P6:219-230 vs OX:333-345 vs OXP:154-174. **Differ** (Section 4, item 2).

21. **Hermiticity guard + symmetrise.** P6:596-604 vs OX:745-756; D0:127-136 vs S:346-355; G:1599-1612. **Agree.**

22. **`expected_gfn2_shell_charge_count` and restart validation.**
    - G:35-56 vs B:667-683; G:1304-1330, B:12524-12553, GP:772-793. **Agree.**

23. **Atom coordinate extraction.**
    - `seccm_atom_coords` SC:469-476 vs S:1058-1062 (verified), B:12318-12322, `hamiltonian.cpp:222-223`. **Agree.**

24. **D4 dispersion.**
    - Absent from every C++ driver and every SECCM adapter (GH:33-34; MD:15, 700, 1005 comments only);
      Python post-SCF only in `methods/gfn2.py:769-779`.
    - **Agree** (absent); the unified provenance record should say so explicitly.

25. **`SeccmReducedState` vs `ReducedState`.** G:500-549 vs PG2:113-152, same layout `[dq; mu(3n); theta(6n)]`. **Agree.**

26. **Madelung state wrapper.** G:450-461 vs S:405-407 vs P6:614-615 (item 4). **Agree.**

## 4. Disagreements: same concern, different behaviour

Each entry: evidence, consequence, and class in {wrong answer, provenance defect, policy inconsistency, cosmetic}.

1. **SCC-DFTB embedded 3-D gradient uses a 2-D Madelung matrix.**
   - Energy: S:414-416 `_madkonst_3d`. Gradient: S:862-868 `dim == 1 ? wire : _madkonst_2d` (both verified).
   - `_madkonst_2d` consumes `translations[0], translations[1]` with no size check (CE:471-473), so the
     `V_mad` entering `M` (S:884-896) is a slab potential of the first two vectors, while
     `_add_madelung_gradient` (S:1062-1064) uses the 3-D lattice (CE:983-1030 `dim == 3`, verified).
   - Reachable: SP:227-236 admits 3-D + Elstner + madelung; T:4241-4268 pins the energy (-0.400383374)
     but requests no gradient.
   - **Wrong answer** (unpinned).

2. **PM6 vs OMx acknowledgement gates.**
   - P6:219-220: `!madelung && !trivial_records && !ack`.
   - OX:333-334 and OXP:154-156: `(group_order > 1 || !trivial) && !ack` (verified).
   - A small molecule in a replicated cell whose WS inventory is all zero-label/unit-weight has
     `trivial == true` and `group_order > 1`: PM6 runs silently, OMx refuses. PM6's message quotes the
     rocksalt error (-43%..+38%); OMx's quotes none.
   - **Policy inconsistency.**

3. **PM6 3-D Madelung refused unconditionally** (P6:213-218) vs SCC-DFTB refusing only the KO form
   (S:255-261) vs GFN2 running 3-D madelung (G:1694-1695; T:4804).
   - PM6's justification is the same R^-3 remainder (P6:207-210, PGH:17-22) and PM6 has no Elstner-type
     alternative, so the refusal is kernel-consistent, but the *shape* of the gate differs.
   - **Policy inconsistency** (the omx map records that D7 lifted the hold on the PM6 3-D arm).

4. **`energy` means Mermin A in SECCM and E elsewhere.**
   - G:1128-1133: `total` includes `-entropy_term`, `free_energy = energy`; GH:59. S:635-636, 727 identical.
   - PG2:546-554 and MD:1039-1053 keep `energy = E`, `free_energy = E - TS` (all verified).
   - The delegation path converts explicitly (`energy = molecular.free_energy / N`, G:1449-1456).
   - Consumers pick by `smearing_temperature` (PY:515-520, 1000-1005). Pinned: T:2223; T:806-852
     (`energy == free_energy`).
   - **Policy inconsistency** (documented, not wrong; one result schema cannot carry both semantics under
     one field name).

5. **Ladder budgets.**
   - SECCM primary `min(500, total)` vs molecular `min(700, total)` with contraction extension vs periodic
     500-step checkpoint plus a single restart; first rung 2000 vs 1800; SECCM has no `auto_stabilize`
     (Section 3 item 2, verified).
   - Pinned only for SECCM: T:2276, 2417.
   - **Policy inconsistency.**

6. **SECCM ladder does not honour an explicit temperature.**
   - G:1757 passes `retry.electronic_temperature` unconditionally; MD:1195-1211 guards with
     `electronic_temperature_explicit` (vibe-qc#3); `GFN2SECCMOptions` has no such flag (GH:165).
   - On delegation the flag stays at its MDH:193 default, so the molecular ladder may also smear.
   - The executed T is reported honestly (G:1135; T:2352; GP:968-970 documents it).
   - **Policy inconsistency** with provenance intact.

7. **DFTB0 pair-table scope: two exception types.**
   - D0:78-86 `std::invalid_argument` (a `ValueError` at the native seam) vs D0P:123-133
     `NotImplementedError` (verified); TD:310-345 pins the Python type.
   - **Cosmetic / policy** (seam callers and wrapper callers see different types for one predicate).

8. **dftb0.py flatten copy skips two checks.**
   - Omits AC:53-56 (covered downstream by B:728-732, SC:111-114) and AC:71-78 `record_translation_orbits()`
     (not covered anywhere on the DFTB0 path).
   - DFTB0 admits a topology with a bound-but-broken orbit reduction that the other four refuse.
   - **Policy inconsistency** (minor, unpinned).

9. **SCC-DFTB provenance gaps.**
   - `gamma_form`, `ewald_gamma`, `madelung`, mixer absent from `SCCDFTBSECCMResult` (SH:48-109;
     SP:333-378, verified).
   - `electrostatics_family` is `"madelung" | "none"` (SP:266) although RY:68 accepts `"ewald_gamma"`; an
     `ewald_gamma=True` run resolves to kernel `"none"` (RY:146-147) and cites no `wire_ewald_1d` /
     `slab_ewald_2d` (RG:706-731 gate on the kernel string).
   - **Provenance defect** (and a citation defect).

10. **PM6 provenance gap.**
    - P6P:188-196 stamps `with_seccm_runtime(periodic_dimension=...)` only (verified), so `madelung=True`
      PM6 runs carry `electrostatics_family="none"` and cite no `ccm_madelung_embedding` row (RG:740-750).
    - **Provenance defect** (same class as item 9).

11. **GFN2 delegation forwards a subset.**
    - G:1401-1412 forwards `max_iter, conv_tol_charge, charge_mixing, electronic_temperature, scc_mixer,
      aes_faithful, aes_damping`; `initial_shell_charges` is refused on this path G:1323-1328 (pinned
      T:5082); `gamma_form` / `ewald_gamma*` are excluded by the gate G:1305-1307;
      `electronic_temperature_explicit` / `auto_stabilize` stay at MDH:193-195 defaults (verified).
    - A restart that works on the supercell path is unavailable on the one-replica torus.
    - **Policy inconsistency** (explicit fail-closed; not wrong).

12. **MSINDO citation selector (#64 = old #702, `docs/issue_renumbering_2026_09.md:155`).**
    - RY:97-100 maps `msindo` 1-D to `truncated_1d`; RG:740-750 selects `ccm_madelung_truncated_1d`
      (Bredow 2001 only, `database.toml:6951`) vs `ccm_madelung_embedding` (verified); pinned TM:1053-1054,
      1063-1092.
    - **Resolved -- cosmetic now.** The remaining asymmetry is physical: MSINDO's 1-D kernel is the truncated
      sum (MC:251-271) while S/G/P6 run the wire Ewald (E1:154-200).

13. **SCC-DFTB Aitken lacks the molecular finite-T branch.**
    - S:576-595 vs `cpp/src/semiempirical/scc_dftb.cpp:291-294` (verified): at T > 0 the SECCM path keeps
      extrapolating where the molecular path only damps.
    - Same fixed point, different iteration count; TB:207-264 pins `n_iter == 92`, TB:330-365 and
      T:761-798 pin `n_iter == 19`.
    - **Policy inconsistency** (visible through the iteration-count pins).

14. **`homo_lumo_gap` recording.**
    - G records the gap before deciding rejection (G:1165-1168) and on delegation (G:1526-1528); S assigns
      it only on the converged path (S:642), so a non-converged S result reports 0.0; P6/OX record only
      when converged (P6:755; OX:896).
    - **Provenance inconsistency.**

15. **Overlap-screening threshold source.**
    - D0:141 and S:356-367 use the `hermiticity_tolerance` argument; G:1625-1635 hard-codes 1e-10. All
      seams hard-code 1e-10 / 1e-8 (B:11769-11770, 11855-11856, 12403, 12463-12474, 12596).
    - **Cosmetic** today (identical numbers); a trap for any future knob.

16. **Convergence residual is method-family specific** (Section 3 item 8) and `conv_tol` defaults differ by
    four orders (1e-8 S, 1e-6 G, 1e-7 P6/OX, 1e-9 MS).
    - **Policy inconsistency** inherent to the families; the unification keeps a per-method residual
      adapter rather than one number.

17. **OMx `max_iter` default** Python 200 (OXP:72) vs C++/binding 100 (OXH:114; B:12488). **Cosmetic.**

18. **Stale text contradicting code.**
    - SH:129 ("rejects 3-D embedding" vs S:255-261); SH:166-186 (coupled-perturbed response vs S:1026-1030
      stationarity); P6H:11-12 and `docs/user_guide/semiempirical.md:614-616` ("no Madelung" vs P6:188-246);
      GH:21-24 (faithful AES "not part" vs GH:246-267); S:449-451 (wrapper "enables DIIS for madelung runs"
      vs SP:130).
    - P6:36 (`gamma_ab_multi` "verbatim from pm6_fock.cpp" vs `pm6_fock.cpp:33-59` differing formula;
      reachable only when `dg.g_ss == 0`, P6:482-488 -- reachability unverified).
    - **Cosmetic**, except the last, a potential wrong answer in an unreached branch.

## 5. Unification target (specification)

Constraints honoured throughout:
- The molecular-limit contract is the acceptance test of every step: T:146-160 (DFTB0 CH4, 1e-13),
  T:193-223 (SCC-DFTB, 1e-12), T:2052-2112 (GFN2 bit parity via delegation), T:1196-1222 (PM6, 5e-12),
  T:1629-1667 (OMx, 1e-12).
- `gamma_form` defaults stay KO on the SECCM routes (D1 held: SP:153-155, GH:239-241).
- `aes_faithful` stays `false` (GH:265-267) unless the maintainer moves it.
- The D2 odd-3-D replica gate stays SCC-DFTB-only (D0:54-58, S:283-291).
- `translation_symmetry_charge_spread` stays ungated (GP:325-335, D3). "D6" is not locatable in this
  checkout (grep over `python/vibeqc/semiempirical/seccm/` and `docs/` finds no D6 marker); it is taken as
  the instruction states.

### 5.1 Single owners

**(a) One topology.**
- Owner: Python `SECCMTopology` (TP:370-393) + `flatten_topology_records` (AC:16-124) +
  `seccm_validated_topology_from_records` (B:774-870) + `validate_common_inputs` (SC:83-246).
- Adapters call: dftb0.py switches to `flatten_topology_records` and `topology_length_unit_scale`
  (AC:127-129) instead of D0P:130-166; the other four already do.
- Method-specific, stays: OMx union WSSC OX:243-265 (only three-centre consumer); P6/OX `R < 1e-12` record
  skip (P6:362; OX:557); G `seccm_pair_weight` G:110-129 until faithful AES has a record-native lattice sum.
- MSINDO stays on bare `build_wigner_seitz` (MC:670); the C++ builder TH:222-269 remains for CE:712 and the
  test seam B:14763-14849; folding MSINDO into the finite-group contract is Section 6, Q8.
- Pins: TD:83-104, 294-307, 354-498; T:226-263; `tests/test_seccm_topology.py`; TM:159-224.

**(b) One shell-gamma kernel.**
- Owner: `periodic_gamma` -- `shell_gamma_pair` / `shell_gamma_onsite` (PGC:172-213) for values,
  `build_periodic_shell_gamma` (PGC:706-767) for the Ewald split, plus one new record-sum builder
  `build_ws_truncated_shell_gamma(sites, positions, spec, records)` = the remainder loop of PGC:737-752
  without the Ewald half and without the `-1/r` (i.e. `sum w gamma(r)` + on-site).
- Adapters call: S default path replaces SC:631-657 by the new builder with `spec{KO, InverseHardnessMean}`
  (the Elstner branch SC:600-627 already uses the shared kernel); G default path replaces G:307-332 by the
  same builder with `spec{KO, HardnessMean}` over `seccm_image_records`; the `ewald_gamma` paths
  (SC:568-584, G:133-172) already share `build_periodic_shell_gamma` and keep only their `GammaSite` layout
  and G's `molecular_onsite` post-subtraction (G:159-170).
- Method-specific, stays: P6 AO-resolved NDDO tensors (P6:390-543); OX `f_ko`-scaled channel blocks
  (OX:719-736); MSINDO `_gamma_shell` (MC:222-227).
- Pins: T:4191-4212 (KO default bitwise), T:4215-4238, T:3680 (tblite), T:3537, T:4271 / 4625 / 5628
  (eta -> 0), T:428-465 (chain convergence), T:2173-2222.

**(c) One Ewald split.**
- Owner: `EwaldCoulombKernel` (PGC:414-661).
- Adapters call: `wire_madkonst_1d` E1:154-200 becomes `sum_records w * kernel.potential(-disp)` with the
  kernel's 1-D branch; `_madkonst_2d` / `_madkonst_3d` (CE:573, 363) become the 2-D/3-D branches
  (PGC:566-599 already says "the validated indo::_madkonst_2d arithmetic"; 3-D background PGC:472-475 =
  CE:379-380); `wire_kernel_gradient` E1:215-301 becomes `kernel.gradient`; `seccm_ewald_alpha` SC:510-532
  is deleted (kernel default).
- One `MadelungState{ews, madkonst, kernel}` builder replaces the four dispatch copies (Section 3 item 1)
  and is the only place that switches on `translations.size()`.
- Method-specific, stays: PM6's 3-D refusal (P6:213-218) runs *before* the builder; MSINDO keeps CE for its
  own energy (oracle pins TM:672-765 at 5e-9) and its frozen truncated 1-D sum unless the maintainer moves
  it (Section 6, Q1).
- Pins: T:806-852, 931-957, 960-992, 1051-1100, 4241-4268 (-0.400383374), 571-596 (-0.114596161176388),
  2903, 2968, 4804, 3758, 374-404, 406-424, 6513-6540; TMC:766-815.

**(d) One mixer state vector.**
- Owner: `ChargeMixer` hierarchy (CM:62-221) plus a core `ReducedState` merged from G:500-549 and
  PG2:113-152, with shared `project_total_charge(dq, charge)` and `physical_extrapolation(dq, n_val)`
  (from S:498-513, 531-554).
- Adapters call: S selects `SimpleMixer` + Aitken option / `ChargeDIISMixer` / `BroydenMixer` through an
  `SCCMixer` enum instead of five booleans (B:11808-11812); G unchanged apart from taking the shared state;
  PG2 takes the shared state.
- Method-specific, stays: Fock-level DIIS for P6/OX/MSINDO (density-matrix state), which may share one
  `FockDIIS` struct with `{max_history, damping_start, damping_length}` as parameters so the P6:261-268 and
  OX:773-775 windows survive as data; G's Newton chord (G:804-887) needs the charge map and stays
  adapter-local (or becomes a `NewtonMixer` taking a closure).
- Pins: T:911-928; TB:207-264 (`n_iter == 92`); TB:330-365 and T:761-798 (`n_iter == 19`); T:4014;
  T:5354-5535; T:2609.

**(e) One occupation/temperature path.**
- Owner: `compute_closed_shell_kpoint_occupations` (KO:249) wrapped by one
  `frontier_guard(eps, n_occ, T, gap_tol) -> FrontierRecord{occupations, entropy, gap, gap_guard_waived,
  aufbau_deviation, aufbau}` built from SC:380-386 + S:599-621 + G:1165-1176 + `record_applied_occupation`
  S:69-93; and one `StabilizationLadder{primary_cap, rungs, honour_explicit_T, auto_stabilize}` shared by
  G:1712-1766 and MD:1163-1221 (PG2 can adopt it with a one-rung table).
- Adapters call: D0/P6/OX call `frontier_guard` with T = 0 (their local `eps(n_occ) - eps(n_occ-1)` equals
  the shared helper whenever `n_occ < n_basis`, enforced at P6:146-156 and OX:367-378); S and G call it with
  their T; `GFN2SECCMOptions` gains `electronic_temperature_explicit`, set by GP:896 on any assignment as
  B:14151-14159 does for the molecular options.
- Method-specific, stays: SECCM `energy = A` and `free_energy = energy` (pinned T:2223, T:806-852) unless
  the maintainer changes it (Section 6, Q2); the SECCM ladder numbers (500 / 2000) stay as the SECCM row
  of the ladder table.
- Pins: T:854-908, 2223, 2276, 2352, 2417, 2533; TD:107-126; TB:207-365; T:6028 (screened spectrum without
  LUMO, parametrised over dftb0 / scc_dftb / gfn2).

**(f) One provenance record.**
- Owner: a Python `SECCMRunProvenance` frozen dataclass in `_adapter_common.py` (or `routes.py`) with
  `{method_key, periodic_dimension, group_order, replicas, normalization, gamma_form, electrostatics_family
  (requested; resolved in {none, madelung, ewald_gamma}), electrostatics_kernel, requested_temperature,
  executed_temperature, mixer, hermiticity_tolerance, gap_tolerance, parameter_identity, parameter_sha256,
  topology_fingerprint, reference_geometry_fingerprint, attempts (optional),
  translation_symmetry_charge_spread (optional, ungated), dispersion = "none"}`, stamped through
  `route_plan.with_seccm_runtime(...)`. The GFN2 `hamiltonian_identity` (RY:739-759) and `run_controls`
  (RY:1068-1089) are the template.
- Adapters call: S adds `gamma_form`, `ewald_gamma`, mixer and emits `electrostatics_family="ewald_gamma"`
  when set; P6 emits `"madelung"` when set; D0/OX emit `"none"` explicitly; MSINDO `run_seccm` propagates
  `parameter_identity/sha256` (constructor already accepts them RN:252-271).
- Method-specific, stays: OX `three_center_weighting`, `truncated_electrostatics_acknowledged` (OX:767-771);
  G `requested_*` echoes and cross-checks (GP:912-963).
- Pins: TR:180-328; T:169-190, 1179-1193, 1576-1585, 2040; TD:501-544; `tests/test_citations.py:1581-1625,
  1660-1682`; TM:1015-1092.

### 5.2 Ordered steps

1. **dftb0.py -> `flatten_topology_records` + `topology_length_unit_scale`; keep `NotImplementedError` for
   the pair table.**
   - Behaviour-preserving by construction for every topology the current tests build (the added
     `record_translation_orbits()` check fires only when `record_orbits_currently_usable`; TD:276-291 covers
     the invalidated-action case).
   - Add a pin anyway: `test_dftb0_seccm_rejects_broken_bound_orbits` on the `_cyclic_h2` fixture (TD:26-67)
     with a bound action whose orbit is broken, mirroring T:226-263.

2. **Hoist the gradient bookkeeping copies (D0:259-267, 292-321, 336-344; S:803-811, 900-929, 944-952)
   onto SC:248-313, 414-419.**
   - Behaviour-preserving by construction (byte-identical loops, Section 3 item 11).
   - Pinned by TD:144-165, T:266-283, 286-320.

3. **One `MadelungState` builder (5.1c, first half: keep the E1/CE kernels, unify only the dispatch) and
   fix S:860-868.**
   - Energy paths S:409-416, G:1684-1696, P6:233-240: behaviour-preserving by construction.
   - S gradient: needs a fail-before/pass-after pin:
     `test_scc_dftb_seccm_3d_elstner_embedded_gradient_matches_fd` -- fixture: the 3-D cell of T:4241-4268
     with `gamma_form="elstner", madelung=True, compute_gradient=True`, central FD as in T:374-404; fails
     before (2-D madkonst), passes after.

4. **`frontier_guard` (5.1e, first half) replacing D0:167-171, S:599-621, G:1165-1176, 1522-1540,
   P6:681-686, OX:889-897, plus S:69-93.**
   - Behaviour-preserving by construction for the throw/waive predicate.
   - The one observable change is `homo_lumo_gap` becoming recorded on non-converged S results (Section 4
     item 14): add `test_scc_dftb_seccm_unconverged_result_records_the_frontier_gap` -- fixture: the
     MgO(100) embedded run of T:761-798 that raises "did not converge", asserting the gap on the carried result.
   - Existing pins: T:854-908; TB; T:6028; TD:107-126.

5. **`weighted_dftb_gamma_klopman_ohno` (SC:631-657) -> `shell_gamma_pair(spec{KO, InverseHardnessMean})`.**
   - Same closed form (PGC:117-125, 181-188) but not guaranteed bit-identical (operation order).
   - Needs a pin: `test_scc_dftb_seccm_klopman_ohno_gamma_matrix_is_unchanged` -- capture `e_scc` and
     `charges` on the T:428-465 chain at 1e-12 before the change. T:4191-4212 (default == explicit KO) remains.

6. **GFN2 default WS gamma loop (G:307-332) -> shared `build_ws_truncated_shell_gamma` with
   `spec{KO, HardnessMean}`.**
   - Needs a pin: capture `e_scc` and `shell_charges` of the T:2173-2222 chain and of T:3537 at 1e-12
     before the change.
   - T:2052-2112 is unaffected (delegation bypasses the WS gamma) but must keep passing.

7. **Delete `seccm_ewald_alpha`; pass `alpha = 0` to `EwaldCoulombKernel`.**
   - Behaviour-preserving by construction (identical formulas, verified SC:510-532 vs PGC:414-427).
   - T:3758 pins alpha-independence regardless.

8. **`wire_madkonst_1d` / `wire_kernel_gradient` -> `EwaldCoulombKernel` 1-D.**
   - Needs a pin: the truncation rules differ (E1:166-167 fixed 8 / 24 vs PGC:428-440, 496 cutoff-derived).
   - Fixtures: T:806-852 (`e_madelung`, `energy`), T:931-957 (`n2 > n4 > n6`), T:6513-6540 (PM6 1-D
     bookkeeping), T:374-404 (embedded gradient FD). Capture `e_madelung` before; if not equal within 1e-10
     the maintainer picks the canonical truncation (Section 6, Q1) and the pin is re-based once.

9. **`_madkonst_2d` / `_madkonst_3d` -> kernel 2-D/3-D for S/G/P6.**
   - Needs a pin: T:960-992, T:4241-4268 (-0.400383374), T:571-596 (-0.114596161176388), T:2903, T:4804
     already pin absolute values; add `test_seccm_madelung_kernel_matches_ccm_engine_bitwise` over the LiH
     sheet / bulk cells of TM:292-314, 390-429 comparing the new matrices to CE at 1e-12.
   - MSINDO keeps CE.

10. **Shared `ReducedState` (G:500-549 + PG2:113-152) in `core/`.**
    - Behaviour-preserving by construction (same layout).
    - Pins T:3947, 4014, 4084.

11. **S mixer selection through `SCCMixer` + shared charge guards; keep `vibeqc::DIIS` behind the interface.**
    - Behaviour-preserving by construction while `vibeqc::DIIS` stays (iteration counts unchanged:
      TB:207-264 `n_iter == 92`, T:761-798 `n_iter == 19`, T:911-928).
    - Replacing it by `ChargeDIISMixer` is a separate decision (Section 6, Q5) and would re-base those pins.

12. **`StabilizationLadder` shared by G and MD, with the SECCM row `{500; 2000 / 500 / 600}` and the
    molecular row `{700 + extension; 1800 / 500 / 600}`.**
    - Behaviour-preserving by construction (tables are data).
    - Pins T:2276, 2352, 2417, 2533 and the molecular attempt tests.

13. **Add `electronic_temperature_explicit` to `GFN2SECCMOptions`, honour it on rungs, forward it on delegation.**
    - Needs a pin: `test_gfn2_seccm_explicit_zero_temperature_is_honoured_by_the_ladder` -- fixture: the
      trivial-retry system of T:2352 with `electronic_temperature=0.0` passed explicitly; before: some attempt
      reports 0.005; after: every attempt reports 0.0 and the run either converges at T = 0 or fails closed.
    - T:2352 (implicit request) keeps its current outcome.

14. **`SECCMRunProvenance` (5.1f).**
    - Additive for D0/G/OX: behaviour-preserving by construction. S and P6 change the resolved family.
    - Needs pins: `test_scc_dftb_seccm_ewald_gamma_route_records_the_kernel` (fixture T:4215-4238; assert
      `route_plan.electrostatics_kernel == "ewald_gamma_wire_1d"` and citation `wire_ewald_1d`) and
      `test_pm6_seccm_madelung_route_records_the_kernel` (fixture T:6513-6540; assert `"madelung_wire_1d"`
      and the `ccm_madelung_embedding` row).
    - TR:180-328 and `tests/test_citations.py` remain.

15. **One acknowledgement predicate for P6 and OX.**
    - Needs a pin either way: T:6491-6510 (PM6), T:1670-1768 (OMx `group_order == 2` with ack),
      T:1629-1667 (OMx trivial records: ack recorded `false`).
    - Which predicate is the maintainer's (Section 6, Q4); the step is written after that answer.

16. **One `SECCMSolverTolerances{hermiticity = 1e-10, gap = 1e-8}` default consumed by every seam
    (B:11769, 11855, 12403, 12463, 12596) and by G:1625-1635.**
    - Behaviour-preserving by construction (same numbers).

17. **Delete stale comments** SH:129, SH:166-186, P6H:11-12, GH:21-24, S:449-451; correct P6:36; update
    `docs/user_guide/semiempirical.md:614-616`.
    - Behaviour-preserving by construction.

18. **MSINDO: propagate `parameter_identity/sha256` through `run_seccm` (RN:1149-1157).**
    - Additive; behaviour-preserving by construction.
    - Pin: TM:918-932 extended with the identity fields.

## 6. Open questions for the maintainer

1. **Canonical 1-D wire truncation** once E1 and PGC merge (fixed `8 / 24` vs `r_cut = sqrt(30)/alpha`,
   `k_cut = sqrt(120) alpha`): which becomes the pinned number for T:806-852 and T:931-957 if they differ
   beyond 1e-10? And does MSINDO's truncated +-2-shell 1-D sum stay frozen (RY:91-92; E1:3-4) after the
   unification, or does `run_ccm` move to the wire kernel with the TM:729-738 oracle re-pinned?
2. **`energy` semantics at T > 0:** keep SECCM `energy = Mermin A` (GH:59; S:635-636; pinned T:2223,
   T:806-852) or align with the periodic/molecular drivers (`energy = E`, `free_energy = E - TS`,
   PG2:551-554)? The shared provenance record cannot carry both under one field name; NEB / optimiser
   consumers pick by `smearing_temperature` (PY:515-520).
3. **Honour an explicit temperature on the SECCM ladder** (step 13) as the molecular driver does under
   vibe-qc#3 (MD:1195-1211)? This changes the documented behaviour at GP:968-970.
4. **Acknowledgement gate predicate:** PM6's `!trivial_records` (P6:219-220) or OMx's
   `group_order > 1 || !trivial` (OX:333-334)? Related: should PM6's 3-D Madelung refusal (P6:213-218)
   become kernel-shaped like SCC-DFTB's (S:255-261) once the D7-unblocked PM6 3-D arm lands, given PM6 has
   no Elstner alternative?
5. **SCC-DFTB charge DIIS:** keep `vibeqc::DIIS` on the charge vector (S:452-455) with its pinned iteration
   counts (TB:207-264, T:761-798), or move to `ChargeDIISMixer` (CM:115-145) and re-pin?
6. **SCC-DFTB 3-D embedded gradient (Section 4 item 1):** fix per step 3, or refuse `compute_gradient` for
   `dim == 3 && madelung` until the FD pin exists?
7. **DFTB0 scope exception type:** is `NotImplementedError` (D0P:123-133, TD:310-345) or the native
   `ValueError` (D0:78-86) the contract for the pair-table predicate?
8. **Is MSINDO in scope** for one topology (bound finite group + length unit, AC:43-60) and for the kernel
   merge (step 9 excludes it), or does the legacy arm stay frozen with its oracle pins (TM 5e-9) as the
   reference?
9. **D3 gate decision on `translation_symmetry_charge_spread`** (GP:333 "The gate decision deferred by D3
   is due"): confirmed ungated for this refactor; is a threshold to be scheduled separately?
10. **`aes_damping` on the SECCM surface:** `ctx.aes_damping` is set (G:1681) but never read on the
    supercell path (GH:270-273); drop it from `GFN2SECCMOptions` (keeping the delegation forward, G:1412)
    or wire it?
