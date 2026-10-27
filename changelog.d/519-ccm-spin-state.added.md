### Added: an explicit spin-state contract for the neutral Γ-CCM producers (#519)

`CCMSystem(..., spin_state=)`, the real-Gamma and neutral-Bloch producers,
both runner arms and `run_periodic_job(aiccm_spin_state=)` name the spin
sector of the Born-von Karman torus: `"legacy"` (the default, each
producer's historical convention, unchanged number for number),
`"per-cell"` (the unit cell's `(n_alpha, n_beta)` replicated with spins
aligned on both producers) or an integer total multiplicity. The sector is
resolved before the shell-aware dispatch, impossible requests fail at
construction, the executed counts are recorded on every result, in the
`.system` manifest (`aiccm_spin_policy`, `aiccm_spin_state`) and in the
QVF convention record, and a same-Hamiltonian comparison is refused when
the two sides executed different sectors
(`assert_compatible_spin_state`, `assert_comparable_neutral_results`).
Under the default a doublet cell on a torus of more than one cell still runs
as a supercell singlet on `real-gamma` and as a per-cell doublet on
`neutral-bloch`; the mismatch is now recorded and refused instead of
silent. AICCM/CCM stays experimental.
