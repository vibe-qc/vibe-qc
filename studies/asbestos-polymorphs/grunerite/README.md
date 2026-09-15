# grunerite

Fe₇²⁺Si₈O₂₂(OH)₂, C2/m. Hawthorne 1983. All-Fe²⁺; high-spin ferromagnetic is the simplest test.

**Status: blocked on `feature/uks-periodic-gdf` landing.**

This mineral has unpaired Fe spins (open-shell) and cannot be run
with the current closed-shell-only GDF spike. Once
`feature/uks-periodic-gdf` merges to main:

  - `grunerite-UKS-PBE0-pobdzvp.py`  (today's default-hybrid; will
                                    be added by that feature chip)
  - `grunerite-UKS-PW1PW-pobdzvp.py` (paper-comparable hybrid; needs
                                    UKS + PW1PW both landing)

For the paper-grade run we'll also want broken-symmetry initial
guesses to characterize antiferromagnetic vs ferromagnetic ground
states — that's an additional knob beyond plain UKS.

CIF should be dropped at `../cifs/grunerite.cif` per
`../cifs/CIFS_NEEDED.md`.
