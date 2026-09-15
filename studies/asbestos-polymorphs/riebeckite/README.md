# riebeckite

Na₂Fe₃²⁺Fe₂³⁺Si₈O₂₂(OH)₂, C2/m. Whittaker 1949. Mixed Fe²⁺/Fe³⁺ — non-trivial spin state; consider broken-symmetry initial guesses.

**Status: blocked on `feature/uks-periodic-gdf` landing.**

This mineral has unpaired Fe spins (open-shell) and cannot be run
with the current closed-shell-only GDF spike. Once
`feature/uks-periodic-gdf` merges to main:

  - `riebeckite-UKS-PBE0-pobdzvp.py`  (today's default-hybrid; will
                                    be added by that feature chip)
  - `riebeckite-UKS-PW1PW-pobdzvp.py` (paper-comparable hybrid; needs
                                    UKS + PW1PW both landing)

For the paper-grade run we'll also want broken-symmetry initial
guesses to characterize antiferromagnetic vs ferromagnetic ground
states — that's an additional knob beyond plain UKS.

CIF should be dropped at `../cifs/riebeckite.cif` per
`../cifs/CIFS_NEEDED.md`.
