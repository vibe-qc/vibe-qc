### Added: an occupation record on the neutral Γ-CCM producers (#519)

Every real-Gamma and neutral-Bloch result records how the spin sector's
electrons were distributed (`result.occupation`: integer or ensemble, the
largest departure from an integer, the Bloch producer's per-k counts, the
frontier gap and whether it is degenerate), in both convention records and
the `.system` manifest (`aiccm_occupation`). A same-Hamiltonian comparison
(`assert_comparable_neutral_results`) now also refuses an ensemble state by
name before any fold: the multi-k zero-temperature fill shares a
frontier-degenerate group equally, which has no determinant on the torus.
AICCM/CCM stays experimental.
