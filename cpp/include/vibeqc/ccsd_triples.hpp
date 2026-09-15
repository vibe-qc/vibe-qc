#pragma once
/// On-the-fly CCSD(T) triples energy correction.
///
/// Computes the Crawford-Schaefer (T) correction E_T without materialising
/// 6-index intermediate tensors.

#include <cstddef>

namespace vibeqc {

double ccsd_t_triples(
    const double* t1, const double* t2,
    const double* eri_as,
    const double* eps_o, const double* eps_v,
    int n_occ, int n_vir, int nso);

}  // namespace vibeqc
