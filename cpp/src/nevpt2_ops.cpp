#include "vibeqc/nevpt2_ops.hpp"

#include <cmath>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeqc {

// Flat index for chemist's 4-index tensor (pq|rs), row-major.
inline int _eri_idx(int p, int q, int r, int s, int norb) {
    return ((p * norb + q) * norb + r) * norb + s;
}

std::unordered_map<uint64_t, double> apply_1body_cpp(
    const std::unordered_map<uint64_t, double>& state,
    const Eigen::MatrixXd& h1,
    int norb,
    const std::vector<int>* idx)
{
    const int n_rng = (idx != nullptr) ? static_cast<int>(idx->size()) : norb;

    // Collect state entries into vector for OpenMP iteration
    std::vector<std::pair<uint64_t, double>> entries(state.begin(), state.end());
    std::unordered_map<uint64_t, double> out;
    out.reserve(state.size() * 2);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::unordered_map<uint64_t, double> local;
        local.reserve(entries.size() * 2 / omp_get_num_threads());
#pragma omp for schedule(static)
        for (size_t idx_entry = 0; idx_entry < entries.size(); ++idx_entry) {
            const auto& [mask, c] = entries[idx_entry];
#else
        for (const auto& [mask, c] : entries) {
#endif
            for (int qi = 0; qi < n_rng; ++qi) {
                int q = (idx != nullptr) ? (*idx)[qi] : qi;
                for (int s = 0; s < 2; ++s) {
                    auto [sgq, mq] = _ann(mask, _so(q, s, norb));
                    if (sgq == 0) continue;
                    for (int pi = 0; pi < n_rng; ++pi) {
                        int p = (idx != nullptr) ? (*idx)[pi] : pi;
                        double hpq = h1(p, q);
                        if (hpq == 0.0) continue;
                        auto [sgp, mp] = _cre(mq, _so(p, s, norb));
                        if (sgp) {
#ifdef _OPENMP
                            local[mp] += static_cast<double>(sgq * sgp) * hpq * c;
#else
                            out[mp] += static_cast<double>(sgq * sgp) * hpq * c;
#endif
                        }
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical
        for (const auto& [m, v] : local) out[m] += v;
    }
#endif
    return out;
}

std::unordered_map<uint64_t, double> apply_2body_cpp(
    const std::unordered_map<uint64_t, double>& state,
    const std::vector<double>& eri,
    int norb,
    const std::vector<int>* idx)
{
    const int n_rng = (idx != nullptr) ? static_cast<int>(idx->size()) : norb;
    std::vector<std::pair<uint64_t, double>> entries(state.begin(), state.end());
    std::unordered_map<uint64_t, double> out;
    out.reserve(state.size() * 4);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::unordered_map<uint64_t, double> local;
        local.reserve(entries.size() * 4 / omp_get_num_threads());
#pragma omp for schedule(static)
        for (size_t idx_e = 0; idx_e < entries.size(); ++idx_e) {
            const auto& [mask, c] = entries[idx_e];
#else
        for (const auto& [mask, c] : entries) {
#endif
            for (int qi = 0; qi < n_rng; ++qi) {
                int q = (idx != nullptr) ? (*idx)[qi] : qi;
                for (int sq = 0; sq < 2; ++sq) {
                    auto [sgq, mq] = _ann(mask, _so(q, sq, norb));
                    if (sgq == 0) continue;
                    for (int si = 0; si < n_rng; ++si) {
                        int s_orb = (idx != nullptr) ? (*idx)[si] : si;
                        for (int st = 0; st < 2; ++st) {
                            auto [sgs, ms] = _ann(mq, _so(s_orb, st, norb));
                            if (sgs == 0) continue;
                            for (int ri = 0; ri < n_rng; ++ri) {
                                int r = (idx != nullptr) ? (*idx)[ri] : ri;
                                auto [sgr, mr] = _cre(ms, _so(r, st, norb));
                                if (sgr == 0) continue;
                                for (int pi = 0; pi < n_rng; ++pi) {
                                    int p = (idx != nullptr) ? (*idx)[pi] : pi;
                                    double v = eri[_eri_idx(p, q, r, s_orb, norb)];
                                    if (v == 0.0) continue;
                                    auto [sgp, mp] = _cre(mr, _so(p, sq, norb));
                                    if (sgp) {
                                        double contrib = 0.5 * static_cast<double>(sgq * sgs * sgr * sgp) * v * c;
#ifdef _OPENMP
                                        local[mp] += contrib;
#else
                                        out[mp] += contrib;
#endif
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical
        for (const auto& [m, v] : local) out[m] += v;
    }
#endif
    return out;
}

double dot_cpp(const std::unordered_map<uint64_t, double>& a,
               const std::unordered_map<uint64_t, double>& b)
{
    const auto* small = &a;
    const auto* large = &b;
    if (a.size() > b.size()) {
        small = &b;
        large = &a;
    }
    double result = 0.0;
    for (const auto& [mask, ca] : *small) {
        auto it = large->find(mask);
        if (it != large->end()) {
            result += ca * it->second;
        }
    }
    return result;
}

std::unordered_map<uint64_t, double> add_cpp(
    const std::unordered_map<uint64_t, double>& a,
    const std::unordered_map<uint64_t, double>& b)
{
    std::unordered_map<uint64_t, double> out = a;
    for (const auto& [mask, cb] : b) {
        out[mask] += cb;
    }
    return out;
}

void state_rdm12_cpp(
    const std::unordered_map<uint64_t, double>& state,
    int norb,
    Eigen::MatrixXd& rdm1,
    Eigen::MatrixXd& rdm2)
{
    // rdm1: (norb, norb), rdm2: (norb^2, norb^2) flat (row-major)
    int norb2 = norb * norb;
    rdm1.setZero(norb, norb);
    rdm2.setZero(norb2, norb2);

    // Precompute E_pq|c> for all (p,q) to avoid O(norb^4) repeated work.
    // T2[p,q,r,s] = <c|E_qp E_rs|c> = (E_qp c) · (E_rs c)
    // For each state, we compute all E_rs actions once.
    std::vector<std::unordered_map<uint64_t, double>> Ec(norb2);
    std::vector<std::unordered_map<uint64_t, double>> EcT(norb2);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int r = 0; r < norb; ++r) {
        for (int s = 0; s < norb; ++s) {
            int idx_rs = r * norb + s;
            Ec[idx_rs].reserve(state.size() * 2);
            EcT[idx_rs].reserve(state.size() * 2);
            // Compute E_rs|c> and E_sr|c> (for bra side as (E_sr c)| = <c|E_rs)
            for (const auto& [mask, c_val] : state) {
                for (int spin = 0; spin < 2; ++spin) {
                    auto [sg_s, m_s] = _ann(mask, _so(s, spin, norb));
                    if (sg_s != 0) {
                        auto [sg_r, m_rs] = _cre(m_s, _so(r, spin, norb));
                        if (sg_r) {
                            double contrib = static_cast<double>(sg_s * sg_r) * c_val;
                            Ec[idx_rs][m_rs] += contrib;
                        }
                    }
                    // Also for bra (p,q) = (s,r): E_sr = (E_rs)^+
                    auto [sg_r2, m_r2] = _ann(mask, _so(r, spin, norb));
                    if (sg_r2 == 0) continue;
                    auto [sg_s2, m_sr2] = _cre(m_r2, _so(s, spin, norb));
                    if (sg_s2) {
                        double contrib = static_cast<double>(sg_r2 * sg_s2) * c_val;
                        EcT[idx_rs][m_sr2] += contrib;
                    }
                }
            }
        }
    }

    // rdm1[p,q] = c · Ec[(p,q)] (diagonal part) + off-diagonal
    for (int p = 0; p < norb; ++p) {
        for (int q = 0; q < norb; ++q) {
            int idx_pq = p * norb + q;
            for (const auto& [mask, c_val] : state) {
                auto it = Ec[idx_pq].find(mask);
                if (it != Ec[idx_pq].end()) {
                    rdm1(p, q) += c_val * it->second;
                }
            }
        }
    }

    // T2[p,q,r,s] = EcT[(p,q)] · Ec[(r,s)]
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int r = 0; r < norb; ++r) {
        for (int s = 0; s < norb; ++s) {
            int idx_rs = r * norb + s;
            const auto& v_rs = Ec[idx_rs];
            for (int p = 0; p < norb; ++p) {
                for (int q = 0; q < norb; ++q) {
                    int idx_pq = p * norb + q;
                    const auto& v_pq = EcT[idx_pq];
                    // Dot product: iterate over smaller map
                    const auto* small = &v_pq;
                    const auto* large = &v_rs;
                    if (v_pq.size() > v_rs.size()) {
                        small = &v_rs; large = &v_pq;
                    }
                    double dot = 0.0;
                    for (const auto& [m, cv] : *small) {
                        auto it = large->find(m);
                        if (it != large->end()) dot += cv * it->second;
                    }
                    // T2 is stored as (pq, rs) matrix: row = p*norb+q, col = r*norb+s
                    rdm2(idx_pq, idx_rs) = dot;
                }
            }
        }
    }

    // Reduce T2 to 2-RDM: rdm2[p,q,r,s] = T2[p,q,r,s] - delta_{qr} * rdm1[p,s]
    for (int p = 0; p < norb; ++p) {
        for (int q = 0; q < norb; ++q) {
            int idx_pq = p * norb + q;
            for (int r = 0; r < norb; ++r) {
                for (int s = 0; s < norb; ++s) {
                    int idx_rs = r * norb + s;
                    if (q == r) {
                        rdm2(idx_pq, idx_rs) -= rdm1(p, s);
                    }
                }
            }
        }
    }
}

// Apply a single E_xv operator (a+_v a_x summed over spins) to a sparse state.
// intermediate = E_xv |input>.
static std::unordered_map<uint64_t, double> _apply_E_xv(
    const std::unordered_map<uint64_t, double>& input,
    int v, int x, int norb)
{
    std::unordered_map<uint64_t, double> out;
    out.reserve(input.size() * 2);
    for (const auto& [mask, c_val] : input) {
        for (int spin = 0; spin < 2; ++spin) {
            auto [sg_x, m_x] = _ann(mask, _so(x, spin, norb));
            if (sg_x == 0) continue;
            auto [sg_v, m_vx] = _cre(m_x, _so(v, spin, norb));
            if (sg_v) {
                out[m_vx] += static_cast<double>(sg_x * sg_v) * c_val;
            }
        }
    }
    return out;
}

// Dot product of two sparse state vectors.
static double _dot_vec(const std::unordered_map<uint64_t, double>& a,
                       const std::unordered_map<uint64_t, double>& b)
{
    const auto* small = &a;
    const auto* large = &b;
    if (a.size() > b.size()) { small = &b; large = &a; }
    double result = 0.0;
    for (const auto& [mask, ca] : *small) {
        auto it = large->find(mask);
        if (it != large->end()) result += ca * it->second;
    }
    return result;
}

// Flat index for 6-index tensor G3[t,u,v,x,y,z] in E-operator convention.
// Convention: G3[t,u,v,x,y,z] = <0|E_tuvxyz|0>  (t,u,v,x,y,z are active-relative)
// Stored: index = ((((t*n + v)*n + x)*n + y)*n + z)*n + u  where n = n_act.
inline int _g3_idx(int t, int u, int v, int x, int y, int z, int n) {
    return ((((t * n + v) * n + x) * n + y) * n + z) * n + u;
}

// Flat index for T3[t,u,v,x,y,z] = <0|E_tu E_vx E_yz|0> (same layout as G3).
inline int _t3_idx(int t, int u, int v, int x, int y, int z, int n) {
    return ((((t * n + v) * n + x) * n + y) * n + z) * n + u;
}

void state_rdm3_cpp(
    const std::unordered_map<uint64_t, double>& state,
    int norb,
    int n_act,
    std::vector<double>& rdm3)
{
    // G3[t,u,v,x,y,z] = <psi| E_tuvxyz |psi>  (E-operator convention, OpenMolcas G3).
    //
    // Algorithm (mkfg3.F90, lines 500-713):
    //   1. Precompute Ec[p,q] = E_pq|c> for all active (p,q).
    //   2. Compute T3[t,u,v,x,y,z] = <0|E_tu E_vx E_yz|0>
    //      using T3 = Ec[u,t]^T · (E_vx · Ec[y,z])  (bra indices are transposed).
    //   3. Apply delta-function subtractions to get G3 from T3:
    //      - if y==x: G3 -= G2[t,u,v,z]; if v==u: G3 -= G1[t,z]
    //      - if v==u: G3 -= G2[t,x,y,z]
    //      - if y==u: G3 -= G2[v,x,t,z]
    //   4. G1[t,u] = dot(c, Ec[t,u]),  G2[t,u,v,x] = dot(Ec[u,t], Ec[v,x])

    const int n = n_act;
    const int n2 = n * n;
    const int n6 = n2 * n2 * n2;  // n_act^6
    rdm3.assign(n6, 0.0);

    if (state.empty() || n_act == 0) return;

    // ---- Step 1: Precompute Ec[p,q] = E_pq|c> for all active (p,q) ----
    std::vector<std::unordered_map<uint64_t, double>> Ec(n2);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int p = 0; p < n_act; ++p) {
        for (int q = 0; q < n_act; ++q) {
            int idx_pq = p * n + q;
            Ec[idx_pq].reserve(state.size() * 2);
            for (const auto& [mask, c_val] : state) {
                for (int spin = 0; spin < 2; ++spin) {
                    auto [sg_q, m_q] = _ann(mask, _so(q, spin, norb));
                    if (sg_q == 0) continue;
                    auto [sg_p, m_pq] = _cre(m_q, _so(p, spin, norb));
                    if (sg_p) {
                        double contrib = static_cast<double>(sg_q * sg_p) * c_val;
                        Ec[idx_pq][m_pq] += contrib;
                    }
                }
            }
        }
    }

    // ---- Step 2: G1[t,u] = <0|E_tu|0> = dot(c, Ec[t,u]) ----
    std::vector<double> G1(n2, 0.0);
    for (int t = 0; t < n_act; ++t) {
        for (int u = 0; u < n_act; ++u) {
            G1[t * n + u] = _dot_vec(state, Ec[t * n + u]);
        }
    }

    // ---- Step 3: G2[t,u,v,x] = <0|E_tu E_vx|0> ----
    // G2[t,u,v,x] = dot(Ec[u,t], Ec[v,x])
    // In E-operator convention, E_tu E_vx = E_tuvx + delta_uv E_tx (no, different)
    // Actually: <0|E_tu E_vx|0> = dot(Ec[u,t], Ec[v,x])  (this IS the raw T2)
    // The proper 2-RDM in E-operator conv: G2_E[t,u,v,x] = T2 - delta_uv G1[t,x]
    // But for delta-function subtractions of G3, we need T2 (pre-delta).
    std::vector<double> T2(n2 * n2, 0.0);  // T2[t,u,v,x]
    for (int t = 0; t < n_act; ++t) {
        for (int u = 0; u < n_act; ++u) {
            const auto& bra = Ec[u * n + t];  // E_ut|c> for bra side
            for (int v = 0; v < n_act; ++v) {
                for (int x = 0; x < n_act; ++x) {
                    const auto& ket = Ec[v * n + x];  // E_vx|c>
                    int idx_tuvx = ((t * n + u) * n + v) * n + x;
                    T2[idx_tuvx] = _dot_vec(bra, ket);
                }
            }
        }
    }

    // ---- Step 4: Compute T3[t,u,v,x,y,z] = <0|E_tu E_vx E_yz|0> ----
    // T3 = dot(Ec[u,t], E_vx(Ec[y,z]))
    // Outer loop: (y,z) — the rightmost E operator on ket.
    // Middle loop: (v,x) — apply to intermediate.
    // Inner loop: (t,u) — dot with bra.
    std::vector<double> T3(n6, 0.0);

    // Collect Ec entries to iterate over them in OpenMP
    std::vector<int> yz_list;
    yz_list.reserve(n2);
    for (int y = 0; y < n_act; ++y)
        for (int z = 0; z < n_act; ++z)
            yz_list.push_back(y * n + z);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i_yz = 0; i_yz < yz_list.size(); ++i_yz) {
        int yz = yz_list[i_yz];
        int y = yz / n;
        int z = yz % n;
        const auto& Ec_yz = Ec[yz];  // E_yz|c>
        if (Ec_yz.empty()) continue;

        // Thread-local T3 accumulation to avoid critical sections
        for (int v = 0; v < n_act; ++v) {
            for (int x = 0; x < n_act; ++x) {
                auto intermediate = _apply_E_xv(Ec_yz, v, x, norb);
                if (intermediate.empty()) continue;
                for (int t = 0; t < n_act; ++t) {
                    for (int u = 0; u < n_act; ++u) {
                        const auto& bra = Ec[u * n + t];  // E_ut|c>
                        if (bra.empty()) continue;
                        double val = _dot_vec(bra, intermediate);
                        if (val != 0.0) {
                            int idx = _t3_idx(t, u, v, x, y, z, n);
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            T3[idx] += val;
                        }
                    }
                }
            }
        }
    }

    // ---- Step 5: Build E-operator 2-RDM G2_E from T2 (mkfg3 lines 601-618) ----
    std::vector<double> G2_E(n2 * n2, 0.0);
    for (int t = 0; t < n_act; ++t) {
        for (int u = 0; u < n_act; ++u) {
            for (int v = 0; v < n_act; ++v) {
                for (int x = 0; x < n_act; ++x) {
                    int idx4 = ((t * n + u) * n + v) * n + x;
                    G2_E[idx4] = T2[idx4];
                    if (u == v) G2_E[idx4] -= G1[t * n + x];
                }
            }
        }
    }

    // ---- Step 6: Delta-function subtractions (mkfg3 lines 706-713) ----
    // Form <0|E_tuvxyz|0> from <0|E_tu E_vx E_yz|0> using corrected G2_E.
    for (int t = 0; t < n_act; ++t) {
        for (int u = 0; u < n_act; ++u) {
            for (int v = 0; v < n_act; ++v) {
                for (int x = 0; x < n_act; ++x) {
                    for (int y = 0; y < n_act; ++y) {
                        for (int z = 0; z < n_act; ++z) {
                            int idx = _g3_idx(t, u, v, x, y, z, n);
                            double val = T3[idx];
                            if (y == x) val -= G2_E[((t * n + u) * n + v) * n + z];
                            if (v == u) val -= G2_E[((t * n + x) * n + y) * n + z];
                            if (y == u) val -= G2_E[((v * n + x) * n + t) * n + z];
                            rdm3[idx] = val;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace vibeqc
