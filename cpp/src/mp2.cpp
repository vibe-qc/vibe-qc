#include "vibeqc/mp2.hpp"

#include <libint2/engine.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "vibeqc/correlation_conventions.hpp"
#include "vibeqc/df.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/thread_pool.hpp"

namespace vibeqc {

namespace {

using RowMat = mp2_detail::RowMatrix;

struct Tensor4 {
    std::vector<double> data;
    std::size_t d1 = 0, d2 = 0, d3 = 0, d4 = 0;

    void resize(std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
        d1 = a;
        d2 = b;
        d3 = c;
        d4 = d;
        data.assign(a * b * c * d, 0.0);
    }

    double operator()(std::size_t a, std::size_t b,
                      std::size_t c, std::size_t d) const {
        return data[((a * d2 + b) * d3 + c) * d4 + d];
    }
};

std::size_t sat_add(std::size_t a, std::size_t b) {
    const auto mx = std::numeric_limits<std::size_t>::max();
    return b > mx - a ? mx : a + b;
}

std::size_t sat_mul(std::size_t a, std::size_t b) {
    const auto mx = std::numeric_limits<std::size_t>::max();
    if (a == 0 || b == 0) return 0;
    return a > mx / b ? mx : a * b;
}

std::size_t bytes_for(std::initializer_list<std::size_t> dims) {
    std::size_t n = sizeof(double);
    for (const auto d : dims) n = sat_mul(n, d);
    return n;
}

std::size_t exact_incore_workspace(std::size_t n, std::size_t no,
                                   std::size_t nv) {
    // C_occ and C_vir are owned copies whose combined size is n*n.
    std::size_t total = bytes_for({n, n});
    total = sat_add(total, bytes_for({n, n, n, n}));
    total = sat_add(total, bytes_for({no, n, n, n}));
    total = sat_add(total, bytes_for({no, nv, n, n}));
    total = sat_add(total, bytes_for({no, nv, no, n}));
    total = sat_add(total, bytes_for({no, nv, no, nv}));
    return total;
}

// The direct four-stage transform releases each predecessor before forming
// the next successor. These are the three live phase overlaps per bra-
// occupied orbital: I1+I2, I2+I3, and I3+the OVOV output panel.
std::size_t exact_direct_bytes_per_i(std::size_t n, std::size_t no,
                                     std::size_t nv) {
    const auto i1 = bytes_for({n, n, n});
    const auto i2 = bytes_for({nv, n, n});
    const auto i3 = bytes_for({nv, no, n});
    const auto panel = bytes_for({nv, no, nv});
    return std::max({sat_add(i1, i2), sat_add(i2, i3),
                     sat_add(i3, panel)});
}

struct DFWorkspace {
    std::size_t construction = 0;
    std::size_t resident = 0;
    std::size_t incore = 0;
};

DFWorkspace df_workspace(std::size_t n, std::size_t naux,
                         std::size_t no, std::size_t nv, bool use_float) {
    const std::size_t fixed_mo = bytes_for({n, n});
    const std::size_t three = bytes_for({naux, n, n});
    const std::size_t metric = bytes_for({naux, naux});
    const std::size_t nov = sat_mul(no, nv);
    const std::size_t b_mo = bytes_for({naux, nov});
    const std::size_t ovov = bytes_for({nov, nov});

    DFWorkspace w;
    // DensityFitting construction overlaps raw T, T_flat, solved B_flat,
    // and the unpacked B_per_P representation while V, LLT storage, and L
    // are all live. It subsequently retains T_flat + B_per_P + L; MP2
    // additionally retains B_mo.
    const std::size_t constructor = sat_add(
        fixed_mo, sat_add(sat_mul(4, three), sat_mul(3, metric)));
    w.resident = sat_add(
        fixed_mo, sat_add(sat_add(sat_mul(2, three), metric), b_mo));
    const std::size_t transform_workers = static_cast<std::size_t>(
        omp_workers_for(naux, omp_max_threads()));
    const std::size_t transform_scratch = sat_mul(
        transform_workers,
        sat_add(bytes_for({no, n}), bytes_for({no, nv})));
    w.construction = std::max(
        constructor, sat_add(w.resident, transform_scratch));
    std::size_t contraction = sat_add(w.resident, ovov);
    if (use_float) {
        contraction = sat_add(contraction, b_mo / 2);
        contraction = sat_add(contraction, ovov / 2);
    }
    w.incore = std::max(w.construction, contraction);
    return w;
}

enum class MemoryMode { Auto, Incore, Direct, Disk };

MemoryMode parse_memory_mode(const std::string& raw) {
    std::string mode;
    mode.reserve(raw.size());
    for (const unsigned char c : raw) {
        mode.push_back(static_cast<char>(std::tolower(c)));
    }
    if (mode.empty() || mode == "auto") return MemoryMode::Auto;
    if (mode == "incore") return MemoryMode::Incore;
    if (mode == "direct") return MemoryMode::Direct;
    if (mode == "disk") return MemoryMode::Disk;
    throw std::invalid_argument(
        "run_mp2: memory_mode must be one of auto, incore, direct, or disk");
}

MemoryMode resolve_memory_mode(MemoryMode requested, std::size_t budget,
                               std::size_t incore_bytes) {
    if (requested == MemoryMode::Auto) {
        if (budget == 0 || incore_bytes <= budget) return MemoryMode::Incore;
        return MemoryMode::Direct;
    }
    if (requested == MemoryMode::Incore && budget != 0 &&
        incore_bytes > budget) {
        throw std::runtime_error(
            "run_mp2: requested in-core MP2 workspace exceeds "
            "requested_memory_bytes; use memory_mode='auto', 'direct', "
            "or 'disk'");
    }
    return requested;
}

const char* mode_name(MemoryMode mode) {
    switch (mode) {
        case MemoryMode::Incore: return "incore";
        case MemoryMode::Direct: return "direct";
        case MemoryMode::Disk: return "disk";
        case MemoryMode::Auto: break;
    }
    return "auto";
}

struct SlabPlan {
    std::size_t occupied_count = 0;
    std::size_t workspace_bytes = 0;
};

SlabPlan plan_occupied_slabs(std::size_t no, std::size_t bytes_per_i,
                             std::size_t resident_bytes,
                             std::size_t construction_bytes,
                             std::size_t budget_bytes,
                             const char* method_name) {
    const std::size_t minimum = std::max(
        construction_bytes, sat_add(resident_bytes, bytes_per_i));
    if (budget_bytes != 0 && minimum > budget_bytes) {
        throw std::runtime_error(
            std::string(method_name) +
            ": requested_memory_bytes is below the minimum one-occupied-"
            "orbital slab workspace for the selected route");
    }
    constexpr std::size_t kDefaultSlabTarget = 64ULL << 20;
    const std::size_t room = budget_bytes == 0
        ? kDefaultSlabTarget
        : budget_bytes - resident_bytes;
    std::size_t count = bytes_per_i == 0 ? no : room / bytes_per_i;
    count = std::max<std::size_t>(1, std::min(no, count));
    const std::size_t live = sat_add(resident_bytes,
                                     sat_mul(count, bytes_per_i));
    return {count, std::max(construction_bytes, live)};
}

class SlabScratchFile {
public:
    explicit SlabScratchFile(const std::string& requested_directory) {
        namespace fs = std::filesystem;
        const fs::path base = requested_directory.empty()
            ? fs::temp_directory_path()
            : fs::path(requested_directory);
        std::error_code ec;
        if (!fs::exists(base, ec) || !fs::is_directory(base, ec)) {
            throw std::runtime_error(
                "run_mp2: scratch_directory does not exist or is not a directory");
        }

        static std::atomic<unsigned long long> sequence{0};
        const auto stamp = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        for (unsigned int attempt = 0; attempt < 100; ++attempt) {
            const auto id = sequence.fetch_add(1, std::memory_order_relaxed);
            directory_ = base / ("vibeqc-mp2-" + std::to_string(stamp) + "-" +
                                 std::to_string(id));
            if (fs::create_directory(directory_, ec)) break;
            directory_.clear();
            ec.clear();
        }
        if (directory_.empty()) {
            throw std::runtime_error(
                "run_mp2: could not create a private scratch directory");
        }
        path_ = directory_ / "occupied-slabs.bin";
        output_.open(path_, std::ios::binary | std::ios::trunc);
        if (!output_) {
            cleanup();
            throw std::runtime_error("run_mp2: could not open MP2 scratch file");
        }
    }

    SlabScratchFile(const SlabScratchFile&) = delete;
    SlabScratchFile& operator=(const SlabScratchFile&) = delete;
    ~SlabScratchFile() { cleanup(); }

    void append(const RowMat& slab) {
        const auto bytes = sat_mul(static_cast<std::size_t>(slab.size()),
                                   sizeof(double));
        output_.write(reinterpret_cast<const char*>(slab.data()),
                      static_cast<std::streamsize>(bytes));
        if (!output_) throw std::runtime_error("run_mp2: MP2 scratch write failed");
        bytes_ = sat_add(bytes_, bytes);
    }

    void begin_read() {
        output_.close();
        input_.open(path_, std::ios::binary);
        if (!input_) throw std::runtime_error("run_mp2: MP2 scratch reopen failed");
    }

    void read(RowMat& slab) {
        const auto bytes = sat_mul(static_cast<std::size_t>(slab.size()),
                                   sizeof(double));
        input_.read(reinterpret_cast<char*>(slab.data()),
                    static_cast<std::streamsize>(bytes));
        if (!input_) throw std::runtime_error("run_mp2: MP2 scratch read failed");
    }

    std::size_t bytes() const noexcept { return bytes_; }

private:
    void cleanup() noexcept {
        output_.close();
        input_.close();
        if (!directory_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(directory_, ec);
        }
    }

    std::filesystem::path directory_;
    std::filesystem::path path_;
    std::ofstream output_;
    std::ifstream input_;
    std::size_t bytes_ = 0;
};

Tensor4 ao_to_mo_ovov(const Eri4D& eri,
                      const Eigen::MatrixXd& C_occ,
                      const Eigen::MatrixXd& C_vir) {
    const std::size_t n = eri.n;
    const std::size_t no = static_cast<std::size_t>(C_occ.cols());
    const std::size_t nv = static_cast<std::size_t>(C_vir.cols());

    Eigen::Map<const RowMat> M_ao(eri.data.data(), n, n * n * n);
    RowMat I1 = C_occ.transpose() * M_ao;
    RowMat I2(no * nv, n * n);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < no; ++i) {
        Eigen::Map<const RowMat> in(I1.data() + i * n * n * n, n, n * n);
        Eigen::Map<RowMat> out(I2.data() + i * nv * n * n, nv, n * n);
        out = C_vir.transpose() * in;
    }
    RowMat I3(no * nv, no * n);
    #pragma omp parallel for schedule(static)
    for (std::size_t ia = 0; ia < no * nv; ++ia) {
        Eigen::Map<const RowMat> in(I2.data() + ia * n * n, n, n);
        Eigen::Map<RowMat> out(I3.data() + ia * no * n, no, n);
        out = C_occ.transpose() * in;
    }
    Tensor4 mo;
    mo.resize(no, nv, no, nv);
    #pragma omp parallel for schedule(static)
    for (std::size_t iaj = 0; iaj < no * nv * no; ++iaj) {
        Eigen::Map<const Eigen::VectorXd> in(I3.data() + iaj * n, n);
        Eigen::Map<Eigen::VectorXd> out(mo.data.data() + iaj * nv, nv);
        out = C_vir.transpose() * in;
    }
    return mo;
}

struct RMP2Energies {
    double os = 0.0;
    double ss = 0.0;
};

RMP2Energies accumulate_incore(const Tensor4& mo,
                               const Eigen::VectorXd& eps,
                               int no, int nv) {
    double e_os = 0.0;
    double e_ss = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:e_os,e_ss)
    for (int i = 0; i < no; ++i) {
        for (int j = 0; j < no; ++j) {
            for (int a = 0; a < nv; ++a) {
                for (int b = 0; b < nv; ++b) {
                    const double iajb = mo(i, a, j, b);
                    const double ibja = mo(i, b, j, a);
                    const double d = eps(i) + eps(j) - eps(no + a) - eps(no + b);
                    const double tos = iajb * iajb / d;
                    e_os += tos;
                    e_ss += tos - iajb * ibja / d;
                }
            }
        }
    }
    return {e_os, e_ss};
}

void accumulate_slab(const RowMat& slab, std::size_t first_i,
                     int no, int nv, const Eigen::VectorXd& eps,
                     RMP2Energies* energies) {
    const int ni = static_cast<int>(slab.rows()) / nv;
    for (int il = 0; il < ni; ++il) {
        const int i = static_cast<int>(first_i) + il;
        for (int j = 0; j < no; ++j) {
            const auto block = slab.block(il * nv, j * nv, nv, nv);
            for (int a = 0; a < nv; ++a) {
                for (int b = 0; b < nv; ++b) {
                    const double iajb = block(a, b);
                    const double ibja = block(b, a);
                    const double d = eps(i) + eps(j) - eps(no + a) - eps(no + b);
                    const double tos = iajb * iajb / d;
                    energies->os += tos;
                    energies->ss += tos - iajb * ibja / d;
                }
            }
        }
    }
}

template <typename Builder>
RMP2Energies contract_occupied_slabs(
    int no, int nv, const Eigen::VectorXd& eps,
    const SlabPlan& plan, bool use_disk,
    const std::string& scratch_directory,
    Builder&& build, std::size_t* disk_bytes) {
    RMP2Energies energies;
    if (use_disk) {
        SlabScratchFile scratch(scratch_directory);
        for (std::size_t first = 0; first < static_cast<std::size_t>(no);
             first += plan.occupied_count) {
            const std::size_t count = std::min(
                plan.occupied_count, static_cast<std::size_t>(no) - first);
            scratch.append(build(first, count));
        }
        *disk_bytes = scratch.bytes();
        scratch.begin_read();
        for (std::size_t first = 0; first < static_cast<std::size_t>(no);
             first += plan.occupied_count) {
            const std::size_t count = std::min(
                plan.occupied_count, static_cast<std::size_t>(no) - first);
            RowMat slab(static_cast<Eigen::Index>(count * nv), no * nv);
            scratch.read(slab);
            accumulate_slab(slab, first, no, nv, eps, &energies);
        }
    } else {
        for (std::size_t first = 0; first < static_cast<std::size_t>(no);
             first += plan.occupied_count) {
            const std::size_t count = std::min(
                plan.occupied_count, static_cast<std::size_t>(no) - first);
            const RowMat slab = build(first, count);
            accumulate_slab(slab, first, no, nv, eps, &energies);
        }
    }
    return energies;
}

}  // namespace

namespace mp2_detail {

RowMatrix build_direct_ovov_i_batch(
    const BasisSet& basis,
    const Eigen::MatrixXd& C_bra_occ,
    const Eigen::MatrixXd& C_bra_vir,
    const Eigen::MatrixXd& C_ket_occ,
    const Eigen::MatrixXd& C_ket_vir,
    std::size_t first_bra_occupied,
    std::size_t occupied_count) {
    const auto nbf = static_cast<Eigen::Index>(basis.nbasis());
    if (C_bra_occ.rows() != nbf || C_bra_vir.rows() != nbf ||
        C_ket_occ.rows() != nbf || C_ket_vir.rows() != nbf) {
        throw std::invalid_argument(
            "build_direct_ovov_i_batch: MO coefficient row mismatch");
    }
    const auto no_bra = static_cast<std::size_t>(C_bra_occ.cols());
    if (first_bra_occupied > no_bra ||
        occupied_count > no_bra - first_bra_occupied) {
        throw std::out_of_range(
            "build_direct_ovov_i_batch: occupied range is out of bounds");
    }
    const std::size_t n = basis.nbasis();
    const std::size_t nvb = static_cast<std::size_t>(C_bra_vir.cols());
    const std::size_t nok = static_cast<std::size_t>(C_ket_occ.cols());
    const std::size_t nvk = static_cast<std::size_t>(C_ket_vir.cols());
    if (occupied_count == 0) return RowMatrix(0, nok * nvk);

    RowMatrix I1(occupied_count, n * n * n);
    I1.setZero();

    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const int nshell = static_cast<int>(shells.size());
    libint2::Engine engine(libint2::Operator::coulomb,
                           shells.max_nprim(), shells.max_l(), 0);
    const auto& buffer = engine.results();

    // Evaluate each distinct AO integral once, expand its unique 8-fold
    // permutations, and contract only the first (bra-occupied) index into I1.
    // Repeating this quartet pass for each occupied slab preserves O(N^5)
    // work across all slabs while bounding memory at O(slab*N^3).
    for (int s1 = 0; s1 < nshell; ++s1) {
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            for (int s3 = 0; s3 <= s1; ++s3) {
                const auto bf3 = shell2bf[s3];
                const auto n3 = shells[s3].size();
                const int s4max = s3 == s1 ? s2 : s3;
                for (int s4 = 0; s4 <= s4max; ++s4) {
                    const auto bf4 = shell2bf[s4];
                    const auto n4 = shells[s4].size();
                    engine.compute(shells[s1], shells[s2],
                                   shells[s3], shells[s4]);
                    const double* values = buffer[0];
                    if (!values) continue;

                    for (std::size_t f1 = 0; f1 < n1; ++f1) {
                        const std::size_t mu = bf1 + f1;
                        for (std::size_t f2 = 0; f2 < n2; ++f2) {
                            const std::size_t nu = bf2 + f2;
                            if (mu < nu) continue;
                            for (std::size_t f3 = 0; f3 < n3; ++f3) {
                                const std::size_t la = bf3 + f3;
                                for (std::size_t f4 = 0; f4 < n4; ++f4) {
                                    const std::size_t si = bf4 + f4;
                                    // The shell loop has already imposed
                                    // (s1,s2) >= (s3,s4). Function-level pair
                                    // ordering is needed only when those shell
                                    // pairs are identical. Applying it when
                                    // s1 == s3 but s2 > s4 can discard valid
                                    // functions: mu and lambda then share a
                                    // shell, while the exchanged shell quartet
                                    // is absent by construction.
                                    const bool identical_shell_pairs =
                                        s1 == s3 && s2 == s4;
                                    if (la < si ||
                                        (identical_shell_pairs &&
                                         mu * n + nu < la * n + si)) {
                                        continue;
                                    }
                                    const double value = values[
                                        ((f1 * n2 + f2) * n3 + f3) * n4 + f4];
                                    if (value == 0.0) continue;

                                    const std::array<std::array<std::size_t, 4>, 8>
                                        candidates = {{{mu, nu, la, si},
                                                       {nu, mu, la, si},
                                                       {mu, nu, si, la},
                                                       {nu, mu, si, la},
                                                       {la, si, mu, nu},
                                                       {si, la, mu, nu},
                                                       {la, si, nu, mu},
                                                       {si, la, nu, mu}}};
                                    std::array<std::array<std::size_t, 4>, 8> unique{};
                                    std::size_t n_unique = 0;
                                    for (const auto& candidate : candidates) {
                                        bool seen = false;
                                        for (std::size_t q = 0; q < n_unique; ++q) {
                                            if (candidate == unique[q]) {
                                                seen = true;
                                                break;
                                            }
                                        }
                                        if (!seen) unique[n_unique++] = candidate;
                                    }
                                    for (std::size_t q = 0; q < n_unique; ++q) {
                                        const auto& x = unique[q];
                                        const auto col = static_cast<Eigen::Index>(
                                            (x[1] * n + x[2]) * n + x[3]);
                                        for (std::size_t il = 0;
                                             il < occupied_count; ++il) {
                                            I1(static_cast<Eigen::Index>(il), col) +=
                                                value * C_bra_occ(
                                                    static_cast<Eigen::Index>(x[0]),
                                                    static_cast<Eigen::Index>(
                                                        first_bra_occupied + il));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    RowMatrix I2(occupied_count * nvb, n * n);
    #pragma omp parallel for schedule(static)
    for (std::size_t il = 0; il < occupied_count; ++il) {
        Eigen::Map<const RowMatrix> in(I1.data() + il * n * n * n,
                                       n, n * n);
        Eigen::Map<RowMatrix> out(I2.data() + il * nvb * n * n,
                                  nvb, n * n);
        out = C_bra_vir.transpose() * in;
    }
    I1.resize(0, 0);

    RowMatrix I3(occupied_count * nvb, nok * n);
    #pragma omp parallel for schedule(static)
    for (std::size_t ila = 0; ila < occupied_count * nvb; ++ila) {
        Eigen::Map<const RowMatrix> in(I2.data() + ila * n * n, n, n);
        Eigen::Map<RowMatrix> out(I3.data() + ila * nok * n, nok, n);
        out = C_ket_occ.transpose() * in;
    }
    I2.resize(0, 0);

    RowMatrix panel(occupied_count * nvb, nok * nvk);
    #pragma omp parallel for schedule(static)
    for (std::size_t ila = 0; ila < occupied_count * nvb; ++ila) {
        for (std::size_t j = 0; j < nok; ++j) {
            Eigen::Map<const Eigen::VectorXd> in(
                I3.data() + (ila * nok + j) * n, n);
            panel.row(static_cast<Eigen::Index>(ila))
                .segment(static_cast<Eigen::Index>(j * nvk),
                         static_cast<Eigen::Index>(nvk)) =
                in.transpose() * C_ket_vir;
        }
    }
    return panel;
}

}  // namespace mp2_detail

MP2Result run_mp2(const Molecule& mol,
                  const BasisSet& basis,
                  const RHFResult& rhf,
                  const MP2Options& opts) {
    if (!rhf.converged) {
        throw std::runtime_error("run_mp2: RHF reference is not converged");
    }
    const int n_elec =
        effective_electron_count(mol, rhf.ecp_total_ncore, "run_mp2");
    if (n_elec % 2 != 0 || mol.multiplicity() != 1) {
        throw std::invalid_argument(
            "run_mp2: RMP2 requires a closed-shell RHF reference (even "
            "electron count, multiplicity = 1)");
    }

    const int n_occ_total = n_elec / 2;
    const int n_frozen = resolve_native_frozen_core(
        mol, opts.n_frozen_core, rhf.ecp_total_ncore, "run_mp2");
    if (n_frozen >= n_occ_total) {
        throw std::invalid_argument(
            "run_mp2: n_frozen_core >= n_occ, all occupied orbitals frozen");
    }
    const int no = n_occ_total - n_frozen;
    const int n = static_cast<int>(rhf.mo_coeffs.cols());
    const int nv = n - n_occ_total;
    if (nv < 1) {
        throw std::runtime_error(
            "run_mp2: no virtual orbitals -- basis too small for correlation");
    }
    const Eigen::MatrixXd C_occ = rhf.mo_coeffs.middleCols(n_frozen, no);
    const Eigen::MatrixXd C_vir = rhf.mo_coeffs.rightCols(nv);
    Eigen::VectorXd eps(no + nv);
    eps.head(no) = rhf.mo_energies.segment(n_frozen, no);
    eps.tail(nv) = rhf.mo_energies.tail(nv);
    const MemoryMode requested_mode = parse_memory_mode(opts.memory_mode);

    MemoryMode mode = MemoryMode::Incore;
    RMP2Energies energies;
    std::size_t workspace = 0;
    std::size_t disk_bytes = 0;

    if (opts.density_fit) {
        if (opts.aux_basis.empty()) {
            throw std::invalid_argument(
                "run_mp2: density_fit=true requires aux_basis to be set "
                "(e.g. \"def2-tzvp-rifit\"). Use "
                "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"ri\") "
                "for autodetection.");
        }
        const BasisSet aux(mol, opts.aux_basis);
        const std::size_t nov = static_cast<std::size_t>(no) * nv;
        const DFWorkspace model = df_workspace(
            static_cast<std::size_t>(n), aux.nbasis(),
            static_cast<std::size_t>(no), static_cast<std::size_t>(nv),
            opts.use_float_intermediates);
        mode = resolve_memory_mode(requested_mode, opts.requested_memory_bytes,
                                   model.incore);

        SlabPlan plan;
        if (mode != MemoryMode::Incore) {
            plan = plan_occupied_slabs(
                static_cast<std::size_t>(no), bytes_for({static_cast<std::size_t>(nv),
                                                         static_cast<std::size_t>(no),
                                                         static_cast<std::size_t>(nv)}),
                model.resident, model.construction,
                opts.requested_memory_bytes, "run_mp2");
        }

        DensityFitting df(basis, aux);
        const RowMat B_mo = df.mo_transform(C_occ, C_vir);
        if (mode == MemoryMode::Incore) {
            Tensor4 mo;
            mo.resize(no, nv, no, nv);
            Eigen::Map<RowMat> view(mo.data.data(), nov, nov);
            if (opts.use_float_intermediates) {
                const Eigen::MatrixXf Bf = B_mo.cast<float>();
                const Eigen::MatrixXf mf = Bf.transpose() * Bf;
                view = mf.cast<double>();
            } else {
                view = B_mo.transpose() * B_mo;
            }
            energies = accumulate_incore(mo, eps, no, nv);
            workspace = model.incore;
        } else {
            auto build = [&](std::size_t first, std::size_t count) {
                RowMat panel = B_mo.middleCols(
                    static_cast<Eigen::Index>(first * nv),
                    static_cast<Eigen::Index>(count * nv)).transpose() * B_mo;
                return panel;
            };
            energies = contract_occupied_slabs(
                no, nv, eps, plan, mode == MemoryMode::Disk,
                opts.scratch_directory, build, &disk_bytes);
            workspace = plan.workspace_bytes;
        }
    } else {
        const std::size_t incore = exact_incore_workspace(n, no, nv);
        mode = resolve_memory_mode(requested_mode, opts.requested_memory_bytes,
                                   incore);
        if (mode == MemoryMode::Incore) {
            const Eri4D eri = compute_eri(basis);
            const Tensor4 mo = ao_to_mo_ovov(eri, C_occ, C_vir);
            energies = accumulate_incore(mo, eps, no, nv);
            workspace = incore;
        } else {
            const std::size_t fixed_mo = bytes_for({
                static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
            const SlabPlan plan = plan_occupied_slabs(
                static_cast<std::size_t>(no),
                exact_direct_bytes_per_i(n, no, nv), fixed_mo, 0,
                opts.requested_memory_bytes, "run_mp2");
            auto build = [&](std::size_t first, std::size_t count) {
                return mp2_detail::build_direct_ovov_i_batch(
                    basis, C_occ, C_vir, C_occ, C_vir, first, count);
            };
            energies = contract_occupied_slabs(
                no, nv, eps, plan, mode == MemoryMode::Disk,
                opts.scratch_directory, build, &disk_bytes);
            workspace = plan.workspace_bytes;
        }
    }

    MP2Result result;
    result.e_hf = rhf.energy;
    result.e_os = energies.os;
    result.e_ss = energies.ss;
    result.e_correlation = opts.c_os * energies.os + opts.c_ss * energies.ss;
    result.e_total = result.e_hf + result.e_correlation;
    result.n_frozen_core = n_frozen;
    result.memory_mode_used = mode_name(mode);
    result.workspace_bytes = workspace;
    result.disk_bytes = disk_bytes;

    if (opts.density_fit && opts.report_ri_residual) {
        const std::size_t fixed_mo = bytes_for({
            static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
        const SlabPlan oracle_plan = plan_occupied_slabs(
            static_cast<std::size_t>(no),
            exact_direct_bytes_per_i(n, no, nv), fixed_mo, 0,
            opts.requested_memory_bytes, "run_mp2 RI residual");
        std::size_t unused_disk = 0;
        auto build = [&](std::size_t first, std::size_t count) {
            return mp2_detail::build_direct_ovov_i_batch(
                basis, C_occ, C_vir, C_occ, C_vir, first, count);
        };
        const RMP2Energies exact = contract_occupied_slabs(
            no, nv, eps, oracle_plan, false, std::string{},
            build, &unused_disk);
        result.e_os_ri_residual = result.e_os - exact.os;
        result.e_ss_ri_residual = result.e_ss - exact.ss;
        result.ri_residual_reported = true;
        result.workspace_bytes = std::max(result.workspace_bytes,
                                          oracle_plan.workspace_bytes);
    }
    return result;
}

}  // namespace vibeqc
