#include "vibeqc/ump2.hpp"

#include <algorithm>
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
#include "vibeqc/integrals.hpp"
#include "vibeqc/mp2.hpp"
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

std::size_t channel_transform_workspace(std::size_t n,
                                        std::size_t no_b, std::size_t nv_b,
                                        std::size_t no_k, std::size_t nv_k) {
    std::size_t total = bytes_for({no_b, n, n, n});
    total = sat_add(total, bytes_for({no_b, nv_b, n, n}));
    total = sat_add(total, bytes_for({no_b, nv_b, no_k, n}));
    total = sat_add(total, bytes_for({no_b, nv_b, no_k, nv_k}));
    return total;
}

std::size_t exact_incore_workspace(std::size_t n,
                                   std::size_t na, std::size_t nva,
                                   std::size_t nb, std::size_t nvb) {
    std::size_t largest = 0;
    if (na >= 2 && nva >= 2) {
        largest = std::max(
            largest, channel_transform_workspace(n, na, nva, na, nva));
    }
    if (nb >= 2 && nvb >= 2) {
        largest = std::max(
            largest, channel_transform_workspace(n, nb, nvb, nb, nvb));
    }
    if (na >= 1 && nb >= 1 && nva >= 1 && nvb >= 1) {
        largest = std::max(
            largest, channel_transform_workspace(n, na, nva, nb, nvb));
    }
    // Alpha and beta occupied/virtual slices are owned copies totaling 2*n*n.
    return sat_add(sat_mul(2, bytes_for({n, n})),
                   sat_add(bytes_for({n, n, n, n}), largest));
}

// The exact integral-direct route keeps one bra-occupied slab through the
// four AO-to-MO transformation stages. Each predecessor is released before
// the next stage is consumed, so the modeled peak is the largest adjacent
// phase overlap.
std::size_t exact_direct_bytes_per_i(std::size_t n, std::size_t nv_b,
                                     std::size_t no_k, std::size_t nv_k) {
    const auto i1 = bytes_for({n, n, n});
    const auto i2 = bytes_for({nv_b, n, n});
    const auto i3 = bytes_for({nv_b, no_k, n});
    const auto panel = bytes_for({nv_b, no_k, nv_k});
    return std::max({sat_add(i1, i2), sat_add(i2, i3),
                     sat_add(i3, panel)});
}

struct DFWorkspace {
    std::size_t construction = 0;
    std::size_t resident = 0;
    std::size_t incore = 0;
};

DFWorkspace df_workspace(std::size_t n, std::size_t naux,
                         std::size_t na, std::size_t nva,
                         std::size_t nb, std::size_t nvb) {
    const std::size_t fixed_mo = sat_mul(2, bytes_for({n, n}));
    const std::size_t three = bytes_for({naux, n, n});
    const std::size_t metric = bytes_for({naux, naux});
    const std::size_t ba = bytes_for({naux, na, nva});
    const std::size_t bb = bytes_for({naux, nb, nvb});
    const std::size_t aa = bytes_for({na, nva, na, nva});
    const std::size_t bbbb = bytes_for({nb, nvb, nb, nvb});
    const std::size_t ab = bytes_for({na, nva, nb, nvb});

    DFWorkspace w;
    // DensityFitting construction overlaps raw T, T_flat, solved B_flat,
    // and unpacked B_per_P while V, LLT storage, and L are all live, then
    // retains T_flat + B_per_P + L. Both spin B-mo matrices remain resident
    // while the three channels are contracted.
    const std::size_t constructor = sat_add(
        fixed_mo, sat_add(sat_mul(4, three), sat_mul(3, metric)));
    const std::size_t df_resident = sat_add(
        fixed_mo, sat_add(sat_mul(2, three), metric));
    w.resident = sat_add(df_resident, sat_add(ba, bb));
    const std::size_t transform_workers = static_cast<std::size_t>(
        omp_workers_for(naux, omp_max_threads()));
    const std::size_t alpha_scratch = sat_mul(
        transform_workers,
        sat_add(bytes_for({na, n}), bytes_for({na, nva})));
    const std::size_t beta_scratch = sat_mul(
        transform_workers,
        sat_add(bytes_for({nb, n}), bytes_for({nb, nvb})));
    const std::size_t alpha_transform = sat_add(
        sat_add(df_resident, ba), alpha_scratch);
    const std::size_t beta_transform = sat_add(w.resident, beta_scratch);
    w.construction = std::max({constructor, alpha_transform, beta_transform});
    w.incore = std::max(w.construction,
                        sat_add(w.resident, std::max({aa, bbbb, ab})));
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
        "run_ump2: memory_mode must be one of auto, incore, direct, or disk");
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
            "run_ump2: requested in-core UMP2 workspace exceeds "
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

SlabPlan plan_occupied_slabs(std::size_t no_bra,
                             std::size_t bytes_per_i,
                             std::size_t resident_bytes,
                             std::size_t construction_bytes,
                             std::size_t budget_bytes,
                             const std::string& channel_name) {
    const std::size_t minimum = std::max(
        construction_bytes, sat_add(resident_bytes, bytes_per_i));
    if (budget_bytes != 0 && minimum > budget_bytes) {
        throw std::runtime_error(
            "run_ump2: requested_memory_bytes is below the minimum "
            "one-occupied-orbital slab workspace for the " + channel_name +
            " channel");
    }
    constexpr std::size_t kDefaultSlabTarget = 64ULL << 20;
    const std::size_t room = budget_bytes == 0
        ? kDefaultSlabTarget
        : budget_bytes - resident_bytes;
    std::size_t count = bytes_per_i == 0
        ? no_bra
        : room / bytes_per_i;
    count = std::max<std::size_t>(1, std::min(no_bra, count));
    const std::size_t live = sat_add(
        resident_bytes, sat_mul(count, bytes_per_i));
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
                "run_ump2: scratch_directory does not exist or is not a directory");
        }

        static std::atomic<unsigned long long> sequence{0};
        const auto stamp = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        for (unsigned int attempt = 0; attempt < 100; ++attempt) {
            const auto id = sequence.fetch_add(1, std::memory_order_relaxed);
            directory_ = base / ("vibeqc-ump2-" + std::to_string(stamp) + "-" +
                                 std::to_string(id));
            if (fs::create_directory(directory_, ec)) break;
            directory_.clear();
            ec.clear();
        }
        if (directory_.empty()) {
            throw std::runtime_error(
                "run_ump2: could not create a private scratch directory");
        }
        path_ = directory_ / "occupied-slabs.bin";
        output_.open(path_, std::ios::binary | std::ios::trunc);
        if (!output_) {
            cleanup();
            throw std::runtime_error("run_ump2: could not open UMP2 scratch file");
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
        if (!output_) throw std::runtime_error("run_ump2: UMP2 scratch write failed");
        bytes_ = sat_add(bytes_, bytes);
    }

    void begin_read() {
        output_.close();
        input_.open(path_, std::ios::binary);
        if (!input_) throw std::runtime_error("run_ump2: UMP2 scratch reopen failed");
    }

    void read(RowMat& slab) {
        const auto bytes = sat_mul(static_cast<std::size_t>(slab.size()),
                                   sizeof(double));
        input_.read(reinterpret_cast<char*>(slab.data()),
                    static_cast<std::streamsize>(bytes));
        if (!input_) throw std::runtime_error("run_ump2: UMP2 scratch read failed");
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

Tensor4 ao_to_mo_ovov_uhf(const Eri4D& eri,
                          const Eigen::MatrixXd& Cb_occ,
                          const Eigen::MatrixXd& Cb_vir,
                          const Eigen::MatrixXd& Ck_occ,
                          const Eigen::MatrixXd& Ck_vir) {
    const std::size_t n = eri.n;
    const std::size_t no1 = static_cast<std::size_t>(Cb_occ.cols());
    const std::size_t nv1 = static_cast<std::size_t>(Cb_vir.cols());
    const std::size_t no2 = static_cast<std::size_t>(Ck_occ.cols());
    const std::size_t nv2 = static_cast<std::size_t>(Ck_vir.cols());

    Eigen::Map<const RowMat> M_ao(eri.data.data(), n, n * n * n);
    RowMat I1 = Cb_occ.transpose() * M_ao;
    RowMat I2(no1 * nv1, n * n);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < no1; ++i) {
        Eigen::Map<const RowMat> in(I1.data() + i * n * n * n, n, n * n);
        Eigen::Map<RowMat> out(I2.data() + i * nv1 * n * n, nv1, n * n);
        out = Cb_vir.transpose() * in;
    }
    RowMat I3(no1 * nv1, no2 * n);
    #pragma omp parallel for schedule(static)
    for (std::size_t ia = 0; ia < no1 * nv1; ++ia) {
        Eigen::Map<const RowMat> in(I2.data() + ia * n * n, n, n);
        Eigen::Map<RowMat> out(I3.data() + ia * no2 * n, no2, n);
        out = Ck_occ.transpose() * in;
    }
    Tensor4 mo;
    mo.resize(no1, nv1, no2, nv2);
    #pragma omp parallel for schedule(static)
    for (std::size_t iaj = 0; iaj < no1 * nv1 * no2; ++iaj) {
        Eigen::Map<const Eigen::VectorXd> in(I3.data() + iaj * n, n);
        Eigen::Map<Eigen::VectorXd> out(mo.data.data() + iaj * nv2, nv2);
        out = Ck_vir.transpose() * in;
    }
    return mo;
}

double accumulate_same_spin_incore(const Tensor4& mo, int no, int nv,
                                   const Eigen::VectorXd& eps) {
    double e = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:e)
    for (int i = 0; i < no; ++i) {
        for (int j = 0; j < no; ++j) {
            for (int a = 0; a < nv; ++a) {
                for (int b = 0; b < nv; ++b) {
                    const double anti = mo(i, a, j, b) - mo(i, b, j, a);
                    e += 0.25 * anti * anti /
                        (eps(i) + eps(j) - eps(no + a) - eps(no + b));
                }
            }
        }
    }
    return e;
}

double accumulate_opposite_spin_incore(const Tensor4& mo,
                                       int na, int nva, int nb, int nvb,
                                       const Eigen::VectorXd& eps_a,
                                       const Eigen::VectorXd& eps_b) {
    double e = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:e)
    for (int i = 0; i < na; ++i) {
        for (int a = 0; a < nva; ++a) {
            for (int j = 0; j < nb; ++j) {
                for (int b = 0; b < nvb; ++b) {
                    const double v = mo(i, a, j, b);
                    e += v * v / (eps_a(i) + eps_b(j)
                                  - eps_a(na + a) - eps_b(nb + b));
                }
            }
        }
    }
    return e;
}

void accumulate_same_spin_slab(const RowMat& slab, std::size_t first_i,
                               int no, int nv, const Eigen::VectorXd& eps,
                               double* energy) {
    const int ni = static_cast<int>(slab.rows()) / nv;
    for (int il = 0; il < ni; ++il) {
        const int i = static_cast<int>(first_i) + il;
        for (int j = 0; j < no; ++j) {
            const auto block = slab.block(il * nv, j * nv, nv, nv);
            for (int a = 0; a < nv; ++a) {
                for (int b = 0; b < nv; ++b) {
                    const double anti = block(a, b) - block(b, a);
                    const double d = eps(i) + eps(j)
                                   - eps(no + a) - eps(no + b);
                    *energy += 0.25 * anti * anti / d;
                }
            }
        }
    }
}

void accumulate_opposite_spin_slab(
    const RowMat& slab, std::size_t first_i,
    int na, int nva, int nb, int nvb,
    const Eigen::VectorXd& eps_a, const Eigen::VectorXd& eps_b,
    double* energy) {
    const int ni = static_cast<int>(slab.rows()) / nva;
    for (int il = 0; il < ni; ++il) {
        const int i = static_cast<int>(first_i) + il;
        for (int j = 0; j < nb; ++j) {
            const auto block = slab.block(il * nva, j * nvb, nva, nvb);
            for (int a = 0; a < nva; ++a) {
                for (int b = 0; b < nvb; ++b) {
                    const double v = block(a, b);
                    const double d = eps_a(i) + eps_b(j)
                                   - eps_a(na + a) - eps_b(nb + b);
                    *energy += v * v / d;
                }
            }
        }
    }
}

template <typename Builder, typename Consumer>
std::size_t run_slab_channel(
    std::size_t no_bra, int nv_bra, int no_ket, int nv_ket,
    const SlabPlan& plan, bool use_disk,
    const std::string& scratch_directory,
    Builder&& build, Consumer&& consume) {
    if (no_bra == 0) return 0;
    if (use_disk) {
        SlabScratchFile scratch(scratch_directory);
        for (std::size_t first = 0; first < no_bra;
             first += plan.occupied_count) {
            const std::size_t count = std::min(
                plan.occupied_count, no_bra - first);
            scratch.append(build(first, count));
        }
        const std::size_t disk_bytes = scratch.bytes();
        scratch.begin_read();
        for (std::size_t first = 0; first < no_bra;
             first += plan.occupied_count) {
            const std::size_t count = std::min(
                plan.occupied_count, no_bra - first);
            RowMat slab(static_cast<Eigen::Index>(count * nv_bra),
                        static_cast<Eigen::Index>(no_ket * nv_ket));
            scratch.read(slab);
            consume(first, slab);
        }
        return disk_bytes;
    }

    for (std::size_t first = 0; first < no_bra;
         first += plan.occupied_count) {
        const std::size_t count = std::min(
            plan.occupied_count, no_bra - first);
        const RowMat slab = build(first, count);
        consume(first, slab);
    }
    return 0;
}

struct ChannelPlans {
    SlabPlan aa;
    SlabPlan bb;
    SlabPlan ab;
    std::size_t workspace = 0;
};

ChannelPlans plan_channels(std::size_t n,
                           int na, int nva, int nb, int nvb,
                           bool use_df,
                           bool have_aa, bool have_bb, bool have_ab,
                           std::size_t resident, std::size_t construction,
                           std::size_t budget) {
    auto bytes_per_i = [&](std::size_t nv_bra,
                           std::size_t no_ket,
                           std::size_t nv_ket) {
        return use_df
            ? bytes_for({nv_bra, no_ket, nv_ket})
            : exact_direct_bytes_per_i(n, nv_bra, no_ket, nv_ket);
    };

    ChannelPlans plans;
    plans.workspace = std::max(resident, construction);
    if (have_aa) {
        plans.aa = plan_occupied_slabs(
            static_cast<std::size_t>(na),
            bytes_per_i(nva, na, nva), resident, construction, budget,
            "alpha-alpha");
        plans.workspace = std::max(plans.workspace, plans.aa.workspace_bytes);
    }
    if (have_bb) {
        plans.bb = plan_occupied_slabs(
            static_cast<std::size_t>(nb),
            bytes_per_i(nvb, nb, nvb), resident, construction, budget,
            "beta-beta");
        plans.workspace = std::max(plans.workspace, plans.bb.workspace_bytes);
    }
    if (have_ab) {
        plans.ab = plan_occupied_slabs(
            static_cast<std::size_t>(na),
            bytes_per_i(nva, nb, nvb), resident, construction, budget,
            "alpha-beta");
        plans.workspace = std::max(plans.workspace, plans.ab.workspace_bytes);
    }
    return plans;
}

struct UMP2Channels {
    double aa = 0.0;
    double bb = 0.0;
    double ab = 0.0;
    std::size_t workspace = 0;
    std::size_t disk_bytes = 0;
};

UMP2Channels contract_slab_channels(
    const BasisSet& basis,
    const Eigen::MatrixXd& Ca_occ, const Eigen::MatrixXd& Ca_vir,
    const Eigen::MatrixXd& Cb_occ, const Eigen::MatrixXd& Cb_vir,
    const Eigen::VectorXd& eps_a, const Eigen::VectorXd& eps_b,
    const RowMat* B_alpha, const RowMat* B_beta,
    MemoryMode mode, const ChannelPlans& plans,
    const std::string& scratch_directory,
    bool have_aa, bool have_bb, bool have_ab) {
    const int na = static_cast<int>(Ca_occ.cols());
    const int nb = static_cast<int>(Cb_occ.cols());
    const int nva = static_cast<int>(Ca_vir.cols());
    const int nvb = static_cast<int>(Cb_vir.cols());
    const bool use_df = B_alpha != nullptr;
    const bool use_disk = mode == MemoryMode::Disk;
    UMP2Channels out;
    out.workspace = plans.workspace;

    auto run_same = [&](const Eigen::MatrixXd& Co,
                        const Eigen::MatrixXd& Cv,
                        const Eigen::VectorXd& eps,
                        const RowMat* B, int no, int nv,
                        const SlabPlan& plan, double* energy) {
        auto build = [&](std::size_t first, std::size_t count) {
            if (!use_df) {
                return mp2_detail::build_direct_ovov_i_batch(
                    basis, Co, Cv, Co, Cv, first, count);
            }
            return RowMat(
                B->middleCols(static_cast<Eigen::Index>(first * nv),
                              static_cast<Eigen::Index>(count * nv))
                    .transpose() * *B);
        };
        auto consume = [&](std::size_t first, const RowMat& slab) {
            accumulate_same_spin_slab(slab, first, no, nv, eps, energy);
        };
        out.disk_bytes = std::max(
            out.disk_bytes,
            run_slab_channel(static_cast<std::size_t>(no), nv, no, nv,
                             plan, use_disk, scratch_directory,
                             build, consume));
    };

    if (have_aa) {
        run_same(Ca_occ, Ca_vir, eps_a, B_alpha, na, nva,
                 plans.aa, &out.aa);
    }
    if (have_bb) {
        run_same(Cb_occ, Cb_vir, eps_b, B_beta, nb, nvb,
                 plans.bb, &out.bb);
    }
    if (have_ab) {
        auto build = [&](std::size_t first, std::size_t count) {
            if (!use_df) {
                return mp2_detail::build_direct_ovov_i_batch(
                    basis, Ca_occ, Ca_vir, Cb_occ, Cb_vir, first, count);
            }
            return RowMat(
                B_alpha->middleCols(
                    static_cast<Eigen::Index>(first * nva),
                    static_cast<Eigen::Index>(count * nva))
                    .transpose() * *B_beta);
        };
        auto consume = [&](std::size_t first, const RowMat& slab) {
            accumulate_opposite_spin_slab(
                slab, first, na, nva, nb, nvb, eps_a, eps_b, &out.ab);
        };
        out.disk_bytes = std::max(
            out.disk_bytes,
            run_slab_channel(static_cast<std::size_t>(na), nva, nb, nvb,
                             plans.ab, use_disk, scratch_directory,
                             build, consume));
    }
    return out;
}

}  // namespace

UMP2Result run_ump2(const Molecule& mol,
                    const BasisSet& basis,
                    const UHFResult& uhf,
                    const UMP2Options& opts) {
    if (!uhf.converged) {
        throw std::runtime_error("run_ump2: UHF reference is not converged");
    }
    const int n_elec =
        effective_electron_count(mol, uhf.ecp_total_ncore, "run_ump2");
    const int mult = mol.multiplicity();
    const int n_alpha_total = (n_elec + mult - 1) / 2;
    const int n_beta_total = (n_elec - mult + 1) / 2;
    const int n_frozen = resolve_native_frozen_core(
        mol, opts.n_frozen_core, uhf.ecp_total_ncore, "run_ump2");
    if (n_frozen >= n_alpha_total) {
        throw std::invalid_argument(
            "run_ump2: n_frozen_core freezes all alpha-occupied orbitals");
    }
    if (n_frozen > n_beta_total) {
        throw std::invalid_argument(
            "run_ump2: n_frozen_core exceeds the beta-occupied orbitals");
    }
    const int n_alpha = n_alpha_total - n_frozen;
    const int n_beta = n_beta_total - n_frozen;
    const int n_orb = static_cast<int>(uhf.mo_coeffs_alpha.cols());
    const int nv_alpha = n_orb - n_alpha_total;
    const int nv_beta = n_orb - n_beta_total;
    if (nv_alpha < 1 && nv_beta < 1) {
        throw std::runtime_error(
            "run_ump2: no virtual orbitals -- basis too small for correlation");
    }

    const Eigen::MatrixXd Ca_occ =
        uhf.mo_coeffs_alpha.middleCols(n_frozen, n_alpha);
    const Eigen::MatrixXd Ca_vir = uhf.mo_coeffs_alpha.rightCols(nv_alpha);
    const Eigen::MatrixXd Cb_occ =
        uhf.mo_coeffs_beta.middleCols(n_frozen, n_beta);
    const Eigen::MatrixXd Cb_vir = uhf.mo_coeffs_beta.rightCols(nv_beta);
    Eigen::VectorXd eps_alpha(n_alpha + nv_alpha);
    eps_alpha.head(n_alpha) =
        uhf.mo_energies_alpha.segment(n_frozen, n_alpha);
    eps_alpha.tail(nv_alpha) = uhf.mo_energies_alpha.tail(nv_alpha);
    Eigen::VectorXd eps_beta(n_beta + nv_beta);
    eps_beta.head(n_beta) =
        uhf.mo_energies_beta.segment(n_frozen, n_beta);
    eps_beta.tail(nv_beta) = uhf.mo_energies_beta.tail(nv_beta);
    const bool have_aa = n_alpha >= 2 && nv_alpha >= 2;
    const bool have_bb = n_beta >= 2 && nv_beta >= 2;
    const bool have_ab = n_alpha >= 1 && n_beta >= 1 &&
                         nv_alpha >= 1 && nv_beta >= 1;

    const MemoryMode requested_mode = parse_memory_mode(opts.memory_mode);
    MemoryMode mode = MemoryMode::Incore;
    UMP2Channels channels;

    if (opts.density_fit) {
        if (opts.aux_basis.empty()) {
            throw std::invalid_argument(
                "run_ump2: density_fit=true requires aux_basis to be set "
                "(e.g. \"def2-tzvp-rifit\"). Use "
                "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"ri\") "
                "for autodetection.");
        }
        const BasisSet aux(mol, opts.aux_basis);
        const DFWorkspace model = df_workspace(
            static_cast<std::size_t>(n_orb), aux.nbasis(),
            static_cast<std::size_t>(n_alpha), static_cast<std::size_t>(nv_alpha),
            static_cast<std::size_t>(n_beta), static_cast<std::size_t>(nv_beta));
        mode = resolve_memory_mode(requested_mode, opts.requested_memory_bytes,
                                   model.incore);
        ChannelPlans plans;
        if (mode != MemoryMode::Incore) {
            plans = plan_channels(
                static_cast<std::size_t>(n_orb),
                n_alpha, nv_alpha, n_beta, nv_beta, true,
                have_aa, have_bb, have_ab,
                model.resident, model.construction,
                opts.requested_memory_bytes);
        }

        DensityFitting df(basis, aux);
        RowMat B_alpha;
        RowMat B_beta;
        if (n_alpha >= 1 && nv_alpha >= 1) {
            B_alpha = df.mo_transform(Ca_occ, Ca_vir);
        }
        if (n_beta >= 1 && nv_beta >= 1) {
            B_beta = df.mo_transform(Cb_occ, Cb_vir);
        }

        if (mode == MemoryMode::Incore) {
            if (have_aa) {
                Tensor4 mo;
                mo.resize(n_alpha, nv_alpha, n_alpha, nv_alpha);
                Eigen::Map<RowMat> view(mo.data.data(), n_alpha * nv_alpha,
                                        n_alpha * nv_alpha);
                view = B_alpha.transpose() * B_alpha;
                channels.aa = accumulate_same_spin_incore(
                    mo, n_alpha, nv_alpha, eps_alpha);
            }
            if (have_bb) {
                Tensor4 mo;
                mo.resize(n_beta, nv_beta, n_beta, nv_beta);
                Eigen::Map<RowMat> view(mo.data.data(), n_beta * nv_beta,
                                        n_beta * nv_beta);
                view = B_beta.transpose() * B_beta;
                channels.bb = accumulate_same_spin_incore(
                    mo, n_beta, nv_beta, eps_beta);
            }
            if (have_ab) {
                Tensor4 mo;
                mo.resize(n_alpha, nv_alpha, n_beta, nv_beta);
                Eigen::Map<RowMat> view(mo.data.data(), n_alpha * nv_alpha,
                                        n_beta * nv_beta);
                view = B_alpha.transpose() * B_beta;
                channels.ab = accumulate_opposite_spin_incore(
                    mo, n_alpha, nv_alpha, n_beta, nv_beta,
                    eps_alpha, eps_beta);
            }
            channels.workspace = model.incore;
        } else {
            channels = contract_slab_channels(
                basis, Ca_occ, Ca_vir, Cb_occ, Cb_vir,
                eps_alpha, eps_beta,
                &B_alpha, &B_beta, mode, plans, opts.scratch_directory,
                have_aa, have_bb, have_ab);
        }
    } else {
        const std::size_t incore = exact_incore_workspace(
            static_cast<std::size_t>(n_orb),
            static_cast<std::size_t>(n_alpha), static_cast<std::size_t>(nv_alpha),
            static_cast<std::size_t>(n_beta), static_cast<std::size_t>(nv_beta));
        mode = resolve_memory_mode(requested_mode, opts.requested_memory_bytes,
                                   incore);
        if (mode == MemoryMode::Incore) {
            const Eri4D eri = compute_eri(basis);
            if (have_aa) {
                const Tensor4 mo = ao_to_mo_ovov_uhf(
                    eri, Ca_occ, Ca_vir, Ca_occ, Ca_vir);
                channels.aa = accumulate_same_spin_incore(
                    mo, n_alpha, nv_alpha, eps_alpha);
            }
            if (have_bb) {
                const Tensor4 mo = ao_to_mo_ovov_uhf(
                    eri, Cb_occ, Cb_vir, Cb_occ, Cb_vir);
                channels.bb = accumulate_same_spin_incore(
                    mo, n_beta, nv_beta, eps_beta);
            }
            if (have_ab) {
                const Tensor4 mo = ao_to_mo_ovov_uhf(
                    eri, Ca_occ, Ca_vir, Cb_occ, Cb_vir);
                channels.ab = accumulate_opposite_spin_incore(
                    mo, n_alpha, nv_alpha, n_beta, nv_beta,
                    eps_alpha, eps_beta);
            }
            channels.workspace = incore;
        } else {
            const std::size_t fixed_mo = sat_mul(
                2, bytes_for({static_cast<std::size_t>(n_orb),
                              static_cast<std::size_t>(n_orb)}));
            const ChannelPlans plans = plan_channels(
                static_cast<std::size_t>(n_orb),
                n_alpha, nv_alpha, n_beta, nv_beta, false,
                have_aa, have_bb, have_ab, fixed_mo, 0,
                opts.requested_memory_bytes);
            channels = contract_slab_channels(
                basis, Ca_occ, Ca_vir, Cb_occ, Cb_vir,
                eps_alpha, eps_beta,
                nullptr, nullptr, mode, plans, opts.scratch_directory,
                have_aa, have_bb, have_ab);
        }
    }

    UMP2Result result;
    result.e_hf = uhf.energy;
    result.e_aa = channels.aa;
    result.e_bb = channels.bb;
    result.e_ab = channels.ab;
    result.e_correlation = opts.c_os * channels.ab +
                           opts.c_ss * (channels.aa + channels.bb);
    result.e_total = result.e_hf + result.e_correlation;
    result.n_frozen_core = n_frozen;
    result.memory_mode_used = mode_name(mode);
    result.workspace_bytes = channels.workspace;
    result.disk_bytes = channels.disk_bytes;

    if (opts.density_fit && opts.report_ri_residual) {
        const std::size_t fixed_mo = sat_mul(
            2, bytes_for({static_cast<std::size_t>(n_orb),
                          static_cast<std::size_t>(n_orb)}));
        const ChannelPlans oracle_plans = plan_channels(
            static_cast<std::size_t>(n_orb),
            n_alpha, nv_alpha, n_beta, nv_beta, false,
            have_aa, have_bb, have_ab, fixed_mo, 0,
            opts.requested_memory_bytes);
        const UMP2Channels exact = contract_slab_channels(
            basis, Ca_occ, Ca_vir, Cb_occ, Cb_vir,
            eps_alpha, eps_beta,
            nullptr, nullptr, MemoryMode::Direct, oracle_plans, std::string{},
            have_aa, have_bb, have_ab);
        result.e_aa_ri_residual = result.e_aa - exact.aa;
        result.e_bb_ri_residual = result.e_bb - exact.bb;
        result.e_ab_ri_residual = result.e_ab - exact.ab;
        result.ri_residual_reported = true;
        result.workspace_bytes = std::max(result.workspace_bytes,
                                          exact.workspace);
    }
    return result;
}

}  // namespace vibeqc
