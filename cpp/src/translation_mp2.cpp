#include "vibeqc/translation_mp2.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace vibeqc {
namespace {

using Matrix = Eigen::MatrixXd;

void require(bool accepted, const char* message) {
    if (!accepted) throw std::invalid_argument(std::string("translation MP2: ") + message);
}

void finite(const Matrix& value, const char* message) {
    if (!value.allFinite()) throw std::runtime_error(std::string("translation MP2: ") + message);
}

bool close(const Matrix& a, const Matrix& b, double tolerance) {
    return a.allFinite() && b.allFinite() && (a - b).cwiseAbs().maxCoeff() <= tolerance;
}

struct Route {
    std::size_t row;
    int shift;
    bool transpose;
};

// This finite-group routing is independent of the number of placed pairs.
class Routes {
public:
    Routes(const std::vector<int>& mesh, int bands,
           const std::vector<int>& first, const std::vector<int>& second)
        : mesh_(mesh), bands_(bands) {
        for (int x = 0; x < mesh[0]; ++x)
            for (int y = 0; y < mesh[1]; ++y)
                for (int z = 0; z < mesh[2]; ++z) cells_.push_back({x, y, z});
        for (std::size_t row = 0; row < first.size(); ++row) {
            const auto key = std::make_pair(first[row], second[row]);
            require(rows_.emplace(key, row).second, "duplicate representative pair");
        }
        std::size_t expected = 0;
        for (int a = 0; a < bands; ++a) for (int b = a; b < bands; ++b) {
            for (int d = 0; d < static_cast<int>(cells_.size()); ++d) {
                if (a == b && d > difference(0, d)) continue;
                require(rows_.count({a, d * bands + b}) == 1,
                        "missing canonical representative pair");
                ++expected;
            }
        }
        require(expected == rows_.size(), "noncanonical representative pair");
    }

    int difference(int a, int b) const {
        const auto& ca = cells_[a];
        const auto& cb = cells_[b];
        std::array<int, 3> d;
        for (int axis = 0; axis < 3; ++axis) {
            d[axis] = ca[axis] - cb[axis];
            if (d[axis] < 0) d[axis] += mesh_[axis];
        }
        return (d[0] * mesh_[1] + d[1]) * mesh_[2] + d[2];
    }

    Route resolve(int i, int j) const {
        int a = i % bands_, b = j % bands_;
        int d = difference(j / bands_, i / bands_);
        const int inverse = difference(i / bands_, j / bands_);
        const bool transpose = a > b || (a == b && d > inverse);
        const int shift = (transpose ? j : i) / bands_;
        if (transpose) { std::swap(a, b); d = inverse; }
        return {rows_.at({a, d * bands_ + b}), shift, transpose};
    }

    Matrix translate(const Matrix& basis, int shift) const {
        const auto local = basis.rows() / static_cast<Eigen::Index>(cells_.size());
        Matrix translated(basis.rows(), basis.cols());
        for (int cell = 0; cell < static_cast<int>(cells_.size()); ++cell)
            translated.middleRows(cell * local, local) =
                basis.middleRows(difference(cell, shift) * local, local);
        return translated;
    }

private:
    std::vector<int> mesh_;
    int bands_;
    std::vector<std::array<int, 3>> cells_;
    std::map<std::pair<int, int>, std::size_t> rows_;
};

/// Read-only metric access for dense and cyclic home-row representations.
/// A block metric obeys S[(R,a),(T,b)] = blocks[T-R](a,b). It never creates
/// the full AO matrix, including while validating its positive definiteness.
class MetricView {
public:
    explicit MetricView(const Matrix& dense) : dense_(&dense) {}
    explicit MetricView(const std::vector<Matrix>& blocks) : blocks_(&blocks) {}

    bool cyclic() const { return blocks_ != nullptr; }
    Eigen::Index rows() const {
        return cyclic() ? static_cast<Eigen::Index>(blocks_->size()) * blocks_->front().rows() : dense_->rows();
    }
    Eigen::Index local_rows() const { return blocks_->front().rows(); }

    void validate_shape(int cells) const {
        if (!cyclic()) {
            require(dense_->rows() > 0 && dense_->rows() == dense_->cols()
                    && dense_->rows() % cells == 0 && dense_->allFinite(), "invalid AO overlap");
            return;
        }
        require(blocks_->size() == static_cast<std::size_t>(cells) && !blocks_->empty(),
                "overlap must supply one home-row block per cell");
        const auto local = blocks_->front().rows();
        require(local > 0 && local <= std::numeric_limits<Eigen::Index>::max() / cells,
                "invalid or overflowing overlap block dimension");
        for (const auto& block : *blocks_)
            require(block.rows() == local && block.cols() == local && block.allFinite(),
                    "overlap blocks must be finite equally sized square matrices");
    }

    double entry(Eigen::Index row, Eigen::Index column, const Routes& routes) const {
        if (!cyclic()) return (*dense_)(row, column);
        const auto local = local_rows();
        return (*blocks_)[routes.difference(static_cast<int>(column / local), static_cast<int>(row / local))]
            (row % local, column % local);
    }

    Matrix row(Eigen::Index index, const Routes& routes) const {
        if (!cyclic()) return dense_->row(index);
        const auto local = local_rows();
        Matrix value(1, rows());
        for (int cell = 0; cell < static_cast<int>(blocks_->size()); ++cell)
            value.middleCols(cell * local, local) =
                (*blocks_)[routes.difference(cell, static_cast<int>(index / local))].row(index % local);
        return value;
    }

    Matrix apply(const Matrix& value, const Routes& routes) const {
        if (!cyclic()) return *dense_ * value;
        const auto local = local_rows();
        Matrix result = Matrix::Zero(rows(), value.cols());
        for (int target = 0; target < static_cast<int>(blocks_->size()); ++target)
            for (int source = 0; source < static_cast<int>(blocks_->size()); ++source)
                result.middleRows(target * local, local).noalias() +=
                    (*blocks_)[routes.difference(source, target)] * value.middleRows(source * local, local);
        return result;
    }

    Matrix left_apply(const Matrix& value, const Routes& routes) const {
        if (!cyclic()) return value.transpose() * *dense_;
        const auto local = local_rows();
        Matrix result = Matrix::Zero(value.cols(), rows());
        for (int target = 0; target < static_cast<int>(blocks_->size()); ++target)
            for (int source = 0; source < static_cast<int>(blocks_->size()); ++source)
                result.middleCols(source * local, local).noalias() +=
                    value.middleRows(target * local, local).transpose() * (*blocks_)[routes.difference(source, target)];
        return result;
    }

    double maximum() const {
        if (!cyclic()) return dense_->cwiseAbs().maxCoeff();
        double value = 0;
        for (const auto& block : *blocks_) value = std::max(value, block.cwiseAbs().maxCoeff());
        return value;
    }

    void validate(const Routes& routes, const std::vector<int>& mesh, int cells, double tolerance) const {
        if (!cyclic()) {
            const auto local = rows() / cells;
            const std::array<int, 3> generator{mesh[1] * mesh[2], mesh[2], 1};
            for (Eigen::Index a = 0; a < rows(); ++a) for (Eigen::Index b = 0; b < rows(); ++b) {
                require(std::abs((*dense_)(a, b) - (*dense_)(b, a)) <= tolerance, "AO overlap must be symmetric");
                for (int axis = 0; axis < 3; ++axis) if (mesh[axis] > 1) {
                    const auto ta = routes.difference(static_cast<int>(a / local), generator[axis]) * local + a % local;
                    const auto tb = routes.difference(static_cast<int>(b / local), generator[axis]) * local + b % local;
                    require(std::abs((*dense_)(a, b) - (*dense_)(ta, tb)) <= tolerance,
                            "AO overlap must be translation covariant");
                }
            }
            const Eigen::LLT<Matrix> cholesky(*dense_);
            require(cholesky.info() == Eigen::Success, "AO overlap must be positive definite");
            return;
        }
        for (int cell = 0; cell < cells; ++cell)
            require(close((*blocks_)[cell], (*blocks_)[routes.difference(0, cell)].transpose(), tolerance),
                    "overlap home-row blocks violate transpose symmetry");
        const auto coordinate = [&](int index) {
            return std::array<int, 3>{index / (mesh[1] * mesh[2]), (index / mesh[2]) % mesh[1], index % mesh[2]};
        };
        // The finite unitary Fourier transform block-diagonalizes a cyclic
        // metric. Validate every exact mesh symbol, without any cutoff or
        // square-AO Cholesky workspace. Unnormalized block sums are its spectra.
        const double two_pi = 2 * std::acos(-1.0);
        for (int wave = 0; wave < cells; ++wave) {
            const auto k = coordinate(wave);
            Eigen::MatrixXcd symbol = Eigen::MatrixXcd::Zero(local_rows(), local_rows());
            for (int cell = 0; cell < cells; ++cell) {
                const auto d = coordinate(cell);
                double phase = 0;
                for (int axis = 0; axis < 3; ++axis)
                    phase += static_cast<double>(k[axis]) * d[axis] / mesh[axis];
                symbol.noalias() += std::polar(1.0, two_pi * phase) * (*blocks_)[cell].cast<std::complex<double>>();
            }
            // Keep the AO-entry transpose-symmetry tolerance above as the
            // sole symmetry guard. Summing allowed entrywise roundoff over
            // cells must not introduce a stricter symbol-Hermiticity cutoff.
            require(symbol.allFinite(), "overlap reciprocal symbol must be finite");
            const Eigen::LLT<Eigen::MatrixXcd> cholesky(symbol);
            require(cholesky.info() == Eigen::Success, "overlap reciprocal symbol must be positive definite");
        }
    }

private:
    const Matrix* dense_ = nullptr;
    const std::vector<Matrix>* blocks_ = nullptr;
};

}  // namespace

TranslationMP2Result translation_mp2_solve(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Matrix>& integrals, const std::vector<Matrix>& virtual_fock,
    const std::vector<Matrix>& seeds, const std::vector<Matrix>& coefficients,
    const Matrix& occupied_fock, const Matrix& overlap,
    const std::vector<double>& energy_weights, int maximum_iterations,
    double energy_tolerance, double residual_tolerance, double damping,
    double symmetry_tolerance) {
    require(mesh.size() == 3 && n_bands > 0, "three positive mesh dimensions and bands required");
    int n_cells = 1;
    for (const int extent : mesh) {
        require(extent > 0 && n_cells <= std::numeric_limits<int>::max() / extent,
                "invalid or overflowing mesh");
        n_cells *= extent;
    }
    require(n_cells <= std::numeric_limits<int>::max() / n_bands, "occupied count overflow");
    const int n_occupied = n_cells * n_bands;
    const auto count = pair_i.size();
    require(count > 0 && pair_j.size() == count && integrals.size() == count
            && virtual_fock.size() == count && seeds.size() == count
            && coefficients.size() == count && energy_weights.size() == count,
            "representative array lengths differ");
    require(maximum_iterations > 0 && std::isfinite(energy_tolerance) && energy_tolerance >= 0
            && std::isfinite(residual_tolerance) && residual_tolerance >= 0
            && std::isfinite(damping) && damping > 0 && damping <= 1
            && std::isfinite(symmetry_tolerance) && symmetry_tolerance > 0,
            "invalid iteration options");
    require(overlap.rows() > 0 && overlap.rows() == overlap.cols()
            && overlap.rows() % n_cells == 0 && overlap.allFinite(), "invalid AO overlap");
    require(occupied_fock.rows() == n_occupied && occupied_fock.cols() == n_occupied
            && occupied_fock.allFinite(), "invalid occupied Fock dimensions or values");
    require(close(overlap, overlap.transpose(), symmetry_tolerance)
            && close(occupied_fock, occupied_fock.transpose(), symmetry_tolerance),
            "overlap and occupied Fock must be symmetric");
    const Eigen::LLT<Matrix> overlap_cholesky(overlap);
    require(overlap_cholesky.info() == Eigen::Success, "AO overlap must be positive definite");
    for (std::size_t p = 0; p < count; ++p)
        require(pair_i[p] >= 0 && pair_i[p] < n_bands && pair_j[p] >= 0
                && pair_j[p] < n_occupied, "representative occupied index out of range");
    const Routes routes(mesh, n_bands, pair_i, pair_j);
    const std::array<int, 3> generator{mesh[1] * mesh[2], mesh[2], 1};
    for (int axis = 0; axis < 3; ++axis) {
        if (mesh[axis] == 1) continue;
        for (const Matrix* value : {&overlap, &occupied_fock}) {
            const Matrix rows = routes.translate(*value, generator[axis]);
            const Matrix translated = routes.translate(rows.transpose(), generator[axis]).transpose();
            require(close(*value, translated, symmetry_tolerance),
                    "AO overlap and occupied Fock must be translation covariant");
        }
    }
    std::vector<Matrix> denominators(count), stabilizers(count);
    for (std::size_t p = 0; p < count; ++p) {
        const auto rank = coefficients[p].cols();
        require(rank > 0 && coefficients[p].rows() == overlap.rows()
                && coefficients[p].allFinite(), "invalid representative AO basis");
        for (const Matrix* block : {&integrals[p], &virtual_fock[p], &seeds[p]})
            require(block->rows() == rank && block->cols() == rank && block->allFinite(),
                    "invalid representative tensor dimensions or values");
        require(close(virtual_fock[p], virtual_fock[p].transpose(), symmetry_tolerance),
                "virtual Fock must be symmetric");
        require(close(coefficients[p].transpose() * overlap * coefficients[p],
                      Matrix::Identity(rank, rank), symmetry_tolerance),
                "representative AO basis must be metric orthonormal");
        const int i = pair_i[p], j = pair_j[p], displacement = j / n_bands;
        const bool half = i == j % n_bands && displacement != 0
            && routes.difference(0, displacement) == displacement;
        const double expected_weight = (i == j || half) ? 1.0 : 2.0;
        require(std::isfinite(energy_weights[p])
                && std::abs(energy_weights[p] - expected_weight) <= symmetry_tolerance,
                "incorrect per-cell representative energy weight");
        if (half) {
            const Matrix translated = routes.translate(coefficients[p], displacement);
            stabilizers[p] = coefficients[p].transpose() * overlap * translated;
            require(close(coefficients[p] * stabilizers[p], translated, symmetry_tolerance),
                    "pair basis is not closed under half translation");
            require(close(stabilizers[p] * virtual_fock[p] * stabilizers[p].transpose(),
                          virtual_fock[p], symmetry_tolerance),
                    "virtual Fock violates half translation");
        } else if (i == j) {
            stabilizers[p] = Matrix::Identity(rank, rank);
        }
        if (stabilizers[p].size()) {
            for (const Matrix* block : {&integrals[p], &seeds[p]})
                require(close(*block, stabilizers[p] * block->transpose() * stabilizers[p].transpose(),
                              symmetry_tolerance), "tensor violates pair exchange sewing");
        }
        denominators[p].resize(rank, rank);
        for (Eigen::Index a = 0; a < rank; ++a) for (Eigen::Index b = 0; b < rank; ++b) {
            const double value = occupied_fock(i, i) + occupied_fock(j, j)
                - virtual_fock[p](a, a) - virtual_fock[p](b, b);
            require(std::isfinite(value) && value != 0, "denominators must be finite and nonzero");
            denominators[p](a, b) = value;
        }
    }
    TranslationMP2Result result;
    result.amplitudes = seeds;
    result.residuals.resize(count);
    double previous_energy = 0;
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        double maximum_residual = 0, energy = 0, energy_compensation = 0;
        for (std::size_t p = 0; p < count; ++p) {
            const int i = pair_i[p], j = pair_j[p];
            const Matrix& amplitude = result.amplitudes[p];
            // Nejad et al., doi:10.1063/5.0290816, Eqs. (42)-(44):
            // source-space amplitudes enter through both cross-pair overlaps.
            Matrix residual = integrals[p] + virtual_fock[p] * amplitude + amplitude * virtual_fock[p];
            const Matrix metric_target = coefficients[p].transpose() * overlap;
            const auto subtract_source = [&](int a, int b, double f) {
                const Route source = routes.resolve(a, b);
                const Matrix link = metric_target * routes.translate(coefficients[source.row], source.shift);
                // Always transport, even when source.row == p: its common
                // translation or exchange orientation can still differ.
                if (source.transpose)
                    residual.noalias() -= f * (link * result.amplitudes[source.row].transpose() * link.transpose());
                else
                    residual.noalias() -= f * (link * result.amplitudes[source.row] * link.transpose());
            };
            for (int k = 0; k < n_occupied; ++k) {
                subtract_source(k, j, occupied_fock(i, k));
                subtract_source(i, k, occupied_fock(k, j));
            }
            finite(residual, "non-finite raw residual");
            maximum_residual = std::max(maximum_residual, residual.cwiseAbs().maxCoeff());
            result.residuals[p] = std::move(residual);
            const double term = energy_weights[p]
                * (integrals[p].array() * (2 * amplitude - amplitude.transpose()).array()).sum();
            const double compensated = term - energy_compensation;
            const double sum = energy + compensated;
            energy_compensation = (sum - energy) - compensated;
            energy = sum;
        }
        if (!std::isfinite(energy)) throw std::runtime_error("translation MP2: non-finite energy");
        const double change = iteration == 1 ? 0.0 : energy - previous_energy;
        result.trace.push_back({iteration, energy, change, maximum_residual});
        result.converged = iteration > 1 && maximum_residual <= residual_tolerance
            && std::abs(change) <= energy_tolerance;
        if (result.converged || iteration == maximum_iterations) break;
        previous_energy = energy;
        for (std::size_t p = 0; p < count; ++p) {
            result.amplitudes[p].array() += damping * result.residuals[p].array() / denominators[p].array();
            finite(result.amplitudes[p], "non-finite amplitude update");
            if (stabilizers[p].size())
                require(close(result.amplitudes[p], stabilizers[p] * result.amplitudes[p].transpose()
                              * stabilizers[p].transpose(), symmetry_tolerance),
                        "Jacobi update violates pair exchange sewing");
        }
    }
    return result;
}

static TranslationMP2Transport prepare_translation_mp2_transport_impl(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Matrix>& virtual_fock, const std::vector<Matrix>& coefficients,
    const Matrix& occupied_fock, const MetricView& overlap,
    const std::vector<double>& energy_weights, double tolerance,
    std::uint64_t maximum_bytes, const std::vector<std::vector<int>>& domains,
    const std::vector<Matrix>& local_coefficients,
    const std::vector<Matrix>& occupied_coefficients, const Matrix& all_occupied, bool occupied_home) {
    require(mesh.size() == 3 && n_bands > 0, "three positive mesh dimensions and bands required");
    int cells = 1;
    for (int extent : mesh) {
        require(extent > 0 && cells <= std::numeric_limits<int>::max() / extent, "invalid or overflowing mesh");
        cells *= extent;
    }
    require(cells <= std::numeric_limits<int>::max() / n_bands, "occupied count overflow");
    const int occupied = cells * n_bands;
    const auto count = pair_i.size();
    const bool recipe_only = coefficients.empty() && !domains.empty();
    require(count && pair_j.size() == count && (recipe_only || coefficients.size() == count)
            && virtual_fock.size() == count && energy_weights.size() == count,
            "representative array lengths differ");
    require(std::isfinite(tolerance) && tolerance > 0 && maximum_bytes > 0,
            "invalid preparation tolerance or byte budget");
    overlap.validate_shape(cells);
    require(occupied_fock.rows() == occupied && occupied_fock.cols() == occupied
            && occupied_fock.allFinite(), "invalid occupied Fock");
    const bool compact = !domains.empty();
    require(!occupied_home || compact, "occupied home columns require projected pair metadata");
    require(!occupied_home || all_occupied.cols() <= overlap.rows()/cells,
            "occupied home dimension exceeds primitive AO count");
    const Eigen::Index total_occupied = all_occupied.cols() * (occupied_home ? cells : 1);
    require((compact && domains.size() == count && local_coefficients.size() == count
             && occupied_coefficients.size() == count && all_occupied.rows() == overlap.rows()
             && total_occupied >= occupied && total_occupied <= overlap.rows()
             && total_occupied % cells == 0 && all_occupied.allFinite())
            || (!compact && local_coefficients.empty() && occupied_coefficients.empty() && !all_occupied.size()),
            "inconsistent projected pair-basis metadata");
    const auto rank_of = [&](std::size_t p) {
        return recipe_only ? local_coefficients[p].cols() : coefficients[p].cols();
    };
    for (std::size_t p = 0; p < count; ++p) {
        require(pair_i[p] >= 0 && pair_i[p] < n_bands && pair_j[p] >= 0 && pair_j[p] < occupied,
                "representative occupied index out of range");
        const auto rank = rank_of(p);
        require(rank > 0 && rank <= overlap.rows() && (recipe_only
                || (coefficients[p].rows() == overlap.rows() && coefficients[p].allFinite())),
                "invalid representative AO basis");
        require(virtual_fock[p].rows() == rank && virtual_fock[p].cols() == rank && virtual_fock[p].allFinite(),
                "invalid virtual Fock");
        if (compact) {
            require(!domains[p].empty() && local_coefficients[p].rows() == static_cast<Eigen::Index>(domains[p].size())
                    && local_coefficients[p].cols() == rank && local_coefficients[p].allFinite()
                    && occupied_coefficients[p].rows() == total_occupied
                    && occupied_coefficients[p].cols() == rank && occupied_coefficients[p].allFinite(),
                    "invalid projected pair-basis dimensions or values");
            auto indices = domains[p];
            std::sort(indices.begin(), indices.end());
            require(indices.front() >= 0 && indices.back() < overlap.rows()
                    && std::adjacent_find(indices.begin(), indices.end()) == indices.end(),
                    "AO domains must have unique indices in range");
        }
    }
    const Routes routes(mesh, n_bands, pair_i, pair_j);
    const auto bytes_add = [](std::uint64_t a, std::uint64_t b) {
        if (a > std::numeric_limits<std::uint64_t>::max() - b)
            throw std::length_error("translation transport byte count overflow");
        return a + b;
    };
    const auto bytes_product = [](std::uint64_t a, std::uint64_t b) {
        if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
            throw std::length_error("translation transport byte count overflow");
        return a * b;
    };
    const auto matrix_bytes = [&](std::uint64_t rows, std::uint64_t columns) {
        return bytes_product(bytes_product(rows, columns), sizeof(double));
    };
    const auto check_budget = [&](std::uint64_t bytes) {
        if (bytes > maximum_bytes)
            throw std::length_error("translation transport exceeds maximum_bytes before link allocation");
    };
    const std::uint64_t slots = bytes_product(bytes_product(count, occupied), 2);
    std::uint64_t retained = bytes_add(sizeof(TranslationMP2Transport),
        bytes_product(slots, sizeof(TranslationMP2Transport::Coupling)));
    retained = bytes_add(retained, bytes_product(count, 3 * sizeof(Matrix) + sizeof(double)));
    check_budget(retained);
    using Key = std::pair<std::size_t, int>;
    // Metadata is also preflighted; no overlap link or target metric has yet
    // been allocated. Slots retain exact p,k,leg order, including zero Fock.
    std::uint64_t workspace = bytes_product(slots, sizeof(Key));
    workspace = bytes_add(workspace, bytes_product(count, sizeof(std::vector<Key>)));
    workspace = bytes_add(workspace, bytes_product(count, 3 * sizeof(double)));
    check_budget(bytes_add(retained, workspace));
    std::vector<std::vector<Key>> keys(count);
    std::uint64_t largest_metric = 0, largest_link_scratch = 0;
    for (std::size_t p = 0; p < count; ++p) {
        const int i = pair_i[p], j = pair_j[p];
        keys[p].reserve(2 * static_cast<std::size_t>(occupied));
        for (int k = 0; k < occupied; ++k) for (const auto pair : {std::make_pair(k, j), std::make_pair(i, k)}) {
            const auto route = routes.resolve(pair.first, pair.second);
            keys[p].emplace_back(route.row, route.shift);
        }
        std::sort(keys[p].begin(), keys[p].end());
        keys[p].erase(std::unique(keys[p].begin(), keys[p].end()), keys[p].end());
        const auto rank = rank_of(p);
        const bool half = i == j % n_bands && j / n_bands != 0
            && routes.difference(0, j / n_bands) == j / n_bands;
        retained = bytes_add(retained, bytes_product(matrix_bytes(rank, rank), (i == j || half) ? 3 : 2));
        largest_metric = std::max(largest_metric, matrix_bytes(rank, overlap.rows()));
        for (const auto& key : keys[p]) {
            const auto source_rank = rank_of(key.first);
            retained = bytes_add(retained, bytes_add(sizeof(TranslationMP2Transport::Link), matrix_bytes(rank, source_rank)));
            auto temporary = matrix_bytes(rank, source_rank);
            if (compact) {
                temporary = bytes_add(temporary, matrix_bytes(rank, domains[key.first].size()));
                temporary = bytes_add(temporary, matrix_bytes(total_occupied, source_rank));
            }
            largest_link_scratch = std::max(largest_link_scratch, temporary);
        }
        check_budget(bytes_add(retained, workspace));
    }
    // Conservative numerical-workspace bound covers validation, a cached S*C,
    // reconstruction, target metric and one link. Caller inputs and pybind
    // conversion copies are deliberately outside this transport-only budget.
    workspace = bytes_add(workspace, overlap.cyclic()
        ? bytes_product(matrix_bytes(overlap.local_rows(), overlap.local_rows()), 8)
        : bytes_product(matrix_bytes(overlap.rows(), overlap.rows()), 3));
    workspace = bytes_add(workspace, bytes_product(matrix_bytes(all_occupied.rows(), all_occupied.cols()), 3));
    // Three streamed column/reconstruction buffers; home and S*home copies
    // above are charged using their actual primitive width.
    if (occupied_home)
        workspace = bytes_add(workspace, bytes_product(matrix_bytes(overlap.rows(), 1), 6));
    // Recipe-only stable links reconstruct one source basis at a time. The
    // extra AO rectangle is charged before any reconstruction, never a pool
    // of all representative bases or a pybind copy of that pool.
    workspace = bytes_add(workspace, bytes_product(largest_metric, recipe_only ? 4 : 3));
    workspace = bytes_add(workspace, largest_link_scratch);
    const auto required = bytes_add(retained, workspace);
    check_budget(required);
    const std::array<int, 3> generator{mesh[1] * mesh[2], mesh[2], 1};
    overlap.validate(routes, mesh, cells, tolerance);
    for (const Matrix* matrix : {&occupied_fock}) {
        const auto local = matrix->rows() / cells;
        for (Eigen::Index a = 0; a < matrix->rows(); ++a) for (Eigen::Index b = 0; b < matrix->cols(); ++b) {
            require(std::abs((*matrix)(a, b) - (*matrix)(b, a)) <= tolerance,
                    "overlap and occupied Fock must be symmetric");
            for (int axis = 0; axis < 3; ++axis) if (mesh[axis] > 1) {
                const auto ta = routes.difference(static_cast<int>(a / local), generator[axis]) * local + a % local;
                const auto tb = routes.difference(static_cast<int>(b / local), generator[axis]) * local + b % local;
                require(std::abs((*matrix)(a, b) - (*matrix)(ta, tb)) <= tolerance,
                        "overlap and occupied Fock must be translation covariant");
            }
        }
    }
    const auto translate_occupied = [&](const Matrix& value, int shift) {
        Matrix shifted(value.rows(), value.cols());
        shifted.topRows(occupied) = routes.translate(value.topRows(occupied), shift);
        if (value.rows() > occupied)
            shifted.bottomRows(value.rows() - occupied) = routes.translate(value.bottomRows(value.rows() - occupied), shift);
        return shifted;
    };
    Matrix metric_occupied;
    double occupied_gram_error = 0, occupied_gauge_error = 0;
    std::vector<double> local_norm(count), tail_norm(count), projection_error(count);
    const auto column = [&](const Matrix& home, Eigen::Index index) -> Matrix {
        if (!occupied_home) return home.col(index);
        const bool active = index < occupied;
        const auto bands = active ? n_bands : all_occupied.cols()-n_bands;
        const auto relative = active ? index : index-occupied;
        return routes.translate(home.col((active ? 0 : n_bands)+relative%bands), relative/bands);
    };
    const auto metric_column = [&](Eigen::Index index) -> Matrix {
        // A dense metric may only be covariant within tolerance. Preserve its
        // actual action instead of silently enforcing exact covariance.
        if (occupied_home && !overlap.cyclic()) return overlap.apply(column(all_occupied, index), routes);
        return column(metric_occupied, index);
    };
    const auto occupied_action = [&](const Matrix& weights, bool metric) -> Matrix {
        if (!occupied_home) return (metric ? metric_occupied : all_occupied) * weights;
        Matrix result = Matrix::Zero(overlap.rows(), weights.cols());
        for (Eigen::Index c = 0; c < total_occupied; ++c)
            result.noalias() += (metric ? metric_column(c) : column(all_occupied, c)) * weights.row(c);
        return result;
    };
    if (compact) {
        metric_occupied = overlap.apply(all_occupied, routes);
        if (occupied_home) {
            // Every Gram and translation entry, not covariance alone or a
            // sampled subblock. Complete selected C and S*C never exist.
            for (Eigen::Index c = 0; c < total_occupied; ++c) {
                const Matrix sc = metric_column(c);
                for (Eigen::Index d = 0; d < total_occupied; ++d) {
                    const double error = std::abs((column(all_occupied, d).array()*sc.array()).sum() - double(c == d));
                    occupied_gram_error = std::max(occupied_gram_error, error);
                }
                const bool active = c < occupied;
                const auto bands = active ? n_bands : all_occupied.cols()-n_bands;
                const auto relative = active ? c : c-occupied;
                for (int axis = 0; axis < 3; ++axis) if (mesh[axis] > 1) {
                    const auto target = routes.difference(relative/bands, routes.difference(0, generator[axis]));
                    const Matrix translated = routes.translate(column(all_occupied, c), generator[axis]);
                    const Matrix expected = column(all_occupied, (active ? 0 : occupied)+target*bands+relative%bands);
                    occupied_gauge_error = std::max(occupied_gauge_error, (translated-expected).cwiseAbs().maxCoeff());
                }
            }
            require(occupied_gram_error <= tolerance, "projected-basis occupied space must be metric orthonormal");
            require(occupied_gauge_error <= tolerance, "projected-basis occupied gauge is not cell-major");
        } else {
            const Matrix gram = all_occupied.transpose() * metric_occupied;
            occupied_gram_error = (gram - Matrix::Identity(gram.rows(), gram.cols())).cwiseAbs().maxCoeff();
            require(close(gram, Matrix::Identity(total_occupied, total_occupied), tolerance),
                    "projected-basis occupied space must be metric orthonormal");
            for (int axis = 0; axis < 3; ++axis) if (mesh[axis] > 1) {
                const Matrix translated = routes.translate(all_occupied, generator[axis]);
                const Matrix expected = translate_occupied(all_occupied.transpose(),
                    routes.difference(0, generator[axis])).transpose();
                occupied_gauge_error = std::max(occupied_gauge_error, (translated-expected).cwiseAbs().maxCoeff());
                require(close(translated, expected, tolerance), "projected-basis occupied gauge is not cell-major");
            }
        }
        for (std::size_t p = 0; p < count; ++p) {
            Matrix projected = Matrix::Zero(total_occupied, rank_of(p));
            if (occupied_home) {
                for (Eigen::Index c = 0; c < total_occupied; ++c) {
                    const Matrix sc = metric_column(c);
                    for (std::size_t a = 0; a < domains[p].size(); ++a)
                        projected.row(c).noalias() += sc(domains[p][a], 0) * local_coefficients[p].row(a);
                }
            } else {
                for (std::size_t a = 0; a < domains[p].size(); ++a)
                    projected.noalias() += metric_occupied.row(domains[p][a]).transpose() * local_coefficients[p].row(a);
            }
            require(close(projected, occupied_coefficients[p], tolerance),
                    "projected pair-basis metadata does not reconstruct its AO basis");
            if (!recipe_only) {
                Matrix reconstructed = -occupied_action(occupied_coefficients[p], false);
                for (std::size_t a = 0; a < domains[p].size(); ++a)
                    reconstructed.row(domains[p][a]) += local_coefficients[p].row(a);
                require(close(reconstructed, coefficients[p], tolerance),
                        "projected pair-basis metadata does not reconstruct its AO basis");
            }
            local_norm[p] = local_coefficients[p].cwiseAbs().colwise().sum().maxCoeff();
            tail_norm[p] = occupied_coefficients[p].cwiseAbs().colwise().sum().maxCoeff();
            projection_error[p] = (projected - occupied_coefficients[p]).cwiseAbs().maxCoeff();
        }
    }
    const auto compact_overlap = [&](std::size_t p, std::size_t q, int shift) {
        const auto local = overlap.rows() / cells;
        const int inverse = routes.difference(0, shift);
        Matrix selected_metric = Matrix::Zero(rank_of(p), domains[q].size());
        for (std::size_t b = 0; b < domains[q].size(); ++b) {
            const int index = domains[q][b];
            const auto translated = routes.difference(index / local, inverse) * local + index % local;
            for (std::size_t a = 0; a < domains[p].size(); ++a)
                selected_metric.col(b).noalias() += overlap.entry(domains[p][a], translated, routes)
                    * local_coefficients[p].row(a).transpose();
        }
        Matrix link = selected_metric * local_coefficients[q]
            - occupied_coefficients[p].transpose() * translate_occupied(occupied_coefficients[q], shift);
        finite(link, "non-finite compact overlap link");
        return link;
    };
    TranslationMP2Transport result;
    result.n_occupied = occupied;
    result.symmetry_tolerance = tolerance;
    result.compact = compact;
    result.recipe_only = recipe_only;
    result.cyclic_metric = overlap.cyclic();
    result.retained_bytes = retained;
    result.workspace_bytes = workspace;
    result.required_bytes = required;
    result.virtual_fock = virtual_fock;
    result.denominators.resize(count);
    result.stabilizers.resize(count);
    result.energy_weights = energy_weights;
    result.couplings.reserve(static_cast<std::size_t>(slots));
    std::size_t n_links = 0;
    for (const auto& row : keys) n_links += row.size();
    result.links.reserve(n_links);
    const double maximum_overlap = overlap.maximum();
    for (std::size_t p = 0; p < count; ++p) {
        const int i = pair_i[p], j = pair_j[p], displacement = j / n_bands;
        const auto rank = rank_of(p);
        require(close(virtual_fock[p], virtual_fock[p].transpose(), tolerance), "virtual Fock must be symmetric");
        Matrix target_metric;
        bool stable = false;
        if (compact) {
            // The two-term projected overlap can subtract O(||A||^2)
            // quantities. Large PAO normalization also amplifies the actual
            // finite-precision C Gram/projection defects. Select an equivalent
            // evaluation before that bound approaches the admission tolerance;
            // this is not a pair, orbital or rank cutoff.
            for (const auto& key : keys[p]) {
                const auto q = key.first;
                const double scale = maximum_overlap * local_norm[p] * local_norm[q]
                    + tail_norm[p] * tail_norm[q];
                const double roundoff = 16 * std::numeric_limits<double>::epsilon()
                    * (domains[p].size() + domains[q].size() + total_occupied + 2) * scale;
                const double identities = occupied_gram_error * tail_norm[p] * tail_norm[q]
                    + projection_error[p] * tail_norm[q] + tail_norm[p] * projection_error[q];
                const double gauge = occupied_gauge_error * overlap.rows() * maximum_overlap
                    * (local_norm[p] + local_norm[q] + tail_norm[p] + tail_norm[q] + 1)
                    * (tail_norm[p] + tail_norm[q]);
                stable = stable || !std::isfinite(roundoff + identities + gauge)
                    || roundoff + identities + gauge > tolerance / 16;
            }
            if (stable) {
                // S*V follows from selected S columns and the shared S*C.
                // Avoid both the large O(A^2)-O(B^2) subtraction and S*V AO
                // multiplication. Actual source AO coefficients retain the
                // admitted finite-precision occupied tails during transport.
                target_metric = -occupied_action(occupied_coefficients[p], true).transpose();
                for (std::size_t a = 0; a < domains[p].size(); ++a)
                    target_metric.noalias() += local_coefficients[p].row(a).transpose()
                        * overlap.row(domains[p][a], routes);
                finite(target_metric, "non-finite selected-column target metric");
                result.stable_links += keys[p].size();
            }
        } else target_metric = overlap.left_apply(coefficients[p], routes);
        const auto link = [&](std::size_t q, int shift) -> Matrix {
            if (compact && !stable) return compact_overlap(p, q, shift);
            Matrix reconstructed;
            if (recipe_only) {
                reconstructed.noalias() = -occupied_action(occupied_coefficients[q], false);
                for (std::size_t a = 0; a < domains[q].size(); ++a)
                    reconstructed.row(domains[q][a]) += local_coefficients[q].row(a);
            }
            const Matrix& source = recipe_only ? reconstructed : coefficients[q];
            Matrix value = Matrix::Zero(rank, rank_of(q));
            const auto local = overlap.rows() / cells;
            for (int cell = 0; cell < cells; ++cell)
                value.noalias() += target_metric.middleCols(cell * local, local)
                    * source.middleRows(routes.difference(cell, shift) * local, local);
            finite(value, "non-finite overlap link");
            return value;
        };
        require(close(link(p, 0), Matrix::Identity(rank, rank), tolerance),
                "representative AO basis must be metric orthonormal");
        const bool half = i == j % n_bands && displacement != 0
            && routes.difference(0, displacement) == displacement;
        require(std::isfinite(energy_weights[p])
                && std::abs(energy_weights[p] - ((i == j || half) ? 1.0 : 2.0)) <= tolerance,
                "incorrect per-cell representative energy weight");
        if (half) {
            result.stabilizers[p] = link(p, displacement);
            require(close(result.stabilizers[p].transpose() * result.stabilizers[p], Matrix::Identity(rank, rank), tolerance),
                    "pair basis is not closed under half translation");
            require(close(result.stabilizers[p] * virtual_fock[p] * result.stabilizers[p].transpose(),
                          virtual_fock[p], tolerance), "virtual Fock violates half translation");
        } else if (i == j) result.stabilizers[p] = Matrix::Identity(rank, rank);
        result.denominators[p].resize(rank, rank);
        for (Eigen::Index a = 0; a < rank; ++a) for (Eigen::Index b = 0; b < rank; ++b) {
            const double denominator = occupied_fock(i, i) + occupied_fock(j, j)
                - virtual_fock[p](a, a) - virtual_fock[p](b, b);
            require(std::isfinite(denominator) && denominator != 0, "denominators must be finite and nonzero");
            result.denominators[p](a, b) = denominator;
        }
        const auto offset = result.links.size();
        for (const auto& key : keys[p]) result.links.push_back({link(key.first, key.second), key.first});
        for (int k = 0; k < occupied; ++k) {
            for (int leg = 0; leg < 2; ++leg) {
                const auto route = leg == 0 ? routes.resolve(k, j) : routes.resolve(i, k);
                const auto position = std::lower_bound(keys[p].begin(), keys[p].end(), Key{route.row, route.shift}) - keys[p].begin();
                result.couplings.push_back({offset + static_cast<std::size_t>(position),
                    leg == 0 ? occupied_fock(i, k) : occupied_fock(k, j), route.transpose});
            }
        }
    }
    return result;
}

TranslationMP2Transport prepare_translation_mp2_transport(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Matrix>& virtual_fock, const std::vector<Matrix>& coefficients,
    const Matrix& occupied_fock, const Matrix& overlap,
    const std::vector<double>& energy_weights, double tolerance,
    std::uint64_t maximum_bytes, const std::vector<std::vector<int>>& domains,
    const std::vector<Matrix>& local_coefficients,
    const std::vector<Matrix>& occupied_coefficients, const Matrix& all_occupied, bool occupied_home) {
    return prepare_translation_mp2_transport_impl(mesh, n_bands, pair_i, pair_j,
        virtual_fock, coefficients, occupied_fock, MetricView(overlap), energy_weights,
        tolerance, maximum_bytes, domains, local_coefficients, occupied_coefficients, all_occupied, occupied_home);
}

TranslationMP2Transport prepare_translation_mp2_transport_blocks(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Matrix>& virtual_fock, const std::vector<Matrix>& coefficients,
    const Matrix& occupied_fock, const std::vector<Matrix>& overlap_home_blocks,
    const std::vector<double>& energy_weights, double tolerance,
    std::uint64_t maximum_bytes, const std::vector<std::vector<int>>& domains,
    const std::vector<Matrix>& local_coefficients,
    const std::vector<Matrix>& occupied_coefficients, const Matrix& all_occupied, bool occupied_home) {
    return prepare_translation_mp2_transport_impl(mesh, n_bands, pair_i, pair_j,
        virtual_fock, coefficients, occupied_fock, MetricView(overlap_home_blocks), energy_weights,
        tolerance, maximum_bytes, domains, local_coefficients, occupied_coefficients, all_occupied, occupied_home);
}

TranslationMP2Result translation_mp2_solve_prepared(
    const TranslationMP2Transport& transport, const std::vector<Matrix>& integrals,
    const std::vector<Matrix>& seeds, int maximum_iterations,
    double energy_tolerance, double residual_tolerance, double damping) {
    const auto count = transport.virtual_fock.size();
    require(count > 0 && integrals.size() == count && seeds.size() == count, "representative array lengths differ");
    require(maximum_iterations > 0 && std::isfinite(energy_tolerance) && energy_tolerance >= 0
            && std::isfinite(residual_tolerance) && residual_tolerance >= 0
            && std::isfinite(damping) && damping > 0 && damping <= 1, "invalid iteration options");
    for (std::size_t p = 0; p < count; ++p) for (const Matrix* block : {&integrals[p], &seeds[p]}) {
        const auto rank = transport.virtual_fock[p].rows();
        require(block->rows() == rank && block->cols() == rank && block->allFinite(), "invalid representative tensor");
        const auto& stabilizer = transport.stabilizers[p];
        if (stabilizer.size())
            require(close(*block, stabilizer * block->transpose() * stabilizer.transpose(), transport.symmetry_tolerance),
                    "tensor violates pair exchange sewing");
    }
    TranslationMP2Result result;
    result.amplitudes = seeds;
    result.residuals.resize(count);
    double previous_energy = 0;
    const auto slots_per_row = 2 * static_cast<std::size_t>(transport.n_occupied);
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        double maximum_residual = 0, energy = 0, compensation = 0;
        for (std::size_t p = 0; p < count; ++p) {
            const auto& amplitude = result.amplitudes[p];
            const auto& fock = transport.virtual_fock[p];
            Matrix residual = integrals[p] + fock * amplitude + amplitude * fock;
            // Riplinger/Neese doi:10.1063/1.4773581 Eq. (30), and Nejad et
            // al. doi:10.1063/5.0290816 Eqs. (42)-(44): source-space transport.
            // The cached graph carries overlaps only, never placed amplitudes.
            for (std::size_t slot = p * slots_per_row; slot < (p + 1) * slots_per_row; ++slot) {
                const auto& coupling = transport.couplings[slot];
                const auto& link = transport.links[coupling.link];
                const auto& source = result.amplitudes[link.source];
                if (coupling.transpose)
                    residual.noalias() -= coupling.fock * (link.overlap * source.transpose() * link.overlap.transpose());
                else
                    residual.noalias() -= coupling.fock * (link.overlap * source * link.overlap.transpose());
            }
            finite(residual, "non-finite raw residual");
            maximum_residual = std::max(maximum_residual, residual.cwiseAbs().maxCoeff());
            result.residuals[p] = std::move(residual);
            const double term = transport.energy_weights[p]
                * (integrals[p].array() * (2 * amplitude - amplitude.transpose()).array()).sum();
            const double corrected = term - compensation;
            const double sum = energy + corrected;
            compensation = (sum - energy) - corrected;
            energy = sum;
        }
        if (!std::isfinite(energy)) throw std::runtime_error("translation MP2: non-finite energy");
        const double change = iteration == 1 ? 0 : energy - previous_energy;
        result.trace.push_back({iteration, energy, change, maximum_residual});
        result.converged = iteration > 1 && maximum_residual <= residual_tolerance && std::abs(change) <= energy_tolerance;
        if (result.converged || iteration == maximum_iterations) break;
        previous_energy = energy;
        for (std::size_t p = 0; p < count; ++p) {
            result.amplitudes[p].array() += damping * result.residuals[p].array() / transport.denominators[p].array();
            finite(result.amplitudes[p], "non-finite amplitude update");
            const auto& stabilizer = transport.stabilizers[p];
            if (stabilizer.size())
                require(close(result.amplitudes[p], stabilizer * result.amplitudes[p].transpose() * stabilizer.transpose(),
                              transport.symmetry_tolerance), "Jacobi update violates pair exchange sewing");
        }
    }
    return result;
}

}  // namespace vibeqc
