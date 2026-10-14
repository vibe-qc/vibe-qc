#include "vibeqc/translation_mp2.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
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

}  // namespace vibeqc
