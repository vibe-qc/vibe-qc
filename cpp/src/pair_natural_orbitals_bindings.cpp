// Bounded internal diagnostics for the native restricted pair-PNO primitive.
// Production numerical callers consume its C++ result directly.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vibeqc/pair_natural_orbitals.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {

constexpr std::size_t kPairPNODiagnosticMaximumDimension = 128U;
constexpr std::uint64_t kPairPNODiagnosticMaximumSweeps = 200U;

py::array_t<double> pair_pno_matrix_copy(
    const std::vector<double>& values, std::size_t rows, std::size_t columns) {
    if (rows > kPairPNODiagnosticMaximumDimension
        || columns > kPairPNODiagnosticMaximumDimension
        || (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows)
        || values.size() != rows * columns) {
        throw std::logic_error("pair PNO diagnostic result has an invalid matrix extent");
    }
    py::array_t<double> copy({static_cast<py::ssize_t>(rows),
                              static_cast<py::ssize_t>(columns)});
    if (!values.empty()) {
        std::memcpy(copy.mutable_data(), values.data(), values.size() * sizeof(double));
    }
    return copy;
}

}  // namespace

void bind_pair_natural_orbitals(py::module_& m) {
    using Kind = vibeqc::RestrictedPairKind;
    using Plan = vibeqc::RestrictedPairPNOMemoryPlan;
    using Result = vibeqc::RestrictedPairPNOResult;

    using MP2Plan = vibeqc::RestrictedPairMP2MemoryPlan;
    using MP2 = vibeqc::RestrictedPairMP2Result;
    py::class_<MP2Plan>(m, "_RestrictedPairMP2MemoryPlan")
        .def_readonly("domain_dimension", &MP2Plan::domain_dimension)
        .def_readonly("input_bytes", &MP2Plan::input_bytes)
        .def_readonly("peak_owned_numerical_bytes", &MP2Plan::peak_owned_numerical_bytes);
    py::class_<MP2>(m, "_RestrictedPairMP2Result")
        .def_readonly("domain_dimension", &MP2::domain_dimension)
        .def_readonly("ordered_pair_energy", &MP2::ordered_pair_energy)
        .def_readonly("minimum_denominator", &MP2::minimum_denominator)
        .def_readonly("maximum_denominator", &MP2::maximum_denominator)
        .def_readonly("maximum_absolute_amplitude", &MP2::maximum_absolute_amplitude)
        .def_readonly("maximum_residual", &MP2::maximum_residual)
        .def_readonly("residual_frobenius_norm", &MP2::residual_frobenius_norm)
        .def_readonly("energy_term_underflow_count", &MP2::energy_term_underflow_count)
        .def_readonly("energy_input_scaling_underflow_count", &MP2::energy_input_scaling_underflow_count)
        .def_property_readonly("memory", [](const MP2& result) { return result.memory; })
        .def("amplitudes_copy", [](const MP2& result) {
            return pair_pno_matrix_copy(result.amplitudes, result.domain_dimension,
                                        result.domain_dimension);
        });
    m.def("_plan_restricted_pair_semicanonical_mp2",
          &vibeqc::plan_restricted_pair_semicanonical_mp2, py::arg("domain_dimension"));
    m.def("_restricted_pair_semicanonical_mp2",
        [](const py::array& integrals, const py::array& energies,
           double fii, double fjj, double denominator_floor, std::uint64_t cap) {
            if (!integrals.dtype().is(py::dtype::of<double>()) || integrals.ndim() != 2
                || !(integrals.flags() & py::array::c_style)
                || integrals.shape(0) != integrals.shape(1)
                || !energies.dtype().is(py::dtype::of<double>()) || energies.ndim() != 1
                || !(energies.flags() & py::array::c_style)
                || energies.shape(0) != integrals.shape(0)) {
                throw std::invalid_argument("pair MP2 requires existing C-contiguous float64 G[n,n] and eps[n]");
            }
            const auto n = static_cast<std::size_t>(integrals.shape(0));
            if (n > kPairPNODiagnosticMaximumDimension) {
                throw std::length_error("pair MP2 diagnostic dimension exceeds its tiny-matrix limit");
            }
            const auto* g = static_cast<const double*>(integrals.data());
            const auto* eps = static_cast<const double*>(energies.data());
            const auto extent = static_cast<std::size_t>(integrals.size());
            py::gil_scoped_release release;
            return vibeqc::restricted_pair_semicanonical_mp2(
                g, extent, eps, n, n, fii, fjj, denominator_floor, cap);
        }, py::arg("integrals").noconvert(), py::arg("virtual_energies").noconvert(),
        py::arg("occupied_fock_ii"), py::arg("occupied_fock_jj"),
        py::arg("denominator_floor"), py::arg("numerical_byte_cap"),
        "Tiny native semi-canonical MP2 initial pair. Inputs must remain immutable; "
        "no coupled solution, pair/cell multiplicity, PNO or energy correction is implied.");

    m.attr("PAIR_PNO_DIAGNOSTIC_MAXIMUM_DIMENSION") =
        py::int_(kPairPNODiagnosticMaximumDimension);
    m.attr("PAIR_PNO_DIAGNOSTIC_MAXIMUM_SWEEPS") =
        py::int_(kPairPNODiagnosticMaximumSweeps);
    py::enum_<Kind>(m, "RestrictedPairKind")
        .value("Diagonal", Kind::Diagonal)
        .value("OffDiagonal", Kind::OffDiagonal);
    py::class_<Plan>(m, "RestrictedPairPNOMemoryPlan")
        .def_readonly("domain_dimension", &Plan::domain_dimension)
        .def_readonly("input_bytes", &Plan::input_bytes)
        .def_readonly("output_bytes_upper_bound", &Plan::output_bytes_upper_bound)
        .def_readonly("eigensolver_workspace_bytes", &Plan::eigensolver_workspace_bytes)
        .def_readonly("peak_owned_numerical_bytes", &Plan::peak_owned_numerical_bytes);
    py::class_<Result>(m, "RestrictedPairPNOResult")
        .def_readonly("pair_kind", &Result::pair_kind)
        .def_readonly("domain_dimension", &Result::domain_dimension)
        .def_readonly("retained_dimension", &Result::retained_dimension)
        .def_readonly("occupation_cutoff", &Result::occupation_cutoff)
        .def_readonly("density_trace", &Result::density_trace)
        .def_readonly("discarded_occupation_sum", &Result::discarded_occupation_sum)
        .def_readonly("minimum_occupation", &Result::minimum_occupation)
        .def_readonly("negative_occupation_count", &Result::negative_occupation_count)
        .def_readonly("input_relative_antisymmetric_norm",
                      &Result::input_relative_antisymmetric_norm)
        .def_readonly("discarded_antisymmetric_norm", &Result::discarded_antisymmetric_norm)
        .def_readonly("amplitude_scaling_underflow_count",
                      &Result::amplitude_scaling_underflow_count)
        .def_readonly("density_underflow_entry_count", &Result::density_underflow_entry_count)
        .def_readonly("eigensystem_relative_residual", &Result::eigensystem_relative_residual)
        .def_readonly("eigenvector_orthogonality_error", &Result::eigenvector_orthogonality_error)
        .def_readonly("output_numerical_bytes", &Result::output_numerical_bytes)
        .def_property_readonly("memory", [](const Result& result) { return result.memory; })
        .def_property_readonly("eigensolver_sweeps", [](const Result& result) {
            return result.eigensolver.sweeps;
        })
        .def_property_readonly("eigensolver_rotations", [](const Result& result) {
            return result.eigensolver.rotations;
        })
        .def("density_copy", [](const Result& result) {
            return pair_pno_matrix_copy(result.density, result.domain_dimension,
                                        result.domain_dimension);
        })
        .def("coefficients_copy", [](const Result& result) {
            return pair_pno_matrix_copy(result.coefficients, result.domain_dimension,
                                        result.retained_dimension);
        })
        .def("occupations_copy", [](const Result& result) {
            if (result.domain_dimension > kPairPNODiagnosticMaximumDimension
                || result.occupations.size() != result.domain_dimension) {
                throw std::logic_error("pair PNO diagnostic result has an invalid occupation extent");
            }
            py::array_t<double> copy(static_cast<py::ssize_t>(result.domain_dimension));
            if (!result.occupations.empty()) {
                std::memcpy(copy.mutable_data(), result.occupations.data(),
                            result.occupations.size() * sizeof(double));
            }
            return copy;
        });

    m.def("plan_restricted_pair_pnos", &vibeqc::plan_restricted_pair_pnos,
          py::arg("domain_dimension"),
          "Count-only native numerical-payload admission; no arrays are allocated.");
    m.def(
        "restricted_pair_pnos",
        [](const py::array& amplitudes, Kind pair_kind, double occupation_cutoff,
           std::uint64_t numerical_byte_cap, std::uint64_t max_sweeps,
           double relative_eigensolver_tolerance) {
            // Require existing, contiguous float64 storage. No forcecast or
            // Eigen conversion can allocate an amplitude copy before admission.
            if (!amplitudes.dtype().is(py::dtype::of<double>())
                || amplitudes.ndim() != 2
                || amplitudes.shape(0) != amplitudes.shape(1)
                || !(amplitudes.flags() & py::array::c_style)) {
                throw std::invalid_argument(
                    "pair PNO diagnostic requires a square C-contiguous float64 array");
            }
            if (amplitudes.shape(0)
                    > static_cast<py::ssize_t>(kPairPNODiagnosticMaximumDimension)) {
                throw std::length_error("pair PNO diagnostic dimension is limited to 128");
            }
            if (max_sweeps > kPairPNODiagnosticMaximumSweeps) {
                throw std::length_error("pair PNO diagnostic is limited to 200 eigensolver sweeps");
            }
            const auto n = static_cast<std::size_t>(amplitudes.shape(0));
            const vibeqc::HermitianJacobiOptions options{
                max_sweeps, relative_eigensolver_tolerance};
            py::gil_scoped_release release;
            return vibeqc::restricted_pair_pnos(
                static_cast<const double*>(amplitudes.data()), n * n, n,
                pair_kind, occupation_cutoff, numerical_byte_cap, options);
        },
        py::arg("amplitudes").noconvert(), py::arg("pair_kind"),
        py::arg("occupation_cutoff"), py::arg("numerical_byte_cap"),
        py::arg("max_sweeps"), py::arg("relative_eigensolver_tolerance"),
        "Bounded native pair-PNO diagnostic. All numerical controls are explicit. "
        "The byte cap covers native-owned payload only; input and arrays returned "
        "by the copy methods are caller-owned. Zero cutoff retains the complete "
        "domain; positive cutoff uses the paper's strict greater-than selection.");

    using SemicanonicalPlan = vibeqc::RestrictedPairSemicanonicalMemoryPlan;
    using SemicanonicalResult = vibeqc::RestrictedPairSemicanonicalResult;
    py::class_<SemicanonicalPlan>(m, "RestrictedPairSemicanonicalMemoryPlan")
        .def_readonly("domain_dimension", &SemicanonicalPlan::domain_dimension)
        .def_readonly("selected_dimension", &SemicanonicalPlan::selected_dimension)
        .def_readonly("input_bytes", &SemicanonicalPlan::input_bytes)
        .def_readonly("output_bytes", &SemicanonicalPlan::output_bytes)
        .def_readonly("projection_phase_bytes", &SemicanonicalPlan::projection_phase_bytes)
        .def_readonly("factorization_phase_bytes", &SemicanonicalPlan::factorization_phase_bytes)
        .def_readonly("rotation_phase_bytes", &SemicanonicalPlan::rotation_phase_bytes)
        .def_readonly("validation_phase_bytes", &SemicanonicalPlan::validation_phase_bytes)
        .def_readonly("peak_owned_numerical_bytes", &SemicanonicalPlan::peak_owned_numerical_bytes);
    py::class_<SemicanonicalResult>(m, "RestrictedPairSemicanonicalResult")
        .def_readonly("domain_dimension", &SemicanonicalResult::domain_dimension)
        .def_readonly("selected_dimension", &SemicanonicalResult::selected_dimension)
        .def_readonly("input_orthonormality_error", &SemicanonicalResult::input_orthonormality_error)
        .def_readonly("output_orthonormality_error", &SemicanonicalResult::output_orthonormality_error)
        .def_readonly("reduced_eigensystem_relative_residual",
                      &SemicanonicalResult::reduced_eigensystem_relative_residual)
        .def_readonly("projected_fock_relative_residual",
                      &SemicanonicalResult::projected_fock_relative_residual)
        .def_readonly("subspace_projector_frobenius_error",
                      &SemicanonicalResult::subspace_projector_frobenius_error)
        .def_readonly("full_space_relative_residual", &SemicanonicalResult::full_space_relative_residual)
        .def_readonly("fock_scale_exponent", &SemicanonicalResult::fock_scale_exponent)
        .def_readonly("fock_scaling_underflow_entries", &SemicanonicalResult::fock_scaling_underflow_entries)
        .def_readonly("eigensolver_performed", &SemicanonicalResult::eigensolver_performed)
        .def_property_readonly("eigensolver_sweeps", [](const SemicanonicalResult& result) {
            return result.eigensolver.sweeps;
        })
        .def_property_readonly("memory", [](const SemicanonicalResult& result) { return result.memory; })
        .def("coefficients_copy", [](const SemicanonicalResult& result) {
            return pair_pno_matrix_copy(result.coefficients, result.domain_dimension,
                                        result.selected_dimension);
        })
        .def("energies_copy", [](const SemicanonicalResult& result) {
            if (result.selected_dimension > kPairPNODiagnosticMaximumDimension
                || result.energies.size() != result.selected_dimension) {
                throw std::logic_error("pair semicanonical result has an invalid energy extent");
            }
            py::array_t<double> copy(static_cast<py::ssize_t>(result.selected_dimension));
            if (!result.energies.empty()) {
                std::memcpy(copy.mutable_data(), result.energies.data(),
                            result.energies.size() * sizeof(double));
            }
            return copy;
        });
    m.def("plan_restricted_pair_semicanonicalization",
          &vibeqc::plan_restricted_pair_semicanonicalization,
          py::arg("domain_dimension"), py::arg("selected_dimension"),
          "Count-only phase memory admission for real selected-space recanonicalization.");
    m.def(
        "restricted_pair_semicanonicalize",
        [](const py::array& coefficients, const py::array& virtual_fock,
           double input_orthonormality_tolerance, std::uint64_t numerical_byte_cap,
           std::uint64_t max_sweeps, double relative_eigensolver_tolerance) {
            if (!coefficients.dtype().is(py::dtype::of<double>())
                || !virtual_fock.dtype().is(py::dtype::of<double>())
                || !(coefficients.flags() & py::array::c_style)
                || !(virtual_fock.flags() & py::array::c_style)
                || coefficients.ndim() != 2 || virtual_fock.ndim() != 2
                || virtual_fock.shape(0) != virtual_fock.shape(1)
                || coefficients.shape(0) != virtual_fock.shape(0)
                || coefficients.shape(0) < 1
                || coefficients.shape(1) > coefficients.shape(0)) {
                throw std::invalid_argument(
                    "pair semicanonical diagnostic requires C-contiguous float64 C(n,r) and F(n,n)");
            }
            if (coefficients.shape(0)
                > static_cast<py::ssize_t>(kPairPNODiagnosticMaximumDimension)) {
                throw std::length_error("pair semicanonical diagnostic dimension is limited to 128");
            }
            if (max_sweeps > kPairPNODiagnosticMaximumSweeps) {
                throw std::length_error("pair semicanonical diagnostic is limited to 200 eigensolver sweeps");
            }
            const auto n = static_cast<std::size_t>(coefficients.shape(0));
            const auto r = static_cast<std::size_t>(coefficients.shape(1));
            const vibeqc::HermitianJacobiOptions options{
                max_sweeps, relative_eigensolver_tolerance};
            const auto* c_data = static_cast<const double*>(coefficients.data());
            const auto* f_data = static_cast<const double*>(virtual_fock.data());
            py::gil_scoped_release release;
            return vibeqc::restricted_pair_semicanonicalize(
                c_data, n * r, f_data, n * n, n, r, input_orthonormality_tolerance,
                numerical_byte_cap, options);
        },
        py::arg("coefficients").noconvert(), py::arg("virtual_fock").noconvert(),
        py::arg("input_orthonormality_tolerance"), py::arg("numerical_byte_cap"),
        py::arg("max_sweeps"), py::arg("relative_eigensolver_tolerance"),
        "Bounded native selected-space Fock recanonicalization with explicit memory cap. "
        "An empty selected space is valid; no PNO density or complete result is copied.");
}
