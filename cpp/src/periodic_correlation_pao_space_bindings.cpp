// Tiny diagnostic boundary; no caller-sized input copy or chemistry job.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "vibeqc/periodic_correlation_pao_space.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_pao_space(py::module_& m) {
    using Options = vibeqc::PeriodicCorrelationPAOSpaceOptions;
    using Memory = vibeqc::PeriodicCorrelationPAOSpaceMemoryPlan;
    using Diagnostics = vibeqc::PeriodicCorrelationPAOSpaceDiagnostics;
    using Space = vibeqc::PeriodicCorrelationPAOSpace;
    using Domain = vibeqc::PeriodicCorrelationPAODomain;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Complex = std::complex<double>;

    m.attr("_PERIODIC_CORRELATION_PAO_SPACE_CONTRACT_VERSION") =
        py::int_(vibeqc::kPeriodicCorrelationPAOSpaceContractVersion);
    py::class_<Options>(m, "_PeriodicCorrelationPAOSpaceOptions")
        .def(py::init<>())
        .def_readwrite("rank_absolute_cutoff", &Options::rank_absolute_cutoff)
        .def_readwrite("rank_relative_cutoff", &Options::rank_relative_cutoff)
        .def_readwrite("negative_absolute_tolerance", &Options::negative_absolute_tolerance)
        .def_readwrite("negative_relative_tolerance", &Options::negative_relative_tolerance)
        .def_readwrite("validation_absolute_tolerance", &Options::validation_absolute_tolerance)
        .def_readwrite("validation_relative_tolerance", &Options::validation_relative_tolerance)
        .def_property("max_sweeps", [](const Options& value) { return value.eigensolver.max_sweeps; },
            [](Options& value, std::uint64_t setting) { value.eigensolver.max_sweeps = setting; })
        .def_property("relative_eigensolver_tolerance",
            [](const Options& value) { return value.eigensolver.relative_offdiagonal_tolerance; },
            [](Options& value, double setting) { value.eigensolver.relative_offdiagonal_tolerance = setting; });
    py::class_<Memory>(m, "_PeriodicCorrelationPAOSpaceMemoryPlan")
        .def_readonly("domain_dimension", &Memory::domain_dimension)
        .def_readonly("retained_dimension", &Memory::retained_dimension)
        .def_readonly("borrowed_domain_index_bytes", &Memory::borrowed_domain_index_bytes)
        .def_readonly("borrowed_domain_matrix_bytes", &Memory::borrowed_domain_matrix_bytes)
        .def_readonly("borrowed_domain_bytes", &Memory::borrowed_domain_bytes)
        .def_readonly("overlap_factorization_phase_bytes", &Memory::overlap_factorization_phase_bytes)
        .def_readonly("compact_orthogonalizer_phase_bytes", &Memory::compact_orthogonalizer_phase_bytes)
        .def_readonly("projected_fock_phase_bytes", &Memory::projected_fock_phase_bytes)
        .def_readonly("fock_factorization_phase_bytes", &Memory::fock_factorization_phase_bytes)
        .def_readonly("rotation_phase_bytes", &Memory::rotation_phase_bytes)
        .def_readonly("validation_phase_bytes", &Memory::validation_phase_bytes)
        .def_readonly("peak_owned_numerical_bytes", &Memory::peak_owned_numerical_bytes)
        .def_readonly("output_numerical_bytes", &Memory::output_numerical_bytes);
    py::class_<Diagnostics>(m, "_PeriodicCorrelationPAOSpaceDiagnostics")
        .def_readonly("effective_rank_cutoff", &Diagnostics::effective_rank_cutoff)
        .def_readonly("effective_negative_tolerance", &Diagnostics::effective_negative_tolerance)
        .def_readonly("minimum_overlap_eigenvalue", &Diagnostics::minimum_overlap_eigenvalue)
        .def_readonly("maximum_overlap_eigenvalue", &Diagnostics::maximum_overlap_eigenvalue)
        .def_readonly("negative_overlap_eigenvalue_count", &Diagnostics::negative_overlap_eigenvalue_count)
        .def_readonly("overlap_eigensystem_relative_residual", &Diagnostics::overlap_eigensystem_relative_residual)
        .def_readonly("overlap_eigenvector_orthogonality_error", &Diagnostics::overlap_eigenvector_orthogonality_error)
        .def_readonly("canonical_metric_frobenius_residual", &Diagnostics::canonical_metric_frobenius_residual)
        .def_readonly("maximum_projected_fock_hermitian_defect", &Diagnostics::maximum_projected_fock_hermitian_defect)
        .def_readonly("maximum_projected_fock_hermitization_correction", &Diagnostics::maximum_projected_fock_hermitization_correction)
        .def_readonly("fock_eigensystem_relative_residual", &Diagnostics::fock_eigensystem_relative_residual)
        .def_readonly("fock_eigenvector_orthogonality_error", &Diagnostics::fock_eigenvector_orthogonality_error)
        .def_readonly("final_metric_frobenius_residual", &Diagnostics::final_metric_frobenius_residual)
        .def_readonly("final_projected_fock_relative_residual", &Diagnostics::final_projected_fock_relative_residual)
        .def_readonly("retained_projector_relative_residual", &Diagnostics::retained_projector_relative_residual)
        .def_readonly("fock_scale_exponent", &Diagnostics::fock_scale_exponent)
        .def_readonly("fock_scaling_underflow_components", &Diagnostics::fock_scaling_underflow_components)
        .def_readonly("energy_rescaling_underflow_count", &Diagnostics::energy_rescaling_underflow_count)
        .def_readonly("required_node_memory_bytes", &Diagnostics::required_node_memory_bytes)
        .def_property_readonly("overlap_solver_sweeps", [](const Diagnostics& value) { return value.overlap_eigensolver.sweeps; })
        .def_property_readonly("fock_solver_sweeps", [](const Diagnostics& value) { return value.fock_eigensolver.sweeps; });
    py::class_<Space>(m, "_PeriodicCorrelationPAOSpace")
        .def_property_readonly("contract_version", &Space::contract_version)
        .def_property_readonly("usable", &Space::usable)
        .def_property_readonly("domain_dimension", &Space::domain_dimension)
        .def_property_readonly("retained_dimension", &Space::retained_dimension)
        .def_property_readonly("state", [](const Space& value) -> std::shared_ptr<State> {
            return std::const_pointer_cast<State>(value.state_handle());
        })
        .def_property_readonly("state_identity_sha256", &Space::state_identity_sha256)
        .def_property_readonly("domain_index_sha256", &Space::domain_index_sha256)
        .def_property_readonly("pao_domain_identity_sha256", &Space::pao_domain_identity_sha256)
        .def_property_readonly("payload_sha256", &Space::payload_sha256)
        .def_property_readonly("pao_space_identity_sha256", &Space::pao_space_identity_sha256)
        .def_property_readonly("allocation_identity", &Space::allocation_identity)
        .def_property_readonly("memory", [](const Space& value) { return value.memory(); })
        .def_property_readonly("options", [](const Space& value) { return value.options(); })
        .def_property_readonly("diagnostics", [](const Space& value) { return value.diagnostics(); })
        .def("coefficient", &Space::coefficient, py::arg("row"), py::arg("orbital"))
        .def("energy", &Space::energy, py::arg("orbital"))
        .def("overlap_eigenvalue", &Space::overlap_eigenvalue, py::arg("index"))
        .def("coefficients_copy", [](const Space& value) {
            if (value.domain_dimension() > 32U) throw std::length_error("PAO space diagnostic copy is limited to 32 columns");
            const auto* source = value.coefficients_data();
            py::array_t<Complex> answer({static_cast<py::ssize_t>(value.domain_dimension()),
                                          static_cast<py::ssize_t>(value.retained_dimension())});
            if (answer.size() != 0) std::memcpy(answer.mutable_data(), source, answer.size() * sizeof(Complex));
            return answer;
        })
        .def("energies_copy", [](const Space& value) {
            if (value.retained_dimension() > 32U) throw std::length_error("PAO space diagnostic copy is limited to 32 orbitals");
            const auto* source = value.energies_data();
            py::array_t<double> answer(static_cast<py::ssize_t>(value.retained_dimension()));
            if (answer.size() != 0) std::memcpy(answer.mutable_data(), source, answer.size() * sizeof(double));
            return answer;
        })
        .def("overlap_eigenvalues_copy", [](const Space& value) {
            if (value.domain_dimension() > 32U) throw std::length_error("PAO space diagnostic copy is limited to 32 columns");
            const auto* source = value.overlap_eigenvalues_data();
            py::array_t<double> answer(static_cast<py::ssize_t>(value.domain_dimension()));
            if (answer.size() != 0) std::memcpy(answer.mutable_data(), source, answer.size() * sizeof(double));
            return answer;
        });
    m.def("_plan_periodic_correlation_pao_space", &vibeqc::plan_periodic_correlation_pao_space,
          py::arg("domain_dimension"), py::arg("retained_dimension"));
    m.def("_make_periodic_correlation_pao_space",
        [](const Reference& reference, const Domain& domain, std::uint64_t cap, const Options& options) {
            if (domain.domain_dimension() > 32U || options.eigensolver.max_sweeps > 200U) {
                throw std::length_error("PAO space diagnostic is limited to 32 columns and 200 sweeps");
            }
            py::gil_scoped_release release;
            return vibeqc::make_periodic_correlation_pao_space(reference, domain, cap, options);
        }, py::arg("reference"), py::arg("domain"), py::arg("owned_numerical_byte_cap"), py::arg("options"),
        "Compact complex PAO orthogonalization/semicanonicalization. Borrows the immutable domain "
        "only during this call; returned coefficients carry its identity, not a copied S/F payload.");
}
