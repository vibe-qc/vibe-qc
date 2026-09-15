// Internal diagnostic binding for the native periodic RHF snapshot contract.
// Production correlation code consumes the C++ state directly.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <utility>

#include "vibeqc/periodic_mean_field_state.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_mean_field_state(py::module_& m) {
    using State = vibeqc::PeriodicRestrictedMeanFieldState;

    m.attr("_PERIODIC_RESTRICTED_MEAN_FIELD_STATE_CONTRACT_VERSION") =
        py::int_(vibeqc::kPeriodicRestrictedMeanFieldStateContractVersion);
    m.attr("_PERIODIC_MEAN_FIELD_STATE_DIGEST_VERSION") =
        py::int_(vibeqc::kPeriodicMeanFieldStateDigestVersion);
    m.attr("_PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION") =
        py::int_(vibeqc::kPeriodicMeanFieldValidationToleranceVersion);

    py::enum_<vibeqc::PeriodicMeanFieldReferenceKind>(
        m, "_PeriodicMeanFieldReferenceKind")
        .value(
            "RESTRICTED_HARTREE_FOCK",
            vibeqc::PeriodicMeanFieldReferenceKind::RestrictedHartreeFock);
    py::enum_<vibeqc::PeriodicMeanFieldNormalizationConvention>(
        m, "_PeriodicMeanFieldNormalizationConvention")
        .value(
            "UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS",
            vibeqc::PeriodicMeanFieldNormalizationConvention::
                UnnormalizedAoBlochSumsUniformFullBzWeights);

    py::class_<vibeqc::PeriodicMeanFieldValidationTolerances>(
        m, "_PeriodicMeanFieldValidationTolerances")
        .def_readonly(
            "version",
            &vibeqc::PeriodicMeanFieldValidationTolerances::version)
        .def_readonly(
            "matrix_absolute",
            &vibeqc::PeriodicMeanFieldValidationTolerances::matrix_absolute)
        .def_readonly(
            "matrix_relative",
            &vibeqc::PeriodicMeanFieldValidationTolerances::matrix_relative)
        .def_readonly(
            "coordinate_absolute",
            &vibeqc::PeriodicMeanFieldValidationTolerances::
                coordinate_absolute)
        .def_readonly(
            "coordinate_relative",
            &vibeqc::PeriodicMeanFieldValidationTolerances::
                coordinate_relative)
        .def_readonly(
            "scalar_absolute",
            &vibeqc::PeriodicMeanFieldValidationTolerances::scalar_absolute)
        .def_readonly(
            "scalar_relative",
            &vibeqc::PeriodicMeanFieldValidationTolerances::scalar_relative)
        .def_readonly(
            "reciprocal_lattice_relative_volume_floor",
            &vibeqc::PeriodicMeanFieldValidationTolerances::
                reciprocal_lattice_relative_volume_floor)
        .def_readonly(
            "overlap_eigenvalue_relative_floor",
            &vibeqc::PeriodicMeanFieldValidationTolerances::
                overlap_eigenvalue_relative_floor);

    py::class_<vibeqc::PeriodicMeanFieldValidationDiagnostics>(
        m, "_PeriodicMeanFieldValidationDiagnostics")
        .def_readonly(
            "maximum_kpoint_cartesian_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_kpoint_cartesian_residual)
        .def_readonly(
            "maximum_weight_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_weight_residual)
        .def_readonly(
            "maximum_overlap_hermiticity_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_overlap_hermiticity_residual)
        .def_readonly(
            "maximum_fock_hermiticity_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_fock_hermiticity_residual)
        .def_readonly(
            "minimum_overlap_eigenvalue",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                minimum_overlap_eigenvalue)
        .def_readonly(
            "maximum_metric_orthonormality_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_metric_orthonormality_residual)
        .def_readonly(
            "maximum_roothaan_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_roothaan_residual)
        .def_readonly(
            "maximum_energy_order_violation",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_energy_order_violation)
        .def_readonly(
            "maximum_occupation_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                maximum_occupation_residual)
        .def_readonly(
            "electron_count_residual",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                electron_count_residual)
        .def_readonly(
            "band_gap_hartree",
            &vibeqc::PeriodicMeanFieldValidationDiagnostics::
                band_gap_hartree);

    py::class_<vibeqc::PeriodicRestrictedMeanFieldInput>(
        m, "_PeriodicRestrictedMeanFieldInput")
        .def(py::init<>())
        .def_readwrite(
            "calculation_identity",
            &vibeqc::PeriodicRestrictedMeanFieldInput::calculation_identity)
        .def_readwrite(
            "reference_kind",
            &vibeqc::PeriodicRestrictedMeanFieldInput::reference_kind)
        .def_readwrite(
            "normalization",
            &vibeqc::PeriodicRestrictedMeanFieldInput::normalization)
        .def_readwrite(
            "periodic_dimension",
            &vibeqc::PeriodicRestrictedMeanFieldInput::periodic_dimension)
        .def_readwrite(
            "mesh", &vibeqc::PeriodicRestrictedMeanFieldInput::mesh)
        .def_readwrite(
            "is_shift", &vibeqc::PeriodicRestrictedMeanFieldInput::is_shift)
        .def_readwrite(
            "reciprocal_lattice",
            &vibeqc::PeriodicRestrictedMeanFieldInput::reciprocal_lattice)
        .def_readwrite(
            "converged",
            &vibeqc::PeriodicRestrictedMeanFieldInput::converged)
        .def_readwrite(
            "symmetry_reduced_input",
            &vibeqc::PeriodicRestrictedMeanFieldInput::
                symmetry_reduced_input)
        .def_readwrite(
            "symmetry_reconstructed_input",
            &vibeqc::PeriodicRestrictedMeanFieldInput::
                symmetry_reconstructed_input)
        .def_readwrite(
            "n_basis", &vibeqc::PeriodicRestrictedMeanFieldInput::n_basis)
        .def_readwrite(
            "n_effective_orbitals",
            &vibeqc::PeriodicRestrictedMeanFieldInput::
                n_effective_orbitals)
        .def_readwrite(
            "electrons_per_cell",
            &vibeqc::PeriodicRestrictedMeanFieldInput::electrons_per_cell)
        .def_readwrite(
            "reference_energy_per_cell",
            &vibeqc::PeriodicRestrictedMeanFieldInput::
                reference_energy_per_cell)
        .def_readwrite(
            "minimum_band_gap_hartree",
            &vibeqc::PeriodicRestrictedMeanFieldInput::
                minimum_band_gap_hartree)
        .def_readwrite(
            "validation_tolerance_version",
            &vibeqc::PeriodicRestrictedMeanFieldInput::
                validation_tolerance_version)
        .def(
            "add_kpoint",
            &vibeqc::PeriodicRestrictedMeanFieldInput::add_kpoint,
            py::arg("k_cartesian"),
            py::arg("weight"),
            py::arg("overlap"),
            py::arg("fock"),
            py::arg("coefficients"),
            py::arg("orbital_energies"),
            py::arg("occupations"),
            py::arg("frozen_core_mask"),
            py::arg("correlated_occupied_mask"),
            py::arg("virtual_mask"))
        .def_property_readonly(
            "kpoint_count",
            &vibeqc::PeriodicRestrictedMeanFieldInput::kpoint_count);

    py::class_<State, std::shared_ptr<State>>(
        m, "_PeriodicRestrictedMeanFieldState")
        .def_property_readonly(
            "contract_version",
            &vibeqc::PeriodicRestrictedMeanFieldState::contract_version)
        .def_property_readonly(
            "converged",
            &vibeqc::PeriodicRestrictedMeanFieldState::converged)
        .def_property_readonly(
            "calculation_identity",
            &vibeqc::PeriodicRestrictedMeanFieldState::calculation_identity)
        .def_property_readonly(
            "digest_version",
            &vibeqc::PeriodicRestrictedMeanFieldState::digest_version)
        .def_property_readonly(
            "numerical_payload_sha256",
            &vibeqc::PeriodicRestrictedMeanFieldState::
                numerical_payload_sha256)
        .def_property_readonly(
            "state_identity_sha256",
            &vibeqc::PeriodicRestrictedMeanFieldState::state_identity_sha256)
        .def_property_readonly(
            "reference_kind",
            &vibeqc::PeriodicRestrictedMeanFieldState::reference_kind)
        .def_property_readonly(
            "normalization",
            &vibeqc::PeriodicRestrictedMeanFieldState::normalization)
        .def_property_readonly(
            "periodic_dimension",
            &vibeqc::PeriodicRestrictedMeanFieldState::periodic_dimension)
        .def_property_readonly(
            "mesh", &vibeqc::PeriodicRestrictedMeanFieldState::mesh)
        .def_property_readonly(
            "is_shift", &vibeqc::PeriodicRestrictedMeanFieldState::is_shift)
        .def_property_readonly(
            "reciprocal_lattice",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state) {
                return state.reciprocal_lattice();
            })
        .def_property_readonly(
            "n_kpoints",
            &vibeqc::PeriodicRestrictedMeanFieldState::n_kpoints)
        .def_property_readonly(
            "n_basis", &vibeqc::PeriodicRestrictedMeanFieldState::n_basis)
        .def_property_readonly(
            "n_effective_orbitals",
            &vibeqc::PeriodicRestrictedMeanFieldState::n_effective_orbitals)
        .def_property_readonly(
            "n_frozen_core",
            &vibeqc::PeriodicRestrictedMeanFieldState::n_frozen_core)
        .def_property_readonly(
            "n_correlated_occupied",
            &vibeqc::PeriodicRestrictedMeanFieldState::
                n_correlated_occupied)
        .def_property_readonly(
            "n_virtual",
            &vibeqc::PeriodicRestrictedMeanFieldState::n_virtual)
        .def_property_readonly(
            "electrons_per_cell",
            &vibeqc::PeriodicRestrictedMeanFieldState::electrons_per_cell)
        .def_property_readonly(
            "reference_energy_per_cell",
            &vibeqc::PeriodicRestrictedMeanFieldState::
                reference_energy_per_cell)
        .def_property_readonly(
            "requested_minimum_band_gap_hartree",
            &vibeqc::PeriodicRestrictedMeanFieldState::
                requested_minimum_band_gap_hartree)
        .def_property_readonly(
            "band_gap_hartree",
            &vibeqc::PeriodicRestrictedMeanFieldState::band_gap_hartree)
        .def_property_readonly(
            "uniform_weight",
            &vibeqc::PeriodicRestrictedMeanFieldState::uniform_weight)
        .def_property_readonly(
            "resident_bytes",
            &vibeqc::PeriodicRestrictedMeanFieldState::resident_bytes)
        .def_property_readonly(
            "validation_tolerances",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state) {
                return state.validation_tolerances();
            })
        .def_property_readonly(
            "diagnostics",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state) {
                return state.diagnostics();
            })
        .def(
            "kpoint_cartesian",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) {
                return state.kpoint_cartesian(index);
            },
            py::arg("index"))
        .def(
            "weight",
            &vibeqc::PeriodicRestrictedMeanFieldState::weight,
            py::arg("index"))
        .def(
            "overlap",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) { return state.overlap(index); },
            py::arg("index"))
        .def(
            "fock",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) { return state.fock(index); },
            py::arg("index"))
        .def(
            "coefficients",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) { return state.coefficients(index); },
            py::arg("index"))
        .def(
            "orbital_energies",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) {
                return state.orbital_energies(index);
            },
            py::arg("index"))
        .def(
            "occupations",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) { return state.occupations(index); },
            py::arg("index"))
        .def(
            "frozen_core_mask",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) {
                return state.frozen_core_mask(index);
            },
            py::arg("index"))
        .def(
            "correlated_occupied_mask",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) {
                return state.correlated_occupied_mask(index);
            },
            py::arg("index"))
        .def(
            "virtual_mask",
            [](const vibeqc::PeriodicRestrictedMeanFieldState& state,
               std::size_t index) { return state.virtual_mask(index); },
            py::arg("index"));

    m.def(
        "_make_periodic_restricted_mean_field_state",
        [](vibeqc::PeriodicRestrictedMeanFieldInput& input) {
            return std::const_pointer_cast<State>(
                vibeqc::make_periodic_restricted_mean_field_state(
                    std::move(input)));
        },
        py::arg("input"));
    m.def(
        "_estimate_periodic_restricted_mean_field_resident_bytes",
        &vibeqc::estimate_periodic_restricted_mean_field_resident_bytes,
        py::arg("mesh"),
        py::arg("n_basis"),
        py::arg("n_effective_orbitals"));
}
