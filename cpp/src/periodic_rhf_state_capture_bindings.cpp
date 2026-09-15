// Internal diagnostic bindings for the experimental native RHF capture gate.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vibeqc/periodic_rhf_state_capture.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_rhf_state_capture(py::module_& m) {
    py::class_<vibeqc::PeriodicRHFStateCaptureRequest>(
        m, "_PeriodicRHFStateCaptureRequest")
        .def(py::init<>())
        .def_readwrite(
            "calculation_identity",
            &vibeqc::PeriodicRHFStateCaptureRequest::calculation_identity)
        .def_readwrite(
            "maximum_retained_numerical_payload_bytes",
            &vibeqc::PeriodicRHFStateCaptureRequest::
                maximum_retained_numerical_payload_bytes)
        .def_readwrite(
            "minimum_band_gap_hartree",
            &vibeqc::PeriodicRHFStateCaptureRequest::
                minimum_band_gap_hartree)
        .def_readwrite(
            "frozen_core_mask_per_k",
            &vibeqc::PeriodicRHFStateCaptureRequest::
                frozen_core_mask_per_k);

    py::class_<vibeqc::PeriodicRHFStateCapturePreflight>(
        m, "_PeriodicRHFStateCapturePreflight")
        .def_readonly(
            "retained_numerical_payload_bytes",
            &vibeqc::PeriodicRHFStateCapturePreflight::
                retained_numerical_payload_bytes)
        .def_readonly(
            "n_frozen_core",
            &vibeqc::PeriodicRHFStateCapturePreflight::n_frozen_core)
        .def_readonly(
            "n_correlated_occupied",
            &vibeqc::PeriodicRHFStateCapturePreflight::
                n_correlated_occupied)
        .def_readonly(
            "n_virtual",
            &vibeqc::PeriodicRHFStateCapturePreflight::n_virtual);

    py::class_<vibeqc::PeriodicRHFPhysicalDensityDiagnostics>(
        m, "_PeriodicRHFPhysicalDensityDiagnostics")
        .def_readonly(
            "commutator_frobenius",
            &vibeqc::PeriodicRHFPhysicalDensityDiagnostics::
                commutator_frobenius)
        .def_readonly(
            "commutator_roundoff_allowance",
            &vibeqc::PeriodicRHFPhysicalDensityDiagnostics::
                commutator_roundoff_allowance)
        .def_readonly(
            "metric_idempotency_frobenius",
            &vibeqc::PeriodicRHFPhysicalDensityDiagnostics::
                metric_idempotency_frobenius)
        .def_readonly(
            "metric_idempotency_relative",
            &vibeqc::PeriodicRHFPhysicalDensityDiagnostics::
                metric_idempotency_relative);

    py::class_<vibeqc::PeriodicRHFDensityFixedPointDiagnostics>(
        m, "_PeriodicRHFDensityFixedPointDiagnostics")
        .def_readonly(
            "frobenius",
            &vibeqc::PeriodicRHFDensityFixedPointDiagnostics::frobenius)
        .def_readonly(
            "relative",
            &vibeqc::PeriodicRHFDensityFixedPointDiagnostics::relative);

    m.def(
        "_preflight_periodic_rhf_state_capture",
        &vibeqc::preflight_periodic_rhf_state_capture,
        py::arg("request"),
        py::arg("periodic_dimension"),
        py::arg("mesh"),
        py::arg("is_shift"),
        py::arg("reciprocal_lattice"),
        py::arg("kpoints"),
        py::arg("weights"),
        py::arg("symmetry_reduced_or_reconstructed"),
        py::arg("n_basis"),
        py::arg("electrons_per_cell"));
    m.def(
        "_periodic_rhf_physical_density_diagnostics",
        &vibeqc::periodic_rhf_physical_density_diagnostics,
        py::arg("overlap"),
        py::arg("physical_fock"),
        py::arg("spin_summed_density"));
    m.def(
        "_periodic_rhf_density_fixed_point_diagnostics",
        &vibeqc::periodic_rhf_density_fixed_point_diagnostics,
        py::arg("spin_summed_density"),
        py::arg("coefficients"),
        py::arg("n_occupied"));
}
