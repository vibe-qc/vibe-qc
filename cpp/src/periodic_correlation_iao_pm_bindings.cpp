// Tiny numerical diagnostic only; no localization optimizer or public driver.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_correlation_iao_pm.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_iao_pm(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationIAOPMOptions;
    using Caps = PeriodicCorrelationIAOPMCaps;
    using Memory = PeriodicCorrelationIAOPMMemoryPlan;
    using Result = PeriodicCorrelationIAOPMResult;
    using Z = std::complex<double>;
    py::class_<Options>(m, "_PeriodicCorrelationIAOPMOptions")
        .def(py::init<>())
        .def_readwrite("gauge_unitarity_tolerance", &Options::gauge_unitarity_tolerance)
        .def_readwrite("charge_normalization_tolerance", &Options::charge_normalization_tolerance);
    py::class_<Caps>(m, "_PeriodicCorrelationIAOPMCaps")
        .def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_work_units", &Caps::maximum_work_units);
    auto memory = py::class_<Memory>(m, "_PeriodicCorrelationIAOPMMemoryPlan");
#define IAO_PM_MEMORY(field) memory.def_readonly(#field, &Memory::field)
    IAO_PM_MEMORY(n_points); IAO_PM_MEMORY(n_minimal); IAO_PM_MEMORY(n_active); IAO_PM_MEMORY(n_occupied);
    IAO_PM_MEMORY(gradient_bytes); IAO_PM_MEMORY(gradient_compensation_bytes);
    IAO_PM_MEMORY(cell_workspace_bytes); IAO_PM_MEMORY(active_index_bytes);
    IAO_PM_MEMORY(peak_owned_numerical_bytes); IAO_PM_MEMORY(borrowed_gauge_bytes);
    IAO_PM_MEMORY(borrowed_owner_pointer_bytes); IAO_PM_MEMORY(live_iao_numerical_bytes);
    IAO_PM_MEMORY(required_node_memory_bytes); IAO_PM_MEMORY(work_units);
#undef IAO_PM_MEMORY
    py::class_<Result>(m, "_PeriodicCorrelationIAOPMResult")
        .def_readonly("memory", &Result::memory)
        .def_readonly("objective", &Result::objective)
        .def_readonly("maximum_charge_normalization_residual", &Result::maximum_charge_normalization_residual)
        .def_readonly("maximum_charge_imaginary_magnitude", &Result::maximum_charge_imaginary_magnitude)
        .def_readonly("maximum_unitarity_residual", &Result::maximum_unitarity_residual)
        .def_readonly("gradient_frobenius_norm", &Result::gradient_frobenius_norm)
        .def("gradient_copy", [](const Result& r) {
            const auto k = r.memory.n_points, o = r.memory.n_active;
            if (k > 64 || o > 8 || r.gradient.size() != k * o * o)
                throw std::length_error("IAO PM diagnostic gradient exceeds tiny extent");
            py::array_t<Z> out({static_cast<py::ssize_t>(k), static_cast<py::ssize_t>(o), static_cast<py::ssize_t>(o)});
            std::memcpy(out.mutable_data(), r.gradient.data(), r.memory.gradient_bytes); return out;
        });
    m.def("_plan_periodic_correlation_iao_pm", &plan_periodic_correlation_iao_pm,
        py::arg("reference"), py::arg("n_minimal"));
    m.def("_evaluate_periodic_correlation_iao_pm", [](
        const PeriodicCorrelationAdmittedReference& reference, const py::list& points,
        const py::array& gauges, const Options& options, const Caps& caps) {
        const auto& state = reference.state();
        const auto nk = state.n_kpoints();
        const auto o = state.n_correlated_occupied();
        if (nk > 64 || o > 8 || state.n_basis() > 32 || points.size() != nk
            || caps.maximum_owned_numerical_bytes > (4U << 20) || caps.maximum_work_units > 1000000000ULL)
            throw std::length_error("IAO PM diagnostic exceeds tiny dimensions or work limits");
        if (!gauges.dtype().is(py::dtype::of<Z>()) || gauges.ndim() != 3 || !(gauges.flags() & py::array::c_style)
            || gauges.shape(0) != static_cast<py::ssize_t>(nk)
            || gauges.shape(1) != static_cast<py::ssize_t>(o) || gauges.shape(2) != static_cast<py::ssize_t>(o))
            throw std::invalid_argument("IAO PM diagnostic requires existing contiguous complex128 [Nk,active,active] gauges");
        std::vector<const PeriodicCorrelationBlochIAO*> owners;
        owners.reserve(nk);
        for (const auto value : points) owners.push_back(&value.cast<const PeriodicCorrelationBlochIAO&>());
        // Keep the GIL: immutable native owners are safe, but Python controls
        // and the raw gauge view must not change midway through this diagnostic.
        return evaluate_periodic_correlation_iao_pm(reference, owners.data(), owners.size(),
            static_cast<const Z*>(gauges.data()), gauges.size(), options, caps);
    }, py::arg("reference"), py::arg("points"), py::arg("gauges").noconvert(),
       py::arg("options"), py::arg("caps"));
}
