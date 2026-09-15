// Included by bindings.cpp. Bounded tiny copied-density diagnostic only.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "vibeqc/periodic_gaussian_fock.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_fock(py::module_& m) {
    using Context=vibeqc::PeriodicGaussianSourceContext;
    using Config=vibeqc::PeriodicGaussianFockConfig;
    using Caps=vibeqc::PeriodicGaussianFockCaps;
    using Live=vibeqc::PeriodicGaussianMetricLiveInventory;
    using Plan=vibeqc::PeriodicGaussianFockPlan;
    using Result=vibeqc::PeriodicGaussianFockResult;
    using Receipt=vibeqc::PeriodicGaussianFockReceipt;
    using Diagnostic=vibeqc::PeriodicGaussianFockMatrixDiagnostics;
    using Progress=vibeqc::PeriodicGaussianFockProgress;
    using Stage=vibeqc::PeriodicGaussianFockStage;
    using Complex=std::complex<double>;
    py::class_<Config>(m,"_PeriodicGaussianFockConfig").def(py::init<>())
        .def_readwrite("auxiliary_block",&Config::auxiliary_block)
        .def_readwrite("ao_column_block",&Config::ao_column_block)
        .def_readwrite("source_caps",&Config::source_caps)
        .def_readwrite("metric",&Config::metric)
        .def_readwrite("metric_caps",&Config::metric_caps)
        .def_readwrite("tile",&Config::tile)
        .def_readwrite("tile_caps",&Config::tile_caps);
    py::class_<Caps>(m,"_PeriodicGaussianFockCaps").def(py::init<>())
        .def_readwrite("resources",&Caps::resources)
        .def_readwrite("maximum_tile_calls",&Caps::maximum_tile_calls)
        .def_readwrite("maximum_progress_callbacks",&Caps::maximum_progress_callbacks)
        .def_readwrite("maximum_image_candidate_evaluations",&Caps::maximum_image_candidate_evaluations);
    py::class_<Plan>(m,"_PeriodicGaussianFockPlan")
        .def_readonly("n_kpoints",&Plan::n_kpoints)
        .def_readonly("n_basis",&Plan::n_basis)
        .def_readonly("n_auxiliary",&Plan::n_auxiliary)
        .def_readonly("auxiliary_block",&Plan::auxiliary_block)
        .def_readonly("ao_column_block",&Plan::ao_column_block)
        .def_readonly("auxiliary_block_count",&Plan::auxiliary_block_count)
        .def_readonly("ao_column_block_count",&Plan::ao_column_block_count)
        .def_readonly("density_element_count",&Plan::density_element_count)
        .def_readonly("borrowed_density_bytes",&Plan::borrowed_density_bytes)
        .def_readonly("response_bytes",&Plan::response_bytes)
        .def_readonly("response_compensation_bytes",&Plan::response_compensation_bytes)
        .def_readonly("hartree_vector_bytes",&Plan::hartree_vector_bytes)
        .def_readonly("exchange_double_panel_bytes",&Plan::exchange_double_panel_bytes)
        .def_readonly("resident_whitener_bytes",&Plan::resident_whitener_bytes)
        .def_readonly("metric_phase_owned_numeric_upper_bound",&Plan::metric_phase_owned_numeric_upper_bound)
        .def_readonly("tile_phase_owned_numeric_upper_bound",&Plan::tile_phase_owned_numeric_upper_bound)
        .def_readonly("owned_numeric_upper_bound",&Plan::owned_numeric_upper_bound)
        .def_readonly("borrowed_basis_active_numeric_bytes",&Plan::borrowed_basis_active_numeric_bytes)
        .def_readonly("macro_fixed_object_bytes",&Plan::macro_fixed_object_bytes)
        .def_readonly("maximum_leaf_fixed_inventory_bytes",&Plan::maximum_leaf_fixed_inventory_bytes)
        .def_readonly("per_replica_inventoried_bytes",&Plan::per_replica_inventoried_bytes)
        .def_readonly("node_inventoried_bytes",&Plan::node_inventoried_bytes)
        .def_readonly("hartree_tile_calls",&Plan::hartree_tile_calls)
        .def_readonly("exchange_tile_calls",&Plan::exchange_tile_calls)
        .def_readonly("total_tile_calls",&Plan::total_tile_calls)
        .def_readonly("progress_callback_upper_bound",&Plan::progress_callback_upper_bound)
        .def_readonly("driver_contraction_terms",&Plan::driver_contraction_terms)
        .def_readonly("reciprocal_candidate_evaluations_upper_bound",&Plan::reciprocal_candidate_evaluations_upper_bound)
        .def_readonly("image_candidate_evaluations_upper_bound",&Plan::image_candidate_evaluations_upper_bound)
        .def_readonly("driver_work_units_upper_bound",&Plan::driver_work_units_upper_bound)
        .def_readonly("work_units_upper_bound",&Plan::work_units_upper_bound)
        .def_property_readonly("config",[](const Plan& p) { return Config(p.config); })
        .def_property_readonly("live",[](const Plan& p) { return Live(p.live); })
        .def_property_readonly("caps",[](const Plan& p) { return Caps(p.caps); })
        .def_property_readonly("plan_identity_sha256",&Plan::plan_identity_sha256);
    py::class_<Diagnostic>(m,"_PeriodicGaussianFockMatrixDiagnostics")
        .def_readonly("maximum_magnitude",&Diagnostic::maximum_magnitude)
        .def_readonly("maximum_hermiticity_residual",&Diagnostic::maximum_hermiticity_residual)
        .def_readonly("relative_hermiticity_residual",&Diagnostic::relative_hermiticity_residual)
        .def_readonly("maximum_time_reversal_residual",&Diagnostic::maximum_time_reversal_residual)
        .def_readonly("relative_time_reversal_residual",&Diagnostic::relative_time_reversal_residual)
        .def_readonly("maximum_diagonal_imaginary",&Diagnostic::maximum_diagonal_imaginary);
    py::class_<Receipt>(m,"_PeriodicGaussianFockReceipt")
        .def_readonly("completed_q_count",&Receipt::completed_q_count)
        .def_readonly("completed_tile_count",&Receipt::completed_tile_count)
        .def_readonly("progress_callback_count",&Receipt::progress_callback_count)
        .def_readonly("source_factory_candidate_evaluations",&Receipt::source_factory_candidate_evaluations)
        .def_readonly("reciprocal_candidate_evaluations",&Receipt::reciprocal_candidate_evaluations)
        .def_readonly("image_candidate_evaluations",&Receipt::image_candidate_evaluations)
        .def_readonly("charged_work_units_upper_bound",&Receipt::charged_work_units_upper_bound)
        .def_readonly("maximum_observed_owned_numeric_bytes",&Receipt::maximum_observed_owned_numeric_bytes)
        .def_readonly("maximum_observed_per_replica_inventoried_bytes",&Receipt::maximum_observed_per_replica_inventoried_bytes);
    py::enum_<Stage>(m,"_PeriodicGaussianFockStage")
        .value("BEGIN",Stage::Begin)
        .value("SOURCE",Stage::Source)
        .value("METRIC",Stage::Metric)
        .value("WHITENING",Stage::Whitening)
        .value("HARTREE_DENSITY",Stage::HartreeDensity)
        .value("HARTREE_APPLY",Stage::HartreeApply)
        .value("EXCHANGE",Stage::Exchange)
        .value("QCOMPLETE",Stage::QComplete)
        .value("COMPLETE",Stage::Complete);
    py::class_<Progress>(m,"_PeriodicGaussianFockProgress")
        .def_readonly("stage",&Progress::stage)
        .def_readonly("q_index",&Progress::q_index)
        .def_readonly("auxiliary_begin",&Progress::auxiliary_begin)
        .def_readonly("k_bra_index",&Progress::k_bra_index)
        .def_readonly("k_ket_index",&Progress::k_ket_index)
        .def_readonly("sigma_begin",&Progress::sigma_begin)
        .def_readonly("completed_q_count",&Progress::completed_q_count)
        .def_readonly("completed_tile_count",&Progress::completed_tile_count)
        .def_readonly("charged_work_units_upper_bound",&Progress::charged_work_units_upper_bound);
    py::class_<Result>(m,"_PeriodicGaussianFockResult")
        .def_property_readonly("context",&Result::context_handle)
        .def_property_readonly("plan",[](const Result& r) { return Plan(r.plan()); })
        .def_property_readonly("receipt",[](const Result& r) { return Receipt(r.receipt()); })
        .def_property_readonly("density_diagnostics",[](const Result& r) { return Diagnostic(r.density_diagnostics()); })
        .def_property_readonly("response_diagnostics",[](const Result& r) { return Diagnostic(r.response_diagnostics()); })
        .def_property_readonly("density_identity_sha256",&Result::density_identity_sha256)
        .def_property_readonly("consumed_factor_identity_sha256",&Result::consumed_factor_identity_sha256)
        .def_property_readonly("payload_identity_sha256",&Result::payload_identity_sha256)
        .def_property_readonly("matrix",[](const Result& r) {
            if(!r.context_handle() || r.plan().n_kpoints>8 || r.plan().n_basis>4
                || r.matrix_row_major().size()!=r.plan().density_element_count) {
                throw std::invalid_argument("Gaussian Fock diagnostic result is moved or outside tiny shape");
            }
            py::array_t<Complex> a({static_cast<py::ssize_t>(r.plan().n_kpoints),
                static_cast<py::ssize_t>(r.plan().n_basis),static_cast<py::ssize_t>(r.plan().n_basis)});
            std::memcpy(a.mutable_data(),r.matrix_row_major().data(),r.plan().response_bytes);
            return a;
        });
    const auto diagnostic_plan=[](const Context& context,const Config& config,const Live& live,const Caps& caps) {
        const auto n=context.inventory().ao.function_count,nk=static_cast<std::uint64_t>(context.mesh().size());
        if(n>4 || nk>8 || context.inventory().auxiliary.function_count>4
            || config.source_caps.maximum_candidates_per_source>65536
            || config.source_caps.maximum_candidate_evaluations>400000
            || config.tile_caps.maximum_image_candidates>65536) {
            throw std::length_error("Gaussian Fock diagnostic exceeds tiny shape/source caps");
        }
        auto copied_live=live;
        const auto bytes=16U*nk*n*n;
        if(bytes>std::numeric_limits<std::uint64_t>::max()-copied_live.other_retained_bytes_per_replica) {
            throw std::overflow_error("Gaussian Fock diagnostic density copy inventory overflow");
        }
        // The native borrowed-D inventory covers one buffer. Charge the
        // second simultaneously live caller/snapshot buffer exactly once.
        copied_live.other_retained_bytes_per_replica+=bytes;
        auto plan=vibeqc::plan_periodic_gaussian_fock(context,config,copied_live,caps);
        if(plan.total_tile_calls>4096) throw std::length_error("Gaussian Fock diagnostic exceeds4096 tile calls");
        return plan;
    };
    m.def("_plan_periodic_gaussian_fock",diagnostic_plan,
        py::arg("context"),py::arg("config"),py::arg("live"),py::arg("caps"),
        "Diagnostic plan includes the second copied/caller density buffer in other_retained.");
    const auto run=[diagnostic_plan](std::shared_ptr<const Context> context,const vibeqc::BasisSet& ao,
        const vibeqc::BasisSet& auxiliary,py::array density,const Config& input_config,
        const Live& input_live,const Caps& input_caps,py::object progress,bool mutate_snapshot) {
        if(!context) throw std::invalid_argument("Gaussian Fock diagnostic requires a native context");
        const auto config=Config(input_config); const auto caps=Caps(input_caps);
        const auto plan=diagnostic_plan(*context,config,input_live,caps);
        const auto live=Live(plan.live);
        if(!density.dtype().is(py::dtype::of<Complex>()) || !(density.flags()&py::array::c_style)
            || density.ndim()!=3 || density.shape(0)!=static_cast<py::ssize_t>(plan.n_kpoints)
            || density.shape(1)!=static_cast<py::ssize_t>(plan.n_basis)
            || density.shape(2)!=static_cast<py::ssize_t>(plan.n_basis)) {
            throw std::invalid_argument("Gaussian Fock diagnostic density must be contiguous complex128 (Nk,n,n)");
        }
        if(!progress.is_none() && !PyCallable_Check(progress.ptr())) throw std::invalid_argument("Gaussian Fock progress must be callable");
        std::vector<Complex> snapshot(static_cast<std::size_t>(plan.density_element_count));
        std::memcpy(snapshot.data(),density.data(),plan.borrowed_density_bytes);
        struct Callback { py::object callable; Complex* snapshot; bool mutate; bool has_callable; };
        Callback callback{progress,snapshot.data(),mutate_snapshot,!progress.is_none()};
        const bool has_callback=!progress.is_none() || mutate_snapshot;
        py::gil_scoped_release release;
        return vibeqc::build_periodic_gaussian_fock(context,ao,auxiliary,
            {snapshot.data(),plan.density_element_count},config,live,caps,
            has_callback ? +[](const Progress& event,void* pointer) {
                auto& c=*static_cast<Callback*>(pointer);
                if(c.mutate) { c.snapshot[0]+=Complex(1.0,0.0); c.mutate=false; }
                if(!c.has_callable) return;
                py::gil_scoped_acquire acquire;
                c.callable(py::cast(Progress(event)));
            } : nullptr,&callback);
    };
    m.def("_build_periodic_gaussian_fock",[run](std::shared_ptr<const Context> context,
        const vibeqc::BasisSet& ao,const vibeqc::BasisSet& auxiliary,py::array density,
        const Config& config,const Live& live,const Caps& caps,py::object progress) {
        return run(std::move(context),ao,auxiliary,density,config,live,caps,progress,false);
    },py::arg("context"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("density").noconvert(),
      py::arg("config"),py::arg("live"),py::arg("caps"),py::arg("progress")=py::none());
    // Failure-path witness only: deliberately violates the C++ borrowed-D
    // contract inside a native callback. No caller-labelled factors accepted.
    m.def("_periodic_gaussian_fock_density_mutation_diagnostic",[run](std::shared_ptr<const Context> context,
        const vibeqc::BasisSet& ao,const vibeqc::BasisSet& auxiliary,py::array density,
        const Config& config,const Live& live,const Caps& caps) {
        return run(std::move(context),ao,auxiliary,density,config,live,caps,py::none(),true);
    },py::arg("context"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("density").noconvert(),
      py::arg("config"),py::arg("live"),py::arg("caps"));
}
