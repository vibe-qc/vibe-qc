// Tiny actual-source diagnostics; no supplied integrals, provider or PNO arrays.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_gram_pair_pnos.hpp"
#include "vibeqc/periodic_gaussian_pair_space.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace periodic_gaussian_gram_pair_pnos_python {
using namespace vibeqc;
using Result = PeriodicGaussianGramPairPNOResult;
using Complex = std::complex<double>;

void tiny(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairDomainGeometry& geometry,
    const PeriodicGaussianGramPairPNOConfig& config,const PeriodicGaussianGramPairPNOCaps& caps) {
    if (!ref.state_handle() || !hf.context_handle())
        throw std::invalid_argument("Gaussian Gram pair PNO diagnostic requires live native source owners");
    if (ref.state().n_kpoints()>8 || ref.state().n_basis()>4
        || hf.context_handle()->inventory().auxiliary.function_count>4
        || basis.memory().occupied_count>4 || basis.memory().virtual_count>4
        || geometry.domain().domain_dimension()>96 || geometry.real_space().space().retained_dimension()>4
        || config.gram.auxiliary_block>4 || caps.maximum_owned_numerical_bytes>(16ULL<<20)
        || caps.maximum_node_bytes>(128ULL<<20) || caps.maximum_work_units>10000000000000000000ULL
        || hf.plan().fock.config.source_caps.maximum_candidate_evaluations>65536)
        throw std::length_error("Gaussian Gram pair PNO diagnostic exceeds tiny shape/source/resource bounds");
}

py::array_t<double> matrix(const std::vector<double>& values,std::uint64_t rows,std::uint64_t columns) {
    if (rows>4 || columns>4 || values.size()!=rows*columns)
        throw std::length_error("Gaussian Gram pair PNO diagnostic copy exceeds tiny shape");
    py::array_t<double> out({static_cast<py::ssize_t>(rows),static_cast<py::ssize_t>(columns)});
    if (!values.empty()) std::memcpy(out.mutable_data(),values.data(),8*values.size());
    out.attr("setflags")(false);
    return out;
}
py::array_t<double> vector(const std::vector<double>& values) {
    if (values.size()>4) throw std::length_error("Gaussian Gram pair PNO diagnostic vector exceeds tiny shape");
    py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) std::memcpy(out.mutable_data(),values.data(),8*values.size());
    out.attr("setflags")(false);
    return out;
}
void tiny_space(const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
    const Result& result,const PeriodicGaussianPairSpaceCaps& caps) {
    if (!ref.state_handle() || ref.state().n_kpoints()>8 || basis.memory().occupied_count>4
        || basis.memory().virtual_count>4 || result.memory().generation_dimension>4
        || caps.maximum_owned_numerical_bytes>(16ULL<<20)
        || caps.maximum_node_inventoried_bytes>(128ULL<<20) || caps.maximum_work_units>100000000000000ULL)
        throw std::length_error("Gaussian direct Gram pair space diagnostic exceeds tiny bounds");
}
} // namespace periodic_gaussian_gram_pair_pnos_python

void bind_periodic_gaussian_gram_pair_pnos(py::module_& m) {
    using namespace vibeqc;
    namespace binding=periodic_gaussian_gram_pair_pnos_python;
    using Config=PeriodicGaussianGramPairPNOConfig;
    using Options=PeriodicGaussianGramPairPNOOptions;
    using Live=PeriodicGaussianGramPairPNOLiveInventory;
    using Caps=PeriodicGaussianGramPairPNOCaps;
    using Plan=PeriodicGaussianGramPairPNOPlan;
    using Diagnostics=PeriodicGaussianGramPairPNODiagnostics;
    using Validation=PeriodicGaussianGramPairPNOPayloadValidationPlan;
    using Result=PeriodicGaussianGramPairPNOResult;
    using Space=PeriodicGaussianPairSpace;
    using Complex=std::complex<double>;
    py::class_<Config>(m,"_PeriodicGaussianGramPairPNOConfig").def(py::init<>()).def_readwrite("gram",&Config::gram);
    py::class_<Options>(m,"_PeriodicGaussianGramPairPNOOptions").def(py::init<>())
        .def_readwrite("local_basis",&Options::local_basis).def_readwrite("gram",&Options::gram)
        .def_readwrite("pno",&Options::pno)
        .def_readwrite("maximum_occupied_fock_difference",&Options::maximum_occupied_fock_difference)
        .def_readwrite("maximum_retained_diagonal_projection_norm",&Options::maximum_retained_diagonal_projection_norm);
    py::class_<Live>(m,"_PeriodicGaussianGramPairPNOLiveInventory").def(py::init<>())
        .def_readwrite("other_live_numerical_bytes_per_worker",&Live::other_live_numerical_bytes_per_worker)
        .def_readwrite("other_live_control_bytes_per_worker",&Live::other_live_control_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker",&Live::fixed_backend_margin_bytes_per_worker);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianGramPairPNOCaps"); caps.def(py::init<>());
#define GPNO_CAP(f) caps.def_readwrite(#f,&Caps::f)
    GPNO_CAP(local_basis);GPNO_CAP(gram);GPNO_CAP(maximum_gram_control_storage_bytes);
    GPNO_CAP(maximum_common_dimension);GPNO_CAP(maximum_generation_dimension);GPNO_CAP(maximum_owned_numerical_bytes);
    GPNO_CAP(maximum_control_storage_bytes);GPNO_CAP(maximum_worker_bytes);GPNO_CAP(maximum_node_bytes);GPNO_CAP(maximum_work_units);
#undef GPNO_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianGramPairPNOPlan");
    plan.def_property_readonly("gram_selection",[](const Plan& p){return p.gram_selection;})
        .def_property_readonly("local_basis",[](const Plan& p){return p.local_basis;});
#define GPNO_PLAN(f) plan.def_readonly(#f,&Plan::f)
    GPNO_PLAN(n_cells);GPNO_PLAN(n_basis);GPNO_PLAN(occupied_count);GPNO_PLAN(common_virtual_dimension);
    GPNO_PLAN(generation_dimension);GPNO_PLAN(occupied_slot_i);GPNO_PLAN(occupied_slot_j);GPNO_PLAN(local_occupied_count);
    GPNO_PLAN(gram_phase_upper_bytes);GPNO_PLAN(copy_phase_bytes);GPNO_PLAN(amplitude_phase_bytes);GPNO_PLAN(density_phase_bytes);
    GPNO_PLAN(semicanonical_phase_upper_bytes);GPNO_PLAN(retained_integral_phase_upper_bytes);GPNO_PLAN(export_phase_upper_bytes);
    GPNO_PLAN(peak_owned_numerical_bytes);GPNO_PLAN(retained_output_upper_bytes);GPNO_PLAN(borrowed_common_basis_bytes);
    GPNO_PLAN(borrowed_geometry_bytes);GPNO_PLAN(borrowed_wannier_bytes);GPNO_PLAN(borrowed_gauge_bytes);
    GPNO_PLAN(borrowed_gaussian_bytes);GPNO_PLAN(complete_borrowed_numerical_bytes);GPNO_PLAN(fixed_control_storage_bytes);
    GPNO_PLAN(control_storage_reservation_bytes);GPNO_PLAN(input_validation_work_units);GPNO_PLAN(numerical_work_units);
    GPNO_PLAN(gram_work_units_upper_bound);GPNO_PLAN(work_units);GPNO_PLAN(replicas_per_node);
    GPNO_PLAN(reference_base_node_bytes);GPNO_PLAN(worker_bytes);GPNO_PLAN(required_node_memory_bytes);
#undef GPNO_PLAN
    auto diag=py::class_<Diagnostics>(m,"_PeriodicGaussianGramPairPNODiagnostics");
    diag.def_property_readonly("local_basis",[](const Diagnostics& d){return d.local_basis;})
        .def_property_readonly("gram_memory",[](const Diagnostics& d){return d.gram_memory;})
        .def_property_readonly("gram",[](const Diagnostics& d){return d.gram;})
        .def_property_readonly("generation",[](const Diagnostics& d){return d.generation;});
#define GPNO_DIAG(f) diag.def_readonly(#f,&Diagnostics::f)
    GPNO_DIAG(retained_dimension);GPNO_DIAG(retained_frame_bytes);GPNO_DIAG(retained_output_bytes);
    GPNO_DIAG(retained_generation_coefficient_bytes);GPNO_DIAG(actual_semicanonical_phase_bytes);
    GPNO_DIAG(actual_retained_integral_phase_bytes);GPNO_DIAG(actual_export_phase_bytes);
    GPNO_DIAG(maximum_raw_exchange_transpose_defect);GPNO_DIAG(diagonal_exchange_projection_frobenius_upper_bound);
    GPNO_DIAG(maximum_raw_retained_transpose_defect);GPNO_DIAG(retained_diagonal_projection_frobenius_upper_bound);
    GPNO_DIAG(occupied_fock_frobenius_upper_bound);GPNO_DIAG(exported_gram_frobenius_upper_bound);
    GPNO_DIAG(exported_fock_frobenius_upper_bound);GPNO_DIAG(exported_subspace_frobenius_upper_bound);
#undef GPNO_DIAG
    py::class_<Validation>(m,"_PeriodicGaussianGramPairPNOPayloadValidationPlan")
        .def_readonly("numerical_lanes",&Validation::numerical_lanes).def_readonly("work_units",&Validation::work_units)
        .def_readonly("control_storage_bytes",&Validation::control_storage_bytes);
    auto result=py::class_<Result>(m,"_PeriodicGaussianGramPairPNOResult");
    result.def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("options",[](const Result& r){return r.options();})
        .def_property_readonly("state",&Result::state_handle).def_property_readonly("context",&Result::context_handle)
        .def_property_readonly("occupied_i",[](const Result& r){auto p=r.occupied_i();return py::make_tuple(p.occupied_index,p.cell);})
        .def_property_readonly("occupied_j",[](const Result& r){auto p=r.occupied_j();return py::make_tuple(p.occupied_index,p.cell);})
        .def("coefficients_copy",[](const Result& r){return binding::matrix(r.coefficients(),r.memory().common_virtual_dimension,r.diagnostics().retained_dimension);})
        .def("generation_coefficients_copy",[](const Result& r){return binding::matrix(r.generation_coefficients(),r.memory().generation_dimension,r.diagnostics().retained_dimension);})
        .def("exchange_integrals_copy",[](const Result& r){return binding::matrix(r.exchange_integrals(),r.diagnostics().retained_dimension,r.diagnostics().retained_dimension);})
        .def("energies_copy",[](const Result& r){return binding::vector(r.energies());})
        .def("original_pno_occupations_copy",[](const Result& r){return binding::vector(r.original_pno_occupations());});
#define GPNO_RESULT(f) result.def_property_readonly(#f,&Result::f)
    GPNO_RESULT(diagonal_pair);GPNO_RESULT(complete_generation_pair_space);GPNO_RESULT(complete_common_virtual_space);
    GPNO_RESULT(retained_numerical_bytes);GPNO_RESULT(retained_receipt_payload_bytes);GPNO_RESULT(identity_sha256);
    GPNO_RESULT(payload_sha256);GPNO_RESULT(basis_identity_sha256);GPNO_RESULT(local_basis_identity_sha256);
    GPNO_RESULT(hf_reference_source_identity_sha256);GPNO_RESULT(source_context_identity_sha256);
    GPNO_RESULT(geometry_identity_sha256);GPNO_RESULT(embedding_identity_sha256);GPNO_RESULT(gram_identity_sha256);
    GPNO_RESULT(gram_payload_sha256);GPNO_RESULT(gram_consumed_sources_identity_sha256);GPNO_RESULT(raw_exchange_identity_sha256);
    GPNO_RESULT(projected_exchange_identity_sha256);GPNO_RESULT(local_fock_identity_sha256);
    GPNO_RESULT(raw_retained_exchange_identity_sha256);GPNO_RESULT(retained_exchange_identity_sha256);
    GPNO_RESULT(initial_amplitude_identity_sha256);GPNO_RESULT(density_identity_sha256);GPNO_RESULT(source_payload_receipt_sha256);
    GPNO_RESULT(matched_finite_gaussian_hf_recipe);GPNO_RESULT(original_provider_projection_reproduced_bitwise);
    GPNO_RESULT(coupled_mp2_solution);GPNO_RESULT(production_dlpno);GPNO_RESULT(infinite_source_accuracy_certified);
#undef GPNO_RESULT
    m.def("_plan_periodic_gaussian_gram_pair_pnos_diagnostic",[](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianPairDomainGeometry& geometry,Config config,Options options,Live live,Caps caps) {
        binding::tiny(hf,reference,basis,geometry,config,caps);
        return plan_periodic_gaussian_gram_pair_pnos(hf,reference,wannier,basis,geometry,config,options,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("basis"),py::arg("geometry"),
        py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_gram_pair_pnos_diagnostic",[](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,const py::array& gauges,
        const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairDomainGeometry& geometry,
        Config config,Options options,Live live,Caps caps) {
        binding::tiny(hf,reference,basis,geometry,config,caps);
        const auto& state=reference.state();
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || !(gauges.flags()&py::array::c_style)
            || gauges.ndim()!=3 || gauges.shape(0)!=static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1)!=static_cast<py::ssize_t>(state.n_correlated_occupied()) || gauges.shape(2)!=gauges.shape(1)
            || reinterpret_cast<std::uintptr_t>(gauges.data())%alignof(Complex))
            throw py::type_error("Gaussian Gram pair PNO gauges require exact C-contiguous complex128 [K,active,active]");
        // GIL held, exact arrays and owner arguments pinned, scalar controls
        // copied. No callback, numerical input copy or typed child escapes.
        return make_periodic_gaussian_gram_pair_pnos(hf,reference,ao,auxiliary,wannier,
            static_cast<const Complex*>(gauges.data()),static_cast<std::size_t>(gauges.size()),
            basis,geometry,config,options,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("wannier"),
        py::arg("gauges").noconvert(),py::arg("basis"),py::arg("geometry"),py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_plan_periodic_gaussian_gram_pair_pno_payload_validation",&plan_periodic_gaussian_gram_pair_pno_payload_validation);
    m.def("_verify_periodic_gaussian_gram_pair_pno_payload",&verify_periodic_gaussian_gram_pair_pno_payload,
        py::arg("result"),py::arg("maximum_work_units"));
    // These overloads return the already registered immutable Space. Its
    // direct seed is never returned as a typed, consuming Python owner.
    m.def("_plan_periodic_gaussian_gram_pair_space_diagnostic",[](
        const PeriodicCorrelationAdmittedReference& reference,const PeriodicCorrelationRealLocalBasis& basis,
        const Result& pno,PeriodicGaussianPairSpaceOptions options,PeriodicGaussianPairSpaceLiveInventory live,
        PeriodicGaussianPairSpaceCaps caps) {
        binding::tiny_space(reference,basis,pno,caps);
        return plan_periodic_gaussian_pair_space(reference,basis,pno,options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("pno"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_make_periodic_gaussian_gram_pair_space_diagnostic",[](
        const PeriodicCorrelationAdmittedReference& reference,const PeriodicCorrelationRealLocalBasis& basis,
        Result& pno,PeriodicGaussianPairSpaceOptions options,PeriodicGaussianPairSpaceLiveInventory live,
        PeriodicGaussianPairSpaceCaps caps) {
        binding::tiny_space(reference,basis,pno,caps);
        return make_periodic_gaussian_pair_space(reference,basis,std::move(pno),options,live,caps);
    },py::arg("reference"),py::arg("basis"),py::arg("pno"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_periodic_gaussian_pair_space_direct_gram_diagnostics",[](const Space& space){return space.direct_gram_pnos().diagnostics();});
    m.def("_periodic_gaussian_pair_space_direct_gram_generation_coefficients_copy",[](const Space& space){
        const auto& seed=space.direct_gram_pnos();
        return binding::matrix(seed.generation_coefficients(),seed.memory().generation_dimension,seed.diagnostics().retained_dimension);
    });
    m.def("_periodic_gaussian_direct_gram_pair_space_storage",&periodic_gaussian_direct_gram_pair_space_storage,
        py::arg("virtual_count"),py::arg("generation_dimension"),py::arg("retained_dimension"));
}
