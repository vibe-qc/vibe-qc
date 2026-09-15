// Included after bounded_restricted_pair_ccsd_amplitudes_bindings.cpp.
// Reuses its tiny pinned-table seam; no borrowed reader escapes to Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cfenv>
#include <cstring>
#include <limits>
#include "vibeqc/bounded_restricted_local_triples_moments.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_local_triples_moments(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Options = BoundedRestrictedLocalTriplesMomentsOptions;
    using Inventory = BoundedRestrictedLocalTriplesMomentsInventory;
    using Caps = BoundedRestrictedLocalTriplesMomentsCaps;
    using Plan = BoundedRestrictedLocalTriplesMomentsMemoryPlan;
    using Result = BoundedRestrictedLocalTriplesMomentsResult;
    auto options = py::class_<Options>(m,"_BoundedRestrictedLocalTriplesMomentsOptions").def(py::init<>());
    options.def_readwrite("coefficient_orthogonality_tolerance",&Options::coefficient_orthogonality_tolerance)
        .def_readwrite("amplitude_symmetry_tolerance",&Options::amplitude_symmetry_tolerance)
        .def_readwrite("maximum_integral_work_units_per_call",&Options::maximum_integral_work_units_per_call);
    auto inventory = py::class_<Inventory>(m,"_BoundedRestrictedLocalTriplesMomentsInventory").def(py::init<>());
#define LTMI(f) inventory.def_readwrite(#f,&Inventory::f)
    LTMI(numerical_replicas); LTMI(external_node_bytes); LTMI(other_live_bytes_per_replica);
    LTMI(fixed_backend_margin_bytes_per_replica);
#undef LTMI
    auto caps = py::class_<Caps>(m,"_BoundedRestrictedLocalTriplesMomentsCaps").def(py::init<>());
#define LTMC(f) caps.def_readwrite(#f,&Caps::f)
    LTMC(maximum_occupied_count); LTMC(maximum_common_virtual_dimension); LTMC(maximum_rank);
    LTMC(maximum_owned_numerical_bytes); LTMC(maximum_per_replica_inventoried_bytes);
    LTMC(maximum_node_inventoried_bytes); LTMC(maximum_common_singles_calls); LTMC(maximum_common_doubles_calls);
    LTMC(maximum_transformed_integral_calls); LTMC(maximum_common_integral_calls); LTMC(maximum_work_units);
#undef LTMC
    auto plan = py::class_<Plan>(m,"_BoundedRestrictedLocalTriplesMomentsMemoryPlan");
#define LTMP(f) plan.def_readonly(#f,&Plan::f)
    LTMP(uniform_rank_upper_bound); LTMP(n_occupied); LTMP(common_virtual_dimension); LTMP(rank); LTMP(moments);
    LTMP(reader_borrowed_numerical_bytes); LTMP(reader_borrowed_table_bytes); LTMP(reader_control_storage_bytes);
    LTMP(target_coefficient_bytes); LTMP(original_f_ov_bytes); LTMP(integral_retained_numerical_bytes);
    LTMP(integral_maximum_transient_numerical_bytes); LTMP(projected_zero_f_ov_bytes); LTMP(output_numerical_bytes);
    LTMP(peak_owned_numerical_bytes); LTMP(control_storage_reservation_bytes); LTMP(per_replica_inventoried_bytes);
    LTMP(required_node_inventoried_bytes); LTMP(common_singles_calls); LTMP(common_doubles_calls);
    LTMP(transformed_integral_calls); LTMP(common_integral_calls); LTMP(one_virtual_integral_calls);
    LTMP(two_virtual_integral_calls); LTMP(three_virtual_integral_calls);
    LTMP(maximum_common_integral_calls_per_transformed_query);
    LTMP(maximum_projected_singles_work_units_per_query); LTMP(maximum_projected_doubles_work_units_per_query);
    LTMP(validation_work_units); LTMP(projection_work_units); LTMP(integral_work_units); LTMP(work_units_upper_bound);
#undef LTMP
    const auto copy = [](const Result& result, bool connected) {
        const U r = result.memory().rank;
        const auto& values = connected ? result.moments().moments.connected : result.moments().moments.singles;
        if (!r || r > 4 || values.size() != r*r*r)
            throw std::length_error("local TNO moments diagnostic output copy exceeds 512 bytes");
        py::array_t<double> out({static_cast<py::ssize_t>(r),static_cast<py::ssize_t>(r),static_cast<py::ssize_t>(r)});
        std::memcpy(out.mutable_data(),values.data(),values.size()*8);
        return out;
    };
    py::class_<Result>(m,"_BoundedRestrictedLocalTriplesMomentsResult")
        .def_property_readonly("memory",&Result::memory,py::return_value_policy::reference_internal)
        .def_property_readonly("moments",&Result::moments,py::return_value_policy::reference_internal)
        .def_property_readonly("common_singles_calls",&Result::common_singles_calls)
        .def_property_readonly("common_doubles_calls",&Result::common_doubles_calls)
        .def_property_readonly("common_integral_calls",&Result::common_integral_calls)
        .def_property_readonly("transformed_integral_calls",&Result::transformed_integral_calls)
        .def_property_readonly("coefficient_orthogonality_error",&Result::coefficient_orthogonality_error)
        .def_property_readonly("reader_snapshot_identity_sha256",&Result::reader_snapshot_identity_sha256)
        .def_property_readonly("target_coefficients_identity_sha256",&Result::target_coefficients_identity_sha256)
        .def_property_readonly("input_identity_sha256",&Result::input_identity_sha256)
        .def_property_readonly("consumed_integral_receipt_sha256",&Result::consumed_integral_receipt_sha256)
        .def_property_readonly("payload_identity_sha256",&Result::payload_identity_sha256)
        .def_property_readonly("connected",[copy](const Result& r){return copy(r,true);})
        .def_property_readonly("singles",[copy](const Result& r){return copy(r,false);});
    m.def("_plan_bounded_restricted_local_triples_moments_upper",&plan_bounded_restricted_local_triples_moments_upper,
        py::arg("n_occupied"),py::arg("common_virtual_dimension"),py::arg("rank"),
        py::arg("maximum_singles_rank"),py::arg("maximum_pair_rank"),
        py::arg("integral_retained_numerical_bytes"),py::arg("integral_maximum_transient_numerical_bytes"),
        py::arg("options"),py::arg("inventory"));
    const auto dispatch = [](U n,py::list sc,py::list st,py::list pc,py::list pt,
        BoundedRestrictedPairCCSDAmplitudesOptions reader_options,
        BoundedRestrictedPairCCSDAmplitudesInventory reader_inventory,
        BoundedRestrictedPairCCSDAmplitudesCaps reader_caps,
        py::array coefficients,py::array fov,py::array eri,std::array<U,3> occupied,U snapshot,
        Options options,Inventory inventory,Caps caps,py::object callback,U fail_before,U transient,
        bool mutate_controls,unsigned floating_fault,bool planning) -> py::object {
        namespace seam = bounded_pair_ccsd_binding;
        seam::array(coefficients,2); seam::array(fov,2); seam::array(eri,4);
        const auto o = static_cast<U>(sc.size());
        const auto r = static_cast<U>(coefficients.shape(1));
        if (!o || o > 4 || !n || n > 4 || !r || r > n
            || coefficients.shape(0) != static_cast<py::ssize_t>(n)
            || fov.shape(0) != static_cast<py::ssize_t>(o) || fov.shape(1) != static_cast<py::ssize_t>(n))
            throw std::invalid_argument("local TNO moments diagnostic has invalid tiny shapes");
        for (int axis = 0; axis < 4; ++axis)
            if (eri.shape(axis) != static_cast<py::ssize_t>(o+n))
                throw std::invalid_argument("local TNO moments diagnostic ERI shape mismatch");
        if (caps.maximum_owned_numerical_bytes > (1U<<20) || caps.maximum_node_inventoried_bytes > (128U<<20)
            || caps.maximum_work_units > 1000000000000000ULL || transient > (1U<<20) || floating_fault > 2)
            throw std::length_error("local TNO moments diagnostic exceeds tiny memory/work bounds");
        if (options.maximum_integral_work_units_per_call != 1)
            throw std::invalid_argument("local TNO moments dense diagnostic integral work declaration must equal one");
        const auto borrowed = seam::borrow(n,sc,st,pc,pt,reader_inventory,reader_caps);
        const auto reader_plan = plan_bounded_restricted_pair_ccsd_amplitudes(borrowed.input(),reader_options,reader_inventory,reader_caps);
        seam::tiny(reader_plan);
        const auto upper = plan_bounded_restricted_local_triples_moments_upper(o,n,r,
            reader_plan.maximum_singles_rank,reader_plan.maximum_pair_rank,eri.nbytes(),transient,options,inventory);
        // Conservative pre-reader admission: no floating reader validation can
        // run with unadmitted outer memory/work. Exact native gates run again.
        const auto bound = [](U value,U cap) {
            if (!cap || value > cap) throw std::length_error("local TNO moments diagnostic pre-reader upper cap exceeded");
        };
        bound(upper.n_occupied,caps.maximum_occupied_count); bound(upper.common_virtual_dimension,caps.maximum_common_virtual_dimension);
        bound(upper.rank,caps.maximum_rank); bound(upper.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes);
        bound(upper.per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes);
        bound(upper.required_node_inventoried_bytes,caps.maximum_node_inventoried_bytes);
        bound(upper.common_singles_calls,caps.maximum_common_singles_calls); bound(upper.common_doubles_calls,caps.maximum_common_doubles_calls);
        bound(upper.transformed_integral_calls,caps.maximum_transformed_integral_calls); bound(upper.common_integral_calls,caps.maximum_common_integral_calls);
        bound(upper.work_units_upper_bound,caps.maximum_work_units);
        if (upper.peak_owned_numerical_bytes > (1U<<20) || upper.required_node_inventoried_bytes > (128U<<20)
            || upper.work_units_upper_bound > 1000000000000000ULL)
            throw std::length_error("local TNO moments diagnostic plan exceeds tiny bounds");
        auto reader = make_bounded_restricted_pair_ccsd_amplitudes(borrowed.input(),reader_options,reader_inventory,reader_caps);
        BoundedRestrictedLocalTriplesMomentsInput input{r,occupied,snapshot,seam::view(coefficients),seam::view(fov)};
        BoundedRestrictedCCSDIntegralProvider provider;
        struct Direct {
            const double* eri;
            U n,fail_before,calls = 0;
            py::object callback;
            const seam::Borrowed* borrowed;
            const py::array* coefficients; const py::array* fov; const py::array* integrals;
            BoundedRestrictedLocalTriplesMomentsInput* input;
            Options* options; Inventory* inventory; Caps* caps; BoundedRestrictedCCSDIntegralProvider* provider;
            bool mutate;
            unsigned floating_fault;
            static double value(U p,U q,U r,U s,void* context) {
                auto& self = *static_cast<Direct*>(context);
                if (self.calls == self.fail_before)
                    throw std::runtime_error("local TNO moments diagnostic injected integral failure");
                if (p >= self.n || q >= self.n || r >= self.n || s >= self.n)
                    throw std::logic_error("local TNO moments diagnostic invalid common integral label");
                if (!self.calls && !self.callback.is_none()) {
                    self.callback();
                    self.borrowed->validate_python_descriptors();
                    seam::array(*self.coefficients,2); seam::array(*self.fov,2); seam::array(*self.integrals,4);
                    if (self.coefficients->data() != self.input->target_coefficients.data
                        || self.coefficients->size() != static_cast<py::ssize_t>(self.input->target_coefficients.element_count)
                        || self.fov->data() != self.input->original_f_ov.data
                        || self.fov->size() != static_cast<py::ssize_t>(self.input->original_f_ov.element_count)
                        || self.integrals->data() != self.eri || static_cast<U>(self.integrals->size()) != self.n*self.n*self.n*self.n)
                        throw std::invalid_argument("local TNO moments diagnostic callback changed array descriptors");
                }
                if (self.mutate) { *self.input={}; *self.options={}; *self.inventory={}; *self.caps={}; *self.provider={}; }
                if (self.floating_fault == 2 && std::fesetround(FE_UPWARD) != 0)
                    throw std::runtime_error("local TNO moments diagnostic cannot set FP fault");
                ++self.calls;
                return self.eri[((p*self.n+q)*self.n+r)*self.n+s];
            }
        } direct{static_cast<const double*>(eri.data()),o+n,fail_before,0,callback,&borrowed,&coefficients,&fov,&eri,
            &input,&options,&inventory,&caps,&provider,mutate_controls,floating_fault};
        provider = {&Direct::value,&direct,static_cast<U>(eri.nbytes()),transient};
        if (planning) return py::cast(plan_bounded_restricted_local_triples_moments(input,reader,provider,options,inventory));
        struct Restore {
            std::fenv_t saved;
            Restore() { if (std::fegetenv(&saved)) throw std::runtime_error("cannot save local TNO diagnostic FP environment"); }
            ~Restore() { std::fesetenv(&saved); }
        } restore;
        if (floating_fault == 1 && std::fesetround(FE_UPWARD) != 0)
            throw std::runtime_error("local TNO moments diagnostic cannot set FP fault");
        // GIL retained for the explicit test-only Python callback; all array
        // and list-element owners are pinned. No Python numerical contraction.
        return py::cast(bounded_restricted_local_triples_moments(input,reader,provider,options,inventory,caps));
    };
    m.def("_bounded_restricted_local_triples_moments_diagnostic",dispatch,
        py::arg("common_virtual_dimension"),py::arg("singles_coefficients"),py::arg("singles_amplitudes"),
        py::arg("pair_coefficients"),py::arg("pair_amplitudes"),py::arg("reader_options"),py::arg("reader_inventory"),py::arg("reader_caps"),
        py::arg("target_coefficients"),py::arg("original_f_ov"),py::arg("integrals"),py::arg("occupied"),py::arg("ccsd_snapshot_id"),
        py::arg("options"),py::arg("inventory"),py::arg("caps"),py::arg("callback")=py::none(),
        py::arg("fail_before_call")=std::numeric_limits<U>::max(),py::arg("provider_transient_bytes")=0,
        py::arg("mutate_control_objects")=false,py::arg("floating_environment_fault")=0,py::arg("planning")=false);
}
