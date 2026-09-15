// Tiny algebra diagnostics only; arrays do not construct a physical source.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <stdexcept>
#include "vibeqc/bounded_restricted_pair_mp2_solver.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_bounded_restricted_pair_mp2_solver(py::module_& m) {
    using namespace vibeqc;
    using Options=BoundedRestrictedPairMP2SolverOptions;
    using Inventory=BoundedRestrictedPairMP2SolverInventory;
    using Caps=BoundedRestrictedPairMP2SolverCaps;
    using Plan=BoundedRestrictedPairMP2SolverMemoryPlan;
    using Progress=BoundedRestrictedPairMP2SolverProgress;
    using Result=BoundedRestrictedPairMP2SolverResult;
    auto options=py::class_<Options>(m,"_BoundedRestrictedPairMP2SolverOptions"); options.def(py::init<>());
#define PAIR_MP2_OPTION(f) options.def_readwrite(#f,&Options::f)
    PAIR_MP2_OPTION(maximum_iterations); PAIR_MP2_OPTION(denominator_floor);
    PAIR_MP2_OPTION(residual_tolerance); PAIR_MP2_OPTION(energy_tolerance);
    PAIR_MP2_OPTION(coefficient_orthogonality_tolerance); PAIR_MP2_OPTION(maximum_diagonal_update_antisymmetry_norm);
#undef PAIR_MP2_OPTION
    auto inventory=py::class_<Inventory>(m,"_BoundedRestrictedPairMP2SolverInventory"); inventory.def(py::init<>());
#define PAIR_MP2_INVENTORY(f) inventory.def_readwrite(#f,&Inventory::f)
    PAIR_MP2_INVENTORY(numerical_replicas); PAIR_MP2_INVENTORY(external_node_bytes);
    PAIR_MP2_INVENTORY(other_live_bytes_per_replica); PAIR_MP2_INVENTORY(fixed_backend_margin_bytes_per_replica);
#undef PAIR_MP2_INVENTORY
    auto caps=py::class_<Caps>(m,"_BoundedRestrictedPairMP2SolverCaps"); caps.def(py::init<>());
#define PAIR_MP2_CAP(f) caps.def_readwrite(#f,&Caps::f)
    PAIR_MP2_CAP(maximum_occupied_count); PAIR_MP2_CAP(maximum_common_virtual_dimension);
    PAIR_MP2_CAP(maximum_pair_count); PAIR_MP2_CAP(maximum_pair_rank);
    PAIR_MP2_CAP(maximum_owned_numerical_bytes); PAIR_MP2_CAP(maximum_per_replica_inventoried_bytes);
    PAIR_MP2_CAP(maximum_node_inventoried_bytes); PAIR_MP2_CAP(maximum_coupling_slots); PAIR_MP2_CAP(maximum_work_units);
#undef PAIR_MP2_CAP
    auto plan=py::class_<Plan>(m,"_BoundedRestrictedPairMP2SolverMemoryPlan");
#define PAIR_MP2_PLAN(f) plan.def_readonly(#f,&Plan::f)
    PAIR_MP2_PLAN(uniform_rank_upper_bound); PAIR_MP2_PLAN(n_occupied); PAIR_MP2_PLAN(common_virtual_dimension);
    PAIR_MP2_PLAN(pair_count); PAIR_MP2_PLAN(maximum_pair_rank); PAIR_MP2_PLAN(maximum_iterations);
    PAIR_MP2_PLAN(total_amplitude_elements); PAIR_MP2_PLAN(amplitude_snapshot_bytes); PAIR_MP2_PLAN(candidate_snapshot_bytes);
    PAIR_MP2_PLAN(retained_pair_record_bytes); PAIR_MP2_PLAN(borrowed_pair_table_bytes);
    PAIR_MP2_PLAN(borrowed_fock_bytes); PAIR_MP2_PLAN(borrowed_pair_numeric_bytes); PAIR_MP2_PLAN(output_numerical_bytes);
    PAIR_MP2_PLAN(initialization_phase_owned_bytes); PAIR_MP2_PLAN(residual_workspace_bytes); PAIR_MP2_PLAN(overlap_workspace_bytes);
    PAIR_MP2_PLAN(iteration_phase_owned_bytes); PAIR_MP2_PLAN(peak_owned_numerical_bytes);
    PAIR_MP2_PLAN(fixed_control_storage_bytes); PAIR_MP2_PLAN(per_replica_inventoried_bytes); PAIR_MP2_PLAN(numerical_replicas);
    PAIR_MP2_PLAN(external_node_bytes); PAIR_MP2_PLAN(required_node_inventoried_bytes);
    PAIR_MP2_PLAN(metadata_work_units); PAIR_MP2_PLAN(validation_work_units); PAIR_MP2_PLAN(initialization_work_units);
    PAIR_MP2_PLAN(work_units_per_snapshot); PAIR_MP2_PLAN(work_units_upper_bound); PAIR_MP2_PLAN(target_evaluations_upper_bound);
    PAIR_MP2_PLAN(coupling_slots_upper_bound); PAIR_MP2_PLAN(overlap_scalar_products_upper_bound);
    PAIR_MP2_PLAN(residual_scalar_products_upper_bound);
#undef PAIR_MP2_PLAN
    auto progress=py::class_<Progress>(m,"_BoundedRestrictedPairMP2SolverProgress");
#define PAIR_MP2_PROGRESS(f) progress.def_readonly(#f,&Progress::f)
    PAIR_MP2_PROGRESS(iteration); PAIR_MP2_PROGRESS(target_evaluations); PAIR_MP2_PROGRESS(coupling_slots);
    PAIR_MP2_PROGRESS(transposed_sources); PAIR_MP2_PROGRESS(zero_rank_sources); PAIR_MP2_PROGRESS(zero_rank_targets);
    PAIR_MP2_PROGRESS(overlap_scalar_products); PAIR_MP2_PROGRESS(residual_scalar_products);
    PAIR_MP2_PROGRESS(input_checks); PAIR_MP2_PROGRESS(charged_work_units); PAIR_MP2_PROGRESS(arithmetic_underflow_count);
    PAIR_MP2_PROGRESS(has_previous_energy); PAIR_MP2_PROGRESS(converged); PAIR_MP2_PROGRESS(correlation_energy);
    PAIR_MP2_PROGRESS(energy_change); PAIR_MP2_PROGRESS(maximum_absolute_residual); PAIR_MP2_PROGRESS(residual_frobenius_norm);
    PAIR_MP2_PROGRESS(maximum_diagonal_update_antisymmetry_norm);
#undef PAIR_MP2_PROGRESS
    py::class_<Result>(m,"_BoundedRestrictedPairMP2SolverResult")
        .def_property_readonly("memory",[](const Result& r) { return r.memory(); })
        .def_property_readonly("final_snapshot",[](const Result& r) { return r.final_snapshot(); })
        .def_property_readonly("minimum_denominator",&Result::minimum_denominator)
        .def_property_readonly("maximum_denominator",&Result::maximum_denominator)
        .def_property_readonly("input_identity_sha256",&Result::input_identity_sha256)
        .def_property_readonly("payload_sha256",&Result::payload_sha256)
        .def("pair_rank",&Result::pair_rank)
        .def("amplitude",&Result::amplitude)
        .def("pair_copy",[](const Result& r,std::uint64_t i,std::uint64_t j) {
            const auto n=r.pair_rank(i,j);
            if (n>4) throw std::length_error("pair MP2 diagnostic copy exceeds tiny rank");
            py::array_t<double> out({static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)});
            for (std::uint64_t a=0;a<n;++a) for (std::uint64_t b=0;b<n;++b)
                out.mutable_data()[a*n+b]=r.amplitude(i,j,a,b);
            return out;
        });
    m.def("_plan_bounded_restricted_pair_mp2_solver_upper",&plan_bounded_restricted_pair_mp2_solver_upper,
        py::arg("n_occupied"),py::arg("common_virtual_dimension"),py::arg("maximum_pair_rank"),
        py::arg("maximum_iterations"),py::arg("inventory"));
    const auto dispatch=[](const py::array& f,py::list gs,py::list energies,py::list coefficients,
        Options options,Inventory inventory,Caps caps,py::object callback,bool planning)->py::object {
        const auto array=[](py::handle object,int dimensions) {
            if (!py::isinstance<py::array>(object)) throw std::invalid_argument("pair MP2 diagnostic requires existing arrays");
            auto a=py::reinterpret_borrow<py::array>(object);
            if (!a.dtype().is(py::dtype::of<double>()) || !(a.flags()&py::array::c_style)
                || a.ndim()!=dimensions || (a.size() && reinterpret_cast<std::uintptr_t>(a.data())%alignof(double)))
                throw std::invalid_argument("pair MP2 diagnostic requires aligned contiguous float64 arrays");
            return a;
        };
        const auto fock=array(f,2);
        if (fock.shape(0)<1 || fock.shape(0)>4 || fock.shape(0)!=fock.shape(1)
            || options.maximum_iterations>128 || caps.maximum_owned_numerical_bytes>(1U<<20)
            || caps.maximum_node_inventoried_bytes>(128U<<20) || caps.maximum_work_units>1000000000000ULL)
            throw std::length_error("pair MP2 diagnostic exceeds tiny dimension, memory or work limits");
        const auto o=static_cast<std::uint64_t>(fock.shape(0)), count=o*(o+1)/2;
        if (gs.size()!=count || energies.size()!=count || coefficients.size()!=count)
            throw std::invalid_argument("pair MP2 diagnostic needs a complete unordered pair table");
        // Fixed tiny wrapper storage; pin all thirty possible array owners
        // even if the callback replaces an entry in the original lists.
        std::array<py::array,30> owners;
        std::array<BoundedRestrictedPairMP2PairView,10> pairs{};
        std::uint64_t common=0;
        for (std::uint64_t k=0;k<count;++k) {
            owners[3*k]=array(gs[k],2); owners[3*k+1]=array(energies[k],1); owners[3*k+2]=array(coefficients[k],2);
            const auto& g=owners[3*k]; const auto& e=owners[3*k+1]; const auto& c=owners[3*k+2];
            const auto rank=g.shape(0);
            if (rank>4 || g.shape(1)!=rank || e.shape(0)!=rank || c.shape(1)!=rank
                || c.shape(0)<1 || c.shape(0)>4 || (k && c.shape(0)!=static_cast<py::ssize_t>(common)))
                throw std::invalid_argument("pair MP2 diagnostic pair shapes are inconsistent");
            common=static_cast<std::uint64_t>(c.shape(0));
            const auto view=[](const py::array& a) { return BoundedRestrictedPairMP2RealView{
                static_cast<const double*>(a.data()),static_cast<std::size_t>(a.size())}; };
            pairs[k]={static_cast<std::uint64_t>(rank),view(g),view(e),view(c)};
        }
        const BoundedRestrictedPairMP2SolverInput input{o,common,
            {static_cast<const double*>(fock.data()),static_cast<std::size_t>(fock.size())},pairs.data(),static_cast<std::size_t>(count)};
        if (planning) return py::cast(plan_bounded_restricted_pair_mp2_solver(input,options,inventory,caps));
        if (!callback.is_none()&&!PyCallable_Check(callback.ptr())) throw std::invalid_argument("pair MP2 progress must be callable or None");
        const auto notify=[](const Progress& p,void* context) { (*static_cast<py::object*>(context))(py::cast(Progress(p))); };
        return py::cast(bounded_restricted_pair_mp2_solve(input,options,inventory,caps,
            callback.is_none()?nullptr:+notify,callback.is_none()?nullptr:&callback));
    };
    m.def("_plan_bounded_restricted_pair_mp2_solver",[dispatch](const py::array& f,py::list g,py::list e,py::list c,
        Options o,Inventory i,Caps caps) { return dispatch(f,g,e,c,o,i,caps,py::none(),true); },
        py::arg("occupied_fock").noconvert(),py::arg("integrals"),py::arg("energies"),py::arg("coefficients"),
        py::arg("options"),py::arg("inventory"),py::arg("caps"));
    m.def("_bounded_restricted_pair_mp2_solve_diagnostic",[dispatch](const py::array& f,py::list g,py::list e,py::list c,
        Options o,Inventory i,Caps caps,py::object callback) { return dispatch(f,g,e,c,o,i,caps,callback,false); },
        py::arg("occupied_fock").noconvert(),py::arg("integrals"),py::arg("energies"),py::arg("coefficients"),
        py::arg("options"),py::arg("inventory"),py::arg("caps"),py::arg("callback")=py::none());
}
