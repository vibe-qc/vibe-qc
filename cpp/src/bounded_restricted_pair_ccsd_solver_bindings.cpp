// Tiny numerical diagnostics. Caller arrays do not certify a Hamiltonian or PNOs.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "vibeqc/bounded_restricted_pair_ccsd_solver.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {
// Native tiny-array fixture only. Never accepts a Python producer callback;
// the declared identity is an arithmetic test identity, not physical lineage.
struct PairCCSDSplitFixture {
    using U = std::uint64_t;
    using Reader = vibeqc::BoundedRestrictedPairCCSDAmplitudes;
    using Plan = vibeqc::BoundedRestrictedPairCCSDSolverMemoryPlan;
    using Visitor = vibeqc::BoundedRestrictedPairCCSDParticleHoleVisitor;
    const double* eri = nullptr;
    double* original_fov = nullptr;
    U o = 0, n = 0, maximum_rank = 0, fault = 0;
    struct Sum {
        double s=0.0,c=0.0;
        void add(double x) { const double t=s+x; c+=std::abs(s)>=std::abs(x)?(s-t)+x:(x-t)+s; s=t; }
        double value() const { return s+c; }
    };
    double integral(U a,U b,U c,U d) const {
        const U m=o+n;
        return eri[((a*m+b)*m+c)*m+d];
    }
    vibeqc::BoundedRestrictedPairCCSDParticleHoleProducer declaration(U extra_transient=0) const {
        using namespace vibeqc;
        if (!o || o>4 || !n || n>4 || maximum_rank>n || extra_transient>(1U<<20) || fault>9)
            throw std::length_error("split pair CCSD fixture exceeds tiny declaration bounds");
        BoundedRestrictedPairCCSDParticleHoleInventory inv{1,0,0,0,1};
        const auto p=plan_bounded_restricted_pair_ccsd_particle_hole(maximum_rank,o,maximum_rank,inv);
        BoundedRestrictedPairCCSDParticleHoleProducer out;
        out.produce=&produce; out.context=this; out.identity_sha256.fill('a');
        // Source K/J/O are generated one at a time; T is borrowed from Reader.
        out.maximum_transient_numerical_bytes_per_target=p.peak_owned_numerical_bytes
            +24*maximum_rank*maximum_rank+extra_transient;
        out.additional_control_storage_bytes=p.fixed_control_storage_bytes+sizeof(PairCCSDSplitFixture)
            +3*sizeof(std::vector<double>)+65536;
        out.maximum_work_units_per_target=p.work_units_upper_bound
            +4096*(o+1)*(n+1)*(n+1)*(maximum_rank+1)*(maximum_rank+1);
        return out;
    }
    static void produce(const Reader& reader,const Plan& admitted,U i,U j,U,
        Visitor visit,void* visitor_context,const void* context) {
        using namespace vibeqc;
        const auto& f=*static_cast<const PairCCSDSplitFixture*>(context);
        if (!admitted.split_bare_particle_hole || admitted.n_occupied!=f.o
            || admitted.common_virtual_dimension!=f.n)
            throw std::logic_error("split fixture received a wrong enclosing plan");
        if (f.fault==1) return; // Missing visit must be rejected even for rank zero.
        if (f.fault==6) throw std::runtime_error("split fixture injected producer exception");
        const auto target=reader.canonical_pair_view(i,j);
        const auto rank=target.rank;
        // This child inventory is a subset; the enclosing solver admits all
        // source arrays, Reader/ERI owners and simultaneous numerical phases.
        BoundedRestrictedPairCCSDParticleHoleInventory inv{1,0,24*rank*f.maximum_rank,0,1};
        const auto p=plan_bounded_restricted_pair_ccsd_particle_hole(rank,f.o,f.maximum_rank,inv);
        BoundedRestrictedPairCCSDParticleHoleCaps caps{rank,f.maximum_rank,f.o,
            p.maximum_borrowed_numerical_bytes,p.peak_owned_numerical_bytes,p.total_control_storage_bytes,
            p.per_replica_inventoried_bytes,p.required_node_inventoried_bytes,
            p.scalar_products_upper_bound,p.work_units_upper_bound};
        const auto snapshot=reader.snapshot_identity_sha256();
        const U result_i=f.fault==4?(i+1)%f.o:i;
        BoundedRestrictedPairCCSDParticleHoleAccumulator accumulator(rank,f.o,result_i,j,f.maximum_rank,inv,caps);
        for (U leg=0;leg<2;++leg) for (U m=0;m<f.o;++m) {
            const U left=leg?j:i, other=leg?i:j;
            const auto source=reader.canonical_pair_view(std::min(left,m),std::max(left,m));
            const U bdim=source.rank;
            std::vector<double> k(bdim*rank),exchange(bdim*rank),overlap(bdim*rank);
            for (U b=0;b<bdim;++b) for (U a=0;a<rank;++a) {
                Sum sk,sj,so;
                for (U x=0;x<f.n;++x) {
                    const double cb=source.coefficients.data[x*bdim+b];
                    so.add(cb*target.coefficients.data[x*rank+a]);
                    for (U y=0;y<f.n;++y) {
                        const double ca=target.coefficients.data[y*rank+a];
                        sk.add((cb*f.integral(m,f.o+x,other,f.o+y))*ca);
                        sj.add((cb*f.integral(m,other,f.o+x,f.o+y))*ca);
                    }
                }
                k[b*rank+a]=sk.value(); exchange[b*rank+a]=sj.value(); overlap[b*rank+a]=so.value();
            }
            accumulator.accumulate({leg,m,bdim,{k.data(),k.size()},
                {exchange.data(),exchange.size()},{overlap.data(),overlap.size()},source.amplitudes,left>m});
        }
        auto result=accumulator.finish();
        if (f.fault==5 && rank) // Deliberately violate the const contract, only in this negative fixture.
            const_cast<double*>(result.residual_data())[0]=std::numeric_limits<double>::quiet_NaN();
        if (f.fault==9 && rank)
            const_cast<double*>(result.residual_data())[0]=std::numeric_limits<double>::infinity();
        visit(result,f.fault==3?std::string(64,'b'):snapshot,visitor_context);
        if (f.fault==2) { // Catching the duplicate exception must not clear the visitor poison.
            try { visit(result,snapshot,visitor_context); } catch (const std::exception&) {}
        }
        if (f.fault==7 && rank) const_cast<double*>(target.amplitudes.data)[0]+=0.125;
        if (f.fault==8) f.original_fov[0]+=0.125;
    }
};
}

void bind_bounded_restricted_pair_ccsd_solver(py::module_& m) {
    using namespace vibeqc;
    using U = std::uint64_t;
    using Options = BoundedRestrictedPairCCSDSolverOptions;
    using Inventory = BoundedRestrictedPairCCSDSolverInventory;
    using Caps = BoundedRestrictedPairCCSDSolverCaps;
    using Plan = BoundedRestrictedPairCCSDSolverMemoryPlan;
    using Progress = BoundedRestrictedPairCCSDSolverProgress;
    using Result = BoundedRestrictedPairCCSDSolverResult;
    using Validation = BoundedRestrictedPairCCSDSolverPayloadValidationPlan;
    py::class_<Validation>(m, "_BoundedRestrictedPairCCSDSolverPayloadValidationPlan")
#define PAIR_CCSD_VALIDATION(f) .def_readonly(#f, &Validation::f)
        PAIR_CCSD_VALIDATION(n_occupied) PAIR_CCSD_VALIDATION(pair_count) PAIR_CCSD_VALIDATION(record_count)
        PAIR_CCSD_VALIDATION(singles_elements) PAIR_CCSD_VALIDATION(doubles_elements) PAIR_CCSD_VALIDATION(numerical_lanes)
        PAIR_CCSD_VALIDATION(retained_numerical_bytes) PAIR_CCSD_VALIDATION(retained_record_bytes)
        PAIR_CCSD_VALIDATION(payload_message_bytes) PAIR_CCSD_VALIDATION(fixed_codec_payload_bytes)
        PAIR_CCSD_VALIDATION(fixed_control_storage_bytes) PAIR_CCSD_VALIDATION(metadata_work_units)
        PAIR_CCSD_VALIDATION(payload_work_units) PAIR_CCSD_VALIDATION(work_units);
#undef PAIR_CCSD_VALIDATION
    const auto payload_guard=[](const Result& r) {
        if (r.memory().n_occupied>4 || r.memory().common_virtual_dimension>4
            || r.memory().amplitude_snapshot_bytes>(1U<<20))
            throw std::length_error("pair CCSD payload diagnostic exceeds tiny owner guards");
    };
    m.def("_plan_bounded_restricted_pair_ccsd_solver_payload_validation",
        [payload_guard](const Result& r) {
            payload_guard(r); return plan_bounded_restricted_pair_ccsd_solver_payload_validation(r);
        }, py::arg("result"));
    m.def("_verify_bounded_restricted_pair_ccsd_solver_payload",
        [payload_guard](const Result& r,U maximum_work_units) {
            payload_guard(r);
            if (maximum_work_units>1000000000ULL)
                throw std::length_error("pair CCSD payload diagnostic exceeds tiny work guard");
            verify_bounded_restricted_pair_ccsd_solver_payload(r,maximum_work_units);
        }, py::arg("result"), py::arg("maximum_work_units"));
    auto options = py::class_<Options>(m, "_BoundedRestrictedPairCCSDSolverOptions");
    options.def(py::init<>());
#define PAIR_CCSD_OPTION(f) options.def_readwrite(#f, &Options::f)
    PAIR_CCSD_OPTION(maximum_iterations); PAIR_CCSD_OPTION(denominator_floor);
    PAIR_CCSD_OPTION(singles_residual_tolerance); PAIR_CCSD_OPTION(doubles_residual_tolerance);
    PAIR_CCSD_OPTION(energy_tolerance); PAIR_CCSD_OPTION(coefficient_orthogonality_tolerance);
    PAIR_CCSD_OPTION(maximum_diagonal_update_antisymmetry_norm);
    PAIR_CCSD_OPTION(maximum_integral_work_units_per_call);
#undef PAIR_CCSD_OPTION
    auto inventory = py::class_<Inventory>(m, "_BoundedRestrictedPairCCSDSolverInventory");
    inventory.def(py::init<>());
#define PAIR_CCSD_INVENTORY(f) inventory.def_readwrite(#f, &Inventory::f)
    PAIR_CCSD_INVENTORY(numerical_replicas); PAIR_CCSD_INVENTORY(external_node_bytes);
    PAIR_CCSD_INVENTORY(other_live_bytes_per_replica); PAIR_CCSD_INVENTORY(fixed_backend_margin_bytes_per_replica);
#undef PAIR_CCSD_INVENTORY
    auto caps = py::class_<Caps>(m, "_BoundedRestrictedPairCCSDSolverCaps");
    caps.def(py::init<>());
#define PAIR_CCSD_CAP(f) caps.def_readwrite(#f, &Caps::f)
    PAIR_CCSD_CAP(maximum_occupied_count); PAIR_CCSD_CAP(maximum_common_virtual_dimension);
    PAIR_CCSD_CAP(maximum_pair_count); PAIR_CCSD_CAP(maximum_singles_rank); PAIR_CCSD_CAP(maximum_pair_rank);
    PAIR_CCSD_CAP(maximum_owned_numerical_bytes); PAIR_CCSD_CAP(maximum_per_replica_inventoried_bytes);
    PAIR_CCSD_CAP(maximum_node_inventoried_bytes); PAIR_CCSD_CAP(maximum_integral_calls);
    PAIR_CCSD_CAP(maximum_singles_calls); PAIR_CCSD_CAP(maximum_doubles_calls); PAIR_CCSD_CAP(maximum_work_units);
    PAIR_CCSD_CAP(maximum_particle_hole_calls);
#undef PAIR_CCSD_CAP
    auto plan = py::class_<Plan>(m, "_BoundedRestrictedPairCCSDSolverMemoryPlan");
#define PAIR_CCSD_PLAN(f) plan.def_readonly(#f, &Plan::f)
    PAIR_CCSD_PLAN(amplitudes); PAIR_CCSD_PLAN(target); PAIR_CCSD_PLAN(uniform_rank_upper_bound);
    PAIR_CCSD_PLAN(n_occupied); PAIR_CCSD_PLAN(common_virtual_dimension); PAIR_CCSD_PLAN(pair_count);
    PAIR_CCSD_PLAN(maximum_singles_rank); PAIR_CCSD_PLAN(maximum_pair_rank); PAIR_CCSD_PLAN(maximum_iterations);
    PAIR_CCSD_PLAN(total_singles_elements); PAIR_CCSD_PLAN(total_doubles_elements);
    PAIR_CCSD_PLAN(amplitude_snapshot_bytes); PAIR_CCSD_PLAN(candidate_snapshot_bytes);
    PAIR_CCSD_PLAN(projected_fock_diagonal_bytes); PAIR_CCSD_PLAN(retained_record_bytes);
    PAIR_CCSD_PLAN(owned_snapshot_table_bytes); PAIR_CCSD_PLAN(borrowed_input_table_bytes);
    PAIR_CCSD_PLAN(borrowed_initial_numerical_bytes); PAIR_CCSD_PLAN(borrowed_fock_bytes);
    PAIR_CCSD_PLAN(provider_retained_numerical_bytes); PAIR_CCSD_PLAN(provider_transient_numerical_bytes);
    PAIR_CCSD_PLAN(target_owned_peak_bytes); PAIR_CCSD_PLAN(output_numerical_bytes);
    PAIR_CCSD_PLAN(peak_owned_numerical_bytes); PAIR_CCSD_PLAN(control_storage_reservation_bytes);
    PAIR_CCSD_PLAN(per_replica_inventoried_bytes); PAIR_CCSD_PLAN(required_node_inventoried_bytes);
    PAIR_CCSD_PLAN(metadata_work_units); PAIR_CCSD_PLAN(validation_work_units); PAIR_CCSD_PLAN(initialization_work_units);
    PAIR_CCSD_PLAN(projection_work_units_per_snapshot); PAIR_CCSD_PLAN(work_units_per_snapshot);
    PAIR_CCSD_PLAN(target_evaluations_upper_bound); PAIR_CCSD_PLAN(integral_calls_upper_bound);
    PAIR_CCSD_PLAN(singles_calls_upper_bound); PAIR_CCSD_PLAN(doubles_calls_upper_bound); PAIR_CCSD_PLAN(work_units_upper_bound);
    PAIR_CCSD_PLAN(split_bare_particle_hole); PAIR_CCSD_PLAN(particle_hole_retained_numerical_bytes);
    PAIR_CCSD_PLAN(particle_hole_transient_numerical_bytes); PAIR_CCSD_PLAN(particle_hole_control_storage_bytes);
    PAIR_CCSD_PLAN(particle_hole_calls_upper_bound); PAIR_CCSD_PLAN(remaining_target_phase_bytes);
    PAIR_CCSD_PLAN(local_particle_hole_phase_bytes); PAIR_CCSD_PLAN(particle_hole_work_units_per_target);
    PAIR_CCSD_PLAN(particle_hole_guard_work_units_per_target); PAIR_CCSD_PLAN(omitted_bare_integral_calls_upper_bound);
#undef PAIR_CCSD_PLAN
    auto progress = py::class_<Progress>(m, "_BoundedRestrictedPairCCSDSolverProgress");
#define PAIR_CCSD_PROGRESS(f) progress.def_readonly(#f, &Progress::f)
    PAIR_CCSD_PROGRESS(iteration); PAIR_CCSD_PROGRESS(target_evaluations); PAIR_CCSD_PROGRESS(integral_calls);
    PAIR_CCSD_PROGRESS(singles_calls); PAIR_CCSD_PROGRESS(doubles_calls); PAIR_CCSD_PROGRESS(input_checks);
    PAIR_CCSD_PROGRESS(charged_work_units); PAIR_CCSD_PROGRESS(has_previous_energy); PAIR_CCSD_PROGRESS(converged);
    PAIR_CCSD_PROGRESS(correlation_energy); PAIR_CCSD_PROGRESS(energy_change);
    PAIR_CCSD_PROGRESS(singles_max_residual); PAIR_CCSD_PROGRESS(doubles_max_residual);
    PAIR_CCSD_PROGRESS(singles_residual_norm); PAIR_CCSD_PROGRESS(doubles_residual_norm);
    PAIR_CCSD_PROGRESS(maximum_diagonal_update_antisymmetry_norm);
    PAIR_CCSD_PROGRESS(particle_hole_calls); PAIR_CCSD_PROGRESS(particle_hole_visits);
    PAIR_CCSD_PROGRESS(particle_hole_source_slots); PAIR_CCSD_PROGRESS(charged_particle_hole_work_units);
#undef PAIR_CCSD_PROGRESS
    py::class_<Result>(m, "_BoundedRestrictedPairCCSDSolverResult")
        .def_property_readonly("memory", [](const Result& r) { return r.memory(); })
        .def_property_readonly("final_snapshot", [](const Result& r) { return r.final_snapshot(); })
        .def_property_readonly("minimum_denominator", &Result::minimum_denominator)
        .def_property_readonly("maximum_denominator", &Result::maximum_denominator)
        .def_property_readonly("input_identity_sha256", &Result::input_identity_sha256)
        .def_property_readonly("payload_sha256", &Result::payload_sha256)
        .def_property_readonly("split_execution_identity_sha256", &Result::split_execution_identity_sha256)
        .def_property_readonly("consumed_particle_hole_identity_sha256", &Result::consumed_particle_hole_identity_sha256)
        .def_property_readonly("particle_hole_physical_source_certified", &Result::particle_hole_physical_source_certified)
        .def("singles_rank", &Result::singles_rank).def("pair_rank", &Result::pair_rank)
        .def("singles_amplitude", &Result::singles_amplitude).def("doubles_amplitude", &Result::doubles_amplitude)
        .def("singles_copy", [](const Result& r, U i) {
            const auto rank = r.singles_rank(i);
            if (rank > 4) throw std::length_error("pair CCSD diagnostic singles copy exceeds tiny rank");
            const auto amplitudes = r.stored_singles_view(i);
            py::array_t<double> out(static_cast<py::ssize_t>(rank));
            for (U a = 0; a < rank; ++a) out.mutable_data()[a] = amplitudes.data[a];
            return out;
        })
        .def("pair_copy", [](const Result& r, U i, U j) {
            const auto rank = r.pair_rank(i, j);
            if (rank > 4) throw std::length_error("pair CCSD diagnostic pair copy exceeds tiny rank");
            const auto amplitudes = r.stored_pair_view(std::min(i, j), std::max(i, j));
            py::array_t<double> out({static_cast<py::ssize_t>(rank), static_cast<py::ssize_t>(rank)});
            for (U a = 0; a < rank; ++a) for (U b = 0; b < rank; ++b)
                out.mutable_data()[a*rank+b] = amplitudes.data[i <= j ? a*rank+b : b*rank+a];
            return out;
        });
    m.def("_plan_bounded_restricted_pair_ccsd_solver_upper", [](U o,U n,U s,U p,
        Options options,Inventory inv,U retained,U transient) {
        return plan_bounded_restricted_pair_ccsd_solver_upper(o,n,s,p,options,inv,retained,transient);
    },
        py::arg("n_occupied"), py::arg("common_virtual_dimension"), py::arg("maximum_singles_rank"),
        py::arg("maximum_pair_rank"), py::arg("options"), py::arg("inventory"),
        py::arg("integral_retained_bytes") = 0, py::arg("integral_transient_bytes") = 0);
    const auto dispatch = [](py::array foo, py::array fvv, py::array fov,
        py::list singles_c, py::list singles_t, py::list pair_c, py::list pair_t,
        py::array integrals, Options options, Inventory inventory, Caps caps,
        py::object callback, U fail_before_call, U transient, bool planning,
        bool split=false,U fault=0,U extra_ph_transient=0) -> py::object {
        const auto array = [](py::handle object, int dimensions) {
            if (!py::isinstance<py::array>(object))
                throw std::invalid_argument("pair CCSD diagnostic requires existing arrays");
            auto a = py::reinterpret_borrow<py::array>(object);
            if (!a.dtype().is(py::dtype::of<double>()) || !(a.flags() & py::array::c_style)
                || a.ndim() != dimensions || (a.size() && reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double)))
                throw std::invalid_argument("pair CCSD diagnostic requires aligned contiguous float64 arrays");
            return a;
        };
        foo = array(foo, 2); fvv = array(fvv, 2); fov = array(fov, 2); integrals = array(integrals, 4);
        if (foo.shape(0) < 1 || foo.shape(0) > 4 || fvv.shape(0) < 1 || fvv.shape(0) > 4
            || foo.shape(0) != foo.shape(1) || fvv.shape(0) != fvv.shape(1)
            || options.maximum_iterations > 128 || caps.maximum_owned_numerical_bytes > (1U << 20)
            || caps.maximum_node_inventoried_bytes > (128U << 20)
            || caps.maximum_work_units > 1000000000000000ULL || transient > (1U << 20))
            throw std::length_error("pair CCSD diagnostic exceeds tiny dimension, memory or work limits");
        if (options.maximum_integral_work_units_per_call != 1)
            throw std::invalid_argument("pair CCSD dense diagnostic integral work declaration must equal one");
        const auto o = static_cast<U>(foo.shape(0)), n = static_cast<U>(fvv.shape(0)), count = o*(o+1)/2;
        if (fov.shape(0) != static_cast<py::ssize_t>(o) || fov.shape(1) != static_cast<py::ssize_t>(n)
            || singles_c.size() != o || singles_t.size() != o || pair_c.size() != count || pair_t.size() != count)
            throw std::invalid_argument("pair CCSD diagnostic needs consistent Fov and complete singles/unordered pair tables");
        for (int axis = 0; axis < 4; ++axis)
            if (integrals.shape(axis) != static_cast<py::ssize_t>(o+n))
                throw std::invalid_argument("pair CCSD diagnostic integral shape mismatch");
        const auto view = [](const py::array& a) { return BoundedRestrictedCCSDRealView{
            static_cast<const double*>(a.data()), static_cast<std::size_t>(a.size())}; };
        // Every input owner survives even when a Python callback replaces all
        // entries of the original lists. Fixed tiny control, no numeric copy.
        std::array<py::array, 28> owners;
        std::array<BoundedRestrictedPairCCSDSinglesView, 4> singles{};
        std::array<BoundedRestrictedPairCCSDPairView, 10> pairs{};
        for (U i = 0; i < o; ++i) {
            owners[2*i] = array(singles_c[i], 2); owners[2*i+1] = array(singles_t[i], 1);
            const auto& c = owners[2*i]; const auto& t = owners[2*i+1];
            const auto rank = t.shape(0);
            if (rank > static_cast<py::ssize_t>(n) || c.shape(0) != static_cast<py::ssize_t>(n)
                || c.shape(1) != rank)
                throw std::invalid_argument("pair CCSD diagnostic singles shapes are inconsistent");
            singles[i] = {static_cast<U>(rank), view(c), view(t)};
        }
        for (U k = 0; k < count; ++k) {
            const auto start = 2*o+2*k;
            owners[start] = array(pair_c[k], 2); owners[start+1] = array(pair_t[k], 2);
            const auto& c = owners[start]; const auto& t = owners[start+1];
            const auto rank = t.shape(0);
            if (rank > static_cast<py::ssize_t>(n) || t.shape(1) != rank
                || c.shape(0) != static_cast<py::ssize_t>(n) || c.shape(1) != rank)
                throw std::invalid_argument("pair CCSD diagnostic pair shapes are inconsistent");
            pairs[k] = {static_cast<U>(rank), view(c), view(t)};
        }
        const BoundedRestrictedPairCCSDSolverInput input{
            {o, n, singles.data(), static_cast<std::size_t>(o), pairs.data(), static_cast<std::size_t>(count)},
            view(foo), view(fvv), view(fov)};
        U maximum_rank=0;
        for (U k=0;k<count;++k) maximum_rank=std::max(maximum_rank,pairs[k].rank);
        PairCCSDSplitFixture fixture{static_cast<const double*>(integrals.data()),
            nullptr,o,n,maximum_rank,fault};
        if (split && fault==8) fixture.original_fov=static_cast<double*>(fov.mutable_data());
        const auto ph=split?fixture.declaration(extra_ph_transient):BoundedRestrictedPairCCSDParticleHoleProducer{};
        if (planning) return py::cast(split?plan_bounded_restricted_pair_ccsd_solver(
            input,options,inventory,caps,ph,static_cast<U>(integrals.nbytes()),transient):
            plan_bounded_restricted_pair_ccsd_solver(input,options,inventory,caps,
                static_cast<U>(integrals.nbytes()),transient));
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("pair CCSD progress must be callable or None");
        struct Direct {
            const double* values;
            U n, fail_before, count = 0;
            static double value(U p, U q, U r, U s, void* context) {
                auto& d = *static_cast<Direct*>(context);
                if (d.count == d.fail_before)
                    throw std::runtime_error("pair CCSD diagnostic injected integral callback failure");
                ++d.count;
                if (p >= d.n || q >= d.n || r >= d.n || s >= d.n)
                    throw std::out_of_range("pair CCSD diagnostic integral label out of range");
                return d.values[((p*d.n+q)*d.n+r)*d.n+s];
            }
        } direct{static_cast<const double*>(integrals.data()), o+n, fail_before_call};
        const BoundedRestrictedCCSDIntegralProvider provider{
            &Direct::value, &direct, static_cast<U>(integrals.nbytes()), transient};
        const auto notify = [](const Progress& p, void* context) {
            (*static_cast<py::object*>(context))(py::cast(Progress(p)));
        };
        return py::cast(split?bounded_restricted_pair_ccsd_solve(input,provider,ph,options,inventory,caps,
            callback.is_none()?nullptr:+notify,callback.is_none()?nullptr:&callback):
            bounded_restricted_pair_ccsd_solve(input,provider,options,inventory,caps,
                callback.is_none()?nullptr:+notify,callback.is_none()?nullptr:&callback));
    };
    m.def("_plan_bounded_restricted_pair_ccsd_solver", [dispatch](
        py::array foo, py::array fvv, py::array fov, py::list sc, py::list st, py::list pc, py::list pt,
        py::array eri, Options options, Inventory inventory, Caps caps, U transient) {
        return dispatch(foo,fvv,fov,sc,st,pc,pt,eri,options,inventory,caps,py::none(),0,transient,true);
    }, py::arg("f_oo").noconvert(), py::arg("f_vv").noconvert(), py::arg("f_ov").noconvert(),
       py::arg("singles_coefficients"), py::arg("initial_singles"), py::arg("pair_coefficients"),
       py::arg("initial_pairs"), py::arg("integrals").noconvert(), py::arg("options"),
       py::arg("inventory"), py::arg("caps"), py::arg("provider_transient_bytes") = 0);
    m.def("_bounded_restricted_pair_ccsd_solve_diagnostic", [dispatch](
        py::array foo, py::array fvv, py::array fov, py::list sc, py::list st, py::list pc, py::list pt,
        py::array eri, Options options, Inventory inventory, Caps caps, py::object callback, U fail, U transient) {
        return dispatch(foo,fvv,fov,sc,st,pc,pt,eri,options,inventory,caps,callback,fail,transient,false);
    }, py::arg("f_oo").noconvert(), py::arg("f_vv").noconvert(), py::arg("f_ov").noconvert(),
       py::arg("singles_coefficients"), py::arg("initial_singles"), py::arg("pair_coefficients"),
       py::arg("initial_pairs"), py::arg("integrals").noconvert(), py::arg("options"),
       py::arg("inventory"), py::arg("caps"), py::arg("callback") = py::none(),
       py::arg("fail_before_call") = std::numeric_limits<U>::max(), py::arg("provider_transient_bytes") = 0);
    m.def("_plan_bounded_restricted_pair_ccsd_solver_split_upper", [](U o,U n,U s,U p,
        Options options,Inventory inv,U retained,U transient,U extra) {
        PairCCSDSplitFixture fixture{nullptr,nullptr,o,n,p,0};
        const auto producer=fixture.declaration(extra);
        return plan_bounded_restricted_pair_ccsd_solver_upper(o,n,s,p,options,inv,producer,retained,transient);
    }, py::arg("n_occupied"),py::arg("common_virtual_dimension"),py::arg("maximum_singles_rank"),
       py::arg("maximum_pair_rank"),py::arg("options"),py::arg("inventory"),
       py::arg("integral_retained_bytes")=0,py::arg("integral_transient_bytes")=0,
       py::arg("extra_particle_hole_transient_bytes")=0);
    m.def("_plan_bounded_restricted_pair_ccsd_solver_split", [dispatch](
        py::array foo,py::array fvv,py::array fov,py::list sc,py::list st,py::list pc,py::list pt,
        py::array eri,Options options,Inventory inventory,Caps caps,U transient,U extra) {
        return dispatch(foo,fvv,fov,sc,st,pc,pt,eri,options,inventory,caps,py::none(),0,transient,true,true,0,extra);
    },py::arg("f_oo").noconvert(),py::arg("f_vv").noconvert(),py::arg("f_ov").noconvert(),
      py::arg("singles_coefficients"),py::arg("initial_singles"),py::arg("pair_coefficients"),
      py::arg("initial_pairs"),py::arg("integrals").noconvert(),py::arg("options"),
      py::arg("inventory"),py::arg("caps"),py::arg("provider_transient_bytes")=0,
      py::arg("extra_particle_hole_transient_bytes")=0);
    m.def("_bounded_restricted_pair_ccsd_solve_split_diagnostic", [dispatch](
        py::array foo,py::array fvv,py::array fov,py::list sc,py::list st,py::list pc,py::list pt,
        py::array eri,Options options,Inventory inventory,Caps caps,py::object callback,U fail,U transient,U fault,U extra) {
        return dispatch(foo,fvv,fov,sc,st,pc,pt,eri,options,inventory,caps,callback,fail,transient,false,true,fault,extra);
    },py::arg("f_oo").noconvert(),py::arg("f_vv").noconvert(),py::arg("f_ov").noconvert(),
      py::arg("singles_coefficients"),py::arg("initial_singles"),py::arg("pair_coefficients"),
      py::arg("initial_pairs"),py::arg("integrals").noconvert(),py::arg("options"),
      py::arg("inventory"),py::arg("caps"),py::arg("callback")=py::none(),
      py::arg("fail_before_call")=std::numeric_limits<U>::max(),py::arg("provider_transient_bytes")=0,
      py::arg("native_producer_fault")=0,py::arg("extra_particle_hole_transient_bytes")=0);
}
