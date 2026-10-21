#include "vibeqc/manta_periodic_ks_grid.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_periodic_quadrature.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
using Complex=std::complex<double>;
void check(const py::array& a,int ndim) {
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=ndim || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic KS sources require contiguous float64 arrays");
}
Count vectors(const py::array& a) {
    check(a,2);if(a.shape(1)!=3) throw py::value_error("MANTA periodic KS vectors require shape (count,3)");return a.shape(0);
}
double item(const py::array& a,py::ssize_t offset) {
    double out;std::memcpy(&out,static_cast<const char*>(a.data())+offset,sizeof(double));return out;
}
std::int64_t boundary(const py::array& a,Count i) {
    std::int64_t out;std::memcpy(&out,static_cast<const char*>(a.data())+i*a.strides(0),sizeof(out));return out;
}
template<class Matrix> Matrix copy(const py::array& a) {
    Matrix out(a.shape(0),a.shape(1));
    for(Eigen::Index i=0;i<out.rows();++i) for(Eigen::Index j=0;j<out.cols();++j)
        out(i,j)=item(a,i*a.strides(0)+j*a.strides(1));return out;
}
template<class T> py::array_t<T> owned(const T* data,std::vector<py::ssize_t> shape) {
    py::array_t<T> out(shape);std::memcpy(out.mutable_data(),data,out.nbytes());return out;
}
}
void bind_manta_periodic_ks(py::module_& m) {
    auto o=py::class_<PeriodicKSOptions>(m,"PeriodicKSOptions").def(py::init<>());
#define O(name) o.def_readwrite(#name,&PeriodicKSOptions::name)
    O(scf);O(alpha);O(nuclear_cutoff_bohr);O(confirmation_fock_tolerance);O(confirmation_energy_tolerance);
    O(reciprocal_batch);O(grid_batch);O(pair_cutoff_bohr);
#undef O
    py::class_<PeriodicGlobalHybridKSOptions>(m,"PeriodicGlobalHybridKSOptions").def(py::init<>())
        .def_readwrite("ks",&PeriodicGlobalHybridKSOptions::ks)
        .def_readwrite("exchange_q0",&PeriodicGlobalHybridKSOptions::exchange_q0)
        .def_readwrite("madelung_alpha",&PeriodicGlobalHybridKSOptions::madelung_alpha)
        .def_readwrite("madelung_tolerance",&PeriodicGlobalHybridKSOptions::madelung_tolerance);
    py::class_<PeriodicRangeSeparatedKSOptions>(m,"PeriodicRangeSeparatedKSOptions").def(py::init<>())
        .def_readwrite("ks",&PeriodicRangeSeparatedKSOptions::ks)
        .def_readwrite("exchange_q0",&PeriodicRangeSeparatedKSOptions::exchange_q0)
        .def_readwrite("madelung_alpha",&PeriodicRangeSeparatedKSOptions::madelung_alpha)
        .def_readwrite("madelung_tolerance",&PeriodicRangeSeparatedKSOptions::madelung_tolerance);
    py::class_<PeriodicRIKSOptions>(m,"PeriodicRIKSOptions").def(py::init<>())
        .def_readwrite("ks",&PeriodicRIKSOptions::ks)
        .def_readwrite("fitting",&PeriodicRIKSOptions::fitting).def_readwrite("ri",&PeriodicRIKSOptions::ri)
        .def_readwrite("ri_accuracy",&PeriodicRIKSOptions::ri_accuracy)
        .def_readwrite("fitting_accuracy",&PeriodicRIKSOptions::fitting_accuracy)
        .def_readwrite("one_electron_accuracy",&PeriodicRIKSOptions::one_electron_accuracy)
        .def_readwrite("exchange_accuracy",&PeriodicRIKSOptions::exchange_accuracy)
        .def_readwrite("nuclear_accuracy",&PeriodicRIKSOptions::nuclear_accuracy);
    py::class_<PeriodicRIGlobalHybridKSOptions>(m,"PeriodicRIGlobalHybridKSOptions").def(py::init<>())
        .def_readwrite("ks",&PeriodicRIGlobalHybridKSOptions::ks)
        .def_readwrite("exchange_q0",&PeriodicRIGlobalHybridKSOptions::exchange_q0)
        .def_readwrite("madelung_alpha",&PeriodicRIGlobalHybridKSOptions::madelung_alpha)
        .def_readwrite("madelung_tolerance",&PeriodicRIGlobalHybridKSOptions::madelung_tolerance)
        .def_readwrite("fitting",&PeriodicRIGlobalHybridKSOptions::fitting).def_readwrite("ri",&PeriodicRIGlobalHybridKSOptions::ri)
        .def_readwrite("ri_accuracy",&PeriodicRIGlobalHybridKSOptions::ri_accuracy)
        .def_readwrite("fitting_accuracy",&PeriodicRIGlobalHybridKSOptions::fitting_accuracy)
        .def_readwrite("one_electron_accuracy",&PeriodicRIGlobalHybridKSOptions::one_electron_accuracy)
        .def_readwrite("exchange_accuracy",&PeriodicRIGlobalHybridKSOptions::exchange_accuracy)
        .def_readwrite("nuclear_accuracy",&PeriodicRIGlobalHybridKSOptions::nuclear_accuracy)
        .def_readwrite("coulomb_algorithm",&PeriodicRIGlobalHybridKSOptions::coulomb_algorithm)
        .def_readwrite("exchange_algorithm",&PeriodicRIGlobalHybridKSOptions::exchange_algorithm);
    py::class_<PeriodicRIRangeSeparatedKSOptions>(m,"PeriodicRIRangeSeparatedKSOptions").def(py::init<>())
        .def_readwrite("ks",&PeriodicRIRangeSeparatedKSOptions::ks)
        .def_readwrite("exchange_q0",&PeriodicRIRangeSeparatedKSOptions::exchange_q0)
        .def_readwrite("madelung_alpha",&PeriodicRIRangeSeparatedKSOptions::madelung_alpha)
        .def_readwrite("madelung_tolerance",&PeriodicRIRangeSeparatedKSOptions::madelung_tolerance)
        .def_readwrite("fitting",&PeriodicRIRangeSeparatedKSOptions::fitting).def_readwrite("ri",&PeriodicRIRangeSeparatedKSOptions::ri)
        .def_readwrite("ri_accuracy",&PeriodicRIRangeSeparatedKSOptions::ri_accuracy)
        .def_readwrite("fitting_accuracy",&PeriodicRIRangeSeparatedKSOptions::fitting_accuracy)
        .def_readwrite("one_electron_accuracy",&PeriodicRIRangeSeparatedKSOptions::one_electron_accuracy)
        .def_readwrite("exchange_accuracy",&PeriodicRIRangeSeparatedKSOptions::exchange_accuracy)
        .def_readwrite("nuclear_accuracy",&PeriodicRIRangeSeparatedKSOptions::nuclear_accuracy)
        .def_readwrite("coulomb_algorithm",&PeriodicRIRangeSeparatedKSOptions::coulomb_algorithm)
        .def_readwrite("exchange_algorithm",&PeriodicRIRangeSeparatedKSOptions::exchange_algorithm);
    py::class_<PeriodicKSExchangeComponentPlan>(m,"PeriodicKSExchangeComponentPlan")
        .def_readonly("kernel",&PeriodicKSExchangeComponentPlan::kernel)
        .def_readonly("coefficient",&PeriodicKSExchangeComponentPlan::coefficient)
        .def_readonly("omega",&PeriodicKSExchangeComponentPlan::omega)
        .def_readonly("exchange",&PeriodicKSExchangeComponentPlan::exchange)
        .def_readonly("fitting",&PeriodicKSExchangeComponentPlan::fitting);
    py::class_<PeriodicKSExchangeComponentReceipt>(m,"PeriodicKSExchangeComponentReceipt")
        .def_readonly("kernel",&PeriodicKSExchangeComponentReceipt::kernel)
        .def_readonly("coefficient",&PeriodicKSExchangeComponentReceipt::coefficient)
        .def_readonly("omega",&PeriodicKSExchangeComponentReceipt::omega)
        .def_readonly("energy",&PeriodicKSExchangeComponentReceipt::energy)
        .def_readonly("exchange",&PeriodicKSExchangeComponentReceipt::exchange)
        .def_readonly("fitting_info",&PeriodicKSExchangeComponentReceipt::fitting_info)
        .def_readonly("fitted_fock",&PeriodicKSExchangeComponentReceipt::fitted_fock);
    auto p=py::class_<PeriodicKSPlan>(m,"PeriodicKSPlan");
#define P(name) p.def_readonly(#name,&PeriodicKSPlan::name)
    P(overlap);P(kinetic);P(nuclear);P(hartree);P(fitting);P(xc);P(solver);P(density_envelope);P(fitting_accuracy);P(one_electron_accuracy);P(exchange_accuracy);P(nuclear_accuracy);P(spins);P(n_alpha);P(n_beta);P(n_basis);P(momenta);P(electrons);
    P(driver_live_bytes);P(provider_workspace_bytes);P(modeled_peak_bytes);P(fock_calls_upper_bound);P(shell_calls);
    P(primitive_pairs);P(primitive_points);P(primitive_work);P(scalar_terms);P(kernel_terms);
    P(nuclear_image_candidates_upper_bound);P(history_capacity);P(history_live_bytes);
    P(domain_live_bytes);P(domain_scalar_terms);P(domain_kernel_terms);P(quadrature);P(quadrature_live_bytes);
    P(exchange);P(madelung);P(exact_exchange_fraction);P(exchange_correction_kernel_terms_per_call);P(provider_kernel_terms_per_call);
    P(range_separated);P(exchange_components);P(long_range_exchange_fraction);P(short_range_exchange_fraction);
#undef P
    auto r=py::class_<PeriodicKSReceipt>(m,"PeriodicKSReceipt");
#define R(name) r.def_readonly(#name,&PeriodicKSReceipt::name)
    R(quadrature);R(grid_radial_points);R(grid_lebedev_order);R(grid_partition_radius_bohr);R(grid_stratmann);
    R(provider);R(algorithm);R(boundary);R(source_identity);R(overlap_identity);R(kinetic_identity);R(nuclear_identity);R(nuclear_energy_identity);
    R(domain_selection);R(domain_identity);R(reciprocal_energy_cutoff);R(xc_image_cutoff_bohr);
    R(domain_translations);R(domain_hartree_vectors);R(domain_xc_images);R(domain_conjugacy_vectors);
    R(domain_exchange_vectors);
    R(final_density_identity);R(final_fock_identity);R(hartree);R(xc);R(confirmed);R(nuclear_real);R(nuclear_reciprocal);R(nuclear_self);
    R(nuclear_background);R(nuclear_energy);R(one_electron_energy);R(coulomb_energy);R(xc_energy);R(energy);
    R(confirmation_fock_error);R(confirmation_energy_error);R(maximum_grid_population_error);
    R(nuclear_pair_images);R(fock_calls);R(shell_calls);R(primitive_pairs);R(primitive_points);R(modeled_peak_bytes);
    R(exchange);R(exchange_q0);R(exchange_correction_identity);R(exact_exchange_fraction);R(madelung);
    R(madelung_real_vectors);R(madelung_reciprocal_vectors);
    R(uncorrected_exchange_energy);R(exchange_correction_energy);R(exchange_energy);
    R(range_separated);R(exchange_components);R(long_range_exchange_fraction);R(short_range_exchange_fraction);R(exchange_q0_requested);
    R(accuracy_density);R(one_electron_accuracy);R(exchange_accuracy);R(nuclear_accuracy);R(coulomb_algorithm);R(exchange_algorithm);R(coulomb_identity);R(exchange_identity);R(fitting_info);R(fitted_fock);
#undef R
    py::class_<PeriodicKSResult>(m,"PeriodicKSResult")
        .def_readonly("plan",&PeriodicKSResult::plan).def_readonly("receipt",&PeriodicKSResult::receipt)
        .def_property_readonly("converged",&PeriodicKSResult::converged)
        .def_property_readonly("solver_snapshot",[](const PeriodicKSResult& r){return r.rks?py::cast(r.rks->final_snapshot(),py::return_value_policy::copy):py::cast(r.uks->final_snapshot(),py::return_value_policy::copy);})
        .def_property_readonly("solver_diagnostics",[](const PeriodicKSResult& r){return r.rks?r.rks->diagnostics():r.uks->diagnostics();})
        .def_property_readonly("solver_memory",[](const PeriodicKSResult& r){return r.rks?py::cast(r.rks->memory(),py::return_value_policy::copy):py::cast(r.uks->memory(),py::return_value_policy::copy);})
        .def_property_readonly("scf_history",[](const PeriodicKSResult& r){return r.rks?py::cast(r.rks_history,py::return_value_policy::copy):py::cast(r.uks_history,py::return_value_policy::copy);})
        .def_property_readonly("density",[](const PeriodicKSResult& r){const auto& p=r.plan;return owned(r.rks?r.rks->density_data():r.uks->density_data(),
            {static_cast<py::ssize_t>(p.spins),static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});})
        .def_property_readonly("fock",[](const PeriodicKSResult& r){const auto& p=r.plan;return owned(r.rks?r.rks->fock_data():r.uks->fock_data(),
            {static_cast<py::ssize_t>(p.spins),static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});})
        .def_property_readonly("coefficients",[](const PeriodicKSResult& r){const auto& p=r.plan;const auto rank=r.rks?r.rks->memory().retained_rank:r.uks->memory().retained_rank;
            return owned(r.rks?r.rks->coefficients_data():r.uks->coefficients_data(),{static_cast<py::ssize_t>(p.spins),static_cast<py::ssize_t>(p.momenta),
                static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(rank)});})
        .def_property_readonly("orbital_energies",[](const PeriodicKSResult& r){const auto& p=r.plan;const auto rank=r.rks?r.rks->memory().retained_rank:r.uks->memory().retained_rank;
            return owned(r.rks?r.rks->orbital_energies_data():r.uks->orbital_energies_data(),{static_cast<py::ssize_t>(p.spins),static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(rank)});})
        .def_property_readonly("overlap",[](const PeriodicKSResult& r){const auto& p=r.plan;return owned(r.overlap.data(),
            {static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});})
        .def_property_readonly("hcore",[](const PeriodicKSResult& r){const auto& p=r.plan;return owned(r.hcore.data(),
            {static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});});
    m.def("plan_periodic_ks",&plan_periodic_ks,py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),
        py::arg("reciprocal_vectors"),py::arg("cells"),py::arg("xc_images"),py::arg("grid_points"),
        py::arg("options")=PeriodicKSOptions{},py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ks",[](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,const vibeqc::RegularKMesh& mesh,
        const vibeqc::Functional& f,const py::array& g,const py::array& cells,const py::array& images,const py::array& points,const py::array& qw,
        const PeriodicKSOptions& o,const ResourceBudget& budget) {
        const Count ng=vectors(g),nc=vectors(cells),ni=vectors(images),np=vectors(points);check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=np) throw py::value_error("MANTA periodic KS grid weight extent mismatch");
        plan_periodic_ks(basis,system,mesh,f,ng,nc,ni,np,o,budget);
        // All array payload copies follow complete count-only admission.
        // Borrowed Functional/system/basis and arrays remain immutable; retain GIL.
        const auto reciprocal=copy<md::Matrix>(g),translations=copy<md::Matrix>(cells);
        const auto grid_images=copy<Eigen::MatrixXd>(images),grid=copy<Eigen::MatrixXd>(points);
        std::vector<double> weights(np);for(Count i=0;i<np;++i) weights[i]=item(qw,i*qw.strides(0));
        return run_periodic_ks(basis,system,mesh,f,reciprocal,translations,grid_images,grid,weights,o,budget);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_vectors").noconvert(),
      py::arg("translations").noconvert(),py::arg("xc_images").noconvert(),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options")=PeriodicKSOptions{},py::arg("budget")=ResourceBudget{});

    m.def("plan_periodic_ri_ks",&plan_periodic_ri_ks,py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),
        py::arg("reciprocal_vectors"),py::arg("cells"),py::arg("xc_images"),py::arg("grid_points"),
        py::arg("options")=PeriodicRIKSOptions{},py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ri_ks",[](const vibeqc::BasisSet& basis,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,const vibeqc::Functional& f,const py::array& g,const py::array& cells,
        const py::array& images,const py::array& points,const py::array& qw,const PeriodicRIKSOptions& o,const ResourceBudget& budget) {
        const Count ng=vectors(g),nc=vectors(cells),ni=vectors(images),np=vectors(points);check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=np) throw py::value_error("MANTA RI KS grid weight extent mismatch");
        plan_periodic_ri_ks(basis,auxiliary,system,mesh,f,ng,nc,ni,np,o,budget);
        const auto reciprocal=copy<md::Matrix>(g),translations=copy<md::Matrix>(cells);
        const auto grid_images=copy<Eigen::MatrixXd>(images),grid=copy<Eigen::MatrixXd>(points);
        std::vector<double> weights(np);for(Count i=0;i<np;++i) weights[i]=item(qw,i*qw.strides(0));
        return run_periodic_ri_ks(basis,auxiliary,system,mesh,f,reciprocal,translations,grid_images,grid,weights,o,budget);
    },py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_vectors").noconvert(),
      py::arg("translations").noconvert(),py::arg("xc_images").noconvert(),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options")=PeriodicRIKSOptions{},py::arg("budget")=ResourceBudget{});
    auto bind_ri_hybrid=[&](auto defaults,auto planner,auto runner,const char* plan_name,const char* run_name) {
    using Options=decltype(defaults);
    m.def(plan_name,planner,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),
        py::arg("reciprocal_vectors"),py::arg("exchange_vectors_per_pair"),py::arg("cells"),py::arg("xc_images"),py::arg("grid_points"),
        py::arg("options")=defaults,py::arg("budget")=ResourceBudget{});
    m.def(run_name,[planner,runner](const vibeqc::BasisSet& basis,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,const vibeqc::Functional& f,const py::array& g,const py::array& q,const py::array& offsets,
        const py::array& cells,const py::array& images,const py::array& points,const py::array& qw,
        const Options& o,const ResourceBudget& budget) {
        using namespace admission;
        const Count ng=vectors(g),nq=vectors(q),nc=vectors(cells),ni=vectors(images),np=vectors(points),pairs=square(mesh.size());check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=np) throw py::value_error("MANTA RI hybrid KS grid weight extent mismatch");
        if(!offsets.dtype().is(py::dtype::of<std::int64_t>()) || offsets.ndim()!=1 || static_cast<Count>(offsets.shape(0))!=add(pairs,1)
           || !(offsets.flags()&(py::array::c_style|py::array::f_style)))
            throw py::value_error("MANTA RI hybrid KS offsets require contiguous int64 (nk*nk+1)");
        planner(basis,auxiliary,system,mesh,f,ng,o.exchange_algorithm=="ri" && mesh.size()>1?1:0,nc,ni,np,o,budget);
        Count maximum=0,previous=0;
        if(boundary(offsets,0)!=0) throw py::value_error("MANTA RI hybrid KS offsets must start at zero");
        for(Count i=1;i<=pairs;++i) {
            const auto next=boundary(offsets,i);
            if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                throw py::value_error("MANTA RI hybrid KS offsets must be monotone and bounded");
            maximum=std::max(maximum,static_cast<Count>(next)-previous);previous=next;
        }
        if(previous!=nq) throw py::value_error("MANTA RI hybrid KS offsets must end at the vector count");
        planner(basis,auxiliary,system,mesh,f,ng,maximum,nc,ni,np,o,budget);
        const auto reciprocal=copy<md::Matrix>(g),exchange=copy<md::Matrix>(q),translations=copy<md::Matrix>(cells);
        const auto grid_images=copy<Eigen::MatrixXd>(images),grid=copy<Eigen::MatrixXd>(points);
        std::vector<Count> bounds(pairs+1);for(Count i=0;i<=pairs;++i) bounds[i]=boundary(offsets,i);
        std::vector<double> weights(np);for(Count i=0;i<np;++i) weights[i]=item(qw,i*qw.strides(0));
        return runner(basis,auxiliary,system,mesh,f,reciprocal,exchange,bounds,translations,grid_images,grid,weights,o,budget);
    },py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_vectors").noconvert(),
      py::arg("exchange_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),py::arg("translations").noconvert(),
      py::arg("xc_images").noconvert(),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options")=defaults,py::arg("budget")=ResourceBudget{});
    };
    bind_ri_hybrid(PeriodicRIGlobalHybridKSOptions{},&plan_periodic_ri_global_hybrid_ks,&run_periodic_ri_global_hybrid_ks,
        "plan_periodic_ri_global_hybrid_ks","run_periodic_ri_global_hybrid_ks");
    bind_ri_hybrid(PeriodicRIRangeSeparatedKSOptions{},&plan_periodic_ri_range_separated_ks,&run_periodic_ri_range_separated_ks,
        "plan_periodic_ri_range_separated_ks","run_periodic_ri_range_separated_ks");

    auto bind_hybrid=[&](auto defaults,auto planner,auto runner,const char* plan_name,const char* run_name) {
    using Options=decltype(defaults);
    m.def(plan_name,planner,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),
        py::arg("reciprocal_vectors"),py::arg("exchange_vectors_per_pair"),py::arg("cells"),py::arg("xc_images"),py::arg("grid_points"),
        py::arg("options")=defaults,py::arg("budget")=ResourceBudget{});
    m.def(run_name,[planner,runner](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,const vibeqc::RegularKMesh& mesh,
        const vibeqc::Functional& f,const py::array& g,const py::array& q,const py::array& offsets,const py::array& cells,
        const py::array& images,const py::array& points,const py::array& qw,const Options& o,const ResourceBudget& budget) {
        using namespace admission;
        const Count ng=vectors(g),nq=vectors(q),nc=vectors(cells),ni=vectors(images),np=vectors(points),pairs=square(mesh.size());check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=np) throw py::value_error("MANTA hybrid KS grid weight extent mismatch");
        if(!offsets.dtype().is(py::dtype::of<std::int64_t>()) || offsets.ndim()!=1 || static_cast<Count>(offsets.shape(0))!=add(pairs,1)
           || !(offsets.flags()&(py::array::c_style|py::array::f_style)))
            throw py::value_error("MANTA hybrid KS offsets require contiguous int64 (nk*nk+1)");
        planner(basis,system,mesh,f,ng,0,nc,ni,np,o,budget);
        Count maximum=0,previous=0;
        if(boundary(offsets,0)!=0) throw py::value_error("MANTA hybrid KS offsets must start at zero");
        for(Count i=1;i<=pairs;++i) {
            const auto next=boundary(offsets,i);
            if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                throw py::value_error("MANTA hybrid KS offsets must be monotone and bounded");
            maximum=std::max(maximum,static_cast<Count>(next)-previous);previous=next;
        }
        if(previous!=nq) throw py::value_error("MANTA hybrid KS offsets must end at the vector count");
        planner(basis,system,mesh,f,ng,maximum,nc,ni,np,o,budget);
        // Full admission precedes every numerical copy. Keep borrowed sources
        // immutable and retain the GIL throughout native evaluation.
        const auto reciprocal=copy<md::Matrix>(g),exchange=copy<md::Matrix>(q),translations=copy<md::Matrix>(cells);
        const auto grid_images=copy<Eigen::MatrixXd>(images),grid=copy<Eigen::MatrixXd>(points);
        std::vector<Count> bounds(pairs+1);for(Count i=0;i<=pairs;++i) bounds[i]=boundary(offsets,i);
        std::vector<double> weights(np);for(Count i=0;i<np;++i) weights[i]=item(qw,i*qw.strides(0));
        return runner(basis,system,mesh,f,reciprocal,exchange,bounds,translations,grid_images,grid,weights,o,budget);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_vectors").noconvert(),
      py::arg("exchange_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),py::arg("translations").noconvert(),
      py::arg("xc_images").noconvert(),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options")=defaults,py::arg("budget")=ResourceBudget{});
    };
    bind_hybrid(PeriodicGlobalHybridKSOptions{},&plan_periodic_global_hybrid_ks,&run_periodic_global_hybrid_ks,
        "plan_periodic_global_hybrid_ks","run_periodic_global_hybrid_ks");
    bind_hybrid(PeriodicRangeSeparatedKSOptions{},&plan_periodic_range_separated_ks,&run_periodic_range_separated_ks,
        "plan_periodic_range_separated_ks","run_periodic_range_separated_ks");

    auto dp=py::class_<PeriodicKSDomainPlan>(m,"PeriodicKSDomainPlan");
#define D(name) dp.def_readonly(#name,&PeriodicKSDomainPlan::name)
    D(ks);D(reciprocal_energy_cutoff);D(pair_cutoff_bohr);D(translation_radius_bohr);D(xc_image_cutoff_bohr);
    D(translation_candidates);D(hartree_candidates);D(xc_image_candidates);D(candidate_visits);
    D(domain_live_bytes);D(domain_scalar_terms);D(domain_kernel_terms);
    D(modeled_peak_bytes);D(shell_calls);D(primitive_work);D(scalar_terms);D(kernel_terms);
#undef D
    auto dr=py::class_<PeriodicKSDomainReceipt>(m,"PeriodicKSDomainReceipt");
#define D(name) dr.def_readonly(#name,&PeriodicKSDomainReceipt::name)
    D(identity);D(translations);D(hartree_vectors);D(xc_images);D(conjugacy_vectors);
#undef D
    auto matrix=[](const auto& a) {
        py::array_t<double> out({static_cast<py::ssize_t>(a.rows()),py::ssize_t{3}});
        for(Eigen::Index i=0;i<a.rows();++i)for(int j=0;j<3;++j)out.mutable_data()[3*i+j]=a(i,j);return out;
    };
    py::class_<PeriodicKSDomains>(m,"PeriodicKSDomains")
        .def_readonly("plan",&PeriodicKSDomains::plan).def_readonly("receipt",&PeriodicKSDomains::receipt)
        .def_property_readonly("translations",[matrix](const PeriodicKSDomains& d){return matrix(d.translations);})
        .def_property_readonly("reciprocal_vectors",[matrix](const PeriodicKSDomains& d){return matrix(d.reciprocal_vectors);})
        .def_property_readonly("xc_images",[matrix](const PeriodicKSDomains& d){return matrix(d.xc_images);});
    m.def("plan_periodic_ks_from_cutoffs",&plan_periodic_ks_from_cutoffs,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("build_periodic_ks_domains",&build_periodic_ks_domains,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ks_from_cutoffs",[](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,const vibeqc::Functional& f,double cutoff,double xc_radius,
        const py::array& points,const py::array& qw,const PeriodicKSOptions& o,const ResourceBudget& budget) {
        const Count n=vectors(points);check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=n)throw py::value_error("MANTA KS cutoff grid weight extent mismatch");
        plan_periodic_ks_from_cutoffs(basis,system,mesh,f,cutoff,xc_radius,n,o,budget);
        const auto grid=copy<Eigen::MatrixXd>(points);std::vector<double> weights(n);
        for(Count i=0;i<n;++i)weights[i]=item(qw,i*qw.strides(0));
        return run_periodic_ks_from_cutoffs(basis,system,mesh,f,cutoff,xc_radius,grid,weights,o,budget);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
      py::arg("xc_image_cutoff_bohr"),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options"),py::arg("budget")=ResourceBudget{});

    m.def("plan_periodic_ri_ks_from_cutoffs",&plan_periodic_ri_ks_from_cutoffs,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("build_periodic_ri_ks_domains",&build_periodic_ri_ks_domains,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ri_ks_from_cutoffs",[](const vibeqc::BasisSet& basis,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,const vibeqc::Functional& f,double cutoff,double xc_radius,
        const py::array& points,const py::array& qw,const PeriodicRIKSOptions& o,const ResourceBudget& budget) {
        const Count n=vectors(points);check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=n)throw py::value_error("MANTA KS cutoff grid weight extent mismatch");
        plan_periodic_ri_ks_from_cutoffs(basis,auxiliary,system,mesh,f,cutoff,xc_radius,n,o,budget);
        const auto grid=copy<Eigen::MatrixXd>(points);std::vector<double> weights(n);
        for(Count i=0;i<n;++i)weights[i]=item(qw,i*qw.strides(0));
        return run_periodic_ri_ks_from_cutoffs(basis,auxiliary,system,mesh,f,cutoff,xc_radius,grid,weights,o,budget);
    },py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
      py::arg("xc_image_cutoff_bohr"),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options"),py::arg("budget")=ResourceBudget{});

    auto bind_hybrid_domains=[&](auto options,auto planner,auto builder,auto runner,const std::string& type,const std::string& fn) {
    using Options=decltype(options);using DomainPlan=PeriodicHybridKSDomainPlan<Options>;
    using DomainReceipt=PeriodicHybridKSDomainReceipt<Options>;using Domains=PeriodicHybridKSDomains<Options>;
    auto hp=py::class_<DomainPlan>(m,(type+"DomainPlan").c_str());
#define H(name) hp.def_readonly(#name,&DomainPlan::name)
    H(ks);H(reciprocal_energy_cutoff);H(pair_cutoff_bohr);H(translation_radius_bohr);H(xc_image_cutoff_bohr);
    H(momenta);H(translation_candidates);H(hartree_candidates);H(xc_image_candidates);H(maximum_exchange_candidates);
    H(exchange_candidates);H(candidate_visits);H(domain_live_bytes);H(domain_scalar_terms);H(domain_kernel_terms);
    H(modeled_peak_bytes);H(shell_calls);H(primitive_pairs);H(primitive_points);H(primitive_work);H(scalar_terms);H(kernel_terms);
#undef H
    auto hr=py::class_<DomainReceipt>(m,(type+"DomainReceipt").c_str());
#define H(name) hr.def_readonly(#name,&DomainReceipt::name)
    H(identity);H(pair_identity);H(xc_image_identity);H(translations);H(hartree_vectors);H(exchange_vectors);H(xc_images);H(conjugacy_vectors);
#undef H
    py::class_<Domains>(m,(type+"Domains").c_str())
        .def_readonly("plan",&Domains::plan).def_readonly("receipt",&Domains::receipt)
        .def_property_readonly("translations",[matrix](const Domains& d){return matrix(d.translations);})
        .def_property_readonly("reciprocal_vectors",[matrix](const Domains& d){return matrix(d.reciprocal_vectors);})
        .def_property_readonly("exchange_vectors",[matrix](const Domains& d){return matrix(d.exchange_vectors);})
        .def_property_readonly("xc_images",[matrix](const Domains& d){return matrix(d.xc_images);})
        .def_property_readonly("transfer_offsets",[](const Domains& d){
            py::array_t<std::int64_t> out(d.transfer_offsets.size());
            for(Count i=0;i<d.transfer_offsets.size();++i)out.mutable_data()[i]=d.transfer_offsets[i];return out;
        });
    if constexpr(std::is_same_v<Options,PeriodicRIGlobalHybridKSOptions> ||
                 std::is_same_v<Options,PeriodicRIRangeSeparatedKSOptions>) {
    m.def(("plan_"+fn+"_from_cutoffs").c_str(),planner,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def(("build_"+fn+"_domains").c_str(),builder,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def(("run_"+fn+"_from_cutoffs").c_str(),[planner,runner](const vibeqc::BasisSet& basis,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,const vibeqc::Functional& f,double cutoff,double xc_radius,
        const py::array& points,const py::array& qw,const Options& o,const ResourceBudget& budget) {
        const Count n=vectors(points);check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=n)throw py::value_error("MANTA hybrid KS cutoff grid weight extent mismatch");
        planner(basis,auxiliary,system,mesh,f,cutoff,xc_radius,n,o,budget);
        const auto grid=copy<Eigen::MatrixXd>(points);std::vector<double> weights(n);
        for(Count i=0;i<n;++i)weights[i]=item(qw,i*qw.strides(0));
        return runner(basis,auxiliary,system,mesh,f,cutoff,xc_radius,grid,weights,o,budget);
    },py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
      py::arg("xc_image_cutoff_bohr"),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options"),py::arg("budget")=ResourceBudget{});
    } else {
    m.def(("plan_"+fn+"_from_cutoffs").c_str(),planner,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def(("build_"+fn+"_domains").c_str(),builder,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_points"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def(("run_"+fn+"_from_cutoffs").c_str(),[planner,runner](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const vibeqc::RegularKMesh& mesh,const vibeqc::Functional& f,double cutoff,double xc_radius,
        const py::array& points,const py::array& qw,const Options& o,const ResourceBudget& budget) {
        const Count n=vectors(points);check(qw,1);
        if(static_cast<Count>(qw.shape(0))!=n)throw py::value_error("MANTA hybrid KS cutoff grid weight extent mismatch");
        planner(basis,system,mesh,f,cutoff,xc_radius,n,o,budget);
        const auto grid=copy<Eigen::MatrixXd>(points);std::vector<double> weights(n);
        for(Count i=0;i<n;++i)weights[i]=item(qw,i*qw.strides(0));
        return runner(basis,system,mesh,f,cutoff,xc_radius,grid,weights,o,budget);
    },py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
      py::arg("xc_image_cutoff_bohr"),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("options"),py::arg("budget")=ResourceBudget{});
    }

    };
    bind_hybrid_domains(PeriodicGlobalHybridKSOptions{},&plan_periodic_global_hybrid_ks_from_cutoffs,
        &build_periodic_global_hybrid_ks_domains,&run_periodic_global_hybrid_ks_from_cutoffs,
        "PeriodicGlobalHybridKS","periodic_global_hybrid_ks");
    bind_hybrid_domains(PeriodicRangeSeparatedKSOptions{},&plan_periodic_range_separated_ks_from_cutoffs,
        &build_periodic_range_separated_ks_domains,&run_periodic_range_separated_ks_from_cutoffs,
        "PeriodicRangeSeparatedKS","periodic_range_separated_ks");

    bind_hybrid_domains(PeriodicRIGlobalHybridKSOptions{},&plan_periodic_ri_global_hybrid_ks_from_cutoffs,
        &build_periodic_ri_global_hybrid_ks_domains,&run_periodic_ri_global_hybrid_ks_from_cutoffs,
        "PeriodicRIGlobalHybridKS","periodic_ri_global_hybrid_ks");
    bind_hybrid_domains(PeriodicRIRangeSeparatedKSOptions{},&plan_periodic_ri_range_separated_ks_from_cutoffs,
        &build_periodic_ri_range_separated_ks_domains,&run_periodic_ri_range_separated_ks_from_cutoffs,
        "PeriodicRIRangeSeparatedKS","periodic_ri_range_separated_ks");

    auto qo=py::class_<PeriodicQuadratureOptions>(m,"PeriodicQuadratureOptions").def(py::init<>());
#define Q(name) qo.def_readwrite(#name,&PeriodicQuadratureOptions::name)
    Q(radial_points);Q(lebedev_order);Q(partition_radius_bohr);Q(stratmann);
#undef Q
    auto qp=py::class_<PeriodicQuadraturePlan>(m,"PeriodicQuadraturePlan");
#define Q(name) qp.def_readonly(#name,&PeriodicQuadraturePlan::name)
    Q(atoms);Q(angular_points);Q(grid_points);Q(threads);Q(atlas_candidates);Q(neighbor_candidates);Q(candidate_visits);
    Q(output_bytes);Q(workspace_bytes);Q(modeled_peak_bytes);Q(scalar_terms);Q(kernel_terms);
#undef Q
    auto qr=py::class_<PeriodicQuadratureReceipt>(m,"PeriodicQuadratureReceipt");
#define Q(name) qr.def_readonly(#name,&PeriodicQuadratureReceipt::name)
    Q(algorithm);Q(source_identity);Q(nonzero_weights);Q(weight_sum);Q(cell_volume);
#undef Q
    py::class_<PeriodicQuadratureResult>(m,"PeriodicQuadratureResult")
        .def_readonly("plan",&PeriodicQuadratureResult::plan).def_readonly("receipt",&PeriodicQuadratureResult::receipt)
        .def_property_readonly("points",[matrix](const PeriodicQuadratureResult& r){return matrix(r.grid.points);})
        .def_property_readonly("weights",[](const PeriodicQuadratureResult& r){return owned(r.grid.weights.data(),{r.grid.weights.size()});});
    m.def("plan_periodic_quadrature",&plan_periodic_quadrature,py::arg("system"),
        py::arg("options")=PeriodicQuadratureOptions{},py::arg("budget")=ResourceBudget{});
    m.def("build_periodic_quadrature",&build_periodic_quadrature,py::arg("system"),
        py::arg("options")=PeriodicQuadratureOptions{},py::arg("budget")=ResourceBudget{});

    auto gp=py::class_<PeriodicKSGridPlan>(m,"PeriodicKSGridPlan");
#define G(name) gp.def_readonly(#name,&PeriodicKSGridPlan::name)
    G(quadrature);G(domains);G(grid_live_bytes);G(modeled_peak_bytes);G(shell_calls);G(primitive_work);G(scalar_terms);G(kernel_terms);
#undef G
    m.def("plan_periodic_ks_with_quadrature",&plan_periodic_ks_with_quadrature,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ks_with_quadrature",&run_periodic_ks_with_quadrature,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});

    m.def("plan_periodic_ri_ks_with_quadrature",&plan_periodic_ri_ks_with_quadrature,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ri_ks_with_quadrature",&run_periodic_ri_ks_with_quadrature,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});

    auto bind_hybrid_grid=[&](auto options,auto planner,auto runner,const std::string& type,const std::string& fn) {
    using GridPlan=PeriodicHybridKSGridPlan<decltype(options)>;
    auto hgp=py::class_<GridPlan>(m,(type+"GridPlan").c_str());
#define G(name) hgp.def_readonly(#name,&GridPlan::name)
    G(quadrature);G(domains);G(grid_live_bytes);G(modeled_peak_bytes);G(shell_calls);G(primitive_work);G(scalar_terms);G(kernel_terms);
#undef G
    if constexpr(std::is_same_v<decltype(options),PeriodicRIGlobalHybridKSOptions> ||
                 std::is_same_v<decltype(options),PeriodicRIRangeSeparatedKSOptions>) {
    m.def(("plan_"+fn+"_with_quadrature").c_str(),planner,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def(("run_"+fn+"_with_quadrature").c_str(),runner,
        py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    } else {
    m.def(("plan_"+fn+"_with_quadrature").c_str(),planner,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def(("run_"+fn+"_with_quadrature").c_str(),runner,
        py::arg("basis"),py::arg("system"),py::arg("mesh"),py::arg("functional"),py::arg("reciprocal_energy_cutoff"),
        py::arg("xc_image_cutoff_bohr"),py::arg("grid_options"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    }
    };
    bind_hybrid_grid(PeriodicGlobalHybridKSOptions{},&plan_periodic_global_hybrid_ks_with_quadrature,
        &run_periodic_global_hybrid_ks_with_quadrature,"PeriodicGlobalHybridKS","periodic_global_hybrid_ks");
    bind_hybrid_grid(PeriodicRangeSeparatedKSOptions{},&plan_periodic_range_separated_ks_with_quadrature,
        &run_periodic_range_separated_ks_with_quadrature,"PeriodicRangeSeparatedKS","periodic_range_separated_ks");
    bind_hybrid_grid(PeriodicRIGlobalHybridKSOptions{},&plan_periodic_ri_global_hybrid_ks_with_quadrature,
        &run_periodic_ri_global_hybrid_ks_with_quadrature,"PeriodicRIGlobalHybridKS","periodic_ri_global_hybrid_ks");
    bind_hybrid_grid(PeriodicRIRangeSeparatedKSOptions{},&plan_periodic_ri_range_separated_ks_with_quadrature,
        &run_periodic_ri_range_separated_ks_with_quadrature,"PeriodicRIRangeSeparatedKS","periodic_ri_range_separated_ks");
}
