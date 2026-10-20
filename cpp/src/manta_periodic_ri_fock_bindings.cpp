#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/manta_admission.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
using Complex=std::complex<double>;
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T result;std::memcpy(&result,static_cast<const char*>(a.data())+offset,sizeof(T));return result;
}
bool contiguous(const py::array& a) {return a.flags()&(py::array::c_style|py::array::f_style);}
Count vectors(const py::array& a) {
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=2 || a.shape(1)!=3 || !contiguous(a))
        throw py::value_error("MANTA RI Fock vectors require C- or F-contiguous float64 (n,3) arrays");
    return a.shape(0);
}
md::Matrix copy_vectors(const py::array& a) {
    md::Matrix result(a.shape(0),3);
    for(Eigen::Index i=0;i<result.rows();++i)for(int j=0;j<3;++j)result(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));return result;
}
std::vector<Complex> matrix(const py::array& a,Count spins,Count nk,Count n,bool spin) {
    if(!a.dtype().is(py::dtype::of<Complex>()) || a.ndim()!=(spin?4:3) || !contiguous(a) ||
       (spin && a.shape(0)!=py::ssize_t(spins)) || a.shape(spin)!=py::ssize_t(nk) ||
       a.shape(spin+1)!=py::ssize_t(n) || a.shape(spin+2)!=py::ssize_t(n))
        throw py::value_error("MANTA RI Fock requires complex128 H[k,AO,AO] and D[2,k,AO,AO]");
    std::vector<Complex> result(a.size());Count out=0;
    for(Count s=0;s<(spin?spins:1);++s)for(Count k=0;k<nk;++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)
        result[out++]=item<Complex>(a,(spin?s*a.strides(0):0)+k*a.strides(spin)+i*a.strides(spin+1)+j*a.strides(spin+2));
    return result;
}
py::array_t<Complex> owned(const std::vector<Complex>& values,std::vector<py::ssize_t> shape) {
    py::array_t<Complex> out(shape);if(!values.empty())std::memcpy(out.mutable_data(),values.data(),values.size()*sizeof(Complex));return out;
}
}
void bind_manta_periodic_ri_fock(py::module_& m) {
    auto p=py::class_<PeriodicRIFockPlan>(m,"PeriodicRIFockPlan");
#define P(name) p.def_readonly(#name,&PeriodicRIFockPlan::name)
    P(has_j);P(has_k);P(n_basis);P(n_auxiliary);P(momenta);P(channels);P(retained_source_bytes);P(driver_live_bytes);P(modeled_peak_bytes);
    P(metric_shell_calls);P(metric_primitive_products);P(build_shell_calls);P(build_primitive_products);
    P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(gamma);P(pair);
#undef P
    auto i=py::class_<PeriodicRIFockInfo>(m,"PeriodicRIFockInfo");
#define I(name) i.def_readonly(#name,&PeriodicRIFockInfo::name)
    I(accuracy);I(exchange_kernel);I(physical_omega);I(has_j);I(has_k);I(source_identity);I(frame_identity);I(weight_identity);I(metric_shell_calls);I(metric_primitive_products);
    I(reverse_metric_residual);I(reverse_projector_residual);I(reverse_inverse_residual);I(metrics);
#undef I
    auto r=py::class_<PeriodicRIFockReceipt,PeriodicFockReceipt>(m,"PeriodicRIFockReceipt");
#define R(name) r.def_readonly(#name,&PeriodicRIFockReceipt::name)
    R(reference_coupling_identity);R(coupling_error_bound);R(approximation_j_bound);R(approximation_energy_bound);R(coupling_multipole_partition_builds);R(coupling_multipole_partition_reuses);R(coupling_multipole_partition_bypasses);R(coupling_multipole_trials);R(coupling_multipole_images);R(coupling_multipole_tree_nodes);R(coupling_multipole_far_blocks);R(coupling_multipole_translations);R(coupling_multipole_moment_panels);R(coupling_multipole_envelope_evaluations);R(coupling_multipole_reused_moment_panels);R(coupling_multipole_reused_extents);R(coupling_multipole_global_pairs);R(coupling_multipole_global_pair_reuses);R(coupling_multipole_global_auxiliary);R(coupling_multipole_global_projections);R(coupling_multipole_global_auxiliary_projections);R(coupling_multipole_pair_groups);R(coupling_multipole_compressed_pairs);R(coupling_multipole_compressed_nodes);R(coupling_multipole_action_products);R(coupling_multipole_action_translations);
    R(exchange_kernel);R(physical_omega);R(has_j);R(has_k);R(frame_identity);R(weight_identity);R(shell_calls);R(primitive_products);R(coupling_passes);
#undef R
    py::class_<PeriodicRIFockResult>(m,"PeriodicRIFockResult")
        .def_readonly("receipt",&PeriodicRIFockResult::receipt).def_readonly("components",&PeriodicRIFockResult::components)
        .def_property_readonly("J",[](const PeriodicRIFockResult& r){return owned(r.J,{py::ssize_t(r.receipt.has_j?r.momenta:0),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});})
        .def_property_readonly("K",[](const PeriodicRIFockResult& r){return owned(r.K,{py::ssize_t(r.receipt.has_k?2:0),py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});})
        .def_property_readonly("values",[](const PeriodicRIFockResult& r){return owned(r.values,{2,py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});});
    m.def("plan_periodic_ri_fock",&plan_periodic_ri_fock,py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),
        py::arg("gamma_vectors"),py::arg("maximum_pair_vectors"),py::arg("momenta"),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("ri_options")=RIOptions{},py::arg("budget")=ResourceBudget{},
        py::arg("make_j")=true,py::arg("make_k")=true,py::arg("exchange_kernel")=PeriodicGDFKernelOptions{},py::arg("accuracy")=PeriodicRIAccuracyOptions{});
    py::class_<PeriodicRIFockBuilder>(m,"PeriodicRIFockBuilder")
        .def(py::init([](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,
            const py::array& k,const py::array& w,const py::array& g,const py::array& q,const py::array& boundaries,
            const PeriodicGDFShortRangeOptions& o,const RIOptions& ri,const ResourceBudget& budget,bool make_j,bool make_k,const PeriodicGDFKernelOptions& physical,const PeriodicRIAccuracyOptions& accuracy) {
            using namespace admission;const Count nk=vectors(k),ng=vectors(g),nq=vectors(q),pairs=square(nk);
            if(!w.dtype().is(py::dtype::of<double>()) || w.ndim()!=1 || w.shape(0)!=py::ssize_t(nk) || !contiguous(w))
                throw py::value_error("MANTA RI Fock weights require contiguous float64 (nk,)");
            if(!boundaries.dtype().is(py::dtype::of<std::int64_t>()) || boundaries.ndim()!=1 ||
               boundaries.shape(0)!=py::ssize_t(add(pairs,1)) || !contiguous(boundaries))
                throw py::value_error("MANTA RI Fock offsets require contiguous int64 (nk*nk+1,)");
            plan_periodic_ri_fock(orbital,auxiliary,system,ng,make_k && nk>1?1:0,nk,o,ri,budget,make_j,make_k,physical,accuracy);
            Count maximum=0,previous=0;
            if(item<std::int64_t>(boundaries,0)!=0)throw py::value_error("MANTA RI Fock offsets must start at zero");
            for(Count p=0;p<pairs;++p) {
                const auto next=item<std::int64_t>(boundaries,(p+1)*boundaries.strides(0));
                if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                    throw py::value_error("MANTA RI Fock offsets must be monotone and bounded");
                const Count length=next-previous;
                if(!make_k && length)throw py::value_error("MANTA RI Fock J-only requires an empty pair domain");
                if(make_k && ((p/nk==p%nk && length) || (p/nk!=p%nk && !length)))
                    throw py::value_error("MANTA RI Fock K requires empty diagonal and nonempty off-diagonal segments");
                maximum=std::max(maximum,length);previous=next;
            }
            if(previous!=nq)throw py::value_error("MANTA RI Fock offsets must end at the vector count");
            plan_periodic_ri_fock(orbital,auxiliary,system,ng,maximum,nk,o,ri,budget,make_j,make_k,physical,accuracy);
            auto momenta=copy_vectors(k),gamma=copy_vectors(g),pair=copy_vectors(q);
            std::vector<double> weights(nk);std::memcpy(weights.data(),w.data(),nk*sizeof(double));
            std::vector<Count> offsets(pairs+1);for(Count p=0;p<=pairs;++p)offsets[p]=item<std::int64_t>(boundaries,p*boundaries.strides(0));
            return PeriodicRIFockBuilder(orbital,auxiliary,system,momenta,weights,gamma,pair,offsets,o,ri,budget,make_j,make_k,physical,accuracy);
        }),py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("momenta").noconvert(),py::arg("weights").noconvert(),
            py::arg("gamma_vectors").noconvert(),py::arg("pair_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),
            py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("ri_options")=RIOptions{},py::arg("budget")=ResourceBudget{},
            py::arg("make_j")=true,py::arg("make_k")=true,py::arg("exchange_kernel")=PeriodicGDFKernelOptions{},py::arg("accuracy")=PeriodicRIAccuracyOptions{})
        .def_property_readonly("plan",&PeriodicRIFockBuilder::plan,py::return_value_policy::copy)
        .def_property_readonly("info",&PeriodicRIFockBuilder::info,py::return_value_policy::copy)
        .def("build",[](const PeriodicRIFockBuilder& b,const py::array& h,const py::array& d,double scale) {
            const auto& p=b.plan();auto hcore=matrix(h,1,p.momenta,p.n_basis,false),density=matrix(d,2,p.momenta,p.n_basis,true);
            py::gil_scoped_release release;return b.build(hcore,density,scale);
        },py::arg("hcore").noconvert(),py::arg("spin_densities").noconvert(),py::arg("exchange_scale")=1.);
}
