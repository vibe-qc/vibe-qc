#include "vibeqc/manta.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_md.hpp"
#include "vibeqc/manta_one_electron.hpp"
#include "vibeqc/manta_fitting.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/manta_fourier.hpp"
#include "vibeqc/manta_derivatives.hpp"
#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>

namespace py = pybind11;
using namespace vibeqc::manta;

namespace {
struct TileSession {
    std::shared_ptr<OrderedJKBuilder> builder;
    Workspace workspace;
    explicit TileSession(std::shared_ptr<OrderedJKBuilder> b)
        : builder(std::move(b)), workspace(builder->plan()) {}
};

py::dict evaluate(const OrderedJKBuilder& b,Workspace& workspace,const std::array<std::size_t,4>& indices) {
    const auto& shells=b.provider().basis().libint();
    std::vector<py::ssize_t> shape;
    for (auto i:indices) {
        if (i>=shells.size()) throw py::index_error("MANTA shell index");
        shape.push_back(static_cast<py::ssize_t>(shells[i].size()));
    }
    const auto tile=b.provider().evaluate(indices,workspace);
    py::array_t<double> values(shape);
    if (tile.values) std::copy_n(tile.values,values.size(),values.mutable_data());
    else std::fill_n(values.mutable_data(),values.size(),0.);
    py::dict result;
    result["values"]=std::move(values);
    result["shell_calls"]=1;
    result["null_tiles"]=tile.values?0:1;
    Count primitives=1;
    for (auto i:indices) primitives*=shells[i].alpha.size();
    result["primitive_quartets_requested"]=primitives;
    result["primitive_precision"]=0.;
    result["source_identity"]=b.provider().source_identity();
    result["provider"]=b.provider().name();
    const auto request=b.provider().kernel_request();
    result["kernel"]=request.kernel;
    result["omega"]=request.omega;
    return result;
}

struct FittingAxes {
    std::vector<py::object> owners;
    std::vector<const vibeqc::BasisSet*> values;
};
FittingAxes fitting_axes(const py::sequence& sequence,bool derivative=false) {
    const auto count=py::len(sequence);
    if(count<2 || count>(derivative?4:3))
        throw py::value_error(derivative?"MANTA derivatives require two, three or four basis axes":"MANTA fitting tiles require two or three basis axes");
    FittingAxes axes;
    axes.owners.reserve(count); axes.values.reserve(count);
    for(py::size_t i=0;i<count;++i) {
        axes.owners.push_back(sequence[i]);
        axes.values.push_back(&axes.owners.back().cast<const vibeqc::BasisSet&>());
    }
    return axes;
}

md::Matrix fourier_vectors(const vibeqc::BasisSet& bra,const vibeqc::BasisSet* ket,
                           const py::array& array,const ResourceBudget& budget) {
    if(!array.dtype().is(py::dtype::of<double>()) || array.ndim()!=2 || array.shape(1)!=3)
        throw py::value_error("MANTA Fourier vectors must be a float64 ndarray of shape (n,3)");
    if(!(array.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA Fourier vectors must be C- or F-contiguous");
    plan_fourier(bra,ket,array.shape(0),budget);
    md::Matrix vectors(array.shape(0),3);
    if(array.size()) {
        const auto* data=static_cast<const char*>(array.data());
        if(array.flags() & py::array::c_style)
            std::memcpy(vectors.data(),data,array.size()*sizeof(double));
        else for(Eigen::Index i=0;i<vectors.rows();++i) for(int j=0;j<3;++j)
            std::memcpy(&vectors(i,j),data+(j*vectors.rows()+i)*sizeof(double),sizeof(double));
    }
    return vectors;
}

Eigen::MatrixXd density(const OrderedJKBuilder& builder, const py::array& array) {
    const auto n = static_cast<py::ssize_t>(builder.plan().n_basis);
    // No forcecast: refuse logical/dtype/layout errors before copying payload.
    if (!array.dtype().is(py::dtype::of<double>()) || array.ndim()!=2 || array.shape(0)!=n || array.shape(1)!=n)
        throw py::value_error("MANTA density must be a square float64 ndarray matching the basis");
    if (!(array.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA density must be C- or F-contiguous");
    Eigen::MatrixXd d(n,n);
    const auto* ptr = static_cast<const double*>(array.data());
    if (array.flags() & py::array::f_style)
        d = Eigen::Map<const Eigen::MatrixXd>(ptr,n,n);
    else
        d = Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(ptr,n,n);
    return d;
}
} // namespace

void bind_manta_ri(py::module_&);
void bind_manta_molecular_source(py::module_&);
void bind_manta_bloch_fourier(py::module_&);
void bind_manta_bloch_one_electron(py::module_&);
void bind_manta_periodic_hartree(py::module_&);
void bind_manta_periodic_fitting(py::module_&);
void bind_manta_periodic_ri(py::module_&);
void bind_manta_periodic_ri_fitting_bounds(py::module_&);
void bind_manta_periodic_ri_fock(py::module_&);
void bind_manta_periodic_nuclear(py::module_&);
void bind_manta_periodic_exchange(py::module_&);
void bind_manta_periodic_semilocal_xc(py::module_&);
void bind_manta_periodic_xc_refinement(py::module_&);
void bind_manta_periodic_local_exchange(py::module_&);
void bind_manta_periodic_fock(py::module_&);
void bind_manta_periodic_rhf(py::module_&);
void bind_manta_periodic_ks(py::module_&);
void bind_manta_periodic_uhf(py::module_&);
void bind_manta_periodic_ri_uhf(py::module_&);
void bind_manta_periodic_ri_rhf(py::module_&);
void bind_manta_ri_gradient(py::module_&);
void bind_manta_gradient(py::module_&);
void bind_manta_scf(py::module_&);
void bind_manta_extents(py::module_&);
void bind_manta_multipole(py::module_&);
void bind_manta_lattice_tail(py::module_&);
void bind_manta_lattice_errors(py::module_&);
void bind_manta_periodic_pairs(py::module_&);
void bind_manta_bubbles(py::module_&);
void bind_manta_periodic_bubbles(py::module_&);
void bind_manta_periodic_ewald(py::module_&);
void bind_manta_periodic_bupo(py::module_&);
void bind_manta_periodic_electrostatic(py::module_&);
void bind_manta_periodic_bupo_hf(py::module_&);
void bind_manta_periodic_one_electron_bounds(py::module_&);
void bind_manta_periodic_exchange_bounds(py::module_&);
void bind_manta_periodic_support_selection(py::module_&);
void bind_manta_moments(py::module_&);
void bind_manta_bupo_errors(py::module_&);
void bind_manta_tiles(py::module_&);
void bind_manta_bupo(py::module_&);
void bind_manta_bupo_fitting(py::module_&);
void bind_manta_bupo_ri(py::module_&);
void bind_manta(py::module_& parent) {
    auto m = parent.def_submodule("_manta", "Experimental integral provider reference boundary");
    py::class_<KernelRequest>(m,"KernelRequest")
        .def(py::init<>())
        .def_readwrite("kernel",&KernelRequest::kernel)
        .def_readwrite("omega",&KernelRequest::omega)
        .def_readwrite("scalar",&KernelRequest::scalar)
        .def_readwrite("centers",&KernelRequest::centers)
        .def_readwrite("derivative",&KernelRequest::derivative)
        .def_readwrite("images",&KernelRequest::images);
    py::class_<ProducerRequest>(m,"ProducerRequest")
        .def(py::init<>())
        .def_readwrite("producer",&ProducerRequest::producer)
        .def_readwrite("order",&ProducerRequest::order);
    py::class_<Capability>(m,"Capability")
        .def_readonly("supported",&Capability::supported)
        .def_readonly("maximum_l",&Capability::maximum_l)
        .def_readonly("maximum_auxiliary_l",&Capability::maximum_auxiliary_l)
        .def_readonly("reason",&Capability::reason);
    py::class_<ResourceBudget>(m,"ResourceBudget")
        .def(py::init<>())
        .def_readwrite("memory_bytes",&ResourceBudget::memory_bytes)
        .def_readwrite("shell_calls",&ResourceBudget::shell_calls)
        .def_readwrite("primitive_quartets",&ResourceBudget::primitive_quartets)
        .def_readwrite("scalar_terms",&ResourceBudget::scalar_terms)
        .def_readwrite("kernel_terms",&ResourceBudget::kernel_terms)
        .def_readwrite("other_live_bytes",&ResourceBudget::other_live_bytes)
        .def_readwrite("backend_margin_bytes",&ResourceBudget::backend_margin_bytes);
    bind_manta_ri(m);
    bind_manta_molecular_source(m);
    bind_manta_bloch_fourier(m);
    bind_manta_bloch_one_electron(m);
    bind_manta_periodic_hartree(m);
    bind_manta_periodic_fitting(m);
    bind_manta_periodic_ri(m);
    bind_manta_periodic_nuclear(m);
    bind_manta_periodic_exchange(m);
    bind_manta_periodic_local_exchange(m);
    bind_manta_periodic_semilocal_xc(m);
    bind_manta_periodic_fock(m);
    bind_manta_periodic_ri_fock(m);
    bind_manta_periodic_rhf(m);
    bind_manta_periodic_uhf(m);
    bind_manta_periodic_ri_uhf(m);
    bind_manta_periodic_ri_rhf(m);
    bind_manta_periodic_ks(m);
    bind_manta_periodic_xc_refinement(m);
    bind_manta_ri_gradient(m);
    bind_manta_gradient(m);
    bind_manta_extents(m);
    bind_manta_multipole(m);
    bind_manta_lattice_tail(m);
    bind_manta_lattice_errors(m);
    bind_manta_periodic_pairs(m);
    bind_manta_bubbles(m);
    bind_manta_periodic_bubbles(m);
    bind_manta_periodic_ewald(m);
    bind_manta_periodic_bupo(m);
    bind_manta_periodic_ri_fitting_bounds(m);
    bind_manta_periodic_electrostatic(m);
    bind_manta_periodic_one_electron_bounds(m);
    bind_manta_periodic_exchange_bounds(m);
    bind_manta_periodic_support_selection(m);
    bind_manta_periodic_bupo_hf(m);
    bind_manta_moments(m);
    bind_manta_bupo_errors(m);
    bind_manta_tiles(m);
    bind_manta_bupo(m);
    bind_manta_bupo_fitting(m);
    bind_manta_bupo_ri(m);
    bind_manta_scf(m);
    py::class_<BuildPlan>(m,"BuildPlan")
        .def_readonly("schedule",&BuildPlan::schedule)
        .def_readonly("channel_batch",&BuildPlan::channel_batch)
        .def_readonly("provider",&BuildPlan::provider)
        .def_readonly("kernel",&BuildPlan::kernel)
        .def_readonly("omega",&BuildPlan::omega)
        .def_readonly("kernel_terms",&BuildPlan::kernel_terms)
        .def_readonly("n_basis",&BuildPlan::n_basis)
        .def_readonly("n_shells",&BuildPlan::n_shells)
        .def_readonly("maximum_primitives",&BuildPlan::maximum_primitives)
        .def_readonly("maximum_l",&BuildPlan::maximum_l)
        .def_readonly("basis_payload_bytes",&BuildPlan::basis_payload_bytes)
        .def_readonly("engine_bytes",&BuildPlan::engine_bytes)
        .def_readonly("target_bytes",&BuildPlan::target_bytes)
        .def_readonly("tile_elements",&BuildPlan::tile_elements)
        .def_readonly("modeled_peak_bytes",&BuildPlan::modeled_peak_bytes)
        .def_readonly("shell_calls",&BuildPlan::shell_calls)
        .def_readonly("primitive_quartets_upper_bound",&BuildPlan::primitive_quartets_upper_bound)
        .def_readonly("scalar_terms",&BuildPlan::scalar_terms);
    py::class_<BuildReceipt>(m,"BuildReceipt")
        .def_readonly("primitive_quartets_evaluated",&BuildReceipt::primitive_quartets_evaluated)
        .def_readonly("primitive_evaluations_measured",&BuildReceipt::primitive_evaluations_measured)
        .def_readonly("contraction_strategy",&BuildReceipt::contraction_strategy)
        .def_readonly("kernel",&BuildReceipt::kernel)
        .def_readonly("omega",&BuildReceipt::omega)
        .def_readonly("provider",&BuildReceipt::provider)
        .def_readonly("provider_version",&BuildReceipt::provider_version)
        .def_readonly("source_identity",&BuildReceipt::source_identity)
        .def_readonly("density_identity",&BuildReceipt::density_identity)
        .def_readonly("normalization",&BuildReceipt::normalization)
        .def_readonly("algorithm",&BuildReceipt::algorithm)
        .def_readonly("shell_calls",&BuildReceipt::shell_calls)
        .def_readonly("null_tiles",&BuildReceipt::null_tiles)
        .def_readonly("integral_elements_consumed",&BuildReceipt::integral_elements_consumed)
        .def_readonly("primitive_quartets_requested",&BuildReceipt::primitive_quartets_requested)
        .def_readonly("density_terms_accumulated",&BuildReceipt::density_terms_accumulated)
        .def_readonly("modeled_peak_bytes",&BuildReceipt::modeled_peak_bytes)
        .def_readonly("primitive_precision",&BuildReceipt::primitive_precision);
    py::class_<JKResult>(m,"JKResult")
        .def_readonly("J",&JKResult::J)
        .def_readonly("K",&JKResult::K)
        .def_readonly("receipt",&JKResult::receipt);
    py::class_<BatchPlan>(m,"BatchPlan")
        .def_readonly("n_densities",&BatchPlan::n_densities)
        .def_readonly("shell_calls",&BatchPlan::shell_calls)
        .def_readonly("primitive_quartets_upper_bound",&BatchPlan::primitive_quartets_upper_bound)
        .def_readonly("scalar_terms",&BatchPlan::scalar_terms)
        .def_readonly("kernel_terms",&BatchPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&BatchPlan::modeled_peak_bytes);
    py::class_<BatchResult>(m,"BatchResult")
        .def_readonly("J",&BatchResult::J)
        .def_readonly("K",&BatchResult::K)
        .def_readonly("density_identities",&BatchResult::density_identities)
        .def_readonly("receipt",&BatchResult::receipt);
    py::class_<TileSession,std::shared_ptr<TileSession>>(m,"TileSession")
        .def("evaluate",[](TileSession& s,const std::array<std::size_t,4>& indices){
            return evaluate(*s.builder,s.workspace,indices);
        },py::arg("ordered_quartet"));
    py::class_<OrderedJKBuilder,vibeqc::JKBuilder,std::shared_ptr<OrderedJKBuilder>>(m,"OrderedJKBuilder")
        .def_property_readonly("plan",&OrderedJKBuilder::plan,py::return_value_policy::copy)
        .def_property_readonly("source_identity",[](const OrderedJKBuilder& b){return b.provider().source_identity();})
        .def_property_readonly("shells",[](const OrderedJKBuilder& b){return b.provider().basis().shells();})
        .def("session",[](std::shared_ptr<OrderedJKBuilder> b){return std::make_shared<TileSession>(std::move(b));})
        .def("build",[](const OrderedJKBuilder& b,const py::array& d){return b.build(density(b,d));},py::arg("density").noconvert())
        .def("plan_batch",&OrderedJKBuilder::plan_batch,py::arg("n_densities"),py::arg("make_j")=true,py::arg("make_k")=true)
        .def("build_batch",[](const OrderedJKBuilder& b,const py::sequence& values,bool make_j,bool make_k) {
            const auto count=py::len(values);
            b.plan_batch(count,make_j,make_k);
            std::vector<Eigen::MatrixXd> densities; densities.reserve(count);
            for(py::size_t i=0;i<count;++i) {
                auto item=values[i];
                if(!py::isinstance<py::array>(item)) throw py::value_error("MANTA batch density must be an ndarray");
                densities.push_back(density(b,py::reinterpret_borrow<py::array>(item)));
            }
            return b.build_batch(densities,make_j,make_k);
        },py::arg("densities"),py::arg("make_j")=true,py::arg("make_k")=true)
        .def("build_J",[](const OrderedJKBuilder& b,const py::array& d){return b.build_J(density(b,d));},py::arg("density").noconvert())
        .def("build_K",[](const OrderedJKBuilder& b,const py::array& d){return b.build_K(density(b,d));},py::arg("density").noconvert())
        .def("build_J_slot",[](const OrderedJKBuilder& b,const py::array& d,int slot){return b.build_J_slot(density(b,d),slot);},
             py::arg("density").noconvert(),py::arg("slot"))
        .def("build_K_slot",[](const OrderedJKBuilder& b,const py::array& d,int slot){return b.build_K_slot(density(b,d),slot);},
             py::arg("density").noconvert(),py::arg("slot"))
        .def("build_K_pair",[](const OrderedJKBuilder& b,const py::array& a,const py::array& beta){
            b.plan_batch(2,false,true);
            const auto da=density(b,a),db=density(b,beta);
            return b.build_K_pair(da,db);
        },py::arg("density_alpha").noconvert(),py::arg("density_beta").noconvert())
        .def("build_g_rhf",[](const OrderedJKBuilder& b,const py::array& d,double alpha){return b.build_g_rhf(density(b,d),alpha);},
             py::arg("density").noconvert(),py::arg("alpha_hf")=1.)
        .def("evaluate",[](const OrderedJKBuilder& b,const std::array<std::size_t,4>& indices){
            for(auto i:indices) if(i>=b.plan().n_shells) throw py::index_error("MANTA shell index");
            Workspace workspace(b.plan());
            return evaluate(b,workspace,indices);
        },py::arg("ordered_quartet"));
    py::class_<ReferenceJKBuilder,OrderedJKBuilder,std::shared_ptr<ReferenceJKBuilder>>(m,"ReferenceJKBuilder")
        .def(py::init<const vibeqc::BasisSet&,const KernelRequest&,const ResourceBudget&>(),
             py::arg("basis"),py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{});
    py::class_<DirectJKBuilder,OrderedJKBuilder,std::shared_ptr<DirectJKBuilder>>(m,"DirectJKBuilder")
        .def(py::init<const vibeqc::BasisSet&,const KernelRequest&,const ResourceBudget&,Count>(),
             py::arg("basis"),py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{},py::arg("channel_batch")=1)
        .def("build_K_erf",[](const DirectJKBuilder& builder,const py::array& d,double omega) {
            return builder.build_K_erf(density(builder,d),omega);
        },py::arg("density").noconvert(),py::arg("omega"));
    py::class_<OneElectronRequest>(m,"OneElectronRequest")
        .def(py::init<>())
        .def_readwrite("operation",&OneElectronRequest::operation)
        .def_readwrite("charges",&OneElectronRequest::charges)
        .def_readwrite("origin",&OneElectronRequest::origin)
        .def_readwrite("powers",&OneElectronRequest::powers)
        .def_readwrite("kernel",&OneElectronRequest::kernel)
        .def_readwrite("omega",&OneElectronRequest::omega);
    py::class_<OneElectronDerivative>(m,"OneElectronDerivative")
        .def(py::init<>())
        .def_readwrite("target",&OneElectronDerivative::target)
        .def_readwrite("axis",&OneElectronDerivative::axis)
        .def_readwrite("charge_index",&OneElectronDerivative::charge_index);
    py::class_<OneElectronPlan>(m,"OneElectronPlan")
        .def_readonly("shape",&OneElectronPlan::shape)
        .def_readonly("shell_calls",&OneElectronPlan::shell_calls)
        .def_readonly("primitive_pairs",&OneElectronPlan::primitive_pairs)
        .def_readonly("primitive_evaluations_upper_bound",&OneElectronPlan::primitive_evaluations_upper_bound)
        .def_readonly("kernel_terms",&OneElectronPlan::kernel_terms)
        .def_readonly("tile_elements",&OneElectronPlan::tile_elements)
        .def_readonly("output_bytes",&OneElectronPlan::output_bytes)
        .def_readonly("modeled_peak_bytes",&OneElectronPlan::modeled_peak_bytes);
    py::class_<OneElectronReceipt>(m,"OneElectronReceipt")
        .def_readonly("provider",&OneElectronReceipt::provider)
        .def_readonly("provider_version",&OneElectronReceipt::provider_version)
        .def_readonly("normalization",&OneElectronReceipt::normalization)
        .def_readonly("operation",&OneElectronReceipt::operation)
        .def_readonly("kernel",&OneElectronReceipt::kernel)
        .def_readonly("omega",&OneElectronReceipt::omega)
        .def_readonly("derivative_target",&OneElectronReceipt::derivative_target)
        .def_readonly("derivative_axis",&OneElectronReceipt::derivative_axis)
        .def_readonly("derivative_charge_index",&OneElectronReceipt::derivative_charge_index)
        .def_readonly("source_identity",&OneElectronReceipt::source_identity)
        .def_readonly("shell_calls",&OneElectronReceipt::shell_calls)
        .def_readonly("primitive_pairs",&OneElectronReceipt::primitive_pairs)
        .def_readonly("primitive_integrals_evaluated",&OneElectronReceipt::primitive_integrals_evaluated)
        .def_readonly("integral_elements",&OneElectronReceipt::integral_elements)
        .def_readonly("modeled_peak_bytes",&OneElectronReceipt::modeled_peak_bytes);
    py::class_<OneElectronResult>(m,"OneElectronResult")
        .def_readonly("values",&OneElectronResult::values)
        .def_readonly("receipt",&OneElectronResult::receipt);
    m.def("one_electron_capability",&one_electron_capability,py::arg("request")=OneElectronRequest{});
    m.def("plan_one_electron",&plan_one_electron,py::arg("bra"),py::arg("ket"),
          py::arg("request")=OneElectronRequest{},py::arg("budget")=ResourceBudget{});
    m.def("compute_one_electron",&compute_one_electron,py::arg("bra"),py::arg("ket"),
          py::arg("request")=OneElectronRequest{},py::arg("budget")=ResourceBudget{});
    m.def("plan_one_electron_derivative",&plan_one_electron_derivative,py::arg("bra"),py::arg("ket"),
          py::arg("request")=OneElectronRequest{},py::arg("derivative")=OneElectronDerivative{},py::arg("budget")=ResourceBudget{});
    m.def("compute_one_electron_derivative",&compute_one_electron_derivative,py::arg("bra"),py::arg("ket"),
          py::arg("request")=OneElectronRequest{},py::arg("derivative")=OneElectronDerivative{},py::arg("budget")=ResourceBudget{});
    py::class_<MetricConstructionPlan>(m,"MetricConstructionPlan")
        .def_readonly("modeled_peak_bytes",&MetricConstructionPlan::modeled_peak_bytes)
        .def_readonly("scalar_terms",&MetricConstructionPlan::scalar_terms)
        .def_readonly("kernel_terms",&MetricConstructionPlan::kernel_terms)
        .def_readonly("shell_calls",&MetricConstructionPlan::shell_calls)
        .def_readonly("primitive_products",&MetricConstructionPlan::primitive_products);
    py::class_<MetricConstructionCertificate>(m,"MetricConstructionCertificate")
        .def_readonly("evaluated",&MetricConstructionCertificate::evaluated)
        .def_readonly("lower",&MetricConstructionCertificate::lower)
        .def_readonly("upper",&MetricConstructionCertificate::upper)
        .def_readonly("error_bounds",&MetricConstructionCertificate::error_bounds);
    m.def("plan_metric_construction",&plan_metric_construction,py::arg("bra"),py::arg("ket"),
        py::arg("request")=fitting_request(2),py::arg("budget")=ResourceBudget{});
    m.def("certify_metric_construction",&certify_metric_construction,py::arg("bra"),py::arg("ket"),
        py::arg("request")=fitting_request(2),py::arg("budget")=ResourceBudget{});
    py::class_<FittingPlan>(m,"FittingPlan")
        .def_readonly("shape",&FittingPlan::shape)
        .def_readonly("shell_extents",&FittingPlan::shell_extents)
        .def_readonly("shell_calls",&FittingPlan::shell_calls)
        .def_readonly("primitive_products",&FittingPlan::primitive_products)
        .def_readonly("kernel_terms",&FittingPlan::kernel_terms)
        .def_readonly("integral_elements",&FittingPlan::integral_elements)
        .def_readonly("tile_elements",&FittingPlan::tile_elements)
        .def_readonly("modeled_peak_bytes",&FittingPlan::modeled_peak_bytes);
    py::class_<FittingTiles>(m,"FittingTiles")
        .def(py::init([](const py::sequence& sequence,const ResourceBudget& budget,const py::object& requested) {
            const auto axes=fitting_axes(sequence);
            const auto default_request=fitting_request(axes.values.size());
            const auto& request=requested.is_none()?default_request:requested.cast<const KernelRequest&>();
            return std::make_unique<FittingTiles>(axes.values,request,budget);
        }),py::arg("basis_axes"),py::arg("budget")=ResourceBudget{},py::arg("request")=py::none())
        .def_property_readonly("plan",&FittingTiles::plan,py::return_value_policy::copy)
        .def_property_readonly("source_identity",&FittingTiles::source_identity)
        .def("evaluate",[](const FittingTiles& provider,const py::sequence& indices) {
            if(py::len(indices)!=provider.plan().shape.size()) throw py::value_error("MANTA fitting shell-index arity mismatch");
            std::vector<std::size_t> native_indices;
            for(std::size_t i=0;i<provider.plan().shape.size();++i) native_indices.push_back(indices[i].cast<std::size_t>());
            const auto tile=provider.evaluate(native_indices);
            std::vector<py::ssize_t> shape(tile.shape.begin(),tile.shape.end());
            py::array_t<double> values(shape);
            std::copy_n(tile.values.data(),values.size(),values.mutable_data());
            py::dict result;
            result["values"]=std::move(values);
            result["provider"]="manta";
            result["provider_version"]="md-fp64-v1";
            result["kernel"]=provider.kernel_request().kernel;
            result["omega"]=provider.kernel_request().omega;
            result["source_identity"]=provider.source_identity();
            result["normalization"]="libint-shell-axial-v1";
            result["shell_calls"]=1;
            result["null_tiles"]=0;
            result["primitive_products_requested"]=tile.primitive_products;
            result["modeled_peak_bytes"]=provider.plan().modeled_peak_bytes;
            return result;
        },py::arg("ordered_shell_indices"));
    m.def("plan_fitting",[](const py::sequence& sequence,const ResourceBudget& budget,const py::object& requested) {
        const auto axes=fitting_axes(sequence);
        const auto default_request=fitting_request(axes.values.size());
        const auto& request=requested.is_none()?default_request:requested.cast<const KernelRequest&>();
        return plan_fitting(axes.values,request,budget);
    },py::arg("basis_axes"),py::arg("budget")=ResourceBudget{},py::arg("request")=py::none());
    py::class_<DerivativePlan>(m,"DerivativePlan")
        .def_readonly("shape",&DerivativePlan::shape)
        .def_readonly("shell_extents",&DerivativePlan::shell_extents)
        .def_readonly("components",&DerivativePlan::components)
        .def_readonly("shell_calls",&DerivativePlan::shell_calls)
        .def_readonly("primitive_products",&DerivativePlan::primitive_products)
        .def_readonly("primitive_evaluations_upper_bound",&DerivativePlan::primitive_evaluations_upper_bound)
        .def_readonly("integral_elements",&DerivativePlan::integral_elements)
        .def_readonly("tile_elements",&DerivativePlan::tile_elements)
        .def_readonly("kernel_terms",&DerivativePlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&DerivativePlan::modeled_peak_bytes);
    py::class_<DerivativeTiles>(m,"DerivativeTiles")
        .def(py::init([](const py::sequence& sequence,const ResourceBudget& budget,const py::object& requested) {
            const auto axes=fitting_axes(sequence,true);
            const auto default_request=derivative_request(axes.values.size());
            const auto& request=requested.is_none()?default_request:requested.cast<const KernelRequest&>();
            return std::make_unique<DerivativeTiles>(axes.values,request,budget);
        }),py::arg("basis_axes"),py::arg("budget")=ResourceBudget{},py::arg("request")=py::none())
        .def_property_readonly("plan",&DerivativeTiles::plan,py::return_value_policy::copy)
        .def_property_readonly("source_identity",&DerivativeTiles::source_identity)
        .def("evaluate",[](const DerivativeTiles& provider,const py::sequence& indices,int center,int axis) {
            if(py::len(indices)!=provider.plan().shape.size()) throw py::value_error("MANTA derivative shell-index arity mismatch");
            std::vector<std::size_t> native_indices;
            for(std::size_t i=0;i<provider.plan().shape.size();++i) native_indices.push_back(indices[i].cast<std::size_t>());
            const auto tile=provider.evaluate(native_indices,center,axis);
            std::vector<py::ssize_t> shape(tile.shape.begin(),tile.shape.end());
            py::array_t<double> values(shape);
            std::copy_n(tile.values.data(),values.size(),values.mutable_data());
            py::dict result;
            result["values"]=std::move(values);
            result["provider"]="manta"; result["provider_version"]="md-fp64-v1";
            result["kernel"]=provider.kernel_request().kernel; result["omega"]=provider.kernel_request().omega;
            result["source_identity"]=provider.source_identity(); result["normalization"]="libint-shell-axial-v1";
            result["derivative_order"]=1; result["derivative_center"]=center; result["derivative_axis"]=axis;
            result["shell_calls"]=1; result["primitive_products_requested"]=tile.primitive_products;
            result["primitive_integrals_evaluated"]=tile.primitive_evaluations;
            result["modeled_peak_bytes"]=provider.plan().modeled_peak_bytes;
            return result;
        },py::arg("ordered_shell_indices"),py::arg("center"),py::arg("axis"));
    m.def("plan_derivatives",[](const py::sequence& sequence,const ResourceBudget& budget,const py::object& requested) {
        const auto axes=fitting_axes(sequence,true);
        const auto default_request=derivative_request(axes.values.size());
        const auto& request=requested.is_none()?default_request:requested.cast<const KernelRequest&>();
        return plan_derivatives(axes.values,request,budget);
    },py::arg("basis_axes"),py::arg("budget")=ResourceBudget{},py::arg("request")=py::none());
    py::class_<FourierPlan>(m,"FourierPlan")
        .def_readonly("shape",&FourierPlan::shape)
        .def_readonly("shell_calls",&FourierPlan::shell_calls)
        .def_readonly("primitive_pairs",&FourierPlan::primitive_pairs)
        .def_readonly("vector_primitive_products",&FourierPlan::vector_primitive_products)
        .def_readonly("kernel_terms",&FourierPlan::kernel_terms)
        .def_readonly("tile_elements",&FourierPlan::tile_elements)
        .def_readonly("output_bytes",&FourierPlan::output_bytes)
        .def_readonly("modeled_peak_bytes",&FourierPlan::modeled_peak_bytes);
    py::class_<FourierReceipt>(m,"FourierReceipt")
        .def_readonly("provider",&FourierReceipt::provider)
        .def_readonly("provider_version",&FourierReceipt::provider_version)
        .def_readonly("normalization",&FourierReceipt::normalization)
        .def_readonly("convention",&FourierReceipt::convention)
        .def_readonly("source_identity",&FourierReceipt::source_identity)
        .def_readonly("shell_calls",&FourierReceipt::shell_calls)
        .def_readonly("primitive_pairs",&FourierReceipt::primitive_pairs)
        .def_readonly("vector_primitive_products",&FourierReceipt::vector_primitive_products)
        .def_readonly("integral_elements",&FourierReceipt::integral_elements)
        .def_readonly("modeled_peak_bytes",&FourierReceipt::modeled_peak_bytes);
    py::class_<FourierResult>(m,"FourierResult")
        .def_readonly("shape",&FourierResult::shape)
        .def_readonly("receipt",&FourierResult::receipt)
        .def_property_readonly("values",[](const FourierResult& result) {
            std::vector<py::ssize_t> shape(result.shape.begin(),result.shape.end());
            py::array_t<std::complex<double>> values(shape);
            if(!result.values.empty()) std::copy(result.values.begin(),result.values.end(),values.mutable_data());
            return values;
        });
    py::class_<BlochPhasePlan>(m,"BlochPhasePlan")
        .def_readonly("modeled_peak_bytes",&BlochPhasePlan::modeled_peak_bytes)
        .def_readonly("scalar_terms",&BlochPhasePlan::scalar_terms)
        .def_readonly("kernel_terms",&BlochPhasePlan::kernel_terms);
    py::class_<BlochPhaseCertificate>(m,"BlochPhaseCertificate")
        .def_readonly("real_lower",&BlochPhaseCertificate::real_lower)
        .def_readonly("real_upper",&BlochPhaseCertificate::real_upper)
        .def_readonly("imag_lower",&BlochPhaseCertificate::imag_lower)
        .def_readonly("imag_upper",&BlochPhaseCertificate::imag_upper)
        .def_readonly("error_bound",&BlochPhaseCertificate::error_bound)
        .def_readonly("squarings",&BlochPhaseCertificate::squarings);
    m.def("plan_bloch_phase",&plan_bloch_phase,py::arg("budget")=ResourceBudget{});
    m.def("certify_bloch_phase",&certify_bloch_phase,py::arg("momentum"),py::arg("translation"),
        py::arg("evaluated_phase"),py::arg("budget")=ResourceBudget{});
    py::class_<FourierConstructionPlan>(m,"FourierConstructionPlan")
        .def_readonly("modeled_peak_bytes",&FourierConstructionPlan::modeled_peak_bytes)
        .def_readonly("scalar_terms",&FourierConstructionPlan::scalar_terms)
        .def_readonly("kernel_terms",&FourierConstructionPlan::kernel_terms)
        .def_readonly("shell_calls",&FourierConstructionPlan::shell_calls)
        .def_readonly("primitive_products",&FourierConstructionPlan::primitive_products);
    py::class_<FourierConstructionCertificate>(m,"FourierConstructionCertificate")
        .def_readonly("evaluated",&FourierConstructionCertificate::evaluated)
        .def_readonly("real_lower",&FourierConstructionCertificate::real_lower)
        .def_readonly("real_upper",&FourierConstructionCertificate::real_upper)
        .def_readonly("imag_lower",&FourierConstructionCertificate::imag_lower)
        .def_readonly("imag_upper",&FourierConstructionCertificate::imag_upper)
        .def_readonly("error_bounds",&FourierConstructionCertificate::error_bounds);
    m.def("plan_auxiliary_fourier_construction",&plan_auxiliary_fourier_construction,
        py::arg("auxiliary"),py::arg("n_vectors"),py::arg("budget")=ResourceBudget{});
    m.def("certify_auxiliary_fourier_panel",[](const vibeqc::BasisSet& auxiliary,const py::array& vectors,const ResourceBudget& budget) {
        if(vectors.ndim()!=2)throw py::value_error("MANTA Fourier vectors must have shape (n,3)");
        plan_auxiliary_fourier_construction(auxiliary,vectors.shape(0),budget);
        const auto q=fourier_vectors(auxiliary,nullptr,vectors,budget);
        return certify_auxiliary_fourier_panel(auxiliary,q,budget);
    },py::arg("auxiliary"),py::arg("wave_vectors").noconvert(),py::arg("budget")=ResourceBudget{});
    m.def("plan_fourier",[](const vibeqc::BasisSet& bra,const vibeqc::BasisSet& ket,Count count,const ResourceBudget& budget) {
        return plan_fourier(bra,&ket,count,budget);
    },py::arg("bra"),py::arg("ket"),py::arg("n_vectors"),py::arg("budget")=ResourceBudget{});
    m.def("plan_auxiliary_fourier",[](const vibeqc::BasisSet& auxiliary,Count count,const ResourceBudget& budget) {
        return plan_fourier(auxiliary,nullptr,count,budget);
    },py::arg("auxiliary"),py::arg("n_vectors"),py::arg("budget")=ResourceBudget{});
    m.def("compute_fourier_panel",[](const vibeqc::BasisSet& bra,const vibeqc::BasisSet& ket,const py::array& vectors,const ResourceBudget& budget) {
        const auto q=fourier_vectors(bra,&ket,vectors,budget);
        return compute_fourier_panel(bra,&ket,q,budget);
    },py::arg("bra"),py::arg("ket"),py::arg("wave_vectors").noconvert(),py::arg("budget")=ResourceBudget{});
    m.def("compute_auxiliary_fourier_panel",[](const vibeqc::BasisSet& auxiliary,const py::array& vectors,const ResourceBudget& budget) {
        const auto q=fourier_vectors(auxiliary,nullptr,vectors,budget);
        return compute_fourier_panel(auxiliary,nullptr,q,budget);
    },py::arg("auxiliary"),py::arg("wave_vectors").noconvert(),py::arg("budget")=ResourceBudget{});
    m.def("capability",&libint_capability,py::arg("request")=KernelRequest{});
    m.def("plan",&plan_libint,py::arg("basis"),py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{});
    m.def("md_capability",&md_capability,py::arg("request")=KernelRequest{});
    m.def("md_producer_capability",&md_producer_capability,py::arg("request")=ProducerRequest{});
    m.def("md_plan",&plan_md,py::arg("basis"),py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{},py::arg("channel_batch")=1);
    m.def("boys",&md::boys,py::arg("maximum_n"),py::arg("argument"));
    m.def("range_boys",[](int n,double t,double rho,const std::string& kernel,double omega) {
        KernelRequest request; request.kernel=kernel; request.omega=omega;
        const auto error=md_request_error(request,4);
        if(!error.empty()) throw py::value_error(error);
        if(!std::isfinite(rho) || rho<=0) throw py::value_error("MANTA Boys rho must be finite and positive");
        return md::radial_boys(n,t,rho,radial_kernel(request),omega);
    },py::arg("maximum_n"),py::arg("argument"),py::arg("rho"),py::arg("kernel"),py::arg("omega"));
    py::class_<RepresentationCertificate>(m,"RepresentationCertificate")
        .def_readonly("evaluated",&RepresentationCertificate::evaluated)
        .def_readonly("lower",&RepresentationCertificate::lower)
        .def_readonly("upper",&RepresentationCertificate::upper)
        .def_readonly("error_bounds",&RepresentationCertificate::error_bounds);
    m.def("plan_representation",&plan_representation,py::arg("angular_momentum"),py::arg("pure")=true,py::arg("budget")=ResourceBudget{});
    m.def("certify_representation",&certify_representation,py::arg("angular_momentum"),py::arg("pure")=true,py::arg("budget")=ResourceBudget{});
    m.def("cartesian_to_spherical",[](int l){return md::representation(l,true);},py::arg("angular_momentum"));
}
