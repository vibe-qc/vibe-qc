#include "vibeqc/manta.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <libint2/engine.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vibeqc::manta {
namespace {
Count add(Count a, Count b) {
    if (b > std::numeric_limits<Count>::max() - a)
        throw std::overflow_error("MANTA resource sum overflow");
    return a + b;
}
Count mul(Count a, Count b) {
    if (a && b > std::numeric_limits<Count>::max() / a)
        throw std::overflow_error("MANTA resource product overflow");
    return a * b;
}
Count fourth(Count n) { const auto n2 = mul(n, n); return mul(n2, n2); }
void cap(Count n, Count limit, const char* what) {
    if (n > limit) throw std::length_error(what);
}
void finite(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("MANTA requires finite basis/density data");
}
std::string identity(const std::string& text) {
    detail::Sha256 digest;
    digest.update(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    return digest.finish_hex();
}
std::string density_identity(const Eigen::MatrixXd& d) {
    detail::Sha256 digest;
    // Canonical row-major, big-endian binary64; dimensions belong to source.
    for (Eigen::Index i = 0; i < d.rows(); ++i) for (Eigen::Index j = 0; j < d.cols(); ++j) {
        double v = d(i,j) == 0 ? 0 : d(i,j);
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        std::array<std::uint8_t,8> bytes{};
        for (int k = 0; k < 8; ++k) bytes[k] = static_cast<std::uint8_t>(bits >> (56-8*k));
        digest.update(bytes.data(), bytes.size());
    }
    return digest.finish_hex();
}
void admit_density(const BuildPlan& p,const Eigen::MatrixXd& d) {
    if(d.rows()!=static_cast<Eigen::Index>(p.n_basis) || d.cols()!=d.rows())
        throw std::invalid_argument("MANTA density dimensions");
    if(!d.allFinite()) throw std::invalid_argument("MANTA density must be finite");
    const double scale=std::max(1.,d.cwiseAbs().maxCoeff());
    for(Eigen::Index i=0;i<d.rows();++i) for(Eigen::Index j=0;j<i;++j)
        if(std::abs(d(i,j)-d(j,i))>1e-12*scale)
            throw std::invalid_argument("MANTA ordered consumer requires symmetric real density");
}
} // namespace

Capability libint_capability(const KernelRequest& r) {
    Capability c;
    c.maximum_l = std::min(6, LIBINT2_MAX_AM_eri);
    if (r.kernel != "coulomb" || r.omega != 0 || r.centers != 4 || r.derivative != 0 || r.scalar != "real") {
        c.reason = "Reference provider admits real, zeroth-derivative, full-Coulomb 4c tiles";
        return c;
    }
    for (const auto& image : r.images) for (auto shift : image) if (shift != 0) {
        c.reason = "Molecular reference provider does not admit image translations";
        return c;
    }
    c.supported = true;
    return c;
}

namespace {
BuildPlan plan_libint_impl(const BasisSet& basis, const KernelRequest& request, const ResourceBudget& b,bool tiles_only) {
    const auto capability = libint_capability(request);
    if (!capability.supported) throw std::invalid_argument(capability.reason);
    if (b.backend_margin_bytes < (Count{8} << 20))
        throw std::invalid_argument("MANTA libint backend margin must be at least 8 MiB");
    BuildPlan p;
    if(tiles_only) p.schedule="shell-tiles";
    p.n_basis = basis.nbasis(); p.n_shells = basis.nshells();
    p.basis_payload_bytes = basis.name().size();
    if (!p.n_basis || !p.n_shells) throw std::invalid_argument("MANTA requires a nonempty basis");
    p.shell_calls = tiles_only?1:fourth(p.n_shells);
    p.scalar_terms = tiles_only?0:mul(2, fourth(p.n_basis));
    cap(p.shell_calls, b.shell_calls, "MANTA shell-call budget exceeded");
    cap(p.scalar_terms, b.scalar_terms, "MANTA scalar-contraction budget exceeded");
    p.target_bytes = tiles_only?0:mul(16, mul(p.n_basis, p.n_basis));
    // Conservatively reserve ten AO matrices for two targets, borrowed
    // densities, binding conversions and paired-spin/fused result lifetimes.
    p.modeled_peak_bytes = add(mul(5, p.target_bytes), add(b.other_live_bytes, b.backend_margin_bytes));
    cap(p.modeled_peak_bytes, b.memory_bytes, "MANTA matrix/input memory budget exceeded");
    Count max_size = 0, ao_count = 0;
    for (const auto& shell : basis.libint()) {
        if (shell.alpha.empty() || shell.contr.size() != 1 || shell.max_ln_coeff.size() != shell.alpha.size())
            throw std::invalid_argument("MANTA reference provider requires nonempty single-contraction shells");
        const auto& c = shell.contr[0];
        if (c.l < 0 || c.l > capability.maximum_l || c.coeff.size() != shell.alpha.size())
            throw std::invalid_argument("MANTA shell angular momentum/contraction exceeds provider capability");
        p.maximum_l = std::max(p.maximum_l, c.l);
        p.maximum_primitives = std::max<Count>(p.maximum_primitives, shell.alpha.size());
        max_size = std::max<Count>(max_size, shell.size());
        ao_count = add(ao_count, shell.size());
        p.basis_payload_bytes = add(p.basis_payload_bytes, add(24, mul(24, shell.alpha.size())));
    }
    if (ao_count != p.n_basis) throw std::invalid_argument("MANTA basis AO census mismatch");
    const Count primitives = fourth(p.maximum_primitives);
    cap(primitives, Count{1} << 53, "MANTA engine primitive extent exceeds exact integer range");
    p.primitive_quartets_upper_bound = mul(p.shell_calls, primitives);
    cap(p.primitive_quartets_upper_bound, b.primitive_quartets, "MANTA primitive-quartet budget exceeded");
    p.tile_elements = fourth(max_size);
    if(tiles_only) {
        p.scalar_terms=p.tile_elements;
        cap(p.scalar_terms,b.scalar_terms,"MANTA tile-element budget exceeded");
    }
    const Count cart = (p.maximum_l + 1) * (p.maximum_l + 2) / 2;
    // Libint's known allocations plus explicit allowance for shared tables,
    // allocator capacity and opaque controls. This model is not an RSS proof.
    p.engine_bytes = mul(primitives, sizeof(Libint_t));
    p.engine_bytes = add(p.engine_bytes, mul(mul(2, mul(p.maximum_primitives,p.maximum_primitives)), sizeof(libint2::ShellPair::PrimPairData)));
    p.engine_bytes = add(p.engine_bytes, mul(8, LIBINT2_PREFIXED_NAME(libint2_need_memory_eri)(p.maximum_l)));
    p.engine_bytes = add(p.engine_bytes, mul(16, fourth(cart)));
    p.engine_bytes = add(p.engine_bytes, mul(24, 4*p.maximum_l+1));
    const Count controls = add(4096, mul(p.n_shells, sizeof(libint2::Shell)+sizeof(libint2::Shell::Contraction)+3*sizeof(std::size_t)));
    p.modeled_peak_bytes = add(p.modeled_peak_bytes, add(mul(8,p.tile_elements), add(p.engine_bytes, add(mul(2,p.basis_payload_bytes), mul(2,controls)))));
    cap(p.modeled_peak_bytes, b.memory_bytes, "MANTA provider/workspace memory budget exceeded");
    // Numerical payload scans follow count-based resource admission.
    for (const auto& shell : basis.libint()) {
        for (auto x : shell.O) finite(x);
        for (auto x : shell.alpha) { finite(x); if (x <= 0) throw std::invalid_argument("MANTA requires positive exponents"); }
        for (auto x : shell.contr[0].coeff) finite(x);
        for (auto x : shell.max_ln_coeff)
            if (std::isnan(x) || x == std::numeric_limits<double>::infinity())
                throw std::invalid_argument("MANTA invalid derived shell screening coefficients");
    }
    return p;
}
} // namespace
BuildPlan plan_libint(const BasisSet& basis,const KernelRequest& request,const ResourceBudget& budget) {
    return plan_libint_impl(basis,request,budget,false);
}
BuildPlan plan_libint_tiles(const BasisSet& basis,const KernelRequest& request,const ResourceBudget& budget) {
    return plan_libint_impl(basis,request,budget,true);
}

struct Workspace::Impl {
    libint2::Engine engine;
    std::size_t primitives;
    int l;
    explicit Impl(const BuildPlan& p)
        : engine(libint2::Operator::coulomb, p.maximum_primitives, p.maximum_l, 0),
          primitives(p.maximum_primitives), l(p.maximum_l) { engine.set_precision(0); }
};
Workspace::Workspace(const BuildPlan& p) : backend_(p.provider) {
    if(backend_=="libint") {
        ensure_libint_initialized(); impl_ = std::make_unique<Impl>(p);
    } else if(backend_=="manta") values_.resize(mul(p.tile_elements,p.channel_batch));
    else throw std::invalid_argument("MANTA workspace provider");
}
Workspace::~Workspace() = default;
TileBatch IntegralProvider::evaluate_batch(const std::vector<std::array<std::size_t,4>>& indices,Workspace& workspace) const {
    if(indices.size()!=1) throw std::invalid_argument("MANTA reference provider admits one quartet per workspace evaluation");
    TileBatch result; result.tiles.push_back(evaluate(indices[0],workspace)); return result;
}

LibintProvider::LibintProvider(const BasisSet& basis, const KernelRequest& r, const ResourceBudget& b,bool tiles_only)
    : plan_(plan_libint_impl(basis,r,b,tiles_only)), basis_(basis) {
    source_ = identity("manta.libint.source.v1;" + auxiliary_basis_content_identity_sha256(basis_) +
        ";" LIBINT_VERSION ";coulomb;real;derivative=0;images=0;precision=0;libint-shell-axial-v1");
}
TileView LibintProvider::evaluate(const std::array<std::size_t,4>& indices, Workspace& workspace) const {
    const auto& shells = basis_.libint();
    for (auto i : indices) if (i >= shells.size()) throw std::out_of_range("MANTA shell index");
    if (workspace.backend_ != "libint" || workspace.impl_->primitives < plan_.maximum_primitives || workspace.impl_->l < plan_.maximum_l)
        throw std::invalid_argument("MANTA workspace is too small for provider");
    auto& engine = workspace.impl_->engine;
    engine.compute(shells[indices[0]],shells[indices[1]],shells[indices[2]],shells[indices[3]]);
    TileView tile;
    for (std::size_t i=0;i<4;++i) tile.shape[i] = shells[indices[i]].size();
    tile.values = engine.results()[0];
    if (tile.values) {
        std::size_t count=1;
        for (auto n:tile.shape) count*=n;
        for (std::size_t i=0;i<count;++i) finite(tile.values[i]);
    }
    return tile;
}

ReferenceJKBuilder::ReferenceJKBuilder(const BasisSet& basis, const KernelRequest& r, const ResourceBudget& b)
    : OrderedJKBuilder(std::make_shared<LibintProvider>(basis,r,b),b) {}
std::string LibintProvider::version() const { return LIBINT_VERSION; }
void OrderedJKBuilder::validate_density(const Eigen::MatrixXd& d) const { admit_density(plan(),d); }
JKResult OrderedJKBuilder::build(const Eigen::MatrixXd& d, bool make_j, bool make_k) const {
    if(!make_j && !make_k) throw std::invalid_argument("MANTA target selection");
    auto batch=contract({&d},make_j,make_k,plan().modeled_peak_bytes);
    JKResult result;
    if(make_j) result.J=std::move(batch.J[0]);
    if(make_k) result.K=std::move(batch.K[0]);
    result.receipt=std::move(batch.receipt);
    return result;
}
BatchPlan OrderedJKBuilder::plan_batch(Count count,bool make_j,bool make_k) const {
    if(!count) throw std::invalid_argument("MANTA density batch must be nonempty");
    if(!make_j && !make_k) throw std::invalid_argument("MANTA target selection");
    const auto& p=plan();
    BatchPlan batch;
    batch.n_densities=count; batch.shell_calls=p.shell_calls;
    batch.primitive_quartets_upper_bound=p.primitive_quartets_upper_bound;
    batch.kernel_terms=p.kernel_terms;
    batch.scalar_terms=mul(mul(count,static_cast<Count>(make_j)+static_cast<Count>(make_k)),fourth(p.n_basis));
    cap(batch.scalar_terms,budget_.scalar_terms,"MANTA batch scalar-contraction budget exceeded");
    // The single-density plan reserves ten matrices. Each further right side
    // reserves eight more for caller inputs, copies, targets, exports and
    // expression temporaries, plus controls and identity strings. No density
    // or target count multiplies the integral workspace or shell-call count.
    const Count extra=add(mul(64,mul(p.n_basis,p.n_basis)),512);
    batch.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(count-1,extra));
    cap(batch.modeled_peak_bytes,budget_.memory_bytes,"MANTA batch matrix/input memory budget exceeded");
    return batch;
}
BatchResult OrderedJKBuilder::build_batch(const std::vector<Eigen::MatrixXd>& densities,bool make_j,bool make_k) const {
    const auto batch=plan_batch(densities.size(),make_j,make_k);
    std::vector<const Eigen::MatrixXd*> views; views.reserve(densities.size());
    for(const auto& d:densities) views.push_back(&d);
    return contract(views,make_j,make_k,batch.modeled_peak_bytes);
}
BatchResult OrderedJKBuilder::contract(const std::vector<const Eigen::MatrixXd*>& densities,bool make_j,bool make_k,Count peak) const {
    const auto& p=plan();
    for(const auto* d:densities) admit_density(p,*d);
    BatchResult result;
    if(make_j) result.J.assign(densities.size(),Eigen::MatrixXd::Zero(p.n_basis,p.n_basis));
    if(make_k) result.K.assign(densities.size(),Eigen::MatrixXd::Zero(p.n_basis,p.n_basis));
    for(const auto* d:densities) result.density_identities.push_back(density_identity(*d));
    auto& receipt = result.receipt;
    receipt.provider = provider_->name(); receipt.provider_version = provider_->version();
    const auto request=provider_->kernel_request();
    receipt.kernel=request.kernel; receipt.omega=request.omega;
    receipt.source_identity = provider_->source_identity();
    if(densities.size()==1) receipt.density_identity=result.density_identities[0];
    else {
        std::string payload="manta.density-batch.v1;";
        for(const auto& id:result.density_identities) payload+=id+";";
        receipt.density_identity=identity(payload);
        receipt.algorithm="ordered-direct-batch";
    }
    receipt.modeled_peak_bytes = peak;
    if(p.channel_batch>1) receipt.contraction_strategy="shared-channels";
    Workspace workspace(p);
    const auto& shells = provider_->basis().libint();
    const auto offsets = shells.shell2bf();
    auto consume=[&](const std::array<std::size_t,4>& indices,const TileView& tile) {
        const auto a=indices[0],b=indices[1],c=indices[2],e=indices[3];
        ++receipt.shell_calls;
        receipt.primitive_quartets_requested += shells[a].alpha.size()*shells[b].alpha.size()*shells[c].alpha.size()*shells[e].alpha.size();
        if (!tile.values) { ++receipt.null_tiles; return; }
        std::size_t index=0;
        for (std::size_t i=0;i<tile.shape[0];++i) for (std::size_t j=0;j<tile.shape[1];++j)
        for (std::size_t k=0;k<tile.shape[2];++k) for (std::size_t l=0;l<tile.shape[3];++l) {
            const double v = tile.values[index++]; finite(v);
            const auto mu=offsets[a]+i, nu=offsets[b]+j, la=offsets[c]+k, si=offsets[e]+l;
            for(std::size_t rhs=0;rhs<densities.size();++rhs) {
                const auto& d=*densities[rhs];
                if (make_j) result.J[rhs](mu,nu) += d(la,si)*v;
                if (make_k) result.K[rhs](mu,la) += d(nu,si)*v;
            }
            ++receipt.integral_elements_consumed;
            receipt.density_terms_accumulated += densities.size()*(static_cast<Count>(make_j)+static_cast<Count>(make_k));
        }
    };
    std::vector<std::array<std::size_t,4>> batch;
    batch.reserve(p.channel_batch);
    auto flush=[&]() {
        const auto tiles=provider_->evaluate_batch(batch,workspace);
        receipt.primitive_evaluations_measured=tiles.primitive_evaluations_measured;
        receipt.primitive_quartets_evaluated+=tiles.primitive_quartets_evaluated;
        for(std::size_t i=0;i<batch.size();++i) consume(batch[i],tiles.tiles[i]);
        batch.clear();
    };
    // Preserve the ordered digestion schedule. Compatible contraction channels
    // can share primitive generation within a bounded consecutive batch.
    for(std::size_t a=0;a<shells.size();++a) for(std::size_t b=0;b<shells.size();++b)
    for(std::size_t c=0;c<shells.size();++c) for(std::size_t e=0;e<shells.size();++e) {
        batch.push_back({a,b,c,e});
        if(batch.size()==p.channel_batch) flush();
    }
    if(!batch.empty()) flush();
    for(const auto& values:result.J) if(!values.allFinite()) throw std::overflow_error("MANTA nonfinite J result");
    for(const auto& values:result.K) if(!values.allFinite()) throw std::overflow_error("MANTA nonfinite K result");
    return result;
}
Eigen::MatrixXd OrderedJKBuilder::build_J(const Eigen::MatrixXd& d) const { return build(d,true,false).J; }
Eigen::MatrixXd OrderedJKBuilder::build_K(const Eigen::MatrixXd& d) const { return build(d,false,true).K; }
std::pair<Eigen::MatrixXd,Eigen::MatrixXd> OrderedJKBuilder::build_K_pair(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) const {
    const auto batch=plan_batch(2,false,true);
    auto result=contract({&a,&b},false,true,batch.modeled_peak_bytes);
    return {std::move(result.K[0]),std::move(result.K[1])};
}
Eigen::MatrixXd OrderedJKBuilder::build_g_rhf(const Eigen::MatrixXd& d,double alpha) const {
    finite(alpha);
    auto result=build(d,true,alpha!=0);
    if (alpha!=0) result.J.noalias()-=0.5*alpha*result.K;
    if(!result.J.allFinite()) throw std::overflow_error("MANTA nonfinite fused J/K result");
    return std::move(result.J);
}
} // namespace vibeqc::manta
