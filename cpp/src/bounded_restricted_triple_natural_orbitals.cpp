#include "vibeqc/bounded_restricted_triple_natural_orbitals.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/detail/sha256.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "bounded TNO geometry does not support fast/finite-only math"
#endif

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Complex = std::complex<double>;
static_assert(sizeof(double) == 8 && sizeof(Complex) == 16
    && std::numeric_limits<double>::is_iec559 && FLT_EVAL_METHOD == 0
    && std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53,
    "bounded TNO geometry requires binary64 without excess evaluation precision");
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("bounded TNO count sum overflows");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("bounded TNO count product overflows");
    return a * b;
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("bounded TNO arithmetic is nonfinite");
    return x;
}
double product(double a, double b) {
    const double x = finite(a*b);
    if (a != 0.0 && b != 0.0 && x == 0.0)
        throw std::overflow_error("bounded TNO product loses a nonzero value to underflow");
    return x;
}
double rescale(double x, int exponent) {
    const double y = finite(std::scalbn(x, exponent));
    if (x != 0.0 && y == 0.0)
        throw std::overflow_error("bounded TNO rescaling loses a nonzero value");
    return y;
}
void environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double normal = std::numeric_limits<double>::min(), one = 1.0, half = 0.5;
    if (std::fegetround() != FE_TONEAREST || tiny*one != tiny || normal*half == 0.0)
        throw std::invalid_argument("bounded TNO requires round-to-nearest and gradual underflow");
}
struct Sum {
    double sum = 0.0, correction = 0.0;
    void include(double x) {
        finite(x);
        const double next = finite(sum+x);
        correction = finite(correction + (std::abs(sum) >= std::abs(x)
            ? finite(finite(sum-next)+x) : finite(finite(x-next)+sum)));
        sum = next;
    }
    double value() const { return finite(sum+correction); }
};
void accumulate(double x, double& value, double& correction) {
    Sum s{value, correction}; s.include(x); value = s.sum; correction = s.correction;
}
double norm_lane(double norm, double value) { return finite(std::hypot(norm, finite(value))); }
void fraction(double x) {
    if (!std::isfinite(x) || x <= 0.0 || x >= 1.0)
        throw std::invalid_argument("bounded TNO audit tolerances must lie strictly in (0,1)");
}
void nonnegative(double x) {
    if (!std::isfinite(x) || x < 0.0)
        throw std::invalid_argument("bounded TNO thresholds must be finite and nonnegative");
}
std::array<double, 13> option_values(const BoundedRestrictedTNOOptions& x) {
    return {x.occupation_cutoff, x.union_absolute_rank_cutoff, x.union_relative_rank_cutoff,
        x.maximum_union_column_reconstruction_error, x.input_orthonormality_tolerance,
        x.eigensystem_relative_reconstruction_tolerance, x.eigenvector_orthogonality_tolerance,
        x.union_negative_absolute_tolerance, x.union_negative_relative_tolerance,
        x.density_negative_absolute_tolerance, x.density_negative_relative_tolerance,
        x.occupation_ambiguity_absolute_guard, x.occupation_ambiguity_relative_guard};
}
void options_valid(const BoundedRestrictedTNOOptions& x) {
    for (double value : option_values(x)) nonnegative(value);
    fraction(x.maximum_union_column_reconstruction_error);
    fraction(x.input_orthonormality_tolerance);
    fraction(x.eigensystem_relative_reconstruction_tolerance);
    fraction(x.eigenvector_orthogonality_tolerance);
    if (x.union_relative_rank_cutoff >= 1.0 || x.union_negative_relative_tolerance >= 1.0
        || x.density_negative_relative_tolerance >= 1.0 || x.occupation_ambiguity_relative_guard >= 1.0)
        throw std::invalid_argument("bounded TNO relative selection controls must be below one");
    for (const auto* e : {&x.union_eigensolver, &x.density_eigensolver, &x.fock_eigensolver}) {
        if (!e->max_sweeps) throw std::invalid_argument("bounded TNO Jacobi sweeps must be positive");
        fraction(e->relative_offdiagonal_tolerance);
    }
}
void extent(U count, U lane = 8) {
    const U bytes = mul(count, lane);
    if (bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<U>::max()/8-16384)
        throw std::length_error("bounded TNO payload exceeds native or hash extent");
}
void view(const BoundedRestrictedTNORealView& v, U count) {
    extent(count);
    const auto address = reinterpret_cast<std::uintptr_t>(v.data);
    const U bytes = mul(8, count);
    if (v.element_count != count || (count && (!v.data || address % alignof(double)
        || bytes-1 > std::numeric_limits<std::uintptr_t>::max()-address)))
        throw std::invalid_argument("bounded TNO view has invalid exact extent/alignment");
}
bool same(const BoundedRestrictedTNOEdge& a, const BoundedRestrictedTNOEdge& b) {
    return a.rank == b.rank && a.coefficients.data == b.coefficients.data
        && a.coefficients.element_count == b.coefficients.element_count
        && a.connected_doubles.data == b.connected_doubles.data
        && a.connected_doubles.element_count == b.connected_doubles.element_count;
}
struct Census {
    std::array<U, 3> unique{}, multiplicity{}, prefix{};
    U count = 0, columns = 0, maximum_rank = 0, input_elements = 0;
};
Census census(const BoundedRestrictedTNOInput& in) {
    const U n = in.common_dimension;
    if (!n || !in.n_occupied || in.occupied[0] > in.occupied[1]
        || in.occupied[1] > in.occupied[2] || in.occupied[2] >= in.n_occupied)
        throw std::invalid_argument("bounded TNO requires positive dimensions and sorted occupied multiset");
    view(in.virtual_fock, mul(n,n));
    Census c; c.input_elements = mul(n,n);
    const std::array<U,3> left{in.occupied[0],in.occupied[0],in.occupied[1]};
    const std::array<U,3> right{in.occupied[1],in.occupied[2],in.occupied[2]};
    for (U e = 0; e < 3; ++e) {
        const auto& edge = in.edges[e];
        if (edge.occupied_i != left[e] || edge.occupied_j != right[e] || edge.rank > n)
            throw std::invalid_argument("bounded TNO edges must be (ij,ik,jk) with rank <= common dimension");
        view(edge.coefficients, mul(n,edge.rank));
        view(edge.connected_doubles, mul(edge.rank,edge.rank));
        U previous = c.count;
        for (U q = 0; q < c.count; ++q) {
            const auto& old = in.edges[c.unique[q]];
            if (old.occupied_i == edge.occupied_i && old.occupied_j == edge.occupied_j) {
                if (!same(old,edge))
                    throw std::invalid_argument("bounded TNO repeated edge must alias the identical pair view");
                previous = q; break;
            }
        }
        if (previous < c.count) { ++c.multiplicity[previous]; continue; }
        c.unique[c.count] = e; c.prefix[c.count] = c.columns; c.multiplicity[c.count] = 1;
        c.columns = add(c.columns,edge.rank); c.maximum_rank = std::max(c.maximum_rank,edge.rank);
        c.input_elements = add(c.input_elements,add(mul(n,edge.rank),mul(edge.rank,edge.rank)));
        ++c.count;
    }
    extent(c.input_elements);
    return c;
}
double column(const BoundedRestrictedTNOInput& in, const Census& c, U row, U col) {
    for (U q = 0; q < c.count; ++q) {
        const auto& e = in.edges[c.unique[q]];
        if (col >= c.prefix[q] && col-c.prefix[q] < e.rank)
            return e.coefficients.data[row*e.rank+col-c.prefix[q]];
    }
    throw std::logic_error("bounded TNO union column census is inconsistent");
}
class Digest {
public:
    void integer(U value) {
        std::array<std::uint8_t,8> bytes{};
        for (unsigned j=0;j<8;++j) bytes[j]=value>>(56-8*j);
        sha_.update(bytes.data(),bytes.size());
    }
    void real(double x) {
        finite(x); if (x == 0.0) x = 0.0;
        U bits=0; std::memcpy(&bits,&x,8); integer(bits);
    }
    void string(const std::string& s) {
        integer(s.size()); sha_.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
    }
    std::string finish() { return sha_.finish_hex(); }
private:
    detail::Sha256 sha_;
};
std::string input_hash(const BoundedRestrictedTNOInput& in,
    const BoundedRestrictedTNOOptions& options, const Census& c) {
    Digest d; d.string("vibeqc.bounded.restricted.triple-natural-orbitals.input.v1");
    d.integer(in.n_occupied); d.integer(in.common_dimension);
    for (U x : in.occupied) d.integer(x);
    for (double x : option_values(options)) d.real(x);
    for (const auto* e : {&options.union_eigensolver,&options.density_eigensolver,&options.fock_eigensolver}) {
        d.integer(e->max_sweeps); d.real(e->relative_offdiagonal_tolerance);
    }
    for (U q=0;q<c.count;++q) {
        const auto& e=in.edges[c.unique[q]];
        d.integer(e.occupied_i); d.integer(e.occupied_j); d.integer(e.rank); d.integer(c.multiplicity[q]);
        for (U x=0;x<e.coefficients.element_count;++x) d.real(e.coefficients.data[x]);
        for (U x=0;x<e.connected_doubles.element_count;++x) d.real(e.connected_doubles.data[x]);
    }
    for (U x=0;x<in.virtual_fock.element_count;++x) d.real(in.virtual_fock.data[x]);
    return d.finish();
}
double gram_error(const double* C, U n, U r) {
    double error=0.0;
    for (U a=0;a<r;++a) for (U b=0;b<r;++b) {
        Sum dot; for (U x=0;x<n;++x) dot.include(product(C[x*r+a],C[x*r+b]));
        error=norm_lane(error,finite(dot.value()-(a==b?1.0:0.0)));
    }
    return error;
}
struct Eigensystem { std::vector<Complex> vectors; std::vector<double> values; };
Eigensystem eigensystem(const std::vector<double>& matrix, U d,
    const HermitianJacobiOptions& options, const BoundedRestrictedTNOOptions& audits,
    BoundedRestrictedTNOEigensystemAudit& audit) {
    Eigensystem e; e.vectors.resize(mul(d,d)); e.values.resize(d);
    std::vector<Complex> scratch(mul(d,d));
    double maximum=0.0;
    for (U x=0;x<mul(d,d);++x) { scratch[x]=matrix[x]; maximum=std::max(maximum,std::abs(matrix[x])); }
    audit.eigensolver=hermitian_jacobi_in_place(scratch.data(),scratch.size(),e.vectors.data(),
        e.vectors.size(),e.values.data(),e.values.size(),d,options);
    if (audit.eigensolver.status != HermitianJacobiStatus::Success
        || audit.eigensolver.scaling_underflow_components)
        throw std::runtime_error("bounded TNO eigensolver failed or lost scaled input");
    if (maximum) std::frexp(maximum,&audit.matrix_scale_exponent);
    for (const auto& value:e.vectors)
        if (!std::isfinite(value.real()) || value.imag()!=0.0)
            throw std::runtime_error("bounded TNO real matrix produced nonreal eigenvectors");
    double max_lambda=0.0;
    for (double value:e.values) max_lambda=std::max(max_lambda,std::abs(rescale(value,-audit.matrix_scale_exponent)));
    for (U a=0;a<d;++a) for (U b=0;b<d;++b) {
        Sum reconstructed, overlap;
        for (U x=0;x<d;++x) {
            reconstructed.include(product(product(e.vectors[a*d+x].real(),
                rescale(e.values[x],-audit.matrix_scale_exponent)),e.vectors[b*d+x].real()));
            overlap.include(product(e.vectors[x*d+a].real(),e.vectors[x*d+b].real()));
        }
        const double original=rescale(matrix[a*d+b],-audit.matrix_scale_exponent);
        audit.scaled_matrix_frobenius_norm=norm_lane(audit.scaled_matrix_frobenius_norm,original);
        audit.scaled_reconstruction_frobenius_error=norm_lane(audit.scaled_reconstruction_frobenius_error,
            finite(original-reconstructed.value()));
        audit.orthogonality_frobenius_error=norm_lane(audit.orthogonality_frobenius_error,
            finite(overlap.value()-(a==b?1.0:0.0)));
    }
    audit.relative_reconstruction_error=audit.scaled_matrix_frobenius_norm==0.0
        ? audit.scaled_reconstruction_frobenius_error
        : finite(audit.scaled_reconstruction_frobenius_error/audit.scaled_matrix_frobenius_norm);
    if (audit.relative_reconstruction_error>audits.eigensystem_relative_reconstruction_tolerance
        || audit.orthogonality_frobenius_error>audits.eigenvector_orthogonality_tolerance)
        throw std::runtime_error("bounded TNO eigensystem failed independent reconstruction/orthogonality audit");
    // If eta=||U^T U-I||_F<1, polar Q obeys ||U-Q||_F <=
    // eta/(1+sqrt(1-eta)). Thus ||U L U^T-Q L Q^T||_F <=
    // ||L||_2 delta (sqrt(1+eta)+1). Norms below are measured, not
    // outward-rounded intervals: this is explicitly a numerical guard.
    const double eta=audit.orthogonality_frobenius_error;
    const double delta=eta/(1.0+std::sqrt(1.0-eta));
    audit.scaled_selection_error_guard=finite(audit.scaled_reconstruction_frobenius_error
        + max_lambda*delta*(std::sqrt(1.0+eta)+1.0));
    return e;
}
void release(std::vector<double>& x) { std::vector<double>().swap(x); }
void release(std::vector<Complex>& x) { std::vector<Complex>().swap(x); }
void limit(U amount,U cap,const char* text) { if (amount>cap) throw std::length_error(text); }
double budget(double absolute,double relative,double scale) {
    return finite(absolute+product(relative,scale));
}
double half_combination(double x,double y,bool difference) {
    const double sum=finite(difference?x-y:x+y), half=std::scalbn(sum,-1);
    if (std::scalbn(half,1)!=sum)
        throw std::overflow_error("bounded TNO symmetric/antisymmetric half-sum loses input");
    return half;
}
}  // namespace

BoundedRestrictedTNOMemoryPlan plan_bounded_restricted_triple_natural_orbitals(
    const BoundedRestrictedTNOInput& input,const BoundedRestrictedTNOOptions& options,
    const BoundedRestrictedTNOInventory& inventory) {
    options_valid(options); const Census c=census(input);
    if (!inventory.numerical_replicas || !inventory.backend_margin_bytes_per_replica)
        throw std::invalid_argument("bounded TNO replicas and backend margin must be positive");
    const U n=input.common_dimension,m=c.columns,u=std::min(n,m),R=c.maximum_rank;
    const U nn=mul(n,n),mm=mul(m,m),uu=mul(u,u),nu=mul(n,u);
    extent(mm,16); extent(uu,16); extent(nu);
    BoundedRestrictedTNOMemoryPlan p;
    p.common_dimension=n;p.unique_edge_count=c.count;p.union_columns=m;
    p.union_rank_upper_bound=u;p.maximum_edge_rank=R;
    p.borrowed_numerical_bytes=mul(8,c.input_elements);
    if (m) {
        p.gram_phase_bytes=add(mul(40,mm),mul(8,m));
        p.union_phase_bytes=add(mul(8,nu),add(mul(16,mm),mul(8,m)));
        p.density_phase_bytes=add(mul(8,nu),add(mul(16,uu),add(mul(8,mul(u,R)),mul(16,u))));
        p.density_eigen_phase_bytes=add(mul(8,nu),add(mul(40,uu),mul(8,u)));
        p.selection_phase_bytes=add(mul(16,nu),add(mul(16,uu),mul(8,u)));
        const auto semi=plan_restricted_pair_semicanonicalization(n,u);
        p.semicanonical_phase_bytes=add(mul(8,nu),add(mul(8,u),semi.peak_owned_numerical_bytes));
        p.peak_owned_numerical_bytes=std::max({p.gram_phase_bytes,p.union_phase_bytes,p.density_phase_bytes,
            p.density_eigen_phase_bytes,p.selection_phase_bytes,p.semicanonical_phase_bytes});
        p.output_numerical_bytes_upper_bound=add(mul(8,nu),mul(16,u));
    }
    // Fixed object/scalar reservation, including vector descriptors, SHA
    // state/short string payloads and nested semicanonical result. Allocator
    // implementation bookkeeping is covered only by the caller's margin.
    p.fixed_inventoried_object_bytes=8192U+3U*sizeof(BoundedRestrictedTNOInput)
        +2U*sizeof(BoundedRestrictedTNOResult)+2U*sizeof(Eigensystem)+sizeof(Census)
        +2U*sizeof(RestrictedPairSemicanonicalResult)+sizeof(p);
    p.control_storage_bytes_per_replica=add(p.fixed_inventoried_object_bytes,
        inventory.other_live_control_bytes_per_replica);
    U replica=add(p.borrowed_numerical_bytes,p.peak_owned_numerical_bytes);
    replica=add(replica,inventory.other_live_numerical_bytes_per_replica);
    replica=add(replica,p.control_storage_bytes_per_replica);
    replica=add(replica,inventory.backend_margin_bytes_per_replica);
    p.total_node_bytes=add(inventory.external_node_bytes,mul(inventory.numerical_replicas,replica));
    // Conservative abstract native work, not FLOPs. A full scalar-Jacobi
    // sweep and independent matrix reconstruction are cubic. These upper
    // counts also include finite/hash scans and original-column replay.
    U work=add(1,add(c.input_elements,nn));
    work=add(work,mul(n,mm));work=add(work,mul(add(options.union_eigensolver.max_sweeps,8),mul(mm,m)));
    work=add(work,mul(n,mul(m,u)));work=add(work,mul(n,mul(u,m)));
    work=add(work,mul(u,mul(m,R)));work=add(work,mul(uu,m));
    work=add(work,mul(add(options.density_eigensolver.max_sweeps,8),mul(uu,u)));
    work=add(work,mul(n,uu));work=add(work,mul(nn,u));
    work=add(work,mul(add(options.fock_eigensolver.max_sweeps,16),mul(uu,u)));
    p.work_units_upper_bound=mul(4096,work);
    return p;
}

BoundedRestrictedTNOResult bounded_restricted_triple_natural_orbitals(
    const BoundedRestrictedTNOInput& caller_input,const BoundedRestrictedTNOOptions& caller_options,
    const BoundedRestrictedTNOInventory& caller_inventory,const BoundedRestrictedTNOCaps& caller_caps) {
    const auto in=caller_input;const auto options=caller_options;
    const auto inventory=caller_inventory;const auto caps=caller_caps;
    BoundedRestrictedTNOResult out;out.options=options;out.occupied=in.occupied;
    out.memory=plan_bounded_restricted_triple_natural_orbitals(in,options,inventory);
    const auto& p=out.memory;
    limit(p.common_dimension,caps.maximum_common_dimension,"bounded TNO common dimension cap");
    limit(p.union_columns,caps.maximum_union_columns,"bounded TNO union column cap");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"bounded TNO owned numerical cap");
    limit(p.control_storage_bytes_per_replica,caps.maximum_control_storage_bytes_per_replica,"bounded TNO control cap");
    limit(p.total_node_bytes,caps.maximum_node_bytes,"bounded TNO node cap");
    limit(p.work_units_upper_bound,caps.maximum_work_units,"bounded TNO work cap");
    environment();const Census c=census(in);const U n=in.common_dimension,m=c.columns;
    out.input_identity_sha256=input_hash(in,options,c);
    for (U a=0;a<n;++a) for (U b=0;b<n;++b)
        if (in.virtual_fock.data[a*n+b]!=in.virtual_fock.data[b*n+a])
            throw std::invalid_argument("bounded TNO original virtual Fock must be exactly symmetric");
    double max_amplitude=0.0;
    for (U q=0;q<c.count;++q) {
        const auto& e=in.edges[c.unique[q]];
        const double orth=gram_error(e.coefficients.data,n,e.rank);
        out.maximum_input_orthonormality_error=std::max(out.maximum_input_orthonormality_error,orth);
        if (orth>options.input_orthonormality_tolerance)
            throw std::invalid_argument("bounded TNO pair columns are not orthonormal");
        for (U a=0;a<e.rank;++a) for (U b=0;b<e.rank;++b) {
            const double value=e.connected_doubles.data[a*e.rank+b];
            max_amplitude=std::max(max_amplitude,std::abs(value));
            if (e.occupied_i==e.occupied_j && value!=e.connected_doubles.data[b*e.rank+a])
                throw std::invalid_argument("bounded TNO diagonal connected doubles must be exactly symmetric");
        }
    }
    if (max_amplitude) std::frexp(max_amplitude,&out.amplitude_scale_exponent);
    for (U q=0;q<c.count;++q) {
        const auto& e=in.edges[c.unique[q]];
        for (U x=0;x<e.connected_doubles.element_count;++x) {
            const double value=e.connected_doubles.data[x];
            const double scaled=rescale(value,-out.amplitude_scale_exponent);
            if (std::scalbn(scaled,out.amplitude_scale_exponent)!=value)
                throw std::overflow_error("bounded TNO common amplitude scaling loses input");
        }
    }
    if (m) {
        std::vector<double> gram(mul(m,m));
        for (U a=0;a<m;++a) for (U b=a;b<m;++b) {
            Sum dot;for(U x=0;x<n;++x)dot.include(product(column(in,c,x,a),column(in,c,x,b)));
            gram[a*m+b]=gram[b*m+a]=dot.value();
        }
        auto g=eigensystem(gram,m,options.union_eigensolver,options,out.union_audit);
        const double top=std::max(0.0,g.values.back());
        const double negative=budget(options.union_negative_absolute_tolerance,options.union_negative_relative_tolerance,top);
        const double threshold=std::max(options.union_absolute_rank_cutoff,product(options.union_relative_rank_cutoff,top));
        U first=0;
        for(double value:g.values) if(value < -negative)
            throw std::runtime_error("bounded TNO union Gram has a materially negative eigenvalue");
        while(first<m && g.values[first]<=threshold)++first;
        const U u=m-first;
        if (!u || u>n) throw std::runtime_error("bounded TNO nonempty union has inadmissible numerical rank");
        out.union_rank=u;release(gram);
        std::vector<double> Q(mul(n,u));
        for(U a=0;a<u;++a) {
            const double inverse=finite(1.0/std::sqrt(g.values[first+a]));
            for(U x=0;x<n;++x) {
                Sum dot;for(U b=0;b<m;++b)dot.include(product(column(in,c,x,b),g.vectors[b*m+first+a].real()));
                Q[x*u+a]=product(dot.value(),inverse);
            }
        }
        out.union_orthonormality_error=gram_error(Q.data(),n,u);
        if(out.union_orthonormality_error>options.input_orthonormality_tolerance)
            throw std::runtime_error("bounded TNO union orthogonalization failed its original-space audit");
        // Reuse eigenvalue storage as the sole u-vector scratch. Every UNIQUE
        // original column is checked once, independently of Gram residuals.
        for(U b=0;b<m;++b) {
            for(U a=0;a<u;++a) {
                Sum dot;for(U x=0;x<n;++x)dot.include(product(Q[x*u+a],column(in,c,x,b)));
                g.values[a]=dot.value();
            }
            for(U x=0;x<n;++x) {
                Sum reconstructed;for(U a=0;a<u;++a)reconstructed.include(product(Q[x*u+a],g.values[a]));
                out.union_column_reconstruction_frobenius_error=norm_lane(
                    out.union_column_reconstruction_frobenius_error,finite(column(in,c,x,b)-reconstructed.value()));
            }
        }
        if(out.union_column_reconstruction_frobenius_error>options.maximum_union_column_reconstruction_error)
            throw std::runtime_error("bounded TNO union discarded relevant original pair columns");
        release(g.vectors);release(g.values);
        std::vector<double> density(mul(u,u),0.0);
        {
            std::vector<double> compensation(mul(u,u),0.0), overlap(mul(u,c.maximum_rank));
            std::vector<double> symmetric(u),antisymmetric(u);
            for(U q=0;q<c.count;++q) {
                const auto& e=in.edges[c.unique[q]];const U r=e.rank;
                for(U a=0;a<u;++a)for(U b=0;b<r;++b) {
                    Sum dot;for(U x=0;x<n;++x)dot.include(product(Q[x*u+a],e.coefficients.data[x*r+b]));
                    overlap[a*r+b]=dot.value();
                }
                const double weight=double(c.multiplicity[q])/(e.occupied_i==e.occupied_j?2.0:1.0);
                for(U l=0;l<r;++l) {
                    for(U a=0;a<u;++a) {
                        Sum s,anti;
                        for(U b=0;b<r;++b) {
                            const double x=rescale(e.connected_doubles.data[b*r+l],-out.amplitude_scale_exponent);
                            const double y=rescale(e.connected_doubles.data[l*r+b],-out.amplitude_scale_exponent);
                            s.include(product(overlap[a*r+b],half_combination(x,y,false)));
                            anti.include(product(overlap[a*r+b],half_combination(x,y,true)));
                        }
                        symmetric[a]=s.value();antisymmetric[a]=anti.value();
                    }
                    for(U a=0;a<u;++a)for(U b=a;b<u;++b) {
                        const double term=product(weight,finite(4.0*product(symmetric[a],symmetric[b])
                            +12.0*product(antisymmetric[a],antisymmetric[b])));
                        accumulate(term,density[a*u+b],compensation[a*u+b]);
                    }
                }
            }
            for(U a=0;a<u;++a)for(U b=a;b<u;++b) {
                const double raw=finite(density[a*u+b]+compensation[a*u+b]);
                const double value=finite(raw/3.0);
                if(raw!=0.0 && value==0.0)throw std::overflow_error("bounded TNO density average underflows");
                density[a*u+b]=density[b*u+a]=value;
            }
        }
        auto d=eigensystem(density,u,options.density_eigensolver,options,out.density_audit);
        release(density);
        const int physical_exponent=2*out.amplitude_scale_exponent;
        double max_occupation=0.0;
        for(double& value:d.values) {value=rescale(value,physical_exponent);max_occupation=std::max(max_occupation,std::abs(value));}
        const double density_negative=budget(options.density_negative_absolute_tolerance,options.density_negative_relative_tolerance,max_occupation);
        const double measured_guard=rescale(out.density_audit.scaled_selection_error_guard,
            out.density_audit.matrix_scale_exponent+physical_exponent);
        const double selection_guard=finite(measured_guard+budget(options.occupation_ambiguity_absolute_guard,
            options.occupation_ambiguity_relative_guard,max_occupation));
        U retained=0;Sum discarded;
        out.minimum_occupation=d.values.front();
        for(double value:d.values) {
            if(value < -density_negative)throw std::runtime_error("bounded TNO density has a materially negative eigenvalue");
            if(value<0.0)++out.negative_occupation_count;
            if(options.occupation_cutoff>0.0 && std::abs(finite(value-options.occupation_cutoff))<=selection_guard)
                throw std::runtime_error("bounded TNO positive occupation cutoff is numerically ambiguous");
            if(options.occupation_cutoff==0.0 || value>options.occupation_cutoff)++retained;
            else discarded.include(value);
        }
        out.retained_rank=retained;out.usable=retained!=0;
        out.discarded_occupation_sum=discarded.value();
        std::vector<double> selected(mul(n,retained));
        for(U a=0;a<retained;++a)for(U x=0;x<n;++x) {
            Sum dot;for(U b=0;b<u;++b)dot.include(product(Q[x*u+b],d.vectors[b*u+u-1-a].real()));
            selected[x*retained+a]=dot.value();
        }
        release(Q);release(d.vectors);
        std::reverse(d.values.begin(),d.values.end());out.occupations=std::move(d.values);
        const auto semi=plan_restricted_pair_semicanonicalization(n,retained);
        out.semicanonical=restricted_pair_semicanonicalize(selected.data(),selected.size(),
            in.virtual_fock.data,in.virtual_fock.element_count,n,retained,
            options.input_orthonormality_tolerance,semi.peak_owned_numerical_bytes,options.fock_eigensolver);
        if(out.semicanonical.fock_scaling_underflow_entries)
            throw std::runtime_error("bounded TNO semicanonical Fock scaling lost input");
    } else {
        out.semicanonical=restricted_pair_semicanonicalize(nullptr,0,in.virtual_fock.data,
            in.virtual_fock.element_count,n,0,options.input_orthonormality_tolerance,0,options.fock_eigensolver);
    }
    out.output_numerical_bytes=mul(8,add(add(mul(n,out.retained_rank),out.retained_rank),out.union_rank));
    if(out.output_numerical_bytes>p.output_numerical_bytes_upper_bound)
        throw std::logic_error("bounded TNO output exceeded its admitted count-only plan");
    environment();
    if(input_hash(in,options,c)!=out.input_identity_sha256)
        throw std::runtime_error("bounded TNO borrowed input changed during evaluation");
    Digest result;result.string("vibeqc.bounded.restricted.triple-natural-orbitals.result.v1");
    result.string(out.input_identity_sha256);result.integer(out.union_rank);result.integer(out.retained_rank);
    for(double x:out.occupations)result.real(x);
    for(double x:out.semicanonical.coefficients)result.real(x);
    for(double x:out.semicanonical.energies)result.real(x);
    out.result_identity_sha256=result.finish();
    return out;
}
}  // namespace vibeqc
