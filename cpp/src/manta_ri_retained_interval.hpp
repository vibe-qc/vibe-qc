#pragma once
#include "manta_ri_interval.hpp"
#include "vibeqc/manta_admission.hpp"

namespace vibeqc::manta::retained_interval {
using namespace spectral_interval;
using Complex=std::complex<double>;
inline ComplexInterval conjugate(ComplexInterval x) { return {x.real,{-x.imag.hi,-x.imag.lo}}; }
// Sum exact stored channel values, keeping their admitted complex L1 radii
// separate from accumulation roundoff. Each radius is propagated once.
class Accumulation {
    std::vector<ComplexInterval> values_;
    std::vector<double> radii_;
public:
    explicit Accumulation(Count count):values_(count),radii_(count) { require_arithmetic(); }
    void add_entry(Count q,Complex value,double radius) {
        if(q>=values_.size() || !std::isfinite(value.real()) || !std::isfinite(value.imag()) || !std::isfinite(radius) || radius<0.)
            throw std::invalid_argument("MANTA channel arithmetic requires finite values and nonnegative entry radii");
        values_[q]=add(values_[q],point(value));radii_[q]=plus_up(radii_[q],radius);
    }
    std::vector<double> errors(const std::vector<Complex>& actual) const {
        if(actual.size()!=values_.size())throw std::logic_error("MANTA channel arithmetic extent mismatch");
        std::vector<double> out(actual.size());
        for(Count q=0;q<out.size();++q)out[q]=plus_up(magnitude(sub(point(actual[q]),values_[q])),radii_[q]);
        return out;
    }
};
// Rump, Acta Numerica 19 (2010), DOI 10.1017/S096249291000005X.
// Same stored F/T and Sun2017 RI convention as the physical retained sweep.
// This owns interval mode blocks, not another coupling tensor or source pass.
class Audit {
    Count n_,nk_,ns_,n2_,mesh_,modes_=0;
    const std::vector<Complex>& d_;
    const std::vector<double>& w_;
    std::vector<ComplexInterval> b_,j_,k_,weighted_;
    Interval energy_;
public:
    Audit(Count n,Count nk,Count ns,const std::vector<Complex>& d,const std::vector<double>& w,bool j,bool k)
        :n_(n),nk_(nk),ns_(ns),n2_(admission::square(n)),mesh_(admission::mul(nk,n2_)),d_(d),w_(w),
         j_(j?mesh_:0),k_(k?admission::mul(ns,mesh_):0),weighted_(j?mesh_:0) {
        require_arithmetic();
        if(j)for(Count q=0;q<mesh_;++q) {
            ComplexInterval total;
            for(Count spin=0;spin<ns_;++spin)total=add(total,point(d_[spin*mesh_+q]));
            weighted_[q]=product(point(Complex(w_[q/n2_])),total);
        }
    }
    void block(Count modes) { modes_=modes;b_.assign(admission::mul(modes,mesh_),{}); }
    void fitted(Count k,Count mode,Count pair,Complex factor,Complex tile) {
        auto& value=b_[(k*modes_+mode)*n2_+pair];
        value=add(value,product(std::conj(factor),tile));
    }
    void contract() {
        for(Count mode=0;mode<modes_;++mode) {
            if(!j_.empty()) {
                ComplexInterval rho;
                for(Count k=0;k<nk_;++k)for(Count pair=0;pair<n2_;++pair)
                    rho=add(rho,product(conjugate(b_[(k*modes_+mode)*n2_+pair]),weighted_[k*n2_+pair]));
                energy_=add(energy_,product({.5,.5},product(conjugate(rho),rho).real));
                for(Count k=0;k<nk_;++k)for(Count pair=0;pair<n2_;++pair) {
                    auto& value=j_[k*n2_+pair];value=add(value,product(b_[(k*modes_+mode)*n2_+pair],rho));
                }
            }
            if(!k_.empty())for(Count k=0;k<nk_;++k)for(Count spin=0;spin<ns_;++spin) {
                const Count offset=(k*modes_+mode)*n2_,out=(spin*nk_+k)*n2_;
                std::vector<ComplexInterval> bd(n2_);
                for(Count i=0;i<n_;++i)for(Count j=0;j<n_;++j)for(Count a=0;a<n_;++a)
                    bd[i*n_+j]=add(bd[i*n_+j],product(b_[offset+i*n_+a],point(d_[out+a*n_+j])));
                for(Count i=0;i<n_;++i)for(Count j=0;j<n_;++j) {
                    ComplexInterval value;
                    for(Count a=0;a<n_;++a)value=add(value,product(bd[i*n_+a],conjugate(b_[offset+j*n_+a])));
                    k_[out+i*n_+j]=add(k_[out+i*n_+j],product(point(Complex(w_[k])),value));
                }
            }
        }
    }
    std::vector<double> errors(const std::vector<Complex>& actual,bool exchange) const {
        const auto& expected=exchange?k_:j_;
        if(actual.size()!=expected.size())throw std::logic_error("MANTA retained arithmetic extent mismatch");
        std::vector<double> out(actual.size());
        for(Count q=0;q<out.size();++q)out[q]=magnitude(sub(point(actual[q]),expected[q]));
        return out;
    }
    double energy_error(double actual) const {
        const auto error=sub({actual,actual},energy_);return std::max(std::abs(error.lo),std::abs(error.hi));
    }
};
} // namespace vibeqc::manta::retained_interval
