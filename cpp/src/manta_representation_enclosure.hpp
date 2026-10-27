#pragma once
#include "manta_verified_math.hpp"
#include "vibeqc/manta_md.hpp"
namespace vibeqc::manta {
// Same exact real Racah harmonics as md::representation: DLMF14.30.1/2,
// Rodrigues polynomial, no Condon-Shortley phase, rows m=-l..l.
// Contract5.65. Bounds certify coefficient construction, not equivariance.
inline std::vector<spectral_interval::Interval> representation_enclosure(int l,bool pure) {
    namespace si=spectral_interval;namespace vm=verified_math;using I=si::Interval;
    if(l<0 || l>7)throw std::invalid_argument("MANTA representation angular order");
    si::require_arithmetic();const auto point=[](double x)->I{return {x,x};};
    const auto& powers=md::cartesians(l);const Count nc=powers.size();
    std::vector<I> out((pure?2*l+1:nc)*nc);
    if(!pure) {for(Count i=0;i<nc;++i)out[i*nc+i]=point(1.);return out;}
    // 0!..14! are exactly representable integers below 2^53.
    double f[15];f[0]=1.;for(int i=1;i<15;++i)f[i]=f[i-1]*i;
    for(int row=-l;row<=l;++row) {
        const int m=std::abs(row);
        const auto scale=vm::sqrt_positive(vm::divide(point((m?2.:1.)*f[l-m]),point(f[l+m])));
        for(int k=0;k<=(l-m)/2;++k) {
            auto denominator=si::product(si::product(point(std::ldexp(1.,l)),point(f[k])),si::product(point(f[l-k]),point(f[l-m-2*k])));
            const auto legendre=vm::divide(point((k%2?-1.:1.)*f[2*l-2*k]),denominator);
            for(int j=0;j<=m;++j) {
                const int phase=row<0?(j%4==1?1:j%4==3?-1:0):(j%4==0?1:j%4==2?-1:0);
                if(!phase)continue;
                const auto azimuth=vm::divide(point(phase*f[m]),si::product(point(f[j]),point(f[m-j])));
                for(int x=0;x<=k;++x)for(int y=0;y<=k-x;++y) {
                    const int z=k-x-y;const md::Index term{m-j+2*x,j+2*y,l-m-2*k+2*z};
                    const Count col=std::find(powers.begin(),powers.end(),term)-powers.begin();
                    const auto multinomial=vm::divide(point(f[k]),si::product(si::product(point(f[x]),point(f[y])),point(f[z])));
                    out[(row+l)*nc+col]=si::add(out[(row+l)*nc+col],si::product(si::product(legendre,azimuth),multinomial));
                }
            }
        }
        for(Count col=0;col<nc;++col)out[(row+l)*nc+col]=si::product(out[(row+l)*nc+col],scale);
    }
    return out;
}
} // namespace vibeqc::manta
