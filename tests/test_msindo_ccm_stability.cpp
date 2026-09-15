// Standalone regression for #249. No libint or Python extension is needed.
// c++ -O1 -std=c++17 -Icpp/include -I<Eigen include directory> \
//   tests/test_msindo_ccm_stability.cpp cpp/src/davidson.cpp -o /tmp/ccm-stability
// /tmp/ccm-stability python/vibeqc/semiempirical/methods/msindo_params.json
#include <fstream>
#include <iomanip>
#include <iostream>
#include "vibeqc/semiempirical/methods/indo/ccm_engine.hpp"

using namespace vibeqc::semiempirical::indo;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "pass the MSINDO parameter JSON path");
        std::ifstream file(argv[1]);
        require(bool(file), "parameter file unavailable");
        std::string json((std::istreambuf_iterator<char>(file)), {});
        const auto p = load_msindo_params_from_json(json);
        int cases = 0;
        for (const auto spec : {std::pair<int, double>{3, 1.28}, {7, 1.28},
                                {9, 1.3}, {9, 1.375}, {9, 1.575}}) {
            const int n = spec.first;
            const double d = spec.second;
            std::vector<int> Z(n, 6);
            for (bool madelung : {false, true}) {
                double reference = 0;
                for (int variant = 0; variant < 4; ++variant) {
                    const Eigen::Vector3d axis = variant == 2
                        ? Eigen::Vector3d(0.36, 0.48, 0.8)
                        : Eigen::Vector3d(1, 0, 0);
                    const Eigen::Vector3d cell = n * d * axis;
                    std::vector<std::array<double, 3>> T{{cell.x(), cell.y(), cell.z()}}, C;
                    for (int i = 0; i < n; ++i) {
                        const int index = variant == 3 ? (i + 2) % n : i;
                        Eigen::Vector3d c = index * d * axis;
                        if (variant == 1) c *= 1.0 + 4.4e-10;
                        if (variant == 3) c += Eigen::Vector3d(2.3, -0.7, 0.5);
                        C.push_back({c.x(), c.y(), c.z()});
                    }
                    auto r = run_ccm_core(Z, C, T, p, madelung, 400, 1e-9);
                    require(r.converged, "chain did not converge");
                    require(r.n_iter <= 400, "CCM exceeded total SCF budget");
                    require(r.stability_checked && r.stability_analysis_converged,
                            "ambiguous-guess chain was not stability checked");
                    require(r.stability_eigenvalue >= -4e-6, "unstable saddle accepted");
                    const double e = r.total_energy / n;
                    if (variant == 0) reference = e;
                    require(std::abs(e - reference) < 1e-8,
                            "energy changed under coordinate noise, rotation or permutation");
                    ++cases;
                    std::cout << std::setprecision(15) << "C" << n << " d=" << d
                              << " madelung=" << madelung << " variant=" << variant
                              << " E/atom=" << e << " iterations=" << r.n_iter
                              << " curvature=" << r.stability_eigenvalue << '\n';
                }
            }
        }

        // Check the response, including affine Madelung terms, against an
        // energy curvature. C3 is a cheap witness with a negative mode.
        const int n = 3;
        std::vector<int> Z(n, 6), cz(n, 4), uat(12);
        std::vector<std::array<int, 2>> blocks{{0,4},{4,8},{8,12}};
        std::vector<Eigen::Vector3d> C, T{Eigen::Vector3d(3*1.28*ANGSTROM_TO_BOHR,0,0)};
        for (int i=0; i<n; ++i) C.push_back(Eigen::Vector3d(i*1.28*ANGSTROM_TO_BOHR,0,0));
        for (int i=0; i<12; ++i) uat[i]=i/4;
        const auto ws=build_wigner_seitz(C,T);
        Eigen::MatrixXd H,G;
        build_core_and_gamma_ccm(Z,C,blocks,12,p,ws,H,G);
        const auto ews=_ewald_ws_cells(ws);
        for (bool embedded : {false,true}) {
            CCMFockExtra extra;
            if (embedded) extra = [&](const Eigen::MatrixXd& P) {
                auto mad=_madelung_potential_1d(_net_charges(P,blocks,cz),ews,T[0]);
                Eigen::MatrixXd f=Eigen::MatrixXd::Zero(12,12);
                for(int i=0;i<12;++i)f(i,i)=-mad(i/4);
                return std::make_pair(f,2.0*mad.sum());
            };
            Eigen::MatrixXd seed=Eigen::MatrixXd::Identity(12,12);
            auto saddle=scf_rhf_driver(H,G,blocks,Z,6,p,400,1e-10,extra,&seed);
            require(saddle.converged,"curvature witness did not converge");
            auto st=ccm_orbital_stability(H,G,blocks,Z,6,p,saddle.density,extra);
            require(st.converged && st.curvature < -1e-3,"missing unstable witness");
            const auto energy=[&](const Eigen::MatrixXd& P) {
                Eigen::MatrixXd F=build_fock(H,G,P,uat,blocks,Z,p);
                double addition=0;
                if(extra){auto fe=extra(P);F+=fe.first;addition=fe.second;}
                return 0.5*(P.array()*(H+F).array()).sum()+addition;
            };
            const double e0=energy(ccm_stability_rotated_density(st,0));
            for(double h : {2e-3,1e-3,5e-4}) {
                const double fd=(energy(ccm_stability_rotated_density(st,h))
                    +energy(ccm_stability_rotated_density(st,-h))-2*e0)/(h*h);
                require(std::abs(fd-4*st.curvature)<2e-5,"orbital Hessian is not energy curvature");
            }
        }
        std::vector<std::array<double,3>> coords{{0,0,0},{1.28,0,0},{2.56,0,0}};
        std::vector<std::array<double,3>> translations{{3.84,0,0}};
        coords[1][0] += 0.003;  // a nonzero force, not a symmetry-zero check
        for (bool embedded : {false,true}) {
            const auto analytic=ccm_gradient_analytic(Z,coords,translations,p,
                embedded,0,400,1e-10);
            const auto fd=ccm_gradient_fd(Z,coords,translations,p,
                embedded,{},400,1e-10,1e-3,0);
            require(analytic.size()==3 && fd.size()==3,"CCM gradient unavailable");
            double force_norm=0, max_error=0;
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            {
                force_norm+=analytic[i][j]*analytic[i][j];
                max_error=std::max(max_error,std::abs(analytic[i][j]-fd[i][j]));
            }
            require(force_norm>1e-10,"gradient witness has only symmetry-zero forces");
            require(max_error<2e-6,"stable-state analytic and FD forces differ");
            std::cout << "gradient embedded=" << embedded << " max_error=" << max_error << '\n';
        }
        auto capped=run_ccm_core(Z,coords,translations,p,false,1,1e-9);
        require(!capped.converged && capped.n_iter==1,"one-step budget ignored");
        bool refused=false;
        try { (void)scf_rhf_ccm_driver(H,G,blocks,Z,6,p,26,1e-9,{}); }
        catch(const std::runtime_error& e) {
            refused=std::string(e.what()).find("unstable SCF saddle")!=std::string::npos;
        }
        require(refused,"an unstable short-budget result was silently returned");
        std::cout << "PASS: " << cases
                  << " chain cases, both curvature/gradient controls, and budget guards\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
