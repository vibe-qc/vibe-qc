// Standalone regression for #249. No libint or Python extension is needed.
// c++ -O1 -std=c++17 -Icpp/include -I<Eigen include directory> \
//   tests/test_msindo_ccm_stability.cpp cpp/src/davidson.cpp \
//   cpp/src/soscf.cpp cpp/src/quadratic_scf.cpp -o /tmp/ccm-stability
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
        for (int nocc : {1, 3})
            for (int budget : {1, 7, 26, 400}) {
                Eigen::VectorXd v(4);
                v << 1, 2, 3, 4;
                v.normalize();
                Eigen::MatrixXd target = 2 * v * v.transpose();
                if (nocc == 3)
                    target = 2 * Eigen::MatrixXd::Identity(4, 4) - target;
                int calls = 0;
                auto seed = ccm_frontier_seed(
                    Eigen::MatrixXd::Zero(4, 4), nocc, budget, [&](const Eigen::MatrixXd &P) {
                        ++calls;
                        Eigen::MatrixXd F = P - target;
                        return std::make_pair(F, .5 * F.squaredNorm());
                    });
                require(calls == seed.evaluations && calls <= budget, "unaccounted seed work");
                require((seed.density * seed.density - 2 * seed.density).norm() < 1e-12,
                        "nonphysical projector");
                if (budget == 400)
                    require(seed.completed && seed.energy < 1e-16,
                            "known quadratic minimizer not recovered");
                else
                    require(!seed.completed && seed.evaluations == budget,
                            "inconsistent seed exhaustion");
            }
        for (bool hole : {false, true})
            for (int budget : {1, 7, 26, 400}) {
                const int n = hole ? 5 : 4, nocc = hole ? 4 : 1;
                Eigen::VectorXd v(n), eps(n);
                if (hole) {
                    v << 3, 4, 1, 1, 1;
                    eps << -1e-5, -1e-5, 0, 0, 0;
                } else {
                    v << 1, 2, 3, 4;
                    eps << 0, 0, 1e-5, 1e-5;
                }
                v.normalize();
                Eigen::MatrixXd target = 2 * v * v.transpose();
                if (hole) target = 2 * Eigen::MatrixXd::Identity(n, n) - target;
                int calls = 0;
                const auto seed = ccm_frontier_seed(
                    eps.asDiagonal(), nocc, budget, [&](const Eigen::MatrixXd &P) {
                        ++calls;
                        Eigen::MatrixXd F = P - target;
                        return std::make_pair(F, .5 * F.squaredNorm());
                    });
                require(calls == seed.evaluations && calls <= budget,
                        "unaccounted adaptive seed work");
                require((seed.density * seed.density - 2 * seed.density).norm() < 1e-12,
                        "nonphysical adaptive projector");
                require(std::abs(seed.density.trace() - 2 * nocc) < 1e-12,
                        "adaptive seed changed occupations");
                require((seed.fock - seed.density + target).norm() < 1e-12,
                        "adaptive exhaustion lost its physical Fock");
                if (budget == 400)
                    require(seed.completed && seed.energy < 1e-16 && seed.expansions == 1
                            && seed.orbital_end - seed.orbital_begin == n,
                            "adaptive known minimizer not recovered");
                else
                    require(!seed.completed && seed.evaluations == budget,
                            "inconsistent adaptive seed exhaustion");
            }
        for (int nocc : {1, 2, 3})
            for (int budget : {0, 1, 3, 200}) {
                Eigen::VectorXd v(4); v << 1, 2, 3, 4; v.normalize();
                const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4, 4) - 2 * v * v.transpose();
                const Eigen::MatrixXd target = 2 * Q.leftCols(nocc) * Q.leftCols(nocc).transpose();
                Eigen::MatrixXd P = Eigen::MatrixXd::Zero(4, 4);
                for (int i = 0; i < nocc; ++i) P(i, i) = 2;
                const Eigen::MatrixXd F = P - target;
                int calls = 0;
                const auto r = ccm_minimize_orbitals(P, F, .5 * F.squaredNorm(), nocc,
                    budget, 1e-10, [&](const Eigen::MatrixXd &trial) {
                        ++calls;
                        require((trial * trial - 2 * trial).norm() < 1e-12,
                                "nonphysical orbital trial");
                        require(std::abs(trial.trace() - 2 * nocc) < 1e-12,
                                "orbital continuation changed occupations");
                        Eigen::MatrixXd f = trial - target;
                        return std::make_pair(f, .5 * f.squaredNorm());
                    });
                require(calls == r.n_iter && calls <= budget, "unaccounted orbital trial");
                require(std::abs(r.electronic_energy - .5 * (r.density-target).squaredNorm()) < 1e-12,
                        "orbital exhaustion lost physical energy");
                if (budget == 200)
                    require(r.converged && r.orbital_optimization_used && r.electronic_energy < 1e-16,
                            "orbital known minimizer not recovered");
                else
                    require(!r.converged && r.n_iter == budget,
                            "inconsistent orbital exhaustion");
            }
        {
            std::vector<int> Z(3, 6);
            std::vector<std::array<double, 3>> C;
            for (int i = 0; i < 3; ++i)
                C.push_back({(i * 1.28 * ANGSTROM_TO_BOHR) / ANGSTROM_TO_BOHR, 0, 0});
            auto r = run_ccm_core(Z, C, {{3.84, 0, 0}}, p, false, 200, 1e-9);
            require(r.converged, "C3 roundtrip default budget failed");
        }
        // Frozen density-continuation energies of locally distorted C19 (#249).
        // The old fresh seed was 0.006867 Ha higher despite a stable final Hessian.
        for (const auto spec : {std::pair<double, double>{1e-4, -111.11041903701442},
                                {3e-4, -111.11041874136663},
                                {1e-3, -111.11041537835774},
                                {3e-3, -111.11038581419405},
                                {4e-3, -111.11035994671295}}) {
            for (int variant = 0; variant < 3; ++variant) {
                const Eigen::Vector3d axis = variant == 1
                    ? Eigen::Vector3d(.36, .48, .8) : Eigen::Vector3d(1, 0, 0);
                std::vector<std::array<double, 3>> coords;
                for (int i = 0; i < 19; ++i) {
                    const int index = variant == 2 ? (i + 2) % 19 : i;
                    Eigen::Vector3d c = (index * 1.28 + (index == 1 ? spec.first : 0.)) * axis;
                    if (variant == 2) c += Eigen::Vector3d(2.3, -.7, .5);
                    coords.push_back({c.x(), c.y(), c.z()});
                }
                const Eigen::Vector3d cell = 19 * 1.28 * axis;
                auto r = run_ccm_core(std::vector<int>(19, 6), coords,
                    {{cell.x(), cell.y(), cell.z()}}, p, false, 400, 1e-9, 0, true);
                std::cout << std::setprecision(16) << "distorted " << spec.first
                          << " variant " << variant << " converged=" << r.converged
                          << " iterations=" << r.n_iter << " seed=" << r.frontier_seed_evaluations
                          << " E=" << r.total_energy << '\n';
                require(r.converged && r.stability_analysis_converged,
                        "distorted C19 did not converge and audit");
                require(std::abs(r.total_energy - spec.second) < 1e-8,
                        "distorted C19 selected higher stable state");
                require(r.frontier_seed_orbitals == 4 && r.frontier_seed_expansions >= 1,
                        "distorted C19 did not report adaptive frontier");
                require(r.n_iter <= 400 && r.stability_eigenvalue >= -4e-6,
                        "distorted C19 budget/stability failed");
            }
        }
        int cases = 0;
        for (const auto spec : {std::pair<int, double>{3, 1.28},
                                {7, 1.28},
                                {9, 1.3},
                                {9, 1.375},
                                {9, 1.575},
                                {19, 1.28}}) {
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

        // #249 (verifier, 2026-09-26): under the lowest_stable policy a C3
        // chain bent by dy at atom 0 stays on the continuous lower branch at
        // every Hcore gap; from the Hcore guess 3e-3 bohr lands 16.6 mHa above
        // it. The reference policy returns that state but flags it.
        for (bool embedded : {false, true}) {
            std::vector<int> Zc(3, 6);
            double previous = 0, linear = 0;
            int index = 0;
            for (double dy : {0.0, 1e-4, 1e-3, 3e-3, 1e-2, 5e-2}) {
                std::vector<std::array<double, 3>> coords{
                    {0, dy / ANGSTROM_TO_BOHR, 0}, {1.283, 0, 0}, {2.56, 0, 0}};
                std::vector<std::array<double, 3>> translations{{3.84, 0, 0}};
                auto r = run_ccm_core(Zc, coords, translations, p, embedded, 400, 1e-9,
                                      0, /*lowest_stable=*/true);
                require(r.converged && r.stability_checked && r.stability_analysis_converged,
                        "bent chain was not converged and audited");
                if (index == 0) linear = r.total_energy;
                else require(std::abs(r.total_energy - previous) < 2e-4,
                             "bent chain energy is discontinuous");
                require(r.total_energy < linear + 1e-9,
                        "bent chain landed above the linear chain");
                std::cout << "bent C3 embedded=" << embedded << " dy=" << dy
                          << " E=" << r.total_energy << " iterations=" << r.n_iter
                          << " curvature=" << r.stability_eigenvalue << '\n';
                previous = r.total_energy;
                ++index;
            }
        }

        {   // Reference policy: the resolved-gap saddle is returned, flagged.
            std::vector<int> Zc(3, 6);
            std::vector<std::array<double, 3>> coords{
                {0, 3e-3 / ANGSTROM_TO_BOHR, 0}, {1.283, 0, 0}, {2.56, 0, 0}};
            std::vector<std::array<double, 3>> translations{{3.84, 0, 0}};
            auto r = run_ccm_core(Zc, coords, translations, p, false, 400, 1e-9);
            require(r.converged && r.stability_checked && r.internal_instability,
                    "reference policy hid a resolved-gap saddle");
            require(r.stability_eigenvalue < -1e-2, "flagged state is not a saddle");
            require(std::abs(r.total_energy - (-17.5358307608)) < 1e-8,
                    "reference policy left the reference trajectory's state");
            std::cout << "reference policy dy=0.003 E=" << r.total_energy
                      << " flagged curvature=" << r.stability_eigenvalue << '\n';
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
        Eigen::MatrixXd unstable_witness;
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
            if (!embedded) unstable_witness = saddle.density;
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
        require(capped.frontier_seed_evaluations==1 && !capped.frontier_seed_completed,
                "initial frontier exceeded one-step budget");
        require(std::isfinite(capped.total_energy) && capped.total_energy!=0,
                "exhausted frontier lost its last energy");
        auto seed_limited=scf_rhf_ccm_driver(H,G,blocks,Z,6,p,26,1e-9,{});
        require(!seed_limited.converged && seed_limited.n_iter==26
                && seed_limited.frontier_seed_evaluations==26,
                "initial frontier hid an exhausted budget");
        require(seed_limited.density.rows()==12 && std::isfinite(seed_limited.electronic_energy),
                "exhausted frontier lost its last density");
        // Keep the original Hcore control. Its degenerate projector can land
        // on a stable state on another compiler, so require either a verified
        // stable result or explicit refusal, never an accepted saddle.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> initial(H);
        const Eigen::MatrixXd P_initial=2*initial.eigenvectors().leftCols(6)
                                           *initial.eigenvectors().leftCols(6).transpose();
        bool refused=false;
        try {
            auto r=scf_rhf_ccm_driver(H,G,blocks,Z,6,p,26,1e-9,{},false,&P_initial);
            require(r.converged && r.stability_analysis_converged
                    && r.stability_eigenvalue >= -4e-6,
                    "Hcore control returned an unverified or unstable state");
            refused=true;
        }
        catch(const std::runtime_error& e) {
            refused=std::string(e.what()).find("unstable SCF saddle")!=std::string::npos;
        }
        require(refused,"an unstable short-budget result was silently returned");
        // This independently verified saddle is a compiler-independent
        // exhaustion control. Two iterations certify stationarity, leaving
        // no budget for the mandatory stability restart.
        refused=false;
        try { (void)scf_rhf_ccm_driver(H,G,blocks,Z,6,p,2,1e-9,{},false,&unstable_witness); }
        catch(const std::runtime_error& e) {
            refused=std::string(e.what()).find("unstable SCF saddle")!=std::string::npos;
        }
        require(refused,"stationary saddle escaped the restart budget guard");
        std::cout << "PASS: " << cases
                  << " chain cases, the bent-chain continuity sweep, both "
                     "curvature/gradient controls, and budget guards\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
