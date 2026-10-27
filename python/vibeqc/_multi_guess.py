"""Multi-guess state selection shared by the legacy and selected open-shell routes.

The base SCF result competes with one alternate per seed, each a restarted
native SCF (``restart_opts.enabled`` with that seed). The lowest-energy
converged state with an internal-stability verdict wins; duplicate states are
recognised by their fingerprint. The caller supplies how one alternate runs,
so the legacy and selected molecular routes share exactly one policy.
"""
from __future__ import annotations


def select_lowest_stable_state(result, molecule, options, options_cls, run_alternate, seeds=None):
    """Return the base ``result`` or the best seeded alternate.

    ``seeds`` defaults to ``options.multi_guess_seeds``; the selected route
    passes the seeds it removed from the native options explicitly.
    """
    extra_seeds = list(options.multi_guess_seeds if seeds is None else seeds)
    if not extra_seeds or not result.converged:
        return result
    from .state_fingerprint import compute_state_fingerprint

    best = result
    seen: set[str] = set()
    n_electrons = molecule.n_electrons() - int(getattr(best, "ecp_total_ncore", 0) or 0)
    multiplicity = molecule.multiplicity
    n_alpha = (n_electrons + multiplicity - 1) // 2
    n_beta = (n_electrons - multiplicity + 1) // 2
    symbols = [atom.symbol for atom in molecule.atoms]

    def fingerprint(state):
        return compute_state_fingerprint(
            energy=float(state.energy), n_alpha=int(n_alpha), n_beta=int(n_beta),
            s_squared=float(getattr(state, "s_squared", 0.0)), molecule_symbols=symbols)

    seen.add(fingerprint(best))
    for seed in extra_seeds:
        # Value-copy the complete native option surface. A whitelist can
        # silently change the Hamiltonian (ECP centers, DFT+U) or turn an
        # implicit stability default into an explicit request that raises.
        opts_alt = options_cls(options)
        opts_alt.restart_opts.enabled = True
        opts_alt.restart_opts.seed = int(seed)
        try:
            alt = run_alternate(opts_alt)
        except RuntimeError:
            continue
        if not alt.converged:
            continue
        key = fingerprint(alt)
        if key in seen:
            continue
        if (getattr(alt, "internal_instability", False)
                    or getattr(alt, "external_instability", False)):
            continue
        # GitLab #566: a false instability flag can mean either a certified
        # stable channel or no verdict at all. A seed whose analysis
        # was requested but reached no verdict must not win on energy alone;
        # one that was never examined stays eligible, exactly as before.
        if getattr(alt, "stability_checked", False) and not getattr(alt, "stability_analysis_converged", False):
            continue
        # Only an eligible, verdict-carrying state participates in duplicate
        # suppression: a later seed may reach the same state with a verdict.
        seen.add(key)
        if alt.energy < best.energy - 1e-8:
            best = alt
    return best
