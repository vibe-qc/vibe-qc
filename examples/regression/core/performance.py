"""Declared-matrix full-SCF timing assessment (#574), independent of native QC.

Receipts are assertions by the campaign producer, not executable attestation.
The report preserves evidence identities and refuses incomplete comparisons.
"""
from __future__ import annotations

import hashlib
import json
import math
from statistics import median

from .case import CodeRow
from .compare import annotate_against_reference, paired_wall_ratio
from .spec import ExpectedRef


def protocol_digest(manifest: dict) -> str:
    """Content identity of the comparison protocol, excluding its digest field."""
    value = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _identity(value):
    return isinstance(value, str) and bool(value.strip())


def assess_performance(manifest: dict, samples: list[dict]) -> dict:
    """Assess all required cases; no averaging can hide a missing or slow case.

    Ratios include full setup and terminal confirmation. The observed range
    describes repeat variability, not a statistical confidence interval.
    """
    if not isinstance(manifest, dict) or not isinstance(samples, list):
        raise ValueError("protocol must be an object and samples a list")
    if any(not isinstance(s, dict) for s in samples):
        raise ValueError("each sample must be an object")
    if manifest.get("schema") != "vibeqc-performance-v1":
        raise ValueError("unsupported performance manifest schema")
    if manifest.get("sha256") != protocol_digest(manifest):
        raise ValueError("manifest digest does not match the frozen protocol")
    repeats = manifest.get("repetitions")
    if type(repeats) is not int or repeats < 3:
        raise ValueError("at least three declared repetitions are required")
    if manifest.get("cache_mode") not in ("cold", "warm"):
        raise ValueError("cache_mode must be cold or warm")
    if manifest.get("target_ratio") != 1.0 or isinstance(manifest["target_ratio"], bool):
        raise ValueError("CRYSTAL23 parity target_ratio must be 1.0")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(c, dict) for c in cases):
        raise ValueError("declare a nonempty required case matrix")
    ids = [case.get("id") for case in cases]
    if any(not _identity(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("required case ids must be unique nonempty strings")
    unknown = [s.get("case_id") for s in samples if s.get("case_id") not in ids]
    if unknown:
        raise ValueError("samples outside the declared case matrix")
    output = []
    used_runs = set()
    for case in cases:
        expected = ExpectedRef(**case["expected"])
        if not all(_identity(x) for x in (expected.system_id, expected.basis, expected.method_id)):
            raise ValueError("case system, basis and method identities must be nonempty")
        if not _positive(expected.tolerance_ha):
            raise ValueError("case energy tolerance must be finite and positive")
        reference = case.get("reference_code")
        if not _identity(reference) or reference == "vibeqc":
            raise ValueError("declare an external reference code")
        skala = "skala" in expected.method_id.lower()
        if skala and reference == "crystal":
            raise ValueError("CRYSTAL is not a same-functional SKALA reference")
        if not skala and reference != "crystal":
            raise ValueError("ordinary parity cases require CRYSTAL23 as reference")
        if skala and not _identity(case.get("checkpoint_identity")):
            raise ValueError("SKALA requires a real checkpoint identity")
        if not _identity(case.get("accuracy_evidence")):
            raise ValueError("case needs a qualified-accuracy evidence identity")
        pairs = [s for s in samples if s.get("case_id") == case["id"]]
        reasons, accepted, identities = [], [], None
        repetition_ids = [s.get("repeat") for s in pairs]
        if (any(type(n) is not int for n in repetition_ids)
                or sorted(repetition_ids) != list(range(1, repeats + 1))):
            reasons.append("missing, duplicated or unexpected repetition")
        for pair in pairs:
            try:
                ratio, detail = _pair(manifest, case, expected, pair, used_runs)
                current = tuple(pair[side]["receipt"][key]
                                for side in ("candidate", "reference")
                                for key in ("runtime_identity", "input_identity", "hardware_id",
                                            "cpus", "memory_limit_mb", "ranks", "threads_per_rank",
                                            "accelerator_identity"))
                if identities is not None and current != identities:
                    raise ValueError("runtime, input or allocation changed between repetitions")
                identities = current
                accepted.append(dict(repeat=pair["repeat"], upper_ratio=ratio, **detail))
            except (ValueError, TypeError, KeyError) as exc:
                reasons.append(f"repeat {pair.get('repeat')}: {exc}")
        ratios = [p["upper_ratio"] for p in accepted]
        status = "blocked" if reasons else (
            "pass" if max(ratios) <= 1. else
            "slow" if any(p["observed_ratio"] > 1. for p in accepted) else "uncertain")
        output.append(dict(
            case_id=case["id"], status=status, reasons=reasons, pairs=accepted,
            comparison_kind="same-functional checkpoint" if skala else "CRYSTAL23",
            upper_ratio_median=median(ratios) if ratios else None,
            upper_ratio_range=[min(ratios), max(ratios)] if ratios else None,
            accuracy_evidence=case["accuracy_evidence"],
        ))
    return dict(
        schema="vibeqc-performance-report-v1", protocol_sha256=manifest["sha256"],
        status="pass" if all(c["status"] == "pass" for c in output) else "incomplete",
        scope="declared matrix only; producer receipts are not independent attestation",
        variability="observed min/max; not a confidence interval",
        cache_mode=manifest["cache_mode"], cases=output,
        worst_upper_ratio=max((p["upper_ratio"] for c in output for p in c["pairs"]), default=None),
        worst_observed_ratio=max((p["observed_ratio"] for c in output for p in c["pairs"]), default=None),
    )


def _pair(manifest, case, expected, pair, used_runs):
    if pair.get("protocol_sha256") != manifest["sha256"]:
        raise ValueError("sample belongs to a different protocol")
    rows, receipts = [], []
    for side, code in (("candidate", "vibeqc"), ("reference", case["reference_code"])):
        data = pair[side]
        if not isinstance(data, dict):
            raise ValueError(f"{side} must be an object")
        row = CodeRow(**data["row"])
        receipt = data["receipt"]
        if not isinstance(receipt, dict):
            raise ValueError("receipt must be an object")
        if row.code != code:
            raise ValueError("unexpected code")
        # This is the canonical banner returned by runner_crystal, not a
        # guessed numeric prefix that could also admit CRYSTAL230.
        if code == "crystal" and row.code_version != "CRYSTAL23":
            raise ValueError("reference is not CRYSTAL23")
        if row.input_qualification != ("qualified" if side == "candidate" else "reference"):
            raise ValueError("physical input qualification missing")
        if row.n_retries != 0:
            raise ValueError("retry cost needs a separately qualified protocol")
        if not _identity(row.run_id):
            raise ValueError("missing campaign run_id")
        if type(row.n_iter) is not int or row.n_iter < 1:
            raise ValueError("complete SCF iteration count missing")
        for key in ("hardware_id", "runtime_identity", "input_identity", "output_identity",
                    "qualification_evidence", "measurement_evidence", "execution_identity",
                    "accelerator_identity"):
            if not _identity(receipt.get(key)):
                raise ValueError(f"missing {key}")
        execution = (code, receipt["execution_identity"])
        if execution in used_runs:
            raise ValueError("reused execution identity")
        used_runs.add(execution)
        if receipt.get("timing_scope") != "setup-through-final-confirmation":
            raise ValueError("timing excludes setup or final confirmation")
        if receipt.get("cache_mode") != manifest["cache_mode"]:
            raise ValueError("cache modes differ")
        if ("checkpoint_identity" in case
                and receipt.get("checkpoint_identity") != case["checkpoint_identity"]):
            raise ValueError("SKALA checkpoint identities differ or are missing")
        if type(receipt.get("cpus")) is not int or receipt["cpus"] < 1:
            raise ValueError("CPU allocation missing")
        for key in ("ranks", "threads_per_rank"):
            if type(receipt.get(key)) is not int or receipt[key] < 1:
                raise ValueError(f"positive integer {key} missing")
        if receipt["ranks"] * receipt["threads_per_rank"] > receipt["cpus"]:
            raise ValueError("rank/thread allocation oversubscribes CPUs")
        for key in ("memory_limit_mb", "peak_rss_mb", "wall_uncertainty_s"):
            if not _positive(receipt.get(key)):
                raise ValueError(f"positive measured {key} missing")
        phases = receipt.get("phase_wall_s", {})
        if (not isinstance(phases, dict) or set(phases) != {"setup", "scf", "final"}
                or not all(_positive(v) for v in phases.values())):
            raise ValueError("complete disjoint setup/scf/final timings missing")
        if not _positive(row.wall_s) or abs(sum(phases.values()) - row.wall_s) > receipt["wall_uncertainty_s"]:
            raise ValueError("phase accounting does not match full wall time")
        if receipt["wall_uncertainty_s"] >= row.wall_s:
            raise ValueError("timing uncertainty spans the measurement")
        rows.append(row)
        receipts.append(receipt)
    for key in ("hardware_id", "cpus", "memory_limit_mb", "ranks", "threads_per_rank",
                "accelerator_identity"):
        if receipts[0][key] != receipts[1][key]:
            raise ValueError(f"unmatched {key}")
    annotate_against_reference(rows, case["reference_code"], expected)
    ratio = paired_wall_ratio(*rows)
    if ratio is None:
        raise ValueError("convergence, case identity or energy comparison failed")
    # Conservative upper ratio includes declared timing measurement uncertainty.
    upper = ((rows[0].wall_s + receipts[0]["wall_uncertainty_s"])
             / (rows[1].wall_s - receipts[1]["wall_uncertainty_s"]))
    if not math.isfinite(upper):
        raise ValueError("uncertainty-adjusted timing ratio is nonfinite")
    return upper, dict(observed_ratio=ratio, candidate=pair["candidate"], reference=pair["reference"])
