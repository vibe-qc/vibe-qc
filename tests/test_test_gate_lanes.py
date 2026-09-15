"""Unit tests for scripts/test_gate lane selection.

The full gate is intentionally expensive; these tests pin the pure manifest
logic without spawning pytest subprocesses.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "test_gate" / "run_full_suite.py"
)
MANIFEST_PATH = RUNNER_PATH.with_name("lane_manifest.json")
SUITE_MANIFEST_PATH = RUNNER_PATH.with_name("suite_manifest.json")
GENERATOR_PATH = RUNNER_PATH.with_name("update_suite_manifest.py")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_module(RUNNER_PATH, "run_full_suite_under_test")
generator = _load_module(GENERATOR_PATH, "update_suite_manifest_under_test")


def test_per_target_runner_sandboxes_persistent_state_and_scrubs_pytest_opts(
    monkeypatch,
):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("HOME", "/caller/home")
    monkeypatch.setenv("VQ_STATE_DIR", "/caller/live-vq-state")
    monkeypatch.setenv(
        "VQ_TEST_SYSTEM_CONFIG_FILE",
        "/caller/live-system-config",
    )
    monkeypatch.setenv("VQ_WEB_TOKEN_FILE", "/caller/live-web-token")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--noconftest")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "foreign parent test")
    monkeypatch.setenv("PYTEST_VERSION", "foreign parent version")

    with runner._isolated_test_environment(
        repo,
        "vibe-queue/tests/test_paths.py::TestStateRoot::test_env_override_takes_precedence",
    ) as env:
        sandbox_home = Path(env["HOME"])
        assert sandbox_home != Path("/caller/home")
        assert Path(env["VQ_STATE_DIR"]) != Path("/caller/live-vq-state")
        assert "PYTEST_ADDOPTS" not in env
        assert "PYTEST_CURRENT_TEST" not in env
        assert "PYTEST_VERSION" not in env
        for name in (
            "HOME",
            "TMPDIR",
            "XDG_DATA_HOME",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "VQ_STATE_DIR",
            "VQ_CONFIG_DIR",
            "VQ_ARCHIVE_DIR",
            "VQ_MULTI_USER_ROOT",
            "VQ_TEST_SANDBOX_ROOT",
        ):
            assert Path(env[name]).is_dir()
            assert Path(env[name]).resolve().is_relative_to(
                Path(env["VQ_TEST_SANDBOX_ROOT"]).resolve()
            )
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(
            (repo / "vibe-queue" / "src").resolve()
        )
        assert Path(env["VQ_WEB_TOKEN_FILE"]) == (
            Path(env["VQ_CONFIG_DIR"]) / "web-token"
        )
        assert Path(env["VQ_TEST_SYSTEM_CONFIG_FILE"]) == (
            Path(env["VQ_CONFIG_DIR"]) / "system-config.toml"
        )
        assert env["VQ_TEST_SYSTEM_CONFIG_FILE"] != "/caller/live-system-config"
        assert env["VQ_WEB_TOKEN_FILE"] != "/caller/live-web-token"
        assert Path(env["VQ_TEST_SHORT_TMPDIR"]) == Path(env["TMPDIR"])

    assert not sandbox_home.exists()


@pytest.mark.skipif(
    not (RUNNER_PATH.parents[2] / "vibe-queue/tests/test_rpc.py").is_file(),
    reason="queue RPC integration belongs to the separate vibe-queue repository",
)
def test_per_target_runner_executes_rpc_socket_node_in_short_sandbox():
    repo = Path(__file__).resolve().parents[1]
    result = runner.run_one(
        "vibe-queue/tests/test_rpc.py::"
        "TestServerLifecycle::test_start_creates_socket_with_right_mode_single_user",
        repo,
        sys.executable,
        file_timeout=45,
        test_timeout=20,
    )

    assert result["status"] == "PASS", result["tail"]


def test_per_target_runner_preserves_non_vq_environment(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("HOME", "/caller/core-home")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k caller-core-selection")

    with runner._isolated_test_environment(
        repo,
        "tests/test_banner.py",
    ) as env:
        assert env["HOME"] == "/caller/core-home"
        assert env["PYTEST_ADDOPTS"] == "-k caller-core-selection"
        assert "VQ_TEST_SANDBOX_ROOT" not in env
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["OMP_NUM_THREADS"]
        assert env["OPENBLAS_NUM_THREADS"]


def test_per_target_runner_keeps_the_platform_temporary_parent(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    real_temporary_directory = runner.tempfile.TemporaryDirectory
    calls = []

    def recording_temporary_directory(*args, **kwargs):
        calls.append((args, kwargs))
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(
        runner.tempfile,
        "TemporaryDirectory",
        recording_temporary_directory,
    )
    with runner._isolated_test_environment(
        repo,
        "vibe-queue/tests/test_paths.py",
    ):
        pass

    assert len(calls) == 1
    assert "dir" not in calls[0][1]


@pytest.mark.parametrize("spelling", ["dot", "absolute"])
def test_per_target_runner_normalizes_equivalent_vq_targets(
    spelling,
    monkeypatch,
):
    repo = Path(__file__).resolve().parents[1]
    relative = "vibe-queue/tests/test_paths.py"
    target = (
        f"./{relative}::TestStateRoot::test_env_override_takes_precedence"
        if spelling == "dot"
        else f"{repo / relative}::TestStateRoot::test_env_override_takes_precedence"
    )
    monkeypatch.setenv("HOME", "/caller/live-home")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--noconftest")

    with runner._isolated_test_environment(repo, target) as env:
        assert env["HOME"] != "/caller/live-home"
        assert "PYTEST_ADDOPTS" not in env
        assert Path(env["VQ_STATE_DIR"]).is_relative_to(
            Path(env["VQ_TEST_SANDBOX_ROOT"])
        )


def test_per_target_runner_rejects_targets_outside_worktree():
    repo = Path(__file__).resolve().parents[1]
    with (
        pytest.raises(ValueError, match="outside worktree"),
        runner._isolated_test_environment(
            repo,
            "/tmp/outside-vq-test.py::test_escape",
        ),
    ):
        pass


def test_lane_manifest_has_required_lanes():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    lanes = manifest["lanes"]
    for name in (
        "smoke",
        "release-core-cheap",
        "molecular-scf-dft",
        "semiempirical-molecular",
        "semiempirical-periodic",
        "semiempirical-aiccm",
        "semiempirical",
        "pbc-core",
        "pbc-gdf",
        "pbc-bipole",
        "pbc-gapw-gpw",
        "aiccm-molecular",
        "aiccm-periodic",
        "aiccm-experimental",
        "full-fast",
        "slow-nightly",
    ):
        assert name in lanes


def test_basis_integrals_lane_allows_the_measured_basis_filter_node():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    lane = manifest["lanes"]["basis-ecp-integrals"]

    # test_lih_filter_unblocks_scf_run measured 301-306 seconds on two
    # independent lane executions.  The old 300-second ceiling therefore
    # made the lane deterministically red before its 900-second file budget.
    assert lane["test_timeout_s"] >= 600
    assert lane["file_timeout_s"] >= 2400


def test_lane_manifest_has_release_core_profile():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    profiles = manifest["profiles"]

    assert "release-core" in profiles
    assert runner.profile_lane_names(manifest, "release-core", "blocking") == [
        "release-core-cheap",
    ]
    advisory = runner.profile_lane_names(manifest, "release-core", "advisory")
    assert "aiccm-molecular" not in advisory
    assert "aiccm-periodic" not in advisory
    assert "semiempirical-aiccm" not in advisory
    assert runner.profile_lane_names(manifest, "release-core", "all")[:1] == [
        "release-core-cheap",
    ]


def test_lane_manifest_declares_validation_policy_metadata():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    policy = manifest["policy"]

    assert set(policy["method_maturity_states"]) == {
        "production",
        "verified",
        "under-review",
        "experimental",
    }
    assert set(policy["lane_classes"]) == {
        "pre-cut blocking",
        "post-cut evaluation",
        "implementing-chat-only",
    }
    assert policy["scientific_acceptance"]["pytest_or_ci_can_accept_method"] is False


def test_every_lane_has_maturity_lane_class_and_full_calculation_note():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    allowed_maturity = set(manifest["policy"]["method_maturity_states"])
    allowed_classes = set(manifest["policy"]["lane_classes"])

    for name, lane in manifest["lanes"].items():
        assert lane["method_maturity"] in allowed_maturity, name
        assert lane["lane_class"] in allowed_classes, name
        assert lane["scientific_acceptance"] is False, name
        assert lane["target_release"], name
        assert lane["global_items"], name
        assert lane["required_full_calculation"], name


def test_release_profile_lanes_match_pre_cut_blocking_metadata():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    lanes = manifest["lanes"]
    profile = manifest["profiles"]["release-core"]
    blocking = set(profile["blocking_lanes"])

    assert blocking == {
        name
        for name, lane in lanes.items()
        if lane["lane_class"] == "pre-cut blocking"
    }
    assert {lanes[name]["method_maturity"] for name in blocking} == {"verified"}


def test_experimental_lanes_are_implementing_chat_only():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    lanes = manifest["lanes"]
    profile_lanes = set()
    for profile in manifest["profiles"].values():
        profile_lanes.update(profile.get("blocking_lanes") or [])
        profile_lanes.update(profile.get("advisory_lanes") or [])

    for name, lane in lanes.items():
        if lane["method_maturity"] == "experimental":
            assert lane["lane_class"] == "implementing-chat-only", name
            assert lane.get("impact_default") is False, name
            assert lane.get("release_impact_default") is not True, name
            assert name not in profile_lanes


def test_select_lane_files_applies_include_and_exclude(tmp_path):
    wt = tmp_path
    files = [
        wt / "tests" / "test_periodic_rhf.py",
        wt / "tests" / "test_periodic_k_gdf.py",
        wt / "tests" / "test_periodic_gapw_grid.py",
        wt / "tests" / "test_rhf.py",
    ]
    lane = {
        "include": ["tests/test_periodic*.py"],
        "exclude": ["tests/test_periodic_*gdf*.py", "tests/test_periodic_gapw*.py"],
    }

    selected = runner.select_lane_files(files, wt, lane)

    assert [p.name for p in selected] == ["test_periodic_rhf.py"]


def test_select_lane_set_files_dedupes_across_lanes(tmp_path):
    wt = tmp_path
    files = [
        wt / "tests" / "test_alpha.py",
        wt / "tests" / "test_beta.py",
        wt / "tests" / "test_gamma.py",
    ]
    lanes = [
        {"include": ["tests/test_alpha.py", "tests/test_beta.py"]},
        {"include": ["tests/test_beta.py", "tests/test_gamma.py"]},
    ]

    selected = runner.select_lane_set_files(files, wt, lanes)

    assert [p.name for p in selected] == [
        "test_alpha.py",
        "test_beta.py",
        "test_gamma.py",
    ]


def test_select_lane_set_targets_keeps_node_ids_and_skips_redundant_nodes(tmp_path):
    wt = tmp_path
    files = [
        wt / "tests" / "test_alpha.py",
        wt / "tests" / "test_beta.py",
    ]
    lanes = [
        {
            "include": ["tests/test_alpha.py"],
            "nodes": [
                "tests/test_alpha.py::test_fast",
                "tests/test_beta.py::test_fast",
            ],
        },
        {"include": [], "nodes": ["tests/test_beta.py::test_fast"]},
    ]

    selected = runner.select_lane_set_targets(files, wt, lanes)

    assert selected == [
        "tests/test_alpha.py",
        "tests/test_beta.py::test_fast",
    ]


def test_discover_test_files_includes_basisset_dev(tmp_path):
    tests_dir = tmp_path / "tests"
    basisset_dir = tests_dir / "basisset_dev"
    tests_dir.mkdir()
    basisset_dir.mkdir()
    top_level = tests_dir / "test_alpha.py"
    basisset = basisset_dir / "test_beta.py"
    ignored = tests_dir / "helper.py"
    top_level.write_text("", encoding="utf-8")
    basisset.write_text("", encoding="utf-8")
    ignored.write_text("", encoding="utf-8")

    selected = runner.discover_test_files(tmp_path)

    assert selected == [top_level, basisset]


def test_suite_manifest_classifies_every_discovered_file():
    root = RUNNER_PATH.parents[2]
    manifest = runner.load_suite_manifest(SUITE_MANIFEST_PATH)
    classified = {row["file"] for row in manifest["tests"]}
    discovered = {
        path.relative_to(root).as_posix()
        for path in runner.discover_test_files(root)
    }
    assert classified == discovered


def test_experimental_files_are_t3_and_blockers_have_owners():
    manifest = runner.load_suite_manifest(SUITE_MANIFEST_PATH)
    for row in manifest["tests"]:
        if row["maturity"] == "experimental":
            assert row["tier"] == "T3", row["file"]
        if row["tier"] in {"T0", "T1"}:
            assert row["owner"], row["file"]


def test_suite_manifest_matches_a_fresh_generator_run():
    """Regenerating the manifest must not change any classification field.

    ``update_suite_manifest.py`` rebuilds all ~700 rows from scratch, and every
    chat that adds a test file has to run it: ``tests/conftest.py`` fails
    collection on an unclassified file. Before the generator learned to carry
    curated rows forward, that obligatory rerun silently retiered another
    chat's test -- ``tests/test_periodic_molden_gamma_export.py`` went from
    blocking T1 to "demote T2", losing its hand-written rationale, inside a
    diff far too large to eyeball. Because the release cut selects its blocking
    set with ``--tier T1`` off this manifest, that clobber quietly narrowed the
    release gate.

    Pinning regeneration as a no-op turns any such rewrite into a failure here.
    The measurement fields and ``generated_at_sha`` are excluded: those are
    expected to move as the gate is rerun and as commits land.
    """
    fields = ("maturity", "tier", "owner", "disposition", "rationale")
    lanes, blocking_files = generator.load_lanes()
    paths = generator.discover_paths()
    expected = [
        dict(zip(fields, generator.classify(path, lanes, blocking_files)), file=path)
        for path in paths
    ]

    manifest = runner.load_suite_manifest(SUITE_MANIFEST_PATH)
    actual = [
        {key: row[key] for key in (*fields, "file")} for row in manifest["tests"]
    ]

    assert actual == expected, (
        "scripts/test_gate/suite_manifest.json disagrees with a fresh "
        "`python scripts/test_gate/update_suite_manifest.py` run. If a row is "
        "deliberately hand-classified, record it in that script's CURATED "
        "table so regeneration preserves it -- do not make this green by "
        "committing a regenerated manifest, which discards the "
        "classification. The generator refuses to write a weakened row, so "
        "rerunning it is safe."
    )
    assert manifest["test_file_count"] == len(expected)


def test_generator_refuses_to_silently_weaken_a_committed_row():
    """Regeneration must not be able to demote an unregistered hand-classification.

    ``CURATED`` protects only rows somebody remembered to register, and the
    guard above cannot distinguish the two ways to make it green: registering
    the row, or regenerating over it.  ``detect_demotions`` compares against the
    committed manifest so the destructive resolution is refused rather than
    silently rewarded -- the mechanism that lost
    ``tests/test_periodic_molden_gamma_export.py`` its blocking-T1 row.
    """
    prior = {
        "tests/test_a.py": {
            "tier": "T1",
            "maturity": "verified",
            "rationale": "C++ far-field contractor energy and Fock parity",
        },
        "tests/test_b.py": {
            "tier": "T2",
            "maturity": "verified",
            "rationale": generator.DEFAULT_RATIONALE["T2"].format(maturity="verified"),
        },
    }
    rows = [
        {
            "file": "tests/test_a.py",
            "tier": "T2",
            "maturity": "under-review",
            "rationale": generator.DEFAULT_RATIONALE["T2"].format(
                maturity="under-review"
            ),
        },
        {
            "file": "tests/test_b.py",
            "tier": "T2",
            "maturity": "verified",
            "rationale": generator.DEFAULT_RATIONALE["T2"].format(maturity="verified"),
        },
    ]

    findings = generator.detect_demotions(rows, prior)

    assert len(findings) == 1
    finding = findings[0]
    assert "tests/test_a.py" in finding
    assert "tier T1 -> T2" in finding
    assert "drops out of the blocking set" in finding
    assert "maturity verified -> under-review" in finding
    assert "hand-written rationale replaced by tier boilerplate" in finding

    assert generator.detect_demotions(rows, prior, allowed={"tests/test_a.py"}) == []


def test_generator_allows_promotions_and_new_files():
    """Only weakening is refused; strengthening and new rows flow through."""
    prior = {
        "tests/test_a.py": {
            "tier": "T2",
            "maturity": "under-review",
            "rationale": generator.DEFAULT_RATIONALE["T2"].format(
                maturity="under-review"
            ),
        }
    }
    rows = [
        {
            "file": "tests/test_a.py",
            "tier": "T1",
            "maturity": "verified",
            "rationale": generator.DEFAULT_RATIONALE["T1"],
        },
        {
            "file": "tests/test_new.py",
            "tier": "T2",
            "maturity": "under-review",
            "rationale": generator.DEFAULT_RATIONALE["T2"].format(
                maturity="under-review"
            ),
        },
    ]

    assert generator.detect_demotions(rows, prior) == []


def test_committed_manifest_survives_a_fresh_regeneration_unweakened():
    """The real tree must have no pending demotion waiting for a regeneration."""
    fields = ("maturity", "tier", "owner", "disposition", "rationale")
    lanes, blocking_files = generator.load_lanes()
    rows = [
        dict(zip(fields, generator.classify(path, lanes, blocking_files)), file=path)
        for path in generator.discover_paths()
    ]
    prior = {
        row["file"]: row
        for row in runner.load_suite_manifest(SUITE_MANIFEST_PATH)["tests"]
    }

    assert generator.detect_demotions(rows, prior) == []


def test_curated_rows_are_live_and_policy_clean():
    """The CURATED table must not outlive the files it classifies."""
    lanes, blocking_files = generator.load_lanes()
    paths = generator.discover_paths()

    generator.validate(paths, lanes, blocking_files)

    assert set(generator.CURATED) <= set(paths)


def test_tier_selection_is_exhaustive_and_disjoint():
    manifest = runner.load_suite_manifest(SUITE_MANIFEST_PATH)
    selected = []
    for tier in ("T0", "T1", "T2", "T3"):
        selected.extend(runner.select_tier_targets(manifest, [tier]))
    assert len(selected) == len(set(selected)) == manifest["test_file_count"]


@pytest.mark.parametrize('source', [
    'python/vibeqc/periodic_mdf.py',
    'python/vibeqc/periodic_gdf_gradient.py',
    'python/vibeqc/periodic_gdf_properties.py',
    'python/vibeqc/periodic_k_gdf.py',
    'python/vibeqc/periodic_rohf_gdf.py',
    'cpp/src/periodic_gdf_short_range.cpp',
    'cpp/src/periodic_gdf_short_range_bindings.cpp',
])
def test_private_mdf_source_and_gradient_changes_trigger_gdf_lane(source):
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    assert 'pbc-gdf' in runner.affected_lane_names(manifest, [source])


def test_manifest_routes_gdf_and_aiccm_files_to_area_lanes():
    root = RUNNER_PATH.parents[2]
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    files = runner.discover_test_files(root)

    def lane_files(name: str) -> set[str]:
        return {
            p.relative_to(root).as_posix()
            for p in runner.select_lane_files(files, root, manifest["lanes"][name])
        }

    pbc_gdf = lane_files("pbc-gdf")
    aiccm_molecular = lane_files("aiccm-molecular")
    aiccm_periodic = lane_files("aiccm-periodic")
    aiccm_aggregate = lane_files("aiccm-experimental")
    smoke = lane_files("smoke")
    pbc_core = lane_files("pbc-core")

    assert "tests/test_runner_gamma_gdf_routing.py" in pbc_gdf
    assert "tests/test_periodic_mdf_source.py" in pbc_gdf
    assert "tests/test_aiccm2026_testset.py" in aiccm_molecular
    assert "tests/test_aiccm2026_testset.py" in aiccm_aggregate
    assert "tests/test_periodic_aiccm2026dev_b.py" in aiccm_periodic
    assert "tests/test_periodic_aiccm2026dev_b.py" in aiccm_aggregate
    assert "tests/test_aiccm2026_testset.py" not in smoke
    assert "tests/test_aiccm2026_testset.py" not in pbc_core
    assert "tests/test_periodic_aiccm2026dev_b.py" not in pbc_core


def test_aiccm_lanes_are_manual_only_not_impact_or_release_profile():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    lanes = manifest["lanes"]
    profile = manifest["profiles"]["release-core"]
    release_profile_lanes = set(profile["blocking_lanes"] + profile["advisory_lanes"])

    for lane_name in (
        "aiccm-molecular",
        "aiccm-periodic",
        "aiccm-experimental",
        "semiempirical-aiccm",
    ):
        assert lanes[lane_name].get("impact_default") is False
        assert lane_name not in release_profile_lanes


def test_manifest_splits_semiempirical_molecular_periodic_and_aiccm():
    root = RUNNER_PATH.parents[2]
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    files = runner.discover_test_files(root)

    def lane_files(name: str) -> set[str]:
        return {
            p.relative_to(root).as_posix()
            for p in runner.select_lane_files(files, root, manifest["lanes"][name])
        }

    molecular = lane_files("semiempirical-molecular")
    periodic = lane_files("semiempirical-periodic")
    aiccm = lane_files("semiempirical-aiccm")
    aggregate = lane_files("semiempirical")
    pbc_core = lane_files("pbc-core")

    assert "tests/test_msindo.py" in molecular
    assert "tests/test_neb_semiempirical.py" in molecular
    assert "tests/test_semiempirical_periodic_route_adapter.py" in periodic
    assert "tests/test_periodic_runner_semiempirical.py" in periodic
    assert "tests/test_msindo_ccm.py" in aiccm
    assert "tests/test_dftb0_seccm.py" in aiccm
    assert "tests/test_ccm_semiempirical.py" in aiccm
    assert "tests/test_seccm_topology.py" in aiccm

    assert "tests/test_semiempirical_periodic_route_adapter.py" not in molecular
    assert "tests/test_msindo.py" not in periodic
    assert "tests/test_msindo_ccm.py" not in molecular
    assert "tests/test_periodic_runner_semiempirical.py" not in pbc_core

    for expected in (
        "tests/test_msindo.py",
        "tests/test_neb_semiempirical.py",
        "tests/test_semiempirical_periodic_route_adapter.py",
        "tests/test_periodic_runner_semiempirical.py",
    ):
        assert expected in aggregate
    assert "tests/test_msindo_ccm.py" not in aggregate
    assert "tests/test_dftb0_seccm.py" not in aggregate
    assert "tests/test_seccm_topology.py" not in aggregate


def test_shipped_and_release_lanes_exclude_experimental_tests():
    root = RUNNER_PATH.parents[2]
    lane_manifest = runner.load_lane_manifest(MANIFEST_PATH)
    suite_manifest = runner.load_suite_manifest(SUITE_MANIFEST_PATH)
    files = runner.discover_test_files(root)

    def lane_files(name: str) -> set[str]:
        return {
            p.relative_to(root).as_posix()
            for p in runner.select_lane_files(
                files,
                root,
                lane_manifest["lanes"][name],
            )
        }

    experimental = {
        row["file"]
        for row in suite_manifest["tests"]
        if row["maturity"] == "experimental"
    }
    shipped_lane_names = {"full-fast", "slow-nightly"}
    for profile in lane_manifest.get("profiles", {}).values():
        shipped_lane_names.update(profile.get("blocking_lanes") or [])
        shipped_lane_names.update(profile.get("advisory_lanes") or [])

    leaks = {}
    for lane_name in sorted(shipped_lane_names):
        leaked = sorted(lane_files(lane_name) & experimental)
        if leaked:
            leaks[lane_name] = leaked
    assert leaks == {}, f"shipped/release lanes select experimental tests: {leaks}"


def test_release_core_cheap_lane_is_blocking_and_non_experimental():
    root = RUNNER_PATH.parents[2]
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    files = runner.discover_test_files(root)
    release_core = set(
        runner.select_lane_set_targets(
            files,
            root,
            [manifest["lanes"]["release-core-cheap"]],
        )
    )

    assert release_core == {"tests/test_release_sentinels.py"}
    assert "tests/test_aiccm2026_testset.py" not in release_core
    assert "tests/test_periodic_aiccm2026dev_b.py" not in release_core
    assert "tests/test_periodic_gapw_grid.py" not in release_core
    assert "tests/test_msindo_ccm.py" not in release_core


def test_affected_lanes_use_triggers_and_skip_aggregates():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)

    dispersion = runner.affected_lane_names(
        manifest,
        ["python/vibeqc/dispersion.py", "tests/test_dispersion.py"],
    )
    semi_periodic = runner.affected_lane_names(
        manifest,
        ["tests/test_semiempirical_periodic_route_adapter.py"],
    )
    aiccm_periodic = runner.affected_lane_names(
        manifest,
        ["tests/test_periodic_aiccm2026dev_b.py"],
    )
    gate = runner.affected_lane_names(
        manifest,
        ["scripts/test_gate/run_full_suite.py", "tests/test_test_gate_lanes.py"],
    )

    assert "molecular-scf-dft" in dispersion
    assert "semiempirical-periodic" in semi_periodic
    assert "semiempirical" not in semi_periodic
    assert "aiccm-periodic" not in aiccm_periodic
    assert "aiccm-experimental" not in aiccm_periodic
    assert "smoke" in gate
    assert "full-fast" not in gate


def test_direct_incremental_fock_changes_run_their_regressions():
    """The DirectJKBuilder trigger and its fast regression stay connected."""
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    root = RUNNER_PATH.parents[2]
    lane = manifest["lanes"]["molecular-scf-dft"]
    files = [root / "tests" / "test_incremental_fock_open_shell.py"]

    for source in (
        "cpp/src/jk_builder.cpp",
        "cpp/src/jk_direct.cpp",
        "cpp/include/vibeqc/jk_builder.hpp",
        "cpp/include/vibeqc/jk_direct.hpp",
    ):
        assert "molecular-scf-dft" in runner.affected_lane_names(
            manifest, [source]
        ), source
    assert runner.select_lane_files(files, root, lane) == files


def test_skala_shared_paths_trigger_changed_since_periodic_xc_lanes():
    """Changed-since inference covers executing and fail-closed XC routes."""
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    expected = {"pbc-core", "pbc-gdf", "pbc-bipole", "pbc-gapw-gpw"}

    for path in (
        "python/vibeqc/skala.py",
        "python/vibeqc/periodic_runner.py",
        "cpp/include/vibeqc/grid.hpp",
        "cpp/include/vibeqc/xc.hpp",
        "cpp/src/bindings.cpp",
        "cpp/src/grid.cpp",
        "cpp/src/xc.cpp",
    ):
        affected = set(runner.affected_lane_names(manifest, [path]))
        assert expected <= affected, (path, sorted(affected))


def test_external_xc_host_contract_runs_in_molecular_and_pbc_core_lanes():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)
    root = RUNNER_PATH.parents[2]
    files = [root / "tests" / "test_xc_external_provider.py"]

    for lane_name in ("molecular-scf-dft", "pbc-core"):
        selected = runner.select_lane_files(
            files,
            root,
            manifest["lanes"][lane_name],
        )
        assert selected == files, lane_name


def test_release_impact_mode_skips_calculation_lanes():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)

    dev = runner.affected_lane_names(
        manifest,
        ["python/vibeqc/periodic_runner.py"],
    )
    release = runner.affected_lane_names(
        manifest,
        ["python/vibeqc/periodic_runner.py"],
        impact_mode="release",
    )
    release_output = runner.affected_lane_names(
        manifest,
        ["python/vibeqc/output/writer.py"],
        impact_mode="release",
    )

    assert "pbc-core" in dev
    assert "semiempirical-periodic" in dev
    assert "pbc-core" not in release
    assert "semiempirical-periodic" not in release
    assert release == []
    assert release_output == []


def test_affected_lanes_use_node_file_targets():
    manifest = {
        "lanes": {
            "node-lane": {
                "nodes": ["tests/test_alpha.py::test_fast"],
            },
        },
    }

    affected = runner.affected_lane_names(manifest, ["tests/test_alpha.py"])

    assert affected == ["node-lane"]


def test_release_impact_mode_requires_explicit_lane_opt_in():
    manifest = {
        "lanes": {
            "node-lane": {
                "nodes": ["tests/test_alpha.py::test_fast"],
            },
            "release-node-lane": {
                "release_impact_default": True,
                "nodes": ["tests/test_beta.py::test_fast"],
            },
        },
    }

    affected = runner.affected_lane_names(
        manifest,
        ["tests/test_alpha.py", "tests/test_beta.py"],
        impact_mode="release",
    )

    assert affected == ["release-node-lane"]


def test_combined_lane_defaults_are_conservative():
    lanes = [
        {
            "markexpr": "not slow",
            "jobs": 6,
            "file_timeout_s": 600,
            "test_timeout_s": 300,
        },
        {
            "markexpr": "not slow",
            "jobs": 2,
            "file_timeout_s": 1800,
            "test_timeout_s": 900,
            "heavy_env": True,
        },
    ]

    merged = runner.combined_lane_defaults(lanes)

    assert merged["markexpr"] == "not slow"
    assert merged["jobs"] == 2
    assert merged["file_timeout_s"] == 1800
    assert merged["test_timeout_s"] == 900
    assert merged["heavy_env"] is True


def test_combined_lane_defaults_reject_conflicting_markexprs():
    lanes = [{"markexpr": "not slow"}, {"markexpr": "slow"}]

    try:
        runner.combined_lane_defaults(lanes)
    except ValueError as exc:
        assert "different markexpr" in str(exc)
    else:
        raise AssertionError("expected markexpr conflict")


def test_profile_lane_names_reject_unknown_profile():
    manifest = runner.load_lane_manifest(MANIFEST_PATH)

    try:
        runner.profile_lane_names(manifest, "does-not-exist", "blocking")
    except ValueError as exc:
        assert "unknown profile" in str(exc)
    else:
        raise AssertionError("expected unknown profile error")


def test_print_selected_files_reports_relative_paths(tmp_path, capsys):
    test_file = tmp_path / "tests" / "test_alpha.py"
    test_file.parent.mkdir()
    test_file.write_text("", encoding="utf-8")

    runner.print_selected_files([test_file], tmp_path, "smoke")

    out = capsys.readouterr().out
    assert "[triage] lane=smoke dry-run selected 1 targets" in out
    assert "tests/test_alpha.py" in out


def test_print_lanes_reports_maturity_and_lane_class(capsys):
    manifest = runner.load_lane_manifest(MANIFEST_PATH)

    runner.print_lanes(manifest)

    out = capsys.readouterr().out
    assert "smoke" in out
    assert "verified" in out
    assert "pre-cut blocking" in out
    assert "aiccm-experimental" in out
    assert "experimental" in out
    assert "implementing-chat-only" in out


def test_lane_defaults_do_not_override_explicit_cli_values():
    args = Namespace(
        markexpr="custom",
        jobs=9,
        file_timeout=99.0,
        test_timeout=88,
        heavy_env=False,
    )
    lane = {
        "markexpr": "not slow",
        "jobs": 2,
        "file_timeout_s": 300,
        "test_timeout_s": 120,
        "heavy_env": True,
    }

    resolved = runner.apply_lane_defaults(args, lane)

    assert resolved.markexpr == "custom"
    assert resolved.jobs == 9
    assert resolved.file_timeout == 99.0
    assert resolved.test_timeout == 88
    assert resolved.heavy_env is True


def test_lane_defaults_fill_unset_values():
    args = Namespace(
        markexpr="",
        jobs=None,
        file_timeout=None,
        test_timeout=None,
        heavy_env=False,
    )
    lane = {
        "markexpr": "slow",
        "jobs": 1,
        "file_timeout_s": 2400,
        "test_timeout_s": 1800,
    }

    resolved = runner.apply_lane_defaults(args, lane)

    assert resolved.markexpr == "slow"
    assert resolved.jobs == 1
    assert resolved.file_timeout == 2400.0
    assert resolved.test_timeout == 1800


def test_periodic_ccm_test_files_carry_experimental_pytestmark():
    """Every test file exercising vibeqc.periodic.ccm (the Γ-CCM -a line) must
    carry the module-level ``experimental`` pytest marker, so marker-based
    selection stays in sync with the manual-research lane policy. The -b
    stream's files are excluded: they import -a helpers only inside slow
    head-to-head tests and are labeled by their own warning contract."""
    import re

    tests_dir = Path(__file__).resolve().parent
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if "aiccm2026dev_b" in path.name:
            continue
        src = path.read_text(encoding="utf-8")
        if not re.search(r"^\s*(from|import)\s+vibeqc\.periodic\.ccm", src, re.M):
            continue
        if not re.search(r"^pytestmark\s*=.*experimental", src, re.M):
            offenders.append(path.name)
    assert not offenders, (
        "Γ-CCM test files missing 'pytestmark = pytest.mark.experimental': "
        f"{offenders}"
    )


def test_chi_and_ccm_lines_import_nothing_from_each_other():
    """D56/D74 firewall as a gate (M1 of
    handovers/HANDOVER_AICCM_STANDARD_METHOD.md): no module under
    python/vibeqc/periodic/chi imports vibeqc.periodic.ccm and no module under
    periodic/ccm imports vibeqc.periodic.chi, at any relative level. The
    front door (``method="aiccm"``) is a layer above both lines and lives in
    periodic_runner / periodic_jk_method; a convenience import between the
    packages would merge the constructions the plan keeps separate. Pure
    filesystem and ``ast``; imports nothing from vibeqc."""
    import ast

    periodic = Path(__file__).resolve().parents[1] / "python" / "vibeqc" / "periodic"

    def imported_modules(path, pkg):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        own = ["vibeqc", "periodic", pkg]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield node.lineno, alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    base = node.module or ""
                else:
                    # level 1 = this package, 2 = vibeqc.periodic, 3 = vibeqc
                    parent = own[: len(own) - (node.level - 1)]
                    base = ".".join(parent + ([node.module] if node.module else []))
                yield node.lineno, base
                for alias in node.names:
                    yield node.lineno, f"{base}.{alias.name}"

    offenders = []
    for pkg, other in (("chi", "ccm"), ("ccm", "chi")):
        pkg_dir = periodic / pkg
        assert pkg_dir.is_dir(), pkg_dir
        forbidden = f"vibeqc.periodic.{other}"
        for path in sorted(pkg_dir.rglob("*.py")):
            for lineno, name in imported_modules(path, pkg):
                if name == forbidden or name.startswith(forbidden + "."):
                    offenders.append(f"{path.relative_to(periodic.parents[1])}:{lineno}: {name}")
    assert not offenders, (
        "periodic/chi and periodic/ccm must stay import-independent "
        "(D56/D74 in docs/aiccm2026dev_b_decisions.md):\n" + "\n".join(offenders)
    )


def test_experimental_marker_registered_in_pyproject():
    """The ``experimental`` marker must stay registered so the module-level
    pytestmark above is not an unknown-marker warning."""
    import re

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert re.search(r'^\s*"experimental:', text, re.M), (
        "pyproject.toml [tool.pytest.ini_options] markers must register "
        "'experimental'"
    )
