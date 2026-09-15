from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'agentic-loop/fleet/prepare-user-daemon-state.py'
spec = importlib.util.spec_from_file_location('prepare_user_daemon_state', SCRIPT)
assert spec and spec.loader
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def record(queue: Path, name: str = 'job1', **values: object) -> Path:
    queue.mkdir(exist_ok=True)
    path = queue / f'{name}.json'
    path.write_text(json.dumps({'id': name, 'state': 'completed', **values}))
    return path


@pytest.mark.parametrize('state', sorted(migration.TERMINAL))
def test_terminal_history_keeps_exact_bytes(tmp_path: Path, state: str) -> None:
    source, target = tmp_path / 'old', tmp_path / 'new'
    path = record(source, state=state, pid=12345)
    target.mkdir()
    expected = migration.inventory(source, allow_pending=False)
    migration.copy_records(source, target, expected)
    assert (target / path.name).read_bytes() == path.read_bytes()
    assert migration.inventory(target, allow_pending=False) == expected


@pytest.mark.parametrize('state', ['running', 'suspended', 'unknown', 'pending'])
def test_unsafe_historical_state_refuses(tmp_path: Path, state: str) -> None:
    record(tmp_path, state=state)
    with pytest.raises(ValueError, match='not quiescent'):
        migration.inventory(tmp_path, allow_pending=False)


def test_only_unlaunched_pending_jobs_can_migrate(tmp_path: Path) -> None:
    record(tmp_path, state='pending', pid=None, scheduler_job_id=None)
    assert len(migration.inventory(tmp_path, allow_pending=True)) == 1
    record(tmp_path, state='pending', scheduler_job_id='123.server')
    with pytest.raises(ValueError, match='not quiescent'):
        migration.inventory(tmp_path, allow_pending=True)


def test_conflicting_ids_require_review_even_if_bytes_match() -> None:
    with pytest.raises(ValueError, match='overlapping'):
        migration.combine({'a.json': 'same'}, {'a.json': 'same'})
    assert migration.combine({'a.json': 'a'}, {'b.json': 'b'}) == {'a.json': 'a', 'b.json': 'b'}


def test_symlink_record_and_false_job_identity_refuse(tmp_path: Path) -> None:
    path = record(tmp_path, id='different')
    with pytest.raises(ValueError, match='identity mismatch'):
        migration.inventory(tmp_path, allow_pending=True)
    path.unlink()
    path.symlink_to(__file__)
    with pytest.raises(ValueError, match='nonregular'):
        migration.inventory(tmp_path, allow_pending=True)


def test_changed_source_and_existing_destination_never_overwrite(tmp_path: Path) -> None:
    source, target = tmp_path / 'old', tmp_path / 'new'
    original = record(source)
    target.mkdir()
    expected = migration.inventory(source, allow_pending=False)
    original.write_text(original.read_text() + '\n')
    with pytest.raises(ValueError, match='changed before copy'):
        migration.copy_records(source, target, expected)
    assert list(target.iterdir()) == []
    expected = migration.inventory(source, allow_pending=False)
    occupied = target / original.name
    occupied.write_text('other evidence')
    with pytest.raises(FileExistsError):
        migration.copy_records(source, target, expected)
    assert occupied.read_text() == 'other evidence'


@pytest.mark.parametrize('change', [
    {'enabled': False}, {'full_dispatch': False}, {'duration_seconds': 60},
    {'expires_at': '2099-01-01'},
])
def test_only_permanent_full_hold_is_accepted(change: dict) -> None:
    drain = {'enabled': True, 'full_dispatch': True, 'duration_seconds': None}
    migration.validate_drain(drain)
    with pytest.raises(ValueError, match='non-expiring full drain'):
        migration.validate_drain({**drain, **change})
