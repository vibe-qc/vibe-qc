"""Molecular input and persisted-cache migration without running chemistry."""
import pytest
from vibe_basis.backends.crystal_atom import emit_input_atom
from vibe_basis.cache import EnergyCache, atom_key
from vibe_basis.engine import EngineEnergy
from vibe_basis.engines.crystal23 import Crystal23Engine
from vibe_basis.pipeline import CohesivePipeline


@pytest.mark.parametrize("z,unpaired", [(12, 0), (8, 2)])
def test_molecular_geometry_and_preserved_spin(z, unpaired):
    deck = emit_input_atom(z, "99 0", n_unpaired=unpaired)
    lines = deck.splitlines()
    assert lines[1:5] == ["MOLECULE", "1", "1", f"{z} 0.0 0.0 0.0"]
    assert "CRYSTAL" not in lines and "SHRINK" not in lines
    for keyword in ("UHF", "SYMMREMO", "SPINLOCK"):
        assert (keyword in lines) == bool(unpaired)
    if unpaired:
        assert lines[lines.index("SPINLOCK") + 1] == "2 50"
    assert emit_input_atom(z, "99 0", n_unpaired=unpaired, box=20) == deck


def test_resumed_periodic_cache_is_not_a_molecular_result(tmp_path, monkeypatch):
    journal = tmp_path / "energies.jsonl"
    old_key = atom_key("crystal23", 12, "99 0", "rhf")
    old = EngineEnergy(-197.3195141238, True, "crystal23",
                       detail={"atom_reference": "p1_box_closed_shell"})
    EnergyCache(journal).put(old_key, old)
    engine = Crystal23Engine(transport=None)
    calls = []

    def capture(deck, name, **detail):
        calls.append(deck)
        # No synthetic molecular energy: failure proves the legacy converged
        # result was refused and the transport was called with a new deck.
        return EngineEnergy.failed("crystal23", "transport_error", **detail)

    monkeypatch.setattr(engine, "_run_deck", capture)
    cache = EnergyCache(journal)
    result, hit = CohesivePipeline(engine, cache=cache)._atom_energy("99 0", 12, "rhf")
    assert not hit and not result.ok
    assert result.detail["atom_reference"] == "molecule_c1_closed_shell"
    assert calls[0].splitlines()[1] == "MOLECULE"
    assert cache.get(old_key) == old
    again, hit = CohesivePipeline(engine, cache=EnergyCache(journal))._atom_energy(
        "99 0", 12, "rhf")
    assert hit and again == result and len(calls) == 1


def test_unchanged_counterpoise_key_reuses_result(tmp_path, monkeypatch):
    class Host:
        name = "MgO"

    engine = Crystal23Engine(transport=None)
    assert engine.atom_cache_settings(host=Host()) == {}
    cache = EnergyCache(tmp_path / "energies.jsonl")
    old = EngineEnergy(-1.0, True, "crystal23")
    cache.put(atom_key("crystal23", 12, "basis", "rhf", host="MgO"), old)

    def forbidden(*args, **kwargs):
        pytest.fail("Unchanged counterpoise cache must not rerun")

    monkeypatch.setattr(engine, "atom_energy", forbidden)
    result, hit = CohesivePipeline(engine, cache=cache)._atom_energy(
        "basis", 12, "rhf", host=Host())
    assert hit and result == old
