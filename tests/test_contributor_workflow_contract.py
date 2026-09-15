"""Core contributor policy retained when loop tests moved to their owner."""

from pathlib import Path

CONTRIBUTOR_GUIDE = Path(__file__).resolve().parents[1] / "CONTRIBUTING.md"


def test_automated_fixer_posts_its_brief_before_landing() -> None:
    """Human post-landing comments must not reverse the automated lease fence."""
    normalized = " ".join(CONTRIBUTOR_GUIDE.read_text(encoding="utf-8").split())
    assert "An automated fixer instead posts the same brief" in normalized
    assert "brief after its push succeeds and before `bugctl land`" in normalized
    assert (
        "because `land` closes the lease that authorizes `bugctl progress`"
        in normalized
    )
