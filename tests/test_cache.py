from multiagent.cache import (
    get_novelty_entry,
    get_relation_entry,
    novelty_entry_covers,
    project_cache_to_abc,
    put_novelty_decision,
    put_relation_verification,
    relation_entry_covers,
)


def test_relation_cache_tracks_temporal_pair_and_depth():
    cache = put_relation_verification(
        {},
        {
            "entity_a": "Migraine",
            "entity_b": "Vascular tone",
            "before_year": 1985,
            "relation_supported": True,
            "relation_type": "associated_with",
            "confidence": 0.8,
            "supporting_pmids": ["1", "2"],
            "evidence": [{"pmid": "1"}, {"pmid": "2"}],
        },
        top_k=8,
        iteration=1,
    )

    entry = get_relation_entry(cache, "migraine", "vascular tone", 1985)

    assert entry is not None
    assert relation_entry_covers(entry, 8)
    assert not relation_entry_covers(entry, 12)
    assert entry["supporting_pmids"] == ["1", "2"]


def test_novelty_cache_is_separate_from_relation_cache():
    cache = put_novelty_decision(
        {},
        {
            "entity_a": "Migraine",
            "entity_c": "Magnesium",
            "before_year": 1985,
            "direct_relation_known": False,
            "novel": True,
            "confidence": 0.9,
            "known_relation_pmids": [],
            "evidence": [],
        },
        top_k=8,
        iteration=2,
    )

    novelty = get_novelty_entry(cache, "Migraine", "Magnesium", 1985)
    relation = get_relation_entry(cache, "Migraine", "Magnesium", 1985)

    assert novelty is not None
    assert novelty_entry_covers(novelty, 8)
    assert relation is None


def test_cache_projection_keeps_only_current_abc_entries():
    cache = {}
    cache = put_relation_verification(
        cache,
        {
            "entity_a": "Migraine",
            "entity_b": "Vascular tone",
            "before_year": 1985,
            "relation_supported": True,
            "evidence": [],
        },
        top_k=8,
        iteration=1,
    )
    cache = put_relation_verification(
        cache,
        {
            "entity_a": "Unrelated A",
            "entity_b": "Unrelated B",
            "before_year": 1985,
            "relation_supported": True,
            "evidence": [],
        },
        top_k=8,
        iteration=1,
    )

    projected = project_cache_to_abc(
        cache,
        "Migraine",
        "Vascular tone",
        "Magnesium",
        1985,
    )

    assert len(projected) == 1
    assert next(iter(projected.values()))["entity_a"] == "Migraine"
