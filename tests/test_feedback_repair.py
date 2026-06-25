from copy import deepcopy

from ktem.utils.feedback_repair import (
    append_feedback_event,
    apply_feedback_repair_settings,
    build_feedback_repair_prompt,
)


BASE_SETTINGS = {
    "index.options.files.num_retrieval": 15,
    "index.options.files.candidate_multiplier": 20,
    "index.options.files.context_expansion_mode": "none",
    "index.options.files.sibling_window": 1,
    "index.options.files.enable_query_expansion": False,
    "index.options.files.mmr": False,
    "index.options.files.use_reranking": False,
    "reasoning.options.simple.system_prompt": "Base system",
    "reasoning.options.simple.qa_prompt": "Base QA {context} {question} {lang}",
    "reasoning.max_context_length": 32000,
}


def test_incomplete_feedback_applies_broader_retrieval_preset():
    repaired = apply_feedback_repair_settings(BASE_SETTINGS, "incomplete")

    assert repaired["index.options.files.num_retrieval"] > 15
    assert repaired["index.options.files.candidate_multiplier"] > 20
    assert repaired["index.options.files.context_expansion_mode"] == "siblings"
    assert repaired["index.options.files.sibling_window"] == 2
    assert repaired["index.options.files.enable_query_expansion"] is True
    assert "previous answer was incomplete" in repaired[
        "reasoning.options.simple.system_prompt"
    ]


def test_hallucination_feedback_is_strict_without_aggressive_top_k_increase():
    repaired = apply_feedback_repair_settings(BASE_SETTINGS, "hallucination")

    assert repaired["index.options.files.num_retrieval"] == 15
    assert repaired["index.options.files.use_reranking"] is True
    assert repaired["index.options.files.mmr"] is False
    assert "unsupported claims" in build_feedback_repair_prompt("hallucination")
    assert "unsupported claims" in repaired["reasoning.options.simple.qa_prompt"]


def test_bad_format_feedback_does_not_change_retrieval_settings():
    repaired = apply_feedback_repair_settings(BASE_SETTINGS, "bad_format")

    for key, value in BASE_SETTINGS.items():
        if key.startswith("index.options."):
            assert repaired[key] == value
    assert "clearer structure" in repaired["reasoning.options.simple.qa_prompt"]


def test_append_feedback_event_is_backward_compatible_with_likes():
    data_source = {"likes": [[0, "old answer", True]]}

    event_id = append_feedback_event(
        data_source,
        message_index=0,
        liked=False,
        reason="incomplete",
        question="What are the rules?",
        old_answer="Old",
        settings_before=deepcopy(BASE_SETTINGS),
    )

    assert data_source["likes"] == [[0, "old answer", True]]
    assert data_source["feedback_events"][0]["event_id"] == event_id
    assert data_source["feedback_events"][0]["reason"] == "incomplete"
    assert data_source["feedback_events"][0]["old_answer"] == "Old"
