from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.cognition import (
    CognitionResponse,
    CognitionResponseKind,
    SpokenDialogueAct,
    SpokenDialoguePolicy,
    SpokenDialoguePolicyConfig,
    SpokenDialoguePolicyDecision,
    SpokenDialogueTone,
    SpokenResponseStyle,
)


def make_response(
    *,
    request_id: str = "request-1",
    text: str = "Hello sir.",
    kind: CognitionResponseKind = CognitionResponseKind.SPOKEN_REPLY,
) -> CognitionResponse:
    return CognitionResponse(
        request_id=request_id,
        text=text,
        kind=kind,
    )


def test_spoken_dialogue_policy_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        SpokenDialoguePolicyConfig(name=" ").validate()

    with pytest.raises(ValueError):
        SpokenDialoguePolicyConfig(concise_max_chars=0).validate()

    with pytest.raises(ValueError):
        SpokenDialoguePolicyConfig(normal_max_chars=0).validate()

    with pytest.raises(ValueError):
        SpokenDialoguePolicyConfig(detailed_max_chars=0).validate()

    with pytest.raises(ValueError):
        SpokenDialoguePolicyConfig(concise_max_sentences=0).validate()

    with pytest.raises(ValueError):
        SpokenDialoguePolicyConfig(normal_max_sentences=0).validate()

    with pytest.raises(ValueError):
        SpokenDialoguePolicyConfig(fallback_text=" ").validate()

def test_spoken_dialogue_policy_decision_rejects_empty_spoken_text() -> None:
    with pytest.raises(ValidationError):
        SpokenDialoguePolicyDecision(
            original_text="hello",
            spoken_text=" ",
        )


def test_spoken_dialogue_policy_prepares_concise_text() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(
        "First sentence. Second sentence. Third sentence.",
        request_id="request-1",
        style=SpokenResponseStyle.CONCISE,
    )

    assert decision.request_id == "request-1"
    assert decision.spoken_text == "First sentence. Second sentence."
    assert decision.style == SpokenResponseStyle.CONCISE
    assert decision.sentence_count == 2
    assert decision.truncated is False


def test_spoken_dialogue_policy_allows_normal_more_sentences() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(
        "One. Two. Three.",
        style=SpokenResponseStyle.NORMAL,
    )

    assert decision.spoken_text == "One. Two. Three."
    assert decision.sentence_count == 3


def test_spoken_dialogue_policy_allows_detailed_full_text() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(
        "One. Two. Three. Four. Five. Six.",
        style=SpokenResponseStyle.DETAILED,
    )

    assert decision.spoken_text == "One. Two. Three. Four. Five. Six."
    assert decision.sentence_count == 6


def test_spoken_dialogue_policy_truncates_by_style_limit() -> None:
    policy = SpokenDialoguePolicy(
        config=SpokenDialoguePolicyConfig(concise_max_chars=12)
    )
    decision = policy.prepare(
        "This response is too long for voice.",
        style=SpokenResponseStyle.CONCISE,
    )

    assert decision.truncated is True
    assert decision.spoken_text == "This resp..."
    assert len(decision.spoken_text) <= 12


def test_spoken_dialogue_policy_uses_fallback_for_empty_direct_reply() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(" ")

    assert decision.spoken_text == "emptyspokenpolicyoutput."
    assert decision.act == SpokenDialogueAct.DIRECT_REPLY


def test_spoken_dialogue_policy_uses_clarification_fallback() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(
        " ",
        act=SpokenDialogueAct.CLARIFICATION,
    )

    assert decision.spoken_text == "What should I focus on, sir?"


def test_spoken_dialogue_policy_uses_refusal_fallback() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(
        " ",
        act=SpokenDialogueAct.REFUSAL,
    )

    assert decision.spoken_text == "I cannot help with that, sir."


def test_spoken_dialogue_policy_uses_failure_fallback() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(
        " ",
        act=SpokenDialogueAct.FAILURE_FALLBACK,
    )

    assert decision.spoken_text == "spokenpolicygenerationfailed."


def test_spoken_dialogue_policy_removes_basic_markdown() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare(
        """
        # Summary
        - **Presence** is complete.
        - `Cognition` is active.
        """,
        style=SpokenResponseStyle.NORMAL,
    )

    assert "#" not in decision.spoken_text
    assert "*" not in decision.spoken_text
    assert "`" not in decision.spoken_text
    assert "Presence is complete." in decision.spoken_text


def test_spoken_dialogue_policy_adds_terminal_punctuation() -> None:
    policy = SpokenDialoguePolicy()
    decision = policy.prepare("Ready sir")

    assert decision.spoken_text == "Ready sir."


def test_spoken_dialogue_policy_apply_to_response() -> None:
    policy = SpokenDialoguePolicy()
    response = make_response(
        text="First. Second. Third.",
    )

    shaped = policy.apply_to_response(response)

    assert shaped.request_id == response.request_id
    assert shaped.response_id == response.response_id
    assert shaped.text == "First. Second."
    assert shaped.metadata["spoken_policy"] == "spoken_dialogue_policy"
    assert shaped.metadata["spoken_act"] == "direct_reply"


def test_spoken_dialogue_policy_apply_to_clarification_response() -> None:
    policy = SpokenDialoguePolicy()
    response = make_response(
        text="Could you clarify that?",
        kind=CognitionResponseKind.CLARIFICATION,
    )

    shaped = policy.apply_to_response(response)

    assert shaped.text == "Could you clarify that?"
    assert shaped.metadata["spoken_act"] == "clarification"


def test_spoken_dialogue_policy_apply_to_refusal_response() -> None:
    policy = SpokenDialoguePolicy()
    response = make_response(
        text="I cannot help with that.",
        kind=CognitionResponseKind.REFUSAL,
    )

    shaped = policy.apply_to_response(response)

    assert shaped.text == "I cannot help with that."
    assert shaped.metadata["spoken_act"] == "refusal"


def test_spoken_dialogue_policy_reads_style_from_response_metadata() -> None:
    policy = SpokenDialoguePolicy()
    response = make_response(
        text="One. Two. Three.",
    ).model_copy(
        update={
            "metadata": {
                "spoken_style": "normal",
            }
        }
    )

    shaped = policy.apply_to_response(response)

    assert shaped.text == "One. Two. Three."
    assert shaped.metadata["spoken_style"] == "normal"


def test_spoken_dialogue_policy_snapshot_counts() -> None:
    policy = SpokenDialoguePolicy(
        config=SpokenDialoguePolicyConfig(concise_max_chars=12)
    )

    policy.prepare("One.", style=SpokenResponseStyle.CONCISE)
    policy.prepare("Two.", style=SpokenResponseStyle.NORMAL)
    policy.prepare("Three.", style=SpokenResponseStyle.DETAILED)
    policy.prepare("This response is too long.", style=SpokenResponseStyle.CONCISE)
    policy.prepare(" ", act=SpokenDialogueAct.FAILURE_FALLBACK)

    snapshot = policy.snapshot()

    assert snapshot.prepared_count == 5
    assert snapshot.concise_count == 3
    assert snapshot.normal_count == 1
    assert snapshot.detailed_count == 1
    assert snapshot.truncated_count == 2
    assert snapshot.fallback_count == 1
    assert snapshot.last_act == SpokenDialogueAct.FAILURE_FALLBACK


def test_spoken_dialogue_policy_reset_clears_counters() -> None:
    policy = SpokenDialoguePolicy()

    policy.prepare("Ready.")

    policy.reset()
    snapshot = policy.snapshot()

    assert snapshot.prepared_count == 0
    assert snapshot.concise_count == 0
    assert snapshot.normal_count == 0
    assert snapshot.detailed_count == 0
    assert snapshot.truncated_count == 0
    assert snapshot.fallback_count == 0
    assert snapshot.last_request_id is None
    assert snapshot.last_error is None


def test_spoken_dialogue_enum_values_are_stable() -> None:
    assert SpokenDialogueAct.DIRECT_REPLY.value == "direct_reply"
    assert SpokenDialogueAct.CLARIFICATION.value == "clarification"
    assert SpokenDialogueAct.FAILURE_FALLBACK.value == "failure_fallback"
    assert SpokenDialogueTone.DIRECT.value == "direct"
    assert SpokenDialogueTone.SUPPORTIVE.value == "supportive"