from __future__ import annotations

from jarvis.cognitive import (
    BehaviorIntent,
    BehaviorPolicy,
    BehaviorRequest,
    BehaviorRisk,
    BehaviorRuntimeStatus,
    BehaviorStance,
    BehaviorTone,
    PersonalityRuntime,
    default_behavior_policy,
    default_jarvis_personality,
)
from jarvis.cognitive.contracts import utc_now


def test_default_personality_matches_jarvis_presence() -> None:
    profile = default_jarvis_personality()

    assert profile.name == "JARVIS"
    assert "calm" in profile.traits
    assert "protective" in profile.traits
    assert "carefully_challenging" in profile.traits
    assert profile.confirmation_phrase == "Certainly, sir."


def test_default_behavior_policy_is_safe_and_concise() -> None:
    policy = default_behavior_policy()

    assert policy.max_reply_sentences == 3
    assert policy.interrupt_only_when_important is True
    assert policy.ask_when_instruction_incomplete is True
    assert policy.truth_over_comfort is True


def test_personality_confirmation_is_calm_and_concise() -> None:
    runtime = PersonalityRuntime()

    result = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.CONFIRMATION,
            message="Running the checks.",
        )
    )

    assert result.status == BehaviorRuntimeStatus.READY
    assert result.text == "Certainly, sir. Running the checks."
    assert result.directive.tone == BehaviorTone.CALM
    assert result.directive.stance == BehaviorStance.SUPPORTIVE


def test_personality_clarifies_incomplete_instruction() -> None:
    runtime = PersonalityRuntime()

    result = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.CLARIFICATION,
            message="Which project should I use?",
            instruction_complete=False,
        )
    )

    assert result.directive.should_clarify is True
    assert result.directive.tone == BehaviorTone.CLARIFYING
    assert result.text.startswith("I need one detail before proceeding.")


def test_personality_warns_for_high_risk() -> None:
    runtime = PersonalityRuntime()

    result = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.WARNING,
            message="Deleting that folder may remove project files.",
            risk=BehaviorRisk.HIGH,
        )
    )

    assert result.directive.should_warn is True
    assert result.directive.stance == BehaviorStance.PROTECTIVE
    assert result.text.startswith("I would advise caution.")


def test_personality_challenges_carefully_when_truth_required() -> None:
    runtime = PersonalityRuntime()

    result = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.CHALLENGE,
            message="That shortcut would bypass the safety boundary.",
            requires_truth_challenge=True,
        )
    )

    assert result.directive.should_challenge is True
    assert result.directive.tone == BehaviorTone.PROTECTIVE
    assert result.text.startswith("I would challenge that carefully, sir.")


def test_personality_stays_silent_when_user_busy_and_low_risk() -> None:
    runtime = PersonalityRuntime()

    result = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.STATUS,
            message="Background scan finished.",
            risk=BehaviorRisk.LOW,
            user_is_busy=True,
        )
    )

    assert result.should_speak is False
    assert result.text == ""
    assert result.directive.stance == BehaviorStance.SILENT


def test_personality_does_not_stay_silent_for_critical_warning() -> None:
    runtime = PersonalityRuntime()

    result = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.WARNING,
            message="Battery is critically low.",
            risk=BehaviorRisk.CRITICAL,
            user_is_busy=True,
        )
    )

    assert result.should_speak is True
    assert result.directive.should_warn is True
    assert "Battery is critically low." in result.text


def test_personality_interruption_response_is_short() -> None:
    runtime = PersonalityRuntime()

    result = runtime.respond(
        BehaviorRequest(intent=BehaviorIntent.INTERRUPTION)
    )

    assert result.text == "Stopping. Listening now."


def test_personality_allows_dry_humor_only_when_safe() -> None:
    runtime = PersonalityRuntime()

    safe = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.HUMOR,
            message="The build failed again.",
            allow_humor=True,
            risk=BehaviorRisk.LOW,
        )
    )
    risky = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.HUMOR,
            message="System failure detected.",
            allow_humor=True,
            risk=BehaviorRisk.HIGH,
        )
    )

    assert "tests are honest" in safe.text
    assert "tests are honest" not in risky.text


def test_personality_bounds_sentence_count() -> None:
    runtime = PersonalityRuntime(
        policy=BehaviorPolicy(
            max_reply_sentences=1,
            interrupt_only_when_important=True,
            ask_when_instruction_incomplete=True,
            allow_dry_humor=True,
            truth_over_comfort=True,
            created_at=utc_now(),
        )
    )

    result = runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.STATUS,
            message="One. Two. Three.",
        )
    )

    assert result.text == "One."


def test_personality_updates_profile_and_policy() -> None:
    runtime = PersonalityRuntime()
    profile = default_jarvis_personality()
    policy = default_behavior_policy()

    profile_result = runtime.update_profile(profile)
    policy_result = runtime.update_policy(policy)

    assert profile_result.status == BehaviorRuntimeStatus.READY
    assert policy_result.status == BehaviorRuntimeStatus.READY


def test_personality_snapshot_tracks_counts() -> None:
    runtime = PersonalityRuntime()

    runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.WARNING,
            message="Unsafe action.",
            risk=BehaviorRisk.HIGH,
        )
    )
    runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.CLARIFICATION,
            instruction_complete=False,
            message="Which file?",
        )
    )
    runtime.respond(
        BehaviorRequest(
            intent=BehaviorIntent.CHALLENGE,
            requires_truth_challenge=True,
            message="That is risky.",
        )
    )

    snapshot = runtime.snapshot()

    assert snapshot.status == BehaviorRuntimeStatus.READY
    assert snapshot.decision_count == 3
    assert snapshot.warning_count == 1
    assert snapshot.clarification_count == 1
    assert snapshot.challenge_count == 1


def test_personality_enum_values_are_stable() -> None:
    assert BehaviorIntent.CONFIRMATION.value == "confirmation"
    assert BehaviorRisk.CRITICAL.value == "critical"
    assert BehaviorRuntimeStatus.READY.value == "ready"