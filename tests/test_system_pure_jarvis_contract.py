from __future__ import annotations

import pytest

from jarvis.system import (
    PCOSCoreComponent,
    PCOSCoreComponentKind,
    PCOSFlowStep,
    PCOSFlowStepKind,
    PCOSLayer,
    PCOSLayerKind,
    PureJarvisAutonomyLevel,
    PureJarvisCapability,
    PureJarvisCapabilityKind,
    PureJarvisRequirementStatus,
    PureJarvisSafetyClass,
    default_pcos_architecture,
    default_pure_jarvis_manifest,
    pure_jarvis_requirement_justifications,
)


def test_default_pure_jarvis_manifest_covers_tony_style_operating_system() -> None:
    manifest = default_pure_jarvis_manifest()

    assert manifest.ready_for_pure_runtime is True
    assert manifest.missing_required_capabilities() == ()
    assert manifest.real_time_capability_count >= 12
    assert manifest.proactive_capability_count >= 10
    assert manifest.doctrine.user_label == "Tony"
    assert manifest.doctrine.assistant_label == "JARVIS"
    assert "calm" in manifest.doctrine.personality


def test_pure_jarvis_doctrine_makes_architecture_and_speed_first_class() -> None:
    manifest = default_pure_jarvis_manifest()
    non_negotiables = " ".join(manifest.doctrine.non_negotiables)

    assert "Architecture and speed are first-class requirements" in non_negotiables
    assert "one owner, one boundary, and one latency role" in non_negotiables
    assert "outside the daily-driver hot path" in non_negotiables


def test_pure_jarvis_speed_capability_protects_hot_path() -> None:
    manifest = default_pure_jarvis_manifest()
    speed = manifest.capability(PureJarvisCapabilityKind.SPEED)

    assert speed.real_time_required is True
    assert speed.phase_owner == "latency + streaming + router"
    assert "streams response chunks" in speed.acceptance
    assert "keeps deep diagnostics and warmups off the live hot path" in (
        speed.acceptance
    )


def test_pure_jarvis_manifest_keeps_voice_inside_cognition_boundary() -> None:
    manifest = default_pure_jarvis_manifest()
    step50 = manifest.capability(PureJarvisCapabilityKind.STEP50_INTEGRATION)

    assert step50.real_time_required is True
    assert any(
        "voice never invents response text" in rule for rule in step50.acceptance
    )
    assert step50.safety_class == PureJarvisSafetyClass.HIGH_IMPACT


def test_pure_jarvis_high_impact_automatic_actions_require_confirmation() -> None:
    with pytest.raises(ValueError, match="high-impact automatic"):
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.SYSTEM_CONTROL,
            purpose="mutate the system",
            autonomy=PureJarvisAutonomyLevel.SAFE_AUTOMATIC,
            safety_class=PureJarvisSafetyClass.SYSTEM_MUTATION,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="actions",
            acceptance=("would mutate state",),
        )


def test_pure_jarvis_communication_is_confirm_then_act() -> None:
    manifest = default_pure_jarvis_manifest()
    communication = manifest.capability(PureJarvisCapabilityKind.COMMUNICATION)

    assert communication.human_confirmation_required is True
    assert communication.autonomy == PureJarvisAutonomyLevel.CONFIRM_THEN_ACT
    assert communication.safety_class == PureJarvisSafetyClass.EXTERNAL_COMMUNICATION


def test_pure_jarvis_requirement_justifications_cover_all_capabilities() -> None:
    manifest = default_pure_jarvis_manifest()
    justifications = pure_jarvis_requirement_justifications()
    justified = {
        capability
        for justification in justifications
        for capability in justification.capabilities
    }

    assert set(manifest.required_capabilities).issubset(justified)
    assert all(justification.evidence for justification in justifications)


def test_pure_jarvis_speed_contract_is_honest_about_milliseconds() -> None:
    justifications = pure_jarvis_requirement_justifications()
    instant = next(
        item
        for item in justifications
        if item.request_line == "respond instantly with short intelligent replies"
    )
    millisecond = next(
        item
        for item in justifications
        if item.request_line == "true milliseconds for complete movie-like answers"
    )
    interruption = next(
        item
        for item in justifications
        if item.request_line == "speak while Tony is working and allow interruption"
    )

    assert instant.status == PureJarvisRequirementStatus.HARDWARE_LIMITED
    assert any("off the blocking launch path" in line for line in instant.evidence)
    assert millisecond.status == PureJarvisRequirementStatus.PHYSICS_LIMITED
    assert millisecond.blocked_or_gated is True
    assert interruption.status == PureJarvisRequirementStatus.SATISFIED
    assert interruption.fully_satisfied is True


def test_pcos_architecture_defines_three_layer_model() -> None:
    architecture = default_pcos_architecture()

    assert tuple(layer.kind for layer in architecture.layers) == (
        PCOSLayerKind.CORE,
        PCOSLayerKind.CAPABILITY,
        PCOSLayerKind.INTELLIGENCE_GROWTH,
    )
    assert architecture.layers[0].allowed_to_replace is False
    assert architecture.layers[1].allowed_to_replace is True
    assert architecture.layers[2].allowed_to_replace is True


def test_pcos_core_components_are_protected_and_complete() -> None:
    architecture = default_pcos_architecture()

    assert architecture.core_is_protected is True
    assert set(component.kind for component in architecture.core_components) == set(
        PCOSCoreComponentKind
    )
    assert architecture.core_component(PCOSCoreComponentKind.SAFETY).protected is True
    assert "cannot be weakened" in (
        architecture.core_component(PCOSCoreComponentKind.SAFETY).mutation_policy
    )
    assert "reversible" in (
        architecture.core_component(PCOSCoreComponentKind.MEMORY).mutation_policy
    )


def test_pcos_flow_forces_context_memory_safety_and_feedback() -> None:
    architecture = default_pcos_architecture()

    assert architecture.critical_flow_has_no_bypass is True
    assert architecture.supports_continuity is True
    assert tuple(step.kind for step in architecture.flow) == (
        PCOSFlowStepKind.USER,
        PCOSFlowStepKind.VOICE,
        PCOSFlowStepKind.PERCEPTION,
        PCOSFlowStepKind.MISSION_CONTEXT,
        PCOSFlowStepKind.MEMORY,
        PCOSFlowStepKind.REASONING,
        PCOSFlowStepKind.PLANNING,
        PCOSFlowStepKind.SAFETY,
        PCOSFlowStepKind.EXECUTION,
        PCOSFlowStepKind.FEEDBACK,
        PCOSFlowStepKind.MEMORY_UPDATE,
    )


def test_pcos_growth_cannot_modify_core_or_safety_without_review() -> None:
    architecture = default_pcos_architecture()

    forbidden = " ".join(architecture.forbidden_self_modifications)
    growth = " ".join(architecture.growth_mechanisms)

    assert "rewrite protected core architecture" in forbidden
    assert "grant unrestricted permissions" in forbidden
    assert "weaken safety" in forbidden
    assert "replace capability adapters" in growth
    assert "governed memory" in growth


def test_pcos_rejects_unprotected_core_component() -> None:
    with pytest.raises(ValueError, match="must be protected"):
        PCOSCoreComponent(
            kind=PCOSCoreComponentKind.IDENTITY,
            purpose="identity",
            protected=False,
            must_be_observable=True,
            mutation_policy="requires review",
        )


def test_pcos_rejects_replaceable_core_layer() -> None:
    with pytest.raises(ValueError, match="core layer cannot"):
        PCOSLayer(
            kind=PCOSLayerKind.CORE,
            purpose="core",
            stability_rule="stable",
            allowed_to_replace=True,
        )


def test_pcos_rejects_bypassed_critical_flow_step() -> None:
    with pytest.raises(ValueError, match="cannot be bypassed"):
        PCOSFlowStep(
            kind=PCOSFlowStepKind.SAFETY,
            purpose="safety",
            bypass_allowed=True,
        )
