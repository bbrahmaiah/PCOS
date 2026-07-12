from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class PureJarvisCapabilityKind(StrEnum):
    CONVERSATION = "conversation"
    ALWAYS_ON_VOICE = "always_on_voice"
    PRESENCE = "presence"
    INTERRUPTION = "interruption"
    MEMORY = "memory"
    RESEARCH = "research"
    ENGINEERING = "engineering"
    SYSTEM_CONTROL = "system_control"
    SAFETY = "safety"
    SECURITY = "security"
    COMMUNICATION = "communication"
    VISION = "vision"
    SIMULATION = "simulation"
    PROACTIVITY = "proactivity"
    AUTONOMY = "autonomy"
    PERSONALITY = "personality"
    SPEED = "speed"
    RECOVERY = "recovery"
    STEP50_INTEGRATION = "step50_integration"


class PureJarvisAutonomyLevel(StrEnum):
    OBSERVE_ONLY = "observe_only"
    SUGGEST = "suggest"
    CONFIRM_THEN_ACT = "confirm_then_act"
    SAFE_AUTOMATIC = "safe_automatic"
    EMERGENCY_INTERRUPT = "emergency_interrupt"


class PureJarvisSafetyClass(StrEnum):
    LOW_RISK = "low_risk"
    PERSONAL_DATA = "personal_data"
    SYSTEM_MUTATION = "system_mutation"
    EXTERNAL_COMMUNICATION = "external_communication"
    HIGH_IMPACT = "high_impact"


class PureJarvisRequirementStatus(StrEnum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    SAFETY_GATED = "safety_gated"
    NEEDS_EXTERNAL_INTEGRATION = "needs_external_integration"
    HARDWARE_LIMITED = "hardware_limited"
    PHYSICS_LIMITED = "physics_limited"


class PCOSLayerKind(StrEnum):
    CORE = "core"
    CAPABILITY = "capability"
    INTELLIGENCE_GROWTH = "intelligence_growth"


class PCOSCoreComponentKind(StrEnum):
    IDENTITY = "identity"
    MEMORY = "memory"
    REASONING = "reasoning"
    SAFETY = "safety"
    MISSION_CONTEXT = "mission_context"
    EVENT_BUS = "event_bus"
    STATE_MANAGEMENT = "state_management"
    TOOL_FRAMEWORK = "tool_framework"


class PCOSFlowStepKind(StrEnum):
    USER = "user"
    VOICE = "voice"
    PERCEPTION = "perception"
    MISSION_CONTEXT = "mission_context"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    SAFETY = "safety"
    EXECUTION = "execution"
    FEEDBACK = "feedback"
    MEMORY_UPDATE = "memory_update"


@dataclass(frozen=True, slots=True)
class PCOSCoreComponent:
    kind: PCOSCoreComponentKind
    purpose: str
    protected: bool
    must_be_observable: bool
    mutation_policy: str

    def __post_init__(self) -> None:
        if not self.purpose.strip():
            raise ValueError("PCOS core component purpose cannot be empty.")
        if not self.mutation_policy.strip():
            raise ValueError("PCOS core component mutation_policy cannot be empty.")
        if not self.protected:
            raise ValueError("PCOS core components must be protected.")
        if not self.must_be_observable:
            raise ValueError("PCOS core components must be observable.")


@dataclass(frozen=True, slots=True)
class PCOSLayer:
    kind: PCOSLayerKind
    purpose: str
    stability_rule: str
    allowed_to_replace: bool

    def __post_init__(self) -> None:
        if not self.purpose.strip():
            raise ValueError("PCOS layer purpose cannot be empty.")
        if not self.stability_rule.strip():
            raise ValueError("PCOS layer stability_rule cannot be empty.")
        if self.kind == PCOSLayerKind.CORE and self.allowed_to_replace:
            raise ValueError("PCOS core layer cannot be freely replaceable.")


@dataclass(frozen=True, slots=True)
class PCOSFlowStep:
    kind: PCOSFlowStepKind
    purpose: str
    bypass_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.purpose.strip():
            raise ValueError("PCOS flow step purpose cannot be empty.")
        if self.kind in {
            PCOSFlowStepKind.MISSION_CONTEXT,
            PCOSFlowStepKind.MEMORY,
            PCOSFlowStepKind.SAFETY,
            PCOSFlowStepKind.MEMORY_UPDATE,
        } and self.bypass_allowed:
            raise ValueError("critical PCOS flow steps cannot be bypassed.")


@dataclass(frozen=True, slots=True)
class PCOSArchitecture:
    name: str
    layers: tuple[PCOSLayer, ...]
    core_components: tuple[PCOSCoreComponent, ...]
    flow: tuple[PCOSFlowStep, ...]
    capability_surfaces: tuple[PureJarvisCapabilityKind, ...]
    growth_mechanisms: tuple[str, ...]
    forbidden_self_modifications: tuple[str, ...]

    @property
    def core_is_protected(self) -> bool:
        return all(component.protected for component in self.core_components)

    @property
    def critical_flow_has_no_bypass(self) -> bool:
        return all(not step.bypass_allowed for step in self.flow)

    @property
    def supports_continuity(self) -> bool:
        flow_kinds = tuple(step.kind for step in self.flow)
        return (
            PCOSFlowStepKind.MEMORY in flow_kinds
            and PCOSFlowStepKind.MISSION_CONTEXT in flow_kinds
            and PCOSFlowStepKind.MEMORY_UPDATE in flow_kinds
        )

    def core_component(
        self,
        kind: PCOSCoreComponentKind,
    ) -> PCOSCoreComponent:
        for component in self.core_components:
            if component.kind == kind:
                return component
        raise KeyError(kind.value)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("PCOS architecture name cannot be empty.")
        if tuple(layer.kind for layer in self.layers) != (
            PCOSLayerKind.CORE,
            PCOSLayerKind.CAPABILITY,
            PCOSLayerKind.INTELLIGENCE_GROWTH,
        ):
            raise ValueError("PCOS architecture must define the three layers.")
        if set(component.kind for component in self.core_components) != set(
            PCOSCoreComponentKind
        ):
            raise ValueError("PCOS architecture must cover every core component.")
        if not self.flow:
            raise ValueError("PCOS architecture flow cannot be empty.")
        if not self.capability_surfaces:
            raise ValueError("PCOS architecture capability_surfaces cannot be empty.")
        if not self.growth_mechanisms:
            raise ValueError("PCOS architecture growth_mechanisms cannot be empty.")
        if not self.forbidden_self_modifications:
            raise ValueError(
                "PCOS architecture forbidden_self_modifications cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class PureJarvisCapability:
    kind: PureJarvisCapabilityKind
    purpose: str
    autonomy: PureJarvisAutonomyLevel
    safety_class: PureJarvisSafetyClass
    real_time_required: bool
    proactive_allowed: bool
    human_confirmation_required: bool
    phase_owner: str
    acceptance: tuple[str, ...]
    safety_rules: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.purpose.strip():
            raise ValueError("capability purpose cannot be empty.")
        if not self.phase_owner.strip():
            raise ValueError("capability phase_owner cannot be empty.")
        if not self.acceptance:
            raise ValueError("capability acceptance cannot be empty.")
        if (
            self.autonomy == PureJarvisAutonomyLevel.SAFE_AUTOMATIC
            and self.safety_class
            in {
                PureJarvisSafetyClass.EXTERNAL_COMMUNICATION,
                PureJarvisSafetyClass.HIGH_IMPACT,
                PureJarvisSafetyClass.SYSTEM_MUTATION,
            }
            and not self.human_confirmation_required
        ):
            raise ValueError(
                "high-impact automatic capability must require confirmation."
            )


@dataclass(frozen=True, slots=True)
class PureJarvisRequirementJustification:
    request_line: str
    capabilities: tuple[PureJarvisCapabilityKind, ...]
    status: PureJarvisRequirementStatus
    evidence: tuple[str, ...]
    remaining_work: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fully_satisfied(self) -> bool:
        return self.status == PureJarvisRequirementStatus.SATISFIED

    @property
    def blocked_or_gated(self) -> bool:
        return self.status in {
            PureJarvisRequirementStatus.SAFETY_GATED,
            PureJarvisRequirementStatus.NEEDS_EXTERNAL_INTEGRATION,
            PureJarvisRequirementStatus.HARDWARE_LIMITED,
            PureJarvisRequirementStatus.PHYSICS_LIMITED,
        }

    def __post_init__(self) -> None:
        if not self.request_line.strip():
            raise ValueError("requirement request_line cannot be empty.")
        if not self.capabilities:
            raise ValueError("requirement capabilities cannot be empty.")
        if not self.evidence:
            raise ValueError("requirement evidence cannot be empty.")
        if self.status != PureJarvisRequirementStatus.SATISFIED:
            if not self.remaining_work:
                raise ValueError(
                    "unsatisfied/gated requirement must explain remaining work."
                )


@dataclass(frozen=True, slots=True)
class PureJarvisDoctrine:
    name: str
    user_label: str
    assistant_label: str
    mission: str
    personality: tuple[str, ...]
    non_negotiables: tuple[str, ...]
    confidence_policy: tuple[str, ...]
    response_style: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("doctrine name cannot be empty.")
        if not self.user_label.strip():
            raise ValueError("doctrine user_label cannot be empty.")
        if not self.assistant_label.strip():
            raise ValueError("doctrine assistant_label cannot be empty.")
        if not self.mission.strip():
            raise ValueError("doctrine mission cannot be empty.")
        if not self.personality:
            raise ValueError("doctrine personality cannot be empty.")
        if not self.non_negotiables:
            raise ValueError("doctrine non_negotiables cannot be empty.")
        if not self.confidence_policy:
            raise ValueError("doctrine confidence_policy cannot be empty.")
        if not self.response_style:
            raise ValueError("doctrine response_style cannot be empty.")


@dataclass(frozen=True, slots=True)
class PureJarvisManifest:
    doctrine: PureJarvisDoctrine
    capabilities: tuple[PureJarvisCapability, ...]
    minimum_real_time_capabilities: int
    required_capabilities: tuple[PureJarvisCapabilityKind, ...]

    @property
    def capability_kinds(self) -> frozenset[PureJarvisCapabilityKind]:
        return frozenset(capability.kind for capability in self.capabilities)

    @property
    def real_time_capability_count(self) -> int:
        return sum(
            1 for capability in self.capabilities if capability.real_time_required
        )

    @property
    def proactive_capability_count(self) -> int:
        return sum(
            1 for capability in self.capabilities if capability.proactive_allowed
        )

    @property
    def covers_required_capabilities(self) -> bool:
        return set(self.required_capabilities).issubset(self.capability_kinds)

    @property
    def ready_for_pure_runtime(self) -> bool:
        return (
            self.covers_required_capabilities
            and self.real_time_capability_count >= self.minimum_real_time_capabilities
            and any(
                capability.kind == PureJarvisCapabilityKind.SAFETY
                and capability.proactive_allowed
                for capability in self.capabilities
            )
            and any(
                capability.kind == PureJarvisCapabilityKind.STEP50_INTEGRATION
                for capability in self.capabilities
            )
        )

    def capability(self, kind: PureJarvisCapabilityKind) -> PureJarvisCapability:
        for capability in self.capabilities:
            if capability.kind == kind:
                return capability
        raise KeyError(kind.value)

    def missing_required_capabilities(self) -> tuple[PureJarvisCapabilityKind, ...]:
        present = self.capability_kinds
        return tuple(kind for kind in self.required_capabilities if kind not in present)


REQUIRED_PURE_JARVIS_CAPABILITIES: tuple[PureJarvisCapabilityKind, ...] = (
    PureJarvisCapabilityKind.CONVERSATION,
    PureJarvisCapabilityKind.ALWAYS_ON_VOICE,
    PureJarvisCapabilityKind.PRESENCE,
    PureJarvisCapabilityKind.INTERRUPTION,
    PureJarvisCapabilityKind.MEMORY,
    PureJarvisCapabilityKind.RESEARCH,
    PureJarvisCapabilityKind.ENGINEERING,
    PureJarvisCapabilityKind.SYSTEM_CONTROL,
    PureJarvisCapabilityKind.SAFETY,
    PureJarvisCapabilityKind.SECURITY,
    PureJarvisCapabilityKind.COMMUNICATION,
    PureJarvisCapabilityKind.VISION,
    PureJarvisCapabilityKind.SIMULATION,
    PureJarvisCapabilityKind.PROACTIVITY,
    PureJarvisCapabilityKind.AUTONOMY,
    PureJarvisCapabilityKind.PERSONALITY,
    PureJarvisCapabilityKind.SPEED,
    PureJarvisCapabilityKind.RECOVERY,
    PureJarvisCapabilityKind.STEP50_INTEGRATION,
)


def default_pcos_architecture() -> PCOSArchitecture:
    return PCOSArchitecture(
        name="Personal Cognitive Operating System protected core architecture",
        layers=(
            PCOSLayer(
                kind=PCOSLayerKind.CORE,
                purpose=(
                    "Stable foundation that owns identity, memory, reasoning, "
                    "safety, mission context, events, state, and tool trust."
                ),
                stability_rule=(
                    "Never rewrite casually; change only by reviewed migration."
                ),
                allowed_to_replace=False,
            ),
            PCOSLayer(
                kind=PCOSLayerKind.CAPABILITY,
                purpose=(
                    "Swappable speech, vision, research, engineering, "
                    "simulation, communication, automation, security, and "
                    "health adapters."
                ),
                stability_rule="Adapters may be upgraded without changing core policy.",
                allowed_to_replace=True,
            ),
            PCOSLayer(
                kind=PCOSLayerKind.INTELLIGENCE_GROWTH,
                purpose=(
                    "Learning, adaptation, prediction, habits, preferences, "
                    "and self-optimization through governed memory."
                ),
                stability_rule=(
                    "Growth may tune behavior, but must not bypass protected core."
                ),
                allowed_to_replace=True,
            ),
        ),
        core_components=(
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.IDENTITY,
                purpose="Know who is being helped and which assistant is acting.",
                protected=True,
                must_be_observable=True,
                mutation_policy="profile changes require explicit user approval",
            ),
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.MEMORY,
                purpose=(
                    "Preserve continuity across projects, days, habits, and "
                    "missions."
                ),
                protected=True,
                must_be_observable=True,
                mutation_policy="memory writes are governed, classed, and reversible",
            ),
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.REASONING,
                purpose="Turn context into decisions without bypassing safety.",
                protected=True,
                must_be_observable=True,
                mutation_policy="model/provider swaps must preserve response boundary",
            ),
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.SAFETY,
                purpose="Decide what is safe, blocked, warned, or confirmation-gated.",
                protected=True,
                must_be_observable=True,
                mutation_policy="safety policy cannot be weakened automatically",
            ),
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.MISSION_CONTEXT,
                purpose="Track current work, unfinished tasks, and active objective.",
                protected=True,
                must_be_observable=True,
                mutation_policy="mission updates require evidence or user intent",
            ),
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.EVENT_BUS,
                purpose="Coordinate perception, cognition, tools, and feedback.",
                protected=True,
                must_be_observable=True,
                mutation_policy="events must remain typed, auditable, and ordered",
            ),
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.STATE_MANAGEMENT,
                purpose="Maintain current runtime, user, task, and subsystem state.",
                protected=True,
                must_be_observable=True,
                mutation_policy="state transitions must be explicit and inspectable",
            ),
            PCOSCoreComponent(
                kind=PCOSCoreComponentKind.TOOL_FRAMEWORK,
                purpose=(
                    "Route actions through planner, validator, executor, and "
                    "audit."
                ),
                protected=True,
                must_be_observable=True,
                mutation_policy="new tools must declare risk and permission policy",
            ),
        ),
        flow=(
            PCOSFlowStep(
                kind=PCOSFlowStepKind.USER,
                purpose="Tony speaks, acts, works, or creates an observable signal.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.VOICE,
                purpose="Speech enters through the realtime voice capability.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.PERCEPTION,
                purpose="Convert raw signals into confidence-scored intent state.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.MISSION_CONTEXT,
                purpose="Attach current project, objective, and unfinished work.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.MEMORY,
                purpose="Retrieve relevant short-term and long-term continuity.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.REASONING,
                purpose=(
                    "Reason over intent, context, memory, constraints, and "
                    "evidence."
                ),
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.PLANNING,
                purpose="Create or update a plan before action when needed.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.SAFETY,
                purpose="Classify risk, confidence, confirmation need, and safe path.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.EXECUTION,
                purpose="Execute approved action or produce spoken/visual response.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.FEEDBACK,
                purpose="Report useful result, blocker, warning, or next step.",
            ),
            PCOSFlowStep(
                kind=PCOSFlowStepKind.MEMORY_UPDATE,
                purpose="Write governed continuity back into memory when appropriate.",
            ),
        ),
        capability_surfaces=(
            PureJarvisCapabilityKind.ALWAYS_ON_VOICE,
            PureJarvisCapabilityKind.VISION,
            PureJarvisCapabilityKind.RESEARCH,
            PureJarvisCapabilityKind.ENGINEERING,
            PureJarvisCapabilityKind.SIMULATION,
            PureJarvisCapabilityKind.COMMUNICATION,
            PureJarvisCapabilityKind.AUTONOMY,
            PureJarvisCapabilityKind.SECURITY,
            PureJarvisCapabilityKind.SAFETY,
        ),
        growth_mechanisms=(
            "learn habits from repeated approved behavior",
            "adapt preferences through governed memory",
            "predict useful next actions from mission context",
            "optimize model, prompt, and latency settings under policy",
            "replace capability adapters without rewriting protected core",
        ),
        forbidden_self_modifications=(
            "rewrite protected core architecture without review",
            "grant unrestricted permissions to itself",
            "weaken safety, privacy, or confirmation policies automatically",
            "execute destructive or external actions outside the tool framework",
        ),
    )


def default_pure_jarvis_doctrine() -> PureJarvisDoctrine:
    return PureJarvisDoctrine(
        name="Pure JARVIS personal cognitive operating system",
        user_label="Tony",
        assistant_label="JARVIS",
        mission=(
            "Act as a calm always-available cognitive operating layer that listens, "
            "understands context, remembers, reasons, monitors, warns, assists, "
            "automates safe work, and protects the user's environment."
        ),
        personality=(
            "calm",
            "respectful",
            "precise",
            "loyal",
            "patient",
            "dryly witty when appropriate",
            "willing to challenge unsafe or incomplete instructions",
        ),
        non_negotiables=(
            "Architecture and speed are first-class requirements; every live "
            "subsystem must have one owner, one boundary, and one latency role.",
            "Keep heavy diagnostics, factory dry-runs, model warmups, and probes "
            "outside the daily-driver hot path unless explicitly requested.",
            "Never pretend a subsystem is live when it is not verified.",
            "Never execute destructive, privacy-sensitive, financial, medical, "
            "or external communication actions without the configured "
            "confirmation policy.",
            "Stay silent when there is no useful signal, but interrupt "
            "immediately for safety, security, or high-priority user "
            "welfare events.",
            "Prefer short spoken responses and richer visual/data overlays "
            "when detail is needed.",
            "Keep memory governed, explainable, and reversible.",
        ),
        confidence_policy=(
            "Act automatically only for low-risk reversible tasks with "
            "high confidence.",
            "Ask a clarifying question when intent, target, or safety "
            "impact is incomplete.",
            "Warn instead of acting when confidence is low or consequences are high.",
            "Escalate to interruption when delay can harm the user, "
            "system, data, or ongoing work.",
        ),
        response_style=(
            "Use concise spoken replies by default.",
            "Acknowledge safe tasks with brief calm confirmations",
            "Use natural conversation instead of command-only parsing.",
            "Show only the important ranked information, not raw search noise.",
        ),
    )


def default_pure_jarvis_manifest() -> PureJarvisManifest:
    capabilities = (
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.CONVERSATION,
            purpose="Understand casual speech, preserve context, and answer naturally.",
            autonomy=PureJarvisAutonomyLevel.SUGGEST,
            safety_class=PureJarvisSafetyClass.LOW_RISK,
            real_time_required=True,
            proactive_allowed=False,
            human_confirmation_required=False,
            phase_owner="conversation + cognition + voice",
            acceptance=(
                "responds in short natural turns",
                "asks clarifying questions for incomplete instructions",
                "continues multi-turn context",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.ALWAYS_ON_VOICE,
            purpose="Keep the microphone-to-STT-to-cognition-to-TTS loop ready.",
            autonomy=PureJarvisAutonomyLevel.OBSERVE_ONLY,
            safety_class=PureJarvisSafetyClass.PERSONAL_DATA,
            real_time_required=True,
            proactive_allowed=False,
            human_confirmation_required=False,
            phase_owner="voice Step 51",
            acceptance=(
                "captures real microphone frames",
                "detects speech and silence",
                "recovers from microphone, STT, TTS, and playback faults",
            ),
            safety_rules=("do not store raw audio unless explicitly configured",),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.PRESENCE,
            purpose=(
                "Remain available, aware of work state, and silent when "
                "not needed."
            ),
            autonomy=PureJarvisAutonomyLevel.OBSERVE_ONLY,
            safety_class=PureJarvisSafetyClass.PERSONAL_DATA,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="presence + environment",
            acceptance=(
                "tracks active user/work context",
                "does not spam low-value notifications",
                "can speak while the user is working",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.INTERRUPTION,
            purpose="Interrupt safely for urgent events and barge-in user corrections.",
            autonomy=PureJarvisAutonomyLevel.EMERGENCY_INTERRUPT,
            safety_class=PureJarvisSafetyClass.HIGH_IMPACT,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="voice + conversation + safety",
            acceptance=(
                "stops or pauses speech on user barge-in",
                "interrupts for high-severity safety/security signals",
                "records why interruption happened",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.MEMORY,
            purpose=(
                "Remember projects, preferences, commands, habits, and "
                "mission history."
            ),
            autonomy=PureJarvisAutonomyLevel.CONFIRM_THEN_ACT,
            safety_class=PureJarvisSafetyClass.PERSONAL_DATA,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=True,
            phase_owner="memory + cognitive working memory",
            acceptance=(
                "retrieves relevant history before reasoning",
                "writes governed memory with policy classification",
                "supports correction and deletion",
            ),
            safety_rules=("memory writes must be policy-governed",),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.RESEARCH,
            purpose="Retrieve, rank, compare, and summarize relevant information.",
            autonomy=PureJarvisAutonomyLevel.SUGGEST,
            safety_class=PureJarvisSafetyClass.LOW_RISK,
            real_time_required=False,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="tools + cognition + memory",
            acceptance=(
                "filters source noise",
                "shows ranked relevant details",
                "separates known facts from inference",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.ENGINEERING,
            purpose="Help design, build, inspect, and verify technical work.",
            autonomy=PureJarvisAutonomyLevel.CONFIRM_THEN_ACT,
            safety_class=PureJarvisSafetyClass.SYSTEM_MUTATION,
            real_time_required=False,
            proactive_allowed=True,
            human_confirmation_required=True,
            phase_owner="developer + cognition + actions",
            acceptance=(
                "plans and edits code with tests",
                "runs verification before reporting success",
                "flags uncertainty and risky changes",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.SYSTEM_CONTROL,
            purpose="Operate laptop apps, files, processes, and tools safely.",
            autonomy=PureJarvisAutonomyLevel.CONFIRM_THEN_ACT,
            safety_class=PureJarvisSafetyClass.SYSTEM_MUTATION,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=True,
            phase_owner="actions + environment + runtime",
            acceptance=(
                "executes reversible low-risk tasks quickly",
                "requires confirmation for destructive or privileged actions",
                "audits actions and outcomes",
            ),
            safety_rules=("never bypass permission and trust gates",),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.SAFETY,
            purpose="Warn before failure, unsafe instruction, or harmful consequence.",
            autonomy=PureJarvisAutonomyLevel.EMERGENCY_INTERRUPT,
            safety_class=PureJarvisSafetyClass.HIGH_IMPACT,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="safety + runtime readiness + health recovery",
            acceptance=(
                "warns for low battery, failing subsystems, unsafe plans, "
                "and bad calculations",
                "blocks unsafe execution",
                "explains the minimum useful reason",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.SECURITY,
            purpose=(
                "Protect files, credentials, system access, and suspicious "
                "activity."
            ),
            autonomy=PureJarvisAutonomyLevel.EMERGENCY_INTERRUPT,
            safety_class=PureJarvisSafetyClass.HIGH_IMPACT,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=True,
            phase_owner="security + trust + actions",
            acceptance=(
                "detects suspicious access or risky file changes",
                "requires confirmation before sensitive disclosure",
                "keeps an audit trail",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.COMMUNICATION,
            purpose="Assist with calls, messages, alerts, and communication filtering.",
            autonomy=PureJarvisAutonomyLevel.CONFIRM_THEN_ACT,
            safety_class=PureJarvisSafetyClass.EXTERNAL_COMMUNICATION,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=True,
            phase_owner="communication + actions + presence",
            acceptance=(
                "surfaces important incoming messages",
                "drafts replies without sending blindly",
                "places calls only under policy",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.VISION,
            purpose=(
                "Act as second eyes over screen, documents, camera, and "
                "visual state."
            ),
            autonomy=PureJarvisAutonomyLevel.OBSERVE_ONLY,
            safety_class=PureJarvisSafetyClass.PERSONAL_DATA,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="environment vision + awareness",
            acceptance=(
                "understands visible workflow state",
                "spots visual errors or warnings",
                "respects privacy boundaries",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.SIMULATION,
            purpose="Test ideas, predict impact, and compare designs before action.",
            autonomy=PureJarvisAutonomyLevel.SUGGEST,
            safety_class=PureJarvisSafetyClass.HIGH_IMPACT,
            real_time_required=False,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="environment simulation + cognition",
            acceptance=(
                "runs what-if analysis before high-impact actions",
                "reports assumptions and uncertainty",
                "compares alternatives",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.PROACTIVITY,
            purpose="Prepare and warn before being asked without becoming annoying.",
            autonomy=PureJarvisAutonomyLevel.SUGGEST,
            safety_class=PureJarvisSafetyClass.LOW_RISK,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="orchestration + attention + presence",
            acceptance=(
                "ranks alerts by urgency",
                "suppresses noise",
                "offers useful next actions",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.AUTONOMY,
            purpose="Execute safe background tasks and maintain long-running work.",
            autonomy=PureJarvisAutonomyLevel.SAFE_AUTOMATIC,
            safety_class=PureJarvisSafetyClass.LOW_RISK,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="orchestration + operations + actions",
            acceptance=(
                "continues approved background work",
                "reports completion or blockers",
                "does not escalate risk without approval",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.PERSONALITY,
            purpose="Maintain a calm loyal presence with respectful wit and challenge.",
            autonomy=PureJarvisAutonomyLevel.SUGGEST,
            safety_class=PureJarvisSafetyClass.LOW_RISK,
            real_time_required=True,
            proactive_allowed=False,
            human_confirmation_required=False,
            phase_owner="cognitive personality + dialogue",
            acceptance=(
                "uses calm confirmations",
                "challenges unsafe assumptions politely",
                "does not roleplay fake subsystem capability",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.SPEED,
            purpose=(
                "Keep interaction fluid through streaming, prefetch, and "
                "adaptive quality."
            ),
            autonomy=PureJarvisAutonomyLevel.OBSERVE_ONLY,
            safety_class=PureJarvisSafetyClass.LOW_RISK,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="latency + streaming + router",
            acceptance=(
                "starts useful work before final transcript when safe",
                "streams response chunks",
                "degrades quality intentionally under latency pressure",
                "keeps deep diagnostics and warmups off the live hot path",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.RECOVERY,
            purpose=(
                "Detect failures, recover subsystems, and tell the user "
                "what changed."
            ),
            autonomy=PureJarvisAutonomyLevel.SAFE_AUTOMATIC,
            safety_class=PureJarvisSafetyClass.LOW_RISK,
            real_time_required=True,
            proactive_allowed=True,
            human_confirmation_required=False,
            phase_owner="runtime health + recovery",
            acceptance=(
                "recovers microphone, STT, TTS, playback, workers, and event paths",
                "continues after recoverable interruption",
                "reports unrecoverable blockers clearly",
            ),
        ),
        PureJarvisCapability(
            kind=PureJarvisCapabilityKind.STEP50_INTEGRATION,
            purpose="Preserve the Step 50 response boundary and Phase 1-9 integration.",
            autonomy=PureJarvisAutonomyLevel.OBSERVE_ONLY,
            safety_class=PureJarvisSafetyClass.HIGH_IMPACT,
            real_time_required=True,
            proactive_allowed=False,
            human_confirmation_required=False,
            phase_owner="Step 50 + Phase 1-9 runtime",
            acceptance=(
                "voice never invents response text outside cognition boundary",
                "actions pass through safety and trust gates",
                "Phase 1-9 organs remain typed and observable",
            ),
        ),
    )

    return PureJarvisManifest(
        doctrine=default_pure_jarvis_doctrine(),
        capabilities=capabilities,
        minimum_real_time_capabilities=12,
        required_capabilities=REQUIRED_PURE_JARVIS_CAPABILITIES,
    )


def pure_jarvis_requirement_justifications() -> tuple[
    PureJarvisRequirementJustification,
    ...,
]:
    return (
        PureJarvisRequirementJustification(
            request_line="personal cognitive operating system",
            capabilities=(
                PureJarvisCapabilityKind.PRESENCE,
                PureJarvisCapabilityKind.MEMORY,
                PureJarvisCapabilityKind.AUTONOMY,
                PureJarvisCapabilityKind.STEP50_INTEGRATION,
            ),
            status=PureJarvisRequirementStatus.PARTIAL,
            evidence=(
                "Phase 1-9 organs are represented in the manifest.",
                "Step 50 response boundary is required before voice speaks.",
                "Start Control supervises connected runtime organs.",
            ),
            remaining_work=(
                "full OS-level presence requires more external app, device, "
                "calendar, message, and sensor connectors",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="always-on voice interaction",
            capabilities=(
                PureJarvisCapabilityKind.ALWAYS_ON_VOICE,
                PureJarvisCapabilityKind.CONVERSATION,
                PureJarvisCapabilityKind.INTERRUPTION,
            ),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "microphone, VAD, STT, cognition, TTS, and playback are wired",
                "speech is gated through attention and safety policy",
                "async playback allows the loop to keep listening while speaking",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="respond instantly with short intelligent replies",
            capabilities=(
                PureJarvisCapabilityKind.CONVERSATION,
                PureJarvisCapabilityKind.SPEED,
                PureJarvisCapabilityKind.PERSONALITY,
            ),
            status=PureJarvisRequirementStatus.HARDWARE_LIMITED,
            evidence=(
                "short response policy exists",
                "STT and Ollama warmups are kept off the blocking launch path",
                "diagnostics expose component latency instead of hiding delay",
            ),
            remaining_work=(
                "full answer latency depends on local CPU/GPU speed, model size, "
                "and streaming LLM/TTS availability",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="true milliseconds for complete movie-like answers",
            capabilities=(PureJarvisCapabilityKind.SPEED,),
            status=PureJarvisRequirementStatus.PHYSICS_LIMITED,
            evidence=(
                "VAD and interruption can operate in millisecond ranges",
                "complete answers require STT, reasoning, synthesis, and playback",
                "local Whisper/Ollama/Piper cannot finish all stages in true ms",
            ),
            remaining_work=(
                "use streaming first-speech, GPU/NPU acceleration, echo "
                "cancellation, and faster realtime models for movie-like feel",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="understand casual speech, not only commands",
            capabilities=(
                PureJarvisCapabilityKind.CONVERSATION,
                PureJarvisCapabilityKind.MEMORY,
            ),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "companion attention policy can accept wake-free natural speech",
                "conversation context and memory are required capabilities",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="continue conversations across context",
            capabilities=(
                PureJarvisCapabilityKind.CONVERSATION,
                PureJarvisCapabilityKind.MEMORY,
            ),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "multi-turn context is an acceptance rule",
                "memory retrieval before reasoning is an acceptance rule",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="speak while Tony is working and allow interruption",
            capabilities=(
                PureJarvisCapabilityKind.PRESENCE,
                PureJarvisCapabilityKind.INTERRUPTION,
            ),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "async Windows playback returns immediately",
                "session loop keeps assistant_speaking active until audio ends",
                "barge-in stops playback and can answer the new question",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="stay silent when not needed",
            capabilities=(
                PureJarvisCapabilityKind.PRESENCE,
                PureJarvisCapabilityKind.PROACTIVITY,
            ),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "attention gate rejects unattended speech",
                "proactivity acceptance requires ranking alerts and suppressing noise",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="ask clarifying questions when instructions are incomplete",
            capabilities=(PureJarvisCapabilityKind.CONVERSATION,),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "doctrine confidence policy requires clarifying questions",
                "conversation acceptance requires incomplete-instruction handling",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="control my laptop safely",
            capabilities=(
                PureJarvisCapabilityKind.SYSTEM_CONTROL,
                PureJarvisCapabilityKind.SECURITY,
            ),
            status=PureJarvisRequirementStatus.SAFETY_GATED,
            evidence=(
                "system control is present as a required capability",
                "system mutation actions require confirmation and audit",
            ),
            remaining_work=(
                "privileged, destructive, or external effects must stay behind "
                "permission gates",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="real-time decision support and second brain",
            capabilities=(
                PureJarvisCapabilityKind.MEMORY,
                PureJarvisCapabilityKind.RESEARCH,
                PureJarvisCapabilityKind.SAFETY,
            ),
            status=PureJarvisRequirementStatus.PARTIAL,
            evidence=(
                "memory, research, and safety are required capabilities",
                "confidence policy separates action, warning, and clarification",
            ),
            remaining_work=(
                "domain-specific health, finance, engineering, and legal tools "
                "need verified connectors and specialist policies",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="engineering help to design, build, and verify everything",
            capabilities=(PureJarvisCapabilityKind.ENGINEERING,),
            status=PureJarvisRequirementStatus.SAFETY_GATED,
            evidence=(
                "engineering is a required capability",
                "engineering acceptance requires edits with tests",
                "risky technical changes must flag uncertainty",
            ),
            remaining_work=(
                "physical builds, combat systems, and high-impact engineering "
                "must use simulation, review, and human approval gates",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="research, retrieve, rank, compare, and summarize",
            capabilities=(
                PureJarvisCapabilityKind.RESEARCH,
                PureJarvisCapabilityKind.MEMORY,
            ),
            status=PureJarvisRequirementStatus.PARTIAL,
            evidence=(
                "research capability requires filtering source noise",
                "memory capability requires relevant history retrieval",
            ),
            remaining_work=(
                "live web/news/document connectors must be configured and "
                "source-attribution rules enforced",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="remember projects, preferences, commands, and habits",
            capabilities=(PureJarvisCapabilityKind.MEMORY,),
            status=PureJarvisRequirementStatus.SAFETY_GATED,
            evidence=(
                "memory is required and policy-governed",
                "memory acceptance requires correction and deletion",
            ),
            remaining_work=(
                "personal memory writes must remain explainable, reversible, "
                "and privacy-controlled",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="protect files, system access, and sensitive data",
            capabilities=(
                PureJarvisCapabilityKind.SECURITY,
                PureJarvisCapabilityKind.SAFETY,
            ),
            status=PureJarvisRequirementStatus.SAFETY_GATED,
            evidence=(
                "security capability requires audit trails",
                "safety capability blocks unsafe execution",
            ),
            remaining_work=(
                "full endpoint protection needs OS security event connectors "
                "and explicit trust policy",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="communication officer for calls and messages",
            capabilities=(PureJarvisCapabilityKind.COMMUNICATION,),
            status=PureJarvisRequirementStatus.NEEDS_EXTERNAL_INTEGRATION,
            evidence=(
                "communication capability requires confirmation before sending",
                "external communication is classified separately for safety",
            ),
            remaining_work=(
                "phone, SMS, email, and notification APIs must be connected "
                "with consent and confirmation policy",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="display overlays and visualize calculations",
            capabilities=(
                PureJarvisCapabilityKind.VISION,
                PureJarvisCapabilityKind.SIMULATION,
            ),
            status=PureJarvisRequirementStatus.PARTIAL,
            evidence=(
                "vision and simulation are required capabilities",
                "response style prefers visual/data overlays for detail",
            ),
            remaining_work=(
                "overlay renderer and per-app display permissions need live wiring",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="simulation engine and impact prediction",
            capabilities=(PureJarvisCapabilityKind.SIMULATION,),
            status=PureJarvisRequirementStatus.PARTIAL,
            evidence=(
                "simulation capability requires what-if analysis",
                "simulation acceptance requires assumptions and uncertainty",
            ),
            remaining_work=(
                "specific structural, physics, performance, and engineering "
                "solvers must be installed per domain",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="autonomous background task execution",
            capabilities=(
                PureJarvisCapabilityKind.AUTONOMY,
                PureJarvisCapabilityKind.RECOVERY,
            ),
            status=PureJarvisRequirementStatus.SAFETY_GATED,
            evidence=(
                "safe automatic autonomy exists for low-risk tasks",
                "recovery capability requires reporting blockers and failures",
            ),
            remaining_work=(
                "higher-risk autonomous actions must remain confirm-then-act",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="proactive warnings without being annoying",
            capabilities=(
                PureJarvisCapabilityKind.PROACTIVITY,
                PureJarvisCapabilityKind.SAFETY,
            ),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "proactivity acceptance requires ranking alerts and suppressing noise",
                "safety capability permits urgent interruption",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="calm, loyal, witty, respectful personality",
            capabilities=(PureJarvisCapabilityKind.PERSONALITY,),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "doctrine personality includes calm, respectful, precise, loyal, "
                "patient, and dryly witty traits",
                "personality acceptance forbids fake subsystem capability",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="vision understands screen and surroundings",
            capabilities=(PureJarvisCapabilityKind.VISION,),
            status=PureJarvisRequirementStatus.PARTIAL,
            evidence=(
                "vision is a required observe-only capability",
                "acceptance requires visual workflow state and privacy boundaries",
            ),
            remaining_work=(
                "camera/screen OCR permissions and realtime visual model routing "
                "must be enabled per device",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="health-related warnings",
            capabilities=(PureJarvisCapabilityKind.SAFETY,),
            status=PureJarvisRequirementStatus.SAFETY_GATED,
            evidence=(
                "safety can warn for high-priority welfare events",
                "confidence policy warns instead of acting under uncertainty",
            ),
            remaining_work=(
                "health advice/monitoring requires validated devices, emergency "
                "policy, and medical-safety disclaimers",
            ),
        ),
        PureJarvisRequirementJustification(
            request_line="integration with Step 50 and Phase 1-9",
            capabilities=(PureJarvisCapabilityKind.STEP50_INTEGRATION,),
            status=PureJarvisRequirementStatus.SATISFIED,
            evidence=(
                "Step 50 integration is a required capability",
                "manifest requires voice text to originate inside cognition boundary",
                "preflight verifies required runtime imports and source fingerprint",
            ),
        ),
    )
