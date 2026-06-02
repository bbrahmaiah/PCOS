from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.system.bootstrap import (
    JarvisBootstrapConfig,
    JarvisSystemBootstrap,
    JarvisSystemFactoryBundle,
)
from jarvis.system.contracts import utc_now


class JarvisComponentKind(StrEnum):
    CONFIG = "config"
    KERNEL = "kernel"
    EVENT_BUS = "event_bus"
    MEMORY = "memory"
    COGNITION = "cognition"
    CONVERSATION = "conversation"
    PRESENCE = "presence"
    ORCHESTRATION = "orchestration"
    JARVIS_SYSTEM = "jarvis_system"


class JarvisComponentRequirement(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class JarvisComponentMode(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    DEGRADED = "degraded"


class JarvisComponentValidationStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    DISABLED = "disabled"


class JarvisDependencyGraphStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class JarvisComponentSpec:
    component_id: str
    kind: JarvisComponentKind
    requirement: JarvisComponentRequirement
    mode: JarvisComponentMode = JarvisComponentMode.ENABLED
    dependencies: tuple[str, ...] = ()
    factory_present: bool = True
    can_degrade: bool = False
    description: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id cannot be empty.")
        if self.component_id in self.dependencies:
            raise ValueError("component cannot depend on itself.")


@dataclass(frozen=True, slots=True)
class JarvisComponentValidation:
    component_id: str
    kind: JarvisComponentKind
    status: JarvisComponentValidationStatus
    requirement: JarvisComponentRequirement
    mode: JarvisComponentMode
    missing_dependencies: tuple[str, ...]
    reason: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def blocks_boot(self) -> bool:
        return (
            self.status == JarvisComponentValidationStatus.BLOCKED
            and self.requirement == JarvisComponentRequirement.REQUIRED
        )


@dataclass(frozen=True, slots=True)
class JarvisDependencyGraphReport:
    status: JarvisDependencyGraphStatus
    components: tuple[JarvisComponentValidation, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def boot_allowed(self) -> bool:
        return self.status in {
            JarvisDependencyGraphStatus.READY,
            JarvisDependencyGraphStatus.DEGRADED,
        }

    @property
    def ready_count(self) -> int:
        return sum(
            1
            for component in self.components
            if component.status == JarvisComponentValidationStatus.READY
        )

    @property
    def degraded_count(self) -> int:
        return sum(
            1
            for component in self.components
            if component.status == JarvisComponentValidationStatus.DEGRADED
        )

    @property
    def blocked_count(self) -> int:
        return sum(
            1
            for component in self.components
            if component.status == JarvisComponentValidationStatus.BLOCKED
        )

    @property
    def disabled_count(self) -> int:
        return sum(
            1
            for component in self.components
            if component.status == JarvisComponentValidationStatus.DISABLED
        )


@dataclass(frozen=True, slots=True)
class JarvisCompositionOverride:
    component_id: str
    mode: JarvisComponentMode | None = None
    factory_present: bool | None = None
    can_degrade: bool | None = None
    requirement: JarvisComponentRequirement | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("override component_id cannot be empty.")


@dataclass(frozen=True, slots=True)
class JarvisCompositionRootSnapshot:
    name: str
    dependency_graph: JarvisDependencyGraphReport
    boot_allowed: bool
    created_at: datetime


class JarvisCompositionRoot:
    """
    Step 46B composition root.

    This is the single intentional construction boundary before live startup.

    Responsibilities:
    - represent the live dependency graph
    - validate required vs optional components
    - classify degraded mode
    - block unsafe boot when required dependencies are missing
    - build JarvisSystemBootstrap only after validation

    It does not start hardware, run cognition, execute tools, or hide global
    dependencies.
    """

    def __init__(
        self,
        *,
        config: JarvisBootstrapConfig,
        factories: JarvisSystemFactoryBundle,
        overrides: Iterable[JarvisCompositionOverride] = (),
    ) -> None:
        self._config = config
        self._factories = factories
        self._overrides = tuple(overrides)
        self._last_report: JarvisDependencyGraphReport | None = None

    @property
    def config(self) -> JarvisBootstrapConfig:
        return self._config

    @property
    def factories(self) -> JarvisSystemFactoryBundle:
        return self._factories

    def component_specs(self) -> tuple[JarvisComponentSpec, ...]:
        specs = _default_component_specs(
            config=self._config,
            factories=self._factories,
        )
        return _apply_overrides(specs=specs, overrides=self._overrides)

    def validate_dependency_graph(self) -> JarvisDependencyGraphReport:
        report = validate_dependency_graph(self.component_specs())
        self._last_report = report
        return report

    def build_bootstrap(self) -> JarvisSystemBootstrap:
        report = self.validate_dependency_graph()
        if not report.boot_allowed:
            blocked = [
                component.component_id
                for component in report.components
                if component.blocks_boot
            ]
            raise RuntimeError(
                "Jarvis composition root blocked boot. "
                f"blocked_components={blocked}"
            )

        return JarvisSystemBootstrap(
            config=self._config,
            factories=self._factories,
        )

    def snapshot(self) -> JarvisCompositionRootSnapshot:
        report = self._last_report or self.validate_dependency_graph()
        return JarvisCompositionRootSnapshot(
            name=self._config.name,
            dependency_graph=report,
            boot_allowed=report.boot_allowed,
            created_at=utc_now(),
        )


def validate_dependency_graph(
    specs: Iterable[JarvisComponentSpec],
) -> JarvisDependencyGraphReport:
    component_specs = tuple(specs)
    by_id = {spec.component_id: spec for spec in component_specs}

    validations: list[JarvisComponentValidation] = []
    blocked_ids: set[str] = set()

    for spec in component_specs:
        validation = _validate_component(
            spec=spec,
            known_component_ids=set(by_id),
            blocked_ids=blocked_ids,
        )
        validations.append(validation)

        if validation.status == JarvisComponentValidationStatus.BLOCKED:
            blocked_ids.add(spec.component_id)

    graph_status = _graph_status(validations)

    return JarvisDependencyGraphReport(
        status=graph_status,
        components=tuple(validations),
        created_at=utc_now(),
        metadata={
            "component_count": len(validations),
            "blocked_components": tuple(
                validation.component_id
                for validation in validations
                if validation.status
                == JarvisComponentValidationStatus.BLOCKED
            ),
            "degraded_components": tuple(
                validation.component_id
                for validation in validations
                if validation.status
                == JarvisComponentValidationStatus.DEGRADED
            ),
        },
    )


def _validate_component(
    *,
    spec: JarvisComponentSpec,
    known_component_ids: set[str],
    blocked_ids: set[str],
) -> JarvisComponentValidation:
    if spec.mode == JarvisComponentMode.DISABLED:
        return JarvisComponentValidation(
            component_id=spec.component_id,
            kind=spec.kind,
            status=JarvisComponentValidationStatus.DISABLED,
            requirement=spec.requirement,
            mode=spec.mode,
            missing_dependencies=(),
            reason="component disabled by composition config",
            created_at=utc_now(),
            metadata=spec.metadata,
        )

    missing_dependencies = tuple(
        dependency
        for dependency in spec.dependencies
        if dependency not in known_component_ids or dependency in blocked_ids
    )

    if missing_dependencies:
        if spec.can_degrade:
            return JarvisComponentValidation(
                component_id=spec.component_id,
                kind=spec.kind,
                status=JarvisComponentValidationStatus.DEGRADED,
                requirement=spec.requirement,
                mode=JarvisComponentMode.DEGRADED,
                missing_dependencies=missing_dependencies,
                reason="component degraded because dependencies are missing",
                created_at=utc_now(),
                metadata=spec.metadata,
            )

        return JarvisComponentValidation(
            component_id=spec.component_id,
            kind=spec.kind,
            status=JarvisComponentValidationStatus.BLOCKED,
            requirement=spec.requirement,
            mode=spec.mode,
            missing_dependencies=missing_dependencies,
            reason="component blocked because dependencies are missing",
            created_at=utc_now(),
            metadata=spec.metadata,
        )

    if not spec.factory_present:
        if spec.can_degrade:
            return JarvisComponentValidation(
                component_id=spec.component_id,
                kind=spec.kind,
                status=JarvisComponentValidationStatus.DEGRADED,
                requirement=spec.requirement,
                mode=JarvisComponentMode.DEGRADED,
                missing_dependencies=(),
                reason="component degraded because factory is missing",
                created_at=utc_now(),
                metadata=spec.metadata,
            )

        return JarvisComponentValidation(
            component_id=spec.component_id,
            kind=spec.kind,
            status=JarvisComponentValidationStatus.BLOCKED,
            requirement=spec.requirement,
            mode=spec.mode,
            missing_dependencies=(),
            reason="component blocked because factory is missing",
            created_at=utc_now(),
            metadata=spec.metadata,
        )

    if spec.mode == JarvisComponentMode.DEGRADED:
        return JarvisComponentValidation(
            component_id=spec.component_id,
            kind=spec.kind,
            status=JarvisComponentValidationStatus.DEGRADED,
            requirement=spec.requirement,
            mode=spec.mode,
            missing_dependencies=(),
            reason="component explicitly configured in degraded mode",
            created_at=utc_now(),
            metadata=spec.metadata,
        )

    return JarvisComponentValidation(
        component_id=spec.component_id,
        kind=spec.kind,
        status=JarvisComponentValidationStatus.READY,
        requirement=spec.requirement,
        mode=spec.mode,
        missing_dependencies=(),
        reason="component ready",
        created_at=utc_now(),
        metadata=spec.metadata,
    )


def _graph_status(
    validations: list[JarvisComponentValidation],
) -> JarvisDependencyGraphStatus:
    if any(validation.blocks_boot for validation in validations):
        return JarvisDependencyGraphStatus.BLOCKED

    if any(
        validation.status == JarvisComponentValidationStatus.DEGRADED
        for validation in validations
    ):
        return JarvisDependencyGraphStatus.DEGRADED

    return JarvisDependencyGraphStatus.READY


def _default_component_specs(
    *,
    config: JarvisBootstrapConfig,
    factories: JarvisSystemFactoryBundle,
) -> tuple[JarvisComponentSpec, ...]:
    conversation_enabled = (
        config.attach_conversation
        and factories.conversation_runtime is not None
    )
    presence_enabled = (
        config.attach_presence
        and factories.presence_engine is not None
    )
    orchestration_enabled = (
        config.attach_orchestration
        and factories.orchestration_runtime is not None
    )

    return (
        JarvisComponentSpec(
            component_id="config",
            kind=JarvisComponentKind.CONFIG,
            requirement=JarvisComponentRequirement.REQUIRED,
            description="bootstrap configuration",
        ),
        JarvisComponentSpec(
            component_id="kernel",
            kind=JarvisComponentKind.KERNEL,
            requirement=JarvisComponentRequirement.REQUIRED,
            dependencies=("config",),
            factory_present=factories.kernel is not None,
            description="runtime kernel factory",
        ),
        JarvisComponentSpec(
            component_id="event_bus",
            kind=JarvisComponentKind.EVENT_BUS,
            requirement=JarvisComponentRequirement.REQUIRED,
            dependencies=("kernel",),
            description="event bus owned by runtime kernel",
        ),
        JarvisComponentSpec(
            component_id="memory",
            kind=JarvisComponentKind.MEMORY,
            requirement=JarvisComponentRequirement.REQUIRED,
            dependencies=("config",),
            factory_present=factories.memory_gateway is not None,
            description="governed memory gateway",
        ),
        JarvisComponentSpec(
            component_id="cognition",
            kind=JarvisComponentKind.COGNITION,
            requirement=JarvisComponentRequirement.REQUIRED,
            dependencies=("config", "memory"),
            factory_present=factories.cognition_worker is not None,
            description="cognition worker",
        ),
        JarvisComponentSpec(
            component_id="conversation",
            kind=JarvisComponentKind.CONVERSATION,
            requirement=JarvisComponentRequirement.OPTIONAL,
            mode=(
                JarvisComponentMode.ENABLED
                if conversation_enabled
                else JarvisComponentMode.DISABLED
            ),
            dependencies=("cognition",),
            factory_present=conversation_enabled,
            can_degrade=True,
            description="conversation runtime",
        ),
        JarvisComponentSpec(
            component_id="presence",
            kind=JarvisComponentKind.PRESENCE,
            requirement=JarvisComponentRequirement.OPTIONAL,
            mode=(
                JarvisComponentMode.ENABLED
                if presence_enabled
                else JarvisComponentMode.DISABLED
            ),
            dependencies=("conversation",),
            factory_present=presence_enabled,
            can_degrade=True,
            description="presence engine",
        ),
        JarvisComponentSpec(
            component_id="orchestration",
            kind=JarvisComponentKind.ORCHESTRATION,
            requirement=JarvisComponentRequirement.OPTIONAL,
            mode=(
                JarvisComponentMode.ENABLED
                if orchestration_enabled
                else JarvisComponentMode.DISABLED
            ),
            dependencies=("event_bus",),
            factory_present=orchestration_enabled,
            can_degrade=True,
            description="orchestration runtime",
        ),
        JarvisComponentSpec(
            component_id="jarvis_system",
            kind=JarvisComponentKind.JARVIS_SYSTEM,
            requirement=JarvisComponentRequirement.REQUIRED,
            dependencies=("kernel", "event_bus", "memory", "cognition"),
            description="assembled JarvisSystem",
        ),
    )


def _apply_overrides(
    *,
    specs: tuple[JarvisComponentSpec, ...],
    overrides: tuple[JarvisCompositionOverride, ...],
) -> tuple[JarvisComponentSpec, ...]:
    override_by_id = {override.component_id: override for override in overrides}
    updated: list[JarvisComponentSpec] = []

    for spec in specs:
        override = override_by_id.get(spec.component_id)
        if override is None:
            updated.append(spec)
            continue

        metadata = {
            **spec.metadata,
            **override.metadata,
            "overridden": True,
        }
        updated.append(
            JarvisComponentSpec(
                component_id=spec.component_id,
                kind=spec.kind,
                requirement=override.requirement or spec.requirement,
                mode=override.mode or spec.mode,
                dependencies=spec.dependencies,
                factory_present=(
                    spec.factory_present
                    if override.factory_present is None
                    else override.factory_present
                ),
                can_degrade=(
                    spec.can_degrade
                    if override.can_degrade is None
                    else override.can_degrade
                ),
                description=spec.description,
                metadata=metadata,
            )
        )

    unknown_overrides = tuple(
        component_id
        for component_id in override_by_id
        if component_id not in {spec.component_id for spec in specs}
    )
    if unknown_overrides:
        raise ValueError(f"unknown composition overrides: {unknown_overrides}")

    return tuple(updated)