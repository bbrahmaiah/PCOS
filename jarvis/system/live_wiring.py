from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.system.bootstrap import (
    JarvisBootstrapConfig,
    JarvisBootstrapResult,
    JarvisSystemBootstrap,
    JarvisSystemFactoryBundle,
)
from jarvis.system.composition import (
    JarvisCompositionOverride,
    JarvisCompositionRoot,
    JarvisCompositionRootSnapshot,
    JarvisDependencyGraphReport,
)
from jarvis.system.contracts import utc_now


class LiveDependencyProfile(StrEnum):
    TEST = "test"
    LOCAL = "local"
    DEGRADED = "degraded"


class LiveDependencyWiringStatus(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    BOOTSTRAP_READY = "bootstrap_ready"
    DRY_RUN_PASSED = "dry_run_passed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LiveDependencyWiringConfig:
    """
    Step 46C live dependency wiring config.

    This config describes how the real organism should be assembled.
    It does not construct dependencies by itself.
    """

    name: str = "jarvis_live_system"
    profile: LiveDependencyProfile = LiveDependencyProfile.LOCAL
    dry_run: bool = True
    attach_conversation: bool = True
    attach_presence: bool = True
    attach_orchestration: bool = True
    allow_degraded_presence: bool = True
    allow_degraded_orchestration: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("live dependency wiring name cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveDependencyWiringReport:
    status: LiveDependencyWiringStatus
    dependency_graph: JarvisDependencyGraphReport | None
    bootstrap_result: JarvisBootstrapResult | None
    composition_snapshot: JarvisCompositionRootSnapshot | None
    created_at: datetime
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            LiveDependencyWiringStatus.VALIDATED,
            LiveDependencyWiringStatus.BOOTSTRAP_READY,
            LiveDependencyWiringStatus.DRY_RUN_PASSED,
        } and self.error is None


class LiveDependencyWiring:
    """
    Step 46C real dependency wiring boundary.

    This layer connects explicit factories into the validated composition root.
    It answers:

    - Are all real dependency factories present?
    - Can required components boot?
    - Which optional components can degrade?
    - Can JarvisSystemBootstrap be built?
    - Can the system dry-run start/stop with these factories?

    It does not hide dependencies and does not execute tools.
    """

    def __init__(
        self,
        *,
        config: LiveDependencyWiringConfig,
        factories: JarvisSystemFactoryBundle,
        overrides: Iterable[JarvisCompositionOverride] = (),
    ) -> None:
        self._config = config
        self._factories = factories
        self._overrides = tuple(overrides)
        self._composition_root: JarvisCompositionRoot | None = None

    @property
    def config(self) -> LiveDependencyWiringConfig:
        return self._config

    @property
    def factories(self) -> JarvisSystemFactoryBundle:
        return self._factories

    def bootstrap_config(self) -> JarvisBootstrapConfig:
        return JarvisBootstrapConfig(
            name=self._config.name,
            dry_run=self._config.dry_run,
            attach_conversation=self._config.attach_conversation,
            attach_presence=self._config.attach_presence,
            attach_orchestration=self._config.attach_orchestration,
            metadata={
                **self._config.metadata,
                "profile": self._config.profile.value,
                "wiring": "live_dependency_wiring",
            },
        )

    def composition_root(self) -> JarvisCompositionRoot:
        if self._composition_root is None:
            self._composition_root = JarvisCompositionRoot(
                config=self.bootstrap_config(),
                factories=self._factories,
                overrides=self._effective_overrides(),
            )

        return self._composition_root

    def validate(self) -> LiveDependencyWiringReport:
        try:
            root = self.composition_root()
            graph = root.validate_dependency_graph()

            if not graph.boot_allowed:
                return LiveDependencyWiringReport(
                    status=LiveDependencyWiringStatus.BLOCKED,
                    dependency_graph=graph,
                    bootstrap_result=None,
                    composition_snapshot=root.snapshot(),
                    created_at=utc_now(),
                    error="dependency graph blocked boot",
                    metadata=self._metadata(),
                )

            return LiveDependencyWiringReport(
                status=LiveDependencyWiringStatus.VALIDATED,
                dependency_graph=graph,
                bootstrap_result=None,
                composition_snapshot=root.snapshot(),
                created_at=utc_now(),
                error=None,
                metadata=self._metadata(),
            )

        except Exception as exc:
            return LiveDependencyWiringReport(
                status=LiveDependencyWiringStatus.FAILED,
                dependency_graph=None,
                bootstrap_result=None,
                composition_snapshot=None,
                created_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._metadata(),
            )

    def build_bootstrap(self) -> JarvisSystemBootstrap:
        return self.composition_root().build_bootstrap()

    def validate_bootstrap_ready(self) -> LiveDependencyWiringReport:
        validation = self.validate()
        if not validation.succeeded:
            return validation

        try:
            root = self.composition_root()
            self.build_bootstrap()

            return LiveDependencyWiringReport(
                status=LiveDependencyWiringStatus.BOOTSTRAP_READY,
                dependency_graph=validation.dependency_graph,
                bootstrap_result=None,
                composition_snapshot=root.snapshot(),
                created_at=utc_now(),
                error=None,
                metadata=self._metadata(),
            )

        except Exception as exc:
            return LiveDependencyWiringReport(
                status=LiveDependencyWiringStatus.FAILED,
                dependency_graph=validation.dependency_graph,
                bootstrap_result=None,
                composition_snapshot=validation.composition_snapshot,
                created_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._metadata(),
            )

    def run_dry_boot(self) -> LiveDependencyWiringReport:
        validation = self.validate_bootstrap_ready()
        if not validation.succeeded:
            return validation

        try:
            bootstrap = self.build_bootstrap()
            result = bootstrap.start()
            root = self.composition_root()

            if not result.succeeded:
                return LiveDependencyWiringReport(
                    status=LiveDependencyWiringStatus.FAILED,
                    dependency_graph=validation.dependency_graph,
                    bootstrap_result=result,
                    composition_snapshot=root.snapshot(),
                    created_at=utc_now(),
                    error=result.error or "dry boot failed",
                    metadata=self._metadata(),
                )

            return LiveDependencyWiringReport(
                status=LiveDependencyWiringStatus.DRY_RUN_PASSED,
                dependency_graph=validation.dependency_graph,
                bootstrap_result=result,
                composition_snapshot=root.snapshot(),
                created_at=utc_now(),
                error=None,
                metadata=self._metadata(),
            )

        except Exception as exc:
            return LiveDependencyWiringReport(
                status=LiveDependencyWiringStatus.FAILED,
                dependency_graph=validation.dependency_graph,
                bootstrap_result=None,
                composition_snapshot=validation.composition_snapshot,
                created_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._metadata(),
            )

    def _effective_overrides(self) -> tuple[JarvisCompositionOverride, ...]:
        overrides = list(self._overrides)

        if self._config.allow_degraded_presence:
            overrides.append(
                JarvisCompositionOverride(
                    component_id="presence",
                    can_degrade=True,
                    metadata={"source": "live_dependency_wiring"},
                )
            )

        if self._config.allow_degraded_orchestration:
            overrides.append(
                JarvisCompositionOverride(
                    component_id="orchestration",
                    can_degrade=True,
                    metadata={"source": "live_dependency_wiring"},
                )
            )

        return tuple(overrides)

    def _metadata(self) -> dict[str, object]:
        return {
            **self._config.metadata,
            "profile": self._config.profile.value,
            "dry_run": self._config.dry_run,
            "attach_conversation": self._config.attach_conversation,
            "attach_presence": self._config.attach_presence,
            "attach_orchestration": self._config.attach_orchestration,
        }