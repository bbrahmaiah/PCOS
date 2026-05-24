from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.memory.models import (
    MemoryModel,
    MemoryPolicyClassification,
    MemoryRetention,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryWriteDecisionKind(StrEnum):
    """
    Final write-policy decision kind.
    """

    ALLOW = "allow"
    BLOCK = "block"
    DOWNGRADE = "downgrade"
    REQUIRE_CONFIRMATION = "require_confirmation"


class MemoryWriteRiskLevel(StrEnum):
    """
    Risk level for a memory write request.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryWritePolicyDecision(MemoryModel):
    """
    Decision returned by MemoryWritePolicy.

    The gateway uses this decision before any memory reaches the store.
    """

    request: MemoryWriteRequest
    effective_request: MemoryWriteRequest | None = None
    decision: MemoryWriteDecisionKind
    risk_level: MemoryWriteRiskLevel
    allowed: bool
    blocked: bool = False
    requires_confirmation: bool = False
    reason: str
    policy_classification: MemoryPolicyClassification
    decided_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class MemoryWritePolicyConfig:
    """
    Configuration for MemoryWritePolicy.

    Defaults are conservative:
    - sensitive writes are blocked unless explicitly allowed
    - pinned memories require explicit user source
    - very low confidence memories are blocked
    """

    name: str = "memory_write_policy"
    allow_sensitive_writes: bool = False
    allow_low_confidence_writes: bool = False
    min_write_confidence: float = 0.2
    require_user_explicit_for_pinned: bool = True
    downgrade_system_source_to_internal: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.min_write_confidence < 0.0 or self.min_write_confidence > 1.0:
            raise ValueError("min_write_confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class MemoryWritePolicySnapshot:
    """
    Observable diagnostics for MemoryWritePolicy.
    """

    name: str
    evaluated_count: int
    allowed_count: int
    blocked_count: int
    downgraded_count: int
    confirmation_required_count: int
    last_request_id: str | None
    last_decision: MemoryWriteDecisionKind | None
    last_risk_level: MemoryWriteRiskLevel | None
    last_error: str | None


class MemoryWritePolicy:
    """
    Governs memory write requests before they reach storage.

    Responsibilities:
    - classify write risk
    - block unsafe memory writes
    - downgrade safe-but-internal writes when needed
    - preserve explainable reasons
    - keep diagnostics

    Non-responsibilities:
    - no persistence
    - no retrieval
    - no embeddings
    - no cognition logic
    - no direct LLM calls
    """

    def __init__(
        self,
        *,
        config: MemoryWritePolicyConfig | None = None,
    ) -> None:
        self._config = config or MemoryWritePolicyConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.write_policy")

        self._evaluated_count = 0
        self._allowed_count = 0
        self._blocked_count = 0
        self._downgraded_count = 0
        self._confirmation_required_count = 0
        self._last_request_id: str | None = None
        self._last_decision: MemoryWriteDecisionKind | None = None
        self._last_risk_level: MemoryWriteRiskLevel | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def evaluate(self, request: MemoryWriteRequest) -> MemoryWritePolicyDecision:
        """
        Evaluate a memory write request before storage.
        """

        risk_level = self._risk_level(request)

        if self._blocks_sensitive_write(request):
            decision = MemoryWritePolicyDecision(
                request=request,
                effective_request=None,
                decision=MemoryWriteDecisionKind.BLOCK,
                risk_level=MemoryWriteRiskLevel.HIGH,
                allowed=False,
                blocked=True,
                reason="blocked sensitive memory write by gateway policy",
                policy_classification=MemoryPolicyClassification.BLOCKED,
                metadata={
                    "policy": self.name,
                    "sensitivity": request.sensitivity.value,
                },
            )
            self._record(decision)

            return decision

        if self._blocks_low_confidence_write(request):
            decision = MemoryWritePolicyDecision(
                request=request,
                effective_request=None,
                decision=MemoryWriteDecisionKind.BLOCK,
                risk_level=MemoryWriteRiskLevel.MEDIUM,
                allowed=False,
                blocked=True,
                reason="blocked low-confidence memory write by policy",
                policy_classification=MemoryPolicyClassification.BLOCKED,
                metadata={
                    "policy": self.name,
                    "confidence": request.confidence,
                    "min_write_confidence": self._config.min_write_confidence,
                },
            )
            self._record(decision)

            return decision

        if self._blocks_implicit_pinned_write(request):
            decision = MemoryWritePolicyDecision(
                request=request,
                effective_request=None,
                decision=MemoryWriteDecisionKind.BLOCK,
                risk_level=MemoryWriteRiskLevel.HIGH,
                allowed=False,
                blocked=True,
                reason="blocked pinned memory without explicit user source",
                policy_classification=MemoryPolicyClassification.BLOCKED,
                metadata={
                    "policy": self.name,
                    "retention": request.retention.value,
                    "source": request.source.value,
                },
            )
            self._record(decision)

            return decision

        effective_request = self._effective_request(request)

        if effective_request != request:
            decision = MemoryWritePolicyDecision(
                request=request,
                effective_request=effective_request,
                decision=MemoryWriteDecisionKind.DOWNGRADE,
                risk_level=risk_level,
                allowed=True,
                blocked=False,
                reason="memory write allowed with policy downgrade",
                policy_classification=MemoryPolicyClassification.ALLOWED,
                metadata={
                    "policy": self.name,
                    "original_sensitivity": request.sensitivity.value,
                    "effective_sensitivity": effective_request.sensitivity.value,
                },
            )
            self._record(decision)

            return decision

        decision = MemoryWritePolicyDecision(
            request=request,
            effective_request=request,
            decision=MemoryWriteDecisionKind.ALLOW,
            risk_level=risk_level,
            allowed=True,
            blocked=False,
            reason="memory write allowed by gateway policy",
            policy_classification=self._policy_classification(request),
            metadata={
                "policy": self.name,
                "sensitivity": request.sensitivity.value,
                "retention": request.retention.value,
            },
        )
        self._record(decision)

        return decision

    def snapshot(self) -> MemoryWritePolicySnapshot:
        """
        Return write-policy diagnostics.
        """

        with self._lock:
            return MemoryWritePolicySnapshot(
                name=self.name,
                evaluated_count=self._evaluated_count,
                allowed_count=self._allowed_count,
                blocked_count=self._blocked_count,
                downgraded_count=self._downgraded_count,
                confirmation_required_count=self._confirmation_required_count,
                last_request_id=self._last_request_id,
                last_decision=self._last_decision,
                last_risk_level=self._last_risk_level,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset policy counters.
        """

        with self._lock:
            self._evaluated_count = 0
            self._allowed_count = 0
            self._blocked_count = 0
            self._downgraded_count = 0
            self._confirmation_required_count = 0
            self._last_request_id = None
            self._last_decision = None
            self._last_risk_level = None
            self._last_error = None

        self._logger.info("memory_write_policy_reset", policy=self.name)

    def _blocks_sensitive_write(self, request: MemoryWriteRequest) -> bool:
        return (
            request.sensitivity == MemorySensitivity.SENSITIVE
            and not self._config.allow_sensitive_writes
        )

    def _blocks_low_confidence_write(self, request: MemoryWriteRequest) -> bool:
        return (
            request.confidence < self._config.min_write_confidence
            and not self._config.allow_low_confidence_writes
        )

    def _blocks_implicit_pinned_write(self, request: MemoryWriteRequest) -> bool:
        return (
            self._config.require_user_explicit_for_pinned
            and request.retention == MemoryRetention.PINNED
            and request.source != MemorySource.USER_EXPLICIT
        )

    def _effective_request(
        self,
        request: MemoryWriteRequest,
    ) -> MemoryWriteRequest:
        if (
            self._config.downgrade_system_source_to_internal
            and request.source == MemorySource.SYSTEM
            and request.sensitivity == MemorySensitivity.PRIVATE
        ):
            return request.model_copy(
                update={
                    "sensitivity": MemorySensitivity.INTERNAL,
                    "metadata": {
                        **request.metadata,
                        "write_policy_downgraded": True,
                        "downgrade_reason": "system source private memory downgraded",
                    },
                }
            )

        return request

    @staticmethod
    def _risk_level(request: MemoryWriteRequest) -> MemoryWriteRiskLevel:
        if request.sensitivity == MemorySensitivity.SENSITIVE:
            return MemoryWriteRiskLevel.HIGH

        if request.retention == MemoryRetention.PINNED:
            return MemoryWriteRiskLevel.HIGH

        if request.sensitivity == MemorySensitivity.PRIVATE:
            return MemoryWriteRiskLevel.MEDIUM

        return MemoryWriteRiskLevel.LOW

    @staticmethod
    def _policy_classification(
        request: MemoryWriteRequest,
    ) -> MemoryPolicyClassification:
        if request.sensitivity == MemorySensitivity.SENSITIVE:
            return MemoryPolicyClassification.RESTRICTED

        return MemoryPolicyClassification.ALLOWED

    def _record(self, decision: MemoryWritePolicyDecision) -> None:
        with self._lock:
            self._evaluated_count += 1
            self._last_request_id = decision.request.request_id
            self._last_decision = decision.decision
            self._last_risk_level = decision.risk_level
            self._last_error = None if decision.allowed else decision.reason

            if decision.blocked:
                self._blocked_count += 1

            elif decision.decision == MemoryWriteDecisionKind.DOWNGRADE:
                self._allowed_count += 1
                self._downgraded_count += 1

            elif decision.requires_confirmation:
                self._confirmation_required_count += 1

            else:
                self._allowed_count += 1

        self._logger.info(
            "memory_write_policy_evaluated",
            policy=self.name,
            request_id=decision.request.request_id,
            decision=decision.decision.value,
            risk_level=decision.risk_level.value,
            allowed=decision.allowed,
            blocked=decision.blocked,
            reason=decision.reason,
        )