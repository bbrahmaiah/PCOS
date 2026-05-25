from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol

from pydantic import Field, field_validator, model_validator

from jarvis.tools.audit import (
    ActionAuditActor,
    ActionAuditEventKind,
    ActionAuditLog,
    ActionAuditOutcome,
)
from jarvis.tools.ids import new_action_result_id, utc_now
from jarvis.tools.models import ActionRisk, ActionStatus, ToolModel


class ToolMemoryEventKind(StrEnum):
    """
    Tool-memory event kind.

    These events describe what happened. They are not direct memory writes.
    """

    ACTION_REQUESTED = "action_requested"
    PLAN_PROPOSED = "plan_proposed"
    APPROVAL_RECORDED = "approval_recorded"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    ACTION_CANCELLED = "action_cancelled"
    ROLLBACK_COMPLETED = "rollback_completed"
    USER_PREFERENCE_LEARNED = "user_preference_learned"
    WORKFLOW_PATTERN_LEARNED = "workflow_pattern_learned"


class ToolMemoryDecision(StrEnum):
    """
    Memory integration decision.
    """

    STORE = "store"
    SKIP = "skip"
    REDACT_AND_STORE = "redact_and_store"
    BLOCK = "block"


class ToolMemoryReason(StrEnum):
    """
    Machine-readable memory integration reason.
    """

    SAFE_ACTION_SUMMARY = "safe_action_summary"
    SAFE_USER_PREFERENCE = "safe_user_preference"
    SAFE_WORKFLOW_PATTERN = "safe_workflow_pattern"
    LOW_VALUE_EVENT_SKIPPED = "low_value_event_skipped"
    SENSITIVE_DATA_REDACTED = "sensitive_data_redacted"
    SENSITIVE_DATA_BLOCKED = "sensitive_data_blocked"
    HIGH_RISK_EVENT_BLOCKED = "high_risk_event_blocked"
    MEMORY_GATEWAY_UNAVAILABLE = "memory_gateway_unavailable"
    MEMORY_WRITE_SUCCEEDED = "memory_write_succeeded"
    MEMORY_WRITE_FAILED = "memory_write_failed"


class ToolMemoryPolicyClass(StrEnum):
    """
    Policy classification for memory payloads.

    This classification must travel with every memory write proposal.
    """

    PUBLIC = "public"
    WORKSPACE = "workspace"
    USER_PRIVATE = "user_private"
    SENSITIVE_REDACTED = "sensitive_redacted"
    DO_NOT_STORE = "do_not_store"


class ToolMemoryImportance(StrEnum):
    """
    Importance band for memory write proposals.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolMemoryEvent(ToolModel):
    """
    Tool event that may become memory.

    This is an input to the integration layer. It is not yet memory.
    """

    event_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    kind: ToolMemoryEventKind
    summary: str
    risk: ActionRisk = ActionRisk.LOW
    status: ActionStatus | None = None
    user_visible: bool = True
    approved_by_user: bool = False
    source_runtime: str | None = None
    data: dict[str, object] = Field(default_factory=dict)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "action_id", "summary")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("source_runtime")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class ToolMemoryWriteProposal(ToolModel):
    """
    Sanitized memory write proposal.

    This is what can be sent to Memory Gateway.
    """

    proposal_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    event_id: str
    decision: ToolMemoryDecision
    reason: ToolMemoryReason
    policy_class: ToolMemoryPolicyClass
    importance: ToolMemoryImportance
    content: str
    source: str = "tool_memory_integration"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    tags: tuple[str, ...] = ()
    redacted_fields: tuple[str, ...] = ()
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("proposal_id", "action_id", "event_id", "content", "source")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _validate_storage_shape(self) -> ToolMemoryWriteProposal:
        if self.decision == ToolMemoryDecision.STORE:
            if self.policy_class == ToolMemoryPolicyClass.DO_NOT_STORE:
                raise ValueError("stored memory cannot use DO_NOT_STORE policy.")

        if self.decision == ToolMemoryDecision.BLOCK:
            if self.policy_class != ToolMemoryPolicyClass.DO_NOT_STORE:
                raise ValueError("blocked memory must use DO_NOT_STORE policy.")

        return self


class ToolMemoryWriteResult(ToolModel):
    """
    Result of submitting a proposal to Memory Gateway.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    proposal_id: str
    action_id: str
    event_id: str
    decision: ToolMemoryDecision
    reason: ToolMemoryReason
    success: bool
    stored: bool
    memory_id: str | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "proposal_id", "action_id", "event_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("memory_id")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


@dataclass(frozen=True, slots=True)
class ToolMemoryIntegrationConfig:
    """
    Tool memory integration configuration.
    """

    name: str = "tool_memory_integration"
    allow_memory_writes: bool = True
    require_user_visible_event: bool = True
    store_low_value_events: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class ToolMemoryIntegrationSnapshot:
    """
    Runtime diagnostics.
    """

    name: str
    event_count: int
    proposal_count: int
    write_count: int
    stored_count: int
    skipped_count: int
    blocked_count: int
    redacted_count: int
    last_decision: ToolMemoryDecision | None
    last_reason: ToolMemoryReason | None
    last_error: str | None


class MemoryGatewayWriter(Protocol):
    """
    Narrow protocol for Memory Gateway integration.

    The real Memory Gateway adapter should implement this protocol. Tool
    runtimes must never write directly to memory stores.
    """

    def write_tool_memory(
        self,
        *,
        content: str,
        source: str,
        confidence: float,
        policy_class: str,
        reason: str,
        tags: tuple[str, ...],
        metadata: dict[str, object],
    ) -> str:
        ...


class NullMemoryGatewayWriter:
    """
    Safe default writer.

    It intentionally does not store anything. This prevents accidental direct
    memory writes before the real Memory Gateway adapter is wired in.
    """

    def write_tool_memory(
        self,
        *,
        content: str,
        source: str,
        confidence: float,
        policy_class: str,
        reason: str,
        tags: tuple[str, ...],
        metadata: dict[str, object],
    ) -> str:
        del content, source, confidence, policy_class, reason, tags, metadata

        raise RuntimeError("memory gateway writer is not configured")


class ToolMemoryPolicy:
    """
    Conservative policy for turning tool events into memory proposals.
    """

    _SENSITIVE_KEYS = {
        "password",
        "passcode",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "otp",
        "card",
        "cvv",
        "pin",
        "private_key",
    }
    _SENSITIVE_TEXT = {
        "password",
        "secret",
        "api key",
        "token",
        "otp",
        "credit card",
        "debit card",
        "cvv",
        "private key",
    }

    def evaluate(
        self,
        event: ToolMemoryEvent,
        *,
        store_low_value_events: bool,
    ) -> ToolMemoryWriteProposal:
        redacted_data, redacted_fields = self._redact(event.data)
        sensitive_text = self._contains_sensitive_text(event.summary)

        if event.risk == ActionRisk.CRITICAL:
            return self._proposal(
                event=event,
                decision=ToolMemoryDecision.BLOCK,
                reason=ToolMemoryReason.HIGH_RISK_EVENT_BLOCKED,
                policy_class=ToolMemoryPolicyClass.DO_NOT_STORE,
                importance=ToolMemoryImportance.LOW,
                content="critical-risk tool event blocked from memory storage",
                redacted_fields=tuple(redacted_fields),
            )

        if sensitive_text and not redacted_fields:
            return self._proposal(
                event=event,
                decision=ToolMemoryDecision.BLOCK,
                reason=ToolMemoryReason.SENSITIVE_DATA_BLOCKED,
                policy_class=ToolMemoryPolicyClass.DO_NOT_STORE,
                importance=ToolMemoryImportance.LOW,
                content="sensitive tool event blocked from memory storage",
            )

        if self._is_low_value(event) and not store_low_value_events:
            return self._proposal(
                event=event,
                decision=ToolMemoryDecision.SKIP,
                reason=ToolMemoryReason.LOW_VALUE_EVENT_SKIPPED,
                policy_class=ToolMemoryPolicyClass.DO_NOT_STORE,
                importance=ToolMemoryImportance.LOW,
                content="low-value tool event skipped",
            )

        reason = self._reason_for(event)
        importance = self._importance_for(event)
        policy_class = self._policy_class_for(event)

        if redacted_fields:
            return self._proposal(
                event=event,
                decision=ToolMemoryDecision.REDACT_AND_STORE,
                reason=ToolMemoryReason.SENSITIVE_DATA_REDACTED,
                policy_class=ToolMemoryPolicyClass.SENSITIVE_REDACTED,
                importance=importance,
                content=self._content_for(event, redacted_data),
                redacted_fields=tuple(redacted_fields),
            )

        return self._proposal(
            event=event,
            decision=ToolMemoryDecision.STORE,
            reason=reason,
            policy_class=policy_class,
            importance=importance,
            content=self._content_for(event, redacted_data),
        )

    def _proposal(
        self,
        *,
        event: ToolMemoryEvent,
        decision: ToolMemoryDecision,
        reason: ToolMemoryReason,
        policy_class: ToolMemoryPolicyClass,
        importance: ToolMemoryImportance,
        content: str,
        redacted_fields: tuple[str, ...] = (),
    ) -> ToolMemoryWriteProposal:
        return ToolMemoryWriteProposal(
            action_id=event.action_id,
            event_id=event.event_id,
            decision=decision,
            reason=reason,
            policy_class=policy_class,
            importance=importance,
            content=content,
            confidence=self._confidence_for(event),
            tags=self._tags_for(event),
            redacted_fields=redacted_fields,
            metadata={
                "event_kind": event.kind.value,
                "risk": event.risk.value,
                "status": event.status.value if event.status else None,
                "source_runtime": event.source_runtime,
            },
        )

    @staticmethod
    def _is_low_value(event: ToolMemoryEvent) -> bool:
        return event.kind in {
            ToolMemoryEventKind.ACTION_REQUESTED,
            ToolMemoryEventKind.PLAN_PROPOSED,
        } and event.risk == ActionRisk.LOW

    @staticmethod
    def _reason_for(event: ToolMemoryEvent) -> ToolMemoryReason:
        if event.kind == ToolMemoryEventKind.USER_PREFERENCE_LEARNED:
            return ToolMemoryReason.SAFE_USER_PREFERENCE

        if event.kind == ToolMemoryEventKind.WORKFLOW_PATTERN_LEARNED:
            return ToolMemoryReason.SAFE_WORKFLOW_PATTERN

        return ToolMemoryReason.SAFE_ACTION_SUMMARY

    @staticmethod
    def _importance_for(event: ToolMemoryEvent) -> ToolMemoryImportance:
        if event.kind in {
            ToolMemoryEventKind.USER_PREFERENCE_LEARNED,
            ToolMemoryEventKind.WORKFLOW_PATTERN_LEARNED,
        }:
            return ToolMemoryImportance.HIGH

        if event.risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
            return ToolMemoryImportance.MEDIUM

        return ToolMemoryImportance.LOW

    @staticmethod
    def _policy_class_for(event: ToolMemoryEvent) -> ToolMemoryPolicyClass:
        if event.source_runtime in {"file_system_runtime", "ide_runtime"}:
            return ToolMemoryPolicyClass.WORKSPACE

        return ToolMemoryPolicyClass.USER_PRIVATE

    @staticmethod
    def _confidence_for(event: ToolMemoryEvent) -> float:
        if event.approved_by_user:
            return 0.95

        if event.status == ActionStatus.SUCCEEDED:
            return 0.85

        if event.status == ActionStatus.FAILED:
            return 0.7

        return 0.75

    @staticmethod
    def _tags_for(event: ToolMemoryEvent) -> tuple[str, ...]:
        tags = ["tool_action", event.kind.value]

        if event.source_runtime:
            tags.append(event.source_runtime)

        if event.status:
            tags.append(event.status.value)

        return tuple(tags)

    @classmethod
    def _content_for(
        cls,
        event: ToolMemoryEvent,
        data: dict[str, object],
    ) -> str:
        parts = [
            f"Tool event: {event.kind.value}",
            f"Action: {event.action_id}",
            f"Summary: {event.summary}",
            f"Risk: {event.risk.value}",
        ]

        if event.status is not None:
            parts.append(f"Status: {event.status.value}")

        if event.source_runtime is not None:
            parts.append(f"Runtime: {event.source_runtime}")

        if data:
            parts.append(f"Metadata: {cls._stable_data_summary(data)}")

        return "\n".join(parts)

    @staticmethod
    def _stable_data_summary(data: dict[str, object]) -> str:
        items = []

        for key in sorted(data):
            value = data[key]
            items.append(f"{key}={value}")

        return "; ".join(items)

    def _redact(self, data: dict[str, object]) -> tuple[dict[str, object], list[str]]:
        redacted: list[str] = []
        safe = self._redact_value(data, redacted, prefix="data")

        if not isinstance(safe, dict):
            return {}, redacted

        return safe, redacted

    def _redact_value(
        self,
        value: object,
        redacted: list[str],
        *,
        prefix: str,
    ) -> object:
        if isinstance(value, dict):
            output: dict[str, object] = {}

            for key, item in value.items():
                key_text = str(key)
                child_prefix = f"{prefix}.{key_text}"

                if self._is_sensitive_key(key_text):
                    output[key_text] = "[REDACTED]"
                    redacted.append(child_prefix)
                    continue

                output[key_text] = self._redact_value(
                    item,
                    redacted,
                    prefix=child_prefix,
                )

            return output

        if isinstance(value, list):
            return [
                self._redact_value(item, redacted, prefix=f"{prefix}[]")
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._redact_value(item, redacted, prefix=f"{prefix}[]")
                for item in value
            )

        return value

    def _is_sensitive_key(self, key: str) -> bool:
        normalized = key.casefold()

        return any(token in normalized for token in self._SENSITIVE_KEYS)

    def _contains_sensitive_text(self, value: str) -> bool:
        normalized = value.casefold()

        return any(token in normalized for token in self._SENSITIVE_TEXT)


class ToolMemoryIntegrationRuntime:
    """
    Safe bridge from Tool Runtime to Memory Gateway.

    Responsibilities:
    - classify tool events before memory
    - redact sensitive values
    - block unsafe memory writes
    - submit only sanitized proposals to Memory Gateway
    - audit memory write decisions

    Non-responsibilities:
    - no direct memory-store access
    - no vector DB writes
    - no bypass of Memory Gateway
    - no storing secrets
    """

    def __init__(
        self,
        *,
        config: ToolMemoryIntegrationConfig | None = None,
        policy: ToolMemoryPolicy | None = None,
        memory_gateway: MemoryGatewayWriter | None = None,
        audit_log: ActionAuditLog | None = None,
    ) -> None:
        self._config = config or ToolMemoryIntegrationConfig()
        self._config.validate()

        self._policy = policy or ToolMemoryPolicy()
        self._memory_gateway = memory_gateway or NullMemoryGatewayWriter()
        self._audit_log = audit_log
        self._lock = RLock()

        self._event_count = 0
        self._proposal_count = 0
        self._write_count = 0
        self._stored_count = 0
        self._skipped_count = 0
        self._blocked_count = 0
        self._redacted_count = 0
        self._last_decision: ToolMemoryDecision | None = None
        self._last_reason: ToolMemoryReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def propose(self, event: ToolMemoryEvent) -> ToolMemoryWriteProposal:
        """
        Convert a tool event into a memory write proposal.

        This does not write memory.
        """

        with self._lock:
            self._event_count += 1

        if self._config.require_user_visible_event and not event.user_visible:
            proposal = ToolMemoryWriteProposal(
                action_id=event.action_id,
                event_id=event.event_id,
                decision=ToolMemoryDecision.BLOCK,
                reason=ToolMemoryReason.SENSITIVE_DATA_BLOCKED,
                policy_class=ToolMemoryPolicyClass.DO_NOT_STORE,
                importance=ToolMemoryImportance.LOW,
                content="hidden tool event blocked from memory storage",
                confidence=0.0,
                tags=("tool_action", "blocked"),
            )
        else:
            proposal = self._policy.evaluate(
                event,
                store_low_value_events=self._config.store_low_value_events,
            )

        self._record_proposal(proposal)

        return proposal

    def write(self, proposal: ToolMemoryWriteProposal) -> ToolMemoryWriteResult:
        """
        Submit an approved sanitized proposal to Memory Gateway.
        """

        with self._lock:
            self._write_count += 1
            self._last_error = None

        if not self._config.allow_memory_writes:
            result = self._result(
                proposal=proposal,
                success=False,
                stored=False,
                reason=ToolMemoryReason.MEMORY_GATEWAY_UNAVAILABLE,
                message="tool memory writes are disabled",
            )
            self._record_result(result)

            return result

        if proposal.decision == ToolMemoryDecision.SKIP:
            result = self._result(
                proposal=proposal,
                success=True,
                stored=False,
                reason=proposal.reason,
                message="memory write skipped by policy",
            )
            self._record_result(result)

            return result

        if proposal.decision == ToolMemoryDecision.BLOCK:
            result = self._result(
                proposal=proposal,
                success=False,
                stored=False,
                reason=proposal.reason,
                message="memory write blocked by policy",
            )
            self._record_result(result)

            return result

        try:
            memory_id = self._memory_gateway.write_tool_memory(
                content=proposal.content,
                source=proposal.source,
                confidence=proposal.confidence,
                policy_class=proposal.policy_class.value,
                reason=proposal.reason.value,
                tags=proposal.tags,
                metadata={
                    **proposal.metadata,
                    "proposal_id": proposal.proposal_id,
                    "action_id": proposal.action_id,
                    "event_id": proposal.event_id,
                    "importance": proposal.importance.value,
                    "redacted_fields": proposal.redacted_fields,
                },
            )
            result = self._result(
                proposal=proposal,
                success=True,
                stored=True,
                reason=ToolMemoryReason.MEMORY_WRITE_SUCCEEDED,
                message="tool memory stored through Memory Gateway",
                memory_id=memory_id,
            )
            self._record_result(result)

            return result

        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"

            result = self._result(
                proposal=proposal,
                success=False,
                stored=False,
                reason=ToolMemoryReason.MEMORY_WRITE_FAILED,
                message=f"{type(exc).__name__}: {exc}",
            )
            self._record_result(result)

            return result

    def process(self, event: ToolMemoryEvent) -> ToolMemoryWriteResult:
        """
        Propose and write in one governed path.
        """

        proposal = self.propose(event)

        return self.write(proposal)

    def snapshot(self) -> ToolMemoryIntegrationSnapshot:
        with self._lock:
            return ToolMemoryIntegrationSnapshot(
                name=self.name,
                event_count=self._event_count,
                proposal_count=self._proposal_count,
                write_count=self._write_count,
                stored_count=self._stored_count,
                skipped_count=self._skipped_count,
                blocked_count=self._blocked_count,
                redacted_count=self._redacted_count,
                last_decision=self._last_decision,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        with self._lock:
            self._event_count = 0
            self._proposal_count = 0
            self._write_count = 0
            self._stored_count = 0
            self._skipped_count = 0
            self._blocked_count = 0
            self._redacted_count = 0
            self._last_decision = None
            self._last_reason = None
            self._last_error = None

    def _record_proposal(self, proposal: ToolMemoryWriteProposal) -> None:
        with self._lock:
            self._proposal_count += 1
            self._last_decision = proposal.decision
            self._last_reason = proposal.reason

            if proposal.decision == ToolMemoryDecision.SKIP:
                self._skipped_count += 1

            elif proposal.decision == ToolMemoryDecision.BLOCK:
                self._blocked_count += 1

            elif proposal.decision == ToolMemoryDecision.REDACT_AND_STORE:
                self._redacted_count += 1

        self._audit_proposal(proposal)

    def _record_result(self, result: ToolMemoryWriteResult) -> None:
        with self._lock:
            self._last_decision = result.decision
            self._last_reason = result.reason

            if result.stored:
                self._stored_count += 1

    @staticmethod
    def _result(
        *,
        proposal: ToolMemoryWriteProposal,
        success: bool,
        stored: bool,
        reason: ToolMemoryReason,
        message: str,
        memory_id: str | None = None,
    ) -> ToolMemoryWriteResult:
        return ToolMemoryWriteResult(
            proposal_id=proposal.proposal_id,
            action_id=proposal.action_id,
            event_id=proposal.event_id,
            decision=proposal.decision,
            reason=reason,
            success=success,
            stored=stored,
            memory_id=memory_id,
            message=message,
        )

    def _audit_proposal(self, proposal: ToolMemoryWriteProposal) -> None:
        if self._audit_log is None:
            return

        self._audit_log.record(
            action_id=proposal.action_id,
            event_kind=ActionAuditEventKind.MEMORY_CONTEXT_USED,
            actor=ActionAuditActor.MEMORY,
            outcome=(
                ActionAuditOutcome.BLOCKED
                if proposal.decision == ToolMemoryDecision.BLOCK
                else ActionAuditOutcome.INFO
            ),
            message="tool memory proposal evaluated",
            source_runtime=self.name,
            data={
                "proposal_id": proposal.proposal_id,
                "decision": proposal.decision.value,
                "reason": proposal.reason.value,
                "policy_class": proposal.policy_class.value,
                "importance": proposal.importance.value,
                "redacted_fields": proposal.redacted_fields,
            },
        )