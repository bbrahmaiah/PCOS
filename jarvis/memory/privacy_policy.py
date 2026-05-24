from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator

from jarvis.memory.models import (
    MemoryModel,
    MemoryPolicyClassification,
    MemoryRecord,
    MemorySensitivity,
    MemoryWriteRequest,
    utc_now,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryPrivacyRiskLevel(StrEnum):
    """
    Privacy risk level for memory content.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryPrivacyDecisionKind(StrEnum):
    """
    Privacy policy decision kind.
    """

    ALLOW = "allow"
    RESTRICT = "restrict"
    REDACT = "redact"
    BLOCK = "block"


class MemoryPrivacySubject(StrEnum):
    """
    The object evaluated by the privacy policy.
    """

    WRITE_REQUEST = "write_request"
    RECORD = "record"


class MemoryPrivacyPolicyDecision(MemoryModel):
    """
    Privacy/sensitivity decision for a memory item.

    This is deliberately explainable. Memory privacy decisions must never be
    hidden or implicit.
    """

    subject: MemoryPrivacySubject
    text: str
    decision: MemoryPrivacyDecisionKind
    risk_level: MemoryPrivacyRiskLevel
    sensitivity: MemorySensitivity
    policy_classification: MemoryPolicyClassification
    allowed: bool
    blocked: bool = False
    redacted_text: str | None = None
    reasons: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    decided_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("text cannot be empty.")

        return cleaned

    @field_validator("reasons")
    @classmethod
    def _reasons_required(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(reason.strip() for reason in value if reason.strip())

        if not cleaned:
            raise ValueError("privacy decision reasons cannot be empty.")

        return cleaned

    @field_validator("matched_terms")
    @classmethod
    def _clean_matched_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(term.strip().casefold() for term in value if term.strip())

        return tuple(dict.fromkeys(cleaned))


@dataclass(frozen=True, slots=True)
class MemoryPrivacyPolicyConfig:
    """
    Configuration for MemoryPrivacyPolicy.

    Defaults are conservative but not destructive:
    - high-risk secret-like content is restricted
    - critical content is blocked
    - optional redaction can be enabled later
    """

    name: str = "memory_privacy_policy"
    block_critical_content: bool = True
    redact_high_risk_content: bool = False
    secret_terms: tuple[str, ...] = (
        "password",
        "passcode",
        "otp",
        "one time password",
        "api key",
        "secret key",
        "private key",
        "access token",
        "refresh token",
        "credit card",
        "debit card",
        "cvv",
        "pin number",
        "bank account",
    )
    critical_terms: tuple[str, ...] = (
        "seed phrase",
        "recovery phrase",
        "private key",
        "ssh private key",
        "master password",
    )
    private_terms: tuple[str, ...] = (
        "address",
        "phone number",
        "email",
        "personal",
        "family",
        "friend",
        "birthday",
        "location",
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        self._validate_terms("secret_terms", self.secret_terms)
        self._validate_terms("critical_terms", self.critical_terms)
        self._validate_terms("private_terms", self.private_terms)

    @staticmethod
    def _validate_terms(name: str, terms: tuple[str, ...]) -> None:
        for term in terms:
            if not term.strip():
                raise ValueError(f"{name} cannot contain empty terms.")


@dataclass(frozen=True, slots=True)
class MemoryPrivacyPolicySnapshot:
    """
    Observable diagnostics for MemoryPrivacyPolicy.
    """

    name: str
    evaluated_count: int
    allowed_count: int
    restricted_count: int
    redacted_count: int
    blocked_count: int
    last_decision: MemoryPrivacyDecisionKind | None
    last_risk_level: MemoryPrivacyRiskLevel | None
    last_error: str | None


class MemoryPrivacyPolicy:
    """
    Classifies privacy and sensitivity risk for memory content.

    Responsibilities:
    - detect secret-like and private content patterns
    - assign risk levels
    - assign policy classification
    - optionally redact high-risk content
    - block critical content when configured
    - produce explainable privacy decisions
    - keep diagnostics

    Non-responsibilities:
    - no storage
    - no retrieval ranking
    - no gateway routing
    - no LLM calls
    - no irreversible deletion
    """

    def __init__(
        self,
        *,
        config: MemoryPrivacyPolicyConfig | None = None,
    ) -> None:
        self._config = config or MemoryPrivacyPolicyConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.privacy_policy")

        self._evaluated_count = 0
        self._allowed_count = 0
        self._restricted_count = 0
        self._redacted_count = 0
        self._blocked_count = 0
        self._last_decision: MemoryPrivacyDecisionKind | None = None
        self._last_risk_level: MemoryPrivacyRiskLevel | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def evaluate_write_request(
        self,
        request: MemoryWriteRequest,
    ) -> MemoryPrivacyPolicyDecision:
        """
        Evaluate a memory write request.
        """

        decision = self._evaluate(
            text=request.text,
            declared_sensitivity=request.sensitivity,
            subject=MemoryPrivacySubject.WRITE_REQUEST,
            metadata={
                "request_id": request.request_id,
                "source": request.source.value,
                "kind": request.kind.value,
            },
        )
        self._record(decision)

        return decision

    def evaluate_record(
        self,
        record: MemoryRecord,
    ) -> MemoryPrivacyPolicyDecision:
        """
        Evaluate an existing memory record.
        """

        decision = self._evaluate(
            text=record.text,
            declared_sensitivity=record.sensitivity,
            subject=MemoryPrivacySubject.RECORD,
            metadata={
                "memory_id": record.memory_id,
                "source": record.source.value,
                "kind": record.kind.value,
            },
        )
        self._record(decision)

        return decision

    def snapshot(self) -> MemoryPrivacyPolicySnapshot:
        """
        Return privacy-policy diagnostics.
        """

        with self._lock:
            return MemoryPrivacyPolicySnapshot(
                name=self.name,
                evaluated_count=self._evaluated_count,
                allowed_count=self._allowed_count,
                restricted_count=self._restricted_count,
                redacted_count=self._redacted_count,
                blocked_count=self._blocked_count,
                last_decision=self._last_decision,
                last_risk_level=self._last_risk_level,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset privacy-policy counters.
        """

        with self._lock:
            self._evaluated_count = 0
            self._allowed_count = 0
            self._restricted_count = 0
            self._redacted_count = 0
            self._blocked_count = 0
            self._last_decision = None
            self._last_risk_level = None
            self._last_error = None

        self._logger.info("memory_privacy_policy_reset", policy=self.name)

    def _evaluate(
        self,
        *,
        text: str,
        declared_sensitivity: MemorySensitivity,
        subject: MemoryPrivacySubject,
        metadata: dict[str, object],
    ) -> MemoryPrivacyPolicyDecision:
        normalized_text = self._normalize(text)

        critical_matches = self._matched_terms(
            normalized_text,
            self._config.critical_terms,
        )
        secret_matches = self._matched_terms(
            normalized_text,
            self._config.secret_terms,
        )
        private_matches = self._matched_terms(
            normalized_text,
            self._config.private_terms,
        )

        matched_terms = tuple(
            dict.fromkeys(
                (
                    *critical_matches,
                    *secret_matches,
                    *private_matches,
                )
            )
        )

        if critical_matches and self._config.block_critical_content:
            return MemoryPrivacyPolicyDecision(
                subject=subject,
                text=text,
                decision=MemoryPrivacyDecisionKind.BLOCK,
                risk_level=MemoryPrivacyRiskLevel.CRITICAL,
                sensitivity=MemorySensitivity.SENSITIVE,
                policy_classification=MemoryPolicyClassification.BLOCKED,
                allowed=False,
                blocked=True,
                reasons=(
                    "critical secret-like content detected",
                    "blocked by privacy policy",
                ),
                matched_terms=matched_terms,
                metadata={
                    **metadata,
                    "policy": self.name,
                },
            )

        if secret_matches:
            if self._config.redact_high_risk_content:
                redacted_text = self._redact(text, secret_matches)

                return MemoryPrivacyPolicyDecision(
                    subject=subject,
                    text=text,
                    decision=MemoryPrivacyDecisionKind.REDACT,
                    risk_level=MemoryPrivacyRiskLevel.HIGH,
                    sensitivity=MemorySensitivity.SENSITIVE,
                    policy_classification=MemoryPolicyClassification.REDACTED,
                    allowed=True,
                    blocked=False,
                    redacted_text=redacted_text,
                    reasons=(
                        "secret-like content detected",
                        "high-risk content redacted by privacy policy",
                    ),
                    matched_terms=matched_terms,
                    metadata={
                        **metadata,
                        "policy": self.name,
                    },
                )

            return MemoryPrivacyPolicyDecision(
                subject=subject,
                text=text,
                decision=MemoryPrivacyDecisionKind.RESTRICT,
                risk_level=MemoryPrivacyRiskLevel.HIGH,
                sensitivity=MemorySensitivity.SENSITIVE,
                policy_classification=MemoryPolicyClassification.RESTRICTED,
                allowed=True,
                blocked=False,
                reasons=(
                    "secret-like content detected",
                    "restricted by privacy policy",
                ),
                matched_terms=matched_terms,
                metadata={
                    **metadata,
                    "policy": self.name,
                },
            )

        if declared_sensitivity == MemorySensitivity.SENSITIVE:
            return MemoryPrivacyPolicyDecision(
                subject=subject,
                text=text,
                decision=MemoryPrivacyDecisionKind.RESTRICT,
                risk_level=MemoryPrivacyRiskLevel.HIGH,
                sensitivity=MemorySensitivity.SENSITIVE,
                policy_classification=MemoryPolicyClassification.RESTRICTED,
                allowed=True,
                blocked=False,
                reasons=(
                    "memory was declared sensitive",
                    "restricted by privacy policy",
                ),
                matched_terms=matched_terms,
                metadata={
                    **metadata,
                    "policy": self.name,
                },
            )

        if private_matches or declared_sensitivity == MemorySensitivity.PRIVATE:
            return MemoryPrivacyPolicyDecision(
                subject=subject,
                text=text,
                decision=MemoryPrivacyDecisionKind.ALLOW,
                risk_level=MemoryPrivacyRiskLevel.MEDIUM,
                sensitivity=MemorySensitivity.PRIVATE,
                policy_classification=MemoryPolicyClassification.ALLOWED,
                allowed=True,
                blocked=False,
                reasons=(
                    "private memory allowed with normal gateway controls",
                ),
                matched_terms=matched_terms,
                metadata={
                    **metadata,
                    "policy": self.name,
                },
            )

        return MemoryPrivacyPolicyDecision(
            subject=subject,
            text=text,
            decision=MemoryPrivacyDecisionKind.ALLOW,
            risk_level=MemoryPrivacyRiskLevel.LOW,
            sensitivity=declared_sensitivity,
            policy_classification=MemoryPolicyClassification.ALLOWED,
            allowed=True,
            blocked=False,
            reasons=(
                "no high-risk privacy terms detected",
            ),
            matched_terms=matched_terms,
            metadata={
                **metadata,
                "policy": self.name,
            },
        )

    def _record(self, decision: MemoryPrivacyPolicyDecision) -> None:
        with self._lock:
            self._evaluated_count += 1
            self._last_decision = decision.decision
            self._last_risk_level = decision.risk_level
            self._last_error = None if decision.allowed else "; ".join(
                decision.reasons
            )

            if decision.decision == MemoryPrivacyDecisionKind.BLOCK:
                self._blocked_count += 1

            elif decision.decision == MemoryPrivacyDecisionKind.RESTRICT:
                self._restricted_count += 1

            elif decision.decision == MemoryPrivacyDecisionKind.REDACT:
                self._redacted_count += 1
                self._allowed_count += 1

            else:
                self._allowed_count += 1

        self._logger.info(
            "memory_privacy_policy_evaluated",
            policy=self.name,
            subject=decision.subject.value,
            decision=decision.decision.value,
            risk_level=decision.risk_level.value,
            allowed=decision.allowed,
            blocked=decision.blocked,
            policy_classification=decision.policy_classification.value,
        )

    @staticmethod
    def _matched_terms(
        normalized_text: str,
        terms: tuple[str, ...],
    ) -> tuple[str, ...]:
        matches = tuple(
            term.strip().casefold()
            for term in terms
            if term.strip() and term.strip().casefold() in normalized_text
        )

        return tuple(dict.fromkeys(matches))

    @staticmethod
    def _redact(text: str, terms: tuple[str, ...]) -> str:
        redacted = text

        for term in sorted(terms, key=len, reverse=True):
            redacted = redacted.replace(term, "[REDACTED]")
            redacted = redacted.replace(term.title(), "[REDACTED]")
            redacted = redacted.replace(term.upper(), "[REDACTED]")

        return redacted

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())