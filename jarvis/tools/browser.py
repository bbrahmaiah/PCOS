from __future__ import annotations

import time
import webbrowser
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol
from urllib.parse import quote_plus, urlparse

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_action_id, new_action_result_id, utc_now
from jarvis.tools.models import (
    ActionKind,
    ActionPlan,
    ActionRisk,
    ActionScope,
    ActionStatus,
    ActionStep,
    PermissionDecision,
    ToolCapability,
    ToolModel,
)
from jarvis.tools.registry import (
    ToolAvailability,
    ToolDescriptor,
    ToolHealth,
    ToolRegistry,
)
from jarvis.tools.validation import (
    ActionValidationDecision,
    ActionValidationResult,
    ActionValidator,
    ActionValidatorConfig,
)


class BrowserActionKind(StrEnum):
    """
    Supported governed browser action kinds.

    Step 7 intentionally starts with visible, low-control browser operations.
    """

    OPEN_URL = "open_url"
    SEARCH_WEB = "search_web"
    READ_PAGE_TITLE = "read_page_title"
    SUMMARIZE_PAGE = "summarize_page"
    DOWNLOAD_FILE = "download_file"
    FILL_FORM = "fill_form"


class BrowserActionDecision(StrEnum):
    """
    Browser runtime decision.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_APPROVAL = "require_approval"


class BrowserActionReason(StrEnum):
    """
    Machine-readable browser policy/runtime reason.
    """

    SAFE_OPEN_ALLOWED = "safe_open_allowed"
    SAFE_SEARCH_ALLOWED = "safe_search_allowed"
    READ_ONLY_ALLOWED = "read_only_allowed"
    DOWNLOAD_REQUIRES_APPROVAL = "download_requires_approval"
    FORM_FILL_REQUIRES_APPROVAL = "form_fill_requires_approval"
    PAYMENT_BLOCKED = "payment_blocked"
    PASSWORD_ENTRY_BLOCKED = "password_entry_blocked"
    ACCOUNT_CHANGE_BLOCKED = "account_change_blocked"
    SENSITIVE_FORM_BLOCKED = "sensitive_form_blocked"
    UNKNOWN_DOWNLOAD_BLOCKED = "unknown_download_blocked"
    UNSAFE_URL_BLOCKED = "unsafe_url_blocked"
    UNSUPPORTED_SCHEME_BLOCKED = "unsupported_scheme_blocked"
    URL_REQUIRED = "url_required"
    QUERY_REQUIRED = "query_required"
    APPROVAL_MISSING = "approval_missing"
    CONFIRMATION_MISSING = "confirmation_missing"
    VALIDATION_BLOCKED = "validation_blocked"
    ACTION_SUCCEEDED = "action_succeeded"
    ACTION_FAILED = "action_failed"


class BrowserActionRequest(ToolModel):
    """
    Typed request for a governed browser action.

    Browser runtime is not hidden automation. It produces visible, policy-gated
    actions and blocks sensitive operations by default.
    """

    action_id: str = Field(default_factory=new_action_id)
    kind: BrowserActionKind
    url: str | None = None
    query: str | None = None
    page_title: str | None = None
    page_text: str | None = None
    form_fields: dict[str, str] = Field(default_factory=dict)
    download_filename: str | None = None
    confirmed: bool = False
    approved: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("action_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("action_id cannot be empty.")

        return cleaned

    @field_validator("url", "query", "page_title", "page_text", "download_filename")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def _validate_payload(self) -> BrowserActionRequest:
        if self.kind in {
            BrowserActionKind.OPEN_URL,
            BrowserActionKind.READ_PAGE_TITLE,
            BrowserActionKind.SUMMARIZE_PAGE,
            BrowserActionKind.DOWNLOAD_FILE,
            BrowserActionKind.FILL_FORM,
        } and self.url is None:
            raise ValueError("url is required for this browser action.")

        if self.kind == BrowserActionKind.SEARCH_WEB and self.query is None:
            raise ValueError("query is required for web search.")

        if self.kind == BrowserActionKind.FILL_FORM and not self.form_fields:
            raise ValueError("form_fields are required for form fill.")

        return self


class BrowserActionPolicyResult(ToolModel):
    """
    Browser policy result.
    """

    decision: BrowserActionDecision
    permission_decision: PermissionDecision
    reason: BrowserActionReason
    risk: ActionRisk
    explanation: str
    sanitized_url: str | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("explanation")
    @classmethod
    def _explanation_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("explanation cannot be empty.")

        return cleaned

    @property
    def allowed(self) -> bool:
        return self.decision == BrowserActionDecision.ALLOW


class BrowserActionResult(ToolModel):
    """
    Observable result of a governed browser action.
    """

    result_id: str = Field(default_factory=new_action_result_id)
    action_id: str
    kind: BrowserActionKind
    status: ActionStatus
    success: bool
    decision: BrowserActionDecision
    reason: BrowserActionReason
    url: str | None = None
    title: str | None = None
    summary: str | None = None
    output: str = ""
    policy_result: BrowserActionPolicyResult
    validation_result: ActionValidationResult | None = None
    started_at: object = Field(default_factory=utc_now)
    completed_at: object = Field(default_factory=utc_now)
    duration_ms: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("result_id", "action_id", "output")
    @classmethod
    def _required_text_fields(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class BrowserRuntimeConfig:
    """
    Configuration for BrowserRuntime.
    """

    name: str = "browser_runtime"
    search_base_url: str = "https://www.google.com/search?q="
    register_default_browser_tool: bool = True
    visible_actions_only: bool = True
    max_summary_chars: int = 1_200

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if not self.search_base_url.strip():
            raise ValueError("search_base_url cannot be empty.")

        if self.max_summary_chars <= 0:
            raise ValueError("max_summary_chars must be positive.")


@dataclass(frozen=True, slots=True)
class BrowserRuntimeSnapshot:
    """
    Observable diagnostics for BrowserRuntime.
    """

    name: str
    action_count: int
    success_count: int
    blocked_count: int
    failed_count: int
    approval_required_count: int
    confirmation_required_count: int
    last_status: ActionStatus | None
    last_reason: BrowserActionReason | None
    last_error: str | None


class BrowserLauncher(Protocol):
    """
    Browser launcher protocol.

    Tests can inject a fake launcher. Production uses WebBrowserLauncher.
    """

    def open(self, url: str) -> bool:
        ...


class WebBrowserLauncher:
    """
    Visible browser launcher using the system browser.

    This intentionally opens a real visible browser tab. It must never become
    hidden browser automation.
    """

    def open(self, url: str) -> bool:
        return webbrowser.open(url, new=2, autoraise=True)


class BrowserPolicy:
    """
    Conservative browser policy.

    It allows visible open/search/read-only actions. It gates downloads and
    form fill behind approval. It blocks payments, passwords, account changes,
    and sensitive forms.
    """

    _PAYMENT_WORDS = (
        "payment",
        "checkout",
        "pay",
        "billing",
        "card",
        "credit",
        "debit",
        "upi",
        "bank",
    )
    _PASSWORD_WORDS = (
        "password",
        "passcode",
        "otp",
        "token",
        "secret",
        "api_key",
        "apikey",
    )
    _ACCOUNT_CHANGE_WORDS = (
        "delete account",
        "close account",
        "change email",
        "change password",
        "security settings",
        "2fa",
        "mfa",
    )
    _SENSITIVE_FORM_WORDS = (
        "ssn",
        "aadhaar",
        "passport",
        "pan",
        "medical",
        "health",
        "tax",
        "salary",
    )
    _BLOCKED_SCHEMES = {
        "file",
        "javascript",
        "data",
        "vbscript",
        "ftp",
    }
    _ALLOWED_SCHEMES = {
        "http",
        "https",
    }

    def evaluate(self, request: BrowserActionRequest) -> BrowserActionPolicyResult:
        sensitive_reason = self._sensitive_reason(request)

        if sensitive_reason is not None:
            return self._deny(
                reason=sensitive_reason,
                explanation="sensitive browser operation is blocked",
            )

        if request.kind == BrowserActionKind.SEARCH_WEB:
            return self._search_decision(request)

        if request.url is None:
            return self._deny(
                reason=BrowserActionReason.URL_REQUIRED,
                explanation="url is required for browser action",
            )

        safe_url, reason = self._sanitize_url(request.url)

        if safe_url is None:
            return self._deny(
                reason=reason,
                explanation="browser URL is not safe",
            )

        if request.kind == BrowserActionKind.OPEN_URL:
            return self._allow(
                reason=BrowserActionReason.SAFE_OPEN_ALLOWED,
                explanation="visible URL open is allowed",
                url=safe_url,
            )

        if request.kind in {
            BrowserActionKind.READ_PAGE_TITLE,
            BrowserActionKind.SUMMARIZE_PAGE,
        }:
            return self._allow(
                reason=BrowserActionReason.READ_ONLY_ALLOWED,
                explanation="read-only browser action is allowed",
                url=safe_url,
            )

        if request.kind == BrowserActionKind.DOWNLOAD_FILE:
            if not request.approved:
                return self._approval_required(
                    reason=BrowserActionReason.DOWNLOAD_REQUIRES_APPROVAL,
                    explanation="downloads require explicit approval",
                    url=safe_url,
                )

            return self._allow(
                reason=BrowserActionReason.DOWNLOAD_REQUIRES_APPROVAL,
                explanation="approved visible download action is allowed",
                url=safe_url,
                risk=ActionRisk.HIGH,
                permission=PermissionDecision.REQUIRE_APPROVAL,
            )

        if request.kind == BrowserActionKind.FILL_FORM:
            if not request.approved:
                return self._approval_required(
                    reason=BrowserActionReason.FORM_FILL_REQUIRES_APPROVAL,
                    explanation="form fill requires explicit approval",
                    url=safe_url,
                )

            return self._allow(
                reason=BrowserActionReason.FORM_FILL_REQUIRES_APPROVAL,
                explanation="approved non-sensitive form fill is allowed",
                url=safe_url,
                risk=ActionRisk.HIGH,
                permission=PermissionDecision.REQUIRE_APPROVAL,
            )

        return self._deny(
            reason=BrowserActionReason.ACTION_FAILED,
            explanation="unsupported browser action",
        )

    def _search_decision(
        self,
        request: BrowserActionRequest,
    ) -> BrowserActionPolicyResult:
        query = request.query or ""

        if not query.strip():
            return self._deny(
                reason=BrowserActionReason.QUERY_REQUIRED,
                explanation="query is required for web search",
            )

        if self._contains_any(query, self._PAYMENT_WORDS):
            return self._deny(
                reason=BrowserActionReason.PAYMENT_BLOCKED,
                explanation="payment-related web search action is blocked",
            )

        safe_url = "https://www.google.com/search?q=" + quote_plus(query)

        return self._allow(
            reason=BrowserActionReason.SAFE_SEARCH_ALLOWED,
            explanation="visible web search is allowed",
            url=safe_url,
        )

    def _sanitize_url(
        self,
        url: str,
    ) -> tuple[str | None, BrowserActionReason]:
        parsed = urlparse(url.strip())

        if not parsed.scheme:
            parsed = urlparse("https://" + url.strip())

        if parsed.scheme.casefold() in self._BLOCKED_SCHEMES:
            return None, BrowserActionReason.UNSUPPORTED_SCHEME_BLOCKED

        if parsed.scheme.casefold() not in self._ALLOWED_SCHEMES:
            return None, BrowserActionReason.UNSUPPORTED_SCHEME_BLOCKED

        if not parsed.netloc:
            return None, BrowserActionReason.UNSAFE_URL_BLOCKED

        return parsed.geturl(), BrowserActionReason.SAFE_OPEN_ALLOWED

    def _sensitive_reason(
        self,
        request: BrowserActionRequest,
    ) -> BrowserActionReason | None:
        haystack = " ".join(
            [
                request.url or "",
                request.query or "",
                request.page_title or "",
                request.page_text or "",
                request.download_filename or "",
                " ".join(request.form_fields.keys()),
                " ".join(request.form_fields.values()),
            ]
        ).casefold()

        if self._contains_any(haystack, self._PASSWORD_WORDS):
            return BrowserActionReason.PASSWORD_ENTRY_BLOCKED

        if self._contains_any(haystack, self._PAYMENT_WORDS):
            return BrowserActionReason.PAYMENT_BLOCKED

        if self._contains_any(haystack, self._ACCOUNT_CHANGE_WORDS):
            return BrowserActionReason.ACCOUNT_CHANGE_BLOCKED

        if self._contains_any(haystack, self._SENSITIVE_FORM_WORDS):
            return BrowserActionReason.SENSITIVE_FORM_BLOCKED

        return None

    @staticmethod
    def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
        normalized = value.casefold()

        return any(needle in normalized for needle in needles)

    @staticmethod
    def _allow(
        *,
        reason: BrowserActionReason,
        explanation: str,
        url: str | None = None,
        risk: ActionRisk = ActionRisk.LOW,
        permission: PermissionDecision = PermissionDecision.ALLOW,
    ) -> BrowserActionPolicyResult:
        return BrowserActionPolicyResult(
            decision=BrowserActionDecision.ALLOW,
            permission_decision=permission,
            reason=reason,
            risk=risk,
            explanation=explanation,
            sanitized_url=url,
        )

    @staticmethod
    def _deny(
        *,
        reason: BrowserActionReason,
        explanation: str,
    ) -> BrowserActionPolicyResult:
        return BrowserActionPolicyResult(
            decision=BrowserActionDecision.DENY,
            permission_decision=PermissionDecision.DENY,
            reason=reason,
            risk=ActionRisk.CRITICAL,
            explanation=explanation,
        )

    @staticmethod
    def _approval_required(
        *,
        reason: BrowserActionReason,
        explanation: str,
        url: str,
    ) -> BrowserActionPolicyResult:
        return BrowserActionPolicyResult(
            decision=BrowserActionDecision.REQUIRE_APPROVAL,
            permission_decision=PermissionDecision.REQUIRE_APPROVAL,
            reason=reason,
            risk=ActionRisk.HIGH,
            explanation=explanation,
            sanitized_url=url,
        )


class BrowserRuntime:
    """
    Governed browser runtime.

    Responsibilities:
    - visible URL open/search actions
    - read-only title/summary contracts
    - approval-gated download/form-fill actions
    - sensitive action blocking
    - typed action plan creation
    - validation before browser interaction
    - observable audit-ready results

    Non-responsibilities:
    - no hidden browser automation
    - no password entry
    - no payments
    - no account changes
    - no sensitive form submission
    """

    def __init__(
        self,
        *,
        config: BrowserRuntimeConfig | None = None,
        registry: ToolRegistry | None = None,
        policy: BrowserPolicy | None = None,
        validator: ActionValidator | None = None,
        launcher: BrowserLauncher | None = None,
    ) -> None:
        self._config = config or BrowserRuntimeConfig()
        self._config.validate()

        self._registry = registry or ToolRegistry()
        self._policy = policy or BrowserPolicy()
        self._launcher: BrowserLauncher = launcher or WebBrowserLauncher()

        if self._config.register_default_browser_tool:
            self._register_default_browser_tool()

        self._validator = validator or ActionValidator(
            config=ActionValidatorConfig(require_policy_evaluation=False),
            registry=self._registry,
        )
        self._lock = RLock()

        self._action_count = 0
        self._success_count = 0
        self._blocked_count = 0
        self._failed_count = 0
        self._approval_required_count = 0
        self._confirmation_required_count = 0
        self._last_status: ActionStatus | None = None
        self._last_reason: BrowserActionReason | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def execute(self, request: BrowserActionRequest) -> BrowserActionResult:
        """
        Execute one governed visible browser action.
        """

        with self._lock:
            self._action_count += 1
            self._last_error = None

        started = utc_now()
        monotonic_start = time.monotonic()

        try:
            policy_result = self._policy.evaluate(request)

            if policy_result.decision != BrowserActionDecision.ALLOW:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
                    started_at=started,
                    monotonic_start=monotonic_start,
                )
                self._record(result)

                return result

            plan = self._build_plan(
                request=request,
                policy_result=policy_result,
            )
            validation = self._validator.validate_plan(plan)

            if validation.decision == ActionValidationDecision.BLOCK:
                result = self._blocked_result(
                    request=request,
                    policy_result=policy_result,
                    started_at=started,
                    monotonic_start=monotonic_start,
                    validation_result=validation,
                    reason=BrowserActionReason.VALIDATION_BLOCKED,
                )
                self._record(result)

                return result

            result = self._execute_allowed(
                request=request,
                policy_result=policy_result,
                validation_result=validation,
                started_at=started,
                monotonic_start=monotonic_start,
            )
            self._record(result)

            return result

        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            policy_result = BrowserActionPolicyResult(
                decision=BrowserActionDecision.DENY,
                permission_decision=PermissionDecision.DENY,
                reason=BrowserActionReason.ACTION_FAILED,
                risk=ActionRisk.HIGH,
                explanation=f"{type(exc).__name__}: {exc}",
            )
            result = self._failed_result(
                request=request,
                policy_result=policy_result,
                started_at=started,
                monotonic_start=monotonic_start,
                output=f"{type(exc).__name__}: {exc}",
            )
            self._record(result)

            return result

    def snapshot(self) -> BrowserRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        with self._lock:
            return BrowserRuntimeSnapshot(
                name=self.name,
                action_count=self._action_count,
                success_count=self._success_count,
                blocked_count=self._blocked_count,
                failed_count=self._failed_count,
                approval_required_count=self._approval_required_count,
                confirmation_required_count=self._confirmation_required_count,
                last_status=self._last_status,
                last_reason=self._last_reason,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset runtime diagnostics.
        """

        with self._lock:
            self._action_count = 0
            self._success_count = 0
            self._blocked_count = 0
            self._failed_count = 0
            self._approval_required_count = 0
            self._confirmation_required_count = 0
            self._last_status = None
            self._last_reason = None
            self._last_error = None

    def _execute_allowed(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> BrowserActionResult:
        if request.kind in {
            BrowserActionKind.OPEN_URL,
            BrowserActionKind.SEARCH_WEB,
            BrowserActionKind.DOWNLOAD_FILE,
        }:
            return self._open_visible(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == BrowserActionKind.READ_PAGE_TITLE:
            return self._read_page_title(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == BrowserActionKind.SUMMARIZE_PAGE:
            return self._summarize_page(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        if request.kind == BrowserActionKind.FILL_FORM:
            return self._form_fill_visible(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
            )

        raise ValueError(f"unsupported browser action: {request.kind.value}")

    def _open_visible(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> BrowserActionResult:
        url = policy_result.sanitized_url

        if url is None:
            raise ValueError("sanitized URL missing.")

        opened = self._launcher.open(url)

        if not opened:
            return self._failed_result(
                request=request,
                policy_result=policy_result,
                validation_result=validation_result,
                started_at=started_at,
                monotonic_start=monotonic_start,
                output="browser launcher failed",
            )

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            url=url,
            output=f"opened visible browser URL: {url}",
        )

    def _read_page_title(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> BrowserActionResult:
        title = request.page_title or self._title_from_url(policy_result.sanitized_url)

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            url=policy_result.sanitized_url,
            title=title,
            output=f"page title available: {title}",
        )

    def _summarize_page(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> BrowserActionResult:
        page_text = request.page_text or ""
        summary = self._summarize_text(page_text)

        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            url=policy_result.sanitized_url,
            summary=summary,
            output="page summary prepared from provided page text",
        )

    def _form_fill_visible(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
    ) -> BrowserActionResult:
        return self._success_result(
            request=request,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            monotonic_start=monotonic_start,
            url=policy_result.sanitized_url,
            output=(
                "form-fill plan prepared for visible browser session; "
                "no hidden submission performed"
            ),
            metadata={
                "field_count": len(request.form_fields),
                "submitted": False,
            },
        )

    def _build_plan(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
    ) -> ActionPlan:
        action_kind = self._action_kind(request.kind)
        capability = self._capability(request.kind)
        risk = policy_result.risk
        scope = ActionScope.BROWSER
        timeout_ms = 30_000 if risk in {ActionRisk.HIGH, ActionRisk.CRITICAL} else None
        requires_approval = (
            request.approved
            or policy_result.decision == BrowserActionDecision.REQUIRE_APPROVAL
            or policy_result.permission_decision == PermissionDecision.REQUIRE_APPROVAL
            or risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}
        )
        arguments: dict[str, object] = {}

        if request.url is not None:
            arguments["path"] = request.url

        if request.query is not None:
            arguments["query"] = request.query

        step = ActionStep(
            action_id=request.action_id,
            order=0,
            kind=action_kind,
            capability=capability,
            scope=scope,
            risk=risk,
            description=f"execute governed browser action: {request.kind.value}",
            arguments=arguments,
            timeout_ms=timeout_ms,
            interruptible=True,
            rollback_supported=False,
        )

        return ActionPlan(
            action_id=request.action_id,
            goal=f"execute browser action: {request.kind.value}",
            steps=(step,),
            risk=risk,
            scope=scope,
            requires_approval=requires_approval,
            permission_decision=policy_result.permission_decision,
            status=ActionStatus.PLANNED,
        )

    def _register_default_browser_tool(self) -> None:
        self._registry.register(
            ToolDescriptor(
                tool_id="tool_browser_runtime",
                name="browser runtime",
                description="Governed visible browser runtime",
                capabilities=(
                    ToolCapability.OPEN_BROWSER,
                    ToolCapability.SEARCH_WEB,
                ),
                supported_action_kinds=(
                    ActionKind.BROWSER_OPEN,
                    ActionKind.BROWSER_SEARCH,
                    ActionKind.READ,
                ),
                scopes=(ActionScope.BROWSER,),
                max_risk=ActionRisk.HIGH,
                required_permission=PermissionDecision.REQUIRE_APPROVAL,
                availability=ToolAvailability.AVAILABLE,
                health=ToolHealth.HEALTHY,
                enabled=True,
            )
        )

    @staticmethod
    def _action_kind(kind: BrowserActionKind) -> ActionKind:
        return {
            BrowserActionKind.OPEN_URL: ActionKind.BROWSER_OPEN,
            BrowserActionKind.SEARCH_WEB: ActionKind.BROWSER_SEARCH,
            BrowserActionKind.READ_PAGE_TITLE: ActionKind.READ,
            BrowserActionKind.SUMMARIZE_PAGE: ActionKind.READ,
            BrowserActionKind.DOWNLOAD_FILE: ActionKind.BROWSER_OPEN,
            BrowserActionKind.FILL_FORM: ActionKind.BROWSER_OPEN,
        }[kind]

    @staticmethod
    def _capability(kind: BrowserActionKind) -> ToolCapability:
        return {
            BrowserActionKind.OPEN_URL: ToolCapability.OPEN_BROWSER,
            BrowserActionKind.SEARCH_WEB: ToolCapability.SEARCH_WEB,
            BrowserActionKind.READ_PAGE_TITLE: ToolCapability.OPEN_BROWSER,
            BrowserActionKind.SUMMARIZE_PAGE: ToolCapability.OPEN_BROWSER,
            BrowserActionKind.DOWNLOAD_FILE: ToolCapability.OPEN_BROWSER,
            BrowserActionKind.FILL_FORM: ToolCapability.OPEN_BROWSER,
        }[kind]

    @staticmethod
    def _title_from_url(url: str | None) -> str:
        if not url:
            return "Untitled page"

        parsed = urlparse(url)

        return parsed.netloc or "Untitled page"

    def _summarize_text(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())

        if not cleaned:
            return "No page text was provided for summarization."

        if len(cleaned) <= self._config.max_summary_chars:
            return cleaned

        return cleaned[: self._config.max_summary_chars].rstrip() + "..."

    def _success_result(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        validation_result: ActionValidationResult,
        started_at: object,
        monotonic_start: float,
        output: str,
        url: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BrowserActionResult:
        return BrowserActionResult(
            action_id=request.action_id,
            kind=request.kind,
            status=ActionStatus.SUCCEEDED,
            success=True,
            decision=BrowserActionDecision.ALLOW,
            reason=BrowserActionReason.ACTION_SUCCEEDED,
            url=url,
            title=title,
            summary=summary,
            output=output,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": BrowserActionReason.ACTION_SUCCEEDED.value,
                **(metadata or {}),
            },
        )

    def _blocked_result(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        started_at: object,
        monotonic_start: float,
        validation_result: ActionValidationResult | None = None,
        reason: BrowserActionReason | None = None,
    ) -> BrowserActionResult:
        final_reason = reason or policy_result.reason

        return BrowserActionResult(
            action_id=request.action_id,
            kind=request.kind,
            status=ActionStatus.BLOCKED,
            success=False,
            decision=policy_result.decision,
            reason=final_reason,
            url=policy_result.sanitized_url,
            output=policy_result.explanation,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": final_reason.value,
            },
        )

    def _failed_result(
        self,
        *,
        request: BrowserActionRequest,
        policy_result: BrowserActionPolicyResult,
        started_at: object,
        monotonic_start: float,
        output: str,
        validation_result: ActionValidationResult | None = None,
    ) -> BrowserActionResult:
        return BrowserActionResult(
            action_id=request.action_id,
            kind=request.kind,
            status=ActionStatus.FAILED,
            success=False,
            decision=BrowserActionDecision.DENY,
            reason=BrowserActionReason.ACTION_FAILED,
            output=output,
            policy_result=policy_result,
            validation_result=validation_result,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=self._duration_ms(monotonic_start),
            metadata={
                "runtime": self.name,
                "reason": BrowserActionReason.ACTION_FAILED.value,
            },
        )

    def _record(self, result: BrowserActionResult) -> None:
        with self._lock:
            self._last_status = result.status
            self._last_reason = result.reason

            if result.success:
                self._success_count += 1

            elif result.decision == BrowserActionDecision.REQUIRE_APPROVAL:
                self._approval_required_count += 1
                self._blocked_count += 1

            elif result.decision == BrowserActionDecision.REQUIRE_CONFIRMATION:
                self._confirmation_required_count += 1
                self._blocked_count += 1

            elif result.status == ActionStatus.BLOCKED:
                self._blocked_count += 1

            else:
                self._failed_count += 1

    @staticmethod
    def _duration_ms(start: float) -> int:
        return max(0, int((time.monotonic() - start) * 1000))