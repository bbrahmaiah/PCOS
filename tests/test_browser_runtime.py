from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionStatus,
    BrowserActionDecision,
    BrowserActionKind,
    BrowserActionReason,
    BrowserActionRequest,
    BrowserPolicy,
    BrowserRuntime,
    BrowserRuntimeConfig,
)


class FakeBrowserLauncher:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.urls: list[str] = []

    def open(self, url: str) -> bool:
        self.urls.append(url)

        return self.opened


def test_browser_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        BrowserRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        BrowserRuntimeConfig(search_base_url=" ").validate()

    with pytest.raises(ValueError):
        BrowserRuntimeConfig(max_summary_chars=0).validate()


def test_request_requires_url_for_open() -> None:
    with pytest.raises(ValidationError):
        BrowserActionRequest(kind=BrowserActionKind.OPEN_URL)


def test_request_requires_query_for_search() -> None:
    with pytest.raises(ValidationError):
        BrowserActionRequest(kind=BrowserActionKind.SEARCH_WEB)


def test_request_requires_form_fields_for_form_fill() -> None:
    with pytest.raises(ValidationError):
        BrowserActionRequest(
            kind=BrowserActionKind.FILL_FORM,
            url="https://example.com/form",
        )


def test_policy_allows_safe_open_url() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="example.com",
        )
    )

    assert result.decision == BrowserActionDecision.ALLOW
    assert result.reason == BrowserActionReason.SAFE_OPEN_ALLOWED
    assert result.sanitized_url == "https://example.com"


def test_policy_allows_safe_search() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.SEARCH_WEB,
            query="python pathlib docs",
        )
    )

    assert result.decision == BrowserActionDecision.ALLOW
    assert result.reason == BrowserActionReason.SAFE_SEARCH_ALLOWED
    assert "python+pathlib+docs" in (result.sanitized_url or "")


def test_policy_blocks_javascript_url() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="javascript:alert(1)",
        )
    )

    assert result.decision == BrowserActionDecision.DENY
    assert result.reason == BrowserActionReason.UNSUPPORTED_SCHEME_BLOCKED


def test_policy_blocks_payment_context() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="https://example.com/checkout",
        )
    )

    assert result.decision == BrowserActionDecision.DENY
    assert result.reason == BrowserActionReason.PAYMENT_BLOCKED


def test_policy_blocks_password_form() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.FILL_FORM,
            url="https://example.com/login",
            form_fields={"password": "secret"},
            approved=True,
        )
    )

    assert result.decision == BrowserActionDecision.DENY
    assert result.reason == BrowserActionReason.PASSWORD_ENTRY_BLOCKED


def test_policy_blocks_account_change() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="https://example.com/change-password",
        )
    )

    assert result.decision == BrowserActionDecision.DENY
    assert result.reason == BrowserActionReason.PASSWORD_ENTRY_BLOCKED


def test_download_requires_approval() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.DOWNLOAD_FILE,
            url="https://example.com/file.zip",
        )
    )

    assert result.decision == BrowserActionDecision.REQUIRE_APPROVAL
    assert result.reason == BrowserActionReason.DOWNLOAD_REQUIRES_APPROVAL


def test_form_fill_requires_approval() -> None:
    result = BrowserPolicy().evaluate(
        BrowserActionRequest(
            kind=BrowserActionKind.FILL_FORM,
            url="https://example.com/contact",
            form_fields={"name": "Bala"},
        )
    )

    assert result.decision == BrowserActionDecision.REQUIRE_APPROVAL
    assert result.reason == BrowserActionReason.FORM_FILL_REQUIRES_APPROVAL


def test_runtime_opens_visible_url() -> None:
    launcher = FakeBrowserLauncher()
    runtime = BrowserRuntime(launcher=launcher)

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="https://example.com",
        )
    )

    assert result.success is True
    assert result.status == ActionStatus.SUCCEEDED
    assert launcher.urls == ["https://example.com"]


def test_runtime_searches_web_visibly() -> None:
    launcher = FakeBrowserLauncher()
    runtime = BrowserRuntime(launcher=launcher)

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.SEARCH_WEB,
            query="jarvis architecture",
        )
    )

    assert result.success is True
    assert launcher.urls
    assert "jarvis+architecture" in launcher.urls[0]


def test_runtime_blocks_sensitive_action_without_launcher() -> None:
    launcher = FakeBrowserLauncher()
    runtime = BrowserRuntime(launcher=launcher)

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="https://example.com/billing",
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.BLOCKED
    assert not launcher.urls


def test_runtime_read_page_title() -> None:
    runtime = BrowserRuntime()

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.READ_PAGE_TITLE,
            url="https://docs.python.org",
            page_title="Python Docs",
        )
    )

    assert result.success is True
    assert result.title == "Python Docs"


def test_runtime_summarize_page_text() -> None:
    runtime = BrowserRuntime(
        config=BrowserRuntimeConfig(max_summary_chars=20)
    )

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.SUMMARIZE_PAGE,
            url="https://example.com",
            page_text="This is a long page text that should be shortened.",
        )
    )

    assert result.success is True
    assert result.summary == "This is a long page..."


def test_runtime_download_without_approval_is_blocked() -> None:
    launcher = FakeBrowserLauncher()
    runtime = BrowserRuntime(launcher=launcher)

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.DOWNLOAD_FILE,
            url="https://example.com/file.zip",
        )
    )

    assert result.success is False
    assert result.decision == BrowserActionDecision.REQUIRE_APPROVAL
    assert not launcher.urls


def test_runtime_download_with_approval_opens_visible_url() -> None:
    launcher = FakeBrowserLauncher()
    runtime = BrowserRuntime(launcher=launcher)

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.DOWNLOAD_FILE,
            url="https://example.com/file.zip",
            approved=True,
        )
    )

    assert result.success is True
    assert launcher.urls == ["https://example.com/file.zip"]


def test_runtime_form_fill_with_approval_does_not_submit_hidden() -> None:
    runtime = BrowserRuntime()

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.FILL_FORM,
            url="https://example.com/contact",
            form_fields={"name": "Bala"},
            approved=True,
        )
    )

    assert result.success is True
    assert result.metadata["submitted"] is False


def test_runtime_launcher_failure_returns_failed() -> None:
    launcher = FakeBrowserLauncher(opened=False)
    runtime = BrowserRuntime(launcher=launcher)

    result = runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="https://example.com",
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.FAILED


def test_snapshot_and_reset() -> None:
    runtime = BrowserRuntime(launcher=FakeBrowserLauncher())

    runtime.execute(
        BrowserActionRequest(
            kind=BrowserActionKind.OPEN_URL,
            url="https://example.com",
        )
    )
    snapshot = runtime.snapshot()

    assert snapshot.action_count == 1
    assert snapshot.success_count == 1

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.action_count == 0
    assert reset_snapshot.last_status is None


def test_enum_values_are_stable() -> None:
    assert BrowserActionKind.OPEN_URL.value == "open_url"
    assert BrowserActionDecision.REQUIRE_APPROVAL.value == "require_approval"
    assert BrowserActionReason.PAYMENT_BLOCKED.value == "payment_blocked"