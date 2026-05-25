from __future__ import annotations


class ToolRuntimeError(Exception):
    """
    Base exception for Tool & Action Runtime failures.
    """


class ToolContractError(ToolRuntimeError):
    """
    Raised when a tool/action contract is invalid.
    """


class ToolPermissionError(ToolRuntimeError):
    """
    Raised when policy denies or blocks an action.
    """


class ToolValidationError(ToolRuntimeError):
    """
    Raised when an action plan fails validation before execution.
    """


class ToolExecutionError(ToolRuntimeError):
    """
    Raised when a governed tool execution fails.
    """


class ToolCancellationError(ToolRuntimeError):
    """
    Raised when an action is cancelled or cannot be cancelled safely.
    """


class ToolRegistryError(ToolRuntimeError):
    """
    Raised when tool registration or lookup fails.
    """


class ToolApprovalError(ToolRuntimeError):
    """
    Raised when required human approval is missing, expired, or rejected.
    """