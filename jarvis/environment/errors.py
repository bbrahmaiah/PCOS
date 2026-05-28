from __future__ import annotations


class EnvironmentError(Exception):
    """
    Base error for Phase 8 environment cognition.

    Phase 8 errors are intentionally isolated from vision/OCR/tool errors.
    Environment cognition is its own runtime boundary.
    """


class EnvironmentContractError(EnvironmentError):
    """
    Raised when an environment contract is malformed or unsafe.
    """


class EnvironmentConfidenceError(EnvironmentError):
    """
    Raised when confidence/trust is too low for a requested operation.
    """


class EnvironmentPrivacyError(EnvironmentError):
    """
    Raised when an environment operation violates privacy boundaries.
    """


class EnvironmentPolicyError(EnvironmentError):
    """
    Raised when an environment action violates governance policy.
    """


class EnvironmentStateError(EnvironmentError):
    """
    Raised when environment state is missing, stale, or contradictory.
    """