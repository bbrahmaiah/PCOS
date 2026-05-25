from __future__ import annotations


class OrchestrationError(Exception):
    """
    Base exception for Phase 6 orchestration runtime errors.
    """


class OrchestrationContractError(OrchestrationError):
    """
    Raised when an orchestration contract is invalid.
    """


class InvalidTaskStateError(OrchestrationContractError):
    """
    Raised when a task state transition or state shape is invalid.
    """


class TaskDependencyError(OrchestrationContractError):
    """
    Raised when task dependencies are invalid or cyclic.
    """


class WorkerContractError(OrchestrationContractError):
    """
    Raised when a worker contract is invalid.
    """


class ResourceBudgetError(OrchestrationContractError):
    """
    Raised when a resource budget contract is invalid.
    """


class ResourceBudgetExceededError(ResourceBudgetError):
    """
    Raised when a resource budget would be exceeded.
    """


class OrchestratorStateError(OrchestrationContractError):
    """
    Raised when orchestrator state is invalid.
    """