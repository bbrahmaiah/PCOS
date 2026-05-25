from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from pydantic import Field, field_validator, model_validator

from jarvis.tools.ids import new_tool_id, utc_now
from jarvis.tools.models import (
    ActionKind,
    ActionRisk,
    ActionScope,
    PermissionDecision,
    ToolCapability,
    ToolId,
    ToolModel,
)


class ToolAvailability(StrEnum):
    """
    Whether a tool is currently available to the runtime.
    """

    AVAILABLE = "available"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class ToolHealth(StrEnum):
    """
    Health state of a registered tool.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ToolRegistrationStatus(StrEnum):
    """
    Result of a tool registration operation.
    """

    REGISTERED = "registered"
    UPDATED = "updated"
    REJECTED = "rejected"


class ToolLookupStatus(StrEnum):
    """
    Result of a tool lookup operation.
    """

    FOUND = "found"
    NOT_FOUND = "not_found"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


class ToolDescriptor(ToolModel):
    """
    Static contract for one registered tool.

    This descriptor does not execute anything. It only declares what the tool is,
    what it can do, what risk it carries, and what policy decisions are required
    before future execution runtimes may use it.
    """

    tool_id: str = Field(default_factory=new_tool_id)
    name: str
    description: str
    capabilities: tuple[ToolCapability, ...]
    supported_action_kinds: tuple[ActionKind, ...]
    scopes: tuple[ActionScope, ...]
    max_risk: ActionRisk = ActionRisk.LOW
    required_permission: PermissionDecision = PermissionDecision.ALLOW
    availability: ToolAvailability = ToolAvailability.AVAILABLE
    health: ToolHealth = ToolHealth.UNKNOWN
    enabled: bool = True
    version: str = "1.0.0"
    owner: str = "jarvis"
    created_at: object = Field(default_factory=utc_now)
    updated_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("tool_id", "name", "description", "version", "owner")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @field_validator("capabilities")
    @classmethod
    def _capabilities_required(
        cls,
        value: tuple[ToolCapability, ...],
    ) -> tuple[ToolCapability, ...]:
        if not value:
            raise ValueError("tool descriptor requires at least one capability.")

        if len(value) != len(set(value)):
            raise ValueError("tool capabilities cannot contain duplicates.")

        return value

    @field_validator("supported_action_kinds")
    @classmethod
    def _action_kinds_required(
        cls,
        value: tuple[ActionKind, ...],
    ) -> tuple[ActionKind, ...]:
        if not value:
            raise ValueError("tool descriptor requires at least one action kind.")

        if len(value) != len(set(value)):
            raise ValueError("supported action kinds cannot contain duplicates.")

        return value

    @field_validator("scopes")
    @classmethod
    def _scopes_required(
        cls,
        value: tuple[ActionScope, ...],
    ) -> tuple[ActionScope, ...]:
        if not value:
            raise ValueError("tool descriptor requires at least one scope.")

        if len(value) != len(set(value)):
            raise ValueError("tool scopes cannot contain duplicates.")

        return value

    @model_validator(mode="after")
    def _validate_availability_consistency(self) -> ToolDescriptor:
        if not self.enabled:
            if self.availability == ToolAvailability.AVAILABLE:
                raise ValueError("disabled tools cannot be marked available.")

        if self.health == ToolHealth.UNHEALTHY:
            if self.availability == ToolAvailability.AVAILABLE:
                raise ValueError("unhealthy tools cannot be marked available.")

        if self.max_risk in {ActionRisk.HIGH, ActionRisk.CRITICAL}:
            allowed = {
                PermissionDecision.REQUIRE_APPROVAL,
                PermissionDecision.REQUIRE_CONFIRMATION,
                PermissionDecision.DENY,
                PermissionDecision.SANDBOX_ONLY,
            }

            if self.required_permission not in allowed:
                raise ValueError(
                    "high and critical risk tools require approval, "
                    "confirmation, sandbox, or denial policy."
                )

        return self

    @property
    def is_available(self) -> bool:
        return (
            self.enabled
            and self.availability == ToolAvailability.AVAILABLE
            and self.health in {ToolHealth.HEALTHY, ToolHealth.UNKNOWN}
        )

    @property
    def is_healthy(self) -> bool:
        return self.health == ToolHealth.HEALTHY

    def supports_capability(self, capability: ToolCapability) -> bool:
        return capability in self.capabilities

    def supports_action_kind(self, action_kind: ActionKind) -> bool:
        return action_kind in self.supported_action_kinds

    def supports_scope(self, scope: ActionScope) -> bool:
        return scope in self.scopes


class ToolRegistrationResult(ToolModel):
    """
    Result of registering or updating a tool descriptor.
    """

    tool_id: str
    status: ToolRegistrationStatus
    descriptor: ToolDescriptor | None = None
    reason: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("tool_id", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class ToolLookupResult(ToolModel):
    """
    Result of looking up a tool by id, name, or capability.
    """

    status: ToolLookupStatus
    descriptor: ToolDescriptor | None = None
    reason: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("reason cannot be empty.")

        return cleaned

    @property
    def found(self) -> bool:
        return self.status == ToolLookupStatus.FOUND and self.descriptor is not None


@dataclass(frozen=True, slots=True)
class ToolRegistryConfig:
    """
    Configuration for ToolRegistry.
    """

    name: str = "tool_registry"
    allow_updates: bool = True
    require_unique_names: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class ToolRegistrySnapshot:
    """
    Observable diagnostics for the registry.
    """

    name: str
    tool_count: int
    enabled_count: int
    available_count: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    registration_count: int
    lookup_count: int
    last_registered_tool_id: str | None
    last_lookup_status: ToolLookupStatus | None
    last_error: str | None


class ToolRegistry:
    """
    Safe discovery runtime for tools.

    Responsibilities:
    - register explicit tool descriptors
    - reject duplicate/invalid hidden tools
    - support lookup by id, name, capability, kind, and scope
    - track availability and health
    - expose diagnostics

    Non-responsibilities:
    - no tool execution
    - no shell calls
    - no filesystem mutation
    - no browser automation
    - no policy execution
    """

    def __init__(
        self,
        *,
        config: ToolRegistryConfig | None = None,
        descriptors: Iterable[ToolDescriptor] = (),
    ) -> None:
        self._config = config or ToolRegistryConfig()
        self._config.validate()

        self._lock = RLock()
        self._tools_by_id: dict[str, ToolDescriptor] = {}
        self._tool_ids_by_name: dict[str, str] = {}

        self._registration_count = 0
        self._lookup_count = 0
        self._last_registered_tool_id: str | None = None
        self._last_lookup_status: ToolLookupStatus | None = None
        self._last_error: str | None = None

        for descriptor in descriptors:
            self.register(descriptor)

    @property
    def name(self) -> str:
        return self._config.name

    def register(self, descriptor: ToolDescriptor) -> ToolRegistrationResult:
        """
        Register or update one explicit tool descriptor.
        """

        with self._lock:
            self._registration_count += 1
            self._last_error = None

            existing = self._tools_by_id.get(descriptor.tool_id)
            normalized_name = self._normalize_name(descriptor.name)
            existing_name_tool_id = self._tool_ids_by_name.get(normalized_name)

            if (
                self._config.require_unique_names
                and existing_name_tool_id is not None
                and existing_name_tool_id != descriptor.tool_id
            ):
                self._last_error = "tool name already registered"

                return ToolRegistrationResult(
                    tool_id=descriptor.tool_id,
                    status=ToolRegistrationStatus.REJECTED,
                    descriptor=None,
                    reason=(
                        "tool name already registered: "
                        f"{descriptor.name}"
                    ),
                )

            if existing is not None and not self._config.allow_updates:
                self._last_error = "tool updates are disabled"

                return ToolRegistrationResult(
                    tool_id=descriptor.tool_id,
                    status=ToolRegistrationStatus.REJECTED,
                    descriptor=existing,
                    reason="tool already registered and updates are disabled",
                )

            status = (
                ToolRegistrationStatus.UPDATED
                if existing is not None
                else ToolRegistrationStatus.REGISTERED
            )
            updated_descriptor = descriptor.model_copy(
                update={
                    "updated_at": utc_now(),
                }
            )

            self._tools_by_id[descriptor.tool_id] = updated_descriptor
            self._tool_ids_by_name[normalized_name] = descriptor.tool_id
            self._last_registered_tool_id = descriptor.tool_id

            return ToolRegistrationResult(
                tool_id=descriptor.tool_id,
                status=status,
                descriptor=updated_descriptor,
                reason=f"tool {status.value}",
            )

    def unregister(self, tool_id: str) -> ToolRegistrationResult:
        """
        Remove one registered tool from the registry.
        """

        cleaned = self._clean_id(tool_id)

        with self._lock:
            descriptor = self._tools_by_id.pop(cleaned, None)

            if descriptor is None:
                return ToolRegistrationResult(
                    tool_id=cleaned,
                    status=ToolRegistrationStatus.REJECTED,
                    descriptor=None,
                    reason="tool is not registered",
                )

            self._tool_ids_by_name.pop(
                self._normalize_name(descriptor.name),
                None,
            )

            return ToolRegistrationResult(
                tool_id=cleaned,
                status=ToolRegistrationStatus.UPDATED,
                descriptor=descriptor,
                reason="tool unregistered",
            )

    def get(self, tool_id: str) -> ToolLookupResult:
        """
        Lookup a tool by id.
        """

        cleaned = self._clean_id(tool_id)

        with self._lock:
            self._lookup_count += 1
            descriptor = self._tools_by_id.get(cleaned)

        result = self._lookup_result(descriptor)
        self._record_lookup(result)

        return result

    def get_by_name(self, name: str) -> ToolLookupResult:
        """
        Lookup a tool by registered name.
        """

        normalized_name = self._normalize_name(name)

        with self._lock:
            self._lookup_count += 1
            tool_id = self._tool_ids_by_name.get(normalized_name)
            descriptor = (
                self._tools_by_id.get(tool_id)
                if tool_id is not None
                else None
            )

        result = self._lookup_result(descriptor)
        self._record_lookup(result)

        return result

    def find_by_capability(
        self,
        capability: ToolCapability,
        *,
        available_only: bool = True,
    ) -> tuple[ToolDescriptor, ...]:
        """
        Return tools that support a capability.
        """

        with self._lock:
            self._lookup_count += 1
            descriptors = tuple(
                descriptor
                for descriptor in self._tools_by_id.values()
                if descriptor.supports_capability(capability)
                and (descriptor.is_available or not available_only)
            )

        self._last_lookup_status = (
            ToolLookupStatus.FOUND
            if descriptors
            else ToolLookupStatus.NOT_FOUND
        )

        return descriptors

    def find_by_action_kind(
        self,
        action_kind: ActionKind,
        *,
        available_only: bool = True,
    ) -> tuple[ToolDescriptor, ...]:
        """
        Return tools that support an action kind.
        """

        with self._lock:
            self._lookup_count += 1
            descriptors = tuple(
                descriptor
                for descriptor in self._tools_by_id.values()
                if descriptor.supports_action_kind(action_kind)
                and (descriptor.is_available or not available_only)
            )

        self._last_lookup_status = (
            ToolLookupStatus.FOUND
            if descriptors
            else ToolLookupStatus.NOT_FOUND
        )

        return descriptors

    def find_by_scope(
        self,
        scope: ActionScope,
        *,
        available_only: bool = True,
    ) -> tuple[ToolDescriptor, ...]:
        """
        Return tools that support an action scope.
        """

        with self._lock:
            self._lookup_count += 1
            descriptors = tuple(
                descriptor
                for descriptor in self._tools_by_id.values()
                if descriptor.supports_scope(scope)
                and (descriptor.is_available or not available_only)
            )

        self._last_lookup_status = (
            ToolLookupStatus.FOUND
            if descriptors
            else ToolLookupStatus.NOT_FOUND
        )

        return descriptors

    def set_availability(
        self,
        tool_id: str,
        availability: ToolAvailability,
    ) -> ToolRegistrationResult:
        """
        Update availability for a registered tool.
        """

        return self._update_descriptor(
            tool_id=tool_id,
            updates={
                "availability": availability,
                "updated_at": utc_now(),
            },
            reason="tool availability updated",
        )

    def set_health(
        self,
        tool_id: str,
        health: ToolHealth,
    ) -> ToolRegistrationResult:
        """
        Update health for a registered tool.
        """

        availability = (
            ToolAvailability.UNAVAILABLE
            if health == ToolHealth.UNHEALTHY
            else None
        )
        updates: dict[str, object] = {
            "health": health,
            "updated_at": utc_now(),
        }

        if availability is not None:
            updates["availability"] = availability

        return self._update_descriptor(
            tool_id=tool_id,
            updates=updates,
            reason="tool health updated",
        )

    def enable(self, tool_id: str) -> ToolRegistrationResult:
        """
        Enable a registered tool.
        """

        return self._update_descriptor(
            tool_id=tool_id,
            updates={
                "enabled": True,
                "availability": ToolAvailability.AVAILABLE,
                "updated_at": utc_now(),
            },
            reason="tool enabled",
        )

    def disable(self, tool_id: str) -> ToolRegistrationResult:
        """
        Disable a registered tool.
        """

        return self._update_descriptor(
            tool_id=tool_id,
            updates={
                "enabled": False,
                "availability": ToolAvailability.DISABLED,
                "updated_at": utc_now(),
            },
            reason="tool disabled",
        )

    def all_tools(self) -> tuple[ToolDescriptor, ...]:
        """
        Return all registered tools.
        """

        with self._lock:
            return tuple(self._tools_by_id.values())

    def available_tools(self) -> tuple[ToolDescriptor, ...]:
        """
        Return only enabled and available tools.
        """

        with self._lock:
            return tuple(
                descriptor
                for descriptor in self._tools_by_id.values()
                if descriptor.is_available
            )

    def snapshot(self) -> ToolRegistrySnapshot:
        """
        Return registry diagnostics.
        """

        with self._lock:
            descriptors = tuple(self._tools_by_id.values())

            return ToolRegistrySnapshot(
                name=self.name,
                tool_count=len(descriptors),
                enabled_count=sum(1 for item in descriptors if item.enabled),
                available_count=sum(
                    1
                    for item in descriptors
                    if item.availability == ToolAvailability.AVAILABLE
                ),
                healthy_count=sum(
                    1 for item in descriptors if item.health == ToolHealth.HEALTHY
                ),
                degraded_count=sum(
                    1 for item in descriptors if item.health == ToolHealth.DEGRADED
                ),
                unhealthy_count=sum(
                    1
                    for item in descriptors
                    if item.health == ToolHealth.UNHEALTHY
                ),
                registration_count=self._registration_count,
                lookup_count=self._lookup_count,
                last_registered_tool_id=self._last_registered_tool_id,
                last_lookup_status=self._last_lookup_status,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Clear all tools and diagnostics.
        """

        with self._lock:
            self._tools_by_id.clear()
            self._tool_ids_by_name.clear()
            self._registration_count = 0
            self._lookup_count = 0
            self._last_registered_tool_id = None
            self._last_lookup_status = None
            self._last_error = None

    def _update_descriptor(
        self,
        *,
        tool_id: str,
        updates: dict[str, object],
        reason: str,
    ) -> ToolRegistrationResult:
        cleaned = self._clean_id(tool_id)

        with self._lock:
            descriptor = self._tools_by_id.get(cleaned)

            if descriptor is None:
                return ToolRegistrationResult(
                    tool_id=cleaned,
                    status=ToolRegistrationStatus.REJECTED,
                    descriptor=None,
                    reason="tool is not registered",
                )

            updated = descriptor.model_copy(update=updates)
            self._tools_by_id[cleaned] = updated
            self._last_registered_tool_id = cleaned

            return ToolRegistrationResult(
                tool_id=cleaned,
                status=ToolRegistrationStatus.UPDATED,
                descriptor=updated,
                reason=reason,
            )

    @staticmethod
    def _lookup_result(
        descriptor: ToolDescriptor | None,
    ) -> ToolLookupResult:
        if descriptor is None:
            return ToolLookupResult(
                status=ToolLookupStatus.NOT_FOUND,
                descriptor=None,
                reason="tool not found",
            )

        if not descriptor.enabled:
            return ToolLookupResult(
                status=ToolLookupStatus.DISABLED,
                descriptor=descriptor,
                reason="tool is disabled",
            )

        if descriptor.availability in {
            ToolAvailability.UNAVAILABLE,
            ToolAvailability.DEGRADED,
        }:
            return ToolLookupResult(
                status=ToolLookupStatus.UNAVAILABLE,
                descriptor=descriptor,
                reason=f"tool is {descriptor.availability.value}",
            )

        return ToolLookupResult(
            status=ToolLookupStatus.FOUND,
            descriptor=descriptor,
            reason="tool found",
        )

    def _record_lookup(self, result: ToolLookupResult) -> None:
        with self._lock:
            self._last_lookup_status = result.status

    @staticmethod
    def _normalize_name(name: str) -> str:
        cleaned = " ".join(name.strip().casefold().split())

        if not cleaned:
            raise ValueError("tool name cannot be empty.")

        return cleaned

    @staticmethod
    def _clean_id(tool_id: str) -> str:
        cleaned = tool_id.strip()

        if not cleaned:
            raise ValueError("tool id cannot be empty.")

        ToolId(value=cleaned)

        return cleaned