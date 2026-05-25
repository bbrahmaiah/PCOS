from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionKind,
    ActionRisk,
    ActionScope,
    PermissionDecision,
    ToolAvailability,
    ToolCapability,
    ToolDescriptor,
    ToolHealth,
    ToolLookupStatus,
    ToolRegistrationStatus,
    ToolRegistry,
    ToolRegistryConfig,
)


def descriptor(
    *,
    tool_id: str = "tool_test",
    name: str = "test tool",
    capability: ToolCapability = ToolCapability.READ_FILE,
    action_kind: ActionKind = ActionKind.READ,
    scope: ActionScope = ActionScope.WORKSPACE,
    risk: ActionRisk = ActionRisk.LOW,
    permission: PermissionDecision = PermissionDecision.ALLOW,
    enabled: bool = True,
    availability: ToolAvailability = ToolAvailability.AVAILABLE,
    health: ToolHealth = ToolHealth.HEALTHY,
) -> ToolDescriptor:
    return ToolDescriptor(
        tool_id=tool_id,
        name=name,
        description="Test tool descriptor",
        capabilities=(capability,),
        supported_action_kinds=(action_kind,),
        scopes=(scope,),
        max_risk=risk,
        required_permission=permission,
        enabled=enabled,
        availability=availability,
        health=health,
    )


def test_registry_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ToolRegistryConfig(name=" ").validate()


def test_descriptor_requires_name_description_and_version() -> None:
    with pytest.raises(ValidationError):
        ToolDescriptor(
            name=" ",
            description="valid",
            capabilities=(ToolCapability.READ_FILE,),
            supported_action_kinds=(ActionKind.READ,),
            scopes=(ActionScope.WORKSPACE,),
        )

    with pytest.raises(ValidationError):
        ToolDescriptor(
            name="tool",
            description=" ",
            capabilities=(ToolCapability.READ_FILE,),
            supported_action_kinds=(ActionKind.READ,),
            scopes=(ActionScope.WORKSPACE,),
        )


def test_descriptor_requires_capability_action_kind_and_scope() -> None:
    with pytest.raises(ValidationError):
        ToolDescriptor(
            name="tool",
            description="valid",
            capabilities=(),
            supported_action_kinds=(ActionKind.READ,),
            scopes=(ActionScope.WORKSPACE,),
        )

    with pytest.raises(ValidationError):
        ToolDescriptor(
            name="tool",
            description="valid",
            capabilities=(ToolCapability.READ_FILE,),
            supported_action_kinds=(),
            scopes=(ActionScope.WORKSPACE,),
        )

    with pytest.raises(ValidationError):
        ToolDescriptor(
            name="tool",
            description="valid",
            capabilities=(ToolCapability.READ_FILE,),
            supported_action_kinds=(ActionKind.READ,),
            scopes=(),
        )


def test_descriptor_rejects_duplicate_capabilities_kinds_and_scopes() -> None:
    with pytest.raises(ValidationError):
        ToolDescriptor(
            name="tool",
            description="valid",
            capabilities=(ToolCapability.READ_FILE, ToolCapability.READ_FILE),
            supported_action_kinds=(ActionKind.READ,),
            scopes=(ActionScope.WORKSPACE,),
        )

    with pytest.raises(ValidationError):
        ToolDescriptor(
            name="tool",
            description="valid",
            capabilities=(ToolCapability.READ_FILE,),
            supported_action_kinds=(ActionKind.READ, ActionKind.READ),
            scopes=(ActionScope.WORKSPACE,),
        )

    with pytest.raises(ValidationError):
        ToolDescriptor(
            name="tool",
            description="valid",
            capabilities=(ToolCapability.READ_FILE,),
            supported_action_kinds=(ActionKind.READ,),
            scopes=(ActionScope.WORKSPACE, ActionScope.WORKSPACE),
        )


def test_high_risk_tool_requires_stronger_permission() -> None:
    with pytest.raises(ValidationError):
        descriptor(
            risk=ActionRisk.HIGH,
            permission=PermissionDecision.ALLOW,
        )


def test_high_risk_tool_accepts_approval_permission() -> None:
    item = descriptor(
        risk=ActionRisk.HIGH,
        permission=PermissionDecision.REQUIRE_APPROVAL,
    )

    assert item.max_risk == ActionRisk.HIGH
    assert item.required_permission == PermissionDecision.REQUIRE_APPROVAL


def test_disabled_tool_cannot_be_available() -> None:
    with pytest.raises(ValidationError):
        descriptor(
            enabled=False,
            availability=ToolAvailability.AVAILABLE,
        )


def test_unhealthy_tool_cannot_be_available() -> None:
    with pytest.raises(ValidationError):
        descriptor(
            health=ToolHealth.UNHEALTHY,
            availability=ToolAvailability.AVAILABLE,
        )


def test_descriptor_support_helpers() -> None:
    item = descriptor()

    assert item.supports_capability(ToolCapability.READ_FILE)
    assert item.supports_action_kind(ActionKind.READ)
    assert item.supports_scope(ActionScope.WORKSPACE)
    assert item.is_available is True
    assert item.is_healthy is True


def test_register_tool() -> None:
    registry = ToolRegistry()
    item = descriptor()

    result = registry.register(item)
    snapshot = registry.snapshot()

    assert result.status == ToolRegistrationStatus.REGISTERED
    assert result.descriptor is not None
    assert snapshot.tool_count == 1
    assert snapshot.registration_count == 1


def test_update_existing_tool_when_allowed() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    updated = item.model_copy(update={"description": "Updated descriptor"})
    result = registry.register(updated)

    assert result.status == ToolRegistrationStatus.UPDATED
    assert result.descriptor is not None
    assert result.descriptor.description == "Updated descriptor"


def test_reject_update_when_disabled_by_config() -> None:
    item = descriptor()
    registry = ToolRegistry(
        config=ToolRegistryConfig(allow_updates=False),
        descriptors=(item,),
    )

    result = registry.register(
        item.model_copy(update={"description": "Updated descriptor"})
    )

    assert result.status == ToolRegistrationStatus.REJECTED
    assert result.reason == "tool already registered and updates are disabled"


def test_reject_duplicate_tool_name() -> None:
    registry = ToolRegistry()

    registry.register(descriptor(tool_id="tool_one", name="same"))
    result = registry.register(descriptor(tool_id="tool_two", name="same"))

    assert result.status == ToolRegistrationStatus.REJECTED
    assert "tool name already registered" in result.reason


def test_get_tool_by_id() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    result = registry.get(item.tool_id)

    assert result.status == ToolLookupStatus.FOUND
    assert result.found is True
    assert result.descriptor is not None
    assert result.descriptor.tool_id == item.tool_id
    assert result.descriptor.name == item.name
    assert result.descriptor.capabilities == item.capabilities


def test_get_tool_by_name() -> None:
    registry = ToolRegistry()
    item = descriptor(name="Read File Tool")

    registry.register(item)
    result = registry.get_by_name(" read   file tool ")

    assert result.status == ToolLookupStatus.FOUND
    assert result.descriptor is not None
    assert result.descriptor.name == "Read File Tool"


def test_get_missing_tool() -> None:
    registry = ToolRegistry()

    result = registry.get("tool_missing")

    assert result.status == ToolLookupStatus.NOT_FOUND
    assert result.found is False


def test_disabled_tool_lookup_status() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    registry.disable(item.tool_id)
    result = registry.get(item.tool_id)

    assert result.status == ToolLookupStatus.DISABLED
    assert result.descriptor is not None
    assert result.descriptor.enabled is False


def test_unavailable_tool_lookup_status() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    registry.set_availability(item.tool_id, ToolAvailability.UNAVAILABLE)
    result = registry.get(item.tool_id)

    assert result.status == ToolLookupStatus.UNAVAILABLE


def test_find_by_capability() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    found = registry.find_by_capability(ToolCapability.READ_FILE)

    assert len(found) == 1
    assert found[0].tool_id == item.tool_id


def test_find_by_action_kind() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    found = registry.find_by_action_kind(ActionKind.READ)

    assert len(found) == 1
    assert found[0].tool_id == item.tool_id


def test_find_by_scope() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    found = registry.find_by_scope(ActionScope.WORKSPACE)

    assert len(found) == 1
    assert found[0].tool_id == item.tool_id


def test_find_can_include_disabled_tools() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    registry.disable(item.tool_id)

    assert len(registry.find_by_capability(ToolCapability.READ_FILE)) == 0
    assert (
        len(
            registry.find_by_capability(
                ToolCapability.READ_FILE,
                available_only=False,
            )
        )
        == 1
    )


def test_set_health_unhealthy_makes_tool_unavailable() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    result = registry.set_health(item.tool_id, ToolHealth.UNHEALTHY)

    assert result.status == ToolRegistrationStatus.UPDATED
    assert result.descriptor is not None
    assert result.descriptor.health == ToolHealth.UNHEALTHY
    assert result.descriptor.availability == ToolAvailability.UNAVAILABLE


def test_enable_and_disable_tool() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    disabled = registry.disable(item.tool_id)
    enabled = registry.enable(item.tool_id)

    assert disabled.descriptor is not None
    assert enabled.descriptor is not None
    assert disabled.descriptor.enabled is False
    assert enabled.descriptor.enabled is True


def test_unregister_tool() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    result = registry.unregister(item.tool_id)

    assert result.status == ToolRegistrationStatus.UPDATED
    assert registry.snapshot().tool_count == 0


def test_unregister_missing_tool_is_rejected() -> None:
    registry = ToolRegistry()

    result = registry.unregister("tool_missing")

    assert result.status == ToolRegistrationStatus.REJECTED


def test_update_missing_tool_is_rejected() -> None:
    registry = ToolRegistry()

    result = registry.set_health("tool_missing", ToolHealth.HEALTHY)

    assert result.status == ToolRegistrationStatus.REJECTED


def test_all_and_available_tools() -> None:
    registry = ToolRegistry()
    first = descriptor(tool_id="tool_one", name="one")
    second = descriptor(tool_id="tool_two", name="two")

    registry.register(first)
    registry.register(second)
    registry.disable(second.tool_id)

    assert len(registry.all_tools()) == 2
    assert len(registry.available_tools()) == 1


def test_snapshot_and_reset() -> None:
    registry = ToolRegistry()
    item = descriptor()

    registry.register(item)
    registry.get(item.tool_id)
    snapshot = registry.snapshot()

    assert snapshot.tool_count == 1
    assert snapshot.lookup_count == 1
    assert snapshot.last_lookup_status == ToolLookupStatus.FOUND

    registry.reset()
    reset_snapshot = registry.snapshot()

    assert reset_snapshot.tool_count == 0
    assert reset_snapshot.registration_count == 0
    assert reset_snapshot.lookup_count == 0


def test_empty_tool_id_rejected_on_lookup() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        registry.get(" ")


def test_registry_enum_values_are_stable() -> None:
    assert ToolAvailability.AVAILABLE.value == "available"
    assert ToolHealth.HEALTHY.value == "healthy"
    assert ToolRegistrationStatus.REGISTERED.value == "registered"
    assert ToolLookupStatus.FOUND.value == "found"