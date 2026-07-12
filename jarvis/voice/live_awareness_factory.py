from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from jarvis.voice.awareness_runtime import (
    VoiceAwarenessFact,
    VoiceAwarenessPriority,
    VoiceAwarenessProvider,
    VoiceAwarenessRequest,
    VoiceAwarenessRuntime,
    VoiceAwarenessSource,
    VoiceResponseBoundaryAwarenessProvider,
)


class MemoryRuntimeProtocol(Protocol):
    def retrieve(self, query: object) -> object:
        raise NotImplementedError


class GoalRuntimeProtocol(Protocol):
    def snapshot(self) -> object:
        raise NotImplementedError


class ToolRegistryProtocol(Protocol):
    def snapshot(self) -> object:
        raise NotImplementedError


class ToolPlannerProtocol(Protocol):
    def snapshot(self) -> object:
        raise NotImplementedError


def _text(value: object, fallback: str = "") -> str:
    text = str(value if value is not None else fallback).strip()
    return text or fallback


def _default_memory() -> MemoryRuntimeProtocol:
    from jarvis.cognition.memory import InMemoryShortTermMemoryStore

    return cast(MemoryRuntimeProtocol, InMemoryShortTermMemoryStore())


def _default_goals() -> GoalRuntimeProtocol:
    from jarvis.cognitive.goals import GoalRuntime

    return cast(GoalRuntimeProtocol, GoalRuntime())


def _default_tool_registry() -> ToolRegistryProtocol:
    from jarvis.tools import ToolRegistry

    return cast(ToolRegistryProtocol, ToolRegistry())


def _default_tool_planner() -> ToolPlannerProtocol:
    from jarvis.cognition.action_planning import ToolActionPlanner

    return cast(ToolPlannerProtocol, ToolActionPlanner())


@dataclass(slots=True)
class LiveMemoryAwarenessProvider(VoiceAwarenessProvider):
    memory: MemoryRuntimeProtocol = field(default_factory=_default_memory)
    source: VoiceAwarenessSource = VoiceAwarenessSource.MEMORY

    def collect(self, request: VoiceAwarenessRequest) -> tuple[VoiceAwarenessFact, ...]:
        from jarvis.cognition.memory import ShortTermMemoryQuery

        result = self.memory.retrieve(
            ShortTermMemoryQuery(
                query_text=request.transcript.text,
                max_items=5,
            )
        )
        items = tuple(getattr(result, "items", ()) or ())

        facts: list[VoiceAwarenessFact] = [
            VoiceAwarenessFact(
                source=self.source,
                key="memory_runtime_state",
                value=f"short-term memory active; retrieved_items={len(items)}",
                confidence=1.0,
                priority=VoiceAwarenessPriority.NORMAL,
                metadata={
                    "provider": type(self).__name__,
                    "retrieval_performed": True,
                    "retrieved_items": len(items),
                },
            )
        ]

        for index, item in enumerate(items[:5], start=1):
            facts.append(
                VoiceAwarenessFact(
                    source=self.source,
                    key=f"retrieved_memory_{index}",
                    value=_text(getattr(item, "text", item)),
                    confidence=float(getattr(item, "confidence", 0.85) or 0.85),
                    priority=VoiceAwarenessPriority.NORMAL,
                    metadata={
                        "provider": type(self).__name__,
                        "memory_id": _text(getattr(item, "memory_id", "")),
                    },
                )
            )

        return tuple(facts)


@dataclass(slots=True)
class LiveGoalAwarenessProvider(VoiceAwarenessProvider):
    goals: GoalRuntimeProtocol = field(default_factory=_default_goals)
    source: VoiceAwarenessSource = VoiceAwarenessSource.GOALS

    def collect(self, request: VoiceAwarenessRequest) -> tuple[VoiceAwarenessFact, ...]:
        snapshot = self.goals.snapshot()

        return (
            VoiceAwarenessFact(
                source=self.source,
                key="goal_runtime_state",
                value=str(snapshot),
                confidence=1.0,
                priority=VoiceAwarenessPriority.NORMAL,
                metadata={
                    "provider": type(self).__name__,
                    "snapshot_type": type(snapshot).__name__,
                },
            ),
        )


@dataclass(slots=True)
class LiveToolAwarenessProvider(VoiceAwarenessProvider):
    registry: ToolRegistryProtocol = field(default_factory=_default_tool_registry)
    planner: ToolPlannerProtocol = field(default_factory=_default_tool_planner)
    source: VoiceAwarenessSource = VoiceAwarenessSource.TOOLS

    def collect(self, request: VoiceAwarenessRequest) -> tuple[VoiceAwarenessFact, ...]:
        registry_snapshot = self.registry.snapshot()
        planner_snapshot = self.planner.snapshot()

        return (
            VoiceAwarenessFact(
                source=self.source,
                key="tool_control_contract",
                value=(
                    "tool_planner_validator_executor_pipeline=active; "
                    "llm_direct_tool_execution_allowed=false"
                ),
                confidence=1.0,
                priority=VoiceAwarenessPriority.HIGH,
                metadata={
                    "provider": type(self).__name__,
                    "llm_direct_tool_execution_allowed": False,
                    "approval_required_for_risky_actions": True,
                },
            ),
            VoiceAwarenessFact(
                source=self.source,
                key="tool_registry_state",
                value=(
                    f"registered_tools={getattr(registry_snapshot, 'tool_count', 0)}; "
                    "available_tools="
                    f"{getattr(registry_snapshot, 'available_count', 0)}; "
                    f"healthy_tools={getattr(registry_snapshot, 'healthy_count', 0)}"
                ),
                confidence=1.0,
                priority=VoiceAwarenessPriority.NORMAL,
                metadata={
                    "provider": type(self).__name__,
                    "snapshot_type": type(registry_snapshot).__name__,
                },
            ),
            VoiceAwarenessFact(
                source=self.source,
                key="tool_planner_state",
                value=(
                    f"name={_text(getattr(planner_snapshot, 'name', 'unknown'))}; "
                    f"planned_count={getattr(planner_snapshot, 'planned_count', 0)}; "
                    "execution_state=planning_only"
                ),
                confidence=1.0,
                priority=VoiceAwarenessPriority.NORMAL,
                metadata={
                    "provider": type(self).__name__,
                    "snapshot_type": type(planner_snapshot).__name__,
                },
            ),
        )


@dataclass(slots=True)
class LiveEnvironmentAwarenessProvider(VoiceAwarenessProvider):
    source: VoiceAwarenessSource = VoiceAwarenessSource.ENVIRONMENT

    def collect(self, request: VoiceAwarenessRequest) -> tuple[VoiceAwarenessFact, ...]:
        user32: Any = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()

        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)

        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        title = buffer.value.strip() or "unknown"

        return (
            VoiceAwarenessFact(
                source=self.source,
                key="active_windows_foreground_window",
                value=f"title={title}; pid={pid.value}; hwnd={hwnd}",
                confidence=1.0,
                priority=VoiceAwarenessPriority.NORMAL,
                metadata={
                    "provider": type(self).__name__,
                    "title": title,
                    "pid": int(pid.value),
                    "hwnd": int(hwnd),
                    "fake_capture": False,
                },
            ),
        )


@dataclass(slots=True)
class LivePersonalityAwarenessProvider(VoiceAwarenessProvider):
    source: VoiceAwarenessSource = VoiceAwarenessSource.PERSONALITY

    def collect(self, request: VoiceAwarenessRequest) -> tuple[VoiceAwarenessFact, ...]:
        from jarvis.cognitive.personality import default_jarvis_personality

        profile = default_jarvis_personality()

        return (
            VoiceAwarenessFact(
                source=self.source,
                key="personality_profile",
                value=(
                    f"profile={profile}; "
                    "spoken_style=concise; "
                    "fixed_conversational_responses_allowed=false"
                ),
                confidence=1.0,
                priority=VoiceAwarenessPriority.HIGH,
                metadata={
                    "provider": type(self).__name__,
                    "profile_type": type(profile).__name__,
                },
            ),
        )


def build_live_voice_awareness_runtime() -> VoiceAwarenessRuntime:
    return VoiceAwarenessRuntime(
        providers=(
            LiveMemoryAwarenessProvider(),
            LiveEnvironmentAwarenessProvider(),
            LiveGoalAwarenessProvider(),
            LiveToolAwarenessProvider(),
            LivePersonalityAwarenessProvider(),
            VoiceResponseBoundaryAwarenessProvider(),
        )
    )
