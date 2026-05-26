from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import (
    TaskId,
    WorkerId,
    utc_now,
    validate_task_id,
    validate_worker_id,
)
from jarvis.orchestration.models import (
    OrchestrationModel,
    TaskPriority,
)


def new_wait_edge_id() -> str:
    return f"waitedge_{uuid4().hex}"


def new_deadlock_id() -> str:
    return f"deadlock_{uuid4().hex}"


class ResourceLockKind(StrEnum):
    """
    Canonical resource lock classes.

    Workers must acquire locks in canonical order.
    """

    ATTENTION = "attention"
    CONTEXT = "context"
    MEMORY = "memory"
    FILE_SYSTEM = "file_system"
    BROWSER = "browser"
    IDE = "ide"
    TOOL = "tool"
    WORKER = "worker"
    BACKGROUND = "background"


class DeadlockDetectionDecision(StrEnum):
    """
    Deadlock detector decision.
    """

    CLEAR = "clear"
    DETECTED = "detected"
    REJECTED = "rejected"
    RESOLVED = "resolved"


class DeadlockDetectionReason(StrEnum):
    """
    Machine-readable deadlock reason.
    """

    NO_DEADLOCK = "no_deadlock"
    CYCLE_DETECTED = "cycle_detected"
    LOCK_ORDER_VALID = "lock_order_valid"
    LOCK_ORDER_VIOLATION = "lock_order_violation"
    WAIT_EDGE_ADDED = "wait_edge_added"
    WAIT_EDGE_REMOVED = "wait_edge_removed"
    WAIT_EDGE_NOT_FOUND = "wait_edge_not_found"
    WAIT_TIMEOUT_DETECTED = "wait_timeout_detected"
    DEADLOCK_RESOLVED_BY_CANCELLING_LOWEST_PRIORITY = (
        "deadlock_resolved_by_cancelling_lowest_priority"
    )
    DEADLOCK_NOT_FOUND = "deadlock_not_found"


class DeadlockResolutionAction(StrEnum):
    """
    Resolution action proposed by the deadlock resolver.
    """

    NONE = "none"
    CANCEL_AND_RETRY_TASK = "cancel_and_retry_task"
    TIMEOUT_WAIT = "timeout_wait"
    ESCALATE = "escalate"


class ResourceLockOrdering(IntEnum):
    """
    Canonical lock ordering.

    Lower value must be acquired before higher value.
    """

    ATTENTION = 0
    CONTEXT = 10
    MEMORY = 20
    FILE_SYSTEM = 30
    BROWSER = 40
    IDE = 50
    TOOL = 60
    WORKER = 70
    BACKGROUND = 80


_LOCK_KIND_ORDER: dict[ResourceLockKind, ResourceLockOrdering] = {
    ResourceLockKind.ATTENTION: ResourceLockOrdering.ATTENTION,
    ResourceLockKind.CONTEXT: ResourceLockOrdering.CONTEXT,
    ResourceLockKind.MEMORY: ResourceLockOrdering.MEMORY,
    ResourceLockKind.FILE_SYSTEM: ResourceLockOrdering.FILE_SYSTEM,
    ResourceLockKind.BROWSER: ResourceLockOrdering.BROWSER,
    ResourceLockKind.IDE: ResourceLockOrdering.IDE,
    ResourceLockKind.TOOL: ResourceLockOrdering.TOOL,
    ResourceLockKind.WORKER: ResourceLockOrdering.WORKER,
    ResourceLockKind.BACKGROUND: ResourceLockOrdering.BACKGROUND,
}


class ResourceLockRequest(OrchestrationModel):
    """
    Ordered lock acquisition request.

    Prevention rule:
    all requested locks must already be in canonical order.
    """

    task_id: TaskId
    worker_id: WorkerId
    locks: tuple[ResourceLockKind, ...]
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @model_validator(mode="after")
    def _validate_locks(self) -> ResourceLockRequest:
        if not self.locks:
            raise ValueError("lock request requires at least one lock.")

        return self

    @property
    def ordered(self) -> bool:
        values = [_LOCK_KIND_ORDER[item] for item in self.locks]

        return values == sorted(values)


class WaitEdge(OrchestrationModel):
    """
    Directed wait edge.

    waiting_worker_id waits for blocking_worker_id through a resource/task.
    """

    edge_id: str = Field(default_factory=new_wait_edge_id)
    waiting_worker_id: WorkerId
    blocking_worker_id: WorkerId
    waiting_task_id: TaskId
    blocking_task_id: TaskId
    resource: ResourceLockKind
    waiting_task_priority: TaskPriority
    blocking_task_priority: TaskPriority
    created_at: object = Field(default_factory=utc_now)
    timeout_ms: int = Field(default=2_000, gt=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("edge_id")
    @classmethod
    def _validate_edge_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("edge_id cannot be empty.")

        if not cleaned.startswith("waitedge_"):
            raise ValueError("edge_id must start with 'waitedge_'.")

        return cleaned

    @field_validator("waiting_worker_id", "blocking_worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return validate_worker_id(value)

    @field_validator("waiting_task_id", "blocking_task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @model_validator(mode="after")
    def _validate_edge(self) -> WaitEdge:
        if self.waiting_worker_id == self.blocking_worker_id:
            raise ValueError("worker cannot wait on itself.")

        if self.waiting_task_id == self.blocking_task_id:
            raise ValueError("task cannot wait on itself.")

        return self


class WaitGraphSnapshot(OrchestrationModel):
    """
    Immutable wait graph snapshot.
    """

    edges: tuple[WaitEdge, ...] = ()
    created_at: object = Field(default_factory=utc_now)

    @property
    def worker_ids(self) -> set[WorkerId]:
        found: set[WorkerId] = set()

        for edge in self.edges:
            found.add(edge.waiting_worker_id)
            found.add(edge.blocking_worker_id)

        return found

    @property
    def task_ids(self) -> set[TaskId]:
        found: set[TaskId] = set()

        for edge in self.edges:
            found.add(edge.waiting_task_id)
            found.add(edge.blocking_task_id)

        return found


class WaitGraph(OrchestrationModel):
    """
    Directed graph of worker-to-worker waits.
    """

    edges: tuple[WaitEdge, ...] = ()

    def add_edge(self, edge: WaitEdge) -> WaitGraph:
        if any(existing.edge_id == edge.edge_id for existing in self.edges):
            raise ValueError("wait edge already exists.")

        return self.model_copy(update={"edges": self.edges + (edge,)})

    def remove_edge(self, edge_id: str) -> WaitGraph:
        next_edges = tuple(edge for edge in self.edges if edge.edge_id != edge_id)

        if len(next_edges) == len(self.edges):
            raise ValueError("wait edge not found.")

        return self.model_copy(update={"edges": next_edges})

    def snapshot(self) -> WaitGraphSnapshot:
        return WaitGraphSnapshot(edges=self.edges)

    def edges_for_worker(self, worker_id: WorkerId) -> tuple[WaitEdge, ...]:
        validated_worker_id = validate_worker_id(worker_id)

        return tuple(
            edge
            for edge in self.edges
            if edge.waiting_worker_id == validated_worker_id
            or edge.blocking_worker_id == validated_worker_id
        )

    def find_cycles(self) -> tuple[tuple[WaitEdge, ...], ...]:
        """
        Find cycles in worker wait graph.

        Returns cycles as edge tuples.
        """

        adjacency: dict[str, list[WaitEdge]] = {}

        for edge in self.edges:
            adjacency.setdefault(edge.waiting_worker_id, []).append(edge)

        cycles: list[tuple[WaitEdge, ...]] = []

        def visit(
            worker_id: str,
            path_workers: list[str],
            path_edges: list[WaitEdge],
        ) -> None:
            for edge in adjacency.get(worker_id, []):
                next_worker = edge.blocking_worker_id

                if next_worker in path_workers:
                    start = path_workers.index(next_worker)
                    cycle_edges = tuple(path_edges[start:] + [edge])
                    cycle_key = tuple(sorted(item.edge_id for item in cycle_edges))

                    if not any(
                        tuple(sorted(item.edge_id for item in found)) == cycle_key
                        for found in cycles
                    ):
                        cycles.append(cycle_edges)

                    continue

                visit(
                    next_worker,
                    path_workers + [next_worker],
                    path_edges + [edge],
                )

        for worker_id in adjacency:
            visit(worker_id, [worker_id], [])

        return tuple(cycles)


class DeadlockEvent(OrchestrationModel):
    """
    Detected deadlock event with full wait graph evidence.
    """

    deadlock_id: str = Field(default_factory=new_deadlock_id)
    cycle: tuple[WaitEdge, ...]
    wait_graph_snapshot: WaitGraphSnapshot
    detected_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("deadlock_id")
    @classmethod
    def _validate_deadlock_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("deadlock_id cannot be empty.")

        if not cleaned.startswith("deadlock_"):
            raise ValueError("deadlock_id must start with 'deadlock_'.")

        return cleaned

    @model_validator(mode="after")
    def _validate_event(self) -> DeadlockEvent:
        if not self.cycle:
            raise ValueError("deadlock event requires cycle edges.")

        return self

    @property
    def worker_ids(self) -> tuple[WorkerId, ...]:
        found: list[WorkerId] = []

        for edge in self.cycle:
            if edge.waiting_worker_id not in found:
                found.append(edge.waiting_worker_id)

            if edge.blocking_worker_id not in found:
                found.append(edge.blocking_worker_id)

        return tuple(found)

    @property
    def task_ids(self) -> tuple[TaskId, ...]:
        found: list[TaskId] = []

        for edge in self.cycle:
            if edge.waiting_task_id not in found:
                found.append(edge.waiting_task_id)

            if edge.blocking_task_id not in found:
                found.append(edge.blocking_task_id)

        return tuple(found)


class DeadlockResolution(OrchestrationModel):
    """
    Resolution proposal for a detected deadlock.
    """

    deadlock_id: str
    action: DeadlockResolutionAction
    task_id: TaskId | None = None
    worker_id: WorkerId | None = None
    reason: DeadlockDetectionReason
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("deadlock_id")
    @classmethod
    def _validate_deadlock_id(cls, value: str) -> str:
        if not value.startswith("deadlock_"):
            raise ValueError("deadlock_id must start with 'deadlock_'.")

        return value

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_task_id(value)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return validate_worker_id(value)


class DeadlockDetectionResult(OrchestrationModel):
    """
    Result from detector/prevention/resolution operations.
    """

    decision: DeadlockDetectionDecision
    reason: DeadlockDetectionReason
    success: bool
    message: str
    graph: WaitGraph | None = None
    event: DeadlockEvent | None = None
    resolution: DeadlockResolution | None = None
    lock_request: ResourceLockRequest | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _required_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("message cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class DeadlockDetectorConfig:
    """
    Deadlock detector configuration.
    """

    name: str = "deadlock_detector"
    max_wait_ms: int = 2_000

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be positive.")


@dataclass(frozen=True, slots=True)
class DeadlockDetectorSnapshot:
    """
    Deadlock detector diagnostics.
    """

    name: str
    wait_edge_count: int
    detected_count: int
    resolved_count: int
    rejected_count: int
    timeout_count: int
    last_reason: DeadlockDetectionReason | None


class DeadlockResolver:
    """
    Deadlock resolution policy.

    Resolution:
    cancel and retry the lowest-priority task in the cycle.
    """

    @staticmethod
    def resolve(event: DeadlockEvent) -> DeadlockResolution:
        victim = DeadlockResolver._victim_edge(event.cycle)

        return DeadlockResolution(
            deadlock_id=event.deadlock_id,
            action=DeadlockResolutionAction.CANCEL_AND_RETRY_TASK,
            task_id=victim.waiting_task_id,
            worker_id=victim.waiting_worker_id,
            reason=(
                DeadlockDetectionReason
                .DEADLOCK_RESOLVED_BY_CANCELLING_LOWEST_PRIORITY
            ),
            metadata={
                "victim_priority": victim.waiting_task_priority.value,
                "cycle_edge_count": len(event.cycle),
            },
        )

    @staticmethod
    def _victim_edge(cycle: tuple[WaitEdge, ...]) -> WaitEdge:
        return sorted(
            cycle,
            key=lambda edge: (
                int(edge.waiting_task_priority) * -1,
                edge.created_at,
                edge.edge_id,
            ),
        )[0]


class DeadlockDetector:
    """
    Phase 6 Deadlock Detection & Prevention.

    Responsibilities:
    - maintain directed wait graph
    - enforce canonical lock ordering
    - detect circular waits
    - detect waits exceeding timeout
    - propose deterministic resolution
    - log full wait graph snapshot with each event

    Non-responsibilities:
    - no direct task execution
    - no direct worker cancellation
    - no rollback execution
    """

    def __init__(
        self,
        *,
        config: DeadlockDetectorConfig | None = None,
        resolver: DeadlockResolver | None = None,
    ) -> None:
        self._config = config or DeadlockDetectorConfig()
        self._config.validate()

        self._resolver = resolver or DeadlockResolver()
        self._graph = WaitGraph()
        self._events: list[DeadlockEvent] = []
        self._resolutions: list[DeadlockResolution] = []
        self._lock = RLock()

        self._detected_count = 0
        self._resolved_count = 0
        self._rejected_count = 0
        self._timeout_count = 0
        self._last_reason: DeadlockDetectionReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def validate_lock_order(
        self,
        request: ResourceLockRequest,
    ) -> DeadlockDetectionResult:
        """
        Enforce canonical alphabetical/stable lock ordering.
        """

        if request.ordered:
            result = DeadlockDetectionResult(
                decision=DeadlockDetectionDecision.CLEAR,
                reason=DeadlockDetectionReason.LOCK_ORDER_VALID,
                success=True,
                message="lock order is valid",
                graph=self._graph,
                lock_request=request,
            )
            self._record(result)

            return result

        result = DeadlockDetectionResult(
            decision=DeadlockDetectionDecision.REJECTED,
            reason=DeadlockDetectionReason.LOCK_ORDER_VIOLATION,
            success=False,
            message="lock acquisition order violates canonical ordering",
            graph=self._graph,
            lock_request=request,
        )
        self._record(result)

        return result

    def add_wait(self, edge: WaitEdge) -> DeadlockDetectionResult:
        """
        Add a wait edge and immediately detect cycles.
        """

        with self._lock:
            self._graph = self._graph.add_edge(edge)
            graph = self._graph

        detection = self.detect()

        if detection.event is not None:
            return detection

        result = DeadlockDetectionResult(
            decision=DeadlockDetectionDecision.CLEAR,
            reason=DeadlockDetectionReason.WAIT_EDGE_ADDED,
            success=True,
            message="wait edge added",
            graph=graph,
        )
        self._record(result)

        return result

    def remove_wait(self, edge_id: str) -> DeadlockDetectionResult:
        """
        Remove a wait edge.
        """

        with self._lock:
            try:
                self._graph = self._graph.remove_edge(edge_id)
            except ValueError:
                result = DeadlockDetectionResult(
                    decision=DeadlockDetectionDecision.REJECTED,
                    reason=DeadlockDetectionReason.WAIT_EDGE_NOT_FOUND,
                    success=False,
                    message="wait edge not found",
                    graph=self._graph,
                )
                self._record(result)

                return result

            result = DeadlockDetectionResult(
                decision=DeadlockDetectionDecision.CLEAR,
                reason=DeadlockDetectionReason.WAIT_EDGE_REMOVED,
                success=True,
                message="wait edge removed",
                graph=self._graph,
            )
            self._record(result)

            return result

    def detect(self) -> DeadlockDetectionResult:
        """
        Detect circular waits in the current graph.
        """

        with self._lock:
            graph = self._graph
            cycles = graph.find_cycles()

            if not cycles:
                result = DeadlockDetectionResult(
                    decision=DeadlockDetectionDecision.CLEAR,
                    reason=DeadlockDetectionReason.NO_DEADLOCK,
                    success=True,
                    message="no deadlock detected",
                    graph=graph,
                )
                self._record(result)

                return result

            event = DeadlockEvent(
                cycle=cycles[0],
                wait_graph_snapshot=graph.snapshot(),
            )
            self._events.append(event)
            self._detected_count += 1

            result = DeadlockDetectionResult(
                decision=DeadlockDetectionDecision.DETECTED,
                reason=DeadlockDetectionReason.CYCLE_DETECTED,
                success=False,
                message="deadlock cycle detected",
                graph=graph,
                event=event,
            )
            self._record(result)

            return result

    def resolve(self, deadlock_id: str) -> DeadlockDetectionResult:
        """
        Resolve a detected deadlock by selecting the lowest-priority task.
        """

        with self._lock:
            event = next(
                (item for item in self._events if item.deadlock_id == deadlock_id),
                None,
            )

            if event is None:
                result = DeadlockDetectionResult(
                    decision=DeadlockDetectionDecision.REJECTED,
                    reason=DeadlockDetectionReason.DEADLOCK_NOT_FOUND,
                    success=False,
                    message="deadlock not found",
                    graph=self._graph,
                )
                self._record(result)

                return result

            resolution = self._resolver.resolve(event)
            self._resolutions.append(resolution)
            self._resolved_count += 1

            result = DeadlockDetectionResult(
                decision=DeadlockDetectionDecision.RESOLVED,
                reason=resolution.reason,
                success=True,
                message="deadlock resolution proposed",
                graph=self._graph,
                event=event,
                resolution=resolution,
            )
            self._record(result)

            return result

    def detect_timeouts(self) -> tuple[DeadlockDetectionResult, ...]:
        """
        Detect wait edges exceeding max wait.
        """

        now = utc_now()
        results: list[DeadlockDetectionResult] = []

        with self._lock:
            edges = self._graph.edges

        for edge in edges:
            if not self._edge_timed_out(edge=edge, now=now):
                continue

            result = DeadlockDetectionResult(
                decision=DeadlockDetectionDecision.DETECTED,
                reason=DeadlockDetectionReason.WAIT_TIMEOUT_DETECTED,
                success=False,
                message="wait edge exceeded maximum wait timeout",
                graph=self._graph,
                metadata={
                    "edge_id": edge.edge_id,
                    "waiting_worker_id": edge.waiting_worker_id,
                    "blocking_worker_id": edge.blocking_worker_id,
                    "timeout_ms": edge.timeout_ms,
                },
            )
            self._timeout_count += 1
            self._record(result)
            results.append(result)

        return tuple(results)

    def graph(self) -> WaitGraph:
        with self._lock:
            return self._graph

    def events(self) -> tuple[DeadlockEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def resolutions(self) -> tuple[DeadlockResolution, ...]:
        with self._lock:
            return tuple(self._resolutions)

    def snapshot(self) -> DeadlockDetectorSnapshot:
        with self._lock:
            return DeadlockDetectorSnapshot(
                name=self.name,
                wait_edge_count=len(self._graph.edges),
                detected_count=self._detected_count,
                resolved_count=self._resolved_count,
                rejected_count=self._rejected_count,
                timeout_count=self._timeout_count,
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._graph = WaitGraph()
            self._events.clear()
            self._resolutions.clear()
            self._detected_count = 0
            self._resolved_count = 0
            self._rejected_count = 0
            self._timeout_count = 0
            self._last_reason = None

    @staticmethod
    def _edge_timed_out(*, edge: WaitEdge, now: object) -> bool:
        if not isinstance(now, datetime):
            return False

        if not isinstance(edge.created_at, datetime):
            return False

        elapsed = now - edge.created_at
        elapsed_ms = int(elapsed.total_seconds() * 1000)

        return elapsed_ms >= edge.timeout_ms

    def _record(self, result: DeadlockDetectionResult) -> None:
        self._last_reason = result.reason

        if result.decision == DeadlockDetectionDecision.REJECTED:
            self._rejected_count += 1