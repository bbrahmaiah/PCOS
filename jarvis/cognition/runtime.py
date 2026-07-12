from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from jarvis.cognition.action_planning import (
    ToolActionPlan,
    ToolActionPlanner,
)
from jarvis.cognition.adapters import (
    CognitionAdapterResult,
    StreamingCognitionAdapter,
)
from jarvis.cognition.local_llm_adapter import LocalLLMAdapter
from jarvis.cognition.memory import (
    InMemoryShortTermMemoryStore,
    ShortTermMemoryKind,
    ShortTermMemoryPriority,
)
from jarvis.cognition.models import (
    CognitionFailure,
    CognitionRequest,
    CognitionResponse,
    CognitionRuntimePolicy,
    SpokenResponseStyle,
)
from jarvis.cognition.planning import (
    ResponsePlanner,
    ResponsePlanningDecision,
)
from jarvis.cognition.session_context import ConversationSessionStore
from jarvis.cognition.spoken_policy import SpokenDialoguePolicy
from jarvis.cognition.streaming import (
    StreamingTokenPipeline,
    StreamingTokenPipelineResult,
)
from jarvis.cognitive.mission_context import (
    MissionContextRuntime,
    mission_context_input_from_request,
)
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class CognitionRuntimeConfig:
    """
    Configuration for the assembled cognition runtime.

    Defaults are conservative:
    - no direct tools
    - memory lookup enabled
    - concise spoken responses
    - optional streaming
    """

    name: str = "cognition_runtime"
    enable_mission_context: bool = True
    enable_memory_enrichment: bool = True
    enable_action_planning: bool = True
    remember_user_turns: bool = False
    remember_assistant_turns: bool = False
    default_allow_tools: bool = False
    default_allow_memory_lookup: bool = True
    default_spoken_style: SpokenResponseStyle = SpokenResponseStyle.CONCISE

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(slots=True)
class CognitionRuntimeComponents:
    """
    Runtime dependencies.

    These are injected so the runtime remains testable and swappable.
    """

    adapter: StreamingCognitionAdapter
    session_store: ConversationSessionStore
    memory_store: InMemoryShortTermMemoryStore
    response_planner: ResponsePlanner
    action_planner: ToolActionPlanner
    spoken_policy: SpokenDialoguePolicy
    mission_context: MissionContextRuntime | None = None
    streaming_pipeline: StreamingTokenPipeline | None = None


@dataclass(frozen=True, slots=True)
class CognitionRuntimeTurnResult:
    """
    Result of one assembled cognition turn.
    """

    request: CognitionRequest
    enriched_request: CognitionRequest
    planning_decision: ResponsePlanningDecision
    action_plan: ToolActionPlan | None = None
    adapter_result: CognitionAdapterResult | None = None
    streaming_result: StreamingTokenPipelineResult | None = None
    response: CognitionResponse | None = None
    failure: CognitionFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.response is not None and self.failure is None

    @property
    def failed(self) -> bool:
        return self.failure is not None

    @property
    def streamed(self) -> bool:
        return self.streaming_result is not None


@dataclass(frozen=True, slots=True)
class CognitionRuntimeSnapshot:
    """
    Observable diagnostics for the assembled runtime.
    """

    name: str
    turn_count: int
    success_count: int
    failure_count: int
    streamed_count: int
    action_planned_count: int
    last_request_id: str | None
    last_response_id: str | None
    last_failure_id: str | None
    last_error: str | None
    session_turn_count: int
    memory_item_count: int
    mission_context_update_count: int = 0
    mission_context_urgency: str | None = None
    mission_context_policy: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedRuntimeRequest:
    enriched_request: CognitionRequest
    planning_decision: ResponsePlanningDecision
    action_plan: ToolActionPlan | None


class CognitionRuntime:
    """
    Assembled cognition runtime.

    Responsibilities:
    - create cognition requests from user text
    - attach session context
    - attach short-term memory context
    - run response planning
    - run safe action planning when tools are allowed
    - call the local LLM adapter boundary
    - shape responses for voice
    - update session context
    - expose diagnostics

    Non-responsibilities:
    - no microphone/STT/TTS internals
    - no direct laptop control
    - no shell execution
    - no file operation execution
    - no long-term memory persistence
    """

    def __init__(
        self,
        *,
        components: CognitionRuntimeComponents,
        config: CognitionRuntimeConfig | None = None,
    ) -> None:
        self._config = config or CognitionRuntimeConfig()
        self._config.validate()

        self._components = components

        if self._components.mission_context is None:
            self._components.mission_context = MissionContextRuntime()

        if self._components.streaming_pipeline is None:
            self._components.streaming_pipeline = StreamingTokenPipeline(
                adapter=self._components.adapter,
            )

        self._lock = RLock()
        self._logger = get_logger("cognition.runtime")

        self._turn_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._streamed_count = 0
        self._action_planned_count = 0
        self._last_request_id: str | None = None
        self._last_response_id: str | None = None
        self._last_failure_id: str | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def components(self) -> CognitionRuntimeComponents:
        return self._components

    def process_text(
        self,
        text: str,
        *,
        request_id: str | None = None,
        turn_id: str | None = None,
        allow_tools: bool | None = None,
        allow_memory_lookup: bool | None = None,
        spoken_style: SpokenResponseStyle | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CognitionRuntimeTurnResult:
        """
        Process one non-streaming cognition text turn.
        """

        request = self._make_request(
            text=text,
            request_id=request_id,
            turn_id=turn_id,
            allow_tools=allow_tools,
            allow_memory_lookup=allow_memory_lookup,
            streaming_enabled=False,
            spoken_style=spoken_style,
            metadata=metadata,
        )

        prepared = self._prepare_request(request)
        adapter_result = self._components.adapter.generate(
            prepared.enriched_request,
        )

        if adapter_result.response is not None:
            response = self._shape_and_store_response(
                adapter_result.response,
                prepared.enriched_request,
            )
            self._record_success(
                request_id=request.request_id,
                response_id=response.response_id,
                streamed=False,
                action_planned=prepared.action_plan is not None,
            )

            return CognitionRuntimeTurnResult(
                request=request,
                enriched_request=prepared.enriched_request,
                planning_decision=prepared.planning_decision,
                action_plan=prepared.action_plan,
                adapter_result=adapter_result,
                response=response,
            )

        failure = adapter_result.failure
        self._record_failure_turn(
            request=request,
            failure=failure,
            action_planned=prepared.action_plan is not None,
        )

        return CognitionRuntimeTurnResult(
            request=request,
            enriched_request=prepared.enriched_request,
            planning_decision=prepared.planning_decision,
            action_plan=prepared.action_plan,
            adapter_result=adapter_result,
            failure=failure,
        )

    def process_text_streaming(
        self,
        text: str,
        *,
        request_id: str | None = None,
        turn_id: str | None = None,
        allow_tools: bool | None = None,
        allow_memory_lookup: bool | None = None,
        spoken_style: SpokenResponseStyle | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CognitionRuntimeTurnResult:
        """
        Process one streaming cognition text turn.

        This prepares speak-early behavior. Live TTS chunking comes later.
        """

        request = self._make_request(
            text=text,
            request_id=request_id,
            turn_id=turn_id,
            allow_tools=allow_tools,
            allow_memory_lookup=allow_memory_lookup,
            streaming_enabled=True,
            spoken_style=spoken_style,
            metadata=metadata,
        )

        prepared = self._prepare_request(request)
        streaming_pipeline = self._components.streaming_pipeline

        if streaming_pipeline is None:
            raise RuntimeError("streaming pipeline is not configured.")

        streaming_result = streaming_pipeline.stream_request(
            prepared.enriched_request,
        )

        if streaming_result.response is not None:
            response = self._shape_and_store_response(
                streaming_result.response,
                prepared.enriched_request,
            )
            self._record_success(
                request_id=request.request_id,
                response_id=response.response_id,
                streamed=True,
                action_planned=prepared.action_plan is not None,
            )

            return CognitionRuntimeTurnResult(
                request=request,
                enriched_request=prepared.enriched_request,
                planning_decision=prepared.planning_decision,
                action_plan=prepared.action_plan,
                streaming_result=streaming_result,
                response=response,
            )

        failure = streaming_result.failure
        self._record_failure_turn(
            request=request,
            failure=failure,
            action_planned=prepared.action_plan is not None,
        )

        return CognitionRuntimeTurnResult(
            request=request,
            enriched_request=prepared.enriched_request,
            planning_decision=prepared.planning_decision,
            action_plan=prepared.action_plan,
            streaming_result=streaming_result,
            failure=failure,
        )

    def remember(
        self,
        text: str,
        *,
        kind: ShortTermMemoryKind = ShortTermMemoryKind.SESSION_FACT,
        priority: ShortTermMemoryPriority = ShortTermMemoryPriority.NORMAL,
    ) -> None:
        """
        Store one temporary runtime memory note.
        """

        self._components.memory_store.remember_text(
            text,
            kind=kind,
            priority=priority,
            source=self.name,
        )

    def snapshot(self) -> CognitionRuntimeSnapshot:
        """
        Return runtime diagnostics.
        """

        session_snapshot = self._components.session_store.snapshot()
        memory_snapshot = self._components.memory_store.snapshot()
        mission_snapshot = (
            self._components.mission_context.snapshot()
            if self._components.mission_context is not None
            else None
        )

        with self._lock:
            return CognitionRuntimeSnapshot(
                name=self.name,
                turn_count=self._turn_count,
                success_count=self._success_count,
                failure_count=self._failure_count,
                streamed_count=self._streamed_count,
                action_planned_count=self._action_planned_count,
                last_request_id=self._last_request_id,
                last_response_id=self._last_response_id,
                last_failure_id=self._last_failure_id,
                last_error=self._last_error,
                session_turn_count=session_snapshot.turn_count,
                memory_item_count=memory_snapshot.item_count,
                mission_context_update_count=(
                    mission_snapshot.update_count
                    if mission_snapshot is not None
                    else 0
                ),
                mission_context_urgency=(
                    mission_snapshot.state.urgency.value
                    if mission_snapshot is not None
                    else None
                ),
                mission_context_policy=(
                    mission_snapshot.state.interruption_policy.value
                    if mission_snapshot is not None
                    else None
                ),
            )

    def reset(self) -> None:
        """
        Reset runtime counters and attached runtime stores.
        """

        with self._lock:
            self._turn_count = 0
            self._success_count = 0
            self._failure_count = 0
            self._streamed_count = 0
            self._action_planned_count = 0
            self._last_request_id = None
            self._last_response_id = None
            self._last_failure_id = None
            self._last_error = None

        self._components.session_store.reset()
        self._components.memory_store.clear()
        if self._components.mission_context is not None:
            self._components.mission_context.clear()

        if self._components.streaming_pipeline is not None:
            self._components.streaming_pipeline.reset()

        self._logger.info("cognition_runtime_reset", runtime=self.name)

    def _prepare_request(
        self,
        request: CognitionRequest,
    ) -> _PreparedRuntimeRequest:
        self._components.session_store.add_user_request(request)

        if self._config.remember_user_turns:
            self._components.memory_store.remember_text(
                request.text,
                kind=ShortTermMemoryKind.SESSION_FACT,
                source=self.name,
            )

        enriched_request = self._components.session_store.enrich_request(request)

        if (
            self._config.enable_memory_enrichment
            and enriched_request.policy.allow_memory_lookup
        ):
            enriched_request = self._components.memory_store.enrich_request(
                enriched_request,
            )

        if (
            self._config.enable_mission_context
            and self._components.mission_context is not None
        ):
            self._components.mission_context.update(
                mission_context_input_from_request(enriched_request)
            )
            enriched_request = self._components.mission_context.enrich_request(
                enriched_request,
            )

        planning_decision = self._components.response_planner.plan(
            enriched_request,
        )

        action_plan: ToolActionPlan | None = None

        if self._should_plan_action(enriched_request, planning_decision):
            action_plan = self._components.action_planner.plan(enriched_request)

        return _PreparedRuntimeRequest(
            enriched_request=enriched_request,
            planning_decision=planning_decision,
            action_plan=action_plan,
        )

    def _shape_and_store_response(
        self,
        response: CognitionResponse,
        request: CognitionRequest,
    ) -> CognitionResponse:
        shaped = self._components.spoken_policy.apply_to_response(
            response,
            style=request.policy.spoken_style,
        )

        self._components.session_store.add_assistant_response(shaped)

        if self._config.remember_assistant_turns:
            self._components.memory_store.remember_text(
                shaped.text,
                kind=ShortTermMemoryKind.SESSION_FACT,
                source=self.name,
            )

        return shaped

    def _record_failure_turn(
        self,
        *,
        request: CognitionRequest,
        failure: CognitionFailure | None,
        action_planned: bool,
    ) -> None:
        if failure is not None:
            self._components.session_store.add_failure(failure)
            failure_id = failure.failure_id
            error = failure.message

        else:
            failure_id = None
            error = "unknown cognition runtime failure"

        with self._lock:
            self._turn_count += 1
            self._failure_count += 1
            self._last_request_id = request.request_id
            self._last_response_id = None
            self._last_failure_id = failure_id
            self._last_error = error

            if action_planned:
                self._action_planned_count += 1

        self._logger.error(
            "cognition_runtime_turn_failed",
            runtime=self.name,
            request_id=request.request_id,
            failure_id=failure_id,
            error=error,
        )

    def _record_success(
        self,
        *,
        request_id: str,
        response_id: str,
        streamed: bool,
        action_planned: bool,
    ) -> None:
        with self._lock:
            self._turn_count += 1
            self._success_count += 1
            self._last_request_id = request_id
            self._last_response_id = response_id
            self._last_failure_id = None
            self._last_error = None

            if streamed:
                self._streamed_count += 1

            if action_planned:
                self._action_planned_count += 1

        self._logger.info(
            "cognition_runtime_turn_completed",
            runtime=self.name,
            request_id=request_id,
            response_id=response_id,
            streamed=streamed,
            action_planned=action_planned,
        )

    def _should_plan_action(
        self,
        request: CognitionRequest,
        decision: ResponsePlanningDecision,
    ) -> bool:
        return (
            self._config.enable_action_planning
            and request.policy.allow_tools
            and decision.tool_planning_recommended
        )

    def _make_request(
        self,
        *,
        text: str,
        request_id: str | None,
        turn_id: str | None,
        allow_tools: bool | None,
        allow_memory_lookup: bool | None,
        streaming_enabled: bool,
        spoken_style: SpokenResponseStyle | None,
        metadata: dict[str, Any] | None,
    ) -> CognitionRequest:
        policy = CognitionRuntimePolicy(
            allow_tools=(
                self._config.default_allow_tools
                if allow_tools is None
                else allow_tools
            ),
            allow_memory_lookup=(
                self._config.default_allow_memory_lookup
                if allow_memory_lookup is None
                else allow_memory_lookup
            ),
            streaming_enabled=streaming_enabled,
            spoken_style=spoken_style or self._config.default_spoken_style,
        )

        kwargs: dict[str, Any] = {
            "text": text,
            "policy": policy,
            "metadata": metadata or {},
        }

        if request_id is not None:
            kwargs["request_id"] = request_id

        if turn_id is not None:
            kwargs["turn_id"] = turn_id

        return CognitionRequest(**kwargs)


def create_cognition_runtime(
    *,
    adapter: LocalLLMAdapter,
    config: CognitionRuntimeConfig | None = None,
) -> CognitionRuntime:
    """
    Factory for the default assembled cognition runtime.
    """

    components = CognitionRuntimeComponents(
        adapter=adapter,
        session_store=ConversationSessionStore(),
        memory_store=InMemoryShortTermMemoryStore(),
        response_planner=ResponsePlanner(),
        action_planner=ToolActionPlanner(),
        spoken_policy=SpokenDialoguePolicy(),
        mission_context=MissionContextRuntime(),
    )

    return CognitionRuntime(
        components=components,
        config=config,
    )
