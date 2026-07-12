from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from urllib import request as http_request
from urllib.error import URLError

from jarvis.live import (
    LiveResponse,
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveResponseGenerator,
    LiveSessionConfig,
    LiveSessionMode,
    LiveSessionRunner,
    LiveSessionRunnerConfig,
    LiveSessionRunnerResult,
    LiveSessionRunnerStatus,
    LiveSessionStateRuntime,
    LiveWakeEngagementPolicy,
    LiveWakeEngagementRuntime,
)
from jarvis.voice.contracts import (
    VoiceTranscript,
    VoiceTranscriptKind,
    utc_now,
)


class VoiceCognitionStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    THINKING = "thinking"
    IGNORED = "ignored"
    DEGRADED = "degraded"
    FAILED = "failed"


class VoiceCognitionOperation(StrEnum):
    PREPARE = "prepare"
    THINK_FROM_TRANSCRIPT = "think_from_transcript"
    PREFETCH_FROM_PARTIAL = "prefetch_from_partial"
    SNAPSHOT = "snapshot"


class VoiceCognitionTranscriptSafety(StrEnum):
    PREDICTION_ONLY = "prediction_only"
    SAFE_FOR_DIALOGUE = "safe_for_dialogue"
    SAFE_FOR_ACTION_CANDIDATE = "safe_for_action_candidate"
    NEEDS_CLARIFICATION = "needs_clarification"
    BLOCKED_FOR_ACTION = "blocked_for_action"


class VoiceCognitionContextKind(StrEnum):
    MEMORY = "memory"
    WORKING_MEMORY = "working_memory"
    ATTENTION = "attention"
    GOAL = "goal"
    PLANNING = "planning"
    PERSONALITY = "personality"
    ENVIRONMENT = "environment"
    DEVELOPER = "developer"
    SAFETY = "safety"


@dataclass(frozen=True, slots=True)
class VoiceCognitionLatencyBudget:
    prewarm_timeout_ms: int = 2_000
    context_build_budget_ms: int = 120
    response_budget_ms: int = 4_000
    total_budget_ms: int = 5_000
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prewarm_timeout_ms <= 0:
            raise ValueError("prewarm_timeout_ms must be positive.")
        if self.context_build_budget_ms <= 0:
            raise ValueError("context_build_budget_ms must be positive.")
        if self.response_budget_ms <= 0:
            raise ValueError("response_budget_ms must be positive.")
        if self.total_budget_ms <= 0:
            raise ValueError("total_budget_ms must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceOllamaGeneratorConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout_seconds: int = 45
    max_sentences: int = 3
    temperature: float = 0.35
    num_predict: int = 180
    keep_alive: str = "10m"
    prewarm_on_prepare: bool = True
    stream_response: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("ollama base_url cannot be empty.")
        if not self.model.strip():
            raise ValueError("ollama model cannot be empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("ollama timeout_seconds must be positive.")
        if self.max_sentences <= 0:
            raise ValueError("ollama max_sentences must be positive.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("ollama temperature must be 0..2.")
        if self.num_predict <= 0:
            raise ValueError("ollama num_predict must be positive.")
        if not self.keep_alive.strip():
            raise ValueError("ollama keep_alive cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceCognitionPolicy:
    min_dialogue_confidence: float = 0.45
    min_action_confidence: float = 0.70
    allow_partial_dialogue: bool = False
    allow_partial_actions: bool = False
    max_context_items_per_kind: int = 4
    max_context_item_chars: int = 420
    max_response_sentences: int = 3
    require_final_for_response: bool = True
    require_wake_word_when_sleeping: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_dialogue_confidence <= 1.0:
            raise ValueError("min_dialogue_confidence must be 0..1.")
        if not 0.0 <= self.min_action_confidence <= 1.0:
            raise ValueError("min_action_confidence must be 0..1.")
        if self.max_context_items_per_kind <= 0:
            raise ValueError("max_context_items_per_kind must be positive.")
        if self.max_context_item_chars <= 0:
            raise ValueError("max_context_item_chars must be positive.")
        if self.max_response_sentences <= 0:
            raise ValueError("max_response_sentences must be positive.")


@dataclass(frozen=True, slots=True)
class VoiceCognitionContextItem:
    kind: VoiceCognitionContextKind
    text: str
    confidence: float = 1.0
    source: str = "runtime"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("voice cognition context text cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("voice cognition context confidence must be 0..1.")
        if not self.source.strip():
            raise ValueError("voice cognition context source cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceCognitionContextPack:
    memory: tuple[str, ...] = ()
    working_memory: tuple[str, ...] = ()
    attention: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    planning: tuple[str, ...] = ()
    personality: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    developer: tuple[str, ...] = ()
    safety: tuple[str, ...] = ()
    latency_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("context pack latency_ms cannot be negative.")


@dataclass(frozen=True, slots=True)
class VoiceCognitionRequest:
    transcript: VoiceTranscript
    user_label: str = "Balu"
    assistant_name: str = "JARVIS"
    allow_action_candidate: bool = False
    partial_prediction_only: bool = True
    memory_context: tuple[str, ...] = ()
    working_memory_context: tuple[str, ...] = ()
    attention_context: tuple[str, ...] = ()
    goal_context: tuple[str, ...] = ()
    planning_context: tuple[str, ...] = ()
    personality_context: tuple[str, ...] = ()
    environment_context: tuple[str, ...] = ()
    developer_context: tuple[str, ...] = ()
    safety_context: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_label.strip():
            raise ValueError("voice cognition user_label cannot be empty.")
        if not self.assistant_name.strip():
            raise ValueError("voice cognition assistant_name cannot be empty.")


@dataclass(frozen=True, slots=True)
class VoiceCognitionPrefetchResult:
    accepted: bool
    transcript: VoiceTranscript
    predicted_text: str
    context_pack: VoiceCognitionContextPack
    safety: VoiceCognitionTranscriptSafety
    latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceCognitionResult:
    status: VoiceCognitionStatus
    operation: VoiceCognitionOperation
    transcript: VoiceTranscript
    response: LiveResponse | None
    context_pack: VoiceCognitionContextPack
    safety: VoiceCognitionTranscriptSafety
    message: str
    latency_ms: float
    context_latency_ms: float
    response_latency_ms: float
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return (
            self.status == VoiceCognitionStatus.THINKING
            and self.response is not None
        )


@dataclass(frozen=True, slots=True)
class VoiceCognitionSnapshot:
    status: VoiceCognitionStatus
    prepared: bool
    responses: int
    ignored: int
    degraded: int
    failed: int
    prefetches: int
    last_text: str | None
    last_response_text: str | None
    last_latency_ms: float | None
    last_context_latency_ms: float | None
    last_response_latency_ms: float | None
    last_safety: VoiceCognitionTranscriptSafety | None
    last_error: str | None
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


class VoiceCognitionContextProvider(Protocol):
    def build_context(
        self,
        request: VoiceCognitionRequest,
    ) -> tuple[VoiceCognitionContextItem, ...]:
        raise NotImplementedError


class EmptyVoiceCognitionContextProvider:
    def build_context(
        self,
        request: VoiceCognitionRequest,
    ) -> tuple[VoiceCognitionContextItem, ...]:
        return ()


class StaticVoiceCognitionContextProvider:
    def __init__(
        self,
        items: tuple[VoiceCognitionContextItem, ...],
    ) -> None:
        self._items = items

    def build_context(
        self,
        request: VoiceCognitionRequest,
    ) -> tuple[VoiceCognitionContextItem, ...]:
        return self._items


class VoiceOllamaResponseGenerator(LiveResponseGenerator):
    """
    Real Ollama-backed generator for voice cognition.

    It produces only a LiveResponseDraft. It does not speak, execute tools,
    access memory directly, or bypass Step 50 response boundary.
    """

    def __init__(
        self,
        config: VoiceOllamaGeneratorConfig | None = None,
    ) -> None:
        self._config = config or VoiceOllamaGeneratorConfig()
        self._prewarmed = False

    @property
    def prewarmed(self) -> bool:
        return self._prewarmed

    def prewarm(self) -> None:
        if not self._config.prewarm_on_prepare:
            return

        payload = {
            "model": self._config.model,
            "prompt": "Ready.",
            "stream": False,
            "keep_alive": self._config.keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": 1,
            },
        }
        self._call_ollama(payload)
        self._prewarmed = True

    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        prompt = _build_ollama_prompt(
            request=request,
            max_sentences=self._config.max_sentences,
        )
        started = time.perf_counter()
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": self._config.stream_response,
            "keep_alive": self._config.keep_alive,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.num_predict,
            },
        }
        streamed = self._config.stream_response
        stream_chunk_count = 0
        first_token_latency_ms: float | None = None
        if streamed:
            text, stream_chunk_count, first_token_latency_ms = self._stream_ollama(
                payload,
                started=started,
            )
        else:
            text = self._call_ollama(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0

        return LiveResponseDraft(
            text=_limit_sentences(text, self._config.max_sentences),
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={
                "provider": "ollama",
                "model": self._config.model,
                "latency_ms": latency_ms,
                "voice_cognition": True,
                "prewarmed": self._prewarmed,
                "streamed": streamed,
                "stream_chunk_count": stream_chunk_count,
                "first_token_latency_ms": first_token_latency_ms,
            },
        )

    def _call_ollama(self, payload: dict[str, object]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = http_request.Request(
            url=f"{self._config.base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with http_request.urlopen(
                req,
                timeout=self._config.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(
                "Ollama request failed. Confirm Ollama is running."
            ) from exc

        data = json.loads(raw)
        text = str(data.get("response", "")).strip()

        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        return text

    def _stream_ollama(
        self,
        payload: dict[str, object],
        *,
        started: float,
    ) -> tuple[str, int, float | None]:
        body = json.dumps(payload).encode("utf-8")
        req = http_request.Request(
            url=f"{self._config.base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        chunks: list[str] = []
        first_token_latency_ms: float | None = None

        try:
            with http_request.urlopen(
                req,
                timeout=self._config.timeout_seconds,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    part = str(item.get("response", ""))
                    if part:
                        if first_token_latency_ms is None:
                            first_token_latency_ms = (
                                time.perf_counter() - started
                            ) * 1000.0
                        chunks.append(part)
                    if item.get("done") is True:
                        break
        except URLError as exc:
            raise RuntimeError(
                "Ollama request failed. Confirm Ollama is running."
            ) from exc

        text = "".join(chunks).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty streamed response.")

        return text, len(chunks), first_token_latency_ms


class VoiceCognitionResponseRuntime:
    """
    Step 51E movie-level voice cognition bridge.

    This runtime routes final voice transcripts through Step 50:
    LiveSessionRunner -> Dialogue Runtime -> Response Boundary -> LiveResponse.

    It supports compact context packs, prefetch from partial transcripts,
    latency budgets, confidence policy, and generated responses only.
    """

    def __init__(
        self,
        *,
        runner: LiveSessionRunner | None = None,
        response_generator: LiveResponseGenerator | None = None,
        context_provider: VoiceCognitionContextProvider | None = None,
        policy: VoiceCognitionPolicy | None = None,
        latency_budget: VoiceCognitionLatencyBudget | None = None,
    ) -> None:
        self._policy = policy or VoiceCognitionPolicy()
        self._latency_budget = latency_budget or VoiceCognitionLatencyBudget()
        self._response_generator = (
            response_generator or VoiceOllamaResponseGenerator()
        )
        self._context_provider = (
            context_provider or EmptyVoiceCognitionContextProvider()
        )
        self._runner = runner
        self._status = VoiceCognitionStatus.CREATED
        self._prepared = False
        self._responses = 0
        self._ignored = 0
        self._degraded = 0
        self._failed = 0
        self._prefetches = 0
        self._last_text: str | None = None
        self._last_response_text: str | None = None
        self._last_latency_ms: float | None = None
        self._last_context_latency_ms: float | None = None
        self._last_response_latency_ms: float | None = None
        self._last_safety: VoiceCognitionTranscriptSafety | None = None
        self._last_error: str | None = None
        self._last_prefetch: VoiceCognitionPrefetchResult | None = None

    def prepare(
        self,
        *,
        user_label: str = "Balu",
        assistant_name: str = "JARVIS",
    ) -> None:
        started = time.perf_counter()

        if hasattr(self._response_generator, "prewarm"):
            try:
                self._response_generator.prewarm()
            except Exception as exc:
                self._status = VoiceCognitionStatus.DEGRADED
                self._degraded += 1
                self._last_error = f"prewarm failed: {exc}"

        if self._runner is None:
            session_config = LiveSessionConfig(
                mode=LiveSessionMode.REAL_VOICE,
                user_label=user_label,
                assistant_name=assistant_name,
                real_microphone_enabled=True,
                real_stt_enabled=True,
                real_tts_enabled=True,
            )
            live_state = LiveSessionStateRuntime(config=session_config)
            wake = LiveWakeEngagementRuntime(
                live_state=live_state,
                policy=LiveWakeEngagementPolicy(
                    wake_word=session_config.wake_word,
                    require_wake_word_when_sleeping=(
                        self._policy.require_wake_word_when_sleeping
                    ),
                ),
            )
            self._runner = LiveSessionRunner(
                config=LiveSessionRunnerConfig(
                    session_config=session_config,
                    auto_prepare_audio=False,
                    auto_health_check=False,
                    auto_recover=False,
                ),
                response_generator=self._response_generator,
                live_state=live_state,
                wake=wake,
            )

        result = self._runner.start()
        latency_ms = (time.perf_counter() - started) * 1000.0

        if result.status == LiveSessionRunnerStatus.FAILED:
            self._status = VoiceCognitionStatus.FAILED
            self._failed += 1
            self._last_error = result.reason
            self._last_latency_ms = latency_ms
            return

        self._status = VoiceCognitionStatus.READY
        self._prepared = True
        if self._last_error is None:
            self._last_error = None
        self._last_latency_ms = latency_ms

    def prefetch_from_partial(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionPrefetchResult:
        started = time.perf_counter()
        safety = _transcript_safety(
            transcript=request.transcript,
            allow_action_candidate=False,
            policy=self._policy,
        )
        context_pack = self._build_context_pack(request)
        latency_ms = (time.perf_counter() - started) * 1000.0

        accepted = request.transcript.kind == VoiceTranscriptKind.PARTIAL
        if accepted:
            self._prefetches += 1

        result = VoiceCognitionPrefetchResult(
            accepted=accepted,
            transcript=request.transcript,
            predicted_text=request.transcript.text,
            context_pack=context_pack,
            safety=safety,
            latency_ms=latency_ms,
            created_at=utc_now(),
            metadata={
                "prediction_only": True,
                "safe_for_action": False,
            },
        )
        self._last_prefetch = result
        return result

    def think_from_transcript(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionResult:
        started = time.perf_counter()
        self._last_text = request.transcript.text

        safety = _transcript_safety(
            transcript=request.transcript,
            allow_action_candidate=request.allow_action_candidate,
            policy=self._policy,
        )
        self._last_safety = safety

        if safety in {
            VoiceCognitionTranscriptSafety.PREDICTION_ONLY,
            VoiceCognitionTranscriptSafety.NEEDS_CLARIFICATION,
            VoiceCognitionTranscriptSafety.BLOCKED_FOR_ACTION,
        }:
            self._ignored += 1
            self._status = VoiceCognitionStatus.IGNORED
            return self._ignored_result(
                request=request,
                started=started,
                safety=safety,
                message="transcript not eligible for response generation",
            )

        context_started = time.perf_counter()
        context_pack = self._build_context_pack(request)
        context_latency_ms = (time.perf_counter() - context_started) * 1000.0
        self._last_context_latency_ms = context_latency_ms

        if context_latency_ms > self._latency_budget.context_build_budget_ms:
            self._degraded += 1
            self._status = VoiceCognitionStatus.DEGRADED

        if not self._prepared:
            self.prepare(
                user_label=request.user_label,
                assistant_name=request.assistant_name,
            )

        if self._runner is None:
            self._failed += 1
            self._status = VoiceCognitionStatus.FAILED
            return self._failure(
                request=request,
                started=started,
                context_pack=context_pack,
                safety=safety,
                message="voice cognition runner unavailable",
            )

        response_started = time.perf_counter()
        try:
            result = self._runner.ingest_text(
                text=request.transcript.text,
                speech_probability=0.99,
                confidence=request.transcript.confidence,
                metadata={
                    "source": "voice_transcript",
                    "transcript_id": str(request.transcript.transcript_id),
                    "memory": "\n".join(context_pack.memory),
                    "working_memory": "\n".join(context_pack.working_memory),
                    "attention": "\n".join(context_pack.attention),
                    "goal": "\n".join(context_pack.goals),
                    "planning": "\n".join(context_pack.planning),
                    "personality": "\n".join(context_pack.personality),
                    "environment": "\n".join(context_pack.environment),
                    "developer": "\n".join(context_pack.developer),
                    "safety": "\n".join(context_pack.safety),
                    **request.metadata,
                },
            )
        except Exception as exc:
            self._failed += 1
            self._status = VoiceCognitionStatus.FAILED
            self._last_error = str(exc)
            return self._failure(
                request=request,
                started=started,
                context_pack=context_pack,
                safety=safety,
                message="voice cognition failed",
                metadata={"error": str(exc)},
            )

        response_latency_ms = (time.perf_counter() - response_started) * 1000.0
        self._last_response_latency_ms = response_latency_ms

        response = None
        if (
            result.dialogue_result is not None
            and result.dialogue_result.turn is not None
        ):
            response = result.dialogue_result.turn.response

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms

        if response is None:
            self._ignored += 1
            self._status = VoiceCognitionStatus.IGNORED
            return VoiceCognitionResult(
                status=self._status,
                operation=VoiceCognitionOperation.THINK_FROM_TRANSCRIPT,
                transcript=request.transcript,
                response=None,
                context_pack=context_pack,
                safety=safety,
                message="voice transcript produced no live response",
                latency_ms=latency_ms,
                context_latency_ms=context_latency_ms,
                response_latency_ms=response_latency_ms,
                created_at=utc_now(),
                metadata=_runner_diagnostic_metadata(result),
            )

        self._responses += 1
        self._status = VoiceCognitionStatus.THINKING
        self._last_response_text = response.text
        self._last_error = None

        if latency_ms > self._latency_budget.total_budget_ms:
            self._degraded += 1

        return VoiceCognitionResult(
            status=self._status,
            operation=VoiceCognitionOperation.THINK_FROM_TRANSCRIPT,
            transcript=request.transcript,
            response=response,
            context_pack=context_pack,
            safety=safety,
            message="voice cognition response produced",
            latency_ms=latency_ms,
            context_latency_ms=context_latency_ms,
            response_latency_ms=response_latency_ms,
            created_at=utc_now(),
            metadata={
                **_runner_diagnostic_metadata(result),
                "response_id": str(response.response_id),
                "within_total_budget": (
                    latency_ms <= self._latency_budget.total_budget_ms
                ),
                "within_response_budget": (
                    response_latency_ms <= self._latency_budget.response_budget_ms
                ),
            },
        )

    def snapshot(self) -> VoiceCognitionSnapshot:
        return VoiceCognitionSnapshot(
            status=self._status,
            prepared=self._prepared,
            responses=self._responses,
            ignored=self._ignored,
            degraded=self._degraded,
            failed=self._failed,
            prefetches=self._prefetches,
            last_text=self._last_text,
            last_response_text=self._last_response_text,
            last_latency_ms=self._last_latency_ms,
            last_context_latency_ms=self._last_context_latency_ms,
            last_response_latency_ms=self._last_response_latency_ms,
            last_safety=self._last_safety,
            last_error=self._last_error,
            created_at=utc_now(),
        )

    def _build_context_pack(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionContextPack:
        started = time.perf_counter()
        provider_items = self._context_provider.build_context(request)
        grouped = _group_context_items(
            items=provider_items,
            policy=self._policy,
        )

        memory = _merge_context(
            request.memory_context,
            grouped[VoiceCognitionContextKind.MEMORY],
            self._policy,
        )
        working_memory = _merge_context(
            request.working_memory_context,
            grouped[VoiceCognitionContextKind.WORKING_MEMORY],
            self._policy,
        )
        attention = _merge_context(
            request.attention_context,
            grouped[VoiceCognitionContextKind.ATTENTION],
            self._policy,
        )
        goals = _merge_context(
            request.goal_context,
            grouped[VoiceCognitionContextKind.GOAL],
            self._policy,
        )
        planning = _merge_context(
            request.planning_context,
            grouped[VoiceCognitionContextKind.PLANNING],
            self._policy,
        )
        personality = _merge_context(
            request.personality_context,
            grouped[VoiceCognitionContextKind.PERSONALITY],
            self._policy,
        )
        environment = _merge_context(
            request.environment_context,
            grouped[VoiceCognitionContextKind.ENVIRONMENT],
            self._policy,
        )
        developer = _merge_context(
            request.developer_context,
            grouped[VoiceCognitionContextKind.DEVELOPER],
            self._policy,
        )
        safety = _merge_context(
            request.safety_context,
            grouped[VoiceCognitionContextKind.SAFETY],
            self._policy,
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return VoiceCognitionContextPack(
            memory=memory,
            working_memory=working_memory,
            attention=attention,
            goals=goals,
            planning=planning,
            personality=personality,
            environment=environment,
            developer=developer,
            safety=safety,
            latency_ms=latency_ms,
            metadata={
                "compact": True,
                "provider_items": len(provider_items),
            },
        )

    def _ignored_result(
        self,
        *,
        request: VoiceCognitionRequest,
        started: float,
        safety: VoiceCognitionTranscriptSafety,
        message: str,
    ) -> VoiceCognitionResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms
        empty_pack = VoiceCognitionContextPack(latency_ms=0.0)
        return VoiceCognitionResult(
            status=self._status,
            operation=VoiceCognitionOperation.THINK_FROM_TRANSCRIPT,
            transcript=request.transcript,
            response=None,
            context_pack=empty_pack,
            safety=safety,
            message=message,
            latency_ms=latency_ms,
            context_latency_ms=0.0,
            response_latency_ms=0.0,
            created_at=utc_now(),
            metadata={
                "confidence": request.transcript.confidence,
                "kind": request.transcript.kind.value,
            },
        )

    def _failure(
        self,
        *,
        request: VoiceCognitionRequest,
        started: float,
        context_pack: VoiceCognitionContextPack,
        safety: VoiceCognitionTranscriptSafety,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> VoiceCognitionResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_latency_ms = latency_ms
        return VoiceCognitionResult(
            status=VoiceCognitionStatus.FAILED,
            operation=VoiceCognitionOperation.THINK_FROM_TRANSCRIPT,
            transcript=request.transcript,
            response=None,
            context_pack=context_pack,
            safety=safety,
            message=message,
            latency_ms=latency_ms,
            context_latency_ms=context_pack.latency_ms,
            response_latency_ms=0.0,
            created_at=utc_now(),
            metadata=metadata or {},
        )


def _transcript_safety(
    *,
    transcript: VoiceTranscript,
    allow_action_candidate: bool,
    policy: VoiceCognitionPolicy,
) -> VoiceCognitionTranscriptSafety:
    if transcript.kind == VoiceTranscriptKind.PARTIAL:
        if policy.require_final_for_response and not policy.allow_partial_dialogue:
            return VoiceCognitionTranscriptSafety.PREDICTION_ONLY
        if allow_action_candidate or policy.allow_partial_actions:
            return VoiceCognitionTranscriptSafety.BLOCKED_FOR_ACTION

    if transcript.confidence < policy.min_dialogue_confidence:
        return VoiceCognitionTranscriptSafety.NEEDS_CLARIFICATION

    if allow_action_candidate:
        if transcript.confidence < policy.min_action_confidence:
            return VoiceCognitionTranscriptSafety.BLOCKED_FOR_ACTION
        return VoiceCognitionTranscriptSafety.SAFE_FOR_ACTION_CANDIDATE

    return VoiceCognitionTranscriptSafety.SAFE_FOR_DIALOGUE


def _group_context_items(
    *,
    items: tuple[VoiceCognitionContextItem, ...],
    policy: VoiceCognitionPolicy,
) -> dict[VoiceCognitionContextKind, tuple[str, ...]]:
    grouped: dict[VoiceCognitionContextKind, list[str]] = {
        kind: [] for kind in VoiceCognitionContextKind
    }

    sorted_items = sorted(
        items,
        key=lambda item: item.confidence,
        reverse=True,
    )

    for item in sorted_items:
        bucket = grouped[item.kind]
        if len(bucket) >= policy.max_context_items_per_kind:
            continue
        bucket.append(_compact_text(item.text, policy.max_context_item_chars))

    return {
        kind: tuple(values)
        for kind, values in grouped.items()
    }


def _merge_context(
    direct_items: tuple[str, ...],
    provider_items: tuple[str, ...],
    policy: VoiceCognitionPolicy,
) -> tuple[str, ...]:
    merged: list[str] = []
    for item in (*direct_items, *provider_items):
        compact = _compact_text(item, policy.max_context_item_chars)
        if compact and compact not in merged:
            merged.append(compact)
        if len(merged) >= policy.max_context_items_per_kind:
            break
    return tuple(merged)


def _compact_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _runner_diagnostic_metadata(
    result: LiveSessionRunnerResult,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "runner_status": result.status.value,
        "runner_reason": result.reason,
    }

    if result.wake_result is not None:
        metadata.update(
            {
                "wake_status": result.wake_result.status.value,
                "wake_decision": result.wake_result.decision.value,
                "wake_reason": result.wake_result.reason.value,
                "wake_engaged": result.wake_result.engaged,
            }
        )

    return metadata


def _build_ollama_prompt(
    *,
    request: LiveResponseGenerationRequest,
    max_sentences: int,
) -> str:
    context = request.context
    state = context.live_state

    memory = "\n".join(context.memory_context) or "No retrieved memory."
    working_memory = (
        "\n".join(context.working_memory_context)
        or "No active working memory."
    )
    attention = "\n".join(context.attention_context) or "No attention signal."
    goals = "\n".join(context.goal_context) or "No active goal context."
    planning = "\n".join(context.planning_context) or "No active plan."
    environment = (
        "\n".join(context.environment_context)
        or "No environment context."
    )
    developer = "\n".join(context.developer_context) or "No developer context."

    return f"""
You are {state.assistant_name}, Balu's local voice-first cognitive assistant.

You are not a chatbot.
You are a calm, concise, loyal, technically sharp executive assistant.
You must respond naturally for spoken conversation.
Keep the response to at most {max_sentences} short sentences unless asked.
Prefer one sentence for tiny follow-ups or status checks.
Be fast, direct, and useful.
Ask one clarifying question if the instruction is incomplete.
If the user interrupts with a smaller question, answer that question first.
If the user asks whether you can hear them, answer from the fact that this
transcript reached you; do not claim you lack audio awareness.
Do not claim an action was performed unless the runtime actually did it.
Do not invent memory, files, screen state, tool results, or system status.
Do not mention transcripts, prompts, runtime internals, or boundaries unless asked.
Do not output markdown tables for spoken replies.

User said:
{context.user_text}

Situation:
{context.situation_summary}

Memory:
{memory}

Working memory:
{working_memory}

Attention:
{attention}

Goals:
{goals}

Planning:
{planning}

Environment:
{environment}

Developer context:
{developer}

Generate the next natural spoken response.
""".strip()


def _limit_sentences(text: str, max_sentences: int) -> str:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return normalized

    sentences: list[str] = []
    current = ""

    for char in normalized:
        current += char
        if char in {".", "?", "!"}:
            sentences.append(current.strip())
            current = ""
            if len(sentences) >= max_sentences:
                break

    if not sentences:
        return normalized

    return " ".join(sentences)
