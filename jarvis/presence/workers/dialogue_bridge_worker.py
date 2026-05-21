from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from jarvis.presence.models import Transcript, TranscriptKind
from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import EventCategory, EventType
from jarvis.runtime.workers import BaseWorker


def new_dialogue_request_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class DialogueBridgePolicy:
    """
    Voice-first response policy.

    This is not cognition. It is a small contract that tells the future
    dialogue/cognition layer how to respond for natural voice interaction.
    """

    response_style: str = "concise_human"
    max_response_words_hint: int = 45
    allow_clarifying_question: bool = True
    interruptible: bool = True
    prefer_short_spoken_response: bool = True


@dataclass(frozen=True, slots=True)
class DialogueBridgeWorkerSnapshot:
    """
    Immutable diagnostic snapshot for DialogueBridgeWorker.
    """

    name: str
    subscribed: bool
    processed_transcripts: int
    response_requests: int
    ignored_events: int
    rejected_transcripts: int
    last_request_id: str | None
    last_transcript_id: str | None
    last_segment_id: str | None
    last_text: str | None
    last_error: str | None


class DialogueBridgeWorker(BaseWorker):
    """
    Event-driven bridge from final transcript to response request.

    Design:
    - consumes presence.transcript_final events
    - emits dialogue.response_requested events
    - carries voice-first response policy
    - does not call LLM/cognition directly
    - does not synthesize speech
    - does not play audio
    - does not mutate Presence state

    Future flow:
        STTWorker -> presence.transcript_final
        DialogueBridgeWorker -> dialogue.response_requested
        Cognition/Dialogue layer -> dialogue.response_ready
        TTSWorker -> audio chunks
        PlaybackWorker -> spoken output
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        policy: DialogueBridgePolicy | None = None,
        name: str = "dialogue_bridge_worker",
        tick_interval_seconds: float = 0.05,
        daemon: bool = True,
        auto_subscribe: bool = True,
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("DialogueBridgeWorker name cannot be empty.")

        if tick_interval_seconds <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero.")

        super().__init__(
            name=clean_name,
            event_bus=event_bus,
            tick_interval_seconds=tick_interval_seconds,
            daemon=daemon,
        )

        self._policy = policy or DialogueBridgePolicy()
        self._auto_subscribe = auto_subscribe

        self._lock = RLock()
        self._subscribed = False
        self._processed_transcripts = 0
        self._response_requests = 0
        self._ignored_events = 0
        self._rejected_transcripts = 0
        self._last_request_id: str | None = None
        self._last_transcript_id: str | None = None
        self._last_segment_id: str | None = None
        self._last_text: str | None = None
        self._last_error: str | None = None

        self._logger = get_logger("presence.dialogue_bridge_worker")

    @property
    def policy(self) -> DialogueBridgePolicy:
        return self._policy

    def on_start(self) -> None:
        """
        Subscribe to final transcript events when the worker starts.
        """

        if not self._auto_subscribe:
            return

        with self._lock:
            if self._subscribed:
                return

            self.event_bus.subscribe(
                event_type=EventType.PRESENCE_TRANSCRIPT_FINAL,
                subscriber_name=self.name,
                callback=self.handle_transcript_event,
            )
            self._subscribed = True

        self._logger.info(
            "dialogue_bridge_subscribed",
            worker=self.name,
            event_type=EventType.PRESENCE_TRANSCRIPT_FINAL.value,
        )

    def on_stop(self) -> None:
        """
        Event-driven bridge has no external adapter to stop.
        """

        self._logger.info("dialogue_bridge_stopped", worker=self.name)

    def run_once(self) -> None:
        """
        Event-driven worker loop placeholder.

        Dialogue bridge work happens through handle_transcript_event().
        """

    def handle_transcript_event(self, event: RuntimeEvent) -> None:
        """
        Consume one presence.transcript_final event.
        """

        if event.event_type != EventType.PRESENCE_TRANSCRIPT_FINAL:
            self._record_ignored_event()
            return

        transcript = self._extract_transcript(event)

        if transcript is None:
            self._record_ignored_event()
            return

        self.process_transcript(
            transcript=transcript,
            source_event=event,
        )

    def process_transcript(
        self,
        *,
        transcript: Transcript,
        source_event: RuntimeEvent | None = None,
    ) -> str | None:
        """
        Convert a final transcript into a dialogue response request.
        """

        if transcript.kind != TranscriptKind.FINAL:
            self._record_rejected_transcript(transcript)
            return None

        text = transcript.text.strip()

        if not text:
            self._record_rejected_transcript(transcript)
            return None

        request_id = new_dialogue_request_id()

        payload = self._build_response_request_payload(
            request_id=request_id,
            transcript=transcript,
            text=text,
            source_event=source_event,
        )

        self._publish_response_request(
            payload=payload,
            source_event=source_event,
        )

        with self._lock:
            self._processed_transcripts += 1
            self._response_requests += 1
            self._last_request_id = request_id
            self._last_transcript_id = transcript.transcript_id
            self._last_segment_id = transcript.segment_id
            self._last_text = text
            self._last_error = None

        self._logger.info(
            "dialogue_response_requested",
            worker=self.name,
            request_id=request_id,
            transcript_id=transcript.transcript_id,
            segment_id=transcript.segment_id,
            language=transcript.language,
            confidence=transcript.confidence,
        )

        return request_id

    def dialogue_snapshot(self) -> DialogueBridgeWorkerSnapshot:
        """
        Return DialogueBridgeWorker-specific diagnostics.
        """

        with self._lock:
            return DialogueBridgeWorkerSnapshot(
                name=self.name,
                subscribed=self._subscribed,
                processed_transcripts=self._processed_transcripts,
                response_requests=self._response_requests,
                ignored_events=self._ignored_events,
                rejected_transcripts=self._rejected_transcripts,
                last_request_id=self._last_request_id,
                last_transcript_id=self._last_transcript_id,
                last_segment_id=self._last_segment_id,
                last_text=self._last_text,
                last_error=self._last_error,
            )

    def _build_response_request_payload(
        self,
        *,
        request_id: str,
        transcript: Transcript,
        text: str,
        source_event: RuntimeEvent | None,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "transcript": transcript,
            "transcript_id": transcript.transcript_id,
            "segment_id": transcript.segment_id,
            "text": text,
            "language": transcript.language,
            "confidence": transcript.confidence,
            "source_event_id": source_event.event_id if source_event else None,
            "response_style": self._policy.response_style,
            "max_response_words_hint": self._policy.max_response_words_hint,
            "allow_clarifying_question": self._policy.allow_clarifying_question,
            "interruptible": self._policy.interruptible,
            "prefer_short_spoken_response": (
                self._policy.prefer_short_spoken_response
            ),
            "voice_instructions": (
                "Respond naturally for real-time voice conversation. "
                "Prefer short, clear, human-feeling responses. "
                "Avoid long paragraphs unless the user asks for detail."
            ),
            "metadata": {
                "bridge": self.name,
                "transcript_metadata": transcript.metadata,
            },
        }

    def _publish_response_request(
        self,
        *,
        payload: dict[str, Any],
        source_event: RuntimeEvent | None,
    ) -> None:
        if source_event is None:
            event = RuntimeEvent(
                event_type=EventType.ASSISTANT_RESPONSE_REQUESTED,
                category=EventCategory.DIALOGUE,
                source=self.name,
                payload=payload,
            )
        else:
            event = RuntimeEvent(
                event_type=EventType.ASSISTANT_RESPONSE_REQUESTED,
                category=EventCategory.DIALOGUE,
                source=self.name,
                correlation_id=source_event.correlation_id,
                payload=payload,
            )

        self.event_bus.publish(event)

    def _record_ignored_event(self) -> None:
        with self._lock:
            self._ignored_events += 1

    def _record_rejected_transcript(self, transcript: Transcript) -> None:
        with self._lock:
            self._rejected_transcripts += 1
            self._last_transcript_id = transcript.transcript_id
            self._last_segment_id = transcript.segment_id
            self._last_text = transcript.text
            self._last_error = (
                f"Transcript kind {transcript.kind.value!r} is not final."
            )

    @staticmethod
    def _extract_transcript(event: RuntimeEvent) -> Transcript | None:
        transcript = event.payload.get("transcript")

        if isinstance(transcript, Transcript):
            return transcript

        return None