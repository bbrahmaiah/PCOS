from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from jarvis.live import LiveResponse
from jarvis.voice import (
    VoiceAwarenessCognitionBridge,
    VoiceAwarenessCognitionBridgePolicy,
    VoiceAwarenessCognitionBridgeStatus,
    VoiceAwarenessFact,
    VoiceAwarenessPacket,
    VoiceAwarenessPriority,
    VoiceAwarenessRequest,
    VoiceAwarenessSource,
    VoiceAwarenessStatus,
    VoiceCognitionRequest,
    VoiceCognitionResult,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)


def _transcript(text: str = "continue") -> VoiceTranscript:
    return VoiceTranscript(
        transcript_id=make_voice_transcript_id(),
        session_id=make_voice_session_id(),
        segment_id=make_voice_segment_id(),
        kind=VoiceTranscriptKind.FINAL,
        text=text,
        confidence=0.96,
        created_at=utc_now(),
    )


def _request(text: str = "continue") -> VoiceCognitionRequest:
    return VoiceCognitionRequest(
        transcript=_transcript(text),
        user_label="Balu",
        assistant_name="JARVIS",
        metadata={"existing": "metadata"},
    )


def _fact() -> VoiceAwarenessFact:
    return VoiceAwarenessFact(
        source=VoiceAwarenessSource.MEMORY,
        key="project",
        value="JARVIS voice runtime",
        confidence=0.9,
        priority=VoiceAwarenessPriority.NORMAL,
        created_at=utc_now(),
    )


def _packet(
    *,
    status: VoiceAwarenessStatus = VoiceAwarenessStatus.READY,
) -> VoiceAwarenessPacket:
    request = VoiceAwarenessRequest(
        transcript=_transcript(),
        session_id="session",
        user_label="Balu",
        assistant_name="JARVIS",
    )
    missing = (
        (VoiceAwarenessSource.MEMORY,)
        if status == VoiceAwarenessStatus.FAILED
        else ()
    )
    return VoiceAwarenessPacket(
        status=status,
        request=request,
        facts=(_fact(),),
        cognition_context="JARVIS_AWARENESS_CONTEXT\nproject=JARVIS voice runtime",
        signature="awareness_signature_test",
        highest_priority=VoiceAwarenessPriority.NORMAL,
        missing_sources=missing,
        provider_errors=("provider_error",)
        if status == VoiceAwarenessStatus.DEGRADED
        else (),
        latency_ms=1.0,
        created_at=utc_now(),
    )


@dataclass
class FakeAwareness:
    status: VoiceAwarenessStatus = VoiceAwarenessStatus.READY
    builds: int = 0

    def build(self, request: VoiceAwarenessRequest) -> VoiceAwarenessPacket:
        self.builds += 1
        return _packet(status=self.status)


@dataclass
class FakeCognition:
    prepared: bool = False
    calls: int = 0
    last_request: VoiceCognitionRequest | None = None

    def prepare(self, *, user_label: str, assistant_name: str) -> object:
        self.prepared = True
        return object()

    def prefetch_from_partial(self, request: VoiceCognitionRequest) -> object:
        self.last_request = request
        return object()

    def think_from_transcript(
        self,
        request: VoiceCognitionRequest,
    ) -> VoiceCognitionResult:
        self.calls += 1
        self.last_request = request
        response = cast(
            LiveResponse,
            SimpleNamespace(
                response_id="response_id",
                text="generated_by_cognition",
                created_at=utc_now(),
                metadata={},
            ),
        )
        return cast(
            VoiceCognitionResult,
            SimpleNamespace(response=response),
        )

    def snapshot(self) -> object:
        return object()


def test_awareness_cognition_bridge_prepares_underlying_cognition() -> None:
    cognition = FakeCognition()
    bridge = VoiceAwarenessCognitionBridge(
        awareness=FakeAwareness(),
        cognition=cognition,
    )

    bridge.prepare(user_label="Balu", assistant_name="JARVIS")

    assert cognition.prepared is True
    assert bridge.snapshot().prepared is True


def test_awareness_cognition_bridge_attaches_awareness_to_cognition_request() -> None:
    awareness = FakeAwareness()
    cognition = FakeCognition()
    bridge = VoiceAwarenessCognitionBridge(
        awareness=awareness,
        cognition=cognition,
    )

    result = bridge.think_with_awareness(_request("continue"))

    assert result.status == VoiceAwarenessCognitionBridgeStatus.READY
    assert result.cognition_result is not None
    assert awareness.builds == 1
    assert cognition.calls == 1
    assert cognition.last_request is not None
    assert cognition.last_request.metadata["existing"] == "metadata"
    assert cognition.last_request.metadata["awareness_enabled"] is True
    assert (
        cognition.last_request.metadata["awareness_context"]
        == "JARVIS_AWARENESS_CONTEXT\nproject=JARVIS voice runtime"
    )
    assert (
        cognition.last_request.metadata["awareness_signature"]
        == "awareness_signature_test"
    )


def test_awareness_cognition_bridge_blocks_failed_awareness() -> None:
    cognition = FakeCognition()
    bridge = VoiceAwarenessCognitionBridge(
        awareness=FakeAwareness(status=VoiceAwarenessStatus.FAILED),
        cognition=cognition,
    )

    result = bridge.think_with_awareness(_request())

    assert result.status == VoiceAwarenessCognitionBridgeStatus.FAILED
    assert result.cognition_result is None
    assert cognition.calls == 0
    assert "blocked" in result.reason


def test_awareness_cognition_bridge_allows_degraded_awareness_by_policy() -> None:
    cognition = FakeCognition()
    bridge = VoiceAwarenessCognitionBridge(
        awareness=FakeAwareness(status=VoiceAwarenessStatus.DEGRADED),
        cognition=cognition,
        policy=VoiceAwarenessCognitionBridgePolicy(
            allow_degraded_awareness=True
        ),
    )

    result = bridge.think_with_awareness(_request())

    assert result.status == VoiceAwarenessCognitionBridgeStatus.DEGRADED
    assert result.cognition_result is not None
    assert cognition.calls == 1
    assert bridge.snapshot().degraded_awareness_count == 1


def test_awareness_cognition_bridge_prefetches_partial_with_awareness() -> None:
    awareness = FakeAwareness()
    cognition = FakeCognition()
    bridge = VoiceAwarenessCognitionBridge(
        awareness=awareness,
        cognition=cognition,
    )

    bridge.prefetch_from_partial(_request("partial"))

    assert awareness.builds == 1
    assert cognition.last_request is not None
    assert cognition.last_request.metadata["awareness_partial"] is True


def test_awareness_cognition_bridge_snapshot_tracks_counts() -> None:
    bridge = VoiceAwarenessCognitionBridge(
        awareness=FakeAwareness(),
        cognition=FakeCognition(),
    )

    bridge.think_with_awareness(_request())

    snapshot = bridge.snapshot()

    assert snapshot.awareness_builds == 1
    assert snapshot.cognition_calls == 1
    assert snapshot.last_awareness_signature == "awareness_signature_test"


def test_awareness_cognition_bridge_does_not_generate_final_speech() -> None:
    bridge = VoiceAwarenessCognitionBridge(
        awareness=FakeAwareness(),
        cognition=FakeCognition(),
    )

    result = bridge.think_with_awareness(_request())

    assert result.cognition_result is not None
    assert not hasattr(result.awareness_packet, "response_text")
    assert not hasattr(result.awareness_packet, "spoken_text")


def test_awareness_cognition_bridge_enum_values_are_stable() -> None:
    assert VoiceAwarenessCognitionBridgeStatus.READY.value == "ready"