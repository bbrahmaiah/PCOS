from __future__ import annotations

from dataclasses import dataclass

from jarvis.live import (
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerationSource,
    LiveResponseGenerator,
)
from jarvis.runtime import JarvisHttpBridgeConfig, JarvisHttpBridgeRuntime
from jarvis.voice import (
    VoiceAwarenessCognitionBridge,
    VoiceAwarenessFact,
    VoiceAwarenessPacket,
    VoiceAwarenessPriority,
    VoiceAwarenessRequest,
    VoiceAwarenessSource,
    VoiceAwarenessStatus,
    VoiceCognitionPolicy,
    VoiceCognitionResponseRuntime,
    utc_now,
)


class EchoResponseGenerator(LiveResponseGenerator):
    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        text = f"Understood. {request.context.user_text}"
        return LiveResponseDraft(
            text=text,
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={
                "test_generator": True,
                "first_token_latency_ms": 1.5,
            },
        )


@dataclass
class FakeAwareness:
    builds: int = 0

    def build(self, request: VoiceAwarenessRequest) -> VoiceAwarenessPacket:
        self.builds += 1
        return VoiceAwarenessPacket(
            status=VoiceAwarenessStatus.READY,
            request=request,
            facts=(
                VoiceAwarenessFact(
                    source=VoiceAwarenessSource.MEMORY,
                    key="project",
                    value="real JARVIS runtime",
                    confidence=0.95,
                    priority=VoiceAwarenessPriority.NORMAL,
                    created_at=utc_now(),
                ),
            ),
            cognition_context="project=real JARVIS runtime",
            signature="fake-awareness-signature",
            highest_priority=VoiceAwarenessPriority.NORMAL,
            missing_sources=(),
            provider_errors=(),
            latency_ms=1.0,
            created_at=utc_now(),
        )


def _runtime() -> JarvisHttpBridgeRuntime:
    cognition = VoiceAwarenessCognitionBridge(
        awareness=FakeAwareness(),
        cognition=VoiceCognitionResponseRuntime(
            response_generator=EchoResponseGenerator(),
            policy=VoiceCognitionPolicy(
                min_dialogue_confidence=0.55,
                max_response_sentences=2,
                require_wake_word_when_sleeping=False,
            ),
        ),
    )
    return JarvisHttpBridgeRuntime(
        config=JarvisHttpBridgeConfig(port=9876),
        cognition=cognition,
    )


def test_http_bridge_health_is_real_runtime_not_fake_fallback() -> None:
    runtime = _runtime()

    health = runtime.health()

    assert health["runtime"] == "jarvis_http_bridge"
    assert health["realRuntime"] is True
    assert health["fakeFallbackEnabled"] is False


def test_http_bridge_routes_turn_through_cognition_boundary() -> None:
    runtime = _runtime()

    result = runtime.handle_turn(
        {
            "message": "Jarvis explain AGI briefly.",
            "history": [{"role": "user", "content": "We discussed AI."}],
            "telemetry": {"battery": 88},
            "activeRole": "Personal Cognitive Operating System",
        }
    )

    assert result.status_code == 200
    assert result.body["source"] == "real_jarvis_runtime"
    assert result.body["fakeFallbackEnabled"] is False
    assert result.body["command"] is None
    assert "Jarvis explain AGI briefly." in str(result.body["text"])
    metadata = result.body["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["bridgeStatus"] == "ready"
    assert metadata["cognitionStatus"] == "thinking"


def test_http_bridge_rejects_empty_message_without_fake_reply() -> None:
    runtime = _runtime()

    result = runtime.handle_turn({"message": " "})

    assert result.status_code == 400
    assert result.body["source"] == "jarvis_http_bridge"
    assert result.body["fakeFallbackEnabled"] is False
    assert result.body["command"] is None
