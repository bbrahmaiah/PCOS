from __future__ import annotations

from jarvis.cognition import (
    LocalLLMAdapter,
    SpokenResponseStyle,
    ToolActionType,
    ValidationLocalLLMBackend,
    VoiceCognitionIO,
    VoiceCognitionPlaybackResult,
    VoiceCognitionSmokeConfig,
    VoiceCognitionSmokeRunner,
    VoiceCognitionTranscript,
    create_cognition_runtime,
)


class FakeVoiceCognitionIO:
    def __init__(
        self,
        *,
        transcript: str,
        playback_completed: bool = True,
    ) -> None:
        self.transcript = transcript
        self.playback_completed = playback_completed
        self.spoken_text: str | None = None

    @property
    def name(self) -> str:
        return "fake_voice_cognition_io"

    def listen_once(self) -> VoiceCognitionTranscript:
        return VoiceCognitionTranscript(
            text=self.transcript,
            confidence=1.0 if self.transcript.strip() else 0.0,
            source="fake_voice",
        )

    def speak(self, text: str) -> VoiceCognitionPlaybackResult:
        self.spoken_text = text

        return VoiceCognitionPlaybackResult(
            text=text,
            started=True,
            completed=self.playback_completed,
        )


def make_runner(
    *,
    transcript: str = "What is the cognition status?",
    streaming: bool = False,
    allow_tools: bool = False,
    playback_completed: bool = True,
) -> tuple[VoiceCognitionSmokeRunner, FakeVoiceCognitionIO]:
    backend = ValidationLocalLLMBackend()
    adapter = LocalLLMAdapter(backend=backend)
    runtime = create_cognition_runtime(adapter=adapter)
    voice_io = FakeVoiceCognitionIO(
        transcript=transcript,
        playback_completed=playback_completed,
    )

    runner = VoiceCognitionSmokeRunner(
        runtime=runtime,
        voice_io=voice_io,
        config=VoiceCognitionSmokeConfig(
            streaming=streaming,
            allow_tools=allow_tools,
            allow_memory_lookup=True,
            spoken_style=SpokenResponseStyle.CONCISE,
        ),
    )

    return runner, voice_io


def test_fake_voice_io_satisfies_protocol() -> None:
    _voice_io: VoiceCognitionIO = FakeVoiceCognitionIO(transcript="hello")


def test_voice_cognition_smoke_non_streaming_passes() -> None:
    runner, voice_io = make_runner()

    report = runner.run_once()
    snapshot = runner.snapshot()

    assert report.passed is True
    assert report.heard_text == "What is the cognition status?"
    assert report.response_text is not None
    assert "cognition runtime is connected" in report.response_text
    assert voice_io.spoken_text == report.response_text
    assert snapshot["passed_count"] == 1


def test_voice_cognition_smoke_streaming_passes() -> None:
    runner, voice_io = make_runner(
        transcript="Stream the cognition status.",
        streaming=True,
    )

    report = runner.run_once()

    assert report.passed is True
    assert report.runtime_result is not None
    assert report.runtime_result.streamed is True
    assert report.response_text is not None
    assert "Streaming cognition is connected." in report.response_text
    assert voice_io.spoken_text == report.response_text


def test_voice_cognition_smoke_empty_transcript_fails() -> None:
    runner, _voice_io = make_runner(transcript=" ")

    report = runner.run_once()

    assert report.passed is False
    assert report.reason == "empty transcript"
    assert report.runtime_result is None


def test_voice_cognition_smoke_playback_failure_fails() -> None:
    runner, _voice_io = make_runner(playback_completed=False)

    report = runner.run_once()

    assert report.passed is False
    assert report.reason == "playback did not complete"


def test_voice_cognition_smoke_action_planning_is_plan_only() -> None:
    runner, _voice_io = make_runner(
        transcript="open diagnostics",
        allow_tools=True,
    )

    report = runner.run_once()

    assert report.passed is True
    assert report.runtime_result is not None
    assert report.runtime_result.action_plan is not None
    assert report.runtime_result.action_plan.metadata[
        "llm_direct_execution_allowed"
    ] is False
    assert report.runtime_result.action_plan.proposals[0].action_type == (
        ToolActionType.OPEN_APPLICATION
    )