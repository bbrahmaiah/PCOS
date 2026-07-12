from __future__ import annotations

import tempfile
import threading
import time
import winsound
from pathlib import Path
from typing import Any

from jarvis.voice.contracts import (
    VoicePlaybackState,
    VoicePlaybackStatus,
    VoiceTTSChunk,
    make_voice_playback_id,
    utc_now,
)
from jarvis.voice.playback_runtime import (
    VoicePlaybackOperation,
    VoicePlaybackResult,
    VoicePlaybackRuntime,
    VoicePlaybackRuntimeStatus,
    VoicePlaybackSnapshot,
)


class WindowsAudiblePlayback:
    """
    Real audible playback adapter for Windows.

    It does not create text.
    It only plays already-generated Piper WAV chunks through Windows speaker.
    Playback is non-blocking so the voice loop can keep listening and stop
    speech during a user barge-in.
    """

    def __init__(self) -> None:
        self._inner = VoicePlaybackRuntime()
        self._queued_chunks: tuple[VoiceTTSChunk, ...] = ()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._current_playback: VoicePlaybackState | None = None
        self._status = VoicePlaybackRuntimeStatus.CREATED
        self._played_chunks = 0
        self._failed_chunks = 0
        self._stopped_count = 0
        self._last_latency_ms: float | None = None
        self._last_first_audio_latency_ms: float | None = None
        self._last_error: str | None = None

    def prepare(self) -> object:
        result = self._inner.prepare()
        if isinstance(result, VoicePlaybackResult):
            with self._lock:
                self._status = result.status
                self._last_latency_ms = result.latency_ms
                error = result.metadata.get("error")
                self._last_error = str(error) if error is not None else None
        return result

    def enqueue_chunks(self, chunks: Any) -> object:
        queued = tuple(chunks) if isinstance(chunks, tuple) else ()
        result = self._inner.enqueue_chunks(queued)
        if isinstance(result, VoicePlaybackResult):
            with self._lock:
                self._status = result.status
                self._last_latency_ms = result.latency_ms
                if result.succeeded:
                    self._queued_chunks = queued
                    self._last_error = None
                else:
                    self._queued_chunks = ()
                    self._last_error = result.message
        return result

    def play_all(self) -> object:
        if not self._queued_chunks:
            result = self._inner.play_all()
            if isinstance(result, VoicePlaybackResult):
                with self._lock:
                    self._status = result.status
                    self._last_latency_ms = result.latency_ms
                    self._last_first_audio_latency_ms = (
                        result.first_audio_latency_ms
                    )
            return result

        started = time.perf_counter()

        with self._lock:
            self._complete_worker_if_done_locked()
            if self._worker is not None and self._worker.is_alive():
                return self._result(
                    operation=VoicePlaybackOperation.PLAY_ALL,
                    status=VoicePlaybackRuntimeStatus.PLAYING,
                    playback_state=self._current_playback,
                    played_chunks=(),
                    message="windows playback already active",
                    started=started,
                    metadata={"async_playback": True},
                )

            chunks = self._queued_chunks
            self._queued_chunks = ()
            playback_state = VoicePlaybackState(
                playback_id=make_voice_playback_id(),
                session_id=chunks[0].session_id,
                status=VoicePlaybackStatus.PLAYING,
                chunk_id=chunks[0].chunk_id,
                started_at=utc_now(),
                stopped_at=None,
                metadata={
                    "provider": "winsound",
                    "async_playback": True,
                    "chunk_count": len(chunks),
                    "estimated_duration_ms": sum(
                        chunk.duration_ms for chunk in chunks
                    ),
                },
            )
            self._current_playback = playback_state
            self._status = VoicePlaybackRuntimeStatus.PLAYING
            self._last_error = None
            self._stop_event.clear()

            worker = threading.Thread(
                target=self._play_chunks_in_background,
                args=(chunks, playback_state),
                name="JarvisWindowsPlayback",
                daemon=True,
            )
            self._worker = worker

        worker.start()
        clear_result = self._inner.clear()

        metadata: dict[str, object] = {
            "async_playback": True,
            "chunk_count": len(chunks),
            "estimated_duration_ms": sum(chunk.duration_ms for chunk in chunks),
        }
        if isinstance(clear_result, VoicePlaybackResult):
            metadata["inner_clear_status"] = clear_result.status.value

        return self._result(
            operation=VoicePlaybackOperation.PLAY_ALL,
            status=VoicePlaybackRuntimeStatus.PLAYING,
            playback_state=playback_state,
            played_chunks=(),
            message="windows playback started asynchronously",
            started=started,
            first_audio_latency_ms=0.0,
            metadata=metadata,
        )

    def stop(self) -> object:
        started = time.perf_counter()
        self._stop_event.set()
        self._stop_windows_audio()

        try:
            inner_result = self._inner.stop()
        except Exception as exc:
            inner_result = None
            with self._lock:
                self._last_error = str(exc)

        worker: threading.Thread | None
        interrupted_state: VoicePlaybackState | None
        with self._lock:
            worker = self._worker
            interrupted_state = None
            if self._current_playback is not None:
                interrupted_state = VoicePlaybackState(
                    playback_id=self._current_playback.playback_id,
                    session_id=self._current_playback.session_id,
                    status=VoicePlaybackStatus.INTERRUPTED,
                    chunk_id=self._current_playback.chunk_id,
                    started_at=self._current_playback.started_at,
                    stopped_at=utc_now(),
                    metadata={
                        **self._current_playback.metadata,
                        "stop_requested": True,
                    },
                )
            self._current_playback = None
            self._queued_chunks = ()
            self._status = VoicePlaybackRuntimeStatus.STOPPED
            self._stopped_count += 1

        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=0.25)

        metadata: dict[str, object] = {
            "async_playback": True,
            "stop_requested": True,
        }
        if isinstance(inner_result, VoicePlaybackResult):
            metadata["inner_stop_status"] = inner_result.status.value
            metadata["inner_stop_message"] = inner_result.message

        return self._result(
            operation=VoicePlaybackOperation.STOP,
            status=VoicePlaybackRuntimeStatus.STOPPED,
            playback_state=interrupted_state,
            played_chunks=(),
            message="windows playback stopped",
            started=started,
            metadata=metadata,
        )

    def reset(self) -> object:
        self.stop()
        result = self._inner.reset()
        with self._lock:
            self._queued_chunks = ()
            self._current_playback = None
            self._worker = None
            self._stop_event.clear()
            self._status = VoicePlaybackRuntimeStatus.CREATED
            self._last_error = None
        return result

    def snapshot(self) -> object:
        inner = self._inner.snapshot()
        with self._lock:
            self._complete_worker_if_done_locked()
            return VoicePlaybackSnapshot(
                status=self._status,
                speaker=inner.speaker,
                queued_chunks=len(self._queued_chunks),
                played_chunks=self._played_chunks,
                failed_chunks=self._failed_chunks,
                stopped_count=self._stopped_count,
                current_playback=(
                    self._current_playback
                    if self._status == VoicePlaybackRuntimeStatus.PLAYING
                    else None
                ),
                last_latency_ms=self._last_latency_ms,
                last_first_audio_latency_ms=self._last_first_audio_latency_ms,
                last_error=self._last_error,
                created_at=utc_now(),
                metadata={
                    "provider": "winsound",
                    "async_playback": (
                        self._status == VoicePlaybackRuntimeStatus.PLAYING
                    ),
                    "inner_status": inner.status.value,
                },
            )

    def _play_chunks_in_background(
        self,
        chunks: tuple[VoiceTTSChunk, ...],
        playback_state: VoicePlaybackState,
    ) -> None:
        played_chunks = 0
        temp_paths: list[Path] = []
        first_audio_recorded = False
        started = time.perf_counter()

        try:
            for index, chunk in enumerate(chunks):
                if self._stop_event.is_set():
                    break

                wav_path = self._write_temp_wav(chunk.audio)
                temp_paths.append(wav_path)

                if index == 0:
                    first_latency_ms = (time.perf_counter() - started) * 1000.0
                    with self._lock:
                        self._last_first_audio_latency_ms = first_latency_ms
                    first_audio_recorded = True

                winsound.PlaySound(
                    str(wav_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
                self._wait_for_chunk_or_stop(chunk.duration_ms)
                played_chunks += 1

            self._stop_windows_audio()
            final_status = (
                VoicePlaybackRuntimeStatus.STOPPED
                if self._stop_event.is_set()
                else VoicePlaybackRuntimeStatus.READY
            )
            with self._lock:
                self._played_chunks += played_chunks
                self._status = final_status
                self._current_playback = None
                self._last_error = None
                self._last_latency_ms = (time.perf_counter() - started) * 1000.0
                if not first_audio_recorded:
                    self._last_first_audio_latency_ms = self._last_latency_ms
                if self._worker is threading.current_thread():
                    self._worker = None
        except Exception as exc:
            self._stop_windows_audio()
            with self._lock:
                self._failed_chunks += 1
                self._status = VoicePlaybackRuntimeStatus.FAILED
                self._current_playback = None
                self._last_error = str(exc)
                self._last_latency_ms = (time.perf_counter() - started) * 1000.0
                if self._worker is threading.current_thread():
                    self._worker = None
        finally:
            for wav_path in temp_paths:
                wav_path.unlink(missing_ok=True)

    def _wait_for_chunk_or_stop(self, duration_ms: int) -> None:
        deadline = time.perf_counter() + max(duration_ms, 1) / 1000.0
        while time.perf_counter() < deadline:
            if self._stop_event.wait(timeout=0.02):
                return

    def _write_temp_wav(self, audio: bytes) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            file.write(audio)
            file.flush()
            return Path(file.name)

    def _stop_windows_audio(self) -> None:
        flags = getattr(winsound, "SND_PURGE", 0)
        for candidate in (flags, 0):
            try:
                winsound.PlaySound(None, candidate)
                return
            except RuntimeError:
                continue

    def _complete_worker_if_done_locked(self) -> None:
        if self._worker is not None and not self._worker.is_alive():
            self._worker = None

    def _result(
        self,
        *,
        operation: VoicePlaybackOperation,
        status: VoicePlaybackRuntimeStatus,
        playback_state: VoicePlaybackState | None,
        played_chunks: tuple[VoiceTTSChunk, ...],
        message: str,
        started: float,
        first_audio_latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VoicePlaybackResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._status = status
            self._last_latency_ms = latency_ms
            if first_audio_latency_ms is not None:
                self._last_first_audio_latency_ms = first_audio_latency_ms
            if status != VoicePlaybackRuntimeStatus.FAILED:
                self._last_error = None

        inner = self._inner.snapshot()
        return VoicePlaybackResult(
            status=status,
            operation=operation,
            playback_state=playback_state,
            played_chunks=played_chunks,
            queued_chunks=len(self._queued_chunks),
            speaker=inner.speaker,
            message=message,
            latency_ms=latency_ms,
            first_audio_latency_ms=first_audio_latency_ms,
            created_at=utc_now(),
            metadata=metadata or {},
        )



