from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime import (  # noqa: E402
    JarvisRuntimeBindingVerificationMode,
    JarvisRuntimeBindingVerificationReport,
    JarvisRuntimeBindingVerificationStatus,
    JarvisRuntimeBindingVerifier,
    JarvisRuntimeBindingVerifierConfig,
    JarvisStartControlRuntime,
    build_connected_start_control_from_plan,
    build_plan_from_import_bindings,
    read_runtime_binding_imports,
    summarize_binding_report,
)
from jarvis.voice import (  # noqa: E402
    VoiceRuntimeLauncher,
    VoiceRuntimeLauncherConfig,
)

BINDINGS_PATH = PROJECT_ROOT / "config" / "runtime_bindings.env"


@dataclass(slots=True)
class ConnectedJarvisEntrypointConfig:
    bindings_path: Path = BINDINGS_PATH
    verify_dry_run: bool = True
    health_interval_seconds: float = 2.0
    run_voice_forever: bool = True
    run_daily_driver_gate: bool = True
    allow_degraded_voice_gate: bool = False

    def __post_init__(self) -> None:
        if self.health_interval_seconds <= 0:
            raise ValueError("health_interval_seconds must be positive.")


class ConnectedJarvisEntrypoint:
    """
    Final connected JARVIS runtime entrypoint.

    This starts the connected personal cognitive OS:
    Phase 1-9 organs + Step 51 voice daily-driver launcher.

    It never generates conversational text. Final speech remains:
    STT -> awareness -> cognition/Ollama -> response boundary -> TTS.
    """

    def __init__(self, config: ConnectedJarvisEntrypointConfig) -> None:
        self._config = config
        self._stop_requested = False
        self._start_control: JarvisStartControlRuntime | None = None

    def run(self) -> int:
        if not self._verify_bindings():
            return 1

        try:
            self._start_control = self._build_start_control()
        except Exception as exc:
            print(f"[runtime] build failed: {exc}")
            return 1

        result = self._start_control.start_all()
        if not result.succeeded:
            print(f"[runtime] start blocked: {result.reason}")
            return 1

        print("[runtime] connected JARVIS started")

        try:
            self._health_loop()
        except KeyboardInterrupt:
            self._stop_requested = True

        return self._shutdown()

    def request_stop(self) -> None:
        self._stop_requested = True

    def _verify_bindings(self) -> bool:
        resolve_report = self._verify(
            mode=JarvisRuntimeBindingVerificationMode.RESOLVE_ONLY
        )
        if resolve_report.status == JarvisRuntimeBindingVerificationStatus.FAILED:
            print(summarize_binding_report(resolve_report))
            return False

        if not self._config.verify_dry_run:
            return True

        dry_run_report = self._verify(
            mode=JarvisRuntimeBindingVerificationMode.FACTORY_DRY_RUN
        )
        if dry_run_report.status == JarvisRuntimeBindingVerificationStatus.FAILED:
            print(summarize_binding_report(dry_run_report))
            return False

        return True

    def _verify(
        self,
        *,
        mode: JarvisRuntimeBindingVerificationMode,
    ) -> JarvisRuntimeBindingVerificationReport:
        verifier = JarvisRuntimeBindingVerifier(
            config=JarvisRuntimeBindingVerifierConfig(
                bindings_path=self._config.bindings_path,
                mode=mode,
                metadata={"entrypoint": "run_connected_jarvis"},
            )
        )
        return verifier.verify()

    def _build_start_control(self) -> JarvisStartControlRuntime:
        import_bindings = read_runtime_binding_imports(self._config.bindings_path)

        voice_launcher = VoiceRuntimeLauncher(
            config=VoiceRuntimeLauncherConfig(
                run_forever=self._config.run_voice_forever,
                run_daily_driver_gate=self._config.run_daily_driver_gate,
                allow_degraded_gate=self._config.allow_degraded_voice_gate,
                metadata={"entrypoint": "run_connected_jarvis"},
            )
        )

        plan = build_plan_from_import_bindings(
            import_bindings=import_bindings,
            voice_launcher=voice_launcher,
        )

        return build_connected_start_control_from_plan(plan)

    def _health_loop(self) -> None:
        if self._start_control is None:
            raise RuntimeError("start control was not built")

        while not self._stop_requested:
            health = self._start_control.health()
            if not health.succeeded:
                print(f"[runtime] health failed: {health.reason}")
                self._stop_requested = True
                break

            time.sleep(self._config.health_interval_seconds)

    def _shutdown(self) -> int:
        if self._start_control is None:
            return 0

        stop_result = self._start_control.stop_all()
        if not stop_result.succeeded:
            print(f"[runtime] shutdown failed: {stop_result.reason}")
            return 1

        print("[runtime] connected JARVIS stopped")
        return 0


def main() -> int:
    entrypoint = ConnectedJarvisEntrypoint(
        ConnectedJarvisEntrypointConfig()
    )

    def _handle_signal(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        entrypoint.request_stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    return entrypoint.run()


if __name__ == "__main__":
    raise SystemExit(main())