from __future__ import annotations

from dataclasses import dataclass
from threading import Timer

import pytest

from jarvis.runtime.bootstrap import JarvisBootstrapper, build_argument_parser, main


@dataclass
class FakeKernel:
    started: bool = False
    stopped: bool = False
    start_count: int = 0
    stop_count: int = 0

    def start(self) -> None:
        self.started = True
        self.start_count += 1

    def stop(self) -> None:
        self.stopped = True
        self.stop_count += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "started": self.started,
            "stopped": self.stopped,
            "start_count": self.start_count,
            "stop_count": self.stop_count,
        }


def test_bootstrapper_starts_kernel_once() -> None:
    kernel = FakeKernel()
    bootstrapper = JarvisBootstrapper(
        kernel=kernel,
        install_signal_handlers=False,
    )

    first = bootstrapper.start()
    second = bootstrapper.start()

    assert first is kernel
    assert second is kernel
    assert kernel.start_count == 1
    assert bootstrapper.running is True


def test_bootstrapper_stops_kernel_once() -> None:
    kernel = FakeKernel()
    bootstrapper = JarvisBootstrapper(
        kernel=kernel,
        install_signal_handlers=False,
    )

    bootstrapper.start()
    bootstrapper.stop()
    bootstrapper.stop()

    assert kernel.stop_count == 1
    assert bootstrapper.running is False


def test_bootstrapper_snapshot_reports_state() -> None:
    kernel = FakeKernel()
    bootstrapper = JarvisBootstrapper(
        kernel=kernel,
        install_signal_handlers=False,
    )

    bootstrapper.start()
    snapshot = bootstrapper.snapshot()

    assert snapshot.running is True
    assert snapshot.booted_at is not None
    assert snapshot.kernel_snapshot is not None

    bootstrapper.stop()

    stopped_snapshot = bootstrapper.snapshot()

    assert stopped_snapshot.running is False
    assert stopped_snapshot.stopped_at is not None


def test_bootstrapper_request_shutdown() -> None:
    kernel = FakeKernel()
    bootstrapper = JarvisBootstrapper(
        kernel=kernel,
        install_signal_handlers=False,
    )

    bootstrapper.request_shutdown("test shutdown")

    snapshot = bootstrapper.snapshot()

    assert snapshot.shutdown_requested is True


def test_bootstrapper_run_forever_exits_when_shutdown_requested() -> None:
    kernel = FakeKernel()
    bootstrapper = JarvisBootstrapper(
        kernel=kernel,
        install_signal_handlers=False,
    )

    timer = Timer(
        0.01,
        lambda: bootstrapper.request_shutdown("test"),
    )
    timer.start()

    try:
        exit_code = bootstrapper.run_forever(poll_interval_seconds=0.001)
    finally:
        timer.cancel()

    assert exit_code == 0
    assert kernel.started is True
    assert kernel.stopped is True
    assert bootstrapper.running is False


def test_bootstrapper_rejects_invalid_poll_interval() -> None:
    bootstrapper = JarvisBootstrapper(
        kernel=FakeKernel(),
        install_signal_handlers=False,
    )

    with pytest.raises(ValueError):
        bootstrapper.run_forever(poll_interval_seconds=0)


def test_argument_parser_accepts_check_flag() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(["--check", "--no-signal-handlers"])

    assert args.check is True
    assert args.no_signal_handlers is True


def test_main_check_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    class TestBootstrapper(JarvisBootstrapper):
        def __init__(self, *, install_signal_handlers: bool = True) -> None:
            super().__init__(
                kernel=FakeKernel(),
                install_signal_handlers=install_signal_handlers,
            )

    monkeypatch.setattr(
        "jarvis.runtime.bootstrap.JarvisBootstrapper",
        TestBootstrapper,
    )

    exit_code = main(["--check", "--no-signal-handlers"])

    assert exit_code == 0