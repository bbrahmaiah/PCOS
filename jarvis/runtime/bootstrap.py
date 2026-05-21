from __future__ import annotations

import argparse
import signal
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from jarvis.runtime.kernel import RuntimeKernel
from jarvis.runtime.observability.structured_logger import get_logger


def utc_now() -> datetime:
    return datetime.now(UTC)


class KernelLike(Protocol):
    """
    Minimal runtime kernel protocol used by the bootstrapper.

    RuntimeKernel satisfies this protocol, and tests can inject lightweight
    fake kernels without booting the full runtime.
    """

    def start(self) -> None:
        """Start the kernel."""

    def stop(self) -> None:
        """Stop the kernel."""

    def snapshot(self) -> Any:
        """Return a diagnostic kernel snapshot."""


@dataclass(frozen=True, slots=True)
class BootstrapSnapshot:
    """
    Immutable snapshot of bootstrap state.
    """

    running: bool
    booted_at: datetime | None
    stopped_at: datetime | None
    shutdown_requested: bool
    kernel_snapshot: Any | None


class JarvisBootstrapper:
    """
    Production bootstrapper for the JARVIS runtime.

    Responsibilities:
    - create the RuntimeKernel
    - start the kernel exactly once
    - stop the kernel safely
    - support Ctrl+C / termination shutdown
    - expose bootstrap diagnostics
    """

    def __init__(
        self,
        *,
        kernel: KernelLike | None = None,
        install_signal_handlers: bool = True,
    ) -> None:
        self._kernel = kernel
        self._install_signal_handlers = install_signal_handlers

        self._lock = threading.RLock()
        self._shutdown_requested = threading.Event()

        self._running = False
        self._booted_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._signals_installed = False

        self._logger = get_logger("bootstrap")

    @property
    def kernel(self) -> KernelLike:
        with self._lock:
            if self._kernel is None:
                self._kernel = RuntimeKernel()

            return self._kernel

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> KernelLike:
        """
        Boot the runtime kernel.

        Safe to call multiple times.
        """

        with self._lock:
            if self._running:
                return self.kernel

            if self._install_signal_handlers:
                self._install_shutdown_signal_handlers()

            kernel = self.kernel

            self._logger.info("jarvis_bootstrap_starting")

            kernel.start()

            self._running = True
            self._booted_at = utc_now()
            self._stopped_at = None
            self._shutdown_requested.clear()

            self._logger.info(
                "jarvis_bootstrap_started",
                booted_at=self._booted_at.isoformat(),
            )

            return kernel

    def stop(self) -> None:
        """
        Stop the runtime kernel.

        Safe to call multiple times.
        """

        with self._lock:
            if not self._running and self._kernel is None:
                return

            kernel = self._kernel

            if kernel is None:
                self._running = False
                self._stopped_at = utc_now()
                return

            if not self._running:
                return

            self._logger.info("jarvis_bootstrap_stopping")

            try:
                kernel.stop()
            finally:
                self._running = False
                self._stopped_at = utc_now()

                self._logger.info(
                    "jarvis_bootstrap_stopped",
                    stopped_at=self._stopped_at.isoformat(),
                )

    def request_shutdown(self, reason: str = "Shutdown requested.") -> None:
        """
        Request graceful shutdown.

        This is used by signal handlers and future UI/runtime controls.
        """

        clean_reason = reason.strip() if reason else "Shutdown requested."

        self._logger.info(
            "jarvis_shutdown_requested",
            reason=clean_reason,
        )

        self._shutdown_requested.set()

    def run_forever(self, *, poll_interval_seconds: float = 0.25) -> int:
        """
        Start JARVIS and block until shutdown is requested.

        Returns process exit code.
        """

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")

        self.start()

        try:
            while not self._shutdown_requested.wait(poll_interval_seconds):
                continue

            return 0

        except KeyboardInterrupt:
            self.request_shutdown("KeyboardInterrupt")
            return 130

        finally:
            self.stop()

    def snapshot(self) -> BootstrapSnapshot:
        with self._lock:
            kernel_snapshot = self._kernel.snapshot() if self._kernel else None

            return BootstrapSnapshot(
                running=self._running,
                booted_at=self._booted_at,
                stopped_at=self._stopped_at,
                shutdown_requested=self._shutdown_requested.is_set(),
                kernel_snapshot=kernel_snapshot,
            )

    def _install_shutdown_signal_handlers(self) -> None:
        if self._signals_installed:
            return

        if threading.current_thread() is not threading.main_thread():
            return

        def handle_signal(signum: int, _frame: object | None) -> None:
            self.request_shutdown(f"Signal received: {signum}")

        signal.signal(signal.SIGINT, handle_signal)

        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)

        self._signals_installed = True


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis-bootstrap",
        description="Boot the JARVIS cognitive runtime.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Start the runtime, take a snapshot, stop, and exit.",
    )

    parser.add_argument(
        "--no-signal-handlers",
        action="store_true",
        help="Disable signal handler installation.",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Runtime shutdown polling interval in seconds.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    bootstrapper = JarvisBootstrapper(
        install_signal_handlers=not args.no_signal_handlers,
    )

    if args.check:
        bootstrapper.start()
        bootstrapper.snapshot()
        bootstrapper.stop()
        print("JARVIS bootstrap check passed.")
        return 0

    return bootstrapper.run_forever(
        poll_interval_seconds=args.poll_interval,
    )