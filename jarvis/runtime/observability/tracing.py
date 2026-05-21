from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class TraceSpan:
    """
    Completed trace span.

    Used to understand how long runtime operations take.
    """

    trace_id: str
    span_id: str
    name: str
    duration_ms: float
    success: bool
    correlation_id: str | None = None
    error: str | None = None
    attributes: dict[str, str | int | float | bool | None] = field(default_factory=dict)


class Tracer:
    """
    Lightweight local tracer.

    Later this can be replaced or extended with OpenTelemetry,
    but the contract stays stable.
    """

    def __init__(self, service_name: str = "jarvis_runtime") -> None:
        self.service_name = service_name
        self.logger = get_logger("observability.tracing")

    @contextmanager
    def span(
        self,
        name: str,
        *,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        **attributes: str | int | float | bool | None,
    ) -> Iterator[TraceSpan]:
        actual_trace_id = trace_id or uuid4().hex
        span_id = uuid4().hex
        start = perf_counter()
        success = False
        error: str | None = None

        try:
            yield TraceSpan(
                trace_id=actual_trace_id,
                span_id=span_id,
                name=name,
                duration_ms=0.0,
                success=True,
                correlation_id=correlation_id,
                attributes=attributes,
            )
            success = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = (perf_counter() - start) * 1000

            completed = TraceSpan(
                trace_id=actual_trace_id,
                span_id=span_id,
                name=name,
                duration_ms=duration_ms,
                success=success,
                correlation_id=correlation_id,
                error=error,
                attributes=attributes,
            )

            self.logger.info(
                "trace_span_completed",
                service=self.service_name,
                trace_id=completed.trace_id,
                span_id=completed.span_id,
                span_name=completed.name,
                duration_ms=round(completed.duration_ms, 3),
                success=completed.success,
                correlation_id=completed.correlation_id,
                error=completed.error,
                attributes=completed.attributes,
            )


_default_tracer = Tracer()


def get_tracer() -> Tracer:
    return _default_tracer