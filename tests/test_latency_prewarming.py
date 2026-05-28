from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.latency import (
    HotCache,
    HotCacheConfig,
    PrewarmConnectionPoolRuntime,
    PrewarmEventKind,
    PrewarmReason,
    PrewarmRuntimeConfig,
    PrewarmStatus,
    PrewarmTarget,
    WarmResource,
    WarmResourceRole,
)


def test_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PrewarmRuntimeConfig(name=" ").validate()


def test_config_requires_two_llm_connections() -> None:
    with pytest.raises(ValueError):
        PrewarmRuntimeConfig(llm_connection_pool_size=1).validate()


def test_config_rejects_invalid_tts_workers() -> None:
    with pytest.raises(ValueError):
        PrewarmRuntimeConfig(tts_worker_pool_size=0).validate()


def test_config_rejects_invalid_ping_interval() -> None:
    with pytest.raises(ValueError):
        PrewarmRuntimeConfig(idle_ping_interval_ms=0).validate()


def test_hot_cache_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        HotCacheConfig(max_entries=0).validate()


def test_hot_cache_lru() -> None:
    cache = HotCache(config=HotCacheConfig(max_entries=2))

    cache.set("one", "1")
    cache.set("two", "2")
    cache.set("three", "3")

    assert cache.size() == 2
    assert cache.get("one") is None
    assert cache.get("two") == "2"
    assert cache.get("three") == "3"


def test_warm_resource_requires_name() -> None:
    with pytest.raises(ValidationError):
        WarmResource(
            target=PrewarmTarget.LLM_CONNECTION,
            role=WarmResourceRole.PRIMARY,
            name=" ",
        )


def test_runtime_creates_session() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    assert state.status == PrewarmStatus.WARMING
    assert runtime.snapshot().session_count == 1


def test_runtime_warms_llm_connection_pool() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    result = runtime.warm_llm_connections(state.session_id)
    pool = runtime.pool_for(PrewarmTarget.LLM_CONNECTION)

    assert result.success is True
    assert pool is not None
    assert len(pool.resources) == 2
    assert any(resource.role == WarmResourceRole.PRIMARY for resource in pool.resources)
    assert any(resource.role == WarmResourceRole.STANDBY for resource in pool.resources)
    assert any(
        resource.metadata.get("system_prompt_loaded") is True
        for resource in pool.resources
    )


def test_runtime_sends_idle_ping() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    runtime.warm_llm_connections(state.session_id)
    result = runtime.send_idle_ping(state.session_id)
    pool = runtime.pool_for(PrewarmTarget.LLM_CONNECTION)

    assert result.success is True
    assert result.reason == PrewarmReason.IDLE_PING_SENT
    assert pool is not None
    assert all(resource.last_ping_at_ns is not None for resource in pool.resources)


def test_runtime_warms_tts_pool_and_phoneme_cache() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    result = runtime.warm_tts_pool(state.session_id)
    pool = runtime.pool_for(PrewarmTarget.TTS_ENGINE)

    assert result.success is True
    assert pool is not None
    assert len(pool.resources) == 2
    assert runtime.hot_cache_size() > 0


def test_runtime_warms_audio_playback_buffer() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    result = runtime.warm_audio_playback(state.session_id)
    pool = runtime.pool_for(PrewarmTarget.AUDIO_PLAYBACK)

    assert result.success is True
    assert pool is not None
    assert pool.resources[0].metadata["prebuffer_ms"] == 200.0
    assert pool.resources[0].metadata["stream_open"] is True


def test_runtime_warms_memory_hot_cache() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    result = runtime.warm_memory_cache(state.session_id)
    pool = runtime.pool_for(PrewarmTarget.MEMORY_HOT_CACHE)

    assert result.success is True
    assert pool is not None
    assert runtime.hot_cache_size() == 5


def test_runtime_warms_workspace_index() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    result = runtime.warm_workspace_index(state.session_id)
    pool = runtime.pool_for(PrewarmTarget.WORKSPACE_INDEX)

    assert result.success is True
    assert pool is not None
    assert pool.resources[0].metadata["preindexed"] is True


def test_runtime_warm_all() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    results = runtime.warm_all(state.session_id)

    assert len(results) == 5
    assert all(result.success for result in results)
    assert runtime.snapshot().warm_pool_count == 5
    assert runtime.snapshot().warm_resource_count >= 7


def test_complete_session_builds_report() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    runtime.warm_all(state.session_id)
    runtime.send_idle_ping(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert report.status == PrewarmStatus.WARM
    assert report.warmed_target_count == 5
    assert report.warmed_resource_count >= 7
    assert report.ping_count == 2
    assert report.profiler_report is not None


def test_complete_rejects_missing_session() -> None:
    runtime = PrewarmConnectionPoolRuntime()

    with pytest.raises(ValueError):
        runtime.complete_session("missing")


def test_invalidate_hot_cache() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    runtime.warm_memory_cache(state.session_id)
    assert runtime.hot_cache_size() > 0

    result = runtime.invalidate_hot_cache(state.session_id)

    assert result.success is True
    assert runtime.hot_cache_size() == 0
    assert runtime.snapshot().last_reason == PrewarmReason.CACHE_INVALIDATED


def test_cancel_session() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    result = runtime.cancel_session(state.session_id)

    assert result.success is True
    assert result.status == PrewarmStatus.CANCELLED


def test_fail_session() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    result = runtime.fail_session(state.session_id, error="prewarm failed")

    assert result.success is True
    assert result.status == PrewarmStatus.FAILED


def test_reports_are_queryable() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    runtime.warm_all(state.session_id)
    report = runtime.complete_session(state.session_id)

    assert runtime.latest_report() == report
    assert runtime.reports() == (report,)


def test_snapshot_tracks_counts() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    runtime.warm_all(state.session_id)
    runtime.complete_session(state.session_id)
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 1
    assert snapshot.report_count == 1
    assert snapshot.warm_pool_count == 5


def test_reset_clears_runtime() -> None:
    runtime = PrewarmConnectionPoolRuntime()
    state = runtime.create_session()

    runtime.warm_all(state.session_id)
    runtime.reset()
    snapshot = runtime.snapshot()

    assert snapshot.session_count == 0
    assert snapshot.warm_pool_count == 0
    assert snapshot.cache_size == 0
    assert snapshot.last_reason == PrewarmReason.RUNTIME_RESET


def test_event_values_are_stable() -> None:
    assert PrewarmTarget.LLM_CONNECTION.value == "llm_connection"
    assert PrewarmStatus.WARM.value == "warm"
    assert PrewarmReason.IDLE_PING_SENT.value == "idle_ping_sent"
    assert PrewarmEventKind.AUDIO_BUFFER_READY.value == "audio_buffer_ready"