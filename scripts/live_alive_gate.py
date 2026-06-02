from __future__ import annotations

from jarvis.system.bootstrap import create_unconfigured_bootstrap


def main() -> int:
    """
    Step 46 live gate entry point.

    The reusable AliveGate is implemented in jarvis.system.alive_gate.
    This script intentionally does not guess production factories.

    Step 46 tests prove the gate with explicit factories.
    The next live wiring step should provide real MemoryGateway, CognitionWorker,
    PresenceEngine, and orchestration factories.
    """
    bootstrap = create_unconfigured_bootstrap(dry_run=True)
    result = bootstrap.start()

    if result.succeeded:
        print("JARVIS live alive gate placeholder passed.")
        return 0

    print(
        "JARVIS live alive gate requires configured live factories: "
        f"{result.error}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())