from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime import (  # noqa: E402
    JarvisRuntimeBindingVerificationMode,
    JarvisRuntimeBindingVerificationStatus,
    JarvisRuntimeBindingVerifier,
    JarvisRuntimeBindingVerifierConfig,
    summarize_binding_report,
)

BINDINGS_PATH = PROJECT_ROOT / "config" / "runtime_bindings.env"


def main() -> int:
    mode = (
        JarvisRuntimeBindingVerificationMode.FACTORY_DRY_RUN
        if "--dry-run" in sys.argv
        else JarvisRuntimeBindingVerificationMode.RESOLVE_ONLY
    )

    verifier = JarvisRuntimeBindingVerifier(
        config=JarvisRuntimeBindingVerifierConfig(
            bindings_path=BINDINGS_PATH,
            mode=mode,
            metadata={"entrypoint": "verify_runtime_bindings"},
        )
    )
    report = verifier.verify()

    print(summarize_binding_report(report))

    return 0 if report.status == JarvisRuntimeBindingVerificationStatus.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())