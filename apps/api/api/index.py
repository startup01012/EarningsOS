from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

# Make the monorepo root importable when Vercel executes this function from
# apps/api as its project root.
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Keep a statically discoverable top-level ASGI entrypoint for Vercel.
app = FastAPI(title="EarningsOS API")

try:
    from apps.api.main import app as application  # noqa: E402
    app = application
except Exception as exc:  # pragma: no cover - deployment diagnostic
    _bootstrap_error_type = type(exc).__name__
    _bootstrap_error_message = str(exc)

    @app.get("/{path:path}")
    def bootstrap_error(
        path: str,
        error_type: str = _bootstrap_error_type,
        error_message: str = _bootstrap_error_message,
    ):
        return {
            "status": "error",
            "phase": "api_bootstrap",
            "error_type": error_type,
            "error": error_message,
            "path": path,
        }
