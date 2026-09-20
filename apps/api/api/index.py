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
    @app.get("/{path:path}")
    def bootstrap_error(path: str):
        return {
            "status": "error",
            "phase": "api_bootstrap",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "path": path,
        }
