from __future__ import annotations

import sys
from pathlib import Path

# Make the monorepo root importable when Vercel executes this function from
# apps/api as its project root.
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from apps.api.main import app  # noqa: E402
