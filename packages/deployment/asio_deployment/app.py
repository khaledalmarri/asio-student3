"""Stand-alone dev server for the S3 routes (no Docker access, like the real API process).

    ASIO_AUTH_MODE=development uv run uvicorn asio_deployment.app:app --port 8003

`ASIO_DATABASE_URL` selects PostgreSQL (tables from `sql/0001_s3_proposed.sql`); without it the
store is in memory. `ASIO_S1_BASE_URL` enables the allocation pre-check against S1.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from .api import build_router
from .controller import DeploymentController
from .errors import install_error_handlers
from .s1 import HttpS1Client
from .store import MemoryStore, PostgresStore, Store


def store_from_env() -> Store:
    dsn = os.environ.get("ASIO_DATABASE_URL")
    return PostgresStore(dsn) if dsn else MemoryStore()


def create_app(controller: DeploymentController) -> FastAPI:
    app = FastAPI(title="ASIO S3 — Deployment Controller (dev)", version="0.1.0")
    app.include_router(build_router(controller))
    install_error_handlers(app)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _default() -> FastAPI:
    s1 = HttpS1Client.from_env() if os.environ.get("ASIO_S1_BASE_URL") else None
    return create_app(DeploymentController(store_from_env(), s1=s1))


app = _default()
