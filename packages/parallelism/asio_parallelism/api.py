"""S3 P1 routes: `/api/v1/parallelism/*`. Mount with `build_router(service)`.

Errors use the contract §3.1 envelope (`{"error": {code, message, details, request_id}}`); mount
S1's/S2's `install_error_handlers` on the app, or `install_plan_error_handler` below.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.responses import JSONResponse

from .contracts import ParallelismPlan, PlanRequest
from .service import PlanError, PlannerService


def _envelope(
    request: Request, status_code: int, code: str, message: str, details: dict | None = None
) -> JSONResponse:
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    return JSONResponse(
        status_code=status_code,
        headers={"X-Request-ID": rid},
        content={
            "error": {"code": code, "message": message, "details": details or {}, "request_id": rid}
        },
    )


def install_plan_error_handler(app: FastAPI) -> None:
    async def on_plan_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, PlanError)
        return _envelope(request, exc.status, exc.code, exc.message, exc.details)

    app.add_exception_handler(PlanError, on_plan_error)


def build_router(service: PlannerService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/parallelism", tags=["parallelism (S3)"])

    @router.post("/plan", response_model=ParallelismPlan)
    def create_plan(body: PlanRequest) -> ParallelismPlan:
        """Side-effect free apart from recording the plan so a deployment can cite its id."""
        return service.plan(body)

    @router.get("/plans/{plan_id}", response_model=ParallelismPlan)
    def get_plan(plan_id: str) -> ParallelismPlan:
        p = service.get(plan_id)
        if p is None:
            raise PlanError(
                404, "PLAN_NOT_FOUND", f"plan {plan_id!r} not found", {"plan_id": plan_id}
            )
        return p

    @router.post("/benchmarks", status_code=status.HTTP_202_ACCEPTED)
    def create_benchmark() -> None:
        raise PlanError(
            501,
            "NOT_IMPLEMENTED",
            "benchmarks run under an S1 EXPERIMENT allocation; wiring is P1 (Days 4-7)",
            {"planned_runner": "bench.py via the shared job runner"},
        )

    @router.get("/benchmarks/{benchmark_id}")
    def get_benchmark(benchmark_id: str) -> None:
        raise PlanError(
            501, "NOT_IMPLEMENTED", "benchmarks are P1 (Days 4-7)", {"benchmark_id": benchmark_id}
        )

    return router


def create_app() -> FastAPI:
    app = FastAPI(title="ASIO S3 — Parallelism planner (dev)", version="0.1.0")
    app.include_router(build_router(PlannerService()))
    install_plan_error_handler(app)
    return app


app = create_app()
