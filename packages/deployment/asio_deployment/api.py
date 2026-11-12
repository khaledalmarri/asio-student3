"""S3 HTTP routes (contract §5.4): `/api/v1/deployments`.

Mount into S1's gateway (or the stand-alone `asio_deployment.app`)::

    app.include_router(build_router(controller))
    install_error_handlers(app)

This router runs in the API process and never touches Docker: it only writes the deployment, its
STARTING pool record and a DEPLOY/STOP_DEPLOY job. The privileged worker does the rest.

Authentication is S1's. S1's middleware sets `request.state.caller = Caller(...)`; with
`ASIO_AUTH_MODE=development` (and only then) the `X-ASIO-Caller: <id>:<role>` header is accepted.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, Field

from .contracts import Deployment, JobState, ObservedState
from .controller import CreateDeploymentRequest, DeploymentController
from .errors import ApiError, ErrorResponse

DEPLOYER_ROLES = {"admin", "system"}


class Caller(BaseModel):
    id: str
    role: str


def get_caller(request: Request) -> Caller:
    caller = getattr(request.state, "caller", None)
    if isinstance(caller, Caller):
        return caller
    if os.environ.get("ASIO_AUTH_MODE") == "development":
        raw = request.headers.get("X-ASIO-Caller", "dev-admin:admin")
        cid, _, role = raw.partition(":")
        return Caller(id=cid or "dev-admin", role=role or "admin")
    raise ApiError(401, "UNAUTHENTICATED", "no authenticated caller")


def require_deployer(caller: Annotated[Caller, Depends(get_caller)]) -> Caller:
    if caller.role not in DEPLOYER_ROLES:
        raise ApiError(
            403,
            "FORBIDDEN",
            "only admin/system callers manage serving deployments",
            {"role": caller.role},
        )
    return caller


class DeploymentAccepted(BaseModel):
    id: str
    observed_state: ObservedState
    status_url: str
    job_id: str
    job_state: JobState
    replayed: bool = False


class StopRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=200)
    drain_timeout_s: float | None = Field(default=None, ge=0)


def build_router(controller: DeploymentController) -> APIRouter:
    router = APIRouter(prefix="/api/v1/deployments", tags=["deployments (S3)"])
    errors: dict[int | str, dict[str, object]] = {
        c: {"model": ErrorResponse} for c in (400, 401, 403, 404, 409, 422, 503)
    }

    def accepted(
        d: Deployment, job_id: str, job_state: JobState, replayed: bool
    ) -> DeploymentAccepted:
        return DeploymentAccepted(
            id=d.id,
            observed_state=d.observed_state,
            status_url=f"/api/v1/deployments/{d.id}",
            job_id=job_id,
            job_state=job_state,
            replayed=replayed,
        )

    @router.post(
        "",
        response_model=DeploymentAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        responses=errors,
    )
    def create_deployment(
        body: CreateDeploymentRequest,
        response: Response,
        caller: Annotated[Caller, Depends(require_deployer)],
    ) -> DeploymentAccepted:
        d, job, created = controller.create(caller.id, body)
        response.headers["Location"] = f"/api/v1/deployments/{d.id}"
        return accepted(d, job.id, job.state, replayed=not created)

    @router.get("", response_model=list[Deployment], responses=errors)
    def list_deployments(_: Annotated[Caller, Depends(get_caller)]) -> list[Deployment]:
        return controller.store.list_deployments()

    @router.get("/{deployment_id}", response_model=Deployment, responses=errors)
    def get_deployment(deployment_id: str, _: Annotated[Caller, Depends(get_caller)]) -> Deployment:
        return controller.get(deployment_id)

    @router.post(
        "/{deployment_id}/drain",
        response_model=DeploymentAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        responses=errors,
    )
    def drain_deployment(
        deployment_id: str,
        _: Annotated[Caller, Depends(require_deployer)],
        body: StopRequest | None = None,
    ) -> DeploymentAccepted:
        b = body or StopRequest()
        d, job = controller.request_drain(
            deployment_id, b.reason or "OPERATOR_DRAIN", b.drain_timeout_s
        )
        return accepted(d, job.id, job.state, replayed=False)

    @router.post(
        "/{deployment_id}/stop",
        response_model=DeploymentAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        responses=errors,
    )
    def stop_deployment(
        deployment_id: str,
        _: Annotated[Caller, Depends(require_deployer)],
        body: StopRequest | None = None,
    ) -> DeploymentAccepted:
        b = body or StopRequest()
        d, job = controller.request_stop(deployment_id, b.reason or "OPERATOR_STOP")
        return accepted(d, job.id, job.state, replayed=False)

    return router
