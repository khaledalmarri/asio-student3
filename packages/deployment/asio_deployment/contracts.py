"""PROVISIONAL copies of shared models (contract §2.2, §4.2, §4.5, §4.6).

S3 owns `Deployment`, `Job` runner semantics and the WRITE side of `ServingPool`. `Allocation`
is S1's. None of these is in `packages/common` yet, so S3 codes against the contract text here,
with the same field names S2 uses in `packages/sla/asio_sla/contracts.py`. When S1 publishes the
canonical classes, replace this module with re-exports from `packages/common`.

Fields the contract leaves open are marked "S3 assumption" and listed in the README.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResourceClass(StrEnum):
    GPU_SERVING = "GPU_SERVING"
    ROUTING_ONLY = "ROUTING_ONLY"


class Backend(StrEnum):
    VLLM = "VLLM"
    DYNAMO = "DYNAMO"
    SWITCHYARD = "SWITCHYARD"


class Readiness(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"


class DesiredState(StrEnum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class ObservedState(StrEnum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


TERMINAL_STATES = {ObservedState.STOPPED, ObservedState.FAILED}


class JobKind(StrEnum):
    DEPLOY = "DEPLOY"
    STOP_DEPLOY = "STOP_DEPLOY"
    START_WORKSPACE = "START_WORKSPACE"
    STOP_WORKSPACE = "STOP_WORKSPACE"


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class AllocationKind(StrEnum):
    SERVING = "SERVING"
    RESERVATION = "RESERVATION"
    EXPERIMENT = "EXPERIMENT"
    MAINTENANCE = "MAINTENANCE"


class AllocationState(StrEnum):
    COMMITTED = "COMMITTED"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"
    RELEASED = "RELEASED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class Allocation(BaseModel):
    """Contract §4.2, read side only. S1 is authoritative; S3 never writes it."""

    model_config = ConfigDict(extra="allow")

    id: str
    kind: AllocationKind
    owner_id: str | None = None
    gpu_uuids: list[str]
    node_id: str | None = None
    start_at: datetime
    end_at: datetime
    state: AllocationState
    version: int = 0


class ServingPool(BaseModel):
    """Contract §4.6. S3 is the single writer; S1 serves the read API and caches it for S2."""

    model_config = ConfigDict(extra="allow")

    id: str
    deployment_id: str
    allocation_id: str | None = None
    resource_class: ResourceClass
    backend: Backend
    endpoint_url: str
    supported_model_ids: list[str]
    readiness: Readiness
    rated_capacity_rps: float | None = Field(default=None, gt=0)
    health_checked_at: datetime | None = None
    generation: int = 0
    updated_at: datetime | None = None


class DeploymentConfig(BaseModel):
    """`Deployment.config`. The contract example only fixes `served_model_name` and `port`;
    everything else is an S3 assumption needed to build a GPU-pinned vLLM command."""

    model_config = ConfigDict(extra="allow")

    served_model_name: str
    port: int = Field(ge=1024, le=65535)
    hf_model: str | None = Field(
        default=None, description="Hugging Face repo id of the weights; defaults to model_id"
    )
    model_path: str | None = Field(
        default=None, description="Container path of the weights; default: cached HF snapshot"
    )
    image_profile: str = "vllm-0.31-a100"
    vllm_args: list[str] = Field(default_factory=list)
    rated_capacity_rps: float | None = Field(default=None, gt=0)
    endpoint_url: str | None = Field(
        default=None, description="ROUTING_ONLY only: the upstream entry point (no container)"
    )


class Deployment(BaseModel):
    """Contract §4.5."""

    id: str
    owner_id: str
    request_ref: str
    resource_class: ResourceClass
    allocation_id: str | None = None
    parallelism_plan_id: str | None = None
    model_id: str
    backend: Backend
    desired_state: DesiredState = DesiredState.RUNNING
    observed_state: ObservedState = ObservedState.PENDING
    pool_id: str
    gpu_uuids: list[str] = Field(default_factory=list)
    generation: int = 0
    last_error: str | None = None
    config: DeploymentConfig
    container_name: str | None = None
    created_at: datetime
    updated_at: datetime


class Job(BaseModel):
    """Contract §2.2 durable job. Columns beyond the contract list are S3 assumptions for
    S1's migration: `max_attempts`, `run_after`, `worker_id`, `last_error`, `payload`."""

    id: str
    kind: JobKind
    owner_module: str
    request_ref: str
    allocation_id: str | None = None
    target_ref: str
    state: JobState = JobState.QUEUED
    generation: int = 0
    lease_until: datetime | None = None
    attempt: int = 0
    max_attempts: int = 5
    run_after: datetime
    worker_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    last_error: str | None = None
    shutdown_verified_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
