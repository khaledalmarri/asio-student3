"""ParallelismPlan (contract §4.4), owned by S3.

`deployment_template` is what `POST /api/v1/deployments` needs in `config` (vLLM args, concurrency
cap, rated capacity), and `gpu_constraints` is what S3 passes to S1 when requesting the allocation:
the plan comes *before* S1 grants GPU UUIDs (§7.2), so it never names GPUs itself.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceLabel(StrEnum):
    MOCKED = "MOCKED"
    SIMULATED = "SIMULATED"
    REAL_GPU = "REAL_GPU"
    NOT_VALIDATED = "NOT_VALIDATED"


class Workload(BaseModel):
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    max_model_len: int = Field(default=4096, gt=0)


class SLATarget(BaseModel):
    tpot_p95_ms: float = Field(gt=0)


class PlanRequest(BaseModel):
    """`POST /api/v1/parallelism/plan` body (S3 assumption until reviewed)."""

    model_id: str = Field(min_length=1, description="ASIO model id / served name, e.g. qwen-demo")
    hf_model: str = Field(min_length=1, description="architecture: HF repo id or config.json path")
    workload: Workload
    sla: SLATarget
    gpu_count: int = Field(ge=1, le=8, description="most GPUs the deployment may use")
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=0.95)
    target_concurrency: int | None = Field(default=None, gt=0)
    moe_placement_plan_id: str | None = None


class ParallelismPlan(BaseModel):
    id: str
    model_id: str
    hf_model: str
    gpu_count: int
    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    ep: int = 1
    gpu_constraints: dict[str, Any] = Field(default_factory=dict)
    moe_placement_plan_id: str | None = None
    feasible: bool
    reason_codes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    deployment_template: dict[str, Any] = Field(default_factory=dict)
    evidence_label: EvidenceLabel = EvidenceLabel.SIMULATED
    created_at: datetime
