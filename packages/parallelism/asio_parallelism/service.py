"""Turns the calibrated planner (`core/`, measured on gpu-02, vLLM 0.31) into ParallelismPlans."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from .contracts import ParallelismPlan, PlanRequest
from .core import plan as core_plan

GPU_MODEL = "A100-SXM4-80GB"
GPU_MEM_MIB = 81920
IMAGE_PROFILE = "vllm-0.31-a100"


class PlanError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


class PlannerService:
    def __init__(self) -> None:
        self._plans: dict[str, ParallelismPlan] = {}
        self._lock = threading.Lock()

    def get(self, plan_id: str) -> ParallelismPlan | None:
        with self._lock:
            return self._plans.get(plan_id)

    def plan(self, req: PlanRequest) -> ParallelismPlan:
        try:
            out = core_plan.plan(
                core_plan.PlanRequest(
                    model=req.hf_model,
                    prompt_tokens=req.workload.prompt_tokens,
                    output_tokens=req.workload.output_tokens,
                    tpot_p95_ms=req.sla.tpot_p95_ms,
                    gpus=list(range(req.gpu_count)),
                    target_concurrency=req.target_concurrency,
                    max_model_len=req.workload.max_model_len,
                    gpu_mem_util=req.gpu_memory_utilization,
                )
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise PlanError(
                422,
                "MODEL_UNKNOWN",
                f"cannot read the architecture of {req.hf_model!r}",
                {"hf_model": req.hf_model, "error": str(e)},
            ) from e

        now = datetime.now(UTC)
        warnings = list(out.get("warnings", []))
        if req.moe_placement_plan_id:
            warnings.append(
                "MoE placement plan recorded but not applied: expert parallelism is "
                "not in the performance model yet (CT-15 with S4)"
            )
        if out["status"] != "ok":
            raise PlanError(
                422,
                "GPU_CONFIGURATION_INFEASIBLE",
                "no TP/PP/DP layout fits memory and the SLA on the GPUs given",
                {
                    "gpu_count": req.gpu_count,
                    "explanation": out["explanation"],
                    "rejected": out.get("rejected", []),
                },
            )

        ch = out["chosen"]
        n = int(ch["gpus"])
        args = [a for a in ch["vllm_args"]]
        rated = round(ch["throughput_tok_s"] / req.workload.output_tokens, 2)
        plan = ParallelismPlan(
            id=f"plan_{uuid.uuid4().hex[:16]}",
            model_id=req.model_id,
            hf_model=req.hf_model,
            gpu_count=n,
            tp=ch["tp"],
            pp=ch["pp"],
            dp=ch["dp"],
            gpu_constraints={
                "count": n,
                "gpu_model": GPU_MODEL,
                "same_node": True,
                # gpu-02 has NVSwitch: every GPU pair is NVLink-connected, so any n GPUs work.
                "interconnect": "NVLINK_ANY_PAIR",
                "min_free_mib_per_gpu": int(req.gpu_memory_utilization * GPU_MEM_MIB),
            },
            moe_placement_plan_id=req.moe_placement_plan_id,
            feasible=True,
            constraints=[
                f"max {ch['concurrency_per_replica']} concurrent requests per engine "
                f"(--max-num-seqs) to hold TPOT p95 <= {req.sla.tpot_p95_ms} ms"
            ],
            explanation=out["explanation"],
            warnings=warnings,
            deployment_template={
                "backend": "VLLM",
                "image_profile": IMAGE_PROFILE,
                "vllm_args": args,
                "max_num_seqs": ch["concurrency_per_replica"],
                "max_concurrency_total": ch["concurrency_total"],
                "rated_capacity_rps": rated if rated > 0 else None,
                "predicted": {
                    "tpot_p95_ms": ch["tpot_p95_ms"],
                    "throughput_tok_s": round(ch["throughput_tok_s"]),
                    "per_gpu_tok_s": round(ch["per_gpu_tok_s"]),
                },
                "calibration": out["calibration"],
            },
            created_at=now,
        )
        with self._lock:
            self._plans[plan.id] = plan
        return plan
