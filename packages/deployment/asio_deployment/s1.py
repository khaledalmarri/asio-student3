"""What S3 needs from S1 (contract §5.3, §5.5): read an allocation, activate it after a verified
launch, release it only with shutdown evidence, and invalidate S1's cached pool eligibility before
S3 stops anything.

`HttpS1Client` calls S1's routes. `InProcessS1Client` wraps S1's Python functions once both live in
the same FastAPI monolith. `FakeS1` stands in for S1 in tests and in the GPU smoke test (MOCKED).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .contracts import Allocation, AllocationState


class S1Error(Exception):
    """S1 refused or could not be reached. Transient unless `permanent` is set."""

    def __init__(self, message: str, *, status: int | None = None, permanent: bool = False):
        super().__init__(message)
        self.status = status
        self.permanent = permanent


class S1Client(Protocol):
    def get_allocation(self, allocation_id: str) -> Allocation: ...
    def activate_allocation(self, allocation_id: str, idempotency_key: str) -> None: ...
    def release_allocation(
        self, allocation_id: str, idempotency_key: str, evidence: dict[str, Any]
    ) -> None: ...
    def invalidate_pool(self, pool_id: str, minimum_generation: int, reason: str) -> bool:
        """True only when S1 acknowledges that no new request will be dispatched to the pool."""
        ...


class HttpS1Client:
    """S1 HTTP API. The invalidation path is an S3 assumption until S1 publishes it."""

    INVALIDATE_PATH = "/internal/v1/serving-pools/{pool_id}/invalidate"

    def __init__(self, base_url: str, token: str | None = None, timeout_s: float = 5.0) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s)

    @classmethod
    def from_env(cls) -> HttpS1Client:
        return cls(os.environ["ASIO_S1_BASE_URL"], os.environ.get("ASIO_S3_SERVICE_TOKEN"))

    def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise S1Error(f"S1 unreachable: {e}") from e
        if r.status_code >= 400:
            raise S1Error(
                f"S1 {method} {path} -> {r.status_code}: {r.text[:300]}",
                status=r.status_code,
                permanent=r.status_code in (400, 403, 404, 409, 422),
            )
        return r

    def get_allocation(self, allocation_id: str) -> Allocation:
        return Allocation.model_validate(
            self._call("GET", f"/api/v1/allocations/{allocation_id}").json()
        )

    def activate_allocation(self, allocation_id: str, idempotency_key: str) -> None:
        self._call(
            "POST",
            f"/api/v1/allocations/{allocation_id}/activate",
            headers={"Idempotency-Key": idempotency_key},
        )

    def release_allocation(
        self, allocation_id: str, idempotency_key: str, evidence: dict[str, Any]
    ) -> None:
        self._call(
            "POST",
            f"/api/v1/allocations/{allocation_id}/release",
            headers={"Idempotency-Key": idempotency_key},
            json={"reason": "S3_VERIFIED_SHUTDOWN", "shutdown_evidence": evidence},
        )

    def invalidate_pool(self, pool_id: str, minimum_generation: int, reason: str) -> bool:
        r = self._call(
            "POST",
            self.INVALIDATE_PATH.format(pool_id=pool_id),
            json={"minimum_generation": minimum_generation, "reason": reason},
        )
        return bool(r.json().get("acknowledged", False))


@dataclass
class InProcessS1Client:
    """Adapter for the monolith: pass S1's own callables in when `packages/platform` exists."""

    get_allocation_fn: Callable[[str], Allocation]
    activate_fn: Callable[[str, str], None]
    release_fn: Callable[[str, str, dict[str, Any]], None]
    invalidate_fn: Callable[[str, int, str], bool]

    def get_allocation(self, allocation_id: str) -> Allocation:
        return self.get_allocation_fn(allocation_id)

    def activate_allocation(self, allocation_id: str, idempotency_key: str) -> None:
        self.activate_fn(allocation_id, idempotency_key)

    def release_allocation(
        self, allocation_id: str, idempotency_key: str, evidence: dict[str, Any]
    ) -> None:
        self.release_fn(allocation_id, idempotency_key, evidence)

    def invalidate_pool(self, pool_id: str, minimum_generation: int, reason: str) -> bool:
        return self.invalidate_fn(pool_id, minimum_generation, reason)


@dataclass
class FakeS1:
    """Records every call. `ack_invalidation=False` simulates S1 not acknowledging."""

    allocations: dict[str, Allocation] = field(default_factory=dict)
    ack_invalidation: bool = True
    calls: list[tuple[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_allocation(self, allocation_id: str) -> Allocation:
        with self._lock:
            self.calls.append(("get", allocation_id))
            if allocation_id not in self.allocations:
                raise S1Error(f"allocation {allocation_id} not found", status=404, permanent=True)
            return self.allocations[allocation_id].model_copy()

    def activate_allocation(self, allocation_id: str, idempotency_key: str) -> None:
        with self._lock:
            self.calls.append(("activate", allocation_id))
            a = self.allocations[allocation_id]
            if a.state == AllocationState.COMMITTED:
                a.state = AllocationState.ACTIVE

    def release_allocation(
        self, allocation_id: str, idempotency_key: str, evidence: dict[str, Any]
    ) -> None:
        with self._lock:
            self.calls.append(("release", allocation_id))
            self.allocations[allocation_id].state = AllocationState.RELEASED

    def invalidate_pool(self, pool_id: str, minimum_generation: int, reason: str) -> bool:
        with self._lock:
            self.calls.append(("invalidate", (pool_id, minimum_generation, reason)))
            return self.ack_invalidation

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]
