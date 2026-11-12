"""The privileged deployment worker: the only ASIO process allowed to use the Docker socket.

    ASIO_DATABASE_URL=postgresql://... ASIO_S1_BASE_URL=http://... \
        uv run python -m asio_deployment.worker

It claims DEPLOY / STOP_DEPLOY jobs (S5 adds its workspace handlers to the same runner), and every
`expiry_check_s` it starts drains for serving deployments whose S1 allocation is about to end.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time

from .contracts import JobKind
from .controller import DeploymentController
from .runner import JobRunner
from .runtime import DockerRuntime
from .s1 import HttpS1Client
from .store import PostgresStore

log = logging.getLogger("asio_deployment.worker")


def build(dsn: str) -> tuple[DeploymentController, JobRunner]:
    store = PostgresStore(dsn)
    controller = DeploymentController(store, s1=HttpS1Client.from_env(), runtime=DockerRuntime())
    runner = JobRunner(store, worker_id=f"s3-{socket.gethostname()}-{os.getpid()}")
    runner.register(JobKind.DEPLOY, controller.handle_deploy)
    runner.register(JobKind.STOP_DEPLOY, controller.handle_stop)
    return controller, runner


def main() -> None:
    ap = argparse.ArgumentParser(description="ASIO S3 deployment worker")
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--expiry-check-s", type=float, default=15.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    controller, runner = build(os.environ["ASIO_DATABASE_URL"])
    next_expiry = 0.0
    while True:
        if time.monotonic() >= next_expiry:
            for dep in controller.enqueue_expiring():
                log.info("serving lease ending: drain started for %s", dep)
            next_expiry = time.monotonic() + a.expiry_check_s
        job = runner.run_once()
        if job is None:
            time.sleep(a.poll_s)
        else:
            log.info(
                "job %s %s (attempt %d) -> %s",
                job.id,
                job.kind,
                job.attempt,
                (runner.store.get_job(job.id) or job).state,
            )


if __name__ == "__main__":
    main()
