-- PROPOSED schema for S1's migration (S1 owns migrations; S3 owns the runner semantics).
-- Column names follow contract §2.2 / §4.5 / §4.6. Columns marked "S3 assumption" are not in the
-- contract text yet and need S1 sign-off.

CREATE TABLE IF NOT EXISTS jobs (
    id                   text PRIMARY KEY,
    kind                 text NOT NULL CHECK (kind IN ('DEPLOY','STOP_DEPLOY','START_WORKSPACE','STOP_WORKSPACE')),
    owner_module         text NOT NULL,                 -- 'deployment' (S3) | 'workspaces' (S5) | 'platform' (S1 reconciler)
    request_ref          text NOT NULL,
    allocation_id        text,
    target_ref           text NOT NULL,                 -- deployment id or workspace id
    state                text NOT NULL CHECK (state IN ('QUEUED','RUNNING','SUCCEEDED','FAILED')),
    generation           integer NOT NULL DEFAULT 0,    -- fencing token: +1 on every claim
    lease_until          timestamptz,
    attempt              integer NOT NULL DEFAULT 0,
    max_attempts         integer NOT NULL DEFAULT 5,    -- S3 assumption
    run_after            timestamptz NOT NULL,          -- S3 assumption (retry backoff)
    worker_id            text,                          -- S3 assumption
    payload              jsonb NOT NULL DEFAULT '{}',   -- S3 assumption (e.g. drain timeout)
    result               jsonb,
    last_error           text,                          -- S3 assumption
    shutdown_verified_at timestamptz,
    created_at           timestamptz NOT NULL,
    updated_at           timestamptz NOT NULL,
    UNIQUE (owner_module, request_ref)
);
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs (state, run_after);
CREATE INDEX IF NOT EXISTS jobs_target_idx ON jobs (target_ref);

CREATE TABLE IF NOT EXISTS deployments (
    id                  text PRIMARY KEY,
    owner_id            text NOT NULL,
    request_ref         text NOT NULL,
    resource_class      text NOT NULL CHECK (resource_class IN ('GPU_SERVING','ROUTING_ONLY')),
    allocation_id       text,
    parallelism_plan_id text,
    model_id            text NOT NULL,
    backend             text NOT NULL CHECK (backend IN ('VLLM','DYNAMO','SWITCHYARD')),
    desired_state       text NOT NULL CHECK (desired_state IN ('RUNNING','STOPPED')),
    observed_state      text NOT NULL CHECK (observed_state IN
                          ('PENDING','STARTING','READY','DRAINING','STOPPING','STOPPED','FAILED')),
    pool_id             text NOT NULL,
    gpu_uuids           jsonb NOT NULL DEFAULT '[]',
    generation          integer NOT NULL DEFAULT 0,
    last_error          text,
    config              jsonb NOT NULL,
    container_name      text,                           -- S3 assumption
    created_at          timestamptz NOT NULL,
    updated_at          timestamptz NOT NULL,
    UNIQUE (owner_id, request_ref),                     -- caller-scoped request_ref (contract §3)
    CHECK (resource_class = 'ROUTING_ONLY' OR allocation_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS serving_pools (
    id                  text PRIMARY KEY,
    deployment_id       text NOT NULL UNIQUE,
    allocation_id       text,
    resource_class      text NOT NULL,
    backend             text NOT NULL,
    endpoint_url        text NOT NULL,
    supported_model_ids jsonb NOT NULL DEFAULT '[]',
    readiness           text NOT NULL CHECK (readiness IN ('STARTING','READY','DRAINING','UNHEALTHY','STOPPED')),
    rated_capacity_rps  double precision,
    health_checked_at   timestamptz,
    generation          integer NOT NULL DEFAULT 0,
    updated_at          timestamptz
);
