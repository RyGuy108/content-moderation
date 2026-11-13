-- Content moderation pipeline schema. Postgres is the source of truth for every decision,
-- the review queue, and the protected-identity / logo registries. Redis only holds the job
-- queue and a decision cache that can be dropped at any time.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS checks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant          text NOT NULL,
    stage           text NOT NULL CHECK (stage IN ('pre', 'post')),
    subject         text NOT NULL CHECK (subject IN ('prompt', 'image', 'video', 'mixed')),
    content_hash    text NOT NULL,
    input           jsonb NOT NULL,              -- prompt text, media urls, magic hour project id
    status          text NOT NULL DEFAULT 'done' CHECK (status IN ('queued', 'running', 'done', 'error')),
    action          text CHECK (action IN ('allow', 'review', 'block')),
    final_action    text CHECK (final_action IN ('allow', 'block')),  -- after human review
    scores          jsonb,
    reasons         jsonb,
    policy_version  text,
    model_versions  jsonb,
    latency_ms      real,
    frames_checked  int,
    error           text,
    webhook_url     text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz
);
CREATE INDEX IF NOT EXISTS checks_tenant_time ON checks (tenant, created_at DESC);
CREATE INDEX IF NOT EXISTS checks_hash ON checks (content_hash);

CREATE TABLE IF NOT EXISTS review_items (
    id           bigserial PRIMARY KEY,
    check_id     uuid NOT NULL UNIQUE REFERENCES checks(id),
    tenant       text NOT NULL,
    priority     int NOT NULL,                  -- higher first: csam_risk > real_person_sexual > ...
    status       text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'claimed', 'resolved')),
    claimed_by   text,
    lease_until  timestamptz,
    attempts     int NOT NULL DEFAULT 0,
    decision     text CHECK (decision IN ('allow', 'block')),
    labels       text[],                        -- reviewer-corrected labels, exported for retraining
    note         text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    resolved_at  timestamptz
);
-- the claim query walks this index; resolved rows drop out of it
CREATE INDEX IF NOT EXISTS review_open ON review_items (priority DESC, created_at)
    WHERE status <> 'resolved';

CREATE TABLE IF NOT EXISTS protected_identities (
    id           text PRIMARY KEY,
    tenant       text,                          -- NULL = protected for every tenant
    display_name text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS identity_embeddings (
    id           bigserial PRIMARY KEY,
    identity_id  text NOT NULL REFERENCES protected_identities(id) ON DELETE CASCADE,
    vec          real[] NOT NULL CHECK (array_length(vec, 1) = 128),
    source       text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS logo_refs (
    id           text PRIMARY KEY,
    tenant       text,
    display_name text NOT NULL,
    image        bytea NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id      bigserial PRIMARY KEY,
    actor   text NOT NULL,
    action  text NOT NULL,
    target  text,
    detail  jsonb,
    at      timestamptz NOT NULL DEFAULT now()
);

-- Workers keep the face gallery and logo index in memory; a change bumps this and NOTIFYs
-- 'registry_changed' so every worker reloads without a restart.
CREATE TABLE IF NOT EXISTS registry_version (one boolean PRIMARY KEY DEFAULT true CHECK (one), v bigint NOT NULL);
INSERT INTO registry_version VALUES (true, 1) ON CONFLICT DO NOTHING;
