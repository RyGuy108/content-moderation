-- Tenant API keys. Only a SHA-256 of the key is stored; the plaintext is shown once at creation.
CREATE TABLE IF NOT EXISTS api_keys (
    id          bigserial PRIMARY KEY,
    tenant      text NOT NULL,
    name        text NOT NULL,
    role        text NOT NULL DEFAULT 'client' CHECK (role IN ('client', 'reviewer', 'admin')),
    key_hash    text NOT NULL UNIQUE,
    prefix      text NOT NULL,                 -- first chars of the key, for humans and logs
    rate_per_s  real NOT NULL DEFAULT 20,      -- token bucket refill
    burst       integer NOT NULL DEFAULT 40,   -- token bucket size
    created_at  timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz,
    revoked_at  timestamptz
);
CREATE INDEX IF NOT EXISTS api_keys_tenant ON api_keys (tenant) WHERE revoked_at IS NULL;
