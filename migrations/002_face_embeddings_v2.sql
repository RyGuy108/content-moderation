-- Face embeddings v2: [SFace(128) | dlib(128)] = 256 floats (see modpipe/image/faces.py).
-- Older 128-d rows are kept but ignored by the matcher until the identity is re-enrolled.
ALTER TABLE identity_embeddings DROP CONSTRAINT IF EXISTS identity_embeddings_vec_check;
ALTER TABLE identity_embeddings DROP CONSTRAINT IF EXISTS identity_embeddings_vec_dims;
ALTER TABLE identity_embeddings ADD CONSTRAINT identity_embeddings_vec_dims
    CHECK (array_length(vec, 1) IN (128, 256));
ALTER TABLE identity_embeddings ADD COLUMN IF NOT EXISTS model text NOT NULL DEFAULT 'dlib-resnet-v1';
