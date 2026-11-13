-- Why an item is in the review queue: the detectors flagged it, or it is a random audit of an
-- allowed check. Audits are the only way false negatives ever reach a human, which is what the
-- retraining loop (modpipe/feedback.py) needs.
ALTER TABLE review_items ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'flagged';
ALTER TABLE review_items DROP CONSTRAINT IF EXISTS review_items_source_check;
ALTER TABLE review_items ADD CONSTRAINT review_items_source_check CHECK (source IN ('flagged', 'audit'));
