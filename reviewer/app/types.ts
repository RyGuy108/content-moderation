export type Reason = {
  label: string;
  score?: number;
  action: "allow" | "review" | "block";
  flags?: string[];
  evidence?: {
    parts?: { class: string; score: number; box: [number, number, number, number] }[];
    identity?: string;
    similarity?: number;
    box?: [number, number, number, number];
    logo?: string;
    inliers?: number;
    quad?: [number, number][];
    mentions?: string[];
    t?: number;
  };
};

export type Check = {
  id: string;
  tenant: string;
  stage: "pre" | "post";
  subject: "prompt" | "image" | "video" | "mixed";
  input: { prompt?: string | null; media?: string[]; magic_hour_project_id?: string | null };
  action: "allow" | "review" | "block";
  scores: Record<string, number>;
  reasons: Reason[];
  created_at: string;
};

export type Item = {
  id: number;
  priority: number;
  attempts: number;
  lease_until: string;
  claimed_by: string;
  source: "flagged" | "audit";   // audit = random sample of allowed checks
  media_urls: string[];   // signed, short-lived: /v1/media/<name>?exp=..&sig=..
  check: Check;
};

export type Stats = {
  pending: number;
  claimed: number;
  expired_leases: number;
  resolved: number;
  p50_time_to_decision_s: number | null;
};
