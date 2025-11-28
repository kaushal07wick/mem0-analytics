-- =======================================================
-- 1. Core Raw Metrics Table Template
-- =======================================================
DROP TABLE IF EXISTS mem0_metrics CASCADE;
DROP TABLE IF EXISTS mem0_metrics_chat CASCADE;
DROP TABLE IF EXISTS mem0_metrics_agent CASCADE;
DROP TABLE IF EXISTS mem0_metrics_summary CASCADE;
DROP TABLE IF EXISTS mem0_metrics_summary_all CASCADE;

-- Base schema used for both chat and agent tables
CREATE TABLE mem0_metrics (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP DEFAULT NOW(),

    -- Operation metadata
    function_name TEXT,
    duration_ms DOUBLE PRECISION,
    success BOOLEAN,
    error_message TEXT,

    -- LLM / embedder info
    provider_llm TEXT,
    model_llm TEXT,
    provider_embedder TEXT,
    model_embedder TEXT,

    -- Vector store info
    provider_vectorstore TEXT,
    vectorstore_collection TEXT,
    vector_backend TEXT,

    -- Resource usage
    cpu_percent DOUBLE PRECISION,
    mem_used_mb DOUBLE PRECISION,
    disk_read_kb DOUBLE PRECISION,
    disk_write_kb DOUBLE PRECISION,
    output_size BIGINT,

    -- Token metrics
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    embed_batch_size INTEGER,
    embed_latency_ms DOUBLE PRECISION,
    vector_latency_ms DOUBLE PRECISION,
    cache_hit_ratio DOUBLE PRECISION,

    -- Contextual identifiers
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,

    -- Additional analytics metadata
    insert_count INTEGER,
    memory_hash TEXT,
    estimated_cost_usd DOUBLE PRECISION,
    ttfr_ms DOUBLE PRECISION
);

CREATE INDEX idx_mem0_fn ON mem0_metrics(function_name);
CREATE INDEX idx_mem0_user ON mem0_metrics(user_id);
CREATE INDEX idx_mem0_success ON mem0_metrics(success);
CREATE INDEX idx_mem0_ts ON mem0_metrics(ts);

-- =======================================================
-- 2. Dedicated Tables for Chat and Agent Metrics
-- =======================================================
CREATE TABLE mem0_metrics_chat (
    LIKE mem0_metrics INCLUDING ALL
);

CREATE TABLE mem0_metrics_agent (
    LIKE mem0_metrics INCLUDING ALL
);

-- Optional separation by future context (multi-agent)
COMMENT ON TABLE mem0_metrics_chat IS 'Raw telemetry from mem0 chat client with analytics tracking';
COMMENT ON TABLE mem0_metrics_agent IS 'Raw telemetry from mem0 scrape agent with analytics tracking';

-- =======================================================
-- 3. Summary Aggregation Table (for daemon.py)
-- =======================================================
CREATE TABLE mem0_metrics_summary_all (
    id SERIAL PRIMARY KEY,
    ts_minute TIMESTAMP,
    ts_hour TIMESTAMP,
    source TEXT DEFAULT 'chat',

    -- Keys
    function_name TEXT,
    provider_llm TEXT,
    model_llm TEXT,
    provider_vectorstore TEXT,

    -- Aggregated metrics
    usage_count BIGINT,
    avg_latency_ms DOUBLE PRECISION,
    latency_p95 DOUBLE PRECISION,
    avg_cpu_percent DOUBLE PRECISION,
    avg_mem_used_mb DOUBLE PRECISION,
    avg_embed_latency DOUBLE PRECISION,
    avg_vector_latency DOUBLE PRECISION,
    avg_prompt_tokens DOUBLE PRECISION,
    avg_total_tokens DOUBLE PRECISION,
    cache_hit_ratio_avg DOUBLE PRECISION,
    avg_cost_usd DOUBLE PRECISION,
    error_rate DOUBLE PRECISION,
    cost_latency_frontier DOUBLE PRECISION,
    token_efficiency DOUBLE PRECISION,
    vector_contribution DOUBLE PRECISION,
    embed_efficiency DOUBLE PRECISION,
    cache_effectiveness DOUBLE PRECISION,
    reliability_index DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT NOW()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'unique_summary_key_all'
    ) THEN
        ALTER TABLE mem0_metrics_summary_all
        ADD CONSTRAINT unique_summary_key_all
        UNIQUE (ts_hour, source, function_name, provider_llm, model_llm, provider_vectorstore);
    END IF;
END $$;

COMMENT ON TABLE mem0_metrics_summary_all IS 'Aggregated hourly/minute metrics computed by daemon for PostHog sync';
