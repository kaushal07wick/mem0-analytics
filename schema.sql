-- ==============================
-- Base Metrics Table
-- ==============================
CREATE TABLE IF NOT EXISTS mem0_metrics (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP DEFAULT NOW(),

    -- Core operation metadata
    function_name TEXT,
    duration_ms DOUBLE PRECISION,
    success BOOLEAN,
    error_message TEXT,

    -- LLM and Embedder info
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

    -- Tokens / embeddings
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
    run_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_mem0_fn ON mem0_metrics(function_name);
CREATE INDEX IF NOT EXISTS idx_mem0_user ON mem0_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_mem0_success ON mem0_metrics(success);
CREATE INDEX IF NOT EXISTS idx_mem0_ts ON mem0_metrics(ts);

-- Ensure all expected columns exist even after prior versions
ALTER TABLE mem0_metrics
ADD COLUMN IF NOT EXISTS ts TIMESTAMP DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER,
ADD COLUMN IF NOT EXISTS completion_tokens INTEGER,
ADD COLUMN IF NOT EXISTS total_tokens INTEGER,
ADD COLUMN IF NOT EXISTS embed_batch_size INTEGER,
ADD COLUMN IF NOT EXISTS embed_latency_ms DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS vector_backend TEXT,
ADD COLUMN IF NOT EXISTS vector_latency_ms DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS cache_hit_ratio DOUBLE PRECISION;


-- ==============================
-- Summary Aggregation Table
-- ==============================
CREATE TABLE IF NOT EXISTS mem0_metrics_summary (
    id SERIAL PRIMARY KEY,

    -- Aggregated timestamps
    ts_minute TIMESTAMP,
    ts_hour TIMESTAMP,

    -- Key identifiers
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
    error_rate DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Add unique key for upsert operations (minute-level resolution)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'unique_summary_ts'
    ) THEN
        ALTER TABLE mem0_metrics_summary
        ADD CONSTRAINT unique_summary_ts
        UNIQUE (ts_minute, function_name, provider_llm, model_llm, provider_vectorstore);
    END IF;
END $$;
