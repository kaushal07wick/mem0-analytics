import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from contextlib import contextmanager
from sqlalchemy import create_engine

load_dotenv()

PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
engine = create_engine(PG_DSN)

# Optional cost multipliers
COST_WEIGHT = float(os.getenv("COST_LAT_WEIGHT", 0.5))  # λ term for cost-latency frontier

@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()

# ----------------- BASE AGGREGATION -----------------
def compute_aggregates():
    query = """
    SELECT
        source,
        function_name,
        provider_llm,
        model_llm,
        provider_vectorstore,
        DATE_TRUNC('hour', COALESCE(ts, NOW())) AS ts_hour,
        COUNT(*) AS usage_count,
        AVG(duration_ms) AS avg_latency_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS latency_p95,
        AVG(cpu_percent) AS avg_cpu_percent,
        AVG(mem_used_mb) AS avg_mem_used_mb,
        AVG(embed_latency_ms) AS avg_embed_latency,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY embed_latency_ms) AS embed_latency_p95,
        AVG(vector_latency_ms) AS avg_vector_latency,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY vector_latency_ms) AS vector_latency_p95,
        AVG(prompt_tokens) AS avg_prompt_tokens,
        AVG(total_tokens) AS avg_total_tokens,
        AVG(cache_hit_ratio) AS cache_hit_ratio_avg,
        AVG(estimated_cost_usd) AS avg_cost_usd,
        SUM(CASE WHEN success THEN 0 ELSE 1 END)::float / COUNT(*) AS error_rate
    FROM (
        SELECT 'chat' AS source, * FROM mem0_metrics_chat
        UNION ALL
        SELECT 'agent' AS source, * FROM mem0_metrics_agent
    ) combined
    WHERE COALESCE(ts, NOW()) > NOW() - INTERVAL '7 days'
    GROUP BY source, function_name, provider_llm, model_llm, provider_vectorstore, ts_hour
    ORDER BY ts_hour DESC;
    """
    df = pd.read_sql(query, engine)
    return df

# ----------------- ADVANCED METRIC COMPUTATION -----------------
def compute_kpis(df):
    if df.empty:
        return df

    df["cost_latency_frontier"] = df["latency_p95"].fillna(0) + COST_WEIGHT * df["avg_cost_usd"].fillna(0)
    df["token_efficiency"] = (df["avg_latency_ms"] / df["avg_total_tokens"].replace(0, None)) * 1000
    df["vector_contribution"] = df["avg_vector_latency"] / df["avg_latency_ms"].replace(0, None)
    df["embed_efficiency"] = df["avg_embed_latency"] / df["avg_prompt_tokens"].replace(0, None)
    df["cache_effectiveness"] = df["cache_hit_ratio_avg"].fillna(0)
    df["reliability_index"] = (
        (1 - df["error_rate"].fillna(0))
        * (1 / df["latency_p95"].replace(0, None))
        * (1 / df["avg_cost_usd"].replace(0, None))
    )
    df["reliability_index"] = df["reliability_index"].fillna(0)
    return df

# ----------------- UPSERT INTO SUMMARY -----------------
def upsert_summary(df):
    if df.empty:
        print("No metrics to aggregate.")
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS mem0_metrics_summary_all (
        id SERIAL PRIMARY KEY,
        ts_hour TIMESTAMP,
        source TEXT,
        function_name TEXT,
        provider_llm TEXT,
        model_llm TEXT,
        provider_vectorstore TEXT,
        usage_count BIGINT,
        avg_latency_ms DOUBLE PRECISION,
        latency_p95 DOUBLE PRECISION,
        avg_cpu_percent DOUBLE PRECISION,
        avg_mem_used_mb DOUBLE PRECISION,
        avg_embed_latency DOUBLE PRECISION,
        embed_latency_p95 DOUBLE PRECISION,
        avg_vector_latency DOUBLE PRECISION,
        vector_latency_p95 DOUBLE PRECISION,
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
        reliability_index DOUBLE PRECISION
    );
    """

    constraint = """
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
    """

    insert_sql = """
    INSERT INTO mem0_metrics_summary_all (
        ts_hour, source, function_name, provider_llm, model_llm, provider_vectorstore,
        usage_count, avg_latency_ms, latency_p95, avg_cpu_percent, avg_mem_used_mb,
        avg_embed_latency, embed_latency_p95, avg_vector_latency, vector_latency_p95,
        avg_prompt_tokens, avg_total_tokens, cache_hit_ratio_avg, avg_cost_usd, error_rate,
        cost_latency_frontier, token_efficiency, vector_contribution, embed_efficiency,
        cache_effectiveness, reliability_index
    )
    VALUES (
        %(ts_hour)s, %(source)s, %(function_name)s, %(provider_llm)s, %(model_llm)s, %(provider_vectorstore)s,
        %(usage_count)s, %(avg_latency_ms)s, %(latency_p95)s, %(avg_cpu_percent)s, %(avg_mem_used_mb)s,
        %(avg_embed_latency)s, %(embed_latency_p95)s, %(avg_vector_latency)s, %(vector_latency_p95)s,
        %(avg_prompt_tokens)s, %(avg_total_tokens)s, %(cache_hit_ratio_avg)s, %(avg_cost_usd)s, %(error_rate)s,
        %(cost_latency_frontier)s, %(token_efficiency)s, %(vector_contribution)s, %(embed_efficiency)s,
        %(cache_effectiveness)s, %(reliability_index)s
    )
    ON CONFLICT (ts_hour, source, function_name, provider_llm, model_llm, provider_vectorstore)
    DO UPDATE SET
        usage_count = EXCLUDED.usage_count,
        avg_latency_ms = EXCLUDED.avg_latency_ms,
        latency_p95 = EXCLUDED.latency_p95,
        avg_cpu_percent = EXCLUDED.avg_cpu_percent,
        avg_mem_used_mb = EXCLUDED.avg_mem_used_mb,
        avg_embed_latency = EXCLUDED.avg_embed_latency,
        embed_latency_p95 = EXCLUDED.embed_latency_p95,
        avg_vector_latency = EXCLUDED.avg_vector_latency,
        vector_latency_p95 = EXCLUDED.vector_latency_p95,
        avg_prompt_tokens = EXCLUDED.avg_prompt_tokens,
        avg_total_tokens = EXCLUDED.avg_total_tokens,
        cache_hit_ratio_avg = EXCLUDED.cache_hit_ratio_avg,
        avg_cost_usd = EXCLUDED.avg_cost_usd,
        error_rate = EXCLUDED.error_rate,
        cost_latency_frontier = EXCLUDED.cost_latency_frontier,
        token_efficiency = EXCLUDED.token_efficiency,
        vector_contribution = EXCLUDED.vector_contribution,
        embed_efficiency = EXCLUDED.embed_efficiency,
        cache_effectiveness = EXCLUDED.cache_effectiveness,
        reliability_index = EXCLUDED.reliability_index;
    """

    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(constraint)
            for _, row in df.iterrows():
                cur.execute(insert_sql, row.to_dict())
        conn.commit()
    print(f"[aggregate] {len(df)} combined summary rows upserted ✅")


# ----------------- MAIN -----------------
if __name__ == "__main__":
    df = compute_aggregates()
    df = compute_kpis(df)
    print(df.head(5))
    upsert_summary(df)
