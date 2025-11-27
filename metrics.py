# metrics.py
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from contextlib import contextmanager
from sqlalchemy import create_engine

# ---- Load Environment ----
load_dotenv()
PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")

# Use SQLAlchemy engine for cleaner integration with pandas
engine = create_engine(PG_DSN)

# ---- DB connection helper ----
@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()

# ---- Core aggregation ----
def compute_aggregates():
    query = """
    SELECT
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
        SUM(CASE WHEN success THEN 0 ELSE 1 END)::float / COUNT(*) AS error_rate
    FROM mem0_metrics
    WHERE COALESCE(ts, NOW()) > NOW() - INTERVAL '7 days'
    GROUP BY function_name, provider_llm, model_llm, provider_vectorstore, ts_hour
    ORDER BY ts_hour DESC;
    """
    # pandas handles SQLAlchemy engine cleanly
    df = pd.read_sql(query, engine)
    return df

# ---- Upsert summary ----
def upsert_summary(df):
    if df.empty:
        print("No metrics to aggregate.")
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS mem0_metrics_summary (
        id SERIAL PRIMARY KEY,
        ts_hour TIMESTAMP,
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
        error_rate DOUBLE PRECISION
    );
    """

    unique_constraint = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'unique_summary_key'
        ) THEN
            ALTER TABLE mem0_metrics_summary
            ADD CONSTRAINT unique_summary_key
            UNIQUE (ts_hour, function_name, provider_llm, model_llm, provider_vectorstore);
        END IF;
    END $$;
    """

    insert_sql = """
    INSERT INTO mem0_metrics_summary (
        ts_hour, function_name, provider_llm, model_llm, provider_vectorstore,
        usage_count, avg_latency_ms, latency_p95,
        avg_cpu_percent, avg_mem_used_mb,
        avg_embed_latency, embed_latency_p95,
        avg_vector_latency, vector_latency_p95,
        avg_prompt_tokens, avg_total_tokens,
        cache_hit_ratio_avg, error_rate
    ) VALUES (
        %(ts_hour)s, %(function_name)s, %(provider_llm)s, %(model_llm)s, %(provider_vectorstore)s,
        %(usage_count)s, %(avg_latency_ms)s, %(latency_p95)s,
        %(avg_cpu_percent)s, %(avg_mem_used_mb)s,
        %(avg_embed_latency)s, %(embed_latency_p95)s,
        %(avg_vector_latency)s, %(vector_latency_p95)s,
        %(avg_prompt_tokens)s, %(avg_total_tokens)s,
        %(cache_hit_ratio_avg)s, %(error_rate)s
    )
    ON CONFLICT (ts_hour, function_name, provider_llm, model_llm, provider_vectorstore)
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
        error_rate = EXCLUDED.error_rate;
    """

    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(unique_constraint)
            for _, row in df.iterrows():
                cur.execute(insert_sql, row.to_dict())
        conn.commit()
    print(f"[aggregate] {len(df)} summary rows upserted ✅")


# ---- Main ----
if __name__ == "__main__":
    df = compute_aggregates()
    print(df.head(5))
    upsert_summary(df)
