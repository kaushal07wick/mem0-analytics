import os
import time
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
from contextlib import contextmanager
from datetime import datetime

load_dotenv()

PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")
POSTHOG_URL = os.getenv("POSTHOG_URL", "https://app.posthog.com")

engine = create_engine(PG_DSN)


# ---- PostgreSQL helpers ----
@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()


# ---- Aggregation Logic ----
def compute_aggregates():
    query = """
    SELECT
        function_name,
        provider_llm,
        model_llm,
        provider_vectorstore,
        DATE_TRUNC('minute', COALESCE(ts, NOW())) AS ts_minute,
        DATE_TRUNC('hour', COALESCE(ts, NOW())) AS ts_hour,
        COUNT(*) AS usage_count,
        AVG(duration_ms) AS avg_latency_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS latency_p95,
        AVG(cpu_percent) AS avg_cpu_percent,
        AVG(mem_used_mb) AS avg_mem_used_mb,
        AVG(embed_latency_ms) AS avg_embed_latency,
        AVG(vector_latency_ms) AS avg_vector_latency,
        AVG(prompt_tokens) AS avg_prompt_tokens,
        AVG(total_tokens) AS avg_total_tokens,
        SUM(CASE WHEN success THEN 0 ELSE 1 END)::float / COUNT(*) AS error_rate
    FROM mem0_metrics
    WHERE COALESCE(ts, NOW()) > NOW() - INTERVAL '2 hours'
    GROUP BY function_name, provider_llm, model_llm, provider_vectorstore, ts_minute, ts_hour
    ORDER BY ts_minute DESC;
    """
    df = pd.read_sql(query, engine)
    return df


# ---- Postgres upsert for summaries ----
def upsert_to_summary(df):
    if df.empty:
        print(f"[aggregate] No metrics found at {datetime.now()}")
        return

    # Step 1: Ensure the table exists with correct structure
    ddl = """
    CREATE TABLE IF NOT EXISTS mem0_metrics_summary (
        id SERIAL PRIMARY KEY,
        ts_minute TIMESTAMP,
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
        avg_vector_latency DOUBLE PRECISION,
        avg_prompt_tokens DOUBLE PRECISION,
        avg_total_tokens DOUBLE PRECISION,
        error_rate DOUBLE PRECISION
    );
    """

    # Step 2: Add unique constraint *after* table exists
    constraint_sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'unique_summary_ts'
        ) THEN
            BEGIN
                ALTER TABLE mem0_metrics_summary
                ADD CONSTRAINT unique_summary_ts
                UNIQUE (ts_minute, function_name, provider_llm, model_llm, provider_vectorstore);
            EXCEPTION WHEN others THEN
                RAISE NOTICE 'Constraint already exists or failed to create.';
            END;
        END IF;
    END $$;
    """

    # Step 3: Upsert data
    insert_sql = """
    INSERT INTO mem0_metrics_summary (
        ts_minute, ts_hour, function_name, provider_llm, model_llm, provider_vectorstore,
        usage_count, avg_latency_ms, latency_p95,
        avg_cpu_percent, avg_mem_used_mb,
        avg_embed_latency, avg_vector_latency,
        avg_prompt_tokens, avg_total_tokens, error_rate
    ) VALUES (
        %(ts_minute)s, %(ts_hour)s, %(function_name)s, %(provider_llm)s, %(model_llm)s, %(provider_vectorstore)s,
        %(usage_count)s, %(avg_latency_ms)s, %(latency_p95)s,
        %(avg_cpu_percent)s, %(avg_mem_used_mb)s,
        %(avg_embed_latency)s, %(avg_vector_latency)s,
        %(avg_prompt_tokens)s, %(avg_total_tokens)s, %(error_rate)s
    )
    ON CONFLICT (ts_minute, function_name, provider_llm, model_llm, provider_vectorstore)
    DO UPDATE SET
        usage_count = EXCLUDED.usage_count,
        avg_latency_ms = EXCLUDED.avg_latency_ms,
        latency_p95 = EXCLUDED.latency_p95,
        avg_cpu_percent = EXCLUDED.avg_cpu_percent,
        avg_mem_used_mb = EXCLUDED.avg_mem_used_mb,
        avg_embed_latency = EXCLUDED.avg_embed_latency,
        avg_vector_latency = EXCLUDED.avg_vector_latency,
        avg_prompt_tokens = EXCLUDED.avg_prompt_tokens,
        avg_total_tokens = EXCLUDED.avg_total_tokens,
        error_rate = EXCLUDED.error_rate;
    """

    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()  # commit table creation first
            cur.execute(constraint_sql)
            conn.commit()
            for _, row in df.iterrows():
                cur.execute(insert_sql, row.to_dict())
            conn.commit()

    print(f"[aggregate] Upserted {len(df)} rows at {datetime.now():%H:%M:%S}")


# ---- PostHog Sync ----
def push_to_posthog(df):
    if df.empty or not POSTHOG_API_KEY:
        return

    headers = {
        "Authorization": f"Bearer {POSTHOG_API_KEY}",
        "Content-Type": "application/json",
    }

    events = []
    for _, row in df.iterrows():
        events.append({
            "event": "mem0_function_usage",
            "distinct_id": row.get("function_name", "unknown"),
            "properties": {
                "function": row.get("function_name"),
                "provider_llm": row.get("provider_llm"),
                "model_llm": row.get("model_llm"),
                "latency_ms": float(row.get("avg_latency_ms", 0)),
                "error_rate": float(row.get("error_rate", 0)),
                "usage_count": int(row.get("usage_count", 0)),
                "ts_minute": str(row.get("ts_minute")),
            },
        })

    payload = {"api_key": POSTHOG_API_KEY, "batch": events}
    try:
        r = requests.post(f"{POSTHOG_URL}/batch/", json=payload, timeout=10)
        if r.status_code == 200:
            print(f"[posthog] Synced {len(events)} events ✅")
        else:
            print(f"[posthog] Error: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[posthog] Failed to sync: {e}")


# ---- Scheduler ----
def run_scheduler(interval_sec=60):
    print(f"[scheduler] Starting Mem0 analytics aggregator every {interval_sec}s...")
    while True:
        try:
            df = compute_aggregates()
            upsert_to_summary(df)
            push_to_posthog(df)
        except Exception as e:
            print(f"[scheduler] Error: {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    run_scheduler(interval_sec=60)
