import os
import time
import json
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
COST_WEIGHT = float(os.getenv("COST_LAT_WEIGHT", 0.5))
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
engine = create_engine(PG_DSN)


@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()


# ----------------- AGGREGATION -----------------
def compute_aggregates():
    query = """
    SELECT
        source,
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
        AVG(cache_hit_ratio) AS cache_hit_ratio_avg,
        AVG(estimated_cost_usd) AS avg_cost_usd,
        SUM(CASE WHEN success THEN 0 ELSE 1 END)::float / COUNT(*) AS error_rate
    FROM (
        SELECT 'chat' AS source, * FROM mem0_metrics_chat
        UNION ALL
        SELECT 'agent' AS source, * FROM mem0_metrics_agent
    ) combined
    WHERE COALESCE(ts, NOW()) > NOW() - INTERVAL '2 hours'
    GROUP BY source, function_name, provider_llm, model_llm, provider_vectorstore, ts_minute, ts_hour
    ORDER BY ts_minute DESC;
    """
    return pd.read_sql(query, engine)


# ----------------- KPI COMPUTATION -----------------
def compute_kpis(df):
    if df.empty:
        return df
    df["cost_latency_frontier"] = df["latency_p95"].fillna(0) + COST_WEIGHT * df["avg_cost_usd"].fillna(0)
    df["token_efficiency"] = (df["avg_latency_ms"] / df["avg_total_tokens"].replace(0, None)) * 1000
    df["vector_contribution"] = df["avg_vector_latency"] / df["avg_latency_ms"].replace(0, None)
    df["cache_effectiveness"] = df["cache_hit_ratio_avg"].fillna(0)
    df["reliability_index"] = (
        (1 - df["error_rate"].fillna(0))
        * (1 / df["latency_p95"].replace(0, None))
        * (1 / df["avg_cost_usd"].replace(0, None))
    ).fillna(0)
    return df


# ----------------- UPSERT TO SUMMARY -----------------
def upsert_summary(df):
    if df.empty:
        print(f"[aggregate] No metrics found at {datetime.now()}")
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS mem0_metrics_summary_all (
        id SERIAL PRIMARY KEY,
        ts_minute TIMESTAMP,
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
        avg_vector_latency DOUBLE PRECISION,
        avg_prompt_tokens DOUBLE PRECISION,
        avg_total_tokens DOUBLE PRECISION,
        cache_hit_ratio_avg DOUBLE PRECISION,
        avg_cost_usd DOUBLE PRECISION,
        error_rate DOUBLE PRECISION,
        cost_latency_frontier DOUBLE PRECISION,
        token_efficiency DOUBLE PRECISION,
        vector_contribution DOUBLE PRECISION,
        cache_effectiveness DOUBLE PRECISION,
        reliability_index DOUBLE PRECISION
    );
    """

    constraint = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'unique_summary_key_sched'
        ) THEN
            ALTER TABLE mem0_metrics_summary_all
            ADD CONSTRAINT unique_summary_key_sched
            UNIQUE (ts_hour, source, function_name, provider_llm, model_llm, provider_vectorstore);
        END IF;
    END $$;
    """

    insert_sql = """
    INSERT INTO mem0_metrics_summary_all (
        ts_minute, ts_hour, source, function_name, provider_llm, model_llm, provider_vectorstore,
        usage_count, avg_latency_ms, latency_p95, avg_cpu_percent, avg_mem_used_mb,
        avg_embed_latency, avg_vector_latency, avg_prompt_tokens, avg_total_tokens,
        cache_hit_ratio_avg, avg_cost_usd, error_rate,
        cost_latency_frontier, token_efficiency, vector_contribution, cache_effectiveness, reliability_index
    )
    VALUES (
        %(ts_minute)s, %(ts_hour)s, %(source)s, %(function_name)s, %(provider_llm)s, %(model_llm)s, %(provider_vectorstore)s,
        %(usage_count)s, %(avg_latency_ms)s, %(latency_p95)s, %(avg_cpu_percent)s, %(avg_mem_used_mb)s,
        %(avg_embed_latency)s, %(avg_vector_latency)s, %(avg_prompt_tokens)s, %(avg_total_tokens)s,
        %(cache_hit_ratio_avg)s, %(avg_cost_usd)s, %(error_rate)s,
        %(cost_latency_frontier)s, %(token_efficiency)s, %(vector_contribution)s, %(cache_effectiveness)s, %(reliability_index)s
    )
    ON CONFLICT (ts_hour, source, function_name, provider_llm, model_llm, provider_vectorstore)
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
        cache_hit_ratio_avg = EXCLUDED.cache_hit_ratio_avg,
        avg_cost_usd = EXCLUDED.avg_cost_usd,
        error_rate = EXCLUDED.error_rate,
        cost_latency_frontier = EXCLUDED.cost_latency_frontier,
        token_efficiency = EXCLUDED.token_efficiency,
        vector_contribution = EXCLUDED.vector_contribution,
        cache_effectiveness = EXCLUDED.cache_effectiveness,
        reliability_index = EXCLUDED.reliability_index;
    """

    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()
            cur.execute(constraint)
            conn.commit()
            for _, row in df.iterrows():
                cur.execute(insert_sql, row.to_dict())
            conn.commit()

    print(f"[aggregate] Upserted {len(df)} rows at {datetime.now():%H:%M:%S}")


# ----------------- LOCAL SAVE -----------------
def save_locally(df):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"metrics_{ts}.csv")
    parquet_path = os.path.join(DATA_DIR, "metrics_latest.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"[local] Saved {csv_path} and updated {parquet_path}")


# ----------------- POSTHOG SYNC -----------------
def safe_float(x):
    try:
        return float(x) if x is not None else 0.0
    except Exception:
        return 0.0

def safe_int(x):
    try:
        return int(x) if x is not None else 0
    except Exception:
        return 0

def push_to_posthog(df):
    if df.empty or not POSTHOG_API_KEY:
        return
    headers = {"Authorization": f"Bearer {POSTHOG_API_KEY}", "Content-Type": "application/json"}

    events = []
    for _, row in df.iterrows():
        events.append({
            "event": "mem0_kpi_update",
            "distinct_id": row.get("function_name", "unknown"),
            "properties": {
                "source": row.get("source"),
                "function": row.get("function_name"),
                "provider_llm": row.get("provider_llm"),
                "model_llm": row.get("model_llm"),
                "vectorstore": row.get("provider_vectorstore"),
                "latency_ms": safe_float(row.get("avg_latency_ms")),
                "cost_usd": safe_float(row.get("avg_cost_usd")),
                "error_rate": safe_float(row.get("error_rate")),
                "cache_effectiveness": safe_float(row.get("cache_effectiveness")),
                "reliability_index": safe_float(row.get("reliability_index")),
                "usage_count": safe_int(row.get("usage_count")),
                "ts_minute": str(row.get("ts_minute")),
            },
        })

    payload = {"api_key": POSTHOG_API_KEY, "batch": events}
    try:
        r = requests.post(f"{POSTHOG_URL}/batch/", json=payload, timeout=10)
        if r.status_code == 200:
            print(f"[posthog] Synced {len(events)} KPI events ✅")
        else:
            print(f"[posthog] Error: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[posthog] Failed to sync: {e}")


# ----------------- SCHEDULER LOOP -----------------
def run_scheduler(interval_sec=60):
    print(f"[scheduler] Running Mem0 analytics every {interval_sec}s")
    while True:
        try:
            df = compute_aggregates()
            df = compute_kpis(df)
            upsert_summary(df)
            save_locally(df)
            push_to_posthog(df)
        except Exception as e:
            print(f"[scheduler] Error: {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    run_scheduler(interval_sec=60)
