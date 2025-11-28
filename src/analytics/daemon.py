import os
import time
import math
import pandas as pd
import psycopg2
import requests
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from contextlib import contextmanager
from datetime import datetime

load_dotenv()

PG_DSN = os.getenv("PG_DSN")
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")
POSTHOG_URL = os.getenv("POSTHOG_URL", "https://app.posthog.com")
COST_WEIGHT = float(os.getenv("COST_LAT_WEIGHT", 0.5))
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
engine = create_engine(PG_DSN)
pd.options.future.no_silent_downcasting = True


@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()


def compute_aggregates(since_ts=None):
    time_filter = ""
    if since_ts:
        time_filter = f"WHERE ts > '{since_ts}'"

    query = f"""
    SELECT
        'chat' AS source,
        function_name,
        provider_llm,
        model_llm,
        provider_embedder,
        model_embedder,
        provider_vectorstore,
        vectorstore_collection,
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
        AVG(cache_hit_ratio) AS avg_cache_hit_ratio,
        AVG(estimated_cost_usd) AS avg_cost_usd,
        SUM(CASE WHEN success THEN 0 ELSE 1 END)::float / COUNT(*) AS error_rate
    FROM mem0_metrics_chat
    {time_filter}
    GROUP BY function_name, provider_llm, model_llm,
             provider_embedder, model_embedder,
             provider_vectorstore, vectorstore_collection,
             ts_minute, ts_hour
    ORDER BY ts_minute DESC;
    """
    try:
        df = pd.read_sql(query, engine)
        df = df.convert_dtypes().infer_objects(copy=False).replace({pd.NA: 0, np.nan: 0})
        print(f"[daemon] Aggregated {len(df)} rows (since {since_ts or 'beginning'})")
        return df
    except Exception as e:
        print(f"[daemon] Query error: {e}")
        return pd.DataFrame()


def compute_kpis(df):
    if df.empty:
        return df
    df = df.copy()
    df = df.replace([float("inf"), float("-inf")], 0).replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False)
    df["cost_latency_frontier"] = (
        df["latency_p95"].astype(float).fillna(0)
        + COST_WEIGHT * df["avg_cost_usd"].astype(float).fillna(0)
    )
    df["token_efficiency"] = (
        df["avg_latency_ms"].astype(float) / df["avg_total_tokens"].replace(0, pd.NA)
    ).fillna(0) * 1000
    df["vector_contribution"] = (
        df["avg_vector_latency"].astype(float) / df["avg_latency_ms"].replace(0, pd.NA)
    ).fillna(0)
    df["embed_efficiency"] = (
        df["avg_embed_latency"].astype(float) / df["avg_prompt_tokens"].replace(0, pd.NA)
    ).fillna(0)
    df["cache_effectiveness"] = df["avg_cache_hit_ratio"].astype(float).fillna(0)
    df["reliability_index"] = (
        (1 - df["error_rate"].astype(float).fillna(0))
        * (1 / df["latency_p95"].replace(0, pd.NA))
        * (1 / df["avg_cost_usd"].replace(0, pd.NA))
    ).astype(float).fillna(0)
    return df.infer_objects(copy=False)


def safe_float(x):
    import numbers
    try:
        if x is None:
            return 0.0
        if isinstance(x, (pd._libs.missing.NAType, np.floating)):
            return 0.0
        if isinstance(x, numbers.Number):
            if math.isnan(x) or math.isinf(x):
                return 0.0
            return float(x)
        xf = float(str(x))
        if math.isnan(xf) or math.isinf(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0


def safe_int(x):
    import numbers
    try:
        if x is None:
            return 0
        if isinstance(x, (pd._libs.missing.NAType, np.floating)):
            return 0
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, numbers.Number):
            if math.isnan(x) or math.isinf(x):
                return 0
            return int(x)
        return int(float(str(x)))
    except Exception:
        return 0


def insert_into_summary(df):
    if df.empty:
        return
    insert_sql = """
    INSERT INTO mem0_metrics_summary_all (
        ts_minute, ts_hour, source, function_name,
        provider_llm, model_llm, provider_vectorstore,
        usage_count, avg_latency_ms, latency_p95,
        avg_cpu_percent, avg_mem_used_mb,
        avg_embed_latency, avg_vector_latency,
        avg_prompt_tokens, avg_total_tokens,
        avg_cost_usd, error_rate, cache_effectiveness,
        reliability_index, cost_latency_frontier
    )
    VALUES (
        %(ts_minute)s, %(ts_hour)s, %(source)s, %(function_name)s,
        %(provider_llm)s, %(model_llm)s, %(provider_vectorstore)s,
        %(usage_count)s, %(avg_latency_ms)s, %(latency_p95)s,
        %(avg_cpu_percent)s, %(avg_mem_used_mb)s,
        %(avg_embed_latency)s, %(avg_vector_latency)s,
        %(avg_prompt_tokens)s, %(avg_total_tokens)s,
        %(avg_cost_usd)s, %(error_rate)s, %(cache_effectiveness)s,
        %(reliability_index)s, %(cost_latency_frontier)s
    )
    ON CONFLICT DO NOTHING;
    """
    with pg_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(insert_sql, {
                    "ts_minute": row.get("ts_minute"),
                    "ts_hour": row.get("ts_hour"),
                    "source": row.get("source"),
                    "function_name": row.get("function_name"),
                    "provider_llm": row.get("provider_llm"),
                    "model_llm": row.get("model_llm"),
                    "provider_vectorstore": row.get("provider_vectorstore"),
                    "usage_count": safe_int(row.get("usage_count")),
                    "avg_latency_ms": safe_float(row.get("avg_latency_ms")),
                    "latency_p95": safe_float(row.get("latency_p95")),
                    "avg_cpu_percent": safe_float(row.get("avg_cpu_percent")),
                    "avg_mem_used_mb": safe_float(row.get("avg_mem_used_mb")),
                    "avg_embed_latency": safe_float(row.get("avg_embed_latency")),
                    "avg_vector_latency": safe_float(row.get("avg_vector_latency")),
                    "avg_prompt_tokens": safe_float(row.get("avg_prompt_tokens")),
                    "avg_total_tokens": safe_float(row.get("avg_total_tokens")),
                    "avg_cost_usd": safe_float(row.get("avg_cost_usd")),
                    "error_rate": safe_float(row.get("error_rate")),
                    "cache_effectiveness": safe_float(row.get("cache_effectiveness")),
                    "reliability_index": safe_float(row.get("reliability_index")),
                    "cost_latency_frontier": safe_float(row.get("cost_latency_frontier")),
                })
        conn.commit()
    print(f"[db] Inserted {len(df)} rows into mem0_metrics_summary_all ✅")


def push_to_posthog(df):
    if df.empty or not POSTHOG_API_KEY:
        return
    df = df.replace([float("inf"), float("-inf")], 0).replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False)
    headers = {"Authorization": f"Bearer {POSTHOG_API_KEY}", "Content-Type": "application/json"}
    events = []
    for _, row in df.iterrows():
        events.append({
            "event": "mem0_chat_kpi_update",
            "distinct_id": row.get("function_name", "unknown"),
            "properties": {
                "function": row.get("function_name"),
                "provider_llm": row.get("provider_llm"),
                "model_llm": row.get("model_llm"),
                "vectorstore": row.get("provider_vectorstore"),
                "latency_ms": safe_float(row.get("avg_latency_ms")),
                "latency_p95": safe_float(row.get("latency_p95")),
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
            print(f"[posthog] Synced {len(events)} chat KPI events ✅")
        else:
            print(f"[posthog] Error: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[posthog] Failed to sync: {e}")


def save_locally(df):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DATA_DIR, f"metrics_chat_{ts}.csv")
    df.to_csv(path, index=False)
    print(f"[local] Saved {path}")


def run_daemon(interval_sec=60):
    print(f"[daemon] Running Mem0 chat analytics every {interval_sec}s")
    last_ts = None
    while True:
        try:
            df = compute_aggregates(since_ts=last_ts)
            if df.empty:
                print(f"[daemon] No new data at {datetime.now():%H:%M:%S}")
            else:
                last_ts = df["ts_minute"].max()
                df = compute_kpis(df)
                insert_into_summary(df)
                save_locally(df)
                push_to_posthog(df)
        except Exception as e:
            print(f"[daemon] Error: {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    run_daemon(interval_sec=60)
