import os, time, math, sqlite3, pandas as pd, numpy as np, threading
from datetime import datetime

DB_PATH = os.path.expanduser("~/.mem0_metrics.db")
AGG_INTERVAL = 60
pd.options.future.no_silent_downcasting = True


def get_conn():
    path = os.path.abspath(DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def ensure_kpi_table():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mem0_kpi (
            ts TIMESTAMP,
            function_name TEXT,
            provider_llm TEXT,
            model_llm TEXT,
            provider_embedder TEXT,
            model_embedder TEXT,
            provider_vectorstore TEXT,
            vectorstore_collection TEXT,
            usage_count INTEGER,
            avg_latency_ms REAL,
            latency_p95 REAL,
            avg_cpu_percent REAL,
            avg_mem_used_mb REAL,
            avg_embed_latency REAL,
            avg_vector_latency REAL,
            avg_prompt_tokens REAL,
            avg_total_tokens REAL,
            avg_cost_usd REAL,
            cache_effectiveness REAL,
            error_rate REAL,
            success_rate REAL
        );
        """)
        conn.commit()


def safe_float(x):
    try:
        if x in (None, np.nan, pd.NA):
            return 0.0
        xf = float(x)
        return 0.0 if math.isnan(xf) or math.isinf(xf) else xf
    except:
        return 0.0


def collect_raw():
    with get_conn() as conn:
        try:
            q = """
            SELECT
                function_name,
                provider_llm,
                model_llm,
                provider_embedder,
                model_embedder,
                provider_vectorstore,
                vectorstore_collection,
                duration_ms,
                cpu_percent,
                mem_used_mb,
                embed_latency_ms,
                vector_latency_ms,
                prompt_tokens,
                total_tokens,
                cache_hit_ratio,
                estimated_cost_usd,
                success
            FROM mem0_met;
            """
            return pd.read_sql(q, conn)
        except:
            return pd.DataFrame()


def compute_kpis(df):
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby([
        "function_name", "provider_llm", "model_llm",
        "provider_embedder", "model_embedder",
        "provider_vectorstore", "vectorstore_collection"
    ])

    rows = []
    for keys, g in grouped:
        d = g["duration_ms"].dropna().sort_values()
        n = len(d)
        p95 = float(d.iloc[int(0.95 * n) - 1]) if n > 0 else 0.0
        rows.append({
            "ts": datetime.now(),
            "function_name": keys[0],
            "provider_llm": keys[1],
            "model_llm": keys[2],
            "provider_embedder": keys[3],
            "model_embedder": keys[4],
            "provider_vectorstore": keys[5],
            "vectorstore_collection": keys[6],
            "usage_count": n,
            "avg_latency_ms": safe_float(g["duration_ms"].mean()),
            "latency_p95": safe_float(p95),
            "avg_cpu_percent": safe_float(g["cpu_percent"].mean()),
            "avg_mem_used_mb": safe_float(g["mem_used_mb"].mean()),
            "avg_embed_latency": safe_float(g["embed_latency_ms"].mean()),
            "avg_vector_latency": safe_float(g["vector_latency_ms"].mean()),
            "avg_prompt_tokens": safe_float(g["prompt_tokens"].mean()),
            "avg_total_tokens": safe_float(g["total_tokens"].mean()),
            "avg_cost_usd": safe_float(g["estimated_cost_usd"].mean()),
            "cache_effectiveness": safe_float(g["cache_hit_ratio"].mean()),
            "error_rate": safe_float((g["success"] == 0).mean()),
            "success_rate": safe_float((g["success"] == 1).mean()),
        })
    return pd.DataFrame(rows)


def save_to_db(df):
    if df.empty:
        return
    ensure_kpi_table()
    with get_conn() as conn:
        df.to_sql("mem0_kpi", conn, if_exists="append", index=False)
        conn.commit()


def _loop():
    ensure_kpi_table()
    while True:
        df = collect_raw()
        if not df.empty:
            df_out = compute_kpis(df)
            save_to_db(df_out)
        time.sleep(AGG_INTERVAL)


def start_background_aggregator():
    threading.Thread(target=_loop, daemon=True).start()
