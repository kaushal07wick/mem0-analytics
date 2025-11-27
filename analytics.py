import os
import time
import psutil
import hashlib
import functools
import psycopg2
from contextlib import contextmanager
from collections import defaultdict
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Target table switch: 'chat' or 'agent'
METRICS_TARGET = os.getenv("METRICS_TARGET", "chat").lower()

# Token cost (for estimated_cost_usd)
COST_RATE_PROMPT = float(os.getenv("COST_RATE_PROMPT_USD", 0))
COST_RATE_COMPLETION = float(os.getenv("COST_RATE_COMPLETION_USD", 0))


def get_table_name() -> str:
    if METRICS_TARGET == "agent":
        return "mem0_metrics_agent"
    return "mem0_metrics_chat"


# ----------------- DB -----------------
@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()


def insert_metric(row: dict):
    table = get_table_name()
    sql = f"""
    INSERT INTO {table} (
        function_name, duration_ms, success, error_message,
        provider_llm, model_llm, provider_embedder, model_embedder,
        provider_vectorstore, vectorstore_collection,
        cpu_percent, mem_used_mb, disk_read_kb, disk_write_kb,
        output_size, prompt_tokens, completion_tokens, total_tokens,
        embed_batch_size, embed_latency_ms, vector_backend, vector_latency_ms,
        cache_hit_ratio, user_id, agent_id, run_id,
        insert_count, memory_hash, estimated_cost_usd, ttfr_ms
    )
    VALUES (
        %(function_name)s, %(duration_ms)s, %(success)s, %(error_message)s,
        %(provider_llm)s, %(model_llm)s, %(provider_embedder)s, %(model_embedder)s,
        %(provider_vectorstore)s, %(vectorstore_collection)s,
        %(cpu_percent)s, %(mem_used_mb)s, %(disk_read_kb)s, %(disk_write_kb)s,
        %(output_size)s, %(prompt_tokens)s, %(completion_tokens)s, %(total_tokens)s,
        %(embed_batch_size)s, %(embed_latency_ms)s, %(vector_backend)s, %(vector_latency_ms)s,
        %(cache_hit_ratio)s, %(user_id)s, %(agent_id)s, %(run_id)s,
        %(insert_count)s, %(memory_hash)s, %(estimated_cost_usd)s, %(ttfr_ms)s
    );
    """
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, row)
        conn.commit()


# ----------------- HELPERS -----------------
def _name(obj):
    try:
        return obj.__class__.__name__
    except Exception:
        return None


def _cfg(obj, key):
    try:
        cfg = getattr(obj, "config", None)
        if isinstance(cfg, dict):
            return cfg.get(key)
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump().get(key)
    except Exception:
        return None
    return None


def providers_snapshot(memory: Memory):
    llm = getattr(memory, "llm", None)
    emb = getattr(memory, "embedding_model", None)
    vs = getattr(memory, "vector_store", None)
    return {
        "provider_llm": _name(llm),
        "model_llm": _cfg(llm, "model"),
        "provider_embedder": _name(emb),
        "model_embedder": _cfg(emb, "model"),
        "provider_vectorstore": _name(vs),
        "vectorstore_collection": (
            _cfg(getattr(memory, "config", None), "collection_name")
            or _cfg(vs, "collection_name")
        ),
    }


def extract_token_usage(output, memory: Memory):
    try:
        if isinstance(output, dict):
            usage = output.get("usage")
            if isinstance(usage, dict):
                return (
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                )
        if hasattr(output, "usage"):
            u = getattr(output, "usage")
            if isinstance(u, dict):
                return (
                    u.get("prompt_tokens", 0),
                    u.get("completion_tokens", 0),
                    u.get("total_tokens", 0),
                )
    except Exception:
        pass

    try:
        usage = getattr(memory.llm, "_last_usage", None)
        if isinstance(usage, dict):
            return (
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )
    except Exception:
        pass
    return (0, 0, 0)


_cache_hits = defaultdict(int)
_cache_total = defaultdict(int)


def update_cache_metrics(fn_name, query, result):
    key = (fn_name, str(query))
    _cache_total[key] += 1
    if _cache_total[key] > 1:
        _cache_hits[key] += 1
    return _cache_hits[key] / _cache_total[key]


# ----------------- WRAPS -----------------
def wrap_embedder(embedder):
    if not embedder or not hasattr(embedder, "embed"):
        return embedder
    if getattr(embedder, "_mem0_wrapped", False):
        return embedder

    orig = embedder.embed

    @functools.wraps(orig)
    def _wrapped(text, task):
        start = time.time()
        batch = len(text) if isinstance(text, list) else 1
        out = orig(text, task)
        elapsed = (time.time() - start) * 1000.0
        setattr(embedder, "_last_embed_batch", batch)
        setattr(embedder, "_last_embed_latency", elapsed)
        return out

    embedder.embed = _wrapped
    embedder._mem0_wrapped = True
    return embedder


def wrap_vectorstore(vs):
    if not vs:
        return vs
    if getattr(vs, "_mem0_wrapped", False):
        return vs

    def wrap_method(method, name):
        @functools.wraps(method)
        def _wrapped(*args, **kwargs):
            start = time.time()
            first_result_time = None
            out = method(*args, **kwargs)
            if hasattr(out, "__iter__"):
                first_result_time = time.time()
            elapsed = (time.time() - start) * 1000.0
            ttfr = (first_result_time - start) * 1000.0 if first_result_time else elapsed
            setattr(vs, "_last_vector_latency", elapsed)
            setattr(vs, "_last_vector_backend", vs.__class__.__name__)
            setattr(vs, "_last_ttfr", ttfr)
            return out
        return _wrapped

    for n in ["insert", "search", "update", "delete", "reset", "list", "get"]:
        if hasattr(vs, n):
            setattr(vs, n, wrap_method(getattr(vs, n), n))

    vs._mem0_wrapped = True
    return vs


def wrap_llm(llm):
    if not llm or not hasattr(llm, "generate_response"):
        return llm
    if getattr(llm, "_mem0_wrapped", False):
        return llm

    orig = llm.generate_response

    @functools.wraps(orig)
    def _wrapped(*args, **kwargs):
        start = time.time()
        resp = orig(*args, **kwargs)
        elapsed = (time.time() - start) * 1000.0
        setattr(llm, "_last_llm_latency_ms", elapsed)
        usage = None
        try:
            if isinstance(resp, dict) and "usage" in resp:
                usage = resp["usage"]
            elif hasattr(resp, "usage"):
                usage = resp.usage
        except Exception:
            pass
        setattr(llm, "_last_usage", usage or {})
        return resp

    llm.generate_response = _wrapped
    llm._mem0_wrapped = True
    return llm


# ----------------- MAIN TRACK DECORATOR -----------------
def track(memory: Memory, fn_name: str):
    orig = getattr(memory, fn_name)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        wrap_embedder(memory.embedding_model)
        wrap_vectorstore(memory.vector_store)
        wrap_llm(memory.llm)

        process = psutil.Process(os.getpid())
        io_before = process.io_counters()
        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = time.time()

        meta = providers_snapshot(memory)
        user_id = kwargs.get("user_id")
        agent_id = kwargs.get("agent_id")
        run_id = kwargs.get("run_id")

        try:
            out = orig(*args, **kwargs)
            duration = (time.time() - t0) * 1000.0
            io_after = process.io_counters()
            mem_after = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent(interval=0.05)
            disk_read_kb = (io_after.read_bytes - io_before.read_bytes) / 1024.0
            disk_write_kb = (io_after.write_bytes - io_before.write_bytes) / 1024.0
            mem_used_mb = max(0.0, mem_after - mem_before)
            output_size = len(str(out)) if out is not None else 0

            prompt_tokens, completion_tokens, total_tokens = extract_token_usage(out, memory)
            embed_batch_size = getattr(memory.embedding_model, "_last_embed_batch", None)
            embed_latency_ms = getattr(memory.embedding_model, "_last_embed_latency", None)
            vector_backend = getattr(memory.vector_store, "_last_vector_backend", None)
            vector_latency_ms = getattr(memory.vector_store, "_last_vector_latency", None)
            ttfr_ms = getattr(memory.vector_store, "_last_ttfr", None)

            estimated_cost = (
                (prompt_tokens * COST_RATE_PROMPT) +
                (completion_tokens * COST_RATE_COMPLETION)
            ) / 1000.0 if (COST_RATE_PROMPT or COST_RATE_COMPLETION) else None

            cache_hit_ratio = (
                update_cache_metrics(fn_name, kwargs.get("query"), out)
                if fn_name == "search" else None
            )

            insert_count = 0
            memory_hash = None
            if fn_name == "add" and args:
                payload = str(args[0])
                insert_count = len(args[0]) if isinstance(args[0], list) else 1
                memory_hash = hashlib.sha1(payload.encode()).hexdigest()

            insert_metric({
                "function_name": fn_name,
                "duration_ms": duration,
                "success": True,
                "error_message": None,
                **meta,
                "cpu_percent": cpu_percent,
                "mem_used_mb": mem_used_mb,
                "disk_read_kb": disk_read_kb,
                "disk_write_kb": disk_write_kb,
                "output_size": output_size,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "embed_batch_size": embed_batch_size,
                "embed_latency_ms": embed_latency_ms,
                "vector_backend": vector_backend,
                "vector_latency_ms": vector_latency_ms,
                "cache_hit_ratio": cache_hit_ratio,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "insert_count": insert_count,
                "memory_hash": memory_hash,
                "estimated_cost_usd": estimated_cost,
                "ttfr_ms": ttfr_ms,
            })

            return out

        except Exception as e:
            duration = (time.time() - t0) * 1000.0
            insert_metric({
                "function_name": fn_name,
                "duration_ms": duration,
                "success": False,
                "error_message": str(e)[:1000],
                **meta,
                "cpu_percent": None,
                "mem_used_mb": None,
                "disk_read_kb": None,
                "disk_write_kb": None,
                "output_size": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "embed_batch_size": None,
                "embed_latency_ms": None,
                "vector_backend": None,
                "vector_latency_ms": None,
                "cache_hit_ratio": None,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "insert_count": None,
                "memory_hash": None,
                "estimated_cost_usd": None,
                "ttfr_ms": None,
            })
            raise

    return wrapper


def patch_memory(memory: Memory):
    for fn in ["add", "search", "get", "get_all", "update", "delete", "delete_all", "reset"]:
        if hasattr(memory, fn):
            setattr(memory, fn, track(memory, fn))
    wrap_embedder(memory.embedding_model)
    wrap_vectorstore(memory.vector_store)
    wrap_llm(memory.llm)
    return memory
