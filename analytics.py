# analytics.py
import time
import functools
import os
import psutil
import psycopg2
from contextlib import contextmanager
from collections import defaultdict
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

# ---- ENV ----
PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- DB HELPERS ----
@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()

def insert_metric(row: dict):
    sql = """
    INSERT INTO mem0_metrics
    (function_name, duration_ms, success, error_message,
     provider_llm, model_llm, provider_embedder, model_embedder,
     provider_vectorstore, vectorstore_collection,
     cpu_percent, mem_used_mb, disk_read_kb, disk_write_kb,
     output_size, prompt_tokens, completion_tokens, total_tokens,
     embed_batch_size, embed_latency_ms, vector_backend, vector_latency_ms,
     cache_hit_ratio, user_id, agent_id, run_id)
    VALUES
    (%(function_name)s, %(duration_ms)s, %(success)s, %(error_message)s,
     %(provider_llm)s, %(model_llm)s, %(provider_embedder)s, %(model_embedder)s,
     %(provider_vectorstore)s, %(vectorstore_collection)s,
     %(cpu_percent)s, %(mem_used_mb)s, %(disk_read_kb)s, %(disk_write_kb)s,
     %(output_size)s, %(prompt_tokens)s, %(completion_tokens)s, %(total_tokens)s,
     %(embed_batch_size)s, %(embed_latency_ms)s, %(vector_backend)s, %(vector_latency_ms)s,
     %(cache_hit_ratio)s, %(user_id)s, %(agent_id)s, %(run_id)s)
    """
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, row)
        conn.commit()

# ---- SAFE EXTRACTORS ----
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
            _cfg(getattr(memory, "config", None), "collection_name") or _cfg(vs, "collection_name")
        ),
    }

# ---- TOKEN EXTRACTION ----
def extract_token_usage(output):
    """Extract token counts from OpenAI-like responses."""
    try:
        if isinstance(output, dict):
            usage = output.get("usage")
            if usage:
                return (
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                )
        if hasattr(output, "usage"):
            usage = getattr(output, "usage", None)
            if usage and isinstance(usage, dict):
                return (
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                )
    except Exception:
        pass
    return (0, 0, 0)

# ---- EMBEDDING MONITOR ----
try:
    from mem0.embeddings.openai import OpenAIEmbedding
    orig_embed = OpenAIEmbedding.embed

    def tracked_embed(self, text, task):
        start = time.time()
        batch_size = len(text) if isinstance(text, list) else 1
        result = orig_embed(self, text, task)
        elapsed = (time.time() - start) * 1000
        print(f"[embed] {batch_size} embeddings in {elapsed:.2f} ms")
        self._last_embed_batch = batch_size
        self._last_embed_latency = elapsed
        return result

    OpenAIEmbedding.embed = tracked_embed
except Exception:
    pass

# ---- VECTORSTORE MONITOR ----
try:
    from mem0.vector_stores.base import VectorStoreBase

    def timed_call(method):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            out = method(self, *args, **kwargs)
            elapsed = (time.time() - start) * 1000
            self._last_vector_latency = elapsed
            self._last_vector_backend = self.__class__.__name__
            print(f"[vectorstore] {self._last_vector_backend}.{method.__name__} {elapsed:.2f} ms")
            return out
        return wrapper

    for attr in ["add", "query", "delete", "reset"]:
        if hasattr(VectorStoreBase, attr):
            setattr(VectorStoreBase, attr, timed_call(getattr(VectorStoreBase, attr)))
except Exception:
    pass

# ---- CACHE METRICS ----
_cache_hits = defaultdict(int)
_cache_total = defaultdict(int)

def update_cache_metrics(fn_name, query, result):
    """Approximate cache performance by repeated query detection."""
    key = (fn_name, str(query))
    _cache_total[key] += 1
    if _cache_total[key] > 1:
        _cache_hits[key] += 1
    return _cache_hits[key] / _cache_total[key]

# ---- MAIN TRACK DECORATOR ----
def track(memory: Memory, fn_name: str):
    orig = getattr(memory, fn_name)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
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
            cpu_percent = process.cpu_percent(interval=0.1)
            disk_read_kb = (io_after.read_bytes - io_before.read_bytes) / 1024
            disk_write_kb = (io_after.write_bytes - io_before.write_bytes) / 1024
            mem_used_mb = mem_after - mem_before
            output_size = len(str(out)) if out is not None else 0

            prompt_tokens, completion_tokens, total_tokens = extract_token_usage(out)
            embed_batch_size = getattr(memory.embedding_model, "_last_embed_batch", None)
            embed_latency_ms = getattr(memory.embedding_model, "_last_embed_latency", None)
            vector_backend = getattr(memory.vector_store, "_last_vector_backend", None)
            vector_latency_ms = getattr(memory.vector_store, "_last_vector_latency", None)

            cache_hit_ratio = (
                update_cache_metrics(fn_name, kwargs.get("query"), out)
                if fn_name == "search" else None
            )

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
            })

            print(f"[mem0-metrics] {fn_name} ok {duration:.2f} ms | CPU {cpu_percent:.1f}% | Mem Δ {mem_used_mb:.2f}MB")
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
            })
            print(f"[mem0-metrics] {fn_name} FAIL {duration:.2f} ms :: {e}")
            raise
    return wrapper

# ---- PATCH ----
def patch_memory(memory: Memory):
    for fn in ["add", "search", "get", "get_all", "update", "delete", "delete_all", "reset"]:
        if hasattr(memory, fn):
            setattr(memory, fn, track(memory, fn))
    return memory

# ---- DEMO ----
if __name__ == "__main__":
    print("[mem0-metrics] Running advanced tracker demo...")
    memory = patch_memory(Memory())
    user_id = "user_1"

    messages = [
        {"role": "user", "content": "What is computational geometry, give me the basic maths as well?"},
        {"role": "assistant", "content": "Computational geometry is a field of computer science and mathematics that focuses on developing algorithms and data structures to solve problems involving geometric shapes"},
    ]

    memory.add(messages, user_id=user_id)
    memory.search("What did I ask about computational geometry?", user_id=user_id)
    if hasattr(memory, "get_all"):
        memory.get_all(user_id=user_id)
    print("[mem0-metrics] demo complete ✅")
