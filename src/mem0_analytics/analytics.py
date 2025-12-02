import os, time, psutil, functools, sqlite3, threading, sys
from contextlib import contextmanager
from collections import defaultdict

DB_PATH = os.path.expanduser("~/.mem0_metrics.db")
COST_RATE_PROMPT = float(os.getenv("COST_RATE_PROMPT_USD", 0))
COST_RATE_COMPLETION = float(os.getenv("COST_RATE_COMPLETION_USD", 0))
ENABLED = os.getenv("MEM0_ANALYTICS_ENABLED", "true").lower() == "true"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

try:
    from mem0 import Memory, MemoryClient
    from mem0.embeddings.base import EmbeddingBase
except Exception:
    Memory = MemoryClient = None
    class EmbeddingBase: pass

try:
    from mem0_analytics.aggregate import start_background_aggregator
except Exception:
    start_background_aggregator = None


def get_conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    return c


def _init_db():
    with get_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS mem0_met (
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            function_name TEXT,
            duration_ms REAL,
            success BOOLEAN,
            error_message TEXT,
            provider_llm TEXT, model_llm TEXT,
            provider_embedder TEXT, model_embedder TEXT,
            provider_vectorstore TEXT, vectorstore_collection TEXT,
            cpu_percent REAL, mem_used_mb REAL,
            disk_read_kb REAL, disk_write_kb REAL,
            input_chars INTEGER, output_chars INTEGER,
            output_size_bytes INTEGER,
            prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
            embed_batch_size INTEGER, embed_latency_ms REAL,
            vector_backend TEXT, vector_latency_ms REAL, ttfr_ms REAL,
            cache_hit_ratio REAL, estimated_cost_usd REAL
        );
        """)
        c.execute("CREATE INDEX IF NOT EXISTS ix_met_fn ON mem0_met(function_name);")
        c.execute("CREATE INDEX IF NOT EXISTS ix_met_model ON mem0_met(model_llm);")
        c.commit()


@contextmanager
def db_conn():
    c = get_conn()
    try:
        yield c
    finally:
        c.close()


def insert_metric(row):
    with db_conn() as c:
        cols = ",".join(row.keys())
        vals = ",".join(["?"] * len(row))
        c.execute(f"INSERT INTO mem0_met ({cols}) VALUES ({vals})", tuple(row.values()))
        c.commit()


def _name(o): return getattr(o, "__class__", type(None)).__name__ or "unknown"


def _cfg(o, k):
    c = getattr(o, "config", None)
    if isinstance(c, dict): return c.get(k)
    if hasattr(c, k): return getattr(c, k)
    if hasattr(c, "model_dump"): return c.model_dump().get(k)
    return None


def find_embedder(m):
    for n in ("embedding_model", "embedder", "embedding", "embed_model"):
        if hasattr(m, n): return getattr(m, n)
    for _, v in vars(m).items():
        if isinstance(v, EmbeddingBase) or hasattr(v, "embed"): return v
    return None


def providers_snapshot(m):
    llm, emb, vs = getattr(m, "llm", None), find_embedder(m), getattr(m, "vector_store", None)
    vc = _cfg(getattr(m, "config", None), "collection_name") or _cfg(vs, "collection_name") or "unknown"
    return dict(
        provider_llm=_name(llm),
        model_llm=_cfg(llm, "model") or "unknown",
        provider_embedder=_name(emb),
        model_embedder=_cfg(emb, "model") or "unknown",
        provider_vectorstore=_name(vs),
        vectorstore_collection=vc
    )


def extract_token_usage(out, m):
    try:
        if isinstance(out, dict) and "usage" in out:
            u = out["usage"] or {}
            return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0)
        if hasattr(out, "usage") and isinstance(out.usage, dict):
            u = out.usage
            return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0)
    except:
        pass
    u = getattr(getattr(m, "llm", None), "_last_usage", {}) or {}
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0)


_cache_hits, _cache_total = defaultdict(int), defaultdict(int)


def update_cache(fn, q, _r):
    k = (fn, str(q))
    _cache_total[k] += 1
    if _cache_total[k] > 1:
        _cache_hits[k] += 1
    return _cache_hits[k] / _cache_total[k]


def safe_len(o):
    try:
        if o is None: return 0
        if isinstance(o, (bytes, bytearray)): return len(o)
        return len(str(o))
    except:
        return 0


def wrap_embedder(e):
    if not e or getattr(e, "_mem0_wrapped", False): return e
    if not hasattr(e, "embed"): e._mem0_wrapped = True; return e
    orig = e.embed
    @functools.wraps(orig)
    def _wrap(t, a=None):
        s = time.time()
        out = orig(t, a)
        e._last_embed_batch = len(t) if isinstance(t, list) else 1
        e._last_embed_latency = (time.time() - s) * 1000
        return out
    e.embed = _wrap
    e._mem0_wrapped = True
    return e


def wrap_vectorstore(vs):
    if not vs or getattr(vs, "_mem0_wrapped", False): return vs
    def wrap(m):
        @functools.wraps(m)
        def _w(*a, **k):
            s = time.time()
            out = m(*a, **k)
            el = (time.time() - s) * 1000
            vs._last_vector_latency = el
            vs._last_vector_backend = vs.__class__.__name__
            vs._last_ttfr = el
            return out
        return _w
    for n in ("insert","search","update","delete","reset","list","get"):
        if hasattr(vs, n): setattr(vs, n, wrap(getattr(vs, n)))
    vs._mem0_wrapped = True
    return vs


def wrap_llm(llm):
    if not llm or getattr(llm, "_mem0_wrapped", False): return llm
    for cand in ("generate_response", "generate", "__call__"):
        if hasattr(llm, cand):
            orig = getattr(llm, cand)
            @functools.wraps(orig)
            def _wrap(*a, **k):
                s = time.time()
                r = orig(*a, **k)
                llm._last_llm_latency_ms = (time.time() - s) * 1000
                if hasattr(r, "usage") and isinstance(r.usage, dict): llm._last_usage = r.usage
                elif isinstance(r, dict) and "usage" in r: llm._last_usage = r["usage"]
                return r
            setattr(llm, cand, _wrap)
            llm._mem0_wrapped = True
            return llm
    llm._mem0_wrapped = True
    return llm


def track(m, fn):
    o = getattr(m, fn)
    @functools.wraps(o)
    def w(*a, **k):
        emb = find_embedder(m)
        wrap_embedder(emb)
        wrap_vectorstore(getattr(m, "vector_store", None))
        wrap_llm(getattr(m, "llm", None))
        p = psutil.Process()
        io_b = p.io_counters()
        mem_b = p.memory_info().rss / (1024*1024)
        t0 = time.time()
        meta = providers_snapshot(m)
        try:
            out = o(*a, **k)
            d = (time.time() - t0) * 1000
            io_a = p.io_counters()
            mem_a = p.memory_info().rss / (1024*1024)
            pt, ct, tt = extract_token_usage(out, m)
            vs = getattr(m, "vector_store", None)
            insert_metric({
                **meta,
                "function_name": fn,
                "duration_ms": d,
                "success": True,
                "error_message": None,
                "cpu_percent": p.cpu_percent(interval=0.0),
                "mem_used_mb": max(0.0, mem_a - mem_b),
                "disk_read_kb": (io_a.read_bytes - io_b.read_bytes) / 1024,
                "disk_write_kb": (io_a.write_bytes - io_b.write_bytes) / 1024,
                "input_chars": safe_len(a) + safe_len(k),
                "output_chars": safe_len(out),
                "output_size_bytes": sys.getsizeof(out),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "embed_batch_size": getattr(emb, "_last_embed_batch", None),
                "embed_latency_ms": getattr(emb, "_last_embed_latency", None),
                "vector_backend": getattr(vs, "_last_vector_backend", None),
                "vector_latency_ms": getattr(vs, "_last_vector_latency", None),
                "ttfr_ms": getattr(vs, "_last_ttfr", None),
                "cache_hit_ratio": update_cache(fn, k.get("query"), out) if fn == "search" else None,
                "estimated_cost_usd": ((pt*COST_RATE_PROMPT)+(ct*COST_RATE_COMPLETION))/1000 if (COST_RATE_PROMPT or COST_RATE_COMPLETION) else None
            })
            return out
        except Exception as e:
            insert_metric({
                **meta,
                "function_name": fn,
                "duration_ms": (time.time() - t0) * 1000,
                "success": False,
                "error_message": str(e)[:300]
            })
            raise
    return w


def patch_memory(m):
    for fn in ("add","search","get","get_all","update","delete","delete_all","reset","query","retrieve","fetch","save","load","sync","stats","count","summary"):
        if hasattr(m, fn):
            setattr(m, fn, track(m, fn))
    wrap_embedder(find_embedder(m))
    wrap_vectorstore(getattr(m, "vector_store", None))
    wrap_llm(getattr(m, "llm", None))
    return m


def _autopatch():
    try:
        _init_db()
        if Memory and hasattr(Memory, "__init__"):
            o = Memory.__init__
            @functools.wraps(o)
            def _i(self, *a, **k):
                o(self, *a, **k)
                patch_memory(self)
            Memory.__init__ = _i
        if MemoryClient and hasattr(MemoryClient, "__init__"):
            o = MemoryClient.__init__
            @functools.wraps(o)
            def _ic(self, *a, **k): o(self, *a, **k)
            MemoryClient.__init__ = _ic
        if start_background_aggregator:
            start_background_aggregator()
    except:
        pass


if ENABLED:
    _autopatch()
