# 🧠 Mem0 Analytics

> **Real-time analytics and monitoring infrastructure for the Mem0 ecosystem.**
> Plug it in once — and it automatically tracks every memory, model, vector store, and embedder you use.

[![PyPI](https://img.shields.io/pypi/v/mem0-analytics.svg?color=blue)](https://pypi.org/project/mem0-analytics/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-local%20metrics-lightgrey?logo=sqlite)](https://sqlite.org/)
[![PostHog](https://img.shields.io/badge/PostHog-cloud%20dashboards-orange?logo=posthog)](https://posthog.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)](#-contributing)

---

## 🧩 Overview

**Mem0 Analytics** is the **official data, analytics, and monitoring layer** for [Mem0](https://github.com/mem0ai/mem0).
It automatically traces every memory interaction, measures latency and efficiency across the entire stack,
and presents insights through a **rich in-terminal dashboard** or **PostHog cloud visualization**.

No setup, no configuration — just:

📦 Install via PyPI
```bash
pip install mem0-analytics
```

and **Mem0 Analytics automatically activates**.
Every `add`, `search`, `update`, `reset`, and `query` operation is tracked — across **all** supported LLMs, embedders, and vector stores.

---

## ⚙️ What It Does

* 🧠 **Autoinstruments Mem0** — wraps every memory call transparently
* ⚡ **Tracks performance** — latency, tail (P95), TTFR, and system load
* 💾 **Monitors all layers** — LLM, embedder, and vector database
* 🔁 **Aggregates KPIs** every 60 s locally (SQLite store)
* 📊 **Visualizes metrics** in a live, auto-updating terminal dashboard
* ☁️ **Optionally syncs** to [PostHog](https://posthog.com) for team dashboards

---

## 🖥 Dashboard

![dashboard](./static/terminal.png)

Real-time monitoring of:

* ⚡ **Latency (avg & P95)** by operation
* 🧩 **Embedding & Vector performance**
* 💾 **Cache effectiveness**
* 🧠 **TTFR (Time-to-First-Response)**
* 🧮 **Success, error, and resource metrics**
* ✅ **System stability indicator**

Runs completely local — powered by `rich`.
No servers, no dependencies beyond SQLite.

---

## ☁️ Cloud Analytics (Optional)

For org-wide tracking, enable **PostHog sync**:

```bash
export POSTHOG_API_KEY=<your_key>
export POSTHOG_HOST=https://app.posthog.com
```

Analytics are automatically batched and sent every minute.

---

## 📊 Metrics Tracked

| Category              | Metrics                                    | Description                  |
| --------------------- | ------------------------------------------ | ---------------------------- |
| **Performance**       | `avg_latency_ms`, `latency_p95`, `ttfr_ms` | End-to-end and tail latency  |
| **Embedder / Vector** | `avg_embed_latency`, `avg_vector_latency`  | Stage-wise breakdown         |
| **Efficiency**        | `cache_effectiveness`, `usage_count`       | Cache reuse and throughput   |
| **System Health**     | `cpu_percent`, `mem_used_mb`               | Runtime system stats         |
| **Reliability**       | `success_rate`, `error_rate`               | Stability and health signals |

---

## 🧱 Architecture

```
Mem0 (any model, vector, embedder)
   ↓
mem0-analytics → captures metrics automatically
   ↓
SQLite (~/.mem0_metrics.db) → local store
   ↓
Live CLI Dashboard  ←  Aggregator updates every 60 s
   ↓
(Optional) PostHog sync for cloud dashboards
```

> **Local-first, privacy-safe, fully offline by default.**

---

## 🚀 Quick Start

```bash
pip install mem0 mem0-analytics
```

That’s it — analytics auto-activates with Mem0.

### View the live dashboard

```bash
python -m mem0_analytics.dashboard
```

Data is stored locally at:

```
~/.mem0_metrics.db
```

Updated automatically every minute.

---

## 🧠 Ecosystem Coverage

**Mem0 Analytics** supports **all major backends** out of the box:

| Layer             | Supported                                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| **LLMs**          | OpenAI (`gpt-4o`, `gpt-5-nano`), Ollama (`smollm2`, `smollm2:135m`), Claude, Gemini, Groq, Llama, DeepSeek, etc. |
| **Vector Stores** | Qdrant, ChromaDB, FAISS, Weaviate, Pinecone, Milvus, Redis, LanceDB                                              |
| **Embedders**     | OpenAI, Ollama, Hugging Face, Sentence-Transformers, InstructorXL, BGE, etc.                                     |

If it works with Mem0 — **it’s already tracked** by Mem0 Analytics.

---

## 🔬 Engineering Highlights

* 🪶 Lightweight (no external DB required)
* 🧱 Built on SQLite + `rich` for local telemetry
* 🔁 Background aggregator with rolling KPIs
* ☁️ Optional PostHog sync for teams
* 🧩 Pluggable architecture (add any provider)
* 💡 Minimal overhead — <1 ms per operation

---

## 🧭 Roadmap

* [x] Local SQLite metrics layer
* [x] Terminal dashboard
* [x] PostHog publishing
* [ ] Cost & token usage metrics
* [ ] Prometheus exporter
* [ ] Alerting / anomaly detection
* [ ] Multi-agent comparison mode

---

## 🤝 Contributing

Contributions are open — help extend analytics across new backends, metrics, or visualizations.

## 📜 License

Released under the **MIT License**.
See [`LICENSE`](./LICENSE) for details.
