# 🧠 Mem0 Analytics

> Real-time analytics intelligence for memory-driven AI systems

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/Postgres-analytics%20backend-blue?logo=postgresql)](https://www.postgresql.org/)
[![PostHog](https://img.shields.io/badge/PostHog-dashboard-orange?logo=posthog)](https://posthog.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](#-contributing)

---

### 🧩 Overview

**Mem0 Analytics** provides **observability, telemetry, and performance analytics** for the [Mem0](https://github.com/mem0ai/mem0) ecosystem — an intelligent memory layer for LLMs.

It captures **live metrics** from memory operations (`add`, `search`, etc.), aggregates data in **PostgreSQL**, and pushes **real-time KPIs** to **PostHog dashboards**.
Built for **engineers**, **data scientists**, and **infra teams** optimizing RAG and chat pipelines.

---

## 🚀 Live Dashboard

🔗 [**View Real-Time Mem0 Dashboard on PostHog →**](https://us.posthog.com/shared/0_gFtZ5fE8WhDNVXKlTHh2i4v31uSQ)

**Tracks:**

* ⚡ Latency (avg & P95) by model and provider
* 🧠 Embedding, vector, and LLM latency distribution
* 💾 Cache effectiveness and token efficiency
* 💰 Cost, token throughput, and reliability index
* 🧩 CPU & memory utilization per function

---

## ⚙️ Architecture

```mermaid
graph TD
    A[Mem0 Chat / Agent Apps] -->|Analytics wrapper| B[(PostgreSQL)]
    B -->|Aggregates per minute| C[Daemon (Aggregator)]
    C -->|Push batch metrics| D[PostHog Dashboard]
    D --> E[Insights / Alerts / Visuals]
```

**Core Components**

* `analytics.py` — instruments Mem0 calls, logs metrics to PostgreSQL
* `daemon.py` — aggregates data, computes KPIs, syncs to PostHog
* `schema.sql` — defines tables for raw & aggregated metrics

---

## 📊 Metrics Tracked

| Category                 | Metrics                                                       | Description                         |
| ------------------------ | ------------------------------------------------------------- | ----------------------------------- |
| **Performance**          | `latency_ms`, `latency_p95`, `ttfr_ms`                        | Total, tail, and cold-start latency |
| **Tokens & Cost**        | `prompt_tokens`, `completion_tokens`, `estimated_cost_usd`    | Token usage and per-call cost       |
| **Resource Utilization** | `cpu_percent`, `mem_used_mb`, `disk_read_kb`, `disk_write_kb` | System-level stats                  |
| **Reliability**          | `error_rate`, `reliability_index`                             | Operational stability               |
| **Efficiency**           | `cache_hit_ratio`, `token_efficiency`, `vector_contribution`  | Throughput and cache health         |

---

## 📈 Sample Insights (Live)

* **🚀 smolm2** is **4.7× faster** than gpt-5-nano
* **⚠️ gpt-4o-mini** shows **6.9× latency spikes** — circuit breaker recommended
* **💾 Cache hit rate <1%** — huge optimization opportunity
* **📊 Vector stores (Qdrant / ChromaDB)** perform <10 ms, no bottleneck
* **🧠 TTFR <10 ms** — zero cold-start overhead

---

## 🔧 Quick Start

```bash
# 1️⃣ Clone repo
git clone https://github.com/mem0ai/mem0-analytics.git
cd mem0-analytics

# 2️⃣ Configure environment
cp .env.example .env
# Add PG_DSN, POSTHOG_API_KEY, and other variables

# 3️⃣ Initialize database
psql -U <user> -d mem0_analytics -f schema.sql

# 4️⃣ Run analytics tracker
python analytics.py

# 5️⃣ Start the continuous aggregator
python daemon.py
```

---

## 💻 Example Dashboard Visuals

| Metric                     | Visualization | Insight                          |
| -------------------------- | ------------- | -------------------------------- |
| Avg & P95 Latency by Model | Line chart    | Detect tail performance drift    |
| Pipeline Breakdown         | Stacked bar   | Time in embedding → vector → LLM |
| Cache Hit Rate (%)         | Area          | Track caching improvements       |
| Token Usage vs Latency     | Scatter       | Efficiency across models         |
| CPU & Memory by Function   | Bar           | Resource footprint monitoring    |

---

## 🔬 Engineering Highlights

* Built with **PostgreSQL** + **SQLAlchemy**
* Real-time sync to **PostHog** via batch API
* Clean modular structure (daemon, analytics, schema)
* Configurable via `.env`
* CSV + Parquet data export for offline analysis
* Fully extensible for **custom metrics**

---

## 🧭 Roadmap & Future Scope
## 🧩 Integration Roadmap — LLMs & Vector Stores

### 🔮 Planned LLM Integrations

| Provider         | Model / API                        | Status       | Notes                                                      |
| ---------------- | ---------------------------------- | ------------ | ---------------------------------------------------------- |
| ✅ **OpenAI**     | `gpt-4o-mini`, `gpt-5-nano`        | ✅ Integrated | Fully instrumented, latency & cost tracked                 |
| ✅ **Ollama**     | `smollm2`, `smollm2:135m`          | ✅ Integrated | Local inference, cost-free tracking                        |
| 🔲 **Anthropic** | `claude-3-opus`, `claude-3-sonnet` | ⏳ Planned    | Add API latency & token-level cost                         |
| 🔲 **Groq**      | `mixtral`, `llama3-groq`           | ⏳ Planned    | Measure sub-10ms ultra-low latency benchmarks              |
| 🔲 **xAI**       | `Grok-2`                           | ⏳ Planned    | Integrate via REST, track reliability index                |
| 🔲 **Meta**      | `Llama-3.1`, `Llama-4` (local)     | ⏳ Planned    | Local benchmarking with Ollama + CPU usage metrics         |
| 🔲 **Google**    | `Gemini-2`                         | ⏳ Planned    | Compare cost-to-performance vs OpenAI                      |
| 🔲 **DeepSeek**  | `DeepSeek-Coder`, `DeepSeek-Chat`  | ⏳ Planned    | Token-efficient models to benchmark memory cost efficiency |

---

### 🧠 Planned Vector Store Integrations

| Vector Store        | Type           | Status       | Notes                                                    |
| ------------------- | -------------- | ------------ | -------------------------------------------------------- |
| ✅ **Qdrant**        | Remote (Rust)  | ✅ Integrated | Fastest in production (avg <10ms latency)                |
| ✅ **ChromaDB**      | Local (Python) | ✅ Integrated | Ideal for lightweight dev workloads                      |
| 🔲 **Pinecone**     | Cloud          | ⏳ Planned    | Enterprise-grade, multi-tenant metrics                   |
| 🔲 **Weaviate**     | Cloud/Local    | ⏳ Planned    | Measure hybrid query latency                             |
| 🔲 **Milvus**       | Local/Cluster  | ⏳ Planned    | Benchmark with high vector throughput                    |
| 🔲 **Redis Vector** | In-memory      | ⏳ Planned    | Low-latency cache-style retrieval benchmarking           |
| 🔲 **LanceDB**      | Local          | ⏳ Planned    | Evaluate performance with Arrow-based storage            |
| 🔲 **FAISS**        | Local          | ⏳ Planned    | Offline RAG experimentation and embedding cache baseline |

---

### 🧰 Additional Infrastructure Targets

| Category                           | Tool / Layer                        | Purpose |
| ---------------------------------- | ----------------------------------- | ------- |
| 🔲 **Prometheus + Grafana**        | Real-time resource observability    |         |
| 🔲 **Kubernetes Metrics Exporter** | Track memory, CPU, I/O per Mem0 pod |         |
| 🔲 **S3 + MinIO Data Lake**        | Long-term metrics archival          |         |
| 🔲 **Airflow / Prefect**           | Scheduled metric aggregation jobs   |         |
| 🔲 **OpenTelemetry**               | Unified tracing for RAG workflows   |         |



## 🤝 Contributing

Pull requests are welcome!
If you’d like to add new metrics, providers, or integrations, open an issue or start a discussion.

```bash
git checkout -b feature/add-groq-support
git commit -am "Add Groq inference metrics"
git push origin feature/add-groq-support
```

## 📜 License

Released under the **MIT License**.
See [`LICENSE`](./LICENSE) for details.


