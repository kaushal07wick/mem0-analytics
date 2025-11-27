import os
import requests
from dotenv import load_dotenv

load_dotenv()

POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")
POSTHOG_PROJECT_ID = os.getenv("POSTHOG_PROJECT_ID")
POSTHOG_URL = os.getenv("POSTHOG_URL", "https://app.posthog.com")

if not POSTHOG_API_KEY or not POSTHOG_PROJECT_ID:
    raise SystemExit("❌ Missing POSTHOG_API_KEY or POSTHOG_PROJECT_ID in environment.")

headers = {
    "Authorization": f"Bearer {POSTHOG_API_KEY}",
    "Content-Type": "application/json",
}

dashboard_name = "Mem0 Performance Monitor"
dashboard_payload = {
    "name": dashboard_name,
    "filters": {},
    "tags": ["mem0", "performance", "analytics"],
    "description": "Monitor Mem0 LLM performance, errors, and resource metrics.",
}

# --- Step 1: Create Dashboard ---
r = requests.post(
    f"{POSTHOG_URL}/api/projects/{POSTHOG_PROJECT_ID}/dashboards/",
    headers=headers,
    json=dashboard_payload
)
if r.status_code != 201:
    raise SystemExit(f"❌ Failed to create dashboard: {r.status_code} {r.text}")

dashboard = r.json()
dashboard_id = dashboard["id"]
print(f"✅ Created dashboard: {dashboard_name} (id={dashboard_id})")

# --- Step 2: Define Insights (Modern 'query' format) ---
insights = [
    {
        "name": "Function Latency Trend",
        "query": {
            "kind": "TrendsQuery",
            "series": [
                {
                    "event": "mem0_function_usage",
                    "math": "avg",
                    "math_property": "latency_ms",
                    "name": "avg latency_ms",
                }
            ],
            "interval": "minute",
            "breakdown": {"property": "function", "type": "event"},
        },
        "display": "LineGraph",
    },
    {
        "name": "Error Rate by Function",
        "query": {
            "kind": "TrendsQuery",
            "series": [
                {
                    "event": "mem0_function_usage",
                    "math": "avg",
                    "math_property": "error_rate",
                    "name": "avg error_rate",
                }
            ],
            "breakdown": {"property": "function", "type": "event"},
        },
        "display": "BarChart",
    },
    {
        "name": "LLM Usage Distribution",
        "query": {
            "kind": "TrendsQuery",
            "series": [
                {
                    "event": "mem0_function_usage",
                    "math": "sum",
                    "math_property": "usage_count",
                    "name": "sum usage_count",
                }
            ],
            "breakdown": {"property": "llm_model", "type": "event"},
        },
        "display": "BarChart",
    },
    {
        "name": "Vector Store Latency",
        "query": {
            "kind": "TrendsQuery",
            "series": [
                {
                    "event": "mem0_function_usage",
                    "math": "avg",
                    "math_property": "vector_latency_ms",
                    "name": "avg vector_latency_ms",
                }
            ],
            "breakdown": {"property": "vectorstore", "type": "event"},
        },
        "display": "BarChart",
    },
    {
        "name": "Tokens vs Latency (by LLM)",
        "query": {
            "kind": "TrendsQuery",
            "series": [
                {
                    "event": "mem0_function_usage",
                    "math": "avg",
                    "math_property": "latency_ms",
                    "name": "avg latency_ms",
                }
            ],
            "breakdown": {"property": "llm_model", "type": "event"},
        },
        "display": "ScatterPlot",
    },
    {
        "name": "CPU & Memory Usage Over Time",
        "query": {
            "kind": "TrendsQuery",
            "series": [
                {
                    "event": "mem0_function_usage",
                    "math": "avg",
                    "math_property": "cpu_percent",
                    "name": "avg cpu_percent",
                },
                {
                    "event": "mem0_function_usage",
                    "math": "avg",
                    "math_property": "mem_mb",
                    "name": "avg mem_mb",
                },
            ],
            "interval": "minute",
        },
        "display": "LineGraph",
    },
]

# --- Step 3: Create Each Insight ---
for insight in insights:
    payload = {
        "name": insight["name"],
        "dashboard": dashboard_id,
        "query": insight["query"],
        "derived_name": insight["name"],
        "filters": {},  # required placeholder for backward compat
        "display": insight.get("display", "LineGraph"),
    }

    r = requests.post(
        f"{POSTHOG_URL}/api/projects/{POSTHOG_PROJECT_ID}/insights/",
        headers=headers,
        json=payload,
    )

    if r.status_code == 201:
        print(f"✅ Added insight: {insight['name']}")
    else:
        print(f"❌ Failed to create insight {insight['name']}: {r.status_code} {r.text}")

# --- Step 4: Print final link ---
print(f"\n🎯 Done. View it at: {POSTHOG_URL}/project/{POSTHOG_PROJECT_ID}/dashboard/{dashboard_id}")
