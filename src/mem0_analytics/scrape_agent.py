import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mem0 import Memory
from analytics import patch_memory
import threading, time, datetime

load_dotenv()

os.environ["METRICS_TARGET"] = "agent"

def start_heartbeat(agent_name: str, interval_sec: int = 300):
    """Background thread to emit heartbeat logs every few minutes."""
    def beat():
        while True:
            print(f"[heartbeat] {agent_name} alive @ {datetime.now():%H:%M:%S}")
            time.sleep(interval_sec)
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()


SCRAPE_SITES = [
    "https://news.ycombinator.com/",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
]

LLM_PROVIDER = os.getenv("SCRAPE_LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("SCRAPE_LLM_MODEL", "gpt-4o-mini")
EMB_PROVIDER = os.getenv("SCRAPE_EMB_PROVIDER", "openai")
EMB_MODEL = os.getenv("SCRAPE_EMB_MODEL", "text-embedding-3-small")
VS_COLLECTION = os.getenv("SCRAPE_VS_COLLECTION", "mem0_scrape_agent")

USER_ID = "agent_scraper_1"
AGENT_ID = "scrape_notes_agent"
os.environ["METRICS_TARGET"] = "agent"


def build_mem0():
    cfg = {
        "llm": {"provider": LLM_PROVIDER, "config": {"model": LLM_MODEL, "temperature": 0.3}},
        "embedder": {"provider": EMB_PROVIDER, "config": {"model": EMB_MODEL}},
        "vector_store": {
            "provider": "qdrant",
            "config": {"host": "localhost", "port": 6333, "collection_name": VS_COLLECTION},
        },
    }
    memory = Memory.from_config(cfg)
    return patch_memory(memory)


def scrape_url(url: str) -> str:
    try:
        print(f"[scraper] fetching {url}")
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join(soup.stripped_strings)[:5000]
    except Exception as e:
        print(f"[scraper] fail {url}: {e}")
        return ""


def summarize(memory: Memory, text: str) -> str:
    try:
        prompt = f"Summarize the following into concise technical notes:\n\n{text[:4000]}"
        resp = memory.llm.generate_response(messages=[
            {"role": "system", "content": "You are a concise summarizer for technical or scientific content."},
            {"role": "user", "content": prompt},
        ])
        if isinstance(resp, dict):
            return resp.get("content") or resp.get("text") or str(resp)
        return str(resp)
    except Exception as e:
        print(f"[llm] summarization failed: {e}")
        return ""


def main():
    start_heartbeat("mem0_scrape_agent")
    start = time.time()
    memory = build_mem0()
    success, fail = 0, 0

    for url in SCRAPE_SITES:
        raw = scrape_url(url)
        if not raw:
            fail += 1
            continue

        notes = summarize(memory, raw)
        if not notes:
            fail += 1
            continue

        payload = [
            {"role": "user", "content": f"source: {url}"},
            {"role": "assistant", "content": notes},
        ]

        try:
            memory.add(payload, user_id=USER_ID, agent_id=AGENT_ID)
            print(f"[store] added notes from {url}")
            success += 1
        except Exception as e:
            print(f"[store] error: {e}")
            fail += 1

        time.sleep(2)

    total_ms = int((time.time() - start) * 1000)
    print(f"✅ scrape_agent complete — stored {success}/{len(SCRAPE_SITES)}, failed {fail}, time {total_ms} ms")
    print("✅ Metrics logged automatically to mem0_metrics_agent.")


if __name__ == "__main__":
    main()
