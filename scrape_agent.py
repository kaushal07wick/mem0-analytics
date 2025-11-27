import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mem0 import Memory
from analytics import patch_memory

load_dotenv()

# --- CONFIG ---
SCRAPE_SITES = [
    "https://news.ycombinator.com/",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
]

LLM_PROVIDER = os.getenv("SCRAPE_LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("SCRAPE_LLM_MODEL", "gpt-4o-mini")
EMB_PROVIDER = os.getenv("SCRAPE_EMB_PROVIDER", "openai")
EMB_MODEL = os.getenv("SCRAPE_EMB_MODEL", "text-embedding-3-small")
VS_PROVIDER = os.getenv("SCRAPE_VS_PROVIDER", "pgvector")
VS_COLLECTION = os.getenv("SCRAPE_VS_COLLECTION", "mem0_scrape_agent")

USER_ID = "agent_scraper_1"
AGENT_ID = "scrape_notes_agent"

# --- HELPER: build mem0 instance ---
def build_mem0():
    config = {
        "llm": {
            "provider": LLM_PROVIDER,
            "config": {"model": LLM_MODEL, "temperature": 0.3},
        },
        "embedder": {
            "provider": EMB_PROVIDER,
            "config": {"model": EMB_MODEL},
        },
        "vector_store": {
            "provider": VS_PROVIDER,
            "config": {"collection_name": VS_COLLECTION},
        },
    }
    memory = Memory.from_config(config)
    return patch_memory(memory)

# --- HELPER: simple scraper ---
def scrape_url(url: str) -> str:
    try:
        print(f"[scraper] fetching {url}")
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # grab visible text, compress whitespace
        text = " ".join(soup.stripped_strings)
        return text[:5000]  # cap at 5k chars to avoid token flood
    except Exception as e:
        print(f"[scraper] fail {url}: {e}")
        return ""

# --- HELPER: summarize content ---
def summarize(memory: Memory, text: str) -> str:
    try:
        prompt = f"Summarize the following content into concise study notes with key ideas:\n\n{text[:4000]}"
        response = memory.llm.generate_response(messages=[
            {"role": "system", "content": "You are a concise summarizer for technical articles."},
            {"role": "user", "content": prompt},
        ])
        if isinstance(response, dict):
            return response.get("content") or response.get("text") or str(response)
        return str(response)
    except Exception as e:
        print(f"[llm] summarization failed: {e}")
        return ""

# --- MAIN ---
def main():
    memory = build_mem0()

    for url in SCRAPE_SITES:
        raw_text = scrape_url(url)
        if not raw_text:
            continue

        notes = summarize(memory, raw_text)
        if not notes:
            continue

        payload = [
            {"role": "user", "content": f"source: {url}"},
            {"role": "assistant", "content": notes},
        ]

        try:
            memory.add(payload, user_id=USER_ID, agent_id=AGENT_ID)
            print(f"[store] added notes from {url}")
        except Exception as e:
            print(f"[store] error storing memory: {e}")

        time.sleep(2)

    print("✅ scrape_agent complete — data stored in vector DB.")


if __name__ == "__main__":
    main()
