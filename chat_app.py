import os
import sys
import signal
import argparse
import threading
import time
from datetime import datetime
from dotenv import load_dotenv
from mem0 import Memory
from analytics import patch_memory

load_dotenv()

def start_heartbeat(agent_name: str, interval_sec: int = 300):
    def beat():
        while True:
            print(f"[heartbeat] {agent_name} alive @ {datetime.now():%H:%M:%S}")
            time.sleep(interval_sec)
    threading.Thread(target=beat, daemon=True).start()


def build_config(provider: str, model: str, collection_name: str) -> dict:
    embedder = {"provider": "openai", "config": {"model": "text-embedding-3-small"}}
    if provider == "openai":
        vector_store = {"provider": "qdrant", "config": {"url": "http://localhost:6333", "collection_name": collection_name}}
    else:
        vector_store = {"provider": "chroma", "config": {"path": "./chroma_store", "collection_name": collection_name}}
    return {
        "llm": {"provider": provider, "config": {"model": model, "temperature": 0.2, "max_tokens": 800}},
        "embedder": embedder,
        "vector_store": vector_store,
    }


def banner(args):
    print(f"""
mem0 chat (cli)
-------------------------------------------
  LLM:           {args.provider}:{args.model}
  Vector Store:  {'qdrant' if args.provider == 'openai' else 'chroma'}
-------------------------------------------
  /help       show commands
  /search q   semantic search
  /get        list memories
  /reset      wipe vector store
  /switch     toggle provider/model
  /exit       quit
""")


def handle_help():
    print("""
/help       show commands
/search q   semantic search
/get        list memories
/reset      wipe vector store
/switch     toggle provider/model
/exit       quit
""")


def switch_provider(args):
    if args.provider == "openai":
        args.provider = "ollama"
        args.model = os.getenv("OLLAMA_CHAT_MODEL", "smollm2:135m")
        args.vs_collection = "mem0_cli_chat_ollama"
    else:
        args.provider = "openai"
        args.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano-2025-08-07")
        args.vs_collection = "mem0_cli_chat_openai"
    print(f"Switched to {args.provider}:{args.model}")
    return args


def make_memory(args):
    cfg = build_config(args.provider, args.model, args.vs_collection)
    mem = Memory.from_config(cfg)
    return patch_memory(mem)


def parse_args():
    p = argparse.ArgumentParser(description="mem0 CLI chat with analytics tracking")
    p.add_argument("--user-id", default="cli_user", help="session user ID")
    p.add_argument("--provider", default=os.getenv("CHAT_PROVIDER", "openai"), choices=["openai", "ollama"])
    p.add_argument("--model", default=os.getenv("CHAT_MODEL", "gpt-5-nano-2025-08-07"))
    p.add_argument("--vs-collection", default=os.getenv("CHAT_VS_COLLECTION", "mem0_cli_chat"))
    return p.parse_args()


def main():
    os.environ["METRICS_TARGET"] = "chat"
    start_heartbeat("mem0_cli_chat")
    args = parse_args()
    banner(args)
    mem = make_memory(args)
    user_id = args.user_id
    history = []
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    system_msg = {"role": "system", "content": "Be concise, factual, and context-aware."}

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue

        if user_text == "/help":
            handle_help()
            continue

        if user_text.startswith("/search "):
            q = user_text[len("/search "):].strip()
            try:
                res = mem.search(q, user_id=user_id)
                for i, r in enumerate(res.get("results", [])[:5], 1):
                    print(f"{i}. {r.get('memory', '')[:160]}")
            except Exception as e:
                print(f"search error: {e}")
            continue

        if user_text == "/get":
            try:
                res = mem.get_all(user_id=user_id)
                items = res.get("results", [])
                print(f"{len(items)} memories found.")
                for i, r in enumerate(items[:10], 1):
                    print(f"{i}. {r.get('memory', '')[:160]}")
            except Exception as e:
                print(f"get error: {e}")
            continue

        if user_text == "/reset":
            confirm = input("type YES to wipe: ").strip()
            if confirm == "YES":
                try:
                    mem.reset()
                    history.clear()
                    print("reset complete.")
                except Exception as e:
                    print(f"reset error: {e}")
            else:
                print("reset aborted.")
            continue

        if user_text == "/switch":
            args = switch_provider(args)
            banner(args)
            mem = make_memory(args)
            history.clear()
            continue

        if user_text == "/exit":
            print("exiting.")
            break

        msg_user = {"role": "user", "content": user_text}
        convo = [system_msg] + history + [msg_user]

        try:
            mem.add([msg_user], user_id=user_id, metadata={"app": "cli_chat"})
        except Exception as e:
            print(f"add error: {e}")

        try:
            _ = mem.search("What are this user's preferences?", user_id=user_id)
        except Exception as e:
            print(f"search error: {e}")

        try:
            reply = mem.llm.generate_response(messages=convo)
        except Exception as e:
            reply = f"[llm error] {e}"

        history.extend([msg_user, {"role": "assistant", "content": reply}])
        print(f"{datetime.now():%H:%M:%S} bot> {reply}\n")


if __name__ == "__main__":
    main()
