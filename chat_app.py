import os
import sys
import signal
import argparse
from datetime import datetime
from dotenv import load_dotenv

from mem0 import Memory
from analytics import patch_memory

load_dotenv()

# --------------------------------------------------------------------
# CONFIG BUILDER
# --------------------------------------------------------------------
def build_config(provider: str, model: str, collection_name: str) -> dict:
    """Construct Memory configuration dynamically."""

    embedder = {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    }

    if provider == "openai":
        # QDRANT: for cloud / OpenAI setup
        vector_store = {
            "provider": "qdrant",
            "config": {
                "url": "http://localhost:6333",  # default Qdrant endpoint
                "collection_name": collection_name,
            },
        }
    else:
        # CHROMA: for local Ollama pipeline
        vector_store = {
            "provider": "chroma",
            "config": {
                "path": "./chroma_store",        # on-disk persistent
                "collection_name": collection_name,
            },
        }

    return {
        "llm": {
            "provider": provider,
            "config": {
                "model": model,
                "temperature": 0.2,
                "max_tokens": 800,
            },
        },
        "embedder": embedder,
        "vector_store": vector_store,
    }


# --------------------------------------------------------------------
# HELPERS / UI
# --------------------------------------------------------------------
def banner(args):
    print(f"""
=======================================================================
                         mem0 chat (cli)
=======================================================================
  llm:          {args.provider}:{args.model}
  vector_store: {"qdrant" if args.provider == "openai" else "chroma"}
-----------------------------------------------------------------------
  /help       show commands
  /search q   semantic search
  /get        list all memories
  /reset      wipe vector store
  /switch     toggle openai/ollama
  /exit       quit
=======================================================================
""")


def handle_help():
    print("""
/help       show commands
/search q   semantic search
/get        list all memories
/reset      wipe vector store
/switch     toggle provider/model
/exit       quit
""")


def switch_provider(args):
    """Switch between OpenAI+Qdrant and Ollama+Chroma dynamically."""
    if args.provider == "openai":
        args.provider = "ollama"
        args.model = os.getenv("OLLAMA_CHAT_MODEL", "smollm2:135m")
        args.vs_collection = "mem0_cli_chat_ollama"
    else:
        args.provider = "openai"
        args.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano-2025-08-07")
        args.vs_collection = "mem0_cli_chat_openai"

    print(f"switched to {args.provider}:{args.model}")
    return args



# --------------------------------------------------------------------
# MEMORY INITIALIZER
# --------------------------------------------------------------------
def make_memory(args):
    cfg = build_config(args.provider, args.model, args.vs_collection)
    mem = Memory.from_config(cfg)
    return patch_memory(mem)


# --------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="mem0 cli chat with analytics tracking")
    p.add_argument("--user-id", default="cli_user", help="session user_id")
    p.add_argument("--provider", default=os.getenv("CHAT_PROVIDER", "openai"),
                   choices=["openai", "ollama"])
    p.add_argument("--model", default=os.getenv("CHAT_MODEL", "gpt-5-nano-2025-08-07"))
    p.add_argument("--vs-collection", default=os.getenv("CHAT_VS_COLLECTION", "mem0_cli_chat"))
    return p.parse_args()


def main():
    args = parse_args()
    banner(args)
    mem = make_memory(args)
    user_id = args.user_id
    history = []

    def sigint(_sig, _frm):
        print("\nexiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint)
    system_msg = {"role": "system", "content": "Be concise, factual, and context-aware."}

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue

        # ------------------ Commands ------------------
        if user_text == "/help":
            handle_help()
            continue

        if user_text.startswith("/search "):
            q = user_text[len("/search "):].strip()
            try:
                res = mem.search(q, user_id=user_id)
                print("search results:")
                for i, r in enumerate(res.get("results", [])[:5], 1):
                    print(f"{i}. {r.get('memory', '')[:160]}")
            except Exception as e:
                print(f"search! {e}")
            continue

        if user_text == "/get":
            try:
                res = mem.get_all(user_id=user_id)
                items = res.get("results", [])
                print(f"{len(items)} memories found.")
                for i, r in enumerate(items[:10], 1):
                    print(f"{i}. {r.get('memory', '')[:160]}")
            except Exception as e:
                print(f"get! {e}")
            continue

        if user_text == "/reset":
            try:
                confirm = input("type YES to wipe: ").strip()
                if confirm == "YES":
                    mem.reset()
                    history.clear()
                    print("reset done.")
                else:
                    print("reset aborted.")
            except Exception as e:
                print(f"reset! {e}")
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

        # ------------------ Chat Flow ------------------
        msg_user = {"role": "user", "content": user_text}
        convo = [system_msg] + history + [msg_user]

        try:
            mem.add([msg_user], user_id=user_id, metadata={"app": "cli_chat"})
        except Exception as e:
            print(f"add! {e}")

        try:
            _ = mem.search("What are this user's preferences?", user_id=user_id)
        except Exception as e:
            print(f"search! {e}")

        try:
            reply = mem.llm.generate_response(messages=convo)
        except Exception as e:
            reply = f"[llm error] {e}"

        history.append(msg_user)
        history.append({"role": "assistant", "content": reply})
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"{ts} bot> {reply}\n")


# --------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
