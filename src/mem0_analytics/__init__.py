import importlib
import threading
import os

def _autopatch():
    try:
        mem0 = importlib.import_module("mem0")
        from mem0_analytics.analytics import patch_memory
        if hasattr(mem0, "Memory"):
            patch_memory(mem0.Memory)
            print("[mem0-analytics] auto-tracking enabled ✅")
    except Exception as e:
        print(f"[mem0-analytics] autopatch failed: {e}")

threading.Thread(target=_autopatch, daemon=True).start()
