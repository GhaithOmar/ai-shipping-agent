# tests/unit/test_memory_buffer.py
import os

os.environ.setdefault("AGENT_OFFLINE", "1")

# Try canonical module first; fallback to flat memory.py if needed
try:
    from backend.agent.memory import ShortMemory
except Exception:
    from memory import ShortMemory  # type: ignore


def test_short_memory_window_and_order():
    mem = ShortMemory(max_turns=3)
    mem.add("user", "hi")
    mem.add("assistant", "hello")
    mem.add("user", "track 123")
    # buffer not full yet
    assert mem.as_lines() == ["user: hi", "assistant: hello", "user: track 123"]

    mem.add("assistant", "please share carrier")
    # now only last 3 are kept
    lines = mem.as_lines()
    assert lines == ["assistant: hello", "user: track 123", "assistant: please share carrier"]

    mem.clear()
    assert mem.as_lines() == []
