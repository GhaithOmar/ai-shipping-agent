from collections import deque
from typing import Deque, Dict, List

class ShortMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._buf: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self._buf.append({"role": role, "content": content})

    def as_lines(self) -> List[str]:
        return [f"{m['role']}: {m['content']}" for m in self._buf]

    def clear(self):
        self._buf.clear()
