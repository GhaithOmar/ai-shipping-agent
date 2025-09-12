import os
os.environ["AGENT_OFFLINE"] = "1"

from fastapi.testclient import TestClient
from backend.main import app

def test_stream_chat():
    client = TestClient(app)
    with client.stream(
        "POST", "/chat?agent=0&stream=1",
        json={"message": "Track order 12345678 with Shipping_B", "top_k": 1},
    ) as r:
        assert r.status_code == 200
        # consume a few lines of the SSE stream without asserting content
        chunks = []
        for i, line in enumerate(r.iter_lines()):
            if line:
                chunks.append(line)
            if i > 10:
                break
        assert chunks  # we got something from the stream
