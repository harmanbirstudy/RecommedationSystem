"""
tests/test_llm_ollama.py

Runs rerank_with_llm against a fake Ollama server (no model needed) to check the
request we send (/api/chat with a JSON-schema `format`) and that the streamed
reply is parsed into LlmRanking.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app import llm
from app.config import get_settings
from app.features import UserProfile
from app.llm import Candidate, LlmRanking, rerank_with_llm

REPLY = {
    "recommendations": [
        {"product_id": "b", "relevance_score": 90, "reason": "Goes well with your books."},
        {"product_id": "a", "relevance_score": 70, "reason": "A favourite in your area."},
    ]
}


class _FakeOllama(BaseHTTPRequestHandler):
    """Answers POST /api/chat the way Ollama streams it: NDJSON chunks, then done."""

    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server naming
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.requests.append({"path": self.path, "body": body})
        content = json.dumps(REPLY)
        chunks = [
            {"model": body["model"], "created_at": "2026-01-01T00:00:00Z",
             "message": {"role": "assistant", "content": content[:40]}, "done": False},
            {"model": body["model"], "created_at": "2026-01-01T00:00:01Z",
             "message": {"role": "assistant", "content": content[40:]}, "done": False},
            {"model": body["model"], "created_at": "2026-01-01T00:00:02Z",
             "message": {"role": "assistant", "content": ""}, "done": True,
             "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20},
        ]
        payload = "".join(json.dumps(c) + "\n" for c in chunks).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        """Keep test output quiet."""


@pytest.fixture
def fake_ollama(monkeypatch):
    """Start the fake server and point the settings at it."""
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    get_settings.cache_clear()
    llm.get_chat_model.cache_clear()
    _FakeOllama.requests.clear()
    yield _FakeOllama.requests
    server.shutdown()
    get_settings.cache_clear()
    llm.get_chat_model.cache_clear()


def _candidate(pid: str) -> Candidate:
    """Minimal candidate."""
    return Candidate(pid, f"Product {pid}", "Books", 10.0, None, 0.5, 0.1, 0.1)


def test_ollama_request_and_structured_reply(fake_ollama):
    """We call /api/chat with the LlmRanking JSON schema and parse the streamed reply."""
    profile = UserProfile("Texas", 3, 120.0, {"Books": 1.0}, ["Clean Code"], set())
    ranking = rerank_with_llm(profile, [_candidate("a"), _candidate("b")], count=2)

    assert isinstance(ranking, LlmRanking)
    assert [p.product_id for p in ranking.recommendations] == ["b", "a"]

    [request] = fake_ollama
    assert request["path"] == "/api/chat"
    body = request["body"]
    assert body["model"] == "qwen2.5-coder:7b"
    # Ollama structured outputs: the JSON schema is sent as `format`
    assert body["format"]["properties"].keys() == {"recommendations"}
    assert body["options"]["temperature"] == 0
    # no PII: the prompt carries the anonymised profile only
    prompt = json.dumps(body["messages"])
    assert "Texas" in prompt and "@" not in prompt
