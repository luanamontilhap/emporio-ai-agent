"""The OpenAI-compatible client against a local fake server speaking the real HTTP protocol."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from fakes import make_policy_rag, make_spy_tools, settings
from emporio_agent.agent import SupportAgent
from emporio_agent.llm_client import LLMError, OpenAICompatibleClient
from emporio_agent.tool_registry import TOOL_DEFINITIONS, ToolDispatcher


def completion(message: dict, finish_reason: str = "stop") -> dict:
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", **message}}]}


def tool_call_message(name: str, arguments: dict) -> dict:
    return completion({"content": None, "tool_calls": [
        {"id": "call_abc", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        "tool_calls")


class FakeOpenAIServer:
    def __init__(self):
        self.requests: list[dict] = []
        self.responder = lambda body: (200, completion({"content": "ok"}))
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                status, payload = outer.responder(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    fake = FakeOpenAIServer()
    yield fake
    fake.close()


def client_for(server, **overrides):
    cfg = settings(LLM_BASE_URL=server.base_url, LLM_MODEL="test-model", LLM_MAX_RETRIES=0, **overrides)
    return OpenAICompatibleClient(cfg)


def test_request_follows_the_chat_completions_protocol_and_tool_calls_are_parsed(server):
    server.responder = lambda body: (200, tool_call_message("get_product_price", {"product_id": 127}))
    client = client_for(server, LLM_API_KEY="sk-test", LLM_TEMPERATURE="0.1")

    reply = client.complete([{"role": "user", "content": "oi"}], TOOL_DEFINITIONS)

    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions" and request["auth"] == "Bearer sk-test"
    assert request["body"]["model"] == "test-model" and request["body"]["tool_choice"] == "auto"
    assert request["body"]["temperature"] == 0.1
    assert [t["function"]["name"] for t in request["body"]["tools"]] == [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert reply.content is None
    assert (reply.tool_calls[0].id, reply.tool_calls[0].name) == ("call_abc", "get_product_price")
    assert json.loads(reply.tool_calls[0].arguments) == {"product_id": 127}


def test_temperature_is_omitted_when_not_configured_and_local_servers_get_a_placeholder_key(server):
    client = client_for(server)

    client.complete([{"role": "user", "content": "oi"}], TOOL_DEFINITIONS)

    assert "temperature" not in server.requests[0]["body"]
    assert server.requests[0]["auth"] == "Bearer not-needed"


def test_reasoning_blocks_are_removed_from_the_answer(server):
    server.responder = lambda body: (200, completion({"content": "<think>vou calcular 99</think>Olá! Posso ajudar?"}))

    reply = client_for(server).complete([{"role": "user", "content": "oi"}], TOOL_DEFINITIONS)

    assert reply.content == "Olá! Posso ajudar?"


def test_server_errors_and_empty_responses_become_llm_errors(server):
    server.responder = lambda body: (500, {"error": {"message": "overloaded"}})
    with pytest.raises(LLMError, match="LLM request failed"):
        client_for(server).complete([{"role": "user", "content": "oi"}], TOOL_DEFINITIONS)

    server.responder = lambda body: (200, {**completion({"content": "x"}), "choices": []})
    with pytest.raises(LLMError):
        client_for(server).complete([{"role": "user", "content": "oi"}], TOOL_DEFINITIONS)


def test_full_agent_round_trip_over_http(server):
    def respond(body):
        last = body["messages"][-1]
        if last["role"] == "tool":
            price = json.loads(last["content"])
            return 200, completion({"content": f"Com PIX, o Kalani sai por {price['final_price_formatted']}."})
        return 200, tool_call_message("get_product_price", {"product_id": 127, "payment_method": "pix"})

    server.responder = respond
    cfg = settings(LLM_BASE_URL=server.base_url, LLM_MODEL="test-model", LLM_MAX_RETRIES=0)
    agent = SupportAgent(OpenAICompatibleClient(cfg),
                         ToolDispatcher(make_spy_tools(), make_policy_rag()), cfg)

    response = agent.ask("Quanto custa o Kalani KAL-700T no PIX?")

    assert response.status == "answered" and "R$ 323,10" in response.text
    second = server.requests[1]["body"]["messages"]
    assert second[0]["role"] == "system" and second[1]["role"] == "user"
    assistant, tool = second[-2], second[-1]
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["function"]["name"] == "get_product_price"
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_abc"
