"""Small Ollama HTTP client. No agent framework or cloud API is required."""
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model="qwen3.5:9b", host="http://127.0.0.1:11434", timeout=180):
        parts = urlparse(host)
        if parts.scheme != "http" or parts.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Use a local Ollama HTTP address, e.g. http://127.0.0.1:11434")
        if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
            raise ValueError("Ollama host must be a plain local origin")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout
        self.http = build_opener(ProxyHandler({}))

    def request(self, path, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        req = Request(self.host + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.http.open(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise OllamaError(f"Could not read Ollama at {self.host}: {exc}. Start Ollama and check the model.") from exc

    def check(self):
        info = self.request("/api/show", {"model": self.model})
        if "tools" not in info.get("capabilities", []):
            raise OllamaError(f"{self.model} does not advertise tool support")
        # Refuse remote/cloud models: the demo promises local inference.
        if info.get("remote_host") or info.get("remote_model"):
            raise OllamaError("Use a locally downloaded model, not an Ollama cloud model")
        return {"model": self.model, "capabilities": info.get("capabilities", [])}

    def chat(self, messages, tools):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 700},
                "keep_alive": "5m"}
        # Qwen3.5/Qwen3 support this. Omit it for alternative models such as Llama 3.2.
        if self.model.startswith("qwen3"):
            body["think"] = False
        if tools:
            body["tools"] = tools
        self.last_usage = {}
        response = self.request("/api/chat", body)
        self.last_usage = {key: value for key in ("prompt_eval_count", "eval_count")
                           if isinstance(value := response.get(key), int) and value >= 0}
        if response.get("done_reason") == "length":
            raise OllamaError("LLM response hit the output limit; use a shorter question or fewer observations")
        message = response.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise OllamaError("Ollama returned no valid assistant message")
        return message
