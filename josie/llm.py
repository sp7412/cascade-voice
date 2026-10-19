import json
from dataclasses import dataclass, field
import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMEvent:
    kind: str  # text | tool_calls | error
    text: str = ""
    calls: list = field(default_factory=list)


class ThinkStripper:
    """Removes <think>...</think> across arbitrary chunk boundaries so reasoning never reaches TTS."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, s):
        self.buf += s
        out = ""
        while True:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i < 0:
                    self.buf = self.buf[-(len(self.CLOSE) - 1):]
                    return out
                self.buf = self.buf[i + len(self.CLOSE):].lstrip()
                self.inside = False
            else:
                i = self.buf.find(self.OPEN)
                if i >= 0:
                    out += self.buf[:i]
                    self.buf = self.buf[i + len(self.OPEN):]
                    self.inside = True
                    continue
                keep = 0
                for k in range(min(len(self.OPEN) - 1, len(self.buf)), 0, -1):
                    if self.OPEN.startswith(self.buf[-k:]):
                        keep = k
                        break
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                return out

    def flush(self):
        out = "" if self.inside else self.buf
        self.buf = ""
        return out


class OllamaLLM:
    """Streams from Ollama's OpenAI-compatible endpoint (local, or cloud with OLLAMA_API_KEY)."""

    def __init__(self, cfg):
        self.cfg = cfg
        h = {"Authorization": f"Bearer {cfg.llm_key}"} if cfg.llm_key else {}
        self.client = httpx.AsyncClient(base_url=cfg.llm_base, headers=h, timeout=httpx.Timeout(60, connect=5))

    async def stream(self, messages, tools=None):
        payload = {"model": self.cfg.llm_model, "messages": messages, "stream": True,
                   "temperature": self.cfg.temperature}
        if tools:
            payload["tools"] = tools
        acc = {}
        async with self.client.stream("POST", "/v1/chat/completions", json=payload) as r:
            if r.status_code >= 400:
                yield LLMEvent("error", text=f"LLM HTTP {r.status_code}: {(await r.aread())[:200]!r}")
                return
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                choices = json.loads(d).get("choices") or [{}]
                delta = choices[0].get("delta", {})
                if delta.get("content"):  # reasoning fields are intentionally ignored
                    yield LLMEvent("text", text=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    a = acc.setdefault(tc.get("index", 0), {"id": tc.get("id") or "", "name": "", "args": ""})
                    f = tc.get("function", {})
                    if f.get("name"):
                        a["name"] = f["name"]
                    a["args"] += f.get("arguments") or ""
        if acc:
            calls = []
            for i, a in sorted(acc.items()):
                try:
                    args = json.loads(a["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(a["id"] or f"call_{i}", a["name"], args))
            yield LLMEvent("tool_calls", calls=calls)
