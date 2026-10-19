"""Tool registry tuned for voice: parallel execution, hard deadlines, interruption cancels in-flight tools."""
import asyncio, os, shutil, subprocess, sys, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Awaitable[Any]]
    deadline: float = 15.0


class ToolRegistry:
    def __init__(self, default_deadline=15.0):
        self.tools, self.default_deadline = {}, default_deadline

    def register(self, name, description, properties=None, required=None, deadline=None):
        def deco(fn):
            self.tools[name] = Tool(name, description, {"type": "object", "properties": properties or {},
                                    "required": required or []}, fn, deadline or self.default_deadline)
            return fn
        return deco

    def specs(self, exclude=()):
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.parameters}}
                for t in self.tools.values() if t.name not in exclude]

    async def run_calls(self, calls, cancel: asyncio.Event, emit=lambda *a, **k: None):
        async def one(c):
            t = self.tools.get(c.name)
            if not t:
                return c, f"error: unknown tool {c.name}"
            emit("tool_start", name=c.name, args=c.args)
            t0 = time.time()
            try:
                out = str(await asyncio.wait_for(t.fn(**c.args), t.deadline))
            except asyncio.TimeoutError:
                out = f"error: {c.name} timed out after {t.deadline:.0f}s"
            except TypeError as e:
                out = f"error: bad arguments: {e}"
            except Exception as e:
                out = f"error: {e}"
            emit("tool_end", name=c.name, ms=int((time.time() - t0) * 1000), result=out[:200])
            return c, out[:6000]

        tasks = [asyncio.create_task(one(c)) for c in calls]
        gather = asyncio.gather(*tasks, return_exceptions=True)
        waiter = asyncio.create_task(cancel.wait())
        await asyncio.wait({gather, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if cancel.is_set():
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            waiter.cancel()
            raise asyncio.CancelledError
        waiter.cancel()
        return [r for r in gather.result() if not isinstance(r, BaseException)]


def _safe(ws: Path, p: str) -> Path:
    full = (ws / p).resolve()
    if ws.resolve() not in full.parents and full != ws.resolve():
        raise ValueError("path escapes workspace")
    return full


async def _sh(*cmd, timeout=10):
    p = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
    except asyncio.TimeoutError:
        p.kill()
        await p.wait()  # Reap the timed-out child before returning control.
        raise
    return out.decode(errors="replace").strip()


def builtin_tools(reg: ToolRegistry, workspace="workspace"):
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    @reg.register("get_time", "Current local date and time.")
    async def get_time():
        return datetime.now().strftime("%A %B %d %Y, %I:%M %p")

    @reg.register("web_search", "Search the web; returns top results with snippets.",
                  {"query": {"type": "string"}}, ["query"])
    async def web_search(query: str):
        def go():
            from ddgs import DDGS
            return [f"{r['title']}: {r['body']}" for r in DDGS().text(query, max_results=4)]
        return "\n".join(await asyncio.to_thread(go))

    @reg.register("run_python", "Run a short Python snippet and return stdout/stderr.",
                  {"code": {"type": "string"}}, ["code"], deadline=25)
    async def run_python(code: str):
        return await _sh(sys.executable, "-c", code, timeout=20) or "(no output)"

    @reg.register("read_file", "Read a text file from the workspace.", {"path": {"type": "string"}}, ["path"])
    async def read_file(path: str):
        return _safe(ws, path).read_text()[:5000]

    @reg.register("write_file", "Write a text file in the workspace.",
                  {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])
    async def write_file(path: str, content: str):
        p = _safe(ws, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    @reg.register("list_files", "List files in the workspace.", {"path": {"type": "string"}})
    async def list_files(path: str = "."):
        return "\n".join(sorted(x.name for x in _safe(ws, path).iterdir())) or "(empty)"

    @reg.register("calendar_today", "List today's calendar events (macOS; needs icalBuddy).")
    async def calendar_today():
        if shutil.which("icalBuddy"):
            return await _sh("icalBuddy", "-n", "eventsToday") or "No events today."
        return "error: install icalBuddy (brew install ical-buddy) for calendar access"

    @reg.register("set_volume", "Set macOS output volume 0-100.", {"level": {"type": "integer"}}, ["level"])
    async def set_volume(level: int):
        await _sh("osascript", "-e", f"set volume output volume {max(0, min(100, int(level)))}")
        return f"volume {level}"

    @reg.register("open_app", "Open a macOS application by name.", {"name": {"type": "string"}}, ["name"])
    async def open_app(name: str):
        return await _sh("open", "-a", name) or f"opened {name}"

    return reg
