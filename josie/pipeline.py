"""Orchestrator: mic -> AEC/echo gate -> VAD -> Smart Turn (+speculation) -> ASR -> LLM/tools
-> sentence chunker -> TTS -> player, with barge-in and played-only history."""
import asyncio, difflib, re, time
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from .asr import make_asr
from .audio_io import AEC, Mic
from .chunker import SentenceChunker
from .config import Config
from .events import EventBus
from .executor import SerialExecutor
from .llm import LLMEvent, OllamaLLM, ThinkStripper
from .playback import Player
from .speaker import AssistantVoice, SpeakerEmbedder
from .tools import ToolRegistry, builtin_tools
from .tts import RTFTracker, make_tts
from .turn import Decision, SmartTurn, SmartTurnModel
from .vad import load_vad

FRAME_MS = 32


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def same_utterance(a, b):
    na, nb = norm(a), norm(b)
    return bool(na) and (na == nb or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.92)


def trim_history(history, max_chars):
    """Drop oldest turns first so 100+ tool-call sessions never overflow the context."""
    total = sum(len(m["content"]) for m in history)
    while history and total > max_chars:
        total -= len(history.pop(0)["content"])
    while history and history[0]["role"] != "user":
        history.pop(0)


class Speculation:
    """LLM run started on a partial transcript. Output is buffered in a queue; the response task
    consumes it if the final transcript matches. Only round 1 is speculative and tools are NOT
    executed here (no side effects from guesses)."""

    def __init__(self, text, agen):
        self.text, self.q = text, asyncio.Queue()
        self.task = asyncio.create_task(self._pump(agen))

    async def _pump(self, agen):
        try:
            async for ev in agen:
                await self.q.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.q.put(LLMEvent("error", text=str(e)))
        finally:
            await self.q.put(None)

    async def events(self):
        while (ev := await self.q.get()) is not None:
            yield ev

    def cancel(self):
        self.task.cancel()


@dataclass
class Utt:
    frames: list
    silence: int = 0
    voiced_ms: int = 0
    partial: str = ""
    last_partial: float = 0.0
    partial_task: asyncio.Task | None = None
    p_task: asyncio.Task | None = None
    p_complete: float | None = None


@dataclass
class TurnState:
    user: str
    msgs: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)


class Voice:
    def __init__(self, cfg: Config, bus: EventBus, audio=True):
        self.cfg, self.bus = cfg, bus
        self.exec = SerialExecutor()
        self.vad, self.turn, self.aec = load_vad(), SmartTurn(), AEC()
        self.turn_model = SmartTurnModel(cfg.smart_turn_onnx) if cfg.smart_turn_onnx else None
        self.asr, self.tts = make_asr(cfg), make_tts(cfg)
        self.llm, self.rtf = OllamaLLM(cfg), RTFTracker()
        self.player = Player(self.tts.sample_rate)
        self.voice_id = AssistantVoice(SpeakerEmbedder(cfg.speaker_onnx), cfg.speaker_threshold)
        self.tools = builtin_tools(ToolRegistry(cfg.tool_deadline_s), cfg.workspace)
        self._register_agents()
        self.history, self.spec, self.resp_task, self.turn_state = [], None, None, None
        self._cancel = asyncio.Event()
        self.bg_results, self.bg_n = asyncio.Queue(), 0
        self.in_utt, self.idle_since = False, time.monotonic()
        self._bi = {"frames": [], "voiced": 0, "gap": 0}
        self._asides = 0
        self.audio = audio

    # ------------------------------------------------------------------ main loop
    async def run(self):
        loop = asyncio.get_running_loop()
        self.player.bind_loop(loop)
        q = asyncio.Queue(maxsize=600)
        if self.audio:
            self.player.open()
            mic = Mic(self.cfg, loop, q)
            mic.start()
        bg = asyncio.create_task(self._bg_announcer())
        pre, utt = deque(maxlen=10), None
        self.bus.emit("ready")
        try:
            while True:
                frame = await q.get()
                frame = self.aec.process(frame, self.player.reference(len(frame)))
                speaking = self.player.active or bool(self.resp_task and not self.resp_task.done())
                thr = self.cfg.vad_threshold + (self.cfg.echo_gate_boost if speaking else 0)
                voiced = self.vad(frame) >= thr
                if speaking and utt is None:
                    heard = await self._bargein(frame, voiced)
                    if heard is None:
                        continue
                    await self.interrupt()
                    utt = Utt(frames=[heard[i:i + self.cfg.frame] for i in range(0, len(heard), self.cfg.frame)],
                              voiced_ms=self.cfg.bargein_ms)
                    self.in_utt = True
                    continue
                utt = await self._step(utt, frame, voiced, pre)
        finally:
            bg.cancel()

    async def _step(self, utt, frame, voiced, pre):
        if utt is None:
            pre.append(frame)
            if voiced:
                utt = Utt(frames=list(pre))
                pre.clear()
                self.in_utt = True
                self.bus.emit("speech_start")
            return utt
        utt.frames.append(frame)
        if voiced:
            utt.silence, utt.voiced_ms = 0, utt.voiced_ms + FRAME_MS
            utt.p_complete = None
            if self.spec:  # user kept talking: speculation is stale
                self.spec.cancel(); self.spec = None
                self.bus.emit("speculation_cancelled", reason="speech_resumed")
        else:
            utt.silence += FRAME_MS
        now = time.monotonic()
        if now - utt.last_partial >= self.cfg.partial_interval_ms / 1000 and not (
                utt.partial_task and not utt.partial_task.done()):
            utt.last_partial = now
            utt.partial_task = asyncio.create_task(self._partial(utt))
        if self.turn_model and utt.silence >= 192 and utt.p_complete is None and not (utt.p_task and not utt.p_task.done()):
            utt.p_task = asyncio.create_task(self._p_complete(utt))
        d = self.turn.decide(utt.silence, utt.partial, utt.voiced_ms, utt.p_complete)
        if d == Decision.DISCARD:
            self.in_utt = False
            return None
        if d == Decision.SPECULATE and not self.spec and utt.partial:
            self.spec = Speculation(utt.partial, self.llm.stream(self._msgs(utt.partial), self.tools.specs()))
            self.bus.emit("speculate", text=utt.partial)
        elif d == Decision.END:
            self.in_utt = False
            await self._end_turn(utt)
            return None
        return utt

    async def _partial(self, utt):
        audio = np.concatenate(utt.frames)
        text = await self.exec.run(self.asr.transcribe, audio)
        utt.partial = text
        self.bus.emit("partial", text=text)

    async def _p_complete(self, utt):
        audio = np.concatenate(utt.frames)
        utt.p_complete = await asyncio.get_running_loop().run_in_executor(None, self.turn_model.p_complete, audio)
        self.bus.emit("smart_turn", p_complete=round(utt.p_complete, 2))

    async def _end_turn(self, utt):
        audio = np.concatenate(utt.frames)
        text = await self.exec.run(self.asr.transcribe, audio)
        self.bus.emit("final", text=text)
        if not norm(text):
            if self.spec:
                self.spec.cancel(); self.spec = None
            return
        spec, self.spec = self.spec, None
        if spec and same_utterance(spec.text, text):
            self.bus.emit("speculation_hit", partial=spec.text, final=text)
        else:
            if spec:
                spec.cancel()
                self.bus.emit("speculation_miss", partial=spec.text, final=text)
            spec = None
        self.resp_task = asyncio.create_task(self._respond(text, spec))

    # ------------------------------------------------------------------ barge-in
    async def _bargein(self, frame, voiced):
        b = self._bi
        if voiced:
            b["frames"].append(frame); b["voiced"] += 1; b["gap"] = 0
        elif b["frames"]:
            b["frames"].append(frame); b["gap"] += 1
            if b["gap"] > 5:
                b.update(frames=[], voiced=0, gap=0)
            return None
        if b["voiced"] * FRAME_MS < self.cfg.bargein_ms:
            return None
        audio = np.concatenate(b["frames"])
        b.update(frames=[], voiced=0, gap=0)
        ref = self.player.reference(len(audio))
        mine = await asyncio.get_running_loop().run_in_executor(None, self.voice_id.is_assistant, audio, ref)
        self.bus.emit("bargein_check", is_assistant=mine)
        return None if mine else audio

    async def interrupt(self):
        self.bus.emit("interrupt")
        self.player.fade_out(self.cfg.fade_ms)
        self._cancel.set()
        if self.spec:
            self.spec.cancel(); self.spec = None
        await asyncio.sleep(self.cfg.fade_ms / 1000 + 0.02)
        t = self.resp_task
        if t and not t.done():
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        self._commit(interrupted=True)  # history = exactly what the user heard
        self.player.flush()

    # ------------------------------------------------------------------ response
    def _msgs(self, user_text, extra=()):
        return [{"role": "system", "content": self.cfg.persona.system}, *self.history,
                {"role": "user", "content": user_text}, *extra]

    async def _respond(self, user_text, spec=None):
        self._cancel = cancel = asyncio.Event()
        self.player.flush()
        self.turn_state = ts = TurnState(user_text)
        self._asides = 0
        tts_q = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_worker(tts_q))
        chunker, strip, filler_said = SentenceChunker(), ThinkStripper(), False
        rounds = self.cfg.max_tool_rounds
        try:
            for rnd in range(rounds + 1):
                tools = self.tools.specs() if rnd < rounds else None
                events = spec.events() if (rnd == 0 and spec) else self.llm.stream(
                    self._msgs(user_text, ts.msgs), tools)
                calls, text_round = None, ""
                async for ev in events:
                    if ev.kind == "text":
                        t = strip.feed(ev.text)
                        text_round += t
                        for s in chunker.feed(t):
                            await tts_q.put(s)
                    elif ev.kind == "tool_calls":
                        calls = ev.calls
                    elif ev.kind == "error":
                        self.bus.emit("error", where="llm", msg=ev.text)
                        await tts_q.put("Sorry, I hit a problem reaching my brain.")
                for s in chunker.flush(strip.flush()):
                    await tts_q.put(s)
                if not calls:
                    break
                if not text_round.strip() and not filler_said:  # filler BEFORE the first step
                    await tts_q.put(self.cfg.persona.filler())
                    filler_said = True
                if self.cfg.tool_waits_for_preamble and (text_round.strip() or filler_said):
                    await self.player.wait_started(1.5)  # speak first, then start the tool
                ts.msgs.append({"role": "assistant", "content": text_round, "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": __import__("json").dumps(c.args)}} for c in calls]})
                ticker = asyncio.create_task(self._progress(tts_q))
                try:
                    results = await self.tools.run_calls(calls, cancel, self.bus.emit)
                finally:
                    ticker.cancel()
                for c, out in results:
                    ts.tools_used.append(f"{c.name}({', '.join(map(str, c.args.values()))[:40]})")
                    ts.msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            await tts_q.put(None)
            await tts_task
            await self.player.wait_idle()
            self._commit(interrupted=False)
        except asyncio.CancelledError:
            tts_task.cancel()
            raise

    async def _progress(self, tts_q):
        """Short bounded asides while tools run: every ~12 s, max 3 per turn."""
        while self._asides < self.cfg.max_asides:
            await asyncio.sleep(self.cfg.progress_every_s)
            self._asides += 1
            await tts_q.put(self.cfg.persona.aside())

    async def _tts_worker(self, q):
        first = True
        while (s := await q.get()) is not None:
            t0 = time.perf_counter()
            audio = await self.exec.run(self.tts.synth, s)
            wall = time.perf_counter() - t0
            dur = len(audio) / self.tts.sample_rate
            self.rtf.update(dur, wall)
            self.bus.emit("tts", text=s, audio_s=round(dur, 2), wall_s=round(wall, 2),
                          speed=round(self.rtf.speed or 0, 2))
            asyncio.get_running_loop().run_in_executor(None, self.voice_id.observe, audio, self.tts.sample_rate)
            if first:
                pending = sum(len(x) for x in list(q._queue) if x) / 15.0  # ~15 chars/s of speech
                self.player.arm(self.rtf.preroll(dur + pending))
                first = False
            self.player.enqueue(s, audio)
            self.bus.emit("assistant_sentence", text=s)

    def _commit(self, interrupted):
        ts, self.turn_state = self.turn_state, None
        if not ts:
            return
        spoken = self.player.spoken_text()
        if interrupted:
            spoken = (spoken + " [interrupted]").strip()
        if ts.tools_used:
            spoken += f"\n(tools used: {'; '.join(ts.tools_used[-8:])})"
        self.history += [{"role": "user", "content": ts.user}, {"role": "assistant", "content": spoken or "(silent)"}]
        trim_history(self.history, self.cfg.max_history_chars)
        self.bus.emit("turn_committed", user=ts.user, assistant=spoken, interrupted=interrupted)

    # ------------------------------------------------------------------ background sub-agents
    def _register_agents(self):
        @self.tools.register("spawn_agent", "Start a background sub-agent for a long task. Returns immediately; "
                             "the result is announced when ready.", {"task": {"type": "string"}}, ["task"])
        async def spawn_agent(task: str):
            self.bg_n += 1
            n = self.bg_n

            async def work():
                try:
                    res = await asyncio.wait_for(self._mini_agent(task), 180)
                except Exception as e:
                    res = f"failed: {e}"
                await self.bg_results.put((n, task, res))
            asyncio.create_task(work())
            self.bus.emit("bg_started", n=n, task=task)
            return f"background agent {n} started"

    async def _mini_agent(self, task):
        msgs = [{"role": "system", "content": "Complete the task using tools. Reply with a 2 sentence result."},
                {"role": "user", "content": task}]
        for _ in range(8):
            text, calls = "", None
            async for ev in self.llm.stream(msgs, self.tools.specs(exclude=("spawn_agent",))):
                if ev.kind == "text":
                    text += ev.text
                elif ev.kind == "tool_calls":
                    calls = ev.calls
            if not calls:
                return ThinkStripper().feed(text) or text
            msgs.append({"role": "assistant", "content": text, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": __import__("json").dumps(c.args)}}
                for c in calls]})
            for c, out in await self.tools.run_calls(calls, asyncio.Event(), self.bus.emit):
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
        return "ran out of steps"

    async def _bg_announcer(self):
        """Announce background results only after a real quiet period (default 1.2 s)."""
        while True:
            await asyncio.sleep(0.1)
            busy = self.in_utt or self.player.active or (self.resp_task and not self.resp_task.done())
            if busy:
                self.idle_since = time.monotonic()
                continue
            if time.monotonic() - self.idle_since >= self.cfg.bg_announce_quiet_s and not self.bg_results.empty():
                n, task, res = self.bg_results.get_nowait()
                self.bus.emit("bg_done", n=n)
                note = f"[background agent {n} finished task '{task}': {res}] Tell the user briefly."
                self.resp_task = asyncio.create_task(self._respond(note))

    # ------------------------------------------------------------------ text mode (no mic) for debugging
    async def run_text(self, line):
        self.player.bind_loop(asyncio.get_running_loop())
        if self.audio and self.player.stream is None:
            self.player.open()
        self.resp_task = asyncio.create_task(self._respond(line))
        await self.resp_task
