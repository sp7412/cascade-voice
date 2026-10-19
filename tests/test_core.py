import asyncio
from josie.chunker import SentenceChunker, clean_for_tts
from josie.llm import ThinkStripper, ToolCall
from josie.pipeline import same_utterance, trim_history
from josie.playback import Player, Segment
from josie.tools import ToolRegistry
from josie.tts import RTFTracker
from josie.turn import Decision, SmartTurn
import numpy as np


def test_think_stripper_across_chunks():
    s, out = ThinkStripper(), ""
    for c in ["Hi <th", "ink>secret reason", "ing</thi", "nk> there."]:
        out += s.feed(c)
    assert (out + s.flush()).strip() == "Hi  there." or (out + s.flush()).replace("  ", " ").strip() == "Hi there."
    assert "secret" not in out


def test_chunker():
    c = SentenceChunker()
    got = c.feed("Sure. Dr. Smith will see you at 3.5 hours. Next") + c.flush()
    assert got[0] == "Sure." and "Dr. Smith will see you at 3.5 hours." in got[1]
    assert clean_for_tts("**Hi** see https://x.com now") == "Hi see a link now"


def test_rtf_preroll():
    r = RTFTracker()
    r.update(2.0, 1.0)
    assert r.preroll(5) == 0.0
    r = RTFTracker(); r.update(1.0, 2.0)  # s = 0.5
    assert abs(r.preroll(4) - (4 * .5 * 1.1 + .15)) < 1e-6
    r = RTFTracker(); r.update(0.1, 10)
    assert r.preroll(100) == 3.0


def test_turn_decisions():
    t = SmartTurn()
    assert t.decide(100, "what time is it?", 900) == Decision.WAIT
    assert t.decide(250, "what time is it?", 900) == Decision.SPECULATE
    assert t.decide(550, "what time is it?", 900) == Decision.END
    assert t.decide(800, "I was thinking and", 900) == Decision.SPECULATE  # hold word: keep waiting
    assert t.decide(100, "", 64) == Decision.WAIT and t.decide(400, "", 64) == Decision.DISCARD


def test_spoken_text_cut():
    p = Player(24000)
    seg = Segment("one two three four", np.zeros(24000, np.float32))
    p.done, p.cur, p.pos = [Segment("First sentence.", np.zeros(10, np.float32))], seg, 12000
    assert p.spoken_text().startswith("First sentence. one two")
    assert "four" not in p.spoken_text()


def test_history_trim_and_match():
    h = [{"role": "user", "content": "a" * 50}, {"role": "assistant", "content": "b" * 50},
         {"role": "user", "content": "c" * 10}, {"role": "assistant", "content": "d" * 10}]
    trim_history(h, 40)
    assert h[0]["role"] == "user" and len(h) == 2
    assert same_utterance("open the pod bay doors", "Open the pod bay doors.")


def test_tools_parallel_deadline_cancel():
    async def go():
        reg = ToolRegistry(0.2)

        @reg.register("fast", "x")
        async def fast(): return "ok"

        @reg.register("slow", "x")
        async def slow(): await asyncio.sleep(5)
        res = await reg.run_calls([ToolCall("1", "fast", {}), ToolCall("2", "slow", {})], asyncio.Event())
        assert dict((c.name, o) for c, o in res)["fast"] == "ok"
        assert "timed out" in dict((c.name, o) for c, o in res)["slow"]
        ev = asyncio.Event()
        asyncio.get_running_loop().call_later(0.05, ev.set)
        reg.tools["slow"].deadline = 5
        try:
            await reg.run_calls([ToolCall("3", "slow", {})], ev)
            assert False
        except asyncio.CancelledError:
            pass
    asyncio.run(go())


def test_http_api_bearer_auth_and_audio_helpers():
    from josie.server import authorized, pcm16, split_sentences, wav_header

    assert authorized("", None)  # Loopback-only mode is allowed without a key.
    assert authorized("secret", "Bearer secret")
    assert not authorized("secret", None)
    assert not authorized("secret", "Bearer wrong")
    assert not authorized("secret", "Basic secret")
    assert pcm16(np.array([0.0, 1.0, -1.0], dtype=np.float32)) == (
        bytes([0, 0, 255, 127, 1, 128])
    )
    assert split_sentences("Hello there. How are you?") == [
        "Hello there.", "How are you?"
    ]
    header = wav_header(24000)
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert len(header) == 44
