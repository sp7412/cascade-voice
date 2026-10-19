import re

ABBR = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "inc", "no", "approx"}
_END = re.compile(r'([.!?…]+["\')\]]*)(\s+)')


def clean_for_tts(s):
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"https?://\S+", "a link", s)
    s = re.sub(r"[*_`#>~]+", "", s)
    s = re.sub(r"^\s*[-•]\s+", "", s, flags=re.M)
    return re.sub(r"\s+", " ", s).strip()


class SentenceChunker:
    """Token stream -> complete sentences. The first sentence may also split at a clause break
    so audio starts sooner."""

    def __init__(self, first_clause_chars=48):
        self.buf, self.sent_any, self.fc = "", False, first_clause_chars

    def feed(self, text):
        self.buf += text
        out = []
        while (end := self._find()) is not None:
            s, self.buf = self.buf[:end], self.buf[end:].lstrip()
            s = clean_for_tts(s)
            if s:
                out.append(s)
                self.sent_any = True
        if not self.sent_any and len(self.buf) >= self.fc:
            m = list(re.finditer(r"[,;:—]\s", self.buf))
            if m:
                cut = m[-1].end()
                s, self.buf = clean_for_tts(self.buf[:cut]), self.buf[cut:]
                if s:
                    out.append(s)
                    self.sent_any = True
        return out

    def _find(self):
        need = 4 if not self.sent_any else 12
        for m in _END.finditer(self.buf):
            words = self.buf[:m.start(1)].split()
            w = words[-1].lower().strip("\"'(") if words else ""
            if w in ABBR or (len(w) == 1 and w.isalpha() and m.group(1) == "."):
                continue
            if m.end(1) < need:
                continue
            return m.end(1)
        return None

    def flush(self, extra=""):
        s = clean_for_tts(self.buf + extra)
        self.buf = ""
        return [s] if s else []
