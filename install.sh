#!/usr/bin/env bash
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[search,dev]"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  pip install -e ".[mlx,kokoro,speaker,aec,smartturn]" || true
  command -v icalBuddy >/dev/null || echo "optional: brew install ical-buddy (calendar tool)"
else
  pip install -e ".[kokoro]" || true   # Linux/Windows dev path: faster-whisper + Kokoro
fi
python -m josie models
