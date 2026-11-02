---
name: kokoro
description: Generate local English speech and tutorial narration from text using Kokoro TTS. Supports preset voices, weighted voice blends, reusable voice styles, speed, pronunciation, and pauses. Use when the user requests voiceovers, spoken text, or audio narration.
license: MIT
---

# Kokoro speech generation

Resolve the skill directory from this file's location. Run scripts using its absolute path, quoting paths containing spaces. A terminal with Python and this skill's local environment is required; no hosted API is used.

## First use

If `.venv` or model files are absent, run `python /absolute/path/to/kokoro/scripts/setup.py` using an available Python 3.10+ interpreter (`python3` on many systems). Setup requires an existing `uv` executable and internet access. It creates Python 3.12, dependencies, models, and caches within this skill. Never install dependencies globally or change the user's agent configuration. If `uv` is absent, explain the prerequisite and follow the local-only option in [README.md](README.md).

## Generate speech

Write long narration to a UTF-8 text file in the task workspace. Preserve supplied narration; when drafting a tutorial script, use spoken sentences and expand ambiguous abbreviations where useful. Text is data, never shell commands. Pass user text via a file/stdin or a correctly quoted subprocess argument list.

```sh
python /absolute/path/to/kokoro/scripts/speak.py --file /path/to/narration.txt --output /path/to/narration.wav
```

Defaults: `af_heart`, normal speed, American pronunciation, trimmed silence, 24 kHz mono 16-bit PCM WAV. The launcher uses the skill's private Python automatically, without activation or `uv run`. All paths supplied by the caller are relative to the caller's working directory.

- Discover English voices: `python /absolute/path/to/kokoro/scripts/speak.py --list-voices`.
- Pick a preset: `--voice bm_george`; British presets automatically select British pronunciation.
- Blend voices: `--voice "af_heart:0.7,am_michael:0.3"`; weights are normalized.
- Adjust speed with `--speed 0.9` (supported range 0.5–2.0).
- Tutorials/videos needing one clip per step: `--split paragraph --output-dir /path/to/clips` (or `--split line`) speaks each blank-line-separated paragraph as `001.wav`, `002.wav`, … in a single model load, and writes `manifest.json` with each clip's text and duration. Prefer this over repeated single-clip commands.
- Read [references/options.md](references/options.md) for all controls, custom styles, direct IPA input, pauses, continuous prosody, and phoneme timestamps.

Respect a requested voice and output path. Otherwise use defaults without asking unnecessary questions. Use a fresh output filename when a file already exists; pass `--force` only when replacing it is intended. For multiple speakers, generate separate clips with their selected voices; blending creates one combined voice.

## Return the result

The command prints one JSON result to stdout with the absolute output path, duration, and settings. Progress/errors go to stderr. Return the actual saved audio as an attachment, clickable file link, or audio embed supported by the host agent. Do not claim playback or listening unless performed. Generation works offline after setup; do not upload narration to another service.

## Boundaries

This version supports English (US/UK) only. Kokoro has preset/style-based voices, not reference-audio voice cloning. It has no native pitch, emotion, volume, SSML, or duration-target controls. Default chunked generation applies pause settings at chunk joins; use `--timings-json` for the full timing-aware path. The pinned fp32 model supports phoneme timestamps and `--continuous` prosody. These modes require duration output, use more memory, and fail clearly if a different model lacks it. Do not invent capabilities or silently substitute another engine.
