#!/usr/bin/env python3
"""Generate local 24 kHz PCM WAV speech using Kokoro; stdout is a JSON result."""

import argparse
import ctypes
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

from common import (ROOT, configure_runtime, emit_json, environment_python,
                    local_environment, verify_asset)


def finite_number(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Must be a number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("Must be finite")
    return number


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to speak (quote it as one argument)")
    source.add_argument("--file", type=Path, help="UTF-8 text file; otherwise read stdin")
    parser.add_argument("--output", type=Path, help="Destination .wav file")
    voice = parser.add_mutually_exclusive_group()
    voice.add_argument("--voice", help="English voice ID or blend, e.g. af_heart:0.7,am_michael:0.3")
    voice.add_argument("--style", type=Path, help="Custom/blended voice style .npy (no pickle)")
    parser.add_argument("--save-style", type=Path, help="Export chosen voice/blend to .npy")
    parser.add_argument("--speed", type=finite_number, default=1.0, help="0.5–2.0; default 1.0")
    parser.add_argument("--lang", choices=("en-us", "en-gb"), help="Pronunciation; inferred from first voice")
    parser.add_argument("--phonemes", action="store_true", help="Treat input as Kokoro IPA phonemes")
    parser.add_argument("--trim", action=argparse.BooleanOptionalAction, default=True,
                        help="Trim chunk-edge silence (default on)")
    parser.add_argument("--sentence-pause", type=finite_number, default=0.25, metavar="SECONDS")
    parser.add_argument("--clause-pause", type=finite_number, default=0.1, metavar="SECONDS")
    parser.add_argument("--continuous", action="store_true",
                        help="Overlapping-window prosody; requires duration output (fp32 supports it)")
    parser.add_argument("--timings-json", type=Path,
                        help="Phoneme timestamps; requires duration output (fp32 supports it)")
    parser.add_argument("--model-variant", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--model", type=Path, help="Trusted custom ONNX model; overrides --model-variant")
    parser.add_argument("--voices-file", type=Path, help="Trusted custom voice bundle in NumPy NPZ format")
    parser.add_argument("--vocab", type=Path, help='Custom vocabulary JSON containing a "vocab" object')
    parser.add_argument("--espeak-library", type=Path, help="Explicit local eSpeak shared library")
    parser.add_argument("--espeak-data", type=Path, help="Explicit local eSpeak data directory")
    parser.add_argument("--list-voices", action="store_true", help="List English voices as JSON")
    parser.add_argument("--force", action="store_true", help="Allow replacing existing output files")
    return parser


def parse_blend(expression):
    entries = []
    for component in expression.split(","):
        parts = component.strip().split(":")
        if len(parts) > 2 or not parts[0]:
            raise ValueError("Use voice IDs or voice:weight pairs separated by commas")
        weight = float(parts[1]) if len(parts) == 2 else 1.0
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Voice weights must be finite and nonnegative")
        entries.append((parts[0], weight))
    total = sum(weight for _, weight in entries)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Voice weights must have a positive finite sum")
    return [(name, weight / total) for name, weight in entries]


def destination(path, suffix, force):
    path = path.expanduser().absolute()
    if path.suffix.lower() != suffix:
        raise ValueError(f"Output must have a {suffix} extension: {path}")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Output must be a regular file, not a directory or symlink: {path}")
    if path.exists() and not force:
        raise ValueError(f"Output exists: {path}. Use --force to overwrite.")
    return path


def publish(temporary, output, force):
    if force:
        os.replace(temporary, output)
    else:
        # Hard-link creation is atomic and refuses a competing writer's existing file.
        os.link(temporary, output)
        Path(temporary).unlink()


def temporary_for(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, path = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".part", dir=output.parent)
    os.close(descriptor)
    return Path(path)


def english_voice(name):
    return name.startswith(("af_", "am_", "bf_", "bm_"))


def generate(engine, text, voice, args, temporary):
    import numpy as np

    frames, chunks = 0, 0
    timestamps = []
    kwargs = dict(voice=voice, speed=args.speed, lang=args.lang,
                  is_phonemes=args.phonemes, trim=args.trim,
                  sentence_pause=args.sentence_pause, clause_pause=args.clause_pause)

    def audio_chunks():
        nonlocal timestamps
        if args.continuous or args.timings_json:
            audio, rate, timestamps = engine.create_timed(text, continuous=args.continuous, **kwargs)
            yield audio, rate
        else:
            # The public async stream depends on executor/event-loop wakeups,
            # which can hang in restricted agent sandboxes. Use the same batch
            # implementation synchronously. These two internal methods are
            # deliberately tied to the locked kokoro-onnx 0.6.1 dependency.
            style, _, batches = engine._prepare(
                text, voice, args.speed, args.lang, args.phonemes,
                args.sentence_pause, args.clause_pause)
            for phonemes, pause in batches:
                audio, _ = engine._create_batch(phonemes, style, args.speed, args.trim, pause)
                yield audio, 24000

    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        for samples, rate in audio_chunks():
            samples = np.asarray(samples).reshape(-1)
            if rate != 24000 or samples.size == 0 or not np.isfinite(samples).all():
                raise ValueError("Engine returned invalid audio")
            pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
            output.writeframes(pcm.tobytes())
            frames += samples.size
            chunks += 1
            print(f"Generated chunk {chunks} ({frames / 24000:.1f}s total)", file=sys.stderr)
    if not frames:
        raise ValueError("No speech was generated from the input")
    return frames, chunks, timestamps


def run(args):
    import numpy as np
    import espeakng_loader
    from kokoro_onnx import EspeakConfig, Kokoro
    import onnxruntime as ort

    ort.disable_telemetry_events()

    if not 0.5 <= args.speed <= 2:
        raise ValueError("Speed must be between 0.5 and 2.0")
    if args.sentence_pause < 0 or args.clause_pause < 0:
        raise ValueError("Pauses must be nonnegative seconds")
    voices_path = args.voices_file or verify_asset("voices")
    # Read names/styles without loading the model, and never allow pickle data.
    with np.load(voices_path, allow_pickle=False) as bundle:
        names = sorted(name for name in bundle.files if english_voice(name))
        if args.list_voices:
            emit_json({"voices": [{"id": name, "lang": "en-gb" if name[0] == "b" else "en-us"}
                                  for name in names]})
            return
        blend = parse_blend(args.voice or "af_heart")
        if args.style:
            voice = np.load(args.style, allow_pickle=False)
            if not isinstance(voice, np.ndarray):
                voice.close()
                raise ValueError("Style must be a NumPy .npy array")
        else:
            for name, _ in blend:
                if name not in names:
                    raise ValueError(f"Unknown English voice: {name}. Use --list-voices.")
            voice = sum(bundle[name].astype(np.float32) * weight for name, weight in blend)
        if (not names or voice.shape != bundle[names[0]].shape or
                not np.issubdtype(voice.dtype, np.floating) or not np.isfinite(voice).all()):
            raise ValueError("Style must be a finite floating-point array matching the voice bundle shape")
        voice = voice.astype(np.float32)
    args.lang = args.lang or ("en-gb" if not args.style and blend[0][0][0] == "b" else "en-us")
    outputs = []
    if args.output:
        outputs.append(destination(args.output, ".wav", args.force))
    elif not args.save_style:
        raise ValueError("--output is required for speech generation")
    style_output = destination(args.save_style, ".npy", args.force) if args.save_style else None
    timing_output = destination(args.timings_json, ".json", args.force) if args.timings_json else None
    for path in (style_output, timing_output):
        if path:
            outputs.append(path)
    inputs = [path.expanduser().resolve() for path in
              (args.file, args.style, args.model, args.voices_file, args.vocab) if path]
    if any(path.resolve() in inputs for path in outputs):
        raise ValueError("An output path must not replace an input file")
    if not args.output:
        if args.timings_json or args.continuous or args.text is not None or args.file:
            raise ValueError("Provide --output when supplying speech input or requesting timings")
        temporary = temporary_for(style_output)
        try:
            with temporary.open("wb") as output:
                np.save(output, voice, allow_pickle=False)
            publish(temporary, style_output, args.force)
        finally:
            temporary.unlink(missing_ok=True)
        emit_json({"style": str(style_output), "shape": list(voice.shape)})
        return
    output_path = outputs[0]
    if args.text is not None:
        text = args.text
    elif args.file:
        text = args.file.read_text(encoding="utf-8-sig")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ValueError("Supply --text, --file, or text on stdin")
    if not text.strip():
        raise ValueError("Input text is empty")
    model_path = args.model or verify_asset(args.model_variant)
    lib = str(args.espeak_library or espeakng_loader.get_library_path())
    data = str(args.espeak_data or espeakng_loader.get_data_path())
    ctypes.CDLL(lib)  # Fail locally rather than letting upstream fall back to system eSpeak.
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    # Keep initialization lightweight for short, one-shot narration commands.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    engine = Kokoro.from_session(session, str(voices_path), espeak_config=EspeakConfig(lib, data),
                                 vocab_config=str(args.vocab) if args.vocab else None)
    if (args.continuous or timing_output) and not engine.has_timings:
        raise ValueError("This model has no duration output. --continuous and --timings-json "
                         "require a timing-capable model; use the pinned fp32 model or a "
                         "compatible custom --model. See references/options.md.")
    if args.phonemes:
        unknown = set(text) - set(engine.tokenizer.vocab) - set("\r\n\t")
        if unknown:
            raise ValueError(f"Unsupported phoneme characters: {''.join(sorted(unknown))!r}")
    staged = []
    try:
        temporary = temporary_for(output_path)
        staged.append((temporary, output_path))
        frames, chunks, timestamps = generate(engine, text, voice, args, temporary)
        if style_output:
            temporary = temporary_for(style_output)
            staged.append((temporary, style_output))
            with temporary.open("wb") as output:
                np.save(output, voice, allow_pickle=False)
        if timing_output:
            temporary = temporary_for(timing_output)
            staged.append((temporary, timing_output))
            temporary.write_text(json.dumps({"unit": "seconds", "timings": [asdict(t) for t in timestamps]},
                                            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Stage everything first. Publish the primary WAV last.
        for temporary, target in staged[1:] + staged[:1]:
            publish(temporary, target, args.force)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    emit_json({"output": str(output_path), "duration_seconds": round(frames / 24000, 3),
               "sample_rate": 24000, "channels": 1, "sample_width_bits": 16,
               "chunks": chunks, "lang": args.lang, "speed": args.speed,
               "voice": str(args.style) if args.style else blend,
               "style": str(style_output) if style_output else None,
               "timings": str(timing_output) if timing_output else None})


def main():
    parser = parser_for_cli()
    args = parser.parse_args()
    python = environment_python()
    # A system Python can launch the skill without activation or uv/network access.
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        if not python.is_file():
            parser.error("Local environment is missing. Run: python scripts/setup.py")
        return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
                               env=local_environment())
    configure_runtime()
    try:
        run(args)
    except Exception as error:
        print(f"Speech error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        sys.exit(130)
