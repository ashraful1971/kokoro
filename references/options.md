# Complete synthesis options (kokoro-onnx 0.6.1)

Run `python scripts/speak.py --help` for command syntax. Paths below are
relative to the caller's current directory, not the skill directory.

| Flag | Meaning / default |
| --- | --- |
| `--text TEXT` | Literal text; mutually exclusive with `--file` |
| `--file PATH` | UTF-8 input, with optional BOM; otherwise stdin |
| `--output PATH.wav` | Required for speech; 24 kHz mono 16-bit WAV |
| `--split paragraph/line` | Speak each blank-line-separated paragraph (or each line) as its own clip; needs `--output-dir` |
| `--output-dir DIR` | Folder for `--split` clips (`001.wav`, …) and `manifest.json`; replaces `--output` |
| `--list-voices` | English preset IDs and their default pronunciation, as JSON |
| `--voice ID` | Preset voice; default `af_heart` |
| `--voice "ID:WEIGHT,ID:WEIGHT"` | Arbitrary weighted voice blend; weights normalized to sum to one |
| `--style PATH.npy` | Reuse a custom voice-style array; mutually exclusive with `--voice` |
| `--save-style PATH.npy` | Export the selected style; can run without text/output |
| `--speed NUMBER` | 0.5–2.0, default 1.0; lower is slower |
| `--lang en-us` / `en-gb` | Pronunciation; inferred from first blend/preset voice, or US for imported styles |
| `--phonemes` | Input is Kokoro IPA symbols; skips text phonemization |
| `--trim` / `--no-trim` | Trim leading/trailing chunk silence; default on |
| `--sentence-pause SECONDS` | Default 0.25; finite nonnegative number |
| `--clause-pause SECONDS` | Default 0.1; finite nonnegative number |
| `--continuous` | Continuous overlapping-window prosody; requires duration-capable model (pinned fp32 supports it) |
| `--timings-json PATH.json` | Phoneme/start/end timestamps in seconds; requires duration-capable model |
| `--model-variant fp32/fp16/int8` | Bundled model variant; default fp32; set up that variant first |
| `--model PATH.onnx` | Trusted custom ONNX model; overrides the variant |
| `--voices-file PATH` | Custom NPZ voice bundle (upstream `.bin` is NPZ data) |
| `--vocab PATH.json` | Override model vocabulary using a JSON object with a `vocab` mapping |
| `--espeak-library PATH` | Explicit local pronunciation shared library |
| `--espeak-data DIRECTORY` | Its matching eSpeak data directory |
| `--force` | Allow replacing output WAV, style, and timing files |

## Choosing and blending voices

English presets start with `af_` (American female), `am_` (American male),
`bf_` (British female), or `bm_` (British male). List the actual downloaded
bundle rather than assuming a particular name exists. Other languages in
the bundle are outside this skill's English-first scope.

Blending changes timbre and vocal character. Use any number of supported
English presets; equal weights can be written as `--voice "af_heart,af_bella"`.
Weights must be finite, nonnegative, and have a positive sum. A blend is one
voice, not a multi-speaker dialogue. Mixing accents is allowed; `--lang` sets
text pronunciation, while embeddings retain their learned vocal traits.

```sh
python scripts/speak.py --voice "af_heart:3,af_bella:1" --save-style outputs/narrator.npy
python scripts/speak.py --style outputs/narrator.npy --lang en-gb --file narration.txt --output outputs/narration.wav
```

Style files are finite floating-point NumPy arrays with exactly the same shape
as styles in the selected bundle (normally `[510, 1, 256]`). Pickle loading is
disabled. This is style reuse/blending, not cloning from a recording.

## Pronunciation, rhythm, and long scripts

Use punctuation and paragraph breaks in ordinary text. For precise
pronunciation, `--phonemes` accepts a full IPA phoneme transcript using the
model's vocabulary. It does not accept SSML or per-word markup. Unknown IPA
symbols produce an error rather than being silently removed.

```sh
python scripts/speak.py --text 'həlˈoʊ wˈɜːld.' --phonemes --output outputs/phonemes.wav
python scripts/speak.py --file tutorial.txt --speed 0.9 --sentence-pause 0.4 --clause-pause 0.15 --output outputs/tutorial.wav
```

Kokoro splits long input at its 510-phoneme window, preferring punctuation
and word boundaries. The default streaming path writes each audio chunk
directly to disk. Original line breaks are normalized as whitespace by the
engine; they are not explicit timed silences.

With the default chunked path, pauses are inserted at chunk joins when trimming is
enabled. They do **not** guarantee a pause after every sentence/clause inside
a chunk. With `--no-trim`, these chunk-join pause controls have no effect.
For pauses at each punctuation mark, use the timing-aware path with
`--timings-json`. For separately controlled segments, render separate clips
and assemble them in the tutorial's audio/video workflow.

## One clip per paragraph or line

For tutorials where each step needs its own audio (to time against video), use
`--split` instead of running the command once per clip. The model loads once and
every clip uses the same voice and settings.

```sh
python scripts/speak.py --file tutorial.txt --split paragraph --output-dir outputs/clips
```

This writes `001.wav`, `002.wav`, … (three digits, more if needed) and a
`manifest.json`:

```json
{"unit":"seconds","sample_rate":24000,"lang":"en-us","speed":1.0,"voice":[["af_heart",1.0]],
 "total_duration_seconds":12.4,"clips":[{"index":1,"file":"001.wav","text":"...","duration_seconds":6.1,"chunks":1}]}
```

`paragraph` splits on blank lines (line breaks inside a paragraph become
spaces); `line` treats every non-empty line as a clip. Empty segments are
skipped. `--split` cannot be combined with `--output`, `--timings-json` or
`--save-style`. Existing clips need `--force`; clips left over from an earlier,
longer run are not deleted, so treat `manifest.json` as the list of current
clips. If any clip fails, nothing is published. To get one joined file, use the
FFmpeg concat recipe below.

## Continuous prosody and timestamps

The pinned fp32 model has both `waveform` and `duration` outputs and supports
`--continuous` and `--timings-json`. Older or custom exports may lack durations;
these options fail clearly if the selected model cannot provide them.

```sh
python scripts/speak.py --file tutorial.txt --continuous --timings-json outputs/timings.json --output outputs/continuous.wav
```

For a custom model, pass `--model` and, if needed, matching voices/vocabulary.
See the upstream [export tool](https://github.com/thewh1teagle/kokoro-onnx/blob/main/scripts/export.py).
This skill does not download extra export dependencies automatically.

These advanced modes synthesize the complete waveform in memory through
`create_timed`; continuous prosody uses overlapping windows and more CPU work.
The upstream continuous path does not apply the ordinary chunk-edge trim flag.
Timestamp JSON has the shape:

```json
{"unit":"seconds","timings":[{"phoneme":"h","start":0.0,"end":0.08}]}
```

Timings are phoneme alignment estimates from that model, not ready-made word
subtitles. The pinned fp32 model receives a real continuous/timing generation
test. Custom models still need validation for their own outputs and vocabulary.

## Native limits

Kokoro does not expose pitch, emotion/style prompt, loudness, reference-audio
cloning, SSML, sample-rate selection, or a fixed output duration. Speed and
voice/style selection influence delivery but cannot promise a precise emotion
or accent transformation. Audio gain, resampling, MP3 conversion, and timeline
assembly are separate post-processing tasks. This skill does not install
FFmpeg or invent engine parameters for them. If FFmpeg is already available,
common follow-ups are:

```sh
ffmpeg -i narration.wav -codec:a libmp3lame -q:a 2 narration.mp3   # MP3
ffmpeg -i narration.wav -af "volume=1.5" louder.wav                 # gain
ffmpeg -i narration.wav -ar 44100 narration-44k.wav                 # resample
```

To join clips, list them in a text file (`file 'a.wav'` per line) and run
`ffmpeg -f concat -safe 0 -i list.txt -c copy joined.wav`.

Kokoro's voice bundle also contains non-English voices (French, Spanish, Hindi,
Italian, Portuguese, Japanese, Chinese). They are not exposed
here because this skill is built and tested for English only; Japanese and
Chinese would also need extra text-processing packages that are not installed.
Bengali is not supported by Kokoro.
