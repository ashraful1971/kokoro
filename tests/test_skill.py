"""Run fast checks everywhere; opt in to real offline inference after setup."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import common
import speak


class LocalTest(unittest.TestCase):
    def setUp(self):
        directory = ROOT / ".cache" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test with spaces ", dir=directory)
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)


class PortableBehaviorTests(LocalTest):
    def test_blend_weights_are_normalized_and_order_is_preserved(self):
        result = speak.parse_blend("af_heart:70,bm_george:30")
        self.assertEqual(result, [("af_heart", 0.7), ("bm_george", 0.3)])
        for invalid in ("af_heart:0", "af_heart:-1", "af_heart:nan", "af_heart:inf", "af_heart:1:2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                speak.parse_blend(invalid)

    def test_split_segments_by_paragraph_and_line(self):
        text = "First step.\nstill first.\r\n\r\n\n  Second step.  \n\nThird."
        self.assertEqual(speak.split_segments(text, "paragraph"),
                         ["First step. still first.", "Second step.", "Third."])
        self.assertEqual(speak.split_segments("A.\n\nB.\n  \nC.", "line"), ["A.", "B.", "C."])
        self.assertEqual(speak.split_segments(" \n\n ", "paragraph"), [])

    def test_competing_output_is_not_overwritten(self):
        output = self.directory / "voice.wav"
        temporary = self.directory / "stage.part"
        temporary.write_bytes(b"new")
        output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            speak.publish(temporary, output, force=False)
        self.assertEqual(output.read_bytes(), b"existing")
        speak.publish(temporary, output, force=True)
        self.assertEqual(output.read_bytes(), b"new")

    def test_runtime_and_cache_paths_are_local_even_with_external_settings(self):
        with patch.dict(os.environ, {"UV_PROJECT_ENVIRONMENT": "/outside", "UV_CACHE_DIR": "/elsewhere",
                                    "PYTHONPATH": "/injected", "UV_PROJECT": "/wrong"}):
            env = common.local_environment()
        for key in ("UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "UV_PYTHON_INSTALL_DIR",
                    "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "TMPDIR"):
            self.assertTrue(Path(env[key]).is_relative_to(ROOT), key)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("UV_PROJECT", env)

    def test_checksum_rejection_preserves_existing_asset_and_removes_partial(self):
        payload = b"verified model"
        expected = hashlib.sha256(payload).hexdigest()
        asset = ("model.onnx", len(payload), expected)
        target = self.directory / "model.onnx"
        target.write_bytes(b"old")
        with patch.object(common, "MODELS", self.directory), patch.object(common, "ASSETS", {"fp32": asset}):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    common.download_asset("fp32")
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(self.directory.glob("*.part")), [])
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)), redirect_stdout(io.StringIO()):
                common.download_asset("fp32")
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network was used")):
                self.assertEqual(common.download_asset("fp32", offline=True), target)


@unittest.skipUnless(os.environ.get("KOKORO_INTEGRATION") == "1", "Set KOKORO_INTEGRATION=1 after setup")
class OfflineSpeechTests(LocalTest):
    def cli(self, *arguments, text=None, success=True):
        # Any Python network attempt fails, even when a developer has internet.
        guard = (
            "import sys,socket,runpy; "
            "reject=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network forbidden in offline test')); "
            "socket.socket.connect=reject; socket.socket.connect_ex=reject; socket.getaddrinfo=reject; "
            "script=sys.argv[1]; sys.argv=sys.argv[1:]; "
            "sys.path.insert(0,str(__import__('pathlib').Path(script).parent)); "
            "runpy.run_path(script,run_name='__main__')"
        )
        result = subprocess.run([str(common.environment_python()), "-c", guard,
                                 str(ROOT / "scripts" / "speak.py"), *map(str, arguments)],
                                input=text, capture_output=True, text=True, encoding="utf-8",
                                env=common.local_environment(), cwd=self.directory, timeout=240)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.directory.glob("*.part")), [])
        return result.stderr

    def check_audio(self, path, result):
        import numpy as np
        with wave.open(str(path)) as audio:
            self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()), (24000, 1, 2))
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            self.assertAlmostEqual(samples.size / 24000, result["duration_seconds"], places=2)
        self.assertGreater(samples.size, 2400)
        self.assertGreater(np.sqrt(np.mean(samples.astype(float) ** 2)), 20)

    def test_short_speech_and_output_protection(self):
        path = self.directory / "short voice.wav"
        result = self.cli("--text", "Welcome to this tutorial. Let's begin!", "--output", path)
        self.check_audio(path, result)
        digest = common.sha256(path)
        error = self.cli("--text", "Replacement", "--output", path, success=False)
        self.assertIn("--force", error)
        self.assertEqual(common.sha256(path), digest)
        replaced = self.cli("--text", "A different recording.", "--output", path, "--force")
        self.check_audio(path, replaced)
        self.assertNotEqual(common.sha256(path), digest)

    def test_long_tutorial_from_utf8_file_has_multiple_chunks(self):
        source = self.directory / "tutorial script.txt"
        source.write_text(
            "Welcome to this practical tutorial. First, create a folder for your project. "
            "Next, open a terminal and check the current directory. Keep your files organized, "
            "and save your work before moving to the next step. " * 9,
            encoding="utf-8")
        path = self.directory / "long tutorial.wav"
        result = self.cli("--file", source, "--output", path,
                          "--sentence-pause", "0.35", "--clause-pause", "0.15")
        self.check_audio(path, result)
        self.assertGreater(result["chunks"], 2)
        self.assertGreater(result["duration_seconds"], 40)

    def test_split_writes_clips_and_manifest_and_protects_existing(self):
        source = self.directory / "steps.txt"
        source.write_text("Open the terminal.\n\nRun the setup script.\nThen wait.\n\nDone.", encoding="utf-8")
        clips = self.directory / "clips dir"
        result = self.cli("--file", source, "--split", "paragraph", "--output-dir", clips)
        self.assertEqual(result["clips"], 3)
        manifest = json.loads((clips / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([clip["file"] for clip in manifest["clips"]], ["001.wav", "002.wav", "003.wav"])
        self.assertEqual(manifest["clips"][1]["text"], "Run the setup script. Then wait.")
        for clip in manifest["clips"]:
            path = clips / clip["file"]
            self.check_audio(path, {"duration_seconds": clip["duration_seconds"]})
        self.assertAlmostEqual(manifest["total_duration_seconds"],
                               sum(clip["duration_seconds"] for clip in manifest["clips"]), places=2)
        digest = common.sha256(clips / "001.wav")
        error = self.cli("--file", source, "--split", "paragraph", "--output-dir", clips, success=False)
        self.assertIn("--force", error)
        self.assertEqual(common.sha256(clips / "001.wav"), digest)
        lines = self.cli("--file", source, "--split", "line", "--output-dir", self.directory / "lines")
        self.assertEqual(lines["clips"], 4)

    def test_split_option_conflicts_fail_clearly(self):
        for flags, expected in ((["--split", "line"], "requires --output-dir"),
                                (["--output-dir", str(self.directory / "x")], "requires --split"),
                                (["--split", "line", "--output-dir", str(self.directory / "x"),
                                  "--output", str(self.directory / "x.wav")], "cannot be combined")):
            with self.subTest(flags=flags):
                self.assertIn(expected, self.cli("--text", "Hello", *flags, success=False))
        self.assertFalse((self.directory / "x").exists())

    def test_british_stdin_speed_and_untrimmed_audio(self):
        path = self.directory / "british.wav"
        result = self.cli("--voice", "bm_george", "--speed", "1.2", "--no-trim", "--output", path,
                          text="Here is a British voice reading from standard input.")
        self.check_audio(path, result)
        self.assertEqual(result["lang"], "en-gb")
        self.assertEqual(result["speed"], 1.2)

    def test_blend_style_export_import_and_phonemes(self):
        style = self.directory / "custom style.npy"
        exported = self.cli("--voice", "af_heart:70,am_michael:30", "--save-style", style)
        self.assertTrue(style.exists())
        self.assertEqual(exported["shape"], [510, 1, 256])
        path = self.directory / "phonemes.wav"
        result = self.cli("--style", style, "--text", "həlˈoʊ wˈɜːld.", "--phonemes", "--output", path)
        self.check_audio(path, result)

    def test_voices_are_discoverable_and_inputs_fail_clearly(self):
        result = self.cli("--list-voices")
        self.assertIn("af_heart", [voice["id"] for voice in result["voices"]])
        target = self.directory / "invalid.wav"
        cases = [
            (["--text", "  "], "empty"),
            (["--text", "Hello", "--voice", "af_unknown"], "Unknown English voice"),
            (["--text", "Hello", "--speed", "0.1"], "Speed"),
            (["--text", "Hello", "--clause-pause", "-1"], "nonnegative"),
            (["--text", "☃", "--phonemes"], "Unsupported phoneme"),
        ]
        for index, (flags, expected) in enumerate(cases):
            with self.subTest(flags=flags):
                target = self.directory / f"invalid-{index}.wav"
                error = self.cli(*flags, "--output", target, success=False)
                self.assertIn(expected, error)
                self.assertFalse(target.exists())
        self.assertFalse((self.directory / "timings.json").exists())

    def test_invalid_custom_style_is_rejected_without_output(self):
        import numpy as np
        style = self.directory / "invalid.npy"
        np.save(style, np.zeros((2, 3)))
        target = self.directory / "invalid.wav"
        error = self.cli("--style", style, "--text", "Hello", "--output", target, success=False)
        self.assertIn("Style must be", error)
        self.assertFalse(target.exists())

    def test_timing_capable_model_adapter_and_continuous_options(self):
        # Exercise option forwarding separately from actual model inference.
        common.configure_runtime()
        import numpy as np
        from kokoro_onnx.sliding import Timing
        engine = SimpleNamespace(has_timings=True, create_timed=Mock(return_value=(
            np.sin(np.arange(24000) * 0.1).astype(np.float32) * 0.2,
            24000, [Timing("h", 0.0, 0.1)])))
        target = self.directory / "timed.wav"
        timing = self.directory / "timed.json"
        args = speak.parser_for_cli().parse_args([
            "--text", "Hello", "--continuous", "--speed", "0.8", "--lang", "en-gb",
            "--sentence-pause", "0.4", "--clause-pause", "0.2",
            "--output", str(target), "--timings-json", str(timing)])
        with patch("onnxruntime.InferenceSession"), patch("kokoro_onnx.Kokoro.from_session", return_value=engine):
            with redirect_stdout(io.StringIO()) as output:
                speak.run(args)
        result = json.loads(output.getvalue())
        self.check_audio(target, result)
        self.assertEqual(json.loads(timing.read_text())["timings"],
                         [{"phoneme": "h", "start": 0.0, "end": 0.1}])
        kwargs = engine.create_timed.call_args.kwargs
        self.assertEqual((kwargs["continuous"], kwargs["speed"], kwargs["lang"],
                          kwargs["sentence_pause"], kwargs["clause_pause"]),
                         (True, 0.8, "en-gb", 0.4, 0.2))

    def test_real_continuous_speech_with_phoneme_timestamps(self):
        path = self.directory / "continuous.wav"
        timing = self.directory / "alignment.json"
        result = self.cli("--text", "This is continuous narration. Timing data helps align the audio.",
                          "--continuous", "--timings-json", timing, "--output", path)
        self.check_audio(path, result)
        timings = json.loads(timing.read_text())["timings"]
        self.assertGreater(len(timings), 10)
        for entry in timings:
            self.assertLessEqual(0, entry["start"])
            self.assertLessEqual(entry["start"], entry["end"])
            self.assertLessEqual(entry["end"], result["duration_seconds"] + 0.001)

    def test_model_without_durations_is_rejected_for_advanced_options(self):
        common.configure_runtime()
        target = self.directory / "unsupported.wav"
        for flags in (["--continuous"], ["--timings-json", str(self.directory / "timing.json")]):
            args = speak.parser_for_cli().parse_args([
                "--text", "Hello", "--output", str(target), *flags])
            with patch("onnxruntime.InferenceSession"), patch("kokoro_onnx.Kokoro.from_session",
                                                              return_value=SimpleNamespace(has_timings=False)):
                with self.assertRaisesRegex(ValueError, "no duration output"):
                    speak.run(args)
            self.assertFalse(target.exists())

    def test_batch_failure_leaves_no_partial_wav_or_style(self):
        common.configure_runtime()
        import numpy as np
        engine = SimpleNamespace(
            has_timings=False,
            _prepare=Mock(return_value=(None, "h h", [("h", 0.2), ("h", 0)])),
            _create_batch=Mock(side_effect=[(np.ones(2400, dtype=np.float32) * 0.1, None),
                                           RuntimeError("inference failed")]))
        target = self.directory / "failed.wav"
        style = self.directory / "failed.npy"
        args = speak.parser_for_cli().parse_args([
            "--text", "Hello", "--output", str(target), "--save-style", str(style)])
        with patch("onnxruntime.InferenceSession"), patch("kokoro_onnx.Kokoro.from_session", return_value=engine):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                speak.run(args)
        self.assertFalse(target.exists())
        self.assertFalse(style.exists())
        self.assertEqual(list(self.directory.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
