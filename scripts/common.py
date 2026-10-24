"""Shared paths, local runtime settings, and verified public model downloads."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
# SHA-256 digests published by the upstream GitHub Release Assets API.
ASSETS = {
    "fp32": ("kokoro-v1.0.onnx", 325505369,
             "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"),
    "fp16": ("kokoro-v1.0.fp16.onnx", 163527961,
             "f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96"),
    "int8": ("kokoro-v1.0.int8.onnx", 114119327,
             "ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c"),
    "voices": ("voices-v1.0.bin", 28214398,
               "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}


def local_environment():
    """Override external cache/env settings for our own processes only."""
    cache = ROOT / ".cache"
    values = {
        "UV_CACHE_DIR": cache / "uv",
        "UV_PYTHON_INSTALL_DIR": cache / "python",
        "UV_PYTHON_BIN_DIR": cache / "bin",
        "UV_TOOL_DIR": cache / "tools",
        "UV_TOOL_BIN_DIR": cache / "bin",
        "UV_PROJECT_ENVIRONMENT": ROOT / ".venv",
        "XDG_CACHE_HOME": cache,
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "TMPDIR": cache / "tmp",
        "TEMP": cache / "tmp",
        "TMP": cache / "tmp",
    }
    for value in values.values():
        value.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in values.items()})
    env.update({"UV_LINK_MODE": "copy", "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "ONNX_PROVIDER": "CPUExecutionProvider"})
    # Do not inherit a caller's Python or alternate uv project configuration.
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_CONFIG_FILE",
                "UV_WORKING_DIR", "UV_PROJECT"):
        env.pop(key, None)
    return env


def configure_runtime():
    env = local_environment()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        os.environ.pop(key, None)
    os.environ.update(env)
    tempfile.tempdir = env["TMPDIR"]


def environment_python():
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(key):
    name, size, digest = ASSETS[key]
    path = MODELS / name
    if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Missing or damaged {name}. Run scripts/setup.py --model-variant "
                         f"{key if key != 'voices' else 'fp32'} to repair it.")
    return path


def download_asset(key, offline=False):
    try:
        return verify_asset(key)
    except ValueError:
        if offline:
            raise
    MODELS.mkdir(parents=True, exist_ok=True)
    name, size, expected = ASSETS[key]
    print(f"Downloading {name} ({size / 1_000_000:.0f} MB)...", flush=True)
    descriptor, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=MODELS)
    try:
        request = urllib.request.Request(RELEASE + name, headers={"User-Agent": "kokoro-tts-skill/0.1"})
        digest = hashlib.sha256()
        received = 0
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > size:
                        raise ValueError(f"Unexpected download size for {name}")
                    digest.update(block)
                    output.write(block)
        if received != size or digest.hexdigest() != expected:
            raise ValueError(f"Checksum/size mismatch for {name}; download rejected")
        os.replace(temporary, MODELS / name)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return MODELS / name


def emit_json(value):
    print(json.dumps(value, ensure_ascii=False))
