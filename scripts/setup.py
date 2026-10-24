#!/usr/bin/env python3
"""Install only into this skill, then download and verify its model assets."""

import argparse
import shutil
import subprocess
import sys

from common import ROOT, download_asset, environment_python, local_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-variant", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--offline", action="store_true", help="Use only local packages and verified assets")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if not uv:
        parser.error("uv is required. Provide an existing uv executable on PATH; "
                     "this skill does not install tools globally. See README.md.")
    if not (ROOT / "uv.lock").is_file():
        parser.error("uv.lock is missing. Obtain the complete skill repository.")
    env = local_environment()
    command = [uv, "sync", "--project", str(ROOT), "--python", "3.12", "--locked",
               "--no-dev", "--no-install-project", "--no-build", "--no-config",
               "--default-index", "https://pypi.org/simple"]
    if args.offline:
        command += ["--offline", "--no-python-downloads"]
    subprocess.run(command, env=env, cwd=ROOT, check=True)
    # Fail here if the packaged phonemizer cannot load; never install a system fallback.
    subprocess.run([str(environment_python()), "-c",
                    "import ctypes, espeakng_loader; "
                    "ctypes.CDLL(espeakng_loader.get_library_path()); "
                    "print('Local pronunciation library: OK')"], env=env, check=True)
    for key in (args.model_variant, "voices"):
        download_asset(key, offline=args.offline)
    print(f"Ready. Run: python scripts/speak.py --text \"Hello!\" "
          f"--model-variant {args.model_variant} --output outputs/hello.wav")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        sys.exit(1)
