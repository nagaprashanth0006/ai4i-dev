#!/usr/bin/env python3
"""Run a short live demo with the same Python interpreter as this script."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCENARIOS = {
    "single": ("single.json", "Assess sample-a with the ML model. Explain its score and alert cutoff in simple terms."),
    "compare": ("compare.json", "Use the model to assess both sample-a and sample-b. Which gets the higher score? Mention any warnings."),
    "missing": ("missing.json", "Assess missing-sensor. Explain how the missing reading affects this request."),
    "invalid": ("invalid.json", "Assess bad-sensor using the model. If it cannot run, explain what I need to fix."),
    "no-data": (None, "Is machine M-017 currently showing a failure?"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIOS, nargs="?", default="single")
    parser.add_argument("--model", default="qwen3.5:9b")
    args = parser.parse_args()
    filename, prompt = SCENARIOS[args.scenario]
    command = [sys.executable, str(HERE / "app.py"), "--model", args.model, "--prompt", prompt, "--trace"]
    if filename:
        command += ["--observations", str(HERE / "examples" / filename)]
    if args.scenario in ("single", "compare", "missing"):
        command += ["--require-prediction"]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
