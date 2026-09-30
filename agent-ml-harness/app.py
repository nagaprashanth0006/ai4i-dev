#!/usr/bin/env python3
"""CLI entry point for the local Ollama + AI4I demo."""
import argparse
import json
from pathlib import Path
import sys

from agent import run_agent
from ml_tool import MLTool, load_observations
from ollama_client import OllamaClient, OllamaError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Plain-English question about the supplied observations")
    parser.add_argument("--observations", type=Path, help="One Kafka-shaped JSON observation or a list")
    parser.add_argument("--model", default="qwen3.5:9b", help="Local Ollama model (default: qwen3.5:9b)")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180, help="Seconds per Ollama request")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--trace", action="store_true", help="Print workflow steps to stderr")
    parser.add_argument("--json", action="store_true", help="Print the full structured response")
    parser.add_argument("--check", action="store_true", help="Check Ollama and load the saved ML pipeline")
    parser.add_argument("--require-prediction", action="store_true", help="Exit nonzero if no ML prediction completes")
    args = parser.parse_args()
    if not args.check and not args.prompt:
        parser.error("--prompt is required unless using --check")
    try:
        client = OllamaClient(args.model, args.host, args.timeout)
        info = client.check()
        print(info)
        tool = MLTool(load_observations(args.observations))
        if args.check:
            tool._load()
            print(json.dumps({"ollama": info, "model_id": tool.model_id,
                              "threshold": tool.metadata["threshold"], "features": tool.metadata["features"]}, indent=2))
            return 0
        trace = (lambda event: print(json.dumps(event), file=sys.stderr, flush=True)) if args.trace else None
        result = run_agent(args.prompt, tool, client, args.max_rounds, trace)
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            print(f"Workflow status: {result['status']}")
            if not result["raw_ml_outputs"]:
                print("No ML prediction was produced for this request.")
            print("\nLLM explanation\n" + result["explanation"])
            print("\nOriginal ML output (copied directly from the tool)")
            print(json.dumps(result["raw_ml_outputs"], indent=2, allow_nan=False))
            if result["tool_errors"]:
                print("\nTool errors\n" + json.dumps(result["tool_errors"], indent=2))
            if result["error"]:
                print("\nError: " + result["error"])
        failed = result["status"] in ("error", "tool_error", "partial")
        return 2 if failed or (args.require_prediction and not result["raw_ml_outputs"]) else 0
    except (ValueError, OSError, ImportError, OllamaError) as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc), "raw_ml_outputs": []}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
