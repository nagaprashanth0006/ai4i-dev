#!/usr/bin/env python3
"""Live Ollama test. Checks actual tool execution and exact saved-model results."""
import argparse
from pathlib import Path
import json
from agent import run_agent
from ml_tool import MLTool, load_observations
from ollama_client import OllamaClient

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5:9b")
    args = parser.parse_args()
    client = OllamaClient(args.model)
    client.check()
    cases = [
        ("single.json", "Assess sample-a using the ML model. Explain its score and cutoff simply.", {"sample-a"}),
        ("compare.json", "Run the model on both sample-a and sample-b. Which gets the higher score?", {"sample-a", "sample-b"}),
        ("missing.json", "Assess missing-sensor with the model and explain any warnings.", {"missing-sensor"}),
        ("invalid.json", "Assess bad-sensor using the model. Explain any input error.", set()),
        (None, "Is machine M-017 currently showing a failure?", set()),
    ]
    for filename, prompt, expected_ids in cases:
        tool = MLTool(load_observations(HERE / "examples" / filename if filename else None))
        result = run_agent(prompt, tool, client)
        actual_ids = {item["observation_id"] for item in result["raw_ml_outputs"]}
        if actual_ids != expected_ids:
            raise AssertionError(f"{filename}: expected {expected_ids}, got {actual_ids}: {result}")
        if expected_ids and result["status"] != "predicted":
            raise AssertionError(result)
        if filename == "invalid.json" and result["status"] != "tool_error":
            raise AssertionError(result)
        if filename is None and result["status"] != "no_prediction":
            raise AssertionError(result)
        # A separate adapter instance provides expected direct-inference output.
        direct = MLTool(tool.observations)
        for raw in result["raw_ml_outputs"]:
            if raw != direct.predict({"observation_id": raw["observation_id"]}):
                raise AssertionError("Agent result differs from direct model output")
        print(json.dumps({"case": filename or "no-data", "status": result["status"],
                          "explanation": result["explanation"], "raw_ml_outputs": result["raw_ml_outputs"]}), flush=True)
    print("PASS: all five live scenarios. Review the explanations for wording as well.")


if __name__ == "__main__":
    main()
