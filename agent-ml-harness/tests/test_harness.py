import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import run_agent
from ml_tool import MLTool, load_observations, strict_json
from ollama_client import OllamaClient, OllamaError

BASE = {"air_temperature": 300.1, "process_temperature": 310.1, "rpm": 1500,
        "torque": 40.0, "tool_wear": 100, "vibration_rms": 3.0, "type": "L"}


def call(name="predict_machine_failure", arguments=None):
    return {"role": "assistant", "tool_calls": [{"function": {
        "name": name, "arguments": {"observation_id": "sample"} if arguments is None else arguments}}]}


class FakeClient:
    model = "test-model"
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
    def chat(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeTool:
    def __init__(self):
        self.calls = 0
    def catalog(self):
        return [{"observation_id": "sample", "machine_id": "M-001"}]
    def schema(self):
        return {"type": "function"}
    def predict(self, arguments):
        self.calls += 1
        if arguments != {"observation_id": "sample"}:
            raise ValueError("Invalid arguments")
        return {"observation_id": "sample", "p_failure": 0.123456789,
                "prediction": 0, "label": "NORMAL", "threshold": 0.13}


class WorkflowTests(unittest.TestCase):
    def test_real_tool_result_survives_llm_explanation(self):
        client = FakeClient([call(), {"role": "assistant", "content": "Here is my explanation."}])
        result = run_agent("Assess sample", FakeTool(), client)
        self.assertEqual(result["status"], "predicted")
        self.assertEqual(result["raw_ml_outputs"][0]["p_failure"], 0.123456789)
        self.assertEqual(client.requests[1][-1]["role"], "tool")
        self.assertIn('"ok": true', client.requests[1][-1]["content"])

    def test_no_fake_prediction_when_llm_skips_tool(self):
        tool = FakeTool()
        client = FakeClient([{"role": "assistant", "content": "NORMAL"}])
        result = run_agent("Assess sample", tool, client)
        self.assertEqual(result["status"], "no_prediction")
        self.assertEqual(result["raw_ml_outputs"], [])
        self.assertEqual(tool.calls, 0)

    def test_unknown_tool_never_executes(self):
        tool = FakeTool()
        client = FakeClient([call("run_shell"), {"role": "assistant", "content": "Tool failed."}])
        result = run_agent("Assess", tool, client)
        self.assertEqual(tool.calls, 0)
        self.assertEqual(result["status"], "tool_error")

    def test_sensor_edits_rejected(self):
        client = FakeClient([call(arguments={"observation_id": "sample", "rpm": 20}),
                             {"role": "assistant", "content": "Cannot modify readings."}])
        result = run_agent("Assess", FakeTool(), client)
        self.assertEqual(result["raw_ml_outputs"], [])
        self.assertEqual(result["status"], "tool_error")

    def test_explanation_failure_preserves_result(self):
        client = FakeClient([call(), OllamaError("timeout")])
        result = run_agent("Assess", FakeTool(), client)
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["raw_ml_outputs"]), 1)

    def test_loop_limit(self):
        result = run_agent("Assess", FakeTool(), FakeClient([call(), call()]), max_rounds=2)
        self.assertEqual(result["error"], "round_limit")
        self.assertEqual(len(result["raw_ml_outputs"]), 1)

    def test_multiple_tool_calls(self):
        tool = FakeTool()
        message = call()
        message["tool_calls"] += call()["tool_calls"]
        result = run_agent("Assess", tool, FakeClient([message, {"role": "assistant", "content": "Done"}]))
        self.assertEqual(tool.calls, 2)
        self.assertEqual(len(result["raw_ml_outputs"]), 1)

    def test_malformed_function_is_a_tool_error(self):
        client = FakeClient([{"role": "assistant", "tool_calls": [{"function": "broken"}]},
                             {"role": "assistant", "content": "Could not call tool."}])
        result = run_agent("Assess", FakeTool(), client)
        self.assertEqual(result["status"], "tool_error")

    def test_missing_content_is_a_clear_error(self):
        result = run_agent("Assess", FakeTool(), FakeClient([{"role": "assistant", "content": None}]))
        self.assertEqual(result["error"], "empty_response")

    def test_nonlocal_host_rejected(self):
        with self.assertRaises(ValueError):
            OllamaClient(host="https://example.com")

    def test_strict_json(self):
        for text in ['{"rpm": NaN}', '{"rpm": 1, "rpm": 2}']:
            with self.assertRaises(ValueError):
                strict_json(text)

    def test_duplicate_ids_rejected_and_labels_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            event = {"observation_id": "a", "features": BASE, "actual": 1}
            path.write_text(json.dumps(event))
            self.assertNotIn("actual", load_observations(path)["a"])
            path.write_text(json.dumps([event, event]))
            with self.assertRaises(ValueError):
                load_observations(path)


class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = MLTool({"sample": {"machine_id": "M-001", "features": copy.deepcopy(BASE)}})
        cls.tool._load()

    def make_tool(self, features):
        tool = MLTool({"sample": {"machine_id": "M-001", "features": features}})
        tool.model, tool.predict_event, tool.model_id = self.tool.model, self.tool.predict_event, self.tool.model_id
        return tool

    def test_exact_kafka_parity(self):
        result = self.tool.predict({"observation_id": "sample"})
        expected = self.tool.predict_event({"features": BASE}, self.tool.model,
                                          self.tool.metadata, self.tool.metadata["threshold"])
        self.assertEqual((result["p_failure"], result["prediction"]), expected[:2])
        self.assertEqual(result["prediction"], int(result["p_failure"] >= result["threshold"]))

    def test_missing_and_unknown_values_are_flagged(self):
        data = dict(BASE, torque=None, type="UNKNOWN")
        result = self.make_tool(data).predict({"observation_id": "sample"})
        self.assertIn("torque missing/imputed", result["notices"])
        self.assertIn("type unknown category", result["notices"])

    def test_bad_readings_rejected(self):
        for value in [True, "fast", float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_tool(dict(BASE, rpm=value)).predict({"observation_id": "sample"})
        with self.assertRaises(ValueError):
            self.make_tool({"type": "L"}).predict({"observation_id": "sample"})

    def test_leaky_features_do_not_change_result(self):
        result = self.make_tool(dict(BASE, machine_failure=1, post_failure_alarm=1)).predict({"observation_id": "sample"})
        self.assertEqual(result["p_failure"], self.tool.predict({"observation_id": "sample"})["p_failure"])
        self.assertEqual(result["ignored_fields"], ["machine_failure", "post_failure_alarm"])

    def test_invalid_id_and_threshold_override_rejected(self):
        for args in [{"observation_id": "invented"}, {"observation_id": "sample", "threshold": 0.9}]:
            with self.assertRaises(ValueError):
                self.tool.predict(args)


if __name__ == "__main__":
    unittest.main()
