"""Read supplied observations and call the SAME inference function as Kafka."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def strict_json(text):
    def invalid(value):
        raise ValueError(f"Invalid JSON number: {value}; use null for missing readings")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=invalid, object_pairs_hook=unique)


def load_observations(path):
    """Accept one Kafka-shaped event or a list. Never take sensor values from an LLM."""
    if path is None:
        return {}
    data = strict_json(Path(path).read_text())
    events = data if isinstance(data, list) else [data]
    if len(events) > 20:
        raise ValueError("Use at most 20 observations per request")
    result = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get("features"), dict):
            raise ValueError("Each observation needs a 'features' JSON object")
        key = event.get("observation_id", f"observation-{index + 1}")
        if not isinstance(key, str) or not key.strip() or len(key) > 100:
            raise ValueError("observation_id must be a nonempty string of at most 100 characters")
        if key in result:
            raise ValueError(f"Duplicate observation_id: {key}")
        machine = event.get("machine_id", key)
        if not isinstance(machine, str) or len(machine) > 100:
            raise ValueError("machine_id must be a string of at most 100 characters")
        # Discard actual labels, timestamps and other top-level context.
        result[key] = {"machine_id": machine, "features": event["features"]}
    return result


class MLTool:
    def __init__(self, observations, model_path=ARTIFACTS / "machine_failure_pipeline.joblib",
                 metadata_path=ARTIFACTS / "machine_failure_metadata.json"):
        self.observations = observations
        self.model_path = Path(model_path)
        self.metadata = strict_json(Path(metadata_path).read_text())
        self.model = None
        self.predict_event = None
        self.cache = {}

    def catalog(self):
        return [{"observation_id": key, "machine_id": event["machine_id"]}
                for key, event in self.observations.items()]

    def schema(self):
        return {"type": "function", "function": {
            "name": "predict_machine_failure",
            "description": ("Run the saved ML pipeline on one supplied observation. "
                            "Use it to assess current machine failure, get a score, or compare observations. "
                            "The app holds the original readings. Pass only their observation_id. "
                            "It cannot forecast future failures or explain a root cause."),
            "parameters": {"type": "object", "properties": {
                "observation_id": {"type": "string", "enum": list(self.observations),
                                   "description": "An exact ID from the supplied observation list"}},
                "required": ["observation_id"], "additionalProperties": False}}}

    def _load(self):
        import joblib
        import sklearn
        from sklearn.exceptions import InconsistentVersionWarning
        expected = self.metadata["versions"]["sklearn"]
        if sklearn.__version__ != expected:
            raise ValueError(f"Model needs scikit-learn {expected}; installed {sklearn.__version__}. "
                             "Use this folder's requirements.txt or retrain and export matching artifacts.")
        # Only load the trusted local project artifact. Joblib is not an untrusted interchange format.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(self.model_path)
        if list(model.feature_names_in_) != self.metadata["features"]:
            raise ValueError("Model and metadata feature lists differ")
        threshold = self.metadata["threshold"]
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("Saved threshold must be between 0 and 1")
        source = ROOT / "src" / "consumer.py"
        spec = importlib.util.spec_from_file_location("ai4i_kafka_consumer", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model.named_steps["model"].n_jobs = 1
        self.model = model
        self.predict_event = module.predict_event
        self.model_id = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:16]

    def predict(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {"observation_id"}:
            raise ValueError("Tool accepts only observation_id; sensor edits and threshold overrides are not allowed")
        key = arguments["observation_id"]
        if not isinstance(key, str) or key not in self.observations:
            raise ValueError("Unknown observation_id. Choose an ID from the supplied list")
        if key in self.cache:
            return self.cache[key]
        if self.model is None:
            self._load()
        event = self.observations[key]
        # Consumer allowlists metadata features. It handles nulls and unknown categories.
        # Never pass actual into inference or expose it to the orchestrator.
        probability, prediction, _, notices = self.predict_event(
            {"features": event["features"]}, self.model, self.metadata, self.metadata["threshold"])
        ignored = sorted(set(event["features"]) - set(self.metadata["features"]))
        result = {"observation_id": key, "machine_id": event["machine_id"],
                  "p_failure": probability, "prediction": prediction,
                  "label": "FAILURE" if prediction else "NORMAL",
                  "threshold": self.metadata["threshold"], "notices": notices,
                  "ignored_fields": ignored, "model_id": self.model_id,
                  "scope": self.metadata["scope"]}
        self.cache[key] = result
        return result
