#!/usr/bin/env python3
"""Create demo inputs from the SAME held-out split used by the Kafka producer."""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_tool import ROOT, ARTIFACTS


def main():
    metadata = json.loads((ARTIFACTS / "machine_failure_metadata.json").read_text())
    frame = pd.read_csv(ROOT / "data" / "processed" / "ai4i_plus.csv")
    target, seed = metadata["target"], metadata["seed"]
    _, holdout = train_test_split(frame, test_size=0.4, stratify=frame[target], random_state=seed)
    _, test = train_test_split(holdout, test_size=0.5, stratify=holdout[target], random_state=seed)
    examples = Path(__file__).resolve().parent / "examples"
    examples.mkdir(exist_ok=True)
    events = []
    # Choose by known label ONLY to illustrate two situations, never to estimate quality.
    for label, name in [(0, "sample-a"), (1, "sample-b")]:
        row = test[test[target] == label].iloc[0]
        features = {}
        for feature in metadata["features"]:
            value = row[feature]
            features[feature] = None if pd.isna(value) else str(value) if feature == "type" else float(value)
        events.append({"observation_id": name, "machine_id": str(row["machine_id"]), "features": features})
    def save(name, value):
        (examples / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    save("single.json", events[0])
    save("compare.json", events)
    missing = json.loads(json.dumps(events[0]))
    missing["observation_id"] = "missing-sensor"
    missing["features"]["torque"] = None
    save("missing.json", missing)
    invalid = json.loads(json.dumps(events[0]))
    invalid["observation_id"] = "bad-sensor"
    invalid["features"]["rpm"] = "fast"
    save("invalid.json", invalid)
    print("Created four example files from held-out rows. Outcomes are not included in the requests.")


if __name__ == "__main__":
    main()
