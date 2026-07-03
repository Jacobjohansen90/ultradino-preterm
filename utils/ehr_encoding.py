#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json


def load_ehr_encodings(path):
    """Load patient encodings from JSON.

    Supported formats:
      {"patient_id": [0.1, 0.2, ...], ...}
      [{"id": "patient_id", "encoding": [...]}, ...]
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k): list(v) for k, v in data.items()}

    if isinstance(data, list):
        encodings = {}
        for entry in data:
            patient_id = entry.get("id") or entry.get("patient_id") or entry.get("CPR_BARN")
            vector = entry.get("encoding") or entry.get("embedding") or entry.get("vector")
            if patient_id is None or vector is None:
                raise ValueError(
                    "Each list entry must include an id and encoding vector"
                )
            encodings[str(patient_id)] = list(vector)
        return encodings

    raise ValueError(f"Unsupported EHR encoding JSON format in {path}")


def load_ehr_encodings_from_cfg(cfg):
    """Merge train/test encoding JSON files from config, if set."""
    encodings = {}
    for key in ("ehr_encoding_train_path", "ehr_encoding_test_path"):
        path = cfg.data.get(key)
        if path:
            encodings.update(load_ehr_encodings(path))
    return encodings
