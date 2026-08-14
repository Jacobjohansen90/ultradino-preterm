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
            patient_id = entry.get("id") or entry.get("patient_id") or entry.get("CPR_CHILD")
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


def log_encoding_id_coverage(
    id_to_idx,
    dataframes,
    split_names,
    id_column="CPR_CHILD",
    max_examples=20,
):
    """Print how many population IDs match the patient-encoding lookup table."""
    encoding_ids = set(id_to_idx)

    for split_name, df in zip(split_names, dataframes):
        if id_column not in df.columns:
            print(f"[{split_name}] Skipping encoding ID check: missing '{id_column}'")
            continue

        data_ids = {str(patient_id) for patient_id in df[id_column].unique().to_list()}
        matched = data_ids & encoding_ids
        missing = sorted(data_ids - encoding_ids)
        n_data = len(data_ids)
        n_matched = len(matched)
        pct = 100.0 * n_matched / n_data if n_data else 0.0

        print(
            f"[{split_name}] EHR encoding ID coverage: "
            f"{n_matched}/{n_data} unique {id_column} ({pct:.1f}%)"
        )
        if missing:
            print(
                f"[{split_name}] Missing from encoding JSON "
                f"({len(missing)} total), e.g.: {missing[:max_examples]}"
            )
        if n_matched == 0 and n_data > 0:
            print(
                f"[{split_name}] WARNING: no {id_column} values matched the encoding JSON. "
                "Check ID formatting (str/int, leading zeros, etc.)."
            )
