#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

import polars as pl

logger = logging.getLogger(__name__)


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


def _log_coverage_stats(
    split_name,
    n_matched,
    n_data,
    missing,
    source,
    id_column,
    max_examples=20,
):
    pct = 100.0 * n_matched / n_data if n_data else 0.0
    logger.info(
        "[%s] %s ID coverage: %d/%d unique %s (%.1f%%)",
        split_name,
        source,
        n_matched,
        n_data,
        id_column,
        pct,
    )
    if missing:
        logger.warning(
            "[%s] Missing from %s (%d total), e.g.: %s",
            split_name,
            source,
            len(missing),
            missing[:max_examples],
        )
    if n_matched == 0 and n_data > 0:
        logger.warning(
            "[%s] No %s values matched %s. "
            "Check ID formatting (str/int, leading zeros, etc.).",
            split_name,
            id_column,
            source,
        )


def log_encoding_id_coverage(
    id_to_idx,
    dataframes,
    split_names,
    id_column="CPR_CHILD",
    max_examples=20,
):
    """Log how many population IDs match the patient-encoding lookup table."""
    encoding_ids = set(id_to_idx)

    for split_name, df in zip(split_names, dataframes):
        if id_column not in df.columns:
            logger.warning(
                "[%s] Skipping encoding ID check: missing '%s'",
                split_name,
                id_column,
            )
            continue

        data_ids = {str(patient_id) for patient_id in df[id_column].unique().to_list()}
        missing = sorted(data_ids - encoding_ids)
        n_data = len(data_ids)
        n_matched = n_data - len(missing)
        _log_coverage_stats(
            split_name,
            n_matched,
            n_data,
            missing,
            "encoding JSON",
            id_column,
            max_examples=max_examples,
        )


def log_tabular_ehr_coverage(
    dataframes,
    split_names,
    ehr_cols,
    id_column="CPR_CHILD",
    max_examples=20,
):
    """Log how many unique IDs have non-null tabular EHR features after the join."""
    ehr_cols = list(ehr_cols or [])
    if not ehr_cols:
        logger.info("Skipping tabular EHR ID check: data.ehr_data is empty")
        return

    for split_name, df in zip(split_names, dataframes):
        if id_column not in df.columns:
            logger.warning(
                "[%s] Skipping tabular EHR ID check: missing '%s'",
                split_name,
                id_column,
            )
            continue

        missing_cols = [col for col in ehr_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                "[%s] Tabular EHR columns missing from dataframe: %s",
                split_name,
                missing_cols,
            )
            continue

        data_ids = {str(patient_id) for patient_id in df[id_column].unique().to_list()}
        unmatched = df.filter(
            pl.any_horizontal([pl.col(col).is_null() for col in ehr_cols])
        )
        missing = sorted(
            {str(patient_id) for patient_id in unmatched[id_column].unique().to_list()}
        )
        n_data = len(data_ids)
        n_matched = n_data - len(missing)
        _log_coverage_stats(
            split_name,
            n_matched,
            n_data,
            missing,
            "tabular EHR",
            id_column,
            max_examples=max_examples,
        )
