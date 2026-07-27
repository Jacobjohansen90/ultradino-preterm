#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
import torch


class TabularEhrModel(nn.Module):
    """Pass-through EHR model for static tabular features from the dataloader."""

    input_type = "tabular"

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        return x


class PatientIdEhrModel(nn.Module):
    """Look up fixed patient encodings by ID from a JSON-derived table."""

    input_type = "patient_id"

    def __init__(self, encodings, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.id_to_idx = {str(patient_id): i for i, patient_id in enumerate(encodings)}

        vectors = [encodings[patient_id] for patient_id in encodings]
        if not vectors:
            raise ValueError("EHR encoding table is empty")

        table = torch.tensor(vectors, dtype=torch.float32)
        if table.shape[1] != embed_dim:
            raise ValueError(
                f"Expected embedding dim {embed_dim}, got {table.shape[1]}"
            )

        self.register_buffer("table", table)

    def forward(self, patient_ids):
        device = self.table.device
        rows = []
        for patient_id in patient_ids:
            idx = self.id_to_idx.get(str(patient_id))
            if idx is None:
                rows.append(torch.zeros(self.embed_dim, device=device))
            else:
                rows.append(self.table[idx])

        return torch.stack(rows).unsqueeze(1)


class PatientLookupTabularEhrModel(nn.Module):
    """Look up patient encodings and prepend tabular EHR features."""

    input_type = "patient_id_tabular"

    def __init__(self, encodings, encoding_dim, tabular_dim):
        super().__init__()
        if tabular_dim <= 0:
            raise ValueError("tabular_dim must be > 0 for patient_lookup_tabular")

        self.lookup = PatientIdEhrModel(encodings, encoding_dim)
        self.tabular_dim = tabular_dim
        self.embed_dim = tabular_dim + encoding_dim

    def forward(self, ehr, patient_ids):
        lookup = self.lookup(patient_ids)
        if ehr.dim() == 2:
            ehr = ehr.unsqueeze(1)
        if ehr.shape[-1] != self.tabular_dim:
            raise ValueError(
                f"Expected {self.tabular_dim} tabular EHR features, got {ehr.shape[-1]}"
            )
        return torch.cat([ehr, lookup], dim=-1)
