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
