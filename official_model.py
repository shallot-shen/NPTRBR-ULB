"""Official TabM wrapper and public train-constant feature filtering."""
from dataclasses import replace

import numpy as np
import tabm
import torch
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

from mecg_star_tabm import _engine as engine
from mecg_star_tabm.model import MultiEvidenceConsensusGate


class OfficialTransfer(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        if tabm.__version__ != "0.0.3":
            raise RuntimeError("Install tabm==0.0.3")
        self.cards = c.categorical_cardinalities
        self.network = tabm.TabM.make(
            n_num_features=c.n_numeric, cat_cardinalities=self.cards,
            num_embeddings=PiecewiseLinearEmbeddings(
                [torch.tensor(b, dtype=torch.float32) for b in c.numeric_bin_edges],
                d_embedding=c.numeric_embedding_dim, activation=False, version="B"),
            d_out=1, k=c.members, n_blocks=c.n_blocks, d_block=c.hidden_dim,
            dropout=c.dropout, arch_type="tabm")
        self.mecg_gate = MultiEvidenceConsensusGate(
            c.n_numeric, len(self.cards), c.mecg_group_ids, c.mecg_feature_prior,
            c.mecg_group_prior, c.mecg_numeric_redundancy, c.mecg_temperature,
            c.mecg_stretch_low, c.mecg_stretch_high, c.mecg_group_gates,
            c.mecg_stochastic_gates, not c.mecg_enabled, c.mecg_target_active_ratio)
        self.register_buffer("widths", torch.tensor([c.numeric_embedding_dim]*c.n_numeric + list(self.cards)))
        self.register_buffer("field_index", torch.repeat_interleave(torch.arange(len(self.widths)), self.widths))
        self.field = None
        self.network.backbone.register_forward_pre_hook(self.apply_gate)

    def apply_gate(self, module, inputs):
        if self.field is None:
            return None
        return (inputs[0] * self.field[self.field_index],)

    def forward(self, xn, xc, context, history, return_masks=False, return_components=False):
        gate, reg = self.mecg_gate()
        self.field = None if self.mecg_gate.identity else gate
        try:
            logits = self.network(xn, xc if self.cards else None).squeeze(-1)
        finally:
            self.field = None
        result = (logits, logits.new_zeros(()), None)
        return (*result, reg) if return_components else result

class InputView:
    """Drop train-constant numeric fields on either dataset, then freeze."""
    def __init__(self, base):
        self.base = base

    def __getattr__(self, key):
        if key == "base":
            raise AttributeError(key)
        return getattr(self.base, key)

    def pack(self, a):
        n = len(a.x_numeric)
        return replace(a, x_numeric=np.ascontiguousarray(a.x_numeric[:,self.keep]),
                       x_context=np.zeros((n,0),np.float32),
                       x_history=np.zeros((n,0),np.float32),
                       matched_key=np.zeros(n,np.int64), deviation_bin=np.zeros(n,np.int64))

    def fit_transform(self, frame):
        a = self.base.fit_transform(frame)
        self.keep = np.flatnonzero(np.ptp(a.x_numeric.astype(np.float64),axis=0)>1e-8)
        if not self.keep.size:
            raise ValueError("No variable numeric inputs")
        self.numeric_columns = [self.base.numeric_columns[i] for i in self.keep]
        self.numeric_bin_edges = [self.base.numeric_bin_edges[i] for i in self.keep]
        self.context_columns = []
        # The inherited legacy guard checks this list even with FAM disabled.
        # All fields qualify for that guard; the official model does not use FAM.
        engine.QUICK_CHECK_NUMERIC = self.numeric_columns
        return self.pack(a)

    def transform(self, frame):
        return self.pack(self.base.transform(frame))
