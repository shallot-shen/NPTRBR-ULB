"""Atomic epoch-boundary recovery for this user's locally trained runs only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

FORMAT_VERSION = 1
STATE_FIELDS = (
    "best_smoothed_ap", "best_epoch", "top_checkpoints", "validation_ap_window",
    "patience_counter", "hard_bank", "topk_state", "topk_threshold_logits",
    "topk_adaptive_weights", "topk_state_history", "nptrbr_state",
    "nptrbr_threshold_logits", "nptrbr_fpr_duals", "nptrbr_kl_dual",
    "nptrbr_state_history", "nptrbr_dual_history", "history",
)


def source_fingerprint():
    root = Path(__file__).parent
    names = ("_engine.py", "cli.py", "model.py", "losses.py", "sampling.py",
             "preprocessing.py", "evidence.py", "data.py", "training_state.py")
    return {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def ensure_finite(name, value, *, epoch=None, batch=None):
    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(
            f"Non-finite {name}; epoch={epoch}, batch={batch}. "
            "No parameter/dual update is permitted from this batch; "
            "resume from the last completed epoch after diagnosing the cause."
        )


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def save_epoch_checkpoint(path, namespace):
    """Write only after validation, selection and hard-bank refresh finish."""
    path = Path(path)
    request = json.loads((path.parent/"run_request.json").read_text(encoding="utf-8"))
    teacher = namespace["nptrbr_teacher"]
    payload = {
        "format_version": FORMAT_VERSION,
        "request_sha256": request["request_sha256"],
        "source_sha256": source_fingerprint(),
        "epoch": namespace["epoch_number"],
        "elapsed_seconds": time.time()-namespace["start_time"],
        "model": _cpu(namespace["model"].state_dict()),
        "optimizer": _cpu(namespace["optimizer"].state_dict()),
        "scheduler": namespace["scheduler"].state_dict(),
        "grad_scaler": namespace["grad_scaler"].state_dict(),
        "sampler": namespace["sampler_engine"].recovery_state(),
        "teacher": None if teacher is None else _cpu(teacher.state_dict()),
        "training": {k: _cpu(namespace[k]) for k in STATE_FIELDS},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    for name, tensor in payload["model"].items():
        ensure_finite("checkpoint parameter "+name, tensor, epoch=payload["epoch"])
    fd, temporary = tempfile.mkstemp(prefix="epoch_state_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_epoch_checkpoint(path, expected_request):
    # This is the user's own local checkpoint, never an external pretrained file.
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported epoch checkpoint format")
    if state.get("request_sha256") != expected_request:
        raise ValueError("Epoch checkpoint recipe/data/device does not match this request")
    if state.get("source_sha256") != source_fingerprint():
        raise ValueError("Training code changed since checkpoint; do not mix execution versions")
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA checkpoint cannot be resumed as a CPU run")
        torch.cuda.set_rng_state_all(state["cuda"])
