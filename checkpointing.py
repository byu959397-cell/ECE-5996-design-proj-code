from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _save_json_atomic(payload: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def save_checkpoint(
    output_dir,
    task_id,
    model,
    extra: Optional[Dict[str, Any]] = None,
):
    ckpt_dir = Path(output_dir) / "checkpoints" / f"task_{task_id:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save the full model state. For OrthoHist-LoRA this includes the frozen
    # historical basis_pool; for Sequential LoRA it includes the PEFT LoRA state.
    state = model.state_dict()
    tmp_path = ckpt_dir / "model_state.pt.tmp"
    final_path = ckpt_dir / "model_state.pt"
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)

    if extra:
        _save_json_atomic(extra, ckpt_dir / "meta.json")


def load_checkpoint(output_dir, task_id, model):
    ckpt_dir = Path(output_dir) / "checkpoints" / f"task_{task_id:02d}"
    ckpt_path = ckpt_dir / "model_state.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    if unexpected:
        print(f"[Checkpoint WARNING] unexpected keys while loading {ckpt_path}:")
        for key in unexpected[:20]:
            print(f"  - {key}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")

    if missing:
        print(f"[Checkpoint WARNING] missing keys while loading {ckpt_path}:")
        for key in missing[:20]:
            print(f"  - {key}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    return incompatible
