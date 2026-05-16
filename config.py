from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

# 必须加类型注解!
@dataclass
class TrainConfig:
    # paths and model
    seed: int = 42
    output_dir: str = "./outputs/run"
    trace_root: str = "./TRACE"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    use_fast_tokenizer: bool = False
    padding_side: str = "left"
    max_length: int = 1024

    # training
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    bf16: bool = True
    compile_model: bool = False

    # logging and checkpointing
    log_every: int = 10
    eval_every_epoch: bool = False
    save_every_task: bool = True
    debug_eval: bool = False

    # method: "ortho_hist" | "sequential_lora"
    method: str = "ortho_hist"

    # LoRA
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # OrthoHist-specific
    mt_rank: int = 8

    # task schedule
    train_tasks: List[str] = field(default_factory=lambda: [
        "C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds",
    ])
    eval_tasks: List[str] = field(default_factory=lambda: [
        "C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds",
    ])


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    valid_keys = set(TrainConfig.__dataclass_fields__.keys())
    unknown = sorted(set(raw.keys()) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")

    cfg = TrainConfig(**raw)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg
