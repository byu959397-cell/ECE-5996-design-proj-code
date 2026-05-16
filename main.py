#!/usr/bin/env python3
import argparse
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config
from src.ortho_hist_lora import OrthoHistManager
from src.trainers import ContinualTrainer, SequentialLoRATrainer, set_seed


def build_model_and_tokenizer(cfg):
    print(f"Loading {cfg.model_name}")

    tok = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=cfg.use_fast_tokenizer,
        padding_side=cfg.padding_side,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Training does not need KV cache. This also avoids warnings in some models.
    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, tok


def inject_peft_lora(model, cfg):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--mt_rank", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.method is not None:
        cfg.method = args.method
    if args.mt_rank is not None:
        cfg.mt_rank = args.mt_rank
    if args.rank is not None:
        cfg.rank = args.rank
    return cfg


def main():
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)

    pathlib.Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"method: {cfg.method}")
    print(f"output_dir: {cfg.output_dir}")
    print(f"tasks: {', '.join(cfg.train_tasks)}")

    model, tok = build_model_and_tokenizer(cfg)
    model.to(device)

    if cfg.method == "ortho_hist":
        manager = OrthoHistManager(
            model,
            target_modules=cfg.target_modules,
            rank=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            mt_rank=cfg.mt_rank,
            device=device,
        )
        manager.inject()
        ContinualTrainer(cfg, tok, model, manager, device).train_all_tasks()

    elif cfg.method == "sequential_lora":
        model = inject_peft_lora(model, cfg)
        model.to(device)
        SequentialLoRATrainer(cfg, tok, model, device).train_all_tasks()

    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    print(f"\nDone. Results in {cfg.output_dir}")


if __name__ == "__main__":
    main()