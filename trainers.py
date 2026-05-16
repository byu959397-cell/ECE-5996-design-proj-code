import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .analysis_utils import collect_reuse_matrix, save_reuse_heatmap, save_score_matrix
from .checkpointing import load_checkpoint, save_checkpoint
from .data_trace import build_task_loaders
from .eval_trace import (
    average_accuracy,
    backward_transfer,
    build_debug_rows,
    debug_parse_examples,
    forward_transfer,
    score_task,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_optimizer_and_scheduler(params, cfg, n_steps):
    opt = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup = int(cfg.warmup_ratio * n_steps)
    sch = get_linear_schedule_with_warmup(opt, warmup, n_steps)
    return opt, sch


def _find_last_done_task(output_dir, n_tasks):
    """Return the last consecutively completed task id."""
    last_done = 0
    ckpt_root = Path(output_dir) / "checkpoints"

    for tid in range(1, n_tasks + 1):
        ckpt_dir = ckpt_root / f"task_{tid:02d}"
        done_flag = ckpt_dir / "DONE"
        state_file = ckpt_dir / "model_state.pt"

        if done_flag.exists() and state_file.exists():
            last_done = tid
        else:
            break

    return last_done


class JsonLogger:
    def __init__(self, output_dir):
        p = Path(output_dir) / "logs"
        p.mkdir(parents=True, exist_ok=True)
        self.path = p / "events.jsonl"
        self.path.touch(exist_ok=True)

    def log(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ContinualTrainer:
    def __init__(self, cfg, tokenizer, model, manager, device):
        self.cfg = cfg
        self.tok = tokenizer
        self.model = model
        self.manager = manager
        self.device = device
        self.logger = JsonLogger(cfg.output_dir)

        self.loaders = build_task_loaders(
            trace_root=cfg.trace_root,
            tasks=cfg.train_tasks,
            tokenizer=tokenizer,
            batch_size=cfg.per_device_batch_size,
            max_length=cfg.max_length,
        )

        self.score_matrix = {}
        self.zero_shot_scores = {}
        self.reuse_snapshots = []
        self.history = []

    def _save_history(self):
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _save_score_matrix(self):
        save_score_matrix(
            self.score_matrix,
            self.cfg.train_tasks,
            self.cfg.output_dir,
        )

        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "score_matrix_raw.json", "w", encoding="utf-8") as f:
            json.dump(self.score_matrix, f, indent=2, ensure_ascii=False)

    def _save_reuse_snapshots(self):
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "reuse_snapshots.json", "w", encoding="utf-8") as f:
            json.dump(self.reuse_snapshots, f, indent=2, ensure_ascii=False)

    def _restore_history_and_scores(self):
        out = Path(self.cfg.output_dir)

        hp = out / "history.json"
        if hp.exists():
            with open(hp, encoding="utf-8") as f:
                self.history = json.load(f)

        smp = out / "score_matrix_raw.json"
        if smp.exists():
            with open(smp, encoding="utf-8") as f:
                self.score_matrix = json.load(f)
            print(f"[Resume] restored score_matrix from {smp}")
        else:
            print("[Resume WARNING] score_matrix_raw.json not found; rebuilding from history.")
            self.score_matrix = {}
            for entry in self.history:
                key = entry.get("task_name") or entry.get("task")
                if key is not None:
                    self.score_matrix[key] = dict(entry.get("seen_scores", {}))

    def _restore_reuse_snapshots(self):
        snap_path = Path(self.cfg.output_dir) / "reuse_snapshots.json"
        if snap_path.exists():
            with open(snap_path, encoding="utf-8") as f:
                self.reuse_snapshots = json.load(f)
            print(f"[Resume] restored {len(self.reuse_snapshots)} reuse snapshots")

    def _preallocate_orthohist_pool_for_resume(self, n_committed_tasks):
        """Create basis_pool slots before loading an OrthoHist checkpoint."""
        if n_committed_tasks <= 0:
            return

        for layer in self.manager.layers.values():
            layer.basis_pool = torch.nn.ParameterList()
            device = layer.lora_B.device
            dtype = layer.lora_B.dtype

            for _ in range(n_committed_tasks):
                basis = torch.empty(
                    layer.out_features,
                    layer.rank,
                    device=device,
                    dtype=dtype,
                )
                coeff = torch.empty(
                    layer.rank,
                    layer.in_features,
                    device=device,
                    dtype=dtype,
                )
                layer.basis_pool.append(torch.nn.Parameter(basis, requires_grad=False))
                layer.basis_pool.append(torch.nn.Parameter(coeff, requires_grad=False))

            layer._n_pool = n_committed_tasks

    def _mark_done(self, task_id):
        done_flag = (
            Path(self.cfg.output_dir)
            / "checkpoints"
            / f"task_{task_id:02d}"
            / "DONE"
        )
        done_flag.parent.mkdir(parents=True, exist_ok=True)
        done_flag.touch()

    def _train_one_epoch(self, task_name, optimizer, scheduler, epoch):
        self.model.train()
        loader = self.loaders[task_name]["train"]
        total_loss = 0.0
        batch_cnt = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"[{task_name}] epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, 1):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / self.cfg.grad_accum_steps
            loss.backward()

            if step % self.cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.manager.trainable_parameters(), self.cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(out.loss.item())
            batch_cnt += 1
            if batch_cnt % self.cfg.log_every == 0:
                pbar.set_postfix(loss=f"{total_loss / batch_cnt:.4f}")

        if len(loader) % self.cfg.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.manager.trainable_parameters(), self.cfg.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return total_loss / max(1, batch_cnt)

    @torch.no_grad()
    def _generate(self, task_name, split="test"):
        self.model.eval()
        loader = self.loaders[task_name][split]
        preds, refs, prompts = [], [], []

        for batch in tqdm(loader, desc=f"eval {task_name}/{split}", leave=False):
            prompt_texts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch["prompt_text"]
            ]

            enc = self.tok(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                add_special_tokens=False,
            ).to(self.device)

            if task_name in {"C-STANCE", "FOMC"}:
                max_new_tokens = 8
            elif task_name == "ScienceQA":
                max_new_tokens = 64
            else:
                max_new_tokens = 128

            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
                do_sample=False,
            )

            input_len = enc["input_ids"].shape[1]
            for i in range(out.size(0)):
                pred = self.tok.decode(out[i][input_len:], skip_special_tokens=True)
                preds.append(pred.strip())

            refs.extend(batch["answer_text"])
            prompts.extend(batch["prompt_text"])

        if getattr(self.cfg, "debug_eval", False):
            debug_parse_examples(task_name, preds, refs, n=10)

        return {"preds": preds, "refs": refs, "prompts": prompts}

    def _save_raw_predictions(self, row_key, task_name, split, gen_results):
        debug_dir = Path(self.cfg.output_dir) / "debug_predictions"
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_row = str(row_key).replace("/", "_")
        safe_task = str(task_name).replace("/", "_")
        out_path = debug_dir / f"{safe_row}__{safe_task}__{split}.jsonl"

        rows = build_debug_rows(
            task_name=task_name,
            preds=gen_results["preds"],
            refs=gen_results["refs"],
            prompts=gen_results["prompts"],
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _eval_tasks(self, task_list, row_key, split="test"):
        self.score_matrix.setdefault(row_key, {})
        task_scores = {}

        for task in task_list:
            gen_results = self._generate(task, split=split)
            self._save_raw_predictions(
                row_key=row_key,
                task_name=task,
                split=split,
                gen_results=gen_results,
            )

            res = score_task(
                task,
                gen_results["preds"],
                gen_results["refs"],
                gen_results["prompts"],
            )

            task_scores[task] = res.metric_value
            self.score_matrix[row_key][task] = res.metric_value

            self.logger.log({
                "event": "task_eval",
                "row": row_key,
                "task": task,
                "split": split,
                "metric": res.metric_name,
                "value": res.metric_value,
                "num_examples": res.num_examples,
            })

        return task_scores

    def train_one_task(self, task_name, task_id, seen):
        self.manager.prepare_for_task(task_id)

        n_steps = max(
            1,
            math.ceil(
                len(self.loaders[task_name]["train"]) * self.cfg.num_epochs
                / self.cfg.grad_accum_steps
            ),
        )
        opt, sch = _make_optimizer_and_scheduler(
            self.manager.trainable_parameters(), self.cfg, n_steps
        )

        self.logger.log({
            "event": "task_start",
            "task": task_name,
            "task_id": task_id,
            "trainable_params": sum(p.numel() for p in self.manager.trainable_parameters()),
        })

        for epoch in range(1, self.cfg.num_epochs + 1):
            loss = self._train_one_epoch(task_name, opt, sch, epoch)
            self.logger.log({
                "event": "epoch_end",
                "task": task_name,
                "epoch": epoch,
                "loss": loss,
            })

        print(f"[{task_name}] training done, running eval...")
        self._eval_tasks(self.cfg.train_tasks, row_key=task_name, split="test")

        if hasattr(self.manager, "collect_reuse_analysis"):
            snapshot = self.manager.collect_reuse_analysis(task_name)
            self.reuse_snapshots.append(snapshot)
            self._save_reuse_snapshots()

        self.manager.commit_task()

        scores = {t: self.score_matrix[task_name][t] for t in seen}
        aa = average_accuracy(scores)
        bwt = backward_transfer(self.score_matrix, seen)
        fwt = forward_transfer(self.zero_shot_scores, self.score_matrix, seen)

        self.history.append({
            "task_id": task_id,
            "task_name": task_name,
            "seen_scores": scores,
            "average_accuracy": aa,
            "bwt": bwt,
            "fwt": fwt,
        })

        self._save_history()
        self._save_score_matrix()

        self.logger.log({
            "event": "task_end",
            "task": task_name,
            "aa": aa,
            "bwt": bwt,
            "fwt": fwt,
        })
        print(f"[{task_name}] AA={aa:.4f}  BWT={bwt:.4f}  FWT={fwt:.4f}")

        if self.cfg.save_every_task:
            save_checkpoint(
                self.cfg.output_dir,
                task_id,
                self.model,
                extra={
                    "task": task_name,
                    "aa": aa,
                    "bwt": bwt,
                    "fwt": fwt,
                },
            )
            self._mark_done(task_id)

    def train_all_tasks(self):
        seen = []
        last_done = _find_last_done_task(self.cfg.output_dir, len(self.cfg.train_tasks))

        if last_done > 0:
            print(
                f"[Resume] Loading checkpoint from task {last_done} "
                f"({self.cfg.train_tasks[last_done - 1]})"
            )
            self._preallocate_orthohist_pool_for_resume(last_done)
            load_checkpoint(self.cfg.output_dir, last_done, self.model)

            for layer in self.manager.layers.values():
                layer._n_pool = last_done

            self._restore_history_and_scores()
            self._restore_reuse_snapshots()

        for task_id, task in enumerate(self.cfg.train_tasks, 1):
            seen.append(task)
            if task_id <= last_done:
                print(f"[Resume] skipping {task} (already done)")
                continue
            self.train_one_task(task, task_id, seen)

        self._finalise()
        return {"history": self.history, "score_matrix": self.score_matrix}

    def _finalise(self):
        self._save_history()
        self._save_score_matrix()

        if self.reuse_snapshots:
            matrix = collect_reuse_matrix(self.reuse_snapshots, self.cfg.train_tasks)
            save_reuse_heatmap(matrix, self.cfg.train_tasks, self.cfg.output_dir)


class SequentialLoRATrainer:
    def __init__(self, cfg, tokenizer, model, device):
        self.cfg = cfg
        self.tok = tokenizer
        self.model = model
        self.device = device
        self.loaders = build_task_loaders(
            trace_root=cfg.trace_root,
            tasks=cfg.train_tasks,
            tokenizer=tokenizer,
            batch_size=cfg.per_device_batch_size,
            max_length=cfg.max_length,
        )
        self.score_matrix = {}
        self.zero_shot_scores = {}
        self.history = []
        self.logger = JsonLogger(cfg.output_dir)

        self._set_lora_trainable_only()

    def _set_lora_trainable_only(self):
        for _, p in self.model.named_parameters():
            p.requires_grad_(False)

        trainable = []
        for name, p in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad_(True)
                trainable.append(p)

        if not trainable:
            raise RuntimeError("No LoRA parameters found in SequentialLoRATrainer.")

    def _trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _save_history(self):
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _save_score_matrix(self):
        save_score_matrix(
            self.score_matrix,
            self.cfg.train_tasks,
            self.cfg.output_dir,
        )

        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "score_matrix_raw.json", "w", encoding="utf-8") as f:
            json.dump(self.score_matrix, f, indent=2, ensure_ascii=False)

    def _restore_history_and_scores(self):
        out = Path(self.cfg.output_dir)

        hp = out / "history.json"
        if hp.exists():
            with open(hp, encoding="utf-8") as f:
                self.history = json.load(f)

        smp = out / "score_matrix_raw.json"
        if smp.exists():
            with open(smp, encoding="utf-8") as f:
                self.score_matrix = json.load(f)
            print(f"[Resume] restored score_matrix from {smp}")
        else:
            print("[Resume WARNING] score_matrix_raw.json not found; rebuilding from history.")
            self.score_matrix = {}
            for entry in self.history:
                key = entry.get("task_name") or entry.get("task")
                if key is not None:
                    self.score_matrix[key] = dict(entry.get("seen_scores", {}))

    def _mark_done(self, task_id):
        done_flag = (
            Path(self.cfg.output_dir)
            / "checkpoints"
            / f"task_{task_id:02d}"
            / "DONE"
        )
        done_flag.parent.mkdir(parents=True, exist_ok=True)
        done_flag.touch()

    @torch.no_grad()
    def _generate(self, task_name, split="test"):
        self.model.eval()
        loader = self.loaders[task_name][split]
        preds, refs, prompts = [], [], []

        for batch in tqdm(loader, desc=f"eval {task_name}/{split}", leave=False):
            prompt_texts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch["prompt_text"]
            ]

            enc = self.tok(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                add_special_tokens=False,
            ).to(self.device)

            if task_name in {"C-STANCE", "FOMC"}:
                max_new_tokens = 8
            elif task_name == "ScienceQA":
                max_new_tokens = 64
            else:
                max_new_tokens = 128

            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
                do_sample=False,
            )

            input_len = enc["input_ids"].shape[1]
            for i in range(out.size(0)):
                pred = self.tok.decode(out[i][input_len:], skip_special_tokens=True)
                preds.append(pred.strip())

            refs.extend(batch["answer_text"])
            prompts.extend(batch["prompt_text"])

        if getattr(self.cfg, "debug_eval", False):
            debug_parse_examples(task_name, preds, refs, n=10)

        return {"preds": preds, "refs": refs, "prompts": prompts}

    def _save_raw_predictions(self, row_key, task_name, split, gen_results):
        debug_dir = Path(self.cfg.output_dir) / "debug_predictions"
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_row = str(row_key).replace("/", "_")
        safe_task = str(task_name).replace("/", "_")
        out_path = debug_dir / f"{safe_row}__{safe_task}__{split}.jsonl"

        rows = build_debug_rows(
            task_name=task_name,
            preds=gen_results["preds"],
            refs=gen_results["refs"],
            prompts=gen_results["prompts"],
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _eval_tasks(self, task_list, row_key, split="test"):
        self.score_matrix.setdefault(row_key, {})
        task_scores = {}

        for task in task_list:
            gen_results = self._generate(task, split=split)
            self._save_raw_predictions(
                row_key=row_key,
                task_name=task,
                split=split,
                gen_results=gen_results,
            )

            res = score_task(
                task,
                gen_results["preds"],
                gen_results["refs"],
                gen_results["prompts"],
            )

            task_scores[task] = res.metric_value
            self.score_matrix[row_key][task] = res.metric_value

            self.logger.log({
                "event": "task_eval",
                "row": row_key,
                "task": task,
                "split": split,
                "metric": res.metric_name,
                "value": res.metric_value,
                "num_examples": res.num_examples,
            })

        return task_scores

    def _train_one_task(self, task_name, task_id):
        self._set_lora_trainable_only()
        trainable = self._trainable_parameters()
        loader = self.loaders[task_name]["train"]

        n_steps = max(
            1,
            math.ceil(
                len(loader) * self.cfg.num_epochs
                / self.cfg.grad_accum_steps
            ),
        )
        opt, sch = _make_optimizer_and_scheduler(trainable, self.cfg, n_steps)
        self.model.train()
        opt.zero_grad(set_to_none=True)

        self.logger.log({
            "event": "task_start",
            "task": task_name,
            "task_id": task_id,
            "trainable_params": sum(p.numel() for p in trainable),
        })

        for epoch in range(1, self.cfg.num_epochs + 1):
            total_loss = 0.0
            batch_cnt = 0
            pbar = tqdm(loader, desc=f"seq_lora [{task_name}] ep{epoch}", leave=False)

            for step, batch in enumerate(pbar, 1):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                (out.loss / self.cfg.grad_accum_steps).backward()

                if step % self.cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, self.cfg.max_grad_norm)
                    opt.step()
                    sch.step()
                    opt.zero_grad(set_to_none=True)

                total_loss += float(out.loss.item())
                batch_cnt += 1
                if batch_cnt % self.cfg.log_every == 0:
                    pbar.set_postfix(loss=f"{total_loss / batch_cnt:.4f}")

            if len(loader) % self.cfg.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(trainable, self.cfg.max_grad_norm)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)

            self.logger.log({
                "event": "epoch_end",
                "task": task_name,
                "epoch": epoch,
                "loss": total_loss / max(1, batch_cnt),
            })

    def train_all_tasks(self):
        seen = []
        last_done = _find_last_done_task(self.cfg.output_dir, len(self.cfg.train_tasks))

        if last_done > 0:
            print(
                f"[Resume] Loading Sequential LoRA checkpoint from task {last_done} "
                f"({self.cfg.train_tasks[last_done - 1]})"
            )
            load_checkpoint(self.cfg.output_dir, last_done, self.model)
            self.model.to(self.device)
            self._set_lora_trainable_only()
            self._restore_history_and_scores()

        for task_id, task in enumerate(self.cfg.train_tasks, 1):
            seen.append(task)
            if task_id <= last_done:
                print(f"[Resume] skipping {task} (already done)")
                continue

            self._train_one_task(task, task_id)
            self._eval_tasks(self.cfg.train_tasks, row_key=task, split="test")

            scores = {t: self.score_matrix[task][t] for t in seen}
            aa = average_accuracy(scores)
            bwt = backward_transfer(self.score_matrix, seen)
            fwt = forward_transfer(self.zero_shot_scores, self.score_matrix, seen)

            self.history.append({
                "task_id": task_id,
                "task_name": task,
                "seen_scores": scores,
                "average_accuracy": aa,
                "bwt": bwt,
                "fwt": fwt,
            })

            self._save_history()
            self._save_score_matrix()

            self.logger.log({
                "event": "task_end",
                "task": task,
                "aa": aa,
                "bwt": bwt,
                "fwt": fwt,
            })
            print(f"[seq_lora {task}] AA={aa:.4f}  BWT={bwt:.4f}  FWT={fwt:.4f}")

            if self.cfg.save_every_task:
                save_checkpoint(
                    self.cfg.output_dir,
                    task_id,
                    self.model,
                    extra={
                        "task": task,
                        "aa": aa,
                        "bwt": bwt,
                        "fwt": fwt,
                    },
                )
                self._mark_done(task_id)

        self._finalise()
        return {"history": self.history, "score_matrix": self.score_matrix}

    def _finalise(self):
        self._save_history()
        self._save_score_matrix()
