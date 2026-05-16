from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


TRACE_TASKS = [
    "C-STANCE",
    "FOMC",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
]

TASK_METRICS = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "exact_match",
    "NumGLUE-ds": "exact_match",
}


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    prompt_text: str
    answer_text: str
    task_name: str


class TraceSFTDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        max_length=1024,
        task_name=None,
        split="train",
        max_eval_samples=100,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TRACE split file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.split = split
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.task_name = task_name or path.parent.name

        if split in {"eval", "test"} and max_eval_samples is not None:
            self.data = self.data[: int(max_eval_samples)]

    def __len__(self):
        return len(self.data)

    def _build_prompt_text(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_full_text(self, prompt, answer):
        return self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def _encode_with_mask(self, prompt, answer):
        prompt_text = self._build_prompt_text(prompt)
        full_text = self._build_full_text(prompt, answer)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
        answer_ids = full_ids[len(prompt_ids):]

        if len(full_ids) <= self.max_length:
            kept_prompt_ids = prompt_ids
            kept_answer_ids = answer_ids
        else:
            min_answer_tokens = max(1, min(len(answer_ids), 128))
            prompt_budget = max(0, self.max_length - min_answer_tokens)

            kept_prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            answer_budget = self.max_length - len(kept_prompt_ids)
            kept_answer_ids = answer_ids[:answer_budget]

            if kept_answer_ids and len(answer_ids) > len(kept_answer_ids):
                eos_ids = self.tokenizer(
                    self.tokenizer.eos_token or "<|im_end|>",
                    add_special_tokens=False,
                ).input_ids
                if eos_ids:
                    kept_answer_ids[-1] = eos_ids[-1]

        kept_ids = kept_prompt_ids + kept_answer_ids
        labels = [-100] * len(kept_prompt_ids) + kept_answer_ids.copy()

        return EncodedExample(
            input_ids=torch.tensor(kept_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            prompt_text=prompt,
            answer_text=answer,
            task_name=self.task_name,
        )

    def __getitem__(self, idx):
        sample = self.data[idx]

        if "prompt" not in sample or "answer" not in sample:
            raise KeyError(
                f"Each TRACE sample must contain 'prompt' and 'answer'. "
                f"Bad sample in {self.task_name}/{self.split}: {sample}"
            )

        if self.split in {"eval", "test"}:
            return {
                "prompt_text": sample["prompt"],
                "answer_text": sample["answer"],
                "task_name": self.task_name,
            }

        encoded = self._encode_with_mask(sample["prompt"], sample["answer"])
        return {
            "input_ids": encoded.input_ids,
            "labels": encoded.labels,
            "prompt_text": encoded.prompt_text,
            "answer_text": encoded.answer_text,
            "task_name": encoded.task_name,
        }


def left_pad_stack(seqs, pad_value):
    max_len = max(x.size(0) for x in seqs)
    out = []
    for x in seqs:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_value, dtype=x.dtype)
            out.append(torch.cat([pad, x], dim=0))
        else:
            out.append(x)
    return torch.stack(out, dim=0)


def make_trace_collator(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate(features):
        if "input_ids" not in features[0]:
            return {
                "prompt_text": [f["prompt_text"] for f in features],
                "answer_text": [f["answer_text"] for f in features],
                "task_name": [f["task_name"] for f in features],
            }

        input_ids = left_pad_stack([f["input_ids"] for f in features], pad_token_id)
        labels = left_pad_stack([f["labels"] for f in features], -100)
        attention_mask = (input_ids != pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_text": [f["prompt_text"] for f in features],
            "answer_text": [f["answer_text"] for f in features],
            "task_name": [f["task_name"] for f in features],
        }

    return collate


def build_trace_loader(
    trace_root,
    task_name,
    split,
    tokenizer,
    batch_size,
    max_length,
    shuffle=False,
    num_workers=0,
):
    path = Path(trace_root) / task_name / f"{split}.json"

    dataset = TraceSFTDataset(
        path,
        tokenizer=tokenizer,
        max_length=max_length,
        task_name=task_name,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_trace_collator(tokenizer),
        pin_memory=torch.cuda.is_available(),
    )


def build_task_loaders(
    trace_root,
    tasks,
    tokenizer,
    batch_size,
    max_length,
):
    loaders = {}

    for task in tasks:
        loaders[task] = {
            "train": build_trace_loader(
                trace_root,
                task,
                "train",
                tokenizer,
                batch_size,
                max_length,
                shuffle=True,
            ),
            "eval": build_trace_loader(
                trace_root,
                task,
                "eval",
                tokenizer,
                20,
                max_length,
                shuffle=False,
            ),
            "test": build_trace_loader(
                trace_root,
                task,
                "test",
                tokenizer,
                20,
                max_length,
                shuffle=False,
            ),
        }

    return loaders
