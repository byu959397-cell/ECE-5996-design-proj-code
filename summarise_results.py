#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

TASKS = [
    "C-STANCE",
    "FOMC",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
]

SHORT = ["C-ST", "FOMC", "SciQ", "Num-cm", "Num-ds"]

RUNS = {
    "Sequential LoRA": "outputs/sequential_lora",
    "OrthoHist k=4": "outputs/ortho_hist_k4",
    "OrthoHist k=8": "outputs/ortho_hist_k8",
    "OrthoHist k=16": "outputs/ortho_hist_k16",
}


def load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history(run_dir: str):
    return load_json(Path(run_dir) / "history.json")


def final_row_from_history(history):
    if not history:
        return {
            "scores": {},
            "aa": None,
            "bwt": None,
            "fwt": None,
        }

    last = history[-1]

    # New trainer format.
    scores = last.get("seen_scores")
    aa = last.get("average_accuracy")

    # Backward compatibility with older SequentialLoRATrainer history.
    if scores is None:
        scores = last.get("scores", {})
    if aa is None:
        aa = last.get("aa")

    return {
        "scores": scores or {},
        "aa": aa,
        "bwt": last.get("bwt"),
        "fwt": last.get("fwt"),
    }


def fmt(v):
    return "—" if v is None else f"{v * 100:.1f}"


def build_rows():
    rows = []
    for name, run_dir in RUNS.items():
        history = load_history(run_dir)
        row = {"name": name, "run_dir": run_dir}
        row.update(final_row_from_history(history))
        rows.append(row)
    return rows


def print_console_table(rows):
    header = (
        f"{'Method':<22}"
        + "".join(f"{s:>8}" for s in SHORT)
        + f"{'AA':>8}{'BWT':>8}{'FWT':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['name']:<22}"
        for task in TASKS:
            line += f"{fmt(row['scores'].get(task)):>8}"
        line += f"{fmt(row['aa']):>8}"
        line += f"{fmt(row['bwt']):>8}"
        line += f"{fmt(row['fwt']):>8}"
        print(line)

    missing = [r for r in rows if r["aa"] is None]
    if missing:
        print("\nMissing runs:")
        for r in missing:
            print(f"  - {r['name']}: {r['run_dir']}/history.json not found")

    print("\nValues are percentages. BWT < 0 indicates forgetting.")


def print_latex_table(rows):
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    cols = "l" + "c" * (len(TASKS) + 3)
    print(r"\begin{tabular}{" + cols + r"}")
    print(r"\toprule")
    print("Method & " + " & ".join(SHORT) + r" & AA & BWT & FWT \\")
    print(r"\midrule")

    for row in rows:
        cells = [row["name"]]
        for task in TASKS:
            cells.append(fmt(row["scores"].get(task)))
        cells.append(fmt(row["aa"]))
        cells.append(fmt(row["bwt"]))
        cells.append(fmt(row["fwt"]))
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Continual learning results on five TRACE tasks "
        r"(Qwen2.5-3B-Instruct, LoRA rank=16). "
        r"AA denotes average accuracy over seen tasks after the full task sequence. "
        r"BWT$<$0 indicates forgetting.}"
    )
    print(r"\label{tab:main_results}")
    print(r"\end{table}")


def main():
    rows = build_rows()
    print_console_table(rows)
    print()
    print_latex_table(rows)


if __name__ == "__main__":
    main()
