from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def collect_reuse_matrix(reuse_snapshots, task_names):
    T = len(task_names)
    matrix = np.full((T, T), np.nan)
    task_to_idx = {name: idx for idx, name in enumerate(task_names)}

    for snap in reuse_snapshots:
        task_name = snap.get("task_name")
        if task_name not in task_to_idx:
            continue

        t_idx = task_to_idx[task_name]
        if t_idx == 0:
            continue

        layer_data = snap.get("layers", {})
        layer_means = []
        for norms in layer_data.values():
            if norms and len(norms) == t_idx:
                layer_means.append(norms)

        if not layer_means:
            continue

        arr = np.array(layer_means, dtype=float)       # [n_layers, t_idx]
        mean_per_hist = arr.mean(axis=0)               # [t_idx]
        for i, val in enumerate(mean_per_hist):
            matrix[t_idx, i] = val

    return matrix


def save_reuse_heatmap(matrix, task_names, output_dir):
    out = Path(output_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    serialisable = {
        task_names[t]: {
            task_names[i]: float(matrix[t, i])
            for i in range(t)
            if not np.isnan(matrix[t, i])
        }
        for t in range(len(task_names))
    }

    with open(out / "reuse_matrix.json", "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="Mean M_t block norm (Frobenius)")
        ax.set_xticks(range(len(task_names)))
        ax.set_yticks(range(len(task_names)))
        ax.set_xticklabels(task_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(task_names, fontsize=8)
        ax.set_xlabel("Historical task")
        ax.set_ylabel("Current task")
        ax.set_title("M_t reuse strength")

        for t in range(len(task_names)):
            for i in range(len(task_names)):
                v = matrix[t, i]
                if not np.isnan(v):
                    ax.text(i, t, f"{v:.2f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()
        plt.savefig(out / "reuse_heatmap.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[analysis] could not save reuse heatmap: {e}")


def save_score_matrix(score_matrix, task_names, output_dir):
    out = Path(output_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "score_matrix.json", "w", encoding="utf-8") as f:
        json.dump(score_matrix, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt

        T = len(task_names)
        mat = np.full((T, T), np.nan)

        for r_idx, r_name in enumerate(task_names):
            row = score_matrix.get(r_name, {})
            for c_idx, c_name in enumerate(task_names):
                if c_name in row:
                    mat[r_idx, c_idx] = row[c_name]

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Score")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(task_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(task_names, fontsize=8)
        ax.set_xlabel("Evaluated task")
        ax.set_ylabel("Trained up to task")
        ax.set_title("Continual learning score matrix")

        for r in range(T):
            for c in range(T):
                v = mat[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()
        plt.savefig(out / "score_matrix.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[analysis] could not save score matrix: {e}")


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
