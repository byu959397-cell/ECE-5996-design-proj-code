# TGR (OrthoHist-LoRA:) Continual Learning for Small Language Models

This repository contains a continual learning pipeline for parameter-efficient supervised fine-tuning of small instruction-tuned language models. The project compares a standard Sequential LoRA baseline with an OrthoHist-LoRA method that stores previous task updates in a frozen historical basis pool and learns a task-conditioned low-rank historical reuse operator.

The implementation uses Qwen2.5-3B-Instruct and a five-task TRACE-style continual learning sequence:

1. C-STANCE
2. FOMC
3. ScienceQA
4. NumGLUE-cm
5. NumGLUE-ds

The current default configuration files use the full local `./TRACE` directory and train for 3 epochs. For faster Colab validation, create a smaller `TRACE_fast` directory and change `trace_root` in the YAML files to `./TRACE_fast`.

---

## Repository Structure

```text
orthohist-lora/
├── README.md
├── requirements.txt
├── main.py
├── run_experiments.sh
├── summarise_results.py
│
├── configs/
│   ├── sequential_lora.yaml
│   ├── ortho_hist_k4.yaml
│   ├── ortho_hist_k8.yaml
│   └── ortho_hist_k16.yaml
│
├── src/
│   ├── config.py
│   ├── data_trace.py
│   ├── eval_trace.py
│   ├── trainers.py
│   ├── ortho_hist_lora.py
│   ├── checkpointing.py
│   └── analysis_utils.py
│
├── results/
│   └── README.md
│
└── docs/
```

---

## Environment Setup

### Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab setup

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Move into the project directory:

```python
%cd /content/drive/MyDrive/project
```

Install dependencies:

```python
!pip install -r requirements.txt
```

Check the GPU:

```python
!nvidia-smi
```

The project was tested mainly on an NVIDIA A100 GPU.

If PEFT raises an incompatible `torchao` error, uninstall `torchao`:

```python
!pip uninstall -y torchao
```

Then restart the Colab runtime and rerun the setup cells.

---

## Dataset Preparation

The code expects TRACE-style JSON files in this structure:

```text
TRACE/
├── C-STANCE/
│   ├── train.json
│   ├── eval.json
│   └── test.json
├── FOMC/
│   ├── train.json
│   ├── eval.json
│   └── test.json
├── ScienceQA/
├── NumGLUE-cm/
└── NumGLUE-ds/
```

Each JSON example should contain:

```json
{
  "prompt": "...",
  "answer": "..."
}
```

TRACE is a benchmark for continual learning in large language models. The official TRACE repository is `BeyonderXX/TRACE`, and the paper describes TRACE as an eight-dataset benchmark standardized for continual learning evaluation.

This project uses a five-task subset for the final implementation and controlled validation.

---

## Optional: Create a Smaller TRACE_fast Subset

For Colab runtime-limited validation, create a smaller dataset:

```python
import json
from pathlib import Path

src_root = Path("./TRACE")
dst_root = Path("./TRACE_fast")

tasks = ["C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds"]

for task in tasks:
    (dst_root / task).mkdir(parents=True, exist_ok=True)

    for split, n in [("train", 200), ("eval", 100), ("test", 100)]:
        src = src_root / task / f"{split}.json"
        dst = dst_root / task / f"{split}.json"

        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        data = data[:n]

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(task, split, len(data))
```

Then edit the YAML files:

```yaml
trace_root: ./TRACE_fast
```

For faster validation, also reduce:

```yaml
max_length: 512
num_epochs: 1
per_device_batch_size: 4
grad_accum_steps: 1
```

---

## Configuration Files

### Sequential LoRA

```text
configs/sequential_lora.yaml
```

This trains a single LoRA adapter continuously across all tasks.

Important fields:

```yaml
method: sequential_lora
rank: 16
save_every_task: true
```

### OrthoHist-LoRA

```text
configs/ortho_hist_k8.yaml
```

Important fields:

```yaml
method: ortho_hist
rank: 16
mt_rank: 8
save_every_task: true
```

Ablation configs:

```text
configs/ortho_hist_k4.yaml
configs/ortho_hist_k16.yaml
```

---

## Running Experiments

### Main run: Sequential LoRA + OrthoHist-LoRA k=8

```bash
bash run_experiments.sh 0 8
```

Arguments:

```text
0  -> GPU id
8  -> OrthoHist mt_rank value
```

This runs:

1. Sequential LoRA
2. OrthoHist-LoRA with `mt_rank=8`

### Run only OrthoHist-LoRA

```bash
RUN_SEQ=0 bash run_experiments.sh 0 8
```

Run k=4 only:

```bash
RUN_SEQ=0 bash run_experiments.sh 0 4
```

Run k=16 only:

```bash
RUN_SEQ=0 bash run_experiments.sh 0 16
```

### Run only Sequential LoRA

```bash
RUN_ORTHO=0 bash run_experiments.sh 0
```

### Direct commands

Sequential LoRA:

```bash
python main.py --config configs/sequential_lora.yaml
```

OrthoHist-LoRA k=8:

```bash
python main.py --config configs/ortho_hist_k8.yaml
```

---

## Output Files

Each run writes outputs to the directory specified by `output_dir` in its YAML file.

Example:

```text
outputs/ortho_hist_k8/
├── history.json
├── score_matrix_raw.json
├── reuse_snapshots.json
├── logs/
│   └── events.jsonl
├── checkpoints/
│   ├── task_01/
│   │   ├── model_state.pt
│   │   ├── meta.json
│   │   └── DONE
│   └── ...
└── analysis/
    ├── score_matrix.json
    ├── score_matrix.png
    ├── reuse_matrix.json
    └── reuse_heatmap.png
```

Important files:

- `history.json`: task-level continual learning metrics after each task.
- `score_matrix_raw.json`: raw score matrix used for AA and BWT.
- `analysis/score_matrix.png`: heatmap of continual learning performance.
- `analysis/reuse_heatmap.png`: OrthoHist historical reuse heatmap.
- `logs/events.jsonl`: JSONL training and evaluation logs.
- `checkpoints/task_XX/DONE`: task completion marker for Colab recovery.

Sequential LoRA does not produce `reuse_heatmap.png` because it has no historical mixing module.

---

## Reading Final Metrics

```python
import json
from pathlib import Path

runs = [
    "sequential_lora",
    "ortho_hist_k4",
    "ortho_hist_k8",
    "ortho_hist_k16",
]

for name in runs:
    p = Path(f"outputs/{name}/history.json")
    if not p.exists():
        print(name, "not finished")
        continue

    hist = json.load(open(p, "r", encoding="utf-8"))
    last = hist[-1]

    print("\n" + name)
    print("Final AA:", last["average_accuracy"])
    print("Final BWT:", last["bwt"])
    print("Final per-task scores:", last["seen_scores"])
```

You can also generate a summary table:

```bash
python summarise_results.py
```

---

## Reproducibility Notes

Recommended experiment order:

```bash
bash run_experiments.sh 0 8
```

If more time is available:

```bash
RUN_SEQ=0 bash run_experiments.sh 0 4
RUN_SEQ=0 bash run_experiments.sh 0 16
```

If Colab disconnects, rerun the same command. The trainer checks task-level checkpoints and skips tasks that already have a `DONE` flag.

---

## Metrics

The final validation focuses on:

- **Average Accuracy / Average Score (AA)**  
  Average score across tasks seen so far.

- **Backward Transfer (BWT)**  
  Measures forgetting by comparing old-task performance after later training.

- **Score Matrix**  
  Visualizes the continual learning trajectory.

- **Reuse Heatmap**  
  Visualizes historical basis reuse in OrthoHist-LoRA.

Forward Transfer (FWT) is logged but not emphasized because the current pipeline does not perform a full zero-shot evaluation before task training.

---

## Notes on GitHub Storage

Do not commit:

```text
TRACE/
TRACE_fast/
outputs/
checkpoints/
*.pt
*.bin
*.safetensors
debug_predictions/
```

Small result artifacts such as `history.json`, `score_matrix_raw.json`, `score_matrix.png`, and `reuse_heatmap.png` can be copied into `results/` if needed for the report.

---

## Citation / Acknowledgment

This project uses Qwen2.5-3B-Instruct as the base model and TRACE-style continual learning tasks for validation. If using TRACE, follow the official TRACE repository instructions for data access and cite the TRACE paper.
