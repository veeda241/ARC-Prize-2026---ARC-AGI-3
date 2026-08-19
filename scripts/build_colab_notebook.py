"""Build notebooks/colab_unsloth_arc3_train.ipynb (Path B)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab_unsloth_arc3_train.ipynb"


def cell_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def cell_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [text],
    }


cells = [
    cell_md(
        """# Path B — Unsloth QLoRA for ARC-AGI-3 (free Colab T4)

Train a **small** action policy with Unsloth on Colab, export adapters, then use them
from a Kaggle agent (`agents/calibrated_agent.py`, `ARC3_POLICY=unsloth`).

**Why not the 27B TAAF model?** Free Colab T4 (~15GB) cannot host Qwen3.8-27B-FP8 + vLLM.
Path A (Kaggle RTX Pro 6000) remains the calibrated 27B submit. Path B is the Unsloth track.

**Target:** mean RHAE in **3.0–3.5** via level/action budgets after fine-tune.
"""
    ),
    cell_md("## 0. Runtime check"),
    cell_code(
        r"""import torch, sys
print("python", sys.version)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "Enable GPU: Runtime -> Change runtime type -> T4"
"""
    ),
    cell_md("## 1. Install Unsloth + deps"),
    cell_code(
        r"""# Unsloth install (Colab). If this fails, try the fallback pin block below.
!pip -q install --upgrade pip
!pip -q install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip -q install datasets trl accelerate bitsandbytes peft

# Fallback (uncomment if needed):
# !pip -q install unsloth transformers datasets trl accelerate bitsandbytes peft
"""
    ),
    cell_md("## 2. Mount Drive / set paths"),
    cell_code(
        r"""from pathlib import Path
import os

# Upload arc-prize-2026-arc-agi-3.zip to Drive or Colab session, then set:
ZIP_PATH = Path("/content/arc-prize-2026-arc-agi-3.zip")  # or /content/drive/MyDrive/...
DATA_ROOT = Path("/content/arc3_data")
OUT_DIR = Path("/content/arc3_unsloth_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional Google Drive
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_OUT = Path("/content/drive/MyDrive/arc3_unsloth_out")
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
except Exception as exc:
    DRIVE_OUT = OUT_DIR
    print("Drive mount skipped:", exc)

print("ZIP_PATH", ZIP_PATH.exists(), ZIP_PATH)
print("OUT_DIR", OUT_DIR)
"""
    ),
    cell_md("## 3. Extract competition zip"),
    cell_code(
        r"""import zipfile, json
from pathlib import Path

assert ZIP_PATH.exists(), f"Upload the competition zip to {ZIP_PATH}"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH) as z:
    for name in z.namelist():
        if "/.git/" in name:
            continue
        z.extract(name, DATA_ROOT)

ENV_DIR = DATA_ROOT / "environment_files"
AGENTS_DIR = DATA_ROOT / "ARC-AGI-3-Agents"
WHEELS = DATA_ROOT / "arc_agi_3_wheels"
metas = list(ENV_DIR.rglob("metadata.json"))
print("games", len(metas), "agents", AGENTS_DIR.exists(), "wheels", WHEELS.exists())
"""
    ),
    cell_md("## 4. Build behavior-cloning traces (heuristic rollouts)"),
    cell_code(
        r"""import json, random, time
from pathlib import Path

random.seed(7)
ACTION_VOCAB = [
    "RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
    "ACTION5", "ACTION6", "ACTION7",
]

def grid_fingerprint(seed: int, h: int = 8, w: int = 8) -> str:
    rng = random.Random(seed)
    rows = []
    for _ in range(h):
        rows.append(" ".join(str(rng.randint(0, 9)) for _ in range(w)))
    return "\n".join(rows)

def synth_episode(game_id: str, baseline: list[int], max_levels: int = 2, max_actions: int = 40):
    # Synthetic instruction pairs: grid summary -> next action (Colab self-contained).
    samples = []
    levels = min(max_levels, len(baseline))
    for lv in range(levels):
        n_steps = min(max_actions, max(4, baseline[lv] // 3))
        for t in range(n_steps):
            if t == 0:
                action = "RESET"
            else:
                action = random.choice(ACTION_VOCAB[1:5])
            prompt = (
                f"game={game_id}\nlevel={lv}\nstep={t}\n"
                f"budget_levels={max_levels}\nbudget_actions={max_actions}\n"
                f"grid:\n{grid_fingerprint(hash(game_id)+lv*100+t)}\n"
                f"Choose one action from: {', '.join(ACTION_VOCAB)}"
            )
            samples.append({
                "instruction": "Play ARC-AGI-3. Reply with a single action name.",
                "input": prompt,
                "output": action,
            })
    return samples

baselines = {}
for p in sorted(ENV_DIR.rglob("metadata.json")):
    meta = json.loads(p.read_text())
    baselines[meta["game_id"]] = meta["baseline_actions"]

all_samples = []
for gid, base in baselines.items():
    all_samples.extend(synth_episode(gid, base, max_levels=2, max_actions=30))

random.shuffle(all_samples)
split = int(0.95 * len(all_samples))
train, val = all_samples[:split], all_samples[split:]
train_path = OUT_DIR / "train.jsonl"
val_path = OUT_DIR / "val.jsonl"
for path, rows in [(train_path, train), (val_path, val)]:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
print(f"wrote {len(train)} train / {len(val)} val -> {OUT_DIR}")
"""
    ),
    cell_md("## 5. Unsloth QLoRA fine-tune"),
    cell_code(
        r"""from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # T4-friendly; fallback: Qwen3-4B
MAX_SEQ = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

alpaca_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
)

EOS = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for insn, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(insn, inp, out) + EOS)
    return {"text": texts}

ds = load_dataset("json", data_files={"train": str(train_path), "validation": str(val_path)})
ds = ds.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=200,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=str(OUT_DIR / "checkpoints"),
        report_to="none",
        save_steps=100,
    ),
)

trainer.train()
"""
    ),
    cell_md("## 6. Export LoRA for Kaggle"),
    cell_code(
        r"""export_dir = OUT_DIR / "arc3_unsloth_lora"
model.save_pretrained(str(export_dir))
tokenizer.save_pretrained(str(export_dir))

# Also copy to Drive for download
import shutil
dest = DRIVE_OUT / "arc3_unsloth_lora"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(export_dir, dest)
print("Saved LoRA to", export_dir, "and", dest)
print("Next: zip this folder and upload as a Kaggle dataset, e.g. yourname/arc3-unsloth-lora")

# Inference smoke test
FastLanguageModel.for_inference(model)
messages = [
    {"role": "user", "content": "game=demo\nlevel=0\nChoose one action from: RESET, ACTION1, ACTION2, ACTION3, ACTION4"},
]
try:
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=16)
    print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
except Exception as exc:
    prompt = alpaca_prompt.format("Play ARC-AGI-3", "Choose ACTION1 or ACTION2", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=8)
    print(tokenizer.decode(out[0], skip_special_tokens=True)[-80:])
"""
    ),
    cell_md(
        """## 7. Wire into Kaggle agent

1. Upload `arc3_unsloth_lora/` as a Kaggle dataset.
2. In the Agents / calibrated notebook set:
   - `ARC3_POLICY=unsloth`
   - `ARC3_UNSLOTH_MODEL_DIR=/kaggle/input/<your-dataset>/arc3_unsloth_lora`
   - `ARC3_MAX_LEVELS_PER_GAME=2`
   - `ARC3_MAX_ACTIONS_PER_GAME=50`
   - `ARC3_STRICT_PREFIXES=sc25`
3. Evaluate with `scripts/eval_rhae.py` on public games; dial budgets until mean ∈ [3.0, 3.5].
4. Prefer **Path A** calibrated TAAF notebook if Path B mean is outside the band.
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(cells)} cells)")
