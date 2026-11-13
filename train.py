"""LoRA fine-tune of a small instruct model on the smart-home energy dataset.

Usage:
    python train.py            # full run (2 epochs, ~1,560 steps)
    python train.py --smoke    # 20-step trial: checks memory and speed, writes to out/smoke
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "Design-Thinking-SLM/smart-home-energy-slm"

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="20-step trial run")
args = parser.parse_args()
out = "out/smoke" if args.smoke else "out/qwen0.5b-energy-lora"


def to_chat(r):
    # Conversational prompt/completion: the chat template is applied and
    # loss is computed on the assistant completion only.
    return {"prompt": [{"role": "user", "content": r["prompt"]}],
            "completion": [{"role": "assistant", "content": r["completion"]}]}


ds = load_dataset(DATASET).map(to_chat, remove_columns=["task", "answerable", "label"])
eval_ds = ds["validation"].select(range(200)) if args.smoke else ds["validation"]

trainer = SFTTrainer(
    model=BASE,
    train_dataset=ds["train"],
    eval_dataset=eval_ds,
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules="all-linear", task_type="CAUSAL_LM"),
    args=SFTConfig(
        output_dir=out,
        num_train_epochs=2,
        max_steps=20 if args.smoke else -1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,      # effective batch 32
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2 if args.smoke else 50,
        max_length=512,
        bf16=True,
        eval_strategy="steps",
        eval_steps=10 if args.smoke else 200,
        save_strategy="steps",
        save_steps=10 if args.smoke else 200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=5 if args.smoke else 20,
        report_to="none",
    ),
)
trainer.train()
trainer.save_model(f"{out}/final")
