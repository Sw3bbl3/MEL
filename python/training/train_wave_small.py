# python/training/train_wave_small.py
import os, random, json, inspect
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

class DataCollatorForCausalLMMEL:
    def __init__(self, tokenizer, pad_label_id: int = -100, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_label_id = pad_label_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # pull labels out before tokenizer.pad
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        max_len = batch["input_ids"].size(1)

        import torch
        out_labels = torch.full((len(labels), max_len), self.pad_label_id, dtype=torch.long)
        for i, lab in enumerate(labels):
            lab = lab[:max_len]
            out_labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
        batch["labels"] = out_labels
        return batch

# Repro
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "mel_pairs_squad"
TOK_PATH  = REPO_ROOT / "gpt2_mel_tok"
BASE_MODEL = "gpt2"
OUT_DIR   = REPO_ROOT / "wave_small_mel"

def format_example(tok, ex):
    prompt = ex["input_text"] + "\n"
    target = ex["target_text"]
    full = prompt + target
    ids = tok(full, truncation=True, max_length=256)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    labels = [-100]*len(prompt_ids) + ids["input_ids"][len(prompt_ids):]
    ids["labels"] = labels[:len(ids["input_ids"])]
    return ids

def main():
    if not TOK_PATH.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {TOK_PATH}. Run prep_tokenizer.py first.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_PATH}. Run build_dataset.py first.")

    tok = AutoTokenizer.from_pretrained(str(TOK_PATH), local_files_only=True)
    # GPT-2 has no pad token by default
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # add this

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tok))
    # Make sure model knows the pad token id
    model.config.pad_token_id = tok.pad_token_id

    ds = load_from_disk(str(DATA_PATH))

    cols = ["input_ids", "attention_mask", "labels"]

    def _fmt(ex):
        return format_example(tok, ex)

    train_ds = ds["train"].map(
        _fmt,
        remove_columns=list(ds["train"].features)
    ).with_format(type="torch", columns=cols)

    val_ds = ds["validation"].map(
        _fmt,
        remove_columns=list(ds["validation"].features)
    ).with_format(type="torch", columns=cols)

    collator = DataCollatorForCausalLMMEL(tok)

    # Build TrainingArguments with version compatibility
    ta_kwargs = dict(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,
        bf16=False,
        report_to="none",
    )
    import inspect
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["eval_strategy"] = "epoch"

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,  # FutureWarning is fine
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    print(f"Saved fine tuned model to {OUT_DIR}")

if __name__ == "__main__":
    main()
