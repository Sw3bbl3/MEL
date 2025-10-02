from pathlib import Path
import random, numpy as np, torch, json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_IN  = REPO_ROOT / "wave_small_mel"
MODEL_OUT = REPO_ROOT / "wave_small_mel_align"
tok = AutoTokenizer.from_pretrained(str(MODEL_IN))
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

def make_pairs(n=200):
    # Synthetic QA prompts using the bridge for gold MEL
    samples = [
        "What is the tallest mountain in Europe?",
        "Who wrote War and Peace?",
        "What is the capital of Japan?",
        "How many continents are there?",
        "What is H2O commonly known as?",
    ]
    data = []
    for i in range(n):
        q = random.choice(samples)
        mel = {
            "type": "TASK_REQUEST",
            "task": {
                "task_id": f"T-align-{i:04d}",
                "intent": "qa",
                "inputs": [{"name": "text", "kind": "text", "lang": "en", "value": q}],
                "expected": [{"name": "answer", "kind": "text"}],
                "constraints": {"max_latency_ms": 200, "max_tokens": 128, "device_pref": ["NPU","CPU"], "deterministic": True},
                "hints": {}
            }
        }
        prompt = f"ENGLISH_TO_MEL:\n{q}\nOUTPUT:\n<mel_json>\n"
        target = json.dumps(mel) + "\n</mel_json>"
        full = prompt + target
        ids = tok(full, truncation=True, max_length=256)
        # label mask so loss is on the target portion only
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        labels = [-100]*len(prompt_ids) + ids["input_ids"][len(prompt_ids):]
        ids["labels"] = labels[:len(ids["input_ids"])]
        data.append(ids)
    return Dataset.from_list(data)

def main():
    train = make_pairs(400)
    val = make_pairs(50)
    ds = DatasetDict({"train": train, "validation": val}).with_format("torch")

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_IN))
    model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id

    args = TrainingArguments(
        output_dir=str(MODEL_OUT),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,  # fast
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=False, bf16=False
    )

    from transformers import Trainer
    class Collator:
        def __init__(self, tok): self.tok = tok
        def __call__(self, features):
            labels = [f.pop("labels") for f in features]
            batch = self.tok.pad(features, return_tensors="pt")
            import torch as T
            max_len = batch["input_ids"].size(1)
            out_labels = T.full((len(labels), max_len), -100, dtype=T.long)
            for i, lab in enumerate(labels):
                lab = lab[:max_len]
                out_labels[i, :len(lab)] = T.tensor(lab, dtype=T.long)
            batch["labels"] = out_labels
            return batch

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], data_collator=Collator(tok), tokenizer=tok)
    trainer.train()
    trainer.save_model(str(MODEL_OUT))
    tok.save_pretrained(str(MODEL_OUT))
    print("Saved aligned model to", MODEL_OUT)

if __name__ == "__main__":
    main()
