# python/training/build_dataset.py
import json
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict
from mel.mel_bridge import english_to_mel

SYSTEM_PREFIX = "<mel_json>\n"
SYSTEM_SUFFIX = "\n</mel_json>"

def build_pairs_squad(train_limit=2000, val_limit=500):
    ds = load_dataset("squad")
    def ex_to_pair(ex):
        q = ex["question"].strip()
        mel = english_to_mel(q, intent="qa")
        src = f"ENGLISH_TO_MEL:\n{q}\nOUTPUT:"
        tgt = SYSTEM_PREFIX + json.dumps(mel, ensure_ascii=False) + SYSTEM_SUFFIX
        return {"input_text": src, "target_text": tgt}
    train = ds["train"].select(range(min(train_limit, len(ds["train"])))).map(
        ex_to_pair, remove_columns=ds["train"].column_names
    )
    valid = ds["validation"].select(range(min(val_limit, len(ds["validation"])))).map(
        ex_to_pair, remove_columns=ds["validation"].column_names
    )
    return DatasetDict({"train": train, "validation": valid})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--out", type=str, default="mel_pairs_squad")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.out
    dsd = build_pairs_squad(args.train, args.val)
    dsd.save_to_disk(str(out_dir))
    print(f"Saved dataset to {out_dir}")
