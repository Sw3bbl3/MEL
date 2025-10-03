# python/training/build_dataset.py
import json
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict
from mel.mel_bridge import english_to_mel

SYSTEM_PREFIX = "<mel_json>\n"
SYSTEM_SUFFIX = "\n</mel_json>"

def _compact(obj) -> str:
    # Stable, compact JSON without spaces to keep sequences short
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def build_pairs_squad(train_limit=2000, val_limit=500):
    """
    For each SQuAD example, produce a supervised pair:
      input_text  = "ENGLISH_TO_MEL:\n{question}\nOUTPUT:"
      target_text = <mel_json>{TASK_REQUEST}</mel_json>\n<mel_json>{TASK_RESULT with answer}</mel_json>
    """
    ds = load_dataset("squad")

    def ex_to_pair(ex):
        q = ex["question"].strip()
        # Use the first annotated answer as ground truth
        # (SQuAD provides multiple; you can vote/normalize later if desired)
        a = ex["answers"]["text"][0].strip() if ex["answers"]["text"] else ""

        # Build a canonical TASK_REQUEST via the bridge, then pin task_id to the SQuAD id
        req = english_to_mel(q, intent="qa")
        req["task"]["task_id"] = f"T-{ex['id']}"

        # Supervised TASK_RESULT with the answer
        res = {
            "type": "TASK_RESULT",
            "task_id": req["task"]["task_id"],
            "status": "ok",
            "outputs": [
                {
                    "name": "answer",
                    "kind": "text",
                    "value": a,
                }
            ],
            # Keep metrics optional and minimal for training
            "metrics": {}
        }

        # Compose two MEL blocks: request then result
        tgt = (
            SYSTEM_PREFIX + _compact(req) + SYSTEM_SUFFIX + "\n" +
            SYSTEM_PREFIX + _compact(res) + SYSTEM_SUFFIX
        )
        src = f"ENGLISH_TO_MEL:\n{q}\nOUTPUT:"
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
