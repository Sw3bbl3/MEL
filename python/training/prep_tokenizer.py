from transformers import AutoTokenizer

TOK_BASE = "gpt2"
TOK_OUT = "gpt2_mel_tok"
SPECIALS = ["<mel_json>", "</mel_json>", "<mel_end>"]

def main():
    tok = AutoTokenizer.from_pretrained(TOK_BASE)
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    tok.save_pretrained(TOK_OUT)
    print(f"Saved tokenizer to {TOK_OUT} with specials: {SPECIALS}")

if __name__ == "__main__":
    main()
