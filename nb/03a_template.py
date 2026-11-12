from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
END = "<|im_end|>"
END_ID = tokenizer.convert_tokens_to_ids(END)
assert tokenizer.eos_token == END


def prompt_messages(row, system=None):
    return [{"role": "system", "content": system or system_text(row)}, {"role": "user", "content": user_text(row)}]


def prompt_string(row, system=None):
    return tokenizer.apply_chat_template(prompt_messages(row, system), tokenize=False, add_generation_prompt=True)


def encode_example(row):
    prompt_ids = tokenizer(prompt_string(row), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(label_text(row) + END, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "prompt_len": len(prompt_ids),
        "target_len": len(target_ids),
    }


example = train_clean[0]
print(prompt_string(example) + label_text(example) + END)

mismatch = 0
for r in train_clean:
    full = tokenizer.apply_chat_template(r["messages"], tokenize=False)
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    enc = encode_example(r)
    if full_ids[: len(enc["input_ids"])] != enc["input_ids"]:
        mismatch += 1
print(f"\nprompt + target tokenization identical to the full chat template on {len(train_clean) - mismatch}/{len(train_clean)} rows")
assert mismatch == 0

enc = encode_example(example)
trained_on = tokenizer.decode([t for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100])
print("tokens with loss:", repr(trained_on))
assert trained_on == label_text(example) + END
