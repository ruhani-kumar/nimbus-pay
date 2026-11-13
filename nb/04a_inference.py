import gc
import shutil
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_KWARGS = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_model(model_id=BASE_MODEL, four_bit=False, adapter=None):
    kwargs = {"dtype": DTYPE}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not four_bit:
        model.to(DEVICE)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model


def length_batches(encoded, token_budget, max_batch):
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    batches, current = [], []
    for i in order:
        longest = max([len(encoded[j]) for j in current] + [len(encoded[i])])
        if current and ((longest + MAX_NEW_TOKENS) * (len(current) + 1) > token_budget or len(current) == max_batch):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


@torch.no_grad()
def generate_predictions(model, rows, system=None, token_budget=24_000, max_batch=32, extra_gen=None, desc="generate"):
    tokenizer.padding_side = "left"
    prompts = [prompt_string(r, system) for r in rows]
    encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    outputs = [None] * len(rows)
    batches = length_batches(encoded, token_budget, max_batch)
    start = time.time()
    for b, idx in enumerate(batches):
        batch = tokenizer.pad({"input_ids": [encoded[i] for i in idx]}, return_tensors="pt").to(model.device)
        gen = model.generate(**batch, **GEN_KWARGS, **(extra_gen or {}), pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen[:, batch["input_ids"].shape[1]:]
        for i, toks in zip(idx, new_tokens):
            outputs[i] = tokenizer.decode(toks, skip_special_tokens=True).strip()
        print(f"\r{desc}: batch {b + 1}/{len(batches)} ({time.time() - start:.0f}s)", end="")
    print()
    return [{"id": r["id"], "output": o} for r, o in zip(rows, outputs)]


def effective_generation_config(model):
    cfg = model.generation_config.to_dict()
    cfg.update(GEN_KWARGS)
    keep = ["do_sample", "max_new_tokens", "repetition_penalty", "temperature", "top_p", "top_k", "eos_token_id"]
    return {k: cfg.get(k) for k in keep}


SMALL_FILES = ["dev_predictions.jsonl", "dev_scores.json", "stress_predictions.jsonl", "stress_scores.json", "train_log.csv", "train_meta.json"]


def export_run(name):
    for fname in SMALL_FILES:
        src = RUNS / name / fname
        if src.exists():
            (RESULTS / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(src, RESULTS / name / fname)


def run_eval(name, model, rows=None, system=None, change="", extra=None, force=False, **gen):
    rows = rows if rows is not None else dev
    out_dir = RUNS / name
    pred_path = out_dir / "dev_predictions.jsonl"
    if pred_path.exists() and not force:
        preds = load_jsonl(pred_path)
        print(f"{name}: reusing saved predictions from {pred_path}")
    else:
        preds = generate_predictions(model, rows, system=system, desc=name, **gen)
        save_jsonl(preds, pred_path)
    res = evaluate(rows, preds)
    (out_dir / "dev_scores.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    show_scores(res, prefixes=["category="])
    log_experiment({
        "run": name, "change": change,
        "dev_exact": round(res["all"]["exact_match"], 4),
        "dev_field_acc": round(res["all"]["mean_field_acc"], 4),
        "dev_json_valid": round(res["all"]["json_valid"], 4),
        **(extra or {}),
    })
    export_run(name)
    return preds, res
