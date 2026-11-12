import contextlib
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import get_cosine_schedule_with_warmup

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAINABLE_CAP = 50_000_000


def collate(examples):
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for k, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[k, :n] = torch.tensor(e["input_ids"])
        labels[k, :n] = torch.tensor(e["labels"])
        mask[k, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def plan_steps(lengths, rows_per_step, token_budget, rng):
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i] // 32, rng.random()))
    chunks = [order[i:i + rows_per_step] for i in range(0, len(order), rows_per_step)]
    rng.shuffle(chunks)
    steps = []
    for chunk in chunks:
        micro, current = [], []
        for i in sorted(chunk, key=lambda i: lengths[i]):
            if current and lengths[i] * (len(current) + 1) > token_budget:
                micro.append(current)
                current = []
            current.append(i)
        micro.append(current)
        steps.append(micro)
    return steps


def autocast():
    if torch.cuda.is_available() and DTYPE != torch.float32:
        return torch.autocast("cuda", dtype=DTYPE)
    return contextlib.nullcontext()


def answer_loss(model, batch):
    inner = model.get_base_model()
    hidden = inner.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    targets = batch["labels"][:, 1:]
    keep = targets != -100
    logits = inner.lm_head(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, targets[keep], reduction="sum"), int(keep.sum())


@torch.no_grad()
def dev_loss(model, encoded, token_budget):
    model.eval()
    lengths = [len(e["input_ids"]) for e in encoded]
    total, count = 0.0, 0
    for micro in plan_steps(lengths, len(encoded), token_budget, random.Random(0)):
        for mb in micro:
            batch = {k: v.to(model.device) for k, v in collate([encoded[i] for i in mb]).items()}
            with autocast():
                loss_sum, n = answer_loss(model, batch)
            total += loss_sum.item()
            count += n
    model.train()
    return total / count


def train_lora(name, rows, rank=16, epochs=3, lr=2e-4, four_bit=False, rows_per_step=16, token_budget=8192,
               warmup_ratio=0.03, seed=SEED, model_id=None, force=False):
    out_dir = RUNS / name
    adapter_dir = out_dir / "adapter"
    meta_path = out_dir / "train_meta.json"
    if meta_path.exists() and (adapter_dir / "adapter_config.json").exists() and not force:
        print(f"{name}: reusing trained adapter from {adapter_dir}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    set_seed(seed)
    free_memory()
    model = load_model(model_id or BASE_MODEL, four_bit=four_bit)
    model.config.use_cache = False
    checkpointing = {"use_reentrant": False}
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=checkpointing)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing)
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    assert trainable <= TRAINABLE_CAP, trainable
    assert all(p.dtype == torch.float32 for p in params)
    print(f"{name}: {len(rows)} rows, {'4-bit NF4' if four_bit else str(DTYPE).split('.')[-1]} base, LoRA r={rank}, trainable params {trainable:,}")

    encoded = [encode_example(r) for r in rows]
    lengths = [len(e["input_ids"]) for e in encoded]
    dev_encoded = [encode_example(r) for r in dev]
    total_steps = math.ceil(len(rows) / rows_per_step) * epochs
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and DTYPE == torch.float16)
    rng = random.Random(seed)
    history, step = [], 0
    start = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        for micro in plan_steps(lengths, rows_per_step, token_budget, rng):
            n_target = sum(encoded[i]["target_len"] for mb in micro for i in mb)
            step_loss = 0.0
            for mb in micro:
                batch = {k: v.to(model.device) for k, v in collate([encoded[i] for i in mb]).items()}
                with autocast():
                    loss_sum, _ = answer_loss(model, batch)
                scaler.scale(loss_sum / n_target).backward()
                step_loss += loss_sum.item()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            history.append({"step": step, "epoch": epoch, "loss": step_loss / n_target,
                            "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm, "dev_loss": None})
            print(f"\r{name}: epoch {epoch}/{epochs} step {step}/{total_steps} loss {step_loss / n_target:.4f} "
                  f"({(time.time() - start) / 60:.1f} min)", end="")
        history[-1]["dev_loss"] = dev_loss(model, dev_encoded, token_budget)
        print(f"\n{name}: epoch {epoch} done, dev loss {history[-1]['dev_loss']:.4f}")

    minutes = (time.time() - start) / 60
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    for p in params:
        p.data = p.data.to(torch.float16)
    model.save_pretrained(adapter_dir)
    size_mb = sum(f.stat().st_size for f in adapter_dir.iterdir()) / 1e6
    meta = {"run": name, "train_rows": len(rows), "trainable_params": trainable, "peak_gpu_gb": round(peak, 2),
            "train_minutes": round(minutes, 1), "rank": rank, "epochs": epochs, "lr": lr, "four_bit": four_bit,
            "rows_per_step": rows_per_step, "steps": total_steps, "adapter_mb": round(size_mb, 1)}
    pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: trained in {minutes:.1f} min, peak GPU {peak:.1f} GB, adapter {size_mb:.0f} MB")
    del model, optimizer, scheduler, scaler, params
    free_memory()
    return meta


def train_and_eval(name, rows, change, four_bit=False, model_id=None, force=False, **train_kwargs):
    meta = train_lora(name, rows, four_bit=four_bit, model_id=model_id, force=force, **train_kwargs)
    model = load_model(model_id or BASE_MODEL, four_bit=four_bit, adapter=RUNS / name / "adapter")
    extra = {k: meta[k] for k in ["train_rows", "peak_gpu_gb", "train_minutes", "trainable_params"]}
    preds, res = run_eval(name, model, change=change, extra=extra, force=force)
    del model
    free_memory()
    return meta, preds, res


def plot_training(names, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for name in names:
        log = pd.read_csv(RUNS / name / "train_log.csv")
        axes[0].plot(log["step"], log["loss"].rolling(5, min_periods=1).mean(), label=name)
        ends = log.dropna(subset=["dev_loss"])
        axes[1].plot(ends["epoch"], ends["dev_loss"], marker="o", label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("train loss (5-step mean)")
    axes[0].set_title("Training loss on answer tokens")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("dev loss")
    axes[1].set_title("Dev loss after each epoch")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def experiments_table():
    return pd.read_csv(EXPERIMENTS_CSV).drop(columns=["timestamp"]).set_index("run")
