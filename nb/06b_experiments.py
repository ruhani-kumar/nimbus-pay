RUN_ALL_EXPERIMENTS = True

train_e1 = [r for r in train_noleak if r["id"] not in findings["label_not_json"]]
EXPERIMENTS = [
    ("E1_raw_lora_r16", train_e1, "raw train (leaks and unparsable labels removed, nothing fixed)", {}),
    ("E2_clean_lora_r16", train_clean, "cleaned train, fp16 base, LoRA r16 (reference)", {}),
    ("E3_clean_qlora_r16", train_clean, "cleaned train, 4-bit NF4 base (QLoRA)", {"four_bit": True}),
    ("E4_clean_lora_r16_2ep", train_clean, "cleaned train, 2 epochs instead of 3", {"epochs": 2}),
    ("E5_aug_lora_r16", train_aug, f"cleaned train + {len(train_aug_extra)} rule-checked synthetic rows", {}),
]

if RUN_ALL_EXPERIMENTS:
    for name, rows, change, overrides in EXPERIMENTS:
        print(f"\n===== {name}: {change} ({len(rows)} rows)")
        train_and_eval(name, rows, change, stress=dev_stress, **{**REFERENCE, **overrides})
else:
    print("RUN_ALL_EXPERIMENTS is False: showing the results logged in results/ (set it to True to retrain E1, E3, E4, E5: about 2.5 h on a T4)")

names = [n for n, *_ in EXPERIMENTS]
table = experiments_table()
summary = table[["change", "train_rows", "dev_exact", "dev_field_acc", "stress_exact", "stress_field_acc",
                 "peak_gpu_gb", "train_minutes", "trainable_params"]]
print(summary.to_string())


def load_scores(name, kind):
    path = RESULTS / name / f"{kind}_scores.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


fields = pd.DataFrame({n: load_scores(n, "dev")["all"]["fields"] for n in ["B2_base_rules_in_prompt"] + names}).round(3)
print("\nper-field dev accuracy:")
print(fields.to_string())

slices = pd.DataFrame({n: {k.split("=", 1)[1]: v["exact_match"] for k, v in load_scores(n, "stress").items() if k.startswith("slice=")}
                       for n in names}).round(3)
print("\ndev stress set, exact match per transformation:")
print(slices.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
order = ["B1_base_fixed_prompt", "B2_base_rules_in_prompt"] + names
x = np.arange(len(order))
axes[0].bar(x - 0.2, [table.loc[n, "dev_exact"] for n in order], 0.4, label="dev", color="#2471a3")
axes[0].bar(x + 0.2, [table.loc[n, "stress_exact"] if n in names else 0 for n in order], 0.4, label="dev stress", color="#c0392b")
axes[0].set_xticks(x, [n.split("_")[0] for n in order])
axes[0].set_ylabel("exact match")
axes[0].set_ylim(0, 1)
axes[0].set_title("Exact match by run")
axes[0].legend(fontsize=8)
for name in names:
    log = pd.read_csv(RESULTS / name / "train_log.csv").dropna(subset=["dev_loss"])
    axes[1].plot(log["epoch"], log["dev_loss"], marker="o", label=name.split("_")[0])
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("dev loss")
axes[1].set_title("Dev loss after each epoch")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS / "experiments.png", dpi=150)
plt.show()
