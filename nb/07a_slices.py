BEST_RUN = "E5_aug_lora_r16"


def parsed_predictions(run, kind="dev"):
    preds = {p["id"]: p["output"] for p in load_jsonl(RESULTS / run / f"{kind}_predictions.jsonl")}
    out = {}
    for rid, text in preds.items():
        try:
            y = json.loads(text)
            out[rid] = y if isinstance(y, dict) else None
        except json.JSONDecodeError:
            out[rid] = None
    return out


def is_exact(gold, pred):
    return pred is not None and set(pred) == set(KEYS) and all(type(pred[k]) is type(gold[k]) and pred[k] == gold[k] for k in KEYS)


dev_gold = {r["id"]: json.loads(label_text(r)) for r in dev}
best = parsed_predictions(BEST_RUN)
dev_frame = pd.DataFrame([{
    "id": r["id"],
    "category": dev_gold[r["id"]]["category"],
    "language": dev_gold[r["id"]]["language"],
    "source": ticket_source(r),
    "length": "long (>512 tokens, quoted email)" if prompt_len(r) > 512 else "short",
    "exact": is_exact(dev_gold[r["id"]], best[r["id"]]),
} for r in dev])

for col in ["category", "language", "source", "length"]:
    t = dev_frame.groupby(col)["exact"].agg(n="size", exact="mean").round(3)
    t["wrong"] = (dev_frame.groupby(col)["exact"].size() - dev_frame.groupby(col)["exact"].sum()).astype(int)
    print(f"\n{BEST_RUN} dev exact match by {col}:")
    print(t.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, key, values in [(axes[0], "category", CATEGORIES), (axes[1], "priority", PRIORITIES)]:
    matrix = pd.crosstab(pd.Categorical([dev_gold[i][key] for i in best], values),
                         pd.Categorical([(best[i] or {}).get(key) for i in best], values), dropna=False)
    ax.imshow(np.log1p(matrix.values), cmap="Blues")
    for (a, b), v in np.ndenumerate(matrix.values):
        if v:
            ax.text(b, a, v, ha="center", va="center", fontsize=8, color="white" if a == b else "#c0392b")
    ax.set_xticks(range(len(values)), values, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(values)), values, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(f"{key} confusion on dev ({BEST_RUN.split('_')[0]})")
    print(f"\n{key} confusion (rows gold, columns predicted):")
    print(matrix.to_string())
plt.tight_layout()
plt.savefig(RESULTS / "confusion.png", dpi=150)
plt.show()
