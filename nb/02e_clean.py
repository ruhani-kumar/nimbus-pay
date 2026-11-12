import matplotlib.pyplot as plt

dev_ids_leaked = findings["leak_dev_exact"] | findings["leak_dev_same_body"] | findings["leak_test_exact"] | findings["leak_test_same_body"]
train_clean = [
    {"id": r["id"], "messages": [r["messages"][0], r["messages"][1], {"role": "assistant", "content": canonical(fixed_labels[r["id"]])}]}
    for r in train_raw if r["id"] in fixed_labels
]
train_noleak = [r for r in train_raw if r["id"] not in dev_ids_leaked]

CLEAN_DIR = RUNS / "data"
save_jsonl(train_clean, CLEAN_DIR / "train_clean.jsonl")
save_jsonl(train_noleak, CLEAN_DIR / "train_raw_noleak.jsonl")
log.to_csv(ROOT / "cleaning_log.csv", index=False)


def violations(rows):
    out = Counter()
    for r in rows:
        y = parse_label(label_text(r))
        if y is None or list(y) != KEYS:
            out["structure"] += 1
            continue
        if label_text(r) != canonical(y):
            out["format"] += 1
        if not (y["txn_id"] is None or TXN_ID.fullmatch(y["txn_id"])):
            out["txn_id_format"] += 1
        if not (y["txn_date"] is None or ISO_DATE.fullmatch(y["txn_date"])):
            out["date_format"] += 1
        ref = reference_fields(r, y["category"])
        amount = ref["amount_paise"] if type(ref["amount_paise"]) is int else y["amount_paise"]
        pri = rule_priority(y["category"], amount, clean_body(r))
        for k in ["amount_paise", "txn_date", "txn_id", "channel", "language"]:
            if ref[k] != y[k]:
                out[k] += 1
        out["priority"] += pri != y["priority"]
        out["needs_human"] += rule_needs_human(pri, clean_body(r)) != y["needs_human"]
    return out


before, after = violations(train_noleak), violations(train_clean)
summary = pd.DataFrame({"raw (dev leaks removed)": before, "clean": after}).fillna(0).astype(int)
summary.loc["rows"] = [len(train_noleak), len(train_clean)]
print(summary.to_string())

dev_check = violations(dev)
dev_bad = {k: v for k, v in dev_check.items() if v}
print(f"\nreference labeller vs clean dev labels ({len(dev)} rows), disagreements per field: {dev_bad or 'none'}")

users = [norm_text(user_text(r)) for r in train_clean]
assert len(users) == len(set(users)), "duplicates left"
assert not (set(users) & dev_users), "dev leak left"
assert sum(after.values()) == 0, "clean set still breaks a rule"

counts = Counter(code for reason in log["reason"] for code in reason.split(";"))
codes = [c for c, _ in counts.most_common()][::-1]
colors = ["#c0392b" if any(c == d for d, _ in DROP_ORDER) else "#2471a3" for c in codes]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(codes, [counts[c] for c in codes], color=colors)
for i, c in enumerate(codes):
    ax.text(counts[c] + 1, i, str(counts[c]), va="center", fontsize=9)
ax.set_xlabel("training rows")
ax.set_title(f"Where the noise was: {sum(log.action == 'drop')} rows dropped (red), {sum(log.action == 'fix')} rows fixed (blue)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RUNS / "audit_noise.png", dpi=150)
plt.show()

print(f"train_clean: {len(train_clean)} rows -> {CLEAN_DIR / 'train_clean.jsonl'}")
print(f"train_raw_noleak: {len(train_noleak)} rows (only dev leaks removed, for the raw-vs-clean experiment)")
print(f"cleaning_log.csv: {len(log)} rows -> {ROOT / 'cleaning_log.csv'}")
