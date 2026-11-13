DECISIONS_LAST = ["category", "amount_paise", "txn_date", "txn_id", "channel", "language", "priority", "needs_human"]
E6_RUN = "E6_aug_decisions_last"


def with_key_order(rows, keys):
    return [{**r, "messages": [r["messages"][0], r["messages"][1],
                               {"role": "assistant", "content": json.dumps({k: json.loads(label_text(r))[k] for k in keys}, ensure_ascii=False)}]}
            for r in rows]


train_e6 = with_key_order(train_aug, DECISIONS_LAST)
assert all(json.loads(label_text(a)) == json.loads(label_text(b)) for a, b in zip(train_aug, train_e6))
print("E5 target:", label_text(train_aug[0]))
print("E6 target:", label_text(train_e6[0]))

if RUN_ALL_EXPERIMENTS:
    train_and_eval(E6_RUN, train_e6, "E5 data, keys reordered so priority and needs_human come last", stress=dev_stress, **REFERENCE)

e6_vs_e5 = pd.DataFrame([paired_comparison(E6_RUN, BEST_RUN, gold, kind) for kind, gold in [("dev", dev_gold), ("stress", stress_gold)]])
print("\nE6 vs E5 on the same tickets:")
print(e6_vs_e5.round(3).to_string(index=False))

e6_failures = failure_table(E6_RUN)
causes = pd.DataFrame({"E5 (schema order)": failures["cause"].value_counts(), "E6 (decisions last)": e6_failures["cause"].value_counts()}).fillna(0).astype(int)
causes.loc["total"] = causes.sum()
print("\nfailures by root cause, dev + stress:")
print(causes.to_string())
print("\npriority errors with amount_paise exactly right: E5 {}/{}, E6 {}/{}".format(*amount_correct_share(failures), *amount_correct_share(e6_failures)))

fields6 = pd.DataFrame({run: load_scores(run, "dev")["all"]["fields"] for run in [BEST_RUN, E6_RUN]}).round(3)
print("\nper-field dev accuracy:")
print(fields6.to_string())

for _, f in e6_failures.iterrows():
    print(f"\n[{f['set']}] {f['id']}  cause: {f['cause']} ({f['detail']})")
    print(f"  wrong:  {f['wrong fields']}")
