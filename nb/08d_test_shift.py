SHIFT_PHRASES = {
    "order number": re.compile(r"order number \d+", re.I),
    "receipt reference": re.compile(r"reference (number )?on the receipt|receipt pe reference", re.I),
    "scratch card ka inaam (Hinglish)": re.compile(r"scratch card ka inaam|srcatch card ka ianam", re.I),
    "last night": re.compile(r"last night", re.I),
    "last month on the Nth": re.compile(r"last month on the \d+", re.I),
    "followed up more times than I can count": re.compile(r"more times than i can count", re.I),
}
splits = {"train": train_raw, "dev": dev, "test": test}
shift = pd.DataFrame({name: {split: sum(bool(p.search(user_text(r))) for r in rows) for split, rows in splits.items()}
                      for name, p in SHIFT_PHRASES.items()}).T
print("tickets containing each phrasing:")
print(shift.to_string())

flagged = [rid for rid, v in issues.items() if v]
test_by_id = {r["id"]: r for r in test}
on_order = [rid for rid in flagged if SHIFT_PHRASES["order number"].search(user_text(test_by_id[rid]))]
print(f"\nflagged test outputs: {len(flagged)}; on tickets with an order number: {len(on_order)}")
for rid in flagged:
    out = json.loads(next(p["output"] for p in test_single if p["id"] == rid))
    print(f"  {rid}  category={out['category']:<15} txn_id={str(out['txn_id']):<14} channel={str(out['channel']):<10} | {user_text(test_by_id[rid]).split(chr(10) * 2, 1)[-1][:90]!r}")
