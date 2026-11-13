RELATIVE = re.compile(r"yesterday|days? ago|din pehle|\bparso\b|\bkal\b|\btoday\b|\baaj\b", re.I)


def same_value(a, b):
    return type(a) is type(b) and a == b


def as_date(value):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def diagnose(row, gold, pred):
    if pred is None:
        return "invalid JSON", "invalid JSON"
    wrong = [k for k in KEYS if not same_value(pred.get(k), gold[k])]
    text = clean_body(row)
    if "category" in wrong:
        return "category confusion", f"{gold['category']} read as {pred.get('category')}"
    if "priority" in wrong:
        if (gold["category"] in ("payment_failed", "refund") and gold["amount_paise"] and "amount_paise" not in wrong
                and not LEGAL.search(text)):
            return "amount threshold", f"amount ₹{gold['amount_paise'] / 100:,.2f} read correctly, but gave {pred.get('priority')} instead of {gold['priority']}"
        if LEGAL.search(text):
            return "legal threat", "legal threat not turned into P1"
        return "priority rule", f"{gold['priority']} read as {pred.get('priority')}"
    if "txn_date" in wrong:
        g, p, received = as_date(gold["txn_date"]), as_date(pred.get("txn_date")), received_date(row)
        if p is None:
            return "date missed", f"date {gold['txn_date']} not extracted"
        if g is None:
            return "date invented", f"gave {pred.get('txn_date')} where the ticket states no date"
        if p == received:
            return "date: used Received", f"used the Received date {p} instead of {g}"
        if RELATIVE.search(text) and g.month != received.month:
            return "relative date across a month boundary", f"gave {p}, expected {g}; Received {received} is in the next month"
        return "date arithmetic", f"{(p - g).days:+d} days off ({g} expected)"
    if "amount_paise" in wrong:
        return "amount", f"{gold['amount_paise']} read as {pred.get('amount_paise')}"
    if "needs_human" in wrong:
        return ("repeat contact missed", "repeat-contact phrase not flagged") if REPEAT.search(text) else ("needs_human", "set without a reason")
    return "other", ", ".join(wrong)


def failure_table(run):
    out = []
    for kind, rows in [("dev", dev), ("stress", dev_stress)]:
        preds = parsed_predictions(run, kind)
        for r in rows:
            gold, pred = json.loads(label_text(r)), preds.get(r["id"])
            if is_exact(gold, pred):
                continue
            family, detail = diagnose(r, gold, pred)
            out.append({
                "set": kind, "id": r["id"], "cause": family, "detail": detail,
                "wrong fields": "; ".join(f"{k}: {json.dumps(gold[k])} -> {json.dumps((pred or {}).get(k))}" for k in KEYS
                                          if not same_value((pred or {}).get(k), gold[k])),
                "ticket": re.sub(r"\s+", " ", clean_body(r))[:220],
            })
    out = pd.DataFrame(out, columns=["set", "id", "cause", "detail", "wrong fields", "ticket"])
    out.to_csv(RESULTS / f"failures_{run}.csv", index=False)
    return out


def amount_correct_share(table):
    priority_errors = table[table["wrong fields"].str.contains("priority:")]
    return len(priority_errors[~priority_errors["wrong fields"].str.contains("amount_paise:")]), len(priority_errors)


failures = failure_table(BEST_RUN)

print(f"{BEST_RUN}: {sum(failures['set'] == 'dev')} wrong on dev (of {len(dev)}), {sum(failures['set'] == 'stress')} wrong on the stress set (of {len(dev_stress)})\n")
print("root causes:")
print(pd.crosstab(failures["cause"], failures["set"], margins=True, margins_name="total").sort_values("total", ascending=False).to_string())

print("\npriority errors where amount_paise was still exactly right: {}/{}".format(*amount_correct_share(failures)))
print("The canonical key order puts priority (2nd key) before amount_paise (3rd), so the model has to choose the "
      "threshold bucket before it writes out the number it is comparing.")

for _, f in failures.iterrows():
    print(f"\n[{f['set']}] {f['id']}  cause: {f['cause']} ({f['detail']})")
    print(f"  ticket: {f['ticket']}")
    print(f"  wrong:  {f['wrong fields']}")
