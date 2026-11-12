DROP_ORDER = [
    ("leak_dev_exact", "same ticket as a dev row; training on it would leak dev"),
    ("leak_dev_same_body", "same ticket body as a dev row; training on it would leak dev"),
    ("leak_test_exact", "same ticket as a test row"),
    ("leak_test_same_body", "same ticket body as a test row"),
    ("junk_ticket", "placeholder text such as 'test' or 'asdf' with a made-up label"),
    ("label_not_json", "label is cut off and does not parse as JSON"),
    ("duplicate_conflicting_label", "exact duplicate ticket with a different label"),
    ("duplicate_same_label", "exact duplicate of an earlier row"),
]
V1_PRIORITY = {"high": "P1", "medium": "P2", "low": "P3"}

decisions = {}
fixed_labels = {}
for r in train_raw:
    rid = r["id"]
    drop = [code for code, _ in DROP_ORDER if rid in findings[code]]
    if drop:
        decisions[rid] = ("drop", drop[0])
        continue
    y = dict(labels[rid])
    reasons = []
    if "prio" in y:
        y["priority"] = V1_PRIORITY[y.pop("prio")]
        reasons.append("schema_v1_prio_key")
    if rid in findings["date_dd_mm_yyyy"]:
        y["txn_date"] = iso_from_label_date(y["txn_date"])
        reasons.append("date_dd_mm_yyyy")
    if rid in findings["txn_id_not_normalized"]:
        y["txn_id"] = label_txn_id(y["txn_id"])
        reasons.append("txn_id_not_normalized")
    if rid in unit_errors:
        y["amount_paise"] = reference[rid]["amount_paise"]
        reasons.append("amount_in_rupees_not_paise")
    if y["priority"] != reference[rid]["priority"]:
        y["priority"] = reference[rid]["priority"]
        reasons.append("priority_breaks_rules")
    if y["needs_human"] != reference[rid]["needs_human"]:
        y["needs_human"] = reference[rid]["needs_human"]
        reasons.append("needs_human_breaks_rules")
    fixed_labels[rid] = {k: y[k] for k in KEYS}
    if reasons:
        decisions[rid] = ("fix", ";".join(reasons))

REASON_TEXT = dict(DROP_ORDER) | {
    "schema_v1_prio_key": "old v1 schema: key 'prio' with low/medium/high; renamed to 'priority' and mapped to P3/P2/P1",
    "date_dd_mm_yyyy": "txn_date written DD/MM/YYYY; converted to YYYY-MM-DD",
    "txn_id_not_normalized": "txn_id had lowercase, spaces or hyphens; normalized to NP + 10 digits",
    "amount_in_rupees_not_paise": "amount stored in rupees (decimals cut off); replaced with the paise amount stated in the ticket",
    "priority_breaks_rules": "priority contradicts the spec given category, amount and legal threat; recomputed",
    "needs_human_breaks_rules": "needs_human contradicts the spec; recomputed",
}

log = pd.DataFrame([(rid, a, reason) for rid, (a, reason) in sorted(decisions.items())], columns=["id", "action", "reason"])
print(log["action"].value_counts().to_string())
reason_counts = Counter(code for reason in log["reason"] for code in reason.split(";"))
print(pd.DataFrame([(c, n, REASON_TEXT[c]) for c, n in reason_counts.most_common()], columns=["reason", "rows", "meaning"]).to_string(index=False))
