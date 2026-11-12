def iso_from_label_date(d):
    m = DMY_DATE.fullmatch(d) if isinstance(d, str) else None
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else d


def label_txn_id(t):
    return "NP" + re.sub(r"\D", "", t) if isinstance(t, str) else t


def is_unit_error(label_amount, rule_amount):
    return (type(label_amount) is int and type(rule_amount) is int and label_amount > 0
            and label_amount != rule_amount and rule_amount // 100 == label_amount)


usable = [r for r in train_raw if r["id"] in labels and r["id"] not in findings["junk_ticket"]]
reference = {}
disagree = defaultdict(list)
for r in usable:
    rid, y = r["id"], labels[r["id"]]
    text = clean_body(r)
    ref = reference_fields(r, y["category"])
    amount = ref["amount_paise"] if type(ref["amount_paise"]) is int else y.get("amount_paise")
    ref["priority"] = rule_priority(y["category"], amount, text)
    ref["needs_human"] = rule_needs_human(ref["priority"], text)
    reference[rid] = ref
    given = {**y, "txn_date": iso_from_label_date(y.get("txn_date")), "txn_id": label_txn_id(y.get("txn_id"))}
    for k in ["amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]:
        if given.get(k) != ref[k]:
            disagree[k].append(rid)
    if "priority" in y and y["priority"] != ref["priority"]:
        disagree["priority"].append(rid)

unit_errors = [rid for rid in disagree["amount_paise"] if is_unit_error(labels[rid].get("amount_paise"), reference[rid]["amount_paise"])]
print(f"rows compared: {len(usable)}")
print(pd.DataFrame(
    [(k, len(v), len(v) / len(usable)) for k, v in disagree.items()] + [("amount_paise: of which rupees-not-paise", len(unit_errors), len(unit_errors) / len(usable))],
    columns=["field", "label != rule", "rate"],
).to_string(index=False))

print("\nv1 'prio' value vs rule priority:")
v1_ids = sorted(findings["schema_v1_prio_key"])
print(pd.crosstab(pd.Series([labels[i]["prio"] for i in v1_ids], name="prio"),
                  pd.Series([reference[i]["priority"] for i in v1_ids], name="rule")))

print("\npriority given vs rule (non-v1 rows):")
pri_rows = [r["id"] for r in usable if "priority" in labels[r["id"]]]
print(pd.crosstab(pd.Series([labels[i]["priority"] for i in pri_rows], name="given"),
                  pd.Series([reference[i]["priority"] for i in pri_rows], name="rule")))

bad_pri = set(disagree["priority"])
nh_ok = sum(labels[i]["needs_human"] == reference[i]["needs_human"] for i in bad_pri)
print(f"\nrows whose priority breaks the rules: {len(bad_pri)}; needs_human agrees with the RULE priority on {nh_ok} of them")
print("so the corruption hit priority only, and needs_human still carries the original priority")

GREETING = re.compile(r"^((\[ivr transcript\]|@nimbuspaycare|urgent!!|hi team|hello ji|good morning|dear nimbuspay support|namaste|hello|hey|bhai|sir|hi)\s*[,!]?\s*)+", re.I)


def issue_sentence(row):
    s = sentences(GREETING.sub("", re.sub(r"\s+", " ", clean_body(row)).strip()))
    out = s[0] if s else ""
    if len(out.split()) < 4 and len(s) > 1:
        out += " " + s[1]
    return out.lower()


by_issue = defaultdict(list)
for r in usable:
    by_issue[issue_sentence(r)].append(r["id"])
multi = {k: v for k, v in by_issue.items() if len(v) > 1}
conflicts = {k: Counter(labels[i]["category"] for i in v) for k, v in multi.items() if len({labels[i]["category"] for i in v}) > 1}
print(f"\ncategory consistency: {len(multi)} issue sentences shared by 2+ rows, {len(conflicts)} with mixed categories")
for k, c in conflicts.items():
    print("  ", dict(c), "|", k[:100])

for k in ["priority", "amount_paise"]:
    print(f"\nexamples, {k}:")
    for rid in disagree[k][:5]:
        r = next(x for x in usable if x["id"] == rid)
        print(f"  {rid} given={labels[rid].get(k)} rule={reference[rid][k]} | {re.sub(chr(10), ' ', clean_body(r))[:160]}")
