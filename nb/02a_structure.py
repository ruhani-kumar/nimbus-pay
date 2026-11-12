import re
from collections import Counter, defaultdict

import pandas as pd

FIXED_SYSTEM = system_text(dev[0])
findings = defaultdict(set)


def flag(code, rid):
    findings[code].add(rid)


def parse_label(text):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def norm_text(s):
    return re.sub(r"\s+", " ", s).strip().lower()


ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
DMY_DATE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
TXN_ID = re.compile(r"NP\d{10}")

labels = {}
for r in train_raw:
    rid = r["id"]
    if [m["role"] for m in r["messages"]] != ["system", "user", "assistant"]:
        flag("bad_roles", rid)
    if system_text(r) != FIXED_SYSTEM:
        flag("bad_system_prompt", rid)
    y = parse_label(label_text(r))
    if y is None:
        flag("label_not_json", rid)
        continue
    labels[rid] = y
    if "prio" in y:
        flag("schema_v1_prio_key", rid)
    elif set(y) != set(KEYS):
        flag("wrong_keys", rid)
    if y.get("category") not in CATEGORIES:
        flag("bad_category_value", rid)
    if "priority" in y and y["priority"] not in PRIORITIES:
        flag("bad_priority_value", rid)
    a = y.get("amount_paise")
    if a is not None and type(a) is not int:
        flag("amount_wrong_type", rid)
    d = y.get("txn_date")
    if d is not None:
        if isinstance(d, str) and DMY_DATE.fullmatch(d):
            flag("date_dd_mm_yyyy", rid)
        elif not (isinstance(d, str) and ISO_DATE.fullmatch(d)):
            flag("date_bad_format", rid)
        elif d > str(received_date(r)):
            flag("date_after_received", rid)
    t = y.get("txn_id")
    if t is not None and not (isinstance(t, str) and TXN_ID.fullmatch(t)):
        flag("txn_id_not_normalized", rid)
    if y.get("channel") not in [None, *CHANNELS]:
        flag("bad_channel_value", rid)
    if y.get("language") not in LANGUAGES:
        flag("bad_language_value", rid)
    if type(y.get("needs_human")) is not bool:
        flag("needs_human_wrong_type", rid)
    if y.get("category") in NON_TXN_CATEGORIES and any(y.get(k) is not None for k in TXN_FIELDS):
        flag("txn_fields_on_non_txn_category", rid)

dev_users = {norm_text(user_text(r)) for r in dev}
test_users = {norm_text(user_text(r)) for r in test}
dev_bodies = {norm_text(ticket_body(r)) for r in dev}
test_bodies = {norm_text(ticket_body(r)) for r in test}
for r in train_raw:
    u, b = norm_text(user_text(r)), norm_text(ticket_body(r))
    if u in dev_users:
        flag("leak_dev_exact", r["id"])
    elif b in dev_bodies:
        flag("leak_dev_same_body", r["id"])
    if u in test_users:
        flag("leak_test_exact", r["id"])
    elif b in test_bodies:
        flag("leak_test_same_body", r["id"])

groups = defaultdict(list)
for r in train_raw:
    groups[norm_text(user_text(r))].append(r)
dup_groups = [g for g in groups.values() if len(g) > 1]
for g in dup_groups:
    distinct = {label_text(r) for r in g}
    for r in g[1:]:
        flag("duplicate_same_label" if len(distinct) == 1 else "duplicate_conflicting_label", r["id"])

JUNK = re.compile(r"(test|asdf|qwerty|hello|hi|\d|\W|_)*", re.I)
for r in train_raw:
    body = re.sub(r"^(\[IVR transcript\]|@NimbusPayCare)", "", ticket_body(r).strip()).strip()
    if len(body) < 20 and JUNK.fullmatch(body):
        flag("junk_ticket", r["id"])

print(f"train rows {len(train_raw)}, parsed labels {len(labels)}, duplicate groups {len(dup_groups)}")
structure_table = pd.DataFrame(
    [(k, len(v), ", ".join(sorted(v)[:4])) for k, v in sorted(findings.items(), key=lambda x: -len(x[1]))],
    columns=["finding", "rows", "examples"],
)
structure_table
