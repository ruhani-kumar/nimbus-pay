from datetime import timedelta

DATE_SPAN = re.compile(
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+" + MON + r"\b(?:\s+\d{4})?"
    r"|\b" + MON + r"\s+\d{1,2}(?:st|nd|rd|th)?\b(?:,?\s+\d{4})?"
    r"|\b\d+\s*(?:days? ago|din pehle)"
    r"|day before yesterday|\bparso\b|\byesterday\b|\bkal\b|\btoday\b|\baaj\b",
    re.I,
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")
THRESHOLD_RUPEES = [4900, 4999, 5000, 5001, 5100, 49500, 49999, 50000, 50001, 51000, 125000, 150000, 800, 2500, 20000]
PHRASES = {
    "legal": {
        "en": ["If this is not resolved I will escalate to the RBI ombudsman.", "I am sending a legal notice.",
               "My lawyer will contact you about this.", "I will take this to consumer court."],
        "hi-en": ["Warna main consumer court jaunga.", "RBI ko complaint kar dunga.", "Mera vakil legal notice bhejega."],
    },
    "repeat": {
        "en": ["This is the third time I am writing about this.", "I have contacted support about this before.",
               "This is my second complaint on the same issue."],
        "hi-en": ["Main baar baar complaint kar raha hoon.", "Teesri baar likh raha hoon.", "Pehle bhi complaint ki thi."],
    },
}


def indian(n):
    s = str(n)
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + groups + [tail])


def render_amount(rupees, rng):
    styles = [f"Rs {rupees}", f"Rs.{rupees}/-", f"₹{indian(rupees)}", f"INR {indian(rupees)}", f"{rupees} rupees"]
    if rupees % 100 == 0 and rupees < 100_000:
        styles.append(f"{rupees / 1000:g}k")
    if rupees % 1000 == 0 and rupees >= 100_000:
        styles.append(f"{rupees / 100_000:g} lakh")
    return rng.choice(styles)


def split_ticket(row):
    return user_text(row).split("\n\n", 1)


def make_row(row, body, label, kind):
    head = split_ticket(row)[0]
    return {"id": f"{row['id']}-{kind}", "slice": kind,
            "messages": [row["messages"][0], {"role": "user", "content": head + "\n\n" + body},
                         {"role": "assistant", "content": canonical(label)}]}


def rule_label(row, category):
    text = clean_body(row)
    ref = reference_fields(row, category)
    priority = rule_priority(category, ref["amount_paise"], text)
    return {"category": category, "priority": priority, "amount_paise": ref["amount_paise"], "txn_date": ref["txn_date"],
            "txn_id": ref["txn_id"], "channel": ref["channel"], "language": ref["language"],
            "needs_human": rule_needs_human(priority, text)}


def insert_sentence(body, sentence, rng):
    cuts = [m.start() for m in SENTENCE_END.finditer(body)]
    if not cuts:
        return None
    at = rng.choice(cuts[: max(1, len(cuts) - 1)])
    return body[:at] + " " + sentence + body[at:]


def relabel(label, row, body):
    text = clean_body(make_row(row, body, label, "draft"))
    priority = rule_priority(label["category"], label["amount_paise"], text)
    return {**label, "priority": priority, "needs_human": rule_needs_human(priority, text)}


def aug_txn_id(row, body, label, rng):
    matches = list(TXN_ID_LOOSE.finditer(body))
    if not label["txn_id"] or len(matches) != 1:
        return None
    m = matches[0]
    end = m.end() - (len(m.group(1)) - len(m.group(1).rstrip(" \t\r\n-")))
    d = label["txn_id"][2:]
    new = rng.choice([f"np {d[:5]} {d[5:]}", f"NP-{d}", f"N P {d[:4]} {d[4:7]} {d[7:]}", f"np{d}", f"NP#{d}", f"Np-{d[:3]}-{d[3:6]}-{d[6:]}"])
    return body[:m.start()] + new + body[end:], label


def aug_amount(row, body, label, rng):
    if label["amount_paise"] is None:
        return None
    masked = NOT_MONEY.sub(lambda m: " " * len(m.group(0)), body)
    spans = [m.span() for m in MONEY.finditer(masked)
             if to_paise(m.group("a") or m.group("b") or m.group("c"), m.group("am") or m.group("bm")) == label["amount_paise"]]
    if not spans:
        spans = [m.span("d") if m.group("d") else m.span("e") for m in BARE_AMOUNT.finditer(masked)
                 if to_paise(m.group("d") or m.group("e"), None) == label["amount_paise"]]
    if len(spans) != 1:
        return None
    start, end = spans[0]
    end = start + len(body[start:end].rstrip())
    rupees = rng.choice(THRESHOLD_RUPEES)
    new_body = body[:start] + render_amount(rupees, rng) + body[end:]
    return new_body, relabel({**label, "amount_paise": rupees * 100}, row, new_body)


def aug_relative_date(row, body, label, rng):
    matches = list(DATE_SPAN.finditer(body))
    if not label["txn_date"] or len(matches) != 1:
        return None
    m = matches[0]
    days = rng.choice([1, 2, 3, 4, 5, 6, 7, 10])
    if label["language"] == "hi-en":
        phrase = {1: "kal", 2: "parso"}.get(days, f"{days} din pehle")
    else:
        phrase = {1: "yesterday", 2: "the day before yesterday"}.get(days, f"{days} days ago")
    start = m.start() - 3 if body[max(0, m.start() - 3):m.start()].lower() == "on " else m.start()
    new_date = (received_date(row) - timedelta(days=days)).isoformat()
    return body[:start] + phrase + body[m.end():], {**label, "txn_date": new_date}


def aug_distractor(row, body, label, rng):
    rupees = rng.choice([258, 1200, 7450, 18000, 64000, 120000])
    if label["language"] == "hi-en":
        options = [f"Account mein abhi {indian(rupees)} rupees pade hain.", f"Mera daily limit {rng.choice([10, 25, 50])}k hai."]
    else:
        options = [f"FYI my account balance is ₹{indian(rupees)} right now.", f"My daily limit is {rng.choice([10, 25, 50])}k.",
                   f"Bank ref {rng.randint(10 ** 9, 10 ** 11)}.", f"There was a fee of Rs {rng.choice([10, 25, 49])} as well."]
    new_body = insert_sentence(body, rng.choice(options), rng)
    return (new_body, label) if new_body else None


def aug_quoted_email(row, body, label, rng):
    if ticket_source(row) != "email":
        return None
    old = received_date(row) - timedelta(days=rng.randint(20, 60))
    quote = (f"\n\n> On {old.day} {old.strftime('%b %Y')}, NimbusPay Support wrote:\n"
             f"> We have registered your earlier request about a payment of Rs {rng.randint(500, 90000)} "
             f"dated {old.strftime('%d/%m/%Y')} under ticket #{rng.randint(10000, 99999)}.")
    return body.rstrip() + quote, label


def aug_phrase(kind):
    def apply(row, body, label, rng):
        new_body = insert_sentence(body, rng.choice(PHRASES[kind][label["language"]]), rng)
        if new_body is None:
            return None
        new_label = relabel(label, row, new_body)
        return (new_body, new_label) if new_label != label else None
    return apply


AUGMENTERS = {
    "txn_id_format": aug_txn_id,
    "amount_threshold": aug_amount,
    "relative_date": aug_relative_date,
    "distractor": aug_distractor,
    "quoted_email": aug_quoted_email,
    "legal_threat": aug_phrase("legal"),
    "repeat_contact": aug_phrase("repeat"),
}
TRAIN_QUOTA = {"txn_id_format": 90, "amount_threshold": 150, "relative_date": 100, "distractor": 100,
               "quoted_email": 50, "legal_threat": 70, "repeat_contact": 60}
STRESS_QUOTA = {kind: 25 for kind in AUGMENTERS}


def augment(rows, quota, seed):
    rng = random.Random(seed)
    trusted = []
    for r in rows:
        y = json.loads(label_text(r))
        if not any(ln.lstrip().startswith(">") for ln in user_text(r).split("\n")) and rule_label(r, y["category"]) == y:
            trusted.append((r, y))
    out, stats = [], {}
    for kind, n in quota.items():
        made = rejected = 0
        for r, y in rng.sample(trusted, len(trusted)):
            if made == n:
                break
            result = AUGMENTERS[kind](r, split_ticket(r)[1], y, rng)
            if result is None:
                continue
            new = make_row(r, result[0], result[1], kind)
            if rule_label(new, y["category"]) != result[1] or (label_text(new) == label_text(r) and user_text(new) == user_text(r)):
                rejected += 1
                continue
            out.append(new)
            made += 1
        stats[kind] = {"made": made, "rejected by the checker": rejected}
    return out, len(trusted), pd.DataFrame(stats).T


train_aug_extra, trusted_train, aug_stats = augment(train_clean, TRAIN_QUOTA, SEED)
train_aug = train_clean + train_aug_extra
save_jsonl(train_aug, RUNS / "data" / "train_aug.jsonl")
print(f"train rows the rule checker fully agrees with: {trusted_train}/{len(train_clean)}")
print(aug_stats.to_string())
print(f"E5 training set: {len(train_clean)} cleaned + {len(train_aug_extra)} synthetic = {len(train_aug)} rows")

dev_stress, trusted_dev, stress_stats = augment(dev, STRESS_QUOTA, SEED + 1)
print(f"\ndev stress set (evaluation only, never trained on): {len(dev_stress)} rows from {trusted_dev} dev tickets")
print(stress_stats.to_string())

shown = set()
for r in train_aug_extra:
    if r["slice"] in shown:
        continue
    shown.add(r["slice"])
    src = next(x for x in train_clean if x["id"] == r["id"].rsplit("-", 1)[0])
    print(f"\n[{r['slice']}] {src['id']}")
    print("  before:", ticket_body(src).replace("\n", " ")[:260])
    print("  after: ", ticket_body(r).replace("\n", " ")[:260])
    print("  label: ", label_text(r))
