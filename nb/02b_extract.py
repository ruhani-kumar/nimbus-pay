from datetime import date, timedelta

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"

LEGAL = re.compile(r"\brbi\b|ombudsman|consumer court|lawyer|legal notice|legal action|\bvakil|\bcourt\b", re.I)
REPEAT = re.compile(
    r"third time|teesri baar|again and again|baar baar|contacted support about this before|"
    r"already written \d+ times|pehle bhi complaint|multiple times about this|second complaint|\d+(st|nd|rd|th)? (time|baar)\b",
    re.I,
)
HINGLISH = re.compile(
    r"\b(hai|hain|nahi|mera|mere|meri|maine|kya|kar|karo|karna|ho|hua|gaya|gaye|paise|paisa|tha|thi|kat|"
    r"wapas|abhi|baat|ki|ka|ke|ko|se|mein|bhi|kaise|kab|kyun|jaldi|raha|rahi|hoon|kal|parso|aaj|din|pehle|wala|"
    r"chahiye|lekin|liye|diya|dunga|jaunga|bol|baar|karun|karu|kitna|kyu|lagega|mila|mile|atka|hota|dhanyavaad)\b",
    re.I,
)
DISTRACTOR = re.compile(r"balance|limit|\bfee|charges? of|pade hain|abhi .* hain", re.I)
TXN_ID_LOOSE = re.compile(r"\bn\s*[\-.]?\s*p[\s\-_.:#]*((?:\d[\s\-]*){10})(?!\d)", re.I)
NUM = r"\d[\d,]*(?:\.\d+)?"
MONEY = re.compile(
    rf"(?:(?:rs\.?|inr|₹)\s*(?P<a>{NUM})\s*(?P<am>k\b|lakhs?\b)?(?:/-)?)"
    rf"|(?:(?P<b>{NUM})\s*(?P<bm>k|lakhs?)\b(?:\s*(?:rupees|rupaye|rs))?)"
    rf"|(?:(?P<c>{NUM})\s*(?:rupees|rupaye|rupaiye|rs\b))",
    re.I,
)
BARE_AMOUNT = re.compile(
    rf"(?:amount|rakam|total|value|it was for|it is)\D{{0,12}}?(?P<d>{NUM})(?![\d/\-])"
    rf"|(?P<e>{NUM})\s*(?:is the amount|was the value|in total|kat gaye|ka transaction|ka tha|tha\b)",
    re.I,
)
NOT_MONEY = re.compile(
    r"\bn\s*[\-.]?\s*p[\s\-_.:#]*(?:\d[\s\-]*){10}|#\d+|\b\d{9,}\b|\d+(?:\.\d+){2,}|\d{1,4}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|version \S+|(?:note|iphone|realme|pixel|vivo y|samsung m|samsung a|galaxy s|nord|oneplus)\s*\d+\w*"
    r"|\b\d{1,2}(st|nd|rd|th)?\s+" + MON + r"|" + MON + r"\s+\d{1,2}(st|nd|rd|th)?\b(\s+\d{4})?"
    r"|\d+\s*(days?|din|times|baar)\b|\b20\d\d\b",
    re.I,
)


def clean_body(row):
    body = ticket_body(row)
    lines = [ln for ln in body.split("\n") if not ln.lstrip().startswith(">") and not re.match(r"\s*On .* wrote:\s*$", ln)]
    lines = [ln for ln in lines if not ln.lower().startswith("subject:")]
    return "\n".join(lines)


def sentences(text):
    text = re.sub(r"\b(rs)\.\s*(?=\d)", r"\1 ", text, flags=re.I)
    text = re.sub(r"\s+(?=(fyi|my account balance|account mein abhi|ref:|id:|bank ref)\b)", "\n", text, flags=re.I)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def to_paise(num, mult):
    value = float(num.replace(",", ""))
    mult = (mult or "").lower()
    if mult == "k":
        value *= 1_000
    elif mult.startswith("lakh"):
        value *= 100_000
    return int(round(value * 100))


def extract_amount(text):
    found = []
    for s in sentences(text):
        if DISTRACTOR.search(s):
            continue
        masked = NOT_MONEY.sub(lambda m: " " * len(m.group(0)), s)
        hits = [to_paise(m.group("a") or m.group("b") or m.group("c"), m.group("am") or m.group("bm")) for m in MONEY.finditer(masked)]
        if not hits:
            hits = [to_paise(m.group("d") or m.group("e"), None) for m in BARE_AMOUNT.finditer(masked)]
        found.extend(hits)
    found = list(dict.fromkeys(found))
    return found[0] if len(found) == 1 else (None if not found else "ambiguous")


def latest_on_or_before(day, month, ref):
    for year in (ref.year, ref.year - 1):
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        if d <= ref:
            return d
    return None


def extract_date(text, ref):
    t = text.lower()
    hits = []
    for m in re.finditer(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b", t):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        try:
            hits.append(date(y, mo, d))
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + MON + r"\b(?:\s+(\d{4}))?", t):
        day, mon, yr = int(m.group(1)), MONTHS[m.group(2)], m.group(3)
        hits.append(date(int(yr), mon, day) if yr else latest_on_or_before(day, mon, ref))
    for m in re.finditer(r"\b" + MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?", t):
        mon, day, yr = MONTHS[m.group(1)], int(m.group(2)), m.group(3)
        hits.append(date(int(yr), mon, day) if yr else latest_on_or_before(day, mon, ref))
    for m in re.finditer(r"\b(\d+)\s*(?:days? ago|din pehle)", t):
        hits.append(ref - timedelta(days=int(m.group(1))))
    if re.search(r"day before yesterday|\bparso\b", t):
        hits.append(ref - timedelta(days=2))
    elif re.search(r"\byesterday\b|\bkal\b", t):
        hits.append(ref - timedelta(days=1))
    if re.search(r"\btoday\b|\baaj\b", t):
        hits.append(ref)
    hits = list(dict.fromkeys(h for h in hits if h))
    return hits[0].isoformat() if len(hits) == 1 else (None if not hits else "ambiguous")


def extract_txn_id(text):
    ids = list(dict.fromkeys("NP" + re.sub(r"\D", "", m.group(1)) for m in TXN_ID_LOOSE.finditer(text)))
    return ids[0] if len(ids) == 1 else (None if not ids else "ambiguous")


def extract_channel(text):
    t = text.lower()
    found = set()
    if re.search(r"\bupi\b", t):
        found.add("upi")
    if re.search(r"net\s?banking|internet banking", t):
        found.add("netbanking")
    if re.search(r"\bwallet\b", t):
        found.add("wallet")
    t_card = re.sub(r"\bs\w*c\w*h card", " ", t)
    if re.search(r"\bcard\b|\bvisa\b|mastercard", t_card):
        found.add("card")
    return found.pop() if len(found) == 1 else (None if not found else "ambiguous")


def is_hinglish(text):
    body = re.sub(r"^(\[IVR transcript\]|@NimbusPayCare)\s*", "", text.strip())
    body = re.sub(r"^(hi|hello|hey|sir|bhai|namaste|hello ji|hi team|good morning|dear nimbuspay support|urgent!!)[,!]?\s*", "", body, flags=re.I)
    words = re.findall(r"[a-z]+", body.lower())
    hits = sum(bool(HINGLISH.fullmatch(w)) for w in words)
    return hits >= 2


def rule_priority(category, amount, text):
    amount = amount if type(amount) is int else 0
    if category == "fraud" or amount >= 5_000_000 or LEGAL.search(text):
        return "P1"
    if (category in ("payment_failed", "refund") and amount >= 500_000) or category == "account_access":
        return "P2"
    return "P3"


def rule_needs_human(priority, text):
    return priority == "P1" or bool(REPEAT.search(text))


def reference_fields(row, category):
    text = clean_body(row)
    ref = received_date(row)
    out = {
        "language": "hi-en" if is_hinglish(text) else "en",
        "legal": bool(LEGAL.search(text)),
        "repeat": bool(REPEAT.search(text)),
    }
    if category in NON_TXN_CATEGORIES:
        out.update(amount_paise=None, txn_date=None, txn_id=None, channel=None)
    else:
        out.update(amount_paise=extract_amount(text), txn_date=extract_date(text, ref),
                   txn_id=extract_txn_id(text), channel=extract_channel(text))
    return out


demo = train_raw[0]
print(user_text(demo))
print(reference_fields(demo, "payment_failed"))
