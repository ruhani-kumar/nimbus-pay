def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def system_text(row):
    return row["messages"][0]["content"]


def user_text(row):
    return row["messages"][1]["content"]


def label_text(row):
    return row["messages"][2]["content"]


def ticket_body(row):
    return user_text(row).split("\n\n", 1)[-1]


def received_date(row):
    return datetime.strptime(user_text(row).split("\n", 1)[0].split()[1], "%Y-%m-%d").date()


def ticket_source(row):
    return user_text(row).split("\n")[1].split(": ", 1)[1].strip()


def canonical(label):
    return json.dumps({k: label[k] for k in KEYS}, ensure_ascii=False)


_spec = importlib.util.spec_from_file_location("nimbus_score", PACK / "score.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)


def evaluate(gold_rows, pred_rows):
    return scorer.score(gold_rows, pred_rows)


def show_scores(res, prefixes=None):
    print(f"{'group':28} {'n':>5} {'valid':>7} {'exact':>7} {'fields':>7}")
    for name in sorted(res, key=lambda x: (x != "all", x)):
        if prefixes is not None and name != "all" and not name.startswith(tuple(prefixes)):
            continue
        r = res[name]
        print(f"{name:28} {r['n']:5d} {r['json_valid']:7.1%} {r['exact_match']:7.1%} {r['mean_field_acc']:7.1%}")
    print("per-field accuracy:", {k: round(v, 3) for k, v in res["all"]["fields"].items()})


EXPERIMENTS_CSV = RESULTS / "experiments.csv"
EXPERIMENT_COLUMNS = ["run", "change", "train_rows", "dev_exact", "dev_field_acc", "dev_json_valid",
                      "stress_exact", "stress_field_acc",
                      "peak_gpu_gb", "train_minutes", "trainable_params", "timestamp"]


def log_experiment(record):
    record = {**record, "timestamp": datetime.now().isoformat(timespec="seconds")}
    rows = []
    if EXPERIMENTS_CSV.exists():
        with open(EXPERIMENTS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    old = next((r for r in rows if r["run"] == record["run"]), {})
    record = {**old, **record}
    rows = [r for r in rows if r["run"] != record["run"]]
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows + [record])
    return record


train_raw = load_jsonl(DATA / "train.jsonl")
dev = load_jsonl(DATA / "dev.jsonl")
test = load_jsonl(DATA / "test_inputs.jsonl")
print(len(train_raw), len(dev), len(test))
print(user_text(dev[0]))
print(label_text(dev[0]))
