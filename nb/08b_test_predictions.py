import subprocess

TEST_PREDICTIONS = ROOT / "predictions_test.jsonl"


def cached_predictions(path, make):
    if path.exists():
        print(f"reusing {path}")
        return load_jsonl(path)
    preds = make()
    save_jsonl(preds, path)
    return preds


test_single = cached_predictions(FINAL_DIR / "test_one_by_one.jsonl",
                                 lambda: generate_predictions(final_model, test, max_batch=1, desc="test, one ticket at a time"))
test_batched = cached_predictions(FINAL_DIR / "test_batched.jsonl",
                                  lambda: generate_predictions(final_model, test, desc="test, batched"))
save_jsonl(test_single, TEST_PREDICTIONS)
shutil.copy(TEST_PREDICTIONS, RESULTS / "predictions_test.jsonl")

batched_by_id = {p["id"]: p["output"] for p in test_batched}
differ = [p["id"] for p in test_single if batched_by_id[p["id"]] != p["output"]]
print(f"\nbatched vs one-at-a-time on {len(test)} test tickets: {len(test) - len(differ)} identical, {len(differ)} differ {differ[:5]}")


def schema_problems(text):
    try:
        y = json.loads(text)
    except json.JSONDecodeError:
        return ["not JSON"]
    if not isinstance(y, dict) or set(y) != set(KEYS):
        return ["wrong keys"]
    problems = []
    allowed = {"category": CATEGORIES, "priority": PRIORITIES, "language": LANGUAGES}
    problems += [k for k, values in allowed.items() if y[k] not in values]
    problems += [] if y["channel"] is None or y["channel"] in CHANNELS else ["channel"]
    problems += [] if y["amount_paise"] is None or type(y["amount_paise"]) is int else ["amount_paise"]
    problems += [] if y["txn_date"] is None or (isinstance(y["txn_date"], str) and ISO_DATE.fullmatch(y["txn_date"])) else ["txn_date"]
    problems += [] if y["txn_id"] is None or (isinstance(y["txn_id"], str) and TXN_ID.fullmatch(y["txn_id"])) else ["txn_id"]
    problems += [] if type(y["needs_human"]) is bool else ["needs_human"]
    if y["category"] in NON_TXN_CATEGORIES and any(y[k] is not None for k in TXN_FIELDS):
        problems.append("txn fields on a non-transaction category")
    return problems


test_ids = [r["id"] for r in test]
assert [p["id"] for p in test_single] == test_ids
issues = {p["id"]: schema_problems(p["output"]) for p in test_single}
print(f"\n{TEST_PREDICTIONS.name}: {len(test_single)} rows, ids match test_inputs.jsonl in order")
print(f"outputs that are valid JSON with the 8 keys and allowed values: {sum(not v for v in issues.values())}/{len(issues)}")
for rid, v in issues.items():
    if v:
        print("  ", rid, v)

test_labels = [json.loads(p["output"]) for p in test_single if "not JSON" not in issues[p["id"]]]
spread = pd.DataFrame({
    "dev gold": pd.Series([dev_gold[i]["category"] for i in dev_gold]).value_counts(normalize=True),
    "test predicted": pd.Series([y.get("category") for y in test_labels]).value_counts(normalize=True),
}).round(3)
print("\ncategory mix, dev gold vs test predictions:")
print(spread.to_string())
print("test predicted priority mix:", pd.Series([y.get("priority") for y in test_labels]).value_counts().to_dict())

rp1 = cached_predictions(FINAL_DIR / "dev_repetition_penalty_1.jsonl",
                         lambda: generate_predictions(final_model, dev, extra_gen={"repetition_penalty": 1.0}, desc="dev, repetition_penalty=1.0"))
rp1_res = evaluate(dev, rp1)
same_outputs = sum(a["output"] == b["output"] for a, b in zip(sorted(rp1, key=lambda p: p["id"]), sorted(final_preds, key=lambda p: p["id"])))
print(f"\ndev with repetition_penalty=1.0: exact {rp1_res['all']['exact_match']:.1%} vs {final_res['all']['exact_match']:.1%} "
      f"with the model default {effective_generation_config(final_model)['repetition_penalty']}; identical outputs on {same_outputs}/{len(dev)} tickets")

print("\nofficial scorer, final dev predictions:")
result = subprocess.run([sys.executable, str(PACK / "score.py"), "--gold", str(DATA / "dev.jsonl"),
                         "--pred", str(RUNS / FINAL_RUN / "dev_predictions.jsonl")], capture_output=True, text=True)
print(result.stdout or result.stderr)

del final_model
free_memory()
print(f"notebook wall-clock so far: {(time.time() - NOTEBOOK_START) / 60:.0f} min")
