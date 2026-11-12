SCHEMA_TEXT = (PACK / "SCHEMA.md").read_text(encoding="utf-8")
RULES_SYSTEM = FIXED_SYSTEM + "\n\nFollow this labelling spec exactly and output only the JSON object.\n\n" + SCHEMA_TEXT
print(f"rules prompt: {len(tokenizer(RULES_SYSTEM)['input_ids'])} tokens added to every request")

free_memory()
base = load_model()
print("effective generation settings:", effective_generation_config(base))

print("\nBaseline 1: fixed prompt")
b1_preds, b1 = run_eval("B1_base_fixed_prompt", base, change="base model, fixed prompt, no fine-tuning")

print("\nBaseline 2: rules in the prompt")
b2_preds, b2 = run_eval("B2_base_rules_in_prompt", base, system=RULES_SYSTEM, change="base model, SCHEMA.md in system prompt",
                        token_budget=16_000, max_batch=8)

baseline_table = pd.DataFrame({
    "B1 fixed prompt": {"json_valid": b1["all"]["json_valid"], "exact_match": b1["all"]["exact_match"],
                        "mean_field_acc": b1["all"]["mean_field_acc"], **b1["all"]["fields"]},
    "B2 rules in prompt": {"json_valid": b2["all"]["json_valid"], "exact_match": b2["all"]["exact_match"],
                           "mean_field_acc": b2["all"]["mean_field_acc"], **b2["all"]["fields"]},
}).round(3)
print(baseline_table.to_string())

for name, preds in [("B1", b1_preds), ("B2", b2_preds)]:
    print(f"\n{name} sample outputs:")
    for p, g in list(zip(preds, dev))[:3]:
        print(f"  {p['id']} gold: {label_text(g)}")
        print(f"  {' ' * len(p['id'])} pred: {p['output'][:300]!r}")

del base
free_memory()
