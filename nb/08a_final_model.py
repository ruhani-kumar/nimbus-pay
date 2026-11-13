FINAL_RUN = BEST_RUN
FINAL_ROWS = train_aug
ADAPTER_DIR = ROOT / "adapter"
FINAL_DIR = RUNS / FINAL_RUN

logged = experiments_table().loc[FINAL_RUN]
final_meta = train_lora(FINAL_RUN, FINAL_ROWS, **REFERENCE)
shutil.copytree(FINAL_DIR / "adapter", ADAPTER_DIR, dirs_exist_ok=True)

adapter_files = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(ADAPTER_DIR.iterdir())}
adapter_config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
print("adapter/ files (MB):", adapter_files)
print({k: adapter_config[k] for k in ["base_model_name_or_path", "peft_type", "r", "lora_alpha", "target_modules"]})
assert adapter_config["base_model_name_or_path"] == BASE_MODEL
assert max(adapter_files.values()) < 100
assert final_meta["trainable_params"] <= TRAINABLE_CAP

free_memory()
final_model = load_model(adapter=ADAPTER_DIR)
print(f"fresh {str(DTYPE).split('.')[-1]} base + adapter loaded from {ADAPTER_DIR}")
print("effective generation settings:", effective_generation_config(final_model))

check_ids = tokenizer.apply_chat_template(prompt_messages(test[0]), add_generation_prompt=True)
check_ids = check_ids["input_ids"] if hasattr(check_ids, "keys") else check_ids
assert list(check_ids) == tokenizer(prompt_string(test[0]), add_special_tokens=False)["input_ids"]
print("prompt tokens identical to tokenizer.apply_chat_template(messages, add_generation_prompt=True)")

final_preds, final_res = run_eval(FINAL_RUN, final_model, change=logged["change"], force=True)
print(f"\ndev exact with the reloaded adapter: {final_res['all']['exact_match']:.1%} (logged when the run was trained: {logged['dev_exact']:.1%})")
