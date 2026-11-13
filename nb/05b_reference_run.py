train_lengths = [len(encode_example(r)["input_ids"]) for r in train_clean]
train_targets = [encode_example(r)["target_len"] for r in train_clean]
plan = plan_steps(train_lengths, 16, 8192, random.Random(SEED))
padded = sum(len(mb) * max(train_lengths[i] for i in mb) for micro in plan for mb in micro)
shuffled = list(range(len(train_clean)))
random.Random(SEED).shuffle(shuffled)
naive = sum(16 * max(train_lengths[i] for i in shuffled[k:k + 16]) for k in range(0, len(shuffled), 16))
print(f"optimizer steps per epoch: {len(plan)}, forward passes per epoch: {sum(len(m) for m in plan)}, "
      f"largest micro-batch: {max(len(mb) for m in plan for mb in m)} rows")
print(f"padding efficiency: {sum(train_lengths) / padded:.1%} with length grouping vs {sum(train_lengths) / naive:.1%} with random batches")
print(f"LM head runs on {sum(train_targets) / sum(train_lengths):.1%} of token positions (answer tokens only)")

REFERENCE = {"rank": 16, "epochs": 3, "lr": 2e-4, "rows_per_step": 16}
e2_meta, e2_preds, e2 = train_and_eval("E2_clean_lora_r16", train_clean, change="cleaned train, fp16 base, LoRA r16 (reference)", **REFERENCE)
plot_training(["E2_clean_lora_r16"], RESULTS / "training_curves_reference.png")
print(experiments_table().to_string())
