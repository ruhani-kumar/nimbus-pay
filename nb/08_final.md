## 7. Final model and test predictions

**Final model: E5** (cleaned train plus 620 rule-checked synthetic rows; fp16 Qwen2.5-3B-Instruct; LoRA r16; 3 epochs). It has the best dev score, the best stress score and a significant improvement over the E2 reference. E6 matched it but did not beat it, and E5 keeps the schema's key order.

The test predictions are produced the way the reviewers will produce them:

1. The adapter is copied to `adapter/`. It is saved in fp16 (≈60 MB), so it fits on GitHub without LFS.
2. A **fresh** fp16 base model is loaded, and `PeftModel.from_pretrained(base, "adapter")` is applied on top. Nothing from training stays in memory.
3. Every test ticket is generated **one at a time** with the chat template, `do_sample=False` and `max_new_tokens=200`. The output is only stripped of whitespace.

Reproducibility checks:

- **Batching:** the same 400 test tickets are generated again in length-grouped batches, and the outputs are compared with the one-at-a-time outputs.
- **Reloading:** dev is scored again with the reloaded adapter. On a fresh run, where E5 is retrained here, this shows how closely the retrained model reproduces the logged score.
- **Decoding default:** Qwen's `generation_config` sets `repetition_penalty=1.05`, which still applies under greedy decoding. Dev is also scored with it set to 1.0, so we know whether a reviewer's setting could change our results.
- **Output check:** every test output is checked against the schema (valid JSON, exactly the 8 keys, allowed values). The outputs are **reported, never modified**.
- **Official scorer:** `score.py` is run from the command line on the final dev predictions.
