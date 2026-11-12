## 4. Part C: fine-tuning pipeline

A short, hand-written training loop rather than a framework trainer, so every choice is visible and easy to defend:

| Choice | Value | Why |
|---|---|---|
| Base | `Qwen2.5-3B-Instruct` in fp16 | the T4 has no native bf16, and the 3B model fits in fp16 with room to spare |
| Adapter | LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections | ≈30M trainable params, under the 50M cap (r=32 would exceed it) |
| Loss | cross-entropy on the answer JSON and `<|im_end|>` only | the model learns to answer, not to repeat tickets, and learns to stop |
| Batches | 16 tickets per optimizer step, grouped by length, split into micro-batches of at most 8,192 tokens | short tickets do not pay for the padding of quoted-email tickets, and one long row never runs out of memory |
| Logits | the LM head runs only on answer positions | a 152k-word vocabulary over every prompt token would be the largest tensor in training; this skips it |
| Optimizer | AdamW, LR 2e-4, cosine schedule, 3% warmup, no weight decay, gradient clip 1.0, 3 epochs, seed 42 | standard LoRA settings; Part D tests rank and epochs |
| Precision | fp16 autocast with a gradient scaler; LoRA weights kept in fp32 | a stable fp16 recipe for pre-Ampere GPUs |
| Memory | gradient checkpointing | room for the 2,240-token rows |

Each optimizer step averages the loss over every answer token in its 16 tickets, however they are split into micro-batches, so the update does not depend on the batching. Dev loss is measured after each epoch to monitor fitting, never to train.

After training, the adapter is saved in fp16, under 100 MB, the same way it ships in `adapter/`. It is then **loaded fresh onto a clean base model** and scored on dev with the same protocol as the baselines, as a reviewer would do. Every run writes `train_log.csv`, `train_meta.json`, its adapter, its dev predictions and scores under `runs/<name>/`, and appends one row to `runs/experiments.csv`. A run whose adapter is already on Drive is reused, so a Colab disconnect costs nothing.
