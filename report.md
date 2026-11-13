# NimbusPay ticket triage: report

**Final model:** `Qwen/Qwen2.5-3B-Instruct`, loaded in fp16 (not 4-bit), with a LoRA adapter (r16, all linear layers, 29.9M trainable parameters). It was trained on the cleaned train set plus 620 rule-checked synthetic rows. **Dev: 95.5% exact match, 99.1% mean field accuracy, 100% valid JSON.** On a 175-row dev stress set it scores 93.1%. Every number below comes from `notebook.ipynb` and `results/`.

## 1. Data audit (Part A)

I checked `train.jsonl` (1,445 rows) in three layers:
- **Structure:** each label must parse, have the 8 keys, and use valid types and formats. Rows must not repeat a dev or test ticket, and must not be duplicates or junk.
- **Rules:** a reference labeller, written only from SCHEMA.md and train phrasing, recomputes the mechanical fields (priority, needs_human, txn_id, amount, date, channel) and flags every disagreement. It is an audit tool only and is never used at inference. It was checked once against dev afterwards, with no tuning: it agrees with the dev labels on every field of all 200 rows.
- **Consistency:** 123 issue sentences are shared by two or more rows, and none of them gets mixed categories.

| Problem | Rows | Action |
|---|---|---|
| Same ticket as a dev row (leak) | 37 | drop |
| Exact duplicate with the same label | 77 | drop |
| Label cut off, not valid JSON | 25 | drop |
| Junk ticket (`test`, `asdf`) with a made-up label | 10 | drop |
| Old v1 schema: key `prio` with low/medium/high | 70 | fix: rename, map to P3/P2/P1 |
| … of which also have `txn_date` as DD/MM/YYYY | 30 | fix: convert to ISO |
| Priority contradicts the spec (category, amount, legal threat) | 60 | fix: recompute |
| Amount stored in rupees instead of paise | 25 | fix: paise from the ticket |
| txn_id not normalized (lowercase, spaces, hyphens) | 20 | fix: `NP` + 10 digits |

In total, 149 rows are dropped and 175 fixed, and all of them are listed in `cleaning_log.csv`. This leaves **1,296 clean rows**. The rule fixes recompute a field only where the spec defines it mechanically. Category and language were never changed, because no check found them noisy.

## 2. Pipeline (Part C)

- **Model:** Qwen2.5-3B-Instruct. It is a strong instruct model under 4B with a native system role, it handles Hinglish, and it fits a T4 in fp16 with room for activations. The T4 has no bf16, so training uses fp16 autocast with fp32 LoRA weights and a GradScaler.
- **Stack:** PEFT plus a short hand-written PyTorch loop, with no Trainer, TRL or Unsloth. This gives exact control over three things:
  - which tokens get loss
  - fp16 versus fp32 parameters
  - running the LM head only on answer positions. A 151k-entry vocabulary makes full-sequence logits the main memory cost, and the answer is only 29.9% of positions.
- **Chat template:** the prompt is `apply_chat_template([system, user], add_generation_prompt=True)` and the target is `label_json + "<|im_end|>"`. Loss is on the target only, and `<|im_end|>` is included so the model learns to stop. The notebook asserts that prompt plus target tokenizes identically to the full template on 1,296/1,296 rows.
- **Sequence length, from measurement:** the longest train sequence is 2,214 tokens, so `MAX_LEN = 2240` and nothing is truncated. The long tail is the 36 rows with quoted emails (2.8%). The test set has 11 prompts over 512 tokens (longest 2,089), so cutting at 512 would have removed exactly the hard cases. The longest answer is 84 tokens, well within `max_new_tokens=200`.
- **Batching:** each optimizer step takes 16 rows grouped by length, split into micro-batches of at most 8,192 tokens. Padding efficiency is 93.6%, against 24.6% with random batches.
- **Settings:** LR 2e-4 (the usual LoRA value), cosine schedule with 3% warmup, 3 epochs, weight decay 0, gradient clip 1.0, seed 42. Rank 16 with alpha 32 on q, k, v, o, gate, up and down gives 29.9M parameters; r32 would exceed the 50M cap.
- **Inference:** exactly the reviewers' protocol: chat template, greedy decoding, `max_new_tokens=200`, output only stripped of whitespace. Qwen's `generation_config` keeps `repetition_penalty=1.05`, which still applies when decoding greedily. Setting it to 1.0 gives identical outputs on 200/200 dev tickets.

## 3. Baselines and experiments (Parts B and D)

All runs below have 29,933,568 trainable parameters. "Stress" is a dev stress set: 175 rows made from dev tickets by 7 transformations × 25 rows, used only for evaluation and never for training.

| Run | What changed (one thing) | Train rows | Dev exact | Dev field acc | Stress exact | Peak GPU | Train min |
|---|---|---|---|---|---|---|---|
| B1 | base model, fixed prompt | – | 0.0% | 0.0% | – | – | – |
| B2 | base model, SCHEMA.md in the prompt | – | 26.5% | 82.5% | – | – | – |
| E1 | raw train (only leaks and unparsable labels removed) | 1,383 | 91.5% | 98.6% | 73.7% | 10.4 GB | 25.3 |
| **E2** | **cleaned train, fp16 LoRA r16 (reference)** | 1,296 | 92.0% | 98.3% | 78.9% | 10.3 GB | 24.0 |
| E3 | 4-bit NF4 base (QLoRA) | 1,296 | 94.5% | 98.9% | 78.9% | 8.1 GB | 28.7 |
| E4 | 2 epochs instead of 3 | 1,296 | 86.5% | 97.4% | 70.9% | 10.3 GB | 15.9 |
| **E5** | **+620 rule-checked synthetic rows (own idea)** | 1,916 | **95.5%** | **99.1%** | **93.1%** | 10.4 GB | 33.1 |
| E6 | E5 data, `priority` and `needs_human` moved to the last keys | 1,916 | 95.5% | 98.9% | 92.6% | 10.4 GB | 32.6 |

B1 gives 0.5% valid JSON: without tuning the model wraps its answer in a markdown fence and invents its own fields. B2 is valid but weak on priority (50.5%) and needs_human (75.5%).

**Significance.** 200 dev rows is a small sample, so every run is compared with E2 on the same tickets, using a paired bootstrap with 10k resamples and an exact McNemar test:
- **E5:** +3.5 points on dev [95% CI +1.0, +6.5], p = 0.039. +14.3 points on stress, p < 0.001.
- **E4:** significantly worse, p = 0.007.
- **E1 (raw vs clean) and E3 (QLoRA):** neither difference is significant.

Cleaning is an honest null result on dev (+0.5 points). On stress the raw model is 5.2 points lower, but that is not significant either: the model learns most of the spec despite the noise. QLoRA saves 2.3 GB at a cost of 4.7 more minutes and no measurable loss of accuracy.

**E5, the augmentation.** It generates 7 kinds of rows from trusted train rows, meaning rows where the reference labeller agrees with the cleaned label:
- txn-id format variants
- amounts near ₹5k and ₹50k
- relative dates
- distractor amounts and bank reference numbers
- quoted earlier emails
- legal threats
- repeat-contact phrases

Each new row is relabelled by the spec and checked again by the labeller, which rejected 19 rows. Only train and SCHEMA.md are used. The stress set uses the same transformations, but on dev tickets, so it measures what the augmentation was meant to fix. The biggest stress gains are quoted emails (32% → 88%), distractors (84% → 96%) and amount thresholds (80% → 92%).

**E6** tests the error analysis below. Making the model write the amount before the priority cut amount-threshold errors from 11 to 8, but added others, so the net effect was zero (McNemar p = 1.0). E6's dev loss of 0.39 is an artefact: dev loss is computed in the schema's key order, which E6 was trained not to produce. **E5 is the final model.**

## 4. Error analysis (Part E, best model E5)

The weakest fields are priority (97.0%) and needs_human (98.0%), followed by category and txn_date (99.0%). By category, fraud is weakest (86.7% exact, 2 of 15). Hinglish scores 95.9% and English 95.4%, so language is not a weakness. E5 gets 9 of 200 dev rows and 12 of 175 stress rows wrong. Every failure is in `results/failures_E5_aug_lora_r16.csv`.

| Ticket | Wrong field | Why |
|---|---|---|
| dv-00011 | priority P2 (gold P1) | ₹84,137 read correctly, but not judged ≥ ₹50,000 |
| dv-00034 | priority P1 (gold P2) | ₹36,737 judged ≥ ₹50,000 |
| dv-00037 | priority | ₹2,84,669 read correctly; the threshold comparison failed |
| dv-00098 | priority | ₹41,500 read correctly; the comparison failed |
| dv-00127 | priority | ₹4,836 judged ≥ ₹5,000 |
| dv-00006 | txn_date 07-01 (gold 06-30) | "day before yesterday" on 2 July: the model subtracted from the day number only |
| dv-00006 (relative-date stress) | txn_date | the same month-boundary mistake, on a different offset |
| dv-00191 (relative-date stress) | txn_date | off by 10 days |
| dv-00087 | txn_date null | a long paragraph with a balance, an id and "4 days ago": the date was dropped |
| dv-00114 (quoted-email stress) | txn_date | gave the Received date instead of "Aug 30" |
| dv-00157 | category account_access (gold fraud) | "asked for my OTP and took my money": the word OTP pulls towards account_access |
| dv-00002 | category other (gold kyc) | "What type of ID do you accept?" reads as a general question |

**Root causes.**
- **Amount thresholds cause 11 of the 21 failures.** In 12 of 12 priority errors the amount itself was right, so the model extracts well but compares magnitudes badly. The augmentation used only 15 fixed amounts, and the misses fall between them.
- **Calendar arithmetic across month boundaries:** 4 failures.
- **Dates lost in long tickets:** 3 failures.
- **Category ambiguity:** 2 failures.

One stress row (dv-00011, txn-id format) was scored on a version with a glued "441Regards" from an augmenter bug. The bug was fixed later; train was unaffected.

**Test-set shift.** The test set does not follow dev. A schema check on the 400 test outputs, which reports problems and modifies nothing, flags 24 outputs:
- **18 of them** are on tickets that give an order number. **26 test tickets do this, against 0 in train and dev.** The model copies the number into `txn_id`, and on kyc and other non-transaction tickets it also fills transaction fields.
- **5** are Hinglish scratch-card tickets ("Scratch card ka inaam…": 8 in test, 0 in train).
- **1** is a "Google Pay" channel.

Other test-only phrasings probably cost well-formed outputs too:
- receipt reference ids: 33 in test, 0 in train
- "last night": 9 in test, 0 in train
- "followed up more times than I can count": 9 in test, 0 in train

I found this by looking at test outputs, so I **did not retrain on it**. E5 is submitted unchanged, and I expect test exact match to fall below dev.

**Checks on the submission.**
- The predictions file was generated one ticket at a time from a freshly loaded base model plus `adapter/`.
- Batched generation gives identical outputs on all 400 tickets.
- The reloaded adapter scores 95.5% on dev, exactly as logged.
- The ids match `test_inputs.jsonl` in order.

## 5. Next week, in order of expected gain

1. **Reference-number and phrasing coverage.** Add augmenters built from SCHEMA.md for order, ticket and bank reference numbers on every category, receipt-style and spaced NP ids, "last night" and "last month on the Nth", and Hinglish offers tickets. Pair each with a dev-stress slice, so the gain is measured on dev and never on test.
2. **Continuous-amount augmentation.** Draw amounts log-uniformly from ₹100 to ₹5 lakh, with extra density within ±10% of each threshold.
3. **A date-arithmetic curriculum.** Use Received dates in the first week of a month, including February and year boundaries.
4. **Hard negatives for category.** Fraud tickets that mention OTP or PIN, and generic KYC questions.
5. **Variance.** Run 3 seeds of E2 and E5, and try Llama-3.2-3B and Phi-3.5-mini in the same pipeline. Differences of 1–2 points on 200 rows are within noise.

## 6. AI tools used

I used **Claude Code (Anthropic, Claude Opus)** as a pair programmer throughout:
- planning the steps
- writing the notebook cells in `nb/` and `tools/build_notebook.py`
- the audit checks, the reference labeller, the augmenters and the training loop
- drafting the notebook text, this report and the README

I ran every notebook cell on Colab myself, read the outputs, and chose what to do at each step: which runs to compare, E5 as the final model, testing E6, and not retraining after the test-set check. Every number in this report was checked against the notebook outputs and `results/`. No AI model or API produced any training label or prediction. All labels come from `train.jsonl`, the rule fixes and the spec-checked augmentation, and all predictions come from the fine-tuned adapter.
