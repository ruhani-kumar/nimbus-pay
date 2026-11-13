## 5. Part D: controlled experiments

Every run starts from the **E2 reference** (cleaned train, fp16 base, LoRA r16, 3 epochs, LR 2e-4, seed 42) and changes exactly one thing:

| Run | One change | Question it answers |
|---|---|---|
| E1 | raw train: only dev/test leaks and unparsable labels removed, no fixes, no deduplication | What does the label noise found in the audit cost? |
| E2 | none (reference) | |
| E3 | base loaded in 4-bit NF4 (QLoRA) | How much accuracy do we trade for memory? |
| E4 | 2 epochs instead of 3 | Is the third epoch worth its 8 minutes? |
| E5 | + 620 synthetic rows from rule-based augmentation | Does teaching the rules on harder surface forms help? |

**E5, rule-based augmentation (my own idea).** The test set is described as harder. Instead of guessing at its contents, I rewrite **train** tickets in the ways the spec says make a ticket hard, and compute the new label from the rules:

- **txn_id_format**: `NP1234567890` becomes `np 12345 67890`, `N P 1234 567 890`, `Np-123-456-7890`.
- **amount_threshold**: the amount is replaced by one at or around the ₹5,000 and ₹50,000 priority cut-offs, written as `Rs.4999/-`, `₹49,999`, `49.5k` or `1.5 lakh`. Priority and needs_human are recomputed.
- **relative_date**: the date becomes `yesterday`, `parso` or `6 din pehle`, and `txn_date` is recomputed from Received.
- **distractor**: a balance, limit, fee or bank reference sentence that must be ignored.
- **quoted_email**: a quoted earlier email with an old date and amount that must be ignored.
- **legal_threat**: a sentence mentioning RBI, a lawyer or consumer court, so the ticket becomes P1 and needs a human.
- **repeat_contact**: a sentence such as "third time" or "baar baar", so the ticket needs a human.

Two safeguards keep this from adding new label noise:
- Only train tickets where the audit's reference labeller agrees with the cleaned label on **all eight fields** are rewritten.
- Every rewritten ticket is labelled again independently by the reference labeller. It is kept only if that label matches the label the transformation produced.

The labeller is used only to build training data and is never part of inference.

**Dev stress set.** The same seven transformations applied to dev tickets give a second benchmark of 175 rows. It is used only for evaluation and never for training or tuning. It shows how each model holds up on the kinds of variation the test set is said to contain, and it is scored per transformation through the scorer's `slice` field.

**Running cost.** E1–E5 together take about 2.5 hours on a T4. With `RUN_ALL_EXPERIMENTS = False` the notebook shows the logged results from `results/` instead of retraining, so a fresh top-to-bottom run stays under the 3-hour limit. All runs are saved, so a Colab disconnect only costs the run in progress.
