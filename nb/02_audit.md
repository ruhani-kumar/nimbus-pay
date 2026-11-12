## 1. Part A: data audit

The training labels came out of a noisy pipeline. The audit has three layers:

1. **Structure**: the label must parse as JSON, have the 8 spec keys, use the right types and formats, not repeat dev or test tickets, and not be a duplicate or a junk ticket.
2. **Rules**: a reference labeller written only from `SCHEMA.md` and the phrasing seen in `train.jsonl` reads each ticket and works out the fields the spec defines mechanically (priority, needs_human, txn_id, amount, date, channel). Every disagreement with the given label is a finding.
3. **Consistency**: tickets that open with the same issue sentence should share a category and a language. Rows that go against their group are flagged.

The reference labeller is only an audit tool. It is never used at inference. It was built from train only. Dev is used once at the end to measure how accurate the labeller is, not to tune it.
