**E6 result: the hypothesis was only partly right, and E5 stays the final model.**

- Amount-threshold failures went **down, as predicted: from 11 to 8**. But new errors appeared:
  - two more payment failures read as fraud
  - one ticket where the model copied the transaction id's digits into `amount_paise`. With the amount written right after the category, the model has seen less of its own output before writing it.
- Exact match did not change: dev **95.5% vs 95.5%** (5 tickets moved each way, McNemar p = 1.0) and stress **92.6% vs 93.1%**.
- The most informative number: even when the amount is written *before* the priority, **11 of 13** priority errors still come with a correctly written amount. So the problem is not generation order. It is the comparison of magnitudes themselves (₹36,737 judged ≥ ₹50,000; ₹4,836 judged ≥ ₹5,000). That points next week's work at item 2, continuous-amount augmentation that covers the whole range near both thresholds, rather than at the output format.
- E6's dev loss (≈0.39 against E5's 0.002) is a measurement artefact, not a sign of over-fitting. Dev loss is computed with teacher forcing against dev labels in the schema's key order, which E6 was trained *not* to produce. Its generated outputs score the same as E5.

E5 keeps the schema's canonical key order and is just as accurate, so **E5 is the final model**.
