### What the errors say

E5 gets 9 of 200 dev tickets wrong and 12 of 175 stress tickets wrong. Every failure falls into one of four causes:

1. **Amount thresholds: 11 of 21 failures.** In **12 of 12** priority errors, `amount_paise` is *exactly right*. The model reads ₹84,137 correctly and still says P2, or reads ₹4,836 and still says P2. So the failure is not in extraction but in the comparison with ₹5,000 and ₹50,000. The cause is visible in how the output is generated: the schema's key order puts `priority` (2nd key) **before** `amount_paise` (3rd), so the model has to commit to a threshold bucket before it has written out the number it is comparing. The augmentation also used only 15 fixed amounts. Errors at ₹36,737, ₹41,500, ₹84,137 and ₹2,84,669 fall *between* those values.
2. **Relative dates across a month boundary: 4 failures, plus 1 other date-arithmetic slip.** "Day before yesterday" on 2 July should be 30 June; the model says 1 July. Every off-by-one case crosses into the previous month, so the model treats subtraction as working on the day number, not the calendar.
3. **Dates missed or confused in long tickets: 3 failures.** One ticket has a balance, a txn id, a repeat-contact line and "4 days ago" in a single paragraph, and the model drops the date. In another, the model gave the Received date instead of "Aug 30".
4. **Category ambiguity: 2 failures.** "Someone asked for my OTP and took my money" is fraud, but the word OTP pulls it towards account_access. "What type of ID do you accept?" is KYC, but reads as a general question. Both are hard to tell apart even with the spec in hand.

Nothing fails on language, source or length: Hinglish (95.9%) matches English (95.4%), and the 3 long quoted-email dev tickets are all correct.

### Next week, in order of expected gain

1. **Write the evidence before the decision. (Tested right below as E6.)** Reorder the target keys so `amount_paise`, `txn_date`, `txn_id`, `channel` and `language` come first, and `priority` and `needs_human` come last. JSON key order does not affect the scorer, and the model then compares against a number it has already written. This targets 11 of the 21 failures and costs one training run.
2. **Continuous-amount augmentation.** Draw amounts log-uniformly from ₹100 to ₹5 lakh, with extra density within ±10% of each threshold, instead of 15 fixed values.
3. **A date-arithmetic curriculum.** Add relative dates whose Received date falls on the 1st–7th of a month, including February and year boundaries, so that every one of them crosses a month boundary.
4. **Hard negatives for category.** Fraud tickets that mention OTP, PIN or login, and KYC questions phrased generically.
5. **Model and seed variance.** Try Llama-3.2-3B and Phi-3.5-mini under the same pipeline, and rerun E2 and E5 with 3 seeds. A 1–2 point difference between runs on 200 dev rows is within noise, as the paired tests above show.
