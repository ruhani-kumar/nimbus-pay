### E6: testing the analysis, with decisions written last

**Hypothesis from the error analysis:** priority errors happen because the model writes `priority` before `amount_paise`. If the target JSON puts the extracted evidence first and the rule-based decisions last, the model can condition on the amount it has already written:

`category, amount_paise, txn_date, txn_id, channel, language, priority, needs_human`

**One change from E5:** the same 1,916 training rows with the same labels and the same hyperparameters. Only the order of the keys in the target string changes. The output is still a single JSON object with exactly the 8 keys and the same values. `score.py` parses the JSON and compares key by key, so the order never affects scoring.

**Prediction:** fewer `amount threshold` failures, and the other failure types roughly unchanged.
