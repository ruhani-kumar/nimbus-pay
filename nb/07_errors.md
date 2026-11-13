## 6. Part E: error analysis

All of the analysis below is on the best run, **E5** (cleaned train plus rule-checked augmentation). It looks at dev, the 200 real tickets, and at the dev stress set. It works from the saved predictions in `results/`, so it needs no GPU.

1. **Where the errors are**: per-field accuracy, slices by category, language, source and ticket length, and confusion matrices for category and priority.
2. **Are the differences real?** Dev has only 200 rows, so one ticket is worth 0.5 points. Each run is compared with E2 on the *same* tickets, with a paired bootstrap confidence interval and an exact McNemar test.
3. **Every failure, with its cause.** Each wrong prediction on dev and on the stress set is listed with the ticket, the wrong fields and a root-cause tag. The tag is assigned by a small set of rules from the spec, not by hand, so it is reproducible.
