from math import comb


def exact_vector(run, gold, kind="dev"):
    preds = parsed_predictions(run, kind)
    return np.array([is_exact(gold[i], preds.get(i)) for i in gold])


def mcnemar_p(only_a, only_b):
    n, k = only_a + only_b, min(only_a, only_b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_comparison(run, reference, gold, kind, resamples=10_000):
    a, b = exact_vector(run, gold, kind), exact_vector(reference, gold, kind)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    ref = reference.split("_")[0]
    return {"run": run, "set": kind, "exact": a.mean(), f"vs {ref}": a.mean() - b.mean(),
            "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "only this right": int((a & ~b).sum()),
            f"only {ref} right": int((~a & b).sum()), "McNemar p": round(mcnemar_p(int((a & ~b).sum()), int((~a & b).sum())), 4)}


stress_gold = {r["id"]: json.loads(label_text(r)) for r in dev_stress}
comparisons = pd.DataFrame([paired_comparison(run, "E2_clean_lora_r16", gold, kind)
                            for kind, gold in [("dev", dev_gold), ("stress", stress_gold)]
                            for run in ["E1_raw_lora_r16", "E3_clean_qlora_r16", "E4_clean_lora_r16_2ep", "E5_aug_lora_r16"]])
print("each run vs the E2 reference on the same tickets (paired bootstrap, 10,000 resamples; exact McNemar test):")
print(comparisons.round(3).to_string(index=False))
