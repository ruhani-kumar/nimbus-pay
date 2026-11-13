def prompt_len(row):
    return len(tokenizer(prompt_string(row), add_special_tokens=False)["input_ids"])


lengths = {
    "train prompt": [prompt_len(r) for r in train_clean],
    "dev prompt": [prompt_len(r) for r in dev],
    "test prompt": [prompt_len(r) for r in test],
    "train answer": [encode_example(r)["target_len"] for r in train_clean],
    "dev answer": [len(tokenizer(label_text(r) + END, add_special_tokens=False)["input_ids"]) for r in dev],
}
pcts = [50, 90, 95, 99, 100]
table = pd.DataFrame({name: np.percentile(v, pcts).round().astype(int) for name, v in lengths.items()}, index=[f"p{p}" for p in pcts]).T
table["mean"] = [round(float(np.mean(v)), 1) for v in lengths.values()]
print(table.to_string())

train_total = [p + a for p, a in zip(lengths["train prompt"], lengths["train answer"])]
quoted = sum(any(ln.lstrip().startswith(">") for ln in user_text(r).split("\n")) for r in train_clean)
long_rows = sum(t > 512 for t in train_total)
print(f"\ntrain rows with a quoted email: {quoted}; train sequences over 512 tokens: {long_rows} ({long_rows / len(train_total):.1%})")
print(f"test prompts over 512 tokens: {sum(t > 512 for t in lengths['test prompt'])}")

MAX_LEN = int(np.ceil(max(train_total) / 64) * 64)
assert max(lengths["train answer"] + lengths["dev answer"]) < MAX_NEW_TOKENS
print(f"\nlongest train sequence {max(train_total)} tokens -> MAX_LEN = {MAX_LEN} (no row is truncated)")
print(f"longest answer {max(lengths['train answer'] + lengths['dev answer'])} tokens, well inside max_new_tokens={MAX_NEW_TOKENS}")
print(f"longest test prompt {max(lengths['test prompt'])} tokens; inference never truncates, so this only needs to fit in memory")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
bins = np.logspace(np.log10(40), np.log10(max(lengths["test prompt"] + train_total) * 1.1), 40)
for name, color in [("train prompt", "#2471a3"), ("dev prompt", "#27ae60"), ("test prompt", "#c0392b")]:
    axes[0].hist(lengths[name], bins=bins, alpha=0.55, label=name, color=color)
axes[0].set_xscale("log")
axes[0].axvline(MAX_LEN, color="black", ls="--", lw=1)
axes[0].text(MAX_LEN, axes[0].get_ylim()[1] * 0.9, f" MAX_LEN={MAX_LEN}", fontsize=8)
axes[0].set_xlabel("prompt tokens (log scale)")
axes[0].set_ylabel("tickets")
axes[0].legend(fontsize=8)
axes[0].set_title("Prompt length: short tickets plus a long quoted-email tail")
axes[1].hist(lengths["train answer"], bins=30, color="#7d3c98")
axes[1].set_xlabel("answer tokens (JSON + <|im_end|>)")
axes[1].set_title("Answer length")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS / "token_lengths.png", dpi=150)
plt.show()
