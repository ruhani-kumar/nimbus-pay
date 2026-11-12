## 2. Chat template and sequence length

At test time the model gets the `system` and `user` messages formatted with Qwen's own chat template, followed by the assistant header. It must then write the JSON and stop. Training uses exactly the same formatting:

`prompt = apply_chat_template([system, user], add_generation_prompt=True)`, then `target = label_json + "<|im_end|>"`

Loss is computed only on `target`. `<|im_end|>` is included so the model learns to stop, because it is the tokenizer's EOS token and `generate` halts on it.

The training sequence length is chosen from the measured token lengths below, not guessed. Long tickets carry quoted email threads, and the test set has them too, so they cannot be cut.
