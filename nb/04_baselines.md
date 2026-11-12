## 3. Part B: baselines

Two reference points on dev, with the untouched base model in fp16:

1. **Fixed prompt**: exactly what the fine-tuned model will see (the one-line system prompt and the ticket).
2. **Rules in the prompt**: the full `SCHEMA.md` appended to the system prompt. This is allowed only here, as a reference for how far prompting alone gets.

Inference follows the evaluation protocol: Qwen's chat template, `model.generate` with greedy decoding (`do_sample=False`) and `max_new_tokens=200`, and the output only stripped of whitespace. Everything else stays at the model's own `generation_config`. That includes Qwen's default `repetition_penalty=1.05`, which still applies under greedy decoding. This is what plain `model.generate` does, so it is what the evaluation will do.

Batches are formed from tickets of similar length and left-padded, to keep generation fast on a T4. Step 7 checks that batched outputs equal one-at-a-time outputs.
