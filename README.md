# NimbusPay ticket triage: LoRA fine-tune

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Loaded in 4-bit | **No.** fp16 base, LoRA weights trained in fp32 and saved in fp16. 4-bit QLoRA was tested as experiment E3. |
| Adapter | `adapter/`: PEFT LoRA, r16, alpha 32, all linear layers, 29,933,568 trainable parameters, 60 MB |
| Best result | **Dev: 95.5% exact match, 99.1% mean field accuracy, 100% valid JSON** (E5: cleaned train plus 620 rule-checked synthetic rows). Significantly better than the cleaned-data reference E2 (92.0%, McNemar p = 0.039). |

## Repository layout

| Path | What it is |
|---|---|
| `notebook.ipynb` | The full pipeline with outputs: audit, sequence length, baselines, training, experiments, error analysis, final model, test predictions |
| `report.md` | Audit findings, experiment table, error analysis, next steps, AI tools used |
| `cleaning_log.csv` | Every dropped or fixed training row (`id,action,reason`): 149 dropped, 175 fixed |
| `predictions_test.jsonl` | Raw model outputs for the 400 test tickets, in `test_inputs.jsonl` order |
| `adapter/` | The final LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`) |
| `results/` | Logged results for every run: dev and stress predictions, scores, training logs, charts, `experiments.csv`, failure tables |
| `nb/`, `tools/build_notebook.py` | The notebook's cell sources and the script that assembles `notebook.ipynb` from them |
| `requirements.txt` | Pinned library versions |

The training data and the base model are not included.

## Loading the adapter and generating

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.float16, device_map="cuda")
model = PeftModel.from_pretrained(base, "adapter").eval()

row = json.loads(open("test_inputs.jsonl").readline())
ids = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**ids, do_sample=False, max_new_tokens=200)
print(tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

The notebook checks that:
- our prompt tokens are identical to `apply_chat_template(messages, add_generation_prompt=True)`
- batched generation and one-ticket-at-a-time generation give identical outputs on all 400 test tickets
- the result is the same with Qwen's default `repetition_penalty=1.05` and with 1.0 (identical on 200/200 dev tickets)

`predictions_test.jsonl` was generated one ticket at a time on a T4 in fp16, with no post-processing beyond `.strip()`.

## Running the notebook on Colab

1. In Google Drive, create `MyDrive/nimbus_triage/` and upload these into it:
   - this repository's `results/` folder
   - the `candidate_pack/` folder with `data/`, `SCHEMA.md` and `score.py`
2. Open `notebook.ipynb` in Colab with a **T4 GPU** runtime and choose **Run all**.
   - The first cell pins the library versions and removes Colab's preinstalled `torchao`, which breaks `peft`.
   - It asks for Drive access.
3. Everything is written to `MyDrive/nimbus_triage/`:
   - `runs/` for adapters and predictions
   - `results/` for small logs and charts
   - `adapter/` and `predictions_test.jsonl` for the final outputs

**Run time.** The notebook runs in **reviewer mode** by default (`RUN_ALL_EXPERIMENTS = False` in section 5). In this mode a fresh run trains:
- the reference E2 (about 24 min)
- the final model E5 (about 33 min)

and reads E1, E3, E4 and E6 from the logs in `results/`. With the baselines, dev scoring and test generation (about 45 min for 400 tickets one at a time), the total is about 2.5 hours on a T4. Set the flag to `True` to retrain every experiment, which adds about 2.5 hours. Finished runs are saved to Drive, so a disconnected session can resume without retraining them.

## Results

| Run | What changed | Dev exact | Dev field acc | Stress exact | Peak GPU | Train min |
|---|---|---|---|---|---|---|
| B1 | base model, fixed prompt | 0.0% | 0.0% | – | – | – |
| B2 | base model, rules in the prompt | 26.5% | 82.5% | – | – | – |
| E1 | raw train | 91.5% | 98.6% | 73.7% | 10.4 GB | 25.3 |
| E2 | cleaned train (reference) | 92.0% | 98.3% | 78.9% | 10.3 GB | 24.0 |
| E3 | 4-bit QLoRA | 94.5% | 98.9% | 78.9% | 8.1 GB | 28.7 |
| E4 | 2 epochs | 86.5% | 97.4% | 70.9% | 10.3 GB | 15.9 |
| **E5** | **+620 synthetic rows (final)** | **95.5%** | **99.1%** | **93.1%** | 10.4 GB | 33.1 |
| E6 | E5, decisions as the last keys | 95.5% | 98.9% | 92.6% | 10.4 GB | 32.6 |

"Stress" is a 175-row set built from dev tickets for evaluation only. [report.md](report.md) has the audit, the significance tests, the error analysis and a test-set distribution shift that the submission does not fix.
