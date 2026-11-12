# NimbusPay ticket triage: LoRA fine-tune

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Loaded in 4-bit | TBD |
| Best dev result | TBD |
| Adapter | `adapter/` (PEFT format) |

## Repository layout

| Path | What it is |
|---|---|
| `notebook.ipynb` | Full pipeline with outputs: audit, baselines, training, experiments, error analysis, test predictions |
| `report.md` | Findings, experiment table, error analysis, next steps, AI tool usage |
| `cleaning_log.csv` | Every dropped or fixed training row (`id,action,reason`) |
| `predictions_test.jsonl` | Raw model outputs for the 400 test tickets |
| `adapter/` | Final LoRA adapter |
| `nb/`, `tools/build_notebook.py` | Notebook cell sources and the script that assembles `notebook.ipynb` |

## Running on Colab

1. Upload `candidate_pack/` to `MyDrive/nimbus_triage/candidate_pack/` in Google Drive.
2. Open `notebook.ipynb` in Colab with a T4 GPU runtime.
3. Run all cells. Runs, adapters and predictions are saved to `MyDrive/nimbus_triage/runs/`.

## Loading the adapter

TBD
