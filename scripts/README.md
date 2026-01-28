# `scripts/` — Training & evaluation code (code release)

This folder contains the implementation of our **encoder–decoder** approach for learning **model representations** from query–model evaluation signals and using them for:

- **Correctness prediction:** predict whether a model will answer a query correctly.
- **Routing:** select a good model for a query via top-k accuracy / hit@k.

These scripts are written to work with the **concise EmbedLLM parquet splits** produced by the `data/` folder.

---

## Expected data layout

These scripts assume you have already created the concise parquets via `data/` and placed them under the repository’s dataset directory (relative paths only). The expected layout is:

```

data/
datasets/
embedllm/
concise/
model_order.csv
concise.train.parquet
concise.val.parquet
concise.test.parquet

````

Each parquet contains one row per prompt, with:

- metadata: `prompt_id`, `prompt`, `category_id`, `category`
- one or more embedding columns (e.g., `emb_all_mpnet_base_v2`)
- label columns `label_<model_id>` for all models in `model_order.csv`

> Note: the exact dataset base path is defined by `scripts/config.py` (relative to the repo), and `scripts/data_embedllm.py` loads splits from that base.

---

## Quickstart

### 1) Install dependencies

Minimum requirements:

- `numpy`, `pandas`
- `torch`
- `scikit-learn` (for ROC-AUC)
- `pyarrow` (recommended for parquet)

### 2) Run training (defaults)

From the repository root:

```bash
python scripts/training_script.py
````

By default, training writes checkpoints and logs to a *relative* directory:

```
locus_encoder_decoder/models_<ANCHORS_PER_MODEL>/
```

### 3) Run training with custom flags

`training_script.py` exposes key knobs via `argparse` (with the current default values baked in).

Example: change anchor budget and training length:

```bash
python scripts/training_script.py \
  --anchors-per-model 2048 \
  --noise-end-epoch 800 \
  --max-epochs 1500 \
  --early-stopping-patience 300
```

Example: restrict to a subset of datasets / tasks (comma-separated):

```bash
python scripts/training_script.py \
  --tasks-train mmlu,gsm8k \
  --tasks-val mmlu,gsm8k \
  --tasks-test mmlu,gsm8k
```

Example: restrict to a subset of models (0-based indices into `label_*` columns):

```bash
python scripts/training_script.py \
  --model-idx-train 0,1,2,3,4 \
  --model-idx-val 0,1,2,3,4 \
  --model-idx-test 0,1,2,3,4
```

Example: override output directory:

```bash
python scripts/training_script.py --models-dir ./locus_encoder_decoder/my_run/
```

---

## File guide

### `config.py`

Central configuration used throughout the codebase:

* dataset base directory (concise EmbedLLM parquets) — **relative path**
* default embedding column name (`EMB_COL`)
* RNG seed and device selection (`cuda` / `mps` / `cpu`)
* dtype / AMP flags
* model order CSV path
* optional artifact directory used by auxiliary scripts

If you change `EMB_COL`, ensure that embedding column exists in the concise parquets.

---

### `data_embedllm.py`

Dataset loader for the concise EmbedLLM parquets.

Key behavior:

* loads `concise.{split}.parquet` for `split ∈ {train, val, test}`
* parses the chosen embedding column into a float32 numpy array `X`
* collects `label_<model_id>` columns into a matrix `Y`
* supports optional filtering:

  * by task/dataset name (with prefix merging for `mmlu* → mmlu`, `gpqa* → gpqa`)
  * by a subset of models (`model_indices`, 0-based indices into label columns)

Returns a dictionary containing `X`, `Y`, `category` labels, and bookkeeping info.

---

### `dataset_model_batches.py`

Defines the training dataset and data loaders used for encoder–decoder training.

* `ModelBatchDataset`: iterates over **models** and produces anchor/target query batches per model.
* `make_loaders(...)`: builds train/val/test dataloaders and auxiliary arrays used by evaluation.

Anchor selection uses `select_anchors(...)` from `decoder_and_losses.py`.

---

### `decoder_and_losses.py`

Decoder and anchor-selection utilities.

* `BilinearMLPDecoder`: predicts correctness logits for (model-embedding, query-embedding) pairs.
* `select_anchors(...)`: anchor-query selection strategies (`random` / `variance` / `entropy`).



---

### `encoder_attention_modules.py`

Encoder attention modules used to implement the set encoder.
We motivate design and notation of latent bottleneck attention block and
learned-query attention aggregation block from Lee et. al. : https://arxiv.org/pdf/1810.00825.

This file provides:

* Multihead Attention Block (MAB)
* Transformer-style self-attention block (SAB)
* Latent bottleneck attention block (ISAB)
* Learned query aggregation block (PMA pooling)



---

### `router_model.py`

Defines the full encoder–decoder model and configuration.

* `Config` dataclass: architecture and regularization knobs (including optional anchor subsampling and noise flags)
* `ModelEmbedRouter`: combines the encoder and decoder
* `RouterWrapper`: helper wrapper used when loading pre-trained encoder/decoder modules for evaluation

---

### `evaluation.py`

Evaluation utilities for correctness prediction and routing.

Main functions:

* `embed_all_models(...)`: computes model representations for all (selected) models given anchor queries
* `score_queries_against_models(...)`: computes logits for each query against each model embedding (chunked to avoid OOM)
* `compute_val_metrics(...)`: reports:

  * correctness prediction metrics (micro-pair accuracy, BCE-with-logits per pair, Brier, ROC-AUC)
  * routing metrics (hit@1/3/5)
  * dataset-wise breakdowns (with MMLU/GPQA prefix merging)

---

### `train_utils.py`

One-epoch training routine:

* runs forward pass on batches produced by `dataset_model_batches.py`
* computes BCE-with-logits loss (optionally with `pos_weight`)
* performs gradient clipping and optimizer step

---

### `utils.py`

Small helper utilities:

* `to_device_with_dtype(...)`: moves tensors to the chosen device and casts floating tensors to the configured dtype.

---

### `model_category_lists.py`

Curated lists of model indices grouped by:

* family (e.g., Llama variants, Mistral variants, etc.)
* specialization tags (code/math/safety/etc.)
* organization groups
* size buckets

These lists are used for analysis/ablations and grouping models for experiments.

---

## Outputs produced by `training_script.py`

In `locus_encoder_decoder/models_<ANCHORS_PER_MODEL>/`, you should expect:

* `encoder.pt` — encoder checkpoint (CPU state_dict)
* `decoder.pt` — decoder checkpoint (CPU state_dict)
* `meta.json` / `meta.pt` — training metadata (config, anchor indices, bookkeeping)
* `val_metrics_train_<ANCHORS_PER_MODEL>.csv` — per-epoch summary row
* `training_summary.json` — best checkpoint summary including overall + dataset-wise metrics (and `args` for reproducibility)

---

## Notes

* The code is designed to be **portable**: paths are relative to the repository (no absolute paths).
* For best performance, run on GPU (`cuda`) when available.
* If you override `--emb-col`, ensure the corresponding `emb_*` column exists in the concise parquets.

