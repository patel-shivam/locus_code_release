# `data/` — EmbedLLM dataset preparation (code release)

This folder contains everything needed to:
1) download the **minimal** EmbedLLM metadata from Hugging Face (**only CSVs / ordering files**),  
2) convert each split into a **concise, pivoted** format, and  
3) optionally compute **local sentence-encoder embeddings** for each prompt (instead of downloading large precomputed embedding artifacts).

All paths are **portable** and **relative** to the repository root (no absolute paths).

---

## What gets stored (directory layout)

After running the download + build steps, you will have:

```

data/
datasets/
embedllm/
raw/
train.csv
val.csv
test.csv
model_order.csv
question_order.csv
concise/
model_order.csv
concise.train.parquet
concise.val.parquet
concise.test.parquet
(optional) concise.*.csv

````

### `raw/` contents
Downloaded directly from the Hugging Face dataset repo:
- `train.csv`, `val.csv`, `test.csv`: query–model correctness rows (+ prompt text and dataset/category metadata)
- `model_order.csv`: stable ordering of models (used to align label columns)
- `question_order.csv`: stable ordering of prompts/questions (optional but useful)

**Important:** We intentionally do **not** download large `.pth` artifacts (e.g., `train_x.pth`, `train_y.pth`) because this repo recomputes embeddings locally.

### `concise/` contents
One file per split:
- `concise.{split}.parquet` where `split ∈ {train,val,test}`

Each parquet row corresponds to a single `prompt_id` (one prompt), with:
- meta columns: `prompt_id`, `prompt`, `category_id`, `category`
- optional embedding columns: `emb_<encoder_slug>` (vector per prompt) or flattened `emb_<slug>_<dim>`
- label columns: `label_<model_id>` for every model in `model_order.csv`

---

## Quickstart

### 0) Install dependencies
You will need (at minimum):
- `huggingface_hub`
- `pandas`, `numpy`
- `sentence-transformers` (only if computing embeddings)
- parquet engine: `pyarrow` (recommended) or `fastparquet`

Example:
```bash
pip install huggingface_hub pandas numpy pyarrow sentence-transformers
````

If you only want to build the pivoted dataset without embeddings:

```bash
pip install huggingface_hub pandas numpy pyarrow
```

---

### 1) Download minimal EmbedLLM CSV metadata

This downloads only:
`train.csv`, `val.csv`, `test.csv`, `model_order.csv`, `question_order.csv`.

```bash
python data/download_embedllm_metadata.py
```

This writes to:
`data/datasets/embedllm/raw/`

**Reproducibility tip:** pin a revision/commit hash:

```bash
python data/download_embedllm_metadata.py --revision <COMMIT_OR_TAG>
```

---

### 2) Build concise splits (pivoted labels), optionally with embeddings

#### Option A: Build concise splits **with embeddings**

This computes local embeddings for each prompt using multiple SentenceTransformer models and stores them in the parquet output:

```bash
python data/build_concise_dataset.py --device cuda
```

If you don’t have a GPU:

```bash
python data/build_concise_dataset.py --device cpu
```

#### Option B: Build concise splits **without embeddings**

This produces only the pivoted label matrix + meta columns:

```bash
python data/build_concise_dataset.py --skip-embeddings
```

---

## What each file does

### `download_embedllm_metadata.py`

Downloads only the small CSV metadata files from the Hugging Face dataset repository (and **avoids** the large `.pth` files).

**Output:**

* `data/datasets/embedllm/raw/{train,val,test}.csv`
* `data/datasets/embedllm/raw/model_order.csv`
* `data/datasets/embedllm/raw/question_order.csv`

**Usage:**

```bash
python data/download_embedllm_metadata.py
python data/download_embedllm_metadata.py --revision <REVISION>
python data/download_embedllm_metadata.py --out-dir data/datasets/embedllm/raw
```

---

### `make_concise_split.py`

Converts one split (`train`, `val`, or `test`) from the raw row format into a concise pivoted format.

Key behaviors:

* Reads `raw/{split}.csv` and `raw/model_order.csv`
* Keeps only the last occurrence of each `(prompt_id, model_id)` pair
* Produces one row per `prompt_id`
* Creates label columns `label_<model_id>` in the exact order given by `model_order.csv`
* Optionally computes embeddings for each prompt using SentenceTransformers, producing `emb_<slug>` columns

**Output (default):**

* `data/datasets/embedllm/concise/concise.<split>.parquet`
* also copies `model_order.csv` into `concise/` for convenience

**Usage:**

```bash
python data/make_concise_split.py --split train
python data/make_concise_split.py --split val --skip-embeddings
python data/make_concise_split.py --split test --device cuda
```

**Optional flags:**

* `--skip-embeddings`: only meta + labels
* `--flat-embeddings`: expands embedding vectors into many scalar columns
* `--csv`: also writes a CSV version (parquet is preferred)

---

### `build_concise_dataset.py`

Convenience wrapper that runs `make_concise_split.py` for all three splits: `train`, `val`, `test`.

**Output:**

* `concise.train.parquet`, `concise.val.parquet`, `concise.test.parquet` in `data/datasets/embedllm/concise/`

**Usage:**

```bash
python data/build_concise_dataset.py --device cuda
python data/build_concise_dataset.py --skip-embeddings
python data/build_concise_dataset.py --flat-embeddings
```

---

### `examine_pth.py`

Utility script to inspect `.pth` files (shapes/types) if you happen to have them locally.
This repo does not require `.pth` files and does not download them by default.

Security behavior:

* Attempts `torch.load(..., weights_only=True)` first
* Refuses unsafe pickle loading unless `--i-trust-this-file` is provided

**Usage:**

```bash
python data/examine_pth.py
python data/examine_pth.py --root data/datasets/embedllm/raw
python data/examine_pth.py --i-trust-this-file
```

---

## Notes / troubleshooting

* **Hugging Face auth:** If the dataset ever requires authentication, set `HF_TOKEN` in your environment or log in via `huggingface-cli login`. (Most public datasets won’t require this.)
* **Parquet engine:** If you see errors writing parquet, install `pyarrow`.
* **Embedding compute time:** Computing embeddings for all splits can take time. Use `--skip-embeddings` if you only need labels, or run on GPU with `--device cuda`.
* **Disk size:** Parquet with embeddings can be larger than without embeddings. Using `--flat-embeddings` can increase file size substantially.

---

## Typical end-to-end commands

Minimal (labels only):

```bash
python data/download_embedllm_metadata.py
python data/build_concise_dataset.py --skip-embeddings
```

Full (labels + embeddings):

```bash
python data/download_embedllm_metadata.py
python data/build_concise_dataset.py --device cuda
```

