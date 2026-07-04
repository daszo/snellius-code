---
name: dsi-100k-thread
description: Explains this DSI-QG thesis repo's layout and how to run experiments through run.py and run_scripts/, with a focus on the 100k_thread experiment (data prep -> train -> test). Use when navigating the repo, editing an experiment script, or running/debugging the 100k_thread pipeline, whether on Snellius (SLURM/A100) or the homelab GPU box.
---

# DSI-QG (Enron email search thesis) — repo layout

This is a thesis project adapting **DSI-QG** (differentiable search index with query
generation) to search a personal Enron-style email corpus. It was developed on the
Snellius HPC cluster (SLURM, A100 GPUs) and is now also being run on a homelab machine.
Everything reads from and writes to a single SQLite database, `data/enron.db`.

## Entry point: `run.py`

Every experiment — query generation, DSI training, and generation-time inference — goes
through `run.py`, launched via `torchrun` (or `python -m torch.distributed.launch` in the
older scripts). It's driven by `--task`, one of:

- **`docTquery`** — train a model that generates queries for a document (query generation /
  "d2q" model).
- **`generation`** — run a trained docTquery model over a table/corpus to produce generated
  queries, written back into the DB as `<table_name>_d2q_q<N>` (or a custom name via
  `--same_mid`).
- **`DSI`** — train the actual DSI model (T5/mT5) to map query text -> document id. When
  `--table_name`/`--db_name` are given, `run.py` calls `split_train_validate_test()` to pull
  the table from the DB and write out train/validate/test JSONL files itself. After training
  it also runs `CE/utils/test/gr_evaluation.py`'s `DSIEmailSearchEvaluator` to score the
  held-out test set and save results.
- **`test`** — a partial/legacy path for BM25-style evaluation (see `CE/utils/test/evaluation_BM25.py`).

Key `RunArguments` fields (in `run.py`, on top of standard HF `TrainingArguments`):
`model_name`/`model_path`, `train_file`/`valid_file` (or `table_name`+`db_name` to read from
SQLite instead of files), `max_length`, `id_max_length`, `remove_prompt`, `thread`,
`same_mid`, and the `save_size`/`save_experiment_type`/`save_version` triple used to tag
results when they're written to the `experiment_results` table.

## `run_scripts/` — SLURM launchers that queue experiments

Each `.sh` file here is a self-contained SLURM batch script (`#SBATCH` headers) that
activates the project venv, optionally stages `data/enron.db` onto node-local scratch
(`$TMPDIR`, since Snellius home/project dirs are network-mounted and slow for SQLite), and
calls `run.py` with a fixed set of hyperparameters and a `--table_name`. They're one-shot,
copy-pasted-and-tweaked configs rather than a shared harness — naming tells you the
experiment:

- `train.sh`, `train_2.sh`, `train_after_fix.sh` — early 10k "classic" (non-thread) runs.
- `train_10k.sh` / `train_10k_thread.sh` / `train_10k_thread_same_mid.sh` — 10k-scale runs,
  varying whether email threads are kept intact (`_thread`) and whether the same message-id
  split is reused (`_same_mid`).
- **`train_100k_thread.sh`** — the 100k-scale, thread-preserving run (see below).
- `resume_from_checkpoint.sh` — same shape as `train.sh` but adds `--resume_from_checkpoint`.
- `make_queries.sh` / `make_queries_db.sh` — run `--task generation` standalone against a
  file or a DB table without training.
- `reserve_gpu.sh` / `reserve_train.sh` / `reserve_train_new.sh` / `go_to_node.sh` —
  interactive SLURM session helpers (`srun`/`tmux`) and a `cd` shortcut into the Snellius
  project path.
- `test.sh` — a minimal SLURM sanity check (activates venv, prints `torch.__version__`).

## Focus: the `100k_thread` experiment

This is the current main experiment: 100k emails, sampled with threads kept intact (as
opposed to individual messages), trained as a DSI model with generated queries as input.
It spans three directories and runs in this order:

1. **Data prep** — `pipelines/pre_pipeline.sh` (SLURM) drives:
   - `pipelines/run_stage1.py` (`pipelines.run_stage1 --thread`) — cleans email bodies
     (`pipelines/clean_files.py`) keeping thread history, then TextRank-summarizes
     (`pipelines/run_textrank.py`) into `full_text_rank_thread`.
   - `run.py --task generation --table_name full_text_rank_thread ...` — generates a title/query
     per document with the `t5-headline` model, writing `full_text_rank_thread_d2q_q1`.
   - `pipelines/prep_datasets.py` (`pipelines.prep_datasets --thread`) — stratified-samples
     10k/100k subsets (by `sender_folder`) into the `N10k_thread` / **`N100k_thread`** tables
     via `CE/utils/database.py`'s `load_db`/`write_to_db`.
2. **Train** — **`run_scripts/train_100k_thread.sh`**: copies `enron.db` to `$TMPDIR`, then
   runs `run.py --task DSI --table_name N100k_thread --thread` style training (t5-base,
   `max_length 512`, `save_experiment_type "thread"`, `save_size "100K"`). `run.py` internally
   splits `N100k_thread` into train/validate/test JSONL under `data/` before training, and
   evaluates + saves results (`experiment_results` table) at the end via
   `DSIEmailSearchEvaluator`.
3. **Test / baselines**:
   - `pipelines/test_dsi_100k_thread.sh` runs `test_100k_thread_dsi.py`, which reloads a
     trained checkpoint and reruns the `DSIEmailSearchEvaluator` retrieval + metrics pass
     standalone (useful for re-scoring without retraining).
   - `pipelines/test_bm25_100k_thread.sh` runs `CE/utils/test/evaluation_BM25.py --table_name
     "N100k_thread" --thread` as the sparse-retrieval (BM25) baseline for comparison.

Table naming convention to keep straight: `N<10k|100k>[_thread][_same_mid]` is the sampled
corpus table; appending `_d2q_q<N>` means "with N generated queries per doc joined in".

## Other top-level pieces

- **`CE/`** — cross-encoder / reranking side of the project (`train_ranker.py`, `re-rank.py`,
  `tries.py` for the docid trie used to constrain beam search) plus `CE/utils/`:
  `database.py` (all SQLite read/write/view helpers), `data_utils.py`, `dist_utils.py`,
  `model_utils.py`, `options.py` (older argparse-based CLI, mostly superseded by `run.py`'s
  `HfArgumentParser`), and `CE/utils/test/` (`gr_evaluation.py` — DSI generative-retrieval
  evaluator; `evaluation_BM25.py` — BM25 baseline; `general.py` — shared metric calculator).
- **`data/`** — `enron.db` (the SQLite DB everything reads from; gitignored, large),
  `msmarco_data/`, plus `process_marco.py`/`process_xorqa.py` for the original DSI-QG public
  datasets (mostly unused now that the project targets the Enron corpus).
- **`sql/`** — one-off SQL (`make_as_view.sql`, `make_experiment_results.sql` — schema for the
  `experiment_results` table that `save_result()` writes to).
- **`trainer.py`** — `DSITrainer` (constrained beam search + Hits@1/10 during eval) and
  `DocTqueryTrainer` (query generation / sampling) — both subclass HF `Trainer`.
- **`data.py`** — dataset/collator classes (`IndexingTrainDataset`, `GenerateDataset`,
  `IndexingCollator`, `QueryEvalCollator`) used by `run.py`.
- **`get_data.sh`**, **`download_model.py`** — bootstrap scripts for the original public
  datasets / base HF models; not part of the Enron 100k_thread flow.
- **`wandb/`, `logs/`, `cache/`** — run artifacts; gitignored.

## Snellius vs. homelab differences to watch for

The `run_scripts/*.sh` files are written for Snellius and hardcode:
`ENV_PATH="/gpfs/work5/0/prjs1828/DSI-QG"`, `.venv` activation from that path, `#SBATCH`
directives, and `$TMPDIR` scratch-copying of `enron.db`. On the homelab box there's no SLURM
and no `$TMPDIR` scratch semantics, `local_models/` and `.venv` are gitignored (so they need
to be recreated locally), and `enron.db` lives directly under `data/` rather than being
staged. When adapting a `run_scripts/*.sh` for homelab use, the SLURM header and the
`cp ... $TMPDIR` step are the parts to strip/replace — the actual `torchrun run.py ...`
invocation and `--table_name`/`--db_name` arguments stay the same.
