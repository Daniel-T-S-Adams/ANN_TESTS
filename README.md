# ANN Benchmark

Benchmarks brute-force, HNSW, IVF-Flat, and CAGRA nearest-neighbor search using FAISS and cuVS.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` now includes CAGRA dependencies (`cupy-cuda12x`, `cuvs-cu12`) for CUDA 12 environments.

## Datasets

Download a dataset before running benchmarks on it:

```bash
python download_data.py --dataset <DATASET>
```

| Key | Name | Dimensions | Vectors | Download size |
|---|---|---|---|---|
| `glove` | GloVe 6B 100d | 100 | ~400K | ~862 MB |
| `sift` | SIFT1M | 128 | 1M | ~500 MB |
| `gist` | GIST1M | 960 | 1M | ~2.6 GB |
| `sift10m` | SIFT10M | 128 | 10M | ~10 GB |
| `sift100m` | SIFT100M | 128 | 100M | ~10 GB |

Each dataset is stored under `data/<key>/` as `vectors.npy` + `dataset_info.json`.
BigANN-derived datasets (`sift10m`, `sift100m`) are converted with streaming writes, so preprocessing does not require loading the full float32 array in RAM.

## Recall measurement

For approximate algorithms (HNSW, CAGRA), recall@k is estimated automatically at benchmark time by running brute-force exact search on a random sample of queries against the actual database. This avoids any precomputation step and works correctly regardless of database size (`--num-db`) or query count.

By default 2000 queries are sampled. Use `--recall-sample` to change the sample size, or `--recall-sample 0` to disable recall estimation entirely.
If you pass `--build-only`, recall is skipped automatically.

## Running benchmarks

```bash
python run_benchmark.py --algorithm <ALGORITHM> --dataset <DATASET> [OPTIONS]
```

### Algorithms and parameters

#### `brute_force`

Exact search using FAISS `IndexFlatL2`. Uses GPU if available, otherwise CPU. No tunable parameters — it computes all pairwise distances.

**Times recorded:** `search_time_s`

```bash
python run_benchmark.py --algorithm brute_force --dataset sift
```

#### `hnsw`

Approximate search using FAISS `IndexHNSWFlat` (CPU only in FAISS).

**Times recorded:** `build_time_s` (index construction), `search_time_s`

| Parameter | Flag | Default | Description |
|---|---|---|---|
| M | `--hnsw-m` | 32 | Connections per node per layer. Higher = better recall, more memory, slower build. |
| efConstruction | `--hnsw-ef-construction` | 40 | Search depth during index build. Higher = better graph quality, slower build. |
| efSearch | `--hnsw-ef-search` | 64 | Search depth at query time. Higher = better recall, slower search. Must be >= k. |

```bash
python run_benchmark.py --algorithm hnsw --dataset sift --hnsw-m 32 --hnsw-ef-construction 40 --hnsw-ef-search 64
```

#### `cagra`

Approximate search using NVIDIA CAGRA via cuVS (GPU only). Builds a graph index on GPU using an IVF-PQ-based construction algorithm, then searches it with a GPU-optimized traversal.

**Recorded metrics:** `build_time_s`, `search_time_s` (when search runs), `index_memory_bytes`, `peak_build_memory_bytes`

`index_memory_bytes` / `peak_build_memory_bytes` are estimated from GPU used-memory deltas measured around `cagra.build(...)`.

| Parameter | Flag | Default | Description |
|---|---|---|---|
| graph_degree | `--cagra-graph-degree` | 64 | Degree of the final optimized graph. Higher = better recall, more memory. |
| intermediate_graph_degree | `--cagra-intermediate-graph-degree` | 128 | Degree of the intermediate kNN graph before pruning. Must be >= graph_degree. |
| itopk_size | `--cagra-itopk-size` | 64 | Intermediate results retained during search. Higher = better recall, slower search. Must be >= k. |
| build_algo | `--cagra-build-algo` | nn_descent | Graph construction algorithm: `nn_descent` (robust, recommended) or `ivf_pq` (faster but may overflow on large-magnitude vectors). |

```bash
python run_benchmark.py --algorithm cagra --dataset sift --cagra-graph-degree 64 --cagra-intermediate-graph-degree 128 --cagra-itopk-size 64
```

Build-only CAGRA (no search):

```bash
python run_benchmark.py --algorithm cagra --dataset sift --num-db 500000 --num-queries 0 \
    --build-only --seeds 42
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `glove` | Dataset to benchmark on (`glove`, `sift`, `gist`, `sift10m`, `sift100m`). |
| `--num-queries` | 1000 | Number of vectors held out as queries (Q). |
| `--num-db` | all remaining | Number of vectors to use as the database (M). Must satisfy M + Q <= N. If omitted, all non-query vectors are used. |
| `--seeds` | 42 | One or more random seeds for the query/database split. Times are averaged over seeds. |
| `-k` | 10 | Number of nearest neighbors to retrieve. |
| `--recall-sample` | 2000 | Number of queries to brute-force for recall estimation (0 to disable). Only used for approximate algorithms. |
| `--build-only` | off | Build index only (skip search and recall). Not valid with `brute_force`. |

### Examples

```bash
# Download SIFT, then run brute force with 3 seeds
python download_data.py --dataset sift
python run_benchmark.py --algorithm brute_force --dataset sift --num-queries 1000 --seeds 42 43 44

# HNSW on GIST with custom params
python download_data.py --dataset gist
python run_benchmark.py --algorithm hnsw --dataset gist --num-queries 500 --seeds 42 43 44 \
    --hnsw-m 64 --hnsw-ef-construction 100 --hnsw-ef-search 128 -k 20

# CAGRA on SIFT with custom params
python download_data.py --dataset sift
python run_benchmark.py --algorithm cagra --dataset sift --num-queries 1000 --seeds 42 43 44 \
    --cagra-graph-degree 32 --cagra-intermediate-graph-degree 64 --cagra-itopk-size 128

# Compare across datasets
python run_benchmark.py --algorithm brute_force --dataset glove --num-queries 1000 --seeds 42 43 44
python run_benchmark.py --algorithm brute_force --dataset sift --num-queries 1000 --seeds 42 43 44
```

## CAGRA build-only size sweep

Use the dedicated sweep runner to benchmark build-time and memory versus database size:

```bash
python run_cagra_sweep.py --dataset sift --seeds 42 43 44
```

Override CAGRA build settings directly from CLI, for example:

```bash
python run_cagra_sweep.py --dataset sift --seeds 42 43 44 \
    --cagra-graph-degree 32 --cagra-intermediate-graph-degree 96 --cagra-build-algo nn_descent
```

Default `--num-db-sizes` values are:

```text
100000 250000 500000 750000 1000000
```

The script writes both a consolidated JSON and a CSV summary to `results/`.

### Sweeping up to 10M vectors (SIFT family)

```bash
# One-time data prep (writes data/sift10m/vectors.npy with 10M vectors)
python download_data.py --dataset sift10m

# Build-only CAGRA sweep from 2M to 10M vectors
python run_cagra_sweep.py --dataset sift10m \
    --num-db-sizes 2000000 4000000 6000000 8000000 10000000 \
    --num-queries 0 --seeds 42 43 44
```

By default the sweep uses memory-mapped reads (`--mmap-read`) to avoid loading the whole `.npy` into RAM.

## Results

Each run writes a JSON file to `results/` with filename pattern:
```
{algorithm}_{dataset}_db{num_db}_nq{num_queries}_{mode}_{YYYYMMDD_HHMMSS}.json
```

`mode` is `full` or `buildonly`.

The JSON contains all experiment metadata (algorithm, dataset, parameters, device, seeds) and per-seed metrics plus means. For approximate algorithms, recall@k is included when search is enabled and `--recall-sample` > 0. For CAGRA, memory metrics are included in bytes. Example:

```json
{
  "algorithm": "cagra",
  "mode": "build_only",
  "dataset": "SIFT1M",
  "dimension": 128,
  "num_database_vectors": 500000,
  "num_queries": 0,
  "k": 10,
  "device": "gpu",
  "seeds": [42],
  "num_seeds": 1,
  "parameters": {
    "graph_degree": 64,
    "intermediate_graph_degree": 128,
    "itopk_size": 64,
    "build_algo": "nn_descent"
  },
  "per_seed": [
    {
      "seed": 42,
      "build_time_s": 2.31,
      "gpu_memory_before_build_bytes": 368640000,
      "gpu_memory_after_build_bytes": 457179136,
      "index_memory_bytes": 88539136,
      "peak_build_memory_bytes": 121159680
    }
  ],
  "mean_build_time_s": 2.31,
  "mean_index_memory_bytes": 88539136,
  "mean_peak_build_memory_bytes": 121159680,
  "timestamp": "2026-02-06T14:30:00"
}
```
