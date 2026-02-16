# ANN Benchmark

Benchmarks brute-force, HNSW, and CAGRA nearest-neighbor search using FAISS and cuVS.

## Setup

```bash
pip install -r requirements.txt
```

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
| `sift100m` | SIFT100M | 128 | 100M | ~10 GB |

Each dataset is stored under `data/<key>/` as `vectors.npy` + `dataset_info.json`.

## Recall measurement

For approximate algorithms (HNSW, CAGRA), recall@k is estimated automatically at benchmark time by running brute-force exact search on a random sample of queries against the actual database. This avoids any precomputation step and works correctly regardless of database size (`--num-db`) or query count.

By default 2000 queries are sampled. Use `--recall-sample` to change the sample size, or `--recall-sample 0` to disable recall estimation entirely.

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

**Times recorded:** `build_time_s` (index construction), `search_time_s`

| Parameter | Flag | Default | Description |
|---|---|---|---|
| graph_degree | `--cagra-graph-degree` | 64 | Degree of the final optimized graph. Higher = better recall, more memory. |
| intermediate_graph_degree | `--cagra-intermediate-graph-degree` | 128 | Degree of the intermediate kNN graph before pruning. Must be >= graph_degree. |
| itopk_size | `--cagra-itopk-size` | 64 | Intermediate results retained during search. Higher = better recall, slower search. Must be >= k. |
| build_algo | `--cagra-build-algo` | nn_descent | Graph construction algorithm: `nn_descent` (robust, recommended) or `ivf_pq` (faster but may overflow on large-magnitude vectors). |

```bash
python run_benchmark.py --algorithm cagra --dataset sift --cagra-graph-degree 64 --cagra-intermediate-graph-degree 128 --cagra-itopk-size 64
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `glove` | Dataset to benchmark on (`glove`, `sift`, `gist`, `sift100m`). |
| `--num-queries` | 1000 | Number of vectors held out as queries (Q). |
| `--num-db` | all remaining | Number of vectors to use as the database (M). Must satisfy M + Q <= N. If omitted, all non-query vectors are used. |
| `--seeds` | 42 | One or more random seeds for the query/database split. Times are averaged over seeds. |
| `-k` | 10 | Number of nearest neighbors to retrieve. |
| `--recall-sample` | 2000 | Number of queries to brute-force for recall estimation (0 to disable). Only used for approximate algorithms. |

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

## Results

Each run writes a JSON file to `results/` with filename pattern:
```
{algorithm}_{dataset}_nq{num_queries}_{num_seeds}seeds_{YYYYMMDD_HHMMSS}.json
```

The JSON contains all experiment metadata (algorithm, dataset, parameters, device, seeds) and per-seed timings plus means. For approximate algorithms, recall@k is included when `--recall-sample` > 0. Example:

```json
{
  "algorithm": "hnsw",
  "dataset": "SIFT1M",
  "dimension": 128,
  "num_database_vectors": 999000,
  "num_queries": 1000,
  "k": 10,
  "device": "gpu",
  "seeds": [42, 43, 44],
  "num_seeds": 3,
  "parameters": {
    "M": 32,
    "efConstruction": 40,
    "efSearch": 64
  },
  "per_seed": [
    {"seed": 42, "build_time_s": 12.3, "search_time_s": 0.05, "recall_at_k": 0.9825},
    {"seed": 43, "build_time_s": 12.1, "search_time_s": 0.04, "recall_at_k": 0.9810},
    {"seed": 44, "build_time_s": 12.5, "search_time_s": 0.06, "recall_at_k": 0.9832}
  ],
  "mean_build_time_s": 12.3,
  "mean_search_time_s": 0.05,
  "mean_recall_at_k": 0.9822,
  "timestamp": "2026-02-06T14:30:00"
}
```
