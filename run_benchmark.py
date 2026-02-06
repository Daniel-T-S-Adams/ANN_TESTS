"""
Benchmark brute-force vs HNSW nearest-neighbor search using FAISS.

Usage examples:
    python run_benchmark.py --algorithm brute_force --dataset sift --num-queries 1000 --seeds 42 43 44
    python run_benchmark.py --algorithm hnsw --dataset glove --num-queries 500 --seeds 42 43 44 \
        --hnsw-m 32 --hnsw-ef-construction 40 --hnsw-ef-search 64

Results are written as JSON files to the results/ directory.
"""

import argparse
import json
import os
import time
from datetime import datetime

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

AVAILABLE_DATASETS = ["glove", "sift", "gist"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(dataset_key):
    dataset_dir = os.path.join(DATA_DIR, dataset_key)
    vectors_file = os.path.join(dataset_dir, "vectors.npy")
    info_file = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(vectors_file):
        raise FileNotFoundError(
            f"{vectors_file} not found. Run: python download_data.py --dataset {dataset_key}"
        )
    with open(info_file) as f:
        info = json.load(f)
    vectors = np.load(vectors_file)
    return vectors, info


def split_queries(vectors, num_queries, seed=42):
    """Randomly split the dataset into database vectors and query vectors."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vectors))
    query_idx = idx[:num_queries]
    db_idx = idx[num_queries:]
    return vectors[db_idx].copy(), vectors[query_idx].copy()


# ---------------------------------------------------------------------------
# Brute-force (GPU)
# ---------------------------------------------------------------------------
def brute_force_gpu(database, queries, k):
    d = database.shape[1]
    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexFlatL2(res, d)
    gpu_index.add(database)

    # warm-up run
    gpu_index.search(queries[:1], k)

    t0 = time.perf_counter()
    gpu_index.search(queries, k)
    search_time = time.perf_counter() - t0

    return search_time


# ---------------------------------------------------------------------------
# Brute-force (CPU fallback)
# ---------------------------------------------------------------------------
def brute_force_cpu(database, queries, k):
    d = database.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(database)

    # warm-up
    index.search(queries[:1], k)

    t0 = time.perf_counter()
    index.search(queries, k)
    search_time = time.perf_counter() - t0

    return search_time


# ---------------------------------------------------------------------------
# HNSW (graph-based, CPU only in FAISS)
# ---------------------------------------------------------------------------
def hnsw_search(database, queries, k, M, ef_construction, ef_search):
    d = database.shape[1]

    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction

    # --- index build ---
    t0 = time.perf_counter()
    index.add(database)
    build_time = time.perf_counter() - t0

    # --- search ---
    index.hnsw.efSearch = ef_search

    # warm-up
    index.search(queries[:1], k)

    t0 = time.perf_counter()
    index.search(queries, k)
    search_time = time.perf_counter() - t0

    return build_time, search_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark nearest-neighbor search algorithms with FAISS."
    )
    p.add_argument(
        "--algorithm",
        required=True,
        choices=["brute_force", "hnsw"],
        help="Algorithm to benchmark.",
    )
    p.add_argument(
        "--dataset",
        default="glove",
        choices=AVAILABLE_DATASETS,
        help="Dataset to use (default: glove). Run download_data.py --dataset <name> first.",
    )
    p.add_argument(
        "--num-queries",
        type=int,
        default=1000,
        help="Number of query vectors to hold out (default: 1000).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for the query/database split (default: 42).",
    )
    p.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10).",
    )
    # HNSW-specific
    p.add_argument("--hnsw-m", type=int, default=32, help="HNSW M (default: 32).")
    p.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=40,
        help="HNSW efConstruction (default: 40).",
    )
    p.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=64,
        help="HNSW efSearch (default: 64).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    vectors, info = load_data(args.dataset)
    print(f"Dataset   : {info['name']}")
    print(f"Vectors   : {info['num_vectors']}")
    print(f"Dimension : {info['dimension']}")
    print(f"Algorithm : {args.algorithm}")
    print(f"Queries   : {args.num_queries}")
    print(f"k         : {args.k}")
    print(f"Seeds     : {args.seeds}")

    use_gpu = faiss.get_num_gpus() > 0
    device = "gpu" if use_gpu else "cpu"

    per_seed = []

    for seed in args.seeds:
        database, queries = split_queries(vectors, args.num_queries, seed=seed)
        print(f"\n--- seed {seed}  (db={database.shape[0]}, queries={queries.shape[0]}) ---")

        if args.algorithm == "brute_force":
            if use_gpu:
                search_t = brute_force_gpu(database, queries, args.k)
            else:
                search_t = brute_force_cpu(database, queries, args.k)
            print(f"  search_time: {search_t:.4f} s")
            per_seed.append({"seed": seed, "search_time_s": round(search_t, 6)})

        else:  # hnsw
            build_t, search_t = hnsw_search(
                database, queries, args.k,
                args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search,
            )
            print(f"  build_time : {build_t:.4f} s")
            print(f"  search_time: {search_t:.4f} s")
            per_seed.append({
                "seed": seed,
                "build_time_s": round(build_t, 6),
                "search_time_s": round(search_t, 6),
            })

    # --- Compute means ---
    mean_search = sum(r["search_time_s"] for r in per_seed) / len(per_seed)

    result = {
        "algorithm": args.algorithm,
        "dataset": info["name"],
        "dimension": info["dimension"],
        "num_database_vectors": info["num_vectors"] - args.num_queries,
        "num_queries": args.num_queries,
        "k": args.k,
        "device": device,
        "seeds": args.seeds,
        "num_seeds": len(args.seeds),
        "per_seed": per_seed,
        "mean_search_time_s": round(mean_search, 6),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if args.algorithm == "hnsw":
        result["parameters"] = {
            "M": args.hnsw_m,
            "efConstruction": args.hnsw_ef_construction,
            "efSearch": args.hnsw_ef_search,
        }
        mean_build = sum(r["build_time_s"] for r in per_seed) / len(per_seed)
        result["mean_build_time_s"] = round(mean_build, 6)

    # --- Print summary ---
    print(f"\n{'=' * 50}")
    print(f"  Mean search time : {mean_search:.4f} s")
    if args.algorithm == "hnsw":
        print(f"  Mean build time  : {mean_build:.4f} s")
    print(f"{'=' * 50}")

    # --- Write result file ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.algorithm}_{args.dataset}_nq{args.num_queries}_{len(args.seeds)}seeds_{ts}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {filepath}")


if __name__ == "__main__":
    main()
