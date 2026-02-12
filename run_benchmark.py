"""
Benchmark nearest-neighbor search algorithms using FAISS and cuVS.

Usage examples:
    python run_benchmark.py --algorithm brute_force --dataset sift --num-queries 1000 --seeds 42 43 44
    python run_benchmark.py --algorithm hnsw --dataset glove --num-queries 500 --seeds 42 43 44 \
        --hnsw-m 32 --hnsw-ef-construction 40 --hnsw-ef-search 64
    python run_benchmark.py --algorithm cagra --dataset sift --num-queries 1000 --seeds 42 43 44 \
        --cagra-graph-degree 64 --cagra-intermediate-graph-degree 128 --cagra-itopk-size 64

Results are written as JSON files to the results/ directory.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

AVAILABLE_DATASETS = ["glove", "sift", "gist", "sift100m"]


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


def split_queries(vectors, num_queries, num_db=None, seed=42):
    """Randomly split the dataset into database vectors and query vectors.

    Parameters
    ----------
    vectors : ndarray, shape (N, D)
    num_queries : int
        Number of vectors to hold out as queries (Q).
    num_db : int or None
        Number of vectors to use as the database (M).  If None, all remaining
        vectors after the query hold-out are used (M = N - Q).
    seed : int
        Random seed for the permutation.

    Returns (database, queries, db_idx, query_idx) where db_idx and query_idx
    are the original row indices into the full vectors array.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vectors))
    query_idx = idx[:num_queries]
    if num_db is None:
        db_idx = idx[num_queries:]
    else:
        db_idx = idx[num_queries : num_queries + num_db]
    return vectors[db_idx].copy(), vectors[query_idx].copy(), db_idx, query_idx


def estimate_recall(database, queries, approx_indices, k, sample_size, seed=42):
    """Estimate recall@k by running brute-force exact search on a query sample.

    A random subset of queries is searched exactly against the database using
    FAISS IndexFlatL2 (GPU if available).  The fraction of true top-k neighbors
    found by the approximate algorithm is averaged over the sample.

    Parameters
    ----------
    database : ndarray, shape (M, D)
    queries : ndarray, shape (Q, D)
    approx_indices : ndarray, shape (Q, k)
        Indices returned by the approximate algorithm (database-local).
    k : int
    sample_size : int
        Number of queries to sample for recall estimation.
    seed : int
        Random seed for selecting the query sample.

    Returns
    -------
    recall : float
        Mean recall@k over the sampled queries.
    num_sampled : int
        Actual number of queries used (may be < sample_size if Q < sample_size).
    """
    num_queries = len(queries)
    sample_size = min(sample_size, num_queries)

    if sample_size >= num_queries:
        sample_idx = np.arange(num_queries)
    else:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(num_queries, size=sample_size, replace=False)

    sample_queries = queries[sample_idx]

    # Build exact-search index
    d = database.shape[1]
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(database)

    _distances, exact_indices = index.search(sample_queries, k)

    recall_sum = 0.0
    for i, qi in enumerate(sample_idx):
        exact_set = set(exact_indices[i].tolist())
        approx_set = set(approx_indices[qi].tolist())
        recall_sum += len(exact_set & approx_set) / k

    return recall_sum / len(sample_idx), len(sample_idx)


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
    _distances, indices = gpu_index.search(queries, k)
    search_time = time.perf_counter() - t0

    return search_time, indices


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
    _distances, indices = index.search(queries, k)
    search_time = time.perf_counter() - t0

    return search_time, indices


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
    _distances, indices = index.search(queries, k)
    search_time = time.perf_counter() - t0

    return build_time, search_time, indices


# ---------------------------------------------------------------------------
# CAGRA (graph-based, GPU, via cuVS)
# ---------------------------------------------------------------------------
def cagra_search(database, queries, k, graph_degree, intermediate_graph_degree,
                 itopk_size, build_algo):
    import cupy as cp
    from cuvs.neighbors import cagra

    database_gpu = cp.asarray(database)
    queries_gpu = cp.asarray(queries)

    build_params = cagra.IndexParams(
        metric="sqeuclidean",
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
        build_algo=build_algo,
    )

    # --- index build ---
    t0 = time.perf_counter()
    index = cagra.build(build_params, database_gpu)
    cp.cuda.Device(0).synchronize()
    build_time = time.perf_counter() - t0

    # --- search ---
    search_params = cagra.SearchParams(itopk_size=itopk_size)

    # warm-up
    cagra.search(search_params, index, queries_gpu[:1], k)
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    _distances, neighbors = cagra.search(search_params, index, queries_gpu, k)
    cp.cuda.Device(0).synchronize()
    search_time = time.perf_counter() - t0

    indices = cp.asnumpy(cp.asarray(neighbors))
    return build_time, search_time, indices


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
        choices=["brute_force", "hnsw", "cagra"],
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
        "--num-db",
        type=int,
        default=None,
        help="Number of database vectors to use. Must satisfy num-db + num-queries <= N. "
             "If omitted, all remaining vectors after the query hold-out are used.",
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
    p.add_argument(
        "--recall-sample",
        type=int,
        default=2000,
        help="Number of queries to brute-force for recall estimation "
             "(default: 2000, 0 to disable). Only used for approximate algorithms.",
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
    # CAGRA-specific
    p.add_argument(
        "--cagra-graph-degree",
        type=int,
        default=64,
        help="CAGRA graph degree (default: 64).",
    )
    p.add_argument(
        "--cagra-intermediate-graph-degree",
        type=int,
        default=128,
        help="CAGRA intermediate graph degree (default: 128).",
    )
    p.add_argument(
        "--cagra-itopk-size",
        type=int,
        default=64,
        help="CAGRA itopk_size for search (default: 64). Must be >= k.",
    )
    p.add_argument(
        "--cagra-build-algo",
        default="nn_descent",
        choices=["ivf_pq", "nn_descent"],
        help="CAGRA graph build algorithm (default: nn_descent).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    vectors, info = load_data(args.dataset)
    total = info["num_vectors"]

    if args.num_queries > total:
        print(f"Error: --num-queries ({args.num_queries}) exceeds dataset size ({total}).")
        sys.exit(1)
    if args.num_db is not None and args.num_db + args.num_queries > total:
        print(f"Error: --num-db ({args.num_db}) + --num-queries ({args.num_queries}) "
              f"exceeds dataset size ({total}).")
        sys.exit(1)

    num_db = args.num_db if args.num_db is not None else total - args.num_queries

    print(f"Dataset   : {info['name']}")
    print(f"Vectors   : {total}")
    print(f"Dimension : {info['dimension']}")
    print(f"Algorithm : {args.algorithm}")
    print(f"DB size   : {num_db}")
    print(f"Queries   : {args.num_queries}")
    print(f"k         : {args.k}")
    print(f"Seeds     : {args.seeds}")

    use_gpu = faiss.get_num_gpus() > 0
    if args.algorithm == "cagra":
        device = "gpu"
    else:
        device = "gpu" if use_gpu else "cpu"

    is_approximate = args.algorithm in ("hnsw", "cagra")

    per_seed = []

    for seed in args.seeds:
        database, queries, db_idx, query_idx = split_queries(
            vectors, args.num_queries, num_db=args.num_db, seed=seed)
        print(f"\n--- seed {seed}  (db={database.shape[0]}, queries={queries.shape[0]}) ---")

        if args.algorithm == "brute_force":
            if use_gpu:
                search_t, _indices = brute_force_gpu(database, queries, args.k)
            else:
                search_t, _indices = brute_force_cpu(database, queries, args.k)
            print(f"  search_time: {search_t:.4f} s")
            per_seed.append({"seed": seed, "search_time_s": round(search_t, 6)})

        elif args.algorithm == "hnsw":
            build_t, search_t, indices = hnsw_search(
                database, queries, args.k,
                args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search,
            )
            print(f"  build_time : {build_t:.4f} s")
            print(f"  search_time: {search_t:.4f} s")
            seed_result = {
                "seed": seed,
                "build_time_s": round(build_t, 6),
                "search_time_s": round(search_t, 6),
            }
            if args.recall_sample > 0:
                recall, n_sampled = estimate_recall(
                    database, queries, indices, args.k,
                    args.recall_sample, seed=seed)
                seed_result["recall_at_k"] = round(recall, 6)
                print(f"  recall@{args.k}   : {recall:.4f}  (sampled {n_sampled} queries)")
            per_seed.append(seed_result)

        elif args.algorithm == "cagra":
            build_t, search_t, indices = cagra_search(
                database, queries, args.k,
                args.cagra_graph_degree,
                args.cagra_intermediate_graph_degree,
                args.cagra_itopk_size,
                args.cagra_build_algo,
            )
            print(f"  build_time : {build_t:.4f} s")
            print(f"  search_time: {search_t:.4f} s")
            seed_result = {
                "seed": seed,
                "build_time_s": round(build_t, 6),
                "search_time_s": round(search_t, 6),
            }
            if args.recall_sample > 0:
                recall, n_sampled = estimate_recall(
                    database, queries, indices, args.k,
                    args.recall_sample, seed=seed)
                seed_result["recall_at_k"] = round(recall, 6)
                print(f"  recall@{args.k}   : {recall:.4f}  (sampled {n_sampled} queries)")
            per_seed.append(seed_result)

    # --- Compute means ---
    mean_search = sum(r["search_time_s"] for r in per_seed) / len(per_seed)
    recall_values = [r["recall_at_k"] for r in per_seed if "recall_at_k" in r]
    mean_recall = sum(recall_values) / len(recall_values) if recall_values else None

    result = {
        "algorithm": args.algorithm,
        "dataset": info["name"],
        "dimension": info["dimension"],
        "num_database_vectors": num_db,
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

    elif args.algorithm == "cagra":
        result["parameters"] = {
            "graph_degree": args.cagra_graph_degree,
            "intermediate_graph_degree": args.cagra_intermediate_graph_degree,
            "itopk_size": args.cagra_itopk_size,
            "build_algo": args.cagra_build_algo,
        }
        mean_build = sum(r["build_time_s"] for r in per_seed) / len(per_seed)
        result["mean_build_time_s"] = round(mean_build, 6)

    if mean_recall is not None:
        result["mean_recall_at_k"] = round(mean_recall, 6)

    # --- Print summary ---
    print(f"\n{'=' * 50}")
    print(f"  Mean search time : {mean_search:.4f} s")
    if args.algorithm in ("hnsw", "cagra"):
        print(f"  Mean build time  : {mean_build:.4f} s")
    if mean_recall is not None:
        print(f"  Mean recall@{args.k}   : {mean_recall:.4f}")
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
