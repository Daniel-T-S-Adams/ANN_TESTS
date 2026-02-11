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
    """Randomly split the dataset into database vectors and query vectors.

    Returns (database, queries, db_idx, query_idx) where db_idx and query_idx
    are the original row indices into the full vectors array.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vectors))
    query_idx = idx[:num_queries]
    db_idx = idx[num_queries:]
    return vectors[db_idx].copy(), vectors[query_idx].copy(), db_idx, query_idx


def load_ground_truth(dataset_key):
    """Load precomputed ground truth indices, or return None if not found."""
    filepath = os.path.join(DATA_DIR, dataset_key, "ground_truth.npy")
    if not os.path.exists(filepath):
        return None
    return np.load(filepath)


def compute_recall(gt_all, approx_indices, query_idx, db_idx, k):
    """Compute mean recall@k using precomputed full-dataset ground truth.

    For each query, looks up its precomputed neighbors in gt_all, filters out
    any that are not in the database (i.e. are in the query set for this split),
    and checks how many of the top-k true neighbors appear in the approximate
    result (whose indices reference positions in the database array, not the
    original dataset).

    Parameters
    ----------
    gt_all : ndarray, shape (num_total_vectors, gt_k)
        Precomputed nearest neighbors for every vector (original dataset indices).
    approx_indices : ndarray, shape (num_queries, k)
        Indices returned by the approximate algorithm (database-local, 0-indexed).
    query_idx : ndarray
        Original dataset indices of the query vectors.
    db_idx : ndarray
        Original dataset indices of the database vectors.
    k : int
        Number of neighbors to evaluate recall for.
    """
    query_set = set(query_idx.tolist())
    # Map database-local index -> original dataset index
    # approx_indices[i][j] is position in the database array, db_idx maps that
    # back to the original vector index.
    num_queries = len(query_idx)
    recall_sum = 0.0
    for i in range(num_queries):
        orig_query = query_idx[i]
        # Get precomputed neighbors, filter out those in the query set
        gt_neighbors = gt_all[orig_query]
        filtered = [n for n in gt_neighbors if n not in query_set]
        true_top_k = set(filtered[:k])
        if len(true_top_k) < k:
            # Not enough neighbors after filtering (very unlikely)
            true_top_k_count = len(true_top_k)
        else:
            true_top_k_count = k
        # Convert approximate results from database-local to original indices
        approx_orig = set(db_idx[approx_indices[i]].tolist())
        hits = len(true_top_k & approx_orig)
        recall_sum += hits / true_top_k_count if true_top_k_count > 0 else 1.0
    return recall_sum / num_queries


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
    print(f"Dataset   : {info['name']}")
    print(f"Vectors   : {info['num_vectors']}")
    print(f"Dimension : {info['dimension']}")
    print(f"Algorithm : {args.algorithm}")
    print(f"Queries   : {args.num_queries}")
    print(f"k         : {args.k}")
    print(f"Seeds     : {args.seeds}")

    use_gpu = faiss.get_num_gpus() > 0
    if args.algorithm == "cagra":
        device = "gpu"
    else:
        device = "gpu" if use_gpu else "cpu"

    # --- Ground truth for recall ---
    is_approximate = args.algorithm in ("hnsw", "cagra")
    gt_all = None
    if is_approximate:
        gt_all = load_ground_truth(args.dataset)
        if gt_all is None:
            print(f"\nWARNING: No ground truth found for dataset '{args.dataset}'. "
                  f"Recall will not be computed.\n"
                  f"Run: python compute_ground_truth.py --dataset {args.dataset}")

    per_seed = []

    for seed in args.seeds:
        database, queries, db_idx, query_idx = split_queries(
            vectors, args.num_queries, seed=seed)
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
            if gt_all is not None:
                recall = compute_recall(gt_all, indices, query_idx, db_idx, args.k)
                seed_result["recall_at_k"] = round(recall, 6)
                print(f"  recall@{args.k}   : {recall:.4f}")
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
            if gt_all is not None:
                recall = compute_recall(gt_all, indices, query_idx, db_idx, args.k)
                seed_result["recall_at_k"] = round(recall, 6)
                print(f"  recall@{args.k}   : {recall:.4f}")
            per_seed.append(seed_result)

    # --- Compute means ---
    mean_search = sum(r["search_time_s"] for r in per_seed) / len(per_seed)
    recall_values = [r["recall_at_k"] for r in per_seed if "recall_at_k" in r]
    mean_recall = sum(recall_values) / len(recall_values) if recall_values else None

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
