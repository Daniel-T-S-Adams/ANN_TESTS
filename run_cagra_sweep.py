"""
Run CAGRA build-only benchmarks across multiple database sizes.

This script measures:
  - CAGRA build time
  - Estimated index memory (steady-state)
  - Peak temporary GPU memory during build

It writes:
  - Consolidated JSON with per-size and per-seed results
  - CSV summary (one row per database size)
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import faiss

from run_benchmark import AVAILABLE_DATASETS
from run_benchmark import RESULTS_DIR
from run_benchmark import cagra_search
from run_benchmark import load_data
from run_benchmark import split_queries

DEFAULT_NUM_DB_SIZES = [100_000, 250_000, 500_000, 750_000, 1_000_000]


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep CAGRA build-only benchmarking across database sizes."
    )
    p.add_argument(
        "--dataset",
        default="sift",
        choices=AVAILABLE_DATASETS,
        help="Dataset to use (default: sift). Run download_data.py first.",
    )
    p.add_argument(
        "--num-db-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_DB_SIZES,
        help="Database sizes to benchmark "
             "(default: 100000 250000 500000 750000 1000000).",
    )
    p.add_argument(
        "--num-queries",
        type=int,
        default=0,
        help="Held-out query vectors for split reproducibility (default: 0).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for DB/query split (default: 42).",
    )
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
        "--cagra-build-algo",
        default="nn_descent",
        choices=["ivf_pq", "nn_descent"],
        help="CAGRA build algorithm (default: nn_descent).",
    )
    p.add_argument(
        "--mmap-read",
        dest="mmap_read",
        action="store_true",
        default=True,
        help="Read vectors.npy using numpy mmap (default: enabled).",
    )
    p.add_argument(
        "--no-mmap-read",
        dest="mmap_read",
        action="store_false",
        help="Disable mmap and fully load vectors.npy into RAM.",
    )
    return p.parse_args()


def mean_or_none(values):
    return (sum(values) / len(values)) if values else None


def main():
    args = parse_args()

    if faiss.get_num_gpus() <= 0:
        print("Error: CAGRA sweep requires a GPU but none was detected.")
        sys.exit(1)
    if args.num_queries < 0:
        print("Error: --num-queries must be >= 0.")
        sys.exit(1)
    if args.cagra_intermediate_graph_degree < args.cagra_graph_degree:
        print("Error: --cagra-intermediate-graph-degree must be >= --cagra-graph-degree.")
        sys.exit(1)
    if not args.num_db_sizes:
        print("Error: provide at least one --num-db-sizes value.")
        sys.exit(1)
    if any(s <= 0 for s in args.num_db_sizes):
        print("Error: all --num-db-sizes values must be > 0.")
        sys.exit(1)

    vectors, info = load_data(args.dataset, mmap_mode="r" if args.mmap_read else None)
    total = info["num_vectors"]

    max_required = max(args.num_db_sizes) + args.num_queries
    if max_required > total:
        hint = ""
        if args.dataset == "sift":
            hint = " (use --dataset sift10m or --dataset sift100m for larger sweeps)"
        print(
            f"Error: max(num_db_sizes) + num_queries = {max_required} exceeds dataset size "
            f"{total}.{hint}"
        )
        sys.exit(1)

    print(f"Dataset   : {info['name']}")
    print(f"Vectors   : {total}")
    print(f"Dimension : {info['dimension']}")
    print("Algorithm : cagra")
    print("Mode      : build_only")
    print(f"Load mode : {'mmap' if args.mmap_read else 'in_memory'}")
    print("CAGRA params:")
    print(f"  graph_degree              : {args.cagra_graph_degree}")
    print(f"  intermediate_graph_degree : {args.cagra_intermediate_graph_degree}")
    print(f"  build_algo                : {args.cagra_build_algo}")
    print(f"DB sizes  : {args.num_db_sizes}")
    print(f"Queries   : {args.num_queries}")
    print(f"Seeds     : {args.seeds}")

    per_size = []

    for num_db in args.num_db_sizes:
        print(f"\n=== num_db={num_db} ===")
        per_seed = []

        for seed in args.seeds:
            database, queries, _db_idx, _query_idx = split_queries(
                vectors, args.num_queries, num_db=num_db, seed=seed
            )

            build_t, _search_t, _indices, memory_stats = cagra_search(
                database=database,
                queries=queries,
                k=1,
                graph_degree=args.cagra_graph_degree,
                intermediate_graph_degree=args.cagra_intermediate_graph_degree,
                itopk_size=1,
                build_algo=args.cagra_build_algo,
                run_search=False,
                track_memory=True,
            )

            seed_result = {
                "seed": seed,
                "build_time_s": round(build_t, 6),
            }
            seed_result.update(memory_stats)
            per_seed.append(seed_result)

            print(f"  seed {seed} build_time: {build_t:.4f} s")
            if "index_memory_bytes" in memory_stats:
                print(f"  seed {seed} index_mem : {memory_stats['index_memory_bytes'] / 1e6:.2f} MB")
                print(f"  seed {seed} peak_build: {memory_stats['peak_build_memory_bytes'] / 1e6:.2f} MB")

            del database
            del queries

        mean_build = mean_or_none([r["build_time_s"] for r in per_seed])
        mean_index_memory = mean_or_none([r["index_memory_bytes"] for r in per_seed])
        mean_peak_build_memory = mean_or_none(
            [r["peak_build_memory_bytes"] for r in per_seed]
        )

        size_result = {
            "num_database_vectors": num_db,
            "num_queries": args.num_queries,
            "per_seed": per_seed,
            "mean_build_time_s": round(mean_build, 6) if mean_build is not None else None,
            "mean_index_memory_bytes": (
                int(round(mean_index_memory)) if mean_index_memory is not None else None
            ),
            "mean_peak_build_memory_bytes": (
                int(round(mean_peak_build_memory))
                if mean_peak_build_memory is not None else None
            ),
        }
        per_size.append(size_result)

    result = {
        "algorithm": "cagra",
        "mode": "build_only",
        "dataset": info["name"],
        "dimension": info["dimension"],
        "num_db_sizes": args.num_db_sizes,
        "num_queries": args.num_queries,
        "device": "gpu",
        "seeds": args.seeds,
        "num_seeds": len(args.seeds),
        "parameters": {
            "graph_degree": args.cagra_graph_degree,
            "intermediate_graph_degree": args.cagra_intermediate_graph_degree,
            "build_algo": args.cagra_build_algo,
        },
        "sizes": per_size,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cagra_sweep_{args.dataset}_{ts}"
    json_path = os.path.join(RESULTS_DIR, f"{base_name}.json")
    csv_path = os.path.join(RESULTS_DIR, f"{base_name}.csv")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_database_vectors",
                "mean_build_time_s",
                "mean_index_memory_bytes",
                "mean_peak_build_memory_bytes",
                "graph_degree",
                "intermediate_graph_degree",
                "build_algo",
                "num_queries",
                "num_seeds",
            ],
        )
        writer.writeheader()
        for row in per_size:
            writer.writerow(
                {
                    "num_database_vectors": row["num_database_vectors"],
                    "mean_build_time_s": row["mean_build_time_s"],
                    "mean_index_memory_bytes": row["mean_index_memory_bytes"],
                    "mean_peak_build_memory_bytes": row["mean_peak_build_memory_bytes"],
                    "graph_degree": args.cagra_graph_degree,
                    "intermediate_graph_degree": args.cagra_intermediate_graph_degree,
                    "build_algo": args.cagra_build_algo,
                    "num_queries": row["num_queries"],
                    "num_seeds": len(args.seeds),
                }
            )

    print(f"\nConsolidated JSON written to {json_path}")
    print(f"CSV summary written to {csv_path}")


if __name__ == "__main__":
    main()
