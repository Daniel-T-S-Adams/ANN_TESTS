"""
Precompute exact nearest-neighbor ground truth for benchmark datasets.

For each vector in the dataset, computes its top-K nearest neighbors among
all other vectors using brute-force IndexFlatL2 (exact L2 search).
Uses GPU if available.

Usage:
    python compute_ground_truth.py --dataset sift
    python compute_ground_truth.py --dataset sift --gt-k 200 --batch-size 50000
    python compute_ground_truth.py --dataset gist --force

Output is saved to data/<dataset>/ground_truth.npy with shape (num_vectors, gt_k).
"""

import argparse
import os
import time

import faiss
import numpy as np

from run_benchmark import AVAILABLE_DATASETS, load_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_GT_K = 100


def ground_truth_filepath(dataset_key):
    return os.path.join(DATA_DIR, dataset_key, "ground_truth.npy")


def compute_ground_truth(vectors, gt_k, batch_size):
    """Compute exact top-gt_k neighbors for every vector against the full dataset.

    Searches for gt_k + 1 neighbors and removes the self-match (distance 0).
    """
    n, d = vectors.shape
    use_gpu = faiss.get_num_gpus() > 0

    if use_gpu:
        print(f"  Building index on GPU (IndexFlatL2, {n} vectors, dim={d})")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        print(f"  Building index on CPU (IndexFlatL2, {n} vectors, dim={d})")
        index = faiss.IndexFlatL2(d)

    index.add(vectors)

    search_k = gt_k + 1  # +1 to account for self-match
    all_indices = np.empty((n, gt_k), dtype=np.int64)

    num_batches = (n + batch_size - 1) // batch_size
    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        print(f"  Searching batch {b + 1}/{num_batches} "
              f"(vectors {start}-{end - 1})...")

        _distances, indices = index.search(vectors[start:end], search_k)

        # Remove self-match: for each query i, remove index (start + i) from results
        for local_i in range(end - start):
            global_i = start + local_i
            row = indices[local_i]
            mask = row != global_i
            filtered = row[mask]
            # Take first gt_k after removing self
            all_indices[global_i] = filtered[:gt_k]

    return all_indices


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute exact nearest-neighbor ground truth."
    )
    p.add_argument(
        "--dataset", required=True, choices=AVAILABLE_DATASETS,
        help="Dataset to compute ground truth for.",
    )
    p.add_argument(
        "--gt-k", type=int, default=DEFAULT_GT_K,
        help=f"Number of exact neighbors to compute per vector (default: {DEFAULT_GT_K}).",
    )
    p.add_argument(
        "--batch-size", type=int, default=10000,
        help="Number of query vectors per search batch (default: 10000).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Recompute even if ground truth file already exists.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    filepath = ground_truth_filepath(args.dataset)
    if os.path.exists(filepath) and not args.force:
        print(f"Ground truth already exists at {filepath}")
        print("Use --force to recompute.")
        return

    vectors, info = load_data(args.dataset)
    print(f"Dataset : {info['name']}")
    print(f"Vectors : {info['num_vectors']}")
    print(f"Dimension: {info['dimension']}")
    print(f"GT k    : {args.gt_k}")

    t0 = time.perf_counter()
    gt_indices = compute_ground_truth(vectors, args.gt_k, args.batch_size)
    elapsed = time.perf_counter() - t0

    np.save(filepath, gt_indices)
    print(f"\nGround truth shape: {gt_indices.shape}")
    print(f"Compute time: {elapsed:.2f} s")
    print(f"Saved to: {filepath}")


if __name__ == "__main__":
    main()
