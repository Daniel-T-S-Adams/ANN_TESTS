"""
Download and prepare datasets for nearest-neighbor benchmarks.

Usage:
    python download_data.py --dataset glove
    python download_data.py --dataset sift
    python download_data.py --dataset gist
    python download_data.py --dataset sift100m

Each dataset is stored under data/<dataset_key>/ as:
    vectors.npy        - float32 numpy array of shape (N, D)
    dataset_info.json  - metadata (name, dimension, num_vectors, etc.)
"""

import argparse
import gzip
import json
import os
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "glove": {
        "name": "GloVe 6B 100d",
        "source": "Wikipedia 2014 + Gigaword 5 (6B tokens)",
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
        "archive": "glove.6B.zip",
        "download_size": "~862 MB",
        "dimension": 100,
    },
    "sift": {
        "name": "SIFT1M",
        "source": "ANN benchmark (INRIA TEXMEX corpus)",
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "archive": "sift.tar.gz",
        "download_size": "~500 MB",
        "dimension": 128,
        "base_file": "sift/sift_base.fvecs",
    },
    "gist": {
        "name": "GIST1M",
        "source": "ANN benchmark (INRIA TEXMEX corpus)",
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        "archive": "gist.tar.gz",
        "download_size": "~2.6 GB",
        "dimension": 960,
        "base_file": "gist/gist_base.fvecs",
    },
    "sift100m": {
        "name": "SIFT100M",
        "source": "ANN benchmark (INRIA TEXMEX corpus, BigANN first 100M)",
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz",
        "archive": "bigann_base.bvecs.gz",
        "download_size": "~26 GB",
        "dimension": 128,
        "num_vectors": 100_000_000,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _download_progress(count, block_size, total_size):
    downloaded = count * block_size
    if total_size > 0:
        pct = min(downloaded * 100.0 / total_size, 100.0)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {mb:.1f} / {total_mb:.1f} MB  ({pct:.1f}%)", end="", flush=True)
    else:
        mb = downloaded / 1e6
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


def download_file(url, dest, size_hint=""):
    """Download url to dest if dest doesn't already exist."""
    if os.path.exists(dest):
        print(f"Archive already exists: {dest}")
        return
    print(f"Downloading {url} ...")
    if size_hint:
        print(f"  (This is {size_hint} and may take a while.)")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_download_progress)
        print()  # newline after progress
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        print(f"\nDownload failed: {e}")
        sys.exit(1)
    print("Download complete.")


def read_fvecs(fname):
    """Read vectors from the .fvecs binary format (INRIA TEXMEX corpus)."""
    a = np.fromfile(fname, dtype=np.float32)
    d = a[0].view(np.int32)
    return a.reshape(-1, d + 1)[:, 1:].copy()


def read_bvecs_gz(gz_path, num_vectors, dim):
    """Read the first `num_vectors` from a gzipped .bvecs file as float32.

    The .bvecs format stores each vector as: 4-byte int32 (dimension) followed
    by `dim` uint8 values.  We stream-decompress to avoid extracting the full
    file to disk.
    """
    record_size = 4 + dim
    chunk_vecs = 10_000
    result = np.empty((num_vectors, dim), dtype=np.float32)
    read_count = 0

    with gzip.open(gz_path, "rb") as f:
        while read_count < num_vectors:
            n = min(chunk_vecs, num_vectors - read_count)
            raw = f.read(n * record_size)
            actual = len(raw) // record_size
            if actual == 0:
                break
            raw = raw[: actual * record_size]
            chunk = np.frombuffer(raw, dtype=np.uint8).reshape(actual, record_size)
            result[read_count : read_count + actual] = chunk[:, 4:].astype(np.float32)
            read_count += actual
            if read_count % 1_000_000 < chunk_vecs:
                print(f"  Read {read_count:,} / {num_vectors:,} vectors...")

    return result[:read_count]


def save_dataset(out_dir, vectors, name, source):
    """Save vectors.npy and dataset_info.json."""
    npy_path = os.path.join(out_dir, "vectors.npy")
    np.save(npy_path, vectors)
    print(f"  Saved vectors  -> {npy_path}  (shape {vectors.shape})")

    info = {
        "name": name,
        "source": source,
        "num_vectors": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "dtype": "float32",
    }
    info_path = os.path.join(out_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved metadata -> {info_path}")


# ---------------------------------------------------------------------------
# Per-dataset preparation
# ---------------------------------------------------------------------------
def prepare_glove(out_dir, archive_path):
    """Extract GloVe 100d vectors from the zip and convert to npy."""
    npy_path = os.path.join(out_dir, "vectors.npy")
    if os.path.exists(npy_path):
        print(f"Parsed data already exists at {npy_path}, skipping.")
        return

    txt_name = "glove.6B.100d.txt"
    print(f"Extracting {txt_name} from zip and parsing vectors ...")
    words = []
    vectors = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        with zf.open(txt_name) as f:
            for i, line in enumerate(f):
                parts = line.decode("utf-8", errors="replace").rstrip().split(" ")
                words.append(parts[0])
                vectors.append([float(x) for x in parts[1:]])
                if (i + 1) % 100_000 == 0:
                    print(f"  parsed {i + 1} vectors ...")

    vectors = np.array(vectors, dtype=np.float32)

    # Save word list (GloVe-specific)
    words_path = os.path.join(out_dir, "words.txt")
    with open(words_path, "w") as f:
        for w in words:
            f.write(w + "\n")
    print(f"  Saved words    -> {words_path}")

    ds = DATASETS["glove"]
    save_dataset(out_dir, vectors, ds["name"], ds["source"])


def prepare_fvecs(out_dir, archive_path, dataset_key):
    """Extract base vectors from a TEXMEX tar.gz archive and convert to npy."""
    npy_path = os.path.join(out_dir, "vectors.npy")
    if os.path.exists(npy_path):
        print(f"Parsed data already exists at {npy_path}, skipping.")
        return

    ds = DATASETS[dataset_key]
    base_file = ds["base_file"]
    print(f"Extracting {base_file} from archive ...")

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extract(base_file, path=out_dir)

    extracted_path = os.path.join(out_dir, base_file)
    print("Parsing .fvecs binary format ...")
    vectors = read_fvecs(extracted_path)
    print(f"  Total vectors: {vectors.shape[0]}, dimension: {vectors.shape[1]}")

    # Clean up extracted fvecs file (we now have the npy)
    os.remove(extracted_path)
    # Remove the empty subdirectory left by tar extraction
    extracted_subdir = os.path.join(out_dir, base_file.split("/")[0])
    if os.path.isdir(extracted_subdir) and not os.listdir(extracted_subdir):
        os.rmdir(extracted_subdir)

    save_dataset(out_dir, vectors, ds["name"], ds["source"])


def prepare_sift100m(out_dir, archive_path):
    """Extract first 100M vectors from BigANN .bvecs.gz and convert to npy."""
    npy_path = os.path.join(out_dir, "vectors.npy")
    if os.path.exists(npy_path):
        print(f"Parsed data already exists at {npy_path}, skipping.")
        return

    ds = DATASETS["sift100m"]
    num_vectors = ds["num_vectors"]
    dim = ds["dimension"]

    ram_gb = num_vectors * dim * 4 / 1e9
    print(f"Reading first {num_vectors:,} vectors from {archive_path} ...")
    print(f"  (This requires ~{ram_gb:.0f} GB of RAM for the float32 array)")
    vectors = read_bvecs_gz(archive_path, num_vectors, dim)
    print(f"  Total vectors: {vectors.shape[0]:,}, dimension: {vectors.shape[1]}")

    save_dataset(out_dir, vectors, ds["name"], ds["source"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare a dataset for nearest-neighbor benchmarks."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to download and prepare.",
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    out_dir = os.path.join(DATA_DIR, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    archive_path = os.path.join(out_dir, ds["archive"])
    download_file(ds["url"], archive_path, size_hint=ds["download_size"])

    if args.dataset == "glove":
        prepare_glove(out_dir, archive_path)
    elif args.dataset == "sift100m":
        prepare_sift100m(out_dir, archive_path)
    else:
        prepare_fvecs(out_dir, archive_path, args.dataset)

    print(f"\nDone. {ds['name']} is ready for benchmarking.")


if __name__ == "__main__":
    main()
