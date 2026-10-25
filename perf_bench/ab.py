"""A/B benchmark of SentenceTransformer.encode between two source trees.

Usage:
    python perf_bench/ab.py <tree_a> <tree_b>

Each measurement runs in a fresh subprocess that imports sentence_transformers from one tree via PYTHONPATH,
and the two trees alternate within every round so that runner drift affects both equally. Every run also hashes
the full embedding matrix, so the two trees must produce bit-for-bit identical embeddings.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys

CHILD = r"""
import hashlib, json, random, sys, time
import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer

model_name, n, batch_size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
random.seed(0)
words = (
    "the quick brown fox jumps over a lazy dog while search engines rank documents by relevance to "
    "queries and users read the most useful results first"
).split()
sentences = [" ".join(random.choices(words, k=random.randint(4, 40))) for _ in range(n)]
model = SentenceTransformer(model_name, device="cpu")
model.encode(sentences[:1000], batch_size=batch_size)
start = time.perf_counter()
embeddings = model.encode(sentences, batch_size=batch_size)
elapsed = time.perf_counter() - start
print(json.dumps({
    "elapsed": elapsed,
    "sha256": hashlib.sha256(np.ascontiguousarray(embeddings).tobytes()).hexdigest(),
    "module": sentence_transformers.__file__,
}))
"""

CASES = [
    ("sentence-transformers/static-retrieval-mrl-en-v1", 200_000, 32, 7),
    ("sentence-transformers/static-retrieval-mrl-en-v1", 200_000, 1024, 7),
    ("sentence-transformers/all-MiniLM-L6-v2", 5_000, 32, 3),
]


def run(tree: str, model_name: str, n: int, batch_size: int) -> dict:
    tree_root = os.path.join(os.path.abspath(tree), "")
    env = {**os.environ, "PYTHONPATH": tree_root}
    # -P keeps the current directory off sys.path, so only PYTHONPATH decides which tree is imported
    out = subprocess.run(
        [sys.executable, "-P", "-c", CHILD, model_name, str(n), str(batch_size)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()[-1]
    result = json.loads(out)
    assert os.path.abspath(result["module"]).startswith(tree_root), (tree_root, result["module"])
    return result


def main() -> None:
    trees = {"main": sys.argv[1], "fix": sys.argv[2]}
    print("| model | sentences | batch size | main min s | main median s | fix min s | fix median s | speedup (min) | identical embeddings |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    all_identical = True
    for model_name, n, batch_size, rounds in CASES:
        times = {"main": [], "fix": []}
        hashes = {"main": set(), "fix": set()}
        for round_idx in range(rounds):
            order = ("main", "fix") if round_idx % 2 == 0 else ("fix", "main")
            for name in order:
                result = run(trees[name], model_name, n, batch_size)
                times[name].append(result["elapsed"])
                hashes[name].add(result["sha256"])
        identical = len(hashes["main"]) == 1 and hashes["main"] == hashes["fix"]
        all_identical &= identical
        speedup = min(times["main"]) / min(times["fix"])
        print(
            f"| `{model_name.split('/')[-1]}` | {n:,} | {batch_size} | {min(times['main']):.2f} | "
            f"{statistics.median(times['main']):.2f} | {min(times['fix']):.2f} | {statistics.median(times['fix']):.2f} | "
            f"{speedup:.2f}x | {'yes' if identical else 'NO'} |"
        )
    print()
    print(f"Rounds alternate the order of main and fix. All embeddings identical: {all_identical}")
    if not all_identical:
        sys.exit(1)


if __name__ == "__main__":
    main()
