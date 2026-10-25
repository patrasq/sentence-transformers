"""Throughput benchmarks for SentenceTransformer.encode on CPU.

Usage:
    python perf_bench/bench.py tokenizer      # per-call tokenizer cost vs batch size (respects TOKENIZERS_PARALLELISM)
    python perf_bench/bench.py encode MODEL N  # end-to-end encode throughput + profile breakdown
"""

from __future__ import annotations

import cProfile
import os
import pstats
import random
import statistics
import sys
import time

random.seed(0)
WORDS = (
    "the quick brown fox jumps over a lazy dog while search engines rank documents by relevance to "
    "queries and users read the most useful results first"
).split()


def make_sentences(n: int) -> list[str]:
    return [" ".join(random.choices(WORDS, k=random.randint(4, 40))) for _ in range(n)]


def timed(fn, repeats: int) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times)


def bench_tokenizer() -> None:
    from sentence_transformers import SentenceTransformer

    tokenizer = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")[0].tokenizer
    sentences = make_sentences(200_000)
    print(f"#### Tokenizer `encode_batch`, 200k sentences, TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}")
    print("| batch size | calls | min s | median s | us/sentence | ms/call |")
    print("|---:|---:|---:|---:|---:|---:|")
    for batch_size in (1, 8, 32, 128, 1024, 8192, 200_000):
        batches = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]
        tokenizer.encode_batch(batches[0], add_special_tokens=False)
        best, median = timed(lambda: [tokenizer.encode_batch(b, add_special_tokens=False) for b in batches], 5)
        print(
            f"| {batch_size} | {len(batches)} | {best:.2f} | {median:.2f} | "
            f"{best / len(sentences) * 1e6:.1f} | {best / len(batches) * 1e3:.3f} |"
        )
    print()


def bench_encode(model_name: str, n: int) -> None:
    import torch

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    sentences = make_sentences(n)
    model.encode(sentences[:1000], batch_size=32)
    print(f"#### `encode()` {model_name}, {n} sentences, torch threads={torch.get_num_threads()}")
    print("| batch size | min s | median s | sentences/s |")
    print("|---:|---:|---:|---:|")
    for batch_size in (32, 128, 1024, 4096):
        best, median = timed(lambda: model.encode(sentences, batch_size=batch_size), 3)
        print(f"| {batch_size} | {best:.2f} | {median:.2f} | {n / best:,.0f} |")
    print()

    profiler = cProfile.Profile()
    profiler.enable()
    model.encode(sentences, batch_size=32)
    profiler.disable()
    stats = pstats.Stats(profiler)
    total = stats.total_tt
    print(f"Profile at batch size 32 (total {total:.2f}s), top functions by own time:")
    print()
    print("| own s | share | calls | function |")
    print("|---:|---:|---:|---|")
    rows = sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[:12]
    for (filename, line, name), (_, ncalls, tottime, _, _) in rows:
        short = filename.replace("\\", "/").split("site-packages/")[-1].split("sentence-transformers/")[-1]
        print(f"| {tottime:.2f} | {tottime / total:.0%} | {ncalls} | `{short}:{line}({name})` |")
    print()


if __name__ == "__main__":
    if sys.argv[1] == "tokenizer":
        bench_tokenizer()
    else:
        bench_encode(sys.argv[2], int(sys.argv[3]))
