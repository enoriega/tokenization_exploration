import json
import statistics
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer


BIORED_PATH = Path("data/biored/Train.BioC.JSON")
CUSTOM_TOKENIZER_PATH = Path("data/tokenizer_test")
MODERNBERT_MODEL = "answerdotai/ModernBERT-base"


def load_texts(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    texts = []
    for doc in data["documents"]:
        for passage in doc["passages"]:
            text = passage.get("text", "").strip()
            if text:
                texts.append(text)
    return texts


def compute_stats(token_counts: list[int]) -> dict:
    sorted_counts = sorted(token_counts)
    n = len(sorted_counts)
    return {
        "count": n,
        "total_tokens": sum(sorted_counts),
        "mean": statistics.mean(sorted_counts),
        "median": statistics.median(sorted_counts),
        "stdev": statistics.stdev(sorted_counts) if n > 1 else 0.0,
        "min": sorted_counts[0],
        "p25": sorted_counts[n // 4],
        "p75": sorted_counts[(3 * n) // 4],
        "p95": sorted_counts[int(0.95 * n)],
        "p99": sorted_counts[int(0.99 * n)],
        "max": sorted_counts[-1],
    }


def tokenize_with_hf(tokenizer, texts: list[str]) -> list[int]:
    counts = []
    for text in texts:
        encoded = tokenizer(text, truncation=False, add_special_tokens=True)
        counts.append(len(encoded["input_ids"]))
    return counts


def tokenize_with_custom(tokenizer: Tokenizer, texts: list[str]) -> list[int]:
    encodings = tokenizer.encode_batch(texts)
    return [len(enc.ids) for enc in encodings]


def print_stats(label: str, stats: dict, vocab_size: int) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Vocabulary size : {vocab_size:,}")
    print(f"  Texts tokenized : {stats['count']:,}")
    print(f"  Total tokens    : {stats['total_tokens']:,}")
    print(f"  Mean  / text    : {stats['mean']:.1f}")
    print(f"  Median / text   : {stats['median']:.1f}")
    print(f"  Std dev         : {stats['stdev']:.1f}")
    print(f"  Min             : {stats['min']}")
    print(f"  P25             : {stats['p25']}")
    print(f"  P75             : {stats['p75']}")
    print(f"  P95             : {stats['p95']}")
    print(f"  P99             : {stats['p99']}")
    print(f"  Max             : {stats['max']}")


def print_comparison(modernbert_stats: dict, custom_stats: dict) -> None:
    print(f"\n{'='*60}")
    print("  Comparison: ModernBERT vs Custom BPE")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'ModernBERT':>15} {'Custom BPE':>15} {'Ratio (C/M)':>12}")
    print(f"  {'-'*62}")
    metrics = [
        ("Total tokens", "total_tokens"),
        ("Mean / text", "mean"),
        ("Median / text", "median"),
        ("Std dev", "stdev"),
        ("P95", "p95"),
        ("P99", "p99"),
        ("Max", "max"),
    ]
    for label, key in metrics:
        m_val = modernbert_stats[key]
        c_val = custom_stats[key]
        ratio = c_val / m_val if m_val else float("inf")
        print(f"  {label:<20} {m_val:>15.1f} {c_val:>15.1f} {ratio:>12.3f}")
    print()


def main():
    print("Loading texts from BioRED Train.BioC.JSON …")
    texts = load_texts(BIORED_PATH)
    print(f"  Found {len(texts):,} text passages")

    print(f"\nLoading ModernBERT tokenizer ({MODERNBERT_MODEL}) …")
    modernbert = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    mb_counts = tokenize_with_hf(modernbert, texts)
    mb_stats = compute_stats(mb_counts)
    print_stats("ModernBERT", mb_stats, modernbert.vocab_size)

    print(f"\nLoading custom BPE tokenizer ({CUSTOM_TOKENIZER_PATH}) …")
    custom = Tokenizer.from_file(str(CUSTOM_TOKENIZER_PATH))
    custom_counts = tokenize_with_custom(custom, texts)
    custom_stats = compute_stats(custom_counts)
    print_stats("Custom BPE (PubMed)", custom_stats, custom.get_vocab_size())

    print_comparison(mb_stats, custom_stats)


if __name__ == "__main__":
    main()
