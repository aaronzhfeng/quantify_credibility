#!/usr/bin/env python3
"""
Download and cache datasets for NLI clustering experiments.

This script downloads TriviaQA and SQuAD v2 datasets from HuggingFace
and saves them locally for offline use.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets triviaqa squad_v2
    python scripts/download_datasets.py --limit 200  # Download subset only
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset


def download_squad_v2(output_dir: str, split: str = "validation", limit: int = None):
    """Download SQuAD v2 dataset."""
    print(f"\n📥 Downloading SQuAD v2 ({split})...")
    
    # Load from HuggingFace
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"   Limited to {len(ds)} examples")
    else:
        print(f"   Total examples: {len(ds)}")
    
    # Save as JSON lines
    output_file = Path(output_dir) / "squad_v2" / f"{split}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for ex in ds:
            # Extract relevant fields
            data = {
                "id": ex["id"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": ex["answers"]["text"],
                "is_impossible": ex.get("is_impossible", False),
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"   ✓ Saved to: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return len(ds)


def download_triviaqa(output_dir: str, split: str = "validation", limit: int = None):
    """Download TriviaQA dataset (rc.nocontext subset)."""
    print(f"\n📥 Downloading TriviaQA ({split})...")
    
    # Load from HuggingFace (rc.nocontext = no evidence documents)
    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"   Limited to {len(ds)} examples")
    else:
        print(f"   Total examples: {len(ds)}")
    
    # Save as JSON lines
    output_file = Path(output_dir) / "triviaqa" / f"{split}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for ex in ds:
            # Extract relevant fields
            data = {
                "question_id": ex["question_id"],
                "question": ex["question"],
                "answers": ex["answer"]["aliases"],  # Multiple acceptable answers
                "value": ex["answer"]["value"],  # Primary answer
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"   ✓ Saved to: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return len(ds)


def main():
    parser = argparse.ArgumentParser(description="Download datasets for NLI clustering")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["squad_v2", "triviaqa", "all"],
        default=["all"],
        help="Which datasets to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to download (train, validation, test)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    
    args = parser.parse_args()
    
    # Normalize dataset list
    if "all" in args.datasets:
        datasets_to_download = ["squad_v2", "triviaqa"]
    else:
        datasets_to_download = args.datasets
    
    print("=" * 80)
    print("Dataset Download Script")
    print("=" * 80)
    print(f"Datasets: {', '.join(datasets_to_download)}")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit if args.limit else 'None (full dataset)'}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    total_examples = 0
    
    # Download each dataset
    if "squad_v2" in datasets_to_download:
        count = download_squad_v2(args.output_dir, args.split, args.limit)
        total_examples += count
    
    if "triviaqa" in datasets_to_download:
        count = download_triviaqa(args.output_dir, args.split, args.limit)
        total_examples += count
    
    print("\n" + "=" * 80)
    print("✅ Download complete!")
    print("=" * 80)
    print(f"Total examples downloaded: {total_examples}")
    print(f"Location: {args.output_dir}/")
    print()
    print("Files:")
    for dataset in datasets_to_download:
        filepath = Path(args.output_dir) / dataset / f"{args.split}.jsonl"
        if filepath.exists():
            print(f"  - {filepath}")
    print()
    print("💡 Tip: The datasets are cached by HuggingFace and stored in ~/.cache/huggingface/")
    print("   The .jsonl files above are simplified versions for easy inspection.")
    print("=" * 80)


if __name__ == "__main__":
    main()

