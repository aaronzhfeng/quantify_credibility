import argparse
import json
import csv
from pathlib import Path
import logging

from .llm_client_local import LocalLlamaClient
from .datasets import load_arc_challenge, load_arc_easy, load_openbookqa
from .calibration import (
    evaluate_mcq_with_mi,
    evaluate_mcq_greedy_baseline,
    evaluate_mcq_self_consistency,
    evaluate_mcq_semantic_entropy,
    evaluate_mcq_self_verification
)
from .detailed_logger import DetailedLogger


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MI & Iterative Prompting on MCQ Benchmarks"
    )
    
    # Method selection
    parser.add_argument(
        "--method",
        choices=["mi", "greedy", "self-consistency", "semantic-entropy", "self-verification"],
        default="mi",
        help="Evaluation method: 'mi' (MI-based, default), 'greedy', 'self-consistency', 'semantic-entropy', 'self-verification'"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=["arc-challenge", "arc-easy", "openbookqa"],
        required=True,
        help="Benchmark dataset to evaluate"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (test/validation)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization (saves memory)"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    
    # MI parameters (using paper's notation)
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of independent chains per question (paper default: 10)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Chain length / pseudo joint dimension (paper default: 2)"
    )
    parser.add_argument(
        "--mi-method",
        choices=["plugin", "listing"],
        default="listing",
        help="MI estimator to use"
    )
    parser.add_argument(
        "--confidence-method",
        choices=["inverse", "exp", "normalized"],
        default="inverse",
        help="How to convert MI to confidence"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens per generation"
    )
    parser.add_argument(
        "--answer-format",
        type=str,
        default="default",
        choices=["default", "strict", "codeblock"],
        help="Answer format: 'default' (verbose), 'strict' (only A/B/C/D), 'codeblock' (```A```)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    # Caching
    parser.add_argument(
        "--cache-path",
        type=str,
        default=".cache/llm_cache.sqlite",
        help="SQLite cache path (default: .cache/llm_cache.sqlite)"
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="readwrite",
        choices=["readwrite", "read", "write", "off"],
        help="Cache mode: readwrite (default), read, write, or off"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Load dataset
    logging.info(f"Loading {args.dataset} ({args.split} split)...")
    if args.dataset == "arc-challenge":
        examples = load_arc_challenge(args.split, args.limit)
    elif args.dataset == "arc-easy":
        examples = load_arc_easy(args.split, args.limit)
    else:
        examples = load_openbookqa(args.split, args.limit)
    
    logging.info(f"Loaded {len(examples)} examples")
    
    # Initialize cache
    from .cache import SQLiteCache
    cache = None
    if args.cache_mode != "off":
        cache = SQLiteCache(args.cache_path, mode=args.cache_mode)
        logging.info(f"Cache enabled: {args.cache_path} (mode: {args.cache_mode})")
    else:
        logging.info("Cache disabled")
    
    # Initialize model
    logging.info(f"Initializing model: {args.model}")
    client = LocalLlamaClient(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        cache=cache
    )
    
    # Verify logprobs support
    if not client.supports_logprobs():
        logging.warning("Client doesn't support logprobs - falling back to approximate method")
    
    # Create detailed logger for saving per-question traces
    detailed_logger = DetailedLogger.create_from_output_path(args.output, args.method)
    logging.info(f"Detailed logs will be saved to: {detailed_logger.output_dir}")
    
    # Run evaluation based on method
    if args.method == "greedy":
        logging.info(f"Starting GREEDY BASELINE evaluation...")
        logging.info(f"This will make {len(examples)} API calls")
        
        metrics, results = evaluate_mcq_greedy_baseline(
            client=client,
            examples=examples,
            max_tokens=args.max_tokens,
            answer_format=args.answer_format,
            detailed_logger=detailed_logger,
            verbose=True
        )
    
    elif args.method == "self-consistency":
        logging.info(f"Starting SELF-CONSISTENCY BASELINE evaluation with k={args.k}...")
        logging.info(f"This will make approximately {len(examples) * args.k} API calls")
        
        metrics, results = evaluate_mcq_self_consistency(
            client=client,
            examples=examples,
            k=args.k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            answer_format=args.answer_format,
            detailed_logger=detailed_logger,
            verbose=True
        )
    
    elif args.method == "semantic-entropy":
        logging.info(f"Starting SEMANTIC ENTROPY evaluation with k={args.k}...")
        logging.info(f"This will make approximately {len(examples) * args.k} API calls")
        
        metrics, results = evaluate_mcq_semantic_entropy(
            client=client,
            examples=examples,
            k=args.k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            answer_format=args.answer_format,
            detailed_logger=detailed_logger,
            verbose=True
        )
    
    elif args.method == "self-verification":
        logging.info(f"Starting SELF-VERIFICATION evaluation with k={args.k}...")
        logging.info(f"This will make approximately {len(examples) * (args.k + 1)} API calls")
        
        metrics, results = evaluate_mcq_self_verification(
            client=client,
            examples=examples,
            k=args.k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            answer_format=args.answer_format,
            detailed_logger=detailed_logger,
            verbose=True
        )
    
    else:  # mi
        logging.info(f"Starting MI METHOD evaluation with k={args.k}, n={args.n}...")
        logging.info(f"This will make approximately {len(examples) * args.k * args.n} API calls")
        
        metrics, results = evaluate_mcq_with_mi(
            client=client,
            examples=examples,
            k=args.k,
            n=args.n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            mi_method=args.mi_method,
            confidence_method=args.confidence_method,
            answer_format=args.answer_format,
            detailed_logger=detailed_logger,
            verbose=True
        )
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*60 + "\n")
    
    # Save results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'question', 'predicted', 'gold', 'correct',
            'confidence', 'mi_score', 'agreement'
        ])
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'question': r.question,
                'predicted': r.predicted,
                'gold': r.gold,
                'correct': int(r.correct),
                'confidence': r.confidence,
                'mi_score': r.mi_score,
                'agreement': r.agreement
            })
    
    # Save metrics to JSON
    metrics_path = output_path.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")
    logging.info(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

