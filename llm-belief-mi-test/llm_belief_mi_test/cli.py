import argparse
import json
import csv
from pathlib import Path
import logging

from .llm_client_local import LocalLlamaClient
from .datasets import load_arc_challenge, load_arc_easy, load_openbookqa, load_squad_v2, load_truthfulqa_mc1, load_truthfulqa_mc2, load_triviaqa
from .calibration import (
    evaluate_mcq_with_mi,
    evaluate_mcq_greedy_baseline,
    evaluate_mcq_self_consistency,
    evaluate_mcq_semantic_entropy,
    evaluate_mcq_self_verification,
    evaluate_extractive_qa_with_mi,
    evaluate_extractive_qa_greedy,
    evaluate_extractive_qa_self_consistency,
    evaluate_truthfulqa_with_correctness_mi,
    evaluate_truthfulqa_mc2_with_correctness_mi,
    evaluate_triviaqa_with_mi
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
        choices=["arc-challenge", "arc-easy", "openbookqa", "squad-v2", "truthfulqa-mc1", "truthfulqa-mc2", "triviaqa"],
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
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N examples (for multi-GPU parallelism)"
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU parallelism (auto-splits work across available GPUs)"
    )
    parser.add_argument(
        "--log-base-path",
        type=str,
        default=None,
        help="Base path for detailed logs (used internally by multi-GPU workers)"
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
    
    # NLI clustering parameters
    parser.add_argument(
        "--use-nli-clustering",
        action="store_true",
        help="Enable NLI-based semantic clustering for MI computation (measures semantic uncertainty)"
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="Threshold for NLI mutual entailment (default: 0.5)"
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default="microsoft/deberta-v2-xlarge-mnli",
        help="NLI model for semantic clustering (default: microsoft/deberta-v2-xlarge-mnli)"
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
    
    # Check for multi-GPU mode
    if args.multi_gpu:
        from .multi_gpu import run_multi_gpu_evaluation
        return run_multi_gpu_evaluation(args)
    
    # Load dataset
    logging.info(f"Loading {args.dataset} ({args.split} split)...")
    if args.dataset == "arc-challenge":
        examples = load_arc_challenge(args.split, args.limit, args.offset)
        dataset_type = "mcq"
    elif args.dataset == "arc-easy":
        examples = load_arc_easy(args.split, args.limit, args.offset)
        dataset_type = "mcq"
    elif args.dataset == "openbookqa":
        examples = load_openbookqa(args.split, args.limit, args.offset)
        dataset_type = "mcq"
    elif args.dataset == "truthfulqa-mc1":
        examples = load_truthfulqa_mc1(args.split, args.limit, args.offset)
        dataset_type = "mcq"
    elif args.dataset == "truthfulqa-mc2":
        examples = load_truthfulqa_mc2(args.split, args.limit, args.offset)
        dataset_type = "mcq"
    elif args.dataset == "squad-v2":
        examples = load_squad_v2(args.split, args.limit, args.offset)
        dataset_type = "extractive"
    elif args.dataset == "triviaqa":
        examples = load_triviaqa(args.split, args.limit, args.offset)
        dataset_type = "triviaqa"  # Special type for TriviaQA
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
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
    # Use log_base_path if provided (for multi-GPU workers), otherwise use output path
    log_path = args.log_base_path if args.log_base_path else args.output
    detailed_logger = DetailedLogger.create_from_output_path(log_path, args.method)
    logging.info(f"Detailed logs will be saved to: {detailed_logger.output_dir}")
    
    # Run evaluation based on method
    if args.method == "greedy":
        logging.info(f"Starting GREEDY BASELINE evaluation...")
        logging.info(f"This will make {len(examples)} API calls")
        
        # Route based on dataset type
        if dataset_type in ["extractive", "triviaqa"]:
            from .iterative_prompting import compose_prompt_extractive, compose_prompt_trivia
            prompt_composer = compose_prompt_trivia if dataset_type == "triviaqa" else compose_prompt_extractive
            metrics, results = evaluate_extractive_qa_greedy(
                client=client,
                examples=examples,
                max_tokens=args.max_tokens,
                prompt_composer=prompt_composer,
                dataset_name=args.dataset,
                detailed_logger=detailed_logger,
                offset=args.offset,
                verbose=True
            )
        else:  # MCQ datasets (including TruthfulQA MC1/MC2)
            metrics, results = evaluate_mcq_greedy_baseline(
                client=client,
                examples=examples,
                max_tokens=args.max_tokens,
                answer_format=args.answer_format,
                detailed_logger=detailed_logger,
                offset=args.offset,
                verbose=True
            )
    
    elif args.method == "self-consistency":
        logging.info(f"Starting SELF-CONSISTENCY BASELINE evaluation with k={args.k}...")
        logging.info(f"This will make approximately {len(examples) * args.k} API calls")
        
        # Route based on dataset type
        if dataset_type in ["extractive", "triviaqa"]:
            from .iterative_prompting import compose_prompt_extractive, compose_prompt_trivia
            prompt_composer = compose_prompt_trivia if dataset_type == "triviaqa" else compose_prompt_extractive
            metrics, results = evaluate_extractive_qa_self_consistency(
                client=client,
                examples=examples,
                k=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                prompt_composer=prompt_composer,
                dataset_name=args.dataset,
                detailed_logger=detailed_logger,
                offset=args.offset,
                verbose=True
            )
        else:  # MCQ datasets (including TruthfulQA MC1/MC2)
            metrics, results = evaluate_mcq_self_consistency(
                client=client,
                examples=examples,
                k=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                answer_format=args.answer_format,
                detailed_logger=detailed_logger,
                offset=args.offset,
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
            offset=args.offset,
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
            offset=args.offset,
            verbose=True
        )
    
    else:  # mi
        logging.info(f"Starting MI METHOD evaluation with k={args.k}, n={args.n}...")
        logging.info(f"This will make approximately {len(examples) * args.k * args.n} API calls")
        
        # Choose evaluation function based on dataset type
        if dataset_type == "extractive":
            metrics, results = evaluate_extractive_qa_with_mi(
                client=client,
                examples=examples,
                k=args.k,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                mi_method=args.mi_method,
                confidence_method=args.confidence_method,
                offset=args.offset,
                detailed_logger=detailed_logger,
                verbose=True,
                use_nli_clustering=args.use_nli_clustering,
                nli_threshold=args.nli_threshold,
                nli_model=args.nli_model
            )
        elif dataset_type == "triviaqa":
            # TriviaQA uses correctness-based MI (similar to SQuAD but no context)
            logging.info("Using correctness-based MI for TriviaQA (open-domain trivia)")
            metrics, results = evaluate_triviaqa_with_mi(
                client=client,
                examples=examples,
                k=args.k,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                mi_method=args.mi_method,
                confidence_method=args.confidence_method,
                offset=args.offset,
                detailed_logger=detailed_logger,
                verbose=True,
                use_nli_clustering=args.use_nli_clustering,
                nli_threshold=args.nli_threshold,
                nli_model=args.nli_model
            )
        elif args.dataset == "truthfulqa-mc1":
            # TruthfulQA MC1 uses correctness-based MI instead of choice-based MI
            logging.info("Using correctness-based MI for TruthfulQA MC1 (measures agreement on truthfulness)")
            metrics, results = evaluate_truthfulqa_with_correctness_mi(
                client=client,
                examples=examples,
                k=args.k,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                mi_method=args.mi_method,
                confidence_method=args.confidence_method,
                answer_format=args.answer_format,
                offset=args.offset,
                detailed_logger=detailed_logger,
                verbose=True
            )
        elif args.dataset == "truthfulqa-mc2":
            # TruthfulQA MC2 uses correctness-based MI with multi-label support
            logging.info("Using correctness-based MI for TruthfulQA MC2 (multi-true, measures agreement on truthfulness)")
            metrics, results = evaluate_truthfulqa_mc2_with_correctness_mi(
                client=client,
                examples=examples,
                k=args.k,
                n=args.n,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                mi_method=args.mi_method,
                confidence_method=args.confidence_method,
                answer_format=args.answer_format,
                offset=args.offset,
                detailed_logger=detailed_logger,
                verbose=True
            )
        else:  # standard mcq
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
                offset=args.offset,
                detailed_logger=detailed_logger,
                verbose=True
            )
    
    # Print logprob extraction statistics if available
    if "logprob_stats" in metrics:
        stats = metrics["logprob_stats"]
        print("\n" + "="*60)
        print("LOGPROB EXTRACTION STATUS")
        print("="*60)
        print(f"Total inferences     : {stats['total_inferences']:,}")
        print(f"Successfully captured: {stats['captured']:,} ({stats['capture_rate']*100:.1f}%)")
        print(f"Fallback (0.0)       : {stats['fallback']:,} ({(1-stats['capture_rate'])*100:.1f}%)")
        
        # Warning if fallback rate is high
        if stats['capture_rate'] < 0.9:
            print(f"\n⚠️  WARNING: Low logprob capture rate!")
            print(f"   This may affect methods that rely on probabilities:")
            print(f"   - Greedy: Confidence values may be incorrect")
            print(f"   - Semantic Entropy: Results may be invalid")
            print(f"   See docs/LOGPROB_DIAGNOSTIC.md for details")
        print("="*60)
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        # Skip logprob_stats in the main metrics display (already shown above)
        if key != "logprob_stats":
            if isinstance(value, (int, float)):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")
    print("="*60 + "\n")
    
    # Save results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        if dataset_type == "extractive":
            # Extractive QA results (dict format)
            fieldnames = ['id', 'question', 'predicted', 'gold_answers', 'exact_match', 'f1',
                         'confidence', 'mi_score', 'agreement', 'is_impossible']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                writer.writerow({
                    'id': r.get('id', ''),
                    'question': r['question'],
                    'predicted': r['predicted'],
                    'gold_answers': str(r['gold_answers']),
                    'exact_match': r['exact_match'],
                    'f1': r['f1'],
                    'confidence': r['confidence'],
                    'mi_score': r['mi_score'],
                    'agreement': r['agreement'],
                    'is_impossible': r['is_impossible']
                })
        elif dataset_type == "triviaqa":
            # TriviaQA results (dict format, similar to extractive but no is_impossible)
            fieldnames = ['id', 'question', 'predicted', 'gold_answers', 'exact_match', 'f1',
                         'confidence', 'mi_score', 'agreement']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                writer.writerow({
                    'id': r.get('id', ''),
                    'question': r['question'],
                    'predicted': r['predicted'],
                    'gold_answers': str(r['gold_answers']),
                    'exact_match': r['exact_match'],
                    'f1': r['f1'],
                    'confidence': r['confidence'],
                    'mi_score': r['mi_score'],
                    'agreement': r['agreement']
                })
        else:
            # MCQ results (EvaluationResult objects)
            fieldnames = ['question', 'predicted', 'gold', 'correct',
                         'confidence', 'mi_score', 'agreement']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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

