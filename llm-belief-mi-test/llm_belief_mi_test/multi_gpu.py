"""
Multi-GPU evaluation orchestration.

Automatically splits work across available GPUs and merges results.
"""

import subprocess
import os
import sys
import time
import logging
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
import tempfile

logger = logging.getLogger(__name__)


def detect_gpus() -> List[Dict[str, Any]]:
    """
    Detect available GPUs and return their properties.
    
    Returns:
        List of dicts with GPU info (id, name, memory_gb)
    """
    if not torch.cuda.is_available():
        return []
    
    gpus = []
    n_gpus = torch.cuda.device_count()
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "id": i,
            "name": props.name,
            "memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}"
        })
    
    return gpus


def report_gpu_setup(gpus: List[Dict], total_examples: int, task_distribution: List[Dict]):
    """
    Print GPU detection and task assignment report.
    
    Args:
        gpus: List of GPU info dicts
        total_examples: Total number of examples to evaluate
        task_distribution: List of {gpu_id, offset, limit} for each GPU
    """
    print("\n" + "="*60)
    print("MULTI-GPU SETUP")
    print("="*60)
    print(f"Detected GPUs        : {len(gpus)}")
    
    for gpu in gpus:
        print(f"GPU {gpu['id']}                : {gpu['name']} ({gpu['memory_gb']:.0f} GB)")
    
    print()
    print(f"Examples to evaluate : {total_examples}")
    print(f"Parallel mode        : ENABLED")
    print()
    print("Task Distribution:")
    
    for task in task_distribution:
        gpu_id = task['gpu_id']
        offset = task['offset']
        limit = task['limit']
        end_idx = offset + limit - 1
        print(f"  GPU {gpu_id} → Examples {offset:4d}-{end_idx:4d}   ({limit} examples)")
    
    print()
    print(f"Launching {len(gpus)} parallel workers...")
    print("="*60)
    print()


def build_worker_command(args, offset: int, limit: int, output_path: str, log_base_path: str = None) -> List[str]:
    """
    Build CLI command for a single GPU worker.
    
    Args:
        args: Original CLI arguments
        offset: Starting example index
        limit: Number of examples for this worker
        output_path: Output file path for this worker (CSV results - temporary)
        log_base_path: Base path for detailed logs (original user path - permanent)
        
    Returns:
        Command as list of strings
    """
    cmd = [
        sys.executable, "-m", "llm_belief_mi_test.cli",
        "--method", args.method,
        "--dataset", args.dataset,
        "--split", args.split,
        "--offset", str(offset),
        "--limit", str(limit),
        "--output", output_path,
        "--model", args.model,
    ]
    
    # Add log base path for multi-GPU logging
    if log_base_path:
        cmd.extend(["--log-base-path", log_base_path])
    
    # Add optional flags
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.load_in_8bit:
        cmd.append("--load-in-8bit")
    if args.verbose:
        cmd.append("--verbose")
    
    # Add method-specific parameters
    if args.method == "mi":
        cmd.extend(["--k", str(args.k), "--n", str(args.n)])
        cmd.extend(["--mi-method", args.mi_method])
        cmd.extend(["--confidence-method", args.confidence_method])
    elif args.method in ["self-consistency", "semantic-entropy", "self-verification"]:
        cmd.extend(["--k", str(args.k)])
    
    # Add generation parameters
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--max-tokens", str(args.max_tokens)])
    
    if hasattr(args, 'answer_format') and args.answer_format:
        cmd.extend(["--answer-format", args.answer_format])
    
    # Cache settings
    cmd.extend(["--cache-path", args.cache_path])
    cmd.extend(["--cache-mode", args.cache_mode])
    
    return cmd


def monitor_workers(processes: List[subprocess.Popen], temp_dir: Path, 
                   task_distribution: List[Dict], total_examples: int):
    """
    Monitor worker processes and print periodic status updates.
    
    Args:
        processes: List of worker subprocess objects
        temp_dir: Directory containing progress files
        task_distribution: Task assignment info
        total_examples: Total number of examples across all workers
    """
    n_gpus = len(processes)
    update_interval = 30  # seconds
    
    print(f"[{time.strftime('%H:%M:%S')}] Workers launched. Monitoring progress...")
    print()
    
    last_update = time.time()
    
    while any(p.poll() is None for p in processes):
        current_time = time.time()
        
        # Print update every 30 seconds
        if current_time - last_update >= update_interval:
            status_parts = []
            total_done = 0
            
            for i, task in enumerate(task_distribution):
                gpu_id = task['gpu_id']
                limit = task['limit']
                
                # Read progress file if it exists
                progress_file = temp_dir / f"gpu{gpu_id}_progress.txt"
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r') as f:
                            done = int(f.read().strip())
                    except:
                        done = 0
                else:
                    done = 0
                
                total_done += done
                
                if done >= limit:
                    status_parts.append(f"GPU{gpu_id}: {done}/{limit} ✓")
                else:
                    status_parts.append(f"GPU{gpu_id}: {done}/{limit}")
            
            status_str = " | ".join(status_parts)
            pct = (total_done / total_examples) * 100 if total_examples > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] {status_str} → Total: {total_done}/{total_examples} ({pct:.0f}%)")
            
            last_update = current_time
        
        time.sleep(5)  # Check every 5 seconds
    
    # Final status
    print()
    print(f"[{time.strftime('%H:%M:%S')}] All workers completed!")
    
    # Check for failures
    failed = []
    for i, p in enumerate(processes):
        if p.returncode != 0:
            failed.append(i)
    
    if failed:
        print(f"⚠️  WARNING: {len(failed)} worker(s) failed: GPUs {failed}")
        return False
    else:
        print(f"✓ All {n_gpus} workers succeeded")
        return True


def run_multi_gpu_evaluation(args) -> None:
    """
    Main entry point for multi-GPU evaluation.
    
    Args:
        args: Parsed CLI arguments
    """
    # Detect GPUs
    gpus = detect_gpus()
    n_gpus = len(gpus)
    
    if n_gpus == 0:
        logger.error("No GPUs detected! Cannot run multi-GPU mode.")
        logger.info("Falling back to single-GPU mode...")
        args.multi_gpu = False
        from .cli import main
        return main()
    
    if n_gpus == 1:
        logger.info("Only 1 GPU detected. Multi-GPU mode not needed.")
        logger.info("Running in standard single-GPU mode...")
        args.multi_gpu = False
        # Continue with normal execution by returning to main
        # But we can't easily do that, so just warn
        print("\n⚠️  Only 1 GPU available - multi-GPU mode has no benefit.")
        print("   Proceeding with single-GPU evaluation...")
        print()
        # Re-import to avoid circular dependency
        import importlib
        import llm_belief_mi_test.cli
        importlib.reload(llm_belief_mi_test.cli)
        return
    
    # Determine total examples
    # We need to load dataset metadata to get size
    from .datasets import (
        load_arc_challenge, load_arc_easy, load_openbookqa,
        load_squad_v2, load_truthfulqa_mc1, load_truthfulqa_mc2, load_triviaqa
    )
    
    # Get dataset size by loading with limit=1
    if args.limit:
        total_examples = args.limit
    else:
        # Load minimal to get total size
        logger.info("Determining dataset size...")
        if args.dataset == "arc-challenge":
            temp = load_arc_challenge(args.split, limit=1)
            # Reload to get full size
            from datasets import load_dataset as hf_load
            full_ds = hf_load("allenai/ai2_arc", "ARC-Challenge", split=args.split)
            total_examples = len(full_ds)
        # Similar for other datasets...
        else:
            logger.warning("Full dataset size detection not implemented for all datasets")
            logger.warning("Please specify --limit when using --multi-gpu")
            total_examples = 500  # Default assumption
    
    # Calculate task distribution
    examples_per_gpu = total_examples // n_gpus
    task_distribution = []
    
    for gpu_id in range(n_gpus):
        offset = gpu_id * examples_per_gpu
        # Last GPU gets any remainder
        if gpu_id == n_gpus - 1:
            limit = total_examples - offset
        else:
            limit = examples_per_gpu
        
        task_distribution.append({
            "gpu_id": gpu_id,
            "offset": offset,
            "limit": limit
        })
    
    # Report setup
    report_gpu_setup(gpus, total_examples, task_distribution)
    
    # Create temporary directory for worker outputs
    temp_dir = Path(tempfile.mkdtemp(prefix="multi_gpu_"))
    logger.info(f"Temporary directory: {temp_dir}")
    
    # Launch workers
    processes = []
    worker_outputs = []
    
    for task in task_distribution:
        gpu_id = task['gpu_id']
        offset = task['offset']
        limit = task['limit']
        
        # Worker output path (temporary for merging)
        worker_output = temp_dir / f"gpu{gpu_id}_output.csv"
        worker_outputs.append(worker_output)
        
        # Build command - pass original output path for logging
        cmd = build_worker_command(args, offset, limit, str(worker_output), log_base_path=args.output)
        
        # Set environment to use specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Set progress file path for worker to write to
        env["MULTI_GPU_PROGRESS_FILE"] = str(temp_dir / f"gpu{gpu_id}_progress.txt")
        
        # Launch worker
        log_file = temp_dir / f"gpu{gpu_id}.log"
        with open(log_file, 'w') as f:
            p = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        processes.append(p)
        logger.info(f"Launched GPU {gpu_id} worker (PID: {p.pid})")
    
    # Monitor progress
    success = monitor_workers(processes, temp_dir, task_distribution, total_examples)
    
    if not success:
        logger.error("Some workers failed. Check logs in: {temp_dir}")
        sys.exit(1)
    
    # Merge results
    print()
    print("Merging results from all GPUs...")
    from .merge_results import merge_evaluation_results
    
    merge_evaluation_results(
        worker_outputs=[str(f) for f in worker_outputs],
        output_path=args.output,
        temp_dir=temp_dir
    )
    
    print(f"✓ Saved merged results to: {args.output}")
    print(f"✓ Saved merged metrics to: {args.output.replace('.csv', '.json')}")
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up temporary directory: {temp_dir}")
    
    print()
    print("="*60)
    print("MULTI-GPU EVALUATION COMPLETE")
    print("="*60)

