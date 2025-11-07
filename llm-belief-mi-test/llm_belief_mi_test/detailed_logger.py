"""
Detailed logging module for saving comprehensive execution traces.

Similar to demo system but designed for actual evaluation runs.
Saves per-question details including prompts, raw outputs, and decision processes.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class DetailedLogger:
    """Logger for saving detailed per-question execution traces."""
    
    def __init__(self, output_dir: Path, method_name: str):
        """
        Initialize detailed logger.
        
        Args:
            output_dir: Directory to save question JSON files
            method_name: Name of method being logged (e.g., "greedy", "mi_method")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name
        
    def log_question(
        self,
        question_id: int,
        question_text: str,
        choices: List[str],
        choice_texts: List[str],
        gold_answer: str,
        method_data: Dict[str, Any]
    ):
        """
        Save detailed trace for a single question.
        
        Args:
            question_id: Question index
            question_text: The question text
            choices: Choice letters (e.g., ["A", "B", "C", "D"])
            choice_texts: Choice text descriptions
            gold_answer: Correct answer letter
            method_data: Method-specific data containing:
                - raw_inputs: List of prompts sent to model
                - raw_outputs: List of model responses
                - decision_process: How decision was made
                - final_metrics: Predicted answer, correctness, confidence, etc.
        """
        # Format choices for display
        formatted_choices = [
            f"{letter}: {text}" 
            for letter, text in zip(choices, choice_texts)
        ]
        
        # Build JSON structure (similar to demo)
        question_data = {
            "question_id": question_id,
            "question_text": question_text,
            "choices": formatted_choices,
            "gold_answer": gold_answer,
            "methods": {
                self.method_name: method_data
            }
        }
        
        # Save to file
        output_file = self.output_dir / f"question_{question_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, indent=2, ensure_ascii=False)
            
    @staticmethod
    def create_from_output_path(output_csv_path: str, method: str) -> 'DetailedLogger':
        """
        Create logger from output CSV path.
        
        Args:
            output_csv_path: Path to output CSV (e.g., "outputs/results/dataset/method_200.csv")
            method: Method name (e.g., "mi", "greedy", "self-consistency")
            
        Returns:
            DetailedLogger instance
            
        Example:
            output_csv_path = "outputs/results/truthfulqa/greedy_200.csv"
            → logs saved to "outputs/logs/truthfulqa_greedy_200/question_0.json", etc.
        """
        csv_path = Path(output_csv_path)
        run_name = csv_path.stem  # e.g., "greedy_200"
        
        # Extract dataset name from parent folder (if it exists)
        # For paths like "outputs/results/truthfulqa/greedy_200.csv"
        # we want "truthfulqa_greedy_200" as the log folder name
        dataset_folder = csv_path.parent.name
        if dataset_folder != "results" and dataset_folder != "test":
            # Include dataset name in log folder
            log_folder_name = f"{dataset_folder}_{run_name}"
        else:
            # Fallback for flat structure (e.g., "outputs/results/test.csv")
            log_folder_name = run_name
        
        # Determine base directory - go up to find "outputs" level
        # Navigate from results folder to logs folder
        if "results" in csv_path.parts:
            # Find the index of "results" and replace with "logs"
            parts = list(csv_path.parts)
            results_idx = parts.index("results")
            base_dir = Path(*parts[:results_idx]) / "logs" / log_folder_name
        else:
            # Fallback: put logs next to output file parent
            base_dir = csv_path.parent.parent / "logs" / log_folder_name
        
        # Map method names to standard format
        method_map = {
            "mi": "mi_method",
            "greedy": "greedy",
            "self-consistency": "self_consistency",
            "semantic-entropy": "semantic_entropy",
            "self-verification": "self_verification"
        }
        method_name = method_map.get(method, method)
        
        return DetailedLogger(base_dir, method_name)

