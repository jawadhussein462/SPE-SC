import torch
from rouge_score import rouge_scorer
from typing import Dict

# Initialize ROUGE-L scorer globally
rougeL_scorer = rouge_scorer.RougeScorer(["rougeL"])

# Device configuration (assuming CUDA availability check)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Move data dictionary to the specified device (GPU/CPU)."""
    return {k: v.to(DEVICE) for k, v in data.items()}


def rouge_recall(instruction: str, completion: str) -> float:
    """Calculate ROUGE-L recall between instruction and completion."""
    rouge_eval = rougeL_scorer.score(instruction, completion)
    return rouge_eval["rougeL"].recall


def approx_match(prediction: str, ground_truth: str,
                 threshold: float = 0.9) -> bool:
    """
    Approximate match metric based on ROUGE-L score.

    Returns 1 if the ratio of LCS(tokens(p), tokens(g)) / |tokens(p)|
    >= threshold, 0 otherwise.

    This is implemented using ROUGE-L recall which measures:
    LCS(prediction_tokens, ground_truth_tokens) / len(prediction_tokens)

    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        threshold: Threshold for approximate match (default: 0.9)

    Returns:
        bool: True if approximate match criterion is satisfied
    """
    rouge_recall_score = rouge_recall(prediction, ground_truth)
    return rouge_recall_score >= threshold