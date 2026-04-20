"""
Evaluation Module for Edge Detection.

Provides metrics for comparing edge detection results against ground truth
and evaluating the performance of different edge detection methods.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy
        }


class EdgeEvaluator:
    """
    Evaluates edge detection results against ground truth.
    """
    
    def __init__(self, tolerance: int = 2):
        """
        Initialize the evaluator.
        
        Args:
            tolerance: Pixel tolerance for edge matching (default 2)
                      Edges within this distance are considered matches
        """
        self.tolerance = tolerance
    
    def evaluate(self, 
                 predicted: np.ndarray, 
                 ground_truth: np.ndarray,
                 use_tolerance: bool = True) -> EvaluationMetrics:
        """
        Evaluate predicted edges against ground truth.
        
        Args:
            predicted: Predicted edge image (binary or 0-255)
            ground_truth: Ground truth edge image
            use_tolerance: Whether to use pixel tolerance for matching
            
        Returns:
            EvaluationMetrics object
        """
        # Binarize images
        pred_binary = (predicted > 0).astype(np.uint8)
        gt_binary = (ground_truth > 0).astype(np.uint8)
        
        if use_tolerance and self.tolerance > 0:
            # Dilate ground truth for tolerance-based matching
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (2 * self.tolerance + 1, 2 * self.tolerance + 1)
            )
            gt_dilated = cv2.dilate(gt_binary, kernel)
            pred_dilated = cv2.dilate(pred_binary, kernel)
            
            # True positives: predicted edges within tolerance of ground truth
            tp = np.sum((pred_binary == 1) & (gt_dilated == 1))
            # False positives: predicted edges not near any ground truth
            fp = np.sum((pred_binary == 1) & (gt_dilated == 0))
            # False negatives: ground truth edges not detected
            fn = np.sum((gt_binary == 1) & (pred_dilated == 0))
        else:
            # Exact pixel matching
            tp = np.sum((pred_binary == 1) & (gt_binary == 1))
            fp = np.sum((pred_binary == 1) & (gt_binary == 0))
            fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy
        )
    
    def evaluate_batch(self,
                       predictions: List[np.ndarray],
                       ground_truths: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted edge images
            ground_truths: List of ground truth edge images
            
        Returns:
            Dictionary with mean metrics
        """
        all_metrics = []
        
        for pred, gt in zip(predictions, ground_truths):
            metrics = self.evaluate(pred, gt)
            all_metrics.append(metrics)
        
        # Calculate mean metrics
        return {
            'mean_precision': np.mean([m.precision for m in all_metrics]),
            'mean_recall': np.mean([m.recall for m in all_metrics]),
            'mean_f1': np.mean([m.f1_score for m in all_metrics]),
            'mean_accuracy': np.mean([m.accuracy for m in all_metrics]),
            'std_f1': np.std([m.f1_score for m in all_metrics])
        }
    
    def compare_methods(self,
                        image: np.ndarray,
                        ground_truth: np.ndarray,
                        methods: Dict[str, np.ndarray]) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple edge detection methods.
        
        Args:
            image: Original image (for reference)
            ground_truth: Ground truth edges
            methods: Dictionary mapping method names to edge images
            
        Returns:
            Dictionary mapping method names to their metrics
        """
        results = {}
        
        for name, edges in methods.items():
            metrics = self.evaluate(edges, ground_truth)
            results[name] = metrics
        
        return results


def compute_precision_recall_curve(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve for edge detection.
    
    Args:
        predicted: Predicted edge probability map (0-255)
        ground_truth: Binary ground truth
        thresholds: Thresholds to evaluate (optional)
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 255, 51)
    
    gt_binary = (ground_truth > 0).astype(np.uint8)
    
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        pred_binary = (predicted >= thresh).astype(np.uint8)
        
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls), thresholds


def plot_comparison_results(
    results: Dict[str, EvaluationMetrics],
    title: str = "Edge Detection Method Comparison",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot comparison bar chart of different methods.
    
    Args:
        results: Dictionary mapping method names to metrics
        title: Plot title
        save_path: Path to save figure (optional)
    """
    methods = list(results.keys())
    metrics = ['precision', 'recall', 'f1_score']
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        values = [getattr(results[m], metric) for m in methods]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def evaluate_robustness(
    detector,
    image: np.ndarray,
    ground_truth: np.ndarray,
    evaluator: EdgeEvaluator
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate detector robustness under various conditions.
    
    Args:
        detector: Edge detector with detect() method
        image: Original image
        ground_truth: Ground truth edges
        evaluator: EdgeEvaluator instance
        
    Returns:
        Dictionary with metrics for each test condition
    """
    from .utils import create_noisy_image, create_blurred_image, adjust_brightness
    
    results = {}
    
    # Original
    edges_orig = detector.detect(image).edges
    results['original'] = evaluator.evaluate(edges_orig, ground_truth).to_dict()
    
    # Noisy (Gaussian)
    noisy_img = create_noisy_image(image, 'gaussian', 25)
    edges_noisy = detector.detect(noisy_img).edges
    results['gaussian_noise'] = evaluator.evaluate(edges_noisy, ground_truth).to_dict()
    
    # Salt & Pepper noise
    sp_img = create_noisy_image(image, 'salt_pepper', 25)
    edges_sp = detector.detect(sp_img).edges
    results['salt_pepper_noise'] = evaluator.evaluate(edges_sp, ground_truth).to_dict()
    
    # Blurred
    blurred_img = create_blurred_image(image, 9)
    edges_blur = detector.detect(blurred_img).edges
    results['blurred'] = evaluator.evaluate(edges_blur, ground_truth).to_dict()
    
    # Bright
    bright_img = adjust_brightness(image, 1.3, 30)
    edges_bright = detector.detect(bright_img).edges
    results['bright'] = evaluator.evaluate(edges_bright, ground_truth).to_dict()
    
    # Dark
    dark_img = adjust_brightness(image, 0.7, -30)
    edges_dark = detector.detect(dark_img).edges
    results['dark'] = evaluator.evaluate(edges_dark, ground_truth).to_dict()
    
    return results
