"""
Adaptive Canny Edge Detector Module.

Implements a self-tuning edge detector that automatically selects
optimal thresholds based on image quality features.

Fixes applied:
  1. find_optimal_thresholds: added L2gradient=True so training labels are
     computed with the same gradient norm as inference.
  2. FixedThresholdDetector.detect: added L2gradient=True for a fair
     apples-to-apples comparison with the adaptive method.
  3. OtsuThresholdDetector.detect: same fix as above.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass

from .feature_extraction import FeatureExtractor, ImageFeatures
from .threshold_predictor import ThresholdPredictor, ThresholdPrediction


@dataclass
class EdgeDetectionResult:
    """Container for edge detection results."""
    edges: np.ndarray
    low_threshold: float
    high_threshold: float
    features: ImageFeatures
    method: str = "adaptive"


class AdaptiveCannyDetector:
    """
    Self-tuning Canny edge detector using image quality-aware threshold prediction.

    Automatically adjusts thresholds based on the image's quality
    characteristics (brightness, contrast, noise, etc.) to produce
    consistent edge detection results across varying imaging conditions.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.feature_extractor = FeatureExtractor()
        self.threshold_predictor: Optional[ThresholdPredictor] = None
        if model_path is not None:
            self.threshold_predictor = ThresholdPredictor(model_path)

    def set_predictor(self, predictor: ThresholdPredictor) -> None:
        self.threshold_predictor = predictor

    def detect(self,
               image: np.ndarray,
               aperture_size: int = 3,
               L2gradient: bool = True) -> EdgeDetectionResult:
        """
        Detect edges using adaptive thresholds.

        Args:
            image: Input image (grayscale or BGR).
            aperture_size: Sobel aperture size (3, 5, or 7).
            L2gradient: Use L2 norm for gradient calculation (default True).

        Returns:
            EdgeDetectionResult containing edges and metadata.
        """
        if self.threshold_predictor is None:
            raise RuntimeError(
                "No threshold predictor set. Load a model or call set_predictor()."
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        features = self.feature_extractor.extract(gray)
        prediction = self.threshold_predictor.predict(features.to_array())
        low_thresh, high_thresh = prediction.to_int_tuple()

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, low_thresh, high_thresh,
                          apertureSize=aperture_size, L2gradient=L2gradient)

        return EdgeDetectionResult(
            edges=edges,
            low_threshold=low_thresh,
            high_threshold=high_thresh,
            features=features,
            method="adaptive",
        )

    def detect_with_thresholds(self,
                               image: np.ndarray,
                               low_threshold: int,
                               high_threshold: int,
                               aperture_size: int = 3,
                               L2gradient: bool = True) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        return cv2.Canny(blurred, low_threshold, high_threshold,
                         apertureSize=aperture_size, L2gradient=L2gradient)


class FixedThresholdDetector:
    """
    Standard Canny detector with fixed thresholds for baseline comparison.

    Fix: L2gradient=True is now used to match the adaptive detector and the
    training-label generation routine so all methods operate on the same
    gradient norm.
    """

    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def detect(self, image: np.ndarray) -> EdgeDetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        # FIX: was cv2.Canny(blurred, ...) with default L2gradient=False.
        # Now matches AdaptiveCannyDetector for a fair comparison.
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold,
                          L2gradient=True)
        return EdgeDetectionResult(
            edges=edges,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            features=None,
            method="fixed",
        )


class OtsuThresholdDetector:
    """
    Canny detector using a sigma-median heuristic for automatic threshold selection.

    Note: despite the class name, this uses the median-sigma rule, not Otsu's
    method on the gradient magnitude. Rename to SigmaMedianDetector if desired.

    Fix: L2gradient=True added for consistency with AdaptiveCannyDetector.
    """

    def __init__(self, sigma: float = 0.33):
        self.sigma = sigma

    def detect(self, image: np.ndarray) -> EdgeDetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        median = np.median(gray)
        low_threshold = int(max(0, (1.0 - self.sigma) * median))
        high_threshold = int(min(255, (1.0 + self.sigma) * median))

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        # FIX: was cv2.Canny(blurred, ...) with default L2gradient=False.
        edges = cv2.Canny(blurred, low_threshold, high_threshold, L2gradient=True)

        return EdgeDetectionResult(
            edges=edges,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            features=None,
            method="otsu",
        )


def find_optimal_thresholds(
    image: np.ndarray,
    ground_truth: np.ndarray,
    low_range: Tuple[int, int] = (10, 100),
    high_range: Tuple[int, int] = (50, 200),
    step: int = 10,
) -> Tuple[int, int, float]:
    """
    Find optimal Canny thresholds via grid search.

    Used for generating training labels by finding the threshold pair that
    maximises F1 against the ground truth.

    Fix: L2gradient=True is now passed to cv2.Canny so the training labels are
    generated under exactly the same gradient-norm assumption used at inference
    time.  The previous default (L2gradient=False) caused a systematic shift
    in the threshold distribution that degraded prediction quality.

    Args:
        image: Input image (BGR or grayscale).
        ground_truth: Binary ground-truth edge image (uint8, 0/255).
        low_range: (min, max) for low threshold search.
        high_range: (min, max) for high threshold search.
        step: Grid-search step size.

    Returns:
        (best_low, best_high, best_f1_score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    gt_binary = (ground_truth > 0).astype(np.uint8)

    best_low, best_high, best_f1 = 50, 150, 0.0

    for low in range(low_range[0], low_range[1] + 1, step):
        for high in range(max(high_range[0], low + 10), high_range[1] + 1, step):
            # FIX: was cv2.Canny(blurred, low, high) — default L2gradient=False.
            # Must match AdaptiveCannyDetector.detect() which uses L2gradient=True.
            edges = cv2.Canny(blurred, low, high, L2gradient=True)
            edges_binary = (edges > 0).astype(np.uint8)

            tp = np.sum((edges_binary == 1) & (gt_binary == 1))
            fp = np.sum((edges_binary == 1) & (gt_binary == 0))
            fn = np.sum((edges_binary == 0) & (gt_binary == 1))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            if f1 > best_f1:
                best_f1 = f1
                best_low = low
                best_high = high

    return best_low, best_high, best_f1