"""
Image Quality Feature Extraction Module.

Extracts 6 features from images for threshold prediction:
  1. Brightness - mean normalised intensity
  2. Contrast - std deviation normalised
  3. Noise level - high-frequency residual std
  4. Entropy - Shannon entropy of histogram
  5. Blur level - inverse of Laplacian variance
  6. Edge density - fraction of edge pixels in pre-blurred image
"""

import cv2
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ImageFeatures:
    """Container for extracted image features."""
    brightness: float
    contrast: float
    noise_level: float
    entropy: float
    blur_level: float
    edge_density: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.brightness,
            self.contrast,
            self.noise_level,
            self.entropy,
            self.blur_level,
            self.edge_density,
        ])

    def to_dict(self) -> Dict[str, float]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "noise_level": self.noise_level,
            "entropy": self.entropy,
            "blur_level": self.blur_level,
            "edge_density": self.edge_density,
        }


class FeatureExtractor:
    """
    Extracts image quality features for adaptive threshold prediction.
    """

    FEATURE_NAMES = ["brightness", "contrast", "noise_level",
                     "entropy", "blur_level", "edge_density"]

    def __init__(self):
        pass  # No initialization needed

    def extract(self, image: np.ndarray) -> ImageFeatures:
        """
        Extract all 6 features from an image.

        Args:
            image: Grayscale (HxW) or BGR (HxWx3) uint8 image.

        Returns:
            ImageFeatures dataclass.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        gray_f = gray.astype(np.float64)

        return ImageFeatures(
            brightness=self._extract_brightness(gray_f),
            contrast=self._extract_contrast(gray_f),
            noise_level=self._extract_noise_level(gray),
            entropy=self._extract_entropy(gray),
            blur_level=self._extract_blur_level(gray),
            edge_density=self._extract_edge_density(gray),
        )

    def extract_array(self, image: np.ndarray) -> np.ndarray:
        return self.extract(image).to_array()

    def _extract_brightness(self, gray: np.ndarray) -> float:
        """Mean normalised intensity (0-1)."""
        return float(np.mean(gray) / 255.0)

    def _extract_contrast(self, gray: np.ndarray) -> float:
        """Std deviation normalised by half range."""
        return float(np.std(gray) / 128.0)

    def _extract_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise via high-frequency residual std (0-1 range)."""
        smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float64)
        residual = gray.astype(np.float64) - smoothed
        sigma = float(np.std(residual))
        return min(sigma / 25.0, 1.0)

    def _extract_entropy(self, gray: np.ndarray) -> float:
        """Shannon entropy of the intensity histogram, normalised to [0, 1]."""
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        h = hist.astype(np.float64)
        h /= h.sum()
        h = h[h > 0]
        entropy = float(-np.sum(h * np.log2(h)))
        return entropy / 8.0  # max possible entropy for 256 bins is 8 bits

    def _extract_blur_level(self, gray: np.ndarray) -> float:
        """Blur level via inverse Laplacian variance (0=sharp, 1=blurry)."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(lap.var())
        sharpness = min(variance / 500.0, 1.0)
        return 1.0 - sharpness

    def _extract_edge_density(self, gray: np.ndarray) -> float:
        """Fraction of edge pixels detected with loose Canny thresholds (0-1)."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, 30, 100)
        return float(np.sum(edges > 0) / edges.size)




def extract_features(image: np.ndarray) -> np.ndarray:
    """Return 1-D feature array (6,) for a single image."""
    return FeatureExtractor().extract_array(image)


def extract_features_batch(images: List[np.ndarray]) -> np.ndarray:
    """Return feature matrix of shape (n_images, 6)."""
    extractor = FeatureExtractor()
    return np.array([extractor.extract_array(img) for img in images])