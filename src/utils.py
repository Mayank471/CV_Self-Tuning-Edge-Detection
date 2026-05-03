"""
Utility functions for image loading, saving, and visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple, List, Optional


def load_image(path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        path: Path to the image file
        grayscale: If True, load as grayscale
        
    Returns:
        Image as numpy array (BGR if color, grayscale otherwise)
    """
    path = str(path)
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    
    return img


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save an image to disk.
    
    Args:
        image: Image array to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 range."""
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 0:
        return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return image.astype(np.uint8)  # Already normalized


def visualize_comparison(
    original: np.ndarray,
    edges_adaptive: np.ndarray,
    edges_fixed: np.ndarray,
    edges_otsu: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Edge Detection Comparison",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Visualize comparison between different edge detection methods.
    
    Args:
        original: Original image
        edges_adaptive: Edges from adaptive method
        edges_fixed: Edges from fixed threshold
        edges_otsu: Edges from Otsu method (optional)
        ground_truth: Ground truth edges (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    n_cols = 3
    if edges_otsu is not None:
        n_cols += 1
    if ground_truth is not None:
        n_cols += 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Original image
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Adaptive edges
    axes[1].imshow(edges_adaptive, cmap='gray')
    axes[1].set_title('Adaptive (Ours)')
    axes[1].axis('off')
    
    # Fixed threshold edges
    axes[2].imshow(edges_fixed, cmap='gray')
    axes[2].set_title('Fixed Threshold')
    axes[2].axis('off')
    
    col_idx = 3
    
    if edges_otsu is not None:
        axes[col_idx].imshow(edges_otsu, cmap='gray')
        axes[col_idx].set_title('Otsu Method')
        axes[col_idx].axis('off')
        col_idx += 1
    
    if ground_truth is not None:
        axes[col_idx].imshow(ground_truth, cmap='gray')
        axes[col_idx].set_title('Ground Truth')
        axes[col_idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_noisy_image(image: np.ndarray, noise_type: str = 'gaussian', 
                       intensity: float = 25.0) -> np.ndarray:
    """
    Add noise to an image for robustness testing.
    
    Args:
        image: Input image
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle')
        intensity: Noise intensity
        
    Returns:
        Noisy image
    """
    img = image.astype(np.float32)
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, img.shape)
        noisy = img + noise
    elif noise_type == 'salt_pepper':
        noisy = img.copy()
        # Salt
        salt_mask = np.random.random(img.shape) < intensity / 500
        noisy[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(img.shape) < intensity / 500
        noisy[pepper_mask] = 0
    elif noise_type == 'speckle':
        noise = np.random.randn(*img.shape) * intensity / 100
        noisy = img + img * noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return np.clip(noisy, 0, 255).astype(np.uint8)


def create_blurred_image(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input image
        kernel_size: Blur kernel size (odd number)
        
    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness(image: np.ndarray, factor: float = 1.0, 
                      offset: int = 0) -> np.ndarray:
    """
    Adjust image brightness.
    
    Args:
        image: Input image
        factor: Multiplicative factor
        offset: Additive offset
        
    Returns:
        Brightness-adjusted image
    """
    adjusted = image.astype(np.float32) * factor + offset
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def get_image_files(directory: Union[str, Path], 
                    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Path]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: Valid image extensions
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(files)
