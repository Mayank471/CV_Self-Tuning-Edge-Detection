"""
Comprehensive edge-detector benchmark on BSDS500.
Compares quality metrics + speed + robustness aspects.

New additions vs. original:
    - sobel, scharr, log, laplacian, morph_gradient, skimage_canny
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from skimage.feature import canny as sk_canny
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

sys.path.insert(0, str(Path(__file__).parent))

from src.edge_detector import AdaptiveCannyDetector, FixedThresholdDetector, OtsuThresholdDetector
from src.evaluation import EdgeEvaluator
from src.utils import (
    load_image,
    get_image_files,
    create_noisy_image,
    create_blurred_image,
    adjust_brightness,
)
from train import resolve_bsds_paths, load_ground_truth, find_gt_for_image


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if needed; return unchanged if already gray."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _binarize(edges: np.ndarray) -> np.ndarray:
    """Convert edges to binary (0 or 255)."""
    return (edges > 0).astype(np.uint8) * 255


def _otsu_thresholds(gray: np.ndarray) -> tuple:
    """Derive Canny thresholds from Otsu's method (low=0.5*T, high=T)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t_otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 0.5 * t_otsu, t_otsu


def detect_sobel(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude with Otsu threshold."""
    gray = _to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, np.empty_like(mag), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges.astype(np.uint8)


def detect_scharr(image: np.ndarray) -> np.ndarray:
    """Scharr gradient magnitude with Otsu threshold."""
    gray = _to_gray(image)
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, np.empty_like(mag), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges.astype(np.uint8)


def detect_log(image: np.ndarray) -> np.ndarray:
    """Laplacian-of-Gaussian with Otsu threshold."""
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    abs_lap = np.abs(lap)
    lap_u8 = cv2.normalize(abs_lap, np.empty_like(abs_lap), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(lap_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges.astype(np.uint8)


def detect_laplacian(image: np.ndarray) -> np.ndarray:
    """Laplacian with Otsu threshold."""
    gray = _to_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.abs(lap)
    lap_u8 = cv2.normalize(abs_lap, np.empty_like(abs_lap), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(lap_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges.astype(np.uint8)


def detect_morph_gradient(image: np.ndarray) -> np.ndarray:
    """Morphological gradient with Otsu threshold."""
    gray = _to_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges.astype(np.uint8)


def detect_skimage_canny(image: np.ndarray) -> np.ndarray:
    """scikit-image Canny with sigma=1.0 (requires scikit-image)."""
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image is not installed; skimage_canny unavailable.")
    global sk_canny
    gray = _to_gray(image).astype(np.float32) / 255.0
    edges = sk_canny(gray, sigma=1.0)
    return edges.astype(np.uint8) * 255


def make_detectors(model_path: Path) -> Dict:
    """
    Build and return the full dictionary of named detector callables.

    All callables accept a single np.ndarray (BGR image) and return a
    uint8 binary edge map (values 0 or 255).
    """
    adaptive = AdaptiveCannyDetector(model_path)
    fixed    = FixedThresholdDetector(50, 150)
    otsu     = OtsuThresholdDetector()

    detectors = {
        "adaptive_canny"           : lambda im: adaptive.detect(im).edges,
        "fixed_canny_50_150"       : lambda im: fixed.detect(im).edges,
        "otsu_canny"               : lambda im: otsu.detect(im).edges,
        "sobel"                    : detect_sobel,
        "scharr"                   : detect_scharr,
        "log"                      : detect_log,
        "laplacian"                : detect_laplacian,
        "morph_gradient"           : detect_morph_gradient,

    }

    if HAS_SKIMAGE:
        detectors["skimage_canny"] = detect_skimage_canny

    return detectors


def evaluate_method(
    evaluator: EdgeEvaluator,
    detector_fn,
    image: np.ndarray,
    gt: np.ndarray,
) -> Dict:
    """Run detector, measure wall time, compute metrics against ground truth."""
    t0 = time.perf_counter()
    edges = detector_fn(image)
    dt = (time.perf_counter() - t0) * 1000.0

    metrics               = evaluator.evaluate(_binarize(edges), gt).to_dict()
    metrics["runtime_ms"] = dt
    metrics["edge_density"] = float(np.mean(_binarize(edges) > 0))
    return metrics


def robustness_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate degraded image variants for robustness testing."""
    return {
        "original"          : image,
        "gaussian_noise"    : create_noisy_image(image, "gaussian", 25),
        "salt_pepper_noise" : create_noisy_image(image, "salt_pepper", 25),
        "blurred"           : create_blurred_image(image, 9),
        "bright"            : adjust_brightness(image, 1.3, 30),
        "dark"              : adjust_brightness(image, 0.7, -30),
    }


def summarize_values(values: List[float]) -> Dict[str, float]:
    """Calculate mean and std of float values."""
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def run_benchmark(
    images_root : Path,
    gt_root     : Path,
    split       : str,
    model_path  : Path,
    max_samples : Optional[int],
) -> Dict:
    """
    Iterate over images, run every detector, collect metrics per method.
    Returns metrics aggregated as mean±std per method.
    """
    evaluator = EdgeEvaluator(tolerance=2)
    detectors = make_detectors(model_path)

    image_paths = get_image_files(images_root / split)
    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    per_method = {
        name: {
            "precision"           : [],
            "recall"              : [],
            "f1_score"            : [],
            "accuracy"            : [],
            "runtime_ms"          : [],
            "edge_density"        : [],
            "robustness_f1_drop"  : [],
        }
        for name in detectors.keys()
    }

    sample_count = 0
    for img_path in image_paths:
        gt_path = find_gt_for_image(img_path, gt_root / split)
        if gt_path is None:
            continue
        image = load_image(img_path)
        gt = load_ground_truth(gt_path)
        if gt is None:
            continue
        if gt.shape[:2] != image.shape[:2]:
            gt = cv2.resize(
                gt, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        variants = robustness_variants(image)
        sample_count += 1

        for name, fn in detectors.items():
            base = evaluate_method(evaluator, fn, variants["original"], gt)
            for k in ("precision", "recall", "f1_score", "accuracy",
                      "runtime_ms", "edge_density"):
                per_method[name][k].append(base[k])

            robustness_f1 = []
            for vname, vimg in variants.items():
                if vname == "original":
                    continue
                vm = evaluate_method(evaluator, fn, vimg, gt)
                robustness_f1.append(vm["f1_score"])
            drop = base["f1_score"] - float(np.mean(robustness_f1))
            per_method[name]["robustness_f1_drop"].append(drop)

    if sample_count == 0:
        raise RuntimeError(f"No valid samples found in split '{split}'.")

    summary = {}
    for name, vals in per_method.items():
        summary[name] = {
            metric: summarize_values(metric_vals)
            for metric, metric_vals in vals.items()
        }
        summary[name]["samples"] = sample_count

    return summary


def format_table(summary: Dict[str, Dict]) -> str:
    """Render a plain-text summary table for console output."""
    headers = [
        "Method",
        "F1(mean±std)",
        "Precision",
        "Recall",
        "Runtime(ms)",
        "RobustDrop",
    ]
    lines = [" | ".join(headers), "-" * 120]
    for method, s in summary.items():
        lines.append(
            " | ".join([
                f"{method:<35}",
                f"{s['f1_score']['mean']:.4f}±{s['f1_score']['std']:.4f}",
                f"{s['precision']['mean']:.4f}",
                f"{s['recall']['mean']:.4f}",
                f"{s['runtime_ms']['mean']:.2f}",
                f"{s['robustness_f1_drop']['mean']:.4f}",
            ])
        )
    return "\n".join(lines)


def save_method_comparisons(
    images_root: Path,
    gt_root: Path,
    split: str,
    detectors: Dict[str, Callable],
    out_dir: Path,
    max_images: Optional[int] = None,
) -> int:
    """Save side-by-side visual comparisons for all valid images in a split."""
    image_paths = get_image_files(images_root / split)
    saved = 0

    for img_path in image_paths:
        if max_images is not None and saved >= max_images:
            break

        gt_path = find_gt_for_image(img_path, gt_root / split)
        if gt_path is None:
            continue

        image = load_image(img_path)
        gt = load_ground_truth(gt_path)
        if gt is None:
            continue
        if gt.shape[:2] != image.shape[:2]:
            gt = cv2.resize(gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        method_names = list(detectors.keys())
        n_panels = 2 + len(method_names)  # original + gt + methods
        n_cols = 4
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
        axes = np.array(axes).reshape(-1)

        # Original image
        if image.ndim == 3:
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(image, cmap="gray")
        axes[0].set_title("original")
        axes[0].axis("off")

        # Ground truth
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("ground_truth")
        axes[1].axis("off")

        # Method outputs
        for idx, name in enumerate(method_names, start=2):
            edges = _binarize(detectors[name](image))
            axes[idx].imshow(edges, cmap="gray")
            axes[idx].set_title(name)
            axes[idx].axis("off")

        # Hide any extra axes
        for idx in range(n_panels, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(f"All-technique comparison: {img_path.name}", fontsize=12)
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"comparison_all_{img_path.stem}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive BSDS500 edge-detector benchmark."
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="BSDS root directory containing images/ and groundTruth/",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--model", type=str, default="models/threshold_model.pkl",
        help="Path to the trained AdaptiveCannyDetector model (.pkl)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation to first N images (useful for quick testing)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to append the text summary log (default: results/logs/compare_<split>.log)",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Optional path to save the full JSON summary",
    )
    parser.add_argument(
        "--save-comparisons", action="store_true",
        help="Save visual grids showing original, GT, and all methods",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    images_root, gt_root = resolve_bsds_paths(
        project_dir,
        Path(args.data_root) if args.data_root else None,
    )
    if images_root is None or gt_root is None:
        raise FileNotFoundError(
            "BSDS dataset not found. "
            "Expected images/ and groundTruth/ (or ground_truth/) under the data root."
        )

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_dir / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    if args.save_comparisons:
        comparison_dir = project_dir / "results" / "visualizations"
        detectors = make_detectors(model_path)
        saved = save_method_comparisons(
            images_root=images_root,
            gt_root=gt_root,
            split=args.split,
            detectors=detectors,
            out_dir=comparison_dir,
        )
        print(f"Saved {saved} all-technique comparison image(s) to: {comparison_dir}")

    summary = run_benchmark(
        images_root = images_root,
        gt_root     = gt_root,
        split       = args.split,
        model_path  = model_path,
        max_samples = args.max_samples,
    )

    if not HAS_SKIMAGE:
        print("\nNote: scikit-image is not installed — 'skimage_canny' was skipped.")

    timestamp = datetime.now().isoformat(timespec="seconds")
    header = "\n".join([
        f"Timestamp : {timestamp}",
        f"Model     : {model_path}",
        f"Data root : {images_root.parent}",
        f"Split     : {args.split}",
        "",
        format_table(summary),
    ])
    print("\n" + header)

    # --- write text log ---
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = project_dir / log_path
    else:
        log_path = project_dir / "results" / "logs" / f"compare_{args.split}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(header + "\n" + ("=" * 120) + "\n")
    print(f"\nAppended log to: {log_path}")

    # --- optional JSON dump ---
    if args.json_out:
        json_path = Path(args.json_out)
        if not json_path.is_absolute():
            json_path = project_dir / json_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved JSON summary to: {json_path}")


if __name__ == "__main__":
    main()