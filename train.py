"""
Train threshold predictor strictly on BSDS500.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_extraction import FeatureExtractor
from src.threshold_predictor import ThresholdPredictor
from src.edge_detector import find_optimal_thresholds
from src.utils import load_image, get_image_files


def resolve_bsds_paths(project_dir: Path, preferred_root: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    candidates = []
    if preferred_root:
        candidates.append(preferred_root)
    candidates.append(project_dir / "data" / "raw" / "BSDS500")

    for root in candidates:
        if not root.exists():
            continue
        img_dir = root / "images"
        gt_dir = root / "groundTruth"
        if not gt_dir.exists():
            alt = root / "ground_truth"
            if alt.exists():
                gt_dir = alt
        if img_dir.exists() and gt_dir.exists():
            return img_dir, gt_dir
    return None, None


def load_ground_truth(gt_path: Path) -> np.ndarray:
    if gt_path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        data = loadmat(str(gt_path))
        gt_data = data.get("groundTruth")
        if gt_data is None:
            return None
        boundaries = []
        for i in range(gt_data.shape[1]):
            boundaries.append(gt_data[0, i]["Boundaries"][0, 0])
        gt = np.mean(boundaries, axis=0)
        return (gt > 0.5).astype(np.uint8) * 255
    return load_image(gt_path, grayscale=True)


def find_gt_for_image(image_path: Path, gt_split_dir: Path) -> Optional[Path]:
    stem = image_path.stem
    for ext in (".mat", ".png", ".jpg", ".jpeg", ".bmp"):
        p = gt_split_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_bsds_training_dataset(
    images_root: Path,
    gt_root: Path,
    split: str = "train",
    max_samples: Optional[int] = None,
    search_step: int = 10,
):
    image_split_dir = images_root / split
    gt_split_dir = gt_root / split
    image_paths = get_image_files(image_split_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_split_dir}")
    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    extractor = FeatureExtractor()
    features, thresholds = [], []

    for img_path in tqdm(image_paths, desc=f"Building {split} dataset"):
        gt_path = find_gt_for_image(img_path, gt_split_dir)
        if gt_path is None:
            continue

        image = load_image(img_path)
        gt = load_ground_truth(gt_path)
        if gt is None:
            continue

        if gt.shape[:2] != image.shape[:2]:
            gt = cv2.resize(gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        feat = extractor.extract_array(image)
        low, high, _ = find_optimal_thresholds(
            image=image,
            ground_truth=gt,
            low_range=(10, 120),
            high_range=(40, 250),
            step=search_step,
        )
        features.append(feat)
        thresholds.append([low, high])

    if not features:
        raise RuntimeError("No valid BSDS image/ground-truth pairs found.")

    return np.array(features), np.array(thresholds)


def visualize_training_results(predictor, features, thresholds, save_dir: Path):
    preds = predictor.predict_batch(features)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(thresholds[:, 0], preds[:, 0], alpha=0.5)
    axes[0].set_title("Low threshold")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Pred")
    axes[1].scatter(thresholds[:, 1], preds[:, 1], alpha=0.5)
    axes[1].set_title("High threshold")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Pred")
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "training_results.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Train threshold predictor on BSDS500 only.")
    parser.add_argument("--data-root", type=str, default=None, help="BSDS root containing images/ and groundTruth/")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit BSDS train images")
    parser.add_argument("--search-step", type=int, default=10, help="Grid-search step for optimal thresholds")
    parser.add_argument("--tune-hyperparams", action="store_true", help="Enable RF hyperparameter tuning")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    models_dir = project_dir / "models"
    results_dir = project_dir / "results" / "visualizations"
    models_dir.mkdir(parents=True, exist_ok=True)

    images_root, gt_root = resolve_bsds_paths(
        project_dir,
        Path(args.data_root) if args.data_root else None,
    )
    if images_root is None or gt_root is None:
        raise FileNotFoundError(
            "BSDS500 dataset not found. Place it under data/raw/BSDS500 with images/ and groundTruth/."
        )

    print(f"Using BSDS images: {images_root}")
    print(f"Using BSDS ground truth: {gt_root}")
    features, thresholds = build_bsds_training_dataset(
        images_root=images_root,
        gt_root=gt_root,
        split="train",
        max_samples=args.max_train_samples,
        search_step=args.search_step,
    )

    n_train = int(0.8 * len(features))
    idx = np.random.permutation(len(features))
    train_x, val_x = features[idx[:n_train]], features[idx[n_train:]]
    train_y, val_y = thresholds[idx[:n_train]], thresholds[idx[n_train:]]

    predictor = ThresholdPredictor()
    predictor.train(train_x, train_y, tune_hyperparams=args.tune_hyperparams, verbose=True)
    model_path = models_dir / "threshold_model.pkl"
    predictor.save(model_path)

    preds = predictor.predict_batch(val_x)
    mae = np.mean(np.abs(val_y - preds))
    print(f"Validation MAE: {mae:.2f}")
    visualize_training_results(predictor, val_x, val_y, results_dir)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
