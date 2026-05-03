# Self-Tuning Edge Detection (BSDS500)

This project builds and evaluates a self-tuning Canny edge detector.

The core idea is to predict Canny low/high thresholds from image features instead of using fixed values. A regression model is trained on BSDS500 so threshold selection adapts per image.

## What This Repository Contains

- Training pipeline for threshold prediction on BSDS500: `train.py`
- Benchmark script comparing adaptive and classical edge methods: `compare_methods.py`
- Reusable detector, feature, and evaluation modules in `src/`

## Method Overview

1. Extract global image features (contrast, statistics, gradients, etc.).
2. For each BSDS training image, search for near-optimal Canny thresholds against ground truth boundaries.
3. Train a Random Forest regressor to map features -> `(low_threshold, high_threshold)`.
4. Use predicted thresholds at inference time for adaptive Canny.
5. Compare against baselines on BSDS split(s) using quality, runtime, and robustness metrics.

## Repository Structure

```text
Project/
├── src/
│   ├── edge_detector.py
│   ├── evaluation.py
│   ├── feature_extraction.py
│   ├── threshold_predictor.py
│   └── utils.py
├── data/
│   ├── raw/BSDS500/
│   └── processed/
├── models/
├── results/
│   ├── logs/
│   └── visualizations/
├── train.py
├── compare_methods.py
├── requirements.txt
└── README.md
```

## Dataset Layout (Required)

Expected BSDS500 layout:

```text
data/raw/BSDS500/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── groundTruth/        # or: ground_truth/
    ├── train/
    ├── val/
    └── test/
```

Ground truth can be `.mat` (BSDS native) or image files.

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Quick Start (End-to-End)

If your BSDS500 folder is already available at `data/raw/BSDS500`, run:

```powershell
# 1) Train the threshold predictor
python train.py --data-root data/raw/BSDS500 --max-train-samples 200

# 2) Benchmark all methods on test split
python compare_methods.py --data-root data/raw/BSDS500 --split test --max-samples 50 --json-out results/logs/compare_test_50.json

# 3) (Optional) Save visual grids for qualitative comparison
python compare_methods.py --data-root data/raw/BSDS500 --split test --max-samples 20 --save-comparisons
```

## Training

Train the adaptive threshold predictor:

```powershell
python train.py --data-root data/raw/BSDS500 --max-train-samples 200
```

Optional hyperparameter tuning:

```powershell
python train.py --data-root data/raw/BSDS500 --max-train-samples 200 --tune-hyperparams
```

Useful training arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-root` | `str` | auto-detect | BSDS root containing `images/` and `groundTruth/` |
| `--max-train-samples` | `int` | `None` | Limit number of train images for faster experiments |
| `--search-step` | `int` | `10` | Threshold grid-search step (smaller = finer but slower) |
| `--tune-hyperparams` | flag | `False` | Enable Random Forest hyperparameter tuning |

Training outputs:

- Model: `models/threshold_model.pkl`
- Validation plot: `results/visualizations/training_results.png`

## Benchmark and Comparison

Run benchmark on BSDS test split:

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test
```

Quick benchmark (fewer images):

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test --max-samples 50
```

Save JSON summary:

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test --json-out results/logs/compare_test.json
```

Save visual comparison grids:

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test --save-comparisons
```

Custom log file path:

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test --log-file results/logs/my_run.log
```

Benchmark arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-root` | `str` | auto-detect | BSDS root containing images and ground truth |
| `--split` | `str` | `test` | One of `train`, `val`, `test` |
| `--model` | `str` | `models/threshold_model.pkl` | Path to trained adaptive model |
| `--max-samples` | `int` | `None` | Evaluate only first N images |
| `--log-file` | `str` | `results/logs/compare_<split>.log` | Text benchmark log destination |
| `--json-out` | `str` | `None` | Save full metrics summary as JSON |
| `--save-comparisons` | flag | `False` | Save side-by-side visual grids |

## Methods Compared

Depending on installed packages, benchmark includes:

- `adaptive_canny` (learned thresholds)
- `fixed_canny_50_150`
- `otsu_canny`
- `sobel`
- `scharr`
- `log` (Laplacian of Gaussian)
- `laplacian`
- `morph_gradient`
- `skimage_canny` (only when `scikit-image` is installed)

## Reported Metrics

Per method, the benchmark summarizes:

- Precision
- Recall
- F1 score
- Accuracy
- Runtime per image (ms)
- Edge density
- Robustness F1 drop under degraded variants (noise, blur, brightness changes)

## Results and Logs

Default output locations:

- Text log: `results/logs/compare_<split>.log`
- Optional JSON: path provided via `--json-out`
- Visual comparisons: `results/visualizations/`

Example summary format (console/log):

```text
Method                              | F1(mean+-std)   | Precision | Recall  | Runtime(ms) | RobustDrop
--------------------------------------------------------------------------------------------------------
adaptive_canny                      | 0.7312+-0.0581  | 0.7460    | 0.7218  | 7.41        | 0.0435
fixed_canny_50_150                  | 0.6845+-0.0649  | 0.7017    | 0.6732  | 1.96        | 0.0698
otsu_canny                          | 0.7011+-0.0612  | 0.7194    | 0.6901  | 2.58        | 0.0561
```

## How To Interpret Results

- Higher F1 usually means better balance between precision and recall.
- Lower runtime means faster inference.
- Lower robustness drop means more stable behavior under noise/blur/lighting changes.
- For practical deployment, prioritize methods on a Pareto trade-off: quality (F1) vs speed (runtime).

## Suggested Experiment Protocol

1. Train once on BSDS train split.
2. Evaluate primarily on BSDS test split.
3. Use `--max-samples` for quick iteration, then run full split for final report.
4. Export JSON summaries and keep run logs for traceability.
5. Include both quantitative metrics and saved visual comparisons in your report.

## Project Limitations

- Current training relies on hand-crafted global image features.
- Threshold search can be compute-heavy when `--search-step` is small.
- Ground truth quality and annotation ambiguity may affect upper-bound metrics.
- No GPU-specific acceleration is used in this baseline workflow.

## Future Improvements

- Add cross-validation and fixed random seeds for stronger reproducibility.
- Add per-category analysis (indoor/outdoor, texture level, lighting conditions).
- Benchmark additional modern edge methods.
- Add experiment tracking (for example with CSV/MLflow/WandB-style logs).
- Package as a small CLI tool for easier reuse.

## Citation

If you use this repository in academic coursework or a report, cite it as:

```text
Self-Tuning Edge Detection (BSDS500), Mayank471, GitHub repository.
```

## Common Issues

1. `BSDS dataset not found`
Place dataset at `data/raw/BSDS500` or pass correct `--data-root`.

2. `Trained model not found`
Run `train.py` first, or pass `--model <path-to-pkl>` in `compare_methods.py`.

3. `skimage_canny` missing
Install `scikit-image`; benchmark continues without that method if unavailable.

## Reproducibility Notes

- Training uses randomized train/validation split via `numpy.random.permutation`.
- For strict reproducibility, set fixed random seeds in code before running experiments.

## License and Data

Code in this repository is project code.
BSDS500 dataset licensing and redistribution terms are governed by the original dataset providers.

