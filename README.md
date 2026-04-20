# Self-Tuning Edge Detection (BSDS-only)

This project keeps:
- Training (`train.py`) on **BSDS500 only**
- Method comparison (`compare_methods.py`)
- Core library code in `src/`

## Structure

```text
Project/
├── src/
├── data/raw/BSDS500/
├── train.py
├── compare_methods.py
├── requirements.txt
└── README.md
```

## BSDS500 format

```text
data/raw/BSDS500/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── groundTruth/   (or ground_truth/)
    ├── train/
    ├── val/
    └── test/
```

## Install

```powershell
py -m pip install -r requirements.txt
```

## Train (BSDS only)

```powershell
python train.py --data-root data/raw/BSDS500 --max-train-samples 200
```

Optional:

```powershell
python train.py --data-root data/raw/BSDS500 --max-train-samples 200 --tune-hyperparams
```

## Comparison (BSDS only)

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test
```

Quick run:

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test --max-samples 50
```

## Logs

Comparison logs are saved in:

```text
results/logs/compare_<split>.log
```

Custom log:

```powershell
python compare_methods.py --data-root data/raw/BSDS500 --split test --log-file results/logs/my_run.log
```

