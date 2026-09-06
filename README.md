# Semi-Supervised Wafer Defect Classification

## Purpose

This project extends the previous supervised wafer defect classification study
to a **Semi-Supervised Learning setting** using both labeled and unlabeled WM-811K data.

The goal is to explore whether large-scale unlabeled wafer maps can be effectively utilized
for 9-class wafer defect classification.

> This is a personal follow-up experiment and is not included in the IJPR manuscript.

---

## Workflow

![workflow](figures/ssl_workflow.png)

---

## Key Contributions

- **FixMatch-style Semi-Supervised Learning**
  using weak/strong augmentation and confidence-based pseudo labeling.

- **Wafer-aware Strong Augmentation**
  with Cutout and Masked Bernoulli Noise to perturb wafer regions while preserving the background.

- **Confidence-based Pseudo Label Filtering**
  using a threshold of `0.90` to reduce the influence of unreliable pseudo labels.

- **Adaptive Loss Weighting**
  to dynamically balance supervised and unsupervised objectives during training.

- **Large-scale Unlabeled Data Utilization**
  using approximately `638K` unlabeled wafer maps in addition to labeled WM-811K data.

---

## Key Results

- **Validation Accuracy**: `0.9787`
- **Validation Macro-F1**: `0.8974`
- **Test Accuracy**: `0.9789`
- **Test Macro-F1**: `0.8984`

---

## Limitation & Future Work

Pseudo labels were heavily concentrated on the dominant `None` class,
indicating that the severe class imbalance in WM-811K also affects pseudo-label generation.

Future work includes:

- Applying **AdaSH-style Adaptive Thresholding** to reduce dominant-class pseudo-label bias.
- Evaluating class-wise pseudo-label distributions and acceptance rates.
- Comparing **Supervised-only vs. Semi-Supervised Learning** under multiple labeled-data ratios.
- Quantitatively evaluating the actual contribution of unlabeled wafer maps.

WBM_Classification/
├── supervised/
│   └── ...
├── semi_supervised/
│   ├── main.py
│   ├── trainer.py
│   └── ...
├── figures/
│   ├── workflow.png
│   └── ssl_workflow.png
└── README.md