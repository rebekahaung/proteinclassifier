# Protein Function Classifier

## Overview

This project builds an end-to-end machine learning pipeline that predicts a protein's functional class directly from its amino acid sequence — no structural data required. It replicates the core feature engineering workflow used in real bioinformatics research, using Biopython's ProtParam module to derive physicochemical descriptors, then benchmarks Random Forest against SVM across standard classification metrics.

**Functional classes predicted:**

| Class | Biological Role | UniProt Keyword |
|---|---|---|
| Enzyme | Catalysis (hydrolases) | KW-0378 |
| Transporter | Membrane transport | KW-0813 |
| Transcription Factor | Gene regulation | KW-0804 |
| Structural Protein | Extracellular matrix | KW-0261 |

## Results

| Model | Precision | Recall | F1 | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|
| Random Forest | 0.7341 | 0.7325 | 0.7323 | 0.9160 | 0.7467 ± 0.0122 |
| SVM (RBF, C=10) | 0.7183 | 0.7150 | 0.7153 | 0.9177 | 0.7308 ± 0.0273 |

## Per-Class Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Enzyme | 0.60 | 0.65 | 0.62 |
| Structural Protein | 0.94 | 0.94 | 0.94 |
| Transcription Factor | 0.72 | 0.74 | 0.73 |
| Transporter | 0.68 | 0.60 | 0.64 |

## Features Engineered (45 total)

Features are computed with `src/features.py`, which implements the same algorithms as Biopython's `Bio.SeqUtils.ProtParam`.

| Category | Features | Count |
|---|---|---|
| Amino acid composition | Fractional frequency of each standard AA | 20 |
| Physicochemical | Molecular weight, GRAVY, aromaticity, instability index, isoelectric point, molar extinction coefficient, net charge at pH 7 | 7 |
| Secondary structure | Chou-Fasman estimated helix / turn / sheet fractions | 3 |
| Sequence properties | Length | 1 |
| Dipeptide composition | Frequencies of 14 selected dipeptides (KK, EE, FF, PP, …) | 14 |


## Visualisations

The output dashboard covers:

- **Confusion matrices** — normalised, for both models
- **ROC curves** — per-class AUC breakdown
- **t-SNE projection** — 2D class separability of the feature space
- **Feature importances** — top 15 by Random Forest Gini importance
- **Cross-validation boxplots** — 5-fold F1 distribution per model
- **Metric summary cards** — precision, recall, ROC-AUC at a glance

