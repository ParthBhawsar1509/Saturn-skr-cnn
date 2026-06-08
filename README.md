# Predicting Saturn's Kilometric Radio Emission with a 1D CNN

Non-linear regression on calibrated Cassini RPWS HFR spectral data. A 1D
convolutional neural network predicts spectral power density at **300 kHz**
(within the Saturn Kilometric Radiation band) from the remaining 95 frequency
channels at each time step.

## Overview

The Cassini Radio and Plasma Wave Science (RPWS) High Frequency Receiver
measured electric-field spectral densities from 3.6 kHz to 16.1 MHz across 96
logarithmically spaced channels. This project treats channel 48 (300 kHz) as a
regression target and learns to reconstruct it from the surrounding 95 channels,
exploiting the local frequency correlations of plasma radio emissions.

A 1D CNN is used rather than an MLP or RNN because adjacent frequency channels
are physically correlated, convolutional kernels act as learned spectral filters,
and the model stays compact (47,969 parameters) while generalising well from a
single day of data.

## Results

Evaluated on 1,600 temporally held-out records (physical units,
log₁₀(V² m⁻² Hz⁻¹)):

| Metric | Value |
|---|---|
| R² | 0.982 |
| RMSE | 0.091 |
| MAE | 0.068 |
| Residual mean | −0.003 |
| Residual std | 0.091 |

Residuals are zero-centred and approximately Gaussian, with training and
validation losses converging without overfitting.

## Architecture

Input (95 × 1) →
Conv1D(32, k=7) → BN → ReLU → MaxPool(2) →
Conv1D(64, k=5) → BN → ReLU → MaxPool(2) →
Conv1D(128, k=3) → BN → ReLU → GlobalAvgPool →
Dense(128) → ReLU → Dropout(0.3) →
Dense(64) → ReLU → Dropout(0.2) →
Dense(1, linear)

Loss: MSE. Optimiser: Adam (lr = 1e-3). Callbacks: EarlyStopping
(patience 20) and ReduceLROnPlateau (factor 0.5, patience 10).

## Data

Cassini RPWS Calibrated Low-Rate Full-Resolution archive
(dataset ID: `CO-V/E/J/S/SS-RPWS-4-SUMM-KEY60S-V1.0`), from the NASA
Planetary Data System Plasma Physics Interactions node. This repository uses one
continuous day of Saturn-orbit data (2005 DOY 060), ~8,000 records after filtering.

> Note: the raw data file is not committed to this repo (see `.gitignore`).
> Download it from the PDS and place it where the notebook expects it.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open `Deep_learning_final_code.ipynb` and run the cells top to bottom.

## Repository contents

```
.
├── Deep_learning_final_code.ipynb   # full pipeline: preprocessing, model, evaluation
├── Deep_Learning_final.pdf          # project report
├── requirements.txt
├── .gitignore
└── README.md
```

## Pre-processing summary

- Remove records containing NaN/Inf (< 0.3% of records).
- Temporal 80/20 train/test split (no shuffling, to avoid temporal leakage).
- StandardScaler on the 95 inputs and a separate scaler on the target, both fit
  on the training set only.
- All metrics and plots reported in physical units after inverse-transforming.

## References

Gurnett et al. (2004); Zarka et al. (2004); Fischer et al. (2006);
Kurth et al. (2014); Ioffe & Szegedy (2015); Kingma & Ba (2015);
Srivastava et al. (2014); Glorot et al. (2011); Gal & Ghahramani (2016).
