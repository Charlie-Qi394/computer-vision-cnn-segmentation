# Computer Vision Classification and Segmentation

## Overview
This repository is a sanitized computer-vision portfolio project covering classical image processing, CNN image classification and semantic segmentation model design.

It is reconstructed from university learning work and keeps the focus on reusable implementation, derived experiment evidence and clear technical explanation. It excludes assignment briefs, submitted reports, student identifiers, datasets, checkpoints and large generated binaries.

## What Is Included
- Classical Harris corner detection and Canny edge detection implementations.
- Lightweight metric utilities for classification and segmentation evaluation.
- Optional TensorFlow/Keras model builders for CIFAR-style classification and segmentation experiments.
- Derived CSV/JSON experiment artifacts from completed runs.
- Generated SVG plots from the included metrics.
- Tests for the reusable Python utilities.

## Methods
Classical CV:
- Harris response calculation from image gradients and Gaussian-smoothed structure tensors.
- Non-maximum suppression for selecting corner candidates.
- Canny-style edge detection with Gaussian smoothing, Sobel gradients, interpolated non-maximum suppression and hysteresis.

CNN classification:
- CIFAR-100-style baseline and architecture variants.
- Experiments compared data augmentation, batch normalization, filter changes, depth changes, residual blocks and inception-style blocks.
- Final tuned experiment used stronger augmentation and a bottleneck-style model variant.

Semantic segmentation:
- FCN-style baseline.
- Attention + ASPP U-Net style model.
- FPN + ASPP + attention style model.
- MeanIoU-style evaluation and visual inspection workflow.

## Key Results
- Baseline CIFAR-100 model: final test accuracy `35.39%`.
- Best controlled architecture comparison: `Res_v1_basic`, final test accuracy `45.23%`.
- Final tuned CIFAR-100 experiment: final test accuracy `74.18%`, final test loss `1.8027`.

Segmentation work covered FCN, U-Net and FPN-style architectures. Raw datasets and model weights are not redistributed.

## Repository Structure
```text
computer-vision-cnn-segmentation/
  README.md
  academic-integrity.md
  requirements.txt
  pyproject.toml
  data/
    README.md
  src/
    cv_portfolio/
      classical.py
      cnn_models.py
      metrics.py
      plot_artifacts.py
      segmentation.py
  artifacts/
    metrics/
    plots/
  tests/
```

## How To Run
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
python -m cv_portfolio.plot_artifacts
```

TensorFlow/Keras is optional and only needed for building the deep-learning model definitions:
```bash
pip install -e ".[deep-learning]"
```

## Artifact Notes
The files in `artifacts/metrics/` are derived experiment summaries. Checkpoints, datasets and raw assignment submissions are intentionally excluded.

## Lessons Learned
- Classical image-processing methods are interpretable but sensitive to thresholds and preprocessing choices.
- CNN performance depends heavily on augmentation, architecture, regularization and training control.
- Segmentation requires both numeric metrics and visual validation because pixel-level errors can be spatially concentrated.

## Future Improvements
- Add small public-domain image demos for the classical CV methods.
- Add reproducible training scripts for the classification variants.
- Add lightweight segmentation inference examples using user-provided images.
- Add experiment tracking and configuration files for repeatable runs.
