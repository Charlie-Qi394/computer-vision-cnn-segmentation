# Computer Vision: Classical Algorithms, CNNs and Semantic Segmentation

## Overview
This repository presents three applied computer-vision projects: classical feature detection, deep-learning image classification and deep-learning semantic segmentation.

It is reconstructed from university learning work and keeps the focus on reusable implementation, derived experiment evidence and clear technical explanation. It excludes assignment briefs, submitted reports, student identifiers, datasets, checkpoints and large generated binaries.

> **Scope:** This is hands-on model implementation, training and evaluation work. It does not claim novel research architectures, production deployment, or model pruning.

## Three Projects at a Glance

| Project | Practical capability | Evidence |
| --- | --- | --- |
| **1. Classical feature detection** | Implemented Harris corner detection and a Canny-style edge-detection pipeline from image gradients, non-maximum suppression and hysteresis. | Reusable algorithms and unit tests in `src/cv_portfolio/classical.py`. |
| **2. CNN image classification** | Built a CIFAR-100 baseline and compared 12 CNN variants spanning augmentation, batch normalisation, regularisation, filter width, depth, residual blocks and Inception-style modules. | Best controlled architecture: `45.23%` test accuracy; final tuned experiment: `74.18%`. |
| **3. Semantic segmentation** | Designed and evaluated FCN, attention + ASPP U-Net, and FPN + ASPP + attention segmentation architectures using staged training and mean IoU evaluation. | Validation mIoU improved from `0.5332` to `0.6849` within a `<15M` parameter constraint. |

## What Is Included
- Classical Harris corner detection and Canny edge detection implementations.
- Lightweight metric utilities for classification and segmentation evaluation.
- Optional TensorFlow/Keras model builders for CIFAR-style classification and segmentation experiments.
- Derived CSV/JSON experiment artifacts from completed runs.
- Generated SVG plots from the included metrics.
- Tests for the reusable Python utilities.

## What I Built and Evaluated

### 1. Classical computer vision
- Harris response calculation from image gradients and Gaussian-smoothed structure tensors.
- Non-maximum suppression for selecting corner candidates.
- Canny-style edge detection with Gaussian smoothing, Sobel gradients, interpolated non-maximum suppression and hysteresis.

### 2. Deep-learning image classification
- CIFAR-100-style baseline and architecture variants.
- Experiments compared data augmentation, batch normalization, filter changes, depth changes, residual blocks and inception-style blocks.
- Final tuned experiment used stronger augmentation and a bottleneck-style model variant, with checkpoints, learning-rate reduction and early stopping.

### 3. Deep-learning semantic segmentation
- FCN-style baseline.
- Attention + ASPP U-Net style model.
- FPN + ASPP + attention style model.
- Separable convolutions, channel-width choices, attention-gated skip connections, combined loss design, staged fine-tuning, mean-IoU evaluation and visual inspection.
- The architecture uses computationally efficient design choices; it is **not** a pruning implementation.

## Results

### Classification
- Baseline CIFAR-100 model: final test accuracy `35.39%`.
- Best controlled architecture comparison: `Res_v1_basic`, final test accuracy `45.23%`.
- Final tuned CIFAR-100 experiment: final test accuracy `74.18%`, final test loss `1.8027`.

### Segmentation
- Baseline validation mIoU: `0.5332`.
- Final FPN + ASPP + attention model validation mIoU: `0.6849`.
- The final model remained within the coursework's `<15M` parameter budget.

Raw datasets and model weights are not redistributed.

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
The files in `artifacts/metrics/` are derived experiment summaries, including the classification comparison and segmentation summary. Checkpoints, datasets and raw assignment submissions are intentionally excluded.

## Lessons Learned
- Classical image-processing methods are interpretable but sensitive to thresholds and preprocessing choices.
- CNN performance depends heavily on augmentation, architecture, regularization and training control.
- Segmentation requires both numeric metrics and visual validation because pixel-level errors can be spatially concentrated.

## Future Improvements
- Add small public-domain image demos for the classical CV methods.
- Add reproducible training scripts for the classification variants.
- Add lightweight segmentation inference examples using user-provided images.
- Add experiment tracking and configuration files for repeatable runs.
