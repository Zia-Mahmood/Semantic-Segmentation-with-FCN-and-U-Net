# Semantic Segmentation with FCN and U-Net

## Overview

This repository contains an academic, team-based implementation of **semantic segmentation models** using **Fully Convolutional Networks (FCN)** and **U-Net architectures**.

The project focuses on:

* understanding architectural differences between FCN variants and U-Net models,
* analyzing segmentation quality using **Mean Intersection over Union (mIoU)**,
* visualizing predictions and failure cases on urban scene data.

All implementations are done in **Python using PyTorch**.

---

## Tasks Covered

The repository is organized into two main parts corresponding to the assignment questions.

### Q1: Fully Convolutional Networks (FCN)

Implemented and evaluated the following FCN variants:

* **FCN-32s**
* **FCN-16s**
* **FCN-8s**

For each variant:

* Models were trained with **frozen backbones** and **fine-tuned backbones**
* Training loss and mIoU curves were plotted
* Sample predictions were visualized and compared
* Performance differences across variants were analyzed

The backbone used was a **VGG-based architecture pretrained on ImageNet**.

---

### Q2: Semantic Segmentation using U-Net

Implemented and evaluated U-Net-based architectures, including:

* **Vanilla U-Net**
* **U-Net without skip connections**
* **Residual U-Net**
* **Gated Attention U-Net**

Each model was:

* Trained for multiple epochs until convergence
* Evaluated using **mIoU on the test set**
* Compared visually against ground-truth segmentation masks

The impact of skip connections, residual blocks, and attention gates on segmentation quality was studied.

---

## Dataset

* Images and segmentation masks were provided as part of the course assignment. [Dataset](https://drive.google.com/file/d/10v-yWWb6NdEEOntcyG0hfVTAzlg9UkcX/view)
* Urban scene images contain **13 semantic classes**, including road, sidewalk, vegetation, vehicles, pedestrians, and traffic signs.
* Dataset exploration includes:

  * Class-wise pixel distribution
  * Per-class mask visualization
  * Sample image–mask pairs

---

## Evaluation Metrics

* **Mean Intersection over Union (mIoU)** was used as the primary evaluation metric.
* Training and validation curves were generated for loss and mIoU.
* Visual inspection of predictions was used to analyze failure modes such as:

  * occlusions,
  * class imbalance,
  * low-contrast regions.

---

## Repository Structure

```
.
├── Q1
│   ├── Q1.ipynb
│   └── Outputs
│       ├── fcn_results
│       │   ├── FCN32s_frozen / finetuned
│       │   ├── FCN16s_frozen / finetuned
│       │   ├── FCN8s_frozen / finetuned
│       │   ├── plots
│       │   └── predictions
│       └── visualization_output
│           ├── class_visualizations
│           ├── class_distribution
│           ├── overlays
│           └── side_by_side_comparisons
│
├── Q2
│   ├── Q2.ipynb
│   └── Outputs
│       ├── training curves
│       ├── metric tables
│       └── qualitative prediction comparisons
│
└── README.md
```

---

## How to Run

1. Clone the repository:

```bash
git clone <repo-url>
cd Semantic-Segmentation-FCN-UNet
```

2. Install required dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib opencv-python torchmetrics
```

3. Run the notebooks:

* `Q1/Q1.ipynb` for FCN experiments
* `Q2/Q2.ipynb` for U-Net experiments

All outputs (plots, predictions, comparisons) are saved under the corresponding `Outputs/` directories.

---

## Notes

* This repository reflects an **academic assignment** and focuses on **understanding and analysis**, not production deployment.
* The work was performed in a **team setting**, with contributions spanning model implementation, experimentation, visualization, and evaluation.
* Any use of external resources or model references follows the course guidelines.

---

## References

* Fully Convolutional Networks for Semantic Segmentation
  [https://arxiv.org/abs/1411.4038](https://arxiv.org/abs/1411.4038)
* U-Net: Convolutional Networks for Biomedical Image Segmentation
  [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
* Attention U-Net
  [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999)


