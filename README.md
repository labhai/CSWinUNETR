<h1 align="center">CSWinUNETR: Segmentation of Thin Anatomical Structures in Medical Images</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=_mD5n_UAAAAJ"><strong>Junho Moon</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://scholar.google.com/citations?user=O-oZnIwAAAAJ"><strong>Haejun Chung</strong></a>
  <span title="Corresponding author">&#9993;</span>
  &nbsp;&middot;&nbsp;
  <a href="https://scholar.google.com/citations?user=1rBh9xkAAAAJ"><strong>Ikbeom Jang</strong></a>
  <span title="Corresponding author">&#9993;</span>
  <br>
  <sub>&#9993; Corresponding authors</sub>
</p>

<p align="center">
  <img src="./assets/main_architecture.png" width="900">
</p>

Official repository for the paper:

> **[CSWinUNETR: Segmentation of Thin Anatomical Structures in Medical Images](https://arxiv.org/abs/2606.19824)**<br>
> MICCAI 2026<br>
> Junho Moon, Haejun Chung, Ikbeom Jang<br>
> ([arXiv ver](https://arxiv.org/abs/2606.19824))

## Overview

*Accurate segmentation of thin, tortuous anatomical structures, such as retinal vessels, cerebral vasculature, and facial wrinkles, remains challenging due to low contrast, frequent discontinuities, and severe class imbalance. Although recent convolutional and Transformer-based models have improved performance, they often yield fragmented predictions and fail to recover fine branches. We propose CSWinUNETR, a task-oriented 2D/3D backbone for thin-structure segmentation. It employs cross-shaped stripe self-attention to model long-range principal-axis context and incorporates cyclic shifts to enhance information exchange across stripes. To better preserve fine-grained details, we further introduce a detail-enhanced multi-scale self-attention module that aggregates contextual features from multi-resolution representations. In addition, we propose sparse-control dynamic snake convolution, which reconstructs reliable dense curvilinear kernels from sparsely predicted control points to better follow tortuous geometry. Extensive experiments on four benchmarks across ophthalmology, neurovascular imaging, and dermatology demonstrate that CSWinUNETR consistently outperforms state-of-the-art methods without task-specific post-processing or topology-aware losses.*

## Code

We provide the PyTorch implementation of CSWinUNETR for binary and multiclass
segmentation in both 2D and 3D. As a usage example, we provide training and
inference scripts for 2D binary segmentation on
[FFHQ-Wrinkle](https://github.com/labhai/ffhq-wrinkle-dataset).

### Installation

We use Python 3.10/3.11 and PyTorch 2.5.1. Please install the appropriate
[PyTorch build](https://pytorch.org/get-started/previous-versions/) for your CUDA
version, then install the dependencies:

```bash
git clone https://github.com/labhai/CSWinUNETR.git
cd CSWinUNETR
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
```

For model inference only, use `python -m pip install -e .`.

### Repository Structure

```text
CSWinUNETR/
├── assets/                       # Architecture figure
├── cswinunetr/                   # Source code
│   ├── models/                   # CSWinUNETR, attention, and SDSConv
│   ├── datasets/                 # FFHQ-Wrinkle loader
│   │   └── splits/ffhq/           # Training, validation, and test IDs
│   ├── engine/                   # Training and inference routines
│   ├── utils/                    # Loss functions and evaluation metrics
│   ├── train.py                  # Training entry point
│   └── inference.py              # Inference and evaluation entry point
└── pyproject.toml                # Dependencies and package configuration
```

### Data Preparation

Please follow the instructions in our
[FFHQ-Wrinkle dataset repository](https://github.com/labhai/ffhq-wrinkle-dataset)
to download the wrinkle annotations and texture maps, obtain the FFHQ images,
and prepare the masked face images. The data directory should be organized as follows:

```text
data/
├── masked_face_images/
│   └── 00001.png
├── manual_wrinkle_masks/
│   └── 00001.png
└── weak_wrinkle_masks/
    └── 00000/
        └── 00001.png
```

We use masked RGB images and their grayscale texture maps as four-channel inputs
at 1024 × 1024 resolution. Manual wrinkle masks provide binary supervision.
Images and masks are matched by their filenames; flat texture directories are also supported.
Please refer to the dataset repository for its [license](https://github.com/labhai/ffhq-wrinkle-dataset#license).

The [split files](cswinunetr/datasets/splits/ffhq) contain 800 training, 100 validation,
and 100 test images. The test split follows the dataset repository's
[`test_file_lists.txt`](https://github.com/labhai/ffhq-wrinkle-dataset/blob/main/test_file_lists.txt).

### Training

Run the following command from the repository root:

```bash
python cswinunetr/train.py \
  --data-root /path/to/ffhq-wrinkle-dataset/data \
  --output-dir runs/ffhq \
  --device cuda
```

The checkpoint with the best validation Dice is saved to `runs/ffhq/best.pt`.
To resume an interrupted run, add `--resume` with the same output directory
and training arguments.

Adjust `--batch-size` to fit the available GPU memory, or enable
`--use-checkpoint` to reduce activation memory. Use `--split-dir` to specify
another directory containing `train.txt`, `val.txt` and `test.txt`.
Run `python cswinunetr/train.py --help` for all training arguments.

### Inference

To predict and evaluate the test images:

```bash
python cswinunetr/inference.py \
  --checkpoint runs/ffhq/best.pt \
  --image-dir /path/to/ffhq-wrinkle-dataset/data/masked_face_images \
  --texture-dir /path/to/ffhq-wrinkle-dataset/data/weak_wrinkle_masks \
  --mask-dir /path/to/ffhq-wrinkle-dataset/data/manual_wrinkle_masks \
  --ids-file cswinunetr/datasets/splits/ffhq/test.txt \
  --output-dir outputs/ffhq-test \
  --device cuda
```

For unlabeled images, omit `--mask-dir`. Omit `--ids-file` to process all images
in the input directory. Texture maps are required for both training and inference.

Predictions are saved as PNG masks with class IDs `0` (background) and `1`
(wrinkle). When ground-truth masks are provided, `predictions.json` records
per-image foreground Dice and its mean across the evaluated images.

Install `python -m pip install -e '.[eval]'` and add `--all-metrics` to also
report clDice, Betti error and HD95.

### Using CSWinUNETR

```python
import torch
from cswinunetr import CSWinUNETR

# 2D binary segmentation (background and foreground)
model = CSWinUNETR(in_channels=3, out_channels=2, spatial_dims=2).eval()
with torch.inference_mode():
    logits = model(torch.randn(1, 3, 64, 64))
    prediction = logits.argmax(dim=1)

# 3D multiclass segmentation
model_3d = CSWinUNETR(in_channels=1, out_channels=14, spatial_dims=3)
```

Set `spatial_dims` to `2` or `3`, and `out_channels` to the number of classes
including background (`2` for binary segmentation). Use `argmax(dim=1)` on the
output logits to obtain class labels in either case.

Inputs follow `NCHW` for images and `NCDHW` for volumes. Spatial dimensions
must be multiples of 32 and at least 64. The model returns logits at the input
resolution. For large volumes, use sliding-window inference with appropriately
preprocessed patches.

## BibTex (to cite our paper)

If this work is useful for your research, please cite:

```bibtex
@article{moon2026cswinunetr,
  title={CSWinUNETR: Segmentation of Thin Anatomical Structures in Medical Images},
  author={Moon, Junho and Chung, Haejun and Jang, Ikbeom},
  journal={arXiv preprint arXiv:2606.19824},
  year={2026}
}
```

## Contact

For questions about the paper or repository, please open an issue or contact [**Junho Moon**](mailto:jhmoon6807@gmail.com).
