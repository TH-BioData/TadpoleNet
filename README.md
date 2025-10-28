# TadpoleNet: Xenopus Embryo Segmentation and Area Calculation

This repository contains a Python-based semantic segmentation project to identify *Xenopus* embryos (tadpoles) in images and calculate their surface area.

The core of the project is a **UNet** model with an **EfficientNet-B0** encoder (pre-trained on ImageNet), implemented using the `segmentation-models-pytorch` library.

The repository includes a **Demo Notebook** (`tadpolenet_demo.ipynb`) that loads the pre-trained model, segments new images, and calculates the area (in cm²) of the detected embryos.

---

## Key Features

* **Architecture:** UNet with a `timm-efficientnet-b0` encoder.
* **Framework:** Python and `segmentation-models-pytorch`.
* **Training:**
    * Combined Loss Function (BCE + Dice Loss) for robust binary segmentation.
    * Intensive data augmentation using `albumentations`.
    * Mixed-Precision (AMP) training for speed and VRAM efficiency.
    * Saves the best model based on validation loss.
* **Inference:**
    * Handles images of any size.
    * Uses aspect-aware resizing (pad-to-square) for prediction.
    * Post-processes the mask back to the original image dimensions.
* **Area Calculation:**
    * Converts the predicted mask's pixel count (`cv2.countNonZero`) to a real-world area (e.g., cm²) using a user-defined calibration scale.

---

## Requirements

You can install the main dependencies using `pip`:

```bash
pip install -U segmentation-models-pytorch==0.3.3 albumentations==1.4.18 opencv-python==4.10.0.84 timm==0.9.2 torchsummary==1.5.1 torch
