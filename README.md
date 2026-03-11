## Offroad Semantic Segmentation – Training Report

This document explains the **segmentation training pipeline**, **model architecture**, **training setup**, and **evaluation results** used in `train_segmentation.py`, together with the recorded stats in `train_stats3/`.

---

### 1. Problem & Dataset

- **Task**: Multi-class semantic segmentation of offroad scenes (trees, bushes, grass, rocks, sky, etc.).
- **Source**: `Offroad_Segmentation_Training_Dataset` with `train/` and `val/` splits.
- **Images**:
  - RGB color images are stored in `Color_Images/`.
  - Corresponding segmentation masks are stored in `Segmentation/` with the **same filenames** as the images.
- **Mask encoding & remapping**:
  - Raw mask pixel values are mapped to **10 semantic classes** using `value_map` in `train_segmentation.py`:
    - `0` → background  
    - `100` → Trees  
    - `200` → Lush Bushes  
    - `300` → Dry Grass  
    - `500` → Dry Bushes  
    - `550` → Ground Clutter  
    - `700` → Logs  
    - `800` → Rocks  
    - `7100` → Landscape  
    - `10000` → Sky  
  - After conversion, masks are stored as integer class IDs in \[0, 9\].

---

### 2. Data Pipeline

- **Custom Dataset**: `MaskDataset`
  - Reads paired `(image, mask)` from `Color_Images/` and `Segmentation/`.
  - Applies **separate transforms** to images and masks.
- **Image preprocessing**:
  - Resize to \((h, w)\) where:
    - `w = int(((960 / 2) // 14) * 14)`  
    - `h = int(((540 / 2) // 14) * 14)`  
    This makes height/width divisible by 14, which matches the DINOv2 patch size.
  - Convert to tensor with `transforms.ToTensor()`.
  - Normalize with ImageNet statistics:
    - mean = \([0.485, 0.456, 0.406]\)
    - std  = \([0.229, 0.224, 0.225]\)
- **Mask preprocessing**:
  - Apply `convert_mask` to turn raw values into class IDs.
  - Resize with `transforms.Resize((h, w))` followed by `transforms.ToTensor()`.
  - Multiply by 255 so that class IDs become integers again after scaling.
  - In the training loop, masks are squeezed and cast to `long` before loss computation.
- **Data loaders**:
  - `batch_size = 16`
  - `train_loader`: data from `train/`.
  - `val_loader`: data from `val/`.
  - Shuffling is **enabled** for training and **disabled** for validation.

---

### 3. Model Architecture

The model is a **two-stage system**:

- **Backbone**: Pretrained **DINOv2 ViT** from Facebook Research:
  - Backbone size: `"base"` (config string `vitb14_reg`).
  - Loaded via:
    - `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")`
  - We use the **patch token features** from:
    - `backbone_model.forward_features(imgs)["x_norm_patchtokens"]`
  - Output shape: `(B, N, C)` where:
    - \(B\): batch size  
    - \(N = (H/14) * (W/14)\): number of patches  
    - \(C\): embedding dimension (logged as `n_embedding`).

- **Segmentation head**: `SegmentationHeadConvNeXt`
  - Purpose: Convert DINOv2 patch tokens into dense per-pixel class logits.
  - Steps:
    1. **Reshape** token sequence back to a 2D feature map:
       - Reshape to `(B, H', W', C)` and permute to `(B, C, H', W')`
       - `H' = tokenH = h / 14`, `W' = tokenW = w / 14`
    2. **Stem**:
       - `Conv2d(in_channels=C, out_channels=128, kernel_size=7, padding=3)`
       - `GELU` activation.
    3. **ConvNeXt-style block**:
       - Depthwise `Conv2d(128, 128, kernel_size=7, padding=3, groups=128)`
       - `GELU`
       - Pointwise `Conv2d(128, 128, kernel_size=1)`
       - `GELU`
    4. **Classifier**:
       - `Conv2d(128, n_classes, kernel_size=1)` → per-patch class logits.
    5. **Upsampling**:
       - Bilinear interpolation with `F.interpolate` to original image resolution `(H, W)`.

---

### 4. Loss Functions & Training Tricks

- **Combined loss**:
  - **Cross-Entropy Loss** (`nn.CrossEntropyLoss`):
    - Encourages correct per-pixel classification.
  - **Dice Loss** (`DiceLoss` custom class):
    - Uses softmax probabilities and one-hot encoded targets.
    - Computes Dice coefficient across spatial dimensions and classes, then returns `1 - mean_dice`.
  - **Final loss per batch**:
    - `loss = loss_ce + loss_dice`
    - This combination balances region overlap quality (Dice) with pixel-wise correctness (CE).

- **Dice loss details**:
  - `probs = softmax(logits, dim=1)`
  - Targets are one-hot encoded with `F.one_hot`.
  - Intersection and cardinality are summed over batch and spatial dims.
  - Smooth term `smooth = 1e-6` avoids division by zero.

- **Optimizer & regularization**:
  - **AdamW** optimizer:
    - Parameters: both the segmentation head and any trainable backbone parameters:
      - `params = list(classifier.parameters()) + list(filter(lambda p: p.requires_grad, backbone_model.parameters()))`
    - Learning rate: `lr = 1e-4`
    - Weight decay: `1e-4`
  - This is a strong choice for transformer-style backbones and stabilizes training with weight decay.

- **Other training design choices / tricks**:
  - **Patch-aligned resizing**:
    - Image dimensions are forced to be multiples of 14 to align with DINOv2 patch size, avoiding awkward reshapes.
  - **Non-interactive plotting**:
    - `matplotlib` is set to `Agg` backend so plots can be generated in non-GUI environments.
  - **Full-dataset metric evaluation per epoch**:
    - After each epoch, metrics are recomputed on the full train and validation sets using `evaluate_metrics` (this is more expensive but gives reliable curves).

---

### 5. Training Configuration

- **Device**:
  - Uses GPU if available: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
- **Hyperparameters**:
  - `batch_size = 16`
  - `n_epochs = 10`
  - `lr = 1e-4`
  - `weight_decay = 1e-4`
- **Loop structure**:
  - For each epoch:
    - **Training phase**:
      - Backbone features computed for each batch.
      - Segmentation head outputs logits → upsampled to image size.
      - Labels squeezed to `(B, H, W)` and cast to `long`.
      - Loss = CE + Dice, then backward, optimizer step, zero grad.
    - **Validation phase**:
      - Model set to `eval()`; gradients disabled with `torch.no_grad()`.
      - Same forward path as training but **no backprop**.
    - **Metric evaluation** (on full train and val):
      - Computed via `evaluate_metrics`:
        - Mean IoU across classes.
        - Mean Dice across classes.
        - Pixel accuracy.
    - Metrics and losses are appended to `history`.
- **Output artifacts**:
  - `segmentation_head.pth`: saved segmentation head weights in the scripts directory.
  - Training curves and metrics stored in `train_stats/` (for the run corresponding to this script).

---

### 6. Evaluation Metrics

Metrics are implemented in `train_segmentation.py` and reused for both train/val:

- **IoU (Intersection over Union)** – `compute_iou`:
  - Predictions: `argmax` over class dimension.
  - For each class, IoU = intersection / union over all pixels.
  - Per-class IoUs are averaged (ignoring classes with zero union).

- **Dice coefficient** – `compute_dice`:
  - Similar to IoU calculation, but uses:
    - \(\text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}\)
  - Computed per class and averaged.

- **Pixel accuracy** – `compute_pixel_accuracy`:
  - Fraction of pixels where `pred == target`.

- **Aggregate evaluation** – `evaluate_metrics`:
  - Iterates over a given data loader.
  - For each batch:
    - Forward pass through backbone and segmentation head.
    - Upsample to image resolution.
    - Compute IoU, Dice, and pixel accuracy for that batch.
  - Returns the **mean** of each metric across all batches.

---

### 7. Training Curves & Saved Stats (`train_stats3/`)

For the referenced run, you provided `train_stats3/evaluation_metrics.txt`, which is generated by `save_history_to_file`. Key contents:

- **Final metrics (Epoch 10)**:
  - **Train Loss**: 0.7349  
  - **Val Loss**: 0.7485  
  - **Train IoU**: 0.5762  
  - **Val IoU**: 0.5624  
  - **Train Dice**: 0.7074  
  - **Val Dice**: 0.6950  
  - **Train Pixel Accuracy**: 0.8445  
  - **Val Pixel Accuracy**: 0.8428  

- **Best validation performance**:
  - **Best Val IoU**: 0.5629 (Epoch 9)
  - **Best Val Dice**: 0.6971 (Epoch 9)
  - **Best Val Accuracy**: 0.8428 (Epoch 10)
  - **Lowest Val Loss**: 0.7485 (Epoch 10)

- **Per-epoch trends (high level)**:
  - **Train & Val Loss**:
    - Start around ~1.33 (train) / ~1.00 (val) at Epoch 1.
    - Decrease monotonically to ~0.73–0.75 by Epoch 10.
    - Indicates **stable convergence** without overfitting spikes.
  - **IoU & Dice**:
    - IoU improves from ~0.45 (val) to ~0.56 (val).
    - Dice improves from ~0.57 (val) to ~0.70 (val).
    - Both metrics rise consistently, showing better region overlap and segmentation quality over epochs.
  - **Pixel Accuracy**:
    - Starts around ~0.81 on val and grows to ~0.84.
    - High accuracy but lower IoU/Dice reflects that background and majority classes are easier, while boundary and minority classes remain harder.

- **Overall interpretation**:
  - With **10 epochs** and the combined **CE + Dice** loss, the model:
    - Achieves **~0.56 mIoU**, **~0.70 Dice**, and **~84% pixel accuracy** on validation.
    - Shows no obvious severe overfitting: train/val curves track closely.
  - Additional epochs, stronger augmentations, or fine-tuning of the DINOv2 backbone could further improve metrics.

---

### 8. How to Run Training

From the `Scripts/` directory:

```bash
python train_segmentation.py
```

This will:

- Load the DINOv2 backbone.
- Build the `MaskDataset` for train and validation splits.
- Train the ConvNeXt-style segmentation head for **10 epochs**.
- Save:
  - Model weights: `segmentation_head.pth`
  - Training plots (`training_curves.png`, `iou_curves.png`, `dice_curves.png`, `all_metrics_curves.png`) in `train_stats/`
  - Text report of metrics in `train_stats/evaluation_metrics.txt`

If you want to reproduce or compare against the stats in `train_stats3/`, ensure that:

- Dataset directory structure matches the one expected in `main()`.
- Hyperparameters (batch size, learning rate, epochs) are left unchanged.

---

### 9. Summary

- **Model**: DINOv2 ViT backbone (`vitb14_reg`) + ConvNeXt-style convolutional decoder head.
- **Loss**: Sum of Cross-Entropy and Dice loss for better overlap quality and class balance.
- **Epochs**: 10, with per-epoch full-dataset metric evaluation.
- **Results** (validation):
  - ~0.56 mean IoU  
  - ~0.70 mean Dice  
  - ~0.84 pixel accuracy  
- **Tricks**:
  - Patch-aligned resizing to 14-pixel multiples.
  - Combined CE + Dice loss.
  - AdamW optimizer with weight decay.
  - Full-history plotting and detailed metrics logging for analysis and debugging.

