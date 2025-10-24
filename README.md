# Autoencoder-Based MRI Anomaly Detection

An inpainting autoencoder for unsupervised brain MRI anomaly detection, designed for FLAIR images with tumor segmentations. Trained on BrainMetShare-3. Compares pixel-perfect, latent, and fusion methods.

### Requirements
```bash
pip install torch torchvision nibabel pytorch-msssim matplotlib numpy
```

### Dataset Structure
```
base_dir/
 ├── subject_001/
 │   ├── flair.nii.gz
 │   └── seg.nii.gz
 ├── subject_002/
 │   ├── flair.nii.gz
 │   └── seg.nii.gz
 └── ...
```

---

## Usage

### Train from scratch
```bash
python autoencoder.py --base_dir /path/to/dataset --epochs 25
```

### Load pretrained model
```bash
python autoencoder.py --base_dir /path/to/dataset --load_model flair_inpainting_ae.pth
```

### Show scoring modes
```bash
python autoencoder.py --help_modes
```

---

## Output

After running, results are saved in:
```
base_dir/inpainting_outputs/
 ├── subject_001_score_pixel.nii.gz
 ├── subject_001_score_latent.nii.gz
 ├── subject_001_score_fusion.nii.gz
```

Each map can be viewed in any NIfTI viewer (e.g., FSLeyes, ITK-SNAP).

---

## Evaluation Metrics

| Mode   | Metric | Description |
|---------|---------|-------------|
| pixel  | Dice/Jaccard | Based on reconstruction error |
| latent | Dice/Jaccard | Based on encoder difference |
| fusion | Dice/Jaccard | Weighted combination of both |

---

## Command-line Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--base_dir` | Path to dataset root (required) | — |
| `--epochs` | Training epochs | 25 |
| `--load_model` | Path to pretrained `.pth` model | None |
| `--alpha` | Fusion weight for pixel/latent | 0.5 |
| `--batch_size` | Batch size | 16 |
| `--help_modes` | Show scoring mode info | False |
