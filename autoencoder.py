#!/usr/bin/env python3
"""
autoencoder.py
---------------
PyTorch-based pipeline for anomaly detection on brain FLAIR MRI images
using inpainting autoencoders. Masks tumor regions, trains to reconstruct
the missing areas, and computes reconstruction-based anomaly maps. Originally trained on BrainMetShare-3 data.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from torchvision.transforms import InterpolationMode


#Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InpaintingFlairDataset(Dataset):
    """
    Custom PyTorch Dataset for FLAIR MRI inpainting.
    Loads (flair.nii.gz, seg.nii.gz) pairs from each subject folder and
    returns masked input, full target, segmentation, subject name, and slice index.
    """

    def __init__(self, base_dir, size=128, mask_prob=1.0):
        self.items = []
        self.size = size
        self.mask_prob = mask_prob

        for subj in os.listdir(base_dir):
            subj_path = os.path.join(base_dir, subj)
            flair_path = os.path.join(subj_path, "flair.nii.gz")
            seg_path = os.path.join(subj_path, "seg.nii.gz")
            if not (os.path.exists(flair_path) and os.path.exists(seg_path)):
                continue

            # Load and normalize image
            flair = nib.load(flair_path).get_fdata()
            flair = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

            seg = (nib.load(seg_path).get_fdata() > 0).astype(np.uint8)

            for i in range(flair.shape[2]):
                f = flair[:, :, i]
                s = seg[:, :, i]

                # Skip empty slices
                if np.max(f) < 0.05:
                    continue

                # Randomly mask tumor region
                if np.random.rand() < self.mask_prob and s.any():
                    masked = f * (1 - s)
                else:
                    masked = f

                # Resize all images to target dimensions
                f_t = TF.resize(torch.tensor(f, dtype=torch.float32).unsqueeze(0), [size, size]).squeeze(0)
                m_t = TF.resize(torch.tensor(masked, dtype=torch.float32).unsqueeze(0), [size, size]).squeeze(0)
                s_t = TF.resize(torch.tensor(s, dtype=torch.float32).unsqueeze(0), [size, size]).squeeze(0)
                self.items.append((m_t, f_t, s_t, subj, i))

        print(f"✅ Loaded {len(self.items)} slices total for inpainting training.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        m, f, s, subj, i = self.items[idx]
        return m.unsqueeze(0), f.unsqueeze(0), s.unsqueeze(0), subj, i


class SmallAE(nn.Module):
    """
    Simple convolutional autoencoder for inpainting.
    - Encoder: extracts compressed latent representation
    - Decoder: reconstructs the masked region
    """

    def __init__(self, dropout=0.05):
        super().__init__()

        # Encoder: Downsample spatial dimensions
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),    # 128 -> 64
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 16, 3, 1, 1),   # keep 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 8, 3, 1, 1),   # bottleneck
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout2d(dropout)
        )

        # Decoder: Upsample back to original size
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),  # 64 -> 128
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_latent=False):
        z = self.enc(x)
        out = self.dec(z)
        return (out, z) if return_latent else out


def train_inpainting(model, loader, epochs=20, lr=1e-3):
    """
    Train the autoencoder using SSIM loss between reconstruction and ground truth.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for masked, full, _, _, _ in loader:
            masked, full = masked.to(DEVICE), full.to(DEVICE)
            recon = model(masked)

            loss = 1 - ssim(recon, full, data_range=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * masked.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.5f}")

    return model


def compute_scores(model, masked, full, mode="pixel", alpha=0.5):
    """
    Compute anomaly score maps.
    - pixel: |reconstructed - original|
    - latent: difference between latent encodings
    - fusion: normalized blend of both
    """
    recon, z_rec = model(masked, return_latent=True)
    pixel_diff = torch.abs(recon - full)

    if mode == "pixel":
        return pixel_diff

    _, z_in = model(full, return_latent=True)
    latent_diff = torch.norm(z_in - z_rec, dim=1, keepdim=True)
    latent_up = F.interpolate(latent_diff, size=pixel_diff.shape[2:], mode='bilinear', align_corners=False)

    if mode == "latent":
        return latent_up

    # Fusion mode: combine both normalized maps
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    return alpha * norm(latent_up) + (1 - alpha) * norm(pixel_diff)


def temporal_smooth3d(vol):
    """Apply temporal smoothing across slices to reduce noise."""
    v = vol.copy()
    for k in range(1, vol.shape[2]-1):
        v[:, :, k] = 0.25*vol[:, :, k-1] + 0.5*vol[:, :, k] + 0.25*vol[:, :, k+1]
    return v


def run_inference(model, loader, base_dir, mode="pixel", alpha=0.5, smooth=False):
    """
    Run inference for all slices, compute score maps, and save NIfTI results.
    """
    out_dir = os.path.join(base_dir, "inpainting_outputs")
    os.makedirs(out_dir, exist_ok=True)
    subj_vols = {}
    model.eval()

    with torch.no_grad():
        for masked, full, seg, subj, idx in loader:
            masked, full = masked.to(DEVICE), full.to(DEVICE)
            score = compute_scores(model, masked, full, mode=mode, alpha=alpha).cpu().numpy()

            for b in range(len(subj)):
                sname, sidx = subj[b], int(idx[b])
                flair_path = os.path.join(base_dir, sname, "flair.nii.gz")

                if sname not in subj_vols:
                    shape = nib.load(flair_path).shape
                    subj_vols[sname] = np.zeros(shape, dtype=np.float32)

                sl = TF.resize(torch.tensor(score[b, 0]).unsqueeze(0),
                               [subj_vols[sname].shape[0], subj_vols[sname].shape[1]])
                subj_vols[sname][:, :, sidx] = sl.squeeze().numpy().astype(np.float32)

    for subj, vol in subj_vols.items():
        v = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        if smooth:
            v = temporal_smooth3d(v)
        ref = nib.load(os.path.join(base_dir, subj, "flair.nii.gz"))
        out = os.path.join(out_dir, f"{subj}_score_{mode}.nii.gz")
        nib.save(nib.Nifti1Image(v, ref.affine), out)
        print(f"Saved {mode} score map for {subj}")
    return out_dir

import random
import random

def show_random_tumor_comparison(model, base_dir, alpha=0.5, size=128, overlay_pct=99):
    """
    Picks a random subject with tumor, a random tumor slice, runs reconstruction and
    anomaly scoring, and shows a 2x3 figure:
      [Original | Reconstruction | Pixel overlay]
      [Latent overlay | Fusion overlay | Tumor GT]
    """
    # --- Find subjects that contain tumors ---
    tumor_subjects = []
    for subj in os.listdir(base_dir):
        seg_path = os.path.join(base_dir, subj, "seg.nii.gz")
        flair_path = os.path.join(base_dir, subj, "flair.nii.gz")
        if not (os.path.exists(seg_path) and os.path.exists(flair_path)):
            continue
        seg_vol = (nib.load(seg_path).get_fdata() > 0).astype(np.uint8)
        if np.any(seg_vol):
            tumor_subjects.append(subj)

    if not tumor_subjects:
        print("[!] No subjects with tumors found.")
        return

    subj = random.choice(tumor_subjects)
    subj_dir = os.path.join(base_dir, subj)
    flair_vol = nib.load(os.path.join(subj_dir, "flair.nii.gz")).get_fdata()
    seg_vol = (nib.load(os.path.join(subj_dir, "seg.nii.gz")).get_fdata() > 0).astype(np.uint8)

    tumor_slices = np.where(seg_vol.sum(axis=(0, 1)) > 0)[0]
    if len(tumor_slices) == 0:
        print(f"[!] No tumor slices found for {subj}")
        return
    slice_idx = int(random.choice(tumor_slices))

    flair_slice = flair_vol[:, :, slice_idx].astype(np.float32)
    seg_slice = seg_vol[:, :, slice_idx].astype(np.float32)

    flair_slice = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)

    flair_disp = TF.resize(torch.from_numpy(flair_slice).unsqueeze(0), [size, size]).squeeze(0).numpy()
    seg_disp = TF.resize(
        torch.from_numpy(seg_slice).unsqueeze(0),
        [size, size],
        interpolation=InterpolationMode.NEAREST
    ).squeeze(0).numpy()

    masked_disp = flair_disp * (1.0 - seg_disp)

    full_tensor   = torch.tensor(flair_disp,  dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    masked_tensor = torch.tensor(masked_disp, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        recon = model(masked_tensor)[0, 0].cpu().numpy()

        pixel_map  = compute_scores(model, masked_tensor, full_tensor, mode="pixel").cpu().numpy()[0, 0]
        latent_map = compute_scores(model, masked_tensor, full_tensor, mode="latent").cpu().numpy()[0, 0]
        fusion_map = compute_scores(model, masked_tensor, full_tensor, mode="fusion", alpha=alpha).cpu().numpy()[0, 0]

    def norm(x): 
        x = x.astype(np.float32)
        mn, mx = np.min(x), np.max(x)
        return (x - mn) / (mx - mn + 1e-8)

    pixel_map_n  = norm(pixel_map)
    latent_map_n = norm(latent_map)
    fusion_map_n = norm(fusion_map)

    p_thr  = np.percentile(pixel_map_n,  overlay_pct)
    l_thr  = np.percentile(latent_map_n, overlay_pct)
    f_thr  = np.percentile(fusion_map_n, overlay_pct)
    pixel_ol  = (pixel_map_n  >= p_thr).astype(np.float32)
    latent_ol = (latent_map_n >= l_thr).astype(np.float32)
    fusion_ol = (fusion_map_n >= f_thr).astype(np.float32)

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Subject: {subj} | Slice: {slice_idx}", fontsize=14)

    plt.subplot(2, 3, 1)
    plt.imshow(flair_disp, cmap='gray')
    plt.title("Original FLAIR")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(recon, cmap='gray')
    plt.title("Reconstruction (from masked)")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(flair_disp, cmap='gray')
    plt.imshow(pixel_ol, cmap='Reds', alpha=0.4)
    plt.title("Pixel Anomaly (≥99th pct)")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(flair_disp, cmap='gray')
    plt.imshow(latent_ol, cmap='Reds', alpha=0.4)
    plt.title("Latent Anomaly (≥99th pct)")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(flair_disp, cmap='gray')
    plt.imshow(fusion_ol, cmap='Reds', alpha=0.4)
    plt.title("Fusion Anomaly (≥99th pct)")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(flair_disp, cmap='gray')
    plt.imshow(seg_disp, cmap='Greens', alpha=0.35)
    plt.title("Tumor Ground Truth")
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()



def evaluate_scores(base_dir, modes=("pixel", "latent", "fusion")):
    """
    Compute Dice and Jaccard metrics comparing predicted anomalies and tumor masks.
    """
    results = {m: {"dice": [], "jacc": []} for m in modes}
    out_dir = os.path.join(base_dir, "inpainting_outputs")

    for mode in modes:
        print(f"\nEvaluating {mode.upper()} score maps...")

        for subj in os.listdir(base_dir):
            subj_dir = os.path.join(base_dir, subj)
            seg_path = os.path.join(subj_dir, "seg.nii.gz")
            pred_path = os.path.join(out_dir, f"{subj}_score_{mode}.nii.gz")

            if not (os.path.exists(seg_path) and os.path.exists(pred_path)):
                continue

            seg = (nib.load(seg_path).get_fdata() > 0).astype(np.uint8)
            pred = nib.load(pred_path).get_fdata()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            thr = np.percentile(pred, 99)
            pred_mask = (pred > thr).astype(np.uint8)

            inter = np.sum(pred_mask * seg)
            dice = (2.0 * inter) / (np.sum(pred_mask) + np.sum(seg) + 1e-8)
            jacc = inter / (np.sum((pred_mask + seg) > 0) + 1e-8)

            results[mode]["dice"].append(dice)
            results[mode]["jacc"].append(jacc)

        if results[mode]["dice"]:
            d, j = np.mean(results[mode]["dice"]), np.mean(results[mode]["jacc"])
            print(f"{mode:<8} Dice={d:.3f}  Jaccard={j:.3f}")
        else:
            print(f"No valid subjects for mode {mode}")

    print("RESULTS")
    for mode in modes:
        if results[mode]["dice"]:
            d = np.mean(results[mode]["dice"])
            j = np.mean(results[mode]["jacc"])
            print(f"{mode:<8} | Dice={d:.3f} | Jaccard={j:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate an inpainting autoencoder on FLAIR MRI data."
    )
    parser.add_argument("--base_dir", required=True, help="Directory containing subject subfolders.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a pretrained model (optional).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fusion weight between pixel and latent modes.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and inference.")
    parser.add_argument("--help_modes", action="store_true", help="Show available scoring modes and exit.")

    args = parser.parse_args()

    if args.help_modes:
        print("""
Scoring Modes:
  pixel   - Uses raw reconstruction difference.
  latent  - Uses difference between latent encodings.
  fusion  - Combines both maps (controlled by --alpha).
""")
        return

    ds = InpaintingFlairDataset(args.base_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = SmallAE().to(DEVICE)

    # Train or load
    if args.load_model and os.path.exists(args.load_model):
        model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
        print(f"Loaded pretrained model from {args.load_model}")
    else:
        model = train_inpainting(model, loader, epochs=args.epochs)
        torch.save(model.state_dict(), "flair_inpainting_ae.pth")
        print("Saved new model: flair_inpainting_ae.pth")

    # Run inference for all scoring modes
    for mode in ["pixel", "latent", "fusion"]:
        print(f"\nRunning {mode.upper()} anomaly scoring...")
        run_inference(model, loader, args.base_dir, mode=mode, alpha=args.alpha, smooth=True)

    # Evaluate
    evaluate_scores(args.base_dir, modes=["pixel", "latent", "fusion"])
    show_random_tumor_comparison(model, args.base_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
