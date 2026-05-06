from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image  # used internally for torchvision transforms
from scipy import linalg

import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3


from ignite.metrics import FID
from ignite.engine import Engine


def load_images_from_folder(folder: str | Path) -> np.ndarray:
    """
    Load all images from a folder into a numpy array.

    Parameters
    ----------
    folder : str | Path
        Path to the folder containing images.

    Returns
    -------
    np.ndarray
        Shape (N, H, W, 3), dtype uint8.
    """
    folder = Path(folder)
    imgs = []
    for fpath in sorted(folder.iterdir()):
        if fpath.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
            img = cv.imread(str(fpath))
            if img is None:
                raise ValueError(f"Could not read image: {fpath}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            imgs.append(img)
    if not imgs:
        raise ValueError(f"No images found in folder: {folder}")
    return np.stack(imgs)  # (N, H, W, 3)


def get_inception_features(
    images_np: np.ndarray,
    batch_size: int = 32,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Extract 2048-d Inception-v3 pool features from a set of images.

    Parameters
    ----------
    images_np : np.ndarray
        Shape (N, H, W, 3), dtype uint8.
    batch_size : int
        Number of images per inference batch.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        Shape (N, 2048), dtype float32.
    """
    model = inception_v3(weights='DEFAULT', transform_input=False)
    model.fc = torch.nn.Identity()  # strip classifier → 2048-d pool features
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    feats = []
    with torch.no_grad():
        for i in range(0, len(images_np), batch_size):
            batch = images_np[i: i + batch_size]
            tensors = torch.stack([
                preprocess(Image.fromarray(img)) for img in batch
            ]).to(device)
            out = model(tensors)  # (B, 2048)
            feats.append(out.cpu().numpy())

    return np.concatenate(feats, axis=0)


def compute_fid(feats_real: np.ndarray, feats_fake: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute the Fréchet Inception Distance (FID) between two feature sets.

    Parameters
    ----------
    feats_real : np.ndarray
        Shape (N, 2048) – features from the real image distribution.
    feats_fake : np.ndarray
        Shape (M, 2048) – features from the generated/simulated image distribution.
    eps : float
        Small value added to the diagonal of covariance matrices for numerical stability.

    Returns
    -------
    float
        FID score (lower is better).
    """
    mu1, sigma1 = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
    mu2, sigma2 = feats_fake.mean(axis=0), np.cov(feats_fake, rowvar=False)

    diff = mu1 - mu2

    # Regularise covariance matrices for numerical stability
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # Matrix square-root via eigendecomposition
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):  # clip tiny imaginary artefacts
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


def calc_fid_from_folders(
    real_folder: str | Path,
    fake_folder: str | Path,
    batch_size: int = 32,
    device: str | None = None,
) -> float:
    """
    High-level helper: load images from two folders and return their FID score.

    Parameters
    ----------
    real_folder : str
        Path to the folder with real images.
    fake_folder : str
        Path to the folder with generated/simulated images.
    batch_size : int
        Batch size for Inception inference.
    device : str or None
        'cpu', 'cuda', or None (auto-detect).

    Returns
    -------
    float
        FID score.

    Example
    -------
    fid = calc_fid_from_folders("path/to/real", "path/to/sim")
    print(f"FID = {fid:.4f}")
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Loading real images from:  {real_folder}")
    real_images = load_images_from_folder(real_folder)
    print(f"  → {len(real_images)} images loaded")

    print(f"Loading fake/sim images from: {fake_folder}")
    fake_images = load_images_from_folder(fake_folder)
    print(f"  → {len(fake_images)} images loaded")

    print("Extracting Inception features for real images …")
    feats_real = get_inception_features(real_images, batch_size=batch_size, device=device)

    print("Extracting Inception features for fake/sim images …")
    feats_fake = get_inception_features(fake_images, batch_size=batch_size, device=device)

    fid_score = compute_fid(feats_real, feats_fake)
    print(f"FID = {fid_score:.4f}")
    return fid_score

# ── Ignite-based FID ───────────────────────────────────────────────────────

def _images_to_tensor_dataset(
    images_np: np.ndarray,
    device: str,
) -> torch.utils.data.TensorDataset:
    """Convert (N, H, W, 3) uint8 numpy array to a resized, normalised TensorDataset."""
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensors = torch.stack([
        preprocess(Image.fromarray(img)) for img in images_np
    ])  # (N, 3, 299, 299)
    return torch.utils.data.TensorDataset(tensors)

def calc_fid_ignite_from_folders(
        real_folder: str | Path,
        fake_folder: str | Path,
        batch_size: int = 32,
        device: str | None = None,
) -> float:
    """
    Compute FID using pytorch-ignite's ignite.metrics.FID.

    Requires: pip install pytorch-ignite

    Parameters
    ----------
    real_folder : str | Path
        Path to real images folder.
    fake_folder : str | Path
        Path to generated/simulated images folder.
    batch_size : int
        Batch size for Inception inference.
    device : str | None
        'cpu', 'cuda', or None (auto-detect).

    Returns
    -------
    float
        FID score (lower is better).

    Example
    -------
    fid = calc_fid_ignite_from_folders("path/to/real", "path/to/sim")
    print(f"FID = {fid:.4f}")
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    print(f"Loading real images from:  {real_folder}")
    real_images = load_images_from_folder(real_folder)
    print(f"  → {len(real_images)} images loaded")

    print(f"Loading fake/sim images from: {fake_folder}")
    fake_images = load_images_from_folder(fake_folder)
    print(f"  → {len(fake_images)} images loaded")

    real_ds = _images_to_tensor_dataset(real_images, device)
    fake_ds = _images_to_tensor_dataset(fake_images, device)

    real_loader = torch.utils.data.DataLoader(real_ds, batch_size=batch_size)
    fake_loader = torch.utils.data.DataLoader(fake_ds, batch_size=batch_size)

    # ignite.metrics.FID needs a dummy engine whose output is (fake, real)
    fid_metric = FID(device=device)

    def eval_step(engine, batch):
        return batch  # batch is already a tuple of tensors from TensorDataset

    # Feed paired batches; zip stops at the shorter loader
    engine = Engine(eval_step)
    fid_metric.attach(engine, "fid")

    # Manually update the metric by iterating both loaders in sync
    fid_metric.reset()
    for (real_batch,), (fake_batch,) in zip(real_loader, fake_loader):
        real_batch = real_batch.to(device)
        fake_batch = fake_batch.to(device)
        fid_metric.update((fake_batch, real_batch))

    fid_score = float(fid_metric.compute())
    print(f"FID (ignite) = {fid_score:.4f}")
    return fid_score


# ── Entry point (optional: run as a standalone script) ─────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Compute FID between two image folders.")
    parser.add_argument('real_folder', type=str, help="Path to real images folder.")
    parser.add_argument('fake_folder', type=str, help="Path to generated/sim images folder.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'])
    args = parser.parse_args()

    calc_fid_from_folders(
        real_folder=args.real_folder,
        fake_folder=args.fake_folder,
        batch_size=args.batch_size,
        device=args.device,
    )

