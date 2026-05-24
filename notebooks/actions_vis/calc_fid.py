from pathlib import Path
import numpy as np
from PIL import Image
from scipy import linalg
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3

from inception import InceptionV3


def load_images_from_folder(folder: str | Path) -> np.ndarray:
    """Load all images from a folder into a numpy array.

    Args:
        folder: Path to the folder containing images.

    Returns:
        np.ndarray of shape (N, H, W, 3), dtype uint8.
    """
    folder = Path(folder)
    imgs = []
    for fpath in sorted(folder.iterdir()):
        if fpath.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
            img = Image.open(fpath).convert('RGB')
            imgs.append(np.array(img))
    if not imgs:
        raise ValueError(f"No images found in folder: {folder}")
    return np.stack(imgs)  # (N, H, W, 3)


def get_inception_features(
    images_np: np.ndarray,
    batch_size: int = 32,
    device: str = 'cpu',
) -> np.ndarray:
    """Extract 2048-d Inception-v3 pool features from a set of images.

    Args:
        images_np: Shape (N, H, W, 3), dtype uint8.
        batch_size: Number of images per inference batch.
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        np.ndarray of shape (N, 2048), dtype float32.
    """
    model = inception_v3(weights='DEFAULT', transform_input=True)
    model.fc = torch.nn.Identity()  # strip classifier → 2048-d pool features
    model.aux_logits = False
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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


def save_features(feats: np.ndarray, path: str | Path) -> None:
    """Save Inception features to a .npy file.

    Args:
        feats: Shape (N, 2048) feature array to save.
        path: Destination file path (e.g. ``"feats_real.npy"``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    print(f"Saved features → {path}  ({len(feats)} vectors)")


def load_features(path: str | Path) -> np.ndarray:
    """Load Inception features previously saved with :func:`save_features`.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        np.ndarray of shape (N, 2048), dtype float32.
    """
    path = Path(path)
    feats = np.load(path)
    print(f"Loaded features ← {path}  ({len(feats)} vectors)")
    return feats


def get_or_compute_features(
    images_folder: str | Path,
    cache_path: str | Path,
    batch_size: int = 32,
    device: str = 'cpu',
    force_recompute: bool = False,
) -> np.ndarray:
    """Load cached features if available, otherwise compute and save them.

    Args:
        images_folder: Folder containing the source images.
        cache_path: Path where the ``.npy`` cache file is stored/read.
        batch_size: Batch size for Inception inference (only used when computing).
        device: ``'cpu'`` or ``'cuda'`` (only used when computing).
        force_recompute: If ``True``, ignore the cache and always recompute.

    Returns:
        np.ndarray of shape (N, 2048).
    """
    cache_path = Path(cache_path)
    if not force_recompute and cache_path.exists():
        return load_features(cache_path)

    images = load_images_from_folder(images_folder)
    print(f"  → {len(images)} images loaded, extracting features …")
    feats = get_inception_features(images, batch_size=batch_size, device=device)
    save_features(feats, cache_path)
    return feats


# ── pytorch-fid reference extractor (original TF weights) ─────────────────

def get_pytorch_fid_features(
    images_np: np.ndarray,
    batch_size: int = 32,
    device: str = 'cpu',
) -> np.ndarray:
    """Extract 2048-d features using the *exact* pytorch-fid Inception model.

    Uses the original TensorFlow FID weights (ported to PyTorch) and the
    patched Inception architecture from ``inception.py``. Results are directly
    comparable to published FID scores that use the pytorch-fid reference
    implementation.

    Differences vs :func:`get_inception_features`:

    * Weights: ``pt_inception-2015-12-05-6726825d.pth`` (TF port)
      vs torchvision's ImageNet weights.
    * Architecture: patched pooling layers (``FIDInceptionA/C/E``) that
      replicate TensorFlow's average-pool behaviour.
    * Normalisation: model applies ``2x - 1`` internally
      (``normalize_input=True``); input must be in ``[0, 1]``.

    Args:
        images_np: Shape (N, H, W, 3), dtype uint8.
        batch_size: Number of images per inference batch.
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        np.ndarray of shape (N, 2048), dtype float32.
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3(
        output_blocks=[block_idx],
        resize_input=True,       # bilinear resize to 299×299 internally
        normalize_input=True,    # scales [0, 1] → [-1, 1] internally
        use_fid_inception=True,  # download & use original TF FID weights
    ).to(device)
    model.eval()

    # Only ToTensor — model handles resize + normalisation itself
    preprocess = transforms.ToTensor()

    feats = []
    with torch.no_grad():
        for i in range(0, len(images_np), batch_size):
            batch = images_np[i: i + batch_size]
            tensors = torch.stack([
                preprocess(Image.fromarray(img)) for img in batch
            ]).to(device)
            out = model(tensors)[0]             # InceptionV3 returns a list
            out = out.squeeze(3).squeeze(2)     # (B, 2048, 1, 1) → (B, 2048)
            feats.append(out.cpu().numpy())

    return np.concatenate(feats, axis=0)


def get_or_compute_pytorch_fid_features(
    images_folder: str | Path,
    cache_path: str | Path,
    batch_size: int = 32,
    device: str = 'cpu',
    force_recompute: bool = False,
) -> np.ndarray:
    """Load cached pytorch-fid features if available, otherwise compute and save.

    Identical interface to :func:`get_or_compute_features` but uses
    :func:`get_pytorch_fid_features` (original TF weights) under the hood.
    Use a distinct cache filename (e.g. ``feats_real_pfid.npy``) to avoid
    mixing caches from the two extractors.

    Args:
        images_folder: Folder containing source images.
        cache_path: Path for the ``.npy`` cache file.
        batch_size: Batch size for Inception inference (only used when computing).
        device: ``'cpu'`` or ``'cuda'`` (only used when computing).
        force_recompute: If ``True``, ignore cache and always recompute.

    Returns:
        np.ndarray of shape (N, 2048).
    """
    cache_path = Path(cache_path)
    if not force_recompute and cache_path.exists():
        return load_features(cache_path)

    images = load_images_from_folder(images_folder)
    print(f"  → {len(images)} images loaded, extracting pytorch-fid features …")
    feats = get_pytorch_fid_features(images, batch_size=batch_size, device=device)
    save_features(feats, cache_path)
    return feats


def compute_fid(feats_real: np.ndarray, feats_fake: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Fréchet Inception Distance (FID) between two feature sets.

    Args:
        feats_real: Shape (N, 2048) – features from the real image distribution.
        feats_fake: Shape (M, 2048) – features from the generated/simulated distribution.
        eps: Small value added to the diagonal of covariance matrices for numerical
            stability.

    Returns:
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


# ── 1-D Gaussian bell-curve visualisation ─────────────────────────────────

def plot_gaussian_1d(
    feats_real: np.ndarray,
    feats_fake: np.ndarray,
    title: str = "FID: Gaussian distributions (PC1 projection)",
    labels: tuple[str, str] = ("Real", "Sim/Fake"),
    colors: tuple[str, str] = ("steelblue", "tomato"),
    show_histogram: bool = True,
    n_bins: int = 40,
    pca=None,
    figsize: tuple[float, float] = (10, 5),
    fontsize: int = 9,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> None:
    """Project inception features onto PC1 and draw 1-D Gaussian bell curves.

    The mean (μ) and standard deviation (σ) are computed from the PC1 projections.
    The distance between the two means corresponds to the mean-shift term of the
    FID (Wasserstein-2) distance: ``FID = ||μ₁ − μ₂||² + Tr(Σ₁ + Σ₂ − 2·sqrtm(Σ₁·Σ₂))``.
    The shaded overlap area is proportional to the Bhattacharyya coefficient — less
    overlap means higher FID.

    Args:
        feats_real: Shape (N, 2048) – Inception features of the real images.
        feats_fake: Shape (M, 2048) – Inception features of the sim/fake images.
        title: Plot title.
        labels: Display labels for (real, fake) distributions.
        colors: Matplotlib colours for (real, fake) distributions.
        show_histogram: If ``True``, also draw a normalised histogram under the curve.
        n_bins: Number of histogram bins (only used when ``show_histogram=True``).
        pca: Pre-fitted sklearn PCA with ``n_components=1``. If ``None``, a new PCA is
            fitted on the concatenation of both feature sets. Pass a pre-fitted object
            to keep the projection axis consistent across multiple calls.
        figsize: Figure size as ``(width, height)`` in inches.
        fontsize: Base font size used for legend labels, axis annotations, and the
            Δμ annotation text.
        save_path: If given, the figure is saved to this path before displaying.
            The file format is inferred from the extension (e.g. ``.pdf``, ``.png``).
        dpi: Rasterisation DPI for data artists when saving to PDF. Has no effect
            on vector elements (axes, labels, text).
    """
    # Project onto PC1
    if pca is None:
        all_feats = np.concatenate([feats_real, feats_fake], axis=0)
        pca = PCA(n_components=1)
        pca.fit(all_feats)

    real_1d = pca.transform(feats_real).ravel()
    fake_1d = pca.transform(feats_fake).ravel()

    mu_r, sig_r = float(real_1d.mean()), float(real_1d.std())
    mu_f, sig_f = float(fake_1d.mean()), float(fake_1d.std())

    x_min = min(mu_r - 4 * sig_r, mu_f - 4 * sig_f)
    x_max = max(mu_r + 4 * sig_r, mu_f + 4 * sig_f)
    x = np.linspace(x_min, x_max, 500)

    pdf_r = norm.pdf(x, mu_r, sig_r)
    pdf_f = norm.pdf(x, mu_f, sig_f)

    fig, ax = plt.subplots(figsize=figsize)

    # Optional normalised histograms
    if show_histogram:
        ax.hist(real_1d, bins=n_bins, density=True,
                color=colors[0], alpha=0.25, label=f"{labels[0]} samples",
                rasterized=True)
        ax.hist(fake_1d, bins=n_bins, density=True,
                color=colors[1], alpha=0.25, label=f"{labels[1]} samples",
                rasterized=True)

    # Bell curves
    ax.plot(x, pdf_r, color=colors[0], lw=2.5,
            label=f"{labels[0]}  μ={mu_r:.2f}, σ={sig_r:.2f}", rasterized=True)
    ax.plot(x, pdf_f, color=colors[1], lw=2.5,
            label=f"{labels[1]}  μ={mu_f:.2f}, σ={sig_f:.2f}", rasterized=True)

    # Overlap fill (grey) + individual non-overlapping tails
    overlap = np.minimum(pdf_r, pdf_f)
    ax.fill_between(x, overlap, alpha=0.20, color='grey', label="Overlap", rasterized=True)
    ax.fill_between(x, pdf_r, overlap, alpha=0.15, color=colors[0], rasterized=True)
    ax.fill_between(x, pdf_f, overlap, alpha=0.15, color=colors[1], rasterized=True)

    # Vertical mean lines
    ax.axvline(mu_r, color=colors[0], linestyle='--', lw=1.5, alpha=0.8)
    ax.axvline(mu_f, color=colors[1], linestyle='--', lw=1.5, alpha=0.8)

    # Double-headed arrow for |Δμ|
    y_arrow = max(pdf_r.max(), pdf_f.max()) * 1.05
    ax.annotate(
        "", xy=(mu_f, y_arrow), xytext=(mu_r, y_arrow),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.8),
    )
    ax.text(
        (mu_r + mu_f) / 2, y_arrow * 1.02,
        f"|Δμ| = {abs(mu_r - mu_f):.2f}",
        ha='center', va='bottom', fontsize=fontsize,
    )

    explained = pca.explained_variance_ratio_[0] * 100
    ax.set_xlabel(f"PC1  ({explained:.1f} % of total variance)", fontsize=fontsize)
    ax.set_ylabel("Probability density", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    plt.show()


# ── 2-D Gaussian distribution visualisation  ───────────────────────────────────

def plot_gaussian_ellipses(
    feats_real: np.ndarray,
    feats_fake: np.ndarray,
    n_std: float = 2.0,
    title: str = "FID: Gaussian distributions (PCA projection)",
    labels: tuple[str, str] = ("Real", "Sim/Fake"),
    colors: tuple[str, str] = ("steelblue", "tomato"),
    pca=None,
    figsize: tuple[float, float] = (8, 6),
    fontsize: int = 9,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> None:
    """Project inception features to 2D via PCA and visualise the two Gaussian distributions.

    The Fréchet (Wasserstein-2) distance between the distributions is the FID score:
    ``FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ − 2·sqrtm(Σ₁·Σ₂))``. The arrow shows the
    mean-shift term; ellipse size/orientation differences represent the covariance term.

    Args:
        feats_real: Shape (N, 2048) – features from the real image distribution.
        feats_fake: Shape (M, 2048) – features from the generated/simulated distribution.
        n_std: Number of standard deviations for the ellipse radius.
        title: Plot title.
        labels: Display labels for (real, fake) distributions.
        colors: Matplotlib colours for (real, fake) distributions.
        pca: Pre-fitted sklearn PCA with ``n_components=2``. If ``None``, a new PCA is
            fitted on the concatenation of both feature sets. Pass a pre-fitted object
            to keep the projection axis consistent across multiple calls.
        figsize: Figure size as ``(width, height)`` in inches.
        fontsize: Base font size used for legend labels and axis annotations.
        save_path: If given, the figure is saved to this path before displaying.
            The file format is inferred from the extension (e.g. ``.pdf``, ``.png``).
        dpi: Rasterisation DPI for data artists when saving to PDF. Has no effect
            on vector elements (axes, labels, text).
    """
    if pca is None:
        all_feats = np.concatenate([feats_real, feats_fake], axis=0)
        pca = PCA(n_components=2)
        pca.fit(all_feats)

    real_2d = pca.transform(feats_real)
    fake_2d = pca.transform(feats_fake)

    mu_real = real_2d.mean(axis=0)
    mu_fake = fake_2d.mean(axis=0)
    cov_real = np.cov(real_2d, rowvar=False)
    cov_fake = np.cov(fake_2d, rowvar=False)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter raw projected points
    ax.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.3, s=10,
               color=colors[0], label=labels[0], rasterized=True)
    ax.scatter(fake_2d[:, 0], fake_2d[:, 1], alpha=0.3, s=10,
               color=colors[1], label=labels[1], rasterized=True)

    # Draw Gaussian ellipses
    for mu, cov, color, label in [
        (mu_real, cov_real, colors[0], labels[0]),
        (mu_fake, cov_fake, colors[1], labels[1]),
    ]:
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0))
        ellipse = Ellipse(
            xy=mu, width=width, height=height, angle=angle,
            edgecolor=color, facecolor=color, alpha=0.15,
            linewidth=2.5, linestyle='--',
            label=f"{label} Gaussian ({n_std}σ)",
            rasterized=True,
        )
        ellipse_border = Ellipse(
            xy=mu, width=width, height=height, angle=angle,
            edgecolor=color, facecolor='none',
            linewidth=2.5, rasterized=True,
        )
        ax.add_patch(ellipse)
        ax.add_patch(ellipse_border)
        ax.plot(*mu, marker='x', color=color, markersize=12, markeredgewidth=2.5,
                zorder=5, rasterized=True)

    # Arrow between the two means (mean-shift term of FID)
    ax.annotate(
        "", xy=mu_fake, xytext=mu_real,
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.8,
                        connectionstyle="arc3,rad=0.0"),
    )
    mid = (mu_real + mu_fake) / 2
    mean_dist = float(np.linalg.norm(mu_real - mu_fake))
    ax.text(mid[0], mid[1], f"  Δμ = {mean_dist:.2f}", fontsize=fontsize,
            va='bottom', ha='left')

    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0] * 100:.1f} % var)", fontsize=fontsize)
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1] * 100:.1f} % var)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)
    ax.legend(loc="upper right", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    plt.show()


# ── 3-D scatter + ellipsoid visualisation ─────────────────────────────────

def plot_gaussian_3d(
    feats_real: np.ndarray,
    feats_fake: np.ndarray,
    n_std: float = 2.0,
    title: str = "FID: Gaussian distributions (PCA 3D projection)",
    labels: tuple[str, str] = ("Real", "Sim/Fake"),
    colors: tuple[str, str] = ("steelblue", "tomato"),
    pca=None,
    figsize: tuple[float, float] = (10, 7),
    fontsize: int = 9,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> None:
    """Project inception features to 3D via PCA and visualise as scatter clouds with ellipsoids.

    Args:
        feats_real: Shape (N, 2048) – features from the real image distribution.
        feats_fake: Shape (M, 2048) – features from the generated/simulated distribution.
        n_std: Number of standard deviations for the ellipsoid radius.
        title: Plot title.
        labels: Display labels for (real, fake) distributions.
        colors: Matplotlib colours for (real, fake) distributions.
        pca: Pre-fitted sklearn PCA with ``n_components=3``. If ``None``, a new PCA is
            fitted on the concatenation of both feature sets. Pass a pre-fitted object
            to keep the coordinate system consistent across multiple calls.
        figsize: Figure size as ``(width, height)`` in inches.
        fontsize: Base font size used for legend labels and axis annotations.
        save_path: If given, the figure is saved to this path before displaying.
            The file format is inferred from the extension (e.g. ``.pdf``, ``.png``).
        dpi: Rasterisation DPI for data artists when saving to PDF. Has no effect
            on vector elements (axes, labels, text).
    """
    if pca is None:
        all_feats = np.concatenate([feats_real, feats_fake], axis=0)
        pca = PCA(n_components=3)
        pca.fit(all_feats)

    real_3d = pca.transform(feats_real)
    fake_3d = pca.transform(feats_fake)

    mu_real = real_3d.mean(axis=0)
    mu_fake = fake_3d.mean(axis=0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter
    ax.scatter(real_3d[:, 0], real_3d[:, 1], real_3d[:, 2],
               alpha=0.25, s=8, color=colors[0], label=labels[0], rasterized=True)
    ax.scatter(fake_3d[:, 0], fake_3d[:, 1], fake_3d[:, 2],
               alpha=0.25, s=8, color=colors[1], label=labels[1], rasterized=True)

    # Ellipsoids via parametric sphere transformed by covariance eigenvectors
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([sphere_x.ravel(), sphere_y.ravel(), sphere_z.ravel()], axis=1)

    for feats_3d, mu, color, label in [
        (real_3d, mu_real, colors[0], labels[0]),
        (fake_3d, mu_fake, colors[1], labels[1]),
    ]:
        cov = np.cov(feats_3d, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)
        radii = n_std * np.sqrt(eigvals)
        ellipsoid = (eigvecs @ (radii[:, None] * sphere.T)).T + mu
        ex = ellipsoid[:, 0].reshape(sphere_x.shape)
        ey = ellipsoid[:, 1].reshape(sphere_y.shape)
        ez = ellipsoid[:, 2].reshape(sphere_z.shape)
        ax.plot_surface(ex, ey, ez, color=color, alpha=0.10, linewidth=0, rasterized=True)
        ax.plot_wireframe(ex, ey, ez, color=color, alpha=0.20, linewidth=0.4,
                          rstride=4, cstride=4, rasterized=True)
        ax.scatter(*mu, color=color, s=80, marker='x', zorder=5, linewidths=2, rasterized=True)

    # Arrow between means
    ax.quiver(*mu_real, *(mu_fake - mu_real), color='black', linewidth=1.5,
              arrow_length_ratio=0.1)

    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=fontsize)
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=fontsize)
    ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    plt.show()

