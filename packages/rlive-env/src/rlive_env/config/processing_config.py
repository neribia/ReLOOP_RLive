"""Processing configuration for ball localisation - environment variables with defaults."""
import os

# --- Gaussian Blur ---
BLUR_KERNEL_SIZE: int = int(os.getenv("BALL_BLUR_KERNEL_SIZE", "9"))
"""
Gaussian blur kernel size (must be odd).
default: 9
"""

BLUR_SIGMA: float = float(os.getenv("BALL_BLUR_SIGMA", "2.0"))
"""
Gaussian blur sigma (standard deviation).
default: 2.0
"""

# --- Canny Edge Detection ---
CANNY_THRESHOLD1: int = int(os.getenv("BALL_CANNY_THRESHOLD1", "50"))
"""
Canny edge detection lower threshold.
default: 50
"""

CANNY_THRESHOLD2: int = int(os.getenv("BALL_CANNY_THRESHOLD2", "150"))
"""
Canny edge detection upper threshold.
default: 150
"""

CANNY_APERTURE: int = int(os.getenv("BALL_CANNY_APERTURE", "3"))
"""
Canny aperture size (3, 5, or 7).
default: 3
"""

# --- Laplacian Edge Detection ---
LAPLACIAN_KSIZE: int = int(os.getenv("BALL_LAPLACIAN_KSIZE", "3"))
"""
Laplacian kernel size (1, 3, 5, or 7).
default: 3
"""

LAPLACIAN_THRESHOLD: int = int(os.getenv("BALL_LAPLACIAN_THRESHOLD", "30"))
"""
Laplacian edge detection threshold.
default: 30
"""

# --- Binary Threshold ---
THRESHOLD_VALUE: int = int(os.getenv("BALL_THRESHOLD_VALUE", "127"))
"""
Binary threshold value (0-255).
default: 127
"""

THRESHOLD_MAX: int = int(os.getenv("BALL_THRESHOLD_MAX", "255"))
"""
Maximum value for thresholding.
default: 255
"""

# --- Dilation ---
DILATION_KERNEL_WIDTH: int = int(os.getenv("BALL_DILATION_KERNEL_WIDTH", "3"))
"""
Dilation kernel width.
default: 3
"""

DILATION_KERNEL_HEIGHT: int = int(os.getenv("BALL_DILATION_KERNEL_HEIGHT", "3"))
"""
Dilation kernel height.
default: 3
"""

DILATION_ITERATIONS: int = int(os.getenv("BALL_DILATION_ITERATIONS", "2"))
"""
Number of dilation iterations.
default: 2
"""

# --- FFT Bandpass ---
FFT_LOW_FREQ: int = int(os.getenv("BALL_FFT_LOW_FREQ", "10"))
"""
FFT bandpass filter lower frequency.
default: 10
"""

FFT_HIGH_FREQ: int = int(os.getenv("BALL_FFT_HIGH_FREQ", "100"))
"""
FFT bandpass filter upper frequency.
default: 100
"""

# --- Pipeline Control ---
ENABLE_PROCESSING_PIPELINE: bool = os.getenv("BALL_ENABLE_PROCESSING_PIPELINE", "true").lower() == "true"
"""
Enable or disable the image processing pipeline.
default: true
"""
