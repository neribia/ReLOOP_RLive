# rlive-demo

Demo utilities and example scripts for the ReLoop_RLive project.

## 📦 What's Included

This package provides demonstration and example code for using the ReLoop_RLive ecosystem:

- Example scripts and workflows
- Demo implementations

## Installation

To install this package, run:

```bash
uv sync 
```

# Demo Pipelines

Pipelines and extractors for detecting a Sphero Bolt+ robot (transparent shell, blue LED strips)
inside a gray plastic box viewed from a top-down camera.

The main challenge: the ball is gray/transparent on a gray surface → low contrast.
The blue LED strip is the most reliable visual feature.

---

## Available Processors

| Processor | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `GrayscaleProcessor` | BGR (3ch) | Gray (1ch) | Color to grayscale |
| `GaussianBlurProcessor` | any | same | Noise reduction |
| `CannyProcessor` | Gray (1ch) | Gray (1ch) | Edge detection |
| `LaplacianProcessor` | Gray (1ch) | Gray (1ch) | Edge detection |
| `ThresholdProcessor` | Gray (1ch) | Binary (1ch) | Binary threshold |
| `DilationProcessor` | Binary/Gray | same | Morphological grow |
| `ErosionProcessor` | Binary/Gray | same | Morphological shrink |
| `CLAHEProcessor` | Gray (1ch) | Gray (1ch) | Local contrast enhancement |
| `HSVProcessor` | BGR (3ch) | HSV (3ch) or Mask (1ch) | Color space / color filter |
| `ChannelExtractorProcessor` | Multi (3ch) | Single (1ch) | Extract one channel |
| `ImageAbsDiff` | any | same | Frame-to-frame difference |
| `MOG2BackgroundSubtractorProcessor` | BGR (3ch) | Binary (1ch) | Learned background subtraction |
| `FFTBandpassProcessor` | Gray (1ch) | Gray (1ch) | Frequency filtering |

## Available Extractors

| Extractor | Strategy | Best suited for |
|-----------|----------|-----------------|
| `ContourExtractor` | Largest contour centroid | Clean binary masks |
| `HoughCircleExtractor` | Hough circle transform | Circular edges after blur |
| `MomentsExtractor` | Most-circular contour via `cv.moments()` | Noisy masks, rejects box corners |
| `PolygonApproxExtractor` | `cv.approxPolyDP()` vertex matching | Contours with known polygon shape |
| `ImageDiffExtractor` | Motion contours from frame diff | Moving objects between frames |

---

## Pipeline 1 — Blue LED HSV Masking + Contour ✅ WORKS

**Status:** ✅ Confirmed working.

**Idea:** The blue LED strip on the Sphero is the strongest color feature.
Isolate the blue hue range, threshold, then find the largest contour.

```python
pipeline = ImagePipeline([
    HSVProcessor(
        lower_hue=90,  upper_hue=130,   # blue hue range
        lower_sat=50,  upper_sat=255,   # require some color (not gray)
        lower_val=50,  upper_val=255,   # require some brightness (not black)
        apply_mask=True,
    ),
    DilationProcessor(kernel_size=(7, 7), iterations=3),
])
extractor = ContourExtractor(min_contour_area=50)
```

**Why it works:** The blue LEDs are the only saturated blue in the gray box.
The HSV mask isolates them cleanly. Dilation merges nearby LED pixels into one blob.

**When it fails:** LEDs are off, LED color changes, or strong blue reflections on the box walls.

---

## Pipeline 2 — Grayscale + Laplacian + Threshold + Moments ✅ WORKS (with tuning)

**Status:** ✅ Works with correct parameters. The ball is clearly visible in the mask.

**Key fix:** `MomentsExtractor` now has a `min_circularity` parameter that rejects
box corners (circularity < 0.3) and only picks truly round contours.

```python
pipeline = ImagePipeline([
    GrayscaleProcessor(),
    LaplacianProcessor(ksize=3),
    ThresholdProcessor(threshold_value=100, max_value=255),
    DilationProcessor(kernel_size=(5, 5), iterations=2),
])
extractor = MomentsExtractor(min_contour_area=800, min_circularity=0.4)
```

**Why it works:** Laplacian detects edges. The ball produces a round contour.
Box corners produce contours with low circularity (< 0.3).
`min_circularity=0.4` rejects all rectangular/angular shapes and only accepts round blobs.

**Tuning tips:**
- Increase `min_circularity` (e.g. 0.5) if box corners still get selected.
- Decrease `min_contour_area` if ball appears small in frame.
- The debug image now shows: gray = rejected by circularity, green = passed, red = selected.

---

## Pipeline 3 — MOG2 Background Subtraction ✅ WORKS (video only)

**Status:** ✅ Confirmed working for video streams.

**New feature:** You can now save and load the background model to skip the warm-up phase.

```python
pipeline = ImagePipeline([
    MOG2BackgroundSubtractorProcessor(
        var_threshold=25.0,
        detect_shadows=False,
    ),
    ErosionProcessor(kernel_size=(3, 3), iterations=1),
    DilationProcessor(kernel_size=(7, 7), iterations=3),
])
extractor = ContourExtractor(min_contour_area=100)
```

**Saving and loading the background model:**
```python
# Get the MOG2 processor from the pipeline
mog2 = pipeline.steps[0]

# After ~50 frames, save the learned background model
mog2.save_model("bg_model.png")

# Later, load it to skip the warm-up phase
mog2.load_model("bg_model.png")
```

**When it fails:** First ~20 frames (learning phase). Not suitable for single images.

---

## Recommendations

| Scenario | Recommended Pipeline |
|----------|---------------------|
| **LEDs on, any mode** | **Pipeline 1** (HSV blue mask) — most reliable |
| **LEDs on, video stream** | Pipeline 1 or Pipeline 4 (MOG2) |
| **LEDs off, static image** | **Pipeline 3** (Laplacian + Moments with circularity) |
| **LEDs off, video stream** | **Pipeline 4** (MOG2 with saved model) |
| **Ball moving in video** | Pipeline 4 (MOG2) |
| **Saturation-based fallback** | Pipeline 7 (HSV saturation + Moments) |

### Quick Start — Most Reliable (LEDs on)

```python
from rlive_env.localisation.processors import *
from rlive_env.localisation.extractors import ContourExtractor

pipeline = ImagePipeline([
    HSVProcessor(
        lower_hue=90,  upper_hue=130,
        lower_sat=50,  upper_sat=255,
        lower_val=50,  upper_val=255,
        apply_mask=True,
    ),
    DilationProcessor(kernel_size=(7, 7), iterations=3),
])
extractor = ContourExtractor(min_contour_area=50)
```

### Quick Start — LEDs off

```python
from rlive_env.localisation.processors import *
from rlive_env.localisation.extractors import MomentsExtractor

pipeline = ImagePipeline([
    GrayscaleProcessor(),
    LaplacianProcessor(ksize=3),
    ThresholdProcessor(threshold_value=100, max_value=255),
    DilationProcessor(kernel_size=(5, 5), iterations=2),
])
extractor = MomentsExtractor(min_contour_area=800, min_circularity=0.4)
```

