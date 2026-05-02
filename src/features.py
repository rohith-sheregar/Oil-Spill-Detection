"""
Module 2 — Feature Extraction
Extracts geometric, morphological, and contextual features from a binary
segmentation mask. Features are used downstream by the Random Forest
look-alike rejection classifier.
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class PatchFeatures:
    """Geometric, morphological and contextual features for one connected patch."""
    area_pixels:    float
    area_km2:       float
    perimeter:      float
    elongation:     float
    aspect_ratio:   float
    compactness:    float
    solidity:       float
    extent:         float
    hu_moment_1:    float
    hu_moment_2:    float
    mean_intensity: float
    std_intensity:  float
    is_night:       float  # 1.0 = night, 0.0 = day


def extract_features(
    mask: np.ndarray,
    image: np.ndarray,
    acquisition_hour: int = 2,
) -> list:
    """
    Extract geometric, morphological, and contextual features from each
    connected component in a binary segmentation mask.

    Args:
        mask:             Binary numpy array of shape (H, W), values 0 or 1.
        image:            Original SAR grayscale image of shape (H, W).
        acquisition_hour: UTC hour the image was acquired (default 2 = night).

    Returns:
        List of PatchFeatures, one per valid connected component (area >= 50 px).
    """
    # Ensure correct dtypes
    mask_u8 = (mask > 0).astype(np.uint8)

    # Determine night flag once — applies to all patches in this image
    is_night = 1.0 if (acquisition_hour < 6 or acquisition_hour >= 20) else 0.0

    # Connected components analysis
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    features: list[PatchFeatures] = []

    for label_id in range(1, num_labels):  # skip background label 0
        # Area in pixels
        area_pixels = float(stats[label_id, cv2.CC_STAT_AREA])
        if area_pixels < 50:
            continue  # skip noise components

        # Isolate this component's binary mask
        component_mask = (labels_map == label_id).astype(np.uint8)

        # ── Bounding box ─────────────────────────────────────────────────────
        bx = stats[label_id, cv2.CC_STAT_LEFT]
        by = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]

        bounding_rect_area = float(bw * bh)
        aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

        # ── Contour-based features ────────────────────────────────────────────
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(contour, closed=True)
        perimeter = max(perimeter, 1e-6)  # avoid divide-by-zero

        # Compactness (circularity): 1.0 = perfect circle
        compactness = (4.0 * math.pi * area_pixels) / (perimeter ** 2)

        # ── Ellipse fitting → elongation ─────────────────────────────────────
        elongation = 1.0  # default for degenerate cases
        if len(contour) >= 5:
            try:
                (_, _), (minor_axis, major_axis), _ = cv2.fitEllipse(contour)
                if minor_axis > 0:
                    elongation = float(major_axis) / float(minor_axis)
                else:
                    elongation = float(major_axis) if major_axis > 0 else 1.0
            except cv2.error:
                elongation = 1.0

        # ── Solidity ─────────────────────────────────────────────────────────
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area_pixels) / float(hull_area) if hull_area > 0 else 1.0

        # ── Extent ───────────────────────────────────────────────────────────
        extent = float(area_pixels) / bounding_rect_area if bounding_rect_area > 0 else 1.0

        # ── Hu Moments ───────────────────────────────────────────────────────
        moments = cv2.moments(component_mask)
        hu = cv2.HuMoments(moments).flatten()
        hu_moment_1 = float(hu[0])
        hu_moment_2 = float(hu[1])

        # ── Intensity statistics inside the patch ─────────────────────────────
        # Ensure image is 2D grayscale
        if image.ndim == 3:
            gray_image = image[:, :, 0].astype(np.float32)
        else:
            gray_image = image.astype(np.float32)

        pixel_vals = gray_image[component_mask == 1]
        mean_intensity = float(np.mean(pixel_vals)) if len(pixel_vals) > 0 else 0.0
        std_intensity  = float(np.std(pixel_vals))  if len(pixel_vals) > 0 else 0.0

        # ── Area in km² (Sentinel-1: 10m resolution) ─────────────────────────
        area_km2 = area_pixels * (10.0 * 10.0) / 1e6

        features.append(PatchFeatures(
            area_pixels=area_pixels,
            area_km2=area_km2,
            perimeter=perimeter,
            elongation=elongation,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            solidity=solidity,
            extent=extent,
            hu_moment_1=hu_moment_1,
            hu_moment_2=hu_moment_2,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            is_night=is_night,
        ))

    return features


def features_to_array(features: list) -> np.ndarray:
    """
    Convert a list of PatchFeatures to a 2D numpy array of shape (N, 13).

    Args:
        features: List of PatchFeatures objects.

    Returns:
        np.ndarray of shape (N, 13), or (0, 13) if features is empty.
    """
    if not features:
        return np.empty((0, 13), dtype=np.float32)

    rows = []
    for f in features:
        rows.append([
            f.area_pixels,
            f.area_km2,
            f.perimeter,
            f.elongation,
            f.aspect_ratio,
            f.compactness,
            f.solidity,
            f.extent,
            f.hu_moment_1,
            f.hu_moment_2,
            f.mean_intensity,
            f.std_intensity,
            f.is_night,
        ])
    return np.array(rows, dtype=np.float32)


def describe_patch(f) -> str:
    """
    Return a human-readable summary string for a single PatchFeatures instance.

    Args:
        f: A PatchFeatures dataclass instance.

    Returns:
        Formatted description string.
    """
    return (
        f"Area: {f.area_km2:.3f}km² | "
        f"Elongation: {f.elongation:.2f} | "
        f"Compactness: {f.compactness:.3f} | "
        f"Night: {bool(f.is_night)}"
    )
