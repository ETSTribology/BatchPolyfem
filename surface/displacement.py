"""
displacement.py

Provides functions to generate and save a 2D displacement map using arbitrary
noise functions, with support for PNG or TIFF output.
"""

import os
import math
import logging

import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler

# Configure logging for this file
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]  # Colorful logs
)
logger = logging.getLogger(__name__)
console = Console()  # If you want direct Rich printing

def generate_displacement_map(
    noise_func,
    map_size=256,
    noise_params=None,
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
    normalize=True
):
    """
    Generate a 2D displacement map using a given noise function.
    ...
    """
    if noise_params is None:
        noise_params = {}

    min_x, max_x = x_range
    min_y, max_y = y_range

    # Prepare linearly spaced coordinates
    xs = np.linspace(min_x, max_x, map_size)
    ys = np.linspace(min_y, max_y, map_size)

    displacement = np.zeros((map_size, map_size), dtype=np.float32)

    # Evaluate noise for each pixel
    for i in range(map_size):
        for j in range(map_size):
            x_val = xs[i]
            y_val = ys[j]
            displacement[i, j] = noise_func(x_val, y_val, **noise_params)

    # Optionally normalize to [0,1]
    if normalize:
        dmin, dmax = displacement.min(), displacement.max()
        if dmax > dmin:  # avoid division by zero
            displacement = (displacement - dmin) / (dmax - dmin)
        else:
            displacement[:] = 0.0

    return displacement

def save_displacement_map(displacement, filename="displacement.png"):
    """
    Save a 2D numpy array (values in [0,1]) as a grayscale image (PNG or TIFF).
    ...
    """
    disp_clipped = np.clip(displacement, 0.0, 1.0)
    disp_8u = (disp_clipped * 255).astype(np.uint8)
    img = Image.fromarray(disp_8u, mode='L')

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".tiff":
        img.save(filename, format="TIFF")
    else:
        img.save(filename, format="PNG")

    logger.info(f"Saved displacement map to: {filename}")
