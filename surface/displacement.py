"""
displacement.py

Provides functions to generate and save a 2D displacement map using arbitrary
noise functions, with support for PNG or TIFF output. Also includes functionality
to convert a normal map into a displacement map.
"""

import os
import math
import logging

import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler

# Import additional libraries for normal map conversion
import inspect
from scipy import fftpack

# Configure logging for this file
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]  # Colorful logs
)
logger = logging.getLogger(__name__)
console = Console()  # If you want direct Rich printing

def generate_displacement_map(noise_func, map_size, noise_params, normalize=True):
    # Inspect the noise function's signature
    sig = inspect.signature(noise_func)
    valid_params = {k: v for k, v in noise_params.items() if k in sig.parameters}
    
    # Create grid of coordinates
    x, y = np.meshgrid(
        np.linspace(0, 1, map_size, endpoint=False),
        np.linspace(0, 1, map_size, endpoint=False)
    )
    
    try:
        # Try calling noise_func directly (assuming it supports array inputs)
        disp_map = noise_func(x, y, **valid_params)
    except TypeError as te:
        logger.warning(f"Noise function '{noise_func.__name__}' is not vectorized. Applying np.vectorize. Error: {te}")
        # Fallback: vectorize the function on the fly,
        # explicitly converting inputs to floats.
        vectorized_func = np.vectorize(
            lambda xi, yi: noise_func(float(xi), float(yi), **valid_params)
        )
        disp_map = vectorized_func(x, y)
    except Exception as e:
        logger.error(f"Error in noise_func '{noise_func.__name__}' with params {noise_params}: {e}")
        raise

    if normalize:
        disp_map = (disp_map - np.min(disp_map)) / (np.max(disp_map) - np.min(disp_map))
    
    return disp_map.astype(np.float32)


def save_displacement_map(displacement, filename="displacement.png"):
    """
    Save a 2D numpy array (values in [0,1]) as a grayscale image (PNG or TIFF).

    Parameters:
    -----------
    displacement : np.ndarray
        2D array representing the displacement map. Values should be in [0,1].
    filename     : str
        Output filename. Supports PNG and TIFF formats based on the extension.
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
