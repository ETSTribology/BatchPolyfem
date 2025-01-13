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

def generate_displacement_map(noise_funcs, map_size, noise_params_list, normalize=True, blend_weights=None):
    """
    Generate a 2D displacement map using one or multiple noise functions.

    Parameters:
    -----------
    noise_funcs : list of callables
        A list of noise functions to apply (e.g., [sine_noise, fbm_noise]).
    map_size : int
        Size of the output displacement map (map_size x map_size).
    noise_params_list : list of dict
        A list of parameter dictionaries, one for each noise function.
    normalize : bool
        Whether to normalize the final map to [0, 1].
    blend_weights : list of float or None
        Weights for blending multiple noise maps. If None, equal weights are applied.

    Returns:
    --------
    np.ndarray
        The generated displacement map as a 2D array.
    """
    if not isinstance(noise_funcs, list) or not isinstance(noise_params_list, list):
        raise ValueError("noise_funcs and noise_params_list must be lists.")

    if len(noise_funcs) != len(noise_params_list):
        raise ValueError("noise_funcs and noise_params_list must have the same length.")

    if blend_weights is None:
        blend_weights = [1.0 / len(noise_funcs)] * len(noise_funcs)

    if len(blend_weights) != len(noise_funcs):
        raise ValueError("blend_weights must have the same length as noise_funcs.")

    # Create grid of coordinates
    x, y = np.meshgrid(
        np.linspace(0, 1, map_size, endpoint=False),
        np.linspace(0, 1, map_size, endpoint=False)
    )

    # Generate displacement maps for each noise function
    displacement_maps = []
    for noise_func, noise_params, weight in zip(noise_funcs, noise_params_list, blend_weights):
        sig = inspect.signature(noise_func)
        valid_params = {k: v for k, v in noise_params.items() if k in sig.parameters}
        try:
            noise_map = noise_func(x, y, **valid_params)
        except TypeError as te:
            logger.warning(f"Noise function '{noise_func.__name__}' is not vectorized. Applying np.vectorize. Error: {te}")
            vectorized_func = np.vectorize(
                lambda xi, yi: noise_func(float(xi), float(yi), **valid_params)
            )
            noise_map = vectorized_func(x, y)
        displacement_maps.append(weight * noise_map)

    # Combine displacement maps
    combined_map = sum(displacement_maps)

    # Normalize the combined map if required
    if normalize:
        min_val, max_val = np.min(combined_map), np.max(combined_map)
        if max_val == min_val:
            logger.warning(f"Combined noise output has no variation: min={min_val}, max={max_val}, params={noise_params_list} map_size={map_size} blend_weights={blend_weights} noise_funcs={noise_funcs}")
            combined_map = np.zeros_like(combined_map)
        else:
            combined_map = (combined_map - min_val) / (max_val - min_val)

    return combined_map.astype(np.float32)


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

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".tiff":
        img.save(filename, format="TIFF")
    else:
        img.save(filename, format="PNG")

    if not os.path.exists(filename):
        raise IOError(f"File not found after saving: {filename}")

    logger.debug(f"Saved displacement map to: {filename}")

def normal_to_displacement(normal_map, regularization=1e-5):
    """
    Convert a normal map to a displacement (height) map using a frequency-domain
    approach based on gradient field integration.
    
    This implementation uses the Fourier transform method to solve the Poisson equation,
    which reconstructs the height field from the partial derivatives (normal map).
    
    Parameters:
    -----------
    normal_map : np.ndarray
        RGB normal map as a HxWx3 array with values in [0,1].
        The RGB channels represent the XYZ components of the normal vectors.
    regularization : float
        Regularization parameter to prevent division by zero in frequency domain.
        Larger values create smoother results but may lose detail.
        
    Returns:
    --------
    np.ndarray
        Reconstructed displacement map as a 2D array with values normalized to [0,1].
    """
    if normal_map.ndim != 3 or normal_map.shape[2] != 3:
        raise ValueError("Input normal map must be an RGB image (HxWx3)")
    
    # Extract normal vector components and convert to [-1,1] range
    normals = normal_map * 2.0 - 1.0
    
    # Extract partial derivatives from normal map
    # Normal = (-dz/dx, -dz/dy, 1) / sqrt((-dz/dx)^2 + (-dz/dy)^2 + 1)
    # So: dz/dx = -Nx/Nz, dz/dy = -Ny/Nz
    dx = -normals[..., 0] / np.maximum(normals[..., 2], 1e-5)
    dy = -normals[..., 1] / np.maximum(normals[..., 2], 1e-5)
    
    # Get image dimensions
    height, width = normal_map.shape[:2]
    
    # Compute frequencies
    freq_x = fftpack.fftfreq(width)
    freq_y = fftpack.fftfreq(height)
    freq_grid_x, freq_grid_y = np.meshgrid(freq_x, freq_y)
    
    # Compute Fourier transforms of derivatives
    dx_f = fftpack.fft2(dx)
    dy_f = fftpack.fft2(dy)
    
    # Solve Poisson equation in frequency domain
    # The solution minimizes: (dz/dx - dx)^2 + (dz/dy - dy)^2
    # Leading to: (d^2/dx^2 + d^2/dy^2)z = d(dx)/dx + d(dy)/dy
    # In Fourier domain: -(wx^2 + wy^2)Z = iwx*Dx + iwy*Dy
    denom = 2.0 * np.pi * (freq_grid_x**2 + freq_grid_y**2) + regularization
    denom[0, 0] = 1.0  # Avoid division by zero at DC
    
    # Compute Z in frequency domain
    numerator = (2.0j * np.pi) * (freq_grid_x * dx_f + freq_grid_y * dy_f)
    Z_f = numerator / denom
    
    # Transform back to spatial domain
    displacement = np.real(fftpack.ifft2(Z_f))
    
    # Normalize to [0,1] range
    displacement -= np.min(displacement)
    displacement /= np.maximum(np.max(displacement), 1e-5)
    
    return displacement.astype(np.float32)

def load_normal_map(filename):
    """
    Load a normal map from an image file.
    
    Parameters:
    -----------
    filename : str
        Path to the normal map image.
        
    Returns:
    --------
    np.ndarray
        Normal map as a HxWx3 array with values in [0,1].
    """
    try:
        img = Image.open(filename)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img).astype(np.float32) / 255.0
    except Exception as e:
        logger.error(f"Failed to load normal map {filename}: {e}")
        raise
