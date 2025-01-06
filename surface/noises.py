"""
noises.py

Collection of noise functions for generating 2D height/displacement values.
Suitable for creating displacement maps and other procedural textures.
"""

import math
import logging
from rich.console import Console
from rich.logging import RichHandler

import numpy as np
import noise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
console = Console()


# ------------------------------------------------------------------------
# 2D/3D Spatial Noise Functions
# ------------------------------------------------------------------------

def sine_noise(x, y, frequency=10.0, amplitude=1.0):
    """
    Simple sine wave in the x-direction.

    Parameters:
    -----------
    x, y       : float
        2D coordinates (y is unused in this function).
    frequency  : float
        Controls the number of wave cycles per unit in x.
    amplitude  : float
        Scales the resulting sine value.

    Returns:
    --------
    float
        Noise value at (x, y).
    """
    return amplitude * math.sin(frequency * x)

def square_noise(x, y, frequency=10.0, amplitude=1.0):
    """
    Simple square wave derived from a sine wave sign.

    Parameters:
    -----------
    x, y       : float
        2D coordinates (y is unused).
    frequency  : float
        Controls the wave frequency in x.
    amplitude  : float
        Scales the resulting wave.

    Returns:
    --------
    float
        +amplitude or -amplitude at each (x, y).
    """
    return amplitude * np.sign(math.sin(frequency * x))

def fbm_noise(x, y, z=0.0, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    Fractal Brownian Motion (fBm) using Perlin noise internally (pnoise3).

    Summation of several octaves of Perlin noise, each octave scaled in
    frequency and amplitude.

    Parameters:
    -----------
    x, y, z    : float
        3D coordinates (z can be used as time or extra dimension).
    octaves    : int
        Number of layers of noise to sum.
    persistence: float
        Amplitude reduction per octave.
    lacunarity : float
        Frequency increase per octave.

    Returns:
    --------
    float
        fBm noise value at (x, y, z).
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    for _ in range(octaves):
        value += amplitude * noise.pnoise3(x * frequency, y * frequency, z * frequency)
        amplitude *= persistence
        frequency *= lacunarity
    return value

def fractal_noise(x, y, **kwargs):
    """
    Alias for fbm_noise(), if you prefer a different function name.
    """
    return fbm_noise(x, y, **kwargs)

def perlin_noise(x, y, z=0.0, scale=1.0):
    """
    Basic Perlin noise (from 'noise' library).

    Parameters:
    -----------
    x, y, z : float
        3D coordinates.
    scale   : float
        Scales input coordinates for different zoom levels.

    Returns:
    --------
    float
        Perlin noise value at (x, y, z).
    """
    return noise.pnoise3(x * scale, y * scale, z * scale)

def gabor_noise(x, y, frequency=5.0, sigma=0.2):
    """
    Simplified Gabor-like noise: a Gaussian envelope multiplied by a sinusoid.

    Parameters:
    -----------
    x, y      : float
        2D coordinates.
    frequency : float
        Frequency of the sinusoid.
    sigma     : float
        Std. dev. of the Gaussian envelope.

    Returns:
    --------
    float
        Gabor-like noise at (x, y).
    """
    gauss = math.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    wave = math.cos(2.0 * math.pi * frequency * x)
    return gauss * wave

def simple_3d_noise(x, y, t=0.0, scale=0.5):
    """
    3D Perlin noise for (x, y, t).

    Parameters:
    -----------
    x, y : float
        2D coordinates.
    t    : float
        Additional dimension (e.g., time).
    scale: float
        Input scaling factor.

    Returns:
    --------
    float
        3D Perlin noise value at (x, y, t).
    """
    return noise.pnoise3(x * scale, y * scale, t)

def mandelbrot_noise(x, y, L=10.0, D_f=1.5, gamma=1.2, M=10, N_max=10, rng=None):
    """
    Ausloos-Berman Weierstrass-Mandelbrot fractal noise function.

    Summation across n=1..N_max and m=1..M, each with random phases.

    Parameters:
    -----------
    x, y   : float
        2D coordinates to evaluate the fractal.
    L      : float
        Scale factor for x, y input.
    D_f    : float
        Fractal dimension controlling amplitude scaling.
    gamma  : float
        Frequency scaling factor for each octave.
    M      : int
        Number of angular divisions.
    N_max  : int
        Max frequency index.
    rng    : np.random.Generator or None
        RNG for stable random phases. If None, uses default_rng().

    Returns:
    --------
    float
        Mandelbrot-like fractal noise value at (x, y).
    """
    if rng is None:
        rng = np.random.default_rng()

    log_gamma = math.log(gamma)
    scale_factor = (L ** (4.0 - 2.0 * D_f)) * log_gamma
    noise_value = 0.0

    for n in range(1, N_max + 1):
        freq_scale = gamma ** (n - 1)
        amp_scale = freq_scale ** (2.0 * (D_f - 2.0))
        # random phases for each angular division m
        phi_mn = rng.uniform(0.0, 2.0 * math.pi, M)

        for m in range(1, M + 1):
            cos_part = math.cos(math.pi * m / M)
            phase = phi_mn[m - 1]

            arg_x = (2.0 * math.pi * freq_scale * (x / L)) - cos_part + phase
            arg_y = (2.0 * math.pi * freq_scale * (y / L)) - cos_part + phase

            noise_value += amp_scale * (math.cos(arg_x) + math.cos(arg_y))

    return scale_factor * noise_value


# ------------------------------------------------------------------------
# Microfacet Distribution Functions (for advanced 2D patterns)
# ------------------------------------------------------------------------

def beckmann_noise_alt(x, y, alpha=0.5):  # Renamed function
    """
    Beckmann microfacet distribution function (D-term).

    Interprets (x, y) as slopes in a local tangent space.
    """
    r2 = x**2 + y**2
    if r2 < 1e-12:
        return 1.0  # avoid dividing by zero near the center

    tan2_theta = r2 / (alpha**2)
    cos2_theta = 1.0 / (1.0 + tan2_theta)
    cos4_theta = cos2_theta * cos2_theta

    # D ~ exp(-tan^2(θ) / alpha^2) / (π α^2 cos^4(θ))
    return math.exp(-tan2_theta) / (math.pi * alpha**2 * cos4_theta)

def ggx_noise_alt(x, y, alpha=0.5):  # Renamed function
    """
    GGX (Trowbridge-Reitz) microfacet distribution function (D-term).

    Interprets (x, y) as slopes in local tangent space.
    """
    r2 = x**2 + y**2
    if r2 < 1e-12:
        return 1.0

    tan2_theta = r2 / (alpha**2)
    cos2_theta = 1.0 / (1.0 + tan2_theta)
    cos4_theta = cos2_theta * cos2_theta

    denom = (alpha**2 + r2)**2
    return (alpha**2) / (math.pi * cos4_theta * denom)


# ------------------------------------------------------------------------
# Alternate Beckmann/GGX Versions
# ------------------------------------------------------------------------

def beckmann_noise_alt(mx, my, alpha=0.5):
    """
    Alternate version of Beckmann microfacet distribution D(m).
    """
    r2 = mx*mx + my*my
    cos2theta = 1.0 / (1.0 + r2)
    if cos2theta <= 0:
        return 0.0
    cos4theta = cos2theta * cos2theta
    exponent = -r2 / (alpha*alpha)
    return math.exp(exponent) / (math.pi * alpha*alpha * cos4theta)

def ggx_noise_alt(mx, my, alpha=0.5):
    """
    Alternate version of GGX microfacet distribution D(m).
    """
    r2 = mx*mx + my*my
    cos2theta = 1.0 / (1.0 + r2)
    if cos2theta <= 0:
        return 0.0
    cos4theta = cos2theta * cos2theta
    denom = (alpha*alpha + r2)
    return (alpha*alpha) / (math.pi * cos4theta * denom*denom)


# ------------------------------------------------------------------------
# Usage Example (Generate a small 2D array):
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demo of generating a 2D displacement map with Perlin noise
    map_size = 16
    displacement_map = np.zeros((map_size, map_size))

    for i in range(map_size):
        for j in range(map_size):
            # Convert (i, j) into normalized coordinates for demonstration
            x = i / map_size * 2.0 - 1.0
            y = j / map_size * 2.0 - 1.0
            displacement_map[i, j] = perlin_noise(x, y, scale=2.0)

    logger.info("Sample 2D Perlin noise map (16x16):")
    logger.info(displacement_map)
