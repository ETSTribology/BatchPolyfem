"""
noises.py

Collection of noise functions for generating 2D height/displacement values.
Suitable for creating displacement maps and other procedural textures.
"""

import math
import logging
from rich.console import Console
from rich.logging import RichHandler
from itertools import product

import numpy as np
import noise
from scipy.stats import qmc

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
# Helper Functions
# ------------------------------------------------------------------------

def vectorize_function(func):
    return np.vectorize(func)

def radical_inverse(index, base):
    """
    Compute the radical inverse of an index in a given base.

    Parameters:
    -----------
    index : int
        The index of the sequence.
    base  : int
        The base for the radical inverse.

    Returns:
    --------
    float
        Radical inverse of the index in the specified base.
    """
    inverse = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        inverse += f * (i % base)
        i = i // base
        f /= base
    return inverse


# ------------------------------------------------------------------------
# 2D/3D Spatial Noise Functions
# ------------------------------------------------------------------------
def sine_noise(x, y, frequency=10.0, amplitude=1.0):
    """
    Simple sine wave in the x-direction.

    Parameters:
    -----------
    x, y       : float or np.ndarray
        2D coordinates (y is unused in this function).
    frequency  : float
        Controls the number of wave cycles per unit in x.
    amplitude  : float
        Scales the resulting sine value.

    Returns:
    --------
    float or np.ndarray
        Noise value at (x, y).
    """
    return amplitude * np.sin(frequency * x)

def square_noise(x, y, frequency=10.0, amplitude=1.0, duty_cycle=0.5):
    """
    Vectorized Square wave with a specified duty cycle.

    Parameters:
    -----------
    x, y        : float or np.ndarray
        2D coordinates (y is unused in this function).
    frequency   : float
        Controls the wave frequency in x (cycles per unit).
    amplitude   : float
        Scales the resulting wave.
    duty_cycle  : float
        Fraction of the period where the wave is positive (0 < duty_cycle < 1).

    Returns:
    --------
    float or np.ndarray
        +amplitude during the positive phase, -amplitude otherwise.
    """
    if not (0.0 < duty_cycle < 1.0):
        raise ValueError("duty_cycle must be between 0 and 1.")

    phase = (frequency * x) % 1.0
    return np.where(phase < duty_cycle, amplitude, -amplitude)

def fbm_noise(x, y, z=0.0, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    Fractal Brownian Motion (fBm) using Perlin noise internally (pnoise3).

    Summation of several octaves of Perlin noise, each octave scaled in
    frequency and amplitude.

    Parameters:
    -----------
    x, y, z    : float or np.ndarray
        3D coordinates (z can be used as time or extra dimension).
    scale      : float
        Scales the input coordinates to control the frequency of the noise.
    octaves    : int
        Number of layers of noise to sum.
    persistence: float
        Amplitude reduction per octave.
    lacunarity : float
        Frequency increase per octave.

    Returns:
    --------
    float or np.ndarray
        fBm noise value at (x, y, z).
    """
    def single_fbm(_x, _y, _z):
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        for _ in range(octaves):
            value += amplitude * noise.pnoise3(_x * frequency * scale, 
                                               _y * frequency * scale, 
                                               _z * frequency * scale)
            amplitude *= persistence
            frequency *= lacunarity
        return value
    
    # Vectorize the single_fbm function
    vectorized_fbm = np.vectorize(single_fbm)
    return vectorized_fbm(x, y, z)

def fractal_noise(x, y, **kwargs):
    """
    Alias for fbm_noise(), if you prefer a different function name.
    """
    return fbm_noise(x, y, **kwargs)

def worley_noise(x, y, z=0.0, scale=1.0, num_features=10):
    """
    Worley (Cellular) noise based on distance to nearest feature point.

    Parameters:
    -----------
    x, y, z    : float
        3D coordinates.
    scale      : float
        Scales the input coordinates.
    num_features: int
        Number of feature points to consider.

    Returns:
    --------
    float
        Worley noise value at (x, y, z).
    """
    min_dist = float('inf')
    for _ in range(num_features):
        # Randomly place feature points within the grid cell
        fx = noise.pnoise3(x * scale, y * scale, z * scale, repeatx=1024, repeaty=1024, repeatz=1024)
        fy = noise.pnoise3((x * scale) + 100, (y * scale) + 100, (z * scale) + 100, repeatx=1024, repeaty=1024, repeatz=1024)
        fz = noise.pnoise3((x * scale) + 200, (y * scale) + 200, (z * scale) + 200, repeatx=1024, repeaty=1024, repeatz=1024)
        # Scale feature points to [0, 1]
        fx = (fx + 1) / 2.0
        fy = (fy + 1) / 2.0
        fz = (fz + 1) / 2.0
        # Compute distance to feature point
        dist = math.sqrt((x * scale - fx) ** 2 + (y * scale - fy) ** 2 + (z * scale - fz) ** 2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def wavelet_noise(x, y, z=0.0, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    Wavelet-based noise using Perlin noise and wavelet transformations.

    Parameters:
    -----------
    x, y, z    : float
        3D coordinates.
    scale      : float
        Scales the input coordinates.
    octaves    : int
        Number of noise layers.
    persistence: float
        Amplitude reduction per octave.
    lacunarity : float
        Frequency increase per octave.

    Returns:
    --------
    float
        Wavelet-based noise value at (x, y, z).
    """
    sum = 0.0
    frequency = 1.0
    amplitude = 1.0

    for _ in range(octaves):
        n = noise.pnoise3(x * frequency * scale, y * frequency * scale, z * frequency * scale)
        # Apply a wavelet transformation (e.g., Haar wavelet)
        n = abs(n)
        sum += amplitude * n
        frequency *= lacunarity
        amplitude *= persistence

    return sum

def white_noise(x, y, z=0.0, scale=1.0, seed=0):
    """
    White noise with no correlation between values.

    Parameters:
    -----------
    x, y, z    : float
        3D coordinates.
    scale      : float
        Scales the input coordinates.
    seed       : int
        Seed for random number generation.

    Returns:
    --------
    float
        White noise value at (x, y, z).
    """
    rng = np.random.default_rng(seed + int(x * scale) * 10000 + int(y * scale) * 10000 + int(z * scale))
    return rng.uniform(-1.0, 1.0)


import numpy as np
import noise

def domain_warp_noise(x, y, z=0.0, scale=1.0, warp_scale=0.5, octaves=2, persistence=0.5, lacunarity=2.0):
    """
    Vectorized domain warping using multiple layers of Perlin noise.
    
    Parameters:
    -----------
    x, y, z      : array-like or float
        3D coordinates. Can be scalars or NumPy arrays.
    scale        : float
        Scales the input coordinates.
    warp_scale   : float
        Scales the warping noise.
    octaves      : int
        Number of noise layers for warping.
    persistence  : float
        Amplitude reduction per octave for warping.
    lacunarity   : float
        Frequency increase per octave for warping.

    Returns:
    --------
    np.ndarray or float
        Domain-warped noise value at (x, y, z).
    """
    # Vectorized wrapper for noise.pnoise3
    pnoise3_vec = np.vectorize(
        lambda x, y, z: noise.pnoise3(
            x, y, z, octaves=octaves, persistence=persistence, lacunarity=lacunarity
        )
    )
    
    # Apply warp
    warp_x = pnoise3_vec(x * warp_scale * scale, y * warp_scale * scale, z * warp_scale * scale)
    warp_y = pnoise3_vec(x * warp_scale * scale + 100, y * warp_scale * scale + 100, z * warp_scale * scale + 100)
    
    # Final domain-warped noise
    return pnoise3_vec((x + warp_x) * scale, (y + warp_y) * scale, z * scale)

def diamond_square_noise(x, y, z=0.0, scale=1.0, size=256, roughness=0.5):
    """
    Diamond-Square noise for generating heightmaps.

    Parameters:
    -----------
    x, y, z      : float
        3D coordinates (z can be used for multiple heightmaps).
    scale        : float
        Scales the input coordinates.
    size         : int
        Size of the grid (must be 2^n + 1).
    roughness    : float
        Controls the roughness of the terrain.

    Returns:
    --------
    float
        Height value at (x, y, z).
    """
    # Initialize grid
    grid_size = size
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Seed corners
    grid[0, 0] = np.random.uniform(-1, 1)
    grid[0, -1] = np.random.uniform(-1, 1)
    grid[-1, 0] = np.random.uniform(-1, 1)
    grid[-1, -1] = np.random.uniform(-1, 1)
    
    step_size = grid_size - 1
    scale_factor = roughness
    
    while step_size > 1:
        half_step = step_size // 2
        
        # Diamond step
        for i in range(half_step, grid_size - 1, step_size):
            for j in range(half_step, grid_size - 1, step_size):
                avg = (grid[i - half_step, j - half_step] +
                       grid[i - half_step, j + half_step] +
                       grid[i + half_step, j - half_step] +
                       grid[i + half_step, j + half_step]) / 4.0
                grid[i, j] = avg + np.random.uniform(-scale_factor, scale_factor)
        
        # Square step
        for i in range(0, grid_size, half_step):
            for j in range((i + half_step) % step_size, grid_size, step_size):
                s = []
                if i - half_step >= 0:
                    s.append(grid[i - half_step, j])
                if i + half_step < grid_size:
                    s.append(grid[i + half_step, j])
                if j - half_step >= 0:
                    s.append(grid[i, j - half_step])
                if j + half_step < grid_size:
                    s.append(grid[i, j + half_step])
                avg = np.mean(s)
                grid[i, j] = avg + np.random.uniform(-scale_factor, scale_factor)
        
        step_size = half_step
        scale_factor *= roughness
    
    # Normalize grid to [-1, 1]
    grid = (grid - grid.min()) / (grid.max() - grid.min()) * 2.0 - 1.0
    
    # Interpolate the value at (x, y)
    xi = x * (grid_size - 1)
    yi = y * (grid_size - 1)
    x0 = int(math.floor(xi))
    y0 = int(math.floor(yi))
    x1 = min(x0 + 1, grid_size - 1)
    y1 = min(y0 + 1, grid_size - 1)
    
    dx = xi - x0
    dy = yi - y0
    
    value = (grid[x0, y0] * (1 - dx) * (1 - dy) +
             grid[x1, y0] * dx * (1 - dy) +
             grid[x0, y1] * (1 - dx) * dy +
             grid[x1, y1] * dx * dy)
    
    return value


def turbulence_noise(x, y, z=0.0, scale=1.0, octaves=5, persistence=0.5, lacunarity=2.0):
    """
    Turbulence noise by summing absolute values of Perlin noise layers.

    Parameters:
    -----------
    x, y, z      : float
        3D coordinates.
    scale        : float
        Scales the input coordinates.
    octaves      : int
        Number of noise layers.
    persistence  : float
        Amplitude reduction per octave.
    lacunarity   : float
        Frequency increase per octave.

    Returns:
    --------
    float
        Turbulence noise value at (x, y, z).
    """
    sum = 0.0
    frequency = 1.0
    amplitude = 1.0

    for _ in range(octaves):
        n = noise.pnoise3(x * frequency * scale, y * frequency * scale, z * frequency * scale)
        sum += abs(n) * amplitude
        frequency *= lacunarity
        amplitude *= persistence

    return sum


def ridged_multifractal_noise(x, y, z=0.0, scale=1.0, octaves=6, lacunarity=2.0, gain=0.5):
    """
    Ridged multifractal noise using Perlin noise.

    Parameters:
    -----------
    x, y, z    : float
        3D coordinates.
    scale      : float
        Scales the input coordinates.
    octaves    : int
        Number of noise layers.
    lacunarity : float
        Frequency increase per octave.
    gain       : float
        Controls the sharpness of ridges.

    Returns:
    --------
    float
        Ridged multifractal noise value at (x, y, z).
    """
    sum = 0.0
    frequency = 1.0
    amplitude = 1.0
    weight = 1.0

    for _ in range(octaves):
        n = noise.pnoise3(x * frequency * scale, y * frequency * scale, z * frequency * scale)
        n = 1.0 - abs(n)  # Invert to create ridges
        n *= n  # Square to increase contrast
        n *= weight
        sum += n * amplitude

        weight = n * gain
        weight = max(min(weight, 1.0), 0.0)  # Clamp between 0 and 1

        frequency *= lacunarity
        amplitude *= gain

    return sum

def perlin_noise(x, y, z=0.0, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
    """
    Basic Perlin noise (from 'noise' library) with support for multiple octaves.

    Parameters:
    -----------
    x, y, z      : float or np.ndarray
        Input coordinates for noise generation.
    scale        : float
        Scale of the noise.
    octaves      : int
        Number of octaves for fractal noise.
    persistence  : float
        Amplitude reduction factor for each octave.
    lacunarity   : float
        Frequency increase factor for each octave.

    Returns:
    --------
    float or np.ndarray
        Perlin noise value(s) at the given coordinates.
    """
    # Validate parameters
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError(f"scale must be a positive float, got {scale}")
    if not isinstance(octaves, int) or octaves <= 0:
        raise ValueError(f"octaves must be a positive integer, got {octaves}")
    if not isinstance(persistence, (int, float)) or not (0 < persistence < 1):
        raise ValueError(f"persistence must be between 0 and 1, got {persistence}")
    if not isinstance(lacunarity, (int, float)) or lacunarity <= 0:
        raise ValueError(f"lacunarity must be a positive float, got {lacunarity}")

    # Vectorize the noise function to handle array inputs
    vectorized_noise = np.vectorize(
        lambda xi, yi, zi: noise.pnoise3(
            xi * scale,
            yi * scale,
            zi * scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
    )

    # Apply vectorized function
    return vectorized_noise(x, y, z)

def gabor_noise(x, y, frequency=5.0, theta=0.0, sigma_x=1.0, sigma_y=1.0, offset=0.0):
    """
    Gabor-like noise: a Gaussian envelope multiplied by a sinusoid.

    Parameters:
    -----------
    x, y      : float
        2D coordinates.
    frequency : float
        Frequency of the sinusoid.
    theta     : float
        Orientation of the sinusoid in degrees.
    sigma_x   : float
        Standard deviation of the Gaussian envelope in x.
    sigma_y   : float
        Standard deviation of the Gaussian envelope in y.
    offset    : float
        Phase offset of the sinusoid.

    Returns:
    --------
    float
        Gabor-like noise at (x, y).
    """
    theta_rad = math.radians(theta)
    x_rot = x * math.cos(theta_rad) + y * math.sin(theta_rad)
    gauss = math.exp(-((x_rot**2) / (2.0 * sigma_x**2) + (y**2) / (2.0 * sigma_y**2)))
    wave = math.cos(2.0 * math.pi * frequency * x_rot + offset)
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
# Additional Noise Functions
# ------------------------------------------------------------------------

def random_walk_noise(x, y, scale=1.0, step_size=1.0, seed=0):
    """
    Simulated Random Walk Noise.

    Generates noise resembling a random walk by accumulating pseudo-random steps
    based on the grid position.

    Parameters:
    -----------
    x, y        : float
        2D coordinates.
    scale       : float
        Scales input coordinates to control the frequency of the noise.
    step_size   : float
        Size of each random step.
    seed        : int
        Seed for the random number generator to ensure reproducibility.

    Returns:
    --------
    float
        Random walk noise value at (x, y).
    """
    if not isinstance(step_size, (int, float)):
        raise ValueError(f"step_size must be a non-negative float, got {type(step_size)}")
    if step_size < 0:
        raise ValueError("step_size must be non-negative")

    rng = np.random.default_rng(seed + int(x * scale) * 10000 + int(y * scale))
    steps = int(math.hypot(x, y) * scale)
    if steps == 0:
        return 0.0
    directions = rng.uniform(0, 2 * math.pi, steps)
    displacement = np.sum(step_size * (np.cos(directions) + np.sin(directions)))
    return displacement


def ornstein_uhlenbeck_noise(x, y, theta=0.15, mu=0.0, sigma=0.3, scale=1.0):
    """
    Ornstein-Uhlenbeck Noise.

    Generates mean-reverting noise based on the Ornstein-Uhlenbeck process.

    Parameters:
    -----------
    x, y     : float
        2D coordinates.
    theta    : float
        Speed of mean reversion.
    mu       : float
        Long-term mean value.
    sigma    : float
        Volatility parameter.
    scale    : float
        Scales input coordinates to control the frequency of the noise.

    Returns:
    --------
    float
        Ornstein-Uhlenbeck noise value at (x, y).
    """
    # Use Perlin noise as the driving noise
    driving_noise = noise.pnoise3(x * scale, y * scale, 0.0)
    # Mean-reverting equation: dX = theta*(mu - X)*dt + sigma*dW
    # Discretized for a grid
    # Since we don't have state, approximate with a damped noise
    return theta * (mu - driving_noise) + sigma * driving_noise


def vasicek_noise(x, y, alpha=0.1, beta=0.1, sigma=0.1, scale=1.0):
    """
    Vasicek Noise.

    Generates noise based on the Vasicek model, commonly used in financial applications
    for interest rate modeling. It is similar to the Ornstein-Uhlenbeck process.

    Parameters:
    -----------
    x, y     : float
        2D coordinates.
    alpha    : float
        Speed of mean reversion.
    beta     : float
        Long-term mean.
    sigma    : float
        Volatility parameter.
    scale    : float
        Scales input coordinates to control the frequency of the noise.

    Returns:
    --------
    float
        Vasicek noise value at (x, y).
    """
    # Use Perlin noise as the driving noise
    driving_noise = noise.pnoise3(x * scale, y * scale, 0.0)
    # Mean-reverting equation: dr = alpha*(beta - r)*dt + sigma*dW
    # Discretized for a grid
    # Approximate with a damped noise similar to Ornstein-Uhlenbeck
    return alpha * (beta - driving_noise) + sigma * driving_noise


def blue_noise(x, y, scale=1.0, seed=0):
    """
    Blue Noise.

    Generates noise with minimal low-frequency components and maximized high-frequency content,
    resembling blue noise patterns. This is an approximation suitable for procedural textures.

    Parameters:
    -----------
    x, y     : float
        2D coordinates.
    scale    : float
        Scales input coordinates to control the frequency of the noise.
    seed     : int
        Seed for the random number generator to ensure reproducibility.

    Returns:
    --------
    float
        Blue noise value at (x, y).
    """
    # High-pass filter approximation using Perlin noise
    low_freq = noise.pnoise3(x * scale * 0.5, y * scale * 0.5, 0.0)
    high_freq = noise.pnoise3(x * scale * 2.0, y * scale * 2.0, 0.0)
    return high_freq - low_freq


def halton_noise(x, y, base1=2, base2=3):
    """
    Generate noise based on the Halton sequence.

    Parameters:
    -----------
    x, y    : float
        2D coordinates, typically integer indices.
    base1   : int
        Base for the first dimension (default: 2).
    base2   : int
        Base for the second dimension (default: 3).

    Returns:
    --------
    float
        Halton sequence-based noise value at (x, y).
    """
    index = int(x) + int(y) * 10000  # Example index mapping
    seq1 = radical_inverse(index, base1)
    seq2 = radical_inverse(index, base2)
    # Combine the two dimensions, e.g., average or sum
    return (seq1 + seq2) / 2.0


def hammersley_noise(x, y, base=2, total_samples=100000):
    """
    Generate noise based on the Hammersley sequence.

    Parameters:
    -----------
    x, y          : float
        2D coordinates, typically integer indices.
    base          : int
        Base for the second dimension (default: 2).
    total_samples : int
        Total number of samples in the sequence (default: 100,000).

    Returns:
    --------
    float
        Hammersley sequence-based noise value at (x, y).
    """
    index = int(x) + int(y) * 10000  # Example index mapping
    seq1 = index / total_samples  # Normalized index
    seq2 = radical_inverse(index, base)
    return (seq1 + seq2) / 2.0


def sobol_noise(x, y, scramble=False, seed=None):
    """
    Generate noise based on the Sobol sequence.

    Parameters:
    -----------
    x, y     : float
        2D coordinates, typically integer indices.
    scramble : bool
        Whether to scramble the sequence for better randomness (default: False).
    seed     : int or None
        Seed for scrambling (default: None).

    Returns:
    --------
    float
        Sobol sequence-based noise value at (x, y).
    """
    sampler = qmc.Sobol(d=2, scramble=scramble, seed=seed)
    index = int(x) + int(y) * 10000  # Example index mapping

    # Since scipy's Sobol does not support jumping to an arbitrary index directly,
    # we generate the required number of samples up to the desired index.
    # Note: This can be inefficient for large indices.

    try:
        # Determine the number of bits needed to represent the index
        bits = int(math.ceil(math.log2(index + 1))) if index > 0 else 1
        m = bits
        sampler.reset()
        samples = sampler.random_base2(m=m)
        if index < len(samples):
            sample = samples[index]
        else:
            # Generate additional samples as needed
            additional_bits = int(math.ceil(math.log2(index + 1 - len(samples))))
            additional_samples = sampler.random_base2(m=additional_bits)
            sample = additional_samples[-1]
    except Exception as e:
        logger.error(f"Sobol sequence generation failed at index {index}: {e}")
        return 0.0

    # Combine the two dimensions, e.g., average or sum
    return np.sum(sample) / len(sample)


def cellular_noise(x, y, z=0.0, scale=1.0, jitter=0.5, mode="F1"):
    """
    Cellular (Worley) noise.

    Parameters:
    -----------
    x, y, z : float or np.ndarray
        3D coordinates.
    scale   : float
        Scale of the grid to control the size of cells.
    jitter  : float
        Jitter factor for feature points within the cells (0.0 to 1.0).
    mode    : str
        Determines the type of noise value to return:
        - "F1": Closest distance (default)
        - "F2": Second closest distance
        - "F2 - F1": Difference between second and first closest distances.

    Returns:
    --------
    float or np.ndarray
        Cellular noise value at (x, y, z).
    """
    x = np.asarray(x) * scale
    y = np.asarray(y) * scale
    z = np.asarray(z) * scale

    # Grid coordinates
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    z0 = np.floor(z).astype(int)

    # Initialize distances
    min_distances = [float("inf"), float("inf")]  # F1 and F2 distances

    for dx, dy, dz in product([-1, 0, 1], repeat=3):
        # Neighbor cell coordinates
        neighbor_x = x0 + dx
        neighbor_y = y0 + dy
        neighbor_z = z0 + dz

        # Random feature point within the cell
        rng = np.random.default_rng(hash((neighbor_x, neighbor_y, neighbor_z)) & 0xFFFFFFFF)
        feature_x = neighbor_x + rng.uniform(-jitter, jitter)
        feature_y = neighbor_y + rng.uniform(-jitter, jitter)
        feature_z = neighbor_z + rng.uniform(-jitter, jitter)

        # Compute distance to the feature point
        dist = np.sqrt((x - feature_x) ** 2 + (y - feature_y) ** 2 + (z - feature_z) ** 2)

        # Update F1 and F2 distances
        if dist < min_distances[0]:
            min_distances[1] = min_distances[0]
            min_distances[0] = dist
        elif dist < min_distances[1]:
            min_distances[1] = dist

    # Return the noise value based on the mode
    if mode == "F1":
        return min_distances[0]
    elif mode == "F2":
        return min_distances[1]
    elif mode == "F2 - F1":
        return min_distances[1] - min_distances[0]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def koch_curve(iterations, length=1.0):
    """
    Generate a Koch curve fractal.

    Parameters:
    -----------
    iterations : int
        Number of iterations to apply the fractal rule.
    length     : float
        Length of the initial line segment.

    Returns:
    --------
    list of tuple
        List of points (x, y) representing the Koch curve.
    """
    def divide_segment(p1, p2):
        """Divide a segment into four parts as per Koch curve rules."""
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        # Points dividing the line into thirds
        pA = (x1 + dx / 3, y1 + dy / 3)
        pB = (x1 + 2 * dx / 3, y1 + 2 * dy / 3)

        # Point forming the peak of the equilateral triangle
        angle = math.atan2(dy, dx) - math.pi / 3
        length = math.sqrt(dx ** 2 + dy ** 2) / 3
        pC = (pA[0] + length * math.cos(angle), pA[1] + length * math.sin(angle))

        return [p1, pA, pC, pB, p2]

    # Initialize with a single line segment
    points = [(0, 0), (length, 0)]

    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            new_points.extend(divide_segment(points[i], points[i + 1])[:-1])
        new_points.append(points[-1])
        points = new_points

    return points


# Define noise functions mapping with their parameters
noise_variations = {
    "sine": [
        (sine_noise, {"amplitude": amp, "frequency": freq})
        for amp, freq in product(
            [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0],
            [15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 1000.0]
        )
    ],
    "square": [
    (square_noise, {"amplitude": amp, "frequency": freq, "duty_cycle": duty})
        for amp, freq, duty in product(
            [2.0, 3.0, 5.0, 10.0, 50.0, 100.0],  # Amplitude: Controls the height of the wave
            [1.0, 2.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0],  # Frequency: Controls the number of cycles per unit
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]  # Duty cycle: Fraction of the period the signal is high
        )
    ],
    "perlin": [
        (perlin_noise, {"scale": scale, "octaves": octaves, "persistence": persistence, "lacunarity": lacunarity})
        for scale, octaves, persistence, lacunarity in product(
            [0.5, 0.8, 1.0, 1.5, 2.0],  # Scale
            [2, 3, 4, 5],  # Octaves
            [0.3, 0.5, 0.6],  # Persistence
            [1.5, 2.0, 2.5]  # Lacunarity
        )
    ],
    "fbm": [
        (fbm_noise, {"scale": scale, "octaves": octaves, "persistence": persistence, "lacunarity": lacunarity})
        for scale, octaves, persistence, lacunarity in product(
            [0.5, 0.8, 1.0, 1.2, 1.5],  # Scale
            [3, 4, 5, 6],  # Octaves
            [0.3, 0.5, 0.6],  # Persistence
            [1.5, 2.0, 2.5]  # Lacunarity
        )
    ],
    "gabor": [
        (gabor_noise, {"frequency": freq, "theta": theta, "sigma_x": sigma_x, "sigma_y": sigma_y, "offset": offset})
        for freq, theta, sigma_x, sigma_y, offset in product(
            [3.0, 5.0, 7.0, 10.0],  # Frequency
            [15, 30, 45, 60, 90],  # Theta
            [0.5, 1.0, 1.5],  # Sigma X
            [0.5, 1.0, 1.5],  # Sigma Y
            [0.0, 0.5, 1.0]  # Offset
        )
    ],
    "random_walk": [
        (random_walk_noise, {"scale": scale, "step_size": step, "seed": seed})
        for scale, step, seed in product(
            [0.5, 1.0, 1.5],  # Scale
            [0.05, 0.1, 0.2],  # Step size
            [7, 24, 42]  # Seed
        )
    ],
    "ornstein_uhlenbeck": [
        (ornstein_uhlenbeck_noise, {"theta": theta, "mu": mu, "sigma": sigma, "scale": scale})
        for theta, mu, sigma, scale in product(
            [0.1, 0.15, 0.2],  # Theta
            [-0.1, 0.0, 0.1],  # Mu
            [0.25, 0.3, 0.35],  # Sigma
            [0.8, 1.0, 1.2]  # Scale
        )
    ],
    "vasicek": [
        (vasicek_noise, {"alpha": alpha, "beta": beta, "sigma": sigma, "scale": scale})
        for alpha, beta, sigma, scale in product(
            [0.1, 0.15, 0.2],  # Alpha
            [0.05, 0.1, 0.15],  # Beta
            [0.09, 0.1, 0.11],  # Sigma
            [0.9, 1.0, 1.1]  # Scale
        )
    ],
    "blue_noise": [
        (blue_noise, {"scale": scale, "seed": seed})
        for scale, seed in product(
            [10, 15, 20],  # Scale
            [0, 1, 2]  # Seed
        )
    ],
    "halton": [
        (halton_noise, {"base1": base1, "base2": base2})
        for base1, base2 in product([2, 3, 5], [3, 5, 7])
    ],
    "wavelet": [
        (wavelet_noise, {"scale": scale, "octaves": octaves, "persistence": persistence, "lacunarity": lacunarity})
        for scale, octaves, persistence, lacunarity in product(
            [0.7, 1.0, 1.3],  # Scale
            [3, 4, 5],  # Octaves
            [0.4, 0.5, 0.6],  # Persistence
            [1.5, 2.0, 2.5]  # Lacunarity
        )
    ],
    "domain_warp": [
        (domain_warp_noise, {"scale": scale, "warp_scale": warp_scale, "octaves": octaves, "persistence": persistence, "lacunarity": lacunarity})
        for scale, warp_scale, octaves, persistence, lacunarity in product(
            [0.8, 1.0, 1.2],  # Scale
            [0.3, 0.5, 0.7],  # Warp scale
            [2, 3, 4],  # Octaves
            [0.4, 0.5, 0.6],  # Persistence
            [1.8, 2.0, 2.5]  # Lacunarity
        )
    ],
    "cellular" : [
        (cellular_noise, {"scale": scale, "jitter": jitter, "mode": mode})
        for scale, jitter, mode in product(
            [0.5, 1.0, 1.5],  # Scale
            [0.3, 0.5, 0.7],  # Jitter
            ["F1", "F2", "F2 - F1"]  # Mode
        )
    ],
    "koch_curve" : [
        (koch_curve, {"iterations": iterations, "length": length})
        for iterations, length in product(
            [1, 2, 3, 4, 5],  # Iterations
            [0.5, 1.0, 1.5, 2.0]  # Length
        )
    ],
}
