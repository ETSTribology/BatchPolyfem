#!/usr/bin/env python3
"""
generate.py

Demonstration script that:
1) Generates 2D displacement maps from various noise functions.
2) Converts displacement PNG -> STL using 'hmm'.
3) Runs fTetWild on that STL to produce a tetrahedral mesh (.msh).
4) Uploads all results to MinIO.

Now enhanced to STOP immediately if any error occurs.
"""

import os
import tempfile
import logging
import subprocess
import typer

from rich.console import Console
from rich.logging import RichHandler

from minio import Minio, S3Error

# Local imports (assuming these files exist in the same directory)
from displacement import generate_displacement_map, save_displacement_map
from noises import (
    sine_noise, square_noise, perlin_noise,
    fbm_noise, fractal_noise, gabor_noise,
)
from hmm import run_hmm
from tetrahedron import run_ftetwild
from storage import setup_minio_client

###############################################################################
# Configure Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
console = Console()

###############################################################################
# Typer CLI
###############################################################################
app = typer.Typer(help="CLI to generate 2D displacement -> STL -> Tetrahedral Mesh, then upload to MinIO.")


###############################################################################
# Main Steps
###############################################################################
def generate_and_process(
    noise_name: str,
    noise_func,
    map_size: int,
    client: Minio
):
    """
    1) Generate a displacement map (PNG),
    2) Convert to STL (via hmm),
    3) Run fTetWild for .msh,
    4) Upload to MinIO in different buckets.

    If any step fails, raise typer.Exit(1) to stop the entire script.
    """
    logger.info(f"=== Processing noise: {noise_name}, size={map_size} ===")

    # Create a temporary folder for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Generate displacement map
        try:
            disp = generate_displacement_map(
                noise_func=noise_func,
                map_size=map_size,
                normalize=True
            )
            png_path = os.path.join(tmpdir, f"{noise_name}_{map_size}.png")
            save_displacement_map(disp, png_path)
            logger.info(f"Generated PNG: {png_path}")
        except Exception as e:
            logger.error(f"Failed generating displacement for {noise_name}_{map_size}: {e}")
            raise typer.Exit(1)

        # Step 2: Convert PNG -> STL via hmm
        stl_path = os.path.join(tmpdir, f"{noise_name}_{map_size}.stl")
        try:
            run_hmm(
                input_file=png_path,
                output_file=stl_path,
                error="0.001",
                triangles=None,
                quiet=True
            )
            logger.info(f"Generated STL: {stl_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"hmm failed for {png_path}: {e}")
            raise typer.Exit(1)

        # Step 3: Tetrahedral mesh via fTetWild
        tet_dir = tempfile.mkdtemp(prefix="ftetwild_", dir=tmpdir)
        msh_path = os.path.join(tet_dir, f"{noise_name}_{map_size}.msh")
        try:
            logger.info(f"Running fTetWild on STL -> MSH: {msh_path}")
            run_ftetwild(
                input_mesh=stl_path,
                ideal_edge_length=0.02,
                epsilon=0.0001,
                stop_energy=10.0,
                max_iterations=30,  # shorter run for demo
                docker=False,
                upload_to_minio=True
            )
            if not os.path.exists(msh_path):
                logger.warning("Could not find the .msh output; ensure run_ftetwild is adapted.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"fTetWild failed for {stl_path}: {e}")
            raise typer.Exit(1)
        except Exception as e:
            logger.error(f"fTetWild encountered an error: {e}")
            raise typer.Exit(1)

        # Step 4: Upload results to MinIO
        png_object_name = os.path.basename(png_path)
        stl_object_name = os.path.basename(stl_path)
        msh_object_name = os.path.basename(msh_path)
        try:
            logger.info("[MinIO] Uploading PNG...")
            client.fput_object(
                bucket_name="displacement",
                object_name=png_object_name,
                file_path=png_path
            )

            logger.info("[MinIO] Uploading STL...")
            client.fput_object(
                bucket_name="stl",
                object_name=stl_object_name,
                file_path=stl_path
            )

            if os.path.exists(msh_path):
                logger.info("[MinIO] Uploading MSH...")
                client.fput_object(
                    bucket_name="tetrahedral",
                    object_name=msh_object_name,
                    file_path=msh_path
                )
            else:
                logger.warning(f"No MSH file found for {noise_name}_{map_size}.")

        except S3Error as s3e:
            logger.error(f"MinIO upload failed for {noise_name}_{map_size}: {s3e}")
            raise typer.Exit(1)

    logger.info(f"=== Done with noise: {noise_name}, size={map_size} ===\n")


@app.command("all")
def generate_all():
    """
    Generate displacement -> PNG -> STL -> MSH, then upload to MinIO
    for multiple noise functions & map sizes.
    Stops immediately if any step fails.
    """
    # If anything in this function fails, we exit
    client = setup_minio_client()

    # Define some noise variations
    noise_variations = {
        "sine": sine_noise,
        "square": square_noise,
        "perlin": perlin_noise,
        "fbm": fbm_noise,
        "fractal": fractal_noise,
        "gabor": gabor_noise,
    }

    # Example map sizes
    map_sizes = [64, 128]

    # Loop
    for noise_name, noise_func in noise_variations.items():
        for size in map_sizes:
            generate_and_process(
                noise_name=noise_name,
                noise_func=noise_func,
                map_size=size,
                client=client
            )

    logger.info("All noise functions completed.")


if __name__ == "__main__":
    app()
