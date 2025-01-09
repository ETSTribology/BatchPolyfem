#!/usr/bin/env python3
"""
generate.py

Demonstration script that:
1) Generates 2D displacement maps from various noise functions or scan images.
2) Converts displacement PNG -> STL using 'hmm'.
3) Runs fTetWild on that STL to produce a tetrahedral mesh (.msh).
4) Uploads all results to MinIO.
5) Collects and uploads metadata as JSON to MinIO.

Enhanced to STOP immediately if any error occurs, organizes files into noise-specific folders,
ensures precise naming based on all parameters, skips generating/uploading existing files,
adds rebuild features, utilizes threading for tetrahedral mesh processing,
and includes random rotation in scan image processing mapped to original scan size.
"""

from datetime import datetime
import os
import sys
import tempfile
import logging
import subprocess
import typer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import uniform

import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

from minio import Minio, S3Error

# Local imports
from displacement import generate_displacement_map, save_displacement_map
from noises import noise_variations
from storage import (
    REQUIRED_BUCKETS,
    setup_minio_client,
    ensure_bucket_exists,
    upload_file_to_minio,
    check_file_exists_in_minio,
    list_scan_images,
    download_scan_image,
    build_filename
)
from hmm import run_hmm
from tetrahedron import run_ftetwild
from stats import collect_and_upload_metadata

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
app = typer.Typer(
    help="CLI to generate 2D displacement -> STL -> Tetrahedral Mesh, then upload to MinIO."
)

###############################################################################
# Common Pipeline Function
###############################################################################
def common_pipeline(
    png_preparation_func,
    preparation_args: dict,
    identifier: str,
    params: dict,
    map_size: int,
    client: Minio,
    required_buckets: list,
    ftetwild_bin: str,
    rebuild: bool,
    rebuild_metadata: bool,
    progress_bar: tqdm,
    metadata_extra: dict = {}
):
    """
    Runs the common steps from PNG generation to metadata upload.

    Args:
        png_preparation_func: Function to generate and save PNG. Should return path to PNG.
        preparation_args (dict): Arguments for png_preparation_func.
        identifier (str): Name identifier (noise_name or scan_name).
        params (dict): Parameters for metadata.
        map_size (int): Size of the displacement map.
        client (Minio): MinIO client instance.
        required_buckets (list): List of required bucket names.
        ftetwild_bin (str): Path for the fTetWild binary.
        rebuild (bool): Flag to force regeneration.
        rebuild_metadata (bool): Flag to force metadata regeneration.
        progress_bar (tqdm): Progress bar instance.
        metadata_extra (dict): Additional metadata parameters.
    """
    logger.info(f"=== Processing {identifier}, size: {map_size} ===")
    generation_start_time = datetime.now().timestamp()

    # Ensure required buckets
    for bucket in required_buckets:
        ensure_bucket_exists(client, bucket)

    # Define filenames and object names
    png_filename = build_filename(identifier, params, map_size, "png")
    stl_filename = build_filename(identifier, params, map_size, "stl")
    msh_filename = build_filename(identifier, params, map_size, "msh")

    # Determine object paths based on type (noise or scan)
    if params.get("scan"):
        base_path = f"scan/{identifier}"
    else:
        base_path = identifier

    displacement_obj = f"{base_path}/{png_filename}"
    stl_obj = f"{base_path}/{stl_filename}"
    msh_obj = f"{base_path}/{msh_filename}"

    # Check existence in MinIO unless rebuilding
    if not rebuild:
        try:
            png_exists = check_file_exists_in_minio(client, "displacement", displacement_obj)
            stl_exists = check_file_exists_in_minio(client, "stl", stl_obj)
            msh_exists = check_file_exists_in_minio(client, "tetrahedral", msh_obj)

            if png_exists and stl_exists and msh_exists:
                logger.info(f"All files already exist in MinIO for {identifier}, size: {map_size}. Skipping generation.")
                progress_bar.update(1)
                return
            else:
                logger.info(f"Files missing in MinIO for {identifier}, size: {map_size}. Proceeding with generation.")
        except S3Error as s3e:
            logger.error(f"MinIO check failed for {identifier}, size {map_size}: {s3e}")
            typer.Exit(code=1)
    else:
        logger.info(f"Rebuild enabled. Regenerating files for {identifier}, size: {map_size}.")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Step 1: Prepare PNG (noise generation or scan processing)
            png_path = png_preparation_func(tmpdir=tmpdir, **preparation_args)
            logger.info(f"Generated PNG: {png_path}")
        except Exception as e:
            logger.error(f"PNG preparation failed for {identifier}, size: {map_size}: {e}")
            typer.Exit(code=1)

        # Step 2: Convert PNG -> STL via hmm
        stl_path = os.path.join(tmpdir, stl_filename)
        hmm_start_time = datetime.now().timestamp()
        try:
            run_hmm(
                input_file=png_path,
                output_file=stl_path,
                error="0.001",
                z_exagg=10.0 if not params.get("scan") else 2.0,
                z_value=3 if not params.get("scan") else 10.0,
                base=1 if not params.get("scan") else 0.001,
                triangles=None,
                quiet=True
            )
            logger.info(f"Generated STL: {stl_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"hmm failed for {png_path}: {e}")
            typer.Exit(code=1)

        # Step 3: Tetrahedral mesh via fTetWild
        tet_dir = os.path.join(tmpdir, "ftetwild_output")
        os.makedirs(tet_dir, exist_ok=True)
        msh_path = os.path.join(tet_dir, msh_filename)
        ftetwild_start_time = datetime.now().timestamp()
        try:
            logger.info(f"Running fTetWild on STL -> MSH: {msh_path}")
            run_ftetwild(
                input_mesh=stl_path,
                output_dir=tet_dir,
                ideal_edge_length=0.02,
                epsilon=0.0001,
                stop_energy=10.0,
                max_iterations=30,
                docker=False,
                ftetwild_bin=ftetwild_bin,
                upload_to_minio=False
            )
            if not os.path.exists(msh_path):
                logger.error(f"Failed to generate .msh at {msh_path}. Aborting.")
                typer.Exit(code=1)
            logger.info(f"Generated MSH: {msh_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"fTetWild failed for {stl_path}: {e}")
            typer.Exit(code=1)
        except Exception as e:
            logger.error(f"fTetWild error: {e}")
            typer.Exit(code=1)

        # Step 4: Upload results to MinIO
        try:
            logger.info("[MinIO] Uploading PNG...")
            upload_file_to_minio(client, "displacement", png_path, displacement_obj, overwrite=rebuild)

            logger.info("[MinIO] Uploading STL...")
            upload_file_to_minio(client, "stl", stl_path, stl_obj, overwrite=rebuild)

            logger.info("[MinIO] Uploading MSH...")
            upload_file_to_minio(client, "tetrahedral", msh_path, msh_obj, overwrite=rebuild)
        except S3Error as s3e:
            logger.error(f"MinIO upload failed for {identifier}, size {map_size}: {s3e}")
            typer.Exit(code=1)

        # Step 5: Collect and Upload Metadata
        try:
            collect_and_upload_metadata(
                noise_name=identifier,
                noise_params=params,
                map_size=map_size,
                client=client,
                displacement_obj=displacement_obj,
                stl_obj=stl_obj,
                msh_obj=msh_obj,
                displacement_path=png_path,
                stl_path=stl_path,
                msh_path=msh_path,
                generation_start_time=generation_start_time,
                hmm_start_time=hmm_start_time,
                ftetwild_start_time=ftetwild_start_time,
                hmm_triangles=None,
                hmm_error=0.0001 if not params.get("scan") else 0.001,
                hmm_z_exaggeration=1 if not params.get("scan") else 2,
                hmm_z_value=10.0 if not params.get("scan") else 10.0,
                hmm_base=1 if not params.get("scan") else 1,
                ftetwild_bin=ftetwild_bin,
                rebuild_metadata=rebuild_metadata,
                **metadata_extra
            )
        except Exception as e:
            logger.error(f"Metadata upload failed for {identifier}, size {map_size}: {e}")
            typer.Exit(code=1)

    logger.info(f"=== Done with {identifier}, size {map_size} ===\n")
    progress_bar.update(1)

###############################################################################
# Specific Preparation Functions
###############################################################################
def prepare_noise_png(tmpdir, noise_func, noise_params, map_size):
    disp = generate_displacement_map(
        noise_func=noise_func,
        map_size=map_size,
        noise_params=noise_params,
        normalize=True
    )
    png_filename = build_filename(noise_params.get("name", "noise"), noise_params, map_size, "png")
    png_path = os.path.join(tmpdir, png_filename)
    save_displacement_map(disp, png_path)
    return png_path

def prepare_scan_png(tmpdir, scan_object, map_size):
    # Download scan image
    scan_local_path = os.path.join(tmpdir, Path(scan_object).name)
    # Assuming 'client', 'scan_bucket' are globally accessible or passed separately if needed.
    # For simplicity in this function, we will assume the scan image has been downloaded beforehand.
    # Adjust as necessary based on your environment.
    download_scan_image(client, params["scan_bucket"], scan_object, scan_local_path)

    # Open the scan image and apply random rotation
    disp_image = Image.open(scan_local_path).convert("L")
    original_size = disp_image.size
    random_angle = uniform(0, 360)
    disp_image = disp_image.rotate(random_angle, resample=Image.BICUBIC, expand=False)
    logger.info(f"Applied random rotation: {random_angle:.2f} degrees.")

    disp_image = disp_image.resize((map_size, map_size), Image.LANCZOS)
    png_filename = build_filename(Path(scan_object).stem, {"scan": True}, map_size, "png")
    png_path = os.path.join(tmpdir, png_filename)
    disp_image.save(png_path)
    return png_path

###############################################################################
# Processing Functions for Noise and Scan
###############################################################################
def process_noise(
    noise_name: str,
    noise_func,
    noise_params: dict,
    map_size: int,
    client: Minio,
    required_buckets: list,
    ftetwild_bin: str,
    rebuild: bool,
    rebuild_metadata: bool,
    progress_bar: tqdm
):
    params = noise_params.copy()
    params["name"] = noise_name
    common_pipeline(
        png_preparation_func=prepare_noise_png,
        preparation_args={
            "noise_func": noise_func,
            "noise_params": noise_params,
            "map_size": map_size
        },
        identifier=noise_name,
        params=params,
        map_size=map_size,
        client=client,
        required_buckets=required_buckets,
        ftetwild_bin=ftetwild_bin,
        rebuild=rebuild,
        rebuild_metadata=rebuild_metadata,
        progress_bar=progress_bar
    )

def process_scan(
    scan_name: str,
    scan_object: str,
    scan_bucket: str,
    map_size: int,
    client: Minio,
    required_buckets: list,
    ftetwild_bin: str,
    rebuild: bool,
    rebuild_metadata: bool,
    progress_bar: tqdm
):
    params = {"scan": True, "scan_bucket": scan_bucket}
    # Pass global variables needed by prepare_scan_png
    global client_global, params_global
    client_global = client
    params_global = params

    common_pipeline(
        png_preparation_func=prepare_scan_png,
        preparation_args={
            "scan_object": scan_object,
            "map_size": map_size
        },
        identifier=scan_name,
        params=params,
        map_size=map_size,
        client=client,
        required_buckets=required_buckets,
        ftetwild_bin=ftetwild_bin,
        rebuild=rebuild,
        rebuild_metadata=rebuild_metadata,
        progress_bar=progress_bar
    )

###############################################################################
# Main Steps
###############################################################################
@app.command("all")
def generate_all(
    map_sizes: list[int] = typer.Option(
        [64, 128, 256, 512, 1024],
        help="List of map sizes to generate.",
        show_default=True
    ),
    noise_types: list[str] = typer.Option(
        [
            "sine", "square", "perlin", "fbm", "gabor", 
            "random_walk", "ornstein_uhlenbeck", "vasicek", 
            "blue_noise", "halton", "wavelet", "domain_warp"
        ],
        help="List of noise types to generate displacement maps.",
        show_default=True
    ),
    ftetwild_bin: str = typer.Option(
        "/home/antoine/code/fTetWild/build/FloatTetwild_bin",
        "--ftetwild-bin",
        help="Path or alias for the fTetWild binary.",
        show_default=True
    ),
    process_scans: bool = typer.Option(
        False,
        "--process-scans",
        help="Enable processing of scan images from MinIO.",
        show_default=False
    ),
    scan_bucket: str = typer.Option(
        "scan",
        "--scan-bucket",
        help="Name of the MinIO bucket containing scan images.",
        show_default=True
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Rebuild all displacement maps, STL, and tetrahedral meshes.",
        show_default=False
    ),
    rebuild_metadata: bool = typer.Option(
        False,
        "--rebuild-metadata",
        help="Rebuild metadata JSON files.",
        show_default=False
    ),
):
    """
    Generate displacement maps for multiple noise functions and map sizes,
    convert them to STL, create tetrahedral meshes, and upload all results to MinIO.
    Stops immediately if any step fails.
    """
    unsupported_noises = [noise for noise in noise_types if noise not in noise_variations]
    if unsupported_noises:
        logger.error(f"Unsupported noise types: {unsupported_noises}")
        raise typer.Exit(code=1)

    scan_images = []
    scan_tasks = 0
    if process_scans:
        client = setup_minio_client(bucket_names=REQUIRED_BUCKETS + [scan_bucket])
        scan_images = list_scan_images(client, scan_bucket)
        scan_tasks = len(scan_images) * len(map_sizes)

    noise_tasks = sum(len(noise_variations[noise]) * len(map_sizes) for noise in noise_types)

    total_tasks = scan_tasks + noise_tasks

    if total_tasks == 0:
        logger.info("No tasks to process. Exiting.")
        sys.exit(0)

    if not process_scans and not noise_types:
        logger.info("No noise types or scan processing enabled. Exiting.")
        sys.exit(0)

    if not process_scans:
        client = setup_minio_client(bucket_names=REQUIRED_BUCKETS)

    with tqdm(total=total_tasks, desc="Overall Progress", bar_format='{l_bar}{bar} [ time left: {remaining} ]') as progress_bar:
        tasks = []
        max_workers = min(16, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if process_scans:
                for scan_object in scan_images:
                    scan_name = Path(scan_object).stem
                    for size in map_sizes:
                        task = executor.submit(
                            process_scan,
                            scan_name=scan_name,
                            scan_object=scan_object,
                            scan_bucket=scan_bucket,
                            map_size=size,
                            client=client,
                            required_buckets=REQUIRED_BUCKETS,
                            ftetwild_bin=ftetwild_bin,
                            rebuild=rebuild,
                            rebuild_metadata=rebuild_metadata,
                            progress_bar=progress_bar
                        )
                        tasks.append(task)

            for noise_name in noise_types:
                for noise_func, noise_params in noise_variations[noise_name]:
                    for size in map_sizes:
                        task = executor.submit(
                            process_noise,
                            noise_name=noise_name,
                            noise_func=noise_func,
                            noise_params=noise_params,
                            map_size=size,
                            client=client,
                            required_buckets=REQUIRED_BUCKETS,
                            ftetwild_bin=ftetwild_bin,
                            rebuild=rebuild,
                            rebuild_metadata=rebuild_metadata,
                            progress_bar=progress_bar
                        )
                        tasks.append(task)

            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"An error occurred during processing: {e}")
                    typer.Exit(code=1)

    logger.info("All processing tasks completed successfully.")

###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    app()