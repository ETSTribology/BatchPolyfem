#!/usr/bin/env python3
"""
tetrahedron.py

1) run_ftetwild()  - fTetWild (faster version of TetWild)
"""

from storage import (
    REQUIRED_BUCKETS,
    setup_minio_client,
    ensure_bucket_exists,
    upload_file_to_minio
)

import os
import shutil
import logging
import subprocess
from minio import S3Error
from rich.console import Console
from rich.logging import RichHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

import ray
from ray.util import ActorPool

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Set to WARNING to suppress INFO logs
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
console = Console()

# ----------------------------------------------------------------------------
# fTetWild
# ----------------------------------------------------------------------------
def run_ftetwild(
    input_mesh: str,
    output_dir: str,
    ideal_edge_length: float = 0.02,
    epsilon: float = 0.0001,
    stop_energy: float = 10.0,
    max_iterations: int = 80,
    docker: bool = False,
    ftetwild_bin: str = "/home/antoine/code/fTetWild/build/FloatTetwild_bin",  # Default to 'ftetwild', configurable via CLI
    upload_to_minio: bool = False
):
    """
    Runs fTetWild, producing .msh in the specified output directory.
    Waits for all output files to be created before uploading them to MinIO.

    Args:
        input_mesh (str): Path to the input STL file.
        output_dir (str): Directory where the .msh file will be saved.
        ideal_edge_length (float): Ideal edge length parameter for fTetWild.
        epsilon (float): Epsilon parameter for fTetWild.
        stop_energy (float): Stop energy parameter for fTetWild.
        max_iterations (int): Maximum iterations for fTetWild.
        docker (bool): Whether to run fTetWild in Docker mode.
        ftetwild_bin (str): Path or alias for the fTetWild binary.
        upload_to_minio (bool): Whether to upload the output files to MinIO.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output .msh name
    basename = os.path.splitext(os.path.basename(input_mesh))[0]  
    output_mesh_with_suffix = os.path.join(output_dir, f"{basename}_ftet.msh")

    if docker:
        raise NotImplementedError("Docker mode for fTetWild not implemented here.")
    else:
        if not shutil.which(ftetwild_bin):
            err_msg = (
                f"[fTetWild] The specified ftetwild binary '{ftetwild_bin}' "
                "was not found. Install fTetWild or provide a full path."
            )
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.warning(f"[fTetWild-Local] Using binary: {ftetwild_bin}")

        cmd = [
            ftetwild_bin,
            "-i", input_mesh,
            "-o", output_mesh_with_suffix
        ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"[fTetWild] Exited with code {process.returncode}")
            logger.error(f"[fTetWild] STDERR:\n{stderr}")
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"[fTetWild] Failed with exit code {e.returncode}")
        raise
    except FileNotFoundError:
        logger.error(f"[fTetWild] Could not find the binary '{ftetwild_bin}'.")
        raise

    logger.warning(f"[fTetWild] Successfully generated: {output_mesh_with_suffix}")

    output_mesh_final = os.path.join(output_dir, f"{basename}.msh")
    if os.path.exists(output_mesh_with_suffix):
        os.rename(output_mesh_with_suffix, output_mesh_final)
        logger.warning(f"Renamed '{output_mesh_with_suffix}' to '{output_mesh_final}'")
    else:
        logger.error(f"Expected output mesh '{output_mesh_with_suffix}' not found.")
        raise FileNotFoundError(f"Expected output mesh '{output_mesh_with_suffix}' not found.")

    if upload_to_minio:
        try:
            logger.warning("[fTetWild] Uploading files to MinIO...")
            minio_client, minio_bucket = setup_minio_client()
            ensure_bucket_exists(minio_client, minio_bucket)

            upload_file_to_minio(
                client=minio_client,
                bucket_name=minio_bucket,
                file_path=output_mesh_final,
                object_name=os.path.basename(output_mesh_final)
            )
            logger.warning(f"Uploaded: {output_mesh_final}")
        except S3Error as s3e:
            logger.error(f"[fTetWild] Upload failed: {s3e}")
            raise

@ray.remote
def run_ftetwild_ray(
    input_mesh: str,
    output_dir: str,
    ideal_edge_length: float = 0.02,
    epsilon: float = 0.0001,
    stop_energy: float = 10.0,
    max_iterations: int = 80,
    ftetwild_bin: str = "ftetwild"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_mesh))[0]
    output_mesh_with_suffix = os.path.join(output_dir, f"{basename}_ftet.msh")
    output_mesh_final = os.path.join(output_dir, f"{basename}.msh")

    if not shutil.which(ftetwild_bin):
        raise FileNotFoundError(f"The specified ftetwild binary '{ftetwild_bin}' was not found.")

    cmd = [
        ftetwild_bin,
        "-i", input_mesh,
        "-o", output_mesh_with_suffix,
        "-l", str(ideal_edge_length),
        "-e", str(epsilon),
        "-s", str(stop_energy),
        "-m", str(max_iterations)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.path.exists(output_mesh_with_suffix):
            os.rename(output_mesh_with_suffix, output_mesh_final)
            logger.debug(f"[fTetWild] Successfully generated: {output_mesh_final}")
        else:
            raise FileNotFoundError(f"Expected output mesh '{output_mesh_with_suffix}' not found.")
    except Exception as e:
        logger.error(f"[fTetWild] Error processing {input_mesh}: {e}")
        raise

    return output_mesh_final

def distribute_ftetwild_tasks(tasks, ftetwild_bin="ftetwild", upload_to_minio=False):
    logger.debug(f"Connected to Ray cluster: {ray.cluster_resources()}")

    ray_tasks = [
        run_ftetwild_ray.remote(task[0], task[1], ftetwild_bin=ftetwild_bin)
        for task in tasks
    ]

    results = []
    while ray_tasks:
        done, ray_tasks = ray.wait(ray_tasks, num_returns=1, timeout=30)
        for result in done:
            try:
                output_file = ray.get(result)
                results.append(output_file)
                logger.debug(f"Task completed successfully: {output_file}")
            except Exception as e:
                logger.error(f"Task failed: {e}")

    if upload_to_minio:
        upload_results_to_minio(results)

    ray.shutdown()

def upload_results_to_minio(file_paths: list) -> None:
    minio_client, minio_bucket = setup_minio_client()
    ensure_bucket_exists(minio_client, minio_bucket)

    def upload_file(file_path):
        object_name = os.path.basename(file_path)
        upload_file_to_minio(minio_client, minio_bucket, file_path, object_name)
        logger.debug(f"Uploaded {file_path} to MinIO.")

    with ThreadPoolExecutor(max_workers=min(8, len(file_paths))) as executor:
        futures = [executor.submit(upload_file, file) for file in file_paths]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Failed to upload file: {e}")

