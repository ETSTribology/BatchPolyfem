#!/usr/bin/env python3
"""
tetrahedron.py

1) run_ftetwild()  - fTetWild (faster version of TetWild)
"""

from storage import (
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
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure Logging
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to suppress INFO logs
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
    ftetwild_bin: str = "ftetwild",  # Default to 'ftetwild', configurable via CLI
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
        # Docker approach (omitted)
        raise NotImplementedError("Docker mode for fTetWild not implemented here.")
    else:
        # Check the local binary or PATH
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
            "-o", output_mesh_with_suffix,
            # Uncomment and adjust parameters as needed
            # "-l", str(ideal_edge_length),
            # "-e", str(epsilon),
            # "-s", str(stop_energy),
            # "-m", str(max_iterations)
        ]

    # Initialize the tqdm progress bar
    pbar = tqdm(total=100, desc="[fTetWild] Processing", bar_format='{l_bar}{bar} [ time left: {remaining} ]')

    try:
        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
            universal_newlines=True
        )

        # Function to update the progress bar based on stdout
        def update_pbar():
            for line in process.stdout:
                # Parse lines like "Progress: 45%"
                if "Progress:" in line:
                    try:
                        progress_str = line.strip().split("Progress:")[1].strip().replace("%", "")
                        progress = int(progress_str)
                        pbar.n = progress
                        pbar.refresh()
                    except (IndexError, ValueError):
                        pass
            process.stdout.close()

        # Start the progress updater thread
        thread = threading.Thread(target=update_pbar, daemon=True)
        thread.start()

        # Wait for the process to complete
        process.wait()
        thread.join()

        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"[fTetWild] Exited with code {process.returncode}")
            logger.error(f"[fTetWild] STDERR:\n{stderr}")
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        logger.error(f"[fTetWild] Failed with exit code {e.returncode}")
        logger.error("[fTetWild] STDERR:\n" + (e.stderr or ""))
        raise
    except FileNotFoundError:
        logger.error(f"[fTetWild] Could not find the binary '{ftetwild_bin}'.")
        raise
    finally:
        # Ensure the progress bar is closed
        pbar.close()

    logger.warning(f"[fTetWild] Successfully generated: {output_mesh_with_suffix}")

    # Rename the .msh file to remove the '_ftet' suffix
    output_mesh_final = os.path.join(output_dir, f"{basename}.msh")
    if os.path.exists(output_mesh_with_suffix):
        os.rename(output_mesh_with_suffix, output_mesh_final)
        logger.warning(f"Renamed '{output_mesh_with_suffix}' to '{output_mesh_final}'")
    else:
        logger.error(f"Expected output mesh '{output_mesh_with_suffix}' not found.")
        raise FileNotFoundError(f"Expected output mesh '{output_mesh_with_suffix}' not found.")

    # Gather all potential extra files based on fTetWild's actual output
    extra_suffixes = [
        "_.csv",
        "__sf.obj",
        "__tracked_surface.stl"
    ]
    all_files_to_upload = [output_mesh_final]  # Always upload the main .msh
    for suf in extra_suffixes:
        candidate = output_mesh_final + suf
        if os.path.exists(candidate):
            all_files_to_upload.append(candidate)
        else:
            logger.warning(f"[fTetWild] Expected sidecar file not found: {candidate}")

    # Optionally upload to MinIO using threading for faster uploads
    if upload_to_minio:
        try:
            logger.warning("[fTetWild] Uploading files to MinIO concurrently...")
            minio_client, minio_bucket = setup_minio_client()
            ensure_bucket_exists(minio_client, minio_bucket)

            def upload_single_file(local_file):
                obj_name = os.path.basename(local_file)
                upload_file_to_minio(
                    client=minio_client,
                    bucket_name=minio_bucket,
                    file_path=local_file,
                    object_name=obj_name
                )
                return local_file

            # Use ThreadPoolExecutor to upload files concurrently
            with ThreadPoolExecutor(max_workers=min(8, len(all_files_to_upload))) as executor:
                futures = [executor.submit(upload_single_file, lf) for lf in all_files_to_upload]
                for future in as_completed(futures):
                    try:
                        uploaded_file = future.result()
                        logger.warning(f"Uploaded: {uploaded_file}")
                    except Exception as e:
                        logger.error(f"Error uploading file: {e}")

            # Delete local files after successful upload
            for local_file in all_files_to_upload:
                try:
                    os.remove(local_file)
                    logger.warning(f"Deleted local file: {local_file}")
                except OSError as e:
                    logger.warning(f"Could not delete file {local_file}: {e}")

        except S3Error as s3e:
            logger.error(f"[fTetWild] Upload failed: {s3e}")
            raise