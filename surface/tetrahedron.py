#!/usr/bin/env python3
"""
tetrahedron.py

1) run_tetwild()   - Original TetWild
2) run_ftetwild()  - fTetWild (faster version of TetWild)
3) run_tetgen()    - TetGen support

Each function can run in local or Docker mode. 
"""

from storage import (
    setup_minio_client,
    ensure_bucket_exists,
    upload_file_to_minio
)

import sys
import os
import shutil
import logging
import subprocess
from minio import S3Error
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
import threading
import time

logging.basicConfig(
    level=logging.INFO,
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
    ideal_edge_length: float = 0.02,
    epsilon: float = 0.0001,
    stop_energy: float = 10.0,
    max_iterations: int = 80,
    docker: bool = False,
    ftetwild_bin: str = "/home/antoine/code/fTetWild/build/FloatTetwild_bin",
    upload_to_minio: bool = False
):
    """
    Runs fTetWild, producing .msh in 'stl/' directory by default.
    Waits for all output files to be created before uploading them to MinIO.
    """
    # Ensure we store final result in a stable location, e.g., "stl" subfolder
    stl_dir = "stl"
    os.makedirs(stl_dir, exist_ok=True)

    # Construct the output .msh name
    basename = os.path.splitext(os.path.basename(input_mesh))[0]  # e.g., "sine_64"
    output_mesh = os.path.join(stl_dir, f"{basename}_ftet.msh")

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

        logger.info(f"[fTetWild-Local] Using binary: {ftetwild_bin}")
        cmd = [
            ftetwild_bin,
            "-i", input_mesh,
            "-o", output_mesh
        ]

    logger.info("[fTetWild] Command: " + " ".join(cmd))

    # Initialize the tqdm progress bar
    pbar = tqdm(total=100, desc="[fTetWild] Processing", bar_format='{l_bar}{bar} [ time left: {remaining} ]')

    # Function to update the progress bar periodically
    def update_pbar():
        while not process.poll() and not pbar.n >= 100:
            pbar.update(1)
            time.sleep(0.1)  # Adjust as needed for smoother progress

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

        # Start the progress bar updater thread
        thread = threading.Thread(target=update_pbar, daemon=True)
        thread.start()

        # Read stdout in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"[fTetWild] {line.strip()}")

        # Read any remaining stderr
        stderr = process.stderr.read()
        if stderr.strip():
            logger.warning(f"[fTetWild] STDERR:\n{stderr}")

        retcode = process.wait()
        if retcode != 0:
            logger.error(f"[fTetWild] Exited with code {retcode}")
            raise subprocess.CalledProcessError(retcode, cmd)

    except subprocess.CalledProcessError as e:
        logger.error(f"[fTetWild] Failed with exit code {e.returncode}")
        logger.error("[fTetWild] STDOUT:\n" + (e.stdout or ""))
        logger.error("[fTetWild] STDERR:\n" + (e.stderr or ""))
        raise
    except FileNotFoundError:
        logger.error(f"[fTetWild] Could not find the binary '{ftetwild_bin}'.")
        raise
    finally:
        # Ensure the progress bar is closed
        pbar.close()
        if 'thread' in locals():
            thread.join()

    # --------------------------------------------------------------------------
    # 1) By the time we reach here, fTetWild has completed. We expect the .msh
    #    plus any additional sidecar files (CSV, OBJ, STL) to exist.
    # --------------------------------------------------------------------------
    logger.info(f"[fTetWild] Successfully generated: {output_mesh}")

    # Gather all potential extra files. 
    # If fTetWild is generating these exact suffixes:
    extra_suffixes = [
        "_.csv",                   # e.g. "..._ftet.msh_.csv"
        "__sf.obj",                # e.g. "..._ftet.msh__sf.obj"
        "__tracked_surface.stl"    # e.g. "..._ftet.msh__tracked_surface.stl"
    ]
    all_files_to_upload = [output_mesh]  # Always upload the main .msh
    for suf in extra_suffixes:
        candidate = output_mesh + suf
        if os.path.exists(candidate):
            all_files_to_upload.append(candidate)
        else:
            logger.warning(f"[fTetWild] Expected sidecar file not found: {candidate}")

    # --------------------------------------------------------------------------
    # 2) Now, optionally upload them to MinIO.
    # --------------------------------------------------------------------------
    if upload_to_minio:
        try:
            logger.info("[fTetWild] Uploading files to MinIO...")
            minio_client, minio_bucket = setup_minio_client()
            ensure_bucket_exists(minio_client, minio_bucket)

            for local_file in all_files_to_upload:
                obj_name = os.path.basename(local_file)
                upload_file_to_minio(
                    client=minio_client,
                    bucket_name=minio_bucket,
                    file_path=local_file,
                    object_name=obj_name
                )

            # 3) Delete local files after successful upload
            for local_file in all_files_to_upload:
                try:
                    os.remove(local_file)
                    logger.info(f"Deleted local file: {local_file}")
                except OSError as e:
                    logger.warning(f"Could not delete file {local_file}: {e}")

        except S3Error as s3e:
            logger.error(f"[fTetWild] Upload failed: {s3e}")
            raise