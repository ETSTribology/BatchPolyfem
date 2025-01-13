#!/usr/bin/env python3
"""
Storage Module

Handles MinIO client setup, bucket operations, file uploads/downloads, and filename construction.
Includes dedicated functions for interacting with a temporary ("temp") bucket.
"""

import os
import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Generator
import logging
from rich.logging import RichHandler
import json

import s3fs
import pyarrow.fs
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from minio import Minio, S3Error

import bisect
from functools import lru_cache
from typing import List, Optional

# Configure logger
logging.basicConfig(
    level=logging.INFO,  # or INFO as needed
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

from enum import Enum

class Buckets(Enum):
    DISPLACEMENT = "displacement"
    STL = "stl"
    TETRAHEDRAL = "tetrahedral"
    METADATA = "metadata"
    SIMULATION = "simulation"
    TEMP = "temp"
    BASE_SIMULATION_PARAMS = "simulation-base-params"
    SIMULATION_RESULTS = "simulation-results"
    ANALYSIS_RESULTS = "analysis-results"

REQUIRED_BUCKETS = [bucket.value for bucket in Buckets]

BUCKET_CONFIG = {
    "displacement": {"description": "Stores displacement data", "region": "us-east-1"},
    "stl": {"description": "Stores STL files", "region": "us-east-1"},
    "tetrahedral": {"description": "Stores tetrahedral mesh files", "region": "us-east-1"},
    "metadata": {"description": "Stores metadata for simulations", "region": "us-east-1"},
    "simulation": {"description": "Primary simulation data", "region": "us-east-1"},
    "temp": {"description": "Temporary files", "region": "us-east-1"},
    "simulation-base-params": {"description": "Base simulation parameters", "region": "us-east-1"},
    "simulation-results": {"description": "Simulation results data", "region": "us-east-1"},
    "analysis-results": {"description": "Analysis results", "region": "us-east-1"},
}

REQUIRED_BUCKETS = list(BUCKET_CONFIG.keys())

# Ensure results directory exists
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def setup_minio_client(bucket_names: List[str]) -> Minio:
    """
    Sets up and returns a configured MinIO client.
    Ensures that the specified buckets exist.
    """
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "192.168.0.20:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
        region="canada"  # Set the expected region
    )
    for bucket in bucket_names:
        ensure_bucket_exists(client, bucket)
    return client

def ensure_bucket_exists(client: Minio, bucket: str | Buckets) -> None:
    """Ensures bucket existence; creates it if necessary."""
    bucket_name = get_bucket_name(bucket)
    try:
        if not client.bucket_exists(bucket_name):
            try:
                # Specify location/region when creating a bucket
                client.make_bucket(bucket_name, location="canada")
                logger.debug(f"Created bucket: {bucket_name}")
            except S3Error as e:
                # Ignore error if bucket already exists and is owned by you
                if e.code == "BucketAlreadyOwnedByYou":
                    logger.debug(f"Bucket {bucket_name} already exists and is owned by you.")
                else:
                    logger.error(f"Error creating bucket {bucket_name}: {e}")
                    raise
        else:
            logger.debug(f"Bucket already exists: {bucket_name}")
    except S3Error as e:
        logger.error(f"Error checking existence of bucket {bucket_name}: {e}")
        raise

def upload_file_to_minio(client: Minio, bucket: str | Buckets, file_path: str, object_name: str, overwrite: bool = False) -> None:
    """Uploads a file to a specified MinIO bucket."""
    bucket_name = get_bucket_name(bucket)
    try:
        if not overwrite and check_file_exists_in_minio(client, bucket_name, object_name):
            logger.debug(f"File {object_name} already exists in bucket {bucket_name}. Skipping upload.")
            return
        client.fput_object(bucket_name, object_name, file_path)
        logger.debug(f"Uploaded {file_path} to {bucket_name}/{object_name}.")
    except S3Error as err:
        logger.error(f"Failed to upload {file_path} to bucket {bucket_name}: {err}")
        raise

@lru_cache(maxsize=10)
def list_files_in_bucket(client: Minio, bucket: str | Buckets) -> list:
    bucket_name = get_bucket_name(bucket)
    try:
        # Create a sorted list of file names instead of a set
        files = sorted({obj.object_name for obj in client.list_objects(bucket_name, recursive=True)})
        logger.info(f"Cached {len(files)} files from bucket '{bucket_name}'")
        return files
    except S3Error as e:
        logger.error(f"Error listing files in bucket '{bucket_name}': {e}")
        raise

def check_file_exists_in_minio(
    client: Minio,
    bucket: str | Buckets,
    object_name: str,
    use_cache: bool = True
) -> bool:
    """
    Checks if a specific file exists in a MinIO bucket using cached file lists for faster lookup.

    Args:
        client (Minio): MinIO client instance.
        bucket (str | Buckets): Bucket name as a string or `Buckets` enum.
        object_name (str): Object name to check.
        use_cache (bool): Whether to use the cached file list for faster lookup. Default is True.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    bucket_name = get_bucket_name(bucket)

    if use_cache:
        try:
            cached_files = list_files_in_bucket(client, bucket_name)
            index = bisect.bisect_left(cached_files, object_name)
            if index < len(cached_files) and cached_files[index] == object_name:
                logger.debug(f"File '{object_name}' found in cache for bucket '{bucket_name}'")
                return True
        except Exception as e:
            logger.warning(f"Failed to use cached file list for bucket '{bucket_name}': {e}")

    # Fallback to direct lookup using MinIO stat_object
    try:
        client.stat_object(bucket_name, object_name)
        return True
    except S3Error as e:
        if e.code == 'NoSuchKey':
            return False
        logger.error(f"Error checking existence of '{object_name}' in bucket '{bucket_name}': {e}")
        raise

def list_scan_images(client: Minio, scan_bucket: str) -> List[Path]:
    """Lists all scan image objects in the specified MinIO bucket."""
    try:
        objects = client.list_objects(scan_bucket, recursive=True)
        return [
            Path(obj.object_name)
            for obj in objects 
            if obj.object_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    except S3Error as s3e:
        logger.error(f"Failed to list scan images in bucket '{scan_bucket}': {s3e}")
        raise

def download_scan_image(client: Minio, scan_bucket: str, object_name: str, download_path: str) -> None:
    """Downloads a scan image from MinIO to a local path."""
    try:
        client.fget_object(scan_bucket, object_name, download_path)
        logger.debug(f"Downloaded scan image: {object_name} to {download_path}")
    except S3Error as s3e:
        logger.error(f"Failed to download scan image '{object_name}': {s3e}")
        raise

def build_filename(noise_name: str, noise_params: dict, map_size: int, extension: str) -> str:
    """Constructs a filename based on noise parameters and map size."""
    param_parts = [f"{key}_{value}" for key, value in noise_params.items()]
    param_str = "_".join(param_parts)
    return f"{noise_name}_{param_str}_size_{map_size}.{extension}"

def generate_folder_simulation_name(config: dict) -> str:
    """Generates a folder name based on simulation parameters."""
    materials = "__".join(f"{mat['type']}_{mat['E']}_{mat['nu']}_{mat['rho']}" for mat in config["materials"])
    contact_params = f"{config['contact']['dhat']}__{config['contact']['epsv']}"
    time_params = f"{config['time']['tend']}_{config['time']['dt']}"
    solver_params = "__".join([
        config['solver']['linear']['solver'],
        config['solver']['contact']['CCD']['broad_phase'],
        config['solver']['nonlinear']['solver'],
        config['solver']['nonlinear']['line_search']['method']
    ])
    return f"{materials}__{contact_params}__{time_params}__{solver_params}"

def save_and_upload_results(client: Minio, config_path: Path, output_dir: Path, folder_name: str) -> None:
    """Uploads simulation results and configuration file to MinIO."""
    bucket_path = f"{Buckets.SIMULATION.value}/{folder_name}"

    for result_file in output_dir.iterdir():
        object_name = f"{bucket_path}/{result_file.name}"
        upload_file_to_minio(client, Buckets.SIMULATION, str(result_file), object_name)

    config_name = f"{bucket_path}/{config_path.name}"
    upload_file_to_minio(client, Buckets.SIMULATION, str(config_path), config_name)
    logger.debug(f"Uploaded all results to MinIO bucket: {Buckets.SIMULATION.value}/{folder_name}")

def safe_upload_file(client: Minio, bucket_name: str, file_path: str, object_name: str) -> None:
    """Safely uploads a file and logs the outcome."""
    try:
        upload_file_to_minio(client, bucket_name, file_path, object_name)
        logger.debug(f"Uploaded: {object_name}")
    except Exception as e:
        logger.error(f"Failed to upload {object_name}: {e}")

@contextmanager
def temp_directory() -> Generator[str, None, None]:
    """Context manager for a temporary directory that auto-deletes on exit."""
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def create_noise_folder(noise_name: str) -> Path:
    """Creates and returns a folder for a specific noise type under results."""
    folder = RESULTS_DIR / noise_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def build_displacement_filename(noise_name: str, noise_params: dict, map_size: int) -> str:
    param_str = format_params(noise_params, sep="_")
    filename = f"{noise_name}_{param_str}.png"
    return f"{noise_name}/{map_size}/{filename}"

def build_stl_filename(noise_name: str, noise_params: dict, program_stl: str, stl_params: dict, map_size: int) -> str:
    noise_param_str = format_params(noise_params, sep="_")
    stl_param_str = format_params(stl_params, sep="_")
    filename = f"{noise_name}_{noise_param_str}_{program_stl}_{stl_param_str}.stl"
    return f"{noise_name}/{map_size}/{filename}"

def build_tetrahedral_filename(
    noise_name: str,
    noise_params: dict,
    program_stl: str,
    stl_params: dict,
    program_tet: str,
    tet_params: dict,
    map_size: int,
) -> str:
    """
    Build a filename for tetrahedral (.msh) files.
    """
    noise_param_str = format_params(noise_params, sep="_")
    stl_param_str = format_params(stl_params, sep="_")
    tet_param_str = format_params(tet_params, sep="_")
    
    filename = (
        f"{noise_name}_{noise_param_str}_{program_stl}_{stl_param_str}_{program_tet}_{tet_param_str}.msh"
    )
    return f"{noise_name}/{map_size}/{filename}"

def build_metadata_filename(noise_name: str, noise_params: dict, map_size: int) -> str:
    param_str = format_params(noise_params, sep="_")
    filename = f"{noise_name}_{param_str}.json"
    return f"{noise_name}/{map_size}/{filename}"

def build_simulation_params_filename(
    noise_name: str,
    noise_params: dict,
    program_stl: str,
    stl_params: dict,
    program_tet: str,
    tet_params: dict,
    map_size: int,
    material_params: dict,
    contact_params: dict,
    time_params: dict,
    solver_params: dict,
    type_: str,
) -> str:
    noise_param_str = format_params(noise_params, sep="_")
    stl_param_str = format_params(stl_params, sep="_")
    tet_param_str = format_params(tet_params, sep="_")
    material_param_str = format_params(material_params, sep="_")
    contact_param_str = format_params(contact_params, sep="_")
    time_param_str = format_params(time_params, sep="_")
    solver_param_str = format_params(solver_params, sep="_")
    
    filename = (
        f"{noise_name}_{noise_param_str}_{program_stl}_{stl_param_str}_{program_tet}_{tet_param_str}"
        f"_{material_param_str}_{contact_param_str}_{time_param_str}_{solver_param_str}_{type_}.json"
    )
    return f"{noise_name}/{map_size}/{filename}"

def build_simulation_results_filename(
    noise_name: str,
    noise_params: dict,
    program_stl: str,
    stl_params: dict,
    program_tet: str,
    tet_params: dict,
    map_size: int,
    material_params: dict,
    contact_params: dict,
    time_params: dict,
    solver_params: dict,
    type_: str,
) -> str:
    noise_param_str = format_params(noise_params, sep="_")
    stl_param_str = format_params(stl_params, sep="_")
    tet_param_str = format_params(tet_params, sep="_")
    material_param_str = format_params(material_params, sep="_")
    contact_param_str = format_params(contact_params, sep="_")
    time_param_str = format_params(time_params, sep="_")
    solver_param_str = format_params(solver_params, sep="_")
    
    filename = (
        f"{noise_name}_{noise_param_str}_{program_stl}_{stl_param_str}_{program_tet}_{tet_param_str}"
        f"_{material_param_str}_{contact_param_str}_{time_param_str}_{solver_param_str}_{type_}.json"
    )
    return f"{noise_name}/{map_size}/{filename}"

def upload_file_to_temp(client: Minio, file_path: str, object_name: str, overwrite: bool = False) -> None:
    """
    Uploads a file to the 'temp' bucket in MinIO using the Buckets enum.
    """
    upload_file_to_minio(client, Buckets.TEMP, file_path, object_name, overwrite)

def check_file_exists_in_temp(client: Minio, object_name: str) -> bool:
    """
    Checks if a specific file exists in the 'temp' bucket using the Buckets enum.
    """
    return check_file_exists_in_minio(client, Buckets.TEMP, object_name)

def download_file_from_temp(client: Minio, object_name: str, download_path: str) -> None:
    """
    Downloads a file from the 'temp' bucket in MinIO to a local path using the Buckets enum.
    """
    try:
        client.fget_object(Buckets.TEMP.value, object_name, download_path)
        logger.debug(f"Downloaded temp file: {object_name} to {download_path}")
    except S3Error as s3e:
        logger.error(f"Failed to download temp file '{object_name}': {s3e}")
        raise

def list_files_in_temp(client: Minio) -> List[Path]:
    """
    Lists all files in the 'temp' bucket using the Buckets enum.
    """
    try:
        objects = client.list_objects(Buckets.TEMP.value, recursive=True)
        return [Path(obj.object_name) for obj in objects]
    except S3Error as s3e:
        logger.error(f"Failed to list files in bucket '{Buckets.TEMP.value}': {s3e}")
        raise

def format_params(params: dict, sep: str = "_") -> str:
    """
    Formats a dictionary of parameters into a sorted, delimited string.
    Example: {"z_value": 3, "error": "0.001"} -> "base-1_error-0.001_z-exagg-1.0_z-value-3"
    """
    # Sort keys for consistency
    sorted_items = sorted(params.items())
    # Create key-value pairs joined by '-'
    return sep.join(f"{key}-{value}" for key, value in sorted_items)


def setup_s3_connection(
    endpoint_url: str,
    access_key: str,
    secret_key: str
) -> pyarrow.fs.FileSystem:
    """
    Sets up an S3 connection using s3fs and returns a PyArrow filesystem.

    Args:
        endpoint_url (str): URL of the S3-compatible storage endpoint.
        access_key (str): Access key for authentication.
        secret_key (str): Secret key for authentication.

    Returns:
        pyarrow.fs.FileSystem: A PyArrow filesystem for interacting with the S3 bucket.
    """
    try:
        # Set up s3fs with the provided credentials
        s3_fs = s3fs.S3FileSystem(
            key=access_key,
            secret=secret_key,
            client_kwargs={"endpoint_url": endpoint_url}
        )
        # Create and return a PyArrow filesystem
        return pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))
    except Exception as e:
        logging.error(f"Failed to set up S3 connection: {e}")
        raise

def read_file_from_s3(
    bucket_name: str,
    file_path: str,
    s3_filesystem: pyarrow.fs.FileSystem,
    file_format: str = "csv"
):
    """
    Reads a file (CSV or JSON) from an S3 bucket using PyArrow and returns its content.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_path (str): Path to the file within the bucket.
        s3_filesystem (pyarrow.fs.FileSystem): A PyArrow filesystem for the S3 connection.
        file_format (str): File format ('csv' or 'json').

    Returns:
        Union[pd.DataFrame, dict]: The file contents as a Pandas DataFrame or dictionary.
    """
    try:
        # Define the full path to the file in the bucket
        storage_path = f"{bucket_name}/{file_path}"

        if file_format == "csv":
            # Read the file as a dataset using PyArrow
            dataset = ds.dataset(storage_path, filesystem=s3_filesystem, format=file_format)
            return dataset.to_table().to_pandas()
        elif file_format == "json":
            # Open the file and parse it as JSON
            with s3_filesystem.open_input_file(storage_path) as file:
                return json.load(file)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    except Exception as e:
        logging.error(f"Failed to read {file_format} file from S3: {e}")
        raise


def upload_files_to_minio(
    client: Minio,
    bucket: Buckets,
    file_paths: List[str],
    object_names: Optional[List[str]] = None,
    overwrite: bool = False,
    max_workers: int = 4
) -> None:
    """
    Uploads multiple files to MinIO using multithreading with a fallback to single-threaded execution.

    Args:
        client (Minio): MinIO client instance.
        bucket (Buckets): Target bucket as a `Buckets` enum.
        file_paths (List[str]): List of local file paths to upload.
        object_names (Optional[List[str]]): List of object names for the uploaded files. 
                                            Defaults to the file names in `file_paths`.
        overwrite (bool): Whether to overwrite existing files. Default is False.
        max_workers (int): Maximum number of threads for multithreaded execution.
    """
    bucket_name = bucket.value
    object_names = object_names or [Path(file).name for file in file_paths]

    def upload_single(file_path, object_name):
        """Uploads a single file to MinIO."""
        try:
            if not overwrite and check_file_exists_in_minio(client, bucket, object_name):
                logger.debug(f"File {object_name} already exists in bucket {bucket_name}. Skipping upload.")
                return
            client.fput_object(bucket_name, object_name, file_path)
            logger.info(f"Uploaded {file_path} to {bucket_name}/{object_name}.")
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to {bucket_name}/{object_name}: {e}")
            raise

    # Attempt multithreaded upload
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(upload_single, file_path, object_name): file_path
                for file_path, object_name in zip(file_paths, object_names)
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()  # Raises exception if the task failed
                except Exception as e:
                    logger.error(f"Error during upload of {file_path}: {e}")
    except Exception as e:
        logger.warning(f"Multithreaded upload failed, switching to single-threaded mode. Error: {e}")

        # Fallback to single-threaded upload
        for file_path, object_name in zip(file_paths, object_names):
            try:
                upload_single(file_path, object_name)
            except Exception as e:
                logger.error(f"Failed to upload {file_path} in single-threaded mode: {e}")

def get_bucket_name(bucket: str | Buckets) -> str:
    if isinstance(bucket, Buckets):
        return bucket.value
    if isinstance(bucket, str):
        return bucket
    raise ValueError(f"Invalid bucket type: {type(bucket)}")

