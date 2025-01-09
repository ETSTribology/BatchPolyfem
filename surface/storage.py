# storage.py

import os
import logging
from pathlib import Path
from minio import Minio, S3Error

logger = logging.getLogger(__name__)

# Constants
SIMULATION_BUCKET = "simulation"
NUM_THREADS = 4  # Adjust based on your system capabilities
RESULTS_DIR = Path("results")

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Define all required buckets
REQUIRED_BUCKETS = ["displacement", "stl", "tetrahedral", "metadata", "simulation"]


def setup_minio_client(bucket_names: list) -> Minio:
    """
    Sets up and returns a MinIO client.

    Args:
        bucket_names (list): List of bucket names to ensure exist.

    Returns:
        Minio: Configured MinIO client.
    """
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "192.168.0.20:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False  # Set to True if using HTTPS
    )
    # Ensure all required buckets exist
    for bucket in bucket_names:
        ensure_bucket_exists(client, bucket)
    return client

def ensure_bucket_exists(client: Minio, bucket_name: str):
    """
    Ensures that a bucket exists in MinIO. Creates it if it does not exist.

    Args:
        client (Minio): MinIO client instance.
        bucket_name (str): Name of the bucket.
    """
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Created bucket: {bucket_name}")
        else:
            logger.debug(f"Bucket already exists: {bucket_name}")
    except S3Error as e:
        logger.error(f"Error checking/creating bucket {bucket_name}: {e}")
        raise e

def upload_file_to_minio(client: Minio, bucket_name: str, file_path: str, object_name: str, overwrite: bool = False):
    """
    Uploads a file to a specified MinIO bucket.

    Args:
        client (Minio): MinIO client instance.
        bucket_name (str): Name of the bucket.
        file_path (str): Local path to the file.
        object_name (str): Object name in MinIO.
        overwrite (bool): Whether to overwrite the file if it exists.
    """
    try:
        if not overwrite:
            if check_file_exists_in_minio(client, bucket_name, object_name):
                logger.info(f"File {object_name} already exists in bucket {bucket_name}. Skipping upload.")
                return
        client.fput_object(bucket_name, object_name, file_path)
        logger.info(f"Uploaded {file_path} to {bucket_name}/{object_name}.")
    except S3Error as err:
        raise Exception(f"Failed to upload {file_path} to bucket {bucket_name}: {err}")


def check_file_exists_in_minio(client: Minio, bucket_name: str, object_name: str) -> bool:
    """
    Checks if a specific file exists in a MinIO bucket.

    Args:
        client (Minio): MinIO client instance.
        bucket_name (str): Name of the bucket.
        object_name (str): Object name to check.

    Returns:
        bool: True if the object exists, False otherwise.
    """
    try:
        client.stat_object(bucket_name, object_name)
        return True
    except S3Error as e:
        if e.code == 'NoSuchKey':
            return False
        else:
            logger.error(f"Error checking existence of '{object_name}' in bucket '{bucket_name}': {e}")
            raise e

def list_scan_images(client: Minio, scan_bucket: str) -> list[Path]:
    """
    Lists all scan image objects in the specified MinIO bucket.

    Args:
        client (Minio): MinIO client instance.
        scan_bucket (str): Name of the scan bucket.

    Returns:
        list[Path]: List of Paths representing scan image object names.
    """
    try:
        objects = client.list_objects(scan_bucket, recursive=True)
        scan_images = [obj.object_name for obj in objects if obj.object_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        return scan_images
    except S3Error as s3e:
        logger.error(f"Failed to list scan images in bucket '{scan_bucket}': {s3e}")
        raise s3e

def download_scan_image(client: Minio, scan_bucket: str, object_name: str, download_path: str):
    """
    Downloads a scan image from MinIO.

    Args:
        client (Minio): MinIO client instance.
        scan_bucket (str): Name of the scan bucket.
        object_name (str): Object name of the scan image.
        download_path (str): Local path to download the image.
    """
    try:
        client.fget_object(scan_bucket, object_name, download_path)
        logger.info(f"Downloaded scan image: {object_name} to {download_path}")
    except S3Error as s3e:
        logger.error(f"Failed to download scan image '{object_name}': {s3e}")
        raise s3e


def build_filename(noise_name: str, noise_params: dict, map_size: int, extension: str) -> str:
    """
    Constructs a precise filename based on noise parameters and map size.

    Args:
        noise_name (str): Name of the noise function.
        noise_params (dict): Parameters of the noise function.
        map_size (int): Size of the map.
        extension (str): File extension (e.g., 'png', 'stl', 'msh').

    Returns:
        str: Constructed filename.
    """
    # Construct parameter part
    param_parts = [f"{key}_{value}" for key, value in noise_params.items()]
    param_str = "_".join(param_parts)
    filename = f"{noise_name}_{param_str}_size_{map_size}.{extension}"
    return filename

def generate_folder_simulation_name(config: dict) -> str:
    """
    Generate a folder name based on the simulation parameters in the config.

    Args:
        config (dict): Simulation configuration.

    Returns:
        str: Folder name.
    """
    materials = "__".join([
        f"{mat['type']}_{mat['E']}_{mat['nu']}_{mat['rho']}"
        for mat in config["materials"]
    ])
    contact_params = f"{config['contact']['dhat']}__{config['contact']['epsv']}"
    time_params = f"{config['time']['tend']}_{config['time']['dt']}"
    solver_params = f"{config['solver']['linear']['solver']}__{config['solver']['contact']['CCD']['broad_phase']}__{config['solver']['nonlinear']['solver']}__{config['solver']['nonlinear']['line_search']['method']}"

    return f"{materials}__{contact_params}__{time_params}__{solver_params}"


def save_and_upload_results(client: Minio, config_path: Path, output_dir: Path, folder_name: str):
    """
    Save results locally and upload them to MinIO.

    Args:
        config_path (Path): Path to the simulation configuration file.
        output_dir (Path): Directory containing simulation results.
        folder_name (str): Folder name for the results.
    """
    bucket_path = f"{SIMULATION_BUCKET}/{folder_name}"

    # Upload all files in the output directory
    for result_file in output_dir.iterdir():
        object_name = f"{bucket_path}/{result_file.name}"
        upload_file_to_minio(client, SIMULATION_BUCKET, result_file, object_name)

    # Upload the simulation configuration file
    config_name = f"{bucket_path}/{config_path.name}"
    upload_file_to_minio(client, SIMULATION_BUCKET, config_path, config_name)

    logger.info(f"Uploaded all results to MinIO bucket: {SIMULATION_BUCKET}/{folder_name}")
