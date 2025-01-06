# storage.py

#!/usr/bin/env python3
"""
storage.py

All MinIO-related logic is placed here:
- Setting up a MinIO client
- Ensuring a bucket exists
- Uploading files
"""

import logging
from minio import Minio, S3Error
from rich.console import Console
from rich.logging import RichHandler

# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
console = Console()


def setup_minio_client(
    endpoint: str = "192.168.0.20:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    secure: bool = False,
    bucket_name: str = "ftetwild"  # Default bucket name
) -> tuple:
    """
    Returns a MinIO client object and the bucket name.
    Ensures the bucket exists.
    """
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        logger.info(f"[MinIO] Created bucket: '{bucket_name}'")
    else:
        logger.info(f"[MinIO] Bucket '{bucket_name}' already exists.")

    return client, bucket_name


def upload_file_to_minio(
    client: Minio,
    bucket_name: str,
    file_path: str,
    object_name: str
) -> None:
    """
    Uploads a local file (file_path) to MinIO under the given bucket/object name.
    """
    try:
        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path
        )
        logger.info(f"[MinIO] Uploaded '{file_path}' to '{bucket_name}/{object_name}'")
    except S3Error as s3e:
        logger.error(f"[MinIO] Upload failed for {file_path}: {s3e}")
        raise


def ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    """
    Ensures the specified bucket exists; creates it if it doesn't.
    """
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        logger.info(f"[MinIO] Created bucket: '{bucket_name}'")
    else:
        logger.info(f"[MinIO] Bucket '{bucket_name}' already exists.")
