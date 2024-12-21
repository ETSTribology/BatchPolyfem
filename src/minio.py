import threading
import time
import jsonschema
from typing import List, Dict
from kafka import KafkaProducer, KafkaConsumer
from minio import Minio
from minio.error import S3Error
from pymongo import MongoClient, errors
import json
import logging
from rich.console import Console
from rich.logging import RichHandler


class MinIOConnect(metaclass=SingletonMeta):
    """
    Singleton class to interact with MinIO server.
    """
    def __init__(self, config_dict: dict):
        """
        Initializes the MinIOConnect instance with the provided configuration.
        
        :param config_dict: Dictionary containing configuration parameters.
        """
        self.config = config_dict
        self.console = Console()
        self.logger = self.setup_logging(self.config.get("verbose", False))
        
        try:
            self.client = Minio(
                endpoint=self.config["endpoint"],
                access_key=self.config["access_key"],
                secret_key=self.config["secret_key"],
                secure=self.config.get("secure", True),
                region=self.config.get("region", "us-east-1"),
                session_token=self.config.get("session_token"),
                http_client=None,  # Use default http client
            )
            self.logger.info("Successfully connected to MinIO server.")
        except Exception as e:
            self.logger.error(f"Failed to connect to MinIO server: {e}")
            raise

    def setup_logging(self, verbose: bool):
        """
        Sets up logging configuration with RichHandler.
        If verbose is True, set log level to DEBUG, else INFO.
        """
        log_level = logging.DEBUG if verbose else logging.INFO

        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console)]
        )
        logger = logging.getLogger("MinIOConnect")
        return logger

    def list_buckets(self) -> list:
        """
        Lists all buckets in the MinIO server.
        
        :return: List of bucket names.
        """
        self.logger.debug("Listing all buckets.")
        try:
            buckets = self.client.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            self.logger.info(f"Buckets: {bucket_names}")
            return bucket_names
        except S3Error as e:
            self.logger.error(f"Error listing buckets: {e}")
            raise

    def create_bucket(self, bucket_name: str, region: str = None):
        """
        Creates a new bucket in the MinIO server.
        
        :param bucket_name: Name of the bucket to create.
        :param region: (Optional) Region for the bucket.
        """
        self.logger.debug(f"Creating bucket: {bucket_name}")
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name, location=region)
                self.logger.info(f"Bucket '{bucket_name}' created successfully.")
            else:
                self.logger.warning(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            self.logger.error(f"Error creating bucket '{bucket_name}': {e}")
            raise

    def delete_bucket(self, bucket_name: str):
        """
        Deletes a bucket from the MinIO server.
        The bucket must be empty before deletion.
        
        :param bucket_name: Name of the bucket to delete.
        """
        self.logger.debug(f"Deleting bucket: {bucket_name}")
        try:
            if self.client.bucket_exists(bucket_name):
                # Ensure the bucket is empty
                objects = self.client.list_objects(bucket_name, recursive=True)
                for obj in objects:
                    self.client.remove_object(bucket_name, obj.object_name)
                self.client.remove_bucket(bucket_name)
                self.logger.info(f"Bucket '{bucket_name}' deleted successfully.")
            else:
                self.logger.warning(f"Bucket '{bucket_name}' does not exist.")
        except S3Error as e:
            self.logger.error(f"Error deleting bucket '{bucket_name}': {e}")
            raise

    def upload_object(self, bucket_name: str, object_name: str, file_path: str, content_type: str = "application/octet-stream"):
        """
        Uploads an object to a specified bucket.
        
        :param bucket_name: Name of the bucket.
        :param object_name: Name of the object in the bucket.
        :param file_path: Path to the file to upload.
        :param content_type: MIME type of the object.
        """
        self.logger.debug(f"Uploading object '{object_name}' to bucket '{bucket_name}'.")
        try:
            self.client.fput_object(bucket_name, object_name, file_path, content_type=content_type)
            self.logger.info(f"Object '{object_name}' uploaded successfully to bucket '{bucket_name}'.")
        except S3Error as e:
            self.logger.error(f"Error uploading object '{object_name}' to bucket '{bucket_name}': {e}")
            raise

    def download_object(self, bucket_name: str, object_name: str, file_path: str):
        """
        Downloads an object from a specified bucket.
        
        :param bucket_name: Name of the bucket.
        :param object_name: Name of the object in the bucket.
        :param file_path: Path to save the downloaded file.
        """
        self.logger.debug(f"Downloading object '{object_name}' from bucket '{bucket_name}'.")
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            self.logger.info(f"Object '{object_name}' downloaded successfully from bucket '{bucket_name}'.")
        except S3Error as e:
            self.logger.error(f"Error downloading object '{object_name}' from bucket '{bucket_name}': {e}")
            raise

    def list_objects(self, bucket_name: str) -> list:
        """
        Lists all objects in a specified bucket.
        
        :param bucket_name: Name of the bucket.
        :return: List of object names.
        """
        self.logger.debug(f"Listing objects in bucket '{bucket_name}'.")
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            object_names = [obj.object_name for obj in objects]
            self.logger.info(f"Objects in bucket '{bucket_name}': {object_names}")
            return object_names
        except S3Error as e:
            self.logger.error(f"Error listing objects in bucket '{bucket_name}': {e}")
            raise

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """
        Checks if an object exists in a specified bucket.
        
        :param bucket_name: Name of the bucket.
        :param object_name: Name of the object.
        :return: True if the object exists, False otherwise.
        """
        self.logger.debug(f"Checking existence of object '{object_name}' in bucket '{bucket_name}'.")
        try:
            exists = self.client.stat_object(bucket_name, object_name) is not None
            self.logger.info(f"Object '{object_name}' exists in bucket '{bucket_name}': {exists}")
            return exists
        except S3Error as e:
            if e.code == "NoSuchKey":
                self.logger.info(f"Object '{object_name}' does not exist in bucket '{bucket_name}'.")
                return False
            else:
                self.logger.error(f"Error checking object '{object_name}' in bucket '{bucket_name}': {e}")
                raise

    def delete_object(self, bucket_name: str, object_name: str):
        """
        Deletes an object from a specified bucket.
        
        :param bucket_name: Name of the bucket.
        :param object_name: Name of the object to delete.
        """
        self.logger.debug(f"Deleting object '{object_name}' from bucket '{bucket_name}'.")
        try:
            self.client.remove_object(bucket_name, object_name)
            self.logger.info(f"Object '{object_name}' deleted successfully from bucket '{bucket_name}'.")
        except S3Error as e:
            self.logger.error(f"Error deleting object '{object_name}' from bucket '{bucket_name}': {e}")
            raise

    def make_bucket_if_not_exists(self, bucket_name: str, region: str = None):
        """
        Creates a bucket if it does not already exist.
        
        :param bucket_name: Name of the bucket.
        :param region: (Optional) Region for the bucket.
        """
        self.logger.debug(f"Ensuring bucket '{bucket_name}' exists.")
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name, location=region)
                self.logger.info(f"Bucket '{bucket_name}' created successfully.")
            else:
                self.logger.info(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            self.logger.error(f"Error ensuring bucket '{bucket_name}' exists: {e}")
            raise


class MinioEventConsumer:
    """
    A class to consume events about uploads to MinIO and handle them.
    """
    def __init__(self, kafka_consumer: KafkaConsumer, minio_client: Minio, bucket_name: str, logger: logging.Logger, mongo_client: MongoClient):
        self.kafka_consumer = kafka_consumer
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.logger = logger
        self.mongo_client = mongo_client

    def handle_event(self, event_data: dict):
        """
        Handles an upload completion event.

        :param event_data: Event data from Kafka.
        """
        object_name = event_data.get("object_name")
        if not object_name:
            self.logger.error("Invalid event data: missing 'object_name'.")
            return

        try:
            # Process the uploaded object
            self.logger.info(f"Processing completed upload: {object_name}")

            # Interact with MongoDB to save the event data
            db = self.mongo_client["minio_events"]
            collection = db["uploads"]
            collection.insert_one(event_data)
            self.logger.info(f"Event data saved to MongoDB: {event_data}")

            # Example: list objects in the bucket
            objects = self.minio_client.list_objects(self.bucket_name)
            for obj in objects:
                self.logger.info(f"Found object: {obj.object_name}")

        except S3Error as e:
            self.logger.error(f"Error processing MinIO event: {e}")
        except errors.PyMongoError as e:
            self.logger.error(f"Error interacting with MongoDB: {e}")

    def consume_events(self):
        """
        Consumes events from Kafka and handles them.
        """
        self.logger.info("Starting to consume events from Kafka.")
        for message in self.kafka_consumer:
            try:
                event_data = json.loads(message.value)
                self.handle_event(event_data)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode Kafka message: {e}")
            except Exception as e:
                self.logger.error(f"Error handling Kafka message: {e}")

