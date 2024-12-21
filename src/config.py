import os
from dotenv import load_dotenv

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            print("Creating new Config instance")
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.settings = {}
            cls._instance.load_settings()
        else:
            print("Using existing Config instance")
        return cls._instance

    def load_settings(self):
        load_dotenv()

        self.settings = {
            # General Settings
            'database_url': os.getenv('DATABASE_URL', 'localhost:5432'),
            'api_key': os.getenv('API_KEY', '1234567890abcdef'),
            'debug': os.getenv('DEBUG', 'True').lower() in ['true', '1', 't'],

            # MinIO Environment Variables
            'MINIO_ROOT_USER': os.getenv('MINIO_ROOT_USER', 'minioadmin'),
            'MINIO_ROOT_PASSWORD': os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin'),
            'MINIO_PROMETHEUS_AUTH_TYPE': os.getenv('MINIO_PROMETHEUS_AUTH_TYPE', 'public'),

            # Zookeeper Environment Variables
            'ZOOKEEPER_CLIENT_PORT': int(os.getenv('ZOOKEEPER_CLIENT_PORT', '2181')),
            'ZOOKEEPER_TICK_TIME': int(os.getenv('ZOOKEEPER_TICK_TIME', '2000')),

            # Kafka Environment Variables
            'KAFKA_BROKER_ID': int(os.getenv('KAFKA_BROKER_ID', '1')),
            'KAFKA_ZOOKEEPER_CONNECT': os.getenv('KAFKA_ZOOKEEPER_CONNECT', 'zookeeper:2181'),
            'KAFKA_ADVERTISED_LISTENERS': os.getenv('KAFKA_ADVERTISED_LISTENERS', 'PLAINTEXT://localhost:9092'),
            'KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR': int(os.getenv('KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR', '1')),

            # MongoDB Environment Variables
            'MONGO_INITDB_ROOT_USERNAME': os.getenv('MONGO_INITDB_ROOT_USERNAME', 'root'),
            'MONGO_INITDB_ROOT_PASSWORD': os.getenv('MONGO_INITDB_ROOT_PASSWORD', 'example'),

            # Qdrant Environment Variables
            'QDRANT__SERVICE__HTTP_PORT': int(os.getenv('QDRANT__SERVICE__HTTP_PORT', '6333')),
            'QDRANT__COLLECTIONS__DEFAULT__NAME': os.getenv('QDRANT__COLLECTIONS__DEFAULT__NAME', 'default'),

            # Grafana Environment Variables
            'GF_SECURITY_ADMIN_USER': os.getenv('GF_SECURITY_ADMIN_USER', 'admin'),
            'GF_SECURITY_ADMIN_PASSWORD': os.getenv('GF_SECURITY_ADMIN_PASSWORD', 'admin'),
        }

    def get_setting(self, key):
        return self.settings.get(key)

    def set_setting(self, key, value):
        self.settings[key] = value
