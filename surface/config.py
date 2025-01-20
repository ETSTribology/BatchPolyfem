# config.py

from typing import List, Dict
from dataclasses import dataclass, field
import logging
from rich.logging import RichHandler

# Import noise variations from the noises module
from noises import noise_variations
from storage import REQUIRED_BUCKETS  # Added import for noise variations

logging.basicConfig(
    level=logging.INFO,  # or INFO as needed
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

class BaseSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BaseSingleton, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, *args, **kwargs):
        """To be overridden by subclasses for initialization."""
        pass

class Config(BaseSingleton):
    def _initialize(self):
        self.minio_endpoint: str = "192.168.0.20:9000"
        self.minio_access_key: str = "minioadmin"
        self.minio_secret_key: str = "minioadmin"
        self.required_buckets: List[str] = REQUIRED_BUCKETS

        self.ftetwild_bin: str = "/home/antoine/code/fTetWild/build/FloatTetwild_bin"
        self.polyfem_bin: str = "/home/antoine/code/polyfem/build/PolyFEM_bin"
        self.hmm_bin: str = "hmm"

        self.map_sizes: List[int] = [8, 16, 32, 64, 128, 256, 512, 1024]

        # Automatically populate noise_types from available noise_variations keys
        self.noise_types: List[str] = list(noise_variations.keys())

        self.stl_params: Dict = {
            "z_value": 5.0,
            "error": "0.001",
            "base": 1,
            "z_exagg": 1.0,
        }
        self.tet_params: Dict = {
        }
        self.metadata_params: Dict = {
            "rebuild": False
        }

        self.ray_cluster_address: str = "auto"

        # Add noise variations to the configuration
        self.noise_variations = noise_variations

    def parse_arguments(self, minio_endpoint: str = None, ray_cluster_address: str = None):
        if minio_endpoint:
            self.minio_endpoint = minio_endpoint
        if ray_cluster_address:
            self.ray_cluster_address = ray_cluster_address

# Singleton instance of the configuration
config = Config()

