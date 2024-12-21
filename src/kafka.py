import requests
from requests.auth import HTTPBasicAuth
import threading
import logging
from rich.console import Console
from rich.logging import RichHandler

class KafkaConnect(metaclass=SingletonMeta):
    """
    Singleton class to interact with Kafka Connect REST API.
    """
    def __init__(self, config_dict: dict):
        """
        Initializes the KafkaConnect instance with the provided configuration.
        
        :param config_dict: Dictionary containing configuration parameters.
        """
        self.config = config_dict
        self.base_url = self.config.get("url", "http://localhost:8083")
        self.auth = None
        if "username" in self.config and "password" in self.config:
            self.auth = HTTPBasicAuth(self.config["username"], self.config["password"])
        self.verify_ssl = self.config.get("verify_ssl", True)
        self.timeout = self.config.get("timeout", 30)
        
        # Setup Rich Logging
        self.console = Console()
        self.logger = logging.getLogger("KafkaConnect")
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.DEBUG if self.config.get("verbose", False) else logging.INFO,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(console=self.console)]
            )
        
    def _request(self, method: str, endpoint: str, **kwargs):
        """
        Internal method to make HTTP requests to the Kafka Connect REST API.
        
        :param method: HTTP method (GET, POST, DELETE, etc.).
        :param endpoint: API endpoint (e.g., "/connectors").
        :param kwargs: Additional arguments for the requests.request method.
        :return: Response object.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(
                method,
                url,
                auth=self.auth,
                verify=self.verify_ssl,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err} - {response.text}")
            raise
        except requests.exceptions.RequestException as err:
            self.logger.error(f"Request exception: {err}")
            raise

    def list_connectors(self) -> list:
        """
        Retrieves a list of all connectors.
        
        :return: List of connector names.
        """
        self.logger.debug("Fetching list of connectors.")
        response = self._request("GET", "/connectors")
        connectors = response.json()
        self.logger.info(f"Found connectors: {connectors}")
        return connectors

    def get_connector_config(self, connector_name: str) -> dict:
        """
        Retrieves the configuration of a specific connector.
        
        :param connector_name: Name of the connector.
        :return: Configuration dictionary.
        """
        self.logger.debug(f"Fetching configuration for connector: {connector_name}")
        response = self._request("GET", f"/connectors/{connector_name}/config")
        config = response.json()
        self.logger.info(f"Configuration for '{connector_name}': {config}")
        return config

    def create_connector(self, connector_name: str, config: dict) -> dict:
        """
        Creates a new connector with the given configuration.
        
        :param connector_name: Name of the new connector.
        :param config: Configuration dictionary for the connector.
        :return: Response JSON.
        """
        self.logger.debug(f"Creating connector: {connector_name}")
        payload = {
            "name": connector_name,
            "config": config
        }
        response = self._request("POST", "/connectors", json=payload)
        result = response.json()
        self.logger.info(f"Connector '{connector_name}' created successfully.")
        return result

    def delete_connector(self, connector_name: str) -> dict:
        """
        Deletes a connector.
        
        :param connector_name: Name of the connector to delete.
        :return: Response JSON.
        """
        self.logger.debug(f"Deleting connector: {connector_name}")
        response = self._request("DELETE", f"/connectors/{connector_name}")
        result = response.json()
        self.logger.info(f"Connector '{connector_name}' deleted successfully.")
        return result

    def get_connector_status(self, connector_name: str) -> dict:
        """
        Retrieves the status of a specific connector.
        
        :param connector_name: Name of the connector.
        :return: Status dictionary.
        """
        self.logger.debug(f"Fetching status for connector: {connector_name}")
        response = self._request("GET", f"/connectors/{connector_name}/status")
        status = response.json()
        self.logger.info(f"Status for '{connector_name}': {status}")
        return status

    def update_connector_config(self, connector_name: str, new_config: dict) -> dict:
        """
        Updates the configuration of an existing connector.
        
        :param connector_name: Name of the connector.
        :param new_config: New configuration dictionary.
        :return: Response JSON.
        """
        self.logger.debug(f"Updating configuration for connector: {connector_name}")
        response = self._request("PUT", f"/connectors/{connector_name}/config", json=new_config)
        result = response.json()
        self.logger.info(f"Configuration for '{connector_name}' updated successfully.")
        return result
