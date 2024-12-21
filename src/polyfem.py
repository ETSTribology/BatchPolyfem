import os
import json
import subprocess
import logging
import time
from typing import List, Dict, Optional
from jsonschema import validate, ValidationError, SchemaError


class PolyfemRunner:
    """
    A helper class to run Polyfem using a JSON configuration file.
    """
    def __init__(self, polyfem_path: str, logger: logging.Logger, mode: str):
        """
        Initialize the PolyfemRunner instance.
        
        :param polyfem_path: Path to the Polyfem executable or Docker image.
        :param logger: Logger instance for logging information.
        :param mode: Execution mode ('docker' or 'local').
        """
        self.polyfem_path = os.path.abspath(polyfem_path)
        self.logger = logger
        self.mode = mode.lower()
        if self.mode not in ['docker', 'local']:
            raise ValueError("Mode must be 'docker' or 'local'")
        self.logger.info(f"Initialized PolyfemRunner in {self.mode} mode with path: {self.polyfem_path}")

    def load_schema(self, schema_dir: str, schema_name: str) -> Dict:
        """
        Load the JSON schema from the specified directory.
        
        :param schema_dir: Directory containing the schema file.
        :param schema_name: Name of the schema file.
        :return: Loaded schema as a dictionary.
        """
        schema_path = os.path.join(schema_dir, schema_name)
        if not os.path.exists(schema_path):
            self.logger.error(f"Schema file does not exist: {schema_path}")
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
            self.logger.info(f"Successfully loaded schema from {schema_path}")
            return schema
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in schema file {schema_path}: {e}")
            raise

    def parse_and_validate_json(self, config_file: str, schema: Dict) -> Dict:
        """
        Parse and validate a JSON configuration file.
        
        :param config_file: Path to the configuration file.
        :param schema: JSON schema for validation.
        :return: Validated configuration data.
        """
        if not os.path.exists(config_file):
            self.logger.error(f"Configuration file not found: {config_file}")
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            validate(instance=config_data, schema=schema)
            self.logger.info(f"Configuration file {config_file} validated successfully.")
            return config_data
        except (ValidationError, SchemaError) as e:
            self.logger.error(f"JSON validation error: {e.message}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in {config_file}: {e}")
            raise

    def modify_config(self, config_data: Dict, updates: Dict) -> Dict:
        """
        Modify specific parameters in the configuration data.
        
        :param config_data: Original configuration data.
        :param updates: Updates to apply.
        :return: Updated configuration data.
        """
        for key, value in updates.items():
            if key in config_data:
                self.logger.info(f"Updating parameter '{key}' to '{value}'")
                config_data[key] = value
            else:
                self.logger.warning(f"Key '{key}' not found in the configuration. Skipping.")
        return config_data

    def save_config(self, config_data: Dict, config_file: str) -> None:
        """
        Save the modified configuration data to a file.
        
        :param config_data: Modified configuration data.
        :param config_file: Path to save the configuration file.
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """
        Helper method to run a subprocess command.
        
        :param command: Command to execute.
        :return: CompletedProcess instance with the results.
        """
        self.logger.debug(f"Executing command: {' '.join(command)}")
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            self.logger.debug(f"Command output: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with error: {e.stderr}")
            raise

    def run_simulation(self, config_file: str, schema: Dict, updates: Optional[Dict] = None) -> None:
        """
        Execute Polyfem simulation with the given configuration file.
        
        :param config_file: Path to the configuration file.
        :param schema: JSON schema for validation.
        :param updates: Optional updates to apply before execution.
        """
        # Validate and modify configuration
        config_data = self.parse_and_validate_json(config_file, schema)
        if updates:
            config_data = self.modify_config(config_data, updates)
            self.save_config(config_data, config_file)

        # Determine command based on execution mode
        if self.mode == "docker":
            command = ["docker", "run", "-v", f"{os.path.abspath(config_file)}:{config_file}", "polyfem", "--json", config_file]
        else:
            command = [self.polyfem_path, "--json", config_file]

        # Run simulation
        start_time = time.time()
        try:
            result = self._run_command(command)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Simulation completed successfully in {elapsed_time:.2f} seconds")
            self.logger.info(result.stdout)
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
