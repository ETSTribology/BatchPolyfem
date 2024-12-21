class PolyfemRunner:
    """
    A helper class to run Polyfem using a JSON configuration file.
    """
    def __init__(self, polyfem_path: str, logger, mode: str):
        self.polyfem_path = polyfem_path
        self.logger = logger
        self.mode = mode  # 'docker' or 'local'

    def parse_and_validate_json(self, config_file: str, schema: Dict):
        """
        Parses and validates the JSON configuration file against a schema.

        :param config_file: Path to the JSON configuration file.
        :param schema: JSON schema for validation.
        :return: Parsed JSON data.
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            validate(instance=config_data, schema=schema)
            self.logger.info("JSON configuration validated successfully.")
            return config_data
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in {config_file}: {e}")
            raise
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"JSON validation error: {e.message}")
            raise

    def modify_config(self, config_data: Dict, updates: Dict):
        """
        Modifies specific parameters in the configuration data.

        :param config_data: Original configuration data.
        :param updates: Dictionary of updates to apply to the configuration.
        :return: Updated configuration data.
        """
        try:
            for key, value in updates.items():
                if key in config_data:
                    self.logger.info(f"Updating parameter: {key} -> {value}")
                    config_data[key] = value
                else:
                    self.logger.warning(f"Key '{key}' not found in the configuration. Adding new key.")
                    config_data[key] = value
            return config_data
        except Exception as e:
            self.logger.error(f"An error occurred while modifying the configuration: {e}")
            raise

    def save_config(self, config_data: Dict, config_file: str):
        """
        Saves the modified configuration data back to a file.

        :param config_data: Modified configuration data.
        :param config_file: Path to the JSON configuration file.
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.logger.info(f"Configuration saved successfully to {config_file}.")
        except Exception as e:
            self.logger.error(f"An error occurred while saving the configuration: {e}")
            raise

    def run_simulation(self, config_file: str, schema: Dict, updates: Dict = None):
        """
        Executes Polyfem with the given configuration file and tracks execution time.

        :param config_file: Path to the JSON configuration file.
        :param schema: JSON schema for validation.
        :param updates: Optional dictionary of updates to modify in the configuration before execution.
        """
        import subprocess

        # Parse and validate JSON
        config_data = self.parse_and_validate_json(config_file, schema)

        # Modify configuration if updates are provided
        if updates:
            config_data = self.modify_config(config_data, updates)
            self.save_config(config_data, config_file)

        try:
            self.logger.info(f"Running Polyfem in {self.mode} mode with config: {config_file}")
            command = [self.polyfem_path, "--json", config_file]
            if self.mode == "docker":
                command = ["docker", "run", "-v", f"{config_file}:{config_file}", "polyfem", "--json", config_file]

            # Track start time
            start_time = time.time()

            # Execute Polyfem
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Track end time
            elapsed_time = time.time() - start_time
            self.logger.info(f"Execution time: {elapsed_time:.2f} seconds")

            if result.returncode == 0:
                self.logger.info(f"Polyfem simulation completed successfully:\n{result.stdout}")
            else:
                self.logger.error(f"Polyfem simulation failed with errors:\n{result.stderr}")
                raise RuntimeError("Polyfem simulation failed")

        except FileNotFoundError:
            self.logger.error("Polyfem executable not found at the specified path.")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred while running Polyfem: {e}")
            raise
