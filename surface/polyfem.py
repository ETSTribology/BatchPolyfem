#!/usr/bin/env python3
"""
run_polyfem_modified_typer.py

Usage Example (once installed Typer + Rich):
  python run_polyfem_modified_typer.py input_config.json output_folder --polyfem-bin /path/to/PolyFEM

Steps:
1) Use Typer to parse CLI (JSON file, output folder, optional PolyFEM binary).
2) Read and parse the original JSON config (log or review it).
3) Modify the loaded JSON (example: config["modified"] = True).
4) Save the modified JSON to a new file in the same directory (repo).
5) Run PolyFEM with the new JSON, placing outputs in 'output_folder'.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(help="Script to run PolyFEM with a modified JSON config using Typer + Rich.")

# Configure logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

# A Rich console (if you want to do non-logging prints)
console = Console()


def read_json_config(json_file: Path) -> dict:
    """
    Read and return the JSON configuration from 'json_file'.
    If the file is not valid or does not exist, exit.
    """
    if not json_file.is_file():
        logger.error(f"JSON file does not exist: {json_file}")
        raise typer.Exit(code=1)

    try:
        config_text = json_file.read_text()
        config = json.loads(config_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON {json_file}: {e}")
        raise typer.Exit(code=1)

    return config


def save_modified_config(config: dict, original_json: Path) -> Path:
    """
    Save the modified JSON config to a new file in the same directory.
    Returns the path to the new JSON file.
    """
    dir_name = original_json.parent
    base_name = original_json.stem  # filename without extension
    new_file_path = dir_name / f"{base_name}_modified.json"

    try:
        new_file_path.write_text(json.dumps(config, indent=2))
    except IOError as e:
        logger.error(f"Could not write modified JSON to {new_file_path}: {e}")
        raise typer.Exit(code=1)

    return new_file_path


def run_polyfem(json_file: Path, output_folder: Path, polyfem_bin: str):
    """
    Run the PolyFEM command with the given JSON file and output folder.
    """
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    cmd = [
        polyfem_bin,
        "--json", str(json_file),
        "--output", str(output_folder)
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"PolyFEM failed with exit code {e.returncode}.")
        raise typer.Exit(code=e.returncode)
    logger.info("PolyFEM run completed successfully.")


@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to the original JSON config file."),
    output_folder: Path = typer.Argument(..., help="Output folder where results will be placed."),
    polyfem_bin: str = typer.Option("polyfem", "--polyfem-bin", help="Path to the PolyFEM binary.")
):
    """
    Main Typer entry point:
    1) Reads original JSON
    2) Logs or modifies it
    3) Saves the modified JSON to a new file
    4) Calls PolyFEM
    """
    # 1) Read the original JSON
    original_config = read_json_config(json_file)
    
    # 2) Review/log the JSON config
    #    For example, log some keys or the entire config
    logger.info(f"Original config loaded. Keys: {list(original_config.keys())}")
    # You can also print the entire config if you want:
    # console.print(original_config)

    # 3) Modify the config
    original_config["modified"] = True
    logger.info("Modifying JSON config: added 'modified': True")

    # 4) Save the modified config
    new_json_file = save_modified_config(original_config, json_file)
    logger.info(f"Modified config saved to: {new_json_file}")

    # 5) Run PolyFEM with the new JSON
    run_polyfem(
        json_file=new_json_file,
        output_folder=output_folder,
        polyfem_bin=polyfem_bin
    )


if __name__ == "__main__":
    app()
