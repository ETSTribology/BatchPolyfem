from dataclasses import dataclass
import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

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

@dataclass
class SimulationConfig:
    materials: list
    contact_dhat: float
    contact_epsv: float
    contact_friction_coefficient: float
    use_convergent_formulation: bool
    time_integrator: str
    time_tend: float
    time_dt: float
    space_bc_method: str
    space_quadrature_order: int
    boundary_conditions: dict
    linear_solver: str
    nonlinear_solver: str
    nonlinear_line_search_method: str
    nonlinear_grad_norm: float
    nonlinear_max_iterations: int
    nonlinear_x_delta: float
    contact_friction_convergence_tol: int
    contact_friction_iterations: int
    contact_CCD_broad_phase: str


def parse_config(file_path):
    """
    Parses the given JSON configuration file and extracts relevant parameters.

    Args:
        file_path (str or Path): Path to the JSON configuration file.

    Returns:
        SimulationConfig: Extracted parameters as a dataclass.
    """
    # Ensure the file exists
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the JSON file
    with file_path.open("r") as f:
        config = json.load(f)

    # Extract materials
    materials = [
        {
            "type": mat["type"],
            "E": mat["E"],
            "nu": mat["nu"],
            "rho": mat["rho"]
        }
        for mat in config["materials"]
    ]

    # Extract other parameters
    contact = config["contact"]
    time = config["time"]
    space = config["space"]["advanced"]
    solver = config["solver"]

    boundary_conditions = config["boundary_conditions"]

    # Create and return the dataclass
    return SimulationConfig(
        materials=materials,
        contact_dhat=contact["dhat"],
        contact_epsv=contact["epsv"],
        contact_friction_coefficient=contact["friction_coefficient"],
        use_convergent_formulation=contact["use_convergent_formulation"],
        time_integrator=time["integrator"],
        time_tend=time["tend"],
        time_dt=time["dt"],
        space_bc_method=space["bc_method"],
        space_quadrature_order=space["quadrature_order"],
        boundary_conditions=boundary_conditions,
        linear_solver=solver["linear"]["solver"],
        nonlinear_solver=solver["nonlinear"]["solver"],
        nonlinear_line_search_method=solver["nonlinear"]["line_search"]["method"],
        nonlinear_grad_norm=solver["nonlinear"]["grad_norm"],
        nonlinear_max_iterations=solver["nonlinear"]["max_iterations"],
        nonlinear_x_delta=solver["nonlinear"]["x_delta"],
        contact_friction_convergence_tol=solver["contact"]["friction_convergence_tol"],
        contact_friction_iterations=solver["contact"]["friction_iterations"],
        contact_CCD_broad_phase=solver["contact"]["CCD"]["broad_phase"]
    )


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


def parse_config(file_path):
    """
    Parses the given JSON configuration file and extracts relevant parameters.

    Args:
        file_path (str or Path): Path to the JSON configuration file.

    Returns:
        SimulationConfig: Extracted parameters as a dataclass.
    """
    # Ensure the file exists
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the JSON file
    with file_path.open("r") as f:
        config = json.load(f)

    # Extract materials
    materials = [
        {
            "type": mat["type"],
            "E": mat["E"],
            "nu": mat["nu"],
            "rho": mat["rho"]
        }
        for mat in config["materials"]
    ]

    # Extract other parameters
    contact = config["contact"]
    time = config["time"]
    space = config["space"]["advanced"]
    solver = config["solver"]

    boundary_conditions = config["boundary_conditions"]

    # Create and return the dataclass
    return SimulationConfig(
        materials=materials,
        contact_dhat=contact["dhat"],
        contact_epsv=contact["epsv"],
        contact_friction_coefficient=contact["friction_coefficient"],
        use_convergent_formulation=contact["use_convergent_formulation"],
        time_integrator=time["integrator"],
        time_tend=time["tend"],
        time_dt=time["dt"],
        space_bc_method=space["bc_method"],
        space_quadrature_order=space["quadrature_order"],
        boundary_conditions=boundary_conditions,
        linear_solver=solver["linear"]["solver"],
        nonlinear_solver=solver["nonlinear"]["solver"],
        nonlinear_line_search_method=solver["nonlinear"]["line_search"]["method"],
        nonlinear_grad_norm=solver["nonlinear"]["grad_norm"],
        nonlinear_max_iterations=solver["nonlinear"]["max_iterations"],
        nonlinear_x_delta=solver["nonlinear"]["x_delta"],
        contact_friction_convergence_tol=solver["contact"]["friction_convergence_tol"],
        contact_friction_iterations=solver["contact"]["friction_iterations"],
        contact_CCD_broad_phase=solver["contact"]["CCD"]["broad_phase"]
    )

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

def run_simulation(config_path: Path, output_folder: Path, polyfem_bin: str = "polyfem_bin"):
    """
    Runs a single PolyFEM simulation using the specified JSON configuration file.

    Args:
        config_path (Path): Path to the JSON configuration file.
        output_folder (Path): Path to the folder where output files will be stored.
        polyfem_bin (str): Path to the PolyFEM binary executable (default: "polyfem_bin").
    """
    logger.info(f"Starting simulation for: {config_path}")
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise typer.Exit(code=1)

    # Run the PolyFEM simulation
    run_polyfem(config_path, output_folder, polyfem_bin)
    logger.info(f"Simulation completed for: {config_path}")


def modify_config_with_new_materials(
    original_config: Path,
    output_dir: Path,
    new_materials: List[dict],
    polyfem_bin: str = "polyfem_bin"
):
    """
    Modifies the configuration file to use new materials, saves it, and runs the simulation.

    Args:
        original_config (Path): Path to the original JSON configuration file.
        output_dir (Path): Directory to save output and modified config.
        new_materials (List[dict]): List of new material properties to replace the old ones.
        polyfem_bin (str): Path to the PolyFEM binary executable (default: "polyfem_bin").
    """
    # Load the original configuration
    config = read_json_config(original_config)

    # Modify the materials in the configuration
    config["materials"] = new_materials

    # Save the modified configuration
    modified_config_path = save_modified_config(config, original_config)

    # Run the simulation
    run_simulation(modified_config_path, output_dir, polyfem_bin)

def run_multiple_simulations(configs: List[Path], output_dir: Path):
    """
    Runs multiple simulations concurrently, each saving results to its own folder.

    Args:
        configs (List[Path]): List of JSON configuration file paths.
        output_dir (Path): Base directory for all simulation results.
    """
    from concurrent.futures import ThreadPoolExecutor

    def run_single_simulation(config_path: Path):
        simulation_output_dir = output_dir / config_path.stem
        run_simulation(config_path, simulation_output_dir)

    with ThreadPoolExecutor() as executor:
        executor.map(run_single_simulation, configs)

def modify_meshes_in_config(
    original_config: Path,
    output_dir: Path,
    new_mesh_paths: List[str],
    polyfem_bin: str = "polyfem_bin"
):
    """
    Modifies the configuration file to use new mesh files for each geometry, saves it, and runs the simulation.

    Args:
        original_config (Path): Path to the original JSON configuration file.
        output_dir (Path): Directory to save output and modified config.
        new_mesh_paths (List[str]): List of new mesh file paths to replace existing ones (one per geometry item).
        polyfem_bin (str): Path to the PolyFEM binary executable.
    """
    # Load the original configuration
    config = read_json_config(original_config)

    # Modify the mesh file paths for each geometry
    for idx, geometry_item in enumerate(config.get("geometry", [])):
        if idx < len(new_mesh_paths):
            geometry_item["mesh"] = new_mesh_paths[idx]
        else:
            logger.warning(f"No new mesh provided for geometry index {idx}. Skipping.")

    # Save the modified configuration
    modified_config_path = save_modified_config(config, original_config)

    # Run the simulation
    run_simulation(modified_config_path, output_dir, polyfem_bin)

