#!/usr/bin/env python3
"""
generate.py

Enhanced version using Ray for distributed processing with separate actors for
each step: noise generation, STL conversion, and tetrahedral mesh creation.
Uses Ray queues to parallelize and decouple processing stages, with task chunking
for improved concurrency. Additionally:
  - Generates both normal & inverted displacement,
  - Creates STL & tetrahedral mesh for both.
"""

import asyncio
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import logging
from tqdm import tqdm
import typer
import tempfile
import shutil
from rich.logging import RichHandler
import ray
from ray.util import ActorPool
from ray.util.queue import Queue

from cluster import init_ray_cluster, shutdown_ray_cluster
from storage import (
    check_file_exists_in_minio,
    get_bucket_name,
    setup_minio_client,
    upload_file_to_minio,
    download_scan_image,
    create_noise_folder,
    build_displacement_filename,
    build_stl_filename,
    build_tetrahedral_filename,
    Buckets,
    temp_directory
)
from displacement import generate_displacement_map, invert_displacement_map, save_displacement_map
from noises import noise_variations
from hmm import run_hmm_sync, run_hmm_ray_task_sync
from tetrahedron import run_ftetwild
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
app = typer.Typer()

# Initialize queues for inter-stage communication
stl_queue = Queue()
tet_queue = Queue()
stats_queue = Queue()
simulation_parameters_queue = Queue()
simulation_results_queue = Queue()
simulation_analysis_queue = Queue()



def is_inverted_object_name(obj_name: str) -> bool:
    """
    Checks if an object name has '_inverted' in the final path stem,
    e.g. "bar_inverted.png".
    """
    return "_inverted" in Path(obj_name).stem


@ray.remote
class NoiseActor:
    """
    Ray Actor for generating displacement (normal + inverted) 
    and uploading them to DISPLACEMENT bucket if they don't already exist.
    """
    def __init__(self):
        self.client = setup_minio_client(config.required_buckets)

    def process_noise(self, noise_name: str, noise_config: Tuple[Any, Dict], map_size: int):
        """
        1) Build the normal and inverted displacement filenames
        2) If not in MinIO, generate & upload
        3) Return a tuple of (normal_disp_obj_name, inverted_disp_obj_name) or None on failure
        """
        # noise_config is (noise_fn, {param_dict})
        noise_fn, noise_params = noise_config

        object_name = build_displacement_filename(noise_name, noise_params, inverse=False, map_size=map_size)
        object_name_inverted = build_displacement_filename(noise_name, noise_params, inverse=True,  map_size=map_size)

        # If normal displacement file already exists in MinIO, skip
        if check_file_exists_in_minio(self.client, Buckets.DISPLACEMENT, object_name) and check_file_exists_in_minio(self.client, Buckets.DISPLACEMENT, object_name_inverted):
            logger.debug(f"Displacement map already exists: {object_name}")
        else:
            try:
                # Generate displacement
                disp_map = generate_displacement_map([noise_fn], map_size, [noise_params], True)
                disp_map_inverted = invert_displacement_map(disp_map)

                # Create local paths under a noise folder (temp)
                noise_folder = create_noise_folder(noise_name)
                normal_path = noise_folder / f"{map_size}.npy"
                inverted_path = noise_folder / f"{map_size}_inverted.npy"

                save_displacement_map(disp_map, str(normal_path))
                save_displacement_map(disp_map_inverted, str(inverted_path))

                # Upload
                upload_file_to_minio(self.client, Buckets.DISPLACEMENT, str(normal_path), object_name)
                upload_file_to_minio(self.client, Buckets.DISPLACEMENT, str(inverted_path), object_name_inverted)

                # Clean up
                normal_path.unlink(missing_ok=True)
                inverted_path.unlink(missing_ok=True)

            except Exception as e:
                logger.error(f"Error generating noise for {noise_name}, size {map_size}: {e}")
                return None

        return (object_name, object_name_inverted)

@ray.remote
class STLActor:
    """
    Ray Actor for converting displacement (normal or inverted) to STL.
    """
    def __init__(self):
        self.client = setup_minio_client(config.required_buckets)

    def process_stl(self, noise_name: str, disp_object: str, noise_params: Dict, map_size: int):
        """
        1) Build STL filename.
        2) If not in MinIO, download displacement, generate STL, upload.
        3) Return stl_object_name or None on failure.
        """
        # Is it an inverted displacement object?
        inverse = is_inverted_object_name(disp_object)

        stl_object_name = build_stl_filename(
            noise_name,
            noise_params,
            "hmm",
            config.stl_params,
            inverse=False,
            map_size=map_size)
        
        # Check if STL already exists in MinIO
        stl_object_name_inverted = build_stl_filename(
            noise_name,
            noise_params,
            "hmm",
            config.stl_params,
            inverse=True,
            map_size=map_size)

        if check_file_exists_in_minio(self.client, Buckets.STL, stl_object_name) and check_file_exists_in_minio(self.client, Buckets.STL, stl_object_name_inverted):
            logger.debug(f"STL already exists: {stl_object_name}")
            return stl_object_name

        # Download displacement and create STL
        with temp_directory() as tmpdir:
            local_disp_path = Path(tmpdir) / Path(disp_object).name
            download_scan_image(self.client, Buckets.DISPLACEMENT.value, disp_object, str(local_disp_path))

            stl_path = Path(tmpdir) / "output.stl"
            try:
                run_hmm_sync(str(local_disp_path), str(stl_path), **config.stl_params)
                if not stl_path.exists():
                    logger.error(f"STL was not created at {stl_path}")
                    return None

                # Upload
                upload_file_to_minio(self.client, Buckets.STL, str(stl_path), stl_object_name)
                upload_file_to_minio(self.client, Buckets.STL, str(stl_path), stl_object_name_inverted)
                return stl_object_name

            except Exception as e:
                logger.error(f"Error creating STL for {disp_object}: {e}")
                return None

@ray.remote
class TetActor:
    """
    Ray Actor for converting STL to Tetrahedral mesh.
    """
    def __init__(self):
        self.client = setup_minio_client(config.required_buckets)

    def process_tetrahedral(
        self,
        noise_name: str,
        stl_object: str,
        noise_params: Dict,
        map_size: int
    ):
        """
        1) Build .msh filename.
        2) If not in MinIO, download STL, generate .msh, upload.
        3) Return tetra_object_name or None on failure.
        """
        # Check if it's inverted

        tetra_object_name = build_tetrahedral_filename(
            noise_name=noise_name,
            noise_params=noise_params,
            program_stl="hmm",
            stl_params=config.stl_params,
            program_tet="ftetwild",
            tet_params=config.tet_params,
            inverse=False,
            map_size=map_size
        )
        
        tetra_object_name_inverted = build_tetrahedral_filename(
            noise_name=noise_name,
            noise_params=noise_params,
            program_stl="hmm",
            stl_params=config.stl_params,
            program_tet="ftetwild",
            tet_params=config.tet_params,
            inverse=True,
            map_size=map_size
        )

        if check_file_exists_in_minio(self.client, Buckets.TETRAHEDRAL, tetra_object_name) and check_file_exists_in_minio(self.client, Buckets.TETRAHEDRAL, tetra_object_name_inverted):
            logger.debug(f"Tetra mesh already exists: {tetra_object_name}")
            return tetra_object_name

        with temp_directory() as tmpdir:
            local_stl_path = Path(tmpdir) / Path(stl_object).name
            download_scan_image(self.client, Buckets.STL.value, stl_object, str(local_stl_path))

            output_dir = Path(tmpdir) / "tet_output"
            output_dir.mkdir(exist_ok=True)

            try:
                run_ftetwild(str(local_stl_path), str(output_dir), **config.tet_params)
                msh_path = next(output_dir.glob("*.msh"), None)
                if not msh_path or not msh_path.exists():
                    raise FileNotFoundError("No .msh file found after tetrahedral generation.")

                # Upload
                upload_file_to_minio(self.client, Buckets.TETRAHEDRAL, str(msh_path), tetra_object_name)
                return tetra_object_name

            except Exception as e:
                logger.error(f"Error creating tetrahedral mesh: {e}")
                return None

@app.command("all")
def generate_all(ray_cluster_address: Optional[str] = typer.Option(None, help="Address of the Ray cluster")):
    """
    Generates displacement maps, converts them to STL, and creates tetrahedral meshes using Ray actors.
    """
    init_ray_cluster(address=ray_cluster_address or "auto")

    # Create actor pools for each stage
    noise_pool = ActorPool([NoiseActor.remote() for _ in range(4)])
    stl_pool   = ActorPool([STLActor.remote() for _ in range(4)])
    tet_pool   = ActorPool([TetActor.remote() for _ in range(4)])

    async def main():
        #######################################################################
        # Stage 1: Noise Generation
        #######################################################################
        noise_tasks = [
            (noise_name, noise_config, size)
            for noise_name in config.noise_types
            for noise_config in noise_variations[noise_name]
            for size in config.map_sizes
        ]
        total_noise_tasks = len(noise_tasks)
        noise_pbar = tqdm(total=total_noise_tasks, desc="Displacement")

        # Process all noise tasks
        for result in noise_pool.map(
            lambda actor, args: actor.process_noise.remote(*args),
            noise_tasks
        ):
            (noise_name, noise_config, size) = noise_tasks.pop(0)
            # If successful, 'result' is (obj_name, inverted_obj_name) or None
            if result:
                (normal_obj, inverted_obj) = result
                # Enqueue them for STL processing
                stl_queue.put((noise_name, normal_obj,  noise_config[1], size))
                stl_queue.put((noise_name, inverted_obj, noise_config[1], size))
            else:
                logger.warning(f"Failed noise processing for {noise_name} at size {size}.")
            noise_pbar.update(1)
        noise_pbar.close()

        #######################################################################
        # Stage 2: STL Conversion
        #######################################################################
        stl_tasks = []
        while not stl_queue.empty():
            stl_tasks.append(stl_queue.get_nowait())

        total_stl_tasks = len(stl_tasks)
        stl_pbar = tqdm(total=total_stl_tasks, desc="STL Conversion")

        # stl_tasks = [(noise_name, disp_object, noise_params, map_size), ...]
        for stl_result in stl_pool.map(
            lambda actor, args: actor.process_stl.remote(*args),
            stl_tasks
        ):
            (noise_name, disp_object, noise_params, size) = stl_tasks.pop(0)
            if stl_result:
                # Enqueue for Tetrahedral
                tet_queue.put((noise_name, stl_result, noise_params, size))
            else:
                logger.warning(f"Failed STL processing for {noise_name} at size {size}.")
            stl_pbar.update(1)
        stl_pbar.close()

        #######################################################################
        # Stage 3: Tetrahedral Mesh Creation
        #######################################################################
        tet_tasks = []
        while not tet_queue.empty():
            tet_tasks.append(tet_queue.get_nowait())

        total_tet_tasks = len(tet_tasks)
        tet_pbar = tqdm(total=total_tet_tasks, desc="Tetrahedral Mesh")

        # tet_tasks = [(noise_name, stl_object, noise_params, map_size), ...]
        for _ in tet_pool.map(
            lambda actor, args: actor.process_tetrahedral.remote(*args),
            tet_tasks
        ):
            tet_pbar.update(1)
        tet_pbar.close()

    asyncio.run(main())
    shutdown_ray_cluster()


# Entry point for CLI
if __name__ == "__main__":
    app()