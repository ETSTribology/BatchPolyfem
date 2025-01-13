#!/usr/bin/env python3
"""
generate.py

Enhanced version using Ray for distributed processing with separate actors for
each step: noise generation, STL conversion, and tetrahedral mesh creation.
Uses Ray queues to parallelize and decouple processing stages, with task chunking for improved concurrency.
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
    Buckets
)
from displacement import generate_displacement_map, save_displacement_map
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

@ray.remote
class NoiseActor:
    def __init__(self):
        self.client = setup_minio_client(bucket_names=config.required_buckets)

    def process_noise(self, noise_name: str, noise_config: Tuple[Any, Dict], map_size: int) -> Optional[str]:
        noise_folder = create_noise_folder(noise_name)
        object_name = build_displacement_filename(noise_name, noise_config[1], map_size)

        if check_file_exists_in_minio(self.client, Buckets.DISPLACEMENT, object_name):
            logger.debug(f"Displacement map already exists: {object_name}. Skipping generation.")
            return object_name

        try:
            disp_map = generate_displacement_map([noise_config[0]], map_size, [noise_config[1]], True)
            png_path = noise_folder / f"{map_size}.png"
            save_displacement_map(disp_map, str(png_path))

            if not png_path.exists():
                raise FileNotFoundError(f"Displacement map not created: {png_path}")

            logger.debug(f"Uploading displacement map to 'displacement' bucket: {object_name}")
            upload_file_to_minio(self.client, Buckets.DISPLACEMENT, str(png_path), object_name)

            return object_name
        except Exception as e:
            logger.error(f"Error in process_noise: {e}")
            return None
        finally:
            if png_path and png_path.exists():
                png_path.unlink()

@ray.remote
class STLActor:
    def __init__(self):
        self.client = setup_minio_client(bucket_names=[bucket.value for bucket in Buckets])

    def process_stl(self, noise_name: str, displacement_object: str, noise_params: dict, stl_params: dict, map_size: int) -> Optional[str]:
        stl_object_name = build_stl_filename(noise_name, noise_params[1], "hmm", stl_params, map_size)
        if check_file_exists_in_minio(self.client, Buckets.STL.value, stl_object_name):
            return stl_object_name
        tmpdir = Path(tempfile.mkdtemp())
        try:
            local_path = tmpdir / Path(displacement_object).name
            logger.debug(f"Downloading displacement {displacement_object} to {local_path}")
            download_scan_image(self.client, Buckets.DISPLACEMENT.value, displacement_object, str(local_path))
            stl_path = tmpdir / f"{local_path.stem}.stl"
            logger.info(f"Running HMM for {local_path} to {stl_path} with parameters {stl_params}")
            run_hmm_sync(str(local_path), str(stl_path), **stl_params)
            if not stl_path.exists():
                logger.error(f"Expected STL not found at {stl_path}")
                return None
            logger.debug(f"Uploading STL {stl_path} to bucket 'stl' as {stl_object_name}")
            upload_file_to_minio(self.client, Buckets.STL.value, str(stl_path), stl_object_name)
            return stl_object_name
        except Exception as e:
            logger.error(f"Error in process_stl: {e}")
            return None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

@ray.remote
class TetActor:
    def __init__(self):
        self.client = setup_minio_client(bucket_names=config.required_buckets)

    def process_tetrahedral(
        self,
        noise_name: str,
        stl_object: str,
        noise_params: dict,
        stl_params: dict,
        program_stl: str,
        program_tet: str,
        tet_params: dict,
        map_size: int,
    ) -> Optional[str]:
        tetra_object_name = build_tetrahedral_filename(
            noise_name=noise_name,
            noise_params=noise_params[1],
            program_stl=program_stl,
            stl_params=stl_params,
            program_tet=program_tet,
            tet_params=tet_params,
            map_size=map_size,
        )
        bucket_name = get_bucket_name(Buckets.TETRAHEDRAL)

        if check_file_exists_in_minio(self.client, bucket_name, tetra_object_name):
            logger.debug(f"Tetrahedral mesh already exists: {tetra_object_name}. Skipping generation.")
            return tetra_object_name

        tmpdir = tempfile.mkdtemp()
        try:
            stl_bucket_name = get_bucket_name(Buckets.STL)
            local_stl_path = Path(tmpdir) / Path(stl_object).name
            download_scan_image(self.client, stl_bucket_name, stl_object, str(local_stl_path))

            output_dir = Path(tmpdir) / "tet_output"
            output_dir.mkdir(exist_ok=True)

            run_ftetwild(str(local_stl_path), str(output_dir), **tet_params)

            msh_path = next(output_dir.glob("*.msh"), None)
            if not msh_path or not msh_path.exists():
                raise FileNotFoundError(f"Tetrahedral mesh not created in: {output_dir}")

            upload_file_to_minio(self.client, bucket_name, str(msh_path), tetra_object_name)
            return tetra_object_name
        except Exception as e:
            logger.error(f"Error in process_tetrahedral: {e}")
            return None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

@app.command("all")
def generate_all(ray_cluster_address: Optional[str] = typer.Option(None, help="Address of the Ray cluster")):
    """
    Generates displacement maps, converts them to STL, and creates tetrahedral meshes using Ray actors.
    """
    init_ray_cluster(address=ray_cluster_address or "auto")

    # Create actor pools for each stage
    noise_pool = ActorPool([NoiseActor.remote() for _ in range(4)])
    stl_pool = ActorPool([STLActor.remote() for _ in range(4)])
    tet_pool = ActorPool([TetActor.remote() for _ in range(4)])

    async def main():
        # Stage 1: Noise Generation with task chunking
        noise_tasks = [
            (noise_name, noise_config, size)
            for noise_name in config.noise_types
            for noise_config in noise_variations[noise_name]
            for size in config.map_sizes
        ]
        total_noise_tasks = len(noise_tasks)
        noise_pbar = tqdm(total=total_noise_tasks, desc="Displacement")

        # Define a chunk size for splitting tasks
        chunk_size = 100
        task_chunks = [noise_tasks[i:i + chunk_size] for i in range(0, total_noise_tasks, chunk_size)]

        async def process_chunk(chunk):
            # Process a single chunk of tasks using the noise_pool
            for result in noise_pool.map(
                lambda actor, args: actor.process_noise.remote(*args),
                chunk
            ):
                noise_name, noise_config, size = chunk.pop(0)
                if result:
                    stl_queue.put((noise_name, result, size, noise_config))
                else:
                    logger.warning(f"Failed noise processing for {noise_name} at size {size}.")
                noise_pbar.update(1)

        # Process all chunks concurrently
        await asyncio.gather(*(process_chunk(chunk) for chunk in task_chunks))
        noise_pbar.close()

        # Stage 2: STL Conversion
        stl_tasks = []
        while not stl_queue.empty():
            stl_tasks.append(stl_queue.get_nowait())
        total_stl_tasks = len(stl_tasks)
        stl_pbar = tqdm(total=total_stl_tasks, desc="STL Conversion")

        for stl_result in stl_pool.map(
            lambda actor, args: actor.process_stl.remote(*args),
            stl_tasks
        ):
            noise_name, displacement_object, size, noise_config = stl_tasks.pop(0)
            if stl_result:
                tet_queue.put((noise_name, stl_result, size, noise_config))
            else:
                logger.warning(f"Failed STL processing for {noise_name} at size {size}.")
            stl_pbar.update(1)
        stl_pbar.close()

        # Stage 3: Tetrahedral Mesh Creation
        tet_tasks = []
        while not tet_queue.empty():
            tet_tasks.append(tet_queue.get_nowait())
        total_tet_tasks = len(tet_tasks)
        tet_pbar = tqdm(total=total_tet_tasks, desc="Tetrahedral Mesh")

        for _ in tet_pool.map(
            lambda actor, args: actor.process_tetrahedral.remote(
                noise_name=args[0],
                stl_object=args[1],
                noise_params=args[3],
                stl_params=config.stl_params,
                program_stl="hmm",
                program_tet="ftetwild",
                tet_params=config.tet_params,
                map_size=args[2]
            ),
            tet_tasks
        ):
            tet_pbar.update(1)
        tet_pbar.close()

    asyncio.run(main())
    shutdown_ray_cluster()

if __name__ == "__main__":
    app()

