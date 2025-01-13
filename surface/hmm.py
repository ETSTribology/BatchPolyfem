#!/usr/bin/env python3
import argparse
import subprocess
import sys
import logging
import asyncio
from rich.console import Console
from rich.logging import RichHandler
import ray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
console = Console()

def run_hmm_sync(
    input_file: str,
    output_file: str,
    z_value: int = 3,
    base: int = 1,
    error: str = "0.001",
    triangles: str = None,
    quiet: bool = False,
    invert: bool = False,
    normal_map: str = None,
    z_exagg: float = 1.0,
    extra_args=None
) -> None:
    """
    Run the 'hmm' command-line tool synchronously with options for invert, normal map, and z exaggeration.
    """
    if extra_args is None:
        extra_args = []

    cmd = [
        "hmm",
        str(input_file),
        str(output_file),
        "-z", str(z_value),
        "-b", str(base),
        "-e", str(error)
    ]

    if triangles:
        cmd += ["-t", str(triangles)]
    if quiet:
        cmd += ["-q"]
    if invert:
        cmd += ["--invert"]
    if normal_map:
        cmd += ["--normal-map", str(normal_map)]
    if z_exagg != 1.0:
        cmd += ["-x", str(z_exagg)]

    cmd += [str(arg) for arg in extra_args]

    logger.debug(f"Constructed HMM command: {' '.join(cmd)}")

    try:
        logger.info(f"Starting HMM execution for input: {input_file}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"HMM execution finished for input: {input_file}")

        if result.returncode != 0:
            logger.error(f"'hmm' failed with exit code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

        logger.info(f"HMM output: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("Error: The 'hmm' executable was not found in PATH.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"'hmm' returned non-zero exit status {e.returncode}.")
        raise

async def run_hmm_async(
    input_file: str,
    output_file: str,
    z_value: int = 3,
    base: int = 1,
    error: str = "0.001",
    triangles: str = None,
    quiet: bool = False,
    invert: bool = False,
    normal_map: str = None,
    z_exagg: float = 1.0,
    extra_args=None
) -> None:
    """
    Asynchronous version of the 'hmm' command-line tool execution.
    """
    if extra_args is None:
        extra_args = []

    cmd = [
        "hmm",
        str(input_file),
        str(output_file),
        "-z", str(z_value),
        "-b", str(base),
        "-e", str(error)
    ]

    if triangles:
        cmd += ["-t", str(triangles)]
    if quiet:
        cmd += ["-q"]
    if invert:
        cmd += ["--invert"]
    if normal_map:
        cmd += ["--normal-map", str(normal_map)]
    if z_exagg != 1.0:
        cmd += ["-x", str(z_exagg)]

    cmd += [str(arg) for arg in extra_args]

    logger.debug(f"Constructed command: {' '.join(cmd)}")

    try:
        logger.info("Executing command: " + " ".join(cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"'hmm' failed with exit code {process.returncode}")
            logger.error(stderr.decode().strip())
            raise subprocess.CalledProcessError(process.returncode, cmd)

        logger.info(stdout.decode().strip())

    except FileNotFoundError:
        logger.error("Error: The 'hmm' executable was not found in PATH.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"'hmm' returned non-zero exit status {e.returncode}.")
        raise

@ray.remote
def run_hmm_ray_task_sync(
    input_file: str,
    output_file: str,
    z_value: int = 3,
    base: int = 1,
    error: str = "0.001",
    triangles: str = None,
    quiet: bool = False,
    invert: bool = False,
    normal_map: str = None,
    z_exagg: float = 1.0,
    extra_args=None
) -> None:
    """
    Ray-compatible synchronous function for running the hmm command.
    """
    run_hmm_sync(
        input_file=input_file,
        output_file=output_file,
        z_value=z_value,
        base=base,
        error=error,
        triangles=triangles,
        quiet=quiet,
        invert=invert,
        normal_map=normal_map,
        z_exagg=z_exagg,
        extra_args=extra_args
    )

