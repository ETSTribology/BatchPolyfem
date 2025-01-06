"""
hmm.py

A Python script to call the 'hmm' heightmap meshing utility with
default arguments -z 3 -b 3, plus any overrides.
"""

import argparse
import subprocess
import sys
import logging
from rich.console import Console
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
console = Console()

def run_hmm(
    input_file: str,
    output_file: str,
    error: str = "0.001",
    triangles: str = None,
    quiet: bool = False,
    extra_args=None
) -> None:
    """
    Run the 'hmm' command-line tool with default -z 3 -b 3 and optional overrides.
    ...
    """
    if extra_args is None:
        extra_args = []

    cmd = [
        "hmm",
        input_file,
        output_file,
        "-z", "3",
        "-b", "3",
        "-e", error
    ]
    if triangles:
        cmd += ["-t", triangles]
    if quiet:
        cmd += ["-q"]
    cmd += extra_args

    logger.info("Running hmm command:")
    logger.info(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("Error: The 'hmm' executable was not found in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"hmm returned non-zero exit status {e.returncode}.")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run 'hmm' with default '-z 3 -b 3' for heightmap meshing."
    )
    parser.add_argument("input_file", help="Path to input heightmap.")
    parser.add_argument("output_file", help="Path to output STL file")
    parser.add_argument("-e", "--error", default="0.001", help="Max triangulation error.")
    parser.add_argument("-t", "--triangles", default=None, help="Max number of triangles.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output.")

    args, extra = parser.parse_known_args()

    run_hmm(
        input_file=args.input_file,
        output_file=args.output_file,
        error=args.error,
        triangles=args.triangles,
        quiet=args.quiet,
        extra_args=extra
    )

if __name__ == "__main__":
    main()
