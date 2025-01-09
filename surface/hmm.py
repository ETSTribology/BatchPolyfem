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
    Run the 'hmm' command-line tool with options for invert, normal map, and z exaggeration.
    """
    if extra_args is None:
        extra_args = []

    cmd = [
        "hmm",
        input_file,
        output_file,
        "-z", str(z_value),
        "-b", str(base),
        "-e", error
    ]

    if triangles:
        cmd += ["-t", triangles]
    if quiet:
        cmd += ["-q"]
    if invert:
        cmd += ["--invert"]
    if normal_map:
        cmd += ["--normal-map", normal_map]
    if z_exagg != 1.0:
        cmd += ["-x", str(z_exagg)]

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
        description="Run 'hmm' with options for invert, normal map, and z exaggeration."
    )
    parser.add_argument("input_file", help="Path to input heightmap.")
    parser.add_argument("output_file", help="Path to output STL file.")
    parser.add_argument("-z", "--z_value", default=30, type=int, help="Z value.")
    parser.add_argument("-b", "--base", default=3, type=int, help="Base value.")
    parser.add_argument("-e", "--error", default="0.001", help="Max triangulation error.")
    parser.add_argument("-t", "--triangles", default=None, help="Max number of triangles.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output.")
    parser.add_argument("--invert", action="store_true", help="Invert the heightmap.")
    parser.add_argument("--normal-map", default=None, help="Path to save the normal map as PNG.")
    parser.add_argument("-x", "--z_exagg", default=1.0, type=float, help="Z exaggeration multiplier.")

    args, extra = parser.parse_known_args()

    run_hmm(
        input_file=args.input_file,
        output_file=args.output_file,
        z_value=args.z_value,
        base=args.base,
        error=args.error,
        triangles=args.triangles,
        quiet=args.quiet,
        invert=args.invert,
        normal_map=args.normal_map,
        z_exagg=args.z_exagg,
        extra_args=extra
    )

if __name__ == "__main__":
    main()
