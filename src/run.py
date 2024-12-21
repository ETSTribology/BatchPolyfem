import json
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from tqdm import tqdm
import argparse
import logging
from typing import Optional

app = typer.Typer()
console = Console()

def setup_logging(verbose: bool):
    """
    Sets up logging configuration with RichHandler.
    If verbose is True, set log level to DEBUG, else INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger("rich")
    return logger

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    This is for demonstration; Typer can handle arguments itself.
    """
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="config.json",
        help="Path to the JSON file to process.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    args = parser.parse_args()
    return args

@app.command()
def run(
    file: str = typer.Option(
        "config.json",
        "--file",
        "-f",
        help="Path to the JSON file to process.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output.",
    ),
):
    """
    Load and process a JSON file with progress bars and rich output.
    """
    # Set up logging
    logger = setup_logging(verbose)

    logger.debug(f"Starting the run command with file: {file} and verbose={verbose}")

    # Load JSON data
    try:
        with open(file, "r") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON file: {file}")
    except FileNotFoundError:
        logger.error(f"File not found: {file}")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file}: {e}")
        raise typer.Exit(code=1)

    # Check if 'tasks' key exists and is a list
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        logger.error("JSON does not contain a 'tasks' list.")
        raise typer.Exit(code=1)

    logger.debug(f"Number of tasks to process: {len(tasks)}")

    # Display tasks using Rich Table
    table = Table(title="Tasks Overview")

    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="magenta")
    table.add_column("Description", style="green")

    for task in tasks:
        table.add_row(
            str(task.get("id", "N/A")),
            task.get("name", "N/A"),
            task.get("description", "N/A"),
        )

    console.print(table)

    # Process tasks with Rich Progress
    logger.debug("Starting processing with Rich Progress.")
    if verbose:
        console.print("[bold yellow]Processing tasks with Rich Progress...[/bold yellow]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_progress = progress.add_task("[green]Processing...", total=len(tasks))
        for task in tasks:
            # Simulate some processing
            import time
            time.sleep(0.5)
            progress.advance(task_progress)
            logger.debug(f"Processed task ID: {task.get('id', 'N/A')}")

    logger.info("Completed processing with Rich Progress.")

    # Alternatively, using tqdm for a simple progress bar
    logger.debug("Starting processing with tqdm.")
    if verbose:
        console.print("[bold yellow]Processing tasks with tqdm...[/bold yellow]")

    for task in tqdm(tasks, desc="Processing tasks", unit="task"):
        # Simulate some processing
        import time
        time.sleep(0.5)
        logger.debug(f"Processed task ID with tqdm: {task.get('id', 'N/A')}")

    logger.info("Completed processing with tqdm.")

    console.print("[bold green]All tasks have been processed successfully![/bold green]")
    logger.debug("Run command completed successfully.")

if __name__ == "__main__":
    args = parse_arguments()
    app()
