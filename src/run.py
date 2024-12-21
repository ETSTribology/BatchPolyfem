import json
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from tqdm import tqdm
import logging
from typing import Optional, Dict
from pymongo import MongoClient
from minio import Minio
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from mongo import MongoConnect
from minio import MinIOConnect
from kafka import KafkaConnect
from config import Config

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

    # Load configuration using Config class
    config = Config(file, verbose=verbose).settings

    # Initialize connections using respective classes
    mongo_client = MongoConnect(config["mongo"])
    minio_client = MinIOConnect(config["minio"])
    kafka_producer = KafkaConnect(config["kafka"])

    logger.debug("Connections to MongoDB, MinIO, and Kafka established successfully.")

    # Example task processing logic
    tasks = config.get("tasks", [])
    if not tasks:
        logger.error("No tasks found in the configuration.")
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
            if task.get("type") == "file_processing":
                process_file_and_dump_to_mongo(
                    minio_client,
                    mongo_client,
                    task.get("bucket_name"),
                    task.get("object_name")
                )
            import time
            time.sleep(0.5)
            progress.advance(task_progress)
            logger.debug(f"Processed task ID: {task.get('id', 'N/A')}")

    logger.info("Completed processing with Rich Progress.")

if __name__ == "__main__":
    app()
