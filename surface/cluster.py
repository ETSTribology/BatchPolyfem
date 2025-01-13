import ray
import logging
from rich.logging import RichHandler
from typing import Optional, List, Dict, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def init_ray_cluster(address: Optional[str] = "localhost", num_cpus: Optional[int] = 8) -> None:
    try:
        if address:
            ray.init(address=address, ignore_reinit_error=True, dashboard_host="0.0.0.0")
        else:
            ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
        resources = ray.cluster_resources()
        logger.info(f"Ray cluster initialized successfully with resources: {resources}")
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster at address {address}. Error: {e}")
        raise

def shutdown_ray_cluster() -> None:
    """
    Shutdown the Ray cluster.
    """
    try:
        ray.shutdown()
        logger.info("Ray cluster has been shut down.")
    except Exception as e:
        logger.error(f"Error shutting down Ray cluster: {e}")

def monitor_cluster(duration: float = 60.0, interval: float = 10.0) -> None:
    """
    Monitor the Ray cluster for a specified duration, printing cluster resources at each interval.
    
    Args:
        duration (float): Total time to monitor the cluster (in seconds).
        interval (float): Time between checks (in seconds).
    """
    import time
    start_time = time.time()
    logger.info("Starting cluster monitoring...")
    try:
        while time.time() - start_time < duration:
            resources = ray.cluster_resources()
            nodes = ray.nodes()
            # Safely access 'Alive' and 'State'
            active_nodes = [
                node for node in nodes 
                if node.get("Alive", False) and node.get("State") == "ALIVE"
            ]
            logger.info(f"Cluster resources: {resources}")
            logger.info(f"Active nodes count: {len(active_nodes)}")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Cluster monitoring interrupted.")
    finally:
        logger.info("Cluster monitoring completed.")

def get_node_resources() -> List[dict]:
    try:
        nodes = ray.nodes()
        active_nodes = [
            {
                "Node Manager Address": node.get("NodeManagerAddress", ""),
                "CPUs": node.get("Resources", {}).get("CPU", 0),
                "GPUs": node.get("Resources", {}).get("GPU", 0),
                "Node ID": node.get("NodeID", ""),
                "State": node.get("State", ""),
                "Alive": node.get("Alive", False),
            }
            for node in nodes
        ]
        return active_nodes
    except Exception as e:
        logger.error(f"Failed to get node resources: {e}")
        return []

def enforce_task_resource_allocation(task_resources: Dict[str, int]) -> ray.util.placement_group:
    """
    Create a placement group to ensure tasks use specific resources.

    Args:
        task_resources (Dict[str, int]): Required resources (e.g., {"CPU": 2, "GPU": 1}).

    Returns:
        ray.util.placement_group: The created placement group.
    """
    try:
        bundles = [task_resources]
        placement_group = ray.util.placement_group(bundles, strategy="STRICT_PACK")
        ray.get(placement_group.ready())
        logger.info(f"Placement group created: {placement_group.id} with resources: {task_resources}")
        return placement_group
    except Exception as e:
        logger.error(f"Failed to create placement group with resources {task_resources}. Error: {e}")
        raise

def enforce_task_resource_allocation(task_resources: dict):
    try:
        bundles = [task_resources]
        placement_group = ray.util.placement_group(bundles, strategy="STRICT_PACK")
        ray.get(placement_group.ready())
        logger.info(f"Placement group created: {placement_group.id} with resources: {task_resources}")
        return placement_group
    except Exception as e:
        logger.error(f"Failed to create placement group with resources {task_resources}. Error: {e}")
        raise

@ray.remote
def task_with_specific_resources(task_id: int) -> str:
    """
    Example task that runs with specific resources.

    Args:
        task_id (int): The ID of the task.

    Returns:
        str: Task completion message.
    """
    logger.info(f"Running task {task_id} on assigned resources.")
    return f"Task {task_id} completed."


def main():
    # Initialize Ray cluster
    init_ray_cluster(address="auto")

    # Fetch and log node details
    logger.info("Fetching node details...")
    nodes = get_node_resources()
    logger.info(f"Number of nodes in the cluster: {len(nodes)}")
    for node in nodes:
        logger.info(f"Node Details: {node}")

    # Define required resources and create a placement group
    required_resources = {"CPU": 2, "GPU": 1}
    placement_group = enforce_task_resource_allocation(required_resources)

    # Assign tasks to the placement group
    tasks = [
        task_with_specific_resources.options(
            placement_group=placement_group, placement_group_bundle_index=0
        ).remote(task_id=i)
        for i in range(5)
    ]

    # Wait for tasks to complete and log results
    results = ray.get(tasks)
    for result in results:
        logger.info(result)

    # Shutdown Ray cluster
    ray.shutdown()


if __name__ == "__main__":
    main()

