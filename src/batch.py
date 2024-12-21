import ray
import threading
import json
import time
from typing import List, Dict, Union
from kafka import KafkaProducer


class PolyfemBatchRunner:
    """
    A class to run multiple PolyfemRunner instances concurrently using threads or Ray.
    """
    def __init__(self, runners: List["PolyfemRunner"], schema: Dict, kafka_config: Dict, mode: str = "thread"):
        """
        Initialize the batch runner.

        :param runners: List of PolyfemRunner instances.
        :param schema: JSON schema for validation.
        :param kafka_config: Kafka configuration dictionary.
        :param mode: Execution mode - 'thread' for local threading, 'ray' for distributed Ray.
        """
        if not all(isinstance(runner, PolyfemRunner) for runner in runners):
            raise ValueError("All elements in 'runners' must be instances of PolyfemRunner.")
        if mode not in ["thread", "ray"]:
            raise ValueError("Mode must be 'thread' or 'ray'.")

        self.runners = runners
        self.schema = schema
        self.mode = mode.lower()
        self.kafka_config = kafka_config
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_config["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        # Initialize Ray workers if using Ray
        if self.mode == "ray":
            ray.init(ignore_reinit_error=True)
            self.workers = [
                PolyfemWorker.remote(runner, schema, kafka_config) for runner in self.runners
            ]

    def run_all(self, config_file: str, updates_list: List[Dict]):
        """
        Runs multiple PolyfemRunner instances based on the selected mode.

        :param config_file: Base configuration file.
        :param updates_list: List of parameter updates for each runner.
        """
        if self.mode == "thread":
            self._run_with_threads(config_file, updates_list)
        elif self.mode == "ray":
            self._run_with_ray(config_file, updates_list)

    def _run_with_threads(self, config_file: str, updates_list: List[Dict]):
        """
        Runs simulations concurrently using threads.
        """
        threads = []
        progress = {}

        def thread_task(runner, updates, thread_id):
            try:
                runner.logger.info(f"Thread {thread_id}: Starting simulation.")
                progress[thread_id] = "Running"
                runner.run_simulation(config_file, self.schema, updates)

                # Send success event to Kafka
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                event_data = {
                    "thread_id": thread_id,
                    "timestamp": timestamp,
                    "status": "Completed",
                    "updates": updates,
                }
                self.kafka_producer.send("polyfem_events", value=event_data)
                progress[thread_id] = "Completed"
                runner.logger.info(f"Thread {thread_id}: Simulation completed and event sent to Kafka.")
            except Exception as e:
                progress[thread_id] = "Failed"
                runner.logger.error(f"Thread {thread_id}: Error - {e}")
                event_data = {
                    "thread_id": thread_id,
                    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                    "status": "Failed",
                    "error": str(e),
                }
                self.kafka_producer.send("polyfem_events", value=event_data)

        for i, (runner, updates) in enumerate(zip(self.runners, updates_list)):
            thread = threading.Thread(target=thread_task, args=(runner, updates, i))
            threads.append(thread)
            progress[i] = "Not Started"
            thread.start()

        for thread in threads:
            thread.join()

        self.kafka_producer.flush()
        self.runners[0].logger.info("All simulations completed (threads). Final progress states:")
        for thread_id, state in progress.items():
            self.runners[0].logger.info(f"Thread {thread_id}: {state}")

    def _run_with_ray(self, config_file: str, updates_list: List[Dict]):
        """
        Runs simulations concurrently using Ray.
        """
        futures = [
            worker.run_simulation.remote(config_file, updates, thread_id)
            for thread_id, (worker, updates) in enumerate(zip(self.workers, updates_list))
        ]

        results = ray.get(futures)

        # Print results and ensure Kafka messages are flushed
        for result in results:
            print(result)

        for worker in self.workers:
            ray.get(worker.kafka_producer.flush.remote())

        self.runners[0].logger.info("All simulations completed (Ray).")


@ray.remote
class PolyfemWorker:
    """
    A Ray worker that runs a PolyfemRunner simulation.
    """
    def __init__(self, runner, schema, kafka_config):
        self.runner = runner
        self.schema = schema
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_config["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def run_simulation(self, config_file: str, updates: Dict, thread_id: int):
        """
        Executes the simulation for a given runner and sends events to Kafka.
        """
        try:
            self.runner.logger.info(f"Worker {thread_id}: Starting simulation.")
            self.runner.run_simulation(config_file, self.schema, updates)

            # Send success event to Kafka
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            event_data = {
                "thread_id": thread_id,
                "timestamp": timestamp,
                "status": "Completed",
                "updates": updates,
            }
            self.kafka_producer.send("polyfem_events", value=event_data)
            self.runner.logger.info(f"Worker {thread_id}: Simulation completed successfully.")
            return {"thread_id": thread_id, "status": "Completed"}
        except Exception as e:
            self.runner.logger.error(f"Worker {thread_id}: Error - {e}")
            event_data = {
                "thread_id": thread_id,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "status": "Failed",
                "error": str(e),
            }
            self.kafka_producer.send("polyfem_events", value=event_data)
            return {"thread_id": thread_id, "status": "Failed", "error": str(e)}
