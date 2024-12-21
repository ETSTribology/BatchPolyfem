import threading
import time
import jsonschema
from typing import List, Dict
from kafka import KafkaProducer
import json

class PolyfemBatchRunner:
    """
    A class to run multiple PolyfemRunner instances concurrently using threads.
    """
    def __init__(self, runners: List[PolyfemRunner], schema: Dict, kafka_producer: KafkaProducer):
        if not all(isinstance(runner, PolyfemRunner) for runner in runners):
            raise ValueError("All elements in 'runners' must be instances of PolyfemRunner.")
        self.runners = runners
        self.schema = schema
        self.kafka_producer = kafka_producer

    def run_all(self, config_file: str, updates_list: List[Dict]):
        """
        Runs multiple PolyfemRunner instances concurrently.

        :param config_file: Base configuration file.
        :param updates_list: List of parameter updates for each runner.
        """
        threads = []
        progress = {}

        def thread_task(runner, updates, thread_id):
            try:
                runner.logger.info(f"Thread {thread_id}: Starting simulation.")
                progress[thread_id] = "Running"
                runner.run_simulation(config_file, self.schema, updates)

                # Send event to Kafka with run details
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                event_data = {
                    "thread_id": thread_id,
                    "timestamp": timestamp,
                    "status": "Completed",
                    "updates": updates
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
                    "error": str(e)
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
        self.logger.info("All simulations completed. Final progress states:")
        for thread_id, state in progress.items():
            self.logger.info(f"Thread {thread_id}: {state}")
