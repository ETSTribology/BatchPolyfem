# BatchPolyfem

**Batch Polyfem Frictional Contact**  
A robust infrastructure for performing batch simulations of frictional contact using Polyfem and supporting services.

---

## Overview

**BatchPolyfem** is designed to streamline batch processing and frictional contact simulations. The project utilizes a containerized architecture to integrate Polyfem with various supporting services for efficient data processing, storage, and visualization.

---

## Infrastructure

The following table provides an overview of the infrastructure components, their Docker images, versions, and exposed ports:

| **Service**  | **Docker Image**                       | **Version** | **Port(s)**                                |
|--------------|----------------------------------------|-------------|--------------------------------------------|
| **MinIO**    | `minio/minio:latest`                   | Latest      | `9000:9000` (API) <br> `20091:9001` (Console) <br> `9001:9000` (Website) |
| **Zookeeper**| `confluentinc/cp-zookeeper:latest`     | Latest      | `2181:2181`                                 |
| **Kafka**    | `confluentinc/cp-kafka:latest`         | Latest      | `9092:9092`                                 |
| **MongoDB**  | `mongo:latest`                         | Latest      | `27017:27017`                               |
| **Qdrant**   | `qdrant/qdrant:latest`                 | Latest      | `6333:6333`                                 |


[![](https://mermaid.ink/img/pako:eNqNUbtuwzAM_BWBc_IDHjoEWoIiKIoUGVplYCzaFlJThiwFSJP8e2nLdsZ24vF4Oj50g9JbggLqgF2jPrRhw0r16ZSJLVcB-xhSGVOgoaTUzvH27cvAGA0cM_uK1RmFHePC7jzXXm8GdUZL5d0G5CiFDCae2OYJRnO1Xr_cDezxQmrjGMNVaYzifp-NR2mGszj6QOpApUT3Q3Z5MfXhadRJrp1s504p0tN63OsfuucIfyhza1hBS6FFZ-Xet-GZgdhQSwYKgRbD2YDhh-gwRb-_cgmFnJ5WEHyqGygq_O4lS53FSNqh_FG7sB3yp_dz_vgFly6djQ?type=png)](https://mermaid.live/edit#pako:eNqNUbtuwzAM_BWBc_IDHjoEWoIiKIoUGVplYCzaFlJThiwFSJP8e2nLdsZ24vF4Oj50g9JbggLqgF2jPrRhw0r16ZSJLVcB-xhSGVOgoaTUzvH27cvAGA0cM_uK1RmFHePC7jzXXm8GdUZL5d0G5CiFDCae2OYJRnO1Xr_cDezxQmrjGMNVaYzifp-NR2mGszj6QOpApUT3Q3Z5MfXhadRJrp1s504p0tN63OsfuucIfyhza1hBS6FFZ-Xet-GZgdhQSwYKgRbD2YDhh-gwRb-_cgmFnJ5WEHyqGygq_O4lS53FSNqh_FG7sB3yp_dz_vgFly6djQ)


## Event Flow

The following sequence demonstrates the interaction between services during the lifecycle of a batch processing event. It highlights how events flow from Kafka to PolyRunnerBatch, Polyfem instances, MinIO, and finally MongoDB.

### Process Flow Description

1. **Trigger Event**: An event is published to Kafka to initiate a PolyRunnerBatch.
2. **PolyRunnerBatch**: The PolyRunnerBatch service processes the event and spawns multiple Polyfem instances.
3. **Polyfem Instances**: Each Polyfem instance processes its assigned task. Upon successful completion, it uploads results to MinIO.
4. **MinIO Notification**: Once the upload to MinIO is complete, an event is triggered to process the uploaded data.
5. **Data Processing**: A service retrieves the data from MinIO, processes it, and stores the processed data in MongoDB.


### Explanation

- **Kafka**: Acts as the event broker, triggering batch simulations.
- **PolyRunnerBatch**: Manages the orchestration of simulation tasks by creating events for each Polyfem instance.
- **Polyfem Instances**: Perform the computational tasks for the simulations.
- **MinIO**: Serves as the storage backend for the simulation outputs.
- **MongoDB**: Stores the final processed data, making it available for querying and visualization.

[![](https://mermaid.ink/img/pako:eNqVkbFuwjAQhl_FugkkGMqYgYGmQ1VBEaFL5eWUXJyIxBdd7EoIePc6TlvUqqqKF_s_fZ8t350g54IgASPYVWqfaqvCesLygJNJ3KZTNZ8vz3upjSFRD29k3VltuTnuvLUkK3R5NWo_itG7F0JH_ej1o1hSe3ezsbjZ2Gh7dYY3I_zSNYyFynyeUx_gdW0fn79xi39ym7-4kYzh1_4JD3iKDj_uvBYinzkWUjvqfTN8ac3WcLqCGbQkLdZFmNlpMDW4ilrSkIRjgXLQoO0lcOgdZ0ebQ-LE0wyEvakgKbHpQ_JdEVqW1hgG335VO7SvzJ_58g4eibin?type=png)](https://mermaid.live/edit#pako:eNqVkbFuwjAQhl_FugkkGMqYgYGmQ1VBEaFL5eWUXJyIxBdd7EoIePc6TlvUqqqKF_s_fZ8t350g54IgASPYVWqfaqvCesLygJNJ3KZTNZ8vz3upjSFRD29k3VltuTnuvLUkK3R5NWo_itG7F0JH_ej1o1hSe3ezsbjZ2Gh7dYY3I_zSNYyFynyeUx_gdW0fn79xi39ym7-4kYzh1_4JD3iKDj_uvBYinzkWUjvqfTN8ac3WcLqCGbQkLdZFmNlpMDW4ilrSkIRjgXLQoO0lcOgdZ0ebQ-LE0wyEvakgKbHpQ_JdEVqW1hgG335VO7SvzJ_58g4eibin)

## Getting Started

### Prerequisites

1. **Docker**: Ensure Docker is installed and running on your system.
2. **Docker Compose**: Install Docker Compose to manage multi-container setups.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ETSTribology/batchpolyfem.git
   cd batchpolyfem
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Verify the services:
   ```bash
   docker-compose ps
   ```

### Access Services

- **MinIO API**: [http://localhost:9000](http://localhost:9000)
- **MinIO Console**: [http://localhost:20091](http://localhost:20091)
- **Kafka**: Port `9092`
- **MongoDB**: Port `27017`
- **Qdrant**: Port `6333`
