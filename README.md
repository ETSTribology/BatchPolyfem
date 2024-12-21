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
