# DATASET.md

# ClusterMind Dataset Specification

## Dataset Strategy

Production GPU-cluster telemetry containing hardware failures, workload degradation events, root-cause labels, and recovery actions is rarely available publicly.

ClusterMind therefore uses a fully synthetic but reproducible telemetry dataset.

The dataset simulates a realistic enterprise GPU cluster operating training and inference workloads while experiencing normal operations, degradation events, and failure scenarios.

All telemetry, incidents, workload metadata, recovery actions, and operational records are generated programmatically using documented GPU monitoring concepts and scenario-based fault injection.

The dataset generator is considered a core project component and provides complete reproducibility.

---

# Simulated Environment

The dataset represents:

```text
10 GPU Compute Nodes

4 GPUs Per Node

40 Total GPUs

Training Workloads

Fine-Tuning Workloads

Inference Workloads

30 Days Of Operation
```

---

# Telemetry Sampling

```text
Sampling Interval:
1 Minute

Duration:
30 Days

Expected Records:
Approximately 1.7 Million GPU-Level Records

Normal Operations:
85%

Degraded Operations:
10%

Failure Events:
5%
```

Development Configuration:

```text
Duration:
7 Days

Sampling Interval:
5 Minutes

Expected Records:
Approximately 80,000 Records
```

---

# Dataset Components

The dataset generator produces five independent datasets.

```text
1. GPU Telemetry Dataset

2. Node Telemetry Dataset

3. Workload Dataset

4. Incident Dataset

5. Agent Evaluation Dataset
```

---

# Dataset 1: GPU Telemetry

## Purpose

Represents raw GPU-level monitoring data.

## Fields

```text
timestamp

node_id

gpu_id

gpu_utilization_percent

memory_utilization_percent

gpu_temperature_c

memory_temperature_c

power_draw_watts

sm_clock_mhz

memory_clock_mhz

ecc_correctable_errors

ecc_uncorrectable_errors

xid_error_code

nvlink_tx_mbps

nvlink_rx_mbps

anomaly_label
```

## Example Record

```json
{
  "timestamp": "2026-01-01T10:00:00",
  "node_id": "gpu-node-07",
  "gpu_id": "gpu-02",
  "gpu_utilization_percent": 94.3,
  "memory_utilization_percent": 91.2,
  "gpu_temperature_c": 88,
  "memory_temperature_c": 92,
  "power_draw_watts": 312,
  "sm_clock_mhz": 1450,
  "memory_clock_mhz": 1590,
  "ecc_correctable_errors": 0,
  "ecc_uncorrectable_errors": 0,
  "xid_error_code": 0,
  "nvlink_tx_mbps": 24300,
  "nvlink_rx_mbps": 24100,
  "anomaly_label": "thermal_throttling"
}
```

---

# Dataset 2: Node Telemetry

## Purpose

Represents compute-node health and operating conditions.

## Fields

```text
timestamp

node_id

cpu_utilization_percent

host_memory_percent

disk_io_percent

network_tx_mbps

network_rx_mbps

fan_speed_percent

node_health_status

active_workloads

healthy_gpu_count
```

## Example Record

```json
{
  "timestamp": "2026-01-01T10:00:00",
  "node_id": "gpu-node-07",
  "cpu_utilization_percent": 78,
  "host_memory_percent": 71,
  "disk_io_percent": 58,
  "network_tx_mbps": 8400,
  "network_rx_mbps": 8100,
  "fan_speed_percent": 96,
  "node_health_status": "degraded",
  "active_workloads": 3,
  "healthy_gpu_count": 3
}
```

---

# Dataset 3: Workload Dataset

## Purpose

Represents AI training and inference workloads.

## Fields

```text
job_id

job_type

priority

assigned_node

allocated_gpus

workload_status

checkpoint_available

training_throughput

tokens_per_second

step_duration_ms

training_loss

inference_latency_ms

error_rate_percent

restart_count
```

## Example Record

```json
{
  "job_id": "train-118",
  "job_type": "llm_finetuning",
  "priority": "high",
  "assigned_node": "gpu-node-07",
  "allocated_gpus": 4,
  "workload_status": "running",
  "checkpoint_available": true,
  "training_throughput": 1240,
  "tokens_per_second": 3100,
  "step_duration_ms": 420,
  "training_loss": 1.87,
  "inference_latency_ms": null,
  "error_rate_percent": 0,
  "restart_count": 0
}
```

---

# Dataset 4: Incident Dataset

## Purpose

Provides ground-truth incident records generated during fault windows.

## Fields

```text
incident_id

node_id

gpu_id

incident_type

severity

start_timestamp

end_timestamp

root_cause

affected_workload

gpu_hours_lost

recovery_action

incident_status
```

## Example Record

```json
{
  "incident_id": "INC-2026-0042",
  "node_id": "gpu-node-07",
  "gpu_id": "gpu-02",
  "incident_type": "thermal_throttling",
  "severity": "high",
  "start_timestamp": "2026-01-01T09:40:00",
  "end_timestamp": "2026-01-01T12:10:00",
  "root_cause": "cooling_degradation",
  "affected_workload": "train-118",
  "gpu_hours_lost": 6.8,
  "recovery_action": "resume_from_checkpoint",
  "incident_status": "resolved"
}
```

---

# Dataset 5: Agent Evaluation Dataset

## Purpose

Used to evaluate anomaly detection, diagnosis, and recovery-planning performance.

## Fields

```text
incident_id

expected_root_cause

expected_severity

expected_recovery_action

expected_workload_impact

detection_timestamp

ground_truth_label
```

---

# Failure Scenarios

The dataset generator supports scenario-based fault injection.

---

## Scenario 1: Thermal Throttling

### Pattern

```text
GPU Temperature Increases

Memory Temperature Increases

Clock Speed Decreases

Training Throughput Decreases

Power Remains High
```

### Expected Diagnosis

```text
Thermal Throttling
```

### Expected Recovery

```text
Move Workload

Quarantine Node

Resume From Checkpoint
```

---

## Scenario 2: ECC Memory Degradation

### Pattern

```text
ECC Errors Increase

Memory XID Events Appear

Training Failures Increase

Job Restarts Increase
```

### Expected Diagnosis

```text
GPU Memory Degradation
```

### Expected Recovery

```text
Quarantine GPU

Migrate Workload
```

---

## Scenario 3: Memory Saturation

### Pattern

```text
Memory Utilization Near 100%

OOM Events Occur

Step Duration Increases
```

### Expected Diagnosis

```text
Memory Saturation
```

### Expected Recovery

```text
Reduce Batch Size

Restart Job
```

---

## Scenario 4: Power Capping

### Pattern

```text
Power Draw Reaches Limit

Clock Frequency Oscillates

Training Throughput Declines
```

### Expected Diagnosis

```text
Power Limitation
```

### Expected Recovery

```text
Performance Review

Continue Monitoring
```

---

## Scenario 5: NVLink Degradation

### Pattern

```text
NVLink Throughput Drops

Distributed Training Slows

GPU Utilization Becomes Uneven
```

### Expected Diagnosis

```text
Interconnect Degradation
```

### Expected Recovery

```text
Workload Migration
```

---

## Scenario 6: Idle GPU Waste

### Pattern

```text
GPU Allocated

Utilization Near Zero

Memory Reserved

No Useful Throughput
```

### Expected Diagnosis

```text
Resource Waste
```

### Expected Recovery

```text
Release Allocation

Reschedule Job
```

---

## Scenario 7: Healthy High Utilization

### Purpose

Prevents false positives.

### Pattern

```text
GPU Utilization Increases

Power Increases

Temperature Increases

Training Throughput Improves

No Errors Exist
```

### Label

```text
Healthy
```

---

# Dataset Generation Requirements

The generator supports:

```text
Deterministic Random Seeds

Configurable Cluster Size

Configurable Number Of GPUs

Noise Injection

Failure Injection

Ground Truth Labels

Incident Windows

Scenario Severity Levels

Normal Operating Ranges

Recovery Outcomes
```

---

# Final Dataset Statement

ClusterMind uses a fully synthetic, reproducible GPU-cluster telemetry dataset representing 10 GPU nodes, 40 GPUs, AI training and inference workloads, infrastructure incidents, workload impact, and recovery actions.

The dataset contains realistic telemetry, labeled fault scenarios, ground-truth root causes, incident timelines, severity levels, and recovery outcomes, enabling anomaly detection, agent orchestration, root-cause analysis, and recovery-planning evaluation without requiring access to confidential production infrastructure data.
