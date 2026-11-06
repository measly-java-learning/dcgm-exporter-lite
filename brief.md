# Project Brief

The Nvidia [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) project is a Golang service that reads GPU metrics via DCGM and exports
those metrics via Prometheus.  It is packaged as a container that requires root privileges and additional CDI hooks.

This project should provide a lightweight alternative to DCGM Exporter.  It will use `nvidia-ml-py` to retrieve available GPU metrics and publish them with `prometheus-client`.
The published metrics should match those from DCGM Exporter (https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html) as closely as possible given the different GPU metrics data sources.  The DCGM Exporter project also includes a sample Grafana dashboard here: https://github.com/NVIDIA/dcgm-exporter/tree/main/grafana.  This project should create a similar dashboard with the metrics that are readily available via `nvidia-ml-py`.
