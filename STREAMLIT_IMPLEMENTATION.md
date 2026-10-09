# ClusterMind Streamlit console

Entry point: `app/ui/main.py`

Local command:

`streamlit run app/ui/main.py --server.address 0.0.0.0 --server.port 18501`

Environment variables:

- `CLUSTERMIND_API_URL=http://localhost:18000`
- `CLUSTERMIND_API_PUBLIC_URL=http://localhost:18000`
- `CLUSTERMIND_GRAFANA_URL=http://localhost:13000`
- `CLUSTERMIND_JAEGER_URL=http://localhost:16686`
- `CLUSTERMIND_PROMETHEUS_URL=http://localhost:19090`
- `CLUSTERMIND_NTFY_URL=http://localhost:18080/clustermind-alerts`
