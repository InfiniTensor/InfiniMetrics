# InfiniMetrics Dashboard User Guide
## 1. Dashboard Overview

InfiniMetrics Dashboard provides a unified interface to visualize benchmark and evaluation results of AI accelerators across the following scenarios:

- Communication (NCCL / Collective Communication)

- Training (Training / Distributed Training)

- Inference (Direct / Service Inference)

- Operator (Core Operator Performance)

The benchmark framework produces two types of outputs:

```
JSON  -> configuration / environment / scalar metrics
CSV   -> curves / time-series data
```
The Dashboard automatically loads test results and provides unified analysis capabilities, including:

- un ID fuzzy search: locate specific test runs using partial Run IDs

- General filters: filter results by framework, model, device count, etc.

- Multi-run comparison: select multiple runs to compare performance

- Performance visualization: display curves such as latency / throughput / loss

- Statistics and configuration view: inspect throughput statistics, runtime configuration, and environment details

For example, you can enter:
```
allreduce
service
```
to perform fuzzy matching on Run IDs

Example screenshot:
![Run_ID research](https://private-user-images.githubusercontent.com/94775646/561443423-ac18e86d-b6e3-4ae5-a386-15d9cf34bf69.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0MjMtYWMxOGU4NmQtYjZlMy00YWU1LWEzODYtMTVkOWNmMzRiZjY5LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI3NDFjM2QzZThjMGY3MjM3ZTJkY2IxMDY4ZDc2ZWYxZmYzOGQzZDAyYjQyZDE4NGZmZmEwZDRjOTQzN2IzOGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.pOFgY_W8dwv-cMMSBP4yee_2l0we_iCL4jptlMLIsnk)
## 2. Running the Dashboard
### 2.1 Environment Requirements

Before using the Dashboard, install the following dependencies:
```
streamlit
plotly
pandas
```
### 2.2 Start the Dashboard

Run the following command in the project root directory:
```
python -m streamlit run dashboard/app.py
```
Access URL after startup:
```
Local URL:    http://localhost:8501
Network URL:  http://<server-ip>:8501
```
Explanation:

Local URL: accessible only on the local machine

Network URL: accessible from other machines within the same network

## 3. Communication Test Analysis

Path:
```
Dashboard → Communication Performance Test
```
Supported features:
```
Bandwidth analysis curve - peak bandwidth

Latency analysis curve - average latency

Test duration

GPU memory usage

Communication configuration analysis
```
Example screenshot:
![Communication Test](https://private-user-images.githubusercontent.com/94775646/561443439-b3e3f409-6bb9-4c32-bee8-9928781fcd39.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0MzktYjNlM2Y0MDktNmJiOS00YzMyLWJlZTgtOTkyODc4MWZjZDM5LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM1MGU2YTcxNjcxNWM4ZWIwYWJiZTVkNmU2NTIxNWNkZDI3NGRiNTNmNWI0YWY4NWUxZmI0MzBjZjc4OGVkZjgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.AaFHdRt4SGyldFCZR1hNBhySRoR2cGDHQ0aAVNG4dq8)
## 4. Inference Test Analysis

Path:
```
Dashboard → Inference Performance Test
```
Modes:
```
Direct Inference
Service Inference
```
Displayed metrics:
```
TTFT

Latency

Throughput

GPU memory usage

Inference configuration analysis
```
Example screenshot:
![Inference Test](https://private-user-images.githubusercontent.com/94775646/561443450-9b57045c-d320-4da6-9b23-6e0f9a6c96cb.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0NTAtOWI1NzA0NWMtZDMyMC00ZGE2LTliMjMtNmUwZjlhNmM5NmNiLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcxYTVjOWY2MzE4NjNjYjJlNGI3ZjkzMDk4MzJmZWNkY2ZiZWI2ZjRiZjcyZjVkZTAzNGE0YzljYjVhZmJhZDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0._LAZMfKkStSoWgw-1t22fOdlbkng-xZnV9rHIpd4v-M)
## 5. Training Test Analysis

Path:
```
Dashboard → Training Performance Test
```
Supported features:
```
Loss curve

Perplexity curve

Throughput curve

GPU memory usage

Training configuration analysis
```
Example screenshot:
![Training Test](https://private-user-images.githubusercontent.com/94775646/561443390-5a789775-9fd4-4146-932d-df2436c20977.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDMzOTAtNWE3ODk3NzUtOWZkNC00MTQ2LTkzMmQtZGYyNDM2YzIwOTc3LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc4ZjFkZGFjYWE1ZWM0ZDRjOWQ3M2QxYzM4YzRjNzUyZmMzZGIwMTQwNjBjMDU4N2QyMmUyOTRlYzU3MWRkN2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.3YfafricSVbHzguSeDhhKiQ0S68AQDY48a1HRpMHTj4)
## 6. Operator Test Analysis

Path:
```
Dashboard → Operator Performance Test
```
Supported metrics:
```
latency

flops

bandwidth
```
Example screenshot:
![Operator Test](https://private-user-images.githubusercontent.com/94775646/561443431-58d7f944-8e6b-4f2a-8fdc-671d09653da8.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMTc3MTAsIm5iZiI6MTc3MzIxNzQxMCwicGF0aCI6Ii85NDc3NTY0Ni81NjE0NDM0MzEtNThkN2Y5NDQtOGU2Yi00ZjJhLThmZGMtNjcxZDA5NjUzZGE4LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzExVDA4MjMzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBjNzNhNGI5YjhlMDc3MjNiMzU3ODQ5YTNjMDBjYTcyN2E1NjIxNjAyNGU3ZDM5ODMwMGVlZTMyMWY0OGE3ZTcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OpDxdAaEsdHYX_jOWNmmdPB-jKp92gbLh1VFs0H2j6c)