#!/usr/bin/env python3
"""
Service Inference Runner Implementation
Start service and run trace testing
"""
import asyncio
import logging
import time
import json
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional

from utils.accelerator_monitor import create_accelerator_monitor
from infer_runner_base import InferRunnerBase, TimeseriesMetric, ScalarMetric
from infer_config import InferConfig, ServiceInferArgs
from utils.trace_client import TraceClient, TraceClientConfig, RequestTrace
from utils.prompt_generator import create_prompt_generator

logger = logging.getLogger(__name__)

# default value
DEFAULT_SERVICE_PORT = 8000
DEFAULT_MAX_WAIT_TIME = 120
DEFAULT_WAIT_INTERVAL = 3

class ServiceInferRunner(InferRunnerBase):
    """Service Inference Runner"""

    def __init__(self, config: InferConfig, adapter):
        super().__init__(config, adapter)
        self.infer_args: ServiceInferArgs = config.infer_args

        # Trace related
        self.traces: List[RequestTrace] = []
        self.trace_stats: Dict[str, Any] = {}

        # Add accelerator monitor
        self.accelerator_monitor = None

        # Service manager
        self.service_manager = None

        # select ServiceManager from the framework
        self.service_manager_class = self._get_service_manager_class()

        logger.info(f"ServiceInferRunner created for trace: {self.infer_args.request_trace}")
        logger.info(f"Concurrency: {self.infer_args.concurrency}")
        logger.info(f"Max sequence length: {self.infer_args.max_seq_len}")
    
    def _get_service_manager_class(self):
        """Get the corresponding ServiceManager class according to framework"""
        framework = self.config.framework.value
        
        try:
            if framework == "infinilm":
                from service_manager import InfiniLMServiceManager
                return InfiniLMServiceManager
            elif framework == "vllm":
                from service_manager import VLLMServiceManager
                return VLLMServiceManager
            else:
                 raise ValueError(f"Unsupported framework for service mode: {framework}")

        except (ImportError, ValueError) as e:
            logger.warning(f"Cannot get ServiceManager for {framework}: {e}")
            return None

    def setup(self) -> None:
        """Set up service inference environment"""
        logger.info("Setting up service inference environment")

        # Create accelerator monitor
        device_ids = self.config.device.device_ids
        if self.config.device.cpu_only:
            logger.info("CPU-only mode, accelerator monitoring disabled")
            self.accelerator_monitor = None
        else:
            self.accelerator_monitor = create_accelerator_monitor(
                accelerator_type=self.config.device.accelerator,
                device_ids=device_ids
            )

        # Start accelerator monitoring
        if self.accelerator_monitor:
            self.accelerator_monitor.start_monitoring()
            logger.info(f"accelerator monitoring started for devices: {device_ids}")

        # Create and start the service manager
        if self.service_manager_class:
            self.service_manager = self.service_manager_class(self.config)
            self.service_manager.start_service(port=DEFAULT_SERVICE_PORT)
        else:
            logger.warning(f"Service manager for {self.config.framework.value} not available, "
                          f"assuming service is already running")
            
            # check service readiness
            if not self._check_service_ready():
                raise RuntimeError("Inference service is not ready")

        # Load trace data
        self._load_trace_data()

    def execute(self) -> None:
        """Execute service inference test"""
        logger.info("Executing service inference benchmark")

        # Run asynchronous trace test
        asyncio.run(self._run_trace_async())

    async def _run_trace_async(self):
        """Asynchronously run trace test"""
        try:
            # Create trace client configuration
            client_config = TraceClientConfig(
                api_url="http://localhost:8000",
                model_name=self.config.model,
                timeout_ms=self.infer_args.timeout_ms,
                warmup_requests=min(10, len(self.traces) // 10)  # 10% of requests for warmup
            )

            # Use trace client to run test
            async with TraceClient(client_config) as client:
                # Run trace
                processed_traces, stats = await client.run_trace(
                    traces=self.traces,
                    concurrency=self.infer_args.concurrency,
                    warmup_requests=client_config.warmup_requests
                )

                # Save results
                self.traces = processed_traces
                self.trace_stats = stats

                # Save trace results to CSV
                if self.infer_dir:
                    client.save_results_to_csv(
                        processed_traces,
                        self.infer_dir,
                        self.config.run_id
                    )

        except Exception as e:
            logger.error(f"Trace test failed: {e}", exc_info=True)
            raise

        finally:
            # Stop accelerator monitoring
            if self.accelerator_monitor:
                self.accelerator_monitor.stop_monitoring()
                peak_memory_gb = self.accelerator_monitor.get_peak_memory_gb()
                logger.info(f"Peak accelerator memory usage during test: {peak_memory_gb} GB")
                self.result.peak_memory_usage = peak_memory_gb

    def collect_metrics(self) -> None:
        """Collect service inference metrics"""
        logger.info("Collecting service inference metrics")

        if not self.trace_stats:
            logger.warning("No trace statistics available")
            return

        # Metric mapping
        metric_map = {
            'avg_ttft': ('infer.avg_ttft', 'ms'),
            'avg_e2e_latency': ('infer.avg_e2e_latency', 'ms'),
            'throughput_tps': ('infer.avg_throughput_tps', 'tokens/s'),
            'total_requests': ('infer.total_requests', 'requests'),
        }

        # Collect TTFT and E2E latency data (single pass)
        for trace in self.traces:
            if trace.success:
                if trace.ttft is not None:
                    self.result.add_ttft(trace.ttft)
                if trace.e2e_latency is not None:
                    self.result.add_latency(trace.e2e_latency)

        # Add standard scalar metrics
        for key, (name, unit) in metric_map.items():
            val = self.trace_stats.get(key)
            if val is not None:
                self.result.add_metric(ScalarMetric(
                    name=name,
                    value=val,
                    unit=unit
                ))

        # Handle special metrics
        self.result.add_metric(ScalarMetric(
            name="infer.success_rate",
            value=self.trace_stats.get('success_rate', 0) * 100,  
            unit="%"
        ))

        # Collect throughput data
        if self.trace_stats.get('total_duration', 0) > 0:
            self.result.add_throughput(self.trace_stats.get('requests_per_second', 0))

        # Record peak memory usage
        if hasattr(self.adapter, 'get_peak_memory_usage'):
            peak_memory = self.adapter.get_peak_memory_usage()
            if peak_memory:
                self.result.peak_memory_usage = peak_memory
                logger.info(f"Peak accelerator memory usage: {peak_memory:.2f} GB")

    def _load_trace_data(self):
        """Load trace data"""
        trace_file = self.infer_args.request_trace

        if not Path(trace_file).exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        logger.info(f"Loading trace data from: {trace_file}")

        # Create temporary prompt generator
        temp_prompt_generator = create_prompt_generator(method="random")

        # Use trace client's method to load trace file
        self.traces = TraceClient.load_trace_file(
            trace_file, 
            temp_prompt_generator
        )

        logger.info(f"Loaded {len(self.traces)} requests from trace file")

        # Validate trace data
        self._validate_trace_data()

    def _validate_trace_data(self):
        """Validate trace data"""
        if not self.traces:
            raise ValueError("No trace data loaded")

        # Check if maximum tokens exceed model limits
        max_input_tokens = max(t.input_token_num for t in self.traces)
        max_output_tokens = max(t.output_token_num for t in self.traces)

        if max_input_tokens > self.infer_args.max_seq_len:
            logger.warning(f"Max input tokens ({max_input_tokens}) exceeds max_seq_len "
                          f"({self.infer_args.max_seq_len})")

        if max_output_tokens > self.infer_args.max_seq_len:
            logger.warning(f"Max output tokens ({max_output_tokens}) exceeds max_seq_len "
                          f"({self.infer_args.max_seq_len})")

        # Check timestamp order
        timestamps = [t.arrival_timestamp_ms for t in self.traces]
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            logger.warning("Trace timestamps are not sorted. Sorting now...")
            self.traces.sort(key=lambda x: x.arrival_timestamp_ms)
    
    def _check_service_ready(self) -> bool:
        """Check that the service is ready"""        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', DEFAULT_SERVICE_PORT))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Port check failed: {e}")
            return False

    def dump_json(self) -> str:
        """Override dump_json to generate standard format metrics"""
        if not self.infer_dir:
            raise ValueError("Output directory not prepared")

        # Use base class method to generate basic JSON
        json_file = super().dump_json()

        # Read base class generated JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build standard metrics array
        standard_metrics = []

        # accuracy_mmlu (placeholder)
        metric_files = {
            'infer.e2e_latency': f"{self.config.run_id}_infer_latency.csv",
            'infer.ttft': f"{self.config.run_id}_infer_ttft.csv",
            'infer.response_per_second': f"{self.config.run_id}_infer_throughput.csv",
            'infer.compute_latency': f"{self.config.run_id}_infer_compute_latency.csv",
            'infer.max_throughput_tps': f"{self.config.run_id}_infer_max_throughput.csv",
        }

        # Accuracy placeholder
        standard_metrics.append({
            "name": "infer.accuracy_mmlu",
            "type": "scalar",
            "value": None,
            "unit": None
        })

        # Dynamically generate time-series metrics
        for metric_name, filename in metric_files.items():
            metric_file = self.infer_dir / filename
            unit = "ms" if "latency" in metric_name or "ttft" in metric_name else None
            
            if metric_file.exists():
                standard_metrics.append({
                    "name": metric_name,
                    "type": "timeseries",
                    "raw_data_url": f"./infer/{filename}",
                    "unit": unit
                })
            else:
                # Handle special logic
                if metric_name == "infer.max_throughput_tps":
                    max_throughput = 0.0
                    if hasattr(self.result, 'throughput_data') and self.result.throughput_data:
                        max_throughput = max(self.result.throughput_data)
                    
                    standard_metrics.append({
                        "name": metric_name,
                        "type": "scalar",
                        "value": max_throughput,
                        "unit": "tokens/s/gpu"
                    })
                else:
                    standard_metrics.append({
                        "name": metric_name,
                        "type": "timeseries",
                        "raw_data_url": None,
                        "unit": unit
                    })

        # Peak memory usage (scalar)
        peak_memory = 0.0
        if self.accelerator_monitor:
            try:
                peak_memory = self.accelerator_monitor.get_peak_memory_gb()
                logger.info(f"Real peak accelerator memory usage: {peak_memory} GB")
            except Exception as e:
                logger.warning(f"Failed to get peak memory from accelerator monitor: {e}")
        else:
            logger.warning("Accelerator monitor not available")

        standard_metrics.append({
            "name": "infer.peak_memory_usage",
            "type": "scalar",
            "value": peak_memory,
            "unit": "GB"
        })

        # Update metrics data
        data["metrics"] = standard_metrics

        # Save back to file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Service metrics saved to: {json_file}")
        return str(json_file)

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up service inference resources")

        # Stop inference service
        if self.service_manager:
            try:
                self.service_manager.stop_service()
            except Exception as e:
                logger.warning(f"Error stopping service via manager: {e}")

