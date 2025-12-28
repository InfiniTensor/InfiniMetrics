#!/usr/bin/env python3
"""
Service Manager for Inference Frameworks
Responsible for starting, stopping, and monitoring services
"""

import subprocess
import threading
import time
import socket
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import signal
import sys
import os

logger = logging.getLogger(__name__)

# Constant definitions
DEFAULT_SERVICE_PORT = 8000
DEFAULT_MAX_WAIT_TIME = 120   # Maximum wait time (seconds)
DEFAULT_WAIT_INTERVAL = 3     # Wait interval (seconds)
DEFAULT_SERVICE_TIMEOUT = 30  # Service stop timeout (seconds)

class BaseServiceManager:
    """Base service manager abstract class"""
    
    def __init__(self, config: Any):
        self.config = config
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = DEFAULT_SERVICE_PORT
        self.service_started = False
        self.output_thread: Optional[threading.Thread] = None
        
    def start_service(self, port: int = DEFAULT_SERVICE_PORT) -> None:
        """Start service (abstract method)"""
        raise NotImplementedError
    
    def stop_service(self) -> None:
        """Stop service"""
        if self.server_process:
            logger.info("Stopping inference service")
            
            try:
                # Graceful shutdown
                self.server_process.terminate()
                self.server_process.wait(timeout=DEFAULT_SERVICE_TIMEOUT)
                logger.info("Service stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Service did not stop gracefully, forcing kill")
                self.server_process.kill()
                self.server_process.wait()
            
            self.server_process = None
        
        self.service_started = False
    
    def is_service_ready(self, port: Optional[int] = None) -> bool:
        """Check whether the service is ready"""
        if not self.service_started or not self.server_process:
            return False
        
        # Check if process is alive
        if self.server_process.poll() is not None:
            return_code = self.server_process.returncode
            logger.error(f"Server process died with return code: {return_code}")
            return False
        
        # Check if port is open
        check_port = port or self.server_port
        return self._check_port_open(check_port)
    
    def get_service_url(self) -> str:
        """Get service URL"""
        return f"http://localhost:{self.server_port}"
    
    def wait_for_service_ready(self, timeout: int = DEFAULT_MAX_WAIT_TIME, 
                               check_interval: int = DEFAULT_WAIT_INTERVAL) -> bool:
        """Wait until the service is ready"""
        logger.info("Waiting for service to be ready...")
        
        max_checks = timeout // check_interval
        for i in range(max_checks):
            if self.is_service_ready():
                logger.info("Service is ready")
                return True
            
            elapsed_time = (i + 1) * check_interval
            logger.debug(f"Waiting... ({elapsed_time}s elapsed)")
            time.sleep(check_interval)
        
        logger.error(f"Service failed to start within {timeout} seconds")
        return False
    
    def _check_port_open(self, port: int) -> bool:
        """Check whether a port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Port check failed: {e}")
            return False
    
    def _start_output_reader(self, process_stdout):
        """Start output reader thread"""
        def read_output():
            for line in process_stdout:
                logger.info(f"[Service] {line.strip()}")
        
        self.output_thread = threading.Thread(target=read_output, daemon=True)
        self.output_thread.start()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure service is stopped"""
        self.stop_service()


class InfiniLMServiceManager(BaseServiceManager):
    """InfiniLM service manager"""
    
    def _locate_infinilm_scripts(self) -> Optional[Path]:
        """Locate InfiniLM scripts directory"""
        search_paths = []
        
        # 1. Environment variable
        env_path = os.environ.get("INFINILM_PATH")
        if env_path:
            search_paths.append(Path(env_path))
        
        # 2. Current working directory
        search_paths.append(Path.cwd())
        
        # 3. Common locations
        search_paths.extend([
            Path.home() / "InfiniLM",
            Path("/opt/InfiniLM"),
            Path("/usr/local/InfiniLM"),
        ])
        
        # Search
        for base_path in search_paths:
            scripts_dir = base_path / "scripts"
            if scripts_dir.exists():
                logger.info(f"Found InfiniLM scripts at: {scripts_dir}")
                return scripts_dir
        
        logger.error("Cannot locate InfiniLM scripts directory")
        return None
    
    def _build_start_command(self, launch_script: Path, port: int) -> List[str]:
        """Build start command"""
        cmd = [
            sys.executable, str(launch_script),
            "--model-path", str(self.config.model_path),
            "--ndev", str(self.config.infer_args.parallel.tp),
            "--max-batch", "4"
        ]
        
        # Device type
        accelerator = self.config.device.accelerator.value.lower()
        cmd.extend(["--dev", accelerator])
        
        # Port
        if port != DEFAULT_SERVICE_PORT:
            cmd.extend(["--port", str(port)])
        
        # Maximum sequence length
        max_seq_len = self.config.infer_args.max_seq_len
        if max_seq_len and max_seq_len != 4096:
            cmd.extend(["--max-tokens", str(max_seq_len)])
        
        return cmd
    
    def start_service(self, port: int = DEFAULT_SERVICE_PORT) -> None:
        """Start InfiniLM service"""
        logger.info(f"Starting InfiniLM service on port {port}")
        
        # Locate scripts directory
        scripts_dir = self._locate_infinilm_scripts()
        if not scripts_dir:
            raise RuntimeError("Cannot locate InfiniLM scripts directory")
        
        launch_script = scripts_dir / "launch_server.py"
        if not launch_script.exists():
            raise FileNotFoundError(f"Launch script not found: {launch_script}")
        
        # Build start command
        cmd = self._build_start_command(launch_script, port)
        
        # Start service process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.server_port = port
        self.service_started = True
        
        # Start output reader thread
        self._start_output_reader(self.server_process.stdout)
        
        logger.info(f"InfiniLM service started with PID: {self.server_process.pid}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # 5. Wait for service to be ready
        if not self.wait_for_service_ready():
            self.stop_service()
            raise TimeoutError("InfiniLM service failed to start within timeout")
