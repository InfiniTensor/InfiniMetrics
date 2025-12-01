import json
import math
import os
import uuid
from abc import ABC, abstractmethod

class TrainingRunner(ABC):
    """Abstract base class for training runners"""
    
    def __init__(self, config_manager, gpu_monitor):
        self.config = config_manager
        self.gpu_monitor = gpu_monitor
        self.run_id = f"train.{self.config.model_name}.{uuid.uuid4()}"
        self.setup_output_files()
    
    def setup_output_files(self):
        """Setup output file paths"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.config.output_dir, f"{self.run_id}_train.log")
        self.loss_csv = os.path.join(self.config.output_dir, f"{self.run_id}_train_loss.csv")
        self.ppl_csv = os.path.join(self.config.output_dir, f"{self.run_id}_train_ppl.csv")
        self.throughput_csv = os.path.join(self.config.output_dir, f"{self.run_id}_train_throughput.csv")
        self.result_json = os.path.join(self.config.output_dir, f"{self.run_id}_result.json")
    
    @abstractmethod
    def build_training_command(self):
        """Build training command"""
        pass
    
    @abstractmethod
    def parse_training_output(self, line, metrics):
        """Parse training output and extract metrics"""
        pass
    
    def run(self):
        """Execute training process"""
        import subprocess
        
        # Build training command
        cmd = self.build_training_command()
        
        # Prepare metrics collection
        metrics = {
            'losses_by_iter': {},
            'throughput_by_iter': {},
            'last_seen_iter': None
        }
        
        # Start training process
        print("Launching:", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Start GPU monitoring
        monitor_thread = self.gpu_monitor.monitor_process(proc)
    
        # Process training output
        with open(self.log_file, "w") as flog:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                print(line)
                flog.write(line + "\n")
                self.parse_training_output(line, metrics)
        
        # Wait for process to finish
        proc.wait()
        monitor_thread.join(timeout=1.0)
        
        # Post-processing and data saving
        self.save_results(metrics)
        return self.result_json
    
    def save_results(self, metrics):
        """Save training results"""
        # Calculate PPL
        ppls_by_iter = {}
        for it, loss in metrics['losses_by_iter'].items():
            try:
                ppls_by_iter[it] = float(math.exp(loss))
            except Exception:
                ppls_by_iter[it] = float("inf")
        
        # Save CSV files
        self._save_csv(self.loss_csv, "iteration,loss", metrics['losses_by_iter'])
        self._save_csv(self.ppl_csv, "iteration,ppl", ppls_by_iter)
        self._save_csv(self.throughput_csv, "iteration,throughput", metrics['throughput_by_iter'])
        
        # Save model configuration
        self._save_model_config()
        
        # Save result JSON
        self._save_result_json()
    
    def _save_csv(self, filename, header, data_dict):
        """Save data to CSV file"""
        with open(filename, "w") as f:
            f.write(header + "\n")
            for it, v in sorted(data_dict.items(), key=lambda x: x[0]):
                f.write(f"{it},{v}\n")
    
    def _save_model_config(self):
        """Save model configuration"""
        model_config_path = os.path.join(self.config.output_dir, f"train_{self.config.model_name}_config.json")
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
        with open(model_config_path, "w") as f:
            json.dump(self.config.get_model_config(), f, indent=2)
    
    def _save_result_json(self):
        """Save result JSON file"""
        result = {
            "config": {
                "command": " ".join(self.build_training_command()),
                "model": self.config.model_name,
                "model_config": os.path.join(self.config.output_dir, f"train_{self.config.model_name}_config.json"),
                "train_dataset": self.config.train_dataset if self.config.train_dataset is not None else "mock",
                "validation_dataset": self.config.validation_dataset,
                "test_dataset": self.config.test_dataset,
                "train_args": self.config.train_args,
                "timeout_ms": self.config.timeout_ms,
                "warmup_iterations": self.config.warmup_iterations,
                "measured_iterations": self.config.measured_iterations
            },
            "metrics": [
                {
                    "name": "train.throughput",
                    "type": "timeseries",
                    "raw_data_url": self.throughput_csv,
                    "unit": "tokens/s/gpu"
                },
                {
                    "name": "train.peak_memory_usage",
                    "type": "scalar",
                    "value": self.gpu_monitor.get_peak_memory_gb(),
                    "unit": "GB"
                },
                {
                    "name": "train.loss",
                    "type": "timeseries",
                    "raw_data_url": self.loss_csv,
                    "unit": ""
                },
                {
                    "name": "train.ppl",
                    "type": "timeseries",
                    "raw_data_url": self.ppl_csv,
                    "unit": None
                }
            ]
        }
        
        with open(self.result_json, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResult JSON written to: {self.result_json}")
        print("Log written to:", self.log_file)
        print("CSV files:", self.loss_csv, self.ppl_csv, self.throughput_csv)
        print(f"Peak GPU memory: {self.gpu_monitor.get_peak_memory_mib()} MiB ({self.gpu_monitor.get_peak_memory_gb():.4f} GiB)")
