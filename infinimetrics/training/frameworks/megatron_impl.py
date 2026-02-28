import os
import re
import random
import subprocess
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MegatronImpl:
    """
    Megatron-LM training executor.

    Pure execution - no monitoring responsibilities.
    Receives resolved device count from adapter.
    """

    MODEL_SCRIPT_MAP = {
        "gpt": "pretrain_gpt.py",
        "bert": "pretrain_bert.py",
        "llama": "pretrain_gpt.py",
        "t5": "pretrain_t5.py",
        "ict": "pretrain_ict.py",
        "mamba": "pretrain_mamba.py",
        "retro": "pretrain_retro.py",
        "vision_classify": "pretrain_vision_classify.py",
        "vision_dino": "pretrain_vision_dino.py",
        "vision_inpaint": "pretrain_vision_inpaint.py",
        "vlm": "pretrain_vlm.py",
    }

    def __init__(
        self,
        config: Dict[str, Any],
        resolved_device_count: int,
        run_id: Optional[str] = None,
    ):
        """
        Args:
            config: Full configuration dict
            resolved_device_count: Number of available devices (from monitor)
        """
        self.config = config
        self.resolved_device_count = resolved_device_count
        self.logger = logger

        # Extract run_id and output paths
        self.run_id = run_id if run_id is not None else config.get("run_id", "unknown")
        self.logger.info(f"MegatronRunner using run_id: {self.run_id}")

        self.output_dir = Path(config.get("output_dir", "./output")) / "training"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training args
        train_args = config.get("train_args", {})
        self.model_name = config.get("model", "gpt")
        self.megatron_path = config.get("megatron_path", "")

        # Parallel config
        parallel = train_args.get("parallel", {})
        self.tp = parallel.get("tp", 1)
        self.pp = parallel.get("pp", 1)
        self.dp = parallel.get("dp", 1)
        self.sp = parallel.get("sp", 0)

        # Model config
        self.num_layers = train_args.get("num_layers", 12)
        self.hidden_size = train_args.get("hidden_size", 768)
        self.num_attention_heads = train_args.get("num_attention_heads", 12)
        self.seq_len = train_args.get("seq_len", 1024)
        self.vocab_size = train_args.get("vocab_size", 32000)
        self.max_position_embeddings = train_args.get(
            "max_position_embeddings", self.seq_len
        )

        # Training hyperparams
        self.mbs = train_args.get("mbs", 1)
        self.gbs = train_args.get("gbs", self.mbs)
        self.train_iters = train_args.get("train_iters", 10)
        self.lr = train_args.get("lr", 1e-4)
        self.precision = train_args.get("precision", "fp16")
        self.optimizer = train_args.get("optimizer", "adam")
        self.weight_decay = train_args.get("weight_decay", 0.0)
        self.clip_grad = train_args.get("clip_grad", 1.0)
        self.beta1 = train_args.get("beta1", 0.9)
        self.beta2 = train_args.get("beta2", 0.999)
        self.lr_scheduler = train_args.get("lr_scheduler", "cosine")
        self.min_lr = train_args.get("min_lr", 0.0)
        self.warmup_iterations = train_args.get("warmup_iterations", 0)

        # Dataset
        self.train_dataset = config.get("train_dataset", "mock")
        self.validation_dataset = config.get("validation_dataset")
        self.test_dataset = config.get("test_dataset")

        # Evaluation
        self.eval_interval = train_args.get("eval_interval", 100)
        self.eval_iters = train_args.get("eval_iters", 10)

        # Checkpoint
        self.save_interval = train_args.get("save_interval", 1000)

        # Calculate nproc_per_node
        self.nproc_per_node = self._calculate_nproc_per_node()
        self.master_port = random.randint(20000, 60000)

        # Regex patterns
        self.iter_pattern = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        self.loss_pattern = re.compile(
            r"lm loss:\s*([+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)", re.IGNORECASE
        )
        self.elapsed_pattern = re.compile(
            r"elapsed time per iteration \(ms\):\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
        )

        # Output files
        self.log_file = self.output_dir / f"{self.run_id}_train.log"
        self.loss_csv = self.output_dir / f"{self.run_id}_train_loss.csv"
        self.ppl_csv = self.output_dir / f"{self.run_id}_train_ppl.csv"
        self.throughput_csv = self.output_dir / f"{self.run_id}_train_throughput.csv"

    def run(self) -> Dict[str, Any]:
        """Execute Megatron training."""
        cmd = self._build_command()
        self.logger.info(f"Launching: {' '.join(cmd)}")

        # Setup environment (CUDA_VISIBLE_DEVICES)
        env = self._get_env()

        # Metrics collection
        metrics = {
            "losses_by_iter": {},
            "throughput_by_iter": {},
            "last_seen_iter": None,
        }

        # Start process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # Process output
        with open(self.log_file, "w") as f:
            for line in proc.stdout:
                line = line.rstrip("\n")
                self.logger.info(line)
                f.write(line + "\n")
                self._parse_output(line, metrics)

        proc.wait()

        # Save results
        self._save_results(metrics)

        return {
            "log_file": str(self.log_file),
            "loss_csv": str(self.loss_csv),
            "ppl_csv": str(self.ppl_csv),
            "throughput_csv": str(self.throughput_csv),
            "metrics": metrics,
        }

    def _build_command(self) -> list:
        """Build torchrun command."""
        train_script = self.MODEL_SCRIPT_MAP.get(self.model_name, "pretrain_gpt.py")

        if self.megatron_path:
            train_script = f"{self.megatron_path}/{train_script}"

        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={self.nproc_per_node}",
            f"--master_port={self.master_port}",
        ]

        megatron_args = [train_script]

        # Parallel config
        megatron_args += [
            f"--tensor-model-parallel-size={self.tp}",
            f"--pipeline-model-parallel-size={self.pp}",
        ]

        # Model config
        megatron_args += [
            f"--num-layers={self.num_layers}",
            f"--hidden-size={self.hidden_size}",
            f"--num-attention-heads={self.num_attention_heads}",
            f"--seq-length={self.seq_len}",
            f"--max-position-embeddings={self.max_position_embeddings}",
            f"--vocab-size={self.vocab_size}",
        ]

        # Training params
        megatron_args += [
            f"--micro-batch-size={self.mbs}",
            f"--global-batch-size={self.gbs}",
            f"--train-iters={self.train_iters}",
            f"--lr={self.lr}",
        ]

        # Precision
        if self.precision == "bf16":
            megatron_args.append("--bf16")
        elif self.precision == "fp16":
            megatron_args.append("--fp16")

        # Optimizer
        megatron_args.append(f"--optimizer={self.optimizer}")
        if self.weight_decay > 0:
            megatron_args.append(f"--weight-decay={self.weight_decay}")
        if self.clip_grad > 0:
            megatron_args.append(f"--clip-grad={self.clip_grad}")
        if self.optimizer.lower() in ["adam", "adamw"]:
            megatron_args.append(f"--adam-beta1={self.beta1}")
            megatron_args.append(f"--adam-beta2={self.beta2}")

        # LR scheduler
        megatron_args.append(f"--lr-decay-style={self.lr_scheduler}")
        if self.min_lr > 0:
            megatron_args.append(f"--min-lr={self.min_lr}")

        # Warmup (simplified - in real code would handle ratio/iters/samples)
        if self.warmup_iterations > 0:
            megatron_args.append(f"--lr-warmup-iters={self.warmup_iterations}")

        # Dataset
        if self.train_dataset == "mock":
            megatron_args += ["--mock-data", "--tokenizer-type", "NullTokenizer"]
        else:
            megatron_args.append(f"--data-path={self.train_dataset}")
            if self.validation_dataset:
                megatron_args.append(
                    f"--validation-data-path={self.validation_dataset}"
                )
            if self.test_dataset:
                megatron_args.append(f"--test-data-path={self.test_dataset}")

        # Common args
        megatron_args += [
            "--transformer-impl",
            "local",
            "--no-gradient-accumulation-fusion",
            "--no-persist-layer-norm",
            "--log-interval",
            "1",
            "--log-throughput",
        ]

        # Evaluation
        if self.eval_interval > 0:
            megatron_args.append(f"--eval-interval={self.eval_interval}")
            megatron_args.append(f"--eval-iters={self.eval_iters}")

        # Checkpoint
        if self.save_interval > 0:
            megatron_args.append(f"--save-interval={self.save_interval}")

        # Sequence parallel
        if self.sp == 1:
            megatron_args.append("--sequence-parallel")

        # Extra args
        extra_args = self.config.get("train_args", {}).get("extra_args", [])
        if extra_args:
            megatron_args.extend(extra_args)

        return torchrun_cmd + megatron_args

    def _get_env(self) -> Dict[str, str]:
        """Get environment variables for subprocess."""
        env = os.environ.copy()

        # Set CUDA_VISIBLE_DEVICES if specified
        device_config = self.config.get("device", {})
        device_ids = device_config.get("device_ids")
        if device_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
            self.logger.info(f"Set CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

        return env

    def _parse_output(self, line: str, metrics: Dict[str, Any]) -> None:
        """Parse Megatron output line."""
        # Update iteration
        m_iter = self.iter_pattern.search(line)
        if m_iter:
            try:
                metrics["last_seen_iter"] = int(m_iter.group(1))
            except Exception:
                pass

        # Extract loss
        m_loss = self.loss_pattern.search(line)
        if m_loss and metrics["last_seen_iter"] is not None:
            try:
                val = float(m_loss.group(1))
                if metrics["last_seen_iter"] > self.warmup_iterations:
                    metrics["losses_by_iter"][metrics["last_seen_iter"]] = val
            except Exception:
                pass

        # Extract throughput
        m_elapsed = self.elapsed_pattern.search(line)
        if m_elapsed and metrics["last_seen_iter"] is not None:
            try:
                elapsed_ms = float(m_elapsed.group(1))
                # tokens per iteration = mbs * seq_len * dp
                tokens_per_iter = self.mbs * self.seq_len * self.dp
                if elapsed_ms > 0:
                    throughput = (
                        tokens_per_iter / (elapsed_ms / 1000.0) / self.nproc_per_node
                    )
                    if metrics["last_seen_iter"] > self.warmup_iterations:
                        metrics["throughput_by_iter"][
                            metrics["last_seen_iter"]
                        ] = throughput
            except Exception:
                pass

    def _save_results(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to CSV."""
        # Save loss
        with open(self.loss_csv, "w") as f:
            f.write("iteration,loss\n")
            for it, val in sorted(metrics["losses_by_iter"].items()):
                f.write(f"{it},{val}\n")

        # Save PPL
        with open(self.ppl_csv, "w") as f:
            f.write("iteration,ppl\n")
            for it, loss in sorted(metrics["losses_by_iter"].items()):
                try:
                    ppl = float(math.exp(loss))
                except Exception:
                    ppl = float("inf")
                f.write(f"{it},{ppl}\n")

        # Save throughput
        with open(self.throughput_csv, "w") as f:
            f.write("iteration,throughput\n")
            for it, val in sorted(metrics["throughput_by_iter"].items()):
                f.write(f"{it},{val}\n")

        self.logger.info(f"Results saved to {self.output_dir}")

    def _calculate_nproc_per_node(self) -> int:
        """Calculate number of processes per node using resolved device count."""
        desired_world = self.dp * self.tp * max(1, self.pp)
        return (
            min(desired_world, self.resolved_device_count)
            if self.resolved_device_count > 0
            else desired_world
        )
