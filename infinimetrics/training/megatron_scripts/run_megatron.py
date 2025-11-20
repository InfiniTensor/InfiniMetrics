#!/usr/bin/env python3
import json
import subprocess
import argparse
import os
import re
import uuid
import shutil
import torch
import threading
import time


# -------------------------
# GPU Memory Monitor
# -------------------------
gpu_peak_mem_mb = [0]

def monitor_gpu_memory(proc):
    """Monitor GPU peak memory while training is running."""
    while True:
        if proc.poll() is not None:  # training finished
            break
        try:
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            mem_list = [int(x) for x in smi_out.strip().split("\n")]
            gpu_peak_mem_mb[0] = max(gpu_peak_mem_mb[0], max(mem_list))
        except Exception:
            pass
        time.sleep(0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------
    # Load Config
    # -------------------------
    with open(args.config, "r") as f:
        cfg = json.load(f)

    train = cfg["config"].get("train_args", {})
    parallel = train.get("parallel", {})

    dp = parallel.get("dp", 1)
    tp = parallel.get("tp", 1)
    pp = parallel.get("pp", {}).get("value", 1)
    sp = parallel.get("sp", 0)

    # Training parameters
    mbs = train.get("mbs", 1)
    seq = train.get("seq_len", 128)
    lr = train.get("lr", 0.00015)
    step = train.get("step", 10)
    num_layers = train.get("num_layers", 2)
    hidden_size = train.get("hidden_size", 512)
    num_attention_heads = train.get("num_attention_heads", 8)
    max_position_embeddings = train.get("max_position_embeddings", seq)
    vocab_size = train.get("vocab_size", 128256)

    run_id = "{run_id}"

    # -------------------------
    # Distributed world size
    # -------------------------
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    desired_world = max(1, dp * tp * pp)
    nproc_per_node = min(desired_world, available_gpus) if available_gpus > 0 else 1

    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "--master_port=29501"
    ]

    # -------------------------
    # Megatron Arguments
    # -------------------------
    megatron_args = [
        "pretrain_gpt.py",
        f"--tensor-model-parallel-size={tp}",
        f"--pipeline-model-parallel-size={pp}",
        f"--micro-batch-size={mbs}",
        f"--seq-length={seq}",
        f"--lr={lr}",
        f"--train-iters={step}",
        f"--num-layers={num_layers}",
        f"--hidden-size={hidden_size}",
        f"--num-attention-heads={num_attention_heads}",
        f"--max-position-embeddings={max_position_embeddings}",
        f"--vocab-size={vocab_size}",
        "--mock-data",
        "--tokenizer-type", "NullTokenizer",
        "--transformer-impl", "local",
        "--bf16",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--log-interval", "1",
        "--log-throughput"
    ]

    if sp == 1:
        megatron_args.append("--sequence-parallel")

    cmd = torchrun_cmd + megatron_args
    print("Launching:", " ".join(cmd))

    # -------------------------
    # Output paths
    # -------------------------
    output_dir = "./train"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"{run_id}_train.log")
    loss_csv = os.path.join(output_dir, f"{run_id}_train_loss.csv")
    ppl_csv = os.path.join(output_dir, f"{run_id}_train_ppl.csv")
    throughput_csv = os.path.join(output_dir, f"{run_id}_train_throughput.csv")
    result_json = os.path.join(output_dir, f"{run_id}_result.json")

    # -------------------------
    # Log Parsing Regex
    # -------------------------
    loss_pattern = re.compile(r"lm loss: ([\d\.\-E+]+)")
    ppl_pattern = re.compile(r"lm loss PPL: ([\d\.\-E+]+)")
    elapsed_pattern = re.compile(r"elapsed time per iteration \(ms\): ([\d\.]+)")

    losses = []
    ppls = []
    throughputs = []

    # -------------------------
    # Launch Training Process
    # -------------------------
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    # Start GPU memory monitor thread
    monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(process,))
    monitor_thread.start()

    # -------------------------
    # Read and Parse Output Log
    # -------------------------
    with open(log_file, "w") as f_log:
        for line in process.stdout:
            print(line, end="")
            f_log.write(line)

            if (m := loss_pattern.search(line)):
                losses.append(float(m.group(1)))

            if (m := ppl_pattern.search(line)):
                ppls.append(float(m.group(1)))

            if (m := elapsed_pattern.search(line)):
                elapsed_ms = float(m.group(1))
                tokens_per_iter = mbs * seq
                throughput = tokens_per_iter / (elapsed_ms / 1000.0)
                throughputs.append(throughput)

    process.wait()
    monitor_thread.join()

    peak_memory_gb = gpu_peak_mem_mb[0] / 1024.0

    # -------------------------
    # Write CSV Files
    # -------------------------
    with open(loss_csv, "w") as f:
        f.write("iteration,loss\n")
        for i, v in enumerate(losses, 1):
            f.write(f"{i},{v}\n")

    with open(ppl_csv, "w") as f:
        f.write("iteration,ppl\n")
        for i, v in enumerate(ppls, 1):
            f.write(f"{i},{v}\n")

    with open(throughput_csv, "w") as f:
        f.write("iteration,throughput\n")
        for i, v in enumerate(throughputs, 1):
            f.write(f"{i},{v}\n")

    # -------------------------
    # Write Result JSON
    # -------------------------
    result = {
        "config": {
            "command": " ".join(cmd),
            "model": cfg["config"].get("model", "Megatron-GPT"),
            "model_config": cfg["config"].get("model_config", ""),
            "train_dataset": cfg["config"].get("train_dataset", "mock"),
            "train_args": train,
        },
        "metrics": [
            {
                "name": "train.throughput",
                "type": "timeseries",
                "raw_data_url": throughput_csv,
                "unit": "tokens/s/gpu"
            },
            {
                "name": "train.peak_memory_usage",
                "type": "scalar",
                "value": peak_memory_gb,
                "unit": "GB"
            },
            {
                "name": "train.loss",
                "type": "timeseries",
                "raw_data_url": loss_csv,
                "unit": ""
            },
            {
                "name": "train.ppl",
                "type": "timeseries",
                "raw_data_url": ppl_csv,
                "unit": None
            }
        ]
    }

    with open(result_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult JSON written to {result_json}")
    print("Log written to", log_file)
    print("CSV files:", loss_csv, ppl_csv, throughput_csv)


if __name__ == "__main__":
    main()

