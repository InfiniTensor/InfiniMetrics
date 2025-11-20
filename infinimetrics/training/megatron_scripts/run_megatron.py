#!/usr/bin/env python3
import json
import subprocess
import argparse
import os
import re
import time
import math
import threading
import uuid
import torch

# GPU Memory Monitor
gpu_peak_mem_mb = [0]

def monitor_gpu_memory(proc, poll_interval=0.5):
    """Monitor GPU peak memory via nvidia-smi while proc is running.
    Updates global gpu_peak_mem_mb[0] with max MiB seen.
    """
    while True:
        if proc.poll() is not None:
            break
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL
            )
            # parse lines -> ints
            mems = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
            if mems:
                gpu_peak_mem_mb[0] = max(gpu_peak_mem_mb[0], max(mems))
        except Exception:
            # ignore transient read errors (e.g., nvidia-smi not present)
            pass
        time.sleep(poll_interval)

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()

    # load config
    with open(args.config, "r") as f:
        cfg = json.load(f)

    train = cfg["config"].get("train_args", {})
    parallel = train.get("parallel", {})

    dp = parallel.get("dp", 1)
    tp = parallel.get("tp", 1)
    pp = parallel.get("pp", {}).get("value", 1)
    sp = parallel.get("sp", 0)

    # training params with defaults
    mbs = train.get("mbs", 1)
    gbs = train.get("gbs", 1)
    seq = train.get("seq_len", 128)
    lr = train.get("lr", 0.00015)
    step = train.get("step", 10)
    num_layers = train.get("num_layers", 2)
    hidden_size = train.get("hidden_size", 512)
    num_attention_heads = train.get("num_attention_heads", 8)
    max_position_embeddings = train.get("max_position_embeddings", seq)
    vocab_size = train.get("vocab_size", 128256)

    # keep placeholder run_id
    run_id = "{run_id}"

    # determine world size / nproc_per_node
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    desired_world = max(1, dp * tp * pp)
    if available_gpus > 0:
        nproc_per_node = min(desired_world, available_gpus)
    else:
        nproc_per_node = max(1, desired_world)

    # build torchrun + megatron args
    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "--master_port=29501"
    ]

    megatron_args = [
        "pretrain_gpt.py",
        f"--tensor-model-parallel-size={tp}",
        f"--pipeline-model-parallel-size={pp}",
        f"--micro-batch-size={mbs}",
        f"--global-batch-size={gbs}",
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

    # output paths
    output_dir = "./train"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{run_id}_train.log")
    loss_csv = os.path.join(output_dir, f"{run_id}_train_loss.csv")
    ppl_csv = os.path.join(output_dir, f"{run_id}_train_ppl.csv")
    throughput_csv = os.path.join(output_dir, f"{run_id}_train_throughput.csv")
    result_json = os.path.join(output_dir, f"{run_id}_result.json")

    # regex patterns
    loss_pattern = re.compile(r"lm loss:\s*([+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)", re.IGNORECASE)
    ppl_pattern_alt = re.compile(r"lm loss PPL:\s*([+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)", re.IGNORECASE)
    elapsed_pattern = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9]*\.?[0-9]+)")

    losses = []
    throughputs = []

    # launch process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # start gpu monitor thread
    monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(process,), daemon=True)
    monitor_thread.start()

    # read stdout line-by-line and parse
    with open(log_file, "w") as flog:
        for line in process.stdout:
            # print to console and write to log
            print(line, end="")
            flog.write(line)

            # try match loss
            m = loss_pattern.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    losses.append(val)
                except Exception:
                    pass

            # try match elapsed -> throughput
            me = elapsed_pattern.search(line)
            if me:
                try:
                    elapsed_ms = float(me.group(1))
                    tokens_per_iter = mbs * seq 
                    # throughput tokens per second
                    throughput = tokens_per_iter / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
                    throughputs.append(throughput)
                except Exception:
                    pass

    # wait for process end and thread join
    process.wait()
    monitor_thread.join()

    peak_memory_gb = gpu_peak_mem_mb[0] / 1024.0

    ppls = []
    for loss in losses:
        try:
            # loss is natural-log-based in Megatron; compute exp(loss)
            ppls.append(float(math.exp(loss)))
        except OverflowError:
            ppls.append(float("inf"))

    with open(loss_csv, "w") as f:
        f.write("iteration,loss\n")
        for i, v in enumerate(losses, start=1):
            f.write(f"{i},{v}\n")

    with open(ppl_csv, "w") as f:
        f.write("iteration,ppl\n")
        for i, v in enumerate(ppls, start=1):
            f.write(f"{i},{v}\n")

    with open(throughput_csv, "w") as f:
        f.write("iteration,throughput\n")
        for i, v in enumerate(throughputs, start=1):
            f.write(f"{i},{v}\n")

    # create result json
    result = {
        "config": {
            "command": " ".join(cmd),
            "model": cfg.get("config", {}).get("model", "Megatron-GPT"),
            "model_config": cfg.get("config", {}).get("model_config", ""),
            "train_dataset": cfg.get("config", {}).get("train_dataset", "mock"),
            "validation_dataset": cfg.get("config", {}).get("validation_dataset", None),
            "test_dataset": cfg.get("config", {}).get("test_dataset", None),
            "train_args": train,
            "timeout_ms": train.get("timeout_ms", 10000),
            "warmup_iterations": train.get("warmup_iterations", 100),
            "measured_iterations": train.get("measured_iterations", step)
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
    print(f"Peak GPU memory (GiB): {peak_memory_gb:.6f}")

if __name__ == "__main__":
    main()

