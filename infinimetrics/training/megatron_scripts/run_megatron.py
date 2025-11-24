#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import subprocess
import threading
import time
import uuid

# optional: try to import torch to detect GPUs; if not available, fallback to 0
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# Select a training script
MODEL_SCRIPT_MAP = {
    "gpt": "pretrain_gpt.py",
    "bert": "pretrain_bert.py",
    "llama": "pretrain_gpt.py",  # some repos keep llama under gpt entry; adjust if you have separate LLaMA entry
    "t5": "pretrain_t5.py",
    "ict": "pretrain_ict.py",
    "mamba": "pretrain_mamba.py",
    "retro": "pretrain_retro.py",
    "vision_classify": "pretrain_vision_classify.py",
    "vision_dino": "pretrain_vision_dino.py",
    "vision_inpaint": "pretrain_vision_inpaint.py",
    "vlm": "pretrain_vlm.py"
}

def pick_random_port(low=20000, high=60000):
    return random.randint(low, high)

def monitor_gpu_memory(proc, poll_interval=0.5, peak_mem_holder=None):
    if peak_mem_holder is None:
        return
    while True:
        if proc.poll() is not None:
            break
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            mems = [int(x) for x in lines] if lines else []
            if mems:
                peak_mem_holder[0] = max(peak_mem_holder[0], max(mems))
        except Exception:
            pass
        time.sleep(poll_interval)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to config.json")
    return p.parse_args()

def main():
    args = parse_args()

    # load config
    with open(args.config, "r") as f:
        cfg = json.load(f)

    conf = cfg.get("config", {})

    # model selection
    model_name = conf.get("model", "gpt").lower()
    train_script = MODEL_SCRIPT_MAP.get(model_name, MODEL_SCRIPT_MAP["gpt"])

    # train_args and parallel
    train_args = conf.get("train_args", {})
    parallel = train_args.get("parallel", {})
    dp = int(parallel.get("dp", 1))
    tp = int(parallel.get("tp", 1))

    pp_raw = parallel.get("pp", 1)

    if isinstance(pp_raw, dict):
        pp_size = int(pp_raw.get("size", 1))
        pp_type = str(pp_raw.get("type", "default")).lower()
    else:
        pp_size = int(pp_raw)
        pp_type = "default"

    ALLOWED_TYPES = ["default", "1f1b", "interleaved", "gpipe"]
    if pp_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported PP type '{pp_type}'. Allowed = {ALLOWED_TYPES}")

    # Megatron-supported mapping
    if pp_type in ["default", "1f1b"]:
        # default Megatron behavior → 1F1B
        use_virtual_pp = False
        virtual_pp_value = None

    elif pp_type == "interleaved":
        # require virtual pipeline, user must supply train_args.virtual_pp or default = 2
        use_virtual_pp = True
        virtual_pp_value = int(train_args.get("virtual_pp", 2))
        if virtual_pp_value <= 1:
            raise ValueError("interleaved PP requires virtual_pp > 1")

    elif pp_type == "gpipe":
        # Megatron cannot support GPipe direcly
        raise ValueError(
            "PP type 'gpipe' is not supported by Megatron-LM. "
            "It must be handled externally by another framework."
        )

    else:
        raise ValueError(f"Invalid pp_type = {pp_type}")
    pp_val = pp_size

    sp = int(parallel.get("sp", 0))

    # training fields with defaults
    mbs = int(train_args.get("mbs", 1))
    gbs = int(train_args.get("gbs", max(1, mbs)))
    seq_len = int(train_args.get("seq_len", train_args.get("seq", 128)))
    lr = train_args.get("lr", 0.00015)
    train_iters = int(train_args.get("step", train_args.get("train_iters", train_args.get("train-iters", 10))))
    num_layers = int(train_args.get("num_layers", train_args.get("num-layers", 2)))
    hidden_size = int(train_args.get("hidden_size", train_args.get("hidden-size", 512)))
    num_attention_heads = int(train_args.get("num_attention_heads", train_args.get("num-attention-heads", 8)))
    max_position_embeddings = int(train_args.get("max_position_embeddings", train_args.get("max-position-embeddings", seq_len)))
    vocab_size = int(train_args.get("vocab_size", train_args.get("vocab-size", 128256)))
    warmup_iters = int(train_args.get("warmup_iterations", train_args.get("warmup", 0)))

    # datasets
    train_dataset = conf.get("train_dataset", None)
    validation_dataset = conf.get("validation_dataset", None)
    test_dataset = conf.get("test_dataset", None)

    # run id and output files
    model_name = conf.get("model", model_name)
    run_id = f"train.{model_name}.{uuid.uuid4()}"
    out_dir = conf.get("output_dir", "./train")
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, f"{run_id}_train.log")
    loss_csv = os.path.join(out_dir, f"{run_id}_train_loss.csv")
    ppl_csv = os.path.join(out_dir, f"{run_id}_train_ppl.csv")
    throughput_csv = os.path.join(out_dir, f"{run_id}_train_throughput.csv")
    result_json = os.path.join(out_dir, f"{run_id}_result.json")

    # compute world size desired
    desired_world = max(1, dp * tp * max(1, pp_val))
    if _TORCH_AVAILABLE:
        available_gpus = torch.cuda.device_count()
    else:
        available_gpus = 0

    # choose nproc_per_node
    if available_gpus > 0:
        nproc_per_node = min(desired_world, available_gpus)
    else:
        # assume single-node run; let torchrun spawn desired_world processes
        nproc_per_node = desired_world

    # random master_port to avoid collisions
    master_port = pick_random_port()

    # build base torchrun command
    torchrun_cmd = ["torchrun", f"--nproc_per_node={nproc_per_node}", f"--master_port={master_port}"]

    # construct megatron args list
    megatron_args = [train_script]

    # add parallel flags
    megatron_args.append(f"--tensor-model-parallel-size={tp}")
    # pipeline
    megatron_args.append(f"--pipeline-model-parallel-size={pp_size}")

    # add basic hyperparams
    megatron_args += [
        f"--micro-batch-size={mbs}",
        f"--global-batch-size={gbs}",
        f"--seq-length={seq_len}",
        f"--lr={lr}",
        f"--train-iters={train_iters}",
        f"--num-layers={num_layers}",
        f"--hidden-size={hidden_size}",
        f"--num-attention-heads={num_attention_heads}",
        f"--max-position-embeddings={max_position_embeddings}",
        f"--vocab-size={vocab_size}",
    ]

    # data handling
    data_args = []
    if train_dataset is None or (isinstance(train_dataset, str) and train_dataset.lower() == "mock"):
        # mock-data quick path
        data_args += ["--mock-data", "--tokenizer-type", "NullTokenizer", "--vocab-size", str(vocab_size)]
    else:
        # real data path given. Megatron commonly uses --data-path + --split for train/valid/test splitting.
        data_args += [f"--data-path={train_dataset}"]
        # if user provided explicit validation/test dataset, pass them (best-effort)
        if validation_dataset:
            data_args += [f"--validation-data-path={validation_dataset}"]
        if test_dataset:
            data_args += [f"--test-data-path={test_dataset}"]
        # if neither validation_dataset nor test_dataset provided, ask Megatron to split using --split (default example)
        if not validation_dataset and not test_dataset:
            # use the default split or config-defined split
            data_args += ["--split", conf.get("data_split", "99,1,0")]

    megatron_args += data_args

    # common optional flags
    megatron_args += [
        "--transformer-impl", "local",
        "--bf16",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--log-interval", "1",
        "--log-throughput"
    ]

    if sp == 1:
        megatron_args.append("--sequence-parallel")

    # final command
    cmd = torchrun_cmd + megatron_args

    # logging regex patterns
    iter_pattern = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    loss_pattern = re.compile(r"lm loss:\s*([+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)", re.IGNORECASE)
    elapsed_pattern = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

    losses_by_iter = {}
    ppls_by_iter = {}
    throughput_by_iter = {}
    last_seen_iter = None

    # peak mem holder in MiB
    peak_mem = [0]

    # start process
    print("Launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # start GPU monitor thread (if nvidia-smi available)
    monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(proc, 0.5, peak_mem), daemon=True)
    monitor_thread.start()

    # read stdout
    with open(log_file, "w") as flog:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            # write and mirror to stdout
            print(line)
            flog.write(line + "\n")

            # update iteration
            m_iter = iter_pattern.search(line)
            if m_iter:
                try:
                    last_seen_iter = int(m_iter.group(1))
                except Exception:
                    last_seen_iter = None

            # loss
            m_loss = loss_pattern.search(line)
            if m_loss and last_seen_iter is not None:
                try:
                    val = float(m_loss.group(1))
                    # skip warmup iterations
                    if last_seen_iter > warmup_iters:
                        losses_by_iter[last_seen_iter] = val
                except Exception:
                    pass

            # elapsed -> throughput
            m_elapsed = elapsed_pattern.search(line)
            if m_elapsed and last_seen_iter is not None:
                try:
                    elapsed_ms = float(m_elapsed.group(1))
                    tokens_per_iter_total = mbs * seq_len * nproc_per_node
                    throughput_gpu_normalized = 0.0
                    if elapsed_ms > 0:
                        throughput_gpu_normalized = tokens_per_iter_total / (elapsed_ms / 1000.0) / nproc_per_node  # tokens/s/GPU
                        # skip warmup
                    if last_seen_iter > warmup_iters:
                        throughput_by_iter[last_seen_iter] = throughput_gpu_normalized
                except Exception:
                    pass

    # wait finish
    proc.wait()
    monitor_thread.join(timeout=1.0)

    # compute PPL from loss for each iteration we captured
    for it, loss in losses_by_iter.items():
        try:
            ppls_by_iter[it] = float(math.exp(loss))
        except Exception:
            ppls_by_iter[it] = float("inf")

    # write CSVs
    def dict_to_sorted_list(d):
        return sorted(d.items(), key=lambda x: x[0])

    with open(loss_csv, "w") as f:
        f.write("iteration,loss\n")
        for it, v in dict_to_sorted_list(losses_by_iter):
            f.write(f"{it},{v}\n")

    with open(ppl_csv, "w") as f:
        f.write("iteration,ppl\n")
        for it, v in dict_to_sorted_list(ppls_by_iter):
            f.write(f"{it},{v}\n")

    with open(throughput_csv, "w") as f:
        f.write("iteration,throughput\n")
        for it, v in dict_to_sorted_list(throughput_by_iter):
            f.write(f"{it},{v}\n")

    # write model_config.json
    model_config_path = os.path.join(out_dir, f"train_{model_name}_config.json")
    model_config = {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "max_position_embeddings": max_position_embeddings,
        "vocab_size": vocab_size
    }
    os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # prepare result json
    result = {
        "config": {
            "command": " ".join(cmd),
            "model": model_name,
            "model_config": model_config_path,
            "train_dataset": train_dataset if train_dataset is not None else "mock",
            "validation_dataset": validation_dataset,
            "test_dataset": test_dataset,
            "train_args": train_args,
            "timeout_ms": train_args.get("timeout_ms", 10000),
            "warmup_iterations": warmup_iters,
            "measured_iterations": train_iters
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
                "value": round((peak_mem[0] / 1024.0), 6),
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

    print(f"\nResult JSON written to: {result_json}")
    print("Log written to:", log_file)
    print("CSV files:", loss_csv, ppl_csv, throughput_csv)
    print(f"Peak GPU memory: {peak_mem[0]} MiB ({peak_mem[0]/1024.0:.4f} GiB)")

if __name__ == "__main__":
    main()

