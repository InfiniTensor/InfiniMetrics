def extract_core_metrics(run: dict) -> dict:
    """
    Extract commonly used metrics from run["data"] after load_test_result
    """
    metrics = run.get("data", {}).get("metrics", []) or []

    out = {
        "bandwidth_gbps": None,
        "latency_us": None,
        "latency_ms": None,
        "ttft_ms": None,
        "throughput": None,
        "duration_ms": None,
    }

    for m in metrics:
        name = m.get("name", "")
        val = m.get("value")
        df = m.get("data")

        # Process bandwidth
        if name == "comm.bandwidth" and df is not None:
            if hasattr(df, "columns") and "bandwidth_gbs" in df.columns:
                out["bandwidth_gbps"] = df["bandwidth_gbs"].max()

        # Process communication delay
        elif name == "comm.latency" and df is not None:
            if hasattr(df, "columns") and "latency_us" in df.columns:
                out["latency_us"] = df["latency_us"].mean()

        # Process duration
        elif name == "comm.duration" and val is not None:
            out["duration_ms"] = val

        elif "bandwidth" in name and val is not None:
            out["bandwidth_gbps"] = val
        elif "latency_us" in name:
            out["latency_us"] = val
        elif "latency_ms" in name:
            out["latency_ms"] = val
        elif "duration" in name:
            out["duration_ms"] = val
        elif "ttft" in name:
            out["ttft_ms"] = val
        elif "throughput" in name:
            out["throughput"] = val

    return out
