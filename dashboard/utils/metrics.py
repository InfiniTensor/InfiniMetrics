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
    }

    for m in metrics:
        name = m.get("name", "")
        val = m.get("value")

        if "bandwidth" in name and val is not None:
            out["bandwidth_gbps"] = val
        elif "latency_us" in name:
            out["latency_us"] = val
        elif "latency_ms" in name:
            out["latency_ms"] = val
        elif "ttft" in name:
            out["ttft_ms"] = val
        elif "throughput" in name:
            out["throughput"] = val

    return out
