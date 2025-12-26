import uuid
import time
import copy
from datetime import datetime

from ..adapters.base import BaseAdapter


class TestRunner:
    """
    The Runner is responsible for the specific execution context.
    It has its own ID, can record state, and can include common pre/post-processing logic.
    """

    def __init__(self, adapter: BaseAdapter):
        # Initialization properties for the Runner
        self.runner_id = str(uuid.uuid4())[:8]
        self.adapter = adapter
        self.created_at = time.time()
        self.execution_history = []  # Record tasks run by this Runner

    def run(self, payload: dict) -> dict:
        """
        Execute the test logic.
        """
        start_ts = time.time()
        print(f" >> [Runner:{self.runner_id}] Processing task...")

        # Add Runner-level logic here, e.g., validating payload integrity
        if "config" not in payload:
            raise ValueError("Invalid payload: missing config")

        # Call the Adapter for core processing
        try:
            result = self.adapter.process(payload)
        except Exception as e:
            print(f" !! [Runner:{self.runner_id}] Error occurred: {e}")
            result = copy.deepcopy(payload)
            result["success"] = 0
            result["error_msg"] = str(e)

        # Record Runner audit logs/status
        duration = time.time() - start_ts
        self.execution_history.append(
            {
                "ts": datetime.now().isoformat(),
                "status": "success" if result.get("success") else "fail",
                "duration": duration,
            }
        )

        print(f" >> [Runner:{self.runner_id}] Task completed, duration {duration:.4f}s")
        return result
