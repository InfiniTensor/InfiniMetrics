from .runner import TestRunner
from ..adapters.infinicore import InfiniCoreAdapter


class WorkloadDispatcher:
    """
    WorkloadDispatcher:
    Responsible for identifying the test type (Operator vs Training vs Inference)
    and dispatching it to the correct Adapter via a Runner.
    """

    def __init__(self):

        self.adapters = {
            "operator": InfiniCoreAdapter(),
            # "training": InfiniCoreTrainAdapter(), # Future extension
        }
        print("[Dispatcher] System initialized. Adapters loaded.")

    def dispatch(self, request_json: dict) -> dict:
        """
        Route the request to the appropriate adapter based on the testcase name or config.
        """
        testcase_name = request_json.get("testcase", "").lower()

        # Simple Routing Logic
        # Default to operator if not specified, or add logic to detect 'train'
        category = "operator"

        if "operator" != category:

            print(
                f"[Dispatcher] Warning: {category} adapter not yet implemented, falling back to Operator for demo."
            )
            pass

        adapter = self.adapters.get(category)

        if not adapter:
            raise ValueError(f"No adapter found for category: {category}")

        print(
            f"[Dispatcher] Routing task '{testcase_name}' to -> {category.upper()} Adapter"
        )

        # Instantiate the Runner (Project Manager) with the selected Adapter (Worker)
        runner = TestRunner(adapter)

        # Execute
        return runner.run(request_json)
