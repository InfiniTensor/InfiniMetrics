class TestGateway:
    """
    The Gateway is the facade of the system.
    Currently in Direct Call mode, can be easily wrapped as an HTTP Controller in the future.
    """
    def __init__(self):
        # Configure dependency injection here, deciding which Adapter to use.
        # If there are multiple backends, different Adapters can be loaded based on configuration.
        self.default_adapter = InfiniCoreAdapter()
        print("[Gateway] System initialized.")

    def dispatch(self, request_json: dict) -> dict:
        """
        Dispatch the request:
        1. Receive request
        2. Instantiate/Get Runner (Demonstrating creating a new Runner per request, could also use a Runner Pool)
        3. Execute
        """
        # Global authentication, rate limiting, and route dispatching can be done here.
        # e.g., if request_json['target'] == 'nvidia': use NvidiaAdapter...
        
        # Create Runner
        runner = TestRunner(self.default_adapter)
        
        # Execute
        response = runner.run(request_json)
        
        return response
